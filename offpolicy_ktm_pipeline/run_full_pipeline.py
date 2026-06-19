from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy
from offpolicy_ktm_pipeline.src.dataframe_processing import build_ktm_dataframe
from offpolicy_ktm_pipeline.src.estimators_and_training import (
    compute_log_mixture_propensity,
    dm_policy_value,
    optimal_irt_value,
)
from offpolicy_ktm_pipeline.src.gaussian_regression import fit_gaussian_by_order, poly_eval
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.utils import ensure_parent_dir, get_default_device, set_global_seed

_GL_CACHE: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray]] = {}

def _gauss_legendre_nodes_weights(delta_min: float, delta_max: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic 1D quadrature nodes/weights on [delta_min, delta_max].
    """
    n = max(int(n_points), 2)
    key = (float(delta_min), float(delta_max), n)
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    x, w = np.polynomial.legendre.leggauss(n)
    a = float(delta_min)
    b = float(delta_max)
    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    _GL_CACHE[key] = (nodes.astype(np.float32), weights.astype(np.float32))
    return _GL_CACHE[key]


def _compute_estimators(
    policy: GlobalGaussianPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    log_prop_b: torch.Tensor,
    *,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    max_weight: float,
    clip_c: float,
    device: torch.device,
    context_w: torch.Tensor | None = None,
    precomputed_q_pi: torch.Tensor | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        th = theta.to(device)
        de = delta.to(device)
        rw = reward.to(device)
        lb = log_prop_b.to(device)
        cw = torch.ones_like(rw) if context_w is None else context_w.to(device)
        log_p_new = policy.log_prob(delta=de, theta=th)
        log_ratio = torch.clamp(log_p_new - lb, min=-20.0, max=float(np.log(max_weight)))
        w = torch.exp(log_ratio)
        w_clip = torch.clamp(w, max=float(clip_c))
        wc = cw * w
        wc_clip = cw * w_clip
        cw_sum = torch.sum(cw).clamp_min(1e-12)

        ips = torch.mean(wc * rw)
        snips = torch.sum(wc * rw) / torch.sum(wc).clamp_min(1e-12)
        cips = torch.mean(wc_clip * rw)
        csnips = torch.sum(wc_clip * rw) / torch.sum(wc_clip).clamp_min(1e-12)

        # IRT direct model: q_hat(theta, delta) = sigmoid(theta-delta) * (delta-delta_min)
        q_logged = torch.sigmoid(th - de) * (de - float(delta_min))
        if precomputed_q_pi is not None:
            q_pi = precomputed_q_pi.to(device)
        else:
            q_pi = _dm_expected_reward_per_theta(
                policy,
                theta=th,
                delta_min=delta_min,
                delta_max=delta_max,
                n_grid=dm_delta_grid,
            )
        resid = rw - q_logged
        dr = torch.sum(cw * (q_pi + w * resid)) / cw_sum
        sndr = (torch.sum(cw * q_pi) / cw_sum) + (torch.sum(wc * resid) / torch.sum(wc).clamp_min(1e-12))
        cdr = torch.sum(cw * (q_pi + w_clip * resid)) / cw_sum
        csndr = (torch.sum(cw * q_pi) / cw_sum) + (
            torch.sum(wc_clip * resid) / torch.sum(wc_clip).clamp_min(1e-12)
        )

        ess = (torch.sum(wc) ** 2) / torch.sum(wc ** 2).clamp_min(1e-12)
        ess_clip = (torch.sum(wc_clip) ** 2) / torch.sum(wc_clip ** 2).clamp_min(1e-12)

    return {
        "ips": float(ips.item()),
        "snips": float(snips.item()),
        "cips": float(cips.item()),
        "csnips": float(csnips.item()),
        "dr": float(dr.item()),
        "sndr": float(sndr.item()),
        "cdr": float(cdr.item()),
        "csndr": float(csndr.item()),
        "ess": float(ess.item()),
        "ess_clip": float(ess_clip.item()),
        "mean_w": float(torch.mean(w).item()),
        "std_w": float(torch.std(w).item()),
    }


def _dm_expected_reward_per_theta(
    policy: GlobalGaussianPolicy,
    *,
    theta: torch.Tensor,
    delta_min: float,
    delta_max: float,
    n_grid: int,
) -> torch.Tensor:
    """
    Per-theta direct-method expected reward:
      E_{delta ~ pi(.|theta)} [ sigmoid(theta-delta) * (delta-delta_min) ].
    Uses a differentiable 1D quadrature over delta.
    """
    nodes_np, w_np = _gauss_legendre_nodes_weights(delta_min=delta_min, delta_max=delta_max, n_points=n_grid)
    nodes = torch.as_tensor(nodes_np, device=theta.device, dtype=theta.dtype)
    quad_w = torch.as_tensor(w_np, device=theta.device, dtype=theta.dtype).clamp_min(1e-12)
    log_quad_w = torch.log(quad_w)[None, :]
    n = int(nodes.shape[0])
    bsz = int(theta.shape[0])
    max_rows = max(1, int(2_000_000 // max(n, 1)))
    out_chunks: list[torch.Tensor] = []
    for start in range(0, bsz, max_rows):
        end = min(start + max_rows, bsz)
        th = theta[start:end, None].expand(end - start, n)
        de = nodes[None, :].expand(end - start, n)
        logp = policy.log_prob(delta=de.reshape(-1), theta=th.reshape(-1)).reshape(end - start, n)
        reward = torch.sigmoid(th - de) * (de - float(delta_min))
        norm_w = torch.softmax(logp + log_quad_w, dim=1)
        out_chunks.append(torch.sum(norm_w * reward, dim=1))
    return torch.cat(out_chunks, dim=0)


def _compute_context_ratio_hist(
    *,
    theta_ref: np.ndarray,
    theta_cur: np.ndarray,
    theta_eval: np.ndarray,
    n_bins: int,
    ratio_clip: float,
    eps: float = 1e-8,
) -> np.ndarray:
    lo = float(min(theta_ref.min(), theta_cur.min(), theta_eval.min()))
    hi = float(max(theta_ref.max(), theta_cur.max(), theta_eval.max()))
    if hi <= lo:
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, max(4, int(n_bins)) + 1)
    q_hist, _ = np.histogram(theta_ref, bins=edges, density=True)
    p_hist, _ = np.histogram(theta_cur, bins=edges, density=True)
    ratio_bins = (q_hist + eps) / (p_hist + eps)

    if ratio_clip > 1.0:
        ratio_bins = np.clip(ratio_bins, 1.0 / ratio_clip, ratio_clip)

    idx = np.clip(np.digitize(theta_eval, bins=edges[1:-1], right=False), 0, len(ratio_bins) - 1)
    out = ratio_bins[idx].astype(np.float32)
    # Normalize to keep average weight around 1 for stability.
    out = out / np.clip(float(out.mean()), 1e-8, None)
    return out


def _hist_prob(theta: np.ndarray, edges: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    counts, _ = np.histogram(theta, bins=edges)
    p = counts.astype(float) + eps
    p = p / np.clip(float(p.sum()), 1e-12, None)
    return p


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    m = 0.5 * (p + q)
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    m_safe = np.clip(m, eps, None)
    kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
    return float(0.5 * (kl_pm + kl_qm))


def _select_theta_reference(
    *,
    offpolicy_df: pd.DataFrame,
    order_list: list[int],
    context_target: str,
    context_k_rounds: int,
    context_bins: int,
) -> tuple[np.ndarray | None, str]:
    if context_target == "none":
        return None, "none"
    if context_target == "global":
        return offpolicy_df["proficiency"].to_numpy(dtype=float), "global"
    if context_target == "round1":
        first_order = int(order_list[0])
        theta = offpolicy_df[offpolicy_df["order_sequence"] == first_order]["proficiency"].to_numpy(dtype=float)
        return theta, f"round1(order={first_order})"

    k = max(1, min(int(context_k_rounds), len(order_list)))
    first_k = [int(x) for x in order_list[:k]]
    first_k_df = offpolicy_df[offpolicy_df["order_sequence"].isin(first_k)].copy()
    if first_k_df.empty:
        return None, f"{context_target}(empty)"

    if context_target == "pooled_first_k":
        theta = first_k_df["proficiency"].to_numpy(dtype=float)
        return theta, f"pooled_first_{k}"

    if context_target == "best_round_in_k":
        lo = float(first_k_df["proficiency"].min())
        hi = float(first_k_df["proficiency"].max())
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, max(4, int(context_bins)) + 1)
        round_dists: dict[int, np.ndarray] = {}
        for o in first_k:
            th = offpolicy_df[offpolicy_df["order_sequence"] == o]["proficiency"].to_numpy(dtype=float)
            round_dists[o] = _hist_prob(th, edges)

        best_order = first_k[0]
        best_score = float("inf")
        for cand in first_k:
            p_c = round_dists[cand]
            score = 0.0
            for o in first_k:
                score += _js_divergence(p_c, round_dists[o])
            score /= float(len(first_k))
            if score < best_score:
                best_score = score
                best_order = cand

        theta = offpolicy_df[offpolicy_df["order_sequence"] == best_order]["proficiency"].to_numpy(dtype=float)
        return theta, f"best_round_in_first_{k}(order={best_order},mean_js={best_score:.6f})"

    raise ValueError(f"Unknown context_target: {context_target}")


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.clip(np.sum(w), 1e-12, None))


def _train_policy_one_round(
    policy: GlobalGaussianPolicy,
    *,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    log_prop_b: torch.Tensor,
    context_w: torch.Tensor | None,
    objective: str,
    epochs: int,
    batch_size: int,
    lr: float,
    l2_coef: float,
    variance_penalty_coef: float,
    max_weight: float,
    clip_c: float,
    num_workers: int,
    device: torch.device,
    balanced_minibatch: bool,
    theta_bins: int,
    sampling_weights: torch.Tensor | None,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    train_dm_delta_grid: int = 0,
) -> None:
    # Use a coarser grid during training to reduce quadrature cost; fall back to dm_delta_grid.
    _train_grid = int(train_dm_delta_grid) if int(train_dm_delta_grid) > 0 else int(dm_delta_grid)
    if context_w is None:
        context_w = torch.ones_like(reward)

    dataset = TensorDataset(theta, delta, reward, log_prop_b, context_w)
    sampler_weights = None
    if sampling_weights is not None:
        sampler_weights = sampling_weights.detach().cpu().numpy().astype(float)
    if balanced_minibatch:
        th_np = theta.detach().cpu().numpy().astype(float)
        # Quantile bins create approximately balanced theta buckets.
        q_edges = np.quantile(th_np, np.linspace(0.0, 1.0, max(2, int(theta_bins) + 1)))
        edges = np.unique(q_edges)
        if len(edges) > 2:
            bin_idx = np.clip(np.digitize(th_np, bins=edges[1:-1], right=False), 0, len(edges) - 2)
            counts = np.bincount(bin_idx, minlength=max(bin_idx) + 1).astype(float)
            balance_w = 1.0 / np.clip(counts[bin_idx], 1.0, None)
            if sampler_weights is None:
                sampler_weights = balance_w
            else:
                sampler_weights = sampler_weights * balance_w

    sampler = None
    shuffle = True
    if sampler_weights is not None:
        sampler_weights = sampler_weights / np.clip(float(np.mean(sampler_weights)), 1e-8, None)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sampler_weights.astype(np.float64)),
            num_samples=len(theta),
            replacement=True,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for _ in range(epochs):
        for b_theta, b_delta, b_reward, b_logb, b_ctx in loader:
            b_theta = b_theta.to(device)
            b_delta = b_delta.to(device)
            b_reward = b_reward.to(device)
            b_logb = b_logb.to(device)
            b_ctx = b_ctx.to(device)

            optimizer.zero_grad(set_to_none=True)
            if objective == "dm":
                dm_vals = _dm_expected_reward_per_theta(
                    policy,
                    theta=b_theta,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    n_grid=_train_grid,
                )
                obj = torch.sum(b_ctx * dm_vals) / torch.sum(b_ctx).clamp_min(1e-12)
                contrib = (b_ctx / torch.sum(b_ctx).clamp_min(1e-12)) * dm_vals
            else:
                log_p_new = policy.log_prob(delta=b_delta, theta=b_theta)
                log_ratio = torch.clamp(log_p_new - b_logb, min=-20.0, max=float(np.log(max_weight)))
                w = torch.exp(log_ratio)
                wc = b_ctx * w

                ips_obj = torch.mean(wc * b_reward)
                snips_obj = torch.sum(wc * b_reward) / torch.sum(wc).clamp_min(1e-12)
                cw_sum = torch.sum(b_ctx).clamp_min(1e-12)
                q_pi = None
                resid = None
                if objective == "dr":
                    q_logged = torch.sigmoid(b_theta - b_delta) * (b_delta - float(delta_min))
                    q_pi = _dm_expected_reward_per_theta(
                        policy,
                        theta=b_theta,
                        delta_min=delta_min,
                        delta_max=delta_max,
                        n_grid=_train_grid,
                    )
                    resid = b_reward - q_logged
                    dr_obj = torch.sum(b_ctx * (q_pi + w * resid)) / cw_sum

                if objective == "ips":
                    obj = ips_obj
                    contrib = wc * b_reward
                elif objective == "snips":
                    obj = snips_obj
                    contrib = (wc / torch.sum(wc).clamp_min(1e-12)) * b_reward
                else:  # dr
                    assert q_pi is not None and resid is not None
                    obj = dr_obj
                    contrib = (b_ctx / cw_sum) * (q_pi + w * resid)
            var_est = torch.var(contrib, unbiased=False)
            t_samples = float(max(int(b_theta.shape[0]), 1))
            var_reg = (
                float(variance_penalty_coef)
                * np.sqrt(1.0 / t_samples)
                * torch.sqrt(var_est.clamp_min(1e-12))
            )

            reg = l2_coef * (torch.mean(policy.beta_mu ** 2) + torch.mean(policy.beta_sigma ** 2))
            loss = -obj + reg + var_reg
            loss.backward()
            optimizer.step()


def _plot_gaussian_fit_quality(fit_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    x = fit_df["order_sequence"].to_numpy(dtype=int)

    axes[0].plot(x, fit_df["nll"].to_numpy(dtype=float), color="#1f77b4", lw=2.0, marker="o", markersize=3)
    axes[0].set_ylabel("NLL")
    axes[0].set_title("Gaussian Regression Fit Quality by order_sequence")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, fit_df["rmse"].to_numpy(dtype=float), color="#ff7f0e", lw=2.0, marker="o", markersize=3, label="rmse")
    axes[1].plot(x, fit_df["mae"].to_numpy(dtype=float), color="#2ca02c", lw=2.0, marker="o", markersize=3, label="mae")
    axes[1].set_xlabel("order_sequence")
    axes[1].set_ylabel("error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_policy_values(round_df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = round_df["round"].to_numpy(dtype=int)
    ax.plot(x, round_df["behavior_empirical_batch_mean"], color="#444444", lw=2.0, label="behavior empirical")
    ax.plot(x, round_df["behavior_irt_value"], color="#ff7f0e", lw=2.0, label="behavior IRT value")
    ax.plot(x, round_df["learned_irt_value"], color="#1f77b4", lw=2.3, label="learned policy IRT value")
    ax.plot(x, round_df["optimal_irt_value"], color="#2ca02c", lw=2.3, linestyle="--", label="optimal policy IRT value")
    ax.set_xlabel("round")
    ax.set_ylabel("value")
    ax.set_title("Round-by-Round Policy Values")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _extract_degree(cols: list[str], prefix: str) -> int:
    idx = [int(c.split("_")[-1]) for c in cols if c.startswith(prefix)]
    if not idx:
        raise ValueError(f"Missing coefficients with prefix: {prefix}")
    return max(idx)


def _optimal_curve_irt(theta_grid: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    d_grid = np.linspace(delta_min, delta_max, 1200)
    z = theta_grid[:, None] - d_grid[None, :]
    p = 1.0 / (1.0 + np.exp(-z))
    exp_reward = p * (d_grid[None, :] - delta_min)
    return d_grid[np.argmax(exp_reward, axis=1)]


def _bound_mu_np(mu_raw: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    width = float(delta_max - delta_min)
    x = np.clip(mu_raw, -40.0, 40.0)
    return float(delta_min) + width * (1.0 / (1.0 + np.exp(-x)))


def _make_policy_animation(
    *,
    round_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    out_gif: str,
    fps: int,
    max_scatter: int,
    y_min: float | None,
    y_max: float | None,
) -> None:
    mu_deg = _extract_degree(coeff_df.columns.tolist(), "beta_mu_")
    sig_deg = _extract_degree(coeff_df.columns.tolist(), "beta_sigma_")

    theta_min = float(tuples_df["proficiency"].min())
    theta_max = float(tuples_df["proficiency"].max())
    theta_grid = np.linspace(theta_min, theta_max, 450)
    delta_min = float(tuples_df["difficulties"].min())
    delta_max = float(tuples_df["difficulties"].max())
    opt_curve = _optimal_curve_irt(theta_grid, delta_min=delta_min, delta_max=delta_max)

    rng = np.random.default_rng(42)
    sampled_xy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order_val, g in tuples_df.groupby("order_sequence", sort=False):
        if len(g) > max_scatter:
            idx = rng.choice(len(g), size=max_scatter, replace=False)
            gg = g.iloc[idx]
        else:
            gg = g
        sampled_xy[int(order_val)] = (
            gg["proficiency"].to_numpy(dtype=float),
            gg["difficulties"].to_numpy(dtype=float),
        )

    fig, ax = plt.subplots(figsize=(9.2, 5.9))

    def draw(i: int) -> None:
        ax.clear()
        r = round_df.iloc[i]
        round_no = int(r["round"])
        order_val = int(r["order_sequence"])
        crow = coeff_df[coeff_df["round"] == round_no].iloc[0]
        brow = fit_df[fit_df["order_sequence"] == order_val].iloc[0]

        b_mu = np.array([float(brow.get(f"beta_mu_{d}", 0.0)) for d in range(mu_deg + 1)], dtype=float)
        b_sig = np.array([float(brow.get(f"beta_sigma_{d}", 0.0)) for d in range(sig_deg + 1)], dtype=float)
        l_mu = np.array([float(crow.get(f"beta_mu_{d}", 0.0)) for d in range(mu_deg + 1)], dtype=float)
        l_sig = np.array([float(crow.get(f"beta_sigma_{d}", 0.0)) for d in range(sig_deg + 1)], dtype=float)

        beh_mu = poly_eval(theta_grid, b_mu)
        beh_sig = np.maximum(np.exp(poly_eval(theta_grid, b_sig)), 1e-4)
        pol_mu_raw = poly_eval(theta_grid, l_mu)
        pol_mu = _bound_mu_np(pol_mu_raw, delta_min=delta_min, delta_max=delta_max)
        pol_sig = np.maximum(np.exp(poly_eval(theta_grid, l_sig)), 1e-4)

        xs, ys = sampled_xy.get(order_val, (np.array([]), np.array([])))
        ax.scatter(xs, ys, s=8, alpha=0.12, color="#8c8c8c")
        ax.plot(theta_grid, beh_mu, color="#ff7f0e", lw=2.0, label="behavior mean")
        ax.fill_between(theta_grid, beh_mu - beh_sig, beh_mu + beh_sig, color="#ff7f0e", alpha=0.14)
        ax.plot(theta_grid, pol_mu, color="#1f77b4", lw=2.2, label="learned mean")
        ax.fill_between(theta_grid, pol_mu - pol_sig, pol_mu + pol_sig, color="#1f77b4", alpha=0.14)
        ax.plot(theta_grid, opt_curve, color="#2ca02c", lw=2.2, linestyle="--", label="optimal (IRT)")

        if y_min is not None and y_max is not None:
            ax.set_ylim(float(y_min), float(y_max))
        ax.set_xlabel("theta")
        ax.set_ylabel("delta")
        ax.set_title(f"Round {round_no}/{len(round_df)} | order={order_val}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

        info = (
            f"Estimator values\\n"
            f"Behavior empirical: {float(r['behavior_empirical_batch_mean']):.3f}\\n"
            f"Learned IPS: {float(r['ips']):.3f} | SNIPS: {float(r['snips']):.3f}\\n"
            f"Learned CIPS: {float(r['cips']):.3f} | CSNIPS: {float(r['csnips']):.3f}\\n"
            f"IRT values -> behavior: {float(r['behavior_irt_value']):.3f}, "
            f"learned: {float(r['learned_irt_value']):.3f}, optimal: {float(r['optimal_irt_value']):.3f}"
        )
        ax.text(
            0.99,
            0.01,
            info,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
        )

    ani = animation.FuncAnimation(fig, draw, frames=len(round_df), interval=1000 / max(fps, 1))
    ensure_parent_dir(out_gif)
    ani.save(out_gif, writer=animation.PillowWriter(fps=max(fps, 1)))
    plt.close(fig)


def _build_round_bucket_mapping(
    order_series: pd.Series,
    *,
    max_rounds: int,
    strategy: str,
) -> pd.DataFrame:
    counts = order_series.astype(int).value_counts().sort_index().rename("n_obs")
    raw_orders = counts.index.to_numpy(dtype=int)

    if max_rounds <= 0 or len(raw_orders) <= max_rounds:
        return pd.DataFrame(
            {
                "order_sequence_raw": raw_orders,
                "order_round": np.arange(len(raw_orders), dtype=int),
                "n_obs": counts.to_numpy(dtype=int),
            }
        )

    if max_rounds == 1:
        return pd.DataFrame(
            {
                "order_sequence_raw": raw_orders,
                "order_round": np.zeros(len(raw_orders), dtype=int),
                "n_obs": counts.to_numpy(dtype=int),
            }
        )

    if strategy == "tail":
        # Keep earliest rounds separate, merge long-tail rounds into last bucket.
        order_rank = np.arange(len(raw_orders), dtype=int)
        order_round = np.where(order_rank < (max_rounds - 1), order_rank, max_rounds - 1)
    elif strategy == "tail_count_at_n":
        # Keep earliest rounds [0..N-1] separate, then group the tail into
        # windows whose observation count is close to count(order=N).
        # Here N is max_rounds in 1-based user language.
        anchor_idx = int(max_rounds - 1)
        target_count = int(counts.iloc[anchor_idx])
        if target_count <= 0:
            target_count = 1

        order_round = np.zeros(len(raw_orders), dtype=int)
        order_round[: anchor_idx + 1] = np.arange(anchor_idx + 1, dtype=int)

        cur_round = anchor_idx + 1
        acc = 0
        for i in range(anchor_idx + 1, len(raw_orders)):
            order_round[i] = cur_round
            acc += int(counts.iloc[i])
            if acc >= target_count:
                cur_round += 1
                acc = 0
    elif strategy == "quantile":
        # Weighted quantile buckets by observation counts.
        cum = counts.cumsum().to_numpy(dtype=float)
        total = float(cum[-1])
        edges = np.linspace(0.0, total, max_rounds + 1)
        order_round = np.digitize(cum, bins=edges[1:-1], right=True).astype(int)
    else:
        raise ValueError(f"Unknown round cap strategy: {strategy}")

    return pd.DataFrame(
        {
            "order_sequence_raw": raw_orders,
            "order_round": order_round.astype(int),
            "n_obs": counts.to_numpy(dtype=int),
        }
    )


def _apply_round_cap(
    df_ktm: pd.DataFrame,
    *,
    max_rounds: int,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cap number of rounds by mapping raw order_sequence to order_round buckets.
    """
    out = df_ktm.copy()
    out["order_sequence_raw"] = out["order_sequence"].astype(int)
    mapping = _build_round_bucket_mapping(
        out["order_sequence_raw"],
        max_rounds=max_rounds,
        strategy=strategy,
    )
    mapper = dict(
        zip(
            mapping["order_sequence_raw"].astype(int).tolist(),
            mapping["order_round"].astype(int).tolist(),
        )
    )
    out["order_sequence"] = out["order_sequence_raw"].map(mapper).astype(int)
    return out, mapping


def _prepare_ktm_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_is_ktm:
        df = pd.read_csv(args.input_csv).copy()
        required = {"user", "item", "correct", "answer_number", "order_sequence", "proficiency", "difficulties"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"--input-is-ktm set but missing columns: {missing}")
        return df

    return build_ktm_dataframe(
        processed_csv=args.input_csv,
        user_col=args.user_col,
        item_col=args.item_col,
        correct_col=args.correct_col,
        skill_col=args.skill_col,
        answer_order_col=args.answer_order_col,
        derive_answer_number_per_user=args.derive_answer_number_per_user,
        sample_users=args.sample_users,
        sample_rows=args.sample_rows,
        seed=args.seed,
        c=args.ktm_c,
        fit_intercept=args.ktm_fit_intercept,
        target_std=args.ktm_target_std,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: KTM dataframe -> Gaussian by-order fit -> propensities -> round-wise off-policy learning."
    )
    parser.add_argument("--input-csv", type=str, default="pix_mapping/pix_processed.csv")
    parser.add_argument("--input-is-ktm", action="store_true")
    parser.add_argument("--output-dir", type=str, default="pix_mapping/full_pipeline_run")

    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="outcome")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument(
        "--derive-answer-number-per-user",
        action="store_true",
        help="Force answer_number = 1..N per user in row order.",
    )

    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Maximum number of training/evaluation rounds. 0 means no cap.",
    )
    parser.add_argument(
        "--round-cap-strategy",
        type=str,
        default="tail",
        choices=["tail", "quantile", "tail_count_at_n"],
        help=(
            "tail: merge long-tail into last bucket; "
            "quantile: balanced buckets by data count; "
            "tail_count_at_n: keep rounds up to N (N=max_rounds), then group later rounds "
            "into windows of size approx count(order=N)."
        ),
    )

    parser.add_argument("--ktm-c", type=float, default=0.1)
    parser.add_argument("--ktm-fit-intercept", action="store_true")
    parser.add_argument("--ktm-target-std", type=float, default=0.0)

    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument(
        "--behavior-dist",
        type=str,
        default="truncated_gaussian",
        choices=["truncated_gaussian"],
        help="Distribution used for behavior policy fit by order (fixed to truncated_gaussian).",
    )
    parser.add_argument(
        "--sigma-floor",
        type=float,
        default=0.2,
        help="Hard lower bound for sigma via clamp_min in Gaussian policy/propensity computations.",
    )
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument(
        "--fit-n-jobs",
        type=int,
        default=-1,
        help="Number of joblib workers for Gaussian fit by order (-1 uses all cores, 1 disables multiprocessing).",
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="snips",
        choices=["ips", "mis", "snips", "dr", "dm"],
    )
    parser.add_argument("--denominator", type=str, default="logged", choices=["logged", "mixture"])
    parser.add_argument("--train-scope", type=str, default="cumulative", choices=["cumulative", "current"])
    parser.add_argument(
        "--train-only-last-round",
        action="store_true",
        help="If set, skip policy optimization on earlier rounds and train only at the final round.",
    )
    parser.add_argument(
        "--context-target",
        type=str,
        default="none",
        choices=["none", "round1", "global", "pooled_first_k", "best_round_in_k"],
        help="Fixed q(theta) target used for context reweighting in training/evaluation.",
    )
    parser.add_argument(
        "--context-k-rounds",
        type=int,
        default=5,
        help="K used by pooled_first_k / best_round_in_k context-target selection.",
    )
    parser.add_argument("--context-bins", type=int, default=120, help="Histogram bins for q(theta)/p(theta) ratio.")
    parser.add_argument(
        "--context-ratio-clip",
        type=float,
        default=20.0,
        help="Clip q/p ratio to [1/clip, clip] for stability.",
    )
    parser.add_argument(
        "--balanced-minibatch",
        action="store_true",
        help="Use theta-balanced mini-batch sampling (quantile bins).",
    )
    parser.add_argument(
        "--context-match-sampling",
        action="store_true",
        help="Sample tuples in training to match q(theta) using q/p ratio weights.",
    )
    parser.add_argument("--theta-bins", type=int, default=20, help="Number of theta bins for balanced mini-batches.")
    parser.add_argument("--epochs-per-round", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--dynamic-batch-size",
        action="store_true",
        help="Set round batch size from current n_train to keep target steps/epoch.",
    )
    parser.add_argument(
        "--target-steps-per-epoch",
        type=int,
        default=300,
        help="Used when --dynamic-batch-size: batch_size ~= ceil(n_train / target_steps_per_epoch).",
    )
    parser.add_argument(
        "--batch-size-min",
        type=int,
        default=512,
        help="Lower clip for dynamic batch size.",
    )
    parser.add_argument(
        "--batch-size-max",
        type=int,
        default=4096,
        help="Upper clip for dynamic batch size.",
    )
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument(
        "--variance-penalty-coef",
        type=float,
        default=0.0,
        help="Sample-variance penalty coefficient on batch IS contribution terms.",
    )
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--clip-c", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--irt-mc-samples", type=int, default=128)
    parser.add_argument("--dm-delta-grid", type=int, default=121, help="Delta grid size for direct-method objective.")
    parser.add_argument(
        "--train-dm-delta-grid",
        type=int,
        default=31,
        help="Coarser delta grid used during training for DR/DM objectives (0 = same as --dm-delta-grid).",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--make-animation", action="store_true")
    parser.add_argument("--animation-fps", type=int, default=1)
    parser.add_argument("--animation-max-scatter", type=int, default=2500)
    parser.add_argument("--animation-name", type=str, default="policy_round_animation.gif")
    parser.add_argument("--animation-y-min", type=float, default=None)
    parser.add_argument("--animation-y-max", type=float, default=None)
    args = parser.parse_args()

    if args.denominator == "mixture":
        raise ValueError(
            "--denominator mixture is deprecated for run_full_pipeline; "
            "use --objective mis to optimize a dedicated mixture-denominator objective."
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.target_steps_per_epoch <= 0:
        raise ValueError("--target-steps-per-epoch must be > 0")
    if args.batch_size_min <= 0 or args.batch_size_max <= 0:
        raise ValueError("--batch-size-min/--batch-size-max must be > 0")
    if args.batch_size_min > args.batch_size_max:
        raise ValueError("--batch-size-min must be <= --batch-size-max")

    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_std = None if args.ktm_target_std <= 0 else float(args.ktm_target_std)
    args.ktm_target_std = target_std

    df_ktm = _prepare_ktm_dataframe(args)
    df_ktm, round_mapping = _apply_round_cap(
        df_ktm,
        max_rounds=args.max_rounds,
        strategy=args.round_cap_strategy,
    )
    ktm_path = out_dir / "ktm_dataframe.csv"
    df_ktm.to_csv(ktm_path, index=False)
    round_map_path = out_dir / "round_bucket_mapping.csv"
    round_mapping.to_csv(round_map_path, index=False)

    fit_df = fit_gaussian_by_order(
        df_ktm,
        order_col="order_sequence",
        theta_col="proficiency",
        delta_col="difficulties",
        mu_degree=args.mu_degree,
        sigma_degree=args.sigma_degree,
        sigma_floor=args.sigma_floor,
        min_obs_per_order=args.min_obs_per_order,
        distribution=args.behavior_dist,
        delta_min=float(df_ktm["difficulties"].min()),
        delta_max=float(df_ktm["difficulties"].max()),
        n_jobs=args.fit_n_jobs,
    )
    if fit_df.empty:
        raise ValueError("Gaussian fit returned no order_sequence (check min_obs_per_order or data).")
    fit_path = out_dir / "gaussian_fit_by_order.csv"
    fit_df.to_csv(fit_path, index=False)

    offpolicy_df = build_offpolicy_dataset(df_ktm, fit_df, sigma_floor=args.sigma_floor)
    offpolicy_path = out_dir / "offpolicy_tuples.csv"
    offpolicy_df.to_csv(offpolicy_path, index=False)

    if args.denominator == "logged" and "log_propensity" not in offpolicy_df.columns:
        raise ValueError("Missing log_propensity in offpolicy tuples for denominator='logged'")

    order_list = sorted(set(offpolicy_df["order_sequence"].astype(int).tolist()) & set(fit_df["order_sequence"].astype(int).tolist()))
    if not order_list:
        raise ValueError("No overlapping orders between tuples and Gaussian fits.")

    delta_min = float(offpolicy_df["difficulties"].min())
    delta_max = float(offpolicy_df["difficulties"].max())
    device = get_default_device(args.device if args.device else None)

    theta_ref_np, context_ref_label = _select_theta_reference(
        offpolicy_df=offpolicy_df,
        order_list=order_list,
        context_target=args.context_target,
        context_k_rounds=args.context_k_rounds,
        context_bins=args.context_bins,
    )

    policy: GlobalGaussianPolicy | None = None
    if args.train_scope == "cumulative":
        policy = GlobalGaussianPolicy(
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            delta_min=delta_min,
            delta_max=delta_max,
            bound_mean_to_action_range=True,
        ).to(device)
        first_row = fit_df.sort_values("order_sequence").iloc[0]
        policy.init_from_behavior_row(first_row)

    summary_rows: list[dict[str, object]] = []
    coeff_rows: list[dict[str, object]] = []

    # Pre-group off-policy data by order to avoid O(n²) pandas filtering inside the loop.
    _order_arrays: dict[int, dict[str, np.ndarray]] = {}
    for _ov in order_list:
        _g = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == int(_ov)]
        _order_arrays[int(_ov)] = {
            "theta": _g["proficiency"].to_numpy(dtype=np.float32),
            "delta": _g["difficulties"].to_numpy(dtype=np.float32),
            "reward": _g["reward"].to_numpy(dtype=np.float32),
            "log_propensity": _g["log_propensity"].to_numpy(dtype=np.float32),
            "order_sequence": _g["order_sequence"].to_numpy(dtype=int),
        }

    # Cumulative numpy arrays extended incrementally each round (replaces growing seen DataFrame).
    _cum_theta = np.empty(0, dtype=np.float32)
    _cum_delta = np.empty(0, dtype=np.float32)
    _cum_reward = np.empty(0, dtype=np.float32)
    _cum_logb = np.empty(0, dtype=np.float32)
    _cum_orders = np.empty(0, dtype=int)

    for round_idx, order_val in enumerate(order_list, start=1):
        _bg = _order_arrays[int(order_val)]
        _cum_theta = np.concatenate([_cum_theta, _bg["theta"]])
        _cum_delta = np.concatenate([_cum_delta, _bg["delta"]])
        _cum_reward = np.concatenate([_cum_reward, _bg["reward"]])
        _cum_logb = np.concatenate([_cum_logb, _bg["log_propensity"]])
        _cum_orders = np.concatenate([_cum_orders, _bg["order_sequence"]])
        behavior_row = fit_df[fit_df["order_sequence"] == order_val].iloc[0]

        if args.train_scope == "current":
            policy = GlobalGaussianPolicy(
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                sigma_floor=args.sigma_floor,
                delta_min=delta_min,
                delta_max=delta_max,
                bound_mean_to_action_range=True,
            ).to(device)
            policy.init_from_behavior_row(behavior_row)
            theta_np = _bg["theta"]
            delta_np = _bg["delta"]
            reward_np = _bg["reward"]
            _logb_logged_np = _bg["log_propensity"]
            _train_orders = _bg["order_sequence"]
        else:
            theta_np = _cum_theta
            delta_np = _cum_delta
            reward_np = _cum_reward
            _logb_logged_np = _cum_logb
            _train_orders = _cum_orders

        assert policy is not None
        n_train = int(len(theta_np))
        n_seen = int(len(_cum_theta))
        batch_size_used = int(args.batch_size)
        if args.dynamic_batch_size:
            raw_bs = int(np.ceil(float(max(n_train, 1)) / float(args.target_steps_per_epoch)))
            batch_size_used = int(np.clip(raw_bs, int(args.batch_size_min), int(args.batch_size_max)))
            batch_size_used = max(1, min(batch_size_used, max(n_train, 1)))

        ctx_train_np = np.ones_like(theta_np, dtype=np.float32)
        sample_w_np: np.ndarray | None = None
        if theta_ref_np is not None and len(theta_np) > 0:
            ctx_train_np = _compute_context_ratio_hist(
                theta_ref=theta_ref_np.astype(float),
                theta_cur=theta_np.astype(float),
                theta_eval=theta_np.astype(float),
                n_bins=args.context_bins,
                ratio_clip=args.context_ratio_clip,
            )
            if args.context_match_sampling:
                sample_w_np = np.clip(
                    ctx_train_np.astype(float),
                    1.0 / max(args.context_ratio_clip, 1.0),
                    max(args.context_ratio_clip, 1.0),
                ).astype(np.float32)

        logb_mix_np = compute_log_mixture_propensity(
            theta=theta_np.astype(float),
            delta=delta_np.astype(float),
            orders=_train_orders,
            fit_df=fit_df,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
        )
        logb_np = _logb_logged_np
        n_components = int(np.unique(_train_orders).size)

        theta_t = torch.from_numpy(theta_np)
        delta_t = torch.from_numpy(delta_np)
        reward_t = torch.from_numpy(reward_np)
        logb_t = torch.from_numpy(logb_np.astype(np.float32))
        logb_mix_t = torch.from_numpy(logb_mix_np.astype(np.float32))
        ctx_train_t = torch.from_numpy(ctx_train_np.astype(np.float32))
        sample_w_t = None if sample_w_np is None else torch.from_numpy(sample_w_np.astype(np.float32))

        did_train_round = (not bool(args.train_only_last_round)) or (int(round_idx) == int(len(order_list)))
        if did_train_round:
            train_objective = "ips" if args.objective == "mis" else args.objective
            train_logb_t = logb_mix_t if args.objective == "mis" else logb_t
            _train_policy_one_round(
                policy,
                theta=theta_t,
                delta=delta_t,
                reward=reward_t,
                log_prop_b=train_logb_t,
                context_w=ctx_train_t,
                objective=train_objective,
                epochs=args.epochs_per_round,
                batch_size=batch_size_used,
                lr=args.lr,
                l2_coef=args.l2_coef,
                variance_penalty_coef=args.variance_penalty_coef,
                max_weight=args.max_weight,
                clip_c=args.clip_c,
                num_workers=args.num_workers,
                device=device,
                balanced_minibatch=args.balanced_minibatch,
                theta_bins=args.theta_bins,
                sampling_weights=sample_w_t,
                delta_min=delta_min,
                delta_max=delta_max,
                dm_delta_grid=args.dm_delta_grid,
                train_dm_delta_grid=args.train_dm_delta_grid,
            )

        with torch.no_grad():
            _shared_q_pi = _dm_expected_reward_per_theta(
                policy,
                theta=theta_t.to(device),
                delta_min=delta_min,
                delta_max=delta_max,
                n_grid=args.dm_delta_grid,
            ).cpu()

        est = _compute_estimators(
            policy,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            log_prop_b=logb_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=args.dm_delta_grid,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
            context_w=None,
            precomputed_q_pi=_shared_q_pi,
        )
        est_q = _compute_estimators(
            policy,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            log_prop_b=logb_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=args.dm_delta_grid,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
            context_w=ctx_train_t,
            precomputed_q_pi=_shared_q_pi,
        )
        est_mix = _compute_estimators(
            policy,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            log_prop_b=logb_mix_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=args.dm_delta_grid,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
            context_w=None,
            precomputed_q_pi=_shared_q_pi,
        )
        est_mix_q = _compute_estimators(
            policy,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            log_prop_b=logb_mix_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=args.dm_delta_grid,
            max_weight=args.max_weight,
            clip_c=args.clip_c,
            device=device,
            context_w=ctx_train_t,
            precomputed_q_pi=_shared_q_pi,
        )
        if args.objective == "dm":
            cw_t = ctx_train_t.to(device)
            objective_estimate = float(_shared_q_pi.mean().item())
            objective_estimate_q = float(
                (torch.sum(cw_t.cpu() * _shared_q_pi) / torch.sum(cw_t.cpu()).clamp_min(1e-12)).item()
            )
        elif args.objective == "mis":
            objective_estimate = float(est_mix["ips"])
            objective_estimate_q = float(est_mix_q["ips"])
        else:
            objective_estimate = float(est[args.objective])
            objective_estimate_q = float(est_q[args.objective])

        with torch.no_grad():
            beta_mu = policy.beta_mu.detach().cpu().numpy().astype(float)
            beta_sigma = policy.beta_sigma.detach().cpu().numpy().astype(float)

        b_mu = np.array([float(behavior_row.get(f"beta_mu_{d}", 0.0)) for d in range(args.mu_degree + 1)], dtype=float)
        b_sig = np.array([float(behavior_row.get(f"beta_sigma_{d}", 0.0)) for d in range(args.sigma_degree + 1)], dtype=float)

        batch_theta = _bg["theta"].astype(float)
        behavior_emp = float(_bg["reward"].mean())
        behavior_emp_q = behavior_emp
        est_batch_q = {
            "ips": float("nan"),
            "snips": float("nan"),
            "dr": float("nan"),
        }
        est_batch_mix_q = {
            "ips": float("nan"),
        }
        if len(batch_theta) > 0:
            batch_ctx_np = np.ones_like(batch_theta, dtype=np.float32)
            if theta_ref_np is not None:
                batch_ctx_np = _compute_context_ratio_hist(
                    theta_ref=theta_ref_np.astype(float),
                    theta_cur=batch_theta,
                    theta_eval=batch_theta,
                    n_bins=args.context_bins,
                    ratio_clip=args.context_ratio_clip,
                )
            behavior_emp_q = _weighted_mean(_bg["reward"].astype(float), batch_ctx_np.astype(float))
            logb_batch_np = _bg["log_propensity"]
            logb_batch_mix_np = compute_log_mixture_propensity(
                theta=batch_theta,
                delta=_bg["delta"].astype(float),
                orders=_bg["order_sequence"],
                fit_df=fit_df,
                mu_degree=args.mu_degree,
                sigma_degree=args.sigma_degree,
                sigma_floor=args.sigma_floor,
            )
            est_batch_q = _compute_estimators(
                policy,
                theta=torch.from_numpy(_bg["theta"]),
                delta=torch.from_numpy(_bg["delta"]),
                reward=torch.from_numpy(_bg["reward"]),
                log_prop_b=torch.from_numpy(logb_batch_np.astype(np.float32)),
                delta_min=delta_min,
                delta_max=delta_max,
                dm_delta_grid=args.dm_delta_grid,
                max_weight=args.max_weight,
                clip_c=args.clip_c,
                device=device,
                context_w=torch.from_numpy(batch_ctx_np.astype(np.float32)),
            )
            est_batch_mix_q = _compute_estimators(
                policy,
                theta=torch.from_numpy(_bg["theta"]),
                delta=torch.from_numpy(_bg["delta"]),
                reward=torch.from_numpy(_bg["reward"]),
                log_prop_b=torch.from_numpy(logb_batch_mix_np.astype(np.float32)),
                delta_min=delta_min,
                delta_max=delta_max,
                dm_delta_grid=args.dm_delta_grid,
                max_weight=args.max_weight,
                clip_c=args.clip_c,
                device=device,
                context_w=torch.from_numpy(batch_ctx_np.astype(np.float32)),
            )
        behavior_policy = GlobalGaussianPolicy(
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            delta_min=delta_min,
            delta_max=delta_max,
            bound_mean_to_action_range=False,
        ).to(device)
        with torch.no_grad():
            behavior_policy.beta_mu.copy_(torch.tensor(b_mu, dtype=behavior_policy.beta_mu.dtype, device=device))
            behavior_policy.beta_sigma.copy_(torch.tensor(b_sig, dtype=behavior_policy.beta_sigma.dtype, device=device))

        behavior_irt = dm_policy_value(
            behavior_policy,
            theta=batch_theta,
            delta_min=delta_min,
            delta_max=delta_max,
            n_delta_grid=args.dm_delta_grid,
            device=device,
        )
        learned_irt = dm_policy_value(
            policy,
            theta=batch_theta,
            delta_min=delta_min,
            delta_max=delta_max,
            n_delta_grid=args.dm_delta_grid,
            device=device,
        )
        optimal_irt = optimal_irt_value(
            theta=batch_theta,
            delta_min=delta_min,
            delta_max=delta_max,
        )
        behavior_irt_q = behavior_irt
        learned_irt_q = learned_irt
        optimal_irt_q = optimal_irt
        if theta_ref_np is not None and len(theta_ref_np) > 0:
            behavior_irt_q = dm_policy_value(
                behavior_policy,
                theta=theta_ref_np.astype(float),
                delta_min=delta_min,
                delta_max=delta_max,
                n_delta_grid=args.dm_delta_grid,
                device=device,
            )
            learned_irt_q = dm_policy_value(
                policy,
                theta=theta_ref_np.astype(float),
                delta_min=delta_min,
                delta_max=delta_max,
                n_delta_grid=args.dm_delta_grid,
                device=device,
            )
            optimal_irt_q = optimal_irt_value(
                theta=theta_ref_np.astype(float),
                delta_min=delta_min,
                delta_max=delta_max,
            )

        coeff_rec: dict[str, object] = {
            "round": int(round_idx),
            "order_sequence": int(order_val),
            "n_train": int(n_train),
            "n_seen": int(n_seen),
            "n_components": int(n_components),
            "objective": args.objective,
            "denominator": args.denominator,
            "objective_denominator": "mixture" if args.objective == "mis" else "logged",
            "train_scope": args.train_scope,
            "context_target": args.context_target,
            "context_ref_label": context_ref_label,
            "balanced_minibatch": bool(args.balanced_minibatch),
            "context_match_sampling": bool(args.context_match_sampling),
            "train_only_last_round": bool(args.train_only_last_round),
            "dynamic_batch_size": bool(args.dynamic_batch_size),
            "target_steps_per_epoch": int(args.target_steps_per_epoch),
            "batch_size_used": int(batch_size_used),
        }
        for d, v in enumerate(beta_mu):
            coeff_rec[f"beta_mu_{d}"] = float(v)
        for d, v in enumerate(beta_sigma):
            coeff_rec[f"beta_sigma_{d}"] = float(v)
        coeff_rows.append(coeff_rec)

        summary_rows.append(
            {
                "round": int(round_idx),
                "order_sequence": int(order_val),
                "n_batch": int(len(batch_theta)),
                "n_train": int(n_train),
                "n_seen": int(n_seen),
                "n_components": int(n_components),
                "objective": args.objective,
                "denominator": args.denominator,
                "objective_denominator": "mixture" if args.objective == "mis" else "logged",
                "train_scope": args.train_scope,
                "context_target": args.context_target,
                "context_ref_label": context_ref_label,
                "balanced_minibatch": bool(args.balanced_minibatch),
                "context_match_sampling": bool(args.context_match_sampling),
                "train_only_last_round": bool(args.train_only_last_round),
                "dynamic_batch_size": bool(args.dynamic_batch_size),
                "target_steps_per_epoch": int(args.target_steps_per_epoch),
                "batch_size_used": int(batch_size_used),
                "did_train_round": bool(did_train_round),
                "behavior_empirical_batch_mean": behavior_emp,
                "behavior_empirical_batch_mean_q": behavior_emp_q,
                "behavior_irt_value": behavior_irt,
                "behavior_irt_value_q": behavior_irt_q,
                "learned_irt_value": learned_irt,
                "learned_irt_value_q": learned_irt_q,
                "optimal_irt_value": optimal_irt,
                "optimal_irt_value_q": optimal_irt_q,
                "objective_estimate": objective_estimate,
                "objective_estimate_q": objective_estimate_q,
                "mis": est_mix["ips"],
                "mis_q_batch": est_batch_mix_q["ips"],
                "ips_q_batch": est_batch_q["ips"],
                "snips_q_batch": est_batch_q["snips"],
                "dr_q_batch": est_batch_q["dr"],
                "mis_q_train": float(est_mix_q["ips"]),
                **{
                    f"{k}_q_train": float(v)
                    for k, v in est_q.items()
                    if k in {"ips", "snips", "dr", "ess", "ess_clip"}
                },
                **est,
            }
        )

        print(
            f"round={round_idx}/{len(order_list)} order={order_val} "
            f"batch_size={batch_size_used} n_train={n_train} "
            f"trained={int(did_train_round)} "
            f"learned_irt={learned_irt:.4f} behavior_irt={behavior_irt:.4f} "
            f"optimal_irt={optimal_irt:.4f} {args.objective}={objective_estimate:.4f} "
            f"{args.objective}_q={objective_estimate_q:.4f} learned_irt_q={learned_irt_q:.4f}"
        )

    round_df = pd.DataFrame(summary_rows)
    coeff_df = pd.DataFrame(coeff_rows)

    round_path = out_dir / "policy_round_results.csv"
    coeff_path = out_dir / "policy_coefficients_by_round.csv"
    round_df.to_csv(round_path, index=False)
    coeff_df.to_csv(coeff_path, index=False)
    value_table_cols = [
        "round",
        "order_sequence",
        "objective",
        "objective_estimate",
        "objective_estimate_q",
        "context_target",
        "context_ref_label",
        "balanced_minibatch",
        "context_match_sampling",
        "dynamic_batch_size",
        "target_steps_per_epoch",
        "batch_size_used",
        "behavior_empirical_batch_mean",
        "behavior_empirical_batch_mean_q",
        "behavior_irt_value",
        "behavior_irt_value_q",
        "learned_irt_value",
        "learned_irt_value_q",
        "optimal_irt_value",
        "optimal_irt_value_q",
        "mis",
        "ips",
        "snips",
        "dr",
        "mis_q_batch",
        "ips_q_batch",
        "snips_q_batch",
        "dr_q_batch",
        "mis_q_train",
        "ips_q_train",
        "snips_q_train",
        "dr_q_train",
        "ess_q_train",
        "ess_clip_q_train",
        "ess",
        "ess_clip",
    ]
    value_table = round_df[[c for c in value_table_cols if c in round_df.columns]].copy()
    value_table_path = out_dir / "policy_value_table.csv"
    value_table.to_csv(value_table_path, index=False)

    fit_merge = fit_df[["order_sequence", "n_obs", "nll", "rmse", "mae"]].rename(
        columns={
            "n_obs": "gaussian_n_obs",
            "nll": "gaussian_nll",
            "rmse": "gaussian_rmse",
            "mae": "gaussian_mae",
        }
    )
    merged_table = round_df.merge(fit_merge, on="order_sequence", how="left")
    merged_path = out_dir / "policy_round_results_with_gaussian_fit.csv"
    merged_table.to_csv(merged_path, index=False)

    fit_summary = pd.DataFrame(
        [
            {
                "orders": int(fit_df["order_sequence"].nunique()),
                "gaussian_nll_mean": float(fit_df["nll"].mean()),
                "gaussian_rmse_mean": float(fit_df["rmse"].mean()),
                "gaussian_mae_mean": float(fit_df["mae"].mean()),
            }
        ]
    )
    fit_summary_path = out_dir / "gaussian_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)

    if not args.skip_plots:
        _plot_gaussian_fit_quality(fit_df, str(out_dir / "gaussian_fit_quality_by_order.png"))
        _plot_policy_values(round_df, str(out_dir / "policy_values_by_round.png"))

    animation_path = None
    if args.make_animation:
        animation_path = out_dir / args.animation_name
        _make_policy_animation(
            round_df=round_df,
            coeff_df=coeff_df,
            fit_df=fit_df,
            tuples_df=offpolicy_df,
            out_gif=str(animation_path),
            fps=args.animation_fps,
            max_scatter=args.animation_max_scatter,
            y_min=args.animation_y_min,
            y_max=args.animation_y_max,
        )

    print(f"saved_ktm={ktm_path.resolve()}")
    print(f"saved_round_mapping={round_map_path.resolve()}")
    print(f"saved_gaussian_fit={fit_path.resolve()}")
    print(f"saved_offpolicy_tuples={offpolicy_path.resolve()}")
    print(f"saved_round_results={round_path.resolve()}")
    print(f"saved_policy_value_table={value_table_path.resolve()}")
    print(f"saved_coefficients={coeff_path.resolve()}")
    print(f"saved_round_plus_fit={merged_path.resolve()}")
    print(f"saved_gaussian_fit_summary={fit_summary_path.resolve()}")
    if not args.skip_plots:
        print(f"saved_plot_gaussian={ (out_dir / 'gaussian_fit_quality_by_order.png').resolve() }")
        print(f"saved_plot_policy={ (out_dir / 'policy_values_by_round.png').resolve() }")
    if animation_path is not None:
        print(f"saved_animation={animation_path.resolve()}")


if __name__ == "__main__":
    main()
