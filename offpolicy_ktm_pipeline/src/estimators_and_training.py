from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.special import lambertw, logsumexp, ndtr, ndtri
from torch.utils.data import DataLoader, TensorDataset

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy
from .utils import get_default_device, set_global_seed

_GL_CACHE: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray]] = {}
_DR_OBJECTIVES = {"dr", "sndr", "cdr", "csndr"}


def poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def _log_gaussian_pdf(delta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (delta - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z ** 2)


def _extract_degree(cols: Iterable[str], prefix: str) -> int:
    idx = [int(c.split("_")[-1]) for c in cols if c.startswith(prefix)]
    if not idx:
        raise ValueError(f"Missing {prefix}* columns")
    return max(idx)


def _build_behavior_lookup(
    fit_df: pd.DataFrame,
    *,
    mu_degree: int,
    sigma_degree: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    fit = fit_df.copy()
    if "order_sequence" not in fit.columns:
        if "order_min" in fit.columns and "order_max" in fit.columns:
            fit = fit[fit["order_min"] == fit["order_max"]].copy()
            fit["order_sequence"] = fit["order_min"].astype(int)
        else:
            raise ValueError("fit_df must contain order_sequence or order_min/order_max")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in fit.iterrows():
        order_val = int(row["order_sequence"])
        b_mu = np.array([float(row.get(f"beta_mu_{d}", 0.0)) for d in range(mu_degree + 1)], dtype=float)
        b_sig = np.array([float(row.get(f"beta_sigma_{d}", 0.0)) for d in range(sigma_degree + 1)], dtype=float)
        out[order_val] = (b_mu, b_sig)
    return out


def _log_diff_exp(log_x: np.ndarray, log_y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.empty_like(log_x)
    mask = log_x > log_y
    out[mask] = log_x[mask] + np.log1p(-np.exp(log_y[mask] - log_x[mask]))
    out[~mask] = np.log(eps)
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _gauss_legendre_nodes_weights(delta_min: float, delta_max: float, n_points: int) -> tuple[np.ndarray, np.ndarray]:
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


def compute_log_mixture_propensity(
    *,
    theta: np.ndarray,
    delta: np.ndarray,
    orders: np.ndarray,
    fit_df: pd.DataFrame,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
) -> np.ndarray:
    """
    log p_mix(delta|theta) where p_mix = sum_m rho_m p_m and rho_m are order frequencies.
    Vectorized: all components evaluated with a single batched matmul.
    """
    lookup = _build_behavior_lookup(fit_df, mu_degree=mu_degree, sigma_degree=sigma_degree)
    comp_counts = pd.Series(orders).value_counts().sort_index()
    comp_orders = comp_counts.index.to_numpy(dtype=int)
    comp_weights = comp_counts.to_numpy(dtype=float)
    comp_weights = comp_weights / float(comp_weights.sum())

    dist = "gaussian"
    if "distribution" in fit_df.columns:
        vals = fit_df["distribution"].dropna().astype(str)
        if len(vals) > 0:
            dist = vals.iloc[0]
    delta_min = float(fit_df["delta_min"].iloc[0]) if ("delta_min" in fit_df.columns) else float(np.min(delta))
    delta_max = float(fit_df["delta_max"].iloc[0]) if ("delta_max" in fit_df.columns) else float(np.max(delta))

    # Filter to orders present in the lookup table.
    valid_pairs = [(int(o), float(rho)) for o, rho in zip(comp_orders, comp_weights) if int(o) in lookup]
    if not valid_pairs:
        return np.full(theta.shape[0], np.log(1e-12), dtype=np.float32)

    valid_orders = [o for o, _ in valid_pairs]
    valid_rhos = np.array([rho for _, rho in valid_pairs], dtype=float)

    # Stack coefficient matrices: (n_comp, degree+1)
    all_b_mu = np.stack([lookup[o][0] for o in valid_orders])   # (n_comp, mu_deg+1)
    all_b_sig = np.stack([lookup[o][1] for o in valid_orders])  # (n_comp, sig_deg+1)

    # Polynomial design matrices: (n, degree+1)
    theta_f = theta.astype(float)
    delta_f = delta.astype(float)
    x_mu = np.column_stack([theta_f ** d for d in range(mu_degree + 1)])    # (n, mu_deg+1)
    x_sig = np.column_stack([theta_f ** d for d in range(sigma_degree + 1)])  # (n, sig_deg+1)

    # All means and log-sigmas in one matmul: (n, n_comp)
    mu_all = x_mu @ all_b_mu.T                                      # (n, n_comp)
    sigma_all = np.maximum(np.exp(x_sig @ all_b_sig.T), sigma_floor)  # (n, n_comp)

    # Gaussian log-pdf: (n, n_comp)
    z = (delta_f[:, None] - mu_all) / sigma_all
    log_p_all = -0.5 * np.log(2.0 * np.pi) - np.log(sigma_all) - 0.5 * (z ** 2)

    if dist == "truncated_gaussian":
        alpha = (float(delta_min) - mu_all) / sigma_all   # (n, n_comp)
        beta_nd = (float(delta_max) - mu_all) / sigma_all  # (n, n_comp)
        cdf_beta = np.clip(ndtr(beta_nd), 1e-12, 1.0)
        cdf_alpha = np.clip(ndtr(alpha), 1e-12, 1.0)
        log_z = _log_diff_exp(np.log(cdf_beta), np.log(cdf_alpha), eps=1e-12)
        log_p_all = log_p_all - log_z
        in_support = (delta_f[:, None] >= float(delta_min)) & (delta_f[:, None] <= float(delta_max))
        log_p_all = np.where(in_support, log_p_all, np.log(1e-12))

    # Weighted logsumexp over components: (n,)
    log_mix = logsumexp(log_p_all + np.log(valid_rhos)[None, :], axis=1)
    return log_mix.astype(np.float32)


def _dm_expected_reward_per_theta(
    policy: GlobalGaussianPolicy,
    *,
    theta: torch.Tensor,
    delta_min: float,
    delta_max: float,
    n_grid: int,
) -> torch.Tensor:
    nodes_np, w_np = _gauss_legendre_nodes_weights(delta_min=delta_min, delta_max=delta_max, n_points=n_grid)
    nodes = torch.as_tensor(nodes_np, device=theta.device, dtype=theta.dtype)
    quad_w = torch.as_tensor(w_np, device=theta.device, dtype=theta.dtype).clamp_min(1e-12)
    log_quad_w = torch.log(quad_w)[None, :]
    n = int(nodes.shape[0])
    bsz = int(theta.shape[0])
    # Keep memory bounded on large datasets while preserving differentiability.
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


@torch.no_grad()
def _evaluate_estimators(
    policy: GlobalGaussianPolicy,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    log_prop_b: torch.Tensor,
    *,
    max_weight: float,
    clip_c: float,
    device: torch.device,
) -> dict[str, float]:
    policy.eval()
    th = theta.to(device)
    de = delta.to(device)
    rw = reward.to(device)
    lb = log_prop_b.to(device)

    log_p_new = policy.log_prob(delta=de, theta=th)
    log_ratio = torch.clamp(log_p_new - lb, min=-20.0, max=float(np.log(max_weight)))
    w = torch.exp(log_ratio)
    w_clip = torch.clamp(w, max=float(clip_c))

    ips = torch.mean(w * rw)
    snips = torch.sum(w * rw) / torch.sum(w).clamp_min(1e-12)
    cips = torch.mean(w_clip * rw)
    csnips = torch.sum(w_clip * rw) / torch.sum(w_clip).clamp_min(1e-12)
    delta_min = float(policy.delta_min) if policy.delta_min is not None else float(torch.min(de).item())
    delta_max = float(policy.delta_max) if policy.delta_max is not None else float(torch.max(de).item())
    q_logged = torch.sigmoid(th - de) * (de - delta_min)
    q_pi = _dm_expected_reward_per_theta(
        policy,
        theta=th,
        delta_min=delta_min,
        delta_max=delta_max,
        n_grid=121,
    )
    resid = rw - q_logged
    dr = torch.mean(q_pi + w * resid)
    sndr = torch.mean(q_pi) + (torch.sum(w * resid) / torch.sum(w).clamp_min(1e-12))
    cdr = torch.mean(q_pi + w_clip * resid)
    csndr = torch.mean(q_pi) + (torch.sum(w_clip * resid) / torch.sum(w_clip).clamp_min(1e-12))
    ess = (torch.sum(w) ** 2) / torch.sum(w ** 2).clamp_min(1e-12)
    ess_clip = (torch.sum(w_clip) ** 2) / torch.sum(w_clip ** 2).clamp_min(1e-12)
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


def get_policy_coefficients(policy: GlobalGaussianPolicy) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        b_mu = policy.beta_mu.detach().cpu().numpy().astype(float).copy()
        b_sig = policy.beta_sigma.detach().cpu().numpy().astype(float).copy()
    return b_mu, b_sig


def train_global_policy(
    df: pd.DataFrame,
    *,
    fit_df: pd.DataFrame | None = None,
    objective: str = "snips",
    denominator: str = "logged",
    theta_col: str = "proficiency",
    delta_col: str = "difficulties",
    reward_col: str = "reward",
    order_col: str = "order_sequence",
    log_prop_col: str = "log_propensity",
    mu_degree: int = 1,
    sigma_degree: int = 2,
    sigma_floor: float = 0.2,
    max_weight: float = 20.0,
    clip_c: float = 10.0,
    epochs: int = 5,
    batch_size: int = 8192,
    lr: float = 0.005,
    l2_coef: float = 1e-6,
    variance_penalty_coef: float = 0.0,
    num_workers: int = 0,
    seed: int = 42,
    init_from_first_behavior: bool = True,
    device: str | None = None,
) -> tuple[GlobalGaussianPolicy, pd.DataFrame, dict[str, np.ndarray]]:
    """
    Train a single global Gaussian policy with IS/DR objectives.
    """
    if objective not in {"ips", "snips", "cips", "csnips", "dr", "sndr", "cdr", "csndr"}:
        raise ValueError(f"Unsupported objective: {objective}")
    if denominator not in {"logged", "mixture"}:
        raise ValueError(f"Unsupported denominator: {denominator}")

    set_global_seed(seed)

    work = df[[theta_col, delta_col, reward_col, order_col, log_prop_col]].dropna().copy()
    theta_np = work[theta_col].to_numpy(dtype=np.float32)
    delta_np = work[delta_col].to_numpy(dtype=np.float32)
    reward_np = work[reward_col].to_numpy(dtype=np.float32)

    if denominator == "mixture":
        if fit_df is None:
            raise ValueError("fit_df is required for denominator='mixture'")
        log_prop_b_np = compute_log_mixture_propensity(
            theta=theta_np.astype(float),
            delta=delta_np.astype(float),
            orders=work[order_col].to_numpy(dtype=int),
            fit_df=fit_df,
            mu_degree=mu_degree,
            sigma_degree=sigma_degree,
            sigma_floor=sigma_floor,
        )
    else:
        log_prop_b_np = work[log_prop_col].to_numpy(dtype=np.float32)

    theta = torch.from_numpy(theta_np)
    delta = torch.from_numpy(delta_np)
    reward = torch.from_numpy(reward_np)
    log_prop_b = torch.from_numpy(log_prop_b_np.astype(np.float32))

    dataset = TensorDataset(theta, delta, reward, log_prop_b)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    target_device = get_default_device(device)
    delta_min = float(np.min(delta_np))
    delta_max = float(np.max(delta_np))
    policy = GlobalGaussianPolicy(
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=sigma_floor,
        delta_min=delta_min,
        delta_max=delta_max,
        bound_mean_to_action_range=True,
    ).to(target_device)
    if init_from_first_behavior and fit_df is not None and len(fit_df) > 0:
        fit = fit_df.copy()
        if "order_sequence" not in fit.columns and "order_min" in fit.columns and "order_max" in fit.columns:
            fit = fit[fit["order_min"] == fit["order_max"]].copy()
            fit["order_sequence"] = fit["order_min"].astype(int)
        if "order_sequence" in fit.columns and not fit.empty:
            first_row = fit.sort_values("order_sequence").iloc[0]
            policy.init_from_behavior_row(first_row)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    init_metrics = _evaluate_estimators(
        policy,
        theta=theta,
        delta=delta,
        reward=reward,
        log_prop_b=log_prop_b,
        max_weight=max_weight,
        clip_c=clip_c,
        device=target_device,
    )
    history.append({"epoch": 0, "train_loss": np.nan, "objective": objective, "denominator": denominator, **init_metrics})

    for epoch in range(1, epochs + 1):
        policy.train()
        running = 0.0
        n_seen = 0
        for b_theta, b_delta, b_reward, b_logb in loader:
            b_theta = b_theta.to(target_device)
            b_delta = b_delta.to(target_device)
            b_reward = b_reward.to(target_device)
            b_logb = b_logb.to(target_device)

            optimizer.zero_grad(set_to_none=True)
            log_p_new = policy.log_prob(delta=b_delta, theta=b_theta)
            log_ratio = torch.clamp(log_p_new - b_logb, min=-20.0, max=float(np.log(max_weight)))
            w = torch.exp(log_ratio)
            w_clip = torch.clamp(w, max=float(clip_c))

            ips_obj = torch.mean(w * b_reward)
            snips_obj = torch.sum(w * b_reward) / torch.sum(w).clamp_min(1e-12)
            cips_obj = torch.mean(w_clip * b_reward)
            csnips_obj = torch.sum(w_clip * b_reward) / torch.sum(w_clip).clamp_min(1e-12)
            q_pi = None
            resid = None
            if objective in _DR_OBJECTIVES:
                q_logged = torch.sigmoid(b_theta - b_delta) * (b_delta - float(delta_min))
                q_pi = _dm_expected_reward_per_theta(
                    policy,
                    theta=b_theta,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    n_grid=121,
                )
                resid = b_reward - q_logged
                dr_obj = torch.mean(q_pi + w * resid)
                sndr_obj = torch.mean(q_pi) + (torch.sum(w * resid) / torch.sum(w).clamp_min(1e-12))
                cdr_obj = torch.mean(q_pi + w_clip * resid)
                csndr_obj = torch.mean(q_pi) + (torch.sum(w_clip * resid) / torch.sum(w_clip).clamp_min(1e-12))

            if objective == "ips":
                obj = ips_obj
            elif objective == "snips":
                obj = snips_obj
            elif objective == "cips":
                obj = cips_obj
            elif objective == "csnips":
                obj = csnips_obj
            elif objective == "dr":
                obj = dr_obj
            elif objective == "sndr":
                obj = sndr_obj
            elif objective == "cdr":
                obj = cdr_obj
            else:
                obj = csndr_obj

            if objective == "ips":
                contrib = w * b_reward
            elif objective == "snips":
                contrib = (w / torch.sum(w).clamp_min(1e-12)) * b_reward
            elif objective == "cips":
                contrib = w_clip * b_reward
            elif objective == "csnips":
                contrib = (w_clip / torch.sum(w_clip).clamp_min(1e-12)) * b_reward
            elif objective == "dr":
                assert q_pi is not None and resid is not None
                contrib = q_pi + w * resid
            elif objective == "sndr":
                assert q_pi is not None and resid is not None
                contrib = q_pi + (w / torch.sum(w).clamp_min(1e-12)) * resid
            elif objective == "cdr":
                assert q_pi is not None and resid is not None
                contrib = q_pi + w_clip * resid
            else:
                assert q_pi is not None and resid is not None
                contrib = q_pi + (w_clip / torch.sum(w_clip).clamp_min(1e-12)) * resid
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

            bs = b_theta.shape[0]
            running += float(loss.item()) * bs
            n_seen += bs

        metrics = _evaluate_estimators(
            policy,
            theta=theta,
            delta=delta,
            reward=reward,
            log_prop_b=log_prop_b,
            max_weight=max_weight,
            clip_c=clip_c,
            device=target_device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(1, n_seen),
                "objective": objective,
                "denominator": denominator,
                **metrics,
            }
        )

    b_mu, b_sig = get_policy_coefficients(policy)
    coeffs = {"beta_mu": b_mu, "beta_sigma": b_sig}
    return policy, pd.DataFrame(history), coeffs


def mc_policy_value(
    *,
    theta: np.ndarray,
    beta_mu: np.ndarray,
    beta_sigma: np.ndarray,
    delta_min: float,
    delta_max: float | None = None,
    sigma_floor: float = 1e-4,
    bound_mean_to_action_range: bool = False,
    n_mc: int = 512,
    seed: int = 42,
) -> float:
    """
    Monte-Carlo estimate of E[r] for:
      r = sigmoid(theta - delta) * (delta - delta_min), delta ~ policy.
    If delta_max is provided, sampling is done from a truncated Gaussian on
    [delta_min, delta_max].
    """
    mu_raw = poly_eval(theta, beta_mu)
    if bound_mean_to_action_range and delta_max is not None and float(delta_max) > float(delta_min):
        width = float(delta_max - delta_min)
        mu = float(delta_min) + width * _sigmoid_np(mu_raw)
    else:
        mu = mu_raw
    sigma = np.maximum(np.exp(poly_eval(theta, beta_sigma)), sigma_floor)
    rng = np.random.default_rng(seed)
    if delta_max is not None and float(delta_max) > float(delta_min):
        a_std = (float(delta_min) - mu) / sigma
        b_std = (float(delta_max) - mu) / sigma
        cdf_a = np.clip(ndtr(a_std), 1e-12, 1.0 - 1e-12)
        cdf_b = np.clip(ndtr(b_std), 1e-12, 1.0 - 1e-12)
        u = rng.uniform(size=(n_mc, theta.shape[0]))
        p = cdf_a[None, :] + u * np.clip((cdf_b - cdf_a)[None, :], 1e-12, None)
        z = ndtri(np.clip(p, 1e-12, 1.0 - 1e-12))
        delta = mu[None, :] + sigma[None, :] * z
        delta = np.clip(delta, float(delta_min), float(delta_max))
    else:
        eps = rng.standard_normal(size=(n_mc, 1))
        delta = mu[None, :] + sigma[None, :] * eps
    prob = 1.0 / (1.0 + np.exp(-(theta[None, :] - delta)))
    reward = prob * (delta - delta_min)
    return float(np.mean(reward))


@torch.no_grad()
def dm_policy_value(
    policy: GlobalGaussianPolicy,
    *,
    theta: np.ndarray,
    delta_min: float,
    delta_max: float,
    n_delta_grid: int = 121,
    device: str | torch.device | None = None,
) -> float:
    """
    Deterministic direct-method estimate of E_theta[E_{delta~pi(.|theta)} r(theta, delta)].
    Uses the exact theta sample provided rather than a histogram approximation.
    """
    theta_np = np.asarray(theta, dtype=np.float32)
    if theta_np.size == 0:
        return float("nan")
    target_device = get_default_device(device)
    theta_t = torch.from_numpy(theta_np).to(target_device)
    q_pi = _dm_expected_reward_per_theta(
        policy,
        theta=theta_t,
        delta_min=delta_min,
        delta_max=delta_max,
        n_grid=n_delta_grid,
    )
    return float(q_pi.mean().item())


def _optimal_delta_star(
    *,
    theta: np.ndarray,
    delta_min: float,
    delta_max: float,
) -> np.ndarray:
    theta_np = np.asarray(theta, dtype=np.float64)
    if theta_np.size == 0:
        return theta_np

    # Closed-form optimizer of sigmoid(theta - delta) * (delta - delta_min):
    # delta*(theta) = delta_min + 1 + W(exp(theta - delta_min - 1)),
    # clipped to the feasible action interval.
    lambert_arg = np.exp(theta_np - float(delta_min) - 1.0)
    delta_star = float(delta_min) + 1.0 + np.real(lambertw(lambert_arg))
    return np.clip(delta_star, float(delta_min), float(delta_max))


def optimal_irt_value(
    *,
    theta: np.ndarray,
    delta_min: float,
    delta_max: float,
    n_delta_grid: int = 1200,
) -> float:
    del n_delta_grid  # Kept for API compatibility; the optimum is now analytic.
    theta_np = np.asarray(theta, dtype=np.float64)
    if theta_np.size == 0:
        return float("nan")
    delta_star = _optimal_delta_star(theta=theta_np, delta_min=delta_min, delta_max=delta_max)
    z = theta_np - delta_star
    p = _sigmoid_np(z)
    best = p * (delta_star - float(delta_min))
    return float(np.mean(best))
