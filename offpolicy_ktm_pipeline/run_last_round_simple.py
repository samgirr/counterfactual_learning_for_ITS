from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.offpolicy_gaussian_policy import GlobalGaussianPolicy
from offpolicy_ktm_pipeline.src.dataframe_processing import build_ktm_dataframe
from offpolicy_ktm_pipeline.src.estimators_and_training import compute_log_mixture_propensity
from offpolicy_ktm_pipeline.src.gaussian_regression import fit_gaussian_by_order
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.utils import get_default_device, set_global_seed
from offpolicy_ktm_pipeline.src.visualization import plot_objective_comparison, plot_behavior_evolution, _anchor_indices

_GL_CACHE: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray]] = {}


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


def _dm_expected_reward_per_theta(
    policy: object,
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
    return ratio_bins[idx].astype(np.float32)


def _select_target_round_closest_to_last(
    offpolicy_df: pd.DataFrame,
    *,
    order_col: str,
    theta_col: str,
    n_bins: int,
) -> tuple[int, int, list[int], float]:
    order_list = sorted([int(x) for x in offpolicy_df[order_col].dropna().unique().tolist()])
    if len(order_list) == 0:
        raise ValueError("No rounds found in offpolicy dataframe")
    if len(order_list) == 1:
        return order_list[0], order_list[0], [order_list[0]], 0.0

    last_order = int(order_list[-1])
    k = max(1, len(order_list) // 2)
    first_half = [int(x) for x in order_list[:k]]

    theta_last = offpolicy_df[offpolicy_df[order_col] == last_order][theta_col].to_numpy(dtype=float)
    if theta_last.size == 0:
        raise ValueError(f"No rows for last round order={last_order}")

    theta_first_half = offpolicy_df[offpolicy_df[order_col].isin(first_half)][theta_col].to_numpy(dtype=float)
    lo = float(min(theta_last.min(), theta_first_half.min()))
    hi = float(max(theta_last.max(), theta_first_half.max()))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, max(4, int(n_bins)) + 1)
    p_last = _hist_prob(theta_last, edges)

    best_order = first_half[0]
    best_js = float("inf")
    for cand in first_half:
        theta_c = offpolicy_df[offpolicy_df[order_col] == cand][theta_col].to_numpy(dtype=float)
        if theta_c.size == 0:
            continue
        p_c = _hist_prob(theta_c, edges)
        js = _js_divergence(p_c, p_last)
        if js < best_js:
            best_js = js
            best_order = int(cand)
    return best_order, last_order, first_half, float(best_js)


class IRTOptimalGaussianPolicy:
    def __init__(self, *, delta_min: float, delta_max: float, sigma: float) -> None:
        self.delta_min = float(delta_min)
        self.delta_max = float(delta_max)
        self.sigma = float(max(sigma, 1e-6))

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _log_diff_exp(log_x: torch.Tensor, log_y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        out = torch.full_like(log_x, float(math.log(eps)))
        mask = log_x > log_y
        out[mask] = log_x[mask] + torch.log1p(-torch.exp(log_y[mask] - log_x[mask]))
        return out

    def _optimal_delta(self, theta: torch.Tensor) -> torch.Tensor:
        x = theta - float(self.delta_min)
        t = torch.clamp(x + 1.0, min=1e-5)
        for _ in range(12):
            exp_term = torch.exp(torch.clamp(x - t, min=-40.0, max=40.0))
            f = t - 1.0 - exp_term
            fp = 1.0 + exp_term
            t = t - (f / fp)
        d = float(self.delta_min) + t
        return torch.clamp(d, float(self.delta_min), float(self.delta_max))

    def log_prob(self, delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mu = self._optimal_delta(theta)
        sigma = torch.full_like(mu, float(self.sigma))
        z = (delta - mu) / sigma
        log_pdf = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (z ** 2)

        a = float(self.delta_min)
        b = float(self.delta_max)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        cdf_beta = torch.clamp(self._normal_cdf(beta), min=1e-12, max=1.0)
        cdf_alpha = torch.clamp(self._normal_cdf(alpha), min=1e-12, max=1.0)
        log_z = self._log_diff_exp(torch.log(cdf_beta), torch.log(cdf_alpha), eps=1e-12)
        log_p = log_pdf - log_z
        in_support = (delta >= a) & (delta <= b)
        return torch.where(in_support, log_p, torch.full_like(log_p, float(math.log(1e-12))))


@torch.no_grad()
def _evaluate_estimators(
    policy: object,
    *,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    context_w: torch.Tensor,
    log_prop_logged: torch.Tensor,
    log_prop_mix: torch.Tensor | None,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
    max_weight: float,
    device: torch.device,
) -> dict[str, float]:
    th = theta.to(device)
    de = delta.to(device)
    rw = reward.to(device)
    cw = context_w.to(device)
    lb = log_prop_logged.to(device)
    lm = log_prop_mix.to(device) if log_prop_mix is not None else None

    logp = policy.log_prob(delta=de, theta=th)
    log_ratio_logged = torch.clamp(logp - lb, min=-20.0, max=float(np.log(max_weight)))
    w_logged = torch.exp(log_ratio_logged)
    if lm is not None:
        log_ratio_mix = torch.clamp(logp - lm, min=-20.0, max=float(np.log(max_weight)))
        w_mix = torch.exp(log_ratio_mix)
    else:
        w_mix = None

    cw_sum = torch.sum(cw).clamp_min(1e-12)
    cw_w_sum = torch.sum(cw * w_logged).clamp_min(1e-12)

    q_logged = torch.sigmoid(th - de) * (de - float(delta_min))
    q_pi = _dm_expected_reward_per_theta(
        policy,
        theta=th,
        delta_min=delta_min,
        delta_max=delta_max,
        n_grid=dm_delta_grid,
    )
    resid = rw - q_logged

    ips = torch.sum(cw * w_logged * rw) / cw_sum
    snips = torch.sum(cw * w_logged * rw) / cw_w_sum
    dr = torch.sum(cw * (q_pi + w_logged * resid)) / cw_sum
    dm = torch.sum(cw * q_pi) / cw_sum
    if w_mix is not None:
        mis = torch.sum(cw * w_mix * rw) / cw_sum
        ess_mix = (torch.sum(cw * w_mix) ** 2) / torch.sum((cw * w_mix) ** 2).clamp_min(1e-12)
        mis_val = float(mis.item())
        ess_mix_val = float(ess_mix.item())
    else:
        mis_val = float("nan")
        ess_mix_val = float("nan")

    ess_logged = (torch.sum(cw * w_logged) ** 2) / torch.sum((cw * w_logged) ** 2).clamp_min(1e-12)

    return {
        "ips": float(ips.item()),
        "snips": float(snips.item()),
        "dr": float(dr.item()),
        "mis": mis_val,
        "dm": float(dm.item()),
        "ess_logged": float(ess_logged.item()),
        "ess_mix": ess_mix_val,
    }


def _train_last_round_policy(
    *,
    objective: str,
    theta: torch.Tensor,
    delta: torch.Tensor,
    reward: torch.Tensor,
    context_w: torch.Tensor,
    log_prop_logged: torch.Tensor,
    log_prop_mix: torch.Tensor,
    delta_min: float,
    delta_max: float,
    policy_sigma_floor: float,
    init_row: pd.Series | None,
    dm_delta_grid: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l2_coef: float,
    max_weight: float,
    device: torch.device,
) -> tuple[GlobalGaussianPolicy, pd.DataFrame]:
    policy = GlobalGaussianPolicy(
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(policy_sigma_floor),
        delta_min=float(delta_min),
        delta_max=float(delta_max),
        bound_mean_to_action_range=True,
    ).to(device)
    if init_row is not None:
        policy.init_from_behavior_row(init_row)

    dataset = TensorDataset(theta, delta, reward, context_w, log_prop_logged, log_prop_mix)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        policy.train()
        running = 0.0
        n_seen = 0
        for b_th, b_de, b_rw, b_cw, b_lb, b_lm in loader:
            b_th = b_th.to(device)
            b_de = b_de.to(device)
            b_rw = b_rw.to(device)
            b_cw = b_cw.to(device)
            b_lb = b_lb.to(device)
            b_lm = b_lm.to(device)

            optimizer.zero_grad(set_to_none=True)
            logp = policy.log_prob(delta=b_de, theta=b_th)
            log_ratio_logged = torch.clamp(logp - b_lb, min=-20.0, max=float(np.log(max_weight)))
            log_ratio_mix = torch.clamp(logp - b_lm, min=-20.0, max=float(np.log(max_weight)))
            w_logged = torch.exp(log_ratio_logged)
            w_mix = torch.exp(log_ratio_mix)

            cw_sum = torch.sum(b_cw).clamp_min(1e-12)
            if objective == "ips":
                obj = torch.sum(b_cw * w_logged * b_rw) / cw_sum
            elif objective == "snips":
                obj = torch.sum(b_cw * w_logged * b_rw) / torch.sum(b_cw * w_logged).clamp_min(1e-12)
            elif objective == "mis":
                obj = torch.sum(b_cw * w_mix * b_rw) / cw_sum
            elif objective == "dm":
                q_pi = _dm_expected_reward_per_theta(
                    policy,
                    theta=b_th,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    n_grid=dm_delta_grid,
                )
                obj = torch.sum(b_cw * q_pi) / cw_sum
            else:  # dr
                q_logged = torch.sigmoid(b_th - b_de) * (b_de - float(delta_min))
                q_pi = _dm_expected_reward_per_theta(
                    policy,
                    theta=b_th,
                    delta_min=delta_min,
                    delta_max=delta_max,
                    n_grid=dm_delta_grid,
                )
                obj = torch.sum(b_cw * (q_pi + w_logged * (b_rw - q_logged))) / cw_sum

            reg = float(l2_coef) * (torch.mean(policy.beta_mu ** 2) + torch.mean(policy.beta_sigma ** 2))
            loss = -obj + reg
            loss.backward()
            optimizer.step()

            bs = int(b_th.shape[0])
            running += float(loss.item()) * bs
            n_seen += bs

        history.append(
            {
                "epoch": int(epoch),
                "objective": objective,
                "train_loss": float(running / max(1, n_seen)),
            }
        )
    return policy, pd.DataFrame(history)


def _print_estimator_formulas() -> None:
    print("Context weights: c_i = q_target(x_i) / q_logged(x_i)")
    print("All estimators are means (expectations), not raw sums.")
    print("IPS_ctx   = (sum_i c_i w_i r_i) / (sum_i c_i)")
    print("SNIPS_ctx = (sum_i c_i w_i r_i) / (sum_i c_i w_i)")
    print("DR_ctx    = (sum_i c_i [vhat_pi(x_i) + w_i(r_i-qhat_i)]) / (sum_i c_i)")
    print("MIS_ctx   = (sum_i c_i w_i_mix r_i) / (sum_i c_i)")
    print("DM_ctx    = (sum_i c_i vhat_pi(x_i)) / (sum_i c_i)")


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(x * w) / np.clip(np.sum(w), 1e-12, None))


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
        order_rank = np.arange(len(raw_orders), dtype=int)
        order_round = np.where(order_rank < (max_rounds - 1), order_rank, max_rounds - 1)
    elif strategy == "tail_count_at_n":
        anchor_idx = int(max_rounds - 1)
        target_count = int(counts.iloc[anchor_idx]) if anchor_idx < len(counts) else int(counts.iloc[-1])
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


def _prepare_ktm_dataframe(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    if args.input_is_ktm:
        df = pd.read_csv(args.input_csv)
        required = {"user", "item", "correct", "answer_number", "order_sequence", "proficiency", "difficulties"}
        missing = sorted(required.difference(set(df.columns)))
        if missing:
            raise ValueError(f"--input-is-ktm set but missing columns: {missing}")
        return df.copy()

    return build_ktm_dataframe(
        processed_csv=str(args.input_csv),
        user_col=str(args.user_col),
        item_col=str(args.item_col),
        correct_col=str(args.correct_col),
        skill_col=str(args.skill_col),
        answer_order_col=str(args.answer_order_col),
        derive_answer_number_per_user=bool(args.derive_answer_number_per_user),
        sample_users=int(args.sample_users),
        sample_rows=int(args.sample_rows),
        seed=int(args.seed),
        reduce=False,
        top_skills=100,
        item_quantile=0.75,
        min_user_obs=10,
        c=float(args.ktm_c),
        fit_intercept=False,
        target_std=None,
        clip_quantiles=None,
        strict_1pl=True,
    )


def _build_behavior_policy_from_row(
    row: pd.Series,
    *,
    delta_min: float,
    delta_max: float,
    sigma_floor: float,
    device: torch.device,
) -> GlobalGaussianPolicy:
    policy = GlobalGaussianPolicy(
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=float(sigma_floor),
        delta_min=float(delta_min),
        delta_max=float(delta_max),
        bound_mean_to_action_range=False,
    ).to(device)
    policy.init_from_behavior_row(row)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simplified last-round pipeline: fit KTM (no intercept, strict 1PL), "
            "fit truncated-Gaussian behavior by round, select context target among first N//2 rounds, "
            "train one policy on last round only, evaluate IPS/SNIPS/DR/MIS/DM."
        )
    )
    parser.add_argument("--input-csv", type=str, default="data/pix_data.csv")
    parser.add_argument("--input-is-ktm", action="store_true")
    parser.add_argument("--output-dir", type=str, default="pix_mapping/last_round_simple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="answer_result")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument("--derive-answer-number-per-user", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument(
        "--round-cap-strategy",
        type=str,
        default="tail",
        choices=["tail", "tail_count_at_n", "quantile"],
    )
    parser.add_argument(
        "--truncate-at-round",
        type=int,
        default=0,
        help="Hard-cut: drop all rows with answer_number > N before any processing (0 = disabled).",
    )

    parser.add_argument("--ktm-c", type=float, default=0.1)
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--fit-sigma-floor", type=float, default=1e-4)
    parser.add_argument("--policy-sigma-floor", type=float, default=0.2)
    parser.add_argument("--optimal-sigma", type=float, default=0.2)

    parser.add_argument(
        "--objectives",
        type=str,
        nargs="+",
        default=["ips", "snips", "dr", "mis"],
        choices=["ips", "snips", "dr", "mis"],
        help="Training objectives to run in a single pass (preprocessing is shared).",
    )
    parser.add_argument("--train-scope", type=str, default="cumulative", choices=["cumulative", "current"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--dm-delta-grid", type=int, default=121)

    parser.add_argument("--context-bins", type=int, default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show-formulas", action="store_true")
    parser.add_argument("--skip-adaptive-behavior", action="store_true")
    parser.add_argument(
        "--reuse-artifacts-dir",
        type=str,
        default="",
        help=(
            "Reuse cached preprocessing artifacts from a prior run (ktm_dataframe, gaussian_fit_by_order, "
            "offpolicy_tuples, train_eval_tuples, run_summary). Useful to train different objectives quickly."
        ),
    )
    args = parser.parse_args()

    if args.show_formulas:
        _print_estimator_formulas()

    set_global_seed(int(args.seed))
    device = get_default_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    reuse_dir = Path(args.reuse_artifacts_dir).resolve() if str(args.reuse_artifacts_dir).strip() else None
    objectives: list[str] = list(args.objectives)

    if reuse_dir is None:
        df_ktm = _prepare_ktm_dataframe(args, out_dir)
        if int(args.truncate_at_round) > 0:
            n_before = len(df_ktm)
            df_ktm = df_ktm[df_ktm["answer_number"] <= int(args.truncate_at_round)].copy()
            print(f"truncate_at_round={args.truncate_at_round}: {n_before} -> {len(df_ktm)} rows")
        df_ktm, round_mapping = _apply_round_cap(
            df_ktm,
            max_rounds=int(args.max_rounds),
            strategy=str(args.round_cap_strategy),
        )
        df_ktm.to_csv(out_dir / "ktm_dataframe.csv", index=False)
        round_mapping.to_csv(out_dir / "round_bucket_mapping.csv", index=False)

        delta_min = float(df_ktm["difficulties"].min())
        delta_max = float(df_ktm["difficulties"].max())

        fit_df = fit_gaussian_by_order(
            df_ktm,
            order_col="order_sequence",
            theta_col="proficiency",
            delta_col="difficulties",
            mu_degree=1,
            sigma_degree=2,
            sigma_floor=float(args.fit_sigma_floor),
            min_obs_per_order=int(args.min_obs_per_order),
            distribution="truncated_gaussian",
            delta_min=delta_min,
            delta_max=delta_max,
        )
        if fit_df.empty:
            raise ValueError("No behavior policy fit rows; decrease --min-obs-per-order or check data")
        fit_df.to_csv(out_dir / "gaussian_fit_by_order.csv", index=False)

        offpolicy_df = build_offpolicy_dataset(df_ktm, fit_df, sigma_floor=float(args.fit_sigma_floor))
        offpolicy_df.to_csv(out_dir / "offpolicy_tuples.csv", index=False)

        target_order, last_order, first_half, best_js = _select_target_round_closest_to_last(
            offpolicy_df,
            order_col="order_sequence",
            theta_col="proficiency",
            n_bins=int(args.context_bins),
        )

        target_theta = offpolicy_df[offpolicy_df["order_sequence"] == int(target_order)]["proficiency"].to_numpy(dtype=float)
        last_df = offpolicy_df[offpolicy_df["order_sequence"] == int(last_order)].copy()
        if last_df.empty:
            raise ValueError(f"No rows for last round order={last_order}")

        if args.train_scope == "current":
            train_eval_df = last_df.copy()
        else:
            train_eval_df = offpolicy_df[offpolicy_df["order_sequence"] <= int(last_order)].copy()
        if train_eval_df.empty:
            raise ValueError("No rows available for selected train-scope")

        theta_cur = train_eval_df["proficiency"].to_numpy(dtype=float)
        c_w = _compute_context_ratio_hist(
            theta_ref=target_theta,
            theta_cur=theta_cur,
            theta_eval=theta_cur,
            n_bins=int(args.context_bins),
            ratio_clip=float(args.context_ratio_clip),
        )
        train_eval_df["context_weight"] = c_w

        log_mix_train = compute_log_mixture_propensity(
            theta=train_eval_df["proficiency"].to_numpy(dtype=float),
            delta=train_eval_df["difficulties"].to_numpy(dtype=float),
            orders=train_eval_df["order_sequence"].to_numpy(dtype=int),
            fit_df=fit_df,
            mu_degree=1,
            sigma_degree=2,
            sigma_floor=float(args.fit_sigma_floor),
        )
        train_eval_df["log_propensity_mix"] = log_mix_train.astype(np.float32)
        train_eval_df.to_csv(out_dir / "train_eval_tuples.csv", index=False)
    else:
        required_files = [
            reuse_dir / "ktm_dataframe.csv",
            reuse_dir / "gaussian_fit_by_order.csv",
            reuse_dir / "offpolicy_tuples.csv",
            reuse_dir / "train_eval_tuples.csv",
            reuse_dir / "run_summary.csv",
        ]
        missing = [str(p) for p in required_files if not p.exists()]
        if missing:
            raise ValueError(f"--reuse-artifacts-dir missing required files: {missing}")

        df_ktm = pd.read_csv(reuse_dir / "ktm_dataframe.csv")
        fit_df = pd.read_csv(reuse_dir / "gaussian_fit_by_order.csv")
        offpolicy_df = pd.read_csv(reuse_dir / "offpolicy_tuples.csv")
        train_eval_df = pd.read_csv(reuse_dir / "train_eval_tuples.csv")
        src_summary = pd.read_csv(reuse_dir / "run_summary.csv").iloc[0]

        src_train_scope = str(src_summary.get("train_scope", ""))
        if src_train_scope and src_train_scope != str(args.train_scope):
            raise ValueError(
                f"train_scope mismatch with reused artifacts: src={src_train_scope}, requested={args.train_scope}"
            )

        delta_min = float(df_ktm["difficulties"].min())
        delta_max = float(df_ktm["difficulties"].max())
        target_order = int(src_summary["target_round_first_half"])
        last_order = int(src_summary["round_last"])
        best_js = float(src_summary["target_selection_mean_js_to_last"])
        target_theta = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == int(target_order)][
            "proficiency"
        ].to_numpy(dtype=float)
        if target_theta.size == 0:
            raise ValueError(f"Reused offpolicy tuples contain no rows for target_round_first_half={target_order}")

        first_half_raw = str(src_summary.get("first_half_rounds", ""))
        first_half = [int(x) for x in first_half_raw.split(",") if str(x).strip() != ""]

        last_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == int(last_order)].copy()
        if last_df.empty:
            raise ValueError(f"Reused offpolicy tuples contain no rows for round_last={last_order}")

        # Keep I/O light in reuse mode: avoid rewriting large cached artifacts.
        pd.DataFrame(
            [
                {
                    "reused_from": str(reuse_dir),
                    "objectives": ",".join(objectives),
                    "train_scope": str(args.train_scope),
                }
            ]
        ).to_csv(out_dir / "reuse_manifest.csv", index=False)

    theta_t = torch.from_numpy(train_eval_df["proficiency"].to_numpy(np.float32))
    delta_t = torch.from_numpy(train_eval_df["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(train_eval_df["reward"].to_numpy(np.float32))
    cw_t = torch.from_numpy(train_eval_df["context_weight"].to_numpy(np.float32))
    lb_t = torch.from_numpy(train_eval_df["log_propensity"].to_numpy(np.float32))
    lm_t = torch.from_numpy(train_eval_df["log_propensity_mix"].to_numpy(np.float32))

    fit_last = fit_df[fit_df["order_sequence"].astype(int) == int(last_order)]
    init_row = fit_last.iloc[0] if not fit_last.empty else fit_df.sort_values("order_sequence").iloc[-1]

    # Keep adaptive behavior policy values across rounds.
    behavior_rows: list[dict[str, float | int]] = []
    if reuse_dir is None and not bool(args.skip_adaptive_behavior):
        order_list = sorted([int(x) for x in offpolicy_df["order_sequence"].dropna().astype(int).unique().tolist()])
        anchor_idxs = set(_anchor_indices(len(order_list)))
        for i, order_val in enumerate(order_list):
            if i not in anchor_idxs:
                continue
            seen_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= int(order_val)].copy()
            batch_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == int(order_val)].copy()
            if seen_df.empty or batch_df.empty:
                continue

            c_seen = _compute_context_ratio_hist(
                theta_ref=target_theta,
                theta_cur=seen_df["proficiency"].to_numpy(dtype=float),
                theta_eval=seen_df["proficiency"].to_numpy(dtype=float),
                n_bins=int(args.context_bins),
                ratio_clip=float(args.context_ratio_clip),
            ).astype(np.float32)
            c_batch = _compute_context_ratio_hist(
                theta_ref=target_theta,
                theta_cur=batch_df["proficiency"].to_numpy(dtype=float),
                theta_eval=batch_df["proficiency"].to_numpy(dtype=float),
                n_bins=int(args.context_bins),
                ratio_clip=float(args.context_ratio_clip),
            ).astype(np.float32)

            log_mix_seen = compute_log_mixture_propensity(
                theta=seen_df["proficiency"].to_numpy(dtype=float),
                delta=seen_df["difficulties"].to_numpy(dtype=float),
                orders=seen_df["order_sequence"].to_numpy(dtype=int),
                fit_df=fit_df,
                mu_degree=1,
                sigma_degree=2,
                sigma_floor=float(args.fit_sigma_floor),
            )

            brow = fit_df[fit_df["order_sequence"].astype(int) == int(order_val)]
            if brow.empty:
                continue
            b_policy = _build_behavior_policy_from_row(
                brow.iloc[0],
                delta_min=delta_min,
                delta_max=delta_max,
                sigma_floor=float(args.fit_sigma_floor),
                device=device,
            )

            est_seen = _evaluate_estimators(
                b_policy,
                theta=torch.from_numpy(seen_df["proficiency"].to_numpy(np.float32)),
                delta=torch.from_numpy(seen_df["difficulties"].to_numpy(np.float32)),
                reward=torch.from_numpy(seen_df["reward"].to_numpy(np.float32)),
                context_w=torch.from_numpy(c_seen),
                log_prop_logged=torch.from_numpy(seen_df["log_propensity"].to_numpy(np.float32)),
                log_prop_mix=torch.from_numpy(log_mix_seen.astype(np.float32)),
                delta_min=delta_min,
                delta_max=delta_max,
                dm_delta_grid=int(args.dm_delta_grid),
                max_weight=float(args.max_weight),
                device=device,
            )

            behavior_rows.append(
                {
                    "round_order_sequence": int(order_val),
                    "n_seen": int(len(seen_df)),
                    "n_batch": int(len(batch_df)),
                    "target_round_first_half": int(target_order),
                    "target_selection_mean_js_to_last": float(best_js),
                    "batch_reward_mean": float(batch_df["reward"].mean()),
                    "batch_reward_mean_ctx": _weighted_mean(
                        batch_df["reward"].to_numpy(dtype=float),
                        c_batch.astype(float),
                    ),
                    "ips_ctx_seen": float(est_seen["ips"]),
                    "snips_ctx_seen": float(est_seen["snips"]),
                    "dr_ctx_seen": float(est_seen["dr"]),
                    "mis_ctx_seen": float(est_seen["mis"]),
                    "dm_ctx_seen": float(est_seen["dm"]),
                    "ess_logged_ctx_seen": float(est_seen["ess_logged"]),
                    "ess_mix_ctx_seen": float(est_seen["ess_mix"]),
                }
            )

    behavior_df = pd.DataFrame(behavior_rows)
    if reuse_dir is not None and not bool(args.skip_adaptive_behavior):
        cached_behavior = reuse_dir / "adaptive_behavior_by_round.csv"
        if cached_behavior.exists():
            behavior_df = pd.read_csv(cached_behavior)
    behavior_df.to_csv(out_dir / "adaptive_behavior_by_round.csv", index=False)

    # --- Train one policy per objective (preprocessing is shared) ---
    train_start = time.perf_counter()
    learned_policies: dict[str, GlobalGaussianPolicy] = {}
    all_histories: list[pd.DataFrame] = []
    all_coef_rows: list[dict] = []

    for obj in objectives:
        print(f"training objective={obj} ...")
        policy, hist_df = _train_last_round_policy(
            objective=obj,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            context_w=cw_t,
            log_prop_logged=lb_t,
            log_prop_mix=lm_t,
            delta_min=delta_min,
            delta_max=delta_max,
            policy_sigma_floor=float(args.policy_sigma_floor),
            init_row=init_row,
            dm_delta_grid=int(args.dm_delta_grid),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            l2_coef=float(args.l2_coef),
            max_weight=float(args.max_weight),
            device=device,
        )
        learned_policies[obj] = policy
        all_histories.append(hist_df)
        with torch.no_grad():
            all_coef_rows.append({
                "objective": obj,
                "beta_mu_0": float(policy.beta_mu[0].item()),
                "beta_mu_1": float(policy.beta_mu[1].item()),
                "beta_sigma_0": float(policy.beta_sigma[0].item()),
                "beta_sigma_1": float(policy.beta_sigma[1].item()),
                "beta_sigma_2": float(policy.beta_sigma[2].item()),
            })

    train_time = time.perf_counter() - train_start
    pd.concat(all_histories, ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    coeff_df = pd.DataFrame(all_coef_rows)
    coeff_df.to_csv(out_dir / "learned_policy_coefficients.csv", index=False)

    behavior_policy = _build_behavior_policy_from_row(
        init_row,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor),
        device=device,
    )
    optimal_policy = IRTOptimalGaussianPolicy(
        delta_min=delta_min,
        delta_max=delta_max,
        sigma=float(args.optimal_sigma),
    )

    eval_rows: list[dict[str, float | str]] = []
    named_policies: list[tuple[str, str, object]] = [
        ("behavior", "n/a", behavior_policy),
        ("optimal_irt_gaussian", "n/a", optimal_policy),
    ] + [(f"learned_{obj}", obj, pol.eval()) for obj, pol in learned_policies.items()]

    for name, trained_obj, pol in named_policies:
        t_eval0 = time.perf_counter()
        m = _evaluate_estimators(
            pol,
            theta=theta_t,
            delta=delta_t,
            reward=reward_t,
            context_w=cw_t,
            log_prop_logged=lb_t,
            log_prop_mix=lm_t,
            delta_min=delta_min,
            delta_max=delta_max,
            dm_delta_grid=int(args.dm_delta_grid),
            max_weight=float(args.max_weight),
            device=device,
        )
        t_eval = time.perf_counter() - t_eval0
        eval_rows.append(
            {
                "policy": name,
                "trained_objective": trained_obj,
                "train_scope": args.train_scope,
                "round_last": int(last_order),
                "target_round_first_half": int(target_order),
                "target_selection_mean_js_to_last": float(best_js),
                "n_train_eval": int(len(train_eval_df)),
                "n_last_round": int(len(last_df)),
                "ips": float(m["ips"]),
                "snips": float(m["snips"]),
                "dr": float(m["dr"]),
                "mis": float(m["mis"]),
                "dm": float(m["dm"]),
                "ess_logged": float(m["ess_logged"]),
                "ess_mix": float(m["ess_mix"]),
                "eval_time_s": float(t_eval),
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "estimator_results_last_round.csv", index=False)

    # --- Plots ---
    theta_np = train_eval_df["proficiency"].to_numpy(dtype=float)
    theta_grid = np.linspace(float(theta_np.min()), float(theta_np.max()), 400)
    _irt_oracle = IRTOptimalGaussianPolicy(delta_min=delta_min, delta_max=delta_max, sigma=float(args.optimal_sigma))
    with torch.no_grad():
        optimal_delta = _irt_oracle._optimal_delta(
            torch.from_numpy(theta_grid.astype(np.float32))
        ).numpy().astype(float)

    fig1, _ = plot_objective_comparison(
        theta_grid=theta_grid,
        fit_df=fit_df,
        coeff_df=coeff_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.policy_sigma_floor),
        optimal_delta=optimal_delta,
        title=f"Learned Policies vs Behavior (last round={last_order})",
    )
    fig1.savefig(str(out_dir / "policy_comparison_by_objective.png"), dpi=170)
    plt.close(fig1)

    fig2, _ = plot_behavior_evolution(
        theta_grid=theta_grid,
        fit_df=fit_df,
        delta_min=delta_min,
        delta_max=delta_max,
        sigma_floor=float(args.fit_sigma_floor),
        title="Behavior Policy Evolution by Round",
    )
    fig2.savefig(str(out_dir / "behavior_evolution_by_round.png"), dpi=170)
    plt.close(fig2)

    total_time = time.perf_counter() - t0
    summary = pd.DataFrame(
        [
            {
                "objectives": ",".join(objectives),
                "train_scope": args.train_scope,
                "n_rounds_total": int(len(sorted(offpolicy_df["order_sequence"].dropna().unique()))),
                "first_half_rounds": ",".join(str(x) for x in first_half),
                "round_last": int(last_order),
                "target_round_first_half": int(target_order),
                "target_selection_mean_js_to_last": float(best_js),
                "n_train_eval": int(len(train_eval_df)),
                "n_last_round": int(len(last_df)),
                "train_time_s": float(train_time),
                "total_time_s": float(total_time),
                "formula_normalization": "sum_cx",
            }
        ]
    )
    summary.to_csv(out_dir / "run_summary.csv", index=False)

    if (out_dir / "reuse_manifest.csv").exists():
        print(f"reused_artifacts_manifest={str((out_dir / 'reuse_manifest.csv').resolve())}")
    else:
        print(f"saved_ktm={str((out_dir / 'ktm_dataframe.csv').resolve())}")
        print(f"saved_gaussian_fit={str((out_dir / 'gaussian_fit_by_order.csv').resolve())}")
        print(f"saved_offpolicy_tuples={str((out_dir / 'offpolicy_tuples.csv').resolve())}")
        print(f"saved_train_eval_tuples={str((out_dir / 'train_eval_tuples.csv').resolve())}")
    print(f"saved_adaptive_behavior={str((out_dir / 'adaptive_behavior_by_round.csv').resolve())}")
    print(f"saved_training_history={str((out_dir / 'training_history.csv').resolve())}")
    print(f"saved_learned_policy_coeffs={str((out_dir / 'learned_policy_coefficients.csv').resolve())}")
    print(f"saved_estimators={str((out_dir / 'estimator_results_last_round.csv').resolve())}")
    print(f"saved_plot_policy_comparison={str((out_dir / 'policy_comparison_by_objective.png').resolve())}")
    print(f"saved_plot_behavior_evolution={str((out_dir / 'behavior_evolution_by_round.png').resolve())}")
    print(f"saved_summary={str((out_dir / 'run_summary.csv').resolve())}")


if __name__ == "__main__":
    main()
