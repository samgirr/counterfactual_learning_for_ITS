"""
Clean OPE experiment runner.

Setup:
- 80/20 user-level train/test split (no val)
- Truncated Gaussian behavior policy fitted on train split only
- Target context = middle round (by index among valid rounds)
- Learned policies: IPS, SNIPS, DR, MIS (truncated Gaussian, same parameterisation)
- Bad rounds (RMSE > rmse_threshold) excluded from BOTH training and test estimators
- Outputs: two HTML result tables + CSV artefacts in results/<dataset>/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offpolicy_ktm_pipeline.run_last_round_simple import (
    IRTOptimalGaussianPolicy,
    _apply_round_cap,
    _build_behavior_policy_from_row,
    _compute_context_ratio_hist,
    _evaluate_estimators,
    _train_last_round_policy,
)
from offpolicy_ktm_pipeline.src.estimators_and_training import compute_log_mixture_propensity
from offpolicy_ktm_pipeline.src.gaussian_regression import fit_gaussian_by_order
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.utils import get_default_device, set_global_seed


# ── helpers ──────────────────────────────────────────────────────────────────

def assign_user_splits(users: np.ndarray, *, train_frac: float, seed: int) -> pd.DataFrame:
    """80/20 user-level split — no validation set."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(users)
    n_train = max(1, int(round(train_frac * len(perm))))
    n_train = min(n_train, len(perm) - 1)
    labels = ["train"] * n_train + ["test"] * (len(perm) - n_train)
    return pd.DataFrame({"user": perm, "split": labels})


def select_middle_round(valid_orders: list[int]) -> int:
    """Pick the round closest to the midpoint of valid rounds."""
    s = sorted(valid_orders)
    return s[len(s) // 2]


def compute_log_mixture_batched(
    eval_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    sigma_floor: float,
    batch: int = 200_000,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(eval_df), batch):
        cur = eval_df.iloc[start : start + batch]
        chunks.append(
            compute_log_mixture_propensity(
                theta=cur["proficiency"].to_numpy(float),
                delta=cur["difficulties"].to_numpy(float),
                orders=cur["order_sequence"].to_numpy(int),
                fit_df=fit_df,
                mu_degree=1,
                sigma_degree=2,
                sigma_floor=sigma_floor,
            ).astype(np.float32)
        )
    return np.concatenate(chunks)


def build_eval_frame(
    offpolicy_df: pd.DataFrame,
    *,
    fit_df: pd.DataFrame,
    target_theta: np.ndarray,
    last_order: int,
    context_bins: int,
    context_ratio_clip: float,
    sigma_floor: float,
    need_mix: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cumulative eval frame with context weights and (optionally) MIS log-propensity."""
    eval_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= last_order].copy()
    last_df = eval_df[eval_df["order_sequence"].astype(int) == last_order].copy()

    theta_cur = eval_df["proficiency"].to_numpy(float)
    cw = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_cur,
        theta_eval=theta_cur,
        n_bins=context_bins,
        ratio_clip=context_ratio_clip,
    )
    eval_df["context_weight"] = cw

    if need_mix:
        eval_df["log_propensity_mix"] = compute_log_mixture_batched(eval_df, fit_df, sigma_floor)
    else:
        eval_df["log_propensity_mix"] = eval_df["log_propensity"].to_numpy(np.float32)

    return eval_df, last_df


def eval_policy(
    policy,
    eval_df: pd.DataFrame,
    *,
    need_mix: bool,
    delta_min: float,
    delta_max: float,
    max_weight: float,
    device: torch.device,
) -> dict[str, float]:
    return _evaluate_estimators(
        policy,
        theta=torch.from_numpy(eval_df["proficiency"].to_numpy(np.float32)),
        delta=torch.from_numpy(eval_df["difficulties"].to_numpy(np.float32)),
        reward=torch.from_numpy(eval_df["reward"].to_numpy(np.float32)),
        context_w=torch.from_numpy(eval_df["context_weight"].to_numpy(np.float32)),
        log_prop_logged=torch.from_numpy(eval_df["log_propensity"].to_numpy(np.float32)),
        log_prop_mix=(
            torch.from_numpy(eval_df["log_propensity_mix"].to_numpy(np.float32))
            if need_mix else None
        ),
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=121,
        max_weight=max_weight,
        device=device,
    )


def behavior_value_at_round(
    offpolicy_df: pd.DataFrame,
    round_order: int,
    target_theta: np.ndarray,
    *,
    context_bins: int,
    context_ratio_clip: float,
) -> tuple[float, float, int]:
    """Context-reweighted mean reward at a specific round."""
    sub = offpolicy_df[offpolicy_df["order_sequence"].astype(int) == round_order].copy()
    if sub.empty:
        return float("nan"), float("nan"), 0
    theta_cur = sub["proficiency"].to_numpy(float)
    cw = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_cur,
        theta_eval=theta_cur,
        n_bins=context_bins,
        ratio_clip=context_ratio_clip,
    )
    r = sub["reward"].to_numpy(float)
    raw_mean = float(r.mean())
    weighted = float(np.sum(cw * r) / np.sum(cw))
    return raw_mean, weighted, len(sub)


# ── HTML report ──────────────────────────────────────────────────────────────

_CSS = """
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:980px;margin:36px auto;padding:0 20px;color:#222;line-height:1.55}
h1{border-bottom:2px solid #333;padding-bottom:6px}
h2{border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:40px;color:#2c3e50}
table{border-collapse:collapse;width:100%;margin:10px 0 20px;font-size:.91em}
th{background:#2c3e50;color:#fff;padding:7px 11px;text-align:right}
th:first-child{text-align:left}
td{padding:6px 11px;border-bottom:1px solid #e8e8e8;text-align:right}
td:first-child{text-align:left}
tr.beh td{background:#eaf3fb;color:#555;font-style:italic}
tr.best td{font-weight:bold}
tr.oracle td{background:#fafafa;color:#999;font-style:italic}
tr.ground td{background:#fdf9e3;font-weight:bold}
.note{font-size:.84em;color:#666;margin:2px 0 14px}
.warn{background:#fff3cd;border-left:4px solid #f0ad4e;padding:8px 12px;margin:10px 0;font-size:.88em;border-radius:0 4px 4px 0}
.good{background:#e9f7ef;border-left:4px solid #27ae60;padding:8px 12px;margin:10px 0;font-size:.88em;border-radius:0 4px 4px 0}
.meta{color:#888;font-size:.83em}
</style>
"""


def fmt(v: float, decimals: int = 3) -> str:
    if np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def make_html_report(
    dataset: str,
    last_order: int,
    first_order: int,
    target_order: int,
    excluded_orders: list[int],
    n_train: int,
    n_test: int,
    n_test_last: int,
    # table 1 data
    policy_rows: list[dict],
    # table 2 data
    beh_first_raw: float,
    beh_first_wtd: float,
    n_first: int,
    beh_last_raw: float,
    beh_last_wtd: float,
    n_last: int,
    beh_cumul: dict[str, float],
    # comments
    comments: list[str],
) -> str:
    excl_str = ", ".join(str(o) for o in sorted(excluded_orders)) if excluded_orders else "none"

    # ── table 1 ──────────────────────────────────────────────────────────────
    t1_rows = ""
    best_snips = max(r["snips"] for r in policy_rows if r["policy"] != "optimal_irt")
    for r in policy_rows:
        cls = ""
        if r["policy"] == "behavior":
            cls = "beh"
        elif r["policy"] == "optimal_irt":
            cls = "oracle"
        elif abs(r["snips"] - best_snips) < 1e-6:
            cls = "best"
        t1_rows += (
            f'<tr class="{cls}">'
            f'<td>{r["policy"]}</td>'
            f'<td>{r.get("objective","—")}</td>'
            f'<td>{fmt(r["ips"])}</td>'
            f'<td>{fmt(r["snips"])}</td>'
            f'<td>{fmt(r["dr"])}</td>'
            f'<td>{fmt(r["mis"])}</td>'
            f'<td>{fmt(r["dm"])}</td>'
            f'<td>{int(r["ess"])}</td>'
            f"</tr>\n"
        )

    # ── table 2 ──────────────────────────────────────────────────────────────
    def delta_str(val: float, ref: float) -> str:
        d = val - ref
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.3f})"

    t2_rows = (
        f'<tr>'
        f'<td>Round {first_order} — first</td>'
        f'<td>{n_first}</td>'
        f'<td>{fmt(beh_first_raw)}</td>'
        f'<td>{fmt(beh_first_wtd)}</td>'
        f'<td>{fmt(beh_cumul["ips"])} {delta_str(beh_cumul["ips"], beh_first_wtd)}</td>'
        f'<td>{fmt(beh_cumul["snips"])} {delta_str(beh_cumul["snips"], beh_first_wtd)}</td>'
        f'<td>{fmt(beh_cumul["dr"])} {delta_str(beh_cumul["dr"], beh_first_wtd)}</td>'
        f"</tr>\n"
        f'<tr class="ground">'
        f'<td>Round {last_order} — last ★</td>'
        f'<td>{n_last}</td>'
        f'<td>{fmt(beh_last_raw)}</td>'
        f'<td>{fmt(beh_last_wtd)}</td>'
        f'<td>{fmt(beh_cumul["ips"])} {delta_str(beh_cumul["ips"], beh_last_wtd)}</td>'
        f'<td>{fmt(beh_cumul["snips"])} {delta_str(beh_cumul["snips"], beh_last_wtd)}</td>'
        f'<td>{fmt(beh_cumul["dr"])} {delta_str(beh_cumul["dr"], beh_last_wtd)}</td>'
        f"</tr>\n"
    )

    comment_html = ""
    for c in comments:
        tag = "warn" if c.startswith("⚠") else "good"
        comment_html += f'<div class="{tag}">{c}</div>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>OPE — {dataset}</title>{_CSS}</head>
<body>
<h1>OPE Results — {dataset}</h1>
<p class="meta">
  Train/test split: 80/20 (user-level) &nbsp;|&nbsp;
  Rounds: {first_order}–{last_order} &nbsp;|&nbsp;
  Target context round: {target_order} (middle) &nbsp;|&nbsp;
  Excluded rounds: {excl_str}<br>
  Train users: {n_train:,} &nbsp;|&nbsp; Test users: {n_test:,} &nbsp;|&nbsp;
  Test samples at last round: {n_test_last}
</p>

{comment_html}

<h2>Table 1 — Learned policy values (test set, cumulative, context-reweighted)</h2>
<p class="note">
  All estimates use the full test trajectory (cumulative), reweighted by
  q<sub>target</sub>(θ) / q<sub>all</sub>(θ) where q<sub>target</sub> is the
  training distribution at round {target_order}.
  Bold = best learned policy by SNIPS.
</p>
<table>
  <tr>
    <th>Policy</th><th>Objective</th>
    <th>IPS</th><th>SNIPS</th><th>DR</th><th>MIS</th><th>DM</th><th>ESS</th>
  </tr>
  {t1_rows}
</table>

<h2>Table 2 — Behavior policy: first vs. last round vs. cumulative estimator</h2>
<p class="note">
  The yellow row (last round ★) is the empirical context-reweighted reward at the
  last round — the ground-truth we aim to beat.
  Parentheses show Δ = cumulative estimator − per-round value.
  A large positive Δ means the cumulative estimator over-estimates relative to the last round.
</p>
<table>
  <tr>
    <th>Round</th><th>n (test)</th>
    <th>Raw mean reward</th><th>Ctx-reweighted reward</th>
    <th>Cumul. IPS (Δ)</th><th>Cumul. SNIPS (Δ)</th><th>Cumul. DR (Δ)</th>
  </tr>
  {t2_rows}
</table>

<hr>
<p class="meta">geometric_flow / experiments · {time.strftime('%Y-%m-%d')}</p>
</body>
</html>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-is-ktm", action="store_true")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--truncate-at-round", type=int, default=0,
                        help="Keep only rows with answer_number <= this value.")
    parser.add_argument("--max-rounds", type=int, default=50,
                        help="Max rounds per student (tail cap).")
    parser.add_argument("--rmse-threshold", type=float, default=5.0,
                        help="Auto-exclude rounds with behavior RMSE above this.")
    parser.add_argument("--objectives", nargs="+",
                        default=["ips", "snips", "dr", "mis"],
                        choices=["ips", "snips", "dr", "mis"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--sample-users", type=int, default=0,
                        help="Randomly sample this many users before splitting (0 = use all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--context-bins", type=int, default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = get_default_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need_mix = "mis" in args.objectives
    t0 = time.perf_counter()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"{'='*60}")

    if args.input_is_ktm:
        df_ktm = pd.read_csv(args.input_csv)
    else:
        from offpolicy_ktm_pipeline.run_last_round_simple import _prepare_ktm_dataframe
        df_ktm = _prepare_ktm_dataframe(args, out_dir)

    if args.sample_users > 0:
        all_users = df_ktm["user"].dropna().unique()
        if args.sample_users < len(all_users):
            rng = np.random.default_rng(args.seed)
            sampled = rng.choice(all_users, size=args.sample_users, replace=False)
            df_ktm = df_ktm[df_ktm["user"].isin(sampled)].copy()
            print(f"sample_users={args.sample_users}: kept {df_ktm['user'].nunique():,} users, {len(df_ktm):,} rows")

    if args.truncate_at_round > 0:
        n0 = len(df_ktm)
        df_ktm = df_ktm[df_ktm["answer_number"] <= args.truncate_at_round].copy()
        print(f"truncate_at_round={args.truncate_at_round}: {n0:,} → {len(df_ktm):,} rows")

    df_ktm, round_mapping = _apply_round_cap(df_ktm, max_rounds=args.max_rounds, strategy="tail")
    round_mapping.to_csv(out_dir / "round_bucket_mapping.csv", index=False)

    # ── 2. Train/test split ───────────────────────────────────────────────────
    users = df_ktm["user"].dropna().unique()
    user_split_df = assign_user_splits(users, train_frac=args.train_frac, seed=args.seed)
    user_to_split = dict(zip(user_split_df["user"], user_split_df["split"]))
    df_ktm["split"] = df_ktm["user"].map(user_to_split)
    user_split_df.to_csv(out_dir / "user_split_assignments.csv", index=False)

    train_df = df_ktm[df_ktm["split"] == "train"].drop(columns=["split"]).copy()
    test_df  = df_ktm[df_ktm["split"] == "test"].drop(columns=["split"]).copy()
    print(f"Train: {train_df['user'].nunique():,} users, {len(train_df):,} rows")
    print(f"Test:  {test_df['user'].nunique():,} users, {len(test_df):,} rows")

    # support bounds (from train, slightly expanded)
    delta_min = float(train_df["difficulties"].min()) - 1e-4
    delta_max = float(train_df["difficulties"].max()) + 1e-4

    # ── 3. Fit behavior policy (train only) ───────────────────────────────────
    print("Fitting behavior policy on train split...")
    fit_df = fit_gaussian_by_order(
        train_df,
        mu_degree=1,
        sigma_degree=2,
        sigma_floor=1e-4,
        min_obs_per_order=args.min_obs_per_order,
        distribution="truncated_gaussian",
        delta_min=delta_min,
        delta_max=delta_max,
    )
    fit_df.to_csv(out_dir / "gaussian_fit_train_by_order.csv", index=False)

    # ── 4. Auto-exclude bad rounds ────────────────────────────────────────────
    excluded_orders: set[int] = set()
    if args.rmse_threshold > 0:
        bad = set(fit_df.loc[fit_df["rmse"] > args.rmse_threshold, "order_sequence"].astype(int))
        if bad:
            print(f"Auto-excluding rounds {sorted(bad)} (RMSE > {args.rmse_threshold})")
        excluded_orders |= bad

    fit_df_clean = fit_df[~fit_df["order_sequence"].astype(int).isin(excluded_orders)].copy()

    # ── 5. Build off-policy datasets ──────────────────────────────────────────
    def build_and_filter(frame: pd.DataFrame) -> pd.DataFrame:
        off = build_offpolicy_dataset(frame, fit_df, sigma_floor=1e-4)
        off["reward"] = (
            off["correct"].to_numpy(float) * (off["difficulties"].to_numpy(float) - delta_min)
        )
        if excluded_orders:
            off = off[~off["order_sequence"].astype(int).isin(excluded_orders)].copy()
        return off

    train_off = build_and_filter(train_df)
    test_off  = build_and_filter(test_df)

    # ── 6. Target context round = middle valid round ───────────────────────────
    common_last = int(min(train_df["order_sequence"].max(), test_df["order_sequence"].max()))
    valid_orders = sorted(fit_df_clean["order_sequence"].astype(int).unique())
    valid_orders = [o for o in valid_orders if o <= common_last]
    target_order = select_middle_round(valid_orders)
    first_order  = valid_orders[0]
    print(f"Rounds: first={first_order}, target(middle)={target_order}, last={common_last}")

    target_theta = train_off[
        train_off["order_sequence"].astype(int) == target_order
    ]["proficiency"].to_numpy(float)

    # ── 7. Build eval frames ──────────────────────────────────────────────────
    print("Building train eval frame...")
    train_eval, _ = build_eval_frame(
        train_off, fit_df=fit_df_clean, target_theta=target_theta,
        last_order=common_last, context_bins=args.context_bins,
        context_ratio_clip=args.context_ratio_clip, sigma_floor=1e-4,
        need_mix=need_mix,
    )
    print("Building test eval frame...")
    test_eval, test_last_df = build_eval_frame(
        test_off, fit_df=fit_df_clean, target_theta=target_theta,
        last_order=common_last, context_bins=args.context_bins,
        context_ratio_clip=args.context_ratio_clip, sigma_floor=1e-4,
        need_mix=need_mix,
    )
    print(f"Test eval rows: {len(test_eval):,}  |  last-round rows: {len(test_last_df)}")

    # ── 8. Train policies ─────────────────────────────────────────────────────
    fit_last = fit_df_clean[fit_df_clean["order_sequence"].astype(int) == common_last]
    init_row = (
        fit_last.iloc[0] if not fit_last.empty
        else fit_df_clean.sort_values("order_sequence").iloc[-1]
    )

    theta_t  = torch.from_numpy(train_eval["proficiency"].to_numpy(np.float32))
    delta_t  = torch.from_numpy(train_eval["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(train_eval["reward"].to_numpy(np.float32))
    cw_t     = torch.from_numpy(train_eval["context_weight"].to_numpy(np.float32))
    lb_t     = torch.from_numpy(train_eval["log_propensity"].to_numpy(np.float32))
    lm_t     = torch.from_numpy(train_eval["log_propensity_mix"].to_numpy(np.float32))

    learned: dict[str, object] = {}
    history_rows: list[pd.DataFrame] = []
    coef_rows: list[dict] = []
    for obj in args.objectives:
        print(f"Training {obj}...")
        policy, hist = _train_last_round_policy(
            objective=obj,
            theta=theta_t, delta=delta_t, reward=reward_t,
            context_w=cw_t, log_prop_logged=lb_t, log_prop_mix=lm_t,
            delta_min=delta_min, delta_max=delta_max,
            policy_sigma_floor=0.2, init_row=init_row,
            dm_delta_grid=121, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, l2_coef=1e-6,
            max_weight=args.max_weight, device=device,
        )
        hist["split_trained_on"] = "train"
        history_rows.append(hist)
        learned[obj] = policy.eval()
        coef_rows.append({
            "objective": obj,
            "beta_mu_0": float(policy.beta_mu[0].item()),
            "beta_mu_1": float(policy.beta_mu[1].item()),
            "beta_sigma_0": float(policy.beta_sigma[0].item()),
            "beta_sigma_1": float(policy.beta_sigma[1].item()),
            "beta_sigma_2": float(policy.beta_sigma[2].item()),
        })

    pd.concat(history_rows, ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out_dir / "learned_policy_coefficients.csv", index=False)

    # ── 9. Reference policies ─────────────────────────────────────────────────
    behavior_policy = _build_behavior_policy_from_row(
        init_row, delta_min=delta_min, delta_max=delta_max, sigma_floor=1e-4, device=device,
    )
    optimal_policy = IRTOptimalGaussianPolicy(
        delta_min=delta_min, delta_max=delta_max, sigma=0.2,
    )

    named: list[tuple[str, str, object]] = [
        ("behavior",    "—",  behavior_policy),
        ("optimal_irt", "—",  optimal_policy),
    ] + [(f"learned_{obj}", obj, p) for obj, p in learned.items()]

    # ── 10. Evaluate on TEST set ──────────────────────────────────────────────
    eval_rows: list[dict] = []
    for name, obj, policy in named:
        m = eval_policy(
            policy, test_eval, need_mix=need_mix,
            delta_min=delta_min, delta_max=delta_max,
            max_weight=args.max_weight, device=device,
        )
        eval_rows.append({
            "policy": name, "objective": obj,
            "ips": m["ips"], "snips": m["snips"],
            "dr": m["dr"], "mis": m["mis"], "dm": m["dm"],
            "ess": int(m["ess_logged"]),
        })
        print(f"  {name:25s}  IPS={m['ips']:.3f}  SNIPS={m['snips']:.3f}  DR={m['dr']:.3f}  MIS={m['mis']:.3f}  DM={m['dm']:.3f}")

    pd.DataFrame(eval_rows).to_csv(out_dir / "estimator_results_test.csv", index=False)

    # ── 11. Per-round behavior values (test set) ───────────────────────────────
    beh_first_raw, beh_first_wtd, n_first = behavior_value_at_round(
        test_off, first_order, target_theta,
        context_bins=args.context_bins, context_ratio_clip=args.context_ratio_clip,
    )
    beh_last_raw, beh_last_wtd, n_last = behavior_value_at_round(
        test_off, common_last, target_theta,
        context_bins=args.context_bins, context_ratio_clip=args.context_ratio_clip,
    )
    beh_cumul = next(r for r in eval_rows if r["policy"] == "behavior")

    print(f"\nBehavior @ round {first_order} (first): raw={beh_first_raw:.3f}  ctx-weighted={beh_first_wtd:.3f}  n={n_first}")
    print(f"Behavior @ round {common_last} (last):  raw={beh_last_raw:.3f}  ctx-weighted={beh_last_wtd:.3f}  n={n_last}")
    print(f"Behavior cumulative: IPS={beh_cumul['ips']:.3f}  SNIPS={beh_cumul['snips']:.3f}  DR={beh_cumul['dr']:.3f}")

    # ── 12. Auto-comments ─────────────────────────────────────────────────────
    comments: list[str] = []
    if excluded_orders:
        comments.append(
            f"⚠ Rounds {sorted(excluded_orders)} excluded (behavior RMSE > {args.rmse_threshold}). "
            "These rounds had degenerate fits (likely bimodal difficulty distribution with a spike at δ_min). "
            "They are removed from both training and test estimators."
        )
    if n_last < 100:
        comments.append(
            f"⚠ Only {n_last} test samples at last round (round {common_last}). "
            "The per-round behavior value is noisy. The cumulative estimator is more reliable "
            "but aggregates across all rounds, causing an upward bias relative to last-round value."
        )
    # check if cumulative IPS overestimates last-round value by > 20%
    if not np.isnan(beh_last_wtd) and beh_last_wtd > 0:
        overest = (beh_cumul["ips"] - beh_last_wtd) / beh_last_wtd
        if abs(overest) > 0.15:
            direction = "over" if overest > 0 else "under"
            comments.append(
                f"⚠ Cumulative IPS {direction}estimates the last-round behavior value by "
                f"{abs(overest)*100:.0f}% "
                f"(IPS={beh_cumul['ips']:.3f} vs. last-round={beh_last_wtd:.3f}). "
                "This is expected: the cumulative estimator averages rewards across all rounds, "
                "while the last-round value reflects only students who persisted to the final round. "
                "Rewards grow with round number (student attrition selects for motivated/capable students), "
                "so early rounds pull the cumulative estimate up."
            )
    # check if any learned policy beats behavior
    best_learned_snips = max(
        r["snips"] for r in eval_rows if r["policy"].startswith("learned_")
    )
    if best_learned_snips > beh_cumul["snips"]:
        pct = (best_learned_snips - beh_cumul["snips"]) / max(beh_cumul["snips"], 1e-6) * 100
        comments.append(
            f"✓ Best learned policy improves SNIPS over behavior by {pct:.1f}% "
            f"({best_learned_snips:.3f} vs. {beh_cumul['snips']:.3f})."
        )
    else:
        comments.append(
            "⚠ No learned policy improves SNIPS over behavior. "
            "Consider more epochs, different learning rate, or larger dataset."
        )

    # ── 13. Save HTML ─────────────────────────────────────────────────────────
    html = make_html_report(
        dataset=args.dataset_name,
        last_order=common_last, first_order=first_order,
        target_order=target_order, excluded_orders=sorted(excluded_orders),
        n_train=int(train_df["user"].nunique()),
        n_test=int(test_df["user"].nunique()),
        n_test_last=n_last,
        policy_rows=eval_rows,
        beh_first_raw=beh_first_raw, beh_first_wtd=beh_first_wtd, n_first=n_first,
        beh_last_raw=beh_last_raw,  beh_last_wtd=beh_last_wtd,   n_last=n_last,
        beh_cumul=beh_cumul, comments=comments,
    )
    html_path = out_dir / "report.html"
    html_path.write_text(html)
    print(f"\nReport saved: {html_path}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
