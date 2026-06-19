from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offpolicy_ktm_pipeline.run_last_round_simple import (
    IRTOptimalGaussianPolicy,
    _apply_round_cap,
    _build_behavior_policy_from_row,
    _compute_context_ratio_hist,
    _evaluate_estimators,
    _prepare_ktm_dataframe,
    _select_target_round_closest_to_last,
    _train_last_round_policy,
)
from offpolicy_ktm_pipeline.src.estimators_and_training import compute_log_mixture_propensity
from offpolicy_ktm_pipeline.src.gaussian_regression import fit_gaussian_by_order
from offpolicy_ktm_pipeline.src.propensity import build_offpolicy_dataset
from offpolicy_ktm_pipeline.src.utils import get_default_device, set_global_seed


def _validate_split_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split fractions must sum to 1.0, got train+val+test={total:.6f}"
        )


def _assign_user_splits(
    users: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(users)
    n_users = int(len(perm))
    n_train = int(round(train_frac * n_users))
    n_val = int(round(val_frac * n_users))
    n_train = min(max(n_train, 1), max(n_users - 2, 1))
    n_val = min(max(n_val, 1), max(n_users - n_train - 1, 1))
    n_test = n_users - n_train - n_val
    if n_test <= 0:
        raise ValueError("User split leaves no users for test; adjust split fractions.")

    train_users = perm[:n_train]
    val_users = perm[n_train : n_train + n_val]
    test_users = perm[n_train + n_val :]
    return pd.DataFrame(
        {
            "user": np.concatenate([train_users, val_users, test_users]),
            "split": (
                ["train"] * len(train_users)
                + ["val"] * len(val_users)
                + ["test"] * len(test_users)
            ),
        }
    )


def _override_reward(offpolicy_df: pd.DataFrame, *, delta_min: float) -> pd.DataFrame:
    out = offpolicy_df.copy()
    out["reward"] = (
        out["correct"].to_numpy(dtype=float)
        * (out["difficulties"].to_numpy(dtype=float) - float(delta_min))
    )
    return out


def _compute_log_mixture_propensity_batched(
    *,
    eval_df: pd.DataFrame,
    fit_df_train: pd.DataFrame,
    fit_sigma_floor: float,
    mix_batch_rows: int,
) -> np.ndarray:
    n = int(len(eval_df))
    if n == 0:
        return np.empty(0, dtype=np.float32)

    batch_rows = max(int(mix_batch_rows), 1)
    chunks: list[np.ndarray] = []
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        cur = eval_df.iloc[start:end]
        chunks.append(
            compute_log_mixture_propensity(
                theta=cur["proficiency"].to_numpy(dtype=float),
                delta=cur["difficulties"].to_numpy(dtype=float),
                orders=cur["order_sequence"].to_numpy(dtype=int),
                fit_df=fit_df_train,
                mu_degree=1,
                sigma_degree=2,
                sigma_floor=float(fit_sigma_floor),
            ).astype(np.float32)
        )
    return np.concatenate(chunks, axis=0)


def _build_eval_frame(
    offpolicy_df: pd.DataFrame,
    *,
    fit_df_train: pd.DataFrame,
    target_theta: np.ndarray,
    last_order: int,
    eval_scope: str,
    context_bins: int,
    context_ratio_clip: float,
    fit_sigma_floor: float,
    mix_batch_rows: int,
    need_mix: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = offpolicy_df[offpolicy_df["order_sequence"].astype(int) <= int(last_order)].copy()
    if split_df.empty:
        raise ValueError("No rows left in split after aligning to common last round.")

    last_df = split_df[split_df["order_sequence"].astype(int) == int(last_order)].copy()
    if eval_scope == "current":
        eval_df = last_df.copy()
    else:
        eval_df = split_df.copy()
    if eval_df.empty:
        raise ValueError(f"No rows available for eval_scope={eval_scope}.")

    theta_cur = eval_df["proficiency"].to_numpy(dtype=float)
    eval_df["context_weight"] = _compute_context_ratio_hist(
        theta_ref=target_theta,
        theta_cur=theta_cur,
        theta_eval=theta_cur,
        n_bins=int(context_bins),
        ratio_clip=float(context_ratio_clip),
    )
    if need_mix:
        eval_df["log_propensity_mix"] = _compute_log_mixture_propensity_batched(
            eval_df=eval_df,
            fit_df_train=fit_df_train,
            fit_sigma_floor=float(fit_sigma_floor),
            mix_batch_rows=int(mix_batch_rows),
        )
    else:
        eval_df["log_propensity_mix"] = eval_df["log_propensity"].to_numpy(dtype=np.float32)
    return eval_df, last_df


def _evaluate_policy_on_frame(
    policy: object,
    *,
    eval_df: pd.DataFrame,
    need_mix: bool,
    delta_min: float,
    delta_max: float,
    dm_delta_grid: int,
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
            if need_mix
            else None
        ),
        delta_min=delta_min,
        delta_max=delta_max,
        dm_delta_grid=int(dm_delta_grid),
        max_weight=float(max_weight),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "User-split last-round offline policy learning: split users into train/val/test, "
            "fit behavior on train users, train policies on train users only, and evaluate "
            "frozen policies on held-out users with IPS/SNIPS/DR/MIS/DM."
        )
    )
    parser.add_argument("--input-csv", type=str, default="pix_mapping/full_pipeline_pix_processed/ktm_dataframe.csv")
    parser.add_argument("--input-is-ktm", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/pix_user_split_last_round")
    parser.add_argument(
        "--reuse-run-dir",
        type=str,
        default="",
        help="Reuse user_split_assignments.csv and gaussian_fit_train_by_order.csv from an earlier user-split run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)

    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--user-col", type=str, default="user_id")
    parser.add_argument("--item-col", type=str, default="challenge_id")
    parser.add_argument("--correct-col", type=str, default="answer_result")
    parser.add_argument("--skill-col", type=str, default="skill_id")
    parser.add_argument("--answer-order-col", type=str, default="answer_number")
    parser.add_argument("--derive-answer-number-per-user", action="store_true")

    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument(
        "--round-cap-strategy",
        type=str,
        default="tail",
        choices=["tail", "tail_count_at_n", "quantile"],
    )
    parser.add_argument("--truncate-at-round", type=int, default=0)
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
    )
    parser.add_argument("--train-scope", type=str, default="cumulative", choices=["cumulative", "current"])
    parser.add_argument("--eval-scope", type=str, default="same", choices=["same", "cumulative", "current"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--dm-delta-grid", type=int, default=121)
    parser.add_argument("--context-bins", type=int, default=120)
    parser.add_argument("--context-ratio-clip", type=float, default=20.0)
    parser.add_argument(
        "--report-splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which held-out splits to evaluate and save in estimator_results_by_split.csv.",
    )
    parser.add_argument("--mix-batch-rows", type=int, default=200000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-split-tuples", action="store_true")
    parser.add_argument(
        "--exclude-orders",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Order sequences to drop from training and evaluation before policy learning "
            "(e.g. rounds with degenerate behavior-policy fits). "
            "Rows with these order_sequence values are removed from every split's offpolicy "
            "dataset, and the corresponding rounds are also removed from the MIS mixture fit."
        ),
    )
    parser.add_argument(
        "--auto-exclude-rmse-threshold",
        type=float,
        default=0.0,
        help=(
            "Automatically exclude any order_sequence whose behavior-policy RMSE in "
            "gaussian_fit_train_by_order.csv exceeds this value. "
            "Applied on top of --exclude-orders. Set to 0 (default) to disable."
        ),
    )
    args = parser.parse_args()

    _validate_split_fractions(args.train_frac, args.val_frac, args.test_frac)
    set_global_seed(int(args.seed))
    device = get_default_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_scope = str(args.train_scope) if str(args.eval_scope) == "same" else str(args.eval_scope)
    need_mix = "mis" in list(args.objectives)
    report_splits = list(dict.fromkeys(args.report_splits))

    t0 = time.perf_counter()
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
    round_mapping.to_csv(out_dir / "round_bucket_mapping.csv", index=False)

    reuse_dir = Path(args.reuse_run_dir).resolve() if str(args.reuse_run_dir).strip() else None
    if reuse_dir is not None:
        split_path = reuse_dir / "user_split_assignments.csv"
        fit_path = reuse_dir / "gaussian_fit_train_by_order.csv"
        missing = [str(p) for p in [split_path, fit_path] if not p.exists()]
        if missing:
            raise ValueError(f"--reuse-run-dir missing required files: {missing}")
        user_split_df = pd.read_csv(split_path)
        fit_df_train = pd.read_csv(fit_path)
    else:
        user_split_df = _assign_user_splits(
            df_ktm["user"].dropna().unique(),
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            seed=int(args.split_seed),
        )
    user_to_split = dict(zip(user_split_df["user"], user_split_df["split"].astype(str)))
    df_ktm["split"] = df_ktm["user"].map(user_to_split)
    if df_ktm["split"].isna().any():
        raise ValueError("Some users were not assigned to a split.")
    user_split_df.to_csv(out_dir / "user_split_assignments.csv", index=False)

    split_frames = {
        split: df_ktm[df_ktm["split"] == split].drop(columns=["split"]).copy()
        for split in ["train", "val", "test"]
    }
    common_last_order = min(
        int(frame["order_sequence"].max()) for frame in split_frames.values() if not frame.empty
    )

    train_df = split_frames["train"]
    # Expand support by a tiny epsilon so that observations exactly at the
    # empirical boundary are never rejected as out-of-support after float32
    # round-trip inside GlobalGaussianPolicy.log_prob.
    delta_min = float(train_df["difficulties"].min()) - 1e-4
    delta_max = float(train_df["difficulties"].max()) + 1e-4

    if reuse_dir is None:
        fit_df_train = fit_gaussian_by_order(
            train_df,
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
        if fit_df_train.empty:
            raise ValueError("No behavior policy fit rows on train split.")
    fit_df_train.to_csv(out_dir / "gaussian_fit_train_by_order.csv", index=False)

    offpolicy_by_split: dict[str, pd.DataFrame] = {}
    for split, frame in split_frames.items():
        off_df = build_offpolicy_dataset(frame, fit_df_train, sigma_floor=float(args.fit_sigma_floor))
        off_df = _override_reward(off_df, delta_min=delta_min)
        offpolicy_by_split[split] = off_df
        if bool(args.save_split_tuples):
            off_df.to_csv(out_dir / f"offpolicy_tuples_{split}.csv", index=False)

    exclude_orders_set = set(int(o) for o in (args.exclude_orders or []))
    if float(args.auto_exclude_rmse_threshold) > 0.0:
        bad_rounds = set(
            int(r) for r in fit_df_train.loc[
                fit_df_train["rmse"] > float(args.auto_exclude_rmse_threshold), "order_sequence"
            ]
        )
        if bad_rounds:
            print(
                f"auto_exclude_rmse_threshold={args.auto_exclude_rmse_threshold}: "
                f"auto-excluding rounds {sorted(bad_rounds)} (RMSE > threshold)"
            )
        exclude_orders_set = exclude_orders_set | bad_rounds

    if exclude_orders_set:
        print(f"Excluding order_sequences {sorted(exclude_orders_set)} from all splits and MIS fit.")
        for split in list(offpolicy_by_split.keys()):
            n_before = len(offpolicy_by_split[split])
            offpolicy_by_split[split] = offpolicy_by_split[split][
                ~offpolicy_by_split[split]["order_sequence"].astype(int).isin(exclude_orders_set)
            ].copy()
            print(f"  split={split}: {n_before} -> {len(offpolicy_by_split[split])} rows")
    fit_df_for_eval = (
        fit_df_train[~fit_df_train["order_sequence"].astype(int).isin(exclude_orders_set)].copy()
        if exclude_orders_set
        else fit_df_train
    )

    train_offpolicy_ref = offpolicy_by_split["train"]
    train_offpolicy_ref = train_offpolicy_ref[
        train_offpolicy_ref["order_sequence"].astype(int) <= int(common_last_order)
    ].copy()
    target_order, _, first_half, best_js = _select_target_round_closest_to_last(
        train_offpolicy_ref,
        order_col="order_sequence",
        theta_col="proficiency",
        n_bins=int(args.context_bins),
    )
    target_theta = train_offpolicy_ref[
        train_offpolicy_ref["order_sequence"].astype(int) == int(target_order)
    ]["proficiency"].to_numpy(dtype=float)
    if target_theta.size == 0:
        raise ValueError(f"No train rows for target round {target_order}.")

    split_rows: list[dict[str, float | int | str]] = []
    print("building eval frame for split=train ...")
    train_eval_df, train_last_df = _build_eval_frame(
        offpolicy_by_split["train"],
        fit_df_train=fit_df_for_eval,
        target_theta=target_theta,
        last_order=common_last_order,
        eval_scope=eval_scope,
        context_bins=int(args.context_bins),
        context_ratio_clip=float(args.context_ratio_clip),
        fit_sigma_floor=float(args.fit_sigma_floor),
        mix_batch_rows=int(args.mix_batch_rows),
        need_mix=need_mix,
    )
    if bool(args.save_split_tuples):
        train_eval_df.to_csv(out_dir / "train_eval_tuples_train.csv", index=False)

    eval_frames: dict[str, pd.DataFrame] = {}
    last_rows_by_split: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        if split == "train":
            eval_df = train_eval_df
            last_df = train_last_df
        elif split in report_splits:
            print(f"building eval frame for split={split} ...")
            eval_df, last_df = _build_eval_frame(
                offpolicy_by_split[split],
                fit_df_train=fit_df_for_eval,
                target_theta=target_theta,
                last_order=common_last_order,
                eval_scope=eval_scope,
                context_bins=int(args.context_bins),
                context_ratio_clip=float(args.context_ratio_clip),
                fit_sigma_floor=float(args.fit_sigma_floor),
                mix_batch_rows=int(args.mix_batch_rows),
                need_mix=need_mix,
            )
            if bool(args.save_split_tuples):
                eval_df.to_csv(out_dir / f"train_eval_tuples_{split}.csv", index=False)
        else:
            eval_df = pd.DataFrame()
            last_df = pd.DataFrame()

        if split in report_splits:
            eval_frames[split] = eval_df
            last_rows_by_split[split] = last_df

        split_rows.append(
            {
                "split": split,
                "n_users": int(split_frames[split]["user"].nunique()),
                "n_rows": int(len(split_frames[split])),
                "n_eval_rows": int(len(eval_df)),
                "n_last_round": int(len(last_df)),
                "max_order_sequence": int(split_frames[split]["order_sequence"].max()),
                "common_last_order": int(common_last_order),
                "reported": bool(split in report_splits),
            }
        )
    split_summary_df = pd.DataFrame(split_rows)
    split_summary_df.to_csv(out_dir / "split_summary.csv", index=False)

    theta_t = torch.from_numpy(train_eval_df["proficiency"].to_numpy(np.float32))
    delta_t = torch.from_numpy(train_eval_df["difficulties"].to_numpy(np.float32))
    reward_t = torch.from_numpy(train_eval_df["reward"].to_numpy(np.float32))
    cw_t = torch.from_numpy(train_eval_df["context_weight"].to_numpy(np.float32))
    lb_t = torch.from_numpy(train_eval_df["log_propensity"].to_numpy(np.float32))
    lm_t = torch.from_numpy(train_eval_df["log_propensity_mix"].to_numpy(np.float32))

    fit_last = fit_df_train[fit_df_train["order_sequence"].astype(int) == int(common_last_order)]
    init_row = fit_last.iloc[0] if not fit_last.empty else fit_df_train.sort_values("order_sequence").iloc[-1]

    learned_policies: dict[str, object] = {}
    history_rows: list[pd.DataFrame] = []
    coef_rows: list[dict[str, float | str]] = []
    train_start = time.perf_counter()
    for obj in args.objectives:
        print(f"training objective={obj} on train split...")
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
        hist_df = hist_df.copy()
        hist_df["split_trained_on"] = "train"
        history_rows.append(hist_df)
        learned_policies[obj] = policy.eval()
        coef_rows.append(
            {
                "objective": obj,
                "beta_mu_0": float(policy.beta_mu[0].item()),
                "beta_mu_1": float(policy.beta_mu[1].item()),
                "beta_sigma_0": float(policy.beta_sigma[0].item()),
                "beta_sigma_1": float(policy.beta_sigma[1].item()),
                "beta_sigma_2": float(policy.beta_sigma[2].item()),
            }
        )
    train_time = time.perf_counter() - train_start

    pd.concat(history_rows, ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(out_dir / "learned_policy_coefficients.csv", index=False)

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

    named_policies: list[tuple[str, str, object]] = [
        ("behavior", "n/a", behavior_policy),
        ("optimal_irt_gaussian", "n/a", optimal_policy),
    ] + [(f"learned_{obj}", obj, pol) for obj, pol in learned_policies.items()]

    eval_rows: list[dict[str, float | int | str]] = []
    for split, eval_df in eval_frames.items():
        for name, trained_obj, policy in named_policies:
            m = _evaluate_policy_on_frame(
                policy,
                eval_df=eval_df,
                need_mix=need_mix,
                delta_min=delta_min,
                delta_max=delta_max,
                dm_delta_grid=int(args.dm_delta_grid),
                max_weight=float(args.max_weight),
                device=device,
            )
            eval_rows.append(
                {
                    "split": split,
                    "policy": name,
                    "trained_objective": trained_obj,
                    "train_scope": args.train_scope,
                    "eval_scope": eval_scope,
                    "target_round_train_first_half": int(target_order),
                    "common_last_round": int(common_last_order),
                    "target_selection_mean_js_to_last_train": float(best_js),
                    "n_eval": int(len(eval_df)),
                    "n_last_round_split": int(len(last_rows_by_split[split])),
                    "ips": float(m["ips"]),
                    "snips": float(m["snips"]),
                    "dr": float(m["dr"]),
                    "mis": float(m["mis"]),
                    "dm": float(m["dm"]),
                    "ess_logged": float(m["ess_logged"]),
                    "ess_mix": float(m["ess_mix"]),
                }
            )
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_dir / "estimator_results_by_split.csv", index=False)

    total_time = time.perf_counter() - t0
    pd.DataFrame(
        [
            {
                "input_csv": str(args.input_csv),
                "input_is_ktm": bool(args.input_is_ktm),
                "train_frac": float(args.train_frac),
                "val_frac": float(args.val_frac),
                "test_frac": float(args.test_frac),
                "max_rounds": int(args.max_rounds),
                "truncate_at_round": int(args.truncate_at_round),
                "round_cap_strategy": str(args.round_cap_strategy),
                "train_scope": str(args.train_scope),
                "eval_scope": str(eval_scope),
                "need_mix": bool(need_mix),
                "report_splits": ",".join(report_splits),
                "common_last_round": int(common_last_order),
                "target_round_train_first_half": int(target_order),
                "target_selection_mean_js_to_last_train": float(best_js),
                "n_train_eval": int(len(train_eval_df)),
                "train_time_s": float(train_time),
                "total_time_s": float(total_time),
                "reused_from": str(reuse_dir) if reuse_dir is not None else "",
                "excluded_orders": ",".join(str(o) for o in sorted(exclude_orders_set)) if exclude_orders_set else "",
                "auto_exclude_rmse_threshold": float(args.auto_exclude_rmse_threshold),
            }
        ]
    ).to_csv(out_dir / "run_summary.csv", index=False)

    print(f"saved_split_assignments={str((out_dir / 'user_split_assignments.csv').resolve())}")
    print(f"saved_split_summary={str((out_dir / 'split_summary.csv').resolve())}")
    print(f"saved_behavior_fit={str((out_dir / 'gaussian_fit_train_by_order.csv').resolve())}")
    print(f"saved_training_history={str((out_dir / 'training_history.csv').resolve())}")
    print(f"saved_policy_coeffs={str((out_dir / 'learned_policy_coefficients.csv').resolve())}")
    print(f"saved_estimators={str((out_dir / 'estimator_results_by_split.csv').resolve())}")
    print(f"saved_summary={str((out_dir / 'run_summary.csv').resolve())}")


if __name__ == "__main__":
    main()
