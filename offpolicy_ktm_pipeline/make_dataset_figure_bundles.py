from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_irt_values(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    x = df["round"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(x, df["behavior_irt_value"], lw=2.0, color="#ff7f0e", label="behavior IRT")
    ax.plot(x, df["learned_irt_value"], lw=2.2, color="#1f77b4", label="learned IRT")
    ax.plot(x, df["optimal_irt_value"], lw=2.2, color="#2ca02c", linestyle="--", label="optimal IRT")
    if "behavior_irt_value_q" in df.columns and df["behavior_irt_value_q"].notna().any():
        ax.plot(x, df["behavior_irt_value_q"], lw=1.8, color="#ff7f0e", linestyle=":", label="behavior IRT (q)")
    if "learned_irt_value_q" in df.columns and df["learned_irt_value_q"].notna().any():
        ax.plot(x, df["learned_irt_value_q"], lw=1.8, color="#1f77b4", linestyle=":", label="learned IRT (q)")
    if "optimal_irt_value_q" in df.columns and df["optimal_irt_value_q"].notna().any():
        ax.plot(x, df["optimal_irt_value_q"], lw=1.8, color="#2ca02c", linestyle=":", label="optimal IRT (q)")
    ax.set_xlabel("round")
    ax.set_ylabel("value")
    ax.set_title(f"{title_prefix} IRT Values by Round")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_ope(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    x = df["round"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(x, df["ips"], lw=2.0, color="#1f77b4", label="IPS")
    ax.plot(x, df["snips"], lw=2.0, color="#ff7f0e", label="SNIPS")
    ax.plot(x, df["cips"], lw=2.0, color="#2ca02c", label="CIPS")
    ax.plot(x, df["csnips"], lw=2.0, color="#d62728", label="CSNIPS")
    if "ips_q_train" in df.columns and df["ips_q_train"].notna().any():
        ax.plot(x, df["ips_q_train"], lw=1.5, color="#1f77b4", linestyle=":", label="IPS q-train")
    if "snips_q_train" in df.columns and df["snips_q_train"].notna().any():
        ax.plot(x, df["snips_q_train"], lw=1.5, color="#ff7f0e", linestyle=":", label="SNIPS q-train")
    if "cips_q_train" in df.columns and df["cips_q_train"].notna().any():
        ax.plot(x, df["cips_q_train"], lw=1.5, color="#2ca02c", linestyle=":", label="CIPS q-train")
    if "csnips_q_train" in df.columns and df["csnips_q_train"].notna().any():
        ax.plot(x, df["csnips_q_train"], lw=1.5, color="#d62728", linestyle=":", label="CSNIPS q-train")
    ax.set_xlabel("round")
    ax.set_ylabel("estimate")
    ax.set_title(f"{title_prefix} OPE Estimators by Round")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_ess(df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    x = df["round"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(x, df["ess"], lw=2.0, color="#1f77b4", label="ESS")
    ax.plot(x, df["ess_clip"], lw=2.0, color="#2ca02c", label="ESS clip")
    if "ess_q_train" in df.columns and df["ess_q_train"].notna().any():
        ax.plot(x, df["ess_q_train"], lw=1.8, color="#ff7f0e", linestyle=":", label="ESS q-train")
    if "ess_clip_q_train" in df.columns and df["ess_clip_q_train"].notna().any():
        ax.plot(x, df["ess_clip_q_train"], lw=1.8, color="#d62728", linestyle=":", label="ESS clip q-train")
    ax.set_xlabel("round")
    ax.set_ylabel("effective sample size")
    ax.set_title(f"{title_prefix} ESS by Round")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_gaussian_fit(fit_df: pd.DataFrame, out_path: Path, title_prefix: str) -> None:
    x = fit_df["order_sequence"].to_numpy(dtype=int)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    axes[0].plot(x, fit_df["nll"], color="#1f77b4", lw=2.0, marker="o", markersize=3)
    axes[0].set_ylabel("NLL")
    axes[0].set_title(f"{title_prefix} Truncated-Gaussian Fit Quality by order_sequence")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, fit_df["rmse"], color="#ff7f0e", lw=2.0, marker="o", markersize=3, label="rmse")
    axes[1].plot(x, fit_df["mae"], color="#2ca02c", lw=2.0, marker="o", markersize=3, label="mae")
    axes[1].set_xlabel("order_sequence")
    axes[1].set_ylabel("error")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _pick_round_indices(n: int, k: int = 4) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, k).round().astype(int).tolist()
    idx = sorted(set(idx))
    if idx[-1] != n - 1:
        idx[-1] = n - 1
    return idx


def _pick_anchor_round_indices(n: int) -> list[int]:
    """
    Select anchor rounds as requested:
    1, N/4, N/2, 3N/4, N (1-based), then convert to 0-based indices.
    Duplicates are removed while preserving order.
    """
    if n <= 0:
        return []
    anchors_1based = [
        1,
        int(math.ceil(n / 4.0)),
        int(math.ceil(n / 2.0)),
        int(math.ceil(3.0 * n / 4.0)),
        n,
    ]
    out: list[int] = []
    seen: set[int] = set()
    for r in anchors_1based:
        idx = max(0, min(n - 1, r - 1))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def _make_axes_grid(num_panels: int) -> tuple[plt.Figure, np.ndarray]:
    if num_panels <= 4:
        ncols = 2
    else:
        ncols = 3
    nrows = int(math.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.1 * ncols, 4.3 * nrows), sharex=True, sharey=True)
    axes_arr = np.array(axes).reshape(-1)
    return fig, axes_arr


def _poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += float(c) * (x ** d)
    return y


def _extract_degree(cols: list[str], prefix: str) -> int:
    vals = [int(c.split("_")[-1]) for c in cols if c.startswith(prefix)]
    if not vals:
        raise ValueError(f"Missing coefficient columns for prefix {prefix}")
    return max(vals)


def _optimal_curve(theta_grid: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    d_grid = np.linspace(delta_min, delta_max, 1000)
    z = theta_grid[:, None] - d_grid[None, :]
    p = 1.0 / (1.0 + np.exp(-z))
    exp_reward = p * (d_grid[None, :] - delta_min)
    return d_grid[np.argmax(exp_reward, axis=1)]


def _bound_mu_np(mu_raw: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    width = float(delta_max - delta_min)
    x = np.clip(mu_raw, -40.0, 40.0)
    return float(delta_min) + width * (1.0 / (1.0 + np.exp(-x)))


def _plot_policy_curves_selected_rounds(
    *,
    value_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    k_rounds: int = 5,
) -> pd.DataFrame:
    n = len(value_df)
    idxs = _pick_anchor_round_indices(n)
    if len(idxs) > k_rounds:
        idxs = idxs[:k_rounds]
    theta_min = float(tuples_df["proficiency"].min())
    theta_max = float(tuples_df["proficiency"].max())
    delta_min = float(tuples_df["difficulties"].min())
    delta_max = float(tuples_df["difficulties"].max())
    theta_grid = np.linspace(theta_min, theta_max, 400)
    opt = _optimal_curve(theta_grid, delta_min=delta_min, delta_max=delta_max)

    mu_deg = _extract_degree(coeff_df.columns.tolist(), "beta_mu_")
    sig_deg = _extract_degree(coeff_df.columns.tolist(), "beta_sigma_")

    fig, axes = _make_axes_grid(max(1, len(idxs)))
    summary_rows = []
    for ax_idx, row_idx in enumerate(idxs):
        ax = axes[ax_idx]
        rv = value_df.iloc[row_idx]
        round_no = int(rv["round"])
        order_val = int(rv["order_sequence"])
        cb = coeff_df[coeff_df["round"] == round_no]
        fb = fit_df[fit_df["order_sequence"] == order_val]
        if cb.empty or fb.empty:
            continue
        cb = cb.iloc[0]
        fb = fb.iloc[0]

        b_mu = np.array([float(fb[f"beta_mu_{d}"]) for d in range(mu_deg + 1)], dtype=float)
        b_sig = np.array([float(fb[f"beta_sigma_{d}"]) for d in range(sig_deg + 1)], dtype=float)
        l_mu = np.array([float(cb[f"beta_mu_{d}"]) for d in range(mu_deg + 1)], dtype=float)
        l_sig = np.array([float(cb[f"beta_sigma_{d}"]) for d in range(sig_deg + 1)], dtype=float)

        b_mu_curve = _poly_eval(theta_grid, b_mu)
        b_sig_curve = np.maximum(np.exp(_poly_eval(theta_grid, b_sig)), 1e-8)
        l_mu_raw_curve = _poly_eval(theta_grid, l_mu)
        l_mu_curve = _bound_mu_np(l_mu_raw_curve, delta_min=delta_min, delta_max=delta_max)
        l_sig_curve = np.maximum(np.exp(_poly_eval(theta_grid, l_sig)), 1e-8)

        ax.plot(theta_grid, b_mu_curve, color="#ff7f0e", lw=2.0, label="behavior mean")
        ax.fill_between(theta_grid, b_mu_curve - b_sig_curve, b_mu_curve + b_sig_curve, color="#ff7f0e", alpha=0.12)
        ax.plot(theta_grid, l_mu_curve, color="#1f77b4", lw=2.2, label="learned mean")
        ax.fill_between(theta_grid, l_mu_curve - l_sig_curve, l_mu_curve + l_sig_curve, color="#1f77b4", alpha=0.12)
        ax.plot(theta_grid, opt, color="#2ca02c", lw=2.0, linestyle="--", label="optimal")
        ax.set_title(f"round={round_no} order={order_val}")
        ax.grid(alpha=0.25)

        summary_rows.append(
            {
                "round": round_no,
                "order_sequence": order_val,
                "behavior_irt_value": float(rv["behavior_irt_value"]),
                "learned_irt_value": float(rv["learned_irt_value"]),
                "optimal_irt_value": float(rv["optimal_irt_value"]),
                "snips": float(rv["snips"]),
                "ips": float(rv["ips"]),
                "cips": float(rv["cips"]),
                "csnips": float(rv["csnips"]),
            }
        )

    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{title_prefix} Policy Curves (selected rounds)", y=0.98)
    fig.text(0.5, 0.03, "theta (proficiency)", ha="center")
    fig.text(0.03, 0.5, "delta (difficulty)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def _plot_behavior_fit_cloud_selected_rounds(
    *,
    value_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    out_path: Path,
    title_prefix: str,
    k_rounds: int = 5,
    max_points: int = 3000,
) -> pd.DataFrame:
    n = len(value_df)
    idxs = _pick_anchor_round_indices(n)
    if len(idxs) > k_rounds:
        idxs = idxs[:k_rounds]
    theta_min = float(tuples_df["proficiency"].min())
    theta_max = float(tuples_df["proficiency"].max())
    delta_min = float(tuples_df["difficulties"].min())
    delta_max = float(tuples_df["difficulties"].max())
    theta_grid = np.linspace(theta_min, theta_max, 400)
    rng = np.random.default_rng(42)

    mu_deg = _extract_degree(fit_df.columns.tolist(), "beta_mu_")
    sig_deg = _extract_degree(fit_df.columns.tolist(), "beta_sigma_")

    fig, axes = _make_axes_grid(max(1, len(idxs)))
    rows = []
    for ax_idx, row_idx in enumerate(idxs):
        ax = axes[ax_idx]
        rv = value_df.iloc[row_idx]
        round_no = int(rv["round"])
        order_val = int(rv["order_sequence"])
        fb = fit_df[fit_df["order_sequence"] == order_val]
        if fb.empty:
            continue
        fb = fb.iloc[0]

        g = tuples_df[tuples_df["order_sequence"] == order_val]
        if len(g) > max_points:
            sel = rng.choice(len(g), size=max_points, replace=False)
            g = g.iloc[sel]
        xs = g["proficiency"].to_numpy(dtype=float)
        ys = g["difficulties"].to_numpy(dtype=float)

        b_mu = np.array([float(fb[f"beta_mu_{d}"]) for d in range(mu_deg + 1)], dtype=float)
        b_sig = np.array([float(fb[f"beta_sigma_{d}"]) for d in range(sig_deg + 1)], dtype=float)
        mu_curve = _poly_eval(theta_grid, b_mu)
        sig_curve = np.maximum(np.exp(_poly_eval(theta_grid, b_sig)), 1e-8)

        ax.scatter(xs, ys, s=8, alpha=0.16, color="#777777")
        ax.plot(theta_grid, mu_curve, color="#ff7f0e", lw=2.2, label="behavior mean")
        ax.fill_between(theta_grid, mu_curve - sig_curve, mu_curve + sig_curve, color="#ff7f0e", alpha=0.15)
        ax.set_title(
            f"round={round_no} order={order_val} n={len(g)} nll={float(fb['nll']):.4f}"
        )
        ax.grid(alpha=0.25)
        rows.append(
            {
                "round": round_no,
                "order_sequence": order_val,
                "n_obs_selected_cloud": int(len(g)),
                "fit_nll": float(fb["nll"]),
                "fit_rmse": float(fb["rmse"]),
                "fit_mae": float(fb["mae"]),
            }
        )

    for j in range(len(idxs), len(axes)):
        axes[j].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{title_prefix} Behavior Fit vs Data Cloud (selected rounds)", y=0.98)
    fig.text(0.5, 0.03, "theta (proficiency)", ha="center")
    fig.text(0.03, 0.5, "delta (difficulty)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return pd.DataFrame(rows)


def _plot_shift_histograms(
    *,
    value_df: pd.DataFrame,
    tuples_df: pd.DataFrame,
    out_theta: Path,
    out_delta: Path,
    title_prefix: str,
    k_rounds: int = 5,
) -> pd.DataFrame:
    idxs = _pick_anchor_round_indices(len(value_df))
    if len(idxs) > k_rounds:
        idxs = idxs[:k_rounds]
    chosen = [value_df.iloc[i] for i in idxs]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(chosen))))
    rows = []

    fig1, ax1 = plt.subplots(figsize=(10.2, 5.4))
    for c, rv in zip(cmap, chosen):
        order_val = int(rv["order_sequence"])
        rr = int(rv["round"])
        th = tuples_df.loc[tuples_df["order_sequence"] == order_val, "proficiency"].to_numpy(dtype=float)
        ax1.hist(th, bins=50, density=True, alpha=0.26, color=c, label=f"r{rr}/o{order_val} (n={len(th)})")
        rows.append({"round": rr, "order_sequence": order_val, "n_theta": int(len(th)), "theta_mean": float(np.mean(th))})
    ax1.set_title(f"{title_prefix} Proficiency Distribution Shift (selected rounds)")
    ax1.set_xlabel("theta (proficiency)")
    ax1.set_ylabel("density")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="best", fontsize=8)
    fig1.tight_layout()
    fig1.savefig(out_theta, dpi=170)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10.2, 5.4))
    for c, rv in zip(cmap, chosen):
        order_val = int(rv["order_sequence"])
        rr = int(rv["round"])
        dd = tuples_df.loc[tuples_df["order_sequence"] == order_val, "difficulties"].to_numpy(dtype=float)
        ax2.hist(dd, bins=50, density=True, alpha=0.26, color=c, label=f"r{rr}/o{order_val} (n={len(dd)})")
        rows[-1]["n_delta"] = int(len(dd))
        rows[-1]["delta_mean"] = float(np.mean(dd))
    ax2.set_title(f"{title_prefix} Difficulty Distribution Shift (selected rounds)")
    ax2.set_xlabel("delta (difficulty)")
    ax2.set_ylabel("density")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_delta, dpi=170)
    plt.close(fig2)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-dataset figure bundles from run_full_pipeline outputs.")
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="pix_mapping/exp_all_quantile20_truncgauss_snips_ep2",
    )
    parser.add_argument("--out-subdir", type=str, default="dataset_figures")
    args = parser.parse_args()

    exp_root = Path(args.experiment_root).resolve()
    out_root = exp_root / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in exp_root.iterdir() if p.is_dir() and p.name.endswith("_snips")])
    if not run_dirs:
        raise ValueError(f"No dataset run directories found in {exp_root}")

    for run_dir in run_dirs:
        dataset = run_dir.name.replace("_snips", "")
        pval = run_dir / "policy_value_table.csv"
        pfit = run_dir / "gaussian_fit_by_order.csv"
        pcoeff = run_dir / "policy_coefficients_by_round.csv"
        ptuples = run_dir / "offpolicy_tuples.csv"
        if not pval.exists() or not pfit.exists() or not pcoeff.exists() or not ptuples.exists():
            continue
        df = pd.read_csv(pval)
        fit_df = pd.read_csv(pfit)
        coeff_df = pd.read_csv(pcoeff)
        tuples_df = pd.read_csv(ptuples)
        if df.empty or fit_df.empty or coeff_df.empty or tuples_df.empty:
            continue

        dst = out_root / dataset
        dst.mkdir(parents=True, exist_ok=True)

        # 1) Most important table: per-round estimator values + model-based policy values.
        round_cols = [
            "round",
            "order_sequence",
            "objective",
            "objective_estimate",
            "objective_estimate_q",
            "ips",
            "snips",
            "cips",
            "csnips",
            "ips_q_batch",
            "snips_q_batch",
            "cips_q_batch",
            "csnips_q_batch",
            "ips_q_train",
            "snips_q_train",
            "cips_q_train",
            "csnips_q_train",
            "behavior_empirical_batch_mean",
            "behavior_empirical_batch_mean_q",
            "behavior_irt_value",
            "behavior_irt_value_q",
            "learned_irt_value",
            "learned_irt_value_q",
            "optimal_irt_value",
            "optimal_irt_value_q",
            "ess",
            "ess_q_train",
            "ess_clip",
            "ess_clip_q_train",
        ]
        round_table = df[[c for c in round_cols if c in df.columns]].copy()
        round_table.to_csv(dst / "policy_estimators_and_irt_by_round.csv", index=False)

        # Keep key source tables alongside figures for quick inspection.
        df.to_csv(dst / "policy_value_table.csv", index=False)
        fit_df.to_csv(dst / "gaussian_fit_by_order.csv", index=False)
        coeff_df.to_csv(dst / "policy_coefficients_by_round.csv", index=False)

        _plot_irt_values(df, dst / "irt_values_by_round.png", dataset)
        _plot_ope(df, dst / "ope_estimators_by_round.png", dataset)
        _plot_ess(df, dst / "ess_by_round.png", dataset)
        _plot_gaussian_fit(fit_df, dst / "gaussian_fit_quality_by_order.png", dataset)

        # 2) Policy curves: behavior vs learned vs optimal for selected rounds.
        selected_policy = _plot_policy_curves_selected_rounds(
            value_df=df,
            coeff_df=coeff_df,
            fit_df=fit_df,
            tuples_df=tuples_df,
            out_path=dst / "policy_curves_selected_rounds.png",
            title_prefix=dataset,
            k_rounds=5,
        )
        selected_policy.to_csv(dst / "policy_curves_selected_rounds_table.csv", index=False)

        # 3) Behavior fit quality cloud (4 representative rounds).
        selected_fit = _plot_behavior_fit_cloud_selected_rounds(
            value_df=df,
            fit_df=fit_df,
            tuples_df=tuples_df,
            out_path=dst / "behavior_fit_cloud_selected_rounds.png",
            title_prefix=dataset,
            k_rounds=5,
            max_points=3000,
        )
        selected_fit.to_csv(dst / "behavior_fit_selected_rounds_table.csv", index=False)

        # 4) Context shift histograms (theta and difficulty).
        shift_df = _plot_shift_histograms(
            value_df=df,
            tuples_df=tuples_df,
            out_theta=dst / "theta_hist_shift_selected_rounds.png",
            out_delta=dst / "difficulty_hist_shift_selected_rounds.png",
            title_prefix=dataset,
            k_rounds=5,
        )
        shift_df.to_csv(dst / "selected_rounds_distribution_shift_table.csv", index=False)

        final = df.iloc[-1]
        summary_lines = [
            f"dataset: {dataset}",
            f"rounds: {len(df)}",
            f"final_round: {int(final['round'])}",
            f"final_order_sequence: {int(final['order_sequence'])}",
            f"final_snips: {float(final['snips']):.6f}",
            f"final_ips: {float(final['ips']):.6f}",
            f"final_cips: {float(final['cips']):.6f}",
            f"final_csnips: {float(final['csnips']):.6f}",
            f"final_behavior_irt: {float(final['behavior_irt_value']):.6f}",
            f"final_learned_irt: {float(final['learned_irt_value']):.6f}",
            f"final_optimal_irt: {float(final['optimal_irt_value']):.6f}",
            "artifacts:",
            "- policy_estimators_and_irt_by_round.csv",
            "- policy_curves_selected_rounds.png",
            "- policy_curves_selected_rounds_table.csv",
            "- behavior_fit_cloud_selected_rounds.png",
            "- behavior_fit_selected_rounds_table.csv",
            "- theta_hist_shift_selected_rounds.png",
            "- difficulty_hist_shift_selected_rounds.png",
            "- selected_rounds_distribution_shift_table.csv",
        ]
        (dst / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"saved_bundle={dst}")

    print(f"all_bundles_root={out_root}")


if __name__ == "__main__":
    main()
