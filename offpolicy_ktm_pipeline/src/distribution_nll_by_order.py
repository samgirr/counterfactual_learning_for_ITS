from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .distribution_nll_comparison import fit_distribution
except ImportError:  # pragma: no cover - allows direct script execution
    from distribution_nll_comparison import fit_distribution


def compare_distributions_by_order(
    df: pd.DataFrame,
    *,
    order_col: str,
    theta_col: str,
    delta_col: str,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    max_iter: int,
    min_obs_per_order: int,
    free_student_t_dof: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    distributions = ["normal", "laplace", "logistic", "student_t"]

    for order_val, g in df.groupby(order_col, sort=True):
        if len(g) < min_obs_per_order:
            continue
        theta = g[theta_col].to_numpy(dtype=float)
        delta = g[delta_col].to_numpy(dtype=float)
        for dist in distributions:
            fit = fit_distribution(
                theta=theta,
                delta=delta,
                mu_degree=mu_degree,
                sigma_degree=sigma_degree,
                distribution=dist,
                sigma_floor=sigma_floor,
                max_iter=max_iter,
                free_student_t_dof=free_student_t_dof,
            )
            row: dict[str, object] = {
                "order_sequence": int(order_val),
                "distribution": dist,
                "n_obs": int(len(g)),
                "nll_mean": fit.nll_mean,
                "nll_total": fit.nll_total,
                "rmse_mu": fit.rmse_mu,
                "mae_mu": fit.mae_mu,
                "success": fit.success,
                "message": fit.message,
                "dof": fit.dof,
                "mu_degree": int(mu_degree),
                "sigma_degree": int(sigma_degree),
            }
            for d, c in enumerate(fit.beta_mu):
                row[f"beta_mu_{d}"] = float(c)
            for d, c in enumerate(fit.beta_scale):
                row[f"beta_scale_{d}"] = float(c)
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["order_sequence", "nll_mean"]).reset_index(drop=True)
    return out


def build_winner_table(detail_df: pd.DataFrame) -> pd.DataFrame:
    idx = detail_df.groupby("order_sequence")["nll_mean"].idxmin()
    win = detail_df.loc[idx].copy()
    win = win.sort_values("order_sequence").reset_index(drop=True)

    normal = detail_df[detail_df["distribution"] == "normal"][
        ["order_sequence", "nll_mean"]
    ].rename(columns={"nll_mean": "nll_normal"})
    win = win.merge(normal, on="order_sequence", how="left")
    win["delta_vs_normal"] = win["nll_normal"] - win["nll_mean"]
    return win


def plot_nll_curves(
    detail_df: pd.DataFrame,
    *,
    out_plot: str,
) -> None:
    pivot = detail_df.pivot(
        index="order_sequence",
        columns="distribution",
        values="nll_mean",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    color_map = {
        "normal": "#1f77b4",
        "laplace": "#ff7f0e",
        "logistic": "#2ca02c",
        "student_t": "#d62728",
    }
    for dist in ["normal", "laplace", "logistic", "student_t"]:
        if dist in pivot.columns:
            ax.plot(
                pivot.index.to_numpy(dtype=int),
                pivot[dist].to_numpy(dtype=float),
                lw=2.0,
                marker="o",
                markersize=3,
                color=color_map.get(dist, None),
                label=dist,
            )
    ax.set_xlabel("order_sequence")
    ax.set_ylabel("mean NLL")
    ax.set_title("Per-Order Distribution Fit (Lower is Better)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    fig.savefig(out_plot, dpi=170)
    plt.close(fig)


def plot_gain_vs_normal(
    winners_df: pd.DataFrame,
    *,
    out_plot: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    x = winners_df["order_sequence"].to_numpy(dtype=int)
    y = winners_df["delta_vs_normal"].to_numpy(dtype=float)
    ax.plot(x, y, color="#444444", lw=2.0, label="best NLL gain vs Normal")
    ax.axhline(0.0, color="#999999", lw=1.0, linestyle="--")
    ax.set_xlabel("order_sequence")
    ax.set_ylabel("NLL(normal) - NLL(best)")
    ax.set_title("Best Distribution Improvement vs Normal by Order")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    fig.savefig(out_plot, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare probabilistic regression distributions per order_sequence."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="pix_mapping/ktm_gaussian_propensity_order_u10k_mu1_sigma2.csv",
    )
    parser.add_argument("--order-col", type=str, default="order_sequence")
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--min-obs-per-order", type=int, default=200)
    parser.add_argument("--fixed-student-t-dof", action="store_true")
    parser.add_argument(
        "--out-detail-csv",
        type=str,
        default="pix_mapping/probabilistic_distribution_nll_by_order.csv",
    )
    parser.add_argument(
        "--out-winners-csv",
        type=str,
        default="pix_mapping/probabilistic_distribution_nll_by_order_winners.csv",
    )
    parser.add_argument(
        "--out-nll-plot",
        type=str,
        default="pix_mapping/probabilistic_distribution_nll_by_order_curves.png",
    )
    parser.add_argument(
        "--out-gain-plot",
        type=str,
        default="pix_mapping/probabilistic_distribution_nll_by_order_gain_vs_normal.png",
    )
    args = parser.parse_args()

    usecols = [args.order_col, args.theta_col, args.delta_col]
    df = pd.read_csv(args.data, usecols=usecols).dropna().copy()
    df[args.order_col] = df[args.order_col].astype(int)

    if args.sample_rows > 0 and args.sample_rows < len(df):
        df = df.sample(n=args.sample_rows, random_state=args.seed).copy()

    detail = compare_distributions_by_order(
        df,
        order_col=args.order_col,
        theta_col=args.theta_col,
        delta_col=args.delta_col,
        mu_degree=args.mu_degree,
        sigma_degree=args.sigma_degree,
        sigma_floor=args.sigma_floor,
        max_iter=args.max_iter,
        min_obs_per_order=args.min_obs_per_order,
        free_student_t_dof=not args.fixed_student_t_dof,
    )
    if detail.empty:
        raise ValueError("No order had enough observations for fitting.")

    winners = build_winner_table(detail)

    os.makedirs(os.path.dirname(args.out_detail_csv), exist_ok=True)
    detail.to_csv(args.out_detail_csv, index=False)
    winners.to_csv(args.out_winners_csv, index=False)

    plot_nll_curves(detail, out_plot=args.out_nll_plot)
    plot_gain_vs_normal(winners, out_plot=args.out_gain_plot)

    n_orders = int(winners["order_sequence"].nunique())
    best_counts = winners["distribution"].value_counts().to_dict()
    mean_gain = float(winners["delta_vs_normal"].mean())
    print(f"n_orders_compared={n_orders}")
    print(f"best_distribution_counts={best_counts}")
    print(f"mean_best_gain_vs_normal={mean_gain:.6f}")
    print(f"saved_detail_csv={args.out_detail_csv}")
    print(f"saved_winners_csv={args.out_winners_csv}")
    print(f"saved_nll_plot={args.out_nll_plot}")
    print(f"saved_gain_plot={args.out_gain_plot}")


if __name__ == "__main__":
    main()
