"""
plot_policies.py — Visualise learned vs behaviour policies from run_ope.py output.

Reads from a results directory produced by run_ope.py and generates a two-panel
figure: policy means μ(θ) only, and μ(θ) ± 1σ(θ) with uncertainty bands.

Usage:
    python experiments/plot_policies.py --results-dir experiments/results/skill_builder
    python experiments/plot_policies.py --results-dir experiments/results/attempts
    python experiments/plot_policies.py --results-dir experiments/results/assistments8000
    python experiments/plot_policies.py --results-dir experiments/results/skill_builder --output policies.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid


# ── policy curve helpers ──────────────────────────────────────────────────────

def eval_mu(theta: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """μ(θ) = β0 + β1·θ  (degree-1 polynomial)."""
    return b0 + b1 * theta


def eval_sigma(theta: np.ndarray, b0: float, b1: float, b2: float,
               sigma_floor: float) -> np.ndarray:
    """σ(θ) = exp(β0 + β1·θ + β2·θ²) + floor  (degree-2 log-σ polynomial)."""
    log_sig = b0 + b1 * theta + b2 * theta ** 2
    return np.exp(log_sig) + sigma_floor


def policy_mu_sigma(theta: np.ndarray, row: pd.Series, *,
                    sigma_floor: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    mu    = eval_mu(theta, row["beta_mu_0"], row["beta_mu_1"])
    sigma = eval_sigma(theta, row["beta_sigma_0"], row["beta_sigma_1"],
                       row["beta_sigma_2"], sigma_floor)
    return mu, sigma


def irt_optimal_delta(theta: float, delta_min: float, delta_max: float) -> float:
    """δ*(θ) = argmax_{δ∈[δ_min,δ_max]} sigmoid(θ−δ)·(δ−δ_min).
    Solved with bounded Brent's method; falls back to θ on failure."""
    def neg_reward(delta: float) -> float:
        return -(expit(theta - delta) * (delta - delta_min))
    try:
        res = minimize_scalar(neg_reward,
                              bounds=(delta_min + 1e-4, delta_max - 1e-4),
                              method="bounded")
        return float(np.clip(res.x, delta_min, delta_max))
    except Exception:
        return float(np.clip(theta, delta_min, delta_max))


# ── colours / labels ─────────────────────────────────────────────────────────

_OBJ_COLOR = {
    "behavior": "#555555",
    "ips":      "#1f77b4",
    "snips":    "#ff7f0e",
    "dr":       "#e377c2",
    "mis":      "#9467bd",
    "oracle":   "#2ca02c",
}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot learned vs behaviour policies from run_ope.py output."
    )
    parser.add_argument("--results-dir", required=True,
                        help="Path to a results directory, e.g. experiments/results/skill_builder")
    parser.add_argument("--theta-min",   type=float, default=-3.0,
                        help="Left boundary for θ grid (default: -3)")
    parser.add_argument("--theta-max",   type=float, default=3.0,
                        help="Right boundary for θ grid (default: 3)")
    parser.add_argument("--n-points",    type=int,   default=300,
                        help="Number of θ points (default: 300)")
    parser.add_argument("--sigma-floor", type=float, default=1e-4,
                        help="Minimum σ value (must match run_ope.py, default: 1e-4)")
    parser.add_argument("--output",      type=str,   default=None,
                        help="Save figure here (e.g. policies.png). "
                             "Defaults to <results-dir>/policies.png")
    args = parser.parse_args()

    results_dir  = Path(args.results_dir)
    dataset_name = results_dir.name

    # ── load artefacts ────────────────────────────────────────────────────────
    fit_df  = pd.read_csv(results_dir / "gaussian_fit_train_by_order.csv")
    coef_df = pd.read_csv(results_dir / "learned_policy_coefficients.csv")
    est_df  = pd.read_csv(results_dir / "estimator_results_test.csv")

    delta_min = float(fit_df["delta_min"].iloc[0])
    delta_max = float(fit_df["delta_max"].iloc[0])

    # behaviour policy parameters = last (non-excluded) round in fit file
    last_order = int(fit_df["order_sequence"].max())
    beh_row    = fit_df[fit_df["order_sequence"] == last_order].iloc[0]

    snips_vals = dict(zip(est_df["policy"], est_df["snips"]))

    theta = np.linspace(args.theta_min, args.theta_max, args.n_points)

    # ── IRT oracle curve ──────────────────────────────────────────────────────
    oracle_delta = np.array(
        [irt_optimal_delta(t, delta_min, delta_max) for t in theta]
    )

    # ── two-panel figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Learned vs behaviour policies — {dataset_name}",
                 fontsize=13, fontweight="bold")

    for ax, show_sigma in zip(axes, [False, True]):
        # behavior policy (last round)
        mu_b, sig_b = policy_mu_sigma(theta, beh_row, sigma_floor=args.sigma_floor)
        mu_b = np.clip(mu_b, delta_min, delta_max)
        snips_b = snips_vals.get("behavior", float("nan"))
        ax.plot(theta, mu_b, color=_OBJ_COLOR["behavior"], lw=2, ls="--",
                label=f"Behavior  SNIPS={snips_b:.3f}")
        if show_sigma:
            ax.fill_between(
                theta,
                np.clip(mu_b - sig_b, delta_min, delta_max),
                np.clip(mu_b + sig_b, delta_min, delta_max),
                alpha=0.12, color=_OBJ_COLOR["behavior"],
            )

        # IRT oracle (no σ band — it's a point policy)
        ax.plot(theta, oracle_delta, color=_OBJ_COLOR["oracle"], lw=1.5, ls=":",
                label=f"IRT oracle  SNIPS={snips_vals.get('optimal_irt', float('nan')):.3f}")

        # learned policies
        for _, row in coef_df.iterrows():
            obj   = str(row["objective"])
            color = _OBJ_COLOR.get(obj, "gray")
            mu_l, sig_l = policy_mu_sigma(theta, row, sigma_floor=args.sigma_floor)
            mu_l = np.clip(mu_l, delta_min, delta_max)
            snips_l = snips_vals.get(f"learned_{obj}", float("nan"))
            ax.plot(theta, mu_l, color=color, lw=2,
                    label=f"Learned {obj.upper()}  SNIPS={snips_l:.3f}")
            if show_sigma:
                ax.fill_between(
                    theta,
                    np.clip(mu_l - sig_l, delta_min, delta_max),
                    np.clip(mu_l + sig_l, delta_min, delta_max),
                    alpha=0.12, color=color,
                )

        # δ support boundaries
        ax.axhline(delta_min, color="black", lw=0.6, ls="--", alpha=0.35)
        ax.axhline(delta_max, color="black", lw=0.6, ls="--", alpha=0.35)

        ax.set_xlabel("Student proficiency θ", fontsize=11)
        ax.set_ylabel("Assigned difficulty δ", fontsize=11)
        ax.set_title("Policy means μ(θ)" if not show_sigma else "Policy means ± 1σ(θ)",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()

    out_path = Path(args.output) if args.output else results_dir / "policies.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
