from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def poly_eval(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, b in enumerate(beta):
        y += float(b) * (x ** d)
    return y


def log_gaussian_pdf(delta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (delta - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z ** 2)


def build_density_ratio(
    theta_ref: np.ndarray,
    theta_cur: np.ndarray,
    theta_eval: np.ndarray,
    *,
    n_bins: int = 120,
    eps: float = 1e-8,
) -> np.ndarray:
    lo = float(min(theta_ref.min(), theta_cur.min(), theta_eval.min()))
    hi = float(max(theta_ref.max(), theta_cur.max(), theta_eval.max()))
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    q_counts, _ = np.histogram(theta_ref, bins=edges, density=True)
    p_counts, _ = np.histogram(theta_cur, bins=edges, density=True)
    ratio_bins = (q_counts + eps) / (p_counts + eps)
    idx = np.clip(np.digitize(theta_eval, edges[1:-1], right=False), 0, n_bins - 1)
    return ratio_bins[idx]


def extract_coefficients(row: pd.Series, prefix: str) -> np.ndarray:
    cols = [c for c in row.index if c.startswith(prefix)]
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    return np.array([float(row[c]) for c in cols], dtype=float)


def run_reweighted_eval(
    *,
    run_dir: str,
    reference: str = "round1",
    n_bins: int = 120,
    sigma_floor: float = 1e-4,
    out_csv: str | None = None,
    out_plot: str | None = None,
) -> pd.DataFrame:
    run_path = Path(run_dir)
    tuples = pd.read_csv(run_path / "offpolicy_tuples.csv")
    rounds = pd.read_csv(run_path / "policy_round_results.csv")
    coeffs = pd.read_csv(run_path / "policy_coefficients_by_round.csv")

    tuples["order_sequence"] = tuples["order_sequence"].astype(int)
    rounds = rounds.sort_values("round").reset_index(drop=True)

    if reference == "round1":
        first_order = int(rounds.iloc[0]["order_sequence"])
        ref_df = tuples[tuples["order_sequence"] == first_order].copy()
    elif reference == "global":
        ref_df = tuples.copy()
    else:
        raise ValueError("reference must be 'round1' or 'global'")

    theta_ref = ref_df["proficiency"].to_numpy(dtype=float)
    out_rows: list[dict[str, float]] = []

    seen_orders: list[int] = []
    for _, r in rounds.iterrows():
        round_idx = int(r["round"])
        order_val = int(r["order_sequence"])
        seen_orders.append(order_val)

        cur = tuples[tuples["order_sequence"].isin(seen_orders)].copy()
        theta = cur["proficiency"].to_numpy(dtype=float)
        delta = cur["difficulties"].to_numpy(dtype=float)
        reward = cur["reward"].to_numpy(dtype=float)
        log_mu = cur["log_propensity"].to_numpy(dtype=float)

        w_ctx = build_density_ratio(theta_ref, theta, theta, n_bins=n_bins)

        c = coeffs[coeffs["round"] == round_idx].iloc[0]
        beta_mu = extract_coefficients(c, "beta_mu_")
        beta_sigma = extract_coefficients(c, "beta_sigma_")
        mu = poly_eval(theta, beta_mu)
        sigma = np.maximum(np.exp(poly_eval(theta, beta_sigma)), sigma_floor)
        log_pi = log_gaussian_pdf(delta=delta, mu=mu, sigma=sigma)
        w_a = np.exp(np.clip(log_pi - log_mu, -20.0, np.log(50.0)))

        w_b = w_ctx
        w_pi = w_ctx * w_a

        beh_rew = float(np.sum(w_b * reward) / np.sum(w_b))
        ips_rew = float(np.mean(w_pi * reward))
        snips_rew = float(np.sum(w_pi * reward) / np.sum(w_pi))

        out_rows.append(
            {
                "round": round_idx,
                "order_sequence": order_val,
                "n_seen": int(len(cur)),
                "behavior_value_reweighted": beh_rew,
                "learned_ips_reweighted": ips_rew,
                "learned_snips_reweighted": snips_rew,
                "behavior_empirical_batch_mean_original": float(r.get("behavior_empirical_batch_mean", np.nan)),
                "ips_original": float(r.get("ips", np.nan)),
                "snips_original": float(r.get("snips", np.nan)),
            }
        )

    out = pd.DataFrame(out_rows)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    if out_plot is not None:
        fig, ax = plt.subplots(figsize=(10.2, 5.4))
        x = out["round"].to_numpy(dtype=int)
        ax.plot(x, out["behavior_empirical_batch_mean_original"], lw=2.0, color="#999999", label="behavior original")
        ax.plot(x, out["behavior_value_reweighted"], lw=2.0, color="#ff7f0e", label="behavior reweighted q(theta)")
        ax.plot(x, out["ips_original"], lw=2.0, color="#7f7f7f", linestyle="--", label="learned IPS original")
        ax.plot(x, out["learned_ips_reweighted"], lw=2.0, color="#1f77b4", label="learned IPS reweighted")
        ax.plot(x, out["snips_original"], lw=2.0, color="#2ca02c", linestyle="--", label="learned SNIPS original")
        ax.plot(x, out["learned_snips_reweighted"], lw=2.0, color="#d62728", label="learned SNIPS reweighted")
        ax.set_xlabel("round")
        ax.set_ylabel("value")
        ax.set_title(f"Reweighted Evaluation (reference={reference})")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=170)
        plt.close(fig)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Reweighted OPE evaluation with fixed reference q(theta).")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--reference", type=str, default="round1", choices=["round1", "global"])
    parser.add_argument("--n-bins", type=int, default=120)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--out-plot", type=str, default="")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    out_csv = args.out_csv if args.out_csv else str(run_path / f"reweighted_eval_{args.reference}.csv")
    out_plot = args.out_plot if args.out_plot else str(run_path / f"reweighted_eval_{args.reference}.png")

    out = run_reweighted_eval(
        run_dir=args.run_dir,
        reference=args.reference,
        n_bins=args.n_bins,
        sigma_floor=args.sigma_floor,
        out_csv=out_csv,
        out_plot=out_plot,
    )
    print(f"rows={len(out)}")
    print(f"saved_csv={Path(out_csv).resolve()}")
    print(f"saved_plot={Path(out_plot).resolve()}")


if __name__ == "__main__":
    main()
