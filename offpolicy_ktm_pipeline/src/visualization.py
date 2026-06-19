from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gaussian_regression import poly_eval

_OBJECTIVE_COLORS = {
    "ips": "#1f77b4",
    "snips": "#ff7f0e",
    "dr": "#e377c2",
    "mis": "#9467bd",
}


def _bound_mu(mu_raw: np.ndarray, delta_min: float, delta_max: float) -> np.ndarray:
    width = float(delta_max - delta_min)
    x = np.clip(mu_raw, -40.0, 40.0)
    return float(delta_min) + width / (1.0 + np.exp(-x))


def _extract_coeffs(row: pd.Series, prefix: str) -> np.ndarray:
    cols = sorted([c for c in row.index if c.startswith(prefix)], key=lambda c: int(c.split("_")[-1]))
    return np.array([float(row[c]) for c in cols], dtype=float)


def _anchor_indices(n: int) -> list[int]:
    """Return 0-based indices for anchor rounds 1, N/4, N/2, 3N/4, N (1-based)."""
    if n <= 0:
        return []
    anchors = [1, int(math.ceil(n / 4)), int(math.ceil(n / 2)), int(math.ceil(3 * n / 4)), n]
    seen: set[int] = set()
    out = []
    for r in anchors:
        idx = max(0, min(n - 1, r - 1))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def plot_training_history(
    history_df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = ("ips", "snips", "cips", "csnips", "dr", "sndr", "cdr", "csndr"),
    title: str = "Off-Policy Training Metrics",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = history_df["epoch"].to_numpy(dtype=int)
    for m in metrics:
        if m in history_df.columns:
            ax.plot(x, history_df[m], linewidth=2.0, label=m)
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_policy_curves(
    *,
    theta_grid: np.ndarray,
    behavior_mu: np.ndarray,
    behavior_sigma: np.ndarray,
    learned_mu: np.ndarray,
    learned_sigma: np.ndarray,
    optimal_delta: np.ndarray | None = None,
    y_min: float = -4.0,
    y_max: float = 4.0,
    title: str = "Behavior vs Learned Policy",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    ax.plot(theta_grid, behavior_mu, color="#ff7f0e", lw=2.0, label="behavior mean")
    ax.fill_between(
        theta_grid,
        behavior_mu - behavior_sigma,
        behavior_mu + behavior_sigma,
        color="#ff7f0e",
        alpha=0.16,
        label="behavior +/- sigma",
    )

    ax.plot(theta_grid, learned_mu, color="#1f77b4", lw=2.2, label="learned mean")
    ax.fill_between(
        theta_grid,
        learned_mu - learned_sigma,
        learned_mu + learned_sigma,
        color="#1f77b4",
        alpha=0.16,
        label="learned +/- sigma",
    )

    if optimal_delta is not None:
        ax.plot(theta_grid, optimal_delta, color="#2ca02c", lw=2.2, linestyle="--", label="optimal (IRT)")

    ax.set_xlabel("theta")
    ax.set_ylabel("delta")
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_objective_comparison(
    *,
    theta_grid: np.ndarray,
    fit_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    delta_min: float,
    delta_max: float,
    sigma_floor: float = 1e-4,
    optimal_delta: np.ndarray | None = None,
    title: str = "Learned Policies vs Behavior (Last Round)",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot E[delta|theta] = mu(theta) ± 2*sigma(theta) for each trained objective,
    plus behavior at first and last round, and the optimal IRT curve.

    fit_df:   rows indexed by order_sequence with beta_mu_* / beta_sigma_* columns.
    coeff_df: rows indexed by objective with beta_mu_* / beta_sigma_* columns.
    """
    fig, ax = plt.subplots(figsize=(10.0, 5.8))

    # Behavior at first and last round.
    fit_sorted = fit_df.sort_values("order_sequence").reset_index(drop=True)
    for label, row, color, ls in [
        ("behavior (round 1)", fit_sorted.iloc[0], "#8c564b", ":"),
        ("behavior (last round)", fit_sorted.iloc[-1], "#7f7f7f", ":"),
    ]:
        b_mu_raw = poly_eval(theta_grid, _extract_coeffs(row, "beta_mu_"))
        b_mu = _bound_mu(b_mu_raw, delta_min, delta_max)
        b_sig = np.maximum(np.exp(poly_eval(theta_grid, _extract_coeffs(row, "beta_sigma_"))), sigma_floor)
        ax.plot(theta_grid, b_mu, color=color, lw=2.0, linestyle=ls, label=label)
        ax.fill_between(theta_grid, b_mu - 2 * b_sig, b_mu + 2 * b_sig, color=color, alpha=0.10)

    # Learned policy per objective.
    for _, row in coeff_df.iterrows():
        obj = str(row["objective"])
        color = _OBJECTIVE_COLORS.get(obj, "#17becf")
        l_mu_raw = poly_eval(theta_grid, _extract_coeffs(row, "beta_mu_"))
        l_mu = _bound_mu(l_mu_raw, delta_min, delta_max)
        l_sig = np.maximum(np.exp(poly_eval(theta_grid, _extract_coeffs(row, "beta_sigma_"))), sigma_floor)
        ax.plot(theta_grid, l_mu, color=color, lw=2.2, label=f"learned ({obj})")
        ax.fill_between(theta_grid, l_mu - 2 * l_sig, l_mu + 2 * l_sig, color=color, alpha=0.12)

    if optimal_delta is not None:
        ax.plot(theta_grid, optimal_delta, color="#2ca02c", lw=2.2, linestyle="--", label="optimal (IRT)")

    margin = 0.05 * (delta_max - delta_min)
    ax.set_ylim(delta_min - margin, delta_max + margin)
    ax.set_xlabel("theta (proficiency)")
    ax.set_ylabel("delta (difficulty)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_behavior_evolution(
    *,
    theta_grid: np.ndarray,
    fit_df: pd.DataFrame,
    delta_min: float,
    delta_max: float,
    sigma_floor: float = 1e-4,
    title: str = "Behavior Policy Evolution by Round",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot mu(theta) ± 2*sigma(theta) of the behavior policy at anchor rounds
    (1, N/4, N/2, 3N/4, N), one colored curve per round.
    """
    fit_sorted = fit_df.sort_values("order_sequence").reset_index(drop=True)
    n = len(fit_sorted)
    idxs = _anchor_indices(n)

    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(idxs) - 1, 1)) for i in range(len(idxs))]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for i, idx in enumerate(idxs):
        row = fit_sorted.iloc[idx]
        order_val = int(row["order_sequence"])
        b_mu_raw = poly_eval(theta_grid, _extract_coeffs(row, "beta_mu_"))
        b_mu = _bound_mu(b_mu_raw, delta_min, delta_max)
        b_sig = np.maximum(np.exp(poly_eval(theta_grid, _extract_coeffs(row, "beta_sigma_"))), sigma_floor)
        c = colors[i]
        ax.plot(theta_grid, b_mu, color=c, lw=2.0, label=f"round {order_val}")
        ax.fill_between(theta_grid, b_mu - 2 * b_sig, b_mu + 2 * b_sig, color=c, alpha=0.12)

    margin = 0.05 * (delta_max - delta_min)
    ax.set_ylim(delta_min - margin, delta_max + margin)
    ax.set_xlabel("theta (proficiency)")
    ax.set_ylabel("delta (difficulty)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax


def build_policy_curves(
    *,
    theta_grid: np.ndarray,
    beta_mu_behavior: np.ndarray,
    beta_sigma_behavior: np.ndarray,
    beta_mu_learned: np.ndarray,
    beta_sigma_learned: np.ndarray,
    sigma_floor: float = 1e-4,
) -> dict[str, np.ndarray]:
    b_mu = poly_eval(theta_grid, beta_mu_behavior)
    b_sigma = np.maximum(np.exp(poly_eval(theta_grid, beta_sigma_behavior)), sigma_floor)
    l_mu = poly_eval(theta_grid, beta_mu_learned)
    l_sigma = np.maximum(np.exp(poly_eval(theta_grid, beta_sigma_learned)), sigma_floor)
    return {
        "behavior_mu": b_mu,
        "behavior_sigma": b_sigma,
        "learned_mu": l_mu,
        "learned_sigma": l_sigma,
    }
