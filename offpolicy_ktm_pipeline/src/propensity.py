from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from .gaussian_regression import poly_eval


def _infer_degree(columns: list[str], prefix: str) -> int:
    values = [int(c.split("_")[-1]) for c in columns if c.startswith(prefix)]
    if not values:
        raise ValueError(f"Missing columns with prefix {prefix}")
    return max(values)


def _nearest_available_order(x: np.ndarray, available: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(available, x)
    pos = np.clip(pos, 0, len(available) - 1)
    left = np.maximum(pos - 1, 0)
    right = pos
    choose_left = np.abs(x - available[left]) <= np.abs(x - available[right])
    return np.where(choose_left, available[left], available[right])


def _log_diff_exp(log_x: np.ndarray, log_y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = np.empty_like(log_x)
    mask = log_x > log_y
    out[mask] = log_x[mask] + np.log1p(-np.exp(log_y[mask] - log_x[mask]))
    out[~mask] = np.log(eps)
    return out


def attach_behavior_params(
    df_ktm: pd.DataFrame,
    fit_df: pd.DataFrame,
    *,
    order_col: str = "order_sequence",
    fill_missing_nearest_order: bool = True,
) -> pd.DataFrame:
    """
    Attach per-order Gaussian behavior coefficients to KTM rows.
    """
    work = df_ktm.copy()
    fit = fit_df.copy()

    if "order_sequence" not in fit.columns:
        if "order_min" in fit.columns and "order_max" in fit.columns:
            fit = fit[fit["order_min"] == fit["order_max"]].copy()
            fit["order_sequence"] = fit["order_min"].astype(int)
        else:
            raise ValueError("fit_df must contain order_sequence or order_min/order_max")

    meta_cols = [c for c in ["distribution", "delta_min", "delta_max"] if c in fit.columns]
    keep_cols = ["subset", "order_sequence"] + meta_cols + [c for c in fit.columns if c.startswith("beta_")]
    fit = fit[keep_cols].copy()
    fit["order_sequence"] = fit["order_sequence"].astype(int)

    merged = work.merge(
        fit,
        left_on=order_col,
        right_on="order_sequence",
        how="left",
        suffixes=("", "_fit"),
    )
    if "order_sequence_fit" in merged.columns:
        merged = merged.drop(columns=["order_sequence_fit"])

    if fill_missing_nearest_order:
        miss = merged["subset"].isna()
        if miss.any():
            available = np.sort(fit["order_sequence"].unique().astype(int))
            target = merged.loc[miss, order_col].to_numpy(dtype=int)
            nearest = _nearest_available_order(target, available)
            fallback = fit.rename(columns={"order_sequence": "nearest_order"})
            miss_df = pd.DataFrame(
                {"nearest_order": nearest},
                index=merged.index[miss],
            )
            miss_df = miss_df.merge(fallback, on="nearest_order", how="left").set_index(miss_df.index)
            for c in [x for x in keep_cols if x != "order_sequence"]:
                merged.loc[miss, c] = miss_df[c].to_numpy()

    return merged


def compute_propensity_and_reward(
    df_with_params: pd.DataFrame,
    *,
    theta_col: str = "proficiency",
    delta_col: str = "difficulties",
    correct_col: str = "correct",
    sigma_floor: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute Gaussian propensity and reward = correct * (difficulty - min_difficulty).
    """
    out = df_with_params.copy()
    theta = out[theta_col].to_numpy(dtype=float)
    delta = out[delta_col].to_numpy(dtype=float)

    mu_degree = _infer_degree(out.columns.tolist(), "beta_mu_")
    sigma_degree = _infer_degree(out.columns.tolist(), "beta_sigma_")

    mu = np.zeros_like(theta, dtype=float)
    log_sigma = np.zeros_like(theta, dtype=float)
    for d in range(mu_degree + 1):
        mu += out[f"beta_mu_{d}"].to_numpy(dtype=float) * (theta ** d)
    for d in range(sigma_degree + 1):
        log_sigma += out[f"beta_sigma_{d}"].to_numpy(dtype=float) * (theta ** d)

    sigma = np.maximum(np.exp(log_sigma), sigma_floor)
    z = (delta - mu) / sigma
    log_prop = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z ** 2)

    dist = "gaussian"
    if "distribution" in out.columns:
        first = out["distribution"].dropna().astype(str)
        if len(first) > 0:
            dist = first.iloc[0]

    if dist == "truncated_gaussian":
        if "delta_min" not in out.columns or "delta_max" not in out.columns:
            raise ValueError("truncated_gaussian requires delta_min and delta_max columns in fit dataframe")
        a = out["delta_min"].to_numpy(dtype=float)
        b = out["delta_max"].to_numpy(dtype=float)
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        # Match GlobalGaussianPolicy.log_prob exactly:
        # use log_ndtr for numerical stability, and switch to the survival-function
        # branch (log P(X > alpha) = log_ndtr(-alpha)) when alpha > 0 to avoid
        # catastrophic cancellation in the far upper tail.
        log_cdf_beta = log_ndtr(beta)
        log_cdf_alpha = log_ndtr(alpha)
        log_sf_alpha = log_ndtr(-alpha)
        log_sf_beta = log_ndtr(-beta)
        log_z_cdf = _log_diff_exp(log_cdf_beta, log_cdf_alpha, eps=1e-300)
        log_z_sf = _log_diff_exp(log_sf_alpha, log_sf_beta, eps=1e-300)
        log_z = np.where(alpha > 0.0, log_z_sf, log_z_cdf)
        log_prop = log_prop - log_z
        in_support = (delta >= a) & (delta <= b)
        log_prop = np.where(in_support, log_prop, np.log(1e-12))

    prop = np.exp(log_prop)

    out["mu_hat"] = mu
    out["sigma_hat"] = sigma
    out["log_propensity"] = log_prop
    out["propensity"] = prop

    diff_min = float(np.nanmin(delta))
    out["reward"] = out[correct_col].to_numpy(dtype=float) * (delta - diff_min)
    return out


def build_offpolicy_dataset(
    df_ktm: pd.DataFrame,
    fit_df: pd.DataFrame,
    *,
    order_col: str = "order_sequence",
    sigma_floor: float = 1e-4,
    fill_missing_nearest_order: bool = True,
) -> pd.DataFrame:
    """
    End-to-end: attach behavior params, compute propensities, and return OPE tuples.
    """
    merged = attach_behavior_params(
        df_ktm,
        fit_df,
        order_col=order_col,
        fill_missing_nearest_order=fill_missing_nearest_order,
    )
    out = compute_propensity_and_reward(merged, sigma_floor=sigma_floor)
    out["sequence"] = out["user"]

    ordered_cols = [
        "sequence",
        "user",
        "item",
        "correct",
        "reward",
        "answer_number",
        "sequence_position",
        "order_sequence",
        "sequence_length",
        "proficiency",
        "difficulties",
        "subset",
        "mu_hat",
        "sigma_hat",
        "log_propensity",
        "propensity",
    ]
    return out[[c for c in ordered_cols if c in out.columns]].copy()
