from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import ndtr
from scipy.optimize import minimize


@dataclass
class GaussianRegressionFit:
    beta_mu: np.ndarray
    beta_sigma: np.ndarray
    nll: float
    rmse: float
    mae: float
    success: bool
    message: str


def _log_diff_exp(log_x: np.ndarray, log_y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Stable log(exp(log_x) - exp(log_y)) for log_x >= log_y.
    """
    out = np.empty_like(log_x)
    mask = log_x > log_y
    out[mask] = log_x[mask] + np.log1p(-np.exp(log_y[mask] - log_x[mask]))
    out[~mask] = np.log(eps)
    return out


def _log_truncnorm_pdf(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    a: float,
    b: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Log-density for Truncated Normal(x | mu, sigma, [a,b]).
    """
    z = (x - mu) / sigma
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    cst = 0.5 * np.log(2.0 * np.pi)
    log_phi = -cst - np.log(sigma) - 0.5 * (z ** 2)

    cdf_beta = np.clip(ndtr(beta), eps, 1.0)
    cdf_alpha = np.clip(ndtr(alpha), eps, 1.0)
    log_z = _log_diff_exp(np.log(cdf_beta), np.log(cdf_alpha), eps=eps)

    log_pdf = log_phi - log_z
    in_support = (x >= a) & (x <= b)
    # Values outside support get a very small likelihood.
    log_pdf = np.where(in_support, log_pdf, np.log(eps))
    return log_pdf


def poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def poly_eval(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for d, c in enumerate(coeffs):
        y += c * (x ** d)
    return y


def fit_gaussian_regression(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    mu_degree: int = 1,
    sigma_degree: int = 2,
    sigma_floor: float = 1e-4,
    distribution: str = "truncated_gaussian",
    delta_min: float | None = None,
    delta_max: float | None = None,
) -> GaussianRegressionFit:
    """
    Fit delta | theta with polynomial mu/log-sigma.

    Supported distributions:
    - gaussian
    - truncated_gaussian (support [delta_min, delta_max])
    """
    if distribution not in {"gaussian", "truncated_gaussian"}:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if distribution == "truncated_gaussian":
        if delta_min is None:
            delta_min = float(np.min(delta))
        if delta_max is None:
            delta_max = float(np.max(delta))
        if not float(delta_min) < float(delta_max):
            raise ValueError("For truncated_gaussian, delta_min must be < delta_max")

    x_mu = poly_design(theta, mu_degree)
    x_sig = poly_design(theta, sigma_degree)
    k_mu = x_mu.shape[1]
    cst = 0.5 * np.log(2.0 * np.pi)

    beta_mu0 = np.linalg.lstsq(x_mu, delta, rcond=None)[0]
    resid = delta - x_mu @ beta_mu0
    beta_sig0 = np.zeros(x_sig.shape[1], dtype=float)
    beta_sig0[0] = float(np.log(np.std(resid) + 1e-3))
    p0 = np.concatenate([beta_mu0, beta_sig0])

    def nll(params: np.ndarray) -> float:
        b_mu = params[:k_mu]
        b_sig = params[k_mu:]
        mu = x_mu @ b_mu
        sigma = np.maximum(np.exp(x_sig @ b_sig), sigma_floor)
        if distribution == "gaussian":
            z = (delta - mu) / sigma
            return float(np.mean(cst + np.log(sigma) + 0.5 * (z ** 2)))

        log_pdf = _log_truncnorm_pdf(
            x=delta,
            mu=mu,
            sigma=sigma,
            a=float(delta_min),
            b=float(delta_max),
        )
        return float(-np.mean(log_pdf))

    out = minimize(nll, p0, method="L-BFGS-B")
    p = out.x if out.success else p0
    beta_mu = p[:k_mu]
    beta_sigma = p[k_mu:]
    mu_hat = x_mu @ beta_mu
    rmse = float(np.sqrt(np.mean((delta - mu_hat) ** 2)))
    mae = float(np.mean(np.abs(delta - mu_hat)))

    return GaussianRegressionFit(
        beta_mu=beta_mu,
        beta_sigma=beta_sigma,
        nll=float(nll(p)),
        rmse=rmse,
        mae=mae,
        success=bool(out.success),
        message=str(out.message),
    )


def _fit_one_order(
    order_val: int,
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    distribution: str,
    delta_min: float | None,
    delta_max: float | None,
) -> dict[str, object]:
    fit = fit_gaussian_regression(
        theta,
        delta,
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=sigma_floor,
        distribution=distribution,
        delta_min=delta_min,
        delta_max=delta_max,
    )
    rec: dict[str, object] = {
        "subset": f"order_{int(order_val)}",
        "order_sequence": int(order_val),
        "order_min": int(order_val),
        "order_max": int(order_val),
        "n_obs": int(len(theta)),
        "distribution": distribution,
        "delta_min": float(delta_min) if delta_min is not None else float(np.nan),
        "delta_max": float(delta_max) if delta_max is not None else float(np.nan),
        "mu_degree": int(mu_degree),
        "sigma_degree": int(sigma_degree),
        "nll": fit.nll,
        "rmse": fit.rmse,
        "mae": fit.mae,
        "success": fit.success,
        "message": fit.message,
    }
    for d, val in enumerate(fit.beta_mu):
        rec[f"beta_mu_{d}"] = float(val)
    for d, val in enumerate(fit.beta_sigma):
        rec[f"beta_sigma_{d}"] = float(val)
    return rec


def fit_gaussian_by_order(
    df_ktm: pd.DataFrame,
    *,
    order_col: str = "order_sequence",
    theta_col: str = "proficiency",
    delta_col: str = "difficulties",
    mu_degree: int = 1,
    sigma_degree: int = 2,
    sigma_floor: float = 1e-4,
    min_obs_per_order: int = 200,
    distribution: str = "truncated_gaussian",
    delta_min: float | None = None,
    delta_max: float | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Fit one Gaussian regression model per order_sequence (parallelized across orders).
    """
    if distribution == "truncated_gaussian":
        if delta_min is None:
            delta_min = float(df_ktm[delta_col].min())
        if delta_max is None:
            delta_max = float(df_ktm[delta_col].max())

    tasks = [
        (int(order_val), g[theta_col].to_numpy(dtype=float), g[delta_col].to_numpy(dtype=float))
        for order_val, g in df_ktm.groupby(order_col, sort=True)
        if len(g) >= min_obs_per_order
    ]

    delayed_jobs = (
        delayed(_fit_one_order)(
            order_val,
            theta,
            delta,
            mu_degree=mu_degree,
            sigma_degree=sigma_degree,
            sigma_floor=sigma_floor,
            distribution=distribution,
            delta_min=delta_min,
            delta_max=delta_max,
        )
        for order_val, theta, delta in tasks
    )
    try:
        rows: list[dict[str, object]] = Parallel(n_jobs=n_jobs)(delayed_jobs)
    except PermissionError as exc:
        warnings.warn(
            f"Parallel fit failed ({exc}); falling back to single-process execution.",
            RuntimeWarning,
            stacklevel=2,
        )
        rows = [
            _fit_one_order(
                order_val,
                theta,
                delta,
                mu_degree=mu_degree,
                sigma_degree=sigma_degree,
                sigma_floor=sigma_floor,
                distribution=distribution,
                delta_min=delta_min,
                delta_max=delta_max,
            )
            for order_val, theta, delta in tasks
        ]

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("order_sequence").reset_index(drop=True)
    return out
