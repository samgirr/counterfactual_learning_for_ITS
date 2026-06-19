from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def poly_design(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_softplus(y: float) -> float:
    if y > 20:
        return float(y)
    return float(np.log(np.expm1(y)))


@dataclass
class DistFit:
    distribution: str
    nll_mean: float
    nll_total: float
    rmse_mu: float
    mae_mu: float
    success: bool
    message: str
    dof: float | None
    beta_mu: np.ndarray
    beta_scale: np.ndarray


def _negative_log_likelihood(
    params: np.ndarray,
    x_mu: np.ndarray,
    x_s: np.ndarray,
    y: np.ndarray,
    distribution: str,
    sigma_floor: float,
    free_student_t_dof: bool,
) -> float:
    k_mu = x_mu.shape[1]
    k_s = x_s.shape[1]
    beta_mu = params[:k_mu]
    beta_s = params[k_mu : k_mu + k_s]
    mu = x_mu @ beta_mu
    scale = np.maximum(np.exp(x_s @ beta_s), sigma_floor)
    z = (y - mu) / scale

    if distribution == "normal":
        nll = 0.5 * np.log(2.0 * np.pi) + np.log(scale) + 0.5 * (z ** 2)
        return float(np.mean(nll))
    if distribution == "laplace":
        nll = np.log(2.0 * scale) + np.abs(z)
        return float(np.mean(nll))
    if distribution == "logistic":
        # log f = -log(s) - z - 2*softplus(-z)
        nll = np.log(scale) + z + 2.0 * _softplus(-z)
        return float(np.mean(nll))
    if distribution == "student_t":
        if free_student_t_dof:
            raw_nu = params[k_mu + k_s]
            nu = 2.0 + float(_softplus(np.array([raw_nu]))[0])
        else:
            nu = 5.0
        log_pdf = (
            gammaln((nu + 1.0) * 0.5)
            - gammaln(nu * 0.5)
            - 0.5 * np.log(nu * np.pi)
            - np.log(scale)
            - ((nu + 1.0) * 0.5) * np.log1p((z ** 2) / nu)
        )
        return float(-np.mean(log_pdf))
    raise ValueError(f"Unknown distribution: {distribution}")


def fit_distribution(
    *,
    theta: np.ndarray,
    delta: np.ndarray,
    mu_degree: int,
    sigma_degree: int,
    distribution: str,
    sigma_floor: float,
    max_iter: int,
    free_student_t_dof: bool,
) -> DistFit:
    x_mu = poly_design(theta, mu_degree)
    x_s = poly_design(theta, sigma_degree)
    k_mu = x_mu.shape[1]
    k_s = x_s.shape[1]

    beta_mu0 = np.linalg.lstsq(x_mu, delta, rcond=None)[0]
    resid = delta - x_mu @ beta_mu0
    beta_s0 = np.zeros(k_s, dtype=float)
    beta_s0[0] = float(np.log(np.std(resid) + 1e-3))

    if distribution == "student_t" and free_student_t_dof:
        p0 = np.concatenate([beta_mu0, beta_s0, np.array([_inv_softplus(3.0)], dtype=float)])
    else:
        p0 = np.concatenate([beta_mu0, beta_s0])

    out = minimize(
        _negative_log_likelihood,
        p0,
        args=(
            x_mu,
            x_s,
            delta,
            distribution,
            sigma_floor,
            free_student_t_dof,
        ),
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )
    p = out.x if out.success else p0
    beta_mu = p[:k_mu]
    beta_s = p[k_mu : k_mu + k_s]

    mu_hat = x_mu @ beta_mu
    rmse_mu = float(np.sqrt(np.mean((delta - mu_hat) ** 2)))
    mae_mu = float(np.mean(np.abs(delta - mu_hat)))
    nll_mean = _negative_log_likelihood(
        p,
        x_mu,
        x_s,
        delta,
        distribution,
        sigma_floor,
        free_student_t_dof,
    )

    dof = None
    if distribution == "student_t":
        if free_student_t_dof:
            dof = float(2.0 + _softplus(np.array([p[k_mu + k_s]]))[0])
        else:
            dof = 5.0

    return DistFit(
        distribution=distribution,
        nll_mean=float(nll_mean),
        nll_total=float(nll_mean * len(delta)),
        rmse_mu=rmse_mu,
        mae_mu=mae_mu,
        success=bool(out.success),
        message=str(out.message),
        dof=dof,
        beta_mu=beta_mu,
        beta_scale=beta_s,
    )


def compare_distributions(
    *,
    theta: np.ndarray,
    delta: np.ndarray,
    mu_degree: int = 1,
    sigma_degree: int = 2,
    sigma_floor: float = 1e-4,
    max_iter: int = 300,
    free_student_t_dof: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    distributions = ["normal", "laplace", "logistic", "student_t"]
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
            "distribution": fit.distribution,
            "n_obs": int(len(theta)),
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
    out = pd.DataFrame(rows).sort_values("nll_mean").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare probabilistic regression distributions by negative log-likelihood."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="pix_mapping/ktm_gaussian_propensity_order_u10k_mu1_sigma2.csv",
    )
    parser.add_argument("--theta-col", type=str, default="proficiency")
    parser.add_argument("--delta-col", type=str, default="difficulties")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--fixed-student-t-dof", action="store_true")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="pix_mapping/probabilistic_distribution_nll_comparison.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data, usecols=[args.theta_col, args.delta_col]).dropna().copy()
    if args.sample_rows > 0 and args.sample_rows < len(df):
        df = df.sample(n=args.sample_rows, random_state=args.seed).copy()

    theta = df[args.theta_col].to_numpy(dtype=float)
    delta = df[args.delta_col].to_numpy(dtype=float)

    comp = compare_distributions(
        theta=theta,
        delta=delta,
        mu_degree=args.mu_degree,
        sigma_degree=args.sigma_degree,
        sigma_floor=args.sigma_floor,
        max_iter=args.max_iter,
        free_student_t_dof=not args.fixed_student_t_dof,
    )
    comp.to_csv(args.out_csv, index=False)

    best = comp.iloc[0]
    print(f"n_obs={len(theta)}")
    print(f"best_distribution={best['distribution']} nll_mean={best['nll_mean']:.6f}")
    print(f"saved_csv={args.out_csv}")


if __name__ == "__main__":
    main()
