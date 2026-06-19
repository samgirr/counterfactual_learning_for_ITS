from __future__ import annotations

import math

import pandas as pd
import torch
import torch.nn as nn


def poly_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    cols = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return torch.stack(cols, dim=1)


class GaussianOrderPolicy(nn.Module):
    """
    Gaussian policy over continuous action delta, conditioned on (theta, order):
      pi(delta | theta, order) = N(mu_order(theta), sigma_order(theta)^2)

    Parameters are exactly polynomial coefficients for:
      mu_order(theta)
      log_sigma_order(theta)
    """

    def __init__(
        self,
        n_orders: int,
        mu_degree: int,
        sigma_degree: int,
        sigma_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_orders = int(n_orders)
        self.mu_degree = int(mu_degree)
        self.sigma_degree = int(sigma_degree)
        self.sigma_floor = float(sigma_floor)

        self.beta_mu = nn.Parameter(torch.zeros(self.n_orders, self.mu_degree + 1))
        self.beta_sigma = nn.Parameter(torch.zeros(self.n_orders, self.sigma_degree + 1))

    def init_from_fit_csv(
        self,
        fit_df: pd.DataFrame,
        order_to_idx: dict[int, int],
    ) -> None:
        """
        Initialize policy coefficients from existing Gaussian regression fits.
        Supports rows keyed either by:
        - order_sequence
        - order_min/order_max (uses order_min if min==max)
        """
        with torch.no_grad():
            for _, row in fit_df.iterrows():
                if "order_sequence" in fit_df.columns:
                    order_val = int(row["order_sequence"])
                elif "order_min" in fit_df.columns and "order_max" in fit_df.columns:
                    if int(row["order_min"]) != int(row["order_max"]):
                        continue
                    order_val = int(row["order_min"])
                else:
                    continue

                if order_val not in order_to_idx:
                    continue
                idx = order_to_idx[order_val]

                for d in range(self.mu_degree + 1):
                    c = f"beta_mu_{d}"
                    if c in row:
                        self.beta_mu[idx, d] = float(row[c])
                for d in range(self.sigma_degree + 1):
                    c = f"beta_sigma_{d}"
                    if c in row:
                        self.beta_sigma[idx, d] = float(row[c])

    def mu_sigma(
        self,
        theta: torch.Tensor,
        order_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_mu = poly_basis(theta, self.mu_degree)
        x_sigma = poly_basis(theta, self.sigma_degree)
        beta_mu = self.beta_mu[order_idx]
        beta_sigma = self.beta_sigma[order_idx]
        mu = (x_mu * beta_mu).sum(dim=1)
        log_sigma = (x_sigma * beta_sigma).sum(dim=1)
        sigma = torch.exp(log_sigma).clamp_min(self.sigma_floor)
        return mu, sigma

    def log_prob(
        self,
        delta: torch.Tensor,
        theta: torch.Tensor,
        order_idx: torch.Tensor,
    ) -> torch.Tensor:
        mu, sigma = self.mu_sigma(theta=theta, order_idx=order_idx)
        z = (delta - mu) / sigma
        return -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (z ** 2)


class GlobalGaussianPolicy(nn.Module):
    """
    Single global Gaussian policy (no order conditioning).

    If delta_min/delta_max are provided, this class implements a truncated
    Gaussian density on [delta_min, delta_max], and constrains mu(theta) to
    stay in-range via a sigmoid map.
    """

    def __init__(
        self,
        mu_degree: int,
        sigma_degree: int,
        sigma_floor: float = 1e-4,
        delta_min: float | None = None,
        delta_max: float | None = None,
        bound_mean_to_action_range: bool = True,
    ) -> None:
        super().__init__()
        self.mu_degree = int(mu_degree)
        self.sigma_degree = int(sigma_degree)
        self.sigma_floor = float(sigma_floor)
        self.delta_min = float(delta_min) if delta_min is not None else None
        self.delta_max = float(delta_max) if delta_max is not None else None
        self.bound_mean_to_action_range = bool(bound_mean_to_action_range)
        self._use_truncation = (
            self.delta_min is not None
            and self.delta_max is not None
            and float(self.delta_max) > float(self.delta_min)
        )

        self.beta_mu = nn.Parameter(torch.zeros(self.mu_degree + 1))
        self.beta_sigma = nn.Parameter(torch.zeros(self.sigma_degree + 1))

    @torch.no_grad()
    def init_from_behavior_row(self, row: pd.Series) -> None:
        for d in range(self.mu_degree + 1):
            c = f"beta_mu_{d}"
            if c in row:
                self.beta_mu[d] = float(row[c])
        for d in range(self.sigma_degree + 1):
            c = f"beta_sigma_{d}"
            if c in row:
                self.beta_sigma[d] = float(row[c])

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _normal_logcdf(x: torch.Tensor) -> torch.Tensor:
        return torch.log((0.5 * torch.erfc(-x / math.sqrt(2.0))).clamp_min(1e-300))

    @staticmethod
    def _normal_logsf(x: torch.Tensor) -> torch.Tensor:
        return torch.log((0.5 * torch.erfc(x / math.sqrt(2.0))).clamp_min(1e-300))

    @staticmethod
    def _log_diff_exp(log_x: torch.Tensor, log_y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        out = torch.full_like(log_x, float(math.log(eps)))
        mask = log_x > log_y
        out[mask] = log_x[mask] + torch.log1p(-torch.exp(log_y[mask] - log_x[mask]))
        return out

    def _bound_mu(self, mu_raw: torch.Tensor) -> torch.Tensor:
        if not self._use_truncation or not self.bound_mean_to_action_range:
            return mu_raw
        width = float(self.delta_max - self.delta_min)
        return float(self.delta_min) + width * torch.sigmoid(mu_raw)

    def mu_sigma(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_mu = poly_basis(theta, self.mu_degree)
        x_sigma = poly_basis(theta, self.sigma_degree)
        mu_raw = (x_mu * self.beta_mu[None, :]).sum(dim=1)
        log_sigma = (x_sigma * self.beta_sigma[None, :]).sum(dim=1)
        sigma = torch.exp(log_sigma).clamp_min(self.sigma_floor)
        mu = self._bound_mu(mu_raw)
        return mu, sigma

    def log_prob(self, delta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.mu_sigma(theta=theta)
        z = (delta - mu) / sigma
        log_pdf = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - 0.5 * (z ** 2)
        if not self._use_truncation:
            return log_pdf

        a = float(self.delta_min)
        b = float(self.delta_max)
        # Do the truncation normalization in float64 and switch to the survival
        # function in the far upper tail to avoid catastrophic cancellation when
        # the fitted mean sits far outside the action range.
        mu64 = mu.to(torch.float64)
        sigma64 = sigma.to(torch.float64)
        delta64 = delta.to(torch.float64)
        alpha = (a - mu64) / sigma64
        beta = (b - mu64) / sigma64
        log_cdf_beta = self._normal_logcdf(beta)
        log_cdf_alpha = self._normal_logcdf(alpha)
        log_sf_alpha = self._normal_logsf(alpha)
        log_sf_beta = self._normal_logsf(beta)
        log_z_cdf = self._log_diff_exp(log_cdf_beta, log_cdf_alpha, eps=1e-300)
        log_z_sf = self._log_diff_exp(log_sf_alpha, log_sf_beta, eps=1e-300)
        log_z = torch.where(alpha > 0.0, log_z_sf, log_z_cdf)
        z64 = (delta64 - mu64) / sigma64
        log_pdf64 = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma64) - 0.5 * (z64 ** 2)
        log_p = log_pdf64 - log_z
        # Use a small tolerance so that boundary values that are exactly at
        # delta_min/delta_max in float64 but land just outside after float32
        # round-trip are still treated as in-support.
        eps_support = 1e-5
        in_support = (delta64 >= a - eps_support) & (delta64 <= b + eps_support)
        out = torch.where(in_support, log_p, torch.full_like(log_p, float(math.log(1e-12))))
        return out.to(log_pdf.dtype)
