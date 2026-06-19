from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.ktm import processing_data


def _poly_design_np(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.column_stack(cols).astype(np.float32)


def _poly_design_torch(x: torch.Tensor, degree: int) -> torch.Tensor:
    cols = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return torch.stack(cols, dim=1)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _init_params(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    k: int,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_mu = _poly_design_np(theta, mu_degree)
    p = x_mu.shape[1]
    q = sigma_degree + 1

    beta_mu = np.zeros((k, p), dtype=np.float32)
    beta_sig = np.zeros((k, q), dtype=np.float32)
    logits_pi = np.zeros(k, dtype=np.float32)

    qs = np.quantile(delta, np.linspace(0.0, 1.0, k + 1))
    comp = np.digitize(delta, qs[1:-1], right=True)
    if len(np.unique(comp)) < k:
        comp = rng.integers(0, k, size=len(delta))

    for j in range(k):
        mask = comp == j
        if int(mask.sum()) < p:
            beta = np.linalg.lstsq(x_mu, delta, rcond=None)[0]
            resid = delta - x_mu @ beta
            nj = max(int(mask.sum()), 1)
        else:
            beta = np.linalg.lstsq(x_mu[mask], delta[mask], rcond=None)[0]
            resid = delta[mask] - x_mu[mask] @ beta
            nj = int(mask.sum())

        beta_mu[j] = beta.astype(np.float32)
        s0 = max(float(np.std(resid)), sigma_floor)
        beta_sig[j, 0] = float(np.log(s0))
        logits_pi[j] = float(np.log(nj))

    return beta_mu, beta_sig, logits_pi


def _fit_trunc_mixture_torch(
    theta: np.ndarray,
    delta: np.ndarray,
    *,
    k: int,
    mu_degree: int,
    sigma_degree: int,
    sigma_floor: float,
    lr: float,
    max_epochs: int,
    tol: float,
    patience: int,
    l2_coef: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    th = torch.from_numpy(theta.astype(np.float32)).to(device)
    de = torch.from_numpy(delta.astype(np.float32)).to(device)
    x_mu = _poly_design_torch(th, mu_degree)
    x_sig = _poly_design_torch(th, sigma_degree)
    a = float(delta.min())
    b = float(delta.max())

    bmu0, bsig0, lpi0 = _init_params(
        theta,
        delta,
        k=k,
        mu_degree=mu_degree,
        sigma_degree=sigma_degree,
        sigma_floor=sigma_floor,
        seed=seed,
    )

    beta_mu = torch.nn.Parameter(torch.from_numpy(bmu0).to(device))
    beta_sig = torch.nn.Parameter(torch.from_numpy(bsig0).to(device))
    logits_pi = torch.nn.Parameter(torch.from_numpy(lpi0).to(device))

    optimizer = torch.optim.Adam([beta_mu, beta_sig, logits_pi], lr=lr)

    best = float("inf")
    best_state = None
    bad = 0
    log2pi = math.log(2.0 * math.pi)

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)

        mu = x_mu @ beta_mu.T
        sigma = torch.exp(x_sig @ beta_sig.T).clamp_min(float(sigma_floor))

        z = (de[:, None] - mu) / sigma
        alpha = (float(a) - mu) / sigma
        beta = (float(b) - mu) / sigma

        log_phi = -0.5 * log2pi - torch.log(sigma) - 0.5 * (z ** 2)
        z_norm = (_normal_cdf(beta) - _normal_cdf(alpha)).clamp_min(1e-12)
        log_pdf = log_phi - torch.log(z_norm)

        log_pi = F.log_softmax(logits_pi, dim=0)[None, :]
        log_mix = torch.logsumexp(log_pi + log_pdf, dim=1)
        nll = -torch.mean(log_mix)

        reg = float(l2_coef) * (torch.mean(beta_mu ** 2) + torch.mean(beta_sig ** 2))
        loss = nll + reg
        loss.backward()
        optimizer.step()

        cur = float(nll.detach().cpu().item())
        if cur + tol < best:
            best = cur
            bad = 0
            best_state = (
                beta_mu.detach().cpu().numpy().copy(),
                beta_sig.detach().cpu().numpy().copy(),
                logits_pi.detach().cpu().numpy().copy(),
                epoch,
            )
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        bmu, bsig, lpi, best_epoch = best_state
    else:
        bmu = beta_mu.detach().cpu().numpy()
        bsig = beta_sig.detach().cpu().numpy()
        lpi = logits_pi.detach().cpu().numpy()
        best_epoch = max_epochs

    pi = np.exp(lpi - np.max(lpi))
    pi = pi / np.clip(pi.sum(), 1e-12, None)

    return {
        "nll": float(best),
        "beta_mu": bmu,
        "beta_sig": bsig,
        "pi": pi,
        "best_epoch": int(best_epoch),
    }


def _load_dataset(name: str, path: str) -> pd.DataFrame:
    lname = name.lower()
    if lname in {"assistment8000", "assistments8000", "assistment100", "assistments100"}:
        df = pd.read_csv(path, usecols=["user", "item", "correct"]).copy()
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df

    if lname in {"skillbuilder", "skill_builder_2015", "skillbuilder2015"}:
        df = pd.read_csv(path, usecols=["user_id", "sequence_id", "correct"]).copy()
        df = df.rename(columns={"user_id": "user", "sequence_id": "item"})
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df

    if lname == "attempts":
        df = pd.read_csv(path, usecols=["student", "problem", "solved"]).copy()
        df = df.rename(columns={"student": "user", "problem": "item", "solved": "correct"})
        df["correct"] = df["correct"].astype(int)
        df["answer_number"] = df.groupby("user").cumcount() + 1
        df["skill"] = "NA"
        return df

    if lname in {"pix_data", "pix"}:
        df = pd.read_csv(path, usecols=["user_id", "challenge_id", "answer_result", "answer_number"]).copy()
        amap = {"ok": 1.0, "ko": 0.0, "aband": 0.0}
        df["correct"] = df["answer_result"].map(amap).fillna(0.0)
        df = df.rename(columns={"user_id": "user", "challenge_id": "item"})
        df["skill"] = "NA"
        return df[["user", "item", "correct", "answer_number", "skill"]]

    raise ValueError(f"Unknown dataset name: {name}")


def _build_ktm(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return processing_data(
        df.copy(),
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=False,
        rank_start=0,
        c=0.1,
        random_state=seed,
        fit_intercept=True,
        center_latents=True,
        target_std=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare truncated Gaussian vs truncated GMM(K=3) NLL on datasets.")
    parser.add_argument("--out-csv", type=str, default="pix_mapping/truncated_gmm3_nll_comparison.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap per dataset (0 means all rows).")
    parser.add_argument("--mu-degree", type=int, default=1)
    parser.add_argument("--sigma-degree", type=int, default=2)
    parser.add_argument("--sigma-floor", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--l2-coef", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    datasets = [
        ("assistment8000", os.path.join(PROJECT_ROOT, "data", "data_top8000_items.csv")),
        ("skill_builder_2015", os.path.join(PROJECT_ROOT, "data", "2015_100_skill_builders_main_problems.csv")),
        ("attempts", os.path.join(PROJECT_ROOT, "data", "attempts.csv")),
        ("pix_data", os.path.join(PROJECT_ROOT, "data", "pix_data.csv")),
    ]

    rows: list[dict[str, object]] = []
    for name, path in datasets:
        t0 = time.time()
        raw = _load_dataset(name, path)
        if args.max_rows > 0 and len(raw) > args.max_rows:
            raw = raw.sample(n=int(args.max_rows), random_state=args.seed).sort_index()
        ktm = _build_ktm(raw, seed=args.seed)

        theta = ktm["proficiency"].to_numpy(dtype=np.float32)
        delta = ktm["difficulties"].to_numpy(dtype=np.float32)

        one = _fit_trunc_mixture_torch(
            theta,
            delta,
            k=1,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            lr=args.lr,
            max_epochs=args.max_epochs,
            tol=args.tol,
            patience=args.patience,
            l2_coef=args.l2_coef,
            seed=args.seed,
            device=args.device,
        )
        mix = _fit_trunc_mixture_torch(
            theta,
            delta,
            k=args.k,
            mu_degree=args.mu_degree,
            sigma_degree=args.sigma_degree,
            sigma_floor=args.sigma_floor,
            lr=args.lr,
            max_epochs=args.max_epochs,
            tol=args.tol,
            patience=args.patience,
            l2_coef=args.l2_coef,
            seed=args.seed,
            device=args.device,
        )
        elapsed = time.time() - t0

        rows.append(
            {
                "dataset": name,
                "n_rows_raw": int(len(raw)),
                "n_rows_ktm": int(len(ktm)),
                "delta_min": float(delta.min()),
                "delta_max": float(delta.max()),
                "trunc_gaussian_nll": float(one["nll"]),
                "trunc_gmm3_nll": float(mix["nll"]),
                "nll_gain_gmm3_minus_single": float(mix["nll"] - one["nll"]),
                "single_best_epoch": int(one["best_epoch"]),
                "gmm3_best_epoch": int(mix["best_epoch"]),
                "runtime_sec": float(elapsed),
            }
        )
        print(
            f"{name}: n={len(ktm)} single_nll={one['nll']:.6f} "
            f"gmm3_nll={mix['nll']:.6f} gain={mix['nll'] - one['nll']:.6f}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved={os.path.abspath(args.out_csv)}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
