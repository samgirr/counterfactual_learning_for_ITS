from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class KTMFitResult:
    """
    Container for fitted latent parameters from a user-item logistic model.
    """

    user_theta: pd.Series
    item_difficulty: pd.Series
    intercept: float
    model: Pipeline


def _to_binary(series: pd.Series, threshold: float = 0.5) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return (x >= threshold).astype("int64")


def _fit_user_item_logit(
    df: pd.DataFrame,
    *,
    user_col: str = "user",
    item_col: str = "item",
    correct_col: str = "correct",
    c: float = 0.1,
    random_state: int = 42,
    max_iter: int = 1000,
    fit_intercept: bool = True,
) -> KTMFitResult:
    """
    Fit a regularized logistic model on one-hot(user, item).
    """
    pipe = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    C=c,
                    random_state=random_state,
                    max_iter=max_iter,
                    fit_intercept=fit_intercept,
                ),
            ),
        ]
    )
    y = _to_binary(df[correct_col])
    pipe.fit(df[[user_col, item_col]], y)

    ohe: OneHotEncoder = pipe.named_steps["onehot"]
    lr: LogisticRegression = pipe.named_steps["lr"]
    coef = lr.coef_.ravel()

    user_ids = pd.Index(ohe.categories_[0], name=user_col)
    item_ids = pd.Index(ohe.categories_[1], name=item_col)

    n_users = len(user_ids)
    n_items = len(item_ids)
    theta = pd.Series(coef[:n_users], index=user_ids, name="theta")
    difficulty = pd.Series(-coef[n_users : n_users + n_items], index=item_ids, name="difficulty")
    intercept_arr = np.asarray(lr.intercept_).reshape(-1)
    intercept = float(intercept_arr[0]) if intercept_arr.size else 0.0

    return KTMFitResult(
        user_theta=theta,
        item_difficulty=difficulty,
        intercept=intercept,
        model=pipe,
    )


def _stabilize_latents(
    theta: pd.Series,
    difficulty: pd.Series,
    *,
    center: bool = True,
    target_std: float | None = None,
    clip_quantiles: tuple[float, float] | None = None,
    preserve_1pl: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """
    Post-fit latent stabilization.

    When preserve_1pl=True, transformations are restricted so that theta-delta
    remains unchanged exactly, which keeps p(correct)=sigmoid(theta-delta)
    identical to the fitted no-intercept logistic model.
    """
    th = theta.astype(float).copy()
    diff = difficulty.astype(float).copy()

    if preserve_1pl:
        if target_std is not None:
            raise ValueError("target_std is incompatible with preserve_1pl=True (would change theta-delta scale).")
        if clip_quantiles is not None:
            raise ValueError("clip_quantiles is incompatible with preserve_1pl=True (would change theta-delta).")
        if center:
            # Gauge shift: subtract the same constant from theta and delta.
            # This preserves theta-delta exactly.
            shift = 0.5 * (float(th.mean()) + float(diff.mean()))
            th = th - shift
            diff = diff - shift
        return th, diff

    if center:
        th = th - float(th.mean())
        diff = diff - float(diff.mean())

    if target_std is not None:
        if target_std <= 0:
            raise ValueError("target_std must be > 0 when provided")
        th_std = float(th.std(ddof=0))
        diff_std = float(diff.std(ddof=0))
        if th_std > 0:
            th = th * (float(target_std) / th_std)
        if diff_std > 0:
            diff = diff * (float(target_std) / diff_std)

    if clip_quantiles is not None:
        q_low, q_high = clip_quantiles
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("clip_quantiles must satisfy 0 <= q_low < q_high <= 1")
        th_lo, th_hi = th.quantile([q_low, q_high])
        d_lo, d_hi = diff.quantile([q_low, q_high])
        th = th.clip(lower=float(th_lo), upper=float(th_hi))
        diff = diff.clip(lower=float(d_lo), upper=float(d_hi))

    return th, diff


def _rank_series(series: pd.Series, rank_start: int = 0) -> tuple[dict[Any, int], dict[int, float]]:
    sorted_s = series.sort_values()
    rank = {idx: i + rank_start for i, idx in enumerate(sorted_s.index)}
    value_by_rank = {rank[idx]: float(val) for idx, val in sorted_s.items()}
    return rank, value_by_rank


def _apply_sequence_columns(
    df: pd.DataFrame,
    *,
    user_col: str = "user",
    order_cols: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    sort_cols = [user_col] + (order_cols or [])
    if order_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    out["sequence_position"] = out.groupby(user_col).cumcount()
    out["sequence_length"] = out.groupby(user_col)[user_col].transform("count")
    out["order_sequence"] = out["sequence_position"]
    return out


def processing_data(
    df: pd.DataFrame,
    *,
    user_col: str = "user",
    item_col: str = "item",
    correct_col: str = "correct",
    skill_col: str = "skill",
    order_cols: list[str] | None = None,
    reduce: bool = False,
    top_skills: int = 100,
    item_quantile: float = 0.75,
    min_user_obs: int = 10,
    rank_start: int = 0,
    c: float = 0.1,
    random_state: int = 42,
    fit_intercept: bool = True,
    center_latents: bool = True,
    target_std: float | None = None,
    clip_quantiles: tuple[float, float] | None = None,
    strict_1pl: bool = False,
) -> pd.DataFrame:
    """
    KTM-style preprocessing:
    1) optional frequency-based reduction
    2) fit one-hot user-item logistic model
    3) derive user proficiency and item difficulty
    4) rank user/item ids by latent values
    5) add sequence position/length features

    If strict_1pl=True, enforce fit_intercept=False and only apply shared
    shifts that preserve theta-difficulty, so p(correct)=sigmoid(theta-difficulty)
    remains exactly consistent with the fitted model.
    """
    data = df.copy()

    needed = {user_col, item_col, correct_col}
    missing = sorted(needed - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if reduce:
        if skill_col in data.columns:
            top = data[skill_col].value_counts().head(top_skills).index
            data = data[data[skill_col].isin(top)]

        item_counts = data[item_col].value_counts()
        item_thr = float(item_counts.quantile(item_quantile))
        keep_items = item_counts[item_counts >= item_thr].index
        data = data[data[item_col].isin(keep_items)]

        user_counts = data[user_col].value_counts()
        keep_users = user_counts[user_counts >= min_user_obs].index
        data = data[data[user_col].isin(keep_users)]

    data = data.dropna(subset=[user_col, item_col, correct_col]).copy()
    data[correct_col] = _to_binary(data[correct_col])

    if strict_1pl and fit_intercept:
        raise ValueError("strict_1pl=True requires fit_intercept=False to keep p= sigmoid(theta-delta).")

    fit = _fit_user_item_logit(
        data,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        c=c,
        random_state=random_state,
        fit_intercept=fit_intercept,
    )
    theta, diff = _stabilize_latents(
        fit.user_theta,
        fit.item_difficulty,
        center=center_latents,
        target_std=target_std,
        clip_quantiles=clip_quantiles,
        preserve_1pl=strict_1pl,
    )

    user_rank, prof_lookup = _rank_series(theta, rank_start=rank_start)
    item_rank, diff_lookup = _rank_series(diff, rank_start=rank_start)

    data[user_col] = data[user_col].map(user_rank)
    data[item_col] = data[item_col].map(item_rank)
    data["proficiency"] = data[user_col].map(prof_lookup)
    data["difficulties"] = data[item_col].map(diff_lookup)

    data = _apply_sequence_columns(data, user_col=user_col, order_cols=order_cols)
    return data


def processing_data_2(
    df: pd.DataFrame,
    *,
    user_col: str = "user",
    item_col: str = "item",
    correct_col: str = "correct",
    skill_col: str = "skill",
    order_cols: list[str] | None = None,
    rank_start: int = 0,
    c: float = 0.1,
    random_state: int = 42,
    min_users_per_skill: int = 2,
    min_items_per_skill: int = 2,
    fit_intercept: bool = True,
    center_latents: bool = True,
    target_std: float | None = None,
    clip_quantiles: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    Extended KTM preprocessing with per-skill local latent estimates.

    Adds:
    - proficiency_pt: user proficiency estimated from local model within each skill
    - difficulty_pt: item difficulty estimated from local model within each skill
    """
    data = df.copy()
    needed = {user_col, item_col, correct_col, skill_col}
    missing = sorted(needed - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(subset=[user_col, item_col, correct_col, skill_col]).copy()
    data[correct_col] = _to_binary(data[correct_col])

    raw_user = data[user_col].copy()
    raw_item = data[item_col].copy()

    global_fit = _fit_user_item_logit(
        data,
        user_col=user_col,
        item_col=item_col,
        correct_col=correct_col,
        c=c,
        random_state=random_state,
        fit_intercept=fit_intercept,
    )
    global_theta, global_diff = _stabilize_latents(
        global_fit.user_theta,
        global_fit.item_difficulty,
        center=center_latents,
        target_std=target_std,
        clip_quantiles=clip_quantiles,
    )

    theta_std = float(global_theta.std(ddof=0)) if len(global_theta) > 1 else 1.0
    diff_std = float(global_diff.std(ddof=0)) if len(global_diff) > 1 else 1.0
    if theta_std == 0.0:
        theta_std = 1.0
    if diff_std == 0.0:
        diff_std = 1.0

    prof_rows = []
    diff_rows = []
    for skill_val, g in data.groupby(skill_col, sort=False):
        if g[user_col].nunique() < min_users_per_skill or g[item_col].nunique() < min_items_per_skill:
            continue
        try:
            local_fit = _fit_user_item_logit(
                g,
                user_col=user_col,
                item_col=item_col,
                correct_col=correct_col,
                c=c,
                random_state=random_state,
                fit_intercept=fit_intercept,
            )
        except Exception:
            continue

        theta_loc = local_fit.user_theta.copy()
        diff_loc = local_fit.item_difficulty.copy()

        t_std = float(theta_loc.std(ddof=0))
        d_std = float(diff_loc.std(ddof=0))
        if t_std > 0:
            theta_loc = (theta_loc - theta_loc.mean()) * (theta_std / t_std)
        else:
            theta_loc = theta_loc - theta_loc.mean()
        if d_std > 0:
            diff_loc = (diff_loc - diff_loc.mean()) * (diff_std / d_std)
        else:
            diff_loc = diff_loc - diff_loc.mean()

        theta_loc = theta_loc - theta_loc.mean()
        diff_loc = diff_loc - diff_loc.mean()

        prof_rows.append(
            pd.DataFrame(
                {
                    skill_col: skill_val,
                    user_col: theta_loc.index,
                    "proficiency_pt": theta_loc.values,
                }
            )
        )
        diff_rows.append(
            pd.DataFrame(
                {
                    skill_col: skill_val,
                    item_col: diff_loc.index,
                    "difficulty_pt": diff_loc.values,
                }
            )
        )

    user_rank, prof_lookup = _rank_series(global_theta, rank_start=rank_start)
    item_rank, diff_lookup = _rank_series(global_diff, rank_start=rank_start)
    data[user_col] = data[user_col].map(user_rank)
    data[item_col] = data[item_col].map(item_rank)
    data["proficiency"] = data[user_col].map(prof_lookup)
    data["difficulties"] = data[item_col].map(diff_lookup)

    data["user_raw"] = raw_user.values
    data["item_raw"] = raw_item.values

    if prof_rows:
        prof_df = pd.concat(prof_rows, ignore_index=True)
        data = data.merge(
            prof_df,
            how="left",
            left_on=[skill_col, "user_raw"],
            right_on=[skill_col, user_col],
            suffixes=("", "_tmp_user"),
        )
        if f"{user_col}_tmp_user" in data.columns:
            data = data.drop(columns=[f"{user_col}_tmp_user"])
    else:
        data["proficiency_pt"] = np.nan

    if diff_rows:
        diff_df = pd.concat(diff_rows, ignore_index=True)
        data = data.merge(
            diff_df,
            how="left",
            left_on=[skill_col, "item_raw"],
            right_on=[skill_col, item_col],
            suffixes=("", "_tmp_item"),
        )
        if f"{item_col}_tmp_item" in data.columns:
            data = data.drop(columns=[f"{item_col}_tmp_item"])
    else:
        data["difficulty_pt"] = np.nan

    data = _apply_sequence_columns(data, user_col=user_col, order_cols=order_cols)
    return data
