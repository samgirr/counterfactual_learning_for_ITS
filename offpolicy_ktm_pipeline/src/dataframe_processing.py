from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.pix_processing import process_pix
from models.ktm import processing_data


def process_pix_dataset(
    *,
    input_csv: str = "data/pix_data.csv",
    output_dir: str = "pix_mapping",
    output_csv_name: str = "pix_processed.csv",
    irt_csv_name: str = "pix_irt.csv",
    irt_outcome_csv_name: str = "pix_irt_outcome.csv",
) -> dict[str, object]:
    """
    Run PIX preprocessing and persist integer-id mappings + processed CSVs.
    """
    return process_pix(
        input_csv=input_csv,
        output_dir=output_dir,
        output_csv_name=output_csv_name,
        irt_csv_name=irt_csv_name,
        irt_outcome_csv_name=irt_outcome_csv_name,
    )


def build_ktm_dataframe(
    *,
    processed_csv: str = "pix_mapping/pix_processed.csv",
    user_col: str = "user_id",
    item_col: str = "challenge_id",
    correct_col: str = "outcome",
    skill_col: str = "skill_id",
    answer_order_col: str = "answer_number",
    derive_answer_number_per_user: bool = False,
    sample_users: int = 0,
    sample_rows: int = 0,
    seed: int = 42,
    reduce: bool = False,
    top_skills: int = 100,
    item_quantile: float = 0.75,
    min_user_obs: int = 10,
    c: float = 0.1,
    fit_intercept: bool = False,
    target_std: float | None = None,
    clip_quantiles: tuple[float, float] | None = None,
    strict_1pl: bool = True,
) -> pd.DataFrame:
    """
    Build KTM-ready dataframe with proficiency/difficulties and sequence columns.
    strict_1pl=True keeps p(correct)=sigmoid(theta-difficulty) exactly.
    """
    data_path = Path(processed_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed CSV: {data_path}")

    raw_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    usecols = [user_col, item_col, correct_col]
    if skill_col and skill_col in raw_cols:
        usecols.append(skill_col)
    if answer_order_col in raw_cols:
        usecols.append(answer_order_col)

    df = pd.read_csv(data_path, usecols=usecols).dropna(subset=[user_col, item_col, correct_col]).copy()

    # If order is missing (or user requests override), derive it as 1..len(user) per user.
    if derive_answer_number_per_user or (answer_order_col not in df.columns):
        df[answer_order_col] = df.groupby(user_col).cumcount() + 1

    if sample_users > 0 and sample_users < df[user_col].nunique():
        rng = np.random.default_rng(seed)
        users = df[user_col].dropna().unique()
        keep = set(rng.choice(users, size=sample_users, replace=False).tolist())
        df = df[df[user_col].isin(keep)].copy()
    if sample_rows > 0 and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=seed).copy()

    rename_map = {
        user_col: "user",
        item_col: "item",
        correct_col: "correct",
        answer_order_col: "answer_number",
    }
    df = df.rename(columns=rename_map)
    if skill_col and skill_col in df.columns:
        df = df.rename(columns={skill_col: "skill"})
    elif "skill" not in df.columns:
        df["skill"] = "NA"

    # Convert string-encoded correctness ("ok"/"ko", "yes"/"no", etc.) to 0/1.
    if df["correct"].dtype == object:
        _ok = {"ok", "1", "true", "yes", "correct", "1.0"}
        df["correct"] = df["correct"].astype(str).str.lower().isin(_ok).astype("int64")

    df_ktm = processing_data(
        df,
        user_col="user",
        item_col="item",
        correct_col="correct",
        skill_col="skill",
        order_cols=["answer_number"],
        reduce=reduce,
        top_skills=top_skills,
        item_quantile=item_quantile,
        min_user_obs=min_user_obs,
        rank_start=0,
        c=c,
        random_state=seed,
        fit_intercept=fit_intercept,
        center_latents=True,
        target_std=target_std,
        clip_quantiles=clip_quantiles,
        strict_1pl=strict_1pl,
    )
    return df_ktm
