"""
Feature engineering on application_train / application_test rows.

Mirrors notebook cells 3 and 5, but split into a fit/transform pattern:

  - fit=True   (training):  compute medians, categorical fill value, and
               which numeric columns get a "_MISSING" indicator, based on
               the *training* data. Returns those as `fit_artifacts` so
               they can be pickled and reused at serve time.
  - fit=False  (serving):   apply a previously-fitted `fit_artifacts`
               dict instead of recomputing anything from the row(s) being
               scored. This is required for correctness: a single live
               applicant can't tell you what "the median" or "which
               columns are ever missing" is across the training set.

Both paths return the same set of engineered columns in the same dtypes,
which is what build_matrix.py depends on downstream.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Columns that are engineered before the missing-flag pass runs, so they
# must never get a spurious "_MISSING" indicator column of their own.
_ENGINEERED_BEFORE_MISSING_PASS = [
    "TARGET", "SK_ID_CURR", "AGE_YEARS", "EMPLOYED_YEARS", "EMPLOYMENT_SENTINEL",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_ANNUITY_RATIO", "GOODS_CREDIT_RATIO",
]

_EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

_SECONDARY_FEATURES = [
    "INCOME_AFTER_ANNUITY", "INCOME_PER_FAMILY_MEMBER", "CREDIT_PER_FAMILY_MEMBER",
    "CHILDREN_FAMILY_RATIO", "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX",
    "SOCIAL_CIRCLE_OBS", "SOCIAL_CIRCLE_DEFAULTS",
]


@dataclass
class ApplicationFitArtifacts:
    """Everything learned from the training set that serving must reuse."""
    missing_flag_columns: list = field(default_factory=list)   # numeric cols that get a _MISSING indicator
    numeric_medians: dict = field(default_factory=dict)        # {col: median} for imputation
    categorical_fill_value: str = "Missing"


def _core_ratios_and_age(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 3: age/employment + core financial ratios. No fit/transform split needed —
    these are row-local computations, safe to run identically at train and serve time."""
    df = df.copy()

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # DAYS_EMPLOYED = 365243 is a known sentinel value for "not employed"
    df["EMPLOYMENT_SENTINEL"] = (df["DAYS_EMPLOYED"] == 365243).astype(np.int8)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYED_YEARS"] = -days_employed_clean / 365.25

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"].replace(0, np.nan)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)

    return df


def _secondary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 5: income/family burden, EXT_SOURCE combos, social-circle behavior."""
    df = df.copy()

    df["INCOME_AFTER_ANNUITY"] = df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CREDIT_PER_FAMILY_MEMBER"] = df["AMT_CREDIT"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["CHILDREN_FAMILY_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    df["EXT_SOURCE_MEAN"] = df[_EXT_SOURCE_COLS].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[_EXT_SOURCE_COLS].min(axis=1)
    df["EXT_SOURCE_MAX"] = df[_EXT_SOURCE_COLS].max(axis=1)

    df["SOCIAL_CIRCLE_OBS"] = df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0) + df.get("OBS_60_CNT_SOCIAL_CIRCLE", 0)
    df["SOCIAL_CIRCLE_DEFAULTS"] = df.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) + df.get("DEF_60_CNT_SOCIAL_CIRCLE", 0)

    df[_SECONDARY_FEATURES] = df[_SECONDARY_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def engineer_application_features(
    df: pd.DataFrame,
    fit: bool,
    fit_artifacts: Optional[ApplicationFitArtifacts] = None,
) -> tuple[pd.DataFrame, ApplicationFitArtifacts]:
    """
    Run the full application-level feature pipeline (cells 3 + 5).

    Parameters
    ----------
    df : raw application_train (or a single-row/batch application_test-shaped) DataFrame
    fit : True during training (learns medians / missing-flag columns from `df`),
          False during serving (applies a previously-fitted `fit_artifacts`)
    fit_artifacts : required when fit=False; ignored (and recomputed) when fit=True

    Returns
    -------
    (engineered_df, fit_artifacts)
    fit_artifacts should be pickled after a fit=True call and passed back in
    on every fit=False call at serve time.
    """
    if not fit and fit_artifacts is None:
        raise ValueError("fit_artifacts is required when fit=False (serving mode).")

    df_fe = _core_ratios_and_age(df)
    df_fe = _secondary_features(df_fe)

    df_fe = df_fe.replace([np.inf, -np.inf], np.nan)

    numeric_cols = [c for c in df_fe.select_dtypes(include=[np.number]).columns
                     if c not in ("TARGET", "SK_ID_CURR")]
    categorical_cols = df_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    if fit:
        # Learn missing-flag columns + medians from this (training) data.
        candidate_cols = [c for c in numeric_cols if c not in _ENGINEERED_BEFORE_MISSING_PASS]
        missing_flag_columns = [c for c in candidate_cols if df_fe[c].isna().any()]
        numeric_medians = {c: float(df_fe[c].median()) for c in numeric_cols if df_fe[c].isna().any()}
        fit_artifacts = ApplicationFitArtifacts(
            missing_flag_columns=missing_flag_columns,
            numeric_medians=numeric_medians,
            categorical_fill_value="Missing",
        )
    else:
        missing_flag_columns = fit_artifacts.missing_flag_columns
        numeric_medians = fit_artifacts.numeric_medians

    # Missing indicators: always emit the SAME columns learned at fit time,
    # regardless of whether this particular row happens to be missing them.
    for col in missing_flag_columns:
        df_fe[f"{col}_MISSING"] = df_fe[col].isna().astype(np.int8) if col in df_fe.columns else 1

    # Impute using the FIT-TIME medians (never recompute at serve time).
    for col, median_value in numeric_medians.items():
        if col in df_fe.columns:
            df_fe[col] = df_fe[col].fillna(median_value)

    for col in categorical_cols:
        df_fe[col] = df_fe[col].fillna(fit_artifacts.categorical_fill_value)

    return df_fe, fit_artifacts
