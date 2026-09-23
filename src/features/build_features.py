"""
Assembles the final 413-feature matrix (X_final / feature_names_final_v2)
that model/train.py and model/predict.py both depend on.

Direct port of notebook cell 13 (X_pos: application + prev + install + pos,
OneHotEncoder fit_transform) and cell 18 (X_final = hstack(X_pos,
X_bureau_final, X_cc_final)) — plus the bureau/cc "merge onto the full
applicant population + missing-flag + fillna(0) + column-select" tail that
bureau.py's and credit_card.py's docstrings explicitly leave for this module.

Two entry points, matching application.py's fit/transform pattern:

  build_training_matrix(...)  — TRAINING. Takes the six raw per-table
      DataFrames (df_fe from application.py, prev_agg/install_agg/pos_agg/
      bureau_agg/cc_agg from their own modules), performs every merge, FITS
      the OneHotEncoder, and returns (X_final, feature_names_final_v2,
      fit_artifacts). fit_artifacts must be pickled — predict.py needs it.

  transform_serve_row(...)    — SERVING. Takes ONE flat dict (application
      fields from application_mapper.py, merged via .update() with
      history_mapper.py's output for returning customers) plus a previously
      pickled fit_artifacts, and returns a single-row matrix in the EXACT
      column order X_final was trained on. Never re-fits anything — any
      column the row dict doesn't have is treated as "missing" the same way
      training treats an applicant with no rows in that source table
      (0 for numeric/bureau/cc, "Missing" for categorical). That equivalence
      is what makes cold-start scoring correct rather than just "not crash."
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from .installments import INSTALL_SELECTED_COLUMNS
from .pos_cash import POS_SELECTED_COLUMNS
from .bureau import BUREAU_AGG_COLUMNS, BUREAU_MISSING_FLAG_SOURCE_COLUMNS, BUREAU_MISSING_FLAG_KEEP
from .credit_card import CC_SELECTED_COLUMNS


@dataclass
class BuildMatrixFitArtifacts:
    """Everything learned at training time that serving must reuse verbatim."""
    numeric_cols: list = field(default_factory=list)             # X_pos numeric block, locked order
    categorical_cols: list = field(default_factory=list)         # X_pos categorical block, locked order
    encoder: Optional[OneHotEncoder] = None                      # fitted on categorical_cols
    bureau_cols_final: list = field(default_factory=list)        # BUREAU_AGG_COLUMNS + kept "_MISSING" flags
    cc_cols_final: list = field(default_factory=list)            # CC_SELECTED_COLUMNS
    feature_names_final_v2: list = field(default_factory=list)   # full 413-length column order


def _bureau_block(base_ids: pd.DataFrame, bureau_agg: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Cell 15's merge-onto-base_ids + missing-flag + fillna(0) + column-select tail."""
    merged = base_ids.merge(bureau_agg, on="SK_ID_CURR", how="left")
    for col in BUREAU_MISSING_FLAG_SOURCE_COLUMNS:
        merged[f"{col}_MISSING"] = merged[col].isna().astype(np.int8)
    merged = merged.fillna(0)
    final_cols = BUREAU_AGG_COLUMNS + [f"{c}_MISSING" for c in BUREAU_MISSING_FLAG_KEEP]
    return merged[final_cols].astype(np.float32), final_cols


def _cc_block(base_ids: pd.DataFrame, cc_agg: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Cell 16's merge-onto-base_ids + fillna(0) + column-select tail."""
    merged = base_ids.merge(cc_agg, on="SK_ID_CURR", how="left")
    merged = merged.fillna(0)
    return merged[CC_SELECTED_COLUMNS].astype(np.float32), CC_SELECTED_COLUMNS


def build_training_matrix(
    df_fe: pd.DataFrame,
    prev_agg: pd.DataFrame,
    install_agg: pd.DataFrame,
    pos_agg: pd.DataFrame,
    bureau_agg: pd.DataFrame,
    cc_agg: pd.DataFrame,
):
    """
    Parameters mirror the notebook's df_fe / prev_agg / install_agg /
    pos_agg / bureau_agg / cc_agg exactly — df_fe from application.py's
    engineer_application_features(fit=True, ...), the rest from their own
    build_*_features() functions.

    Returns
    -------
    (X_final, feature_names_final_v2, fit_artifacts) — X_final is the sparse
    413-column training matrix (scipy.sparse.csr_matrix); fit_artifacts
    must be pickled and reused at serve time.
    """
    # ---- cell 13: X_pos (application + FULL prev_agg + selected install/pos) ----
    df_model = df_fe.merge(prev_agg, left_on="SK_ID_CURR", right_index=True, how="left")
    df_model = df_model.merge(install_agg[INSTALL_SELECTED_COLUMNS], left_on="SK_ID_CURR", right_index=True, how="left")
    df_model = df_model.merge(pos_agg[POS_SELECTED_COLUMNS], left_on="SK_ID_CURR", right_index=True, how="left")

    X_df = df_model.drop(columns=["TARGET", "SK_ID_CURR"])
    categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    X_num = X_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    X_cat = X_df[categorical_cols].fillna("Missing").astype(str)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    X_cat_encoded = encoder.fit_transform(X_cat)

    X_num_sparse = sparse.csr_matrix(X_num.values, dtype=np.float32)
    X_pos = sparse.hstack([X_num_sparse, X_cat_encoded], format="csr", dtype=np.float32)
    feature_names = numeric_cols + encoder.get_feature_names_out(categorical_cols).tolist()

    # ---- cells 15/16 tail: bureau + cc, merged onto the FULL applicant population ----
    base_ids = df_fe[["SK_ID_CURR"]].copy()
    X_bureau_df, bureau_cols_final = _bureau_block(base_ids, bureau_agg)
    X_cc_df, cc_cols_final = _cc_block(base_ids, cc_agg)

    X_bureau_final = sparse.csr_matrix(X_bureau_df.values, dtype=np.float32)
    X_cc_final = sparse.csr_matrix(X_cc_df.values, dtype=np.float32)

    # ---- cell 18: final hstack ----
    X_final = sparse.hstack([X_pos, X_bureau_final, X_cc_final], format="csr")
    feature_names_final_v2 = feature_names + bureau_cols_final + cc_cols_final

    assert X_final.shape[1] == len(feature_names_final_v2), \
        "Feature name / column count mismatch — stop and check."

    fit_artifacts = BuildMatrixFitArtifacts(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        encoder=encoder,
        bureau_cols_final=bureau_cols_final,
        cc_cols_final=cc_cols_final,
        feature_names_final_v2=feature_names_final_v2,
    )
    return X_final, feature_names_final_v2, fit_artifacts


def transform_serve_row(row: dict, fit_artifacts: BuildMatrixFitArtifacts):
    """
    Parameters
    ----------
    row : a single flat dict — application_mapper.py's output, merged (via
        .update()) with history_mapper.py's output for returning customers.
        Any column this function needs that ISN'T a key in `row` is treated
        as missing: 0 for numeric/bureau/cc columns (matching training's
        left-join + fillna(0) for an applicant with no rows in that source
        table), "Missing" for categorical columns (matching application.py's
        own categorical fill value).

    fit_artifacts : the BuildMatrixFitArtifacts pickled by
        build_training_matrix(). Supplies the locked column order and the
        already-fitted encoder — nothing here is ever re-fit.

    Returns
    -------
    A single-row sparse matrix, shape (1, len(feature_names_final_v2)),
    column order identical to X_final — feed straight into the trained
    XGBoost / LightGBM models.
    """
    numeric_vals = [float(row.get(c, 0) or 0) for c in fit_artifacts.numeric_cols]
    X_num = sparse.csr_matrix(np.array([numeric_vals], dtype=np.float32))

    X_cat_df = pd.DataFrame({c: [str(row.get(c, "Missing"))] for c in fit_artifacts.categorical_cols})
    X_cat_encoded = fit_artifacts.encoder.transform(X_cat_df)

    X_pos_row = sparse.hstack([X_num, X_cat_encoded], format="csr", dtype=np.float32)

    bureau_vals = [float(row.get(c, 0) or 0) for c in fit_artifacts.bureau_cols_final]
    X_bureau_row = sparse.csr_matrix(np.array([bureau_vals], dtype=np.float32))

    cc_vals = [float(row.get(c, 0) or 0) for c in fit_artifacts.cc_cols_final]
    X_cc_row = sparse.csr_matrix(np.array([cc_vals], dtype=np.float32))

    X_row_final = sparse.hstack([X_pos_row, X_bureau_row, X_cc_row], format="csr")
    assert X_row_final.shape[1] == len(fit_artifacts.feature_names_final_v2), \
        "Serve-time row width doesn't match training feature count — check fit_artifacts."
    return X_row_final