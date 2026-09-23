"""
Feature engineering on credit_card_balance.csv, aggregated to one row per
SK_ID_CURR.

Direct port of notebook cell 16's aggregation logic. Stops at cc_agg,
matching bureau.py / previous_app.py / pos_cash.py's pattern — the
notebook's merge-onto-base_ids + missing-flag + column-drop steps happen
AFTER a left-join against the FULL applicant population (~66% of
applicants have no credit card at all, per the notebook's own comment),
so that belongs in build_matrix.py.

Unlike bureau.py, this one is simple: ALL 3 missing-flag columns the
notebook generates get dropped again in the same cell (cc_drop), so
build_matrix.py does not need to keep any "_MISSING" columns for this
block — just left-join, fillna(0), and select CC_SELECTED_COLUMNS.
CC_UNIQUE_PREV is also dropped (present in cc_agg but excluded from the
final feature set), which is why it's not in CC_SELECTED_COLUMNS below
even though build_cc_features() still returns it.

build_matrix.py's job, in order, once it has cc_agg from this module:
  1. left-join cc_agg onto the full df_fe population on SK_ID_CURR
  2. .fillna(0) on the merged frame
  3. keep CC_SELECTED_COLUMNS as the final credit-card feature block
     (13 columns total)
"""
import numpy as np
import pandas as pd

# Final 13 columns that survive the notebook's cc_drop step (CC_UNIQUE_PREV
# and all 3 "_MISSING" flags are excluded). build_matrix.py should select
# cc_merged[CC_SELECTED_COLUMNS] as the credit-card feature block.
CC_SELECTED_COLUMNS = [
    "CC_MONTH_COUNT", "CC_AVG_UTILIZATION", "CC_MAX_UTILIZATION", "CC_AVG_BALANCE",
    "CC_MAX_BALANCE", "CC_AVG_CREDIT_LIMIT", "CC_DPD_RATIO", "CC_DPD_DEF_RATIO",
    "CC_MAX_DPD", "CC_AVG_MIN_PAYMENT_RATIO", "CC_DRAWING_RATIO", "CC_TOTAL_DRAWINGS",
    "CC_RECENT_1Y_COUNT",
]


def build_cc_features(cc: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    cc : raw credit_card_balance.csv (or a subset of rows)

    Returns
    -------
    cc_agg : DataFrame indexed by SK_ID_CURR, ALL aggregate columns
        (CC_SELECTED_COLUMNS + CC_UNIQUE_PREV) — not yet left-joined onto
        the full applicant population and not yet fillna(0)'d; see module
        docstring, that's build_matrix.py's job.
    """
    cc = cc.copy()

    cc["CC_UTILIZATION"] = np.where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
        cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"], np.nan,
    )
    cc["CC_DPD_FLAG"] = (cc["SK_DPD"] > 0).astype("int8")
    cc["CC_DPD_DEF_FLAG"] = (cc["SK_DPD_DEF"] > 0).astype("int8")
    cc["CC_MIN_PAYMENT_RATIO"] = np.where(
        cc["AMT_INST_MIN_REGULARITY"] > 0,
        cc["AMT_PAYMENT_CURRENT"].fillna(0) / cc["AMT_INST_MIN_REGULARITY"], np.nan,
    )
    cc["CC_DRAWING_ACTIVITY"] = (cc["AMT_DRAWINGS_CURRENT"].fillna(0) > 0).astype("int8")
    cc["CC_RECENT_1Y"] = (cc["MONTHS_BALANCE"] >= -12).astype("int8")

    cc_agg = cc.groupby("SK_ID_CURR").agg(
        CC_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        CC_AVG_UTILIZATION=("CC_UTILIZATION", "mean"),
        CC_MAX_UTILIZATION=("CC_UTILIZATION", "max"),
        CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
        CC_MAX_BALANCE=("AMT_BALANCE", "max"),
        CC_AVG_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DPD_RATIO=("CC_DPD_FLAG", "mean"),
        CC_DPD_DEF_RATIO=("CC_DPD_DEF_FLAG", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_AVG_MIN_PAYMENT_RATIO=("CC_MIN_PAYMENT_RATIO", "mean"),
        CC_DRAWING_RATIO=("CC_DRAWING_ACTIVITY", "mean"),
        CC_TOTAL_DRAWINGS=("AMT_DRAWINGS_CURRENT", "sum"),
        CC_RECENT_1Y_COUNT=("CC_RECENT_1Y", "sum"),
        CC_UNIQUE_PREV=("SK_ID_PREV", "nunique"),
    )

    return cc_agg