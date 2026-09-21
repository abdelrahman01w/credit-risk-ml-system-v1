"""
Feature engineering on POS_CASH_balance.csv, aggregated to one row per
SK_ID_CURR.

Direct port of notebook cell 11. Only 9 columns made it into the final
winning feature set (X_pos) — see POS_SELECTED_COLUMNS below.

IMPORTANT for build_matrix.py: the NAME_CONTRACT_STATUS crosstab below
(POS_STATUS_* columns) will only contain the status categories present
in whatever slice of POS_CASH_balance.csv is passed in. At serve time,
with one applicant's history, you'll almost never see every status
category that existed in the full training set. build_matrix.py MUST
reindex this table's columns against the fitted feature_names_final_v2
list (fill missing categories with 0) rather than trusting whatever
columns come out of pd.crosstab() here — otherwise column order/count
will silently drift between train and serve.
"""
import numpy as np
import pandas as pd

# Only these 9 columns were carried into the final winning feature set (X_pos).
POS_SELECTED_COLUMNS = [
    "POS_AVG_INSTALLMENT", "POS_AVG_INSTALLMENT_FUTURE", "POS_MIN_INSTALLMENT_FUTURE",
    "POS_MAX_INSTALLMENT_FUTURE", "POS_RECORD_COUNT", "POS_CONTRACT_COUNT",
    "POS_MONTH_COUNT", "POS_RECENT_1Y_RATIO", "POS_RECENT_2Y_RATIO",
]


def build_pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    pos : raw POS_CASH_balance.csv (or a subset of rows)

    Returns
    -------
    pos_agg : DataFrame indexed by SK_ID_CURR, ALL aggregate + status-crosstab
        columns (not yet subset to POS_SELECTED_COLUMNS — the selected subset
        drops the crosstab columns entirely, so the reindexing caveat above
        only matters if you keep the full pos_agg somewhere downstream).
    """
    pos = pos.copy()

    pos["POS_DPD_FLAG"] = (pos["SK_DPD"] > 0).astype(np.int8)
    pos["POS_DPD_DEF_FLAG"] = (pos["SK_DPD_DEF"] > 0).astype(np.int8)
    pos["POS_SEVERE_DPD_FLAG"] = (pos["SK_DPD"] > 30).astype(np.int8)
    pos["POS_RECENT_1Y"] = (pos["MONTHS_BALANCE"] >= -12).astype(np.int8)
    pos["POS_RECENT_2Y"] = (pos["MONTHS_BALANCE"] >= -24).astype(np.int8)

    pos_agg = pos.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("SK_ID_PREV", "count"),
        POS_CONTRACT_COUNT=("SK_ID_PREV", "nunique"),
        POS_DPD_COUNT=("POS_DPD_FLAG", "sum"),
        POS_DPD_DEF_COUNT=("POS_DPD_DEF_FLAG", "sum"),
        POS_SEVERE_DPD_COUNT=("POS_SEVERE_DPD_FLAG", "sum"),
        POS_AVG_DPD=("SK_DPD", "mean"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_AVG_DPD_DEF=("SK_DPD_DEF", "mean"),
        POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
        POS_AVG_INSTALLMENT=("CNT_INSTALMENT", "mean"),
        POS_AVG_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_MIN_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "min"),
        POS_MAX_INSTALLMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "max"),
        POS_MONTH_COUNT=("MONTHS_BALANCE", "nunique"),
        POS_RECENT_1Y_COUNT=("POS_RECENT_1Y", "sum"),
        POS_RECENT_2Y_COUNT=("POS_RECENT_2Y", "sum"),
    )

    denom = pos_agg["POS_RECORD_COUNT"].replace(0, np.nan)
    pos_agg["POS_DPD_RATIO"] = pos_agg["POS_DPD_COUNT"] / denom
    pos_agg["POS_DPD_DEF_RATIO"] = pos_agg["POS_DPD_DEF_COUNT"] / denom
    pos_agg["POS_SEVERE_DPD_RATIO"] = pos_agg["POS_SEVERE_DPD_COUNT"] / denom
    pos_agg["POS_RECENT_1Y_RATIO"] = pos_agg["POS_RECENT_1Y_COUNT"] / denom
    pos_agg["POS_RECENT_2Y_RATIO"] = pos_agg["POS_RECENT_2Y_COUNT"] / denom

    status_counts = pd.crosstab(pos["SK_ID_CURR"], pos["NAME_CONTRACT_STATUS"])
    status_counts.columns = ["POS_STATUS_" + str(c).replace(" ", "_") for c in status_counts.columns]
    pos_agg = pos_agg.join(status_counts, how="left")
    for col in status_counts.columns:
        pos_agg[col + "_RATIO"] = pos_agg[col] / pos_agg["POS_RECORD_COUNT"].replace(0, np.nan)

    pos_agg = pos_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in pos_agg.columns:
        pos_agg[col] = pos_agg[col].astype(np.float32)

    return pos_agg