"""
Feature engineering on bureau.csv + bureau_balance.csv, aggregated to one
row per SK_ID_CURR.

Direct port of notebook cell 15's aggregation logic (bb -> bb_agg ->
bureau_agg). Stops at bureau_agg, exactly where previous_app.py /
pos_cash.py stop — the notebook's own merge-onto-base_ids + missing-flag +
column-drop steps happen AFTER a left-join against the FULL applicant
population (not just applicants with bureau history), so that logic
belongs in build_matrix.py, not here. Two constants below tell
build_matrix.py exactly how to replicate that merge-time step:

  BUREAU_MISSING_FLAG_SOURCE_COLUMNS : the 5 bureau_agg columns the
      notebook adds a "_MISSING" indicator for, computed AFTER the
      left-join onto every applicant (so it captures both "no bureau
      history at all" and "has bureau history but this one sub-metric
      came out NaN", e.g. BUREAU_DEBT_TO_CREDIT when AMT_CREDIT_SUM was 0
      for every facility).
  BUREAU_MISSING_FLAG_KEEP : of those 5 generated "_MISSING" flags, only
      these 2 survived the notebook's empirical feature-selection pass
      (cell 15's `bureau_drop` list) and should be kept in the final
      matrix. The other 3 ("_AVG_DPD_RATIO_MISSING",
      "_AVG_MONTH_COUNT_MISSING", "_MAX_DAYS_CREDIT_ENDDATE_MISSING")
      were dropped as zero-importance and must NOT appear in
      feature_names_final_v2.

build_matrix.py's job, in order, once it has bureau_agg from this module:
  1. left-join bureau_agg onto the full df_fe population on SK_ID_CURR
  2. for col in BUREAU_MISSING_FLAG_SOURCE_COLUMNS: add f"{col}_MISSING"
  3. .fillna(0) on the merged frame
  4. keep BUREAU_AGG_COLUMNS + [f"{c}_MISSING" for c in BUREAU_MISSING_FLAG_KEEP]
     as the final bureau feature block (17 columns total)

NOTE (known gap, flagged for later): history_mapper.py's serve-time
BUREAU_* reconstruction for returning customers is built from the much
thinner bureau_facilities[] payload block (no DAYS_CREDIT, no
bureau_balance detail at all) and currently uses different column names
than this module produces. Reconciling the two — so a live /score request
can actually populate these 17 columns instead of falling back to 0 via
imputation — is unresolved and should happen once build_matrix.py exists
and the real column list is locked in.
"""
import numpy as np
import pandas as pd

# The 15 raw aggregate columns bureau_agg always has (everything except
# SK_ID_CURR / the index). Provided so build_matrix.py doesn't have to
# re-derive this list by inspecting the DataFrame.
BUREAU_AGG_COLUMNS = [
    "BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_RATIO", "BUREAU_CLOSED_RATIO", "BUREAU_OVERDUE_RATIO",
    "BUREAU_TOTAL_CREDIT", "BUREAU_TOTAL_DEBT", "BUREAU_TOTAL_OVERDUE", "BUREAU_MAX_OVERDUE",
    "BUREAU_MAX_DAYS_CREDIT_ENDDATE", "BUREAU_RECENT_1Y_COUNT", "BUREAU_RECENT_2Y_COUNT",
    "BUREAU_AVG_DPD_RATIO", "BUREAU_MAX_STATUS", "BUREAU_AVG_MONTH_COUNT", "BUREAU_DEBT_TO_CREDIT",
]

# Missing-flag bookkeeping — see module docstring for how build_matrix.py uses these.
BUREAU_MISSING_FLAG_SOURCE_COLUMNS = [
    "BUREAU_AVG_DPD_RATIO", "BUREAU_MAX_STATUS", "BUREAU_AVG_MONTH_COUNT",
    "BUREAU_MAX_DAYS_CREDIT_ENDDATE", "BUREAU_DEBT_TO_CREDIT",
]
BUREAU_MISSING_FLAG_KEEP = ["BUREAU_MAX_STATUS", "BUREAU_DEBT_TO_CREDIT"]


def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    bureau : raw bureau.csv (or a subset of rows)
    bureau_balance : raw bureau_balance.csv (or a subset of rows)

    Returns
    -------
    bureau_agg : DataFrame indexed by SK_ID_CURR, the 15 BUREAU_AGG_COLUMNS.
        NOT yet left-joined onto the full applicant population and NOT yet
        missing-flagged / fillna(0)'d — see module docstring, that's
        build_matrix.py's job.
    """
    bb = bureau_balance.copy()
    status_num = pd.to_numeric(bb["STATUS"], errors="coerce")
    bb["BB_DPD_FLAG"] = (status_num.fillna(0) > 0).astype("int8")
    bb["BB_CLOSED_FLAG"] = (bb["STATUS"] == "C").astype("int8")

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(
        BB_MONTH_COUNT=("MONTHS_BALANCE", "count"),
        BB_DPD_RATIO=("BB_DPD_FLAG", "mean"),
        BB_CLOSED_RATIO=("BB_CLOSED_FLAG", "mean"),
        BB_MAX_STATUS=("STATUS", lambda x: pd.to_numeric(x, errors="coerce").max()),
    ).reset_index()

    bureau = bureau.copy().merge(bb_agg, on="SK_ID_BUREAU", how="left")

    bureau["BUREAU_ACTIVE_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
    bureau["BUREAU_CLOSED_FLAG"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype("int8")
    bureau["BUREAU_OVERDUE_FLAG"] = (bureau["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0).astype("int8")
    bureau["BUREAU_RECENT_1Y"] = (bureau["DAYS_CREDIT"] >= -365).astype("int8")
    bureau["BUREAU_RECENT_2Y"] = (bureau["DAYS_CREDIT"] >= -730).astype("int8")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_RATIO=("BUREAU_ACTIVE_FLAG", "mean"),
        BUREAU_CLOSED_RATIO=("BUREAU_CLOSED_FLAG", "mean"),
        BUREAU_OVERDUE_RATIO=("BUREAU_OVERDUE_FLAG", "mean"),
        BUREAU_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
        BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_TOTAL_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_MAX_OVERDUE=("AMT_CREDIT_SUM_OVERDUE", "max"),
        BUREAU_MAX_DAYS_CREDIT_ENDDATE=("DAYS_CREDIT_ENDDATE", "max"),
        BUREAU_RECENT_1Y_COUNT=("BUREAU_RECENT_1Y", "sum"),
        BUREAU_RECENT_2Y_COUNT=("BUREAU_RECENT_2Y", "sum"),
        BUREAU_AVG_DPD_RATIO=("BB_DPD_RATIO", "mean"),
        BUREAU_MAX_STATUS=("BB_MAX_STATUS", "max"),
        BUREAU_AVG_MONTH_COUNT=("BB_MONTH_COUNT", "mean"),
    )
    bureau_agg["BUREAU_DEBT_TO_CREDIT"] = (
        bureau_agg["BUREAU_TOTAL_DEBT"] / bureau_agg["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)
    )

    return bureau_agg