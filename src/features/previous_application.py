"""
Feature engineering on previous_application.csv, aggregated to one row
per SK_ID_CURR.

Direct port of notebook cell 7. No fit/serve split needed: this is a
pure groupby().agg() on the applicant's OWN previous-application history,
so it produces the same result whether run over the full training table
or over just one applicant's slice of previous_application.csv at serve
time (missing history -> the caller left-joins and fillna(0), same as here).
"""
import numpy as np
import pandas as pd


def build_previous_application_features(previous: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    previous : raw previous_application.csv (or a subset of rows for one
        or more applicants — must contain SK_ID_CURR, SK_ID_PREV, and the
        original Home Credit columns)

    Returns
    -------
    prev_agg : DataFrame indexed by SK_ID_CURR, one row per applicant,
        ready to be left-joined onto the application-level frame.
    """
    previous = previous.copy()

    previous["PREV_APPROVED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Approved").astype(np.int8)
    previous["PREV_REFUSED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Refused").astype(np.int8)
    previous["PREV_CANCELED_FLAG"] = (previous["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.int8)
    previous["PREV_DAYS_TO_CURRENT"] = -previous["DAYS_DECISION"]
    previous["PREV_RECENT_1Y"] = (previous["PREV_DAYS_TO_CURRENT"] <= 365).astype(np.int8)
    previous["PREV_RECENT_2Y"] = (previous["PREV_DAYS_TO_CURRENT"] <= 730).astype(np.int8)
    previous["PREV_CREDIT_TO_APPLICATION"] = previous["AMT_CREDIT"] / previous["AMT_APPLICATION"].replace(0, np.nan)
    previous["PREV_ANNUITY_TO_CREDIT"] = previous["AMT_ANNUITY"] / previous["AMT_CREDIT"].replace(0, np.nan)

    prev_agg = previous.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("PREV_APPROVED_FLAG", "sum"),
        PREV_REFUSED_COUNT=("PREV_REFUSED_FLAG", "sum"),
        PREV_CANCELED_COUNT=("PREV_CANCELED_FLAG", "sum"),
        PREV_AVG_CREDIT_TO_APPLICATION=("PREV_CREDIT_TO_APPLICATION", "mean"),
        PREV_AVG_ANNUITY=("AMT_ANNUITY", "mean"),
        PREV_AVG_CREDIT=("AMT_CREDIT", "mean"),
        PREV_AVG_APPLICATION=("AMT_APPLICATION", "mean"),
        PREV_AVG_CNT_PAYMENT=("CNT_PAYMENT", "mean"),
        PREV_AVG_DOWN_PAYMENT=("AMT_DOWN_PAYMENT", "mean"),
        PREV_RECENT_1Y_COUNT=("PREV_RECENT_1Y", "sum"),
        PREV_RECENT_2Y_COUNT=("PREV_RECENT_2Y", "sum"),
    )

    prev_agg["PREV_APPROVED_RATIO"] = prev_agg["PREV_APPROVED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_REFUSED_RATIO"] = prev_agg["PREV_REFUSED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_CANCELED_RATIO"] = prev_agg["PREV_CANCELED_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_RECENT_1Y_RATIO"] = prev_agg["PREV_RECENT_1Y_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)
    prev_agg["PREV_RECENT_2Y_RATIO"] = prev_agg["PREV_RECENT_2Y_COUNT"] / prev_agg["PREV_APP_COUNT"].replace(0, np.nan)

    prev_contract_types = pd.crosstab(previous["SK_ID_CURR"], previous["NAME_CONTRACT_TYPE"])
    prev_contract_types.columns = [f"PREV_CONTRACT_{str(c).replace(' ', '_')}" for c in prev_contract_types.columns]

    prev_product_types = pd.crosstab(previous["SK_ID_CURR"], previous["NAME_PORTFOLIO"])
    prev_product_types.columns = [f"PREV_PRODUCT_{str(c).replace(' ', '_')}" for c in prev_product_types.columns]

    prev_agg = prev_agg.join(prev_contract_types, how="left").join(prev_product_types, how="left")
    prev_agg = prev_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in prev_agg.columns:
        prev_agg[col] = prev_agg[col].astype(np.float32)

    return prev_agg