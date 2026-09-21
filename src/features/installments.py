"""
Feature engineering on installments_payments.csv, aggregated to one row
per SK_ID_CURR.

Direct port of notebook cell 9. Note only 17 of these columns made it
into the final winning feature set (X_pos) — see INSTALL_SELECTED_COLUMNS
below, used by build_matrix.py to select just those before merging.
"""
import numpy as np
import pandas as pd

# Only these 17 columns were carried into the final winning feature set (X_pos).
# build_matrix.py should select install_agg[INSTALL_SELECTED_COLUMNS] before merging.
INSTALL_SELECTED_COLUMNS = [
    "INST_PAYMENT_COUNT", "INST_LATE_COUNT", "INST_SEVERE_LATE_COUNT", "INST_UNDERPAYMENT_COUNT",
    "INST_AVG_DAYS_LATE", "INST_MAX_DAYS_LATE", "INST_AVG_PAYMENT_RATIO", "INST_MIN_PAYMENT_RATIO",
    "INST_TOTAL_INSTALLMENT", "INST_TOTAL_PAYMENT", "INST_AVG_INSTALLMENT", "INST_AVG_PAYMENT",
    "INST_UNIQUE_PREV", "INST_LATE_RATIO", "INST_SEVERE_LATE_RATIO", "INST_UNDERPAYMENT_RATIO",
    "INST_TOTAL_PAYMENT_RATIO",
]


def build_installments_features(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    installments : raw installments_payments.csv (or a subset of rows)

    Returns
    -------
    install_agg : DataFrame indexed by SK_ID_CURR, ALL aggregate columns
        (not yet subset to INSTALL_SELECTED_COLUMNS — do that at merge time).
    """
    installments = installments.copy()

    installments["DAYS_LATE"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["PAYMENT_RATIO"] = installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"].replace(0, np.nan)
    installments["LATE_PAYMENT"] = (installments["DAYS_LATE"] > 0).astype(np.int8)
    installments["SEVERE_LATE_PAYMENT"] = (installments["DAYS_LATE"] > 30).astype(np.int8)
    installments["UNDERPAYMENT"] = (installments["PAYMENT_RATIO"] < 0.95).astype(np.int8)
    installments["PAYMENT_RATIO_CAPPED"] = installments["PAYMENT_RATIO"].clip(0, 2)

    install_agg = installments.groupby("SK_ID_CURR").agg(
        INST_PAYMENT_COUNT=("SK_ID_PREV", "count"),
        INST_LATE_COUNT=("LATE_PAYMENT", "sum"),
        INST_SEVERE_LATE_COUNT=("SEVERE_LATE_PAYMENT", "sum"),
        INST_UNDERPAYMENT_COUNT=("UNDERPAYMENT", "sum"),
        INST_AVG_DAYS_LATE=("DAYS_LATE", "mean"),
        INST_MAX_DAYS_LATE=("DAYS_LATE", "max"),
        INST_AVG_PAYMENT_RATIO=("PAYMENT_RATIO_CAPPED", "mean"),
        INST_MIN_PAYMENT_RATIO=("PAYMENT_RATIO", "min"),
        INST_TOTAL_INSTALLMENT=("AMT_INSTALMENT", "sum"),
        INST_TOTAL_PAYMENT=("AMT_PAYMENT", "sum"),
        INST_AVG_INSTALLMENT=("AMT_INSTALMENT", "mean"),
        INST_AVG_PAYMENT=("AMT_PAYMENT", "mean"),
        INST_UNIQUE_PREV=("SK_ID_PREV", "nunique"),
    )

    install_agg["INST_LATE_RATIO"] = install_agg["INST_LATE_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_SEVERE_LATE_RATIO"] = install_agg["INST_SEVERE_LATE_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_UNDERPAYMENT_RATIO"] = install_agg["INST_UNDERPAYMENT_COUNT"] / install_agg["INST_PAYMENT_COUNT"].replace(0, np.nan)
    install_agg["INST_TOTAL_PAYMENT_RATIO"] = install_agg["INST_TOTAL_PAYMENT"] / install_agg["INST_TOTAL_INSTALLMENT"].replace(0, np.nan)

    install_agg = install_agg.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in install_agg.columns:
        install_agg[col] = install_agg[col].astype(np.float32)

    return install_agg