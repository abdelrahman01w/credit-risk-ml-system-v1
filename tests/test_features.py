"""
Unit tests for the per-table feature-engineering modules. These test the
FUNCTIONS in isolation with small synthetic DataFrames — no CSVs, no
trained artifacts, no network. Run with: pytest tests/test_features.py
"""
import numpy as np
import pandas as pd
import pytest

from src.features.application import engineer_application_features
from src.features.previous_application import build_previous_application_features
from src.features.installments import build_installments_features, INSTALL_SELECTED_COLUMNS
from src.features.pos_cash import build_pos_cash_features, POS_SELECTED_COLUMNS
from src.features.bureau import build_bureau_features, BUREAU_AGG_COLUMNS
from src.features.credit_card import build_cc_features, CC_SELECTED_COLUMNS


# ------------------------------------------------------------------
# application.py
# ------------------------------------------------------------------
def _make_application_df(n=5):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "TARGET": rng.integers(0, 2, n),
        "DAYS_BIRTH": -rng.integers(7300, 25550, n),          # ~20-70 years
        "DAYS_EMPLOYED": -rng.integers(30, 10000, n),
        "AMT_CREDIT": rng.uniform(50000, 900000, n),
        "AMT_INCOME_TOTAL": rng.uniform(50000, 400000, n),
        "AMT_ANNUITY": rng.uniform(5000, 50000, n),
        "AMT_GOODS_PRICE": rng.uniform(50000, 900000, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 5, n),
        "CNT_CHILDREN": rng.integers(0, 3, n),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
        "NAME_CONTRACT_TYPE": ["Cash loans"] * n,
    })


def test_engineer_application_features_fit_produces_core_ratios():
    df = _make_application_df()
    df_fe, fit_artifacts = engineer_application_features(df, fit=True)

    assert "AGE_YEARS" in df_fe.columns
    assert "CREDIT_INCOME_RATIO" in df_fe.columns
    assert "EXT_SOURCE_MEAN" in df_fe.columns
    # AGE_YEARS should be positive and roughly plausible (20-70)
    assert (df_fe["AGE_YEARS"] > 15).all() and (df_fe["AGE_YEARS"] < 80).all()
    assert fit_artifacts.categorical_fill_value == "Missing"


def test_engineer_application_features_serve_reuses_fit_artifacts():
    df_train = _make_application_df(n=20)
    _, fit_artifacts = engineer_application_features(df_train, fit=True)

    # A single serve-time row, missing several columns entirely (as a real
    # application_mapper.py row would be) — must not crash, must not re-fit.
    single_row = pd.DataFrame([{
        "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -2000,
        "AMT_CREDIT": 90000.0, "AMT_INCOME_TOTAL": 216000.0, "AMT_ANNUITY": 4600.0,
        "AMT_GOODS_PRICE": 90000.0, "CNT_FAM_MEMBERS": 1, "CNT_CHILDREN": 0,
        "EXT_SOURCE_1": 0.69, "EXT_SOURCE_2": 0.69, "EXT_SOURCE_3": 0.69,
        "NAME_CONTRACT_TYPE": "Cash loans",
    }])
    df_served, _ = engineer_application_features(single_row, fit=False, fit_artifacts=fit_artifacts)
    assert len(df_served) == 1
    assert not df_served["EXT_SOURCE_MEAN"].isna().any()


def test_engineer_application_features_serve_without_fit_artifacts_raises():
    df = _make_application_df()
    with pytest.raises(ValueError):
        engineer_application_features(df, fit=False, fit_artifacts=None)


def test_ext_source_missing_does_not_raise_keyerror():
    """Regression test for the bug fixed in application_mapper.py: EXT_SOURCE_1/2/3
    columns must exist (even as NaN) or _secondary_features()'s .mean(axis=1) KeyErrors."""
    df = _make_application_df()
    df["EXT_SOURCE_1"] = np.nan
    df["EXT_SOURCE_2"] = np.nan
    df["EXT_SOURCE_3"] = np.nan
    df_fe, _ = engineer_application_features(df, fit=True)
    assert "EXT_SOURCE_MEAN" in df_fe.columns


# ------------------------------------------------------------------
# previous_application.py
# ------------------------------------------------------------------
def test_build_previous_application_features_basic_aggregation():
    previous = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [101, 102, 201],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "DAYS_DECISION": [-100, -800, -50],
        "AMT_CREDIT": [10000, 20000, 30000],
        "AMT_APPLICATION": [10000, 20000, 30000],
        "AMT_ANNUITY": [1000, 2000, 3000],
        "CNT_PAYMENT": [12, 24, 12],
        "AMT_DOWN_PAYMENT": [0, 0, 0],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
        "NAME_PORTFOLIO": ["POS", "POS", "Cards"],
    })
    prev_agg = build_previous_application_features(previous)

    assert prev_agg.loc[1, "PREV_APP_COUNT"] == 2
    assert prev_agg.loc[1, "PREV_APPROVED_COUNT"] == 1
    assert prev_agg.loc[1, "PREV_REFUSED_COUNT"] == 1
    assert prev_agg.loc[2, "PREV_APP_COUNT"] == 1
    assert not prev_agg.isna().any().any()  # fillna(0) should leave no NaNs


# ------------------------------------------------------------------
# installments.py
# ------------------------------------------------------------------
def test_build_installments_features_late_payment_flagging():
    installments = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 1],
        "SK_ID_PREV": [101, 101, 102],
        "DAYS_INSTALMENT": [-100, -70, -40],
        "DAYS_ENTRY_PAYMENT": [-100, -30, -40],  # row 2 is 40 days late
        "AMT_INSTALMENT": [1000, 1000, 1000],
        "AMT_PAYMENT": [1000, 1000, 1000],
    })
    install_agg = build_installments_features(installments)

    assert install_agg.loc[1, "INST_PAYMENT_COUNT"] == 3
    assert install_agg.loc[1, "INST_LATE_COUNT"] == 1
    assert install_agg.loc[1, "INST_SEVERE_LATE_COUNT"] == 1  # >30 days late
    # every selected column should exist and be numeric
    for col in INSTALL_SELECTED_COLUMNS:
        assert col in install_agg.columns


# ------------------------------------------------------------------
# pos_cash.py
# ------------------------------------------------------------------
def test_build_pos_cash_features_selected_columns_present():
    pos = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [101, 101, 201],
        "MONTHS_BALANCE": [-1, -2, -1],
        "CNT_INSTALMENT": [12, 12, 24],
        "CNT_INSTALMENT_FUTURE": [10, 9, 20],
        "SK_DPD": [0, 5, 0],
        "SK_DPD_DEF": [0, 0, 0],
        "NAME_CONTRACT_STATUS": ["Active", "Active", "Completed"],
    })
    pos_agg = build_pos_cash_features(pos)
    for col in POS_SELECTED_COLUMNS:
        assert col in pos_agg.columns
    assert pos_agg.loc[1, "POS_RECORD_COUNT"] == 2


# ------------------------------------------------------------------
# bureau.py
# ------------------------------------------------------------------
def test_build_bureau_features_selected_columns_present():
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_BUREAU": [9001, 9002],
        "CREDIT_ACTIVE": ["Active", "Closed"],
        "DAYS_CREDIT": [-100, -900],
        "DAYS_CREDIT_ENDDATE": [200, -100],
        "AMT_CREDIT_SUM": [50000, 20000],
        "AMT_CREDIT_SUM_DEBT": [10000, 0],
        "AMT_CREDIT_SUM_OVERDUE": [0, 0],
    })
    bureau_balance = pd.DataFrame({
        "SK_ID_BUREAU": [9001, 9001, 9002],
        "MONTHS_BALANCE": [-1, -2, -1],
        "STATUS": ["0", "1", "C"],
    })
    bureau_agg = build_bureau_features(bureau, bureau_balance)
    for col in BUREAU_AGG_COLUMNS:
        assert col in bureau_agg.columns
    assert bureau_agg.loc[1, "BUREAU_LOAN_COUNT"] == 2


# ------------------------------------------------------------------
# credit_card.py
# ------------------------------------------------------------------
def test_build_cc_features_selected_columns_present():
    cc = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "SK_ID_PREV": [301, 301],
        "MONTHS_BALANCE": [-1, -2],
        "AMT_BALANCE": [5000, 4000],
        "AMT_CREDIT_LIMIT_ACTUAL": [20000, 20000],
        "SK_DPD": [0, 0],
        "SK_DPD_DEF": [0, 0],
        "AMT_INST_MIN_REGULARITY": [500, 500],
        "AMT_PAYMENT_CURRENT": [500, 500],
        "AMT_DRAWINGS_CURRENT": [1000, 0],
    })
    cc_agg = build_cc_features(cc)
    for col in CC_SELECTED_COLUMNS:
        assert col in cc_agg.columns
    assert cc_agg.loc[1, "CC_MONTH_COUNT"] == 2