"""
Central configuration for the credit-risk-v1 project.

Every path and business constant that used to be a hardcoded literal
scattered across notebook cells lives here, so train.py and predict.py
(and the API) all read from one source of truth.
"""
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# Project root = credit-risk-v1/ (this file lives at credit-risk-v1/src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    "/home/seg/Documents/projects/credix/credit_risk_prediction_system/"
    "credit-risk-v1/datasets/raw"
)

ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

APPLICATION_TRAIN_PATH = DATA_DIR / "application_train.csv"
PREVIOUS_APPLICATION_PATH = DATA_DIR / "previous_application.csv"
INSTALLMENTS_PATH = DATA_DIR / "installments_payments.csv"
POS_CASH_PATH = DATA_DIR / "POS_CASH_balance.csv"
BUREAU_PATH = DATA_DIR / "bureau.csv"
BUREAU_BALANCE_PATH = DATA_DIR / "bureau_balance.csv"
CREDIT_CARD_PATH = DATA_DIR / "credit_card_balance.csv"

# ------------------------------------------------------------------
# Model artifact filenames (all live under ARTIFACTS_DIR)
# ------------------------------------------------------------------
XGB_MODEL_FILE = "xgb_final.pkl"
LGB_MODEL_FILE = "lgb_final.pkl"
ENCODER_FILE = "onehot_encoder.pkl"
FEATURE_NAMES_FILE = "feature_names_final_v2.pkl"
FIT_ARTIFACTS_FILE = "fit_artifacts.pkl"  # medians, missing-flag columns, etc. (see application.py)

# ------------------------------------------------------------------
# Blend weights (from cell 22's blend-weight search — overwritten by
# train.py each time it reruns the search, this is just the last-known value)
# ------------------------------------------------------------------
BLEND_W_XGB = 0.3
BLEND_W_LGB = 0.7

# ------------------------------------------------------------------
# Business decision cutoffs (from cell 35 — "Bank Policy Cutoffs based
# on P&L Simulation"). These are policy, not statistics: confirm with
# risk team before changing.
# ------------------------------------------------------------------
CUTOFF_APPROVE = 0.0723   # PD below this -> Auto-Approve (NPL ~ 3.14%)
CUTOFF_REJECT = 0.2000    # PD at/above this -> Auto-Reject
# Between CUTOFF_APPROVE and CUTOFF_REJECT -> Manual Underwriting Review

# Cost-sensitive analysis assumptions (cell 27 / cell 34)
LGD = 0.45                # Loss Given Default: fraction of AMT_CREDIT lost on default
INTEREST_MARGIN = 0.10    # Net interest margin on a performing loan

# Credit score mapping (cell 35)
SCORE_MIN = 300
SCORE_MAX = 850

RANDOM_STATE = 42

# ------------------------------------------------------------------
# Payload-mapping assumptions (application_mapper.py) — these are
# BUSINESS DECISIONS standing in for data we don't have. Confirm with
# risk team before relying on them; revisit if score distributions
# look off after deployment.
# ------------------------------------------------------------------
# ASSUMPTION 1: Kaggle's AMT_INCOME_TOTAL is annual income (median ~147k vs
# AMT_CREDIT median ~513k implies a ~3.5x ratio consistent with annual
# income against a multi-year loan, not monthly). declared_net_salary in
# the payload is monthly -> multiply by 12 to match the training scale.
# If this is wrong, CREDIT_INCOME_RATIO / ANNUITY_INCOME_RATIO will be off
# by ~12x for every applicant, which will visibly distort the score.
INCOME_ANNUALIZATION_MULTIPLIER = 12

# ASSUMPTION 2: EXT_SOURCE_1/2/3 (the model's single strongest feature) has
# no true equivalent in this payload. Proxied by min-max normalizing the
# I-Score credit_score into the same 0-1 range EXT_SOURCE values occupy,
# and using that one proxy value for all three EXT_SOURCE_1/2/3 slots.
# This is a stopgap, not a validated substitute — re-check the
# cost-optimal threshold (CUTOFF_APPROVE/CUTOFF_REJECT above) against
# real outcomes once live I-Score-based scores start coming in.
ISCORE_MIN = 300
ISCORE_MAX = 850