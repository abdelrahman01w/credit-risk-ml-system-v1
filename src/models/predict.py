"""
End-to-end scoring for ONE live application. This is the first place in
the codebase where a real payload actually becomes a credit decision —
everything built so far (application_mapper, history_mapper, application.py,
build_features.py, thresholds.py, reason_codes.py) gets wired together here.

Direct port of notebook cell 35's credit_decision_engine(), restructured to
start from a raw intake payload instead of an already-engineered training
row, and using the pickled fit_artifacts from train.py instead of in-memory
training-time state.

Artifacts are loaded ONCE at import time (module-level), not per request —
api/main.py should import this module at startup, not per-request, so the
~5 pickle loads (2 models + 3 fit_artifacts) happen once.
"""
import pickle

import pandas as pd

from .. import config
from ..adapters.application_mapper import map_payload_to_application_row
from ..features.application import engineer_application_features
from ..features.build_features import transform_serve_row
from ..scoring.thresholds import compute_credit_score, classify_decision
from ..scoring.reason_codes import generate_reason_codes
from ..adapters.history_mapper import build_history_features_for_customer
MODEL_VERSION = "credit-risk-xgb-lgb-blend-v1"


def _load_artifacts():
    artifacts_dir = config.ARTIFACTS_DIR
    with open(artifacts_dir / config.XGB_MODEL_FILE, "rb") as f:
        xgb_model = pickle.load(f)
    with open(artifacts_dir / config.LGB_MODEL_FILE, "rb") as f:
        lgb_model = pickle.load(f)
    with open(artifacts_dir / config.APPLICATION_FIT_ARTIFACTS_FILE, "rb") as f:
        application_fit_artifacts = pickle.load(f)
    with open(artifacts_dir / config.BUILD_MATRIX_FIT_ARTIFACTS_FILE, "rb") as f:
        build_matrix_fit_artifacts = pickle.load(f)
    with open(artifacts_dir / config.BLEND_WEIGHTS_FILE, "rb") as f:
        blend_weights = pickle.load(f)
    return xgb_model, lgb_model, application_fit_artifacts, build_matrix_fit_artifacts, blend_weights


# Module-level singletons — loaded once when this module is first imported.
_XGB_MODEL, _LGB_MODEL, _APPLICATION_FIT_ARTIFACTS, _BUILD_MATRIX_FIT_ARTIFACTS, _BLEND_WEIGHTS = _load_artifacts()


def _engineer_row(payload: dict) -> dict:
    """
    payload -> application_mapper -> application.py (fit=False) -> merged
    with history_mapper's output for returning customers.

    Returns a single flat dict with every engineered feature this
    applicant's row could produce (EXT_SOURCE_MEAN, CREDIT_INCOME_RATIO,
    EMPLOYED_YEARS, and — for returning customers only — PREV_*/INST_*/
    BUREAU_* from history_mapper.py). Ready for build_features.transform_serve_row().
    """
    application_row = map_payload_to_application_row(payload)
    df_row = pd.DataFrame([application_row])
    df_engineered, _ = engineer_application_features(
        df_row, fit=False, fit_artifacts=_APPLICATION_FIT_ARTIFACTS,
    )
    engineered_row = df_engineered.iloc[0].to_dict()

    internal_history = payload.get("internal_history", {})
    is_returning = payload.get("is_returning_customer", False) or not internal_history.get(
        "internal_history_missing", 1
    )
    if is_returning:
        bureau_facilities = payload.get("iscore_report_fields", {}).get("bureau_facilities", [])
        history_features = build_history_features_for_customer(internal_history, bureau_facilities)
        engineered_row.update(history_features)

    return engineered_row


def score_application(payload: dict) -> dict:
    """
    Parameters
    ----------
    payload : a NewCustomerApplicationPayload or ReturningCustomerApplicationPayload,
        as a dict (schemas.py's .model_dump(), or raw request JSON already
        validated against those schemas upstream in the API layer).

    Returns
    -------
    dict shaped like schemas.CreditDecisionResponse:
        application_id, credit_score, default_probability, decision,
        risk_tier, reason_codes, model_version
    """
    engineered_row = _engineer_row(payload)

    X_row = transform_serve_row(engineered_row, _BUILD_MATRIX_FIT_ARTIFACTS)

    prob_xgb = _XGB_MODEL.predict_proba(X_row)[:, 1][0]
    prob_lgb = _LGB_MODEL.predict_proba(X_row)[:, 1][0]
    prob_default = (_BLEND_WEIGHTS["xgb"] * prob_xgb) + (_BLEND_WEIGHTS["lgb"] * prob_lgb)

    credit_score = compute_credit_score(prob_default)
    decision, risk_tier = classify_decision(prob_default)
    reason_codes = generate_reason_codes(engineered_row)

    return {
        "application_id": payload.get("application_id"),
        "credit_score": credit_score,
        "default_probability": round(float(prob_default), 4),
        "decision": decision,
        "risk_tier": risk_tier,
        "reason_codes": reason_codes,
        "model_version": MODEL_VERSION,
    }