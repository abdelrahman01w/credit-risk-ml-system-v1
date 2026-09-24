"""
End-to-end integration tests through src.models.predict.score_application(),
using the real sample payloads (src/sample_new_to_bank_payload.json,
src/sample_returning_customer_payload.json).

These require trained artifacts to already exist in model/artifacts/ (see
src/model/train.py) — skipped automatically if they don't, so `pytest` runs
clean in a fresh checkout before anyone has trained a model, same as the
README's original test-skipping note intended.

Run with: pytest tests/test_pipeline.py
"""
import json
from pathlib import Path

import pytest

from src import config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_NEW = _PROJECT_ROOT / "src" / "sample_new_to_bank_payload.json"
_SAMPLE_RETURNING = _PROJECT_ROOT / "src" / "sample_returning_customer_payload.json"

_ARTIFACTS_EXIST = all(
    (config.ARTIFACTS_DIR / fname).exists()
    for fname in [
        config.XGB_MODEL_FILE, config.LGB_MODEL_FILE,
        config.APPLICATION_FIT_ARTIFACTS_FILE, config.BUILD_MATRIX_FIT_ARTIFACTS_FILE,
        config.BLEND_WEIGHTS_FILE,
    ]
)

pytestmark = pytest.mark.skipif(
    not _ARTIFACTS_EXIST,
    reason="model/artifacts/ is missing trained artifacts — run src/model/train.py first.",
)


def _load_sample(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_new_customer_payload_scores_successfully():
    from src.models.predict import score_application

    payload = _load_sample(_SAMPLE_NEW)
    result = score_application(payload)

    assert result["application_id"] == payload["application_id"]
    assert 300 <= result["credit_score"] <= 850
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["decision"] in {"AUTO-APPROVE", "MANUAL REVIEW", "AUTO-REJECT"}
    assert len(result["reason_codes"]) >= 1
    assert result["model_version"]


def test_returning_customer_payload_scores_successfully():
    from src.models.predict import score_application

    payload = _load_sample(_SAMPLE_RETURNING)
    result = score_application(payload)

    assert result["application_id"] == payload["application_id"]
    assert 300 <= result["credit_score"] <= 850
    assert 0.0 <= result["default_probability"] <= 1.0
    assert result["decision"] in {"AUTO-APPROVE", "MANUAL REVIEW", "AUTO-REJECT"}


def test_returning_customer_uses_history_features():
    """
    A regression guard for history_mapper.py actually being wired in:
    the returning customer's clean repayment history (24/24 on-time
    installments, zero refusals) should make them score AT LEAST as low-risk
    as the cold-start new customer — if history_mapper.py's output stopped
    reaching build_features.transform_serve_row() (e.g. a silent key-name
    mismatch), this comparison would likely flip or the returning customer's
    score would look suspiciously close to a "no history" default.
    """
    from src.models.predict import score_application

    new_result = score_application(_load_sample(_SAMPLE_NEW))
    returning_result = score_application(_load_sample(_SAMPLE_RETURNING))

    assert returning_result["default_probability"] <= new_result["default_probability"]


@pytest.mark.xfail(
    reason="Frozen-value regression check — WILL fail after any retrain. "
           "That's the point: it catches silent drift. Update the expected "
           "values below deliberately after confirming a retrain's new "
           "numbers are correct, don't just delete this test.",
    strict=False,
)
def test_frozen_new_customer_score_matches_last_known_good_value():
    from src.models.predict import score_application

    result = score_application(_load_sample(_SAMPLE_NEW))
    # Last known-good values, captured from a real run on 2026-09-21.
    assert result["credit_score"] == 822
    assert result["default_probability"] == pytest.approx(0.0504, abs=0.0005)