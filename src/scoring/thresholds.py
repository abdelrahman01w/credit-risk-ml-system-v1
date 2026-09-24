"""
Credit-score mapping + decision/tier classification.

Direct port of notebook cell 35's compute_credit_score() and the
decision-tier if/elif block from credit_decision_engine(). The two cutoff
values (0.0723 / 0.2000) are the notebook's cost-optimal threshold from
cell 27 (instance-dependent cost matrix using AMT_CREDIT * LGD vs
AMT_CREDIT * INTEREST_MARGIN) cross-checked against cell 34's P&L
simulation across acceptance rates — NOT re-derived here, just consumed.
They already live in config.py (CUTOFF_APPROVE, CUTOFF_REJECT, SCORE_MIN,
SCORE_MAX) since they're policy constants shared with reason_codes.py and
predict.py, so this module reads them from there rather than duplicating.

Deliberately NOT ported here: cell 27's cost-curve scan and cell 34's
acceptance-rate P&L table are one-time model-risk analyses used to ARRIVE
at CUTOFF_APPROVE/CUTOFF_REJECT during training/validation — they don't
run at serve time and don't belong in the request path. If the model is
ever retrained, those two cells (or their equivalent) should be re-run
against the new validation predictions and config.py's cutoffs updated
deliberately, not silently.
"""
from .. import config


def compute_credit_score(
    prob_default: float,
    min_score: int = config.SCORE_MIN,
    max_score: int = config.SCORE_MAX,
) -> int:
    """Linear mapping: prob_default=0 -> max_score (850), prob_default=1 -> min_score (300).
    Exact port of cell 35's compute_credit_score()."""
    score = max_score - (prob_default * (max_score - min_score))
    return int(max(min_score, min(max_score, round(score))))


def classify_decision(prob_default: float) -> tuple[str, str]:
    """
    Returns (decision, risk_tier), exact port of cell 35's if/elif block.

    decision     : "AUTO-APPROVE" | "MANUAL REVIEW" | "AUTO-REJECT"
    risk_tier    : human-readable grade + guidance string
    """
    if prob_default < config.CUTOFF_APPROVE:
        return "AUTO-APPROVE", "Low Risk (Grade A/B)"
    elif prob_default < config.CUTOFF_REJECT:
        return "MANUAL REVIEW", "Medium Risk (Grade C/D) - Request Collateral/Guarantor"
    else:
        return "AUTO-REJECT", "High Risk (Grade E) - Decline Application"