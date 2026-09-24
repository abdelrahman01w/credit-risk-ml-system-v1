"""
POST /api/v1/score/returning-customer

Same shape as new_customer.py, validating against
schemas.ReturningCustomerApplicationPayload instead — the only difference
being the richer internal_history block, which predict.py's
_engineer_row() detects via is_returning_customer / internal_history_missing
and routes through history_mapper.py accordingly. No branching needed here.
"""
from fastapi import APIRouter, HTTPException

from src.models.predict import score_application
from ..schemas import ReturningCustomerApplicationPayload, CreditDecisionResponse

router = APIRouter()


@router.post("/score/returning-customer", response_model=CreditDecisionResponse)
def score_returning_customer(payload: ReturningCustomerApplicationPayload) -> CreditDecisionResponse:
    try:
        result = score_application(payload.model_dump(mode="json"))
    except Exception as exc:
        # See new_customer.py's NOTE — same caveat applies here.
        raise HTTPException(status_code=500, detail=str(exc))
    return result