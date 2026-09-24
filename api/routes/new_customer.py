"""
POST /api/v1/score/new-customer

Validates the request against schemas.NewCustomerApplicationPayload, then
hands it straight to src.models.predict.score_application() — no
transformation happens here, this layer is purely: validate -> call ->
return. All the actual mapping/feature-engineering/scoring logic lives in
predict.py and the modules it wires together.
"""
from fastapi import APIRouter, HTTPException

from src.models.predict import score_application
from ..schemas import NewCustomerApplicationPayload, CreditDecisionResponse

router = APIRouter()


@router.post("/score/new-customer", response_model=CreditDecisionResponse)
def score_new_customer(payload: NewCustomerApplicationPayload) -> CreditDecisionResponse:
    try:
        result = score_application(payload.model_dump(mode="json"))
    except Exception as exc:
        # NOTE: this is intentionally broad for now — score_application()
        # can fail for many reasons (missing artifact files, malformed
        # nested payload fields, etc.) and none of them are distinguished
        # yet. Once this goes further than local testing, replace with
        # specific exception types + appropriate status codes (422 for
        # payload-shape issues application_mapper.py can't handle, 503 if
        # artifacts failed to load at startup, etc.) rather than a blanket 500.
        raise HTTPException(status_code=500, detail=str(exc))
    return result