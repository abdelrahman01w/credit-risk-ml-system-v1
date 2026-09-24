"""
FastAPI entry point. Run with:
    uvicorn api.main:app --reload --port 8000
from the project root (so `src` and `api` both resolve as top-level packages).

-> http://localhost:8000/docs for interactive Swagger UI.

NOTE: this deliberately does NOT expose the raw-Kaggle-columns POST /score
shape the original README's Quickstart example showed. That shape assumed
callers would send application_train.csv-style fields directly; what's
actually been built instead is the two-payload intake system
(application_mapper.py + history_mapper.py) matching the real document/OCR
pipeline shape (sample_new_to_bank_payload.json /
sample_returning_customer_payload.json). The README's example request
should be updated to match /api/v1/score/new-customer once this is settled
— it currently describes an endpoint this code doesn't implement.
"""
from fastapi import FastAPI

from .routes.new_customer import router as new_customer_router
from .routes.returning_customer import router as returning_customer_router

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Scores new-to-bank and returning-customer loan applications "
                "via an XGBoost + LightGBM blend trained on Home Credit Default Risk.",
)

app.include_router(new_customer_router, prefix="/api/v1", tags=["scoring"])
app.include_router(returning_customer_router, prefix="/api/v1", tags=["scoring"])


@app.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness check. Does NOT verify artifacts loaded correctly — src.models.predict
    loads them at import time, so if they failed, the app would have already
    crashed on startup rather than serving a false-positive health check here."""
    return {"status": "ok"}