# Credit Risk Scoring System

A credit-default risk model (XGBoost + LightGBM ensemble) trained on the
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
dataset, served behind a FastAPI scoring API with two intake endpoints —
one for new-to-bank applicants, one for returning customers with real
internal loan/bureau history.

Validation performance (on this training run): **ROC-AUC ≈ 0.789**.

## Project structure

```
credit_risk_ml_system_v1/
├── api/
│   ├── main.py                    # FastAPI app + router wiring
│   ├── schemas.py                 # Pydantic request/response models
│   └── routes/
│       ├── new_customer.py        # POST /api/v1/score/new-customer
│       └── returning_customer.py  # POST /api/v1/score/returning-customer
├── src/
│   ├── config.py                  # paths, cutoffs, business-assumption constants
│   ├── adapters/
│   │   ├── application_mapper.py  # payload -> Kaggle-shaped application row
│   │   └── history_mapper.py      # returning-customer history -> PREV_*/INST_*/BUREAU_*
│   ├── features/                  # one module per Home Credit source table
│   │   ├── application.py
│   │   ├── previous_application.py
│   │   ├── installments.py
│   │   ├── pos_cash.py
│   │   ├── bureau.py
│   │   ├── credit_card.py
│   │   └── build_features.py      # assembles the final feature matrix
│   ├── models/
│   │   ├── train.py                # trains + saves artifacts
│   │   └── predict.py              # scores one live application
│   └── scoring/
│       ├── thresholds.py           # PD -> credit score, decision, risk tier
│       └── reason_codes.py         # adverse-action explanations
├── model/artifacts/                # trained model + fit artifacts (see Training)
├── data/raw/                       # Home Credit CSVs (not committed — see Data)
├── tests/
│   ├── test_features.py            # unit tests per feature module
│   └── test_pipeline.py            # end-to-end tests via predict.score_application()
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a dedicated environment
conda create -n credit_risk_ml_system_v1 python=3.14
conda activate credit_risk_ml_system_v1

# 2. Install dependencies
pip install -r requirements.txt
```

## Data

This repo does not ship the training data. Download the Home Credit Default
Risk dataset and place these 7 files under `data/raw/`:

```
application_train.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
bureau.csv
bureau_balance.csv
credit_card_balance.csv
```

## Training

```bash
python -m src.models.train --data-dir data/raw
```

Writes 5 artifacts to `model/artifacts/`:

| File | Contents |
|---|---|
| `xgb_final.pkl` | trained XGBClassifier |
| `lgb_final.pkl` | trained LGBMClassifier |
| `application_fit_artifacts.pkl` | medians / missing-flag columns learned from training data |
| `build_matrix_fit_artifacts.pkl` | fitted OneHotEncoder + locked column order + final feature list |
| `blend_weights.pkl` | XGBoost/LightGBM blend weights from the validation-set search |

All 5 are required before `predict.py` (and therefore the API) can run —
it loads them once at import time and will fail on startup if any are
missing.

## Running the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: `http://localhost:8000/docs`

### Endpoints

- `POST /api/v1/score/new-customer` — cold-start applicant, no internal
  bank history. Request shape: `src/sample_new_to_bank_payload.json`.
- `POST /api/v1/score/returning-customer` — existing customer with real
  internal loan/bureau history. Request shape:
  `src/sample_returning_customer_payload.json`.
- `GET /health` — liveness check.

Both scoring endpoints return:

```json
{
  "application_id": "APP-2026-EG-00202",
  "credit_score": 822,
  "default_probability": 0.0504,
  "decision": "AUTO-APPROVE",
  "risk_tier": "Low Risk (Grade A/B)",
  "reason_codes": ["No critical risk flags detected; standard portfolio profile"],
  "model_version": "credit-risk-xgb-lgb-blend-v1"
}
```

- `credit_score`: 300–850
- `default_probability`: 0–1
- `decision`: `AUTO-APPROVE` / `MANUAL REVIEW` / `AUTO-REJECT`, based on the
  cost-optimal cutoffs in `src/config.py` (`CUTOFF_APPROVE`, `CUTOFF_REJECT`)
- `reason_codes`: up to 3 human-readable explanations (see Known limitations)

## Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 -e PORT=8000 credit-risk-api
```

The image bakes in whatever is currently in `model/artifacts/` at build
time — train first, confirm all 5 files exist, **then** build.

## Deploying (Railway)

1. Confirm `.gitignore` excludes `data/raw/*.csv` and `__pycache__/`, but
   does **not** exclude `model/artifacts/*.pkl` (Railway builds from what's
   committed).
2. Push to GitHub, then in Railway: New Project → Deploy from GitHub repo.
   Railway auto-detects the `Dockerfile`.
3. No environment variables are required — the app has no API keys or
   secrets today. Railway injects `PORT` automatically; the Dockerfile
   already listens on it.

## Tests

```bash
pytest tests/ -v
```

- `test_features.py` — unit tests per feature module, synthetic data, no
  trained artifacts required.
- `test_pipeline.py` — end-to-end tests through `predict.score_application()`
  using the real sample payloads. Skipped automatically if
  `model/artifacts/` isn't populated yet.

## Known limitations

- **Feature count**: the notebook this was ported from reports 413 final
  features; this pipeline's `build_features.py` produces 409 on the last
  training run. Not yet root-caused — likely a small difference in which
  rare categorical values appear in this environment's data slice.
  Worth confirming before relying on this in production.
- **`reason_codes`**: of the 6 signals in `src/scoring/reason_codes.py`,
  two (late-payment history, prior-refusal history) only have real data for
  **returning** customers, via `history_mapper.py`. A third
  (credit-card utilization) has no data source yet for either applicant
  type — there's no credit-card adapter built. That rule can't fire today.
- **Payload-mapping assumptions**: `src/adapters/application_mapper.py`
  makes two business-decision assumptions (monthly-to-annual income
  scaling, I-Score-as-EXT_SOURCE-proxy) documented in `src/config.py`.
  These are stopgaps pending real risk-team sign-off and outcome data.
- **Model risk governance**: population validation, model-risk sign-off,
  and regulatory approval have not happened. This is a working technical
  pipeline, not a production-approved lending decision system.