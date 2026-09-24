# Credit Risk Scoring API — production image.
# Local build:  docker build -t credit-risk-api .
# Local run:    docker run -p 8000:8000 -e PORT=8000 credit-risk-api
# Railway: connect the repo, Railway auto-detects this Dockerfile and sets
# $PORT itself — no manual configuration needed beyond what's below.
#
# IMPORTANT: this image bakes in whatever is currently in model/artifacts/
# at build time (xgb_final.pkl, lgb_final.pkl, application_fit_artifacts.pkl,
# build_matrix_fit_artifacts.pkl, blend_weights.pkl). Run
# `python -m src.models.train --data-dir <path>` locally FIRST and confirm
# model/artifacts/ is populated AND committed to git before deploying —
# src/models/predict.py loads these at import time, so the container will
# crash on startup if they're missing, not fail gracefully per-request.
# Double-check .gitignore does NOT exclude model/artifacts/*.pkl.

FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost / lightgbm wheels (libgomp1 is required by both
# at runtime, not just build time — skipping this causes an import-time
# OSError inside the container even though `pip install` succeeds).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + already-trained artifacts. Raw CSVs under data/raw/
# are deliberately NOT copied in — training happens outside the image,
# only the serving path needs to run in production.
COPY api/ ./api/
COPY src/ ./src/
COPY model/artifacts/ ./model/artifacts/

# Railway (and most PaaS hosts) inject PORT at runtime and route traffic to
# it — the container must listen on THAT port, not a hardcoded one. Default
# to 8000 so `docker run` without -e PORT still works locally.
ENV PORT=8000
EXPOSE 8000

# No --reload in production. Shell form (not exec-array form) so $PORT
# actually gets expanded — the array form does NOT do shell substitution.
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT