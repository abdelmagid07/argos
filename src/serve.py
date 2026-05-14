"""Stage 5: FastAPI prediction service.

Run:
    uvicorn src.serve:app --port 8000

Endpoints:
    GET  /health   liveness check
    POST /predict  fraud score for a single transaction
    GET  /stats    in-process counters

Diagnostics under load
----------------------
- ``GET /stats`` exposes ``in_flight`` and rolling latency percentiles.
- ``POST /predict?timings=1`` adds ``breakdown_ms`` (feature store, prep,
  scaler, model) so you can see CPU vs I/O.

"""
from __future__ import annotations

import json
import logging
import pickle
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from src.feature_store import FeatureStore, load_feature_store
from src.model import FraudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve")


class PredictRequest(BaseModel):
    user_id: int = Field(..., ge=0)
    merchant_id: int = Field(..., ge=0)
    amount: float = Field(..., gt=0)


class PredictResponse(BaseModel):
    fraud_score: float
    risk_label: str
    model_version: str
    latency_ms: float
    used_user_features: bool
    used_merchant_features: bool
    breakdown_ms: Optional[dict[str, float]] = None


class _State:
    """Process-wide singleton holding the model, scaler, feature store, and
    rolling counters. Loaded once on startup."""

    def __init__(self) -> None:
        self.model: Optional[FraudDetector] = None
        self.scaler = None
        self.feature_columns: list[str] = []
        self.store: Optional[FeatureStore] = None
        self.model_version = "v1"

        self.counters = {
            "total": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "missing_user_features": 0,
            "missing_merchant_features": 0,
        }
        self.latencies_ms: deque[float] = deque(maxlen=1000)
        self.in_flight = 0
        self.in_flight_max = 0

    def load(self) -> None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        log.info("Loading model from %s", MODEL_PATH)
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            input_dim = ckpt["input_dim"]
            state_dict = ckpt["state_dict"]
        else:
            # Backwards-compatible: bare state_dict, infer input_dim from columns.
            input_dim = len(FEATURE_COLUMNS)
            state_dict = ckpt
        self.model = FraudDetector(input_dim=input_dim)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        log.info("Loading scaler from %s", SCALER_PATH)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH) as f:
                self.feature_columns = json.load(f)
        else:
            self.feature_columns = FEATURE_COLUMNS

        self.store = load_feature_store()
        log.info("State loaded. model_version=%s feature_dim=%d backend=%s",
                 self.model_version, input_dim, self.store.backend_name)

    def record(self, *, label: str, latency_ms: float,
               missing_user: bool, missing_merchant: bool) -> None:
        # All callers run on the event loop (single thread), so no lock needed.
        self.counters["total"] += 1
        self.counters[label] += 1
        if missing_user:
            self.counters["missing_user_features"] += 1
        if missing_merchant:
            self.counters["missing_merchant_features"] += 1
        self.latencies_ms.append(latency_ms)


STATE = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE.load()
    except FileNotFoundError as e:
        log.error("Failed to load model artifacts: %s", e)
        log.error("Run: python -m src.ingest && python -m src.features && python -m src.train")
        raise
    yield


app = FastAPI(title="Argos Fraud Detection (MVP)", lifespan=lifespan)


def _risk_label(score: float) -> str:
    if score > 0.7:
        return "high"
    if score > 0.4:
        return "medium"
    return "low"


@app.get("/health")
async def health() -> dict:
    store = STATE.store
    return {
        "status": "ok",
        "model_loaded": STATE.model is not None,
        "feature_store": store.backend_name if store else "none",
        "users_in_store": store.num_users if store else 0,
        "merchants_in_store": store.num_merchants if store else 0,
    }


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True)
async def predict(
    req: PredictRequest,
    timings: bool = Query(False, description="Include breakdown_ms (server-only timing)."),
) -> PredictResponse:
    if STATE.model is None or STATE.store is None or STATE.scaler is None:
        raise HTTPException(503, "Model not loaded")

    STATE.in_flight += 1
    STATE.in_flight_max = max(STATE.in_flight_max, STATE.in_flight)

    try:
        started = time.perf_counter()
        bd: dict[str, float] | None = {} if timings else None

        # Feature lookup — in-memory dict or Redis.
        user_feats = STATE.store.get_user_features(req.user_id)
        merchant_feats = STATE.store.get_merchant_features(req.merchant_id)
        if bd is not None:
            bd["features_ms"] = round((time.perf_counter() - started) * 1000, 3)
        t_prep0 = time.perf_counter()

        # MVP policy: cold-start users/merchants get zeros and a flag in the
        # response. Production will likely reject or route to a fallback model.
        feature_lookup = {
            "amount": req.amount,
            "user_avg_amount": user_feats.get("avg_transaction_amount", 0.0),
            "user_total_txns": user_feats.get("total_transactions", 0.0),
            "user_fraud_rate": user_feats.get("fraud_rate", 0.0),
            "velocity_24h": user_feats.get("transaction_velocity_24h", 0.0),
            "unique_merchants": user_feats.get("unique_merchants", 0.0),
            "unique_countries": user_feats.get("unique_countries", 0.0),
            "merchant_fraud_rate": merchant_feats.get("merchant_fraud_rate", 0.0),
            "merchant_avg_amount": merchant_feats.get("merchant_avg_amount", 0.0),
            "merchant_total_txns": merchant_feats.get("merchant_total_transactions", 0.0),
            "merchant_id": req.merchant_id,
        }
        vec = [[float(feature_lookup[c]) for c in STATE.feature_columns]]
        if bd is not None:
            bd["prep_ms"] = round((time.perf_counter() - t_prep0) * 1000, 3)
        t_scale0 = time.perf_counter()

        # Scale + infer — pure CPU, <2ms for an 11-feature MLP.
        scaled = STATE.scaler.transform(vec)
        if bd is not None:
            bd["scaler_ms"] = round((time.perf_counter() - t_scale0) * 1000, 3)
        t_model0 = time.perf_counter()

        tensor = torch.from_numpy(scaled).float()
        score = float(STATE.model.predict_proba(tensor).item())
        if bd is not None:
            bd["model_ms"] = round((time.perf_counter() - t_model0) * 1000, 3)

        label = _risk_label(score)
        latency_ms = (time.perf_counter() - started) * 1000

        STATE.record(
            label=label,
            latency_ms=latency_ms,
            missing_user=not user_feats,
            missing_merchant=not merchant_feats,
        )

        out_breakdown: Optional[dict[str, float]] = bd if timings else None
        return PredictResponse(
            fraud_score=round(score, 4),
            risk_label=label,
            model_version=STATE.model_version,
            latency_ms=round(latency_ms, 3),
            used_user_features=bool(user_feats),
            used_merchant_features=bool(merchant_feats),
            breakdown_ms=out_breakdown,
        )
    finally:
        STATE.in_flight -= 1


@app.get("/stats")
async def stats() -> JSONResponse:
    lats = sorted(STATE.latencies_ms)
    if lats:
        def pct(p: float) -> float:
            return round(lats[min(len(lats) - 1, int(len(lats) * p))], 3)
        latency = {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
                   "samples": len(lats)}
    else:
        latency = {"p50": None, "p95": None, "p99": None, "samples": 0}
    return JSONResponse({
        "counters": STATE.counters,
        "latency_ms": latency,
        "in_flight": {"current": STATE.in_flight, "max_since_boot": STATE.in_flight_max},
        "model_version": STATE.model_version,
    })
