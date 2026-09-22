"""Remote Laya System-1 HTTP service for GPU2.

Run on GPU2 only; Jev on the Pi accesses POST /predict with a 50-ms budget.
"""
import logging
import os
import time
from typing import Literal

import laya
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jev.laya")
MODEL_NAME = os.getenv("LAYA_MODEL", "convaiinnovations/laya")


class PredictionRequest(BaseModel):
    query: str = Field(min_length=1, max_length=16000)


class PredictionResponse(BaseModel):
    strategy: Literal["direct_cmd", "database_search", "graph_lightrag", "complex_llm", "general_fallback"]
    domain: Literal["raspberry_pi", "automotive", "general"]
    confidence: float
    status: str
    latency_ms: float


app = FastAPI(title="Jev Laya Tier-1", version="1.0.0")
agent = None
load_error: str | None = None


@app.on_event("startup")
def load_model() -> None:
    global agent, load_error
    try:
        logger.info("Loading Laya model %s; CUDA=%s", MODEL_NAME, torch.cuda.is_available())
        agent = laya.load(MODEL_NAME)
        logger.info("Laya model is ready")
    except Exception as error:
        load_error = str(error)
        logger.exception("Laya model failed to load")


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if agent is not None else "unavailable", "model": MODEL_NAME, "cuda": torch.cuda.is_available(), "error": load_error}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    t0 = time.perf_counter()
    if agent is None:
        return PredictionResponse(strategy="general_fallback", domain="general", confidence=0.0, status="unavailable", latency_ms=0.0)
    schema = {
        "strategy": {"type": "choice", "options": ["direct_cmd", "database_search", "graph_lightrag", "complex_llm"]},
        "domain": {"type": "choice", "options": ["raspberry_pi", "automotive", "general"]},
    }
    try:
        result = agent.predict({"query": request.query}, schema)
        confidence = min(float(result["strategy"]["confidence"]), float(result["domain"]["confidence"]))
        return PredictionResponse(strategy=result["strategy"]["answer"], domain=result["domain"]["answer"], confidence=confidence, status="success", latency_ms=(time.perf_counter() - t0) * 1000)
    except Exception:
        logger.exception("Laya prediction failed")
        return PredictionResponse(strategy="general_fallback", domain="general", confidence=0.0, status="error", latency_ms=(time.perf_counter() - t0) * 1000)
