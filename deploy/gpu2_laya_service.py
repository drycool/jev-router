"""Remote Laya System-1 HTTP service for GPU2.

Runs on GPU2 only. Jev on the Pi calls `POST /predict` with a ~50 ms budget and
falls back to its local tiers whenever this service is slow or unavailable, so
this process must never block or crash the gateway.

Laya 0.3.x runtime API:

    questions = {qid: {"type": "choice"|"score"|"noul",
                       "instructions": str,   # required
                       "criteria": dict|list}}
    result  = agent.predict(state, questions)   # state: text, dict or turn list
    result["answers"][qid]["choice"]        # choice questions
    result["answers"][qid]["confidence"]    # calibrated 0..1
    result["answers"][qid]["probabilities"]

Two run modes:

  router (default)  `laya.Router` preloads several checkpoints and picks one per
                    request from the detected script/language. Required for
                    non-English traffic: the English checkpoint does not degrade
                    gracefully off English, it collapses *while staying confident*,
                    so confidence gating cannot catch the mistake.
  single            one fixed checkpoint (`LAYA_SUBFOLDER`), useful for A/B
                    benchmarks of english vs multilingual vs routed.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, cast

import laya
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("jev.laya")

MODE = os.getenv("LAYA_MODE", "router").strip().lower()
MODEL_NAME = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
MODEL_SUBFOLDER = os.getenv("LAYA_SUBFOLDER") or None
PRELOAD = [n.strip() for n in os.getenv("LAYA_PRELOAD", "english,multilingual").split(",") if n.strip()]
DEVICE = os.getenv("LAYA_DEVICE") or None

# The watchdog on GPU2 powers the box off after N idle minutes. Every /predict
# refreshes this lease so an actively used Laya keeps the machine awake, while an
# idle one still lets the box sleep after the usual timeout.
LEASE_PATH = os.path.expanduser(os.getenv("LAYA_LEASE_PATH", "~/logs/activity.lease"))
LEASE_ENABLED = os.getenv("LAYA_LEASE_ENABLED", "1") != "0"

# Strategy/domain labels must stay in sync with core/laya_client.py on the Pi.
STRATEGIES = ("direct_cmd", "database_search", "graph_lightrag", "complex_llm")
DOMAINS = ("raspberry_pi", "automotive", "general")
FALLBACK_STRATEGY = "general_fallback"
Strategy = Literal["direct_cmd", "database_search", "graph_lightrag", "complex_llm", "general_fallback"]
Domain = Literal["raspberry_pi", "automotive", "general"]

# The state key is `request` because the shipped presets reference it by name in
# their instructions ("How hard is `request` for a language model?").
QUESTIONS: dict[str, dict[str, Any]] = {
    "strategy": {
        "type": "choice",
        "instructions": "Which retrieval strategy should handle the `request`?",
        "criteria": {
            "direct_cmd": "run a shell/system command immediately: a status check, restart or one-line action, no knowledge lookup needed",
            "database_search": "look up one specific known record in the local database or index",
            "graph_lightrag": "answer from the knowledge graph; needs several related facts or multi-hop connections",
            "complex_llm": "needs reasoning, synthesis, explanation or code generation by a large language model",
        },
    },
    "domain": {
        "type": "choice",
        "instructions": "What subject domain does the `request` belong to?",
        "criteria": {
            "raspberry_pi": "Raspberry Pi hardware, Linux on ARM, GPIO, cases, cables, power supplies, Pi accessories",
            "automotive": "Daewoo Espero car repair: engine, electrical, bodywork, parts, fault codes",
            "general": "anything that is not clearly about Pi hardware or the car",
        },
    },
}

# The block below is `laya.router_questions()` verbatim (question wording, option
# labels and order).  This matters: the checkpoint is trained with RL against
# these exact workflows, and an invented taxonomy degrades it badly.  Measured on
# Russian traffic, our own labels produced confidences of 0.003-0.37 with
# near-random answers, while this preset answered "chitchat" at 0.988 and "code"
# at 0.926 for the same queries.  Do not "improve" these strings.
QUESTIONS.update(
    {
        "difficulty": {
            "type": "score",
            "instructions": "How hard is `request` for a language model?",
            "criteria": [
                "trivial: a lookup or one-liner",
                "easy: short answer, no reasoning",
                "moderate: several steps",
                "hard: long multi-step reasoning or specialist knowledge",
            ],
        },
        "task": {
            "type": "choice",
            "instructions": "What domain does `request` belong to?",
            "criteria": {
                "code": "software engineering, programming, refactoring, architecture, debugging",
                "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
                "writing": "creative writing, essays, emails, blog posts, copywriting",
                "factual_lookup": "facts, definitions, trivia, history",
                "data_analysis": "statistics, SQL, data manipulation, metrics",
                "chitchat": "casual conversation, greetings, small talk",
            },
        },
        "needs_tools": {
            "type": "noul",
            "instructions": "Does answering `request` require external tools, search or private data?",
        },
        "is_sensitive": {
            "type": "noul",
            "instructions": "Does `request` involve money, legal, medical or safety consequences?",
        },
    }
)


# One GPU, one model set: serialise inference so concurrent requests cannot thrash
# VRAM or double-book the forward pass.
INFERENCE_LOCK = threading.Lock()

ENGINE: Any = None
ENGINE_DESCRIPTION: str = "not loaded"
LOAD_ERROR: str | None = None
LOAD_MS: float | None = None

STATS: dict[str, Any] = {
    "requests": 0,
    "errors": 0,
    "statuses": {},
    "checkpoints": {},  # which checkpoint actually answered (routing observability)
    "inference_ms": [],  # bounded ring of the last 200 timings
}


def touch_lease() -> None:
    """Mark GPU2 as busy for the idle-shutdown watchdog."""
    if not LEASE_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(LEASE_PATH), exist_ok=True)
        with open(LEASE_PATH, "a"):
            os.utime(LEASE_PATH, None)
    except Exception:
        logger.warning("could not refresh lease %s", LEASE_PATH, exc_info=True)


def _build_engine() -> tuple[Any, str]:
    if MODE == "single":
        agent = laya.load(MODEL_NAME, device=DEVICE, subfolder=MODEL_SUBFOLDER)
        return agent, f"single:{MODEL_SUBFOLDER or 'root'}({MODEL_NAME})"
    router = laya.Router(device=DEVICE)
    router.preload(PRELOAD)
    return router, f"router:preload={','.join(PRELOAD)}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, ENGINE_DESCRIPTION, LOAD_ERROR, LOAD_MS
    touch_lease()
    started = time.perf_counter()
    try:
        logger.info("loading Laya mode=%s preload=%s CUDA=%s", MODE, PRELOAD if MODE != "single" else MODEL_SUBFOLDER, torch.cuda.is_available())
        ENGINE, ENGINE_DESCRIPTION = _build_engine()
        LOAD_MS = (time.perf_counter() - started) * 1000
        logger.info("Laya ready (%s) in %.0f ms", ENGINE_DESCRIPTION, LOAD_MS)
    except Exception as error:  # keep serving /health so the Pi can report *why*
        LOAD_ERROR = f"{type(error).__name__}: {error}"
        LOAD_MS = (time.perf_counter() - started) * 1000
        logger.exception("Laya model failed to load")
    touch_lease()
    yield
    ENGINE = None


app = FastAPI(title="Jev Laya Tier-1", version="2.1.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    query: str = Field(min_length=1, max_length=16000)
    # Router mode only: force one preloaded checkpoint for this call so a
    # benchmark can A/B english vs multilingual vs routed without restarts.
    model: str | None = Field(default=None, max_length=64)


class PredictionResponse(BaseModel):
    strategy: Strategy
    domain: Domain
    confidence: float
    status: str
    latency_ms: float
    inference_ms: float = 0.0
    checkpoint: str | None = None      # which Laya checkpoint answered
    routing_reason: str | None = None  # why the router picked it
    probabilities: dict[str, dict[str, float]] | None = None
    # Shipped-preset signals (in-distribution for this checkpoint).
    difficulty: float | None = None
    task: str | None = None
    task_confidence: float | None = None
    needs_tools: float | None = None
    is_sensitive: float | None = None


def _fallback(status: str, latency_ms: float, strategy: str = FALLBACK_STRATEGY, domain: str = "general") -> PredictionResponse:
    return PredictionResponse(
        strategy=cast(Strategy, strategy),
        domain=cast(Domain, domain),
        confidence=0.0,
        status=status,
        latency_ms=latency_ms,
    )


@app.get("/health")
def health() -> dict:
    ready = ENGINE is not None
    return {
        "status": "ok" if ready else "unavailable",
        "mode": MODE,
        "engine": ENGINE_DESCRIPTION,
        "model": MODEL_NAME,
        "subfolder": MODEL_SUBFOLDER,
        "loaded": ready,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "load_ms": round(LOAD_MS, 1) if LOAD_MS is not None else None,
        "error": LOAD_ERROR,
        "lease": LEASE_PATH if LEASE_ENABLED else None,
        "requests": STATS["requests"],
        "errors": STATS["errors"],
        "checkpoints": STATS["checkpoints"],
    }


@app.get("/schema")
def schema() -> dict:
    """The exact question schema this service sends to Laya (for benchmarks)."""
    return {
        "mode": MODE,
        "engine": ENGINE_DESCRIPTION,
        "strategies": list(STRATEGIES),
        "domains": list(DOMAINS),
        "questions": QUESTIONS,
    }


@app.get("/stats")
def stats() -> dict:
    timings = STATS["inference_ms"]
    return {
        "requests": STATS["requests"],
        "errors": STATS["errors"],
        "statuses": STATS["statuses"],
        "checkpoints": STATS["checkpoints"],
        "inference_ms_avg": round(sum(timings) / len(timings), 2) if timings else None,
        "inference_ms_max": round(max(timings), 2) if timings else None,
        "samples": len(timings),
    }


def _record(status: str, inference_ms: float, checkpoint: str | None = None) -> None:
    STATS["requests"] += 1
    STATS["statuses"][status] = STATS["statuses"].get(status, 0) + 1
    if checkpoint:
        STATS["checkpoints"][checkpoint] = STATS["checkpoints"].get(checkpoint, 0) + 1
    if status != "success":
        STATS["errors"] += 1
    if inference_ms:
        STATS["inference_ms"].append(inference_ms)
        del STATS["inference_ms"][:-200]


def _score(query: str, model: str | None = None) -> dict[str, Any]:
    """Run one Laya forward pass. Raises on any runtime failure.

    All questions are evaluated in a single non-autoregressive forward pass, so
    the preset block costs head tokens rather than extra round trips.
    """
    started = time.perf_counter()
    with INFERENCE_LOCK:
        if model and MODE == "router":
            result = ENGINE.predict({"request": query}, QUESTIONS, model=model)
        else:
            result = ENGINE.predict({"request": query}, QUESTIONS)
    inference_ms = (time.perf_counter() - started) * 1000

    answers = result["answers"]
    strategy = str(answers["strategy"]["choice"])
    domain = str(answers["domain"]["choice"])
    if strategy not in STRATEGIES:
        raise ValueError(f"model returned unknown strategy {strategy!r}")
    if domain not in DOMAINS:
        raise ValueError(f"model returned unknown domain {domain!r}")
    routing = result.get("routing") or {}
    return {
        "strategy": strategy,
        "domain": domain,
        # The weaker of the two verdicts gates the pair, so one confident field
        # cannot carry an uncertain one past the acceptance threshold.
        "confidence": min(float(answers["strategy"]["confidence"]), float(answers["domain"]["confidence"])),
        "inference_ms": inference_ms,
        "checkpoint": routing.get("model"),
        "routing_reason": routing.get("reason"),
        "probabilities": {
            "strategy": {k: float(v) for k, v in answers["strategy"].get("probabilities", {}).items()},
            "domain": {k: float(v) for k, v in answers["domain"].get("probabilities", {}).items()},
        },
        # Shipped-preset signals. These are the in-distribution ones: they are what
        # should drive the strategy choice, while subject domain stays with the
        # local regex classifier, which is both free and exact on this corpus.
        "difficulty": float(answers.get("difficulty", {}).get("score", 0.0)),
        "task": answers.get("task", {}).get("choice"),
        "task_confidence": float(answers.get("task", {}).get("confidence", 0.0)),
        "needs_tools": float(answers.get("needs_tools", {}).get("noul", 0.0)),
        "is_sensitive": float(answers.get("is_sensitive", {}).get("noul", 0.0)),
    }



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    touch_lease()
    t0 = time.perf_counter()
    if ENGINE is None:
        _record("unavailable", 0.0)
        return _fallback("unavailable", (time.perf_counter() - t0) * 1000)
    try:
        scored = _score(request.query, request.model)
    except Exception:
        logger.exception("Laya prediction failed")
        _record("error", 0.0)
        return _fallback("error", (time.perf_counter() - t0) * 1000)
    _record("success", scored["inference_ms"], scored["checkpoint"])
    return PredictionResponse(
        strategy=cast(Strategy, scored["strategy"]),
        domain=cast(Domain, scored["domain"]),
        confidence=scored["confidence"],
        status="success",
        latency_ms=(time.perf_counter() - t0) * 1000,
        inference_ms=scored["inference_ms"],
        checkpoint=scored["checkpoint"],
        routing_reason=scored["routing_reason"],
        probabilities=scored["probabilities"],
        difficulty=scored["difficulty"],
        task=scored["task"],
        task_confidence=scored["task_confidence"],
        needs_tools=scored["needs_tools"],
        is_sensitive=scored["is_sensitive"],
    )


@app.post("/warmup")
def warmup() -> dict:
    """Pay the first-call cost (CUDA kernels, lazy buffers) before a benchmark."""
    if ENGINE is None:
        return {"status": "unavailable", "error": LOAD_ERROR}
    t0 = time.perf_counter()
    try:
        _score("status check warmup")
    except Exception as error:
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    return {"status": "ok", "engine": ENGINE_DESCRIPTION, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


if __name__ == "__main__":  # quick smoke test without uvicorn
    print(json.dumps({"mode": MODE, "preload": PRELOAD, "questions": QUESTIONS}, ensure_ascii=False, indent=2))
