"""
Jev API Server
==============
FastAPI server for the 4-tier Jev router pipeline.
"""
import asyncio
import time
import os
import hashlib
import json
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.decision_engine import (
    DECISION_ENGINE_LOW_CONFIDENCE,
    DECISION_ENGINE_URL,
    DECISION_SCHEMAS,
    DecisionEngineClient,
)
from core.laya_client import LAYA_CONFIDENCE_THRESHOLD, LAYA_URL
from core.router import AgentType, JevRouter, RoutingResult, Strategy
from core.shadow import ShadowProbe, ShadowTarget
from agents.base import (
    GeneralAgent, CodeAgent, DBAgent, TroubleshooterAgent,
    AgentResponse,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────
LLM_HOST = os.getenv("JEV_LLM_HOST", "http://192.168.11.87:11434")
LLM_MODEL = os.getenv("JEV_LLM_MODEL", "ornith-1.5-9b-256k")
LIGHTRAG_API = os.getenv("JEV_LIGHTRAG_API", "http://localhost:8020")
DECISION_LOG_PATH = os.getenv("JEV_DECISION_LOG_PATH", os.path.join(PROJECT_ROOT, "jev_decisions.jsonl"))
LIGHTRAG_CHUNKS_PATH = os.getenv(
    "JEV_LIGHTRAG_CHUNKS_PATH",
    "/home/dry/LightRag/car_index_espero_clean/kv_store_text_chunks.json",
)
LOG_RAW_QUERY = os.getenv("JEV_LOG_RAW_QUERY", "false").lower() == "true"

# ── Shadow mode ───────────────────────────────────────────────────────
# A candidate decision engine probed on real traffic, off the request path.
# "off" is the default: nothing is probed until someone opts in per target.
SHADOW_MODE = os.getenv("JEV_SHADOW_MODE", "off").strip().lower()
if SHADOW_MODE not in {"off", "laya", "decision"}:
    print(f"[Jev] unknown JEV_SHADOW_MODE={SHADOW_MODE!r}; shadow probes disabled")
    SHADOW_MODE = "off"
SHADOW_DEFAULT_URL = {"laya": LAYA_URL, "decision": DECISION_ENGINE_URL}.get(SHADOW_MODE, "")
SHADOW_URL = os.getenv("JEV_SHADOW_URL") or SHADOW_DEFAULT_URL
SHADOW_TIMEOUT_S = float(os.getenv("JEV_SHADOW_TIMEOUT_S", "1.0"))
SHADOW_MAX_INFLIGHT = int(os.getenv("JEV_SHADOW_MAX_INFLIGHT", "1"))
SHADOW_SAMPLE_RATE = float(os.getenv("JEV_SHADOW_SAMPLE_RATE", "1.0"))

decision_logger = logging.getLogger("jev.decisions")
decision_logger.setLevel(logging.INFO)
if not decision_logger.handlers:
    handler = logging.FileHandler(DECISION_LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    decision_logger.addHandler(handler)
    decision_logger.propagate = False


# ── Lifespan ──────────────────────────────────────────────────────────
router: JevRouter = None
decision_engine = DecisionEngineClient()


def _record_shadow(event: dict) -> None:
    """Persist one shadow observation and fold it into the live counters.

    Shadow probes are the only place a candidate engine's own answer is recorded,
    so the counters here are what a promotion decision gets argued from.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "shadow",
        **event,
    }
    decision_logger.info(json.dumps(payload, ensure_ascii=False))

    _stats["shadow_requests"] += 1
    _stats["shadow_latency_ms"] += float(event.get("latency_ms", 0.0))
    status = str(event.get("status", "error"))
    if status != "success":
        _stats["shadow_errors"] += 1
    if status == "timeout":
        _stats["shadow_timeouts"] += 1
    if event.get("low_confidence"):
        _stats["shadow_low_confidence"] += 1
    choice = event.get("choice")
    if choice:
        _stats["shadow_choices"][str(choice)] = _stats["shadow_choices"].get(str(choice), 0) + 1


shadow_probe = ShadowProbe(
    target=cast(ShadowTarget, SHADOW_MODE),
    url=SHADOW_URL,
    timeout_s=SHADOW_TIMEOUT_S,
    max_inflight=SHADOW_MAX_INFLIGHT,
    sample_rate=SHADOW_SAMPLE_RATE,
    low_confidence_threshold=DECISION_ENGINE_LOW_CONFIDENCE,
    on_result=_record_shadow,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    router = JevRouter()

    # Register agents
    router.tier4.register_agent(
        AgentType.GENERAL,
        GeneralAgent(LLM_HOST, LLM_MODEL)
    )
    router.tier4.register_agent(
        AgentType.CODE, CodeAgent(LLM_HOST, LLM_MODEL)
    )
    router.tier4.register_agent(
        AgentType.DB, DBAgent(LLM_HOST, LLM_MODEL)
    )
    router.tier4.register_agent(
        AgentType.TROUBLESHOOTER,
        TroubleshooterAgent(LLM_HOST, LLM_MODEL)
    )

    # Index existing LightRAG chunks into FTS5
    await _index_lightrag_chunks()

    yield
    await shadow_probe.aclose()
    await router.laya.aclose()
    router.tier2.close()


async def _index_lightrag_chunks():
    """Index chunks from LightRAG into FTS5 for fast search."""
    import json
    from pathlib import Path

    chunks_path = Path(LIGHTRAG_CHUNKS_PATH)
    if not chunks_path.exists():
        return

    try:
        with open(chunks_path) as f:
            data = json.load(f)

        # This is a derived index.  Rebuild it atomically in one transaction so
        # repeated restarts cannot multiply every document in FTS5.
        router.tier2.clear()
        count = 0
        for chunk_id, chunk_data in data.items():
            content = chunk_data.get("content", "")
            if content:
                router.tier2.index_chunk(
                    chunk_id=chunk_id,
                    content=content[:2000],
                    source=chunk_data.get("file_path", ""),
                )
                count += 1
        router.tier2.commit()

        print(f"[Jev] Indexed {count} chunks into FTS5")
    except Exception as e:
        print(f"[Jev] FTS5 indexing error: {e}")


# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Jev Multi-Tier Router",
    description="4-tier pipeline: Fast Router → FTS5/Vector → LightRAG → LLM",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=16000, description="User query")
    execute: bool = Field(default=True, description="Execute target agent after routing")


class QueryResponse(BaseModel):
    routing_decision: dict
    extracted_metadata: dict
    rag_configuration: dict
    target_agent: str
    context_preview: str = ""
    agent_response: str = ""
    elapsed_ms: float = 0.0
    degraded: bool = False
    fallback_reason: str | None = None


class StatsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    tier1_exits: int
    tier2_hits: int
    tier3_hits: int
    degraded_requests: int
    agent_errors: int
    laya_predictions: int
    laya_accepted: int
    laya_not_awaited: int
    decision_engine_requests: int
    decision_engine_errors: int
    decision_engine_low_confidence: int
    decision_engine_latency_ms: float
    shadow_requests: int
    shadow_errors: int
    shadow_timeouts: int
    shadow_low_confidence: int
    shadow_latency_ms_avg: float
    shadow_choices: dict[str, int]
    shadow: dict


class DecisionTestRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, max_length=16000)
    candidates: list[str] = Field(
        default_factory=lambda: ["exact_fts", "vector_fast", "graph_lightrag", "general_llm"],
        min_length=1,
        max_length=16,
    )
    schema_name: str = "routing_v1"
    context: str = Field(default="", max_length=16000)

    @model_validator(mode="before")
    @classmethod
    def accept_schema_alias(cls, data):
        if isinstance(data, dict) and "schema" in data and "schema_name" not in data:
            data = dict(data)
            data["schema_name"] = data.pop("schema")
        return data


class DecisionTestResponse(BaseModel):
    choice: str
    confidence: float
    latency_ms: float
    engine: str
    status: str
    low_confidence: bool
    fallback_reason: str | None = None


# ── Stats tracking ────────────────────────────────────────────────────
_stats = {
    "total": 0,
    "tier1": 0,
    "tier2": 0,
    "tier3": 0,
    "total_ms": 0.0,
    "degraded": 0,
    "agent_errors": 0,
    "laya_predictions": 0,
    "laya_accepted": 0,
    "laya_not_awaited": 0,
    "decision_engine_requests": 0,
    "decision_engine_errors": 0,
    "decision_engine_low_confidence": 0,
    "decision_engine_latency_ms": 0.0,
    "shadow_requests": 0,
    "shadow_errors": 0,
    "shadow_timeouts": 0,
    "shadow_low_confidence": 0,
    "shadow_latency_ms": 0.0,
    "shadow_choices": {},
}


def _laya_accepted(result: RoutingResult) -> bool:
    """Whether the GPU2 classifier's verdict cleared the acceptance threshold.

    Recorded per request because "how often would the System-1 tier have been
    trusted?" is the number that decides whether its threshold is calibrated.

    Gated on the shipped-preset ``task`` signal only.  The service's
    ``confidence`` is the minimum over two hand-written questions this
    checkpoint was never trained on, so a confident-but-wrong verdict clears it -
    which is exactly why raising that number was never going to help.
    """
    laya = result.laya_result or {}
    return (
        laya.get("status") == "success"
        and float(laya.get("task_confidence", 0.0)) >= LAYA_CONFIDENCE_THRESHOLD
    )


def _laya_status(result: RoutingResult) -> str:
    return str((result.laya_result or {}).get("status", "absent"))


def _record_decision(query: str, result: RoutingResult, elapsed_ms: float, agent_error: bool = False) -> None:
    """Write one privacy-preserving JSONL event for later evaluation/ML labels."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "decision": {
            "strategy": result.routing_decision.strategy.value,
            "confidence": result.routing_decision.confidence_score,
            "keywords": result.extracted_metadata.keywords,
            "entities": result.extracted_metadata.entities,
            "domain": result.extracted_metadata.domain,
        },
        "execution": {
            "latency_ms": round(elapsed_ms, 2),
            "degraded": result.degraded,
            "fallback_reason": result.fallback_reason,
            "agent_error": agent_error,
            "laya_accepted": _laya_accepted(result),
            "decision_engine_used": False,
            "decision_engine_candidate_count": 0,
            "decision_engine_confidence": None,
        },
        "laya_result": result.laya_result,
    }
    if LOG_RAW_QUERY:
        event["raw_query"] = query
    decision_logger.info(json.dumps(event, ensure_ascii=False))


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "jev-router",
        "tiers": ["fast_router", "fts5_vector", "lightrag", "llm"],
        "lightrag_api": LIGHTRAG_API,
        "llm_host": LLM_HOST,
        "decision_engine_url": DECISION_ENGINE_URL,
        "decision_schemas": sorted(DECISION_SCHEMAS),
        "shadow": {
            "mode": SHADOW_MODE,
            "enabled": shadow_probe.enabled,
            "url": SHADOW_URL or None,
            "timeout_s": SHADOW_TIMEOUT_S,
            "max_inflight": SHADOW_MAX_INFLIGHT,
            "sample_rate": SHADOW_SAMPLE_RATE,
            "requests": _stats["shadow_requests"],
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Main entry point. Routes through 4-tier pipeline.
    """
    t0 = time.perf_counter()
    _stats["total"] += 1

    # Tier 1-3: Route
    result: RoutingResult = await router.route(req.query)

    # Shadow probe: dispatched off the request path.  It is scheduled here, after
    # routing, so it overlaps Tier 4 and never adds latency to this response —
    # failures are counted, not raised.
    shadow_probe.submit(req.query, context=result.context[:2000] if result.context else "")

    # Track stats
    if result.routing_decision.strategy == Strategy.DIRECT_ACTION:
        _stats["tier1"] += 1
    elif result.routing_decision.strategy in (Strategy.EXACT_FTS, Strategy.VECTOR_FAST):
        _stats["tier2"] += 1
    elif result.routing_decision.strategy == Strategy.GRAPH_LIGHTRAG:
        _stats["tier3"] += 1
    if result.degraded:
        _stats["degraded"] += 1
    if result.laya_result:
        if _laya_status(result) == "not_awaited":
            # The local path answered without waiting for GPU2.  Counting this as
            # a prediction would make the classifier look busier than it is.
            _stats["laya_not_awaited"] += 1
        else:
            _stats["laya_predictions"] += 1
            if _laya_accepted(result):
                _stats["laya_accepted"] += 1

    # Tier 4: Execute agent if requested
    agent_response = ""
    agent_error = False
    if req.execute:
        try:
            resp: AgentResponse = await router.tier4.execute(result)
            agent_response = resp.answer
        except Exception as e:
            agent_response = f"Agent error: {e}"
            agent_error = True
            _stats["agent_errors"] += 1

    elapsed = (time.perf_counter() - t0) * 1000
    _stats["total_ms"] += elapsed
    _record_decision(req.query, result, elapsed, agent_error)

    return QueryResponse(
        routing_decision={
            "strategy": result.routing_decision.strategy.value,
            "confidence_score": result.routing_decision.confidence_score,
            "fast_path_exit": result.routing_decision.fast_path_exit,
        },
        extracted_metadata={
            "intent": result.extracted_metadata.intent,
            "keywords": result.extracted_metadata.keywords,
            "entities": result.extracted_metadata.entities,
            "domain": result.extracted_metadata.domain,
        },
        rag_configuration={
            "lightrag_required": result.rag_configuration.lightrag_required,
            "lightrag_mode": result.rag_configuration.lightrag_mode,
            "similarity_threshold": result.rag_configuration.similarity_threshold,
        },
        target_agent=result.target_agent.value,
        context_preview=result.context[:500] if result.context else "",
        agent_response=agent_response,
        elapsed_ms=round(elapsed, 2),
        degraded=result.degraded,
        fallback_reason=result.fallback_reason,
    )


@app.get("/stats")
async def stats():
    """Pipeline statistics."""
    total = max(_stats["total"], 1)
    return StatsResponse(
        total_requests=_stats["total"],
        avg_latency_ms=round(_stats["total_ms"] / total, 2),
        tier1_exits=_stats["tier1"],
        tier2_hits=_stats["tier2"],
        tier3_hits=_stats["tier3"],
        degraded_requests=_stats["degraded"],
        agent_errors=_stats["agent_errors"],
        laya_predictions=_stats["laya_predictions"],
        laya_accepted=_stats["laya_accepted"],
        laya_not_awaited=_stats["laya_not_awaited"],
        decision_engine_requests=_stats["decision_engine_requests"],
        decision_engine_errors=_stats["decision_engine_errors"],
        decision_engine_low_confidence=_stats["decision_engine_low_confidence"],
        decision_engine_latency_ms=round(_stats["decision_engine_latency_ms"], 2),
        shadow_requests=_stats["shadow_requests"],
        shadow_errors=_stats["shadow_errors"],
        shadow_timeouts=_stats["shadow_timeouts"],
        shadow_low_confidence=_stats["shadow_low_confidence"],
        shadow_latency_ms_avg=round(
            _stats["shadow_latency_ms"] / max(_stats["shadow_requests"], 1), 2
        ),
        shadow_choices=dict(_stats["shadow_choices"]),
        shadow=shadow_probe.stats(),
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus text exposition; labels are intentionally low-cardinality."""
    lines = [
        "# TYPE jev_requests_total counter",
        f'jev_requests_total{{tier="tier1"}} {_stats["tier1"]}',
        f'jev_requests_total{{tier="tier2"}} {_stats["tier2"]}',
        f'jev_requests_total{{tier="tier3"}} {_stats["tier3"]}',
        "# TYPE jev_degraded_requests_total counter",
        f'jev_degraded_requests_total {_stats["degraded"]}',
        "# TYPE jev_agent_errors_total counter",
        f'jev_agent_errors_total {_stats["agent_errors"]}',
        "# TYPE jev_laya_predictions_total counter",
        f'jev_laya_predictions_total {_stats["laya_predictions"]}',
        "# TYPE jev_laya_accepted_total counter",
        f'jev_laya_accepted_total {_stats["laya_accepted"]}',
        "# TYPE jev_laya_not_awaited_total counter",
        f'jev_laya_not_awaited_total {_stats["laya_not_awaited"]}',
        "# TYPE jev_request_latency_ms_total counter",
        f'jev_request_latency_ms_total {_stats["total_ms"]:.3f}',
        "# TYPE jev_decision_engine_requests_total counter",
        f'jev_decision_engine_requests_total {_stats["decision_engine_requests"]}',
        "# TYPE jev_decision_engine_errors_total counter",
        f'jev_decision_engine_errors_total {_stats["decision_engine_errors"]}',
        "# TYPE jev_decision_low_confidence_total counter",
        f'jev_decision_low_confidence_total {_stats["decision_engine_low_confidence"]}',
        "# TYPE jev_decision_engine_latency_ms_total counter",
        f'jev_decision_engine_latency_ms_total {_stats["decision_engine_latency_ms"]:.3f}',
    ]

    # Shadow probes: what a candidate engine would have answered on live traffic.
    probe = shadow_probe.stats()
    target = str(probe["target"])
    lines += [
        "# TYPE jev_shadow_enabled gauge",
        f'jev_shadow_enabled{{target="{target}"}} {1 if probe["enabled"] else 0}',
        "# TYPE jev_shadow_requests_total counter",
        f'jev_shadow_requests_total{{target="{target}",status="success"}} {probe["success"]}',
        f'jev_shadow_requests_total{{target="{target}",status="timeout"}} {probe["timeout"]}',
        f'jev_shadow_requests_total{{target="{target}",status="unavailable"}} {probe["unavailable"]}',
        f'jev_shadow_requests_total{{target="{target}",status="error"}} {probe["error"]}',
        "# TYPE jev_shadow_low_confidence_total counter",
        f'jev_shadow_low_confidence_total{{target="{target}"}} {probe["low_confidence"]}',
        "# TYPE jev_shadow_skipped_total counter",
        f'jev_shadow_skipped_total{{target="{target}",reason="busy"}} {probe["skipped_busy"]}',
        "# TYPE jev_shadow_latency_ms_total counter",
        f'jev_shadow_latency_ms_total{{target="{target}"}} {_stats["shadow_latency_ms"]:.3f}',
    ]
    for choice, count in sorted(probe["choices"].items()):
        lines.append(f'jev_shadow_choice_total{{target="{target}",choice="{choice}"}} {count}')
    return PlainTextResponse("\n".join(lines) + "\n")


@app.get("/route-only")
async def route_only(query: str = Query(..., min_length=1, max_length=16000)):
    """Route without executing agent (diagnostic mode)."""
    result = await router.route(query)
    return {
        "routing_decision": {
            "strategy": result.routing_decision.strategy.value,
            "confidence_score": result.routing_decision.confidence_score,
            "fast_path_exit": result.routing_decision.fast_path_exit,
        },
        "extracted_metadata": {
            "intent": result.extracted_metadata.intent,
            "keywords": result.extracted_metadata.keywords,
            "entities": result.extracted_metadata.entities,
            "domain": result.extracted_metadata.domain,
        },
        "rag_configuration": {
            "lightrag_required": result.rag_configuration.lightrag_required,
            "lightrag_mode": result.rag_configuration.lightrag_mode,
        },
        "target_agent": result.target_agent.value,
        "elapsed_ms": round(result.elapsed_ms, 2),
        "degraded": result.degraded,
        "fallback_reason": result.fallback_reason,
    }


@app.post("/decision-test", response_model=DecisionTestResponse)
async def decision_test(req: DecisionTestRequest):
    """Diagnostic decision-engine probe; does not affect production routing."""
    result = await decision_engine.decide(
        query=req.query,
        candidates=req.candidates,
        schema_name=req.schema_name,
        context=req.context,
    )
    _stats["decision_engine_requests"] += 1
    _stats["decision_engine_latency_ms"] += result.latency_ms
    if result.status != "success":
        _stats["decision_engine_errors"] += 1
    if result.low_confidence:
        _stats["decision_engine_low_confidence"] += 1
    return DecisionTestResponse(
        choice=result.choice,
        confidence=result.confidence,
        latency_ms=round(result.latency_ms, 2),
        engine=result.engine,
        status=result.status,
        low_confidence=result.low_confidence,
        fallback_reason=result.error,
    )


# ── Main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Jev Multi-Tier Router API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()

    print(f"[Jev] Starting on {args.host}:{args.port}")
    print(f"[Jev] LLM: {LLM_MODEL} @ {LLM_HOST}")
    print(f"[Jev] LightRAG: {LIGHTRAG_API}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
