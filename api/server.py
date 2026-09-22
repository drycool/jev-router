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

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.decision_engine import (
    DECISION_ENGINE_LOW_CONFIDENCE,
    DECISION_ENGINE_URL,
    DECISION_SCHEMAS,
    DecisionEngineClient,
)
from core.laya_client import LAYA_CONFIDENCE_THRESHOLD
from core.router import AgentType, JevRouter, RoutingResult, Strategy
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
    decision_engine_requests: int
    decision_engine_errors: int
    decision_engine_low_confidence: int
    decision_engine_latency_ms: float


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
    "decision_engine_requests": 0,
    "decision_engine_errors": 0,
    "decision_engine_low_confidence": 0,
    "decision_engine_latency_ms": 0.0,
}


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
        _stats["laya_predictions"] += 1
        if (
            result.laya_result.get("status") == "success"
            and result.laya_result.get("confidence", 0) >= LAYA_CONFIDENCE_THRESHOLD
        ):
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
        decision_engine_requests=_stats["decision_engine_requests"],
        decision_engine_errors=_stats["decision_engine_errors"],
        decision_engine_low_confidence=_stats["decision_engine_low_confidence"],
        decision_engine_latency_ms=round(_stats["decision_engine_latency_ms"], 2),
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
    from fastapi.responses import PlainTextResponse
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
