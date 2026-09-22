"""Optional llama.cpp decision endpoint client.

The decision engine is diagnostic for now: it can evaluate fixed-choice
schemas, but it does not control the production routing path.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx


DECISION_ENGINE_URL = os.getenv("JEV_DECISION_ENGINE_URL", "http://127.0.0.1:8080")
DECISION_ENGINE_TIMEOUT_S = float(os.getenv("JEV_DECISION_ENGINE_TIMEOUT_S", "0.3"))
DECISION_ENGINE_LOW_CONFIDENCE = float(os.getenv("JEV_DECISION_ENGINE_LOW_CONFIDENCE", "0.80"))

SchemaName = Literal["routing_v1", "agent_selection_v1", "rag_mode_v1"]

DECISION_SCHEMAS: dict[str, dict[str, Any]] = {
    "routing_v1": {
        "choice": {
            "type": "choice",
            "options": ["direct_action", "exact_fts", "vector_fast", "graph_lightrag", "general_llm"],
        }
    },
    "agent_selection_v1": {
        "choice": {
            "type": "choice",
            "options": ["general_agent", "code_agent", "db_agent", "troubleshooter_agent"],
        }
    },
    "rag_mode_v1": {
        "choice": {
            "type": "choice",
            "options": ["skip", "local", "global", "hybrid"],
        }
    },
}


@dataclass
class DecisionEngineResult:
    choice: str
    confidence: float
    latency_ms: float
    engine: str = "fallback"
    status: str = "fallback"
    low_confidence: bool = True
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "choice": self.choice,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 2),
            "engine": self.engine,
            "status": self.status,
            "low_confidence": self.low_confidence,
        }
        if self.error:
            payload["error"] = self.error
        if self.raw:
            payload["raw"] = self.raw
        return payload


class DecisionEngineClient:
    """HTTP client for a llama.cpp-style `/v1/decision` endpoint."""

    def __init__(self, base_url: str = DECISION_ENGINE_URL, timeout_s: float = DECISION_ENGINE_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def decide(
        self,
        query: str,
        candidates: list[str],
        schema_name: str = "routing_v1",
        context: str = "",
    ) -> DecisionEngineResult:
        t0 = time.perf_counter()
        candidates = [item for item in candidates if item]
        if not candidates:
            return DecisionEngineResult(
                choice="",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                status="invalid_request",
                error="candidates must not be empty",
            )
        schema = self._schema_with_candidates(schema_name, candidates)
        if schema is None:
            return DecisionEngineResult(
                choice=candidates[0],
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                status="invalid_schema",
                error=f"unknown schema: {schema_name}",
            )

        try:
            timeout = httpx.Timeout(self.timeout_s)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/decision",
                    json={
                        "input": {"query": query, "context": context},
                        "schema": schema,
                        "candidates": candidates,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException as error:
            return self._fallback(candidates[0], t0, "timeout", str(error))
        except Exception as error:
            return self._fallback(candidates[0], t0, "unavailable", str(error))

        choice = str(payload.get("choice") or payload.get("decision") or candidates[0])
        if choice not in candidates:
            choice = candidates[0]
        confidence = self._confidence(payload)
        return DecisionEngineResult(
            choice=choice,
            confidence=confidence,
            latency_ms=(time.perf_counter() - t0) * 1000,
            engine="llama_cpp_decision",
            status="success",
            low_confidence=confidence < DECISION_ENGINE_LOW_CONFIDENCE,
            raw=payload,
        )

    def _schema_with_candidates(self, schema_name: str, candidates: list[str]) -> dict[str, Any] | None:
        schema = DECISION_SCHEMAS.get(schema_name)
        if schema is None:
            return None
        result = {"choice": dict(schema["choice"])}
        allowed = set(result["choice"]["options"])
        result["choice"]["options"] = [item for item in candidates if item in allowed]
        if not result["choice"]["options"]:
            result["choice"]["options"] = candidates
        return result

    def _fallback(self, choice: str, t0: float, status: str, error: str) -> DecisionEngineResult:
        return DecisionEngineResult(
            choice=choice,
            confidence=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            status=status,
            error=error,
        )

    def _confidence(self, payload: dict[str, Any]) -> float:
        if "confidence" in payload:
            return float(payload["confidence"])
        scores = payload.get("scores")
        if isinstance(scores, dict) and scores:
            return max(float(value) for value in scores.values())
        return 0.0
