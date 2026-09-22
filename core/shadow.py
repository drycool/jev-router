"""Fire-and-forget shadow probes for candidate decision engines.

Shadow mode answers one question without risking production: *what would this
engine have said, and how fast?*  A probe is dispatched as a background task and
never influences the response the caller receives; the result only lands in the
decision log and in Prometheus counters.  That makes it safe to collect honest
latency/confidence statistics for a candidate engine on real traffic before
anything is allowed to steer routing.

Two targets are supported:

  ``laya``      ``POST {url}/predict``  the GPU2 System-1 classifier
  ``decision``  ``POST {url}/v1/decision``  a llama.cpp-style decision endpoint

The probe is deliberately fail-quiet: a shadow engine that is slow, down or
returning nonsense must cost the production path nothing but a skipped counter.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Literal

import httpx

ShadowTarget = Literal["off", "laya", "decision"]

LAYA_PATH = "/predict"
DECISION_PATH = "/v1/decision"

# The routing labels a Laya shadow probe may return; anything else is a probe
# failure rather than a usable observation.
LAYA_STRATEGIES = {"direct_cmd", "database_search", "graph_lightrag", "complex_llm"}
LAYA_DOMAINS = {"raspberry_pi", "automotive", "general"}


class ShadowProbe:
    """Runs candidate-engine probes off the request path and counts the results."""

    def __init__(
        self,
        target: ShadowTarget = "off",
        url: str = "",
        timeout_s: float = 1.0,
        max_inflight: int = 1,
        sample_rate: float = 1.0,
        low_confidence_threshold: float = 0.80,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.target: ShadowTarget = target
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_inflight = max(1, int(max_inflight))
        self.sample_rate = min(1.0, max(0.0, sample_rate))
        self.low_confidence_threshold = low_confidence_threshold
        self.on_result = on_result
        self._transport = transport

        self._tasks: set[asyncio.Task] = set()
        self._inflight = 0
        self._latency_ms: list[float] = []
        self._counters: dict[str, int] = {
            "requests": 0,
            "success": 0,
            "timeout": 0,
            "unavailable": 0,
            "error": 0,
            "low_confidence": 0,
            "skipped_busy": 0,
            "skipped_sampled_out": 0,
            "skipped_no_loop": 0,
        }
        self._choices: dict[str, int] = {}

    # ── public API ────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self.target != "off" and bool(self.url)

    def submit(self, query: str, *, context: str = "") -> bool:
        """Queue a probe. Returns True when a task was actually scheduled.

        Never raises: shadow work must not be able to fail a production request.
        """
        try:
            if not self.enabled or not query:
                return False
            if self.sample_rate < 1.0 and random.random() >= self.sample_rate:
                self._counters["skipped_sampled_out"] += 1
                return False
            if self._inflight >= self.max_inflight:
                # Saturation is expected on bursty traffic: dropping the probe is
                # the correct trade for a diagnostic-only signal.
                self._counters["skipped_busy"] += 1
                return False
            task = asyncio.get_running_loop().create_task(self._probe(query, context))
        except Exception:
            self._counters["skipped_no_loop"] += 1
            return False
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def drain(self) -> None:
        """Wait for in-flight probes to finish (never cancels them)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel in-flight probes (called on server shutdown)."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def stats(self) -> dict[str, Any]:
        latencies = self._latency_ms
        return {
            "target": self.target,
            "url": self.url or None,
            "enabled": self.enabled,
            "requests": self._counters["requests"],
            "success": self._counters["success"],
            "timeout": self._counters["timeout"],
            "unavailable": self._counters["unavailable"],
            "error": self._counters["error"],
            "low_confidence": self._counters["low_confidence"],
            "skipped_busy": self._counters["skipped_busy"],
            "skipped_kind": {
                key.removeprefix("skipped_"): value
                for key, value in self._counters.items()
                if key.startswith("skipped_")
            },
            "inflight": self._inflight,
            "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95),
            "samples": len(latencies),
            "choices": dict(sorted(self._choices.items())),
        }

    # ── internals ─────────────────────────────────────────────────────────
    async def _probe(self, query: str, context: str) -> None:
        self._inflight += 1
        started = time.perf_counter()
        status = "error"
        confidence: float | None = None
        choice: str | None = None
        detail: dict[str, Any] = {}
        try:
            timeout = httpx.Timeout(self.timeout_s)
            async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
                url, payload = self._request(query, context)
                response = await client.post(url, json=payload)
                response.raise_for_status()
                status, confidence, choice, detail = self._parse(response.json())
        except httpx.TimeoutException as error:
            status = "timeout"
            detail = {"error": str(error)}
        except Exception as error:  # never propagate: shadow work is diagnostic
            status = "unavailable"
            detail = {"error": f"{type(error).__name__}: {error}"}
        finally:
            self._inflight -= 1

        latency_ms = (time.perf_counter() - started) * 1000
        self._counters["requests"] += 1
        self._counters[status if status in self._counters else "error"] += 1
        low_confidence = confidence is not None and confidence < self.low_confidence_threshold
        if status == "success":
            self._latency_ms.append(latency_ms)
            del self._latency_ms[:-200]
            if low_confidence:
                self._counters["low_confidence"] += 1
            if choice:
                self._choices[choice] = self._choices.get(choice, 0) + 1

        if self.on_result is not None:
            try:
                self.on_result(
                    {
                        "target": self.target,
                        "status": status,
                        "confidence": confidence,
                        "choice": choice,
                        "low_confidence": low_confidence,
                        "latency_ms": round(latency_ms, 2),
                        **detail,
                    }
                )
            except Exception:
                pass

    def _request(self, query: str, context: str) -> tuple[str, dict[str, Any]]:
        if self.target == "laya":
            return f"{self.url}{LAYA_PATH}", {"query": query}
        return (
            f"{self.url}{DECISION_PATH}",
            {"input": {"query": query, "context": context}, "schema": {"choice": {"type": "choice"}}},
        )

    def _parse(self, payload: dict[str, Any]) -> tuple[str, float | None, str | None, dict[str, Any]]:
        if self.target == "laya":
            strategy = str(payload.get("strategy", ""))
            domain = str(payload.get("domain", ""))
            status = str(payload.get("status", "success"))
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            detail = {"domain": domain, "checkpoint": payload.get("checkpoint")}
            if status != "success" or strategy not in LAYA_STRATEGIES:
                # An unavailable engine answers with a fallback label; counting it
                # as a successful observation would flatter the benchmark.
                return "unavailable", None, None, detail
            if domain not in LAYA_DOMAINS:
                return "error", None, None, detail
            return "success", confidence, strategy, detail
        choice = payload.get("choice") or payload.get("decision")
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        return ("success", confidence, str(choice) if choice else None, {"engine": payload.get("engine")})


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return round(ordered[index], 2)
