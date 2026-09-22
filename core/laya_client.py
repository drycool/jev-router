"""Optional remote System-1 classifier hosted on GPU2.

The gateway never imports PyTorch or Laya: a slow/unavailable ML service must
not affect its lightweight routing process on the Pi.
"""
import asyncio
import os
import time
from dataclasses import dataclass

import httpx

LAYA_URL = os.getenv("JEV_LAYA_URL", "http://192.168.11.87:8031")
LAYA_TIMEOUT_S = float(os.getenv("JEV_LAYA_TIMEOUT_S", "0.15"))
LAYA_CONFIDENCE_THRESHOLD = float(os.getenv("JEV_LAYA_CONFIDENCE_THRESHOLD", "0.82"))


@dataclass
class LayaDecision:
    strategy: str = "general_fallback"
    domain: str = "general"
    confidence: float = 0.0
    status: str = "disabled"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "domain": self.domain,
            "confidence": self.confidence, "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
        }


class LayaTier1Client:
    """Calls the GPU2 Laya service with a strict total latency budget."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or LAYA_URL).rstrip("/")
        self.timeout_s = LAYA_TIMEOUT_S if timeout_s is None else timeout_s
        self._transport = transport
        self._http: httpx.AsyncClient | None = None

    def _client(self) -> httpx.AsyncClient:
        """One pooled client for the process.

        Opening a fresh connection per call cost ~15 ms of a ~44 ms round trip
        (measured from the Pi), which matters badly against a 50 ms-class budget.
        """
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s), transport=self._transport
            )
        return self._http

    async def aclose(self) -> None:
        """Release the pooled connection (call on shutdown)."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def predict_routing(self, query: str) -> LayaDecision:
        """Ask GPU2 for a routing verdict, within a hard deadline.

        ``timeout_s`` is enforced as a *total* budget via ``asyncio.timeout``, not
        only as an httpx per-operation timeout. httpx read timeouts apply per
        socket read, so a response that trickles in can legitimately return well
        after the configured value — a 50 ms setting was observed returning after
        181 ms. Since this call sits inline in the routing path, the budget has to
        be a real ceiling on what the tier can add to a request.
        """
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_s):
                response = await self._client().post(f"{self.base_url}/predict", json={"query": query})
                response.raise_for_status()
                payload = response.json()
            return LayaDecision(
                strategy=str(payload.get("strategy", "general_fallback")),
                domain=str(payload.get("domain", "general")),
                confidence=float(payload.get("confidence", 0.0)),
                status=str(payload.get("status", "success")),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except (TimeoutError, httpx.TimeoutException):
            return LayaDecision(status="timeout", latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception:
            return LayaDecision(status="unavailable", latency_ms=(time.perf_counter() - t0) * 1000)

