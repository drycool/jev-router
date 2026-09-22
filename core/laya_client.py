"""Optional remote System-1 classifier hosted on GPU2.

The gateway never imports PyTorch or Laya: a slow/unavailable ML service must
not affect its lightweight routing process on the Pi.
"""
import os
import time
from dataclasses import dataclass

import httpx


LAYA_URL = os.getenv("JEV_LAYA_URL", "http://192.168.11.87:8031")
LAYA_TIMEOUT_S = float(os.getenv("JEV_LAYA_TIMEOUT_S", "0.05"))
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
    """Calls the GPU2 Laya service with a strict latency budget."""

    async def predict_routing(self, query: str) -> LayaDecision:
        t0 = time.perf_counter()
        timeout = httpx.Timeout(LAYA_TIMEOUT_S)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{LAYA_URL}/predict", json={"query": query})
                response.raise_for_status()
                payload = response.json()
            return LayaDecision(
                strategy=str(payload.get("strategy", "general_fallback")),
                domain=str(payload.get("domain", "general")),
                confidence=float(payload.get("confidence", 0.0)),
                status=str(payload.get("status", "success")),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.TimeoutException:
            return LayaDecision(status="timeout", latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception:
            return LayaDecision(status="unavailable", latency_ms=(time.perf_counter() - t0) * 1000)
