#!/usr/bin/env python3
"""Measure what the GPU2 System-1 classifier adds to a routing decision.

Run this before and after changing where the classifier sits in the pipeline.
Run it from the repository root:

    python3 scripts/measure_route_latency.py

The point is to separate the two costs the classifier imposes:

  * ``classifier_ms`` - the GPU2 round trip itself, as reported by the client;
  * the total ``route()`` time for a query the local index can answer exactly.

If the classifier sits on the blocking path, the second number contains the
first one. Nothing in the local path needs that round trip to succeed.
"""
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.router as router_mod  # noqa: E402
from core.router import JevRouter  # noqa: E402

# Two shapes of traffic: one the local FTS index can answer exactly, and one
# that has to fall through to the expensive tiers.
EXACT_QUERY = "Raspberry Pi cable"
EXACT_DOC = "Raspberry Pi cable cable cable GPIO pinout expansion header"
MISS_QUERY = "quantum flux capacitor calibration drift"

ITERATIONS = 30


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


async def _run(router: JevRouter, query: str) -> None:
    await router.route(query)  # warm the process, discard
    totals: list[float] = []
    classifier_ms: list[float] = []
    statuses: dict[str, int] = {}
    result = None
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        result = await router.route(query)
        totals.append((time.perf_counter() - started) * 1000)
        laya = result.laya_result or {}
        status = str(laya.get("status", "absent"))
        statuses[status] = statuses.get(status, 0) + 1
        if laya.get("latency_ms"):
            classifier_ms.append(float(laya["latency_ms"]))

    print(f"\n{query!r}")
    print(f"  strategy            : {result.routing_decision.strategy.value}")
    print(f"  route() p50 / p95 / max ms : "
          f"{statistics.median(totals):.2f} / {_percentile(totals, 0.95):.2f} / {max(totals):.2f}")
    if classifier_ms:
        print(f"  classifier_ms p50 / max    : "
              f"{statistics.median(classifier_ms):.2f} / {max(classifier_ms):.2f}")
    else:
        print("  classifier_ms              : (never surfaced to the caller)")
    print(f"  laya status counts  : {statuses}")
    print(f"  laya_result         : {result.laya_result}")


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        router_mod.FTS5_DB_PATH = str(Path(directory) / "index.db")
        router = JevRouter()
        router.tier2.index_chunk("one", EXACT_DOC)
        router.tier2.commit()

        # Isolate the classifier's cost from the embedding server and LightRAG,
        # both of which are unrelated to where the classifier sits.
        async def no_embedding(_):
            return None

        async def no_graph(*_, **__):
            return {"response": ""}

        router.get_embedding = no_embedding
        router.tier3.search = no_graph

        await _run(router, EXACT_QUERY)
        # Let anything the previous phase left in flight finish, so a slow phase
        # cannot be blamed on the one before it.
        await asyncio.sleep(1.0)
        await _run(router, MISS_QUERY)
        router.tier2.close()


if __name__ == "__main__":
    asyncio.run(main())
