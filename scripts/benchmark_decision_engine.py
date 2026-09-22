#!/usr/bin/env python3
"""Benchmark the diagnostic `/decision-test` endpoint.

Inputs:
- A JSONL decision log with `raw_query` fields.
- Or a plain text file with one query per line.

The script writes JSONL results and prints a compact summary. It deliberately
uses `/decision-test`, so production `/query` routing is not affected.
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_CANDIDATES = ["exact_fts", "vector_fast", "graph_lightrag", "general_llm"]


@dataclass
class BenchmarkItem:
    query: str
    source: str
    expected_strategy: str | None = None


def load_queries(log_path: Path | None = None, queries_path: Path | None = None) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    if log_path:
        items.extend(_load_log_queries(log_path))
    if queries_path:
        items.extend(_load_text_queries(queries_path))
    return items


def _load_log_queries(path: Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = event.get("raw_query")
            if not query:
                continue
            decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
            items.append(
                BenchmarkItem(
                    query=str(query),
                    source=f"{path}:{line_number}",
                    expected_strategy=decision.get("strategy"),
                )
            )
    return items


def _load_text_queries(path: Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            query = line.strip()
            if not query or query.startswith("#"):
                continue
            items.append(BenchmarkItem(query=query, source=f"{path}:{line_number}"))
    return items


async def run_benchmark(
    items: list[BenchmarkItem],
    endpoint: str,
    output: Path,
    candidates: list[str],
    schema_name: str,
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(timeout_s)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            _probe(client, semaphore, item, endpoint, candidates, schema_name)
            for item in items
        ]
        results = await asyncio.gather(*tasks)

    with output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    return summarize(results, total_elapsed_ms=(time.perf_counter() - started) * 1000)


async def _probe(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    item: BenchmarkItem,
    endpoint: str,
    candidates: list[str],
    schema_name: str,
) -> dict[str, Any]:
    request = {
        "query": item.query,
        "candidates": candidates,
        "schema": schema_name,
    }
    t0 = time.perf_counter()
    async with semaphore:
        try:
            response = await client.post(endpoint, json=request)
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status", "unknown"))
            error = payload.get("fallback_reason")
        except Exception as exc:
            payload = {}
            status = "client_error"
            error = str(exc)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    choice = payload.get("choice")
    expected = item.expected_strategy
    return {
        "query": item.query,
        "source": item.source,
        "expected_strategy": expected,
        "choice": choice,
        "matches_expected": expected is not None and choice == expected,
        "confidence": float(payload.get("confidence", 0.0) or 0.0),
        "latency_ms": round(float(payload.get("latency_ms", elapsed_ms) or elapsed_ms), 2),
        "client_elapsed_ms": round(elapsed_ms, 2),
        "engine": payload.get("engine", "client"),
        "status": status,
        "low_confidence": bool(payload.get("low_confidence", True)),
        "error": error,
    }


def summarize(results: list[dict[str, Any]], total_elapsed_ms: float = 0.0) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in results]
    labeled = [row for row in results if row.get("expected_strategy")]
    matches = [row for row in labeled if row.get("matches_expected")]
    statuses: dict[str, int] = {}
    for row in results:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "total": len(results),
        "labeled": len(labeled),
        "accuracy": round(len(matches) / len(labeled), 4) if labeled else None,
        "success": statuses.get("success", 0),
        "fallback_or_error": len(results) - statuses.get("success", 0),
        "low_confidence": sum(1 for row in results if row.get("low_confidence")),
        "avg_latency_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "statuses": statuses,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSONL decision log with raw_query fields")
    parser.add_argument("--queries", type=Path, help="Plain text file with one query per line")
    parser.add_argument("--output", type=Path, default=Path("decision_benchmark.jsonl"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8030/decision-test")
    parser.add_argument("--schema", default="routing_v1")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input and not args.queries:
        raise SystemExit("provide --input jev_decisions.jsonl or --queries queries.txt")
    items = load_queries(args.input, args.queries)
    if not items:
        raise SystemExit("no benchmarkable queries found; JSONL logs need raw_query entries")
    summary = asyncio.run(
        run_benchmark(
            items=items,
            endpoint=args.endpoint,
            output=args.output,
            candidates=args.candidates,
            schema_name=args.schema,
            concurrency=max(1, args.concurrency),
            timeout_s=args.timeout,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
