#!/usr/bin/env python3
"""Benchmark the GPU2 Laya System-1 service on a labeled fixture.

This measures the tier that is actually wired into production routing, and it
measures it the way the router uses it: domain and strategy separately, latency
per call, and how often the calibrated confidence clears the acceptance
threshold.

The English checkpoint is at the repo root; ``multilingual`` is a bundled
subfolder.  Because querying only one of them silently produces a wrong answer
for the other language, the benchmark can force each one per request and report
them side by side:

    python3 scripts/benchmark_laya.py                                  # routed
    python3 scripts/benchmark_laya.py --models english multilingual    # A/B

``--include-baseline`` also scores the *local* regex domain classifier
(``core.router.detect_domain``) on the same fixture, which is the honest
comparison: the tier Laya is meant to replace.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "decision_queries.txt"
DEFAULT_URL = "http://192.168.11.87:8031"
STRATEGIES = ("direct_cmd", "database_search", "graph_lightrag", "complex_llm")
DOMAINS = ("raspberry_pi", "automotive", "general")


@dataclass
class FixtureItem:
    query: str
    expected_strategy: str | None = None
    expected_domain: str | None = None
    line_number: int = 0


@dataclass
class ModelReport:
    model: str
    items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def answered(self) -> list[dict[str, Any]]:
        return [row for row in self.items if row["status"] == "success"]

    def accuracy(self, field_name: str) -> tuple[int, int]:
        """(correct, scored) over answered rows that carry an expectation."""
        key = f"expected_{field_name}"
        graded = [row for row in self.answered if row.get(key)]
        correct = [row for row in graded if row.get(field_name) == row[key]]
        return len(correct), len(graded)

    def summary(self) -> dict[str, Any]:
        answered = self.answered
        latencies = [row["latency_ms"] for row in answered]
        confidences = sorted(row["confidence"] for row in answered)
        statuses: dict[str, int] = {}
        for row in self.items:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        checkpoints: dict[str, int] = {}
        for row in answered:
            name = row.get("checkpoint") or "unknown"
            checkpoints[name] = checkpoints.get(name, 0) + 1
        strategy_correct, strategy_scored = self.accuracy("strategy")
        domain_correct, domain_scored = self.accuracy("domain")
        return {
            "model": self.model,
            "total": len(self.items),
            "answered": len(answered),
            "statuses": statuses,
            "strategy_accuracy": _ratio(strategy_correct, strategy_scored),
            "strategy_scored": strategy_scored,
            "domain_accuracy": _ratio(domain_correct, domain_scored),
            "domain_scored": domain_scored,
            "confidence_mean": round(statistics.fmean(confidences), 4) if confidences else None,
            "confidence_median": round(statistics.median(confidences), 4) if confidences else None,
            "confidence_max": round(confidences[-1], 4) if confidences else None,
            "accepted_ge_threshold": None,  # filled in by the caller (threshold is CLI-level)
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "checkpoints": checkpoints,
            "domain_confusion": _confusion([row for row in answered if row.get("expected_domain")], "domain"),
        }


def load_fixture(path: Path) -> list[FixtureItem]:
    items: list[FixtureItem] = []
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [part.strip() for part in line.split("\t")]
            query = parts[0]
            if not query:
                continue
            expected_strategy = _clean(parts[1]) if len(parts) > 1 else None
            expected_domain = _clean(parts[2]) if len(parts) > 2 else None
            if expected_strategy and expected_strategy not in STRATEGIES:
                raise SystemExit(f"{path}:{number}: unknown strategy label {expected_strategy!r}")
            if expected_domain and expected_domain not in DOMAINS:
                raise SystemExit(f"{path}:{number}: unknown domain label {expected_domain!r}")
            items.append(FixtureItem(query, expected_strategy, expected_domain, number))
    return items


def _clean(value: str) -> str | None:
    value = value.strip()
    return None if value in ("", "-") else value


async def probe_model(
    client: httpx.AsyncClient,
    item: FixtureItem,
    url: str,
    model: str,
    repeat: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": item.query}
    if model:
        payload["model"] = model
    latencies: list[float] = []
    last: dict[str, Any] = {}
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        try:
            response = await client.post(f"{url}/predict", json=payload)
            response.raise_for_status()
            last = response.json()
        except Exception as error:
            last = {"status": "client_error", "error": f"{type(error).__name__}: {error}"}
        latencies.append((time.perf_counter() - started) * 1000)
        if last.get("status") != "success":
            break  # repeats measure latency, not retries
    return {
        "query": item.query,
        "line": item.line_number,
        "expected_strategy": item.expected_strategy,
        "expected_domain": item.expected_domain,
        "strategy": last.get("strategy"),
        "domain": last.get("domain"),
        "confidence": float(last.get("confidence", 0.0) or 0.0),
        "status": str(last.get("status", "client_error")),
        "checkpoint": last.get("checkpoint"),
        "routing_reason": last.get("routing_reason"),
        "latency_ms": round(statistics.fmean(latencies), 2),
        "latency_ms_min": round(min(latencies), 2),
        "error": last.get("error"),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    items = load_fixture(args.fixture)
    if not items:
        raise SystemExit(f"no benchmarkable queries in {args.fixture}")

    models = args.models or [""]
    timeout = httpx.Timeout(args.timeout)
    reports: dict[str, ModelReport] = {}
    baseline: dict[str, Any] | None = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.warmup:
            try:
                response = await client.post(f"{args.url}/warmup")
                print(f"[warmup] {response.json()}")
            except Exception as error:
                print(f"[warmup] failed: {type(error).__name__}: {error}")

        try:
            health = (await client.get(f"{args.url}/health")).json()
            print(f"[engine] {health.get('engine')} loaded={health.get('loaded')}")
        except Exception as error:
            raise SystemExit(f"cannot reach the Laya service at {args.url}: {error}")

        for model in models:
            label = model or "routed"
            report = ModelReport(model=label)
            for item in items:
                row = await probe_model(client, item, args.url, model, args.repeat)
                report.items.append(row)
            reports[label] = report
            print(f"[{label}] {len(report.answered)}/{len(report.items)} answered")

    summaries = []
    for report in reports.values():
        summary = report.summary()
        answered = report.answered
        summary["accepted_ge_threshold"] = _ratio(
            sum(1 for row in answered if row["confidence"] >= args.threshold), len(answered)
        )
        summaries.append(summary)

    if args.include_baseline:
        baseline = _baseline(items)

    result = {
        "url": args.url,
        "fixture": str(args.fixture),
        "threshold": args.threshold,
        "repeat": args.repeat,
        "models": summaries,
        "baseline_regex_domain": baseline,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for report in reports.values():
            for row in report.items:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_report(result, reports, items)
    return result


def _baseline(items: list[FixtureItem]) -> dict[str, Any]:
    """Score the local regex domain classifier on the same fixture.

    Only the domain is comparable: Laya's strategy labels are its own, while
    ``detect_domain`` is a keyword pre-filter that never claims a strategy.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from core.router import detect_domain  # imported late so --help stays fast

    graded = [item for item in items if item.expected_domain]
    correct = [item for item in graded if detect_domain(item.query) == item.expected_domain]
    confusion = _confusion(
        [
            {"expected_domain": item.expected_domain, "domain": detect_domain(item.query)}
            for item in graded
        ],
        "domain",
    )
    return {
        "name": "regex detect_domain (local tier)",
        "domain_accuracy": _ratio(len(correct), len(graded)),
        "domain_scored": len(graded),
        "domain_confusion": confusion,
    }


def _print_report(result: dict[str, Any], reports: dict[str, ModelReport], items: list[FixtureItem]) -> None:
    threshold = result["threshold"]
    print()
    print(f"{'model':<14}{'answ':>7}{'strat':>9}{'domain':>9}{'conf med':>10}{'≥thr':>7}{'p50 ms':>9}{'p95 ms':>9}")
    print("-" * 74)
    for summary in result["models"]:
        print(
            f"{summary['model']:<14}"
            f"{summary['answered']:>4}/{summary['total']:<2}"
            f"{_fmt_ratio(summary['strategy_accuracy']):>9}"
            f"{_fmt_ratio(summary['domain_accuracy']):>9}"
            f"{_fmt_num(summary['confidence_median']):>10}"
            f"{_fmt_ratio(summary['accepted_ge_threshold']):>7}"
            f"{_fmt_num(summary['latency_ms_p50']):>9}"
            f"{_fmt_num(summary['latency_ms_p95']):>9}"
        )
    if result.get("baseline_regex_domain"):
        base = result["baseline_regex_domain"]
        print(
            f"{'regex local':<14}{'-':>7}{'-':>9}"
            f"{_fmt_ratio(base['domain_accuracy']):>9}{'-':>10}{'-':>7}{'-':>9}{'-':>9}"
        )
    print(f"\nconfidence threshold = {threshold}; latency measured over the service's own /predict")

    for summary in result["models"]:
        if summary["answered"] < summary["total"]:
            print(f"\n! {summary['model']}: {summary['total'] - summary['answered']} call(s) did not answer "
                  f"({summary['statuses']}) — scored accuracy covers only answered rows.")
        confusion = summary.get("domain_confusion") or {}
        if confusion:
            print(f"\ndomain confusion [{summary['model']}] (expected -> got)")
            _print_confusion(confusion)

    errors = [row for report in reports.values() for row in report.items if row["status"] != "success"]
    if errors:
        print("\nfailed calls:")
        for row in errors[:20]:
            print(f"  line {row['line']}: {row['status']} — {row['error'] or ''}")

    _print_disagreements(reports, items)


def _print_disagreements(reports: dict[str, ModelReport], items: list[FixtureItem]) -> None:
    """Show every item the model got wrong; a silent summary would hide the pattern."""
    for label, report in reports.items():
        wrong = [
            row
            for row in report.answered
            if (row.get("expected_strategy") and row["strategy"] != row["expected_strategy"])
            or (row.get("expected_domain") and row["domain"] != row["expected_domain"])
        ]
        if not wrong:
            continue
        print(f"\ndisagreements [{label}] ({len(wrong)} of {len(report.answered)} answered):")
        for row in wrong:
            bits = []
            if row.get("expected_strategy") and row["strategy"] != row["expected_strategy"]:
                bits.append(f"strategy {row['strategy']} (expected {row['expected_strategy']})")
            if row.get("expected_domain") and row["domain"] != row["expected_domain"]:
                bits.append(f"domain {row['domain']} (expected {row['expected_domain']})")
            print(f"  conf={row['confidence']:.3f} {'; '.join(bits)}\n    {row['query']}")


def _confusion(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for row in rows:
        expected = row.get(f"expected_{key}")
        if not expected or row.get("status") == "client_error":
            continue
        predicted = str(row.get(key))
        table.setdefault(str(expected), {})
        table[str(expected)][predicted] = table[str(expected)].get(predicted, 0) + 1
    return table


def _print_confusion(table: dict[str, dict[str, int]]) -> None:
    for expected in sorted(table):
        row = table[expected]
        total = sum(row.values())
        correct = row.get(expected, 0)
        detail = ", ".join(f"{name}={count}" for name, count in sorted(row.items(), key=lambda kv: -kv[1]))
        print(f"  {expected:<14} {correct}/{total}  [{detail}]")


def _ratio(correct: int, total: int) -> float | None:
    return round(correct / total, 4) if total else None


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return round(ordered[index], 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help="Laya service base URL")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=Path("laya_benchmark.jsonl"))
    parser.add_argument("--models", nargs="*", default=[], help="force checkpoints, e.g. english multilingual (default: routed)")
    parser.add_argument("--repeat", type=int, default=3, help="calls per query, for latency stability")
    parser.add_argument("--threshold", type=float, default=0.80, help="confidence acceptance threshold to report against")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="skip POST /warmup")
    parser.add_argument("--include-baseline", action="store_true", help="also score the local regex domain classifier")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
