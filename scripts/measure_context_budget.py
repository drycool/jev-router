#!/usr/bin/env python3
"""Measure what context assembly actually hands the agent, on the real index.

Written to make the dedup and the character budget falsifiable. It exists because the
defect it measures was invisible: the router answered, the answer looked plausible, and
the only clue was the model remarking that its own context was truncated. Nothing in the
logs said how many chunks arrived, how many of them were copies, or how much text the
agent was actually charged for.

Offline mode (default) reads the production FTS5 index directly - no server restart, no
GPU, no LLM - and reports, per query:

    pool     results returned by search_fts5
    exact    results that pass the exact-hit test (what the tier-2 branch would use)
    uniq     distinct texts in that pool
    dup      rows dropped as byte-identical copies
    old      characters the previous `[:3]` slice delivered
    new      characters assemble_context delivers under the current budget
    used     unique chunks that fitted / dropped for size

--agent adds an A/B of the prefill cost: the same query executed by the real agent at two
budgets. That number is the price of the budget, and it is the reason the budget is a
number in .env rather than a hard-coded constant.

Usage:
    python3 scripts/measure_context_budget.py
    python3 scripts/measure_context_budget.py --query "порядок регулировки зазоров клапанов"
    python3 scripts/measure_context_budget.py --agent --budgets 2500,6000
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.router import (  # noqa: E402
    MAX_CONTEXT_CHARS,
    RETRIEVAL_LIMIT,
    Tier2Search,
    _is_fts_exact,
    assemble_context,
)

DEFAULT_QUERIES = [
    "порядок регулировки зазоров клапанов",
    "момент затяжки головки блока цилиндров",
    "как снять распределительный вал",
    "давление в системе смазки двигателя",
]


def measure(search: Tier2Search, query: str, budget: int) -> dict:
    pool = search.search_fts5(query)
    exact = [r for r in pool if _is_fts_exact(query, r)]

    old_chars = len("\n\n".join(r["content"] for r in exact[:3]))
    new_context, stats = assemble_context(exact, budget=budget)

    return {
        "query": query,
        "pool": len(pool),
        "exact": len(exact),
        "pool_chars": sum(len(r["content"]) for r in exact),
        "duplicate": stats["chunks_duplicate"],
        "old_chars": old_chars,
        "new_chars": stats["chars"],
        "used": stats["chunks_used"],
        "dropped": stats["chunks_too_large"],
        "budget_exhausted": stats["budget_exhausted"],
        "context": new_context,
    }


def offline(queries: list[str], budget: int) -> list[dict]:
    search = Tier2Search()
    try:
        return [measure(search, q, budget) for q in queries]
    finally:
        search.close()


async def agent_ab(queries: list[str], budgets: list[int]) -> None:
    """Time the real agent at each budget, twice, to expose order effects.

    The first call of the session pays for loading the model on GPU2, so it is a warm-up
    and its number is reported separately rather than averaged in.
    """
    from core.router import AgentType, JevRouter
    # The LLM endpoint lives in the server's config, not the router's: the router is
    # transport-agnostic and the agent objects are wired up by whichever process hosts it.
    from api.server import LLM_HOST, LLM_MODEL

    router = JevRouter()
    # JevRouter builds the executor but does not populate it; the server registers the
    # agents in its lifespan. Without this the executor returns the string
    # "No agent registered for ..." and every timing below would be a measurement of
    # nothing - which is exactly what the first version of this script reported.
    from agents.base import GeneralAgent

    router.tier4.register_agent(AgentType.GENERAL, GeneralAgent(LLM_HOST, LLM_MODEL))
    query = queries[0]

    print(f"\n=== agent A/B (query: {query!r}) ===")
    print("warm-up call (pays for model load, not comparable):")
    context, _ = assemble_context(
        [r for r in router.tier2.search_fts5(query) if _is_fts_exact(query, r)],
        budget=budgets[-1],
    )
    await _run_agent(router, query, context, label="warm-up")

    # Each budget twice, in both orders: a single pass cannot tell a real prefill
    # difference from GPU2 getting warmer.
    for budget in list(budgets) + list(reversed(budgets)):
        pool = router.tier2.search_fts5(query)
        context, stats = assemble_context(
            [r for r in pool if _is_fts_exact(query, r)], budget=budget
        )
        print(
            f"\nbudget={budget}  chars={stats['chars']}  chunks={stats['chunks_used']}"
        )
        await _run_agent(router, query, context, label=f"budget={budget}")

    await router.laya.aclose()
    router.tier2.close()


async def _run_agent(router, query: str, context: str, label: str) -> None:
    from core.router import (
        AgentType,
        ExtractedMetadata,
        RAGConfiguration,
        RoutingDecision,
        RoutingResult,
        Strategy,
    )

    result = RoutingResult(
        RoutingDecision(Strategy.EXACT_FTS, 0.95, False),
        ExtractedMetadata(intent="exact_search"),
        RAGConfiguration(),
        AgentType.GENERAL,
        query=query,
        context=context,
    )
    t0 = time.perf_counter()
    try:
        response = await router.tier4.execute(result)
        elapsed = (time.perf_counter() - t0) * 1000
        answer = getattr(response, "answer", response)
        print(
            f"  {label:14s} {elapsed:8.0f} ms   chars_in={len(context):5d} "
            f"chars_out={len(answer):5d}"
        )
    except Exception as exc:  # noqa: BLE001 - a failed probe must not hide the table
        print(f"  {label:14s} FAILED: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--budget", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument(
        "--budgets",
        default="",
        help="comma-separated budgets for --agent, e.g. 2500,6000",
    )
    parser.add_argument("--agent", action="store_true", help="add the LLM prefill A/B")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES
    rows = offline(queries, args.budget)

    print(
        f"budget = {args.budget} chars (JEV_MAX_CONTEXT_CHARS)   "
        f"pool = {RETRIEVAL_LIMIT} results (JEV_RETRIEVAL_LIMIT)\n"
    )
    header = f"{'query':44s} {'pool':>4s} {'exact':>5s} {'uniq':>4s} {'dup':>3s} {'old':>6s} {'new':>6s} {'used':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        used = f"{row['used']}/{row['used'] + row['dropped']}"
        print(
            f"{row['query'][:44]:44s} {row['pool']:4d} {row['exact']:5d} "
            f"{row['exact'] - row['duplicate']:4d} {row['duplicate']:3d} "
            f"{row['old_chars']:6d} {row['new_chars']:6d} {used:>9s}"
        )

    total_old = sum(r["old_chars"] for r in rows)
    total_new = sum(r["new_chars"] for r in rows)
    print(f"\ntotal: old {total_old} chars -> new {total_new} chars")
    print(
        "  `old` is the previous [:3] slice; rows with dup>0 are the reason `new` is "
        "larger without asking for more of the corpus."
    )
    budgets_note = "budget was the binding constraint" if any(
        r["budget_exhausted"] for r in rows
    ) else "pool, not the budget, was the binding constraint"
    print(f"  {budgets_note}; raise JEV_MAX_CONTEXT_CHARS only if chunks are being dropped.")

    if args.agent:
        budgets = [int(b) for b in args.budgets.split(",") if b.strip()] or [args.budget]
        asyncio.run(agent_ab(queries, budgets))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
