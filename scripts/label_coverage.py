#!/usr/bin/env python3
"""
Labelling report for the Jev decision log
=========================================

Answers one question honestly: is there enough ground truth to do anything with?

The router's own log cannot answer it. jev_decisions.jsonl records what the router chose
and what it cost - tier, latency, whether an answer came out - and none of that is a
judgement, because the router cannot know whether its answer was right. A label exists only
when a consumer reports one, and those verdicts live in jev_feedback.jsonl, joined by
decision_id.

This script joins the two and reports coverage, which is the number that decides whether
"training" or "calibration" is a real option or a story. It deliberately refuses to report
accuracy: a percentage computed from a handful of self-reported labels would look like a
metric and mean nothing.

Usage
-----
    python3 scripts/label_coverage.py
    python3 scripts/label_coverage.py --min-per-class 20
    python3 scripts/label_coverage.py --json
    python3 scripts/label_coverage.py --show-rejected 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DECISIONS = os.path.join(PROJECT_ROOT, "jev_decisions.jsonl")
DEFAULT_FEEDBACK = os.path.join(PROJECT_ROOT, "jev_feedback.jsonl")

# Who to believe when a decision has been judged more than once. A human verdict replaces an
# agent's; among equal sources the latest one wins. The router's own signal never counts as
# a verdict at all, which is why it does not appear here.
SOURCE_RANK = {"human": 3, "script": 2, "agent": 1}


@dataclass
class Decision:
    decision_id: str
    tier: str
    strategy: str
    schema: str
    timestamp: str
    latency_ms: float
    answered: bool
    preview: str
    verdicts: list[dict] = field(default_factory=list)

    @property
    def effective_verdict(self) -> str | None:
        """The verdict that counts: strongest source first, latest within a source."""
        ruling = self.ruling_verdict
        return ruling.get("verdict") if ruling else None

    @property
    def ruling_verdict(self) -> dict | None:
        """The verdict entry that decides this decision's label."""
        if not self.verdicts:
            return None
        ranked = sorted(
            self.verdicts,
            key=lambda v: (SOURCE_RANK.get(v.get("source", ""), 0), v.get("index", 0)),
        )
        return ranked[-1]

    @property
    def effective_comment(self) -> str:
        """The reason given by whoever issued the ruling verdict.

        Not simply "the first comment": when a human overrides an agent, the human's reason
        is the one that explains the label. Taking the first comment instead attributes the
        agent's optimism to a rejection, which is worse than saying nothing.
        """
        ruling = self.ruling_verdict
        return (ruling or {}).get("comment", "") or ""

    @property
    def labelled_query(self) -> str:
        """The question, if whoever judged attached it.

        The decision log holds only a hash of the query, so an answer preview alone cannot be
        judged: the reader can see what was answered but not what was asked. A label that
        carries its question is self-contained and can be reviewed later without the person
        who made it; one that does not is only as good as the reviewer's memory.
        """
        for verdict in reversed(self.verdicts):
            if verdict.get("query"):
                return verdict["query"]
        return ""


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file, skipping malformed lines instead of dying on them.

    A truncated final line is normal for an append-only log written by a live process, and
    losing the whole report over it would be the wrong trade.
    """
    if not os.path.exists(path):
        return []
    entries: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  ! {os.path.basename(path)}:{number} is not valid JSON; skipped", file=sys.stderr)
    return entries


def load_decisions(path: str) -> list[Decision]:
    decisions = []
    for entry in read_jsonl(path):
        # Only decision records belong here. A v1 line has no schema key; anything else is
        # a different kind of event and must not be counted as a decision.
        if entry.get("schema") not in (None, "decision_v2"):
            continue
        signals = entry.get("signals") or {}
        decision = entry.get("decision") or {}
        execution = entry.get("execution") or {}
        decisions.append(
            Decision(
                decision_id=entry.get("decision_id") or "",
                tier=signals.get("tier", "?"),
                strategy=decision.get("strategy", "?"),
                schema=entry.get("schema", "decision_v1"),
                timestamp=entry.get("timestamp", ""),
                latency_ms=float(execution.get("latency_ms", 0.0)),
                answered=bool(signals.get("answered", False)),
                preview=(signals.get("answer_preview") or signals.get("context_preview") or "")[:200],
            )
        )
    return decisions


def attach_feedback(decisions: list[Decision], path: str) -> tuple[int, int]:
    """Join verdicts onto decisions. Returns (attached, orphaned)."""
    by_id = {d.decision_id: d for d in decisions if d.decision_id}
    attached = orphans = 0

    for index, entry in enumerate(read_jsonl(path)):
        if entry.get("schema") not in (None, "feedback_v1"):
            continue
        verdict = {
            "verdict": entry.get("verdict"),
            "source": entry.get("source", "agent"),
            "comment": entry.get("comment", ""),
            "query": entry.get("query") or "",
            "timestamp": entry.get("timestamp", ""),
            "index": index,
        }
        decision_id = entry.get("decision_id")
        decision = by_id.get(decision_id) if decision_id else None
        if decision is None:
            orphans += 1
            continue
        decision.verdicts.append(verdict)
        attached += 1

    return attached, orphans


def report(decisions: list[Decision], attached: int, orphans: int, min_per_class: int) -> dict[str, Any]:
    labelled = [d for d in decisions if d.effective_verdict]
    unlabelled = [d for d in decisions if not d.effective_verdict]
    verdicts = Counter(d.effective_verdict for d in labelled)

    by_tier: dict[str, Counter] = defaultdict(Counter)
    for decision in decisions:
        by_tier[decision.tier][decision.effective_verdict or "unlabelled"] += 1

    # A decision the router cannot have logged an id for can never be labelled: the verdict
    # has nothing to attach to. Counting these separately keeps the coverage number honest.
    unlabellable = [d for d in decisions if not d.decision_id]

    # Where both an agent and a human judged the same answer, do they agree? This is the
    # only measurement of whether cheap labels are worth collecting at all.
    both = [d for d in decisions if {v["source"] for v in d.verdicts} & {"agent", "human"} == {"agent", "human"}]
    disagreements = [
        d for d in both
        if (agent := [v for v in d.verdicts if v["source"] == "agent"])
        and (human := [v for v in d.verdicts if v["source"] == "human"])
        and agent[-1]["verdict"] != human[-1]["verdict"]
    ]

    coverage = (len(labelled) / len(decisions) * 100) if decisions else 0.0
    thin = {name: count for name, count in verdicts.items() if count < min_per_class}
    self_contained = [d for d in labelled if d.labelled_query]

    return {
        "decisions": len(decisions),
        "schemas": dict(Counter(d.schema for d in decisions)),
        "unlabellable_legacy": len(unlabellable),
        "feedback_entries": attached,
        "feedback_orphans": orphans,
        "labelled": len(labelled),
        "unlabelled": len(unlabelled),
        "coverage_percent": round(coverage, 1),
        "verdicts": dict(verdicts),
        "by_tier": {tier: dict(counts) for tier, counts in sorted(by_tier.items())},
        "judged_by_both": len(both),
        "source_disagreements": len(disagreements),
        "labels_with_query": len(self_contained),
        "thin_classes": thin,
        "min_per_class": min_per_class,
        "labelled_examples": [
            {
                "decision_id": d.decision_id,
                "tier": d.tier,
                "strategy": d.strategy,
                "verdict": d.effective_verdict,
                "comment": d.effective_comment,
                "query": d.labelled_query,
                "preview": d.preview,
            }
            for d in labelled
        ],
    }


def print_report(data: dict[str, Any], decisions: list[Decision], show_rejected: int) -> None:
    print("Jev labelling report")
    print("=" * 60)
    print(f"decisions            {data['decisions']}")
    print(f"  by schema          {data['schemas']}")
    if data["unlabellable_legacy"]:
        print(f"  no decision_id     {data['unlabellable_legacy']}  <- v1 records, unlabellable forever")
    print(f"feedback entries     {data['feedback_entries']}")
    if data["feedback_orphans"]:
        print(f"  orphans            {data['feedback_orphans']}  <- judged an id the log does not have")

    print("\nlabel coverage")
    print(f"  labelled           {data['labelled']} / {data['decisions']}  ({data['coverage_percent']}%)")
    print(f"  unlabelled         {data['unlabelled']}")
    print(f"  effective verdicts {data['verdicts'] or '{}'}")

    print("\nby tier (effective verdict)")
    for tier, counts in data["by_tier"].items():
        print(f"  {tier:<7} {counts}")

    print("\nlabel trust")
    print(f"  judged by both a human and an agent  {data['judged_by_both']}")
    print(f"  of those, they disagreed             {data['source_disagreements']}")
    if data["judged_by_both"] == 0:
        print("  (no overlap yet, so cheap agent labels are unvalidated - they cannot be")
        print("   trusted as a substitute for human ones until some overlap exists)")

    if data["labelled"]:
        print(f"  labels carrying their question       {data['labels_with_query']} / {data['labelled']}")
        if data["labels_with_query"] == 0:
            print("  (the decision log keeps only a query hash, so a label without the question")
            print("   cannot be re-judged by anyone else - pass query= when recording a verdict)")

    print("\ndataset readiness")
    if not data["decisions"]:
        print("  No decisions logged. Nothing is calling the router yet, so there is")
        print("  nothing to label and no point measuring anything else.")
    elif data["labelled"] == 0:
        print(f"  0 labelled decisions out of {data['decisions']}.")
        print("  The verdict channel exists and is wired; it has simply never been used.")
        print("  Until someone reports a verdict, this is a log, not a dataset.")
    elif data["thin_classes"]:
        print(f"  Not usable yet. {data['labelled']} labelled decisions, but these verdict")
        print(f"  classes are below the floor of {data['min_per_class']}: {data['thin_classes']}")
        print("  A class with a handful of examples cannot calibrate a threshold or train")
        print("  anything; it can only mislead. Keep collecting.")
    else:
        print(f"  {data['labelled']} labelled decisions with at least {data['min_per_class']}")
        print("  per verdict class. Enough to look at distributions per tier.")
        print("  Note what this does and does not support: it supports checking whether a")
        print("  tier is worth its latency, and calibrating a threshold. It does not")
        print("  support fine-tuning anything, and a percentage computed from")
        print("  self-reported labels is not an accuracy figure.")

    rejected = [d for d in decisions if d.effective_verdict == "rejected"]
    partial = [d for d in decisions if d.effective_verdict == "partial"]
    for label, group in (("rejected", rejected), ("partial", partial)):
        if not group:
            continue
        shown = group[:show_rejected] if show_rejected else group
        print(f"\n{label} examples ({len(group)} total, showing {len(shown)})")
        for decision in shown:
            comment = decision.effective_comment
            print(f"  [{decision.tier} {decision.strategy}] {decision.decision_id[:12]}")
            if decision.labelled_query:
                print(f"    question: {decision.labelled_query}")
            else:
                print("    question: (not recorded at labelling time)")
            if comment:
                print(f"    reason: {comment}")
            if decision.preview:
                print(f"    answer: {decision.preview[:150]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--feedback", default=DEFAULT_FEEDBACK)
    parser.add_argument("--min-per-class", type=int, default=20,
                        help="verdict classes below this count are reported as too thin to use (default 20)")
    parser.add_argument("--show-rejected", type=int, default=3,
                        help="how many rejected/partial examples to print, 0 for all (default 3)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    decisions = load_decisions(args.decisions)
    attached, orphans = attach_feedback(decisions, args.feedback)
    data = report(decisions, attached, orphans, args.min_per_class)

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_report(data, decisions, args.show_rejected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
