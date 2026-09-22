#!/usr/bin/env python3
"""
Labelling CLI: judge router answers and record the verdict
==========================================================

The gap this fills. `jev_feedback` is an MCP tool, so it is reachable by an agent but
awkward for the person sitting at the terminal who owns the labels: they would have to find
a decision_id in one file, remember the question, and hand-write a JSON body. Measured
conclusion from the design work: the bottleneck on ground truth is not the volume of the log,
it is the reviewer's attention. So the reviewer gets a tool that shows a queue and takes one
keystroke per answer.

What the reviewer sees, and the one thing the log cannot give them: the decision record keeps
only a *hash* of the query, so it shows what was answered but not what was asked. This CLI
asks for the question at the moment of judging, when the reviewer still has it, and attaches
it to the verdict - keeping raw query text out of the log by default while making the label
self-contained for whoever reads the dataset later.

Usage
-----
    python3 scripts/label.py                     # interactive queue, newest first
    python3 scripts/label.py --list 20           # just show the queue
    python3 scripts/label.py --stats             # coverage, label trust, readiness
    python3 scripts/label.py --id <decision_id> --verdict rejected \\
        --comment "wrong torque spec" --query "какой момент затяжки..." --source human

Verdicts describe the answer, not the router: accepted = usable as given, partial = needed
correction or more work, rejected = wrong. There is no "unknown": an abstention carries no
signal and the router rejects it anyway.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import label_coverage  # noqa: E402  (sibling script, reused so the two tools cannot disagree)

DEFAULT_URL = os.getenv("JEV_URL", "http://127.0.0.1:8030")

# One keystroke per answer. Skipping is first-class: a reviewer who is unsure should skip,
# not guess - a guessed label is worse than a missing one, because it looks like data.
KEY_TO_VERDICT = {"a": "accepted", "r": "rejected", "p": "partial"}
SKIP_KEYS = {"s", ""}
QUIT_KEYS = {"q"}


def parse_verdict_key(key: str) -> str | None:
    """Map a keystroke to a verdict. Returns None for skip, and raises for quit."""
    normalised = key.strip().lower()
    if normalised in QUIT_KEYS:
        raise KeyboardInterrupt
    if normalised in SKIP_KEYS:
        return None
    return KEY_TO_VERDICT.get(normalised)


def record(
    decision_id: str,
    verdict: str,
    *,
    comment: str = "",
    query: str = "",
    source: str = "human",
    url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict:
    """POST one verdict to the router. Raises RuntimeError with the response body on failure."""
    payload = {
        "decision_id": decision_id,
        "verdict": verdict,
        "source": source,
        "comment": comment,
    }
    if query.strip():
        payload["query"] = query.strip()

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/feedback",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"router rejected the verdict: HTTP {error.code} {error.read().decode('utf-8', 'replace')[:300]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"router unreachable at {url}: {error.reason}") from error


def load_state(decisions_path: str, feedback_path: str) -> list[label_coverage.Decision]:
    """Load decisions *with* their verdicts already joined.

    The join is not optional and belongs here rather than at any call site: without it every
    decision looks unlabelled, so the queue invites the reviewer to re-judge work already
    done and writes a second verdict that contradicts the first - corrupting the very dataset
    this tool exists to protect. That is precisely what happened before this function
    existed, which is why the joining is a named step with a test that goes through real
    files rather than an in-memory list.
    """
    decisions = label_coverage.load_decisions(decisions_path)
    label_coverage.attach_feedback(decisions, feedback_path)
    return decisions


def pending(decisions: list[label_coverage.Decision]) -> list[label_coverage.Decision]:
    """Decisions that can still be labelled and have not been, newest first.

    v1 records have no decision_id, so no verdict can ever attach to them; listing them would
    be an invitation to waste attention on records that cannot be labelled.
    """
    return [d for d in reversed(decisions) if d.decision_id and not d.verdicts]


def show_queue(decisions: list[label_coverage.Decision], limit: int) -> None:
    total = len(decisions)
    unlabellable = [d for d in decisions if not d.decision_id]
    print(f"decisions {total} · can be labelled {total - len(unlabellable)} · awaiting a verdict {len(pending(decisions))}")
    if unlabellable:
        print(f"  ({len(unlabellable)} v1 records have no decision_id and can never be labelled)")

    queue = pending(decisions)[:limit] if limit else pending(decisions)
    if not queue:
        print("\nnothing awaiting a verdict")
        return

    print()
    for index, decision in enumerate(queue, 1):
        print(f"[{index}/{len(queue)}] {decision.decision_id}")
        print(f"  when      {decision.timestamp}")
        print(f"  route     {decision.tier} / {decision.strategy} · {decision.latency_ms:.0f} ms · "
              f"answered={decision.answered}")
        if decision.preview:
            preview = decision.preview.replace("\n", " ")[:220]
            print(f"  answer    {preview}")
        print()


def run_interactive(decisions: list[label_coverage.Decision], url: str, limit: int) -> int:
    queue = pending(decisions)
    if limit:
        queue = queue[:limit]
    if not queue:
        print("nothing awaiting a verdict — every labellable decision has one")
        return 0

    print(f"{len(queue)} decisions awaiting a verdict. "
          f"[a]ccepted [r]ejected [p]artial [s]kip [q]uit\n")

    labelled = 0
    for index, decision in enumerate(queue, 1):
        print("─" * 68)
        print(f"[{index}/{len(queue)}] {decision.decision_id}")
        print(f"  when      {decision.timestamp}")
        print(f"  route     {decision.tier} / {decision.strategy} · {decision.latency_ms:.0f} ms · "
              f"answered={decision.answered}")
        if decision.preview:
            print(f"  answer    {decision.preview.replace(chr(10), ' ')[:400]}")

        try:
            key = input("\nverdict > ")
            verdict = parse_verdict_key(key)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped")
            break

        if verdict is None:
            print("  skipped (a guessed label is worse than a missing one)")
            continue

        try:
            query = input("question (Enter if not to hand) > ")
            comment = input("reason (Enter to skip) > ")
        except (KeyboardInterrupt, EOFError):
            print("\nstopped before recording")
            break

        try:
            result = record(decision.decision_id, verdict, comment=comment, query=query, url=url)
        except RuntimeError as error:
            print(f"  NOT recorded: {error}")
            continue

        labelled += 1
        note = f" (verdict #{result['verdicts_for_decision']}"
        if result.get("previous_verdict"):
            note += f", replaces {result['previous_verdict']!r}"
        note += ")"
        if not result.get("known_decision"):
            note += " WARNING: the router does not know this decision id"
        print(f"  recorded: {result['verdict']} by {result['source']}{note}")

    print("─" * 68)
    print(f"{labelled} recorded this session")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decisions", default=label_coverage.DEFAULT_DECISIONS)
    parser.add_argument("--feedback", default=label_coverage.DEFAULT_FEEDBACK)
    parser.add_argument("--url", default=DEFAULT_URL, help="Jev base URL (default %(default)s)")
    parser.add_argument("--list", type=int, default=0, metavar="N",
                        help="print the N oldest-awaiting decisions and exit (0 = all)")
    parser.add_argument("--stats", action="store_true", help="print the coverage report and exit")
    parser.add_argument("--id", dest="decision_id", help="record a verdict non-interactively")
    parser.add_argument("--verdict", choices=sorted(KEY_TO_VERDICT.values()))
    parser.add_argument("--comment", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--source", default="human", choices=["human", "script", "agent"])
    args = parser.parse_args()

    if args.stats:
        return subprocess.call([sys.executable, os.path.join(HERE, "label_coverage.py"),
                                "--decisions", args.decisions, "--feedback", args.feedback])

    decisions = load_state(args.decisions, args.feedback)

    if args.decision_id:
        if not args.verdict:
            print("--id requires --verdict (accepted|rejected|partial)", file=sys.stderr)
            return 2
        try:
            result = record(args.decision_id, args.verdict, comment=args.comment,
                            query=args.query, source=args.source, url=args.url)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"recorded: {result['verdict']} by {result['source']} for {result['decision_id']}"
              f" (verdict #{result['verdicts_for_decision']}, known={result['known_decision']})")
        return 0

    if args.list:
        show_queue(decisions, args.list)
        return 0

    return run_interactive(decisions, args.url, limit=0)


if __name__ == "__main__":
    sys.exit(main())
