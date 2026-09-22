#!/usr/bin/env python3
"""Read jev_decisions.jsonl and report how the classifier actually behaves per checkpoint.

The threshold on the preset signal (`task`) can only be set from data, and that data
must be split by checkpoint: Cyrillic traffic goes to `multilingual`, Latin script to
`english`, and the two need not report confidences on the same scale. Pooling them into
one number is how a threshold ends up either never firing or firing on the wrong model.

Usage:

    python3 scripts/analyze_laya_verdicts.py                       # default log
    python3 scripts/analyze_laya_verdicts.py --log other.jsonl
    python3 scripts/analyze_laya_verdicts.py --threshold 0.88

Privacy: the decision log stores query hashes, not queries, so this reads metadata only.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "jev_decisions.jsonl"


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def load(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  (skipping unparseable line {lineno})")
    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the classifier's verdict distribution per checkpoint."
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="the gate being evaluated (default 0.90)")
    args = parser.parse_args()

    events = load(args.log)
    print(f"log        : {args.log}")
    print(f"events     : {len(events)}")

    laya_events = [e.get("laya_result") or {} for e in events if e.get("laya_result")]
    if not laya_events:
        print("\nNo routing events carry a classifier result yet — nothing to measure.")
        print("Run traffic through /query (or /route-only), then re-run this script.")
        return

    statuses = Counter(str(v.get("status", "absent")) for v in laya_events)
    print(f"with verdict: {len(laya_events)}")
    print("\nstatus breakdown")
    for status, count in statuses.most_common():
        note = {
            "not_awaited": "local path answered; the classifier was cancelled",
            "success": "a verdict was returned",
            "timeout": "exceeded JEV_LAYA_TIMEOUT_S",
            "unavailable": "service unreachable or errored",
        }.get(status, "")
        print(f"  {status:<12} {count:>6}   {note}")

    verdicts = [v for v in laya_events if str(v.get("status")) == "success"]
    if not verdicts:
        print("\nNo successful verdicts yet, so no confidence distribution to read.")
        print("The 'not_awaited' share is itself the finding: those requests were")
        print("answered locally and never needed the classifier.")
        return

    # The preset signal is the only gated one; the invented labels are reported
    # alongside purely so the contrast stays visible.
    groups: dict[str, list[dict]] = defaultdict(list)
    for verdict in verdicts:
        groups[str(verdict.get("checkpoint") or "unknown")].append(verdict)

    print(f"\ngate: task_confidence >= {args.threshold}\n")
    header = (f"{'checkpoint':<14}{'n':>5}{'p50':>8}{'p90':>8}{'max':>8}"
              f"{'>=gate':>9}{'invented p50':>14}{'p50 ms':>9}")
    print(header)
    print("-" * len(header))
    for checkpoint in sorted(groups, key=lambda c: -len(groups[c])):
        rows = groups[checkpoint]
        task_conf = [float(v.get("task_confidence") or 0.0) for v in rows]
        invented = [float(v.get("confidence") or 0.0) for v in rows]
        latency = [float(v.get("latency_ms") or 0.0) for v in rows]
        passing = sum(value >= args.threshold for value in task_conf)
        print(f"{checkpoint:<14}{len(rows):>5}"
              f"{statistics.median(task_conf):>8.3f}{percentile(task_conf, 0.9):>8.3f}"
              f"{max(task_conf):>8.3f}{passing / len(rows) * 100:>8.0f}%"
              f"{statistics.median(invented):>14.3f}{statistics.median(latency):>9.1f}")

    print("\npreset `task` labels seen, per checkpoint")
    for checkpoint in sorted(groups):
        labels = Counter(str(v.get("task")) for v in groups[checkpoint])
        rendered = ", ".join(f"{label} x{count}" for label, count in labels.most_common())
        print(f"  {checkpoint:<14} {rendered}")

    print("\nreading this table")
    print("  - a checkpoint whose >=gate share is ~0 makes the tier observational:")
    print("    the threshold is not calibrated, it simply never fires. Raise the sample")
    print("    size before lowering it, and never merge the two checkpoints into one")
    print("    number when their scales differ.")
    print("  - 'invented p50' is the out-of-distribution confidence kept for contrast:")
    print("    it can sit high while the model is wrong, which is why it is not gated.")


if __name__ == "__main__":
    main()
