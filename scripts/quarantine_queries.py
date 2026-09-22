#!/usr/bin/env python3
"""
Quarantine decision records produced by test fixtures
=====================================================

A test that exercises /query appends a real record to the decision log, because the
endpoint cannot tell a fixture from traffic. That happened here: 20 of 71 records (28%)
were the string "Raspberry Pi cable" from ShadowProbeTests, all with empty keywords and
entities because the fixture is degenerate.

The root cause is fixed (tests/__init__.py redirects the log paths), but the records it
already wrote are still sitting in the log, and every statistic computed over that log -
including a "58 records over 28.6 hours" figure used to argue that labelling was a distant
prospect - was inflated by them. This moves them out.

Matching is by sha256 of the query text, which is how the log stores queries. That means
only queries whose exact text is known can be quarantined; a record cannot be classified as
synthetic from its shape alone, and guessing from shape would risk discarding real traffic.

Usage
-----
    # see what would move (default is a dry run)
    python3 scripts/quarantine_queries.py --query "Raspberry Pi cable"

    # do it
    python3 scripts/quarantine_queries.py --query "Raspberry Pi cable" --apply

    # several fixtures, or a file of them, one per line
    python3 scripts/quarantine_queries.py --query-file /path/to/fixture-queries.txt --apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG = os.path.join(PROJECT_ROOT, "jev_decisions.jsonl")


def hashes_for(queries: list[str]) -> dict[str, str]:
    """Map each query's sha256 to the query text, for reporting."""
    return {hashlib.sha256(q.encode("utf-8")).hexdigest(): q for q in queries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default=DEFAULT_LOG, help="decision log to filter")
    parser.add_argument("--query", action="append", default=[], help="a fixture query to quarantine (repeatable)")
    parser.add_argument("--query-file", help="file of fixture queries, one per line")
    parser.add_argument("--apply", action="store_true", help="write the change; without it this is a dry run")
    args = parser.parse_args()

    queries = list(args.query)
    if args.query_file:
        with open(args.query_file, encoding="utf-8") as handle:
            queries.extend(line.strip() for line in handle if line.strip())

    if not queries:
        print("nothing to do: pass --query or --query-file", file=sys.stderr)
        return 2

    if not os.path.exists(args.log):
        print(f"{args.log} does not exist", file=sys.stderr)
        return 1

    wanted = hashes_for(queries)

    keep: list[str] = []
    move: list[str] = []
    kept_unparsed = 0

    with open(args.log, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                # An unreadable line is kept: this tool removes provable fixtures, and a
                # corrupted line is not provably anything.
                keep.append(stripped)
                kept_unparsed += 1
                continue
            if event.get("query_hash") in wanted:
                move.append(stripped)
            else:
                keep.append(stripped)

    print(f"log            {args.log}")
    print(f"fixture hashes {len(wanted)}")
    print(f"records total  {len(keep) + len(move)}")
    print(f"to quarantine  {len(move)}")
    print(f"to keep        {len(keep)}")
    if kept_unparsed:
        print(f"  of which {kept_unparsed} unparseable and kept as-is")

    if not move:
        print("\nnothing matched; no change")
        return 0

    if not args.apply:
        print("\ndry run - nothing was written. Re-run with --apply to move these records.")
        return 0

    quarantine_path = f"{args.log}.quarantine"
    stamp = datetime.now(timezone.utc).isoformat()
    with open(quarantine_path, "a", encoding="utf-8") as handle:
        for line in move:
            handle.write(json.dumps({"quarantined_at": stamp, "reason": "test fixture query", "record": json.loads(line)}, ensure_ascii=False) + "\n")

    # Keep a backup before rewriting: this is the only copy of the log's history.
    backup_path = f"{args.log}.bak-{stamp[:19].replace(':', '')}"
    shutil.copy2(args.log, backup_path)

    tmp_path = f"{args.log}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for line in keep:
            handle.write(line + "\n")
    os.replace(tmp_path, args.log)

    print(f"\nmoved {len(move)} records to {quarantine_path}")
    print(f"backup          {backup_path}")
    print(f"remaining       {len(keep)} records in {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
