#!/usr/bin/env python3
"""Index the working-memory directory into Jev's Tier-2 FTS5 index (CLI).

The chunking and the write live in `core/memory_index.py`, because the service
runs exactly the same code on every start
(`api/server.py::_index_memory_docs`).  Two chunkers would drift apart, so there
is one implementation and this is its command line.

When to use it: a dry run before changing a document, per-file statistics, or
indexing on a machine where the service is not running.  On a machine where the
service runs, the start already does this - see the `[Jev] Indexed N memory
chunks ...` line in the journal.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.memory_index import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    MEMORY_DIR,
    build_chunks,
    index_memory,
)

FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(MEMORY_DIR),
                        help=f"memory directory (default {MEMORY_DIR}, env JEV_MEMORY_DIR)")
    parser.add_argument("--db", type=Path, default=Path(FTS5_DB_PATH))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="cap per chunk (the server indexes LightRAG at 2000)")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"memory directory not found: {args.root}", file=sys.stderr)
        return 2

    print(f"memory root : {args.root}")
    print(f"database    : {args.db}")

    if args.dry_run:
        chunks = build_chunks(args.root, args.max_chars)
        per_file: dict[str, int] = {}
        for _, _, source in chunks:
            per_file[source] = per_file.get(source, 0) + 1
        for source, count in sorted(per_file.items()):
            print(f"  {count:3d} chunks  {source}")
        print(f"total       : {len(chunks)} chunks from {len(per_file)} files "
              f"({sum(len(c[1]) for c in chunks)} chars)")
        print("\n[dry-run] nothing written")
        return 0

    connection = sqlite3.connect(args.db)
    try:
        stats = index_memory(connection, args.root, args.max_chars)
        total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        mine = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE entity_type = 'memory'"
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"removed previous memory rows : {stats['removed']}")
    print(f"inserted                     : {stats['inserted']}"
          f" from {stats['files']} files ({stats['chars']} chars)")
    print(f"index now                    : {total} rows total, {mine} of them memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
