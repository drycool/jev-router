#!/usr/bin/env python3
"""Index one or all registered feeds into Jev's Tier-2 FTS5 index (CLI).

The chunking and the write live in `core/memory_index.py`, because the service
runs exactly the same code on every start (`api/server.py::_index_feeds`).  A
second chunker would drift from the one the service uses, so there is one
implementation and this is its command line.

When to use it: right after a collector has written a feed (`jev-collect`), or on
a machine where the service is not running.  On a machine where the service runs,
a restart already does this - look for the `[Jev] feed <name> (<prefix>): ...`
lines in the journal.

    python3 scripts/index_feeds.py                       # every feed
    python3 scripts/index_feeds.py --feed projects       # just one
    python3 scripts/index_feeds.py --dry-run             # report, write nothing
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.env import load_env  # noqa: E402

# Before the import below: feed directories and the database path are read at
# import time, so a `.env` that points elsewhere has to be applied first.
load_env()

from core.memory_index import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    build_chunks,
    feeds,
    index_feed,
)

FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feed", action="append", default=[],
                        help="feed name (repeatable, comma separated); default every feed")
    parser.add_argument("--db", type=Path, default=Path(FTS5_DB_PATH))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    wanted = [name.strip() for raw in args.feed for name in raw.split(",") if name.strip()]
    known = {spec.name: spec for spec in feeds()}
    if wanted:
        unknown = [name for name in wanted if name not in known]
        if unknown:
            print(f"unknown feed(s): {', '.join(unknown)}; known: {', '.join(known)}",
                  file=sys.stderr)
            return 2
        selected = [known[name] for name in wanted]
    else:
        selected = list(known.values())

    print(f"database : {args.db}")
    if args.dry_run:
        for spec in selected:
            root = Path(spec.directory)
            if not root.is_dir():
                print(f"  {spec.name:<9} {spec.prefix:<5} {root} — нет каталога")
                continue
            chunks = build_chunks(root, args.max_chars, spec.prefix)
            files = len({source for _, _, source in chunks})
            chars = sum(len(content) for _, content, _ in chunks)
            print(f"  {spec.name:<9} {spec.prefix:<5} {len(chunks):5d} чанков из {files} файлов "
                  f"({chars} символов)  {root}")
        print("\n[dry-run] ничего не записано")
        return 0

    connection = sqlite3.connect(args.db)
    try:
        for spec in selected:
            stats = index_feed(connection, spec.name)
            if not stats["exists"]:
                print(f"  {spec.name:<9} {spec.prefix:<5} {stats['root']} — каталога нет, пропущено")
                continue
            print(f"  {spec.name:<9} {spec.prefix:<5} {stats['inserted']:5d} чанков из "
                  f"{stats['files']} файлов (заменено {stats['removed']}, {stats['chars']} символов)")
        total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        by_prefix = {
            prefix: connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE ?", (f"{prefix}%",)
            ).fetchone()[0]
            for prefix in {spec.prefix for spec in selected}
        }
    finally:
        connection.close()

    print(f"\nв индексе всего: {total} строк")
    for prefix, count in sorted(by_prefix.items()):
        print(f"  {prefix:<5} {count}")
    print("\nвектора — отдельный шаг: python3 scripts/build_memory_vectors.py --namespace all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
