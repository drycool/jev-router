#!/usr/bin/env python3
"""Index the imported chat corpus into Jev's Tier-2 FTS5 index (CLI).

The chunking and the write live in `core/memory_index.py`, because the service
runs exactly the same code on every start (`api/server.py::_index_corpus_docs`).
A second chunker would drift from the one the service uses, so there is one
implementation and this is its command line.

When to use it: a dry run after an import (`scripts/import_chat_export.py`), or
on a machine where the service is not running.  On a machine where the service
runs, the start already does this - look for the
`[Jev] Indexed N chat-corpus chunks ...` line in the journal.
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

# Before the import below: JEV_CORPUS_DIR and JEV_FTS5_DB_PATH are read at import
# time, so a `.env` that names another directory has to be applied first.
load_env()

from core.memory_index import (  # noqa: E402
    CORPUS_CHUNK_PREFIX,
    CORPUS_DIR,
    DEFAULT_MAX_CHARS,
    build_chunks,
    index_corpus,
)

FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(CORPUS_DIR),
                        help=f"chat corpus directory (default {CORPUS_DIR}, env JEV_CORPUS_DIR)")
    parser.add_argument("--db", type=Path, default=Path(FTS5_DB_PATH))
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="cap per chunk (the server indexes LightRAG at 2000)")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"corpus directory not found: {args.root}", file=sys.stderr)
        print("import something first: scripts/import_chat_export.py --match <words>",
              file=sys.stderr)
        return 2

    print(f"corpus root : {args.root}")
    print(f"database    : {args.db}")

    if args.dry_run:
        chunks = build_chunks(args.root, args.max_chars, CORPUS_CHUNK_PREFIX)
        per_file: dict[str, int] = {}
        for _, _, source in chunks:
            per_file[source] = per_file.get(source, 0) + 1
        for source, count in sorted(per_file.items()):
            print(f"  {count:4d} chunks  {Path(source).name}")
        print(f"total       : {len(chunks)} chunks from {len(per_file)} files "
              f"({sum(len(c[1]) for c in chunks)} chars)")
        print("\n[dry-run] nothing written")
        return 0

    connection = sqlite3.connect(args.db)
    try:
        stats = index_corpus(connection, args.root, args.max_chars)
        total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        mine = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE ?", (f"{CORPUS_CHUNK_PREFIX}%",)
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"removed previous corpus rows : {stats['removed']}")
    print(f"inserted                     : {stats['inserted']}"
          f" from {stats['files']} files ({stats['chars']} chars)")
    print(f"index now                    : {total} rows total, {mine} of them chat corpus")
    print("\nvectors are a separate step: scripts/build_memory_vectors.py --namespace corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
