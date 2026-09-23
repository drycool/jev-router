#!/usr/bin/env python3
"""Index /home/dry/memory/*.md into the Jev Tier-2 FTS5 index.

MVP scope: this only WRITES to the existing FTS5 table (`chunks`) through the
same schema the router already created.  It changes no routing, no gating and no
search code - the memory documents simply become additional rows that BM25 can
find, tagged `entity_type='memory'` and `source=<absolute path>` so they are
identifiable and removable.

Chunking is by Markdown level-2 headings (`## `).  Each chunk is prefixed with
the document title and its heading, so a chunk retrieved on its own still says
what it is about.  Sections longer than --max-chars are split on blank lines.

Idempotent: rows with entity_type='memory' are deleted before inserting, so
repeated runs do not multiply documents.

Known MVP gap (see decisions/ADR-001-memory-foundation.md): the FTS5 table is a
derived index that the server rebuilds from the LightRAG chunk store on every
start, so these rows survive until the next `systemctl --user restart jev`.
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))
MEMORY_DIR = Path(os.getenv("JEV_MEMORY_DIR", "/home/dry/memory"))
ENTITY_TYPE = "memory"


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs, split on level-2 headings."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading or body:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        elif line.startswith("# ") and not heading and not body:
            continue  # the document title is added separately
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, "\n".join(body).strip()))
    return [(h, b) for h, b in sections if b]


def document_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def split_too_long(block: str, max_chars: int) -> list[str]:
    """Split a block on blank lines so no part exceeds max_chars."""
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in re.split(r"\n\s*\n", block):
        if current and size + len(paragraph) + 2 > max_chars:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


def build_chunks(root: Path, max_chars: int) -> list[tuple[str, str, str]]:
    """Return (chunk_id, content, source) for every chunk under root."""
    chunks: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.md")):
        if ".index" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        title = document_title(text, path.stem)
        rel = path.relative_to(root)
        index = 0
        for heading, body in split_sections(text):
            header = f"{title} :: {heading}" if heading else title
            for part in split_too_long(body, max_chars - len(header) - 4):
                content = f"{header}\n\n{part}"[:max_chars]
                chunks.append((f"mem:{rel}#{index}", content, str(path)))
                index += 1
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=MEMORY_DIR)
    parser.add_argument("--db", type=Path, default=Path(FTS5_DB_PATH))
    parser.add_argument("--max-chars", type=int, default=1800,
                        help="cap per chunk (the server indexes LightRAG at 2000)")
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"memory directory not found: {args.root}", file=sys.stderr)
        return 2

    chunks = build_chunks(args.root, args.max_chars)
    files = {}
    for chunk_id, _, source in chunks:
        files[source] = files.get(source, 0) + 1

    print(f"memory root : {args.root}")
    print(f"database    : {args.db}")
    for source, count in sorted(files.items()):
        print(f"  {count:3d} chunks  {source}")
    print(f"total       : {len(chunks)} chunks from {len(files)} files "
          f"({sum(len(c[1]) for c in chunks)} chars)")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    connection = sqlite3.connect(args.db)
    try:
        with connection:  # one transaction: delete + insert is atomic from readers' view
            removed = connection.execute(
                "DELETE FROM chunks WHERE entity_type = ?", (ENTITY_TYPE,)
            ).rowcount
            connection.executemany(
                "INSERT INTO chunks (chunk_id, content, source, entity_type) VALUES (?, ?, ?, ?)",
                [(cid, content, source, ENTITY_TYPE) for cid, content, source in chunks],
            )
        total = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        mine = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE entity_type = ?", (ENTITY_TYPE,)
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"\nremoved previous memory rows : {removed}")
    print(f"inserted                     : {len(chunks)}")
    print(f"index now                    : {total} rows total, {mine} of them memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
