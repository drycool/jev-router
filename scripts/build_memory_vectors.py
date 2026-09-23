#!/usr/bin/env python3
"""Add or refresh the memory documents' vectors in storage/jev_vectors.npz.

Incremental by design: the 4562 corpus vectors are read from the archive and
written back untouched, only the `mem:` rows are re-embedded.  A full rebuild of
the corpus is the other script's job (`build_vector_index.py`), and it calls the
same merge at the end so a rebuild cannot quietly drop memory.

Idempotent: memory rows are recognised by their `mem:` chunk-id prefix and
replaced, so running this ten times leaves one copy.  Corpus rows are never
touched - that is asserted in tests/test_vector_index.py, because losing them
would empty the index for every corpus query while still looking healthy.

    python3 scripts/build_memory_vectors.py --dry-run     # no GPU, no write
    python3 scripts/build_memory_vectors.py               # refresh the archive
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.router import detect_domain  # noqa: E402
from core.vector_index import (  # noqa: E402
    VectorIndexError,
    corpus_rows,
    embed_texts,
    load_index,
    memory_rows_from_fts5,
    merge_memory,
    prune_excluded,
    save_index,
)

DEFAULT_INDEX = PROJECT_ROOT / "storage" / "jev_vectors.npz"
DEFAULT_DB = PROJECT_ROOT / "storage" / "jev_fts5.db"


def report_changes(index: dict, chunks: list[tuple[str, str, str]]) -> int:
    """How many memory chunks differ from what the archive already holds."""
    stored = {
        str(cid): str(content)
        for cid, content in zip(index["chunk_ids"], index["contents"])
        if str(cid).startswith("mem:")
    }
    if not stored:
        print(f"  archive holds no memory vectors yet: all {len(chunks)} are new")
        return len(chunks)
    changed = sum(1 for cid, content, _ in chunks if stored.get(cid) != content)
    missing = sum(1 for cid, _, _ in chunks if cid not in stored)
    removed = len(set(stored) - {cid for cid, _, _ in chunks})
    print(f"  already present : {len(stored) - changed} unchanged")
    print(f"  changed text    : {changed}")
    print(f"  new             : {missing}")
    print(f"  no longer exists: {removed}")
    return changed + missing + removed


async def build(args: argparse.Namespace) -> int:
    if not args.index.exists():
        print(f"no vector index at {args.index}; run scripts/build_vector_index.py first",
              file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"no FTS5 database at {args.db} - start the router once to build it",
              file=sys.stderr)
        return 2

    index = load_index(args.index)
    total_before = len(index["chunk_ids"])
    index, prune_stats = prune_excluded(index)
    chunks = memory_rows_from_fts5(args.db)
    print(f"index   : {args.index}")
    if prune_stats["pruned"]:
        sources = ", ".join(f"{name} x{count}"
                            for name, count in prune_stats["sources"].items())
        print(f"excluded: dropped {prune_stats['pruned']} of {total_before} vectors "
              f"({sources}) - JEV_VECTOR_EXCLUDE_SOURCES")
    print(f"  model={index['model']} dimension={index['dimension']} "
          f"rows={len(index['chunk_ids'])} (corpus {corpus_rows(index)})")
    print(f"memory  : {len(chunks)} chunks from FTS5")
    if not chunks:
        print("\nnothing to do: the FTS5 table holds no memory rows")
        return 0

    if index["model"] != args.model:
        # The router rejects an index whose model or dimension does not match the
        # configured embedder, so mixing models here would take the whole vector
        # tier offline rather than only the new rows.
        print(f"\nrefusing to merge: the archive was built with {index['model']}, "
              f"this run would embed with {args.model}", file=sys.stderr)
        return 2

    print("\nchanges against the archive:")
    report_changes(index, chunks)

    timeout = httpx.Timeout(connect=args.connect_timeout, read=args.read_timeout,
                            write=args.read_timeout, pool=args.connect_timeout)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        embeddings = await embed_texts(client, args.api, args.model,
                                       [content for _, content, _ in chunks],
                                       batch_size=args.batch_size)
    elapsed = time.perf_counter() - started

    domains = [detect_domain(f"{source}\n{content}") for _, content, source in chunks]
    merged, stats = merge_memory(index, chunks, embeddings, domains)
    save_index(args.index, merged)

    print(f"\nembedded {len(chunks)} chunks in {elapsed * 1000:.0f} ms "
          f"({elapsed * 1000 / len(chunks):.0f} ms/chunk)")
    print(f"corpus vectors kept   : {stats['corpus']}")
    print(f"memory rows replaced  : {stats['memory_replaced']}")
    print(f"memory rows written   : {stats['memory_added']}")
    print(f"total vectors         : {stats['total']}")
    print(f"domains               : "
          f"{json.dumps({d: domains.count(d) for d in sorted(set(domains))})}")
    print(f"wrote {args.index}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--api", default=os.getenv("JEV_EMBEDDING_API",
                                                   "http://192.168.11.87:11434/api/embed"))
    parser.add_argument("--model", default=os.getenv("JEV_EMBEDDING_MODEL", "mxbai-embed-large"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        return _dry(args)
    try:
        return asyncio.run(build(args))
    except VectorIndexError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1


def _dry(args: argparse.Namespace) -> int:
    """Dry run: read both sides and report the delta without touching the embedder."""
    if not args.index.exists():
        print(f"no vector index at {args.index}", file=sys.stderr)
        return 2
    index = load_index(args.index)
    chunks = memory_rows_from_fts5(args.db) if args.db.exists() else []
    print(f"index   : {args.index}")
    print(f"  model={index['model']} dimension={index['dimension']} "
          f"rows={len(index['chunk_ids'])} (corpus {corpus_rows(index)})")
    print(f"memory  : {len(chunks)} chunks in FTS5")
    print("\nchanges against the archive:")
    report_changes(index, chunks)
    print("\n[dry-run] no embeddings requested, nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
