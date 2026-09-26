#!/usr/bin/env python3
"""Add or refresh the vectors of the rows that live only in FTS5.

Two namespaces qualify, and they are the two the server indexes at start: the
working-memory documents (`mem:`) and the imported chat corpus (`gem:`).  The
file name is historical - the memory documents came first - but the command
covers both, because a second near-identical script is a second place for the
dimension check and the corpus-preserving merge to be got wrong.

Incremental by design: the corpus vectors that came from the LightRAG chunk store
are read from the archive and written back untouched, only the selected
namespace's rows are re-embedded.  A full rebuild is the other script's job
(`build_vector_index.py`), and it calls both merges at the end so a rebuild
cannot quietly drop either namespace.

Idempotent: rows are recognised by their chunk-id prefix and replaced, so running
this ten times leaves one copy.  The other namespace is never touched - that is
asserted in tests/test_vector_index.py, because losing it would empty the index
for those queries while still looking healthy.

    python3 scripts/build_memory_vectors.py --dry-run
    python3 scripts/build_memory_vectors.py --namespace corpus
    python3 scripts/build_memory_vectors.py --namespace both
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.env import load_env  # noqa: E402

# Configuration is read at import time by the modules below, so `.env` has to be
# applied before they are imported - otherwise this script would embed against
# the code default while the service embeds against the calibrated value, and
# the two can differ (JEV_EMBEDDING_API points at the CPU instance here).
load_env()

from core.memory_index import ENTITY_TYPE, feed, feeds  # noqa: E402
from core.router import detect_domain  # noqa: E402
from core.vector_index import (  # noqa: E402
    VectorIndexError,
    corpus_rows,
    embed_texts,
    load_index,
    memory_rows_from_fts5,
    merge_namespace,
    reusable_embeddings,
    prune_excluded,
    rows_from_fts5,
    save_index,
)

DEFAULT_INDEX = PROJECT_ROOT / "storage" / "jev_vectors.npz"
DEFAULT_DB = PROJECT_ROOT / "storage" / "jev_fts5.db"

# (namespace, a reader of that feed's FTS5 rows, a merge into the archive, its prefix)
NamespaceSpec = tuple[str, Callable[..., list[tuple[str, str, str]]],
                      Callable[..., tuple[dict, dict]], str]
# (namespace, the rows read, the merge, the prefix) - what a run works from
RowPlan = tuple[str, list[tuple[str, str, str]], Callable[..., tuple[dict, dict]], str]


def _merger(spec) -> Callable[..., tuple[dict, dict]]:
    """A merge for one feed, closing over its id namespace.

    The prefix comes from the registry rather than from the caller, so a run
    cannot replace one feed's vectors with another's content - a prefix that
    matches nothing replaces nothing, which is a silent no-op instead of a silent
    corruption.
    """

    def merge(index, chunks, embeddings, domains=None):
        merged, replaced = merge_namespace(index, chunks, embeddings, domains,
                                           spec.prefix, spec.entity_type)
        return merged, {
            "replaced": replaced,
            "added": len(chunks),
            "total": len(merged["chunk_ids"]),
        }

    return merge


def namespace_plan(name: str) -> NamespaceSpec:
    """How one registered feed is read out of FTS5 and merged into the archive."""
    spec = feed(name)
    reader: Callable[..., list[tuple[str, str, str]]]
    if spec.name == "memory":
        # Memory rows are identified by their tag in FTS5, which is what the
        # indexer writes and what the merge has always used.
        reader = memory_rows_from_fts5
    else:
        reader = lambda db: rows_from_fts5(db, spec.prefix)
    return (spec.name, reader, _merger(spec), spec.prefix)


def report_changes(index: dict, chunks: list[tuple[str, str, str]], prefix: str) -> int:
    """How many of these chunks differ from what the archive already holds."""
    stored = {
        str(cid): str(content)
        for cid, content in zip(index["chunk_ids"], index["contents"])
        if str(cid).startswith(prefix)
    }
    if not stored:
        print(f"  archive holds no rows in this namespace yet: all {len(chunks)} are new")
        return len(chunks)
    changed = sum(1 for cid, content, _ in chunks if stored.get(cid) != content)
    missing = sum(1 for cid, _, _ in chunks if cid not in stored)
    removed = len(set(stored) - {cid for cid, _, _ in chunks})
    print(f"  already present : {len(stored) - changed} unchanged")
    print(f"  changed text    : {changed}")
    print(f"  new             : {missing}")
    print(f"  no longer exists: {removed}")
    return changed + missing + removed


def selected_namespaces(args: argparse.Namespace) -> list[str]:
    if args.namespace == "all":
        return [spec.name for spec in feeds()]
    return [args.namespace]


def read_rows(args: argparse.Namespace, names: list[str]) -> list[RowPlan]:
    """One entry per feed with rows to write: (name, chunks, merge, prefix)."""
    plan = []
    for name in names:
        namespace, reader, merge, prefix = namespace_plan(name)
        chunks = reader(args.db) if args.db.exists() else []
        print(f"{namespace:<9}: {len(chunks)} chunks in FTS5")
        plan.append((namespace, chunks, merge, prefix))
    return plan


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
    print(f"index   : {args.index}")
    if prune_stats["pruned"]:
        sources = ", ".join(f"{name} x{count}"
                            for name, count in prune_stats["sources"].items())
        print(f"excluded: dropped {prune_stats['pruned']} of {total_before} vectors "
              f"({sources}) - JEV_VECTOR_EXCLUDE_SOURCES")
    print(f"  model={index['model']} dimension={index['dimension']} "
          f"rows={len(index['chunk_ids'])} (corpus {corpus_rows(index)})")

    plan = read_rows(args, selected_namespaces(args))
    if not any(chunks for _, chunks, _, _ in plan):
        print("\nnothing to do: the FTS5 table holds no rows in the selected namespaces")
        return 0

    if index["model"] != args.model:
        # The router rejects an index whose model or dimension does not match the
        # configured embedder, so mixing models here would take the whole vector
        # tier offline rather than only the new rows.
        print(f"\nrefusing to merge: the archive was built with {index['model']}, "
              f"this run would embed with {args.model}", file=sys.stderr)
        return 2

    timeout = httpx.Timeout(connect=args.connect_timeout, read=args.read_timeout,
                            write=args.read_timeout, pool=args.connect_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, chunks, merge, prefix in plan:
            if not chunks:
                continue
            print(f"\nchanges against the archive ({name}):")
            report_changes(index, chunks, prefix)

            # Only what changed goes to the embedder.  A daily refresh of a feed with
            # fifteen changed rows must not pay for two and a half thousand unchanged
            # ones: on this host the embedder is a model on another machine that sleeps.
            embeddings, needs = reusable_embeddings(index, chunks, prefix)
            fresh = int(needs.sum())
            print(f"  to embed       : {fresh} rows, {len(chunks) - fresh} reused")
            started = time.perf_counter()
            if fresh:
                new_rows = await embed_texts(
                    client, args.api, args.model,
                    [content for row, (_, content, _) in enumerate(chunks) if needs[row]],
                    batch_size=args.batch_size)
                embeddings[needs] = np.asarray(new_rows, dtype=np.float32)
            elapsed = time.perf_counter() - started

            domains = [detect_domain(f"{source}\n{content}") for _, content, source in chunks]
            index, stats = merge(index, chunks, embeddings, domains)
            save_index(args.index, index)

            print(f"\nembedded {len(chunks)} chunks in {elapsed * 1000:.0f} ms "
                  f"({elapsed * 1000 / len(chunks):.0f} ms/chunk)")
            for key, value in stats.items():
                print(f"{key:<21}: {value}")
            print(f"domains              : "
                  f"{json.dumps({d: domains.count(d) for d in sorted(set(domains))})}")

    print(f"\nwrote {args.index}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--namespace", choices=[spec.name for spec in feeds()] + ["all"],
                        default="memory",
                        help="which feed's vectors to refresh (default memory; 'all' for every feed)")
    parser.add_argument("--api", default=os.getenv("JEV_EMBEDDING_API",
                                                   "http://192.168.11.87:11434/api/embed"))
    parser.add_argument("--model", default=os.getenv("JEV_EMBEDDING_MODEL", "bge-m3"))
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
    print(f"index   : {args.index}")
    print(f"  model={index['model']} dimension={index['dimension']} "
          f"rows={len(index['chunk_ids'])} (corpus {corpus_rows(index)})")
    for name, chunks, _, prefix in read_rows(args, selected_namespaces(args)):
        print(f"\nchanges against the archive ({name}):")
        report_changes(index, chunks, prefix)
    print("\n[dry-run] no embeddings requested, nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
