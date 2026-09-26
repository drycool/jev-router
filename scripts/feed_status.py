#!/usr/bin/env python3
"""What each feed holds right now (CLI): chunks, vectors, lag, and what was skipped.

`jev-feed status` is this script.  It exists because the question "is the base up to
date" has four different answers depending on which layer failed: a collector that did not
run, a feed that indexed but was never embedded (the vector step is deliberately separate,
because the embedder lives on a machine that sleeps), a manifest that reports an error or
a deliberate skip, and a corpus that is simply larger than yesterday.

    python3 scripts/feed_status.py
    python3 scripts/feed_status.py --json

Nothing here writes: the table is read from the FTS5 table, the vector archive and the
`feed.json` manifests, in that order of authority - the index is what the router serves.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.memory_index import feeds  # noqa: E402
from core.router import FTS5_DB_PATH  # noqa: E402
from core.vector_index import load_index  # noqa: E402

# Chunks the router knows about but that carry no feed prefix: the OCR'd Espero manual is
# in FTS5 and excluded from the vector index on purpose (JEV_VECTOR_EXCLUDE_SOURCES), so
# its lag is a policy and not a backlog.
UNPREFIXED = "(без префикса — корпус/мануал)"


def feed_prefixes() -> list[str]:
    """Every registered feed prefix, taken from the registry rather than spelled out here.

    This list used to be five literal ``not like`` clauses.  A sixth feed would then have
    been counted as corpus and its backlog reported as a policy - the same defect this
    project already paid for twice (a hardcoded id format drifting from its definition).
    """
    return [spec.prefix for spec in feeds()]


def prefix_of(chunk_id: str) -> str:
    """The feed prefix a stored chunk id carries, or `UNPREFIXED` if it carries none."""
    head = str(chunk_id).split(":", 1)[0] + ":" if ":" in str(chunk_id)[:12] else ""
    return head if head in feed_prefixes() else UNPREFIXED


def count_corpus_chunks(connection: sqlite3.Connection, prefixes: Sequence[str]) -> int:
    """Chunks in FTS5 that belong to no feed: the OCR'd manual and any older corpus."""
    if not prefixes:
        return connection.execute("select count(*) from chunks").fetchone()[0]
    clause = " and ".join("chunk_id not like ?" for _ in prefixes)
    return connection.execute(
        f"select count(*) from chunks where {clause}",
        tuple(f"{prefix}%" for prefix in prefixes),
    ).fetchone()[0]


def manifest_summary(directory: str) -> dict:
    path = Path(directory) / "feed.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"error": f"манифест нечитаем: {error}"}
    sources = manifest.get("sources") or {}
    notes = []
    for name, state in sorted(sources.items()):
        if state.get("error"):
            notes.append(f"{name}: ОШИБКА {state['error'][:60]}")
        if state.get("skip_reason"):
            notes.append(f"{name}: пропущен — {state['skip_reason'][:60]}")
        if state.get("history_skip_reason"):
            notes.append(f"{name}: без истории — {state['history_skip_reason'][:50]}")
        if state.get("bots_skipped"):
            notes.append(f"{name}: ботов {state['bots_skipped']}")
        if state.get("reasoning_chars_dropped"):
            notes.append(f"{name}: рассуждений отброшено {state['reasoning_chars_dropped']:,}")
    return {
        "updated_at": manifest.get("updated_at", ""),
        "documents": len(manifest.get("documents") or []),
        "sources": len(sources),
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="машиночитаемый вывод")
    args = parser.parse_args()

    db_path = Path(FTS5_DB_PATH)
    if not db_path.exists():
        print(f"нет FTS5-таблицы {db_path} — запустить роутер хотя бы раз", file=sys.stderr)
        return 2

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    index = load_index(PROJECT_ROOT / "storage" / "jev_vectors.npz")

    vector_counts: dict[str, int] = {}
    for chunk_id in index["chunk_ids"]:
        prefix = prefix_of(chunk_id)
        vector_counts[prefix] = vector_counts.get(prefix, 0) + 1

    rows = []
    total_chunks = total_vectors = 0
    for spec in sorted(feeds(), key=lambda item: item.name):
        chunks = connection.execute(
            "select count(*) from chunks where chunk_id like ?", (f"{spec.prefix}%",)
        ).fetchone()[0]
        vectors = vector_counts.get(spec.prefix, 0)
        manifest = manifest_summary(spec.directory)
        rows.append({
            "name": spec.name,
            "prefix": spec.prefix,
            "directory": spec.directory,
            "chunks": chunks,
            "vectors": vectors,
            "lag": chunks - vectors,
            **manifest,
        })
        total_chunks += chunks
        total_vectors += vectors

    corpus_chunks = count_corpus_chunks(connection, feed_prefixes())
    total_chunks += corpus_chunks
    total_vectors += vector_counts.get(UNPREFIXED, 0)

    if args.json:
        print(json.dumps({"feeds": rows, "corpus_chunks_unprefixed": corpus_chunks,
                          "totals": {"chunks": total_chunks, "vectors": total_vectors}},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"{'фид':<10} {'префикс':<7} {'чанков':>8} {'векторов':>9} {'без вектора':>12} "
          f"{'документов':>11}  обновлён")
    for row in rows:
        print(f"{row['name']:<10} {row['prefix']:<7} {row['chunks']:>8} {row['vectors']:>9} "
              f"{row['lag']:>12} {row.get('documents', '-'):>11}  {row.get('updated_at', '-')[:19]}")
    print(f"{'корпус':<10} {'—':<7} {corpus_chunks:>8} {vector_counts.get(UNPREFIXED, 0):>9} "
          f"{corpus_chunks - vector_counts.get(UNPREFIXED, 0):>12} {'—':>11}  "
          "(политика JEV_VECTOR_EXCLUDE_SOURCES)")
    lag = sum(row["lag"] for row in rows if row["lag"] > 0)
    print(f"\nитого: {total_chunks} чанков, {total_vectors} векторов, "
          f"без вектора {total_chunks - total_vectors}"
          + (f" (в фидах {lag} — ждут шага векторизации)" if lag else ""))

    notes = [(row["name"], note) for row in rows for note in row.get("notes", [])]
    if notes:
        print("\nчто фиды сказали о себе:")
        for name, note in notes:
            print(f"  {name:<10} {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
