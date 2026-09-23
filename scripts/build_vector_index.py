#!/usr/bin/env python3
"""Build the Jev Tier-2b vector index from the existing LightRAG chunks.

The index is an NPZ archive, not a pickle: loading it cannot execute code.
It records the embedding model and dimension; the router rejects mismatches.

A full rebuild is the expensive path (one embed call per corpus chunk).  It also
merges the memory documents at the end, because otherwise a rebuild would
silently drop them: the archive would look healthy while the memory documents
became unfindable in the vector tier - the same defect class the FTS5 rebuild
had.  Use scripts/build_memory_vectors.py for the cheap incremental update of
just the memory rows.

    python3 scripts/build_vector_index.py --chunks /path/to/chunks.json
    python3 scripts/build_vector_index.py --chunks ... --no-memory
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.router import detect_domain
from core.vector_index import (
    embed_texts,
    memory_rows_from_fts5,
    merge_memory,
    save_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = Path("/home/dry/LightRag/car_index_espero_clean/kv_store_text_chunks.json")
DEFAULT_INDEX = PROJECT_ROOT / "storage" / "jev_vectors.npz"
DEFAULT_MEMORY_DB = PROJECT_ROOT / "storage" / "jev_fts5.db"


async def build(args: argparse.Namespace) -> None:
    with args.chunks.open(encoding="utf-8") as stream:
        source_data = json.load(stream)
    records = [
        (chunk_id, item.get("content", "")[: args.max_chars], item.get("file_path", ""))
        for chunk_id, item in source_data.items()
        if item.get("content", "")
    ]
    timeout = httpx.Timeout(connect=args.connect_timeout, read=args.read_timeout,
                            write=args.read_timeout, pool=args.connect_timeout)
    batches: list[np.ndarray] = []
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            # Batched through the shared embedder so the count and shape checks
            # live in one place; the progress line stays here because a corpus
            # rebuild takes minutes and silence looks like a hang.
            batches.append(await embed_texts(client, args.api, args.model,
                                             [row[1] for row in batch],
                                             batch_size=args.batch_size))
            print(f"embedded {min(start + len(batch), len(records))}/{len(records)}", flush=True)
        if not batches:
            raise RuntimeError("the chunk store held no content to embed")
        embeddings = np.vstack(batches).astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
            raise RuntimeError("invalid embedding matrix")
        print(f"corpus embedded in {(time.perf_counter() - started) * 1000:.0f} ms")

        index = {
            "embeddings": embeddings,
            "chunk_ids": np.asarray([row[0] for row in records]),
            "contents": np.asarray([row[1] for row in records]),
            "sources": np.asarray([row[2] for row in records]),
            "domains": np.asarray([detect_domain(f"{row[2]}\n{row[1]}") for row in records]),
            "entity_types": np.asarray(["" for _ in records]),
            "model": args.model,
            "dimension": int(embeddings.shape[1]),
        }
        save_index(args.output, index)
        print(f"wrote {len(records)} corpus vectors, dimension={embeddings.shape[1]}, "
              f"model={args.model}: {args.output}")

        if args.include_memory:
            await _merge_memory(client, args, index)


async def _merge_memory(client: httpx.AsyncClient, args: argparse.Namespace, index: dict) -> None:
    """Append the memory documents to a freshly rebuilt corpus index."""
    database = Path(args.memory_db)
    if not database.exists():
        print(f"memory: no FTS5 database at {database}, skipped "
              f"(start the router once to build it)")
        return
    chunks = memory_rows_from_fts5(database)
    if not chunks:
        print(f"memory: no memory rows in {database}, nothing to merge")
        return
    embeddings = await embed_texts(client, args.api, args.model,
                                   [content for _, content, _ in chunks],
                                   batch_size=args.batch_size)
    domains = [detect_domain(f"{source}\n{content}") for _, content, source in chunks]
    merged, stats = merge_memory(index, chunks, embeddings, domains)
    save_index(args.output, merged)
    print(f"memory: merged {stats['memory_added']} chunks from {database.name} "
          f"(replaced {stats['memory_replaced']}) -> {stats['total']} vectors total")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--api", default=os.getenv("JEV_EMBEDDING_API", "http://192.168.11.87:11434/api/embed"))
    parser.add_argument("--model", default=os.getenv("JEV_EMBEDDING_MODEL", "mxbai-embed-large"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--memory-db", type=Path, default=DEFAULT_MEMORY_DB,
                        help="FTS5 database the memory rows are read from")
    parser.add_argument("--no-memory", dest="include_memory", action="store_false",
                        help="build the corpus index only (memory rows are dropped)")
    parser.set_defaults(include_memory=True)
    asyncio.run(build(parser.parse_args()))


if __name__ == "__main__":
    main()
