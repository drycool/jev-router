#!/usr/bin/env python3
"""Build the Jev Tier-2b vector index from the existing LightRAG chunks.

The index is an NPZ archive, not a pickle: loading it cannot execute code.
It records the embedding model and dimension; the router rejects mismatches.
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.router import detect_domain


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = Path("/home/dry/LightRag/car_index_espero_clean/kv_store_text_chunks.json")
DEFAULT_INDEX = PROJECT_ROOT / "storage" / "jev_vectors.npz"


async def embed_batch(client: httpx.AsyncClient, api: str, model: str, texts: list[str]) -> list[list[float]]:
    response = await client.post(api, json={"model": model, "input": texts})
    response.raise_for_status()
    embeddings = response.json().get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError(f"embedding API returned {len(embeddings)} vectors for {len(texts)} texts")
    return embeddings


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
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            vectors.extend(await embed_batch(client, args.api, args.model, [row[1] for row in batch]))
            print(f"embedded {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise RuntimeError("invalid embedding matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            embeddings=embeddings,
            chunk_ids=np.asarray([row[0] for row in records]),
            contents=np.asarray([row[1] for row in records]),
            sources=np.asarray([row[2] for row in records]),
            domains=np.asarray([detect_domain(f"{row[2]}\n{row[1]}") for row in records]),
            model=np.asarray(args.model),
            dimension=np.asarray(embeddings.shape[1]),
        )
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {len(records)} vectors, dimension={embeddings.shape[1]}, model={args.model}: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--api", default=os.getenv("JEV_EMBEDDING_API", "http://192.168.11.87:11434/api/embed"))
    parser.add_argument("--model", default=os.getenv("JEV_EMBEDDING_MODEL", "mxbai-embed-large"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--connect-timeout", type=float, default=2.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    asyncio.run(build(parser.parse_args()))


if __name__ == "__main__":
    main()
