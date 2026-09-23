"""Read, merge and write the Tier-2b vector index (`storage/jev_vectors.npz`).

The index is a derived artefact: `embeddings`, `chunk_ids`, `contents`, `sources`,
`domains`, plus `model` and `dimension`, which the router validates before it
searches anything.  Memory documents are added to the same archive rather than to
a second one, because `search_vector` searches one matrix - a second archive
would need routing code to decide which to use.

Why this module exists rather than two scripts: there are two ways to build this
index (a full rebuild from the LightRAG chunk store, and an incremental update of
just the memory rows) and both must produce the same invariant - the archive
contains the corpus *and* the memory documents.  Two implementations would drift,
and the drift would be silent: a full rebuild that forgot memory would look
exactly like a healthy index, with the memory documents simply unfindable - the
same class of defect that the FTS5 rebuild had.

The merge is a pure function of (index, chunks, embeddings) so it is testable
without a GPU: `merge_memory` never calls the network.
"""
from __future__ import annotations

import collections
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

# The namespace every memory chunk carries, in FTS5 and here.  It is what makes
# the merge idempotent: memory rows are recognised and replaced, corpus rows are
# never touched.
MEMORY_CHUNK_PREFIX = "mem:"
ENTITY_MEMORY = "memory"

# Sources that must not reach the vector index at all.
#
# The OCR'd Espero manual was 4414 of 4562 vectors and scored 0.75-0.83 against
# *every* query - autostart, schema validation, "rewrite the router in Go",
# cylinder-head torque - because its mangled fragments (soft hyphens, broken
# words) sit near the mean of the embedding space. That put the noise floor above
# the correct answers (0.55-0.69), so the 0.80 gate fired on garbage while the
# real content ranked #4536 of 4593. The manual stays in FTS5, where literal term
# matching is unaffected by this; it is removed from the vector index only.
#
# Matched as a substring of the source path, so re-ingesting the manual under
# another directory does not sneak it back in.
EXCLUDED_SOURCE_MARKERS: tuple[str, ...] = tuple(
    marker.strip()
    for marker in os.getenv("JEV_VECTOR_EXCLUDE_SOURCES", "d_espero.pdf").split(",")
    if marker.strip()
)


class VectorIndexError(RuntimeError):
    """A merge that would corrupt the index is refused rather than written."""


def is_excluded_source(source: str, markers: tuple[str, ...] | None = None) -> bool:
    markers = EXCLUDED_SOURCE_MARKERS if markers is None else markers
    return any(marker in source for marker in markers)


def load_index(path: str | Path) -> dict:
    """Load the archive and normalise it, so callers need no `files` checks."""
    data = np.load(path, allow_pickle=True)
    index = {
        "embeddings": data["embeddings"],
        "chunk_ids": data["chunk_ids"],
        "contents": data["contents"],
        "sources": data["sources"],
        "domains": data["domains"],
        "model": str(data["model"]),
        "dimension": int(data["dimension"]),
    }
    # `entity_types` is newer than the archive itself.  It is inferred from the
    # chunk-id namespace instead of invented: anything not prefixed `mem:` is
    # corpus, and the prefix is this module's own.
    if "entity_types" in data.files:
        index["entity_types"] = data["entity_types"]
    else:
        index["entity_types"] = np.asarray(
            [ENTITY_MEMORY if str(cid).startswith(MEMORY_CHUNK_PREFIX) else ""
             for cid in index["chunk_ids"]]
        )
    return index


def save_index(path: str | Path, index: dict) -> None:
    """Write the archive atomically: a reader never sees a half-written index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(
            temporary,
            embeddings=index["embeddings"],
            chunk_ids=index["chunk_ids"],
            contents=index["contents"],
            sources=index["sources"],
            domains=index["domains"],
            entity_types=index["entity_types"],
            model=np.asarray(index["model"]),
            dimension=np.asarray(index["dimension"]),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def memory_rows_from_fts5(db_path: str | Path) -> list[tuple[str, str, str]]:
    """(chunk_id, content, source) for every memory row the server indexed.

    Sources of truth: FTS5, not the directory, because that is the set the router
    can actually return.  Reading the directory would index text that search
    cannot reach.
    """
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            (str(chunk_id), str(content), str(source))
            for chunk_id, content, source in connection.execute(
                "SELECT chunk_id, content, source FROM chunks "
                "WHERE entity_type = ? ORDER BY chunk_id",
                (ENTITY_MEMORY,),
            )
        ]
    finally:
        connection.close()


def corpus_rows(index: dict) -> int:
    """How many rows are not memory rows."""
    return sum(1 for cid in index["chunk_ids"]
               if not str(cid).startswith(MEMORY_CHUNK_PREFIX))


def prune_excluded(index: dict, markers: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    """Drop rows whose source matches an exclusion marker.

    Pure, like the merge: it slices arrays and never touches the network. Running
    it twice is a no-op, and it never rebuilds a kept row - the surviving vectors
    are the ones that were already in the archive, not re-derived ones.
    """
    markers = EXCLUDED_SOURCE_MARKERS if markers is None else markers
    if not markers:
        return index, {"pruned": 0, "kept": len(index["chunk_ids"]), "sources": {}}
    keep = np.asarray([not is_excluded_source(str(source), markers)
                       for source in index["sources"]], dtype=bool)
    dropped = collections.Counter(
        str(source).replace("\\", "/").rsplit("/", 1)[-1]
        for source, keep_it in zip(index["sources"], keep) if not keep_it
    )
    if keep.all():
        return index, {"pruned": 0, "kept": len(index["chunk_ids"]), "sources": {}}
    pruned = {key: value[keep] if isinstance(value, np.ndarray) else value
              for key, value in index.items()}
    stats = {
        "pruned": int((~keep).sum()),
        "kept": int(keep.sum()),
        "sources": dict(dropped.most_common(10)),
    }
    return pruned, stats


def merge_memory(index: dict, chunks: Sequence[tuple[str, str, str]],
                 embeddings: np.ndarray, domains: Iterable[str] | None = None) -> tuple[dict, dict]:
    """Return a new index with the memory rows replaced by these ones.

    Pure: no network, no filesystem.  Idempotent by construction - rows whose id
    carries the memory prefix are dropped before the new ones are appended, so
    running it twice leaves one copy.  Corpus rows are copied through untouched,
    and that is asserted rather than assumed (see the tests): losing them would
    silently empty the index for every corpus query.
    """
    if len(chunks) != len(embeddings):
        raise VectorIndexError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings"
        )
    if embeddings.size and embeddings.ndim != 2:
        raise VectorIndexError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    dimension = int(index["dimension"])
    if embeddings.size and embeddings.shape[1] != dimension:
        raise VectorIndexError(
            f"embedding dimension {embeddings.shape[1]} does not match the index "
            f"({dimension}); the router would reject the whole index"
        )
    domains = list(domains) if domains is not None else ["" for _ in chunks]
    if len(domains) != len(chunks):
        raise VectorIndexError(f"{len(chunks)} chunks but {len(domains)} domains")

    keep = np.asarray([not str(cid).startswith(MEMORY_CHUNK_PREFIX)
                       for cid in index["chunk_ids"]], dtype=bool)
    replaced = int((~keep).sum())

    merged = {
        "embeddings": np.vstack([index["embeddings"][keep],
                                 np.asarray(embeddings, dtype=index["embeddings"].dtype)]),
        "chunk_ids": np.concatenate([index["chunk_ids"][keep],
                                     np.asarray([c[0] for c in chunks])]),
        "contents": np.concatenate([index["contents"][keep],
                                    np.asarray([c[1] for c in chunks])]),
        "sources": np.concatenate([index["sources"][keep],
                                   np.asarray([c[2] for c in chunks])]),
        "domains": np.concatenate([index["domains"][keep], np.asarray(domains)]),
        "entity_types": np.concatenate([index["entity_types"][keep],
                                        np.asarray([ENTITY_MEMORY] * len(chunks))]),
        "model": index["model"],
        "dimension": dimension,
    }
    stats = {
        "corpus": corpus_rows(merged),
        "memory_replaced": replaced,
        "memory_added": len(chunks),
        "total": len(merged["chunk_ids"]),
    }
    return merged, stats


async def embed_texts(client, api: str, model: str, texts: Sequence[str],
                      batch_size: int = 16) -> np.ndarray:
    """Embed texts in batches through an Ollama-compatible /api/embed endpoint.

    One implementation for both builders.  Raises if the server returns a
    different number of vectors than it was given texts - a silent mismatch here
    would attach embeddings to the wrong chunks.
    """
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        response = await client.post(api, json={"model": model, "input": batch})
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])
        if len(embeddings) != len(batch):
            raise VectorIndexError(
                f"embedding API returned {len(embeddings)} vectors for {len(batch)} texts"
            )
        vectors.extend(embeddings)
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)
