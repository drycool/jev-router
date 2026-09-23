#!/usr/bin/env python3
"""Fingerprint the vector archive, so "the corpus was not touched" is evidence.

Prints a sha256 of the corpus rows only (chunk ids, sources, contents and the raw
embedding bytes), so the same numbers before and after a merge prove the merge
preserved them rather than re-derived something that merely looks similar.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/dry/Jev/storage/jev_vectors.npz")
data = np.load(path, allow_pickle=True)
chunk_ids = data["chunk_ids"]
entity_types = (data["entity_types"] if "entity_types" in data.files
                else np.asarray([""] * len(chunk_ids)))

memory_mask = np.asarray([str(cid).startswith("mem:") for cid in chunk_ids])
corpus_mask = ~memory_mask

def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]

print(f"файл        : {path} ({path.stat().st_size / 1e6:.1f} MB, "
      f"mtime {int(path.stat().st_mtime_ns)})")
print(f"модель      : {data['model']} dimension={int(data['dimension'])}")
print(f"всего строк : {len(chunk_ids)}")
print(f"  корпус    : {int(corpus_mask.sum())}")
print(f"  память    : {int(memory_mask.sum())}")
print(f"ключи npz   : {sorted(data.files)}")
print()
print("отпечатки КОРПУСА (должны совпасть до и после слияния):")
print(f"  embeddings : {digest(data['embeddings'][corpus_mask])}   "
      f"shape={data['embeddings'][corpus_mask].shape} dtype={data['embeddings'].dtype}")
print(f"  chunk_ids  : {digest(chunk_ids[corpus_mask])}")
print(f"  contents   : {digest(data['contents'][corpus_mask])}")
print(f"  sources    : {digest(data['sources'][corpus_mask])}")
print(f"  domains    : {digest(data['domains'][corpus_mask])}")
if memory_mask.any():
    print("\nстроки памяти:")
    for cid, etype, source in zip(chunk_ids[memory_mask], entity_types[memory_mask],
                                  data["sources"][memory_mask]):
        print(f"  {etype:8s} {Path(str(source)).name:28s} {cid}")
