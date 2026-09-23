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
    """sha256 of the array's *contents*, independent of numpy's dtype.

    For string arrays numpy pads to the widest element, so `.tobytes()` changes
    when the widest string changes even though no text did: a model migration
    dropped the maximum chunk length from 1901 to 1877 characters and this
    function reported "contents changed" for all 209 rows.  Text is therefore
    hashed as its UTF-8 bytes, not as fixed-width records.
    """
    array = np.ascontiguousarray(array)
    if array.dtype.kind in {"U", "S", "O"}:
        payload = b"\x00".join(str(item).encode("utf-8") for item in array.ravel())
    else:
        payload = array.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]

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
    # Same evidence for the memory slice: a model migration must change the
    # embeddings and nothing else, so these four must match across the rebuild.
    print("\nотпечатки ПАМЯТИ (при смене модели должны совпасть, в отличие от embeddings):")
    print(f"  chunk_ids  : {digest(chunk_ids[memory_mask])}")
    print(f"  contents   : {digest(data['contents'][memory_mask])}")
    print(f"  sources    : {digest(data['sources'][memory_mask])}")
    print(f"  domains    : {digest(data['domains'][memory_mask])}")
    print("\nстроки памяти:")
    for cid, etype, source in zip(chunk_ids[memory_mask], entity_types[memory_mask],
                                  data["sources"][memory_mask]):
        print(f"  {etype:8s} {Path(str(source)).name:28s} {cid}")
