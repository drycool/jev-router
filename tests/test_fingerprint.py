"""The fingerprint instrument must not lie about what changed.

It did once: scripts/fingerprint_vectors.py hashed `array.tobytes()`, and numpy
pads string arrays to the widest element, so changing the *maximum* chunk length
(1901 -> 1877 characters during the bge-m3 migration) changed the digest of all
209 rows even though no text changed.  That produced a false "contents changed"
alarm on a migration that was in fact clean.  These tests pin the fix, because an
instrument that reports false drift is worse than no instrument.
"""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fingerprint_vectors.py"
DIGEST_LINE = re.compile(r"^\s+(embeddings|chunk_ids|contents|sources|domains)\s*:\s*([0-9a-f]{16})")


def write_archive(path: Path, contents_dtype=None, embeddings=None):
    contents = ["короткий текст про systemd"]
    sources = ["/home/dry/memory/projects/linux_systemd.md"]
    np.savez_compressed(
        path,
        embeddings=(np.zeros((1, 4), dtype=np.float32) if embeddings is None
                    else np.asarray(embeddings, dtype=np.float32)),
        chunk_ids=np.asarray(["chunk-1"]),
        contents=(np.asarray(contents, dtype=contents_dtype) if contents_dtype
                  else np.asarray(contents)),
        sources=np.asarray(sources),
        domains=np.asarray(["general"]),
        entity_types=np.asarray([""]),
        model=np.asarray("bge-m3"),
        dimension=np.asarray(4),
    )


def digests(path: Path) -> dict:
    result = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                            capture_output=True, text=True, check=True)
    return {match.group(1): match.group(2)
            for match in (DIGEST_LINE.match(line) for line in result.stdout.splitlines())
            if match}


class FingerprintTests(unittest.TestCase):

    def test_string_dtype_padding_does_not_change_the_digest(self):
        # Same text, different declared width: the old implementation reported
        # these as different.
        with tempfile.TemporaryDirectory() as directory:
            narrow = Path(directory) / "narrow.npz"
            wide = Path(directory) / "wide.npz"
            write_archive(narrow)
            write_archive(wide, contents_dtype="<U400")
            self.assertNotEqual(np.load(narrow, allow_pickle=True)["contents"].dtype,
                                np.load(wide, allow_pickle=True)["contents"].dtype)
            self.assertEqual(digests(narrow)["contents"], digests(wide)["contents"])

    def test_the_digest_still_notices_a_real_text_change(self):
        # The fix must not turn the instrument blind: a real edit has to move it.
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            write_archive(first)
            np.savez_compressed(
                second,
                embeddings=np.zeros((1, 4), dtype=np.float32),
                chunk_ids=np.asarray(["chunk-1"]),
                contents=np.asarray(["другой текст про systemd"]),
                sources=np.asarray(["/home/dry/memory/projects/linux_systemd.md"]),
                domains=np.asarray(["general"]),
                entity_types=np.asarray([""]),
                model=np.asarray("bge-m3"),
                dimension=np.asarray(4),
            )
            self.assertNotEqual(digests(first)["contents"], digests(second)["contents"])

    def test_embeddings_are_still_hashed_as_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.npz"
            right = Path(directory) / "right.npz"
            write_archive(left, embeddings=[[0.0, 0.0, 0.0, 0.0]])
            write_archive(right, embeddings=[[0.5, 0.0, 0.0, 0.0]])
            self.assertNotEqual(digests(left)["embeddings"], digests(right)["embeddings"])


if __name__ == "__main__":
    unittest.main()
