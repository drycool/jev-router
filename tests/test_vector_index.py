"""Tests for the Tier-2b vector index: adding memory rows without disturbing the corpus.

The defect this guards against is the quiet one.  A merge that dropped or
corrupted the 4562 corpus vectors would still produce an archive that loads, still
have the right dimension, still be searched - and would simply stop answering
corpus questions, with nothing in the logs to say why.  So the assertions here are
about what must *not* change, not only about what must appear.

The merge itself is a pure function of (index, chunks, embeddings), which is why
none of this needs a GPU.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.router import EMBEDDING_MODEL, Tier2Search
from core.vector_index import (
    ENTITY_MEMORY,
    reusable_embeddings,
    EXCLUDED_SOURCE_MARKERS,
    VectorIndexError,
    corpus_rows,
    is_excluded_source,
    load_index,
    merge_memory,
    prune_excluded,
    save_index,
)


def _corpus_index(rows: int = 5, dimension: int = 4) -> dict:
    generator = np.random.default_rng(seed=1234)
    return {
        "embeddings": generator.random((rows, dimension), dtype=np.float32),
        "chunk_ids": np.asarray([f"chunk-{i}" for i in range(rows)]),
        "contents": np.asarray([f"corpus text {i}" for i in range(rows)]),
        "sources": np.asarray(["d_espero.pdf"] * rows),
        "domains": np.asarray(["automotive"] * rows),
        "entity_types": np.asarray([""] * rows),
        "model": EMBEDDING_MODEL,
        "dimension": dimension,
    }


def _memory_chunks(count: int = 2) -> list[tuple[str, str, str]]:
    return [
        (f"mem:projects/jev_gateway.md#{i}", f"memory text {i}", "/home/dry/memory/projects/jev_gateway.md")
        for i in range(count)
    ]


class MergeTests(unittest.TestCase):
    def test_merging_keeps_every_corpus_vector_untouched(self):
        index = _corpus_index(rows=5)
        chunks = _memory_chunks(count=2)
        embeddings = np.full((2, 4), 0.5, dtype=np.float32)

        merged, stats = merge_memory(index, chunks, embeddings)

        self.assertEqual(stats["corpus"], 5)
        self.assertEqual(stats["memory_added"], 2)
        self.assertEqual(stats["total"], 7)
        # The corpus rows come through in order, value for value.
        np.testing.assert_array_equal(merged["embeddings"][:5], index["embeddings"])
        self.assertEqual(list(merged["chunk_ids"][:5]), list(index["chunk_ids"]))
        self.assertEqual(list(merged["sources"][:5]), list(index["sources"]))
        # And the memory rows land after them, tagged and findable by source.
        self.assertEqual(list(merged["chunk_ids"][5:]), [c[0] for c in chunks])
        self.assertEqual(list(merged["entity_types"][5:]), [ENTITY_MEMORY] * 2)
        self.assertTrue(str(merged["sources"][5]).endswith("jev_gateway.md"))
        self.assertEqual(merged["model"], index["model"])
        self.assertEqual(merged["dimension"], index["dimension"])

    def test_merging_twice_replaces_its_own_rows_instead_of_duplicating(self):
        index = _corpus_index(rows=5)
        chunks = _memory_chunks(count=2)

        once, first = merge_memory(index, chunks, np.zeros((2, 4), dtype=np.float32))
        twice, second = merge_memory(once, chunks, np.ones((2, 4), dtype=np.float32))

        self.assertEqual(first["total"], 7)
        self.assertEqual(second["total"], 7, "a second merge must not add a second copy")
        self.assertEqual(second["memory_replaced"], 2)
        self.assertEqual(second["corpus"], 5)
        np.testing.assert_array_equal(twice["embeddings"][:5], index["embeddings"])
        # The vectors are the new ones, not the old ones kept alongside them.
        np.testing.assert_array_equal(twice["embeddings"][5:], np.ones((2, 4), dtype=np.float32))

    def test_changed_text_replaces_the_vector_and_removed_chunks_drop_out(self):
        index = _corpus_index(rows=2)
        first_chunks = _memory_chunks(count=3)
        after_first, _ = merge_memory(index, first_chunks, np.zeros((3, 4), dtype=np.float32))

        edited = [
            (first_chunks[0][0], "memory text 0 edited", first_chunks[0][2]),
            (first_chunks[1][0], first_chunks[1][1], first_chunks[1][2]),
        ]
        after_edit, stats = merge_memory(after_first, edited, np.ones((2, 4), dtype=np.float32))

        self.assertEqual(stats["memory_replaced"], 3, "all three old rows are dropped")
        self.assertEqual(stats["memory_added"], 2)
        self.assertEqual(stats["total"], 4)
        self.assertEqual(str(after_edit["contents"][2]), "memory text 0 edited")

    def test_a_dimension_mismatch_is_refused_before_it_reaches_disk(self):
        index = _corpus_index(rows=2, dimension=4)
        with self.assertRaises(VectorIndexError) as caught:
            merge_memory(index, _memory_chunks(1), np.zeros((1, 8), dtype=np.float32))
        self.assertIn("dimension", str(caught.exception))

    def test_a_count_mismatch_is_refused(self):
        index = _corpus_index(rows=2)
        with self.assertRaises(VectorIndexError):
            merge_memory(index, _memory_chunks(3), np.zeros((2, 4), dtype=np.float32))

    def test_corpus_rows_are_counted_apart_from_memory(self):
        merged, _ = merge_memory(_corpus_index(rows=5), _memory_chunks(2),
                                 np.zeros((2, 4), dtype=np.float32))
        self.assertEqual(corpus_rows(merged), 5)


class ExclusionTests(unittest.TestCase):
    """The golden-corpus policy: the OCR'd manual must not reach the vector index."""

    def _mixed_index(self) -> dict:
        index = _corpus_index(rows=3)
        index["sources"] = np.asarray([
            "C:\\Users\\369\\Downloads\\d_espero\\d_espero.pdf",
            "Gemini-Кабель для Raspberry Pi 5_chunk_5.md",
            "d_espero.pdf",
        ])
        return index

    def test_excluded_sources_are_dropped_and_the_rest_survive_untouched(self):
        index = self._mixed_index()
        pruned, stats = prune_excluded(index, markers=("d_espero.pdf",))

        self.assertEqual(stats["pruned"], 2)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(list(pruned["chunk_ids"]), ["chunk-1"])
        np.testing.assert_array_equal(pruned["embeddings"], index["embeddings"][1:2])
        self.assertEqual(list(pruned["sources"]),
                         ["Gemini-Кабель для Raspberry Pi 5_chunk_5.md"])
        self.assertEqual(pruned["model"], index["model"])

    def test_the_marker_matches_a_path_not_only_a_filename(self):
        # Re-ingesting the manual under another directory must not smuggle it back.
        self.assertTrue(is_excluded_source("/mnt/backup/d_espero.pdf", ("d_espero.pdf",)))
        self.assertTrue(is_excluded_source("C:\\Users\\369\\Downloads\\d_espero\\d_espero.pdf",
                                           ("d_espero.pdf",)))
        self.assertFalse(is_excluded_source("/home/dry/memory/global/linux_systemd.md",
                                            ("d_espero.pdf",)))

    def test_pruning_is_a_no_op_without_markers_and_when_nothing_matches(self):
        index = self._mixed_index()
        same, stats = prune_excluded(index, markers=())
        self.assertEqual(stats["pruned"], 0)
        self.assertIs(same, index)
        kept, stats = prune_excluded(index, markers=("nothing-matches-this",))
        self.assertEqual(stats["pruned"], 0)
        self.assertIs(kept, index)

    def test_the_default_policy_names_the_ocr_manual(self):
        self.assertIn("d_espero.pdf", EXCLUDED_SOURCE_MARKERS)

    def test_memory_rows_are_not_excluded_by_the_corpus_policy(self):
        index = _corpus_index(rows=1)
        merged, _ = merge_memory(index, _memory_chunks(1), np.zeros((1, 4), dtype=np.float32))
        pruned, stats = prune_excluded(merged, markers=("d_espero.pdf",))
        self.assertEqual(stats["pruned"], 1, "only the corpus row matches")
        self.assertEqual(list(pruned["entity_types"]), [ENTITY_MEMORY])


class ArchiveTests(unittest.TestCase):
    def test_the_archive_round_trips_including_entity_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            merged, _ = merge_memory(_corpus_index(rows=3), _memory_chunks(2),
                                     np.zeros((2, 4), dtype=np.float32))
            save_index(path, merged)
            reloaded = load_index(path)

            np.testing.assert_array_equal(reloaded["embeddings"], merged["embeddings"])
            self.assertEqual(list(reloaded["chunk_ids"]), list(merged["chunk_ids"]))
            self.assertEqual(list(reloaded["entity_types"]), list(merged["entity_types"]))
            self.assertEqual(reloaded["model"], EMBEDDING_MODEL)
            self.assertEqual(reloaded["dimension"], 4)

    def test_an_archive_without_entity_types_infers_them_from_the_namespace(self):
        # Archives built before the memory tier have no entity_types array; they
        # must still load, with the memory rows recognised by their id prefix.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez_compressed(
                path,
                embeddings=np.zeros((2, 4), dtype=np.float32),
                chunk_ids=np.asarray(["chunk-1", "mem:global/linux_systemd.md#0"]),
                contents=np.asarray(["corpus", "memory"]),
                sources=np.asarray(["d_espero.pdf", "/home/dry/memory/global/linux_systemd.md"]),
                domains=np.asarray(["automotive", "general"]),
                model=np.asarray(EMBEDDING_MODEL),
                dimension=np.asarray(4),
            )
            reloaded = load_index(path)

            self.assertEqual(list(reloaded["entity_types"]), ["", ENTITY_MEMORY])

    def test_the_model_travels_with_the_archive_so_the_router_can_reject_it(self):
        # The merge itself does not police the model - the CLI refuses to mix
        # models, because the router rejects a whole index whose model does not
        # match the configured embedder.  What is asserted here is the part the
        # archive owes that check: it keeps the model string it was given.
        index = _corpus_index(rows=2)
        index["model"] = "some-other-embedder"
        merged, _ = merge_memory(index, _memory_chunks(1), np.zeros((1, 4), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            save_index(path, merged)
            self.assertEqual(load_index(path)["model"], "some-other-embedder")


class RouterVectorTests(unittest.TestCase):
    def test_search_vector_tags_a_memory_hit_and_still_finds_the_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vectors.npz"
            index = _corpus_index(rows=1, dimension=4)
            index["embeddings"] = np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
            merged, _ = merge_memory(index, _memory_chunks(1), np.asarray([[1.0, 0.0, 0.0, 0.0]]))
            save_index(vector_path, merged)

            with patch("core.router.VECTOR_DB_PATH", str(vector_path)), \
                 patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")), \
                 patch("core.router.SIMILARITY_THRESHOLD", 0.5):
                search = Tier2Search()
                try:
                    results = search.search_vector(
                        "любой запрос", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    )
                finally:
                    search.close()

        self.assertTrue(results, "the memory row should clear the patched threshold")
        self.assertEqual(results[0]["entity_type"], ENTITY_MEMORY)
        self.assertEqual(results[0]["search_type"], "vector")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()


class ReusableEmbeddingsTest(unittest.TestCase):
    """Incremental embedding: the daily refresh must not re-embed what it already has.

    Measured on this host: a feed of 2620 chunks with 15 changed ones costs the embedder's
    whole pass (1 m 47 s) if nothing is reused, and the embedder is a model on another
    machine that sleeps by default.  The rules below are what makes the difference between
    a schedule that runs unattended and one that wakes a GPU to recompute identical rows.
    """

    def index(self, rows):
        return {
            "model": EMBEDDING_MODEL,
            "dimension": 3,
            "chunk_ids": [row[0] for row in rows],
            "contents": [row[1] for row in rows],
            "sources": [row[2] for row in rows],
            "domains": ["general"] * len(rows),
            "entity_types": ["memory"] * len(rows),
            "embeddings": np.asarray([row[3] for row in rows], dtype=np.float32),
        }

    def test_an_unchanged_chunk_keeps_its_stored_vector(self):
        stored = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        index = self.index([("mem:a#1", "текст", "/m/a.md", [1.0, 2.0, 3.0])])
        embeddings, needs = reusable_embeddings(index, [("mem:a#1", "текст", "/m/a.md")], "mem:")
        self.assertFalse(bool(needs[0]))
        np.testing.assert_array_equal(embeddings[0], stored[0])

    def test_changed_text_is_not_reused(self):
        index = self.index([("mem:a#1", "старый текст", "/m/a.md", [1.0, 2.0, 3.0])])
        embeddings, needs = reusable_embeddings(index, [("mem:a#1", "новый текст", "/m/a.md")], "mem:")
        self.assertTrue(bool(needs[0]))
        # Zeroed rather than stale: a caller that ignored the mask must embed, not reuse.
        np.testing.assert_array_equal(embeddings[0], np.zeros(3, dtype=np.float32))

    def test_a_new_chunk_is_not_reused(self):
        index = self.index([("mem:a#1", "текст", "/m/a.md", [1.0, 2.0, 3.0])])
        chunks = [("mem:a#1", "текст", "/m/a.md"), ("mem:b#1", "ещё", "/m/b.md")]
        embeddings, needs = reusable_embeddings(index, chunks, "mem:")
        self.assertEqual(list(needs), [False, True])

    def test_a_vector_of_another_feed_is_not_visible(self):
        """Namespaces stay separate: a `prj:` row must not answer for a `mem:` chunk."""
        index = self.index([("mem:a#1", "текст", "/m/a.md", [1.0, 2.0, 3.0])])
        embeddings, needs = reusable_embeddings(index, [("prj:a#1", "текст", "/p/a.md")], "prj:")
        self.assertTrue(bool(needs[0]))
        np.testing.assert_array_equal(embeddings[0], np.zeros(3, dtype=np.float32))

    def test_a_changed_source_does_not_force_a_recompute(self):
        # Reuse is by text: the caller's source is what gets written, so a document that
        # moved must not cost an embedding of text that did not change.
        index = self.index([("mem:a#1", "текст", "/old/a.md", [1.0, 2.0, 3.0])])
        _embeddings, needs = reusable_embeddings(index, [("mem:a#1", "текст", "/new/a.md")], "mem:")
        self.assertFalse(bool(needs[0]))

    def test_the_caller_order_decides_which_row_gets_which_vector(self):
        index = self.index([("mem:a#1", "a", "/m/a.md", [1.0, 0.0, 0.0]),
                            ("mem:b#1", "b", "/m/b.md", [0.0, 1.0, 0.0])])
        chunks = [("mem:b#1", "b", "/m/b.md"), ("mem:a#1", "a", "/m/a.md")]
        embeddings, needs = reusable_embeddings(index, chunks, "mem:")
        self.assertEqual(list(needs), [False, False])
        np.testing.assert_array_equal(embeddings[0], np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[1], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    def test_an_archive_without_a_dimension_reuses_nothing(self):
        # An archive that never had vectors cannot be indexed against; the caller has to
        # build it from scratch, and a zero-width matrix makes that obvious instead of
        # returning rows that would fail the merge's shape check later.
        embeddings, needs = reusable_embeddings({}, [("mem:a#1", "a", "/m/a.md")], "mem:")
        self.assertEqual(embeddings.shape, (1, 0))
        self.assertTrue(bool(needs[0]))

    def test_a_stored_vector_of_the_wrong_width_is_not_reused(self):
        index = self.index([("mem:a#1", "текст", "/m/a.md", [1.0, 2.0, 3.0])])
        # A rectangular archive cannot hold a short row, so this is what a hand-edited or
        # half-migrated npz looks like: it declares dimension=3 and ships a 2-wide matrix.
        index["embeddings"] = np.asarray([[1.0, 2.0]], dtype=np.float32)
        embeddings, needs = reusable_embeddings(index, [("mem:a#1", "текст", "/m/a.md")], "mem:")
        self.assertTrue(bool(needs[0]))
        np.testing.assert_array_equal(embeddings[0], np.zeros(3, dtype=np.float32))
