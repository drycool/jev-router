"""The memory priority in the BM25 ordering.

The boost is a reordering, not a filter: it changes which rows win pool slots
when both a memory chunk and an OCR fragment matched, and it must not change
anything else.  These tests pin the sign convention (bm25 is negative, so
promotion means multiplying by more than 1), the fact that the reported score
stays raw, and that a query with no memory chunk in the pool is untouched.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.router import Tier2Search

# The corpus row repeats the shared terms more often, so it wins on raw bm25 -
# the situation the boost exists for.
DEFAULT_ROWS = [
    ("mem:note.md#1", "autostart user services autostart services",
     "/memory/projects/linux_systemd.md", "memory"),
    ("corpus-1", "autostart services autostart services autostart services",
     "/manual/d_espero.pdf", ""),
]


class _Fixture:
    """A throwaway FTS5 index, patched in for the duration of one test."""

    def __init__(self, rows=None):
        self.directory = tempfile.TemporaryDirectory()
        self.path = str(Path(self.directory.name) / "index.db")
        self.patch = patch("core.router.FTS5_DB_PATH", self.path)
        self.patch.start()
        search = Tier2Search()
        for chunk_id, content, source, entity_type in (rows or DEFAULT_ROWS):
            search.index_chunk(chunk_id, content, source=source, entity_type=entity_type)
        search.commit()
        search.close()

    def search(self, query="autostart services"):
        search = Tier2Search()
        try:
            return search.search_fts5(query)
        finally:
            search.close()

    def cleanup(self):
        self.patch.stop()
        self.directory.cleanup()


class MemoryBoostTests(unittest.TestCase):

    def setUp(self):
        self.fixture = _Fixture()

    def tearDown(self):
        self.fixture.cleanup()

    def test_default_boost_promotes_the_memory_row(self):
        with patch("core.router.FTS_MEMORY_BOOST", 1.5):
            results = self.fixture.search()
        self.assertEqual([r["chunk_id"] for r in results],
                         ["mem:note.md#1", "corpus-1"])

    def test_boost_of_one_reproduces_the_plain_bm25_order(self):
        with patch("core.router.FTS_MEMORY_BOOST", 1.0):
            boosted_off = [r["chunk_id"] for r in self.fixture.search()]
        # Disabled, the corpus row wins on raw bm25 - which is the defect the
        # default factor fixes, and the proof that the factor is what does it.
        self.assertEqual(boosted_off, ["corpus-1", "mem:note.md#1"])

    def test_the_reported_score_is_the_raw_bm25_not_the_weighted_one(self):
        with patch("core.router.FTS_MEMORY_BOOST", 3.0):
            boosted = next(r for r in self.fixture.search()
                           if r["chunk_id"] == "mem:note.md#1")
        with patch("core.router.FTS_MEMORY_BOOST", 1.0):
            plain = next(r for r in self.fixture.search()
                         if r["chunk_id"] == "mem:note.md#1")
        # Same number under both factors: the log must not record a relevance the
        # chunk did not score.
        self.assertAlmostEqual(boosted["score"], plain["score"], places=9)

    def test_a_query_with_no_memory_row_in_the_pool_is_untouched(self):
        with patch("core.router.FTS_MEMORY_BOOST", 5.0):
            results = self.fixture.search("tyre tread damage")
        # Neither row matched, and the boost cannot add one: an empty result set
        # must stay empty, not become a weak memory hit.
        self.assertEqual(results, [])

    def test_the_boost_reorders_and_never_adds_rows(self):
        with patch("core.router.FTS_MEMORY_BOOST", 1.0):
            plain = [r["chunk_id"] for r in self.fixture.search()]
        with patch("core.router.FTS_MEMORY_BOOST", 2.0):
            boosted = [r["chunk_id"] for r in self.fixture.search()]
        self.assertEqual(sorted(plain), sorted(boosted))
        self.assertNotEqual(plain, boosted)

    def test_corpus_rows_keep_their_relative_order(self):
        # Two corpus rows, the second weaker: the boost must not shuffle the
        # corpus against itself, only move memory rows across it.
        fixture = _Fixture(rows=DEFAULT_ROWS + [
            ("corpus-2", "autostart services", "/manual/d_espero.pdf", ""),
        ])
        self.addCleanup(fixture.cleanup)
        with patch("core.router.FTS_MEMORY_BOOST", 3.0):
            results = [r["chunk_id"] for r in fixture.search()]
        corpus = [cid for cid in results if not cid.startswith("mem:")]
        self.assertEqual(corpus, ["corpus-1", "corpus-2"])

    def test_memory_rows_carry_their_entity_type_into_results(self):
        with patch("core.router.FTS_MEMORY_BOOST", 1.5):
            results = self.fixture.search()
        # The boost keys off entity_type, so the field has to survive into the
        # result the router and the decision log read.
        self.assertEqual(results[0]["entity_type"], "memory")


if __name__ == "__main__":
    unittest.main()
