"""Tests for the context-assembly step: dedup and the character budget.

Written against the measured failure that produced them. The router's FTS pool held the
same text twice (the Espero manual was ingested under two source paths: 865 duplicate
groups, 19% of the index), the old `[:3]` slice passed the duplicate through, and the
continuation of the procedure the model needed - rank 6 of 10 - never arrived. The model
then reported its own context as truncated, which is how the defect was found.

These run against the real retrieval layer (a temporary FTS5 database) wherever the point
of the test is the pipeline rather than the pure function, because a test that only feeds
hand-built dicts proves nothing about what search_fts5 returns.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.router import (
    MAX_CONTEXT_CHARS,
    RETRIEVAL_LIMIT,
    Tier2Search,
    assemble_context,
)


def _chunk(content: str, chunk_id: str = "c", score: float = 0.0) -> dict:
    return {"chunk_id": chunk_id, "content": content, "score": score}


class ContextAssemblyTests(unittest.TestCase):
    def test_identical_chunks_are_delivered_once(self):
        context, stats = assemble_context([
            _chunk("Raspberry Pi cable", "a"),
            _chunk("Raspberry Pi cable", "b"),
        ])
        self.assertEqual(context, "Raspberry Pi cable")
        self.assertEqual(stats["chunks_used"], 1)
        self.assertEqual(stats["chunks_duplicate"], 1)

    def test_deduplication_does_not_depend_on_trailing_whitespace(self):
        context, stats = assemble_context([
            _chunk("same text", "a"),
            _chunk("  same text\n", "b"),
        ])
        self.assertEqual(context, "same text")
        self.assertEqual(stats["chunks_duplicate"], 1)

    def test_chunks_differing_in_the_middle_are_not_merged(self):
        """Dedup is exact text, deliberately not whitespace-normalised.

        Two chunks that differ inside the text are different chunks, and collapsing
        interior whitespace could merge distinct code blocks.
        """
        context, stats = assemble_context([
            _chunk("step 25 N*m", "a"),
            _chunk("step 35 N*m", "b"),
        ])
        self.assertEqual(stats["chunks_used"], 2)
        self.assertEqual(stats["chunks_duplicate"], 0)
        self.assertIn("25 N*m", context)
        self.assertIn("35 N*m", context)

    def test_the_budget_bounds_the_context(self):
        # Unique content per chunk: ten copies of one string would be deduplicated
        # before the budget ever came into play, and the test would pass for the wrong
        # reason - which is exactly what the first version of it did.
        chunks = [_chunk(f"chunk{i} " + "x" * 400, f"c{i}") for i in range(10)]
        context, stats = assemble_context(chunks, budget=1000)
        self.assertLessEqual(len(context), 1000)
        self.assertTrue(stats["budget_exhausted"])
        self.assertLess(stats["chunks_used"], len(chunks))

    def test_a_budget_is_read_at_call_time_so_it_stays_patchable(self):
        with patch("core.router.MAX_CONTEXT_CHARS", 120):
            context, stats = assemble_context([_chunk("y" * 200, "a")])
        self.assertEqual(stats["budget"], 120)
        self.assertEqual(context, "y" * 200)

    def test_the_first_chunk_survives_a_budget_smaller_than_itself(self):
        """A budget that returns nothing would silently switch retrieval off."""
        context, stats = assemble_context([_chunk("z" * 5000, "a")], budget=100)
        self.assertEqual(len(context), 5000)
        self.assertEqual(stats["chunks_used"], 1)

    def test_a_chunk_too_large_does_not_hide_a_smaller_one_behind_it(self):
        context, stats = assemble_context(
            [_chunk("first", "a"), _chunk("L" * 5000, "b"), _chunk("third", "c")],
            budget=100,
        )
        self.assertIn("first", context)
        self.assertIn("third", context)
        self.assertNotIn("LLLL", context)
        self.assertEqual(stats["chunks_too_large"], 1)

    def test_empty_and_blank_chunks_are_skipped(self):
        context, stats = assemble_context([_chunk("", "a"), _chunk("   ", "b"), _chunk("real", "c")])
        self.assertEqual(context, "real")
        self.assertEqual(stats["chunks_considered"], 3)
        self.assertEqual(stats["chunks_used"], 1)

    def test_no_results_produce_no_context_and_honest_stats(self):
        context, stats = assemble_context([])
        self.assertEqual(context, "")
        self.assertEqual(stats["chunks_used"], 0)
        self.assertFalse(stats["budget_exhausted"])

    def test_stats_name_the_budget_that_was_in_force(self):
        _, stats = assemble_context([_chunk("a", "a")], budget=4321)
        self.assertEqual(stats["budget"], 4321)
        self.assertEqual(stats["chars"], 1)


class TheSixthChunkTests(unittest.TestCase):
    """The regression itself, through a real FTS5 index rather than hand-built dicts."""

    def _index(self, directory: str) -> Tier2Search:
        search = Tier2Search()
        # The same page twice under two source paths, as the corpus actually has it.
        for chunk_id in ("dup-a", "dup-b"):
            search.index_chunk(chunk_id, "порядок регулировки зазоров клапанов двигателя")
        for i in range(2, 7):
            search.index_chunk(
                f"step-{i}",
                f"порядок регулировки зазоров клапанов этап {i} " + "описание " * 60,
            )
        # The continuation the model asked for, ranked last of the pool. The filler is
        # load-bearing: FTS5's bm25 normalises by document length, so a short version of
        # this chunk ranked 3rd and the old three-chunk slice already carried it. 150
        # repeats of filler was measured to put it 8th of 8, which is the case the
        # regression is about.
        search.index_chunk(
            "step-g",
            "порядок регулировки зазоров клапанов этап г доворачивание на 90 градусов "
            + "описание " * 150,
        )
        search.commit()
        return search

    def test_the_duplicate_does_not_consume_a_slot_in_a_three_chunk_world(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                search = self._index(directory)
                results = search.search_fts5("порядок регулировки зазоров клапанов")
                old = "\n\n".join(r["content"] for r in results[:3])
                new, stats = assemble_context(results, budget=MAX_CONTEXT_CHARS)
                self.assertEqual(stats["chunks_duplicate"], 1)
                self.assertEqual(stats["chunks_considered"], 8)
                self.assertEqual(stats["chunks_used"], 7)
                self.assertGreater(len(new), len(old))
                search.close()

    def test_the_last_ranked_unique_chunk_reaches_the_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                search = self._index(directory)
                results = search.search_fts5("порядок регулировки зазоров клапанов")
                # The old behaviour, for the record: a fixed three chunks.
                old = "\n\n".join(r["content"] for r in results[:3])
                new, _ = assemble_context(results, budget=MAX_CONTEXT_CHARS)
                self.assertNotIn("доворачивание", old)
                self.assertIn("доворачивание", new)
                search.close()


class RetrievalDepthTests(unittest.TestCase):
    """The pool depth decides whether a chunk is in the running at all.

    This is the defect that survived the context-assembly fix: the agent kept reporting an
    incomplete context because the chunk holding the complete tightening sequence sat at
    rank 13 of the query's matches, while the pool stopped at 10. Assembly cannot deliver a
    chunk that retrieval never returned.
    """

    QUERY = "затяжка болтов головки блока цилиндров момент"

    def _index(self, search: Tier2Search, fillers: int) -> int:
        """Index fillers plus one marker chunk; return the marker's 1-based rank.

        The marker is longer than the fillers on purpose: bm25 normalises by document
        length, so a longer document ranks lower, which is how the real corpus buries the
        procedure behind shorter, looser matches.
        """
        for i in range(fillers):
            search.index_chunk(
                f"filler-{i}",
                f"затяжка болтов головки блока цилиндров момент проба {i} " + "текст " * 40,
            )
        search.index_chunk(
            "sequence",
            "затяжка болтов головки блока цилиндров момент в несколько этапов: "
            "а) все болты моментом 25 Н*м, б) затяните моментом, в) доверните на 60 "
            "градусов, г) доворачивание по схеме Рис. 3.20 " + "описание " * 120,
        )
        search.commit()
        ranked = search.search_fts5(self.QUERY, limit=200)
        for position, result in enumerate(ranked, start=1):
            if result["chunk_id"] == "sequence":
                return position
        self.fail("the marker chunk is not retrievable at all")

    def test_the_marker_needs_a_deeper_pool_than_the_old_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                search = Tier2Search()
                rank = self._index(search, fillers=12)
                self.assertGreater(rank, 10, "fixture no longer reproduces the defect")
                self.assertLessEqual(
                    rank, 20, "fixture puts the marker beyond the production pool"
                )

                shallow = [r["chunk_id"] for r in search.search_fts5(self.QUERY, limit=10)]
                self.assertNotIn("sequence", shallow)

                deep = [r["chunk_id"] for r in search.search_fts5(self.QUERY, limit=20)]
                self.assertIn("sequence", deep)
                search.close()

    def test_the_pool_depth_comes_from_the_constant_not_a_bound_default(self):
        """One setting governs every caller, and it stays patchable."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                search = Tier2Search()
                self._index(search, fillers=12)
                with patch("core.router.RETRIEVAL_LIMIT", 4):
                    self.assertEqual(len(search.search_fts5(self.QUERY)), 4)
                with patch("core.router.RETRIEVAL_LIMIT", 20):
                    # The pool is capped by the constant, not by a value bound at import
                    # time, and capped by the available matches rather than by the limit.
                    self.assertEqual(len(search.search_fts5(self.QUERY)), 13)
                search.close()

    def test_the_default_pool_is_wide_enough_for_the_measured_case(self):
        """A guard on the constant itself: 10 was measured to be too shallow."""
        self.assertGreaterEqual(RETRIEVAL_LIMIT, 13)


if __name__ == "__main__":
    unittest.main()
