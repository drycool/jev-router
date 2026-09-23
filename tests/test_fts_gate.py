"""The FTS gate: what may take the exact-FTS early exit.

Two measured defects are pinned here.  First, the raw OCR corpus could take the
exit on its own, so a question about schema validation exited on a fragment about
wheel alignment.  Second, two matched terms out of six is a coincidence of
ordinary words, not a match, so questions about user services exited on "при" and
"загрузке".

The rule under test: memory rows take the exit when they also survived it; the
raw corpus never takes it alone; and a held-back raw row is still carried into
the context when the vector tier answers, because the manual has no vectors and
dropping it there would delete the only copy of its answer.
"""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.router import JevRouter, Strategy, _gate_selection, _is_fts_exact
from scripts.check_memory_contamination import competitors

RAW_SOURCE = "C:/Users/369/Downloads/d_espero/d_espero.pdf"
QUERY = "затяжка болтов головки блока цилиндров момент"


class GateSelectionTests(unittest.TestCase):
    """The pure decision, without a router or a database."""

    @staticmethod
    def row(content, source=""):
        return {"chunk_id": content[:8], "content": content, "source": source,
                "entity_type": "memory" if "memory" in source else "", "score": 1.0}

    def test_two_terms_out_of_six_is_not_a_match(self):
        # The measured coincidence: "при" and "загрузке" out of a six-term query,
        # which is how a question about user services exited on a manual fragment.
        self.assertFalse(_is_fts_exact(QUERY, self.row("при загрузке")))
        self.assertFalse(_is_fts_exact(QUERY, self.row("болтов и цилиндров от руки")))

    def test_the_boundary_is_the_configured_number_of_terms(self):
        # Exactly FTS_GATE_MIN_TERMS is enough.  This is a count, not a weighting,
        # so three *ordinary* words would also pass - the corpus measurement is
        # what keeps that honest, not the arithmetic.
        self.assertTrue(_is_fts_exact(QUERY, self.row("болтов головки цилиндров")))

    def test_three_terms_in_a_long_query_do_match(self):
        chunk = self.row("болтов головки цилиндров последовательность")
        self.assertTrue(_is_fts_exact(QUERY, chunk))

    def test_a_short_query_still_requires_all_its_terms(self):
        # min(FTS_GATE_MIN_TERMS, len(keywords)): a two-term query is unchanged.
        self.assertTrue(_is_fts_exact("Raspberry cable", self.row("Raspberry cable here")))
        self.assertFalse(_is_fts_exact("Raspberry cable", self.row("Raspberry only")))

    def test_memory_rows_win_the_gate_over_the_raw_corpus(self):
        memory = self.row("болтов головки цилиндров memory", source="/memory/x.md")
        raw = self.row("болтов головки цилиндров момент", source=RAW_SOURCE)
        allowed, held = _gate_selection(QUERY, [raw, memory])
        self.assertEqual([r["chunk_id"] for r in allowed], [memory["chunk_id"]])
        self.assertEqual([r["chunk_id"] for r in held], [raw["chunk_id"]])

    def test_raw_rows_alone_take_no_exit_but_are_held_rather_than_dropped(self):
        raw = self.row("болтов головки цилиндров", source=RAW_SOURCE)
        allowed, held = _gate_selection(QUERY, [raw])
        self.assertEqual(allowed, [])
        # Held back, not deleted: the vector tier must still be able to carry it.
        self.assertEqual(held, [raw])

    def test_a_decisive_raw_match_keeps_the_exit_for_itself(self):
        # Four or more content terms is a chunk the query came from, not a
        # coincidence: the manual is the only source for such questions and is
        # excluded from the vector index, so it must keep its local exit.
        decisive = self.row("болтов головки цилиндров момент", source=RAW_SOURCE)
        allowed, held = _gate_selection(QUERY, [decisive])
        self.assertEqual(allowed, [decisive])
        self.assertEqual(held, [])

    def test_a_coincidental_raw_match_does_not(self):
        coincidental = self.row("болтов головки цилиндров", source=RAW_SOURCE)
        allowed, held = _gate_selection(QUERY, [coincidental])
        self.assertEqual(allowed, [])
        self.assertEqual(held, [coincidental])

    def test_a_chunk_without_a_source_is_not_the_raw_corpus(self):
        anonymous = self.row("болтов головки цилиндров")
        allowed, held = _gate_selection(QUERY, [anonymous])
        self.assertEqual(allowed, [anonymous])
        self.assertEqual(held, [])

    def test_no_survivors_means_no_exit_and_nothing_held(self):
        allowed, held = _gate_selection(QUERY, [self.row("совсем другой текст")])
        self.assertEqual(allowed, [])
        self.assertEqual(held, [])

    def test_the_rule_can_be_switched_off(self):
        raw = self.row("болтов головки цилиндров момент", source=RAW_SOURCE)
        with patch("core.router.FTS_GATE_REQUIRE_GOLDEN", False):
            allowed, held = _gate_selection(QUERY, [raw])
        self.assertEqual(allowed, [raw])
        self.assertEqual(held, [])


class GateRoutingTests(unittest.TestCase):
    """The same rule through the router, where the context is what matters."""

    def _router(self, directory):
        patcher = patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db"))
        patcher.start()
        self.addCleanup(patcher.stop)
        router = JevRouter()
        self.addCleanup(router.tier2.close)
        return router

    @staticmethod
    def _offline(router):
        """No embedding server, no graph tier: the local path is what is tested."""
        async def no_embedding(_):
            return None

        async def no_graph(*_, **__):
            return {"response": "", "error_type": "timeout"}

        router.get_embedding = no_embedding
        router.tier3.search = no_graph
        return router

    def test_memory_match_takes_the_exit_and_the_raw_row_stays_out_of_context(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self._offline(self._router(directory))
            router.tier2.index_chunk("mem:1", "болтов головки цилиндров memory",
                                     source="/memory/projects/x.md", entity_type="memory")
            router.tier2.index_chunk("raw-1", "болтов головки цилиндров момент",
                                     source=RAW_SOURCE)
            router.tier2.commit()
            result = asyncio.run(router.route(QUERY))
            self.assertEqual(result.routing_decision.strategy, Strategy.EXACT_FTS)
            self.assertIn("memory", result.context)
            self.assertNotIn("цилиндров момент", result.context)

    def test_a_raw_only_match_does_not_exit_and_keeps_its_content_in_the_request(self):
        with tempfile.TemporaryDirectory() as directory:
            router = self._offline(self._router(directory))
            router.tier2.index_chunk("raw-1", "болтов головки цилиндров",
                                     source=RAW_SOURCE)
            router.tier2.commit()
            result = asyncio.run(router.route(QUERY))
            # The exit is refused, so the request goes on and the degraded
            # fall-through serves the local pool: the manual's own question
            # still gets the manual's own text.
            self.assertNotEqual(result.routing_decision.strategy, Strategy.EXACT_FTS)
            self.assertIn("болтов головки цилиндров", result.context)



class MemoryContaminationTests(unittest.TestCase):
    """The check that keeps the indexed memory from competing with its controls.

    Three times in one session a finding was written into the corpus with the
    control query's own words, and the document describing the measurement then
    won the gate for that query.  This is the guard, tested on the shape that
    actually happened.
    """

    def test_prose_that_repeats_the_query_is_reported(self):
        rows = [("mem:projects/jev_gateway.md#10",
                 "запрос про кабели (Raspberry Pi) ушёл в vector_fast",
                 "/home/dry/memory/projects/jev_gateway.md")]
        found = competitors(rows, "Кабели для Raspberry Pi 5", "Gemini")
        self.assertEqual([chunk_id for chunk_id, _ in found],
                         ["mem:projects/jev_gateway.md#10"])

    def test_the_file_that_should_answer_is_not_a_competitor(self):
        rows = [("mem:projects/garageos.md#2",
                 "garageOS чёрный ящик инцидентов",
                 "/home/dry/memory/projects/garageos.md")]
        self.assertEqual(competitors(
            rows, "Как устроена двухслойная конфигурация garageOS и чёрный ящик инцидентов",
            "garageos.md"), [])

    def test_the_raw_corpus_is_never_reported(self):
        rows = [("raw", "болтов головки блока цилиндров момент", RAW_SOURCE)]
        self.assertEqual(competitors(rows, QUERY, "d_espero.pdf"), [])

    def test_a_clean_description_is_clean(self):
        rows = [("mem:projects/x.md#1", "запрос про периферию Pi5",
                 "/home/dry/memory/projects/x.md")]
        self.assertEqual(competitors(rows, "Кабели для Raspberry Pi 5", "Gemini"), [])


if __name__ == "__main__":
    unittest.main()
