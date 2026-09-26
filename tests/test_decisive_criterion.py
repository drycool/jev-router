"""The decisive criterion: an absolute floor *and* a margin over the corpus's own noise.

One number could not decide this.  JEV_SIMILARITY_THRESHOLD = 0.45 was calibrated when the
corpus held 209 rows; it holds 858 vectors over 5269 chunks now, the noise floor rose with
it, and measured on 2026-09-26 three of six questions the corpus cannot answer at all were
labelled *decisive* - a cat, a currency rate, a tax letter - with more than a thousand
characters of selected text as the evidence.  A consumer told "this is the answer" and
handed wiring instructions for a cat question does not retry; it concludes the base is
useless.

What the tests below pin down:

* the rule itself, including the case where the margin is what refuses the hit and the
  case where the absolute floor is what refuses it (they fail differently and are easy to
  confuse);
* that the margin is measured per query, from the corpus, not one number for everything;
* that a corpus too small for a quantile falls back to the absolute rule, so a fresh
  install or a test fixture does not start refusing everything;
* that the demotion is a *label* change, not a deletion: the material is still served,
  because "there is material but it is not the answer" is the useful thing to say.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from core.router import (  # noqa: E402
    DECISIVE_MARGIN,
    DECISIVE_MIN_CORPUS,
    SIMILARITY_THRESHOLD,
    Tier2Search,
    decisive_hit,
)
from core.vector_index import save_index  # noqa: E402

# 256, not 8: in eight dimensions one lucky dimension lifts a cosine enormously, so the
# fixture's noise tail was far fatter than a real embedder's and the numbers it measured
# were fiction.  At 256 the tail concentrates, like the real corpus (measured: positives
# 0.48-0.64 against floors 0.33-0.55, mean similarity about 0.25 at 1024 dimensions).
DIMENSION = 256


def _index(rows: int, spread: float = 0.0, one_clear_winner: bool = False) -> dict:
    """A synthetic corpus: every chunk near one direction, optionally one real hit.

    ``spread`` sets how far the bulk of the corpus sits from the query: 0.08 puts it at
    cosine ~0.67 and 0.15 at ~0.45, which is the range the real corpus shows.  A dense
    topic is exactly this - many chunks similar to the question without any of them being
    its answer.  ``one_clear_winner`` adds a chunk aligned with the query itself, which is
    what an actual answer looks like.
    """
    rng = np.random.default_rng(7)
    query_direction = np.zeros(DIMENSION, dtype=np.float32)
    query_direction[0] = 1.0
    bulk = np.tile(query_direction, (rows, 1))
    if spread:
        bulk = bulk + rng.normal(0, spread, size=(rows, DIMENSION)).astype(np.float32)
    chunk_ids = [f"mem:тест.md#{i}" for i in range(rows)]
    contents = [f"чанк {i}" for i in range(rows)]
    if one_clear_winner:
        bulk[0] = query_direction
        contents[0] = "чанк с ответом"
    index = {
        "embeddings": bulk.astype(np.float32),
        "chunk_ids": np.array(chunk_ids),
        "contents": np.array(contents),
        "sources": np.array(["тест.md"] * rows),
        "domains": np.array(["general"] * rows),
        "entity_types": np.array(["memory"] * rows),
        "model": "bge-m3",
        "dimension": DIMENSION,
    }
    return index


def _query() -> np.ndarray:
    vector = np.zeros(DIMENSION, dtype=np.float32)
    vector[0] = 1.0
    return vector


class DecisiveRuleTests(unittest.TestCase):
    def test_a_hit_below_the_absolute_floor_is_never_decisive(self):
        # Even when it stands far above the floor, which is the case a margin-only rule
        # gets wrong: measured, "how much is a ticket to Lviv" was that case.
        self.assertFalse(decisive_hit(SIMILARITY_THRESHOLD - 0.01, 0.05))
        self.assertFalse(decisive_hit(0.44, None))

    def test_a_hit_that_does_not_stand_out_is_not_decisive(self):
        floor = 0.42
        self.assertFalse(decisive_hit(floor + DECISIVE_MARGIN - 0.01, floor))

    def test_a_hit_that_stands_out_is_decisive(self):
        floor = 0.42
        self.assertTrue(decisive_hit(floor + DECISIVE_MARGIN + 0.001, floor))

    def test_the_margin_is_what_decides_not_a_higher_threshold(self):
        # Both of these clear the absolute floor by a wide margin; only the second stands
        # out from its own corpus.  A raised absolute threshold would separate neither.
        self.assertFalse(decisive_hit(0.62, 0.55))
        self.assertTrue(decisive_hit(0.62, 0.40))

    def test_an_unmeasurable_floor_falls_back_to_the_absolute_rule(self):
        # A small corpus: the quantile is one of the hits, so there is no floor to compare
        # against and the previous behaviour is the right one.
        self.assertTrue(decisive_hit(0.50, None))
        self.assertFalse(decisive_hit(0.40, None))


class CorpusFloorTests(unittest.TestCase):
    def _tier(self, index: dict):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "vectors.npz"
        save_index(path, index)
        patcher = patch("core.router.VECTOR_DB_PATH", str(path))
        patcher.start()
        self.addCleanup(patcher.stop)
        return Tier2Search()

    def test_a_small_corpus_has_no_measurable_floor(self):
        search = self._tier(_index(rows=DECISIVE_MIN_CORPUS - 1))
        self.assertIsNone(search.corpus_floor(_query()))

    def test_a_dense_corpus_has_a_high_floor(self):
        # Every chunk is nearly parallel to the query: nothing is an answer, everything
        # looks similar.  This is the situation the absolute threshold could not see.
        search = self._tier(_index(rows=DECISIVE_MIN_CORPUS + 50, spread=0.08))
        floor = search.corpus_floor(_query())
        self.assertIsNotNone(floor)
        self.assertGreater(floor, SIMILARITY_THRESHOLD)

    def test_the_floor_is_measured_per_query(self):
        # One corpus, two questions: the one whose topic the corpus covers densely gets a
        # higher floor than the one it does not.  A single number cannot do this, which is
        # the entire argument for the margin.
        search = self._tier(_index(rows=DECISIVE_MIN_CORPUS + 50, spread=0.15))
        on_topic = search.corpus_floor(_query())
        off_topic = np.zeros(DIMENSION, dtype=np.float32)
        off_topic[DIMENSION - 1] = 1.0
        self.assertGreater(on_topic, search.corpus_floor(off_topic))

    def test_a_demoted_hit_keeps_its_material(self):
        # The label changes, the serving does not: "material, but not the answer" is the
        # whole point of the demotion, and dropping the chunks would turn a demotion into
        # a miss.
        search = self._tier(_index(rows=DECISIVE_MIN_CORPUS + 50, spread=0.15))
        neighbours = search.search_vector("q", _query())
        self.assertTrue(neighbours, "the material was dropped instead of demoted")
        floor = search.corpus_floor(_query())
        self.assertIsNotNone(floor)
        self.assertFalse(decisive_hit(neighbours[0]["score"], floor))
        self.assertGreater(neighbours[0]["score"], SIMILARITY_THRESHOLD)

    def test_a_real_hit_in_a_dense_corpus_stays_decisive(self):
        # The bulk of this corpus sits near 0.5 against the query - about where the real
        # corpus sits (positives 0.48-0.64 against floors 0.33-0.55) - and one chunk is
        # exactly the query.  That is an answer, and it has to stay one.
        search = self._tier(_index(rows=DECISIVE_MIN_CORPUS + 50, spread=0.12,
                                   one_clear_winner=True))
        neighbours = search.search_vector("q", _query())
        floor = search.corpus_floor(_query())
        self.assertIsNotNone(floor)
        self.assertLess(floor, SIMILARITY_THRESHOLD + 0.10,
                        "the fixture's bulk drifted above the threshold band")
        self.assertTrue(decisive_hit(neighbours[0]["score"], floor))


class InstrumentVerdictTests(unittest.TestCase):
    """The instrument's own summary line, which used to cry failure for the wrong reason.

    A raw count called both failures the same thing.  A negative answered by the literal
    tier means the corpus *contains* the question's words - a quotation in a note or a
    commit message, which no amount of calibration can fix and only a new question set
    can work around - while one answered by the vector tier means the criterion let noise
    through, which is the criterion's own defect.  Reporting the first as "the criterion
    does not separate" sends the next reader to retune a rule that is working.
    """

    def _row(self, query, decisive, strategy):
        return {"query": query, "decisive": decisive, "strategy": strategy}

    def _instrument(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import measure_decisive_criterion
        return measure_decisive_criterion

    def test_a_quotation_in_the_corpus_is_not_the_criterion_failing(self):
        instrument = self._instrument()
        verdict = instrument.classify(
            positives=[self._row("настоящий вопрос", True, "exact_fts")],
            negatives=[self._row("вопрос, ответа на который нет", True, "exact_fts")],
        )
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertEqual(len(verdict.contaminated), 1)
        self.assertIn("буквальный", verdict.reason)

    def test_noise_through_the_vector_tier_is_the_criterion_failing(self):
        instrument = self._instrument()
        verdict = instrument.classify(
            positives=[self._row("настоящий вопрос", True, "vector_fast")],
            negatives=[self._row("шум", True, "vector_fast")],
        )
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.false_positives, ("шум",))
        self.assertEqual(verdict.contaminated, ())

    def test_a_lost_answer_names_the_question(self):
        instrument = self._instrument()
        verdict = instrument.classify(
            positives=[self._row("потерянный ответ", False, "vector_low_confidence")],
            negatives=[],
        )
        self.assertFalse(verdict.ok)
        self.assertIn("потерянный ответ", verdict.reason)

    def test_a_clean_table_says_so(self):
        instrument = self._instrument()
        verdict = instrument.classify(
            positives=[self._row("а", True, "exact_fts")],
            negatives=[self._row("б", False, "vector_low_confidence")],
        )
        self.assertTrue(verdict.ok, verdict.reason)
        self.assertIn("разделяет", verdict.reason)


if __name__ == "__main__":
    unittest.main()
