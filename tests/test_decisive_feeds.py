"""A conversation may inform a caller; it may not conclude for it.

Measured 2026-09-26, the day the sessions feed was indexed: with 2620 chunks of
conversation in the corpus, "what is the weather tomorrow in Kyiv" - a question the
corpus cannot answer at all - came back DECISIVE at 0.5158 against a floor of 0.3568.
Without that feed the same question was refused at 0.4359, and nothing was retuned in
between.  The corpus changed, and a conversation resembles any question, because
questions are most of what a conversation is.

Dropping reasoning and tool payloads at input took 74% of the history's text away and
was not enough on its own, so the feed loses the right to the *label* while keeping its
material: the chunks are still served, ranked and assembled into the context, and the
decision says which feed stopped it (`decisive_blocked`) instead of leaving that to be
inferred from a label that is missing.

What the tests below pin down:

* the rule itself, including that an unlabelled chunk is not treated as a conversation;
* that a conversation row cannot take the literal (FTS) exit, while a memory row still
  takes it on the same content - so the block is about the feed and not about the query;
* end to end through the router: a session hit that clears the whole decisive criterion
  is demoted and named, and the same hit from a project feed still decides, which is
  what makes this a rule about conversations rather than a second threshold;
* that the material survives the demotion.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from core.router import (  # noqa: E402
    DECISIVE_MIN_CORPUS,
    NON_DECISIVE_PREFIXES,
    SIMILARITY_THRESHOLD,
    JevRouter,
    Strategy,
    _gate_selection,
    decisive_feed,
)
from core.vector_index import save_index  # noqa: E402

DIMENSION = 256


class DecisiveFeedRuleTests(unittest.TestCase):
    def test_a_conversation_may_not_conclude(self):
        self.assertFalse(decisive_feed("ses:cli__20260428.md#3"))
        self.assertFalse(decisive_feed("gem:чат.md#4"))

    def test_every_other_feed_may(self):
        # Documents written to be read: a decision, a commit log, a README.
        for chunk_id in ("mem:заметка.md#1", "prj:Jev__commits.md#2",
                         "gh:drycool__md_reader__readme.md#0"):
            self.assertTrue(decisive_feed(chunk_id), chunk_id)

    def test_the_barred_list_is_about_transcripts_not_directories(self):
        # Two collectors, one class of document.  Pinned as a property, so adding a third
        # transcript feed without thinking about the criterion fails here.
        from core.memory_index import CORPUS_CHUNK_PREFIX, SESSIONS_CHUNK_PREFIX
        self.assertEqual(set(NON_DECISIVE_PREFIXES),
                         {SESSIONS_CHUNK_PREFIX, CORPUS_CHUNK_PREFIX})

    def test_an_unlabelled_chunk_is_not_assumed_to_be_a_conversation(self):
        # The rule is a statement about conversations.  A row whose id names no feed is
        # not one we know to be a conversation, and guessing would silently narrow
        # retrieval for sources that never asked for it.
        self.assertTrue(decisive_feed(""))
        self.assertTrue(decisive_feed("старый_корпус.md#0"))


class LiteralExitTests(unittest.TestCase):
    """The FTS gate: who may take the decisive exit on a literal match."""

    def _survivors(self, chunk_id: str) -> list[dict]:
        # Enough query terms in the chunk for the gate to accept it.
        return [{"chunk_id": chunk_id, "content": "порог подобия в Jev равен 0.45",
                 "source": "sessions.md", "score": 0.95}]

    def test_a_conversation_row_is_held_back_from_the_exit(self):
        decisive, held_back = _gate_selection("какой порог подобия в Jev", self._survivors("ses:cli__x.md#2"))
        self.assertEqual(decisive, [], "разговор забрал решающий выход")
        self.assertEqual(len(held_back), 1, "материал разговора потерян")
        self.assertEqual(held_back[0]["chunk_id"], "ses:cli__x.md#2")

    def test_a_memory_row_takes_it_and_the_conversation_rides_along(self):
        rows = self._survivors("mem:порог.md#1") + self._survivors("ses:cli__x.md#2")
        decisive, held_back = _gate_selection("какой порог подобия в Jev", rows)
        self.assertEqual([r["chunk_id"] for r in decisive], ["mem:порог.md#1"])
        self.assertEqual([r["chunk_id"] for r in held_back], ["ses:cli__x.md#2"])


class SessionHitEndToEnd(unittest.TestCase):
    """Through the router, with a synthetic index and no network."""

    def _router(self, top_chunk_id: str,
                top_text: str = "чанк про порог подобия в Jev"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        index_path = Path(directory.name) / "vectors.npz"
        fts_path = Path(directory.name) / "index.db"

        rng = np.random.default_rng(7)
        query_direction = np.zeros(DIMENSION, dtype=np.float32)
        query_direction[0] = 1.0
        rows = DECISIVE_MIN_CORPUS + 50
        embeddings = np.tile(query_direction, (rows, 1))
        embeddings = embeddings + rng.normal(0, 0.12, size=(rows, DIMENSION)).astype(np.float32)
        # One chunk is the query itself: a real answer, whatever its feed.
        embeddings[0] = query_direction
        chunk_ids = [top_chunk_id] + [f"mem:плотная_тема.md#{i}" for i in range(1, rows)]
        # The top chunk's text shares a content word with the question on purpose: these
        # tests are about which *feed* may conclude, so the lexical floor must be satisfied
        # and the namespace must be the only thing that differs.  A hit with no shared word
        # is a separate test below.
        contents = [top_text] + [f"чанк {i}" for i in range(1, rows)]
        save_index(index_path, {
            "embeddings": embeddings.astype(np.float32),
            "chunk_ids": np.array(chunk_ids),
            "contents": np.array(contents),
            "sources": np.array(["тест.md"] * rows),
            "domains": np.array(["general"] * rows),
            "entity_types": np.array([""] * rows),
            "model": "bge-m3",
            "dimension": DIMENSION,
        })

        patchers = [patch("core.router.FTS5_DB_PATH", str(fts_path)),
                    patch("core.router.VECTOR_DB_PATH", str(index_path))]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        router = JevRouter()
        self.addCleanup(router.tier2.close)
        # An FTS row that does not match the question, so the literal gate cannot fire
        # and the vector tier is what answers.
        router.tier2.index_chunk("mem:не_про_это.md#0", "совершенно другая тема про кабели")
        router.tier2.commit()

        async def embed(_query):
            return query_direction

        async def no_graph(*_args, **_kwargs):
            return {"response": ""}

        router.get_embedding = embed
        router.tier3.search = no_graph
        return router

    def test_a_session_hit_that_clears_the_criterion_is_named_and_demoted(self):
        result = asyncio.run(self._router("ses:cli__20260722.md#5").route("порог подобия в Jev"))
        decision = result.routing_decision
        # The criterion itself is satisfied - this is a hit aligned with the question in a
        # corpus that would otherwise call it decisive.
        self.assertGreater(decision.confidence_score, SIMILARITY_THRESHOLD)
        self.assertIsNotNone(decision.decisive_floor)
        self.assertNotEqual(decision.strategy, Strategy.VECTOR_FAST,
                            "разговор вынес вердикт")
        self.assertFalse(decision.local_material_decisive)
        self.assertEqual(decision.decisive_blocked, "ses:")
        # Материал не потерян: понижение — смена ярлыка, не удаление.
        self.assertIn("чанк про порог подобия", result.context or "")

    def test_a_chat_transcript_cannot_conclude_either(self):
        # The chats namespace holds conversations just as the sessions one does, and the
        # criterion does not care which collector wrote the transcript.  Barred *before*
        # the mass chat import, so the import cannot be the thing that breaks it.
        result = asyncio.run(self._router("gem:chats__export.md#2").route("порог подобия в Jev"))
        decision = result.routing_decision
        self.assertNotEqual(decision.strategy, Strategy.VECTOR_FAST,
                            "транскрипт беседы вынес вердикт")
        self.assertFalse(decision.local_material_decisive)
        self.assertEqual(decision.decisive_blocked, "gem:")
        self.assertIn("чанк про порог подобия", result.context or "")

    def test_a_hit_with_no_shared_words_is_not_a_verdict(self):
        """Measured failure: a decisive label with zero lexical support.

        On the fresh control set a question about a music subscription was called decisive
        at 0.4713 over a floor of 0.3473 - the margin rule was satisfied and the chunk
        shared not one content word with the question.  All seven answerable questions
        shared at least one.
        """
        # Same geometry, same namespace that decided above; only the text changes.
        router = self._router("prj:Jev__commits.md#5",
                              top_text="перечень цен на музыкальные сервисы и тарифы")
        result = asyncio.run(router.route("порог подобия в Jev"))
        decision = result.routing_decision
        self.assertGreater(decision.confidence_score, SIMILARITY_THRESHOLD)
        self.assertNotEqual(decision.strategy, Strategy.VECTOR_FAST)
        self.assertEqual(decision.decisive_blocked, "нет общих слов с вопросом")
        # The material is still handed over: the label was withheld, not the text.
        self.assertIn("музыкальные сервисы", result.context or "")

    def test_the_same_hit_from_a_project_feed_still_decides(self):
        # The control that makes the rule a statement about feeds rather than a second
        # threshold: identical geometry, identical content, different namespace.
        result = asyncio.run(self._router("prj:Jev__commits.md#5").route("порог подобия в Jev"))
        decision = result.routing_decision
        self.assertEqual(decision.strategy, Strategy.VECTOR_FAST)
        self.assertTrue(decision.local_material_decisive)
        self.assertIsNone(decision.decisive_blocked)


if __name__ == "__main__":
    unittest.main()
