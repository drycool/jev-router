"""
Ground truth: what can and cannot be labelled, and why.
=======================================================

The point of this file is the boundary it draws. The router records `signals` about itself
- which tier answered, how long it took, whether an answer came out - and none of that is a
judgement, because the router cannot know whether its own answer was right. A verdict can
only come from a consumer, and it lands in a second append-only file keyed by `decision_id`,
because a verdict arrives after the fact and an append-only decision record cannot be
amended.

These tests exist to keep that boundary from eroding. The day a `verdict` field appears in a
decision record "for convenience", the log starts looking like a dataset while containing no
labels at all - which is exactly the illusion this project has already paid for once.
"""
import asyncio
import json
import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import api.server as server
from agents.base import AgentResponse
from core.router import (
    AgentType,
    ExtractedMetadata,
    RAGConfiguration,
    RoutingDecision,
    RoutingResult,
    Strategy,
)


class CapturedLogs:
    """Capture the JSONL lines a logger would have written to disk.

    The handlers are *swapped out*, not added to: these tests must never append to the
    production decision or feedback logs, and asserting on the line that was actually
    emitted is both cleaner and more truthful than reading back a file afterwards.
    """

    def __init__(self, logger):
        self.logger = logger
        self.lines = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.lines.append(record.getMessage())

    def __enter__(self):
        self.saved = list(self.logger.handlers)
        self.logger.handlers = [self.handler]
        return self

    def __exit__(self, *exc):
        self.logger.handlers = self.saved
        return False

    def events(self):
        return [json.loads(line) for line in self.lines]


def make_result(strategy=Strategy.EXACT_FTS, context="", **kwargs):
    return RoutingResult(
        routing_decision=RoutingDecision(
            strategy=strategy, confidence_score=0.95, fast_path_exit=False
        ),
        extracted_metadata=ExtractedMetadata(
            intent="exact_search", keywords=["момент"], domain="general"
        ),
        rag_configuration=RAGConfiguration(lightrag_required=False, lightrag_mode="skip"),
        target_agent=AgentType.GENERAL,
        context=context,
        **kwargs,
    )


def write_decision_log(entries):
    """Write a throwaway decision log and return its path."""
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for entry in entries:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    handle.close()
    return handle.name


class TestDecisionIdentity(unittest.TestCase):
    def test_identical_queries_get_distinct_decision_ids(self):
        """The reason decision_id exists at all.

        query_hash is sha256 of the query text, so the same question asked twice produces
        two records that cannot be told apart - and therefore no verdict can be attached to
        either. The hash is still there for grouping; the id is what makes labelling
        possible.
        """
        query = "какой момент затяжки болтов головки блока цилиндров"
        with CapturedLogs(server.decision_logger) as captured:
            server._record_decision(query, make_result(), 10.0, False, decision_id="a" * 32)
            server._record_decision(query, make_result(), 20.0, False, decision_id="b" * 32)

        first, second = captured.events()
        self.assertEqual(first["query_hash"], second["query_hash"], "hashes must collide")
        self.assertNotEqual(first["decision_id"], second["decision_id"])


class TestDecisionRecordShape(unittest.TestCase):
    def record(self, **kwargs):
        with CapturedLogs(server.decision_logger) as captured:
            server._record_decision(
                "вопрос", make_result(context="контекст"), 1.0, False, decision_id="c" * 32, **kwargs
            )
            return captured.events()[0]

    def test_record_is_marked_v2(self):
        self.assertEqual(self.record()["schema"], "decision_v2")

    def test_record_contains_no_verdict_or_label_field(self):
        """The boundary this file exists to protect.

        A decision record must not gain a truth-shaped field. If it does, the log looks
        like a dataset while holding no labels, and the next person to read it will
        reasonably assume the answers were checked.
        """
        event = self.record(agent_response="25 Н·м")
        flattened = json.dumps(event, ensure_ascii=False).lower()
        for forbidden in ("verdict", "label", "ground_truth", "user_accepted", "correct"):
            self.assertNotIn(forbidden, flattened, f"decision record must not claim {forbidden!r}")
        self.assertNotIn("outcome", event)

    def test_signals_report_observations_not_judgements(self):
        event = self.record(agent_response="25 Н·м в несколько этапов")
        signals = event["signals"]
        self.assertTrue(signals["answered"])
        self.assertGreater(signals["answer_chars"], 0)
        self.assertEqual(signals["context_chars"], len("контекст"))
        self.assertIn("tier", signals)

    def test_an_empty_answer_is_recorded_as_unanswered(self):
        """execute=false legitimately produces no answer; the log must say so plainly
        rather than leaving a field that looks like a missing value."""
        signals = self.record(execute=False)["signals"]
        self.assertFalse(signals["answered"])
        self.assertEqual(signals["answer_chars"], 0)
        self.assertFalse(signals["execute_requested"])

    def test_tier_is_derived_from_the_strategy(self):
        for strategy, tier in (
            (Strategy.DIRECT_ACTION, "tier1"),
            (Strategy.EXACT_FTS, "tier2"),
            (Strategy.VECTOR_FAST, "tier2"),
            (Strategy.VECTOR_LOW_CONFIDENCE, "tier2"),
            (Strategy.FTS_FALLBACK, "tier2"),
            (Strategy.GRAPH_LIGHTRAG, "tier3"),
            (Strategy.GENERAL_LLM, "tier4"),
        ):
            with self.subTest(strategy=strategy):
                with CapturedLogs(server.decision_logger) as captured:
                    server._record_decision(
                        "q", make_result(strategy=strategy), 1.0, False, decision_id="d" * 32
                    )
                    self.assertEqual(captured.events()[0]["signals"]["tier"], tier)

    def test_every_strategy_names_its_tier_explicitly(self):
        """The bug this test exists for.

        The map's default was "tier4", so when the two local fallbacks were added they were
        logged as LLM answers without a single test noticing: the list above used to be
        written out by hand, and a hand-written list cannot fail for a strategy that is not
        in it.  Exhaustive coverage is what makes the default unreachable, so this test
        enumerates the enum rather than repeating a subset of it.
        """
        unmapped = [s.value for s in Strategy if s.value not in server._TIER_OF]
        self.assertEqual(unmapped, [], f"стратегии без явного яруса: {unmapped}")

    def test_a_strategy_the_map_does_not_know_is_not_reported_as_a_tier(self):
        """A missing tier must not look like a real one, or the log will be trusted."""
        invented = SimpleNamespace(value="invented_strategy")
        # A stub on purpose: the point is a strategy the map has never seen.
        self.assertEqual(server._tier_of(invented), "unknown")  # type: ignore[arg-type]

    def test_preview_is_bounded(self):
        long_answer = "я" * (server.PREVIEW_CHARS * 5)
        signals = self.record(agent_response=long_answer)["signals"]
        self.assertEqual(len(signals["answer_preview"]), server.PREVIEW_CHARS)
        self.assertEqual(signals["answer_chars"], len(long_answer))

    def test_preview_can_be_switched_off(self):
        """The privacy switch has to actually work, or the default is not a choice."""
        with patch.object(server, "LOG_ANSWER_PREVIEW", False):
            signals = self.record(agent_response="секретный ответ")["signals"]
        self.assertIsNone(signals["answer_preview"])
        self.assertIsNone(signals["context_preview"])
        # The size stays: it is a signal about the answer, not the answer itself.
        self.assertGreater(signals["answer_chars"], 0)


class TestFeedbackEndpoint(unittest.TestCase):
    def verdict(self, decision_id="e" * 32, verdict="accepted", source="human", comment="", query=""):
        with CapturedLogs(server.feedback_logger) as captured:
            response = asyncio.run(
                server.feedback(
                    server.FeedbackRequest(
                        decision_id=decision_id,
                        verdict=verdict,
                        source=source,
                        comment=comment,
                        query=query,
                    )
                )
            )
        return response, captured.events()

    def test_the_reviewer_can_attach_the_question_they_judged(self):
        """The decision log keeps only a hash of the query, so a label without the question
        cannot be re-judged by anyone else. The reviewer has it in hand at the moment of
        judging, and this is the only point at which it can be captured without turning on
        raw-query logging globally."""
        _, events = self.verdict(query="какой момент затяжки болтов головки блока цилиндров")
        self.assertEqual(events[0]["query"], "какой момент затяжки болтов головки блока цилиндров")

    def test_an_absent_question_is_null_rather_than_missing(self):
        """A stable shape lets a reader tell 'the reviewer did not provide the question' from
        'this field did not exist yet'."""
        _, events = self.verdict()
        self.assertIn("query", events[0])
        self.assertIsNone(events[0]["query"])

    def test_an_absurdly_long_question_is_rejected(self):
        with self.assertRaises(Exception):
            server.FeedbackRequest(
                decision_id="a" * 32, verdict="accepted", query="я" * 4001
            )

    def test_a_verdict_is_appended_not_merged_into_the_decision_log(self):
        response, events = self.verdict()

        self.assertTrue(response.recorded)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["schema"], "feedback_v1")
        self.assertEqual(events[0]["verdict"], "accepted")
        self.assertEqual(events[0]["source"], "human")

    def test_a_correction_reports_what_it_corrects(self):
        """Two verdicts on one decision is the normal sequence - an agent's guess, then a
        human's correction - and the second must be visibly a correction."""
        with patch.object(server, "_feedback_history", return_value=("accepted", 1)):
            response, _ = self.verdict(verdict="rejected", source="human", comment="не тот момент")

        self.assertEqual(response.previous_verdict, "accepted")
        self.assertEqual(response.verdicts_for_decision, 2)
        self.assertEqual(response.verdict, "rejected")

    def test_an_unknown_decision_is_recorded_but_flagged(self):
        """A human's verdict is the most expensive data here, so it is never dropped over
        our own bookkeeping - but it must not silently inflate the label count either."""
        with patch.object(server, "_decision_id_known", return_value=False):
            before = server._stats["feedback_orphans"]
            response, events = self.verdict(decision_id="f" * 32)

        self.assertTrue(response.recorded)
        self.assertFalse(response.known_decision)
        self.assertFalse(events[0]["known_decision"])
        self.assertEqual(server._stats["feedback_orphans"], before + 1)

    def test_a_known_decision_is_recognised(self):
        path = write_decision_log([{"decision_id": "a1b2c3", "schema": "decision_v2"}])
        try:
            with patch.object(server, "DECISION_LOG_PATH", path):
                self.assertTrue(server._decision_id_known("a1b2c3"))
                self.assertFalse(server._decision_id_known("нет-такого"))
        finally:
            os.unlink(path)

    def test_reading_a_missing_feedback_log_is_not_an_error(self):
        with patch.object(server, "FEEDBACK_LOG_PATH", "/nonexistent/feedback.jsonl"):
            self.assertEqual(server._feedback_history("anything"), (None, 0))

    def test_a_malformed_line_does_not_break_the_reader(self):
        path = write_decision_log([{"decision_id": "keep"}])
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("{это не json}\n")
            with patch.object(server, "DECISION_LOG_PATH", path):
                self.assertTrue(server._decision_id_known("keep"))
        finally:
            os.unlink(path)


class TestFeedbackValidation(unittest.TestCase):
    def test_an_unusable_verdict_is_rejected(self):
        """'unknown' and 'maybe' carry no signal; accepting them would inflate the label
        count with abstentions."""
        for bad in ("unknown", "maybe", "correct", ""):
            with self.subTest(verdict=bad):
                with self.assertRaises(Exception):
                    server.FeedbackRequest(decision_id="a" * 32, verdict=bad)

    def test_an_unknown_source_is_rejected(self):
        with self.assertRaises(Exception):
            server.FeedbackRequest(decision_id="a" * 32, verdict="accepted", source="robot")

    def test_an_empty_decision_id_is_rejected(self):
        with self.assertRaises(Exception):
            server.FeedbackRequest(decision_id="", verdict="accepted")


class TestQueryEndpointWiring(unittest.TestCase):
    class FakeRouter:
        def __init__(self, result, answer="", explode=False):
            self.result = result
            self.answer = answer
            self.explode = explode
            self.tier4 = self

        async def route(self, query):
            return self.result

        async def execute(self, result):
            if self.explode:
                raise RuntimeError("agent tier down")
            return AgentResponse(agent="general_agent", answer=self.answer)

    def run_query(self, result, answer="", explode=False, execute=True):
        fake = self.FakeRouter(result, answer, explode)
        with patch.object(server, "router", fake), CapturedLogs(server.decision_logger) as captured:
            response = asyncio.run(server.query(server.QueryRequest(query="вопрос", execute=execute)))
        return response, captured.events()

    def test_the_response_carries_the_id_of_the_record_it_produced(self):
        """Without this the caller has no handle to attach a verdict to, and the whole
        feedback path is unreachable in practice."""
        response, events = self.run_query(make_result(), answer="ответ", execute=False)

        self.assertTrue(response.decision_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["decision_id"], response.decision_id)

    def test_two_calls_produce_two_distinct_ids(self):
        first, _ = self.run_query(make_result(), answer="ответ", execute=False)
        second, _ = self.run_query(make_result(), answer="ответ", execute=False)
        self.assertNotEqual(first.decision_id, second.decision_id)

    def test_a_successful_answer_is_recorded_as_answered(self):
        _, events = self.run_query(make_result(), answer="25 Н·м в несколько этапов")
        signals = events[0]["signals"]
        self.assertTrue(signals["answered"])
        self.assertIn("25 Н·м", signals["answer_preview"])

    def test_an_agent_failure_is_visible_in_signals_and_execution(self):
        """A failure must not be silently recorded as a normal answer, or the log would
        show a healthy tier where there was none."""
        response, events = self.run_query(make_result(), explode=True)
        signals, execution = events[0]["signals"], events[0]["execution"]

        self.assertTrue(execution["agent_error"])
        self.assertIn("Agent error", response.agent_response)
        # The placeholder text is an answer-shaped string, so `answered` is true; what
        # distinguishes a failure is agent_error, which is why both are recorded.
        self.assertTrue(signals["answered"])
        self.assertTrue(execution["agent_error"])

    def test_stats_expose_the_feedback_counters(self):
        stats = asyncio.run(server.stats())
        self.assertIn("feedback", stats.model_dump())
        self.assertIn("feedback_orphans", stats.model_dump())

    def test_health_declares_how_labelling_is_configured(self):
        health = asyncio.run(server.health())
        ground_truth = health["ground_truth"]

        self.assertIn("feedback_log", ground_truth)
        self.assertIn("accepted", ground_truth["verdicts"])
        self.assertIsInstance(ground_truth["answer_preview_logged"], bool)


class TestProductionLogsAreProtected(unittest.TestCase):
    """The redirect in tests/__init__.py is the only thing standing between the suite and
    the production logs, and it is invisible in any single test file.

    It exists because ShadowProbeTests posts to /query, which appends a decision record for
    its fixture query on every run: 20 of the 71 records in the log at the time of writing,
    28%, were test scaffolding. Removing the redirect must therefore break these tests, not
    quietly resume corrupting the dataset.
    """

    def setUp(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.production_decision_log = os.path.join(project_root, "jev_decisions.jsonl")
        self.production_feedback_log = os.path.join(project_root, "jev_feedback.jsonl")

    def test_the_decision_log_is_redirected_away_from_production(self):
        self.assertNotEqual(
            os.path.abspath(server.DECISION_LOG_PATH),
            os.path.abspath(self.production_decision_log),
            "the test suite would append to the production decision log",
        )

    def test_the_feedback_log_is_redirected_away_from_production(self):
        self.assertNotEqual(
            os.path.abspath(server.FEEDBACK_LOG_PATH),
            os.path.abspath(self.production_feedback_log),
            "the test suite would append to the production feedback log",
        )

    def test_the_redirect_does_not_depend_on_how_the_suite_was_started(self):
        """The bypass this mechanism exists for.

        `python3 -m unittest discover -s tests` imports the test modules as top-level ones,
        so tests/__init__.py never runs and its redirect never happens: the suite appended
        two fixture records, 1824 bytes, to the production decision log, and the assertion
        below reported it only after the writes were already on disk. A canary reports; it
        does not prevent. The server now recognises a test run itself, so the guarantee
        rests on the write site rather than on the invocation.
        """
        self.assertTrue(server._running_under_test())
        self.assertEqual(
            os.path.dirname(os.path.abspath(server.DECISION_LOG_PATH)),
            os.path.join(tempfile.gettempdir(), "jev-test-logs"),
        )

    def test_the_handlers_actually_opened_the_redirected_paths(self):
        """The constant being right is not enough: the FileHandler binds a filename at
        import time, so a logger opened before the redirect would still write to production
        while every configuration value said otherwise."""
        for logger, expected in (
            (server.decision_logger, server.DECISION_LOG_PATH),
            (server.feedback_logger, server.FEEDBACK_LOG_PATH),
        ):
            with self.subTest(logger=logger.name):
                opened = [
                    os.path.abspath(handler.baseFilename)
                    for handler in logger.handlers
                    if hasattr(handler, "baseFilename")
                ]
                if not opened:
                    self.skipTest(f"{logger.name} has no file handler in this process")
                self.assertIn(os.path.abspath(expected), opened)


if __name__ == "__main__":
    unittest.main()
