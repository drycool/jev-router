"""The fast path: answering from local material without calling the model.

The defect these tests exist for: tier 2 already returns the answer text, not a pointer to
it. On the acceptance question the router held 1209 characters of exactly the right material
after 166 ms as `exact_fts` at 0.95 - and `/query` sent that text to a 9B model anyway,
because the only voice in the decision to call tier 4 was the caller's `execute` flag, which
had no idea what routing had found. The request took 12617 ms and came back with the model's
reasoning about the question ("The question is in Russian... Let me base my answer on the
provided context.") instead of the answer to it.

Two properties are pinned here, and they are the same property at two levels: the material
decides whether the model is needed, and the answer says which material decided it.
"""
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import server
from core.router import (
    FAST_PATH_FTS_MIN,
    FAST_PATH_NO_EXIT,
    FAST_PATH_REASON_FTS,
    FAST_PATH_REASON_VECTOR,
    FAST_PATH_VECTOR_MIN,
    SIMILARITY_THRESHOLD,
    AgentType,
    ExtractedMetadata,
    RAGConfiguration,
    RoutingDecision,
    RoutingResult,
    Strategy,
    assemble_context,
    fast_path_exit,
    fast_path_reason,
)


def decision(strategy: Strategy, confidence: float) -> RoutingDecision:
    return RoutingDecision(strategy=strategy, confidence_score=confidence, tier1_exit=False)


def result_for(
    strategy: Strategy = Strategy.EXACT_FTS,
    confidence: float = 0.95,
    context: str = "КОНТЕКСТ ОТВЕТА",
    sources: list | None = None,
) -> RoutingResult:
    return RoutingResult(
        routing_decision=decision(strategy, confidence),
        extracted_metadata=ExtractedMetadata(intent="exact_search", keywords=["systemd"]),
        rag_configuration=RAGConfiguration(lightrag_required=False, lightrag_mode="skip"),
        target_agent=AgentType.GENERAL,
        query="Как у нас настраиваются пользовательские systemd сервисы?",
        context=context,
        context_stats={"sources": sources if sources is not None else ["/home/dry/memory/x.md"]},
    )


class FastPathReasonTests(unittest.TestCase):
    """Which local results earn an exit, derived from the same bit consumers read."""

    def test_an_exact_literal_match_earns_an_exit(self):
        self.assertEqual(fast_path_reason(decision(Strategy.EXACT_FTS, 0.95)),
                         FAST_PATH_REASON_FTS)

    def test_a_decisive_semantic_match_earns_an_exit(self):
        self.assertEqual(fast_path_reason(decision(Strategy.VECTOR_FAST, 0.6032)),
                         FAST_PATH_REASON_VECTOR)

    def test_an_exact_match_below_the_bar_does_not(self):
        """The bar is a formality today (the FTS gate reports a fixed 0.95), and it is here
        so that the day a real confidence arrives it lands somewhere that is already
        tested rather than somewhere that always says yes."""
        self.assertIsNone(fast_path_reason(decision(Strategy.EXACT_FTS, FAST_PATH_FTS_MIN - 0.01)))

    def test_a_semantic_match_below_the_bar_does_not(self):
        self.assertIsNone(
            fast_path_reason(decision(Strategy.VECTOR_FAST, FAST_PATH_VECTOR_MIN - 0.001))
        )

    def test_every_strategy_is_accounted_for(self):
        """A total function over the enum, not a list.

        The strategy list has been written out by hand in this branch before, and a
        hand-written list cannot fail for the strategy that is not in it - which is how two
        local fallbacks came to be logged as LLM answers. Every status must either earn an
        exit or be named in FAST_PATH_NO_EXIT.
        """
        for strategy in Strategy:
            with self.subTest(strategy=strategy):
                earned = fast_path_exit(decision(strategy, 1.0))
                listed = strategy in FAST_PATH_NO_EXIT
                self.assertTrue(earned or listed,
                                f"{strategy.value} earns no exit and is not listed as such")
                self.assertFalse(earned and listed,
                                 f"{strategy.value} both earns an exit and is denied one")

    def test_a_high_confidence_weak_pool_still_does_not_earn_an_exit(self):
        """The number follows the status, never the other way round: 0.99 from a refused
        gate is not material, and no threshold change may make it look like material."""
        self.assertIsNone(fast_path_reason(decision(Strategy.FTS_FALLBACK, 0.99)))
        self.assertIsNone(fast_path_reason(decision(Strategy.VECTOR_LOW_CONFIDENCE, 0.99)))

    def test_the_semantic_bar_is_not_above_the_calibrated_hit_range(self):
        """Guards the reachability of this whole path, which is what 0.90 for both
        strategies would have destroyed.

        bge-m3 compresses related pairs on this corpus to 0.55-0.67: measured decisive hits
        are 0.5574-0.6721. A 0.90 vector bar would never fire in production while passing
        every unit test, because a unit test can set the confidence it wants. The bar must
        therefore never exceed the threshold the tier itself uses to call a hit decisive.
        """
        self.assertLessEqual(
            FAST_PATH_VECTOR_MIN, SIMILARITY_THRESHOLD,
            "векторная планка выше порога, по которому тир считает попадание решающим",
        )
        self.assertLess(FAST_PATH_VECTOR_MIN, 0.5574,
                        "планка выше самого слабого измеренного решающего попадания")


class DirectAnswerTests(unittest.TestCase):
    """What a fast-path answer actually is."""

    def test_the_answer_is_the_material_itself(self):
        answer = server._direct_answer(result_for(context="материал"))
        self.assertIn("материал", answer)

    def test_the_answer_names_its_sources(self):
        """On this path the text IS the answer, so an answer that does not say which file it
        came from cannot be checked against the file."""
        answer = server._direct_answer(
            result_for(sources=["/home/dry/memory/projects/a.md", "/home/dry/Jev/README.md"])
        )
        self.assertIn("/home/dry/memory/projects/a.md", answer)
        self.assertIn("/home/dry/Jev/README.md", answer)

    def test_an_empty_context_produces_no_answer(self):
        """An exit with nothing to return must not manufacture one."""
        self.assertEqual(server._direct_answer(result_for(context="", sources=[])), "")

    def test_assemble_context_records_where_the_used_chunks_came_from(self):
        context, stats = assemble_context([
            {"content": "первый", "source": "/a.md"},
            {"content": "второй", "source": "/b.md"},
            {"content": "первый", "source": "/a.md"},  # duplicate: must not appear twice
        ])
        self.assertEqual(stats["sources"], ["/a.md", "/b.md"])
        self.assertEqual(stats["chunks_used"], 2)


class QueryFastPathTests(unittest.TestCase):
    """The gate itself, through the endpoint a consumer actually calls."""

    class Agent:
        """Stands in for tier 4 and records whether it was asked."""

        def __init__(self):
            self.calls = 0

        async def execute(self, _result):
            self.calls += 1

            class Response:
                answer = "ОТВЕТ ОТ МОДЕЛИ"

            return Response()

    def call(self, result: RoutingResult, execute: bool = True):
        class Router:
            async def route(self, _query):
                return result

        agent = self.Agent()
        router = Router()
        router.tier4 = agent
        with patch.object(server, "router", router):
            client = TestClient(server.app)
            response = client.post("/query", json={"query": result.query, "execute": execute})
        return response, agent

    def test_decisive_material_answers_without_calling_the_model(self):
        """The acceptance criterion: the systemd question must not reach tier 4."""
        response, agent = self.call(result_for())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["routing_decision"]["fast_path_exit"])
        self.assertEqual(payload["routing_decision"]["fast_path_reason"], FAST_PATH_REASON_FTS)
        self.assertEqual(agent.calls, 0, "модель не должна вызываться на быстром пути")
        self.assertIn("КОНТЕКСТ ОТВЕТА", payload["agent_response"])

    def test_a_decisive_semantic_match_also_skips_the_model(self):
        response, agent = self.call(result_for(Strategy.VECTOR_FAST, 0.6032))
        payload = response.json()
        self.assertEqual(payload["routing_decision"]["fast_path_reason"], FAST_PATH_REASON_VECTOR)
        self.assertEqual(agent.calls, 0)

    def test_weak_material_still_reaches_the_model(self):
        """The exit must not become a way to avoid answering: material that did not decide
        anything is exactly when the model is needed."""
        response, agent = self.call(result_for(Strategy.VECTOR_LOW_CONFIDENCE, 0.4231))
        payload = response.json()
        self.assertFalse(payload["routing_decision"]["fast_path_exit"])
        self.assertIsNone(payload["routing_decision"]["fast_path_reason"])
        self.assertEqual(agent.calls, 1)
        self.assertEqual(payload["agent_response"], "ОТВЕТ ОТ МОДЕЛИ")

    def test_a_broken_retriever_still_reaches_the_model(self):
        response, agent = self.call(result_for(Strategy.EMBEDDING_TIMEOUT, 0.0, context=""))
        self.assertEqual(agent.calls, 1)
        self.assertFalse(response.json()["routing_decision"]["fast_path_exit"])

    def test_execute_false_still_returns_no_answer(self):
        """The MCP tool's contract: execute=false means the routing decision and the context,
        with no prose. An exit found while routing must not override the caller here, or the
        tool would receive an answer it explicitly did not ask for."""
        response, agent = self.call(result_for(), execute=False)
        payload = response.json()
        self.assertEqual(agent.calls, 0, "execute=false уже означал отсутствие вызова модели")
        self.assertEqual(payload["agent_response"], "")
        self.assertFalse(payload["routing_decision"]["fast_path_exit"],
                         "выход не состоялся: ответа не просили")

    def test_route_only_reports_both_exit_claims_separately(self):
        """Tier 1's own signal and the request-level outcome are different claims, and a
        diagnostic endpoint that conflates them is how the name collided in the first place.
        """
        class Router:
            async def route(self, _query):
                return result_for()

        with patch.object(server, "router", Router()):
            client = TestClient(server.app)
            routing = client.get("/route-only", params={"query": "q"}).json()["routing_decision"]
        self.assertTrue(routing["fast_path_exit"])
        self.assertIs(routing["tier1_exit"], False)


class FastPathJournalTests(unittest.TestCase):
    """The log must let a reader tell "the caller wanted an agent" from "an agent ran"."""

    def record(self, reason, execute=True):
        import logging
        from io import StringIO

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        server.decision_logger.addHandler(handler)
        try:
            server._record_decision(
                "q",
                result_for(sources=["/a.md"]),
                5.0,
                False,
                decision_id="e" * 32,
                agent_response="ответ",
                execute=execute,
                fast_path=reason,
            )
        finally:
            server.decision_logger.removeHandler(handler)
        return json.loads(stream.getvalue().strip().splitlines()[-1])

    def test_the_reason_is_recorded_next_to_what_the_caller_asked_for(self):
        signals = self.record(FAST_PATH_REASON_FTS)["signals"]
        self.assertTrue(signals["execute_requested"])
        self.assertTrue(signals["fast_path_exit"])
        self.assertEqual(signals["fast_path_reason"], "fts_exact_high_confidence")

    def test_a_request_that_did_reach_the_model_says_so(self):
        signals = self.record(None)["signals"]
        self.assertTrue(signals["execute_requested"])
        self.assertFalse(signals["fast_path_exit"])
        self.assertIsNone(signals["fast_path_reason"])

    def test_the_sources_of_a_fast_path_answer_are_recorded(self):
        """Provenance belongs in the log for the same reason it belongs in the answer: the
        material is the answer on this path, and it is not reproducible from a hash."""
        self.assertEqual(self.record(FAST_PATH_REASON_FTS)["signals"]["context_sources"],
                         ["/a.md"])


class FastPathMetricsTests(unittest.TestCase):
    def test_the_counter_is_exposed_in_prometheus_format(self):
        server._stats["fast_path_exits"] += 1
        server._stats["fast_path_exits_by_strategy"][Strategy.EXACT_FTS.value] = (
            server._stats["fast_path_exits_by_strategy"].get(Strategy.EXACT_FTS.value, 0) + 1
        )
        client = TestClient(server.app)
        body = client.get("/metrics").text
        self.assertIn("# TYPE jev_fast_path_exits_total counter", body)
        self.assertRegex(body, r'jev_fast_path_exits_total\{strategy="exact_fts"\} \d+')

    def test_stats_exposes_the_per_strategy_breakdown(self):
        payload = TestClient(server.app).get("/stats").json()
        self.assertIn("fast_path_exits", payload)
        self.assertIn("fast_path_exits_by_strategy", payload)


if __name__ == "__main__":
    unittest.main()
