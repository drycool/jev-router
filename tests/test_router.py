import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

from api import server
from core.env import load_env
from core.decision_engine import DecisionEngineClient, DecisionEngineResult
from core.laya_client import LayaDecision, LayaTier1Client, not_awaited
from core.shadow import ShadowProbe
from core.router import (
    AgentExecutor,
    AgentType,
    ExtractedMetadata,
    RAGConfiguration,
    RoutingDecision,
    RoutingResult,
    SIMILARITY_THRESHOLD,
    Strategy,
    Tier2Search,
    JevRouter,
    tier1_fast_route,
)


class _Agent:
    async def execute(self, query, context, metadata):
        return query


class RouterTests(unittest.TestCase):
    def test_direct_route_keeps_original_query(self):
        query = "run pytest -q"
        result = tier1_fast_route(query)
        self.assertEqual(result.query, query)
        self.assertEqual(result.target_agent, AgentType.CODE)

    def test_agent_receives_original_query_not_router_intent(self):
        result = RoutingResult(
            RoutingDecision(Strategy.DIRECT_ACTION, 1.0, True),
            ExtractedMetadata(intent="execute_code"),
            RAGConfiguration(),
            AgentType.CODE,
            query="run pytest -q",
        )
        executor = AgentExecutor()
        executor.register_agent(AgentType.CODE, _Agent())
        self.assertEqual(asyncio.run(executor.execute(result)), "run pytest -q")

    def test_fts_rebuild_does_not_duplicate_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "index.db")
            with patch("core.router.FTS5_DB_PATH", db_path):
                search = Tier2Search()
                search.index_chunk("one", "Raspberry Pi cable")
                search.commit()
                search.clear()
                search.index_chunk("one", "Raspberry Pi cable")
                search.commit()
                self.assertEqual(len(search.search_fts5("Raspberry")), 1)
                search.close()

    def test_vector_search_rejects_wrong_dimensions(self):
        # A stale index with a different dimension must not produce a hit.
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vectors.npz"
            np.savez(
                vector_path,
                embeddings=np.array([[1.0, 0.0]]),
                chunk_ids=np.array(["one"]), contents=np.array(["content"]), sources=np.array([""]),
                domains=np.array(["general"]),
                model=np.array("mxbai-embed-large"), dimension=np.array(2),
            )
            with patch("core.router.VECTOR_DB_PATH", str(vector_path)):
                with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                    search = Tier2Search()
                    self.assertEqual(search.search_vector("q", np.array([1.0, 0.0, 0.0])), [])
                    search.close()

    def test_vector_index_rejects_other_embedding_model(self):
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vectors.npz"
            np.savez(
                vector_path,
                embeddings=np.array([[1.0, 0.0]]),
                chunk_ids=np.array(["one"]), contents=np.array(["content"]), sources=np.array([""]),
                domains=np.array(["general"]),
                model=np.array("another-model"), dimension=np.array(2),
            )
            with patch("core.router.VECTOR_DB_PATH", str(vector_path)):
                with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                    search = Tier2Search()
                    self.assertEqual(search.search_vector("q", np.array([1.0, 0.0])), [])
                    search.close()

    def test_exact_fts_does_not_request_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()
                router.tier2.index_chunk("one", "Raspberry Pi cable cable cable")
                router.tier2.commit()
                async def embedding_must_not_run(_):
                    raise AssertionError("embedding should be skipped for FTS hit")
                router.get_embedding = embedding_must_not_run
                result = asyncio.run(router.route("Raspberry Pi cable"))
                self.assertEqual(result.routing_decision.strategy, Strategy.EXACT_FTS)
                router.tier2.close()

    def test_fts_rank_is_not_mistaken_for_vector_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()
                router.tier2.index_chunk("one", "Raspberry unrelated")
                router.tier2.commit()
                async def no_embedding(_):
                    return None
                async def no_graph(*_, **__):
                    return {"response": ""}
                router.get_embedding = no_embedding
                router.tier3.search = no_graph
                result = asyncio.run(router.route("Raspberry Pi cable"))
                self.assertEqual(result.routing_decision.strategy, Strategy.GRAPH_LIGHTRAG)
                router.tier2.close()

    def test_lightrag_failure_degrades_to_fts_context(self):
        # The graph tier is pinned on explicitly. Importing api.server now loads the
        # project's .env (see core/env.py), and a test must not change meaning depending on
        # the developer's local configuration.
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", True):
                    router = JevRouter()
                    router.tier2.index_chunk("one", "Raspberry unrelated")
                    router.tier2.commit()
                    async def no_embedding(_):
                        return None
                    async def timed_out(*_, **__):
                        return {"response": "", "error_type": "timeout"}
                    router.get_embedding = no_embedding
                    router.tier3.search = timed_out
                    result = asyncio.run(router.route("Raspberry Pi cable"))
                    self.assertTrue(result.degraded)
                    self.assertEqual(result.fallback_reason, "lightrag_timeout")
                    self.assertIn("Raspberry", result.context)
                    router.tier2.close()

    def test_laya_direct_command_cannot_bypass_regex_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()
                class Laya:
                    async def predict_routing(self, _):
                        # A fully trusted classifier: even then it must not gain
                        # authority to execute anything.
                        return LayaDecision(
                            strategy="direct_cmd", domain="general", confidence=0.99,
                            task="code", task_confidence=0.99, status="success",
                        )
                async def no_embedding(_): return None
                async def no_graph(*_, **__): return {"response": ""}
                router.laya = Laya()
                router.get_embedding = no_embedding
                router.tier3.search = no_graph
                result = asyncio.run(router.route("please erase all documents"))
                self.assertEqual(result.target_agent, AgentType.GENERAL)
                self.assertEqual(result.laya_result["execution_override"], "direct_cmd_requires_regex_match")
                router.tier2.close()

    def test_local_exact_hit_does_not_wait_for_the_classifier(self):
        """An exactly answerable query measured 56.0 ms end to end, of which
        55.3 ms was the GPU2 round trip - and the answer never used the verdict.
        Nothing in the local path needs the classifier to answer."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()
                router.tier2.index_chunk("one", "Raspberry Pi cable cable cable GPIO pinout")
                router.tier2.commit()

                class SlowLaya:
                    async def predict_routing(self, _):
                        await asyncio.sleep(30)
                        raise AssertionError("an exact local hit must not await the classifier")

                router.laya = SlowLaya()
                started = time.perf_counter()
                result = asyncio.run(router.route("Raspberry Pi cable"))
                elapsed = time.perf_counter() - started
                self.assertEqual(result.routing_decision.strategy, Strategy.EXACT_FTS)
                self.assertLess(elapsed, 1.0)
                self.assertEqual(result.laya_result["status"], "not_awaited")
                router.tier2.close()

    def test_classifier_cannot_veto_a_local_exact_hit(self):
        """The model used to suppress exact local hits whenever it answered
        graph_lightrag - a label from the taxonomy it was never trained on."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()
                router.tier2.index_chunk("one", "Raspberry Pi cable cable cable GPIO pinout")
                router.tier2.commit()

                class Laya:
                    async def predict_routing(self, _):
                        return LayaDecision(
                            strategy="graph_lightrag", domain="raspberry_pi",
                            confidence=0.9999, task="code", task_confidence=0.99,
                            status="success",
                        )

                router.laya = Laya()
                result = asyncio.run(router.route("Raspberry Pi cable"))
                self.assertEqual(result.routing_decision.strategy, Strategy.EXACT_FTS)
                router.tier2.close()

    def test_subject_domain_never_comes_from_the_classifier(self):
        """The classifier's domain question measured 33.3% against 100% for the
        rule, so its answer is recorded for analysis and otherwise ignored."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                router = JevRouter()

                class Laya:
                    async def predict_routing(self, _):
                        return LayaDecision(
                            strategy="complex_llm", domain="raspberry_pi",
                            confidence=0.9995, task="code", task_confidence=0.95,
                            status="success",
                        )

                async def no_embedding(_): return None
                async def no_graph(*_, **__): return {"response": ""}
                router.laya = Laya()
                router.get_embedding = no_embedding
                router.tier3.search = no_graph
                result = asyncio.run(router.route("refactor the go parser"))
                self.assertEqual(result.extracted_metadata.domain, "general")
                self.assertEqual(result.laya_result["domain"], "raspberry_pi")
                router.tier2.close()

    def test_disabled_graph_tier_serves_the_local_answer_without_waiting(self):
        """The graph tier is parked on this hardware: retrieval alone measured 10.9 s
        against a 5 s budget, so calling it is a guaranteed timeout that pays for
        nothing. Disabling it must skip the call, serve the local retrieval, and name
        the tier that actually answered.

        It used to report graph_lightrag with degraded=true, which announced an
        outage that never happened - the call is skipped, elapsed_ms 0.0 - and hid
        which tier produced the context. A parked tier is a configuration, not a
        failure, so it must not be reported as one.

        The embedder here answers and the vector index offers nothing, which is what
        this test was always trying to say: it used to stub the embedder out instead,
        because that was the only way to empty the neighbour list - and that made the
        test pass for a reason it did not claim.  An embedder that does not answer is
        now its own reported state, so the weaker stub would no longer mean "no
        neighbours", it would mean "the semantic search never ran"."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()
                    router.tier2.index_chunk("one", "Torque settings for various fasteners")
                    router.tier2.commit()

                    async def must_not_run(*_, **__):
                        raise AssertionError("the graph tier must not be called when parked")

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="general_fallback", status="success")

                    async def embedding(_):
                        return np.ones(4)

                    def empty_index(*_, **__):
                        # search_vector is synchronous; an async stub here would hand
                        # route() a coroutine and the neighbour list would be one.
                        return []

                    router.tier3.search = must_not_run
                    router.laya = Laya()
                    router.get_embedding = embedding
                    router.tier2.search_vector = empty_index

                    started = time.perf_counter()
                    result = asyncio.run(router.route("torque of the cylinder head bolts"))
                    elapsed = time.perf_counter() - started

                    self.assertEqual(result.routing_decision.strategy, Strategy.FTS_FALLBACK)
                    self.assertEqual(result.routing_decision.confidence_score, 0.5)
                    self.assertFalse(result.degraded)
                    self.assertIsNone(result.fallback_reason)
                    self.assertIn("Torque settings", result.context)
                    self.assertFalse(result.rag_configuration.lightrag_required)
                    self.assertLess(elapsed, 0.5)
                    router.tier2.close()

    def test_nothing_local_at_all_is_not_labelled_as_an_fts_fallback(self):
        """A pool that failed the gate and a corpus that matched nothing are both
        low-confidence, but only one of them has rows behind it.  Labelling the
        empty case fts_fallback would name a tier that produced nothing, which is
        the same substitution this branch already had to be fixed for twice."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="general_fallback", status="success")

                    async def embedding(_):
                        return np.ones(4)

                    def empty_index(*_, **__):
                        return []

                    router.laya = Laya()
                    router.get_embedding = embedding
                    router.tier2.search_vector = empty_index

                    result = asyncio.run(router.route("qwertyuiop zxcvbnm asdfghjkl"))

                    self.assertEqual(result.routing_decision.strategy, Strategy.GENERAL_LLM)
                    self.assertEqual(result.routing_decision.confidence_score, 0.0)
                    self.assertEqual(result.context, "")
                    self.assertFalse(result.degraded)
                    router.tier2.close()

    def test_a_silent_embedder_is_reported_as_a_degradation_not_as_a_weak_pool(self):
        """The difference between "the corpus matched nothing relevant" and "the
        semantic search never happened".

        Both used to arrive as fts_fallback, with the same floor confidence and
        degraded=false, so the only thing telling them apart was latency - 83-149 ms with
        the embedder answering, 2049-2128 ms with it evicted to the CPU - and no consumer
        reads latency. A caller that takes a broken retriever's silence for a verdict on
        its corpus draws the wrong conclusion about the material and retries the wrong
        thing.
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()
                    router.tier2.index_chunk("one", "Torque settings for various fasteners")
                    router.tier2.commit()

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="general_fallback", status="success")

                    async def no_embedding(_):
                        return None

                    router.laya = Laya()
                    router.get_embedding = no_embedding

                    result = asyncio.run(router.route("torque of the cylinder head bolts"))

                    self.assertEqual(result.routing_decision.strategy,
                                     Strategy.EMBEDDING_TIMEOUT)
                    self.assertEqual(result.routing_decision.confidence_score, 0.0)
                    self.assertTrue(result.degraded)
                    self.assertEqual(result.fallback_reason, "embedding_timeout")
                    # The status is the verdict; the local rows are still what gets served.
                    self.assertIn("Torque settings", result.context)
                    router.tier2.close()

    def test_a_missing_vector_index_is_not_reported_as_an_embedder_failure(self):
        """No index to search and an embedder that will not answer are different states.

        The second is an outage to report; the first is a configuration to fix, and
        calling it degraded would send an operator to GPU2 over a file path. So the
        degraded status is gated on the embedder having been asked at all - and this
        asserts the asking, not just the resulting label, because "we did not try" and
        "we tried and failed" are the two states being separated.
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    with patch("core.router.VECTOR_DB_PATH",
                               str(Path(directory) / "absent.npz")):
                        router = JevRouter()
                        router.tier2.index_chunk("one", "Torque settings for various fasteners")
                        router.tier2.commit()

                        class Laya:
                            async def predict_routing(self, _):
                                return LayaDecision(strategy="general_fallback",
                                                    status="success")

                        async def must_not_be_asked(_):
                            raise AssertionError(
                                "the embedder must not be called with no index to search")

                        router.laya = Laya()
                        router.get_embedding = must_not_be_asked

                        result = asyncio.run(router.route("torque of the cylinder head bolts"))

                        self.assertEqual(result.routing_decision.strategy,
                                         Strategy.FTS_FALLBACK)
                        self.assertFalse(result.degraded)
                        self.assertIsNone(result.fallback_reason)
                        router.tier2.close()

    def test_parked_graph_tier_labels_a_sub_threshold_vector_hit_as_low_confidence(self):
        """When the embedding arrives but nothing clears the bar, the vector tier
        offered neighbours and did not decide.  That is a different outcome from
        the decisive vector_fast exit above, and it has to say so: with one label
        for both, a 0.31 neighbour and a 0.67 hit read the same to any caller that
        looks at the status instead of the number.

        The hits are sub-threshold by construction here - clearing the threshold
        would have taken the exit above - so the confidence carries the real
        cosine and claims nothing."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()

                    async def must_not_run(*_, **__):
                        raise AssertionError("the graph tier must not be called when parked")

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="graph_lightrag", status="success")

                    async def embedding(_):
                        return np.ones(4)

                    router.tier3.search = must_not_run
                    router.laya = Laya()
                    router.get_embedding = embedding
                    router.tier2.search_vector = lambda *_, **__: [
                        {"chunk_id": "vec:1", "content": "Sub-threshold neighbour",
                         "source": "/home/dry/memory/projects/x.md", "score": 0.31,
                         "entity_type": "memory", "search_type": "vector"},
                    ]

                    result = asyncio.run(router.route("a query only the parked tier would claim"))

                    self.assertEqual(result.routing_decision.strategy,
                                     Strategy.VECTOR_LOW_CONFIDENCE)
                    self.assertEqual(result.routing_decision.confidence_score, 0.31)
                    self.assertLess(result.routing_decision.confidence_score, SIMILARITY_THRESHOLD)
                    self.assertFalse(result.degraded)
                    self.assertEqual(result.extracted_metadata.intent, "vector_search")
                    router.tier2.close()

    def test_weak_vector_neighbours_are_reported_but_not_served(self):
        """The label describes the tier; the bar still decides the context.

        A neighbour below JEV_SIMILARITY_THRESHOLD is not put in front of the model -
        the calibrated threshold exists to keep it out - but the request must stop
        looking identical to one where the vector tier never ran at all.  So it is
        reported through the status and left out of the context.
        """
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="general_fallback", status="success")

                    async def embedding(_):
                        return np.ones(4)

                    router.laya = Laya()
                    router.get_embedding = embedding
                    router.tier2.search_vector = lambda *_, **__: [
                        {"chunk_id": "vec:1", "content": "WEAK-NEIGHBOUR-MARKER",
                         "source": "/home/dry/memory/projects/x.md", "score": 0.31,
                         "entity_type": "memory", "search_type": "vector"},
                    ]

                    # Nothing in FTS either, so the only material in play is the
                    # weak neighbour - and it must not reach the context.
                    result = asyncio.run(router.route("qwertyuiop zxcvbnm asdfghjkl"))

                    self.assertEqual(result.routing_decision.strategy,
                                     Strategy.VECTOR_LOW_CONFIDENCE)
                    self.assertEqual(result.routing_decision.confidence_score, 0.31)
                    self.assertEqual(result.context, "")
                    self.assertNotIn("WEAK-NEIGHBOUR-MARKER", result.context)
                    router.tier2.close()

    def test_the_vector_search_is_asked_for_unfiltered_neighbours(self):
        """The threshold moved out of search_vector for a reason: applied inside, it
        returned an empty list both when the tier found only weak neighbours and when
        it did not run, so the parked branch could not tell the two apart.  The
        decisive path must keep the same filtering it always had."""
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                with patch("core.router.LIGHTRAG_ENABLED", False):
                    router = JevRouter()
                    calls = []

                    class Laya:
                        async def predict_routing(self, _):
                            return LayaDecision(strategy="general_fallback", status="success")

                    async def embedding(_):
                        return np.ones(4)

                    def record(*_, **kwargs):
                        calls.append(kwargs)
                        return [{"chunk_id": "vec:1", "content": "neighbour",
                                 "source": "/home/dry/memory/projects/x.md", "score": 0.44,
                                 "entity_type": "memory", "search_type": "vector"}]

                    router.laya = Laya()
                    router.get_embedding = embedding
                    router.tier2.search_vector = record

                    result = asyncio.run(router.route("a query the vector tier cannot answer"))

                    self.assertEqual(calls, [{"domain": "general", "apply_threshold": False}])
                    # 0.44 is below the code default here, so it stays out of the
                    # decisive exit and is reported instead.
                    self.assertEqual(result.routing_decision.strategy,
                                     Strategy.VECTOR_LOW_CONFIDENCE)
                    router.tier2.close()

    def test_the_vector_label_follows_the_threshold_on_both_sides(self):
        """The threshold is the entire difference between the two vector labels, so
        the boundary is pinned from both sides rather than from one convenient score.

        The threshold is patched here on purpose.  This suite does not load `.env`
        the way the service does, so unpatched it exercises the code default (0.80)
        while production runs the calibrated value (0.45) - a test written against
        one of them stops describing the other without anyone noticing.
        """
        def route_with(score: float, threshold: float):
            with tempfile.TemporaryDirectory() as directory:
                with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                    with patch("core.router.LIGHTRAG_ENABLED", False):
                        with patch("core.router.SIMILARITY_THRESHOLD", threshold):
                            router = JevRouter()

                            class Laya:
                                async def predict_routing(self, _):
                                    return LayaDecision(strategy="general_fallback", status="success")

                            async def embedding(_):
                                return np.ones(4)

                            router.laya = Laya()
                            router.get_embedding = embedding
                            router.tier2.search_vector = lambda *_, **__: [
                                {"chunk_id": "vec:1", "content": "Neighbour",
                                 "source": "/home/dry/memory/projects/x.md", "score": score,
                                 "entity_type": "memory", "search_type": "vector"},
                            ]
                            result = asyncio.run(router.route("a query the vector tier can answer"))
                            router.tier2.close()
                            return result

        above = route_with(0.46, 0.45)
        self.assertEqual(above.routing_decision.strategy, Strategy.VECTOR_FAST)
        self.assertEqual(above.routing_decision.confidence_score, 0.46)

        below = route_with(0.44, 0.45)
        self.assertEqual(below.routing_decision.strategy, Strategy.VECTOR_LOW_CONFIDENCE)
        self.assertEqual(below.routing_decision.confidence_score, 0.44)
        self.assertFalse(below.degraded)

    def test_vector_index_is_read_once_not_per_request(self):
        """Loading the archive on every request cost ~310 ms of blocked event loop while
        the cosine search itself takes ~13 ms - and being synchronous, that load also
        delayed the classifier's own timeout."""
        with tempfile.TemporaryDirectory() as directory:
            vector_path = Path(directory) / "vectors.npz"
            np.savez(
                vector_path,
                embeddings=np.array([[1.0, 0.0]]),
                chunk_ids=np.array(["one"]), contents=np.array(["content"]), sources=np.array([""]),
                domains=np.array(["general"]),
                model=np.array("mxbai-embed-large"), dimension=np.array(2),
            )
            with patch("core.router.VECTOR_DB_PATH", str(vector_path)):
                with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                    search = Tier2Search()
                    real_load = np.load
                    reads = []

                    def counting_load(*args, **kwargs):
                        reads.append(args[0] if args else kwargs.get("file"))
                        return real_load(*args, **kwargs)

                    with patch("core.router.np.load", counting_load):
                        for _ in range(3):
                            search.search_vector("q", np.array([1.0, 0.0]))

                    self.assertEqual(len(reads), 1, "the index must be read once, not per request")
                    search.close()

    def test_embedding_call_is_bounded_by_its_budget(self):
        """A stalling embedder must cost the budget, not the client default.

        The embedder runs on the node that powers itself off when idle, and it is now the
        fall-through path's only remaining GPU dependency. Under the previous 30 s client
        timeout a sleeping node stalled a request for almost that long (~16 s observed
        while the box was coming up). Here the endpoint accepts the connection and then
        never answers - exactly what a sleeping node presents - and the call must still
        give up inside its budget so the local FTS results get served.
        """

        async def scenario():
            async def handle(reader, writer):
                await asyncio.sleep(30)  # accept, then say nothing

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                with tempfile.TemporaryDirectory() as directory:
                    with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
                        router = JevRouter()
                        with patch("core.router.EMBEDDING_API",
                                   f"http://127.0.0.1:{port}/api/embed"):
                            with patch("core.router.EMBEDDING_TIMEOUT_S", 0.3):
                                started = time.perf_counter()
                                vector = await router.get_embedding("anything")
                                elapsed = time.perf_counter() - started
                        router.tier2.close()
            finally:
                # Server.wait_closed() waits for open handlers on Python 3.12+, and this
                # handler is deliberately stalled; closing is enough, because asyncio.run
                # cancels whatever is left on exit.
                server.close()
            return vector, elapsed

        vector, elapsed = asyncio.run(scenario())
        self.assertIsNone(vector, "a stalled embedder must degrade, not raise")
        self.assertLess(elapsed, 2.0, "the budget must cut the call short")

    def test_stats_exposes_observability_counters(self):
        client = TestClient(server.app)
        response = client.get("/stats")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in (
            "degraded_requests",
            "agent_errors",
            "laya_predictions",
            "laya_accepted",
            "laya_not_awaited",
            "decision_engine_requests",
            "decision_engine_errors",
            "decision_engine_low_confidence",
            "decision_engine_latency_ms",
        ):
            self.assertIn(key, payload)

    def test_route_only_includes_detected_domain(self):
        class Router:
            async def route(self, _):
                return RoutingResult(
                    RoutingDecision(Strategy.DIRECT_ACTION, 1.0, True),
                    ExtractedMetadata(intent="test", domain="raspberry_pi"),
                    RAGConfiguration(),
                    AgentType.GENERAL,
                    query="Raspberry Pi cable",
                )

        with patch.object(server, "router", Router()):
            client = TestClient(server.app)
            response = client.get("/route-only", params={"query": "Raspberry Pi cable"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["extracted_metadata"]["domain"], "raspberry_pi")

    def test_decision_engine_falls_back_when_unavailable(self):
        client = DecisionEngineClient(base_url="http://127.0.0.1:1", timeout_s=0.01)
        result = asyncio.run(client.decide("route this", ["graph_lightrag", "general_llm"]))
        self.assertEqual(result.choice, "graph_lightrag")
        self.assertEqual(result.engine, "fallback")
        self.assertNotEqual(result.status, "success")
        self.assertTrue(result.low_confidence)

    def test_decision_test_endpoint_records_fallback_metrics(self):
        class Engine:
            async def decide(self, **_):
                return DecisionEngineResult(
                    choice="graph_lightrag",
                    confidence=0.0,
                    latency_ms=1.5,
                    status="timeout",
                    error="decision timeout",
                )

        before = server._stats["decision_engine_requests"]
        with patch.object(server, "decision_engine", Engine()):
            client = TestClient(server.app)
            response = client.post(
                "/decision-test",
                json={
                    "query": "choose path",
                    "candidates": ["graph_lightrag", "general_llm"],
                    "schema": "routing_v1",
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choice"], "graph_lightrag")
        self.assertEqual(payload["status"], "timeout")
        self.assertEqual(payload["fallback_reason"], "decision timeout")
        self.assertEqual(server._stats["decision_engine_requests"], before + 1)


def _laya_response(strategy="direct_cmd", domain="general", confidence=0.91, status="success"):
    return httpx.Response(
        200,
        json={
            "strategy": strategy,
            "domain": domain,
            "confidence": confidence,
            "status": status,
            "latency_ms": 28.0,
            "checkpoint": "multilingual",
        },
    )


class ShadowProbeTests(unittest.TestCase):
    def test_disabled_target_schedules_nothing(self):
        probe = ShadowProbe(target="off", url="http://127.0.0.1:1")

        async def scenario():
            return probe.submit("hello")

        self.assertFalse(asyncio.run(scenario()))
        self.assertFalse(probe.enabled)
        self.assertEqual(probe.stats()["requests"], 0)

    def test_probe_records_success_without_touching_the_caller(self):
        seen: list[dict] = []

        def handler(request):
            self.assertEqual(json.loads(request.content)["query"], "status check")
            return _laya_response()

        probe = ShadowProbe(
            target="laya",
            url="http://shadow.test",
            timeout_s=1.0,
            transport=httpx.MockTransport(handler),
            on_result=seen.append,
        )

        async def scenario():
            self.assertTrue(probe.submit("status check"))
            await probe.drain()

        asyncio.run(scenario())
        stats = probe.stats()
        self.assertEqual(stats["success"], 1)
        self.assertEqual(stats["choices"], {"direct_cmd": 1})
        self.assertEqual(stats["inflight"], 0)
        self.assertEqual(seen[0]["confidence"], 0.91)
        self.assertEqual(seen[0]["checkpoint"], "multilingual")
        self.assertFalse(seen[0]["low_confidence"])

    def test_quiet_engine_answer_is_not_counted_as_a_success(self):
        # A sleeping GPU2 answers with a fallback label. Counting that as a
        # successful observation would flatter every shadow benchmark.
        def handler(request):
            return _laya_response(strategy="general_fallback", confidence=0.0, status="unavailable")

        probe = ShadowProbe(target="laya", url="http://shadow.test", transport=httpx.MockTransport(handler))

        async def scenario():
            probe.submit("anything")
            await probe.drain()

        asyncio.run(scenario())
        stats = probe.stats()
        self.assertEqual(stats["success"], 0)
        self.assertEqual(stats["unavailable"], 1)
        self.assertEqual(stats["choices"], {})

    def test_saturated_probe_is_skipped_not_queued(self):
        release = asyncio.Event()

        async def handler(request):
            await release.wait()
            return _laya_response(strategy="database_search", confidence=0.5)

        probe = ShadowProbe(
            target="laya",
            url="http://shadow.test",
            max_inflight=1,
            transport=httpx.MockTransport(handler),
        )

        async def scenario():
            self.assertTrue(probe.submit("first"))
            await asyncio.sleep(0)  # let the first probe reach the transport
            self.assertFalse(probe.submit("second"))
            release.set()
            await probe.drain()

        asyncio.run(scenario())
        stats = probe.stats()
        self.assertEqual(stats["skipped_busy"], 1)
        self.assertEqual(stats["requests"], 1)

    def test_http_failure_is_counted_and_never_raised(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        probe = ShadowProbe(target="laya", url="http://shadow.test", transport=httpx.MockTransport(handler))

        async def scenario():
            probe.submit("anything")
            await probe.drain()

        asyncio.run(scenario())
        self.assertEqual(probe.stats()["unavailable"], 1)

    def test_query_schedules_a_shadow_probe(self):
        submitted: list[str] = []

        class Probe:
            enabled = True

            def submit(self, query, context=""):
                submitted.append(query)
                return True

        class Router:
            async def route(self, _):
                return RoutingResult(
                    RoutingDecision(Strategy.DIRECT_ACTION, 1.0, True),
                    ExtractedMetadata(intent="test"),
                    RAGConfiguration(),
                    AgentType.GENERAL,
                    query="Raspberry Pi cable",
                )

        with patch.object(server, "router", Router()), patch.object(server, "shadow_probe", Probe()):
            client = TestClient(server.app)
            response = client.post("/query", json={"query": "Raspberry Pi cable", "execute": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(submitted, ["Raspberry Pi cable"])

    def test_stats_expose_shadow_counters(self):
        payload = TestClient(server.app).get("/stats").json()
        for key in (
            "shadow_requests",
            "shadow_errors",
            "shadow_timeouts",
            "shadow_low_confidence",
            "shadow_choices",
            "shadow",
        ):
            self.assertIn(key, payload)
        self.assertIn("skipped_busy", payload["shadow"])

    def test_metrics_expose_shadow_families(self):
        body = TestClient(server.app).get("/metrics").text
        self.assertIn("jev_shadow_requests_total", body)
        self.assertIn("jev_shadow_low_confidence_total", body)
        self.assertIn("jev_shadow_skipped_total", body)

    def test_health_reports_shadow_mode(self):
        payload = TestClient(server.app).get("/health").json()
        self.assertIn("shadow", payload)
        self.assertIn("mode", payload["shadow"])


class LayaClientTests(unittest.TestCase):
    def test_total_deadline_is_enforced_not_just_the_read_timeout(self):
        """httpx read timeouts apply per socket read, so a trickling response can
        outlive the configured budget. Since this call is inline in routing, the
        budget has to bound the whole call."""
        async def slow(request):
            await asyncio.sleep(0.6)
            return httpx.Response(
                200,
                json={"strategy": "direct_cmd", "domain": "general", "confidence": 0.9, "status": "success"},
            )

        client = LayaTier1Client(
            base_url="http://shadow.test",
            timeout_s=0.05,
            transport=httpx.MockTransport(slow),
        )
        started = time.perf_counter()
        decision = asyncio.run(client.predict_routing("status check"))
        elapsed = time.perf_counter() - started
        self.assertEqual(decision.status, "timeout")
        self.assertLess(elapsed, 0.5, "the client must give up long before the handler finishes")

    def test_success_path_returns_the_verdict(self):
        def handler(request):
            self.assertEqual(json.loads(request.content)["query"], "rewrite in Go")
            return httpx.Response(
                200,
                json={"strategy": "complex_llm", "domain": "general", "confidence": 0.93, "status": "success"},
            )

        client = LayaTier1Client(base_url="http://shadow.test", transport=httpx.MockTransport(handler))
        decision = asyncio.run(client.predict_routing("rewrite in Go"))
        self.assertEqual(decision.status, "success")
        self.assertEqual(decision.strategy, "complex_llm")
        self.assertAlmostEqual(decision.confidence, 0.93)

    def test_trust_follows_the_preset_signal_not_the_invented_confidence(self):
        """0.9995 on a hand-written question is not evidence of anything: this
        checkpoint was never trained on that taxonomy. Raising the threshold
        cannot filter it, because the wrong answer is the confident one."""
        invented = LayaDecision(confidence=0.9995, task_confidence=0.0, status="success")
        self.assertFalse(invented.trusted)
        preset = LayaDecision(confidence=0.10, task="code", task_confidence=0.95, status="success")
        self.assertTrue(preset.trusted)
        self.assertFalse(not_awaited().trusted)

    def test_preset_signals_are_parsed_from_the_response(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "strategy": "complex_llm", "domain": "raspberry_pi", "confidence": 0.31,
                    "status": "success", "task": "code",
                    "task_confidence": 0.9255, "difficulty": 3.0,
                    "checkpoint": "multilingual", "routing_reason": "non-Latin script (cyrillic)",
                },
            )

        client = LayaTier1Client(base_url="http://shadow.test", transport=httpx.MockTransport(handler))
        decision = asyncio.run(client.predict_routing("rewrite the router in Go"))
        self.assertEqual(decision.task, "code")
        self.assertAlmostEqual(decision.task_confidence, 0.9255)
        self.assertAlmostEqual(decision.difficulty, 3.0)
        self.assertTrue(decision.trusted)
        self.assertEqual(decision.to_dict()["task"], "code")
        # The checkpoint is what makes the threshold measurable: the two checkpoints
        # need not report preset confidence on the same scale, so it must reach the
        # decision log rather than being pooled away.
        self.assertEqual(decision.checkpoint, "multilingual")
        self.assertEqual(decision.routing_reason, "non-Latin script (cyrillic)")
        self.assertEqual(decision.to_dict()["checkpoint"], "multilingual")


class EnvLoaderTests(unittest.TestCase):
    def test_dotenv_values_load_but_do_not_override_the_environment(self):
        """`.env` was documented but never read, so editing it did nothing at all.

        Real environment variables must still win: an operator exporting a value for one
        run should not have it silently overridden by a checked-in file.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# a comment\n"
                "\n"
                "JEV_TEST_PLAIN=value\n"
                'JEV_TEST_QUOTED="http://127.0.0.1:8030"\n'
                "export JEV_TEST_EXPORTED=exported\n"
                "JEV_TEST_EXISTING=from_file\n"
                "NOT_A_ASSIGNMENT\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"JEV_TEST_EXISTING": "from_environment"}, clear=False):
                applied = load_env(path)
                self.assertEqual(os.environ["JEV_TEST_PLAIN"], "value")
                self.assertEqual(os.environ["JEV_TEST_QUOTED"], "http://127.0.0.1:8030")
                self.assertEqual(os.environ["JEV_TEST_EXPORTED"], "exported")
                self.assertEqual(os.environ["JEV_TEST_EXISTING"], "from_environment")
                self.assertIn("JEV_TEST_PLAIN", applied)
                self.assertNotIn("JEV_TEST_EXISTING", applied)
            for key in ("JEV_TEST_PLAIN", "JEV_TEST_QUOTED", "JEV_TEST_EXPORTED"):
                os.environ.pop(key, None)
