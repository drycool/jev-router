import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

from api import server
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
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
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
