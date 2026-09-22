import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api import server
from core.decision_engine import DecisionEngineClient, DecisionEngineResult
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
                        from core.laya_client import LayaDecision
                        return LayaDecision(strategy="direct_cmd", domain="general", confidence=0.99, status="success")
                async def no_embedding(_): return None
                async def no_graph(*_, **__): return {"response": ""}
                router.laya = Laya()
                router.get_embedding = no_embedding
                router.tier3.search = no_graph
                result = asyncio.run(router.route("please erase all documents"))
                self.assertEqual(result.target_agent, AgentType.GENERAL)
                self.assertEqual(result.laya_result["execution_override"], "direct_cmd_requires_regex_match")
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
