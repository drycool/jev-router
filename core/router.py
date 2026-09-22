"""
Jev Multi-Tier Router & Agentic Orchestrator
=============================================
Level 0: Ultra-Low Latency Request Routing
4-Tier Pipeline: Fast Route → FTS/Vector → LightRAG → Agent
"""
import re
import time
import json
import sqlite3
import os
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import numpy as np
from core.laya_client import LAYA_CONFIDENCE_THRESHOLD, LayaTier1Client


# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_THRESHOLD = float(os.getenv("JEV_SIMILARITY_THRESHOLD", "0.80"))
FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))
VECTOR_DB_PATH = os.getenv("JEV_VECTOR_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_vectors.npz"))
LIGHTRAG_API = os.getenv("JEV_LIGHTRAG_API", "http://localhost:8020")
LIGHTRAG_CONNECT_TIMEOUT_S = float(os.getenv("JEV_LIGHTRAG_CONNECT_TIMEOUT_S", "1"))
LIGHTRAG_READ_TIMEOUT_S = float(os.getenv("JEV_LIGHTRAG_READ_TIMEOUT_S", "5"))
EMBEDDING_API = os.getenv("JEV_EMBEDDING_API", "http://192.168.11.87:11434/api/embed")
EMBEDDING_MODEL = os.getenv("JEV_EMBEDDING_MODEL", "mxbai-embed-large")


class Strategy(str, Enum):
    DIRECT_ACTION = "direct_action"
    EXACT_FTS = "exact_fts"
    VECTOR_FAST = "vector_fast"
    GRAPH_LIGHTRAG = "graph_lightrag"
    GENERAL_LLM = "general_llm"


class AgentType(str, Enum):
    DB = "db_agent"
    TROUBLESHOOTER = "troubleshooter_agent"
    CODE = "code_agent"
    GENERAL = "general_agent"


class LightRAGMode(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    SKIP = "skip"


# ── Data Models ────────────────────────────────────────────────────────
@dataclass
class RoutingDecision:
    strategy: Strategy
    confidence_score: float
    fast_path_exit: bool


@dataclass
class ExtractedMetadata:
    intent: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    domain: str = "general"
    action_payload: dict = field(default_factory=dict)


@dataclass
class RAGConfiguration:
    lightrag_required: bool = False
    lightrag_mode: str = "skip"
    similarity_threshold: float = SIMILARITY_THRESHOLD


@dataclass
class RoutingResult:
    routing_decision: RoutingDecision
    extracted_metadata: ExtractedMetadata
    rag_configuration: RAGConfiguration
    target_agent: AgentType
    query: str = ""
    context: str = ""
    elapsed_ms: float = 0.0
    degraded: bool = False
    fallback_reason: Optional[str] = None
    laya_result: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "routing_decision": asdict(self.routing_decision),
                "extracted_metadata": asdict(self.extracted_metadata),
                "rag_configuration": asdict(self.rag_configuration),
                "target_agent": self.target_agent.value,
                "query": self.query,
                "context": self.context[:500] if self.context else "",
                "elapsed_ms": round(self.elapsed_ms, 2),
                "degraded": self.degraded,
                "fallback_reason": self.fallback_reason,
                "laya_result": self.laya_result,
            },
            ensure_ascii=False,
            indent=2,
        )


# ── Tier 1: Fast Single-Pass Router ──────────────────────────────────
# Sub-20ms, non-autoregressive, pattern-based

DIRECT_COMMANDS = {
    # Code execution patterns
    r"^run\s+": ("execute_code", AgentType.CODE),
    r"^exec\s+": ("execute_code", AgentType.CODE),
    r"^python\s+": ("execute_code", AgentType.CODE),
    r"^bash\s+": ("execute_code", AgentType.CODE),
    r"^sudo\s+": ("execute_code", AgentType.CODE),
    r"^pip\s+install": ("install_package", AgentType.CODE),
    r"^npm\s+": ("npm_command", AgentType.CODE),
    r"^docker\s+": ("docker_command", AgentType.CODE),
    # DB commands
    r"^select\s+": ("db_query", AgentType.DB),
    r"^insert\s+": ("db_insert", AgentType.DB),
    r"^update\s+": ("db_update", AgentType.DB),
    r"^delete\s+": ("db_delete", AgentType.DB),
    r"^show\s+tables": ("db_list_tables", AgentType.DB),
    r"^describe\s+": ("db_describe", AgentType.DB),
    # Troubleshooting
    r"^fix\s+": ("fix_issue", AgentType.TROUBLESHOOTER),
    r"^debug\s+": ("debug_issue", AgentType.TROUBLESHOOTER),
    r"^why\s+does": ("diagnose", AgentType.TROUBLESHOOTER),
    r"^error\s+": ("analyze_error", AgentType.TROUBLESHOOTER),
    r"^traceback": ("analyze_error", AgentType.TROUBLESHOOTER),
}

# Binary decisions / confirmations
BINARY_PATTERNS = [
    r"^(yes|no|да|нет|ok|cancel|quit|exit|q)$",
    r"^(y|n)$",
    r"^confirm",
    r"^skip",
]

# Code detection
CODE_INDICATORS = [
    r"```",
    r"def\s+\w+\(",
    r"class\s+\w+",
    r"import\s+\w+",
    r"from\s+\w+\s+import",
    r"if\s+__name__",
    r"for\s+\w+\s+in\s+",
    r"while\s+",
    r"return\s+",
    r"print\(",
]


def tier1_fast_route(query: str) -> Optional[RoutingResult]:
    """
    Tier 1: Sub-20ms single-pass routing.
    Returns RoutingResult for early exit, None to escalate.
    """
    t0 = time.perf_counter()
    query_lower = query.strip().lower()

    # Binary check
    for pattern in BINARY_PATTERNS:
        if re.match(pattern, query_lower):
            return RoutingResult(
                routing_decision=RoutingDecision(
                    strategy=Strategy.DIRECT_ACTION,
                    confidence_score=1.0,
                    fast_path_exit=True,
                ),
                extracted_metadata=ExtractedMetadata(
                    intent="binary_decision",
                    keywords=[query_lower],
                ),
                rag_configuration=RAGConfiguration(lightrag_mode="skip"),
                target_agent=AgentType.GENERAL,
                query=query,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    # Direct command check
    for pattern, (intent, agent) in DIRECT_COMMANDS.items():
        if re.match(pattern, query_lower):
            return RoutingResult(
                routing_decision=RoutingDecision(
                    strategy=Strategy.DIRECT_ACTION,
                    confidence_score=0.95,
                    fast_path_exit=True,
                ),
                extracted_metadata=ExtractedMetadata(
                    intent=intent,
                    keywords=query_lower.split()[:5],
                    action_payload={"command": query_lower.split()[0]},
                ),
                rag_configuration=RAGConfiguration(lightrag_mode="skip"),
                target_agent=agent,
                query=query,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    # Code detection
    code_score = sum(1 for p in CODE_INDICATORS if re.search(p, query))
    if code_score >= 2:
        return RoutingResult(
            routing_decision=RoutingDecision(
                strategy=Strategy.DIRECT_ACTION,
                confidence_score=0.85,
                fast_path_exit=True,
            ),
            extracted_metadata=ExtractedMetadata(
                intent="code_execution",
                keywords=_extract_keywords(query),
            ),
            rag_configuration=RAGConfiguration(lightrag_mode="skip"),
            target_agent=AgentType.CODE,
            query=query,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    # No fast path — escalate to Tier 2
    return None


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text."""
    words = re.findall(r"\b[a-zA-Zа-яА-ЯёЁ]{3,}\b", text)
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "what", "how", "why", "can", "you", "does", "как", "что", "это",
        "для", "когда", "где", "какой", "какая", "какие", "почему",
    }
    return [w for w in words if w.lower() not in stopwords][:10]


def detect_domain(text: str) -> str:
    """Fast, conservative domain classifier used only as a vector pre-filter."""
    value = text.lower()
    raspberry_markers = ("raspberry", "одноплат", "gpio", "micro-hdmi", "nvme", "pi 5", "pi 4")
    automotive_markers = ("автомоб", "двигател", "ecu", "k-line", "зажиган", "esperо", "espero", "диагност")
    if any(marker in value for marker in raspberry_markers):
        return "raspberry_pi"
    if any(marker in value for marker in automotive_markers):
        return "automotive"
    return "general"


def _is_fts_exact(query: str, result: dict) -> bool:
    """Require multiple meaningful query terms in a candidate before early exit."""
    keywords = _extract_keywords(query)
    if not keywords:
        return False
    content_terms = set(re.findall(r"\b[\wа-яА-ЯёЁ]+\b", result["content"].lower()))
    matches = sum(keyword.lower() in content_terms for keyword in keywords)
    return matches >= min(2, len(keywords))


# ── Tier 2: FTS5 + Vector Search ─────────────────────────────────────
class Tier2Search:
    """Fast semantic search: SQLite FTS5 (BM25) + Vector cosine similarity."""

    def __init__(self):
        self._init_fts5()

    def _init_fts5(self):
        """Initialize FTS5 database."""
        Path(FTS5_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(FTS5_DB_PATH, check_same_thread=False)
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                chunk_id UNINDEXED,
                content,
                source,
                entity_type,
                tokenize='unicode61'
            )
        """)
        self.conn.commit()

    def clear(self) -> None:
        """Remove the derived index before a complete rebuild."""
        self.conn.execute("DELETE FROM chunks")

    def index_chunk(self, chunk_id: str, content: str, source: str = "", entity_type: str = ""):
        """Index a text chunk into FTS5."""
        self.conn.execute(
            "INSERT INTO chunks (chunk_id, content, source, entity_type) VALUES (?, ?, ?, ?)",
            (chunk_id, content, source, entity_type),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def search_fts5(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 full-text search. Returns results in ~2-5ms."""
        t0 = time.perf_counter()
        # Escape FTS5 special characters
        safe_query = re.sub(r'[^\w\s]', ' ', query)
        terms = safe_query.split()
        if not terms:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in terms)

        try:
            rows = self.conn.execute(
                """SELECT chunk_id, content, source, entity_type,
                          rank AS score
                   FROM chunks
                   WHERE chunks MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "content": row[1],
                "source": row[2],
                "entity_type": row[3],
                # bm25() returns a negative rank; lower is more relevant.
                "score": abs(float(row[4])),
                "search_type": "fts5",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            })
        return results

    def search_vector(self, query: str, query_embedding: np.ndarray, limit: int = 10, domain: str = "general") -> list[dict]:
        """Vector cosine similarity search. ~10-30ms."""
        t0 = time.perf_counter()
        # Load vector database
        if not Path(VECTOR_DB_PATH).exists():
            return []

        try:
            with np.load(VECTOR_DB_PATH, allow_pickle=False) as db:
                embeddings = db["embeddings"]
                chunk_ids = db["chunk_ids"]
                contents = db["contents"]
                sources = db["sources"]
                domains = db["domains"]
                model = str(db["model"].item())
                dimension = int(db["dimension"].item())
        except (OSError, KeyError, ValueError):
            return []

        if len(embeddings) == 0:
            return []
        if (
            embeddings.ndim != 2
            or model != EMBEDDING_MODEL
            or dimension != embeddings.shape[1]
            or query_embedding.ndim != 1
            or embeddings.shape[1] != query_embedding.shape[0]
        ):
            # A stale index or one built with another model must never be
            # searched: incomparable vectors would create false confidence.
            return []

        # Cosine similarity
        if domain != "general":
            mask = domains.astype(str) == domain
            if not np.any(mask):
                return []
            embeddings = embeddings[mask]
            chunk_ids, contents, sources = chunk_ids[mask], contents[mask], sources[mask]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        q_norm = query_embedding / (np.linalg.norm(query_embedding) or 1)

        scores = normalized @ q_norm
        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < SIMILARITY_THRESHOLD:
                continue
            results.append({
                "chunk_id": str(chunk_ids[idx]),
                "content": str(contents[idx]),
                "source": str(sources[idx]),
                "score": score,
                "search_type": "vector",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            })
        return results

    def search(self, query: str, query_embedding: Optional[np.ndarray] = None) -> list[dict]:
        """Combined Tier 2 search: FTS5 first, then vector if needed."""
        # FTS5 first (fastest)
        fts_results = self.search_fts5(query)

        # Vector search if embedding provided
        vector_results = []
        if query_embedding is not None:
            vector_results = self.search_vector(query, query_embedding)

        # Merge and deduplicate
        all_results = {}
        for r in fts_results + vector_results:
            cid = r["chunk_id"]
            if cid not in all_results or r["score"] > all_results[cid]["score"]:
                all_results[cid] = r

        # FTS5 BM25 and cosine similarity are different scales.  Preserve the
        # FTS result order, then append vector hits rather than comparing them.
        return fts_results + [r for r in vector_results if r["chunk_id"] not in {x["chunk_id"] for x in fts_results}]


# ── Tier 3: LightRAG Graph Search ────────────────────────────────────
class Tier3GraphSearch:
    """LightRAG integration for graph-based retrieval."""

    def __init__(self, api_url: str = LIGHTRAG_API):
        self.api_url = api_url

    async def search(self, query: str, mode: str = "local", top_k: int = 40) -> dict:
        """Query LightRAG API."""
        import httpx

        t0 = time.perf_counter()
        try:
            timeout = httpx.Timeout(
                connect=LIGHTRAG_CONNECT_TIMEOUT_S,
                read=LIGHTRAG_READ_TIMEOUT_S,
                write=LIGHTRAG_READ_TIMEOUT_S,
                pool=LIGHTRAG_CONNECT_TIMEOUT_S,
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self.api_url}/query",
                    json={"query": query, "mode": mode, "top_k": top_k},
                )
                resp.raise_for_status()
                result = resp.json()
                result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
                return result
        except httpx.TimeoutException as e:
            return {
                "response": "", "error": str(e), "error_type": "timeout",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }
        except Exception as e:
            return {
                "response": "",
                "error": str(e),
                "error_type": "error",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            }


# ── Tier 4: Agent Executor ───────────────────────────────────────────
class AgentExecutor:
    """Dispatches to target agent with context."""

    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_type: AgentType, agent):
        self.agents[agent_type] = agent

    async def execute(self, result: RoutingResult):
        """Execute the target agent with gathered context."""
        agent = self.agents.get(result.target_agent)
        if agent is None:
            return f"No agent registered for {result.target_agent}"

        return await agent.execute(
            query=result.query,
            context=result.context,
            metadata=result.extracted_metadata,
        )


# ── Main Router ───────────────────────────────────────────────────────
class JevRouter:
    """
    Jev Multi-Tier Router & Agentic Orchestrator.
    Level 0: Ultra-Low Latency Request Routing.
    """

    def __init__(self):
        self.tier2 = Tier2Search()
        self.tier3 = Tier3GraphSearch()
        self.tier4 = AgentExecutor()
        self.laya = LayaTier1Client()
        self._request_count = 0
        self._total_ms = 0.0

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text. Returns None if embedding service unavailable."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    EMBEDDING_API,
                    json={"model": EMBEDDING_MODEL, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                return np.array(embeddings[0])
        except Exception:
            pass
        return None

    async def route(self, query: str) -> RoutingResult:
        """
        Main routing pipeline.
        Returns RoutingResult with strategy, context, and target agent.
        """
        t0 = time.perf_counter()
        self._request_count += 1

        # ── Tier 1: Fast single-pass routing ──
        tier1_result = tier1_fast_route(query)
        if tier1_result is not None:
            self._total_ms += (time.perf_counter() - t0) * 1000
            return tier1_result

        # Remote ML is advisory: exact rules retain command-execution
        # authority. A timeout/unavailable GPU2 merely falls through to the
        # deterministic retrieval pipeline.
        laya = await self.laya.predict_routing(query)
        laya_data = laya.to_dict()
        laya_accepted = laya.status == "success" and laya.confidence >= LAYA_CONFIDENCE_THRESHOLD
        domain = laya.domain if laya_accepted and laya.domain in {"raspberry_pi", "automotive", "general"} else detect_domain(query)
        laya_strategy = laya.strategy if laya_accepted else "general_fallback"
        # ML may recommend retrieval, but it never gains authority to execute
        # commands. Direct execution remains exclusively rule-gated above.
        if laya_strategy == "direct_cmd":
            laya_data["execution_override"] = "direct_cmd_requires_regex_match"
            laya_strategy = "general_fallback"
        target_agent = AgentType.DB if laya_strategy == "database_search" else AgentType.GENERAL

        # ── Tier 2: FTS5 + Vector search ──
        # FTS is local and is deliberately attempted before embedding.  Calling
        # the embedding server first made an exact local hit wait for the
        # network on every request, defeating the early-exit design.
        fts_results = self.tier2.search_fts5(query)
        fts_exact = [] if laya_strategy == "graph_lightrag" else [r for r in fts_results if _is_fts_exact(query, r)]
        if fts_exact:
            context = "\n\n".join(r["content"] for r in fts_exact[:3])
            return RoutingResult(
                routing_decision=RoutingDecision(
                    strategy=Strategy.EXACT_FTS,
                    confidence_score=0.95,
                    fast_path_exit=False,
                ),
                extracted_metadata=ExtractedMetadata(
                    intent="exact_search",
                    keywords=_extract_keywords(query),
                ),
                rag_configuration=RAGConfiguration(
                    lightrag_required=False,
                    lightrag_mode="skip",
                ),
                target_agent=target_agent,
                query=query,
                context=context,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                laya_result=laya_data,
            )

        # Do not make a remote embedding call when there is no vector index to
        # search.  This installation currently uses FTS5 + LightRAG only.
        query_embedding = await self.get_embedding(query) if Path(VECTOR_DB_PATH).exists() else None
        vector_results = self.tier2.search_vector(query, query_embedding, domain=domain) if query_embedding is not None else []
        tier2_results = fts_results + [
            item for item in vector_results
            if item["chunk_id"] not in {fts["chunk_id"] for fts in fts_results}
        ]

        # Check vector similarity threshold
        best_score = vector_results[0]["score"] if vector_results else 0.0
        if best_score >= SIMILARITY_THRESHOLD and laya_strategy != "graph_lightrag":
            context = "\n\n".join(r["content"] for r in vector_results[:5])
            return RoutingResult(
                routing_decision=RoutingDecision(
                    strategy=Strategy.VECTOR_FAST,
                    confidence_score=best_score,
                    fast_path_exit=False,
                ),
                extracted_metadata=ExtractedMetadata(
                    intent="vector_search",
                    keywords=_extract_keywords(query),
                    domain=domain,
                ),
                rag_configuration=RAGConfiguration(
                    lightrag_required=False,
                    lightrag_mode="skip",
                ),
                target_agent=target_agent,
                query=query,
                context=context,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                laya_result=laya_data,
            )

        # ── Tier 3: LightRAG Graph Search ──
        # Determine mode based on query characteristics
        lightrag_mode = self._determine_lightrag_mode(query)

        tier3_result = await self.tier3.search(query, mode=lightrag_mode)

        context = tier3_result.get("response", "")
        fallback_reason = None
        if not context and tier2_results:
            # Degraded-mode fallback: serve retained local retrieval rather
            # than failing the HTTP request when the graph is unavailable.
            context = "\n\n".join(r["content"] for r in tier2_results[:5])
            fallback_reason = f"lightrag_{tier3_result.get('error_type', 'empty_response')}"
        elif not context:
            fallback_reason = f"lightrag_{tier3_result.get('error_type', 'empty_response')}"

        return RoutingResult(
            routing_decision=RoutingDecision(
                strategy=Strategy.GRAPH_LIGHTRAG,
                confidence_score=0.7,
                fast_path_exit=False,
            ),
            extracted_metadata=ExtractedMetadata(
                intent="graph_search",
                keywords=_extract_keywords(query),
                entities=self._extract_entities(query),
                domain=domain,
            ),
            rag_configuration=RAGConfiguration(
                lightrag_required=True,
                lightrag_mode=lightrag_mode,
            ),
            target_agent=target_agent,
            query=query,
            context=context,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            degraded=fallback_reason is not None,
            fallback_reason=fallback_reason,
            laya_result=laya_data,
        )

    def _determine_lightrag_mode(self, query: str) -> str:
        """Determine LightRAG mode based on query characteristics."""
        query_lower = query.lower()

        # Macro/summary queries → global
        macro_keywords = ["summary", "overview", "обзор", "summarize", "все", "all", "total"]
        if any(kw in query_lower for kw in macro_keywords):
            return "global"

        # Multi-hop / relationship queries → hybrid
        multi_hop_keywords = ["relationship", "связь", "how.*connected", "between", "middle", "chain"]
        if any(re.search(kw, query_lower) for kw in multi_hop_keywords):
            return "hybrid"

        # Default: local (focused)
        return "local"

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential entity names from query."""
        # Simple NER: capitalized words, technical terms
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        # Also match technical terms
        tech = re.findall(r"\b(?:Raspberry\s+Pi|LED|GPIO|SSD|HDMI|USB)\b", query, re.IGNORECASE)
        return list(set(entities + tech))[:5]

    def stats(self) -> dict:
        """Return routing statistics."""
        return {
            "total_requests": self._request_count,
            "avg_latency_ms": round(self._total_ms / max(self._request_count, 1), 2),
        }
