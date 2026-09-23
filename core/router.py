"""
Jev Multi-Tier Router & Agentic Orchestrator
=============================================
Level 0: Ultra-Low Latency Request Routing
4-Tier Pipeline: Fast Route → FTS/Vector → LightRAG → Agent
"""
import asyncio
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
from core.laya_client import LayaDecision, LayaTier1Client, not_awaited


# ── Configuration ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_THRESHOLD = float(os.getenv("JEV_SIMILARITY_THRESHOLD", "0.80"))
FTS5_DB_PATH = os.getenv("JEV_FTS5_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_fts5.db"))
VECTOR_DB_PATH = os.getenv("JEV_VECTOR_DB_PATH", str(PROJECT_ROOT / "storage" / "jev_vectors.npz"))
LIGHTRAG_API = os.getenv("JEV_LIGHTRAG_API", "http://localhost:8020")
LIGHTRAG_CONNECT_TIMEOUT_S = float(os.getenv("JEV_LIGHTRAG_CONNECT_TIMEOUT_S", "1"))
LIGHTRAG_READ_TIMEOUT_S = float(os.getenv("JEV_LIGHTRAG_READ_TIMEOUT_S", "5"))
# On this hardware the graph tier cannot answer inside its own budget: even
# retrieval alone measured 10.9 s against a 5 s read timeout, so calling it is a
# guaranteed timeout and the request pays for nothing. The tier is parked until the
# GPU budget is resolved (see README, "Deferred"). Disabling it skips the call and
# serves exactly the context the timeout path would have served, without the wait.
# Default preserves the existing behaviour.
LIGHTRAG_ENABLED = os.getenv("JEV_LIGHTRAG_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
EMBEDDING_API = os.getenv("JEV_EMBEDDING_API", "http://192.168.11.87:11434/api/embed")
EMBEDDING_MODEL = os.getenv("JEV_EMBEDDING_MODEL", "mxbai-embed-large")
# Total budget for the embedding call, enforced with asyncio.timeout (same reason as the
# classifier: an httpx timeout is per socket read, not a deadline). The embedder lives on
# the node that powers itself off when idle, and this is the fall-through path's only
# remaining GPU dependency - a 30 s client timeout let a sleeping node stall a request for
# almost that long (observed ~16 s while the box was coming up). Healthy cost is ~32 ms,
# so this is ~60x headroom. On expiry the vector layer is skipped and the local FTS
# results are served, exactly as when the embedder is unreachable.
EMBEDDING_TIMEOUT_S = float(os.getenv("JEV_VECTOR_TIMEOUT_S", "2.0"))

# How much retrieved text the agent is handed, in characters.  A budget rather than a
# count, because a count cannot know how long a chunk is: the old `[:3]` delivered
# 1691-3078 characters across four real queries and would deliver 300 on a corpus that
# happened to produce three short chunks.  What the router owes the agent is a
# predictable prefill bill, not a fixed number of rows.
#
# 8000 is where the budget stops being the binding constraint.  Measured on four real
# queries (scripts/measure_context_budget.py), unique chunks delivered out of the pool:
# 4000 chars -> 4/7, 6/9, 7/10, 4/8; 6000 -> 5/7, 9/9, 9/10, 8/8; 8000 -> 7/7, 9/9, 10/10,
# 8/8; 10000 -> byte-identical to 8000, because by then the retrieval limit of 10 results
# is what runs out.  Above this value the setting does nothing at all.
#
# The cost is smaller than it looks, and it is not the deciding factor.  An A/B of the
# same query at 2360 and 7569 characters in (two runs each, alternating) came back at
# 5.8/7.3 s and 7.8/9.6 s - a spread inside one budget as wide as the gap between them,
# plus the confound that longer context produced longer answers and decode dominates.
# What the sample does show: answers were 1325-1640 characters at 2360 in and 1838-2449
# at 7569, i.e. more of the procedure survives.  Prefill is cheap; the budget is set for
# completeness, not for microseconds.
MAX_CONTEXT_CHARS = int(os.getenv("JEV_MAX_CONTEXT_CHARS", "8000"))
CONTEXT_SEPARATOR = "\n\n"


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
    # What assemble_context did with the retrieval pool: how many chunks it considered,
    # how many it used, how many it dropped as duplicates. Without this the dedup and
    # the budget are invisible in production and any future regression is unmeasurable.
    context_stats: dict = field(default_factory=dict)

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
                "context_stats": self.context_stats,
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


def assemble_context(results: list[dict], budget: Optional[int] = None) -> tuple[str, dict]:
    """Join ranked chunks into the context string handed to the agent.

    Two properties here, both written after measuring a real answer come back
    incomplete.

    Deduplication.  This FTS index carries the same text more than once - 865 groups of
    byte-identical chunks, 1730 of 4562 (19%), because the Espero manual was ingested
    twice under two source paths.  On the query that exposed this, two of the three
    chunks delivered to the agent were copies of one text, so a third of the context
    was the same page twice while the step the model actually needed sat at rank 6.
    The same text is never useful twice; dropping it costs nothing.

    A character budget instead of a chunk count.  Three chunks is not a size and is
    not even stable across corpora.

    Dedup is on the text with only the edges stripped, never on normalised interior
    whitespace: chunks that differ mid-text are not duplicates, and collapsing
    interior whitespace could merge distinct code blocks.

    The first unique chunk is always included whole, even if it alone exceeds the
    budget - a budget that returns nothing would silently switch retrieval off.
    Chunks are never split, because half a procedure is worse than a missing one.

    ``budget=None`` reads the module constant at call time rather than binding it as a
    default, which is what lets the tests patch the value.
    """
    if budget is None:
        budget = MAX_CONTEXT_CHARS

    seen: set[str] = set()
    parts: list[str] = []
    used = 0
    duplicates = 0
    too_large = 0

    for result in results:
        content = (result.get("content") or "").strip()
        if not content:
            continue
        if content in seen:
            duplicates += 1
            continue
        seen.add(content)

        separator = len(CONTEXT_SEPARATOR) if parts else 0
        if parts and used + separator + len(content) > budget:
            # Skip, do not stop: a long chunk that does not fit must not hide a
            # shorter unique one ranked behind it.
            too_large += 1
            continue

        parts.append(content)
        used += separator + len(content)

    context = CONTEXT_SEPARATOR.join(parts)
    stats = {
        "chunks_considered": len(results),
        "chunks_used": len(parts),
        "chunks_duplicate": duplicates,
        "chunks_too_large": too_large,
        "chars": len(context),
        "budget": budget,
        # True when a unique chunk was dropped for size, i.e. the budget - not the
        # retrieval pool - was the binding constraint.
        "budget_exhausted": too_large > 0,
    }
    return context, stats


def _settled_laya(task: "asyncio.Task[LayaDecision]") -> dict:
    """Read the classifier verdict if it already landed, else cancel the call.

    Used on paths that must not wait for GPU2.  A fire-and-forget request that
    nobody will read is not free: the GPU2 service scores one request at a time,
    and leaving these in flight measurably queued them ahead of the requests that
    *did* need a verdict - the fall-through path went from 55.6 ms to 120.4 ms
    with half its calls timing out, purely from abandoned work.  Enrichment for
    locally answered queries is what shadow mode is for.
    """
    if not task.done():
        task.cancel()
        return not_awaited().to_dict()
    try:
        return task.result().to_dict()
    except (asyncio.CancelledError, Exception):
        return not_awaited().to_dict()


# ── Tier 2: FTS5 + Vector Search ─────────────────────────────────────
class Tier2Search:
    """Fast semantic search: SQLite FTS5 (BM25) + Vector cosine similarity."""

    def __init__(self):
        self._init_fts5()
        # The vector index is read once and re-read only when the file changes. Loading it
        # per request cost ~310 ms of blocked event loop on this hardware while the cosine
        # search itself takes ~13 ms - and because the load is synchronous it also delayed
        # timer callbacks, which is why the classifier's 150 ms budget was observed firing
        # at ~420 ms.
        self._vector_cache: Optional[tuple[tuple[int, int], dict[str, Any]]] = None

    def _load_vector_index(self) -> Optional[dict[str, Any]]:
        """Return the vector index, reloading only if the file changed on disk."""
        path = Path(VECTOR_DB_PATH)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        if self._vector_cache is not None and self._vector_cache[0] == key:
            return self._vector_cache[1]
        try:
            with np.load(path, allow_pickle=False) as db:
                data: dict[str, Any] = {
                    "embeddings": db["embeddings"],
                    "chunk_ids": db["chunk_ids"],
                    "contents": db["contents"],
                    "sources": db["sources"],
                    "domains": db["domains"],
                    "model": str(db["model"].item()),
                    "dimension": int(db["dimension"].item()),
                }
        except (OSError, KeyError, ValueError):
            return None
        self._vector_cache = (key, data)
        return data

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
        index = self._load_vector_index()
        if index is None:
            return []
        embeddings = index["embeddings"]
        chunk_ids = index["chunk_ids"]
        contents = index["contents"]
        sources = index["sources"]
        domains = index["domains"]
        model = index["model"]
        dimension = index["dimension"]

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
        """Get an embedding vector, or None if the embedder cannot answer in budget.

        None is a supported outcome, not an error: the caller skips the vector layer and
        serves the local FTS results.
        """
        try:
            import httpx
            async with asyncio.timeout(EMBEDDING_TIMEOUT_S):
                async with httpx.AsyncClient(timeout=httpx.Timeout(EMBEDDING_TIMEOUT_S)) as client:
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

        # Remote ML is advisory: exact rules retain both retrieval and
        # command-execution authority.  The GPU2 round trip is *started* here so
        # that it overlaps the local index lookup, but it is awaited only if the
        # local path cannot answer on its own.  Measured on an exactly answerable
        # query: route() took 56.0 ms, of which 55.3 ms was this classifier, and
        # the answer that came back did not use the verdict at all.
        laya_task = asyncio.create_task(self.laya.predict_routing(query))
        # The subject domain is decided locally and only locally.  The
        # classifier's own domain question is hand-written rather than trained
        # for, and measured 33.3% against 100% for this rule.
        domain = detect_domain(query)

        # ── Tier 2: FTS5 + Vector search ──
        # FTS is local and is deliberately attempted before embedding.  Calling
        # the embedding server first made an exact local hit wait for the
        # network on every request, defeating the early-exit design.
        fts_results = self.tier2.search_fts5(query)
        fts_exact = [r for r in fts_results if _is_fts_exact(query, r)]
        if fts_exact:
            # Nothing in this answer needs the classifier, so nothing here waits
            # for it: the verdict rides along only if it already arrived.
            context, context_stats = assemble_context(fts_exact)
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
                target_agent=AgentType.GENERAL,
                query=query,
                context=context,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                laya_result=_settled_laya(laya_task),
                context_stats=context_stats,
            )

        # The vector index is optional and this installation does have one
        # (storage/jev_vectors.npz, mxbai-embed-large, 1024d), so this call does go to
        # GPU2 - measured ~32 ms when the box is healthy. That makes it the one
        # remaining GPU dependency on the fall-through path once the graph tier is
        # parked, and its 30 s client timeout is the worst case it can cost: a
        # rebooting GPU2 was observed stalling a request for ~16 s.
        query_embedding = await self.get_embedding(query) if Path(VECTOR_DB_PATH).exists() else None
        vector_results = self.tier2.search_vector(query, query_embedding, domain=domain) if query_embedding is not None else []
        tier2_results = fts_results + [
            item for item in vector_results
            if item["chunk_id"] not in {fts["chunk_id"] for fts in fts_results}
        ]

        # Nothing local answered the query, so the classifier's verdict is worth
        # having - and this is the first point where anything needs it.  Resolving
        # it here rather than before the embedding call means the GPU2 round trip
        # overlaps that network call instead of serialising in front of it.
        laya = await laya_task
        laya_data = laya.to_dict()
        laya_accepted = laya.trusted
        laya_strategy = laya.strategy if laya_accepted else "general_fallback"
        # ML may recommend retrieval, but it never gains authority to execute
        # commands. Direct execution remains exclusively rule-gated above.
        if laya.strategy == "direct_cmd":
            laya_data["execution_override"] = "direct_cmd_requires_regex_match"
            laya_strategy = "general_fallback"
        target_agent = AgentType.DB if laya_accepted and laya_strategy == "database_search" else AgentType.GENERAL

        # Check vector similarity threshold
        best_score = vector_results[0]["score"] if vector_results else 0.0
        # The classifier no longer vetoes this exit.  Its strategy label comes
        # from the invented taxonomy, and a label that is wrong two times in
        # three must not be able to suppress a good vector hit.
        if best_score >= SIMILARITY_THRESHOLD:
            context, context_stats = assemble_context(vector_results)
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
                context_stats=context_stats,
            )

        # ── Tier 3: LightRAG Graph Search ──
        # Determine mode based on query characteristics
        lightrag_mode = self._determine_lightrag_mode(query)

        if LIGHTRAG_ENABLED:
            tier3_result = await self.tier3.search(query, mode=lightrag_mode)
        else:
            # Parked, not broken: no call is made, and the degraded branch below
            # serves the same local retrieval a timeout would have served.
            tier3_result = {"response": "", "error_type": "disabled", "elapsed_ms": 0.0}

        # The graph tier composes its own context, so neither the dedup nor the budget is
        # applied here: assemble_context works on ranked chunks, and rewriting a
        # synthesised answer is a different decision with a different owner.  The
        # fall-back below is our own retrieval and does go through it.
        context = tier3_result.get("response", "")
        context_stats: dict = {}
        fallback_reason = None
        if not context and tier2_results:
            # Degraded-mode fallback: serve retained local retrieval rather
            # than failing the HTTP request when the graph is unavailable.
            context, context_stats = assemble_context(tier2_results)
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
                lightrag_required=LIGHTRAG_ENABLED,
                lightrag_mode=lightrag_mode,
            ),
            target_agent=target_agent,
            query=query,
            context=context,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            degraded=fallback_reason is not None,
            fallback_reason=fallback_reason,
            laya_result=laya_data,
            context_stats=context_stats,
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
