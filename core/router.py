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
# The raw-corpus policy lives in one place and is consumed here rather than
# re-derived: two definitions of "which sources are raw" would drift.
from core.vector_index import is_excluded_source


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
EMBEDDING_MODEL = os.getenv("JEV_EMBEDDING_MODEL", "bge-m3")
# Priority for the memory documents in the BM25 ordering.
#
# The OCR'd Espero manual is 4414 of the 4623 FTS5 rows and its mangled fragments
# match ordinary Russian words *slightly* better than the memory documents do,
# which is how a question about user systemd services came back with a chunk
# about tyre tread (rank -8.489 against -8.079 for linux_systemd.md).  The
# manual stays in FTS5 - literal matching there is not the problem - so the fix
# belongs in the ordering, not in the corpus.
#
# The factor multiplies bm25() **for memory rows only**, and bm25() is negative
# with lower = better, so a factor **greater than 1** is what promotes a row:
# -8.079 * 1.5 = -12.1, which now beats -8.489.  A factor below 1 would demote
# memory, which is the opposite of the intent and an easy sign error to make.
#
# 1.5 is the smallest measured value that moves the shared-stop-word case, and
# it is deliberately mild: the boost reorders rows that already matched, it does
# not add rows, so a query with no memory chunk in the pool is unaffected (the
# manual query and the off-topic controls have zero memory rows in the pool and
# measured identical output at every factor).  Set JEV_FTS_MEMORY_BOOST=1 to
# disable the reordering entirely.
FTS_MEMORY_BOOST = float(os.getenv("JEV_FTS_MEMORY_BOOST", "1.5"))
# Whether the exact-FTS early exit may be taken by the raw corpus alone.
#
# On, the gate prefers golden/memory rows when both survived it, and a
# pool of *only* raw-corpus rows does not exit early at all: the request
# goes on to the vector tier, which searches the golden corpus.  The raw
# rows are not removed from the pool, so the questions the manual itself
# answers keep their content - measured on "затяжка болтов головки блока
# цилиндров момент", whose complete sequence exists only in the manual.
# Off restores the previous behaviour, where any matching fragment could
# take the exit.
FTS_GATE_REQUIRE_GOLDEN = os.getenv(
    "JEV_FTS_GATE_REQUIRE_GOLDEN", "1").strip().lower() not in {"0", "false", "no", "off"}
# How many query terms a candidate must contain to take the early exit.
#
# It was `min(2, len(keywords))`, and two terms out of six is not a match,
# it is a coincidence of ordinary words: measured on the current corpus,
# "проверка типов на асинхронных маршрутах" cleared the gate on a manual
# fragment containing only проверка + типов (bm25 -6.6), and questions about
# user services cleared it on при + загрузке.  Three terms, or every term
# when the query has fewer, is what the measured data separates: the chunks
# that are genuinely the answer match five and six of six ("затяжка болтов
# головки блока цилиндров момент" -> 5), while stop-word coincidences reach
# two.  A query that only clears two now goes on to the vector tier, which
# searches the golden corpus, instead of exiting on a coincidence.
FTS_GATE_MIN_TERMS = int(os.getenv("JEV_FTS_GATE_MIN_TERMS", "3"))
# Content terms a raw-corpus chunk must contain to keep the exit for itself.
#
# The manual is the only source for some questions and is excluded from the
# vector index, so if it may never take the early exit, those questions pay a
# GPU round trip and lead with unrelated golden-corpus chunks.  The measured
# boundary: a chunk that is genuinely the document a query came from contains
# four or more of its content words (the cylinder-head sequence matches five of
# six), while ordinary-word coincidences reach two.  Above the line the manual
# answers its own question locally; below it, the raw rows are held back and
# the vector tier leads.
FTS_GATE_RAW_EXIT_TERMS = int(os.getenv("JEV_FTS_GATE_RAW_EXIT_TERMS", "4"))
# Total budget for the embedding call, enforced with asyncio.timeout (same reason as the
# classifier: an httpx timeout is per socket read, not a deadline). The embedder lives on
# the node that powers itself off when idle, and this is the fall-through path's only
# remaining GPU dependency - a 30 s client timeout let a sleeping node stall a request for
# almost that long (observed ~16 s while the box was coming up). Healthy cost is ~32 ms,
# so this is ~60x headroom. On expiry the vector layer is skipped and the local FTS
# results are served, exactly as when the embedder is unreachable.
EMBEDDING_TIMEOUT_S = float(os.getenv("JEV_VECTOR_TIMEOUT_S", "2.0"))
# How long the embedder is kept loaded on GPU2.  The embedder and the answer
# model share one 12 GB card and ollama unloads an un-kept model, so without
# this the embedding call reloads bge-m3 (4163 ms measured cold) and busts the
# budget above every time the answer model is resident - the vector tier then
# drops out of every request without saying so.  Measured warm: 59 ms.
#
# 30m was too short in practice.  The cost is paid by the first request after
# idle, which is exactly when a reactive router is being judged: measured after
# a 30-minute gap, the embedder was gone and the call took 4.2 s against this
# budget, so the vector tier was skipped and the query fell to local rows.  24h
# keeps it hot across a working day without pinning it forever, which a -1 would
# do at the expense of the 10 GB model that also lives on this card.
EMBEDDING_KEEP_ALIVE = os.getenv("JEV_EMBEDDING_KEEP_ALIVE", "24h")

# Confidence carried by fts_fallback: local rows were found, none of them matched
# decisively (the gate declined the early exit), so the floor of that status's
# band is used and no relevance is claimed.
FTS_FALLBACK_CONFIDENCE = float(os.getenv("JEV_FTS_FALLBACK_CONFIDENCE", "0.5"))

# How deep retrieval goes before context assembly sees anything.  This is a separate
# decision from the budget below, and it is the one that decides whether a needed chunk is
# in the running at all.
#
# It was 10, and 10 was not enough: for "затяжка болтов головки блока цилиндров момент" the
# chunk carrying the complete tightening sequence (stages а-г, 25 Н·м plus the 60°/180°
# follow-up rotation) sits at rank 13 of the query's matches. At 10 it is absent from the
# pool, so no assembly policy could deliver it - which is exactly why the agent kept
# answering that its context was incomplete while the fix to assembly had already landed.
#
# At 20 that query yields 16 unique chunks (11720 characters after dedup). Measured cost of
# the wider pool: 540 chunks in the index match the query's terms, so this is still a
# shallow slice; bm25 ordering means the depth is not free relevance, and note that the FTS
# query is an OR over the terms - deeper results are looser. The exact-hit test is what keeps
# precision, and it is applied unchanged.
RETRIEVAL_LIMIT = int(os.getenv("JEV_RETRIEVAL_LIMIT", "20"))

# How much retrieved text the agent is handed, in characters.  A budget rather than a
# count, because a count cannot know how long a chunk is: the old `[:3]` delivered
# 1691-3078 characters across four real queries and would deliver 300 on a corpus that
# happened to produce three short chunks.  What the router owes the agent is a
# predictable prefill bill, not a fixed number of rows.
#
# The saturation point moves with the pool size, so these two settings are read together.
# Unique chunks delivered, out of a 20-result pool, across the four measured queries:
#
#     budget   query 1   query 2   query 3   query 4
#       4000      4/7       6/16      7/16      4/16
#       6000      5/7       9/16      9/16      9/16
#       8000      7/7      10/16     11/16     12/16
#      12000      7/7      13/16     15/16     16/16
#      16000      7/7      16/16     16/16     16/16   <- saturation
#      20000    byte-identical to 16000; 28000 likewise
#
# 16000 is the value. 12000 was tempting - the tightening-sequence query fits all 16 of its
# chunks there - but on the other three queries it still drops 3, 1 and 0, so it would have
# repeated the original defect at a smaller scale: a budget quietly cutting the context
# while the answer looked fine. Setting it at the measured saturation makes the retrieval
# pool the only limit, and that one is explicit.
#
# Measured cost of the larger context, on this hardware, is not the deciding factor: an A/B
# at 2360 and 7569 characters in (two alternating runs each) came back 5.8/7.3 s and
# 7.8/9.6 s, a spread within one budget as wide as the gap between them, and a direct LLM
# measurement put 4000 characters at 16.7 s and 12000 at 10.6 s. Prefill is cheap.
MAX_CONTEXT_CHARS = int(os.getenv("JEV_MAX_CONTEXT_CHARS", "16000"))
CONTEXT_SEPARATOR = "\n\n"


class Strategy(str, Enum):
    DIRECT_ACTION = "direct_action"
    EXACT_FTS = "exact_fts"
    VECTOR_FAST = "vector_fast"
    GRAPH_LIGHTRAG = "graph_lightrag"
    GENERAL_LLM = "general_llm"
    # Local rows served without a decisive match: the pool exists but failed the
    # gate, so this is the lower tier taken with nothing relevant in it.  Kept
    # distinct from exact_fts because a caller that cannot tell them apart reads a
    # weak pool as an answer.  Confidence stays at the floor of its band - rows
    # were found, nothing matched decisively, and any number above the floor would
    # claim a relevance the gate declined.
    FTS_FALLBACK = "fts_fallback"
    # Vector neighbours that all scored below JEV_SIMILARITY_THRESHOLD.  The tier
    # ran and returned neighbours, but none cleared the bar, so this is not the
    # decisive hit vector_fast names - and the asymmetry would be worse than the
    # gate's: a 0.31 neighbour and a 0.67 hit would wear the same label while only
    # the number told them apart.  Confidence carries the real cosine, which is
    # below the threshold by construction.
    VECTOR_LOW_CONFIDENCE = "vector_low_confidence"
    # The embedder did not answer, so the vector tier never ran this request.  This is
    # the one state in this branch that is a degradation rather than a retrieval
    # outcome, and it used to be indistinguishable from a weak pool: both arrived as
    # fts_fallback with degraded=false, and the only difference was latency (83-149 ms
    # when the embedder answered, 2049-2128 ms when it had been evicted to the CPU) -
    # which no consumer reads.  A caller told "the corpus matched nothing relevant"
    # when the truth is "the semantic search never happened" will trust the wrong
    # conclusion, so the two are separated here and this one is marked degraded.
    #
    # Fires only when the embedder was actually asked.  A missing or empty vector
    # index is a configuration to fix, not an outage to report, and it is not folded
    # in here - see the embedder_attempted flag in route().
    EMBEDDING_TIMEOUT = "embedding_timeout"


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


# Terms that cannot carry a match on their own.  Counting them is how two
# ordinary words out of six passed for evidence: the measured exits were on
# "при" + "загрузке" (a question about user services) and "проверка" + "типов"
# (a question about schema validation, matched by an OCR fragment).  Function
# words are common to every chunk of a 4626-row corpus, so matching them says
# nothing about relevance; only content words are counted.
FTS_GATE_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "ли", "если",
    "уже", "или", "ни", "быть", "был", "него", "до", "вас", "вам", "ведь",
    "там", "потом", "себя", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "тогда",
    "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним",
    "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас",
    "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец",
    "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через",
    "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три",
    "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда",
    "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда",
    "конечно", "всю", "между",
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is",
    "are", "with", "how", "what", "why", "does", "do", "my", "our",
}


def _is_fts_exact(query: str, result: dict) -> bool:
    """Require enough *content* terms in a candidate before an early exit.

    The threshold is `min(FTS_GATE_MIN_TERMS, len(content terms))`, so a short
    query is unchanged and a long one needs three real words rather than three
    words of any kind.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return False
    content_keywords = [k for k in keywords if k.lower() not in FTS_GATE_STOPWORDS]
    if not content_keywords:
        # A query of nothing but function words is not a retrieval request; the
        # old behaviour of counting them is the only thing that could be done.
        content_keywords = keywords
    return _content_matches(query, result) >= min(FTS_GATE_MIN_TERMS, len(content_keywords))


def _content_matches(query: str, result: dict) -> int:
    """How many of the query's *content* terms the chunk contains.

    Returned as a count so both the exit rule and the raw-corpus rule
    measure the same thing; the denominator they apply differs.
    """
    keywords = _extract_keywords(query)
    content_keywords = [k for k in keywords if k.lower() not in FTS_GATE_STOPWORDS]
    content_terms = set(re.findall(r"\b[\wа-яА-ЯёЁ]+\b", result["content"].lower()))
    return sum(keyword.lower() in content_terms for keyword in content_keywords)


def _gate_selection(query: str, results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the gate survivors into (may take the early exit, raw rows held back).

    The gate asks whether a candidate contains enough query terms, and the OCR'd
    manual answers that question well: its 4414 mangled fragments match ordinary
    Russian words, so a question about schema validation exited on a chunk about
    wheel alignment while the answer sat in `fastapi_pydantic.md`.  The manual
    stays in FTS5 (a torque question is answered *only* by it), so the raw corpus
    is not removed - it is stopped from taking the early exit on its own:

      * memory/golden rows also survived  -> they take it, raw rows are held back
        from the captured pool (memory wins the ordering argument);
      * **only** raw rows survived        -> nothing takes the early exit here;
        the request continues to the vector tier, which searches the golden
        corpus. If that finds nothing, the ordinary fall-through serves the same
        FTS pool, so the manual's own questions keep their answers.

    The held-back rows are returned rather than dropped, because the manual has
    no vectors (it is excluded from that index): a caller that answers from the
    vector tier must still carry them, or the only copy of a manual answer would
    disappear from the request.

    Sources are matched through `core.vector_index.is_excluded_source`, so the
    policy has exactly one definition (`JEV_VECTOR_EXCLUDE_SOURCES`).  A chunk
    indexed without a source is not the raw corpus.
    """
    survivors = [r for r in results if _is_fts_exact(query, r)]
    if not survivors or not FTS_GATE_REQUIRE_GOLDEN:
        return survivors, []
    golden = [r for r in survivors if not is_excluded_source(str(r.get("source") or ""))]
    raw = [r for r in survivors if is_excluded_source(str(r.get("source") or ""))]
    if golden or not raw:
        return golden, raw
    # Nothing but the raw corpus survived.  It keeps the exit only when its
    # best chunk is decisive - otherwise the held-back rows wait for the
    # vector tier, which searches the golden corpus.
    if _content_matches(query, raw[0]) >= FTS_GATE_RAW_EXIT_TERMS:
        return raw, []
    return [], raw


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
                # Optional: it appeared with the memory tier.  An archive built
                # before that is still searched, just without entity tags.
                if "entity_types" in db.files:
                    data["entity_types"] = db["entity_types"]
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

    def search_fts5(self, query: str, limit: Optional[int] = None) -> list[dict]:
        """BM25 full-text search over the top RETRIEVAL_LIMIT matches.

        ``limit=None`` reads the module constant at call time rather than binding it as a
        default, so the pool depth stays patchable and one setting governs every caller.
        """
        if limit is None:
            limit = RETRIEVAL_LIMIT
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
                          rank AS score,
                          rank * (CASE WHEN entity_type = 'memory' THEN ? ELSE 1.0 END)
                              AS weighted
                   FROM chunks
                   WHERE chunks MATCH ?
                   ORDER BY weighted
                   LIMIT ?""",
                (FTS_MEMORY_BOOST, fts_query, limit),
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
                # bm25() returns a negative rank; lower is more relevant.  This is
                # the raw value, not the memory-weighted one used for ordering:
                # the decision log records what the corpus actually scored, and a
                # boosted number there would be a fabricated relevance.
                "score": abs(float(row[4])),
                "search_type": "fts5",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            })
        return results

    def search_vector(self, query: str, query_embedding: np.ndarray, limit: Optional[int] = None, domain: str = "general", apply_threshold: bool = True) -> list[dict]:
        """Vector cosine similarity search. ~10-30ms. Same pool depth as the FTS path.

        ``apply_threshold`` drops the neighbours that score below
        ``SIMILARITY_THRESHOLD``, which is what the decisive exit wants.  Callers
        that need to *report* on the tier rather than act on it pass ``False``: the
        threshold filter used to be unconditional, so a request whose neighbours all
        fell short saw an empty list and could not tell "the tier ran and found only
        weak neighbours" from "the tier did not run at all"."""

        if limit is None:
            limit = RETRIEVAL_LIMIT
        t0 = time.perf_counter()
        index = self._load_vector_index()
        if index is None:
            return []
        embeddings = index["embeddings"]
        chunk_ids = index["chunk_ids"]
        contents = index["contents"]
        sources = index["sources"]
        domains = index["domains"]
        # Newer archives carry it; older ones do not, and an absent tag is not a
        # reason to drop a result.
        entity_types = index.get("entity_types")
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
            if entity_types is not None:
                entity_types = entity_types[mask]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        q_norm = query_embedding / (np.linalg.norm(query_embedding) or 1)

        scores = normalized @ q_norm
        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if apply_threshold and score < SIMILARITY_THRESHOLD:
                continue
            results.append({
                "chunk_id": str(chunk_ids[idx]),
                "content": str(contents[idx]),
                "source": str(sources[idx]),
                # Which tier the chunk came from. The FTS5 path already carries
                # this; without it here, a memory hit on the vector path would be
                # indistinguishable from a corpus hit in the decision log.
                "entity_type": str(entity_types[idx]) if entity_types is not None else "",
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

        `keep_alive` is sent on every request because the embedder and the answer model
        share one 12 GB card, and ollama unloads a model that is not kept.  Measured
        before this: with the 9B answer model resident, every embedding call reloaded
        bge-m3 (4163 ms cold) and blew the 2.0 s budget, so the vector tier silently
        dropped out of *every* request while the service kept answering from the degraded
        path - 15 of 15 control queries fell back in one run.  With the embedder kept the
        warm call is 59 ms and both models stay loaded (5.7 + 1.3 GB).
        """
        try:
            import httpx
            async with asyncio.timeout(EMBEDDING_TIMEOUT_S):
                async with httpx.AsyncClient(timeout=httpx.Timeout(EMBEDDING_TIMEOUT_S)) as client:
                    resp = await client.post(
                        EMBEDDING_API,
                        json={"model": EMBEDDING_MODEL, "input": text,
                              "keep_alive": EMBEDDING_KEEP_ALIVE},
                    )
                    resp.raise_for_status()
                    data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                return np.array(embeddings[0])
        except Exception as error:
            # A silent None is how a dead embedder stayed invisible for a whole
            # session: the request still answers, from the degraded path.  The
            # caller keeps its behaviour; the operator gets one line.
            print(f"[Jev] embedding unavailable ({type(error).__name__}: {error}); "
                  f"vector tier skipped for this request", flush=True)
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
        # Who may take this exit is decided by _gate_selection: memory rows when
        # they also matched, and nobody at all when the only match is the raw
        # OCR corpus - that case continues to the vector tier below, which
        # carries the held-back raw rows into its context.
        fts_exact, raw_held_back = _gate_selection(query, fts_results)
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
        # Two reasons leave query_embedding None, and they are not the same state: the
        # embedder was asked and did not answer in budget (a degradation to report), or
        # there is no vector index to search (a configuration to fix).  Only the first
        # is an outage, so the flag below gates the degraded status on it - the second
        # keeps reporting whatever the local tiers actually did.  Measured on the live
        # host: the embedder had been evicted to the CPU by two other models sharing the
        # 12 GB card and cost 17.2 s against a 2.0 s budget, so every non-literal query
        # landed here and was reported as fts_fallback with degraded=false.
        vector_index_present = Path(VECTOR_DB_PATH).exists()
        query_embedding = await self.get_embedding(query) if vector_index_present else None
        embedder_failed = vector_index_present and query_embedding is None
        # One search, with the threshold applied out here instead of inside it.  The
        # decisive exit below needs the neighbours that cleared the bar; the parked
        # branch needs to know whether the tier found *anything*, because "ran and
        # found only weak neighbours" and "did not run" are different states and used
        # to look identical (an empty list) from here.
        if query_embedding is not None:
            neighbours = self.tier2.search_vector(
                query, query_embedding, domain=domain, apply_threshold=False)
        else:
            neighbours = []
        vector_results = [
            item for item in neighbours if item["score"] >= SIMILARITY_THRESHOLD]
        # The best neighbour regardless of the bar: this is the only number that can
        # describe a tier which ran and did not clear it.
        best_neighbour_score = neighbours[0]["score"] if neighbours else 0.0
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
            # When the gate held raw rows back, they ride along after the
            # vector hits instead of being dropped: the manual is excluded
            # from the vector index on purpose, so dropping them here would
            # delete the only copy of its answers from the request.  They
            # keep their relative order and lose only their claim to the
            # front of the context.
            context_results = vector_results
            if raw_held_back:
                seen = {item["chunk_id"] for item in vector_results}
                context_results = vector_results + [
                    item for item in raw_held_back if item["chunk_id"] not in seen]
            context, context_stats = assemble_context(context_results)
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
        lightrag_mode = self._determine_lightrag_mode(query)

        if not LIGHTRAG_ENABLED:
            # Parked by configuration, so there is no outage to report.  Naming
            # this result graph_lightrag with degraded=true described a failure
            # that did not happen and hid which tier actually served the request:
            # the graph call is skipped entirely (elapsed_ms 0.0), so the context
            # can only have come from the local tiers above.  Measured on a query
            # the parked tier would have claimed: 2.2 s, degraded=true, while the
            # material served was plain local retrieval.
            #
            # The tier that did the work names the result instead.  Sub-threshold
            # vector hits stay in play because the graph tier that would have
            # covered them is off by configuration, and dropping them would serve
            # strictly less than the parked branch served before.  Confidence is
            # left at this branch's historical 0.7 rather than the decisive 0.95
            # of the early exit: the gate declined that exit, so this result does
            # not claim a decisive local hit.
            context = ""
            context_stats: dict = {}
            context_results = vector_results
            if raw_held_back:
                seen = {item["chunk_id"] for item in vector_results}
                context_results = vector_results + [
                    item for item in raw_held_back if item["chunk_id"] not in seen]
            if not context_results:
                context_results = tier2_results
            if context_results:
                context, context_stats = assemble_context(context_results)

            # Name the tier that answered, and say plainly when nothing did.  Five
            # outcomes live here, and running them together under one label is what
            # this branch got wrong three times: first as graph_lightrag (an outage
            # that never happened), then as exact_fts 0.7 (a decisive-hit label on a
            # pool the gate had just refused), then as fts_fallback for a request whose
            # semantic search had not run at all.  A caller that cannot tell a weak
            # pool from an answer reads the weak pool as the answer; a caller that
            # cannot tell a weak pool from a broken retriever retries the wrong thing.
            #
            #   embedder silent            -> embedding_timeout, degraded
            #   neighbours, none clearing  -> vector_low_confidence, best real cosine
            #   local rows, no gate        -> fts_fallback, floor confidence
            #   nothing local at all       -> general_llm, no context
            #
            # The embedder check comes first because it describes why the branch was
            # reached rather than what it served: when it fires, no vector search
            # happened, so no retrieval conclusion can be drawn from the tier at all -
            # including "the corpus matched nothing relevant", which is what
            # fts_fallback would have said.  The context is still local rows, and the
            # consumer is expected to read this status, not the context, as the verdict.
            #
            # Reaching this branch at all means the decisive exits above were not
            # taken, so any neighbour here is below JEV_SIMILARITY_THRESHOLD: the same
            # label as a hit that cleared the bar would hide the only difference there
            # is.  The neighbours are *reported*, not served - the calibrated bar
            # exists to keep them out of the context, and the local rows keep their
            # place in it.
            #
            # The last case is not a fallback to FTS - there is no FTS row behind
            # it - so it is not labelled as one.  It is the only case where the
            # model has to answer by itself, and saying so is the point.
            degraded = False
            fallback_reason = None
            if embedder_failed:
                strategy = Strategy.EMBEDDING_TIMEOUT
                confidence = 0.0
                intent = "vector_search"
                degraded = True
                fallback_reason = "embedding_timeout"
            elif neighbours:
                strategy = Strategy.VECTOR_LOW_CONFIDENCE
                confidence = best_neighbour_score
                intent = "vector_search"
            elif context_results:
                strategy = Strategy.FTS_FALLBACK
                confidence = FTS_FALLBACK_CONFIDENCE
                intent = "exact_search"
            else:
                strategy = Strategy.GENERAL_LLM
                confidence = 0.0
                intent = "general"

            return RoutingResult(
                routing_decision=RoutingDecision(
                    strategy=strategy,
                    confidence_score=confidence,
                    fast_path_exit=False,
                ),
                extracted_metadata=ExtractedMetadata(
                    intent=intent,
                    keywords=_extract_keywords(query),
                    entities=self._extract_entities(query),
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
                degraded=degraded,
                fallback_reason=fallback_reason,
                laya_result=laya_data,
                context_stats=context_stats,
            )

        tier3_result = await self.tier3.search(query, mode=lightrag_mode)

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
