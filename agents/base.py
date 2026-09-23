"""
Jev Agents
===========
Base agent class and specific agent implementations.
"""
import json
import time
import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import httpx

from core.router import MAX_CONTEXT_CHARS

LLM_CONNECT_TIMEOUT_S = float(os.getenv("JEV_LLM_CONNECT_TIMEOUT_S", "2"))
LLM_READ_TIMEOUT_S = float(os.getenv("JEV_LLM_READ_TIMEOUT_S", "60"))

# How much of the assembled context reaches the model, in characters.
#
# This is the third hard-coded slice found in the same pipeline, and the one that made the
# first two inert. The router used to pass `[:3]` chunks; each agent then independently cut
# what it received to 4000 characters (general) or 3000 (code, db, troubleshooter). So the
# assembly could be fixed and made generous and it still would not matter: on the measured
# case the tightening sequence sat past character 4000 of a 6058-character context, and the
# agent answering "my context is incomplete" was reading a prompt that ended before it.
#
# Four different numbers for one decision is also why this was invisible: nothing compared
# them, and the smallest one won silently. There is now one value, it defaults to the
# assembly budget so the two cannot disagree, and it is the only place the context is cut.
LLM_CONTEXT_CHARS = int(os.getenv("JEV_LLM_CONTEXT_CHARS", str(MAX_CONTEXT_CHARS)))


def _clip_context(context: str) -> str:
    """The single place context length is limited before a prompt is built."""
    return context[:LLM_CONTEXT_CHARS]


def _llm_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        connect=LLM_CONNECT_TIMEOUT_S,
        read=LLM_READ_TIMEOUT_S,
        write=LLM_READ_TIMEOUT_S,
        pool=LLM_CONNECT_TIMEOUT_S,
    )


# ── Base Agent ────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    agent: str
    answer: str
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class BaseAgent(ABC):
    """Base class for all Jev agents."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    async def execute(self, query: str, context: str = "", metadata: Any = None) -> AgentResponse:
        pass

    def _elapsed(self, t0: float) -> float:
        return (time.perf_counter() - t0) * 1000


# ── General Agent ─────────────────────────────────────────────────────
class GeneralAgent(BaseAgent):
    """Default agent for general queries using LLM + context."""

    name = "general_agent"
    description = "General-purpose Q&A with RAG context"

    def __init__(self, llm_host: str = "http://192.168.11.87:11434", model: str = "ornith-1.5-9b-256k"):
        self.llm_host = llm_host
        self.model = model

    async def execute(self, query: str, context: str = "", metadata: Any = None) -> AgentResponse:
        t0 = time.perf_counter()

        prompt = f"""Ты — AI-ассистент. Отвечай на основе предоставленного контекста.
Если контекста недостаточно, скажи об этом.

КОНТЕКСТ:
{_clip_context(context) if context else 'Контекст не найден.'}

ВОПРОС: {query}

ОТВЕТ:"""

        try:
            async with httpx.AsyncClient(timeout=_llm_timeout()) as client:
                resp = await client.post(
                    f"{self.llm_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": 16384, "num_predict": 4096},
                    },
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "")
        except Exception as e:
            answer = f"Ошибка LLM: {e}"

        return AgentResponse(
            agent=self.name,
            answer=answer,
            confidence=0.8 if context else 0.3,
            elapsed_ms=self._elapsed(t0),
        )


# ── Code Agent ────────────────────────────────────────────────────────
class CodeAgent(BaseAgent):
    """Agent for code execution and generation."""

    name = "code_agent"
    description = "Code generation, execution, and analysis"

    def __init__(self, llm_host: str = "http://192.168.11.87:11434", model: str = "ornith-1.5-9b-256k"):
        self.llm_host = llm_host
        self.model = model

    async def execute(self, query: str, context: str = "", metadata: Any = None) -> AgentResponse:
        t0 = time.perf_counter()

        prompt = f"""Ты — expert programmer. Отвечай на вопросы по коду, генерируй код, отлаживай.
Всегда давай готовые к запуску примеры.

КОНТЕКСТ:
{_clip_context(context) if context else ''}

ЗАДАЧА: {query}

КОД/ОТВЕТ:"""

        try:
            async with httpx.AsyncClient(timeout=_llm_timeout()) as client:
                resp = await client.post(
                    f"{self.llm_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": 16384, "num_predict": 4096},
                    },
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "")
        except Exception as e:
            answer = f"Ошибка генерации кода: {e}"

        return AgentResponse(
            agent=self.name,
            answer=answer,
            confidence=0.85,
            elapsed_ms=self._elapsed(t0),
        )


# ── DB Agent ──────────────────────────────────────────────────────────
class DBAgent(BaseAgent):
    """Agent for database operations and queries."""

    name = "db_agent"
    description = "Database queries, schema inspection, data operations"

    def __init__(self, llm_host: str = "http://192.168.11.87:11434", model: str = "ornith-1.5-9b-256k"):
        self.llm_host = llm_host
        self.model = model

    async def execute(self, query: str, context: str = "", metadata: Any = None) -> AgentResponse:
        t0 = time.perf_counter()

        # For now, generate SQL/explanation via LLM
        prompt = f"""Ты — database expert. Помогай с SQL-запросами, схемами БД, анализом данных.

КОНТЕКСТ БД:
{_clip_context(context) if context else 'Нет данных о схеме БД.'}

ЗАПРОС: {query}

SQL/ОТВЕТ:"""

        try:
            async with httpx.AsyncClient(timeout=_llm_timeout()) as client:
                resp = await client.post(
                    f"{self.llm_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": 16384, "num_predict": 4096},
                    },
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "")
        except Exception as e:
            answer = f"Ошибка БД агента: {e}"

        return AgentResponse(
            agent=self.name,
            answer=answer,
            confidence=0.8,
            elapsed_ms=self._elapsed(t0),
        )


# ── Troubleshooter Agent ─────────────────────────────────────────────
class TroubleshooterAgent(BaseAgent):
    """Agent for debugging, error analysis, and problem resolution."""

    name = "troubleshooter_agent"
    description = "Error analysis, debugging, root cause investigation"

    def __init__(self, llm_host: str = "http://192.168.11.87:11434", model: str = "ornith-1.5-9b-256k"):
        self.llm_host = llm_host
        self.model = model

    async def execute(self, query: str, context: str = "", metadata: Any = None) -> AgentResponse:
        t0 = time.perf_counter()

        prompt = f"""Ты — senior troubleshooter. Анализируй ошибки, находи root cause, предлагай исправления.
Структурируй ответ: 1) Причина 2) Решение 3) Профилактика.

КОНТЕКСТ:
{_clip_context(context) if context else ''}

ПРОБЛЕМА: {query}

АНАЛИЗ:"""

        try:
            async with httpx.AsyncClient(timeout=_llm_timeout()) as client:
                resp = await client.post(
                    f"{self.llm_host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_ctx": 16384, "num_predict": 4096},
                    },
                )
                resp.raise_for_status()
                answer = resp.json().get("response", "")
        except Exception as e:
            answer = f"Ошибка troubleshooter: {e}"

        return AgentResponse(
            agent=self.name,
            answer=answer,
            confidence=0.85,
            elapsed_ms=self._elapsed(t0),
        )
