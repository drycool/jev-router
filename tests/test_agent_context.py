"""The prompt context limit: one value, applied by every agent.

This is the third hard-coded slice in the same pipeline and the one that made the other two
inert. The router used to send `[:3]` chunks; each agent then independently cut what it
received — 4000 characters for the general agent, 3000 for code, db and troubleshooter. So
the assembly could be fixed, the budget raised and the pool deepened, and the measured case
would still arrive truncated: the tightening sequence sat past character 4000 of a
6058-character context, and the agent reporting "my context is incomplete" was reading a
prompt that ended before it.

The reason four different numbers survived is that nothing compared them and the smallest
one won silently. These tests are that comparison.
"""
import asyncio
import unittest
from unittest.mock import patch

from agents.base import (
    LLM_CONTEXT_CHARS,
    CodeAgent,
    DBAgent,
    GeneralAgent,
    TroubleshooterAgent,
)
from core.router import MAX_CONTEXT_CHARS

AGENTS = (GeneralAgent, CodeAgent, DBAgent, TroubleshooterAgent)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": "ok"}


class _FakeClient:
    """Stands in for httpx.AsyncClient and records what would have been sent."""

    sent: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json: dict | None = None):  # noqa: A002 - mirrors httpx
        assert json is not None, "the agent posted without a body"
        _FakeClient.sent.append(json)
        return _FakeResponse(json)


class _StubbedAgents(unittest.TestCase):
    def setUp(self):
        _FakeClient.sent = []
        self._patch = patch("agents.base.httpx.AsyncClient", _FakeClient)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _prompt_for(self, agent_class, context: str, query: str = "вопрос") -> str:
        """Run one agent against the stub and return the prompt it would have sent."""
        before = len(_FakeClient.sent)
        agent = agent_class()
        asyncio.run(agent.execute(query=query, context=context, metadata=None))
        self.assertEqual(
            len(_FakeClient.sent), before + 1, "the agent did not call the LLM"
        )
        return _FakeClient.sent[-1]["prompt"]


class PromptContextLimitTests(_StubbedAgents):
    TAIL = "ХВОСТ-КОНТЕКСТА-ЗА-ПРЕДЕЛОМ-СТАРОГО-ЛИМИТА"

    def _long_context(self, length: int = 12000) -> str:
        """A context of the requested length whose tail is identifiable."""
        filler = "данные мануала " * ((length - len(self.TAIL)) // 15 + 1)
        return (filler + self.TAIL)[: length - len(self.TAIL)] + self.TAIL

    def test_the_general_agent_sends_the_whole_assembled_context(self):
        """The case that failed: the needed text was past character 4000."""
        context = self._long_context(12000)
        prompt = self._prompt_for(GeneralAgent, context)
        self.assertIn(self.TAIL, prompt)

    def test_no_agent_keeps_a_private_smaller_limit(self):
        """Every agent is compared against each other, not just against its own history."""
        context = self._long_context(12000)
        for agent_class in AGENTS:
            with self.subTest(agent=agent_class.__name__):
                prompt = self._prompt_for(agent_class, context)
                self.assertIn(self.TAIL, prompt)
                self.assertLessEqual(len(prompt), len(context) + 400)

    def test_the_limit_comes_from_the_constant(self):
        """Lowering the one value is what cuts the context - nothing else does."""
        context = self._long_context(12000)
        with patch("agents.base.LLM_CONTEXT_CHARS", 500):
            prompt = self._prompt_for(GeneralAgent, context)
            self.assertIn(context[:400], prompt)
            self.assertNotIn(self.TAIL, prompt)

    def test_the_default_matches_the_assembly_budget(self):
        """Two settings for one decision is how the last defect hid; these cannot drift."""
        self.assertEqual(LLM_CONTEXT_CHARS, MAX_CONTEXT_CHARS)

    def test_a_short_context_is_not_padded_or_mangled(self):
        prompt = self._prompt_for(GeneralAgent, "короткий контекст")
        self.assertIn("короткий контекст", prompt)


if __name__ == "__main__":
    unittest.main()
