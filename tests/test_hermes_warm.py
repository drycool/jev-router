"""Tests for the warm-up script: the parsing, not the call.

The script's own promise is small - "the head of the prefix is warm now" - and a
wrong reading of the log breaks that promise silently in the worst way: it reports
a share that belongs to somebody else's session.  So the checks here are about
attribution (whose call is this?) and about the verdict wording, since the whole
point of the tool is to refuse to claim a saving it did not get.
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hermes_warm import (SESSION_IN_OUTPUT, calls_of,  # noqa: E402
                                 last_session_calls, money, report)


def call_line(number: int, session: str, tokens: int, hit: int, model: str = "deepseek-v4-flash") -> str:
    share = round(100 * hit / tokens) if tokens else 0
    return (f"16:00:{number:02d} - agent.conversation_loop - INFO [{session}] - "
            f"API call #{number}: model={model} provider=deepseek in={tokens} out=2 "
            f"total={tokens + 2} latency=1.5s cache={hit}/{tokens} ({share}%)")


class SessionIdTest(unittest.TestCase):
    """Идентификатор сессии - самое хрупкое место: ошибка стоит чужого расхода."""

    def identifier_in(self, text: str) -> str:
        found = SESSION_IN_OUTPUT.search(text)
        assert found is not None, f"идентификатор не найден в {text!r}"
        return found.group("id")

    def test_the_full_id_is_taken_from_the_stderr_form(self):
        # Жадный [0-9a-f]{6,} откусил бы только «20260928», и расход не нашёлся бы.
        self.assertEqual(self.identifier_in("session_id: 20260928_160736_aeaf44"),
                         "20260928_160736_aeaf44")

    def test_the_full_id_is_taken_from_the_interactive_form(self):
        self.assertEqual(self.identifier_in("Session:        20260928_154725_2f4039"),
                         "20260928_154725_2f4039")

    def test_a_short_identifier_still_matches(self):
        self.assertEqual(self.identifier_in("session_id: mu7zb8zi0lx1uy"), "mu7zb8zi0lx1uy")


class AttributionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log = Path(self.directory.name) / "agent.log"
        self.log.write_text("\n".join([
            call_line(1, "20260901_120000_aaaaaa", 22000, 1300),
            call_line(2, "20260901_120000_aaaaaa", 22100, 21800),
            call_line(1, "20260902_130000_bbbbbb", 21000, 20800),
            "строка без вызова",
        ]) + "\n")

    def tearDown(self):
        self.directory.cleanup()

    def test_calls_are_attributed_to_the_session_that_made_them(self):
        calls = calls_of("20260901_120000_aaaaaa", self.log)
        self.assertEqual([call["number"] for call in calls], [1, 2])
        self.assertEqual(calls[0]["input"], 22000)

    def test_another_session_is_not_mistaken_for_ours(self):
        calls = calls_of("20260902_130000_bbbbbb", self.log)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["hit"], 20800)

    def test_an_unknown_session_yields_nothing_rather_than_everything(self):
        self.assertEqual(calls_of("20260101_000000_zzzzzz", self.log), [])

    def test_a_missing_log_is_not_an_error(self):
        self.assertEqual(calls_of(None, Path(self.directory.name) / "нет.log"), [])

    def test_the_last_session_is_the_newest_one_in_the_log(self):
        calls = last_session_calls(self.log)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["hit"], 20800)


class VerdictTest(unittest.TestCase):
    RATES = {"deepseek-v4-flash": {"input": 0.30, "cache_read": 0.006, "output": 1.20,
                                   "off_peak": {"input": 0.15, "cache_read": 0.003,
                                                "output": 0.60}}}

    def render(self, calls, check=False, json_output=False):
        args = Namespace(check=check, json=json_output)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report(calls, args, "20260901_120000_aaaaaa")
        return buffer.getvalue()

    def test_a_warm_head_says_so(self):
        text = self.render([{"number": 1, "model": "deepseek-v4-flash", "input": 22561,
                             "output": 2, "hit": 22400, "share": 99}])
        self.assertIn("голова тёплая", text)
        self.assertIn("99%", text)

    def test_a_cold_head_names_the_waste_instead_of_hiding_it(self):
        text = self.render([{"number": 1, "model": "deepseek-v4-flash", "input": 21779,
                             "output": 2, "hit": 1280, "share": 6}])
        self.assertIn("ХОЛОДНАЯ", text)
        self.assertIn("оплачен служебным запросом", text)

    def test_the_price_is_a_range_because_the_tariff_has_two_sides(self):
        text = self.render([{"number": 1, "model": "deepseek-v4-flash", "input": 21779,
                             "output": 2, "hit": 1280, "share": 6}])
        peak = money({"model": "deepseek-v4-flash", "input": 21779, "output": 2, "hit": 1280},
                     self.RATES)
        assert peak is not None
        self.assertAlmostEqual(peak[1], peak[0] / 2, places=6)
        self.assertIn("пик", text)

    def test_a_model_without_a_tariff_is_not_given_a_price(self):
        self.assertIsNone(money({"model": "неизвестная", "input": 100, "output": 1, "hit": 0},
                                self.RATES))

    def test_the_json_form_carries_the_verdict(self):
        text = self.render([{"number": 1, "model": "deepseek-v4-flash", "input": 22561,
                             "output": 2, "hit": 22400, "share": 99}], json_output=True)
        self.assertIn('"verdict": "голова тёплая"', text)


if __name__ == "__main__":
    unittest.main()
