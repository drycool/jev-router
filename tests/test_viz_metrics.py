"""Tests for the Jev + sqz + Hermes dashboard: the numbers, not the pixels.

The page is only as honest as the arithmetic under it, so every figure the page shows is
checked here against a fixture whose answer is known by construction.  The most important
case is the last one: a machine where the three sources do not exist yet must render a page
that says so, because a dashboard that crashes on a fresh install is a dashboard nobody
finds out is broken.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from viz.metrics import dashboard, hermes_summary, jev_summary, percent, percentile, sqz_summary  # noqa: E402
from viz.server import render_page  # noqa: E402


def make_hermes(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript("""
        create table sessions(id text primary key, title text, source text, model text,
            started_at real, ended_at real, message_count integer, tool_call_count integer,
            input_tokens integer default 0, output_tokens integer default 0,
            cache_read_tokens integer default 0);
        create table messages(id integer primary key, session_id text, role text,
            content text, tool_name text, token_count integer, timestamp real);
    """)
    now = 1_700_000_000.0
    connection.execute(
        "insert into sessions values ('s1','Работа','cli','deepseek-v4-flash',?,?,40,12,1000,100,9000)",
        (now, now + 60))
    connection.execute(
        "insert into sessions values ('s2','Ещё','telegram','local-9b',?,?,10,2,0,0,0)", (now, now + 60))
    connection.executemany(
        "insert into messages(session_id, role, content, tool_name, timestamp) values (?,?,?,?,?)",
        [("s1", "assistant", "ок", None, now + 1),
         ("s1", "assistant", "ок", None, now + 2),
         ("s1", "tool", "x" * 300, "terminal", now + 3),
         ("s1", "tool", "y" * 100, "read_file", now + 4),
         ("tool-less", "tool", "z" * 500, None, now + 5)])
    connection.commit()
    connection.close()
    return path


def make_sqz(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("""create table compression_log(id integer primary key,
        tokens_original integer, tokens_compressed integer, stages_applied text,
        mode text, created_at text, project_dir text)""")
    stamp = "2099-01-01T00:00:0"
    connection.executemany(
        "insert into compression_log(id, tokens_original, tokens_compressed, created_at, project_dir)"
        " values (?,?,?,?,?)",
        [(1, 1000, 200, stamp + "1+00:00", "/home/dry/Jev"),   # сжатие 80%
         (2, 500, 490, stamp + "2+00:00", "/home/dry/Jev"),    # слабое
         (3, 300, 300, stamp + "3+00:00", "/tmp"),             # без эффекта
         (4, 300, 330, stamp + "4+00:00", "/tmp")])            # выросло - тоже пустое
    connection.commit()
    connection.close()
    return path


def make_jev(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = "2099-01-01T00:00:0"
    decisions = [
        {"schema": "decision_v2", "timestamp": stamp + "1+00:00",
         "decision": {"strategy": "exact_fts"},
         "execution": {"latency_ms": 10.0, "decisive_blocked": None}},
        {"schema": "decision_v2", "timestamp": stamp + "2+00:00",
         "decision": {"strategy": "vector_fast"},
         "execution": {"latency_ms": 20.0, "decisive_blocked": None}},
        {"schema": "decision_v2", "timestamp": stamp + "3+00:00",
         "decision": {"strategy": "vector_low_confidence"},
         "execution": {"latency_ms": 30.0, "decisive_blocked": "ses:"}},
        {"schema": "decision_v2", "timestamp": stamp + "4+00:00",
         "decision": {"strategy": "vector_low_confidence"},
         "execution": {"latency_ms": 40.0, "decisive_blocked": "нет общих слов с вопросом"}},
        {"schema": "decision_v2", "timestamp": stamp + "5+00:00",
         "decision": {"strategy": "graph_lightrag"}, "execution": {"latency_ms": 50.0}},
    ]
    (directory / "jev_decisions.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in decisions), encoding="utf-8")
    (directory / "jev_feedback.jsonl").write_text(
        json.dumps({"label": "accepted"}, ensure_ascii=False) + "\n"
        + json.dumps({"label": "partial"}, ensure_ascii=False) + "\n", encoding="utf-8")
    return directory


class ArithmeticTest(unittest.TestCase):
    def test_share_of_nothing_is_zero_not_a_division_error(self):
        self.assertEqual(percent(5, 0), 0.0)
        self.assertEqual(percent(1, 4), 25.0)

    def test_percentile_of_an_empty_list_is_zero(self):
        self.assertEqual(percentile([], 0.5), 0.0)
        self.assertEqual(percentile([10, 20, 30, 40], 0.5), 30.0)
        self.assertEqual(percentile([10, 20, 30, 40], 0.95), 40.0)


class HermesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = make_hermes(Path(self.directory.name) / "state.db")

    def test_the_ratio_and_the_cache_share_are_computed_per_session(self):
        summary = hermes_summary(hours=0, path=self.path)
        session = next(item for item in summary["sessions"] if item["id"] == "s1")
        self.assertEqual(session["input_per_output"], 10.0)
        self.assertEqual(session["cache_share"], 90.0)     # 9000 кэша из 10000 прочитанного
        # Сессия без расхода в таблицу не попадает: строка «0 токенов» только шумит.
        self.assertEqual([item["id"] for item in summary["sessions"]], ["s1"])

    def test_what_takes_the_room_is_volume_by_tool_not_token_estimates(self):
        summary = hermes_summary(hours=0, path=self.path)
        volume = {item["tool"]: item for item in summary["tool_volume"]}
        self.assertEqual(volume["terminal"]["chars"], 300)
        self.assertEqual(volume["read_file"]["chars"], 100)
        self.assertEqual(volume["(без имени)"]["chars"], 500)
        self.assertAlmostEqual(volume["(без имени)"]["share"], 55.6, places=1)

    def test_events_are_counted_per_day_and_model_calls_are_assistant_messages(self):
        summary = hermes_summary(hours=0, path=self.path)
        day = summary["per_day"][0]
        self.assertEqual(day["assistant_calls"], 2)
        self.assertEqual(day["tool_results"], 3)
        self.assertEqual(summary["by_model"][0]["model"], "cli/deepseek-v4-flash")


class SqzTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = make_sqz(Path(self.directory.name) / "sessions.db")

    def test_a_compression_that_saved_nothing_is_counted_as_such(self):
        summary = sqz_summary(hours=0, path=self.path)
        self.assertEqual(summary["compressions"], 4)
        self.assertEqual(summary["no_op"], 2)               # в том числе та, где текст вырос
        self.assertEqual(summary["tokens_before"], 2100)
        self.assertEqual(summary["tokens_after"], 1320)
        self.assertEqual(summary["saved"], 780)
        self.assertEqual(summary["saving_percent"], 37.1)

    def test_the_biggest_compression_is_named_with_its_directory(self):
        summary = sqz_summary(hours=0, path=self.path)
        self.assertEqual(summary["largest"][0]["before"], 1000)
        self.assertEqual(summary["largest"][0]["after"], 200)
        self.assertIn("Jev", summary["largest"][0]["dir"])


class JevTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.directory_path = make_jev(Path(self.directory.name) / "Jev")

    def test_tiers_are_split_into_local_partial_and_model(self):
        summary = jev_summary(hours=0, directory=self.directory_path)
        self.assertEqual(summary["requests"], 5)
        self.assertEqual(summary["local"], 2)       # exact_fts, vector_fast
        self.assertEqual(summary["partial"], 2)     # vector_low_confidence x2
        self.assertEqual(summary["model"], 1)       # graph_lightrag

    def test_latency_percentiles_and_why_a_verdict_was_withheld(self):
        summary = jev_summary(hours=0, directory=self.directory_path)
        self.assertEqual(summary["median_latency_ms"], 30.0)
        self.assertEqual(summary["p95_latency_ms"], 50.0)
        self.assertEqual(summary["blocked"]["ses:"], 1)
        self.assertEqual(summary["blocked"]["нет общих слов с вопросом"], 1)

    def test_feedback_labels_are_counted(self):
        summary = jev_summary(hours=0, directory=self.directory_path)
        self.assertEqual(summary["feedback"], {"accepted": 1, "partial": 1})


class PageTest(unittest.TestCase):
    def test_a_fresh_machine_renders_a_page_that_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = dashboard(hours=24, hermes_db=root / "нет.db", sqz_db=root / "нет.db",
                                jev_dir=root)
            self.assertFalse(payload["hermes"]["available"])
            self.assertFalse(payload["sqz"]["available"])
            self.assertFalse(payload["jev"]["available"])
            page = render_page(payload, 24)
            self.assertIn("нет базы", page)
            self.assertIn("нет стора", page)
            self.assertIn("Как это читать", page)

    def test_the_page_shows_the_numbers_the_api_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = dashboard(hours=0, hermes_db=make_hermes(root / "state.db"),
                                sqz_db=make_sqz(root / "sessions.db"),
                                jev_dir=make_jev(root / "Jev"))
            page = render_page(payload, 0)
            combined = payload["combined"]
            self.assertIn("40.0%", page)                            # доля локальных ответов
            self.assertIn(str(combined["local_share_percent"]), page)
            self.assertIn(str(combined["cloud_calls"]), page)
            self.assertIn("без эффекта 2", page)
            self.assertIn("exact_fts", page)
            # Цены не выдумываются: без prices.json страница об этом говорит прямо.
            self.assertIn("prices.json", " ".join(payload["notes"]) + page)

    def test_the_page_never_leaks_a_traceback_into_the_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = dashboard(hours=24, hermes_db=root / "нет.db", sqz_db=root / "нет.db",
                                jev_dir=root)
            self.assertNotIn("Traceback", render_page(payload, 24))


if __name__ == "__main__":
    unittest.main()
