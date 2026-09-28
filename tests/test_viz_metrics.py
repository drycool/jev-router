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
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from viz.metrics import (dashboard, estimate_cost, estimate_cost_off_peak,  # noqa: E402
                         hermes_summary, jev_summary, load_prices, percent, percentile,
                         price_for, sqz_plugin_summary, sqz_summary)
from viz.server import bar, cost_block, render_page, rows  # noqa: E402


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


class AnswersVersusProbesTest(unittest.TestCase):
    """Проба (``execute=false``) - замер, а не работа агента.

    Без этого разделения калибровочные прогоны выглядят как использование базы: на этой
    машине 660 из 716 записей - пробы, и «68% локальных ответов» описывает маршрутизатор,
    а не поведение агента.  Цифра, которая описывает не то, что кажется, хуже отсутствующей.
    """

    def test_requests_are_split_into_answers_and_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Jev"
            root.mkdir(parents=True)
            stamp = "2099-01-01T00:00:0"
            records = [
                {"timestamp": stamp + "1+00:00", "decision": {"strategy": "exact_fts"},
                 "signals": {"answered": True, "answer_chars": 900}, "execution": {}},
                {"timestamp": stamp + "2+00:00", "decision": {"strategy": "vector_fast"},
                 "signals": {"answered": True, "answer_chars": 0}, "execution": {}},
                {"timestamp": stamp + "3+00:00", "decision": {"strategy": "vector_fast"},
                 "signals": {"answered": False, "answer_chars": 0}, "execution": {}},
            ]
            (root / "jev_decisions.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in records),
                encoding="utf-8")
            summary = jev_summary(hours=0, directory=root)
        # answered=True считаются ответом даже с пустым текстом: попытка ответа - это
        # ответ; проба от ответа отличается признаком, а не длиной текста.
        self.assertEqual(summary["requests"], 3)
        self.assertEqual(summary["answers"], 2)
        self.assertEqual(summary["probes"], 1)
        self.assertEqual(summary["local_answers"], 2)


class SqzPluginJournalTest(unittest.TestCase):
    """Журнал плагина sqz: экономия считается от ПОПЫТОК, а не от удач.

    База sqz хранит только состоявшиеся сжатия, поэтому «16.3% экономии» ничего не
    говорит о том, чего стоило сжатие.  В журнале плагина лежат и отказы, и вопрос
    «сколько попыток дали пользу» получает честный знаменатель.
    """

    def _rows(self, directory: Path, now: float) -> Path:
        path = directory / "sqz_plugin.jsonl"
        rows = [
            {"ts": now, "tool": "terminal", "action": "compressed", "chars_in": 1000,
             "chars_out": 250, "tokens_in": 500, "tokens_out": 120, "saving_percent": 75.0},
            {"ts": now, "tool": "terminal", "action": "no_gain", "chars_in": 900, "chars_out": 900},
            {"ts": now, "tool": "read_file", "action": "tool_not_listed", "chars_in": 4000},
            {"ts": now, "tool": "execute_code", "action": "compressed", "chars_in": 2000,
             "chars_out": 1000, "tokens_in": 900, "tokens_out": 450, "saving_percent": 50.0},
            {"ts": now - 10 * 86400, "tool": "terminal", "action": "compressed",
             "chars_in": 8000, "chars_out": 100},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return path

    def test_share_is_counted_from_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._rows(root, time.time())
            summary = sqz_plugin_summary(hours=24, path=path)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["attempts"], 4)          # старая запись вне окна
        self.assertEqual(summary["compressed"], 2)
        self.assertEqual(summary["useful_share_percent"], 50.0)
        self.assertEqual(summary["saving_percent"], 58.3)  # 3000 -> 1250
        self.assertEqual(summary["actions"]["tool_not_listed"], 1)
        tools = {row["tool"]: row for row in summary["by_tool"]}
        self.assertEqual(tools["execute_code"]["compressed"], 1)
        self.assertEqual(tools["read_file"]["attempts"], 1)
        self.assertEqual(tools["read_file"]["chars_in"], 0)   # отказ не входит в объём

    def test_all_time_window_includes_everything(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._rows(Path(directory), time.time())
            summary = sqz_plugin_summary(hours=0, path=path)
        self.assertEqual(summary["attempts"], 5)
        self.assertEqual(len(summary["by_tool"]), 3)

    def test_missing_journal_says_so_instead_of_inventing_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = sqz_plugin_summary(hours=24, path=Path(directory) / "нет.jsonl")
        self.assertFalse(summary["available"])
        self.assertIn("журнал", summary["hint"])

    def test_broken_line_does_not_break_the_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sqz_plugin.jsonl"
            path.write_text('{"ts": %f, "tool": "terminal", "action": "compressed", "chars_in": 100, "chars_out": 10}\n'
                            'не json\n' % time.time(), encoding="utf-8")
            summary = sqz_plugin_summary(hours=0, path=path)
        self.assertEqual(summary["attempts"], 1)

    def test_page_shows_plugin_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._rows(root, time.time())
            # Страница собирается из настоящего `dashboard`, а не из собранного
            # руками словаря: иначе тест проверяет выдумку и падает на первом же
            # ключе, который появился в шаблоне позже.
            payload = dashboard(hours=24, hermes_db=root / "нет.db", sqz_db=root / "нет.db",
                                jev_dir=root, plugin_log=path)
            page = render_page(payload, 24)
        self.assertIn("Плагин sqz", page)
        self.assertIn("попыток сжатия", page)
        self.assertIn("execute_code", page)


class TableRenderingTest(unittest.TestCase):
    """Экранирование ячеек не должно убивать диаграммы внутри таблиц.

    При добавлении <thead> я экранировал все ячейки подряд, и полосы-диаграммы
    начали печататься текстом `<svg width=...>`. Тесты этого не заметили, потому
    что искали текст; картинку нашёл глаз на скриншоте. Поэтому проверяем обе
    стороны: готовая SVG остаётся разметкой, данные экранируются.
    """

    def test_bar_survives_and_data_is_escaped(self):
        page = rows([["тир", "доля"], ["<script>alert(1)</script>", bar(30, 120)]])
        self.assertIn("<svg", page)
        self.assertNotIn("&lt;svg", page)
        self.assertIn("&lt;script&gt;", page)

    def test_header_row_is_a_real_thead(self):
        page = rows([["a", "b"], ["1", "2"]])
        self.assertIn("<thead><tr><th>a</th><th>b</th></tr></thead>", page)


class CostTest(unittest.TestCase):
    """Деньги считаются по тарифу владельца, а не по догадке.

    Цену DeepSeek я снял с живой страницы вендора: там пик 01:00-04:00 и 06:00-10:00 UTC
    по будням, всё остальное вне пика и ВДВОЕ дешевле. Одно число тут было бы
    выдумкой, поэтому считается вилка, и обе границы обязаны быть согласованы.
    """

    PEAK = {"input": 0.3, "cache_read": 0.006, "output": 1.2,
            "off_peak": {"input": 0.15, "cache_read": 0.003, "output": 0.6}}
    TOKENS = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 1_000_000}

    def test_off_peak_is_exactly_half(self):
        self.assertEqual(estimate_cost(self.TOKENS, self.PEAK), 1.506)
        self.assertEqual(estimate_cost_off_peak(self.TOKENS, self.PEAK), 0.753)

    def test_rate_set_without_off_peak_gives_none_not_a_guess(self):
        price = {"input": 0.5, "cache_read": 0.05, "output": 3.0}
        self.assertIsNone(estimate_cost_off_peak(self.TOKENS, price))

    def test_missing_rate_means_unpriced_not_free(self):
        """Ноль и «нет данных» - разные вещи: без ставки стоимость не показывается."""
        self.assertIsNone(estimate_cost(self.TOKENS, {"input": 1.0, "output": 1.0}))
        self.assertIsNone(estimate_cost(self.TOKENS, {}))

    def test_price_lookup_by_long_file_name(self):
        prices = {"qwen2.5-3b-instruct-q5_k_m.gguf": {"input": 0.0, "cache_read": 0.0, "output": 0.0}}
        found = price_for(prices, "telegram//home/dry/llama.cpp/models/qwen2.5-3b-instruct-q5_k_m.gguf")
        self.assertIsNotNone(found)

    def test_longest_match_wins(self):
        prices = {"ornith-1.5:9b": {"input": 9.0},
                  "ornith-1.5-9b-256k:latest": {"input": 1.0}}
        self.assertEqual(price_for(prices, "telegram/ornith-1.5-9b-256k:latest"), {"input": 1.0})

    def test_unknown_model_has_no_price(self):
        self.assertIsNone(price_for({"a": {}}, "cli/неизвестная-модель"))

    def test_owner_price_file_is_complete_and_halves_agree(self):
        """Файл владельца: у каждой модели все три ставки, вне пика ровно вдвое ниже."""
        prices = load_prices()
        self.assertTrue(prices, "viz/prices.json не читается")
        for name, price in prices.items():
            for key in ("input", "cache_read", "output"):
                self.assertIn(key, price, f"{name}: нет ставки {key}")
            if "off_peak" in price:
                for key in ("input", "cache_read", "output"):
                    self.assertAlmostEqual(price["off_peak"][key], price[key] / 2, places=9,
                                           msg=f"{name}: внепиковая ставка {key} не вдвое ниже")
        for name in ("deepseek-v4-flash", "deepseek-v4-pro", "ornith-1.5-9b-256k"):
            self.assertIn(name, prices)

    def test_page_shows_both_bounds_and_names_unpriced(self):
        combined = {"cost": [{"model": "cli/deepseek-v4-flash", "cost": 11.52, "cost_off_peak": 5.76,
                              "input_tokens": 9_000_000, "cache_read_tokens": 1_100_000_000,
                              "output_tokens": 2_000_000}],
                    "cost_total": 11.52, "cost_total_off_peak": 5.76,
                    "cost_per_session": 0.0768, "cost_per_session_off_peak": 0.0384,
                    "cost_unpriced_models": ["cli/minimax-m3"]}
        page = cost_block(combined, {})
        self.assertIn("5.76", page)
        self.assertIn("0.0768", page)
        self.assertIn("minimax-m3", page)
        self.assertIn("deepseek-v4-flash", page)

    def test_cost_block_is_empty_without_prices(self):
        self.assertEqual(cost_block({"cost": []}, {}), "")
