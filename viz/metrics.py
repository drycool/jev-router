"""Что показывать и откуда брать: три источника, три разных класса метрик.

Этот модуль только читает и считает.  Он ничего не рисует и ничего не пишет - ни в базы
Hermes, ни в стор sqz, ни в лог решений Jev: инструмент наблюдения, который меняет
наблюдаемое, измеряет себя, а не систему.

Три источника, потому что вопрос «стало ли дешевле» распадается на три:

- **Hermes (`~/.hermes/state.db`)** - что реально ушло в облако.  Токены лежат на уровне
  сессии (`input_tokens`, `output_tokens`, `cache_read_tokens`), и это единственное место,
  где видно и отношение вход/выход, и долю кэша.  Точность здесь такая: сессия целиком,
  с моделью и источником (cli/telegram).
- **sqz (`~/.sqz/sessions.db`)** - сколько текста он сжал.  Каждое сжатие уже записано в
  `compression_log` (токены до/после, режим, каталог, время), включая сжатия без эффекта -
  и это важно: 68% замеренных сжатий не дали ничего, и увидеть это нужнее, чем красивый
  процент.
- **Jev (`jev_decisions.jsonl`, `jev_feedback.jsonl`)** - что база ответила сама.  Тиры,
  задержки, был ли вердикт и почему его сняли.

Чего здесь **нет** и почему:

- **Токенов по дням.**  Точная разбивка сессии по суткам невозможна: сообщения не несут
  счётчиков (`token_count` пуст), а сессия живёт днями.  Поэтому ряд по дням строится из
  событий, которые считаются точно (вызовы, запросы, сжатия), а токены показываются на
  уровне сессии.  Выдумывать суточный ряд делением сессии на длительность - это рисовать
  данные, которых нет.
- **Цены.**  Стоимость зависит от тарифов, которых инструмент не знает.  Токены
  показываются всегда; цена появляется только если рядом лежит `prices.json` с тарифами
  владельца (`viz/prices.json`), иначе - нет, вместо выдуманных чисел.
- **«Skip Cloud LLM» строкой в логе.**  Такой строки в Jev нет и не будет: локальный ответ
  виден как `strategy` в решении (например `vector_fast`), а не как запись об отсечении.
  Инструмент считает долю запросов, которым база ответила сама, и рядом - число облачных
  вызовов агента за тот же период.  Это сопоставление, а не одна метрика, и оно честнее.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

# Пути по умолчанию.  Переопределяются аргументами: у другой машины они другие, а
# инструмент не должен требовать правки кода, чтобы посмотреть на другой стенд.
HERMES_DB = Path(os.getenv("JEV_VIZ_HERMES_DB", "~/.hermes/state.db")).expanduser()
SQZ_DB = Path(os.getenv("JEV_VIZ_SQZ_DB", "~/.sqz/sessions.db")).expanduser()
# Журнал самого плагина sqz: он пишет по строке на КАЖДОЕ решение, включая
# отказы. База sqz показывает только состоявшиеся сжатия и потому льстит себе:
# из неё видно «16.3% экономии», но не видно ни 15% выводов, которые сжатие
# раздуло, ни того, что 68% сжатий не дали ничего.
PLUGIN_LOG = Path(os.getenv("JEV_VIZ_SQZ_PLUGIN_LOG",
                            "~/.hermes/logs/sqz_plugin.jsonl")).expanduser()
# Лог агента: единственное место, где расход виден НА ЗАПРОС, а не на сессию.  В базе
# `sessions` лежит итог по сессии, `messages.token_count` пуст, `session_model_usage`
# режет по задачам (title_generation и т.п.).  Поэтому вся арифметика холодных
# первых запросов считается здесь - из строк вида
#   `API call #1: model=… in=21779 out=2 total=21781 latency=1.8s cache=1280/21779 (6%)`
AGENT_LOG = Path(os.getenv("JEV_VIZ_AGENT_LOG", "~/.hermes/logs/agent.log")).expanduser()
JEV_DIR = Path(os.getenv("JEV_VIZ_JEV_DIR", str(Path(__file__).resolve().parents[1]))).expanduser()
PRICES_FILE = Path(__file__).resolve().parent / "prices.json"

# Порог «большого» первого запроса.  Голова префикса Hermes - около 21.8 тысяч
# токенов (системный промпт 23.2 тысячи символов плюс 33 инструмента), поэтому
# первый запрос заметно выше этой границы несёт в себе не только голову, а
# перезалив истории сессии при возобновлении или сжатии контекста.
BIG_FIRST_CALL_TOKENS = 60000

# Тиры, которые означают «база ответила сама, модель не нужна».  Список явный, потому что
# от него зависит главная цифра отчёта - доля запросов, не потребовавших облака.
LOCAL_TIERS = ("exact_fts", "vector_fast", "local_material_decisive", "direct_action")
# Тиры, которые означают «материал отдан, но вердикт не вынесен» - вызывающий сам решает.
PARTIAL_TIERS = ("vector_low_confidence", "fts_fallback")
# Тиры, которые означают «ответ дала модель».
MODEL_TIERS = ("graph_lightrag", "general_llm", "general_agent", "fallback")


def _connect(path: Path) -> sqlite3.Connection | None:
    """Открыть базу только на чтение.  Чужой стор не должен блокироваться и не должен
    меняться: sqz работает в WAL, и запись из наблюдателя испортила бы его же данные."""
    if not path.exists():
        return None
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _since(hours: float) -> datetime:
    """Начало окна.  ``hours <= 0`` означает «всё время» и даёт начало эпохи.

    Иначе окно «всё время» фильтровало бы по «сейчас» и показывало пустые таблицы: окно,
    которое ничего не показывает, легко принять за отсутствие данных.
    """
    if hours <= 0:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _day(stamp: str | None) -> str:
    return (stamp or "")[:10]


def percent(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def percentile(values: Sequence[float], share: float) -> float:
    """Перцентиль без numpy: список может быть пустым, а округление должно быть видно."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return round(float(ordered[index]), 1)


# ── Hermes: что ушло в облако ────────────────────────────────────────────────────────

def hermes_summary(hours: float = 24.0, path: Path = HERMES_DB) -> dict:
    """Токены по сессиям + события по дням, которые считаются точно.

    Отношение вход/выход и доля кэша считаются **по сессии**, а не по суткам: счётчики
    живут на сессии, и разложить их по дням нечем.  События (вызовы модели, вызовы
    инструментов, объём текста по инструментам) считаются из сообщений и раскладываются
    по дням точно.
    """
    connection = _connect(path)
    if connection is None:
        return {"available": False, "path": str(path)}
    since = _since(hours)

    sessions = []
    for row in connection.execute("""
            select id, title, source, model, started_at, ended_at, message_count,
                   tool_call_count, input_tokens, output_tokens, cache_read_tokens
            from sessions where started_at >= ? and (input_tokens > 0 or output_tokens > 0)
            order by started_at""", (since.timestamp(),)):
        input_tokens = int(row["input_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        cache_read = int(row["cache_read_tokens"] or 0)
        sessions.append({
            "id": row["id"],
            "title": (row["title"] or "")[:60],
            "source": row["source"] or "",
            "model": row["model"] or "",
            "day": datetime.fromtimestamp(row["started_at"], timezone.utc).isoformat()[:10],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "messages": int(row["message_count"] or 0),
            "tool_calls": int(row["tool_call_count"] or 0),
            # Отношение вход/выход: сколько входа пришлось на один токен ответа.  Именно
            # это число должно падать, если контекст стал дешевле.
            "input_per_output": round(input_tokens / output_tokens, 1) if output_tokens else None,
            # Доля кэша: считаем от всего, что модель прочитала (кэш + обычный вход).
            "cache_share": percent(cache_read, cache_read + input_tokens),
        })

    per_day: dict[str, dict] = {}
    for row in connection.execute("""
            select date(timestamp, 'unixepoch') as day, role, tool_name, count(*) as calls,
                   sum(length(coalesce(content, ''))) as chars
            from messages where timestamp >= ? group by day, role, tool_name""", (since.timestamp(),)):
        day = row["day"] or ""
        bucket = per_day.setdefault(day, {"day": day, "assistant_calls": 0, "tool_results": 0,
                                          "tool_chars": 0, "tool_chars_by_tool": Counter()})
        if row["role"] == "assistant":
            bucket["assistant_calls"] += int(row["calls"])
        elif row["role"] == "tool":
            bucket["tool_results"] += int(row["calls"])
            bucket["tool_chars"] += int(row["chars"] or 0)
            bucket["tool_chars_by_tool"][row["tool_name"] or "(без имени)"] += int(row["chars"] or 0)

    # Что съедает контекст: объём текста по инструментам.  Это рычаг, на который sqz может
    # давить, и он виден без токенов - по символам, которые точно посчитаны.
    tool_volume = Counter()
    for bucket in per_day.values():
        tool_volume.update(bucket["tool_chars_by_tool"])
    total_tool_chars = sum(tool_volume.values()) or 1

    connection.close()
    by_model: dict[str, dict] = {}
    for session in sessions:
        key = f'{session["source"]}/{session["model"]}'
        bucket = by_model.setdefault(key, {"model": key, "sessions": 0, "input_tokens": 0,
                                           "output_tokens": 0, "cache_read_tokens": 0})
        bucket["sessions"] += 1
        bucket["input_tokens"] += session["input_tokens"]
        bucket["output_tokens"] += session["output_tokens"]
        bucket["cache_read_tokens"] += session["cache_read_tokens"]
    for bucket in by_model.values():
        bucket["input_per_output"] = (round(bucket["input_tokens"] / bucket["output_tokens"], 1)
                                      if bucket["output_tokens"] else None)
        bucket["cache_share"] = percent(bucket["cache_read_tokens"],
                                        bucket["cache_read_tokens"] + bucket["input_tokens"])

    days = sorted(per_day)
    for bucket in per_day.values():
        bucket["tool_chars_by_tool"] = dict(bucket["tool_chars_by_tool"].most_common(8))
    return {
        "available": True,
        "path": str(path),
        "window_hours": hours,
        "sessions": sessions,
        "by_model": sorted(by_model.values(), key=lambda item: -item["input_tokens"]),
        "per_day": [per_day[day] for day in days],
        "tool_volume": [{"tool": tool, "chars": chars, "share": percent(chars, total_tool_chars)}
                        for tool, chars in tool_volume.most_common(12)],
        "total_tool_chars": total_tool_chars,
    }


# ── sqz: сколько текста он сжал ──────────────────────────────────────────────────────

def sqz_summary(hours: float = 24.0, path: Path = SQZ_DB) -> dict:
    """Сжатия sqz: по дням, по эффекту, по источнику, плюс самые крупные.

    Сжатия без эффекта считаются отдельно и не прячутся: `--no-cache` в плагине Hermes
    отключает дедуп-ссылки, поэтому повторяющийся текст не превращается в короткий
    указатель и остаётся как есть.  Процент экономии без этой цифры вводит в заблуждение.
    """
    connection = _connect(path)
    if connection is None:
        return {"available": False, "path": str(path)}
    since = _since(hours)

    rows = [dict(row) for row in connection.execute("""
            select id, tokens_original, tokens_compressed, stages_applied, mode, created_at,
                   coalesce(project_dir, '') as project_dir
            from compression_log where created_at >= ? order by created_at""",
                                                          (since.isoformat(),))]
    connection.close()

    before = sum(row["tokens_original"] for row in rows)
    after = sum(row["tokens_compressed"] for row in rows)
    per_day: dict[str, dict] = {}
    for row in rows:
        day = _day(row["created_at"])
        bucket = per_day.setdefault(day, {"day": day, "compressions": 0, "tokens_before": 0,
                                          "tokens_after": 0, "no_op": 0})
        bucket["compressions"] += 1
        bucket["tokens_before"] += row["tokens_original"]
        bucket["tokens_after"] += row["tokens_compressed"]
        if row["tokens_compressed"] >= row["tokens_original"]:
            bucket["no_op"] += 1
    for bucket in per_day.values():
        bucket["saved"] = bucket["tokens_before"] - bucket["tokens_after"]
        bucket["saving_percent"] = percent(bucket["saved"], bucket["tokens_before"])
        bucket["no_op_share"] = percent(bucket["no_op"], bucket["compressions"])

    by_source = Counter()
    for row in rows:
        by_source[row["project_dir"] or "(нет)"] += row["tokens_original"]

    return {
        "available": True,
        "path": str(path),
        "window_hours": hours,
        "compressions": len(rows),
        "tokens_before": before,
        "tokens_after": after,
        "saved": before - after,
        "saving_percent": percent(before - after, before),
        "no_op": sum(1 for row in rows if row["tokens_compressed"] >= row["tokens_original"]),
        "per_day": [per_day[day] for day in sorted(per_day)],
        "by_source": [{"dir": directory, "tokens": tokens}
                      for directory, tokens in by_source.most_common(8)],
        "largest": sorted(
            ({"id": row.get("id"), "before": row["tokens_original"],
              "after": row["tokens_compressed"], "when": row["created_at"][:16],
              "dir": row["project_dir"][-40:]}
             for row in rows),
            key=lambda item: -(item["before"] - item["after"]))[:8],
    }


# ── Jev: что база ответила сама ──────────────────────────────────────────────────────

def sqz_plugin_summary(hours: float = 24.0, path: Path = PLUGIN_LOG) -> dict:
    """Что плагин sqz сделал на самом деле: по строке журнала на каждое решение.

    Отличие от `sqz_summary`, которое здесь и есть смысл: база sqz хранит только
    состоявшиеся сжатия. Журнал плагина хранит и отказы - `no_gain` (сжатие
    раздуло текст или не дотянуло до порога), `tool_not_listed` (инструмент вне
    белого списка), `no_payload` (в конверте нет текста), `error`, `timeout`.
    Поэтому доля полезных сжатий здесь считается от ПОПЫТОК, а не от успехов.
    """
    if not path.exists():
        return {"available": False, "path": str(path),
                "hint": "журнал появится после первого вызова инструмента плагином v2"}
    since = _since(hours).timestamp()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if float(row.get("ts") or 0) >= since:
            rows.append(row)

    per_tool: dict[str, dict] = {}
    actions = Counter()
    chars_in = chars_out = tokens_in = tokens_out = 0
    compressed = refs = 0
    for row in rows:
        action = str(row.get("action") or "?")
        actions[action] += 1
        tool = str(row.get("tool") or "?")
        bucket = per_tool.setdefault(tool, {"tool": tool, "attempts": 0, "compressed": 0,
                                            "chars_in": 0, "chars_out": 0,
                                            "tokens_in": 0, "tokens_out": 0})
        bucket["attempts"] += 1
        if action in ("compressed", "dedup_ref"):
            compressed += 1
            bucket["compressed"] += 1
            refs += 1 if action == "dedup_ref" else 0
            for key, value in (("chars_in", row.get("chars_in")), ("chars_out", row.get("chars_out")),
                               ("tokens_in", row.get("tokens_in")), ("tokens_out", row.get("tokens_out"))):
                bucket[key] += int(value or 0)
            chars_in += int(row.get("chars_in") or 0)
            chars_out += int(row.get("chars_out") or 0)
            tokens_in += int(row.get("tokens_in") or 0)
            tokens_out += int(row.get("tokens_out") or 0)
    for bucket in per_tool.values():
        bucket["saving_percent"] = percent(bucket["chars_in"] - bucket["chars_out"], bucket["chars_in"])
        bucket["useful_share"] = percent(bucket["compressed"], bucket["attempts"])

    return {
        "available": True,
        "path": str(path),
        "window_hours": hours,
        "attempts": len(rows),
        "compressed": compressed,
        "refs": refs,
        "useful_share_percent": percent(compressed, len(rows)),
        "chars_in": chars_in,
        "chars_out": chars_out,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "saving_percent": percent(chars_in - chars_out, chars_in),
        "actions": dict(actions.most_common()),
        "by_tool": sorted(per_tool.values(), key=lambda row: -row["chars_in"]),
    }


def jev_summary(hours: float = 24.0, directory: Path = JEV_DIR) -> dict:
    """Решения роутера: тиры, задержки, вердикты, причины отказа — и обратная связь."""
    path = Path(directory) / "jev_decisions.jsonl"
    feedback_path = Path(directory) / "jev_feedback.jsonl"
    if not path.exists():
        return {"available": False, "path": str(path)}
    since = _since(hours).isoformat()
    decisions = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (record.get("timestamp") or "") >= since.replace("+00:00", ""):
                decisions.append(record)

    per_day: dict[str, dict] = {}
    latencies, tiers, blocked, strategies = [], Counter(), Counter(), Counter()
    answered = 0
    local_answers = 0
    for record in decisions:
        decision = record.get("decision") or {}
        execution = record.get("execution") or {}
        signals = record.get("signals") or {}
        # Ответ от пробы отличается тем, что проба не отдаёт материал: `execute=false`
        # (`answer_chars` пусто) - это замер, а не работа агента.  Без этого разделения
        # калибровочные прогоны выглядят как использование базы, и главная цифра отчёта
        # («доля локальных ответов») описывает маршрутизатор, а не поведение агента.
        is_answer = bool(signals.get("answered")) or (signals.get("answer_chars") or 0) > 0
        answered += 1 if is_answer else 0
        strategy = str(decision.get("strategy") or "?")
        # Пересечение двух признаков, а не отношение двух итогов: «локально» считается по
        # тирам среди всех записей, «ответы» - другое подмножество, и их отношение дало бы
        # больше 100% (проверено на живых данных: 869.6%).
        if is_answer and strategy in LOCAL_TIERS:
            local_answers += 1
        day = _day(record.get("timestamp"))
        bucket = per_day.setdefault(day, {"day": day, "requests": 0, "answers": 0, "local": 0,
                                          "partial": 0, "model": 0, "latency_ms": []})
        bucket["requests"] += 1
        bucket["answers"] += 1 if is_answer else 0
        if strategy in LOCAL_TIERS:
            bucket["local"] += 1
        elif strategy in PARTIAL_TIERS:
            bucket["partial"] += 1
        else:
            bucket["model"] += 1
        if execution.get("latency_ms"):
            bucket["latency_ms"].append(float(execution["latency_ms"]))
            latencies.append(float(execution["latency_ms"]))
        tiers[strategy] += 1
        if execution.get("decisive_blocked"):
            blocked[str(execution["decisive_blocked"])] += 1
        if strategy == "vector_low_confidence" and execution.get("decisive_floor") is not None:
            strategies["ниже плана запаса"] += 1

    for bucket in per_day.values():
        bucket["local_share"] = percent(bucket["local"], bucket["requests"])
        bucket["median_latency_ms"] = percentile(bucket["latency_ms"], 0.5)
        bucket["p95_latency_ms"] = percentile(bucket["latency_ms"], 0.95)
        bucket.pop("latency_ms")

    labels = Counter()
    if feedback_path.exists():
        with feedback_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    try:
                        labels[str(json.loads(line).get("label") or json.loads(line).get("verdict")
                                   or "?")] += 1
                    except json.JSONDecodeError:
                        continue

    return {
        "available": True,
        "path": str(path),
        "window_hours": hours,
        "requests": len(decisions),
        # Сколько из них отдали материал, а сколько было пробами (execute=false).
        "answers": answered,
        "local_answers": local_answers,
        "probes": len(decisions) - answered,
        "tiers": dict(tiers.most_common()),
        "local": sum(count for tier, count in tiers.items() if tier in LOCAL_TIERS),
        "partial": sum(count for tier, count in tiers.items() if tier in PARTIAL_TIERS),
        "model": sum(count for tier, count in tiers.items()
                     if tier not in LOCAL_TIERS and tier not in PARTIAL_TIERS),
        "blocked": dict(blocked.most_common()),
        "median_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "per_day": [per_day[day] for day in sorted(per_day)],
        "feedback": dict(labels.most_common()),
    }


# ── Сборка и производные ─────────────────────────────────────────────────────────────

def load_prices(path: Path = PRICES_FILE) -> dict:
    """Тарифы владельца, если он их положил.  Нет файла - нет цен: выдумывать нельзя."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload.get("models", payload) if isinstance(payload, dict) else {}


def estimate_cost(tokens: dict, price: dict) -> float | None:
    """Стоимость по тарифу вида {input, output, cache_read} в единицах за 1M токенов."""
    if not price:
        return None
    total = 0.0
    for key, field in (("input", "input_tokens"), ("output", "output_tokens"),
                       ("cache_read", "cache_read_tokens")):
        rate = price.get(key)
        if rate is None:
            return None
        total += rate * tokens.get(field, 0) / 1_000_000
    return round(total, 4)


def estimate_cost_off_peak(tokens: dict, price: dict) -> float | None:
    """Тот же расчёт по внепиковому тарифу, если он задан.

    У DeepSeek цена вне пика вдвое ниже, а пик занимает лишь 01:00-04:00 и
    06:00-10:00 UTC по будням, то есть около 7 часов из 24 и ноль в выходные.
    Одно число здесь было бы выдумкой: замер идёт по сессиям, а сессия может
    пересекать границу тарифа, поэтому показываем обе границы - верхнюю
    (пиковую) и нижнюю (внепиковую).
    """
    off_peak = (price or {}).get("off_peak")
    if not isinstance(off_peak, dict):
        return None
    return estimate_cost(tokens, off_peak)


def price_for(prices: dict, model: str) -> dict | None:
    """Тариф для имени модели, включая длинные имена моделей-файлов.

    Точный ключ, потом имя после префикса поверхности (`cli/`), потом
    вхождение: имя вида `/home/dry/llama.cpp/models/qwen2.5-3b-instruct-q5_k_m.gguf`
    после отрезания первого слэша не совпадает ни с чем, если хранить его целиком.
    Из совпадений берётся самое длинное, чтобы `ornith-1.5:9b` не перехватило
    `ornith-1.5-9b-256k`.
    """
    if not prices or not model:
        return None
    for key in (model, model.split("/", 1)[-1]):
        if key in prices:
            return prices[key]
    matches = [key for key in prices if key and key in model]
    if not matches:
        return None
    return prices[max(matches, key=len)]


_CALL_LINE = re.compile(
    r"API call #(?P<number>\d+): model=(?P<model>\S+) provider=(?P<provider>\S+) "
    r"in=(?P<input>\d+) out=(?P<output>\d+) total=(?P<total>\d+) latency=(?P<latency>[\d.]+)s "
    r"cache=(?P<hit>\d+)/(?P<prompt>\d+) \((?P<share>\d+)%\)")
_SESSION_STAMP = re.compile(r"\[(?P<day>\d{8})_\d{6}_[0-9a-f]+\]")


def agent_calls(path: Path = AGENT_LOG) -> list[dict]:
    """Расход по каждому запросу к модели - из лога агента.

    Только здесь видно номер запроса внутри сессии, а значит и то, что первый
    запрос стоит дороже всех остальных вместе.  День берётся из идентификатора
    сессии в той же строке: у самого лога даты нет, только время.
    """
    if not path.exists():
        return []
    calls: list[dict] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            found = _CALL_LINE.search(line)
            if not found:
                continue
            stamp = _SESSION_STAMP.search(line)
            day = stamp.group("day") if stamp else ""
            calls.append({
                "number": int(found.group("number")),
                "model": found.group("model"),
                "provider": found.group("provider"),
                "input_tokens": int(found.group("input")),
                "output_tokens": int(found.group("output")),
                "cache_read_tokens": int(found.group("hit")),
                "share": int(found.group("share")),
                "latency": float(found.group("latency")),
                # 20260928 -> 2026-09-28, чтобы день сортировался как текст
                "day": f"{day[:4]}-{day[4:6]}-{day[6:8]}" if len(day) == 8 else "",
            })
    return calls


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def _oldest_day(hours: float) -> str:
    return _since(hours).strftime("%Y-%m-%d")


def cold_prefix_summary(hours: float = 0.0, path: Path = AGENT_LOG,
                        prices: dict | None = None,
                        sessions_path: Path = HERMES_DB) -> dict:
    """Холодный старт: сколько стоит ПЕРВЫЙ запрос сессии и кто именно его ест.

    Префикс-кэш провайдера байтовый, поэтому первый запрос сессии почти всегда
    платит полную цену за то, что живёт в начале контекста.  Замер показал две
    разные причины, и они требуют разных действий:

    * малый первый запрос (меньше порога) - это сама голова: системный промпт,
      список инструментов, индекс скиллов.  Лечится прогревом головы и тем,
      чтобы она не менялась между сессиями;
    * большой первый запрос - это перезалив уже накопленной истории при
      возобновлении сессии или после сжатия контекста.  Прогревом головы не
      лечится вовсе.

    Деньги считаются как «сколько те же токены стоили бы из кэша»: это верхняя
    граница, а не факт - у нового префикса кэша ещё нет и взять его неоткуда.
    """
    all_calls = agent_calls(path)
    oldest = _oldest_day(hours)
    calls = [call for call in all_calls if hours <= 0 or not call["day"] or call["day"] >= oldest]

    by_number: list[dict] = []
    for number in range(1, 13):
        bucket = [call for call in calls if call["number"] == number]
        if bucket:
            by_number.append(_bucket_row(str(number), bucket))
    tail = [call for call in calls if call["number"] > 12]
    if tail:
        by_number.append(_bucket_row("13+", tail))

    first = [call for call in calls if call["number"] == 1]
    small = [call for call in first if call["input_tokens"] < BIG_FIRST_CALL_TOKENS]
    big = [call for call in first if call["input_tokens"] >= BIG_FIRST_CALL_TOKENS]

    unpriced: set[str] = set()
    tax_peak = 0.0
    tax_off_peak = 0.0
    for call in first:
        price = price_for(prices or {}, call["model"])
        if not price:
            if call["model"]:
                unpriced.add(call["model"])
            continue
        missed = max(0, call["input_tokens"] - call["cache_read_tokens"])
        tax_peak += missed * (price["input"] - price["cache_read"]) / 1e6
        off_peak = price.get("off_peak") or price
        tax_off_peak += missed * (off_peak["input"] - off_peak["cache_read"]) / 1e6

    days = sorted({call["day"] for call in calls if call["day"]})
    heads, sessions = _prompt_heads(hours, sessions_path)

    return {
        # available - есть ли лог вообще; calls - сколько запросов попало в окно.
        # Разделять важно: «нет лога» и «в окне нет запросов» требуют разных действий.
        "available": bool(all_calls),
        "path": str(path),
        "in_window": bool(calls),
        "calls": len(calls),
        "window_days": len(days),
        "first_day": days[0] if days else "",
        "last_day": days[-1] if days else "",
        "by_number": by_number,
        "first_calls": {
            "calls": len(first),
            "input_tokens": sum(call["input_tokens"] for call in first),
            "cache_read_tokens": sum(call["cache_read_tokens"] for call in first),
            "share": percent(sum(call["cache_read_tokens"] for call in first),
                             sum(call["input_tokens"] for call in first)),
            "tax_peak": round(tax_peak, 4),
            "tax_off_peak": round(tax_off_peak, 4),
        },
        "small": _bucket_row("малые (только голова)", small),
        "big": _bucket_row("большие (перезалив истории)", big),
        # Размер головы: нижняя четверть малых первых запросов.  Ни медиана, ни
        # минимум здесь не годятся: медиана вбирает сессии, возобновлённые с
        # накопленной историей (41 тысяча вместо реальных 21.8), а минимум ловит
        # сессию с намеренно урезанным набором инструментов (3.9 тысячи).
        "prefix_low": int(percentile([call["input_tokens"] for call in small], 0.25)),
        "prefix_median": int(_median([call["input_tokens"] for call in small])),
        "heads": heads,
        "sessions": sessions,
        "unpriced": sorted(unpriced),
        "threshold": BIG_FIRST_CALL_TOKENS,
    }


def _bucket_row(label: str, calls: Sequence[dict]) -> dict:
    input_tokens = sum(call["input_tokens"] for call in calls)
    hit = sum(call["cache_read_tokens"] for call in calls)
    return {
        "label": label,
        "calls": len(calls),
        "input_tokens": input_tokens,
        "cache_read_tokens": hit,
        "missed_tokens": input_tokens - hit,
        "share": percent(hit, input_tokens),
    }


def _prompt_heads(hours: float, sessions_path: Path) -> tuple[int, int]:
    """Сколько РАЗНЫХ голов было у сессий в окне: хеш системного промпта.

    Каждая новая голова - это гарантированный холодный первый запрос у всех
    сессий, которые её используют.  Дата в конце системного промпта делает
    голову уникальной на сутки, поэтому счётчик растёт.
    """
    connection = _connect(sessions_path)
    if connection is None:
        return 0, 0
    try:
        row = connection.execute(
            "select count(distinct system_prompt_hash), count(*) from sessions "
            "where system_prompt_hash is not null and started_at >= ?",
            (_since(hours).timestamp(),)).fetchone()
    except sqlite3.Error:
        return 0, 0
    finally:
        connection.close()
    return int(row[0] or 0), int(row[1] or 0)


def dashboard(hours: float = 24.0, hermes_db: Path = HERMES_DB, sqz_db: Path = SQZ_DB,
              jev_dir: Path = JEV_DIR, plugin_log: Path = PLUGIN_LOG,
              agent_log: Path = AGENT_LOG) -> dict:
    """Одна страница данных: три источника плюс то, ради чего они вместе.

    Главная производная - сопоставление: сколько запросов база закрыла сама и сколько
    вызовов модели за это же время сделал агент.  Одно число здесь нечестно, пара - даёт
    ответ на вопрос «стало дешевле или нет».
    """
    hermes = hermes_summary(hours, hermes_db)
    sqz = sqz_summary(hours, sqz_db)
    plugin = sqz_plugin_summary(hours, plugin_log)
    jev = jev_summary(hours, jev_dir)
    prices = load_prices()
    cold = cold_prefix_summary(hours, agent_log, prices, hermes_db)

    cloud_calls = sum(day["assistant_calls"] for day in hermes.get("per_day", []))
    tokens = {
        "input_tokens": sum(session["input_tokens"] for session in hermes.get("sessions", [])),
        "output_tokens": sum(session["output_tokens"] for session in hermes.get("sessions", [])),
        "cache_read_tokens": sum(session["cache_read_tokens"]
                                 for session in hermes.get("sessions", [])),
    }
    cost_by_model: list[dict] = []
    unpriced: list[str] = []
    for bucket in hermes.get("by_model", []):
        price = price_for(prices, bucket["model"])
        if price is None:
            # Модель без тарифа называется вслух: молча выкинуть её из счёта
            # значит показать стоимость, которой не существует.
            unpriced.append(bucket["model"])
            continue
        cost = estimate_cost(bucket, price)
        if cost is None:
            unpriced.append(bucket["model"])
            continue
        entry = {"model": bucket["model"], "cost": cost,
                 "sessions": bucket.get("sessions", 0),
                 "input_tokens": bucket.get("input_tokens", 0),
                 "output_tokens": bucket.get("output_tokens", 0),
                 "cache_read_tokens": bucket.get("cache_read_tokens", 0)}
        off_peak = estimate_cost_off_peak(bucket, price)
        if off_peak is not None:
            entry["cost_off_peak"] = off_peak
        cost_by_model.append(entry)
    cost_by_model.sort(key=lambda item: -item["cost"])
    cost_total = round(sum(item["cost"] for item in cost_by_model), 4)
    # Верхняя граница - пиковый тариф, нижняя - внепиковый (там, где он есть).
    cost_total_off_peak = round(sum(item.get("cost_off_peak", item["cost"]) for item in cost_by_model), 4)
    spent_sessions = len(hermes.get("sessions", []))

    return {
        "window_hours": hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hermes": hermes,
        "sqz": sqz,
        "sqz_plugin": plugin,
        "jev": jev,
        "cold": cold,
        "combined": {
            "cloud_calls": cloud_calls,
            "jev_requests": jev.get("requests", 0),
            "jev_local": jev.get("local", 0),
            "jev_partial": jev.get("partial", 0),
            "local_share_percent": percent(jev.get("local", 0), jev.get("requests", 0)),
            # Доля локальных ответов, посчитанная только по настоящим ответам: цифра про
            # маршрутизатор считается по всем запросам, цифра про агента - по ответам.
            "local_share_of_answers": percent(jev.get("local_answers", 0), jev.get("answers", 0)),
            "jev_local_answers": jev.get("local_answers", 0),
            "jev_answers": jev.get("answers", 0),
            "jev_probes": jev.get("probes", 0),
            # Вызовов модели на один запрос к базе.  Обратное отношение (0.25) читается
            # как «четверть вызова на запрос» и путает: смысл в том, сколько раз агент
            # всё равно пошёл в облако, имея базу под рукой.
            "cloud_calls_per_jev_request": (round(cloud_calls / jev.get("requests", 0), 2)
                                            if jev.get("requests") else None),
            "input_per_output": (round(tokens["input_tokens"] / tokens["output_tokens"], 1)
                                 if tokens["output_tokens"] else None),
            "cache_share_percent": percent(tokens["cache_read_tokens"],
                                           tokens["cache_read_tokens"] + tokens["input_tokens"]),
            "sqz_saved_tokens": sqz.get("saved", 0),
            "sqz_saving_percent": sqz.get("saving_percent", 0),
            # Цифры плагина считаются от попыток, а не от успехов: отказ тоже
            # результат, и без него «экономия» выглядит лучше, чем есть.
            "plugin_attempts": plugin.get("attempts", 0),
            "plugin_compressed": plugin.get("compressed", 0),
            "plugin_saving_percent": plugin.get("saving_percent", 0),
            "plugin_useful_share": plugin.get("useful_share_percent", 0),
            "cost": cost_by_model,
            "cost_total": cost_total,
            "cost_total_off_peak": cost_total_off_peak,
            "cost_unpriced_models": unpriced,
            # Цена «за задачу»: делим на сессии с расходом, потому что задача в
            # этих данных и есть сессия - другого знаменателя в базе нет.
            "cost_per_session": round(cost_total / spent_sessions, 4) if spent_sessions else None,
            "cost_per_session_off_peak": (round(cost_total_off_peak / spent_sessions, 4)
                                          if spent_sessions else None),
            "cost_currency": prices.get("_currency", "USD") if isinstance(prices, dict) else "USD",
            "prices_configured": bool(prices),
            # Холодный старт: первый запрос сессии - единственный, который платит
            # полную цену за голову контекста.  Считается по логу агента, потому что
            # только там виден номер запроса внутри сессии.
            "cold_calls": cold.get("calls", 0),
            "cold_first_calls": (cold.get("first_calls") or {}).get("calls", 0),
            "cold_first_share_percent": (cold.get("first_calls") or {}).get("share", 0),
            "cold_tax_peak": (cold.get("first_calls") or {}).get("tax_peak", 0),
            "cold_tax_off_peak": (cold.get("first_calls") or {}).get("tax_off_peak", 0),
            "cold_prefix_tokens": cold.get("prefix_low", 0),
            "cold_prefix_median_tokens": cold.get("prefix_median", 0),
            "prompt_heads": cold.get("heads", 0),
            "prompt_head_sessions": cold.get("sessions", 0),
        },
        "notes": [
            "Токены считаются по сессиям Hermes (в сообщениях счётчиков нет), поэтому ряд "
            "по дням строится из событий: вызовы модели, запросы Jev, сжатия sqz.",
            "Сжатие sqz идёт от попыток, а не от удач: сжатие, которое раздуло текст или "
            "не дотянуло до порога 10%, отбрасывается, и это видно в журнале плагина как "
            "no_gain. Процент экономии без этой цифры льстит себе.",
            "Цифра «доля локальных ответов» - про запросы к базе, а не про шаги агента: "
            "агент может спросить базу и всё равно позвать модель.",
            "Запросы разделены на ответы и пробы: проба (execute=false) материал не отдаёт "
            "и работой агента не является - калибровочные прогоны иначе выглядят как "
            "использование базы.",
            "Стоимость не показывается, пока нет viz/prices.json с тарифами владельца: "
            "цены зависят от тарифа, а выдуманный тариф хуже отсутствующего.",
            "Холодный старт считается по логу агента (в базе номер запроса внутри сессии "
            "не хранится), поэтому окно у него - по дням из идентификаторов сессий, и оно "
            "может быть короче окна страницы: лог ротируется.",
            "Деньги холодного старта - верхняя граница, а не факт: это стоимость тех же "
            "токенов по цене кэша. У нового префикса кэша ещё нет, и взять его неоткуда.",
        ],
    }
