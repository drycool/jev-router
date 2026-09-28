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
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

# Пути по умолчанию.  Переопределяются аргументами: у другой машины они другие, а
# инструмент не должен требовать правки кода, чтобы посмотреть на другой стенд.
HERMES_DB = Path(os.getenv("JEV_VIZ_HERMES_DB", "~/.hermes/state.db")).expanduser()
SQZ_DB = Path(os.getenv("JEV_VIZ_SQZ_DB", "~/.sqz/sessions.db")).expanduser()
JEV_DIR = Path(os.getenv("JEV_VIZ_JEV_DIR", str(Path(__file__).resolve().parents[1]))).expanduser()
PRICES_FILE = Path(__file__).resolve().parent / "prices.json"

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


def dashboard(hours: float = 24.0, hermes_db: Path = HERMES_DB, sqz_db: Path = SQZ_DB,
              jev_dir: Path = JEV_DIR) -> dict:
    """Одна страница данных: три источника плюс то, ради чего они вместе.

    Главная производная - сопоставление: сколько запросов база закрыла сама и сколько
    вызовов модели за это же время сделал агент.  Одно число здесь нечестно, пара - даёт
    ответ на вопрос «стало дешевле или нет».
    """
    hermes = hermes_summary(hours, hermes_db)
    sqz = sqz_summary(hours, sqz_db)
    jev = jev_summary(hours, jev_dir)
    prices = load_prices()

    cloud_calls = sum(day["assistant_calls"] for day in hermes.get("per_day", []))
    tokens = {
        "input_tokens": sum(session["input_tokens"] for session in hermes.get("sessions", [])),
        "output_tokens": sum(session["output_tokens"] for session in hermes.get("sessions", [])),
        "cache_read_tokens": sum(session["cache_read_tokens"]
                                 for session in hermes.get("sessions", [])),
    }
    cost_by_model = []
    for bucket in hermes.get("by_model", []):
        price = prices.get(bucket["model"].split("/", 1)[-1]) or prices.get(bucket["model"])
        cost = estimate_cost(bucket, price) if price else None
        if cost is not None:
            cost_by_model.append({"model": bucket["model"], "cost": cost})

    return {
        "window_hours": hours,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hermes": hermes,
        "sqz": sqz,
        "jev": jev,
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
            "cost": cost_by_model,
            "prices_configured": bool(prices),
        },
        "notes": [
            "Токены считаются по сессиям Hermes (в сообщениях счётчиков нет), поэтому ряд "
            "по дням строится из событий: вызовы модели, запросы Jev, сжатия sqz.",
            "sqz в плагине Hermes работает с --no-cache: дедуп-ссылки отключены, поэтому "
            "повторяющийся текст не сжимается и часть сжатий не даёт ничего.",
            "Цифра «доля локальных ответов» - про запросы к базе, а не про шаги агента: "
            "агент может спросить базу и всё равно позвать модель.",
            "Запросы разделены на ответы и пробы: проба (execute=false) материал не отдаёт "
            "и работой агента не является - калибровочные прогоны иначе выглядят как "
            "использование базы.",
            "Стоимость не показывается, пока нет viz/prices.json с тарифами владельца: "
            "цены зависят от тарифа, а выдуманный тариф хуже отсутствующего.",
        ],
    }
