#!/usr/bin/env python3
"""Веб-слой: одна страница про связку Jev + sqz + Hermes, плюс то же в JSON.

    python3 viz/server.py --port 8035

Страница собирается на сервере, без внешних CDN и без JS-фреймворков: машина живёт в
локальной сети, и дашборд, который не открывается без интернета, не дашборд.  Графики -
инлайновый SVG, посчитанный из тех же чисел, что отдаёт `viz/metrics.py`.  Одно число
проверяется в трёх местах (CLI, JSON, страница), потому что картинка, построенная на
непроверенном числе, врёт убедительнее текста.

    GET /                     страница
    GET /api/metrics?hours=N  данные (N часов; 0 - за всё время)
    GET /health               живость
"""
from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from viz.metrics import dashboard  # noqa: E402

WINDOWS = (("24 ч", 24), ("7 дней", 168), ("30 дней", 720), ("всё время", 0))

app = FastAPI(title="jev-viz", docs_url=None, redoc_url=None)


def human(value: float) -> str:
    for limit, suffix in ((1_000_000_000, "млрд"), (1_000_000, "млн"), (1_000, "тыс")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return f"{value:,.0f}".replace(",", " ")


def bar(value: float, maximum: float, width: int = 120, colour: str = "#58a6ff") -> str:
    """Полоска как SVG-прямоугольник: без библиотек и без магии."""
    span = int(width * (value / maximum)) if maximum else 0
    return (f'<svg width="{width}" height="12" class="bar"><rect width="{width}" height="12" '
            f'fill="#21262d"/><rect width="{max(span, 1)}" height="12" fill="{colour}"/></svg>')


def sparkline(values: list[float], width: int = 240, height: int = 40, colour: str = "#3fb950") -> str:
    """Ряд по дням.  Нормируется по максимуму и подписывается крайними днями."""
    if not values:
        return '<span class="muted">нет данных</span>'
    peak = max(values) or 1
    step = width / max(len(values) - 1, 1)
    points = " ".join(f"{i * step:.1f},{height - (value / peak) * (height - 4) - 2:.1f}"
                      for i, value in enumerate(values))
    return (f'<svg width="{width}" height="{height}" class="spark">'
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.5"/>'
            f'<line x1="0" y1="{height - 1}" x2="{width}" y2="{height - 1}" stroke="#30363d"/>'
            f'</svg>')


def rows(table: list[list[str]]) -> str:
    head, *body = table
    cells = "".join(f"<th>{html.escape(str(value))}</th>" for value in head)
    lines = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
                    for row in body)
    return f'<table><thead><tr>{cells}</tr></thead><tbody>{lines}</tbody></table>'


def plugin_block(plugin: dict) -> str:
    """Цифры плагина sqz: от попыток, а не от удач.

    База sqz хранит только состоявшиеся сжатия и потому показывает экономию без
    знаменателя. Журнал плагина хранит и отказы (`no_gain` - сжатие раздуло текст
    или не дотянуло до порога, `tool_not_listed` - инструмент вне белого списка),
    поэтому доля полезных сжатий здесь честная.
    """
    if not plugin.get("available"):
        return (f'<h3>Плагин sqz</h3><p class="muted">журнала нет: '
                f'{html.escape(plugin.get("path", ""))} — '
                f'{html.escape(plugin.get("hint", ""))}</p>')
    actions = " · ".join(f"{name} {count}" for name, count in (plugin.get("actions") or {}).items())
    tool_rows = [[row["tool"], str(row["attempts"]), str(row["compressed"]),
                  f'{row["useful_share"]}%', human(row["chars_in"]), human(row["chars_out"]),
                  f'{row["saving_percent"]}%'] for row in plugin.get("by_tool", [])]
    return f"""
          <h3>Плагин sqz <span class="muted">версия с журналом решений</span></h3>
          <div class="cards">
            <div class="card"><div class="k">попыток сжатия</div><div class="v">{plugin['attempts']}</div>
              <div class="muted">удачных {plugin['compressed']} ({plugin['useful_share_percent']}%)</div></div>
            <div class="card"><div class="k">экономия</div><div class="v">{plugin['saving_percent']}%</div>
              <div class="muted">{human(plugin['chars_in'] - plugin['chars_out'])} символов</div></div>
            <div class="card"><div class="k">символов</div>
              <div class="v">{human(plugin['chars_in'])} → {human(plugin['chars_out'])}</div>
              <div class="muted">токенов {human(plugin['tokens_in'])} → {human(plugin['tokens_out'])}</div></div>
          </div>
          <p class="muted">решения: {html.escape(actions or "нет")}</p>
          {rows([["инструмент", "попыток", "сжато", "доля", "вход", "выход", "%"]] + tool_rows)
           if tool_rows else ""}"""


def render_page(payload: dict, hours: float) -> str:
    hermes, sqz, jev, combined = payload["hermes"], payload["sqz"], payload["jev"], payload["combined"]
    plugin = payload.get("sqz_plugin") or {"available": False}
    links = " ".join(
        f'<a class="{"on" if label_hours == hours else ""}" href="/?hours={label_hours}">{label}</a>'
        for label, label_hours in WINDOWS)

    sections = []

    # 1. Что ушло в облако
    if hermes.get("available"):
        sessions = hermes["sessions"]
        total_in = sum(s["input_tokens"] for s in sessions)
        total_out = sum(s["output_tokens"] for s in sessions)
        total_cache = sum(s["cache_read_tokens"] for s in sessions)
        model_rows = [[html.escape(bucket["model"]), str(bucket["sessions"]),
                       human(bucket["input_tokens"]), human(bucket["output_tokens"]),
                       str(bucket["input_per_output"]), f'{bucket["cache_share"]}%']
                      for bucket in hermes["by_model"][:8]]
        tool_rows = [[html.escape(item["tool"]), human(item["chars"]), f'{item["share"]}%',
                      bar(item["share"], 100)] for item in hermes["tool_volume"][:8]]
        sections.append(f"""
        <section>
          <h2>Что ушло в облако <span class="muted">Hermes, {len(sessions)} сессий с расходом</span></h2>
          <div class="cards">
            <div class="card"><div class="k">вход</div><div class="v">{human(total_in)}</div>
              <div class="muted">токенов</div></div>
            <div class="card"><div class="k">выход</div><div class="v">{human(total_out)}</div>
              <div class="muted">токенов</div></div>
            <div class="card"><div class="k">кэш-чтение</div><div class="v">{human(total_cache)}</div>
              <div class="muted">токенов, доля {combined['cache_share_percent']}%</div></div>
            <div class="card"><div class="k">вход/выход</div>
              <div class="v">{combined['input_per_output'] if combined['input_per_output'] is not None else '—'}</div>
              <div class="muted">входных на один ответный</div></div>
            <div class="card"><div class="k">вызовов модели</div>
              <div class="v">{combined['cloud_calls']}</div>
              <div class="muted">{combined['cloud_calls_per_jev_request']} на один запрос к базе</div></div>
          </div>
          <h3>По моделям</h3>
          {rows([["модель", "сессий", "вход", "выход", "вх/вых", "кэш"]] + model_rows)}
          <h3>Что занимает место в контексте <span class="muted">по символам, точно</span></h3>
          {rows([["источник", "символов", "доля", ""]] + tool_rows)}
        </section>""")
    else:
        sections.append(f'<section><h2>Hermes</h2><p class="muted">нет базы {html.escape(hermes.get("path", ""))}</p></section>')

    # 2. Сжатие
    if sqz.get("available"):
        day_rows = [[day["day"], str(day["compressions"]), human(day["tokens_before"]),
                     human(day["saved"]), f'{day["saving_percent"]}%', f'{day["no_op_share"]}%']
                    for day in sqz["per_day"][-10:]]
        biggest = [[f'#{item["id"]}', human(item["before"]), human(item["after"]),
                    item["when"], html.escape(item["dir"])] for item in sqz["largest"][:6]]
        no_op_share = round(100 * sqz["no_op"] / sqz["compressions"]) if sqz["compressions"] else 0
        sections.append(f"""
        <section>
          <h2>Сжатие <span class="muted">sqz</span></h2>
          <div class="cards">
            <div class="card"><div class="k">сжатий</div><div class="v">{sqz['compressions']}</div>
              <div class="muted">без эффекта {sqz['no_op']} ({no_op_share}%)</div></div>
            <div class="card"><div class="k">экономия</div><div class="v">{sqz['saving_percent']}%</div>
              <div class="muted">{human(sqz['saved'])} токенов</div></div>
            <div class="card"><div class="k">до / после</div>
              <div class="v">{human(sqz['tokens_before'])} → {human(sqz['tokens_after'])}</div>
              <div class="muted">токенов</div></div>
          </div>
          {sparkline([day["saved"] for day in sqz["per_day"]])}
          <h3>По дням</h3>
          {rows([["день", "сжатий", "вход", "экономия", "%", "пустых"]] + day_rows)}
          <h3>Крупнейшие сжатия</h3>
          {rows([["id", "до", "после", "когда", "каталог"]] + biggest)}
          {plugin_block(plugin)}
        </section>""")
    else:
        # Журнал плагина - отдельный источник: он есть даже когда база sqz
        # недоступна, и прятать его вместе с базой значит терять единственные
        # честные цифры сжатия.
        sections.append(f'<section><h2>Сжатие <span class="muted">sqz</span></h2>'
                        f'<p class="muted">нет стора {html.escape(sqz.get("path", ""))}</p>'
                        f'{plugin_block(plugin)}</section>')

    # 3. Что база ответила сама
    if jev.get("available"):
        tier_rows = [[html.escape(tier), str(count), f'{round(100 * count / max(jev["requests"], 1))}%',
                      bar(count, max(jev["tiers"].values()) if jev["tiers"] else 1)]
                     for tier, count in jev["tiers"].items()]
        day_rows = [[day["day"], str(day["requests"]), str(day["local"]),
                     f'{day["local_share"]}%', f'{day["median_latency_ms"]} мс',
                     f'{day["p95_latency_ms"]} мс'] for day in jev["per_day"][-10:]]
        sections.append(f"""
        <section>
          <h2>Что база ответила сама <span class="muted">Jev, {jev['requests']} запросов</span></h2>
          <div class="cards">
            <div class="card"><div class="k">локально</div>
              <div class="v">{combined['local_share_percent']}%</div>
              <div class="muted">{jev['local']} из {jev['requests']}</div></div>
            <div class="card"><div class="k">ответов / проб</div>
              <div class="v">{jev['answers']} / {jev['probes']}</div>
              <div class="muted">проба = замер, не работа агента · локально из ответов
                {combined['local_share_of_answers']}%</div></div>
            <div class="card"><div class="k">материал без вердикта</div><div class="v">{jev['partial']}</div>
              <div class="muted">вызывающий решает сам</div></div>
            <div class="card"><div class="k">через модель</div><div class="v">{jev['model']}</div>
              <div class="muted">graph / general</div></div>
            <div class="card"><div class="k">медиана</div><div class="v">{jev['median_latency_ms']} мс</div>
              <div class="muted">p95 {jev['p95_latency_ms']} мс</div></div>
          </div>
          {sparkline([day["local_share"] for day in jev["per_day"]], colour="#d29922")}
          <h3>Тиры</h3>
          {rows([["тир", "запросов", "доля", ""]] + tier_rows)}
          <h3>По дням</h3>
          {rows([["день", "запросов", "локально", "доля", "медиана", "p95"]] + day_rows)}
          {f'<p class="muted">вердикт снят: {html.escape(str(jev["blocked"]))}</p>' if jev["blocked"] else ''}
          {f'<p class="muted">обратная связь: {html.escape(str(jev["feedback"]))}</p>' if jev["feedback"] else ''}
        </section>""")

    notes = "".join(f"<li>{html.escape(note)}</li>" for note in payload["notes"])
    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev + sqz + Hermes: что стало дешевле</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ margin:0; padding:1.5rem; background:#0d1117; color:#e6edf3;
        font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }}
 h1 {{ font-size:1.3rem; margin:0 0 .3rem; }}
 h2 {{ font-size:1.05rem; margin:0 0 .8rem; color:#58a6ff; }}
 h3 {{ font-size:.9rem; margin:1.2rem 0 .4rem; color:#8b949e; text-transform:uppercase;
       letter-spacing:.04em; }}
 .muted {{ color:#8b949e; font-weight:400; font-size:.85em; }}
 nav {{ margin:.8rem 0 1.4rem; display:flex; gap:.5rem; flex-wrap:wrap; }}
 nav a {{ color:#8b949e; text-decoration:none; border:1px solid #30363d; border-radius:6px;
          padding:.25rem .7rem; }}
 nav a.on {{ color:#0d1117; background:#58a6ff; border-color:#58a6ff; }}
 section {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
            padding:1.1rem 1.2rem; margin-bottom:1.1rem; }}
 .cards {{ display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:.6rem; }}
 .card {{ background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:.6rem .9rem;
          min-width:8.5rem; }}
 .card .k {{ color:#8b949e; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }}
 .card .v {{ font-size:1.35rem; font-weight:600; }}
 .spark {{ margin:.4rem 0 .2rem; }}
 table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }}
 td, th {{ padding:.28rem .5rem; border-bottom:1px solid #21262d; text-align:left; }}
 thead th, tr:first-child td {{ color:#8b949e; font-size:.8rem; text-transform:uppercase;
                     letter-spacing:.04em; font-weight:400; }}
 .bar {{ vertical-align:middle; }}
 ul {{ margin:0; padding-left:1.2rem; color:#8b949e; }}
</style></head><body>
<h1>Jev + sqz + Hermes: что стало дешевле</h1>
<div class="muted">окно: {"всё время" if hours <= 0 else f"последние {hours:g} ч"} · сформировано {payload['generated_at'][:19]}Z
 · запросов к базе {jev.get('requests', 0)} ({jev.get('answers', 0)} ответов, {jev.get('probes', 0)} проб),
из них локально {combined['local_share_percent']}%
 · облачных вызовов {combined['cloud_calls']} · кэш {combined['cache_share_percent']}%
 · sqz сэкономил {human(combined['sqz_saved_tokens'])} токенов ({combined['sqz_saving_percent']}%)</div>
<nav>{links} <a href="/api/metrics?hours={hours:g}">JSON</a></nav>
{''.join(sections)}
<section><h2>Как это читать</h2><ul>{notes}</ul></section>
</body></html>"""


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "jev-viz"}


@app.get("/api/metrics")
async def api_metrics(hours: float = Query(24.0, ge=0)) -> JSONResponse:
    return JSONResponse(dashboard(hours=hours))


@app.get("/", response_class=HTMLResponse)
async def index(hours: float = Query(24.0, ge=0)) -> HTMLResponse:
    return HTMLResponse(render_page(dashboard(hours=hours), hours))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8035)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
