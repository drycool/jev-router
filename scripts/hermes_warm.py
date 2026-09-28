#!/usr/bin/env python3
"""Прогрев головы префикса перед длинной сессией.

Префикс-кэш провайдера байтовый: если голова контекста (системный промпт, список
инструментов, индекс скиллов) уже лежит в кэше, первый запрос сессии читает её
оттуда и платит только за своё.  Замер на живом контуре: 6% попаданий у первой
сессии против 99% у следующей с той же головой.

    hermes-warm                      # прогреть голову моделью по умолчанию
    hermes-warm --model deepseek-v4-flash --provider deepseek
    hermes-warm --toolsets terminal --in ~/Jev
    hermes-warm --check              # только замерить, не звонить
    hermes-warm --json

ЧЕСТНО О ТОМ, ЧТО ЭТО ПОКУПАЕТ.  Прогрев переносит промах на дешёвый служебный
запрос, а не убирает его: одиночной длинной сессии он не даёт ничего, потому что
та всё равно заплатила бы этот промах один раз.  Он выигрывает, когда одной
головой пользуются НЕСКОЛЬКО запусков (cron, субагенты, серия проб) - тогда один
промах заменяет N.  Многотонные холодные старты (перезалив истории при
возобновлении и сжатии) прогревом головы не лечится вовсе: там голова попадает, а
история нет.  Поэтому скрипт всегда печатает, сколько он на самом деле стоил и
сколько попаданий получила голова.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.metrics import AGENT_LOG, estimate_cost, estimate_cost_off_peak, load_prices, price_for  # noqa: E402

WARM_QUESTION = "ответь одним словом: ок"
CALL_LINE = re.compile(
    r"API call #(?P<number>\d+): model=(?P<model>\S+) provider=(?P<provider>\S+) "
    r"in=(?P<input>\d+) out=(?P<output>\d+) total=(?P<total>\d+) latency=(?P<latency>[\d.]+)s "
    r"cache=(?P<hit>\d+)/(?P<prompt>\d+) \((?P<share>\d+)%\)")
# Hermes печатает идентификатор сессии в stderr как `session_id: 20260928_160736_aeaf44`,
# а в интерактивном режиме - как `Session:        <id>`.  Оба вида нужно поймать:
# без идентификатора расход пришлось бы искать по всему логу и приписать себе чужой.
# Порядок альтернатив важен: `[0-9a-f]{6,}` жадный и от `20260928_160736_aeaf44`
# откусит только дату `20260928`, после чего расход не найдётся ни у одной сессии.
# Полный вид идёт первым.
# Короткие идентификаторы Hermes - не hex (`mu7zb8zi0lx1uy`), поэтому набор символов
# шире: строчные латинские и цифры.
SESSION_IN_OUTPUT = re.compile(
    r"(?:session_id|Session)\s*:\s*(?P<id>\d{8}_\d{6}_[0-9a-f]+|[0-9a-z]{6,})")


def calls_of(session: str | None, path: Path = AGENT_LOG) -> list[dict]:
    """Строки вызовов нашей сессии.  По идентификатору - чтобы не спутать с чужой."""
    if not path.exists():
        return []
    found: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        if session and f"[{session}]" not in line:
            continue
        match = CALL_LINE.search(line)
        if match:
            found.append({key: int(value) if key in ("number", "input", "output", "hit", "share")
                          else value for key, value in match.groupdict().items()})
    return found


def last_session_calls(path: Path = AGENT_LOG) -> list[dict]:
    """Строки последней сессии в логе: нужны для --check, когда мы не звонили сами."""
    if not path.exists():
        return []
    session = None
    for line in reversed(path.read_text(errors="replace").splitlines()):
        stamp = re.search(r"\[(\d{8}_\d{6}_[0-9a-f]+)\]", line)
        if stamp:
            session = stamp.group(1)
            break
    return calls_of(session, path)


def money(call: dict, prices: dict) -> tuple[float, float] | None:
    price = price_for(prices, call["model"])
    if not price:
        return None
    tokens = {"input_tokens": call["input"] - call["hit"], "cache_read_tokens": call["hit"],
              "output_tokens": call["output"]}
    peak = estimate_cost(tokens, price)
    if peak is None:
        return None
    off_peak = estimate_cost_off_peak(tokens, price)
    return peak, (off_peak if off_peak is not None else peak)


def warm(args: argparse.Namespace) -> int:
    command = ["hermes", "chat", "-q", WARM_QUESTION, "-Q"]
    if args.model:
        command += ["-m", args.model]
    if args.provider:
        command += ["--provider", args.provider]
    if args.toolsets:
        command += ["-t", args.toolsets]
    if args.in_dir:
        command += ["--in", args.in_dir]
    command += args.extra or []

    environment = dict(os.environ)
    # Плагин сжатия правит вывод инструментов, а не голову контекста, поэтому его
    # можно оставить включённым: на прогрев он не влияет.
    try:
        finished = subprocess.run(command, capture_output=True, text=True,
                                  timeout=args.timeout, env=environment)
    except subprocess.TimeoutExpired:
        print(f"прогрев не удался: hermes не ответил за {args.timeout} с", file=sys.stderr)
        return 0                                   # никогда не блокируем сессию
    except OSError as error:
        print(f"прогрев не удался: {error}", file=sys.stderr)
        return 0

    session = None
    match = SESSION_IN_OUTPUT.search(finished.stdout + finished.stderr)
    if match:
        session = match.group("id")
    if not session:
        print("прогрев выполнен, но идентификатора сессии в выводе нет - "
              "расход приписать некому, поэтому он не показывается", file=sys.stderr)
        return 0
    calls = calls_of(session)
    if not calls:
        print(f"прогрев выполнен (rc={finished.returncode}), но строки расхода в логе нет: "
              "голова могла не прогреться", file=sys.stderr)
        return 0
    return report(calls, args, session)


def report(calls: list[dict], args: argparse.Namespace, session: str | None) -> int:
    prices = load_prices()
    first = next((call for call in calls if call["number"] == 1), calls[0])
    costs = money(first, prices)
    share = first["share"]
    verdict = "голова тёплая" if share >= 90 else ("голова прогрета" if share >= 50 else "голова ХОЛОДНАЯ")
    if args.json:
        print(json.dumps({"session": session, "model": first["model"], "input_tokens": first["input"],
                          "cache_read_tokens": first["hit"], "share": share,
                          "cost_peak": costs[0] if costs else None,
                          "cost_off_peak": costs[1] if costs else None,
                          "verdict": verdict}, ensure_ascii=False))
        return 0
    print(f"{verdict}: попаданий {share}% ({first['hit']:,} из {first['input']:,} токенов), "
          f"модель {first['model']}")
    if costs:
        print(f"  запрос стоил {costs[0]:.6f} USD (пик) … {costs[1]:.6f} USD (вне пика)")
    else:
        print("  тарифа на эту модель нет - стоимость не считается")
    if args.check:
        print(f"  вызовов в последней сессии лога: {len(calls)}")
    elif len(calls) > 1:
        print(f"  в сессии прогрева вызовов {len(calls)} (ожидался один)")
    if share < 90:
        print("  голова осталась холодной: следующий запуск с той же головой её прочитает, "
              "и вот ЭТОТ промах уже оплачен служебным запросом, а не рабочей сессией")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогреть голову префикса перед длинной сессией")
    parser.add_argument("-m", "--model", help="модель (по умолчанию - та, что у Hermes)")
    parser.add_argument("-p", "--provider", help="маршрут, например deepseek")
    parser.add_argument("-t", "--toolsets", help="набор инструментов, как у будущей сессии")
    parser.add_argument("--in", dest="in_dir", help="рабочий каталог будущей сессии: он входит в промпт")
    parser.add_argument("--check", action="store_true",
                        help="ничего не звонить: показать состояние головы у последней сессии")
    parser.add_argument("--json", action="store_true", help="машиночитаемый вывод")
    parser.add_argument("--timeout", type=int, default=180, help="предел ожидания, секунды")
    parser.add_argument("extra", nargs="*", help="дополнительные аргументы hermes chat")
    args = parser.parse_args()

    if args.check:
        calls = last_session_calls()
        if not calls:
            print("в логе агента нет вызовов модели", file=sys.stderr)
            return 0
        return report(calls, args, None)
    return warm(args)


if __name__ == "__main__":
    sys.exit(main())
