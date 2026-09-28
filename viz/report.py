#!/usr/bin/env python3
"""Отчёт по связке Jev + sqz + Hermes: те же числа, что увидит WebUI, но текстом.

    python3 viz/report.py                 # окно 24 часа
    python3 viz/report.py --hours 168     # неделя
    python3 viz/report.py --json          # машиночитаемо

Печатается текстом, чтобы числа можно было проверить до того, как они станут картинкой:
график, построенный на непроверенном числе, врёт убедительнее текста.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viz.metrics import dashboard  # noqa: E402


def human_size(value: int) -> str:
    for limit, suffix in ((1_000_000_000, "млрд"), (1_000_000, "млн"), (1_000, "тыс")):
        if abs(value) >= limit:
            return f"{value / limit:.1f}{suffix}"
    return str(value)


def show(payload: dict) -> None:
    window = payload["window_hours"]
    print(f"окно: последние {window:g} ч    сформировано {payload['generated_at'][:19]}Z\n")

    hermes, sqz, jev = payload["hermes"], payload["sqz"], payload["jev"]
    combined = payload["combined"]

    print("── ЧТО УШЛО В ОБЛАКО (Hermes) ─────────────────────────────────────────")
    if not hermes.get("available"):
        print(f"  нет базы {hermes.get('path')}")
    else:
        sessions = hermes["sessions"]
        print(f"  сессий с расходом: {len(sessions)}    вызовов модели: {combined['cloud_calls']}")
        print(f"  вход {human_size(combined_input(hermes))} токенов, "
              f"выход {human_size(combined_output(hermes))}, "
              f"кэш-чтение {human_size(combined_cache(hermes))}")
        print(f"  вход/выход: {combined['input_per_output']}    "
              f"доля кэша: {combined['cache_share_percent']}%")
        print("\n  по моделям:")
        for bucket in hermes["by_model"][:6]:
            print(f"    {bucket['model'][:34]:<34} {bucket['sessions']:>3} сессий  "
                  f"вход {human_size(bucket['input_tokens']):>8}  выход "
                  f"{human_size(bucket['output_tokens']):>7}  вх/вых {str(bucket['input_per_output']):>7}"
                  f"  кэш {bucket['cache_share']:>5}%")
        print("\n  что занимает место в контексте (по символам, точно):")
        for item in hermes["tool_volume"][:6]:
            print(f"    {item['tool'][:26]:<26} {human_size(item['chars']):>8} симв  "
                  f"{item['share']:>5}%")

    print("\n── СЖАТИЕ (sqz) ──────────────────────────────────────────────────────")
    if not sqz.get("available"):
        print(f"  нет стора {sqz.get('path')}")
    else:
        print(f"  сжатий {sqz['compressions']}, из них без эффекта {sqz['no_op']} "
              f"({100 - sqz['saving_percent'] if not sqz['compressions'] else round(100 * sqz['no_op'] / max(sqz['compressions'], 1))}%)")
        print(f"  токенов до {human_size(sqz['tokens_before'])}, после "
              f"{human_size(sqz['tokens_after'])}, сэкономлено {human_size(sqz['saved'])} "
              f"({sqz['saving_percent']}%)")
        for day in sqz["per_day"][-5:]:
            print(f"    {day['day']}  {day['compressions']:>4} сжатий  вход "
                  f"{human_size(day['tokens_before']):>8}  экономия "
                  f"{human_size(day['saved']):>7} ({day['saving_percent']}%)  "
                  f"пустых {day['no_op_share']}%")

    print("\n── ЧТО БАЗА ОТВЕТИЛА САМА (Jev) ──────────────────────────────────────")
    if not jev.get("available"):
        print(f"  нет журнала {jev.get('path')}")
    else:
        print(f"  запросов {jev['requests']}: локально {jev['local']} "
              f"({combined['local_share_percent']}%), материал без вердикта {jev['partial']}, "
              f"через модель {jev['model']}")
        print(f"  задержка: медиана {jev['median_latency_ms']} мс, p95 {jev['p95_latency_ms']} мс")
        print(f"  тиры: {jev['tiers']}")
        if jev["blocked"]:
            print(f"  вердикт снят: {jev['blocked']}")
        if jev["feedback"]:
            print(f"  обратная связь: {jev['feedback']}")

    print("\n── СВЯЗКА ────────────────────────────────────────────────────────────")
    print(f"  вызовов модели на один запрос к базе: {combined['cloud_calls_per_jev_request']}")
    print(f"  доля запросов, закрытых локально:     {combined['local_share_percent']}%")
    print(f"  доля кэша во входе модели:            {combined['cache_share_percent']}%")
    print(f"  экономия sqz:                         {human_size(combined['sqz_saved_tokens'])} токенов "
          f"({combined['sqz_saving_percent']}%)")
    if combined["prices_configured"]:
        print(f"  стоимость: {combined['cost']}")
    else:
        print("  стоимость: нет viz/prices.json - считаются только токены")
    print("\n── КАК ЭТО ЧИТАТЬ ────────────────────────────────────────────────────")
    for note in payload["notes"]:
        print(f"  · {note}")


def combined_input(hermes: dict) -> int:
    return sum(session["input_tokens"] for session in hermes["sessions"])


def combined_output(hermes: dict) -> int:
    return sum(session["output_tokens"] for session in hermes["sessions"])


def combined_cache(hermes: dict) -> int:
    return sum(session["cache_read_tokens"] for session in hermes["sessions"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = dashboard(hours=args.hours)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        show(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
