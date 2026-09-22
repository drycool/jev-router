"""Decisive check: official Laya preset vs our custom Jev schema.

Motivation. With our custom ``strategy``/``domain`` criteria the model answered
`raspberry_pi` for a car-engine question and `direct_cmd` for "rewrite the router
in Go" — i.e. it looked biased toward the first option in each list. Two
explanations fit:

  1. the model genuinely cannot decide, and falls back to a position prior, or
  2. our criteria text is out of the distribution the checkpoint was trained on.

This script separates them by asking the SAME Russian queries through the
preset that ships with the package (``laya.router_questions``) and comparing the
answers. Run on GPU2 where the checkpoints live.
"""
import json
import sys

import laya
from laya import Router, router_questions

QUERIES = [
    "Запусти проверку статуса",
    "Проверь свободное место на диске",
    "Какие кабели обсуждались для Raspberry Pi 5?",
    "Какой блок питания нужен для Raspberry Pi 5 с NVMe?",
    "Почему двигатель Espero троит на холодную?",
    "Где находится реле зажигания у Espero?",
    "Перепиши роутер на Go",
    "Объясни, почему падает сервис под нагрузкой",
    "Привет, как дела?",
    "Что такое BM25?",
]


def main() -> None:
    router = Router(device="cuda")
    router.preload(["multilingual"])
    questions = router_questions()

    print("schema: laya.router_questions() — domain is the shipped 6-way preset")
    print(f"{'domain':<15}{'conf':>7}{'difficulty':>11}{'needs_tools':>12}  query")
    print("-" * 96)
    for query in QUERIES:
        result = router.predict({"request": query}, questions)
        answers = result["answers"]
        domain = answers["domain"]
        difficulty = answers["difficulty"]
        tools = answers["needs_tools"]
        print(
            f"{domain['choice']:<15}{domain['confidence']:>7.3f}"
            f"{difficulty['score']:>11.2f}{tools['noul']:>12.2f}  {query}"
        )
        if "--json" in sys.argv:
            print(json.dumps(result, ensure_ascii=False))

    print("\ndomain probabilities per query (position bias check):")
    for query in QUERIES:
        result = router.predict({"request": query}, questions)
        probabilities = result["answers"]["domain"]["probabilities"]
        ordered = ", ".join(f"{name}={value:.2f}" for name, value in probabilities.items())
        print(f"  {query[:44]:<46} {ordered}")


if __name__ == "__main__":
    main()
