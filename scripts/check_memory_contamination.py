#!/usr/bin/env python3
"""Check whether the indexed memory competes with its own control queries.

A document that *describes* a query - in the indexed corpus - competes with the
document that *answers* it. This happened three times in one session, every time
because a finding was written down with the query's own words:

  * the memory document quoted three control queries verbatim, so FTS retrieved the
    quote (bm25 -34.6) instead of the answer, which was not in the pool at all;
  * a paragraph explaining the gate defect repeated two of the query's content
    words and re-created the defect it described;
  * a results table listing "про кабели (Raspberry Pi)" accumulated exactly the two
    content terms the cable query has, which was enough to win the gate.

The rule this enforces: measurement text belongs in the repository (README, commit
bodies, tests), and the indexed memory describes findings without the control
queries' content words.

    python3 scripts/check_memory_contamination.py            # default pairs
    python3 scripts/check_memory_contamination.py --db ... --quiet

Exit code 1 when a competitor exists, so it can be wired into a check.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.router import FTS_GATE_MIN_TERMS, FTS_GATE_STOPWORDS, _extract_keywords  # noqa: E402
from core.vector_index import is_excluded_source  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "storage" / "jev_fts5.db"

# (query, the file that must answer it).  An empty expectation means "no document
# should answer this at all".
CONTROL_QUERIES: list[tuple[str, str]] = [
    ("включать демоны пользователя автоматически при загрузке", "linux_systemd.md"),
    ("проверка типов на асинхронных маршрутах", "fastapi_pydantic.md"),
    ("Как устроена двухслойная конфигурация garageOS и чёрный ящик инцидентов", "garageos.md"),
    ("Кабели для Raspberry Pi 5", "Gemini"),
    ("затяжка болтов головки блока цилиндров момент", "d_espero.pdf"),
    # No document describes the router's own request flow, and the finding that
    # records this is exactly the kind of note that can invent one: writing the
    # question's own words into the corpus would make the note answer the question
    # it says is unanswerable.  Empty expectation = nobody may take the gate.
    ("как идут запросы и есть ли промежуточные сервисы", ""),
    # This one *is* answerable - the board material is the right source - so the
    # expectation names it and the guard only fires on a third document trying to
    # take the gate away from it.
    ("UPS HAT и Orange PI4 Pro возможно взаимодействие?", "carpc"),
    ("Какая погода в Киеве завтра", ""),
    ("Купить билеты на поезд Киев Львов", ""),
]


def competitors(rows, query: str, expected: str) -> list[tuple[str, list[str]]]:
    """Indexed, non-raw chunks that would pass the gate for this query.

    `rows` is an iterable of (chunk_id, content, source).  Pure, so the rule is
    testable without a database.
    """
    keywords = _extract_keywords(query)
    content_keywords = [k for k in keywords if k.lower() not in FTS_GATE_STOPWORDS]
    if not content_keywords:
        return []
    threshold = min(FTS_GATE_MIN_TERMS, len(content_keywords))
    found = []
    for chunk_id, content, source in rows:
        if is_excluded_source(str(source)):
            continue  # the raw corpus is expected to match ordinary words
        if expected and expected in str(chunk_id):
            continue  # the file that should answer is not a competitor
        terms = set(re.findall(r"\b[\wа-яА-ЯёЁ]+\b", str(content).lower()))
        inside = [k for k in content_keywords if k.lower() in terms]
        if len(inside) >= threshold:
            found.append((str(chunk_id), inside))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--quiet", action="store_true", help="print only competitors")
    args = parser.parse_args()
    if not args.db.exists():
        print(f"no FTS5 database at {args.db}", file=sys.stderr)
        return 2

    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = connection.execute("SELECT chunk_id, content, source FROM chunks").fetchall()
    connection.close()

    total = 0
    for query, expected in CONTROL_QUERIES:
        found = competitors(rows, query, expected)
        total += len(found)
        if not args.quiet or found:
            print(f"\n{query}")
            print(f"  ожидает ответа: {expected or 'никто'}")
            for chunk_id, inside in found[:5]:
                print(f"  ЗАРАЖЕНИЕ {chunk_id}  совпало {len(inside)}: {inside}")
            if not found and not args.quiet:
                print("  чисто")
    print(f"\nчанков-конкурентов: {total}")
    if total:
        print("Перепишите описание находки без содержательных слов контрольного запроса.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
