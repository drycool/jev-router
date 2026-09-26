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
bodies, tests), and the indexed **memory** describes findings without the control
queries' content words.

The scope is the memory feed, and that was decided by measurement, not by taste.
Once `jev-collect` indexed this repository, the guard reported 22 competitors and
every one of them was legitimate: a project's README and its commit messages record
the measurements made in this project, so they quote the control queries *by
design*.  Scoring them as contamination would demand rewriting history, and the
questions are genuinely answered there - the router's request flow, the user-level
systemd deployment and the two-layer garageOS configuration are all documented in
the project's own material.  A collected source may contain any words; only the
material written as prose about the system is held to the rule.

Consequence worth remembering when measuring: a control query can now be "answered"
by a commit message that talks about that measurement, so routing numbers taken
with these queries are no longer clean.  New measurements need new questions, the
same rule the LightRAG comparison already learned.

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
from core.memory_index import MEMORY_CHUNK_PREFIX  # noqa: E402
from core.vector_index import is_excluded_source  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "storage" / "jev_fts5.db"

# (query, the file that must answer it).  An empty expectation means "no document
# should answer this at all".  An expectation may name several sources: a question
# can be answerable from the owner's own note *and* from the conversation that note
# summarises, and neither of them is a finding describing the query.
CONTROL_QUERIES: list[tuple[str, str | tuple[str, ...]]] = [
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
    # This one *is* answerable, and from two places since the chat import: the
    # board material is the summary, and the conversation it summarises is the
    # original.  Measured after the import: the conversation takes the gate at
    # 0.95 / 53 ms, which is the right material and not a leak - the guard fires
    # only on a *third* document trying to take it away from both.
    ("UPS HAT и Orange PI4 Pro возможно взаимодействие?", ("carpc", "Gemini")),
    ("Какая погода в Киеве завтра", ""),
    ("Купить билеты на поезд Киев Львов", ""),
]


def is_expected(chunk_id: str, expected: str | tuple[str, ...]) -> bool:
    """Whether this chunk is one of the sources allowed to answer the query."""
    if not expected:
        return False
    sources = (expected,) if isinstance(expected, str) else expected
    return any(source in chunk_id for source in sources)


def competitors(rows, query: str, expected: str | tuple[str, ...],
                namespaces: tuple[str, ...] = (MEMORY_CHUNK_PREFIX,),
                ) -> list[tuple[str, list[str]]]:
    """Indexed, authored chunks that would pass the gate for this query.

    `rows` is an iterable of (chunk_id, content, source).  Pure, so the rule is
    testable without a database.

    `namespaces` is which feeds the rule applies to, and the default is the memory
    feed alone: a *collected source* - the OCR'd manual, an imported conversation, a
    project's README or its commit messages - may contain any words, including the
    control queries themselves, because that material records what this project
    measured.  Only prose written about the system is held to the rule.
    """
    if not namespaces:
        return []
    keywords = _extract_keywords(query)
    content_keywords = [k for k in keywords if k.lower() not in FTS_GATE_STOPWORDS]
    if not content_keywords:
        return []
    threshold = min(FTS_GATE_MIN_TERMS, len(content_keywords))
    found = []
    for chunk_id, content, source in rows:
        if not str(chunk_id).startswith(namespaces):
            continue  # a collected source, not authored memory
        if is_excluded_source(str(source)):
            continue  # the raw corpus is expected to match ordinary words
        if is_expected(str(chunk_id), expected):
            continue  # the source that should answer is not a competitor
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
