#!/usr/bin/env python3
"""Measure the "decisive" criterion for the vector tier against a fixed query set.

Why this exists: the vector tier decides whether local material is *decisive* - whether
the caller may treat it as the answer without a model.  That decision used to be one
absolute cosine number (JEV_SIMILARITY_THRESHOLD = 0.45), calibrated when the corpus was
209 rows.  The corpus is 5269 chunks now, the noise floor rose with it, and measured on
ordinary questions a wrong chunk clears 0.45 and gets labelled decisive: «чем кормить кота
зимой» returned RTC wiring with 1766 characters of citation-grade material.

An absolute number cannot express "this hit stands out", because what stands out depends
on how similar the corpus happens to be to this query *by chance*.  This script measures
that chance level per query: it draws a fixed random sample of corpus vectors, takes a
high quantile of the query's similarity to them, and prints the distance from the top hit
to that floor, alongside the verdict of the current rule and of candidate rules.

The query set is fixed and lives here: positives are questions the corpus can answer (they
must stay decisive), negatives are questions it cannot (they must be refused).  A criterion
that keeps every positive decisive and refuses every negative is the one to ship.

Usage:
    python3 scripts/measure_decisive_criterion.py            # table
    python3 scripts/measure_decisive_criterion.py --json     # machine readable
"""

from __future__ import annotations

import argparse
from typing import NamedTuple
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.env import load_env  # noqa: E402

load_env()

import numpy as np  # noqa: E402

from core.router import EMBEDDING_API, EMBEDDING_MODEL, SIMILARITY_THRESHOLD, VECTOR_DB_PATH  # noqa: E402
from core.vector_index import load_index  # noqa: E402

# The sample the floor is measured on.  Fixed seed and size, so two runs of this script
# compare the same thing: a moving floor would make the numbers uncomparable, which is
# the whole failure mode this instrument exists to catch.
NULL_SAMPLE_SEED = 20260926
NULL_SAMPLE_SIZE = 512
# The floor is taken at several quantiles rather than one: a high quantile asks "is the
# top hit better than the best of 512 random chunks", a lower one asks "is it better than
# the bulk of them".  Which question separates the two sets is a measurement, not a taste.
FLOOR_QUANTILES = (0.90, 0.95, 0.99)

# Questions the corpus answers.  Each one names a piece of material that is really in the
# base, so a refusal here is a false negative and a failure of the criterion.
POSITIVES = [
    "что решено про коммиты и рабочее дерево в фиде",
    "Geekworm X728 V2.5 подходит ли для автомобиля",
    "чем занимается проект aegis-core",
    "какая история решений в garageOS",
    "почему агент уходил в ручной поиск вместо базы",
    "как устроен контракт фида и манифест",
    "какой порог подобия стоит в Jev и почему",
]

# Questions nothing in this corpus can answer.  A decisive verdict here is a false
# positive: the caller would be handed unrelated material as if it were the answer.
NEGATIVES = [
    "чем кормить кота зимой и почему он спит на ноутбуке",
    "курс биткоина к гривне сегодня",
    "рецепт борща с говядиной",
    "какая погода завтра в Киеве",
    "напиши заявление в налоговую о переносе срока",
    "сколько стоит билет на поезд Киев Львов",
]


def embed(text: str) -> np.ndarray:
    body = json.dumps({"model": EMBEDDING_MODEL, "input": [text]}).encode()
    request = urllib.request.Request(EMBEDDING_API, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    vector = payload.get("embeddings") or payload.get("embedding")
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]
    return np.asarray(vector, dtype=np.float32)


def cosine(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    return (matrix @ vector) / np.maximum(matrix_norms * vector_norm, 1e-9)


def document_of(chunk_id: str) -> str:
    """The document a chunk belongs to: everything before the '#' turn number."""
    return chunk_id.split("#", 1)[0]


def words(text: str) -> set[str]:
    """Content words, crudely: enough to ask "does the chunk mention the question"."""
    import re
    stop = {"что", "как", "чем", "для", "или", "это", "его", "она", "они", "при", "про",
            "какой", "какая", "какие", "почему", "стоит", "подходит", "сегодня", "надо",
            "наши", "наших", "было", "были", "того", "чтобы", "ли", "же", "он", "у",
            "почему", "the", "and", "for", "with", "why", "does", "what"}
    return {w for w in re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐ0-9]{4,}", text.lower())
            if w not in stop}


class Verdict(NamedTuple):
    """What the table says, split by what the failure actually is.

    Two different things look the same in a raw count and call for different actions: a
    negative that clears the bar through the literal tier means the corpus *contains* the
    question's words (a quotation in a note or a commit message - the corpus cannot be
    fixed, only the question set can), while one that clears it through the vector tier
    means the criterion let noise through, which is the criterion's own failure.
    """

    ok: bool
    reason: str
    contaminated: tuple[str, ...]
    false_positives: tuple[str, ...]


def classify(positives: list[dict], negatives: list[dict]) -> Verdict:
    kept = [r for r in positives if r["decisive"]]
    missed = [r["query"] for r in positives if not r["decisive"]]
    contaminated = tuple(r["query"] for r in negatives
                         if r["decisive"] and r["strategy"] == "exact_fts")
    false_positives = tuple(r["query"] for r in negatives
                            if r["decisive"] and r["strategy"] != "exact_fts")
    if missed:
        return Verdict(False, f"критерий теряет настоящие ответы: {missed}",
                       contaminated, false_positives)
    if false_positives:
        return Verdict(False, "критерий пропускает шум векторным тиром: "
                              f"{list(false_positives)} - смотреть таблицу",
                       contaminated, false_positives)
    if contaminated:
        return Verdict(True, "критерий разделяет по векторному тиру; "
                             f"{len(contaminated)} негатив(ов) отвечает буквальный тир - "
                             "это цитата в корпусе, а не критерий; набор вопросов нужен новый",
                       contaminated, false_positives)
    return Verdict(True, f"критерий разделяет: {len(kept)}/{len(positives)} позитивов, "
                         f"0/{len(negatives)} негативов", contaminated, false_positives)


def measure(queries: list[str], floor_quantiles: tuple[float, ...], sample_size: int) -> list[dict]:
    index = load_index(VECTOR_DB_PATH)
    embeddings = index["embeddings"]
    contents = index["contents"]
    chunk_ids = index["chunk_ids"]
    rng = np.random.default_rng(NULL_SAMPLE_SEED)
    sample = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)

    rows = []
    for query in queries:
        scores = cosine(embed(query), embeddings)
        order = np.argsort(-scores)
        top = order[:10]
        top_document = document_of(chunk_ids[top[0]])
        # How much of the pool agrees with the top hit.  One stray chunk that happens to
        # be close is noise; a body of material about the same thing is an answer.
        agreement = sum(1 for i in top if document_of(chunk_ids[i]) == top_document)
        query_words = words(query)
        # Lexical support: the question's own words in the material that would be served.
        lexical = max(
            (len(query_words & words(contents[i])) for i in top[:5]), default=0)
        row = {
            "query": query,
            "top1": float(scores[top[0]]),
            "top2": float(scores[top[1]]) if len(top) > 1 else 0.0,
            "gap": float(scores[top[0]] - scores[top[1]]) if len(top) > 1 else 0.0,
            "agreement": agreement,
            "lexical": lexical,
            "lexical_max": len(query_words),
            "current": bool(scores[top[0]] >= SIMILARITY_THRESHOLD),
            "chunk": chunk_ids[top[0]],
        }
        for q in floor_quantiles:
            row[f"floor{q}"] = float(np.quantile(scores[sample], q))
            # The same question asked of the whole corpus instead of a sample.  The sample
            # keeps genuine hits out of the noise estimate; the whole corpus mixes them in,
            # which for a well-represented topic raises the floor.  Which is the better
            # noise estimate is a measurement, so both are measured.
            row[f"full{q}"] = float(np.quantile(scores, q))
        row["mean"] = float(scores.mean())
        row["std"] = float(scores.std())
        row["z"] = (row["top1"] - row["mean"]) / max(row["std"], 1e-9)
        rows.append(row)
    return rows


def calibrate(positives: list[dict], negatives: list[dict], quantile: float) -> list[tuple]:
    """Every (absolute bar, relative margin) pair, scored by what it does to both sets."""
    cells = []
    for bar in (0.45, 0.50, 0.55, 0.60):
        for margin in (0.04, 0.06, 0.08, 0.10, 0.12):
            def decisive(row: dict) -> bool:
                # Two independent conditions, because neither separates these classes on
                # its own: the absolute bar refuses anything weak wherever it came from,
                # and the margin refuses a hit that is no better than this query's own
                # chance level against the corpus.
                return (row["top1"] >= SIMILARITY_THRESHOLD
                        and row["top1"] >= max(bar, row[f"floor{quantile}"] + margin))
            kept = sum(1 for r in positives if decisive(r))
            wrong = sum(1 for r in negatives if decisive(r))
            cells.append((wrong, -kept, bar, margin))
    return sorted(cells)


def via_api(queries: list[str], base: str) -> list[dict]:
    """Ask the live router, so the instrument measures what is actually served.

    The archive-side measurement above says what the numbers are; this says what the
    router does with them, including the label and the floor it reports.  Both are needed:
    a criterion that is calibrated but not wired up looks identical in the first table.
    """
    rows = []
    for query in queries:
        body = json.dumps({"query": query, "execute": False}).encode()
        request = urllib.request.Request(f"{base}/query", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        decision = payload.get("routing_decision") or {}
        rows.append({
            "query": query,
            "strategy": decision.get("strategy"),
            "decisive": bool(decision.get("local_material_decisive")),
            "floor": decision.get("decisive_floor"),
            "top1": decision.get("confidence_score"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--sample", type=int, default=NULL_SAMPLE_SIZE)
    parser.add_argument("--via-api", action="store_true",
                        help="ask the live router instead of measuring the archive")
    parser.add_argument("--base", default="http://127.0.0.1:8030")
    args = parser.parse_args()

    quantiles = FLOOR_QUANTILES
    index = load_index(VECTOR_DB_PATH)

    if args.via_api:
        positives = via_api(POSITIVES, args.base)
        negatives = via_api(NEGATIVES, args.base)
        if args.json:
            print(json.dumps({"positives": positives, "negatives": negatives},
                             ensure_ascii=False, indent=2))
            return 0
        print(f"через живой роутер {args.base} (execute=false):\n")
        for label, rows in (("ПОЗИТИВЫ (должны быть решающими)", positives),
                            ("НЕГАТИВЫ (должны быть не решающими)", negatives)):
            print(label)
            for row in rows:
                floor = f"{row['floor']:.4f}" if row["floor"] is not None else "  —   "
                print(f"  {row['top1'] or 0:.4f} пол {floor} "
                      f"{'РЕШАЮЩИЙ' if row['decisive'] else 'не решающий':>11} "
                      f"{row['strategy']:<20} {row['query'][:40]}")
            verdict = sum(1 for r in rows if r["decisive"])
            # Which tier answered, counted rather than described.  This set cannot be
            # measured twice: an `exact_fts` verdict means the corpus contains the
            # question's own words, and after a round of calibration that is expected -
            # commit messages and notes quote the measured queries.  A table of "7 of 7
            # decisive" would hide that the literal tier did the work, and the vector
            # criterion would get credit it did not earn.
            tiers: dict = {}
            for row in rows:
                if row["decisive"]:
                    tiers[row["strategy"]] = tiers.get(row["strategy"], 0) + 1
            print(f"  решающих: {verdict} из {len(rows)}"
                  + (f"  (по тирам: {tiers})" if tiers else "") + "\n")
        verdict = classify(positives, negatives)
        print("итог:", verdict.reason)
        expected = verdict.ok
        print("напоминание: контрольные запросы нельзя цитировать в индексируемом тексте - "
              "заметки и сообщения коммитов попадают в корпус, и тогда буквальный тир "
              "отвечает на них сам. Для повторной калибровки нужен новый набор вопросов.")
        return 0 if expected else 1

    positives = measure(POSITIVES, quantiles, args.sample)
    negatives = measure(NEGATIVES, quantiles, args.sample)

    if args.json:
        print(json.dumps({"positives": positives, "negatives": negatives,
                          "threshold": SIMILARITY_THRESHOLD, "sample": args.sample},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"корпус: {len(index['embeddings'])} векторов, модель {index['model']}, "
          f"выборка пола: {args.sample} векторов с фиксированным зерном")
    print(f"абсолютный порог сейчас: {SIMILARITY_THRESHOLD}\n")
    for label, rows in (("ПОЗИТИВЫ (материал в базе есть)", positives),
                        ("НЕГАТИВЫ (материала нет)", negatives)):
        print(f"{label}")
        heads = "".join(f"{'пол' + str(q):>9}" for q in quantiles)
        print(f"{'топ-1':>7}{heads}{'запас99':>9}{'согл':>6}{'лекс':>7}{'сейчас':>11}   запрос")
        for row in rows:
            floors = "".join(f"{row[f'floor{q}']:>9.4f}" for q in quantiles)
            verdict = "решающий" if row["current"] else "отвергнут"
            print(f"{row['top1']:>7.4f}{floors}{row['top1'] - row['floor0.99']:>9.4f}"
                  f"{row['agreement']:>6}{row['lexical']:>4}/{row['lexical_max']:<3}"
                  f"{verdict:>11}   {row['query'][:40]}")
        print(f"  топ-1: {min(r['top1'] for r in rows):.4f} … {max(r['top1'] for r in rows):.4f}"
              f" | решающих сейчас: {sum(1 for r in rows if r['current'])} из {len(rows)}\n")

    # The calibration: which pair of conditions keeps the positives and refuses all the
    # negatives.  Sorted by (false positives, positives kept) - a false positive costs
    # more than a demoted positive, because a demoted one still serves its material with
    # "judge it yourself" while a false decisive tells the consumer it is the answer.
    print("КАЛИБРОВКА: (абсолютная планка, запас над полом) → решающих у негативов / у позитивов")
    for q in quantiles:
        cells = calibrate(positives, negatives, q)
        best = [c for c in cells if c[0] == 0]
        line = "  ".join(f"({bar}/{margin})=>{-kept}из{len(positives)}"
                         for wrong, kept, bar, margin in (best[:5] or cells[:5]))
        print(f"  пол по квантилю {q}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
