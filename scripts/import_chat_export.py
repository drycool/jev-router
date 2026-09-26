#!/usr/bin/env python3
"""Import a Google Takeout Gemini export into Jev's corpus directory.

Why this exists: a question the owner had discussed in Gemini was answered with
the nearest unrelated neighbour, because the corpus held exactly ONE Gemini chat
- the only `Gemini-*.md` file that happened to be in `~/Downloads` when
`/home/dry/LightRag/extract_history.py` last ran.  The activity export held the
rest all along: 2969 items, 351 conversations, 6.7M characters of answers.

The export is an activity log, not a chat archive: every item carries the prompt
in `title` and the answer as HTML in `safeHtmlItem[].html`, and the conversation
it belongs to in `details[].url` (`.../app/<16 hex>`).  So a conversation can be
rebuilt from it, which is what this script does - one Markdown file per
conversation, in the same shape the memory documents use (`# title` + `## section`
blocks), because that is the shape the corpus chunker already knows how to split
into self-describing chunks.

The output directory is indexed at server start by the same code path that
indexes the memory directory (`_index_corpus_docs`), so an import survives a
restart.  Writing rows straight into the derived FTS5 table would not: it is
rebuilt from scratch on every start.

    python3 scripts/import_chat_export.py --list
    python3 scripts/import_chat_export.py --match x728,geekworm --dry-run
    python3 scripts/import_chat_export.py --conversation 1f8537fbad48f448
    python3 scripts/import_chat_export.py --all
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# One definition of where the corpus lives, taken from the module the server
# indexes with: an importer with its own default would write Markdown into a
# directory nothing reads, and the import would look like it worked.
from core.env import load_env  # noqa: E402

load_env()

from core.memory_index import CORPUS_DIR  # noqa: E402

DEFAULT_EXPORT = Path("/home/dry/LightRag/takeout_p1/Takeout/Мої дії/Додатки Gemini/MyActivity.json")
DEFAULT_OUT = Path(CORPUS_DIR)

# A conversation id is the 16 hex characters Gemini puts in the chat URL.
_CONVERSATION_ID = re.compile(r"/app/([0-9a-f]{16})")
# Google wraps every answer in markup; these tags carry the line structure.
_BLOCK_TAGS = re.compile(r"</(p|li|h[1-6]|ul|ol|div|tr|blockquote)>", re.IGNORECASE)
_BREAK_TAGS = re.compile(r"<br\s*/?>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")
# "1 вкладений файл." / "1 прикреплённый файл." - a count, not content.
_ATTACHMENT_NOISE = re.compile(r"^\s*\d+\s+(вкладених|вкладений|прикреплённ\w*|прикрепленн\w*|вложенн\w*)\s+файл\w*\.?\s*$",
                               re.IGNORECASE | re.MULTILINE)
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
SAFE_SLUG = 60


def strip_html(raw: str) -> str:
    """Turn an answer's HTML into plain text, keeping paragraph breaks.

    A pure function so the shape of an imported answer is testable without the
    export: the tags that matter for readability are the block ones, and the
    entities (`&quot;`, `&nbsp;`) must be decoded, or the corpus stores markup
    nobody can search for.
    """
    if not raw:
        return ""
    text = _BREAK_TAGS.sub("\n", raw)
    text = _BLOCK_TAGS.sub("\n", text)
    text = _ANY_TAG.sub("", text)
    text = html_lib.unescape(text)
    text = _ATTACHMENT_NOISE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANK_LINES.sub("\n\n", text).strip()


def conversation_id(item: dict) -> str:
    """The conversation an activity item belongs to, or "" when it names none."""
    for detail in item.get("details") or []:
        match = _CONVERSATION_ID.search(str(detail.get("url", "")))
        if match:
            return match.group(1)
    return ""


def item_question(item: dict) -> str:
    """The prompt. The export prefixes it with `Запит:` (and `Промпт:`)."""
    title = str(item.get("title", "")).strip()
    for prefix in ("Запит:", "Промпт:", "Prompt:"):
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    return title.strip()


def item_answer(item: dict) -> str:
    return strip_html(" ".join(str(part.get("html", "")) for part in item.get("safeHtmlItem") or []))


def group_by_conversation(items: list[dict]) -> dict[str, list[tuple[str, str, str]]]:
    """`{conversation_id: [(timestamp, question, answer), ...]}`, oldest first.

    Items without an id, without a question or without an answer are dropped: an
    activity log records every prompt, including the ones whose answer never
    arrived, and a prompt with no answer is not knowledge - it is a question the
    corpus would answer with itself.
    """
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for item in items:
        cid = conversation_id(item)
        question = item_question(item)
        answer = item_answer(item)
        if not cid or not question or len(answer) < 40:
            continue
        stamp = str(item.get("time", ""))[:19]
        grouped.setdefault(cid, []).append((stamp, question, answer))
    for turns in grouped.values():
        turns.sort(key=lambda turn: turn[0])
    return grouped


def conversation_slug(turns: list[tuple[str, str, str]]) -> str:
    """A file name that says which conversation this was, for provenance.

    Falls back to the first question because the export has no chat title.
    """
    opening = turns[0][1] if turns else "chat"
    slug = " ".join(opening.split())[:SAFE_SLUG].strip(" .-—")
    slug = unicodedata.normalize("NFC", slug)
    slug = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", slug)
    return slug or "chat"


def conversation_markdown(cid: str, turns: list[tuple[str, str, str]]) -> tuple[str, str]:
    """(filename, markdown) for one conversation.

    Each turn becomes one `##` section, so the corpus chunker emits a chunk per
    turn whose text opens with the chat and the date - a chunk retrieved alone
    has to say what it is about, and which conversation it came from.  The
    provenance line repeats in every turn rather than sitting in a preamble,
    because a preamble is also a chunk: metadata with no content is a slot spent
    on nothing, and it was the one chunk that could not name its question.
    """
    title = f"Gemini: {conversation_slug(turns)}"
    opened = turns[0][0][:10] if turns and turns[0][0] else "unknown-date"
    closed = turns[-1][0][:10] if turns and turns[-1][0] else opened
    span = opened if opened == closed else f"{opened}..{closed}"
    lines = [f"# {title}", ""]
    for stamp, question, answer in turns:
        moment = stamp[:16].replace("T", " ") or "?"
        head = " ".join(question.split())[:110]
        lines += [
            f"## {moment} — {head}", "",
            f"Источник: https://gemini.google.com/app/{cid} "
            f"(Gemini, {span}, пар в диалоге: {len(turns)})", "",
            "Вопрос:", question, "", "Ответ:", answer, "",
        ]
    filename = f"Gemini-{conversation_slug(turns)}-{opened.replace('-', '')}.md"
    return filename, "\n".join(lines)


def select_conversations(grouped: dict[str, list[tuple[str, str, str]]],
                         ids: list[str], matches: list[str]) -> dict[str, list[tuple[str, str, str]]]:
    """Apply `--conversation` and `--match` (case-insensitive substring).

    Both accept comma-separated values as well as repeated flags: `--match
    x728,geekworm` is what a user types, and an empty selection because the
    comma was taken literally would look like the corpus has nothing.
    """
    ids = [value.strip() for raw in ids for value in raw.split(",") if value.strip()]
    matches = [value.strip().lower() for raw in matches for value in raw.split(",") if value.strip()]
    selected = {cid: turns for cid, turns in grouped.items() if not ids or cid in ids}
    if matches:
        selected = {
            cid: turns for cid, turns in selected.items()
            if any(needle in " ".join(q + " " + a for _, q, a in turns).lower()
                   for needle in matches)
        }
    if ids:
        missing = [cid for cid in ids if cid not in grouped]
        for cid in missing:
            print(f"[import] conversation {cid} not found in the export", file=sys.stderr)
    return selected


def describe(cid: str, turns: list[tuple[str, str, str]]) -> str:
    chars = sum(len(a) for _, _, a in turns)
    return (f"{cid}  {len(turns):4d} пар  {chars:7,d} симв  "
            f"{turns[0][0][:10] if turns else '?'}  {conversation_slug(turns)[:62]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--conversation", action="append", default=[],
                        help="conversation id (repeatable)")
    parser.add_argument("--match", action="append", default=[],
                        help="case-insensitive substring of the conversation text (repeatable)")
    parser.add_argument("--all", action="store_true", help="every conversation in the export")
    parser.add_argument("--list", action="store_true", help="list the conversations and stop")
    parser.add_argument("--dry-run", action="store_true", help="report what would be written")
    args = parser.parse_args()

    if not args.export.exists():
        print(f"[import] export not found: {args.export}", file=sys.stderr)
        return 2
    items = json.loads(args.export.read_text(encoding="utf-8"))
    grouped = group_by_conversation(items)
    print(f"[import] {len(items)} activity items -> {len(grouped)} conversations with an answer; "
          f"{sum(len(t) for t in grouped.values()):,} question-answer pairs, "
          f"{sum(len(a) for t in grouped.values() for _, _, a in t):,} characters")

    if args.list:
        for cid, turns in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            print("   ", describe(cid, turns))
        return 0

    selected = grouped if args.all else select_conversations(grouped, args.conversation, args.match)
    if not selected:
        print("[import] nothing selected: pass --all, --conversation or --match", file=sys.stderr)
        return 2

    written = 0
    for cid, turns in sorted(selected.items()):
        filename, markdown = conversation_markdown(cid, turns)
        size = len(markdown)
        if args.dry_run:
            print(f"[import] would write {args.out / filename} ({len(turns)} turns, {size:,} chars)")
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / filename).write_text(markdown, encoding="utf-8")
        print(f"[import] wrote {filename} ({len(turns)} turns, {size:,} chars)")
        written += 1
    if args.dry_run:
        print(f"[import] dry run: {len(selected)} conversations would be written to {args.out}")
    else:
        print(f"[import] {written} conversations written to {args.out}; "
              f"start jev.service (or run scripts/index_corpus.py) to make them searchable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
