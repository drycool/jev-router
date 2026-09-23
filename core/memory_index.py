"""Index a directory of Markdown documents into Jev's Tier-2 FTS5 index.

One implementation, two callers that must not drift apart:

  * ``api/server.py::_index_memory_docs()`` runs it on every start, so the
    working-memory documents survive a restart of the service;
  * ``scripts/index_memory.py`` runs it from the command line, with a dry-run
    and per-file statistics.

The table it writes to (``chunks``) is the router's own FTS5 virtual table,
created by ``Tier2Search._init_fts5``.  This module issues the same INSERT the
router issues and nothing else: no routing, no gating, no search code.  Rows are
tagged ``entity_type='memory'`` and carry their absolute path in ``source``, so
they are identifiable and removable while sharing one BM25 index with the
corpus.

The chunk prefix ``"<document title> :: <section>"`` is load-bearing.  A chunk
retrieved on its own has to say what it is about; the probe showed the delivered
context for the systemd question starting with
``"Деплой демонов: systemd --user :: Установка юнита"``.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

# Read once at import, like the router's own settings; ``index_memory`` accepts
# an explicit root so callers (and tests) are never forced to touch the env.
MEMORY_DIR = os.getenv("JEV_MEMORY_DIR", "/home/dry/memory")
ENTITY_TYPE = "memory"
# The server indexes LightRAG chunks at 2000 characters; staying under that keeps
# one document from being penalised relative to another by the BM25 length norm.
DEFAULT_MAX_CHARS = 1800


def iter_markdown(root: Path) -> list[Path]:
    """Every .md under root, sorted, skipping index scratch directories."""
    return sorted(p for p in root.rglob("*.md") if ".index" not in p.parts)


def document_title(text: str, fallback: str) -> str:
    """The level-1 heading, or the file's stem."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs, split on level-2 headings."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading or body:
                sections.append((heading, "\n".join(body).strip()))
            heading, body = line[3:].strip(), []
        elif line.startswith("# ") and not heading and not body:
            continue  # the document title is added to every chunk separately
        else:
            body.append(line)
    if heading or body:
        sections.append((heading, "\n".join(body).strip()))
    return [(h, b) for h, b in sections if b]


def split_too_long(block: str, max_chars: int) -> list[str]:
    """Split a block on blank lines so no part exceeds max_chars."""
    if len(block) <= max_chars:
        return [block]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in re.split(r"\n\s*\n", block):
        if current and size + len(paragraph) + 2 > max_chars:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts


def build_chunks(root: Path, max_chars: int = DEFAULT_MAX_CHARS) -> list[tuple[str, str, str]]:
    """Return (chunk_id, content, source) for every chunk under root.

    chunk_id is ``mem:<path relative to root>#<n>`` - stable across runs, so a
    re-index replaces a document rather than accumulating copies of it.
    """
    chunks: list[tuple[str, str, str]] = []
    for path in iter_markdown(root):
        text = path.read_text(encoding="utf-8")
        title = document_title(text, path.stem)
        rel = path.relative_to(root)
        index = 0
        for heading, body in split_sections(text):
            header = f"{title} :: {heading}" if heading else title
            for part in split_too_long(body, max_chars - len(header) - 4):
                content = f"{header}\n\n{part}"[:max_chars]
                chunks.append((f"mem:{rel}#{index}", content, str(path)))
                index += 1
    return chunks


def replace_memory_rows(connection: sqlite3.Connection, chunks: list[tuple[str, str, str]],
                        entity_type: str = ENTITY_TYPE) -> tuple[int, int]:
    """Delete previous memory rows and insert these, in one transaction.

    Idempotent by construction: a second run replaces its own output instead of
    doubling every document.
    """
    with connection:  # one transaction, so readers never see a half-built index
        removed = connection.execute(
            "DELETE FROM chunks WHERE entity_type = ?", (entity_type,)
        ).rowcount
        connection.executemany(
            "INSERT INTO chunks (chunk_id, content, source, entity_type) VALUES (?, ?, ?, ?)",
            [(cid, content, source, entity_type) for cid, content, source in chunks],
        )
    return removed, len(chunks)


def index_memory(connection: sqlite3.Connection, root: str | Path | None = None,
                 max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Index the memory directory; returns statistics, raises on a real failure.

    A missing directory is not a failure: the router must start even on a machine
    where the memory directory has not been created yet.
    """
    root = Path(root or MEMORY_DIR)
    if not root.is_dir():
        return {"root": str(root), "exists": False, "files": 0, "chunks": 0,
                "chars": 0, "removed": 0, "inserted": 0}
    chunks = build_chunks(root, max_chars)
    removed, inserted = replace_memory_rows(connection, chunks)
    return {
        "root": str(root),
        "exists": True,
        "files": len({source for _, _, source in chunks}),
        "chunks": len(chunks),
        "chars": sum(len(content) for _, content, _ in chunks),
        "removed": removed,
        "inserted": inserted,
    }
