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

There are now two such directories - the working-memory documents and the
imported chat corpus (``scripts/import_chat_export.py`` writes it) - and both go
through ``index_directory``.  They differ only in their chunk-id namespace
(``mem:`` / ``gem:``) and in the tag their rows carry.  The namespace is the
row's identity: ``replace_rows`` deletes by it, so re-indexing one directory can
never delete the other's rows, and ``core.vector_index`` recognises memory rows
by the same ``mem:`` prefix.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import NamedTuple

# Read once at import, like the router's own settings; the indexers accept an
# explicit root so callers (and tests) are never forced to touch the env.
DEFAULT_MEMORY_DIR = "/home/dry/memory"
MEMORY_DIR = os.getenv("JEV_MEMORY_DIR", DEFAULT_MEMORY_DIR)
# An answer that lives only in an export is not findable, and the discussion
# about a UPS HAT existed only in the export: 351 conversations, one of them in
# the corpus.  This directory is what the importer writes and the server indexes.
DEFAULT_CHATS_DIR = "/home/dry/LightRag/feeds/chats"
CORPUS_DIR = os.getenv("JEV_CORPUS_DIR", DEFAULT_CHATS_DIR)
# Written by `jev-collect` (Go): the documentation and commit history of the
# repositories on this machine, and later GitHub and the agent's own sessions.
DEFAULT_PROJECTS_DIR = "/home/dry/LightRag/feeds/projects"
# Also `jev-collect`, a different source: what exists only on GitHub - the README
# and commit history of the repositories with no local clone here, plus issues,
# pull requests and releases, which a clone never carries.  A repository that does
# have a clone keeps its README and history in `projects` only: one text under two
# identities dilutes retrieval, because the two chunks compete and the winner is a
# coin toss.
DEFAULT_GITHUB_DIR = "/home/dry/LightRag/feeds/github"
ENTITY_TYPE = "memory"
ENTITY_TYPE_CORPUS = ""
MEMORY_CHUNK_PREFIX = "mem:"
CORPUS_CHUNK_PREFIX = "gem:"
PROJECT_CHUNK_PREFIX = "prj:"
GITHUB_CHUNK_PREFIX = "gh:"
# The server indexes LightRAG chunks at 2000 characters; staying under that keeps
# one document from being penalised relative to another by the BM25 length norm.
DEFAULT_MAX_CHARS = 1800

# Every directory the service indexes, in one place.  A feed is (name, env var,
# default directory, chunk-id namespace, row tag); the namespace is the row's
# identity, so a rebuild of one feed can never delete another's rows, and adding
# a source means adding a line here rather than a new code path in the server.
FEED_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    ("memory", "JEV_MEMORY_DIR", DEFAULT_MEMORY_DIR,
     MEMORY_CHUNK_PREFIX, ENTITY_TYPE),
    ("chats", "JEV_CORPUS_DIR", DEFAULT_CHATS_DIR,
     CORPUS_CHUNK_PREFIX, ENTITY_TYPE_CORPUS),
    ("projects", "JEV_PROJECTS_DIR", DEFAULT_PROJECTS_DIR,
     PROJECT_CHUNK_PREFIX, ENTITY_TYPE_CORPUS),
    ("github", "JEV_GITHUB_DIR", DEFAULT_GITHUB_DIR,
     GITHUB_CHUNK_PREFIX, ENTITY_TYPE_CORPUS),
)


class Feed(NamedTuple):
    """One indexed directory: where it is, and how its rows are identified."""

    name: str
    prefix: str
    directory: str
    entity_type: str

    @property
    def env_var(self) -> str:
        for name, env_var, *_ in FEED_SPECS:
            if name == self.name:
                return env_var
        return ""


def feeds() -> list[Feed]:
    """The registry, with directories read from the environment *now*.

    Read per call rather than at import: the service reads it once at start, and
    a test that patches a directory must not have to reload the module to be
    believed.
    """
    return [Feed(name=name, prefix=prefix, entity_type=entity_type,
                 directory=os.getenv(env_var, default))
            for name, env_var, default, prefix, entity_type in FEED_SPECS]


def feed(name: str) -> Feed:
    for candidate in feeds():
        if candidate.name == name:
            return candidate
    known = ", ".join(candidate.name for candidate in feeds())
    raise ValueError(f"unknown feed {name!r}; known feeds: {known}")


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


def build_chunks(root: Path, max_chars: int = DEFAULT_MAX_CHARS,
                 prefix: str = MEMORY_CHUNK_PREFIX) -> list[tuple[str, str, str]]:
    """Return (chunk_id, content, source) for every chunk under root.

    chunk_id is ``<prefix><path relative to root>#<n>`` - stable across runs, so
    a re-index replaces a document rather than accumulating copies of it.
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
                chunks.append((f"{prefix}{rel}#{index}", content, str(path)))
                index += 1
    return chunks


def replace_rows(connection: sqlite3.Connection, chunks: list[tuple[str, str, str]],
                 prefix: str = MEMORY_CHUNK_PREFIX,
                 entity_type: str = ENTITY_TYPE) -> tuple[int, int]:
    """Delete previous rows in this namespace and insert these, in one transaction.

    Idempotent by construction: a second run replaces its own output instead of
    doubling every document.

    Deletion is by chunk-id prefix, not by tag.  The tag is a ranking hint the
    memory rows share with nothing but each other, while an untagged corpus row
    and an untagged memory row would be indistinguishable - a rebuild that
    deleted by tag would either take the corpus with it or leave duplicates.
    """
    with connection:  # one transaction, so readers never see a half-built index
        removed = connection.execute(
            "DELETE FROM chunks WHERE chunk_id LIKE ?", (f"{prefix}%",)
        ).rowcount
        connection.executemany(
            "INSERT INTO chunks (chunk_id, content, source, entity_type) VALUES (?, ?, ?, ?)",
            [(cid, content, source, entity_type) for cid, content, source in chunks],
        )
    return removed, len(chunks)


def index_directory(connection: sqlite3.Connection, root: str | Path,
                    prefix: str = MEMORY_CHUNK_PREFIX, entity_type: str = ENTITY_TYPE,
                    max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Index one directory; returns statistics, raises on a real failure.

    A missing directory is not a failure: the router must start even on a machine
    where the directory has not been created yet.
    """
    root = Path(root)
    if not root.is_dir():
        return {"root": str(root), "exists": False, "files": 0, "chunks": 0,
                "chars": 0, "removed": 0, "inserted": 0}
    chunks = build_chunks(root, max_chars, prefix)
    removed, inserted = replace_rows(connection, chunks, prefix, entity_type)
    return {
        "root": str(root),
        "exists": True,
        "files": len({source for _, _, source in chunks}),
        "chunks": len(chunks),
        "chars": sum(len(content) for _, content, _ in chunks),
        "removed": removed,
        "inserted": inserted,
    }


def index_feed(connection: sqlite3.Connection, name: str,
               root: str | Path | None = None,
               max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Index one registered feed.  The service's single entry point.

    Every feed goes through this: the id namespace comes from the registry, so a
    feed that is added later cannot be indexed under someone else's namespace by
    a caller that forgot to pass one.
    """
    spec = feed(name)
    directory = Path(root) if root is not None else Path(spec.directory)
    stats = index_directory(connection, directory, spec.prefix, spec.entity_type, max_chars)
    stats["feed"] = spec.name
    stats["prefix"] = spec.prefix
    return stats


def index_memory(connection: sqlite3.Connection, root: str | Path | None = None,
                 max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """The working-memory directory: tagged rows, ``mem:`` ids."""
    return index_directory(connection, Path(root or MEMORY_DIR), MEMORY_CHUNK_PREFIX,
                           ENTITY_TYPE, max_chars)


def index_corpus(connection: sqlite3.Connection, root: str | Path | None = None,
                 max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """The imported chat corpus: untagged rows, ``gem:`` ids.

    Untagged on purpose - ``entity_type='memory'`` is the input to the FTS5
    memory boost, and an imported conversation is not the owner's own note.  It
    is still reachable by the same search, which is the whole point of indexing
    it into the same table.
    """
    return index_directory(connection, Path(root or CORPUS_DIR), CORPUS_CHUNK_PREFIX,
                           ENTITY_TYPE_CORPUS, max_chars)


def index_projects(connection: sqlite3.Connection, root: str | Path | None = None,
                   max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Collected project documentation and history: untagged rows, ``prj:`` ids.

    Untagged like the chat corpus, and for the same reason: a project's README is
    material, not the owner's own note, and the memory boost is for the notes.
    """
    return index_directory(connection, Path(root or feed("projects").directory),
                           PROJECT_CHUNK_PREFIX, ENTITY_TYPE_CORPUS, max_chars)
