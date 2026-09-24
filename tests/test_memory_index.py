"""Tests for indexing the working-memory directory into Tier-2 FTS5.

Written against the measured failure behind the feature.  The memory documents
were indexed by an operator-run script, and `systemctl --user restart jev` took
the index from 4593 rows to 4562 - the FTS5 table is *derived* and the server
rebuilds it from the LightRAG chunk store on every start.  So the two facts worth
protecting are: a restart re-indexes the directory by itself, and re-indexing
replaces its own rows instead of multiplying them.

The pipeline tests run against a real FTS5 table (a temporary database, same
virtual table the router creates), because a test that only checks a returned
list proves nothing about what the index actually contains afterwards.
"""
import asyncio
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.memory_index import (
    DEFAULT_MAX_CHARS,
    ENTITY_TYPE,
    build_chunks,
    index_memory,
)
from core.router import Tier2Search

DOCUMENT = """# Правила окружения

Вступление.

## Установка юнита

```bash
cp deploy/jev.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

## Почему не tmux

Перезагрузка убила сервис и ничто не сообщило об этом.
"""


def _write(root: Path, name: str, text: str = DOCUMENT) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class MemoryChunkingTests(unittest.TestCase):
    def test_every_chunk_says_which_document_and_section_it_came_from(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "global/linux_systemd.md")
            chunks = build_chunks(root)

            self.assertTrue(chunks)
            for _, content, _ in chunks:
                self.assertTrue(
                    content.startswith("Правила окружения"),
                    f"chunk does not carry its document title: {content[:60]!r}",
                )
            headings = [content.split("\n")[0] for _, content, _ in chunks]
            self.assertIn("Правила окружения :: Установка юнита", headings)
            self.assertIn("Правила окружения :: Почему не tmux", headings)

    def test_chunk_ids_name_the_file_and_are_stable_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "projects/jev_gateway.md")

            first = build_chunks(root)
            second = build_chunks(root)

            self.assertEqual(
                [chunk_id for chunk_id, _, _ in first],
                [chunk_id for chunk_id, _, _ in second],
                "re-indexing must reproduce the same ids, or it accumulates copies",
            )
            self.assertTrue(first[0][0].startswith("mem:projects/jev_gateway.md#"))
            self.assertEqual(first[0][2], str(root / "projects" / "jev_gateway.md"))

    def test_a_long_section_is_split_and_every_part_respects_the_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = "\n\n".join(f"Пункт {i} " + "текст " * 30 for i in range(20))
            _write(root, "long.md", f"# Длинный\n\n## Раздел\n\n{body}\n")

            chunks = build_chunks(root, max_chars=800)

            self.assertGreater(len(chunks), 1, "a long section must not be one oversized chunk")
            for _, content, _ in chunks:
                self.assertLessEqual(len(content), 800)
                self.assertTrue(content.startswith("Длинный :: Раздел"))

    def test_the_default_cap_stays_under_the_corpus_indexing_cap(self):
        # The server indexes LightRAG chunks at content[:2000]; a memory chunk
        # larger than that would be the only document type BM25 penalises for
        # length alone.
        self.assertLess(DEFAULT_MAX_CHARS, 2000)

    def test_the_directory_is_configurable_and_defaults_to_the_memory_root(self):
        module = importlib.import_module("core.memory_index")
        # Taken before the patch, so it is what a reload after the patch must
        # restore - on this machine whatever JEV_MEMORY_DIR says, else the root.
        default = os.getenv("JEV_MEMORY_DIR", "/home/dry/memory")
        with patch.dict(os.environ, {"JEV_MEMORY_DIR": "/tmp/some-other-memory"}):
            self.assertEqual(importlib.reload(module).MEMORY_DIR, "/tmp/some-other-memory")
        # Restore outside the patch: reloading inside it re-reads the patched value.
        self.assertEqual(importlib.reload(module).MEMORY_DIR, default)


class MemoryIndexingTests(unittest.TestCase):
    def _search(self, directory: str) -> Tier2Search:
        with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
            return Tier2Search()

    def test_reindexing_replaces_its_own_rows_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            _write(root, "global/linux_systemd.md")
            search = self._search(directory)
            try:
                first = index_memory(search.conn, root)
                second = index_memory(search.conn, root)

                self.assertEqual(first["inserted"], second["inserted"])
                self.assertEqual(first["removed"], 0, "nothing to replace on the first run")
                self.assertEqual(second["removed"], first["inserted"],
                                 "the second run must replace the first run's rows")
                total = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE entity_type = ?", (ENTITY_TYPE,)
                ).fetchone()[0]
                self.assertEqual(total, first["inserted"])
            finally:
                search.close()

    def test_memory_rows_are_findable_and_identifiable_apart_from_the_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            _write(root, "global/linux_systemd.md")
            search = self._search(directory)
            try:
                search.index_chunk("corpus-1", "настраиваются пользовательские сервисы", "d_espero.pdf")
                search.commit()
                index_memory(search.conn, root)

                results = search.search_fts5("пользовательские сервисы systemd", limit=20)
                memory = [r for r in results if r["entity_type"] == ENTITY_TYPE]
                self.assertTrue(memory, "the indexed document is not retrievable")
                self.assertTrue(memory[0]["source"].endswith("linux_systemd.md"))
            finally:
                search.close()

    def test_a_missing_directory_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            search = self._search(directory)
            try:
                stats = index_memory(search.conn, Path(directory) / "does-not-exist")

                self.assertFalse(stats["exists"])
                self.assertEqual(stats["inserted"], 0)
                self.assertEqual(
                    search.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 0
                )
            finally:
                search.close()

    def test_the_root_comes_from_the_module_constant_when_not_given(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "memory"
            _write(root, "global/linux_systemd.md")
            search = self._search(directory)
            try:
                with patch("core.memory_index.MEMORY_DIR", str(root)):
                    stats = index_memory(search.conn)
                self.assertTrue(stats["exists"])
                self.assertEqual(stats["root"], str(root))
                self.assertEqual(stats["files"], 1)
            finally:
                search.close()


class ServerWiringTests(unittest.TestCase):
    """The order in the lifespan is the whole point, so it is asserted, not assumed."""

    def _source(self) -> str:
        return (Path(__file__).resolve().parents[1] / "api" / "server.py").read_text(encoding="utf-8")

    def test_the_server_indexes_memory_after_the_lightrag_rebuild(self):
        source = self._source()
        lightrag_call = source.index("await _index_lightrag_chunks()")
        memory_call = source.index("await _index_memory_docs()")
        self.assertLess(
            lightrag_call, memory_call,
            "the LightRAG rebuild starts with clear(); indexing memory before it "
            "would delete the memory rows on every start",
        )

    def test_the_server_has_the_function_it_calls(self):
        source = self._source()
        self.assertIn("async def _index_memory_docs():", source)
        self.assertIn("from core.memory_index import", source)

    def test_the_embedder_is_warmed_without_being_awaited(self):
        """The warm-up has to happen, and has to happen off the request path.

        Awaiting it would make startup depend on GPU2 being awake - the rule that
        keeps the memory vectors an explicit build step - and a 30 s timeout there
        would hold the port closed while the health check reports nothing.
        """
        source = self._source()
        memory_call = source.index("await _index_memory_docs()")
        warm_call = source.index("asyncio.create_task(_warm_embedder())")
        self.assertLess(memory_call, warm_call)
        self.assertNotIn("await _warm_embedder()", source,
                         "the warm-up must not be awaited in the lifespan")

    def test_a_failing_warm_up_does_not_break_startup(self):
        """GPU2 asleep is a normal state here, not a start-up failure."""
        import api.server as server

        with patch.object(server, "EMBEDDING_WARMUP_TIMEOUT_S", 0.2), \
             patch.object(server, "EMBEDDING_API", "http://127.0.0.1:9/api/embed"):
            # Port 9 is discard: the connection is refused rather than hanging.
            asyncio.run(server._warm_embedder())


if __name__ == "__main__":
    unittest.main()
