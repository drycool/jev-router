"""Tests for the feed registry: one code path for every indexed directory.

Written against what the registry is for.  The corpus arrived in three separate
steps - memory documents, an imported chat export, collected project material -
and each step first came with its own function in the server, its own delete rule
and its own way of being forgotten on restart.  The registry exists so the next
source is a line in `FEED_SPECS` rather than a fourth code path, and the facts
worth protecting are:

  * a row is identified by its chunk-id namespace, so re-indexing one feed cannot
    delete another's rows - the memory documents, the chats and the project
    material share one table;
  * every feed the service indexes is in the registry, and every registered feed
    is indexed on start (asserted in the wiring tests too, from the other side);
  * a feed directory that does not exist yet is not an error, because the router
    has to start on a machine where nothing has been collected.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.memory_index import (
    DEFAULT_PROJECTS_DIR,
    ENTITY_TYPE,
    ENTITY_TYPE_CORPUS,
    FEED_SPECS,
    MEMORY_CHUNK_PREFIX,
    PROJECT_CHUNK_PREFIX,
    build_chunks,
    feed,
    feeds,
    index_feed,
)
from core.router import Tier2Search
from core.vector_index import merge_namespace

PROJECT_DOCUMENT = """# Jev: docs/feeds

## Контракт

Сборщик пишет Markdown и манифест; индексация остаётся в Jev.
"""

MEMORY_DOCUMENT = """# Деплой демонов

## Установка юнита

`systemctl --user enable --now jev.service`
"""


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class RegistryTests(unittest.TestCase):
    def test_every_feed_has_its_own_namespace(self):
        prefixes = [spec.prefix for spec in feeds()]
        self.assertEqual(len(prefixes), len(set(prefixes)),
                         "two feeds sharing an id namespace would overwrite each other")
        self.assertIn(MEMORY_CHUNK_PREFIX, prefixes)
        self.assertIn(PROJECT_CHUNK_PREFIX, prefixes)

    def test_the_projects_feed_points_at_the_collector_output(self):
        """`jev-collect --out` has to be this directory, or the collection lands
        where the indexer never looks."""
        self.assertEqual(feed("projects").directory, DEFAULT_PROJECTS_DIR)
        self.assertEqual(feed("projects").prefix, PROJECT_CHUNK_PREFIX)
        self.assertEqual(feed("projects").entity_type, ENTITY_TYPE_CORPUS,
                         "a project's README is material, not the owner's own note")

    def test_an_unknown_feed_names_the_known_ones(self):
        with self.assertRaises(ValueError) as caught:
            feed("github")
        self.assertIn("projects", str(caught.exception))

    def test_a_directory_can_be_moved_without_reloading_the_module(self):
        """The registry is read per call: a test (or an operator) that points a
        feed elsewhere must be believed immediately."""
        with patch.dict("os.environ", {"JEV_PROJECTS_DIR": "/tmp/other-projects"}):
            self.assertEqual(feed("projects").directory, "/tmp/other-projects")
        self.assertEqual(feed("projects").directory, DEFAULT_PROJECTS_DIR)

    def test_the_registry_is_the_single_definition(self):
        names = [name for name, *_ in FEED_SPECS]
        self.assertEqual(names, [spec.name for spec in feeds()])
        for name, env_var, default, prefix, entity_type in FEED_SPECS:
            self.assertTrue(env_var.startswith("JEV_"))
            self.assertTrue(default.startswith("/"))
            self.assertTrue(prefix.endswith(":"))
            self.assertIn(entity_type, (ENTITY_TYPE, ENTITY_TYPE_CORPUS))


class FeedIndexingTests(unittest.TestCase):
    def _search(self, directory: str) -> Tier2Search:
        with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
            return Tier2Search()

    def test_indexing_a_feed_uses_the_registry_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "projects"
            _write(root, "Jev__docs__feeds.md", PROJECT_DOCUMENT)
            search = self._search(directory)
            try:
                stats = index_feed(search.conn, "projects", root)
                self.assertTrue(stats["exists"])
                self.assertEqual(stats["prefix"], PROJECT_CHUNK_PREFIX)
                rows = search.conn.execute(
                    "SELECT chunk_id, entity_type FROM chunks"
                ).fetchall()
                self.assertTrue(rows)
                for chunk_id, entity_type in rows:
                    self.assertTrue(chunk_id.startswith(PROJECT_CHUNK_PREFIX),
                                    f"row in the wrong namespace: {chunk_id}")
                    self.assertEqual(entity_type, ENTITY_TYPE_CORPUS)
            finally:
                search.close()

    def test_reindexing_one_feed_leaves_the_others_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            projects = Path(directory) / "projects"
            memory = Path(directory) / "memory"
            _write(projects, "a.md", PROJECT_DOCUMENT)
            _write(memory, "global/x.md", MEMORY_DOCUMENT)
            search = self._search(directory)
            try:
                index_feed(search.conn, "projects", projects)
                index_feed(search.conn, "memory", memory)
                before = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE ?",
                    (f"{MEMORY_CHUNK_PREFIX}%",),
                ).fetchone()[0]
                self.assertTrue(before)

                index_feed(search.conn, "projects", projects)
                index_feed(search.conn, "projects", projects)
                after = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE ?",
                    (f"{MEMORY_CHUNK_PREFIX}%",),
                ).fetchone()[0]
                self.assertEqual(before, after, "re-indexing projects ate the memory rows")
            finally:
                search.close()

    def test_a_feed_directory_that_does_not_exist_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            search = self._search(directory)
            try:
                stats = index_feed(search.conn, "projects", Path(directory) / "нет")
                self.assertFalse(stats["exists"])
                self.assertEqual(stats["inserted"], 0)
            finally:
                search.close()

    def test_every_registered_prefix_is_what_its_chunks_carry(self):
        """A feed's prefix and the ids its chunks get must be the same string;
        they come from one place, and this is what proves it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "any"
            _write(root, "a.md", PROJECT_DOCUMENT)
            for spec in feeds():
                chunks = build_chunks(root, 1800, spec.prefix)
                self.assertTrue(chunks)
                for chunk_id, _, _ in chunks:
                    self.assertTrue(chunk_id.startswith(spec.prefix))


class FeedVectorTests(unittest.TestCase):
    def _index(self, ids: list[str]) -> dict:
        rows = len(ids)
        return {
            "embeddings": np.zeros((rows, 4), dtype=np.float32),
            "chunk_ids": np.asarray(ids),
            "contents": np.asarray([f"content {i}" for i in ids]),
            "sources": np.asarray([f"/src/{i}" for i in ids]),
            "domains": np.asarray(["general"] * rows),
            "entity_types": np.asarray([ENTITY_TYPE_CORPUS] * rows),
            "model": "bge-m3",
            "dimension": 4,
        }

    def test_a_third_namespace_merges_without_touching_the_others(self):
        index = self._index([f"{MEMORY_CHUNK_PREFIX}a#0", "gem:chat.md#0"])
        chunks = [(f"{PROJECT_CHUNK_PREFIX}docs/feeds.md#0", "текст", "/p/x.md")]
        merged, replaced = merge_namespace(index, chunks, np.zeros((1, 4), dtype=np.float32),
                                           ["general"], PROJECT_CHUNK_PREFIX, ENTITY_TYPE_CORPUS)
        self.assertEqual(replaced, 0, "nothing was there to replace yet")
        self.assertEqual(len(merged["chunk_ids"]), 3)
        self.assertEqual(merged["chunk_ids"][0], f"{MEMORY_CHUNK_PREFIX}a#0")
        self.assertEqual(merged["chunk_ids"][1], "gem:chat.md#0")
        self.assertEqual(merged["chunk_ids"][2], f"{PROJECT_CHUNK_PREFIX}docs/feeds.md#0")

        # And a second merge replaces exactly its own rows.
        merged, replaced = merge_namespace(merged, chunks, np.ones((1, 4), dtype=np.float32),
                                           ["general"], PROJECT_CHUNK_PREFIX, ENTITY_TYPE_CORPUS)
        self.assertEqual(replaced, 1)
        self.assertEqual(len(merged["chunk_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
