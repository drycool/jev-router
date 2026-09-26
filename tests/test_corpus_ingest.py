"""Tests for importing a Takeout chat export and indexing it into Tier-2 FTS5.

Written against the measured failure: a question whose answer was discussed in
Gemini returned the nearest unrelated neighbour (an OLX listing, at cosine 0.47)
because the whole Gemini part of the corpus was ONE chat - the single
`Gemini-*.md` file that happened to be in `~/Downloads` when the old converter
last ran.  The export held the rest all along.

So the facts worth protecting are:

  * a conversation can be rebuilt from the activity export, and each chunk says
    which conversation and which date it came from (a chunk retrieved alone has
    to say what it is about);
  * the import reaches the index the server actually rebuilds on start, in both
    forms - FTS5 rows and vectors - because FTS5 alone answers a literal query
    while a paraphrase still gets the unrelated neighbour;
  * re-indexing one namespace never deletes another's rows: the memory documents
    and the imported chats share one table and one archive, and `replace_rows`
    deleting by tag instead of by chunk-id namespace would take the LightRAG
    corpus with it (every imported row is untagged, exactly like a LightRAG row).

The pipeline tests run against a real FTS5 table and a real (temporary) archive,
because a test that only checks a returned list proves nothing about what the
index contains afterwards.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.memory_index import (
    CORPUS_CHUNK_PREFIX,
    CORPUS_DIR,
    ENTITY_TYPE,
    ENTITY_TYPE_CORPUS,
    MEMORY_CHUNK_PREFIX,
    build_chunks,
    feed,
    index_corpus,
    index_memory,
)
from core.router import Tier2Search
from core.vector_index import (
    CORPUS_CHUNK_PREFIX as VECTOR_CORPUS_CHUNK_PREFIX,
    ENTITY_CORPUS,
    ENTITY_MEMORY,
    MEMORY_CHUNK_PREFIX as VECTOR_MEMORY_CHUNK_PREFIX,
    corpus_rows_from_fts5,
    merge_corpus,
)
from scripts.import_chat_export import (
    DEFAULT_OUT,
    conversation_id,
    conversation_markdown,
    group_by_conversation,
    item_answer,
    item_question,
    select_conversations,
    strip_html,
)

MEMORY_DOCUMENT = """# Деплой демонов: systemd --user

## Установка юнита

`systemctl --user enable --now jev.service` — и юнит поднимается при загрузке.
"""

CHAT_TITLE = "Запит: Ups hat годиться ои он в качестве буферного безперебойника"
CHAT_ANSWER = ("<p>Плата <strong>Geekworm X728 V2.5</strong> решает большинство проблем, "
               "но для автомобиля не подходит.</p><ul><li>вход строго 5 В DC</li>"
               "<li>скрипты под Raspbian</li></ul>")
CHAT_URL = "https://gemini.google.com/app/d83053a82f419e50"


def _item(minutes: int, question: str = CHAT_TITLE, answer: str = CHAT_ANSWER,
          url: str = CHAT_URL) -> dict:
    return {
        "header": "Додатки Gemini",
        "title": question,
        "time": f"2026-09-13T09:{minutes:02d}:00.000Z",
        "details": [{"name": url, "url": url}] if url else [],
        "safeHtmlItem": [{"html": answer}] if answer else [],
    }


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class HtmlToTextTests(unittest.TestCase):
    def test_block_tags_become_line_breaks_and_entities_are_decoded(self):
        text = strip_html("<p>Первый</p><p>Второй &quot;цитата&quot;</p><ul><li>пункт</li></ul>")
        self.assertEqual(text, 'Первый\nВторой "цитата"\nпункт')

    def test_the_attachment_counter_is_not_indexed_as_content(self):
        text = strip_html("<p>1 вкладений файл.</p><p>Настоящий ответ</p>")
        self.assertEqual(text, "Настоящий ответ")

    def test_empty_input_stays_empty(self):
        self.assertEqual(strip_html(""), "")


class ExportParsingTests(unittest.TestCase):
    def test_the_conversation_id_comes_from_the_chat_url(self):
        self.assertEqual(conversation_id(_item(0)), "d83053a82f419e50")

    def test_an_item_without_a_chat_url_has_no_conversation(self):
        self.assertEqual(conversation_id(_item(0, url="")), "")
        self.assertEqual(conversation_id({"title": "нет деталей"}), "")

    def test_the_prompt_prefix_is_not_part_of_the_question(self):
        self.assertEqual(item_question(_item(0)), CHAT_TITLE[len("Запит:"):].strip())

    def test_an_item_without_an_answer_is_not_knowledge(self):
        # The activity log records every prompt, including the ones whose answer
        # never arrived: a prompt alone would answer the corpus with itself.
        grouped = group_by_conversation([_item(0), _item(1, answer="коротко"),
                                         _item(2, url=""), _item(3, question="")])
        self.assertEqual(list(grouped), ["d83053a82f419e50"])
        self.assertEqual(len(grouped["d83053a82f419e50"]), 1)

    def test_turns_come_back_in_time_order(self):
        grouped = group_by_conversation([_item(30), _item(10), _item(20)])
        stamps = [stamp for stamp, _, _ in grouped["d83053a82f419e50"]]
        self.assertEqual(stamps, sorted(stamps))

    def test_each_chunk_says_which_conversation_and_date_it_came_from(self):
        grouped = group_by_conversation([_item(9)])
        filename, markdown = conversation_markdown("d83053a82f419e50", grouped["d83053a82f419e50"])
        self.assertTrue(filename.startswith("Gemini-"))
        self.assertIn("# Gemini: ", markdown)
        self.assertIn("https://gemini.google.com/app/d83053a82f419e50", markdown)
        self.assertIn("## 2026-09-13 09:09", markdown)
        self.assertIn("Вопрос:", markdown)
        self.assertIn("Ответ:", markdown)

    def test_the_chunk_header_carries_the_conversation_into_the_index(self):
        grouped = group_by_conversation([_item(9)])
        _, markdown = conversation_markdown("d83053a82f419e50", grouped["d83053a82f419e50"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root, "chat.md", markdown)
            chunks = build_chunks(root, 1800, CORPUS_CHUNK_PREFIX)
            self.assertTrue(chunks)
            for _, content, _ in chunks:
                self.assertTrue(content.startswith("Gemini: "),
                                f"chunk does not name its chat: {content[:60]!r}")
                self.assertIn("2026-09-13 09:09", content)

    def test_selection_accepts_repeated_flags_and_comma_lists(self):
        other = "https://gemini.google.com/app/aaaaaaaaaaaaaaaa"
        elsewhere = ("<p>Отдельный разговор про реле и буферное питание бортового "
                     "компьютера в машине, без упоминания платы.</p>")
        grouped = group_by_conversation([_item(0), _item(1, answer=elsewhere, url=other)])
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(select_conversations(grouped, [], ["x728"])), 1)
        self.assertEqual(len(select_conversations(grouped, [], ["реле,x728"])), 2,
                         "a comma-separated list is a list of needles, not one literal")
        self.assertEqual(len(select_conversations(grouped, ["d83053a82f419e50"], [])), 1)
        self.assertEqual(len(select_conversations(grouped, ["d83053a82f419e50,aaaaaaaaaaaaaaaa"], [])), 2)
        self.assertEqual(select_conversations(grouped, [], ["ничего-такого"]), {})


class CorpusIndexingTests(unittest.TestCase):
    def _search(self, directory: str) -> Tier2Search:
        with patch("core.router.FTS5_DB_PATH", str(Path(directory) / "index.db")):
            return Tier2Search()

    def _roots(self, directory: str) -> tuple[Path, Path]:
        memory = Path(directory) / "memory"
        corpus = Path(directory) / "gemini_chats"
        _write(memory, "global/linux_systemd.md", MEMORY_DOCUMENT)
        grouped = group_by_conversation([_item(9)])
        filename, markdown = conversation_markdown("d83053a82f419e50", grouped["d83053a82f419e50"])
        _write(corpus, filename, markdown)
        return memory, corpus

    def test_reindexing_the_corpus_replaces_its_own_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            _, corpus = self._roots(directory)
            search = self._search(directory)
            try:
                first = index_corpus(search.conn, corpus)
                second = index_corpus(search.conn, corpus)
                self.assertEqual(first["removed"], 0)
                self.assertEqual(second["removed"], first["inserted"],
                                 "the second run must replace the first run's rows")
            finally:
                search.close()

    def test_indexing_the_corpus_does_not_delete_the_memory_rows(self):
        """The memory documents share the table, and both are untagged-or-tagged
        neighbours: deleting by tag would take the LightRAG corpus with it."""
        with tempfile.TemporaryDirectory() as directory:
            memory, corpus = self._roots(directory)
            search = self._search(directory)
            try:
                index_memory(search.conn, memory)
                before = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE entity_type = ?", (ENTITY_TYPE,)
                ).fetchone()[0]
                index_corpus(search.conn, corpus)
                index_corpus(search.conn, corpus)
                after = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE entity_type = ?", (ENTITY_TYPE,)
                ).fetchone()[0]
                self.assertEqual(before, after, "memory rows were deleted by the corpus rebuild")
            finally:
                search.close()

    def test_reindexing_memory_does_not_delete_the_imported_chats(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, corpus = self._roots(directory)
            search = self._search(directory)
            try:
                index_corpus(search.conn, corpus)
                self.assertTrue(index_corpus(search.conn, corpus)["inserted"])
                index_memory(search.conn, memory)
                index_memory(search.conn, memory)
                left = search.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE chunk_id LIKE ?",
                    (f"{CORPUS_CHUNK_PREFIX}%",),
                ).fetchone()[0]
                self.assertTrue(left, "the memory rebuild deleted the imported chats")
            finally:
                search.close()

    def test_imported_chats_do_not_carry_the_memory_boost(self):
        with tempfile.TemporaryDirectory() as directory:
            _, corpus = self._roots(directory)
            search = self._search(directory)
            try:
                index_corpus(search.conn, corpus)
                tags = {row[0] for row in search.conn.execute(
                    "SELECT DISTINCT entity_type FROM chunks WHERE chunk_id LIKE ?",
                    (f"{CORPUS_CHUNK_PREFIX}%",),
                )}
                self.assertEqual(tags, {ENTITY_TYPE_CORPUS})
            finally:
                search.close()

    def test_an_imported_chat_is_findable_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            _, corpus = self._roots(directory)
            search = self._search(directory)
            try:
                index_corpus(search.conn, corpus)
                hits = search.search_fts5("X728")
                self.assertTrue(hits, "the imported conversation is not findable by name")
                self.assertTrue(all(hit["chunk_id"].startswith(CORPUS_CHUNK_PREFIX)
                                    for hit in hits))
            finally:
                search.close()

    def test_a_missing_corpus_directory_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            search = self._search(directory)
            try:
                stats = index_corpus(search.conn, Path(directory) / "does-not-exist")
                self.assertFalse(stats["exists"])
                self.assertEqual(stats["inserted"], 0)
            finally:
                search.close()


class CorpusVectorTests(unittest.TestCase):
    """The archive is the other half: without it a paraphrase still fails."""

    def _index(self, ids: list[str], tags: list[str] | None = None) -> dict:
        tags = tags if tags is not None else [ENTITY_CORPUS for _ in ids]
        rows = len(ids)
        return {
            "embeddings": np.zeros((rows, 4), dtype=np.float32),
            "chunk_ids": np.asarray(ids),
            "contents": np.asarray([f"content {i}" for i in ids]),
            "sources": np.asarray([f"/src/{i}.md" for i in ids]),
            "domains": np.asarray(["general"] * rows),
            "entity_types": np.asarray(tags),
            "model": "bge-m3",
            "dimension": 4,
        }

    def test_the_prefixes_agree_between_the_indexer_and_the_archive(self):
        """Two modules spell these ids out; a drift would make one namespace
        invisible to the merge while the index still looked healthy."""
        self.assertEqual(CORPUS_CHUNK_PREFIX, VECTOR_CORPUS_CHUNK_PREFIX)
        self.assertEqual(MEMORY_CHUNK_PREFIX, VECTOR_MEMORY_CHUNK_PREFIX)

    def test_merging_the_corpus_replaces_its_own_rows_only(self):
        index = self._index(
            [f"{MEMORY_CHUNK_PREFIX}a#0",
             f"{CORPUS_CHUNK_PREFIX}row0",
             f"{CORPUS_CHUNK_PREFIX}row1",
             f"{CORPUS_CHUNK_PREFIX}row2"],
            [ENTITY_MEMORY, ENTITY_CORPUS, ENTITY_CORPUS, ENTITY_CORPUS],
        )
        chunks = [(f"{CORPUS_CHUNK_PREFIX}row0", "новый текст", "/src/0.md"),
                  (f"{CORPUS_CHUNK_PREFIX}row1", "новый текст", "/src/1.md")]
        merged, stats = merge_corpus(index, chunks, np.ones((2, 4), dtype=np.float32))

        self.assertEqual(stats["corpus_replaced"], 3, "every old corpus row is dropped")
        self.assertEqual(stats["corpus_added"], 2)
        self.assertEqual(stats["memory"], 1, "the memory row must survive")
        self.assertEqual(list(merged["chunk_ids"]),
                         [f"{MEMORY_CHUNK_PREFIX}a#0",
                          f"{CORPUS_CHUNK_PREFIX}row0",
                          f"{CORPUS_CHUNK_PREFIX}row1"])
        self.assertEqual(list(merged["entity_types"]),
                         [ENTITY_MEMORY, ENTITY_CORPUS, ENTITY_CORPUS])

    def test_merging_the_corpus_twice_leaves_one_copy(self):
        index = self._index([])
        chunks = [(f"{CORPUS_CHUNK_PREFIX}row0", "текст", "/src/0.md")]
        once, _ = merge_corpus(index, chunks, np.zeros((1, 4), dtype=np.float32))
        twice, second = merge_corpus(once, chunks, np.ones((1, 4), dtype=np.float32))
        self.assertEqual(second["corpus_replaced"], 1)
        self.assertEqual(len(twice["chunk_ids"]), 1)

    def test_a_wrong_dimension_is_refused(self):
        from core.vector_index import VectorIndexError
        index = self._index([])
        with self.assertRaises(VectorIndexError):
            merge_corpus(index, [(f"{CORPUS_CHUNK_PREFIX}r", "x", "/s.md")],
                         np.zeros((1, 8), dtype=np.float32))

    def test_fts5_supplies_the_rows_to_embed(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "index.db"
            with patch("core.router.FTS5_DB_PATH", str(db)):
                search = Tier2Search()
                try:
                    corpus = Path(directory) / "chats"
                    grouped = group_by_conversation([_item(9)])
                    filename, markdown = conversation_markdown("d83053a82f419e50",
                                                               grouped["d83053a82f419e50"])
                    _write(corpus, filename, markdown)
                    index_corpus(search.conn, corpus)
                finally:
                    search.close()
            rows = corpus_rows_from_fts5(db)
            self.assertTrue(rows)
            self.assertTrue(all(cid.startswith(CORPUS_CHUNK_PREFIX) for cid, _, _ in rows))


class WiringTests(unittest.TestCase):
    """The import only counts if it reaches the things that are rebuilt on start."""

    def _source(self, relative: str) -> str:
        return (Path(__file__).resolve().parents[1] / relative).read_text(encoding="utf-8")

    def test_the_server_indexes_the_feeds_after_the_lightrag_rebuild(self):
        source = self._source("api/server.py")
        lightrag_call = source.index("await _index_lightrag_chunks()")
        feeds_call = source.index("await _index_feeds()")
        self.assertLess(lightrag_call, feeds_call,
                        "the LightRAG rebuild starts with clear(); indexing a feed "
                        "before it would delete the imported rows on every start")

    def test_the_chat_corpus_is_a_registered_feed(self):
        """The import only counts if the directory the importer writes is the one
        the indexer reads, under its own id namespace."""
        self.assertEqual(feed("chats").directory, str(DEFAULT_OUT))
        self.assertEqual(feed("chats").prefix, CORPUS_CHUNK_PREFIX)
        self.assertEqual(feed("chats").entity_type, ENTITY_TYPE_CORPUS)

    def test_a_full_vector_rebuild_merges_the_imported_corpus(self):
        """Otherwise the next rebuild drops the import: the archive would look
        healthy while those conversations became unfindable again."""
        source = self._source("scripts/build_vector_index.py")
        self.assertIn("merge_corpus(", source)
        self.assertIn("corpus_rows_from_fts5(", source)

    def test_the_importer_writes_where_the_server_reads(self):
        """One definition of the directory, taken from the indexer: an importer
        with its own default would write where nothing reads."""
        source = self._source("scripts/import_chat_export.py")
        self.assertIn("from core.memory_index import CORPUS_DIR", source)
        self.assertIn("DEFAULT_OUT = Path(CORPUS_DIR)", source)
        self.assertEqual(str(DEFAULT_OUT), CORPUS_DIR)


if __name__ == "__main__":
    unittest.main()
