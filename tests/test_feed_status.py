"""Tests for `jev-feed status`: what each feed holds, and what it said about itself.

The table this script prints is the only place where "is the base up to date" has an
answer, so the parts that can be wrong silently are the parts tested here: which chunks
count as a feed's, which count as corpus, and which of a manifest's silences (an error, a
deliberate skip, dropped reasoning) reach the reader.

An unnamed chunk used to be corpus *by falling through five hardcoded `not like` clauses*.
A sixth feed would have been counted as corpus and its backlog reported as a policy, so
the tests below pin the list to the registry instead of to this script.
"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.memory_index import Feed, feeds  # noqa: E402
from core.vector_index import save_index  # noqa: E402
from scripts.feed_status import (  # noqa: E402
    UNPREFIXED,
    count_corpus_chunks,
    feed_prefixes,
    main,
    manifest_summary,
    prefix_of,
)


def make_db(rows, path: Path) -> Path:
    """An FTS5-shaped table with only the column the status query reads."""
    connection = sqlite3.connect(path)
    connection.execute("create table chunks(chunk_id text)")
    connection.executemany("insert into chunks(chunk_id) values (?)", [(row,) for row in rows])
    connection.commit()
    connection.close()
    return path


class PrefixTest(unittest.TestCase):
    def test_the_feed_prefixes_are_the_registry_not_a_list_in_this_script(self):
        prefixes = feed_prefixes()
        self.assertEqual(sorted(prefixes), sorted(spec.prefix for spec in feeds()))
        self.assertIn("ses:", prefixes)
        self.assertIn("gh:", prefixes)

    def test_a_prefix_of_a_chunk_id_is_its_feed(self):
        self.assertEqual(prefix_of("mem:projects/jev_ops_defects.md#3"), "mem:")
        self.assertEqual(prefix_of("prj:Jev__commits.md#5"), "prj:")
        self.assertEqual(prefix_of("ses:hermes/2026-09#1"), "ses:")

    def test_a_chunk_id_without_a_feed_prefix_is_corpus(self):
        self.assertEqual(prefix_of("9f3a1c8e2b7d4a5f"), UNPREFIXED)
        self.assertEqual(prefix_of(""), UNPREFIXED)
        # An unknown prefix is not a feed: it is material the router serves but that no
        # collector claims, and calling it a feed would invent a lag that cannot be fixed.
        self.assertEqual(prefix_of("zzz:1"), UNPREFIXED)

    def test_a_colon_deep_inside_a_corpus_hash_is_not_a_prefix(self):
        # Corpus ids are long hashes; one may contain a colon and must not become a feed.
        self.assertEqual(prefix_of("a1b2c3d4e5f6a7b8:extra"), UNPREFIXED)


class CorpusCountTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = make_db(["mem:1", "mem:2", "ses:1", "9f3a1c8e", "plain"], 
                          Path(self.directory.name) / "fts.db")

    def tearDown(self):
        self.directory.cleanup()

    def test_only_chunks_of_registered_feeds_are_excluded_from_the_corpus(self):
        connection = sqlite3.connect(self.db)
        prefixes = [spec.prefix for spec in feeds()]
        self.assertEqual(count_corpus_chunks(connection, prefixes), 2)
        connection.close()

    def test_a_sixth_feed_is_not_reported_as_corpus(self):
        """The regression the registry-driven count exists to prevent."""
        connection = sqlite3.connect(self.db)
        self.assertEqual(count_corpus_chunks(connection, ["mem:", "ses:", "fb:"]), 2)
        self.assertEqual(count_corpus_chunks(connection, ["mem:", "ses:", "fb:", "9f3a1c8e"]), 1)
        connection.close()

    def test_an_empty_registry_makes_everything_corpus_rather_than_nothing(self):
        # A `where` built from zero clauses is a syntax error, and answering 0 would say
        # "the corpus is empty" - the opposite of the truth.
        connection = sqlite3.connect(self.db)
        self.assertEqual(count_corpus_chunks(connection, []), 5)
        connection.close()


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def write(self, payload) -> str:
        path = Path(self.directory.name) / "feed.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return self.directory.name

    def test_a_feed_without_a_manifest_reports_nothing_rather_than_zero(self):
        # `chats` and `memory` are indexed without the collector, so "no manifest" must be
        # distinguishable from "a manifest that says nothing".
        self.assertEqual(manifest_summary(self.directory.name), {})

    def test_every_kind_of_silence_reaches_the_reader(self):
        directory = self.write({
            "updated_at": "2026-09-26T16:47:18Z",
            "documents": [{"path": "a"}, {"path": "b"}],
            "sources": {
                "espero-kb": {"error": "git rev-parse --short HEAD: fatal: Needed a single revision"},
                "hashcat": {"skip_reason": "форк hashcat/hashcat: 11055 коммитов чужой истории"},
                "hermes-vintazh": {"history_skip_reason": "история апстрима Hermes (6435 коммитов)"},
                "selena": {"bots_skipped": 50},
                "hermes": {"reasoning_chars_dropped": 7751055},
            },
        })
        summary = manifest_summary(directory)
        self.assertEqual(summary["documents"], 2)
        self.assertEqual(summary["sources"], 5)
        self.assertEqual(summary["updated_at"], "2026-09-26T16:47:18Z")
        joined = "\n".join(summary["notes"])
        for fragment in ("ОШИБКА git rev-parse", "пропущен — форк", "без истории — история апстрима",
                         "ботов 50", "рассуждений отброшено 7,751,055"):
            self.assertIn(fragment, joined)

    def test_an_unreadable_manifest_is_reported_not_swallowed(self):
        (Path(self.directory.name) / "feed.json").write_text("{not json", encoding="utf-8")
        self.assertIn("манифест нечитаем", manifest_summary(self.directory.name)["error"])


class MainTest(unittest.TestCase):
    """End to end: the script's only failure mode that tests cannot see is a bad import."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.db = make_db(["mem:1", "mem:2", "ses:1", "9f3a1c8e", "plain"], self.root / "fts.db")
        (self.root / "storage").mkdir()
        save_index(self.root / "storage" / "jev_vectors.npz", {
            "model": "bge-m3",
            "dimension": 2,
            "chunk_ids": ["mem:1", "ses:1", "9f3a1c8e"],
            "contents": ["a", "b", "c"],
            "sources": ["m", "s", "c"],
            "domains": ["general", "general", "general"],
            "entity_types": ["memory", "memory", ""],
            "embeddings": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        })

    def tearDown(self):
        self.directory.cleanup()

    def test_a_missing_database_names_the_next_step_instead_of_printing_zeros(self):
        import io
        from contextlib import redirect_stderr
        stderr = io.StringIO()
        with patch("scripts.feed_status.FTS5_DB_PATH", str(self.root / "нет-такой.db")), \
             patch.object(sys, "argv", ["feed_status.py"]):
            with redirect_stderr(stderr):
                code = main()
        self.assertEqual(code, 2)
        self.assertIn("нет FTS5-таблицы", stderr.getvalue())

    def test_the_json_view_counts_chunks_vectors_and_corpus_per_feed(self):
        import io
        from contextlib import redirect_stdout
        stdout = io.StringIO()
        with patch("scripts.feed_status.FTS5_DB_PATH", str(self.db)), \
             patch("scripts.feed_status.PROJECT_ROOT", self.root), \
             patch.object(sys, "argv", ["feed_status.py", "--json"]), \
             redirect_stdout(stdout):
            code = main()
        self.assertEqual(code, 0)
        payload = json.loads(stdout.getvalue())
        by_name = {row["name"]: row for row in payload["feeds"]}
        self.assertEqual(by_name["memory"]["chunks"], 2)
        self.assertEqual(by_name["memory"]["vectors"], 1)
        # The lag is the product of the design, not a failure: the vector step is separate
        # because the embedder sleeps, so a chunk without a vector must be visible here.
        self.assertEqual(by_name["memory"]["lag"], 1)
        self.assertEqual(by_name["sessions"]["vectors"], 1)
        self.assertEqual(payload["corpus_chunks_unprefixed"], 2)
        self.assertEqual(payload["totals"]["chunks"], 3 + 2)
        self.assertEqual(payload["totals"]["vectors"], 3)

    def test_the_table_says_which_feed_skipped_what(self):
        import io
        from contextlib import redirect_stdout
        manifest = self.root / "memory"
        manifest.mkdir()
        (manifest / "feed.json").write_text(json.dumps({
            "updated_at": "2026-09-26T16:47:18Z",
            "documents": [{"path": "a"}],
            "sources": {"hashcat": {"skip_reason": "форк: 11055 коммитов чужой истории"}},
        }), encoding="utf-8")
        stdout = io.StringIO()
        with patch("scripts.feed_status.FTS5_DB_PATH", str(self.db)), \
             patch("scripts.feed_status.PROJECT_ROOT", self.root), \
             patch("scripts.feed_status.feeds", lambda: [Feed("memory", "mem:", str(manifest), "memory")]), \
             patch.object(sys, "argv", ["feed_status.py"]), \
             redirect_stdout(stdout):
            code = main()
        self.assertEqual(code, 0)
        text = stdout.getvalue()
        self.assertIn("что фиды сказали о себе", text)
        self.assertIn("пропущен — форк: 11055 коммитов чужой истории", text)
        self.assertIn("(политика JEV_VECTOR_EXCLUDE_SOURCES)", text)


if __name__ == "__main__":
    unittest.main()
