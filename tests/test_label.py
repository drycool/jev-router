"""
Tests for the labelling CLI.
============================

The CLI is the reviewer's interface, so the properties worth testing are the ones that keep a
reviewer's attention from being wasted or a label from being silently wrong: that the queue
excludes records which can never be labelled, that skipping is possible, that the question is
attached only when the reviewer supplies one, and that a rejected verdict is reported rather
than swallowed.
"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import label  # noqa: E402
import label_coverage  # noqa: E402


def make_decision(decision_id="", verdicts=None, preview="ответ"):
    return label_coverage.Decision(
        decision_id=decision_id,
        tier="tier2",
        strategy="exact_fts",
        schema="decision_v2" if decision_id else "decision_v1",
        timestamp="2026-09-22T12:00:00+00:00",
        latency_ms=21.0,
        answered=True,
        preview=preview,
        verdicts=list(verdicts or []),
    )


def stub_router(response=None, status=200):
    """Start a one-shot HTTP server that records what the CLI posted to it."""
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 (http.server API)
            length = int(self.headers.get("Content-Length", 0))
            captured.append(json.loads(self.rfile.read(length).decode("utf-8")))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response or {
                "recorded": True, "decision_id": captured[-1]["decision_id"],
                "known_decision": True, "verdict": captured[-1]["verdict"],
                "source": captured[-1]["source"], "verdicts_for_decision": 1,
                "previous_verdict": None,
            }).encode("utf-8"))

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, captured


@contextmanager
def running_router(response=None, status=200):
    """Yield (base_url, captured). Closes the socket, not just the serve loop: an unclosed
    socket turns every run into ResourceWarning noise that hides real warnings."""
    server, captured = stub_router(response, status)
    try:
        yield f"http://127.0.0.1:{server.server_port}", captured
    finally:
        server.shutdown()
        server.server_close()


class TestVerdictKeys(unittest.TestCase):
    def test_the_three_verdicts_have_keys(self):
        self.assertEqual(label.parse_verdict_key("a"), "accepted")
        self.assertEqual(label.parse_verdict_key("r"), "rejected")
        self.assertEqual(label.parse_verdict_key("p"), "partial")

    def test_keys_are_case_insensitive_and_trimmed(self):
        self.assertEqual(label.parse_verdict_key(" R "), "rejected")

    def test_skipping_is_possible(self):
        """A reviewer who is unsure must be able to skip: a guessed label is worse than a
        missing one, because it looks like data."""
        self.assertIsNone(label.parse_verdict_key("s"))
        self.assertIsNone(label.parse_verdict_key(""))

    def test_quit_raises(self):
        with self.assertRaises(KeyboardInterrupt):
            label.parse_verdict_key("q")

    def test_an_invented_key_is_not_a_verdict(self):
        for bad in ("x", "unknown", "1"):
            with self.subTest(key=bad):
                self.assertIsNone(label.parse_verdict_key(bad))


class TestQueue(unittest.TestCase):
    def test_v1_records_are_not_offered(self):
        """They have no decision_id, so no verdict can attach. Listing them would waste the
        reviewer's attention on records that cannot be labelled."""
        decisions = [make_decision(decision_id=""), make_decision(decision_id="a" * 32)]
        self.assertEqual([d.decision_id for d in label.pending(decisions)], ["a" * 32])

    def test_already_labelled_decisions_are_not_offered(self):
        decisions = [
            make_decision(decision_id="a" * 32, verdicts=[{"verdict": "accepted", "source": "human", "index": 0}]),
            make_decision(decision_id="b" * 32),
        ]
        self.assertEqual([d.decision_id for d in label.pending(decisions)], ["b" * 32])

    def test_the_queue_is_newest_first(self):
        decisions = [make_decision(decision_id="old"), make_decision(decision_id="new")]
        self.assertEqual([d.decision_id for d in label.pending(decisions)], ["new", "old"])

    def test_the_summary_states_how_many_can_never_be_labelled(self):
        decisions = [make_decision(decision_id="") for _ in range(3)] + [make_decision(decision_id="a" * 32)]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            label.show_queue(decisions, limit=0)
        output = buffer.getvalue()
        self.assertIn("can never be labelled", output)
        self.assertIn("awaiting a verdict 1", output)

    def test_an_empty_queue_says_so(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            label.show_queue([make_decision(decision_id="a" * 32, verdicts=[{"verdict": "accepted", "source": "human", "index": 0}])], limit=0)
        self.assertIn("nothing awaiting a verdict", buffer.getvalue())


class TestRecordingAVerdict(unittest.TestCase):
    def test_the_verdict_reaches_the_router_with_its_source(self):
        with running_router() as (url, captured):
            result = label.record("a" * 32, "rejected", comment="не тот момент", url=url)

        self.assertEqual(captured[0]["decision_id"], "a" * 32)
        self.assertEqual(captured[0]["verdict"], "rejected")
        # The CLI is the human path; an unattributed label would look like an agent's guess.
        self.assertEqual(captured[0]["source"], "human")
        self.assertEqual(captured[0]["comment"], "не тот момент")
        self.assertTrue(result["recorded"])

    def test_the_question_is_attached_only_when_the_reviewer_supplies_one(self):
        """The decision log keeps only a hash of the query, so the question has to come from
        the reviewer. Sending an empty string would create a field that looks populated."""
        with running_router() as (url, captured):
            label.record("a" * 32, "accepted", url=url)
            label.record("b" * 32, "accepted", query="какой момент затяжки", url=url)

        self.assertNotIn("query", captured[0])
        self.assertEqual(captured[1]["query"], "какой момент затяжки")

    def test_a_whitespace_only_question_counts_as_absent(self):
        with running_router() as (url, captured):
            label.record("a" * 32, "partial", query="   ", url=url)
        self.assertNotIn("query", captured[0])

    def test_a_rejected_verdict_is_reported_not_swallowed(self):
        """If the router refuses the verdict the reviewer must be told, or they will believe
        the label was recorded and never label that answer again."""
        with running_router(response={"detail": "no"}, status=422) as (url, _):
            with self.assertRaises(RuntimeError) as caught:
                label.record("a" * 32, "accepted", url=url)
        self.assertIn("422", str(caught.exception))

    def test_an_unreachable_router_is_reported(self):
        with self.assertRaises(RuntimeError) as caught:
            label.record("a" * 32, "accepted", url="http://127.0.0.1:1", timeout=0.5)
        self.assertIn("unreachable", str(caught.exception))


class TestTheQueueIsJoinedToTheRealVerdictFiles(unittest.TestCase):
    """The bug this class exists for.

    label.py once loaded decisions without joining the feedback file. Everything looked fine,
    and the queue offered decisions that already had verdicts — inviting a second,
    contradictory label and corrupting the dataset the tool exists to protect. The earlier
    test passed anyway because it built Decision objects with verdicts already set in memory:
    it tested the assumption rather than the code. This one goes through real files.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.decisions = os.path.join(self.tmp.name, "decisions.jsonl")
        self.feedback = os.path.join(self.tmp.name, "feedback.jsonl")

        with open(self.decisions, "w", encoding="utf-8") as handle:
            for decision_id in ("a" * 32, "b" * 32):
                handle.write(json.dumps({
                    "schema": "decision_v2", "decision_id": decision_id,
                    "decision": {"strategy": "exact_fts"},
                    "signals": {"tier": "tier2", "answered": True, "answer_preview": "ответ"},
                    "execution": {"latency_ms": 10.0},
                }) + "\n")

        with open(self.feedback, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "schema": "feedback_v1", "decision_id": "a" * 32, "verdict": "rejected",
                "source": "human", "comment": "не тот момент", "query": "какой момент затяжки",
            }) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_decision_with_a_verdict_on_disk_is_not_offered(self):
        decisions = label.load_state(self.decisions, self.feedback)
        offered = [d.decision_id for d in label.pending(decisions)]

        self.assertEqual(offered, ["b" * 32],
                         "a decision that already has a verdict was offered for labelling again")

    def test_the_verdict_and_its_question_survive_the_join(self):
        decisions = label.load_state(self.decisions, self.feedback)
        labelled = next(d for d in decisions if d.decision_id == "a" * 32)

        self.assertEqual(labelled.effective_verdict, "rejected")
        self.assertEqual(labelled.labelled_query, "какой момент затяжки")
        self.assertEqual(labelled.effective_comment, "не тот момент")

    def test_loading_without_the_join_is_what_broke_it(self):
        """Keeps the failure mode visible: the same files, minus the join, offer both
        decisions. If this ever stops being true the helper's reason for existing has gone."""
        without_join = label_coverage.load_decisions(self.decisions)
        self.assertEqual(len(label.pending(without_join)), 2)


if __name__ == "__main__":
    unittest.main()
