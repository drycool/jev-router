"""
Test package initialisation, and a redirect that keeps the dataset honest.
==========================================================================

The decision and feedback logs are the only record of what the router decided and whether
anyone judged it. They must contain real traffic and nothing else.

That was not true. `ShadowProbeTests` posts to /query to exercise the endpoint, so every run
of the suite appended a decision record for its fixture query - 20 of the 71 records in
jev_decisions.jsonl at the time of writing, 28%, all with empty keywords and entities
because the fixture is degenerate. Any statistic computed over that log was quietly
inflated, and a dataset built from it would have been trained on test scaffolding.

Redirecting the paths here, in the package initialiser, is deliberate: this module is
imported before any test module, so no test - including ones written later by someone who
has never read this comment - can write to the production logs by accident. Forgetting to
capture a logger is an easy mistake; a package-level redirect does not depend on anyone
remembering anything.

`tests/test_ground_truth.py::TestProductionLogsAreProtected` asserts that this redirect is
still in force, so that removing it fails loudly instead of silently corrupting the log.
"""
import os
import tempfile

_TEST_LOG_DIR = os.path.join(tempfile.gettempdir(), "jev-test-logs")
os.makedirs(_TEST_LOG_DIR, exist_ok=True)

for _var, _filename in (
    ("JEV_DECISION_LOG_PATH", "decisions.jsonl"),
    ("JEV_FEEDBACK_LOG_PATH", "feedback.jsonl"),
):
    # setdefault, not assignment: an operator who deliberately points these somewhere else
    # to run the suite against a copy should not be overridden.
    os.environ.setdefault(_var, os.path.join(_TEST_LOG_DIR, _filename))
