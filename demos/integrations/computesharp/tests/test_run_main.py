"""Check how the translated-run harness decides that a run qualifies.

Every phase of ``run_tests.main`` is replaced by a stub, so these checks cover
only the decision: a run qualifies only when exactness, accounting, evidence
and every negative control pass, and a deadline or failed preflight leaves a
failed report.
"""

import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

RUN_TESTS = load_module("computesharp_run_tests_main", ROOT / "run_tests.py")
PROFILE = "metal-apple-gpu"


@pytest.fixture
def stubs(monkeypatch):
    """Stub every phase so that, by default, each one passes."""
    state = {
        "preflight": "passed",
        "exactness": "passed",
        "evidence": None,
        "controls": "passed",
        "deadline": False,
    }

    def translate(discovery, artifacts, target, log):
        (artifacts / target).mkdir(parents=True)
        manifest = {"translator": "crosstl", "counts": {"kernels": 1}}
        (artifacts / target / "manifest.json").write_text(json.dumps(manifest))

    def run_tests(**kwargs):
        if state["deadline"]:
            raise RUN_TESTS.DeadlineExceeded("dotnet test exceeded its deadline")
        return Path("run.trx")

    def verify_evidence(**kwargs):
        if state["evidence"]:
            raise RUN_TESTS.evidence_module.EvidenceError([state["evidence"]])
        return {"status": "passed"}

    def analyze(*args, **kwargs):
        return {"status": state["exactness"], "tool": "stub", "totals": {}}

    def controls(**kwargs):
        status = state["controls"]
        return {
            "status": status,
            "controls": [{"fault": "omit-dispatch", "status": status}],
        }

    passed = [{"name": "Case", "outcome": "Passed"}]
    monkeypatch.setattr(RUN_TESTS, "build_native", lambda *args: None)
    monkeypatch.setattr(RUN_TESTS, "build_tests", lambda *args: None)
    monkeypatch.setattr(
        RUN_TESTS, "preflight", lambda *args: {"status": state["preflight"]}
    )
    monkeypatch.setattr(RUN_TESTS, "translate", translate)
    monkeypatch.setattr(RUN_TESTS, "excluded_methods", lambda *args: [])
    monkeypatch.setattr(RUN_TESTS, "run_tests", run_tests)
    monkeypatch.setattr(RUN_TESTS, "summarize_failures", lambda trx: [])
    monkeypatch.setattr(RUN_TESTS, "run_negative_controls", controls)
    monkeypatch.setattr(RUN_TESTS.exactness, "analyze", analyze)
    monkeypatch.setattr(
        RUN_TESTS.baseline, "verify_results", lambda *args: {"status": "passed"}
    )
    monkeypatch.setattr(RUN_TESTS.evidence_module, "verify", verify_evidence)
    monkeypatch.setattr(RUN_TESTS.evidence_module, "trx_cases", lambda trx: passed)
    monkeypatch.setattr(
        RUN_TESTS.collect_outputs,
        "collect",
        lambda *args: {"outputs": [], "references": []},
    )
    return state


def run(tmp_path, *flags):
    work = tmp_path / "work"
    argv = ["--adapted", str(tmp_path), "--discovery", str(tmp_path)]
    argv += ["--profile", PROFILE, "--work", str(work), *flags]
    status = RUN_TESTS.main(argv)
    return status, json.loads((work / "results.json").read_text())


def test_a_complete_run_with_negative_controls_qualifies(stubs, tmp_path):
    status, report = run(tmp_path, "--negative-controls")
    assert status == 0
    assert report["status"] == "passed"
    assert report["nativeTargetExecutionQualified"] is True
    assert report["accounting"] == {"status": "passed"}


def test_a_run_without_negative_controls_does_not_qualify(stubs, tmp_path):
    status, report = run(tmp_path)
    assert status == 0
    assert report["nativeTargetExecutionQualified"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [("exactness", "failed"), ("evidence", "missing record"), ("controls", "failed")],
)
def test_any_failed_phase_fails_the_run(stubs, tmp_path, key, value):
    stubs[key] = value
    status, report = run(tmp_path, "--negative-controls")
    assert status == 1
    assert report["status"] == "failed"
    assert report["nativeTargetExecutionQualified"] is False


@pytest.mark.parametrize(
    "flags", [("--filter", "Name~Case"), ("--no-evidence",), ("--controls-only",)]
)
def test_diagnostic_runs_never_qualify(stubs, tmp_path, flags):
    status, report = run(tmp_path, "--negative-controls", *flags)
    assert status == 1
    assert report["nativeTargetExecutionQualified"] is False


def test_a_deadline_leaves_a_failed_report(stubs, tmp_path):
    stubs["deadline"] = True
    status, report = run(tmp_path, "--negative-controls")
    assert status == 1
    assert report["status"] == "failed"
    assert "deadline" in report["deadlineExceeded"]
    assert "negativeControls" not in report


@pytest.mark.parametrize("device", ["blocked", "capability-mismatch"])
def test_a_failed_preflight_runs_nothing(stubs, tmp_path, device):
    stubs["preflight"] = device
    status, report = run(tmp_path, "--negative-controls")
    assert status == 1
    assert report["status"] == device
    assert not (tmp_path / "work" / "artifacts").exists()
