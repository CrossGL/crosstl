"""Run and qualify the companion upstream test suites alongside the main suite."""

import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import ROOT, load_module  # noqa: E402

COMPANIONS = load_module("computesharp_companions", ROOT / "companions.py")
QUALIFY = load_module("computesharp_qualify_suites", ROOT / "qualify.py")
SUMMARY = load_module("computesharp_results_summary", ROOT / "results_summary.py")
SUITES = sys.modules["suites"]
EXPECTED = {
    "ComputeSharp.Tests.DisableDynamicCompilation",
    "ComputeSharp.Tests.GlobalStatements",
    "ComputeSharp.Tests.Internals",
}


def test_every_ported_project_but_the_main_suite_is_a_companion():
    names = [item.name for item in COMPANIONS.companion_suites()]
    assert names == sorted(EXPECTED)
    assert SUITES.MAIN_SUITE not in names


class FakeDotnet:
    """Record commands and produce the files the real tools would write."""

    def __init__(self, tmp_path, *, test_exit=0, fail=None):
        self.tmp_path = tmp_path
        self.commands = []
        self.test_exit = test_exit
        self.fail = fail

    def __call__(self, command, log, *, deadline, check=True):
        self.commands.append(command)
        if self.fail and self.fail in command:
            raise COMPANIONS.StepError(f"{self.fail} failed")
        stdout = ""
        if command[1] == "msbuild":
            assembly = self.tmp_path / "bin" / (Path(command[2]).stem + ".dll")
            assembly.parent.mkdir(parents=True, exist_ok=True)
            assembly.write_bytes(b"assembly")
            stdout = str(assembly) + "\n"
        return type("Completed", (), {"stdout": stdout, "returncode": 0})()


def test_baseline_runs_each_suite_as_the_upstream_step(tmp_path, monkeypatch):
    fake = FakeDotnet(tmp_path)
    calls = []
    monkeypatch.setattr(
        COMPANIONS.coverage_contract,
        "check",
        lambda discovery, lock, contract: calls.append((lock, contract))
        or {"caseCount": 1, "totals": {}},
    )
    monkeypatch.setattr(
        COMPANIONS.baseline,
        "load_discovery",
        lambda directory: {
            "assembly": {
                "sha256": COMPANIONS._sha256(next((tmp_path / "bin").iterdir()))
            }
        },
    )
    accounted = []

    def verify(discovery, trx, profile, contract):
        accounted.append((trx.name, profile, contract))
        return {
            "expected": {"pass": 1},
            "requiredCasesPassed": 1,
            "requiredGpuCasesPassed": 1,
        }

    monkeypatch.setattr(COMPANIONS.baseline, "verify_results", verify)
    programs = []
    for name in sorted(EXPECTED):
        for stale in (tmp_path / "bin").glob("*.dll"):
            stale.unlink()
        item = SUITES.suite(name)
        report = COMPANIONS.baseline_suite(
            item,
            tmp_path / "upstream",
            tmp_path / "out" / name,
            runner=fake,
            program=lambda *args, **kwargs: programs.append((args, kwargs)),
        )
        assert report["accounting"]["requiredCasesPassed"] == 1
        assert calls[-1] == (item.lock, item.contract)
        assert accounted[-1] == ("baseline.trx", "directx-warp", item.contract)
    builds = [c for c in fake.commands if c[1] == "build"]
    assert len(builds) == 3
    for command in builds:
        assert "-p:Platform=x64" in command
        assert (
            f"-p:CustomAfterMicrosoftCommonTargets={COMPANIONS.AUDIT_PROPS}" in command
        )
    tests = [c for c in fake.commands if c[1] == "test"]
    assert [Path(c[2]).stem for c in tests] == [
        "ComputeSharp.Tests.DisableDynamicCompilation",
        "ComputeSharp.Tests.Internals",
    ]
    assert all("--no-build" in c and "-p:Platform=x64" in c for c in tests)
    ((args, kwargs),) = programs
    assert args[0].name == "ComputeSharp.Tests.GlobalStatements.dll"
    assert kwargs["trx_name"] == "baseline.trx"


def test_a_changed_assembly_fails_the_baseline(tmp_path, monkeypatch):
    fake = FakeDotnet(tmp_path)
    monkeypatch.setattr(
        COMPANIONS.coverage_contract, "check", lambda *a: {"caseCount": 1, "totals": {}}
    )
    monkeypatch.setattr(
        COMPANIONS.baseline,
        "load_discovery",
        lambda d: {"assembly": {"sha256": "0" * 64}},
    )
    item = SUITES.suite("ComputeSharp.Tests.Internals")
    with pytest.raises(COMPANIONS.StepError, match="changed during execution"):
        COMPANIONS.baseline_suite(item, tmp_path, tmp_path / "out", runner=fake)


def test_translated_runs_build_against_the_portable_runtime(tmp_path, monkeypatch):
    fake = FakeDotnet(tmp_path)
    monkeypatch.setattr(
        COMPANIONS.coverage_contract, "check", lambda *a: {"caseCount": 1, "totals": {}}
    )
    item = SUITES.suite("ComputeSharp.Tests.DisableDynamicCompilation")
    directory = tmp_path / "out"
    (directory / "run").mkdir(parents=True)
    (directory / "run" / "results.json").write_text(
        json.dumps({"status": "passed", "accounting": {"requiredCasesPassed": 4}})
    )
    report = COMPANIONS.translated_suite(
        item, tmp_path / "adapted", directory, profile="metal-apple-gpu", runner=fake
    )
    assert report["run"]["requiredCasesPassed"] == 4
    build, discovery, run = fake.commands
    assert (
        f"-p:CustomAfterMicrosoftCommonTargets={COMPANIONS.PORTABLE_TARGETS}" in build
    )
    assert "-p:Platform=x64" not in build
    assert discovery[-2] == str(tmp_path / "adapted" / item.assembly)
    assert run[run.index("--suite") + 1] == item.name
    assert run[run.index("--profile") + 1] == "metal-apple-gpu"
    assert "--negative-controls" in run and "--prebuilt-tests" in run


def test_every_suite_runs_although_one_fails(tmp_path):
    selected = COMPANIONS.companion_suites()

    def run_one(item, directory):
        if item.name.endswith("GlobalStatements"):
            raise COMPANIONS.StepError("the program failed")
        return {"coverage": {"caseCount": 1}}

    report = COMPANIONS.run_all(selected, tmp_path, run_one)
    assert report["status"] == "failed"
    assert set(report["suites"]) == EXPECTED
    assert report["suites"]["ComputeSharp.Tests.GlobalStatements"] == {
        "status": "failed",
        "error": "the program failed",
    }
    written = json.loads((tmp_path / "companions.json").read_text())
    assert written == report
    assert COMPANIONS.run_all([], tmp_path / "none", run_one)["status"] == "failed"


def test_each_suite_qualifies_with_its_own_frozen_files(tmp_path, monkeypatch):
    seen = {}

    def accounting(directory, *, contract_path, lock_path):
        seen["directx"] = (directory, contract_path, lock_path)
        return {"profile": "directx-warp", "coverage": {"status": "passed"}}

    def execution(name, directory, **kwargs):
        seen[name] = (directory, kwargs)
        return {"profile": QUALIFY.TARGETS[name], "run": {"status": "passed"}}

    monkeypatch.setattr(QUALIFY, "directx_accounting", accounting)
    monkeypatch.setattr(QUALIFY, "target_execution", execution)
    item = SUITES.suite("ComputeSharp.Tests.Internals")
    report = QUALIFY.qualify_suite(
        item,
        directx=tmp_path / "directx",
        targets={"metal": tmp_path / "metal", "opengl": tmp_path / "opengl"},
    )
    assert report["status"] == "passed"
    original = tmp_path / "directx" / "suites" / item.name
    assert seen["directx"] == (original, item.contract, item.lock)
    for name in ("metal", "opengl"):
        directory, kwargs = seen[name]
        assert directory == tmp_path / name / "suites" / item.name
        assert kwargs["directx"] == original
        assert kwargs["contract_path"] == item.contract
        assert kwargs["lock_path"] == item.lock
        assert kwargs["policy_path"] == item.evidence_policy
        assert kwargs["controls"] == json.loads(item.negative_controls.read_text())
        # Internals declares that none of its artifacts does float arithmetic.
        assert kwargs["require_float_operations"] is False
    assert report["noFloatArithmeticReason"] == item.no_float_arithmetic_reason
    assert report["evidencePolicySHA256"] == QUALIFY._sha256(item.evidence_policy)

    other = SUITES.suite("ComputeSharp.Tests.DisableDynamicCompilation")
    QUALIFY.qualify_suite(other, directx=tmp_path, targets={"metal": tmp_path})
    assert seen["metal"][1]["require_float_operations"] is True


def test_a_failing_profile_fails_the_suite_and_the_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(
        QUALIFY,
        "directx_accounting",
        lambda directory, **kwargs: {"accounting": {"status": "passed"}},
    )
    monkeypatch.setattr(
        QUALIFY,
        "target_execution",
        lambda name, directory, **kwargs: {
            "negativeControls": {"status": "failed" if name == "opengl" else "passed"}
        },
    )
    companions = {
        name: SUITES.suite(name) for name in ("ComputeSharp.Tests.GlobalStatements",)
    }
    report = QUALIFY.qualify(
        directx=tmp_path / "directx",
        targets={"metal": tmp_path / "metal", "opengl": tmp_path / "opengl"},
        output=tmp_path / "qualification",
        companion_suites=companions,
    )
    suite = report["suites"]["ComputeSharp.Tests.GlobalStatements"]
    assert suite["status"] == "failed"
    assert suite["profiles"]["opengl-mesa-llvmpipe"]["status"] == "failed"
    assert suite["profiles"]["metal-apple-gpu"]["status"] == "passed"
    assert report["status"] == "failed"
    lines = QUALIFY.summary_lines(report)
    assert "  suite ComputeSharp.Tests.GlobalStatements: failed" in lines
    summary = json.loads((tmp_path / "qualification" / "results.json").read_text())
    published = summary["suites"]["ComputeSharp.Tests.GlobalStatements"]
    assert published["status"] == "failed"
    assert published["kind"] == "program"
    assert published["profiles"]["opengl-mesa-llvmpipe"]["checks"] == {
        "negativeControls": "failed"
    }


def test_missing_companion_artifacts_fail_without_crashing(tmp_path):
    item = SUITES.suite("ComputeSharp.Tests.DisableDynamicCompilation")
    report = QUALIFY.qualify_suite(
        item, directx=tmp_path / "directx", targets={"metal": tmp_path / "metal"}
    )
    assert report["status"] == "failed"
    profiles = report["profiles"]
    assert profiles["directx-warp"]["coverage"]["status"] == "failed"
    assert profiles["metal-apple-gpu"]["evidence"]["status"] == "failed"
    assert profiles["opengl-mesa-llvmpipe"]["errors"] == [
        "no opengl artifacts were supplied"
    ]


def test_qualification_command_requires_every_companion_suite(tmp_path, monkeypatch):
    captured = {}

    def qualify(**kwargs):
        captured.update(kwargs)
        return {"status": "passed", "profiles": {}, "suites": {}}

    monkeypatch.setattr(QUALIFY, "qualify", qualify)
    assert QUALIFY.main(["--directx", str(tmp_path), "--output", str(tmp_path)]) == 0
    assert set(captured["companion_suites"]) == EXPECTED


def test_summary_of_a_report_without_companions_has_no_suites():
    assert SUMMARY.summarize({"status": "passed", "profiles": {}})["suites"] == {}
