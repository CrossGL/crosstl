"""Account every upstream test project and run program projects as one case."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "computesharp_fixture_support", TESTS / "fixture_support.py"
)
SUPPORT = importlib.util.module_from_spec(_SUPPORT_SPEC)
_SUPPORT_SPEC.loader.exec_module(SUPPORT)
ROOT = SUPPORT.ROOT
SUITES = SUPPORT.load_module("computesharp_suites", ROOT / "suites.py")
PROGRAMS = SUPPORT.load_module(
    "computesharp_program_runner", ROOT / "program_runner.py"
)
BASELINE = SUPPORT.load_module("computesharp_baseline", ROOT / "baseline.py")
EVIDENCE = SUPPORT.load_module("computesharp_evidence", ROOT / "evidence.py")
RUN_TESTS = SUPPORT.load_module("computesharp_run_tests", ROOT / "run_tests.py")
PROGRAM_CONTRACT = SUPPORT.PROGRAM_CONTRACT
ENTRY = "Program.<Main>$"


def test_inventory_accounts_every_project_with_consistent_contracts():
    inventory = SUITES.load_inventory()
    assert SUITES.check_contracts(inventory) == []
    summary = SUITES.summary(inventory)
    assert summary["projects"] == len(inventory["projects"]) == 15
    ported = SUITES.ported_suites(inventory)
    assert set(ported) == set(summary["scopes"]["ported"])
    assert SUITES.MAIN_SUITE in ported
    assert ported["ComputeSharp.Tests.GlobalStatements"].kind == "program"
    for item in ported.values():
        for path in (
            item.contract,
            item.lock,
            item.evidence_policy,
            item.negative_controls,
        ):
            assert path.is_file(), path
        controls = json.loads(item.negative_controls.read_text(encoding="utf-8"))
        assert controls["controls"], item.name
        assert inventory["projects"][item.name]["upstreamSteps"], item.name


def pinned_tree(root: Path, inventory: dict) -> Path:
    """Write a checkout with every inventoried project and workflow step."""
    for name in inventory["projects"]:
        project = root / "tests" / name
        project.mkdir(parents=True)
        (project / f"{name}.csproj").write_text("<Project />\n")
    jobs: dict = {}
    for name, entry in sorted(inventory["projects"].items()):
        for label in entry["upstreamSteps"]:
            job, step = label.split(": ", 1)
            jobs.setdefault(job, {"steps": []})["steps"].append(
                {"name": step, "run": f"dotnet test tests\\{name}\\{name}.csproj"}
            )
    workflow = root / inventory["workflow"]
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(json.dumps({"jobs": jobs}))
    return root


def test_inventory_check_requires_every_project_and_upstream_step(tmp_path):
    inventory = SUITES.load_inventory()
    source = pinned_tree(tmp_path / "source", inventory)
    assert SUITES.check(source)["projects"] == 15

    extra = source / "tests" / "ComputeSharp.Tests.New"
    extra.mkdir()
    (extra / "ComputeSharp.Tests.New.csproj").write_text("<Project />\n")
    with pytest.raises(ValueError, match="unlisted upstream test project"):
        SUITES.check(source)
    (extra / "ComputeSharp.Tests.New.csproj").unlink()
    extra.rmdir()

    workflow = source / inventory["workflow"]
    jobs = json.loads(workflow.read_text())["jobs"]
    jobs["run-tests"]["steps"].append(
        {
            "name": "Run it again",
            "run": "dotnet test tests\\ComputeSharp.Tests\\x.csproj",
        }
    )
    workflow.write_text(json.dumps({"jobs": jobs}))
    with pytest.raises(ValueError, match="ComputeSharp.Tests: recorded steps"):
        SUITES.check(source)


def test_inventory_check_rejects_a_missing_project(tmp_path):
    inventory = SUITES.load_inventory()
    source = pinned_tree(tmp_path / "source", inventory)
    project = source / "tests" / "ComputeSharp.Tests.Internals"
    (project / "ComputeSharp.Tests.Internals.csproj").unlink()
    with pytest.raises(ValueError, match="Internals is not in the upstream checkout"):
        SUITES.check(source)


def vague_reason(inventory):
    inventory["projects"]["ComputeSharp.Tests.DeviceLost"]["reason"] = "Windows."


def unknown_scope(inventory):
    inventory["projects"]["ComputeSharp.Tests.DeviceLost"]["scope"] = "skipped"


def ported_without_files(inventory):
    del inventory["projects"]["ComputeSharp.Tests.GlobalStatements"]["suite"]


def files_for_an_unported_project(inventory):
    projects = inventory["projects"]
    projects["ComputeSharp.Tests.SourceGenerators"]["suite"] = projects[
        "ComputeSharp.Tests.Internals"
    ]["suite"]


def other_revision(inventory):
    inventory["commit"] = "0" * 40


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (vague_reason, "needs a specific reason"),
        (unknown_scope, "unknown scope"),
        (ported_without_files, "must name its"),
        (files_for_an_unported_project, "not a ported project"),
        (other_revision, "different upstream revision"),
    ],
)
def test_inventory_rejects_incomplete_entries(tmp_path, mutate, message):
    inventory = copy.deepcopy(SUITES.load_inventory())
    mutate(inventory)
    path = tmp_path / "test_projects.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match=message):
        SUITES.load_inventory(path)


def test_suite_contracts_must_share_the_main_profiles(tmp_path, monkeypatch):
    inventory = copy.deepcopy(SUITES.load_inventory())
    entry = inventory["projects"]["ComputeSharp.Tests.GlobalStatements"]
    contract = json.loads((ROOT / entry["suite"]["contract"]).read_text())
    contract["profiles"]["metal-apple-gpu"]["execution"] = "software"
    changed = tmp_path / "coverage.json"
    changed.write_text(json.dumps(contract, indent=2) + "\n")
    entry["suite"]["contract"] = str(changed)
    problems = SUITES.check_contracts(inventory)
    assert any("contract profiles differ" in problem for problem in problems)
    assert any("lock was not frozen" in problem for problem in problems)


def program_discovery(root: Path, assembly: bytes = b"program") -> tuple[Path, Path]:
    """A discovery export of a program, with the assembly it was taken from."""
    discovery = root / "discovery"
    (discovery / "hlsl").mkdir(parents=True)
    hlsl = b"[numthreads(64,1,1)] void Execute(uint3 id : SV_DispatchThreadID) {}\n"
    (discovery / "hlsl" / "shader.hlsl").write_bytes(hlsl)
    signature = ENTRY + "(System.String[])"
    report = {
        "schemaVersion": 1,
        "kind": "computesharp.discovery",
        "assembly": {
            "name": "ProgramFixture",
            "sha256": hashlib.sha256(assembly).hexdigest(),
        },
        "projectKind": "program",
        "testMethodCount": 1,
        "caseCount": 1,
        "shaderCount": 1,
        "methods": [
            {
                "method": ENTRY,
                "signature": signature,
                "source": {"path": "tests/Program/Program.cs", "line": 3},
            }
        ],
        "tests": [
            {
                "id": hashlib.sha256((signature + "\n[]\n1").encode()).hexdigest(),
                "method": ENTRY,
                "signature": signature,
                "displayName": "<Main>$",
                "occurrence": 1,
                "arguments": [],
                "ignored": False,
            }
        ],
        "shaders": [
            {
                "id": "s" * 64,
                "type": "Shader",
                "path": "hlsl/shader.hlsl",
                "sha256": hashlib.sha256(hlsl).hexdigest(),
                "threads": [64, 1, 1],
                "constantBufferSize": 16,
                "staticSamplerRequired": False,
                "resourceDescriptorRanges": [],
            }
        ],
    }
    (discovery / "discovery.json").write_text(json.dumps(report))
    path = root / "ProgramFixture.dll"
    path.write_bytes(assembly)
    return discovery, path


def fake_run(monkeypatch, *, returncode=0, stdout="", stderr="", timeout=False):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if timeout:
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], output=b"partial"
            )
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    monkeypatch.setattr(PROGRAMS.subprocess, "run", run)
    return calls


def test_a_passing_program_is_its_one_accounted_case(tmp_path, monkeypatch):
    discovery, assembly = program_discovery(tmp_path)
    calls = fake_run(monkeypatch, stdout="Test passed successfully!\n")
    trx = PROGRAMS.run(
        assembly, discovery, tmp_path / "trx", log=tmp_path / "run.log", env={}
    )
    ((command, kwargs),) = calls
    assert command == ["dotnet", str(assembly)]
    assert kwargs["env"][PROGRAMS.PROGRAM_ENVIRONMENT] == "ProgramFixture"
    assert EVIDENCE.trx_cases(trx) == [
        {"name": ENTRY, "outcome": "Passed", "message": ""}
    ]
    for profile in ("directx-warp", "translated"):
        report = BASELINE.verify_results(discovery, trx, profile, PROGRAM_CONTRACT)
        assert report["requiredGpuCasesPassed"] == 1
        assert report["warpCasesPassed"] == 0
    assert "Test passed successfully!" in (tmp_path / "run.log").read_text()


def test_a_failing_or_hung_program_fails_its_case(tmp_path, monkeypatch):
    discovery, assembly = program_discovery(tmp_path)
    fake_run(
        monkeypatch,
        returncode=134,
        stderr="Unhandled exception. System.Exception: mismatch\n",
    )
    trx = PROGRAMS.run(
        assembly, discovery, tmp_path / "failed", log=tmp_path / "f.log", env={}
    )
    (case,) = EVIDENCE.trx_cases(trx)
    assert case["outcome"] == "Failed"
    assert (
        "status 134" in case["message"]
        and "System.Exception: mismatch" in case["message"]
    )
    with pytest.raises(
        BASELINE.AccountingError, match="expected pass but reported Failed"
    ):
        BASELINE.verify_results(discovery, trx, "translated", PROGRAM_CONTRACT)

    fake_run(monkeypatch, timeout=True)
    trx = PROGRAMS.run(
        assembly,
        discovery,
        tmp_path / "hung",
        log=tmp_path / "h.log",
        env={},
        deadline=3,
    )
    (case,) = EVIDENCE.trx_cases(trx)
    assert case["outcome"] == "Failed" and "deadline of 3s" in case["message"]


def test_only_the_discovered_program_runs(tmp_path, monkeypatch):
    discovery, assembly = program_discovery(tmp_path)
    calls = fake_run(monkeypatch)
    assembly.write_bytes(b"rebuilt")
    with pytest.raises(
        PROGRAMS.ProgramError, match="not the assembly that discovery exported"
    ):
        PROGRAMS.run(assembly, discovery, tmp_path / "trx", log=tmp_path / "run.log")
    report = json.loads((discovery / "discovery.json").read_text())
    report["projectKind"] = "mstest"
    (discovery / "discovery.json").write_text(json.dumps(report))
    with pytest.raises(PROGRAMS.ProgramError, match="not the discovery of a program"):
        PROGRAMS.run(assembly, discovery, tmp_path / "trx", log=tmp_path / "run.log")
    assert calls == []


def test_a_program_suite_cannot_be_filtered(tmp_path):
    with pytest.raises(SystemExit, match="one case"):
        RUN_TESTS.main(
            [
                "--suite",
                "ComputeSharp.Tests.GlobalStatements",
                "--adapted",
                str(tmp_path),
                "--discovery",
                str(tmp_path),
                "--profile",
                "metal-apple-gpu",
                "--work",
                str(tmp_path / "work"),
                "--filter",
                "Name=Other",
            ]
        )
    with pytest.raises(SystemExit, match="cannot be filtered out"):
        RUN_TESTS.run_tests(
            adapted=tmp_path,
            artifacts=tmp_path,
            target="metal",
            native_library=tmp_path / "lib",
            excluded=[ENTRY],
            results_dir=tmp_path / "trx",
            evidence=None,
            fault=None,
            extra_filter=None,
            log=tmp_path / "run.log",
            program=tmp_path,
        )


def test_discovery_exports_a_top_level_program_as_its_entry_point(
    program_tools, tmp_path
):
    discovery = tmp_path / "discovery"
    result = SUPPORT.export(program_tools, discovery)
    assert result.returncode == 0, result.stderr
    report = BASELINE.load_discovery(discovery)
    assert report["projectKind"] == "program"
    (method,) = report["methods"]
    assert method["method"] == ENTRY
    assert method["source"]["path"] == (
        "demos/integrations/computesharp/tests/program_fixture/Program.cs"
    )
    (case,) = report["tests"]
    assert case["displayName"] == "<Main>$" and case["arguments"] == []
    (shader,) = report["shaders"]
    assert shader["threads"] == [64, 1, 1]

    assembly = Path(program_tools[2])
    trx = PROGRAMS.run(
        assembly, discovery, tmp_path / "pass", log=tmp_path / "pass.log"
    )
    assert EVIDENCE.trx_cases(trx)[0]["outcome"] == "Passed"
    assert "program ProgramFixture" in (tmp_path / "pass.log").read_text()
    BASELINE.verify_results(discovery, trx, "translated", PROGRAM_CONTRACT)

    environment = {**os.environ, "PROGRAM_FIXTURE_MODE": "fail"}
    trx = PROGRAMS.run(
        assembly,
        discovery,
        tmp_path / "fail",
        log=tmp_path / "fail.log",
        env=environment,
    )
    (case,) = EVIDENCE.trx_cases(trx)
    assert case["outcome"] == "Failed"
    assert "does not match the expected results" in case["message"]
