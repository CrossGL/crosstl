"""Reference refreshes require exact coverage and new native compiler evidence."""

import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tools import refresh_artifact_contract as refresh

SOURCE = """shader Example {
    RWStructuredBuffer<float> outputBuffer;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void computeMain() @ stage_entry { outputBuffer[0] = 3.0; }
    }
}
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "source.cgl").write_bytes(SOURCE.encode("utf-8"))
    config = tmp_path / "crosstl.toml"
    config.write_text('[project]\nsource_roots = ["."]\ntargets = ["opengl"]\n')
    contract = {
        "schemaVersion": 1,
        "kind": "test-artifact-contract",
        "source": "source.cgl",
        "sourceSha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
        "target": "opengl",
        "artifactContract": {
            "artifactCount": 1,
            "generatedSizeBytesTotal": 1,
            "generatedSizeRange": {"minimum": {}, "maximum": {}},
        },
        "entries": [
            {
                "entryPoint": "computeMain",
                "sha256": "0" * 64,
                "sizeBytes": 1,
                "customCoverage": {"unchanged": True},
            }
        ],
        "proof": {"numericalGate": "must-remain-enabled"},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    compiler = tmp_path / "compiler.py"
    compiler.write_text(
        "from pathlib import Path\nimport sys\nPath(sys.argv[2]).write_bytes(b'compiled')\n"
    )
    return {
        "root": tmp_path,
        "config_path": config,
        "contract_path": path,
        "work": tmp_path / "audit",
        "compiler_command": [sys.executable, str(compiler), "{artifact}", "{output}"],
    }


def test_refresh_translates_through_public_project_api_and_preserves_contract(project):
    original = project["contract_path"].read_bytes()
    result = refresh.refresh(**project)
    assert result["status"] == "passed", result
    assert result["changedCount"] == 1
    assert result["numericalExecution"] is False
    assert result["fullUpstreamSuite"] is False
    assert project["contract_path"].read_bytes() == original
    candidate = json.loads(Path(result["candidate"]).read_text())
    expected = json.loads(original)
    expected["entries"][0].update(
        {key: candidate["entries"][0][key] for key in ("sha256", "sizeBytes")}
    )
    expected["artifactContract"] = candidate["artifactContract"]
    assert candidate == expected
    record = result["records"][0]
    assert record["returncode"] == 0 and record["status"] == "passed"
    assert record["compiledSha256"] == hashlib.sha256(b"compiled").hexdigest()
    assert (
        record["sha256"]
        == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    )


@pytest.fixture
def unavailable_default_toolchain(monkeypatch):
    from crosstl.project import pipeline

    monkeypatch.setattr(
        pipeline,
        "_tool_status",
        lambda target: {
            "target": target,
            "status": "unavailable",
            "tools": [{"name": "default-compiler", "available": False, "path": None}],
        },
    )


def test_explicit_compiler_replaces_unavailable_default(
    project, unavailable_default_toolchain
):
    result = refresh.refresh(**project)
    assert result["status"] == "passed", result
    assert result["records"][0]["status"] == "passed"
    report = json.loads(Path(result["portabilityReport"]).read_text())
    diagnostics = report["diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "project.validate.toolchain-unavailable"
    assert report["summary"]["diagnosticCounts"]["warning"] == 1
    assert result["defaultToolchainDiagnostics"] == diagnostics
    assert result["numericalExecution"] is False


@pytest.mark.parametrize("failure", ["missing", "nonzero", "empty"])
def test_unavailable_default_does_not_bypass_explicit_compiler_failure(
    project, unavailable_default_toolchain, failure
):
    if failure == "missing":
        project["compiler_command"][0] = str(project["root"] / "missing-compiler")
    else:
        Path(project["compiler_command"][1]).write_text(
            "raise SystemExit(4)\n" if failure == "nonzero" else "pass\n"
        )
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert len(result["defaultToolchainDiagnostics"]) == 1
    assert len(result["records"]) == 1
    assert result["records"][0]["status"] == "failed"
    assert result["records"][0]["error"]
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize(
    "code,severity,target,capabilities",
    [
        ("project.source.unsupported", "warning", "opengl", ["source.translation"]),
        (
            "project.validate.toolchain-unavailable",
            "error",
            "opengl",
            ["toolchain.validation"],
        ),
        (
            "project.validate.toolchain-unavailable",
            "warning",
            "vulkan",
            ["toolchain.validation"],
        ),
        (
            "project.validate.toolchain-unavailable",
            "warning",
            "opengl",
            ["source.translation"],
        ),
    ],
)
def test_explicit_compiler_does_not_suppress_other_diagnostics(
    project,
    unavailable_default_toolchain,
    monkeypatch,
    code,
    severity,
    target,
    capabilities,
):
    import crosstl.project
    from crosstl.project import pipeline

    translate = crosstl.project.translate_project

    def with_diagnostic(*args, **kwargs):
        report = translate(*args, **kwargs)
        report.diagnostics.append(
            pipeline.ProjectDiagnostic(
                severity=severity,
                code=code,
                message="Additional validation diagnostic",
                location=pipeline.SourceLocation(file="source.cgl"),
                target=target,
                missing_capabilities=capabilities,
            )
        )
        return report

    monkeypatch.setattr(crosstl.project, "translate_project", with_diagnostic)
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["records"] == []
    assert (
        result["failures"][-1]["error"] == "Translation reported warnings or failures"
    )
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize("jobs", [1, 2])
def test_complete_multi_entry_source_refreshes_in_serial_and_parallel(project, jobs):
    source = (
        "kernel void first(device float* result [[buffer(0)]]) { result[0] = 1.0; }\n"
        "kernel void second(device float* result [[buffer(0)]]) { result[0] = 2.0; }\n"
    )
    (project["root"] / "source.metal").write_bytes(source.encode("utf-8"))
    contract = json.loads(project["contract_path"].read_text())
    contract.update(
        source="source.metal", sourceSha256=hashlib.sha256(source.encode()).hexdigest()
    )
    first = contract["entries"][0]
    first["entryPoint"] = "first"
    second = deepcopy(first)
    second["entryPoint"] = "second"
    contract["entries"].append(second)
    contract["artifactContract"].update(
        artifactCount=2, hostDispatchWorkgroupSize=[1, 1, 1]
    )
    project["contract_path"].write_text(json.dumps(contract))
    result = refresh.refresh(**project, jobs=jobs)
    assert result["status"] == "passed", result
    assert {record["entryPoint"] for record in result["records"]} == {"first", "second"}
    assert len({record["compiledPath"] for record in result["records"]}) == 2
    candidate = json.loads(Path(result["candidate"]).read_text())
    assert [entry["entryPoint"] for entry in candidate["entries"]] == [
        "first",
        "second",
    ]
    assert len({entry["sha256"] for entry in candidate["entries"]}) == 2


def test_metal_template_roundtrip_retains_entry_launch_contract(project):
    source = (
        "template<typename T> kernel void write_value(device T* result [[buffer(0)]])"
        " { result[0] = T(1); }\n"
        'template [[host_name("write_float")]] kernel void write_value<float>(device float*);\n'
    )
    (project["root"] / "source.metal").write_bytes(source.encode("utf-8"))
    contract = json.loads(project["contract_path"].read_text())
    contract.update(
        source="source.metal",
        sourceSha256=hashlib.sha256(source.encode()).hexdigest(),
        target="metal",
    )
    contract["entries"][0]["entryPoint"] = "write_float"
    contract["artifactContract"].update(
        specializationCount=1, hostDispatchWorkgroupSize=[32, 1, 1]
    )
    project["contract_path"].write_text(json.dumps(contract))
    result = refresh.refresh(**project)
    assert result["status"] == "passed", result
    report = json.loads(Path(result["portabilityReport"]).read_text())
    execution = report["artifacts"][0]["execution"]
    assert execution["entryPoints"][0]["workgroupSize"] == [32, 1, 1]


def test_native_compiler_creates_candidate(project):
    if sys.platform == "darwin":
        target, executable = "metal", "xcrun"
        command = [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            "{artifact}",
            "-o",
            "{output}",
        ]
        magic = b"BC\xc0\xde"
    elif sys.platform == "win32":
        target, executable = "directx", "dxc"
        command = [
            "dxc",
            "-T",
            "cs_6_0",
            "-E",
            "{entry_point}",
            "-WX",
            "-Fo",
            "{output}",
            "{artifact}",
        ]
        magic = b"DXBC"
    else:
        target, executable = "opengl", "glslangValidator"
        command = [
            "glslangValidator",
            "-G",
            "-S",
            "comp",
            "-o",
            "{output}",
            "{artifact}",
        ]
        magic = b"\x03\x02\x23\x07"
    if shutil.which(executable) is None:
        message = f"{executable} is required for the native {target} refresh proof"
        if os.environ.get("CROSTL_REQUIRE_ARTIFACT_REFRESH_COMPILER") == "1":
            pytest.fail(message)
        pytest.skip(message)
    contract = json.loads(project["contract_path"].read_text())
    contract["target"] = target
    project["contract_path"].write_text(json.dumps(contract))
    project["compiler_command"] = command
    result = refresh.refresh(**project)
    assert result["status"] == "passed", result
    record = result["records"][0]
    assert record["returncode"] == 0
    binary = Path(record["compiledPath"]).read_bytes()
    # AIR may be wrapped bitcode rather than starting directly with LLVM's marker.
    assert binary.startswith(magic) or (
        target == "metal" and binary.startswith(b"\xde\xc0\x17\x0b")
    )
    assert Path(result["candidate"]).is_file()
    assert result["numericalExecution"] is False


@pytest.mark.parametrize(
    "program, error",
    [
        ("raise SystemExit(4)\n", "Native compiler failed"),
        ("pass\n", "Native compiler produced no nonempty output"),
        (
            "from pathlib import Path\nimport sys\nPath(sys.argv[2]).touch()\n",
            "Native compiler produced no nonempty output",
        ),
        (
            "from pathlib import Path\nimport sys\nPath(sys.argv[1]).write_text('changed')\nPath(sys.argv[2]).write_bytes(b'ok')\n",
            "Artifact changed during compilation",
        ),
    ],
)
def test_compiler_failure_never_refreshes_a_contract(project, program, error):
    Path(project["compiler_command"][1]).write_text(program)
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["records"][0]["error"] == error
    assert not (project["work"] / "candidate.json").exists()
    assert (project["work"] / "audit.json").is_file()


def test_compiler_timeout_is_retained_as_evidence(project):
    Path(project["compiler_command"][1]).write_text(
        "import sys, time\nprint('started', flush=True)\n"
        "print('waiting', file=sys.stderr, flush=True)\ntime.sleep(5)\n"
    )
    source = project["root"] / "source.cgl"
    record = {
        "entryPoint": "computeMain",
        "targetEntryPoint": "main",
        "path": str(source),
        "sha256": refresh._sha256(source),
    }
    result = refresh._compile(record, project["compiler_command"], project["work"], 0.5)
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
    assert result["stdout"] == "started\n"
    assert result["stderr"] == "waiting\n"
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize(
    "stdout,stderr,expected_stdout,expected_stderr",
    [
        (b"started\n", b"waiting\n", "started\n", "waiting\n"),
        ("started\n", "waiting\n", "started\n", "waiting\n"),
        (b"started\n", "waiting\n", "started\n", "waiting\n"),
        (None, None, "", ""),
        (b"invalid\xff", b"", "invalid\ufffd", ""),
    ],
)
def test_compiler_timeout_accepts_platform_output_types(
    project, monkeypatch, stdout, stderr, expected_stdout, expected_stderr
):
    def timeout(command, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "timeout": 1}
        raise refresh.subprocess.TimeoutExpired(
            command, 1, output=stdout, stderr=stderr
        )

    source = project["root"] / "source.cgl"
    record = {
        "entryPoint": "computeMain",
        "targetEntryPoint": "main",
        "path": str(source),
        "sha256": refresh._sha256(source),
    }
    monkeypatch.setattr(refresh.subprocess, "run", timeout)
    result = refresh._compile(record, project["compiler_command"], project["work"], 1)
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
    assert result["stdout"] == expected_stdout
    assert result["stderr"] == expected_stderr
    assert "compiledSha256" not in result
    evidence = list((project["work"] / "compiled").glob("*/evidence.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8")) == result


def test_source_newline_changes_are_not_accepted_as_pinned_input(project):
    (project["root"] / "source.cgl").write_bytes(
        SOURCE.replace("\n", "\r\n").encode("utf-8")
    )
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["failures"] == [{"error": "Pinned source hash differs"}]
    assert result["records"] == []
    assert not (project["work"] / "candidate.json").exists()


def test_source_mismatch_removes_previous_candidate(project):
    project["work"].mkdir()
    (project["work"] / "candidate.json").write_text("old")
    (project["root"] / "source.cgl").write_text("changed")
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["failures"] == [{"error": "Pinned source hash differs"}]
    assert not (project["work"] / "candidate.json").exists()


def test_repeated_refresh_still_requires_fresh_compiler_output(project):
    assert refresh.refresh(**project)["status"] == "passed"
    Path(project["compiler_command"][1]).write_text("pass\n")
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["records"], result
    assert (
        result["records"][0]["error"] == "Native compiler produced no nonempty output"
    )
    assert not (project["work"] / "candidate.json").exists()


def test_completed_checkpoint_cannot_be_reused_as_new_evidence(project):
    assert refresh.refresh(**project)["status"] == "passed"
    result = refresh.refresh(**project, resume=True)
    assert result["status"] == "failed"
    assert "already-complete" in result["failures"][0]["error"]
    assert result["records"] == []
    assert not (project["work"] / "candidate.json").exists()


def test_worker_failure_retains_failed_audit(project, monkeypatch):
    import crosstl.project

    def fail(*args, **kwargs):
        raise RuntimeError("Translation worker stopped")

    monkeypatch.setattr(crosstl.project, "translate_project", fail)
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["failures"] == [{"error": "Translation worker stopped"}]
    assert (
        json.loads((project["work"] / "audit.json").read_text())["status"] == "failed"
    )
    assert not (project["work"] / "candidate.json").exists()


def test_interruption_records_state_without_publishing_candidate(project, monkeypatch):
    import crosstl.project

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(crosstl.project, "translate_project", interrupt)
    project["work"].mkdir()
    (project["work"] / "candidate.json").write_text("previous")
    with pytest.raises(KeyboardInterrupt):
        refresh.refresh(**project)
    audit = json.loads((project["work"] / "audit.json").read_text())
    assert audit["status"] == "interrupted"
    assert audit["failures"] == [{"error": "Audit interrupted"}]
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("entries", ["missing"]),
        ("entries", ["computeMain", "computeMain"]),
        ("jobs", 0),
    ],
)
def test_invalid_selection_or_workers_fail_before_translation(project, key, value):
    with pytest.raises(ValueError):
        refresh.refresh(**project, **{key: value})


@pytest.mark.parametrize(
    "field,value",
    [("artifactCount", 2), ("specializationCount", 7), ("reflectedResourceCount", 2)],
)
def test_aggregate_drift_is_not_blessed(project, field, value):
    contract = json.loads(project["contract_path"].read_text())
    contract["artifactContract"][field] = value
    project["contract_path"].write_text(json.dumps(contract))
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["failures"][-1]["error"] == f"Aggregate {field} differs"
    assert not (project["work"] / "candidate.json").exists()


def test_resource_drift_is_not_blessed(project):
    contract = json.loads(project["contract_path"].read_text())
    contract["entries"][0]["resources"] = []
    project["contract_path"].write_text(json.dumps(contract))
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["failures"][0] == {
        "entryPoint": "computeMain",
        "error": "Resource ABI differs",
    }
    assert result["records"] == []


def test_partial_audit_retains_evidence_without_a_candidate(project):
    contract = json.loads(project["contract_path"].read_text())
    second = deepcopy(contract["entries"][0])
    second["entryPoint"] = "other"
    contract["entries"].append(second)
    project["contract_path"].write_text(json.dumps(contract))
    result = refresh.refresh(**project, entries=["computeMain"])
    assert result["status"] == "failed"
    assert result["records"][0]["status"] == "passed"
    assert result["expectedCount"] == 2 and result["selectedCount"] == 1
    assert (
        result["failures"][-1]["error"] == "Incomplete or duplicate artifact coverage"
    )
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize(
    "command", [[], ["tool"], ["tool", "{artifact}"], "not-an-array"]
)
def test_compiler_command_requires_explicit_inputs_and_output(project, command):
    project["compiler_command"] = command
    with pytest.raises(ValueError, match="Compiler command"):
        refresh.refresh(**project)


def test_output_must_remain_below_project_root(project):
    project["work"] = project["root"].parent / "outside"
    with pytest.raises(ValueError, match="Path escapes project root"):
        refresh.refresh(**project)


@pytest.mark.parametrize("key", ["contract_path", "config_path"])
def test_work_directory_cannot_overwrite_inputs(project, key):
    original = project[key].read_bytes()
    project["work"].mkdir()
    relocated = project["work"] / "audit.json"
    relocated.write_bytes(original)
    project[key] = relocated
    with pytest.raises(ValueError, match="must be outside the work directory"):
        refresh.refresh(**project)
    assert relocated.read_bytes() == original


def test_work_directory_cannot_contain_the_pinned_source(project):
    project["work"].mkdir()
    source = project["work"] / "audit.json"
    source.write_text(SOURCE)
    contract = json.loads(project["contract_path"].read_text())
    contract["source"] = "audit/audit.json"
    project["contract_path"].write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="Pinned source must be outside"):
        refresh.refresh(**project)
    assert source.read_text() == SOURCE


@pytest.mark.parametrize("key", ["contract_path", "config_path", "source"])
def test_inputs_changed_during_compilation_cannot_refresh_contract(project, key):
    path = project["root"] / "source.cgl" if key == "source" else project[key]
    Path(project["compiler_command"][1]).write_text(
        "from pathlib import Path\nimport sys\n"
        "Path(sys.argv[2]).write_bytes(b'compiled')\n"
        f"Path({str(path)!r}).write_text('changed')\n"
    )
    result = refresh.refresh(**project)
    assert result["status"] == "failed"
    assert result["records"][0]["status"] == "passed"
    assert result["failures"][-1]["error"] == (
        "Pinned source hash differs"
        if key == "source"
        else "Input contract or configuration changed during the audit"
    )
    assert not (project["work"] / "candidate.json").exists()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_timeout_requires_positive_finite_number(project, value):
    with pytest.raises(ValueError, match="positive and finite"):
        refresh.refresh(**project, timeout=value)


def test_canonical_legacy_hash_styles_are_explicit():
    value = {"a": 1}
    for data in (b'{"a":1}', b'{"a":1}\n'):
        assert refresh._matches_json_hash(value, hashlib.sha256(data).hexdigest())
    assert not refresh._matches_json_hash(
        {"a": 2}, hashlib.sha256(b'{"a":1}').hexdigest()
    )
