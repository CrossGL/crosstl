"""Padding payload, addressing, unchanged upstream tests and native CI contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import packages
from demos.integrations.mlx.portable_host import padding_workloads as workloads
from demos.integrations.mlx.portable_host import verify_padding as proof
from demos.integrations.mlx.portable_host.runtime import (
    BOOLEAN_GUARD,
    COPY_GUARD,
    DISPATCH_VERSION,
)


def evidence(target="metal", *, native=True):
    records, trace = [], []
    for case in workloads.cases():
        base, value, _ = workloads.inputs(np, case)
        expected, operations = workloads.stages(np, case)
        operations = operations if native else []
        records.append(
            {
                **case,
                "inputPayloads": [
                    workloads.payload(np, item) for item in (base, value)
                ],
                "inputUnchanged": True,
                "resultPayload": workloads.payload(np, expected),
                "resultShape": list(expected.shape),
                "resultDtype": expected.dtype.name,
                "dispatchCount": len(operations),
            }
        )
        dtype = (
            case["dtype"] if case["dtype"] in {"bool_", "int64", "uint64"} else "uint32"
        )
        for operation in operations:
            source, _, _, _, output = operation
            info = workloads.metadata(*operation)
            values = workloads.words(np, output)
            guard = BOOLEAN_GUARD if dtype == "bool_" else COPY_GUARD
            if dtype == "bool_" and target != "metal":
                values, guard = [int(v) for v in values], [int(v) for v in guard]
            trace.append(
                {
                    "entry": f"ggn2_dynamic_copy{dtype}{dtype}",
                    "target": target,
                    "dispatchVersion": DISPATCH_VERSION,
                    "threads": source.size,
                    "workgroupCount": info["workgroupCount"],
                    "workgroupSize": [1, 1, 1],
                    "copyMetadata": info,
                    "copyValues": values,
                    "copyGuardWords": guard,
                }
            )
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("native", [True, False])
def test_complete_padding_cases(target, native):
    records, trace = evidence(target, native=native)
    assert len(records) == 102
    assert {item["dtype"] for item in records} == set(workloads.DTYPES)
    assert any(item["resultShape"] == [255, 257] for item in records)
    workloads.validate(records, trace, native=native)


@pytest.fixture
def small_evidence(monkeypatch):
    selected = [case for case in workloads.cases() if case["layout"] != "limit"]
    monkeypatch.setattr(workloads, "cases", lambda: iter(selected))
    return evidence()


@pytest.mark.parametrize(
    "fault",
    [
        "missing-case",
        "extra-case",
        "input",
        "result",
        "dtype",
        "shape",
        "preserved",
        "count",
        "bool-count",
        "missing-trace",
        "extra-trace",
        "entry",
        "target",
        "abi",
        "launch",
        "threads",
        "words",
        "guard",
        "offset",
        "stride",
        "preserve-output",
        "wide-word",
        "boolean-word",
    ],
)
def test_padding_rejects_corrupt_evidence(small_evidence, fault):
    records, trace = small_evidence
    if fault == "missing-case":
        records.pop()
    elif fault == "extra-case":
        records.append(records[-1])
    elif fault == "input":
        records[0]["inputPayloads"][0] = ""
    elif fault == "result":
        records[0]["resultPayload"] = ""
    elif fault == "dtype":
        records[0]["resultDtype"] = "float16"
    elif fault == "shape":
        records[0]["resultShape"] = [54]
    elif fault == "preserved":
        records[0]["inputUnchanged"] = False
    elif fault == "count":
        records[0]["dispatchCount"] = 0
    elif fault == "bool-count":
        records[0]["dispatchCount"] = True
    elif fault == "missing-trace":
        trace.pop()
    elif fault == "extra-trace":
        trace.append(trace[-1])
    elif fault == "entry":
        trace[0]["entry"] = "copy"
    elif fault == "target":
        trace[0]["target"] = "cpu"
    elif fault == "abi":
        trace[0]["dispatchVersion"] = 1
    elif fault == "launch":
        trace[0]["workgroupCount"] = [1, 1, 1]
    elif fault == "threads":
        trace[0]["threads"] = 1
    elif fault == "words":
        trace[1]["copyValues"][0] ^= 1
    elif fault == "guard":
        trace[0]["copyGuardWords"] = []
    elif fault == "offset":
        trace[1]["copyMetadata"]["destinationOffset"] = 0
    elif fault == "stride":
        trace[1]["copyMetadata"]["destinationStrides"][0] += 1
    elif fault == "preserve-output":
        trace[1]["copyMetadata"]["preserveDestination"] = False
    elif fault == "wide-word":
        event = next(e for e in trace if e["entry"].endswith("copyint64int64"))
        event["copyValues"][0] = float(event["copyValues"][0])
    else:
        event = next(e for e in trace if e["entry"].endswith("copybool_bool_"))
        event["copyValues"][0] = int(event["copyValues"][0])
    with pytest.raises(ValueError, match="Padding"):
        workloads.validate(records, trace, native=True)


def test_copy_references_preserve_alias_and_signed_destination():
    base, update, region = workloads.inputs(
        np, {"dtype": "uint64", "operation": "update", "layout": "alias"}
    )
    assert np.shares_memory(base, update)
    expected = base.copy()
    expected[region] = update.copy()
    output, operations = workloads.stages(
        np, {"dtype": "uint64", "operation": "update", "layout": "alias"}
    )
    assert np.array_equal(output, expected)
    assert operations[1][3] is True
    _, operations = workloads.stages(
        np,
        {"dtype": "float32", "operation": "update", "layout": "negative-destination"},
    )
    assert operations[1][1] == [-12, -2]
    assert operations[1][2] == 23


def test_padding_ci_requires_native_execution_and_retained_evidence():
    import re

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["integer64-host"]
    assert job["needs"] == "portable-host"
    assert "if" not in job and "continue-on-error" not in job
    assert {
        (item["target"], item["os"]) for item in job["strategy"]["matrix"]["include"]
    } == {
        ("metal", "macos-26"),
        ("directx", "windows-2025"),
        ("opengl", "ubuntu-24.04"),
    }
    steps = {step.get("name"): step for step in job["steps"]}
    companion = steps["Translate padding test companion"]
    execute = steps["Execute padding and slice updates"]
    for step in (companion, execute):
        assert "if" not in step and "continue-on-error" not in step
        assert "set -euo pipefail" in step["run"]
        assert "|| true" not in step["run"]
    assert "--entry all_reduce_andbool_ --width 64" in companion["run"]
    assert "matrix.target" in companion["run"]
    assert "portable_host.verify_padding" in execute["run"]
    assert "--integer64" in execute["run"] and execute["run"].count("--reductions") == 2
    deadlines = [
        int(v)
        for step in job["steps"]
        for v in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
    retained = steps["Retain integer64 execution evidence"]
    assert retained["if"] == "always()"
    assert retained["with"]["include-hidden-files"] is True
    assert retained["with"]["if-no-files-found"] == "error"
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        assert "tests/test_mlx_portable_padding.py" in triggers[event]["paths"]
    assert any(
        "tests/test_mlx_portable_padding.py" in step.get("run", "")
        for step in workflow["jobs"]["portable-host"]["steps"]
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "skip",
        "failure",
        "error",
        "name",
        "no-dispatch",
        "workload-count",
    ],
)
def test_padding_requires_all_unchanged_upstream_tests(native, fault):
    data = {
        "tests": [
            {
                "test": name,
                "testsRun": 1,
                "failures": 0,
                "errors": 0,
                "skips": 0,
                "dispatchCount": 1 if native else 0,
            }
            for name in proof.UPSTREAM_TESTS
        ],
        "workloadDispatchCount": 100 if native else 0,
    }
    if fault == "missing":
        data["tests"].pop()
    elif fault in {"skip", "failure", "error"}:
        data["tests"][0][
            {"skip": "skips", "failure": "failures", "error": "errors"}[fault]
        ] = 1
    elif fault == "name":
        data["tests"][0]["test"] = "other"
    elif fault == "no-dispatch":
        data["tests"][0]["dispatchCount"] = 0 if native else 1
    elif fault == "workload-count":
        data["workloadDispatchCount"] = False
    if fault:
        with pytest.raises(ValueError, match="Padding"):
            proof.validate_upstream(data, native=native)
    else:
        proof.validate_upstream(data, native=native)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "worker",
        "missing-rejection",
        "negative-dispatch",
        "unknown-artifact",
        "sources-changed",
        "adaptation-changed",
    ],
)
def test_padding_parent_checks_workers_sources_and_artifacts(
    tmp_path, monkeypatch, small_evidence, fault
):
    args = SimpleNamespace(mlx_root=tmp_path / "mlx", output_dir=tmp_path / "proof")
    for family, entries in (
        ("packages", packages.ENTRIES),
        ("integer64", packages.INTEGER64_ENTRIES),
        ("reductions", ("w32/all_reduce_andbool_", "w64/all_reduce_andbool_")),
    ):
        directory = tmp_path / family
        directory.mkdir()
        setattr(args, family, directory)
        proof.write_json(
            directory / "index.json",
            {"target": "metal", "descriptors": {entry: {} for entry in entries}},
        )
    args.reductions = [args.reductions]
    snapshots, sources = [], []

    def snapshot(root):
        snapshots.append(root)
        return {
            "hash": (
                "changed"
                if fault == "adaptation-changed" and len(snapshots) > 1
                else "original"
            )
        }

    def upstream_sources(root):
        sources.append(root)
        return {
            "test_ops.py": (
                "changed"
                if fault == "sources-changed" and len(sources) > 1
                else "original"
            )
        }

    monkeypatch.setattr(proof, "verify_prepared", snapshot)
    monkeypatch.setattr(proof, "upstream_test_sources", upstream_sources)
    monkeypatch.setattr(
        proof,
        "load_index",
        lambda directory, target: json.loads((directory / "index.json").read_text()),
    )
    identity, artifacts, commands = [], [], []
    monkeypatch.setattr(
        proof, "verify_native_identity", lambda trace, target: identity.extend(trace)
    )
    monkeypatch.setattr(
        proof,
        "verify_artifacts",
        lambda trace, *args, **kwargs: artifacts.extend(trace),
    )

    def execute(command, **kwargs):
        mode = command[command.index("--worker") + 1]
        commands.append(command)
        directory = Path(command[command.index("--output-dir") + 1])
        directory.mkdir()
        if fault == "worker":
            return SimpleNamespace(returncode=1)
        if mode in proof.NEGATIVE_CHECKS:
            proof.write_json(
                directory / "rejection.json",
                {
                    "check": mode,
                    "error": (
                        "wrong"
                        if fault == "missing-rejection"
                        else proof.NEGATIVE_CHECKS[mode]
                    ),
                    "dispatchCount": int(fault == "negative-dispatch"),
                },
            )
            return SimpleNamespace(returncode=0)
        records, trace = copy.deepcopy(small_evidence)
        workload_count = len(trace)
        if mode == "cpu":
            for record in records:
                record["dispatchCount"] = 0
        proof.write_json(directory / "results.json", records)
        proof.write_json(
            directory / "upstream.json",
            {
                "tests": [
                    {
                        "test": name,
                        "testsRun": 1,
                        "failures": 0,
                        "errors": 0,
                        "skips": 0,
                        "dispatchCount": int(mode == "native"),
                    }
                    for name in proof.UPSTREAM_TESTS
                ],
                "workloadDispatchCount": workload_count if mode == "native" else 0,
            },
        )
        if mode == "native":
            trace.extend(
                [{"entry": packages.COPY_ENTRY, "workgroupSize": [1, 1, 1]}] * 3
            )
            if fault == "unknown-artifact":
                trace[-1] = {"entry": "other", "workgroupSize": [1, 1, 1]}
            (directory / "dispatch.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in trace)
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(proof.subprocess, "run", execute)
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            proof.verify(args)
        assert not (args.output_dir / "evidence.json").exists()
    else:
        result = proof.verify(args)
        assert result["casesPerPath"] == len(small_evidence[0])
        assert (
            result["fullUpstreamSuite"] is False
            and result["fullTranslatedBackend"] is False
        )
        assert len(identity) == len(artifacts) == result["dispatchCount"]
        assert len(snapshots) == len(sources) == 2
        assert len(commands) == 2 + len(proof.NEGATIVE_CHECKS)
        assert all(command.count("--reductions") == 1 for command in commands)
