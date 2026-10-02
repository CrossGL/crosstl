"""Mixed MLX reductions preserve package ownership and native dependency order."""

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import mixed_reduction_workloads as workloads
from demos.integrations.mlx.portable_host import runtime
from demos.integrations.mlx.portable_host.packages import BOOLEAN_COPY_ENTRY, COPY_ENTRY


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "duplicate",
        "duplicate-root",
        "target",
        "missing",
        "empty",
        "implicit",
        "single",
        "shard",
    ],
)
def test_multiple_reduction_packages_keep_owner_directories(
    tmp_path, monkeypatch, target, fault
):
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.json").write_text(
        json.dumps(
            {"target": target, "descriptors": dict.fromkeys(runtime.ENTRIES, {})}
        )
    )
    directories = [tmp_path / "all", tmp_path / "row"]
    entries = ["all_reduce_sumfloat32", "row_reduce_simple_sumfloat32"]
    for directory, entry, family in zip(directories, entries, ("all", "row")):
        directory.mkdir()
        (directory / "index.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "family": family,
                    "widths": [32],
                    "entries": [entry],
                    "descriptors": {"w32/" + entry: {"target": target}},
                }
            )
        )
    for name in (
        "OpenGLComputeRuntime",
        "OpenGLRuntimeParityAdapter",
        "DirectXComputeRuntime",
        "DirectXRuntimeParityAdapter",
        "MetalRuntimeParityAdapter",
        "RuntimeParityExecutor",
    ):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: SimpleNamespace())
    if fault == "duplicate":
        directories.append(directories[0])
    elif fault == "duplicate-root":
        other = tmp_path / "duplicate"
        other.mkdir()
        (other / "index.json").write_bytes((directories[0] / "index.json").read_bytes())
        directories.append(other)
    elif fault == "single":
        directories = directories[:1]
    elif fault == "shard":
        index = json.loads((directories[0] / "index.json").read_text())
        index["widths"] = [64]
        index["descriptors"] = {"w64/" + entries[0]: {"target": target}}
        (directories[1] / "index.json").write_text(json.dumps(index))
    elif fault == "target":
        index = json.loads((directories[1] / "index.json").read_text())
        index["target"] = "other"
        (directories[1] / "index.json").write_text(json.dumps(index))
    elif fault == "missing":
        directories.append(tmp_path / "missing")
    elif fault == "empty":
        directories = []
    if fault not in {None, "implicit", "single", "shard"}:
        with pytest.raises(ValueError):
            runtime.HostRuntime(base, tmp_path / "trace", reductions=directories)
    else:
        host = runtime.HostRuntime(
            base,
            tmp_path / "trace",
            reductions=(
                None
                if fault == "implicit"
                else directories[0] if fault == "single" else directories
            ),
        )
        expected = (
            {}
            if fault == "implicit"
            else {
                "w32/" + entry: directory
                for entry, directory in zip(entries, directories)
            }
        )
        if fault == "shard":
            expected = {
                "w32/" + entries[0]: directories[0],
                "w64/" + entries[0]: directories[1],
            }
        assert host.reduction_directories == expected
        assert host.reduction_descriptors.keys() == expected.keys()


@pytest.fixture(scope="module")
def evidence():
    import numpy as np

    records, trace = [], []
    for case in workloads.cases():
        _, seed, rows, final = workloads.reference(np, case)
        start = len(trace)
        dtype = case["dtype"]
        guard = (
            [index % 2 == 0 for index in range(32)]
            if dtype == "bool_"
            else [
                (
                    struct.unpack("=f", struct.pack("=I", 0x6A15BEEF))[0]
                    if dtype == "float32"
                    else 0x6A15BEEF
                )
            ]
            * 32
        )

        def reduction(entry, size, groups, values):
            return {
                "entry": entry,
                "target": "metal",
                "threads": size,
                "dispatchVersion": 3,
                "workgroupSize": [32, 1, 1],
                "workgroupCount": groups,
                "reductionValues": values,
                "reductionGuardValues": guard.copy(),
            }

        if seed is not None:
            trace.append(reduction(case["entry"], 127, [1, 1, 1], [seed.item()]))
            trace.append(
                {
                    "entry": BOOLEAN_COPY_ENTRY if dtype == "bool_" else COPY_ENTRY,
                    "target": "metal",
                    "threads": 2145,
                    "dispatchVersion": 3,
                    "workgroupSize": [1, 1, 1],
                    "workgroupCount": [33, 33, 1],
                    "copyGuardWords": (
                        [index % 2 == 0 for index in range(32)]
                        if dtype == "bool_"
                        else [0x6A15BEEF] * 32
                    ),
                }
            )
        trace.extend(
            [
                reduction(case["rowEntry"], 2145, [1, 9, 1], rows.tolist()),
                reduction(case["entry"], 33, [1, 1, 1], [final.item()]),
            ]
        )
        records.append(
            case
            | {
                "rows": rows.tolist(),
                "final": final.item(),
                "seed": seed.item() if seed is not None else None,
                "resultDtype": "mlx.core." + dtype.removesuffix("_"),
                "resultShape": [],
                "dispatchStart": start,
                "dispatchEnd": len(trace),
            }
        )
    return records, trace


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing-case",
        "duplicate-case",
        "rows",
        "final",
        "seed",
        "dtype",
        "shape",
        "interval",
        "reorder",
        "missing-copy",
        "copy-guard",
        "reduction-guard",
        "readback",
        "width",
        "float-width",
        "threads",
        "version",
        "trailing",
    ],
)
def test_mixed_evidence_requires_every_native_stage(evidence, fault):
    records, trace = copy.deepcopy(evidence)
    if fault == "missing-case":
        records.pop()
    elif fault == "duplicate-case":
        records[-1] = records[0]
    elif fault == "rows":
        records[0]["rows"][0] += 1
    elif fault == "final":
        records[0]["final"] += 1
    elif fault == "seed":
        records[1]["seed"] += 1
    elif fault == "dtype":
        records[0]["resultDtype"] = "mlx.core.bool"
    elif fault == "shape":
        records[0]["resultShape"] = [1]
    elif fault == "interval":
        records[1]["dispatchStart"] = 0
    elif fault == "reorder":
        trace[0], trace[1] = trace[1], trace[0]
    elif fault == "missing-copy":
        trace.pop(3)
    elif fault == "copy-guard":
        trace[3]["copyGuardWords"][0] = 0
    elif fault == "reduction-guard":
        trace[0]["reductionGuardValues"][0] = 0
    elif fault == "readback":
        trace[0]["reductionValues"][0] += 1
    elif fault == "width":
        trace[0]["workgroupSize"][0] = 64
    elif fault == "float-width":
        trace[0]["workgroupSize"][0] = 32.0
    elif fault == "threads":
        trace[0]["threads"] -= 1
    elif fault == "version":
        trace[0]["dispatchVersion"] = 2.0
    elif fault == "trailing":
        trace.append(trace[-1])
    if fault:
        with pytest.raises(ValueError):
            workloads.validate(records, trace=trace)
    else:
        workloads.validate(records, trace=trace)
        assert len(records) == 28 and len(trace) == 84


def test_mixed_trace_can_follow_other_native_workloads(evidence):
    records, trace = copy.deepcopy(evidence)
    for record in records:
        record["dispatchStart"] += 2
        record["dispatchEnd"] += 2
    workloads.validate(records, trace=[{}, {}] + trace, offset=2)


def test_mixed_ci_requires_companions_without_reducing_standalone_coverage():
    import yaml

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/mlx-portable-host.yml"
    ).read_text()
    for event in ("push", "pull_request"):
        assert (
            "tests/test_mlx_portable_mixed_reductions.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
    job = yaml.safe_load(workflow)["jobs"]["reductions"]
    cases = job["strategy"]["matrix"]["include"]
    assert {case["target"] for case in cases if case["family"] == "row"} == {
        "metal",
        "directx",
        "opengl",
    }
    assert all(
        case["companion_args"] == "--all-reductions .mlx-portable-reductions/companions"
        for case in cases
        if case["family"] == "row"
    )
    steps = {step.get("name"): step for step in job["steps"]}
    translation = steps["Translate reduction launch variants"]["run"]
    assert "--width" not in translation and "--entry" not in translation
    companions = steps["Translate whole-array companions for mixed workloads"]
    assert companions["if"] == "matrix.family == 'row'"
    assert (
        "--family all --width 32" in companions["run"]
        and "--entry" not in companions["run"]
    )
    assert (
        not companions.get("continue-on-error")
        and "set -euo pipefail" in companions["run"]
    )
    execution = steps["Execute MLX reductions"]
    assert "${{ matrix.companion_args }}" in execution["run"] and "if" not in execution
