"""Selection package, callback, source-layout and native evidence contracts."""

import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import (
    absolute_workloads,
    packages,
    prepare,
    runtime,
)
from demos.integrations.mlx.portable_host import selection_workloads as workloads
from demos.integrations.mlx.portable_host import verify_selection as proof


@pytest.fixture(scope="module", params=["metal", "opengl", "directx"])
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("selection-" + request.param)
    source = root / packages.SELECTION_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(
        """template<typename T> kernel void choose(
device const bool* a [[buffer(0)]], device const T* b [[buffer(1)]],
device const T* c [[buffer(2)]], device T* d [[buffer(3)]],
constant uint& size [[buffer(4)]], uint index [[thread_position_in_grid]]) {
if (index < size) d[index] = a[index] ? b[index] : c[index]; }
"""
        + "".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(choose<{metal_type}>) choose<{metal_type}>;\n'
            for (entry, dtype) in packages.SELECTION_ENTRIES.items()
            for metal_type in [
                {"float32": "float", "int32": "int", "uint32": "uint", "bool_": "bool"}[
                    dtype
                ]
            ]
        )
    )
    original = source.read_bytes()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "selection", request.param, family="selection"
        )
    assert source.read_bytes() == original
    assert index["family"] == "selection"
    assert set(index["descriptors"]) == set(packages.SELECTION_ENTRIES)
    assert len(packages.ENTRIES) == 93
    return root / "selection", index


@pytest.fixture
def host(translated, tmp_path):
    directory, index = translated
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.json").write_text(
        json.dumps(
            {
                "target": index["target"],
                "descriptors": {
                    entry: {"target": index["target"]} for entry in packages.ENTRIES
                },
            }
        )
    )
    return runtime.HostRuntime(base, tmp_path / "trace", selection=directory)


@pytest.fixture(scope="module")
def translated_absolute(translated):
    directory, selected = translated
    root = directory.parent
    source = root / packages.UNARY_SOURCE
    source.write_text(
        """template<typename T> kernel void absolute_signature(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) out[index] = in[index]; }
"""
        + "".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(absolute_signature<{metal_type}>) absolute_signature<{metal_type}>;\n'
            for entry, dtype in packages.ABSOLUTE_ENTRIES.items()
            for metal_type in [
                {"int32": "int", "uint32": "uint", "bool_": "bool"}[dtype]
            ]
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "absolute", selected["target"], family="absolute"
        )
    assert set(index["descriptors"]) == set(packages.ABSOLUTE_ENTRIES)
    return root / "absolute", index


@pytest.mark.parametrize("entry", packages.ABSOLUTE_ENTRIES)
@pytest.mark.parametrize("donated", [False, True])
@pytest.mark.parametrize(
    "fault", [None, "dtype", "size", "condition", "guard", "readback-range"]
)
def test_absolute_callback_contract(
    host, translated_absolute, monkeypatch, entry, donated, fault
):
    directory, index = translated_absolute
    host = runtime.HostRuntime(host.directory, host.trace, absolute=directory)
    dtype = packages.ABSOLUTE_ENTRIES[entry]
    ctype = runtime.TYPES[dtype]
    source = (ctype * 3)(0, -1 if dtype == "int32" else 1, 1)
    destination = source if donated else (ctype * 3)()
    size = ctypes.c_uint32(3)
    buffers = (runtime.Buffer * 3)(
        runtime.Buffer(b"in", dtype.encode(), ctypes.addressof(source), 3, 0),
        runtime.Buffer(b"out", dtype.encode(), ctypes.addressof(destination), 3, 1),
        runtime.Buffer(b"size", b"uint32", ctypes.addressof(size), 1, 0),
    )
    if fault == "dtype":
        buffers[0].dtype = b"float32"
    elif fault == "size":
        size.value = 2
    elif fault == "condition":
        buffers[0].output = 2
    storage = runtime.physical_dtype(dtype, host.target)
    values = [False, True, True] if storage == "bool" else [0, 1, 1]
    guard = (
        list(runtime.BOOLEAN_GUARD)
        if storage == "bool"
        else (
            [int(v) for v in runtime.BOOLEAN_GUARD]
            if dtype == "bool_"
            else list(runtime.COPY_GUARD)
        )
    )
    output = values + guard
    if fault == "guard":
        output[-1] = not guard[-1] if storage == "bool" else 0 if guard[-1] else 1
    elif fault == "readback-range":
        output[0] = 2 if dtype == "bool_" else 2**32
    calls = []

    def execute(request):
        calls.append(request)
        binding = next(
            item["name"]
            for item in index["descriptors"][entry]["bindings"]
            if item["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={binding: {"dtype": storage, "shape": [35], "values": output}},
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, 3, 3)
    else:
        host.dispatch(entry, buffers, 3, 3)
        assert list(destination) == values
        event = json.loads(host.trace.read_text())
        assert event["absoluteValues"] == values
        assert event["unaryGuardValues"] == guard
    assert len(calls) == int(fault in {None, "guard", "readback-range"})


@pytest.mark.parametrize("family", ["selection", "absolute"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "extra",
        "family",
        "target",
        "descriptor-target",
        "descriptor-value",
        "not-object",
        "descriptors-not-object",
    ],
)
def test_optional_selection_index(
    host, translated, translated_absolute, tmp_path, family, fault
):
    index = copy.deepcopy(
        (translated if family == "selection" else translated_absolute)[1]
    )
    entries = (
        packages.SELECTION_ENTRIES
        if family == "selection"
        else packages.ABSOLUTE_ENTRIES
    )
    if fault == "missing":
        index["descriptors"].pop(next(iter(index["descriptors"])))
    elif fault == "extra":
        index["descriptors"]["extra"] = {"target": host.target}
    elif fault in {"family", "target"}:
        index[fault] = "wrong"
    elif fault == "descriptor-target":
        next(iter(index["descriptors"].values()))["target"] = "wrong"
    elif fault == "descriptor-value":
        index["descriptors"][next(iter(index["descriptors"]))] = None
    elif fault == "not-object":
        index = []
    elif fault == "descriptors-not-object":
        index["descriptors"] = None
    directory = tmp_path / "optional"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps(index))
    if fault:
        with pytest.raises(ValueError, match=f"{family.title()} packages"):
            runtime.HostRuntime(
                host.directory, tmp_path / "other", **{family: directory}
            )
    else:
        loaded = runtime.HostRuntime(
            host.directory, tmp_path / "other", **{family: directory}
        )
        assert set(loaded.descriptors) == set(packages.ENTRIES) | set(entries)
    legacy = runtime.HostRuntime(host.directory, tmp_path / "legacy")
    with pytest.raises(ValueError, match="No translated package"):
        legacy.dispatch(next(iter(entries)), None, 0, 1)


@pytest.mark.parametrize("entry", packages.SELECTION_ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "condition-dtype",
        "value-dtype",
        "direction",
        "count",
        "null",
        "size",
        "condition-value",
        "guard",
        "readback-type",
        "readback-range",
    ],
)
def test_selection_callback_contract(host, monkeypatch, entry, fault):
    dtype = packages.SELECTION_ENTRIES[entry]
    ctype = runtime.TYPES[dtype]
    left, right = ([1, 0, 1], [0, 1, 0]) if dtype == "bool_" else ([1, 2, 3], [4, 5, 6])
    memory = [
        (ctypes.c_uint8 * 3)(1, 0, 1),
        (ctype * 3)(*left),
        (ctype * 3)(*right),
        (ctype * 3)(),
        ctypes.c_uint32(3),
    ]
    buffers = (runtime.Buffer * 5)(
        *[
            runtime.Buffer(
                name.encode(),
                (
                    "uint32" if name == "size" else "bool_" if name == "a" else dtype
                ).encode(),
                ctypes.addressof(value),
                1 if name == "size" else 3,
                int(name == "d"),
            )
            for name, value in zip(("a", "b", "c", "d", "size"), memory)
        ]
    )
    if fault == "condition-dtype":
        buffers[0].dtype = b"uint32"
    elif fault == "value-dtype":
        buffers[1].dtype = b"int64"
    elif fault == "direction":
        buffers[1].output = 1
    elif fault == "count":
        buffers[0].count = 2
    elif fault == "null":
        buffers[0].data = 0
    elif fault == "size":
        memory[-1].value = 2
    elif fault == "condition-value":
        memory[0][0] = 2
    storage = runtime.physical_dtype(dtype, host.target)
    guard = list(runtime.COPY_GUARD)
    if dtype == "float32":
        guard = np.asarray(guard, dtype="uint32").view("float32").tolist()
    elif dtype == "bool_":
        guard = (
            list(runtime.BOOLEAN_GUARD)
            if storage == "bool"
            else [int(v) for v in runtime.BOOLEAN_GUARD]
        )
    values = [left[0], right[1], left[2]]
    if storage == "bool":
        values = [bool(v) for v in values]
    actual = values + guard
    if fault == "guard":
        actual[-1] = not guard[-1] if storage == "bool" else 0 if guard[-1] else 1
    elif fault == "readback-type":
        actual[0] = "invalid" if dtype == "float32" else 1 if storage == "bool" else 1.0
    elif fault == "readback-range":
        actual[0] = (
            "invalid" if dtype == "float32" else 2 if dtype == "bool_" else 2**32
        )
    calls = []

    def execute(request):
        calls.append(request)
        assert request.artifact_path.is_relative_to(host.selection_directory)
        binding = next(
            item["name"]
            for item in host.descriptors[entry]["bindings"]
            if item["access"] == "read_write"
        )
        return SimpleNamespace(
            status="ok",
            outputs={binding: {"dtype": storage, "shape": [35], "values": actual}},
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, 5, 3)
    else:
        host.dispatch(entry, buffers, 5, 3)
        assert list(memory[3]) == values
        event = json.loads(host.trace.read_text())
        assert event["selectionValues"] == values
        assert event["selectionGuardValues"] == guard
        assert host.dispatch_count == 1
    assert len(calls) == int(
        fault in {None, "guard", "readback-type", "readback-range"}
    )


def synthetic_evidence(target):
    records, trace = [], []
    for case in workloads.cases():
        arrays = workloads.operands(np, case)
        result = np.where(
            arrays[0], arrays[1].astype(case["dtype"]), arrays[2].astype(case["dtype"])
        )
        entries = workloads.expected_entries(np, case)
        records.append(
            {
                **case,
                "inputWords": [workloads.words(np, value) for value in arrays],
                "inputUnchanged": True,
                "resultWords": workloads.words(np, result),
                "resultShape": list(result.shape),
                "resultDtype": case["dtype"],
                "dispatchCount": len(entries),
            }
        )
        trace.extend({"entry": entry} for entry in entries)
        if entries:
            guard = (
                runtime.BOOLEAN_GUARD
                if case["dtype"] == "bool_"
                else runtime.COPY_GUARD
            )
            if case["dtype"] == "float32":
                guard = np.asarray(guard, dtype="uint32").view("float32").tolist()
            elif case["dtype"] == "bool_" and target != "metal":
                guard = [int(v) for v in guard]
            trace[-1].update(
                target=target,
                threads=result.size,
                workgroupCount=[result.size, 1, 1],
                workgroupSize=[1, 1, 1],
                dispatchVersion=runtime.DISPATCH_VERSION,
                selectionValues=[
                    runtime.wire_value(v) for v in result.reshape(-1).tolist()
                ],
                selectionGuardValues=guard,
            )
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "case",
        "input",
        "result",
        "shape",
        "dtype",
        "preserved",
        "count",
        "entry",
        "guard",
        "geometry",
        "native-value",
    ],
)
def test_selection_requires_exact_evidence(target, fault):
    records, trace = synthetic_evidence(target)
    if fault == "case":
        records.pop()
    elif fault == "input":
        records[1]["inputWords"][1][0] ^= 1
    elif fault == "result":
        records[1]["resultWords"][0] ^= 1
    elif fault in {"shape", "dtype"}:
        records[1]["resultShape" if fault == "shape" else "resultDtype"] = "wrong"
    elif fault == "preserved":
        records[1]["inputUnchanged"] = False
    elif fault == "count":
        records[1]["dispatchCount"] = 0
    elif fault == "entry":
        trace[0]["entry"] = "wrong"
    elif fault == "guard":
        trace[0]["selectionGuardValues"] = []
    elif fault == "geometry":
        trace[0]["workgroupSize"] = [2, 1, 1]
    elif fault == "native-value":
        trace[0]["selectionValues"][0] = 1234
    if fault:
        with pytest.raises(ValueError, match="Selection"):
            workloads.validate(records, trace, native=True)
    else:
        workloads.validate(records, trace, native=True)
        assert len(records) == 45
        for record in records:
            record["dispatchCount"] = 0
        workloads.validate(records, [], native=False)


def test_selection_float_contract_preserves_finite_bits_and_classifies_nan():
    assert workloads.equal_words([0x7FC00000], [0x7FC12345], "float32")
    assert not workloads.equal_words([0x7F800000], [0x7FC12345], "float32")
    assert not workloads.equal_words([0], [0x80000000], "float32")
    assert not workloads.equal_words([0], [1], "float32")


def absolute_evidence(target):
    records, trace = [], []
    for dtype, layout, value in absolute_workloads.cases(np):
        result = absolute_workloads.expected(np, value)
        entries = []
        if value.size:
            if layout == "reverse":
                entries.append(
                    packages.BOOLEAN_COPY_ENTRY
                    if dtype == "bool_"
                    else packages.COPY_ENTRY
                )
            entries.append(f"v_Abs{dtype}{dtype}")
        records.append(
            {
                "dtype": dtype,
                "layout": layout,
                "shape": list(result.shape),
                "resultDtype": result.dtype.name,
                "inputWords": workloads.words(np, value),
                "inputUnchanged": True,
                "resultWords": workloads.words(np, result),
                "dispatchCount": len(entries),
            }
        )
        trace.extend({"entry": entry} for entry in entries)
        if entries:
            physical = absolute_workloads.physical(np, value)
            values = absolute_workloads.expected(np, physical).tolist()
            guard = list(
                runtime.BOOLEAN_GUARD if dtype == "bool_" else runtime.COPY_GUARD
            )
            if dtype == "bool_" and target != "metal":
                values, guard = [int(v) for v in values], [int(v) for v in guard]
            trace[-1].update(
                target=target,
                threads=len(values),
                workgroupCount=[len(values), 1, 1],
                workgroupSize=[1, 1, 1],
                dispatchVersion=runtime.DISPATCH_VERSION,
                absoluteValues=values,
                unaryGuardValues=guard,
            )
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "case",
        "input",
        "result",
        "shape",
        "dtype",
        "preserved",
        "count",
        "entry",
        "guard",
        "geometry",
        "native-value",
        "abi",
        "extra-event",
    ],
)
def test_absolute_requires_exact_evidence(target, fault):
    records, trace = absolute_evidence(target)
    if fault == "case":
        records.pop()
    elif fault == "input":
        records[1]["inputWords"][0] ^= 1
    elif fault == "result":
        records[1]["resultWords"][0] ^= 1
    elif fault in {"shape", "dtype"}:
        records[1]["shape" if fault == "shape" else "resultDtype"] = "wrong"
    elif fault == "preserved":
        records[1]["inputUnchanged"] = False
    elif fault == "count":
        records[1]["dispatchCount"] = 0
    elif fault == "entry":
        trace[0]["entry"] = "wrong"
    elif fault == "guard":
        trace[0]["unaryGuardValues"] = []
    elif fault == "geometry":
        trace[0]["workgroupSize"] = [2, 1, 1]
    elif fault == "native-value":
        trace[0]["absoluteValues"][0] = 1234
    elif fault == "abi":
        trace[0]["dispatchVersion"] -= 1
    elif fault == "extra-event":
        trace.append(trace[0])
    if fault:
        with pytest.raises(ValueError, match="Absolute-value"):
            absolute_workloads.validate(records, trace, native=True)
    else:
        absolute_workloads.validate(records, trace, native=True)
        assert len(records) == 24
        for record in records:
            record["dispatchCount"] = 0
        absolute_workloads.validate(records, [], native=False)


def test_ci_requires_selection_and_absolute_execution_on_every_target():
    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {"metal": "macos-26", "opengl": "ubuntu-24.04", "directx": "windows-2025"}
    steps = {step.get("name"): step for step in job["steps"]}
    for name in (
        "Translate selection host packages",
        "Translate absolute-value host packages",
        "Execute selection host operations",
    ):
        step = steps[name]
        assert "if" not in step and "continue-on-error" not in step
        assert (
            "run_bounded_command.py" in step["run"]
            and "set -euo pipefail" in step["run"]
        )
    for family, name in (("selection", "selection"), ("absolute", "absolute-value")):
        assert f"--family {family}" in steps[f"Translate {name} host packages"]["run"]
        assert (
            f"--{family} .mlx-portable-small-rows/{family}-packages"
            in steps["Execute selection host operations"]["run"]
        )
    assert (
        "verify_selection --mlx-root mlx-upstream"
        in steps["Execute selection host operations"]["run"]
    )
    assert (
        "--reductions .mlx-portable-small-rows/concatenate-reductions"
        in steps["Execute selection host operations"]["run"]
    )
    retained = steps["Retain small-row execution evidence"]
    assert (
        retained["if"] == "always()"
        and retained["with"]["include-hidden-files"] is True
    )
    assert retained["with"]["if-no-files-found"] == "error"
    for event in ("push", "pull_request"):
        assert (
            "tests/test_mlx_portable_selection.py"
            in workflow.get("on", workflow.get(True))[event]["paths"]
        )
    assert any(
        "tests/test_mlx_portable_selection.py" in step.get("run", "")
        for step in workflow["jobs"]["portable-host"]["steps"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "worker",
        "upstream-skipped",
        "upstream-failed",
        "cpu-dispatch",
        "native-count",
        "selection-count",
        "missing-rejection",
        "negative-dispatch",
        "unknown-artifact",
        "sources-changed",
        "adaptation-changed",
    ],
)
def test_parent_requires_complete_worker_evidence(tmp_path, monkeypatch, fault):
    args = SimpleNamespace(mlx_root=tmp_path / "mlx", output_dir=tmp_path / "proof")
    for family, entries in (
        ("packages", packages.ENTRIES),
        ("selection", packages.SELECTION_ENTRIES),
        ("absolute", packages.ABSOLUTE_ENTRIES),
        ("reductions", ("w32/all_reduce_andbool_",)),
    ):
        directory = tmp_path / family
        directory.mkdir()
        setattr(args, family, directory)
        proof.write_json(
            directory / "index.json",
            {"target": "metal", "descriptors": {entry: {} for entry in entries}},
        )
    snapshots, sources = [], []

    def prepared(root):
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

    monkeypatch.setattr(proof, "verify_prepared", prepared)
    monkeypatch.setattr(proof, "upstream_test_sources", upstream_sources)
    monkeypatch.setattr(
        proof,
        "load_index",
        lambda directory, target: json.loads((directory / "index.json").read_text()),
    )
    verified_identity, verified_artifacts = [], []
    monkeypatch.setattr(
        proof,
        "verify_native_identity",
        lambda trace, target: verified_identity.extend(trace),
    )
    monkeypatch.setattr(
        proof,
        "verify_artifacts",
        lambda trace, *args, **kwargs: verified_artifacts.extend(trace),
    )
    commands = []

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
        results, selection_trace = synthetic_evidence("metal")
        absolute, absolute_trace = absolute_evidence("metal")
        trace = selection_trace + absolute_trace + [{"entry": "v_Selectint32"}]
        for event in trace:
            event.setdefault("workgroupSize", [1, 1, 1])
        if mode == "cpu":
            for record in results + absolute:
                record["dispatchCount"] = 0
        proof.write_json(directory / "results.json", results)
        proof.write_json(directory / "absolute.json", absolute)
        proof.write_json(
            directory / "upstream.json",
            {
                "test": proof.UPSTREAM_TEST,
                "testsRun": 1,
                "failures": int(fault == "upstream-failed"),
                "errors": 0,
                "skips": int(fault == "upstream-skipped"),
                "dispatchCount": (
                    int(fault == "cpu-dispatch")
                    if mode == "cpu"
                    else 2 if fault == "native-count" else 1
                ),
                "workloadDispatchCount": (
                    0 if mode == "cpu" else len(selection_trace) + len(absolute_trace)
                ),
                "selectionDispatchCount": (
                    0
                    if mode == "cpu"
                    else -1 if fault == "selection-count" else len(selection_trace)
                ),
            },
        )
        if mode == "native":
            if fault == "unknown-artifact":
                trace[-1]["entry"] = "unknown"
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
        evidence = proof.verify(args)
        assert evidence["casesPerPath"] == 45 and evidence["absoluteCasesPerPath"] == 24
        assert (
            evidence["fullTranslatedBackend"] is False
            and evidence["fullUpstreamSuite"] is False
        )
        assert (
            len(verified_identity)
            == len(verified_artifacts)
            == evidence["dispatchCount"]
        )
        assert set(evidence["negativeChecks"]) == set(proof.NEGATIVE_CHECKS)
        assert len(snapshots) == len(sources) == 2
        assert len(commands) == 2 + len(proof.NEGATIVE_CHECKS)
        for command in commands:
            seconds = command[command.index("--timeout-seconds") + 1]
            assert seconds == (
                "900" if command[command.index("--worker") + 1] == "native" else "60"
            )
