"""Typed MLX bitwise packages, callback contracts and numerical evidence."""

import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    binary_workloads,
)
from demos.integrations.mlx.portable_host import bitwise_workloads as workloads
from demos.integrations.mlx.portable_host import packages, prepare, runtime
from demos.integrations.mlx.portable_host import verify_bitwise as proof
from tests.ci_helpers import assert_paths_covered


@pytest.fixture(scope="module", params=["metal", "opengl", "directx"])
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp(f"bitwise-{request.param}")
    source = root / packages.BINARY_SOURCE
    source.parent.mkdir(parents=True)
    symbols = {
        "BitwiseAnd": "&",
        "BitwiseOr": "|",
        "BitwiseXor": "^",
        "LeftShift": "<<",
        "RightShift": ">>",
    }
    definitions = []
    for operation, symbol in symbols.items():
        definitions.append(f"""template<typename T> kernel void {operation}(
device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
device T* c [[buffer(2)]], constant uint& size [[buffer(3)]],
uint index [[thread_position_in_grid]]) {{ if (index < size) c[index] = a[index] {symbol} b[index]; }}
""")
    for entry, dtype in packages.BITWISE_ENTRIES.items():
        operation = entry[3 : -len(dtype)]
        metal_type = {"int32": "int", "uint32": "uint", "bool_": "bool"}[dtype]
        definitions.append(
            f'template [[host_name("{entry}")]] [[kernel]] decltype({operation}<{metal_type}>) {operation}<{metal_type}>;\n'
        )
    source.write_text("".join(definitions))
    unary = root / packages.UNARY_SOURCE
    unary.write_text(
        """template<typename T> kernel void invert(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) out[index] = ~in[index]; }
"""
        + "".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(invert<{"int" if dtype == "int32" else "uint"}>) invert<{"int" if dtype == "int32" else "uint"}>;\n'
            for entry, dtype in packages.BITWISE_INVERT_ENTRIES.items()
        )
    )
    original = source.read_bytes()
    original_unary = unary.read_bytes()
    output = root / "bitwise"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(root, output, request.param, family="bitwise")
    assert source.read_bytes() == original
    assert unary.read_bytes() == original_unary
    assert set(index["descriptors"]) == set(packages.BITWISE_PACKAGE_ENTRIES)
    assert index["family"] == "bitwise"
    assert len(packages.ENTRIES) == 93
    assert not set(packages.BITWISE_PACKAGE_ENTRIES).intersection(packages.ENTRIES)
    return output, index


@pytest.fixture
def host(translated, tmp_path):
    bitwise, index = translated
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
    return runtime.HostRuntime(base, tmp_path / "trace", bitwise=bitwise)


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
def test_optional_index_contract(host, translated, tmp_path, fault):
    _, original = translated
    index = copy.deepcopy(original)
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
        with pytest.raises(ValueError, match="Bitwise packages"):
            runtime.HostRuntime(host.directory, tmp_path / "trace2", bitwise=directory)
    else:
        loaded = runtime.HostRuntime(
            host.directory, tmp_path / "trace2", bitwise=directory
        )
        assert set(loaded.descriptors) == set(packages.ENTRIES) | set(
            packages.BITWISE_PACKAGE_ENTRIES
        )
    legacy = runtime.HostRuntime(host.directory, tmp_path / "legacy")
    assert legacy.bitwise_directory is None
    with pytest.raises(ValueError, match="No translated package"):
        legacy.dispatch("vv_BitwiseAndint32", None, 0, 1)


@pytest.mark.parametrize("entry", packages.BITWISE_ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "dtype",
        "direction",
        "count",
        "null",
        "size",
        "shift-negative",
        "shift-32",
        "guard",
        "readback-type",
        "readback-range",
    ],
)
def test_bitwise_callback_contract(host, monkeypatch, entry, fault):
    dtype = packages.BITWISE_ENTRIES[entry]
    shift = "Shift" in entry
    if fault and fault.startswith("shift") and not shift:
        return
    ctype = runtime.TYPES[dtype]
    left = (
        [True, False, True]
        if dtype == "bool_"
        else [0, 1, -1 if dtype == "int32" else 0xFFFFFFFF]
    )
    right = [False, True, True] if dtype == "bool_" else [0, 1, 2]
    memory = [
        (ctype * 3)(*left),
        (ctype * 3)(*right),
        (ctype * 3)(),
        ctypes.c_uint32(3),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                ("uint32" if name == "size" else dtype).encode(),
                ctypes.addressof(data),
                1 if name == "size" else 3,
                int(name == "c"),
            )
            for name, data in zip(("a", "b", "c", "size"), memory)
        ]
    )
    if fault == "dtype":
        buffers[0].dtype = b"float32"
    elif fault == "direction":
        buffers[1].output = 1
    elif fault == "count":
        buffers[0].count = 2
    elif fault == "null":
        buffers[0].data = 0
    elif fault == "size":
        memory[-1].value = 2
    elif fault == "shift-negative":
        memory[1][0] = -1
    elif fault == "shift-32":
        memory[1][0] = 32
    storage = runtime.physical_dtype(dtype, host.target)
    guard = (
        runtime.COPY_GUARD
        if dtype != "bool_"
        else (
            runtime.BOOLEAN_GUARD
            if storage == "bool"
            else [int(v) for v in runtime.BOOLEAN_GUARD]
        )
    )
    values = [True, False, True] if storage == "bool" else [1, 0, 1]
    output_values = values + guard
    if fault == "guard":
        output_values[-1] = (
            not bool(guard[-1]) if storage == "bool" else 0 if guard[-1] else 1
        )
    elif fault == "readback-type":
        output_values[0] = 1 if storage == "bool" else 1.0
    elif fault == "readback-range":
        output_values[0] = 2 if dtype == "bool_" else 2**32
    calls = []

    def execute(request):
        calls.append(request)
        assert request.artifact_path.is_relative_to(host.bitwise_directory)
        binding = next(
            item["name"]
            for item in host.descriptors[entry]["bindings"]
            if item["access"] == "read_write"
        )
        initial = next(item for item in request.fixture.inputs if item.name == binding)
        poison = (
            True
            if storage == "bool"
            else 1 if dtype == "bool_" else runtime.COPY_GUARD[0]
        )
        assert list(initial.values) == [poison] * 3 + guard
        return SimpleNamespace(
            status="ok",
            outputs={
                binding: {"dtype": storage, "shape": [35], "values": output_values}
            },
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, 4, 3)
        assert len(calls) == int(fault in {"guard", "readback-type", "readback-range"})
        assert not host.trace.exists()
        assert list(memory[2]) == [0, 0, 0]
    else:
        host.dispatch(entry, buffers, 4, 3)
        assert list(memory[2]) == values
        record = json.loads(host.trace.read_text())
        assert record["bitwiseValues"] == values
        assert record["binaryGuardValues"] == guard


@pytest.mark.parametrize("entry", packages.BITWISE_INVERT_ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "dtype",
        "direction",
        "count",
        "size",
        "guard",
        "readback-type",
        "readback-range",
    ],
)
@pytest.mark.parametrize("donated", [False, True])
def test_invert_callback_preserves_integer_storage(
    host, monkeypatch, entry, fault, donated
):
    dtype = packages.BITWISE_INVERT_ENTRIES[entry]
    ctype = runtime.TYPES[dtype]
    values = [-1, 0, -(2**31)] if dtype == "int32" else [0xFFFFFFFF, 0, 0x80000000]
    source = (ctype * 3)(*values)
    destination = source if donated else (ctype * 3)(7, 7, 7)
    before = list(destination)
    size = ctypes.c_uint32(3)
    buffers = (runtime.Buffer * 3)(
        runtime.Buffer(b"in", dtype.encode(), ctypes.addressof(source), 3, 0),
        runtime.Buffer(b"out", dtype.encode(), ctypes.addressof(destination), 3, 1),
        runtime.Buffer(b"size", b"uint32", ctypes.addressof(size), 1, 0),
    )
    if fault == "dtype":
        buffers[0].dtype = b"float32"
    elif fault == "direction":
        buffers[1].output = 0
    elif fault == "count":
        buffers[0].count = 2
    elif fault == "size":
        size.value = 2
    expected = [0, -1, 2**31 - 1] if dtype == "int32" else [0, 0xFFFFFFFF, 0x7FFFFFFF]
    output = expected + runtime.COPY_GUARD
    if fault == "guard":
        output[-1] = 0
    elif fault == "readback-type":
        output[0] = 0.0
    elif fault == "readback-range":
        output[0] = 2**32
    calls = []

    def execute(request):
        calls.append(request)
        assert request.artifact_path.is_relative_to(host.bitwise_directory)
        binding = next(
            item["name"]
            for item in host.descriptors[entry]["bindings"]
            if item["access"] == "read_write"
        )
        initial = next(item for item in request.fixture.inputs if item.name == binding)
        assert list(initial.values) == [runtime.COPY_GUARD[0]] * 35
        input_value = next(
            item
            for item in request.fixture.inputs
            if item.dtype == dtype and len(item.values) == 3
        )
        assert list(input_value.values) == values
        return SimpleNamespace(
            status="ok",
            outputs={binding: {"dtype": dtype, "shape": [35], "values": output}},
            details={},
        )

    monkeypatch.setattr(host.executor, "run", execute)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            host.dispatch(entry, buffers, 3, 3)
        assert list(destination) == before
        assert len(calls) == int(fault in {"guard", "readback-type", "readback-range"})
        assert not host.trace.exists()
    else:
        host.dispatch(entry, buffers, 3, 3)
        assert list(destination) == expected
        trace = json.loads(host.trace.read_text())
        assert trace["bitwiseValues"] == expected
        assert trace["unaryGuardValues"] == runtime.COPY_GUARD


@pytest.mark.parametrize(
    "layout,count", [("transpose", 15), ("broadcast", 5), ("reverse", 17)]
)
def test_invert_dispatch_preserves_stored_unary_layout(layout, count):
    case = {
        "entry": "v_BitwiseInvertint32int32",
        "dtype": "int32",
        "operation": "BitwiseInvert",
        "layout": layout,
    }
    calls = workloads.dispatches(np, case)
    assert calls[-1] == (case["entry"], count, [count, 1, 1])
    assert len(calls) == (2 if layout == "reverse" else 1)
    _, _, expected = workloads.reference(np, case)
    assert len(workloads.stored_values(case, expected)) == count


def expected_evidence(target="metal", native=True):
    records, trace = [], []
    for case in workloads.cases():
        a, b, expected = workloads.reference(np, case)
        calls = workloads.dispatches(np, case) if native else []
        records.append(
            {
                **case,
                "a": a.tolist(),
                "b": b.tolist() if b is not None else None,
                "actual": expected.tolist(),
                "expected": expected.tolist(),
                "resultShape": list(expected.shape),
                "resultDtype": "mlx.core." + case["dtype"].removesuffix("_"),
                "inputUnchanged": True,
                "dispatchCount": len(calls),
            }
        )
        for entry, count, grid in calls:
            guard = (
                runtime.COPY_GUARD
                if case["dtype"] != "bool_"
                else (
                    runtime.BOOLEAN_GUARD
                    if target == "metal"
                    else [int(v) for v in runtime.BOOLEAN_GUARD]
                )
            )
            event = {
                "entry": entry,
                "target": target,
                "threads": count,
                "workgroupCount": grid,
                "workgroupSize": [1, 1, 1],
                "dispatchVersion": runtime.DISPATCH_VERSION,
            }
            event[
                (
                    "binaryGuardValues"
                    if entry in packages.BITWISE_ENTRIES
                    else (
                        "unaryGuardValues"
                        if entry in packages.BITWISE_INVERT_ENTRIES
                        else "copyGuardWords"
                    )
                )
            ] = guard
            if entry in packages.BITWISE_PACKAGE_ENTRIES:
                event["bitwiseValues"] = (
                    [int(v) for v in workloads.stored_values(case, expected)]
                    if case["dtype"] == "bool_" and target != "metal"
                    else workloads.stored_values(case, expected)
                )
            trace.append(event)
    return records, trace


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "value",
        "dtype",
        "shape",
        "input",
        "dispatch-count",
        "trace-missing",
        "trace-extra",
        "trace-value",
        "guard",
        "geometry",
    ],
)
def test_bitwise_evidence_is_exact(target, fault):
    records, trace = expected_evidence(target)
    if fault == "missing":
        records.pop()
    elif fault == "value":
        records[1]["actual"] = 17
    elif fault == "dtype":
        records[1]["resultDtype"] = "mlx.core.uint32"
    elif fault == "shape":
        records[1]["resultShape"] = [1]
    elif fault == "input":
        records[1]["inputUnchanged"] = False
    elif fault == "dispatch-count":
        records[1]["dispatchCount"] = 0
    elif fault == "trace-missing":
        trace.pop()
    elif fault == "trace-extra":
        trace.append(trace[-1])
    elif fault == "trace-value":
        trace[0]["bitwiseValues"] = [999]
    elif fault == "guard":
        trace[0]["binaryGuardValues"] = []
    elif fault == "geometry":
        trace[0]["workgroupCount"] = [2, 1, 1]
    if fault:
        with pytest.raises(ValueError, match="Bitwise"):
            workloads.validate(records, trace, native=True)
    else:
        workloads.validate(records, trace, native=True)
        cpu, empty_trace = expected_evidence(native=False)
        workloads.validate(cpu, empty_trace, native=False)
        assert len(records) == 120
        assert {
            record["entry"]
            for record in trace
            if record["entry"] in packages.BITWISE_PACKAGE_ENTRIES
        } == set(packages.BITWISE_PACKAGE_ENTRIES)


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("fault", ["values", "guard", "layout", "geometry", "input"])
def test_invert_evidence_rejects_corruption(target, fault):
    records, trace = expected_evidence(target)
    entry = "v_BitwiseInvertint32int32"
    record = next(
        item
        for item in records
        if item["entry"] == entry and item["layout"] == "transpose"
    )
    event = next(
        item for item in trace if item["entry"] == entry and item["threads"] == 15
    )
    if fault == "values":
        event["bitwiseValues"][0] += 1
    elif fault == "guard":
        event["unaryGuardValues"] = []
    elif fault == "layout":
        event["bitwiseValues"] = np.asarray(record["actual"]).reshape(-1).tolist()
    elif fault == "geometry":
        event["workgroupCount"] = [5, 3, 1]
    elif fault == "input":
        record["inputUnchanged"] = False
    with pytest.raises(ValueError, match="Bitwise"):
        workloads.validate(records, trace, native=True)


def test_layout_operand_retains_reinterpreted_signed_type():
    value = (
        np.asarray([0xFFFFFFFF, 0x80000000, 2, 3, 4, 5], dtype="uint32")
        .view("int32")
        .reshape(2, 3)
        .T
    )
    observed = {}

    def strided(source, shape, strides, offset):
        observed["dtype"] = source.dtype
        return np.lib.stride_tricks.as_strided(
            source[offset:],
            shape=shape,
            strides=tuple(stride * source.itemsize for stride in strides),
        )

    got = binary_workloads.mlx_operand(
        SimpleNamespace(array=np.array, as_strided=strided), np, value
    )
    assert observed["dtype"] == value.dtype
    np.testing.assert_array_equal(got, value)


def test_failed_worker_keeps_command_and_no_success(tmp_path, monkeypatch):
    base, bitwise = tmp_path / "base", tmp_path / "bitwise"
    base.mkdir()
    bitwise.mkdir()
    (base / "index.json").write_text(json.dumps({"target": "metal"}))
    (bitwise / "index.json").write_text(
        json.dumps(
            {
                "target": "metal",
                "family": "bitwise",
                "descriptors": dict.fromkeys(packages.BITWISE_PACKAGE_ENTRIES, {}),
            }
        )
    )
    monkeypatch.setattr(
        proof, "verify_prepared", lambda root: {"commit": prepare.COMMIT}
    )

    def timeout(command, **kwargs):
        assert command[command.index("--timeout-seconds") + 1] == "60"
        return SimpleNamespace(returncode=124)

    monkeypatch.setattr(proof.subprocess, "run", timeout)
    output = tmp_path / "result"
    args = SimpleNamespace(
        mlx_root=tmp_path, packages=base, bitwise=bitwise, output_dir=output
    )
    with pytest.raises(RuntimeError, match="cpu worker failed"):
        proof.verify(args)
    assert json.loads((output / "cpu.command.json").read_text())["returncode"] == 124
    assert not (output / "evidence.json").exists()


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "unverified",
        "changed-hash",
        "package",
        "target",
        "geometry",
        "adapter",
        "compiler-missing",
        "compiler-failed",
    ],
)
def test_native_evidence_requires_compilation_and_dispatch(target, fault):
    identity = {"sizeBytes": 3, "hash": {"algorithm": "sha256", "value": "1" * 64}}
    actions = {
        "metal": ["compile-metal-for-native-runtime", "link-metal-for-native-runtime"],
        "opengl": ["validate-glsl-for-opengl-runtime"],
        "directx": ["compile-hlsl-for-directx-runtime"],
    }[target]
    record = {
        "artifact": {**identity, "packagePath": "artifact"},
        "workgroupCount": [3, 1, 1],
        "workgroupSize": [1, 1, 1],
    }
    details = {
        "artifactIdentityVerification": {
            "verificationStatus": "verified",
            "target": target,
            "expectedIdentity": identity,
            "observedIdentity": identity,
        },
        "nativeRuntimeDispatch": {
            "artifact": {"packagePath": "artifact", "target": target},
            "dispatch": {
                key: record[key] for key in ("workgroupCount", "workgroupSize")
            },
        },
        "runtimeParityAdapter": {
            "target": target,
            "runtimeAdapter": target + "-native-runtime",
        },
        "adapterSteps": [
            {"action": action, "status": "passed"}
            for action in [
                *actions,
                "dispatch-translated-artifact",
                "collect-runtime-outputs",
            ]
        ],
    }
    record["details"] = details
    if fault == "unverified":
        details["artifactIdentityVerification"]["verificationStatus"] = "skipped"
    elif fault == "changed-hash":
        details["artifactIdentityVerification"]["observedIdentity"] = {}
    elif fault == "package":
        details["nativeRuntimeDispatch"]["artifact"]["packagePath"] = "other"
    elif fault == "target":
        details["nativeRuntimeDispatch"]["artifact"]["target"] = "wrong"
    elif fault == "geometry":
        details["nativeRuntimeDispatch"]["dispatch"]["workgroupCount"] = [2, 1, 1]
    elif fault == "adapter":
        details["runtimeParityAdapter"]["runtimeAdapter"] = "cpu"
    elif fault == "compiler-missing":
        details["adapterSteps"].pop(0)
    elif fault == "compiler-failed":
        details["adapterSteps"][0]["status"] = "failed"
    if fault:
        with pytest.raises(ValueError, match="identity is incomplete"):
            proof.verify_native_identity([record], target)
    else:
        proof.verify_native_identity([record], target)


def test_ci_requires_bitwise_execution_on_every_target():
    import re

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {"metal": "macos-26", "opengl": "ubuntu-24.04", "directx": "windows-2025"}
    assert job["needs"] == "portable-host"
    steps = {step.get("name"): step for step in job["steps"]}
    build, execute = (
        steps["Translate bitwise host packages"],
        steps["Execute bitwise host operations"],
    )
    for step in (build, execute):
        assert "if" not in step and "continue-on-error" not in step
        assert "run_bounded_command.py" in step["run"]
        assert "set -euo pipefail" in step["run"]
    assert "--family bitwise" in build["run"]
    assert "verify_bitwise" in execute["run"]
    assert "--bitwise .mlx-portable-small-rows/bitwise-packages" in execute["run"]
    assert steps["Retain small-row execution evidence"]["if"] == "always()"
    deadlines = [
        int(value)
        for step in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
    for event in ("push", "pull_request"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[event]["paths"],
            "demos/integrations/mlx/tests/host/test_portable_bitwise.py",
        )
    assert any(
        "demos/integrations/mlx/tests/host/test_portable_bitwise.py" in step.get(
            "run", ""
        )
        for step in workflow["jobs"]["portable-host"]["steps"]
    )
