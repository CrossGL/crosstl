"""64-bit package and host callback contracts, independent of native execution."""

import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    integer64_workloads,
    packages,
    prepare,
    runtime,
    verify_integer64,
)


@pytest.fixture(scope="module", params=["metal", "opengl", "directx"])
def translated(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("integer64-" + request.param)
    types = {
        "float32": "float",
        "int32": "int",
        "uint32": "uint",
        "bool_": "bool",
        "int64": "int64_t",
        "uint64": "uint64_t",
    }
    source = root / packages.COPY_SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(
        """template<typename T> kernel void copy_signature(
device const T* src [[buffer(0)]], device T* dst [[buffer(1)]],
constant int* src_shape [[buffer(2)]], constant long* src_strides [[buffer(3)]],
constant long* dst_strides [[buffer(4)]], constant int& ndim [[buffer(5)]],
constant long& src_offset [[buffer(6)]], constant long& dst_offset [[buffer(7)]],
uint index [[thread_position_in_grid]]) { dst[index] = src[index]; }
template<typename T, typename U> kernel void cast_signature(
device const T* src [[buffer(0)]], device U* dst [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) dst[index] = U(src[index]); }
"""
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(copy_signature<{types[dtype]}>) copy_signature<{types[dtype]}>;'
            for entry, dtype in packages.INTEGER64_COPY_ENTRIES.items()
        )
        + "\n"
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(cast_signature<{types[src]}, {types[dst]}>) cast_signature<{types[src]}, {types[dst]}>;'
            for entry, (src, dst) in packages.INTEGER64_CAST_ENTRIES.items()
        )
    )
    (root / packages.UNARY_SOURCE).write_text(
        """template<typename T> kernel void absolute_signature(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant uint& size [[buffer(2)]], uint index [[thread_position_in_grid]]) {
if (index < size) out[index] = in[index]; }
"""
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(absolute_signature<{types[dtype]}>) absolute_signature<{types[dtype]}>;'
            for entry, dtype in packages.INTEGER64_ABSOLUTE_ENTRIES.items()
        )
    )
    (root / packages.BINARY_SOURCE).write_text(
        """template<typename T, typename U, int op> kernel void binary_signature(
device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
device U* c [[buffer(2)]], constant uint& size [[buffer(3)]],
uint index [[thread_position_in_grid]]) { if (index < size) c[index] = U(a[index]); }
"""
        + "\n".join(
            f'template [[host_name("{entry}")]] [[kernel]] decltype(binary_signature<{types[dtype]}, {types[output]}, {ordinal}>) binary_signature<{types[dtype]}, {types[output]}, {ordinal}>;'
            for ordinal, (entry, dtype, output) in enumerate(
                [
                    (entry, dtype, dtype)
                    for entry, dtype in packages.INTEGER64_BINARY_ENTRIES.items()
                ]
                + [
                    (entry, dtype, "bool_")
                    for entry, dtype in packages.INTEGER64_COMPARISON_ENTRIES.items()
                ]
            )
        )
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            packages.subprocess,
            "check_output",
            lambda command, **kwargs: prepare.COMMIT if "rev-parse" in command else "",
        )
        index = packages.build_packages(
            root, root / "integer64", request.param, family="integer64"
        )
    assert set(index["descriptors"]) == set(packages.INTEGER64_ENTRIES)
    assert len(index["descriptors"]) == 44 and len(packages.ENTRIES) == 93
    return root / "integer64", index


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
    return runtime.HostRuntime(base, tmp_path / "trace", integer64=directory)


def values(dtype):
    return {
        "int64": [-(2**63) + 1, 2**53 + 1, 2**63 - 1],
        "uint64": [2**32 + 1, 2**53 + 1, 2**64 - 1],
        "int32": [-(2**31), 0, 2**31 - 1],
        "uint32": [0, 2**31, 2**32 - 1],
        "float32": [-1.5, 0.0, 1.5],
        "bool_": [0, 1, 0],
    }[dtype]


def buffers_for(entry):
    if entry in packages.INTEGER64_COPY_ENTRIES:
        dtype = packages.INTEGER64_COPY_ENTRIES[entry]
        spec = {
            "src": (dtype, values(dtype)),
            "dst": (dtype, [0] * 3),
            "src_shape": ("int32", [1, 3]),
            "src_strides": ("int64", [3, 1]),
            "dst_strides": ("int64", [3, 1]),
            "ndim": ("int32", [2]),
            "src_offset": ("int64", [0]),
            "dst_offset": ("int64", [0]),
        }
        destination, groups = "dst", [2, 1, 1]
    elif entry in packages.INTEGER64_CAST_ENTRIES:
        src, dtype = packages.INTEGER64_CAST_ENTRIES[entry]
        spec = {
            "src": (src, values(src)),
            "dst": (dtype, [0] * 3),
            "size": ("uint32", [3]),
        }
        destination, groups = "dst", [3, 1, 1]
    elif entry in packages.INTEGER64_ABSOLUTE_ENTRIES:
        dtype = packages.INTEGER64_ABSOLUTE_ENTRIES[entry]
        spec = {
            "in": (dtype, values(dtype)),
            "out": (dtype, [0] * 3),
            "size": ("uint32", [3]),
        }
        destination, groups = "out", [3, 1, 1]
    else:
        src = {
            **packages.INTEGER64_BINARY_ENTRIES,
            **packages.INTEGER64_COMPARISON_ENTRIES,
        }[entry]
        dtype = "bool_" if entry in packages.INTEGER64_COMPARISON_ENTRIES else src
        spec = {
            "a": (src, values(src)),
            "b": (src, values(src)),
            "c": (dtype, [0] * 3),
            "size": ("uint32", [3]),
        }
        destination, groups = "c", [3, 1, 1]
    memory = {
        name: (runtime.TYPES[kind] * len(data))(*data)
        for name, (kind, data) in spec.items()
    }
    buffers = (runtime.Buffer * len(spec))(
        *(
            runtime.Buffer(
                name.encode(),
                kind.encode(),
                ctypes.addressof(memory[name]),
                len(data),
                int(name == destination),
            )
            for name, (kind, data) in spec.items()
        )
    )
    return (
        buffers,
        memory,
        destination,
        dtype,
        runtime.Launch((ctypes.c_uint32 * 3)(*groups), (ctypes.c_uint32 * 3)(1, 1, 1)),
    )


@pytest.mark.parametrize("entry", packages.INTEGER64_ENTRIES)
@pytest.mark.parametrize(
    "fault", [None, "dtype", "count", "null", "guard", "readback-range"]
)
def test_integer64_callback_preserves_transport_and_guards(
    host, monkeypatch, entry, fault
):
    buffers, memory, destination, dtype, launch = buffers_for(entry)
    expected = values(dtype)
    storage = runtime.physical_dtype(dtype, host.target)
    expected = [bool(value) for value in expected] if storage == "bool" else expected
    guard = (
        [
            bool(value) if storage == "bool" else int(value)
            for value in runtime.BOOLEAN_GUARD
        ]
        if dtype == "bool_"
        else (
            [
                ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
                for word in runtime.COPY_GUARD
            ]
            if dtype == "float32"
            else list(runtime.COPY_GUARD)
        )
    )
    output = expected + guard
    if fault == "dtype":
        buffers[0].dtype = b"float32" if buffers[0].dtype != b"float32" else b"int32"
    elif fault == "count":
        buffers[0].count = 0
    elif fault == "null":
        buffers[0].data = None
    elif fault == "guard":
        output[-1] = not guard[-1] if storage == "bool" else -1
    elif fault == "readback-range":
        if dtype in {"int64", "uint64"}:
            output[0] = 2**64
        elif dtype == "bool_":
            output[0] = 2
        else:
            output = output[:-1]
    calls = []

    def execute(request):
        calls.append(request)
        binding = next(
            item["name"]
            for item in host.descriptors[entry]["bindings"]
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
            host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory[destination]) == [0] * 3 and not host.trace.exists()
    else:
        host.dispatch(entry, buffers, len(buffers), 3, launch=launch)
        assert list(memory[destination]) == expected
        assert json.loads(host.trace.read_text())["entry"] == entry
    assert len(calls) == int(fault in {None, "guard", "readback-range"})


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "target", "family", "descriptor"]
)
def test_integer64_optional_index_rejects_invalid_contracts(host, tmp_path, fault):
    data = json.loads((host.integer64_directory / "index.json").read_text())
    if fault == "missing":
        data["descriptors"].pop(next(iter(data["descriptors"])))
    elif fault == "extra":
        data["descriptors"]["other"] = {"target": host.target}
    elif fault == "target":
        data["target"] = "wrong"
    elif fault == "family":
        data["family"] = "base"
    else:
        next(iter(data["descriptors"].values()))["target"] = "wrong"
    directory = tmp_path / "changed"
    directory.mkdir()
    (directory / "index.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exact entry set"):
        runtime.HostRuntime(host.directory, tmp_path / "unused", integer64=directory)


def test_missing_integer64_package_is_rejected_before_dispatch(host):
    plain = runtime.HostRuntime(host.directory, host.trace)
    entry = next(iter(packages.INTEGER64_ENTRIES))
    buffers, _, _, _, launch = buffers_for(entry)
    with pytest.raises(ValueError, match="No translated package"):
        plain.dispatch(entry, buffers, len(buffers), 3, launch=launch)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_copy_overlap_checks_use_all_eight_bytes(dtype):
    entry = next(
        entry
        for entry, value in packages.INTEGER64_COPY_ENTRIES.items()
        if value == dtype
    )
    buffers, memory, _, _, _ = buffers_for(entry)
    buffers[1].data = ctypes.addressof(memory["src"]) + 16
    with pytest.raises(ValueError, match="overlap"):
        runtime.copy_layout.validate(
            {buffer.name.decode(): buffer for buffer in buffers}, 3, dtype=dtype
        )


@pytest.fixture(scope="module")
def evidence():
    import numpy as np

    records, events = [], []
    for case in integer64_workloads.cases():
        arrays = integer64_workloads.inputs(np, case)
        expected = integer64_workloads.reference(np, case, arrays)
        entries = integer64_workloads.trace_entries(case, arrays)
        records.append(
            {
                **case,
                "inputPayloads": [
                    integer64_workloads.payload(np, value) for value in arrays
                ],
                "inputUnchanged": True,
                "resultPayload": integer64_workloads.payload(np, expected),
                "resultShape": list(expected.shape),
                "resultDtype": expected.dtype.name,
                "dispatchCount": len(entries),
            }
        )
        for entry in entries:
            events.append(
                {
                    "entry": entry,
                    "target": "metal",
                    "dispatchVersion": runtime.DISPATCH_VERSION,
                    "workgroupSize": [1, 1, 1],
                }
            )
        if entries:
            values = expected
            if case["entry"] in packages.INTEGER64_ABSOLUTE_ENTRIES:
                values = np.abs(integer64_workloads.physical(np, arrays[0]))
            guard = (
                runtime.BOOLEAN_GUARD
                if expected.dtype == np.bool_
                else (
                    np.asarray(runtime.COPY_GUARD, dtype="uint32")
                    .view("float32")
                    .tolist()
                    if expected.dtype == np.float32
                    else runtime.COPY_GUARD
                )
            )
            events[-1].update(
                {
                    "integer64Values": values.reshape(-1).tolist(),
                    "integer64GuardValues": guard,
                    "threads": values.size,
                    "workgroupCount": [values.size, 1, 1],
                }
            )
    return records, events


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_integer64_workload_evidence_requires_exact_high_bits(evidence, target):
    records, trace = copy.deepcopy(evidence)
    for event in trace:
        event["target"] = target
        if (
            target != "metal"
            and event.get("integer64GuardValues") == runtime.BOOLEAN_GUARD
        ):
            event["integer64GuardValues"] = [
                int(value) for value in runtime.BOOLEAN_GUARD
            ]
            event["integer64Values"] = [
                int(value) for value in event["integer64Values"]
            ]
    integer64_workloads.validate(records, trace, native=True)
    assert len(records) == 44 * 8 + 8
    assert set(packages.INTEGER64_ENTRIES) <= {item["entry"] for item in trace}
    for record in records:
        record["dispatchCount"] = 0
    integer64_workloads.validate(records, [], native=False)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "identity",
        "input",
        "preservation",
        "result",
        "shape",
        "dtype",
        "count",
        "trace",
        "abi",
        "workgroup",
        "guard",
        "high-bit",
        "fractional-integer",
        "thread-count",
        "launch",
    ],
)
def test_integer64_workload_evidence_rejects_corruption(evidence, fault):
    records, trace = copy.deepcopy(evidence)
    record = next(
        record
        for record in records
        if record["entry"] == "v_Absint64int64" and record["layout"] == "vector"
    )
    event = next(
        event
        for event in trace
        if event.get("entry") == "v_Absint64int64" and event.get("threads") == 9
    )
    if fault == "missing":
        records.pop()
    elif fault == "extra":
        records.append(record)
    elif fault == "identity":
        record["entry"] = "other"
    elif fault == "input":
        record["inputPayloads"] = []
    elif fault == "preservation":
        record["inputUnchanged"] = False
    elif fault == "result":
        record["resultPayload"] = "00"
    elif fault == "shape":
        record["resultShape"] = [1]
    elif fault == "dtype":
        record["resultDtype"] = "int32"
    elif fault == "count":
        record["dispatchCount"] += 1
    elif fault == "trace":
        trace.append(event)
    elif fault == "abi":
        event["dispatchVersion"] = 0
    elif fault == "workgroup":
        event["workgroupSize"] = [32, 1, 1]
    elif fault == "guard":
        event["integer64GuardValues"] = [0] * 32
    elif fault == "high-bit":
        event["integer64Values"][2] -= 2**32
    elif fault == "fractional-integer":
        event["integer64Values"][0] = float(event["integer64Values"][0])
    elif fault == "thread-count":
        event["threads"] -= 1
    else:
        event["workgroupCount"] = [1, 1, 1]
    with pytest.raises(ValueError, match="Integer64"):
        integer64_workloads.validate(records, trace, native=True)


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
def test_integer64_upstream_proof_requires_both_unchanged_tests(native, fault):
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
            for name in verify_integer64.UPSTREAM_TESTS
        ],
        "workloadDispatchCount": 400 if native else 0,
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
        with pytest.raises(ValueError, match="Integer64"):
            verify_integer64.validate_upstream(data, native=native)
    else:
        verify_integer64.validate_upstream(data, native=native)


def test_integer64_ci_requires_all_three_native_targets():
    import re

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["integer64-host"]
    assert job["needs"] == "portable-host"
    assert "if" not in job and "continue-on-error" not in job
    assert {
        item["target"]: item["os"] for item in job["strategy"]["matrix"]["include"]
    } == {
        "directx": "windows-2025",
        "opengl": "ubuntu-24.04",
        "metal": "macos-26",
    }
    steps = {step.get("name"): step for step in job["steps"]}
    prepare_step = steps["Prepare pinned MLX"]["run"]
    assert prepare_step.index("config core.autocrlf false") < prepare_step.index(
        "checkout --detach"
    )
    build = steps["Translate integer64 host packages"]["run"]
    assert "--family integer64" in build and "--target" in build
    execute = steps["Execute integer64 host operations"]
    assert "if" not in execute and "continue-on-error" not in execute
    assert "portable_host.verify_integer64" in execute["run"]
    assert "--absolute" in execute["run"] and execute["run"].count("--reductions") == 2
    deadlines = [
        int(value)
        for step in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert deadlines == [1800, 7200, 600, 900, 900, 3000]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60 <= 360 * 60
    retained = steps["Retain integer64 execution evidence"]
    assert retained["if"] == "always()"
    assert retained["with"]["include-hidden-files"] is True
    assert retained["with"]["if-no-files-found"] == "error"
    triggers = workflow.get("on", workflow.get(True))
    for event in ("pull_request", "push"):
        assert "tests/test_mlx_portable_integer64.py" in triggers[event]["paths"]


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
def test_integer64_parent_checks_workers_sources_and_artifacts(
    tmp_path, monkeypatch, evidence, fault
):
    proof = verify_integer64
    args = SimpleNamespace(mlx_root=tmp_path / "mlx", output_dir=tmp_path / "proof")
    for family, entries in (
        ("packages", packages.ENTRIES),
        ("integer64", packages.INTEGER64_ENTRIES),
        ("absolute", packages.ABSOLUTE_ENTRIES),
        ("reductions", ("w32/all_reduce_andbool_", "w1/init_reduce_andbool_")),
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
        records, trace = copy.deepcopy(evidence)
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
            trace.extend([{"entry": "v_Absint64int64", "workgroupSize": [1, 1, 1]}] * 2)
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
        assert result["casesPerPath"] == 360
        assert (
            result["fullUpstreamSuite"] is False
            and result["fullTranslatedBackend"] is False
        )
        assert len(identity) == len(artifacts) == result["dispatchCount"]
        assert len(snapshots) == len(sources) == 2
        assert len(commands) == 2 + len(proof.NEGATIVE_CHECKS)
        assert all(command.count("--reductions") == 1 for command in commands)
