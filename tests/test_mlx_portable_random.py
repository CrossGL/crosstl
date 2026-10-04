import ctypes
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx import random_audit
from demos.integrations.mlx.portable_host import (
    random_dispatch,
    random_evidence,
    random_packages,
    random_workloads,
    runtime,
    verify_random,
)


def test_random_audit_preserves_public_source_identity():
    assert random_audit.SOURCE == random_packages.SOURCE
    assert random_audit.SOURCE_SHA256 == random_packages.SOURCE_SHA256


def buffers(per_key=3, shape=(3, 2), strides=(2, 1)):
    count = math.prod(shape) // 2
    span = 1 + sum((size - 1) * step for size, step in zip(shape, strides))
    items = {
        "keys": list(range(span)),
        "out": [42] * (count * per_key),
        "bytes_per_key": [per_key],
        "ndim": [len(shape)],
        "key_shape": shape,
        "key_strides": strides,
    }
    memory = {
        name: (random_dispatch.TYPES[name][1] * len(data))(*data)
        for name, data in items.items()
    }
    supplied = {
        name: runtime.Buffer(
            name.encode(),
            random_dispatch.TYPES[name][0].encode(),
            ctypes.addressof(data),
            len(data),
            int(name == "out"),
        )
        for name, data in memory.items()
    }
    launch = runtime.Launch((count, ((per_key + 3) // 4 + 1) // 2, 1), (1, 1, 1))
    return supplied, memory, launch


@pytest.mark.parametrize("per_key", [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 15, 17, 33])
@pytest.mark.parametrize(
    "shape,strides,entry",
    [
        ((3, 2), (2, 1), "rbitsc"),
        ((3, 2), (1, 3), "rbits"),
        ((3, 2), (0, 1), "rbits"),
        ((2, 3, 2), (0, 2, 1), "rbits"),
    ],
)
def test_random_validates_contiguous_strided_and_broadcast_keys(
    per_key, shape, strides, entry
):
    supplied, memory, launch = buffers(per_key, shape, strides)
    layout, metadata = random_dispatch.validate(
        entry, supplied, len(memory["out"]), launch.execution()
    )
    assert layout.logical_byte_count == len(memory["out"])
    assert layout.native_bytes_per_key == max(4, per_key)
    assert metadata["keyStrides"] == list(strides)
    assert metadata["keyCount"] == math.prod(shape) // 2


@pytest.mark.parametrize(
    "fault",
    [
        "entry",
        "missing",
        "null",
        "dtype",
        "direction",
        "count",
        "rank",
        "rank-length",
        "strides-length",
        "shape",
        "stride",
        "span",
        "byte-count",
        "per-key",
        "overlap",
        "groups",
        "width",
        "contiguous",
    ],
)
def test_random_rejects_invalid_metadata(fault):
    supplied, memory, launch = buffers()
    entry, count = "rbitsc", len(memory["out"])
    execution = launch.execution()
    if fault == "entry":
        entry = "unknown"
    elif fault == "missing":
        supplied.pop("ndim")
    elif fault == "null":
        supplied["keys"].data = None
    elif fault == "dtype":
        supplied["keys"].dtype = b"int32"
    elif fault == "direction":
        supplied["keys"].output = 1
    elif fault == "count":
        supplied["keys"].count = 65536
    elif fault == "rank":
        memory["ndim"][0] = 3
    elif fault == "rank-length":
        supplied["ndim"].count = 2
    elif fault == "strides-length":
        supplied["key_strides"].count = 1
    elif fault == "shape":
        memory["key_shape"][-1] = 3
    elif fault == "stride":
        memory["key_strides"][0] = -1
    elif fault == "span":
        supplied["keys"].count -= 1
    elif fault == "byte-count":
        count += 1
    elif fault == "per-key":
        memory["bytes_per_key"][0] = 4
    elif fault == "overlap":
        supplied["out"].data = supplied["keys"].data
    elif fault == "groups":
        execution["workgroupCount"][1] = 2
    elif fault == "width":
        execution["workgroupSize"][0] = 32
    elif fault == "contiguous":
        supplied, memory, launch = buffers(strides=(1, 3))
    with pytest.raises(ValueError):
        random_dispatch.validate(entry, supplied, count, execution)


def descriptor(target, entry):
    names = ["keys", "out", "odd", "bytes_per_key"]
    if entry == "rbits":
        names.extend(("ndim", "key_shape", "key_strides"))
    bindings = []
    for name in names:
        if name == "out":
            dtype, size = ("int8", 1) if target == "metal" else ("int32", 4)
        elif name == "odd":
            dtype, size = ("bool", 1) if target == "metal" else ("uint32", 4)
        else:
            dtype, ctype = random_dispatch.TYPES[name]
            size = ctypes.sizeof(ctype)
        member = "out_" if name == "out" else name
        bindings.append(
            {
                "name": name,
                "scalarLayout": {
                    "memberName": (
                        entry + "_" + member if target == "directx" else member
                    ),
                    "elementType": dtype,
                    "elementStrideBytes": size,
                },
            }
        )
    return {
        "artifact": {"hash": {"algorithm": "sha256", "value": "a" * 64}},
        "bindings": bindings,
    }


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("entry", random_packages.ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "guard",
        "dtype",
        "shape",
        "encoding",
        "missing",
        "extra",
        "status",
        "short",
        "byte",
        "bool",
        "binding",
        "stride",
        "duplicate",
        "metadata",
        "package",
    ],
)
def test_random_writeback_requires_exact_native_storage(
    tmp_path, monkeypatch, target, entry, fault
):
    supplied, memory, launch = buffers()
    original = bytes(memory["out"])
    desc = descriptor(target, entry)
    if fault == "binding":
        desc["bindings"][0]["scalarLayout"]["memberName"] = "unknown"
    elif fault == "stride":
        desc["bindings"][0]["scalarLayout"]["elementStrideBytes"] = 8
    elif fault == "duplicate":
        desc["bindings"].append(desc["bindings"][0])
    elif fault == "metadata":
        memory["ndim"][0] = 9
    calls = []
    native = [-128, -1, 0, 127, 3, 4, 5, 6, 7, 8, 9, 10]

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        assert kwargs == {"expected_target": target}
        assert inputs["bytes_per_key"]["values"] == [4]
        assert inputs["odd"]["values"] == [1]
        assert outputs["out"]["values"] == [91] * 29
        assert execution == launch.execution()
        calls.append(inputs)
        return SimpleNamespace(inputs=inputs, outputs=outputs)

    def execute(host, request):
        output = dict(request.outputs["out"], values=native + random_dispatch.GUARD)
        if fault == "guard":
            output["values"][-1] = 0
        elif fault == "dtype":
            output["dtype"] = "uint32"
        elif fault == "shape":
            output["shape"] = [3]
        elif fault == "encoding":
            output["encoding"] = "ieee754-binary32"
        elif fault == "short":
            output["values"].pop()
        elif fault == "byte":
            output["values"][0] = 128
        elif fault == "bool":
            output["values"][0] = True
        outputs = {"out": output}
        if fault == "missing":
            outputs = {}
        elif fault == "extra":
            outputs["extra"] = output
        return SimpleNamespace(
            status="error" if fault == "status" else "ok", outputs=outputs, details={}
        )

    monkeypatch.setattr(
        random_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(random_dispatch, "execute", execute)
    host = SimpleNamespace(
        target=target,
        random_directory=None if fault == "package" else tmp_path,
        descriptors={entry: desc},
        dispatch_count=0,
        trace=tmp_path / "trace.jsonl",
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            random_dispatch.dispatch(
                host, entry, table, len(table), len(memory["out"]), launch
            )
        assert bytes(memory["out"]) == original and host.dispatch_count == 0
        assert not host.trace.exists()
        if fault in {"binding", "stride", "duplicate", "metadata", "package"}:
            assert not calls
    else:
        random_dispatch.dispatch(
            host, entry, table, len(table), len(memory["out"]), launch
        )
        expected = bytes([128, 255, 0, 3, 4, 5, 7, 8, 9])
        assert bytes(memory["out"]) == expected and host.dispatch_count == 1
        event = json.loads(host.trace.read_text())
        assert event["randomValues"] == native
        assert event["randomGuardValues"] == random_dispatch.GUARD
        assert event["outputHash"] == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "pin",
        "target",
        "entry",
        "source",
        "backend",
        "hash",
        "swap",
        "descriptor",
        "duplicate",
        "incomplete",
        "invalid",
        "list",
        "null",
    ],
)
def test_random_index_verifies_source_and_package_identity(
    tmp_path, monkeypatch, fault
):
    descriptors = {
        entry: dict(
            descriptor("opengl", entry),
            target="opengl",
            stage="compute",
            entryPoint={"name": "main"},
            source={
                "path": random_packages.SOURCE,
                "backend": "metal",
                "hash": {"algorithm": "sha256", "value": random_packages.SOURCE_SHA256},
            },
        )
        for entry in random_packages.ENTRIES
    }
    units = [
        {"id": entry, "entryPoint": {"source": entry}}
        for entry in random_packages.ENTRIES
    ]
    index = {
        "commit": random_packages.COMMIT,
        "target": "opengl",
        "descriptors": json.loads(json.dumps(descriptors)),
    }
    if fault == "pin":
        index["commit"] = "wrong"
    elif fault == "target":
        index["target"] = "metal"
    elif fault == "entry":
        units[0]["entryPoint"]["source"] = "unknown"
    elif fault in {"source", "backend", "hash"}:
        descriptors["rbitsc"]["source"][
            {"source": "path", "backend": "backend", "hash": "hash"}[fault]
        ] = "wrong"
        index["descriptors"] = descriptors
    elif fault == "swap":
        index["descriptors"] = {
            "rbits": descriptors["rbitsc"],
            "rbitsc": descriptors["rbits"],
        }
    elif fault == "descriptor":
        index["descriptors"]["rbitsc"]["stage"] = "fragment"
    elif fault == "duplicate":
        units = [units[0], units[0]]
    elif fault == "incomplete":
        units.pop()
    elif fault == "list":
        index = []
    elif fault == "null":
        index["descriptors"] = None
    (tmp_path / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(
        random_packages,
        "build_runtime_loader_manifest",
        lambda path: {"success": fault != "invalid", "loadUnits": units},
    )
    monkeypatch.setattr(
        random_packages,
        "build_native_loader_abi_descriptor",
        lambda loader, load_unit_id: descriptors[load_unit_id],
    )
    if fault:
        with pytest.raises(ValueError):
            random_packages.load_index(tmp_path, "opengl")
    else:
        assert random_packages.load_index(tmp_path, "opengl") == descriptors


def workload_records(native):
    records, trace = [], []
    for case in random_workloads.cases():
        reference = random_workloads.expected(np, case)
        count = int(native and reference.size > 0)
        records.append(
            {
                **case,
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
                "words": reference.view(np.uint32).reshape(-1).tolist(),
                "dispatchStart": len(trace),
                "dispatchCount": count,
            }
        )
        if count:
            keys = 3 if case.get("layout") not in (None, "single") else 1
            trace.append(
                {
                    "entry": "rbitsc",
                    "threads": reference.nbytes,
                    "randomMetadata": {
                        "keyCount": keys,
                        "wordCount": reference.size // keys,
                    },
                    "randomValues": reference.view(np.int8).reshape(-1).tolist(),
                }
            )
    return records, trace


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "duplicate",
        "values",
        "dtype",
        "shape",
        "count",
        "boundary",
        "id",
    ],
)
def test_random_workload_requires_exact_inventory_and_results(native, fault):
    records, trace = workload_records(native)
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records[1] = records[0]
    elif fault == "values":
        records[1]["words"][0] ^= 1
    elif fault == "dtype":
        records[1]["dtype"] = "int32"
    elif fault == "shape":
        records[1]["shape"] = [2]
    elif fault == "count":
        records[1]["dispatchCount"] = -1
    elif fault == "boundary":
        records[1]["dispatchStart"] += 1
    elif fault == "id":
        records[1]["id"] = "unknown"
    if fault:
        with pytest.raises(ValueError):
            random_workloads.validate(np, records, trace, native=native)
    else:
        random_workloads.validate(np, records, trace, native=native)
        assert len(records) == 42


def event(target, entry):
    case = random_audit.byte_workload(entry, 3, 3)
    desc = descriptor(target, entry)
    desc["target"] = target
    inputs, outputs = random_audit.dispatch_values(desc, case)
    request = {
        "target": target,
        "entryPoint": {"opengl": "main", "directx": "CSMain"}.get(target, entry),
        "dispatch": case["execution"],
        "buffers": {},
    }
    for binding in desc["bindings"]:
        value = inputs[binding["name"]]
        request["buffers"][binding["name"]] = {
            **value,
            "binding": {"metadata": {"scalarLayout": binding["scalarLayout"]}},
        }
    return {
        "target": target,
        "entry": entry,
        "details": {"request": request},
        "inputs": inputs,
        "randomMetadata": {
            "keyCount": 3,
            "keyShape": [3, 2],
            "keyStrides": [2, 1] if entry == "rbitsc" else [1, 3],
            "logicalBytesPerKey": 3,
            "nativeBytesPerKey": 4,
            "wordCount": 1,
        },
        "randomValues": case["expected"],
        "randomGuardValues": [91] * 17,
        "threads": 9,
        "outputHash": hashlib.sha256(bytes(case["logicalExpected"])).hexdigest(),
        "dispatchVersion": 3,
        **case["execution"],
    }


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("entry", random_packages.ENTRIES)
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "value",
        "padding",
        "guard",
        "poison",
        "key",
        "rank",
        "count",
        "counter",
        "width",
        "entry",
        "hash",
        "dtype",
        "stride",
        "version",
    ],
)
def test_random_audit_checks_padding_metadata_and_native_identity(
    monkeypatch, target, entry, fault
):
    record = event(target, entry)
    calls = []
    monkeypatch.setattr(
        random_evidence, "audit_input_bindings", lambda event: calls.append("bindings")
    )
    monkeypatch.setattr(
        random_evidence, "audit_native_execution", lambda event: calls.append("native")
    )
    if fault == "value":
        record["randomValues"][0] ^= 1
    elif fault == "padding":
        record["randomValues"][3] ^= 1
    elif fault == "guard":
        record["randomGuardValues"][-1] = 0
    elif fault == "poison":
        record["inputs"]["out"]["values"][0] = 0
    elif fault == "key":
        record["inputs"]["keys"]["values"][0] ^= 1
    elif fault == "rank":
        record["randomMetadata"]["keyShape"] = [6]
    elif fault == "count":
        record["randomMetadata"]["keyCount"] = 1
    elif fault == "counter":
        record["inputs"]["odd"]["values"][0] = 0
    elif fault == "width":
        record["workgroupSize"] = [32, 1, 1]
    elif fault == "entry":
        record["details"]["request"]["entryPoint"] = "unknown"
    elif fault == "hash":
        record["outputHash"] = "wrong"
    elif fault == "dtype":
        record["inputs"]["bytes_per_key"]["dtype"] = "uint32"
    elif fault == "stride":
        record["details"]["request"]["buffers"]["keys"]["binding"]["metadata"][
            "scalarLayout"
        ]["elementStrideBytes"] = 8
    elif fault == "version":
        record["dispatchVersion"] = 2
    if fault:
        with pytest.raises(ValueError):
            random_evidence.audit_event(np, record)
        assert not calls
    else:
        logical = random_evidence.audit_event(np, record)
        assert len(logical) == 9
        assert calls == ["bindings", "native"]


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "duplicate",
        "skips",
        "failures",
        "errors",
        "testsRun",
        "dispatchCount",
        "workloads",
    ],
)
def test_random_upstream_proof_cannot_skip_or_replace_tests(native, fault):
    record = {
        "tests": [
            {
                "test": name,
                "testsRun": 1,
                "skips": 0,
                "failures": 0,
                "errors": 0,
                "dispatchCount": int(native),
            }
            for name in verify_random.UPSTREAM_TESTS
        ],
        "workloadDispatchCount": int(native),
    }
    if fault == "missing":
        record["tests"].pop()
    elif fault == "duplicate":
        record["tests"][1] = record["tests"][0]
    elif fault == "workloads":
        record["workloadDispatchCount"] = int(not native)
    elif fault:
        record["tests"][0][fault] = (
            0
            if fault == "testsRun"
            else (int(not native) if fault == "dispatchCount" else 1)
        )
    if fault:
        with pytest.raises(ValueError):
            verify_random.validate_upstream(record, native=native)
    else:
        verify_random.validate_upstream(record, native=native)


def test_random_host_ci_requires_all_three_platforms():
    workflow = yaml.safe_load(
        Path(".github/workflows/mlx-portable-host.yml").read_text()
    )
    job = workflow["jobs"]["small-row-reductions"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    steps = {step.get("name"): step for step in job["steps"]}
    build = steps["Translate random kernels"]
    execute = steps["Execute random workloads and unchanged upstream tests"]
    for step in (build, execute):
        assert not step.get("continue-on-error") and not step.get("if")
        assert "set -euo pipefail" in step["run"]
    assert (
        "--entry all_reduce_andbool_ --width 32"
        in steps["Translate concatenation test reduction"]["run"]
    )
    assert "portable_host.random_packages" in build["run"]
    assert "--target ${{ matrix.target }}" in build["run"]
    assert "portable_host.verify_random" in execute["run"]
    assert "--timeout-seconds 2200" in execute["run"]
    assert "--random .mlx-portable-small-rows/random-packages" in execute["run"]
    assert (
        "--reductions .mlx-portable-small-rows/concatenate-reductions" in execute["run"]
    )
    assert steps["Retain small-row execution evidence"]["if"] == "always()"


def test_random_verifier_preserves_all_failed_worker_attempts(tmp_path, monkeypatch):
    (tmp_path / "index.json").write_text(json.dumps({"target": "metal"}))
    args = SimpleNamespace(
        mlx_root=tmp_path,
        packages=tmp_path,
        random=tmp_path,
        reductions=tmp_path,
        output_dir=tmp_path / "evidence",
    )
    monkeypatch.setattr(verify_random, "verify_prepared", lambda root: {"files": {}})
    monkeypatch.setattr(
        verify_random,
        "upstream_test_sources",
        lambda root, tests: {"test_random.py": "original"},
    )
    monkeypatch.setattr(verify_random, "load_index", lambda root, target: {})
    monkeypatch.setattr(
        verify_random,
        "load_reductions",
        lambda root, target: {"descriptors": {"w32/all_reduce_andbool_": {}}},
    )
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(verify_random.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="worker failed"):
        verify_random.verify(args)
    evidence = json.loads((args.output_dir / "evidence.json").read_text())
    assert not evidence["passed"]
    assert not evidence["fullUpstreamSuite"] and not evidence["fullTranslatedBackend"]
    modes = ("cpu", "native", *verify_random.NEGATIVE_CHECKS)
    assert list(evidence["workers"]) == list(modes)
    for mode, command in zip(modes, commands):
        assert command[command.index("--worker") + 1] == mode
        assert command[command.index("--timeout-seconds") + 1] == (
            "1800" if mode == "native" else "120"
        )
        assert (args.output_dir / (mode + ".command.json")).exists()


def test_random_source_attestation_checks_the_actual_upstream_test_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "python/tests/test_random.py"
    path.parent.mkdir(parents=True)
    original = b"# unchanged test source\n"
    path.write_bytes(original)
    calls = []

    def read(command, **kwargs):
        calls.append(command)
        return original

    monkeypatch.setattr(verify_random.subprocess, "check_output", read)
    assert verify_random.upstream_test_sources(
        tmp_path, verify_random.UPSTREAM_TESTS
    ) == {"python/tests/test_random.py": hashlib.sha256(original).hexdigest()}
    assert calls[0][-1] == "HEAD:python/tests/test_random.py"
    path.write_bytes(b"# modified\n")
    with pytest.raises(ValueError, match="modified"):
        verify_random.upstream_test_sources(tmp_path, verify_random.UPSTREAM_TESTS)
