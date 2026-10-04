"""Storage validation and guarded host dispatch for pinned general gather."""

import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    gather_evidence,
    gather_layout,
    gather_packages,
    gather_workloads,
    runtime,
    verify_gather,
)
from tests.ci_helpers import assert_paths_covered

ENTRY = "gatherfloat32int32_1_1_int"


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize(
    "fault",
    (
        None,
        "missing",
        "duplicate",
        "value",
        "dtype",
        "shape",
        "source",
        "count",
        "start",
    ),
)
def test_gather_verifier_requires_complete_exact_records(native, fault):
    import numpy as np

    records = []
    for i, case in enumerate(gather_workloads.cases()):
        _, _, expected = gather_workloads.reference(np, case)
        records.append(
            {
                **case,
                "dtype": "mlx.core." + case["dtype"].removesuffix("_"),
                "actual": gather_workloads.words(np, expected),
                "expected": gather_workloads.words(np, expected),
                "shape": list(expected.shape),
                "inputUnchanged": True,
                "dispatchStart": i if native else 0,
                "dispatchCount": int(native),
            }
        )
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records[-1] = records[0]
    elif fault == "value":
        records[0]["actual"][0] ^= 1
    elif fault == "dtype":
        records[0]["dtype"] = "mlx.core.int32"
    elif fault == "shape":
        records[0]["shape"] = [18]
    elif fault == "source":
        records[0]["inputUnchanged"] = False
    elif fault == "count":
        records[0]["dispatchCount"] = not native
    elif fault == "start":
        records[1]["dispatchStart"] += 1
    if fault:
        with pytest.raises(ValueError):
            verify_gather.validate_records(np, records, native=native)
    else:
        verify_gather.validate_records(np, records, native=native)


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize(
    "fault",
    (
        None,
        "testsRun",
        "failures",
        "errors",
        "skipped",
        "dispatchStart",
        "dispatchCount",
    ),
)
def test_gather_upstream_requires_execution_without_skips(native, fault):
    summary = {
        "testsRun": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "dispatchStart": 42 if native else 0,
        "dispatchCount": 6 if native else 0,
    }
    if fault:
        summary[fault] = 0 if summary[fault] else 1
        with pytest.raises(ValueError):
            verify_gather.validate_upstream(summary, native=native)
    else:
        verify_gather.validate_upstream(summary, native=native)


def evidence_event(tmp_path):
    import numpy as np

    supplied, memory = buffers()
    inputs = {}
    bindings = {}
    for name, buffer in supplied.items():
        kind = buffer.dtype.decode()
        dtype = "bool" if kind == "bool_" else kind
        values = gather_layout.values(buffer)
        if dtype == "bool":
            values = [bool(value) for value in values]
        if kind == "float32":
            values = np.array(values, dtype=np.float32).view(np.uint32).tolist()
        if name == "out":
            values = [runtime.COPY_GUARD[0]] * 6 + runtime.COPY_GUARD
        value = {"dtype": dtype, "shape": [len(values)], "values": values}
        if kind == "float32":
            value["encoding"] = "ieee754-binary32"
        inputs[name] = value
        bindings[name] = {key: item for key, item in value.items() if key != "values"}
        bindings[name]["binding"] = {
            "metadata": {"scalarLayout": {"elementType": dtype}}
        }
    module = tmp_path / "test.metallib"
    module.write_bytes(b"retained-module")
    air = tmp_path / "test.air"
    air.write_bytes(b"retained-validation")
    artifact = tmp_path / "source.metal"
    artifact.write_bytes(b"retained-source")
    values = np.array([9, 10, 11, 0, 1, 2], dtype=np.float32)
    execution = runtime.Launch((2, 1, 3), (1, 1, 1)).execution()
    event = {
        "entry": ENTRY,
        "target": "metal",
        "threads": 6,
        **execution,
        "dispatchVersion": 3,
        "inputs": inputs,
        "gatherValues": values.view(np.uint32).tolist(),
        "gatherGuardValues": runtime.COPY_GUARD.copy(),
        "gatherStorageType": "float32",
        "gatherMetadata": gather_layout.validate(ENTRY, supplied, 6, execution),
        "outputHash": hashlib.sha256(values.tobytes()).hexdigest(),
        "artifact": {
            "packagePath": artifact.name,
            "sizeBytes": artifact.stat().st_size,
            "hash": {"value": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        },
        "packageRoot": str(tmp_path),
        "details": {
            "request": {
                "entryPoint": ENTRY,
                "target": "metal",
                "buffers": bindings,
                "dispatch": execution,
            },
            "module": {
                "file": str(module),
                "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            },
            "validationModules": [
                {
                    "file": str(air),
                    "sha256": hashlib.sha256(air.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    identity = {key: event["artifact"][key] for key in ("hash", "sizeBytes")}
    event["details"]["artifactIdentityVerification"] = {
        "verificationStatus": "verified",
        "target": "metal",
        "expectedIdentity": identity,
        "observedIdentity": identity,
    }
    event["details"]["nativeRuntimeDispatch"] = event["details"]["request"]
    event["details"]["adapterSteps"] = [
        {"action": action, "status": "passed"}
        for action in (
            "compile-metal-for-native-runtime",
            "link-metal-for-native-runtime",
        )
    ]
    return event


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "result",
        "guard",
        "hash",
        "index",
        "metadata",
        "geometry",
        "module",
        "compiler",
        "source",
        "binding",
        "entry",
        "encoding",
        "validation",
        "identity",
    ),
)
def test_gather_audit_reconciles_native_storage_and_rejects_corruption(tmp_path, fault):
    import numpy as np

    event = evidence_event(tmp_path)
    if fault == "result":
        event["gatherValues"][0] ^= 1
    elif fault == "guard":
        event["gatherGuardValues"][-1] = 0
    elif fault == "hash":
        event["outputHash"] = "0" * 64
    elif fault == "index":
        event["inputs"]["idx0"]["values"][0] = 2
    elif fault == "metadata":
        event["gatherMetadata"]["sourceStrides"][0] = 1
    elif fault == "geometry":
        event["workgroupCount"] = [6, 1, 1]
    elif fault == "module":
        (tmp_path / "test.metallib").write_bytes(b"changed")
    elif fault == "compiler":
        event["details"]["validationModules"] = []
    elif fault == "source":
        (tmp_path / "source.metal").write_bytes(b"changed")
    elif fault == "binding":
        event["details"]["request"]["buffers"]["src"]["shape"] = [11]
    elif fault == "entry":
        event["details"]["request"]["entryPoint"] = "other"
    elif fault == "encoding":
        del event["inputs"]["src"]["encoding"]
    elif fault == "validation":
        event["details"]["adapterSteps"][0]["status"] = "skipped"
    elif fault == "identity":
        event["details"]["artifactIdentityVerification"][
            "verificationStatus"
        ] = "unverified"
    if fault:
        with pytest.raises(ValueError):
            gather_evidence.audit_event(np, event)
    else:
        source, indices, values = gather_evidence.audit_event(np, event)
        assert source.shape == (4, 3) and indices[0].tolist() == [3, 0]
        assert values == event["gatherValues"]


def test_gather_ci_requires_unchanged_tests_on_all_three_targets():
    import re

    import yaml

    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["slice-update-host"]
    assert {
        row["target"]: row["os"] for row in job["strategy"]["matrix"]["include"]
    } == {"directx": "windows-2025", "opengl": "ubuntu-24.04", "metal": "macos-26"}
    assert (
        job["needs"] == "integer64-host"
        and job["if"] == "github.event_name != 'schedule'"
        and "continue-on-error" not in job
    )
    steps = {step.get("name"): step for step in job["steps"]}
    step = steps["Execute general gather host operations"]
    assert "if" not in step and "continue-on-error" not in step
    assert (
        "set -euo pipefail" in step["run"] and "--timeout-seconds 3600" in step["run"]
    )
    assert "portable_host.verify_gather" in step["run"]
    assert (
        "--packages" in step["run"]
        and "--integer64" in step["run"]
        and "--reductions" in step["run"]
    )
    assert "gather-evidence" in step["run"] and "tee" in step["run"]
    retained = steps["Retain slice update execution evidence"]
    assert (
        retained["if"] == "always()"
        and retained["with"]["include-hidden-files"] is True
    )
    assert retained["with"]["if-no-files-found"] == "error"
    deadlines = [
        int(value)
        for step in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", step.get("run", ""))
    ]
    assert sum(deadlines) + 1800 < job["timeout-minutes"] * 60
    for trigger in ("pull_request", "push"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[trigger]["paths"],
            "demos/integrations/mlx/tests/host/test_portable_gather.py",
        )


def buffers(dtype="float32", index_dtype="int32"):
    data = {
        "src": (dtype, list(range(12))),
        "out": (dtype, [-1] * 6),
        "src_shape": ("int32", [4, 3]),
        "src_strides": ("int64", [3, 1]),
        "src_ndim": ("uint64", [2]),
        "slice_sizes": ("int32", [1, 3]),
        "axes": ("int32", [0]),
        "idx_shapes": ("int32", [2]),
        "idx_strides": ("int64", [1]),
        "idx_contigs": ("bool_", [1]),
        "idx_ndim": ("int32", [1]),
        "idx0": (index_dtype, [3, 0]),
    }
    memory = {
        name: (runtime.TYPES[kind] * len(values))(*values)
        for name, (kind, values) in data.items()
    }
    supplied = {
        name: runtime.Buffer(
            name.encode(),
            kind.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == "out"),
        )
        for name, (kind, values) in data.items()
    }
    return supplied, memory


@pytest.mark.parametrize("dtype", gather_layout.TYPES)
@pytest.mark.parametrize("index_dtype", ("int32", "uint32", "int64", "uint64"))
def test_gather_layout_matches_all_supported_storage_types(dtype, index_dtype):
    supplied, memory = buffers(dtype, index_dtype)
    entry = f"gather{dtype}{index_dtype}_1_1_int"
    result = gather_layout.validate(
        entry, supplied, 6, runtime.Launch((2, 1, 3), (1, 1, 1)).execution()
    )
    assert result["maximumIndex"] == 11
    assert result["sourceCount"] == 12 and result["outputCount"] == 6
    if index_dtype.startswith("int"):
        memory["idx0"][0] = -1
        assert (
            gather_layout.validate(
                entry, supplied, 6, runtime.Launch((2, 1, 3), (1, 1, 1)).execution()
            )
            == result
        )


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "dtype",
        "null",
        "direction",
        "rank",
        "rank-length",
        "index-rank",
        "source-span",
        "negative-stride",
        "source-shape",
        "slice",
        "axis",
        "index-span",
        "index-shape",
        "contiguity",
        "positive-index",
        "negative-index",
        "launch",
        "overlap",
    ),
)
def test_gather_rejects_invalid_metadata_before_dispatch(fault):
    supplied, memory = buffers()
    execution = runtime.Launch((2, 1, 3), (1, 1, 1)).execution()
    if fault == "missing":
        del supplied["idx0"]
    elif fault == "dtype":
        supplied["src"].dtype = b"uint32"
    elif fault == "null":
        supplied["src"].data = None
    elif fault == "direction":
        supplied["out"].output = 0
    elif fault == "rank":
        memory["src_ndim"][0] = 1
    elif fault == "rank-length":
        supplied["src_ndim"].count = 2
    elif fault == "index-rank":
        memory["idx_ndim"][0] = 2
    elif fault == "source-span":
        supplied["src"].count = 11
    elif fault == "negative-stride":
        memory["src_strides"][0] = -3
    elif fault == "source-shape":
        memory["src_shape"][0] = 0
    elif fault == "slice":
        memory["slice_sizes"][1] = 4
    elif fault == "axis":
        memory["axes"][0] = 2
    elif fault == "index-span":
        supplied["idx0"].count = 1
    elif fault == "index-shape":
        memory["idx_shapes"][0] = 0
    elif fault == "contiguity":
        memory["idx_contigs"][0] = 2
    elif fault == "positive-index":
        memory["idx0"][0] = 4
    elif fault == "negative-index":
        memory["idx0"][0] = -5
    elif fault == "launch":
        execution["workgroupCount"][0] = 3
    elif fault == "overlap":
        supplied["out"].data = supplied["src"].data
    with pytest.raises(ValueError):
        gather_layout.validate(ENTRY, supplied, 6, execution)


@pytest.mark.parametrize(
    "entry",
    (
        "gatherfloat32int32_0_1_int",
        "gatherfloat32int32_11_1_int",
        "gatherfloat32int32_1_65_int",
        "gatherfloat32int32_01_1_int",
        "gatherfloat32float32_1_1_int",
        "gatherfloat32int32_1_1_int64_t",
    ),
)
def test_gather_rejects_unknown_specializations(entry):
    with pytest.raises(ValueError):
        gather_layout.signature(entry)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "dtype", "shape", "encoding", "missing", "binding", "metadata"),
)
def test_gather_host_writeback_requires_valid_native_readback(
    tmp_path, monkeypatch, target, fault
):
    supplied, memory = buffers()
    original = bytes(memory["out"])
    calls = []
    descriptor = {
        "artifact": {"sha256": "a" * 64},
        "bindings": [
            {
                "name": f"binding{i}",
                "scalarLayout": {
                    "memberName": (
                        (ENTRY + "_" if target == "directx" else "")
                        + ("out_" if name == "out" else name)
                    ),
                    "elementType": runtime.physical_dtype(
                        buffer.dtype.decode(), target
                    ),
                    "elementStrideBytes": (
                        1
                        if buffer.dtype == b"bool_" and target == "metal"
                        else (
                            4
                            if buffer.dtype == b"bool_"
                            else ctypes.sizeof(runtime.TYPES[buffer.dtype.decode()])
                        )
                    ),
                },
            }
            for i, (name, buffer) in enumerate(supplied.items())
        ],
    }
    if fault == "binding":
        descriptor["bindings"][-1]["scalarLayout"]["memberName"] = "unknown"
    if fault == "metadata":
        memory["idx0"][0] = 4

    def package(entry, maximum):
        calls.append((entry, maximum))
        return descriptor, tmp_path

    def request(_descriptor, _directory, inputs, outputs, execution, **kwargs):
        assert kwargs == {"expected_target": target}
        assert inputs["binding0"]["encoding"] == "ieee754-binary32"
        assert inputs["binding0"]["values"][1] == 0x3F800000
        return SimpleNamespace(inputs=inputs, outputs=outputs)

    def execute(_host, request):
        readback = dict(
            request.outputs["binding1"],
            values=[0x7FC01234, 0x80000000, 0, 1, 2, 3] + runtime.COPY_GUARD,
        )
        if fault == "guard":
            readback["values"][-1] = 0
        elif fault == "dtype":
            readback["dtype"] = "int32"
        elif fault == "shape":
            readback["shape"] = [6]
        elif fault == "encoding":
            del readback["encoding"]
        return SimpleNamespace(
            status="ok",
            outputs={} if fault == "missing" else {"binding1": readback},
            details={},
        )

    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=package),
        executor=SimpleNamespace(run=execute),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, ENTRY, table, len(table), 6, runtime.Launch((2, 1, 3), (1, 1, 1))
            )
        assert bytes(memory["out"]) == original and host.dispatch_count == 0
        assert not host.trace.exists()
        if fault == "metadata":
            assert not calls
    else:
        gather_dispatch.dispatch(
            host, ENTRY, table, len(table), 6, runtime.Launch((2, 1, 3), (1, 1, 1))
        )
        words = list(
            ctypes.cast(memory["out"], ctypes.POINTER(ctypes.c_uint32 * 6)).contents
        )
        assert words == [0x7FC01234, 0x80000000, 0, 1, 2, 3]
        assert host.dispatch_count == 1 and calls == [(ENTRY, 11)]
        assert json.loads(host.trace.read_text())["gatherValues"] == words


def test_gather_cache_separates_ranges_and_rechecks_source(tmp_path, monkeypatch):
    cache = gather_packages.GatherPackageCache(tmp_path, tmp_path / "cache", "opengl")
    calls = []
    monkeypatch.setattr(cache, "_require_source", lambda: None)
    monkeypatch.setattr(
        gather_packages, "translation_implementation_hash", lambda: "a" * 64
    )

    def build(entry, output, identity):
        calls.append(identity)
        (output / "index.json").write_text(json.dumps(identity))

    monkeypatch.setattr(cache, "_build", build)
    monkeypatch.setattr(
        cache,
        "_load",
        lambda directory, identity: (
            json.loads((directory / "index.json").read_text()),
            directory,
        ),
    )
    first = cache.get(ENTRY, 11)
    assert cache.get(ENTRY, 11) == first and len(calls) == 1
    assert cache.get(ENTRY, 19)[1] != first[1] and len(calls) == 2

    def reject():
        raise ValueError("Source changed")

    monkeypatch.setattr(cache, "_require_source", reject)
    with pytest.raises(ValueError, match="Source changed"):
        cache.get(ENTRY, 11)


def test_gather_cache_does_not_publish_mixed_implementation(tmp_path, monkeypatch):
    cache = gather_packages.GatherPackageCache(tmp_path, tmp_path / "cache", "metal")
    monkeypatch.setattr(cache, "_require_source", lambda: None)
    hashes = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(
        gather_packages, "translation_implementation_hash", lambda: next(hashes)
    )
    monkeypatch.setattr(cache, "_build", lambda *args: None)
    with pytest.raises(ValueError, match="Translator changed"):
        cache.get(ENTRY, 11)
    assert not list(cache.directory.iterdir())


def test_gather_cache_does_not_publish_mixed_recipe(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.py"
    recipe.write_text("original")
    monkeypatch.setattr(gather_packages, "__file__", str(recipe))
    monkeypatch.setattr(
        gather_packages, "translation_implementation_hash", lambda: "a" * 64
    )
    cache = gather_packages.GatherPackageCache(tmp_path, tmp_path / "cache", "metal")
    monkeypatch.setattr(cache, "_require_source", lambda: None)
    monkeypatch.setattr(cache, "_build", lambda *args: recipe.write_text("changed"))
    with pytest.raises(ValueError, match="recipe changed"):
        cache.get(ENTRY, 11)
    assert not list(cache.directory.iterdir())


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "identity", "record", "target", "stage", "entry", "source", "loader"),
)
def test_gather_cache_validates_source_entry_and_saved_descriptor(
    tmp_path, monkeypatch, target, fault
):
    cache = gather_packages.GatherPackageCache(tmp_path, tmp_path / "cache", target)
    identity = {"entry": ENTRY}
    descriptor = {
        "target": target,
        "stage": "compute",
        "entryPoint": {
            "name": {"metal": ENTRY, "opengl": "main", "directx": "CSMain"}[target]
        },
        "source": {
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(b"wrapper").hexdigest(),
            }
        },
    }
    if fault in {"target", "stage"}:
        descriptor[fault] = "other"
    elif fault == "entry":
        descriptor["entryPoint"]["name"] = "other"
    elif fault == "source":
        descriptor["source"]["hash"]["value"] = "0" * 64
    record = {
        "identity": {} if fault == "identity" else identity,
        "descriptor": {} if fault == "record" else descriptor,
    }
    (tmp_path / "index.json").write_text(json.dumps(record))
    monkeypatch.setattr(gather_packages, "source", lambda root, entry: "wrapper")
    monkeypatch.setattr(
        gather_packages,
        "build_runtime_loader_manifest",
        lambda path: {
            "success": fault != "loader",
            "loadUnits": [{}],
            "diagnostics": [],
        },
    )
    monkeypatch.setattr(
        gather_packages, "build_native_loader_abi_descriptor", lambda loader: descriptor
    )
    if fault:
        with pytest.raises(ValueError):
            cache._load(tmp_path, identity)
    else:
        assert cache._load(tmp_path, identity) == (descriptor, tmp_path / "package")


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_gather_audit_accepts_target_entry_and_native_module_contract(tmp_path, target):
    import numpy as np

    event = evidence_event(tmp_path)
    event["target"] = target
    request = event["details"]["request"]
    request["target"] = target
    request["entryPoint"] = "main" if target == "opengl" else "CSMain"
    for table in (event["inputs"], request["buffers"]):
        table["idx_contigs"]["dtype"] = "uint32"
    event["inputs"]["idx_contigs"]["values"] = [1]
    request["buffers"]["idx_contigs"]["binding"]["metadata"]["scalarLayout"][
        "elementType"
    ] = "uint32"
    event["details"]["artifactIdentityVerification"]["target"] = target
    module = tmp_path / ("native.dxil" if target == "directx" else "native.glsl")
    module.write_bytes(b"native module")
    event["details"]["module"] = {
        "file": str(module),
        "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }
    event["details"]["validationModules"] = []
    event["details"]["adapterSteps"] = [
        {
            "action": (
                "compile-hlsl-for-directx-runtime"
                if target == "directx"
                else "validate-glsl-for-opengl-runtime"
            ),
            "status": "passed",
        }
    ]
    gather_evidence.audit_event(np, event)
    event["details"]["adapterSteps"][0]["status"] = "skipped"
    with pytest.raises(ValueError, match="compilation"):
        gather_evidence.audit_event(np, event)
