"""Scan callbacks retain exact source geometry and validate storage before reads."""

import copy
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    runtime,
    scan_dispatch,
    scan_layout,
    scan_packages,
    verify_scans,
)
from tests import workflow_text


def buffers(dtype="float32", axis=129, stride=1, rows=3):
    entry = (
        f"{'contig' if stride == 1 else 'strided'}_scan_inclusive_sum_{dtype}_{dtype}"
    )
    size = axis * stride * rows
    data = {"in": (dtype, size), "out": (dtype, size), "axis_size": ("uint64", 1)}
    if stride > 1:
        data.update({"stride": ("uint64", 1), "stride_blocks": ("uint64", 1)})
    memory = {
        name: (scan_layout.TYPES[kind] * count)()
        for name, (kind, count) in data.items()
    }
    memory["axis_size"][0] = axis
    if stride > 1:
        memory["stride"][0] = stride
        memory["stride_blocks"][0] = (stride + 31) // 32
    supplied = {
        name: runtime.Buffer(
            name.encode(),
            kind.encode(),
            ctypes.addressof(memory[name]),
            count,
            int(name == "out"),
        )
        for name, (kind, count) in data.items()
    }
    return (
        entry,
        supplied,
        memory,
        size,
        scan_layout.execution(entry, size, axis, stride),
    )


@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16", "int64"))
@pytest.mark.parametrize(
    "axis",
    (1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025, 2048, 4096, 4097, 8192, 8193),
)
def test_scan_geometry_follows_source_item_width(dtype, axis):
    entry, supplied, memory, size, launch = buffers(dtype, axis)
    metadata = scan_layout.validate(entry, supplied, size, launch)
    assert metadata["axisSize"] == axis
    assert launch["workgroupCount"] == [1, 3, 1]
    reads = 2 if dtype == "int64" else 4
    # Independent form of the two source thresholds and subgroup rounding.
    units = axis if axis <= reads * 1024 else axis // 2
    width = min(1024, ((units + reads * 32 - 1) // (reads * 32)) * 32)
    if axis > reads * 2048:
        width = 1024
    assert launch["workgroupSize"] == [width, 1, 1]


@pytest.mark.parametrize(
    "dtype,width", (("float32", 256), ("bfloat16", 256), ("int64", 512))
)
@pytest.mark.parametrize("stride", (2, 31, 32, 33, 65))
@pytest.mark.parametrize("axis", (1, 31, 32, 33))
def test_strided_scan_preserves_partial_column_blocks(dtype, width, stride, axis):
    entry, supplied, memory, size, launch = buffers(dtype, axis, stride)
    metadata = scan_layout.validate(entry, supplied, size, launch)
    assert metadata["strideBlocks"] == (stride + 31) // 32
    assert launch == {
        "workgroupSize": [width, 1, 1],
        "workgroupCount": [1, 3 * ((stride + 31) // 32), 1],
    }


@pytest.mark.parametrize(
    "fault",
    (
        "dtype",
        "count",
        "direction",
        "null",
        "unaligned",
        "overflow",
        "axis",
        "width",
        "groups",
        "overlap",
        "extra",
        "partial",
    ),
)
def test_invalid_scan_requests_are_rejected_before_native_execution(fault):
    entry, supplied, memory, size, launch = buffers()
    if fault == "dtype":
        supplied["in"].dtype = b"int32"
    elif fault == "count":
        supplied["in"].count -= 1
    elif fault == "direction":
        supplied["in"].output = 1
    elif fault == "null":
        supplied["in"].data = None
    elif fault == "unaligned":
        supplied["in"].data += 1
    elif fault == "overflow":
        supplied["in"].data = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 4
    elif fault == "axis":
        memory["axis_size"][0] = 1 << 32
    elif fault == "width":
        launch["workgroupSize"][0] = 32
    elif fault == "groups":
        launch["workgroupCount"][1] += 1
    elif fault == "overlap":
        supplied["out"].data = supplied["axis_size"].data
    elif fault == "extra":
        supplied["unused"] = supplied["in"]
    else:
        supplied["out"].data = supplied["in"].data + 4
    with pytest.raises(ValueError):
        scan_layout.validate(entry, supplied, size, launch)


def test_scan_donation_only_accepts_an_identical_typed_input_view():
    entry, supplied, memory, size, launch = buffers()
    supplied["out"].data = supplied["in"].data
    assert scan_layout.validate(entry, supplied, size, launch)


@pytest.mark.parametrize(
    "entry",
    (
        "contig_scan_inclusive_sum_bool__int32",
        "strided_scan_reverse_exclusive_sum_bool__uint32",
    ),
)
def test_scan_signature_preserves_boolean_sum_promotion(entry):
    assert scan_layout.signature(entry)[1:3] == ("sum", "bool_")


@pytest.mark.parametrize(
    "entry",
    (
        "contig_scan_inclusive_sum_bool__bool_",
        "contig_scan_inclusive_logaddexp_int32_int32",
        "contig_scan_inclusive_sum_complex64_complex64",
        "other",
        None,
    ),
)
def test_unimplemented_host_scalar_types_remain_explicit(entry):
    with pytest.raises(ValueError, match="Unknown scalar"):
        scan_layout.signature(entry)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "dtype,words",
    (
        ("float32", [0, 0x80000000, 1, 0x7FA12345, 0x7F800000]),
        ("float16", [0, 0x8000, 1, 0x7C01, 0x7C00]),
        ("bfloat16", [0, 0x8000, 1, 0x7F81, 0x7F80]),
        ("bool_", [0, 1, 1, 0]),
        ("int32", [-(1 << 31), -1, 0, (1 << 31) - 1]),
        ("uint64", [0, (1 << 64) - 1]),
    ),
)
def test_scan_transport_preserves_source_words(target, dtype, words):
    value = scan_dispatch.packet(dtype, target, words)
    assert scan_dispatch.unpack(dtype, target, value["values"]) == words
    assert value["shape"] == [len(words)]


@pytest.mark.parametrize(
    "dtype,word",
    (("float32", -1), ("int32", 1 << 31), ("bool_", 2), ("uint64", 1 << 64)),
)
def test_scan_readback_rejects_out_of_range_words(dtype, word):
    with pytest.raises(ValueError):
        scan_dispatch.unpack(dtype, "opengl", [word])


@pytest.mark.parametrize(
    "fault", ("count", "launch", "source", "duplicate", "direction")
)
def test_invalid_dispatch_does_not_request_a_package(tmp_path, fault):
    entry, supplied, memory, size, execution = buffers()
    requested = []
    host = SimpleNamespace(
        scans=SimpleNamespace(get=lambda *args: requested.append(args))
    )
    batch = (runtime.Buffer * 3)(*supplied.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    count = 3
    if fault == "count":
        count = 2
    elif fault == "launch":
        launch.workgroup_size[0] = 32
    elif fault == "source":
        host.scans = None
    elif fault == "duplicate":
        batch[1].name = batch[0].name
    else:
        batch[0].output = 1
    with pytest.raises(ValueError):
        scan_dispatch.dispatch(host, entry, batch, count, size, launch)
    assert not requested


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "status",
        "missing",
        "dtype",
        "shape",
        "encoding",
        "count",
        "guard",
        "range",
    ),
)
def test_scan_dispatch_validates_readback_before_writing_host_storage(
    tmp_path, monkeypatch, fault
):
    entry, supplied, memory, size, geometry = buffers("int32", axis=1)
    memory["out"][:] = [-99] * size
    descriptor = {
        "artifact": {},
        "bindings": [
            {
                "name": name,
                "provenance": {"sourceResource": {"parameter": name}},
                "scalarLayout": {
                    "elementType": buffer.dtype.decode(),
                    "elementStrideBytes": ctypes.sizeof(
                        scan_layout.TYPES[buffer.dtype.decode()]
                    ),
                },
            }
            for name, buffer in supplied.items()
        ],
    }
    requests = []

    def request(*args, **kwargs):
        assert kwargs["observe_inputs"] == ("in", "axis_size")
        requests.append(args)
        return {}

    monkeypatch.setattr(scan_dispatch, "build_native_loader_dispatch_request", request)
    output = {
        "dtype": "int32",
        "shape": [size + 32],
        "values": [1] * size + scan_dispatch.guard("int32"),
    }
    result = SimpleNamespace(status="ok", outputs={"out": output}, details={})
    if fault == "status":
        result.status = "failed"
    elif fault == "missing":
        result.outputs = {}
    elif fault == "dtype":
        output["dtype"] = "uint32"
    elif fault == "shape":
        output["shape"] = [size]
    elif fault == "encoding":
        output["encoding"] = "ieee754-binary32"
    elif fault == "count":
        output["values"].pop()
    elif fault == "guard":
        output["values"][-1] = 0
    elif fault == "range":
        output["values"][0] = 1 << 32
    monkeypatch.setattr(scan_dispatch, "execute", lambda *args: result)
    host = SimpleNamespace(
        scans=SimpleNamespace(get=lambda *args: (descriptor, tmp_path)),
        target="metal",
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    batch = (runtime.Buffer * 3)(*supplied.values())
    launch = runtime.Launch(
        tuple(geometry["workgroupCount"]), tuple(geometry["workgroupSize"])
    )
    if fault:
        with pytest.raises((RuntimeError, ValueError)):
            scan_dispatch.dispatch(host, entry, batch, 3, size, launch)
        assert list(memory["out"]) == [-99] * size
        assert host.dispatch_count == 0 and not host.trace.exists()
    else:
        scan_dispatch.dispatch(host, entry, batch, 3, size, launch)
        assert list(memory["out"]) == [1] * size
        assert host.dispatch_count == 1 and host.trace.is_file()
    assert len(requests) == 1


def test_scan_cache_keys_bind_launch_source_and_implementation(tmp_path, monkeypatch):
    root = tmp_path / "source"
    source = root / scan_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("original")
    monkeypatch.setattr(
        scan_packages.ScanPackageCache, "_require_source", lambda self: None
    )
    implementation = ["a" * 64]
    monkeypatch.setattr(
        scan_packages, "translation_implementation_hash", lambda: implementation[0]
    )
    built = []

    def build(self, output, identity):
        built.append(identity)
        (output / "identity.json").write_text(json.dumps(identity))

    def load(self, directory, identity):
        assert json.loads((directory / "identity.json").read_text()) == identity
        return directory

    monkeypatch.setattr(scan_packages.ScanPackageCache, "_build", build)
    monkeypatch.setattr(scan_packages.ScanPackageCache, "_load", load)
    cache = scan_packages.ScanPackageCache(root, tmp_path / "cache", "metal")
    entry, supplied, memory, size, geometry = buffers("int32", axis=1)
    metadata = scan_layout.validate(entry, supplied, size, geometry)
    first = cache.get(entry, metadata, geometry)
    assert cache.get(entry, metadata, geometry) == first and len(built) == 1
    wider = dict(metadata, axisSize=129, storageElements=387)
    assert cache.get(entry, wider, scan_layout.execution(entry, 387, 129)) != first
    second = dict(metadata, axisSize=2, storageElements=6)
    assert cache.get(entry, second, geometry) != first
    more_rows = dict(metadata, storageElements=4)
    assert cache.get(entry, more_rows, scan_layout.execution(entry, 4, 1)) != first
    implementation[0] = "b" * 64
    assert cache.get(entry, metadata, geometry) != first
    source.write_text("changed")
    assert cache.get(entry, metadata, geometry) != first
    assert len(built) == 6


@pytest.mark.parametrize("fault", ("implementation", "recipe"))
def test_scan_cache_does_not_publish_during_source_changes(
    tmp_path, monkeypatch, fault
):
    source = tmp_path / scan_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("original")
    monkeypatch.setattr(
        scan_packages.ScanPackageCache, "_require_source", lambda self: None
    )
    implementation = ["a" * 64]
    monkeypatch.setattr(
        scan_packages, "translation_implementation_hash", lambda: implementation[0]
    )

    def build(self, output, identity):
        if fault == "implementation":
            implementation[0] = "b" * 64
        else:
            identity["recipeHash"] = "0" * 64

    monkeypatch.setattr(scan_packages.ScanPackageCache, "_build", build)
    cache = scan_packages.ScanPackageCache(tmp_path, tmp_path / "cache", "metal")
    entry, supplied, memory, size, geometry = buffers("int32", axis=1)
    metadata = scan_layout.validate(entry, supplied, size, geometry)
    with pytest.raises(ValueError, match="changed during translation"):
        cache.get(entry, metadata, geometry)
    assert not list(cache.directory.iterdir())


def test_scan_verifier_cases_cover_storage_and_launch_boundaries():
    cases = verify_scans.cases()
    assert len(cases) == len({case["id"] for case in cases}) == 54
    assert {case["layout"] for case in cases} == {"row", "column", "strided-column"}
    assert {case["axisSize"] for case in cases} == {1, 31, 32, 33, 65, 129, 257}
    assert {case["operation"] for case in cases} == {"sum", "min", "max"}


def synthetic_event(case, target):
    storage, view = verify_scans.operand(np, case)
    expected = verify_scans.reference(np, case, view)
    saved = {"before": storage.copy(), "after": storage.copy(), "actual": expected}
    dtype = case["dtype"]
    raw = storage[7 : 7 + view.size]
    if dtype == "float32":
        raw = raw.view(np.uint32)
    inputs = {
        "in": scan_dispatch.packet(
            dtype, target, raw.tolist() + scan_dispatch.guard(dtype)
        ),
        "out": scan_dispatch.packet(
            dtype, target, scan_dispatch.guard(dtype)[:1] * (view.size + 32)
        ),
        "axis_size": scan_dispatch.packet("uint64", target, [case["axisSize"]]),
    }
    stride = 33 if case["layout"] == "strided-column" else 1
    if stride > 1:
        inputs.update(
            stride=scan_dispatch.packet("uint64", target, [stride]),
            stride_blocks=scan_dispatch.packet("uint64", target, [2]),
        )
    bindings, observations, conditions = {}, {}, []
    for name, value in inputs.items():
        bindings[name] = {
            **{key: item for key, item in value.items() if key != "values"},
            "binding": {
                "metadata": {
                    "scalarLayout": {
                        "elementType": value["dtype"],
                        "elementStrideBytes": np.dtype(value["dtype"]).itemsize,
                        "memberOffsetBytes": 0,
                    },
                    "provenance": {
                        "sourceResource": {
                            "parameter": name,
                            "entryPoint": case["entry"],
                            "backend": "metal",
                        }
                    },
                }
            },
        }
        if name != "out":
            data = np.asarray(
                value["values"],
                dtype="uint32" if value.get("encoding") else value["dtype"],
            ).tobytes()
            digest = hashlib.sha256(data).hexdigest()
            observations[name] = {
                "unchanged": True,
                "observedBytes": data.hex(),
                "observedSHA256": digest,
                "uploadSnapshot": {
                    "sha256": digest,
                    "sizeBytes": len(data),
                    **{key: item for key, item in value.items() if key != "values"},
                },
            }
            if name != "in":
                requirement = {
                    "dtype": "uint64",
                    "minimum": value["values"][0],
                    "maximum": value["values"][0],
                    "sourceResource": {
                        "schemaVersion": 1,
                        "backend": "metal",
                        "entryPoint": case["entry"],
                        "parameter": name,
                        "parameterIndex": {
                            "axis_size": 2,
                            "stride": 3,
                            "stride_blocks": 4,
                        }[name],
                    },
                }
                bindings[name]["binding"]["metadata"]["provenance"][
                    "metadataPrecondition"
                ] = copy.deepcopy(requirement)
                conditions.append(
                    {
                        "binding": name,
                        "value": value["values"][0],
                        "requirement": requirement,
                        "uploadSnapshot": dict(
                            observations[name]["uploadSnapshot"], elementCount=1
                        ),
                    }
                )
    geometry = scan_layout.execution(case["entry"], view.size, case["axisSize"], stride)
    native = {"metal": case["entry"], "opengl": "main", "directx": "CSMain"}[target]
    ordered = np.ascontiguousarray(
        expected.T if case["layout"] == "column" else expected
    )
    words = ordered.view(np.uint32) if dtype == "float32" else ordered
    event = {
        "entry": case["entry"],
        "target": target,
        "threads": view.size,
        "dispatchVersion": 3,
        **copy.deepcopy(geometry),
        "scanMetadata": {
            "axisSize": case["axisSize"],
            "stride": stride,
            "strideBlocks": (stride + 31) // 32,
            "storageElements": view.size,
            "operation": case["operation"],
            "sourceType": dtype,
            "outputType": dtype,
        },
        "inputs": inputs,
        "outputValues": words.ravel().tolist(),
        "outputGuards": scan_dispatch.guard(dtype),
        "outputHash": hashlib.sha256(ordered.tobytes()).hexdigest(),
        "details": {
            "inputObservations": observations,
            "metadataPreconditions": conditions,
            "request": {
                "target": target,
                "entryPoint": native,
                "buffers": bindings,
                "dispatch": {"entryPoint": native, **copy.deepcopy(geometry)},
            },
        },
    }
    return event, saved


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", verify_scans.cases(), ids=lambda case: case["id"])
def test_scan_audit_checks_every_case_and_requires_native_receipts(
    monkeypatch, target, case
):
    event, saved = synthetic_event(case, target)
    receipts = []
    monkeypatch.setattr(verify_scans, "audit_native_execution", receipts.append)
    verify_scans.audit_event(np, case, event, saved)
    assert receipts == [event]


@pytest.mark.parametrize(
    "fault",
    (
        "entry",
        "launch",
        "request",
        "metadata",
        "input",
        "initialization",
        "binding",
        "origin",
        "output",
        "guard",
        "hash",
        "coercion",
        "source",
        "result",
        "observation",
        "observation-bytes",
        "observation-hash",
        "snapshot",
        "receipt",
        "missing-precondition",
        "precondition-bound",
        "precondition-origin",
        "precondition-value",
        "precondition-upload",
        "precondition-duplicate",
        "precondition-binding",
    ),
)
@pytest.mark.parametrize("layout", ("column", "strided-column"))
def test_scan_audit_rejects_corrupt_evidence(monkeypatch, fault, layout):
    case = next(case for case in verify_scans.cases() if case["layout"] == layout)
    event, saved = synthetic_event(case, "directx")
    monkeypatch.setattr(verify_scans, "audit_native_execution", lambda event: None)
    if fault == "entry":
        event["entry"] = "wrong"
    elif fault == "launch":
        event["workgroupSize"][0] = 32
    elif fault == "request":
        event["details"]["request"]["dispatch"]["entryPoint"] = "wrong"
    elif fault == "metadata":
        event["scanMetadata"]["axisSize"] -= 1
    elif fault in {"input", "initialization"}:
        event["inputs"]["in" if fault == "input" else "out"]["values"][0] += 1
    elif fault == "binding":
        event["details"]["request"]["buffers"]["in"]["binding"]["metadata"][
            "scalarLayout"
        ]["elementStrideBytes"] = 8
    elif fault == "origin":
        event["details"]["request"]["buffers"]["in"]["binding"]["metadata"][
            "provenance"
        ]["sourceResource"]["entryPoint"] = "wrong"
    elif fault in {"output", "guard"}:
        event["outputValues" if fault == "output" else "outputGuards"][0] += 1
    elif fault == "hash":
        event["outputHash"] = "0" * 64
    elif fault == "coercion":
        event["outputValues"][0] = float(event["outputValues"][0])
    elif fault == "source":
        saved["after"][0] += 1
    elif fault == "result":
        saved["actual"].flat[0] += 1
    elif fault == "observation":
        event["details"]["inputObservations"].pop("in")
    elif fault in {"observation-bytes", "observation-hash", "snapshot"}:
        observation = event["details"]["inputObservations"]["in"]
        if fault == "observation-bytes":
            data = bytes(len(bytes.fromhex(observation["observedBytes"])))
            observation["observedBytes"] = data.hex()
            observation["observedSHA256"] = hashlib.sha256(data).hexdigest()
            observation["uploadSnapshot"]["sha256"] = observation["observedSHA256"]
        elif fault == "observation-hash":
            observation["observedSHA256"] = "0" * 64
        else:
            observation["uploadSnapshot"]["shape"] = [1]
    elif fault.startswith("precondition-") or fault == "missing-precondition":
        conditions = event["details"]["metadataPreconditions"]
        if fault == "missing-precondition":
            conditions.pop()
        elif fault == "precondition-bound":
            conditions[0]["requirement"]["maximum"] += 1
        elif fault == "precondition-origin":
            conditions[0]["requirement"]["sourceResource"]["parameterIndex"] += 1
        elif fault == "precondition-value":
            conditions[0]["value"] = float(conditions[0]["value"])
        elif fault == "precondition-upload":
            conditions[0]["uploadSnapshot"]["sha256"] = "0" * 64
        elif fault == "precondition-binding":
            event["details"]["request"]["buffers"]["axis_size"]["binding"]["metadata"][
                "provenance"
            ].pop("metadataPrecondition")
        else:
            conditions.append(copy.deepcopy(conditions[0]))
    else:

        def reject(event):
            raise ValueError("Native receipt differs")

        monkeypatch.setattr(verify_scans, "audit_native_execution", reject)
    with pytest.raises(ValueError):
        verify_scans.audit_event(np, case, event, saved)


@pytest.mark.parametrize(
    "fault",
    (
        None,
        "command",
        "missing",
        "commit",
        "target",
        "count",
        "scope",
        "records",
        "repeated",
        "reference",
    ),
)
def test_scan_bounded_verification_requires_complete_result_records(
    tmp_path, monkeypatch, fault
):
    packages, output = tmp_path / "packages", tmp_path / "proof"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "metal"}))

    def run(command, **kwargs):
        assert command[2:4] == ["--timeout-seconds", "1800"]
        assert command[-2:] == ["--output-dir", str(output)]
        output.mkdir()
        records = []
        for case in verify_scans.cases():
            _, view = verify_scans.operand(np, case)
            result = verify_scans.reference(np, case, view)
            records.append(
                {
                    "case": case["id"],
                    "outputHash": hashlib.sha256(result.tobytes()).hexdigest(),
                    "shape": list(result.shape),
                }
            )
        evidence = {
            "passed": True,
            "commit": verify_scans.COMMIT,
            "target": "metal",
            "casesPerPath": len(records),
            "dispatchCount": len(records),
            "fullUpstreamSuite": False,
            "fullScanRouting": False,
            "cpu": copy.deepcopy(records),
            "native": records,
        }
        if fault in {"commit", "target"}:
            evidence[fault] = "wrong"
        elif fault == "count":
            evidence["dispatchCount"] -= 1
        elif fault == "scope":
            evidence["fullScanRouting"] = True
        elif fault == "records":
            evidence["native"].pop()
        elif fault == "repeated":
            evidence["native"][1] = evidence["native"][0]
        elif fault == "reference":
            evidence["native"][0]["outputHash"] = "0" * 64
            evidence["cpu"] = evidence["native"]
        if fault != "missing":
            (output / "evidence.json").write_text(json.dumps(evidence))
        return SimpleNamespace(returncode=1 if fault == "command" else 0)

    monkeypatch.setattr(verify_scans.subprocess, "run", run)
    if fault:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            verify_scans.run_bounded(tmp_path, packages, output)
    else:
        assert verify_scans.run_bounded(tmp_path, packages, output)["passed"] is True


def test_scan_host_checks_share_existing_platform_jobs_and_deadline():
    root = Path(__file__).resolve().parents[5]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    job = workflow_text.workflow_job_text(workflow, "portable-host")
    step = workflow_text.workflow_step_section(
        job, "Execute upstream tests on the translated backend"
    )
    assert "--scans" in step and "--timeout-seconds 4000" in step
    assert "continue-on-error" not in step
    assert "steps.host-build.outcome == 'success'" in step
    assert "steps.host-packages.outcome == 'success'" in step
    for target in ("metal", "opengl", "directx"):
        assert f"target: {target}" in job
