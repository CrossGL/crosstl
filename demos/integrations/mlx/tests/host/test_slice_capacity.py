"""Large destination coverage and independent slice-update evidence checks."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import slice_capacity_workloads as proof
from demos.integrations.mlx.portable_host import verify_slice_updates as verifier
from demos.integrations.mlx.portable_host.runtime import BOOLEAN_GUARD, COPY_GUARD


def evidence(case, target):
    root, update, region = proof.arrays(np, case)
    base = root[1:-1]
    output = proof.reference(np, case)
    strides = [stride // base.itemsize for stride in output.strides]
    events = [proof.small.copy_event(np, target, base, base, strides)]
    normalized = [part.indices(size) for part, size in zip(region, base.shape)]
    destination = [stride * step for stride, (_, _, step) in zip(strides, normalized)]
    indices = np.arange(base.size).reshape(base.shape)[region].reshape(-1)
    if case["operation"] == "replace":
        events.append(
            proof.small.copy_event(
                np, target, update, output, destination, int(indices[0]), True
            )
        )
        return events
    if not update.flags.c_contiguous:
        dense = np.array(update, copy=True, order="C")
        events.append(
            proof.small.copy_event(
                np,
                target,
                update,
                dense,
                [stride // dense.itemsize for stride in dense.strides],
            )
        )
    shapes = {
        "strided": [(2047, 32), (1, 32)],
        "rank3": [(1, 255, 256), (1, 2, 256)] * 2,
        "boundary": [(1, 1)],
    }.get(case["layout"], [(511, 128), (511, 128), (2, 128)])
    current, wanted = base.copy().reshape(-1), output.reshape(-1)
    first = 0
    for shape in shapes:
        count = math.prod(shape)
        selected = indices[first : first + count]
        current[selected] = wanted[selected]
        guard = list(BOOLEAN_GUARD if case["dtype"] == "bool_" else COPY_GUARD)
        if case["dtype"] == "bool_" and target != "metal":
            guard = list(map(int, guard))
        events.append(
            {
                "target": target,
                "entry": "slice_update_" + case["operation"] + case["dtype"],
                "threads": count,
                "workgroupCount": [count, 1, 1],
                "workgroupSize": [1, 1, 1],
                "sliceUpdateMetadata": {
                    "shape": list(shape),
                    "destinationStrides": destination,
                    "destinationOffset": int(selected[0]),
                    "destinationCount": base.size,
                },
                "inputs": {
                    "updates": {
                        "values": proof.storage_values(
                            np, update.reshape(-1)[first : first + count], target
                        )
                    }
                },
                (
                    "sliceUpdateStorageWords"
                    if case["dtype"] == "float32"
                    else "sliceUpdateValues"
                ): proof.storage_values(np, current, target),
                (
                    "sliceUpdateGuardWords"
                    if case["dtype"] == "float32"
                    else "sliceUpdateGuardValues"
                ): guard,
            }
        )
        first += count
    assert first == update.size
    return events


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", list(proof.cases()))
def test_large_slice_evidence_preserves_each_intermediate_allocation(case, target):
    proof.validate_case(case, evidence(case, target), target)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "reordered",
        "gap",
        "stride",
        "upload",
        "upload-type",
        "storage-type",
        "guard",
        "untouched",
        "target",
        "launch",
    ],
)
def test_large_slice_evidence_rejects_incomplete_or_changed_batches(fault):
    case = {"dtype": "float32", "operation": "min", "layout": "strided"}
    trace = evidence(case, "metal")
    if fault == "missing":
        trace.pop()
    elif fault == "duplicate":
        trace.append(trace[-1])
    elif fault == "reordered":
        trace[1:] = reversed(trace[1:])
    elif fault == "gap":
        trace[1]["sliceUpdateMetadata"]["destinationOffset"] += 1
    elif fault == "stride":
        trace[1]["sliceUpdateMetadata"]["destinationStrides"][0] += 1
    elif fault == "upload":
        trace[1]["inputs"]["updates"]["values"][0] ^= 1
    elif fault == "upload-type":
        values = trace[1]["inputs"]["updates"]["values"]
        values[0] = float(values[0])
    elif fault == "storage-type":
        values = trace[1]["sliceUpdateStorageWords"]
        values[0] = float(values[0])
    elif fault == "guard":
        trace[1]["sliceUpdateGuardWords"].pop()
    elif fault == "untouched":
        trace[1]["sliceUpdateStorageWords"][1] ^= 1
    elif fault == "target":
        trace[1]["target"] = "cpu"
    elif fault == "launch":
        trace[1]["workgroupCount"][0] += 1
    with pytest.raises(ValueError, match="Slice capacity"):
        proof.validate_case(case, trace, "metal")


def test_capacity_workloads_cover_all_storage_types_and_operations():
    cases = list(proof.cases())
    assert len(cases) == 16
    assert {case["dtype"] for case in cases} == set(proof.small.copies.DTYPES)
    assert {case["operation"] for case in cases} == {*proof.small.METHODS, "replace"}
    assert {case["count"] for case in cases if "count" in case} == {65535, 65536, 65537}


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "event-target",
        "request-target",
        "entry",
        "artifact-target",
        "artifact-path",
        "geometry",
        "outside",
        "module",
        "source",
        "identity",
        "compiler",
        "compiler-status",
    ],
)
def test_slice_native_identity_requires_retained_compilation(tmp_path, target, fault):
    package = tmp_path / "package"
    package.mkdir()
    source = package / "source.txt"
    source.write_bytes(b"translated source")
    artifact = {
        "packagePath": source.name,
        "sizeBytes": source.stat().st_size,
        "hash": {
            "algorithm": "sha256",
            "value": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    retained = tmp_path / "native/native-modules"
    retained.mkdir(parents=True)
    modules = []
    for suffix in {
        "metal": [".metallib", ".air"],
        "opengl": [".glsl"],
        "directx": [".dxil"],
    }[target]:
        path = retained / ("compiled" + suffix)
        path.write_bytes(b"compiled test fixture" + suffix.encode())
        modules.append(
            {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    execution = {"workgroupCount": [31, 1, 1], "workgroupSize": [1, 1, 1]}
    entry = "slice_update_sumfloat32"
    request = {
        "target": target,
        "entryPoint": {"metal": entry, "opengl": "main", "directx": "CSMain"}[target],
        "artifact": {"target": target, "packagePath": source.name},
        "dispatch": dict(execution),
    }
    identity = {key: artifact[key] for key in ("hash", "sizeBytes")}
    details = {
        "request": request,
        "nativeRuntimeDispatch": request,
        "module": modules[0],
        "validationModules": modules[1:],
        "artifactIdentityVerification": {
            "verificationStatus": "verified",
            "target": target,
            "expectedIdentity": identity,
            "observedIdentity": identity,
        },
        "adapterSteps": [
            {"action": action, "status": "passed"}
            for action in {
                "metal": [
                    "compile-metal-for-native-runtime",
                    "link-metal-for-native-runtime",
                ],
                "opengl": ["validate-glsl-for-opengl-runtime"],
                "directx": ["compile-hlsl-for-directx-runtime"],
            }[target]
        ],
    }
    event = {
        "entry": entry,
        "target": target,
        "artifact": artifact,
        "details": details,
        **execution,
    }
    if fault == "event-target":
        event["target"] = "cpu"
    elif fault == "request-target":
        request["target"] = "cpu"
    elif fault == "entry":
        request["entryPoint"] = "replacement"
    elif fault == "artifact-target":
        request["artifact"]["target"] = "cpu"
    elif fault == "artifact-path":
        request["artifact"]["packagePath"] = "other.txt"
    elif fault == "geometry":
        request["dispatch"]["workgroupCount"] = [30, 1, 1]
    elif fault == "outside":
        path = tmp_path / "outside"
        path.write_bytes(Path(modules[0]["file"]).read_bytes())
        modules[0]["file"] = str(path)
    elif fault == "module":
        Path(modules[0]["file"]).write_bytes(b"changed")
    elif fault == "source":
        source.write_bytes(b"changed")
    elif fault == "identity":
        details["artifactIdentityVerification"]["observedIdentity"] = {}
    elif fault == "compiler":
        details["adapterSteps"].pop()
    elif fault == "compiler-status":
        details["adapterSteps"][0]["status"] = "skipped"
    if fault:
        with pytest.raises(ValueError):
            verifier.validate_native_event(event, package, target, tmp_path)
    else:
        verifier.validate_native_event(event, package, target, tmp_path)
