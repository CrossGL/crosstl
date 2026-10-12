"""Checked metadata remains attached to the exact artifact and upload."""

import copy
import hashlib
import itertools
import json
import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    inspect_runtime_package,
    load_project_config,
    translate_project,
    validate_project_report,
)
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.metadata_preconditions import check_native_metadata_snapshots
from crosstl.project.native_loader_abi import (
    NativeLoaderABIError,
    generate_native_loader_execution_abi,
)
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    NativeRuntimeParityAdapter,
    RuntimeAllocationView,
    RuntimeExecutionState,
    RuntimeValue,
)
from crosstl.translator.frozen_specializations import (
    frozen_specialization_header,
    parse_frozen_specializations,
)
from crosstl.translator.metadata_preconditions import (
    METADATA_PRECONDITION_PREFIX,
    MetadataRangePrecondition,
    apply_metadata_preconditions,
    checked_metadata_preconditions,
    metadata_precondition_header,
    parse_metadata_precondition_header,
    parse_metadata_preconditions,
    with_metadata_precondition_header,
)
from crosstl.translator.resource_storage import (
    parse_resource_storage_header,
    resource_storage_header,
)

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void bounded_copy(constant uint& count [[buffer(0)]],
                         device uint* output [[buffer(1)]],
                         uint lane [[thread_position_in_grid]]) {
    if (lane < count) { output[lane] = count + lane; }
}
"""
COMBINED_SOURCE = (
    SOURCE.replace(
        "using namespace metal;",
        "using namespace metal;\nconstant bool enabled [[function_constant(0)]];",
    )
    .replace(
        "device uint* output [[buffer(1)]],",
        "device const half* values [[buffer(1)]],\n"
        "                         device half* output [[buffer(2)]],",
    )
    .replace("lane < count", "enabled && lane < count")
    .replace("count + lane", "values[lane]")
)
CONDITION = dict(entryPoint="bounded_copy", parameter="count", minimum=1, maximum=4)
RESOLVED = dict(
    sourceResource=dict(
        schemaVersion=1,
        backend="metal",
        entryPoint="bounded_copy",
        parameter="count",
        parameterIndex=0,
    ),
    dtype="uint32",
    minimum=1,
    maximum=4,
)
TARGETS = ("metal", "directx", "opengl")


def _package(
    tmp_path, target, *, source=SOURCE, conditions=None, enabled=None, formatted=False
):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "bounded.metal").write_text(source)
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("*.metal",),
            targets=(target,),
            workgroup_size=(4, 1, 1),
            metadata_preconditions=[CONDITION] if conditions is None else conditions,
            specialization_constants={} if enabled is None else {"enabled": enabled},
            freeze_specialization_constants=enabled is not None,
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=formatted,
    )
    report.write_json(tmp_path / "report.json")
    assert report.to_json()["summary"]["translatedCount"] == 1, report.to_json()[
        "diagnostics"
    ]
    manifest = build_runtime_artifact_manifest(tmp_path / "report.json")
    assert manifest["success"], manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    package = tmp_path / "package"
    result = build_runtime_package(tmp_path / "manifest.json", package)
    assert result["success"], result
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    return (
        build_native_loader_abi_descriptor(
            loader, load_unit_id=loader["loadUnits"][0]["id"]
        ),
        package,
    )


def _combined_values(descriptor, count=3, enabled=True):
    inputs, outputs = {}, {}
    values = [1.5, -2.0, 3.25, 4.5, 7.0, 8.0, 9.0, 10.0]
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        words = {"count": [count], "values": values, "output": [-11.0] * 8}[parameter]
        layout = binding["scalarLayout"]
        dtype = layout.get("storageEncoding", {}).get(
            "logicalElementType", layout["elementType"]
        )
        inputs[binding["name"]] = dict(dtype=dtype, shape=[len(words)], values=words)
        if parameter == "output":
            outputs[binding["name"]] = dict(
                dtype=dtype,
                shape=[8],
                values=[
                    value if enabled and index < count else -11.0
                    for index, value in enumerate(values)
                ],
            )
    return inputs, outputs


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("enabled", [False, True])
def test_combined_project_contracts_survive_format_package_and_dispatch(
    tmp_path, target, enabled
):
    descriptor, package = _package(
        tmp_path, target, source=COMBINED_SOURCE, enabled=enabled, formatted=True
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    source = artifact.read_text()
    assert parse_frozen_specializations(source) == [
        dict(id=0, name="enabled", dtype="bool", value=enabled, frozen=True)
    ]
    assert parse_metadata_precondition_header(source) == [RESOLVED]
    assert bool(parse_resource_storage_header(source)) == (target == "directx")
    assert reflect_target_host_interface(artifact, target=target)["status"] == "ready"
    assert inspect_runtime_package(package / "runtime-package.json")["success"]
    assert validate_project_report(tmp_path / "report.json")["success"]
    inputs, outputs = _combined_values(descriptor, enabled=enabled)
    request = _request(descriptor, package, values=(inputs, outputs))
    assert not request.adapter_contract.specialization_constants
    for value in (0, 5):
        with pytest.raises(NativeLoaderDispatchError, match="outside"):
            _request(
                descriptor, package, values=_combined_values(descriptor, value, enabled)
            )
    with pytest.raises(NativeLoaderDispatchError, match="override"):
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
            specialization_values={0: not enabled},
        )


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_dispatch_accepts_all_contract_orders_and_rechecks_full_artifact(
    tmp_path, order
):
    descriptor, package = _package(
        tmp_path, "directx", source=COMBINED_SOURCE, enabled=True
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    lines = artifact.read_text().splitlines(keepends=True)
    source = "".join(lines[index] for index in order) + "".join(lines[3:])
    artifact.write_text(source)
    descriptor["artifact"]["sizeBytes"] = artifact.stat().st_size
    descriptor["artifact"]["hash"]["value"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    request = _request(descriptor, package, values=_combined_values(descriptor))
    assert (
        request.adapter_contract.metadata["frozenSpecializations"][0]["value"] is True
    )
    artifact.write_text(source + lines[0])
    descriptor["artifact"]["sizeBytes"] = artifact.stat().st_size
    descriptor["artifact"]["hash"]["value"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    with pytest.raises(
        NativeLoaderDispatchError, match="frozen-specialization-invalid"
    ):
        _request(descriptor, package, values=_combined_values(descriptor))


def test_storage_failure_is_not_replaced_by_missing_resource_identity(tmp_path):
    descriptor, package = _package(
        tmp_path, "directx", source=COMBINED_SOURCE, enabled=True
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    source = artifact.read_text().replace('"ieee754-binary16"', '"unknown"')
    artifact.write_text(source)
    reflected = reflect_target_host_interface(artifact, target="directx")
    assert reflected["status"] == "failed"
    assert (
        reflected["diagnosticRecords"][0]["details"]["contract"] == "resource-storage"
    )
    assert (
        "Unsupported logical storage encoding" in reflected["diagnosticRecords"][0][
            "message"
        ]
    )


@pytest.mark.parametrize("target", ["metal", "opengl"])
def test_storage_contract_cannot_be_ignored_by_another_target(tmp_path, target):
    descriptor, package = _package(
        tmp_path, target, source=COMBINED_SOURCE, enabled=True
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    artifact.write_text(
        resource_storage_header(
            {
                "values": {
                    "logicalElementType": "float16",
                    "encoding": "ieee754-binary16",
                }
            }
        )
        + artifact.read_text()
    )
    reflected = reflect_target_host_interface(artifact, target=target)
    assert reflected["status"] == "failed"
    assert (
        reflected["diagnosticRecords"][0]["details"]["contract"] == "resource-storage"
    )


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("fault", ["remove", "duplicate", "body", "malformed"])
def test_package_rejects_mutations_of_any_combined_contract(tmp_path, index, fault):
    descriptor, package = _package(
        tmp_path, "directx", source=COMBINED_SOURCE, enabled=True
    )
    artifact = package / descriptor["artifact"]["packagePath"]
    lines = artifact.read_text().splitlines(keepends=True)
    if fault == "remove":
        del lines[index]
    elif fault == "duplicate":
        lines.insert(index, lines[index])
    elif fault == "body":
        lines.append(lines.pop(index))
    else:
        lines[index] = lines[index].split(": ", 1)[0] + ": {\n"
    artifact.write_text("".join(lines))
    path = package / "runtime-package.json"
    payload = json.loads(path.read_text())
    record = payload["artifacts"][0]
    record["sizeBytes"] = artifact.stat().st_size
    record["hash"]["value"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    assert not inspect_runtime_package(path)["success"]
    assert not build_runtime_loader_manifest(path)["success"]


@pytest.mark.parametrize("enabled", [False, True])
def test_native_combined_contracts(tmp_path, enabled):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(
        tmp_path, target, source=COMBINED_SOURCE, enabled=enabled
    )
    for count in (1, 3, 4):
        inputs, outputs = _combined_values(descriptor, count, enabled)
        request = _request(descriptor, package, values=(inputs, outputs))
        work = tmp_path / f"native-combined-{count}"
        work.mkdir()
        _execute(request, outputs, work, validate=_validate)
        evidence = json.loads((work / "evidence.json").read_text())
        assert (
            evidence["records"]["generated"]["details"]["metadataPreconditions"][0][
                "value"
            ]
            == count
        )


@pytest.mark.parametrize(
    "fault", ["remove", "encoding", "coordinates", "layout", "header"]
)
def test_direct_dispatch_rechecks_storage_contract(tmp_path, fault):
    descriptor, package = _package(
        tmp_path, "directx", source=COMBINED_SOURCE, enabled=True
    )
    binding = next(
        item
        for item in descriptor["bindings"]
        if "storageEncoding" in item["scalarLayout"]
    )
    if fault == "remove":
        binding["scalarLayout"].pop("storageEncoding")
    elif fault == "encoding":
        binding["scalarLayout"]["storageEncoding"]["encoding"] = "unknown"
    elif fault == "coordinates":
        binding["coordinates"]["binding"] = 15
    elif fault == "layout":
        binding["scalarLayout"]["elementStrideBytes"] = 4
    else:
        artifact = package / descriptor["artifact"]["packagePath"]
        lines = artifact.read_text().splitlines(keepends=True)
        artifact.write_text("".join([lines[0], *lines[2:]]))
        descriptor["artifact"]["sizeBytes"] = artifact.stat().st_size
        descriptor["artifact"]["hash"]["value"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    descriptor["scalarLayout"]["bindings"] = [
        {"binding": item["name"], "layout": copy.deepcopy(item["scalarLayout"])}
        for item in descriptor["bindings"]
    ]
    with pytest.raises(NativeLoaderDispatchError, match="resource-storage-invalid"):
        _request(descriptor, package, values=_combined_values(descriptor))


def _values(descriptor, count=4):
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        words = [count] if parameter == "count" else [123456] * 8
        inputs[binding["name"]] = dict(dtype="uint32", shape=[len(words)], values=words)
        if parameter == "output":
            outputs[binding["name"]] = dict(
                dtype="uint32",
                shape=[8],
                values=[
                    count + i if type(count) is int and i < count else 123456
                    for i in range(8)
                ],
            )
    return inputs, outputs


def _request(descriptor, package, count=4, *, values=None):
    inputs, outputs = values or _values(descriptor, count)
    return build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {
            "workgroupCount": [1, 1, 1],
            "workgroupSize": [4, 1, 1],
        },
    )


@pytest.mark.parametrize("target", TARGETS)
def test_project_package_dispatch_roundtrip(tmp_path, target):
    descriptor, package = _package(tmp_path, target)
    source = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert parse_metadata_precondition_header(source) == [RESOLVED]
    assert validate_project_report(tmp_path / "report.json")["success"]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["project"]["metadataPreconditions"] == [{"source": "*", **CONDITION}]
    for count in (1, 3, 4):
        request = _request(descriptor, package, count)
        state = RuntimeExecutionState(request=request, plan=request.execution_plan)
        adapter = NativeRuntimeParityAdapter()
        adapter.target = target
        buffers = adapter._native_buffer_bindings(state)
        receipts = check_native_metadata_snapshots(
            package / descriptor["artifact"]["packagePath"], target, buffers
        )
        assert len(receipts) == 1
        assert receipts[0]["value"] == count
        assert receipts[0]["requirement"] == RESOLVED
    with pytest.raises(
        NativeLoaderABIError, match="execution-metadata-precondition-unsupported"
    ):
        generate_native_loader_execution_abi(descriptor)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("count", [0, 5, -1, 2**32, True, "4", None])
def test_out_of_range_input_is_rejected(tmp_path, target, count):
    descriptor, package = _package(tmp_path, target)
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, count)


@pytest.mark.parametrize(
    "fault", ["remove", "maximum", "origin", "coordinates", "dtype", "access"]
)
def test_descriptor_cannot_change_obligations(tmp_path, fault):
    descriptor, package = _package(tmp_path, "opengl")
    binding = next(
        item
        for item in descriptor["bindings"]
        if "metadataPrecondition" in item["provenance"]
    )
    if fault == "remove":
        binding["provenance"].pop("metadataPrecondition")
    elif fault == "maximum":
        binding["provenance"]["metadataPrecondition"]["maximum"] = 100
    elif fault == "origin":
        binding["provenance"]["sourceResource"]["parameterIndex"] = 5
    elif fault == "coordinates":
        binding["coordinates"]["binding"] = 15
    elif fault == "dtype":
        binding["scalarLayout"]["elementType"] = "int32"
    else:
        binding["access"] = "read_write"
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package)


@pytest.mark.parametrize("fault", ["declared", "removed-header", "changed-header"])
def test_package_rechecks_metadata_contract(tmp_path, fault):
    descriptor, package = _package(tmp_path, "metal")
    path = package / "runtime-package.json"
    payload = json.loads(path.read_text())
    artifact = payload["artifacts"][0]
    if fault == "declared":
        resource = next(
            item
            for item in artifact["hostInterface"]["resources"]
            if "metadataPrecondition" in item["metadata"]["provenance"]
        )
        resource["metadata"]["provenance"]["metadataPrecondition"]["maximum"] = 10
    else:
        shader = package / descriptor["artifact"]["packagePath"]
        text = shader.read_text()
        text = (
            text.split("\n", 1)[1]
            if fault == "removed-header"
            else text.replace('"maximum":4', '"maximum":10', 1)
        )
        shader.write_text(text)
        artifact["sizeBytes"] = shader.stat().st_size
        artifact["hash"]["value"] = hashlib.sha256(shader.read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    assert not inspect_runtime_package(path)["success"]
    assert not build_runtime_loader_manifest(path)["success"]


def test_shared_allocation_is_rejected(tmp_path):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values(descriptor)
    inputs = {
        name: RuntimeValue(
            name=name, **value, allocation=RuntimeAllocationView("shared", 0, None, 32)
        )
        for name, value in inputs.items()
    }
    outputs = {
        name: RuntimeValue(
            name=name, **value, allocation=RuntimeAllocationView("shared", 0, None, 32)
        )
        for name, value in outputs.items()
    }
    with pytest.raises(NativeLoaderDispatchError, match="share"):
        _request(descriptor, package, values=(inputs, outputs))


def test_native_snapshot_rechecked_after_request_mutation(tmp_path):
    descriptor, package = _package(tmp_path, "metal")
    request = _request(descriptor, package)
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    adapter = NativeRuntimeParityAdapter()
    adapter.target = "metal"
    buffers = adapter._native_buffer_bindings(state)
    key = next(
        name
        for name, item in buffers.items()
        if "metadataPrecondition" in item.binding.metadata["provenance"]
    )
    binding = buffers[key]
    buffers[key] = replace(
        binding,
        upload_snapshot=replace(
            binding.upload_snapshot, payload=(5).to_bytes(4, "little")
        ),
    )
    with pytest.raises(ValueError, match="outside"):
        check_native_metadata_snapshots(
            package / descriptor["artifact"]["packagePath"], "metal", buffers
        )


@pytest.mark.parametrize(
    "replacement",
    ["constant float& count", "device uint& count", "constant uint* count"],
)
def test_only_read_only_integer_scalars_are_accepted(tmp_path, replacement):
    (tmp_path / "bounded.metal").write_text(
        SOURCE.replace("constant uint& count", replacement)
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=("metal",),
            metadata_preconditions=[CONDITION],
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=False,
    )
    assert report.to_json()["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metadata-precondition-unsupported"
        for item in report.to_json()["diagnostics"]
    )


@pytest.mark.parametrize(
    "condition",
    [
        [],
        {},
        {**CONDITION, "other": 1},
        {**CONDITION, "minimum": True},
        {**CONDITION, "maximum": 0},
        {**CONDITION, "parameter": "a.b"},
    ],
)
def test_invalid_configuration_is_rejected(condition):
    with pytest.raises(ValueError):
        parse_metadata_preconditions([condition])


def test_configuration_roundtrip(tmp_path):
    path = tmp_path / "crosstl.toml"
    path.write_text("""[project]
targets = ["metal"]
[[project.metadata_preconditions]]
entryPoint = "bounded_copy"
parameter = "count"
minimum = 1
maximum = 4
""")
    assert load_project_config(tmp_path).metadata_preconditions == (
        MetadataRangePrecondition("bounded_copy", "count", 1, 4),
    )


@pytest.mark.parametrize(
    "dtype,minimum,maximum",
    [
        ("int8", -128, 127),
        ("uint8", 0, 255),
        ("int16", -32768, 32767),
        ("uint16", 0, 65535),
        ("int32", -(2**31), 2**31 - 1),
        ("uint32", 0, 2**32 - 1),
        ("int64", -(2**63), 2**63 - 1),
        ("uint64", 0, 2**64 - 1),
    ],
)
def test_resolved_bounds_preserve_source_width(dtype, minimum, maximum):
    record = {**RESOLVED, "dtype": dtype, "minimum": minimum, "maximum": maximum}
    assert checked_metadata_preconditions([record]) == [record]
    for field, value in (("minimum", minimum - 1), ("maximum", maximum + 1)):
        with pytest.raises(ValueError):
            checked_metadata_preconditions([{**record, field: value}])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "metal_type,dtype", [("int", "int32"), ("long", "int64"), ("ulong", "uint64")]
)
def test_metadata_source_types_survive_lowering(tmp_path, target, metal_type, dtype):
    source = SOURCE.replace("constant uint& count", f"constant {metal_type}& count")
    descriptor, package = _package(tmp_path, target, source=source)
    binding = next(
        item
        for item in descriptor["bindings"]
        if "metadataPrecondition" in item["provenance"]
    )
    assert binding["provenance"]["metadataPrecondition"]["dtype"] == dtype
    inputs, outputs = _values(descriptor)
    inputs[binding["name"]]["dtype"] = dtype
    _request(descriptor, package, values=(inputs, outputs))


@pytest.mark.parametrize(
    "fault",
    [
        "vector",
        "offset",
        "runtime-array",
        "writable",
        "record",
        "renamed",
        "struct-record",
    ],
)
def test_reflection_rejects_incompatible_metadata_layout(fault):
    resource = dict(
        name="metadata",
        kind="constant-buffer",
        access="read",
        metadata={"provenance": {"sourceResource": RESOLVED["sourceResource"]}},
        scalarLayout={
            "elementType": "uint32",
            "elementSizeBytes": 4,
            "elementStrideBytes": 4,
            "memberOffsetBytes": 0,
            "runtimeSized": False,
        },
    )
    if fault == "vector":
        resource["scalarLayout"]["vectorWidth"] = 2
    elif fault == "offset":
        resource["scalarLayout"]["memberOffsetBytes"] = 4
    elif fault == "runtime-array":
        resource["scalarLayout"]["runtimeSized"] = True
    elif fault == "writable":
        resource["access"] = "read_write"
    elif fault == "record":
        resource["scalarLayout"]["members"] = []
    elif fault == "struct-record":
        resource["scalarLayout"]["structMembers"] = []
    else:
        resource["metadata"]["provenance"]["sourceResource"] = {
            **RESOLVED["sourceResource"],
            "parameter": "other",
        }
    with pytest.raises(ValueError):
        apply_metadata_preconditions([resource], [RESOLVED])
    assert "metadataPrecondition" not in resource["metadata"]["provenance"]


def test_implicit_allocation_alias_is_rejected(tmp_path):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values(descriptor)
    name = next(
        item["name"]
        for item in descriptor["bindings"]
        if "metadataPrecondition" in item["provenance"]
    )
    output_name = next(iter(outputs))
    output_binding = next(
        item for item in descriptor["bindings"] if item["name"] == output_name
    )
    coords = output_binding["coordinates"]
    inputs[name] = RuntimeValue(
        name=name,
        **inputs[name],
        allocation=RuntimeAllocationView(
            f"binding:{coords['set']}:{coords['binding']}:-:{output_name}", 0, 4, 32
        ),
    )
    with pytest.raises(NativeLoaderDispatchError, match="share"):
        _request(descriptor, package, values=(inputs, outputs))


def test_input_snapshot_remains_owned_and_is_checked_again(tmp_path):
    descriptor, package = _package(tmp_path, "metal")
    inputs, outputs = _values(descriptor)
    name = next(
        item["name"]
        for item in descriptor["bindings"]
        if "metadataPrecondition" in item["provenance"]
    )
    request = _request(descriptor, package, values=(inputs, outputs))
    inputs[name]["values"][0] = 999
    adapter = NativeRuntimeParityAdapter()
    adapter.target = "metal"
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    buffers = adapter._native_buffer_bindings(state)
    path = package / descriptor["artifact"]["packagePath"]
    assert check_native_metadata_snapshots(path, "metal", buffers)[0]["value"] == 4
    for resource in request.execution_plan.resource_bindings:
        if resource.binding.name == name:
            resource.value.values[0] = 999
    with pytest.raises(ValueError, match="outside"):
        check_native_metadata_snapshots(
            path, "metal", adapter._native_buffer_bindings(state)
        )


def test_native_ci_requires_metadata_execution():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    assert 'CROSTL_REQUIRE_METADATA_PRECONDITIONS: "1"' in workflow
    assert "tests/test_translator/test_metadata_preconditions.py" in workflow


@pytest.mark.parametrize(
    "target", ["webgl", "wgsl", "vulkan", "cuda", "hip", "mojo", "rust", "slang"]
)
def test_unsupported_runtime_targets_reject_preconditions(tmp_path, target):
    (tmp_path / "bounded.metal").write_text(SOURCE)
    report = translate_project(
        ProjectConfig(
            root=tmp_path, targets=(target,), metadata_preconditions=[CONDITION]
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert any(
        item["code"] == "project.translate.metadata-precondition-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("target", TARGETS)
def test_invalid_late_value_stops_native_preparation(tmp_path, target):
    from crosstl.project.runtime_verification import RuntimeAdapterSetupError

    descriptor, package = _package(tmp_path, target)
    request = _request(descriptor, package)
    for resource in request.execution_plan.resource_bindings:
        if "metadataPrecondition" in resource.binding.metadata.get("provenance", {}):
            resource.value.values[0] = 17
    state = RuntimeExecutionState(request=request, plan=request.execution_plan)
    adapter = NativeRuntimeParityAdapter(validate=False)
    adapter.target = target
    try:
        with pytest.raises(RuntimeAdapterSetupError, match="outside") as failure:
            adapter.prepare_buffers(state)
        assert failure.value.details["reasonKind"] == "metadata-precondition-invalid"
        assert "nativeRuntimeDispatch" not in state.details
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()


@pytest.mark.parametrize("prefix", ["", "frozen", "storage"])
def test_header_coexists_with_other_leading_contracts(prefix):
    first = ""
    if prefix == "frozen":
        first = frozen_specialization_header(
            [dict(id=0, name="enabled", dtype="bool", value=True, frozen=True)]
        )
    elif prefix == "storage":
        first = resource_storage_header(
            {"data": dict(logicalElementType="float16", encoding="ieee754-binary16")}
        )
    source = with_metadata_precondition_header(first + "void main() {}\n", [RESOLVED])
    assert parse_metadata_precondition_header(source) == [RESOLVED]
    if prefix == "frozen":
        assert parse_frozen_specializations(source) == parse_frozen_specializations(
            first
        )
    elif prefix == "storage":
        assert parse_resource_storage_header(source) == parse_resource_storage_header(
            first
        )


@pytest.mark.parametrize(
    "mutation", ["misplaced", "duplicate", "empty", "boolean-version", "duplicate-key"]
)
def test_invalid_headers_rejected(mutation):
    header = metadata_precondition_header([RESOLVED])
    if mutation == "misplaced":
        header = "void main() {}\n" + header
    elif mutation == "duplicate":
        header *= 2
    elif mutation == "empty":
        header = METADATA_PRECONDITION_PREFIX + '{"schemaVersion":1,"requirements":[]}'
    elif mutation == "boolean-version":
        header = header.replace('"schemaVersion":1', '"schemaVersion":true')
    else:
        header = header.replace('"maximum":4', '"maximum":4,"maximum":5')
    with pytest.raises(ValueError):
        parse_metadata_precondition_header(header)


def test_native_checked_metadata(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    descriptor, package = _package(tmp_path, target)
    for count in (1, 3, 4):
        inputs, outputs = _values(descriptor, count)
        name = next(
            item["name"]
            for item in descriptor["bindings"]
            if "metadataPrecondition" in item["provenance"]
        )
        inputs[name] = RuntimeValue(
            name=name,
            **inputs[name],
            allocation=RuntimeAllocationView("metadata", 256, None, 512),
        )
        request = _request(descriptor, package, values=(inputs, outputs))
        _, outputs = _values(descriptor, count)
        work = tmp_path / f"native-{count}"
        work.mkdir()
        _execute(
            request,
            outputs,
            work,
            validate=_validate,
            original_source=SOURCE if target == "metal" else None,
            original_entry="bounded_copy",
        )
        evidence = json.loads((work / "evidence.json").read_text())
        assert (
            evidence["records"]["generated"]["details"]["metadataPreconditions"][0][
                "value"
            ]
            == count
        )
        assert (
            evidence["records"]["generated"]["details"]["metadataPreconditions"][0][
                "allocation"
            ]["byteOffset"]
            == 256
        )


def test_native_entry_scoped_metadata(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_project_metadata_entry_scope import SOURCE as source
    from tests.test_translator.test_project_metadata_entry_scope import (
        _config,
        _entry,
        _packages,
    )
    from tests.test_translator.test_project_metadata_entry_scope import (
        _request as entry_request,
    )

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _payload, package, descriptors = _packages(_config(tmp_path, target))
    for descriptor in descriptors:
        entry = _entry(descriptor)
        counts = (1, 3, 4) if entry == "first" else (1, 5, 8)
        for count in counts:
            request, outputs = entry_request(descriptor, package, count)
            work = tmp_path / f"native-entry-{entry}-{count}"
            work.mkdir()
            _execute(
                request,
                outputs,
                work,
                validate=_validate,
                original_source=source if target == "metal" else None,
                original_entry=entry,
            )
            evidence = json.loads((work / "evidence.json").read_text())
            checks = evidence["records"]["generated"]["details"][
                "metadataPreconditions"
            ]
            assert len(checks) == 1
            assert checks[0]["value"] == count
            assert checks[0]["requirement"]["sourceResource"]["entryPoint"] == entry


def test_native_metadata_bounded_collective_loop(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_metadata_loop_bounds import SOURCE as source
    from tests.test_translator.test_metadata_loop_bounds import _loop_config, _packages
    from tests.test_translator.test_metadata_loop_bounds import _request as loop_request

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, package, (descriptor,) = _packages(_loop_config(tmp_path, target))
    for count in (0, 1, 31, 32, 33, 64, 65, 65535):
        request, outputs = loop_request(descriptor, package, count)
        work = tmp_path / f"native-loop-{count}"
        work.mkdir()
        _execute(
            request,
            outputs,
            work,
            validate=_validate,
            original_source=source if target == "metal" else None,
            original_entry="scan_loop",
        )
        evidence = json.loads((work / "evidence.json").read_text())
        for record in evidence["records"].values():
            observations = record["details"]["inputObservations"]
            assert len(observations) == 1
            assert all(item["unchanged"] for item in observations.values())
        checks = evidence["records"]["generated"]["details"]["metadataPreconditions"]
        assert len(checks) == 1 and checks[0]["value"] == count


def test_native_metadata_bounded_indices(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_metadata_index_bounds import (
        SOURCE,
        _index_config,
        _index_request,
        _packages,
    )

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    for mode, body in (
        ("composed", "result[2] = values[offset + index];"),
        ("rebased", "values += offset; result[2] = values[index];"),
    ):
        root = tmp_path / mode
        root.mkdir()
        source = SOURCE.replace("result[0] = values[offset + index];", body)
        _, package, (descriptor,) = _packages(
            _index_config(root, source=source, target=target)
        )
        for index, offset in ((0, 0), (3, 0), (0, 4), (3, 4)):
            request, outputs = _index_request(descriptor, package, index, offset)
            work = root / f"native-index-{index}-{offset}"
            work.mkdir()
            _execute(
                request,
                outputs,
                work,
                validate=_validate,
                original_source=source if target == "metal" else None,
                original_entry="read_value",
            )
            evidence = json.loads((work / "evidence.json").read_text())
            for record in evidence["records"].values():
                observations = record["details"]["inputObservations"]
                assert len(observations) == 3
                assert all(item["unchanged"] for item in observations.values())
            checks = evidence["records"]["generated"]["details"][
                "metadataPreconditions"
            ]
            assert {
                item["requirement"]["sourceResource"]["parameter"]: item["value"]
                for item in checks
            } == {"index": index, "offset": offset}


def test_native_metadata_bounded_division(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_integer_interval_division import (
        DIVISION_CASES,
        DIVISION_SOURCE,
        _division_config,
        _division_request,
    )
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_project_metadata_entry_scope import _packages

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, package, (descriptor,) = _packages(_division_config(tmp_path, target))
    for numerator, divisor in DIVISION_CASES:
        request, outputs = _division_request(descriptor, package, numerator, divisor)
        work = tmp_path / f"native-division-{numerator}-{divisor}"
        work.mkdir()
        _execute(
            request,
            outputs,
            work,
            validate=_validate,
            original_source=DIVISION_SOURCE if target == "metal" else None,
            original_entry="read_value",
        )
        evidence = json.loads((work / "evidence.json").read_text())
        for record in evidence["records"].values():
            observations = record["details"]["inputObservations"]
            assert len(observations) == 3
            assert all(item["unchanged"] for item in observations.values())
        checks = evidence["records"]["generated"]["details"]["metadataPreconditions"]
        assert {
            item["requirement"]["sourceResource"]["parameter"]: item["value"]
            for item in checks
        } == {"index": numerator, "offset": divisor}


def test_native_region_metadata_indices(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_dispatch_region_contract import (
        REGION_INDEX_SOURCE,
        _region_index_config,
        _region_index_request,
    )
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_project_metadata_entry_scope import _packages

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    _, package, (descriptor,) = _packages(_region_index_config(tmp_path, target))
    for index, offset in ((1, 1), (1, 2), (16, 1), (16, 2)):
        request, outputs = _region_index_request(descriptor, package, index, offset)
        work = tmp_path / f"native-region-index-{index}-{offset}"
        work.mkdir()
        _execute(
            request,
            outputs,
            work,
            validate=_validate,
            original_source=REGION_INDEX_SOURCE if target == "metal" else None,
            original_entry="read_value",
        )
        evidence = json.loads((work / "evidence.json").read_text())
        for record in evidence["records"].values():
            observations = record["details"]["inputObservations"]
            assert len(observations) == 3
            assert all(item["unchanged"] for item in observations.values())
        if target != "metal":
            assert (
                evidence["records"]["generated"]["details"]["dispatchRegion"]
                == descriptor["provenance"]["dispatchRegion"]
            )


def test_native_region_metadata_coordinate_widths(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native metadata checks"
        )
    from crosstl.translator.dispatch_regions import plan_dispatch_regions
    from tests.runtime_helpers import _validate
    from tests.test_translator.test_dispatch_coordinate_bounds import (
        _coordinate_config,
        _coordinate_request,
        _coordinate_source,
    )
    from tests.test_translator.test_loop_updates import _execute
    from tests.test_translator.test_project_metadata_entry_scope import _packages

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    cases = [
        (kind, plan_dispatch_regions(grid, (1, 1, 1))[0])
        for kind, grid in (
            ("uint", (3, 1, 1)),
            ("uint2", (3, 2, 1)),
            ("uint3", (3, 2, 2)),
        )
    ]
    if target != "metal":
        cases.extend(
            ("uint2", region) for region in plan_dispatch_regions((5, 3, 1), (4, 2, 1))
        )
    for number, (kind, region) in enumerate(cases):
        grid = region.thread_grid_size
        root = tmp_path / f"{kind}-{number}"
        _, package, (descriptor,) = _packages(
            _coordinate_config(root, target, kind, region)
        )
        request, outputs = _coordinate_request(descriptor, package, region)
        work = root / "native"
        work.mkdir()
        _execute(
            request,
            outputs,
            work,
            validate=_validate,
            original_source=(
                _coordinate_source(kind, grid) if target == "metal" else None
            ),
            original_entry="read_value",
        )
        evidence = json.loads((work / "evidence.json").read_text())
        for record in evidence["records"].values():
            observations = record["details"]["inputObservations"]
            assert len(observations) == 1
            assert all(item["unchanged"] for item in observations.values())
        generated = evidence["records"]["generated"]
        checks = generated["details"]["metadataPreconditions"]
        assert len(checks) == 1 and checks[0]["value"] == 16
        if target != "metal":
            assert generated["details"]["dispatchRegion"] == region.to_json()
