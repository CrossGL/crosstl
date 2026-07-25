import copy
import hashlib
import json

import pytest

import crosstl.project.runtime_variant_dispatch as dispatch_module
from crosstl.project import (
    NATIVE_LOADER_ABI_KIND,
    NATIVE_LOADER_ABI_PACKAGE_KIND,
    NATIVE_LOADER_ABI_PACKAGE_VERSION,
    NATIVE_LOADER_ABI_VERSION,
    RuntimeVariantDispatchError,
    build_runtime_variant_dispatch_request,
    encode_runtime_variant_key,
)
from crosstl.project import pipeline as project_pipeline


def _sha256(content):
    return {"algorithm": "sha256", "value": hashlib.sha256(content).hexdigest()}


def _json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _layout(target):
    return {
        "physicalType": "float",
        "elementType": "float32",
        "elementSizeBytes": 4,
        "elementStrideBytes": 4,
        "alignmentBytes": 4,
        "memberOffsetBytes": 0,
        "storageLayout": "hlsl-structured-buffer" if target == "directx" else "std430",
        "runtimeSized": True,
    }


def _artifact_source(target, entry_point):
    if target == "directx":
        return (
            "StructuredBuffer<float> input_values : register(t0);\n"
            "RWStructuredBuffer<float> output_values : register(u1);\n"
            "[numthreads(4, 1, 1)]\n"
            f"void {entry_point}(uint3 tid : SV_DispatchThreadID) {{\n"
            "  output_values[tid.x] = input_values[tid.x];\n"
            "}\n"
        ).encode()
    return (
        b"#version 450\n"
        b"layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;\n"
        b"layout(std430, binding = 0) readonly buffer InputValues {\n"
        b"  float input_values[];\n"
        b"};\n"
        b"layout(std430, binding = 1) writeonly buffer OutputValues {\n"
        b"  float output_values[];\n"
        b"};\n"
        b"void main() {\n"
        b"  output_values[gl_GlobalInvocationID.x] = "
        b"input_values[gl_GlobalInvocationID.x];\n"
        b"}\n"
    )


def _descriptor(target, artifact_path, artifact_bytes, *, unit_id=None):
    entry_point = "CopyMain" if target == "directx" else "main"
    unit_id = unit_id or f"{target}|copy"
    layout = _layout(target)
    namespaces = (
        ("srv", "uav") if target == "directx" else ("storage-buffer", "storage-buffer")
    )
    bindings = [
        {
            "name": "input_values",
            "kind": "buffer",
            "type": "float[]",
            "namespace": namespaces[0],
            "coordinates": {"set": 0, "binding": 0},
            "access": "read",
            "scalarLayout": copy.deepcopy(layout),
            "provenance": {"parameter": 0},
        },
        {
            "name": "output_values",
            "kind": "buffer",
            "type": "float[]",
            "namespace": namespaces[1],
            "coordinates": {"set": 0, "binding": 1},
            "access": "write",
            "scalarLayout": copy.deepcopy(layout),
            "provenance": {"parameter": 1},
        },
    ]
    specialization = {
        "id": 7,
        "name": "mode",
        "dtype": "uint32",
        "required": True,
        "provenance": {"source": "variant-registry"},
    }
    if target == "directx":
        specialization["value"] = 11
    else:
        specialization["default"] = 3
    return {
        "schemaVersion": NATIVE_LOADER_ABI_VERSION,
        "kind": NATIVE_LOADER_ABI_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "unitId": unit_id,
        "target": target,
        "stage": "compute",
        "entryPoint": {
            "name": entry_point,
            "stage": "compute",
            "executionConfig": {"workgroupSize": [4, 1, 1]},
            "provenance": {"sourceName": "copy"},
        },
        "artifact": {
            "packagePath": artifact_path,
            "format": "HLSL source" if target == "directx" else "GLSL source",
            "hash": _sha256(artifact_bytes),
            "sizeBytes": len(artifact_bytes),
        },
        "source": {
            "path": "kernels/copy.metal",
            "artifactPath": f"out/{target}/copy",
            "backend": "metal",
            "hash": None,
            "remap": None,
        },
        "bindings": bindings,
        "scalarLayout": {
            "constants": [],
            "bindings": [
                {"binding": binding["name"], "layout": copy.deepcopy(layout)}
                for binding in bindings
            ],
        },
        "specializationConstants": [specialization],
        "provenance": {"pipeline": "metal-to-crossgl"},
    }


def _record(descriptor):
    target = descriptor["target"]
    entry_point = descriptor["entryPoint"]["name"]
    execution = {"workgroupSize": [4, 1, 1], "subgroupWidth": None}
    specialization = {
        "id": 7,
        "name": "mode",
        "kind": "specialization-constant",
        "dtype": "uint32",
        "value": 11,
        "valueSource": "concreteValue",
        "required": True,
        "status": "materialized",
        "source": "project.variant",
        "runtimeRole": "pipeline-specialization",
    }
    key = encode_runtime_variant_key(
        "kernels/copy.metal",
        "copy",
        target,
        target_profile="cs_6_0" if target == "directx" else None,
        execution=execution,
        type_arguments={"T": "float"},
        value_arguments={"N": 4},
        specialization_constants=[specialization],
        defines={"COPY_MODE": "1"},
    )
    return {
        "key": key,
        "status": "ready",
        "source": {
            "unit": "kernels/copy.metal",
            "backend": "metal",
            "entry": "copy",
        },
        "target": {
            "backend": target,
            "profile": "cs_6_0" if target == "directx" else None,
            "stage": "compute",
            "entryPoint": entry_point,
        },
        "variant": "float-n4",
        "arguments": {"types": {"T": "float"}, "values": {"N": 4}},
        "defines": {"COPY_MODE": "1"},
        "specializationConstants": [specialization],
        "execution": execution,
        "bindingInterface": {
            "status": "ready",
            "source": "compiled-artifact",
            "parser": f"{target}-reflection",
            "artifactFormat": descriptor["artifact"]["format"],
            "entryPoint": {
                "name": entry_point,
                "stage": "compute",
                "executionConfig": {"workgroupSize": [4, 1, 1]},
            },
            "entryPointCount": 1,
            "resources": [],
            "resourceCount": 0,
            "constants": [],
            "constantCount": 0,
            "specializationConstants": [copy.deepcopy(specialization)],
            "specializationConstantCount": 1,
            "diagnostics": [],
            "diagnosticRecords": [],
        },
        "artifact": {
            "id": descriptor["unitId"],
            "path": descriptor["artifact"]["packagePath"],
            "format": descriptor["artifact"]["format"],
            "hash": copy.deepcopy(descriptor["artifact"]["hash"]),
            "sizeBytes": descriptor["artifact"]["sizeBytes"],
        },
        "provenance": {
            "inputKind": "crosstl-runtime-package",
            "artifactId": descriptor["unitId"],
            "sourceHash": None,
            "sourceSizeBytes": None,
            "translation": {},
            "entryPointNames": "explicit-source-target-entry-point",
            "templateArguments": {},
            "sourceRemap": None,
        },
        "lookup": {"mode": "exact", "eligible": True},
        "blockers": [],
    }


def _registry(records, *, reverse=False):
    ordered = list(records)
    if reverse:
        ordered.reverse()
    variants = {record["key"]: copy.deepcopy(record) for record in ordered}
    key_schema = project_pipeline._runtime_variant_key_schema()
    registry_hash = project_pipeline._runtime_variant_payload_hash(
        {
            "schemaVersion": project_pipeline.REPORT_SCHEMA_VERSION,
            "keySchema": key_schema,
            "variants": variants,
        }
    )
    ready_keys = sorted(variants)
    targets = sorted({record["target"]["backend"] for record in variants.values()})
    return {
        "schemaVersion": project_pipeline.REPORT_SCHEMA_VERSION,
        "kind": project_pipeline.RUNTIME_VARIANT_REGISTRY_KIND,
        "success": True,
        "status": "ready",
        "scope": project_pipeline.RUNTIME_VARIANT_REGISTRY_SCOPE,
        "nonGoals": list(project_pipeline.RUNTIME_VARIANT_REGISTRY_NON_GOALS),
        "source": {
            "kind": project_pipeline.RUNTIME_PACKAGE_KIND,
            "schemaVersion": project_pipeline.REPORT_SCHEMA_VERSION,
        },
        "keySchema": key_schema,
        "summary": {
            "targetCount": len(targets),
            "candidateCount": len(variants),
            "variantCount": len(variants),
            "readyVariantCount": len(variants),
            "blockedVariantCount": 0,
            "staleVariantCount": 0,
            "duplicateKeyCount": 0,
            "conflictingKeyCount": 0,
            "rejectedCandidateCount": 0,
        },
        "targets": [
            {
                "target": target,
                "variantCount": sum(
                    record["target"]["backend"] == target
                    for record in variants.values()
                ),
                "readyVariantCount": sum(
                    record["target"]["backend"] == target
                    for record in variants.values()
                ),
                "blockedVariantCount": 0,
                "staleVariantCount": 0,
                "keys": sorted(
                    record["key"]
                    for record in variants.values()
                    if record["target"]["backend"] == target
                ),
            }
            for target in targets
        ],
        "variants": variants,
        "lookup": {
            "mode": "exact",
            "defaulting": "none",
            "availableKeys": ready_keys,
            "readyKeys": ready_keys,
            "blockedKeys": [],
        },
        "inputInspection": {
            "kind": project_pipeline.RUNTIME_PACKAGE_INSPECTION_KIND,
            "success": True,
        },
        "registryHash": registry_hash,
        "diagnosticCounts": {"error": 0, "warning": 0, "info": 0},
        "diagnostics": [],
    }


def _package_unit(descriptor, descriptor_path, descriptor_bytes):
    return {
        "unitId": descriptor["unitId"],
        "target": descriptor["target"],
        "stage": descriptor["stage"],
        "entryPoint": descriptor["entryPoint"]["name"],
        "artifact": copy.deepcopy(descriptor["artifact"]),
        "descriptorPath": descriptor_path,
        "descriptorHash": _sha256(descriptor_bytes),
        "descriptorSizeBytes": len(descriptor_bytes),
        "declarationsPath": f"targets/{descriptor['target']}/abi.h",
        "declarationsHash": _sha256(b"declarations"),
        "executionABIPath": f"targets/{descriptor['target']}/execution.h",
        "executionABIHash": _sha256(b"execution"),
    }


def _write_fixture(tmp_path, *, reverse_registry=False, reverse_package=False):
    package_root = tmp_path / "package"
    records = []
    units = []
    descriptors = {}
    for target in ("directx", "opengl"):
        entry_point = "CopyMain" if target == "directx" else "main"
        extension = "hlsl" if target == "directx" else "glsl"
        artifact_path = f"artifacts/{target}/copy.{extension}"
        artifact_bytes = _artifact_source(target, entry_point)
        output_artifact = package_root / artifact_path
        output_artifact.parent.mkdir(parents=True, exist_ok=True)
        output_artifact.write_bytes(artifact_bytes)
        descriptor = _descriptor(target, artifact_path, artifact_bytes)
        descriptor_path = f"targets/{target}/copy.native-loader-abi.json"
        descriptor_bytes = _json_bytes(descriptor)
        output_descriptor = package_root / descriptor_path
        output_descriptor.parent.mkdir(parents=True, exist_ok=True)
        output_descriptor.write_bytes(descriptor_bytes)
        records.append(_record(descriptor))
        units.append(_package_unit(descriptor, descriptor_path, descriptor_bytes))
        descriptors[target] = descriptor
    if reverse_package:
        units.reverse()
    registry = _registry(records, reverse=reverse_registry)
    registry_path = "runtime/runtime-variant-registry.json"
    registry_bytes = _json_bytes(registry)
    output_registry = package_root / registry_path
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    output_registry.write_bytes(registry_bytes)
    native_header_path = "native-runtime-variant-registry.hpp"
    native_header_bytes = b"#pragma once\n"
    (package_root / native_header_path).write_bytes(native_header_bytes)
    package = {
        "schemaVersion": NATIVE_LOADER_ABI_PACKAGE_VERSION,
        "kind": NATIVE_LOADER_ABI_PACKAGE_KIND,
        "abiVersion": NATIVE_LOADER_ABI_VERSION,
        "sourceLoaderManifest": "runtime-loader-manifest.json",
        "sourceLoaderManifestHash": _sha256(b"loader-manifest"),
        "success": True,
        "summary": {
            "unitCount": 2,
            "targetCount": 2,
            "targetAdapterCount": 2,
            "unavailableTargetAdapterCount": 0,
            "runtimeVariantCount": len(registry["variants"]),
            "generatedFileCount": 9,
        },
        "units": units,
        "targetAdapters": [],
        "runtimeVariantRegistry": {
            "available": True,
            "path": registry_path,
            "hash": _sha256(registry_bytes),
            "registryHash": copy.deepcopy(registry["registryHash"]),
            "variantCount": len(registry["variants"]),
            "nativeHeader": {
                "available": True,
                "path": native_header_path,
                "hash": _sha256(native_header_bytes),
            },
        },
        "generatedFiles": [],
    }
    manifest_path = package_root / "native-loader-abi-package.json"
    manifest_path.write_bytes(_json_bytes(package))
    return {
        "packageRoot": package_root,
        "manifestPath": manifest_path,
        "package": package,
        "registry": registry,
        "registryPath": output_registry,
        "nativeHeaderPath": package_root / native_header_path,
        "records": {record["target"]["backend"]: record for record in records},
        "descriptors": descriptors,
    }


def _inputs():
    return {
        "input_values": {
            "dtype": "float32",
            "shape": [8],
            "values": [float(index) for index in range(8)],
        }
    }


def _outputs():
    return {"output_values": {"dtype": "float32", "shape": [8]}}


def _build(fixture, target, *, package_input=None, geometry=None):
    record = fixture["records"][target]
    return build_runtime_variant_dispatch_request(
        fixture["registry"],
        record["key"],
        package_input or fixture["packageRoot"],
        _inputs(),
        _outputs(),
        geometry
        or {
            "workgroupCount": [2, 1, 1],
            "globalSize": [8, 1, 1],
            "metadata": {"case": "copy"},
        },
    )


def _rewrite_package(fixture):
    fixture["manifestPath"].write_bytes(_json_bytes(fixture["package"]))


def _rewrite_packaged_registry(fixture):
    registry = fixture["registry"]
    registry_bytes = _json_bytes(registry)
    fixture["registryPath"].write_bytes(registry_bytes)
    reference = fixture["package"]["runtimeVariantRegistry"]
    reference["hash"] = _sha256(registry_bytes)
    reference["registryHash"] = copy.deepcopy(registry["registryHash"])
    reference["variantCount"] = len(registry["variants"])
    fixture["package"]["summary"]["runtimeVariantCount"] = len(registry["variants"])
    _rewrite_package(fixture)


def _rewrite_descriptor(fixture, target):
    descriptor = fixture["descriptors"][target]
    unit = next(
        unit
        for unit in fixture["package"]["units"]
        if unit["unitId"] == descriptor["unitId"]
    )
    descriptor_bytes = _json_bytes(descriptor)
    (fixture["packageRoot"] / unit["descriptorPath"]).write_bytes(descriptor_bytes)
    unit["descriptorHash"] = _sha256(descriptor_bytes)
    unit["descriptorSizeBytes"] = len(descriptor_bytes)
    _rewrite_package(fixture)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("package_form", ["root", "manifest"])
def test_builds_exact_native_request_for_selected_variant(
    tmp_path, target, package_form
):
    fixture = _write_fixture(
        tmp_path,
        reverse_registry=True,
        reverse_package=True,
    )
    package_input = (
        fixture["packageRoot"] if package_form == "root" else fixture["manifestPath"]
    )

    request = _build(fixture, target, package_input=package_input)

    assert request.artifact["target"] == target
    assert request.fixture.selector.artifact_id == f"{target}|copy"
    assert request.fixture.entry_point == (
        "CopyMain" if target == "directx" else "main"
    )
    assert request.execution_plan.dispatch.workgroup_count == (2, 1, 1)
    assert request.execution_plan.dispatch.workgroup_size == (4, 1, 1)
    assert request.execution_plan.dispatch.global_size == (8, 1, 1)
    assert request.execution_plan.dispatch.metadata["case"] == "copy"
    constant = request.adapter_contract.specialization_constants[0]
    assert constant.constant_id == 7
    assert constant.value == 11
    assert constant.value_provenance["source"] == "explicit"


def test_selects_same_variant_independent_of_registry_and_package_order(tmp_path):
    first = _write_fixture(tmp_path / "first")
    reversed_fixture = _write_fixture(
        tmp_path / "reversed",
        reverse_registry=True,
        reverse_package=True,
    )

    first_request = _build(first, "opengl")
    reversed_request = _build(reversed_fixture, "opengl")

    assert first_request.fixture.to_json() == reversed_request.fixture.to_json()
    assert (
        first_request.adapter_contract.to_json()
        == reversed_request.adapter_contract.to_json()
    )


def test_snapshots_runtime_values_before_building_request(tmp_path):
    fixture = _write_fixture(tmp_path)
    inputs = _inputs()
    outputs = {
        "output_values": {
            "dtype": "float32",
            "shape": [8],
            "values": [0.0] * 8,
        }
    }
    record = fixture["records"]["directx"]

    request = build_runtime_variant_dispatch_request(
        fixture["registry"],
        record["key"],
        fixture["packageRoot"],
        inputs,
        outputs,
        (2, 1, 1),
    )
    inputs["input_values"]["values"][0] = 999.0
    outputs["output_values"]["values"][0] = 999.0

    assert request.fixture.inputs[0].values[0] == 0.0
    assert request.fixture.expected_outputs[0].values[0] == 0.0


def test_snapshots_registry_before_loading_package(tmp_path, monkeypatch):
    fixture = _write_fixture(tmp_path)
    supplied_records = copy.deepcopy(list(fixture["registry"]["variants"].values()))
    opengl_record = next(
        record for record in supplied_records if record["target"]["backend"] == "opengl"
    )
    opengl_record["provenance"]["translation"] = {"pipeline": "different"}
    supplied_registry = _registry(supplied_records)
    packaged_hash = copy.deepcopy(
        fixture["package"]["runtimeVariantRegistry"]["registryHash"]
    )
    load_package = dispatch_module._load_package

    def mutate_registry_then_load_package(value):
        supplied_registry["registryHash"] = packaged_hash
        return load_package(value)

    monkeypatch.setattr(
        dispatch_module,
        "_load_package",
        mutate_registry_then_load_package,
    )
    record = fixture["records"]["directx"]

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        build_runtime_variant_dispatch_request(
            supplied_registry,
            record["key"],
            fixture["packageRoot"],
            _inputs(),
            _outputs(),
            (2, 1, 1),
        )

    assert caught.value.code.endswith(".registry-package-mismatch")


def test_rejects_missing_exact_key_with_lookup_diagnostic(tmp_path):
    fixture = _write_fixture(tmp_path)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        build_runtime_variant_dispatch_request(
            fixture["registry"],
            "crosstl-rvk2:missing",
            fixture["packageRoot"],
            _inputs(),
            _outputs(),
            (2, 1, 1),
        )

    assert caught.value.code.endswith(".lookup-failed")
    assert caught.value.path == "$.key"
    assert caught.value.details["lookupDiagnostics"][0]["code"].startswith(
        "project.runtime-variant-registry."
    )


def test_rejects_modified_registry_before_package_selection(tmp_path):
    fixture = _write_fixture(tmp_path)
    record = fixture["registry"]["variants"][fixture["records"]["directx"]["key"]]
    record["artifact"]["path"] = "artifacts/directx/stale.hlsl"

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".lookup-failed")
    reasons = caught.value.details["lookupDiagnostics"][0]["details"]["reasons"]
    assert any("registryHash" in reason for reason in reasons)


def test_rejects_rehashed_registry_with_incomplete_artifact_identity(tmp_path):
    fixture = _write_fixture(tmp_path)
    records = list(fixture["registry"]["variants"].values())
    selected = next(
        record for record in records if record["target"]["backend"] == "directx"
    )
    del selected["artifact"]["path"]
    fixture["registry"] = _registry(records)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".variant-artifact-invalid")
    assert caught.value.path == "$.selectedVariant.artifact.path"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("kind", "other-package", "package-kind-invalid"),
        ("schemaVersion", 99, "package-version-invalid"),
        ("success", False, "package-unsuccessful"),
    ],
)
def test_rejects_invalid_or_unsuccessful_package(tmp_path, field, value, code):
    fixture = _write_fixture(tmp_path)
    fixture["package"][field] = value
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(f".{code}")


def test_requires_runtime_variant_registry_package_metadata(tmp_path):
    fixture = _write_fixture(tmp_path)
    del fixture["package"]["runtimeVariantRegistry"]
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-schema-invalid")
    assert "runtimeVariantRegistry" in caught.value.details["missingFields"]


def test_rejects_runtime_registry_file_hash_mismatch(tmp_path):
    fixture = _write_fixture(tmp_path)
    fixture["registryPath"].write_bytes(fixture["registryPath"].read_bytes() + b" ")

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-registry-hash-mismatch")
    assert caught.value.path.endswith(".runtimeVariantRegistry.hash.value")


def test_rejects_runtime_registry_identity_mismatch(tmp_path):
    fixture = _write_fixture(tmp_path)
    fixture["package"]["runtimeVariantRegistry"]["registryHash"] = _sha256(
        b"different-registry"
    )
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-registry-identity-mismatch")


def test_rejects_valid_registry_from_another_package(tmp_path):
    fixture = _write_fixture(tmp_path)
    record = fixture["registry"]["variants"][fixture["records"]["directx"]["key"]]
    record["provenance"]["translation"] = {"pipeline": "different"}
    fixture["registry"] = _registry(list(fixture["registry"]["variants"].values()))

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".registry-package-mismatch")
    assert caught.value.path == "$.runtimeVariantRegistry.registryHash"


@pytest.mark.parametrize(
    "registry_path",
    [
        "../runtime-variant-registry.json",
        "/tmp/runtime-variant-registry.json",
        "C:/runtime-variant-registry.json",
        "runtime//runtime-variant-registry.json",
        r"runtime\runtime-variant-registry.json",
        "runtime/CON",
        "runtime/registry.json:payload",
        "runtime/registry.json.",
        "runtime/trailing /registry.json",
    ],
)
def test_rejects_unsafe_runtime_registry_paths(tmp_path, registry_path):
    fixture = _write_fixture(tmp_path)
    fixture["package"]["runtimeVariantRegistry"]["path"] = registry_path
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-registry-path-invalid")


def test_rejects_modified_native_registry_header(tmp_path):
    fixture = _write_fixture(tmp_path)
    fixture["nativeHeaderPath"].write_bytes(b"#pragma once\n// modified\n")

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "opengl")

    assert caught.value.code.endswith(".package-registry-header-hash-mismatch")


def test_accepts_explicitly_unavailable_native_registry_header(tmp_path):
    fixture = _write_fixture(tmp_path)
    fixture["package"]["runtimeVariantRegistry"]["nativeHeader"] = {
        "available": False,
        "reason": "target-adapter-unavailable",
        "unavailableTargets": ["metal"],
    }
    _rewrite_package(fixture)

    request = _build(fixture, "opengl")

    assert request.fixture.selector.artifact_id == "opengl|copy"


@pytest.mark.parametrize(
    "invalid_kind",
    ["duplicate-key", "non-finite-number", "overflowed-number"],
)
def test_rejects_noncanonical_packaged_registry_json(tmp_path, invalid_kind):
    fixture = _write_fixture(tmp_path)
    registry_text = fixture["registryPath"].read_text()
    if invalid_kind == "duplicate-key":
        marker = f'  "kind": "{project_pipeline.RUNTIME_VARIANT_REGISTRY_KIND}",\n'
        invalid_text = registry_text.replace(marker, marker + marker, 1)
    elif invalid_kind == "non-finite-number":
        invalid_text = registry_text.replace(
            '    "candidateCount": 2,\n',
            '    "candidateCount": NaN,\n',
            1,
        )
    else:
        invalid_text = registry_text.replace(
            '    "candidateCount": 2,\n',
            '    "candidateCount": 1e999,\n',
            1,
        )
    assert invalid_text != registry_text
    invalid_bytes = invalid_text.encode()
    fixture["registryPath"].write_bytes(invalid_bytes)
    fixture["package"]["runtimeVariantRegistry"]["hash"] = _sha256(invalid_bytes)
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-registry-json-invalid")


def test_requires_exactly_one_matching_package_unit(tmp_path):
    fixture = _write_fixture(tmp_path)
    selected = next(
        unit for unit in fixture["package"]["units"] if unit["unitId"] == "directx|copy"
    )
    fixture["package"]["units"].append(copy.deepcopy(selected))
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".package-unit-match-invalid")
    assert caught.value.details["matchCount"] == 2


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda descriptor: descriptor.update(target="opengl"),
            "identity-mismatch",
        ),
        (
            lambda descriptor: (
                descriptor.update(stage="fragment"),
                descriptor["entryPoint"].update(stage="fragment"),
            ),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["entryPoint"].update(name="OtherMain"),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["artifact"].update(
                packagePath="artifacts/directx/other.hlsl"
            ),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["artifact"].update(format="DXIL bytecode"),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["artifact"].update(
                hash={"algorithm": "sha256", "value": "0" * 64}
            ),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["source"].update(path="kernels/other.metal"),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["source"].update(backend="cuda"),
            "identity-mismatch",
        ),
        (
            lambda descriptor: descriptor["entryPoint"]["executionConfig"].update(
                workgroupSize=[8, 1, 1]
            ),
            "execution-mismatch",
        ),
        (
            lambda descriptor: descriptor["specializationConstants"][0].update(id=8),
            "specialization-identity-mismatch",
        ),
    ],
)
def test_rejects_descriptor_identity_mismatches(tmp_path, mutation, expected_code):
    fixture = _write_fixture(tmp_path)
    mutation(fixture["descriptors"]["directx"])
    _rewrite_descriptor(fixture, "directx")

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(f".{expected_code}")


def test_rejects_package_descriptor_hash_and_size_mismatches(tmp_path):
    fixture = _write_fixture(tmp_path)
    unit = next(
        unit for unit in fixture["package"]["units"] if unit["unitId"] == "directx|copy"
    )
    unit["descriptorHash"]["value"] = "0" * 64
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as hash_error:
        _build(fixture, "directx")
    assert hash_error.value.code.endswith(".descriptor-hash-mismatch")

    unit["descriptorHash"] = _sha256(
        (fixture["packageRoot"] / unit["descriptorPath"]).read_bytes()
    )
    unit["descriptorSizeBytes"] += 1
    _rewrite_package(fixture)
    with pytest.raises(RuntimeVariantDispatchError) as size_error:
        _build(fixture, "directx")
    assert size_error.value.code.endswith(".descriptor-size-mismatch")


def test_rejects_package_without_required_descriptor_size(tmp_path):
    fixture = _write_fixture(tmp_path)
    for unit in fixture["package"]["units"]:
        unit.pop("descriptorSizeBytes")
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "opengl")

    assert caught.value.code.endswith(".package-unit-schema-invalid")
    assert "descriptorSizeBytes" in caught.value.details["missingFields"]


@pytest.mark.parametrize(
    "descriptor_path",
    [
        "../descriptor.json",
        "/tmp/descriptor.json",
        "C:/descriptor.json",
        "targets//descriptor.json",
        "targets/./descriptor.json",
        r"targets\descriptor.json",
    ],
)
def test_rejects_unsafe_descriptor_paths(tmp_path, descriptor_path):
    fixture = _write_fixture(tmp_path)
    unit = next(
        unit for unit in fixture["package"]["units"] if unit["unitId"] == "directx|copy"
    )
    unit["descriptorPath"] = descriptor_path
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".descriptor-path-invalid")


def test_rejects_descriptor_symlink_escape(tmp_path):
    fixture = _write_fixture(tmp_path)
    unit = next(
        unit for unit in fixture["package"]["units"] if unit["unitId"] == "directx|copy"
    )
    descriptor_path = fixture["packageRoot"] / unit["descriptorPath"]
    outside = tmp_path / "outside.json"
    outside.write_bytes(descriptor_path.read_bytes())
    descriptor_path.unlink()
    try:
        descriptor_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "directx")

    assert caught.value.code.endswith(".descriptor-path-escape")


def test_rejects_tampered_descriptor_json_and_schema(tmp_path):
    fixture = _write_fixture(tmp_path)
    unit = next(
        unit for unit in fixture["package"]["units"] if unit["unitId"] == "opengl|copy"
    )
    descriptor_path = fixture["packageRoot"] / unit["descriptorPath"]
    invalid_json = b"{not-json}\n"
    descriptor_path.write_bytes(invalid_json)
    unit["descriptorHash"] = _sha256(invalid_json)
    unit["descriptorSizeBytes"] = len(invalid_json)
    _rewrite_package(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as json_error:
        _build(fixture, "opengl")
    assert json_error.value.code.endswith(".descriptor-json-invalid")

    descriptor = fixture["descriptors"]["opengl"]
    descriptor["kind"] = "other-descriptor"
    _rewrite_descriptor(fixture, "opengl")
    with pytest.raises(RuntimeVariantDispatchError) as schema_error:
        _build(fixture, "opengl")
    assert schema_error.value.code.endswith(".descriptor-schema-invalid")


def test_rejects_specialization_type_role_and_value_mismatches(tmp_path):
    fixture = _write_fixture(tmp_path)
    record = fixture["records"]["opengl"]
    registry_record = fixture["registry"]["variants"][record["key"]]
    registry_record["specializationConstants"][0]["runtimeRole"] = "uniform"
    fixture["registry"] = _registry(list(fixture["registry"]["variants"].values()))
    _rewrite_packaged_registry(fixture)

    with pytest.raises(RuntimeVariantDispatchError) as role_error:
        _build(fixture, "opengl")
    assert role_error.value.code.endswith(".specialization-role-mismatch")

    fixture = _write_fixture(tmp_path / "type")
    fixture["descriptors"]["opengl"]["specializationConstants"][0]["dtype"] = "int32"
    _rewrite_descriptor(fixture, "opengl")
    with pytest.raises(RuntimeVariantDispatchError) as type_error:
        _build(fixture, "opengl")
    assert type_error.value.code.endswith(".specialization-type-mismatch")


def test_rejects_duplicate_numeric_specialization_identity(tmp_path):
    fixture = _write_fixture(tmp_path)
    duplicate = copy.deepcopy(
        fixture["descriptors"]["opengl"]["specializationConstants"][0]
    )
    duplicate["name"] = "alternate_mode"
    fixture["descriptors"]["opengl"]["specializationConstants"].append(duplicate)
    _rewrite_descriptor(fixture, "opengl")

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(fixture, "opengl")

    assert caught.value.code.endswith(".specialization-identity-ambiguous")


def test_enforces_selected_workgroup_and_preserves_caller_extents(tmp_path):
    fixture = _write_fixture(tmp_path)

    request = _build(
        fixture,
        "opengl",
        geometry={
            "workgroupCount": [3, 2, 1],
            "globalSize": [12, 2, 1],
            "gridSize": [12, 2, 1],
            "metadata": {"batch": 2},
        },
    )

    dispatch = request.execution_plan.dispatch
    assert dispatch.workgroup_size == (4, 1, 1)
    assert dispatch.workgroup_count == (3, 2, 1)
    assert dispatch.global_size == (12, 2, 1)
    assert dispatch.grid_size == (12, 2, 1)
    assert dispatch.metadata["batch"] == 2

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(
            fixture,
            "opengl",
            geometry={
                "workgroupCount": [3, 2, 1],
                "workgroupSize": [8, 1, 1],
            },
        )
    assert caught.value.code.endswith(".workgroup-size-override")


@pytest.mark.parametrize("invalid_count", [[True, 1, 1], [1 << 32, 1, 1]])
def test_rejects_workgroup_counts_outside_native_uint32_range(
    tmp_path,
    invalid_count,
):
    fixture = _write_fixture(tmp_path)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        _build(
            fixture,
            "opengl",
            geometry={"workgroupCount": invalid_count},
        )

    assert caught.value.code.endswith(".execution-invalid")
    assert caught.value.path == "$.dispatchGeometry.workgroupCount"


def test_wraps_native_dispatch_value_diagnostics(tmp_path):
    fixture = _write_fixture(tmp_path)

    with pytest.raises(RuntimeVariantDispatchError) as caught:
        build_runtime_variant_dispatch_request(
            fixture["registry"],
            fixture["records"]["directx"]["key"],
            fixture["packageRoot"],
            {},
            _outputs(),
            (2, 1, 1),
        )

    assert caught.value.code.endswith(".request-invalid")
    diagnostic = caught.value.details["dispatchDiagnostic"]
    assert diagnostic["code"].startswith("project.native-loader-dispatch.")
