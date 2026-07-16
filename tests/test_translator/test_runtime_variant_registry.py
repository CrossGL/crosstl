import base64
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    RUNTIME_VARIANT_REGISTRY_KIND,
    build_runtime_loader_manifest,
    build_runtime_variant_registry,
    decode_runtime_variant_key,
    encode_runtime_variant_key,
    lookup_runtime_variant,
)
from crosstl.project import pipeline as project_pipeline

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (
    ("f32-n4", "float", 4, 1),
    ("i32-n8", "int", 8, 2),
)
TARGETS = ("directx", "opengl")


def _artifact_source(target, target_entry, dtype, width, subgroup_width=None):
    if target == "directx":
        wave_size = (
            f"[WaveSize({subgroup_width})]\n" if subgroup_width is not None else ""
        )
        return f"""
RWStructuredBuffer<{dtype}> values : register(u0, space1);
static const uint tileSize = {width}u;

{wave_size}
[numthreads({width}, 1, 1)]
void {target_entry}(uint3 tid : SV_DispatchThreadID) {{
    values[tid.x] = ({dtype})tid.x;
}}
""".strip()
    return f"""
#version 450
layout(local_size_x = {width}, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer Values {{ {dtype} values[]; }};
const uint tileSize = {width}u;

void main() {{
    values[gl_GlobalInvocationID.x] = {dtype}(gl_GlobalInvocationID.x);
}}
""".strip()


def _declared_host_interface(target, target_entry, width, subgroup_width=None):
    execution_config = {"numthreads": [width, 1, 1]}
    if subgroup_width is not None:
        execution_config["subgroupWidth"] = subgroup_width
    return {
        "status": "ready",
        "source": "synthetic-package-contract",
        "parser": "synthetic",
        "artifactFormat": "HLSL source" if target == "directx" else "GLSL source",
        "entryPointCount": 1,
        "resourceCount": 0,
        "constantCount": 0,
        "specializationConstantCount": 0,
        "entryPoints": [
            {
                "name": target_entry,
                "stage": "compute",
                "executionConfig": execution_config,
            }
        ],
        "resources": [],
        "constants": [],
        "specializationConstants": [],
        "diagnostics": [],
        "diagnosticRecords": [],
    }


def _write_runtime_package(
    package_dir,
    *,
    reverse=False,
    variants=VARIANTS,
    targets=TARGETS,
    execution_identity_only=False,
    subgroup_widths=None,
):
    package_dir.mkdir(parents=True)
    artifacts = []
    subgroup_widths = subgroup_widths or {}
    for variant, dtype, width, mode in variants:
        for target in targets:
            subgroup_width = (
                subgroup_widths.get(variant) if target == "directx" else None
            )
            target_entry = (
                (
                    "vector_add_compiled"
                    if execution_identity_only
                    else f"vector_add_{variant.replace('-', '_')}"
                )
                if target == "directx"
                else "main"
            )
            suffix = "hlsl" if target == "directx" else "glsl"
            package_path = f"artifacts/{target}/{variant}.{suffix}"
            artifact_path = package_dir / package_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                _artifact_source(
                    target,
                    target_entry,
                    dtype,
                    width,
                    subgroup_width,
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts.append(
                {
                    "id": f"{target}|{variant}",
                    "status": "packaged",
                    "source": "kernels/vector.metal",
                    "sourcePath": f"out/{target}/{variant}.{suffix}",
                    "packagePath": package_path,
                    "target": target,
                    "sourceBackend": "metal",
                    "stage": "compute",
                    "variant": variant,
                    "defines": {"USE_FAST_PATH": "1"},
                    "entryPoint": {
                        "source": "vector_add",
                        "target": target_entry,
                        "stage": "compute",
                    },
                    "templateMaterialization": {
                        "status": "materialized",
                        "specializations": [
                            {
                                "name": "vector_add",
                                "hostName": target_entry,
                                "materializedName": target_entry,
                                "parameters": {
                                    "N": str(1 if execution_identity_only else width),
                                    "T": dtype,
                                },
                                "parameterSources": {
                                    "N": "project.variant",
                                    "T": "project.variant",
                                },
                                "source": "project.variant",
                            }
                        ],
                    },
                    "specializationMaterialization": {
                        "status": "materialized",
                        "values": {"7": mode},
                    },
                    "specializationConstants": [
                        {
                            "name": "mode",
                            "id": 7,
                            "kind": "specialization-constant",
                            "dtype": "uint",
                            "concreteValue": mode,
                            "required": False,
                            "source": "project.variant",
                        }
                    ],
                    "provenance": {
                        "sourceArtifact": "kernels/vector.metal",
                        "translation": "synthetic-porting-contract",
                    },
                    "sourceHash": project_pipeline._source_hash(artifact_path),
                    "sourceSizeBytes": artifact_path.stat().st_size,
                    "hash": project_pipeline._source_hash(artifact_path),
                    "sizeBytes": artifact_path.stat().st_size,
                    "sourceRemap": None,
                    "hostInterface": _declared_host_interface(
                        target,
                        target_entry,
                        width,
                        subgroup_width,
                    ),
                }
            )
    if reverse:
        artifacts.reverse()
    package = {
        "schemaVersion": 1,
        "kind": project_pipeline.RUNTIME_PACKAGE_KIND,
        "success": True,
        "packageRoot": str(package_dir),
        "project": {"targets": list(targets)},
        "artifacts": artifacts,
        "runtimePlan": {"runtimeReferenceCount": 0},
    }
    manifest_path = package_dir / "runtime-package.json"
    manifest_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _registry_record(registry, *, target, variant):
    return next(
        record
        for record in registry["variants"].values()
        if record["target"]["backend"] == target and record["variant"] == variant
    )


def _raw_runtime_variant_key(prefix, payload):
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return prefix + encoded.rstrip("=")


def test_runtime_variant_registry_is_deterministic_and_exact(tmp_path):
    first_path = _write_runtime_package(tmp_path / "first")
    reversed_path = _write_runtime_package(tmp_path / "reversed", reverse=True)

    first = build_runtime_variant_registry(first_path)
    reversed_registry = build_runtime_variant_registry(reversed_path)

    assert first == reversed_registry
    assert set(first) == project_pipeline.RUNTIME_VARIANT_REGISTRY_FIELDS
    assert first["kind"] == RUNTIME_VARIANT_REGISTRY_KIND
    assert first["success"] is True
    assert first["status"] == "ready"
    assert first["summary"] == {
        "targetCount": 2,
        "candidateCount": 4,
        "variantCount": 4,
        "readyVariantCount": 4,
        "blockedVariantCount": 0,
        "staleVariantCount": 0,
        "duplicateKeyCount": 0,
        "conflictingKeyCount": 0,
        "rejectedCandidateCount": 0,
    }
    assert list(first["variants"]) == sorted(first["variants"])
    assert first["keySchema"]["version"] == 2
    assert first["keySchema"]["prefix"] == "crosstl-rvk2:"
    assert first["keySchema"]["encoding"] == "base64url-canonical-json-v2"
    assert first["lookup"]["mode"] == "exact"
    assert first["lookup"]["defaulting"] == "none"

    directx = _registry_record(first, target="directx", variant="f32-n4")
    assert set(directx) == project_pipeline.RUNTIME_VARIANT_REGISTRY_RECORD_FIELDS
    assert directx["source"] == {
        "unit": "kernels/vector.metal",
        "backend": "metal",
        "entry": "vector_add",
    }
    assert directx["target"] == {
        "backend": "directx",
        "profile": "cs_6_0",
        "stage": "compute",
        "entryPoint": "vector_add_f32_n4",
    }
    assert directx["arguments"] == {
        "types": {"T": "float"},
        "values": {"N": 4},
    }
    assert directx["specializationConstants"][0]["id"] == 7
    assert directx["specializationConstants"][0]["value"] == 1
    assert directx["specializationConstants"][0]["runtimeRole"] == (
        "pipeline-specialization"
    )
    assert directx["execution"] == {
        "workgroupSize": [4, 1, 1],
        "subgroupWidth": None,
    }
    assert directx["bindingInterface"]["source"] == "compiled-artifact"
    assert directx["bindingInterface"]["parser"] == "directx-reflection"
    assert directx["bindingInterface"]["constants"][0]["name"] == "tileSize"
    assert directx["bindingInterface"]["resources"][0]["name"] == "values"
    assert directx["artifact"]["hash"]["algorithm"] == "sha256"
    assert directx["artifact"]["path"] == "artifacts/directx/f32-n4.hlsl"
    assert directx["provenance"]["translation"]["sourceArtifact"] == (
        "kernels/vector.metal"
    )
    assert decode_runtime_variant_key(directx["key"]) == {
        "sourceUnit": "kernels/vector.metal",
        "sourceEntry": "vector_add",
        "target": "directx",
        "targetProfile": "cs_6_0",
        "execution": {"workgroupSize": [4, 1, 1], "subgroupWidth": None},
        "typeArguments": {"T": "float"},
        "valueArguments": {"N": 4},
        "specializationConstants": [{"id": 7, "name": "mode", "value": 1}],
        "defines": {"USE_FAST_PATH": "1"},
    }
    assert _registry_record(first, target="opengl", variant="i32-n8")["target"] == {
        "backend": "opengl",
        "profile": None,
        "stage": "compute",
        "entryPoint": "main",
    }

    found = lookup_runtime_variant(first, directx["key"])
    assert set(found) == project_pipeline.RUNTIME_VARIANT_LOOKUP_FIELDS
    assert found["success"] is True
    assert found["match"] == "exact"
    assert found["record"] == directx

    missing_key = encode_runtime_variant_key(
        "kernels/vector.metal",
        "vector_add",
        "directx",
        target_profile="cs_6_0",
        execution={"workgroupSize": [4, 1, 1], "subgroupWidth": None},
        type_arguments={"T": "float"},
        value_arguments={"N": 64},
        specialization_constants=[{"id": 7, "name": "mode", "value": 1}],
        defines={"USE_FAST_PATH": "1"},
    )
    missing = lookup_runtime_variant(first, missing_key)
    assert missing["success"] is False
    assert missing["status"] == "not-found"
    assert missing["record"] is None
    assert missing["availableKeys"] == sorted(first["variants"])
    assert missing["diagnostics"][0]["code"] == (
        "project.runtime-variant-registry.variant-not-found"
    )


def test_runtime_variant_execution_keys_are_distinct_reorderable_and_exact(tmp_path):
    variants = (
        ("wg32", "float", 32, 1),
        ("wg64", "float", 64, 1),
    )
    first = build_runtime_variant_registry(
        _write_runtime_package(
            tmp_path / "first",
            variants=variants,
            execution_identity_only=True,
        )
    )
    reordered = build_runtime_variant_registry(
        _write_runtime_package(
            tmp_path / "reordered",
            reverse=True,
            variants=tuple(reversed(variants)),
            execution_identity_only=True,
        )
    )

    assert first == reordered
    assert first["success"] is True
    assert first["summary"]["variantCount"] == 4
    for target in TARGETS:
        records = {
            record["variant"]: record
            for record in first["variants"].values()
            if record["target"]["backend"] == target
        }
        assert set(records) == {"wg32", "wg64"}
        assert records["wg32"]["key"] != records["wg64"]["key"]
        assert {
            variant: record["execution"] for variant, record in records.items()
        } == {
            "wg32": {"workgroupSize": [32, 1, 1], "subgroupWidth": None},
            "wg64": {"workgroupSize": [64, 1, 1], "subgroupWidth": None},
        }

        selected = lookup_runtime_variant(first, records["wg32"]["key"])
        assert selected["success"] is True
        assert selected["record"]["variant"] == "wg32"

        requested_execution = {
            "workgroupSize": [48, 1, 1],
            "subgroupWidth": None,
        }
        missing_key = encode_runtime_variant_key(
            "kernels/vector.metal",
            "vector_add",
            target,
            target_profile="cs_6_0" if target == "directx" else None,
            execution=requested_execution,
            type_arguments={"T": "float"},
            value_arguments={"N": 1},
            specialization_constants=[{"id": 7, "name": "mode", "value": 1}],
            defines={"USE_FAST_PATH": "1"},
        )
        missing = lookup_runtime_variant(first, missing_key)
        details = missing["diagnostics"][0]["details"]
        assert missing["status"] == "not-found"
        assert details["requestedExecution"] == requested_execution
        assert {
            tuple(alternative["execution"]["workgroupSize"])
            for alternative in details["availableExecutionAlternatives"]
        } == {(32, 1, 1), (64, 1, 1)}
        assert all(
            alternative["status"] == "ready"
            for alternative in details["availableExecutionAlternatives"]
        )


def test_runtime_variant_execution_uses_only_the_selected_entry(tmp_path):
    variants = (("selected", "float", 32, 1),)
    registries = []
    for directory, reverse_entries in (("first", False), ("reordered", True)):
        package_path = _write_runtime_package(
            tmp_path / directory,
            variants=variants,
            targets=("directx",),
            execution_identity_only=True,
        )
        package = json.loads(package_path.read_text(encoding="utf-8"))
        artifact = package["artifacts"][0]
        selected_entry = artifact["hostInterface"]["entryPoints"][0]
        other_entry = {
            "name": "aaa_unselected",
            "stage": "compute",
            "executionConfig": {"numthreads": [64, 1, 1]},
        }
        entries = [selected_entry, other_entry]
        if reverse_entries:
            entries.reverse()
        artifact["hostInterface"]["entryPoints"] = entries
        artifact["hostInterface"]["entryPointCount"] = 2
        package["project"]["workgroupSize"] = [128, 1, 1]
        package_path.write_text(json.dumps(package), encoding="utf-8")
        registries.append(build_runtime_variant_registry(package_path))

    assert registries[0] == registries[1]
    record = next(iter(registries[0]["variants"].values()))
    assert record["target"]["entryPoint"] == "vector_add_compiled"
    assert record["execution"] == {
        "workgroupSize": [32, 1, 1],
        "subgroupWidth": None,
    }


def test_runtime_variant_subgroup_width_distinguishes_exact_variants(tmp_path):
    variants = (
        ("wave32", "float", 32, 1),
        ("wave64", "float", 32, 1),
    )
    registry = build_runtime_variant_registry(
        _write_runtime_package(
            tmp_path / "package",
            variants=variants,
            targets=("directx",),
            execution_identity_only=True,
            subgroup_widths={"wave32": 32, "wave64": 64},
        )
    )

    assert registry["success"] is True
    records = {record["variant"]: record for record in registry["variants"].values()}
    assert records["wave32"]["key"] != records["wave64"]["key"]
    assert {
        variant: decode_runtime_variant_key(record["key"])["execution"]
        for variant, record in records.items()
    } == {
        "wave32": {"workgroupSize": [32, 1, 1], "subgroupWidth": 32},
        "wave64": {"workgroupSize": [32, 1, 1], "subgroupWidth": 64},
    }

    original_key = records["wave32"]["key"]
    tampered = copy.deepcopy(registry)
    tampered_record = tampered["variants"].pop(original_key)
    tampered_payload = decode_runtime_variant_key(original_key)
    tampered_payload["execution"]["subgroupWidth"] = 16
    tampered_key = _raw_runtime_variant_key("crosstl-rvk2:", tampered_payload)
    tampered_record["key"] = tampered_key
    tampered_record["execution"] = tampered_payload["execution"]
    tampered["variants"][tampered_key] = tampered_record
    tampered["registryHash"] = project_pipeline._runtime_variant_payload_hash(
        {
            "schemaVersion": tampered["schemaVersion"],
            "keySchema": tampered["keySchema"],
            "variants": tampered["variants"],
        }
    )

    lookup = lookup_runtime_variant(tampered, tampered_key)
    reasons = lookup["diagnostics"][0]["details"]["reasons"]
    assert lookup["status"] == "invalid"
    assert any("selected bindingInterface entry point" in reason for reason in reasons)


def test_runtime_variant_key_round_trips_absent_execution_and_reordered_inputs():
    first = encode_runtime_variant_key(
        "kernels/vector.metal",
        "vector_add",
        "directx",
        target_profile="cs_6_0",
        type_arguments={"Z": "uint", "A": "float"},
        value_arguments={"width": 4, "mode": True},
        specialization_constants=[
            {"id": None, "name": "beta", "value": 2},
            {"id": 7, "name": "mode", "value": 1},
        ],
        defines={"ZED": "1", "ALPHA": "1"},
    )
    reordered = encode_runtime_variant_key(
        "kernels/vector.metal",
        "vector_add",
        "directx",
        target_profile="cs_6_0",
        execution={"subgroupWidth": None, "workgroupSize": None},
        type_arguments={"A": "float", "Z": "uint"},
        value_arguments={"mode": True, "width": 4},
        specialization_constants=[
            {"id": 7, "name": "mode", "value": 1},
            {"id": None, "name": "beta", "value": 2},
        ],
        defines={"ALPHA": "1", "ZED": "1"},
    )

    assert first == reordered
    assert first.startswith("crosstl-rvk2:")
    assert decode_runtime_variant_key(first)["execution"] == {
        "workgroupSize": None,
        "subgroupWidth": None,
    }


@pytest.mark.parametrize(
    ("execution", "expected_reason"),
    [
        (
            {"workgroupSize": [32, 1], "subgroupWidth": None},
            "three-component list",
        ),
        (
            {"workgroupSize": [32, 0, 1], "subgroupWidth": None},
            "positive integer",
        ),
        (
            {"workgroupSize": [32, True, 1], "subgroupWidth": None},
            "positive integer",
        ),
        (
            {"workgroupSize": [32, "1", 1], "subgroupWidth": None},
            "positive integer",
        ),
        (
            {"workgroupSize": None, "subgroupWidth": 0},
            "positive integer or null",
        ),
        (
            {"workgroupSize": None, "subgroupWidth": True},
            "positive integer or null",
        ),
        (
            {
                "workgroupSize": None,
                "subgroupWidth": None,
                "backendPolicy": "directx",
            },
            "not allowed",
        ),
        ([], "must be an object"),
    ],
)
def test_runtime_variant_key_rejects_malformed_execution(execution, expected_reason):
    with pytest.raises(ValueError, match=expected_reason):
        encode_runtime_variant_key(
            "kernels/vector.metal",
            "vector_add",
            "directx",
            execution=execution,
        )


@pytest.mark.parametrize(
    ("execution_config", "expected_reason"),
    [
        ({"numthreads": ["WIDTH", 1, 1]}, "exact positive integers"),
        ({"numthreads": [32, 0, 1]}, "exact positive integers"),
        ({"numthreads": [64, 1, 1]}, "conflicting workgroup sizes"),
        (
            {
                "numthreads": [32, 1, 1],
                "workgroupSize": [64, 1, 1],
            },
            "conflicting workgroup sizes",
        ),
        (
            {"numthreads": [32, 1, 1], "subgroupWidth": "WAVE"},
            "exact positive integer",
        ),
        (
            {"numthreads": [32, 1, 1], "subgroupWidth": 64},
            "exact positive integer",
        ),
    ],
)
def test_runtime_variant_registry_rejects_inexact_selected_execution(
    tmp_path, execution_config, expected_reason
):
    package_path = _write_runtime_package(
        tmp_path / "package",
        variants=(("selected", "float", 32, 1),),
        targets=("directx",),
        execution_identity_only=True,
    )
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["artifacts"][0]["hostInterface"]["entryPoints"][0][
        "executionConfig"
    ] = execution_config
    package_path.write_text(json.dumps(package), encoding="utf-8")

    registry = build_runtime_variant_registry(package_path)

    assert registry["success"] is False
    assert registry["summary"]["readyVariantCount"] == 0
    reasons = [
        reason
        for diagnostic in registry["diagnostics"]
        for reason in diagnostic.get("details", {}).get("reasons", [])
    ]
    assert any(expected_reason in reason for reason in reasons)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("payload", "registryHash does not match its variants"),
        ("identity", "identity does not match its canonical key"),
        ("execution", "selected bindingInterface entry point"),
        ("binding-execution", "selected bindingInterface entry point"),
        ("nested-schema", "cannot reconstruct its canonical identity"),
    ],
)
def test_runtime_variant_lookup_rejects_modified_registries(
    tmp_path, mutation, expected_reason
):
    registry = build_runtime_variant_registry(
        _write_runtime_package(tmp_path / "package")
    )
    record = _registry_record(registry, target="directx", variant="f32-n4")
    key = record["key"]
    modified = copy.deepcopy(registry)

    if mutation == "payload":
        modified["variants"][key]["artifact"]["path"] = "artifacts/other.hlsl"
    elif mutation == "identity":
        modified["variants"][key]["arguments"]["values"]["N"] = 64
        modified["registryHash"] = project_pipeline._runtime_variant_payload_hash(
            {
                "schemaVersion": modified["schemaVersion"],
                "keySchema": modified["keySchema"],
                "variants": modified["variants"],
            }
        )
    elif mutation == "execution":
        modified["variants"][key]["execution"]["workgroupSize"] = [64, 1, 1]
        modified["registryHash"] = project_pipeline._runtime_variant_payload_hash(
            {
                "schemaVersion": modified["schemaVersion"],
                "keySchema": modified["keySchema"],
                "variants": modified["variants"],
            }
        )
    elif mutation == "binding-execution":
        modified["variants"][key]["bindingInterface"]["entryPoint"]["executionConfig"][
            "numthreads"
        ] = [64, 1, 1]
        modified["registryHash"] = project_pipeline._runtime_variant_payload_hash(
            {
                "schemaVersion": modified["schemaVersion"],
                "keySchema": modified["keySchema"],
                "variants": modified["variants"],
            }
        )
    else:
        modified["variants"][key]["arguments"]["types"] = []
        modified["registryHash"] = project_pipeline._runtime_variant_payload_hash(
            {
                "schemaVersion": modified["schemaVersion"],
                "keySchema": modified["keySchema"],
                "variants": modified["variants"],
            }
        )

    lookup = lookup_runtime_variant(modified, key)

    assert lookup["success"] is False
    assert lookup["status"] == "invalid"
    assert lookup["record"] is None
    assert lookup["availableKeys"] == []
    assert lookup["diagnostics"][0]["code"] == (
        "project.runtime-variant-registry.lookup-registry-invalid"
    )
    reasons = lookup["diagnostics"][0]["details"]["reasons"]
    assert any(expected_reason in reason for reason in reasons)


def test_runtime_variant_lookup_rejects_reencoded_workgroup_tampering(tmp_path):
    registry = build_runtime_variant_registry(
        _write_runtime_package(tmp_path / "package")
    )
    record = _registry_record(registry, target="directx", variant="f32-n4")
    original_key = record["key"]
    tampered = copy.deepcopy(registry)
    tampered_record = tampered["variants"].pop(original_key)
    tampered_payload = decode_runtime_variant_key(original_key)
    tampered_payload["execution"]["workgroupSize"] = [64, 1, 1]
    tampered_key = _raw_runtime_variant_key("crosstl-rvk2:", tampered_payload)
    tampered_record["key"] = tampered_key
    tampered_record["execution"] = tampered_payload["execution"]
    tampered["variants"][tampered_key] = tampered_record
    tampered["registryHash"] = project_pipeline._runtime_variant_payload_hash(
        {
            "schemaVersion": tampered["schemaVersion"],
            "keySchema": tampered["keySchema"],
            "variants": tampered["variants"],
        }
    )

    lookup = lookup_runtime_variant(tampered, tampered_key)

    assert lookup["status"] == "invalid"
    reasons = lookup["diagnostics"][0]["details"]["reasons"]
    assert any("selected bindingInterface entry point" in reason for reason in reasons)


def test_runtime_variant_registry_accepts_loader_manifest(tmp_path):
    package_path = _write_runtime_package(tmp_path / "package")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    for artifact in package["artifacts"]:
        artifact["hostInterface"] = None
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package_registry = build_runtime_variant_registry(package_path)
    loader = build_runtime_loader_manifest(package_path)
    loader_path = tmp_path / "runtime-loader-manifest.json"
    loader_path.write_text(
        json.dumps(loader, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    loader_registry = build_runtime_variant_registry(loader_path)

    assert loader_registry["success"] is True
    assert loader_registry["source"]["kind"] == (
        project_pipeline.RUNTIME_LOADER_MANIFEST_KIND
    )
    assert list(loader_registry["variants"]) == list(package_registry["variants"])
    record = _registry_record(loader_registry, target="directx", variant="i32-n8")
    assert record["source"]["entry"] == "vector_add"
    assert record["target"]["entryPoint"] == "vector_add_i32_n8"
    assert record["artifact"]["hash"]["algorithm"] == "sha256"
    assert record["provenance"]["inputKind"] == (
        project_pipeline.RUNTIME_LOADER_MANIFEST_KIND
    )


@pytest.mark.parametrize(
    ("conflicting", "diagnostic_code", "summary_field"),
    [
        (
            False,
            "project.runtime-variant-registry.duplicate-key",
            "duplicateKeyCount",
        ),
        (
            True,
            "project.runtime-variant-registry.conflicting-key",
            "conflictingKeyCount",
        ),
    ],
)
def test_runtime_variant_registry_rejects_duplicate_keys(
    tmp_path, conflicting, diagnostic_code, summary_field
):
    package_path = _write_runtime_package(tmp_path / "package")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    duplicate = copy.deepcopy(package["artifacts"][0])
    if conflicting:
        duplicate["id"] = duplicate["id"] + "|conflict"
        old_path = Path(package["packageRoot"]) / duplicate["packagePath"]
        duplicate["packagePath"] = "artifacts/directx/conflict.hlsl"
        new_path = Path(package["packageRoot"]) / duplicate["packagePath"]
        new_path.write_text(old_path.read_text(encoding="utf-8"), encoding="utf-8")
    package["artifacts"].append(duplicate)
    package_path.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    registry = build_runtime_variant_registry(package_path)

    assert registry["success"] is False
    assert registry["status"] == "failed"
    assert registry["summary"][summary_field] == 1
    assert registry["summary"]["candidateCount"] == 5
    assert registry["summary"]["variantCount"] == 3
    assert registry["summary"]["rejectedCandidateCount"] == 2
    assert registry["diagnostics"][0]["code"] == diagnostic_code


def test_runtime_variant_registry_propagates_stale_package_records(tmp_path):
    package_path = _write_runtime_package(tmp_path / "package")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    stale_artifact = package["artifacts"][0]
    stale_path = Path(package["packageRoot"]) / stale_artifact["packagePath"]
    stale_path.write_text("// stale artifact\n", encoding="utf-8")

    registry = build_runtime_variant_registry(package_path)

    assert registry["success"] is False
    assert registry["status"] == "failed"
    assert registry["summary"]["readyVariantCount"] == 3
    assert registry["summary"]["staleVariantCount"] == 1
    stale = next(
        record
        for record in registry["variants"].values()
        if record["status"] == "stale"
    )
    assert stale["lookup"]["eligible"] is False
    assert {blocker["code"] for blocker in stale["blockers"]} >= {
        "project.runtime-package-inspection.artifact-hash-mismatch",
        "project.runtime-package-inspection.artifact-size-mismatch",
    }
    lookup = lookup_runtime_variant(registry, stale["key"])
    assert lookup["success"] is False
    assert lookup["status"] == "blocked"
    assert lookup["record"]["status"] == "stale"


def test_runtime_variant_registry_propagates_loader_blockers(tmp_path):
    package_path = _write_runtime_package(tmp_path / "package")
    loader = build_runtime_loader_manifest(package_path)
    for load_unit in loader["loadUnits"]:
        load_unit["blockers"] = [
            {
                "kind": "host-capability-unavailable",
                "severity": "warning",
                "message": "Required host capability is unavailable.",
            }
        ]
        load_unit["validation"]["loadReady"] = False
    loader_path = tmp_path / "blocked-loader.json"
    loader_path.write_text(
        json.dumps(loader, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    registry = build_runtime_variant_registry(loader_path)

    assert registry["success"] is True
    assert registry["status"] == "blocked"
    assert registry["summary"]["readyVariantCount"] == 0
    assert registry["summary"]["blockedVariantCount"] == 4
    assert registry["diagnosticCounts"]["warning"] == 4
    assert all(
        record["blockers"][0]["code"] == "host-capability-unavailable"
        for record in registry["variants"].values()
    )


def test_runtime_variant_registry_rejects_malformed_inputs_and_keys(tmp_path):
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text('{"schemaVersion": 1,', encoding="utf-8")
    malformed = build_runtime_variant_registry(malformed_json)
    assert malformed["success"] is False
    assert malformed["diagnostics"][0]["code"] == (
        "project.runtime-variant-registry.input-json-invalid"
    )

    package_path = _write_runtime_package(tmp_path / "package")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["artifacts"][0]["unexpectedField"] = True
    package["artifacts"][0]["specializationConstants"].append(
        copy.deepcopy(package["artifacts"][0]["specializationConstants"][0])
    )
    package["artifacts"][1]["hash"] = {"algorithm": "md5", "value": "bad"}
    package_path.write_text(json.dumps(package), encoding="utf-8")
    invalid = build_runtime_variant_registry(package_path)
    assert invalid["success"] is False
    assert invalid["status"] == "failed"
    assert invalid["variants"] == {}
    assert invalid["diagnostics"][0]["code"] == (
        "project.runtime-variant-registry.input-schema-invalid"
    )
    reasons = invalid["diagnostics"][0]["details"]["reasons"]
    assert any("unexpectedField" in reason for reason in reasons)
    assert any("duplicates specialization identity" in reason for reason in reasons)
    assert any("sha256" in reason for reason in reasons)

    with pytest.raises(ValueError, match="payload is invalid"):
        decode_runtime_variant_key("crosstl-rvk2:not-base64!")

    valid_key = encode_runtime_variant_key(
        "kernels/vector.metal",
        "vector_add",
        "directx",
        execution={"workgroupSize": [32, 1, 1], "subgroupWidth": 32},
    )
    malformed_payload = decode_runtime_variant_key(valid_key)
    malformed_payload["execution"]["workgroupSize"] = [32, -1, 1]
    malformed_key = _raw_runtime_variant_key("crosstl-rvk2:", malformed_payload)
    with pytest.raises(ValueError, match="positive integer"):
        decode_runtime_variant_key(malformed_key)


def test_runtime_variant_key_rejects_legacy_schema_with_migration_error(tmp_path):
    registry = build_runtime_variant_registry(
        _write_runtime_package(tmp_path / "package")
    )
    current_key = next(iter(registry["variants"]))
    legacy_payload = decode_runtime_variant_key(current_key)
    legacy_payload.pop("execution")
    legacy_key = _raw_runtime_variant_key("crosstl-rvk1:", legacy_payload)

    with pytest.raises(
        ValueError,
        match="Legacy runtime variant key schema v1.*regenerate",
    ):
        decode_runtime_variant_key(legacy_key)

    lookup = lookup_runtime_variant(registry, legacy_key)
    assert lookup["status"] == "invalid"
    assert lookup["diagnostics"][0]["code"] == (
        "project.runtime-variant-registry.lookup-key-invalid"
    )
    assert "regenerate the key and registry" in lookup["diagnostics"][0]["message"]


def test_runtime_variant_registry_cli_json_and_text(tmp_path):
    package_path = _write_runtime_package(tmp_path / "package")
    output_path = tmp_path / "registry.json"
    json_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "runtime-variant-registry",
            str(package_path),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    text_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crosstl._crosstl",
            "runtime-variant-registry",
            str(package_path),
            "--format",
            "text",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert json_result.returncode == 0
    assert json_result.stdout == f"Wrote {output_path}\n"
    registry = json.loads(output_path.read_text(encoding="utf-8"))
    assert registry["kind"] == RUNTIME_VARIANT_REGISTRY_KIND
    assert registry["summary"]["variantCount"] == 4
    assert text_result.returncode == 0
    assert "Runtime variant registry\n" in text_result.stdout
    assert "Status: ready" in text_result.stdout
    assert "Lookup: exact; defaulting: none" in text_result.stdout
    assert "Summary: 2 targets, 4 variants, 4 ready, 0 blocked, 0 stale" in (
        text_result.stdout
    )
    assert "- directx: 2 variants, 2 ready, 0 blocked, 0 stale" in (text_result.stdout)
