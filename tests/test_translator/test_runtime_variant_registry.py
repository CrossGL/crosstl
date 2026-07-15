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


def _artifact_source(target, target_entry, dtype, width):
    if target == "directx":
        return f"""
RWStructuredBuffer<{dtype}> values : register(u0, space1);
static const uint tileSize = {width}u;

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


def _declared_host_interface(target, target_entry, width):
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
                "executionConfig": {"numthreads": [width, 1, 1]},
            }
        ],
        "resources": [],
        "constants": [],
        "specializationConstants": [],
        "diagnostics": [],
        "diagnosticRecords": [],
    }


def _write_runtime_package(package_dir, *, reverse=False):
    package_dir.mkdir(parents=True)
    artifacts = []
    for variant, dtype, width, mode in VARIANTS:
        for target in TARGETS:
            target_entry = (
                f"vector_add_{variant.replace('-', '_')}"
                if target == "directx"
                else "main"
            )
            suffix = "hlsl" if target == "directx" else "glsl"
            package_path = f"artifacts/{target}/{variant}.{suffix}"
            artifact_path = package_dir / package_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                _artifact_source(target, target_entry, dtype, width) + "\n",
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
                                "parameters": {"N": str(width), "T": dtype},
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
                        target, target_entry, width
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
        "project": {"targets": list(TARGETS)},
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
        decode_runtime_variant_key("crosstl-rvk1:not-base64!")


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
