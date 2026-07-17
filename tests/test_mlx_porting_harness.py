import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINNED_MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
HARNESS_PATH = ROOT / "demos" / "integrations" / "mlx" / "run_mlx_porting.py"
RMS_NORM_HARNESS_PATH = (
    ROOT / "demos" / "integrations" / "mlx" / "prove_rms_norm_specialization.py"
)
MLX_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "mlx-project-porting.yml"
MLX_README_PATH = ROOT / "demos" / "integrations" / "mlx" / "README.md"
RMS_NORM_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "project_porting" / "mlx"


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_mlx_porting", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rms_norm_harness():
    spec = importlib.util.spec_from_file_location(
        "mlx_rms_norm_specialization_proof", RMS_NORM_HARNESS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mlx_porting_contract_uses_exact_pinned_revision():
    module = _load_harness()
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert module.MLX_COMMIT == PINNED_MLX_COMMIT
    assert expected_gaps["commit"] == PINNED_MLX_COMMIT


def test_project_config_writer_emits_general_index_range_assertions(tmp_path):
    from crosstl.project import load_project_config

    module = _load_harness()
    config_path = tmp_path / "frontier.toml"
    assertions = (
        {
            "source": "mlx/backend/metal/kernels/example.metal",
            "function": "gather_values",
            "expression": "base + index",
            "minimum": 0,
            "maximum": 1023,
        },
        {
            "source": "mlx/backend/metal/kernels/other.metal",
            "expression": "position.x",
            "minimum": 4,
            "maximum": 31,
        },
    )

    module._write_project_config(
        config_path,
        include=tuple(assertion["source"] for assertion in assertions),
        targets=("opengl",),
        output_dir="generated",
        index_range_assertions=assertions,
    )

    config = load_project_config(tmp_path, config_path)
    assert [assertion.to_json() for assertion in config.index_range_assertions] == list(
        assertions
    )
    assert config_path.read_text(encoding="utf-8").count(
        "[[project.index_range_assertions]]"
    ) == len(assertions)


def _load_rms_norm_fixture_metadata():
    return json.loads(
        (
            RMS_NORM_FIXTURE_ROOT / "rms_norm_specialization.fixture-metadata.json"
        ).read_text(encoding="utf-8")
    )


def _prepare_reduced_rms_norm_checkout(module, tmp_path, monkeypatch):
    mlx_root = tmp_path / "mlx"
    source_path = mlx_root / module.MLX_RMS_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    host_entry_points = _load_rms_norm_fixture_metadata()["hostNamedEntryPoints"]
    instantiations = []
    for index, entry_point in enumerate(host_entry_points):
        template = (
            "rms_looped_fixture"
            if "_looped" in entry_point
            else "rms_single_row_fixture"
        )
        instantiations.append(
            f'instantiate_kernel("{entry_point}", {template}, {index})'
        )
    source_path.write_text(
        """#include <metal_stdlib>
using namespace metal;

constant bool has_w [[function_constant(20)]];

template <int TAG>
[[kernel]] void rms_single_row_fixture(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  output[index] = (has_w ? 1.0f : 0.0f) + float(TAG);
}

template <int TAG>
[[kernel]] void rms_looped_fixture(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]],
    uint lsize [[threads_per_threadgroup]]) {
  output[index] = (has_w ? 1.0f : 0.0f) + float(TAG) + float(lsize);
}

""" + "\n".join(instantiations) + "\n",
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "MLX_RMS_NORM_SHA256", source_hash)
    work_dir = mlx_root / ".rms-norm-proof"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    report_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return mlx_root, work_dir, report_dir, log_dir


def _full_corpus_report(module, mlx_root, work_dir, *, include_extra_failure=False):
    per_target = {
        target: {
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT - 1,
            "failedCount": 1,
        }
        for target in module.FULL_CORPUS_TARGETS
    }
    diagnostics_by_code = {
        contract["diagnosticCode"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    diagnostics_by_code["project.validate.failed-artifact"] = 3
    missing_capability_counts = {
        contract["missingCapability"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    missing_capability_counts["batch.translation"] = 3
    diagnostics = []
    artifacts = []
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        message = module._atomic_fence_expected_message(contract)
        artifact_path = (
            work_dir
            / "out-full-corpus"
            / target
            / Path(module.MLX_FENCE_SOURCE).with_suffix(extensions[target])
        ).relative_to(mlx_root)
        diagnostics.append(
            {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": message,
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            }
        )
        diagnostics.append(
            {
                "severity": "error",
                "code": "project.validate.failed-artifact",
                "message": f"Artifact translation failed before validation: {message}",
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": ["batch.translation"],
            }
        )
        artifacts.append(
            {
                "source": module.MLX_FENCE_SOURCE,
                "sourceBackend": "metal",
                "target": target,
                "path": artifact_path.as_posix(),
                "status": "failed",
                "error": message,
            }
        )

    translated_count = module.FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
    failed_count = module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    if include_extra_failure:
        per_target["directx"] = {
            "translatedCount": module.EXPECTED_METAL_KERNEL_COUNT - 2,
            "failedCount": 2,
        }
        translated_count -= 1
        failed_count += 1
        diagnostics_by_code["project.translate.failed"] = 1
        diagnostics_by_code["project.validate.failed-artifact"] += 1
        missing_capability_counts["batch.translation"] += 2
        diagnostics.extend(
            [
                {
                    "severity": "error",
                    "code": "project.translate.failed",
                    "message": "unrelated full-corpus translation failure",
                    "location": {"file": "mlx/backend/metal/kernels/other.metal"},
                    "target": "directx",
                    "sourceBackend": "metal",
                    "missingCapabilities": ["batch.translation"],
                },
                {
                    "severity": "error",
                    "code": "project.validate.failed-artifact",
                    "message": "Artifact translation failed before validation",
                    "location": {"file": "mlx/backend/metal/kernels/other.metal"},
                    "target": "directx",
                    "sourceBackend": "metal",
                    "missingCapabilities": ["batch.translation"],
                },
            ]
        )
        artifacts.append(
            {
                "source": "mlx/backend/metal/kernels/other.metal",
                "sourceBackend": "metal",
                "target": "directx",
                "path": (
                    (work_dir / "out-full-corpus/directx/other.hlsl")
                    .relative_to(mlx_root)
                    .as_posix()
                ),
                "status": "failed",
                "error": "unrelated full-corpus translation failure",
            }
        )

    summary = {
        "unitCount": module.EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "translatedCount": translated_count,
        "failedCount": failed_count,
        "diagnosticCounts": {
            "error": len(diagnostics),
            "note": 0,
            "warning": 0,
        },
        "diagnosticsByCode": diagnostics_by_code,
        "missingCapabilityCounts": missing_capability_counts,
        "artifactsByTarget": per_target,
    }
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "artifacts": artifacts,
        "validation": {"summary": {"failedCount": failed_count}},
    }


def _translated_arange_report(module, target):
    return {
        "kind": "crosstl-project-portability-report",
        "project": {"targets": [target]},
        "artifacts": [
            {
                "source": module.MLX_ARANGE_SOURCE,
                "path": f"out/{target}/arange",
                "target": target,
                "sourceBackend": "metal",
                "status": "translated",
            }
        ],
    }


def _runtime_arange_artifact_manifest(module, target, output_name="out"):
    entry_point = module.RUNTIME_READINESS_ENTRY_POINTS[target]
    return {
        "kind": "crosstl-project-runtime-artifact-manifest",
        "project": {"targets": [target]},
        "summary": {
            "artifactCount": 1,
            "entryPointCount": 1,
            "resourceBindingCount": 3,
            "dispatchMetadataCount": 1,
        },
        "artifacts": [
            {
                "id": (
                    f"{module.MLX_ARANGE_SOURCE}|{target}|default|out/{target}/arange"
                ),
                "source": module.MLX_ARANGE_SOURCE,
                "path": f"out/{target}/arange",
                "target": target,
                "sourceBackend": "metal",
                "status": "translated",
                "entryPoints": [
                    {
                        "name": entry_point,
                        "stage": "compute",
                        "workgroupSize": [1, 1, 1],
                    }
                ],
                "resourceBindings": [
                    {
                        "name": "start",
                        "kind": "constant",
                        "binding": 0,
                    },
                    {
                        "name": "step",
                        "kind": "constant",
                        "binding": 1,
                    },
                    {
                        "name": output_name,
                        "kind": "buffer",
                        "binding": 2,
                    },
                ],
                "dispatch": {
                    "entryPoint": entry_point,
                    "workgroupSize": [1, 1, 1],
                    "workgroupCount": [1, 1, 1],
                },
            }
        ],
        "runtimeDiagnosticCounts": {"note": 0, "warning": 0, "error": 0},
        "runtimeDiagnostics": [],
    }


def _write_reference_accessor_report(
    module,
    work_dir,
    report_path,
    *,
    generated_by_target=None,
):
    defaults = {
        "directx": (
            """
            struct ReferenceAccessorTile {
              float val_frags[4];
            };
            struct ReferenceAccessorFragmentTile {
              float2 val_frags[4];
            };
            struct ReferenceAccessorStoreLoop {
              ReferenceAccessorFragmentTile nestedTile;
            };
            RWStructuredBuffer<float> out : register(u0);
            RWStructuredBuffer<float2> nestedOut : register(u1);
            void ReferenceAccessorOps__store(
                inout float value,
                RWStructuredBuffer<float> out) {
              out[1] = value;
            }
            void ReferenceAccessorTile__store(
                in ReferenceAccessorTile self,
                RWStructuredBuffer<float> out,
                int i,
                int j) {
              ReferenceAccessorOps__store(
                  self.val_frags[((i * 2) + j)], out);
            }
            void ReferenceAccessorStoreLoop__store(
                in ReferenceAccessorStoreLoop self,
                inout float2 stored,
                int i,
                int j) {
              for (int k = 0; k < 2; ++k) {
                stored[k] =
                    self.nestedTile.val_frags[((i * 2) + j)][k];
              }
            }
            [numthreads(1, 1, 1)]
            void CSMain() {
              ReferenceAccessorTile tile;
              tile.val_frags[((1 * 2) + 1)] = 73.25f;
              out[0] = tile.val_frags[((1 * 2) + 1)];
              ReferenceAccessorTile__store(tile, out, 1, 1);
              ReferenceAccessorStoreLoop nestedStore;
              nestedStore.nestedTile.val_frags[((1 * 2) + 1)] =
                  float2(11.5f, 29.75f);
              float2 stored;
              ReferenceAccessorStoreLoop__store(nestedStore, stored, 1, 1);
              nestedOut[0] = stored;
            }
        """
        ),
        "opengl": (
            """
            #version 450
            struct ReferenceAccessorTile {
              float val_frags[4];
            };
            struct ReferenceAccessorFragmentTile {
              vec2 val_frags[4];
            };
            struct ReferenceAccessorStoreLoop {
              ReferenceAccessorFragmentTile nestedTile;
            };
            layout(std430, binding = 0) buffer out_block { float out_values[]; };
            layout(std430, binding = 1) buffer nested_out_block {
              vec2 nested_out_values[];
            };
            void ReferenceAccessorOps_store_glsl_out_out_float(
                inout float value,
                int out_offset) {
              out_values[out_offset + 1] = value;
            }
            void ReferenceAccessorTile_store_glsl_out_out_float(
                ReferenceAccessorTile self,
                int i,
                int j,
                int out_offset) {
              ReferenceAccessorOps_store_glsl_out_out_float(
                  self.val_frags[((i * 2) + j)], out_offset);
            }
            void ReferenceAccessorStoreLoop_store(
                ReferenceAccessorStoreLoop self,
                inout vec2 stored,
                int i,
                int j) {
              for (int k = 0; k < 2; ++k) {
                stored[k] =
                    self.nestedTile.val_frags[((i * 2) + j)][k];
              }
            }
            void main() {
              ReferenceAccessorTile tile;
              tile.val_frags[((1 * 2) + 1)] = 73.25;
              out_values[0] = tile.val_frags[((1 * 2) + 1)];
              ReferenceAccessorTile_store_glsl_out_out_float(tile, 1, 1, 0);
              ReferenceAccessorStoreLoop nestedStore;
              nestedStore.nestedTile.val_frags[((1 * 2) + 1)] =
                  vec2(11.5, 29.75);
              vec2 stored;
              ReferenceAccessorStoreLoop_store(
                  nestedStore, stored, 1, 1);
              nested_out_values[0] = stored;
            }
        """
        ),
    }
    generated_by_target = {**defaults, **(generated_by_target or {})}
    project_dir = work_dir / "reference-accessor-project"
    artifacts = []
    for target, suffix in (("directx", ".hlsl"), ("opengl", ".glsl")):
        generated_path = (
            project_dir
            / "generated"
            / target
            / Path(module.REFERENCE_ACCESSOR_FIXTURE_NAME).with_suffix(suffix)
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(
            generated_by_target[target].strip() + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "source": module.REFERENCE_ACCESSOR_FIXTURE_NAME,
                "sourceBackend": "metal",
                "target": target,
                "path": generated_path.relative_to(project_dir).as_posix(),
                "status": "translated",
            }
        )
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "unitCount": 1,
                    "artifactCount": 2,
                    "translatedCount": 2,
                    "failedCount": 0,
                    "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
                },
                "diagnostics": [],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _write_template_member_pointer_report(
    module,
    work_dir,
    report_path,
    *,
    generated_by_target=None,
    targets=None,
):
    defaults = {
        "directx": (
            """
            struct ReducedMMATile { float value; };
            StructuredBuffer<float> src : register(t0);
            RWStructuredBuffer<float> out_ : register(u1);
            float ReducedMMAFrag__load__const_device_float_ptr(
                StructuredBuffer<float> src, int64_t src_offset, int stride) {
              return float(src[uint((src_offset + stride))]);
            }
            void ReducedMMATile__load__float(
                inout ReducedMMATile self,
                StructuredBuffer<float> src,
                int64_t src_offset,
                int index) {
              self.value = ReducedMMAFrag__load__const_device_float_ptr(
                  src, int64_t((src_offset + index)), 1);
            }
            [numthreads(1, 1, 1)]
            void CSMain(uint3 gid_dispatchThreadID : SV_DispatchThreadID) {
              uint gid = gid_dispatchThreadID.x;
              ReducedMMATile tile;
              ReducedMMATile__load__float(tile, src, int64_t(0), int(gid));
              out_[gid] = tile.value;
            }
        """
        ),
        "opengl": (
            """
            #version 450 core
            struct ReducedMMATile { float value; };
            layout(std430, binding = 0) readonly buffer srcBuffer {
              float src[];
            };
            layout(std430, binding = 1) buffer outBuffer { float out_[]; };
            void ReducedMMATile_load_float__glsl_src_src_float(
                inout ReducedMMATile self, int index, int src_offset) {
              self.value =
                  ReducedMMAFrag_load_const_device_float_ptr__glsl_src_src_float(
                      1, int((src_offset + index)));
            }
            float ReducedMMAFrag_load_const_device_float_ptr__glsl_src_src_float(
                int stride, int src_offset) {
              return float(src[(src_offset + stride)]);
            }
            void main() {
              uint gid = uint(gl_GlobalInvocationID.x);
              ReducedMMATile tile;
              ReducedMMATile_load_float__glsl_src_src_float(
                  tile, int(gid), int(0));
              out_[gid] = tile.value;
            }
        """
        ),
    }
    generated_by_target = {**defaults, **(generated_by_target or {})}
    selected_targets = tuple(targets or module.TEMPLATE_MEMBER_POINTER_TARGETS)
    project_dir = work_dir / "template-member-pointer-project"
    artifacts = []
    suffixes = {"directx": ".hlsl", "opengl": ".glsl"}
    for target in selected_targets:
        generated_path = (
            project_dir
            / "generated"
            / target
            / Path(module.TEMPLATE_MEMBER_POINTER_FIXTURE_NAME).with_suffix(
                suffixes[target]
            )
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(
            generated_by_target[target].strip() + "\n",
            encoding="utf-8",
        )
        artifacts.append(
            {
                "source": module.TEMPLATE_MEMBER_POINTER_FIXTURE_NAME,
                "sourceBackend": "metal",
                "target": target,
                "path": generated_path.relative_to(project_dir).as_posix(),
                "status": "translated",
            }
        )
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "unitCount": 1,
                    "artifactCount": len(selected_targets),
                    "translatedCount": len(selected_targets),
                    "failedCount": 0,
                    "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
                },
                "diagnostics": [],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _write_metal_roundtrip_report(
    module,
    mlx_root,
    work_dir,
    report_path,
    *,
    toolchain_unavailable=False,
):
    source_path = mlx_root / module.MLX_METAL_ROUNDTRIP_SOURCE
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "#include <metal_atomic>\n[[kernel]] void fence_wait() {}\n",
        encoding="utf-8",
    )
    generated_path = (
        work_dir / "out-metal-roundtrip" / "metal" / module.MLX_METAL_ROUNDTRIP_SOURCE
    )
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(
        "\n".join(
            (
                "#include <metal_stdlib>",
                "using namespace metal;",
                "kernel void input_coherent(device uint* input [[buffer(0)]], ",
                "    uint index [[thread_position_in_grid]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "kernel void fence_update(device uint* out [[buffer(0)]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "kernel void fence_wait(device uint* out [[buffer(0)]]) {",
                "  metal::atomic_thread_fence(metal::mem_flags::mem_device,",
                "      metal::memory_order_seq_cst, metal::thread_scope_system);",
                "}",
                "",
            )
        ),
        encoding="utf-8",
    )
    artifact_path = generated_path.relative_to(mlx_root).as_posix()
    artifact = {
        "source": module.MLX_METAL_ROUNDTRIP_SOURCE,
        "sourceBackend": "metal",
        "target": "metal",
        "path": artifact_path,
        "status": "translated",
        "provenance": {
            "pipeline": "single-file-translate",
            "intermediate": "crossgl",
        },
        "sourceHash": {
            "algorithm": "sha256",
            "value": module._sha256(source_path),
        },
        "generatedHash": {
            "algorithm": "sha256",
            "value": module._sha256(generated_path),
        },
        "sourceSizeBytes": source_path.stat().st_size,
        "generatedSizeBytes": generated_path.stat().st_size,
    }
    warning_count = 1 if toolchain_unavailable else 0
    diagnostics = []
    if toolchain_unavailable:
        diagnostics.append(
            {
                "code": "project.validate.toolchain-unavailable",
                "message": "No validation toolchain is available for target metal",
                "missingCapabilities": ["toolchain.validation"],
                "severity": "warning",
                "target": "metal",
            }
        )
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "unitCount": 1,
                    "artifactCount": 1,
                    "translatedCount": 1,
                    "failedCount": 0,
                    "diagnosticCounts": {
                        "error": 0,
                        "note": 0,
                        "warning": warning_count,
                    },
                },
                "diagnostics": diagnostics,
                "artifacts": [artifact],
                "validation": {
                    "summary": {
                        "artifactCount": 1,
                        "okCount": 1,
                        "failedCount": 0,
                    },
                    "artifacts": [
                        {
                            "path": artifact_path,
                            "status": "ok",
                            "sourceHashStatus": "ok",
                            "generatedHashStatus": "ok",
                            "sourceSizeStatus": "ok",
                            "generatedSizeStatus": "ok",
                            "sourceMapStatus": "ok",
                            "sourceRemapStatus": "ok",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_reference_accessor_fixture_translates_through_public_project_surface(
    tmp_path,
):
    from crosstl.project import translate_project

    module = _load_harness()
    project_dir = tmp_path / "reference-accessor-project"
    project_dir.mkdir()
    source_path = project_dir / module.REFERENCE_ACCESSOR_FIXTURE_NAME
    source_path.write_bytes(module.REFERENCE_ACCESSOR_FIXTURE_PATH.read_bytes())

    source = source_path.read_text(encoding="utf-8")
    assert "constexpr thread float& frag_at" in source
    assert "constexpr const thread float& frag_at" in source
    assert "constexpr thread float2& frag_at" in source
    assert "constexpr const thread float2& frag_at" in source
    assert "return val_frags[i * width + j];" in source
    assert "ReferenceAccessorOps::store(frag_at(i, j), out);" in source
    outer_declaration = "struct ReferenceAccessorStoreLoop {"
    nested_member = "ReferenceAccessorFragmentTile nestedTile;"
    nested_store = (
        "void store(thread float2& stored, const short i, const short j) const"
    )
    nested_alias = "thread const auto& accum = nestedTile.frag_at(i, j);"
    nested_read = "stored[k] = accum[k];"
    assert source.index(outer_declaration) < source.index(nested_member)
    assert source.index(nested_member) < source.index(nested_store)
    assert source.index(nested_store) < source.index(nested_alias)
    assert source.index(nested_alias) < source.index(nested_read)
    alias_declaration = "using tile_t = ReferenceAccessorTile;"
    receiver_declaration = "tile_t tile;"
    accessor_write = "tile.frag_at(1, 1) = 73.25f;"
    assert source.index(alias_declaration) < source.index(receiver_declaration)
    assert source.index(receiver_declaration) < source.index(accessor_write)
    assert "ReferenceAccessorTile tile;" not in source

    payload = translate_project(
        project_dir,
        targets=list(module.REFERENCE_ACCESSOR_TARGETS),
        output_dir="generated",
    ).to_json()

    assert payload["summary"]["translatedCount"] == 2
    assert payload["summary"]["failedCount"] == 0
    assert payload["diagnostics"] == []
    artifacts = {
        artifact["target"]: artifact
        for artifact in payload["artifacts"]
        if artifact["status"] == "translated"
    }
    assert set(artifacts) == set(module.REFERENCE_ACCESSOR_TARGETS)
    for target, artifact in artifacts.items():
        generated = (project_dir / artifact["path"]).read_text(encoding="utf-8")
        evidence = module._reference_accessor_write_evidence(
            generated,
            target=target,
        )
        assert evidence["status"] == "verified-original-storage-write"
        assert evidence["readBackFromWrittenLvalue"] is True
        assert evidence["readBackLvalue"] == evidence["storageLvalue"]
        assert evidence["valueReturningHelperUsedForWrite"] is False
        const_read = module._reference_accessor_const_read_evidence(
            generated,
            target=target,
        )
        assert const_read["status"] == "verified-original-storage-const-read"
        assert const_read["storageLvalue"].startswith("self.val_frags[")
        assert const_read["passedDirectlyToHelper"] is True
        assert const_read["accessorCallEliminated"] is True
        assert const_read["kernelPathInvoked"] is True
        nested_const_alias = module._reference_accessor_nested_const_alias_evidence(
            generated,
            target=target,
        )
        assert nested_const_alias["status"] == (
            "verified-original-nested-storage-const-alias-read"
        )
        assert nested_const_alias["storageLvalue"].startswith(
            "self.nestedTile.val_frags["
        )
        assert nested_const_alias["storageLvalue"].endswith("][k]")
        assert nested_const_alias["componentReadLowering"] == "lane-helper"
        assert nested_const_alias["aliasEliminated"] is True
        assert nested_const_alias["accessorCallEliminated"] is True
        assert nested_const_alias["readFromOriginalStorage"] is True
        assert nested_const_alias["kernelPathInvoked"] is True


def test_reference_accessor_check_records_structured_proof_and_native_validation(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        if name == "translate-reference-accessor":
            _write_reference_accessor_report(
                module,
                work_dir,
                report_dir / "reference-accessor.json",
            )
        elif name == "validate-reference-accessor-directx":
            Path(command[command.index("-Fo") + 1]).write_bytes(b"DXIL")
        elif name == "validate-reference-accessor-opengl":
            Path(command[command.index("-o") + 1]).write_bytes(b"SPIRV")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    tools = {
        "dxc": "/opt/tools/dxc",
        "glslangValidator": "/opt/tools/glslangValidator",
        "spirv-val": "/opt/tools/spirv-val",
    }
    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", tools.get)

    result = module._check_reference_accessor_lvalue_identity(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_directx_toolchain=True,
        require_opengl_toolchain=True,
    )

    config = (config_dir / "reference-accessor.toml").read_text(encoding="utf-8")
    assert 'source_roots = ["."]' in config
    assert f'include = ["{module.REFERENCE_ACCESSOR_FIXTURE_NAME}"]' in config
    assert 'targets = ["directx", "opengl"]' in config
    assert [name for name, _command in commands] == [
        "translate-reference-accessor",
        "validate-reference-accessor-directx",
        "validate-reference-accessor-opengl",
        "validate-reference-accessor-opengl-spirv",
    ]
    translate_command = commands[0][1]
    assert translate_command[:4] == [
        "python",
        "-m",
        "crosstl",
        "translate-project",
    ]
    assert Path(translate_command[4]) == work_dir / "reference-accessor-project"
    assert commands[1][1][:7] == [
        "/opt/tools/dxc",
        "-T",
        "cs_6_0",
        "-E",
        "CSMain",
        str(
            work_dir
            / "reference-accessor-project"
            / "generated"
            / "directx"
            / "reference_accessor_lvalue.hlsl"
        ),
        "-Fo",
    ]
    assert "-enable-16bit-types" not in commands[1][1]
    assert commands[2][1][0] == "/opt/tools/glslangValidator"
    assert commands[2][1][1:5] == [
        "--target-env",
        "opengl",
        "-S",
        "comp",
    ]
    assert commands[3][1][:3] == [
        "/opt/tools/spirv-val",
        "--target-env",
        "opengl4.5",
    ]

    assert result["status"] == "passed"
    assert result["proofStatus"] == "verified-original-storage-write"
    assert result["constReadProofStatus"] == ("verified-original-storage-const-read")
    assert result["nestedConstAliasProofStatus"] == (
        "verified-original-nested-storage-const-alias-read"
    )
    assert result["translationSurface"] == "crosstl translate-project"
    assert result["accessorContract"]["storageExpression"] == (
        "val_frags[i * width + j]"
    )
    assert result["accessorContract"]["nestedConstAliasRead"] == {
        "outerType": "ReferenceAccessorStoreLoop",
        "tileMember": "nestedTile",
        "fragmentReturnType": "const thread float2&",
        "enclosingMethod": "store(...) const",
        "aliasDeclaration": "thread const auto& accum = nestedTile.frag_at(i, j)",
        "readExpression": "accum[k]",
        "storageExpression": "nestedTile.val_frags[i * width + j][k]",
    }
    assert result["runtimeParityClaimed"] is False
    assert result["upstreamMlxRuntimeExecuted"] is False
    assert result["sourceSha256"] == module._sha256(
        module.REFERENCE_ACCESSOR_FIXTURE_PATH
    )
    for target, tool in (
        ("directx", "dxc"),
        ("opengl", "glslangValidator"),
    ):
        proof = result["targetProofs"][target]
        assert proof["writeEvidence"] == {
            "status": "verified-original-storage-write",
            "storageMember": "val_frags",
            "storageLvalue": "tile.val_frags[((1*2)+1)]",
            "sentinel": "73.25",
            "readBackFromSameStorage": True,
            "readBackFromWrittenLvalue": True,
            "readBackLvalue": "tile.val_frags[((1*2)+1)]",
            "valueReturningHelperUsedForWrite": False,
        }
        const_read = proof["constReadEvidence"]
        assert const_read["status"] == "verified-original-storage-const-read"
        assert const_read["storageMember"] == "val_frags"
        assert const_read["storageLvalue"].startswith("self.val_frags[")
        assert const_read["implicitReceiver"] == "self"
        assert const_read["passedDirectlyToHelper"] is True
        assert const_read["accessorCallEliminated"] is True
        assert const_read["kernelPathInvoked"] is True
        nested_const_alias = proof["nestedConstAliasEvidence"]
        assert nested_const_alias["status"] == (
            "verified-original-nested-storage-const-alias-read"
        )
        assert nested_const_alias["storageMember"] == "val_frags"
        assert nested_const_alias["storagePath"] == "self.nestedTile.val_frags"
        assert nested_const_alias["storageLvalue"] == (
            "self.nestedTile.val_frags[((i*2)+j)][k]"
        )
        assert nested_const_alias["componentReadLowering"] == "direct-index"
        assert nested_const_alias["outerReceiver"] == "self"
        assert nested_const_alias["tileMember"] == "nestedTile"
        assert nested_const_alias["fragmentType"] == "float2"
        assert nested_const_alias["aliasName"] == "accum"
        assert nested_const_alias["aliasEliminated"] is True
        assert nested_const_alias["accessorCallEliminated"] is True
        assert nested_const_alias["indexedAliasRead"] == "accum[k]"
        assert nested_const_alias["readFromOriginalStorage"] is True
        assert nested_const_alias["kernelPathInvoked"] is True
        assert proof["nativeValidation"]["status"] == "validated"
        assert proof["nativeValidation"]["nativeCompiler"] == tool
        assert (mlx_root / proof["artifact"]).is_file()
    directx_validation = result["targetProofs"]["directx"]["nativeValidation"]
    assert directx_validation["profile"] == "cs_6_0"
    assert "compilerArguments" not in directx_validation
    assert "minimumShaderModel" not in directx_validation


def test_reference_accessor_directx_enables_native_16bit_hlsl(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    log_dir = work_dir / "logs"
    artifact_path = work_dir / "generated" / "reference-accessor.hlsl"
    artifact_path.parent.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    artifact_path.write_text("float16_t value;\n", encoding="utf-8")
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True):
        command = list(command)
        commands.append(command)
        Path(command[command.index("-Fo") + 1]).write_bytes(b"DXIL")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, command, 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    result = module._validate_reference_accessor_directx(
        mlx_root,
        work_dir,
        log_dir,
        artifact_path,
        required=True,
    )

    assert commands == [
        [
            "/tools/dxc",
            "-T",
            "cs_6_2",
            "-enable-16bit-types",
            "-E",
            module.REFERENCE_ACCESSOR_DXC_ENTRY_POINT,
            str(artifact_path),
            "-Fo",
            str(work_dir / "validation" / "reference-accessor.dxil"),
        ]
    ]
    assert result["profile"] == "cs_6_2"
    assert result["compilerArguments"] == ["-enable-16bit-types"]
    assert result["minimumShaderModel"] == "6.2"


def test_template_member_pointer_fixture_preserves_mlx_addressed_load_shape():
    module = _load_harness()
    source = module.TEMPLATE_MEMBER_POINTER_FIXTURE_PATH.read_text(encoding="utf-8")

    fragment_template = "template <typename SrcPtrType>"
    fragment_method = "static float load(SrcPtrType src, const int stride)"
    indexed_read = "return static_cast<float>(src[stride]);"
    tile_template = "template <typename U>"
    tile_method = "void load(const device U* src, const int index)"
    addressed_call = "value = ReducedMMAFrag::load(&(src[index]), 1);"
    kernel_call = "tile.load(src, int(gid));"
    assert source.index(fragment_template) < source.index(fragment_method)
    assert source.index(fragment_method) < source.index(indexed_read)
    assert source.index(tile_template, source.index(indexed_read)) < source.index(
        tile_method
    )
    assert source.index(tile_method) < source.index(addressed_call)
    assert source.index(addressed_call) < source.index(kernel_call)


def test_template_member_pointer_check_records_precise_target_evidence(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        _write_template_member_pointer_report(
            module,
            work_dir,
            report_dir / "template-member-pointer.json",
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._check_template_member_buffer_pointer(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
    )

    config = (config_dir / "template-member-pointer.toml").read_text(encoding="utf-8")
    assert 'source_roots = ["."]' in config
    assert f'include = ["{module.TEMPLATE_MEMBER_POINTER_FIXTURE_NAME}"]' in config
    assert 'targets = ["directx", "opengl"]' in config
    assert [name for name, _command in commands] == [
        "translate-template-member-pointer"
    ]
    assert commands[0][1][:4] == [
        "python",
        "-m",
        "crosstl",
        "translate-project",
    ]
    assert Path(commands[0][1][4]) == work_dir / "template-member-pointer-project"

    assert result["status"] == "passed"
    assert result["proofStatus"] == "verified-materialized-pointer-indexed-read"
    assert result["sourceContract"] == {
        "outerType": "ReducedMMATile",
        "outerMethod": "load",
        "outerTemplateParameter": "U",
        "sourcePointerType": "const device U*",
        "fragmentType": "ReducedMMAFrag",
        "fragmentMethod": "load",
        "fragmentTemplateParameter": "SrcPtrType",
        "pointerArgument": "&(src[index])",
        "indexedReadExpression": "src[stride]",
    }
    assert result["targets"] == ["directx", "opengl"]
    assert result["artifactCount"] == 2
    assert result["projectDiagnosticCount"] == 0
    assert result["generatedArtifactEvidenceOnly"] is True
    assert result["nativeToolchainValidationIncluded"] is False
    assert result["upstreamMlxRuntimeExecuted"] is False
    assert result["runtimeIntegrationIncluded"] is False
    assert result["runtimeParityClaimed"] is False
    assert result["sourceSha256"] == module._sha256(
        module.TEMPLATE_MEMBER_POINTER_FIXTURE_PATH
    )

    directx = result["targetProofs"]["directx"]["structuralEvidence"]
    assert directx["status"] == "verified-materialized-pointer-indexed-read"
    assert directx["materializedHelper"] == (
        "ReducedMMAFrag__load__const_device_float_ptr"
    )
    assert directx["materializedOuterHelper"] == "ReducedMMATile__load__float"
    assert directx["sourceView"] == {
        "representation": "structured-buffer-plus-offset",
        "resourceName": "src",
        "parameterName": "src",
        "parameterType": "StructuredBuffer<float>",
        "offsetParameter": "src_offset",
        "scalarizedPointerParameter": False,
    }
    assert directx["sourceIndexExpression"] == "src_offset+index"
    assert directx["indexedReadExpression"] == "src[uint((src_offset+stride))]"
    assert directx["indexedReadIndex"] == "uint((src_offset+stride))"
    assert directx["scalarizedPointerParameter"] is False
    assert directx["unresolvedSourceCallRetained"] is False
    assert directx["sourceIndexPreserved"] is True
    assert directx["kernelPathInvoked"] is True

    opengl = result["targetProofs"]["opengl"]["structuralEvidence"]
    assert opengl["status"] == "verified-materialized-pointer-indexed-read"
    assert opengl["materializedHelper"] == (
        "ReducedMMAFrag_load_const_device_float_ptr__glsl_src_src_float"
    )
    assert opengl["materializedOuterHelper"] == (
        "ReducedMMATile_load_float__glsl_src_src_float"
    )
    assert opengl["sourceView"] == {
        "representation": "global-storage-buffer-plus-offset",
        "resourceName": "src",
        "parameterName": "src_offset",
        "parameterType": "int",
        "offsetParameter": "src_offset",
        "scalarizedPointerParameter": False,
    }
    assert opengl["sourceIndexExpression"] == "src_offset+index"
    assert opengl["indexedReadExpression"] == "src[(src_offset+stride)]"
    assert opengl["indexedReadIndex"] == "(src_offset+stride)"
    assert opengl["sourceViewParameterRetained"] is True
    assert opengl["indexedReadFromSourceView"] is True
    assert opengl["scalarizedPointerParameter"] is False
    assert opengl["unresolvedSourceCallRetained"] is False
    assert opengl["kernelPathInvoked"] is True
    for proof in result["targetProofs"].values():
        assert (mlx_root / proof["artifact"]).is_file()
        assert proof["artifactSha256"] == module._sha256(mlx_root / proof["artifact"])


def test_template_member_pointer_check_requires_both_target_artifacts(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        _write_template_member_pointer_report(
            module,
            work_dir,
            report_dir / "template-member-pointer.json",
            targets=("directx",),
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="did not emit both clean artifacts",
    ):
        module._check_template_member_buffer_pointer(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            "python",
        )


@pytest.mark.parametrize(
    ("target", "generated", "message"),
    [
        pytest.param(
            "directx",
            """
            float stale_path() {
              return ReducedMMAFrag::load(&(src[index]), 1);
            }
            """,
            "retained an unresolved source member call",
            id="retained-source-member-call",
        ),
        pytest.param(
            "opengl",
            """
            void main() {
              ReducedMMATile tile;
              tile.load(src, int(gid));
            }
            """,
            "retained an unresolved source member call",
            id="retained-outer-member-call",
        ),
        pytest.param(
            "directx",
            """
            float ReducedMMAFrag__load__float(float src, int stride) {
              return float(src[stride]);
            }
            """,
            "scalarized float src parameter",
            id="directx-scalarized-pointer-parameter",
        ),
        pytest.param(
            "opengl",
            """
            float ReducedMMAFrag_load_float(float src, int stride) {
              return float(src[stride]);
            }
            """,
            "scalarized float src parameter",
            id="opengl-scalarized-pointer-parameter",
        ),
        pytest.param(
            "directx",
            """
            float ReducedMMAFrag__load__const_device_float_ptr(
                StructuredBuffer<float> src, int stride) {
              return float(src);
            }
            """,
            "does not perform an indexed read",
            id="directx-unindexed-source-read",
        ),
        pytest.param(
            "opengl",
            """
            struct ReducedMMATile { float value; };
            layout(std430, binding = 0) readonly buffer srcBuffer {
              float src[];
            };
            float ReducedMMAFrag_load_const_device_float_ptr__glsl_src_src_float(
                int stride, int src_offset) {
              return float(src[(src_offset + stride)]);
            }
            void ReducedMMATile_load_float__glsl_src_src_float(
                inout ReducedMMATile self, int index, int src_offset) {
              self.value =
                  ReducedMMAFrag_load_const_device_float_ptr__glsl_src_src_float(
                      1, int(src_offset));
            }
            """,
            "does not preserve index in the src_offset buffer view",
            id="opengl-lost-addressed-index",
        ),
    ],
)
def test_template_member_pointer_evidence_rejects_unresolved_or_scalarized_paths(
    target, generated, message
):
    module = _load_harness()

    with pytest.raises(module.PortingCheckError, match=message):
        module._template_member_pointer_evidence(generated, target=target)


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        pytest.param(
            """
            void CSMain() {
              ReferenceAccessorTile__frag_at(tile, 1, 1) = 73.25f;
              out[0] = tile.val_frags[3];
            }
            """,
            "value-return helper",
            id="value-return-helper",
        ),
        pytest.param(
            """
            void main() {
              float value = ReferenceAccessorTile__frag_at(tile, 1, 1);
              value = 73.25;
              // tile.val_frags[3] = 73.25;
              out_values[0] = tile.val_frags[3];
            }
            """,
            "did not write the sentinel directly",
            id="temporary-and-comment",
        ),
        pytest.param(
            """
            void main() {
              ReferenceAccessorTile tile_copy = tile;
              tile_copy.val_frags[3] = 73.25;
              out_values[0] = tile_copy.val_frags[3];
            }
            """,
            "copied receiver",
            id="copied-receiver-storage",
        ),
        pytest.param(
            """
            void main() {
              tile.val_frags[3] = 73.25;
              out_values[0] = tile.val_frags[2];
            }
            """,
            "exact val_frags lvalue",
            id="different-storage-element",
        ),
        pytest.param(
            """
            void main() {
              tile.val_frags[3] = 73.25;
              tile.val_frags[3] = 0.0;
            }
            """,
            "exact val_frags lvalue",
            id="subsequent-write-is-not-readback",
        ),
    ],
)
def test_reference_accessor_evidence_rejects_value_copy_false_positives(
    generated, message
):
    module = _load_harness()

    with pytest.raises(module.PortingCheckError, match=message):
        module._reference_accessor_write_evidence(generated, target="opengl")


@pytest.mark.parametrize(
    ("generated", "message"),
    [
        pytest.param(
            """
            void ReferenceAccessorStoreLoop__store(
                in ReferenceAccessorStoreLoop self) {
              float2 accum =
                  self.nestedTile.val_frags[((i * 2) + j)];
              out[2 + k] = accum[k];
            }
            void CSMain() {
              ReferenceAccessorStoreLoop__store(nestedStore);
            }
            """,
            "accum reference alias",
            id="alias-retained",
        ),
        pytest.param(
            """
            void ReferenceAccessorStoreLoop__store(
                in ReferenceAccessorStoreLoop self) {
              out[2 + k] = ReferenceAccessorFragmentTile__frag_at(
                  self.nestedTile, i, j)[k];
            }
            void CSMain() {
              ReferenceAccessorStoreLoop__store(nestedStore);
            }
            """,
            "frag_at helper or call",
            id="accessor-retained",
        ),
        pytest.param(
            """
            void ReferenceAccessorStoreLoop__store(
                in ReferenceAccessorStoreLoop self) {
              ReferenceAccessorFragmentTile tileCopy = self.nestedTile;
              out[2 + k] = tileCopy.val_frags[((i * 2) + j)][k];
            }
            void CSMain() {
              ReferenceAccessorStoreLoop__store(nestedStore);
            }
            """,
            "self.nestedTile.val_frags storage indexed by k",
            id="copied-nested-tile",
        ),
    ],
)
def test_nested_const_alias_evidence_rejects_unlowered_or_copied_reads(
    generated, message
):
    module = _load_harness()

    with pytest.raises(module.PortingCheckError, match=message):
        module._reference_accessor_nested_const_alias_evidence(
            generated,
            target="directx",
        )


def test_mlx_workflow_accounts_nested_const_alias_native_validation():
    workflow = MLX_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "os: [ubuntu-latest, windows-latest, macOS-latest]" in workflow
    assert '"nestedConstAliasProofStatus"' in workflow
    assert 'proof["nestedConstAliasEvidence"]' in workflow
    assert "verified-original-nested-storage-const-alias-read" in workflow
    assert "nested const-alias evidence is incomplete" in workflow
    assert '"Linux": "opengl"' in workflow
    assert '"Windows": "directx"' in workflow
    assert 'expected_directx_required = os.environ["RUNNER_OS"] == "Windows"' in (
        workflow
    )
    assert 'expected_opengl_required = os.environ["RUNNER_OS"] == "Linux"' in workflow
    assert '"directx": "dxc"' in workflow
    assert '"opengl": "glslangValidator"' in workflow
    assert 'native_validation["spirvValidator"] != "spirv-val"' in workflow
    assert 'native_validation["required"] is not expected_required' in workflow
    assert 'reference_accessor["runtimeParityClaimed"] is not False' in workflow
    assert 'reference_accessor["upstreamMlxRuntimeExecuted"] is not False' in workflow


def test_metal_roundtrip_validates_generated_artifact_natively(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        if name == "translate-metal-roundtrip":
            _write_metal_roundtrip_report(
                module,
                mlx_root,
                work_dir,
                report_dir / "metal-roundtrip.json",
            )
        elif name == "validate-metal-roundtrip-native":
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"AIR")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module,
        "_probe_native_metal_toolchain",
        lambda *args: {
            "status": "available",
            "platform": "darwin",
            "xcrun": "/usr/bin/xcrun",
            "reason": None,
        },
    )

    result = module._check_metal_roundtrip(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_metal_toolchain=True,
    )

    config = (config_dir / "metal-roundtrip.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_METAL_ROUNDTRIP_SOURCE}"]' in config
    assert 'targets = ["metal"]' in config
    assert result["roundTripStages"] == ["metal", "crossgl", "metal"]
    assert result["artifactValidationStatus"] == "validated"
    assert result["fenceContract"] == {
        "memoryFlags": ["mem_device"],
        "memoryOrder": "memory_order_seq_cst",
        "threadScope": "thread_scope_system",
        "occurrences": module.MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
        "preserved": True,
    }
    assert result["semanticReadinessStatus"] == "blocked"
    assert result["semanticTrackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1660"
    ]
    assert result["runtimeParityClaimed"] is False
    assert result["nativeMetalValidation"]["status"] == "validated"
    assert result["nativeMetalValidation"]["required"] is True
    assert result["nativeMetalValidation"]["artifactCompiled"] is True
    assert [name for name, _command in commands] == [
        "translate-metal-roundtrip",
        "validate-metal-roundtrip-native",
    ]
    assert "--validate" in commands[0][1]
    native_command = commands[1][1]
    assert native_command[:5] == [
        "/usr/bin/xcrun",
        "-sdk",
        "macosx",
        "metal",
        "-c",
    ]
    assert Path(native_command[5]) == mlx_root / result["artifact"]
    assert (mlx_root / result["nativeMetalValidation"]["compiledArtifact"]).is_file()


def test_metal_roundtrip_allows_unavailable_toolchain_when_not_required(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        if name == "translate-metal-roundtrip":
            _write_metal_roundtrip_report(
                module,
                mlx_root,
                work_dir,
                report_dir / "metal-roundtrip.json",
                toolchain_unavailable=True,
            )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module,
        "_probe_native_metal_toolchain",
        lambda *args: {
            "status": "toolchain-unavailable",
            "platform": "linux",
            "reason": "native Metal validation requires macOS",
        },
    )

    result = module._check_metal_roundtrip(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_metal_toolchain=False,
    )

    assert result["diagnosticCounts"] == {"error": 0, "note": 0, "warning": 1}
    assert result["nativeMetalValidation"]["status"] == "toolchain-unavailable"
    assert result["nativeMetalValidation"]["required"] is False
    assert result["nativeMetalValidation"]["artifactCompiled"] is False


def test_runtime_readiness_uses_runtime_artifact_manifest_metadata(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    report_dir = mlx_root / ".crosstl-mlx-porting" / "reports"
    report_dir.mkdir(parents=True)
    artifact_report = report_dir / "directx-readiness-artifacts.json"
    artifact_report.write_text(
        json.dumps(_translated_arange_report(module, "directx")),
        encoding="utf-8",
    )

    build_calls = []

    def fake_runtime_artifact_manifest(report_path):
        build_calls.append(Path(report_path))
        return _runtime_arange_artifact_manifest(module, "directx")

    monkeypatch.setattr(
        module,
        "build_runtime_artifact_manifest",
        fake_runtime_artifact_manifest,
    )

    result = module._plan_runtime_readiness_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name="directx-runtime-readiness",
        artifact_report=artifact_report,
        targets=("directx",),
        required_native_runtime_targets=(),
    )

    assert build_calls == [artifact_report]
    assert result["status"] == "planned"
    assert result["trackedRuntimeIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1388",
        "https://github.com/CrossGL/crosstl/issues/1471",
    ]
    assert result["testCount"] == 1
    assert result["diagnosticCounts"] == {"error": 0, "note": 0, "warning": 0}
    assert result["metadataGapCodes"] == []
    assert result["planBlockerCodes"] == []
    assert result["runtimeArtifactSummary"]["resourceBindingCount"] == 3
    assert (mlx_root / result["fixtureMetadata"]).is_file()
    assert (mlx_root / result["runtimeArtifactManifest"]).is_file()
    assert (mlx_root / result["runtimeTestManifest"]).is_file()
    assert (mlx_root / result["runtimeTestPlan"]).is_file()
    assert result["runtimeFixtureExecutionIncluded"] is True
    execution = result["runtimeFixtureExecution"]
    assert execution["status"] == "passed"
    assert execution["summary"]["fixtureCount"] == 1
    assert execution["summary"]["passedCount"] == 1
    assert execution["summary"]["failedCount"] == 0
    assert execution["projectRunnerSummary"]["skippedCount"] == 1
    assert (mlx_root / execution["fixtureMetadata"]).is_file()
    assert (mlx_root / execution["runtimeTestManifest"]).is_file()
    assert (mlx_root / execution["projectTestRunnerPlan"]).is_file()
    assert (mlx_root / execution["projectTestRunnerReport"]).is_file()
    native_execution = result["nativeRuntimeExecution"]
    assert result["nativeRuntimeExecutionIncluded"] is True
    assert native_execution["status"] == "blocked-by-runtime-driver"
    assert native_execution["summary"]["fixtureCount"] == 1
    assert native_execution["summary"]["unavailableCount"] == 1
    assert (mlx_root / native_execution["fixtureMetadata"]).is_file()
    assert (mlx_root / native_execution["runtimeTestManifest"]).is_file()
    assert (mlx_root / native_execution["projectTestRunnerPlan"]).is_file()
    assert (mlx_root / native_execution["projectTestRunnerReport"]).is_file()

    manifest = json.loads((mlx_root / result["runtimeTestManifest"]).read_text())
    assert manifest["success"] is True
    assert manifest["summary"]["testsByTarget"] == {"directx": 1}
    assert manifest["metadata"]["trackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1388",
        "https://github.com/CrossGL/crosstl/issues/1471",
    ]
    assert manifest["tests"][0]["selector"] == {
        "source": module.MLX_ARANGE_SOURCE,
        "target": "directx",
    }
    assert manifest["tests"][0]["entryPoint"] == "CSMain"

    plan = json.loads((mlx_root / result["runtimeTestPlan"]).read_text())
    assert plan["testCases"][0]["runtimeExecution"]["dispatch"]["entryPoint"] == (
        "CSMain"
    )

    runner_report = json.loads(
        (mlx_root / execution["projectTestRunnerReport"]).read_text()
    )
    runtime_result = runner_report["runtimeTestReport"]["results"][0]
    assert runner_report["success"] is True
    assert runtime_result["status"] == "passed"
    assert (
        runtime_result["executor"]["details"]["runtimeParityAdapter"]["runtimeAdapter"]
        == "mlx-arange-reference-runtime"
    )


def test_runtime_fixture_execution_metadata_uses_toolchain_free_adapters():
    module = _load_harness()

    metadata = module._runtime_fixture_execution_metadata(
        ("directx", "opengl", "vulkan")
    )

    assert metadata["metadata"]["runtimeFixtureExecutionIncluded"] is True
    assert {adapter["id"] for adapter in metadata["adapters"]} == {
        "mlx-arange-reference-directx",
        "mlx-arange-reference-opengl",
        "mlx-arange-reference-vulkan",
    }
    assert all(
        adapter["platformRequirements"]["requiredTools"] == []
        for adapter in metadata["adapters"]
    )
    assert all("target" not in adapter for adapter in metadata["adapters"])
    assert {fixture["adapter"] for fixture in metadata["fixtures"]} == {
        "mlx-arange-reference-directx",
        "mlx-arange-reference-opengl",
        "mlx-arange-reference-vulkan",
    }


def test_native_runtime_execution_metadata_uses_target_executors():
    module = _load_harness()

    metadata = module._native_runtime_execution_metadata(
        ("directx", "opengl", "vulkan")
    )

    assert metadata["metadata"]["nativeRuntimeExecutionIncluded"] is True
    assert {
        (adapter["id"], adapter["executor"], adapter["adapterKind"])
        for adapter in metadata["adapters"]
    } == {
        ("mlx-arange-native-directx", "directx", "directx-native-runtime"),
        ("mlx-arange-native-opengl", "opengl", "opengl-native-runtime"),
        ("mlx-arange-native-vulkan", "vulkan", "vulkan-native-runtime"),
    }
    assert all(
        adapter["platformRequirements"]["requiredTools"] == []
        for adapter in metadata["adapters"]
    )
    assert {fixture["adapter"] for fixture in metadata["fixtures"]} == {
        "mlx-arange-native-directx",
        "mlx-arange-native-opengl",
        "mlx-arange-native-vulkan",
    }
    assert len(metadata["fixtures"]) == 5


def test_vulkan_runtime_readiness_covers_supported_numeric_variants():
    module = _load_harness()

    fixtures = module._runtime_readiness_fixtures(("vulkan",))

    assert [fixture["id"] for fixture in fixtures] == [
        "mlx-arange-vulkan-runtime-readiness",
        "mlx-arange-vulkan-int32-runtime-readiness",
        "mlx-arange-vulkan-float32-runtime-readiness",
    ]
    assert [fixture["entryPoint"] for fixture in fixtures] == [
        "arangeuint32",
        "arangeint32",
        "arangefloat32",
    ]
    assert [fixture["inputs"][0]["dtype"] for fixture in fixtures] == [
        "uint32",
        "int32",
        "float32",
    ]
    assert fixtures[0]["expectedOutputs"][0]["values"] == [300, 317, 334, 351]
    assert fixtures[1]["expectedOutputs"][0]["values"] == [-3, -1, 1, 3]
    assert fixtures[2]["expectedOutputs"][0]["values"] == [
        1.5,
        1.75,
        2.0,
        2.25,
    ]
    assert module._runtime_fixture_scalar(1.5, default=0) == 1.5


def test_expected_gaps_tracks_current_frontier_and_runtime_fixture_counts():
    module = _load_harness()
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    frontier = expected_gaps["frontier_status"]
    assert frontier["sources"] == len(module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    assert frontier["artifacts"] == len(
        module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    ) * len(frontier["targets"])
    assert frontier["status"] == "target-split-with-expected-workgroup-blockers"
    assert frontier["scope"] == "target-split-frontier"
    assert frontier["translated_artifacts"] == 16
    assert frontier["failed_artifacts"] == 6
    assert frontier["target_artifacts"] == {
        "directx": {"translated": 5, "failed": 6},
        "vulkan": {"translated": 11, "failed": 0},
    }
    assert frontier["semantic_readiness_status"] == "not-established"
    assert frontier["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert frontier["workgroup_blocked_by"] == (
        module.MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE
    )
    assert frontier["excluded_blocked_sources"] == [module.MLX_FENCE_SOURCE]
    assert frontier["runtime_integration_included"] is False
    assert frontier["runtime_parity_claimed"] is False

    fence = expected_gaps["fence_contract_status"]
    assert fence["status"] == "blocked-as-expected"
    assert fence["source"] == module.MLX_FENCE_SOURCE
    assert fence["targets"] == list(module.MLX_FENCE_TARGET_CONTRACTS)
    assert fence["artifact_records"] == 3
    assert fence["translated_artifacts"] == 0
    assert fence["failed_artifacts"] == 3
    assert fence["emitted_artifacts"] == 0
    assert fence["requested_contract"] == {
        "memory_flags": ["mem_device"],
        "memory_order": "memory_order_seq_cst",
        "thread_scope": "thread_scope_system",
    }
    assert fence["diagnostics"] == {
        target: {
            "code": contract["diagnosticCode"],
            "missing_capability": contract["missingCapability"],
        }
        for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items()
    }
    assert fence["blocked_by"] == list(module.FENCE_CONTRACT_TRACKED_ISSUES)
    assert fence["runtime_parity_claimed"] is False

    arg_reduce = expected_gaps["arg_reduce_status"]
    assert arg_reduce["status"] == "target-dependent"
    assert arg_reduce["source"] == module.MLX_ARG_REDUCE_SOURCE
    assert arg_reduce["artifact_records"] == len(module.FULL_CORPUS_TARGETS)
    assert arg_reduce["translated_artifacts"] == 1
    assert arg_reduce["failed_artifacts"] == 2
    assert arg_reduce["translated_targets"] == ["vulkan"]
    assert arg_reduce["workgroup_blocked_targets"] == ["directx", "opengl"]
    assert arg_reduce["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert arg_reduce["workgroup_blocked_by"] == (
        module.MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE
    )

    opengl_frontier = expected_gaps["opengl_frontier_status"]
    assert opengl_frontier["status"] == (
        "toolchain-validated-with-expected-workgroup-blockers"
    )
    assert opengl_frontier["sources"] == list(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert opengl_frontier["source_count"] == len(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert opengl_frontier["artifact_count"] == len(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert opengl_frontier["translated_sources"] == list(
        module.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    )
    assert opengl_frontier["workgroup_blocked_sources"] == list(
        module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert opengl_frontier["clean_project_diagnostic_count"] == 0
    assert len(module.MLX_OPENGL_FRONTIER_SOURCES) == 8
    assert opengl_frontier["glslang_compiled_artifact_count"] == 3
    assert opengl_frontier["spirv_validated_artifact_count"] == 3
    assert opengl_frontier["glslang_target_environments"] == [
        "opengl",
        "spirv1.3",
    ]
    assert opengl_frontier["spirv_val_target_environment"] == "spv1.3"
    assert opengl_frontier["specialization_constants"] == {
        module.MLX_ROPE_SOURCE: module.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS[
            module.MLX_ROPE_SOURCE
        ]
    }
    index_range_evidence = opengl_frontier["index_range_assertion_evidence"]
    assert index_range_evidence == {
        "assertion_count": len(module.MLX_OPENGL_INDEX_RANGE_ASSERTIONS),
        "inclusive_bounds": {
            "minimum": module.MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
            "maximum": module.MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
        },
        "expressions_by_source": {
            source: list(expressions)
            for source, expressions in (
                module.MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS.items()
            )
        },
        "contract_kind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtime_enforced": False,
    }
    assert (
        sum(
            len(expressions)
            for expressions in index_range_evidence["expressions_by_source"].values()
        )
        == index_range_evidence["assertion_count"]
    )
    assert opengl_frontier["runtime_integration_included"] is False
    assert opengl_frontier["runtime_parity_claimed"] is False

    directx = expected_gaps["directx_toolchain_status"]
    assert directx["compiler"] == {"name": "dxc", "version": "v1.9.2602.24"}
    assert directx["dxc_validated_sources"] == list(
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert directx["expected_entry_point_counts"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
    )
    assert sum(directx["expected_entry_point_counts"].values()) == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
    )
    assert set(directx["directx_toolchain_gaps"]) == set(
        module.MLX_REDUCED_FRONTIER_SOURCES
    ) - set(module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES)
    directx_gaps = directx["directx_toolchain_gaps"]
    assert module.MLX_BINARY_TWO_SOURCE not in directx_gaps
    assert module.MLX_RANDOM_SOURCE not in directx_gaps
    assert module.MLX_TERNARY_SOURCE not in directx_gaps
    assert "issues/1537" in directx_gaps[module.MLX_FENCE_SOURCE]
    assert "issues/1695" not in json.dumps(directx_gaps)
    assert "issues/1518" not in json.dumps(directx_gaps)
    assert directx["native_runtime_executed"] is False
    assert directx["runtime_parity_claimed"] is False

    pointer_reinterpretation = expected_gaps["pointer_reinterpretation_status"]
    assert pointer_reinterpretation["status"] == "partial"
    assert pointer_reinterpretation["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert pointer_reinterpretation["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
        "https://github.com/CrossGL/crosstl/issues/1544",
    ]
    assert set(pointer_reinterpretation["remaining_pointer_cases_blocked_by"]) == {
        "https://github.com/CrossGL/crosstl/issues/1546"
    }

    callback = expected_gaps["captured_callback_status"]
    assert callback["status"] == "partial"
    assert callback["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert callback["remaining_callback_helpers_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1554"
    ]
    assert callback["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    compile_time_loop = expected_gaps["compile_time_loop_status"]
    assert compile_time_loop["status"] == "partial"
    assert compile_time_loop["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert "verified integral_constant" in compile_time_loop["supported_contract"]
    assert compile_time_loop["native_validation"] == {
        "directx": "generated-reduced-fixture-dxc-not-run",
        "opengl": "blocked-by-tracked-issue",
        "vulkan": "validated-reduced-fixture",
    }
    assert compile_time_loop["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    template_alias = expected_gaps["template_alias_status"]
    assert template_alias["status"] == "materialized"
    assert template_alias["target"] == "vulkan"
    assert template_alias["struct_owned_alias_template_supported_contract"] == (
        "concrete non-variadic struct-owned alias templates with declaring-owner "
        "retention, default argument substitution, dependent owner integral "
        "constants, cross-owner alias chains, and namespace-qualified or nested "
        "owner disambiguation"
    )
    assert template_alias["struct_owned_alias_template_tracked_by"] == (
        "https://github.com/CrossGL/crosstl/issues/1490"
    )
    assert template_alias["plain_helper_supported_contract"] == (
        "call-site deduction through unnamed parameters, empty braced type values, "
        "and proven lexical integral constants"
    )
    assert template_alias["callback_handoff_supported_contract"] == (
        "verified dispatch_bool helpers with only reachable lambda calls defer to "
        "structured frontend callback lowering"
    )
    assert template_alias["high_budget_report"] == {
        "unsupported_before": 111,
        "unsupported_after": 0,
        "residual_int_alias_uses": 0,
        "artifact_status": "blocked-after-materialization",
    }
    assert template_alias["remaining_helpers"] == []
    assert template_alias["resolved_helpers"] == [
        "dispatch_bool<F>",
        "const_for_loop<start, stop, step, F>",
        "tile_matmad_nax<CTile, ATile, BTile, transpose_a, transpose_b>",
    ]
    assert template_alias["resolved_value_arguments"] == [
        "BK_padded",
        "BN_padded",
    ]
    assert template_alias["wide_vector_aggregate_lowering"] == {
        "status": "validated-reduced-fixture",
        "source_types": [
            "vec<float,8>",
            "vec<float16_t,8>",
            "vec<bfloat16_t,8>",
        ],
        "representation": (
            "fixed aggregate wrapper with explicit lane storage and element-wise "
            "helpers"
        ),
        "native_validation": {
            "directx": "validated-if-toolchain-available",
            "opengl": "validated-if-toolchain-available",
            "vulkan": "validated-if-toolchain-available",
        },
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1569",
    }
    assert template_alias["post_materialization_translation"] == {
        "status": "blocked-by-tracked-issue",
        "diagnostic_code": "project.translate.unsupported-feature",
        "feature": "empty initializer on unresolved dependent static call",
        "first_unsupported_expression": "metal::bool_constant<false>{}",
        "missing_capability": "spirv.empty_initializer_type_inference",
        "issue": "https://github.com/CrossGL/crosstl/issues/1574",
        "artifact_status": "failed",
    }
    assert template_alias["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]
    assert template_alias["semantic_readiness_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1557",
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    struct_scoped_cast_alias = expected_gaps["struct_scoped_cast_alias_status"]
    assert struct_scoped_cast_alias["status"] == "partial"
    assert struct_scoped_cast_alias["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert struct_scoped_cast_alias["concrete_specializations"] == ["float", "int"]
    assert struct_scoped_cast_alias["qualifier_transport"] == "retained-in-metal-ast"
    assert struct_scoped_cast_alias["strict_crossgl_function_body_parse"] == (
        "passing-reduced-fixture"
    )
    assert struct_scoped_cast_alias["native_validation"] == {
        "directx": "required-on-windows-ci",
        "opengl": "validated-reduced-fixture",
        "vulkan": "validated-reduced-fixture",
    }
    assert struct_scoped_cast_alias["high_budget_report"] == {
        "specialization_count": 722,
        "unsupported_specialization_count": 0,
        "resolved_function": "NAXTile_float_2_2__elems",
        "prior_diagnostic_code": "project.translate.crossgl-function-body-parse-failed",
        "next_diagnostic_code": "project.translate.unsupported-feature",
        "artifact_status": "failed",
    }
    assert struct_scoped_cast_alias["remaining_contract_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1566",
    ]
    assert struct_scoped_cast_alias["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    function_local_alias = expected_gaps["function_local_alias_status"]
    assert function_local_alias["status"] == "partial"
    assert function_local_alias["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert function_local_alias["entry_count"] == 42
    assert function_local_alias["resolved_use_counts"] == {
        "declaration_types": 402,
        "casts": 87,
        "static_members": 42,
    }
    assert function_local_alias["native_validation"] == {
        "directx": "dxc-validated",
        "opengl": "validated",
        "vulkan": "validated",
    }
    assert function_local_alias["vulkan_project_warning_count"] == 0
    assert function_local_alias["single_file_vulkan_unreachable_warning_count"] == 5
    assert function_local_alias["single_file_vulkan_warnings_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1568",
    ]
    assert function_local_alias["remaining_alias_shapes_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1567",
    ]

    generic_member_call = expected_gaps["generic_member_call_status"]
    assert generic_member_call["status"] == "validated-reduced-fixture"
    assert generic_member_call["sources"] == [
        "mlx/backend/metal/kernels/fp_quantized.metal",
        "mlx/backend/metal/kernels/quantized_nax.metal",
    ]
    assert generic_member_call["targets"] == list(module.FULL_CORPUS_TARGETS)
    assert generic_member_call["native_validation"] == {
        "directx": "validated-with-glslang-hlsl",
        "opengl": "validated-with-glslang",
        "vulkan": "validated-with-spirv-tools",
    }
    assert generic_member_call["pinned_vulkan_replay"] == {
        "mlx/backend/metal/kernels/fp_quantized.metal": {
            "status": "blocked-by-tracked-issue",
            "diagnostic_code": "project.translate.metal-struct-method",
            "missing_capability": "struct.template-method",
            "first_unresolved_expression": "frag_at(i, j)",
            "issue": "https://github.com/CrossGL/crosstl/issues/1557",
            "artifact_status": "failed",
        },
        "mlx/backend/metal/kernels/quantized_nax.metal": {
            "status": "blocked-by-tracked-issue",
            "diagnostic_code": "project.translate.unsupported-feature",
            "missing_capability": "spirv.empty_initializer_type_inference",
            "first_unresolved_call": "mma",
            "issue": "https://github.com/CrossGL/crosstl/issues/1574",
            "artifact_status": "failed",
        },
    }
    assert generic_member_call["runtime_integration_included"] is False

    gemv = expected_gaps["vulkan_gemv_toolchain_status"]
    assert gemv == {
        "status": "passed",
        "source": module.MLX_GEMV_SOURCE,
        "target": "vulkan",
        "specialization_count": module.GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "entry_point_count": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "structural_validation_status": "validated",
        "available_validators": ["spirv-as", "spirv-val"],
        "target_environment": "vulkan1.1",
        "semantic_readiness_status": "no-known-codegen-fallbacks",
        "semantic_warning_count": 0,
        "semantic_warnings_by_issue": {},
        "semantic_warning_descriptions_by_issue": {},
        "semantic_blockers": [],
        "report_warning_transport_tracked_by": None,
        "runtime_integration_included": False,
    }

    runtime = expected_gaps["runtime_readiness_status"]
    assert runtime["fixture_count"] == len(
        module._runtime_readiness_fixtures(("directx", "opengl", "vulkan"))
    )

    full_corpus = expected_gaps["full_corpus_scout"]
    assert full_corpus["artifacts"] == module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT
    assert full_corpus["expected_translated_artifacts"] == (
        module.FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
    )
    assert full_corpus["expected_fence_failures"] == (
        module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    )
    assert full_corpus["expected_fence_diagnostics"] == {
        contract["diagnosticCode"]: 1
        for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
    }
    assert (
        "https://github.com/CrossGL/crosstl/issues/1537"
        in full_corpus["semantic_blocked_by"]
    )
    assert full_corpus["runtime_integration_included"] is False
    assert full_corpus["runtime_parity_claimed"] is False
    assert full_corpus["last_completed"]["snapshot_scope"] == (
        "pre-canonical-fence-contract"
    )


def test_arange_reference_runtime_resolves_fixture_resource_aliases():
    module = _load_harness()
    runtime = module.MlxArangeReferenceRuntime("vulkan")
    state = SimpleNamespace(
        resource_values={
            "startUniform": [-3],
            "stepUniform": [2],
            "out_": None,
        },
        plan=SimpleNamespace(
            resource_bindings=[
                SimpleNamespace(
                    source="input",
                    value=SimpleNamespace(name="start", values=[-3]),
                ),
                SimpleNamespace(
                    source="input",
                    value=SimpleNamespace(name="step", values=[2]),
                ),
                SimpleNamespace(
                    source="expectedOutput",
                    value=SimpleNamespace(name="out", values=None),
                ),
            ]
        ),
    )

    prepared = runtime.prepare_buffers(state)

    assert prepared["start"] == [-3]
    assert prepared["step"] == [2]
    assert "out" not in prepared


def test_runtime_report_target_selection_returns_every_fixture_result():
    module = _load_harness()
    runtime_report = {
        "results": [
            {"status": "passed", "artifact": {"target": "vulkan"}},
            {
                "status": "passed",
                "fixture": {"selector": {"target": "vulkan"}},
            },
            {"status": "unavailable", "artifact": {"target": "directx"}},
        ]
    }

    assert len(module._runtime_report_results_for_target(runtime_report, "vulkan")) == 2
    assert (
        len(module._runtime_report_results_for_target(runtime_report, "directx")) == 1
    )


@pytest.mark.parametrize("target", ("opengl", "vulkan"))
def test_required_native_runtime_rejects_any_failed_numeric_variant(target):
    module = _load_harness()
    runtime_report = {
        "results": [
            {
                "status": status,
                "fixture": {"selector": {"target": target}},
            }
            for status in ("passed", "passed", "runtime-failed")
        ]
    }

    with pytest.raises(
        module.PortingCheckError,
        match="every MLX arange fixture",
    ):
        module._require_native_runtime_results(runtime_report, target)


@pytest.mark.parametrize(
    ("target", "entry_point", "dtype"),
    (
        ("directx", "CSMain", "uint8"),
        ("opengl", "main", "uint32"),
        ("vulkan", "arangeuint32", "uint32"),
    ),
)
def test_runtime_readiness_selects_entry_point_independently(
    target, entry_point, dtype
):
    module = _load_harness()

    fixture = module._runtime_readiness_fixture(target)

    assert fixture["selector"] == {
        "source": module.MLX_ARANGE_SOURCE,
        "target": target,
    }
    assert fixture["entryPoint"] == entry_point
    assert fixture["inputs"][0]["dtype"] == dtype
    assert fixture["runtimeAdapter"]["dispatch"] == {
        "globalSize": [4, 1, 1],
    }
    assert "https://github.com/CrossGL/crosstl/issues/1394" not in (
        module.RUNTIME_READINESS_TRACKED_ISSUES
    )
    assert "https://github.com/CrossGL/crosstl/issues/1394" in (
        module.RESOLVED_FRONTIER_ISSUES
    )


def test_static_constant_materialization_issue_is_active():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1491"

    assert issue not in module.RESOLVED_FRONTIER_ISSUES
    assert issue in module.FULL_CORPUS_TRACKED_ISSUES


def test_arg_reduce_remains_in_frontiers_but_fail_closes_native_targets():
    module = _load_harness()

    assert module.MLX_ARG_REDUCE_SOURCE in (module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    assert module.MLX_ARG_REDUCE_SOURCE in (module.MLX_OPENGL_FRONTIER_SOURCES)
    assert module.MLX_ARG_REDUCE_SOURCE in (
        module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert module.MLX_ARG_REDUCE_SOURCE not in (
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert module.MLX_ARG_REDUCE_SOURCE in (
        module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert module.MLX_OPENGL_FRONTIER_SOURCES == (
        module.MLX_ARG_REDUCE_SOURCE,
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_LOGSUMEXP_SOURCE,
        module.MLX_RMS_NORM_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        module.MLX_SOFTMAX_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )
    assert module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES == (
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )
    assert module.MLX_REDUCED_FRONTIER_SOURCES == tuple(
        sorted(
            (
                *module.MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES,
                *module.MLX_BLOCKED_REDUCED_FRONTIER_SOURCES,
            )
        )
    )
    assert "https://github.com/CrossGL/crosstl/issues/1551" in (
        module.RESOLVED_FRONTIER_ISSUES
    )


def test_binary_two_advances_into_opengl_toolchain_frontier_without_source_growth():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1661"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert module.MLX_BINARY_TWO_SOURCE in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    assert module.MLX_BINARY_TWO_SOURCE in (
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert len(module.MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES) == 11
    assert len(module.MLX_REDUCED_FRONTIER_SOURCES) == 12
    assert (
        module.MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES.count(
            module.MLX_BINARY_TWO_SOURCE
        )
        == 1
    )
    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert issue in expected_gaps["resolved_issues"]
    assert issue not in expected_gaps["tracked_issues"]
    binary_two = expected_gaps["opengl_frontier_status"][
        "binary_two_fixed_array_resource_status"
    ]
    assert binary_two == {
        "status": "validated",
        "resolved_issue": issue,
        "resolved_by_commit": "db593d19bfa04b7a58ef9f4b6b224842d802173f",
    }


def test_opengl_toolchain_frontier_matches_pinned_validator_inventory():
    module = _load_harness()
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    expected_frontier_sources = (
        module.MLX_ARG_REDUCE_SOURCE,
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_LOGSUMEXP_SOURCE,
        module.MLX_RMS_NORM_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        module.MLX_SOFTMAX_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )
    expected_sources = (
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )

    assert module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES == expected_sources
    assert module.MLX_OPENGL_FRONTIER_SOURCES == expected_frontier_sources
    assert len(expected_frontier_sources) == 8
    assert len(expected_sources) == 3
    status = expected_gaps["opengl_frontier_status"]
    assert status["sources"] == list(expected_frontier_sources)
    assert status["source_count"] == 8
    assert status["artifact_count"] == 8
    assert status["translated_sources"] == list(expected_sources)
    assert status["glslang_compiled_artifact_count"] == 3
    assert status["spirv_validated_artifact_count"] == 3
    assert status["runtime_integration_included"] is False
    assert status["runtime_parity_claimed"] is False

    workflow = MLX_WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES" in workflow
    assert "if opengl_toolchain_frontier_count != 3:" in workflow
    assert "expected 3 OpenGL toolchain frontier sources" in workflow


def test_fence_is_blocked_outside_non_fence_and_directx_toolchain_frontiers():
    module = _load_harness()

    assert module.MLX_FENCE_SOURCE not in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    assert module.MLX_FENCE_SOURCE not in module.MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES
    assert module.MLX_FENCE_SOURCE not in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    assert module.MLX_BLOCKED_REDUCED_FRONTIER_SOURCES == (module.MLX_FENCE_SOURCE,)
    assert module.MLX_FENCE_SOURCE in module.MLX_REDUCED_FRONTIER_SOURCES
    assert module.FENCE_CONTRACT_TRACKED_ISSUES == (
        "https://github.com/CrossGL/crosstl/issues/1537",
    )


def test_float_subgroup_xor_issue_is_resolved_for_mlx_gemv():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1498"

    assert module.VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES == ()
    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES


def test_nested_return_inlining_issue_is_resolved_for_mlx_gemv():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1561"

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES


def test_generic_member_call_issue_is_resolved_for_pinned_quantized_kernels():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1555"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert issue in expected_gaps["resolved_issues"]
    assert issue not in expected_gaps["tracked_issues"]
    assert issue not in expected_gaps["full_corpus_scout"]["translation_blocked_by"]


def test_empty_initializer_issue_is_resolved_for_pinned_quantized_nax():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1573"
    next_issue = "https://github.com/CrossGL/crosstl/issues/1574"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert issue in module.RESOLVED_FRONTIER_ISSUES
    assert issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert issue in expected_gaps["resolved_issues"]
    assert issue not in expected_gaps["tracked_issues"]
    replay = expected_gaps["generic_member_call_status"]["pinned_vulkan_replay"]
    assert replay["mlx/backend/metal/kernels/quantized_nax.metal"]["issue"] == (
        next_issue
    )


def test_scaled_dot_product_attention_is_vulkan_only_until_runtime_dispatch_maps():
    module = _load_harness()
    source = module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    resolved_issue = "https://github.com/CrossGL/crosstl/issues/1535"
    static_constant_issue = "https://github.com/CrossGL/crosstl/issues/1491"
    function_constant_issue = "https://github.com/CrossGL/crosstl/issues/1538"

    assert source in module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    assert source not in (
        module.MLX_ARANGE_SOURCE,
        module.MLX_ARG_REDUCE_SOURCE,
        module.MLX_GEMV_SOURCE,
    )
    assert resolved_issue in module.RESOLVED_FRONTIER_ISSUES
    assert resolved_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert source in module.MLX_OPENGL_FRONTIER_SOURCES
    assert source in module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    assert source in module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    assert source not in module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    assert source not in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    assert module.OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES == (
        function_constant_issue,
    )
    assert static_constant_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert function_constant_issue in module.FULL_CORPUS_TRACKED_ISSUES


def test_scaled_attention_local_alias_evidence_requires_complete_entries(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    vulkan_path = Path("out/vulkan/scaled_dot_product_attention.spvasm")
    (mlx_root / vulkan_path).parent.mkdir(parents=True, exist_ok=True)

    vulkan = "\n".join(
        f'  OpEntryPoint GLCompute %{index + 1} "sdpa_{index}"' for index in range(42)
    )
    (mlx_root / vulkan_path).write_text(vulkan, encoding="utf-8")
    payload = {
        "artifacts": [
            {
                "source": module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
                "target": "vulkan",
                "path": vulkan_path.as_posix(),
                "status": "translated",
            }
        ]
    }

    evidence = module._scaled_attention_local_alias_evidence(mlx_root, payload)

    assert evidence["target"] == "vulkan"
    assert evidence["entryCount"] == 42
    assert evidence["resolvedDeclarationTypeCount"] == 402
    assert evidence["resolvedCastCount"] == 87
    assert evidence["resolvedStaticMemberCount"] == 42
    assert evidence["vulkanProjectWarningCount"] == 0


def _fence_contract_report(module, mlx_root, work_dir):
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    diagnostics = []
    artifacts = []
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        message = module._atomic_fence_expected_message(contract)
        artifact_path = (
            work_dir
            / "out-fence-contract"
            / target
            / Path(module.MLX_FENCE_SOURCE).with_suffix(extensions[target])
        ).relative_to(mlx_root)
        diagnostics.append(
            {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": message,
                "location": {"file": module.MLX_FENCE_SOURCE},
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            }
        )
        artifacts.append(
            {
                "source": module.MLX_FENCE_SOURCE,
                "sourceBackend": "metal",
                "target": target,
                "path": artifact_path.as_posix(),
                "status": "failed",
                "error": message,
            }
        )
    target_count = len(module.MLX_FENCE_TARGET_CONTRACTS)
    return {
        "summary": {
            "unitCount": 1,
            "artifactCount": target_count,
            "translatedCount": 0,
            "failedCount": target_count,
            "diagnosticCounts": {"error": target_count, "note": 0, "warning": 0},
            "diagnosticsByCode": {
                contract["diagnosticCode"]: 1
                for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "missingCapabilityCounts": {
                contract["missingCapability"]: 1
                for contract in module.MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "artifactsByTarget": {
                target: {
                    "artifactCount": 1,
                    "translatedCount": 0,
                    "failedCount": 1,
                }
                for target in module.MLX_FENCE_TARGET_CONTRACTS
            },
        },
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    }


def _prepare_fence_contract_check(module, tmp_path, monkeypatch, mutate=None):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        payload = _fence_contract_report(module, mlx_root, work_dir)
        if mutate is not None:
            mutate(payload, mlx_root)
        (report_dir / "fence-contract.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        commands.append((name, list(command), check))
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    return mlx_root, work_dir, config_dir, report_dir, log_dir, commands


def test_atomic_fence_contract_records_exact_target_failures(tmp_path, monkeypatch):
    module = _load_harness()
    (
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        commands,
    ) = _prepare_fence_contract_check(module, tmp_path, monkeypatch)

    result = module._check_atomic_fence_contract(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
    )

    assert result["status"] == "blocked-as-expected"
    assert result["source"] == module.MLX_FENCE_SOURCE
    assert result["targets"] == ["directx", "opengl", "vulkan"]
    assert result["artifactRecordCount"] == 3
    assert result["failedArtifactCount"] == 3
    assert result["emittedArtifactCount"] == 0
    assert result["requestedContract"] == {
        "memoryFlags": ["mem_device"],
        "memoryOrder": "memory_order_seq_cst",
        "threadScope": "thread_scope_system",
    }
    assert result["semanticTrackedIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1537"
    ]
    assert result["runtimeParityClaimed"] is False
    assert set(result["targetContracts"]) == {"directx", "opengl", "vulkan"}
    for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items():
        target_result = result["targetContracts"][target]
        assert target_result["diagnosticCode"] == contract["diagnosticCode"]
        assert target_result["missingCapability"] == contract["missingCapability"]
        assert target_result["artifactStatus"] == "failed"
        assert target_result["artifactEmitted"] is False

    config = (config_dir / "fence-contract.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_FENCE_SOURCE}"]' in config
    assert 'targets = ["directx", "opengl", "vulkan"]' in config
    assert commands == [
        (
            "translate-fence-contract",
            [
                "python",
                "-m",
                "crosstl",
                "translate-project",
                str(mlx_root),
                "--config",
                str(config_dir / "fence-contract.toml"),
                "--report",
                str(report_dir / "fence-contract.json"),
            ],
            False,
        )
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload, _root: payload["diagnostics"][0].__setitem__(
                "code", "project.translate.failed"
            ),
            "structured diagnostic changed",
        ),
        (
            lambda payload, _root: payload["diagnostics"][1].__setitem__(
                "missingCapabilities", ["batch.translation"]
            ),
            "structured diagnostic changed",
        ),
        (
            lambda payload, _root: payload["diagnostics"][2].__setitem__(
                "message",
                payload["diagnostics"][2]["message"].replace(
                    "memory_order_seq_cst", "memory_order_acq_rel"
                ),
            ),
            "structured diagnostic changed",
        ),
    ],
)
def test_atomic_fence_contract_rejects_diagnostic_drift(
    tmp_path, monkeypatch, mutation, error
):
    module = _load_harness()
    check = _prepare_fence_contract_check(module, tmp_path, monkeypatch, mutation)

    with pytest.raises(module.PortingCheckError, match=error):
        module._check_atomic_fence_contract(*check[:5], "python")


def test_atomic_fence_contract_rejects_emitted_target_artifact(tmp_path, monkeypatch):
    module = _load_harness()

    def emit_directx_artifact(payload, mlx_root):
        artifact_path = mlx_root / payload["artifacts"][0]["path"]
        artifact_path.parent.mkdir(parents=True)
        artifact_path.write_text("unexpected", encoding="utf-8")

    check = _prepare_fence_contract_check(
        module,
        tmp_path,
        monkeypatch,
        emit_directx_artifact,
    )

    with pytest.raises(module.PortingCheckError, match="unexpectedly emitted"):
        module._check_atomic_fence_contract(*check[:5], "python")


def test_opengl_codegen_fixes_advance_native_validation_frontier():
    module = _load_harness()
    resolved_issues = {
        "https://github.com/CrossGL/crosstl/issues/1535",
        "https://github.com/CrossGL/crosstl/issues/1500",
        "https://github.com/CrossGL/crosstl/issues/1502",
        "https://github.com/CrossGL/crosstl/issues/1503",
        "https://github.com/CrossGL/crosstl/issues/1504",
        "https://github.com/CrossGL/crosstl/issues/1489",
    }

    assert resolved_issues.issubset(module.RESOLVED_FRONTIER_ISSUES)
    assert resolved_issues.isdisjoint(module.FULL_CORPUS_TRACKED_ISSUES)
    assert resolved_issues.isdisjoint(module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES)
    assert module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES == ()


def _prepare_arange_opengl_check(module, tmp_path, generated):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    generated_path = work_dir / "out" / "opengl" / "arange" / "arangeuint32.glsl"
    for path in (config_dir, report_dir, log_dir, generated_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    report = {
        "summary": {"translatedCount": 1, "failedCount": 0},
        "artifacts": [
            {
                "source": module.MLX_ARANGE_SOURCE,
                "target": "opengl",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
                "entryPoint": {
                    "source": "arangeuint32",
                    "target": "main",
                    "stage": "compute",
                },
            }
        ],
    }
    (report_dir / "arange-opengl.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def _arange_opengl_frontier_source():
    return """
    #version 450 core
    struct complex64_t { float real; float imag; };
    layout(std430, binding = 2) buffer out_Buffer { uint out_[]; };
    layout(std140, binding = 0) uniform arangeuint32_start_Args {
        uint start;
    };
    layout(std140, binding = 1) uniform arangeuint32_step_Args {
        uint step;
    };
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {
        uint index = uint(gl_GlobalInvocationID.x);
        out_[index] = (start + (index * step));
    }
    """


def _frontier_unit(source, index):
    return {
        "id": source,
        "path": source,
        "sourceBackend": "metal",
        "sourceHash": {"algorithm": "sha256", "value": f"source-{index}"},
        "sourceSizeBytes": 1000 + index,
    }


def _write_clean_frontier_report(
    module,
    mlx_root,
    output_dir,
    report_path,
    *,
    target,
    sources,
    toolchain_runs=(),
    index_range_assertions=(),
    reverse=False,
):
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    units = [_frontier_unit(source, index) for index, source in enumerate(sources)]
    units_by_source = {unit["path"]: unit for unit in units}
    artifacts = []
    for source in sources:
        generated_path = (output_dir / target / Path(source)).with_suffix(
            extensions[target]
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        expected_constants = (
            module.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS.get(source, {})
            if target == "opengl"
            else {}
        )
        declarations = [
            "layout(constant_id = {}) const {} {} = {};".format(
                constant_id,
                "int" if name == "blocks" else "bool",
                name,
                "1" if name == "blocks" else "false",
            )
            for name, constant_id in expected_constants.items()
        ]
        generated_path.write_text(
            "\n".join(["#version 450 core", *declarations, "void main() {}", ""]),
            encoding="utf-8",
        )
        unit = units_by_source[source]
        artifact = {
            "source": source,
            "sourceHash": unit["sourceHash"],
            "sourceSizeBytes": unit["sourceSizeBytes"],
            "target": target,
            "path": generated_path.relative_to(mlx_root).as_posix(),
            "status": "translated",
        }
        if expected_constants:
            artifact["specializationConstants"] = [
                {"name": name, "id": constant_id}
                for name, constant_id in expected_constants.items()
            ]
            artifact["specializationMaterialization"] = {
                "mode": "deferred",
                "targetSupportsDeferredSpecialization": True,
            }
        artifacts.append(artifact)
    if reverse:
        units.reverse()
        artifacts.reverse()
    source_count = len(sources)
    report = {
        "project": {
            "includePatterns": list(reversed(sources)) if reverse else list(sources),
            "targets": [target],
            "workgroupSizeRules": {},
            "workgroupSizeRuleCount": 0,
            "indexRangeAssertions": [
                dict(assertion) for assertion in index_range_assertions
            ],
            "indexRangeAssertionCount": len(index_range_assertions),
        },
        "summary": {
            "unitCount": source_count,
            "artifactCount": source_count,
            "translatedCount": source_count,
            "failedCount": 0,
            "diagnosticCounts": {"note": 0, "warning": 0, "error": 0},
            "artifactsByTarget": {
                target: {
                    "artifactCount": source_count,
                    "translatedCount": source_count,
                    "failedCount": 0,
                }
            },
        },
        "units": units,
        "artifacts": artifacts,
        "diagnostics": [],
        "validation": {
            "summary": {"failedCount": 0},
            "toolchainRuns": list(toolchain_runs),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _dynamic_workgroup_report(
    module,
    mlx_root,
    output_dir,
    *,
    target,
    sources,
    validated,
    toolchain_available=True,
    reverse=False,
):
    extensions = {"directx": ".hlsl", "opengl": ".glsl"}
    units = [_frontier_unit(source, index) for index, source in enumerate(sources)]
    units_by_source = {unit["path"]: unit for unit in units}
    diagnostics = []
    artifacts = []
    for source in sources:
        entry_count = module.MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS[source]
        specialization_count = module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source][
            "specializationCount"
        ]
        entry_points = [f"entry_{index}" for index in range(entry_count)]
        specializations = [
            {"hostName": entry_point} for entry_point in entry_points
        ] + [
            {"materializedName": f"helper_{index}"}
            for index in range(specialization_count - entry_count)
        ]
        artifact_path = (output_dir / target / Path(source)).with_suffix(
            extensions[target]
        )
        unit = units_by_source[source]
        diagnostics.append(
            {
                "severity": "error",
                "code": module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE,
                "message": module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE,
                "target": target,
                "checkKind": "execution-specialization",
                "missingCapabilities": ["execution.workgroup-size-specialization"],
                "details": {
                    "sourcePath": source,
                    "executionSpecialization": {
                        "reason": "aggregate-entry-size-unproven",
                        "sourceEntryPoints": entry_points,
                    },
                },
            }
        )
        artifacts.append(
            {
                "source": source,
                "sourceHash": unit["sourceHash"],
                "sourceSizeBytes": unit["sourceSizeBytes"],
                "target": target,
                "path": artifact_path.relative_to(mlx_root).as_posix(),
                "status": "failed",
                "error": module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_MESSAGE,
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": specialization_count,
                    "specializations": specializations,
                    "unsupported": [],
                },
            }
        )
        if validated:
            diagnostics.append(
                {
                    "severity": "error",
                    "code": "project.validate.failed-artifact",
                    "target": target,
                    "details": {"sourcePath": source},
                }
            )
    if reverse:
        units.reverse()
        diagnostics.reverse()
        artifacts.reverse()
    expected_diagnostics = {module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE: len(sources)}
    if validated:
        expected_diagnostics["project.validate.failed-artifact"] = len(sources)
    source_count = len(sources)
    report = {
        "project": {
            "includePatterns": list(reversed(sources)) if reverse else list(sources),
            "targets": [target],
            "workgroupSizeRules": {},
            "workgroupSizeRuleCount": 0,
        },
        "summary": {
            "unitCount": source_count,
            "artifactCount": source_count,
            "translatedCount": 0,
            "failedCount": source_count,
            "diagnosticCounts": {
                "error": len(diagnostics),
                "note": 0,
                "warning": 0,
            },
            "diagnosticsByCode": expected_diagnostics,
            "artifactsByTarget": {
                target: {
                    "artifactCount": source_count,
                    "translatedCount": 0,
                    "failedCount": source_count,
                }
            },
        },
        "units": units,
        "artifacts": artifacts,
        "diagnostics": diagnostics,
    }
    if validated:
        tool_name = {"directx": "dxc", "opengl": "glslangValidator"}[target]
        report["validation"] = {
            "summary": {"failedCount": source_count},
            "toolchains": [
                {
                    "target": target,
                    "status": "available" if toolchain_available else "unavailable",
                    "tools": [
                        {
                            "name": tool_name,
                            "available": toolchain_available,
                            "path": (
                                f"/usr/bin/{tool_name}" if toolchain_available else None
                            ),
                        }
                    ],
                }
            ],
        }
        if not toolchain_available:
            warning = {
                "severity": "warning",
                "code": "project.validate.toolchain-unavailable",
                "message": f"No validation toolchain is available for target {target}",
                "target": target,
                "missingCapabilities": ["toolchain.validation"],
            }
            report["diagnostics"].append(warning)
            report["summary"]["diagnosticCounts"]["warning"] = 1
            report["summary"]["diagnosticsByCode"][warning["code"]] = 1
    return report


def _write_dynamic_workgroup_report(module, report_path, *args, **kwargs):
    report = _dynamic_workgroup_report(module, *args, **kwargs)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _saved_report_payloads(report_dir, *names):
    return {
        name: (report_dir / f"{name}.json").read_text(encoding="utf-8")
        for name in names
    }


def _restore_fake_report(module, mlx_root, report_dir, name, payloads):
    payload = payloads.get(name)
    if payload is not None:
        report = json.loads(payload)
        for artifact in report.get("artifacts", []):
            if artifact.get("status") != "translated":
                continue
            generated_path = mlx_root / artifact["path"]
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            expected_constants = module.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS.get(
                artifact.get("source"), {}
            )
            declarations = [
                "layout(constant_id = {}) const {} {} = {};".format(
                    constant_id,
                    "int" if constant_name == "blocks" else "bool",
                    constant_name,
                    "1" if constant_name == "blocks" else "false",
                )
                for constant_name, constant_id in expected_constants.items()
            ]
            generated_path.write_text(
                "\n".join(["#version 450 core", *declarations, "void main() {}", ""]),
                encoding="utf-8",
            )
        (report_dir / f"{name}.json").write_text(payload, encoding="utf-8")


def test_dynamic_workgroup_config_report_join_is_order_independent(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
        reverse=True,
    )

    evidence = module._require_dynamic_workgroup_blocker_report(
        mlx_root,
        output_dir,
        payload,
        target="directx",
        sources=sources,
        validated=True,
    )

    assert set(evidence) == set(sources)
    assert all(item["artifactEmitted"] is False for item in evidence.values())


def test_dynamic_workgroup_config_report_join_rejects_unproved_rule(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
    )
    payload["project"]["workgroupSizeRules"] = {sources[0]: [1, 1, 1]}
    payload["project"]["workgroupSizeRuleCount"] = 1

    with pytest.raises(
        module.PortingCheckError,
        match="gained an unproved workgroup-size rule",
    ):
        module._require_dynamic_workgroup_blocker_report(
            mlx_root,
            output_dir,
            payload,
            target="directx",
            sources=sources,
            validated=True,
        )


def test_dynamic_workgroup_config_report_join_rejects_artifact_mismatch(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
    )
    payload["artifacts"][0]["source"] = sources[1]

    with pytest.raises(
        module.PortingCheckError,
        match="artifact/source join changed",
    ):
        module._require_dynamic_workgroup_blocker_report(
            mlx_root,
            output_dir,
            payload,
            target="directx",
            sources=sources,
            validated=True,
        )


def test_dynamic_workgroup_report_rejects_entry_identity_mismatch(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
    )
    source_entries = payload["diagnostics"][0]["details"]["executionSpecialization"][
        "sourceEntryPoints"
    ]
    source_entries[0] = "different_entry"

    with pytest.raises(
        module.PortingCheckError,
        match="dynamic-workgroup materialization changed",
    ):
        module._require_dynamic_workgroup_blocker_report(
            mlx_root,
            output_dir,
            payload,
            target="directx",
            sources=sources,
            validated=True,
        )


def test_dynamic_workgroup_report_rejects_specialization_count_drift(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
    )
    materialization = payload["artifacts"][0]["templateMaterialization"]
    materialization["specializations"].pop()
    materialization["specializationCount"] -= 1

    with pytest.raises(
        module.PortingCheckError,
        match="dynamic-workgroup materialization changed",
    ):
        module._require_dynamic_workgroup_blocker_report(
            mlx_root,
            output_dir,
            payload,
            target="directx",
            sources=sources,
            validated=True,
        )


def test_dynamic_workgroup_report_rejects_unexpected_warning(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-opengl-workgroup-frontier"
    sources = module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="opengl",
        sources=sources,
        validated=False,
    )
    payload["summary"]["diagnosticCounts"]["warning"] = 1
    payload["summary"]["diagnosticsByCode"]["project.translate.test-warning"] = 1
    payload["diagnostics"].append(
        {
            "severity": "warning",
            "code": "project.translate.test-warning",
            "target": "opengl",
        }
    )

    with pytest.raises(
        module.PortingCheckError,
        match="dynamic-workgroup frontier accounting changed",
    ):
        module._require_dynamic_workgroup_blocker_report(
            mlx_root,
            output_dir,
            payload,
            target="opengl",
            sources=sources,
            validated=False,
        )


def test_dynamic_workgroup_report_accepts_explicitly_unavailable_toolchain(tmp_path):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    output_dir = mlx_root / "out-directx-workgroup-frontier"
    sources = module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    payload = _dynamic_workgroup_report(
        module,
        mlx_root,
        output_dir,
        target="directx",
        sources=sources,
        validated=True,
        toolchain_available=False,
    )

    evidence = module._require_dynamic_workgroup_blocker_report(
        mlx_root,
        output_dir,
        payload,
        target="directx",
        sources=sources,
        validated=True,
    )

    assert set(evidence) == set(sources)


def _prepare_opengl_frontier_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_clean_frontier_report(
        module,
        mlx_root,
        work_dir / "out-opengl-frontier",
        report_dir / "opengl-frontier.json",
        target="opengl",
        sources=module.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES,
        index_range_assertions=module.MLX_OPENGL_INDEX_RANGE_ASSERTIONS,
    )
    _write_dynamic_workgroup_report(
        module,
        report_dir / "opengl-workgroup-frontier.json",
        mlx_root,
        work_dir / "out-opengl-workgroup-frontier",
        target="opengl",
        sources=module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
        validated=False,
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def test_opengl_frontier_required_toolchain_compiles_and_validates_artifacts(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    payloads = _saved_report_payloads(
        paths[3], "opengl-frontier", "opengl-workgroup-frontier"
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        _restore_fake_report(module, paths[0], paths[3], name, payloads)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name.startswith("validate-") and name.endswith("-opengl"):
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    tools = {
        "glslangValidator": "/tools/glslangValidator",
        "spirv-val": "/tools/spirv-val",
    }
    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", tools.get)

    result = module._check_opengl_frontier(
        *paths,
        "python",
        require_toolchain=True,
    )

    assert result["status"] == "passed-with-expected-workgroup-blockers"
    assert result["sources"] == list(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert result["sourceCount"] == len(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert result["artifactCount"] == len(module.MLX_OPENGL_FRONTIER_SOURCES)
    assert result["translatedSources"] == list(
        module.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    )
    assert result["workgroupBlockedSources"] == list(
        module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert all(
        evidence["sourceEntryPointIdentityStatus"] == "matched-materialized-host-names"
        for evidence in result["dynamicWorkgroupDispatchEvidence"].values()
    )
    assert result["projectDiagnosticCount"] == 0
    assert result["toolchainRequired"] is True
    assert result["toolchainValidatedSources"] == list(
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert result["toolchainValidatedArtifactCount"] == len(
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert result["nativeValidationStatus"] == "validated"
    assert result["spirvValidator"] == "spirv-val"
    assert result["specializationConstants"] == {
        module.MLX_ROPE_SOURCE: module.MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS[
            module.MLX_ROPE_SOURCE
        ]
    }
    assert result["indexRangeAssertionEvidence"] == {
        "assertionCount": len(module.MLX_OPENGL_INDEX_RANGE_ASSERTIONS),
        "inclusiveBounds": {
            "minimum": module.MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
            "maximum": module.MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
        },
        "expressionsBySource": {
            source: list(expressions)
            for source, expressions in (
                module.MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS.items()
            )
        },
        "contractKind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtimeEnforced": False,
    }
    assert result["runtimeIntegrationIncluded"] is False
    assert [name for name, _command in commands] == [
        "opengl-frontier",
        "opengl-workgroup-frontier",
        "validate-binary-two-opengl",
        "validate-binary-two-opengl-spirv",
        "validate-rope-opengl",
        "validate-rope-opengl-spirv",
        "validate-ternary-opengl",
        "validate-ternary-opengl-spirv",
    ]
    assert commands[2][1][:5] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
    ]
    assert commands[3][1][:3] == [
        "/tools/spirv-val",
        "--target-env",
        "spv1.3",
    ]
    assert set(result["nativeValidationOutputs"]) == set(
        module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES
    )
    commands_by_name = dict(commands)
    for source in module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES:
        stem = Path(source).stem
        command_name = stem.replace("_", "-")
        generated_path = (
            paths[1] / "out-opengl-frontier" / "opengl" / Path(source)
        ).with_suffix(".glsl")
        output_path = paths[0] / result["nativeValidationOutputs"][source]
        assert (
            str(generated_path) in commands_by_name[f"validate-{command_name}-opengl"]
        )
        assert (
            str(output_path)
            in commands_by_name[f"validate-{command_name}-opengl-spirv"]
        )
    config = (paths[2] / "opengl-frontier.toml").read_text(encoding="utf-8")
    for source in module.MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES:
        assert source in config
    assert module.MLX_TERNARY_SOURCE in config
    assert module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE not in config
    assert 'targets = ["opengl"]' in config
    assert config.count("[[project.index_range_assertions]]") == len(
        module.MLX_OPENGL_INDEX_RANGE_ASSERTIONS
    )
    from crosstl.project import load_project_config

    parsed_config = load_project_config(paths[0], paths[2] / "opengl-frontier.toml")
    assert [
        assertion.to_json() for assertion in parsed_config.index_range_assertions
    ] == list(module.MLX_OPENGL_INDEX_RANGE_ASSERTIONS)
    blocked_config = (paths[2] / "opengl-workgroup-frontier.toml").read_text(
        encoding="utf-8"
    )
    for source in module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES:
        assert source in blocked_config
    assert "[[project.index_range_assertions]]" not in blocked_config


def test_opengl_frontier_skips_toolchain_when_not_required(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    payloads = _saved_report_payloads(
        paths[3], "opengl-frontier", "opengl-workgroup-frontier"
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append(name)
        _restore_fake_report(module, paths[0], paths[3], name, payloads)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: pytest.fail(f"tool lookup should not run for {name}"),
    )

    result = module._check_opengl_frontier(
        *paths,
        "python",
        require_toolchain=False,
    )

    assert commands == ["opengl-frontier", "opengl-workgroup-frontier"]
    assert result["toolchainRequired"] is False
    assert result["toolchainValidatedSources"] == []
    assert result["toolchainValidatedArtifactCount"] == 0
    assert result["nativeValidationStatus"] == "not-required"
    assert result["nativeValidationOutputs"] == {}


def test_opengl_frontier_requires_every_clean_artifact(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    report_path = paths[3] / "opengl-frontier.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["artifactCount"] -= 1
    report["artifacts"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payloads = _saved_report_payloads(
        paths[3], "opengl-frontier", "opengl-workgroup-frontier"
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        _restore_fake_report(module, paths[0], paths[3], name, payloads)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="every clean source artifact",
    ):
        module._check_opengl_frontier(
            *paths,
            "python",
            require_toolchain=False,
        )


def test_opengl_frontier_requires_zero_project_diagnostics(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    report_path = paths[3] / "opengl-frontier.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["diagnosticCounts"]["warning"] = 1
    report["diagnostics"] = [
        {
            "severity": "warning",
            "code": "project.translate.test-warning",
            "message": "unexpected warning",
        }
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payloads = _saved_report_payloads(
        paths[3], "opengl-frontier", "opengl-workgroup-frontier"
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        _restore_fake_report(module, paths[0], paths[3], name, payloads)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(module.PortingCheckError, match="zero project diagnostics"):
        module._check_opengl_frontier(
            *paths,
            "python",
            require_toolchain=False,
        )


@pytest.mark.parametrize("drift", ("count", "content"))
def test_opengl_frontier_requires_exact_index_range_assertion_report(
    tmp_path,
    monkeypatch,
    drift,
):
    module = _load_harness()
    paths = _prepare_opengl_frontier_check(module, tmp_path)
    report_path = paths[3] / "opengl-frontier.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if drift == "count":
        report["project"]["indexRangeAssertionCount"] -= 1
    else:
        report["project"]["indexRangeAssertions"][0]["expression"] = "offset + j"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    payloads = _saved_report_payloads(
        paths[3], "opengl-frontier", "opengl-workgroup-frontier"
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        _restore_fake_report(module, paths[0], paths[3], name, payloads)
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            1 if name == "opengl-workgroup-frontier" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="index-range assertion contract changed",
    ):
        module._check_opengl_frontier(
            *paths,
            "python",
            require_toolchain=False,
        )


def _prepare_fft_opengl_pointer_check(module, tmp_path, *, pointer_overrides=None):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    artifact_path = (
        work_dir / "out-fft-opengl-workgroup-pointer" / "opengl" / module.MLX_FFT_SOURCE
    ).with_suffix(".glsl")
    relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
    pointer_evidence = dict(module.FFT_OPENGL_EXPECTED_POINTER_EVIDENCE)
    pointer_evidence.update(pointer_overrides or {})
    report = {
        "kind": "crosstl-project-portability-report",
        "summary": {
            "unitCount": 1,
            "artifactCount": 1,
            "translatedCount": 0,
            "failedCount": 1,
            "diagnosticCounts": {"error": 1, "note": 0, "warning": 0},
            "diagnosticsByCode": {module.FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE: 1},
            "missingCapabilityCounts": {
                module.FFT_OPENGL_EXPECTED_MISSING_CAPABILITY: 1
            },
        },
        "diagnostics": [
            {
                "severity": "error",
                "code": module.FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
                "message": module.FFT_OPENGL_EXPECTED_MESSAGE,
                "location": {"file": module.MLX_FFT_SOURCE},
                "target": "opengl",
                "sourceBackend": "metal",
                "missingCapabilities": [module.FFT_OPENGL_EXPECTED_MISSING_CAPABILITY],
                "details": {
                    "sourcePath": module.MLX_FFT_SOURCE,
                    "targetArtifact": relative_artifact_path,
                    "workgroupPointer": pointer_evidence,
                },
            }
        ],
        "artifacts": [
            {
                "source": module.MLX_FFT_SOURCE,
                "sourceBackend": "metal",
                "target": "opengl",
                "path": relative_artifact_path,
                "status": "failed",
                "error": module.FFT_OPENGL_EXPECTED_MESSAGE,
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.MLX_FFT_SHA256,
                },
                "sourceSizeBytes": module.MLX_FFT_SOURCE_SIZE_BYTES,
                "specializationConstants": [
                    {"id": index}
                    for index in range(
                        module.FFT_OPENGL_EXPECTED_FUNCTION_CONSTANT_COUNT
                    )
                ],
                "templateMaterialization": {
                    "status": "materialized",
                    "specializations": [
                        {"materializedName": f"specialization_{index}"}
                        for index in range(
                            module.FFT_OPENGL_EXPECTED_SPECIALIZATION_COUNT
                        )
                    ],
                    "unsupported": [],
                },
            }
        ],
    }
    (report_dir / "fft-opengl-workgroup-pointer.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def test_fft_opengl_pointer_frontier_records_next_fail_closed_root(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_fft_opengl_pointer_check(module, tmp_path)
    commands = []

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._check_fft_opengl_workgroup_pointer_frontier(
        *paths,
        "python",
    )

    assert result == {
        "name": "fft-opengl-workgroup-pointer-frontier",
        "status": "blocked-as-expected",
        "report": ".crosstl-mlx-porting/reports/fft-opengl-workgroup-pointer.json",
        "source": module.MLX_FFT_SOURCE,
        "sourceHash": module.MLX_FFT_SHA256,
        "target": "opengl",
        "artifactStatus": "failed",
        "artifactEmitted": False,
        "nativeValidationAttempted": False,
        "nativeValidationStatus": "not-run-no-artifact",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": 117,
        "functionConstantCount": 22,
        "workgroupPointer": module.FFT_OPENGL_EXPECTED_POINTER_EVIDENCE,
        "provenanceStatus": "concrete-backing-preserved",
        "accessRangeStatus": "unprovable",
        "trackedIssue": "https://github.com/CrossGL/crosstl/issues/1671",
        "maxTemplateSpecializations": 4096,
        "maxTemplateMaterializationWork": 2097152,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }
    assert [name for name, _command, _kwargs in commands] == [
        "translate-fft-opengl-workgroup-pointer"
    ]
    assert commands[0][2]["check"] is False
    assert commands[0][2]["timeout_seconds"] == 900
    assert commands[0][1][-1] == "--no-format"
    config = (paths[2] / "fft-opengl-workgroup-pointer.toml").read_text(
        encoding="utf-8"
    )
    assert f'include = ["{module.MLX_FFT_SOURCE}"]' in config
    assert 'targets = ["opengl"]' in config
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_fft_opengl_pointer_frontier_rejects_changed_provenance(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_fft_opengl_pointer_check(
        module,
        tmp_path,
        pointer_overrides={"backingName": "unknown_shared_storage"},
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="lost concrete workgroup-pointer provenance",
    ):
        module._check_fft_opengl_workgroup_pointer_frontier(*paths, "python")


def _prepare_gemv_opengl_frontier_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    artifact_path = (
        work_dir / "out-gemv-opengl-frontier" / "opengl" / module.MLX_GEMV_SOURCE
    ).with_suffix(".glsl")
    relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
    source_hash = {"algorithm": "sha256", "value": module.MLX_GEMV_SHA256}
    report = {
        "kind": "crosstl-project-portability-report",
        "project": {
            "workgroupSizeRules": {
                module.MLX_GEMV_SOURCE: list(module.GEMV_REPORT_WORKGROUP_SIZE_RULE),
            },
            "workgroupSizeRuleCount": 1,
        },
        "summary": {
            "unitCount": 1,
            "skippedCount": 0,
            "targetCount": 1,
            "artifactCount": 1,
            "translatedCount": 0,
            "failedCount": 1,
            "diagnosticCounts": {"error": 1, "note": 0, "warning": 0},
            "diagnosticsByCode": {module.GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE: 1},
            "missingCapabilityCounts": {
                module.GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY: 1
            },
            "artifactProvenanceByPipeline": {"single-file-translate": 1},
            "artifactProvenanceByIntermediate": {"crossgl": 1},
        },
        "units": [
            {
                "id": module.MLX_GEMV_SOURCE,
                "path": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "extension": ".metal",
                "sourceHash": source_hash,
                "sourceSizeBytes": module.MLX_GEMV_SOURCE_SIZE_BYTES,
            }
        ],
        "diagnostics": [
            {
                "severity": "error",
                "code": module.GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
                "message": module.GEMV_OPENGL_EXPECTED_MESSAGE,
                "location": {"file": module.MLX_GEMV_SOURCE},
                "target": "opengl",
                "sourceBackend": "metal",
                "missingCapabilities": [module.GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY],
                "details": {
                    "sourcePath": module.MLX_GEMV_SOURCE,
                    "targetArtifact": relative_artifact_path,
                    "workgroupPointer": module.GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE,
                },
            }
        ],
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "target": "opengl",
                "path": relative_artifact_path,
                "status": "failed",
                "error": module.GEMV_OPENGL_EXPECTED_MESSAGE,
                "sourceHash": source_hash,
                "sourceSizeBytes": module.MLX_GEMV_SOURCE_SIZE_BYTES,
                "provenance": {
                    "intermediate": "crossgl",
                    "pipeline": "single-file-translate",
                },
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT,
                    "specializations": [
                        {"materializedName": f"specialization_{index}"}
                        for index in range(module.GEMV_EXPECTED_SPECIALIZATION_COUNT)
                    ],
                    "unsupported": [],
                },
            }
        ],
    }
    report_path = report_dir / "gemv-opengl-frontier.json"
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return (
        (mlx_root, work_dir, config_dir, report_dir, log_dir),
        report_path,
        artifact_path,
    )


def _stub_gemv_opengl_frontier(
    module,
    monkeypatch,
    commands,
    report_path,
    *,
    returncode=1,
    emitted_path=None,
):
    report_text = report_path.read_text(encoding="utf-8")

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")
        if emitted_path is not None:
            emitted_path.parent.mkdir(parents=True, exist_ok=True)
            emitted_path.write_text("#version 450 core\n", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            returncode,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)


def test_gemv_opengl_frontier_accepts_exact_pinned_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    commands = []
    _stub_gemv_opengl_frontier(module, monkeypatch, commands, report_path)

    result = module._check_gemv_opengl_frontier(*paths, "python")

    assert result == {
        "name": "gemv-opengl-frontier",
        "status": "blocked-as-expected",
        "report": ".crosstl-mlx-porting/reports/gemv-opengl-frontier.json",
        "source": module.MLX_GEMV_SOURCE,
        "sourceHash": module.MLX_GEMV_SHA256,
        "target": "opengl",
        "artifactStatus": "failed",
        "artifactEmitted": False,
        "emittedTargetFileCount": 0,
        "nativeValidationAttempted": False,
        "nativeValidationStatus": "not-run-no-artifact",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": 225,
        "workgroupSizeRule": [32, "BN", "BM"],
        "reportWorkgroupSizeRule": ["32", "BN", "BM"],
        "reportWorkgroupSizeRuleCount": 1,
        "workgroupSizeRuleConfigured": True,
        "diagnosticCode": module.GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
        "missingCapability": module.GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY,
        "workgroupPointer": module.GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE,
        "provenanceStatus": "concrete-backing-preserved",
        "accessRangeStatus": "unprovable",
        "trackedIssues": [
            "https://github.com/CrossGL/crosstl/issues/1671",
            "https://github.com/CrossGL/crosstl/issues/1786",
        ],
        "translationBlockedBy": ["https://github.com/CrossGL/crosstl/issues/1671"],
        "executionContractBlockedBy": [
            "https://github.com/CrossGL/crosstl/issues/1786",
        ],
        "maxTemplateSpecializations": 4096,
        "maxTemplateMaterializationWork": 2097152,
        "runtimeExecutionAttempted": False,
        "runtimeIntegrationIncluded": False,
        "runnableArtifactClaimed": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }
    assert [name for name, _command, _kwargs in commands] == [
        "translate-gemv-opengl-frontier"
    ]
    assert commands[0][2]["check"] is False
    assert commands[0][2]["timeout_seconds"] == 900
    assert commands[0][1][-1] == "--no-format"
    assert not artifact_path.exists()
    config = (paths[2] / "gemv-opengl-frontier.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_GEMV_SOURCE}"]' in config
    assert 'targets = ["opengl"]' in config
    assert "[project.workgroup_size_rules]" in config
    assert f'"{module.MLX_GEMV_SOURCE}" = [32, "BN", "BM"]' in config
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_opengl_frontier_rejects_wrong_diagnostic(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, _artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["diagnosticsByCode"] = {"project.translate.other": 1}
    report["diagnostics"][0]["code"] = "project.translate.other"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_frontier(module, monkeypatch, commands, report_path)

    with pytest.raises(module.PortingCheckError, match="diagnostic summary changed"):
        module._check_gemv_opengl_frontier(*paths, "python")

    assert [name for name, _command, _kwargs in commands] == [
        "translate-gemv-opengl-frontier"
    ]


def test_gemv_opengl_frontier_rejects_missing_report_workgroup_rule(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, _artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["project"]["workgroupSizeRules"] = {}
    report["project"]["workgroupSizeRuleCount"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_frontier(module, monkeypatch, commands, report_path)

    with pytest.raises(module.PortingCheckError, match="exact workgroup-size rule"):
        module._check_gemv_opengl_frontier(*paths, "python")


def test_gemv_opengl_frontier_rejects_emitted_artifact(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    commands = []
    _stub_gemv_opengl_frontier(
        module,
        monkeypatch,
        commands,
        report_path,
        emitted_path=artifact_path,
    )

    with pytest.raises(module.PortingCheckError, match="emitted GLSL"):
        module._check_gemv_opengl_frontier(*paths, "python")


def test_gemv_opengl_frontier_rejects_translation_success(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, _artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    commands = []
    _stub_gemv_opengl_frontier(
        module,
        monkeypatch,
        commands,
        report_path,
        returncode=0,
    )

    with pytest.raises(module.PortingCheckError, match="must remain fail-closed"):
        module._check_gemv_opengl_frontier(*paths, "python")


def test_gemv_opengl_frontier_rejects_missing_report_provenance(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, _artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0].pop("provenance")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_frontier(module, monkeypatch, commands, report_path)

    with pytest.raises(module.PortingCheckError, match="artifact provenance changed"):
        module._check_gemv_opengl_frontier(*paths, "python")


def test_gemv_opengl_frontier_rejects_changed_specialization_count(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, _artifact_path = _prepare_gemv_opengl_frontier_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    materialization = report["artifacts"][0]["templateMaterialization"]
    materialization["specializationCount"] -= 1
    materialization["specializations"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_frontier(module, monkeypatch, commands, report_path)

    with pytest.raises(module.PortingCheckError, match="specialization count changed"):
        module._check_gemv_opengl_frontier(*paths, "python")


def test_gemv_opengl_frontier_flag_replaces_toolchain_flag():
    module = _load_harness()

    args = module.parse_args(
        ["--mlx-root", "/tmp/mlx", "--require-opengl-gemv-frontier"]
    )

    assert args.require_opengl_gemv_frontier is True
    with pytest.raises(SystemExit):
        module.parse_args(["--mlx-root", "/tmp/mlx", "--require-opengl-gemv-toolchain"])


def _test_contract_identity(value):
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def _gemv_directx_contract_fixture(module):
    rule_path = f'project.workgroup_size_rules["{module.MLX_GEMV_SOURCE}"]'
    host_records = []
    execution_entries = []
    sizes = module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    target_entries = list(module.GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
    target_indexes = {name: index for index, name in enumerate(target_entries)}
    for source_index, target_entry in enumerate(reversed(target_entries)):
        target_index = target_indexes[target_entry]
        workgroup_size = sizes[target_index % len(sizes)]
        source_entry = f"gemv_host_{source_index:03d}"
        materialized_entry = f"gemv_materialized_{source_index:03d}"
        parameters = {
            "BM": str(workgroup_size[2]),
            "BN": str(workgroup_size[1]),
            "T": "float",
        }
        parameter_sources = {name: "source-instantiation" for name in parameters}
        record = {
            "name": "GEMVKernel",
            "hostName": source_entry,
            "materializedName": materialized_entry,
            "parameters": parameters,
            "parameterSources": parameter_sources,
        }
        entry = {
            "sourceEntryPoint": source_entry,
            "materializedEntryPoint": materialized_entry,
            "targetEntryPoint": target_entry,
            "workgroupSize": list(workgroup_size),
            "rule": {
                "components": list(module.GEMV_REPORT_WORKGROUP_SIZE_RULE),
                "sourcePattern": module.MLX_GEMV_SOURCE,
                "path": rule_path,
            },
            "parameters": parameters,
            "parameterSources": parameter_sources,
            "materialization": {
                "name": record["name"],
                "hostName": source_entry,
                "materializedName": materialized_entry,
            },
        }
        entry["identity"] = _test_contract_identity(
            f"{source_entry}:{materialized_entry}:{target_entry}:{workgroup_size}"
        )
        host_records.append(record)
        execution_entries.append(entry)
    execution = {
        "sourceEntryPoints": [entry["sourceEntryPoint"] for entry in execution_entries],
        "entryPoints": execution_entries,
        "provenance": {
            "kind": "materialized-template-rule",
            "path": rule_path,
        },
        "identity": _test_contract_identity("gemv-directx-execution"),
    }
    helper_record = {
        "name": "GEMVKernel::run",
        "materializedName": "gemv_helper_materialization",
        "parameters": {"BM": "1", "BN": "1", "T": "float"},
        "parameterSources": {
            "BM": "call-site",
            "BN": "call-site",
            "T": "call-site",
        },
    }
    return [helper_record, *host_records], execution


def _gemv_directx_frontier_source(module, *, replace_entry=None):
    entry_points = list(module.GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
    if replace_entry is not None:
        old_entry, new_entry = replace_entry
        entry_points[entry_points.index(old_entry)] = new_entry
    sizes = module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    return "typedef float16_t GemvNativeHalf;\n" + "\n".join(
        (
            f"[numthreads({size[0]}, {size[1]}, {size[2]})]\n"
            + f"void {entry_point}(uint3 lid : SV_GroupThreadID) {{\n"
            + "}\n"
        )
        for index, entry_point in enumerate(entry_points)
        for size in (sizes[index % len(sizes)],)
    )


def _prepare_gemv_directx_frontier_check(
    module,
    tmp_path,
    monkeypatch,
    *,
    generated=None,
):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    source_path = mlx_root / module.MLX_GEMV_SOURCE
    artifact_path = (
        work_dir
        / "out-gemv-directx-compiler-frontier"
        / "directx"
        / module.MLX_GEMV_SOURCE
    ).with_suffix(".hlsl")
    for path in (
        config_dir,
        report_dir,
        log_dir,
        source_path.parent,
        artifact_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"x" * module.MLX_GEMV_SOURCE_SIZE_BYTES)
    generated = generated or _gemv_directx_frontier_source(module)
    artifact_path.write_text(generated, encoding="utf-8")

    real_sha256 = module._sha256

    def pinned_source_sha256(path):
        if Path(path).resolve() == source_path.resolve():
            return module.MLX_GEMV_SHA256
        return real_sha256(path)

    monkeypatch.setattr(module, "_sha256", pinned_source_sha256)
    source_hash = {"algorithm": "sha256", "value": module.MLX_GEMV_SHA256}
    generated_hash = hashlib.sha256(generated.encode("utf-8")).hexdigest()
    relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
    specializations, execution = _gemv_directx_contract_fixture(module)
    report = {
        "kind": "crosstl-project-portability-report",
        "project": {
            "workgroupSizeRules": {
                module.MLX_GEMV_SOURCE: list(module.GEMV_REPORT_WORKGROUP_SIZE_RULE),
            },
            "workgroupSizeRuleCount": 1,
        },
        "summary": {
            "unitCount": 1,
            "skippedCount": 0,
            "targetCount": 1,
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "diagnosticsByCode": {},
            "missingCapabilityCounts": {},
            "artifactProvenanceByPipeline": {"single-file-translate": 1},
            "artifactProvenanceByIntermediate": {"crossgl": 1},
        },
        "units": [
            {
                "id": module.MLX_GEMV_SOURCE,
                "path": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "extension": ".metal",
                "sourceHash": source_hash,
                "sourceSizeBytes": module.MLX_GEMV_SOURCE_SIZE_BYTES,
            }
        ],
        "diagnostics": [],
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "target": "directx",
                "path": relative_artifact_path,
                "status": "translated",
                "sourceHash": source_hash,
                "sourceSizeBytes": module.MLX_GEMV_SOURCE_SIZE_BYTES,
                "generatedHash": {"algorithm": "sha256", "value": generated_hash},
                "generatedSizeBytes": len(generated.encode("utf-8")),
                "provenance": {
                    "intermediate": "crossgl",
                    "pipeline": "single-file-translate",
                },
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT,
                    "specializations": specializations,
                    "unsupported": [],
                },
                "execution": execution,
            }
        ],
    }
    report_path = report_dir / "gemv-directx-compiler-frontier.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return (
        (mlx_root, work_dir, config_dir, report_dir, log_dir),
        report_path,
        artifact_path,
        generated,
    )


def _stub_gemv_directx_frontier(
    module,
    monkeypatch,
    commands,
    report_path,
    artifact_path,
    generated,
    *,
    emit_translation_artifact=True,
    compiler_failure_entry=None,
    missing_binary_entry=None,
    unexpected_warning_entry=None,
    library_failure=False,
    missing_library=False,
    unexpected_library_warning=False,
):
    report_text = report_path.read_text(encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "translate-gemv-directx-compiler-frontier":
            report_path.write_text(report_text, encoding="utf-8")
            if emit_translation_artifact:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(generated, encoding="utf-8")
            return module.CommandResult(
                name, list(command), 0, stdout_path, stderr_path
            )

        output_path = Path(command[command.index("-Fo") + 1])
        if name == "compile-gemv-directx-all-entries":
            if not library_failure and not missing_library:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"DXIL:all-entries")
            if not library_failure:
                warnings = []
                numthreads_message = (
                    module.GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE
                )
                if unexpected_library_warning:
                    warnings.extend(
                        (
                            "C:/generated/gemv.hlsl:1:5: warning: "
                            "expression result unused [-Wunused-value]",
                            "    lid;",
                            "    ^~~",
                        )
                    )
                numthreads_source_lines = re.findall(
                    r"(?m)^[ \t]*(\[numthreads\([^\r\n]+\)\])[ \t]*$",
                    generated,
                )
                assert len(numthreads_source_lines) == (
                    module.GEMV_DIRECTX_EXPECTED_LIBRARY_WARNING_COUNT
                )
                for line_number, source_line in enumerate(
                    numthreads_source_lines, start=1
                ):
                    warnings.extend(
                        (
                            "C:/generated/gemv.hlsl:"
                            f"{line_number}:2: warning: {numthreads_message}",
                            f"    {source_line}",
                            "     ^",
                        )
                    )
                stderr_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
            return module.CommandResult(
                name,
                list(command),
                1 if library_failure else 0,
                stdout_path,
                stderr_path,
            )

        entry_point = command[command.index("-E") + 1]
        returncode = 1 if entry_point == compiler_failure_entry else 0
        if returncode == 0 and entry_point != missing_binary_entry:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"DXIL:{entry_point}".encode())
        if returncode == 0:
            if entry_point == unexpected_warning_entry:
                stderr_path.write_text(
                    "\n".join(
                        (
                            "C:/generated/gemv.hlsl:1:5: warning: "
                            "expression result unused [-Wunused-value]",
                            "    lid;",
                            "    ^~~",
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name, list(command), returncode, stdout_path, stderr_path
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)


def test_gemv_directx_compiler_frontier_accepts_exact_pinned_artifact(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    commands = []
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        commands,
        report_path,
        artifact_path,
        generated,
    )

    result = module._check_gemv_directx_compiler_frontier(*paths, "python")

    assert result["status"] == "passed"
    assert result["sourceHash"] == module.MLX_GEMV_SHA256
    assert result["sourceSizeBytes"] == module.MLX_GEMV_SOURCE_SIZE_BYTES
    assert result["artifactStatus"] == "translated"
    assert result["artifactPackaging"] == "single-aggregate-artifact"
    assert (
        result["artifactHash"]["value"]
        == hashlib.sha256(generated.encode("utf-8")).hexdigest()
    )
    assert result["artifactSizeBytes"] == len(generated.encode("utf-8"))
    assert result["artifactProvenance"] == {
        "intermediate": "crossgl",
        "pipeline": "single-file-translate",
    }
    assert result["templateSpecializationCount"] == 225
    assert result["unsupportedSpecializationCount"] == 0
    assert result["hostNamedMaterializationCount"] == 224
    assert result["reportExecutionEntryCount"] == 224
    assert result["executionIdentityJoinCount"] == 224
    assert result["generatedTargetEntryIdentityCount"] == 224
    assert result["generatedNumthreadsContractCount"] == 224
    assert result["workgroupSizeRule"] == [32, "BN", "BM"]
    assert result["reportWorkgroupSizeRule"] == ["32", "BN", "BM"]
    assert result["reportWorkgroupSizeRuleCount"] == 1
    assert result["resolvedWorkgroupSizes"] == [
        list(size) for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert result["resolvedWorkgroupSizeCounts"] == [
        {"workgroupSize": list(size), "entryCount": 32}
        for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert result["materializationResidueCount"] == 0
    assert result["bareValueDiscardCount"] == 0
    assert result["computeEntryPointCount"] == 224
    assert result["compilerArguments"] == ["-enable-16bit-types"]
    assert result["minimumShaderModel"] == "6.2"
    assert result["entryProfile"] == "cs_6_2"
    assert result["entryProfileCompilerEntryPoints"] == [
        "CSMain",
        "CSMain_85",
        "CSMain_113",
    ]
    assert result["entryProfileCompiledEntryPointCount"] == 3
    assert result["entryProfileDiagnosticCount"] == 0
    assert result["entryProfileUnusedValueWarningCount"] == 0
    assert result["libraryProfile"] == "lib_6_6"
    assert result["libraryExports"] == list(module.GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
    assert result["libraryExportCount"] == 224
    assert result["libraryCompilerRun"]["status"] == "compiled"
    assert result["libraryCompilerRun"]["outputSizeBytes"] > 0
    assert result["libraryCompilerRun"]["allowedWarningCounts"] == {
        "libraryNumthreads": 224,
    }
    assert result["libraryCompilerRun"]["unusedValueWarningCount"] == 0
    assert result["libraryCompilerRun"]["compilerArguments"] == ["-enable-16bit-types"]
    assert result["libraryCompilerRun"]["minimumShaderModel"] == "6.2"
    assert result["libraryAllowedWarnings"] == [
        {
            "classification": "library-profile-numthreads-ignored",
            "severity": "warning",
            "message": module.GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE,
            "sourceExpression": f"[numthreads({size[0]}, {size[1]}, {size[2]})]",
            "count": 32,
        }
        for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert result["libraryUnusedValueWarningCount"] == 0
    assert result["compilerCoveredEntryPointCount"] == 224
    assert result["uncompiledEntryPointCount"] == 0
    assert result["compilerCoverageStatus"] == "all-exported-entry-points-compiled"
    assert result["libraryCodeGenerationScope"] == "all-exported-functions"
    assert result["wholeArtifactSemanticValidityClaimed"] is False
    assert result["libraryExecutionSemanticsEstablished"] is False
    assert result["observedNumthreadsDirectives"] == [
        f"[numthreads({size[0]}, {size[1]}, {size[2]})]"
        for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert result["numthreadsContractEstablished"] is True
    assert result["exactWorkgroupSizeEstablished"] is True
    assert result["requiredWaveSize"] == 32
    assert result["requiredWaveSizeEstablished"] is False
    assert result["executionContractBlockedBy"] == list(
        module.GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES
    )
    assert result["runtimeExecutionAttempted"] is False
    assert result["runtimeIntegrationIncluded"] is False
    assert result["numericalParityClaimed"] is False
    assert result["runtimeParityClaimed"] is False
    assert [run["entryPoint"] for run in result["entryProfileCompilerRuns"]] == [
        "CSMain",
        "CSMain_85",
        "CSMain_113",
    ]
    assert all(run["outputSizeBytes"] > 0 for run in result["entryProfileCompilerRuns"])
    assert all(
        run["diagnosticCount"] == 0
        and run["unusedValueWarningCount"] == 0
        and run["profile"] == "cs_6_2"
        and run["compilerArguments"] == ["-enable-16bit-types"]
        and run["minimumShaderModel"] == "6.2"
        for run in result["entryProfileCompilerRuns"]
    )
    assert [name for name, _command, _kwargs in commands] == [
        "translate-gemv-directx-compiler-frontier",
        "compile-gemv-directx-csmain",
        "compile-gemv-directx-csmain-85",
        "compile-gemv-directx-csmain-113",
        "compile-gemv-directx-all-entries",
    ]
    assert commands[0][2] == {"check": False, "timeout_seconds": 900}
    assert commands[0][1][-1] == "--no-format"
    for _name, command, kwargs in commands[1:4]:
        assert command[command.index("-T") + 1] == "cs_6_2"
        assert command[command.index("-T") + 2] == "-enable-16bit-types"
        assert Path(command[command.index("-Fo") + 1]).is_file()
        assert kwargs == {"check": False}
    library_command = commands[4][1]
    assert library_command[library_command.index("-T") + 1] == "lib_6_6"
    assert library_command[library_command.index("-T") + 2] == ("-enable-16bit-types")
    exports = library_command[library_command.index("-exports") + 1].split(";")
    assert exports == list(module.GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
    assert len(exports) == len(set(exports)) == 224
    assert Path(library_command[library_command.index("-Fo") + 1]).is_file()
    assert commands[4][2] == {"check": False}
    config = (paths[2] / "gemv-directx-compiler-frontier.toml").read_text(
        encoding="utf-8"
    )
    assert f'include = ["{module.MLX_GEMV_SOURCE}"]' in config
    assert 'targets = ["directx"]' in config
    assert "[project.workgroup_size_rules]" in config
    assert f'"{module.MLX_GEMV_SOURCE}" = [32, "BN", "BM"]' in config
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_directx_execution_join_is_independent_of_report_list_order(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact = report["artifacts"][0]
    artifact["templateMaterialization"]["specializations"].reverse()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    result = module._check_gemv_directx_compiler_frontier(*paths, "python")

    assert result["executionIdentityJoinCount"] == 224
    assert result["generatedNumthreadsContractCount"] == 224


def test_gemv_directx_compiler_frontier_rejects_missing_execution_entry(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    execution = report["artifacts"][0]["execution"]
    execution["entryPoints"].pop()
    execution["sourceEntryPoints"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="exactly 224 execution entries"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_numthreads_contract_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = _gemv_directx_frontier_source(module).replace(
        "[numthreads(32, 1, 1)]",
        "[numthreads(32, 2, 1)]",
        1,
    )
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch, generated=generated
    )
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(
        module.PortingCheckError,
        match="numthreads declaration does not match.*CSMain",
    ):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_wrong_source_hash(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["units"][0]["sourceHash"]["value"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="source-unit provenance"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_materialization_count_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    materialization = report["artifacts"][0]["templateMaterialization"]
    materialization["specializationCount"] -= 1
    materialization["specializations"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="materialization evidence"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_missing_output(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        emit_translation_artifact=False,
    )

    with pytest.raises(module.PortingCheckError, match="artifact is missing"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_artifact_hash_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0]["generatedHash"]["value"] = "f" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="hash or size"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_compiler_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        compiler_failure_entry="CSMain_85",
    )

    with pytest.raises(module.PortingCheckError, match="failed to compile.*CSMain_85"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_missing_compiler_binary(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        missing_binary_entry="CSMain_113",
    )

    with pytest.raises(module.PortingCheckError, match="did not emit a binary"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


@pytest.mark.parametrize("statement", ["lid;", "42;", "probe.value;", "probe[0];"])
def test_gemv_directx_compiler_frontier_rejects_bare_value_discard(
    tmp_path,
    monkeypatch,
    statement,
):
    module = _load_harness()
    generated = _gemv_directx_frontier_source(module)
    generated = generated.replace(
        "void CSMain_85(uint3 lid : SV_GroupThreadID) {\n}\n",
        "void CSMain_85(uint3 lid : SV_GroupThreadID) {\n" f"    {statement}\n" "}\n",
    )
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch, generated=generated
    )
    commands = []
    _stub_gemv_directx_frontier(
        module, monkeypatch, commands, report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="bare value-discard statement"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")

    assert [name for name, _command, _kwargs in commands] == [
        "translate-gemv-directx-compiler-frontier"
    ]


def test_gemv_directx_compiler_frontier_rejects_entry_unused_value_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        unexpected_warning_entry="CSMain_113",
    )

    with pytest.raises(module.PortingCheckError, match="DXC diagnostics changed"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_library_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        library_failure=True,
    )

    with pytest.raises(module.PortingCheckError, match="all-entry library"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_missing_library(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        missing_library=True,
    )

    with pytest.raises(module.PortingCheckError, match="did not emit.*all-entry"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_library_unused_value_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch
    )
    _stub_gemv_directx_frontier(
        module,
        monkeypatch,
        [],
        report_path,
        artifact_path,
        generated,
        unexpected_library_warning=True,
    )

    with pytest.raises(module.PortingCheckError, match="all-entry library"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_wrong_entry_selection(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = _gemv_directx_frontier_source(
        module, replace_entry=("CSMain_85", "CSMain_225")
    )
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch, generated=generated
    )
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="entry selection changed"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_rejects_export_set_drift(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = _gemv_directx_frontier_source(
        module, replace_entry=("CSMain_84", "CSMain_225")
    )
    paths, report_path, artifact_path, generated = _prepare_gemv_directx_frontier_check(
        module, tmp_path, monkeypatch, generated=generated
    )
    _stub_gemv_directx_frontier(
        module, monkeypatch, [], report_path, artifact_path, generated
    )

    with pytest.raises(module.PortingCheckError, match="artifact export set changed"):
        module._check_gemv_directx_compiler_frontier(*paths, "python")


def test_gemv_directx_compiler_frontier_flag_is_reduced_scope_only(tmp_path):
    module = _load_harness()
    args = module.parse_args(
        [
            "--mlx-root",
            str(tmp_path),
            "--require-directx-gemv-compiler-frontier",
        ]
    )
    assert args.require_directx_gemv_compiler_frontier is True
    args.mode = module.FULL_CORPUS_MODE
    with pytest.raises(module.PortingCheckError, match="only valid"):
        module.run_checks(args)


def _prepare_gemv_vulkan_check(module, tmp_path, generated):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    generated_path = work_dir / "out" / "vulkan" / "gemv.spvasm"
    for path in (config_dir, report_dir, log_dir, generated_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(generated, encoding="utf-8")
    report = {
        "summary": {
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"warning": 0},
        },
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "target": "vulkan",
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "status": "translated",
                "templateMaterialization": {
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT
                },
            }
        ],
    }
    (report_dir / "gemv-vulkan.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return mlx_root, work_dir, config_dir, report_dir, log_dir


def _gemv_vulkan_frontier_source():
    entry_points = "\n".join(
        f'  OpEntryPoint GLCompute %main_{index} "compute_main_{index}"'
        for index in range(1, 225)
    )
    return entry_points + "\n"


def _stub_gemv_vulkan_toolchain(module, monkeypatch, commands):
    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "assemble-gemv-vulkan":
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"\x03\x02\x23\x07")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")


def test_gemv_vulkan_toolchain_check_structurally_validates_full_artifact(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source(),
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    result = module._check_gemv_vulkan_toolchain(*paths, "python")

    assert result["status"] == "passed"
    assert result["specializationCount"] == 225
    assert result["entryPointCount"] == 224
    assert result["structuralValidationStatus"] == "validated"
    assert result["semanticReadinessStatus"] == "no-known-codegen-fallbacks"
    assert result["semanticWarningCount"] == 0
    assert result["semanticWarningsByIssue"] == {}
    assert result["semanticBlockers"] == []
    assert result["reportWarningCount"] == 0
    assert result["reportWarningTransportTrackedBy"] is None
    assert result["runtimeIntegrationIncluded"] is False
    assert [name for name, _command in commands] == [
        "translate-gemv-vulkan",
        "assemble-gemv-vulkan",
        "validate-gemv-vulkan-spirv",
    ]
    assert commands[1][1][:3] == [
        "/tools/spirv-as",
        "--target-env",
        "vulkan1.1",
    ]
    assert commands[2][1][:3] == [
        "/tools/spirv-val",
        "--target-env",
        "vulkan1.1",
    ]
    config = (paths[2] / "gemv-vulkan.toml").read_text(encoding="utf-8")
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_vulkan_toolchain_check_rejects_translation_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(module, tmp_path, "")
    _mlx_root, _work_dir, _config_dir, report_dir, log_dir = paths
    report = {
        "summary": {
            "translatedCount": 0,
            "failedCount": 1,
            "diagnosticCounts": {"error": 1},
        },
        "artifacts": [
            {
                "source": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "target": "vulkan",
                "status": "failed",
            }
        ],
        "diagnostics": [
            {
                "code": "project.translate.unsupported-feature",
                "sourceBackend": "metal",
                "target": "vulkan",
                "missingCapabilities": ["spirv.nested_return_storage_buffer_function"],
                "message": (
                    "SPIR-V pointer-preserving function inlining requires returns "
                    "to be top-level statements; helper 'run' contains a nested "
                    "return"
                ),
            }
        ],
    }
    (report_dir / "gemv-vulkan.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    with pytest.raises(
        module.PortingCheckError,
        match="Vulkan GEMV translation failed: .*contains a nested return",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]
    assert not (paths[1] / "validation" / "gemv-vulkan.spv").exists()


def test_gemv_vulkan_toolchain_check_rejects_untracked_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source() + "; WARNING: subgroup behavior changed\n",
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="emitted a semantic warning",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_gemv_vulkan_toolchain_check_rejects_resolved_float_fallback_warning(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = (
        "; WARNING: WaveActiveBitXor requires a compatible arithmetic or "
        "bitwise operand; got float\n" + _gemv_vulkan_frontier_source()
    )
    paths = _prepare_gemv_vulkan_check(module, tmp_path, generated)
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="emitted a semantic warning",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_gemv_vulkan_toolchain_check_rejects_materialization_residue(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_gemv_vulkan_check(
        module,
        tmp_path,
        _gemv_vulkan_frontier_source() + 'OpName %residue "PrimitiveType"\n',
    )
    commands = []
    _stub_gemv_vulkan_toolchain(module, monkeypatch, commands)

    with pytest.raises(
        module.PortingCheckError,
        match="retained unresolved materialization text",
    ):
        module._check_gemv_vulkan_toolchain(*paths, "python")

    assert [name for name, _command in commands] == ["translate-gemv-vulkan"]


def test_arange_opengl_check_confirms_native_validation(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        commands.append((name, list(command)))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    result = module._check_arange_opengl(*paths, "python")

    config = (paths[2] / "arange-opengl.toml").read_text(encoding="utf-8")
    assert "[project.entry_points]" in config
    assert f'"{module.MLX_ARANGE_SOURCE}" = "arangeuint32"' in config
    assert result["selectedEntryPoint"] == "arangeuint32"
    assert result["targetEntryPoint"] == "main"
    assert result["interfaceResourceCount"] == 3
    assert result["standaloneArtifact"] is True
    assert result["arangeDataFlowPreserved"] is True
    assert result["nativeValidationAttempted"] is True
    assert result["nativeValidationBlockerConfirmed"] is False
    assert result["nativeValidationStatus"] == "validated"
    assert result["nativeValidationExitCode"] == 0
    assert result["trackedIssues"] == list(
        module.OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES
    )
    validator_command = commands[1][1]
    assert validator_command[:5] == [
        "/tools/glslangValidator",
        "--target-env",
        "opengl",
        "--target-env",
        "spirv1.3",
    ]


def test_arange_opengl_check_reports_unavailable_validator(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    result = module._check_arange_opengl(*paths, "python")

    assert result["nativeValidationAttempted"] is False
    assert result["nativeValidationBlockerConfirmed"] is False
    assert result["nativeValidationStatus"] == "not-run-tool-unavailable"
    assert result["nativeValidatorStatus"] == "unavailable"
    assert result["trackedIssues"] == []


def test_arange_opengl_check_requires_selected_entry_metadata(tmp_path, monkeypatch):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )
    report_path = paths[3] / "arange-opengl.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0].pop("entryPoint")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(module.PortingCheckError, match="selected compute entry"):
        module._check_arange_opengl(*paths, "python")


def test_arange_opengl_check_rejects_native_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths = _prepare_arange_opengl_check(
        module,
        tmp_path,
        _arange_opengl_frontier_source(),
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text(
            (
                "ERROR: compute shader compilation failed"
                if name == "validate-arange-opengl"
                else ""
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            2 if name == "validate-arange-opengl" else 0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/tools/glslangValidator")

    with pytest.raises(
        module.PortingCheckError,
        match="failed without a tracked validation issue",
    ):
        module._check_arange_opengl(*paths, "python")


def test_arange_opengl_check_rejects_extra_entry_resource(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    generated = _arange_opengl_frontier_source().replace(
        "layout(local_size_x",
        "layout(std430, binding = 4) buffer extra_Buffer { uint extra_[]; };\n"
        "layout(local_size_x",
    )
    paths = _prepare_arange_opengl_check(module, tmp_path, generated)

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name,
            list(command),
            0,
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)

    with pytest.raises(
        module.PortingCheckError,
        match="only start, step, and output resources",
    ):
        module._check_arange_opengl(*paths, "python")


def test_runtime_readiness_reports_tracked_plan_resource_blockers(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    report_dir = mlx_root / ".crosstl-mlx-porting" / "reports"
    report_dir.mkdir(parents=True)
    artifact_report = report_dir / "opengl-readiness-artifacts.json"
    artifact_report.write_text(
        json.dumps(_translated_arange_report(module, "opengl")),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        module,
        "build_runtime_artifact_manifest",
        lambda report_path: _runtime_arange_artifact_manifest(
            module, "opengl", output_name="unrelatedResource"
        ),
    )

    result = module._plan_runtime_readiness_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name="opengl-runtime-readiness",
        artifact_report=artifact_report,
        targets=("opengl",),
        required_native_runtime_targets=(),
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["diagnosticCounts"] == {"error": 0, "note": 0, "warning": 0}
    assert result["metadataGapCodes"] == []
    assert result["planBlockerCodes"] == [
        "project.runtime-verification.resource-unbound"
    ]
    assert (
        result["runtimePlanDiagnosticsByCode"][
            "project.runtime-verification.resource-unbound"
        ]
        == 1
    )
    assert (
        "https://github.com/CrossGL/crosstl/issues/1392"
        not in result["trackedRuntimeIssues"]
    )
    assert result["runtimeFixtureExecution"]["status"] == "blocked-by-tracked-issues"
    assert result["nativeRuntimeExecution"]["status"] in {
        "blocked-by-runtime-driver",
        "blocked-by-tracked-issues",
    }


def test_reduced_runtime_readiness_aggregates_fixture_execution(monkeypatch):
    module = _load_harness()
    calls = []
    reports = [
        {
            "name": "directx-runtime-readiness",
            "status": "planned",
            "testCount": 1,
            "diagnosticsByCode": {},
            "runtimeArtifactDiagnosticsByCode": {},
            "runtimePlanDiagnosticsByCode": {},
            "runtimeFixtureExecution": {
                "status": "passed",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 1,
                    "skippedCount": 0,
                    "unavailableCount": 0,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
            "nativeRuntimeExecution": {
                "status": "blocked-by-runtime-driver",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 0,
                    "unavailableCount": 1,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
        },
        {
            "name": "vulkan-runtime-readiness",
            "status": "planned",
            "testCount": 1,
            "diagnosticsByCode": {},
            "runtimeArtifactDiagnosticsByCode": {},
            "runtimePlanDiagnosticsByCode": {},
            "runtimeFixtureExecution": {
                "status": "passed",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 1,
                    "skippedCount": 0,
                    "unavailableCount": 0,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
            "nativeRuntimeExecution": {
                "status": "blocked-by-runtime-driver",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 0,
                    "unavailableCount": 1,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
        },
        {
            "name": "opengl-runtime-readiness",
            "status": "blocked-by-tracked-issues",
            "testCount": 1,
            "diagnosticsByCode": {},
            "runtimeArtifactDiagnosticsByCode": {},
            "runtimePlanDiagnosticsByCode": {
                "project.runtime-verification.resource-unbound": 1
            },
            "runtimeFixtureExecution": {
                "status": "blocked-by-tracked-issues",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 1,
                    "unavailableCount": 0,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
            "nativeRuntimeExecution": {
                "status": "blocked-by-runtime-driver",
                "summary": {
                    "fixtureCount": 1,
                    "passedCount": 0,
                    "skippedCount": 0,
                    "unavailableCount": 1,
                    "translationFailedCount": 0,
                    "runtimeFailedCount": 0,
                    "comparisonFailedCount": 0,
                    "failedCount": 0,
                },
            },
        },
    ]

    def fake_plan_runtime_readiness(**kwargs):
        calls.append(kwargs)
        return reports.pop(0)

    monkeypatch.setattr(
        module,
        "_plan_runtime_readiness_for_report",
        fake_plan_runtime_readiness,
    )

    result = module._plan_reduced_runtime_readiness(
        Path("/tmp/mlx"),
        Path("/tmp/reports"),
        require_vulkan_native_runtime=True,
        require_opengl_native_runtime=True,
    )

    assert calls[0]["name"] == "directx-runtime-readiness"
    assert calls[0]["artifact_report"] == Path("/tmp/reports/directx-frontier.json")
    assert calls[0]["required_native_runtime_targets"] == ()
    assert calls[1]["name"] == "vulkan-runtime-readiness"
    assert calls[1]["artifact_report"] == Path("/tmp/reports/vulkan-frontier.json")
    assert calls[1]["required_native_runtime_targets"] == ("vulkan",)
    assert calls[2]["required_native_runtime_targets"] == ("opengl",)
    assert result["status"] == "blocked-by-tracked-issues"
    assert result["runtimeFixtureExecutionIncluded"] is True
    assert result["runtimeFixtureExecutionByStatus"] == {
        "blocked-by-tracked-issues": 1,
        "passed": 2,
    }
    assert result["runtimeFixtureExecutionSummary"] == {
        "comparisonFailedCount": 0,
        "failedCount": 0,
        "fixtureCount": 3,
        "passedCount": 2,
        "runtimeFailedCount": 0,
        "skippedCount": 1,
        "translationFailedCount": 0,
        "unavailableCount": 0,
    }
    assert result["nativeRuntimeExecutionIncluded"] is True
    assert result["nativeRuntimeExecutionByStatus"] == {
        "blocked-by-runtime-driver": 3,
    }
    assert result["nativeRuntimeExecutionSummary"] == {
        "comparisonFailedCount": 0,
        "failedCount": 0,
        "fixtureCount": 3,
        "passedCount": 0,
        "runtimeFailedCount": 0,
        "skippedCount": 0,
        "translationFailedCount": 0,
        "unavailableCount": 3,
    }


def test_full_corpus_mode_writes_bounded_config_and_checks_counts(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append(list(command))
        (report_dir / "full-corpus.json").write_text(
            json.dumps(_full_corpus_report(module, mlx_root, work_dir)),
            encoding="utf-8",
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    config = (config_dir / "full-corpus.toml").read_text(encoding="utf-8")
    assert 'include = ["mlx/backend/metal/kernels/**/*.metal"]' in config
    assert 'targets = ["directx", "opengl", "vulkan"]' in config
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 131072" in config
    assert commands == [
        [
            "python",
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_dir / "full-corpus.toml"),
            "--report",
            str(report_dir / "full-corpus.json"),
            "--validate",
        ]
    ]
    assert "--run-toolchains" not in commands[0]
    assert result["unitCount"] == 40
    assert result["artifactCount"] == 120
    assert result["translatedCount"] == 117
    assert result["failedCount"] == 3
    assert result["status"] == "passed-with-expected-fence-blockers"
    assert result["targetCounts"] == {
        "directx": {"translatedCount": 39, "failedCount": 1},
        "opengl": {"translatedCount": 39, "failedCount": 1},
        "vulkan": {"translatedCount": 39, "failedCount": 1},
    }
    assert result["fenceContract"]["status"] == "blocked-as-expected"
    assert set(result["fenceContract"]["targetContracts"]) == {
        "directx",
        "opengl",
        "vulkan",
    }
    assert result["shaderArtifactsOnly"] is True
    assert result["runtimeIntegrationIncluded"] is False
    assert result["runtimeParityClaimed"] is False


def test_full_corpus_mode_rejects_untracked_translation_errors(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(module, "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        report = _full_corpus_report(
            module,
            mlx_root,
            work_dir,
            include_extra_failure=True,
        )
        (report_dir / "full-corpus.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(module.PortingCheckError, match="tracked issue references"):
        module._translate_full_corpus(
            mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
        )


def test_full_corpus_mode_reports_tracked_translation_errors(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(
        module,
        "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES",
        ("https://github.com/CrossGL/crosstl/issues/1354",),
    )
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        report = _full_corpus_report(
            module,
            mlx_root,
            work_dir,
            include_extra_failure=True,
        )
        (report_dir / "full-corpus.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 1, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["translatedCount"] == 116
    assert result["failedCount"] == 4
    assert result["expectedFenceFailureCount"] == 3
    assert result["unexpectedFailedCount"] == 1
    assert result["unexpectedErrorDiagnosticsByCode"] == {
        "project.translate.failed": 1,
        "project.validate.failed-artifact": 1,
    }
    assert result["trackedTranslationIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1354"
    ]


def test_full_corpus_mode_reports_tracked_timeout_without_report(tmp_path, monkeypatch):
    module = _load_harness()
    monkeypatch.setattr(
        module,
        "FULL_CORPUS_TRANSLATION_TRACKED_ISSUES",
        ("https://github.com/CrossGL/crosstl/issues/1376",),
    )
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("timed out", encoding="utf-8")
        return module.CommandResult(name, list(command), 124, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_full_corpus(
        mlx_root, work_dir, config_dir, report_dir, log_dir, "python"
    )

    assert result["status"] == "blocked-by-tracked-issues"
    assert result["reportProduced"] is False
    assert result["returncode"] == 124
    assert result["trackedTranslationIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1376"
    ]


def test_reduced_frontier_accepts_multiple_vulkan_toolchain_runs_per_artifact(
    tmp_path, monkeypatch
):
    module = _load_harness()
    monkeypatch.setattr(module, "FRONTIER_VALIDATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    sources = module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    frontier_count = len(sources)
    commands = []
    alias_evidence = {"source": module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE}
    monkeypatch.setattr(
        module,
        "_scaled_attention_local_alias_evidence",
        lambda *_args: alias_evidence,
    )

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        output_dir = (
            work_dir / "out-vulkan-frontier"
            if name == "vulkan-frontier"
            else work_dir / "out-vulkan-frontier-toolchain"
        )
        toolchain_runs = []
        if name == "validate-vulkan-frontier-toolchain":
            for source in sources:
                artifact_path = (output_dir / "vulkan" / Path(source)).with_suffix(
                    ".spvasm"
                )
                relative_path = artifact_path.relative_to(mlx_root).as_posix()
                toolchain_runs.extend(
                    {
                        "source": source,
                        "target": "vulkan",
                        "path": relative_path,
                        "status": "ok",
                    }
                    for _ in range(2)
                )
        _write_clean_frontier_report(
            module,
            mlx_root,
            output_dir,
            report_dir / f"{name}.json",
            target="vulkan",
            sources=sources,
            toolchain_runs=toolchain_runs,
        )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_vulkan_frontier(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_toolchain=True,
        run_optional_toolchain=False,
    )

    assert result["toolchainRuns"] == frontier_count * 2
    assert result["status"] == "passed"
    assert result["scope"] == "target-split-frontier"
    assert result["vulkanValidationStatus"] == "validated"
    assert result["semanticReadinessStatus"] == "not-established"
    assert result["regressionEvidence"] == [alias_evidence]
    assert result["runtimeParityClaimed"] is False
    assert commands[0][0] == "vulkan-frontier"
    assert "--validate" in commands[0][1]
    assert "--run-toolchains" not in commands[0][1]
    assert commands[1][0] == "validate-vulkan-frontier-toolchain"
    assert "--run-toolchains" in commands[1][1]
    toolchain_config = (
        config_dir / "validate-vulkan-frontier-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert 'targets = ["vulkan"]' in toolchain_config
    assert module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE in toolchain_config


def test_reduced_frontier_requires_all_directx_entries_per_artifact(
    tmp_path, monkeypatch
):
    module = _load_harness()
    monkeypatch.setattr(module, "FRONTIER_VALIDATION_TRACKED_ISSUES", ())
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)

    directx_toolchain_count = len(module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES)
    commands = []

    def fake_run_command(name, command, *, log_dir, check=True, timeout_seconds=None):
        commands.append((name, list(command)))
        returncode = 0
        if name == "directx-workgroup-frontier":
            _write_dynamic_workgroup_report(
                module,
                report_dir / f"{name}.json",
                mlx_root,
                work_dir / "out-directx-workgroup-frontier",
                target="directx",
                sources=module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
                validated=True,
            )
            returncode = 1
        else:
            is_toolchain = name == "validate-directx-frontier-toolchain"
            output_dir = (
                work_dir / "out-directx-frontier-toolchain"
                if is_toolchain
                else work_dir / "out-directx-frontier"
            )
            toolchain_runs = []
            if is_toolchain:
                for source in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES:
                    artifact_path = (output_dir / "directx" / Path(source)).with_suffix(
                        ".hlsl"
                    )
                    relative_path = artifact_path.relative_to(mlx_root).as_posix()
                    for index in range(
                        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS[source]
                    ):
                        entry_point = "CSMain" if index == 0 else f"CSMain_{index + 1}"
                        toolchain_runs.append(
                            {
                                "source": source,
                                "target": "directx",
                                "path": relative_path,
                                "command": [
                                    "dxc",
                                    "-T",
                                    "cs_6_0",
                                    "-E",
                                    entry_point,
                                    relative_path,
                                ],
                                "status": "ok",
                            }
                        )
            _write_clean_frontier_report(
                module,
                mlx_root,
                output_dir,
                report_dir / f"{name}.json",
                target="directx",
                sources=module.MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES,
                toolchain_runs=toolchain_runs,
            )
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(
            name, list(command), returncode, stdout_path, stderr_path
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    result = module._translate_directx_frontier(
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
        require_directx_toolchain=True,
    )

    assert result["toolchainRuns"] == (module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT)
    assert result["status"] == "passed-with-expected-workgroup-blockers"
    assert result["directxToolchainRequired"] is True
    assert result["directxToolchainSources"] == list(
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert result["directxToolchainArtifactCount"] == directx_toolchain_count
    assert result["directxToolchainExpectedEntryPointCounts"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
    )
    assert result["directxToolchainExpectedEntryPointCount"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
    )
    assert result["directxToolchainValidatedSources"] == list(
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert result["directxToolchainValidatedArtifactCount"] == directx_toolchain_count
    assert result["directxToolchainValidatedEntryPointCounts"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
    )
    assert result["directxToolchainValidatedEntryPointCount"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
    )
    assert result["semanticReadinessStatus"] == "not-established"
    assert result["directxValidationStatus"] == "validated"
    assert result["workgroupBlockedSources"] == list(
        module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert all(
        evidence["sourceEntryPointIdentityStatus"] == "matched-materialized-host-names"
        for evidence in result["dynamicWorkgroupDispatchEvidence"].values()
    )
    assert commands[1][0] == "directx-workgroup-frontier"
    assert commands[2][0] == "validate-directx-frontier-toolchain"
    assert "--run-toolchains" in commands[2][1]
    toolchain_config = (
        config_dir / "validate-directx-frontier-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert 'targets = ["directx"]' in toolchain_config
    assert "[project.specialization_constants]" in toolchain_config
    for selector, value in module.MLX_FRONTIER_SPECIALIZATION_CONSTANTS.items():
        assert f"{json.dumps(selector)} = {json.dumps(value)}" in toolchain_config
    for source in module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES:
        assert source in toolchain_config


def test_directx_toolchain_frontier_matches_pinned_dxc_inventory():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    expected_all_sources = (
        module.MLX_ARANGE_SOURCE,
        module.MLX_ARG_REDUCE_SOURCE,
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_LAYER_NORM_SOURCE,
        module.MLX_LOGSUMEXP_SOURCE,
        module.MLX_RANDOM_SOURCE,
        module.MLX_RMS_NORM_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        module.MLX_SOFTMAX_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )
    expected_sources = (
        module.MLX_ARANGE_SOURCE,
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_RANDOM_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_TERNARY_SOURCE,
    )
    assert module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES == expected_sources
    assert module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES == expected_all_sources
    assert tuple(module.MLX_DIRECTX_FRONTIER_ENTRY_POINT_COUNTS) == (
        expected_all_sources
    )
    assert tuple(module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS) == expected_sources
    assert module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS == {
        module.MLX_ARANGE_SOURCE: 11,
        module.MLX_BINARY_TWO_SOURCE: 225,
        module.MLX_RANDOM_SOURCE: 2,
        module.MLX_ROPE_SOURCE: 18,
        module.MLX_TERNARY_SOURCE: 212,
    }
    assert module.MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS == {
        module.MLX_ARG_REDUCE_SOURCE: 24,
        module.MLX_LAYER_NORM_SOURCE: 12,
        module.MLX_LOGSUMEXP_SOURCE: 6,
        module.MLX_RMS_NORM_SOURCE: 12,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: 42,
        module.MLX_SOFTMAX_SOURCE: 10,
    }
    assert len(expected_sources) == 5
    assert module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT == sum(
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS.values()
    )
    assert module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT == 468
    assert sum(module.MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS.values()) == 106
    assert {
        source: evidence["specializationCount"]
        for source, evidence in module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE.items()
    } == {
        module.MLX_ARG_REDUCE_SOURCE: 51,
        module.MLX_LAYER_NORM_SOURCE: 16,
        module.MLX_LOGSUMEXP_SOURCE: 7,
        module.MLX_RMS_NORM_SOURCE: 12,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: 42,
        module.MLX_SOFTMAX_SOURCE: 17,
    }
    directx_status = gaps["directx_toolchain_status"]
    assert directx_status["specialization_constants"] == (
        module.MLX_FRONTIER_SPECIALIZATION_CONSTANTS
    )
    assert directx_status["dxc_validated_sources"] == list(expected_sources)
    assert directx_status["expected_entry_point_counts"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
    )
    assert directx_status["workgroup_blocked_sources"] == list(
        module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert directx_status["workgroup_blocked_entry_point_counts"] == (
        module.MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS
    )
    assert directx_status["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert directx_status["workgroup_blocked_by"] == (
        module.MLX_DYNAMIC_WORKGROUP_TRACKED_ISSUE
    )
    assert directx_status["dispatch_evidence"] == (
        module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE
    )
    assert module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    ]["hostLines"].endswith("613-640,685-753")
    assert set(directx_status["directx_toolchain_gaps"]) == (
        set(module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES) | {module.MLX_FENCE_SOURCE}
    )
    assert {
        f"https://github.com/CrossGL/crosstl/issues/{issue}"
        for issue in (1694, 1695, 1701, 1728)
    } <= set(module.RESOLVED_FRONTIER_ISSUES)
    assert set(module.RESOLVED_FRONTIER_ISSUES) <= set(gaps["resolved_issues"])
    assert not set(module.RESOLVED_FRONTIER_ISSUES) & set(gaps["tracked_issues"])
    assert "https://github.com/CrossGL/crosstl/issues/1696" in gaps["tracked_issues"]


def test_directx_frontier_readme_records_compile_only_scope_and_current_gaps():
    readme = MLX_README_PATH.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "official DXC v1.9.2602.24 on Windows CI" in readme
    assert (
        "the emitted five-source frontier: `arange.metal`, `binary_two.metal`,"
        in normalized_readme
    )
    assert "11, 225, 2, 18, and 212 entries respectively" in normalized_readme
    assert "468 generated compute entries" in normalized_readme
    assert "six excluded aggregate sources cover 106 compute entries" in (
        normalized_readme
    )
    assert "no placeholder workgroup size is restored" in normalized_readme
    for issue in (1694, 1695, 1696, 1701, 1728, 1750, 1542, 1537):
        assert f"https://github.com/CrossGL/crosstl/issues/{issue}" in readme
    assert "does not dispatch these kernels or establish numerical parity" in (
        normalized_readme
    )
    assert "DirectX remains outside the DXC gate" not in readme


def test_opengl_index_range_contract_is_documented_as_a_portability_precondition():
    module = _load_harness()
    readme = MLX_README_PATH.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "24 source-qualified index-range assertions" in normalized_readme
    assert "24 configured index-range assertions" in normalized_readme
    assert "24 audited index-range assertions" not in normalized_readme
    assert "exact assertion count and content" in normalized_readme
    assert "inclusive bounds `[0, 2147483647]`" in normalized_readme
    assert "explicit MLX host/runtime portability preconditions for OpenGL" in (
        normalized_readme
    )
    assert "They are not inferred guarantees" in normalized_readme
    assert "does not enforce them at runtime" in normalized_readme
    assert "do not establish runtime integration or numerical parity" in (
        normalized_readme
    )
    for (
        source,
        expressions,
    ) in module.MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS.items():
        assert f"`{Path(source).name}`" in readme
        for expression in expressions:
            assert f"`{expression}`" in readme


def test_binary_resource_relocation_issue_is_full_corpus_only():
    module = _load_harness()
    issue = "https://github.com/CrossGL/crosstl/issues/1659"

    assert issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert issue not in module.FRONTIER_VALIDATION_TRACKED_ISSUES
    assert issue not in module.RUNTIME_READINESS_TRACKED_ISSUES


def test_new_pin_resource_profile_and_workgroup_contracts_are_tracked():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    resource_issue = "https://github.com/CrossGL/crosstl/issues/1669"
    profile_issue = "https://github.com/CrossGL/crosstl/issues/1670"
    workgroup_issue = "https://github.com/CrossGL/crosstl/issues/1671"
    struct_constant_issue = "https://github.com/CrossGL/crosstl/issues/1672"

    assert resource_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert resource_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert profile_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert workgroup_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert workgroup_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert struct_constant_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert struct_constant_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert resource_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert profile_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert workgroup_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert struct_constant_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert resource_issue in gaps["tracked_issues"]
    assert profile_issue in gaps["tracked_issues"]
    assert workgroup_issue in gaps["tracked_issues"]
    assert struct_constant_issue in gaps["tracked_issues"]
    assert resource_issue in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert profile_issue in gaps["full_corpus_scout"]["validation_blocked_by"]
    assert workgroup_issue in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert struct_constant_issue in gaps["full_corpus_scout"]["translation_blocked_by"]


def test_fft_opengl_evidence_records_provenance_without_artifact_claims():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["opengl_fft_workgroup_pointer_status"]
    assert status["status"] == "blocked-after-provenance"
    assert status["source"] == module.MLX_FFT_SOURCE
    assert status["source_sha256"] == module.MLX_FFT_SHA256
    assert status["target"] == "opengl"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 0,
        "failed_count": 1,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 117,
        "unsupported_specialization_count": 0,
        "function_constant_count": 22,
    }
    assert status["diagnostic"] == {
        "code": module.FFT_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
        "missing_capability": module.FFT_OPENGL_EXPECTED_MISSING_CAPABILITY,
        "message": module.FFT_OPENGL_EXPECTED_MESSAGE,
        "workgroup_pointer": module.FFT_OPENGL_EXPECTED_POINTER_EVIDENCE,
    }
    assert status["provenance_status"] == "concrete-backing-preserved"
    assert status["access_range_status"] == "unprovable"
    assert status["artifact_emitted"] is False
    assert status["native_validation_status"] == "not-run-no-artifact"
    assert status["blocked_by"] == [module.FFT_OPENGL_WORKGROUP_POINTER_ISSUE]
    assert status["runtime_integration_included"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "the complete pinned `fft.metal` source for OpenGL" in readme
    assert "All 117 reachable template specializations materialize" in readme
    assert "workgroup access range cannot be proven" in readme
    assert "No GLSL artifact is emitted" in readme


def test_gemv_directx_gap_records_full_compiler_coverage_without_runtime_claims():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["directx_gemv_compiler_frontier_status"]
    assert status["status"] == "all-exported-entry-points-compiled"
    assert status["source"] == module.MLX_GEMV_SOURCE
    assert status["source_sha256"] == module.MLX_GEMV_SHA256
    assert status["source_size_bytes"] == module.MLX_GEMV_SOURCE_SIZE_BYTES
    assert status["target"] == "directx"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
        "workgroup_size_rule": [32, "BN", "BM"],
        "report_workgroup_size_rule": ["32", "BN", "BM"],
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 225,
        "host_named_specialization_count": 224,
        "unsupported_specialization_count": 0,
        "unresolved_residue_count": 0,
        "bare_value_discard_count": 0,
        "report_execution_entry_count": 224,
        "execution_identity_join": "hostName/materializedName",
    }
    assert status["artifact"]["compute_entry_count"] == 224
    assert status["artifact"]["packaging"] == "single-aggregate-artifact"
    assert status["artifact"]["generated_target_entry_identity_count"] == 224
    assert status["artifact"]["generated_numthreads_contract_count"] == 224
    assert status["artifact"]["resolved_workgroup_sizes"] == [
        list(size) for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert status["artifact"]["provenance"] == {
        "intermediate": "crossgl",
        "pipeline": "single-file-translate",
    }
    compiler = status["compiler"]
    assert compiler["entry_profile"] == {
        "profile": "cs_6_2",
        "compiler_arguments": ["-enable-16bit-types"],
        "entry_points": ["CSMain", "CSMain_85", "CSMain_113"],
        "compiled_binary_count": 3,
        "diagnostic_count": 0,
        "unused_value_warning_count": 0,
    }
    assert compiler["library_profile"] == {
        "profile": "lib_6_6",
        "compiler_arguments": ["-enable-16bit-types"],
        "export_set": "CSMain;CSMain_2;...;CSMain_224",
        "export_count": 224,
        "compiled_library_count": 1,
        "coverage": "all-exported-functions-code-generated",
        "unused_value_warning_count": 0,
    }
    assert set(compiler["allowed_warnings"]) == {"library_numthreads"}
    library_warning = compiler["allowed_warnings"]["library_numthreads"]
    assert library_warning["count"] == 224
    assert library_warning["classification"] == ("library-profile-numthreads-ignored")
    assert library_warning["source_expressions"] == [
        f"[numthreads({size[0]}, {size[1]}, {size[2]})]"
        for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    assert "tracked_by" not in library_warning
    assert compiler["whole_artifact_semantic_validity_claimed"] is False
    assert status["execution_contracts"] == {
        "workgroup_size_rule": [32, "BN", "BM"],
        "resolved_workgroup_sizes": [
            list(size) for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
        ],
        "generated_entry_contract_count": 224,
        "numthreads_contract_established": True,
        "exact_workgroup_size_established": True,
        "required_wave_size": 32,
        "required_wave_size_established": False,
        "library_execution_semantics_established": False,
        "blocked_by": list(module.GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES),
    }
    assert status["tracked_issues"] == list(
        module.GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES
    )
    assert status["runtime_execution_attempted"] is False
    assert status["runtime_integration_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "--require-directx-gemv-compiler-frontier" in readme
    assert "A second `lib_6_6` invocation retains" in readme
    assert "exporting and code-generating all 224 functions" in readme
    assert "Any unused-value warning, error, or other diagnostic fails the gate" in (
        readme
    )
    assert "This establishes exact workgroup-size specialization" in readme
    assert "It does not establish wave semantics" in readme


def test_gemv_opengl_gap_records_strict_expected_frontier():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["opengl_gemv_frontier_status"]
    assert status["status"] == "blocked-as-expected"
    assert status["source"] == module.MLX_GEMV_SOURCE
    assert status["source_sha256"] == module.MLX_GEMV_SHA256
    assert status["source_size_bytes"] == module.MLX_GEMV_SOURCE_SIZE_BYTES
    assert status["target"] == "opengl"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 0,
        "failed_count": 1,
        "emitted_target_file_count": 0,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
        "workgroup_size_rule": [32, "BN", "BM"],
        "report_workgroup_size_rule": ["32", "BN", "BM"],
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 225,
        "unsupported_specialization_count": 0,
    }
    assert status["diagnostic"] == {
        "code": module.GEMV_OPENGL_EXPECTED_DIAGNOSTIC_CODE,
        "missing_capability": module.GEMV_OPENGL_EXPECTED_MISSING_CAPABILITY,
        "message": module.GEMV_OPENGL_EXPECTED_MESSAGE,
        "workgroup_pointer": module.GEMV_OPENGL_EXPECTED_POINTER_EVIDENCE,
    }
    assert status["range_derivation"] == {
        "sgN": "simd_gid % 8",
        "simdM": "simd_gid / 8",
        "bm": "simdM * 1",
        "tgp_results": "tgp_memory + sgN * 2 + bm",
        "source_required_subgroup_width": 32,
        "configured_workgroup_size": [32, 8, 1],
        "source_reachable_simd_gid": [0, 7],
        "base_offset_range": [0, 14],
        "guarded_reduction_max_element_index": 14,
        "backing_element_count": 16,
        "highest_valid_element_index": 15,
    }
    assert status["artifact_emitted"] is False
    assert status["native_validation_attempted"] is False
    assert status["native_validation_status"] == "not-run-no-artifact"
    assert status["translation_blocked_by"] == [module.GEMV_OPENGL_BACKING_RANGE_ISSUE]
    assert status["execution_contracts"] == {
        "workgroup_size_specialization": {
            "status": "configured-in-project-report",
            "rule": [32, "BN", "BM"],
            "report_rule": ["32", "BN", "BM"],
            "emitted_artifact_contract_claimed": False,
        },
        "subgroup_width_specialization": {
            "status": "not-established",
            "blocked_by": module.GEMV_OPENGL_SUBGROUP_WIDTH_ISSUE,
        },
    }
    assert status["blocked_by"] == list(module.GEMV_OPENGL_FRONTIER_TRACKED_ISSUES)
    assert status["runtime_execution_attempted"] is False
    assert status["runtime_integration_included"] is False
    assert status["runnable_artifact_claimed"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    assert module.GEMV_OPENGL_SUBGROUP_WIDTH_ISSUE in gaps["tracked_issues"]

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "--require-opengl-gemv-frontier" in readme
    assert "--require-opengl-gemv-toolchain" not in readme
    assert "all 225 specializations materialized" in readme
    assert "the base offset is in `[0, 14]`" in readme
    assert "at most element `14` of the 16-element backing" in readme
    assert "target subgroup-width contract is also not established" in readme
    assert "attempts no native compiler" in readme
    assert "does not claim an emitted or runnable GEMV artifact" in readme


def test_run_checks_full_corpus_mode_skips_reduced_frontier(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        lambda *args: {"name": "mlx-checkout", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        lambda *args: {"name": "metal-kernel-scan", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_metal_roundtrip",
        lambda *args, **kwargs: {"name": "metal-roundtrip", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_translate_full_corpus",
        lambda *args: {"name": "full-corpus", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_atomic_fence_contract",
        lambda *args: pytest.fail("dedicated fence check should not run twice"),
    )
    monkeypatch.setattr(
        module,
        "_check_reference_accessor_lvalue_identity",
        lambda *args, **kwargs: pytest.fail(
            "reduced reference accessor proof should not run"
        ),
    )
    monkeypatch.setattr(
        module,
        "_check_template_member_buffer_pointer",
        lambda *args: pytest.fail(
            "reduced template member pointer proof should not run"
        ),
    )
    monkeypatch.setattr(
        module,
        "_translate_directx_frontier",
        lambda *args, **kwargs: pytest.fail("reduced DirectX frontier should not run"),
    )
    monkeypatch.setattr(
        module,
        "_translate_vulkan_frontier",
        lambda *args, **kwargs: pytest.fail("reduced Vulkan frontier should not run"),
    )
    monkeypatch.setattr(
        module,
        "_check_arange_opengl",
        lambda *args: pytest.fail("OpenGL smoke check should not run"),
    )
    monkeypatch.setattr(
        module,
        "_check_opengl_frontier",
        lambda *args, **kwargs: pytest.fail("OpenGL frontier should not run"),
    )
    monkeypatch.setattr(
        module,
        "_check_fft_opengl_workgroup_pointer_frontier",
        lambda *args: pytest.fail("OpenGL FFT pointer frontier should not run"),
    )

    result = module.run_checks(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=None,
            no_clean=False,
            python="python",
            require_directx_toolchain=False,
            require_vulkan_toolchain=False,
            mode=module.FULL_CORPUS_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "metal-roundtrip",
        "full-corpus",
    ]
    assert result["scope"]["mode"] == module.FULL_CORPUS_MODE
    assert result["scope"]["metalRoundTripIncluded"] is True
    assert result["scope"]["fullCorpusExpectedUnitCount"] == 40
    assert result["scope"]["fullCorpusExpectedArtifactCount"] == 120
    assert result["scope"]["fullCorpusExpectedTranslatedArtifactCount"] == 117
    assert result["scope"]["fullCorpusExpectedFenceFailureCount"] == 3
    assert result["scope"]["referenceAccessorProofIncluded"] is False
    assert result["scope"]["referenceAccessorTargets"] == []
    assert result["scope"]["referenceAccessorDirectxToolchainRequired"] is False
    assert result["scope"]["referenceAccessorOpenglToolchainRequired"] is False
    assert result["scope"]["templateMemberBufferPointerProofIncluded"] is False
    assert result["scope"]["templateMemberBufferPointerTargets"] == []
    assert (
        result["scope"]["templateMemberBufferPointerNativeValidationIncluded"] is False
    )
    assert result["scope"]["fftOpenGLWorkgroupPointerFrontierIncluded"] is False
    assert result["scope"]["runtimeParityClaimed"] is False


def test_run_checks_reduced_frontier_includes_runtime_readiness(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        lambda *args: {"name": "mlx-checkout", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        lambda *args: {"name": "metal-kernel-scan", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_metal_roundtrip",
        lambda *args, **kwargs: {"name": "metal-roundtrip", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_atomic_fence_contract",
        lambda *args: {
            "name": "atomic-fence-contract",
            "status": "blocked-as-expected",
        },
    )
    reference_accessor_requirements = []

    def fake_reference_accessor_check(
        *args, require_directx_toolchain, require_opengl_toolchain
    ):
        reference_accessor_requirements.append(
            (require_directx_toolchain, require_opengl_toolchain)
        )
        return {
            "name": "reference-accessor-lvalue-identity",
            "status": "passed",
            "runtimeParityClaimed": False,
        }

    monkeypatch.setattr(
        module,
        "_check_reference_accessor_lvalue_identity",
        fake_reference_accessor_check,
    )
    monkeypatch.setattr(
        module,
        "_check_template_member_buffer_pointer",
        lambda *args: {
            "name": "template-member-buffer-pointer",
            "status": "passed",
            "runtimeParityClaimed": False,
        },
    )
    directx_frontier_requirements = []

    def fake_directx_frontier(*args, require_directx_toolchain):
        directx_frontier_requirements.append(require_directx_toolchain)
        return {"name": "directx-frontier", "status": "passed"}

    vulkan_frontier_requirements = []

    def fake_vulkan_frontier(*args, require_toolchain, run_optional_toolchain):
        vulkan_frontier_requirements.append((require_toolchain, run_optional_toolchain))
        return {"name": "vulkan-frontier", "status": "passed"}

    monkeypatch.setattr(module, "_translate_directx_frontier", fake_directx_frontier)
    monkeypatch.setattr(module, "_translate_vulkan_frontier", fake_vulkan_frontier)
    monkeypatch.setattr(
        module,
        "_check_arange_opengl",
        lambda *args: {"name": "arange-opengl", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_directx_compiler_frontier",
        lambda *args: {
            "name": "gemv-directx-compiler-frontier",
            "status": "passed",
            "runtimeParityClaimed": False,
        },
    )
    opengl_frontier_requirements = []

    def fake_opengl_frontier_check(*args, require_toolchain):
        opengl_frontier_requirements.append(require_toolchain)
        return {"name": "opengl-frontier", "status": "passed"}

    monkeypatch.setattr(
        module,
        "_check_opengl_frontier",
        fake_opengl_frontier_check,
    )
    monkeypatch.setattr(
        module,
        "_check_fft_opengl_workgroup_pointer_frontier",
        lambda *args: {
            "name": "fft-opengl-workgroup-pointer-frontier",
            "status": "blocked-as-expected",
        },
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_opengl_frontier",
        lambda *args: {
            "name": "gemv-opengl-frontier",
            "status": "blocked-as-expected",
        },
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_vulkan_toolchain",
        lambda *args: {
            "name": "gemv-vulkan-toolchain",
            "status": "passed",
        },
    )
    runtime_requirements = []

    def fake_runtime_readiness(*args, **kwargs):
        runtime_requirements.append(kwargs)
        return {
            "name": "runtime-readiness",
            "status": "blocked-by-tracked-issues",
        }

    monkeypatch.setattr(
        module,
        "_plan_reduced_runtime_readiness",
        fake_runtime_readiness,
    )
    monkeypatch.setattr(
        module,
        "_translate_full_corpus",
        lambda *args: pytest.fail("full-corpus check should not run"),
    )

    result = module.run_checks(
        SimpleNamespace(
            mlx_root=str(mlx_root),
            work_dir=None,
            no_clean=False,
            python="python",
            require_directx_toolchain=False,
            require_directx_gemv_compiler_frontier=True,
            require_vulkan_toolchain=False,
            require_vulkan_native_runtime=False,
            require_opengl_native_runtime=True,
            require_opengl_frontier_toolchain=True,
            require_opengl_gemv_frontier=True,
            require_vulkan_gemv_toolchain=True,
            mode=module.REDUCED_FRONTIER_MODE,
        )
    )

    assert [check["name"] for check in result["checks"]] == [
        "mlx-checkout",
        "metal-kernel-scan",
        "metal-roundtrip",
        "atomic-fence-contract",
        "reference-accessor-lvalue-identity",
        "template-member-buffer-pointer",
        "directx-frontier",
        "vulkan-frontier",
        "arange-opengl",
        "gemv-directx-compiler-frontier",
        "opengl-frontier",
        "fft-opengl-workgroup-pointer-frontier",
        "gemv-opengl-frontier",
        "gemv-vulkan-toolchain",
        "runtime-readiness",
    ]
    assert reference_accessor_requirements == [(False, True)]
    assert directx_frontier_requirements == [False]
    assert vulkan_frontier_requirements == [(False, True)]
    assert opengl_frontier_requirements == [True]
    assert runtime_requirements == [
        {
            "require_vulkan_native_runtime": False,
            "require_opengl_native_runtime": True,
        }
    ]
    assert result["scope"]["openglFrontierToolchainRequired"] is True
    assert result["scope"]["directxGemvCompilerFrontierRequired"] is True
    assert result["scope"]["fftOpenGLWorkgroupPointerFrontierIncluded"] is True
    assert result["scope"]["openglGemvFrontierRequired"] is True
    assert result["scope"]["openglNativeRuntimeRequired"] is True
    assert result["scope"]["vulkanGemvToolchainRequired"] is True
    assert result["scope"]["runtimeReadinessIncluded"] is True
    assert result["scope"]["runtimeFixtureExecutionIncluded"] is True
    assert result["scope"]["nativeRuntimeExecutionIncluded"] is True
    assert result["scope"]["referenceAccessorProofIncluded"] is True
    assert result["scope"]["referenceAccessorTargets"] == ["directx", "opengl"]
    assert result["scope"]["referenceAccessorDirectxToolchainRequired"] is False
    assert result["scope"]["referenceAccessorOpenglToolchainRequired"] is True
    assert result["scope"]["templateMemberBufferPointerProofIncluded"] is True
    assert result["scope"]["templateMemberBufferPointerTargets"] == [
        "directx",
        "opengl",
    ]
    assert (
        result["scope"]["templateMemberBufferPointerNativeValidationIncluded"] is False
    )
    assert result["scope"]["runtimeParityClaimed"] is False
    assert result["scope"]["nonFenceFrontierSources"] == list(
        module.MLX_NON_FENCE_REDUCED_FRONTIER_SOURCES
    )
    assert "cleanFrontierSources" not in result["scope"]
    assert result["scope"]["blockedFrontierSources"] == [module.MLX_FENCE_SOURCE]
    assert result["scope"]["blockedFrontierIssues"] == [
        "https://github.com/CrossGL/crosstl/issues/1537"
    ]
    assert result["scope"]["directxTranslatedFrontierSources"] == list(
        module.MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
    )
    assert result["scope"]["directxWorkgroupBlockedFrontierSources"] == list(
        module.MLX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert result["scope"]["vulkanTranslatedFrontierSources"] == list(
        module.MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
    )
    assert result["scope"]["openglTranslatedFrontierSources"] == list(
        module.MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES
    )
    assert result["scope"]["openglWorkgroupBlockedFrontierSources"] == list(
        module.MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )


def test_rms_norm_contract_fixture_matches_pinned_harness_configuration():
    module = _load_rms_norm_harness()
    metadata = _load_rms_norm_fixture_metadata()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.MLX_RMS_NORM_SHA256 == (
        "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee"
    )
    assert metadata["upstreamSourceReplacement"] is False
    assert metadata["functionConstant"] == {
        "name": module.RMS_NORM_FUNCTION_CONSTANT_NAME,
        "id": module.RMS_NORM_FUNCTION_CONSTANT_ID,
        "required": True,
    }
    assert module.RMS_NORM_HOST_ENTRY_POINTS == (
        "rms_loopedbfloat16",
        "rms_loopedfloat16",
        "rms_loopedfloat32",
        "rmsbfloat16",
        "rmsfloat16",
        "rmsfloat32",
        "vjp_rms_loopedbfloat16",
        "vjp_rms_loopedfloat16",
        "vjp_rms_loopedfloat32",
        "vjp_rmsbfloat16",
        "vjp_rmsfloat16",
        "vjp_rmsfloat32",
    )
    assert metadata["hostNamedEntryPoints"] == list(module.RMS_NORM_HOST_ENTRY_POINTS)
    assert len(metadata["hostNamedEntryPoints"]) == 12
    assert module.RMS_NORM_EXPECTED_ENTRY_POINT_COUNT == 12
    assert (
        sum(
            "_looped" in entry_point for entry_point in metadata["hostNamedEntryPoints"]
        )
        == module.RMS_NORM_LOOPED_ENTRY_POINT_COUNT
        == 6
    )
    assert metadata["representativeWorkgroupContract"] == {
        "hostSource": "mlx/backend/metal/normalization.cpp",
        "hostLines": "52-91,137-197",
        "singleRowFormula": "[32 * ceil_div(ceil_div(axis_size, 4), 32), 1, 1]",
        "loopedFormula": "[maxTotalThreadsPerThreadgroup, 1, 1]",
        "workgroupSizes": [[32, 1, 1], [64, 1, 1]],
        "upstreamHostValid": True,
        "completeRuntimeCoverage": False,
    }
    assert metadata["directx"] == {
        "profile": module.RMS_NORM_DIRECTX_PROFILE,
        "libraryArtifactCount": 2,
        "hostNamedEntryCountPerArtifact": 12,
        "compiledEntryPointCountPerVariant": 1,
        "compilerRunCount": 2,
        "variants": list(module.RMS_NORM_DIRECTX_VARIANTS),
    }
    assert metadata["opengl"] == {
        "deferred": True,
        "constantId": 20,
        "standaloneArtifactCount": 24,
        "standaloneArtifactCountPerVariant": 12,
        "compiledArtifactCount": 24,
        "validatedArtifactCount": 24,
        "variants": list(module.RMS_NORM_OPENGL_VARIANTS),
    }
    assert metadata["claims"] == {
        "translationAndNativeCompilationOnly": True,
        "representativeHostValidWorkgroupSizes": True,
        "completeRuntimeCoverage": False,
        "numericalRuntimeParity": False,
        "fullMlxTestSuite": False,
        "runtimeBlockedBy": list(module.RMS_NORM_RUNTIME_BLOCKERS),
    }
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    status = expected_gaps["rms_norm_specialization_status"]
    assert status["source"] == module.MLX_RMS_NORM_SOURCE
    assert status["source_sha256"] == module.MLX_RMS_NORM_SHA256
    assert status["host_named_entry_count"] == 12
    assert status["representative_workgroup_contract"] == {
        "host_source": "mlx/backend/metal/normalization.cpp",
        "host_lines": "52-91,137-197",
        "single_row_formula": "[32 * ceil_div(ceil_div(axis_size, 4), 32), 1, 1]",
        "looped_formula": "[maxTotalThreadsPerThreadgroup, 1, 1]",
        "workgroup_sizes": [[32, 1, 1], [64, 1, 1]],
        "upstream_host_valid": True,
        "complete_runtime_coverage": False,
    }
    assert status["directx"]["variants"] == [
        {
            "name": variant["name"],
            "selector": variant["selector"],
            "selector_kind": variant["selectorKind"],
            "value": variant["value"],
            "workgroup_size": variant["workgroupSize"],
        }
        for variant in module.RMS_NORM_DIRECTX_VARIANTS
    ]
    assert status["directx"]["native_compilation"] == ("required-on-windows-ci")
    assert status["directx"]["profile"] == "cs_6_2"
    assert status["directx"]["compiler_arguments"] == ["-enable-16bit-types"]
    assert status["directx"]["library_artifact_count"] == 2
    assert status["directx"]["execution_entry_count_per_artifact"] == 12
    assert status["directx"]["generated_numthreads_count_per_artifact"] == 12
    assert status["directx"]["compiled_entry_count_per_variant"] == 1
    assert status["directx"]["compiler_run_count"] == 2
    assert status["opengl"]["variants"] == [
        {
            "name": variant["name"],
            "workgroup_size": variant["workgroupSize"],
        }
        for variant in module.RMS_NORM_OPENGL_VARIANTS
    ]
    assert status["opengl"]["native_compilation"] == "required-on-linux-ci"
    assert status["opengl"]["standalone_artifact_count"] == 24
    assert status["opengl"]["standalone_artifact_count_per_variant"] == 12
    assert status["opengl"]["compiled_artifact_count"] == 24
    assert status["opengl"]["validated_artifact_count"] == 24
    assert status["numerical_execution_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["runtime_blocked_by"] == list(module.RMS_NORM_RUNTIME_BLOCKERS)
    assert set(status["runtime_blocked_by"]) <= set(expected_gaps["tracked_issues"])


def test_rms_norm_checkout_verifies_revision_and_source_hash(tmp_path, monkeypatch):
    module = _load_rms_norm_harness()
    mlx_root, _work_dir, _report_dir, log_dir = _prepare_reduced_rms_norm_checkout(
        module, tmp_path, monkeypatch
    )
    revision_stdout = log_dir / "mlx-rmsnorm-revision.stdout"
    revision_stderr = log_dir / "mlx-rmsnorm-revision.stderr"
    revision_stdout.write_text(module.MLX_COMMIT + "\n", encoding="utf-8")
    revision_stderr.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_run_command",
        lambda *args, **kwargs: {
            "returncode": 0,
            "stdoutPath": revision_stdout,
            "stderrPath": revision_stderr,
        },
    )

    identity = module._verify_mlx_checkout(mlx_root, log_dir=log_dir)

    assert identity["commit"] == module.MLX_COMMIT
    assert identity["sourceHash"] == {
        "algorithm": "sha256",
        "value": module.MLX_RMS_NORM_SHA256,
    }

    (mlx_root / module.MLX_RMS_NORM_SOURCE).write_text(
        "modified source\n", encoding="utf-8"
    )
    with pytest.raises(module.MlxRmsNormProofError, match="SHA-256 mismatch"):
        module._verify_mlx_checkout(mlx_root, log_dir=log_dir)


def test_rms_norm_directx_variants_translate_through_project_api(tmp_path, monkeypatch):
    module = _load_rms_norm_harness()
    mlx_root, work_dir, report_dir, log_dir = _prepare_reduced_rms_norm_checkout(
        module, tmp_path, monkeypatch
    )

    result = module._check_directx(
        mlx_root,
        work_dir,
        report_dir,
        log_dir,
        require_toolchain=False,
    )

    assert result["status"] == "passed"
    assert result["runtimeBlockedBy"] == list(module.RMS_NORM_RUNTIME_BLOCKERS)
    assert result["artifactCount"] == 2
    assert result["executionEntryCountPerArtifact"] == 12
    assert result["runtimeParityClaimed"] is False
    assert result["numericalExecutionIncluded"] is False
    assert result["nativeCompilation"]["status"] == "not-required"
    assert result["nativeCompilation"]["compiledArtifactCount"] == 0
    assert result["nativeCompilation"]["runs"] == []
    variants = {variant["name"]: variant for variant in result["variants"]}
    assert set(variants) == {
        variant["name"] for variant in module.RMS_NORM_DIRECTX_VARIANTS
    }
    for expected in module.RMS_NORM_DIRECTX_VARIANTS:
        variant = variants[expected["name"]]
        assert variant["selector"] == expected["selector"]
        assert variant["selectorKind"] == expected["selectorKind"]
        assert variant["value"] is expected["value"]
        assert variant["workgroupSize"] == expected["workgroupSize"]
        assert variant["valueProvenance"] == {
            "kind": "project-variant",
            "path": (
                f"project.variants.{expected['name']}.specialization_constants."
                f"{expected['selector']}"
            ),
            "selector": expected["selector"],
            "selectorKind": expected["selectorKind"],
            "variant": expected["name"],
        }
        assert variant["generatedStaticConst"] == (
            "static const bool has_w = " + ("true;" if expected["value"] else "false;")
        )
        assert variant["representativeEntryPoint"] == "CSMain"
        execution = variant["execution"]
        assert execution["hostNamedMaterializationCount"] == 12
        assert execution["executionEntryCount"] == 12
        assert execution["generatedNumthreadsContractCount"] == 12
        assert execution["consumedWorkgroupProjectionCount"] == 6
        assert execution["sourceEntryPoints"] == list(module.RMS_NORM_HOST_ENTRY_POINTS)
        assert execution["executionIdentity"]["algorithm"] == "sha256"
        assert re.fullmatch(r"[0-9a-f]{64}", execution["executionIdentity"]["value"])
    report = json.loads(
        (report_dir / "rms-norm-directx-variants.json").read_text(encoding="utf-8")
    )
    expected_workgroup_sizes = {
        variant["name"]: variant["workgroupSize"]
        for variant in module.RMS_NORM_DIRECTX_VARIANTS
    }
    assert report["project"]["variantWorkgroupSizes"] == expected_workgroup_sizes
    assert report["summary"]["artifactsByVariant"] == {
        variant_name: {
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
        }
        for variant_name in expected_workgroup_sizes
    }
    assert len(report["artifacts"]) == 2
    for artifact in report["artifacts"]:
        variant_name = artifact["variant"]
        workgroup_size = expected_workgroup_sizes[variant_name]
        execution = artifact["execution"]
        assert execution["sourceEntryPoints"] == list(module.RMS_NORM_HOST_ENTRY_POINTS)
        assert execution["provenance"] == {
            "kind": "project-variant",
            "path": f"project.variants.{variant_name}.workgroup_size",
            "variant": variant_name,
        }
        assert len(execution["entryPoints"]) == 12
        assert all(
            entry["workgroupSize"] == workgroup_size
            for entry in execution["entryPoints"]
        )
        generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
        attribute = (
            f"[numthreads({workgroup_size[0]}, {workgroup_size[1]}, "
            f"{workgroup_size[2]})]"
        )
        assert generated.count(attribute) == 12
        assert generated.count(variants[variant_name]["generatedStaticConst"]) == 1


def test_rms_norm_opengl_translation_retains_deferred_specialization(
    tmp_path, monkeypatch
):
    module = _load_rms_norm_harness()
    mlx_root, work_dir, report_dir, log_dir = _prepare_reduced_rms_norm_checkout(
        module, tmp_path, monkeypatch
    )

    result = module._check_opengl(
        mlx_root,
        work_dir,
        report_dir,
        log_dir,
        require_toolchain=False,
    )

    assert result["status"] == "passed"
    assert result["artifactCount"] == 24
    assert result["artifactCountPerVariant"] == 12
    assert result["runtimeParityClaimed"] is False
    assert result["numericalExecutionIncluded"] is False
    assert result["runtimeBlockedBy"] == list(module.RMS_NORM_RUNTIME_BLOCKERS)
    assert result["nativeCompilation"]["status"] == "not-required"
    assert result["nativeCompilation"]["compiledArtifactCount"] == 0
    assert result["nativeCompilation"]["validatedArtifactCount"] == 0
    assert result["nativeCompilation"]["runs"] == []
    specialization = result["specializationConstant"]
    assert specialization["name"] == "has_w"
    assert specialization["id"] == 20
    assert specialization["required"] is True
    assert specialization["deferred"] is True
    assert specialization["valueProvenance"] == {"kind": "runtime-override-required"}
    assert specialization["generatedContract"] == (
        "layout(constant_id = 20) const bool has_w = false;"
    )
    assert specialization["specializationMaterialization"]["mode"] == "deferred"
    variants = {variant["name"]: variant for variant in result["variants"]}
    assert set(variants) == {
        variant["name"] for variant in module.RMS_NORM_OPENGL_VARIANTS
    }
    for expected in module.RMS_NORM_OPENGL_VARIANTS:
        variant = variants[expected["name"]]
        assert variant["workgroupSize"] == expected["workgroupSize"]
        assert variant["artifactCount"] == 12
        assert variant["executionEntryCount"] == 12
        assert variant["generatedLocalSizeContractCount"] == 12
        artifacts = variant["artifacts"]
        assert [artifact["sourceEntryPoint"] for artifact in artifacts] == list(
            module.RMS_NORM_HOST_ENTRY_POINTS
        )
        for artifact in artifacts:
            assert artifact["targetEntryPoint"] == "main"
            assert artifact["workgroupSize"] == expected["workgroupSize"]
            assert artifact["generatedLocalSizeContract"] == (
                "layout(local_size_x = "
                f"{expected['workgroupSize'][0]}, local_size_y = "
                f"{expected['workgroupSize'][1]}, local_size_z = "
                f"{expected['workgroupSize'][2]}) in;"
            )
            assert artifact["consumedWorkgroupProjectionCount"] == (
                1 if "_looped" in artifact["sourceEntryPoint"] else 0
            )
            for identity_key in ("executionIdentity", "executionEntryIdentity"):
                assert artifact[identity_key]["algorithm"] == "sha256"
                assert re.fullmatch(r"[0-9a-f]{64}", artifact[identity_key]["value"])
    report = json.loads(
        (report_dir / "rms-norm-opengl-deferred.json").read_text(encoding="utf-8")
    )
    expected_workgroup_sizes = {
        variant["name"]: variant["workgroupSize"]
        for variant in module.RMS_NORM_OPENGL_VARIANTS
    }
    assert report["project"]["variantWorkgroupSizes"] == expected_workgroup_sizes
    assert report["summary"]["artifactsByVariant"] == {
        variant_name: {
            "artifactCount": 12,
            "translatedCount": 12,
            "failedCount": 0,
        }
        for variant_name in expected_workgroup_sizes
    }
    assert len(report["artifacts"]) == 24
    artifact_identities = set()
    for artifact in report["artifacts"]:
        variant_name = artifact["variant"]
        source_entry = artifact["entryPoint"]["source"]
        artifact_identities.add((variant_name, source_entry))
        assert artifact["entryPoint"] == {
            "source": source_entry,
            "target": "main",
            "stage": "compute",
        }
        assert artifact["provenance"] == {
            "pipeline": "entry-scoped-translate",
            "intermediate": "crossgl",
        }
        execution = artifact["execution"]
        assert execution["sourceEntryPoints"] == [source_entry]
        assert execution["provenance"] == {
            "kind": "project-variant",
            "path": f"project.variants.{variant_name}.workgroup_size",
            "variant": variant_name,
        }
        assert len(execution["entryPoints"]) == 1
        assert execution["entryPoints"][0]["workgroupSize"] == (
            expected_workgroup_sizes[variant_name]
        )
        constant = artifact["specializationConstants"][0]
        assert constant["valueProvenance"] == {"kind": "runtime-override-required"}
        assert "concreteValue" not in constant
        generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
        local_size = variants[variant_name]["artifacts"][
            list(module.RMS_NORM_HOST_ENTRY_POINTS).index(source_entry)
        ]["generatedLocalSizeContract"]
        assert generated.count(local_size) == 1
        assert generated.count(specialization["generatedContract"]) == 1
    assert artifact_identities == {
        (variant["name"], source_entry)
        for variant in module.RMS_NORM_OPENGL_VARIANTS
        for source_entry in module.RMS_NORM_HOST_ENTRY_POINTS
    }


def test_rms_norm_native_toolchain_gates_compile_generated_artifacts(
    tmp_path, monkeypatch
):
    module = _load_rms_norm_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".proof"
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True)
    artifacts_by_variant = {}
    for variant in module.RMS_NORM_DIRECTX_VARIANTS:
        artifact_path = work_dir / "artifacts" / f"{variant['name']}.hlsl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text("float16_t generated_value;\n", encoding="utf-8")
        artifacts_by_variant[variant["name"]] = {
            "path": artifact_path.relative_to(mlx_root).as_posix()
        }
    opengl_artifacts = []
    for variant in module.RMS_NORM_OPENGL_VARIANTS:
        for source_entry in module.RMS_NORM_HOST_ENTRY_POINTS:
            artifact_path = (
                work_dir / "artifacts" / variant["name"] / f"{source_entry}.glsl"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("generated GLSL", encoding="utf-8")
            opengl_artifacts.append(
                {
                    "path": artifact_path.relative_to(mlx_root).as_posix(),
                    "variant": variant["name"],
                    "entryPoint": {
                        "source": source_entry,
                        "target": "main",
                        "stage": "compute",
                    },
                }
            )
    commands = []

    def fake_run_command(name, command, *, log_dir, timeout_seconds=180):
        command = list(command)
        commands.append((name, command))
        if "-Fo" in command:
            Path(command[command.index("-Fo") + 1]).write_bytes(b"DXIL")
        if "-o" in command:
            Path(command[command.index("-o") + 1]).write_bytes(b"SPIR-V")
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "stdoutPath": stdout_path,
            "stderrPath": stderr_path,
        }

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module,
        "_representative_directx_entry_point",
        lambda *_args: "CSMain",
    )
    monkeypatch.setattr(
        module,
        "shutil",
        SimpleNamespace(which=lambda name: f"/tools/{name}"),
    )
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))

    directx = module._compile_directx_variants(
        artifacts_by_variant,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=True,
    )

    assert directx["status"] == "compiled"
    assert directx["compiledArtifactCount"] == 2
    assert directx["profile"] == "cs_6_2"
    assert directx["compilerArguments"] == ["-enable-16bit-types"]
    assert directx["minimumShaderModel"] == "6.2"
    assert len(directx["runs"]) == 2
    assert [run["entryPoint"] for run in directx["runs"]] == [
        "CSMain",
        "CSMain",
    ]
    assert [run["workgroupSize"] for run in directx["runs"]] == [
        [32, 1, 1],
        [64, 1, 1],
    ]
    dxc_commands = [
        command
        for name, command in commands
        if name.startswith("compile-rmsnorm-directx")
    ]
    assert len(dxc_commands) == 2
    assert all(command[command.index("-E") + 1] == "CSMain" for command in dxc_commands)
    assert all(
        command[command.index("-T") + 1 : command.index("-E")]
        == ["cs_6_2", "-enable-16bit-types"]
        for command in dxc_commands
    )
    assert all(
        run["profile"] == "cs_6_2"
        and run["compilerArguments"] == ["-enable-16bit-types"]
        and run["minimumShaderModel"] == "6.2"
        for run in directx["runs"]
    )

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    opengl = module._compile_opengl(
        opengl_artifacts,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=True,
    )

    assert opengl["status"] == "compiled-and-validated"
    assert opengl["compiledArtifactCount"] == 24
    assert opengl["validatedArtifactCount"] == 24
    assert len(opengl["runs"]) == 24
    assert {(run["variant"], run["sourceEntryPoint"]) for run in opengl["runs"]} == {
        (variant["name"], source_entry)
        for variant in module.RMS_NORM_OPENGL_VARIANTS
        for source_entry in module.RMS_NORM_HOST_ENTRY_POINTS
    }
    assert all(run["targetEntryPoint"] == "main" for run in opengl["runs"])
    glslang_commands = [
        command
        for name, command in commands
        if name.startswith("compile-rmsnorm-opengl-")
    ]
    assert len(glslang_commands) == 24
    assert all(
        command[1:7]
        == [
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
        ]
        for command in glslang_commands
    )
    spirv_commands = [
        command
        for name, command in commands
        if name.startswith("validate-rmsnorm-opengl-")
    ]
    assert len(spirv_commands) == 24
    assert all(command[1:3] == ["--target-env", "spv1.3"] for command in spirv_commands)
    assert len(commands) == 50


def test_mlx_workflow_runs_platform_native_rms_norm_specialization_proof():
    workflow = MLX_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert 'MLX_COMMIT: "4367c73b60541ddd5a266ce4644fd93d20223b6e"' in workflow
    assert "python demos/integrations/mlx/prove_rms_norm_specialization.py" in workflow
    assert "--require-directx-toolchain" in workflow
    assert "--require-opengl-toolchain" in workflow
    assert 'if [ "$RUNNER_OS" = "Windows" ]; then' in workflow
    assert 'elif [ "$RUNNER_OS" = "Linux" ]; then' in workflow
    assert "rms-norm-specialization" in workflow
