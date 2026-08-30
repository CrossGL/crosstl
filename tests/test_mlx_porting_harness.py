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
CURRENT_MLX_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
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

    assert module.MLX_REFERENCE_COMMIT == PINNED_MLX_COMMIT
    assert module.MLX_CORPUS_COMMIT == CURRENT_MLX_COMMIT
    assert module.MLX_COMMIT == module.MLX_REFERENCE_COMMIT
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
        subgroup_width_rules={"kernels/a.metal": 32},
        index_range_assertions=assertions,
    )

    config = load_project_config(tmp_path, config_path)
    assert config.subgroup_width_rules == {"kernels/a.metal": "32"}
    assert [assertion.to_json() for assertion in config.index_range_assertions] == list(
        assertions
    )
    assert config_path.read_text(encoding="utf-8").count(
        "[[project.index_range_assertions]]"
    ) == len(assertions)


def test_project_config_writer_emits_workgroup_access_assertions(tmp_path):
    from crosstl.project import load_project_config

    module = _load_harness()
    config_path = tmp_path / "frontier.toml"
    assertions = (
        {
            "source": "mlx/backend/metal/kernels/fft.metal",
            "entry_point": "*mem_256_*",
            "function": "ReadWriter_*",
            "parameter": "crosstl_ptr_buf",
            "minimum": 0,
            "maximum": 255,
        },
    )

    module._write_project_config(
        config_path,
        include="mlx/backend/metal/kernels/fft.metal",
        targets=("opengl",),
        output_dir="generated",
        workgroup_access_assertions=assertions,
    )

    config = load_project_config(tmp_path, config_path)
    assert [
        assertion.to_json() for assertion in config.workgroup_access_assertions
    ] == [
        {
            "source": assertions[0]["source"],
            "entryPoint": assertions[0]["entry_point"],
            "function": assertions[0]["function"],
            "parameter": assertions[0]["parameter"],
            "minimum": assertions[0]["minimum"],
            "maximum": assertions[0]["maximum"],
        }
    ]
    assert config_path.read_text(encoding="utf-8").count(
        "[[project.workgroup_access_assertions]]"
    ) == len(assertions)


def test_project_config_writer_emits_entry_workgroup_size_rules(tmp_path):
    from crosstl.project import load_project_config

    module = _load_harness()
    config_path = tmp_path / "frontier.toml"
    source = "mlx/backend/metal/kernels/gemv.metal"

    module._write_project_config(
        config_path,
        include=source,
        targets=("directx",),
        output_dir="generated",
        variant_specialization_constants={
            "wide_plain": {
                "gemv_wide_has_batch": False,
                "gemv_wide_do_axpby": False,
            },
            "wide_batch_axpby": {
                "gemv_wide_has_batch": True,
                "gemv_wide_do_axpby": True,
            },
        },
        workgroup_size_rules={source: (32, "BN", "BM")},
        entry_workgroup_size_rules={source: {"gemv_wide*": (32, "k_lanes / 8", 1)}},
    )

    config = load_project_config(tmp_path, config_path)
    assert config.workgroup_size_rules == {source: ("32", "BN", "BM")}
    assert config.entry_workgroup_size_rules == {
        source: {"gemv_wide*": ("32", "k_lanes / 8", "1")}
    }
    assert config.variant_specialization_constants == {
        "wide_plain": {
            "gemv_wide_has_batch": False,
            "gemv_wide_do_axpby": False,
        },
        "wide_batch_axpby": {
            "gemv_wide_has_batch": True,
            "gemv_wide_do_axpby": True,
        },
    }


def test_project_config_writer_emits_dispatch_contracts(tmp_path):
    from crosstl.project import load_project_config

    module = _load_harness()
    config_path = tmp_path / "frontier.toml"
    contract_path = "contracts/layer_norm.dispatch.json"
    installed_contract = tmp_path / contract_path
    installed_contract.parent.mkdir(parents=True)
    installed_contract.write_bytes(
        module.MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE.read_bytes()
    )

    module._write_project_config(
        config_path,
        include=module.MLX_LAYER_NORM_SOURCE,
        targets=("directx",),
        output_dir="generated",
        dispatch_contracts=(contract_path,),
    )

    config = load_project_config(tmp_path, config_path)
    assert config.dispatch_contracts == [contract_path]
    assert f'dispatch_contracts = ["{contract_path}"]' in config_path.read_text(
        encoding="utf-8"
    )


def test_layer_norm_dispatch_contract_preparation_copies_verified_manifest(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    source_path = mlx_root / module.MLX_LAYER_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text("kernel void layer_norm_fixture() {}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MLX_LAYER_NORM_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )

    contract = module._prepare_layer_norm_dispatch_contract(
        mlx_root,
        mlx_root / ".crosstl-mlx-porting",
    )

    copied_path = mlx_root / contract["path"]
    assert (
        copied_path.read_bytes()
        == module.MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE.read_bytes()
    )
    assert contract == {
        "path": ".crosstl-mlx-porting/contracts/layer_norm.dispatch.json",
        "contentIdentity": {
            "algorithm": "sha256",
            "value": module.MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix(
                "sha256:"
            ),
        },
        "variantCount": len(module.MLX_LAYER_NORM_DISPATCH_VARIANTS),
    }


def test_layer_norm_dispatch_contract_accepts_crlf_checkout(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    source_path = mlx_root / module.MLX_LAYER_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text("kernel void layer_norm_fixture() {}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MLX_LAYER_NORM_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    contract_source = tmp_path / "layer_norm.dispatch.json"
    normalized = module.MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE.read_text(
        encoding="utf-8"
    )
    contract_source.write_bytes(normalized.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(
        module,
        "MLX_LAYER_NORM_DISPATCH_CONTRACT_SOURCE",
        contract_source,
    )

    contract = module._prepare_layer_norm_dispatch_contract(
        mlx_root,
        mlx_root / ".crosstl-mlx-porting",
    )

    copied_path = mlx_root / contract["path"]
    assert copied_path.read_bytes() == contract_source.read_bytes()
    assert contract["contentIdentity"] == {
        "algorithm": "sha256",
        "value": module.MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix(
            "sha256:"
        ),
    }


def test_rms_norm_dispatch_contract_preparation_copies_verified_manifest(
    tmp_path, monkeypatch
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    source_path = mlx_root / module.MLX_RMS_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text("kernel void rms_norm_fixture() {}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MLX_RMS_NORM_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )

    contract = module._prepare_rms_norm_dispatch_contract(
        mlx_root,
        mlx_root / ".crosstl-mlx-porting",
    )

    copied_path = mlx_root / contract["path"]
    assert (
        copied_path.read_bytes()
        == module.MLX_RMS_NORM_DISPATCH_CONTRACT_SOURCE.read_bytes()
    )
    assert contract == {
        "path": ".crosstl-mlx-porting/contracts/rms_norm.dispatch.json",
        "contentIdentity": {
            "algorithm": "sha256",
            "value": module.MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix(
                "sha256:"
            ),
        },
        "variantCount": len(module.MLX_RMS_NORM_DISPATCH_VARIANTS),
    }


def test_rms_norm_dispatch_contract_accepts_crlf_checkout(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    source_path = mlx_root / module.MLX_RMS_NORM_SOURCE
    source_path.parent.mkdir(parents=True)
    source_path.write_text("kernel void rms_norm_fixture() {}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "MLX_RMS_NORM_SHA256",
        hashlib.sha256(source_path.read_bytes()).hexdigest(),
    )
    contract_source = tmp_path / "rms_norm.dispatch.json"
    normalized = module.MLX_RMS_NORM_DISPATCH_CONTRACT_SOURCE.read_text(
        encoding="utf-8"
    )
    contract_source.write_bytes(normalized.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.setattr(
        module,
        "MLX_RMS_NORM_DISPATCH_CONTRACT_SOURCE",
        contract_source,
    )

    contract = module._prepare_rms_norm_dispatch_contract(
        mlx_root,
        mlx_root / ".crosstl-mlx-porting",
    )

    copied_path = mlx_root / contract["path"]
    assert copied_path.read_bytes() == contract_source.read_bytes()
    assert contract["contentIdentity"] == {
        "algorithm": "sha256",
        "value": module.MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix("sha256:"),
    }


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
        template_prefix = "vjp_rms" if entry_point.startswith("vjp_rms") else "rms"
        template_suffix = (
            "looped_fixture" if "_looped" in entry_point else "single_row_fixture"
        )
        template = f"{template_prefix}_{template_suffix}"
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
    uint index [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  float value = float(TAG + SIMD_SIZE);
  value = simd_sum(value);
  value = simd_sum(value + float(simd_lane_id + simd_group_id));
  output[index] = value;
}

template <int TAG>
[[kernel]] void rms_looped_fixture(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  float value = float(TAG + SIMD_SIZE + lsize);
  value = simd_sum(value);
  value = simd_sum(value + float(simd_lane_id + simd_group_id));
  output[index] = value;
}

template <int TAG>
[[kernel]] void vjp_rms_single_row_fixture(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  float value =
      (has_w ? 1.0f : 0.0f) + float(TAG + SIMD_SIZE + simd_lane_id + simd_group_id);
  value = simd_sum(value);
  value = simd_sum(value);
  value = simd_sum(value);
  value = simd_sum(value);
  output[index] = value;
}

template <int TAG>
[[kernel]] void vjp_rms_looped_fixture(
    device float* output [[buffer(0)]],
    uint index [[thread_position_in_grid]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int SIMD_SIZE = 32;
  float value = (has_w ? 1.0f : 0.0f) +
      float(TAG + SIMD_SIZE + lsize + simd_lane_id + simd_group_id);
  value = simd_sum(value);
  value = simd_sum(value);
  value = simd_sum(value);
  value = simd_sum(value);
  output[index] = value;
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
            "translatedCount": module.FULL_CORPUS_EXPECTED_UNIT_COUNT - 1,
            "failedCount": 1,
        }
        for target in module.FULL_CORPUS_TARGETS
    }
    diagnostics_by_code = {
        contract["diagnosticCode"]: 1
        for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items()
        if target in module.FULL_CORPUS_TARGETS
    }
    diagnostics_by_code["project.validate.failed-artifact"] = (
        module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    )
    missing_capability_counts = {
        contract["missingCapability"]: 1
        for target, contract in module.MLX_FENCE_TARGET_CONTRACTS.items()
        if target in module.FULL_CORPUS_TARGETS
    }
    missing_capability_counts["batch.translation"] = (
        module.FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
    )
    diagnostics = []
    artifacts = []
    extensions = {"directx": ".hlsl", "opengl": ".glsl", "vulkan": ".spvasm"}
    for target in module.FULL_CORPUS_TARGETS:
        contract = module.MLX_FENCE_TARGET_CONTRACTS[target]
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
            "translatedCount": module.FULL_CORPUS_EXPECTED_UNIT_COUNT - 2,
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
        "unitCount": module.FULL_CORPUS_EXPECTED_UNIT_COUNT,
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


def _write_full_corpus_checkpoint(module, path, *, state="running"):
    from crosstl.project import ProjectTranslationCheckpointRecorder

    target_suffixes = {
        "directx": "hlsl",
        "opengl": "glsl",
        "vulkan": "spvasm",
    }
    first_jobs = [
        {
            "source": module.MLX_ARANGE_SOURCE,
            "target": target,
            "path": (
                f".crosstl-mlx-porting/out-full-corpus/{target}/"
                f"mlx/backend/metal/kernels/arange.{target_suffixes[target]}"
            ),
        }
        for target in module.FULL_CORPUS_TARGETS
    ]
    active = {
        "source": module.MLX_ARG_REDUCE_SOURCE,
        "target": "directx",
        "path": (
            ".crosstl-mlx-porting/out-full-corpus/directx/"
            "mlx/backend/metal/kernels/arg_reduce.hlsl"
        ),
    }
    jobs = [*first_jobs, active]
    for index in range(
        len(jobs),
        module.FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
    ):
        target = module.FULL_CORPUS_TARGETS[index % len(module.FULL_CORPUS_TARGETS)]
        jobs.append(
            {
                "source": f"mlx/backend/metal/kernels/pending_{index}.metal",
                "target": target,
                "path": (
                    f".crosstl-mlx-porting/out-full-corpus/{target}/"
                    f"mlx/backend/metal/kernels/pending_{index}."
                    f"{target_suffixes[target]}"
                ),
            }
        )
    recorder = ProjectTranslationCheckpointRecorder(
        path,
        {"root": "/mlx", "targets": list(module.FULL_CORPUS_TARGETS)},
        jobs,
        started_at=100,
        initial_diagnostics=[
            {
                "severity": "warning",
                "code": "project.translate.progress",
            }
        ],
    )
    for job in first_jobs:
        recorder.record_completion(
            job,
            [
                {
                    "source": job["source"],
                    "target": job["target"],
                    "path": job["path"],
                    "status": "translated",
                }
            ],
            [],
        )
    if state == "interrupted":
        return recorder.write_interrupted(active, RuntimeError("timed out"))
    return recorder.write_running(active)


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
    assert frontier["artifacts"] == 35
    assert frontier["status"] == (
        "target-split-with-bounded-dispatch-and-pending-contracts"
    )
    assert frontier["scope"] == "target-split-frontier"
    assert frontier["translated_artifacts"] == 32
    assert frontier["failed_artifacts"] == 3
    assert frontier["target_artifacts"] == {
        "directx": {"translated": 21, "failed": 3},
        "vulkan": {"translated": 11, "failed": 0},
    }
    assert frontier["semantic_readiness_status"] == "not-established"
    assert frontier["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert frontier["host_dispatch_import_resolved_by"] == (
        module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    )
    assert frontier["pending_host_dispatch_sources"] == list(
        module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
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
    assert arg_reduce["artifact_records"] == len(module.MLX_REFERENCE_TARGETS)
    assert arg_reduce["translated_artifacts"] == 1
    assert arg_reduce["failed_artifacts"] == 2
    assert arg_reduce["translated_targets"] == ["vulkan"]
    assert arg_reduce["workgroup_blocked_targets"] == ["directx", "opengl"]
    assert arg_reduce["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert arg_reduce["host_dispatch_import_resolved_by"] == (
        module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    )
    assert arg_reduce["bounded_native_runtime_status"] == (
        module.MLX_ARG_REDUCE_NATIVE_RUNTIME_EVIDENCE["status"]
    )
    assert arg_reduce["bounded_entry_points"] == [
        "argmin_float32",
        "argmax_float32",
    ]
    assert arg_reduce["bounded_workgroup_size"] == [32, 1, 1]
    assert arg_reduce["bounded_dispatch_workgroup_count"] == [1, 2, 1]
    assert (
        arg_reduce["bounded_native_runtime_evidence"]
        == "arg_reduce_native_runtime_status"
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

    opengl_logsumexp = expected_gaps["opengl_logsumexp_dispatch_status"]
    assert opengl_logsumexp["status"] == (
        "translated-toolchain-validated-with-axis32-software-runtime"
    )
    assert opengl_logsumexp["commit"] == module.MLX_CORPUS_COMMIT
    assert opengl_logsumexp["source"] == module.MLX_LOGSUMEXP_SOURCE
    assert opengl_logsumexp["source_sha256"] == module.MLX_LOGSUMEXP_SHA256
    assert opengl_logsumexp["target"] == "opengl"
    assert opengl_logsumexp["dispatch_contract"] == {
        "path": (
            "demos/integrations/mlx/contracts/" "logsumexp.native-loader.dispatch.json"
        ),
        "content_identity": (
            module.MLX_LOGSUMEXP_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY
        ),
        "workload_count": len(module.MLX_LOGSUMEXP_DISPATCH_VARIANTS),
        "workgroup_sizes": [
            variant["workgroupSize"]
            for variant in module.MLX_LOGSUMEXP_DISPATCH_VARIANTS.values()
        ],
        "subgroup_width": 32,
    }
    assert opengl_logsumexp["historical_dispatch_contract"] == {
        "commit": module.MLX_COMMIT,
        "path": "demos/integrations/mlx/contracts/logsumexp.dispatch.json",
        "content_identity": module.MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY,
    }
    assert opengl_logsumexp["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 2,
        "translated_count": 2,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "entry_point": "block_logsumexp_float32",
        "target_entry_point": "main",
    }
    assert opengl_logsumexp["toolchain_validation"]["status"] == "passed"
    assert opengl_logsumexp["toolchain_validation"]["compiled_artifact_count"] == 2
    assert opengl_logsumexp["toolchain_validation"]["validated_artifact_count"] == 2
    assert opengl_logsumexp["runtime_compatibility"] == {
        "adapter_preflight_included": True,
        "compatible_width_device_test": {
            "artifact_specialized_to_reported_width": True,
            "width_source": "GL_SUBGROUP_SIZE_KHR",
        },
        "hardware_logsumexp_required_width": 32,
        "hardware_runtime_execution_attempted": False,
        "software_runtime_execution_attempted": True,
        "software_executed_workloads": ["block-float32-axis-32"],
        "remaining_hardware_only_workloads": ["block-float32-axis-1025"],
        "blocked_by": "https://github.com/CrossGL/crosstl/issues/1894",
    }
    software_logsumexp = opengl_logsumexp["current_corpus_software_runtime"]
    assert software_logsumexp == (module.MLX_OPENGL_LOGSUMEXP_SOFTWARE_RUNTIME_EVIDENCE)
    assert software_logsumexp["generated_glsl"] == {
        "sha256": "813762d4535fdd693ca0a48c3c3f5dc79f6cc298050faae6e180d3cc9f1d60e5",
        "size_bytes": 4676,
    }
    assert software_logsumexp["software_subgroup"]["operations"] == [
        "WaveActiveMax(float)",
        "WaveActiveSum(float)",
    ]
    assert (
        software_logsumexp["software_subgroup"]["control_barrier_instruction_count"]
        == 10
    )
    assert software_logsumexp["runtime_package"]["ready_load_unit_count"] == 1
    assert software_logsumexp["runtime_execution"]["status"] == "required-on-ci"
    assert software_logsumexp["remaining_scope"] == {
        "workload": "block-float32-axis-1025",
        "axis_size": 1025,
        "workgroup_size": [288, 1, 1],
        "status": "hardware-subgroup-only",
        "reason": "software-mode-requires-exactly-one-32-thread-subgroup",
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1894",
    }
    assert opengl_logsumexp["runtime_integration_included"] is True
    assert opengl_logsumexp["selected_workload_numerical_parity_verified"] is True
    assert opengl_logsumexp["full_mlx_test_suite_included"] is False
    assert opengl_logsumexp["numerical_parity_claimed"] is False
    assert opengl_logsumexp["runtime_parity_claimed"] is False

    opengl_quantized = expected_gaps["opengl_quantized_frontier_status"]
    assert opengl_quantized == module.MLX_OPENGL_QUANTIZED_FRONTIER_EVIDENCE
    assert opengl_quantized["commit"] == module.MLX_CORPUS_COMMIT
    assert opengl_quantized["artifact_emitted"] is True
    assert opengl_quantized["native_validation_attempted"] is True
    assert opengl_quantized["native_validation_status"] == "passed"
    assert opengl_quantized["generated_glsl"] == {
        "sha256": "e4d8e5931bfc93f81e2c3686c102a1d676c9a3dcdfd6447e90918aa7581beecb",
        "size_bytes": 6642,
    }
    assert opengl_quantized["software_subgroup"] == {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "activation": "explicit-target-scoped",
        "operations": [
            "WaveActiveMin(float)",
            "WaveActiveMax(float)",
            "WaveShuffleDown(uint,int)",
        ],
        "artifact_marker": "CROSSTL_SOFTWARE_SUBGROUP_WIDTH",
        "control_barrier_instruction_count": 8,
        "group_non_uniform_instruction_count": 0,
        "hardware_subgroup_extensions_emitted": False,
        "hardware_subgroup_marker_emitted": False,
        "hardware_subgroup_execution_metadata_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    }
    assert opengl_quantized["runtime_execution"]["status"] == "passed"
    assert opengl_quantized["runtime_execution"]["outputs"] == {
        "out_Buffer": {
            "dtype": "uint32",
            "shape": [8],
            "values": [27, 27, 27, 27, 27, 27, 27, 27],
        },
        "scalesBuffer": {
            "dtype": "float32",
            "shape": [1],
            "values": [-1.0],
        },
        "biasesBuffer": {
            "dtype": "float32",
            "shape": [1],
            "values": [3.0],
        },
    }
    assert opengl_quantized["runtime_execution_attempted"] is True
    assert opengl_quantized["runtime_integration_included"] is True
    assert opengl_quantized["mlx_host_runtime_integration_included"] is False
    assert opengl_quantized["numerical_parity_claimed"] is True
    assert opengl_quantized["runtime_parity_claimed"] is True

    directx = expected_gaps["directx_toolchain_status"]
    assert directx["compiler"] == {"name": "dxc", "version": "v1.9.2602.24"}
    assert directx["warning_evidence"] == (
        module.MLX_DIRECTX_TOOLCHAIN_WARNING_EVIDENCE
    )
    assert directx["warning_evidence"] == {
        "status": "warning-clean",
        "validatedRunCount": module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT,
        "warningRunCount": 0,
        "observedWarningCount": 0,
        "uniqueContractCount": 0,
        "contracts": [],
    }
    assert directx["contextual_narrowing_evidence"] == (
        module.MLX_DIRECTX_CONTEXTUAL_NARROWING_EVIDENCE
    )
    contextual_narrowing = directx["contextual_narrowing_evidence"]
    assert contextual_narrowing["profile"] == "cs_6_2"
    assert contextual_narrowing["compilerArguments"] == ["-enable-16bit-types"]
    assert [
        contract["classification"]
        for contract in contextual_narrowing["resolvedWarningContracts"]
    ] == [
        "native-int16-destination-conversion",
        "uint64-local-destination-conversion",
    ]
    assert [
        contract["validatedEntryPointCount"]
        for contract in contextual_narrowing["resolvedWarningContracts"]
    ] == [11, 18]
    assert all(
        contract["observedWarningCount"] == 0
        for contract in contextual_narrowing["resolvedWarningContracts"]
    )
    assert all(
        contract["warningsAsErrors"] is True
        for contract in contextual_narrowing["resolvedWarningContracts"]
    )
    native_arithmetic = directx["native_16_bit_arithmetic_evidence"]
    assert native_arithmetic == module.MLX_DIRECTX_NATIVE_16_BIT_ARITHMETIC_EVIDENCE
    assert native_arithmetic["validatedEntryPointCount"] == 11
    assert native_arithmetic["generatedSourceLine"] == (
        "arangefloat16_out[index] = (arangefloat16_start + "
        "(float16_t(index) * arangefloat16_step));"
    )
    assert native_arithmetic["observedWarningCount"] == 0
    assert native_arithmetic["warningsAsErrors"] is True
    assert native_arithmetic["runtimeExecutionAttempted"] is False
    assert native_arithmetic["numericalParityClaimed"] is False
    assert directx["selected_quantized_frontier"] == (
        module.MLX_DIRECTX_QUANTIZED_FRONTIER_EVIDENCE
    )
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
    layer_norm = directx["layer_norm_dispatch_frontier"]
    assert layer_norm["status"] == "translated-dxc-validated"
    assert layer_norm["source"] == module.MLX_LAYER_NORM_SOURCE
    assert layer_norm["source_sha256"] == module.MLX_LAYER_NORM_SHA256
    assert layer_norm["dispatch_contract"] == {
        "path": "demos/integrations/mlx/contracts/layer_norm.dispatch.json",
        "normalized_sha256": module.MLX_LAYER_NORM_DISPATCH_NORMALIZED_SHA256,
        "content_identity": module.MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY,
        "variant_count": len(module.MLX_LAYER_NORM_DISPATCH_VARIANTS),
        "resolved_issue": module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
    }
    assert layer_norm["artifact_count"] == len(module.MLX_LAYER_NORM_DISPATCH_VARIANTS)
    assert set(layer_norm["variants"]) == set(module.MLX_LAYER_NORM_DISPATCH_VARIANTS)
    assert layer_norm["dxc_validated_artifact_count"] == 2
    assert layer_norm["runtime_execution_attempted"] is False
    assert layer_norm["numerical_parity_claimed"] is False
    logsumexp = directx["logsumexp_dispatch_frontier"]
    assert logsumexp["status"] == "translated-dxc-validated"
    assert logsumexp["source"] == module.MLX_LOGSUMEXP_SOURCE
    assert logsumexp["source_sha256"] == module.MLX_LOGSUMEXP_SHA256
    assert logsumexp["test_sources"] == [
        "python/tests/test_ops.py::test_logsumexp",
        "python/tests/test_autograd.py::test_logsumexp_grad",
    ]
    assert logsumexp["dispatch_contract"] == {
        "path": "demos/integrations/mlx/contracts/logsumexp.dispatch.json",
        "normalized_sha256": module.MLX_LOGSUMEXP_DISPATCH_NORMALIZED_SHA256,
        "content_identity": module.MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY,
        "variant_count": len(module.MLX_LOGSUMEXP_DISPATCH_VARIANTS),
        "resolved_issue": module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
    }
    assert logsumexp["artifact_count"] == len(module.MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    assert set(logsumexp["variants"]) == set(module.MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    assert logsumexp["dxc_validated_artifact_count"] == 2
    for workload_id, expected in module.MLX_LOGSUMEXP_DISPATCH_VARIANTS.items():
        variant = logsumexp["variants"][workload_id]
        assert variant["entry_point"] == expected["entryPoint"]
        assert variant["artifact_id"] == expected["artifactId"]
        assert variant["dispatch_variant_id"] == expected["dispatchVariantId"]
        assert variant["inputs"] == expected["inputs"]
        assert variant["workgroup_size"] == expected["workgroupSize"]
        assert variant["dispatch_workgroup_count"] == (
            expected["dispatchWorkgroupCount"]
        )
        assert variant["subgroup_width"] == 32
        assert variant["subgroup_width_enforcement"] == "WaveSize(32)"
        assert len(variant["generated_hlsl"]["sha256"]) == 64
        assert variant["generated_hlsl"]["size_bytes"] > 0
    assert logsumexp["runtime_execution_attempted"] is False
    assert logsumexp["numerical_parity_claimed"] is False
    rms_norm = directx["rms_norm_dispatch_frontier"]
    assert rms_norm["status"] == "translated-dxc-validated"
    assert rms_norm["source"] == module.MLX_RMS_NORM_SOURCE
    assert rms_norm["source_sha256"] == module.MLX_RMS_NORM_SHA256
    assert rms_norm["test_sources"] == [
        "python/tests/test_fast.py::test_rms_norm",
        "python/tests/test_fast.py::test_rms_norm_grad",
    ]
    assert rms_norm["dispatch_contract"] == {
        "path": "demos/integrations/mlx/contracts/rms_norm.dispatch.json",
        "normalized_sha256": module.MLX_RMS_NORM_DISPATCH_NORMALIZED_SHA256,
        "content_identity": module.MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY,
        "variant_count": len(module.MLX_RMS_NORM_DISPATCH_VARIANTS),
        "resolved_issue": module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
    }
    assert rms_norm["artifact_count"] == len(module.MLX_RMS_NORM_DISPATCH_VARIANTS)
    assert set(rms_norm["variants"]) == set(module.MLX_RMS_NORM_DISPATCH_VARIANTS)
    assert rms_norm["dxc_validated_artifact_count"] == 12
    for workload_id, expected in module.MLX_RMS_NORM_DISPATCH_VARIANTS.items():
        variant = rms_norm["variants"][workload_id]
        assert variant["entry_point"] == expected["entryPoint"]
        assert variant["artifact_id"] == expected["artifactId"]
        assert variant["dispatch_variant_id"] == expected["dispatchVariantId"]
        assert variant["inputs"] == expected["inputs"]
        assert variant["workgroup_size"] == expected["workgroupSize"]
        assert variant["dispatch_workgroup_count"] == (
            expected["dispatchWorkgroupCount"]
        )
        assert variant["subgroup_width"] == 32
        assert variant["subgroup_width_enforcement"] == "WaveSize(32)"
        assert variant["specialization_constants"] == (
            expected["specializationConstants"]
        )
        assert len(variant["generated_hlsl"]["normalized_sha256"]) == 64
        assert variant["generated_hlsl"]["size_bytes"] > 0
    assert rms_norm["runtime_execution_attempted"] is False
    assert rms_norm["numerical_parity_claimed"] is False
    assert directx["native_runtime_executed"] is False
    assert directx["runtime_parity_claimed"] is False

    pointer_reinterpretation = expected_gaps["pointer_reinterpretation_status"]
    assert pointer_reinterpretation["status"] == "partial"
    assert pointer_reinterpretation["targets"] == list(module.MLX_REFERENCE_TARGETS)
    assert pointer_reinterpretation["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
        "https://github.com/CrossGL/crosstl/issues/1544",
    ]
    assert set(pointer_reinterpretation["remaining_pointer_cases_blocked_by"]) == {
        "https://github.com/CrossGL/crosstl/issues/1546"
    }

    callback = expected_gaps["captured_callback_status"]
    assert callback["status"] == "partial"
    assert callback["targets"] == list(module.MLX_REFERENCE_TARGETS)
    assert callback["remaining_callback_helpers_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1554"
    ]
    assert callback["next_kernel_blocked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1574",
    ]

    compile_time_loop = expected_gaps["compile_time_loop_status"]
    assert compile_time_loop["status"] == "partial"
    assert compile_time_loop["targets"] == list(module.MLX_REFERENCE_TARGETS)
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
    assert struct_scoped_cast_alias["targets"] == list(module.MLX_REFERENCE_TARGETS)
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
    assert function_local_alias["targets"] == list(module.MLX_REFERENCE_TARGETS)
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
    assert generic_member_call["targets"] == list(module.MLX_REFERENCE_TARGETS)
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
    reference_artifact_count = module.EXPECTED_METAL_KERNEL_COUNT * len(
        module.MLX_REFERENCE_TARGETS
    )
    reference_fence_failure_count = len(module.MLX_REFERENCE_TARGETS)
    assert full_corpus["artifacts"] == reference_artifact_count
    assert full_corpus["expected_translated_artifacts"] == (
        reference_artifact_count - reference_fence_failure_count
    )
    assert full_corpus["expected_fence_failures"] == (reference_fence_failure_count)
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


def test_cooperative_matrix_fragment_mapping_contract_tracks_pinned_source():
    contract = json.loads(
        (
            ROOT
            / "demos"
            / "integrations"
            / "mlx"
            / "contracts"
            / "cooperative-matrix-fragment-mapping.json"
        ).read_text(encoding="utf-8")
    )
    matrix_type = (
        "CooperativeMatrix<float, 8, 8, subgroup, unspecified, unspecified, "
        "metal_thread_elements, 32, 2, metal_thread_elements_reference_view, "
        "tile_4x4_row_pair, mlx_steel_BaseMMAFrag_get_coord>"
    )

    assert contract == {
        "kind": "crosstl-cooperative-matrix-fragment-mapping-contract",
        "schemaVersion": 1,
        "id": "mlx-steel-base-mma-frag-8x8-row-pair",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx",
            "commit": PINNED_MLX_COMMIT,
            "source": "mlx/backend/metal/kernels/steel/gemm/mma.h",
            "sourceLines": {"start": 37, "end": 54},
            "sourceSymbol": "mlx::steel::BaseMMAFrag<T, 8, 8>::get_coord",
            "sourceUrl": (
                "https://github.com/ml-explore/mlx/blob/"
                f"{PINNED_MLX_COMMIT}/mlx/backend/metal/kernels/steel/gemm/"
                "mma.h#L37-L54"
            ),
        },
        "mapping": {
            "name": "tile_4x4_row_pair",
            "matrixShape": [8, 8],
            "subgroupSize": 32,
            "elementsPerLane": 2,
            "coordinateOrder": ["row", "column"],
            "formula": {
                "qid": "lane / 4",
                "row": "(qid & 4) + ((lane / 2) % 4)",
                "columnBase": "(qid & 2) * 2 + (lane % 2) * 2",
                "element": "[row, columnBase + elementIndex]",
                "elementIndexRange": [0, 1],
            },
            "laneCoordinates": [
                [[0, 0], [0, 1]],
                [[0, 2], [0, 3]],
                [[1, 0], [1, 1]],
                [[1, 2], [1, 3]],
                [[2, 0], [2, 1]],
                [[2, 2], [2, 3]],
                [[3, 0], [3, 1]],
                [[3, 2], [3, 3]],
                [[0, 4], [0, 5]],
                [[0, 6], [0, 7]],
                [[1, 4], [1, 5]],
                [[1, 6], [1, 7]],
                [[2, 4], [2, 5]],
                [[2, 6], [2, 7]],
                [[3, 4], [3, 5]],
                [[3, 6], [3, 7]],
                [[4, 0], [4, 1]],
                [[4, 2], [4, 3]],
                [[5, 0], [5, 1]],
                [[5, 2], [5, 3]],
                [[6, 0], [6, 1]],
                [[6, 2], [6, 3]],
                [[7, 0], [7, 1]],
                [[7, 2], [7, 3]],
                [[4, 4], [4, 5]],
                [[4, 6], [4, 7]],
                [[5, 4], [5, 5]],
                [[5, 6], [5, 7]],
                [[6, 4], [6, 5]],
                [[6, 6], [6, 7]],
                [[7, 4], [7, 5]],
                [[7, 6], [7, 7]],
            ],
        },
        "translationEvidence": {
            "source": "mlx/backend/metal/kernels/fp_quantized.metal",
            "elementType": "float",
            "fragmentLayout": "metal_thread_elements",
            "fragmentProvenance": "metal_thread_elements_reference_view",
            "fragmentMapping": "tile_4x4_row_pair",
            "fragmentMappingProvenance": "mlx_steel_BaseMMAFrag_get_coord",
            "cooperativeMatrixType": matrix_type,
            "materializedSpecializationCount": 604,
            "unsupportedSpecializationCount": 0,
            "sourceContracts": {
                "cooperativeMatrixTypeCount": 16,
                "completeTwelveFieldContractCount": 16,
            },
            "operationContracts": {
                "operationCount": 8,
                "element": {
                    "count": 7,
                    "expressionType": "float",
                    "matrixResultTypeCount": 0,
                    "matrixResultTypeRequired": False,
                },
                "multiplyAccumulate": {
                    "count": 1,
                    "expressionType": matrix_type,
                    "resultType": matrix_type,
                    "completeTwelveFieldContractCount": 1,
                    "preservesAccumulatorRepresentation": True,
                },
                "resolvedIssue": "https://github.com/CrossGL/crosstl/issues/1610",
            },
            "softwareLowering": {
                "status": "lane-local-foundation",
                "mode": "explicit-opt-in",
                "targets": ["directx", "opengl"],
                "supportedOperations": [
                    "type-representation",
                    "element",
                    "negate",
                    "elementwise-add",
                    "elementwise-subtract",
                    "elementwise-multiply",
                ],
                "failClosedOperations": [
                    "load",
                    "store",
                    "multiply",
                    "multiply-accumulate",
                ],
                "defaultEnabled": False,
                "projectConfigurationWired": False,
            },
            "replay": {
                "scope": "full-source-exact-high-budget",
                "mode": "explicit-opt-in",
                "elapsedSeconds": 765.091,
                "codegenFactoryOverrides": [
                    "crosstl.project.pipeline.get_codegen",
                    "crosstl._crosstl.get_codegen",
                ],
                "projectConfigurationWired": False,
            },
            "resolvedPrivatePointerContracts": [
                {
                    "target": "directx",
                    "contract": "concrete-ternary-condition-selection",
                    "resolvedIssue": "https://github.com/CrossGL/crosstl/issues/1826",
                },
                {
                    "target": "opengl",
                    "contract": (
                        "compile-time-global-preparation-before-"
                        "private-pointer-analysis"
                    ),
                    "resolvedIssue": "https://github.com/CrossGL/crosstl/issues/1826",
                },
            ],
            "targets": {
                "directx": {
                    "boundaryDiagnostic": (
                        "project.translate.directx-private-pointer-unsupported"
                    ),
                    "missingCapability": "directx.private-pointer-parameter-lowering",
                    "privatePointer": {
                        "function": "qouter_float_2_8_4",
                        "parameter": "w",
                        "reason": "missing-view-backing",
                        "sourceExpression": "(thread uint8_t*)&w_local",
                        "source": "mlx/backend/metal/kernels/fp_quantized.h",
                    },
                    "advancedPast": {
                        "function": "qdot_float_16_4",
                        "parameter": "x_thread",
                        "reason": "unprovable-view-offset",
                    },
                    "blockedBy": "https://github.com/CrossGL/crosstl/issues/1546",
                    "artifactEmitted": False,
                },
                "opengl": {
                    "boundaryDiagnostic": (
                        "project.translate.opengl-private-pointer-unsupported"
                    ),
                    "missingCapability": "opengl.private-pointer-parameter-lowering",
                    "privatePointer": {
                        "function": "qouter_float_2_8_4",
                        "parameter": "w",
                        "reason": "missing-view-backing",
                        "sourceExpression": "(thread uint8_t*)&w_local",
                        "source": "mlx/backend/metal/kernels/fp_quantized.h",
                    },
                    "advancedPast": {
                        "function": "qdot_float_16_4",
                        "parameter": "x_thread",
                        "reason": "unprovable-view-offset",
                    },
                    "blockedBy": "https://github.com/CrossGL/crosstl/issues/1546",
                    "artifactEmitted": False,
                },
            },
            "reducedRuntimeProof": {
                "status": "native-execution-wired-not-locally-executed",
                "source": (
                    "tests/fixtures/runtime_verification/private_pointer_partition/"
                    "private_pointer_partition.cgl"
                ),
                "expectedReadback": [100, 101, 102, 103, 200, 201, 202, 203],
                "targets": {
                    "directx": {
                        "ciPlatform": "windows",
                        "runtime": "direct3d",
                        "nativeExecutionWired": True,
                    },
                    "opengl": {
                        "ciPlatform": "linux",
                        "runtime": "opengl",
                        "nativeExecutionWired": True,
                    },
                },
                "localExecutionAttempted": False,
                "localExecutionVerified": False,
                "mlxRuntimeIncluded": False,
            },
        },
        "scope": {
            "sourceCoordinateMappingVerified": True,
            "expressionResultInferenceImplemented": True,
            "laneLocalSoftwareLoweringFoundationImplemented": True,
            "targetPolicyImplemented": False,
            "fullSoftwareFallbackImplemented": False,
            "defaultTargetBehavior": "fail-closed",
            "projectConfigurationWired": False,
            "runtimeIntegrationIncluded": False,
            "numericalParityVerified": False,
            "resolvedIssues": [
                "https://github.com/CrossGL/crosstl/issues/1610",
                "https://github.com/CrossGL/crosstl/issues/1823",
                "https://github.com/CrossGL/crosstl/issues/1824",
                "https://github.com/CrossGL/crosstl/issues/1826",
            ],
            "trackedIssues": [
                "https://github.com/CrossGL/crosstl/issues/1602",
                "https://github.com/CrossGL/crosstl/issues/1820",
                "https://github.com/CrossGL/crosstl/issues/1546",
            ],
        },
    }

    from crosstl.translator.cooperative_matrix import (
        get_cooperative_matrix_fragment_mapping,
    )

    mapping = get_cooperative_matrix_fragment_mapping(
        contract["mapping"]["name"],
        *contract["mapping"]["matrixShape"],
        contract["mapping"]["subgroupSize"],
        contract["mapping"]["elementsPerLane"],
    )
    assert mapping is not None
    assert contract["mapping"]["laneCoordinates"] == [
        [list(coordinate) for coordinate in lane] for lane in mapping.lane_coordinates
    ]


def test_fp_quantized_contextual_materialization_evidence_tracks_current_boundary():
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    issue = "https://github.com/CrossGL/crosstl/issues/1479"
    dependent_value_issue = "https://github.com/CrossGL/crosstl/issues/1490"
    directx_issue = "https://github.com/CrossGL/crosstl/issues/1538"
    opengl_issue = "https://github.com/CrossGL/crosstl/issues/1491"
    source_specialization_issue = "https://github.com/CrossGL/crosstl/issues/1809"
    constructor_issue = "https://github.com/CrossGL/crosstl/issues/1810"
    indexed_alias_issue = "https://github.com/CrossGL/crosstl/issues/1811"
    const_member_issue = "https://github.com/CrossGL/crosstl/issues/1812"
    member_array_issue = "https://github.com/CrossGL/crosstl/issues/1813"
    pointer_cast_issue = "https://github.com/CrossGL/crosstl/issues/1814"
    whole_fragment_issue = "https://github.com/CrossGL/crosstl/issues/1815"
    fragment_contract_issue = "https://github.com/CrossGL/crosstl/issues/1816"
    cooperative_matrix_issue = "https://github.com/CrossGL/crosstl/issues/1602"
    operation_result_type_issue = "https://github.com/CrossGL/crosstl/issues/1610"
    coordinate_mapping_issue = "https://github.com/CrossGL/crosstl/issues/1820"
    opengl_analyzer_issue = "https://github.com/CrossGL/crosstl/issues/1823"
    source_materialization_issue = "https://github.com/CrossGL/crosstl/issues/1824"
    partition_view_issue = "https://github.com/CrossGL/crosstl/issues/1826"
    byte_reinterpretation_issue = "https://github.com/CrossGL/crosstl/issues/1546"
    local_typedef_issue = "https://github.com/CrossGL/crosstl/issues/1567"
    directx_workgroup_issue = "https://github.com/CrossGL/crosstl/issues/1518"
    opengl_aggregate_issue = "https://github.com/CrossGL/crosstl/issues/1544"
    opengl_workgroup_issue = "https://github.com/CrossGL/crosstl/issues/1671"
    branch_analysis_issue = "https://github.com/CrossGL/crosstl/issues/1829"
    resolved_issue = "https://github.com/CrossGL/crosstl/issues/1807"

    status = expected_gaps["fp_quantized_contextual_materialization_status"]

    assert status == {
        "status": "blocked-at-target-workgroup-pointer-boundaries",
        "source": "mlx/backend/metal/kernels/fp_quantized.metal",
        "repository_commit": PINNED_MLX_COMMIT,
        "targets": ["directx", "opengl"],
        "translation_mode": "full-source-exact-high-budget",
        "replay": {
            "mode": "explicit-opt-in",
            "codegen_factory_overrides": [
                "crosstl.project.pipeline.get_codegen",
                "crosstl._crosstl.get_codegen",
            ],
            "project_configuration_wired": False,
            "intermediate_runs": [
                {
                    "translator_commit": "16d4df7b1bca4412350410a94fd4cc01f15d97a9",
                    "status": "blocked-at-symbolic-local-struct-extent",
                    "diagnostic_code": "project.translate.metal-local-type-unresolved",
                    "local_type": "vec_w",
                    "extent_expression": "tn * bytes_per_pack",
                    "tracked_by": local_typedef_issue,
                    "target_runs": {
                        "directx": {"elapsed_seconds": 292.953},
                        "opengl": {"elapsed_seconds": 286.175},
                    },
                },
                {
                    "translator_commit": "0509feabb863b988649444cd5e559eeb45eb8b54",
                    "status": "advanced-symbolic-local-struct-extent",
                    "local_type": "vec_w",
                    "extent_expression": "(2) * bytes_per_pack",
                    "resolved_values": {"tn": 2},
                    "tracked_by": local_typedef_issue,
                    "target_runs": {
                        "directx": {"elapsed_seconds": 286.54},
                        "opengl": {"elapsed_seconds": 283.718},
                    },
                },
                {
                    "translator_commit": "710a16ee75731d6a12ea48aa9688e1c3223a9691",
                    "status": "blocked-by-branch-insensitive-private-pointer-analysis",
                    "tracked_by": branch_analysis_issue,
                    "target_runs": {
                        "directx": {
                            "elapsed_seconds": 418.54,
                            "diagnostic_code": (
                                "project.translate."
                                "directx-private-pointer-unsupported"
                            ),
                            "function": "qouter_float_2_8_4",
                            "parameter": "w",
                            "reason": "view-out-of-bounds",
                        },
                        "opengl": {
                            "elapsed_seconds": 364.117,
                            "diagnostic_code": (
                                "project.translate."
                                "opengl-private-pointer-unsupported"
                            ),
                            "function": "qouter_float_2_8_4",
                            "parameter": "w",
                            "reason": "view-out-of-bounds",
                        },
                    },
                },
            ],
            "exact_run": {
                "translator_commit": "7b1779a22c49cf445384f02d50c2116d426f43fc",
                "status": "completed",
                "target_runs": {
                    "directx": {"elapsed_seconds": 429.507},
                    "opengl": {"elapsed_seconds": 417.522},
                },
            },
        },
        "resolved_contracts": [
            "helper-array-decay-template-deduction",
            "specialized-struct-constexpr-assertion-evaluation",
            "lexical-receiver-alias-resolution",
            "statement-bounded-member-template-parsing",
            "concrete-constructor-preservation",
            "line-wrapped-qualified-struct-receiver-materialization",
            "contextual-metal-method-receiver-resolution",
            "dependent-template-value-argument-resolution",
            "canonical-non-type-specialization-identity",
            "source-scoped-specialization-values",
            "equivalent-duplicate-static-owner-resolution",
            "constructor-argument-address-space-provenance",
            "equivalent-duplicate-struct-alias-component-type",
            "cv-qualified-constructor-member-initialization",
            "fixed-member-array-constructor-initialization",
            "generic-crossgl-pointer-cast-parsing",
            "whole-fragment-thread-elements-canonicalization",
            "cooperative-matrix-fragment-contract-provenance",
            "source-proven-cooperative-matrix-coordinate-mapping",
            "cooperative-matrix-expression-result-inference",
            "directx-lane-local-cooperative-matrix-software-lowering",
            "opengl-lane-local-cooperative-matrix-software-lowering",
            "opengl-private-pointer-unresolved-base-analysis",
            "transitive-local-constexpr-materialization",
            "directx-concrete-ternary-condition-selection",
            "opengl-compile-time-global-preparation-before-private-pointer-analysis",
            "function-local-struct-hoisting",
            "concrete-constexpr-local-extents",
            "defaulted-zero-argument-helper-materialization",
            "statically-unreachable-private-pointer-branch-pruning",
        ],
        "resolved_contract_evidence": {
            "function-local-struct-hoisting": {
                "translator_commit": "16d4df7b1bca4412350410a94fd4cc01f15d97a9"
            },
            "concrete-constexpr-local-extents": {
                "translator_commit": "0509feabb863b988649444cd5e559eeb45eb8b54"
            },
            "defaulted-zero-argument-helper-materialization": {
                "translator_commit": "a8410478d359a629dac83b94fb153acd6a7c0705"
            },
            "statically-unreachable-private-pointer-branch-pruning": {
                "issue": branch_analysis_issue,
                "status": "resolved",
                "directx_commit": "3386410857e6ed42e4e0e89488b1011858cb38a4",
                "opengl_commit": "88241864b7f79d00c53f979ad5faf60e1c7fa114",
                "project_regression_commit": "7b1779a22c49cf445384f02d50c2116d426f43fc",
            },
        },
        "contextual_receiver_resolved_by_commit": (
            "c7a3c61addf9ca523e09bc81252ab76340c3f82c"
        ),
        "dependent_values_resolved_in": "https://github.com/CrossGL/crosstl/pull/1808",
        "advanced_past": {
            "member_call": "epilogue_op.apply",
            "receiver_declaration": (
                "thread const TransformNone_float_float& epilogue_op"
            ),
            "dependent_expressions": ["BK_padded", "BN_padded", "SIMD_SIZE"],
            "source_specialization": {
                "pattern": "mlx/backend/metal/kernels/fp_quantized.metal",
                "provenance_kind": "project-source-pattern",
                "values": [
                    {
                        "name": "align_M",
                        "id": 200,
                        "source_type": "bool",
                        "value": True,
                    },
                    {
                        "name": "align_N",
                        "id": 201,
                        "source_type": "bool",
                        "value": True,
                    },
                    {
                        "name": "align_K",
                        "id": 202,
                        "source_type": "bool",
                        "value": True,
                    },
                ],
            },
            "static_constant": "BaseMMAFrag_float_8_8::kFragRows",
            "constructor": "QuantizedBlockLoader_float_32_32_36_1_64_16_4",
            "struct_alias_component": {
                "alias": "BaseMMAFrag_float_8_8::frag_type",
                "index_expression": "k",
                "component_type": "float",
            },
            "cv_qualified_constructor_member": (
                "BlockLoader_float_16_32_36_1_64::src_ld"
            ),
            "fixed_member_array_constructor": (
                "MMATile_float_2_1_BaseMMAFrag_float_8_8::val_frags"
            ),
            "generic_pointer_cast": "(vec<bfloat16_t, 4>*)(xv[v] + k0)",
            "cooperative_matrix_fragment_contract": {
                "layout": "metal_thread_elements",
                "subgroup_size": 32,
                "elements_per_lane": 2,
                "provenance": "metal_thread_elements_reference_view",
                "mapping": "tile_4x4_row_pair",
                "mapping_provenance": "mlx_steel_BaseMMAFrag_get_coord",
                "source_contract": "contracts/cooperative-matrix-fragment-mapping.json",
            },
            "transitive_local_constexpr": {
                "symbol": "values_per_thread",
                "fixed_array_extents_materialized": True,
                "resolved_by": source_materialization_issue,
                "resolved_boundaries": [
                    "symbolic-values_per_thread",
                    "missing-fixed-array-extent",
                    "missing-view-backing",
                ],
            },
            "private_pointer_partition_view": {
                "function": "qdot_float_16_4",
                "parameter": "x_thread",
                "reason": "unprovable-view-offset",
                "resolved_by": partition_view_issue,
                "target_contracts": {
                    "directx": "concrete-ternary-condition-selection",
                    "opengl": (
                        "compile-time-global-preparation-before-"
                        "private-pointer-analysis"
                    ),
                },
            },
            "branch_insensitive_private_pointer_analysis": {
                "function": "qouter_float_2_8_4",
                "parameter": "w",
                "previous_reason": "view-out-of-bounds",
                "resolved_by": branch_analysis_issue,
                "project_regression_commit": "7b1779a22c49cf445384f02d50c2116d426f43fc",
            },
        },
        "coordinate_mapping_status": {
            "source_contract": "contracts/cooperative-matrix-fragment-mapping.json",
            "source_coordinates_verified": True,
            "lane_local_software_lowering_foundation_implemented": True,
            "target_policy_implemented": False,
            "full_software_fallback_implemented": False,
            "default_target_behavior": "fail-closed",
            "project_configuration_wired": False,
            "numerical_parity_verified": False,
            "tracked_by": coordinate_mapping_issue,
        },
        "cooperative_matrix_contract_flow": {
            "status": "verified-contract-propagation",
            "ast_node_type": "CooperativeMatrixType",
            "ast_node_count": 16,
            "contract_field_count": 12,
            "complete_contract_node_count_before": 2,
            "complete_contract_node_count_after": 16,
            "fragment_contract": {
                "layout": "metal_thread_elements",
                "subgroup_size": 32,
                "elements_per_lane": 2,
                "provenance": "metal_thread_elements_reference_view",
                "mapping": "tile_4x4_row_pair",
                "mapping_provenance": "mlx_steel_BaseMMAFrag_get_coord",
            },
        },
        "cooperative_matrix_operation_result_types": {
            "status": "resolved-expression-contracts",
            "ast_node_type": "CooperativeMatrixOpNode",
            "ast_node_count": 8,
            "operation_counts": {"element": 7, "multiply_accumulate": 1},
            "expression_type_set_count": 8,
            "element": {
                "count": 7,
                "expression_type": "float",
                "matrix_result_type_set_count": 0,
                "matrix_result_type_intentionally_absent_count": 7,
            },
            "multiply_accumulate": {
                "count": 1,
                "expression_type": (
                    "CooperativeMatrix<float, 8, 8, subgroup, unspecified, "
                    "unspecified, metal_thread_elements, 32, 2, "
                    "metal_thread_elements_reference_view, "
                    "tile_4x4_row_pair, mlx_steel_BaseMMAFrag_get_coord>"
                ),
                "result_type": (
                    "CooperativeMatrix<float, 8, 8, subgroup, unspecified, "
                    "unspecified, metal_thread_elements, 32, 2, "
                    "metal_thread_elements_reference_view, "
                    "tile_4x4_row_pair, mlx_steel_BaseMMAFrag_get_coord>"
                ),
                "complete_contract_field_count": 12,
                "preserves_accumulator_representation": True,
            },
            "resolved_by": operation_result_type_issue,
        },
        "cooperative_matrix_software_lowering": {
            "status": "lane-local-foundation",
            "mode": "explicit-opt-in",
            "registered_mapping": {
                "matrix_shape": [8, 8],
                "subgroup_size": 32,
                "elements_per_lane": 2,
                "mapping": "tile_4x4_row_pair",
            },
            "targets": ["directx", "opengl"],
            "supported_operations": [
                "type-representation",
                "element",
                "negate",
                "elementwise-add",
                "elementwise-subtract",
                "elementwise-multiply",
            ],
            "fail_closed_operations": [
                "load",
                "store",
                "multiply",
                "multiply-accumulate",
            ],
            "default_enabled": False,
            "project_configuration_wired": False,
            "full_fallback_implemented": False,
            "target_policy_implemented": False,
            "runtime_execution_verified": False,
            "numerical_parity_verified": False,
            "tracked_by": [cooperative_matrix_issue, coordinate_mapping_issue],
        },
        "current_boundaries": {
            "directx": {
                "diagnostic_code": (
                    "project.translate.directx-workgroup-pointer-unsupported"
                ),
                "missing_capability": "directx.workgroup-pointer-lowering",
                "workgroup_pointer": {
                    "function": (
                        "BlockMMA_float_float_16_32_32_1_2_" "false_true_36_36__mma"
                    ),
                    "parameter": "As",
                    "reason": "dynamic-control-flow-reassignment",
                },
                "scope_classification": {
                    "groupshared_alias_lowering": directx_workgroup_issue,
                    "alias_reassignment": directx_workgroup_issue,
                    "structured_rejection": directx_workgroup_issue,
                },
                "blocked_by": [directx_workgroup_issue],
            },
            "opengl": {
                "diagnostic_code": (
                    "project.translate.opengl-workgroup-pointer-unsupported"
                ),
                "missing_capability": "opengl.workgroup-pointer-lowering",
                "workgroup_pointer": {
                    "parameter": "dst_",
                    "reason": "bare-pointer-expression",
                },
                "scope_classification": {
                    "pointer_bearing_aggregate_provenance": opengl_aggregate_issue,
                    "helper_backing_provenance": opengl_workgroup_issue,
                },
                "blocked_by": [opengl_aggregate_issue, opengl_workgroup_issue],
            },
        },
        "reduced_runtime_proofs": {
            "partition_view": {
                "status": "native-execution-observed-in-ci",
                "source": (
                    "tests/fixtures/runtime_verification/"
                    "private_pointer_partition/private_pointer_partition.cgl"
                ),
                "expected_readback": [100, 101, 102, 103, 200, 201, 202, 203],
                "ci_run_url": (
                    "https://github.com/CrossGL/crosstl/actions/runs/29641172600"
                ),
                "targets": {
                    "directx": {
                        "ci_platform": "windows-latest",
                        "runtime": "direct3d",
                        "workflow_step": (
                            "Prove Direct3D private-pointer partition writeback"
                        ),
                        "native_execution_wired": True,
                        "native_execution_observed": True,
                    },
                    "opengl": {
                        "ci_platform": "ubuntu-latest",
                        "runtime": "opengl",
                        "workflow_step": (
                            "Prove OpenGL private-pointer partition writeback"
                        ),
                        "native_execution_wired": True,
                        "native_execution_observed": True,
                    },
                },
                "local_execution_attempted": False,
                "local_execution_verified": False,
                "mlx_runtime_included": False,
                "numerical_parity_claimed": False,
            },
            "local_struct_byte_view": {
                "status": "native-execution-observed-in-ci",
                "source": (
                    "tests/fixtures/runtime_verification/"
                    "private_pointer_partition/private_pointer_word_view.metal"
                ),
                "expected_readback": [204],
                "readback_contract": (
                    "sum(byte[index] * (index + 1)) for index 0 through 7"
                ),
                "order_sensitive": True,
                "ci_run_url": (
                    "https://github.com/CrossGL/crosstl/actions/runs/29649620337"
                ),
                "source_validation": {
                    "platform": "macos",
                    "compiler": "xcrun metal",
                    "compiler_version": "32023.918",
                    "language_standard": "metal3.2",
                    "status": "compiled",
                },
                "targets": {
                    "directx": {
                        "ci_platform": "windows-latest",
                        "runtime": "direct3d",
                        "workflow_step": (
                            "Prove Direct3D local-struct byte-view native readback"
                        ),
                        "native_execution_wired": True,
                        "native_execution_observed": True,
                    },
                    "opengl": {
                        "ci_platform": "ubuntu-latest",
                        "runtime": "opengl",
                        "workflow_step": (
                            "Prove OpenGL local-struct byte-view native readback"
                        ),
                        "native_execution_wired": True,
                        "native_execution_observed": True,
                    },
                },
                "local_execution_attempted": False,
                "local_execution_verified": False,
                "mlx_runtime_included": False,
                "full_mlx_test_suite_included": False,
                "numerical_parity_claimed": False,
            },
        },
        "target_results": {
            "directx": {
                "translator_commit": "7b1779a22c49cf445384f02d50c2116d426f43fc",
                "elapsed_seconds": 429.507,
                "template_materialization": {
                    "status": "materialized",
                    "specialization_count": 606,
                    "unsupported_specialization_count": 0,
                },
                "project_translation": {
                    "artifact_record_count": 1,
                    "translated_count": 0,
                    "failed_count": 1,
                    "emitted_target_file_count": 0,
                    "project_diagnostic_count": 1,
                    "error_count": 1,
                },
                "diagnostic": {
                    "code": "project.translate.directx-workgroup-pointer-unsupported",
                    "missing_capability": "directx.workgroup-pointer-lowering",
                    "workgroup_pointer": {
                        "function": (
                            "BlockMMA_float_float_16_32_32_1_2_" "false_true_36_36__mma"
                        ),
                        "parameter": "As",
                        "reason": "dynamic-control-flow-reassignment",
                    },
                    "message": (
                        "DirectX cannot preserve workgroup pointer reassignment "
                        "for 'As' across nested control flow"
                    ),
                },
                "artifact_emitted": False,
                "native_validation_attempted": False,
                "native_validation_status": "not-run-no-artifact",
                "mlx_host_runtime_included": False,
                "runtime_execution_attempted": False,
                "numerical_parity_claimed": False,
            },
            "opengl": {
                "translator_commit": "7b1779a22c49cf445384f02d50c2116d426f43fc",
                "elapsed_seconds": 417.522,
                "template_materialization": {
                    "status": "materialized",
                    "specialization_count": 606,
                    "unsupported_specialization_count": 0,
                },
                "project_translation": {
                    "artifact_record_count": 1,
                    "translated_count": 0,
                    "failed_count": 1,
                    "emitted_target_file_count": 0,
                    "project_diagnostic_count": 1,
                    "error_count": 1,
                },
                "diagnostic": {
                    "code": "project.translate.opengl-workgroup-pointer-unsupported",
                    "missing_capability": "opengl.workgroup-pointer-lowering",
                    "workgroup_pointer": {
                        "parameter": "dst_",
                        "reason": "bare-pointer-expression",
                    },
                    "message": (
                        "OpenGL cannot emit a workgroup pointer as a first-class "
                        "value: dst_"
                    ),
                },
                "artifact_emitted": False,
                "native_validation_attempted": False,
                "native_validation_status": "not-run-no-artifact",
                "mlx_host_runtime_included": False,
                "runtime_execution_attempted": False,
                "numerical_parity_claimed": False,
            },
        },
        "target_artifact_count": 0,
        "resolved_issues": [
            resolved_issue,
            indexed_alias_issue,
            pointer_cast_issue,
            whole_fragment_issue,
            fragment_contract_issue,
            operation_result_type_issue,
            opengl_analyzer_issue,
            source_materialization_issue,
            partition_view_issue,
            branch_analysis_issue,
        ],
        "advanced_issue_contracts": [
            issue,
            dependent_value_issue,
            opengl_issue,
            directx_issue,
            local_typedef_issue,
            source_specialization_issue,
            constructor_issue,
            const_member_issue,
            member_array_issue,
            coordinate_mapping_issue,
        ],
        "remaining_blocked_by": [
            directx_workgroup_issue,
            opengl_aggregate_issue,
            opengl_workgroup_issue,
        ],
        "source_translation_claimed": False,
        "native_validation_attempted": False,
        "mlx_host_runtime_included": False,
        "runtime_integration_included": False,
        "runtime_execution_verified": False,
        "numerical_parity_claimed": False,
        "runtime_parity_claimed": False,
    }
    assert resolved_issue in expected_gaps["resolved_issues"]
    assert resolved_issue not in expected_gaps["tracked_issues"]
    assert indexed_alias_issue in expected_gaps["resolved_issues"]
    assert indexed_alias_issue not in expected_gaps["tracked_issues"]
    assert whole_fragment_issue in expected_gaps["resolved_issues"]
    assert whole_fragment_issue not in expected_gaps["tracked_issues"]
    assert fragment_contract_issue in expected_gaps["resolved_issues"]
    assert fragment_contract_issue not in expected_gaps["tracked_issues"]
    assert issue in expected_gaps["tracked_issues"]
    assert dependent_value_issue in expected_gaps["tracked_issues"]
    assert directx_issue in expected_gaps["tracked_issues"]
    assert opengl_issue in expected_gaps["tracked_issues"]
    assert source_specialization_issue in expected_gaps["tracked_issues"]
    assert constructor_issue in expected_gaps["tracked_issues"]
    assert const_member_issue in expected_gaps["tracked_issues"]
    assert member_array_issue in expected_gaps["tracked_issues"]
    assert pointer_cast_issue in expected_gaps["resolved_issues"]
    assert pointer_cast_issue not in expected_gaps["tracked_issues"]
    assert cooperative_matrix_issue in expected_gaps["tracked_issues"]
    assert coordinate_mapping_issue in expected_gaps["tracked_issues"]
    assert partition_view_issue in expected_gaps["resolved_issues"]
    assert partition_view_issue not in expected_gaps["tracked_issues"]
    assert byte_reinterpretation_issue in expected_gaps["tracked_issues"]
    assert local_typedef_issue in expected_gaps["tracked_issues"]
    assert directx_workgroup_issue in expected_gaps["tracked_issues"]
    assert opengl_aggregate_issue in expected_gaps["tracked_issues"]
    assert opengl_workgroup_issue in expected_gaps["tracked_issues"]
    assert branch_analysis_issue in expected_gaps["resolved_issues"]
    assert branch_analysis_issue not in expected_gaps["tracked_issues"]
    assert source_materialization_issue not in expected_gaps["tracked_issues"]
    assert operation_result_type_issue in expected_gaps["resolved_issues"]
    assert operation_result_type_issue not in expected_gaps["tracked_issues"]
    assert opengl_analyzer_issue in expected_gaps["resolved_issues"]
    assert opengl_analyzer_issue not in expected_gaps["tracked_issues"]
    assert source_materialization_issue in expected_gaps["resolved_issues"]
    assert partition_view_issue in status["resolved_issues"]
    assert pointer_cast_issue in status["resolved_issues"]
    assert branch_analysis_issue in status["resolved_issues"]
    assert local_typedef_issue in status["advanced_issue_contracts"]
    assert local_typedef_issue not in status["resolved_issues"]
    assert status["remaining_blocked_by"] == [
        directx_workgroup_issue,
        opengl_aggregate_issue,
        opengl_workgroup_issue,
    ]
    assert byte_reinterpretation_issue not in status["remaining_blocked_by"]

    assert [
        run["translator_commit"] for run in status["replay"]["intermediate_runs"]
    ] == [
        "16d4df7b1bca4412350410a94fd4cc01f15d97a9",
        "0509feabb863b988649444cd5e559eeb45eb8b54",
        "710a16ee75731d6a12ea48aa9688e1c3223a9691",
    ]
    assert status["replay"]["exact_run"]["translator_commit"] == (
        "7b1779a22c49cf445384f02d50c2116d426f43fc"
    )
    resolved_boundaries = status["advanced_past"]["transitive_local_constexpr"][
        "resolved_boundaries"
    ]
    assert resolved_boundaries == [
        "symbolic-values_per_thread",
        "missing-fixed-array-extent",
        "missing-view-backing",
    ]
    assert status["current_boundaries"]["directx"]["blocked_by"] == [
        directx_workgroup_issue
    ]
    assert status["current_boundaries"]["opengl"]["blocked_by"] == [
        opengl_aggregate_issue,
        opengl_workgroup_issue,
    ]
    assert set(status["target_results"]) == {"directx", "opengl"}
    assert status["target_artifact_count"] == 0

    readme = MLX_README_PATH.read_text(encoding="utf-8")
    assert PINNED_MLX_COMMIT in readme
    assert "`epilogue_op.apply`" in readme
    assert "`thread const TransformNone_float_float& epilogue_op`" in readme
    assert "CrossTL commit\n`c7a3c61ad`" in readme
    assert "`BK_padded`" in readme
    assert "`BN_padded`" in readme
    assert "`SIMD_SIZE`" in readme
    assert "`project-source-pattern`" in readme
    assert "`align_M` (ID 200)" in readme
    assert "`align_N` (ID 201)" in readme
    assert "`align_K` (ID 202)" in readme
    assert "`project.translate.metal-local-type-unresolved`" in readme
    assert "`tn * bytes_per_pack`" in readme
    assert "`(2) * bytes_per_pack`" in readme
    assert "`BaseMMAFrag_float_8_8::kFragRows`" in readme
    assert "CrossGL/crosstl#1479" in readme
    assert "CrossGL/crosstl#1490" in readme
    assert "CrossGL/crosstl#1538" in readme
    assert "CrossGL/crosstl#1491" in readme
    assert "CrossGL/crosstl#1807" in readme
    assert "CrossGL/crosstl#1809" in readme
    assert "CrossGL/crosstl#1810" in readme
    assert "CrossGL/crosstl#1811" in readme
    assert "CrossGL/crosstl#1812" in readme
    assert "CrossGL/crosstl#1813" in readme
    assert "resolves CrossGL/crosstl#1814" in " ".join(readme.split())
    assert "CrossGL/crosstl#1815" in readme
    assert "CrossGL/crosstl#1816" in readme
    assert "CrossGL/crosstl#1602" in readme
    assert "CrossGL/crosstl#1610" in readme
    assert "CrossGL/crosstl#1820" in readme
    assert "CrossGL/crosstl#1824" in readme
    assert "CrossGL/crosstl#1826" in readme
    assert "CrossGL/crosstl#1546" in readme
    assert "CrossGL/crosstl#1518" in readme
    assert "CrossGL/crosstl#1544" in readme
    assert "CrossGL/crosstl#1671" in readme
    assert "CrossGL/crosstl#1829" in readme
    assert "ordered `cooperative_matrix_element` operations" in " ".join(readme.split())
    assert "`metal_thread_elements` layout" in " ".join(readme.split())
    assert "a 32-lane subgroup, two elements per lane" in " ".join(readme.split())
    assert "`metal_thread_elements_reference_view` provenance" in " ".join(
        readme.split()
    )
    assert "`tile_4x4_row_pair` mapping" in " ".join(readme.split())
    assert "`mlx_steel_BaseMMAFrag_get_coord` provenance" in " ".join(readme.split())
    assert "16 source `CooperativeMatrixType` contract nodes" in " ".join(
        readme.split()
    )
    assert "two nodes carried the complete 12-field contract" in " ".join(
        readme.split()
    )
    assert "all 16 carry the `metal_thread_elements` layout" in " ".join(readme.split())
    assert "eight `CooperativeMatrixOpNode` operations" in " ".join(readme.split())
    assert "scalar `expression_type` `float`" in " ".join(readme.split())
    assert "intentionally has no matrix `result_type`" in " ".join(readme.split())
    assert (
        "complete cooperative-matrix `result_type` and `expression_type`"
        in " ".join(readme.split())
    )
    assert "explicit opt-in lane-local cooperative-matrix" in " ".join(readme.split())
    assert "Reduced target tests compile and validate type representation" in " ".join(
        readme.split()
    )
    assert (
        "load, store, multiply, and multiply-accumulate operations remain fail closed"
        in " ".join(readme.split())
    )
    assert "not wired through project profiles or configuration" in " ".join(
        readme.split()
    )
    assert "transitive local `constexpr` materialization contract" in " ".join(
        readme.split()
    )
    assert "proving that `tn` had resolved to `2`" in " ".join(readme.split())
    assert "292.953 seconds" in " ".join(readme.split())
    assert "286.175 seconds" in " ".join(readme.split())
    assert "286.540 seconds" in " ".join(readme.split())
    assert "283.718 seconds" in " ".join(readme.split())
    assert "In the first replay" in " ".join(readme.split())
    assert "In the second replay" in " ".join(readme.split())
    assert "In the third replay" in " ".join(readme.split())
    assert "both code-generation factory paths" in " ".join(readme.split())
    assert "`qdot_float_16_4.x_thread`" in readme
    assert "`unprovable-view-offset`" in readme
    assert "function-local struct hoisting" in " ".join(readme.split())
    assert "concrete `constexpr` local extents" in " ".join(readme.split())
    assert "defaulted zero-argument helper materialization" in " ".join(readme.split())
    assert "are not current-boundary claims" in " ".join(readme.split())
    assert "418.540 seconds" in " ".join(readme.split())
    assert "364.117 seconds" in " ".join(readme.split())
    assert "`view-out-of-bounds`" in readme
    assert "branch-insensitive private-pointer analysis" in " ".join(readme.split())
    assert "`qouter_float_2_8_4.w`" in readme
    assert "DirectX and OpenGL branch pruning" in " ".join(readme.split())
    assert "resolve CrossGL/crosstl#1829" in " ".join(readme.split())
    assert "The completed exact replay" in " ".join(readme.split())
    assert "issue remains open until these changes merge" not in " ".join(
        readme.lower().split()
    )
    assert "429.507 seconds" in " ".join(readme.split())
    assert "417.522 seconds" in " ".join(readme.split())
    assert "no unsupported specializations" in " ".join(readme.split())
    assert "`project.translate.directx-workgroup-pointer-unsupported`" in readme
    assert "`directx.workgroup-pointer-lowering`" in readme
    assert "`BlockMMA_float_float_16_32_32_1_2_false_true_36_36__mma`" in readme
    assert "`dynamic-control-flow-reassignment`" in readme
    assert (
        "`DirectX cannot preserve workgroup pointer reassignment for 'As' across "
        "nested control flow`" in " ".join(readme.split())
    )
    assert "`project.translate.opengl-workgroup-pointer-unsupported`" in readme
    assert "`opengl.workgroup-pointer-lowering`" in readme
    assert "`bare-pointer-expression`" in readme
    assert (
        "`OpenGL cannot emit a workgroup pointer as a first-class value: dst_`"
        in " ".join(readme.split())
    )
    assert (
        "one failed artifact/provenance record, zero translated artifacts, and "
        "one error" in " ".join(readme.split())
    )
    assert "no target artifact was emitted" in " ".join(readme.lower().split())
    assert "Native validation was not attempted" in " ".join(readme.split())
    assert "MLX host runtime integration and execution were not attempted" in " ".join(
        readme.split()
    )
    assert "numerical parity was not evaluated" in " ".join(readme.split())
    assert "it is not the current exact boundary" in " ".join(readme.split())
    assert "CrossGL/crosstl#1567 remains open" in " ".join(readme.split())
    assert "[100, 101, 102, 103, 200, 201, 202, 203]" in readme
    assert "GitHub Actions run 29641172600" in " ".join(readme.split())
    assert "`Prove Direct3D private-pointer partition writeback`" in readme
    assert "`Prove OpenGL private-pointer partition writeback`" in readme
    assert "`private_pointer_word_view.metal`" in readme
    assert "`sum(byte[index] * (index + 1))`" in readme
    assert "required readback is `[204]`" in " ".join(readme.split())
    assert "compiles locally as Metal 3.2 with Apple metal `32023.918`" in " ".join(
        readme.split()
    )
    assert "GitHub Actions run 29649620337" in " ".join(readme.split())
    assert "produced the exact `[204]` readback" in " ".join(readme.split())
    assert "`windows-latest` through Direct3D" in " ".join(readme.split())
    assert "`ubuntu-latest` through OpenGL" in " ".join(readme.split())
    assert "`Prove Direct3D local-struct byte-view native readback`" in readme
    assert "`Prove OpenGL local-struct byte-view native readback`" in readme
    assert "reported zero mismatches with zero absolute and relative tolerance" in (
        " ".join(readme.split())
    )
    assert "do not establish complete MLX artifact translation" in " ".join(
        readme.split()
    )
    assert "full MLX host runtime integration" in " ".join(readme.split())
    assert "full MLX test-suite execution" in " ".join(readme.split())
    assert "numerical parity for MLX workloads" in " ".join(readme.split())


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


def test_scaled_dot_product_attention_aggregate_stays_fail_closed_with_bounded_proof():
    module = _load_harness()
    source = module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    resolved_issue = "https://github.com/CrossGL/crosstl/issues/1535"
    static_constant_issue = "https://github.com/CrossGL/crosstl/issues/1491"
    function_constant_issue = "https://github.com/CrossGL/crosstl/issues/1538"
    expected_gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

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

    evidence = expected_gaps["scaled_dot_product_attention_native_runtime_status"]
    assert evidence == module.MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_RUNTIME_EVIDENCE
    assert evidence["selected_entry_point"] == "sdpa_vector_float_64_64"
    assert evidence["dispatch_contract"]["host_selection"] == "one-pass-vector"
    assert (
        evidence["remaining_scope"]["aggregate_directx_opengl_translation_unblocked"]
        is False
    )
    assert (
        "separate bounded dispatch manifest"
        in expected_gaps["directx_toolchain_status"]["directx_toolchain_gaps"][source]
    )


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
        if target == "directx":
            bfloat16_evidence = module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE[source]
            artifact["bfloat16Lowering"] = dict(bfloat16_evidence["bfloat16Lowering"])
            artifact["requiredCapabilities"] = list(
                bfloat16_evidence["requiredCapabilities"]
            )
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


def _write_layer_norm_dispatch_report(
    module,
    mlx_root,
    output_dir,
    report_path,
    *,
    contract,
    toolchain_runs=(),
):
    unit = {
        "id": module.MLX_LAYER_NORM_SOURCE,
        "path": module.MLX_LAYER_NORM_SOURCE,
        "sourceBackend": "metal",
        "sourceHash": {
            "algorithm": "sha256",
            "value": module.MLX_LAYER_NORM_SHA256,
        },
        "sourceSizeBytes": 4096,
    }
    artifacts = []
    planned_artifacts = []
    for entry_point, expected in module.MLX_LAYER_NORM_DISPATCH_VARIANTS.items():
        variant = "dispatch-" + expected["artifactId"].removeprefix("sha256:")
        generated_path = (
            output_dir
            / "directx"
            / variant
            / "mlx/backend/metal/kernels/layer_norm"
            / f"{entry_point}.hlsl"
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_lines = [
            f"[numthreads({expected['workgroupSize'][0]}, 1, 1)]",
            "[WaveSize(32)]",
        ]
        if expected["specializationConstants"]:
            generated_lines.insert(0, "static const bool has_w = true;")
        generated_lines.extend(("void CSMain() {", "}", ""))
        generated_path.write_text("\n".join(generated_lines), encoding="utf-8")
        generated_hash = hashlib.sha256(generated_path.read_bytes()).hexdigest()
        plan = {
            "artifactId": expected["artifactId"],
            "dispatchVariantIds": [expected["dispatchVariantId"]],
            "entryPoint": entry_point,
            "manifestContentIdentities": [
                module.MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY
            ],
            "source": module.MLX_LAYER_NORM_SOURCE,
            "specializationConstants": dict(expected["specializationConstants"]),
            "subgroupWidth": expected["subgroupWidth"],
            "variant": variant,
            "workgroupSize": list(expected["workgroupSize"]),
        }
        planned_artifacts.append(plan)
        constants = [
            {
                "id": int(constant_id),
                "concreteValue": value,
                "deferred": False,
                "valueProvenance": {
                    "kind": "host-dispatch-contract",
                    "artifactId": expected["artifactId"],
                },
            }
            for constant_id, value in expected["specializationConstants"].items()
        ]
        specializations = [{"hostName": entry_point}]
        specializations.extend(
            {"materializedName": f"helper_{index}"}
            for index in range(expected["specializationCount"] - 1)
        )
        artifacts.append(
            {
                "source": module.MLX_LAYER_NORM_SOURCE,
                "sourceBackend": "metal",
                "sourceHash": unit["sourceHash"],
                "sourceSizeBytes": unit["sourceSizeBytes"],
                "target": "directx",
                "status": "translated",
                "variant": variant,
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": generated_hash,
                },
                "generatedSizeBytes": generated_path.stat().st_size,
                "entryPoint": {
                    "source": entry_point,
                    "target": "CSMain",
                    "stage": "compute",
                },
                "dispatchArtifact": plan,
                "execution": {
                    "sourceEntryPoints": [entry_point],
                    "provenance": {
                        "kind": "host-dispatch-contract",
                        "artifactId": expected["artifactId"],
                    },
                    "subgroupWidthProvenance": {
                        "kind": "host-dispatch-contract",
                    },
                    "subgroupWidthEnforcement": {
                        "mechanism": "hlsl-wave-size-attribute",
                        "minimumShaderModel": "6.6",
                        "entryProfiles": [
                            {"entryPoint": "CSMain", "profile": "cs_6_6"}
                        ],
                    },
                    "entryPoints": [
                        {
                            "sourceEntryPoint": entry_point,
                            "materializedEntryPoint": entry_point,
                            "targetEntryPoint": "CSMain",
                            "workgroupSize": list(expected["workgroupSize"]),
                            "subgroupWidth": expected["subgroupWidth"],
                        }
                    ],
                },
                "specializationConstants": constants,
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": expected["specializationCount"],
                    "specializations": specializations,
                    "unsupported": [],
                },
            }
        )

    artifact_count = len(artifacts)
    identity = contract["contentIdentity"]
    report = {
        "project": {
            "includePatterns": [module.MLX_LAYER_NORM_SOURCE],
            "targets": ["directx"],
            "dispatchContractFiles": [contract["path"]],
            "dispatchContractCount": 1,
            "dispatchVariantCount": artifact_count,
            "dispatchContracts": [
                {
                    "path": contract["path"],
                    "schemaVersion": 1,
                    "contentIdentity": identity,
                    "manifest": {"provenance": {"commit": module.MLX_COMMIT}},
                    "evaluation": {
                        "manifestSource": str((mlx_root / contract["path"]).resolve()),
                        "variantCount": artifact_count,
                    },
                }
            ],
            "dispatchArtifactPlan": {
                "kind": "crosstl-dispatch-artifact-plan",
                "schemaVersion": 1,
                "sourceUnitCount": 1,
                "artifactCount": artifact_count,
                "dispatchVariantCount": artifact_count,
                "artifacts": planned_artifacts,
            },
        },
        "summary": {
            "unitCount": 1,
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "artifactsByTarget": {
                "directx": {
                    "artifactCount": artifact_count,
                    "translatedCount": artifact_count,
                    "failedCount": 0,
                }
            },
        },
        "units": [unit],
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


def _write_logsumexp_dispatch_report(
    module,
    mlx_root,
    output_dir,
    report_path,
    *,
    contract,
    toolchain_runs=(),
):
    unit = {
        "id": module.MLX_LOGSUMEXP_SOURCE,
        "path": module.MLX_LOGSUMEXP_SOURCE,
        "sourceBackend": "metal",
        "sourceHash": {
            "algorithm": "sha256",
            "value": module.MLX_LOGSUMEXP_SHA256,
        },
        "sourceSizeBytes": 593,
    }
    artifacts = []
    planned_artifacts = []
    evaluated_variants = []
    for workload_id, expected in module.MLX_LOGSUMEXP_DISPATCH_VARIANTS.items():
        entry_point = expected["entryPoint"]
        variant = "dispatch-" + expected["artifactId"].removeprefix("sha256:")
        generated_path = (
            output_dir
            / "directx"
            / variant
            / "mlx/backend/metal/kernels/logsumexp"
            / f"{entry_point}.hlsl"
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(
            "\n".join(
                (
                    f"[numthreads({expected['workgroupSize'][0]}, 1, 1)]",
                    "[WaveSize(32)]",
                    "void CSMain() {",
                    "  float maxval = WaveActiveMax(0.0);",
                    "  GroupMemoryBarrierWithGroupSync();",
                    "  float normalizer = WaveActiveSum(exp(maxval));",
                    "  out_[gid] = log(normalizer);",
                    "}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        generated_hash = hashlib.sha256(generated_path.read_bytes()).hexdigest()
        plan = {
            "artifactId": expected["artifactId"],
            "dispatchVariantIds": [expected["dispatchVariantId"]],
            "entryPoint": entry_point,
            "manifestContentIdentities": [
                module.MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY
            ],
            "source": module.MLX_LOGSUMEXP_SOURCE,
            "specializationConstants": {},
            "subgroupWidth": 32,
            "variant": variant,
            "workgroupSize": list(expected["workgroupSize"]),
        }
        planned_artifacts.append(plan)
        evaluated_variants.append(
            {
                "artifactId": expected["artifactId"],
                "variantId": expected["dispatchVariantId"],
                "source": module.MLX_LOGSUMEXP_SOURCE,
                "entryPoint": entry_point,
                "workload": {
                    "id": workload_id,
                    "inputs": dict(expected["inputs"]),
                },
                "workgroupSize": list(expected["workgroupSize"]),
                "subgroupWidth": 32,
                "specializationConstants": {},
                "dispatch": {
                    "workgroupCount": list(expected["dispatchWorkgroupCount"])
                },
            }
        )
        artifacts.append(
            {
                "source": module.MLX_LOGSUMEXP_SOURCE,
                "sourceBackend": "metal",
                "sourceHash": unit["sourceHash"],
                "sourceSizeBytes": unit["sourceSizeBytes"],
                "target": "directx",
                "status": "translated",
                "variant": variant,
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": generated_hash,
                },
                "generatedSizeBytes": generated_path.stat().st_size,
                "entryPoint": {
                    "source": entry_point,
                    "target": "CSMain",
                    "stage": "compute",
                },
                "dispatchArtifact": plan,
                "execution": {
                    "sourceEntryPoints": [entry_point],
                    "provenance": {
                        "kind": "host-dispatch-contract",
                        "artifactId": expected["artifactId"],
                    },
                    "subgroupWidthProvenance": {
                        "kind": "host-dispatch-contract",
                    },
                    "subgroupWidthEnforcement": {
                        "mechanism": "hlsl-wave-size-attribute",
                        "minimumShaderModel": "6.6",
                        "entryProfiles": [
                            {"entryPoint": "CSMain", "profile": "cs_6_6"}
                        ],
                    },
                    "entryPoints": [
                        {
                            "sourceEntryPoint": entry_point,
                            "materializedEntryPoint": entry_point,
                            "targetEntryPoint": "CSMain",
                            "workgroupSize": list(expected["workgroupSize"]),
                            "subgroupWidth": 32,
                        }
                    ],
                },
                "specializationConstants": None,
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": 1,
                    "specializations": [
                        {
                            "hostName": entry_point,
                            "parameters": {
                                "AccT": "float",
                                "N_READS": "4",
                                "T": "float",
                            },
                        }
                    ],
                    "unsupported": [],
                },
            }
        )

    artifact_count = len(artifacts)
    report = {
        "project": {
            "includePatterns": [module.MLX_LOGSUMEXP_SOURCE],
            "targets": ["directx"],
            "dispatchContractFiles": [contract["path"]],
            "dispatchContractCount": 1,
            "dispatchVariantCount": artifact_count,
            "dispatchContracts": [
                {
                    "path": contract["path"],
                    "schemaVersion": 1,
                    "contentIdentity": contract["contentIdentity"],
                    "manifest": {"provenance": {"commit": module.MLX_COMMIT}},
                    "evaluation": {
                        "manifestSource": str((mlx_root / contract["path"]).resolve()),
                        "variantCount": artifact_count,
                        "variants": evaluated_variants,
                    },
                }
            ],
            "dispatchArtifactPlan": {
                "kind": "crosstl-dispatch-artifact-plan",
                "schemaVersion": 1,
                "sourceUnitCount": 1,
                "artifactCount": artifact_count,
                "dispatchVariantCount": artifact_count,
                "artifacts": planned_artifacts,
            },
        },
        "summary": {
            "unitCount": 1,
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "artifactsByTarget": {
                "directx": {
                    "artifactCount": artifact_count,
                    "translatedCount": artifact_count,
                    "failedCount": 0,
                }
            },
        },
        "units": [unit],
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


def _write_rms_norm_dispatch_report(
    module,
    mlx_root,
    output_dir,
    report_path,
    *,
    contract,
    toolchain_runs=(),
):
    unit = {
        "id": module.MLX_RMS_NORM_SOURCE,
        "path": module.MLX_RMS_NORM_SOURCE,
        "sourceBackend": "metal",
        "sourceHash": {
            "algorithm": "sha256",
            "value": module.MLX_RMS_NORM_SHA256,
        },
        "sourceSizeBytes": 12069,
    }
    artifacts = []
    planned_artifacts = []
    evaluated_variants = []
    for workload_id, expected in module.MLX_RMS_NORM_DISPATCH_VARIANTS.items():
        entry_point = expected["entryPoint"]
        variant = "dispatch-" + expected["artifactId"].removeprefix("sha256:")
        generated_path = (
            output_dir
            / "directx"
            / variant
            / "mlx/backend/metal/kernels/rms_norm"
            / f"{entry_point}.hlsl"
        )
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_lines = [
            f"[numthreads({expected['workgroupSize'][0]}, 1, 1)]",
            "[WaveSize(32)]",
        ]
        if expected["specializationConstants"]:
            has_w = str(expected["specializationConstants"]["20"]).lower()
            generated_lines.insert(0, f"static const bool has_w = {has_w};")
        generated_lines.extend(("void CSMain() {", "}", ""))
        generated_path.write_text("\n".join(generated_lines), encoding="utf-8")
        generated_hash = hashlib.sha256(generated_path.read_bytes()).hexdigest()
        plan = {
            "artifactId": expected["artifactId"],
            "dispatchVariantIds": [expected["dispatchVariantId"]],
            "entryPoint": entry_point,
            "manifestContentIdentities": [
                module.MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY
            ],
            "source": module.MLX_RMS_NORM_SOURCE,
            "specializationConstants": dict(expected["specializationConstants"]),
            "subgroupWidth": 32,
            "variant": variant,
            "workgroupSize": list(expected["workgroupSize"]),
        }
        planned_artifacts.append(plan)
        evaluated_variants.append(
            {
                "artifactId": expected["artifactId"],
                "variantId": expected["dispatchVariantId"],
                "source": module.MLX_RMS_NORM_SOURCE,
                "entryPoint": entry_point,
                "workload": {
                    "id": workload_id,
                    "inputs": dict(expected["inputs"]),
                },
                "workgroupSize": list(expected["workgroupSize"]),
                "subgroupWidth": 32,
                "specializationConstants": dict(expected["specializationConstants"]),
                "dispatch": {
                    "workgroupCount": list(expected["dispatchWorkgroupCount"])
                },
            }
        )
        constants = [
            {
                "id": int(constant_id),
                "concreteValue": value,
                "deferred": False,
                "valueProvenance": {
                    "kind": "host-dispatch-contract",
                    "artifactId": expected["artifactId"],
                },
            }
            for constant_id, value in expected["specializationConstants"].items()
        ]
        artifacts.append(
            {
                "source": module.MLX_RMS_NORM_SOURCE,
                "sourceBackend": "metal",
                "sourceHash": unit["sourceHash"],
                "sourceSizeBytes": unit["sourceSizeBytes"],
                "target": "directx",
                "status": "translated",
                "variant": variant,
                "path": generated_path.relative_to(mlx_root).as_posix(),
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": generated_hash,
                },
                "generatedSizeBytes": generated_path.stat().st_size,
                "entryPoint": {
                    "source": entry_point,
                    "target": "CSMain",
                    "stage": "compute",
                },
                "dispatchArtifact": plan,
                "execution": {
                    "sourceEntryPoints": [entry_point],
                    "provenance": {
                        "kind": "host-dispatch-contract",
                        "artifactId": expected["artifactId"],
                    },
                    "subgroupWidthProvenance": {
                        "kind": "host-dispatch-contract",
                    },
                    "subgroupWidthEnforcement": {
                        "mechanism": "hlsl-wave-size-attribute",
                        "minimumShaderModel": "6.6",
                        "entryProfiles": [
                            {"entryPoint": "CSMain", "profile": "cs_6_6"}
                        ],
                    },
                    "entryPoints": [
                        {
                            "sourceEntryPoint": entry_point,
                            "materializedEntryPoint": entry_point,
                            "targetEntryPoint": "CSMain",
                            "workgroupSize": list(expected["workgroupSize"]),
                            "subgroupWidth": 32,
                        }
                    ],
                },
                "specializationConstants": constants,
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": 1,
                    "specializations": [{"hostName": entry_point}],
                    "unsupported": [],
                },
            }
        )

    artifact_count = len(artifacts)
    report = {
        "project": {
            "includePatterns": [module.MLX_RMS_NORM_SOURCE],
            "targets": ["directx"],
            "dispatchContractFiles": [contract["path"]],
            "dispatchContractCount": 1,
            "dispatchVariantCount": artifact_count,
            "dispatchContracts": [
                {
                    "path": contract["path"],
                    "schemaVersion": 1,
                    "contentIdentity": contract["contentIdentity"],
                    "manifest": {"provenance": {"commit": module.MLX_COMMIT}},
                    "evaluation": {
                        "manifestSource": str((mlx_root / contract["path"]).resolve()),
                        "variantCount": artifact_count,
                        "variants": evaluated_variants,
                    },
                }
            ],
            "dispatchArtifactPlan": {
                "kind": "crosstl-dispatch-artifact-plan",
                "schemaVersion": 1,
                "sourceUnitCount": 1,
                "artifactCount": artifact_count,
                "dispatchVariantCount": artifact_count,
                "artifacts": planned_artifacts,
            },
        },
        "summary": {
            "unitCount": 1,
            "artifactCount": artifact_count,
            "translatedCount": artifact_count,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "artifactsByTarget": {
                "directx": {
                    "artifactCount": artifact_count,
                    "translatedCount": artifact_count,
                    "failedCount": 0,
                }
            },
        },
        "units": [unit],
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


def _prepare_fft_directx_toolchain_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    artifact_path = (
        work_dir
        / "out-fft-directx"
        / "directx"
        / Path(module.MLX_FFT_SOURCE).with_suffix("")
        / f"{module.FFT_DIRECTX_ENTRY_POINT}.hlsl"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        """
static const bool inv_ = false;
static const bool is_power_of_2_ = true;
static const int elems_per_thread_ = 4;
static const int radix_4_steps_ = 4;
groupshared float2 fft_mem_256_float2_float2_shared_in[256];
uint3 crossglNumWorkGroups;
int consume_native_short(int16_t value) {
    return int(value);
}
[numthreads(1, 1, 64)]
void CSMain() {
    uint3 grid = (crossglNumWorkGroups * uint3(1, 1, 64));
}
""".lstrip(),
        encoding="utf-8",
    )
    relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
    expected_workgroup_assertions = [
        {
            "source": assertion["source"],
            "entryPoint": assertion["entry_point"],
            "function": assertion["function"],
            "parameter": assertion["parameter"],
            "minimum": assertion["minimum"],
            "maximum": assertion["maximum"],
        }
        for assertion in module.FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS
    ]
    expected_rule_path = (
        f'project.entry_workgroup_size_rules["{module.MLX_FFT_SOURCE}"].'
        f"{module.FFT_DIRECTX_ENTRY_POINT}"
    )
    report = {
        "kind": "crosstl-project-portability-report",
        "summary": {
            "unitCount": 1,
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "diagnosticsByCode": {},
            "missingCapabilityCounts": {},
        },
        "project": {
            "indexRangeAssertionCount": len(module.FFT_INDEX_RANGE_ASSERTIONS),
            "indexRangeAssertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
            "workgroupAccessAssertionCount": len(
                module.FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS
            ),
            "workgroupAccessAssertions": expected_workgroup_assertions,
        },
        "diagnostics": [],
        "units": [
            {
                "id": module.MLX_FFT_SOURCE,
                "path": module.MLX_FFT_SOURCE,
                "sourceBackend": "metal",
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.MLX_FFT_SHA256,
                },
                "sourceSizeBytes": module.MLX_FFT_SOURCE_SIZE_BYTES,
            }
        ],
        "artifacts": [
            {
                "source": module.MLX_FFT_SOURCE,
                "sourceBackend": "metal",
                "target": "directx",
                "path": relative_artifact_path,
                "status": "translated",
                "provenance": {
                    "intermediate": "crossgl",
                    "pipeline": "entry-scoped-translate",
                },
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.MLX_FFT_SHA256,
                },
                "sourceSizeBytes": module.MLX_FFT_SOURCE_SIZE_BYTES,
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": module._sha256(artifact_path),
                },
                "generatedSizeBytes": artifact_path.stat().st_size,
                "entryPoint": {
                    "source": module.FFT_DIRECTX_ENTRY_POINT,
                    "stage": "compute",
                    "target": "CSMain",
                },
                "execution": {
                    "sourceEntryPoints": [module.FFT_DIRECTX_ENTRY_POINT],
                    "provenance": {
                        "kind": "materialized-template-entry-rules",
                        "path": (
                            f'project.entry_workgroup_size_rules["{module.MLX_FFT_SOURCE}"]'
                        ),
                    },
                    "entryPoints": [
                        {
                            "sourceEntryPoint": module.FFT_DIRECTX_ENTRY_POINT,
                            "materializedEntryPoint": module.FFT_DIRECTX_ENTRY_POINT,
                            "targetEntryPoint": "CSMain",
                            "workgroupSize": list(module.FFT_DIRECTX_WORKGROUP_SIZE),
                            "rule": {
                                "components": [
                                    str(value)
                                    for value in module.FFT_DIRECTX_WORKGROUP_SIZE
                                ],
                                "entryPattern": module.FFT_DIRECTX_ENTRY_POINT,
                                "path": expected_rule_path,
                                "sourcePattern": module.MLX_FFT_SOURCE,
                            },
                        }
                    ],
                },
                "specializationConstants": [
                    {
                        "id": constant_id,
                        "concreteValue": value,
                        "deferred": False,
                    }
                    for constant_id, value in (
                        module.FFT_DIRECTX_REACHABLE_SPECIALIZATION_CONSTANTS.items()
                    )
                ],
                "templateMaterialization": {
                    "status": "materialized",
                    "specializations": [
                        {"materializedName": f"specialization_{index}"}
                        for index in range(
                            module.FFT_DIRECTX_EXPECTED_SPECIALIZATION_COUNT
                        )
                    ],
                    "unsupported": [],
                },
            }
        ],
    }
    (report_dir / "fft-directx-toolchain.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return (
        (mlx_root, work_dir, config_dir, report_dir, log_dir),
        artifact_path,
        expected_workgroup_assertions,
    )


def test_fft_directx_toolchain_records_host_plan_and_native_validation(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, artifact_path, expected_workgroup_assertions = (
        _prepare_fft_directx_toolchain_check(module, tmp_path)
    )
    generated = artifact_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        module,
        "FFT_DIRECTX_GENERATED_SHA256",
        module._sha256(artifact_path),
    )
    monkeypatch.setattr(
        module,
        "FFT_DIRECTX_GENERATED_SIZE_BYTES",
        artifact_path.stat().st_size,
    )
    commands = []

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "translate-fft-directx":
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(generated, encoding="utf-8")
        if name == "validate-fft-directx":
            output_path = Path(command[command.index("-Fo") + 1])
            output_path.write_bytes(b"DXIL")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: f"C:/tools/{name}.exe" if name == "dxc" else None,
    )

    result = module._check_fft_directx_toolchain(
        *paths,
        "python",
        require_toolchain=True,
    )

    assert result == {
        "name": "fft-directx-toolchain",
        "status": "passed",
        "report": ".crosstl-mlx-porting/reports/fft-directx-toolchain.json",
        "source": module.MLX_FFT_SOURCE,
        "sourceHash": module.MLX_FFT_SHA256,
        "target": "directx",
        "selectedEntryPoint": module.FFT_DIRECTX_ENTRY_POINT,
        "targetEntryPoint": "CSMain",
        "artifactStatus": "translated",
        "artifactEmitted": True,
        "generatedHash": module._sha256(artifact_path),
        "generatedSizeBytes": artifact_path.stat().st_size,
        "nativeValidationAttempted": True,
        "nativeValidationStatus": "validated",
        "nativeValidationOutput": (
            ".crosstl-mlx-porting/validation/fft-directx-256.dxil"
        ),
        "nativeCompiler": "dxc",
        "entryProfile": "cs_6_2",
        "compilerArguments": ["-enable-16bit-types"],
        "warningsAsErrors": True,
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": 24,
        "configuredFunctionConstantCount": 22,
        "reachableFunctionConstantCount": 21,
        "specializationConstants": dict(module.FFT_DIRECTX_SPECIALIZATION_CONSTANTS),
        "workgroupSize": [1, 1, 64],
        "indexRangeAssertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
        "workgroupAccessAssertions": expected_workgroup_assertions,
        "toolchainRequired": True,
        "trackedIssues": [],
        "maxTemplateSpecializations": 4096,
        "maxTemplateMaterializationWork": 2097152,
        "runtimeIntegrationIncluded": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }
    assert [name for name, _command, _kwargs in commands] == [
        "translate-fft-directx",
        "validate-fft-directx",
    ]
    assert commands[0][2] == {
        "check": False,
        "timeout_seconds": module.FFT_DIRECTX_TRANSLATION_TIMEOUT_SECONDS,
    }
    assert commands[1][2] == {"check": False}
    dxc_command = commands[1][1]
    assert dxc_command[:7] == [
        "C:/tools/dxc.exe",
        "-WX",
        "-T",
        "cs_6_2",
        "-enable-16bit-types",
        "-E",
        "CSMain",
    ]
    config = (paths[2] / "fft-directx-toolchain.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_FFT_SOURCE}"]' in config
    assert 'targets = ["directx"]' in config
    assert f'"{module.MLX_FFT_SOURCE}" = "{module.FFT_DIRECTX_ENTRY_POINT}"' in config
    assert f'[project.entry_workgroup_size_rules."{module.MLX_FFT_SOURCE}"]' in config
    assert f'"{module.FFT_DIRECTX_ENTRY_POINT}" = [1, 1, 64]' in config
    assert config.count("[[project.index_range_assertions]]") == len(
        module.FFT_INDEX_RANGE_ASSERTIONS
    )
    assert config.count("[[project.workgroup_access_assertions]]") == 1
    assert config.count("[[project.specialization_constants]]") == 0
    assert "[project.specialization_constants]" in config
    assert config.count(" = ") >= len(module.FFT_DIRECTX_SPECIALIZATION_CONSTANTS)


def test_fft_directx_toolchain_rejects_changed_workgroup_axis(tmp_path, monkeypatch):
    module = _load_harness()
    paths, artifact_path, _expected_workgroup_assertions = (
        _prepare_fft_directx_toolchain_check(module, tmp_path)
    )
    report_path = paths[3] / "fft-directx-toolchain.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0]["execution"]["entryPoints"][0]["workgroupSize"] = [
        64,
        1,
        1,
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    generated = artifact_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        module,
        "FFT_DIRECTX_GENERATED_SHA256",
        module._sha256(artifact_path),
    )
    monkeypatch.setattr(
        module,
        "FFT_DIRECTX_GENERATED_SIZE_BYTES",
        artifact_path.stat().st_size,
    )

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "translate-fft-directx":
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(generated, encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(module.PortingCheckError, match="execution contract changed"):
        module._check_fft_directx_toolchain(
            *paths,
            "python",
            require_toolchain=False,
        )


def _prepare_fft_opengl_toolchain_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    artifact_path = (
        work_dir / "out-fft-opengl" / "opengl" / module.MLX_FFT_SOURCE
    ).with_suffix(".glsl")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        """#version 450 core
void radix_butterfly_2_radix2(vec2 value) {
}
void main() {
}
""",
        encoding="utf-8",
    )
    relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
    expected_workgroup_assertions = [
        {
            "source": assertion["source"],
            "entryPoint": assertion["entry_point"],
            "function": assertion["function"],
            "parameter": assertion["parameter"],
            "minimum": assertion["minimum"],
            "maximum": assertion["maximum"],
        }
        for assertion in module.FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS
    ]
    report = {
        "kind": "crosstl-project-portability-report",
        "summary": {
            "unitCount": 1,
            "artifactCount": 1,
            "translatedCount": 1,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "diagnosticsByCode": {},
            "missingCapabilityCounts": {},
        },
        "project": {
            "indexRangeAssertionCount": len(module.FFT_INDEX_RANGE_ASSERTIONS),
            "indexRangeAssertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
            "workgroupAccessAssertionCount": len(
                module.FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS
            ),
            "workgroupAccessAssertions": expected_workgroup_assertions,
        },
        "diagnostics": [],
        "units": [
            {
                "id": module.MLX_FFT_SOURCE,
                "path": module.MLX_FFT_SOURCE,
                "sourceBackend": "metal",
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.MLX_FFT_SHA256,
                },
                "sourceSizeBytes": module.MLX_FFT_SOURCE_SIZE_BYTES,
            }
        ],
        "artifacts": [
            {
                "source": module.MLX_FFT_SOURCE,
                "sourceBackend": "metal",
                "target": "opengl",
                "path": relative_artifact_path,
                "status": "translated",
                "provenance": {
                    "intermediate": "crossgl",
                    "pipeline": "single-file-translate",
                },
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": module.MLX_FFT_SHA256,
                },
                "sourceSizeBytes": module.MLX_FFT_SOURCE_SIZE_BYTES,
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": module._sha256(artifact_path),
                },
                "generatedSizeBytes": artifact_path.stat().st_size,
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
    (report_dir / "fft-opengl-toolchain.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return (
        (mlx_root, work_dir, config_dir, report_dir, log_dir),
        artifact_path,
        expected_workgroup_assertions,
    )


def test_fft_opengl_toolchain_records_translation_and_native_validation(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, artifact_path, expected_workgroup_assertions = (
        _prepare_fft_opengl_toolchain_check(module, tmp_path)
    )
    generated = artifact_path.read_text(encoding="utf-8")
    commands = []

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "translate-fft-opengl":
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(generated, encoding="utf-8")
        if name == "validate-fft-opengl":
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"SPIR-V")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)
    monkeypatch.setattr(
        module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )

    result = module._check_fft_opengl_toolchain(
        *paths,
        "python",
        require_toolchain=True,
    )

    assert result == {
        "name": "fft-opengl-toolchain",
        "status": "passed",
        "report": ".crosstl-mlx-porting/reports/fft-opengl-toolchain.json",
        "source": module.MLX_FFT_SOURCE,
        "sourceHash": module.MLX_FFT_SHA256,
        "target": "opengl",
        "artifactStatus": "translated",
        "artifactEmitted": True,
        "generatedHash": module._sha256(artifact_path),
        "generatedSizeBytes": artifact_path.stat().st_size,
        "nativeValidationAttempted": True,
        "nativeValidationStatus": "validated",
        "nativeValidationOutput": ".crosstl-mlx-porting/validation/fft-opengl.spv",
        "nativeCompiler": "glslangValidator",
        "spirvValidator": "spirv-val",
        "templateMaterializationStatus": "materialized",
        "templateSpecializationCount": 99,
        "functionConstantCount": 22,
        "indexRangeAssertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
        "workgroupAccessAssertions": expected_workgroup_assertions,
        "toolchainRequired": True,
        "trackedIssues": [],
        "maxTemplateSpecializations": 4096,
        "maxTemplateMaterializationWork": 2097152,
        "runtimeIntegrationIncluded": False,
        "numericalParityClaimed": False,
        "runtimeParityClaimed": False,
    }
    assert [name for name, _command, _kwargs in commands] == [
        "translate-fft-opengl",
        "validate-fft-opengl",
        "validate-fft-opengl-spirv",
    ]
    assert commands[0][2]["check"] is False
    assert (
        commands[0][2]["timeout_seconds"]
        == module.FFT_OPENGL_TRANSLATION_TIMEOUT_SECONDS
        == 900
    )
    assert commands[0][1][-1] == "--no-format"
    assert commands[1][2] == {"check": False}
    assert commands[2][2] == {"check": False}
    config = (paths[2] / "fft-opengl-toolchain.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_FFT_SOURCE}"]' in config
    assert 'targets = ["opengl"]' in config
    assert config.count("[[project.index_range_assertions]]") == len(
        module.FFT_INDEX_RANGE_ASSERTIONS
    )
    assert config.count("[[project.workgroup_access_assertions]]") == len(
        module.FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS
    )
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_fft_opengl_toolchain_rejects_changed_portability_precondition(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    paths, _artifact_path, _expected_workgroup_assertions = (
        _prepare_fft_opengl_toolchain_check(module, tmp_path)
    )
    report_path = paths[3] / "fft-opengl-toolchain.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["project"]["workgroupAccessAssertions"][0]["maximum"] -= 1
    report_path.write_text(json.dumps(report), encoding="utf-8")

    def fake_run_command(name, command, *, log_dir, **_kwargs):
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return module.CommandResult(name, list(command), 0, stdout_path, stderr_path)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    with pytest.raises(
        module.PortingCheckError,
        match="portability preconditions changed",
    ):
        module._check_fft_opengl_toolchain(
            *paths,
            "python",
            require_toolchain=False,
        )


def _configure_small_gemv_opengl_toolchain(module, monkeypatch):
    monkeypatch.setattr(module, "GEMV_EXPECTED_ENTRY_POINT_COUNT", 2)
    monkeypatch.setattr(module, "GEMV_EXPECTED_SPECIALIZATION_COUNT", 3)
    monkeypatch.setattr(
        module,
        "GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES",
        ((32, 1, 1), (32, 8, 1)),
    )


def _prepare_gemv_opengl_toolchain_check(module, tmp_path):
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for path in (config_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_hash = {"algorithm": "sha256", "value": module.MLX_GEMV_SHA256}
    workgroup_rule_path = f'project.workgroup_size_rules["{module.MLX_GEMV_SOURCE}"]'
    subgroup_rule_path = f'project.subgroup_width_rules["{module.MLX_GEMV_SOURCE}"]'
    generated_files = {}
    artifacts = []
    for index, workgroup_size in enumerate(
        module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES,
        start=1,
    ):
        source_entry = f"gemv_fixture_{index}"
        artifact_path = (
            work_dir
            / "out-gemv-opengl"
            / "opengl"
            / "mlx/backend/metal/kernels/gemv"
            / f"{source_entry}.glsl"
        )
        relative_artifact_path = artifact_path.relative_to(mlx_root).as_posix()
        generated = "\n".join(
            (
                "#version 450 core",
                "#extension GL_KHR_shader_subgroup_basic : require",
                f"#define CROSSTL_REQUIRED_SUBGROUP_WIDTH {module.GEMV_SUBGROUP_WIDTH}u",
                (
                    "layout(local_size_x = "
                    f"{workgroup_size[0]}, local_size_y = {workgroup_size[1]}, "
                    f"local_size_z = {workgroup_size[2]}) in;"
                ),
                "void main() {",
                "    if (gl_SubgroupSize != CROSSTL_REQUIRED_SUBGROUP_WIDTH) {",
                "        return;",
                "    }",
                "}",
                "",
            )
        )
        generated_files[artifact_path] = generated
        artifacts.append(
            {
                "source": module.MLX_GEMV_SOURCE,
                "sourceBackend": "metal",
                "target": "opengl",
                "path": relative_artifact_path,
                "status": "translated",
                "sourceHash": source_hash,
                "sourceSizeBytes": module.MLX_GEMV_SOURCE_SIZE_BYTES,
                "generatedHash": {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
                },
                "generatedSizeBytes": len(generated.encode("utf-8")),
                "provenance": {
                    "intermediate": "crossgl",
                    "pipeline": "entry-scoped-translate",
                },
                "entryPoint": {
                    "source": source_entry,
                    "stage": "compute",
                    "target": "main",
                },
                "execution": {
                    "entryPoints": [
                        {
                            "identity": _test_contract_identity(
                                f"entry-{source_entry}"
                            ),
                            "materialization": {
                                "hostName": source_entry,
                                "materializedName": source_entry,
                                "name": "gemv",
                            },
                            "materializedEntryPoint": source_entry,
                            "sourceEntryPoint": source_entry,
                            "targetEntryPoint": "main",
                            "parameters": {},
                            "parameterSources": {},
                            "rule": {
                                "components": list(
                                    module.GEMV_REPORT_WORKGROUP_SIZE_RULE
                                ),
                                "path": workgroup_rule_path,
                                "sourcePattern": module.MLX_GEMV_SOURCE,
                            },
                            "subgroupWidth": module.GEMV_SUBGROUP_WIDTH,
                            "subgroupWidthRule": {
                                "expression": str(module.GEMV_SUBGROUP_WIDTH),
                                "path": subgroup_rule_path,
                                "sourcePattern": module.MLX_GEMV_SOURCE,
                            },
                            "workgroupSize": list(workgroup_size),
                        }
                    ],
                    "identity": _test_contract_identity(f"artifact-{source_entry}"),
                    "provenance": {
                        "kind": "materialized-template-rule",
                        "path": workgroup_rule_path,
                    },
                    "sourceEntryPoints": [source_entry],
                    "subgroupWidthEnforcement": dict(
                        module.GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT
                    ),
                    "subgroupWidthProvenance": {
                        "kind": "materialized-template-rule",
                        "path": subgroup_rule_path,
                    },
                },
                "templateMaterialization": {
                    "status": "materialized",
                    "specializationCount": module.GEMV_EXPECTED_SPECIALIZATION_COUNT,
                    "specializations": [
                        {"materializedName": f"specialization_{item}"}
                        for item in range(module.GEMV_EXPECTED_SPECIALIZATION_COUNT)
                    ],
                    "unsupported": [],
                },
            }
        )

    report = {
        "kind": "crosstl-project-portability-report",
        "project": {
            "workgroupSizeRules": {
                module.MLX_GEMV_SOURCE: list(module.GEMV_REPORT_WORKGROUP_SIZE_RULE),
            },
            "workgroupSizeRuleCount": 1,
            "subgroupWidthRules": {
                module.MLX_GEMV_SOURCE: str(module.GEMV_SUBGROUP_WIDTH),
            },
            "subgroupWidthRuleCount": 1,
            "indexRangeAssertions": [
                {"source": module.MLX_GEMV_SOURCE, **assertion}
                for assertion in module.GEMV_OPENGL_INDEX_RANGE_ASSERTIONS
            ],
            "indexRangeAssertionCount": len(module.GEMV_OPENGL_INDEX_RANGE_ASSERTIONS),
        },
        "summary": {
            "unitCount": 1,
            "skippedCount": 0,
            "targetCount": 1,
            "artifactCount": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
            "translatedCount": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
            "failedCount": 0,
            "diagnosticCounts": {"error": 0, "note": 0, "warning": 0},
            "diagnosticsByCode": {},
            "missingCapabilityCounts": {},
            "artifactProvenanceByPipeline": {
                "entry-scoped-translate": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
            },
            "artifactProvenanceByIntermediate": {
                "crossgl": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
            },
            "sourceMapCount": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
            "sourceRemapCount": module.GEMV_EXPECTED_ENTRY_POINT_COUNT,
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
        "artifacts": artifacts,
    }
    report_path = report_dir / "gemv-opengl.json"
    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    return (
        (mlx_root, work_dir, config_dir, report_dir, log_dir),
        report_path,
        generated_files,
    )


def _stub_gemv_opengl_toolchain(
    module,
    monkeypatch,
    commands,
    report_path,
    generated_files,
    *,
    returncodes=None,
):
    report_text = report_path.read_text(encoding="utf-8")
    returncodes = returncodes or {}

    monkeypatch.setattr(module.shutil, "which", lambda name: f"/tools/{name}")

    def fake_run_command(name, command, *, log_dir, **kwargs):
        commands.append((name, list(command), kwargs))
        stdout_path = log_dir / f"{name}.stdout"
        stderr_path = log_dir / f"{name}.stderr"
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        if name == "translate-gemv-opengl":
            report_path.write_text(report_text, encoding="utf-8")
            for path, generated in generated_files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(generated, encoding="utf-8")
        elif name.startswith("compile-gemv-opengl-"):
            output_path = Path(command[command.index("-o") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"SPIR-V fixture")
        return module.CommandResult(
            name,
            list(command),
            returncodes.get(name, 0),
            stdout_path,
            stderr_path,
        )

    monkeypatch.setattr(module, "_run_command", fake_run_command)


def test_gemv_opengl_toolchain_accepts_complete_compiler_validation(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    _configure_small_gemv_opengl_toolchain(module, monkeypatch)
    paths, report_path, generated_files = _prepare_gemv_opengl_toolchain_check(
        module, tmp_path
    )
    commands = []
    _stub_gemv_opengl_toolchain(
        module,
        monkeypatch,
        commands,
        report_path,
        generated_files,
    )

    result = module._check_gemv_opengl_toolchain(*paths, "python")

    assert result["name"] == "gemv-opengl-toolchain"
    assert result["status"] == "passed"
    assert result["artifactStatus"] == "translated"
    assert result["artifactCount"] == 2
    assert result["emittedTargetFileCount"] == 2
    assert result["reportExecutionEntryCount"] == 2
    assert result["nativeValidationStatus"] == "validated"
    assert result["toolchainValidatedArtifactCount"] == 2
    assert len(result["toolchainValidationRuns"]) == 2
    assert result["resolvedWorkgroupSizes"] == [[32, 1, 1], [32, 8, 1]]
    assert result["subgroupWidth"] == 32
    assert result["subgroupWidthRuleConfigured"] is True
    assert result["translationBlockedBy"] == []
    assert result["executionContractBlockedBy"] == []
    assert result["runtimeExecutionAttempted"] is False
    assert result["runtimeIntegrationIncluded"] is False
    assert result["numericalParityClaimed"] is False
    assert result["runtimeParityClaimed"] is False
    assert [name for name, _command, _kwargs in commands] == [
        "translate-gemv-opengl",
        "compile-gemv-opengl-001",
        "validate-gemv-opengl-001",
        "compile-gemv-opengl-002",
        "validate-gemv-opengl-002",
    ]
    assert commands[0][2]["check"] is False
    assert (
        commands[0][2]["timeout_seconds"]
        == module.GEMV_OPENGL_TRANSLATION_TIMEOUT_SECONDS
        == 1500
    )
    assert commands[0][1][-1] == "--no-format"
    assert all(path.is_file() for path in generated_files)
    config = (paths[2] / "gemv-opengl.toml").read_text(encoding="utf-8")
    assert f'include = ["{module.MLX_GEMV_SOURCE}"]' in config
    assert 'targets = ["opengl"]' in config
    assert "[project.workgroup_size_rules]" in config
    assert f'"{module.MLX_GEMV_SOURCE}" = [32, "BN", "BM"]' in config
    assert "[project.subgroup_width_rules]" in config
    assert f'"{module.MLX_GEMV_SOURCE}" = "32"' in config
    assert config.count("[[project.index_range_assertions]]") == len(
        module.GEMV_OPENGL_INDEX_RANGE_ASSERTIONS
    )
    assert "max_template_specializations = 4096" in config
    assert "max_template_materialization_work = 2097152" in config


def test_gemv_opengl_toolchain_rejects_incomplete_artifact_accounting(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    _configure_small_gemv_opengl_toolchain(module, monkeypatch)
    paths, report_path, generated_files = _prepare_gemv_opengl_toolchain_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["translatedCount"] = 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_toolchain(
        module, monkeypatch, commands, report_path, generated_files
    )

    with pytest.raises(module.PortingCheckError, match="every entry-scoped artifact"):
        module._check_gemv_opengl_toolchain(*paths, "python")

    assert [name for name, _command, _kwargs in commands] == ["translate-gemv-opengl"]


def test_gemv_opengl_toolchain_rejects_missing_subgroup_rule(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    _configure_small_gemv_opengl_toolchain(module, monkeypatch)
    paths, report_path, generated_files = _prepare_gemv_opengl_toolchain_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["project"]["subgroupWidthRules"] = {}
    report["project"]["subgroupWidthRuleCount"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_toolchain(
        module, monkeypatch, commands, report_path, generated_files
    )

    with pytest.raises(module.PortingCheckError, match="exact subgroup-width rule"):
        module._check_gemv_opengl_toolchain(*paths, "python")


def test_gemv_opengl_toolchain_rejects_generated_hash_mismatch(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    _configure_small_gemv_opengl_toolchain(module, monkeypatch)
    paths, report_path, generated_files = _prepare_gemv_opengl_toolchain_check(
        module, tmp_path
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"][0]["generatedHash"]["value"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    commands = []
    _stub_gemv_opengl_toolchain(
        module, monkeypatch, commands, report_path, generated_files
    )

    with pytest.raises(module.PortingCheckError, match="hash or size"):
        module._check_gemv_opengl_toolchain(*paths, "python")


def test_gemv_opengl_toolchain_rejects_native_compile_failure(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    _configure_small_gemv_opengl_toolchain(module, monkeypatch)
    paths, report_path, generated_files = _prepare_gemv_opengl_toolchain_check(
        module, tmp_path
    )
    commands = []
    _stub_gemv_opengl_toolchain(
        module,
        monkeypatch,
        commands,
        report_path,
        generated_files,
        returncodes={"compile-gemv-opengl-002": 1},
    )

    with pytest.raises(module.PortingCheckError, match="failed to compile"):
        module._check_gemv_opengl_toolchain(*paths, "python")


def test_gemv_opengl_toolchain_flag_is_available(tmp_path, monkeypatch):
    module = _load_harness()

    args = module.parse_args(
        ["--mlx-root", "/tmp/mlx", "--require-opengl-gemv-toolchain"]
    )

    assert args.require_opengl_gemv_toolchain is True
    with pytest.raises(SystemExit):
        module.parse_args(["--mlx-root", "/tmp/mlx", "--require-opengl-gemv-frontier"])


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
    helper_records = [
        {
            "name": "elem_to_loc",
            "materializedName": "elem_to_loc_uint",
            "parameters": {"IdxT": "uint"},
            "parameterSources": {"IdxT": "call-site"},
        },
        {
            "name": "elem_to_loc",
            "materializedName": "elem_to_loc_uint32_t",
            "parameters": {"IdxT": "uint32_t"},
            "parameterSources": {"IdxT": "call-site"},
        },
    ]
    return [*helper_records, *host_records], execution


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
    assert result["templateSpecializationCount"] == 226
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
    assert result["specializationCount"] == 226
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
    assert 'targets = ["directx", "opengl"]' in config
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
            "--checkpoint",
            str(report_dir / "full-corpus.checkpoint.json"),
            "--job-timeout-seconds",
            "120",
            "--validate",
        ]
    ]
    assert "--run-toolchains" not in commands[0]
    assert result["unitCount"] == 42
    assert result["artifactCount"] == 84
    assert result["translatedCount"] == 82
    assert result["failedCount"] == 2
    assert result["status"] == "passed-with-expected-fence-blockers"
    assert result["jobTimeoutSeconds"] == 120
    assert result["targetCounts"] == {
        "directx": {"translatedCount": 41, "failedCount": 1},
        "opengl": {"translatedCount": 41, "failedCount": 1},
    }
    assert result["fenceContract"]["status"] == "blocked-as-expected"
    assert set(result["fenceContract"]["targetContracts"]) == {
        "directx",
        "opengl",
    }
    assert result["shaderArtifactsOnly"] is True
    assert result["runtimeIntegrationIncluded"] is False
    assert result["runtimeParityClaimed"] is False


def test_full_corpus_mode_resumes_existing_progress_checkpoint(
    tmp_path,
    monkeypatch,
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    checkpoint_path = report_dir / "full-corpus.checkpoint.json"
    _write_full_corpus_checkpoint(
        module,
        checkpoint_path,
        state="interrupted",
    )
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
        mlx_root,
        work_dir,
        config_dir,
        report_dir,
        log_dir,
        "python",
    )

    assert commands[0][-1] == "--resume"
    assert result["status"] == "passed-with-expected-fence-blockers"


def test_full_corpus_mode_rejects_invalid_progress_checkpoint(
    tmp_path,
):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    work_dir = mlx_root / ".crosstl-mlx-porting"
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True)
    (report_dir / "full-corpus.checkpoint.json").write_text(
        '{"state": "running"}\n',
        encoding="utf-8",
    )

    with pytest.raises(module.PortingCheckError, match="checkpoint is invalid"):
        module._translate_full_corpus(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            "python",
        )


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
    assert result["translatedCount"] == 81
    assert result["failedCount"] == 3
    assert result["expectedFenceFailureCount"] == 2
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
        _write_full_corpus_checkpoint(
            module,
            report_dir / "full-corpus.checkpoint.json",
        )
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
    assert result["jobTimeoutSeconds"] == 120
    assert result["checkpoint"]["produced"] is True
    assert result["checkpoint"]["state"] == "running"
    assert result["checkpoint"]["completedCount"] == 2
    assert result["checkpoint"]["activeCoordinate"] == {
        "source": module.MLX_ARG_REDUCE_SOURCE,
        "target": "directx",
        "path": (
            ".crosstl-mlx-porting/out-full-corpus/directx/"
            "mlx/backend/metal/kernels/arg_reduce.hlsl"
        ),
    }
    assert result["checkpoint"]["lastCompletedCoordinate"]["source"] == (
        module.MLX_ARANGE_SOURCE
    )
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

    directx_toolchain_count = module.MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT
    layer_norm_contract = {
        "path": ".crosstl-mlx-porting/contracts/layer_norm.dispatch.json",
        "contentIdentity": {
            "algorithm": "sha256",
            "value": module.MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix(
                "sha256:"
            ),
        },
        "variantCount": len(module.MLX_LAYER_NORM_DISPATCH_VARIANTS),
    }
    monkeypatch.setattr(
        module,
        "_prepare_layer_norm_dispatch_contract",
        lambda *_args: layer_norm_contract,
    )
    logsumexp_contract = {
        "path": ".crosstl-mlx-porting/contracts/logsumexp.dispatch.json",
        "contentIdentity": {
            "algorithm": "sha256",
            "value": module.MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY.removeprefix(
                "sha256:"
            ),
        },
        "variantCount": len(module.MLX_LOGSUMEXP_DISPATCH_VARIANTS),
    }
    monkeypatch.setattr(
        module,
        "_prepare_logsumexp_dispatch_contract",
        lambda *_args: logsumexp_contract,
    )
    rms_norm_contract = {
        "path": ".crosstl-mlx-porting/contracts/rms_norm.dispatch.json",
        "contentIdentity": {
            "algorithm": "sha256",
            "value": module.MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY.removeprefix(
                "sha256:"
            ),
        },
        "variantCount": len(module.MLX_RMS_NORM_DISPATCH_VARIANTS),
    }
    monkeypatch.setattr(
        module,
        "_prepare_rms_norm_dispatch_contract",
        lambda *_args: rms_norm_contract,
    )
    commands = []

    def warning_stderr(source, relative_path):
        lines = []
        warning_index = 1
        for contract in module.MLX_DIRECTX_TOOLCHAIN_WARNING_CONTRACTS:
            if contract["source"] != source:
                continue
            for source_line in contract["sourceLines"]:
                for _ in range(source_line["occurrencesPerRun"]):
                    lines.extend(
                        (
                            f"{relative_path}:{warning_index}:1: warning: "
                            f"{contract['message']}",
                            source_line["text"],
                            "^",
                        )
                    )
                    warning_index += 1
        return "\n".join(lines)

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
                sources=module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
                validated=True,
            )
            returncode = 1
        elif "layer-norm-dispatch" in name:
            is_toolchain = name.startswith("validate-")
            output_dir = work_dir / (
                "out-directx-layer-norm-dispatch-toolchain"
                if is_toolchain
                else "out-directx-layer-norm-dispatch-frontier"
            )
            report = _write_layer_norm_dispatch_report(
                module,
                mlx_root,
                output_dir,
                report_dir / f"{name}.json",
                contract=layer_norm_contract,
            )
            if is_toolchain:
                toolchain_runs = [
                    {
                        "source": module.MLX_LAYER_NORM_SOURCE,
                        "target": "directx",
                        "path": artifact["path"],
                        "command": [
                            "dxc",
                            "-T",
                            "cs_6_6",
                            "-E",
                            "CSMain",
                            artifact["path"],
                        ],
                        "status": "ok",
                        "stderr": "",
                    }
                    for artifact in report["artifacts"]
                ]
                _write_layer_norm_dispatch_report(
                    module,
                    mlx_root,
                    output_dir,
                    report_dir / f"{name}.json",
                    contract=layer_norm_contract,
                    toolchain_runs=toolchain_runs,
                )
        elif "logsumexp-dispatch" in name:
            is_toolchain = name.startswith("validate-")
            output_dir = work_dir / (
                "out-directx-logsumexp-dispatch-toolchain"
                if is_toolchain
                else "out-directx-logsumexp-dispatch-frontier"
            )
            report = _write_logsumexp_dispatch_report(
                module,
                mlx_root,
                output_dir,
                report_dir / f"{name}.json",
                contract=logsumexp_contract,
            )
            if is_toolchain:
                toolchain_runs = [
                    {
                        "source": module.MLX_LOGSUMEXP_SOURCE,
                        "target": "directx",
                        "path": artifact["path"],
                        "command": [
                            "dxc",
                            "-T",
                            "cs_6_6",
                            "-E",
                            "CSMain",
                            artifact["path"],
                        ],
                        "status": "ok",
                        "stderr": "",
                    }
                    for artifact in report["artifacts"]
                ]
                _write_logsumexp_dispatch_report(
                    module,
                    mlx_root,
                    output_dir,
                    report_dir / f"{name}.json",
                    contract=logsumexp_contract,
                    toolchain_runs=toolchain_runs,
                )
        elif "rms-norm-dispatch" in name:
            is_toolchain = name.startswith("validate-")
            output_dir = work_dir / (
                "out-directx-rms-norm-dispatch-toolchain"
                if is_toolchain
                else "out-directx-rms-norm-dispatch-frontier"
            )
            report = _write_rms_norm_dispatch_report(
                module,
                mlx_root,
                output_dir,
                report_dir / f"{name}.json",
                contract=rms_norm_contract,
            )
            if is_toolchain:
                toolchain_runs = [
                    {
                        "source": module.MLX_RMS_NORM_SOURCE,
                        "target": "directx",
                        "path": artifact["path"],
                        "command": [
                            "dxc",
                            "-T",
                            "cs_6_6",
                            "-E",
                            "CSMain",
                            artifact["path"],
                        ],
                        "status": "ok",
                        "stderr": "",
                    }
                    for artifact in report["artifacts"]
                ]
                _write_rms_norm_dispatch_report(
                    module,
                    mlx_root,
                    output_dir,
                    report_dir / f"{name}.json",
                    contract=rms_norm_contract,
                    toolchain_runs=toolchain_runs,
                )
        else:
            is_toolchain = name == "validate-directx-frontier-toolchain"
            output_dir = (
                work_dir / "out-directx-frontier-toolchain"
                if is_toolchain
                else work_dir / "out-directx-frontier"
            )
            toolchain_runs = []
            if is_toolchain:
                for source in module.MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES:
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
                                "stderr": warning_stderr(source, relative_path),
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
    assert result["status"] == "passed-with-bounded-dispatch-and-pending-contracts"
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
    assert result["directxToolchainWarningEvidence"] == (
        module.MLX_DIRECTX_TOOLCHAIN_WARNING_EVIDENCE
    )
    assert result["contextualNarrowingEvidence"] == (
        module.MLX_DIRECTX_CONTEXTUAL_NARROWING_EVIDENCE
    )
    assert result["native16BitArithmeticEvidence"] == (
        module.MLX_DIRECTX_NATIVE_16_BIT_ARITHMETIC_EVIDENCE
    )
    assert result["bfloat16LoweringEvidence"] == (
        module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE
    )
    assert result["layerNormDispatchEvidence"]["status"] == ("translated-dxc-validated")
    assert result["layerNormDispatchEvidence"]["artifactCount"] == 2
    assert result["layerNormDispatchEvidence"]["dxcValidatedArtifactCount"] == 2
    assert set(result["layerNormDispatchEvidence"]["variants"]) == set(
        module.MLX_LAYER_NORM_DISPATCH_VARIANTS
    )
    assert result["logsumexpDispatchEvidence"]["status"] == ("translated-dxc-validated")
    assert result["logsumexpDispatchEvidence"]["artifactCount"] == 2
    assert result["logsumexpDispatchEvidence"]["dxcValidatedArtifactCount"] == 2
    assert set(result["logsumexpDispatchEvidence"]["variants"]) == set(
        module.MLX_LOGSUMEXP_DISPATCH_VARIANTS
    )
    assert result["rmsNormDispatchEvidence"]["status"] == ("translated-dxc-validated")
    assert result["rmsNormDispatchEvidence"]["artifactCount"] == 12
    assert result["rmsNormDispatchEvidence"]["dxcValidatedArtifactCount"] == 12
    assert set(result["rmsNormDispatchEvidence"]["variants"]) == set(
        module.MLX_RMS_NORM_DISPATCH_VARIANTS
    )
    assert result["workgroupBlockedSources"] == list(
        module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert all(
        evidence["sourceEntryPointIdentityStatus"] == "matched-materialized-host-names"
        for evidence in result["dynamicWorkgroupDispatchEvidence"].values()
    )
    assert [name for name, _command in commands] == [
        "directx-frontier",
        "directx-layer-norm-dispatch-frontier",
        "directx-logsumexp-dispatch-frontier",
        "directx-rms-norm-dispatch-frontier",
        "directx-workgroup-frontier",
        "validate-directx-frontier-toolchain",
        "validate-directx-layer-norm-dispatch-toolchain",
        "validate-directx-logsumexp-dispatch-toolchain",
        "validate-directx-rms-norm-dispatch-toolchain",
    ]
    assert "--run-toolchains" in commands[5][1]
    assert "--run-toolchains" in commands[6][1]
    assert "--run-toolchains" in commands[7][1]
    assert "--run-toolchains" in commands[8][1]
    toolchain_config = (
        config_dir / "validate-directx-frontier-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert 'targets = ["directx"]' in toolchain_config
    assert "[project.specialization_constants]" in toolchain_config
    for selector, value in module.MLX_FRONTIER_SPECIALIZATION_CONSTANTS.items():
        assert f"{json.dumps(selector)} = {json.dumps(value)}" in toolchain_config
    for source in module.MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES:
        assert source in toolchain_config
    assert module.MLX_LAYER_NORM_SOURCE not in toolchain_config
    layer_norm_config = (
        config_dir / "validate-directx-layer-norm-dispatch-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert module.MLX_LAYER_NORM_SOURCE in layer_norm_config
    assert layer_norm_contract["path"] in layer_norm_config
    logsumexp_config = (
        config_dir / "validate-directx-logsumexp-dispatch-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert module.MLX_LOGSUMEXP_SOURCE in logsumexp_config
    assert logsumexp_contract["path"] in logsumexp_config
    rms_norm_config = (
        config_dir / "validate-directx-rms-norm-dispatch-toolchain.toml"
    ).read_text(encoding="utf-8")
    assert module.MLX_RMS_NORM_SOURCE in rms_norm_config
    assert rms_norm_contract["path"] in rms_norm_config


def test_directx_toolchain_warning_contract_rejects_new_warning():
    module = _load_harness()
    assert module.MLX_DIRECTX_TOOLCHAIN_WARNING_CONTRACTS == ()
    warnings = [("new target warning [-Wconversion]", "value = wider;")]
    stderr = "\n".join(
        line
        for index, (message, source_line) in enumerate(warnings, start=1)
        for line in (
            f"generated.hlsl:{index}:1: warning: {message}",
            source_line,
            "^",
        )
    )

    with pytest.raises(module.PortingCheckError, match="toolchain warnings changed"):
        module._directx_toolchain_warning_evidence(
            [{"source": module.MLX_ARANGE_SOURCE, "stderr": stderr}]
        )


def test_directx_bfloat16_lowering_evidence_matches_pinned_report():
    module = _load_harness()
    native_storage_sources = {
        module.MLX_ARANGE_SOURCE,
        module.MLX_BINARY_TWO_SOURCE,
        module.MLX_RANDOM_SOURCE,
        module.MLX_ROPE_SOURCE,
        module.MLX_TERNARY_SOURCE,
    }
    assert set(module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE) == set(
        module.MLX_DIRECTX_TRANSLATED_FRONTIER_SOURCES
    )
    for source, evidence in module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE.items():
        uses_native_storage = source in native_storage_sources
        assert evidence == {
            "bfloat16Lowering": {
                "status": "exact",
                "approximationUsed": False,
                "registerRepresentation": "uint-low-16-bits",
                "storageRepresentation": (
                    "native-uint16" if uses_native_storage else "not-required"
                ),
                "roundingMode": "round-to-nearest-ties-to-even",
            },
            "requiredCapabilities": (
                ["directx.native-16bit-types"] if uses_native_storage else []
            ),
        }


@pytest.mark.parametrize(
    "drift",
    (
        "missing-lowering",
        "status",
        "approximation",
        "register-representation",
        "storage-representation",
        "rounding-mode",
        "extra-lowering-field",
        "missing-required-capabilities",
        "required-capabilities",
    ),
)
def test_directx_bfloat16_lowering_contract_rejects_report_drift(drift):
    module = _load_harness()
    artifacts = {
        source: {
            "bfloat16Lowering": dict(evidence["bfloat16Lowering"]),
            "requiredCapabilities": list(evidence["requiredCapabilities"]),
        }
        for source, evidence in module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE.items()
    }
    artifact = artifacts[module.MLX_ARANGE_SOURCE]
    lowering = artifact["bfloat16Lowering"]
    if drift == "missing-lowering":
        artifact.pop("bfloat16Lowering")
    elif drift == "status":
        lowering["status"] = "approximate"
    elif drift == "approximation":
        lowering["approximationUsed"] = True
    elif drift == "register-representation":
        lowering["registerRepresentation"] = "float16"
    elif drift == "storage-representation":
        lowering["storageRepresentation"] = "not-required"
    elif drift == "rounding-mode":
        lowering["roundingMode"] = "round-toward-zero"
    elif drift == "extra-lowering-field":
        lowering["conversionRepresentation"] = "helper-functions"
    elif drift == "missing-required-capabilities":
        artifact.pop("requiredCapabilities")
    else:
        artifact["requiredCapabilities"] = []

    with pytest.raises(
        module.PortingCheckError,
        match="DirectX bfloat16 lowering contract changed",
    ):
        module._require_directx_bfloat16_lowering_evidence(artifacts)


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
        module.MLX_LAYER_NORM_SOURCE,
        module.MLX_LOGSUMEXP_SOURCE,
        module.MLX_RANDOM_SOURCE,
        module.MLX_RMS_NORM_SOURCE,
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
        module.MLX_LAYER_NORM_SOURCE: 2,
        module.MLX_LOGSUMEXP_SOURCE: 2,
        module.MLX_RANDOM_SOURCE: 2,
        module.MLX_RMS_NORM_SOURCE: 12,
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
    assert len(expected_sources) == 8
    assert module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT == sum(
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS.values()
    )
    assert module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT == 484
    assert module.MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT == 21
    assert sum(module.MLX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS.values()) == 106
    assert {
        source: evidence["specializationCount"]
        for source, evidence in module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE.items()
    } == {
        module.MLX_ARG_REDUCE_SOURCE: 39,
        module.MLX_LAYER_NORM_SOURCE: 16,
        module.MLX_LOGSUMEXP_SOURCE: 7,
        module.MLX_RMS_NORM_SOURCE: 12,
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: 42,
        module.MLX_SOFTMAX_SOURCE: 14,
    }
    directx_status = gaps["directx_toolchain_status"]
    assert directx_status["specialization_constants"] == (
        module.MLX_FRONTIER_SPECIALIZATION_CONSTANTS
    )
    assert directx_status["dxc_validated_sources"] == list(expected_sources)
    assert directx_status["bfloat16_lowering_evidence"] == (
        module.MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE
    )
    assert directx_status["expected_entry_point_counts"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
    )
    assert directx_status["workgroup_blocked_sources"] == list(
        module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert directx_status["workgroup_blocked_entry_point_counts"] == (
        module.MLX_DIRECTX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS
    )
    assert directx_status["workgroup_blocked_diagnostic"] == (
        module.MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
    )
    assert directx_status["host_dispatch_import_resolved_by"] == (
        module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    )
    assert directx_status["dispatch_evidence"] == (
        {
            source: module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source]
            for source in module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
        }
    )
    assert module.MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    ]["hostLines"].endswith("613-640,685-753")
    assert set(directx_status["directx_toolchain_gaps"]) == (
        set(module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES)
        | {module.MLX_FENCE_SOURCE}
    )
    assert {
        f"https://github.com/CrossGL/crosstl/issues/{issue}"
        for issue in (
            1474,
            1694,
            1695,
            1701,
            1726,
            1728,
            1750,
            1784,
            1787,
            1789,
            1790,
            1792,
        )
    } <= set(module.RESOLVED_FRONTIER_ISSUES)
    assert set(module.RESOLVED_FRONTIER_ISSUES) <= set(gaps["resolved_issues"])
    assert not set(module.RESOLVED_FRONTIER_ISSUES) & set(gaps["tracked_issues"])
    assert "https://github.com/CrossGL/crosstl/issues/1696" in gaps["tracked_issues"]


def test_directx_frontier_readme_records_compile_only_scope_and_current_gaps():
    readme = MLX_README_PATH.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "official DXC v1.9.2602.24 on Windows CI" in readme
    assert "21-artifact frontier representing eight pinned sources" in (
        normalized_readme
    )
    assert "11, 225, 2, 2, 2, 12, 18, and 212 entries respectively" in (
        normalized_readme
    )
    assert "484 generated compute entries" in normalized_readme
    assert "three pending aggregate sources cover 76 compute entries" in (
        normalized_readme
    )
    assert "no placeholder workgroup size is restored" in normalized_readme
    assert "exact per-source bfloat16 report evidence" in normalized_readme
    assert "`status=exact`, `approximationUsed=false`" in normalized_readme
    assert "`uint-low-16-bits` register representation" in normalized_readme
    assert "round-to-nearest, ties-to-even conversion" in normalized_readme
    assert "`directx.native-16bit-types` capability" in normalized_readme
    assert "All five aggregate sources require" in normalized_readme
    assert "fails closed if either field is missing or changes" in normalized_readme
    assert "storage, conversion, report, and compiler evidence only" in (
        normalized_readme
    )
    assert "beyond the current compile-time smoke mapping" not in normalized_readme
    assert "does not execute a bfloat16 workload" in normalized_readme
    assert "does not extend the bounded runtime proof to bfloat16" in (
        normalized_readme
    )
    for issue in (1694, 1695, 1696, 1701, 1728, 1793, 1542, 1537):
        assert f"https://github.com/CrossGL/crosstl/issues/{issue}" in readme
    assert "https://github.com/CrossGL/crosstl/issues/1750" not in readme
    assert "does not dispatch these kernels or establish numerical parity" in (
        normalized_readme
    )
    assert "captures 12 distinct dispatch artifacts exercised by the pinned" in (
        normalized_readme
    )
    assert "test_rms_norm` and `test_rms_norm_grad` workloads" in normalized_readme
    assert "the MLX runtime is not redirected to these artifacts" in (normalized_readme)
    assert "DirectX remains outside the DXC gate" not in readme


def test_selected_quantized_frontiers_record_current_target_boundaries():
    module = _load_harness()
    readme = MLX_README_PATH.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "Native 16-bit HLSL support" in normalized_readme
    assert "Concrete `static_assert` evaluation" in normalized_readme
    assert "Official DXC validation" in normalized_readme
    assert "profile `cs_6_0`, no `-enable-16bit-types`, and `-WX` passes" in (
        normalized_readme
    )
    assert "zero warnings across all 484 entry-point runs" in normalized_readme
    assert "two float32 block-reduction workloads" in normalized_readme
    assert "axis sizes 32 and 1025" in normalized_readme
    assert "current-corpus axis-size-32 artifact opts into" in normalized_readme
    assert "``WaveActiveMax(float)`` and ``WaveActiveSum(float)``" in (
        normalized_readme
    )
    assert "4,676 bytes" in normalized_readme
    assert (
        "`813762d4535fdd693ca0a48c3c3f5dc79f6cc298050faae6e180d3cc9f1d60e5`" in readme
    )
    assert "ten control-barrier instructions" in normalized_readme
    assert "`3.9978051379373145`" in readme
    assert "axis-size-1025 record still produces a ``[288, 1, 1]`` artifact" in (
        normalized_readme
    )
    assert "nine logical 32-lane subgroups" in normalized_readme
    assert "does not redirect the MLX runtime" in normalized_readme
    assert "records this as a warning-clean contract" in normalized_readme
    assert "rejects any newly observed warning" in normalized_readme
    assert "two selected `random.metal` entries compile without" in normalized_readme
    assert "All 18 rope entries compile with DXC profile `cs_6_2`" in (
        normalized_readme
    )
    assert "All 11 arange entries compile without the destination-conversion" in (
        normalized_readme
    )
    assert (
        "`arangeint16_out[index] = int16_t((uint(arangeint16_start) + "
        "(index * uint(arangeint16_step))));`"
    ) in normalized_readme
    assert (
        "`arangefloat16_out[index] = (arangefloat16_start + "
        "(float16_t(index) * arangefloat16_step));`"
    ) in normalized_readme
    assert "`index_1 = uint(((2 * pos.x) + (pos.y * stride)));`" in (normalized_readme)
    assert "`index_1 = uint((pos.x + (pos.y * stride)));`" in normalized_readme
    assert "and with DXC profile `cs_6_2`, `-enable-16bit-types`, and `-WX`" in (
        normalized_readme
    )
    assert "compiler acceptance and a warning-clean diagnostic contract" in (
        normalized_readme
    )
    assert "this selected float specialization contains no live" in normalized_readme
    assert "native-width 16-bit types" in normalized_readme
    assert "its report records `requiredCapabilities=[]`" in normalized_readme
    assert "remains resolved and validated elsewhere" in normalized_readme
    assert "but is not required by this selected specialization" in (normalized_readme)
    assert "needs no native-16-bit profile uplift" in normalized_readme
    assert "wave intrinsics keep the configured project and compiler target" in (
        normalized_readme
    )
    assert "scoped to `directx-12`" in normalized_readme
    assert "this selected `bits = 2` entry resolves" in normalized_readme
    assert "source `uint32_t` and generated HLSL `uint`" in normalized_readme
    assert "cross-version compiler invariant" in normalized_readme
    assert "does not claim runtime execution or numerical parity" in normalized_readme
    assert (
        "`out_[uint((out_index / writes_per_reduce))] = output;`"
    ) in normalized_readme
    assert "also emits one artifact with zero translation diagnostics" in (
        normalized_readme
    )
    assert "`affine_gather_qmv_fast_float_gs_32_b_2`" in readme
    assert "`values_per_thread = 16`" in readme
    assert "`inout float[16]` plus a base offset" in normalized_readme
    assert "all four writes in each `i += 4` iteration" in normalized_readme
    assert "read-only `uint8_t` view over its `uint32_t`" in normalized_readme
    assert "`gather_qmv` host dispatch" in normalized_readme
    assert "`MTL::Size group_dims(bk, 2, 1)`" in normalized_readme
    assert "`[numthreads(32, 2, 1)]`" in normalized_readme
    assert "`[WaveSize(32)]`" in normalized_readme
    assert "15,835 bytes" in normalized_readme
    assert "materializes the overloaded `elem_to_loc` helper" in normalized_readme
    assert "independent shape and stride resource offsets" in normalized_readme
    assert "passes its logical offset as `inout int64_t`" in normalized_readme
    assert "consumes all five updated values" in normalized_readme
    assert "DXC profile `cs_6_6` and `-WX`" in normalized_readme
    assert "does not dispatch the kernel through Direct3D" in normalized_readme
    assert "three source-qualified index-range assertions" in normalized_readme
    assert "`in_index + i`" in readme
    assert "`gindex`" in readme
    assert "`out_index / writes_per_reduce`" in readme
    assert "inclusive bounds `[0, 2147483647]`" in normalized_readme
    assert "6,642 bytes" in normalized_readme
    assert "`CROSSTL_SOFTWARE_SUBGROUP_WIDTH`" in readme
    assert "eight control barriers" in normalized_readme
    assert "no group-nonuniform instruction" in normalized_readme
    assert "OpenGL/SPIR-V 1.3" in normalized_readme
    assert "`spirv-val`" in readme
    assert "not inferred or enforced at runtime" in normalized_readme
    assert "eight packed `uint32` values of `27`" in normalized_readme
    assert "scale `-1`, and bias `3`" in normalized_readme
    assert "one deterministic affine-quantize workload" in normalized_readme
    for issue in (1497, 1515, 1799, 1800, 1801, 1802, 1894):
        assert f"https://github.com/CrossGL/crosstl/issues/{issue}" in readme

    directx = module.MLX_DIRECTX_QUANTIZED_FRONTIER_EVIDENCE
    assert directx["status"] == "translated-dxc-validated"
    assert directx["artifact_count"] == 1
    assert directx["translation_diagnostic_count"] == 0
    assert directx["configured_project_target"] == "directx-12"
    assert directx["compiler_target_profiles"] == ["directx-12"]
    assert directx["required_capabilities"] == []
    assert directx["generated_hlsl"] == {
        "sha256": "52569209d98f1bf2ae7fa645f2e4858a420f3920368e14aecb98c2ba9939ac8f",
        "size_bytes": 4357,
    }
    assert directx["materialization"] == {
        "reachable_specialization_count": 6,
        "concrete_specialization_count": 3,
        "pruned_candidate_count": 110861,
    }
    assert directx["native_16_bit_emission"] == {
        "status": "not-required-for-selected-entry",
        "issue": "https://github.com/CrossGL/crosstl/issues/1799",
        "support_status": "resolved-and-validated-elsewhere",
        "reason": "unreachable-native-width-materializations-pruned",
    }
    assert directx["concrete_static_assertion_evaluation"] == {
        "status": "resolved-for-selected-entry",
        "issue": "https://github.com/CrossGL/crosstl/issues/1800",
        "remaining_static_assertion_count": 0,
    }
    assert directx["compiler_validation"] == {
        "compiler": "dxc",
        "profile": "cs_6_0",
        "compiler_arguments": [],
        "warnings_as_errors": True,
        "status": "passed",
        "observed_failure_count": 0,
        "contextual_narrowing": {
            "status": "not-required-for-selected-entry",
            "issue": "https://github.com/CrossGL/crosstl/issues/1801",
            "resource": "out_",
            "resource_element_type": "uint",
            "source_specialized_type": "uint32_t",
            "generated_value_type": "uint",
            "conversion": "not-required",
            "generated_store": "out_[uint((out_index / writes_per_reduce))] = output;",
        },
    }
    assert directx["runtime_execution_attempted"] is False
    assert directx["numerical_parity_claimed"] is False

    opengl = module.MLX_OPENGL_QUANTIZED_FRONTIER_EVIDENCE
    assert opengl["status"] == "translated-toolchain-validated-native-loader-executed"
    assert opengl["commit"] == module.MLX_CORPUS_COMMIT
    assert opengl["project_translation"] == {
        "unit_count": 1,
        "artifact_record_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "emitted_target_file_count": 1,
        "project_diagnostic_count": 0,
        "workgroup_size": [32, 1, 1],
        "subgroup_width_rule_configured": False,
        "max_template_specializations": 128,
        "max_template_materialization_work": 4096,
    }
    assert opengl["materialization"] == {
        "reachable_specialization_count": 9,
        "concrete_specialization_count": 3,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 104702,
        "selected_parameters": {
            "T": "float",
            "bits": 2,
            "group_size": 32,
            "has_global_scale": False,
        },
    }
    assert opengl["index_range_assertion_evidence"] == {
        "assertion_count": 3,
        "assertions": [
            dict(assertion)
            for assertion in module.MLX_OPENGL_QUANTIZED_INDEX_RANGE_ASSERTIONS
        ],
        "inclusive_bounds": {"minimum": 0, "maximum": 2147483647},
        "contract_kind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtime_enforced": False,
    }
    assert opengl["toolchain_validation"] == {
        "compiler": "glslangValidator",
        "compiler_target": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validator_target": "SPIR-V 1.3",
        "status": "passed",
        "observed_failure_count": 0,
    }
    assert opengl["generated_glsl"] == {
        "sha256": "e4d8e5931bfc93f81e2c3686c102a1d676c9a3dcdfd6447e90918aa7581beecb",
        "size_bytes": 6642,
    }
    assert opengl["software_subgroup"]["width"] == 32
    assert opengl["software_subgroup"]["control_barrier_instruction_count"] == 8
    assert opengl["software_subgroup"]["group_non_uniform_instruction_count"] == 0
    assert opengl["resolved_by"] == [module.OPENGL_QUANTIZED_INDEX_TYPE_RESOLVED_ISSUE]
    assert opengl["runtime_execution"]["status"] == "passed"
    assert opengl["runtime_execution_attempted"] is True
    assert opengl["runtime_integration_included"] is True
    assert opengl["mlx_host_runtime_integration_included"] is False
    assert opengl["numerical_parity_claimed"] is True
    assert opengl["runtime_parity_claimed"] is True

    adjacent = module.MLX_DIRECTX_QUANTIZED_PRIVATE_POINTER_BOUNDARY_EVIDENCE
    assert adjacent["status"] == "translated-dxc-validated"
    assert adjacent["selected_entry_point"] == (
        "affine_gather_qmv_fast_float_gs_32_b_2"
    )
    assert adjacent["project_translation"] == {
        "unit_count": 1,
        "artifact_record_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "emitted_target_file_count": 1,
        "project_diagnostic_count": 0,
    }
    assert adjacent["materialization"] == {
        "reachable_specialization_count": 11,
        "concrete_specialization_count": 8,
        "pruned_candidate_count": 110861,
    }
    assert adjacent["source_contract"] == {
        "helper": "load_vector<T, U, values_per_thread, bits>",
        "caller_array": "thread U x_thread[values_per_thread]",
        "specialized_extent": 16,
        "loop_step": 4,
    }
    assert adjacent["private_array_aliasing"] == {
        "status": "passed",
        "helper": "load_vector_float_float_16_2",
        "parameter_mode": "inout",
        "base_offset_parameter": "x_thread_base",
        "extent": 16,
        "writes_per_iteration": 4,
    }
    assert adjacent["weight_byte_view"] == {
        "status": "passed",
        "helper": "qdot_float_16_2",
        "backing_element_type": "uint32_t",
        "view_element_type": "uint8_t",
        "access": "read",
        "lane_read_count": 4,
        "composed_offset_terms": [
            "w_offset * 4",
            "ws_offset",
            "row * in_vec_size_w",
            "wl_offset",
        ],
    }
    assert adjacent["index_helper_materialization"] == {
        "status": "passed",
        "helper": "elem_to_loc_uint32_t",
        "source_index_type": "uint32_t",
        "generated_index_type": "uint",
        "resource_offsets": ["x_shape_offset", "x_strides_offset"],
    }
    assert adjacent["pointer_reference_offset_writeback"] == {
        "status": "passed",
        "helper": "adjust_matrix_offsets_float",
        "offsets": [
            "x_offset",
            "w_offset",
            "scales_offset",
            "biases_offset",
            "y_offset",
        ],
        "downstream_helper": "qmv_fast_impl_float_32_2",
    }
    assert adjacent["execution_contract"] == {
        "status": "source-verified-and-emitted",
        "workgroup_size": [32, 2, 1],
        "subgroup_width": 32,
        "subgroup_width_enforcement": "WaveSize(32)",
        "minimum_shader_model": "6.6",
        "host_dispatch_provenance": {
            "source": "mlx/backend/metal/quantized.cpp",
            "function": "gather_qmv",
            "workgroup_expression": "MTL::Size group_dims(bk, 2, 1)",
            "bk": 32,
        },
    }
    assert adjacent["generated_hlsl"] == {
        "sha256": "b7d6251d27fcdafc003c85975bf5c5774a1fca0a3d4602b9e9ea5ef62673f76e",
        "size_bytes": 15835,
    }
    assert adjacent["compiler_validation"] == {
        "compiler": "dxc",
        "profile": "cs_6_6",
        "compiler_arguments": [],
        "warnings_as_errors": True,
        "status": "passed",
        "observed_failure_count": 0,
    }
    assert adjacent["artifact_emitted"] is True
    assert adjacent["native_validation_attempted"] is True
    assert adjacent["native_validation_status"] == "passed"
    assert adjacent["tracked_by"] == [
        "https://github.com/CrossGL/crosstl/issues/1497",
        "https://github.com/CrossGL/crosstl/issues/1518",
        "https://github.com/CrossGL/crosstl/issues/1546",
        "https://github.com/CrossGL/crosstl/issues/1786",
    ]
    assert adjacent["runtime_execution_attempted"] is False
    assert adjacent["numerical_parity_claimed"] is False

    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        gaps["directx_toolchain_status"]["selected_quantized_private_pointer_boundary"]
        == adjacent
    )


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


def test_closed_project_blockers_are_recorded_as_resolved():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    resolved_issues = {
        f"https://github.com/CrossGL/crosstl/issues/{number}"
        for number in (
            1312,
            1472,
            1476,
            1515,
            1516,
            1659,
            1672,
            1800,
            1801,
            1802,
        )
    }

    assert resolved_issues <= set(module.RESOLVED_FRONTIER_ISSUES)
    assert resolved_issues <= set(gaps["resolved_issues"])
    assert resolved_issues.isdisjoint(module.FULL_CORPUS_TRACKED_ISSUES)
    assert resolved_issues.isdisjoint(gaps["tracked_issues"])
    assert resolved_issues.isdisjoint(gaps["runtime_readiness_status"]["blocked_by"])
    for blocker_kind in (
        "translation_blocked_by",
        "validation_blocked_by",
        "semantic_blocked_by",
    ):
        assert resolved_issues.isdisjoint(gaps["full_corpus_scout"][blocker_kind])


def test_new_pin_resource_profile_workgroup_and_validation_contracts_are_tracked():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )
    resource_issue = "https://github.com/CrossGL/crosstl/issues/1669"
    profile_issue = "https://github.com/CrossGL/crosstl/issues/1670"
    workgroup_issue = "https://github.com/CrossGL/crosstl/issues/1671"
    checkpoint_issue = "https://github.com/CrossGL/crosstl/issues/1576"
    resolved_frontier_issues = {
        "https://github.com/CrossGL/crosstl/issues/1799",
        "https://github.com/CrossGL/crosstl/issues/1800",
        "https://github.com/CrossGL/crosstl/issues/1801",
        "https://github.com/CrossGL/crosstl/issues/1802",
    }
    native_profile_issue = "https://github.com/CrossGL/crosstl/issues/1799"
    narrowing_issue = "https://github.com/CrossGL/crosstl/issues/1801"
    arithmetic_conversion_issue = "https://github.com/CrossGL/crosstl/issues/1802"
    static_assertion_issue = "https://github.com/CrossGL/crosstl/issues/1800"
    private_pointer_issue = "https://github.com/CrossGL/crosstl/issues/1497"
    opengl_index_issue = module.OPENGL_QUANTIZED_INDEX_TYPE_RESOLVED_ISSUE

    assert resource_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert resource_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert profile_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert workgroup_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert workgroup_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert checkpoint_issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert checkpoint_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert checkpoint_issue in module.RESOLVED_FRONTIER_ISSUES
    assert module.FULL_CORPUS_VALIDATION_TRACKED_ISSUES == (profile_issue,)
    assert native_profile_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert narrowing_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert arithmetic_conversion_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert static_assertion_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert private_pointer_issue in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert private_pointer_issue in module.FULL_CORPUS_TRACKED_ISSUES
    assert opengl_index_issue not in module.FULL_CORPUS_TRANSLATION_TRACKED_ISSUES
    assert opengl_index_issue not in module.FULL_CORPUS_TRACKED_ISSUES
    assert opengl_index_issue in module.RESOLVED_FRONTIER_ISSUES
    assert resolved_frontier_issues <= set(module.RESOLVED_FRONTIER_ISSUES)
    assert resolved_frontier_issues.isdisjoint(module.FULL_CORPUS_TRACKED_ISSUES)
    assert resource_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert profile_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert workgroup_issue not in module.RESOLVED_FRONTIER_ISSUES
    assert native_profile_issue in module.RESOLVED_FRONTIER_ISSUES
    assert narrowing_issue in module.RESOLVED_FRONTIER_ISSUES
    assert arithmetic_conversion_issue in module.RESOLVED_FRONTIER_ISSUES
    assert resource_issue in gaps["tracked_issues"]
    assert profile_issue in gaps["tracked_issues"]
    assert workgroup_issue in gaps["tracked_issues"]
    assert checkpoint_issue not in gaps["tracked_issues"]
    assert checkpoint_issue in gaps["resolved_issues"]
    assert native_profile_issue not in gaps["tracked_issues"]
    assert narrowing_issue not in gaps["tracked_issues"]
    assert arithmetic_conversion_issue not in gaps["tracked_issues"]
    assert native_profile_issue in gaps["resolved_issues"]
    assert narrowing_issue in gaps["resolved_issues"]
    assert arithmetic_conversion_issue in gaps["resolved_issues"]
    assert static_assertion_issue not in gaps["tracked_issues"]
    assert static_assertion_issue in module.RESOLVED_FRONTIER_ISSUES
    assert static_assertion_issue in gaps["resolved_issues"]
    assert private_pointer_issue in gaps["tracked_issues"]
    assert opengl_index_issue not in gaps["tracked_issues"]
    assert opengl_index_issue in gaps["resolved_issues"]
    assert resolved_frontier_issues <= set(gaps["resolved_issues"])
    assert resolved_frontier_issues.isdisjoint(gaps["tracked_issues"])
    assert resource_issue in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert profile_issue in gaps["full_corpus_scout"]["validation_blocked_by"]
    assert workgroup_issue in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert checkpoint_issue not in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert native_profile_issue not in gaps["full_corpus_scout"]["semantic_blocked_by"]
    assert narrowing_issue not in gaps["full_corpus_scout"]["semantic_blocked_by"]
    assert (
        arithmetic_conversion_issue
        not in gaps["full_corpus_scout"]["semantic_blocked_by"]
    )
    assert gaps["full_corpus_scout"]["validation_blocked_by"] == [
        profile_issue,
    ]
    assert private_pointer_issue in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert opengl_index_issue not in gaps["full_corpus_scout"]["translation_blocked_by"]
    assert resolved_frontier_issues.isdisjoint(
        gaps["full_corpus_scout"]["validation_blocked_by"]
    )


def test_latest_full_corpus_attempt_records_interruption_without_coordinate_claim():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    attempt = gaps["full_corpus_scout"]["latest_attempt"]
    assert attempt == {
        "commit": module.MLX_COMMIT,
        "outcome": "timeout-before-report",
        "timeout_seconds": module.FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
        "return_code": 124,
        "report_produced": False,
        "generated_file_count": 8,
        "active_coordinate_claimed": False,
        "completed_target_artifacts": {
            module.MLX_ARANGE_SOURCE: ["directx", "opengl", "vulkan"]
        },
        "partial_target_artifacts": {module.MLX_ARG_REDUCE_SOURCE: ["vulkan"]},
        "blocked_by": [
            "https://github.com/CrossGL/crosstl/issues/1376",
            "https://github.com/CrossGL/crosstl/issues/1576",
        ],
    }


def test_full_corpus_checkpoint_probe_records_verified_resume_coordinate():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    assert gaps["full_corpus_scout"]["checkpoint_probe"] == {
        "commit": module.MLX_COMMIT,
        "targets": ["directx", "opengl"],
        "job_count": 80,
        "completed_count": 4,
        "pending_count": 75,
        "translated_artifact_count": 2,
        "failed_artifact_count": 2,
        "state": "interrupted",
        "active_coordinate": {
            "source": "mlx/backend/metal/kernels/binary.metal",
            "target": "directx",
        },
        "last_completed_coordinate": {
            "source": module.MLX_ARG_REDUCE_SOURCE,
            "target": "opengl",
        },
        "resume_verified": True,
        "checkpoint_schema_validated": True,
        "canonical_report_produced": False,
        "runtime_parity_claimed": False,
    }


def test_arg_reduce_native_runtime_evidence_records_bounded_cross_target_proof():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["arg_reduce_native_runtime_status"]
    assert status == module.MLX_ARG_REDUCE_NATIVE_RUNTIME_EVIDENCE
    assert status["status"] == (
        "translated-packaged-and-cross-target-native-runtime-required"
    )
    assert status["commit"] == module.MLX_CORPUS_COMMIT
    assert status["source"] == module.MLX_ARG_REDUCE_SOURCE
    assert status["source_sha256"] == module.MLX_CURRENT_ARG_REDUCE_SHA256
    assert status["selected_entry_points"] == ["argmin_float32", "argmax_float32"]

    contract = status["dispatch_contract"]
    assert contract["path"] == (
        "demos/integrations/mlx/contracts/" "arg_reduce.native-loader.dispatch.json"
    )
    assert contract["content_identity"] == (
        module.MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY
    )
    assert contract["workload_count"] == 2
    assert contract["bounded_axis_size"] == 32
    assert contract["subgroup_width"] == 32
    assert contract["variants"] == module.MLX_ARG_REDUCE_NATIVE_LOADER_DISPATCH_VARIANTS
    assert set(contract["variants"]) == {
        "argmin-float32-axis-32-two-rows",
        "argmax-float32-axis-32-two-rows",
    }
    for variant in contract["variants"].values():
        assert variant["workgroup_size"] == [32, 1, 1]
        assert variant["dispatch_workgroup_count"] == [1, 2, 1]

    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 2, "opengl": 2},
        "translated_count_by_target": {"directx": 2, "opengl": 2},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
    }
    overload = status["materialization"]["overloaded_helper_selection"]
    assert overload == {
        "source_name": "elem_to_loc",
        "materialized_name": "elem_to_loc_int64_t",
        "selected_first_parameter": "int64_t",
        "rejected_first_parameter": "uint3",
        "resolution": "signature-aware-call-site-materialization",
    }

    directx = status["artifacts"]["directx"]
    assert directx["argmin_float32"] == {
        "sha256": "b63a160f4ca102cc6407d88b84f5c3e6f849840f73e57d372722f9828569a34d",
        "size_bytes": 6655,
    }
    assert directx["argmax_float32"] == {
        "sha256": "99359b364f701a420f1ee918f481eaabac7d5ea9b02c5872c915c7672c60b398",
        "size_bytes": 6657,
    }
    assert directx["compiler"] == "dxc"
    assert directx["compiler_version"] == "1.9.2602.24"
    assert directx["compiler_profile"] == "cs_6_6"
    assert directx["compiler_arguments"] == ["-enable-16bit-types", "-WX"]
    assert directx["compiler_validation_status"] == "passed"
    assert status["directx_relative_shuffle"] == {
        "configuration": (
            "project.source_options.metal.target_options.directx."
            "relative_wave_shuffle_out_of_range"
        ),
        "policy": "self",
        "default_policy": "undefined",
        "activation": "explicit-target-scoped",
        "selected_value_types": ["float", "uint"],
        "invalid_source_lane_result": "calling-lane-value",
        "invalid_source_lane_reads_emitted": False,
        "source_lane_bounds": {
            "down_valid_when": "delta < laneCount - lane",
            "read_lane": "valid ? lane + delta : lane",
        },
        "wave_read_control_flow": "single-unconditional-selected-lane-read",
    }
    opengl = status["artifacts"]["opengl"]
    assert opengl["compiler"] == "glslangValidator"
    assert opengl["validator"] == "spirv-val"
    assert opengl["control_barrier_instruction_count"] == 5
    assert opengl["group_non_uniform_instruction_count"] == 0
    assert opengl["compiler_validation_status"] == "passed"

    assert status["runtime_package"] == {
        "artifact_count_per_variant_and_target": 1,
        "ready_load_unit_count_per_variant_and_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count_by_target": {"directx": 9, "opengl": 8},
        "resource_element_types": [
            "float32",
            "uint32",
            "int32",
            "int64",
            "uint64",
        ],
        "directx_generated_dispatch_binding": "CrossGLDispatchInfo",
        "scalar_64_bit_layout_reflected": True,
    }
    assert status["workloads"]["argmin_expected_indices"] == [5, 7]
    assert status["workloads"]["argmax_expected_indices"] == [3, 2]
    assert status["workloads"]["tie_behavior"] == "lowest index"
    assert status["native_runtime"]["directx"]["status"] == "required-on-ci"
    assert status["native_runtime"]["opengl"]["status"] == "required-on-ci"
    assert status["native_runtime"]["opengl"]["local_linux_arm64_validation"] == (
        "passed"
    )
    assert status["metal_roundtrip_boundary"] == {
        "status": "entry-workgroup-specialization-target-unsupported",
        "diagnostic": "project.translate.workgroup-size-rule-unsupported-target",
        "missing_capability": "execution.workgroup-size-specialization",
    }
    assert all(value is False for value in status["remaining_scope"].values())
    assert status["selected_workloads_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False


def test_scaled_attention_native_runtime_evidence_records_bounded_cross_target_proof():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["scaled_dot_product_attention_native_runtime_status"]
    assert status == module.MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_RUNTIME_EVIDENCE
    assert status["status"] == (
        "translated-packaged-and-cross-target-native-runtime-required"
    )
    assert status["commit"] == module.MLX_CORPUS_COMMIT
    assert status["source"] == module.MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    assert status["source_sha256"] == (
        module.MLX_CURRENT_SCALED_DOT_PRODUCT_ATTENTION_SHA256
    )
    assert status["selected_entry_point"] == "sdpa_vector_float_64_64"

    contract = status["dispatch_contract"]
    assert contract["path"] == (
        "demos/integrations/mlx/contracts/"
        "scaled_dot_product_attention.native-loader.dispatch.json"
    )
    assert contract["content_identity"] == (
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY
    )
    assert contract["workload_count"] == 1
    assert contract["host_selection"] == "one-pass-vector"
    assert contract["subgroup_width"] == 32
    assert contract["variants"] == (
        module.MLX_SCALED_DOT_PRODUCT_ATTENTION_NATIVE_LOADER_DISPATCH_VARIANTS
    )
    variant = contract["variants"]["vector-float32-b1-h1-q1-k4-d64-v64-nomask"]
    assert variant["artifact_id"] == (
        "sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89"
    )
    assert variant["dispatch_variant_id"] == (
        "sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb"
    )
    assert variant["workgroup_size"] == [1024, 1, 1]
    assert variant["dispatch_workgroup_count"] == [1, 1, 1]
    assert variant["specialization_constants"] == {
        str(constant_id): False for constant_id in range(20, 26)
    }

    assert status["materialization"] == {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 753,
        "selected_parameters": {"D": "64", "T": "float", "V": "64"},
    }
    assert set(status["specialization_constants"]) == {
        str(constant_id) for constant_id in range(20, 26)
    }
    assert "26" not in status["specialization_constants"]
    assert all(
        constant["value"] is False
        for constant in status["specialization_constants"].values()
    )

    directx = status["artifacts"]["directx"]
    assert directx["sha256"] == (
        "2182a09b1e03815f11e36c3ab1addb2138257e0bcf69284f99a0c33ec344816b"
    )
    assert directx["size_bytes"] == 8721
    assert directx["compiler_version"] == "1.9.2602.24"
    assert directx["compiler_profile"] == "cs_6_6"
    assert directx["compiler_arguments"] == ["-enable-16bit-types", "-WX"]
    assert directx["compiled_dxil_size_bytes"] == 9000
    assert directx["compiler_validation_status"] == "passed"
    assert directx["subgroup_id_lowering"] == (
        "workgroup-synchronized-physical-wave-allocation"
    )

    opengl = status["artifacts"]["opengl"]
    assert opengl["sha256"] == (
        "9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b"
    )
    assert opengl["size_bytes"] == 12089
    assert opengl["control_barrier_instruction_count"] == 9
    assert opengl["group_non_uniform_instruction_count"] == 0
    assert opengl["specialization_constant_false_count"] == 6
    assert opengl["specialization_materialization"] == "deferred"

    package = status["runtime_package"]
    assert package["resource_count_by_target"] == {"directx": 19, "opengl": 18}
    assert package["specialization_constant_count"] == 6
    assert package["stored_bool_physical_type"] == "uint32"
    assert package["optional_placeholder_resources"] == [
        "bmask",
        "fmask",
        "sinks",
    ]
    assert package["opengl_native_registry_header"] == {
        "available": False,
        "reason": "specialization-requires-deferred-compilation",
    }

    assert status["workload"]["key_length"] == 4
    assert status["workload"]["query_dimension"] == 64
    assert status["workload"]["value_dimension"] == 64
    assert status["workload"]["mask"] == "none"
    assert status["native_runtime"]["directx"]["status"] == "required-on-ci"
    assert status["native_runtime"]["opengl"]["status"] == "required-on-ci"
    assert status["native_runtime"]["opengl"]["local_linux_mesa_validation"] == (
        "passed"
    )
    assert status["native_runtime"]["opengl"]["local_max_absolute_error"] < 5e-8
    assert status["native_runtime"]["opengl"]["local_max_relative_error"] < 5e-6
    assert all(value is False for value in status["remaining_scope"].values())
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects current-pinned `sdpa_vector_float_64_64`" in readme
    assert "partitions 1,024 invocations into 32 logical subgroups" in readme
    assert "six `OpSpecConstantFalse` declarations" in readme
    assert "no bounded attention dispatch, package, or numerical runtime proof" in (
        readme
    )

    guide = " ".join(
        (ROOT / "docs" / "source" / "project-porting.rst")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "bounded one-pass scaled-attention proof" in guide
    assert "inactive subgroups supply typed reduction identities" in guide
    assert "it is not evidence that a bounded attention runtime proof is absent" in (
        guide
    )


def test_softmax_native_runtime_evidence_records_bounded_cross_target_proof():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["softmax_native_runtime_status"]
    assert status == module.MLX_SOFTMAX_NATIVE_RUNTIME_EVIDENCE
    assert status["status"] == (
        "translated-packaged-and-cross-target-native-runtime-required"
    )
    assert status["commit"] == module.MLX_CORPUS_COMMIT
    assert status["source"] == module.MLX_SOFTMAX_SOURCE
    assert status["source_sha256"] == module.MLX_CURRENT_SOFTMAX_SHA256
    assert status["selected_entry_point"] == "block_softmax_float32"

    contract = status["dispatch_contract"]
    assert contract["path"] == (
        "demos/integrations/mlx/contracts/softmax.native-loader.dispatch.json"
    )
    assert contract["content_identity"] == (
        module.MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY
    )
    assert contract["workload_count"] == 2
    assert contract["host_formula"] == "32 * ceilDiv(ceilDiv(axisSize, 4), 32)"
    assert contract["block_limit"] == 4096
    assert contract["subgroup_width"] == 32
    assert contract["variants"] == module.MLX_SOFTMAX_NATIVE_LOADER_DISPATCH_VARIANTS
    assert set(contract["variants"]) == {
        "block-float32-axis-32-two-rows",
        "block-float32-axis-2049",
    }
    assert contract["variants"]["block-float32-axis-32-two-rows"]["workgroup_size"] == [
        32,
        1,
        1,
    ]
    assert contract["variants"]["block-float32-axis-2049"]["workgroup_size"] == [
        544,
        1,
        1,
    ]

    assert status["materialization"] == {
        "concrete_specialization_count": 2,
        "reachable_specialization_count": 5,
        "dependency_discovery_work_count": 11,
        "pruned_candidate_count": 131,
        "selected_parameters": {
            "AccT": "float",
            "N_READS": "SOFTMAX_N_READS",
            "T": "float",
        },
    }
    assert status["guarded_artifacts"]["directx"]["block-float32-axis-32-two-rows"] == {
        "sha256": "de5ae241c037cf9a0b37f456d777d51c544c8f725cf2efe8a51c45ae968a0fc2",
        "size_bytes": 4213,
        "subgroup_id_lowering": "fixed-single-wave-group-index-quotient",
    }
    assert status["guarded_artifacts"]["directx"]["block-float32-axis-2049"] == {
        "sha256": "3a3f443fdb6df38e37bda4828601b967cc6aa216c0805e6dc060be7e7bddd02c",
        "size_bytes": 4784,
        "subgroup_id_lowering": "workgroup-synchronized-physical-wave-allocation",
    }
    directx_artifacts = status["guarded_artifacts"]["directx"]
    assert directx_artifacts["compiler_profile"] == "cs_6_6"
    assert directx_artifacts["compiler_arguments"] == ["-enable-16bit-types"]
    assert directx_artifacts["warnings_as_errors"] is True
    assert directx_artifacts["compiler_validation_status"] == "passed"
    software = status["software_opengl_artifacts"]
    assert software["block-float32-axis-32-two-rows"]["sha256"] == (
        "f69dad597cefc34f7908799aaf0ba2eac47a0dcdd91e5f2bf3d7247172fa84b9"
    )
    assert software["block-float32-axis-32-two-rows"]["size_bytes"] == 5585
    assert software["block-float32-axis-2049"]["sha256"] == (
        "eb195e15089f4e7bade380af55e8b7e167c4b89f80b2f25675eb71196a5468ce"
    )
    assert software["block-float32-axis-2049"]["size_bytes"] == 7204
    assert software["block-float32-axis-2049"]["logical_subgroup_count"] == 17
    assert software["block-float32-axis-2049"]["masked_collective_count"] == 2
    for workload_id in (
        "block-float32-axis-32-two-rows",
        "block-float32-axis-2049",
    ):
        assert software[workload_id]["control_barrier_instruction_count"] == 11
        assert software[workload_id]["group_non_uniform_instruction_count"] == 0

    subgroup = status["software_subgroup"]
    assert subgroup["selected_kernel_operations"] == [
        "WaveActiveMax(float)",
        "WaveActiveSum(float)",
    ]
    assert subgroup["masked_reduction_operations"] == [
        "WaveActiveSum",
        "WaveActiveMin",
        "WaveActiveMax",
    ]
    assert subgroup["typed_mask_identities"]["WaveActiveMax"] == {
        "float": "-infinity",
        "int": "INT_MIN",
        "uint": "0u",
    }
    assert subgroup["masked_shuffle_supported"] is False
    assert status["runtime_package"]["resource_count"] == 3
    assert status["runtime_package"]["blocked_load_unit_count"] == 0
    assert status["native_runtime"]["directx"]["status"] == "required-on-ci"
    assert status["native_runtime"]["opengl"]["status"] == "required-on-ci"
    assert (
        status["native_runtime"]["opengl"]["local_linux_arm64_validation"] == "passed"
    )
    assert status["metal_roundtrip_boundary"] == {
        "status": "entry-scoped-target-unsupported",
        "diagnostic": "project.translate.entry-point-target-unsupported",
        "missing_capability": "artifact.entry-point-selection",
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects current-pinned `block_softmax_float32`" in readme
    assert "Axis size 2049 with one row uses `[544, 1, 1]`" in readme
    assert "17 independent logical subgroups" in readme
    assert "negative infinity for float maximum and zero for float sum" in readme
    assert "does not mean that no bounded Softmax translation" in readme

    guide = " ".join(
        (ROOT / "docs" / "source" / "project-porting.rst")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "a bounded Softmax proof for ``block_softmax_float32``" in guide
    assert "typed inactive-lane identities" in guide
    assert "does not claim a Metal round trip" in guide


def test_rms_norm_native_runtime_evidence_records_bounded_cross_target_proof():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["rms_norm_native_runtime_status"]
    assert status == module.MLX_RMS_NORM_NATIVE_RUNTIME_EVIDENCE
    assert status["status"] == "translated-packaged-and-native-runtime-required"
    assert status["commit"] == module.MLX_CORPUS_COMMIT
    assert status["source"] == module.MLX_RMS_NORM_SOURCE
    assert status["source_sha256"] == module.MLX_CURRENT_RMS_NORM_SHA256
    assert status["selected_entry_point"] == "rmsfloat32"
    assert status["selected_workload"] == "forward-float32-axis-32"
    assert status["dispatch_contract"] == {
        "path": (
            "demos/integrations/mlx/contracts/" "rms_norm.native-loader.dispatch.json"
        ),
        "content_identity": module.MLX_RMS_NORM_NATIVE_LOADER_DISPATCH_CONTENT_IDENTITY,
        "artifact_id": (
            "sha256:00c05fccf276cf11f3fb9b617b8fe0bb3c5f8766c0e4ca1ed990c093e700422e"
        ),
        "workload_count": 1,
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [2, 1, 1],
        "function_constants": {},
    }
    assert status["entry_scoped_specialization_ownership"] == {
        "constant_name": "has_w",
        "constant_id": 20,
        "reachable_from_selected_entry": False,
        "artifact_specialization_constant_count": 0,
        "runtime_manifest_specialization_constant_count": 0,
        "reachable_vjp_constants_preserved": True,
        "resolved_issue": "https://github.com/CrossGL/crosstl/issues/1795",
    }
    assert status["materialization"] == {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 168,
        "selected_parameters": {"N_READS": "RMS_N_READS", "T": "float"},
    }
    assert status["artifacts"]["directx"]["sha256"] == (
        "83f7b6e437122b2afe3dbc5d7649f6bc882d671947bcc72579ae5aa568fb2a5b"
    )
    assert status["artifacts"]["directx"]["size_bytes"] == 3486
    assert status["artifacts"]["directx"]["native_runtime"]["status"] == (
        "required-on-ci"
    )
    assert status["artifacts"]["opengl"]["sha256"] == (
        "3180aba83b64add0ae3c2d471b9297eb5bada4c4ff2bd5c91a3db3698cf0df78"
    )
    assert status["artifacts"]["opengl"]["size_bytes"] == 4393
    assert status["artifacts"]["opengl"]["control_barrier_instruction_count"] == 6
    assert status["artifacts"]["opengl"]["group_non_uniform_instruction_count"] == 0
    assert status["artifacts"]["opengl"]["native_runtime"]["status"] == (
        "required-on-ci"
    )
    assert status["software_subgroup"]["operations"] == ["WaveActiveSum(float)"]
    assert status["software_subgroup"]["width"] == 32
    assert status["runtime_package"]["resource_count"] == 6
    assert status["runtime_package"]["ready_load_unit_count_by_target"] == {
        "directx": 1,
        "opengl": 1,
    }
    assert status["runtime_package"]["blocked_load_unit_count_by_target"] == {
        "directx": 0,
        "opengl": 0,
    }
    assert status["workload"] == {
        "dtype": "float32",
        "shape": [2, 32],
        "axis_size": 32,
        "row_count": 2,
        "weight_shape": [32],
        "epsilon": 0.00001,
        "weight_stride": 1,
        "input_values": "row0=(index-16)/8; row1=((index%9)-4)*0.3125",
        "weight_values": "0.5+(index%5)*0.125",
        "reference": "x*w*rsqrt(mean(x*x)+epsilon)",
        "output_element_count": 64,
        "absolute_tolerance": 0.00003,
        "relative_tolerance": 0.00003,
    }
    assert status["remaining_scope"] == {
        "forward_entries_other_than_rmsfloat32_included": False,
        "vjp_entries_included": False,
        "float16_and_bfloat16_included": False,
        "looped_entries_included": False,
        "other_axis_sizes_included": False,
        "historical_compiler_dispatch_record_count": 12,
        "mlx_host_runtime_integration_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    specialization = gaps["rms_norm_specialization_status"]
    assert specialization["status"] == (
        "translation-native-compilation-and-bounded-runtime-validated"
    )
    assert specialization["numerical_execution_included"] is True
    assert specialization["runtime_integration_included"] is True
    assert specialization["selected_workload_numerical_parity_verified"] is True
    assert specialization["bounded_native_runtime_evidence"] == (
        "rms_norm_native_runtime_status"
    )
    assert specialization["runtime_blocked_by"] == []
    issue = "https://github.com/CrossGL/crosstl/issues/1795"
    assert issue not in gaps["tracked_issues"]
    assert issue in gaps["resolved_issues"]

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects the current-corpus `rmsfloat32` entry" in readme
    assert "two deterministic float32 rows of axis size 32" in readme
    assert "six-buffer native-loader ABI" in readme
    assert "six `OpControlBarrier` instructions" in readme
    assert "Direct3D 12 WARP and Mesa software OpenGL" in readme
    assert "does not cover VJP, looped, float16, or bfloat16 entries" in readme


def test_rms_norm_vjp_native_runtime_evidence_records_deferred_cross_target_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["rms_norm_vjp_native_runtime_status"]
    assert status["status"] == (
        "translated-packaged-deferred-specialized-and-native-runtime-required"
    )
    assert status["commit"] == "846d176227a0ac13d2667e58d2bb68b322109ab0"
    assert status["source"] == "mlx/backend/metal/kernels/rms_norm.metal"
    assert status["source_sha256"] == (
        "b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf"
    )
    assert status["selected_entry_point"] == "vjp_rmsfloat32"
    assert status["selected_workload"] == "vjp-float32-axis-32-one-row-has-w"
    assert status["upstream_test"] == "python/tests/test_fast.py::test_rms_norm_grad"
    assert status["dispatch_contract"] == {
        "path": (
            "demos/integrations/mlx/contracts/"
            "rms_norm_vjp.native-loader.dispatch.json"
        ),
        "content_identity": (
            "sha256:6b80c42a03de10db01881cbf2ca01c119ee4537cb5c221b0be9efcff138edfb3"
        ),
        "dispatch_variant_id": (
            "sha256:5f30015f711d2884061ea69c110e54a5ac2e1c03361315e6cac9de2b2c7891a5"
        ),
        "artifact_id": (
            "sha256:a9be06b43a6156fb9ee1f9a6955d03d6bda0940c2a8223b58f564c2d12bd0cd0"
        ),
        "workload_count": 1,
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [1, 1, 1],
        "function_constants": {"20": True},
    }
    assert status["index_range_contracts"] == [
        {
            "expression": "uint64(row) * axis_size + lid * RMS_N_READS",
            "minimum": 0,
            "maximum": 31,
        },
        {
            "expression": "uint64(gid) * axis_size + lid * RMS_N_READS",
            "minimum": 0,
            "maximum": 31,
        },
    ]
    assert status["materialization"] == {
        "concrete_specialization_count": 1,
        "reachable_specialization_count": 4,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 168,
        "selected_specializations": {
            "vjp_rms_single_row": {
                "N_READS": "RMS_N_READS",
                "T": "float",
            }
        },
    }
    specialization = status["specialization"]
    assert specialization["name"] == "has_w"
    assert specialization["constant_id"] == 20
    assert specialization["value"] is True
    assert specialization["directx_materialization"] == "concrete-crossgl-variant"
    assert specialization["opengl_materialization"] == "deferred-layout-constant-id"
    assert specialization["native_header_available"] is False
    assert specialization["native_header_unavailable_reason"] == (
        "specialization-requires-deferred-compilation"
    )
    assert specialization["deferred_compilation_request_status"] == "ready"
    assert specialization["deferred_output_format"] == "SPIR-V binary"

    directx = status["artifacts"]["directx"]
    assert directx["sha256"] == (
        "008a7a6f1614cb7c087d11c7e65adef919b58504e5fff3bd6479940aebc0aa1d"
    )
    assert directx["size_bytes"] == 6795
    assert directx["wave_active_sum_call_count"] == 4
    assert directx["native_runtime"]["status"] == "required-on-ci"
    opengl = status["artifacts"]["opengl"]
    assert opengl["sha256"] == (
        "2112adeb6c1693fa42c48fe3013cd57637f34a9393c0d468b547ed06ab42cf73"
    )
    assert opengl["size_bytes"] == 7771
    assert opengl["specialization_enforcement"] == (
        "deferred-opengl-spirv-specialization"
    )
    assert opengl["control_barrier_instruction_count"] == 6
    assert opengl["group_non_uniform_instruction_count"] == 0
    assert opengl["local_validation"] == {
        "platform": "linux-arm64",
        "runtime": "mesa-headless-egl-llvmpipe",
        "status": "passed",
        "gx_maximum_absolute_error": 3.5336688131160088e-08,
        "gx_maximum_relative_error": 2.1698385352267783e-06,
        "gw_maximum_absolute_error": 3.3944730581936255e-08,
        "gw_maximum_relative_error": 6.02581746340496e-08,
    }
    assert opengl["native_runtime"]["status"] == "required-on-ci"
    assert status["software_subgroup"]["operations"] == ["WaveActiveSum(float)"]
    assert status["software_subgroup"]["width"] == 32
    assert status["software_subgroup"]["runtime_loop_uniformity"] == {
        "accepted_form": "canonical-int-or-uint-for-loop",
        "initializer_and_bound": "proven-workgroup-uniform",
        "uniform_sources": [
            "compile-time constants",
            "read-only scalar blocks",
            "workgroup builtins",
            "conservative local dataflow",
        ],
        "lane_varying_or_mutating_bounds_rejected": True,
        "escaping_control_flow_rejected": True,
        "unresolved_calls_rejected": True,
    }
    assert status["runtime_package"]["resource_count"] == 10
    assert status["runtime_package"]["specialization_constant_count"] == 1
    assert [
        resource["role"] for resource in status["runtime_package"]["resources"]
    ] == [
        "input",
        "weight",
        "output_cotangent",
        "input_gradient",
        "per_group_weight_gradient",
        "epsilon",
        "axis_size",
        "weight_stride",
        "row_count",
        "rows_per_group",
    ]
    assert status["one_row_boundary"] == {
        "per_group_weight_gradient_equals_final_reduced_gradient": True,
        "follow_on_weight_reduction_dispatch_required": False,
    }
    assert status["remaining_scope"] == {
        "multi_row_weight_reduction_included": False,
        "has_w_false_included": False,
        "float16_and_bfloat16_included": False,
        "looped_entries_included": False,
        "other_axis_sizes_included": False,
        "mlx_host_runtime_integration_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects `vjp_rmsfloat32`" in readme
    assert "deferred OpenGL specialization constant `20`" in readme
    assert "initializer and bound workgroup-uniform" in readme
    assert "ten-resource native-loader ABI" in readme
    assert "one row and one group" in readme
    assert "does not cover multi-row weight reduction" in readme


def test_layer_norm_native_runtime_evidence_records_bounded_cross_target_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["layer_norm_native_runtime_status"]
    assert status["status"] == "translated-packaged-and-native-runtime-required"
    assert status["commit"] == "846d176227a0ac13d2667e58d2bb68b322109ab0"
    assert status["source"] == "mlx/backend/metal/kernels/layer_norm.metal"
    assert status["source_sha256"] == (
        "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
    )
    assert status["selected_entry_point"] == "layer_normfloat32"
    assert status["selected_workload"] == "forward-float32-axis-32"
    assert status["upstream_test"] == "python/tests/test_fast.py::test_layer_norm"
    assert status["dispatch_contract"] == {
        "path": (
            "demos/integrations/mlx/contracts/" "layer_norm.native-loader.dispatch.json"
        ),
        "content_identity": (
            "sha256:320929bc503640b12748a28f72aa571f9a79123e498d5f8cda4a9827c2d01add"
        ),
        "dispatch_variant_id": (
            "sha256:3ffe2a2e8450658c4972b627679caa9a16290f4f596eab2af26888433de9870f"
        ),
        "artifact_id": (
            "sha256:4b1c27c05949e3021e5b28aeb4552e668aad8dd00141a1880fe98e7ebc7b129e"
        ),
        "workload_count": 1,
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [2, 1, 1],
        "function_constants": {},
    }
    assert status["workgroup_access_contract"] == {
        "kind": "explicit-host-runtime-portability-precondition",
        "source": "mlx/backend/metal/kernels/layer_norm.metal",
        "entry_point": "layer_normfloat32",
        "function": "*",
        "parameter": "*",
        "minimum": 0,
        "maximum": 31,
        "inferred": False,
        "runtime_enforced": False,
    }
    assert status["materialization"] == {
        "concrete_specialization_count": 3,
        "reachable_specialization_count": 6,
        "dependency_discovery_work_count": 8,
        "pruned_candidate_count": 194,
        "selected_specializations": {
            "layer_norm_single_row": {"N_READS": "8", "T": "float"},
            "initialize_buffer": {"N": "1"},
            "threadgroup_sum": {"N": "1"},
        },
    }
    assert status["artifacts"]["directx"]["sha256"] == (
        "7fea4cd2ecf9b636ef2aa9a1a588e4186704bff5cb80163820c02fdb113194ba"
    )
    assert status["artifacts"]["directx"]["size_bytes"] == 5216
    assert status["artifacts"]["directx"]["wave_active_sum_call_count"] == 2
    assert status["artifacts"]["directx"]["native_runtime"]["status"] == (
        "required-on-ci"
    )
    assert status["artifacts"]["opengl"]["sha256"] == (
        "f86f83b6835b7d4b07ece9f153df883300f7a131bbcec5d084bf29084c1bf51a"
    )
    assert status["artifacts"]["opengl"]["size_bytes"] == 5914
    assert status["artifacts"]["opengl"]["control_barrier_instruction_count"] == 6
    assert status["artifacts"]["opengl"]["group_non_uniform_instruction_count"] == 0
    assert status["artifacts"]["opengl"]["local_validation"] == {
        "platform": "linux-arm64",
        "runtime": "mesa-headless-egl-llvmpipe",
        "status": "passed",
        "maximum_absolute_error": 8.876972712457132e-08,
        "maximum_relative_error": 2.9057866258306256e-07,
    }
    assert status["artifacts"]["opengl"]["native_runtime"]["status"] == (
        "required-on-ci"
    )
    software = status["software_subgroup"]
    assert software["operations"] == ["WaveActiveSum(float)"]
    assert software["width"] == 32
    assert software["approved_helpers"] == ["threadgroup_sum_1"]
    assert software["rejected_helper_contracts"] == [
        "conditional-or-nested helper calls",
        "indirect helper calls",
        "ambiguous helper identities",
        "subgroup operations in potentially divergent helper control flow",
        "narrower or base-incompatible workgroup proof reuse",
    ]
    assert software["control_barrier_instruction_count"] == 6
    assert software["group_non_uniform_instruction_count"] == 0
    assert software["hardware_subgroup_extensions_emitted"] is False
    assert software["unsupported_contract_behavior"] == (
        "reject-before-artifact-emission"
    )
    assert status["runtime_package"]["resource_count"] == 8
    assert status["runtime_package"]["specialization_constant_count"] == 0
    assert status["runtime_package"]["ready_load_unit_count_by_target"] == {
        "directx": 1,
        "opengl": 1,
    }
    assert status["runtime_package"]["blocked_load_unit_count_by_target"] == {
        "directx": 0,
        "opengl": 0,
    }
    assert [
        resource["role"] for resource in status["runtime_package"]["resources"]
    ] == [
        "input",
        "weight",
        "bias",
        "output",
        "epsilon",
        "axis_size",
        "weight_stride",
        "bias_stride",
    ]
    assert status["workload"] == {
        "dtype": "float32",
        "shape": [2, 32],
        "axis_size": 32,
        "row_count": 2,
        "weight_shape": [32],
        "bias_shape": [32],
        "epsilon": 0.00001,
        "weight_stride": 1,
        "bias_stride": 1,
        "input_values": "row0=(index-16)/8; row1=((index%9)-4)*0.3125",
        "weight_values": "0.5+(index%5)*0.125",
        "bias_values": "((index%7)-3)*0.0625",
        "reference": "((x-mean(x))/sqrt(mean((x-mean(x))^2)+epsilon))*weight+bias",
        "output_element_count": 64,
        "absolute_tolerance": 0.00005,
        "relative_tolerance": 0.00005,
    }
    assert status["remaining_scope"] == {
        "forward_entries_other_than_layer_normfloat32_included": False,
        "vjp_entries_included": False,
        "float16_and_bfloat16_included": False,
        "looped_entries_included": False,
        "other_axis_sizes_included": False,
        "historical_axis_4096_and_vjp_dispatch_frontier_included": False,
        "mlx_host_runtime_integration_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    historical = gaps["directx_toolchain_status"]["layer_norm_dispatch_frontier"]
    assert historical["bounded_axis_32_native_runtime_evidence"] == (
        "layer_norm_native_runtime_status"
    )
    assert historical["runtime_execution_attempted"] is False
    assert historical["runtime_execution_scope"] == (
        "historical-axis-4096-forward-and-vjp-variants-only"
    )

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects the forward `layer_normfloat32` entry" in readme
    assert "workgroup-access precondition bounds specialized helper views" in readme
    assert "one stable non-overloaded source identity" in readme
    assert "eight ready resources on both targets" in readme
    assert "maximum absolute error below `8.88e-8`" in readme
    assert "does not redirect MLX host execution" in readme


def test_layer_norm_vjp_native_runtime_evidence_records_deferred_cross_target_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["layer_norm_vjp_native_runtime_status"]
    assert status["status"] == (
        "translated-packaged-deferred-specialized-and-native-runtime-required"
    )
    assert status["commit"] == "846d176227a0ac13d2667e58d2bb68b322109ab0"
    assert status["source"] == "mlx/backend/metal/kernels/layer_norm.metal"
    assert status["source_sha256"] == (
        "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
    )
    assert status["selected_entry_point"] == "vjp_layer_normfloat32"
    assert status["selected_workload"] == ("vjp-float32-axis-32-one-row-has-w")
    assert status["upstream_test"] == (
        "python/tests/test_fast.py::test_layer_norm_grad"
    )
    assert status["dispatch_contract"] == {
        "path": (
            "demos/integrations/mlx/contracts/"
            "layer_norm_vjp.native-loader.dispatch.json"
        ),
        "content_identity": (
            "sha256:e0e47ea5589f6fa812990617f0ac9f0de4f7749054a7f68ab1959dbcb5f8859c"
        ),
        "dispatch_variant_id": (
            "sha256:7bbb4b949deb3bfcd443016f4e5d8bb46f297ffb5f8508a9fd73619496c41cf3"
        ),
        "artifact_id": (
            "sha256:89b99b1316177ae5a2cae5a0aa57767219e01e9c4efc0497f6831e5e367fd9db"
        ),
        "workload_count": 1,
        "workgroup_size": [32, 1, 1],
        "subgroup_width": 32,
        "dispatch_workgroup_count": [1, 1, 1],
        "function_constants": {"20": True},
    }
    assert status["workgroup_access_contract"] == {
        "kind": "explicit-host-runtime-portability-precondition",
        "source": "mlx/backend/metal/kernels/layer_norm.metal",
        "entry_point": "vjp_layer_normfloat32",
        "function": "*",
        "parameter": "*",
        "minimum": 0,
        "maximum": 95,
        "inferred": False,
        "runtime_enforced": False,
    }
    assert status["materialization"] == {
        "concrete_specialization_count": 4,
        "reachable_specialization_count": 7,
        "dependency_discovery_work_count": 8,
        "pruned_candidate_count": 194,
        "selected_specializations": {
            "vjp_layer_norm_single_row": {"N_READS": "8", "T": "float"},
            "initialize_buffer": {"N": "3"},
            "threadgroup_sum_3": {"N": "3"},
            "threadgroup_sum_1": {"N": "1"},
        },
    }
    specialization = status["specialization"]
    assert specialization["name"] == "has_w"
    assert specialization["constant_id"] == 20
    assert specialization["value"] is True
    assert specialization["directx_materialization"] == "concrete-crossgl-variant"
    assert specialization["opengl_materialization"] == ("deferred-layout-constant-id")
    assert specialization["native_header_available"] is False
    assert specialization["native_header_unavailable_reason"] == (
        "specialization-requires-deferred-compilation"
    )
    assert specialization["deferred_compilation_request_status"] == "ready"
    assert specialization["deferred_output_format"] == "SPIR-V binary"

    directx = status["artifacts"]["directx"]
    assert directx["sha256"] == (
        "7d45e40974cc5419bb93c106708460eb1edd5d68ec21a6afba7e9b7f6f05cf7e"
    )
    assert directx["size_bytes"] == 7504
    assert directx["wave_active_sum_call_count"] == 4
    assert directx["native_runtime"]["status"] == "required-on-ci"
    opengl = status["artifacts"]["opengl"]
    assert opengl["sha256"] == (
        "9e6c4e6201e1c78e981a346275b849c37e6c8d834e7509d662f7aec5782980fa"
    )
    assert opengl["size_bytes"] == 8291
    assert opengl["specialization_enforcement"] == (
        "deferred-opengl-spirv-specialization"
    )
    assert opengl["control_barrier_instruction_count"] == 8
    assert opengl["group_non_uniform_instruction_count"] == 0
    assert opengl["local_validation"] == {
        "platform": "linux-arm64",
        "runtime": "mesa-headless-egl-llvmpipe",
        "status": "passed",
        "gx_maximum_absolute_error": 4.8107191008561756e-08,
        "gx_maximum_relative_error": 6.617471155711408e-06,
        "gw_maximum_absolute_error": 5.43903434513382e-08,
        "gw_maximum_relative_error": 1.198717416402971e-07,
    }
    assert opengl["native_runtime"]["status"] == "required-on-ci"
    assert status["runtime_package"]["resource_count"] == 8
    assert status["runtime_package"]["specialization_constant_count"] == 1
    assert [
        resource["role"] for resource in status["runtime_package"]["resources"]
    ] == [
        "input",
        "weight",
        "output_cotangent",
        "input_gradient",
        "per_row_weight_gradient",
        "epsilon",
        "axis_size",
        "weight_stride",
    ]
    assert status["one_row_boundary"] == {
        "per_row_weight_gradient_equals_final_reduced_gradient": True,
        "follow_on_weight_reduction_dispatch_required": False,
        "bias_gradient_reduction_included": False,
    }
    assert status["remaining_scope"] == {
        "multi_row_weight_reduction_included": False,
        "bias_gradient_reduction_included": False,
        "has_w_false_included": False,
        "float16_and_bfloat16_included": False,
        "looped_entries_included": False,
        "other_axis_sizes_included": False,
        "mlx_host_runtime_integration_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects `vjp_layer_normfloat32` at axis size 32" in readme
    assert "deferred OpenGL specialization constant `20`" in readme
    assert "one-row boundary makes the per-row `gw` temporary" in readme
    assert "eight-resource native-loader ABI" in readme
    assert "does not include the separate bias-gradient reduction" in readme


def test_copy_native_runtime_evidence_records_bounded_cross_target_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["copy_native_runtime_status"]
    assert status["status"] == "translated-packaged-and-native-executed"
    assert status["commit"] == "846d176227a0ac13d2667e58d2bb68b322109ab0"
    assert status["source"] == "mlx/backend/metal/kernels/copy.metal"
    assert status["source_sha256"] == (
        "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba"
    )
    assert status["selected_entry_point"] == "v_copyfloat32float32"
    assert status["template_materialization"] == {
        "template": "copy_v",
        "arguments": {"N": "1", "T": "float", "U": "float"},
        "specializations": [
            {
                "template": "copy_v",
                "arguments": {"N": "1", "T": "float", "U": "float"},
                "source": "source-instantiation",
            },
            {
                "template": "cast_to",
                "materialized_name": "cast_to_float_float",
                "arguments": {"T": "float", "U": "float"},
                "source": "call-site",
            },
        ],
        "specialization_count": 2,
        "reachable_specialization_count": 5,
        "dependency_discovery_work_count": 7,
        "pruned_candidate_count": 62424,
    }
    assert status["artifacts"]["directx"] == {
        "target_entry_point": "CSMain",
        "sha256": "023227a86b82cfeff6c32219e2526efbb658d018a4679b47f06c8369186a3495",
        "size_bytes": 1234,
        "workgroup_size": [1, 1, 1],
        "resource_layouts": {
            "input_storage": "hlsl-structured-buffer",
            "output_storage": "hlsl-structured-buffer",
            "element_type": "float32",
            "element_stride_bytes": 4,
            "constant_storage": "hlsl-constant-buffer",
            "constant_block_size_bytes": 16,
        },
        "native_runtime": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_copy_native_loader.py::"
                "test_pinned_mlx_copy_executes_through_directx_native_loader"
            ),
        },
    }
    assert status["artifacts"]["opengl"] == {
        "target_entry_point": "main",
        "sha256": "fac13358ba17271c622634c3e42f9b3cd0863adb75bcfe71aca7ce13aa5628cb",
        "size_bytes": 1619,
        "workgroup_size": [1, 1, 1],
        "resource_layouts": {
            "input_storage": "std430",
            "output_storage": "std430",
            "element_type": "float32",
            "element_stride_bytes": 4,
            "constant_storage": "std140",
            "constant_block_size_bytes": 16,
        },
        "native_runtime": {
            "platform": "ubuntu-latest",
            "runtime": "headless-egl-software",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_copy_native_loader.py::"
                "test_pinned_mlx_copy_executes_through_opengl_native_loader"
            ),
        },
    }
    assert status["workload"] == {
        "dtype": "float32",
        "shape": [8],
        "dispatch_workgroup_count": [8, 1, 1],
        "expected_output": "exact-input-copy",
        "output_element_count": 8,
    }
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    assert status["remaining_aggregate_layout_scope"] == {
        "entry_point": "s_copycomplex64float32",
        "input_type": "complex64_t",
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1543",
    }

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects `v_copyfloat32float32` from the current corpus" in readme
    assert "materializes `copy_v<float, float, 1>`" in readme
    assert "`cast_to<float, float>`" in readme
    assert "pruning 62,424 unreachable candidate pairs" in readme
    assert "requires exact readback on both targets" in readme
    assert "bounded execution evidence for one scalar copy entry" in readme
    assert "does not cover the complex or bfloat16 entries" in readme


def test_dot_native_runtime_evidence_records_bounded_current_corpus_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["dot_native_runtime_status"]
    assert status["status"] == "translated-packaged-and-cross-target-executed"
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == "mlx/backend/metal/kernels/dot.metal"
    assert status["source_sha256"] == (
        "97bcad13d09c3d5fed87482a0bb9719d6eeff9b21d364967cd6aec5b695b3462"
    )
    assert status["selected_entry_point"] == "dot_product_float32_it32_tg512_sg16"
    assert status["template_arguments"] == {
        "T": "float",
        "ITEMS_PER_THREAD": 32,
        "TG_SIZE": 512,
        "SIMD_GROUPS": 16,
    }
    assert status["execution"] == {
        "workgroup_size": [512, 1, 1],
        "subgroup_width": 32,
        "workgroup_reduction_element_count": 16,
    }
    assert status["artifacts"]["directx"] == {
        "target_entry_point": "CSMain",
        "sha256": "f902bae0e7603d302340327c61e5a82ab392b9ce1afffb5557b1760592a1465f",
        "size_bytes": 5110,
        "subgroup_enforcement": "hlsl-wave-size-attribute",
        "subgroup_id_lowering": "workgroup-synchronized-physical-wave-allocation",
        "native_runtime": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_dot_native_loader.py::"
                "test_pinned_mlx_dot_executes_through_directx_native_loader"
            ),
        },
    }
    assert status["artifacts"]["opengl"] == {
        "target_entry_point": "main",
        "sha256": "ef69a757339fe09897a38804c27be279a19a7db146e2e02f85f0349c59f3168d",
        "size_bytes": 5188,
        "subgroup_enforcement": "glsl-subgroup-size-guard",
        "toolchain_validation": {
            "platform": "ubuntu-latest",
            "compiler": "glslangValidator",
            "validator": "spirv-val",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_dot_native_loader.py::"
                "test_pinned_mlx_dot_translates_to_guarded_opengl_artifact"
            ),
        },
        "native_runtime_executed": False,
        "software_runtime_artifact": {
            "target_entry_point": "main",
            "sha256": (
                "a3c1958daa680419ce3f38559de1a6a2319a7abdac556a049632194c88223a32"
            ),
            "size_bytes": 6275,
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "software_subgroup_width": 32,
            "software_subgroup_count": 16,
            "shared_scratch_element_count": 512,
            "hardware_subgroup_extension_required": False,
            "toolchain_validation": {
                "platform": "ubuntu-latest",
                "compiler": "glslangValidator",
                "validator": "spirv-val",
                "spirv_target_environment": "spv1.3",
                "control_barrier_count": 4,
                "group_non_uniform_instruction_count": 0,
                "status": "required-on-ci",
            },
            "native_runtime": {
                "platform": "ubuntu-latest",
                "runtime": "mesa-llvmpipe-opengl",
                "adapter": "opengl-native-runtime",
                "status": "required-on-ci",
                "test": (
                    "tests/test_translator/test_mlx_dot_native_loader.py::"
                    "test_pinned_mlx_dot_executes_with_opengl_software_subgroups"
                ),
            },
        },
    }
    assert status["artifacts"]["metal"] == {
        "target_entry_point": "dot_product_float32_it32_tg512_sg16",
        "status": "blocked-before-emission",
        "diagnostic_code": "project.translate.pointer-reinterpret-unsupported",
        "missing_capability": "pointer.reinterpretation",
        "reason": "target-lowering-unavailable",
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1903",
        "test": (
            "tests/test_translator/test_mlx_dot_native_loader.py::"
            "test_pinned_mlx_dot_records_metal_roundtrip_boundary"
        ),
    }
    assert status["workload"] == {
        "dtype": "float32",
        "input_element_count": 1024,
        "left_value": 1.0,
        "right_value": 0.25,
        "dispatch_workgroup_count": [1, 1, 1],
        "expected_output": 256.0,
        "absolute_tolerance": 0.00001,
        "relative_tolerance": 0.00001,
    }
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["metal_roundtrip_included"] is False
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["cross_target_selected_workload_numerical_parity_verified"] is True
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "selects `dot_product_float32_it32_tg512_sg16`" in readme
    assert "preserve the read-only `float4` storage views" in readme
    assert "requires a readback of `256.0`" in readme
    assert "separate software artifact is 6,275 bytes" in readme
    assert "sixteen independent 32-lane logical subgroups" in readme
    assert "four control barriers and no group-nonuniform instruction" in readme
    assert "Mesa llvmpipe executes the same workload" in readme
    assert "still fails closed at storage-backed vector pointer lowering" in readme
    assert "do not redirect MLX host dispatch" in readme
    assert "or run the MLX test suite" in readme


def test_unary_native_runtime_evidence_records_selected_entry_proofs():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["unary_square_native_runtime_status"]
    assert status["status"] == "translated-packaged-and-executed"
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == "mlx/backend/metal/kernels/unary.metal"
    assert status["source_sha256"] == (
        "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
    )
    assert status["selected_entry_point"] == "v_Squarefloat32float32"
    assert status["specialization"] == {
        "name": "unary_v",
        "arguments": {
            "T": "float",
            "U": "float",
            "Op": "Square",
            "N": 1,
        },
        "specialization_count": 1,
        "unsupported_specialization_count": 0,
        "pruned_candidate_count": 39509,
    }
    assert status["entry_reachability"] == {
        "unreachable_member_lowering_pruned": True,
        "reachable_unresolved_calls_fail_closed": True,
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1922",
    }
    assert status["execution"] == {
        "workgroup_size": [1, 1, 1],
        "dispatch_workgroup_count": [5, 1, 1],
    }
    assert status["artifacts"]["directx"] == {
        "target_entry_point": "CSMain",
        "sha256": "64540a89c95e39914a4d616aff9bec98b939a5209fa4caef5cc1425511abb4e5",
        "size_bytes": 2314,
        "native_runtime": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "compiler": "dxc",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_unary_native_loader.py::"
                "test_pinned_mlx_unary_square_executes_through_directx_native_loader"
            ),
        },
    }
    assert status["artifacts"]["metal"] == {
        "target_entry_point": "v_Squarefloat32float32",
        "sha256": "244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495",
        "size_bytes": 1015,
        "host_dispatch_workgroup_size": [1, 1, 1],
        "native_roundtrip": {
            "platform": "macos-latest",
            "compiler": "xcrun -sdk macosx metal",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_unary_native_loader.py::"
                "test_pinned_mlx_unary_square_roundtrips_through_metal"
            ),
        },
        "numerical_runtime_included": False,
    }
    assert status["artifacts"]["opengl"] == {
        "target_entry_point": "main",
        "sha256": "2bb46a3bb0858eb849e533bfe46eff1d59b9192436e15b2639c7998698db6a48",
        "size_bytes": 3613,
        "native_runtime": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-opengl-4.3",
            "compiler": "glslangValidator",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_unary_native_loader.py::"
                "test_pinned_mlx_unary_square_executes_through_opengl_native_loader"
            ),
        },
    }
    assert status["workload"] == {
        "dtype": "float32",
        "input": [-3.0, -1.5, 0.0, 2.0, 4.25],
        "expected_output": [9.0, 2.25, 0.0, 4.0, 18.0625],
        "absolute_tolerance": 0.000001,
        "relative_tolerance": 0.000001,
    }
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["metal_roundtrip_included"] is True
    assert status["metal_numerical_runtime_included"] is False
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["other_unary_specializations_included"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    arccos = gaps["unary_arccos_native_runtime_status"]
    assert arccos["status"] == "translated-packaged-and-executed"
    assert arccos["commit"] == CURRENT_MLX_COMMIT
    assert arccos["source"] == "mlx/backend/metal/kernels/unary.metal"
    assert arccos["source_sha256"] == (
        "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b"
    )
    assert arccos["selected_entry_point"] == "v_ArcCosfloat32float32"
    assert arccos["specialization"] == {
        "name": "unary_v",
        "arguments": {
            "T": "float",
            "U": "float",
            "Op": "ArcCos",
            "N": 1,
        },
        "specialization_count": 1,
        "unsupported_specialization_count": 0,
        "pruned_candidate_count": 39509,
    }
    assert arccos["entry_reachability"] == {
        "unselected_out_of_line_overloads_pruned": True,
        "reachable_unresolved_calls_fail_closed": True,
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1924",
    }
    assert arccos["precise_math_lowering"] == {
        "source_operation": "metal::precise::acos",
        "strategy": "portable-float32-range-reduction",
        "no_contraction": True,
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1925",
    }
    assert arccos["execution"] == {
        "workgroup_size": [1, 1, 1],
        "dispatch_workgroup_count": [5, 1, 1],
    }
    assert arccos["artifacts"]["directx"] == {
        "target_entry_point": "CSMain",
        "sha256": "4562332ad4fb951478ca419180ccdf3589f74b9e0956226badc6de877d343239",
        "size_bytes": 4175,
        "native_runtime": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "compiler": "dxc",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_unary_native_loader.py::"
                "test_pinned_mlx_unary_arccos_executes_through_directx_native_loader"
            ),
        },
    }
    assert arccos["artifacts"]["opengl"] == {
        "target_entry_point": "main",
        "sha256": "280864c39e88198cd5e660127db453877349fadb090cb37f022bcc46300660b3",
        "size_bytes": 5965,
        "native_runtime": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-opengl-4.3",
            "compiler": "glslangValidator",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_unary_native_loader.py::"
                "test_pinned_mlx_unary_arccos_executes_through_opengl_native_loader"
            ),
        },
    }
    assert arccos["workload"] == {
        "dtype": "float32",
        "input": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "expected_output": [
            3.141592653589793,
            2.0943951023931957,
            1.5707963267948966,
            1.0471975511965979,
            0.0,
        ],
        "absolute_tolerance": 0.000001,
        "relative_tolerance": 0.00001,
    }
    assert arccos["mlx_host_runtime_included"] is False
    assert arccos["runtime_integration_included"] is True
    assert arccos["selected_workload_numerical_parity_verified"] is True
    assert arccos["other_unary_specializations_included"] is False
    assert arccos["full_mlx_test_suite_included"] is False
    assert arccos["numerical_parity_claimed"] is False
    assert arccos["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "select `v_Squarefloat32float32` and `v_ArcCosfloat32float32`" in readme
    assert "Each entry-scoped artifact emits exactly one" in readme
    assert "keeps the out-of-line complex `ArcCos::operator()` body out" in readme
    assert "Square retains `x * x`" in readme
    assert "portable float32 range-reduction implementation" in readme
    assert "collision-safe `precise` local" in readme
    assert "preserving SPIR-V `NoContraction`" in readme
    assert "target intrinsic's unspecified accuracy" in readme
    assert "executes them through the native loader on Direct3D 12 WARP" in readme
    assert "surfaceless Mesa OpenGL context" in readme
    assert "numerical evidence for two selected unary specializations" in readme
    assert "does not cover the other unary operations or dtypes" in readme
    assert "run the MLX test suite" in readme


def test_fft_directx_evidence_records_selected_native_runtime_proof():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["directx_fft_translation_status"]
    assert status["status"] == "translated-dxc-validated-direct3d-executed"
    assert status["source"] == module.MLX_FFT_SOURCE
    assert status["source_sha256"] == module.MLX_FFT_SHA256
    assert status["source_size_bytes"] == module.MLX_FFT_SOURCE_SIZE_BYTES
    assert status["target"] == "directx"
    assert status["selected_entry_point"] == module.FFT_DIRECTX_ENTRY_POINT
    assert status["target_entry_point"] == "CSMain"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "index_range_assertion_count": 4,
        "workgroup_access_assertion_count": 1,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 24,
        "unsupported_specialization_count": 0,
        "configured_function_constant_count": 22,
        "reachable_function_constant_count": 21,
        "pruned_function_constant_ids": [3],
    }
    assert status["execution"] == {
        "workgroup_size": list(module.FFT_DIRECTX_WORKGROUP_SIZE),
        "host_dispatch_source": "mlx/backend/metal/fft.cpp",
        "host_dispatch_shape": "MTL::Size(1, threadgroup_batch_size, threads_per_fft)",
        "numthreads_declaration": "[numthreads(1, 1, 64)]",
        "workgroup_grid_expression": "crossglNumWorkGroups * uint3(1, 1, 64)",
    }
    assert status["portability_preconditions"] == {
        "index_range_assertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
        "workgroup_access_assertions": list(
            module.FFT_DIRECTX_WORKGROUP_ACCESS_ASSERTIONS
        ),
    }
    assert status["artifact"] == {
        "sha256": module.FFT_DIRECTX_GENERATED_SHA256,
        "size_bytes": module.FFT_DIRECTX_GENERATED_SIZE_BYTES,
        "promoted_native_16_shift_count": 20,
        "first_class_workgroup_pointer_residue": False,
    }
    assert status["native_validation"] == {
        "compiler": "dxc",
        "profile": "cs_6_2",
        "arguments": ["-enable-16bit-types"],
        "warnings_as_errors": True,
        "status": "passed",
    }
    assert status["native_runtime"] == {
        "runtime": "direct3d-12-warp",
        "status": "passed",
        "test": (
            "tests/test_translator/test_mlx_fft_native_loader.py::"
            "test_pinned_mlx_fft_executes_through_directx_native_loader"
        ),
        "workgroup_count": [1, 1, 1],
        "global_size": [1, 1, 64],
        "input": {
            "kind": "complex-unit-impulse",
            "index": 1,
            "shape": [256, 2],
        },
        "expected_output": {
            "kind": "forward-dft-unit-circle",
            "shape": [256, 2],
            "absolute_tolerance": 0.0002,
            "relative_tolerance": 0.0002,
        },
        "resource_layouts": {
            "input_element": "float2",
            "input_stride_bytes": 8,
            "output_element": "float2",
            "output_stride_bytes": 8,
            "dispatch_element": "uint3",
            "dispatch_block_size_bytes": 16,
        },
        "dispatch_binding_source": "dispatch.workgroupCount",
    }
    assert status["aggregate_source_translation"] == {
        "included": False,
        "tracked_by": "https://github.com/CrossGL/crosstl/issues/1916",
    }
    assert status["tracked_issues"] == []
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "forward complex 256-point entry" in readme
    assert "`MTL::Size(1, threadgroup_batch_size, threads_per_fft)`" in readme
    assert "24 reachable template specializations" in readme
    assert "21 of the 22 configured function constants" in readme
    assert "contains no first-class workgroup pointer residue" in readme
    assert "Direct3D 12 WARP" in readme
    assert "index-1 complex unit impulse" in readme
    assert "`2e-4` absolute and relative tolerance" in readme
    assert "does not redirect the MLX host runtime" in readme


def test_fft_current_corpus_evidence_records_native_runtime_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["directx_fft_current_corpus_status"]
    assert status["status"] == "translated-dxc-validated-direct3d-executed"
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == "mlx/backend/metal/kernels/fft.metal"
    assert status["source_sha256"] == (
        "c478eb84283bbdf585c0cb34b2bfde5b0fc32d1740c6ad76e8559698a57b8d2e"
    )
    assert status["source_size_bytes"] == 3436
    assert status["target"] == "directx"
    assert status["selected_entry_point"] == "fft_mem_256_float2_float2"
    assert status["target_entry_point"] == "CSMain"
    assert status["workgroup_size"] == [1, 1, 64]
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "artifact_emitted": True,
        "index_range_assertion_count": 4,
        "workgroup_access_assertion_count": 1,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
    }
    assert status["specialization"] == {
        "configured_function_constant_count": 23,
        "reachable_function_constant_count": 21,
        "pruned_function_constant_ids": [3, 22],
        "new_function_constant": {
            "id": 22,
            "name": "use_bluestein_twiddle_table_",
            "value": False,
        },
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 37,
        "unsupported_specialization_count": 0,
        "reachable_specialization_count": 42,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 2120,
    }
    assert status["artifact"] == {
        "sha256": "dd64cfa562f4463f3ecb237d00c0273560e62839565e8638c2343a922149c6ab",
        "size_bytes": 146763,
        "promoted_native_16_shift_count": 20,
        "first_class_workgroup_pointer_residue": False,
        "workgroup_pointer_transport": "concrete-groupshared-root-plus-integer-offset",
        "overload_body_reachability": "all-compatible-source-overloads",
        "default_null_resource_pointer_transport": "statically-unobserved-chain-pruned",
        "observable_null_resource_pointer_policy": "fail-closed",
    }
    assert status["native_validation"] == {
        "compiler": "dxc",
        "profile": "cs_6_2",
        "arguments": ["-enable-16bit-types"],
        "warnings_as_errors": True,
        "status": "required-on-ci",
    }
    assert status["native_runtime"] == {
        "runtime": "direct3d-12-warp",
        "status": "required-on-ci",
        "test": (
            "tests/test_translator/test_mlx_fft_native_loader.py::"
            "test_current_mlx_fft_executes_through_directx_native_loader"
        ),
        "workgroup_count": [1, 1, 1],
        "global_size": [1, 1, 64],
        "input": {
            "kind": "complex-unit-impulse",
            "index": 1,
            "shape": [256, 2],
        },
        "expected_output": {
            "kind": "forward-dft-unit-circle",
            "shape": [256, 2],
            "absolute_tolerance": 0.0002,
            "relative_tolerance": 0.0002,
        },
        "resource_layouts": {
            "input_element": "float2",
            "input_stride_bytes": 8,
            "output_element": "float2",
            "output_stride_bytes": 8,
            "dispatch_element": "uint3",
            "dispatch_block_size_bytes": 16,
        },
        "dispatch_binding_source": "dispatch.workgroupCount",
    }
    assert status["resolved_blocker"] == {
        "former_diagnostic_code": (
            "project.translate.directx-workgroup-pointer-unsupported"
        ),
        "function": "ReadWriter_float2_float2__load",
        "parameter": "crosstl_ptr_buf",
        "cause": "same-name-overload-body-was-not-analyzed",
        "resolution": "merge-compatible-overload-pointer-analysis",
        "diagnostic_emitted": False,
        "broader_contract_tracked_by": "https://github.com/CrossGL/crosstl/issues/1518",
    }
    assert status["previous_verified_proof"] == {
        "commit": PINNED_MLX_COMMIT,
        "status_key": "directx_fft_translation_status",
    }
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "bounded runtime proof now also covers current corpus commit" in readme
    assert "adds function constant 22" in readme
    assert "materializes 37 specializations" in readme
    assert "records 42 reachable specializations before pruning" in readme
    assert "two compatible `ReadWriter_float2_float2__load` overload bodies" in readme
    assert "transport `crosstl_ptr_buf` as an integer offset" in readme
    assert "carried into the CrossGL intermediate before target lowering" in readme
    assert "removes that unobserved resource parameter" in readme
    assert (
        "null pointer that can be observed or dereferenced still fails closed" in readme
    )
    assert "146,763-byte HLSL artifact" in readme
    assert "All 20 native-16 ``power`` shift counts" in readme
    assert "explicitly promoted to ``int``" in readme
    assert "zero project diagnostics" in readme
    assert "packages and dispatches it through Direct3D 12 WARP" in readme
    assert (
        "roots without a concrete shared-array identity or extent still fail closed"
        in readme
    )
    assert "no longer being used as a substitute for current-corpus evidence" in readme


def test_fft_current_opengl_evidence_records_native_runtime_proof():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["opengl_fft_current_corpus_status"]
    assert status["status"] == "translated-glslang-spirv-val-mesa-executed"
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == "mlx/backend/metal/kernels/fft.metal"
    assert status["source_sha256"] == (
        "c478eb84283bbdf585c0cb34b2bfde5b0fc32d1740c6ad76e8559698a57b8d2e"
    )
    assert status["source_size_bytes"] == 3436
    assert status["target"] == "opengl"
    assert status["selected_entry_point"] == "fft_mem_256_float2_float2"
    assert status["target_entry_point"] == "main"
    assert status["workgroup_size"] == [1, 1, 64]
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "index_range_assertion_count": 5,
        "workgroup_access_assertion_count": 1,
        "source_remap_mapping_count": 84,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
    }
    assert status["specialization"]["deferred_function_constant_count"] == 21
    assert status["specialization"]["pruned_function_constant_ids"] == [3, 22]
    assert status["specialization"]["mode"] == "deferred"
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 37,
        "unsupported_specialization_count": 0,
        "reachable_specialization_count": 42,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 2120,
    }
    assert status["artifact"] == {
        "sha256": "a1ab0c346d9143e6749e391fb971aeaed71bd84e15fedaf7a7e92808a56449bb",
        "size_bytes": 82045,
        "source_remap_mapping_count": 84,
        "pointer_transport": "concrete-workgroup-and-storage-resource-offsets",
        "default_null_resource_pointer_transport": "statically-unobserved-chain-pruned",
        "encoded_generic_vector_constructor_residue": False,
        "all_overload_specialization_declarations_defined": True,
    }
    assert status["native_validation"]["compiler"] == "glslangValidator"
    assert status["native_validation"]["validator"] == "spirv-val"
    assert status["native_validation"]["status"] == "passed"
    assert status["native_validation"]["control_barrier_count"] == 19
    assert status["native_validation"]["group_non_uniform_instruction_count"] == 0
    assert status["runtime_package"] == {
        "resource_binding_count": 4,
        "specialization_constant_count": 21,
        "blocked_variant_count": 0,
        "registry_status": "ready",
        "input_layout": {
            "physical_type": "float2",
            "element_type": "float32",
            "element_size_bytes": 8,
            "element_stride_bytes": 8,
            "alignment_bytes": 8,
            "storage_layout": "std430",
        },
        "output_layout": {
            "physical_type": "float2",
            "element_type": "float32",
            "element_size_bytes": 8,
            "element_stride_bytes": 8,
            "alignment_bytes": 8,
            "storage_layout": "std430",
        },
    }
    runtime = status["native_runtime"]
    assert runtime["platform"] == "ubuntu-latest"
    assert runtime["runtime"] == "mesa-llvmpipe-opengl"
    assert runtime["specialization_mode"] == "deferred-spirv"
    assert runtime["status"] == "required-on-ci"
    assert runtime["test"].endswith(
        "::test_current_mlx_fft_executes_through_opengl_native_loader"
    )
    assert runtime["observed_max_absolute_error"] < 1e-7
    assert runtime["interface_status"] == "verified"
    assert runtime["cache_status"] == "published"
    assert status["resolved_blockers"] == [
        "transitive-statically-dead-null-storage-pointer-specialization",
        "encoded-metal-generic-vector-constructor",
        "overload-specialization-body-name-deduplication",
        "std430-vector-resource-layout-reflection",
    ]
    assert status["mlx_host_runtime_included"] is False
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["cross_target_selected_workload_numerical_parity_verified"] is True
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "current FFT source now also emits an 82,045-byte GLSL artifact" in readme
    assert "21 deferred specialization constants" in readme
    assert "8-byte size, stride, and alignment" in readme
    assert "19 control barriers and no group-nonuniform instructions" in readme
    assert "Mesa llvmpipe" in readme
    assert "`9.264554161336758e-08`" in readme
    assert "does not establish full FFT, MLX host-runtime, or backend parity" in readme


def test_fft_opengl_evidence_records_toolchain_proof_without_runtime_claims():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["opengl_fft_translation_status"]
    assert status["status"] == "translated-glslang-spirv-val-validated"
    assert status["source"] == module.MLX_FFT_SOURCE
    assert status["source_sha256"] == module.MLX_FFT_SHA256
    assert status["source_size_bytes"] == module.MLX_FFT_SOURCE_SIZE_BYTES
    assert status["target"] == "opengl"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 1,
        "translated_count": 1,
        "failed_count": 0,
        "project_diagnostic_count": 0,
        "index_range_assertion_count": 4,
        "workgroup_access_assertion_count": 5,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 99,
        "unsupported_specialization_count": 0,
        "function_constant_count": 22,
    }
    assert status["portability_preconditions"] == {
        "index_range_assertions": list(module.FFT_INDEX_RANGE_ASSERTIONS),
        "workgroup_access_assertions": list(
            module.FFT_OPENGL_WORKGROUP_ACCESS_ASSERTIONS
        ),
    }
    assert status["artifact_emitted"] is True
    assert status["native_validation"] == {
        "compiler": "glslangValidator",
        "compiler_target": "OpenGL SPIR-V 1.3",
        "validator": "spirv-val",
        "validator_target": "SPIR-V 1.3",
        "status": "passed",
    }
    assert status["tracked_issues"] == []
    assert status["runtime_integration_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "the complete pinned `fft.metal` source to one standalone OpenGL" in readme
    assert "99 unique reachable template specializations" in readme
    assert "five entry-point-scoped workgroup access assertions" in readme
    assert "validates the binary with `spirv-val`" in readme
    assert "OpenGL runtime dispatch and numerical parity" in readme


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
        "specialization_count": 226,
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
    assert "does not establish wave semantics" in readme


def test_gemv_opengl_evidence_records_all_entry_toolchain_validation():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["opengl_gemv_toolchain_status"]
    assert status["status"] == "all-entry-points-validated"
    assert status["source"] == module.MLX_GEMV_SOURCE
    assert status["source_sha256"] == module.MLX_GEMV_SHA256
    assert status["source_size_bytes"] == module.MLX_GEMV_SOURCE_SIZE_BYTES
    assert status["target"] == "opengl"
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count": 224,
        "translated_count": 224,
        "failed_count": 0,
        "emitted_target_file_count": 224,
        "max_template_specializations": 4096,
        "max_template_materialization_work": 2097152,
        "workgroup_size_rule": [32, "BN", "BM"],
        "report_workgroup_size_rule": ["32", "BN", "BM"],
        "subgroup_width_rule": 32,
        "index_range_assertion_count": len(module.GEMV_OPENGL_INDEX_RANGE_ASSERTIONS),
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 226,
        "unsupported_specialization_count": 0,
        "artifact_packaging": "entry-scoped-artifacts",
    }
    assert status["compiler"] == {
        "name": "glslangValidator",
        "target_environment": "OpenGL SPIR-V 1.3",
        "compiled_artifact_count": 224,
        "validator": "spirv-val",
        "validator_target_environment": "spv1.3",
        "validated_artifact_count": 224,
    }
    execution = status["execution_contracts"]
    assert execution["workgroup_size_specialization"]["status"] == "established"
    assert execution["workgroup_size_specialization"]["resolved_sizes"] == [
        list(size) for size in module.GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES
    ]
    subgroup = execution["subgroup_width_specialization"]
    assert subgroup["status"] == "guarded"
    assert subgroup["required_width"] == module.GEMV_SUBGROUP_WIDTH
    assert subgroup["enforcement"] == module.GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT
    assert (
        subgroup["fallback_tracked_by"]
        == module.GEMV_OPENGL_SUBGROUP_WIDTH_FALLBACK_ISSUE
    )
    index_ranges = execution["index_range_assertions"]
    assert index_ranges["count"] == len(module.GEMV_OPENGL_INDEX_RANGE_ASSERTIONS)
    assert index_ranges["contract_kind"] == (
        "explicit-host-runtime-portability-preconditions"
    )
    assert index_ranges["inferred"] is False
    assert index_ranges["runtime_enforced"] is False
    assert status["translation_blocked_by"] == []
    assert status["native_validation_attempted"] is True
    assert status["native_validation_status"] == "validated"
    assert status["tracked_issues"] == list(
        module.GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES
    )
    assert status["runtime_execution_attempted"] is False
    assert status["runtime_integration_included"] is False
    assert status["runnable_artifact_claimed"] is False
    assert status["compiler_validated_artifact_claimed"] is True
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    assert module.GEMV_OPENGL_SUBGROUP_WIDTH_FALLBACK_ISSUE in gaps["tracked_issues"]

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "--require-opengl-gemv-toolchain" in readme
    assert "all 226 specializations materialized" in readme
    assert "all 224 entry-scoped GLSL artifacts" in readme
    assert "validates every resulting SPIR-V 1.3 module" in readme
    assert "five explicit host index-range preconditions" in readme
    assert "does not establish runtime execution or numerical parity" in readme


def test_gemv_current_native_runtime_evidence_is_exact_and_bounded():
    module = _load_harness()
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["gemv_native_runtime_status"]
    assert status["status"] == (
        "translated-packaged-and-cross-target-native-runtime-required"
    )
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == module.MLX_GEMV_SOURCE
    assert status["source_sha256"] == (
        "0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda"
    )
    assert status["source_size_bytes"] == 6981
    assert status["selected_entry_point"] == (
        "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
    )
    assert status["selected_workload"] == (
        "vector-matrix-float32-m1-n32-k32-contiguous"
    )

    dispatch = status["dispatch_contract"]
    assert dispatch["path"] == (
        "demos/integrations/mlx/contracts/gemv.native-loader.dispatch.json"
    )
    assert dispatch["content_identity"] == (
        "sha256:6b3bb18d130159f13874f06668b536fe4b9270ffbb2a1f44b6d9aac257aba7e4"
    )
    assert dispatch["workload_count"] == 1
    assert dispatch["host_source"] == "mlx/backend/metal/matmul.cpp"
    assert dispatch["implementation_source"] == ("mlx/backend/metal/kernels/gemv.h")
    assert dispatch["upstream_test"] == (
        "python/tests/test_blas.py::TestBlas::test_matrix_vector"
    )
    assert dispatch["host_selection"] == {
        "BM": 1,
        "BN": 2,
        "SM": 8,
        "SN": 4,
        "TM": 4,
        "TN": 4,
        "kDoNCBatch": False,
        "kDoAxpby": False,
    }
    assert dispatch["subgroup_width"] == 32
    variant = dispatch["variants"][status["selected_workload"]]
    assert variant == {
        "artifact_id": (
            "sha256:34eab189b10cc699f06f4cbed04faae41a2658a2a3665a6866ed987f5946949a"
        ),
        "dispatch_variant_id": (
            "sha256:acaba2ec4813a364b06d95a5136bda80351797591a4d9f0b3d195f85da287fe3"
        ),
        "entry_point": status["selected_entry_point"],
        "inputs": {
            "K": 32,
            "M": 1,
            "N": 32,
            "batchSize": 1,
            "contiguousBatch": True,
            "doAxpby": False,
            "dtype": "float32",
            "transposeA": False,
            "transposeB": False,
        },
        "workgroup_size": [32, 2, 1],
        "dispatch_workgroup_count": [1, 1, 1],
        "specialization_constants": {},
    }
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 1, "opengl": 1},
        "translated_count_by_target": {"directx": 1, "opengl": 1},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
        "opengl_index_range_assertion_count": 1,
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 2,
        "unsupported_specialization_count": 0,
        "materialized_names": [
            status["selected_entry_point"],
            "elem_to_loc_uint",
        ],
        "selected_parameters": {
            "BM": "1",
            "BN": "2",
            "SM": "8",
            "SN": "4",
            "T": "float",
            "TM": "4",
            "TN": "4",
            "kDoAxpby": "0",
            "kDoNCBatch": "0",
        },
    }

    assert status["artifacts"] == {
        "directx": {
            "target_entry_point": "CSMain",
            "sha256": (
                "9972997d87bb4c8c5fac0c0f7182bb19648654ca2c30eacd4b304bfaf18f64d2"
            ),
            "size_bytes": 8188,
            "subgroup_id_lowering": "flattened-logical-software-subgroups",
            "relative_shuffle_out_of_range": "calling-invocation-value",
            "software_subgroup_width": 32,
            "logical_subgroup_count": 2,
            "shared_float_element_count": 64,
            "hardware_wave_read_count": 0,
            "workgroup_size": [32, 2, 1],
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "compiler": "dxc",
            "compiler_version": "1.9.2602.24",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types", "-WX"],
            "compiler_validation_status": "passed",
        },
        "opengl": {
            "target_entry_point": "main",
            "sha256": (
                "f5ef8900ee65d63a6df2818ef111f56b4f269c6366c82d82a9d97c967042f562"
            ),
            "size_bytes": 7705,
            "workgroup_size": [32, 2, 1],
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "compiler": "glslangValidator",
            "compiler_target": "OpenGL/SPIR-V 1.3",
            "validator": "spirv-val",
            "control_barrier_instruction_count": 3,
            "group_non_uniform_instruction_count": 0,
            "compiler_validation_status": "passed",
        },
    }
    assert status["directx_software_subgroup"] == {
        "configuration": (
            "project.source_options.metal.target_options.directx."
            "software_subgroup_width"
        ),
        "width": 32,
        "logical_subgroup_count": 2,
        "shared_float_element_count": 64,
        "logical_invocation_index": "flattened-SV_GroupIndex",
        "selected_kernel_operations": ["WaveShuffleDown(float,uint)"],
        "barrier": "GroupMemoryBarrierWithGroupSync",
        "out_of_range_value": "calling-invocation-value",
        "hardware_wave_reads_emitted": False,
        "unsupported_contract_behavior": "reject-before-artifact-emission",
    }
    assert status["directx_warp_diagnostic"] == {
        "rejected_artifact_sha256": (
            "f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d"
        ),
        "rejected_artifact_size_bytes": 8410,
        "workflow_run": 33268998061,
        "job": 99143984804,
        "mismatched_output_count": 32,
        "output_element_count": 32,
        "max_absolute_error": 1.90625,
        "max_relative_error": 0.1382488479262673,
        "first_obtained": 6.78125,
        "first_expected": 5.84375,
        "reduction_signature": "logical-lanes-5-through-8-replaced-by-21-through-24",
    }
    assert status["software_subgroup"] == {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "logical_subgroup_count": 2,
        "shared_float_element_count": 64,
        "activation": "explicit-target-scoped",
        "selected_kernel_operations": ["WaveShuffleDown(float,uint)"],
        "uniform_halving_loop_contract": {
            "source_condition": "sm >= 1",
            "source_update": "sm >>= 1",
            "canonical_positive_bounds": ["value > 0", "value >= 1"],
            "helper_call_form": "direct-top-level-statement",
            "unsafe_loop_behavior": "reject-before-artifact-emission",
        },
        "hardware_subgroup_extensions_emitted": False,
        "group_non_uniform_instruction_count": 0,
    }
    assert status["index_range_assertions"] == [
        {
            "source": module.MLX_GEMV_SOURCE,
            "expression": "uint64(bm + tm) * marix_ld + out_col + tn",
            "minimum": 0,
            "maximum": (1 << 32) - 1,
        }
    ]
    assert status["runtime_package"] == {
        "artifact_count_per_target": 1,
        "ready_load_unit_count_per_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count_by_target": {"directx": 15, "opengl": 15},
        "resource_element_types": ["float32", "int32", "int64"],
        "scalar_argument_count": 7,
        "disabled_path_placeholders": [
            "bias",
            "vector_batch_stride",
            "matrix_batch_stride",
            "bias_batch_stride",
        ],
    }
    assert status["workload"] == {
        "matrix_shape": [32, 32],
        "vector_shape": [32],
        "output_shape": [32],
        "matrix_values": "(row + 1) / 64 + (column + 1) / 64",
        "vector_values": "(row + 1) / 32",
        "expected_values": "5.5859375 + 0.2578125 * (column + 1)",
        "reference": "float32-vector-matrix-product",
        "absolute_tolerance": 1e-5,
        "relative_tolerance": 1e-5,
    }
    assert status["native_runtime"] == {
        "directx": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_gemv_native_loader.py::"
                "test_current_mlx_gemv_executes_through_directx_native_loader"
            ),
        },
        "opengl": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-headless-egl-software-opengl",
            "status": "required-on-ci",
            "local_linux_arm64_validation": "passed",
            "test": (
                "tests/test_translator/test_mlx_gemv_native_loader.py::"
                "test_current_mlx_gemv_executes_with_opengl_software_subgroups"
            ),
        },
    }
    assert status["metal_roundtrip_boundary"] == {
        "status": "outside-selected-native-loader-proof",
        "aggregate_native_baseline": "separate",
        "selected_entry_compiler_validation_included": False,
    }
    assert status["previous_verified_proofs"] == {
        "directx_aggregate_status_key": "directx_gemv_compiler_frontier_status",
        "opengl_aggregate_status_key": "opengl_gemv_toolchain_status",
        "aggregate_commit": PINNED_MLX_COMMIT,
    }
    assert status["remaining_scope"] == {
        "gather_entries_included": False,
        "wide_entries_included": False,
        "batched_entries_included": False,
        "axpby_entries_included": False,
        "remaining_host_named_entries_included": False,
        "mlx_host_runtime_integration_included": False,
        "full_mlx_test_suite_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "contracts/gemv.native-loader.dispatch.json" in readme
    assert status["selected_entry_point"] in readme
    assert status["artifacts"]["directx"]["sha256"] in readme
    assert status["artifacts"]["opengl"]["sha256"] in readme
    assert "canonical positive-to-zero halving loop" in readme
    assert "does not replace the historical 224-entry aggregate gates" in readme


def test_mxfp4_current_native_runtime_evidence_is_exact_and_bounded():
    gaps = json.loads(
        (ROOT / "demos" / "integrations" / "mlx" / "expected-gaps.json").read_text(
            encoding="utf-8"
        )
    )

    status = gaps["fp_quantized_mxfp4_native_runtime_status"]
    assert status["status"] == (
        "translated-packaged-and-cross-target-native-runtime-required"
    )
    assert status["commit"] == CURRENT_MLX_COMMIT
    assert status["source"] == "mlx/backend/metal/kernels/fp_quantized.metal"
    assert status["source_sha256"] == (
        "ef4ba099710a63a0b5d27d3e5ce69a8528bee8f1757805aa606c8d8e43de18d4"
    )
    assert status["source_size_bytes"] == 9700
    assert status["selected_entry_point"] == (
        "mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false"
    )
    assert status["selected_workload"] == (
        "mxfp4-quantize-dequantize-float32-32-no-global-scale"
    )

    dispatch = status["dispatch_contract"]
    assert dispatch["path"] == (
        "demos/integrations/mlx/contracts/" "fp_quantized.native-loader.dispatch.json"
    )
    assert dispatch["content_identity"] == (
        "sha256:5256e32b364ac303a6873f28b5ac3e9a1a811ac5c38bc41a977bce9191a025ed"
    )
    assert dispatch["workload_count"] == 1
    assert dispatch["host_source"] == "mlx/backend/metal/quantized.cpp"
    assert dispatch["implementation_source"] == (
        "mlx/backend/metal/kernels/fp_quantized.h"
    )
    assert dispatch["kernel_source"] == status["source"]
    assert dispatch["subgroup_width"] == 32
    variant = dispatch["variants"][status["selected_workload"]]
    assert variant == {
        "artifact_id": (
            "sha256:bde1bfa31c116a52a1dc3b6e546dfa2ee43dc968719393ba04f629b4e2d95319"
        ),
        "dispatch_variant_id": (
            "sha256:ebd6ab3f40f5839764f592943180ba64f11a66f10a5561b9468019032c04df8a"
        ),
        "entry_point": status["selected_entry_point"],
        "inputs": {
            "bits": 4,
            "dtype": "float32",
            "elementCount": 32,
            "groupSize": 32,
            "hasGlobalScale": False,
            "isMxfp4": True,
            "rowContiguous": True,
        },
        "workgroup_size": [32, 1, 1],
        "dispatch_workgroup_count": [1, 1, 1],
        "specialization_constants": {},
    }
    assert status["project_translation"] == {
        "unit_count": 1,
        "artifact_count_by_target": {"directx": 1, "opengl": 1},
        "translated_count_by_target": {"directx": 1, "opengl": 1},
        "failed_count_by_target": {"directx": 0, "opengl": 0},
        "project_diagnostic_count": 0,
        "opengl_index_range_assertion_count": 1,
    }
    assert status["materialization"] == {
        "status": "materialized",
        "specialization_count": 1,
        "unsupported_specialization_count": 0,
        "reachable_specialization_count": 9,
        "dependency_discovery_work_count": 0,
        "pruned_candidate_count": 25492,
        "selected_parameters": {
            "T": "float",
            "bits": "4",
            "group_size": "32",
            "has_global_scale": "false",
        },
    }
    assert status["artifacts"] == {
        "directx": {
            "target_entry_point": "CSMain",
            "sha256": (
                "3fe38e171ba8c8ea1adfc8efad20b242ca02dd05e1a5a53a9b9d1e18459d8c7d"
            ),
            "size_bytes": 9123,
            "workgroup_size": [32, 1, 1],
            "subgroup_enforcement": "hlsl-wave-size-attribute",
            "compiler": "dxc",
            "compiler_profile": "cs_6_6",
            "compiler_arguments": ["-enable-16bit-types", "-WX"],
            "compiled_artifact_size_bytes": 4716,
            "compiler_validation_status": "passed",
        },
        "opengl": {
            "target_entry_point": "main",
            "sha256": (
                "cbbe989c40317c04ffe915f1f314f55db8896edfd38f04ad4b8882be53b2a4da"
            ),
            "size_bytes": 9571,
            "workgroup_size": [32, 1, 1],
            "subgroup_enforcement": "explicit-32-lane-software-subgroup",
            "compiler": "glslangValidator",
            "compiler_target": "OpenGL/SPIR-V 1.3",
            "validator": "spirv-val",
            "compiled_artifact_size_bytes": 10488,
            "control_barrier_instruction_count": 3,
            "group_non_uniform_instruction_count": 0,
            "compiler_validation_status": "passed",
        },
    }
    assert status["semantic_contracts"] == {
        "scale_constructor": "fp8_e8m0 source constructor factory",
        "scale_conversion": "selected sibling float conversion operator",
        "finite_test": "single-evaluation IEEE-754 float32 exponent mask",
        "sign_test": "exact IEEE-754 float32 sign bit",
        "private_scalar_struct_view": "read-only exact one-member layout",
        "directx_binary16_bitcast": "target-scoped-integer-ieee754-binary16-to-float32",
        "directx_binary16_mixed_arithmetic": (
            "source-scoped-native-float16-widening-with-float32-arithmetic-"
            "and-no-half-roundtrip"
        ),
        "directx_native_float16_widening": (
            "project.source_options.metal.target_options.directx."
            "widen_native_float16"
        ),
        "directx_native_16_bit_integer_promotion": (
            "scalar-int16-and-uint16-promote-to-int-before-shifts-and-arithmetic"
        ),
        "opengl_binary16_bitcast": "exact-low-16-bit-payload-via-pack-unpack-half",
        "global_scale_read": "statically-eliminated-for-has_global_scale-false",
    }
    assert status["directx_binary16_diagnostics"] == {
        "numeric_cast_rejection": {
            "artifact_sha256": (
                "3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361"
            ),
            "artifact_size_bytes": 7809,
            "workflow_run": 33271117475,
            "job": 99149649480,
            "output_element_count": 32,
            "mismatched_nonzero_output_count": 28,
            "observed_nonzero_outputs": "signed-zero-only",
            "rejected_dxil_conversion": "uitofp-i16-to-half",
        },
        "native_half_subnormal_rejection": {
            "artifact_sha256": (
                "4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352"
            ),
            "artifact_size_bytes": 7809,
            "workflow_run": 33272842347,
            "job": 99154326814,
            "output_element_count": 32,
            "mismatched_nonzero_output_count": 28,
            "observed_nonzero_outputs": "signed-zero-only",
            "dxil_bitcast": "bitcast-i16-to-half",
            "rejected_dxil_arithmetic": "fmul-half-on-binary16-subnormal",
        },
        "legacy_half_decode_rejection": {
            "artifact_sha256": (
                "938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627"
            ),
            "artifact_size_bytes": 7909,
            "workflow_run": 33274360343,
            "job": 99158370210,
            "output_element_count": 32,
            "mismatched_nonzero_output_count": 28,
            "observed_nonzero_outputs": "signed-zero-only",
            "rejected_dxil_decode": "legacy-f16-to-f32",
        },
        "integer_decode_half_roundtrip_rejection": {
            "artifact_sha256": (
                "936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625"
            ),
            "artifact_size_bytes": 9240,
            "workflow_run": 33275550062,
            "job": 99161501105,
            "output_element_count": 32,
            "mismatched_nonzero_output_count": 28,
            "observed_nonzero_outputs": "signed-zero-only",
            "integer_decode": "uitofp-i32-to-float",
            "rejected_dxil_roundtrip": (
                "fptrunc-float-to-half-fsub-half-fpext-half-to-float"
            ),
        },
        "widened_float16_narrow_shift_rejection": {
            "artifact_sha256": (
                "7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a"
            ),
            "artifact_size_bytes": 9118,
            "workflow_run": 33277494856,
            "job": 99166677942,
            "output_element_count": 32,
            "mismatched_nonzero_output_count": 28,
            "observed_nonzero_outputs": "signed-zero-only",
            "native_half_operation_count": 0,
            "rejected_dxil_scale_bit_construction": (
                "native-uint16-shift-23-truncated-to-shift-7-and-mask-0xffff"
            ),
        },
        "corrected_dxil_contract": {
            "artifact_sha256": (
                "3fe38e171ba8c8ea1adfc8efad20b242ca02dd05e1a5a53a9b9d1e18459d8c7d"
            ),
            "artifact_size_bytes": 9123,
            "source_bitcast": "integer-ieee754-binary16-to-float32",
            "subnormal_decode": "integer-ieee754-binary16-to-float32",
            "arithmetic": "fmul-float",
            "logical_float16_representation": "widened-float32",
            "native_half_operation_count": 0,
            "scale_bit_construction": "promoted-int-shift-23-before-asfloat",
            "rounding": "none-in-selected-widened-path",
        },
    }
    assert status["software_subgroup"] == {
        "configuration": (
            "project.source_options.metal.target_options.opengl."
            "software_subgroup_width"
        ),
        "width": 32,
        "logical_subgroup_count": 1,
        "shared_float_element_count": 32,
        "activation": "explicit-target-scoped",
        "selected_kernel_operations": ["WaveActiveMax(float)"],
        "control_barrier_instruction_count": 3,
        "hardware_subgroup_extensions_emitted": False,
        "group_non_uniform_instruction_count": 0,
    }
    assert status["index_range_assertions"] == [
        {
            "source": status["source"],
            "expression": "index",
            "minimum": 0,
            "maximum": 31,
        }
    ]
    assert status["runtime_package"] == {
        "artifact_count_per_target": 1,
        "ready_load_unit_count_per_target": 1,
        "blocked_load_unit_count": 0,
        "resource_count_by_target": {"directx": 4, "opengl": 3},
        "data_resources": [
            {
                "binding": 0,
                "role": "input",
                "dtype": "float32",
                "access": "read",
            },
            {
                "binding": 1,
                "role": "global_scale_inert_placeholder",
                "dtype": "float32",
                "access": "read",
            },
            {
                "binding": 2,
                "role": "output",
                "dtype": "float32",
                "access": "read_write",
            },
        ],
        "directx_binding_namespaces": {
            "dispatch_info": "b0",
            "input": "t0",
            "global_scale": "t1",
            "output": "u2",
        },
        "host_global_scale_binding": "omitted-for-selected-specialization",
        "loader_global_scale_allocation": (
            "inert-allocation-required-by-reflected-generic-loader"
        ),
    }
    values = [
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        6.0,
        4.0,
        3.0,
        2.0,
        1.5,
        1.0,
        0.5,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
        0.0,
        0.0,
    ]
    assert status["workload"] == {
        "dtype": "float32",
        "shape": [32],
        "input_values": values,
        "maximum_absolute_value": 6.0,
        "scale_divisor": 6.0,
        "mx_scale": 1.0,
        "reference": "exact-FP4-E2M1-representable-roundtrip",
        "expected_output": "input_values",
        "absolute_tolerance": 0.0,
        "relative_tolerance": 0.0,
    }
    assert status["native_runtime"] == {
        "directx": {
            "platform": "windows-latest",
            "runtime": "direct3d-12-warp",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_fp_quantized_native_loader.py::"
                "test_current_mlx_mxfp4_executes_through_directx_native_loader"
            ),
        },
        "opengl": {
            "platform": "ubuntu-latest",
            "runtime": "mesa-headless-egl-software-opengl",
            "status": "required-on-ci",
            "test": (
                "tests/test_translator/test_mlx_fp_quantized_native_loader.py::"
                "test_current_mlx_mxfp4_executes_with_opengl_software_subgroups"
            ),
        },
    }
    assert status["metal_roundtrip_boundary"] == {
        "status": "outside-selected-native-loader-proof",
        "aggregate_native_baseline": "separate",
        "selected_entry_compiler_validation_included": False,
    }
    assert status["remaining_scope"] == {
        "other_fp_quantized_entries_included": False,
        "global_scale_variants_included": False,
        "other_group_sizes_included": False,
        "other_bit_widths_included": False,
        "other_dtypes_included": False,
        "mlx_host_runtime_integration_included": False,
        "full_mlx_test_suite_included": False,
    }
    assert status["runtime_execution_attempted"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_required"] is True
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False

    readme = " ".join(MLX_README_PATH.read_text(encoding="utf-8").split())
    assert "contracts/fp_quantized.native-loader.dispatch.json" in readme
    assert status["selected_entry_point"] in readme
    assert status["artifacts"]["directx"]["sha256"] in readme
    assert status["artifacts"]["opengl"]["sha256"] in readme
    assert "exact FP4 E2M1 values" in readme
    assert "does not cover the remaining `fp_quantized` entries" in readme


def test_run_checks_full_corpus_mode_skips_reduced_frontier(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()
    verification_options = []
    scan_options = []

    def verify_checkout(*args, **kwargs):
        verification_options.append(kwargs)
        return {"name": "mlx-checkout", "status": "passed"}

    def scan_kernels(*args, **kwargs):
        scan_options.append(kwargs)
        return {"name": "metal-kernel-scan", "status": "passed"}

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        verify_checkout,
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        scan_kernels,
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
        "_check_fft_directx_toolchain",
        lambda *args, **kwargs: pytest.fail(
            "DirectX FFT toolchain proof should not run"
        ),
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
        "_check_fft_opengl_toolchain",
        lambda *args, **kwargs: pytest.fail(
            "OpenGL FFT toolchain proof should not run"
        ),
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
    assert verification_options == [{"expected_commit": CURRENT_MLX_COMMIT}]
    assert scan_options == [
        {
            "expected_unit_count": 42,
            "targets": ("directx", "opengl"),
        }
    ]
    assert result["scope"]["metalRoundTripIncluded"] is True
    assert result["repository"]["commit"] == CURRENT_MLX_COMMIT
    assert result["scope"]["fullCorpusExpectedUnitCount"] == 42
    assert result["scope"]["fullCorpusExpectedArtifactCount"] == 84
    assert result["scope"]["fullCorpusExpectedTranslatedArtifactCount"] == 82
    assert result["scope"]["fullCorpusExpectedFenceFailureCount"] == 2
    assert result["scope"]["referenceAccessorProofIncluded"] is False
    assert result["scope"]["referenceAccessorTargets"] == []
    assert result["scope"]["referenceAccessorDirectxToolchainRequired"] is False
    assert result["scope"]["referenceAccessorOpenglToolchainRequired"] is False
    assert result["scope"]["templateMemberBufferPointerProofIncluded"] is False
    assert result["scope"]["templateMemberBufferPointerTargets"] == []
    assert (
        result["scope"]["templateMemberBufferPointerNativeValidationIncluded"] is False
    )
    assert result["scope"]["fftDirectXToolchainIncluded"] is False
    assert result["scope"]["fftDirectXToolchainRequired"] is False
    assert result["scope"]["fftOpenGLToolchainIncluded"] is False
    assert result["scope"]["fftOpenGLToolchainRequired"] is False
    assert result["scope"]["runtimeParityClaimed"] is False


def test_run_checks_reduced_frontier_includes_runtime_readiness(tmp_path, monkeypatch):
    module = _load_harness()
    mlx_root = tmp_path / "mlx"
    mlx_root.mkdir()

    monkeypatch.setattr(
        module,
        "_verify_mlx_checkout",
        lambda *args, **kwargs: {"name": "mlx-checkout", "status": "passed"},
    )
    monkeypatch.setattr(
        module,
        "_scan_metal_kernels",
        lambda *args, **kwargs: {
            "name": "metal-kernel-scan",
            "status": "passed",
        },
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

    fft_directx_requirements = []

    def fake_fft_directx_toolchain(*args, require_toolchain):
        fft_directx_requirements.append(require_toolchain)
        return {
            "name": "fft-directx-toolchain",
            "status": "passed",
        }

    vulkan_frontier_requirements = []

    def fake_vulkan_frontier(*args, require_toolchain, run_optional_toolchain):
        vulkan_frontier_requirements.append((require_toolchain, run_optional_toolchain))
        return {"name": "vulkan-frontier", "status": "passed"}

    monkeypatch.setattr(module, "_translate_directx_frontier", fake_directx_frontier)
    monkeypatch.setattr(
        module,
        "_check_fft_directx_toolchain",
        fake_fft_directx_toolchain,
    )
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
    fft_opengl_requirements = []

    def fake_fft_opengl_toolchain(*args, require_toolchain):
        fft_opengl_requirements.append(require_toolchain)
        return {
            "name": "fft-opengl-toolchain",
            "status": "passed",
        }

    monkeypatch.setattr(
        module,
        "_check_fft_opengl_toolchain",
        fake_fft_opengl_toolchain,
    )
    monkeypatch.setattr(
        module,
        "_check_gemv_opengl_toolchain",
        lambda *args: {
            "name": "gemv-opengl-toolchain",
            "status": "passed",
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
            require_directx_toolchain=True,
            require_directx_gemv_compiler_frontier=True,
            require_vulkan_toolchain=False,
            require_vulkan_native_runtime=False,
            require_opengl_native_runtime=True,
            require_opengl_frontier_toolchain=True,
            require_opengl_gemv_toolchain=True,
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
        "fft-directx-toolchain",
        "vulkan-frontier",
        "arange-opengl",
        "gemv-directx-compiler-frontier",
        "opengl-frontier",
        "fft-opengl-toolchain",
        "gemv-opengl-toolchain",
        "gemv-vulkan-toolchain",
        "runtime-readiness",
    ]
    assert reference_accessor_requirements == [(True, True)]
    assert directx_frontier_requirements == [True]
    assert fft_directx_requirements == [True]
    assert vulkan_frontier_requirements == [(False, False)]
    assert opengl_frontier_requirements == [True]
    assert fft_opengl_requirements == [True]
    assert runtime_requirements == [
        {
            "require_vulkan_native_runtime": False,
            "require_opengl_native_runtime": True,
        }
    ]
    assert result["scope"]["openglFrontierToolchainRequired"] is True
    assert result["scope"]["directxGemvCompilerFrontierRequired"] is True
    assert result["scope"]["fftDirectXToolchainIncluded"] is True
    assert result["scope"]["fftDirectXToolchainRequired"] is True
    assert result["scope"]["fftOpenGLToolchainIncluded"] is True
    assert result["scope"]["fftOpenGLToolchainRequired"] is True
    assert result["scope"]["openglGemvToolchainRequired"] is True
    assert result["scope"]["openglNativeRuntimeRequired"] is True
    assert result["scope"]["vulkanGemvToolchainRequired"] is True
    assert result["scope"]["runtimeReadinessIncluded"] is True
    assert result["scope"]["runtimeFixtureExecutionIncluded"] is True
    assert result["scope"]["nativeRuntimeExecutionIncluded"] is True
    assert result["scope"]["referenceAccessorProofIncluded"] is True
    assert result["scope"]["referenceAccessorTargets"] == ["directx", "opengl"]
    assert result["scope"]["referenceAccessorDirectxToolchainRequired"] is True
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
        module.MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    )
    assert result["scope"]["directxTranslatedFrontierArtifactCount"] == (
        module.MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT
    )
    assert result["scope"]["directxWorkgroupBlockedFrontierSources"] == list(
        module.MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES
    )
    assert result["scope"]["hostDispatchImportResolvedIssue"] == (
        module.MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
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


def test_rms_norm_contract_fixture_matches_pinned_translation_scope():
    module = _load_rms_norm_harness()
    metadata = _load_rms_norm_fixture_metadata()

    assert module.MLX_COMMIT == "4367c73b60541ddd5a266ce4644fd93d20223b6e"
    assert module.MLX_RMS_NORM_SHA256 == (
        "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee"
    )
    assert module.RMS_NORM_SUBGROUP_WIDTH == 32
    assert module.RMS_NORM_DIRECTX_PROFILE == "cs_6_6"
    assert metadata["upstreamSourceReplacement"] is False
    assert metadata["functionConstant"] == {
        "name": module.RMS_NORM_FUNCTION_CONSTANT_NAME,
        "id": module.RMS_NORM_FUNCTION_CONSTANT_ID,
        "required": True,
        "requiredByEntryPoints": list(module.RMS_NORM_VJP_ENTRY_POINTS),
        "absentFromEntryPoints": list(module.RMS_NORM_FORWARD_ENTRY_POINTS),
    }
    assert module.RMS_NORM_FORWARD_ENTRY_POINTS == (
        "rms_loopedbfloat16",
        "rms_loopedfloat16",
        "rms_loopedfloat32",
        "rmsbfloat16",
        "rmsfloat16",
        "rmsfloat32",
    )
    assert module.RMS_NORM_VJP_ENTRY_POINTS == (
        "vjp_rms_loopedbfloat16",
        "vjp_rms_loopedfloat16",
        "vjp_rms_loopedfloat32",
        "vjp_rmsbfloat16",
        "vjp_rmsfloat16",
        "vjp_rmsfloat32",
    )
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
        "minimumShaderModel": "6.6",
        "requiredSubgroupWidth": 32,
        "subgroupWidthEnforcement": "WaveSize(32)",
        "libraryArtifactCount": 2,
        "hostNamedEntryCountPerArtifact": 12,
        "generatedWaveSizeCountPerArtifact": 12,
        "compiledEntryPointCountPerVariant": 1,
        "compilerRunCount": 2,
        "variants": list(module.RMS_NORM_DIRECTX_VARIANTS),
    }
    assert metadata["opengl"] == {
        "deferred": True,
        "constantId": 20,
        "standaloneArtifactCount": 24,
        "standaloneArtifactCountPerVariant": 12,
        "deferredEntryPointCountPerVariant": 6,
        "constantFreeEntryPointCountPerVariant": 6,
        "deferredArtifactCount": 12,
        "constantFreeArtifactCount": 12,
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
    assert status["source_required_subgroup_width"] == 32
    assert status["host_named_entry_count"] == 12
    assert status["function_constant"] == {
        "name": module.RMS_NORM_FUNCTION_CONSTANT_NAME,
        "id": module.RMS_NORM_FUNCTION_CONSTANT_ID,
        "required": True,
        "required_by_entry_points": list(module.RMS_NORM_VJP_ENTRY_POINTS),
        "absent_from_entry_points": list(module.RMS_NORM_FORWARD_ENTRY_POINTS),
    }
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
    assert status["directx"]["profile"] == "cs_6_6"
    assert status["directx"]["minimum_shader_model"] == "6.6"
    assert status["directx"]["required_subgroup_width"] == 32
    assert status["directx"]["subgroup_width_enforcement"] == "WaveSize(32)"
    assert status["directx"]["compiler_arguments"] == ["-enable-16bit-types"]
    assert status["directx"]["library_artifact_count"] == 2
    assert status["directx"]["execution_entry_count_per_artifact"] == 12
    assert status["directx"]["generated_wave_size_count_per_artifact"] == 12
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
    assert status["opengl"]["subgroup_width_rule_configured"] is False
    assert status["opengl"]["subgroup_width_enforcement_claimed"] is False
    assert status["opengl"]["subgroup_semantic_parity_claimed"] is False
    assert status["opengl"]["native_compilation"] == "required-on-linux-ci"
    assert status["opengl"]["specialization_materialization"] == (
        "deferred-for-vjp-entries"
    )
    assert status["opengl"]["standalone_artifact_count"] == 24
    assert status["opengl"]["standalone_artifact_count_per_variant"] == 12
    assert status["opengl"]["deferred_entry_point_count_per_variant"] == 6
    assert status["opengl"]["constant_free_entry_point_count_per_variant"] == 6
    assert status["opengl"]["deferred_artifact_count"] == 12
    assert status["opengl"]["constant_free_artifact_count"] == 12
    assert status["opengl"]["compiled_artifact_count"] == 24
    assert status["opengl"]["validated_artifact_count"] == 24
    assert status["numerical_execution_included"] is True
    assert status["runtime_integration_included"] is True
    assert status["selected_workload_numerical_parity_verified"] is True
    assert status["bounded_native_runtime_evidence"] == (
        "rms_norm_native_runtime_status"
    )
    assert status["numerical_parity_claimed"] is False
    assert status["runtime_parity_claimed"] is False
    assert status["complete_runtime_coverage_claimed"] is False
    assert status["full_mlx_test_suite_included"] is False
    assert status["runtime_blocked_by"] == list(module.RMS_NORM_RUNTIME_BLOCKERS)
    assert status["runtime_blocked_by"] == []
    assert "https://github.com/CrossGL/crosstl/issues/1795" in (
        expected_gaps["resolved_issues"]
    )


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
    assert identity["sourceSubgroupContract"] == {
        "status": "passed",
        "requiredSubgroupWidth": 32,
        "widthDeclaration": "constexpr int SIMD_SIZE = 32;",
        "widthDeclarationCount": 4,
        "laneBuiltin": "thread_index_in_simdgroup",
        "laneBuiltinCount": 4,
        "groupBuiltin": "simdgroup_index_in_threadgroup",
        "groupBuiltinCount": 4,
        "reductionBuiltin": "simd_sum",
        "reductionCallCount": 12,
        "requiredDirectXEnforcement": {
            "attribute": "WaveSize(32)",
            "minimumShaderModel": "6.6",
            "profile": "cs_6_6",
        },
    }

    source = (mlx_root / module.MLX_RMS_NORM_SOURCE).read_text(encoding="utf-8")
    with pytest.raises(
        module.MlxRmsNormProofError,
        match="exactly four SIMD_SIZE = 32",
    ):
        module._verify_source_subgroup_contract(
            source.replace("constexpr int SIMD_SIZE = 32;", "", 1)
        )

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
    assert result["sourceRequiredSubgroupWidth"] == 32
    assert result["subgroupWidthEnforced"] is True
    assert result["runtimeParityClaimed"] is False
    assert result["numericalExecutionIncluded"] is False
    assert result["nativeCompilation"]["status"] == "not-required"
    assert result["nativeCompilation"]["profile"] == "cs_6_6"
    assert result["nativeCompilation"]["minimumShaderModel"] == "6.6"
    assert result["nativeCompilation"]["subgroupWidth"] == 32
    assert result["nativeCompilation"]["subgroupWidthEnforcement"] == ("WaveSize(32)")
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
        assert execution["sourceRequiredSubgroupWidth"] == 32
        assert execution["subgroupWidthRule"] == {
            "expression": "32",
            "sourcePattern": module.MLX_RMS_NORM_SOURCE,
            "path": f'project.subgroup_width_rules["{module.MLX_RMS_NORM_SOURCE}"]',
        }
        assert execution["subgroupWidthProvenance"] == {
            "kind": "materialized-template-rule",
            "path": f'project.subgroup_width_rules["{module.MLX_RMS_NORM_SOURCE}"]',
        }
        subgroup_enforcement = execution["subgroupWidthEnforcement"]
        assert subgroup_enforcement["mechanism"] == "hlsl-wave-size-attribute"
        assert subgroup_enforcement["minimumShaderModel"] == "6.6"
        assert len(subgroup_enforcement["entryProfiles"]) == 12
        assert all(
            profile["profile"] == "cs_6_6"
            for profile in subgroup_enforcement["entryProfiles"]
        )
        assert execution["generatedWaveSizeContractCount"] == 12
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
    assert report["project"]["subgroupWidthRules"] == {module.MLX_RMS_NORM_SOURCE: "32"}
    assert report["project"]["subgroupWidthRuleCount"] == 1
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
        subgroup_rule_path = (
            f'project.subgroup_width_rules["{module.MLX_RMS_NORM_SOURCE}"]'
        )
        assert execution["subgroupWidthProvenance"] == {
            "kind": "materialized-template-rule",
            "path": subgroup_rule_path,
        }
        assert execution["subgroupWidthEnforcement"] == {
            "mechanism": "hlsl-wave-size-attribute",
            "minimumShaderModel": "6.6",
            "entryProfiles": [
                {
                    "entryPoint": entry["targetEntryPoint"],
                    "profile": "cs_6_6",
                }
                for entry in execution["entryPoints"]
            ],
        }
        assert all(
            entry["subgroupWidth"] == 32
            and entry["subgroupWidthRule"]
            == {
                "expression": "32",
                "sourcePattern": module.MLX_RMS_NORM_SOURCE,
                "path": subgroup_rule_path,
            }
            for entry in execution["entryPoints"]
        )
        generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
        attribute = (
            f"[numthreads({workgroup_size[0]}, {workgroup_size[1]}, "
            f"{workgroup_size[2]})]"
        )
        assert generated.count(attribute) == 12
        assert generated.count("[WaveSize(32)]") == 12
        assert len(re.findall(r"\[\s*WaveSize\s*\(", generated)) == 12
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
    assert (
        result["report"]
        == (report_dir / "rms-norm-opengl-deferred.json")
        .relative_to(mlx_root)
        .as_posix()
    )
    assert result["artifactCount"] == 24
    assert result["artifactCountPerVariant"] == 12
    assert result["subgroupWidthContract"] == {
        "sourceRequiredWidth": 32,
        "projectRuleConfigured": False,
        "targetEnforcementClaimed": False,
        "semanticParityClaimed": False,
    }
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
    assert specialization["deferredEntryPoints"] == list(
        module.RMS_NORM_VJP_ENTRY_POINTS
    )
    assert specialization["absentEntryPoints"] == list(
        module.RMS_NORM_FORWARD_ENTRY_POINTS
    )
    assert specialization["deferredArtifactCount"] == 12
    assert specialization["constantFreeArtifactCount"] == 12
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
        assert variant["deferredSpecializationArtifactCount"] == 6
        assert variant["constantFreeArtifactCount"] == 6
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
            if artifact["sourceEntryPoint"] in module.RMS_NORM_VJP_ENTRY_POINTS:
                assert artifact["specializationConstant"] == {
                    "name": "has_w",
                    "id": 20,
                    "required": True,
                    "deferred": True,
                    "valueProvenance": {"kind": "runtime-override-required"},
                    "specializationMaterialization": specialization[
                        "specializationMaterialization"
                    ],
                }
            else:
                assert artifact["sourceEntryPoint"] in (
                    module.RMS_NORM_FORWARD_ENTRY_POINTS
                )
                assert artifact["specializationConstant"] is None
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
    assert report["project"]["subgroupWidthRules"] == {}
    assert report["project"]["subgroupWidthRuleCount"] == 0
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
        assert "subgroupWidthProvenance" not in execution
        assert "subgroupWidthEnforcement" not in execution
        assert execution["provenance"] == {
            "kind": "project-variant",
            "path": f"project.variants.{variant_name}.workgroup_size",
            "variant": variant_name,
        }
        assert len(execution["entryPoints"]) == 1
        assert execution["entryPoints"][0]["workgroupSize"] == (
            expected_workgroup_sizes[variant_name]
        )
        assert "subgroupWidth" not in execution["entryPoints"][0]
        assert "subgroupWidthRule" not in execution["entryPoints"][0]
        materialization = artifact["templateMaterialization"]
        assert materialization["specializationCount"] == 12
        assert len(materialization["specializations"]) == 12
        assert {
            record["hostName"] for record in materialization["specializations"]
        } == set(module.RMS_NORM_HOST_ENTRY_POINTS)
        generated = (mlx_root / artifact["path"]).read_text(encoding="utf-8")
        local_size = variants[variant_name]["artifacts"][
            list(module.RMS_NORM_HOST_ENTRY_POINTS).index(source_entry)
        ]["generatedLocalSizeContract"]
        assert generated.count(local_size) == 1
        if source_entry in module.RMS_NORM_VJP_ENTRY_POINTS:
            constant = artifact["specializationConstants"][0]
            assert constant["valueProvenance"] == {"kind": "runtime-override-required"}
            assert "concreteValue" not in constant
            assert (
                artifact["specializationMaterialization"]
                == specialization["specializationMaterialization"]
            )
            assert generated.count(specialization["generatedContract"]) == 1
            assert re.search(r"\bhas_w\b", generated)
        else:
            assert source_entry in module.RMS_NORM_FORWARD_ENTRY_POINTS
            assert artifact.get("specializationConstants", []) == []
            assert "specializationMaterialization" not in artifact
            assert generated.count(specialization["generatedContract"]) == 0
            assert re.search(r"\bhas_w\b", generated) is None
        assert "WaveSize" not in generated
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
        generated_entries = []
        for index in range(module.RMS_NORM_EXPECTED_ENTRY_POINT_COUNT):
            entry_point = "CSMain" if index == 0 else f"CSMain_{index + 1}"
            generated_entries.append(
                f"[numthreads({variant['workgroupSize'][0]}, 1, 1)]\n"
                f"[WaveSize({module.RMS_NORM_SUBGROUP_WIDTH})]\n"
                f"void {entry_point}() {{}}"
            )
        artifact_path.write_text(
            "float16_t generated_value;\n" + "\n".join(generated_entries) + "\n",
            encoding="utf-8",
        )
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
    assert directx["profile"] == "cs_6_6"
    assert directx["compilerArguments"] == ["-enable-16bit-types"]
    assert directx["minimumShaderModel"] == "6.6"
    assert directx["subgroupWidth"] == 32
    assert directx["subgroupWidthEnforcement"] == "WaveSize(32)"
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
    assert all(command[1] == "-WX" for command in dxc_commands)
    assert all(command[command.index("-E") + 1] == "CSMain" for command in dxc_commands)
    assert all(
        command[command.index("-T") + 1 : command.index("-E")]
        == ["cs_6_6", "-enable-16bit-types"]
        for command in dxc_commands
    )
    assert all(
        run["profile"] == "cs_6_6"
        and run["compilerArguments"] == ["-enable-16bit-types"]
        and run["minimumShaderModel"] == "6.6"
        and run["subgroupWidth"] == 32
        and run["subgroupWidthEnforcement"] == "WaveSize(32)"
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
