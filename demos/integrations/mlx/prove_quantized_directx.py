#!/usr/bin/env python3
"""Prove one pinned MLX quantized kernel as a DirectX 12 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.project import ProjectConfig, translate_project
from crosstl.project.directx_toolchain import (
    directx_target_profiles_for_source,
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
    hlsl_requires_native_16bit_types,
)

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_QUANTIZED_SOURCE = f"{MLX_KERNEL_ROOT}/quantized.metal"
MLX_QUANTIZED_HEADER = f"{MLX_KERNEL_ROOT}/quantized.h"
MLX_QUANTIZED_ENTRY_POINT = "affine_quantize_float_gs_32_b_2"
MLX_QUANTIZED_GATHER_ENTRY_POINT = "affine_gather_qmv_fast_float_gs_32_b_2"
MLX_QUANTIZED_GATHER_WORKGROUP_SIZE = (32, 2, 1)
MLX_QUANTIZED_GATHER_SUBGROUP_WIDTH = 32

PINNED_FILE_SHA256 = {
    MLX_QUANTIZED_SOURCE: (
        "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
    ),
    MLX_QUANTIZED_HEADER: (
        "4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f"
    ),
}

DIRECTX_TARGET_PROFILE = "directx-12"
DIRECTX_BASE_SHADER_PROFILE = "cs_6_0"
TEMPLATE_SPECIALIZATION_LIMIT = 128
MATERIALIZATION_WORK_LIMIT = 4096
REACHABLE_SPECIALIZATION_COUNT = 6
CONCRETE_SPECIALIZATION_COUNT = 3
PRUNED_CANDIDATE_COUNT = 110861
ENTRY_CONTRACTS = {
    MLX_QUANTIZED_ENTRY_POINT: {
        "specializationName": "affine_quantize",
        "parameters": {"T": "float", "bits": "2", "group_size": "32"},
        "reachableSpecializationCount": REACHABLE_SPECIALIZATION_COUNT,
        "concreteSpecializationCount": CONCRETE_SPECIALIZATION_COUNT,
        "prunedCandidateCount": PRUNED_CANDIDATE_COUNT,
        "generatedContract": "quantize",
    },
    MLX_QUANTIZED_GATHER_ENTRY_POINT: {
        "specializationName": "affine_gather_qmv_fast",
        "parameters": {"T": "float", "bits": "2", "group_size": "32"},
        "reachableSpecializationCount": 11,
        "concreteSpecializationCount": 8,
        "prunedCandidateCount": PRUNED_CANDIDATE_COUNT,
        "generatedContract": "gather-qmv-fast",
        "workgroupSize": MLX_QUANTIZED_GATHER_WORKGROUP_SIZE,
        "subgroupWidth": MLX_QUANTIZED_GATHER_SUBGROUP_WIDTH,
    },
}
GENERATED_ARTIFACTS = {
    MLX_QUANTIZED_ENTRY_POINT: {
        "sha256": "a0f1a10def581f30dc34ed870b9ce36f70fb12abfd447e9b1b369524efde7438",
        "sizeBytes": 4357,
    },
    MLX_QUANTIZED_GATHER_ENTRY_POINT: {
        "sha256": "c64564b5705aa9ef16769c0d0ffda26a8852399d63460079cd449fe71323b5de",
        "sizeBytes": 16359,
    },
}
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/quantized-directx"
SUMMARY_FILENAME = "summary.json"
NATIVE_16_BIT_CAPABILITY = "directx.native-16bit-types"
PACKED_OUTPUT_STORE = "out_[uint((out_index / writes_per_reduce))] = output;"

NON_RUNTIME_CLAIMS = {
    "runtimeExecution": False,
    "numericalParity": False,
    "mlxUnitTests": False,
    "fullMlxTestSuite": False,
}

_PACKED_OUTPUT_STORE_RE = re.compile(
    r"\bout_\s*\[\s*uint\s*\(\s*\(\s*out_index\s*/\s*writes_per_reduce\s*\)"
    r"\s*\)\s*\]\s*=\s*output\s*;"
)
_PACKED_OUTPUT_DECLARATION_RE = re.compile(
    r"\buint\s+output\s*=\s*0\s*;",
)
_MINIMUM_PRECISION_TYPE_RE = re.compile(
    r"\bmin16(?:float|int|uint)(?:[1-4])?\b",
    flags=re.IGNORECASE,
)
_STATIC_ASSERT_RE = re.compile(r"\bstatic_assert\s*\(")
_COMPUTE_ENTRY_RE = re.compile(
    r"\[\s*numthreads\s*\([^\]]+\)\s*\]\s*"
    r"(?:\[\s*WaveSize\s*\([^\]]+\)\s*\]\s*)?"
    r"void\s+CSMain\s*\(",
    flags=re.MULTILINE,
)
_GATHER_EXECUTION_ENTRY_RE = re.compile(
    r"\[\s*numthreads\s*\(\s*32\s*,\s*2\s*,\s*1\s*\)\s*\]\s*"
    r"\[\s*WaveSize\s*\(\s*32\s*\)\s*\]\s*"
    r"void\s+CSMain\s*\(",
    flags=re.MULTILINE,
)
_GATHER_DIRECT_TYPED_BYTE_READ_RE = re.compile(
    r"\bw\s*\[\s*uint\s*\(\s*\(\s*w_offset\s*\+\s*i\s*\)\s*\)\s*\]"
)
_GATHER_UNMATERIALIZED_INDEX_HELPER_RE = re.compile(
    r"(?<![A-Za-z0-9_])elem_to_loc\s*\("
)
_GATHER_INDEX_HELPER_DEFINITION_RE = re.compile(
    r"\buint\s+elem_to_loc_uint32_t\s*\(\s*uint\s+elem\s*,\s*"
    r"StructuredBuffer<int>\s+shape\s*,\s*int64_t\s+shape_offset\s*,\s*"
    r"StructuredBuffer<int64_t>\s+strides\s*,\s*int64_t\s+strides_offset\s*,\s*"
    r"int\s+ndim\s*\)\s*\{"
)
GATHER_INDEX_HELPER_CALL = (
    "elem_to_loc_uint32_t(x_idx, x_shape, int64_t(x_shape_offset), "
    "x_strides, int64_t(x_strides_offset), x_batch_ndims)"
)
_GATHER_POINTER_OFFSET_WRITEBACK_SIGNATURE_RE = re.compile(
    r"\bvoid\s+adjust_matrix_offsets_float\s*\(\s*"
    r"StructuredBuffer<float>\s+x\s*,\s*inout\s+int64_t\s+x_offset\s*,\s*"
    r"StructuredBuffer<uint>\s+w\s*,\s*inout\s+int64_t\s+w_offset\s*,\s*"
    r"StructuredBuffer<float>\s+scales\s*,\s*"
    r"inout\s+int64_t\s+scales_offset\s*,\s*"
    r"StructuredBuffer<float>\s+biases\s*,\s*"
    r"inout\s+int64_t\s+biases_offset\s*,.*?"
    r"RWStructuredBuffer<float>\s+y\s*,\s*inout\s+int64_t\s+y_offset\b",
    flags=re.DOTALL,
)
_GATHER_POINTER_OFFSET_WRITEBACK_CALL_RE = re.compile(
    r"\badjust_matrix_offsets_float\s*\(\s*x\s*,\s*x_offset\s*,\s*"
    r"w\s*,\s*w_offset\s*,\s*scales\s*,\s*scales_offset\s*,\s*"
    r"biases\s*,\s*biases_offset\s*,.*?\by\s*,\s*y_offset\s*,",
    flags=re.DOTALL,
)
_GATHER_POINTER_OFFSET_DOWNSTREAM_CALL_RE = re.compile(
    r"\bqmv_fast_impl_float_32_2\s*\(\s*"
    r"w\s*,\s*int64_t\s*\(\s*w_offset\s*\)\s*,\s*"
    r"scales\s*,\s*int64_t\s*\(\s*scales_offset\s*\)\s*,\s*"
    r"biases\s*,\s*int64_t\s*\(\s*biases_offset\s*\)\s*,\s*"
    r"x\s*,\s*int64_t\s*\(\s*x_offset\s*\)\s*,\s*"
    r"y\s*,\s*int64_t\s*\(\s*y_offset\s*\)",
)
GATHER_MUTABLE_POINTER_OFFSETS = (
    "x_offset",
    "w_offset",
    "scales_offset",
    "biases_offset",
    "y_offset",
)


class MlxQuantizedDirectXProofError(RuntimeError):
    """Raised when the pinned quantized DirectX proof contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxQuantizedDirectXProofError(message)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relpath(path: Path, root: Path) -> str:
    resolved = path.resolve()
    resolved_root = root.resolve()
    _require(
        _is_relative_to(resolved, resolved_root),
        f"proof path must stay inside the MLX checkout: {resolved}",
    )
    return resolved.relative_to(resolved_root).as_posix()


def _resolve_work_dir(mlx_root: Path, value: str | None) -> Path:
    candidate = Path(value) if value else Path(DEFAULT_WORK_DIR)
    if not candidate.is_absolute():
        candidate = mlx_root / candidate
    resolved = candidate.resolve()
    root = mlx_root.resolve()
    _require(
        resolved != root and _is_relative_to(resolved, root),
        f"work directory must be inside the MLX checkout: {resolved}",
    )
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_revision(mlx_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MlxQuantizedDirectXProofError(
            f"could not resolve the MLX checkout revision: {exc}"
        ) from exc
    _require(
        completed.returncode == 0,
        "could not resolve the MLX checkout revision: " + completed.stderr.strip(),
    )
    return completed.stdout.strip()


def _verify_checkout(mlx_root: Path) -> dict[str, Any]:
    root = mlx_root.resolve()
    _require(root.is_dir(), f"MLX checkout does not exist: {root}")
    revision = _git_revision(root)
    _require(
        revision == MLX_COMMIT,
        f"MLX checkout must be pinned to {MLX_COMMIT}; found {revision}",
    )

    files = []
    for relative_path, expected_hash in PINNED_FILE_SHA256.items():
        path = (root / relative_path).resolve()
        _require(
            _is_relative_to(path, root) and path.is_file(),
            f"pinned MLX file is missing or outside the checkout: {relative_path}",
        )
        actual_hash = _sha256(path)
        _require(
            actual_hash == expected_hash,
            f"pinned MLX file SHA-256 mismatch for {relative_path}: "
            f"expected {expected_hash}, found {actual_hash}",
        )
        files.append(
            {
                "path": relative_path,
                "kind": "source" if relative_path == MLX_QUANTIZED_SOURCE else "header",
                "hash": {"algorithm": "sha256", "value": actual_hash},
            }
        )
    return {"status": "passed", "commit": revision, "files": files}


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    log_dir: Path,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout"
    stderr_path = log_dir / f"{name}.stderr"
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr += f"\n{name} timed out after {timeout_seconds} seconds.\n"
    except OSError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "command": list(command),
        "returncode": returncode,
        "stdoutPath": stdout_path,
        "stderrPath": stderr_path,
    }


def _project_config(
    mlx_root: Path,
    work_dir: Path,
    *,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> ProjectConfig:
    _require(entry_point in ENTRY_CONTRACTS, f"unsupported proof entry: {entry_point}")
    entry_contract = ENTRY_CONTRACTS[entry_point]
    workgroup_size = entry_contract.get("workgroupSize")
    subgroup_width = entry_contract.get("subgroupWidth")
    return ProjectConfig(
        root=mlx_root,
        source_roots=(MLX_KERNEL_ROOT,),
        include_patterns=(MLX_QUANTIZED_SOURCE,),
        targets=(DIRECTX_TARGET_PROFILE,),
        output_dir=_relpath(work_dir / "artifacts", mlx_root),
        source_overrides={MLX_QUANTIZED_SOURCE: "metal"},
        entry_points={MLX_QUANTIZED_SOURCE: entry_point},
        include_dirs=(".",),
        source_options={
            "metal": {
                "max_template_specializations": TEMPLATE_SPECIALIZATION_LIMIT,
                "max_template_materialization_work": MATERIALIZATION_WORK_LIMIT,
            }
        },
        workgroup_size_rules=(
            {MLX_QUANTIZED_SOURCE: [str(value) for value in workgroup_size]}
            if workgroup_size is not None
            else {}
        ),
        subgroup_width_rules=(
            {MLX_QUANTIZED_SOURCE: str(subgroup_width)}
            if subgroup_width is not None
            else {}
        ),
    )


def _translate_report(config: ProjectConfig, *, report_path: Path) -> dict[str, Any]:
    try:
        payload = translate_project(
            config,
            format_output=False,
            validate=False,
        ).to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxQuantizedDirectXProofError(
            f"DirectX project translation raised {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _require_translation_summary(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), "project report summary is missing")
    diagnostic_counts = summary.get("diagnosticCounts")
    _require(
        summary.get("unitCount") == 1
        and summary.get("targetCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and summary.get("skippedCount") == 0
        and diagnostic_counts == {"error": 0, "note": 0, "warning": 0},
        "pinned quantized.metal report must contain one translated artifact "
        "and no diagnostics",
    )
    _require(
        payload.get("diagnostics") == [],
        "pinned quantized.metal translation emitted diagnostics",
    )


def _is_sha256_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("algorithm") == "sha256"
        and re.fullmatch(r"[0-9a-f]{64}", str(value.get("value", ""))) is not None
    )


def _validate_execution_report(
    payload: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    entry_point: str,
    entry_contract: Mapping[str, Any],
) -> None:
    workgroup_size = entry_contract.get("workgroupSize")
    subgroup_width = entry_contract.get("subgroupWidth")
    if workgroup_size is None and subgroup_width is None:
        return

    _require(
        workgroup_size is not None and subgroup_width is not None,
        "quantized execution rules must define workgroup and subgroup together",
    )
    workgroup_size = list(workgroup_size)
    workgroup_rule_path = f'project.workgroup_size_rules["{MLX_QUANTIZED_SOURCE}"]'
    subgroup_rule_path = f'project.subgroup_width_rules["{MLX_QUANTIZED_SOURCE}"]'
    expected_workgroup_rule = {
        "components": [str(value) for value in workgroup_size],
        "sourcePattern": MLX_QUANTIZED_SOURCE,
        "path": workgroup_rule_path,
    }
    expected_subgroup_rule = {
        "expression": str(subgroup_width),
        "sourcePattern": MLX_QUANTIZED_SOURCE,
        "path": subgroup_rule_path,
    }
    project = payload.get("project")
    _require(
        isinstance(project, Mapping)
        and project.get("workgroupSize") is None
        and project.get("workgroupSizeRules")
        == {MLX_QUANTIZED_SOURCE: [str(value) for value in workgroup_size]}
        and project.get("workgroupSizeRuleCount") == 1
        and project.get("subgroupWidthRules")
        == {MLX_QUANTIZED_SOURCE: str(subgroup_width)}
        and project.get("subgroupWidthRuleCount") == 1,
        "quantized project report did not retain the pinned execution rules",
    )

    execution = artifact.get("execution")
    entries = execution.get("entryPoints") if isinstance(execution, Mapping) else None
    _require(
        isinstance(execution, Mapping)
        and execution.get("sourceEntryPoints") == [entry_point]
        and execution.get("provenance")
        == {"kind": "materialized-template-rule", "path": workgroup_rule_path}
        and execution.get("subgroupWidthProvenance")
        == {"kind": "materialized-template-rule", "path": subgroup_rule_path}
        and execution.get("subgroupWidthEnforcement")
        == {
            "mechanism": "hlsl-wave-size-attribute",
            "minimumShaderModel": "6.6",
            "entryProfiles": [{"entryPoint": "CSMain", "profile": "cs_6_6"}],
        }
        and _is_sha256_identity(execution.get("identity"))
        and isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], Mapping),
        "quantized artifact execution metadata changed",
    )
    execution_entry = entries[0]
    _require(
        execution_entry.get("sourceEntryPoint") == entry_point
        and execution_entry.get("materializedEntryPoint") == entry_point
        and execution_entry.get("targetEntryPoint") == "CSMain"
        and execution_entry.get("workgroupSize") == workgroup_size
        and execution_entry.get("rule") == expected_workgroup_rule
        and execution_entry.get("subgroupWidth") == subgroup_width
        and execution_entry.get("subgroupWidthRule") == expected_subgroup_rule
        and execution_entry.get("materialization")
        == {
            "name": entry_contract["specializationName"],
            "hostName": entry_point,
            "materializedName": entry_point,
        }
        and execution_entry.get("parameters") == entry_contract["parameters"]
        and _is_sha256_identity(execution_entry.get("identity")),
        "quantized per-entry execution contract changed",
    )


def _translated_artifact(
    payload: Mapping[str, Any],
    *,
    mlx_root: Path,
    work_dir: Path,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> tuple[Mapping[str, Any], Path]:
    entry_contract = ENTRY_CONTRACTS.get(entry_point)
    _require(entry_contract is not None, f"unsupported proof entry: {entry_point}")
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "translation did not produce a project portability report",
    )
    _require_translation_summary(payload)
    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], Mapping),
        "project report must contain exactly one DirectX artifact record",
    )
    artifact = artifacts[0]
    _require(
        artifact.get("source") == MLX_QUANTIZED_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "directx"
        and artifact.get("status") == "translated"
        and artifact.get("sourceHash")
        == {
            "algorithm": "sha256",
            "value": PINNED_FILE_SHA256[MLX_QUANTIZED_SOURCE],
        }
        and artifact.get("provenance")
        == {"pipeline": "entry-scoped-translate", "intermediate": "crossgl"},
        "DirectX artifact provenance does not match pinned quantized.metal",
    )
    _require(
        artifact.get("entryPoint")
        == {
            "source": entry_point,
            "target": "CSMain",
            "stage": "compute",
        },
        "selected quantized entry-point identity was not preserved",
    )
    _validate_execution_report(
        payload,
        artifact,
        entry_point=entry_point,
        entry_contract=entry_contract,
    )
    materialization = artifact.get("templateMaterialization")
    specializations = (
        materialization.get("specializations")
        if isinstance(materialization, Mapping)
        else None
    )
    accounting = (
        materialization.get("accounting")
        if isinstance(materialization, Mapping)
        else None
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("specializationCount")
        == entry_contract["concreteSpecializationCount"]
        and materialization.get("unsupported") == []
        and isinstance(specializations, list)
        and len(specializations) == entry_contract["concreteSpecializationCount"]
        and isinstance(accounting, Mapping)
        and accounting.get("reachableSpecializationCount")
        == entry_contract["reachableSpecializationCount"]
        and accounting.get("prunedCandidateCount")
        == entry_contract["prunedCandidateCount"],
        "quantized specialization accounting changed",
    )
    selected_specializations = [
        specialization
        for specialization in specializations
        if isinstance(specialization, Mapping)
        and specialization.get("name") == entry_contract["specializationName"]
        and specialization.get("hostName") == entry_point
    ]
    _require(
        len(selected_specializations) == 1
        and selected_specializations[0].get("materializedName") == entry_point
        and selected_specializations[0].get("parameters")
        == entry_contract["parameters"],
        f"selected {entry_contract['specializationName']} specialization changed",
    )

    artifact_path = (mlx_root / str(artifact.get("path", ""))).resolve()
    _require(
        _is_relative_to(artifact_path, work_dir.resolve()) and artifact_path.is_file(),
        f"generated HLSL is missing or outside the work directory: {artifact_path}",
    )
    _require(
        artifact_path.suffix == ".hlsl"
        and artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": _sha256(artifact_path)}
        and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size,
        "generated HLSL identity does not match the project report",
    )
    generated = artifact_path.read_text(encoding="utf-8")
    expected_capabilities = (
        [NATIVE_16_BIT_CAPABILITY]
        if hlsl_requires_native_16bit_types(generated)
        else []
    )
    _require(
        artifact.get("requiredCapabilities") == expected_capabilities,
        "DirectX artifact capabilities do not match the generated HLSL",
    )
    return artifact, artifact_path


def _validate_quantize_generated_hlsl(generated: str) -> dict[str, Any]:
    _require(
        len(_PACKED_OUTPUT_DECLARATION_RE.findall(generated)) == 1,
        "the bits=2 packed output specialization must retain its uint32 type",
    )
    _require(
        len(_PACKED_OUTPUT_STORE_RE.findall(generated)) == 1,
        "the bits=2 packed output must be stored without a width-changing conversion",
    )
    return {
        "typedResourceStore": {
            "status": "passed",
            "resource": "out_",
            "resourceElementType": "uint",
            "sourceSpecializedType": "uint32_t",
            "generatedValueType": "uint",
            "conversion": "not-required",
            "generatedStore": PACKED_OUTPUT_STORE,
        }
    }


def _validate_gather_generated_hlsl(generated: str) -> dict[str, Any]:
    normalized_generated = " ".join(generated.split())
    normalized_generated = re.sub(r"\(\s+", "(", normalized_generated)
    normalized_generated = re.sub(r"\s+\)", ")", normalized_generated)
    load_signature = (
        "float load_vector_float_float_16_2(StructuredBuffer<float> x, "
        "int64_t x_offset, inout float x_thread[16], int x_thread_base)"
    )
    qdot_signature = (
        "float qdot_float_16_2(StructuredBuffer<uint> w, int64_t w_offset, "
        "inout float x_thread[16], int x_thread_base, float scale, float bias, "
        "float sum)"
    )
    private_writes = (
        "x_thread[(x_thread_base + i)] =",
        "x_thread[(x_thread_base + (i + 1))] =",
        "x_thread[(x_thread_base + (i + 2))] =",
        "x_thread[(x_thread_base + (i + 3))] =",
    )
    byte_read = "w[uint(((w_offset + i)) / 4)]"
    lane_shift = "uint((((w_offset + i)) % 4) * 8)"
    call_offset = (
        "int64_t((((w_offset * 4) + "
        "(ws_offset + (row * in_vec_size_w))) + wl_offset))"
    )
    _require(
        load_signature in normalized_generated, "the 16-element load helper changed"
    )
    _require(
        qdot_signature in normalized_generated, "the 16-element qdot helper changed"
    )
    _require(
        all(generated.count(write) == 1 for write in private_writes),
        "the load helper must write all four values in each stepped iteration",
    )
    _require(
        generated.count(byte_read) == 4
        and generated.count(lane_shift) == 4
        and generated.count("& 255u") >= 4,
        "the qdot helper must unpack each uint8 weight from its 32-bit backing word",
    )
    _require(
        call_offset in generated,
        "the qdot call must compose root, byte-view, row, and local offsets",
    )
    _require(
        _GATHER_DIRECT_TYPED_BYTE_READ_RE.search(generated) is None,
        "the qdot helper must not index packed uint8 weights as typed uint words",
    )
    _require(
        len(_GATHER_EXECUTION_ENTRY_RE.findall(generated)) == 1,
        "the gather entry must enforce a 32x2x1 workgroup and 32-lane wave",
    )
    _require(
        len(_GATHER_INDEX_HELPER_DEFINITION_RE.findall(generated)) == 1
        and generated.count(GATHER_INDEX_HELPER_CALL) == 1
        and _GATHER_UNMATERIALIZED_INDEX_HELPER_RE.search(generated) is None,
        "the scalar index helper must be materialized with both resource offsets",
    )
    mutable_offset_declarations = tuple(
        f"int64_t {name} = int64_t(0);" for name in GATHER_MUTABLE_POINTER_OFFSETS
    )
    mutable_offset_updates = tuple(
        f"{name} +=" for name in GATHER_MUTABLE_POINTER_OFFSETS
    )
    _require(
        len(_GATHER_POINTER_OFFSET_WRITEBACK_SIGNATURE_RE.findall(generated)) == 2
        and all(
            generated.count(declaration) == 1
            for declaration in mutable_offset_declarations
        )
        and all(generated.count(update) >= 1 for update in mutable_offset_updates)
        and len(_GATHER_POINTER_OFFSET_WRITEBACK_CALL_RE.findall(generated)) == 1
        and len(_GATHER_POINTER_OFFSET_DOWNSTREAM_CALL_RE.findall(generated)) == 1,
        "the mutable resource pointer offsets must write back into the entry and "
        "feed the downstream quantized helper",
    )
    return {
        "executionContract": {
            "status": "passed",
            "workgroupSize": list(MLX_QUANTIZED_GATHER_WORKGROUP_SIZE),
            "subgroupWidth": MLX_QUANTIZED_GATHER_SUBGROUP_WIDTH,
            "subgroupWidthEnforcement": "WaveSize(32)",
        },
        "privateArrayAliasing": {
            "status": "passed",
            "helper": "load_vector_float_float_16_2",
            "extent": 16,
            "writeCountPerIteration": 4,
            "callBaseOffset": 0,
        },
        "weightByteView": {
            "status": "passed",
            "helper": "qdot_float_16_2",
            "backingElementType": "uint32_t",
            "viewElementType": "uint8_t",
            "laneReadCount": 4,
            "composedOffsetTerms": [
                "w_offset * 4",
                "ws_offset",
                "row * in_vec_size_w",
                "wl_offset",
            ],
        },
        "indexHelperMaterialization": {
            "status": "passed",
            "helper": "elem_to_loc_uint32_t",
            "sourceIndexType": "uint32_t",
            "generatedIndexType": "uint",
            "resourceOffsets": ["x_shape_offset", "x_strides_offset"],
        },
        "pointerReferenceOffsetWriteback": {
            "status": "passed",
            "helper": "adjust_matrix_offsets_float",
            "offsets": list(GATHER_MUTABLE_POINTER_OFFSETS),
            "downstreamHelper": "qmv_fast_impl_float_32_2",
        },
    }


def _validate_generated_hlsl(
    artifact_path: Path,
    *,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated = artifact_path.read_text(encoding="utf-8")
    entry_contract = ENTRY_CONTRACTS.get(entry_point)
    _require(entry_contract is not None, f"unsupported proof entry: {entry_point}")
    _require(
        len(_COMPUTE_ENTRY_RE.findall(generated)) == 1,
        "generated HLSL must define exactly one CSMain compute entry",
    )
    _require(
        _STATIC_ASSERT_RE.search(generated) is None,
        "generated HLSL must not retain Metal static_assert expressions",
    )
    _require(
        _MINIMUM_PRECISION_TYPE_RE.search(generated) is None,
        "generated HLSL must not promote exact source types to min16 types",
    )
    generated_contract = entry_contract["generatedContract"]
    if generated_contract == "quantize":
        entry_checks = _validate_quantize_generated_hlsl(generated)
    elif generated_contract == "gather-qmv-fast":
        entry_checks = _validate_gather_generated_hlsl(generated)
    else:
        raise MlxQuantizedDirectXProofError(
            f"unsupported generated contract: {generated_contract}"
        )

    compatible_target_profiles = directx_target_profiles_for_source(generated)
    profile = dxc_profile_for_source(DIRECTX_BASE_SHADER_PROFILE, generated)
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    expected_profile = (
        "cs_6_6"
        if entry_contract.get("subgroupWidth") is not None
        else DIRECTX_BASE_SHADER_PROFILE
    )
    _require(
        not hlsl_requires_native_16bit_types(generated),
        "pinned quantized HLSL must not require native 16-bit types",
    )
    _require(
        DIRECTX_TARGET_PROFILE in compatible_target_profiles,
        "generated quantized HLSL must remain compatible with DirectX 12",
    )
    _require(
        profile == expected_profile and compiler_arguments == (),
        "generated quantized HLSL compiler requirements must match its features",
    )
    checks = {
        "status": "passed",
        "entryPoint": "CSMain",
        "native16BitTypes": "not-required",
        "staticAssertions": "absent",
        "minimumPrecisionTypes": "absent",
        **entry_checks,
    }
    compiler_contract = {
        "entryPoint": "CSMain",
        "profile": profile,
        "compilerArguments": list(compiler_arguments),
        "targetProfiles": [DIRECTX_TARGET_PROFILE],
        "warningsAsErrors": True,
    }
    return checks, compiler_contract


def _compile_directx_artifact(
    artifact_path: Path,
    compiler_contract: Mapping[str, Any],
    *,
    dxc: str | None,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    required: bool,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    common = {
        "required": required,
        "compiler": "dxc",
        "entryPoint": compiler_contract["entryPoint"],
        "profile": compiler_contract["profile"],
        "compilerArguments": list(compiler_contract["compilerArguments"]),
        "targetProfiles": list(compiler_contract["targetProfiles"]),
        "warningsAsErrors": True,
    }
    if dxc is None:
        _require(
            not required,
            "DirectX quantized proof requires dxc, but it is unavailable",
        )
        return {
            **common,
            "available": False,
            "status": "not-required",
            "reason": "dxc-unavailable",
            "compiledArtifactCount": 0,
        }

    output_path = work_dir / "native" / "directx" / f"{entry_point}.dxil"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    command = [
        dxc,
        "-WX",
        "-T",
        str(compiler_contract["profile"]),
        *[str(value) for value in compiler_contract["compilerArguments"]],
        "-E",
        str(compiler_contract["entryPoint"]),
        str(artifact_path),
        "-Fo",
        str(output_path),
    ]
    result = _run_command(
        "compile-quantized-directx",
        command,
        log_dir=log_dir,
    )
    _require(
        result["returncode"] == 0,
        "DXC rejected the generated quantized artifact",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "DXC did not emit a nonempty quantized DXIL artifact",
    )
    return {
        **common,
        "available": True,
        "status": "compiled",
        "artifact": _relpath(artifact_path, mlx_root),
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compiledArtifactHash": {
            "algorithm": "sha256",
            "value": _sha256(output_path),
        },
        "compiledArtifactCount": 1,
        "stdout": _relpath(result["stdoutPath"], mlx_root),
        "stderr": _relpath(result["stderrPath"], mlx_root),
    }


def run_proof(
    mlx_root: Path,
    work_dir: Path,
    *,
    require_directx_toolchain: bool = False,
    clean: bool = True,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    root = mlx_root.resolve()
    resolved_work_dir = _resolve_work_dir(root, str(work_dir))
    provenance = _verify_checkout(root)
    if clean and resolved_work_dir.exists():
        shutil.rmtree(resolved_work_dir)
    report_dir = resolved_work_dir / "reports"
    log_dir = resolved_work_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "quantized-metal-selected-entry.json"
    payload = _translate_report(
        _project_config(root, resolved_work_dir, entry_point=entry_point),
        report_path=report_path,
    )
    artifact, artifact_path = _translated_artifact(
        payload,
        mlx_root=root,
        work_dir=resolved_work_dir,
        entry_point=entry_point,
    )
    expected_artifact = GENERATED_ARTIFACTS[entry_point]
    _require(
        artifact["generatedHash"]
        == {"algorithm": "sha256", "value": expected_artifact["sha256"]}
        and artifact["generatedSizeBytes"] == expected_artifact["sizeBytes"],
        f"generated DirectX artifact identity changed for {entry_point}",
    )
    generated_checks, compiler_contract = _validate_generated_hlsl(
        artifact_path, entry_point=entry_point
    )
    compiler = _compile_directx_artifact(
        artifact_path,
        compiler_contract,
        dxc=shutil.which("dxc"),
        mlx_root=root,
        work_dir=resolved_work_dir,
        log_dir=log_dir,
        required=require_directx_toolchain,
        entry_point=entry_point,
    )
    entry_contract = ENTRY_CONTRACTS[entry_point]
    translation_scope = {
        "source": MLX_QUANTIZED_SOURCE,
        "selectedEntryPoint": entry_point,
        "sourceBackend": "metal",
        "sourceOverride": "metal",
        "includeDirectories": ["."],
        "target": DIRECTX_TARGET_PROFILE,
        "projectTranslationApi": "crosstl.project.translate_project",
        "materializationLimits": {
            "maxTemplateSpecializations": TEMPLATE_SPECIALIZATION_LIMIT,
            "maxTemplateMaterializationWork": MATERIALIZATION_WORK_LIMIT,
        },
    }
    if entry_contract.get("workgroupSize") is not None:
        translation_scope["workgroupSize"] = list(entry_contract["workgroupSize"])
    if entry_contract.get("subgroupWidth") is not None:
        translation_scope["subgroupWidth"] = entry_contract["subgroupWidth"]

    summary = {
        "schema_version": 1,
        "kind": "crosstl-mlx-quantized-directx-toolchain-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "translation": translation_scope,
            "compiler": {
                "compiler": "dxc",
                "required": require_directx_toolchain,
                "warningsAsErrors": True,
            },
            "runtime": {
                "executionAttempted": False,
                "backendIntegrationAttempted": False,
                "mlxTestsRun": False,
            },
            "numerical": {
                "comparisonAttempted": False,
                "parityClaimed": False,
            },
        },
        "claims": {
            "projectTranslation": True,
            "nativeCompilation": compiler["status"] == "compiled",
            **NON_RUNTIME_CLAIMS,
        },
        "provenance": provenance,
        "translation": {
            "status": "passed",
            "report": _relpath(report_path, root),
            "artifact": _relpath(artifact_path, root),
            "artifactHash": artifact["generatedHash"],
            "entryPoint": artifact["entryPoint"],
            "requiredCapabilities": list(artifact["requiredCapabilities"]),
            "templateMaterialization": artifact["templateMaterialization"],
            "generatedChecks": generated_checks,
            **(
                {"execution": dict(artifact["execution"])}
                if isinstance(artifact.get("execution"), Mapping)
                else {}
            ),
        },
        "compiler": compiler,
        "runtime": {
            "status": "not-attempted",
            "reason": "compile-only proof; no Direct3D dispatch or MLX runtime wiring",
        },
        "numerical": {
            "status": "not-attempted",
            "reason": "no translated kernel execution or reference comparison",
        },
        "status": "passed",
    }
    _write_json(resolved_work_dir / SUMMARY_FILENAME, summary)
    return summary


def _failure_summary(
    *, required: bool, error: str, entry_point: str = MLX_QUANTIZED_ENTRY_POINT
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-quantized-directx-toolchain-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "translation": {
                "source": MLX_QUANTIZED_SOURCE,
                "selectedEntryPoint": entry_point,
                "target": DIRECTX_TARGET_PROFILE,
            },
            "compiler": {"compiler": "dxc", "required": required},
            "runtime": {"executionAttempted": False, "mlxTestsRun": False},
            "numerical": {"comparisonAttempted": False, "parityClaimed": False},
        },
        "claims": {
            "projectTranslation": False,
            "nativeCompilation": False,
            **NON_RUNTIME_CLAIMS,
        },
        "status": "failed",
        "error": error,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove a pinned MLX quantized entry through project translation and "
            "optional required DXC compilation."
        )
    )
    parser.add_argument("--mlx-root", required=True, help="Path to the MLX checkout")
    parser.add_argument(
        "--entry-point",
        choices=tuple(ENTRY_CONTRACTS),
        default=MLX_QUANTIZED_ENTRY_POINT,
        help="Pinned quantized entry to translate and compile.",
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "Generated report/artifact directory inside the MLX checkout. "
            f"Defaults to <mlx-root>/{DEFAULT_WORK_DIR}."
        ),
    )
    parser.add_argument(
        "--require-directx-toolchain",
        action="store_true",
        help="Require DXC compilation instead of accepting translation-only proof.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing files in the generated work directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlx_root = Path(args.mlx_root).resolve()
    try:
        work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    except MlxQuantizedDirectXProofError as exc:
        print(f"MLX quantized DirectX proof failed: {exc}", file=sys.stderr)
        return 1

    summary_path = work_dir / SUMMARY_FILENAME
    try:
        run_proof(
            mlx_root,
            work_dir,
            require_directx_toolchain=args.require_directx_toolchain,
            clean=not args.no_clean,
            entry_point=args.entry_point,
        )
    except MlxQuantizedDirectXProofError as exc:
        _write_json(
            summary_path,
            _failure_summary(
                required=args.require_directx_toolchain,
                error=str(exc),
                entry_point=args.entry_point,
            ),
        )
        print(f"MLX quantized DirectX proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    print(f"MLX quantized DirectX proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
