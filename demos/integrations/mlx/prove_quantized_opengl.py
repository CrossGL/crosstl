#!/usr/bin/env python3
"""Prove one pinned MLX quantized kernel as an OpenGL artifact."""

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

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_QUANTIZED_SOURCE = f"{MLX_KERNEL_ROOT}/quantized.metal"
MLX_QUANTIZED_HEADER = f"{MLX_KERNEL_ROOT}/quantized.h"
MLX_QUANTIZED_ENTRY_POINT = "affine_quantize_float_gs_32_b_2"
MLX_QUANTIZED_GATHER_ENTRY_POINT = "affine_gather_qmv_fast_float_gs_32_b_2"
MLX_QUANTIZED_GATHER_WORKGROUP_SIZE = (32, 2, 1)

PINNED_FILE_SHA256 = {
    MLX_QUANTIZED_SOURCE: (
        "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd"
    ),
    MLX_QUANTIZED_HEADER: (
        "4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f"
    ),
}

TARGET = "opengl"
TEMPLATE_SPECIALIZATION_LIMIT = 128
MATERIALIZATION_WORK_LIMIT = 4096
REACHABLE_SPECIALIZATION_COUNT = 6
CONCRETE_SPECIALIZATION_COUNT = 3
PRUNED_CANDIDATE_COUNT = 110861
INDEX_RANGE_MINIMUM = 0
INDEX_RANGE_MAXIMUM = 2147483647
INDEX_RANGE_EXPRESSIONS = (
    "in_index + i",
    "gindex",
    "out_index / writes_per_reduce",
)
INDEX_RANGE_ASSERTIONS = tuple(
    {
        "source": MLX_QUANTIZED_SOURCE,
        "expression": expression,
        "minimum": INDEX_RANGE_MINIMUM,
        "maximum": INDEX_RANGE_MAXIMUM,
    }
    for expression in INDEX_RANGE_EXPRESSIONS
)
GATHER_INDEX_RANGE_EXPRESSIONS = (
    "tid.z * lhs_strides[0]",
    "tid.z * rhs_strides[0]",
    "idx.x",
    "idx.y",
)
GATHER_INDEX_RANGE_ASSERTIONS = tuple(
    {
        "source": MLX_QUANTIZED_SOURCE,
        "expression": expression,
        "minimum": INDEX_RANGE_MINIMUM,
        "maximum": INDEX_RANGE_MAXIMUM,
    }
    for expression in GATHER_INDEX_RANGE_EXPRESSIONS
)
ENTRY_CONTRACTS = {
    MLX_QUANTIZED_ENTRY_POINT: {
        "specializationName": "affine_quantize",
        "parameters": {"T": "float", "bits": "2", "group_size": "32"},
        "reachableSpecializationCount": REACHABLE_SPECIALIZATION_COUNT,
        "concreteSpecializationCount": CONCRETE_SPECIALIZATION_COUNT,
        "prunedCandidateCount": PRUNED_CANDIDATE_COUNT,
        "generatedContract": "quantize",
        "indexRangeAssertions": INDEX_RANGE_ASSERTIONS,
    },
    MLX_QUANTIZED_GATHER_ENTRY_POINT: {
        "specializationName": "affine_gather_qmv_fast",
        "parameters": {"T": "float", "bits": "2", "group_size": "32"},
        "reachableSpecializationCount": 11,
        "concreteSpecializationCount": 8,
        "prunedCandidateCount": PRUNED_CANDIDATE_COUNT,
        "generatedContract": "gather-qmv-fast",
        "indexRangeAssertions": GATHER_INDEX_RANGE_ASSERTIONS,
        "workgroupSize": MLX_QUANTIZED_GATHER_WORKGROUP_SIZE,
    },
}
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/quantized-opengl"
SUMMARY_FILENAME = "summary.json"

NON_RUNTIME_CLAIMS = {
    "runtimeExecution": False,
    "numericalParity": False,
    "mlxUnitTests": False,
    "fullMlxTestSuite": False,
}

_GENERATED_SEMANTIC_SENTINELS = {
    "minimumReduction": "w_min = subgroupMin(w_min);",
    "maximumReduction": "w_max = subgroupMax(w_max);",
    "scaleCalculation": "float scale = max(((w_max - w_min) / n_bins), eps);",
    "edgeRounding": "float q0 = round((edge / scale));",
    "quantization": (
        "uint val = bitfieldExtract(uint(min(round(((w_thread[i] - bias) / "
        "scale)), n_bins)), 0, 8);"
    ),
    "subgroupPacking": "uint sval = subgroupShuffleDown(val, j);",
}
_GENERATED_INDEX_SENTINELS = {
    "in_index + i": "w[uint((in_index + uint64_t(i)))]",
    "gindex": ("scales[uint(gindex)]", "biases[uint(gindex)]"),
    "out_index / writes_per_reduce": (
        "out_[uint((out_index / uint64_t(writes_per_reduce)))]"
    ),
}
_GATHER_QDOT_DEFINITION_RE = re.compile(
    r"\bfloat\s+qdot_float_16_2[A-Za-z0-9_]*\s*"
    r"\([^)]*\bint\s+w_offset\s*,\s*\bint\s+w_byte_offset\s*\)\s*"
    r"\{(?P<body>.*?)^\}",
    re.MULTILINE | re.DOTALL,
)
_GATHER_QDOT_CALL_RE = re.compile(
    r"\bqdot_float_16_2[A-Za-z0-9_]*\s*"
    r"\([^;]*\bint\s*\(\s*wl_offset\s*\)\s*,\s*"
    r"\bint\s*\(\s*\(\s*w_offset\s*\*\s*4\s*\)\s*\)\s*\)"
)
_GATHER_ADJUST_DEFINITION_RE = re.compile(
    r"\bvoid\s+adjust_matrix_offsets_float[A-Za-z0-9_]*\s*"
    r"\([^)]*\binout\s+int\s+x_offset\b"
    r"[^)]*\binout\s+int\s+w_offset\b"
    r"[^)]*\binout\s+int\s+scales_offset\b"
    r"[^)]*\binout\s+int\s+biases_offset\b"
    r"[^)]*\binout\s+int\s+y_offset\b",
    re.DOTALL,
)
GATHER_MUTABLE_POINTER_OFFSETS = (
    "x_offset",
    "w_offset",
    "scales_offset",
    "biases_offset",
    "y_offset",
)


class MlxQuantizedOpenGLProofError(RuntimeError):
    """Raised when the pinned quantized OpenGL proof contract is not met."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxQuantizedOpenGLProofError(message)


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
        raise MlxQuantizedOpenGLProofError(
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
    entry_contract = ENTRY_CONTRACTS.get(entry_point)
    _require(entry_contract is not None, f"unsupported proof entry: {entry_point}")
    workgroup_size = entry_contract.get("workgroupSize")
    return ProjectConfig(
        root=mlx_root,
        source_roots=(MLX_KERNEL_ROOT,),
        include_patterns=(MLX_QUANTIZED_SOURCE,),
        targets=(TARGET,),
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
        index_range_assertions=entry_contract["indexRangeAssertions"],
    )


def _translate_report(config: ProjectConfig, *, report_path: Path) -> dict[str, Any]:
    try:
        payload = translate_project(
            config,
            format_output=False,
            validate=False,
        ).to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxQuantizedOpenGLProofError(
            f"OpenGL project translation raised {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _require_translation_summary(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), "project report summary is missing")
    _require(
        summary.get("unitCount") == 1
        and summary.get("targetCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and summary.get("skippedCount") == 0
        and summary.get("diagnosticCounts") == {"error": 0, "note": 0, "warning": 0},
        "pinned quantized.metal report must contain one translated artifact "
        "and no diagnostics",
    )
    _require(
        payload.get("diagnostics") == [],
        "pinned quantized.metal translation emitted diagnostics",
    )


def _require_index_range_contract(
    payload: Mapping[str, Any],
    *,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    entry_contract = ENTRY_CONTRACTS.get(entry_point)
    _require(entry_contract is not None, f"unsupported proof entry: {entry_point}")
    project = payload.get("project")
    expected = [dict(assertion) for assertion in entry_contract["indexRangeAssertions"]]
    _require(
        isinstance(project, Mapping)
        and project.get("indexRangeAssertionCount") == len(expected)
        and project.get("indexRangeAssertions") == expected,
        "project report did not preserve the exact quantized index-range contract",
    )
    return {
        "status": "configured",
        "assertions": expected,
        "assertionCount": len(expected),
        "contractKind": "explicit-host-runtime-portability-preconditions",
        "inferred": False,
        "runtimeEnforced": False,
    }


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
    if workgroup_size is None:
        return

    workgroup_size = list(workgroup_size)
    rule_path = f'project.workgroup_size_rules["{MLX_QUANTIZED_SOURCE}"]'
    expected_rule = {
        "components": [str(value) for value in workgroup_size],
        "sourcePattern": MLX_QUANTIZED_SOURCE,
        "path": rule_path,
    }
    project = payload.get("project")
    _require(
        isinstance(project, Mapping)
        and project.get("workgroupSize") is None
        and project.get("workgroupSizeRules")
        == {MLX_QUANTIZED_SOURCE: [str(value) for value in workgroup_size]}
        and project.get("workgroupSizeRuleCount") == 1
        and project.get("subgroupWidthRules") == {}
        and project.get("subgroupWidthRuleCount") == 0,
        "quantized OpenGL report did not retain the pinned workgroup rule",
    )

    execution = artifact.get("execution")
    entries = execution.get("entryPoints") if isinstance(execution, Mapping) else None
    _require(
        isinstance(execution, Mapping)
        and execution.get("sourceEntryPoints") == [entry_point]
        and execution.get("provenance")
        == {"kind": "materialized-template-rule", "path": rule_path}
        and _is_sha256_identity(execution.get("identity"))
        and isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], Mapping),
        "quantized OpenGL artifact execution metadata changed",
    )
    execution_entry = entries[0]
    _require(
        execution_entry.get("sourceEntryPoint") == entry_point
        and execution_entry.get("materializedEntryPoint") == entry_point
        and execution_entry.get("targetEntryPoint") == "main"
        and execution_entry.get("workgroupSize") == workgroup_size
        and execution_entry.get("rule") == expected_rule
        and execution_entry.get("materialization")
        == {
            "name": entry_contract["specializationName"],
            "hostName": entry_point,
            "materializedName": entry_point,
        }
        and execution_entry.get("parameters") == entry_contract["parameters"]
        and _is_sha256_identity(execution_entry.get("identity")),
        "quantized OpenGL per-entry execution contract changed",
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
        "project report must contain exactly one OpenGL artifact record",
    )
    artifact = artifacts[0]
    _require(
        artifact.get("source") == MLX_QUANTIZED_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == TARGET
        and artifact.get("status") == "translated"
        and artifact.get("sourceHash")
        == {
            "algorithm": "sha256",
            "value": PINNED_FILE_SHA256[MLX_QUANTIZED_SOURCE],
        }
        and artifact.get("provenance")
        == {"pipeline": "entry-scoped-translate", "intermediate": "crossgl"}
        and artifact.get("requiredCapabilities") == [],
        "OpenGL artifact provenance does not match pinned quantized.metal",
    )
    _require(
        artifact.get("entryPoint")
        == {
            "source": entry_point,
            "target": "main",
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
        f"generated GLSL is missing or outside the work directory: {artifact_path}",
    )
    _require(
        artifact_path.suffix == ".glsl"
        and artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": _sha256(artifact_path)}
        and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size,
        "generated GLSL identity does not match the project report",
    )
    return artifact, artifact_path


def _validate_quantize_generated_glsl(source: str) -> dict[str, Any]:
    _require(
        source.count("#version 450 core") == 1
        and source.count("void main()") == 1
        and (
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in source
        ),
        "generated GLSL compute entry-point contract is incomplete",
    )
    required_extensions = (
        "GL_ARB_gpu_shader_int64",
        "GL_KHR_shader_subgroup_basic",
        "GL_KHR_shader_subgroup_arithmetic",
        "GL_KHR_shader_subgroup_shuffle_relative",
    )
    _require(
        all(
            f"#extension {extension} : require" in source
            for extension in required_extensions
        ),
        "generated GLSL extension contract is incomplete",
    )
    _require(
        all(sentinel in source for sentinel in _GENERATED_SEMANTIC_SENTINELS.values()),
        "generated GLSL no longer preserves the selected quantization computation",
    )
    for expression, sentinels in _GENERATED_INDEX_SENTINELS.items():
        values = (sentinels,) if isinstance(sentinels, str) else sentinels
        _require(
            all(source.count(sentinel) == 1 for sentinel in values),
            f"generated GLSL index normalization changed for {expression!r}",
        )
    return {
        "status": "passed",
        "entryPoint": "main",
        "requiredExtensions": list(required_extensions),
        "semanticOperations": list(_GENERATED_SEMANTIC_SENTINELS),
        "normalizedIndexExpressions": list(INDEX_RANGE_EXPRESSIONS),
    }


def _validate_gather_generated_glsl(source: str) -> dict[str, Any]:
    required_extensions = (
        "GL_ARB_gpu_shader_int64",
        "GL_KHR_shader_subgroup_basic",
        "GL_KHR_shader_subgroup_arithmetic",
    )
    _require(
        source.count("#version 450 core") == 1
        and source.count("void main()") == 1
        and (
            "layout(local_size_x = 32, local_size_y = 2, "
            "local_size_z = 1) in;" in source
        ),
        "generated gather GLSL compute entry-point contract is incomplete",
    )
    _require(
        all(
            f"#extension {extension} : require" in source
            for extension in required_extensions
        ),
        "generated gather GLSL extension contract is incomplete",
    )
    _require(
        _GATHER_ADJUST_DEFINITION_RE.search(source) is not None
        and all(
            f"int {name} = int(0);" in source and f"{name} += int" in source
            for name in GATHER_MUTABLE_POINTER_OFFSETS
        ),
        "generated gather GLSL does not preserve mutable resource offsets",
    )
    _require(
        source.count("inout float x_thread[16]") == 4
        and "x_thread[(x_thread_base + int(i))] = x[(x_offset + i)];" in source
        and "result[row] = subgroupAdd(result[row]);" in source,
        "generated gather GLSL fixed-array or subgroup computation changed",
    )

    qdot = _GATHER_QDOT_DEFINITION_RE.search(source)
    _require(qdot is not None, "generated gather GLSL qdot contract is missing")
    qdot_body = qdot.group("body")
    _require(
        _GATHER_QDOT_CALL_RE.search(source) is not None
        and "w_byte_offset + (w_offset + i)" in qdot_body
        and "bitfieldExtract(w[int(" in qdot_body
        and "out_row" not in qdot_body
        and "in_vec_size_w" not in qdot_body
        and "simd_lid" not in qdot_body,
        "generated gather GLSL byte-address provenance changed",
    )
    _require(
        "elem_to_loc_uint32_t" in source
        and re.search(r"(?<![A-Za-z0-9_])elem_to_loc\s*\(", source) is None,
        "generated gather GLSL index helper was not fully specialized",
    )
    return {
        "status": "passed",
        "entryPoint": "main",
        "requiredExtensions": list(required_extensions),
        "workgroupSize": list(MLX_QUANTIZED_GATHER_WORKGROUP_SIZE),
        "privateArrayExtent": 16,
        "mutablePointerOffsets": list(GATHER_MUTABLE_POINTER_OFFSETS),
        "byteAddressBaseForwarding": "explicit-parameter",
        "normalizedIndexExpressions": list(GATHER_INDEX_RANGE_EXPRESSIONS),
    }


def _validate_generated_glsl(
    artifact_path: Path,
    *,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    entry_contract = ENTRY_CONTRACTS.get(entry_point)
    _require(entry_contract is not None, f"unsupported proof entry: {entry_point}")
    source = artifact_path.read_text(encoding="utf-8")
    generated_contract = entry_contract["generatedContract"]
    if generated_contract == "quantize":
        return _validate_quantize_generated_glsl(source)
    if generated_contract == "gather-qmv-fast":
        return _validate_gather_generated_glsl(source)
    raise MlxQuantizedOpenGLProofError(
        f"unsupported generated proof contract: {generated_contract}"
    )


def _compile_and_validate(
    artifact_path: Path,
    *,
    glslang: str | None,
    spirv_val: str | None,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    required: bool,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    missing = [
        name
        for name, path in (
            ("glslangValidator", glslang),
            ("spirv-val", spirv_val),
        )
        if path is None
    ]
    common = {
        "required": required,
        "compiler": "glslangValidator",
        "compilerTarget": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validatorTarget": "SPIR-V 1.3",
    }
    if missing:
        _require(
            not required,
            "OpenGL quantized proof requires these tools: " + ", ".join(missing),
        )
        return {
            **common,
            "available": False,
            "status": "not-required",
            "reason": "toolchain-unavailable",
            "missingTools": missing,
            "compiledArtifactCount": 0,
        }

    output_path = work_dir / "native" / "opengl" / f"{entry_point}.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    compile_result = _run_command(
        "compile-quantized-opengl",
        [
            str(glslang),
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
    )
    _require(
        compile_result["returncode"] == 0,
        "glslangValidator rejected the generated quantized artifact",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "glslangValidator did not emit a nonempty quantized SPIR-V module",
    )
    validation_result = _run_command(
        "validate-quantized-opengl",
        [str(spirv_val), "--target-env", "spv1.3", str(output_path)],
        log_dir=log_dir,
    )
    _require(
        validation_result["returncode"] == 0,
        "spirv-val rejected the generated quantized module",
    )
    runs = [
        {
            "name": result["name"],
            "command": result["command"],
            "returncode": result["returncode"],
            "stdout": _relpath(result["stdoutPath"], mlx_root),
            "stderr": _relpath(result["stderrPath"], mlx_root),
        }
        for result in (compile_result, validation_result)
    ]
    return {
        **common,
        "available": True,
        "status": "compiled-and-validated",
        "artifact": _relpath(artifact_path, mlx_root),
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compiledArtifactHash": {
            "algorithm": "sha256",
            "value": _sha256(output_path),
        },
        "compiledArtifactCount": 1,
        "runs": runs,
    }


def run_proof(
    mlx_root: Path,
    work_dir: Path,
    *,
    require_opengl_toolchain: bool = False,
    clean: bool = True,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    _require(entry_point in ENTRY_CONTRACTS, f"unsupported proof entry: {entry_point}")
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
    index_ranges = _require_index_range_contract(payload, entry_point=entry_point)
    artifact, artifact_path = _translated_artifact(
        payload,
        mlx_root=root,
        work_dir=resolved_work_dir,
        entry_point=entry_point,
    )
    generated_checks = _validate_generated_glsl(
        artifact_path,
        entry_point=entry_point,
    )
    toolchain = _compile_and_validate(
        artifact_path,
        glslang=shutil.which("glslangValidator"),
        spirv_val=shutil.which("spirv-val"),
        mlx_root=root,
        work_dir=resolved_work_dir,
        log_dir=log_dir,
        required=require_opengl_toolchain,
        entry_point=entry_point,
    )

    summary = {
        "schema_version": 1,
        "kind": "crosstl-mlx-quantized-opengl-toolchain-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "translation": {
                "source": MLX_QUANTIZED_SOURCE,
                "selectedEntryPoint": entry_point,
                "sourceBackend": "metal",
                "sourceOverride": "metal",
                "includeDirectories": ["."],
                "target": TARGET,
                "projectTranslationApi": "crosstl.project.translate_project",
                "materializationLimits": {
                    "maxTemplateSpecializations": TEMPLATE_SPECIALIZATION_LIMIT,
                    "maxTemplateMaterializationWork": MATERIALIZATION_WORK_LIMIT,
                },
                "indexRangeContract": index_ranges,
            },
            "toolchain": {
                "compiler": "glslangValidator",
                "validator": "spirv-val",
                "required": require_opengl_toolchain,
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
            "nativeCompilation": toolchain["status"] == "compiled-and-validated",
            "spirvValidation": toolchain["status"] == "compiled-and-validated",
            **NON_RUNTIME_CLAIMS,
        },
        "provenance": provenance,
        "translation": {
            "status": "passed",
            "report": _relpath(report_path, root),
            "artifact": _relpath(artifact_path, root),
            "artifactHash": artifact["generatedHash"],
            "artifactSizeBytes": artifact["generatedSizeBytes"],
            "entryPoint": artifact["entryPoint"],
            "requiredCapabilities": list(artifact["requiredCapabilities"]),
            "templateMaterialization": artifact["templateMaterialization"],
            "indexRangeContract": index_ranges,
            "generatedChecks": generated_checks,
        },
        "toolchain": toolchain,
        "runtime": {
            "status": "not-attempted",
            "reason": "compile-only proof; no OpenGL dispatch or MLX runtime wiring",
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
    *,
    required: bool,
    error: str,
    entry_point: str = MLX_QUANTIZED_ENTRY_POINT,
) -> dict[str, Any]:
    entry_contract = ENTRY_CONTRACTS.get(
        entry_point, ENTRY_CONTRACTS[MLX_QUANTIZED_ENTRY_POINT]
    )
    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-quantized-opengl-toolchain-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "translation": {
                "source": MLX_QUANTIZED_SOURCE,
                "selectedEntryPoint": entry_point,
                "target": TARGET,
                "indexRangeAssertions": [
                    dict(assertion)
                    for assertion in entry_contract["indexRangeAssertions"]
                ],
            },
            "toolchain": {
                "compiler": "glslangValidator",
                "validator": "spirv-val",
                "required": required,
            },
            "runtime": {"executionAttempted": False, "mlxTestsRun": False},
            "numerical": {"comparisonAttempted": False, "parityClaimed": False},
        },
        "claims": {
            "projectTranslation": False,
            "nativeCompilation": False,
            "spirvValidation": False,
            **NON_RUNTIME_CLAIMS,
        },
        "status": "failed",
        "error": error,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove a pinned MLX quantized project entry through OpenGL "
            "translation and optional required native toolchain validation."
        )
    )
    parser.add_argument("--mlx-root", required=True, help="Path to the MLX checkout")
    parser.add_argument(
        "--entry-point",
        choices=tuple(ENTRY_CONTRACTS),
        default=MLX_QUANTIZED_ENTRY_POINT,
        help="Pinned quantized Metal entry point to translate.",
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "Generated report/artifact directory inside the MLX checkout. "
            f"Defaults to <mlx-root>/{DEFAULT_WORK_DIR}."
        ),
    )
    parser.add_argument(
        "--require-opengl-toolchain",
        action="store_true",
        help=(
            "Require glslangValidator and spirv-val instead of accepting a "
            "translation-only proof."
        ),
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
    except MlxQuantizedOpenGLProofError as exc:
        print(f"MLX quantized OpenGL proof failed: {exc}", file=sys.stderr)
        return 1

    summary_path = work_dir / SUMMARY_FILENAME
    try:
        run_proof(
            mlx_root,
            work_dir,
            require_opengl_toolchain=args.require_opengl_toolchain,
            clean=not args.no_clean,
            entry_point=args.entry_point,
        )
    except MlxQuantizedOpenGLProofError as exc:
        _write_json(
            summary_path,
            _failure_summary(
                required=args.require_opengl_toolchain,
                error=str(exc),
                entry_point=args.entry_point,
            ),
        )
        print(f"MLX quantized OpenGL proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    print(f"MLX quantized OpenGL proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
