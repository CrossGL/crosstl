#!/usr/bin/env python3
"""Prove pinned MLX RMSNorm specialization translation contracts."""

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

from crosstl.project import (
    ProjectConfig,
    reflect_target_host_interface,
    translate_project,
)
from crosstl.project.directx_toolchain import (
    dxc_compiler_arguments_for_source,
    dxc_profile_for_source,
)

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_RMS_NORM_SHA256 = "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee"
RMS_NORM_FUNCTION_CONSTANT_NAME = "has_w"
RMS_NORM_FUNCTION_CONSTANT_ID = 20
RMS_NORM_HOST_ENTRY_POINTS = (
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
RMS_NORM_EXPECTED_ENTRY_POINT_COUNT = len(RMS_NORM_HOST_ENTRY_POINTS)
RMS_NORM_LOOPED_ENTRY_POINT_COUNT = sum(
    "_looped" in entry_point for entry_point in RMS_NORM_HOST_ENTRY_POINTS
)
RMS_NORM_SUBGROUP_WIDTH = 32
RMS_NORM_DIRECTX_PROFILE = "cs_6_6"
RMS_NORM_RUNTIME_BLOCKERS = (
    "https://github.com/CrossGL/crosstl/issues/1462",
    "https://github.com/CrossGL/crosstl/issues/1735",
)
RMS_NORM_DIRECTX_VARIANTS = (
    {
        "name": "has_w_false_by_name_workgroup_32",
        "selector": RMS_NORM_FUNCTION_CONSTANT_NAME,
        "selectorKind": "name",
        "value": False,
        "workgroupSize": [32, 1, 1],
    },
    {
        "name": "has_w_true_by_id_workgroup_64",
        "selector": str(RMS_NORM_FUNCTION_CONSTANT_ID),
        "selectorKind": "id",
        "value": True,
        "workgroupSize": [64, 1, 1],
    },
)
RMS_NORM_OPENGL_VARIANTS = (
    {
        "name": "workgroup_32",
        "workgroupSize": [32, 1, 1],
    },
    {
        "name": "workgroup_64",
        "workgroupSize": [64, 1, 1],
    },
)
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/rms-norm-specialization"


class MlxRmsNormProofError(RuntimeError):
    """Raised when the pinned RMSNorm proof contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxRmsNormProofError(message)


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
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "command": list(command),
        "returncode": returncode,
        "stdoutPath": stdout_path,
        "stderrPath": stderr_path,
    }


def _verify_mlx_checkout(
    mlx_root: Path,
    *,
    log_dir: Path,
) -> dict[str, Any]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    source_path = mlx_root / MLX_RMS_NORM_SOURCE
    _require(source_path.is_file(), f"pinned MLX source is missing: {source_path}")

    revision_result = _run_command(
        "mlx-rmsnorm-revision",
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
        log_dir=log_dir,
    )
    _require(
        revision_result["returncode"] == 0,
        "could not resolve the MLX checkout revision",
    )
    revision = revision_result["stdoutPath"].read_text(encoding="utf-8").strip()
    _require(
        revision == MLX_COMMIT,
        f"MLX checkout must be pinned to {MLX_COMMIT}; found {revision}",
    )

    source_sha256 = _sha256(source_path)
    _require(
        source_sha256 == MLX_RMS_NORM_SHA256,
        "pinned MLX rms_norm.metal SHA-256 mismatch: "
        f"expected {MLX_RMS_NORM_SHA256}, found {source_sha256}",
    )
    source_subgroup_contract = _verify_source_subgroup_contract(
        source_path.read_text(encoding="utf-8")
    )
    return {
        "name": "pinned-source-identity",
        "status": "passed",
        "repository": MLX_REPOSITORY,
        "commit": revision,
        "source": MLX_RMS_NORM_SOURCE,
        "sourceHash": {
            "algorithm": "sha256",
            "value": source_sha256,
        },
        "sourceSubgroupContract": source_subgroup_contract,
    }


def _verify_source_subgroup_contract(source: str) -> dict[str, Any]:
    width_declarations = [
        int(width)
        for width in re.findall(
            r"\bconstexpr\s+int\s+SIMD_SIZE\s*=\s*(\d+)\s*;",
            source,
        )
    ]
    lane_builtin_count = len(
        re.findall(r"\[\[\s*thread_index_in_simdgroup\s*\]\]", source)
    )
    group_builtin_count = len(
        re.findall(r"\[\[\s*simdgroup_index_in_threadgroup\s*\]\]", source)
    )
    simd_sum_call_count = len(re.findall(r"\bsimd_sum\s*\(", source))
    _require(
        width_declarations == [RMS_NORM_SUBGROUP_WIDTH] * 4,
        "pinned RMSNorm source must declare exactly four SIMD_SIZE = 32 contracts",
    )
    _require(
        lane_builtin_count == group_builtin_count == len(width_declarations),
        "pinned RMSNorm source must retain lane and simdgroup index builtins on "
        "all four kernels",
    )
    _require(
        simd_sum_call_count == 12,
        "pinned RMSNorm source must retain all 12 simd_sum reductions",
    )
    return {
        "status": "passed",
        "requiredSubgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
        "widthDeclaration": "constexpr int SIMD_SIZE = 32;",
        "widthDeclarationCount": len(width_declarations),
        "laneBuiltin": "thread_index_in_simdgroup",
        "laneBuiltinCount": lane_builtin_count,
        "groupBuiltin": "simdgroup_index_in_threadgroup",
        "groupBuiltinCount": group_builtin_count,
        "reductionBuiltin": "simd_sum",
        "reductionCallCount": simd_sum_call_count,
        "requiredDirectXEnforcement": {
            "attribute": f"WaveSize({RMS_NORM_SUBGROUP_WIDTH})",
            "minimumShaderModel": "6.6",
            "profile": RMS_NORM_DIRECTX_PROFILE,
        },
    }


def _project_config(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
) -> ProjectConfig:
    _require(
        target in {"directx", "opengl"},
        f"unsupported RMSNorm proof target: {target}",
    )
    common: dict[str, Any] = {
        "root": mlx_root,
        "source_roots": (MLX_METAL_KERNEL_ROOT,),
        "include_patterns": (MLX_RMS_NORM_SOURCE,),
        "targets": (target,),
        "output_dir": _relpath(work_dir / "artifacts", mlx_root),
        "source_overrides": {"**/*.metal": "metal"},
        "include_dirs": (".",),
    }
    if target == "directx":
        configured_variants = RMS_NORM_DIRECTX_VARIANTS
        variant_specializations = {
            variant["name"]: {variant["selector"]: variant["value"]}
            for variant in RMS_NORM_DIRECTX_VARIANTS
        }
        common["variant_specialization_constants"] = variant_specializations
        common["subgroup_width_rules"] = {
            MLX_RMS_NORM_SOURCE: str(RMS_NORM_SUBGROUP_WIDTH)
        }
    else:
        configured_variants = RMS_NORM_OPENGL_VARIANTS
    variants = {
        variant["name"]: {"workgroup_size": variant["workgroupSize"]}
        for variant in configured_variants
    }
    common.update(
        {
            "variants": variants,
            "selected_variants": tuple(variants),
        }
    )
    return ProjectConfig(**common)


def _translate_report(
    config: ProjectConfig,
    *,
    target: str,
    report_path: Path,
) -> dict[str, Any]:
    try:
        report = translate_project(config, validate=True)
        payload = report.to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxRmsNormProofError(
            f"{target} project translation raised {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _validate_report(
    payload: Mapping[str, Any],
    *,
    target: str,
    expected_artifact_count: int,
) -> list[Mapping[str, Any]]:
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        f"{target} proof did not produce a project portability report",
    )
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), f"{target} report summary is missing")
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == expected_artifact_count
        and summary.get("translatedCount") == expected_artifact_count
        and summary.get("failedCount") == 0,
        f"{target} report does not contain the expected translated artifact set",
    )
    diagnostics = payload.get("diagnostics")
    _require(isinstance(diagnostics, list), f"{target} diagnostics are missing")
    unexpected_diagnostics = [
        diagnostic
        for diagnostic in diagnostics
        if not (
            isinstance(diagnostic, Mapping)
            and diagnostic.get("severity") == "warning"
            and diagnostic.get("code") == "project.validate.toolchain-unavailable"
            and diagnostic.get("target") == target
        )
    ]
    _require(
        not unexpected_diagnostics,
        f"{target} project translation emitted unexpected diagnostics",
    )
    _require(
        summary.get("diagnosticCounts", {}).get("error") == 0,
        f"{target} project translation reported an error diagnostic",
    )
    validation = payload.get("validation")
    _require(isinstance(validation, Mapping), f"{target} validation is missing")
    validation_summary = validation.get("summary")
    _require(
        isinstance(validation_summary, Mapping)
        and validation_summary.get("artifactCount") == expected_artifact_count
        and validation_summary.get("okCount") == expected_artifact_count
        and validation_summary.get("failedCount") == 0,
        f"{target} report artifact validation did not pass",
    )
    artifacts = payload.get("artifacts")
    _require(isinstance(artifacts, list), f"{target} report artifacts are missing")
    _require(
        all(isinstance(artifact, Mapping) for artifact in artifacts),
        f"{target} report artifacts must be JSON objects",
    )
    return artifacts


def _validate_project_variants(
    payload: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> None:
    project = payload.get("project")
    _require(isinstance(project, Mapping), f"{target} project metadata is missing")
    variant_names = [str(variant["name"]) for variant in variants]
    expected_workgroup_sizes = {
        str(variant["name"]): list(variant["workgroupSize"]) for variant in variants
    }
    _require(
        project.get("targets") == [target]
        and project.get("variantCount") == len(variants)
        and project.get("selectedVariants") == variant_names
        and project.get("variants") == {name: {} for name in variant_names}
        and project.get("variantWorkgroupSizes") == expected_workgroup_sizes
        and project.get("workgroupSize") is None
        and project.get("workgroupSizeRuleCount") == 0
        and project.get("workgroupSizeRules") == {},
        f"{target} report did not retain the exact workgroup variants",
    )
    expected_specializations = (
        {
            str(variant["name"]): {
                str(variant["selector"]): variant["value"],
            }
            for variant in variants
        }
        if target == "directx"
        else {}
    )
    _require(
        project.get("specializationConstants") == {}
        and project.get("specializationConstantCount") == 0
        and project.get("variantSpecializationConstants") == expected_specializations,
        f"{target} report specialization-variant configuration changed",
    )
    expected_subgroup_width_rules = (
        {MLX_RMS_NORM_SOURCE: str(RMS_NORM_SUBGROUP_WIDTH)}
        if target == "directx"
        else {}
    )
    _require(
        project.get("subgroupWidthRules") == expected_subgroup_width_rules
        and project.get("subgroupWidthRuleCount") == len(expected_subgroup_width_rules),
        f"{target} report subgroup-width configuration changed",
    )


def _is_sha256_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("algorithm") == "sha256"
        and isinstance(value.get("value"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["value"]) is not None
    )


def _artifact_path(artifact: Mapping[str, Any], mlx_root: Path) -> Path:
    value = artifact.get("path")
    _require(isinstance(value, str) and value, "artifact path is missing")
    path = (mlx_root / value).resolve()
    _require(
        _is_relative_to(path, mlx_root.resolve()),
        f"artifact path resolves outside the MLX checkout: {value}",
    )
    _require(path.is_file(), f"translated artifact is missing: {path}")
    return path


def _validate_common_artifact(
    artifact: Mapping[str, Any],
    *,
    target: str,
    mlx_root: Path,
) -> tuple[Path, str]:
    expected_pipeline = (
        "single-file-translate" if target == "directx" else "entry-scoped-translate"
    )
    _require(
        artifact.get("source") == MLX_RMS_NORM_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == target
        and artifact.get("status") == "translated",
        f"{target} artifact identity or status is incorrect",
    )
    _require(
        artifact.get("sourceHash")
        == {"algorithm": "sha256", "value": MLX_RMS_NORM_SHA256},
        f"{target} report did not retain the pinned source hash",
    )
    _require(
        artifact.get("provenance")
        == {"pipeline": expected_pipeline, "intermediate": "crossgl"},
        f"{target} report did not retain Metal-to-CrossGL project provenance",
    )
    path = _artifact_path(artifact, mlx_root)
    generated_hash = artifact.get("generatedHash")
    _require(
        isinstance(generated_hash, Mapping)
        and generated_hash.get("algorithm") == "sha256"
        and generated_hash.get("value") == _sha256(path),
        f"{target} generated artifact hash is missing or stale",
    )
    generated_size = artifact.get("generatedSizeBytes")
    _require(
        isinstance(generated_size, int)
        and generated_size > 0
        and generated_size == path.stat().st_size,
        f"{target} generated artifact size is missing or stale",
    )
    return path, path.read_text(encoding="utf-8")


def _validate_specialization_materialization(
    artifact: Mapping[str, Any],
    *,
    target: str,
    deferred: bool,
) -> None:
    materialization = artifact.get("specializationMaterialization")
    expected = {
        "status": "deferred" if deferred else "concrete",
        "mode": "deferred" if deferred else "concrete-crossgl-variant",
        "targetSupportsDeferredSpecialization": deferred,
        "constantCount": 1,
        "requiredCount": 1,
        "overriddenCount": 0 if deferred else 1,
        "concreteCount": 0 if deferred else 1,
        "source": "shared-crossgl-specialization",
    }
    _require(
        materialization == expected,
        f"{target} specialization materialization contract does not match",
    )


def _host_materializations(
    artifact: Mapping[str, Any],
    *,
    target: str,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    materialization = artifact.get("templateMaterialization")
    specializations = (
        materialization.get("specializations")
        if isinstance(materialization, Mapping)
        else None
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("specializationCount")
        == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        and isinstance(specializations, list)
        and len(specializations) == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        and materialization.get("unsupported") == [],
        f"{target} RMSNorm artifact did not retain exactly 12 materializations",
    )
    host_materializations: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in specializations:
        _require(
            isinstance(record, Mapping),
            f"{target} RMSNorm materialization records must be objects",
        )
        host_name = record.get("hostName")
        materialized_name = record.get("materializedName")
        _require(
            isinstance(host_name, str)
            and bool(host_name)
            and isinstance(materialized_name, str)
            and bool(materialized_name)
            and isinstance(record.get("name"), str)
            and bool(record.get("name")),
            f"{target} RMSNorm host materialization identity is incomplete",
        )
        identity = (host_name, materialized_name)
        _require(
            identity not in host_materializations,
            f"{target} RMSNorm host materialization identities must be unique",
        )
        host_materializations[identity] = record
    _require(
        set(host_materializations)
        == {(entry_point, entry_point) for entry_point in RMS_NORM_HOST_ENTRY_POINTS},
        f"{target} RMSNorm host-named compute entry set changed",
    )
    return host_materializations


def _execution_entries(
    artifact: Mapping[str, Any],
    host_materializations: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    target: str,
    variant_name: str,
    workgroup_size: Sequence[int],
    expected_source_entry_points: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]]]:
    execution = artifact.get("execution")
    entries = execution.get("entryPoints") if isinstance(execution, Mapping) else None
    expected_size = list(workgroup_size)
    requires_exact_subgroup_width = target.lower() == "directx"
    subgroup_rule_path = f'project.subgroup_width_rules["{MLX_RMS_NORM_SOURCE}"]'
    expected_subgroup_rule = {
        "expression": str(RMS_NORM_SUBGROUP_WIDTH),
        "sourcePattern": MLX_RMS_NORM_SOURCE,
        "path": subgroup_rule_path,
    }
    expected_execution_keys = {
        "sourceEntryPoints",
        "entryPoints",
        "provenance",
        "identity",
    }
    if requires_exact_subgroup_width:
        expected_execution_keys.update(
            {"subgroupWidthProvenance", "subgroupWidthEnforcement"}
        )
    _require(
        isinstance(execution, Mapping)
        and set(execution) == expected_execution_keys
        and execution.get("provenance")
        == {
            "kind": "project-variant",
            "path": f"project.variants.{variant_name}.workgroup_size",
            "variant": variant_name,
        }
        and _is_sha256_identity(execution.get("identity"))
        and execution.get("sourceEntryPoints") == list(expected_source_entry_points)
        and isinstance(entries, list)
        and len(entries) == len(expected_source_entry_points),
        f"{target} variant {variant_name} execution identity or entry set changed",
    )
    entries_by_target: dict[str, Mapping[str, Any]] = {}
    source_identities: set[tuple[str, str]] = set()
    for entry in entries:
        _require(
            isinstance(entry, Mapping),
            f"{target} variant {variant_name} execution entries must be objects",
        )
        source_entry = entry.get("sourceEntryPoint")
        materialized_entry = entry.get("materializedEntryPoint")
        target_entry = entry.get("targetEntryPoint")
        _require(
            isinstance(source_entry, str)
            and bool(source_entry)
            and isinstance(materialized_entry, str)
            and bool(materialized_entry)
            and isinstance(target_entry, str)
            and bool(target_entry),
            f"{target} variant {variant_name} execution entry identity is incomplete",
        )
        source_identity = (source_entry, materialized_entry)
        host_record = host_materializations.get(source_identity)
        _require(
            host_record is not None
            and source_identity not in source_identities
            and target_entry not in entries_by_target,
            f"{target} variant {variant_name} execution identities are not unique",
        )
        expected_materialization = {
            "hostName": source_entry,
            "materializedName": materialized_entry,
            "name": host_record.get("name"),
        }
        _require(
            entry.get("materialization") == expected_materialization
            and entry.get("parameters") == host_record.get("parameters")
            and entry.get("parameterSources") == host_record.get("parameterSources")
            and entry.get("workgroupSize") == expected_size
            and _is_sha256_identity(entry.get("identity"))
            and "rule" not in entry,
            f"{target} variant {variant_name} execution contract changed for "
            f"{source_entry}",
        )
        if requires_exact_subgroup_width:
            _require(
                entry.get("subgroupWidth") == RMS_NORM_SUBGROUP_WIDTH
                and entry.get("subgroupWidthRule") == expected_subgroup_rule,
                f"DirectX variant {variant_name} subgroup-width contract changed "
                f"for {source_entry}",
            )
        else:
            _require(
                "subgroupWidth" not in entry and "subgroupWidthRule" not in entry,
                f"OpenGL variant {variant_name} must not report an unenforceable "
                f"subgroup-width contract for {source_entry}",
            )
        source_identities.add(source_identity)
        entries_by_target[target_entry] = entry
    _require(
        {identity[0] for identity in source_identities}
        == set(expected_source_entry_points),
        f"{target} variant {variant_name} execution source identities changed",
    )
    if requires_exact_subgroup_width:
        expected_subgroup_provenance = {
            "kind": "materialized-template-rule",
            "path": subgroup_rule_path,
        }
        expected_enforcement = {
            "mechanism": "hlsl-wave-size-attribute",
            "minimumShaderModel": "6.6",
            "entryProfiles": [
                {"entryPoint": entry_point, "profile": RMS_NORM_DIRECTX_PROFILE}
                for entry_point in entries_by_target
            ],
        }
        _require(
            execution.get("subgroupWidthProvenance") == expected_subgroup_provenance
            and execution.get("subgroupWidthEnforcement") == expected_enforcement,
            f"DirectX variant {variant_name} subgroup-width provenance or "
            "enforcement metadata changed",
        )
    else:
        _require(
            "subgroupWidthProvenance" not in execution
            and "subgroupWidthEnforcement" not in execution,
            f"OpenGL variant {variant_name} must not report DirectX subgroup-width "
            "enforcement metadata",
        )
    return execution, entries_by_target


def _directx_execution_evidence(
    artifact: Mapping[str, Any],
    generated: str,
    variant: Mapping[str, Any],
) -> dict[str, Any]:
    variant_name = str(variant["name"])
    workgroup_size = list(variant["workgroupSize"])
    host_materializations = _host_materializations(
        artifact,
        target=f"DirectX variant {variant_name}",
    )
    execution, report_entries_by_target = _execution_entries(
        artifact,
        host_materializations,
        target="DirectX",
        variant_name=variant_name,
        workgroup_size=workgroup_size,
        expected_source_entry_points=RMS_NORM_HOST_ENTRY_POINTS,
    )
    generated_entry_pattern = re.compile(
        r"(?m)^[ \t]*\[numthreads\(\s*(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*"
        r"(?P<z>\d+)\s*\)\][ \t]*\r?\n"
        r"[ \t]*\[WaveSize\(\s*(?P<wave>\d+)\s*\)\][ \t]*\r?\n"
        r"[ \t]*void[ \t]+(?P<target_entry>[A-Za-z_]\w*)[ \t]*\("
    )
    generated_entries_by_target: dict[str, dict[str, Any]] = {}
    for match in generated_entry_pattern.finditer(generated):
        target_entry = match.group("target_entry")
        _require(
            target_entry not in generated_entries_by_target,
            f"DirectX variant {variant_name} generated duplicate target entries",
        )
        generated_entries_by_target[target_entry] = {
            "workgroupSize": [
                int(match.group(component)) for component in ("x", "y", "z")
            ],
            "subgroupWidth": int(match.group("wave")),
        }
    all_numthreads_sizes = [
        [int(component) for component in match]
        for match in re.findall(
            r"\[\s*numthreads\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]",
            generated,
        )
    ]
    all_wave_size_widths = [
        int(width)
        for width in re.findall(
            r"\[\s*WaveSize\s*\(\s*(\d+)\s*\)\s*\]",
            generated,
        )
    ]
    all_wave_size_attribute_count = len(re.findall(r"\[\s*WaveSize\s*\(", generated))
    _require(
        len(generated_entries_by_target) == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        and set(generated_entries_by_target) == set(report_entries_by_target),
        f"DirectX variant {variant_name} generated compute entry set changed",
    )
    _require(
        len(all_numthreads_sizes) == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        and all(size == workgroup_size for size in all_numthreads_sizes),
        f"DirectX variant {variant_name} generated an extra numthreads contract",
    )
    _require(
        all_wave_size_attribute_count
        == len(all_wave_size_widths)
        == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        and all(width == RMS_NORM_SUBGROUP_WIDTH for width in all_wave_size_widths),
        f"DirectX variant {variant_name} must emit exactly one WaveSize(32) "
        "contract per generated entry",
    )
    for target_entry, report_entry in report_entries_by_target.items():
        _require(
            generated_entries_by_target[target_entry]["workgroupSize"]
            == report_entry.get("workgroupSize")
            == workgroup_size,
            f"DirectX variant {variant_name} numthreads contract changed for "
            f"{target_entry}",
        )
        _require(
            generated_entries_by_target[target_entry]["subgroupWidth"]
            == report_entry.get("subgroupWidth")
            == RMS_NORM_SUBGROUP_WIDTH,
            f"DirectX variant {variant_name} WaveSize contract changed for "
            f"{target_entry}",
        )
    projection_pattern = re.compile(
        r"\buint\s+lsize\s*=\s*uint\s*\(\s*\(\s*uint3\s*\(\s*"
        r"(?P<x>\d+)\s*,\s*(?P<y>\d+)\s*,\s*(?P<z>\d+)\s*\)\s*\)"
        r"\.x\s*\)\s*;"
    )
    projected_sizes = [
        [int(match.group(component)) for component in ("x", "y", "z")]
        for match in projection_pattern.finditer(generated)
    ]
    lsize_assignments = re.findall(r"\buint\s+lsize\s*=\s*[^;]+;", generated)
    _require(
        len(lsize_assignments)
        == len(projected_sizes)
        == RMS_NORM_LOOPED_ENTRY_POINT_COUNT
        and all(size == workgroup_size for size in projected_sizes),
        f"DirectX variant {variant_name} consumed workgroup-size projection changed",
    )
    return {
        "hostNamedMaterializationCount": len(host_materializations),
        "executionIdentity": dict(execution["identity"]),
        "executionEntryCount": len(report_entries_by_target),
        "generatedNumthreadsContractCount": len(generated_entries_by_target),
        "sourceRequiredSubgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
        "subgroupWidthRule": dict(
            next(iter(report_entries_by_target.values()))["subgroupWidthRule"]
        ),
        "subgroupWidthProvenance": dict(execution["subgroupWidthProvenance"]),
        "subgroupWidthEnforcement": dict(execution["subgroupWidthEnforcement"]),
        "generatedWaveSizeContractCount": len(all_wave_size_widths),
        "consumedWorkgroupProjectionCount": len(projected_sizes),
        "sourceEntryPoints": list(RMS_NORM_HOST_ENTRY_POINTS),
    }


def _representative_directx_entry_point(
    artifact_path: Path,
    workgroup_size: Sequence[int] | None = None,
) -> str:
    reflection = reflect_target_host_interface(artifact_path, target="directx")
    entries = reflection.get("entryPoints")
    _require(
        isinstance(entries, list),
        "DirectX RMSNorm reflection did not report compute entry points",
    )
    compute_entries = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("stage") == "compute"
        and isinstance(entry.get("name"), str)
        and entry.get("name")
    ]
    _require(
        compute_entries,
        "DirectX RMSNorm artifact has no reflected compute entry point",
    )
    if workgroup_size is not None:
        _require(
            compute_entries[0].get("executionConfig")
            == {"numthreads": list(workgroup_size)},
            "DirectX RMSNorm representative reflection workgroup size changed",
        )
    return str(compute_entries[0]["name"])


def _directx_variant_evidence(
    artifact: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    mlx_root: Path,
) -> dict[str, Any]:
    variant_name = str(variant["name"])
    workgroup_size = list(variant["workgroupSize"])
    _require(
        artifact.get("variant") == variant_name,
        f"DirectX report artifact is missing variant {variant_name}",
    )
    artifact_path, generated = _validate_common_artifact(
        artifact,
        target="directx",
        mlx_root=mlx_root,
    )
    constants = artifact.get("specializationConstants")
    _require(
        isinstance(constants, list) and len(constants) == 1,
        f"DirectX variant {variant_name} must report one function constant",
    )
    constant = constants[0]
    _require(
        isinstance(constant, Mapping)
        and constant.get("name") == RMS_NORM_FUNCTION_CONSTANT_NAME
        and constant.get("id") == RMS_NORM_FUNCTION_CONSTANT_ID
        and constant.get("required") is True
        and constant.get("overridden") is True
        and constant.get("deferred") is False
        and constant.get("concreteValue") is variant["value"],
        f"DirectX variant {variant_name} function-constant record is incorrect",
    )
    expected_provenance = {
        "kind": "project-variant",
        "path": (
            f"project.variants.{variant_name}.specialization_constants."
            f"{variant['selector']}"
        ),
        "selector": variant["selector"],
        "selectorKind": variant["selectorKind"],
        "variant": variant_name,
    }
    _require(
        constant.get("valueProvenance") == expected_provenance,
        f"DirectX variant {variant_name} selector provenance is incorrect",
    )
    _validate_specialization_materialization(
        artifact,
        target=f"DirectX variant {variant_name}",
        deferred=False,
    )
    execution_evidence = _directx_execution_evidence(
        artifact,
        generated,
        variant,
    )

    literal = "true" if variant["value"] else "false"
    static_constants = re.findall(
        rf"\bstatic\s+const\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}"
        r"\s*=\s*(true|false)\s*;",
        generated,
    )
    _require(
        static_constants == [literal],
        f"DirectX variant {variant_name} did not emit the expected static const",
    )
    entry_point = _representative_directx_entry_point(
        artifact_path,
        workgroup_size,
    )
    _require(
        re.search(
            rf"\buniform\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}\b",
            generated,
        )
        is None,
        f"DirectX variant {variant_name} lowered has_w as a uniform",
    )
    return {
        "name": variant_name,
        "selector": variant["selector"],
        "selectorKind": variant["selectorKind"],
        "value": variant["value"],
        "workgroupSize": workgroup_size,
        "valueProvenance": expected_provenance,
        "specializationMaterialization": dict(
            artifact["specializationMaterialization"]
        ),
        "execution": execution_evidence,
        "generatedStaticConst": f"static const bool has_w = {literal};",
        "representativeEntryPoint": entry_point,
        "artifact": _relpath(artifact_path, mlx_root),
    }


def _compile_directx_variants(
    artifacts_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "status": "not-required",
            "platform": sys.platform,
            "compiler": "dxc",
            "profile": RMS_NORM_DIRECTX_PROFILE,
            "minimumShaderModel": "6.6",
            "subgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
            "subgroupWidthEnforcement": f"WaveSize({RMS_NORM_SUBGROUP_WIDTH})",
            "compiledArtifactCount": 0,
            "runs": [],
        }
    _require(
        sys.platform == "win32",
        "--require-directx-toolchain is reserved for the Windows proof job",
    )
    dxc = shutil.which("dxc")
    _require(dxc is not None, "DirectX proof requires dxc on Windows")

    output_dir = work_dir / "native" / "directx"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for variant in RMS_NORM_DIRECTX_VARIANTS:
        variant_name = str(variant["name"])
        artifact_path = _artifact_path(artifacts_by_variant[variant_name], mlx_root)
        generated = artifact_path.read_text(encoding="utf-8")
        _require(
            generated.count(f"[WaveSize({RMS_NORM_SUBGROUP_WIDTH})]")
            == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
            and len(re.findall(r"\[\s*WaveSize\s*\(", generated))
            == RMS_NORM_EXPECTED_ENTRY_POINT_COUNT,
            f"DirectX RMSNorm variant {variant_name} must contain exactly one "
            "WaveSize(32) contract per generated entry before DXC validation",
        )
        profile = dxc_profile_for_source(RMS_NORM_DIRECTX_PROFILE, generated)
        _require(
            profile == RMS_NORM_DIRECTX_PROFILE,
            f"DirectX RMSNorm variant {variant_name} must compile with "
            f"{RMS_NORM_DIRECTX_PROFILE}",
        )
        compiler_arguments = dxc_compiler_arguments_for_source(generated)
        entry_point = _representative_directx_entry_point(artifact_path)
        output_path = output_dir / f"{variant_name}.dxil"
        output_path.unlink(missing_ok=True)
        result = _run_command(
            f"compile-rmsnorm-directx-{variant_name}",
            [
                dxc,
                "-WX",
                "-T",
                profile,
                *compiler_arguments,
                "-E",
                entry_point,
                str(artifact_path),
                "-Fo",
                str(output_path),
            ],
            log_dir=log_dir,
        )
        _require(
            result["returncode"] == 0,
            f"DXC failed for DirectX RMSNorm variant {variant_name}",
        )
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            f"DXC did not emit DXIL for DirectX RMSNorm variant {variant_name}",
        )
        runs.append(
            {
                "variant": variant_name,
                "workgroupSize": list(variant["workgroupSize"]),
                "status": "compiled",
                "entryPoint": entry_point,
                "profile": profile,
                "compilerArguments": list(compiler_arguments),
                "minimumShaderModel": "6.6",
                "subgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
                "subgroupWidthEnforcement": f"WaveSize({RMS_NORM_SUBGROUP_WIDTH})",
                "artifact": _relpath(artifact_path, mlx_root),
                "compiledArtifact": _relpath(output_path, mlx_root),
                "stdout": _relpath(result["stdoutPath"], mlx_root),
                "stderr": _relpath(result["stderrPath"], mlx_root),
            }
        )
    profiles = {str(run["profile"]) for run in runs}
    compiler_argument_sets = {tuple(run.get("compilerArguments", ())) for run in runs}
    result = {
        "required": True,
        "status": "compiled",
        "platform": sys.platform,
        "compiler": "dxc",
        "profile": RMS_NORM_DIRECTX_PROFILE,
        "minimumShaderModel": "6.6",
        "subgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
        "subgroupWidthEnforcement": f"WaveSize({RMS_NORM_SUBGROUP_WIDTH})",
        "compiledArtifactCount": len(runs),
        "runs": runs,
    }
    _require(
        profiles == {RMS_NORM_DIRECTX_PROFILE},
        "DirectX RMSNorm variants did not use the required cs_6_6 profile",
    )
    if len(compiler_argument_sets) == 1:
        compiler_arguments = next(iter(compiler_argument_sets))
        result["compilerArguments"] = list(compiler_arguments)
    return result


def _check_directx(
    mlx_root: Path,
    work_dir: Path,
    report_dir: Path,
    log_dir: Path,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    report_path = report_dir / "rms-norm-directx-variants.json"
    payload = _translate_report(
        _project_config(mlx_root, work_dir, target="directx"),
        target="directx",
        report_path=report_path,
    )
    artifacts = _validate_report(
        payload,
        target="directx",
        expected_artifact_count=len(RMS_NORM_DIRECTX_VARIANTS),
    )
    _validate_project_variants(
        payload,
        RMS_NORM_DIRECTX_VARIANTS,
        target="directx",
    )
    artifacts_by_variant = {
        str(artifact.get("variant")): artifact for artifact in artifacts
    }
    expected_variants = {str(variant["name"]) for variant in RMS_NORM_DIRECTX_VARIANTS}
    _require(
        set(artifacts_by_variant) == expected_variants,
        "DirectX report did not materialize both configured RMSNorm variants",
    )
    variant_evidence = [
        _directx_variant_evidence(
            artifacts_by_variant[str(variant["name"])],
            variant,
            mlx_root=mlx_root,
        )
        for variant in RMS_NORM_DIRECTX_VARIANTS
    ]
    native_compilation = _compile_directx_variants(
        artifacts_by_variant,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=require_toolchain,
    )
    return {
        "name": "rms-norm-directx-specialization-variants",
        "status": "passed",
        "source": MLX_RMS_NORM_SOURCE,
        "target": "directx",
        "report": _relpath(report_path, mlx_root),
        "artifactCount": len(artifacts),
        "executionEntryCountPerArtifact": RMS_NORM_EXPECTED_ENTRY_POINT_COUNT,
        "sourceRequiredSubgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
        "subgroupWidthEnforced": True,
        "variants": variant_evidence,
        "nativeCompilation": native_compilation,
        "runtimeParityClaimed": False,
        "numericalExecutionIncluded": False,
        "runtimeBlockedBy": list(RMS_NORM_RUNTIME_BLOCKERS),
    }


def _opengl_evidence(
    artifact: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    mlx_root: Path,
) -> dict[str, Any]:
    variant_name = str(variant["name"])
    workgroup_size = list(variant["workgroupSize"])
    artifact_path, generated = _validate_common_artifact(
        artifact,
        target="opengl",
        mlx_root=mlx_root,
    )
    entry_point = artifact.get("entryPoint")
    source_entry = (
        entry_point.get("source") if isinstance(entry_point, Mapping) else None
    )
    _require(
        artifact.get("variant") == variant_name
        and isinstance(source_entry, str)
        and source_entry in RMS_NORM_HOST_ENTRY_POINTS
        and entry_point
        == {
            "source": source_entry,
            "target": "main",
            "stage": "compute",
        },
        f"OpenGL variant {variant_name} artifact entry identity changed",
    )
    constants = artifact.get("specializationConstants")
    _require(
        isinstance(constants, list) and len(constants) == 1,
        "OpenGL RMSNorm artifact must report one function constant",
    )
    constant = constants[0]
    _require(
        isinstance(constant, Mapping)
        and constant.get("name") == RMS_NORM_FUNCTION_CONSTANT_NAME
        and constant.get("id") == RMS_NORM_FUNCTION_CONSTANT_ID
        and constant.get("required") is True
        and constant.get("overridden") is False
        and constant.get("deferred") is True
        and constant.get("status") == "required"
        and constant.get("valueProvenance") == {"kind": "runtime-override-required"}
        and "concreteValue" not in constant,
        f"OpenGL variant {variant_name} function-constant deferral is incorrect",
    )
    _validate_specialization_materialization(
        artifact,
        target=f"OpenGL variant {variant_name} entry {source_entry}",
        deferred=True,
    )
    host_materializations = _host_materializations(
        artifact,
        target=f"OpenGL variant {variant_name}",
    )
    execution, entries_by_target = _execution_entries(
        artifact,
        host_materializations,
        target="OpenGL",
        variant_name=variant_name,
        workgroup_size=workgroup_size,
        expected_source_entry_points=[source_entry],
    )
    _require(
        set(entries_by_target) == {"main"}
        and entries_by_target["main"].get("sourceEntryPoint") == source_entry,
        f"OpenGL variant {variant_name} execution target changed for {source_entry}",
    )
    declaration_pattern = (
        rf"layout\s*\(\s*constant_id\s*=\s*{RMS_NORM_FUNCTION_CONSTANT_ID}\s*\)"
        rf"\s*const\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}\s*=\s*false\s*;"
    )
    _require(
        len(re.findall(declaration_pattern, generated)) == 1,
        f"OpenGL variant {variant_name} entry {source_entry} lost constant_id = 20",
    )
    _require(
        re.search(
            rf"\buniform\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}\b",
            generated,
        )
        is None,
        f"OpenGL variant {variant_name} entry {source_entry} lowered has_w as a uniform",
    )
    local_size_pattern = re.compile(
        r"layout\s*\(\s*local_size_x\s*=\s*(?P<x>\d+)\s*,\s*"
        r"local_size_y\s*=\s*(?P<y>\d+)\s*,\s*"
        r"local_size_z\s*=\s*(?P<z>\d+)\s*\)\s*in\s*;"
    )
    generated_local_sizes = [
        [int(match.group(component)) for component in ("x", "y", "z")]
        for match in local_size_pattern.finditer(generated)
    ]
    _require(
        generated_local_sizes == [workgroup_size],
        f"OpenGL variant {variant_name} local-size contract changed for "
        f"{source_entry}",
    )
    lsize_assignments = re.findall(r"\buint\s+lsize\s*=\s*[^;]+;", generated)
    consumed_projection_count = len(
        re.findall(
            r"\buint\s+lsize\s*=\s*uint\s*\(\s*gl_WorkGroupSize\.x\s*\)\s*;",
            generated,
        )
    )
    expected_projection_count = 1 if "_looped" in source_entry else 0
    _require(
        len(lsize_assignments)
        == consumed_projection_count
        == expected_projection_count,
        f"OpenGL variant {variant_name} consumed workgroup-size projection changed "
        f"for {source_entry}",
    )
    execution_entry = entries_by_target["main"]
    return {
        "sourceEntryPoint": source_entry,
        "targetEntryPoint": "main",
        "workgroupSize": workgroup_size,
        "executionIdentity": dict(execution["identity"]),
        "executionEntryIdentity": dict(execution_entry["identity"]),
        "generatedLocalSizeContract": (
            f"layout(local_size_x = {workgroup_size[0]}, "
            f"local_size_y = {workgroup_size[1]}, "
            f"local_size_z = {workgroup_size[2]}) in;"
        ),
        "consumedWorkgroupProjectionCount": consumed_projection_count,
        "artifact": _relpath(artifact_path, mlx_root),
    }


def _compile_opengl(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "status": "not-required",
            "platform": sys.platform,
            "compiler": "glslangValidator",
            "validator": "spirv-val",
            "compiledArtifactCount": 0,
            "validatedArtifactCount": 0,
            "runs": [],
        }
    _require(
        sys.platform.startswith("linux"),
        "--require-opengl-toolchain is reserved for the Linux proof job",
    )
    glslang = shutil.which("glslangValidator")
    spirv_val = shutil.which("spirv-val")
    missing = [
        name
        for name, path in (
            ("glslangValidator", glslang),
            ("spirv-val", spirv_val),
        )
        if path is None
    ]
    _require(
        not missing,
        "OpenGL proof requires these Linux tools: " + ", ".join(missing),
    )
    _require(
        len(artifacts)
        == len(RMS_NORM_OPENGL_VARIANTS) * RMS_NORM_EXPECTED_ENTRY_POINT_COUNT,
        "OpenGL native proof must compile all 24 generated artifacts",
    )
    expected_variants = {
        str(variant["name"]): variant for variant in RMS_NORM_OPENGL_VARIANTS
    }
    seen: set[tuple[str, str]] = set()
    runs = []
    for artifact in artifacts:
        variant_name = artifact.get("variant")
        entry_point = artifact.get("entryPoint")
        source_entry = (
            entry_point.get("source") if isinstance(entry_point, Mapping) else None
        )
        identity = (str(variant_name), str(source_entry))
        _require(
            isinstance(variant_name, str)
            and variant_name in expected_variants
            and isinstance(source_entry, str)
            and source_entry in RMS_NORM_HOST_ENTRY_POINTS
            and identity not in seen,
            "OpenGL native proof artifact identity is missing or duplicated",
        )
        seen.add(identity)
        artifact_path = _artifact_path(artifact, mlx_root)
        output_path = (
            work_dir / "native" / "opengl" / variant_name / f"{source_entry}.spv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)
        command_name = f"rmsnorm-opengl-{variant_name}-{source_entry}"
        compile_result = _run_command(
            f"compile-{command_name}",
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
            f"glslangValidator failed for {variant_name}/{source_entry}",
        )
        _require(
            output_path.is_file() and output_path.stat().st_size > 0,
            f"glslangValidator did not emit SPIR-V for {variant_name}/{source_entry}",
        )
        validation_result = _run_command(
            f"validate-{command_name}",
            [str(spirv_val), "--target-env", "spv1.3", str(output_path)],
            log_dir=log_dir,
        )
        _require(
            validation_result["returncode"] == 0,
            f"spirv-val failed for {variant_name}/{source_entry}",
        )
        runs.append(
            {
                "variant": variant_name,
                "sourceEntryPoint": source_entry,
                "targetEntryPoint": "main",
                "workgroupSize": list(expected_variants[variant_name]["workgroupSize"]),
                "status": "compiled-and-validated",
                "artifact": _relpath(artifact_path, mlx_root),
                "compiledArtifact": _relpath(output_path, mlx_root),
                "compileStdout": _relpath(compile_result["stdoutPath"], mlx_root),
                "compileStderr": _relpath(compile_result["stderrPath"], mlx_root),
                "validationStdout": _relpath(validation_result["stdoutPath"], mlx_root),
                "validationStderr": _relpath(validation_result["stderrPath"], mlx_root),
            }
        )
    return {
        "required": True,
        "status": "compiled-and-validated",
        "platform": sys.platform,
        "compiler": "glslangValidator",
        "validator": "spirv-val",
        "entryPoint": "main",
        "compiledArtifactCount": len(runs),
        "validatedArtifactCount": len(runs),
        "runs": runs,
    }


def _check_opengl(
    mlx_root: Path,
    work_dir: Path,
    report_dir: Path,
    log_dir: Path,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    report_path = report_dir / "rms-norm-opengl-deferred.json"
    payload = _translate_report(
        _project_config(mlx_root, work_dir, target="opengl"),
        target="opengl",
        report_path=report_path,
    )
    artifacts = _validate_report(
        payload,
        target="opengl",
        expected_artifact_count=(
            len(RMS_NORM_OPENGL_VARIANTS) * RMS_NORM_EXPECTED_ENTRY_POINT_COUNT
        ),
    )
    _validate_project_variants(
        payload,
        RMS_NORM_OPENGL_VARIANTS,
        target="opengl",
    )
    artifacts_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for artifact in artifacts:
        variant_name = artifact.get("variant")
        entry_point = artifact.get("entryPoint")
        source_entry = (
            entry_point.get("source") if isinstance(entry_point, Mapping) else None
        )
        _require(
            isinstance(variant_name, str)
            and isinstance(source_entry, str)
            and (variant_name, source_entry) not in artifacts_by_identity,
            "OpenGL report artifact variant/entry identity is missing or duplicated",
        )
        artifacts_by_identity[(variant_name, source_entry)] = artifact
    expected_identities = {
        (str(variant["name"]), source_entry)
        for variant in RMS_NORM_OPENGL_VARIANTS
        for source_entry in RMS_NORM_HOST_ENTRY_POINTS
    }
    _require(
        set(artifacts_by_identity) == expected_identities,
        "OpenGL report did not emit 12 standalone artifacts per workgroup variant",
    )
    ordered_artifacts = [
        artifacts_by_identity[(str(variant["name"]), source_entry)]
        for variant in RMS_NORM_OPENGL_VARIANTS
        for source_entry in RMS_NORM_HOST_ENTRY_POINTS
    ]
    variant_evidence = []
    for variant in RMS_NORM_OPENGL_VARIANTS:
        variant_name = str(variant["name"])
        entry_evidence = [
            _opengl_evidence(
                artifacts_by_identity[(variant_name, source_entry)],
                variant,
                mlx_root=mlx_root,
            )
            for source_entry in RMS_NORM_HOST_ENTRY_POINTS
        ]
        variant_evidence.append(
            {
                "name": variant_name,
                "workgroupSize": list(variant["workgroupSize"]),
                "artifactCount": len(entry_evidence),
                "executionEntryCount": len(entry_evidence),
                "generatedLocalSizeContractCount": len(entry_evidence),
                "artifacts": entry_evidence,
            }
        )
    native_compilation = _compile_opengl(
        ordered_artifacts,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=require_toolchain,
    )
    return {
        "name": "rms-norm-opengl-deferred-specialization",
        "status": "passed",
        "source": MLX_RMS_NORM_SOURCE,
        "target": "opengl",
        "report": _relpath(report_path, mlx_root),
        "artifactCount": len(artifacts),
        "artifactCountPerVariant": RMS_NORM_EXPECTED_ENTRY_POINT_COUNT,
        "subgroupWidthContract": {
            "sourceRequiredWidth": RMS_NORM_SUBGROUP_WIDTH,
            "projectRuleConfigured": False,
            "targetEnforcementClaimed": False,
            "semanticParityClaimed": False,
        },
        "specializationConstant": {
            "name": RMS_NORM_FUNCTION_CONSTANT_NAME,
            "id": RMS_NORM_FUNCTION_CONSTANT_ID,
            "required": True,
            "deferred": True,
            "valueProvenance": {"kind": "runtime-override-required"},
            "specializationMaterialization": dict(
                ordered_artifacts[0]["specializationMaterialization"]
            ),
            "generatedContract": "layout(constant_id = 20) const bool has_w = false;",
        },
        "variants": variant_evidence,
        "nativeCompilation": native_compilation,
        "runtimeParityClaimed": False,
        "numericalExecutionIncluded": False,
        "runtimeBlockedBy": list(RMS_NORM_RUNTIME_BLOCKERS),
    }


def run_proof(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    _require(
        not (args.require_directx_toolchain and args.require_opengl_toolchain),
        "DirectX/Windows and OpenGL/Linux toolchains cannot be required together",
    )
    if work_dir.exists() and not args.no_clean:
        shutil.rmtree(work_dir)
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source_identity = _verify_mlx_checkout(mlx_root, log_dir=log_dir)
    directx = _check_directx(
        mlx_root,
        work_dir,
        report_dir,
        log_dir,
        require_toolchain=args.require_directx_toolchain,
    )
    opengl = _check_opengl(
        mlx_root,
        work_dir,
        report_dir,
        log_dir,
        require_toolchain=args.require_opengl_toolchain,
    )
    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-rmsnorm-specialization-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "source": MLX_RMS_NORM_SOURCE,
            "sourceSha256": MLX_RMS_NORM_SHA256,
            "targets": ["directx", "opengl"],
            "projectTranslationApi": "crosstl.project.translate_project",
            "sourceRequiredSubgroupWidth": RMS_NORM_SUBGROUP_WIDTH,
            "translationClaimed": True,
            "nativeCompilationClaimed": bool(
                args.require_directx_toolchain or args.require_opengl_toolchain
            ),
            "runtimeParityClaimed": False,
            "numericalExecutionIncluded": False,
            "fullMlxTestSuiteIncluded": False,
            "runtimeBlockedBy": list(RMS_NORM_RUNTIME_BLOCKERS),
        },
        "checks": [source_identity, directx, opengl],
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove pinned MLX rms_norm.metal specialization translation for "
            "DirectX and OpenGL."
        )
    )
    parser.add_argument("--mlx-root", required=True, help="Path to the MLX checkout")
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
        help=(
            "On Windows, require DXC compilation of a representative compute "
            "entry from both concrete DirectX variants."
        ),
    )
    parser.add_argument(
        "--require-opengl-toolchain",
        action="store_true",
        help=(
            "On Linux, require glslang OpenGL/SPIR-V compilation and spirv-val "
            "validation of the deferred-specialization artifact."
        ),
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing files in the generated work directory.",
    )
    parser.add_argument(
        "--summary",
        help="Summary JSON path. Defaults to <work-dir>/summary.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mlx_root = Path(args.mlx_root).resolve()
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    summary_path = (
        Path(args.summary).resolve() if args.summary else work_dir / "summary.json"
    )
    try:
        summary = run_proof(args)
    except MlxRmsNormProofError as exc:
        summary = {
            "schema_version": 1,
            "kind": "crosstl-mlx-rmsnorm-specialization-proof",
            "repository": {
                "name": "ml-explore/mlx",
                "url": MLX_REPOSITORY,
                "commit": MLX_COMMIT,
            },
            "scope": {
                "source": MLX_RMS_NORM_SOURCE,
                "sourceSha256": MLX_RMS_NORM_SHA256,
                "runtimeParityClaimed": False,
                "numericalExecutionIncluded": False,
                "fullMlxTestSuiteIncluded": False,
                "runtimeBlockedBy": list(RMS_NORM_RUNTIME_BLOCKERS),
            },
            "status": "failed",
            "error": str(exc),
        }
        _write_json(summary_path, summary)
        print(f"MLX RMSNorm specialization proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    _write_json(summary_path, summary)
    print(f"MLX RMSNorm specialization proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
