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

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_RMS_NORM_SHA256 = "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee"
RMS_NORM_FUNCTION_CONSTANT_NAME = "has_w"
RMS_NORM_FUNCTION_CONSTANT_ID = 20
RMS_NORM_DIRECTX_PROFILE = "cs_6_0"
RMS_NORM_DIRECTX_VARIANTS = (
    {
        "name": "has_w_false_by_name",
        "selector": RMS_NORM_FUNCTION_CONSTANT_NAME,
        "selectorKind": "name",
        "value": False,
    },
    {
        "name": "has_w_true_by_id",
        "selector": str(RMS_NORM_FUNCTION_CONSTANT_ID),
        "selectorKind": "id",
        "value": True,
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
    }


def _project_config(
    mlx_root: Path,
    work_dir: Path,
    *,
    target: str,
) -> ProjectConfig:
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
        variants = {variant["name"]: {} for variant in RMS_NORM_DIRECTX_VARIANTS}
        variant_specializations = {
            variant["name"]: {variant["selector"]: variant["value"]}
            for variant in RMS_NORM_DIRECTX_VARIANTS
        }
        common.update(
            {
                "variants": variants,
                "variant_specialization_constants": variant_specializations,
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
        == {"pipeline": "single-file-translate", "intermediate": "crossgl"},
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


def _representative_directx_entry_point(artifact_path: Path) -> str:
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
    return str(compute_entries[0]["name"])


def _directx_variant_evidence(
    artifact: Mapping[str, Any],
    variant: Mapping[str, Any],
    *,
    mlx_root: Path,
) -> dict[str, Any]:
    variant_name = str(variant["name"])
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

    literal = "true" if variant["value"] else "false"
    static_constant = re.findall(
        rf"\bstatic\s+const\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}"
        rf"\s*=\s*{literal}\s*;",
        generated,
    )
    _require(
        len(static_constant) == 1,
        f"DirectX variant {variant_name} did not emit the expected static const",
    )
    entry_point = _representative_directx_entry_point(artifact_path)
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
        "valueProvenance": expected_provenance,
        "specializationMaterialization": dict(
            artifact["specializationMaterialization"]
        ),
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
        entry_point = _representative_directx_entry_point(artifact_path)
        output_path = output_dir / f"{variant_name}.dxil"
        output_path.unlink(missing_ok=True)
        result = _run_command(
            f"compile-rmsnorm-directx-{variant_name}",
            [
                dxc,
                "-T",
                RMS_NORM_DIRECTX_PROFILE,
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
                "status": "compiled",
                "entryPoint": entry_point,
                "artifact": _relpath(artifact_path, mlx_root),
                "compiledArtifact": _relpath(output_path, mlx_root),
                "stdout": _relpath(result["stdoutPath"], mlx_root),
                "stderr": _relpath(result["stderrPath"], mlx_root),
            }
        )
    return {
        "required": True,
        "status": "compiled",
        "platform": sys.platform,
        "compiler": "dxc",
        "profile": RMS_NORM_DIRECTX_PROFILE,
        "compiledArtifactCount": len(runs),
        "runs": runs,
    }


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
        "variants": variant_evidence,
        "nativeCompilation": native_compilation,
        "runtimeParityClaimed": False,
        "numericalExecutionIncluded": False,
    }


def _opengl_evidence(
    artifact: Mapping[str, Any],
    *,
    mlx_root: Path,
) -> tuple[dict[str, Any], Path]:
    artifact_path, generated = _validate_common_artifact(
        artifact,
        target="opengl",
        mlx_root=mlx_root,
    )
    _require(
        "variant" not in artifact,
        "OpenGL RMSNorm proof must remain an unmaterialized project variant",
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
        "OpenGL RMSNorm function-constant deferral record is incorrect",
    )
    _validate_specialization_materialization(
        artifact,
        target="OpenGL",
        deferred=True,
    )
    declaration_pattern = (
        rf"layout\s*\(\s*constant_id\s*=\s*{RMS_NORM_FUNCTION_CONSTANT_ID}\s*\)"
        rf"\s*const\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}\s*=\s*false\s*;"
    )
    _require(
        len(re.findall(declaration_pattern, generated)) == 1,
        "OpenGL RMSNorm artifact did not retain constant_id = 20",
    )
    _require(
        re.search(
            rf"\buniform\s+bool\s+{RMS_NORM_FUNCTION_CONSTANT_NAME}\b",
            generated,
        )
        is None,
        "OpenGL RMSNorm artifact lowered has_w as a uniform",
    )
    return (
        {
            "name": RMS_NORM_FUNCTION_CONSTANT_NAME,
            "id": RMS_NORM_FUNCTION_CONSTANT_ID,
            "required": True,
            "deferred": True,
            "valueProvenance": {"kind": "runtime-override-required"},
            "specializationMaterialization": dict(
                artifact["specializationMaterialization"]
            ),
            "generatedContract": "layout(constant_id = 20) const bool has_w = false;",
            "artifact": _relpath(artifact_path, mlx_root),
        },
        artifact_path,
    )


def _compile_opengl(
    artifact_path: Path,
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

    output_path = work_dir / "native" / "opengl" / "rms_norm.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    compile_result = _run_command(
        "compile-rmsnorm-opengl",
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
        "glslangValidator failed for the generated OpenGL RMSNorm artifact",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "glslangValidator did not emit RMSNorm SPIR-V",
    )
    validation_result = _run_command(
        "validate-rmsnorm-opengl-spirv",
        [str(spirv_val), "--target-env", "spv1.3", str(output_path)],
        log_dir=log_dir,
    )
    _require(
        validation_result["returncode"] == 0,
        "spirv-val failed for the generated OpenGL RMSNorm artifact",
    )
    return {
        "required": True,
        "status": "compiled-and-validated",
        "platform": sys.platform,
        "compiler": "glslangValidator",
        "validator": "spirv-val",
        "entryPoint": "main",
        "compiledArtifactCount": 1,
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compileStdout": _relpath(compile_result["stdoutPath"], mlx_root),
        "compileStderr": _relpath(compile_result["stderrPath"], mlx_root),
        "validationStdout": _relpath(validation_result["stdoutPath"], mlx_root),
        "validationStderr": _relpath(validation_result["stderrPath"], mlx_root),
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
        expected_artifact_count=1,
    )
    specialization, artifact_path = _opengl_evidence(artifacts[0], mlx_root=mlx_root)
    native_compilation = _compile_opengl(
        artifact_path,
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
        "artifactCount": 1,
        "specializationConstant": specialization,
        "nativeCompilation": native_compilation,
        "runtimeParityClaimed": False,
        "numericalExecutionIncluded": False,
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
            "translationClaimed": True,
            "nativeCompilationClaimed": bool(
                args.require_directx_toolchain or args.require_opengl_toolchain
            ),
            "runtimeParityClaimed": False,
            "numericalExecutionIncluded": False,
            "fullMlxTestSuiteIncluded": False,
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
