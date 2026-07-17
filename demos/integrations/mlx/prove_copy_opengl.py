#!/usr/bin/env python3
"""Prove pinned MLX complex-to-float copy translation for OpenGL."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.project import ProjectConfig, translate_project

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_COPY_SOURCE = f"{MLX_KERNEL_ROOT}/copy.metal"
MLX_COPY_ENTRY_POINT = "s_copycomplex64float32"
MLX_COPY_TEMPLATE = "copy_s"
MLX_COPY_TEMPLATE_ARGUMENTS = {"T": "complex64_t", "U": "float", "N": "1"}
MLX_COPY_DECLARED_ENTRY_COUNT = 2496
MLX_COPY_PREPROCESSED_INSTANTIATION_COUNT = 2497

PINNED_FILE_SHA256 = {
    MLX_COPY_SOURCE: "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba",
    f"{MLX_KERNEL_ROOT}/copy.h": (
        "faafc09afc5e190252f3544c966b333b503c30404be3836444abf645f615b1c8"
    ),
    f"{MLX_KERNEL_ROOT}/utils.h": (
        "c30223b42b71068321149eea4fcd319878a4004425fb7cc34cdd296a76fabbfc"
    ),
    f"{MLX_KERNEL_ROOT}/bf16.h": (
        "abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb"
    ),
    f"{MLX_KERNEL_ROOT}/bf16_math.h": (
        "1f374f8380f756eb89acf6a847741cb8fecbe642945e159fb6208d804cc06496"
    ),
    f"{MLX_KERNEL_ROOT}/complex.h": (
        "aa3d29a2a0bb31fc0071493e3ac917387f96ba059d10a8f371a0b6a41a216dd3"
    ),
    f"{MLX_KERNEL_ROOT}/defines.h": (
        "a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163"
    ),
    f"{MLX_KERNEL_ROOT}/logging.h": (
        "fae44781743dbc5eb727e505b090e8445adff5626c1179bcb193b6fe7bedac8f"
    ),
}

MATERIALIZATION_WORK_LIMIT = 64
TEMPLATE_SPECIALIZATION_LIMIT = 16
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/copy-opengl"

NON_RUNTIME_CLAIMS = {
    "runtimeExecution": False,
    "numericalParity": False,
    "mlxUnitTests": False,
    "fullMlxTestSuite": False,
}


class MlxCopyOpenGLProofError(RuntimeError):
    """Raised when the pinned MLX copy proof contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxCopyOpenGLProofError(message)


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
        raise MlxCopyOpenGLProofError(
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

    identities = []
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
        identities.append(
            {
                "path": relative_path,
                "kind": "source" if relative_path == MLX_COPY_SOURCE else "header",
                "hash": {"algorithm": "sha256", "value": actual_hash},
            }
        )
    return {
        "status": "passed",
        "commit": revision,
        "files": identities,
    }


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


def _source_options() -> dict[str, Mapping[str, Any]]:
    return {
        "metal": {
            "max_template_specializations": TEMPLATE_SPECIALIZATION_LIMIT,
            "max_template_materialization_work": MATERIALIZATION_WORK_LIMIT,
        }
    }


def _project_config(
    mlx_root: Path,
    work_dir: Path,
    *,
    source: str,
    output_name: str,
) -> ProjectConfig:
    source_root = Path(source).parent.as_posix()
    return ProjectConfig(
        root=mlx_root,
        source_roots=(source_root,),
        include_patterns=(source,),
        targets=("opengl",),
        output_dir=_relpath(work_dir / output_name, mlx_root),
        source_overrides={source: "metal"},
        entry_points={source: MLX_COPY_ENTRY_POINT},
        include_dirs=(".",),
        source_options=_source_options(),
    )


def _translate_report(
    config: ProjectConfig,
    *,
    report_path: Path,
) -> dict[str, Any]:
    try:
        payload = translate_project(config, validate=False).to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxCopyOpenGLProofError(
            f"OpenGL project translation raised {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _require_real_source_translation(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    _require(
        isinstance(summary, Mapping),
        "pinned copy.metal translation summary is missing",
    )
    diagnostic_counts = summary.get("diagnosticCounts")
    if (
        summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and isinstance(diagnostic_counts, Mapping)
        and diagnostic_counts.get("error") == 0
    ):
        return {"status": "translated", "fallbackUsed": False}

    diagnostics = payload.get("diagnostics")
    diagnostic_records = diagnostics if isinstance(diagnostics, list) else []
    diagnostic_codes = sorted(
        {
            str(diagnostic.get("code"))
            for diagnostic in diagnostic_records
            if isinstance(diagnostic, Mapping) and diagnostic.get("code")
        }
    )
    failure_summary = {
        key: summary.get(key)
        for key in (
            "artifactCount",
            "translatedCount",
            "failedCount",
            "diagnosticCounts",
        )
    }
    raise MlxCopyOpenGLProofError(
        "pinned copy.metal selected-entry translation failed: "
        f"summary={json.dumps(failure_summary, sort_keys=True)}; "
        f"diagnosticCodes={diagnostic_codes}"
    )


def _translated_artifact(
    payload: Mapping[str, Any],
    *,
    mlx_root: Path,
    work_dir: Path,
) -> tuple[Mapping[str, Any], Path]:
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        "translation did not produce a project portability report",
    )
    summary = payload.get("summary")
    diagnostics = payload.get("diagnostics")
    artifacts = payload.get("artifacts")
    _require(
        isinstance(summary, Mapping)
        and summary.get("unitCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and summary.get("diagnosticCounts", {}).get("error") == 0,
        "copy specialization project report did not translate one artifact",
    )
    _require(
        isinstance(diagnostics, list) and not diagnostics,
        "copy specialization project translation emitted diagnostics",
    )
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], Mapping),
        "copy specialization project report must contain one artifact",
    )
    artifact = artifacts[0]
    _require(
        artifact.get("source") == MLX_COPY_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "opengl"
        and artifact.get("status") == "translated"
        and artifact.get("sourceHash")
        == {
            "algorithm": "sha256",
            "value": PINNED_FILE_SHA256[MLX_COPY_SOURCE],
        },
        "copy specialization artifact provenance does not match pinned copy.metal",
    )
    _require(
        artifact.get("entryPoint")
        == {
            "source": MLX_COPY_ENTRY_POINT,
            "target": "main",
            "stage": "compute",
        },
        "copy specialization entry-point selection was not preserved",
    )
    materialization = artifact.get("templateMaterialization")
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("unsupported") == [],
        "copy specialization materialization is incomplete",
    )
    specializations = materialization.get("specializations")
    _require(
        isinstance(specializations, list)
        and materialization.get("specializationCount") == 1
        and len(specializations) == 1
        and any(
            isinstance(record, Mapping)
            and record.get("name") == MLX_COPY_TEMPLATE
            and record.get("hostName") == MLX_COPY_ENTRY_POINT
            and record.get("materializedName") == MLX_COPY_ENTRY_POINT
            and record.get("parameters") == MLX_COPY_TEMPLATE_ARGUMENTS
            for record in specializations
        ),
        "copy_s<complex64_t, float, 1> was not the only materialized specialization",
    )
    accounting = materialization.get("accounting")
    _require(
        isinstance(accounting, Mapping)
        and accounting.get("reachableSpecializationCount") == 1,
        "copy specialization reachability accounting is incomplete",
    )

    artifact_path = (mlx_root / str(artifact.get("path", ""))).resolve()
    _require(
        _is_relative_to(artifact_path, work_dir.resolve()) and artifact_path.is_file(),
        f"generated GLSL artifact is missing or outside the work directory: {artifact_path}",
    )
    generated_hash = artifact.get("generatedHash")
    _require(
        generated_hash == {"algorithm": "sha256", "value": _sha256(artifact_path)},
        "generated GLSL artifact hash does not match the project report",
    )
    return artifact, artifact_path


def _validate_real_projection(artifact_path: Path) -> dict[str, Any]:
    source = artifact_path.read_text(encoding="utf-8")
    expected_store = "dst[(index + uint(i))] = (src[0]).real;"
    _require(
        source.count(expected_store) == 1,
        "generated GLSL must store exactly one projection of src[0].real",
    )
    _require(
        "float(src[0])" not in source and "float((src[0]))" not in source,
        "generated GLSL passed complex64_t directly to a float constructor",
    )
    _require(
        "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;" in source
        and "void main()" in source,
        "generated GLSL compute entry-point contract is incomplete",
    )
    return {
        "status": "passed",
        "sourceExpression": "static_cast<float>(src[0])",
        "generatedExpression": "(src[0]).real",
        "store": expected_store,
        "singleEvaluation": True,
    }


def _compile_and_validate(
    artifact_path: Path,
    *,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
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
        "OpenGL copy proof requires these tools: " + ", ".join(missing),
    )
    output_path = work_dir / "toolchain" / f"{MLX_COPY_ENTRY_POINT}.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    compile_result = _run_command(
        "compile-copy-opengl",
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
        "glslangValidator rejected the generated copy specialization",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "glslangValidator did not emit the copy specialization SPIR-V module",
    )
    validation_result = _run_command(
        "validate-copy-opengl",
        [str(spirv_val), "--target-env", "spv1.3", str(output_path)],
        log_dir=log_dir,
    )
    _require(
        validation_result["returncode"] == 0,
        "spirv-val rejected the generated copy specialization module",
    )

    runs = []
    for result in (compile_result, validation_result):
        runs.append(
            {
                "name": result["name"],
                "command": result["command"],
                "returncode": result["returncode"],
                "stdout": _relpath(result["stdoutPath"], mlx_root),
                "stderr": _relpath(result["stderrPath"], mlx_root),
            }
        )
    return {
        "status": "compiled-and-validated",
        "compiler": "glslangValidator",
        "compilerTarget": "OpenGL/SPIR-V 1.3",
        "validator": "spirv-val",
        "validatorTarget": "SPIR-V 1.3",
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compiledArtifactHash": {
            "algorithm": "sha256",
            "value": _sha256(output_path),
        },
        "runs": runs,
    }


def run_proof(
    mlx_root: Path,
    work_dir: Path,
    *,
    clean: bool = True,
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

    real_report_path = report_dir / "copy-metal-selected-entry-probe.json"
    real_payload = _translate_report(
        _project_config(
            root,
            resolved_work_dir,
            source=MLX_COPY_SOURCE,
            output_name="real-source-artifacts",
        ),
        report_path=real_report_path,
    )
    source_translation = _require_real_source_translation(real_payload)

    artifact, artifact_path = _translated_artifact(
        real_payload,
        mlx_root=root,
        work_dir=resolved_work_dir,
    )
    projection = _validate_real_projection(artifact_path)
    toolchain = _compile_and_validate(
        artifact_path,
        mlx_root=root,
        work_dir=resolved_work_dir,
        log_dir=log_dir,
    )

    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-copy-opengl-toolchain-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "upstreamSource": MLX_COPY_SOURCE,
            "selectedEntryPoint": MLX_COPY_ENTRY_POINT,
            "templateSpecialization": {
                "name": MLX_COPY_TEMPLATE,
                "arguments": MLX_COPY_TEMPLATE_ARGUMENTS,
            },
            "target": "opengl",
            "projectTranslationApi": "crosstl.project.translate_project",
            "translationMode": "pinned-source-selected-entry",
            "realCopyMetalTranslated": True,
            "fallbackUsed": False,
            "copyMetalDeclaredEntryCount": MLX_COPY_DECLARED_ENTRY_COUNT,
            "preprocessedInstantiationCount": MLX_COPY_PREPROCESSED_INSTANTIATION_COUNT,
            "materializationLimits": {
                "maxTemplateSpecializations": TEMPLATE_SPECIALIZATION_LIMIT,
                "maxTemplateMaterializationWork": MATERIALIZATION_WORK_LIMIT,
            },
        },
        "claims": {
            "projectTranslation": True,
            "nativeCompilation": True,
            "spirvValidation": True,
            **NON_RUNTIME_CLAIMS,
        },
        "provenance": provenance,
        "sourceTranslation": {
            **source_translation,
            "report": _relpath(real_report_path, root),
        },
        "translation": {
            "status": "passed",
            "source": MLX_COPY_SOURCE,
            "report": _relpath(real_report_path, root),
            "artifact": _relpath(artifact_path, root),
            "artifactHash": artifact["generatedHash"],
            "entryPoint": artifact["entryPoint"],
            "templateMaterialization": artifact["templateMaterialization"],
            "complexToScalarProjection": projection,
        },
        "toolchain": toolchain,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove pinned MLX copy_s<complex64_t, float, 1> project translation "
            "and OpenGL/SPIR-V 1.3 toolchain acceptance."
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
        summary = run_proof(mlx_root, work_dir, clean=not args.no_clean)
    except MlxCopyOpenGLProofError as exc:
        print(f"MLX OpenGL copy proof failed: {exc}", file=sys.stderr)
        return 1
    summary_path = work_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"MLX OpenGL copy proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
