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
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/quantized-directx"
SUMMARY_FILENAME = "summary.json"
NATIVE_16_BIT_CAPABILITY = "directx.native-16bit-types"
NARROWED_RESOURCE_STORE = (
    "out_[uint((out_index + 4))] = " "uint(((output & 1095216660480ull) >> 32));"
)

NON_RUNTIME_CLAIMS = {
    "runtimeExecution": False,
    "numericalParity": False,
    "mlxUnitTests": False,
    "fullMlxTestSuite": False,
}

_NARROWED_RESOURCE_STORE_RE = re.compile(
    r"\bout_\s*\[\s*uint\s*\(\s*\(\s*out_index\s*\+\s*4\s*\)\s*\)\s*\]"
    r"\s*=\s*uint\s*\(\s*\(\s*\(\s*output\s*&\s*1095216660480ull\s*\)"
    r"\s*>>\s*32\s*\)\s*\)\s*;"
)
_RESOURCE_STORE_WITH_64_BIT_MASK_RE = re.compile(
    r"\bout_\s*\[[^;]*?\]\s*=\s*[^;]*?1095216660480ull[^;]*?;",
    flags=re.DOTALL,
)
_MINIMUM_PRECISION_TYPE_RE = re.compile(
    r"\bmin16(?:float|int|uint)(?:[1-4])?\b",
    flags=re.IGNORECASE,
)
_STATIC_ASSERT_RE = re.compile(r"\bstatic_assert\s*\(")
_COMPUTE_ENTRY_RE = re.compile(
    r"\[\s*numthreads\s*\([^\]]+\)\s*\]\s*" r"void\s+CSMain\s*\(",
    flags=re.MULTILINE,
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


def _project_config(mlx_root: Path, work_dir: Path) -> ProjectConfig:
    return ProjectConfig(
        root=mlx_root,
        source_roots=(MLX_KERNEL_ROOT,),
        include_patterns=(MLX_QUANTIZED_SOURCE,),
        targets=(DIRECTX_TARGET_PROFILE,),
        output_dir=_relpath(work_dir / "artifacts", mlx_root),
        source_overrides={MLX_QUANTIZED_SOURCE: "metal"},
        entry_points={MLX_QUANTIZED_SOURCE: MLX_QUANTIZED_ENTRY_POINT},
        include_dirs=(".",),
        source_options={
            "metal": {
                "max_template_specializations": TEMPLATE_SPECIALIZATION_LIMIT,
                "max_template_materialization_work": MATERIALIZATION_WORK_LIMIT,
            }
        },
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
            "source": MLX_QUANTIZED_ENTRY_POINT,
            "target": "CSMain",
            "stage": "compute",
        },
        "selected quantized entry-point identity was not preserved",
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
        and materialization.get("specializationCount") == CONCRETE_SPECIALIZATION_COUNT
        and materialization.get("unsupported") == []
        and isinstance(specializations, list)
        and len(specializations) == CONCRETE_SPECIALIZATION_COUNT
        and isinstance(accounting, Mapping)
        and accounting.get("reachableSpecializationCount")
        == REACHABLE_SPECIALIZATION_COUNT
        and accounting.get("prunedCandidateCount") == PRUNED_CANDIDATE_COUNT,
        "quantized specialization accounting changed",
    )
    selected_specializations = [
        specialization
        for specialization in specializations
        if isinstance(specialization, Mapping)
        and specialization.get("name") == "affine_quantize"
        and specialization.get("hostName") == MLX_QUANTIZED_ENTRY_POINT
    ]
    _require(
        len(selected_specializations) == 1
        and selected_specializations[0].get("materializedName")
        == MLX_QUANTIZED_ENTRY_POINT
        and selected_specializations[0].get("parameters")
        == {"T": "float", "bits": "2", "group_size": "32"},
        "selected affine_quantize<float, 32, 2> specialization changed",
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


def _validate_generated_hlsl(
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated = artifact_path.read_text(encoding="utf-8")
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
    candidate_stores = _RESOURCE_STORE_WITH_64_BIT_MASK_RE.findall(generated)
    _require(
        len(candidate_stores) == 1
        and _NARROWED_RESOURCE_STORE_RE.fullmatch(candidate_stores[0]) is not None,
        "the 64-bit expression stored in out_ must be explicitly narrowed to uint",
    )

    compatible_target_profiles = directx_target_profiles_for_source(generated)
    profile = dxc_profile_for_source(DIRECTX_BASE_SHADER_PROFILE, generated)
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    _require(
        not hlsl_requires_native_16bit_types(generated),
        "pinned quantized HLSL must not require native 16-bit types",
    )
    _require(
        DIRECTX_TARGET_PROFILE in compatible_target_profiles,
        "generated quantized HLSL must remain compatible with DirectX 12",
    )
    _require(
        profile == DIRECTX_BASE_SHADER_PROFILE and compiler_arguments == (),
        "generated quantized HLSL must compile without native 16-bit options",
    )
    checks = {
        "status": "passed",
        "entryPoint": "CSMain",
        "native16BitTypes": "not-required",
        "staticAssertions": "absent",
        "minimumPrecisionTypes": "absent",
        "typedResourceStoreNarrowing": {
            "status": "passed",
            "resource": "out_",
            "resourceElementType": "uint",
            "sourceExpressionType": "uint64_t",
            "generatedStore": NARROWED_RESOURCE_STORE,
        },
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

    output_path = (
        work_dir / "native" / "directx" / (f"{MLX_QUANTIZED_ENTRY_POINT}.dxil")
    )
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
        _project_config(root, resolved_work_dir),
        report_path=report_path,
    )
    artifact, artifact_path = _translated_artifact(
        payload,
        mlx_root=root,
        work_dir=resolved_work_dir,
    )
    generated_checks, compiler_contract = _validate_generated_hlsl(artifact_path)
    compiler = _compile_directx_artifact(
        artifact_path,
        compiler_contract,
        dxc=shutil.which("dxc"),
        mlx_root=root,
        work_dir=resolved_work_dir,
        log_dir=log_dir,
        required=require_directx_toolchain,
    )

    summary = {
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
                "selectedEntryPoint": MLX_QUANTIZED_ENTRY_POINT,
                "sourceBackend": "metal",
                "sourceOverride": "metal",
                "includeDirectories": ["."],
                "target": DIRECTX_TARGET_PROFILE,
                "projectTranslationApi": "crosstl.project.translate_project",
                "materializationLimits": {
                    "maxTemplateSpecializations": TEMPLATE_SPECIALIZATION_LIMIT,
                    "maxTemplateMaterializationWork": MATERIALIZATION_WORK_LIMIT,
                },
            },
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


def _failure_summary(*, required: bool, error: str) -> dict[str, Any]:
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
                "selectedEntryPoint": MLX_QUANTIZED_ENTRY_POINT,
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
            "Prove pinned MLX affine_quantize_float_gs_32_b_2 project "
            "translation and optional required DXC compilation."
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
        )
    except MlxQuantizedDirectXProofError as exc:
        _write_json(
            summary_path,
            _failure_summary(
                required=args.require_directx_toolchain,
                error=str(exc),
            ),
        )
        print(f"MLX quantized DirectX proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    print(f"MLX quantized DirectX proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
