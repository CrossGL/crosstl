#!/usr/bin/env python3
"""Prove pinned MLX complex-to-bfloat16 copy translation for DirectX."""

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
MLX_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_COPY_SOURCE = f"{MLX_KERNEL_ROOT}/copy.metal"
MLX_COPY_ENTRY_POINT = "s_copycomplex64bfloat16"
MLX_COPY_TEMPLATE = "copy_s"
MLX_COPY_TEMPLATE_ARGUMENTS = {
    "T": "complex64_t",
    "U": "bfloat16_t",
    "N": "1",
}

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
DIRECTX_BASE_PROFILE = "cs_6_0"
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/copy-directx"

NON_RUNTIME_CLAIMS = {
    "runtimeExecution": False,
    "numericalParity": False,
    "mlxUnitTests": False,
    "fullMlxTestSuite": False,
}


class MlxCopyDirectXProofError(RuntimeError):
    """Raised when the pinned MLX copy DirectX proof contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxCopyDirectXProofError(message)


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
        raise MlxCopyDirectXProofError(
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
    return {"status": "passed", "commit": revision, "files": identities}


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
        include_patterns=(MLX_COPY_SOURCE,),
        targets=("directx",),
        output_dir=_relpath(work_dir / "artifacts", mlx_root),
        source_overrides={MLX_COPY_SOURCE: "metal"},
        entry_points={MLX_COPY_SOURCE: MLX_COPY_ENTRY_POINT},
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
        payload = translate_project(config, validate=False).to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxCopyDirectXProofError(
            f"DirectX project translation raised {type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _require_translation(payload: Mapping[str, Any]) -> None:
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), "project report summary is missing")
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and isinstance(summary.get("diagnosticCounts"), Mapping)
        and summary["diagnosticCounts"].get("error") == 0,
        "pinned copy.metal report did not translate exactly one DirectX artifact",
    )
    diagnostics = payload.get("diagnostics")
    _require(
        isinstance(diagnostics, list) and not diagnostics,
        "pinned copy.metal translation emitted diagnostics",
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
    _require_translation(payload)
    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], Mapping),
        "project report must contain exactly one DirectX artifact record",
    )
    artifact = artifacts[0]
    _require(
        artifact.get("source") == MLX_COPY_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "directx"
        and artifact.get("status") == "translated"
        and artifact.get("sourceHash")
        == {
            "algorithm": "sha256",
            "value": PINNED_FILE_SHA256[MLX_COPY_SOURCE],
        }
        and artifact.get("provenance")
        == {"pipeline": "entry-scoped-translate", "intermediate": "crossgl"},
        "DirectX artifact provenance does not match pinned copy.metal",
    )
    _require(
        artifact.get("entryPoint")
        == {
            "source": MLX_COPY_ENTRY_POINT,
            "target": "CSMain",
            "stage": "compute",
        },
        "selected copy entry-point identity was not preserved",
    )

    materialization = artifact.get("templateMaterialization")
    specializations = (
        materialization.get("specializations")
        if isinstance(materialization, Mapping)
        else None
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("specializationCount") == 1
        and materialization.get("unsupported") == []
        and isinstance(specializations, list)
        and len(specializations) == 1,
        "copy specialization materialization must contain exactly one specialization",
    )
    specialization = specializations[0]
    _require(
        isinstance(specialization, Mapping)
        and specialization.get("name") == MLX_COPY_TEMPLATE
        and specialization.get("hostName") == MLX_COPY_ENTRY_POINT
        and specialization.get("materializedName") == MLX_COPY_ENTRY_POINT
        and specialization.get("parameters") == MLX_COPY_TEMPLATE_ARGUMENTS
        and specialization.get("source") == "source-instantiation",
        "selected copy_s<complex64_t, bfloat16_t, 1> metadata changed",
    )
    accounting = materialization.get("accounting")
    _require(
        isinstance(accounting, Mapping)
        and accounting.get("reachableSpecializationCount") == 1,
        "copy specialization reachability accounting must report one specialization",
    )
    _require(
        artifact.get("bfloat16Lowering")
        == {
            "status": "exact",
            "approximationUsed": False,
            "registerRepresentation": "uint-low-16-bits",
            "storageRepresentation": "native-uint16",
            "roundingMode": "round-to-nearest-ties-to-even",
        }
        and artifact.get("requiredCapabilities") == ["directx.native-16bit-types"],
        "DirectX exact bfloat16 lowering contract changed",
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
    return artifact, artifact_path


def _validate_conversion(artifact_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    generated = artifact_path.read_text(encoding="utf-8")
    helper = "__crossgl_bfloat16_from_float"
    helper_definitions = re.findall(
        rf"\buint\s+{re.escape(helper)}\s*\(\s*float\s+\w+\s*\)",
        generated,
    )
    _require(
        len(helper_definitions) == 1,
        "generated HLSL must define the exact bfloat16 conversion helper once",
    )
    stores = re.findall(
        r"\bdst\s*\[\s*\(\s*index\s*\+\s*i\s*\)\s*\]\s*=\s*([^;]+);",
        generated,
    )
    _require(stores, "generated HLSL contains no selected copy store")
    safe_store = re.compile(
        rf"^\s*uint16_t\s*\(\s*{re.escape(helper)}\s*\(\s*float\s*\(\s*"
        r"\(\s*src\s*\.\s*Load\s*\(\s*0\s*\)\s*\)\s*\.\s*real\s*"
        r"\)\s*\)\s*\)\s*$"
    )
    for expression in stores:
        _require(
            safe_store.fullmatch(expression) is not None,
            "generated HLSL copy store must project src.Load(0).real before "
            "exact bfloat16 conversion",
        )
        _require(
            len(re.findall(r"\bsrc\s*\.\s*Load\s*\(\s*0\s*\)", expression)) == 1,
            "generated HLSL must evaluate the source value once per copy store",
        )

    source_loads = re.findall(r"\bsrc\s*\.\s*Load\s*\(\s*0\s*\)", generated)
    helper_uses = re.findall(rf"\b{re.escape(helper)}\s*\(", generated)
    _require(
        len(source_loads) == len(stores)
        and len(helper_uses) == len(stores) + len(helper_definitions),
        "generated HLSL must have one source evaluation and one conversion per store",
    )
    reflection = reflect_target_host_interface(artifact_path, target="directx")
    entry_points = reflection.get("entryPoints")
    _require(
        reflection.get("status") == "ready"
        and reflection.get("entryPointCount") == 1
        and isinstance(entry_points, list)
        and len(entry_points) == 1
        and isinstance(entry_points[0], Mapping)
        and entry_points[0].get("name") == "CSMain"
        and entry_points[0].get("stage") == "compute"
        and entry_points[0].get("executionConfig") == {"numthreads": [1, 1, 1]}
        and reflection.get("diagnostics") == [],
        "reflected DirectX compute entry contract changed",
    )
    entry = entry_points[0]
    profile = dxc_profile_for_source(DIRECTX_BASE_PROFILE, generated)
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    _require(
        profile == "cs_6_2" and compiler_arguments == ("-enable-16bit-types",),
        "generated native 16-bit storage must require DXC cs_6_2 and "
        "-enable-16bit-types",
    )
    conversion = {
        "status": "passed",
        "sourceExpression": "static_cast<bfloat16_t>(src[0])",
        "generatedProjection": "src.Load(0).real",
        "conversionHelper": helper,
        "storeSiteCount": len(stores),
        "sourceValueEvaluationsPerStore": 1,
        "complexStructPassedDirectly": False,
    }
    compiler_contract = {
        "entryPoint": str(entry["name"]),
        "stage": str(entry["stage"]),
        "profile": profile,
        "compilerArguments": list(compiler_arguments),
        "executionConfig": dict(entry["executionConfig"]),
    }
    return conversion, compiler_contract


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
    }
    if dxc is None:
        _require(not required, "DirectX copy proof requires dxc, but it is unavailable")
        return {
            **common,
            "available": False,
            "status": "not-required",
            "reason": "dxc-unavailable",
            "compiledArtifactCount": 0,
        }

    output_path = work_dir / "native" / "directx" / f"{MLX_COPY_ENTRY_POINT}.dxil"
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
        "compile-copy-directx",
        command,
        log_dir=log_dir,
    )
    _require(result["returncode"] == 0, "DXC rejected the generated copy artifact")
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "DXC did not emit the copy DXIL artifact",
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
    require_toolchain: bool = False,
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

    report_path = report_dir / "copy-metal-selected-entry.json"
    payload = _translate_report(
        _project_config(root, resolved_work_dir),
        report_path=report_path,
    )
    artifact, artifact_path = _translated_artifact(
        payload,
        mlx_root=root,
        work_dir=resolved_work_dir,
    )
    conversion, compiler_contract = _validate_conversion(artifact_path)
    toolchain = _compile_directx_artifact(
        artifact_path,
        compiler_contract,
        dxc=shutil.which("dxc"),
        mlx_root=root,
        work_dir=resolved_work_dir,
        log_dir=log_dir,
        required=require_toolchain,
    )

    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-copy-directx-toolchain-proof",
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
            "target": "directx",
            "projectTranslationApi": "crosstl.project.translate_project",
            "translationMode": "pinned-source-selected-entry",
            "fallbackUsed": False,
            "toolchainRequired": require_toolchain,
            "materializationLimits": {
                "maxTemplateSpecializations": TEMPLATE_SPECIALIZATION_LIMIT,
                "maxTemplateMaterializationWork": MATERIALIZATION_WORK_LIMIT,
            },
        },
        "claims": {
            "projectTranslation": True,
            "structuralConversionValidation": True,
            "nativeCompilation": toolchain["status"] == "compiled",
            **NON_RUNTIME_CLAIMS,
        },
        "provenance": provenance,
        "translation": {
            "status": "passed",
            "source": MLX_COPY_SOURCE,
            "report": _relpath(report_path, root),
            "artifact": _relpath(artifact_path, root),
            "artifactHash": artifact["generatedHash"],
            "entryPoint": artifact["entryPoint"],
            "templateMaterialization": artifact["templateMaterialization"],
            "complexToBfloat16Conversion": conversion,
        },
        "compilerContract": compiler_contract,
        "toolchain": toolchain,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove pinned MLX copy_s<complex64_t, bfloat16_t, 1> project "
            "translation and optional DXC compilation."
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
        "--require-toolchain",
        action="store_true",
        help="Require DXC compilation instead of accepting structural validation.",
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
    try:
        work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    except MlxCopyDirectXProofError as exc:
        print(f"MLX DirectX copy proof failed: {exc}", file=sys.stderr)
        return 1
    summary_path = (
        Path(args.summary).resolve() if args.summary else work_dir / "summary.json"
    )
    try:
        summary = run_proof(
            mlx_root,
            work_dir,
            require_toolchain=args.require_toolchain,
            clean=not args.no_clean,
        )
    except MlxCopyDirectXProofError as exc:
        summary = {
            "schema_version": 1,
            "kind": "crosstl-mlx-copy-directx-toolchain-proof",
            "repository": {
                "name": "ml-explore/mlx",
                "url": MLX_REPOSITORY,
                "commit": MLX_COMMIT,
            },
            "scope": {
                "upstreamSource": MLX_COPY_SOURCE,
                "selectedEntryPoint": MLX_COPY_ENTRY_POINT,
                "target": "directx",
                "toolchainRequired": args.require_toolchain,
                "fallbackUsed": False,
            },
            "claims": {
                "projectTranslation": False,
                "structuralConversionValidation": False,
                "nativeCompilation": False,
                **NON_RUNTIME_CLAIMS,
            },
            "status": "failed",
            "error": str(exc),
        }
        _write_json(summary_path, summary)
        print(f"MLX DirectX copy proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    _write_json(summary_path, summary)
    print(f"MLX DirectX copy proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
