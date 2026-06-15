#!/usr/bin/env python3
"""Run pinned MLX project-porting checks through the public CrossTL CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "968d264f2903d578e699c4452a4dbf48633921aa"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_ARANGE_SOURCE = "mlx/backend/metal/kernels/arange.metal"
MLX_DIRECTX_VULKAN_FRONTIER_SOURCES = (
    "mlx/backend/metal/kernels/arange.metal",
    "mlx/backend/metal/kernels/binary_two.metal",
    "mlx/backend/metal/kernels/fence.metal",
    "mlx/backend/metal/kernels/random.metal",
    "mlx/backend/metal/kernels/ternary.metal",
)
EXPECTED_METAL_KERNEL_COUNT = 40
FULL_CORPUS_TARGETS = ("directx", "opengl", "vulkan")
FULL_CORPUS_EXPECTED_ARTIFACT_COUNT = EXPECTED_METAL_KERNEL_COUNT * len(
    FULL_CORPUS_TARGETS
)
FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK = 131072
FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS = 900
REDUCED_FRONTIER_MODE = "reduced-frontier"
FULL_CORPUS_MODE = "full-corpus"
FRONTIER_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRANSLATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRACKED_ISSUES = (*FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,)
RESOLVED_FRONTIER_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1354",
    "https://github.com/CrossGL/crosstl/issues/1355",
    "https://github.com/CrossGL/crosstl/issues/1362",
    "https://github.com/CrossGL/crosstl/issues/1317",
    "https://github.com/CrossGL/crosstl/issues/1300",
    "https://github.com/CrossGL/crosstl/issues/939",
    "https://github.com/CrossGL/crosstl/issues/940",
    "https://github.com/CrossGL/crosstl/issues/941",
    "https://github.com/CrossGL/crosstl/issues/943",
    "https://github.com/CrossGL/crosstl/issues/944",
    "https://github.com/CrossGL/crosstl/issues/945",
    "https://github.com/CrossGL/crosstl/issues/946",
    "https://github.com/CrossGL/crosstl/issues/979",
    "https://github.com/CrossGL/crosstl/issues/980",
    "https://github.com/CrossGL/crosstl/issues/981",
    "https://github.com/CrossGL/crosstl/issues/982",
    "https://github.com/CrossGL/crosstl/issues/983",
    "https://github.com/CrossGL/crosstl/issues/984",
    "https://github.com/CrossGL/crosstl/issues/985",
    "https://github.com/CrossGL/crosstl/issues/1001",
    "https://github.com/CrossGL/crosstl/issues/1002",
    "https://github.com/CrossGL/crosstl/issues/1003",
    "https://github.com/CrossGL/crosstl/issues/1004",
    "https://github.com/CrossGL/crosstl/issues/1006",
    "https://github.com/CrossGL/crosstl/issues/1007",
    "https://github.com/CrossGL/crosstl/issues/1012",
    "https://github.com/CrossGL/crosstl/issues/1013",
    "https://github.com/CrossGL/crosstl/issues/1019",
    "https://github.com/CrossGL/crosstl/issues/1026",
    "https://github.com/CrossGL/crosstl/issues/1027",
    "https://github.com/CrossGL/crosstl/issues/1028",
    "https://github.com/CrossGL/crosstl/issues/1029",
    "https://github.com/CrossGL/crosstl/issues/1030",
    "https://github.com/CrossGL/crosstl/issues/1031",
    "https://github.com/CrossGL/crosstl/issues/1033",
    "https://github.com/CrossGL/crosstl/issues/1034",
    "https://github.com/CrossGL/crosstl/issues/1035",
    "https://github.com/CrossGL/crosstl/issues/1036",
    "https://github.com/CrossGL/crosstl/issues/1032",
    "https://github.com/CrossGL/crosstl/issues/1037",
    "https://github.com/CrossGL/crosstl/issues/1038",
    "https://github.com/CrossGL/crosstl/issues/1039",
    "https://github.com/CrossGL/crosstl/issues/1068",
    "https://github.com/CrossGL/crosstl/issues/1104",
    "https://github.com/CrossGL/crosstl/issues/1105",
    "https://github.com/CrossGL/crosstl/issues/1106",
    "https://github.com/CrossGL/crosstl/issues/1107",
    "https://github.com/CrossGL/crosstl/issues/1110",
    "https://github.com/CrossGL/crosstl/issues/1111",
    "https://github.com/CrossGL/crosstl/issues/1122",
    "https://github.com/CrossGL/crosstl/issues/1124",
    "https://github.com/CrossGL/crosstl/issues/1126",
    "https://github.com/CrossGL/crosstl/issues/1127",
    "https://github.com/CrossGL/crosstl/issues/852",
    "https://github.com/CrossGL/crosstl/issues/1146",
    "https://github.com/CrossGL/crosstl/issues/1155",
    "https://github.com/CrossGL/crosstl/issues/1160",
    "https://github.com/CrossGL/crosstl/issues/1184",
    "https://github.com/CrossGL/crosstl/issues/1203",
    "https://github.com/CrossGL/crosstl/issues/1204",
    "https://github.com/CrossGL/crosstl/issues/1206",
    "https://github.com/CrossGL/crosstl/issues/1205",
    "https://github.com/CrossGL/crosstl/issues/1207",
    "https://github.com/CrossGL/crosstl/issues/1218",
    "https://github.com/CrossGL/crosstl/issues/1222",
    "https://github.com/CrossGL/crosstl/issues/1238",
    "https://github.com/CrossGL/crosstl/issues/1239",
    "https://github.com/CrossGL/crosstl/issues/1240",
    "https://github.com/CrossGL/crosstl/issues/1246",
    "https://github.com/CrossGL/crosstl/issues/1248",
    "https://github.com/CrossGL/crosstl/issues/1249",
    "https://github.com/CrossGL/crosstl/issues/1250",
    "https://github.com/CrossGL/crosstl/issues/1259",
    "https://github.com/CrossGL/crosstl/issues/1260",
    "https://github.com/CrossGL/crosstl/issues/1261",
    "https://github.com/CrossGL/crosstl/issues/1274",
    "https://github.com/CrossGL/crosstl/issues/1287",
    "https://github.com/CrossGL/crosstl/issues/1329",
    "https://github.com/CrossGL/crosstl/issues/1338",
    "https://github.com/CrossGL/crosstl/issues/1340",
    "https://github.com/CrossGL/crosstl/issues/1346",
)


class PortingCheckError(RuntimeError):
    """Raised when an MLX project-porting check fails."""


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PortingCheckError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PortingCheckError(f"{path} must contain a JSON object")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortingCheckError(message)


def _resolve_work_dir(mlx_root: Path, work_dir: str | None) -> Path:
    if work_dir:
        candidate = Path(work_dir)
        if not candidate.is_absolute():
            candidate = mlx_root / candidate
    else:
        candidate = mlx_root / ".crosstl-mlx-porting"
    resolved = candidate.resolve()
    root = mlx_root.resolve()
    _require(
        _is_relative_to(resolved, root) and resolved != root,
        f"work directory must be inside the MLX checkout: {resolved}",
    )
    return resolved


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    log_dir: Path,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> CommandResult:
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
        stderr = stderr + f"\n{name} timed out after {timeout_seconds} seconds.\n"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result = CommandResult(
        name=name,
        command=list(command),
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if check and returncode != 0:
        raise PortingCheckError(
            "{} failed with exit code {}. See {} and {}.".format(
                name,
                returncode,
                stdout_path,
                stderr_path,
            )
        )
    return result


def _write_project_config(
    path: Path,
    *,
    include: str | Sequence[str],
    targets: Sequence[str],
    output_dir: str,
    metal_source_options: Mapping[str, int] | None = None,
    metal_target_options: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    include_values = [include] if isinstance(include, str) else list(include)
    include_list = ", ".join(json.dumps(value) for value in include_values)
    target_list = ", ".join(json.dumps(target) for target in targets)
    lines = [
        "[project]",
        f'source_roots = ["{MLX_METAL_KERNEL_ROOT}"]',
        f"include = [{include_list}]",
        'include_dirs = ["."]',
        f"targets = [{target_list}]",
        f'output_dir = "{output_dir}"',
        "",
        "[project.sources]",
        '"**/*.metal" = "metal"',
        "",
    ]
    if metal_source_options or metal_target_options:
        lines.append("[project.source_options.metal]")
        for key, value in (metal_source_options or {}).items():
            lines.append(f"{key} = {value}")
        lines.append("")
        for target, options in (metal_target_options or {}).items():
            lines.append(f"[project.source_options.metal.target_options.{target}]")
            for key, value in options.items():
                lines.append(f"{key} = {value}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_mlx_checkout(mlx_root: Path, python: str, log_dir: Path) -> dict[str, Any]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    _require(
        (mlx_root / MLX_ARANGE_SOURCE).is_file(),
        f"MLX Metal frontier source is missing: {MLX_ARANGE_SOURCE}",
    )
    for source in MLX_DIRECTX_VULKAN_FRONTIER_SOURCES:
        _require(
            (mlx_root / source).is_file(),
            f"MLX Metal frontier source is missing: {source}",
        )
    result = _run_command(
        "mlx-revision",
        ["git", "-C", str(mlx_root), "rev-parse", "HEAD"],
        log_dir=log_dir,
    )
    revision = result.stdout_path.read_text(encoding="utf-8").strip()
    _require(
        revision == MLX_COMMIT,
        f"MLX checkout must be pinned to {MLX_COMMIT}; found {revision}",
    )
    return {
        "name": "mlx-checkout",
        "status": "passed",
        "repository": MLX_REPOSITORY,
        "commit": revision,
        "python": python,
    }


def _scan_metal_kernels(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "scan-metal-kernels.toml"
    report_path = report_dir / "scan-metal-kernels.json"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=("directx", "opengl", "vulkan"),
        output_dir=_relpath(work_dir / "out-scan", mlx_root),
    )
    _run_command(
        "scan-metal-kernels",
        [
            python,
            "-m",
            "crosstl",
            "scan",
            str(mlx_root),
            "--config",
            str(config_path),
            "--output",
            str(report_path),
        ],
        log_dir=log_dir,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    units = payload.get("units", [])
    _require(isinstance(summary, dict), "scan report summary must be an object")
    _require(isinstance(units, list), "scan report units must be a list")
    _require(
        summary.get("unitCount") == EXPECTED_METAL_KERNEL_COUNT,
        "expected {} MLX Metal kernels, found {}".format(
            EXPECTED_METAL_KERNEL_COUNT,
            summary.get("unitCount"),
        ),
    )
    _require(
        summary.get("diagnosticCounts", {}).get("error", 0) == 0,
        "MLX Metal scan reported errors",
    )
    unit_paths = {unit.get("path") for unit in units if isinstance(unit, dict)}
    for source in MLX_DIRECTX_VULKAN_FRONTIER_SOURCES:
        _require(source in unit_paths, f"{source} was not scanned")
    return {
        "name": "metal-kernel-scan",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "unitCount": summary.get("unitCount"),
        "includeDependencyCount": summary.get("includeDependencyCount"),
        "targets": ["directx", "opengl", "vulkan"],
    }


def _translate_directx_vulkan_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_vulkan_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "directx-vulkan-frontier.toml"
    report_path = report_dir / "directx-vulkan-frontier.json"
    _write_project_config(
        config_path,
        include=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
        targets=("directx", "vulkan"),
        output_dir=_relpath(work_dir / "out-directx-vulkan-frontier", mlx_root),
    )
    run_toolchains = not FRONTIER_VALIDATION_TRACKED_ISSUES
    validation_flag = "--run-toolchains" if run_toolchains else "--validate"
    _run_command(
        "translate-directx-vulkan-frontier",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            validation_flag,
        ],
        log_dir=log_dir,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "translation report summary must be an object")
    frontier_count = len(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
    _require(
        summary.get("unitCount") == frontier_count,
        f"DirectX/Vulkan frontier translation must scan {frontier_count} units",
    )
    _require(
        summary.get("artifactCount") == frontier_count * 2,
        "DirectX/Vulkan frontier translation must emit {} artifacts".format(
            frontier_count * 2
        ),
    )
    _require(
        summary.get("translatedCount") == frontier_count * 2,
        "DirectX/Vulkan frontier translation did not emit every artifact",
    )
    _require(
        summary.get("failedCount") == 0,
        "DirectX/Vulkan frontier translation reported failed artifacts",
    )
    _require(
        summary.get("diagnosticCounts", {}).get("error", 0) == 0,
        "DirectX/Vulkan frontier translation reported errors",
    )
    artifacts_by_target = summary.get("artifactsByTarget", {})
    for target in ("directx", "vulkan"):
        target_summary = artifacts_by_target.get(target, {})
        _require(
            target_summary.get("translatedCount") == frontier_count
            and target_summary.get("failedCount") == 0,
            f"DirectX/Vulkan frontier {target} artifacts were not translated cleanly",
        )
    validation = payload.get("validation", {})
    _require(isinstance(validation, dict), "translation validation must be an object")
    artifact_validation = validation.get("summary", {})
    _require(
        artifact_validation.get("failedCount") == 0,
        "artifact validation reported failures for DirectX/Vulkan frontier outputs",
    )
    toolchain_runs = validation.get("toolchainRuns", [])
    _require(isinstance(toolchain_runs, list), "toolchainRuns must be a list")
    vulkan_runs = [
        run
        for run in toolchain_runs
        if isinstance(run, dict) and run.get("target") == "vulkan"
    ]
    if require_vulkan_toolchain and run_toolchains:
        _require(
            len(vulkan_runs) == frontier_count,
            "Vulkan toolchain validation was required for every frontier artifact",
        )
    for run in vulkan_runs:
        _require(run.get("status") == "ok", "Vulkan toolchain validation failed")
    if require_vulkan_toolchain and not run_toolchains:
        _require(
            not vulkan_runs,
            "Vulkan toolchain validation ran while active validation issues are tracked",
        )
    return {
        "name": "directx-vulkan-frontier",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": frontier_count,
        "artifactCount": frontier_count * 2,
        "targets": ["directx", "vulkan"],
        "toolchainRuns": len(toolchain_runs),
        "vulkanToolchainRequired": require_vulkan_toolchain,
        "vulkanValidationStatus": (
            "validated" if run_toolchains else "blocked-by-tracked-issues"
        ),
        "trackedIssues": list(FRONTIER_VALIDATION_TRACKED_ISSUES),
    }


def _check_arange_opengl(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "arange-opengl.toml"
    report_path = report_dir / "arange-opengl.json"
    _write_project_config(
        config_path,
        include=MLX_ARANGE_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(work_dir / "out-arange-opengl", mlx_root),
    )
    result = _run_command(
        "translate-arange-opengl",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "OpenGL report summary must be an object")
    if result.returncode != 0:
        messages = []
        for diagnostic in payload.get("diagnostics", []):
            if isinstance(diagnostic, dict):
                message = diagnostic.get("message")
                if isinstance(message, str):
                    messages.append(message)
        for artifact in payload.get("artifacts", []):
            if isinstance(artifact, dict):
                error = artifact.get("error")
                if isinstance(error, str):
                    messages.append(error)
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"OpenGL arange translation failed{detail}")

    _require(
        summary.get("translatedCount") == 1 and summary.get("failedCount") == 0,
        "OpenGL arange translation succeeded but the report did not show one clean artifact",
    )
    artifacts = payload.get("artifacts", [])
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("source") == MLX_ARANGE_SOURCE
            and item.get("target") == "opengl"
        ),
        None,
    )
    _require(isinstance(artifact, dict), "OpenGL arange artifact is missing")
    artifact_path = artifact.get("path")
    _require(isinstance(artifact_path, str), "OpenGL arange artifact path is missing")
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(), f"OpenGL arange artifact is missing: {artifact_path}"
    )
    generated = generated_path.read_text(encoding="utf-8")
    generated_lower = generated.lower()
    _require(
        "#include <metal" not in generated_lower
        and "#pragma metal" not in generated_lower,
        "OpenGL arange artifact retained a Metal system preprocessor line",
    )
    return {
        "name": "arange-opengl",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_ARANGE_SOURCE,
        "target": "opengl",
        "metalIncludesFiltered": True,
    }


def _translate_full_corpus(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "full-corpus.toml"
    report_path = report_dir / "full-corpus.json"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=FULL_CORPUS_TARGETS,
        output_dir=_relpath(work_dir / "out-full-corpus", mlx_root),
        metal_source_options={
            "max_template_specializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        },
    )
    result = _run_command(
        "translate-full-corpus",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(mlx_root),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
            "--validate",
        ],
        log_dir=log_dir,
        check=False,
        timeout_seconds=FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
    )
    if not report_path.is_file() and result.returncode:
        _require(
            FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
            "full-corpus translation failed before writing a report without "
            "tracked issue references",
        )
        return {
            "name": "full-corpus",
            "status": "blocked-by-tracked-issues",
            "report": _relpath(report_path, mlx_root),
            "reportProduced": False,
            "unitCount": EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "targets": list(FULL_CORPUS_TARGETS),
            "returncode": result.returncode,
            "timeoutSeconds": FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS,
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    _require(
        report_path.is_file(),
        "full-corpus translation did not produce a project report",
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "full-corpus summary must be an object")
    diagnostic_counts = summary.get("diagnosticCounts", {})
    _require(
        isinstance(diagnostic_counts, dict),
        "full-corpus diagnostic counts must be an object",
    )
    failed_count = summary.get("failedCount")
    error_count = diagnostic_counts.get("error", 0)
    _require(
        summary.get("unitCount") == EXPECTED_METAL_KERNEL_COUNT,
        "full-corpus translation must scan {} units; found {}".format(
            EXPECTED_METAL_KERNEL_COUNT,
            summary.get("unitCount"),
        ),
    )
    _require(
        summary.get("artifactCount") == FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "full-corpus translation must emit {} artifacts; found {}".format(
            FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            summary.get("artifactCount"),
        ),
    )
    artifacts_by_target = summary.get("artifactsByTarget", {})
    _require(
        isinstance(artifacts_by_target, dict),
        "full-corpus artifactsByTarget must be an object",
    )
    target_counts: dict[str, dict[str, int]] = {}
    for target in FULL_CORPUS_TARGETS:
        target_summary = artifacts_by_target.get(target, {})
        _require(
            isinstance(target_summary, dict),
            f"full-corpus target summary is missing for {target}",
        )
        target_counts[target] = {
            "translatedCount": target_summary.get("translatedCount", 0),
            "failedCount": target_summary.get("failedCount", 0),
        }
    validation = payload.get("validation", {})
    _require(isinstance(validation, dict), "full-corpus validation must be an object")
    artifact_validation = validation.get("summary", {})
    _require(
        isinstance(artifact_validation, dict),
        "full-corpus validation summary must be an object",
    )
    has_translation_failures = bool(failed_count or error_count or result.returncode)
    if has_translation_failures:
        _require(
            FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
            "full-corpus translation reported failed artifacts or errors "
            "without tracked issue references",
        )
        return {
            "name": "full-corpus",
            "status": "blocked-by-tracked-issues",
            "report": _relpath(report_path, mlx_root),
            "unitCount": EXPECTED_METAL_KERNEL_COUNT,
            "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "translatedCount": summary.get("translatedCount", 0),
            "failedCount": summary.get("failedCount", 0),
            "diagnosticCounts": diagnostic_counts,
            "diagnosticsByCode": summary.get("diagnosticsByCode", {}),
            "targets": list(FULL_CORPUS_TARGETS),
            "targetCounts": target_counts,
            "validationFailedCount": artifact_validation.get("failedCount", 0),
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    _require(
        summary.get("translatedCount") == FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "full-corpus translation did not emit every expected artifact",
    )
    _require(
        artifact_validation.get("failedCount") == 0,
        "artifact validation reported failures for full-corpus outputs",
    )
    for target, target_count in target_counts.items():
        _require(
            target_count["translatedCount"] == EXPECTED_METAL_KERNEL_COUNT
            and target_count["failedCount"] == 0,
            f"full-corpus {target} artifacts were not translated cleanly",
        )
    return {
        "name": "full-corpus",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "unitCount": EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "targets": list(FULL_CORPUS_TARGETS),
        "targetCounts": target_counts,
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
        "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK,
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    if work_dir.exists() and not args.no_clean:
        shutil.rmtree(work_dir)
    config_dir = work_dir / "configs"
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    for directory in (config_dir, report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, Any]] = [
        _verify_mlx_checkout(mlx_root, args.python, log_dir),
        _scan_metal_kernels(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
        ),
    ]
    if args.mode == REDUCED_FRONTIER_MODE:
        checks.extend(
            [
                _translate_directx_vulkan_frontier(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                    require_vulkan_toolchain=args.require_vulkan_toolchain,
                ),
                _check_arange_opengl(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                ),
            ]
        )
    elif args.mode == FULL_CORPUS_MODE:
        checks.append(
            _translate_full_corpus(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
    else:
        raise PortingCheckError(f"unsupported MLX porting mode: {args.mode}")
    return {
        "schema_version": 1,
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "mode": args.mode,
            "sourceRoot": MLX_METAL_KERNEL_ROOT,
            "frontierSources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
            "fullCorpusTargets": list(FULL_CORPUS_TARGETS),
            "fullCorpusExpectedUnitCount": EXPECTED_METAL_KERNEL_COUNT,
            "fullCorpusExpectedArtifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
        },
        "trackedIssues": list(FULL_CORPUS_TRACKED_ISSUES),
        "resolvedFrontierIssues": list(RESOLVED_FRONTIER_ISSUES),
        "checks": checks,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pinned MLX project-porting checks through CrossTL."
    )
    parser.add_argument("--mlx-root", required=True, help="Path to the MLX checkout")
    parser.add_argument(
        "--mode",
        choices=(REDUCED_FRONTIER_MODE, FULL_CORPUS_MODE),
        default=REDUCED_FRONTIER_MODE,
        help=(
            "Harness scope to run. The default reduced frontier is the pull "
            "request gate; full-corpus is intended for scheduled and manual "
            "artifact-generation scouts."
        ),
    )
    parser.add_argument(
        "--work-dir",
        help=(
            "Generated config/report/output directory. Defaults to "
            "<mlx-root>/.crosstl-mlx-porting."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to invoke `python -m crosstl`.",
    )
    parser.add_argument(
        "--require-vulkan-toolchain",
        action="store_true",
        help="Fail unless the Vulkan SPIR-V smoke check runs successfully.",
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
        summary = run_checks(args)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except PortingCheckError as exc:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repository": {
                        "name": "ml-explore/mlx",
                        "url": MLX_REPOSITORY,
                        "commit": MLX_COMMIT,
                    },
                    "scope": {
                        "mode": args.mode,
                        "shaderArtifactsOnly": True,
                        "runtimeIntegrationIncluded": False,
                    },
                    "status": "failed",
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"MLX project-porting checks failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    print(f"MLX project-porting checks passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
