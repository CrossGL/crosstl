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
from typing import Any, Sequence

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
FRONTIER_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRACKED_ISSUES = (*FRONTIER_VALIDATION_TRACKED_ISSUES,)
RESOLVED_FRONTIER_ISSUES = (
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
    "https://github.com/CrossGL/crosstl/issues/1317",
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


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
) -> CommandResult:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    stdout_path = log_dir / f"{name}.stdout"
    stderr_path = log_dir / f"{name}.stderr"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = CommandResult(
        name=name,
        command=list(command),
        returncode=completed.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if check and completed.returncode != 0:
        raise PortingCheckError(
            "{} failed with exit code {}. See {} and {}.".format(
                name,
                completed.returncode,
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
) -> None:
    include_values = [include] if isinstance(include, str) else list(include)
    include_list = ", ".join(json.dumps(value) for value in include_values)
    target_list = ", ".join(json.dumps(target) for target in targets)
    path.write_text(
        "\n".join(
            [
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
        ),
        encoding="utf-8",
    )


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


def _target_toolchain(payload: dict[str, Any], target: str) -> dict[str, Any] | None:
    validation = payload.get("validation", {})
    if not isinstance(validation, dict):
        return None
    for toolchain in _records(validation.get("toolchains")):
        if toolchain.get("target") == target:
            return toolchain
    return None


def _target_toolchain_runs(
    payload: dict[str, Any],
    target: str,
) -> list[dict[str, Any]]:
    validation = payload.get("validation", {})
    if not isinstance(validation, dict):
        return []
    return [
        run
        for run in _records(validation.get("toolchainRuns"))
        if run.get("target") == target
    ]


def _target_artifacts(payload: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return [
        artifact
        for artifact in _records(payload.get("artifacts"))
        if artifact.get("target") == target and artifact.get("status") == "translated"
    ]


def _missing_tool_names(toolchain: dict[str, Any] | None) -> list[str]:
    if not isinstance(toolchain, dict):
        return []
    missing = []
    for tool in _records(toolchain.get("tools")):
        if tool.get("available") is False and isinstance(tool.get("name"), str):
            missing.append(tool["name"])
    return missing


def _require_target_toolchain_smoke(
    payload: dict[str, Any],
    target: str,
    *,
    label: str,
) -> dict[str, Any]:
    toolchain = _target_toolchain(payload, target)
    _require(
        isinstance(toolchain, dict),
        f"{label} required {target} toolchain smoke checks, but no toolchain status was recorded",
    )
    missing_tools = _missing_tool_names(toolchain)
    _require(
        toolchain.get("status") == "available",
        "{} required {} toolchain smoke checks, but the toolchain was {}{}".format(
            label,
            target,
            toolchain.get("status", "unknown"),
            f" (missing: {', '.join(missing_tools)})" if missing_tools else "",
        ),
    )

    artifacts = _target_artifacts(payload, target)
    expected_paths = {
        str(artifact.get("path"))
        for artifact in artifacts
        if isinstance(artifact.get("path"), str) and artifact.get("path")
    }
    _require(
        bool(expected_paths),
        f"{label} required {target} toolchain smoke checks, but no translated artifacts were recorded",
    )

    runs = _target_toolchain_runs(payload, target)
    failed_runs = [run for run in runs if run.get("status") != "ok"]
    if failed_runs:
        first = failed_runs[0]
        detail = first.get("stderr") or first.get("stdout") or "no tool output"
        raise PortingCheckError(
            "{} required {} toolchain smoke checks, but {} failed: {}".format(
                label,
                target,
                first.get("path", "an artifact"),
                str(detail).splitlines()[0],
            )
        )

    ok_paths = {
        str(run.get("path"))
        for run in runs
        if run.get("status") == "ok" and isinstance(run.get("path"), str)
    }
    missing_paths = sorted(expected_paths - ok_paths)
    _require(
        not missing_paths,
        "{} required {} toolchain smoke checks, but checks were missing or skipped for: {}".format(
            label,
            target,
            ", ".join(missing_paths[:5]),
        ),
    )
    return {
        "target": target,
        "status": "validated",
        "artifactCount": len(expected_paths),
        "runCount": len(runs),
    }


def _toolchain_run_counts_by_target(
    runs: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        target = run.get("target")
        if not isinstance(target, str) or not target:
            continue
        row = counts.setdefault(target, {"runCount": 0, "okCount": 0, "failedCount": 0})
        row["runCount"] += 1
        if run.get("status") == "ok":
            row["okCount"] += 1
        elif run.get("status") == "failed":
            row["failedCount"] += 1
    return {target: counts[target] for target in sorted(counts)}


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
    require_directx_toolchain: bool,
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
    run_toolchains = (
        not FRONTIER_VALIDATION_TRACKED_ISSUES
        or require_directx_toolchain
        or require_vulkan_toolchain
    )
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
    typed_toolchain_runs = _records(toolchain_runs)
    required_toolchains = []
    if require_directx_toolchain:
        required_toolchains.append(
            _require_target_toolchain_smoke(
                payload,
                "directx",
                label="DirectX/Vulkan frontier",
            )
        )
    if require_vulkan_toolchain:
        required_toolchains.append(
            _require_target_toolchain_smoke(
                payload,
                "vulkan",
                label="DirectX/Vulkan frontier",
            )
        )
    return {
        "name": "directx-vulkan-frontier",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": frontier_count,
        "artifactCount": frontier_count * 2,
        "targets": ["directx", "vulkan"],
        "toolchainRuns": len(typed_toolchain_runs),
        "toolchainRunsByTarget": _toolchain_run_counts_by_target(typed_toolchain_runs),
        "requiredToolchains": required_toolchains,
        "directxToolchainRequired": require_directx_toolchain,
        "vulkanToolchainRequired": require_vulkan_toolchain,
        "trackedIssues": list(FRONTIER_VALIDATION_TRACKED_ISSUES),
    }


def _check_arange_opengl(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_opengl_toolchain: bool,
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
            "--run-toolchains" if require_opengl_toolchain else "--validate",
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
    validation = payload.get("validation", {})
    toolchain_runs = (
        _records(validation.get("toolchainRuns"))
        if isinstance(validation, dict)
        else []
    )
    required_toolchains = []
    if require_opengl_toolchain:
        required_toolchains.append(
            _require_target_toolchain_smoke(
                payload,
                "opengl",
                label="OpenGL arange",
            )
        )
    return {
        "name": "arange-opengl",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_ARANGE_SOURCE,
        "target": "opengl",
        "metalIncludesFiltered": True,
        "toolchainRuns": len(toolchain_runs),
        "toolchainRunsByTarget": _toolchain_run_counts_by_target(toolchain_runs),
        "requiredToolchains": required_toolchains,
        "openglToolchainRequired": require_opengl_toolchain,
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

    checks = [
        _verify_mlx_checkout(mlx_root, args.python, log_dir),
        _scan_metal_kernels(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
        ),
        _translate_directx_vulkan_frontier(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
            require_directx_toolchain=args.require_directx_toolchain,
            require_vulkan_toolchain=args.require_vulkan_toolchain,
        ),
        _check_arange_opengl(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
            require_opengl_toolchain=args.require_opengl_toolchain,
        ),
    ]
    return {
        "schema_version": 1,
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "sourceRoot": MLX_METAL_KERNEL_ROOT,
            "frontierSources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
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
        "--require-directx-toolchain",
        action="store_true",
        help="Fail unless the DirectX smoke checks run successfully.",
    )
    parser.add_argument(
        "--require-opengl-toolchain",
        action="store_true",
        help="Fail unless the OpenGL smoke checks run successfully.",
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
