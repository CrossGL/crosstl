#!/usr/bin/env python3
"""Run pinned MLX project-porting checks through the public CrossTL CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from crosstl.project import (
    VulkanComputeRuntime,
    build_project_test_runner_plan,
    build_runtime_artifact_manifest,
    execute_project_test_runner_plan,
    native_runtime_parity_adapters,
)
from crosstl.project.runtime_verification import (
    RuntimeParityAdapter,
    build_runtime_test_manifest,
    plan_runtime_test_manifest,
)

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
FULL_CORPUS_TRANSLATION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1376",
)
RUNTIME_READINESS_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1388",
    "https://github.com/CrossGL/crosstl/issues/1392",
    "https://github.com/CrossGL/crosstl/issues/1396",
)
RUNTIME_READINESS_ENTRY_POINTS = {
    "directx": "CSMain",
    "opengl": "main",
    "vulkan": "arangeuint32",
}
RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND = "mlx-arange-reference-runtime"
NATIVE_RUNTIME_EXECUTION_SCOPE = "native-runtime-execution-readiness"
RUNTIME_READINESS_DIAGNOSTIC_CODES = frozenset(
    (
        "project.runtime-test-manifest.entry-points-unavailable",
        "project.runtime-test-manifest.resource-bindings-unavailable",
        "project.runtime-test-manifest.dispatch-unavailable",
    )
)
RUNTIME_READINESS_PLAN_DIAGNOSTIC_CODES = frozenset(
    ("project.runtime-verification.resource-unbound",)
)
FULL_CORPUS_TRACKED_ISSUES = (
    *FRONTIER_VALIDATION_TRACKED_ISSUES,
    "https://github.com/CrossGL/crosstl/issues/1312",
    *FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
    *RUNTIME_READINESS_TRACKED_ISSUES,
)
RESOLVED_FRONTIER_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1454",
    "https://github.com/CrossGL/crosstl/issues/1453",
    "https://github.com/CrossGL/crosstl/issues/1452",
    "https://github.com/CrossGL/crosstl/issues/1354",
    "https://github.com/CrossGL/crosstl/issues/1362",
    "https://github.com/CrossGL/crosstl/issues/1394",
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
    "https://github.com/CrossGL/crosstl/issues/1355",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
            "--validate",
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
    toolchain_payload = payload
    if run_toolchains:
        toolchain_config_path = config_dir / "vulkan-frontier-toolchain.toml"
        toolchain_report_path = report_dir / "vulkan-frontier-toolchain.json"
        _write_project_config(
            toolchain_config_path,
            include=MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
            targets=("vulkan",),
            output_dir=_relpath(work_dir / "out-vulkan-frontier-toolchain", mlx_root),
        )
        _run_command(
            "validate-vulkan-frontier-toolchain",
            [
                python,
                "-m",
                "crosstl",
                "translate-project",
                str(mlx_root),
                "--config",
                str(toolchain_config_path),
                "--report",
                str(toolchain_report_path),
                "--run-toolchains",
            ],
            log_dir=log_dir,
        )
        toolchain_payload = _load_json(toolchain_report_path)
    toolchain_validation = toolchain_payload.get("validation", {})
    _require(
        isinstance(toolchain_validation, dict),
        "Vulkan toolchain validation must be an object",
    )
    toolchain_runs = toolchain_validation.get("toolchainRuns", [])
    _require(isinstance(toolchain_runs, list), "toolchainRuns must be a list")
    vulkan_runs = [
        run
        for run in toolchain_runs
        if isinstance(run, dict) and run.get("target") == "vulkan"
    ]
    if require_vulkan_toolchain and run_toolchains:
        vulkan_artifact_paths = {
            artifact.get("path")
            for artifact in toolchain_payload.get("artifacts", [])
            if isinstance(artifact, dict)
            and artifact.get("target") == "vulkan"
            and artifact.get("status") == "translated"
            and isinstance(artifact.get("path"), str)
        }
        validated_vulkan_paths = {
            run.get("path")
            for run in vulkan_runs
            if run.get("status") == "ok" and isinstance(run.get("path"), str)
        }
        _require(
            len(vulkan_artifact_paths) == frontier_count
            and vulkan_artifact_paths <= validated_vulkan_paths,
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


def _runtime_readiness_fixture(target: str) -> dict[str, Any]:
    entry_point = RUNTIME_READINESS_ENTRY_POINTS.get(target)
    _require(
        entry_point is not None,
        f"runtime readiness entry point is not configured for target: {target}",
    )
    return {
        "id": f"mlx-arange-{target}-runtime-readiness",
        "selector": {
            "source": MLX_ARANGE_SOURCE,
            "target": target,
        },
        "entryPoint": entry_point,
        "inputs": [
            {
                "name": "start",
                "kind": "scalar",
                "dtype": "uint32",
                "value": 0,
            },
            {
                "name": "step",
                "kind": "scalar",
                "dtype": "uint32",
                "value": 1,
            },
        ],
        "expectedOutputs": [
            {
                "name": "out",
                "kind": "buffer",
                "dtype": "uint32",
                "shape": [4],
                "values": [0, 1, 2, 3],
            }
        ],
        "runtimeAdapter": {
            "dispatch": {
                "globalSize": [4, 1, 1],
            }
        },
        "metadata": {
            "repository": "mlx",
            "source": MLX_ARANGE_SOURCE,
            "purpose": "runtime-readiness-metadata-probe",
        },
    }


def _runtime_readiness_fixture_metadata(targets: Sequence[str]) -> dict[str, Any]:
    return {
        "kind": "crosstl-project-runtime-fixture-metadata",
        "metadata": {
            "repository": "mlx",
            "fixtureSet": "reduced-arange-runtime-readiness",
            "scope": "artifact-execution-metadata-readiness",
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
        },
        "fixtures": [_runtime_readiness_fixture(target) for target in targets],
    }


def _runtime_fixture_execution_adapter_id(target: str) -> str:
    return f"mlx-arange-reference-{target}"


def _runtime_fixture_execution_metadata(targets: Sequence[str]) -> dict[str, Any]:
    metadata = _runtime_readiness_fixture_metadata(targets)
    metadata["metadata"] = {
        **metadata["metadata"],
        "scope": "reference-runtime-fixture-execution",
        "runtimeFixtureExecutionIncluded": True,
    }
    metadata["adapters"] = [
        {
            "id": _runtime_fixture_execution_adapter_id(target),
            "executor": _runtime_fixture_execution_adapter_id(target),
            "adapterKind": RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND,
            "platformRequirements": {"requiredTools": []},
            "metadata": {
                "target": target,
                "scope": "reference-runtime-fixture-execution",
            },
        }
        for target in targets
    ]
    metadata["fixtures"] = [
        {
            **fixture,
            "adapter": _runtime_fixture_execution_adapter_id(
                str(fixture["selector"]["target"])
            ),
        }
        for fixture in metadata["fixtures"]
        if isinstance(fixture.get("selector"), Mapping)
    ]
    return metadata


def _native_runtime_execution_adapter_id(target: str) -> str:
    return f"mlx-arange-native-{target}"


def _native_runtime_execution_metadata(targets: Sequence[str]) -> dict[str, Any]:
    metadata = _runtime_readiness_fixture_metadata(targets)
    metadata["metadata"] = {
        **metadata["metadata"],
        "scope": NATIVE_RUNTIME_EXECUTION_SCOPE,
        "nativeRuntimeExecutionIncluded": True,
    }
    metadata["adapters"] = [
        {
            "id": _native_runtime_execution_adapter_id(target),
            "executor": target,
            "adapterKind": f"{target}-native-runtime",
            "platformRequirements": {"requiredTools": []},
            "metadata": {
                "target": target,
                "scope": NATIVE_RUNTIME_EXECUTION_SCOPE,
            },
        }
        for target in targets
    ]
    metadata["fixtures"] = [
        {
            **fixture,
            "adapter": _native_runtime_execution_adapter_id(
                str(fixture["selector"]["target"])
            ),
        }
        for fixture in metadata["fixtures"]
        if isinstance(fixture.get("selector"), Mapping)
    ]
    return metadata


class MlxArangeReferenceRuntime(RuntimeParityAdapter):
    """Reference executor for MLX reduced arange runtime fixtures."""

    name = RUNTIME_FIXTURE_EXECUTION_ADAPTER_KIND

    def __init__(self, target: str):
        self.target = target

    def prepare_buffers(self, state):
        return dict(state.resource_values)

    def dispatch(self, state, prepared_buffers):
        start = _runtime_fixture_scalar(prepared_buffers.get("start"), default=0)
        step = _runtime_fixture_scalar(prepared_buffers.get("step"), default=1)
        output = state.request.fixture.expected_outputs[0]
        count = _runtime_fixture_output_count(state, output)
        return {output.name: [start + index * step for index in range(count)]}

    def collect_outputs(self, state, dispatch_result):
        outputs = {}
        for output in state.request.fixture.expected_outputs:
            values = dispatch_result.get(output.name, [])
            outputs[output.name] = {
                "dtype": output.dtype,
                "shape": list(output.shape),
                "values": values,
            }
        return outputs


def _runtime_fixture_scalar(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return default
        value = value[0]
    return int(value)


def _runtime_fixture_output_count(state: Any, output: Any) -> int:
    if output.shape:
        return int(output.shape[0])
    if isinstance(output.values, Sequence) and not isinstance(
        output.values, (str, bytes, bytearray)
    ):
        return len(output.values)
    dispatch = state.plan.dispatch
    if dispatch is not None and dispatch.global_size:
        return int(dispatch.global_size[0])
    return 1


def _runtime_fixture_execution_executors(targets: Sequence[str]) -> dict[str, Any]:
    return {
        _runtime_fixture_execution_adapter_id(target): MlxArangeReferenceRuntime(target)
        for target in targets
    }


def _diagnostics_by_code(diagnostics: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        code = diagnostic.get("code")
        if isinstance(code, str) and code:
            counts[code] += 1
    return dict(sorted(counts.items()))


def _runtime_plan_diagnostics(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics: list[Mapping[str, Any]] = []
    for test_case in plan.get("testCases", []):
        if not isinstance(test_case, Mapping):
            continue
        for diagnostic in test_case.get("diagnostics", []):
            if isinstance(diagnostic, Mapping):
                diagnostics.append(diagnostic)
    return diagnostics


def _runtime_report_diagnostics(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostics: list[Mapping[str, Any]] = []
    for result in report.get("results", []):
        if not isinstance(result, Mapping):
            continue
        for diagnostic in result.get("diagnostics", []):
            if isinstance(diagnostic, Mapping):
                diagnostics.append(diagnostic)
    runtime_report = report.get("runtimeTestReport")
    if isinstance(runtime_report, Mapping):
        diagnostics.extend(_runtime_report_diagnostics(runtime_report))
    return diagnostics


def _runtime_report_result_for_target(
    runtime_report: Mapping[str, Any], target: str
) -> Mapping[str, Any] | None:
    for result in runtime_report.get("results", []):
        if not isinstance(result, Mapping):
            continue
        artifact = result.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("target") == target:
            return result
        fixture = result.get("fixture")
        if isinstance(fixture, Mapping):
            selector = fixture.get("selector")
            if isinstance(selector, Mapping) and selector.get("target") == target:
                return result
    return None


def _error_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.get("severity") == "error"
    ]


def _execute_runtime_fixtures_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    runtime_artifact_manifest_path: Path,
    targets: Sequence[str],
) -> dict[str, Any]:
    metadata_path = report_dir / f"{name}.runtime-fixture-execution-metadata.json"
    manifest_path = report_dir / f"{name}.runtime-fixture-execution-manifest.json"
    plan_path = report_dir / f"{name}.runtime-fixture-execution-plan.json"
    report_path = report_dir / f"{name}.runtime-fixture-execution-report.json"
    metadata = _runtime_fixture_execution_metadata(targets)
    _write_json(metadata_path, metadata)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = build_project_test_runner_plan(
        runtime_artifact_manifest_path,
        manifest,
        selected_targets=targets,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    report = execute_project_test_runner_plan(
        plan,
        project_root=mlx_root,
        runtime_executors=_runtime_fixture_execution_executors(targets),
    )
    _write_json(report_path, report)
    project_runner_summary = report.get("summary", {})
    _require(
        isinstance(project_runner_summary, dict),
        "runtime fixture execution project-runner summary missing",
    )
    runtime_report = report.get("runtimeTestReport", {})
    _require(
        isinstance(runtime_report, Mapping),
        "runtime fixture execution runtime report missing",
    )
    summary = runtime_report.get("summary", {})
    _require(isinstance(summary, dict), "runtime fixture execution summary missing")
    diagnostics = _runtime_report_diagnostics(report)
    diagnostics_by_code = _diagnostics_by_code(diagnostics)
    failed_count = int(summary.get("failedCount", 0))
    skipped_count = int(summary.get("skippedCount", 0))
    status = "passed" if failed_count == 0 and skipped_count == 0 else "failed"
    if status == "failed" and RUNTIME_READINESS_TRACKED_ISSUES:
        status = "blocked-by-tracked-issues"
    return {
        "name": f"{name}-runtime-fixture-execution",
        "status": status,
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "projectTestRunnerPlan": _relpath(plan_path, mlx_root),
        "projectTestRunnerReport": _relpath(report_path, mlx_root),
        "targets": list(targets),
        "summary": summary,
        "projectRunnerSummary": project_runner_summary,
        "diagnosticsByCode": diagnostics_by_code,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeIntegrationIncluded": False,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _execute_native_runtime_fixtures_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    runtime_artifact_manifest_path: Path,
    targets: Sequence[str],
    require_vulkan_native_runtime: bool,
) -> dict[str, Any]:
    metadata_path = report_dir / f"{name}.native-runtime-execution-metadata.json"
    manifest_path = report_dir / f"{name}.native-runtime-execution-manifest.json"
    plan_path = report_dir / f"{name}.native-runtime-execution-plan.json"
    report_path = report_dir / f"{name}.native-runtime-execution-report.json"
    metadata = _native_runtime_execution_metadata(targets)
    _write_json(metadata_path, metadata)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = build_project_test_runner_plan(
        runtime_artifact_manifest_path,
        manifest,
        selected_targets=targets,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    report = execute_project_test_runner_plan(
        plan,
        project_root=mlx_root,
        runtime_executors=native_runtime_parity_adapters(
            runtimes={"vulkan": VulkanComputeRuntime()}
        ),
    )
    _write_json(report_path, report)
    project_runner_summary = report.get("summary", {})
    _require(
        isinstance(project_runner_summary, dict),
        "native runtime execution project-runner summary missing",
    )
    runtime_report = report.get("runtimeTestReport", {})
    _require(
        isinstance(runtime_report, Mapping),
        "native runtime execution runtime report missing",
    )
    summary = runtime_report.get("summary", {})
    _require(isinstance(summary, dict), "native runtime execution summary missing")
    diagnostics = _runtime_report_diagnostics(report)
    diagnostics_by_code = _diagnostics_by_code(diagnostics)
    failed_count = int(summary.get("failedCount", 0))
    passed_count = int(summary.get("passedCount", 0))
    unavailable_count = int(summary.get("unavailableCount", 0))
    skipped_count = int(summary.get("skippedCount", 0))
    if require_vulkan_native_runtime:
        vulkan_result = _runtime_report_result_for_target(runtime_report, "vulkan")
        _require(
            isinstance(vulkan_result, Mapping)
            and vulkan_result.get("status") == "passed",
            "Vulkan native runtime execution was required for the MLX arange fixture",
        )
    status = "passed"
    if failed_count:
        status = "blocked-by-tracked-issues"
    elif unavailable_count or skipped_count:
        status = "blocked-by-runtime-driver"
    return {
        "name": f"{name}-native-runtime-execution",
        "status": status,
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "projectTestRunnerPlan": _relpath(plan_path, mlx_root),
        "projectTestRunnerReport": _relpath(report_path, mlx_root),
        "targets": list(targets),
        "summary": summary,
        "passedCount": passed_count,
        "projectRunnerSummary": project_runner_summary,
        "diagnosticsByCode": diagnostics_by_code,
        "nativeRuntimeExecutionIncluded": True,
        "runtimeIntegrationIncluded": False,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _plan_runtime_readiness_for_report(
    *,
    mlx_root: Path,
    report_dir: Path,
    name: str,
    artifact_report: Path,
    targets: Sequence[str],
    require_vulkan_native_runtime: bool,
) -> dict[str, Any]:
    _require(
        artifact_report.is_file(),
        f"runtime readiness artifact report is missing: {artifact_report}",
    )
    metadata_path = report_dir / f"{name}.fixture-metadata.json"
    runtime_artifact_manifest_path = (
        report_dir / f"{name}.runtime-artifact-manifest.json"
    )
    manifest_path = report_dir / f"{name}.runtime-test-manifest.json"
    plan_path = report_dir / f"{name}.runtime-test-plan.json"
    metadata = _runtime_readiness_fixture_metadata(targets)
    _write_json(metadata_path, metadata)
    runtime_artifact_manifest = build_runtime_artifact_manifest(artifact_report)
    _write_json(runtime_artifact_manifest_path, runtime_artifact_manifest)
    manifest = build_runtime_test_manifest(
        runtime_artifact_manifest_path,
        metadata_path,
        project_root=mlx_root,
    )
    _write_json(manifest_path, manifest)
    plan = plan_runtime_test_manifest(
        runtime_artifact_manifest_path,
        manifest,
        project_root=mlx_root,
    )
    _write_json(plan_path, plan)
    runtime_fixture_execution = _execute_runtime_fixtures_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name=name,
        runtime_artifact_manifest_path=runtime_artifact_manifest_path,
        targets=targets,
    )
    native_runtime_execution = _execute_native_runtime_fixtures_for_report(
        mlx_root=mlx_root,
        report_dir=report_dir,
        name=name,
        runtime_artifact_manifest_path=runtime_artifact_manifest_path,
        targets=targets,
        require_vulkan_native_runtime=require_vulkan_native_runtime,
    )

    runtime_artifact_diagnostics_by_code = _diagnostics_by_code(
        runtime_artifact_manifest.get("runtimeDiagnostics", [])
    )
    diagnostic_counts = manifest.get("diagnosticCounts", {})
    _require(
        isinstance(diagnostic_counts, dict),
        "runtime readiness diagnostic counts must be an object",
    )
    _require(
        diagnostic_counts.get("error", 0) == 0,
        "runtime readiness manifest reported fixture or artifact selection errors",
    )
    diagnostics_by_code = _diagnostics_by_code(manifest.get("diagnostics", []))
    metadata_gap_codes = sorted(
        code
        for code in diagnostics_by_code
        if code in RUNTIME_READINESS_DIAGNOSTIC_CODES
    )
    plan_diagnostics = _runtime_plan_diagnostics(plan)
    plan_diagnostics_by_code = _diagnostics_by_code(plan_diagnostics)
    plan_blocker_codes = sorted(
        code
        for code in _diagnostics_by_code(_error_diagnostics(plan_diagnostics))
        if code in RUNTIME_READINESS_PLAN_DIAGNOSTIC_CODES
    )
    if metadata_gap_codes:
        _require(
            RUNTIME_READINESS_TRACKED_ISSUES,
            "runtime readiness manifest reported artifact execution metadata gaps "
            "without tracked issue references",
        )
    if plan_blocker_codes:
        _require(
            RUNTIME_READINESS_TRACKED_ISSUES,
            "runtime readiness plan reported adapter setup blockers without "
            "tracked issue references",
        )
    status = (
        "blocked-by-tracked-issues"
        if metadata_gap_codes or plan_blocker_codes
        else "planned"
    )
    plan_summary = plan.get("summary", {})
    _require(isinstance(plan_summary, dict), "runtime readiness plan summary missing")
    manifest_summary = manifest.get("summary", {})
    _require(
        isinstance(manifest_summary, dict),
        "runtime readiness manifest summary missing",
    )
    return {
        "name": name,
        "status": status,
        "artifactReport": _relpath(artifact_report, mlx_root),
        "fixtureMetadata": _relpath(metadata_path, mlx_root),
        "runtimeArtifactManifest": _relpath(runtime_artifact_manifest_path, mlx_root),
        "runtimeTestManifest": _relpath(manifest_path, mlx_root),
        "runtimeTestPlan": _relpath(plan_path, mlx_root),
        "targets": list(targets),
        "testCount": manifest_summary.get("testCount", 0),
        "runtimeArtifactSummary": runtime_artifact_manifest.get("summary", {}),
        "runtimeArtifactDiagnosticCounts": runtime_artifact_manifest.get(
            "runtimeDiagnosticCounts", {}
        ),
        "runtimeArtifactDiagnosticsByCode": runtime_artifact_diagnostics_by_code,
        "diagnosticCounts": diagnostic_counts,
        "diagnosticsByCode": diagnostics_by_code,
        "runtimePlanDiagnosticsByCode": plan_diagnostics_by_code,
        "runtimePlanSummary": plan_summary,
        "metadataGapCodes": metadata_gap_codes,
        "planBlockerCodes": plan_blocker_codes,
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeFixtureExecution": runtime_fixture_execution,
        "nativeRuntimeExecutionIncluded": True,
        "nativeRuntimeExecution": native_runtime_execution,
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
    }


def _plan_reduced_runtime_readiness(
    mlx_root: Path,
    report_dir: Path,
    *,
    require_vulkan_native_runtime: bool,
) -> dict[str, Any]:
    reports = [
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="directx-vulkan-runtime-readiness",
            artifact_report=report_dir / "directx-vulkan-frontier.json",
            targets=("directx", "vulkan"),
            require_vulkan_native_runtime=require_vulkan_native_runtime,
        ),
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="opengl-runtime-readiness",
            artifact_report=report_dir / "arange-opengl.json",
            targets=("opengl",),
            require_vulkan_native_runtime=False,
        ),
    ]
    status = (
        "blocked-by-tracked-issues"
        if any(report["status"] == "blocked-by-tracked-issues" for report in reports)
        else "planned"
    )
    diagnostics_by_code: Counter[str] = Counter()
    runtime_artifact_diagnostics_by_code: Counter[str] = Counter()
    runtime_plan_diagnostics_by_code: Counter[str] = Counter()
    runtime_fixture_execution_by_status: Counter[str] = Counter()
    runtime_fixture_execution_summary: Counter[str] = Counter()
    native_runtime_execution_by_status: Counter[str] = Counter()
    native_runtime_execution_summary: Counter[str] = Counter()
    for report in reports:
        diagnostics_by_code.update(report.get("diagnosticsByCode", {}))
        runtime_artifact_diagnostics_by_code.update(
            report.get("runtimeArtifactDiagnosticsByCode", {})
        )
        runtime_plan_diagnostics_by_code.update(
            report.get("runtimePlanDiagnosticsByCode", {})
        )
        runtime_fixture_execution = report.get("runtimeFixtureExecution", {})
        if isinstance(runtime_fixture_execution, Mapping):
            runtime_fixture_execution_by_status.update(
                [str(runtime_fixture_execution.get("status", "unknown"))]
            )
            execution_summary = runtime_fixture_execution.get("summary", {})
            if isinstance(execution_summary, Mapping):
                for key in (
                    "fixtureCount",
                    "resultCount",
                    "passedCount",
                    "skippedCount",
                    "unavailableCount",
                    "translationFailedCount",
                    "runtimeFailedCount",
                    "comparisonFailedCount",
                    "failedCount",
                ):
                    if key in execution_summary:
                        runtime_fixture_execution_summary[key] += int(
                            execution_summary.get(key, 0)
                        )
        native_runtime_execution = report.get("nativeRuntimeExecution", {})
        if isinstance(native_runtime_execution, Mapping):
            native_runtime_execution_by_status.update(
                [str(native_runtime_execution.get("status", "unknown"))]
            )
            execution_summary = native_runtime_execution.get("summary", {})
            if isinstance(execution_summary, Mapping):
                for key in (
                    "fixtureCount",
                    "passedCount",
                    "skippedCount",
                    "unavailableCount",
                    "translationFailedCount",
                    "runtimeFailedCount",
                    "comparisonFailedCount",
                    "failedCount",
                ):
                    if key in execution_summary:
                        native_runtime_execution_summary[key] += int(
                            execution_summary.get(key, 0)
                        )
    return {
        "name": "runtime-readiness",
        "status": status,
        "reports": reports,
        "targets": ["directx", "opengl", "vulkan"],
        "testCount": sum(int(report.get("testCount", 0)) for report in reports),
        "diagnosticsByCode": dict(sorted(diagnostics_by_code.items())),
        "runtimeArtifactDiagnosticsByCode": dict(
            sorted(runtime_artifact_diagnostics_by_code.items())
        ),
        "runtimePlanDiagnosticsByCode": dict(
            sorted(runtime_plan_diagnostics_by_code.items())
        ),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeFixtureExecutionIncluded": True,
        "runtimeFixtureExecutionByStatus": dict(
            sorted(runtime_fixture_execution_by_status.items())
        ),
        "runtimeFixtureExecutionSummary": dict(
            sorted(runtime_fixture_execution_summary.items())
        ),
        "nativeRuntimeExecutionIncluded": True,
        "nativeRuntimeExecutionByStatus": dict(
            sorted(native_runtime_execution_by_status.items())
        ),
        "nativeRuntimeExecutionSummary": dict(
            sorted(native_runtime_execution_summary.items())
        ),
        "trackedRuntimeIssues": list(RUNTIME_READINESS_TRACKED_ISSUES),
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
        checks.append(
            _translate_directx_vulkan_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_vulkan_toolchain=args.require_vulkan_toolchain,
            )
        )
        checks.append(
            _check_arange_opengl(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        checks.append(
            _plan_reduced_runtime_readiness(
                mlx_root,
                report_dir,
                require_vulkan_native_runtime=args.require_vulkan_native_runtime,
            )
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
            "runtimeReadinessIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "runtimeFixtureExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "nativeRuntimeExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
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
        "--require-vulkan-native-runtime",
        action="store_true",
        help="Fail unless the MLX-generated Vulkan arange fixture executes natively.",
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
