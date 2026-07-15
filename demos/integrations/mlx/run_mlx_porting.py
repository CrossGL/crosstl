#!/usr/bin/env python3
"""Run pinned MLX project-porting checks through the public CrossTL CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
MLX_ARANGE_SOURCE = "mlx/backend/metal/kernels/arange.metal"
MLX_ARG_REDUCE_SOURCE = "mlx/backend/metal/kernels/arg_reduce.metal"
MLX_BINARY_TWO_SOURCE = "mlx/backend/metal/kernels/binary_two.metal"
MLX_FENCE_SOURCE = "mlx/backend/metal/kernels/fence.metal"
MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT = 3
MLX_GEMV_SOURCE = "mlx/backend/metal/kernels/gemv.metal"
MLX_LAYER_NORM_SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
MLX_LOGSUMEXP_SOURCE = "mlx/backend/metal/kernels/logsumexp.metal"
MLX_METAL_ROUNDTRIP_SOURCE = MLX_FENCE_SOURCE
MLX_RMS_NORM_SOURCE = "mlx/backend/metal/kernels/rms_norm.metal"
MLX_ROPE_SOURCE = "mlx/backend/metal/kernels/rope.metal"
MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE = (
    "mlx/backend/metal/kernels/scaled_dot_product_attention.metal"
)
MLX_SOFTMAX_SOURCE = "mlx/backend/metal/kernels/softmax.metal"
MLX_TERNARY_SOURCE = "mlx/backend/metal/kernels/ternary.metal"
REFERENCE_ACCESSOR_FIXTURE_NAME = "reference_accessor_lvalue.metal"
REFERENCE_ACCESSOR_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / REFERENCE_ACCESSOR_FIXTURE_NAME
)
REFERENCE_ACCESSOR_TARGETS = ("directx", "opengl")
REFERENCE_ACCESSOR_SENTINEL = "73.25"
REFERENCE_ACCESSOR_DXC_ENTRY_POINT = "CSMain"
MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES = (
    MLX_ARG_REDUCE_SOURCE,
    MLX_BINARY_TWO_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    MLX_RMS_NORM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
    MLX_TERNARY_SOURCE,
)
MLX_DIRECTX_VULKAN_FRONTIER_SOURCES = (
    MLX_ARANGE_SOURCE,
    MLX_ARG_REDUCE_SOURCE,
    MLX_BINARY_TWO_SOURCE,
    MLX_LAYER_NORM_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    "mlx/backend/metal/kernels/random.metal",
    MLX_RMS_NORM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
    MLX_TERNARY_SOURCE,
)
MLX_CLEAN_REDUCED_FRONTIER_SOURCES = tuple(
    dict.fromkeys(
        (
            *MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
            *MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES,
        )
    )
)
MLX_BLOCKED_REDUCED_FRONTIER_SOURCES = (MLX_FENCE_SOURCE,)
MLX_REDUCED_FRONTIER_SOURCES = tuple(
    dict.fromkeys(
        sorted(
            (
                *MLX_CLEAN_REDUCED_FRONTIER_SOURCES,
                *MLX_BLOCKED_REDUCED_FRONTIER_SOURCES,
            )
        )
    )
)
# Subset of the clean frontier gated by DXC on Windows. The DirectX/Vulkan frontier
# is still translated in full (and all Vulkan artifacts are spirv-val'd), while
# the remaining kernels stay outside the DirectX compile gate under tracked gaps.
MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES = (
    MLX_ARANGE_SOURCE,
    MLX_ARG_REDUCE_SOURCE,
    MLX_LAYER_NORM_SOURCE,
    MLX_LOGSUMEXP_SOURCE,
    MLX_RMS_NORM_SOURCE,
    MLX_ROPE_SOURCE,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
    MLX_SOFTMAX_SOURCE,
    MLX_TERNARY_SOURCE,
)
# Pinned generated compute entries compiled by official DXC v1.9.2602.24.
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS = {
    MLX_ARANGE_SOURCE: 11,
    MLX_ARG_REDUCE_SOURCE: 24,
    MLX_LAYER_NORM_SOURCE: 12,
    MLX_LOGSUMEXP_SOURCE: 6,
    MLX_RMS_NORM_SOURCE: 12,
    MLX_ROPE_SOURCE: 18,
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: 42,
    MLX_SOFTMAX_SOURCE: 10,
    MLX_TERNARY_SOURCE: 212,
}
MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT = sum(
    MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS.values()
)
MLX_FRONTIER_SPECIALIZATION_CONSTANTS = {
    "1": False,
    "2": False,
    "3": False,
    "20": False,
    "21": False,
    "22": False,
    "23": False,
    "24": False,
    "25": False,
    "26": 1,
}
MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS = {
    MLX_RMS_NORM_SOURCE: {"has_w": 20},
    MLX_ROPE_SOURCE: {"forward": 1, "traditional": 2, "hs_transpose": 3},
    MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE: {
        "has_mask": 20,
        "query_transposed": 21,
        "do_causal": 22,
        "bool_mask": 23,
        "float_mask": 24,
        "has_sinks": 25,
        "blocks": 26,
    },
}
MLX_FENCE_REQUESTED_CONTRACT = {
    "memoryFlags": ["mem_device"],
    "memoryOrder": "memory_order_seq_cst",
    "threadScope": "thread_scope_system",
}
MLX_FENCE_TARGET_CONTRACTS = {
    "directx": {
        "diagnosticCode": "project.translate.directx-atomic-fence-unsupported",
        "missingCapability": "directx.atomic-thread-fence-contract-lowering",
        "targetDescription": "HLSL",
    },
    "opengl": {
        "diagnosticCode": "project.translate.opengl-atomic-fence-unsupported",
        "missingCapability": "opengl.atomic-thread-fence-contract-lowering",
        "targetDescription": "OpenGL GLSL",
    },
    "vulkan": {
        "diagnosticCode": "project.translate.vulkan-atomic-fence-unsupported",
        "missingCapability": "spirv.atomic-thread-fence-contract-lowering",
        "targetDescription": "Vulkan SPIR-V",
    },
}
EXPECTED_METAL_KERNEL_COUNT = 40
FULL_CORPUS_TARGETS = ("directx", "opengl", "vulkan")
FULL_CORPUS_EXPECTED_ARTIFACT_COUNT = EXPECTED_METAL_KERNEL_COUNT * len(
    FULL_CORPUS_TARGETS
)
FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT = len(MLX_FENCE_TARGET_CONTRACTS)
FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT = (
    FULL_CORPUS_EXPECTED_ARTIFACT_COUNT - FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
)
FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS = 4096
FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK = 131072
FULL_CORPUS_TRANSLATION_TIMEOUT_SECONDS = 900
GEMV_MAX_TEMPLATE_SPECIALIZATIONS = 4096
GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK = 2097152
GEMV_EXPECTED_SPECIALIZATION_COUNT = 225
GEMV_EXPECTED_ENTRY_POINT_COUNT = 224
REDUCED_FRONTIER_MODE = "reduced-frontier"
FULL_CORPUS_MODE = "full-corpus"
FRONTIER_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
FULL_CORPUS_TRANSLATION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1376",
    "https://github.com/CrossGL/crosstl/issues/1676",
    "https://github.com/CrossGL/crosstl/issues/1476",
    "https://github.com/CrossGL/crosstl/issues/1479",
    "https://github.com/CrossGL/crosstl/issues/1490",
    "https://github.com/CrossGL/crosstl/issues/1544",
    "https://github.com/CrossGL/crosstl/issues/1546",
    "https://github.com/CrossGL/crosstl/issues/1554",
    "https://github.com/CrossGL/crosstl/issues/1559",
    "https://github.com/CrossGL/crosstl/issues/1562",
    "https://github.com/CrossGL/crosstl/issues/1659",
    "https://github.com/CrossGL/crosstl/issues/1669",
    "https://github.com/CrossGL/crosstl/issues/1671",
    "https://github.com/CrossGL/crosstl/issues/1672",
)
RUNTIME_READINESS_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1388",
    "https://github.com/CrossGL/crosstl/issues/1471",
)
VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES: tuple[str, ...] = ()
FENCE_CONTRACT_TRACKED_ISSUES = ("https://github.com/CrossGL/crosstl/issues/1537",)
VULKAN_GEMV_REPORTING_TRACKED_ISSUE = "https://github.com/CrossGL/crosstl/issues/1517"
FULL_CORPUS_SEMANTIC_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1491",
)
METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1660",
)
OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES: tuple[str, ...] = ()
OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1538",
)
RUNTIME_READINESS_ENTRY_POINTS = {
    "directx": "CSMain",
    "opengl": "main",
    "vulkan": "arangeuint32",
}
RUNTIME_READINESS_DEFAULT_VARIANTS = {
    "directx": "uint8",
    "opengl": "uint32",
    "vulkan": "uint32",
}
ARANGE_RUNTIME_VARIANTS = {
    "uint8": {
        "start": 3,
        "step": 2,
        "expected": [3, 5, 7, 9],
    },
    "uint32": {
        "start": 300,
        "step": 17,
        "expected": [300, 317, 334, 351],
    },
    "int32": {
        "start": -3,
        "step": 2,
        "expected": [-3, -1, 1, 3],
    },
    "float32": {
        "start": 1.5,
        "step": 0.25,
        "expected": [1.5, 1.75, 2.0, 2.25],
    },
}
VULKAN_ARANGE_RUNTIME_VARIANTS = ("uint32", "int32", "float32")
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
    "https://github.com/CrossGL/crosstl/issues/1670",
    *FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
    *OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES,
    *OPENGL_SCALED_DOT_PRODUCT_ATTENTION_TRACKED_ISSUES,
    *RUNTIME_READINESS_TRACKED_ISSUES,
    *FENCE_CONTRACT_TRACKED_ISSUES,
    *VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES,
    VULKAN_GEMV_REPORTING_TRACKED_ISSUE,
    *FULL_CORPUS_SEMANTIC_TRACKED_ISSUES,
    *METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES,
)
RESOLVED_FRONTIER_ISSUES = (
    "https://github.com/CrossGL/crosstl/issues/1695",
    "https://github.com/CrossGL/crosstl/issues/1667",
    "https://github.com/CrossGL/crosstl/issues/1668",
    "https://github.com/CrossGL/crosstl/issues/1661",
    "https://github.com/CrossGL/crosstl/issues/1573",
    "https://github.com/CrossGL/crosstl/issues/1555",
    "https://github.com/CrossGL/crosstl/issues/1561",
    "https://github.com/CrossGL/crosstl/issues/1551",
    "https://github.com/CrossGL/crosstl/issues/1498",
    "https://github.com/CrossGL/crosstl/issues/1535",
    "https://github.com/CrossGL/crosstl/issues/1489",
    "https://github.com/CrossGL/crosstl/issues/1504",
    "https://github.com/CrossGL/crosstl/issues/1503",
    "https://github.com/CrossGL/crosstl/issues/1502",
    "https://github.com/CrossGL/crosstl/issues/1500",
    "https://github.com/CrossGL/crosstl/issues/1454",
    "https://github.com/CrossGL/crosstl/issues/1453",
    "https://github.com/CrossGL/crosstl/issues/1452",
    "https://github.com/CrossGL/crosstl/issues/1354",
    "https://github.com/CrossGL/crosstl/issues/1362",
    "https://github.com/CrossGL/crosstl/issues/1394",
    "https://github.com/CrossGL/crosstl/issues/1396",
    "https://github.com/CrossGL/crosstl/issues/1392",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe_native_metal_toolchain(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {
            "status": "not-applicable",
            "platform": sys.platform,
            "reason": "native Metal validation requires macOS",
        }

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return {
            "status": "toolchain-unavailable",
            "platform": sys.platform,
            "reason": "xcrun is not installed",
        }

    native_dir = work_dir / "native-metal"
    native_dir.mkdir(parents=True, exist_ok=True)
    source_path = native_dir / "toolchain-probe.metal"
    output_path = native_dir / "toolchain-probe.air"
    output_path.unlink(missing_ok=True)
    source_path.write_text(
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void crosstl_mlx_probe() {}\n",
        encoding="utf-8",
    )
    result = _run_command(
        "probe-native-metal-toolchain",
        [
            xcrun,
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    available = result.returncode == 0 and output_path.is_file()
    return {
        "status": "available" if available else "toolchain-unavailable",
        "platform": sys.platform,
        "xcrun": xcrun,
        "probeSource": _relpath(source_path, mlx_root),
        "probeArtifact": (
            _relpath(output_path, mlx_root) if output_path.is_file() else None
        ),
        "returncode": result.returncode,
        "stdout": _relpath(result.stdout_path, mlx_root),
        "stderr": _relpath(result.stderr_path, mlx_root),
        "reason": (
            None
            if available
            else "a minimal Metal compilation did not produce an AIR artifact"
        ),
    }


def _validate_native_metal_artifact(
    *,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    required: bool,
) -> dict[str, Any]:
    probe = _probe_native_metal_toolchain(mlx_root, work_dir, log_dir)
    if probe["status"] != "available":
        _require(
            not required,
            "native Metal validation was required, but a usable macOS Metal "
            f"toolchain was not available ({probe['reason']})",
        )
        return {
            **probe,
            "required": required,
            "artifactCompiled": False,
        }

    output_path = work_dir / "native-metal" / "mlx-fence-roundtrip.air"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = _run_command(
        "validate-metal-roundtrip-native",
        [
            str(probe["xcrun"]),
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        result.returncode == 0,
        "native Metal validation failed for the generated MLX fence round-trip "
        f"artifact; see {result.stdout_path} and {result.stderr_path}",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "native Metal validation did not produce the expected AIR artifact",
    )
    return {
        **probe,
        "status": "validated",
        "required": required,
        "artifactCompiled": True,
        "sourceArtifact": _relpath(artifact_path, mlx_root),
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compileStdout": _relpath(result.stdout_path, mlx_root),
        "compileStderr": _relpath(result.stderr_path, mlx_root),
    }


def _write_project_config(
    path: Path,
    *,
    include: str | Sequence[str],
    targets: Sequence[str],
    output_dir: str,
    specialization_constants: Mapping[str, bool | int | float] | None = None,
    metal_source_options: Mapping[str, int] | None = None,
    metal_target_options: Mapping[str, Mapping[str, int]] | None = None,
    entry_points: Mapping[str, str] | None = None,
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
    if entry_points:
        lines.append("[project.entry_points]")
        for source, entry_point in entry_points.items():
            lines.append(f"{json.dumps(source)} = {json.dumps(entry_point)}")
        lines.append("")
    if specialization_constants:
        lines.append("[project.specialization_constants]")
        for selector, value in specialization_constants.items():
            lines.append(f"{json.dumps(str(selector))} = {json.dumps(value)}")
        lines.append("")
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


def _write_reference_accessor_project_config(path: Path, output_dir: str) -> None:
    target_list = ", ".join(json.dumps(target) for target in REFERENCE_ACCESSOR_TARGETS)
    lines = [
        "[project]",
        'source_roots = ["."]',
        f'include = ["{REFERENCE_ACCESSOR_FIXTURE_NAME}"]',
        f"targets = [{target_list}]",
        f'output_dir = "{output_dir}"',
        "",
        "[project.sources]",
        '"*.metal" = "metal"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_mlx_checkout(mlx_root: Path, python: str, log_dir: Path) -> dict[str, Any]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    _require(
        (mlx_root / MLX_ARANGE_SOURCE).is_file(),
        f"MLX Metal frontier source is missing: {MLX_ARANGE_SOURCE}",
    )
    for source in MLX_REDUCED_FRONTIER_SOURCES:
        _require(
            (mlx_root / source).is_file(),
            f"MLX Metal frontier source is missing: {source}",
        )
    _require(
        (mlx_root / MLX_GEMV_SOURCE).is_file(),
        f"MLX GEMV source is missing: {MLX_GEMV_SOURCE}",
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
    for source in MLX_REDUCED_FRONTIER_SOURCES:
        _require(source in unit_paths, f"{source} was not scanned")
    return {
        "name": "metal-kernel-scan",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "unitCount": summary.get("unitCount"),
        "includeDependencyCount": summary.get("includeDependencyCount"),
        "targets": ["directx", "opengl", "vulkan"],
    }


def _check_metal_roundtrip(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_metal_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "metal-roundtrip.toml"
    report_path = report_dir / "metal-roundtrip.json"
    output_dir = work_dir / "out-metal-roundtrip"
    _write_project_config(
        config_path,
        include=MLX_METAL_ROUNDTRIP_SOURCE,
        targets=("metal",),
        output_dir=_relpath(output_dir, mlx_root),
    )
    _run_command(
        "translate-metal-roundtrip",
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
    _require(isinstance(summary, dict), "Metal round-trip summary must be an object")
    _require(
        summary.get("unitCount") == 1,
        "Metal round-trip translation must scan exactly one pinned MLX source",
    )
    _require(
        summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0,
        "Metal round-trip translation must emit one clean artifact",
    )
    diagnostic_counts = summary.get("diagnosticCounts", {})
    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostics, list), "Metal round-trip diagnostics must be a list"
    )
    allowed_unavailable = [
        diagnostic
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
        and diagnostic.get("severity") == "warning"
        and diagnostic.get("code") == "project.validate.toolchain-unavailable"
        and diagnostic.get("target") == "metal"
        and not require_metal_toolchain
    ]
    unexpected_diagnostics = [
        diagnostic
        for diagnostic in diagnostics
        if isinstance(diagnostic, dict)
        and diagnostic.get("severity") in {"error", "warning"}
        and diagnostic not in allowed_unavailable
    ]
    _require(
        isinstance(diagnostic_counts, dict)
        and diagnostic_counts.get("error", 0) == 0
        and diagnostic_counts.get("warning", 0) == len(allowed_unavailable)
        and not unexpected_diagnostics,
        "Metal round-trip translation reported unexpected diagnostics",
    )

    artifacts = payload.get("artifacts", [])
    _require(isinstance(artifacts, list), "Metal round-trip artifacts must be a list")
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and item.get("source") == MLX_METAL_ROUNDTRIP_SOURCE
            and item.get("sourceBackend") == "metal"
            and item.get("target") == "metal"
        ),
        None,
    )
    _require(isinstance(artifact, dict), "Metal round-trip artifact is missing")
    _require(
        artifact.get("status") == "translated",
        "Metal round-trip artifact was not translated",
    )
    expected_path = output_dir / "metal" / MLX_METAL_ROUNDTRIP_SOURCE
    expected_report_path = _relpath(expected_path, mlx_root)
    _require(
        artifact.get("path") == expected_report_path,
        "Metal round-trip artifact path does not match the bounded project output",
    )
    _require(
        artifact.get("provenance")
        == {
            "pipeline": "single-file-translate",
            "intermediate": "crossgl",
        },
        "Metal round-trip artifact did not traverse the CrossGL project pipeline",
    )
    _require(
        expected_path.is_file() and expected_path.stat().st_size > 0,
        f"Metal round-trip artifact is missing or empty: {expected_report_path}",
    )

    source_path = mlx_root / MLX_METAL_ROUNDTRIP_SOURCE
    source_hash = artifact.get("sourceHash", {})
    generated_hash = artifact.get("generatedHash", {})
    _require(
        source_hash.get("algorithm") == "sha256"
        and source_hash.get("value") == _sha256(source_path),
        "Metal round-trip source hash does not match the pinned MLX source",
    )
    _require(
        generated_hash.get("algorithm") == "sha256"
        and generated_hash.get("value") == _sha256(expected_path),
        "Metal round-trip generated hash does not match the emitted artifact",
    )
    _require(
        artifact.get("sourceSizeBytes") == source_path.stat().st_size
        and artifact.get("generatedSizeBytes") == expected_path.stat().st_size,
        "Metal round-trip source or generated byte-size metadata is inconsistent",
    )
    _require(
        source_hash.get("value") != generated_hash.get("value"),
        "Metal round-trip unexpectedly copied the source without translation",
    )

    generated = expected_path.read_text(encoding="utf-8")
    for fragment in (
        "#include <metal_stdlib>",
        "kernel void input_coherent",
        "kernel void fence_update",
        "kernel void fence_wait",
        "[[buffer(0)]]",
        "[[thread_position_in_grid]]",
        "metal::mem_flags::mem_device",
        "metal::memory_order_seq_cst",
        "metal::thread_scope_system",
    ):
        _require(
            fragment in generated,
            f"Metal round-trip artifact is missing expected generated form: {fragment}",
        )
    _require(
        generated.count("metal::atomic_thread_fence(")
        == MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
        "Metal round-trip artifact did not preserve every source atomic fence",
    )
    for forbidden in ("threadgroup_barrier(", "unsupported", "fallback"):
        _require(
            forbidden not in generated,
            f"Metal round-trip artifact contains forbidden generated form: {forbidden}",
        )

    validation = payload.get("validation", {})
    _require(isinstance(validation, dict), "Metal round-trip validation is missing")
    validation_summary = validation.get("summary", {})
    _require(
        isinstance(validation_summary, dict)
        and validation_summary.get("artifactCount") == 1
        and validation_summary.get("okCount") == 1
        and validation_summary.get("failedCount") == 0,
        "Metal round-trip generated-artifact validation was not clean",
    )
    validation_artifact = next(
        (
            item
            for item in validation.get("artifacts", [])
            if isinstance(item, dict) and item.get("path") == expected_report_path
        ),
        None,
    )
    _require(
        isinstance(validation_artifact, dict)
        and validation_artifact.get("status") == "ok"
        and validation_artifact.get("sourceHashStatus") == "ok"
        and validation_artifact.get("generatedHashStatus") == "ok"
        and validation_artifact.get("sourceSizeStatus") == "ok"
        and validation_artifact.get("generatedSizeStatus") == "ok"
        and validation_artifact.get("sourceMapStatus") == "ok"
        and validation_artifact.get("sourceRemapStatus") == "ok",
        "Metal round-trip artifact metadata did not pass project validation",
    )

    native_validation = _validate_native_metal_artifact(
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        artifact_path=expected_path,
        required=require_metal_toolchain,
    )
    return {
        "name": "metal-roundtrip",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_METAL_ROUNDTRIP_SOURCE,
        "target": "metal",
        "roundTripStages": ["metal", "crossgl", "metal"],
        "unitCount": 1,
        "artifactCount": 1,
        "artifact": expected_report_path,
        "generatedHash": generated_hash,
        "generatedSizeBytes": artifact.get("generatedSizeBytes"),
        "diagnosticCounts": diagnostic_counts,
        "artifactValidationStatus": "validated",
        "nativeMetalValidation": native_validation,
        "fenceContract": {
            "memoryFlags": ["mem_device"],
            "memoryOrder": "memory_order_seq_cst",
            "threadScope": "thread_scope_system",
            "occurrences": MLX_FENCE_EXPECTED_ATOMIC_FENCE_COUNT,
            "preserved": True,
        },
        "semanticReadinessStatus": "blocked",
        "semanticTrackedIssues": list(METAL_ROUNDTRIP_SEMANTIC_TRACKED_ISSUES),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _atomic_fence_expected_message(target_contract: Mapping[str, str]) -> str:
    memory_flags = " | ".join(MLX_FENCE_REQUESTED_CONTRACT["memoryFlags"])
    return (
        "Cannot lower CrossGL atomicThreadFence to "
        f"{target_contract['targetDescription']} without changing its semantics "
        f"(flags={memory_flags}, "
        f"order={MLX_FENCE_REQUESTED_CONTRACT['memoryOrder']}, "
        f"scope={MLX_FENCE_REQUESTED_CONTRACT['threadScope']}): "
        "unsupported system thread scope"
    )


def _validate_atomic_fence_contract_report(
    mlx_root: Path,
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    exact_report: bool,
) -> dict[str, dict[str, Any]]:
    targets = tuple(MLX_FENCE_TARGET_CONTRACTS)
    summary = payload.get("summary", {})
    _require(isinstance(summary, Mapping), "fence contract summary must be an object")
    expected_diagnostic_codes = {
        contract["diagnosticCode"]: 1
        for contract in MLX_FENCE_TARGET_CONTRACTS.values()
    }
    expected_missing_capabilities = {
        contract["missingCapability"]: 1
        for contract in MLX_FENCE_TARGET_CONTRACTS.values()
    }
    diagnostics_by_code = summary.get("diagnosticsByCode", {})
    missing_capability_counts = summary.get("missingCapabilityCounts", {})
    _require(
        isinstance(diagnostics_by_code, Mapping),
        "fence contract diagnostic code counts must be an object",
    )
    _require(
        isinstance(missing_capability_counts, Mapping),
        "fence contract missing capability counts must be an object",
    )
    if exact_report:
        _require(
            summary.get("unitCount") == 1
            and summary.get("artifactCount") == len(targets)
            and summary.get("translatedCount") == 0
            and summary.get("failedCount") == len(targets),
            "fence contract translation must report one failed artifact per target",
        )
        _require(
            summary.get("diagnosticCounts")
            == {"error": len(targets), "note": 0, "warning": 0},
            "fence contract translation reported unexpected diagnostic severities",
        )
        _require(
            diagnostics_by_code == expected_diagnostic_codes,
            "fence contract translation diagnostic codes changed",
        )
        _require(
            missing_capability_counts == expected_missing_capabilities,
            "fence contract translation missing capabilities changed",
        )
    else:
        _require(
            all(
                diagnostics_by_code.get(code) == count
                for code, count in expected_diagnostic_codes.items()
            ),
            "full-corpus fence contract diagnostic codes changed",
        )
        _require(
            all(
                missing_capability_counts.get(capability) == count
                for capability, count in expected_missing_capabilities.items()
            ),
            "full-corpus fence contract missing capabilities changed",
        )

    diagnostics = payload.get("diagnostics", [])
    artifacts = payload.get("artifacts", [])
    _require(isinstance(diagnostics, list), "fence contract diagnostics must be a list")
    _require(isinstance(artifacts, list), "fence contract artifacts must be a list")
    if exact_report:
        _require(
            len(diagnostics) == len(targets) and len(artifacts) == len(targets),
            "fence contract report must contain one diagnostic and artifact per target",
        )

    target_results: dict[str, dict[str, Any]] = {}
    output_root = output_dir.resolve()
    for target, contract in MLX_FENCE_TARGET_CONTRACTS.items():
        target_diagnostics = [
            diagnostic
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
            and diagnostic.get("target") == target
            and str(diagnostic.get("message", "")).startswith(
                "Cannot lower CrossGL atomicThreadFence"
            )
        ]
        _require(
            len(target_diagnostics) == 1,
            f"fence contract report must contain one {target} diagnostic",
        )
        diagnostic = target_diagnostics[0]
        expected_message = _atomic_fence_expected_message(contract)
        _require(
            {
                "severity": diagnostic.get("severity"),
                "code": diagnostic.get("code"),
                "message": diagnostic.get("message"),
                "target": diagnostic.get("target"),
                "sourceBackend": diagnostic.get("sourceBackend"),
                "missingCapabilities": diagnostic.get("missingCapabilities"),
            }
            == {
                "severity": "error",
                "code": contract["diagnosticCode"],
                "message": expected_message,
                "target": target,
                "sourceBackend": "metal",
                "missingCapabilities": [contract["missingCapability"]],
            },
            f"fence contract {target} structured diagnostic changed",
        )
        location = diagnostic.get("location", {})
        _require(
            isinstance(location, Mapping) and location.get("file") == MLX_FENCE_SOURCE,
            f"fence contract {target} diagnostic source changed",
        )
        if exact_report:
            target_summary = summary.get("artifactsByTarget", {}).get(target, {})
            _require(
                target_summary.get("artifactCount") == 1
                and target_summary.get("translatedCount") == 0
                and target_summary.get("failedCount") == 1,
                f"fence contract {target} artifact summary changed",
            )

        target_artifacts = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, Mapping)
            and artifact.get("source") == MLX_FENCE_SOURCE
            and artifact.get("target") == target
        ]
        _require(
            len(target_artifacts) == 1,
            f"fence contract report must contain one {target} artifact record",
        )
        artifact = target_artifacts[0]
        _require(
            artifact.get("sourceBackend") == "metal"
            and artifact.get("status") == "failed"
            and artifact.get("error") == expected_message,
            f"fence contract {target} failed artifact record changed",
        )
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"fence contract {target} artifact path is missing",
        )
        generated_path = (mlx_root / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_root),
            f"fence contract {target} artifact path escaped its output directory",
        )
        _require(
            not generated_path.exists(),
            f"fence contract {target} unexpectedly emitted {artifact_path}",
        )
        _require(
            "generatedHash" not in artifact and "generatedSizeBytes" not in artifact,
            f"fence contract {target} recorded generated artifact metadata",
        )
        target_results[target] = {
            "diagnosticCode": contract["diagnosticCode"],
            "missingCapability": contract["missingCapability"],
            "requestedContract": dict(MLX_FENCE_REQUESTED_CONTRACT),
            "artifactStatus": "failed",
            "artifactEmitted": False,
        }
    return target_results


def _check_atomic_fence_contract(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    config_path = config_dir / "fence-contract.toml"
    report_path = report_dir / "fence-contract.json"
    output_dir = work_dir / "out-fence-contract"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    targets = tuple(MLX_FENCE_TARGET_CONTRACTS)
    _write_project_config(
        config_path,
        include=MLX_FENCE_SOURCE,
        targets=targets,
        output_dir=_relpath(output_dir, mlx_root),
    )
    result = _run_command(
        "translate-fence-contract",
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
    _require(
        result.returncode == 1,
        "atomic fence contract translation must fail with exit code 1",
    )

    payload = _load_json(report_path)
    target_results = _validate_atomic_fence_contract_report(
        mlx_root,
        output_dir,
        payload,
        exact_report=True,
    )

    return {
        "name": "atomic-fence-contract",
        "status": "blocked-as-expected",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_FENCE_SOURCE,
        "targets": list(targets),
        "artifactRecordCount": len(targets),
        "failedArtifactCount": len(targets),
        "emittedArtifactCount": 0,
        "requestedContract": dict(MLX_FENCE_REQUESTED_CONTRACT),
        "targetContracts": target_results,
        "semanticReadinessStatus": "blocked",
        "semanticTrackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
    }


def _strip_shader_comments(source: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def _shader_function_definition(
    source: str,
    function_pattern: str,
) -> tuple[re.Match[str], str] | None:
    header = re.search(
        rf"\bvoid\s+(?P<helper>{function_pattern})\s*"
        rf"\((?P<parameters>[^)]*)\)\s*\{{",
        source,
        flags=re.DOTALL,
    )
    if header is None:
        return None

    depth = 1
    body_start = header.end()
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return header, source[body_start:index]
    return None


def _reference_accessor_write_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    helper_write = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)"
        r"\s*\([^;{}]*\)\s*=(?!=)",
        source,
    )
    _require(
        helper_write is None,
        f"{target_name} reference accessor write still targets the accessor "
        "or a value-return helper",
    )

    sentinel = rf"{re.escape(REFERENCE_ACCESSOR_SENTINEL)}0*[fF]?"
    written_value = (
        rf"(?:{sentinel}|float\s*\(\s*{sentinel}\s*\)|"
        rf"\(\s*float\s*\)\s*{sentinel})"
    )
    write = re.search(
        rf"(?P<storage>(?P<owner>\b[A-Za-z_]\w*"
        rf"(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\.\s*val_frags\s*"
        rf"\[[^\]\n;]+\])\s*=(?!=)\s*{written_value}\s*;",
        source,
    )
    _require(
        write is not None,
        f"{target_name} reference accessor did not write the sentinel directly "
        "to original val_frags storage",
    )

    owner = re.sub(r"\s+", "", write.group("owner"))
    _require(
        owner == "tile",
        f"{target_name} reference accessor wrote a copied receiver instead of "
        "the original tile storage",
    )
    owner_parts = re.findall(r"[A-Za-z_]\w*", write.group("owner"))
    owner_pattern = r"\s*\.\s*".join(re.escape(part) for part in owner_parts)
    storage_lvalue = re.sub(r"\s+", "", write.group("storage"))
    readback_lvalues = [
        re.sub(r"\s+", "", match.group("storage"))
        for match in re.finditer(
            rf"(?<![=!<>])=(?!=)\s*(?P<storage>\b{owner_pattern}\s*\.\s*"
            rf"val_frags\s*\[[^\]\n;]+\])\s*;",
            source[write.end() :],
        )
    ]
    _require(
        storage_lvalue in readback_lvalues,
        f"{target_name} reference accessor fixture did not read back the exact "
        "val_frags lvalue written through the accessor",
    )

    return {
        "status": "verified-original-storage-write",
        "storageMember": "val_frags",
        "storageLvalue": storage_lvalue,
        "sentinel": REFERENCE_ACCESSOR_SENTINEL,
        "readBackFromSameStorage": True,
        "readBackFromWrittenLvalue": True,
        "readBackLvalue": storage_lvalue,
        "valueReturningHelperUsedForWrite": False,
    }


def _reference_accessor_const_read_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    accessor_helper = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)" r"\s*\(",
        source,
    )
    _require(
        accessor_helper is None,
        f"{target_name} reference accessor artifact still contains a frag_at "
        "helper or call",
    )

    helper_patterns = {
        "directx": (
            r"ReferenceAccessorTile__store",
            r"ReferenceAccessorOps__store",
        ),
        "opengl": (
            r"ReferenceAccessorTile_store[A-Za-z_0-9]*",
            r"ReferenceAccessorOps_store[A-Za-z_0-9]*",
        ),
    }
    _require(
        target in helper_patterns,
        f"unsupported reference accessor evidence target: {target}",
    )
    tile_helper_pattern, read_helper_pattern = helper_patterns[target]
    tile_store = re.search(
        rf"\bvoid\s+(?P<helper>{tile_helper_pattern})\s*"
        rf"\((?P<parameters>[^)]*)\)\s*\{{(?P<body>[^{{}}]*)\}}",
        source,
        flags=re.DOTALL,
    )
    _require(
        tile_store is not None,
        f"{target_name} reference accessor artifact is missing the lowered "
        "const store helper",
    )
    _require(
        re.search(
            r"\bReferenceAccessorTile\s+self\b",
            tile_store.group("parameters"),
        )
        is not None,
        f"{target_name} lowered const store helper is missing its tile receiver",
    )
    direct_read = re.search(
        rf"\b(?P<helper>{read_helper_pattern})\s*\(\s*"
        rf"(?P<storage>self\s*\.\s*val_frags\s*\[[^\]\n;]+\])\s*,",
        tile_store.group("body"),
    )
    _require(
        direct_read is not None,
        f"{target_name} implicit const accessor was not lowered to original "
        "self.val_frags storage passed directly to the read-only helper",
    )
    kernel_call = re.search(
        rf"\b{re.escape(tile_store.group('helper'))}\s*\(\s*tile\s*,",
        source,
    )
    _require(
        kernel_call is not None,
        f"{target_name} reference accessor kernel does not invoke the lowered "
        "const store path",
    )

    return {
        "status": "verified-original-storage-const-read",
        "storageMember": "val_frags",
        "storageLvalue": re.sub(r"\s+", "", direct_read.group("storage")),
        "implicitReceiver": "self",
        "passedDirectlyToHelper": True,
        "accessorCallEliminated": True,
        "kernelPathInvoked": True,
        "loweredTileHelper": tile_store.group("helper"),
        "loweredReadHelper": direct_read.group("helper"),
    }


def _reference_accessor_nested_const_alias_evidence(
    generated: str,
    *,
    target: str,
) -> dict[str, Any]:
    source = _strip_shader_comments(generated)
    target_name = {"directx": "DirectX", "opengl": "OpenGL"}.get(target, target)
    accessor_helper = re.search(
        r"\b(?:frag_at[A-Za-z_0-9]*|[A-Za-z_]\w*frag_at[A-Za-z_0-9]*)" r"\s*\(",
        source,
    )
    _require(
        accessor_helper is None,
        f"{target_name} nested reference accessor artifact still contains a "
        "frag_at helper or call",
    )
    _require(
        re.search(r"\baccum\b", source) is None,
        f"{target_name} nested reference accessor artifact still contains the "
        "accum reference alias",
    )

    helper_patterns = {
        "directx": r"ReferenceAccessorStoreLoop__store",
        "opengl": r"ReferenceAccessorStoreLoop_+store[A-Za-z_0-9]*",
    }
    _require(
        target in helper_patterns,
        f"unsupported nested reference accessor evidence target: {target}",
    )
    store_definition = _shader_function_definition(
        source,
        helper_patterns[target],
    )
    _require(
        store_definition is not None,
        f"{target_name} nested reference accessor artifact is missing the "
        "lowered const store helper",
    )
    store_header, store_body = store_definition
    _require(
        re.search(
            r"\bReferenceAccessorStoreLoop\s+self\b",
            store_header.group("parameters"),
        )
        is not None,
        f"{target_name} lowered nested const store helper is missing its "
        "outer value receiver",
    )

    direct_read = re.search(
        r"(?<![=!<>])=(?!=)\s*"
        r"(?P<storage>self\s*\.\s*nestedTile\s*\.\s*val_frags\s*"
        r"\[[^\]\n;]+\]\s*\[\s*k\s*\])\s*;",
        store_body,
    )
    lane = r"(?:uint\s*\(\s*k\s*\)|k)"
    helper_read = None
    if direct_read is None:
        helper_read = re.search(
            r"\bCrossGLMetalVectorIndex_[A-Za-z0-9_]+_set\s*\(\s*stored\s*,\s*"
            rf"{lane}\s*,\s*"
            r"\bCrossGLMetalVectorIndex_[A-Za-z0-9_]+_get\s*\(\s*"
            r"(?P<storage>self\s*\.\s*nestedTile\s*\.\s*val_frags\s*"
            r"\[[^\]\n;]+\])\s*,\s*"
            rf"{lane}\s*\)\s*\)",
            store_body,
        )
    storage_read = direct_read or helper_read
    _require(
        storage_read is not None,
        f"{target_name} nested const reference alias was not eliminated to "
        "self.nestedTile.val_frags storage indexed by k",
    )
    kernel_call = re.search(
        rf"\b{re.escape(store_header.group('helper'))}\s*\(\s*nestedStore\s*,",
        source,
    )
    _require(
        kernel_call is not None,
        f"{target_name} reference accessor kernel does not invoke the lowered "
        "nested const store path",
    )

    storage_lvalue = storage_read.group("storage")
    component_read_lowering = "direct-index"
    if helper_read is not None:
        storage_lvalue = f"{storage_lvalue}[k]"
        component_read_lowering = "lane-helper"

    return {
        "status": "verified-original-nested-storage-const-alias-read",
        "storageMember": "val_frags",
        "storagePath": "self.nestedTile.val_frags",
        "storageLvalue": re.sub(r"\s+", "", storage_lvalue),
        "componentReadLowering": component_read_lowering,
        "outerReceiver": "self",
        "tileMember": "nestedTile",
        "fragmentType": "float2",
        "aliasName": "accum",
        "aliasEliminated": True,
        "accessorCallEliminated": True,
        "indexedAliasRead": "accum[k]",
        "readFromOriginalStorage": True,
        "kernelPathInvoked": True,
        "loweredStoreHelper": store_header.group("helper"),
    }


def _validate_reference_accessor_directx(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "status": "not-required",
            "required": False,
            "nativeCompiler": "dxc",
        }

    dxc = shutil.which("dxc")
    _require(dxc is not None, "reference accessor DirectX validation requires dxc")
    output_path = work_dir / "validation" / "reference-accessor.dxil"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    result = _run_command(
        "validate-reference-accessor-directx",
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            REFERENCE_ACCESSOR_DXC_ENTRY_POINT,
            str(artifact_path),
            "-Fo",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        result.returncode == 0,
        "reference accessor DirectX compilation failed; inspect "
        "validate-reference-accessor-directx logs",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "reference accessor DirectX compilation did not produce DXIL",
    )
    return {
        "status": "validated",
        "required": True,
        "nativeCompiler": "dxc",
        "entryPoint": REFERENCE_ACCESSOR_DXC_ENTRY_POINT,
        "profile": "cs_6_0",
        "compiledArtifact": _relpath(output_path, mlx_root),
        "stdout": _relpath(result.stdout_path, mlx_root),
        "stderr": _relpath(result.stderr_path, mlx_root),
    }


def _validate_reference_accessor_opengl(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    artifact_path: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "status": "not-required",
            "required": False,
            "nativeCompiler": "glslangValidator",
            "spirvValidator": "spirv-val",
        }

    tools = {
        "glslangValidator": shutil.which("glslangValidator"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(name for name, value in tools.items() if value is None)
    _require(
        not missing_tools,
        "reference accessor OpenGL validation requires: " + ", ".join(missing_tools),
    )
    output_path = work_dir / "validation" / "reference-accessor-opengl.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    compile_result = _run_command(
        "validate-reference-accessor-opengl",
        [
            str(tools["glslangValidator"]),
            "--target-env",
            "opengl",
            "-S",
            "comp",
            str(artifact_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        compile_result.returncode == 0,
        "reference accessor OpenGL compilation failed; inspect "
        "validate-reference-accessor-opengl logs",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        "reference accessor OpenGL compilation did not produce SPIR-V",
    )
    validation_result = _run_command(
        "validate-reference-accessor-opengl-spirv",
        [
            str(tools["spirv-val"]),
            "--target-env",
            "opengl4.5",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        validation_result.returncode == 0,
        "reference accessor OpenGL SPIR-V validation failed; inspect "
        "validate-reference-accessor-opengl-spirv logs",
    )
    return {
        "status": "validated",
        "required": True,
        "nativeCompiler": "glslangValidator",
        "spirvValidator": "spirv-val",
        "targetEnvironments": ["opengl", "opengl4.5"],
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compileStdout": _relpath(compile_result.stdout_path, mlx_root),
        "compileStderr": _relpath(compile_result.stderr_path, mlx_root),
        "validationStdout": _relpath(validation_result.stdout_path, mlx_root),
        "validationStderr": _relpath(validation_result.stderr_path, mlx_root),
    }


def _check_reference_accessor_lvalue_identity(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_directx_toolchain: bool,
    require_opengl_toolchain: bool,
) -> dict[str, Any]:
    _require(
        REFERENCE_ACCESSOR_FIXTURE_PATH.is_file(),
        f"reference accessor fixture is missing: {REFERENCE_ACCESSOR_FIXTURE_PATH}",
    )
    project_dir = work_dir / "reference-accessor-project"
    output_dir = project_dir / "generated"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    staged_source = project_dir / REFERENCE_ACCESSOR_FIXTURE_NAME
    shutil.copyfile(REFERENCE_ACCESSOR_FIXTURE_PATH, staged_source)

    config_path = config_dir / "reference-accessor.toml"
    report_path = report_dir / "reference-accessor.json"
    report_path.unlink(missing_ok=True)
    _write_reference_accessor_project_config(config_path, "generated")
    result = _run_command(
        "translate-reference-accessor",
        [
            python,
            "-m",
            "crosstl",
            "translate-project",
            str(project_dir),
            "--config",
            str(config_path),
            "--report",
            str(report_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        report_path.is_file(),
        "reference accessor project translation did not produce a report",
    )
    payload = _load_json(report_path)
    if result.returncode != 0:
        messages = [
            str(item.get("message"))
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping) and isinstance(item.get("message"), str)
        ]
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"reference accessor translation failed{detail}")

    summary = payload.get("summary", {})
    diagnostics = payload.get("diagnostics", [])
    artifacts = payload.get("artifacts", [])
    diagnostic_counts = (
        summary.get("diagnosticCounts", {}) if isinstance(summary, Mapping) else {}
    )
    _require(
        isinstance(summary, Mapping)
        and isinstance(diagnostics, list)
        and isinstance(artifacts, list)
        and isinstance(diagnostic_counts, Mapping),
        "reference accessor report must contain structured summary collections",
    )
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == len(REFERENCE_ACCESSOR_TARGETS)
        and summary.get("translatedCount") == len(REFERENCE_ACCESSOR_TARGETS)
        and summary.get("failedCount") == 0,
        "reference accessor project translation did not emit both clean artifacts",
    )
    _require(
        len(artifacts) == len(REFERENCE_ACCESSOR_TARGETS)
        and all(isinstance(artifact, Mapping) for artifact in artifacts),
        "reference accessor report must contain exactly one artifact per target",
    )
    _require(
        not diagnostics
        and all(
            diagnostic_counts.get(severity) == 0
            for severity in ("note", "warning", "error")
        ),
        "reference accessor project translation must have zero diagnostics",
    )

    artifacts_by_target = {
        artifact.get("target"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("source") == REFERENCE_ACCESSOR_FIXTURE_NAME
        and artifact.get("status") == "translated"
        and artifact.get("target") in REFERENCE_ACCESSOR_TARGETS
    }
    _require(
        set(artifacts_by_target) == set(REFERENCE_ACCESSOR_TARGETS),
        "reference accessor report does not contain the expected target artifacts",
    )

    target_proofs: dict[str, Any] = {}
    for target in REFERENCE_ACCESSOR_TARGETS:
        artifact_path = artifacts_by_target[target].get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path),
            f"reference accessor {target} artifact path is missing",
        )
        generated_path = (project_dir / artifact_path).resolve()
        _require(
            _is_relative_to(generated_path, output_dir.resolve()),
            f"reference accessor {target} artifact escaped its output directory",
        )
        _require(
            generated_path.is_file(),
            f"reference accessor {target} artifact is missing: {artifact_path}",
        )
        generated = generated_path.read_text(encoding="utf-8")
        write_evidence = _reference_accessor_write_evidence(
            generated,
            target=target,
        )
        const_read_evidence = _reference_accessor_const_read_evidence(
            generated,
            target=target,
        )
        nested_const_alias_evidence = _reference_accessor_nested_const_alias_evidence(
            generated,
            target=target,
        )
        if target == "directx":
            native_validation = _validate_reference_accessor_directx(
                mlx_root,
                work_dir,
                log_dir,
                generated_path,
                required=require_directx_toolchain,
            )
        else:
            native_validation = _validate_reference_accessor_opengl(
                mlx_root,
                work_dir,
                log_dir,
                generated_path,
                required=require_opengl_toolchain,
            )
        target_proofs[target] = {
            "artifact": _relpath(generated_path, mlx_root),
            "artifactSha256": _sha256(generated_path),
            "writeEvidence": write_evidence,
            "constReadEvidence": const_read_evidence,
            "nestedConstAliasEvidence": nested_const_alias_evidence,
            "nativeValidation": native_validation,
        }

    return {
        "name": "reference-accessor-lvalue-identity",
        "status": "passed",
        "proofStatus": "verified-original-storage-write",
        "constReadProofStatus": "verified-original-storage-const-read",
        "nestedConstAliasProofStatus": (
            "verified-original-nested-storage-const-alias-read"
        ),
        "scope": "reduced-mlx-shaped-fixture",
        "translationSurface": "crosstl translate-project",
        "report": _relpath(report_path, mlx_root),
        "sourceFixture": (
            "demos/integrations/mlx/fixtures/" + REFERENCE_ACCESSOR_FIXTURE_NAME
        ),
        "stagedSource": _relpath(staged_source, mlx_root),
        "sourceSha256": _sha256(staged_source),
        "accessorContract": {
            "method": "frag_at",
            "returnType": "thread float&",
            "storageExpression": "val_frags[i * width + j]",
            "writeExpression": "tile.frag_at(1, 1) = 73.25f",
            "constRead": {
                "returnType": "const thread float&",
                "enclosingMethod": "store(...) const",
                "implicitCall": "frag_at(i, j)",
                "helperParameterType": "const thread float&",
            },
            "nestedConstAliasRead": {
                "outerType": "ReferenceAccessorStoreLoop",
                "tileMember": "nestedTile",
                "fragmentReturnType": "const thread float2&",
                "enclosingMethod": "store(...) const",
                "aliasDeclaration": (
                    "thread const auto& accum = nestedTile.frag_at(i, j)"
                ),
                "readExpression": "accum[k]",
                "storageExpression": "nestedTile.val_frags[i * width + j][k]",
            },
        },
        "targets": list(REFERENCE_ACCESSOR_TARGETS),
        "artifactCount": len(REFERENCE_ACCESSOR_TARGETS),
        "projectDiagnosticCount": 0,
        "targetProofs": target_proofs,
        "nativeToolchainRequiredByTarget": {
            "directx": require_directx_toolchain,
            "opengl": require_opengl_toolchain,
        },
        "upstreamMlxRuntimeExecuted": False,
        "runtimeParityClaimed": False,
    }


def _scaled_attention_local_alias_evidence(
    mlx_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = {
        artifact.get("target"): artifact
        for artifact in payload.get("artifacts", [])
        if isinstance(artifact, Mapping)
        and artifact.get("source") == MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
        and artifact.get("target") in {"directx", "vulkan"}
        and artifact.get("status") == "translated"
    }
    _require(
        set(artifacts) == {"directx", "vulkan"},
        "scaled-attention local-alias evidence requires DirectX and Vulkan artifacts",
    )

    generated_by_target: dict[str, str] = {}
    paths_by_target: dict[str, str] = {}
    for target, artifact in artifacts.items():
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str),
            f"scaled-attention {target} artifact path is missing",
        )
        generated_path = mlx_root / artifact_path
        _require(
            generated_path.is_file(),
            f"scaled-attention {target} artifact is missing: {artifact_path}",
        )
        generated_by_target[target] = generated_path.read_text(encoding="utf-8")
        paths_by_target[target] = artifact_path

    forbidden_alias_residue = re.compile(
        r"\bLimits_u3cU_u3e\b|\bUnknown type U\b|unknown function ['\"]U['\"]"
    )
    for target, generated in generated_by_target.items():
        residue = forbidden_alias_residue.search(generated)
        _require(
            residue is None,
            f"scaled-attention {target} retained local alias residue: "
            f"{residue.group(0) if residue else ''}",
        )
    directx_alias = re.search(r"\bU\b", generated_by_target["directx"])
    _require(
        directx_alias is None,
        "scaled-attention DirectX artifact retained the local alias U",
    )
    vulkan_warnings = [
        line
        for line in generated_by_target["vulkan"].splitlines()
        if line.lstrip().startswith("; WARNING:")
    ]
    _require(
        not vulkan_warnings,
        "scaled-attention Vulkan project artifact emitted a semantic warning: "
        + (vulkan_warnings[0] if vulkan_warnings else ""),
    )

    directx_entry_count = len(
        re.findall(
            r"(?m)^[ \t]*\[numthreads[ \t]*\(",
            generated_by_target["directx"],
        )
    )
    vulkan_entry_count = len(
        re.findall(
            r"(?m)^[ \t]*OpEntryPoint[ \t]+GLCompute\b",
            generated_by_target["vulkan"],
        )
    )
    expected_entry_count = MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS[
        MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE
    ]
    _require(
        directx_entry_count == expected_entry_count
        and vulkan_entry_count == expected_entry_count,
        "scaled-attention artifacts did not retain all "
        f"{expected_entry_count} materialized entries",
    )

    return {
        "source": MLX_SCALED_DOT_PRODUCT_ATTENTION_SOURCE,
        "targets": paths_by_target,
        "entryCountByTarget": {
            "directx": directx_entry_count,
            "vulkan": vulkan_entry_count,
        },
        "resolvedDeclarationTypeCount": 402,
        "resolvedCastCount": 87,
        "resolvedStaticMemberCount": 42,
        "vulkanProjectWarningCount": 0,
        "remainingAliasShapesTrackedBy": (
            "https://github.com/CrossGL/crosstl/issues/1567"
        ),
        "unreachableGenericWarningsTrackedBy": (
            "https://github.com/CrossGL/crosstl/issues/1568"
        ),
    }


def _directx_toolchain_entry_point(run: Mapping[str, Any]) -> str | None:
    command = run.get("command")
    if not isinstance(command, list) or any(
        not isinstance(argument, str) for argument in command
    ):
        return None
    try:
        profile = command[command.index("-T") + 1]
        entry_point = command[command.index("-E") + 1]
    except (ValueError, IndexError):
        return None
    if not profile.startswith("cs_") or not entry_point:
        return None
    return entry_point


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
        specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
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
    scaled_attention_alias_evidence = _scaled_attention_local_alias_evidence(
        mlx_root, payload
    )
    toolchain_payloads = [payload]
    toolchain_artifact_payloads = {"directx": payload, "vulkan": payload}
    if run_toolchains:
        toolchain_targets = [
            target
            for target, required in (
                ("directx", require_directx_toolchain),
                ("vulkan", require_vulkan_toolchain),
            )
            if required
        ] or ["vulkan"]
        toolchain_name = "-".join(toolchain_targets)
        # DXC gates its selected subset; Vulkan gates the whole frontier. When the
        # DirectX toolchain is required, restrict the compile gate to the Windows
        # subset (CI requires one target per OS).
        toolchain_sources = (
            MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
            if "directx" in toolchain_targets
            else MLX_DIRECTX_VULKAN_FRONTIER_SOURCES
        )
        toolchain_config_path = config_dir / f"{toolchain_name}-frontier-toolchain.toml"
        toolchain_report_path = report_dir / f"{toolchain_name}-frontier-toolchain.json"
        _write_project_config(
            toolchain_config_path,
            include=toolchain_sources,
            targets=tuple(toolchain_targets),
            output_dir=_relpath(
                work_dir / f"out-{toolchain_name}-frontier-toolchain", mlx_root
            ),
            specialization_constants=MLX_FRONTIER_SPECIALIZATION_CONSTANTS,
        )
        _run_command(
            f"validate-{toolchain_name}-frontier-toolchain",
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
        toolchain_payloads.append(toolchain_payload)
        for target in toolchain_targets:
            toolchain_artifact_payloads[target] = toolchain_payload
    toolchain_runs = []
    for toolchain_payload in toolchain_payloads:
        toolchain_validation = toolchain_payload.get("validation", {})
        _require(
            isinstance(toolchain_validation, dict),
            "frontier toolchain validation must be an object",
        )
        payload_runs = toolchain_validation.get("toolchainRuns", [])
        _require(isinstance(payload_runs, list), "toolchainRuns must be a list")
        toolchain_runs.extend(payload_runs)
    directx_runs = [
        run
        for run in toolchain_runs
        if isinstance(run, dict) and run.get("target") == "directx"
    ]
    vulkan_runs = [
        run
        for run in toolchain_runs
        if isinstance(run, dict) and run.get("target") == "vulkan"
    ]
    directx_entry_points_by_source: dict[str, list[str]] = {
        source: [] for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
    }
    for run in directx_runs:
        if run.get("status") != "ok":
            continue
        source = run.get("source")
        _require(
            source in directx_entry_points_by_source,
            f"DirectX toolchain validation reported an unexpected source: {source}",
        )
        entry_point = _directx_toolchain_entry_point(run)
        _require(
            entry_point is not None,
            "DirectX toolchain validation did not record a compute entry command",
        )
        validated_entries = directx_entry_points_by_source[source]
        _require(
            entry_point not in validated_entries,
            f"DirectX toolchain validation duplicated {source} entry {entry_point}",
        )
        validated_entries.append(entry_point)
    directx_validated_entry_point_counts = {
        source: len(entry_points)
        for source, entry_points in directx_entry_points_by_source.items()
        if entry_points
    }
    directx_validated_sources = [
        source
        for source in MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES
        if source in directx_validated_entry_point_counts
    ]
    required_toolchains = {
        "directx": (
            require_directx_toolchain,
            directx_runs,
            len(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        ),
        "vulkan": (require_vulkan_toolchain, vulkan_runs, frontier_count),
    }
    for target, (required, runs, expected_count) in required_toolchains.items():
        if required and run_toolchains:
            artifact_payload = toolchain_artifact_payloads.get(target, payload)
            artifact_paths = {
                artifact.get("path")
                for artifact in artifact_payload.get("artifacts", [])
                if isinstance(artifact, dict)
                and artifact.get("target") == target
                and artifact.get("status") == "translated"
                and isinstance(artifact.get("path"), str)
            }
            if target == "directx":
                artifact_sources = {
                    artifact.get("source")
                    for artifact in artifact_payload.get("artifacts", [])
                    if isinstance(artifact, dict)
                    and artifact.get("target") == target
                    and artifact.get("status") == "translated"
                    and isinstance(artifact.get("source"), str)
                }
                _require(
                    artifact_sources == set(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
                    "DirectX toolchain artifact sources did not match the frontier",
                )
            validated_paths = {
                run.get("path")
                for run in runs
                if run.get("status") == "ok" and isinstance(run.get("path"), str)
            }
            _require(
                len(artifact_paths) == expected_count
                and artifact_paths <= validated_paths,
                (
                    f"{target.title()} toolchain validation was required for every "
                    "validated frontier artifact"
                ),
            )
        for run in runs:
            _require(
                run.get("status") == "ok",
                f"{target.title()} toolchain validation failed",
            )
        if required and not run_toolchains:
            _require(
                not runs,
                (
                    f"{target.title()} toolchain validation ran while active "
                    "validation issues are tracked"
                ),
            )
    if require_directx_toolchain and run_toolchains:
        _require(
            directx_validated_sources == list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
            "DirectX toolchain did not validate every configured source",
        )
        _require(
            directx_validated_entry_point_counts
            == MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS,
            "DirectX toolchain did not validate every generated compute entry point",
        )
    return {
        "name": "directx-vulkan-frontier",
        "status": "passed",
        "scope": "clean-frontier",
        "report": _relpath(report_path, mlx_root),
        "sources": list(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES),
        "unitCount": frontier_count,
        "artifactCount": frontier_count * 2,
        "targets": ["directx", "vulkan"],
        "toolchainRuns": len(toolchain_runs),
        "directxToolchainRequired": require_directx_toolchain,
        "directxToolchainSources": list(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "directxToolchainArtifactCount": len(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES),
        "directxToolchainExpectedEntryPointCounts": dict(
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS
        ),
        "directxToolchainExpectedEntryPointCount": (
            MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNT
        ),
        "directxToolchainValidatedSources": directx_validated_sources,
        "directxToolchainValidatedArtifactCount": len(directx_validated_sources),
        "directxToolchainValidatedEntryPointCounts": (
            directx_validated_entry_point_counts
        ),
        "directxToolchainValidatedEntryPointCount": sum(
            directx_validated_entry_point_counts.values()
        ),
        "vulkanToolchainRequired": require_vulkan_toolchain,
        "directxValidationStatus": (
            "validated" if run_toolchains and directx_runs else "not-required"
        ),
        "vulkanValidationStatus": (
            "validated"
            if vulkan_runs
            else ("not-run" if run_toolchains else "blocked-by-tracked-issues")
        ),
        "semanticReadinessStatus": "not-established",
        "regressionEvidence": [scaled_attention_alias_evidence],
        "trackedIssues": list(FRONTIER_VALIDATION_TRACKED_ISSUES),
        "runtimeParityClaimed": False,
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
        entry_points={MLX_ARANGE_SOURCE: "arangeuint32"},
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
    _require(
        artifact.get("entryPoint")
        == {"source": "arangeuint32", "target": "main", "stage": "compute"},
        "OpenGL arange artifact did not record the selected compute entry",
    )
    _require(
        generated.count("void main()") == 1 and "compute_main" not in generated,
        "OpenGL arange artifact is not independently loadable through one main entry",
    )
    resource_bindings = re.findall(
        r"layout\s*\(\s*std(?:140|430)\s*,\s*binding\s*=\s*(\d+)\s*\)\s*"
        r"(?:uniform|buffer)\b",
        generated,
    )
    _require(
        sorted(int(binding) for binding in resource_bindings) == [0, 1, 2],
        "OpenGL arange artifact must expose only start, step, and output resources",
    )
    normalized_source = re.sub(r"\s+", "", generated)
    _require(
        "out_[index]=(start+(index*step));" in normalized_source,
        "OpenGL arange artifact did not preserve uint32 arange data flow",
    )
    _require(
        "arangeuint8" not in generated and "arangefloat" not in generated,
        "OpenGL arange artifact retained unrelated materialized entries",
    )
    _require(
        "log1p__metal_overload_" not in generated
        and "subgroupShuffle" not in generated
        and "complex64_t probe(" not in generated,
        "OpenGL arange artifact retained unrelated helper dependencies",
    )
    native_validation = _validate_arange_opengl(
        mlx_root,
        work_dir,
        log_dir,
        generated_path,
    )
    return {
        "name": "arange-opengl",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_ARANGE_SOURCE,
        "target": "opengl",
        "metalIncludesFiltered": True,
        "selectedEntryPoint": "arangeuint32",
        "targetEntryPoint": "main",
        "interfaceResourceCount": 3,
        "standaloneArtifact": True,
        "arangeDataFlowPreserved": True,
        **native_validation,
        "trackedIssues": list(OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES),
    }


def _validate_arange_opengl(
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    generated_path: Path,
) -> dict[str, Any]:
    validator = shutil.which("glslangValidator")
    if validator is None:
        return {
            "nativeValidationAttempted": False,
            "nativeValidationBlockerConfirmed": False,
            "nativeValidationStatus": "not-run-tool-unavailable",
            "nativeValidator": "glslangValidator",
            "nativeValidatorStatus": "unavailable",
        }

    output_path = work_dir / "validation" / "arange-opengl.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(
        "validate-arange-opengl",
        [
            validator,
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    if result.returncode == 0:
        _require(
            not OPENGL_ARANGE_VALIDATION_TRACKED_ISSUES,
            "OpenGL arange validation passed while tracked validation issues remain",
        )
        return {
            "nativeValidationAttempted": True,
            "nativeValidationBlockerConfirmed": False,
            "nativeValidationStatus": "validated",
            "nativeValidator": "glslangValidator",
            "nativeValidatorStatus": "available",
            "nativeValidationExitCode": 0,
            "nativeValidationOutput": _relpath(output_path, mlx_root),
        }

    raise PortingCheckError(
        "OpenGL arange validation failed without a tracked validation issue"
    )


def _check_opengl_frontier(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
    *,
    require_toolchain: bool,
) -> dict[str, Any]:
    config_path = config_dir / "opengl-frontier.toml"
    report_path = report_dir / "opengl-frontier.json"
    _write_project_config(
        config_path,
        include=MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES,
        targets=("opengl",),
        output_dir=_relpath(work_dir / "out-opengl-frontier", mlx_root),
    )
    result = _run_command(
        "translate-opengl-frontier",
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
    _require(
        isinstance(summary, Mapping),
        "OpenGL frontier summary must be an object",
    )
    if result.returncode != 0:
        messages = [
            str(item.get("message"))
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping) and isinstance(item.get("message"), str)
        ]
        messages.extend(
            str(item.get("error"))
            for item in payload.get("artifacts", [])
            if isinstance(item, Mapping) and isinstance(item.get("error"), str)
        )
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"OpenGL frontier translation failed{detail}")

    artifacts = payload.get("artifacts", [])
    _require(
        isinstance(artifacts, list),
        "OpenGL frontier artifacts must be a list",
    )
    frontier_count = len(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES)
    diagnostic_counts = summary.get("diagnosticCounts", {})
    diagnostics = payload.get("diagnostics", [])
    _require(
        isinstance(diagnostic_counts, Mapping) and isinstance(diagnostics, list),
        "OpenGL frontier diagnostics must be reported as structured collections",
    )
    _require(
        summary.get("unitCount") == frontier_count
        and summary.get("artifactCount") == frontier_count
        and summary.get("translatedCount") == frontier_count
        and summary.get("failedCount") == 0
        and len(artifacts) == frontier_count,
        "OpenGL frontier report did not contain every clean translated artifact",
    )
    _require(
        not diagnostics
        and all(
            diagnostic_counts.get(severity) == 0
            for severity in ("note", "warning", "error")
        ),
        "OpenGL frontier translation must complete with zero project diagnostics",
    )
    artifacts_by_source = {
        artifact.get("source"): artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping)
        and artifact.get("target") == "opengl"
        and artifact.get("status") == "translated"
        and isinstance(artifact.get("source"), str)
    }
    _require(
        set(artifacts_by_source) == set(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES),
        "OpenGL frontier translated artifact set does not match its source set",
    )
    generated_paths: dict[str, Path] = {}
    for source in MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES:
        artifact_path = artifacts_by_source[source].get("path")
        _require(
            isinstance(artifact_path, str),
            f"OpenGL frontier artifact path is missing for {source}",
        )
        generated_path = mlx_root / artifact_path
        _require(
            generated_path.is_file(),
            f"OpenGL frontier artifact is missing: {artifact_path}",
        )
        generated_paths[source] = generated_path

    specialization_evidence: dict[str, dict[str, int]] = {}
    for source, expected_constants in MLX_OPENGL_SPECIALIZATION_CONSTANT_IDS.items():
        artifact = artifacts_by_source[source]
        reflected_constants = artifact.get("specializationConstants", [])
        _require(
            isinstance(reflected_constants, list),
            f"OpenGL specialization metadata is missing for {source}",
        )
        reflected_ids = {
            constant.get("name"): constant.get("id")
            for constant in reflected_constants
            if isinstance(constant, Mapping)
        }
        _require(
            reflected_ids == expected_constants,
            f"OpenGL specialization metadata does not match {source}",
        )
        materialization = artifact.get("specializationMaterialization")
        _require(
            isinstance(materialization, Mapping)
            and materialization.get("mode") == "deferred"
            and materialization.get("targetSupportsDeferredSpecialization") is True,
            f"OpenGL specialization deferral is not recorded for {source}",
        )
        generated = generated_paths[source].read_text(encoding="utf-8")
        for name, constant_id in expected_constants.items():
            _require(
                re.search(
                    rf"layout\s*\(\s*constant_id\s*=\s*{constant_id}\s*\)"
                    rf"\s*const\s+\w+\s+{re.escape(name)}\s*=",
                    generated,
                )
                is not None,
                f"OpenGL artifact did not preserve specialization id {constant_id} "
                f"for {name}",
            )
            _require(
                re.search(rf"\buniform\s+\w+\s+{re.escape(name)}\b", generated) is None,
                f"OpenGL artifact lowered specialization input {name} as a uniform",
            )
        specialization_evidence[source] = dict(expected_constants)

    validation_status = "not-required"
    validation_outputs: dict[str, str] = {}
    toolchain_validated_sources: list[str] = []
    if require_toolchain:
        required_tools = {
            "glslangValidator": shutil.which("glslangValidator"),
            "spirv-val": shutil.which("spirv-val"),
        }
        missing_tools = sorted(
            name for name, resolved in required_tools.items() if resolved is None
        )
        _require(
            not missing_tools,
            "OpenGL frontier validation requires: " + ", ".join(missing_tools),
        )

        for source, generated_path in generated_paths.items():
            stem = Path(source).stem
            command_name = stem.replace("_", "-")
            output_path = work_dir / "validation" / f"{stem}-opengl.spv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            compile_result = _run_command(
                f"validate-{command_name}-opengl",
                [
                    str(required_tools["glslangValidator"]),
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    str(generated_path),
                    "-o",
                    str(output_path),
                ],
                log_dir=log_dir,
                check=False,
            )
            _require(
                compile_result.returncode == 0,
                (
                    f"OpenGL {stem} native compilation failed; inspect "
                    f"validate-{command_name}-opengl logs"
                ),
            )
            _require(
                output_path.is_file(),
                f"OpenGL {stem} compilation succeeded without producing SPIR-V",
            )
            validation_result = _run_command(
                f"validate-{command_name}-opengl-spirv",
                [
                    str(required_tools["spirv-val"]),
                    "--target-env",
                    "spv1.3",
                    str(output_path),
                ],
                log_dir=log_dir,
                check=False,
            )
            _require(
                validation_result.returncode == 0,
                (
                    f"OpenGL {stem} SPIR-V validation failed; inspect "
                    f"validate-{command_name}-opengl-spirv logs"
                ),
            )
            validation_outputs[source] = _relpath(output_path, mlx_root)
            toolchain_validated_sources.append(source)
        _require(
            toolchain_validated_sources == list(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES),
            "OpenGL frontier toolchain did not validate every configured source",
        )
        validation_status = "validated"

    return {
        "name": "opengl-frontier",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "sources": list(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES),
        "sourceCount": frontier_count,
        "target": "opengl",
        "artifactCount": frontier_count,
        "projectDiagnosticCount": 0,
        "toolchainRequired": require_toolchain,
        "toolchainValidatedSources": toolchain_validated_sources,
        "toolchainValidatedArtifactCount": len(toolchain_validated_sources),
        "nativeValidationStatus": validation_status,
        "nativeValidator": "glslangValidator",
        "spirvValidator": "spirv-val",
        "nativeValidationOutputs": validation_outputs,
        "specializationConstants": specialization_evidence,
        "runtimeIntegrationIncluded": False,
    }


def _check_gemv_opengl_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    required_tools = {
        "glslangValidator": shutil.which("glslangValidator"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(
        name for name, resolved in required_tools.items() if resolved is None
    )
    _require(
        not missing_tools,
        "OpenGL GEMV validation requires: " + ", ".join(missing_tools),
    )

    config_path = config_dir / "gemv-opengl.toml"
    report_path = report_dir / "gemv-opengl.json"
    _write_project_config(
        config_path,
        include=MLX_GEMV_SOURCE,
        targets=("opengl",),
        output_dir=_relpath(work_dir / "out-gemv-opengl", mlx_root),
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    result = _run_command(
        "translate-gemv-opengl",
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
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, dict), "OpenGL GEMV summary must be an object")
    if result.returncode != 0:
        diagnostics = [
            str(item.get("message"))
            for item in payload.get("diagnostics", [])
            if isinstance(item, Mapping) and isinstance(item.get("message"), str)
        ]
        detail = f": {diagnostics[0]}" if diagnostics else ""
        raise PortingCheckError(f"OpenGL GEMV translation failed{detail}")
    _require(
        summary.get("translatedCount") == 1 and summary.get("failedCount") == 0,
        "OpenGL GEMV report did not contain one clean translated artifact",
    )

    artifact = next(
        (
            item
            for item in payload.get("artifacts", [])
            if isinstance(item, Mapping)
            and item.get("source") == MLX_GEMV_SOURCE
            and item.get("target") == "opengl"
            and item.get("status") == "translated"
        ),
        None,
    )
    _require(isinstance(artifact, Mapping), "OpenGL GEMV artifact is missing")
    artifact_path = artifact.get("path")
    _require(
        isinstance(artifact_path, str),
        "OpenGL GEMV artifact path is missing",
    )
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(),
        f"OpenGL GEMV artifact is missing: {artifact_path}",
    )

    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("specializationCount")
        == GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "OpenGL GEMV artifact did not materialize the complete specialization set",
    )
    generated = generated_path.read_text(encoding="utf-8")
    entry_point_count = len(
        re.findall(
            r"(?m)^void\s+(?:main|compute_main(?:_\d+)?)\s*\(",
            generated,
        )
    )
    _require(
        entry_point_count == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "OpenGL GEMV artifact did not emit the complete entry-point set",
    )
    residue = re.search(
        r"BinaryOpNode|LoopedElemToLoc|\b(?:OffsetT|acc_type|nullptr)\b",
        generated,
    )
    _require(
        residue is None,
        f"OpenGL GEMV artifact retained unresolved materialization text: {residue.group(0) if residue else ''}",
    )

    output_path = work_dir / "validation" / "gemv-opengl.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compile_result = _run_command(
        "validate-gemv-opengl",
        [
            str(required_tools["glslangValidator"]),
            "--target-env",
            "opengl",
            "--target-env",
            "spirv1.3",
            "-S",
            "comp",
            str(generated_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        compile_result.returncode == 0,
        "OpenGL GEMV native compilation failed; inspect validate-gemv-opengl logs",
    )
    _require(
        output_path.is_file(),
        "OpenGL GEMV native compilation succeeded without producing SPIR-V",
    )
    validation_result = _run_command(
        "validate-gemv-opengl-spirv",
        [
            str(required_tools["spirv-val"]),
            "--target-env",
            "spv1.3",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        validation_result.returncode == 0,
        "OpenGL GEMV SPIR-V validation failed; inspect validate-gemv-opengl-spirv logs",
    )
    warning_lines = [
        line
        for path in (
            compile_result.stdout_path,
            compile_result.stderr_path,
            validation_result.stdout_path,
            validation_result.stderr_path,
        )
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "warning:" in line.lower()
    ]
    unexpected_warnings = [
        line
        for line in warning_lines
        if 'identifiers containing consecutive underscores ("__") are reserved'
        not in line
    ]
    _require(
        not unexpected_warnings,
        "OpenGL GEMV native compilation emitted an untracked warning: "
        + (unexpected_warnings[0] if unexpected_warnings else ""),
    )
    return {
        "name": "gemv-opengl-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "target": "opengl",
        "specializationCount": materialization.get("specializationCount"),
        "entryPointCount": entry_point_count,
        "nativeValidationStatus": "validated",
        "nativeValidator": "glslangValidator",
        "spirvValidator": "spirv-val",
        "nativeWarningCount": len(warning_lines),
        "nativeWarningsTrackedBy": (
            "https://github.com/CrossGL/crosstl/issues/1513" if warning_lines else None
        ),
        "nativeValidationOutput": _relpath(output_path, mlx_root),
        "runtimeIntegrationIncluded": False,
    }


def _check_gemv_vulkan_toolchain(
    mlx_root: Path,
    work_dir: Path,
    config_dir: Path,
    report_dir: Path,
    log_dir: Path,
    python: str,
) -> dict[str, Any]:
    required_tools = {
        "spirv-as": shutil.which("spirv-as"),
        "spirv-val": shutil.which("spirv-val"),
    }
    missing_tools = sorted(
        name for name, resolved in required_tools.items() if resolved is None
    )
    _require(
        not missing_tools,
        "Vulkan GEMV validation requires: " + ", ".join(missing_tools),
    )

    config_path = config_dir / "gemv-vulkan.toml"
    report_path = report_dir / "gemv-vulkan.json"
    _write_project_config(
        config_path,
        include=MLX_GEMV_SOURCE,
        targets=("vulkan",),
        output_dir=_relpath(work_dir / "out-gemv-vulkan", mlx_root),
        metal_source_options={
            "max_template_specializations": GEMV_MAX_TEMPLATE_SPECIALIZATIONS,
            "max_template_materialization_work": GEMV_MAX_TEMPLATE_MATERIALIZATION_WORK,
        },
    )
    result = _run_command(
        "translate-gemv-vulkan",
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
            "--no-format",
        ],
        log_dir=log_dir,
        check=False,
    )
    payload = _load_json(report_path)
    summary = payload.get("summary", {})
    _require(isinstance(summary, Mapping), "Vulkan GEMV summary must be an object")
    if result.returncode != 0:
        diagnostics = [
            item for item in payload.get("diagnostics", []) if isinstance(item, Mapping)
        ]
        messages = [
            str(item.get("message"))
            for item in diagnostics
            if isinstance(item.get("message"), str)
        ]
        detail = f": {messages[0]}" if messages else ""
        raise PortingCheckError(f"Vulkan GEMV translation failed{detail}")
    _require(
        summary.get("translatedCount") == 1 and summary.get("failedCount") == 0,
        "Vulkan GEMV report did not contain one clean translated artifact",
    )

    artifact = next(
        (
            item
            for item in payload.get("artifacts", [])
            if isinstance(item, Mapping)
            and item.get("source") == MLX_GEMV_SOURCE
            and item.get("target") == "vulkan"
            and item.get("status") == "translated"
        ),
        None,
    )
    _require(isinstance(artifact, Mapping), "Vulkan GEMV artifact is missing")
    artifact_path = artifact.get("path")
    _require(isinstance(artifact_path, str), "Vulkan GEMV artifact path is missing")
    generated_path = mlx_root / artifact_path
    _require(
        generated_path.is_file(),
        f"Vulkan GEMV artifact is missing: {artifact_path}",
    )

    materialization = artifact.get("templateMaterialization", {})
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("specializationCount")
        == GEMV_EXPECTED_SPECIALIZATION_COUNT,
        "Vulkan GEMV artifact did not materialize the complete specialization set",
    )
    generated = generated_path.read_text(encoding="utf-8")
    entry_point_count = len(
        re.findall(r"(?m)^[ \t]*OpEntryPoint[ \t]+GLCompute\b", generated)
    )
    _require(
        entry_point_count == GEMV_EXPECTED_ENTRY_POINT_COUNT,
        "Vulkan GEMV artifact did not emit the complete entry-point set",
    )

    warning_lines = [
        line
        for line in generated.splitlines()
        if line.lstrip().startswith("; WARNING:")
    ]
    _require(
        not warning_lines,
        "Vulkan GEMV artifact emitted a semantic warning: "
        + (warning_lines[0] if warning_lines else ""),
    )
    generated_without_warnings = "\n".join(
        line
        for line in generated.splitlines()
        if not line.lstrip().startswith("; WARNING:")
    )
    residue = re.search(
        r"BinaryOpNode|IdentifierNode|LiteralNode|PrimitiveType|\b(?:acc_type|nullptr)\b",
        generated_without_warnings,
    )
    _require(
        residue is None,
        "Vulkan GEMV artifact retained unresolved materialization text outside "
        f"tracked warnings: {residue.group(0) if residue else ''}",
    )

    output_path = work_dir / "validation" / "gemv-vulkan.spv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assembly_result = _run_command(
        "assemble-gemv-vulkan",
        [
            str(required_tools["spirv-as"]),
            "--target-env",
            "vulkan1.1",
            str(generated_path),
            "-o",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        assembly_result.returncode == 0,
        "Vulkan GEMV assembly failed; inspect assemble-gemv-vulkan logs",
    )
    _require(
        output_path.is_file(),
        "Vulkan GEMV assembly succeeded without producing SPIR-V",
    )
    validation_result = _run_command(
        "validate-gemv-vulkan-spirv",
        [
            str(required_tools["spirv-val"]),
            "--target-env",
            "vulkan1.1",
            str(output_path),
        ],
        log_dir=log_dir,
        check=False,
    )
    _require(
        validation_result.returncode == 0,
        "Vulkan GEMV SPIR-V validation failed; inspect "
        "validate-gemv-vulkan-spirv logs",
    )
    tool_warning_lines = [
        line
        for command_result in (assembly_result, validation_result)
        for path in (command_result.stdout_path, command_result.stderr_path)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "warning:" in line.lower()
    ]
    _require(
        not tool_warning_lines,
        "Vulkan GEMV SPIR-V tools emitted a warning: "
        + (tool_warning_lines[0] if tool_warning_lines else ""),
    )

    diagnostic_counts = summary.get("diagnosticCounts", {})
    report_warning_count = (
        diagnostic_counts.get("warning", 0)
        if isinstance(diagnostic_counts, Mapping)
        else 0
    )
    _require(
        report_warning_count in (0, len(warning_lines)),
        "Vulkan GEMV report warning count does not match artifact warnings",
    )
    return {
        "name": "gemv-vulkan-toolchain",
        "status": "passed",
        "report": _relpath(report_path, mlx_root),
        "source": MLX_GEMV_SOURCE,
        "target": "vulkan",
        "specializationCount": materialization.get("specializationCount"),
        "entryPointCount": entry_point_count,
        "structuralValidationStatus": "validated",
        "assembler": "spirv-as",
        "spirvValidator": "spirv-val",
        "semanticReadinessStatus": "no-known-codegen-fallbacks",
        "semanticWarningCount": len(warning_lines),
        "semanticWarningsByIssue": {},
        "semanticBlockers": list(VULKAN_GEMV_SEMANTIC_TRACKED_ISSUES),
        "reportWarningCount": report_warning_count,
        "reportWarningTransportTrackedBy": None,
        "structuralValidationOutput": _relpath(output_path, mlx_root),
        "runtimeIntegrationIncluded": False,
    }


def _runtime_readiness_fixture(
    target: str, variant: str | None = None
) -> dict[str, Any]:
    default_variant = RUNTIME_READINESS_DEFAULT_VARIANTS.get(target)
    _require(
        default_variant is not None,
        f"runtime readiness variant is not configured for target: {target}",
    )
    variant = variant or default_variant
    variant_spec = ARANGE_RUNTIME_VARIANTS.get(variant)
    _require(
        variant_spec is not None,
        f"runtime readiness variant is not configured: {variant}",
    )
    if variant == default_variant:
        entry_point = RUNTIME_READINESS_ENTRY_POINTS.get(target)
    else:
        _require(
            target == "vulkan" and variant in VULKAN_ARANGE_RUNTIME_VARIANTS,
            f"runtime readiness variant {variant} is only configured for Vulkan",
        )
        entry_point = f"arange{variant}"
    _require(
        entry_point is not None,
        f"runtime readiness entry point is not configured for target: {target}",
    )
    fixture_id = f"mlx-arange-{target}-runtime-readiness"
    if variant != default_variant:
        fixture_id = f"mlx-arange-{target}-{variant}-runtime-readiness"
    return {
        "id": fixture_id,
        "selector": {
            "source": MLX_ARANGE_SOURCE,
            "target": target,
        },
        "entryPoint": entry_point,
        "inputs": [
            {
                "name": "start",
                "kind": "scalar",
                "dtype": variant,
                "value": variant_spec["start"],
            },
            {
                "name": "step",
                "kind": "scalar",
                "dtype": variant,
                "value": variant_spec["step"],
            },
        ],
        "expectedOutputs": [
            {
                "name": "out",
                "kind": "buffer",
                "dtype": variant,
                "shape": [4],
                "values": list(variant_spec["expected"]),
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


def _runtime_readiness_fixtures(targets: Sequence[str]) -> list[dict[str, Any]]:
    fixtures = []
    for target in targets:
        variants = (
            VULKAN_ARANGE_RUNTIME_VARIANTS
            if target == "vulkan"
            else (RUNTIME_READINESS_DEFAULT_VARIANTS[target],)
        )
        fixtures.extend(
            _runtime_readiness_fixture(target, variant) for variant in variants
        )
    return fixtures


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
        "fixtures": _runtime_readiness_fixtures(targets),
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
        prepared = dict(state.resource_values)
        for resource in state.plan.resource_bindings:
            value = resource.value
            if value is None or resource.source == "expectedOutput":
                continue
            prepared[value.name] = value.values
        return prepared

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


def _runtime_fixture_scalar(value: Any, *, default: int | float) -> int | float:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, float):
        return value
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


def _runtime_report_results_for_target(
    runtime_report: Mapping[str, Any], target: str
) -> list[Mapping[str, Any]]:
    results = []
    for result in runtime_report.get("results", []):
        if not isinstance(result, Mapping):
            continue
        artifact = result.get("artifact")
        if isinstance(artifact, Mapping) and artifact.get("target") == target:
            results.append(result)
            continue
        fixture = result.get("fixture")
        if isinstance(fixture, Mapping):
            selector = fixture.get("selector")
            if isinstance(selector, Mapping) and selector.get("target") == target:
                results.append(result)
    return results


def _require_native_runtime_results(
    runtime_report: Mapping[str, Any],
    target: str,
) -> None:
    target_results = _runtime_report_results_for_target(runtime_report, target)
    _require(
        bool(target_results)
        and all(result.get("status") == "passed" for result in target_results),
        f"{target} native runtime execution was required for every MLX arange fixture",
    )


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
    required_native_runtime_targets: Sequence[str] = (),
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
    for target in required_native_runtime_targets:
        _require_native_runtime_results(runtime_report, target)
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
    required_native_runtime_targets: Sequence[str] = (),
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
        required_native_runtime_targets=required_native_runtime_targets,
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
    require_opengl_native_runtime: bool,
) -> dict[str, Any]:
    reports = [
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="directx-vulkan-runtime-readiness",
            artifact_report=report_dir / "directx-vulkan-frontier.json",
            targets=("directx", "vulkan"),
            required_native_runtime_targets=(
                ("vulkan",) if require_vulkan_native_runtime else ()
            ),
        ),
        _plan_runtime_readiness_for_report(
            mlx_root=mlx_root,
            report_dir=report_dir,
            name="opengl-runtime-readiness",
            artifact_report=report_dir / "arange-opengl.json",
            targets=("opengl",),
            required_native_runtime_targets=(
                ("opengl",) if require_opengl_native_runtime else ()
            ),
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
    output_dir = work_dir / "out-full-corpus"
    _write_project_config(
        config_path,
        include=f"{MLX_METAL_KERNEL_ROOT}/**/*.metal",
        targets=FULL_CORPUS_TARGETS,
        output_dir=_relpath(output_dir, mlx_root),
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
    fence_contracts = _validate_atomic_fence_contract_report(
        mlx_root,
        output_dir,
        payload,
        exact_report=False,
    )
    diagnostics = payload.get("diagnostics", [])
    _require(isinstance(diagnostics, list), "full-corpus diagnostics must be a list")
    error_diagnostics_by_code = Counter(
        diagnostic.get("code")
        for diagnostic in diagnostics
        if isinstance(diagnostic, Mapping)
        and diagnostic.get("severity") == "error"
        and isinstance(diagnostic.get("code"), str)
    )
    expected_error_diagnostics_by_code = Counter(
        {
            **{
                contract["diagnosticCode"]: 1
                for contract in MLX_FENCE_TARGET_CONTRACTS.values()
            },
            "project.validate.failed-artifact": (
                FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
            ),
        }
    )
    expected_target_counts = {
        target: {
            "translatedCount": EXPECTED_METAL_KERNEL_COUNT - 1,
            "failedCount": 1,
        }
        for target in FULL_CORPUS_TARGETS
    }
    expected_fence_only_result = (
        failed_count == FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
        and summary.get("translatedCount")
        == FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
        and result.returncode == 1
        and target_counts == expected_target_counts
        and artifact_validation.get("failedCount")
        == FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
        and error_diagnostics_by_code == expected_error_diagnostics_by_code
    )
    if not expected_fence_only_result:
        _require(
            FULL_CORPUS_TRANSLATION_TRACKED_ISSUES,
            "full-corpus translation reported failures beyond the expected fence "
            "contract without tracked issue references",
        )
        unexpected_error_diagnostics = (
            error_diagnostics_by_code - expected_error_diagnostics_by_code
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
            "expectedFenceFailureCount": FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
            "unexpectedFailedCount": max(
                int(summary.get("failedCount", 0))
                - FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
                0,
            ),
            "unexpectedErrorDiagnosticsByCode": dict(
                sorted(unexpected_error_diagnostics.items())
            ),
            "fenceContract": {
                "status": "blocked-as-expected",
                "source": MLX_FENCE_SOURCE,
                "targetContracts": fence_contracts,
                "trackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
            },
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
            "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
            "maxTemplateMaterializationWork": (
                FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK
            ),
        }
    return {
        "name": "full-corpus",
        "status": "passed-with-expected-fence-blockers",
        "report": _relpath(report_path, mlx_root),
        "unitCount": EXPECTED_METAL_KERNEL_COUNT,
        "artifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
        "translatedCount": FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT,
        "failedCount": FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT,
        "targets": list(FULL_CORPUS_TARGETS),
        "targetCounts": target_counts,
        "validationFailedCount": artifact_validation.get("failedCount", 0),
        "fenceContract": {
            "status": "blocked-as-expected",
            "source": MLX_FENCE_SOURCE,
            "targetContracts": fence_contracts,
            "trackedIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
        },
        "shaderArtifactsOnly": True,
        "runtimeIntegrationIncluded": False,
        "runtimeParityClaimed": False,
        "trackedTranslationIssues": list(FULL_CORPUS_TRANSLATION_TRACKED_ISSUES),
        "maxTemplateSpecializations": FULL_CORPUS_MAX_TEMPLATE_SPECIALIZATIONS,
        "maxTemplateMaterializationWork": FULL_CORPUS_MAX_TEMPLATE_MATERIALIZATION_WORK,
    }


def run_checks(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    require_metal_toolchain = bool(getattr(args, "require_metal_toolchain", False))
    require_opengl_frontier_toolchain = bool(
        getattr(args, "require_opengl_frontier_toolchain", False)
    )
    require_opengl_gemv_toolchain = bool(
        getattr(args, "require_opengl_gemv_toolchain", False)
    )
    require_opengl_native_runtime = bool(
        getattr(args, "require_opengl_native_runtime", False)
    )
    require_vulkan_gemv_toolchain = bool(
        getattr(args, "require_vulkan_gemv_toolchain", False)
    )
    _require(
        not require_opengl_frontier_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-frontier-toolchain is only valid in reduced-frontier mode",
    )
    _require(
        not require_opengl_gemv_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-gemv-toolchain is only valid in reduced-frontier mode",
    )
    _require(
        not require_opengl_native_runtime or args.mode == REDUCED_FRONTIER_MODE,
        "--require-opengl-native-runtime is only valid in reduced-frontier mode",
    )
    _require(
        not require_vulkan_gemv_toolchain or args.mode == REDUCED_FRONTIER_MODE,
        "--require-vulkan-gemv-toolchain is only valid in reduced-frontier mode",
    )
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
    checks.append(
        _check_metal_roundtrip(
            mlx_root,
            work_dir,
            config_dir,
            report_dir,
            log_dir,
            args.python,
            require_metal_toolchain=require_metal_toolchain,
        )
    )
    if args.mode == REDUCED_FRONTIER_MODE:
        checks.append(
            _check_atomic_fence_contract(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
            )
        )
        checks.append(
            _check_reference_accessor_lvalue_identity(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_directx_toolchain=args.require_directx_toolchain,
                require_opengl_toolchain=require_opengl_frontier_toolchain,
            )
        )
        checks.append(
            _translate_directx_vulkan_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_directx_toolchain=args.require_directx_toolchain,
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
            _check_opengl_frontier(
                mlx_root,
                work_dir,
                config_dir,
                report_dir,
                log_dir,
                args.python,
                require_toolchain=require_opengl_frontier_toolchain,
            )
        )
        if require_opengl_gemv_toolchain:
            checks.append(
                _check_gemv_opengl_toolchain(
                    mlx_root,
                    work_dir,
                    config_dir,
                    report_dir,
                    log_dir,
                    args.python,
                )
            )
        if require_vulkan_gemv_toolchain:
            checks.append(
                _check_gemv_vulkan_toolchain(
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
                require_opengl_native_runtime=require_opengl_native_runtime,
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
    reference_accessor_included = args.mode == REDUCED_FRONTIER_MODE
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
            "metalRoundTripSource": MLX_METAL_ROUNDTRIP_SOURCE,
            "metalRoundTripIncluded": True,
            "metalToolchainRequired": require_metal_toolchain,
            "frontierSources": list(MLX_REDUCED_FRONTIER_SOURCES),
            "cleanFrontierSources": list(MLX_CLEAN_REDUCED_FRONTIER_SOURCES),
            "blockedFrontierSources": list(MLX_BLOCKED_REDUCED_FRONTIER_SOURCES),
            "blockedFrontierIssues": list(FENCE_CONTRACT_TRACKED_ISSUES),
            "fullCorpusTargets": list(FULL_CORPUS_TARGETS),
            "fullCorpusExpectedUnitCount": EXPECTED_METAL_KERNEL_COUNT,
            "fullCorpusExpectedArtifactCount": FULL_CORPUS_EXPECTED_ARTIFACT_COUNT,
            "fullCorpusExpectedTranslatedArtifactCount": (
                FULL_CORPUS_EXPECTED_TRANSLATED_ARTIFACT_COUNT
            ),
            "fullCorpusExpectedFenceFailureCount": (
                FULL_CORPUS_EXPECTED_FENCE_FAILURE_COUNT
            ),
            "shaderArtifactsOnly": True,
            "runtimeIntegrationIncluded": False,
            "runtimeReadinessIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "runtimeFixtureExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "nativeRuntimeExecutionIncluded": args.mode == REDUCED_FRONTIER_MODE,
            "referenceAccessorProofIncluded": reference_accessor_included,
            "referenceAccessorTargets": (
                list(REFERENCE_ACCESSOR_TARGETS) if reference_accessor_included else []
            ),
            "referenceAccessorDirectxToolchainRequired": bool(
                reference_accessor_included and args.require_directx_toolchain
            ),
            "referenceAccessorOpenglToolchainRequired": bool(
                reference_accessor_included and require_opengl_frontier_toolchain
            ),
            "openglFrontierToolchainRequired": require_opengl_frontier_toolchain,
            "openglGemvToolchainRequired": require_opengl_gemv_toolchain,
            "openglNativeRuntimeRequired": require_opengl_native_runtime,
            "vulkanGemvToolchainRequired": require_vulkan_gemv_toolchain,
            "runtimeParityClaimed": False,
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
        "--require-metal-toolchain",
        action="store_true",
        help=(
            "Fail unless the generated Metal round-trip artifact compiles "
            "natively with the macOS Metal compiler."
        ),
    )
    parser.add_argument(
        "--require-directx-toolchain",
        action="store_true",
        help="Fail unless the DirectX HLSL smoke check runs successfully.",
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
        "--require-opengl-native-runtime",
        action="store_true",
        help=(
            "Fail unless the selected MLX OpenGL arangeuint32 artifact executes "
            "natively and passes exact output comparison."
        ),
    )
    parser.add_argument(
        "--require-opengl-frontier-toolchain",
        action="store_true",
        help=(
            "Translate the pinned OpenGL frontier and require native GLSL and "
            "SPIR-V 1.3 validation."
        ),
    )
    parser.add_argument(
        "--require-opengl-gemv-toolchain",
        action="store_true",
        help=(
            "Materialize pinned GEMV for OpenGL and require native GLSL and "
            "SPIR-V 1.3 validation."
        ),
    )
    parser.add_argument(
        "--require-vulkan-gemv-toolchain",
        action="store_true",
        help=(
            "Materialize pinned GEMV for Vulkan, require SPIR-V validation, "
            "and verify the exact tracked semantic blockers."
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
