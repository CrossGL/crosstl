#!/usr/bin/env python3
"""Compile the pinned upstream MLX Metal corpus into one native metallib."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

MLX_REPOSITORY = "https://github.com/ml-explore/mlx"
MLX_COMMIT = "4367c73b60541ddd5a266ce4644fd93d20223b6e"
MLX_METAL_KERNEL_ROOT = "mlx/backend/metal/kernels"
METAL_STANDARD_PROFILE = "metal3.2"
METAL_NAX_PROFILE = "metal4.0"
MACOS_DEPLOYMENT_TARGET = "26.2"
MINIMUM_MACOS_SDK_VERSION = "26.2"
EXPECTED_SOURCE_COUNT = 40
EXPECTED_STANDARD_SOURCE_COUNT = 33
EXPECTED_NAX_SOURCE_COUNT = 7
EXPECTED_UPSTREAM_TEST_COUNT = 776
EXPECTED_UPSTREAM_SKIP_COUNT = 44
# The pinned suite has two additional tests guarded by skipIf("CI" in os.environ).
EXPECTED_UPSTREAM_CI_SKIP_COUNT = 46
EXPECTED_CPP_TEST_CASE_COUNT = 260
EXPECTED_CPP_ASSERTION_COUNT = 3490
REQUIRED_PYTHON_VERSION = (3, 13)
EVIDENCE_KIND = "crosstl-mlx-native-metal-reference-baseline"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_FILENAME = "evidence.json"
DEFAULT_OUTPUT_DIR = ".crosstl-mlx-native-metal"
DEFAULT_JOBS = min(3, os.cpu_count() or 1)
COMMAND_TIMEOUT_SECONDS = 60
COMPILE_TIMEOUT_SECONDS = 1800
LINK_TIMEOUT_SECONDS = 1800
PACKAGE_BUILD_TIMEOUT_SECONDS = 3600
UPSTREAM_TEST_TIMEOUT_SECONDS = 1800
CPP_BUILD_TIMEOUT_SECONDS = 3600
CPP_TEST_TIMEOUT_SECONDS = 1800

MLX_COMPILE_FLAGS = (
    "-x",
    "metal",
    "-Wall",
    "-Wextra",
    "-fno-fast-math",
    "-Wno-c++17-extensions",
    "-Wno-c++20-extensions",
)

# This is the complete recursive *.metal surface at MLX_COMMIT. The path and
# content hashes make a dirty, incomplete, or expanded source tree fail closed.
EXPECTED_SOURCE_SHA256 = (
    (
        "mlx/backend/metal/kernels/arange.metal",
        "ca29a59005b5ad54dccc369542e32804229490a41eb30d568adcd913d1ee68d1",
    ),
    (
        "mlx/backend/metal/kernels/arg_reduce.metal",
        "377ec607c802122c68f07ec400474c2444cdfb0d4978d8ebe232b3c040a1f278",
    ),
    (
        "mlx/backend/metal/kernels/binary.metal",
        "4dadb612a9b768f9d51b3b394b32fc0129d361a55b35d545b3c014c87e00897e",
    ),
    (
        "mlx/backend/metal/kernels/binary_two.metal",
        "a425eb847614267e04a92ee70009ef9250914a327848c8e3a71519a233266142",
    ),
    (
        "mlx/backend/metal/kernels/conv.metal",
        "e33150f2c1596fd83b613c046fa5c637b40fbb44032648117a51426fe58dae53",
    ),
    (
        "mlx/backend/metal/kernels/copy.metal",
        "ed8a579eb6fe6a14c36560d2c8b548baf99e66fa77d300fb4ad7554883820eba",
    ),
    (
        "mlx/backend/metal/kernels/fence.metal",
        "6d3f3c27dd038a0f731054f7466f951b02134ce4124b1c1bbbdabfd9498d1913",
    ),
    (
        "mlx/backend/metal/kernels/fft.metal",
        "3a1fbb38ed64f50a49a20d0c5adb1748d9d06ea20e5931e99aa26be543cb7825",
    ),
    (
        "mlx/backend/metal/kernels/fp_quantized.metal",
        "4b4d2399569c0fcdd529bde470872139b981e4e040f4ac5c194980f00b4372d6",
    ),
    (
        "mlx/backend/metal/kernels/fp_quantized_nax.metal",
        "9874131586aa02391139839edf15a2781a22537c1e6638eb0a7ae17cfa7e314e",
    ),
    (
        "mlx/backend/metal/kernels/gemv.metal",
        "c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f",
    ),
    (
        "mlx/backend/metal/kernels/gemv_masked.metal",
        "c044719fbf8c8de8b5ec37527be9537a60fdd81220a3415fcda1dc1c4097b249",
    ),
    (
        "mlx/backend/metal/kernels/layer_norm.metal",
        "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f",
    ),
    (
        "mlx/backend/metal/kernels/logsumexp.metal",
        "f9bec5e1e5a23d20bedf9ff8d29a8c03bbb5144bc5d751bbfe906d32ee894817",
    ),
    (
        "mlx/backend/metal/kernels/quantized.metal",
        "292aab5a98e3fc047b8ed91343fc10b66e5a92e12c258cde168929520ab2abfd",
    ),
    (
        "mlx/backend/metal/kernels/quantized_nax.metal",
        "4b00e59bf4fa3a6561ff318149cf30734999624da212b0e2b49dd71179f76b0d",
    ),
    (
        "mlx/backend/metal/kernels/random.metal",
        "f1a19b3f11b7b10203824890f13debc6d627959b4e7f17c219e2e9da553c1bd7",
    ),
    (
        "mlx/backend/metal/kernels/reduce.metal",
        "9f232f0f77281b3105b6b905f3c706ef2a96d962ae317946d3b650a650db3ad9",
    ),
    (
        "mlx/backend/metal/kernels/rms_norm.metal",
        "5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee",
    ),
    (
        "mlx/backend/metal/kernels/rope.metal",
        "2b35221ebea033ff43100b85eaaddec5c0e34e3cfd52e7c692573764b2b7d0ab",
    ),
    (
        "mlx/backend/metal/kernels/scaled_dot_product_attention.metal",
        "2d019f31531cf40fcd2a225fbb09aab74f2ae877a1d2afe06b27386000f99e21",
    ),
    (
        "mlx/backend/metal/kernels/scan.metal",
        "49faeb1a36f346b2aa0f68a5f5fe2544af2109fec5357425550f97a79926b6f3",
    ),
    (
        "mlx/backend/metal/kernels/softmax.metal",
        "d19231c66973edc3944f12529d1cc393029e7f7262b914907c710ef9dbcb39e2",
    ),
    (
        "mlx/backend/metal/kernels/sort.metal",
        "be97ea6106fb2e0c0c26f006037408e246fafa012fd2577bfc20cf356475ee61",
    ),
    (
        "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal",
        "79cbc1164ca440df10dc66288677d1657c027fd88f4d285f48825be365dd6362",
    ),
    (
        "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal",
        "5baaa632aa2fdfdfb50e79820e6a9d9b4bfa2e4badb0d9b1fae4774299b76457",
    ),
    (
        "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal",
        "29ce84c7fbc964cc2257ad7a1f5792493f0af502a2e74223d473a26f83493928",
    ),
    (
        "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_3d.metal",
        "bd9985c4bc4d468b1485edf98065f4fb6319bf912d1de2bb766bf183655791dd",
    ),
    (
        "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv_general.metal",
        "df95502fbc02ad2bdd31ac6e00c30ac8379dc07ab2038e915e02dbb813fb789b",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal",
        "6c6e67f5c7fc6b378ca2c849c36a79d696c93e65b3a60d3f5b1f13928c79029e",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.metal",
        "497550cb5b0896fae66f679a6454676871fff8735901c46b997a781660ae7047",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.metal",
        "191402a9a6a18b6cfd5e74309c15e3c748dd59f2f609b9c8324ea514b872d7ed",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.metal",
        "b829a8226b88fbae47d0393d294dcbb339c96bd2cd7f29af8de8b85b423f65a8",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_masked.metal",
        "6cd1102cdcff715226f095869069e5b84eebd3b7965bd5eb7f4a0f130d47354f",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented.metal",
        "fe8a6ac43fd0383c94dff4cb0acf2d6e0b83b2ef692387a4e3307c824c6d14d6",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_segmented_nax.metal",
        "cde0fce87a930b5dbba885bfafed2bb21ea93aea504611490930051140e77ed2",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.metal",
        "6c3f93a9f55d60ec8e6bd14e1b3e88b37ec65185b424042f2fa6436b397aeaed",
    ),
    (
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk_nax.metal",
        "0d3833d3857ccbf0dc19693e0338c7be4c2d942b6555a343839f584c8ed9329b",
    ),
    (
        "mlx/backend/metal/kernels/ternary.metal",
        "e042fe449dbf49761547d244d521bbe0be883eff2e67cffd33f674b95c3a4dc5",
    ),
    (
        "mlx/backend/metal/kernels/unary.metal",
        "51af04126d68e1f5baee5f467268408650d24a68db66e8c044f7f0be3f15368b",
    ),
)


class MlxNativeMetalProofError(RuntimeError):
    """Raised when the upstream native Metal baseline is not satisfied."""


@dataclass(frozen=True)
class CommandResult:
    """Captured command outcome with stable log locations."""

    name: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    stdout_log: Path
    stderr_log: Path
    cwd: Path | None = None


CommandRunner = Callable[..., CommandResult]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxNativeMetalProofError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    resolved_root = root.resolve()
    _require(
        _is_relative_to(resolved, resolved_root),
        f"evidence path resolves outside the output directory: {resolved}",
    )
    return resolved.relative_to(resolved_root).as_posix()


def _file_evidence(path: Path, *, display_path: str) -> dict[str, Any]:
    _require(path.is_file(), f"expected output is not a file: {path}")
    size = path.stat().st_size
    _require(size > 0, f"expected output is empty: {path}")
    return {
        "path": display_path,
        "sizeBytes": size,
        "sha256": _sha256_file(path),
    }


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _prepare_output(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    air_dir = output_dir / "air"
    log_dir = output_dir / "logs"
    metallib_path = output_dir / "mlx.metallib"
    cpp_build_dir = output_dir / "cpp-build"
    _reset_path(air_dir)
    _reset_path(log_dir)
    _reset_path(metallib_path)
    _reset_path(cpp_build_dir)
    _reset_path(output_dir / EVIDENCE_FILENAME)
    air_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    return air_dir, log_dir, metallib_path, cpp_build_dir


def _run_command(
    name: str,
    command: Sequence[str],
    *,
    log_dir: Path,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> CommandResult:
    _require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) is not None,
        f"invalid command log name: {name}",
    )
    command_tuple = tuple(str(argument) for argument in command)
    stdout_log = log_dir / f"{name}.stdout.log"
    stderr_log = log_dir / f"{name}.stderr.log"
    try:
        completed = subprocess.run(
            command_tuple,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd is not None else None,
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
        stderr = f"{type(exc).__name__}: {exc}\n"
    stdout_log.write_text(stdout, encoding="utf-8")
    stderr_log.write_text(stderr, encoding="utf-8")
    return CommandResult(
        name=name,
        command=command_tuple,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        cwd=cwd,
    )


def _invoke(
    runner: CommandRunner,
    name: str,
    command: Sequence[str],
    *,
    log_dir: Path,
    timeout_seconds: int,
    cwd: Path | None = None,
) -> CommandResult:
    command_tuple = tuple(str(argument) for argument in command)
    result = runner(
        name,
        command_tuple,
        log_dir=log_dir,
        timeout_seconds=timeout_seconds,
        cwd=cwd,
    )
    _require(isinstance(result, CommandResult), f"{name} returned no command result")
    _require(result.name == name, f"{name} returned mismatched command evidence")
    _require(
        result.command == command_tuple,
        f"{name} returned evidence for a different command",
    )
    _require(
        result.cwd == cwd, f"{name} returned mismatched working-directory evidence"
    )
    return result


def _failure_excerpt(result: CommandResult) -> str:
    output = result.stderr.strip() or result.stdout.strip()
    if not output:
        return "no diagnostic output"
    lines = output.splitlines()
    return " | ".join(lines[-4:])


def _require_success(result: CommandResult, purpose: str) -> None:
    _require(
        result.returncode == 0,
        f"{purpose} failed with exit code {result.returncode}: "
        f"{_failure_excerpt(result)}",
    )


def _normalize_command_argument(
    argument: str,
    *,
    mlx_root: Path,
    output_dir: Path,
    python_root: Path | None = None,
) -> str:
    roots = [(output_dir, "$OUTPUT_DIR")]
    if python_root is not None:
        roots.append((python_root, "$MLX_PYTHON_ROOT"))
    roots.append((mlx_root, "$MLX_METAL_ROOT"))
    for root, marker in roots:
        root_text = str(root)
        if argument == root_text:
            return marker
        prefix = root_text + os.sep
        if argument.startswith(prefix):
            return marker + "/" + argument[len(prefix) :].replace(os.sep, "/")
    return argument


def _command_evidence(
    result: CommandResult,
    *,
    mlx_root: Path,
    output_dir: Path,
    python_root: Path | None = None,
) -> dict[str, Any]:
    evidence = {
        "name": result.name,
        "command": [
            _normalize_command_argument(
                argument,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            )
            for argument in result.command
        ],
        "returnCode": result.returncode,
        "stdout": {
            "log": _relative_path(result.stdout_log, output_dir),
            "sizeBytes": len(result.stdout.encode("utf-8")),
            "sha256": _sha256_text(result.stdout),
        },
        "stderr": {
            "log": _relative_path(result.stderr_log, output_dir),
            "sizeBytes": len(result.stderr.encode("utf-8")),
            "sha256": _sha256_text(result.stderr),
        },
    }
    if result.cwd is not None:
        evidence["workingDirectory"] = _normalize_command_argument(
            str(result.cwd),
            mlx_root=mlx_root,
            output_dir=output_dir,
            python_root=python_root,
        )
    return evidence


def _parse_version_tuple(value: str, *, label: str) -> tuple[int, ...]:
    _require(
        re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", value) is not None,
        f"could not parse {label} version: {value!r}",
    )
    return tuple(int(component) for component in value.split("."))


def _version_at_least(value: str, minimum: str, *, label: str) -> bool:
    actual_parts = _parse_version_tuple(value, label=label)
    minimum_parts = _parse_version_tuple(minimum, label=f"minimum {label}")
    width = max(len(actual_parts), len(minimum_parts))
    actual = actual_parts + (0,) * (width - len(actual_parts))
    required = minimum_parts + (0,) * (width - len(minimum_parts))
    return actual >= required


def _parse_xcode_version(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    _require(len(lines) == 2, "xcodebuild -version returned an unexpected shape")
    version_match = re.fullmatch(r"Xcode (\S+)", lines[0])
    build_match = re.fullmatch(r"Build version (\S+)", lines[1])
    _require(
        version_match is not None and build_match is not None,
        "xcodebuild -version returned unrecognized output",
    )
    return {
        "version": version_match.group(1),
        "buildVersion": build_match.group(1),
    }


def _parse_metal_version(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    _require(lines, "metal --version returned no output")
    version_match = re.fullmatch(r"Apple metal version (\S+) \(([^)]+)\)", lines[0])
    target_line = next((line for line in lines if line.startswith("Target: ")), None)
    installed_line = next(
        (line for line in lines if line.startswith("InstalledDir: ")), None
    )
    _require(
        version_match is not None
        and target_line is not None
        and installed_line is not None,
        "metal --version returned unrecognized output",
    )
    return {
        "version": version_match.group(1),
        "frontendVersion": version_match.group(2),
        "target": target_line[len("Target: ") :],
        "installedDirectory": installed_line[len("InstalledDir: ") :],
    }


def _parse_sw_vers(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    required = {"ProductName", "ProductVersion", "BuildVersion"}
    _require(required <= values.keys(), "sw_vers returned incomplete platform identity")
    return {
        "productName": values["ProductName"],
        "productVersion": values["ProductVersion"],
        "buildVersion": values["BuildVersion"],
    }


def _parse_metal_nm_symbols(output: str) -> tuple[str, ...]:
    symbols = tuple(line.strip() for line in output.splitlines() if line.strip())
    _require(symbols, "metal-nm reported no defined symbols")
    return symbols


def _parse_python_version(output: str) -> dict[str, Any]:
    match = re.fullmatch(r"Python ([0-9]+)\.([0-9]+)\.([0-9]+)(.*)", output.strip())
    _require(match is not None, "python --version returned unrecognized output")
    version = tuple(int(match.group(index)) for index in range(1, 4))
    _require(
        version[:2] == REQUIRED_PYTHON_VERSION,
        "native MLX Python baseline requires Python "
        f"{REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}; "
        f"found {version[0]}.{version[1]}",
    )
    return {
        "version": ".".join(str(component) for component in version) + match.group(4),
        "major": version[0],
        "minor": version[1],
        "micro": version[2],
    }


def _parse_mlx_runtime_identity(output: str) -> dict[str, str]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise MlxNativeMetalProofError(
            "MLX runtime identity probe did not return JSON"
        ) from exc
    _require(isinstance(payload, dict), "MLX runtime identity must be a JSON object")
    version = payload.get("mlxVersion")
    device = payload.get("defaultDevice")
    device_type = payload.get("defaultDeviceType")
    _require(isinstance(version, str) and version, "MLX version is missing")
    _require(isinstance(device, str) and device, "MLX default device is missing")
    _require(
        device_type == "gpu",
        f"native macOS MLX baseline requires a GPU default device; found {device_type!r}",
    )
    return {
        "mlxVersion": version,
        "defaultDevice": device,
        "defaultDeviceType": device_type,
    }


def _parse_unittest_summary(
    output: str,
    *,
    ci_environment: bool = False,
) -> dict[str, int]:
    ran_matches = re.findall(r"^Ran ([0-9]+) tests? in .+$", output, flags=re.MULTILINE)
    _require(
        len(ran_matches) == 1,
        "upstream Python unittest output has no unique total count",
    )
    ok_matches = re.findall(r"^OK(?: \(([^)]*)\))?$", output, flags=re.MULTILINE)
    _require(len(ok_matches) == 1, "upstream Python unittest output is not successful")
    details = ok_matches[0]
    skipped_match = re.search(r"(?:^|, )skipped=([0-9]+)(?:,|$)", details)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    total = int(ran_matches[0])
    expected_skipped = (
        EXPECTED_UPSTREAM_CI_SKIP_COUNT
        if ci_environment
        else EXPECTED_UPSTREAM_SKIP_COUNT
    )
    _require(
        total == EXPECTED_UPSTREAM_TEST_COUNT and skipped == expected_skipped,
        "upstream Python unittest accounting changed: "
        f"expected {EXPECTED_UPSTREAM_TEST_COUNT} total and "
        f"{expected_skipped} skipped; found {total} total and "
        f"{skipped} skipped",
    )
    return {
        "total": total,
        "passed": total - skipped,
        "skipped": skipped,
        "failures": 0,
        "errors": 0,
    }


def _parse_doctest_summary(output: str) -> dict[str, Any]:
    cases_match = re.search(
        r"^\[doctest\] test cases:\s*([0-9]+)\s*\|\s*"
        r"([0-9]+) passed\s*\|\s*([0-9]+) failed\s*\|\s*"
        r"([0-9]+) skipped$",
        output,
        flags=re.MULTILINE,
    )
    assertions_match = re.search(
        r"^\[doctest\] assertions:\s*([0-9]+)\s*\|\s*"
        r"([0-9]+) passed\s*\|\s*([0-9]+) failed\s*\|$",
        output,
        flags=re.MULTILINE,
    )
    _require(
        cases_match is not None and assertions_match is not None,
        "aggregate C++ doctest output has no complete count summary",
    )
    total_cases, passed_cases, failed_cases, skipped_cases = (
        int(value) for value in cases_match.groups()
    )
    total_assertions, passed_assertions, failed_assertions = (
        int(value) for value in assertions_match.groups()
    )
    _require(
        total_cases == EXPECTED_CPP_TEST_CASE_COUNT
        and passed_cases == EXPECTED_CPP_TEST_CASE_COUNT
        and failed_cases == 0
        and skipped_cases == 0,
        "aggregate C++ doctest case accounting changed: "
        f"found {passed_cases}/{total_cases} passed, {failed_cases} failed, "
        f"{skipped_cases} skipped",
    )
    _require(
        total_assertions == EXPECTED_CPP_ASSERTION_COUNT
        and passed_assertions == EXPECTED_CPP_ASSERTION_COUNT
        and failed_assertions == 0,
        "aggregate C++ doctest assertion accounting changed: "
        f"found {passed_assertions}/{total_assertions} passed and "
        f"{failed_assertions} failed",
    )
    return {
        "testCases": {
            "total": total_cases,
            "passed": passed_cases,
            "failed": failed_cases,
            "skipped": skipped_cases,
        },
        "assertions": {
            "total": total_assertions,
            "passed": passed_assertions,
            "failed": failed_assertions,
        },
    }


def _profile_for_source(source: str) -> str:
    if source.endswith("_nax.metal"):
        return METAL_NAX_PROFILE
    return METAL_STANDARD_PROFILE


def _expected_source_map() -> dict[str, str]:
    paths = [source for source, _ in EXPECTED_SOURCE_SHA256]
    _require(
        len(EXPECTED_SOURCE_SHA256) == EXPECTED_SOURCE_COUNT,
        "expected source manifest count changed",
    )
    _require(paths == sorted(paths), "expected source manifest is not sorted")
    _require(len(set(paths)) == len(paths), "expected source manifest has duplicates")
    for source, digest in EXPECTED_SOURCE_SHA256:
        _require(
            source.startswith(MLX_METAL_KERNEL_ROOT + "/")
            and source.endswith(".metal"),
            f"invalid expected Metal source path: {source}",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            f"invalid expected SHA-256 for {source}",
        )
    profile_counts = {
        METAL_STANDARD_PROFILE: sum(
            _profile_for_source(source) == METAL_STANDARD_PROFILE for source in paths
        ),
        METAL_NAX_PROFILE: sum(
            _profile_for_source(source) == METAL_NAX_PROFILE for source in paths
        ),
    }
    _require(
        profile_counts
        == {
            METAL_STANDARD_PROFILE: EXPECTED_STANDARD_SOURCE_COUNT,
            METAL_NAX_PROFILE: EXPECTED_NAX_SOURCE_COUNT,
        },
        "expected source profile accounting changed",
    )
    return dict(EXPECTED_SOURCE_SHA256)


def _verify_checkout(
    mlx_root: Path,
    *,
    log_dir: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    kernel_root = mlx_root / MLX_METAL_KERNEL_ROOT
    _require(kernel_root.is_dir(), f"MLX Metal kernel tree is missing: {kernel_root}")

    revision_result = _invoke(
        runner,
        "checkout-revision",
        ("git", "-C", str(mlx_root), "rev-parse", "HEAD"),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    _require_success(revision_result, "MLX checkout revision validation")
    revision = revision_result.stdout.strip()
    _require(
        revision == MLX_COMMIT,
        f"MLX checkout must be pinned to {MLX_COMMIT}; found {revision or 'nothing'}",
    )

    status_result = _invoke(
        runner,
        "checkout-kernel-status",
        (
            "git",
            "-C",
            str(mlx_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            MLX_METAL_KERNEL_ROOT,
        ),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    _require_success(status_result, "MLX kernel tree cleanliness validation")
    _require(
        not status_result.stdout.strip(),
        "MLX Metal kernel tree has tracked or untracked changes",
    )
    return {
        "expectedCommit": MLX_COMMIT,
        "revision": revision,
        "kernelRoot": MLX_METAL_KERNEL_ROOT,
        "kernelTreeClean": True,
        "validation": [
            _command_evidence(
                result,
                mlx_root=mlx_root,
                output_dir=output_dir,
            )
            for result in (revision_result, status_result)
        ],
    }


def _verify_python_checkout(
    python_root: Path,
    *,
    mlx_root: Path,
    log_dir: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    _require(python_root.is_dir(), f"full MLX checkout does not exist: {python_root}")
    _require(
        (python_root / "setup.py").is_file(),
        f"full MLX checkout has no setup.py: {python_root}",
    )
    _require(
        (python_root / "python" / "tests").is_dir(),
        f"full MLX checkout has no Python test tree: {python_root}",
    )
    revision_result = _invoke(
        runner,
        "python-checkout-revision",
        ("git", "-C", str(python_root), "rev-parse", "HEAD"),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    _require_success(revision_result, "full MLX checkout revision validation")
    revision = revision_result.stdout.strip()
    _require(
        revision == MLX_COMMIT,
        f"full MLX checkout must be pinned to {MLX_COMMIT}; "
        f"found {revision or 'nothing'}",
    )
    status_result = _invoke(
        runner,
        "python-checkout-status",
        (
            "git",
            "-C",
            str(python_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    _require_success(status_result, "full MLX checkout cleanliness validation")
    _require(
        not status_result.stdout.strip(),
        "full MLX checkout has tracked or untracked changes",
    )
    return {
        "expectedCommit": MLX_COMMIT,
        "revision": revision,
        "fullCheckout": True,
        "treeCleanBeforeBuild": True,
        "validation": [
            _command_evidence(
                result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            )
            for result in (revision_result, status_result)
        ],
    }


def _discover_source_manifest(mlx_root: Path) -> dict[str, Any]:
    expected = _expected_source_map()
    kernel_root = mlx_root / MLX_METAL_KERNEL_ROOT
    discovered_paths: list[str] = []
    absolute_by_source: dict[str, Path] = {}
    for path in kernel_root.rglob("*.metal"):
        _require(not path.is_symlink(), f"Metal source must not be a symlink: {path}")
        _require(path.is_file(), f"Metal source is not a regular file: {path}")
        source = path.relative_to(mlx_root).as_posix()
        discovered_paths.append(source)
        absolute_by_source[source] = path
    discovered_paths.sort()
    expected_paths = list(expected)
    if discovered_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(discovered_paths))
        unexpected = sorted(set(discovered_paths) - set(expected_paths))
        raise MlxNativeMetalProofError(
            "pinned Metal source manifest changed: "
            f"missing={missing or []}, unexpected={unexpected or []}"
        )

    sources: list[dict[str, Any]] = []
    identity_entries: list[dict[str, str]] = []
    profile_counts = {METAL_STANDARD_PROFILE: 0, METAL_NAX_PROFILE: 0}
    total_source_bytes = 0
    for source in discovered_paths:
        path = absolute_by_source[source]
        digest = _sha256_file(path)
        _require(
            digest == expected[source],
            f"pinned source SHA-256 mismatch for {source}: "
            f"expected {expected[source]}, found {digest}",
        )
        profile = _profile_for_source(source)
        size = path.stat().st_size
        profile_counts[profile] += 1
        total_source_bytes += size
        sources.append(
            {
                "source": source,
                "profile": profile,
                "sizeBytes": size,
                "sha256": digest,
            }
        )
        identity_entries.append(
            {"source": source, "profile": profile, "sha256": digest}
        )

    expected_identity = [
        {
            "source": source,
            "profile": _profile_for_source(source),
            "sha256": digest,
        }
        for source, digest in EXPECTED_SOURCE_SHA256
    ]
    expected_manifest_sha256 = _canonical_json_sha256(expected_identity)
    actual_manifest_sha256 = _canonical_json_sha256(identity_entries)
    _require(
        actual_manifest_sha256 == expected_manifest_sha256,
        "pinned Metal source manifest identity hash changed",
    )
    _require(
        len(sources) == EXPECTED_SOURCE_COUNT
        and profile_counts[METAL_STANDARD_PROFILE] == EXPECTED_STANDARD_SOURCE_COUNT
        and profile_counts[METAL_NAX_PROFILE] == EXPECTED_NAX_SOURCE_COUNT,
        "discovered Metal source/profile accounting changed",
    )
    return {
        "sourceCount": len(sources),
        "profileCounts": profile_counts,
        "totalSourceBytes": total_source_bytes,
        "expectedSha256": expected_manifest_sha256,
        "actualSha256": actual_manifest_sha256,
        "sources": sources,
    }


def _probe_toolchain(
    mlx_root: Path,
    *,
    log_dir: Path,
    output_dir: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    commands = (
        ("machine-architecture", ("uname", "-m")),
        ("sdk-path", ("xcrun", "--sdk", "macosx", "--show-sdk-path")),
        ("sdk-version", ("xcrun", "--sdk", "macosx", "--show-sdk-version")),
        (
            "sdk-build-version",
            ("xcrun", "--sdk", "macosx", "--show-sdk-build-version"),
        ),
        ("metal-path", ("xcrun", "--sdk", "macosx", "--find", "metal")),
        (
            "metal-nm-path",
            ("xcrun", "--sdk", "macosx", "--find", "metal-nm"),
        ),
        ("metal-version", ("xcrun", "--sdk", "macosx", "metal", "--version")),
        (
            "metal-nm-version",
            ("xcrun", "--sdk", "macosx", "metal-nm", "--version"),
        ),
        ("xcode-version", ("xcodebuild", "-version")),
        ("macos-version", ("sw_vers",)),
    )
    results: dict[str, CommandResult] = {}
    for name, command in commands:
        result = _invoke(
            runner,
            f"toolchain-{name}",
            command,
            log_dir=log_dir,
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        )
        _require_success(result, f"toolchain probe {name}")
        results[name] = result

    architecture = results["machine-architecture"].stdout.strip()
    _require(
        architecture == "arm64",
        f"native MLX Metal baseline requires arm64; found {architecture or 'nothing'}",
    )
    sdk_path = results["sdk-path"].stdout.strip()
    sdk_version = results["sdk-version"].stdout.strip()
    sdk_build_version = results["sdk-build-version"].stdout.strip()
    metal_path = results["metal-path"].stdout.strip()
    metal_nm_path = results["metal-nm-path"].stdout.strip()
    _require(sdk_path.startswith("/"), "xcrun returned a non-absolute macOS SDK path")
    _require(metal_path.startswith("/"), "xcrun returned a non-absolute metal path")
    _require(
        metal_nm_path.startswith("/"), "xcrun returned a non-absolute metal-nm path"
    )
    _require(
        _version_at_least(
            sdk_version,
            MINIMUM_MACOS_SDK_VERSION,
            label="macOS SDK",
        ),
        f"macOS SDK {sdk_version} is below required {MINIMUM_MACOS_SDK_VERSION}",
    )
    _require(sdk_build_version, "xcrun returned no macOS SDK build version")
    metal_identity = _parse_metal_version(results["metal-version"].stdout)
    metal_nm_version = results["metal-nm-version"].stdout.strip()
    _require(metal_nm_version, "metal-nm --version returned no output")
    return {
        "architecture": architecture,
        "deploymentTarget": MACOS_DEPLOYMENT_TARGET,
        "macos": _parse_sw_vers(results["macos-version"].stdout),
        "sdk": {
            "name": "macosx",
            "path": sdk_path,
            "version": sdk_version,
            "buildVersion": sdk_build_version,
        },
        "xcode": _parse_xcode_version(results["xcode-version"].stdout),
        "metal": {
            "path": metal_path,
            **metal_identity,
        },
        "metalNm": {
            "path": metal_nm_path,
            "versionOutput": metal_nm_version,
            "versionOutputSha256": _sha256_text(results["metal-nm-version"].stdout),
        },
        "probes": [
            _command_evidence(
                results[name],
                mlx_root=mlx_root,
                output_dir=output_dir,
            )
            for name, _ in commands
        ],
    }


def _air_filename(index: int, source: str) -> str:
    source_suffix = source[len(MLX_METAL_KERNEL_ROOT) + 1 :]
    stem = source_suffix[: -len(".metal")].replace("/", "-")
    return f"{index:02d}-{stem}.air"


def _compile_unit(
    index: int,
    source_entry: Mapping[str, Any],
    *,
    mlx_root: Path,
    output_dir: Path,
    air_dir: Path,
    log_dir: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    source = str(source_entry["source"])
    profile = str(source_entry["profile"])
    source_path = mlx_root / source
    air_path = air_dir / _air_filename(index, source)
    _reset_path(air_path)
    command = (
        "xcrun",
        "--sdk",
        "macosx",
        "metal",
        *MLX_COMPILE_FLAGS[:2],
        f"-std={profile}",
        *MLX_COMPILE_FLAGS[2:],
        f"-mmacosx-version-min={MACOS_DEPLOYMENT_TARGET}",
        "-c",
        str(source_path),
        "-I",
        str(mlx_root),
        "-o",
        str(air_path),
    )
    result = _invoke(
        runner,
        f"compile-{index:02d}",
        command,
        log_dir=log_dir,
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    _require_success(result, f"Metal compilation for {source}")
    air = _file_evidence(
        air_path,
        display_path=_relative_path(air_path, output_dir),
    )
    return {
        "index": index,
        "source": {
            "path": source,
            "sizeBytes": source_entry["sizeBytes"],
            "sha256": source_entry["sha256"],
        },
        "profile": profile,
        "deploymentTarget": MACOS_DEPLOYMENT_TARGET,
        "air": air,
        "invocation": _command_evidence(
            result,
            mlx_root=mlx_root,
            output_dir=output_dir,
        ),
    }


def _compile_all(
    source_manifest: Mapping[str, Any],
    *,
    mlx_root: Path,
    output_dir: Path,
    air_dir: Path,
    log_dir: Path,
    runner: CommandRunner,
    jobs: int,
) -> list[dict[str, Any]]:
    sources = source_manifest["sources"]
    _require(isinstance(sources, list), "source manifest entries are missing")
    _require(len(sources) == EXPECTED_SOURCE_COUNT, "compile source count changed")
    print(
        f"Compiling {len(sources)} pinned MLX Metal units with {jobs} job(s).",
        flush=True,
    )
    compiled: list[Any] = [None] * len(sources)
    failures: list[tuple[int, str, Exception]] = []
    futures: dict[Future[dict[str, Any]], tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for index, source_entry in enumerate(sources):
            source = str(source_entry["source"])
            future = executor.submit(
                _compile_unit,
                index,
                source_entry,
                mlx_root=mlx_root,
                output_dir=output_dir,
                air_dir=air_dir,
                log_dir=log_dir,
                runner=runner,
            )
            futures[future] = (index, source)
        for future in as_completed(futures):
            index, source = futures[future]
            try:
                compiled[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append((index, source, exc))
            else:
                print(f"Compiled {index + 1:02d}/{len(sources)}: {source}", flush=True)
    if failures:
        details = "; ".join(f"{source}: {exc}" for _, source, exc in sorted(failures))
        raise MlxNativeMetalProofError(f"one or more Metal compiles failed: {details}")
    _require(
        all(isinstance(unit, dict) for unit in compiled),
        "compile evidence is incomplete",
    )
    return compiled


def _link_and_inspect(
    units: Sequence[Mapping[str, Any]],
    *,
    mlx_root: Path,
    output_dir: Path,
    metallib_path: Path,
    log_dir: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    _require(len(units) == EXPECTED_SOURCE_COUNT, "link AIR input count changed")
    air_paths = [output_dir / str(unit["air"]["path"]) for unit in units]
    _reset_path(metallib_path)
    link_command = (
        "xcrun",
        "--sdk",
        "macosx",
        "metal",
        f"-mmacosx-version-min={MACOS_DEPLOYMENT_TARGET}",
        *(str(path) for path in air_paths),
        "-o",
        str(metallib_path),
    )
    link_result = _invoke(
        runner,
        "link-metallib",
        link_command,
        log_dir=log_dir,
        timeout_seconds=LINK_TIMEOUT_SECONDS,
    )
    _require_success(link_result, "Metal AIR link")
    metallib = _file_evidence(
        metallib_path,
        display_path=_relative_path(metallib_path, output_dir),
    )

    inspect_command = (
        "xcrun",
        "--sdk",
        "macosx",
        "metal-nm",
        "--defined-only",
        "--just-symbol-name",
        str(metallib_path),
    )
    inspect_result = _invoke(
        runner,
        "inspect-metallib",
        inspect_command,
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    _require_success(inspect_result, "metal-nm metallib inspection")
    symbols = _parse_metal_nm_symbols(inspect_result.stdout)
    air_identity = [
        {
            "source": unit["source"]["path"],
            "airSha256": unit["air"]["sha256"],
        }
        for unit in units
    ]
    return {
        "airInputCount": len(air_paths),
        "airInputManifestSha256": _canonical_json_sha256(air_identity),
        "metallib": metallib,
        "invocation": _command_evidence(
            link_result,
            mlx_root=mlx_root,
            output_dir=output_dir,
        ),
        "inspection": {
            "tool": "metal-nm",
            "definedSymbolCount": len(symbols),
            "definedSymbolsSha256": _canonical_json_sha256(symbols),
            "invocation": _command_evidence(
                inspect_result,
                mlx_root=mlx_root,
                output_dir=output_dir,
            ),
        },
    }


def _run_upstream_python_tests(
    *,
    python_root: Path,
    python_executable: str,
    mlx_root: Path,
    output_dir: Path,
    log_dir: Path,
    runner: CommandRunner,
    ci_environment: bool,
) -> dict[str, Any]:
    version_result = _invoke(
        runner,
        "python-version",
        (python_executable, "--version"),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(version_result, "Python version probe")
    python_identity = _parse_python_version(
        (version_result.stdout + "\n" + version_result.stderr).strip()
    )

    build_result = _invoke(
        runner,
        "python-package-build",
        (
            python_executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "--no-build-isolation",
        ),
        log_dir=log_dir,
        timeout_seconds=PACKAGE_BUILD_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(build_result, "pinned MLX editable package build")

    identity_code = (
        "import json; from importlib.metadata import version; "
        "import mlx.core as mx; device = mx.default_device(); "
        "print(json.dumps({'defaultDevice': str(device), "
        "'defaultDeviceType': device.type.name, 'mlxVersion': version('mlx')}, "
        "sort_keys=True))"
    )
    identity_result = _invoke(
        runner,
        "python-mlx-runtime-identity",
        (python_executable, "-c", identity_code),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(identity_result, "MLX runtime identity probe")
    runtime_identity = _parse_mlx_runtime_identity(identity_result.stdout.strip())

    test_result = _invoke(
        runner,
        "python-unittest",
        (
            python_executable,
            "-m",
            "unittest",
            "discover",
            "python/tests",
        ),
        log_dir=log_dir,
        timeout_seconds=UPSTREAM_TEST_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(test_result, "upstream MLX Python unittest suite")
    unittest_output = test_result.stdout + "\n" + test_result.stderr
    counts = _parse_unittest_summary(
        unittest_output,
        ci_environment=ci_environment,
    )
    return {
        "python": python_identity,
        **runtime_identity,
        "packageBuild": {
            "editable": True,
            "buildIsolation": False,
            "invocation": _command_evidence(
                build_result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            ),
        },
        "unitTests": {
            "framework": "unittest",
            "environment": "ci" if ci_environment else "local",
            "expectedSkipCount": (
                EXPECTED_UPSTREAM_CI_SKIP_COUNT
                if ci_environment
                else EXPECTED_UPSTREAM_SKIP_COUNT
            ),
            "counts": counts,
            "invocation": _command_evidence(
                test_result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            ),
        },
        "identityProbes": [
            _command_evidence(
                result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            )
            for result in (version_result, identity_result)
        ],
    }


def _run_upstream_cpp_tests(
    *,
    python_root: Path,
    cpp_build_dir: Path,
    mlx_root: Path,
    output_dir: Path,
    log_dir: Path,
    runner: CommandRunner,
    jobs: int,
) -> dict[str, Any]:
    cmake_version_result = _invoke(
        runner,
        "cpp-cmake-version",
        ("cmake", "--version"),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(cmake_version_result, "CMake version probe")
    ninja_version_result = _invoke(
        runner,
        "cpp-ninja-version",
        ("ninja", "--version"),
        log_dir=log_dir,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(ninja_version_result, "Ninja version probe")
    _require(
        cmake_version_result.stdout.strip().startswith("cmake version "),
        "cmake --version returned unrecognized output",
    )
    _require(
        re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)+(?:[^\s]*)?",
            ninja_version_result.stdout.strip(),
        )
        is not None,
        "ninja --version returned unrecognized output",
    )

    configure_definitions = (
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MACOS_DEPLOYMENT_TARGET}",
        "-DMLX_BUILD_TESTS=ON",
        "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
        "-DMLX_BUILD_EXAMPLES=OFF",
        "-DMLX_BUILD_BENCHMARKS=OFF",
    )
    configure_result = _invoke(
        runner,
        "cpp-configure",
        (
            "cmake",
            "-S",
            str(python_root),
            "-B",
            str(cpp_build_dir),
            "-G",
            "Ninja",
            *configure_definitions,
        ),
        log_dir=log_dir,
        timeout_seconds=PACKAGE_BUILD_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(configure_result, "upstream MLX C++ test configuration")
    build_result = _invoke(
        runner,
        "cpp-build-tests",
        (
            "cmake",
            "--build",
            str(cpp_build_dir),
            "--target",
            "tests",
            "--parallel",
            str(jobs),
        ),
        log_dir=log_dir,
        timeout_seconds=CPP_BUILD_TIMEOUT_SECONDS,
        cwd=python_root,
    )
    _require_success(build_result, "upstream MLX aggregate C++ test build")
    test_executable = cpp_build_dir / "tests" / "tests"
    executable = _file_evidence(
        test_executable,
        display_path=_relative_path(test_executable, output_dir),
    )
    aggregate_result = _invoke(
        runner,
        "cpp-aggregate-tests",
        (str(test_executable),),
        log_dir=log_dir,
        timeout_seconds=CPP_TEST_TIMEOUT_SECONDS,
        cwd=cpp_build_dir,
    )
    _require_success(aggregate_result, "upstream MLX aggregate C++ doctest suite")
    counts = _parse_doctest_summary(
        aggregate_result.stdout + "\n" + aggregate_result.stderr
    )
    return {
        "configuration": {
            "buildType": "Release",
            "generator": "Ninja",
            "deploymentTarget": MACOS_DEPLOYMENT_TARGET,
            "definitions": list(configure_definitions),
        },
        "tools": {
            "cmakeVersionOutput": cmake_version_result.stdout.strip(),
            "ninjaVersion": ninja_version_result.stdout.strip(),
            "probes": [
                _command_evidence(
                    result,
                    mlx_root=mlx_root,
                    output_dir=output_dir,
                    python_root=python_root,
                )
                for result in (cmake_version_result, ninja_version_result)
            ],
        },
        "configure": _command_evidence(
            configure_result,
            mlx_root=mlx_root,
            output_dir=output_dir,
            python_root=python_root,
        ),
        "build": {
            "target": "tests",
            "executable": executable,
            "invocation": _command_evidence(
                build_result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            ),
        },
        "aggregateTests": {
            "framework": "doctest",
            "processCount": 1,
            "counts": counts,
            "invocation": _command_evidence(
                aggregate_result,
                mlx_root=mlx_root,
                output_dir=output_dir,
                python_root=python_root,
            ),
        },
    }


def _claims() -> dict[str, bool]:
    return {
        "nativeUpstreamSourceCompilation": True,
        "nativeUpstreamPythonTests": True,
        "nativeUpstreamCppTests": True,
        "translatedTargetCorrectness": False,
        "translatedTargetRuntimeExecution": False,
        "runtimeParity": False,
        "numericalParity": False,
    }


def run_proof(
    mlx_root: Path,
    python_root: Path,
    output_dir: Path,
    *,
    python_executable: str = sys.executable,
    runner: CommandRunner = _run_command,
    jobs: int = DEFAULT_JOBS,
    ci_environment: bool = False,
) -> dict[str, Any]:
    """Run the complete native baseline and write deterministic JSON evidence."""

    mlx_root = mlx_root.resolve()
    python_root = python_root.resolve()
    output_dir = output_dir.resolve()
    kernel_root = (mlx_root / MLX_METAL_KERNEL_ROOT).resolve()
    _require(1 <= jobs <= 32, "parallel job count must be between 1 and 32")
    _require(python_executable, "Python executable must not be empty")
    _require(
        not _is_relative_to(output_dir, mlx_root)
        and not _is_relative_to(output_dir, python_root),
        "output directory must be outside both MLX checkouts",
    )
    _require(
        _is_relative_to(kernel_root, mlx_root),
        "MLX Metal kernel root resolves outside its checkout",
    )
    air_dir, log_dir, metallib_path, cpp_build_dir = _prepare_output(output_dir)
    source_checkout = _verify_checkout(
        mlx_root,
        log_dir=log_dir,
        output_dir=output_dir,
        runner=runner,
    )
    source_manifest = _discover_source_manifest(mlx_root)
    python_checkout = _verify_python_checkout(
        python_root,
        mlx_root=mlx_root,
        log_dir=log_dir,
        output_dir=output_dir,
        runner=runner,
    )
    toolchain = _probe_toolchain(
        mlx_root,
        log_dir=log_dir,
        output_dir=output_dir,
        runner=runner,
    )
    units = _compile_all(
        source_manifest,
        mlx_root=mlx_root,
        output_dir=output_dir,
        air_dir=air_dir,
        log_dir=log_dir,
        runner=runner,
        jobs=jobs,
    )
    profile_counts = {
        profile: sum(unit["profile"] == profile for unit in units)
        for profile in (METAL_STANDARD_PROFILE, METAL_NAX_PROFILE)
    }
    _require(
        len(units) == EXPECTED_SOURCE_COUNT
        and profile_counts[METAL_STANDARD_PROFILE] == EXPECTED_STANDARD_SOURCE_COUNT
        and profile_counts[METAL_NAX_PROFILE] == EXPECTED_NAX_SOURCE_COUNT,
        "compiled Metal source/profile accounting changed",
    )
    link = _link_and_inspect(
        units,
        mlx_root=mlx_root,
        output_dir=output_dir,
        metallib_path=metallib_path,
        log_dir=log_dir,
        runner=runner,
    )
    upstream_python_tests = _run_upstream_python_tests(
        python_root=python_root,
        python_executable=python_executable,
        mlx_root=mlx_root,
        output_dir=output_dir,
        log_dir=log_dir,
        runner=runner,
        ci_environment=ci_environment,
    )
    upstream_cpp_tests = _run_upstream_cpp_tests(
        python_root=python_root,
        cpp_build_dir=cpp_build_dir,
        mlx_root=mlx_root,
        output_dir=output_dir,
        log_dir=log_dir,
        runner=runner,
        jobs=jobs,
    )
    payload: dict[str, Any] = {
        "kind": EVIDENCE_KIND,
        "schemaVersion": EVIDENCE_SCHEMA_VERSION,
        "status": "passed",
        "scope": {
            "description": "Pinned upstream MLX native Metal reference baseline only",
            "nativeUpstreamReferenceBaselineOnly": True,
        },
        "claims": _claims(),
        "mlx": {
            "repository": MLX_REPOSITORY,
            "expectedCommit": MLX_COMMIT,
            "sourceCheckout": source_checkout,
            "testCheckout": python_checkout,
        },
        "toolchain": toolchain,
        "sourceCompileLink": {
            "sourceManifest": source_manifest,
            "compile": {
                "expectedUnitCount": EXPECTED_SOURCE_COUNT,
                "attemptedUnitCount": len(units),
                "successfulUnitCount": len(units),
                "profileCounts": profile_counts,
                "parallelJobs": jobs,
                "commonFlags": list(MLX_COMPILE_FLAGS),
                "deploymentTarget": MACOS_DEPLOYMENT_TARGET,
                "units": units,
            },
            "link": link,
        },
        "upstreamPythonTests": upstream_python_tests,
        "upstreamCppTests": upstream_cpp_tests,
    }
    evidence_path = output_dir / EVIDENCE_FILENAME
    _write_json(evidence_path, payload)
    print(f"Evidence: {evidence_path}", flush=True)
    print(f"Evidence SHA-256: {_sha256_file(evidence_path)}", flush=True)
    return payload


def _failure_payload(exc: Exception) -> dict[str, Any]:
    claims = _claims()
    claims["nativeUpstreamSourceCompilation"] = False
    claims["nativeUpstreamPythonTests"] = False
    claims["nativeUpstreamCppTests"] = False
    return {
        "kind": EVIDENCE_KIND,
        "schemaVersion": EVIDENCE_SCHEMA_VERSION,
        "status": "failed",
        "scope": {
            "description": "Pinned upstream MLX native Metal reference baseline only",
            "nativeUpstreamReferenceBaselineOnly": True,
        },
        "claims": claims,
        "mlx": {
            "repository": MLX_REPOSITORY,
            "expectedCommit": MLX_COMMIT,
            "kernelRoot": MLX_METAL_KERNEL_ROOT,
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _positive_jobs(value: str) -> int:
    try:
        jobs = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("jobs must be an integer") from exc
    if not 1 <= jobs <= 32:
        raise argparse.ArgumentTypeError("jobs must be between 1 and 32")
    return jobs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile the pinned upstream MLX Metal source baseline."
    )
    parser.add_argument(
        "--mlx-root",
        type=Path,
        required=True,
        help="Path to the pinned MLX checkout.",
    )
    parser.add_argument(
        "--mlx-python-root",
        type=Path,
        required=True,
        help="Path to a clean full checkout used for upstream Python and C++ tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Evidence and build directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--jobs",
        type=_positive_jobs,
        default=DEFAULT_JOBS,
        help=f"Maximum concurrent Metal compiles (default: {DEFAULT_JOBS}).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python 3.13 interpreter used to build and test MLX.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        run_proof(
            args.mlx_root,
            args.mlx_python_root,
            output_dir,
            python_executable=args.python_executable,
            jobs=args.jobs,
            ci_environment=os.environ.get("CI", "").strip().lower()
            not in {"", "0", "false", "no", "off"},
        )
    except Exception as exc:  # noqa: BLE001
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / EVIDENCE_FILENAME
        _write_json(failure_path, _failure_payload(exc))
        print(f"Native Metal baseline failed: {exc}", file=sys.stderr, flush=True)
        print(f"Failure evidence: {failure_path}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
