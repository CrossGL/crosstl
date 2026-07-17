#!/usr/bin/env python3
"""Prove selected pinned MLX LayerNorm entries as standalone DirectX artifacts."""

from __future__ import annotations

import argparse
import ast
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
MLX_LAYER_NORM_SOURCE = "mlx/backend/metal/kernels/layer_norm.metal"
MLX_LAYER_NORM_SHA256 = (
    "2d243f5abea7353929f9bc838ceb5a98e52a452dfc29609ad4d5974447ea689f"
)
MLX_NORMALIZATION_HOST_SOURCE = "mlx/backend/metal/normalization.cpp"
MLX_NORMALIZATION_HOST_SHA256 = (
    "7c1483b439db051d3f9170dd446314b4602ba1682adb03e804216c3495b49eea"
)
MLX_LAYER_NORM_TEST_SOURCE = "python/tests/test_fast.py"
MLX_LAYER_NORM_TEST_SHA256 = (
    "728142b9d2567a1b5e1ae44fd15d7f008317989a6f44a9d8e51a5a236f45d82e"
)
LAYER_NORM_FUNCTION_CONSTANT_NAME = "has_w"
LAYER_NORM_FUNCTION_CONSTANT_ID = 20
LAYER_NORM_SUBGROUP_WIDTH = 32
LAYER_NORM_DIRECTX_PROFILE = "cs_6_6"
LAYER_NORM_INSTANTIATION_MARKER = "// clang-format off\n#define instantiate_layer_norm"
LAYER_NORM_DIRECTX_CASES = (
    {
        "name": "forward_float32_axis_4099",
        "entryPoint": "layer_normfloat32",
        "templateName": "layer_norm_single_row",
        "hostFunction": "LayerNorm::eval_gpu",
        "testFunction": "test_layer_norm",
        "testAxisPattern": (
            r"dims\s*,\s*dtype\s*,\s*eps\s*=\s*4099\s*,\s*" r"mx\.float32\s*,\s*1e-5"
        ),
        "axisSize": 4099,
        "simdSize": 32,
        "nReads": 8,
        "loopedLimit": 6656,
        "workgroupSize": [544, 1, 1],
        "selector": LAYER_NORM_FUNCTION_CONSTANT_NAME,
        "selectorKind": "name",
        "value": False,
        "usesFunctionConstant": False,
    },
    {
        "name": "vjp_float32_axis_8192_has_w",
        "entryPoint": "vjp_layer_normfloat32",
        "templateName": "vjp_layer_norm_single_row",
        "hostFunction": "LayerNormVJP::eval_gpu",
        "testFunction": "test_layer_norm_grad",
        "testAxisPattern": r"D\s*=\s*8192",
        "axisSize": 8192,
        "simdSize": 32,
        "nReads": 8,
        "loopedLimit": 8192,
        "workgroupSize": [1024, 1, 1],
        "selector": str(LAYER_NORM_FUNCTION_CONSTANT_ID),
        "selectorKind": "id",
        "value": True,
        "usesFunctionConstant": True,
    },
)
DEFAULT_WORK_DIR = ".crosstl-mlx-porting/layer-norm-directx"


class MlxLayerNormDirectXProofError(RuntimeError):
    """Raised when the pinned LayerNorm DirectX proof is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MlxLayerNormDirectXProofError(message)


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _git_blob(
    mlx_root: Path,
    revision: str,
    path: str,
    *,
    log_dir: Path,
) -> bytes:
    name = f"mlx-layernorm-git-show-{Path(path).name}"
    command = ["git", "-C", str(mlx_root), "show", f"{revision}:{path}"]
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout"
    stderr_path = log_dir / f"{name}.stderr"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=180,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or b""
        stderr = (exc.stderr or b"") + b"\ngit show timed out after 180 seconds.\n"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    _require(
        returncode == 0,
        f"could not read pinned MLX evidence blob: {path}",
    )
    return stdout


def _cpp_function_body(source: str, qualified_name: str) -> str:
    signature = re.compile(
        rf"\bvoid\s+{re.escape(qualified_name)}\s*\(",
        flags=re.MULTILINE,
    )
    matches = list(signature.finditer(source))
    _require(
        len(matches) == 1,
        f"host evidence for {qualified_name} must resolve to one function",
    )
    opening = source.find("{", matches[0].end())
    _require(opening >= 0, f"host evidence for {qualified_name} has no body")
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise MlxLayerNormDirectXProofError(
        f"host evidence for {qualified_name} has an unterminated body"
    )


def _metal_function_definition(source: str, function_name: str) -> str:
    signature = re.compile(
        rf"\b(?:inline\s+)?void\s+{re.escape(function_name)}\s*\(",
        flags=re.MULTILINE,
    )
    matches = list(signature.finditer(source))
    _require(
        len(matches) == 1,
        f"source evidence for {function_name} must resolve to one function",
    )
    opening = source.find("{", matches[0].end())
    _require(opening >= 0, f"source evidence for {function_name} has no body")
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[matches[0].start() : index + 1]
    raise MlxLayerNormDirectXProofError(
        f"source evidence for {function_name} has an unterminated body"
    )


def _unique_cpp_integer(body: str, name: str, *, function: str) -> int:
    matches = re.findall(rf"\bint\s+{re.escape(name)}\s*=\s*(\d+)\s*;", body)
    _require(
        len(matches) == 1,
        f"host evidence for {function} must declare one concrete {name}",
    )
    return int(matches[0])


def _derive_single_row_workgroup(
    *,
    axis_size: int,
    simd_size: int,
    n_reads: int,
) -> list[int]:
    _require(axis_size > 0, "LayerNorm axis size must be positive")
    _require(simd_size > 0, "LayerNorm SIMD size must be positive")
    _require(n_reads > 0, "LayerNorm read count must be positive")
    threadgroup_needed = (axis_size + n_reads - 1) // n_reads
    simds_needed = (threadgroup_needed + simd_size - 1) // simd_size
    return [simd_size * simds_needed, 1, 1]


def _python_function_source(source: str, function_name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise MlxLayerNormDirectXProofError(
            f"pinned MLX test evidence is not valid Python: {exc}"
        ) from exc
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    _require(
        len(matches) == 1,
        f"test evidence for {function_name} must resolve to one function",
    )
    segment = ast.get_source_segment(source, matches[0])
    _require(
        isinstance(segment, str) and bool(segment),
        f"test evidence for {function_name} has no source segment",
    )
    return segment


def _verify_dispatch_evidence(
    host_source: str,
    test_source: str,
) -> dict[str, Any]:
    formula_patterns = (
        r"threadgroup_needed\s*=\s*"
        r"\(axis_size\s*\+\s*n_reads\s*-\s*1\)\s*/\s*n_reads\s*;",
        r"simds_needed\s*=\s*"
        r"\(threadgroup_needed\s*\+\s*simd_size\s*-\s*1\)\s*/\s*"
        r"simd_size\s*;",
        r"threadgroup_size\s*=\s*simd_size\s*\*\s*simds_needed\s*;",
    )
    cases = []
    for case in LAYER_NORM_DIRECTX_CASES:
        function = str(case["hostFunction"])
        body = _cpp_function_body(host_source, function)
        simd_size = _unique_cpp_integer(body, "simd_size", function=function)
        n_reads = _unique_cpp_integer(body, "n_reads", function=function)
        looped_limit = _unique_cpp_integer(body, "looped_limit", function=function)
        for pattern in formula_patterns:
            _require(
                len(re.findall(pattern, body)) == 1,
                f"host evidence for {function} has an ambiguous dispatch formula",
            )
        _require(
            len(
                re.findall(
                    r"if\s*\(axis_size\s*<=\s*looped_limit\)\s*\{",
                    body,
                )
            )
            == 1,
            f"host evidence for {function} has an ambiguous single-row branch",
        )
        axis_size = int(case["axisSize"])
        _require(
            axis_size <= looped_limit,
            f"{case['name']} does not select the single-row host branch",
        )
        workgroup_size = _derive_single_row_workgroup(
            axis_size=axis_size,
            simd_size=simd_size,
            n_reads=n_reads,
        )
        _require(
            simd_size == case["simdSize"]
            and n_reads == case["nReads"]
            and looped_limit == case["loopedLimit"]
            and workgroup_size == case["workgroupSize"],
            f"pinned host dispatch evidence changed for {case['name']}",
        )
        _require(
            workgroup_size[0] <= 1024,
            f"{case['name']} exceeds the DirectX compute threadgroup limit",
        )

        test_function = str(case["testFunction"])
        test_body = _python_function_source(test_source, test_function)
        _require(
            len(re.findall(str(case["testAxisPattern"]), test_body)) == 1,
            f"test evidence for {case['name']} does not identify one axis size",
        )
        _require(
            "mx.fast.layer_norm" in test_body,
            f"test evidence for {case['name']} does not exercise fast LayerNorm",
        )
        if bool(case["value"]):
            _require(
                re.search(r"\bw\s*=\s*mx\.random\.uniform\(", test_body) is not None,
                f"test evidence for {case['name']} does not provide a weight",
            )
        cases.append(
            {
                "name": case["name"],
                "entryPoint": case["entryPoint"],
                "hostFunction": function,
                "testFunction": test_function,
                "axisSize": axis_size,
                "simdSize": simd_size,
                "nReads": n_reads,
                "loopedLimit": looped_limit,
                "branch": "single-row",
                "formula": (
                    "simd_size * ceil_div(ceil_div(axis_size, n_reads), " "simd_size)"
                ),
                "workgroupSize": workgroup_size,
            }
        )
    return {
        "name": "pinned-host-dispatch-evidence",
        "status": "passed",
        "hostSource": MLX_NORMALIZATION_HOST_SOURCE,
        "hostSourceHash": {
            "algorithm": "sha256",
            "value": MLX_NORMALIZATION_HOST_SHA256,
        },
        "testSource": MLX_LAYER_NORM_TEST_SOURCE,
        "testSourceHash": {
            "algorithm": "sha256",
            "value": MLX_LAYER_NORM_TEST_SHA256,
        },
        "cases": cases,
        "loopedEntriesIncluded": False,
        "loopedEntriesExcludedBecause": (
            "Pinned host dispatch uses the runtime pipeline value "
            "maxTotalThreadsPerThreadgroup()."
        ),
    }


def _verify_source_subgroup_contract(source: str) -> dict[str, Any]:
    helper_name = "threadgroup_sum"
    helper = _metal_function_definition(source, helper_name)
    simd_sum_call_count = len(re.findall(r"\bsimd_sum\s*\(", helper))
    _require(
        simd_sum_call_count == 2,
        "pinned LayerNorm threadgroup_sum must contain two simd_sum reductions",
    )

    attributes = (
        "thread_index_in_simdgroup",
        "simdgroup_index_in_threadgroup",
    )
    selected_entries = []
    for case in LAYER_NORM_DIRECTX_CASES:
        template_name = str(case["templateName"])
        definition = _metal_function_definition(source, template_name)
        declared_widths = [
            int(value)
            for value in re.findall(
                r"\bconstexpr\s+int\s+SIMD_SIZE\s*=\s*(\d+)\s*;",
                definition,
            )
        ]
        _require(
            declared_widths == [LAYER_NORM_SUBGROUP_WIDTH],
            f"{template_name} must declare exactly SIMD_SIZE = "
            f"{LAYER_NORM_SUBGROUP_WIDTH}",
        )
        attribute_counts = {
            attribute: len(
                re.findall(
                    rf"\[\[\s*{re.escape(attribute)}\s*\]\]",
                    definition,
                )
            )
            for attribute in attributes
        }
        _require(
            all(count == 1 for count in attribute_counts.values()),
            f"{template_name} must declare one lane and one SIMD-group index",
        )
        threadgroup_sum_call_count = len(
            re.findall(r"\bthreadgroup_sum(?:\s*<[^>]+>)?\s*\(", definition)
        )
        _require(
            threadgroup_sum_call_count == 2,
            f"{template_name} must contain two threadgroup_sum reductions",
        )
        selected_entries.append(
            {
                "entryPoint": case["entryPoint"],
                "templateName": template_name,
                "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
                "attributeCounts": attribute_counts,
                "threadgroupSumCallCount": threadgroup_sum_call_count,
            }
        )

    return {
        "name": "pinned-source-subgroup-contract",
        "status": "passed",
        "source": MLX_LAYER_NORM_SOURCE,
        "sourceHash": {
            "algorithm": "sha256",
            "value": MLX_LAYER_NORM_SHA256,
        },
        "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
        "sourceDeclaration": f"constexpr int SIMD_SIZE = {LAYER_NORM_SUBGROUP_WIDTH};",
        "requiredAttributes": list(attributes),
        "reduction": {
            "helper": helper_name,
            "primitive": "simd_sum",
            "simdSumCallCount": simd_sum_call_count,
        },
        "selectedEntries": selected_entries,
        "requiredDirectXEnforcement": {
            "attribute": f"WaveSize({LAYER_NORM_SUBGROUP_WIDTH})",
            "minimumShaderModel": "6.6",
            "profile": LAYER_NORM_DIRECTX_PROFILE,
        },
        "provenance": {
            "kind": "pinned-source-contract",
            "repository": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
            "path": MLX_LAYER_NORM_SOURCE,
        },
    }


def _verify_mlx_checkout(
    mlx_root: Path,
    *,
    log_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(mlx_root.is_dir(), f"MLX checkout does not exist: {mlx_root}")
    source_path = mlx_root / MLX_LAYER_NORM_SOURCE
    _require(source_path.is_file(), f"pinned MLX source is missing: {source_path}")

    revision_result = _run_command(
        "mlx-layernorm-revision",
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

    source_hash = _sha256(source_path)
    _require(
        source_hash == MLX_LAYER_NORM_SHA256,
        "pinned MLX layer_norm.metal SHA-256 mismatch: "
        f"expected {MLX_LAYER_NORM_SHA256}, found {source_hash}",
    )
    source = source_path.read_text(encoding="utf-8")
    _require(
        len(
            re.findall(
                r"\bconstant\s+bool\s+has_w\s*"
                r"\[\[\s*function_constant\(20\)\s*\]\]\s*;",
                source,
            )
        )
        == 1,
        "pinned LayerNorm function-constant declaration is ambiguous",
    )

    host_blob = _git_blob(
        mlx_root,
        revision,
        MLX_NORMALIZATION_HOST_SOURCE,
        log_dir=log_dir,
    )
    test_blob = _git_blob(
        mlx_root,
        revision,
        MLX_LAYER_NORM_TEST_SOURCE,
        log_dir=log_dir,
    )
    _require(
        _sha256_bytes(host_blob) == MLX_NORMALIZATION_HOST_SHA256,
        "pinned MLX normalization.cpp SHA-256 mismatch",
    )
    _require(
        _sha256_bytes(test_blob) == MLX_LAYER_NORM_TEST_SHA256,
        "pinned MLX test_fast.py SHA-256 mismatch",
    )
    dispatch_evidence = _verify_dispatch_evidence(
        host_blob.decode("utf-8"),
        test_blob.decode("utf-8"),
    )
    subgroup_evidence = _verify_source_subgroup_contract(source)
    return (
        {
            "name": "pinned-source-identity",
            "status": "passed",
            "repository": MLX_REPOSITORY,
            "commit": revision,
            "source": MLX_LAYER_NORM_SOURCE,
            "sourceHash": {
                "algorithm": "sha256",
                "value": source_hash,
            },
        },
        dispatch_evidence,
        subgroup_evidence,
    )


def _selected_instantiation_source(
    source: str,
    case: Mapping[str, Any],
) -> tuple[str, str]:
    marker_count = source.count(LAYER_NORM_INSTANTIATION_MARKER)
    _require(
        marker_count == 1,
        "pinned LayerNorm instantiation manifest must have one exact marker",
    )
    preserved_prefix, _separator, _manifest = source.partition(
        LAYER_NORM_INSTANTIATION_MARKER
    )
    projected = (
        preserved_prefix
        + "// clang-format off\n"
        + f'instantiate_kernel("{case["entryPoint"]}", '
        + f'{case["templateName"]}, float)\n'
        + "// clang-format on\n"
    )
    return projected, _sha256_bytes(preserved_prefix.encode("utf-8"))


def _prepare_case_project(
    mlx_root: Path,
    work_dir: Path,
    case: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    source_path = mlx_root / MLX_LAYER_NORM_SOURCE
    _require(source_path.is_file(), f"pinned MLX source is missing: {source_path}")
    source_hash = _sha256(source_path)
    _require(
        source_hash == MLX_LAYER_NORM_SHA256,
        "LayerNorm source changed before project projection",
    )
    source = source_path.read_text(encoding="utf-8")
    projected, preserved_prefix_hash = _selected_instantiation_source(source, case)

    project_root = work_dir / "projects" / str(case["name"])
    if project_root.exists():
        shutil.rmtree(project_root)
    projected_kernel_root = project_root / MLX_METAL_KERNEL_ROOT
    shutil.copytree(
        mlx_root / MLX_METAL_KERNEL_ROOT,
        projected_kernel_root,
    )
    projected_source_path = project_root / MLX_LAYER_NORM_SOURCE
    projected_source_path.write_text(projected, encoding="utf-8")
    projected_hash = _sha256(projected_source_path)
    return (
        project_root,
        {
            "kind": "selected-host-instantiation-project",
            "projectRoot": _relpath(project_root, mlx_root),
            "source": MLX_LAYER_NORM_SOURCE,
            "sourceHash": {
                "algorithm": "sha256",
                "value": projected_hash,
            },
            "derivedFrom": {
                "repository": MLX_REPOSITORY,
                "commit": MLX_COMMIT,
                "source": MLX_LAYER_NORM_SOURCE,
                "sourceHash": {
                    "algorithm": "sha256",
                    "value": source_hash,
                },
            },
            "transform": {
                "kind": "instantiation-manifest-selection",
                "selectedEntryPoint": case["entryPoint"],
                "templateName": case["templateName"],
                "templateArguments": {"T": "float"},
                "preservedPrefixHash": {
                    "algorithm": "sha256",
                    "value": preserved_prefix_hash,
                },
            },
        },
    )


def _project_config(
    project_root: Path,
    case: Mapping[str, Any],
) -> ProjectConfig:
    variant_name = str(case["name"])
    return ProjectConfig(
        root=project_root,
        source_roots=(MLX_METAL_KERNEL_ROOT,),
        include_patterns=(MLX_LAYER_NORM_SOURCE,),
        targets=("directx",),
        output_dir="crosstl-out",
        source_overrides={"**/*.metal": "metal"},
        entry_points={MLX_LAYER_NORM_SOURCE: str(case["entryPoint"])},
        include_dirs=(".",),
        variants={variant_name: {}},
        variant_specialization_constants=(
            {variant_name: {str(case["selector"]): case["value"]}}
            if case["usesFunctionConstant"]
            else {}
        ),
        workgroup_size_rules={
            MLX_LAYER_NORM_SOURCE: [
                str(component) for component in case["workgroupSize"]
            ]
        },
        subgroup_width_rules={MLX_LAYER_NORM_SOURCE: str(LAYER_NORM_SUBGROUP_WIDTH)},
        selected_variants=(variant_name,),
    )


def _translate_case(
    config: ProjectConfig,
    case: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    try:
        payload = translate_project(config, validate=True).to_json()
    except Exception as exc:  # noqa: BLE001
        raise MlxLayerNormDirectXProofError(
            f"DirectX project translation for {case['name']} raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _require(isinstance(payload, Mapping), "project report must be a JSON object")
    normalized = dict(payload)
    _write_json(report_path, normalized)
    return normalized


def _is_sha256_identity(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("algorithm") == "sha256"
        and isinstance(value.get("value"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["value"]) is not None
    )


def _artifact_path(artifact: Mapping[str, Any], project_root: Path) -> Path:
    value = artifact.get("path")
    _require(isinstance(value, str) and value, "artifact path is missing")
    path = (project_root / value).resolve()
    _require(
        _is_relative_to(path, project_root.resolve()),
        f"artifact path resolves outside the proof project: {value}",
    )
    _require(path.is_file(), f"translated artifact is missing: {path}")
    return path


def _validate_project_report(
    payload: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Mapping[str, Any]:
    name = str(case["name"])
    _require(
        payload.get("kind") == "crosstl-project-portability-report",
        f"{name} did not produce a project portability report",
    )
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), f"{name} report summary is missing")
    _require(
        summary.get("unitCount") == 1
        and summary.get("artifactCount") == 1
        and summary.get("translatedCount") == 1
        and summary.get("failedCount") == 0
        and summary.get("diagnosticCounts", {}).get("error") == 0,
        f"{name} report does not contain one successful artifact",
    )
    diagnostics = payload.get("diagnostics")
    _require(isinstance(diagnostics, list), f"{name} diagnostics are missing")
    unexpected = [
        diagnostic
        for diagnostic in diagnostics
        if not (
            isinstance(diagnostic, Mapping)
            and diagnostic.get("severity") == "warning"
            and diagnostic.get("code") == "project.validate.toolchain-unavailable"
            and diagnostic.get("target") == "directx"
        )
    ]
    _require(not unexpected, f"{name} emitted unexpected project diagnostics")

    validation = payload.get("validation")
    validation_summary = (
        validation.get("summary") if isinstance(validation, Mapping) else None
    )
    _require(
        isinstance(validation_summary, Mapping)
        and validation_summary.get("artifactCount") == 1
        and validation_summary.get("okCount") == 1
        and validation_summary.get("failedCount") == 0,
        f"{name} artifact validation did not pass",
    )

    project = payload.get("project")
    _require(isinstance(project, Mapping), f"{name} project metadata is missing")
    expected_variant = {name: {}}
    expected_workgroup_rule = {
        MLX_LAYER_NORM_SOURCE: [str(component) for component in case["workgroupSize"]]
    }
    expected_subgroup_rule = {MLX_LAYER_NORM_SOURCE: str(LAYER_NORM_SUBGROUP_WIDTH)}
    expected_specialization_constants = (
        {name: {str(case["selector"]): case["value"]}}
        if case["usesFunctionConstant"]
        else {}
    )
    _require(
        project.get("targets") == ["directx"]
        and project.get("entryPointSelections")
        == {MLX_LAYER_NORM_SOURCE: case["entryPoint"]}
        and project.get("entryPointSelectionCount") == 1
        and project.get("variants") == expected_variant
        and project.get("variantCount") == 1
        and project.get("selectedVariants") == [name]
        and project.get("variantWorkgroupSizes") == {}
        and project.get("variantSpecializationConstants")
        == expected_specialization_constants
        and project.get("workgroupSize") is None
        and project.get("workgroupSizeRules") == expected_workgroup_rule
        and project.get("workgroupSizeRuleCount") == 1
        and project.get("subgroupWidthRules") == expected_subgroup_rule
        and project.get("subgroupWidthRuleCount") == 1,
        f"{name} report did not retain the exact project execution rules",
    )

    artifacts = payload.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and len(artifacts) == 1
        and isinstance(artifacts[0], Mapping),
        f"{name} report must contain exactly one artifact record",
    )
    return artifacts[0]


def _selected_materialization(
    artifact: Mapping[str, Any],
    case: Mapping[str, Any],
) -> Mapping[str, Any]:
    materialization = artifact.get("templateMaterialization")
    specializations = (
        materialization.get("specializations")
        if isinstance(materialization, Mapping)
        else None
    )
    _require(
        isinstance(materialization, Mapping)
        and materialization.get("status") == "materialized"
        and materialization.get("unsupported") == []
        and isinstance(specializations, list)
        and materialization.get("specializationCount") == len(specializations),
        f"{case['name']} template materialization is incomplete",
    )
    entry_point = str(case["entryPoint"])
    matches = [
        record
        for record in specializations
        if isinstance(record, Mapping)
        and (
            record.get("hostName") == entry_point
            or record.get("materializedName") == entry_point
        )
    ]
    _require(
        len(matches) == 1,
        f"{case['name']} must resolve to one host-named materialization",
    )
    selected = matches[0]
    _require(
        selected
        == {
            "name": case["templateName"],
            "materializedName": entry_point,
            "parameters": {"N_READS": "8", "T": "float"},
            "parameterSources": {
                "N_READS": "source-default",
                "T": "source-instantiation",
            },
            "source": "source-instantiation",
            "hostName": entry_point,
        },
        f"{case['name']} selected materialization metadata changed",
    )
    return selected


def _validate_specialization(
    artifact: Mapping[str, Any],
    generated: str,
    case: Mapping[str, Any],
) -> dict[str, Any] | None:
    constants = artifact.get("specializationConstants")
    if not case["usesFunctionConstant"]:
        _require(
            constants in (None, [])
            and "specializationMaterialization" not in artifact
            and re.search(r"\b(?:static\s+const\s+bool\s+)?has_w\b", generated) is None,
            f"{case['name']} retained an unreachable function constant",
        )
        return None

    _require(
        isinstance(constants, list) and len(constants) == 1,
        f"{case['name']} must report one function constant",
    )
    constant = constants[0]
    expected_provenance = {
        "kind": "project-variant",
        "path": (
            f"project.variants.{case['name']}.specialization_constants."
            f"{case['selector']}"
        ),
        "selector": case["selector"],
        "selectorKind": case["selectorKind"],
        "variant": case["name"],
    }
    _require(
        isinstance(constant, Mapping)
        and constant.get("name") == LAYER_NORM_FUNCTION_CONSTANT_NAME
        and constant.get("id") == LAYER_NORM_FUNCTION_CONSTANT_ID
        and constant.get("sourceType") == "bool"
        and constant.get("required") is True
        and constant.get("overridden") is True
        and constant.get("deferred") is False
        and constant.get("concreteValue") is case["value"]
        and constant.get("valueProvenance") == expected_provenance,
        f"{case['name']} function-constant metadata changed",
    )
    _require(
        artifact.get("specializationMaterialization")
        == {
            "status": "concrete",
            "mode": "concrete-crossgl-variant",
            "targetSupportsDeferredSpecialization": False,
            "constantCount": 1,
            "requiredCount": 1,
            "overriddenCount": 1,
            "concreteCount": 1,
            "source": "shared-crossgl-specialization",
        },
        f"{case['name']} function-constant materialization changed",
    )
    literal = "true" if case["value"] else "false"
    static_constants = re.findall(
        r"\bstatic\s+const\s+bool\s+has_w\s*=\s*(true|false)\s*;",
        generated,
    )
    _require(
        static_constants == [literal],
        f"{case['name']} emitted an incorrect concrete has_w constant set",
    )
    return {
        "name": LAYER_NORM_FUNCTION_CONSTANT_NAME,
        "id": LAYER_NORM_FUNCTION_CONSTANT_ID,
        "selector": case["selector"],
        "selectorKind": case["selectorKind"],
        "value": case["value"],
        "valueProvenance": expected_provenance,
        "emittedInSelectedEntry": True,
    }


def _validate_execution(
    artifact: Mapping[str, Any],
    artifact_path: Path,
    generated: str,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    entry_point = str(case["entryPoint"])
    workgroup_size = list(case["workgroupSize"])
    execution = artifact.get("execution")
    workgroup_rule_path = f'project.workgroup_size_rules["{MLX_LAYER_NORM_SOURCE}"]'
    subgroup_rule_path = f'project.subgroup_width_rules["{MLX_LAYER_NORM_SOURCE}"]'
    expected_workgroup_rule = {
        "components": [str(component) for component in workgroup_size],
        "sourcePattern": MLX_LAYER_NORM_SOURCE,
        "path": workgroup_rule_path,
    }
    expected_subgroup_rule = {
        "expression": str(LAYER_NORM_SUBGROUP_WIDTH),
        "sourcePattern": MLX_LAYER_NORM_SOURCE,
        "path": subgroup_rule_path,
    }
    expected_materialization = {
        "name": case["templateName"],
        "hostName": entry_point,
        "materializedName": entry_point,
    }
    expected_parameters = {"N_READS": "8", "T": "float"}
    expected_parameter_sources = {
        "N_READS": "source-default",
        "T": "source-instantiation",
    }
    expected_provenance = {
        "kind": "materialized-template-rule",
        "path": workgroup_rule_path,
    }
    expected_subgroup_provenance = {
        "kind": "materialized-template-rule",
        "path": subgroup_rule_path,
    }
    expected_enforcement = {
        "mechanism": "hlsl-wave-size-attribute",
        "minimumShaderModel": "6.6",
        "entryProfiles": [
            {"entryPoint": "CSMain", "profile": LAYER_NORM_DIRECTX_PROFILE}
        ],
    }
    _require(
        artifact.get("entryPoint")
        == {"source": entry_point, "target": "CSMain", "stage": "compute"},
        f"{case['name']} artifact entry-point identity changed",
    )
    _require(
        isinstance(execution, Mapping)
        and set(execution)
        == {
            "sourceEntryPoints",
            "entryPoints",
            "provenance",
            "identity",
            "subgroupWidthProvenance",
            "subgroupWidthEnforcement",
        }
        and execution.get("sourceEntryPoints") == [entry_point]
        and execution.get("provenance") == expected_provenance
        and execution.get("subgroupWidthProvenance") == expected_subgroup_provenance
        and execution.get("subgroupWidthEnforcement") == expected_enforcement
        and _is_sha256_identity(execution.get("identity")),
        f"{case['name']} exact execution metadata changed",
    )
    entries = execution.get("entryPoints")
    _require(
        isinstance(entries, list)
        and len(entries) == 1
        and isinstance(entries[0], Mapping),
        f"{case['name']} must report one materialized execution entry",
    )
    execution_entry = entries[0]
    _require(
        set(execution_entry)
        == {
            "sourceEntryPoint",
            "materializedEntryPoint",
            "targetEntryPoint",
            "workgroupSize",
            "rule",
            "parameters",
            "parameterSources",
            "materialization",
            "identity",
            "subgroupWidth",
            "subgroupWidthRule",
        }
        and execution_entry.get("sourceEntryPoint") == entry_point
        and execution_entry.get("materializedEntryPoint") == entry_point
        and execution_entry.get("targetEntryPoint") == "CSMain"
        and execution_entry.get("workgroupSize") == workgroup_size
        and execution_entry.get("rule") == expected_workgroup_rule
        and execution_entry.get("parameters") == expected_parameters
        and execution_entry.get("parameterSources") == expected_parameter_sources
        and execution_entry.get("materialization") == expected_materialization
        and execution_entry.get("subgroupWidth") == LAYER_NORM_SUBGROUP_WIDTH
        and execution_entry.get("subgroupWidthRule") == expected_subgroup_rule
        and _is_sha256_identity(execution_entry.get("identity")),
        f"{case['name']} per-entry execution contract changed",
    )
    numthreads = [
        [int(component) for component in match]
        for match in re.findall(
            r"\[\s*numthreads\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]",
            generated,
        )
    ]
    exact_contract = (
        f"[numthreads({workgroup_size[0]}, {workgroup_size[1]}, "
        f"{workgroup_size[2]})]\n"
        f"[WaveSize({LAYER_NORM_SUBGROUP_WIDTH})]\n"
        "void CSMain("
    )
    _require(
        numthreads == [workgroup_size]
        and generated.count(f"[WaveSize({LAYER_NORM_SUBGROUP_WIDTH})]") == 1
        and len(re.findall(r"\[\s*WaveSize\s*\(", generated)) == 1
        and generated.count(exact_contract) == 1
        and len(re.findall(r"\bvoid\s+CSMain\s*\(", generated)) == 1,
        f"{case['name']} standalone HLSL entry/workgroup contract changed",
    )
    reflection = reflect_target_host_interface(artifact_path, target="directx")
    _require(
        reflection.get("status") == "ready"
        and reflection.get("entryPointCount") == 1
        and reflection.get("entryPoints")
        == [
            {
                "name": "CSMain",
                "stage": "compute",
                "executionConfig": {"numthreads": workgroup_size},
            }
        ]
        and reflection.get("diagnostics") == [],
        f"{case['name']} reflected DirectX entry/workgroup metadata changed",
    )
    return {
        "sourceEntryPoint": entry_point,
        "targetEntryPoint": "CSMain",
        "workgroupSize": workgroup_size,
        "workgroupSizeRule": expected_workgroup_rule,
        "provenance": expected_provenance,
        "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
        "subgroupWidthRule": expected_subgroup_rule,
        "subgroupWidthProvenance": expected_subgroup_provenance,
        "subgroupWidthEnforcement": expected_enforcement,
        "identity": dict(execution_entry["identity"]),
        "aggregateIdentity": dict(execution["identity"]),
    }


def _validate_artifact(
    artifact: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    mlx_root: Path,
    project_root: Path,
    projection: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    name = str(case["name"])
    _require(
        artifact.get("source") == MLX_LAYER_NORM_SOURCE
        and artifact.get("sourceBackend") == "metal"
        and artifact.get("target") == "directx"
        and artifact.get("status") == "translated"
        and artifact.get("variant") == name,
        f"{name} artifact identity or status is incorrect",
    )
    _require(
        artifact.get("sourceHash") == projection.get("sourceHash")
        and artifact.get("provenance")
        == {"pipeline": "entry-scoped-translate", "intermediate": "crossgl"},
        f"{name} artifact provenance changed",
    )
    artifact_path = _artifact_path(artifact, project_root)
    generated = artifact_path.read_text(encoding="utf-8")
    _require(
        artifact.get("generatedHash")
        == {"algorithm": "sha256", "value": _sha256(artifact_path)}
        and artifact.get("generatedSizeBytes") == artifact_path.stat().st_size
        and artifact_path.suffix == ".hlsl",
        f"{name} generated artifact identity is stale",
    )
    materialization = _selected_materialization(artifact, case)
    specialization = _validate_specialization(artifact, generated, case)
    execution = _validate_execution(
        artifact,
        artifact_path,
        generated,
        case,
    )
    return (
        {
            "name": name,
            "axisSize": case["axisSize"],
            "branch": "single-row",
            "artifact": _relpath(artifact_path, mlx_root),
            "sourceProjection": dict(projection),
            "materialization": dict(materialization),
            "specializationConstant": specialization,
            "execution": execution,
        },
        artifact_path,
        generated,
    )


def _find_dxc() -> str | None:
    return shutil.which("dxc")


def _compile_directx_artifact(
    artifact_path: Path,
    generated: str,
    case: Mapping[str, Any],
    *,
    dxc: str | None,
    mlx_root: Path,
    work_dir: Path,
    log_dir: Path,
    required: bool,
) -> dict[str, Any]:
    if dxc is None:
        _require(not required, "DirectX proof requires dxc, but it is unavailable")
        return {
            "required": False,
            "available": False,
            "status": "unavailable",
            "compiler": "dxc",
            "compiledArtifactCount": 0,
        }

    output_dir = work_dir / "native" / "directx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case['name']}.dxil"
    output_path.unlink(missing_ok=True)
    _require(
        generated.count(f"[WaveSize({LAYER_NORM_SUBGROUP_WIDTH})]") == 1
        and len(re.findall(r"\[\s*WaveSize\s*\(", generated)) == 1,
        f"{case['name']} must contain exactly WaveSize("
        f"{LAYER_NORM_SUBGROUP_WIDTH}) before DXC validation",
    )
    profile = dxc_profile_for_source(LAYER_NORM_DIRECTX_PROFILE, generated)
    _require(
        profile == LAYER_NORM_DIRECTX_PROFILE,
        f"{case['name']} must compile with {LAYER_NORM_DIRECTX_PROFILE}",
    )
    compiler_arguments = dxc_compiler_arguments_for_source(generated)
    result = _run_command(
        f"compile-layernorm-directx-{case['name']}",
        [
            dxc,
            "-WX",
            "-T",
            profile,
            *compiler_arguments,
            "-E",
            "CSMain",
            str(artifact_path),
            "-Fo",
            str(output_path),
        ],
        log_dir=log_dir,
    )
    _require(
        result["returncode"] == 0,
        f"DXC failed for LayerNorm case {case['name']}",
    )
    _require(
        output_path.is_file() and output_path.stat().st_size > 0,
        f"DXC did not emit DXIL for LayerNorm case {case['name']}",
    )
    return {
        "required": required,
        "available": True,
        "status": "compiled",
        "compiler": "dxc",
        "profile": profile,
        "compilerArguments": list(compiler_arguments),
        "entryPoint": "CSMain",
        "workgroupSize": list(case["workgroupSize"]),
        "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
        "subgroupWidthEnforcement": f"WaveSize({LAYER_NORM_SUBGROUP_WIDTH})",
        "artifact": _relpath(artifact_path, mlx_root),
        "compiledArtifact": _relpath(output_path, mlx_root),
        "compiledArtifactCount": 1,
        "stdout": _relpath(result["stdoutPath"], mlx_root),
        "stderr": _relpath(result["stderrPath"], mlx_root),
    }


def _check_directx_case(
    mlx_root: Path,
    work_dir: Path,
    report_dir: Path,
    log_dir: Path,
    case: Mapping[str, Any],
    *,
    dxc: str | None,
    require_toolchain: bool,
) -> dict[str, Any]:
    project_root, projection = _prepare_case_project(mlx_root, work_dir, case)
    report_path = report_dir / f"{case['name']}.json"
    payload = _translate_case(
        _project_config(project_root, case),
        case,
        report_path=report_path,
    )
    artifact = _validate_project_report(payload, case)
    evidence, artifact_path, generated = _validate_artifact(
        artifact,
        case,
        mlx_root=mlx_root,
        project_root=project_root,
        projection=projection,
    )
    evidence["report"] = _relpath(report_path, mlx_root)
    evidence["nativeCompilation"] = _compile_directx_artifact(
        artifact_path,
        generated,
        case,
        dxc=dxc,
        mlx_root=mlx_root,
        work_dir=work_dir,
        log_dir=log_dir,
        required=require_toolchain,
    )
    evidence["runtimeParityClaimed"] = False
    evidence["numericalExecutionIncluded"] = False
    return evidence


def run_proof(args: argparse.Namespace) -> dict[str, Any]:
    mlx_root = Path(args.mlx_root).resolve()
    work_dir = _resolve_work_dir(mlx_root, args.work_dir)
    if work_dir.exists() and not args.no_clean:
        shutil.rmtree(work_dir)
    report_dir = work_dir / "reports"
    log_dir = work_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    source_identity, dispatch_evidence, subgroup_evidence = _verify_mlx_checkout(
        mlx_root,
        log_dir=log_dir,
    )
    dxc = _find_dxc()
    cases = [
        _check_directx_case(
            mlx_root,
            work_dir,
            report_dir,
            log_dir,
            case,
            dxc=dxc,
            require_toolchain=args.require_directx_toolchain,
        )
        for case in LAYER_NORM_DIRECTX_CASES
    ]
    native_compilation_claimed = all(
        case["nativeCompilation"]["status"] == "compiled" for case in cases
    )
    return {
        "schema_version": 1,
        "kind": "crosstl-mlx-layernorm-directx-project-proof",
        "repository": {
            "name": "ml-explore/mlx",
            "url": MLX_REPOSITORY,
            "commit": MLX_COMMIT,
        },
        "scope": {
            "source": MLX_LAYER_NORM_SOURCE,
            "sourceSha256": MLX_LAYER_NORM_SHA256,
            "target": "directx",
            "projectTranslationApi": "crosstl.project.translate_project",
            "selectedEntryPoints": [
                str(case["entryPoint"]) for case in LAYER_NORM_DIRECTX_CASES
            ],
            "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
            "subgroupWidthEnforcement": f"WaveSize({LAYER_NORM_SUBGROUP_WIDTH})",
            "minimumShaderModel": "6.6",
            "dxcProfile": LAYER_NORM_DIRECTX_PROFILE,
            "translationClaimed": True,
            "nativeCompilationClaimed": native_compilation_claimed,
            "runtimeParityClaimed": False,
            "numericalExecutionIncluded": False,
            "fullMlxTestSuiteIncluded": False,
            "loopedEntriesIncluded": False,
        },
        "checks": [source_identity, dispatch_evidence, subgroup_evidence],
        "cases": cases,
        "status": "passed",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove selected pinned MLX layer_norm.metal entries as standalone "
            "DirectX HLSL artifacts."
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
        help="Require DXC compilation instead of recording an unavailable toolchain.",
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
    except MlxLayerNormDirectXProofError as exc:
        summary = {
            "schema_version": 1,
            "kind": "crosstl-mlx-layernorm-directx-project-proof",
            "repository": {
                "name": "ml-explore/mlx",
                "url": MLX_REPOSITORY,
                "commit": MLX_COMMIT,
            },
            "scope": {
                "source": MLX_LAYER_NORM_SOURCE,
                "sourceSha256": MLX_LAYER_NORM_SHA256,
                "target": "directx",
                "subgroupWidth": LAYER_NORM_SUBGROUP_WIDTH,
                "subgroupWidthEnforcement": f"WaveSize({LAYER_NORM_SUBGROUP_WIDTH})",
                "minimumShaderModel": "6.6",
                "dxcProfile": LAYER_NORM_DIRECTX_PROFILE,
                "runtimeParityClaimed": False,
                "numericalExecutionIncluded": False,
                "fullMlxTestSuiteIncluded": False,
            },
            "status": "failed",
            "error": str(exc),
        }
        _write_json(summary_path, summary)
        print(f"MLX LayerNorm DirectX proof failed: {exc}", file=sys.stderr)
        print(f"Summary: {summary_path}", file=sys.stderr)
        return 1
    _write_json(summary_path, summary)
    print(f"MLX LayerNorm DirectX proof passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
