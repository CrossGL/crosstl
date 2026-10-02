"""Translate pinned entry sources into immutable native loader packages."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)

SOURCE = "mlx/backend/metal/kernels/arange.metal"
ARANGE_ENTRIES = (
    "arangefloat32",
    "arangeint32",
    "arangeuint32",
    "arangeint64",
    "arangeuint64",
)
UNARY_SOURCE = "mlx/backend/metal/kernels/unary.metal"
UNARY_OPERATIONS = (
    "Abs",
    "ArcCos",
    "ArcCosh",
    "ArcSin",
    "ArcSinh",
    "ArcTan",
    "ArcTanh",
    "Ceil",
    "Cos",
    "Cosh",
    "Exp",
    "Expm1",
    "Floor",
    "Log",
    "Log2",
    "Log10",
    "Log1p",
    "Negative",
    "Sigmoid",
    "Erf",
    "ErfInv",
    "Sign",
    "Sin",
    "Sinh",
    "Square",
    "Sqrt",
    "Rsqrt",
    "Tan",
    "Tanh",
    "Round",
)
UNARY_ENTRIES = tuple(f"v_{op}float32float32" for op in UNARY_OPERATIONS)
COPY_SOURCE = "mlx/backend/metal/kernels/copy.metal"
COPY_ENTRY = "ggn2_dynamic_copyuint32uint32"
BOOLEAN_COPY_ENTRY = "ggn2_dynamic_copybool_bool_"
LOGICAL_NOT_ENTRY = "v_LogicalNotbool_bool_"
CAST_ENTRIES = {
    f"v_copy{source}{destination}": (source, destination)
    for source in ("float32", "int32", "uint32")
    for destination in ("float32", "int32", "uint32")
    if source != destination
}
BOOLEAN_CAST_ENTRIES = {
    f"v_copy{source}{destination}": (source, destination)
    for dtype in ("float32", "int32", "uint32")
    for source, destination in ((dtype, "bool_"), ("bool_", dtype))
}
COMPARISON_OPERATIONS = (
    "Equal",
    "NotEqual",
    "Less",
    "LessEqual",
    "Greater",
    "GreaterEqual",
)
LOGICAL_OPERATIONS = ("LogicalAnd", "LogicalOr")
COMPARISON_ENTRIES = {
    f"vv_{operation}{dtype}": dtype
    for operation in COMPARISON_OPERATIONS
    for dtype in ("float32", "int32", "uint32", "bool_")
}
COMPARISON_ENTRIES.update(
    {f"vv_{operation}bool_": "bool_" for operation in LOGICAL_OPERATIONS}
)
COMPARISON_ENTRIES["vv_NaNEqualfloat32"] = "float32"
BINARY_SOURCE = "mlx/backend/metal/kernels/binary.metal"
BINARY_OPERATIONS = ("Add", "Subtract", "Multiply", "Minimum", "Maximum", "Divide")
BINARY_ENTRIES = {
    f"vv_{operation}{dtype}": dtype
    for operation in BINARY_OPERATIONS
    for dtype in ("float32", "int32", "uint32")
    if operation != "Divide" or dtype == "float32"
}
BITWISE_OPERATIONS = (
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "LeftShift",
    "RightShift",
)
BITWISE_ENTRIES = {
    f"vv_{operation}{dtype}": dtype
    for operation in BITWISE_OPERATIONS
    for dtype in ("int32", "uint32", "bool_")
    if dtype != "bool_" or operation not in {"LeftShift", "RightShift"}
}
BITWISE_INVERT_ENTRIES = {
    f"v_BitwiseInvert{dtype}{dtype}": dtype for dtype in ("int32", "uint32")
}
BITWISE_PACKAGE_ENTRIES = {**BITWISE_ENTRIES, **BITWISE_INVERT_ENTRIES}
ENTRIES = (
    ARANGE_ENTRIES
    + UNARY_ENTRIES
    + (COPY_ENTRY,)
    + tuple(BINARY_ENTRIES)
    + tuple(CAST_ENTRIES)
    + (BOOLEAN_COPY_ENTRY, LOGICAL_NOT_ENTRY)
    + tuple(BOOLEAN_CAST_ENTRIES)
    + tuple(COMPARISON_ENTRIES)
)


def build_packages(root, output, target, *, family="base"):
    root = Path(root).resolve()
    if target not in {"opengl", "directx", "metal"}:
        raise ValueError(f"Unsupported target: {target}")
    if family not in {"base", "bitwise"}:
        raise ValueError(f"Unsupported package family: {family}")
    entries = ENTRIES if family == "base" else tuple(BITWISE_PACKAGE_ENTRIES)
    sources = (
        {
            SOURCE: ARANGE_ENTRIES,
            UNARY_SOURCE: (*UNARY_ENTRIES, LOGICAL_NOT_ENTRY),
            COPY_SOURCE: (
                COPY_ENTRY,
                BOOLEAN_COPY_ENTRY,
                *CAST_ENTRIES,
                *BOOLEAN_CAST_ENTRIES,
            ),
            BINARY_SOURCE: (*BINARY_ENTRIES, *COMPARISON_ENTRIES),
        }
        if family == "base"
        else {
            BINARY_SOURCE: tuple(BITWISE_ENTRIES),
            UNARY_SOURCE: tuple(BITWISE_INVERT_ENTRIES),
        }
    )
    patterns = {
        SOURCE: "arange*",
        UNARY_SOURCE: "v_*",
        COPY_SOURCE: "*copy*",
        BINARY_SOURCE: "vv_*",
    }
    from demos.integrations.mlx.portable_host.prepare import COMMIT

    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    )
    if head != COMMIT or dirty.strip():
        raise ValueError("Translation requires the unchanged pinned MLX kernel tree")
    output = Path(output).resolve()
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=".portable-host-", dir=root) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(
            '[project]\nsource_roots = ["mlx/backend/metal/kernels"]\n'
            f'include = {json.dumps(list(sources))}\ninclude_dirs = ["."]\n'
            f'targets = ["{target}"]\noutput_dir = "{work.name}/out"\n'
            "[project.entry_points]\n"
            + "".join(
                f'"{source}" = {json.dumps(selected)}\n'
                for source, selected in sources.items()
            )
            + "".join(
                f'[project.entry_workgroup_size_rules."{source}"]\n"{patterns[source]}" = [1, 1, 1]\n'
                for source in sources
            )
            + '[project.source_options.metal]\nbinary32_fma_profile = "rne-flush"\n',
            encoding="utf-8",
        )
        try:
            report = translate_project(
                load_project_config(root, config), format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            if payload["summary"]["failedCount"]:
                raise RuntimeError(payload["diagnostics"])
            entries_by_path = {
                artifact["path"]: artifact["entryPoint"]["source"]
                for artifact in payload["artifacts"]
            }
            if len(entries_by_path) != len(entries) or set(
                entries_by_path.values()
            ) != set(entries):
                raise RuntimeError(
                    "Translation did not produce the exact required entries"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise RuntimeError(manifest)
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = output / "package"
            result = build_runtime_package(work / "artifacts.json", package)
            if not result["success"]:
                raise RuntimeError(result)
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            if not loader["success"]:
                raise RuntimeError(loader)
            descriptors = {}
            for unit in loader["loadUnits"]:
                descriptor = build_native_loader_abi_descriptor(
                    loader, load_unit_id=unit["id"]
                )
                entry = entries_by_path.get(descriptor["source"]["artifactPath"])
                if entry is None or entry in descriptors:
                    raise RuntimeError(f"Ambiguous loader entry: {unit['id']}")
                descriptors[entry] = descriptor
            if set(descriptors) != set(entries):
                raise RuntimeError("Missing required loader entries")
            index = {"target": target, "descriptors": descriptors}
            if family != "base":
                index["family"] = family
            (output / "index.json").write_text(
                json.dumps(index, indent=2), encoding="utf-8"
            )
        finally:
            shutil.copytree(work, output / "translation")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target", choices=["opengl", "directx", "metal"], required=True
    )
    parser.add_argument("--family", choices=("base", "bitwise"), default="base")
    args = parser.parse_args()
    build_packages(args.mlx_root, args.output_dir, args.target, family=args.family)
