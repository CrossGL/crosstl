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
ENTRIES = (
    "arangefloat32",
    "arangeint32",
    "arangeuint32",
    "arangeint64",
    "arangeuint64",
)


def build_packages(root, output, target):
    root = Path(root).resolve()
    if target not in {"opengl", "directx"}:
        raise ValueError(f"Unsupported target: {target}")
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
            f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{work.name}/out"
[project.entry_points]
"{SOURCE}" = {json.dumps(ENTRIES)}
[project.entry_workgroup_size_rules."{SOURCE}"]
"arange*" = [1, 1, 1]
""",
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
            if len(entries_by_path) != len(ENTRIES) or set(
                entries_by_path.values()
            ) != set(ENTRIES):
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
            if set(descriptors) != set(ENTRIES):
                raise RuntimeError("Missing required loader entries")
            index = {"target": target, "descriptors": descriptors}
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
    parser.add_argument("--target", choices=["opengl", "directx"], required=True)
    args = parser.parse_args()
    build_packages(args.mlx_root, args.output_dir, args.target)
