"""Package the unchanged pinned random kernels for native host dispatch."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from demos.integrations.mlx.portable_host.random_layout import (
    MAX_KEY_ELEMENTS,
    MAX_NATIVE_BYTES,
)

SOURCE = "mlx/backend/metal/kernels/random.metal"
SOURCE_SHA256 = "f1a19b3f11b7b10203824890f13debc6d627959b4e7f17c219e2e9da553c1bd7"
ENTRIES = ("rbitsc", "rbits")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_source(root):
    require_revision(root)
    if hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned random source hash differs")
    changed = subprocess.check_output(
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
    if changed.strip():
        raise ValueError("Pinned kernel tree has local changes")


def build_packages(root, target, output):
    verify_source(root)
    with tempfile.TemporaryDirectory(prefix=".random-audit-", dir=root) as directory:
        work = Path(directory)
        config = ProjectConfig(
            root=root,
            source_roots=("mlx/backend/metal/kernels",),
            include_patterns=(SOURCE,),
            include_dirs=(".",),
            targets=(target,),
            output_dir=f"{work.name}/out",
            entry_points={SOURCE: ENTRIES},
            workgroup_size=(1, 1, 1),
            index_range_assertions=(
                [
                    {
                        "source": SOURCE,
                        "function": function,
                        "expression": expression,
                        "minimum": 0,
                        "maximum": (
                            MAX_NATIVE_BYTES - 1
                            if expression == "idx + i"
                            else MAX_KEY_ELEMENTS - 1
                        ),
                    }
                    for function, expression in (
                        ("rbitsc", "idx + i"),
                        ("rbits", "idx + i"),
                        ("rbits", "k1_elem"),
                        ("rbits", "k2_elem"),
                    )
                ]
                if target == "opengl"
                else ()
            ),
        )
        report = translate_project(config, format_output=False)
        report.write_json(work / "report.json")
        shutil.copytree(work, output / "translation")
        payload = report.to_json()
        if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 2:
            raise ValueError("Random translation did not produce both entries")
        entries = {
            item["path"]: item["entryPoint"]["source"] for item in payload["artifacts"]
        }
        manifest = build_runtime_artifact_manifest(work / "report.json")
        if not manifest["success"]:
            raise ValueError("Random runtime manifest is incomplete")
        write_json(work / "artifacts.json", manifest)
        package = output / "package"
        result = build_runtime_package(work / "artifacts.json", package)
        if not result["success"]:
            raise ValueError("Random runtime package is incomplete")
        loader = build_runtime_loader_manifest(package / "runtime-package.json")
        if not loader["success"]:
            raise ValueError("Random loader manifest is incomplete")
        descriptors = {}
        for unit in loader["loadUnits"]:
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=unit["id"]
            )
            descriptors[entries[descriptor["source"]["artifactPath"]]] = descriptor
        if set(descriptors) != set(ENTRIES):
            raise ValueError("Random entry coverage differs")
        write_json(
            output / "index.json",
            {
                "commit": COMMIT,
                "target": target,
                "maximumNativeByteCount": MAX_NATIVE_BYTES,
                "descriptors": descriptors,
            },
        )
        return descriptors


def load_index(directory, target):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if (
        not isinstance(index, dict)
        or index.get("commit") != COMMIT
        or index.get("target") != target
        or index.get("maximumNativeByteCount") != MAX_NATIVE_BYTES
        or not isinstance(index.get("descriptors"), dict)
        or set(index.get("descriptors", {})) != set(ENTRIES)
    ):
        raise ValueError("Random packages must match the pin, target and entry set")
    loader = build_runtime_loader_manifest(directory / "package/runtime-package.json")
    if not loader["success"] or len(loader["loadUnits"]) != 2:
        raise ValueError("Random runtime package is incomplete")
    observed = {}
    for unit in loader["loadUnits"]:
        name = unit.get("entryPoint", {}).get("source")
        descriptor = build_native_loader_abi_descriptor(loader, load_unit_id=unit["id"])
        if name not in ENTRIES or index["descriptors"][name] != descriptor:
            raise ValueError("Random descriptor differs from its runtime package")
        if (
            name in observed
            or descriptor["target"] != target
            or descriptor["stage"] != "compute"
            or descriptor["source"]["path"] != SOURCE
            or descriptor["source"]["backend"] != "metal"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(target, name)
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
        ):
            raise ValueError("Random descriptor source or entry identity differs")
        observed[name] = descriptor
    if set(observed) != set(ENTRIES):
        raise ValueError("Random package entry coverage differs")
    return observed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "opengl", "directx"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)
    build_packages(args.mlx_root.resolve(), args.target, args.output_dir.resolve())
    load_index(args.output_dir, args.target)
