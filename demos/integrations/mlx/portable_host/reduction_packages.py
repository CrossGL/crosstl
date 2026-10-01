"""Package unchanged MLX reductions at their concrete upstream launch widths."""

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
from demos.integrations.mlx.portable_host.prepare import require_revision

SOURCE = "mlx/backend/metal/kernels/reduce.metal"
ENTRIES = {
    f"all_reduce_{operation}{dtype}": dtype
    for operation in ("sum", "prod", "min", "max")
    for dtype in ("float32", "int32", "uint32")
}
ENTRIES.update({"all_reduce_andbool_": "bool_", "all_reduce_orbool_": "bool_"})
WIDTHS = tuple(range(32, 1025, 32))


def load_index(directory, target):
    index = json.loads((Path(directory) / "index.json").read_text(encoding="utf-8"))
    widths, entries = index.get("widths"), index.get("entries")
    if (
        index.get("target") != target
        or not isinstance(widths, list)
        or not widths
        or any(type(width) is not int or width not in WIDTHS for width in widths)
        or len(set(widths)) != len(widths)
        or not isinstance(entries, list)
        or not entries
        or any(not isinstance(entry, str) or entry not in ENTRIES for entry in entries)
        or len(set(entries)) != len(entries)
    ):
        raise ValueError("Invalid reduction package target, widths or entries")
    expected = {f"w{width}/{entry}" for width in widths for entry in entries}
    if (
        not isinstance(index.get("descriptors"), dict)
        or set(index["descriptors"]) != expected
    ):
        raise ValueError("Reduction packages do not cover their declared variants")
    for key, descriptor in index["descriptors"].items():
        if not isinstance(descriptor, dict) or descriptor.get("target") != target:
            raise ValueError(f"Invalid reduction descriptor target: {key}")
    return index


def build_packages(root, output, target, *, widths=WIDTHS, entries=None, jobs=1):
    root, output = Path(root).resolve(), Path(output).resolve()
    if target not in {"metal", "opengl", "directx"}:
        raise ValueError(f"Unsupported reduction target: {target}")
    if type(jobs) is not int or jobs < 1:
        raise ValueError("Reduction translation jobs must be a positive integer")
    widths = tuple(widths)
    entries = tuple(ENTRIES if entries is None else entries)
    if (
        not widths
        or len(set(widths)) != len(widths)
        or any(type(width) is not int or width not in WIDTHS for width in widths)
    ):
        raise ValueError("Reduction widths must be unique multiples of 32 up to 1024")
    if (
        not entries
        or len(set(entries)) != len(entries)
        or set(entries) - ENTRIES.keys()
    ):
        raise ValueError("Unknown or duplicate reduction entries")
    require_revision(root)
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
    if dirty.strip():
        raise ValueError("Reduction translation requires unchanged pinned kernels")
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix=".host-reductions-", dir=root) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        variants = "".join(
            f"[project.variants.w{width}]\nworkgroup_size = [{width}, 1, 1]\n"
            for width in widths
        )
        options = ""
        if target != "metal":
            options = (
                f"[project.source_options.metal.target_options.{target}]\n"
                "software_subgroup_width = 32\n"
            )
        if target == "directx":
            options += 'relative_wave_shuffle_out_of_range = "self"\n'
        config.write_text(
            '[project]\nsource_roots = ["mlx/backend/metal/kernels"]\n'
            f'include = ["{SOURCE}"]\ninclude_dirs = ["."]\ntargets = ["{target}"]\n'
            f'output_dir = "{work.name}/out"\n[project.entry_points]\n'
            f'"{SOURCE}" = {json.dumps(entries)}\n'
            + variants
            + "[project.source_options.metal]\nmax_template_specializations = 128\n"
            "max_template_materialization_work = 8192\n" + options,
            encoding="utf-8",
        )
        shutil.copyfile(config, output / "crosstl.toml")
        try:
            report = translate_project(
                load_project_config(root, config),
                format_output=False,
                max_workers=jobs,
                checkpoint_path=output / "checkpoint.json",
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            if payload["summary"]["failedCount"]:
                raise RuntimeError(payload["diagnostics"])
            identities = {
                artifact[
                    "path"
                ]: f'{artifact["variant"]}/{artifact["entryPoint"]["source"]}'
                for artifact in payload["artifacts"]
            }
            expected = {f"w{width}/{entry}" for width in widths for entry in entries}
            if len(identities) != len(expected) or set(identities.values()) != expected:
                raise RuntimeError(
                    "Reduction translation did not preserve every entry and width"
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
                identity = identities.get(descriptor["source"]["artifactPath"])
                if identity is None or identity in descriptors:
                    raise RuntimeError("Ambiguous reduction package identity")
                descriptors[identity] = descriptor
            if set(descriptors) != expected:
                raise RuntimeError("Missing reduction loader variants")
            index = {
                "target": target,
                "widths": list(widths),
                "entries": list(entries),
                "descriptors": descriptors,
            }
            (output / "index.json").write_text(
                json.dumps(index, indent=2), encoding="utf-8"
            )
            return index
        finally:
            shutil.copytree(work, output / "translation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "opengl", "directx"), required=True
    )
    parser.add_argument("--width", type=int, action="append")
    parser.add_argument("--entry", action="append")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    build_packages(
        args.mlx_root,
        args.output_dir,
        args.target,
        widths=args.width or WIDTHS,
        entries=args.entry,
        jobs=args.jobs,
    )
