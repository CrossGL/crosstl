"""Build pinned small-row kernel packages for an exact native launch."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    select_native_loader_dispatch_regions,
    translate_project,
)
from crosstl.translator.dispatch_region_identity import translation_implementation_hash
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from demos.integrations.mlx.portable_host.reduction_packages import ALL_ENTRIES, SOURCE

SMALL_ROW_ENTRIES = {
    f"row_reduce_small_{dimension}_reduce_" + entry.removeprefix("all_reduce_"): dtype
    for dimension in (1, 2, 5)
    for entry, dtype in ALL_ENTRIES.items()
}


class SmallRowPackageCache:
    """Cache translated sources, not native modules or numerical results."""

    def __init__(self, root, directory, target):
        if target not in {"metal", "opengl", "directx"}:
            raise ValueError("Unsupported small-row target")
        self.root = Path(root).resolve()
        self.directory = Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()

    def _require_source(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(self.root),
                "status",
                "--porcelain",
                "--",
                "mlx/backend/metal/kernels",
            ],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError("Small-row translation requires unchanged pinned kernels")

    def get(self, entry, *, thread_grid_size, workgroup_size):
        if entry not in SMALL_ROW_ENTRIES:
            raise ValueError("Unknown small-row entry")
        regions = plan_dispatch_regions(thread_grid_size, workgroup_size)
        nominal = regions[0].source_workgroup_size
        if nominal[1:] != (1, 1) or nominal[0] > 1024:
            raise ValueError("Small-row workgroups require an X width of at most 1024")
        with self._lock:
            self._require_source()
            implementation = translation_implementation_hash()
            items = [None] if self.target == "metal" else regions
            packages = []
            for region in items:
                identity = {
                    "schemaVersion": 1,
                    "revision": COMMIT,
                    "target": self.target,
                    "entry": entry,
                    "implementationHash": implementation,
                    "recipeHash": (
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    ),
                    "workgroupSize": list(
                        nominal if region is None else region.workgroup_size
                    ),
                    "region": region.to_json() if region is not None else None,
                }
                key = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                destination = self.directory / key
                if not destination.exists():
                    self.directory.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        prefix=".small-row-", dir=self.directory
                    ) as temporary:
                        staging = Path(temporary) / "entry"
                        staging.mkdir()
                        self._build(entry, staging, identity, region)
                        # Do not publish a package built while source or translator changed.
                        self._require_source()
                        if translation_implementation_hash() != implementation:
                            raise ValueError(
                                "Translator changed while building a small-row package"
                            )
                        try:
                            staging.rename(destination)
                        except OSError:
                            if not destination.is_dir():
                                raise
                packages.append(self._load(destination, identity))
            if self.target != "metal":
                return select_native_loader_dispatch_regions(
                    packages,
                    thread_grid_size=thread_grid_size,
                    source_workgroup_size=nominal,
                )
            return tuple(packages)

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError(
                "Small-row cache identity does not match the requested launch"
            )
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid small-row runtime package: {loader['diagnostics']}"
            )
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
        ):
            raise ValueError("Small-row cache descriptor does not match its package")
        if identity["region"] is not None:
            provenance = descriptor["provenance"]
            program = provenance.get("dispatchRegionProgram", {})
            if (
                provenance.get("dispatchRegion") != identity["region"]
                or program.get("sourceEntryPoint") != identity["entry"]
                or program.get("implementationHash") != identity["implementationHash"]
            ):
                raise ValueError(
                    "Small-row source program does not match its cache identity"
                )
        return descriptor, directory / "package"

    def _build(self, entry, output, identity, region):
        target_options = {}
        if self.target != "metal":
            target_options["software_subgroup_width"] = 32
            target_options["dispatch_region"] = region.to_json()
        if self.target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        with tempfile.TemporaryDirectory(
            prefix=".small-row-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = ProjectConfig(
                root=self.root,
                source_roots=("mlx/backend/metal/kernels",),
                include_patterns=(SOURCE,),
                include_dirs=(".",),
                targets=(self.target,),
                output_dir=f"{work.name}/out",
                entry_points={SOURCE: entry},
                entry_workgroup_size_rules={
                    SOURCE: {"row_reduce_small_*": identity["workgroupSize"]}
                },
                subgroup_width_rules={SOURCE: 32} if self.target == "directx" else {},
                source_options={
                    "metal": {
                        "max_template_specializations": 128,
                        "max_template_materialization_work": 8192,
                        "target_options": {self.target: target_options},
                    }
                },
            )
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(
                    f"Small-row translation failed: {payload['diagnostics']}"
                )
            if payload["artifacts"][0]["entryPoint"]["source"] != entry:
                raise ValueError(
                    "Small-row translation selected a different source entry"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Small-row manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Small-row packaging failed: {package['diagnostics']}"
                )
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"Small-row loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
