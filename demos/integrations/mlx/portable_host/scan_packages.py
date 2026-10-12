"""Cache unchanged pinned scan entries for an exact source workgroup width."""

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
    translate_project,
)
from crosstl.translator.dispatch_region_identity import translation_implementation_hash
from demos.integrations.mlx.portable_host import scan_layout
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

SOURCE = "mlx/backend/metal/kernels/scan.metal"


class ScanPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError("Unsupported scan target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
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
            raise ValueError("Scan translation requires unchanged pinned kernels")

    def get(self, entry, width):
        scan_layout.signature(entry)
        if type(width) is not int or not 32 <= width <= 1024 or width % 32:
            raise ValueError(
                "Scan workgroups require complete 32-lane source subgroups"
            )
        with self._lock:
            self._require_source()
            identity = {
                "schemaVersion": 1,
                "revision": COMMIT,
                "target": self.target,
                "entry": entry,
                "workgroupSize": [width, 1, 1],
                "implementationHash": translation_implementation_hash(),
                "recipeHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sourceHash": (
                    hashlib.sha256((self.root / SOURCE).read_bytes()).hexdigest()
                ),
            }
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".scan-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity)
                    self._require_source()
                    if (
                        translation_implementation_hash()
                        != identity["implementationHash"]
                        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        != identity["recipeHash"]
                    ):
                        raise ValueError(
                            "Scan translator or packaging recipe changed during translation"
                        )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            return self._load(destination, identity)

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError("Scan cache identity does not match the requested entry")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(f"Invalid scan runtime package: {loader['diagnostics']}")
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": identity["sourceHash"],
            }
            or loader["loadUnits"][0]["entryPoint"]["source"] != identity["entry"]
        ):
            raise ValueError("Scan cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, output, identity):
        target_options = {}
        if self.target != "metal":
            target_options["software_subgroup_width"] = 32
        if self.target == "directx":
            target_options.update(
                {
                    "relative_wave_shuffle_out_of_range": "self",
                    "binary32_additive_profile": "native-precise",
                }
            )
        with tempfile.TemporaryDirectory(
            prefix=".scan-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            report = translate_project(
                ProjectConfig(
                    root=self.root,
                    source_roots=("mlx/backend/metal/kernels",),
                    include_patterns=(SOURCE,),
                    include_dirs=(".",),
                    targets=(self.target,),
                    output_dir=f"{work.name}/out",
                    entry_points={SOURCE: (identity["entry"],)},
                    workgroup_size_rules={SOURCE: tuple(identity["workgroupSize"])},
                    source_options={
                        "metal": {
                            "preserve_resource_origins": True,
                            "max_template_specializations": 256,
                            "max_template_materialization_work": 8192,
                            "target_options": {self.target: target_options},
                        }
                    },
                ),
                format_output=False,
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(f"Scan translation failed: {payload['diagnostics']}")
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"Scan manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"Scan packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"Scan loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
