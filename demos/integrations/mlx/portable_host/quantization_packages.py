"""Cache selected entries from the unchanged pinned quantization source."""

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
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from demos.integrations.mlx.portable_host.quantization_layout import (
    MAX_GROUPS,
    signature,
    workgroup_size,
)

SOURCE = "mlx/backend/metal/kernels/quantized.metal"
INDEX_EXPRESSIONS = (
    "in_index + i",
    "gindex",
    "out_index / writes_per_reduce",
    "out_index",
    "out_index + i / pack_factor",
    "out_index + 1",
    "out_index + 2",
    "out_index + 3",
    "out_index + 4",
    "offset",
    "oindex",
)


class QuantizationPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError("Unsupported affine quantization target")
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
            raise ValueError("Affine translation requires unchanged pinned kernels")

    def get(self, entry):
        signature(entry)
        with self._lock:
            self._require_source()
            identity = {
                "schemaVersion": 1,
                "revision": COMMIT,
                "source": SOURCE,
                "sourceHash": (
                    hashlib.sha256((self.root / SOURCE).read_bytes()).hexdigest()
                ),
                "target": self.target,
                "entry": entry,
                "maximumGroups": MAX_GROUPS,
                "workgroupSize": workgroup_size(entry),
                "implementationHash": translation_implementation_hash(),
                "recipeHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "layoutHash": (
                    hashlib.sha256(
                        Path(__file__).with_name("quantization_layout.py").read_bytes()
                    ).hexdigest()
                ),
            }
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".affine-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(entry, staging)
                    self._require_source()
                    if (
                        identity["implementationHash"]
                        != translation_implementation_hash()
                        or identity["recipeHash"]
                        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        or identity["layoutHash"]
                        != hashlib.sha256(
                            Path(__file__)
                            .with_name("quantization_layout.py")
                            .read_bytes()
                        ).hexdigest()
                    ):
                        raise ValueError(
                            "Affine translation implementation changed while packaging"
                        )
                    loader = build_runtime_loader_manifest(
                        staging / "package/runtime-package.json"
                    )
                    if not loader["success"] or len(loader["loadUnits"]) != 1:
                        raise ValueError(
                            f"Invalid affine package: {loader['diagnostics']}"
                        )
                    descriptor = build_native_loader_abi_descriptor(loader)
                    (staging / "index.json").write_text(
                        json.dumps(
                            {"identity": identity, "descriptor": descriptor}, indent=2
                        )
                    )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            return self._load(destination, identity)

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text())
        if record.get("identity") != identity:
            raise ValueError("Affine cache identity does not match the request")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(f"Invalid affine runtime package: {loader['diagnostics']}")
        if loader["loadUnits"][0]["entryPoint"]["source"] != identity["entry"]:
            raise ValueError("Affine package selected entry does not match the request")
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(
                self.target, identity["entry"]
            )
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": identity["sourceHash"],
            }
        ):
            raise ValueError("Affine cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, entry, output):
        _, _, group, _ = signature(entry)
        options = {
            "max_template_specializations": 128,
            "max_template_materialization_work": 4096,
        }
        if self.target != "metal":
            settings = {
                "software_subgroup_width": 32,
                "software_subgroup_applicability": "when-used",
            }
            if self.target == "directx":
                settings["relative_wave_shuffle_out_of_range"] = "self"
            options["target_options"] = {self.target: settings}
        with tempfile.TemporaryDirectory(
            prefix=".affine-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = ProjectConfig(
                root=self.root,
                source_roots=("mlx/backend/metal/kernels",),
                include_patterns=(SOURCE,),
                include_dirs=(".",),
                targets=(self.target,),
                output_dir=f"{work.name}/out",
                entry_points={SOURCE: (entry,)},
                workgroup_size=(workgroup_size(entry), 1, 1),
                source_options={"metal": options},
                # Include pre-predicate byte offsets for all lanes, not just writers.
                index_range_assertions=(
                    tuple(
                        {
                            "source": SOURCE,
                            "expression": expression,
                            "minimum": 0,
                            "maximum": MAX_GROUPS * group * 5 + 4,
                        }
                        for expression in INDEX_EXPRESSIONS
                    )
                    if self.target == "opengl"
                    else ()
                ),
            )
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(f"Affine translation failed: {payload['diagnostics']}")
            if payload["artifacts"][0]["entryPoint"]["source"] != entry:
                raise ValueError("Affine translation selected a different entry")
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"Affine manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"Affine packaging failed: {package['diagnostics']}")
