"""Cache translated affine matrix-vector entries from the unchanged pinned source."""

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
from demos.integrations.mlx.portable_host.matmul_layout import (
    MAX_ELEMENTS,
    SIMD_SIZE,
    signature,
    workgroup_size,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

SOURCE = "mlx/backend/metal/kernels/quantized.metal"
# Expressions OpenGL may evaluate in 32 bits: adjust_matrix_offsets' batch
# offsets. matmul_layout.plan proves each names an element of a buffer of at
# most MAX_ELEMENTS elements; unbatched kernels never compute them.
BATCH_INDEX_EXPRESSIONS = (
    "x_idx * x_strides[0]",
    "w_idx * w_strides[0]",
    "w_idx * s_strides[0]",
    "w_idx * b_strides[0]",
    "idx.x",
    "idx.y",
    "idx.z",
)


def _recipe_hashes():
    here = Path(__file__)
    return {
        "recipeHash": hashlib.sha256(here.read_bytes()).hexdigest(),
        "layoutHash": (
            hashlib.sha256(here.with_name("matmul_layout.py").read_bytes()).hexdigest()
        ),
    }


def project_config(root, output_dir, target, entry):
    """The translation of one entry, with the current-tree qmv gates' options."""
    options = {
        "directx": {
            "software_subgroup_width": SIMD_SIZE,
            "relative_wave_shuffle_out_of_range": "self",
        },
        "opengl": {"software_subgroup_width": SIMD_SIZE},
    }
    batched = signature(entry)["batched"]
    return ProjectConfig(
        root=root,
        source_roots=("mlx/backend/metal/kernels",),
        include_patterns=(SOURCE,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=output_dir,
        entry_points={SOURCE: (entry,)},
        entry_workgroup_size_rules={SOURCE: {entry: workgroup_size(entry)}},
        index_range_assertions=[
            dict(source=SOURCE, expression=expression, minimum=0, maximum=MAX_ELEMENTS)
            for expression in BATCH_INDEX_EXPRESSIONS
            if target == "opengl" and batched
        ],
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "max_template_specializations": 512,
                "max_template_materialization_work": 65536,
                "target_options": options,
            }
        },
    )


class QuantizedMatmulPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError("Unsupported quantized matrix-vector target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()

    def _require_source(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--"]
            + ["mlx/backend/metal/kernels"],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError(
                "Quantized matrix-vector translation requires unchanged pinned kernels"
            )

    def _identity(self, entry):
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "source": SOURCE,
            "sourceHash": hashlib.sha256((self.root / SOURCE).read_bytes()).hexdigest(),
            "target": self.target,
            "entry": entry,
            "maximumElements": MAX_ELEMENTS,
            "workgroupSize": list(workgroup_size(entry)),
            "implementationHash": translation_implementation_hash(),
            **_recipe_hashes(),
        }

    def get(self, entry):
        signature(entry)
        with self._lock:
            self._require_source()
            identity = self._identity(entry)
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".quantized-matmul-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(entry, staging)
                    self._require_source()
                    if self._identity(entry) != identity:
                        raise ValueError(
                            "Quantized matrix-vector implementation changed while packaging"
                        )
                    loader = build_runtime_loader_manifest(
                        staging / "package/runtime-package.json"
                    )
                    if not loader["success"] or len(loader["loadUnits"]) != 1:
                        raise ValueError(
                            f"Invalid quantized matrix-vector package: {loader['diagnostics']}"
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
            raise ValueError("Quantized matrix-vector cache identity differs")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid quantized matrix-vector runtime package: {loader['diagnostics']}"
            )
        if loader["loadUnits"][0]["entryPoint"]["source"] != identity["entry"]:
            raise ValueError("Quantized matrix-vector package selects another entry")
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
            raise ValueError(
                "Quantized matrix-vector cache descriptor does not match its package"
            )
        return descriptor, directory / "package"

    def _build(self, entry, output):
        with tempfile.TemporaryDirectory(
            prefix=".quantized-matmul-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = project_config(self.root, f"{work.name}/out", self.target, entry)
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(
                    f"Quantized matrix-vector translation failed: {payload['diagnostics']}"
                )
            (artifact,) = payload["artifacts"]
            if artifact["entryPoint"]["source"] != entry:
                raise ValueError(
                    "Quantized matrix-vector translation selected another entry"
                )
            (execution,) = artifact["execution"]["entryPoints"]
            if execution["workgroupSize"] != list(workgroup_size(entry)):
                raise ValueError(
                    "Quantized matrix-vector translation changed the workgroup size"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Quantized matrix-vector manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Quantized matrix-vector packaging failed: {package['diagnostics']}"
                )
