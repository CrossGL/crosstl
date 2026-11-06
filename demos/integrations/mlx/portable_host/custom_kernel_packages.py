"""Translate the kernels ``mx.fast.metal_kernel`` builds, one package per source.

A custom kernel has no pinned source file: the user's code builds it, and
metal_kernel.cpp's ``write_signature`` wraps it in the kernel's signature. MLX
prepends metal::utils(), the pinned ``utils.h``, and compiles the result with
the kernel's math mode; the cache translates that program once per source,
math mode and threadgroup shape. Its identity covers the translator, this
recipe, the layout parser and the pinned headers, so any change translates
again. The source is kept in the package directory, so an audit can derive the
package again from the hash a trace records.

Metal runs the edge threadgroups of a grid that is not a whole number of
them, and the kernels index by the thread's grid position without checking
the grid. Direct3D and OpenGL dispatch whole workgroups only, so each launch
runs as the dispatch regions its exact grid splits into, each a package of
its own. MLX's simdgroups are 32 lanes wide; the Direct3D and OpenGL packages
of a program with subgroup operations lower them to 32-lane software
subgroups.
"""

import gzip
import hashlib
import json
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
from demos.integrations.mlx.portable_host import custom_kernel_layout as layout
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

CACHE_DIRECTORY = "custom-kernels"
SOURCE_FILE = "kernel.metal"
TARGETS = frozenset({"metal", "directx", "opengl"})
KERNELS = "mlx/backend/metal/kernels"
HEADERS = (f"{KERNELS}/utils.h",)
SIMD_SIZE = 32
# MLX compiles a fast-math kernel with fast math enabled
# (MTL::CompileOptions::setFastMathEnabled), under which the Metal compiler
# defines __FAST_MATH__; the safe and relaxed modes do not enable it.
FAST_MATH_DEFINES = {"fast": {"__FAST_MATH__": "1"}}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def target_options(target, region, subgroups):
    """The Metal frontend's options for ``target``'s package of one region."""
    if target == "metal":
        return {}
    options = {"dispatch_region": region.to_json()}
    if subgroups:
        options["software_subgroup_width"] = SIMD_SIZE
        if target == "directx":
            options["relative_wave_shuffle_out_of_range"] = "self"
    return {target: options}


def project_config(root, work, identity, region, subgroups):
    """The single-entry project for a custom kernel's program in ``work``."""
    path = f"{work}/{SOURCE_FILE}"
    target = identity["target"]
    return ProjectConfig(
        root=root,
        source_roots=(work,),
        include_patterns=(path,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work}/out",
        entry_points={path: (identity["entry"],)},
        workgroup_size=tuple(identity["workgroupSize"]),
        defines=dict(FAST_MATH_DEFINES.get(identity["mathMode"], {})),
        source_options={
            "metal": {
                "preserve_resource_origins": True,
                "target_options": target_options(target, region, subgroups),
            }
        },
    )


def _subgroups_absent(payload, target):
    """Whether translation failed only because no subgroup operation exists."""
    errors = {
        item["code"]
        for item in payload["diagnostics"]
        if item.get("severity") == "error"
    }
    return errors == {f"project.translate.{target}-software-subgroup-invalid"}


class CustomKernelPackageCache:
    """One target's packages for the kernels mx.fast.metal_kernel builds."""

    def __init__(self, root, directory, target):
        if target not in TARGETS:
            raise ValueError("Unsupported custom-kernel target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

    def _require_headers(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--", KERNELS],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError("Custom kernel translation requires unchanged kernels")

    def identity(self, entry, source_hash, mode, workgroup_size, region):
        here = Path(__file__)
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "entry": entry,
            "sourceHash": source_hash,
            "mathMode": mode,
            "workgroupSize": list(workgroup_size),
            "region": None if region is None else region.to_json(),
            "softwareSubgroupWidth": None if self.target == "metal" else SIMD_SIZE,
            "implementationHash": translation_implementation_hash(),
            "recipeHash": _sha256(here.read_bytes()),
            "layoutHash": _sha256(
                here.with_name("custom_kernel_layout.py").read_bytes()
            ),
            "headerHashes": {
                header: _sha256((self.root / header).read_bytes()) for header in HEADERS
            },
        }

    def get(self, entry, source, *, math_mode, thread_grid_size, workgroup_size):
        """The ``(descriptor, package directory)`` pairs that run one launch.

        Metal runs the exact grid from one package; Direct3D and OpenGL get
        one package per dispatch region, in dispatch order.
        """
        layout.parse(entry, source)
        mode = layout.MATH_MODES[math_mode]
        data = source.encode("utf-8")
        regions = plan_dispatch_regions(thread_grid_size, workgroup_size)
        nominal = regions[0].source_workgroup_size
        items = [None] if self.target == "metal" else regions
        with self._lock:
            packages = []
            for region in items:
                size = nominal if region is None else region.workgroup_size
                key = (entry, _sha256(data), mode, tuple(size), region)
                cached = self._loaded.get(key)
                if cached is None:
                    cached = self._package(entry, data, mode, size, region)
                    self._loaded[key] = cached
                packages.append(cached)
            if self.target == "metal":
                return tuple(packages)
            return select_native_loader_dispatch_regions(
                packages,
                thread_grid_size=thread_grid_size,
                source_workgroup_size=nominal,
            )

    def _package(self, entry, data, mode, size, region):
        self._require_headers()
        identity = self.identity(entry, _sha256(data), mode, size, region)
        key = _sha256(json.dumps(identity, sort_keys=True).encode())
        destination = self.directory / key
        if not destination.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".custom-kernel-package-", dir=self.directory
            ) as temporary:
                staging = Path(temporary) / "entry"
                staging.mkdir()
                self._build(staging, identity, data, region)
                # Do not publish a package built while its inputs changed.
                self._require_headers()
                if self.identity(entry, identity["sourceHash"], mode, size, region) != (
                    identity
                ):
                    raise ValueError(
                        "Custom kernel translator or recipe changed during translation"
                    )
                try:
                    staging.rename(destination)
                except OSError:
                    if not destination.is_dir():
                        raise
        return self._load(destination, identity)

    def find(self, entry, source_hash, math_mode, *, thread_grid_size, workgroup_size):
        """The packages and source a trace's ``source_hash`` names."""
        for directory in sorted(self.directory.glob("[0-9a-f]" * 64)):
            record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
            identity = record.get("identity", {})
            if (
                identity.get("entry") == entry
                and identity.get("sourceHash") == source_hash
            ):
                source = self.source(directory, source_hash)
                packages = self.get(
                    entry,
                    source,
                    math_mode=math_mode,
                    thread_grid_size=thread_grid_size,
                    workgroup_size=workgroup_size,
                )
                return packages, source
        raise ValueError("No cached custom-kernel package for the traced source")

    @staticmethod
    def source(directory, source_hash):
        """The source a package translated, with MLX's prepended headers."""
        data = (Path(directory) / SOURCE_FILE).read_bytes()
        prefix = layout.PREAMBLE.encode("utf-8")
        if not data.startswith(prefix) or _sha256(data[len(prefix) :]) != source_hash:
            raise ValueError("Cached custom kernel source changed after translation")
        return data[len(prefix) :].decode("utf-8")

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError("Custom kernel cache identity does not match the source")
        program = layout.PREAMBLE + self.source(directory, identity["sourceHash"])
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid custom-kernel runtime package: {loader['diagnostics']}"
            )
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": _sha256(program.encode("utf-8")),
            }
            or loader["loadUnits"][0]["entryPoint"]["source"] != identity["entry"]
            or type(record.get("softwareSubgroups")) is not bool
            or (
                record["softwareSubgroups"]
                and identity["softwareSubgroupWidth"] is None
            )
        ):
            raise ValueError(
                "Custom kernel cache descriptor does not match its package"
            )
        if identity["region"] is not None and (
            descriptor["provenance"].get("dispatchRegion") != identity["region"]
        ):
            raise ValueError("Custom kernel region does not match its cache identity")
        return descriptor, directory / "package"

    def _translate(self, work, identity, region):
        """Translate with software subgroups where the program has subgroup
        operations; the translator refuses them for a program without any."""
        subgroups = identity["softwareSubgroupWidth"] is not None
        while True:
            report = translate_project(
                project_config(self.root, work.name, identity, region, subgroups),
                format_output=False,
            )
            payload = report.to_json()
            if subgroups and _subgroups_absent(payload, self.target):
                subgroups = False
                continue
            return report, payload, subgroups

    def _build(self, output, identity, data, region):
        entry = identity["entry"]
        program = layout.PREAMBLE.encode("utf-8") + data
        with tempfile.TemporaryDirectory(
            prefix=".custom-kernel-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            (work / SOURCE_FILE).write_bytes(program)
            report, payload, subgroups = self._translate(work, identity, region)
            report.write_json(work / "report.json")
            if (
                payload["summary"]["failedCount"]
                or len(payload["artifacts"]) != 1
                or payload["artifacts"][0]["entryPoint"]["source"] != entry
            ):
                raise ValueError(
                    f"Custom kernel translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Custom kernel manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Custom kernel packaging failed: {package['diagnostics']}"
                )
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(
                    f"Custom kernel loading failed: {loader['diagnostics']}"
                )
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / SOURCE_FILE).write_bytes(program)
            (output / "report.json.gz").write_bytes(
                gzip.compress((work / "report.json").read_bytes(), mtime=0)
            )
            (output / "index.json").write_text(
                json.dumps(
                    {
                        "identity": identity,
                        "descriptor": descriptor,
                        "softwareSubgroups": subgroups,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
