"""Translate the kernels MLX generates for fused tapes, one package per source.

A compiled kernel has no pinned source file: the host generates it from the
tape with MLX's ``build_kernel``. Each source is translated once into a cache
whose identity covers the translator, this recipe, the layout parser, the
source's SHA-256 and the float profiles its operations need, so equal sources
share a package and any change translates again. The source is kept in the
package directory, so an audit can derive the package again from the hash a
trace records.
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
    translate_project,
)
from crosstl.translator.dispatch_region_identity import translation_implementation_hash
from demos.integrations.mlx.portable_host import compiled_layout
from demos.integrations.mlx.portable_host.elementwise_packages import (
    PROFILES,
    operation_profiles,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

CACHE_DIRECTORY = "compiled"
SOURCE_FILE = "kernel.metal"
TARGETS = frozenset({"metal", "directx", "opengl"})
# The pinned headers every compiled source includes.
HEADERS = tuple(
    line.split('"')[1] for line in compiled_layout.PREAMBLE.strip().split("\n")
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def profiles(source):
    """The float profiles of every tape operation, as the vector families pin them.

    A compiled kernel computes each operation as its unary, binary or ternary
    vector does, so it takes the union of those vectors' profiles.
    """
    selected = dict(PROFILES)
    for kind, operation, dtype in compiled_layout.operations(source):
        if kind == "cast":
            continue
        for option, value in operation_profiles(kind, operation, dtype):
            if selected.get(option, value) != value:
                raise ValueError("Compiled operations need conflicting float profiles")
            selected[option] = value
    return tuple(sorted(selected.items()))


def project_config(root, work, target, entry, options):
    """The single-entry project for a generated source in ``work``."""
    path = f"{work}/{SOURCE_FILE}"
    return ProjectConfig(
        root=root,
        source_roots=(work,),
        include_patterns=(path,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work}/out",
        entry_points={path: (entry,)},
        workgroup_size=(1, 1, 1),
        source_options={"metal": {**dict(options), "preserve_resource_origins": True}},
    )


class CompiledPackageCache:
    """One target's packages for generated compiled kernels."""

    def __init__(self, root, directory, target):
        if target not in TARGETS:
            raise ValueError("Unsupported compiled-kernel target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

    def _require_headers(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--", *HEADERS],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError("Compiled translation requires unchanged pinned headers")

    def identity(self, entry, source_hash, options):
        compiled_layout.variant(entry)
        here = Path(__file__)
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "entry": entry,
            "sourceHash": source_hash,
            "profiles": [list(item) for item in options],
            "workgroupSize": [1, 1, 1],
            "implementationHash": translation_implementation_hash(),
            "recipeHash": _sha256(here.read_bytes()),
            "layoutHash": _sha256(here.with_name("compiled_layout.py").read_bytes()),
            "headerHashes": {
                header: _sha256((self.root / header).read_bytes()) for header in HEADERS
            },
        }

    def _key(self, identity):
        return _sha256(json.dumps(identity, sort_keys=True).encode())

    def get(self, entry, source):
        """The ``(descriptor, package directory)`` that runs ``source``."""
        compiled_layout.parse(entry, source)
        data = source.encode("utf-8")
        options = profiles(source)
        with self._lock:
            cached = self._loaded.get((entry, _sha256(data)))
            if cached is not None:
                return cached
            self._require_headers()
            identity = self.identity(entry, _sha256(data), options)
            destination = self.directory / self._key(identity)
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".compiled-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity, data, options)
                    self._require_headers()
                    if (
                        self.identity(entry, identity["sourceHash"], options)
                        != identity
                    ):
                        raise ValueError(
                            "Compiled translator or recipe changed during translation"
                        )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            loaded = self._load(destination, identity)
            self._loaded[(entry, identity["sourceHash"])] = loaded
            return loaded

    def find(self, entry, source_hash):
        """The cached package and source a trace's ``source_hash`` names."""
        for directory in sorted(self.directory.glob("[0-9a-f]" * 64)):
            record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
            identity = record.get("identity", {})
            if (
                identity.get("entry") == entry
                and identity.get("sourceHash") == source_hash
            ):
                source = (directory / SOURCE_FILE).read_bytes()
                if _sha256(source) != source_hash:
                    raise ValueError("Cached compiled source changed after translation")
                return self.get(entry, source.decode("utf-8")), source.decode("utf-8")
        raise ValueError("No cached compiled package for the traced source")

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError("Compiled cache identity does not match the source")
        if _sha256((directory / SOURCE_FILE).read_bytes()) != identity["sourceHash"]:
            raise ValueError("Cached compiled source changed after translation")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid compiled runtime package: {loader['diagnostics']}"
            )
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
            raise ValueError("Compiled cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, output, identity, data, options):
        entry = identity["entry"]
        with tempfile.TemporaryDirectory(
            prefix=".compiled-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            (work / SOURCE_FILE).write_bytes(data)
            report = translate_project(
                project_config(self.root, work.name, self.target, entry, options),
                format_output=False,
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            if (
                payload["summary"]["failedCount"]
                or len(payload["artifacts"]) != 1
                or payload["artifacts"][0]["entryPoint"]["source"] != entry
            ):
                raise ValueError(
                    f"Compiled translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"Compiled manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"Compiled packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"Compiled loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / SOURCE_FILE).write_bytes(data)
            (output / "report.json.gz").write_bytes(
                gzip.compress((work / "report.json").read_bytes(), mtime=0)
            )
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
