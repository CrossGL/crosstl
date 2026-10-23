"""Translate unchanged pinned MLX kernel entries on first use, one package each.

A recipe names the pinned source, how to recognize its entries and the project
options an entry needs. Each entry is translated once and cached under a
content identity that covers the translator, the recipe module, this module and
the kernel source, so a changed translator or recipe never reuses an older
package. Loading checks that the cached descriptor still matches its package.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.translator.dispatch_region_identity import translation_implementation_hash
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

TARGETS = frozenset({"metal", "directx", "opengl"})
KERNELS = "mlx/backend/metal/kernels"


@dataclass(frozen=True)
class KernelRecipe:
    """How one pinned source's entries are recognized and translated.

    ``signature`` raises ValueError for an entry the recipe does not cover;
    ``options`` returns the extra project TOML for an entry on a target.
    """

    label: str
    source: str
    signature: Callable[[str], object]
    options: Callable[[str, str], str]
    module: str


def index_range_assertions(source, expressions):
    """Host-enforced ``[0, INT32_MAX]`` bounds for OpenGL's signed indices."""
    return "".join(
        "[[project.index_range_assertions]]\n"
        f'source = "{source}"\n'
        f'expression = "{expression}"\n'
        "minimum = 0\n"
        f"maximum = {2**31 - 1}\n"
        for expression in expressions
    )


def project_config(output, target, source, entry, options=""):
    """The single-entry project for ``entry`` with a one-thread workgroup."""
    return (
        "[project]\n"
        f'source_roots = ["{KERNELS}"]\n'
        f'include = ["{source}"]\n'
        'include_dirs = ["."]\n'
        f'targets = ["{target}"]\n'
        f'output_dir = "{output}"\n'
        "[project.entry_points]\n"
        f'"{source}" = "{entry}"\n'
        f'[project.entry_workgroup_size_rules."{source}"]\n'
        f'"{entry}" = [1, 1, 1]\n'
        "[project.source_options.metal]\n"
        "max_template_specializations = 512\n"
        "max_template_materialization_work = 65536\n"
        f"{options}"
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LazyKernelCache:
    """One target's packages for a recipe's entries, translated on first use."""

    def __init__(self, root, directory, target, recipe):
        if target not in TARGETS:
            raise ValueError(f"Unsupported {recipe.label} target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target, self.recipe = target, recipe
        self._lock = threading.RLock()
        self._loaded = {}

    def _require_source(self):
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--", KERNELS],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError(
                f"{self.recipe.label} translation requires unchanged pinned kernels"
            )

    def identity(self, entry):
        self.recipe.signature(entry)
        return {
            "schemaVersion": 2,
            "revision": COMMIT,
            "target": self.target,
            "source": self.recipe.source,
            "entry": entry,
            "workgroupSize": [1, 1, 1],
            "implementationHash": translation_implementation_hash(),
            "recipeHash": _sha256(self.recipe.module),
            "cacheHash": _sha256(__file__),
            "sourceHash": _sha256(self.root / self.recipe.source),
        }

    def get(self, entry):
        """The ``(descriptor, package directory)`` of ``entry``."""
        with self._lock:
            if entry in self._loaded:
                return self._loaded[entry]
            self._require_source()
            identity = self.identity(entry)
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".lazy-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity)
                    self._require_source()
                    if self.identity(entry) != identity:
                        raise ValueError(
                            f"{self.recipe.label} translator or recipe changed "
                            "during translation"
                        )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            self._loaded[entry] = self._load(destination, identity)
            return self._loaded[entry]

    def _load(self, directory, identity):
        label = self.recipe.label
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError(f"{label} cache identity does not match the entry")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid {label} runtime package: {loader['diagnostics']}"
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
            raise ValueError(f"{label} cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, output, identity):
        entry, label = identity["entry"], self.recipe.label
        with tempfile.TemporaryDirectory(
            prefix=".lazy-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = work / "crosstl.toml"
            config.write_text(
                project_config(
                    f"{work.name}/out",
                    self.target,
                    self.recipe.source,
                    entry,
                    self.recipe.options(self.target, entry),
                ),
                encoding="utf-8",
            )
            report = translate_project(
                load_project_config(self.root, config), format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if (
                payload["summary"]["failedCount"]
                or len(payload["artifacts"]) != 1
                or payload["artifacts"][0]["entryPoint"]["source"] != entry
            ):
                raise ValueError(
                    f"{label} translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"{label} manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"{label} packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"{label} loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
