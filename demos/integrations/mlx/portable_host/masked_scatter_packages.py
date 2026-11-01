"""Translate the ``masked_assign`` kernels MLX generates for masked scatters.

``MaskedScatter::eval_gpu`` in the pinned ``mlx/backend/metal/indexing.cpp``
builds one library per value type and source contiguity: MLX's JIT
``utils`` and ``masked_scatter`` sources, followed by the
``masked_assign_kernel`` template of ``mlx/backend/metal/jit/indexing.h``
formatted with the kernel name, the value type and the contiguity. The host
includes the same two unchanged kernel headers and formats the unchanged
template exactly as ``fmt::format`` does. Each entry is translated once into a
cache whose identity covers the translator, this recipe, the template file's
hash and the formatted source, so any change translates again. The formatted
source is kept with its package, so an audit can derive the package again from
a trace.
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
from demos.integrations.mlx.portable_host.dynamic_offset_packages import format_source
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

CACHE = "masked-scatters"
TEMPLATE_SOURCE = "mlx/backend/metal/jit/indexing.h"
KERNELS = "mlx/backend/metal/kernels"
SOURCE_FILE = "kernel.metal"
ENTRY_PREFIX = "masked_assign_"
TARGETS = frozenset({"metal", "directx", "opengl"})
# type_to_name and get_type_string of every dtype a GPU array can hold.
TYPES = {
    "bool_": "bool",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "float16": "float16_t",
    "float32": "float",
    "bfloat16": "bfloat16_t",
    "complex64": "complex64_t",
}
ENTRIES = {
    f"{ENTRY_PREFIX}{dtype}_{contiguous}": (dtype, contiguous == "true")
    for dtype in TYPES
    for contiguous in ("true", "false")
}
# metal::utils() and metal::masked_scatter(), which indexing.cpp prepends.
HEADERS = (
    '#include "mlx/backend/metal/kernels/utils.h"\n'
    '#include "mlx/backend/metal/kernels/indexing/masked_scatter.h"\n'
)
DECLARATION = "constexpr std::string_view masked_assign_kernel = "
RAW_START, RAW_END = 'R"(', ')"'
# OpenGL lowers the contiguous read's int64 source index to a signed int.
# masked_scatter_dispatch.validate bounds the source span by INT32_MAX and the
# kernel reads only below batch_idx * src_batch_size + src_batch_size, so the
# subscript's precondition is host-enforced.
OPENGL_INDEX_ASSERTIONS = ("batch_idx * src_batch_size + src_index",)
SIGNED_MAX = 2**31 - 1


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def signature(entry):
    """The ``(dtype, source contiguous)`` of a masked assign entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown masked assign entry") from error


def template(text):
    """The raw-string ``masked_assign_kernel`` template in ``text``."""
    if text.count(DECLARATION) != 1:
        raise ValueError("The pinned indexing JIT source has no single masked assign")
    body = text[text.index(DECLARATION) + len(DECLARATION) :]
    if not body.startswith(RAW_START):
        raise ValueError("The pinned masked assign kernel has no source template")
    end = body.find(RAW_END, len(RAW_START))
    if end < 0:
        raise ValueError("The pinned masked assign template is unterminated")
    return body[len(RAW_START) : end]


def render(text, entry):
    """The kernel source MLX generates for ``entry`` from indexing.h ``text``."""
    dtype, contiguous = signature(entry)
    return HEADERS + format_source(
        template(text), (entry, TYPES[dtype], "true" if contiguous else "false")
    )


def project_config(root, work, target, entry):
    """The single-entry project for a formatted masked assign kernel in ``work``."""
    path = f"{work}/{SOURCE_FILE}"
    _, contiguous = signature(entry)
    assertions = ()
    if target == "opengl" and contiguous:
        assertions = tuple(
            {
                "source": path,
                "expression": expression,
                "minimum": 0,
                "maximum": SIGNED_MAX,
            }
            for expression in OPENGL_INDEX_ASSERTIONS
        )
    return ProjectConfig(
        root=root,
        source_roots=(work,),
        include_patterns=(path,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work}/out",
        entry_points={path: (entry,)},
        workgroup_size=(1, 1, 1),
        index_range_assertions=assertions,
        source_options={"metal": {"preserve_resource_origins": True}},
    )


class MaskedScatterPackageCache:
    """One target's translated ``masked_assign`` kernels."""

    def __init__(self, root, directory, target):
        if target not in TARGETS:
            raise ValueError("Unsupported masked scatter target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

    def pinned_text(self):
        """The unchanged pinned indexing.h, refused if it or a kernel changed."""
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--", KERNELS],
            text=True,
            timeout=30,
        )
        pristine = subprocess.check_output(
            ["git", "-C", str(self.root), "show", f"HEAD:{TEMPLATE_SOURCE}"],
            timeout=30,
        )
        path = self.root / TEMPLATE_SOURCE
        if dirty.strip() or path.is_symlink() or path.read_bytes() != pristine:
            raise ValueError(
                "Masked scatters require the unchanged pinned kernels and indexing.h"
            )
        return pristine.decode("utf-8")

    def identity(self, entry, text):
        source = render(text, entry).encode("utf-8")
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "entry": entry,
            "templateSource": TEMPLATE_SOURCE,
            "templateHash": _sha256(text.encode("utf-8")),
            "sourceHash": _sha256(source),
            "workgroupSize": [1, 1, 1],
            "implementationHash": translation_implementation_hash(),
            "recipeHash": _sha256(Path(__file__).read_bytes()),
        }

    def get(self, entry):
        """The ``(descriptor, package directory)`` that runs ``entry``."""
        signature(entry)
        with self._lock:
            cached = self._loaded.get(entry)
            if cached is not None:
                return cached
            text = self.pinned_text()
            identity = self.identity(entry, text)
            key = _sha256(json.dumps(identity, sort_keys=True).encode())
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".masked-scatter-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity, render(text, entry))
                    if self.identity(entry, self.pinned_text()) != identity:
                        raise ValueError(
                            "Masked scatter translator or recipe changed during "
                            "translation"
                        )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            loaded = self._load(destination, identity)
            self._loaded[entry] = loaded
            return loaded

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError("Masked scatter cache identity does not match the source")
        if _sha256((directory / SOURCE_FILE).read_bytes()) != identity["sourceHash"]:
            raise ValueError("Cached masked scatter source changed after translation")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid masked scatter runtime package: {loader['diagnostics']}"
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
            raise ValueError(
                "Masked scatter cache descriptor does not match its package"
            )
        return descriptor, directory / "package"

    def _build(self, output, identity, source):
        entry = identity["entry"]
        data = source.encode("utf-8")
        with tempfile.TemporaryDirectory(
            prefix=".masked-scatter-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            (work / SOURCE_FILE).write_bytes(data)
            report = translate_project(
                project_config(self.root, work.name, self.target, entry),
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
                    f"Masked scatter translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Masked scatter manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Masked scatter packaging failed: {package['diagnostics']}"
                )
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(
                    f"Masked scatter loading failed: {loader['diagnostics']}"
                )
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / SOURCE_FILE).write_bytes(data)
            (output / "report.json.gz").write_bytes(
                gzip.compress((work / "report.json").read_bytes(), mtime=0)
            )
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
