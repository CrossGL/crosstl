"""Translate the start-offset kernel MLX generates for dynamic slices.

``compute_dynamic_offset`` in the pinned ``mlx/backend/metal/slicing.cpp``
formats a one-thread kernel per index dtype from a source template. The
kernel sums each start index times the stride of its axis into one int64
offset, which the dynamic copy kernels then read. The template is taken from
the unchanged pinned file and formatted exactly as ``fmt::format`` does, and
each dtype's kernel is translated once into a cache. The cache identity covers
the translator, this recipe, the template file's hash and the formatted
source, so any change translates again. The formatted source is kept with its
package, so an audit can derive the package again from a trace.
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
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

CACHE = "dynamic-offsets"
TEMPLATE_SOURCE = "mlx/backend/metal/slicing.cpp"
SOURCE_FILE = "kernel.metal"
ENTRY_PREFIX = "compute_dynamic_offset_"
TARGETS = frozenset({"metal", "directx", "opengl"})
# type_to_name and get_type_string of the index dtypes the host binds.
TYPES = {
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
}
ENTRIES = {ENTRY_PREFIX + name: name for name in TYPES}
# The function that formats the template, and the raw string it formats.
FUNCTION = "array compute_dynamic_offset("
RAW_START, RAW_END = 'R"(', ')"'


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def signature(entry):
    """The index dtype of a dynamic offset entry."""
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown dynamic offset entry") from error


def template(text):
    """The raw-string template ``compute_dynamic_offset`` formats in ``text``."""
    if text.count(FUNCTION) != 1:
        raise ValueError("The pinned slicing source has no single offset kernel")
    body = text[text.index(FUNCTION) :]
    if RAW_START not in body:
        raise ValueError("The pinned offset kernel has no source template")
    start = body.index(RAW_START) + len(RAW_START)
    end = body.find(RAW_END, start)
    if end < 0:
        raise ValueError("The pinned offset kernel template is unterminated")
    return body[start:end]


def format_source(pattern, arguments):
    """Format ``pattern`` as ``fmt::format`` does with positional ``arguments``.

    Only doubled braces and ``{N}`` fields occur in the pinned template; any
    other replacement field is refused rather than guessed.
    """
    output, i = [], 0
    while i < len(pattern):
        pair = pattern[i : i + 2]
        if pair in ("{{", "}}"):
            output.append(pair[0])
            i += 2
            continue
        if pattern[i] == "}":
            raise ValueError("Unmatched closing brace in the offset template")
        if pattern[i] != "{":
            output.append(pattern[i])
            i += 1
            continue
        end = pattern.find("}", i)
        field = pattern[i + 1 : end] if end > 0 else ""
        if not field.isdigit() or int(field) >= len(arguments):
            raise ValueError("Unsupported field in the offset template")
        output.append(arguments[int(field)])
        i = end + 1
    return "".join(output)


def render(text, entry):
    """The kernel source MLX generates for ``entry`` from slicing.cpp ``text``."""
    dtype = signature(entry)
    return format_source(template(text), (dtype, TYPES[dtype]))


def project_config(root, work, target, entry):
    """The single-entry project for a formatted offset kernel in ``work``."""
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
        source_options={"metal": {"preserve_resource_origins": True}},
    )


class DynamicOffsetPackageCache:
    """One target's translated ``compute_dynamic_offset`` kernels."""

    def __init__(self, root, directory, target):
        if target not in TARGETS:
            raise ValueError("Unsupported dynamic offset target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

    def pinned_text(self):
        """The unchanged pinned slicing.cpp, refused if it was modified."""
        require_revision(self.root)
        pristine = subprocess.check_output(
            ["git", "-C", str(self.root), "show", f"HEAD:{TEMPLATE_SOURCE}"],
            timeout=30,
        )
        path = self.root / TEMPLATE_SOURCE
        if path.is_symlink() or path.read_bytes() != pristine:
            raise ValueError("Dynamic offsets require the unchanged pinned slicing.cpp")
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
                    prefix=".dynamic-offset-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity, render(text, entry))
                    if self.identity(entry, self.pinned_text()) != identity:
                        raise ValueError(
                            "Dynamic offset translator or recipe changed during "
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
            raise ValueError("Dynamic offset cache identity does not match the source")
        if _sha256((directory / SOURCE_FILE).read_bytes()) != identity["sourceHash"]:
            raise ValueError("Cached dynamic offset source changed after translation")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid dynamic offset runtime package: {loader['diagnostics']}"
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
                "Dynamic offset cache descriptor does not match its package"
            )
        return descriptor, directory / "package"

    def _build(self, output, identity, source):
        entry = identity["entry"]
        data = source.encode("utf-8")
        with tempfile.TemporaryDirectory(
            prefix=".dynamic-offset-translate-", dir=self.root
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
                    f"Dynamic offset translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Dynamic offset manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Dynamic offset packaging failed: {package['diagnostics']}"
                )
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(
                    f"Dynamic offset loading failed: {loader['diagnostics']}"
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
