"""Translate unchanged ``binary_two.metal`` DivMod entries on first use.

``DivMod`` is MLX's only two-output elementwise primitive. Its host port picks
the same kernel families as ``binary_op_gpu`` (scalar, vector and general
strided layouts), so each kernel/dtype pair is translated once from the pinned
source and cached under a content identity that covers the translator, this
recipe and the kernel source.
"""

import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

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

SOURCE = "mlx/backend/metal/kernels/binary_two.metal"
CACHE = "divmod"
# binary_two.metal instantiates these N=1 and int-indexed kernels for each type.
KERNELS = ("ss", "sv", "vs", "vv", "g1", "g2", "g3", "gn2")
DTYPES = (
    "float16",
    "float32",
    "bfloat16",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
)
ENTRIES = {
    f"{kernel}_DivMod{dtype}": (kernel, dtype) for kernel in KERNELS for dtype in DTYPES
}
# The pinned per-type profiles reproduce Metal's binary32 division, comparison,
# remainder and additive results, and its binary16 remainder quotient.
BINARY32_PROFILES = (
    ("binary32_division_profile", "rne-flush"),
    ("binary32_comparison_profile", "flush-subnormals"),
    ("binary32_remainder_profile", "flush-arithmetic-subnormals"),
    ("binary32_additive_profile", "rne-flush"),
)
PROFILES = {
    "float32": BINARY32_PROFILES,
    "bfloat16": BINARY32_PROFILES,
    "float16": (("binary16_remainder_profile", "binary32-quotient"),),
}
# OpenGL lowers the kernels' element indices to signed ints. divmod_dispatch
# bounds every output element and input span below 2**31, so these hold.
OPENGL_INDEX_EXPRESSIONS = (
    "offset + i",
    "a_idx",
    "b_idx",
    "out_idx",
    "out_idx++",
    "idx.x",
    "idx.y",
)


def signature(entry):
    try:
        return ENTRIES[entry]
    except (KeyError, TypeError) as error:
        raise ValueError("Unknown DivMod entry") from error


def _config(output, target, entry):
    _, dtype = signature(entry)
    profiles = "".join(
        f'{option} = "{value}"\n' for option, value in PROFILES.get(dtype, ())
    )
    assertions = "".join(
        "[[project.index_range_assertions]]\n"
        f'source = "{SOURCE}"\n'
        f'expression = "{expression}"\n'
        "minimum = 0\n"
        f"maximum = {2**31 - 1}\n"
        for expression in (OPENGL_INDEX_EXPRESSIONS if target == "opengl" else ())
    )
    return (
        "[project]\n"
        'source_roots = ["mlx/backend/metal/kernels"]\n'
        f'include = ["{SOURCE}"]\n'
        'include_dirs = ["."]\n'
        f'targets = ["{target}"]\n'
        f'output_dir = "{output}"\n'
        "[project.entry_points]\n"
        f'"{SOURCE}" = "{entry}"\n'
        f'[project.entry_workgroup_size_rules."{SOURCE}"]\n'
        f'"{entry}" = [1, 1, 1]\n'
        "[project.source_options.metal]\n"
        "max_template_specializations = 512\n"
        "max_template_materialization_work = 65536\n"
        f"{profiles}{assertions}"
    )


class DivModPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError("Unsupported DivMod target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

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
            raise ValueError("DivMod translation requires unchanged pinned kernels")

    def identity(self, entry):
        signature(entry)
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "entry": entry,
            "workgroupSize": [1, 1, 1],
            "implementationHash": translation_implementation_hash(),
            "recipeHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sourceHash": hashlib.sha256((self.root / SOURCE).read_bytes()).hexdigest(),
        }

    def get(self, entry):
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
                    prefix=".divmod-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity)
                    self._require_source()
                    if self.identity(entry) != identity:
                        raise ValueError(
                            "DivMod translator or recipe changed during translation"
                        )
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            self._loaded[entry] = self._load(destination, identity)
            return self._loaded[entry]

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if record.get("identity") != identity:
            raise ValueError("DivMod cache identity does not match the entry")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(f"Invalid DivMod runtime package: {loader['diagnostics']}")
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
            raise ValueError("DivMod cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, output, identity):
        entry = identity["entry"]
        with tempfile.TemporaryDirectory(
            prefix=".divmod-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = work / "crosstl.toml"
            config.write_text(
                _config(f"{work.name}/out", self.target, entry), encoding="utf-8"
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
                raise ValueError(f"DivMod translation failed: {payload['diagnostics']}")
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"DivMod manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"DivMod packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"DivMod loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )
