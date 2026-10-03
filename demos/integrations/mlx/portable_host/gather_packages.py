"""Build exact pinned general-gather JIT specializations on demand."""

import hashlib
import json
import re
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
from demos.integrations.mlx.portable_host import gather_axis_layout, gather_layout
from demos.integrations.mlx.portable_host.gather_layout import MAX_ELEMENTS
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

JIT_HEADER = "mlx/backend/metal/jit/indexing.h"
SOURCE_TYPES = {
    "float32": "float",
    "int32": "int",
    "uint32": "uint",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "bool_": "bool",
}


def signature(entry):
    layout = gather_axis_layout if entry.startswith("gather_axis") else gather_layout
    return layout.signature(entry)


def source(root, entry):
    if entry.startswith("gather_axis"):
        dtype, index_dtype, source_contiguous, index_contiguous = signature(entry)
        arguments = ", ".join(
            (
                SOURCE_TYPES[dtype],
                SOURCE_TYPES[index_dtype],
                "int",
                str(source_contiguous).lower(),
                str(index_contiguous).lower(),
            )
        )
        return (
            '#include "mlx/backend/metal/kernels/utils.h"\n'
            '#include "mlx/backend/metal/kernels/indexing/gather_axis.h"\n'
            f'template [[host_name("{entry}")]] [[kernel]]\n'
            f"decltype(gather_axis<{arguments}>) gather_axis<{arguments}>;\n"
        )
    dtype, index_dtype, count, ndim = signature(entry)
    template = re.search(
        r'gather_kernels = R"\((.*?)\)";', (root / JIT_HEADER).read_text(), re.S
    )
    if template is None:
        raise ValueError("Pinned general-gather JIT definition is missing")
    kind = SOURCE_TYPES[index_dtype]
    arguments = "\n".join(
        f"const device {kind}* idx{i} [[buffer({20 + i})]]," for i in range(count)
    )
    wrapper = template.group(1).format(
        dtype + index_dtype,
        SOURCE_TYPES[dtype],
        kind,
        count,
        arguments,
        ", ".join(f"idx{i}" for i in range(count)),
        ndim,
        "int",
    )
    return (
        '#include "mlx/backend/metal/kernels/utils.h"\n'
        '#include "mlx/backend/metal/kernels/indexing/gather.h"\n' + wrapper
    )


class GatherPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError("Unsupported gather target")
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
                JIT_HEADER,
            ],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError(
                "Gather translation requires unchanged pinned kernels and JIT definitions"
            )

    def get(self, entry, maximum_index):
        signature(entry)
        if type(maximum_index) is not int or not 0 <= maximum_index < MAX_ELEMENTS:
            raise ValueError("Gather index bound exceeds the validated host contract")
        with self._lock:
            self._require_source()
            implementation = translation_implementation_hash()
            identity = {
                "schemaVersion": 1,
                "revision": COMMIT,
                "target": self.target,
                "entry": entry,
                "maximumIndex": maximum_index,
                "implementationHash": implementation,
                "recipeHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            key = hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            destination = self.directory / key
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".gather-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(entry, staging, identity)
                    self._require_source()
                    if translation_implementation_hash() != implementation:
                        raise ValueError(
                            "Translator changed while building a gather package"
                        )
                    if (
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                        != identity["recipeHash"]
                    ):
                        raise ValueError(
                            "Gather packaging recipe changed during translation"
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
            raise ValueError("Gather cache identity does not match the request")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(f"Invalid gather runtime package: {loader['diagnostics']}")
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(
                self.target, identity["entry"]
            )
            or descriptor["source"]["hash"]
            != {
                "algorithm": "sha256",
                "value": (
                    hashlib.sha256(
                        source(self.root, identity["entry"]).encode("utf-8")
                    ).hexdigest()
                ),
            }
        ):
            raise ValueError("Gather cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, entry, output, identity):
        with tempfile.TemporaryDirectory(
            prefix=".gather-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            wrapper = work / "gather.metal"
            wrapper.write_bytes(source(self.root, entry).encode("utf-8"))
            relative = wrapper.relative_to(self.root).as_posix()
            config = ProjectConfig(
                root=self.root,
                source_roots=(work.name,),
                include_patterns=(relative,),
                include_dirs=(".",),
                targets=(self.target,),
                output_dir=f"{work.name}/out",
                entry_points={relative: (entry,)},
                workgroup_size=(1, 1, 1),
                index_range_assertions=(
                    (
                        {
                            "source": relative,
                            "expression": "reference.offset + index",
                            "minimum": 0,
                            "maximum": identity["maximumIndex"],
                        },
                    )
                    if self.target == "opengl" and not entry.startswith("gather_axis")
                    else ()
                ),
            )
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(f"Gather translation failed: {payload['diagnostics']}")
            if payload["artifacts"][0]["entryPoint"]["source"] != entry:
                raise ValueError("Gather translation selected a different entry")
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"Gather manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"Gather packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"Gather loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2)
            )
