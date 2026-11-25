"""Translate the Hadamard kernels MLX generates, one package per stage.

``hadamard_mn_contiguous`` in the pinned ``mlx/backend/metal/hadamard.cpp``
has no pinned library: it writes the ``hadamard_radix_m`` codelet from the
Hadamard matrix strings of ``mlx/backend/common/hadamard.h`` and instantiates
``hadamard_n`` and ``hadamard_m`` of ``kernels/hadamard.h`` for one size and
type. ``hadamard_layout.render`` builds the same library from the unchanged
pinned files, and each stage is translated once into a cache whose identity
covers the translator, this recipe, the layout, the pinned headers and the
rendered source, so any change translates again. The source is kept with
its package, so an audit can derive the package again from a trace. Run as
a module, it translates entries ahead of execution, several at a time.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
from demos.integrations.mlx.portable_host import hadamard_layout as layout
from demos.integrations.mlx.portable_host.matmul_packages import (
    build_record,
    run_builds,
)
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision

CACHE = "hadamard"
SOURCE_FILE = "kernel.metal"
TARGETS = frozenset({"metal", "directx", "opengl"})
ENTRIES = layout.ENTRIES
# The kernel directory hadamard.h includes from, and the matrix strings.
PINNED_PATHS = ("mlx/backend/metal/kernels", layout.COMMON_HEADER)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def project_config(root, work, target, entry):
    """The single-entry project for one rendered stage in ``work``."""
    path = f"{work}/{SOURCE_FILE}"
    return ProjectConfig(
        root=root,
        source_roots=(work,),
        include_patterns=(path,),
        include_dirs=(".",),
        targets=(target,),
        output_dir=f"{work}/out",
        entry_points={path: (entry,)},
        workgroup_size=layout.workgroup_size(entry),
        source_options={"metal": {"preserve_resource_origins": True}},
    )


class HadamardPackageCache:
    """One target's translated Hadamard stages."""

    def __init__(self, root, directory, target):
        if target not in TARGETS:
            raise ValueError("Unsupported Hadamard target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()
        self._loaded = {}

    def pinned_text(self):
        """The unchanged pinned matrix header; refuses any modified input."""
        require_revision(self.root)
        dirty = subprocess.check_output(
            ["git", "-C", str(self.root), "status", "--porcelain", "--", *PINNED_PATHS],
            text=True,
            timeout=30,
        )
        if dirty.strip():
            raise ValueError("Hadamard translation requires unchanged pinned headers")
        return (self.root / layout.COMMON_HEADER).read_text(encoding="utf-8")

    def identity(self, entry, text):
        here = Path(__file__)
        source = layout.render(entry, text).encode("utf-8")
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "entry": entry,
            "sourceHash": _sha256(source),
            "workgroupSize": list(layout.workgroup_size(entry)),
            "headerHashes": {
                header: _sha256((self.root / header).read_bytes())
                for header in layout.HEADERS
            },
            "implementationHash": translation_implementation_hash(),
            "recipeHash": _sha256(here.read_bytes()),
            "layoutHash": _sha256(here.with_name("hadamard_layout.py").read_bytes()),
        }

    def get(self, entry):
        """The ``(descriptor, package directory)`` that runs ``entry``."""
        layout.signature(entry)
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
                    prefix=".hadamard-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity, layout.render(entry, text))
                    if self.identity(entry, self.pinned_text()) != identity:
                        raise ValueError(
                            "Hadamard translator or recipe changed during translation"
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
            raise ValueError("Hadamard cache identity does not match the source")
        if _sha256((directory / SOURCE_FILE).read_bytes()) != identity["sourceHash"]:
            raise ValueError("Cached Hadamard source changed after translation")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid Hadamard runtime package: {loader['diagnostics']}"
            )
        descriptor = build_native_loader_abi_descriptor(loader)
        entry = identity["entry"]
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(self.target, entry)
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": identity["sourceHash"],
            }
            or loader["loadUnits"][0]["entryPoint"]["source"] != entry
        ):
            raise ValueError("Hadamard cache descriptor does not match its package")
        return descriptor, directory / "package"

    def _build(self, output, identity, source):
        entry = identity["entry"]
        data = source.encode("utf-8")
        with tempfile.TemporaryDirectory(
            prefix=".hadamard-translate-", dir=self.root
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
                    f"Hadamard translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(f"Hadamard manifest failed: {manifest['diagnostics']}")
            (work / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(f"Hadamard packaging failed: {package['diagnostics']}")
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError(f"Hadamard loading failed: {loader['diagnostics']}")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / SOURCE_FILE).write_bytes(data)
            (output / "report.json.gz").write_bytes(
                gzip.compress((work / "report.json").read_bytes(), mtime=0)
            )
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2),
                encoding="utf-8",
            )


def _build_one(root, directory, target, entry):
    return build_record(HadamardPackageCache(root, directory, target), entry)


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        layout.signature,
        "Hadamard",
        root,
        Path(packages).resolve() / CACHE,
        target,
        entries,
        jobs,
    )


def verification_entries():
    """Every stage the unchanged upstream Hadamard tests launch.

    ``test_hadamard`` transforms 2**k points for k up to 13 and 28 * 2**k
    points for k up to 8 in float32 and float16 (and bfloat16 alongside
    float16), and m * 2**k float32 points for k from 14 to 16 and m in 1, 3,
    5 and 7; ``test_hadamard_grad_vmap`` transforms 4 to 128 float32 points.
    """
    sizes = {(2**k, dtype) for k in range(1, 14) for dtype in layout.DTYPES}
    sizes |= {(28 * 2**k, dtype) for k in range(1, 9) for dtype in layout.DTYPES}
    sizes |= {(m * 2**k, "float32") for k in range(14, 17) for m in (1, 3, 5, 7)}
    return sorted(
        {entry for size, dtype in sizes for entry in layout.stages(size, dtype)}
    )


def warm_entries(root, packages, target, entries, jobs, budget):
    """Translate ``entries`` ``jobs`` at a time, starting none after ``budget`` seconds.

    Returns the translated records in sorted order and the entries left for
    the host to translate on demand.
    """
    entries = sorted(set(entries))
    if not entries or type(jobs) is not int or jobs < 1 or not budget > 0:
        raise ValueError("Hadamard warming needs entries, jobs and a positive budget")
    for entry in entries:
        layout.signature(entry)
    directory = str(Path(packages).resolve() / CACHE)
    deadline = time.monotonic() + budget
    records, pending = [], list(entries)
    with ProcessPoolExecutor(max_workers=min(jobs, len(entries))) as pool:
        running = set()
        while pending or running:
            while pending and len(running) < jobs and time.monotonic() < deadline:
                running.add(
                    pool.submit(
                        _build_one, str(root), directory, target, pending.pop(0)
                    )
                )
            if not running:
                break
            done, running = wait(running, return_when=FIRST_COMPLETED)
            records.extend(future.result() for future in done)
    return sorted(records, key=lambda record: record["entry"]), pending


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument(
        "--verification",
        action="store_true",
        help="add every stage the upstream Hadamard tests launch",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        help="start no translation after this many seconds; the host translates the rest on demand",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    entries = list(args.entry) + (verification_entries() if args.verification else [])
    if args.budget_seconds is None:
        records, deferred = (
            build_entries(
                args.mlx_root, args.packages, args.target, entries, args.jobs
            ),
            None,
        )
    else:
        records, deferred = warm_entries(
            args.mlx_root,
            args.packages,
            args.target,
            entries,
            args.jobs,
            args.budget_seconds,
        )
    for record in records:
        print(json.dumps(record), flush=True)
    if deferred is not None:
        print(
            json.dumps({"translated": len(records), "deferred": deferred}), flush=True
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
