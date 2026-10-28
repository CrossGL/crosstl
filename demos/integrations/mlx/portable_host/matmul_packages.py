"""Cache translated affine matrix-vector entries from the unchanged pinned source.

Run as a module, it translates entries into a host package directory's cache
ahead of execution, several processes at a time.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
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


class TranslatedEntryCache:
    """Translate and package single entries of one unchanged pinned source.

    Subclasses name the source and supply each entry's validation, workgroup
    size, translation config and recipe hashes.
    """

    LABEL = "Translated entry"
    SOURCE = None
    PREFIX = "entry"

    def signature(self, entry):
        raise NotImplementedError

    def workgroup_size(self, entry):
        raise NotImplementedError

    def project_config(self, output_dir, entry):
        raise NotImplementedError

    def recipe_hashes(self):
        raise NotImplementedError

    def source_entry(self, entry):
        """The source entry point an entry translates; entries may add constants."""
        return entry

    def __init__(self, root, directory, target):
        if target not in {"metal", "directx", "opengl"}:
            raise ValueError(f"Unsupported {self.LABEL.lower()} target")
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
                f"{self.LABEL} translation requires unchanged pinned kernels"
            )

    def _identity(self, entry):
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "source": self.SOURCE,
            "sourceHash": (
                hashlib.sha256((self.root / self.SOURCE).read_bytes()).hexdigest()
            ),
            "target": self.target,
            "entry": entry,
            "maximumElements": MAX_ELEMENTS,
            "workgroupSize": list(self.workgroup_size(entry)),
            "implementationHash": translation_implementation_hash(),
            **self.recipe_hashes(),
        }

    def get(self, entry):
        self.signature(entry)
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
                    prefix=f".{self.PREFIX}-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(entry, staging)
                    self._require_source()
                    if self._identity(entry) != identity:
                        raise ValueError(
                            f"{self.LABEL} implementation changed while packaging"
                        )
                    loader = build_runtime_loader_manifest(
                        staging / "package/runtime-package.json"
                    )
                    if not loader["success"] or len(loader["loadUnits"]) != 1:
                        raise ValueError(
                            f"Invalid {self.LABEL.lower()} package: {loader['diagnostics']}"
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
            raise ValueError(f"{self.LABEL} cache identity differs")
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError(
                f"Invalid {self.LABEL.lower()} runtime package: {loader['diagnostics']}"
            )
        source_entry = self.source_entry(identity["entry"])
        if loader["loadUnits"][0]["entryPoint"]["source"] != source_entry:
            raise ValueError(f"{self.LABEL} package selects another entry")
        descriptor = build_native_loader_abi_descriptor(loader)
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(self.target, source_entry)
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": identity["sourceHash"],
            }
        ):
            raise ValueError(
                f"{self.LABEL} cache descriptor does not match its package"
            )
        return descriptor, directory / "package"

    def _build(self, entry, output):
        with tempfile.TemporaryDirectory(
            prefix=f".{self.PREFIX}-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            config = self.project_config(f"{work.name}/out", entry)
            report = translate_project(config, format_output=False)
            report.write_json(work / "report.json")
            payload = report.to_json()
            shutil.copytree(work, output / "translation")
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(
                    f"{self.LABEL} translation failed: {payload['diagnostics']}"
                )
            (artifact,) = payload["artifacts"]
            if artifact["entryPoint"]["source"] != self.source_entry(entry):
                raise ValueError(f"{self.LABEL} translation selected another entry")
            (execution,) = artifact["execution"]["entryPoints"]
            if execution["workgroupSize"] != list(self.workgroup_size(entry)):
                raise ValueError(f"{self.LABEL} translation changed the workgroup size")
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"{self.LABEL} manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"{self.LABEL} packaging failed: {package['diagnostics']}"
                )


class QuantizedMatmulPackageCache(TranslatedEntryCache):
    LABEL = "Quantized matrix-vector"
    SOURCE = SOURCE
    PREFIX = "quantized-matmul"

    def signature(self, entry):
        return signature(entry)

    def workgroup_size(self, entry):
        return workgroup_size(entry)

    def project_config(self, output_dir, entry):
        return project_config(self.root, output_dir, self.target, entry)

    def recipe_hashes(self):
        return _recipe_hashes()


# The directory under a host package root that HostRuntime caches entries in.
CACHE_DIRECTORY = "quantized-matmul"


def build_record(cache, entry):
    """Translate one entry through ``cache`` and describe its package."""
    start = time.monotonic()
    descriptor, package = cache.get(entry)
    return {
        "entry": entry,
        "target": cache.target,
        "package": str(package),
        "artifact": descriptor["artifact"],
        "seconds": round(time.monotonic() - start, 1),
    }


def _build_one(root, directory, target, entry):
    return build_record(QuantizedMatmulPackageCache(root, directory, target), entry)


def run_builds(builder, check, label, root, directory, target, entries, jobs):
    """Run ``builder(root, directory, target, entry)`` over ``entries``.

    Returns one record per entry in sorted order. The caches publish each
    entry by renaming a complete staging directory, so builders may overlap.
    """
    entries = sorted(set(entries))
    if not entries or type(jobs) is not int or jobs < 1:
        raise ValueError(f"{label} packaging needs entries and jobs")
    for entry in entries:
        check(entry)
    with ProcessPoolExecutor(max_workers=min(jobs, len(entries))) as pool:
        futures = [
            pool.submit(builder, str(root), str(directory), target, entry)
            for entry in entries
        ]
        return [future.result() for future in futures]


def build_entries(root, packages, target, entries, jobs):
    """Translate ``entries`` into ``packages``' cache, ``jobs`` at a time."""
    return run_builds(
        _build_one,
        signature,
        "Quantized matrix-vector",
        root,
        Path(packages).resolve() / CACHE_DIRECTORY,
        target,
        entries,
        jobs,
    )


def verification_entries():
    from demos.integrations.mlx.portable_host import verify_quantized_matmul

    return [
        verify_quantized_matmul.case_record(case)["entry"]
        for case in verify_quantized_matmul.CASES
    ]


def cli(argv, description, build, verification):
    """The package builders' command line: ``build`` and ``verification``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "directx", "opengl"), required=True
    )
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument(
        "--verification",
        action="store_true",
        help="add every entry the host verification routes",
    )
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    entries = list(args.entry) + (verification() if args.verification else [])
    records = build(args.mlx_root, args.packages, args.target, entries, args.jobs)
    for record in records:
        print(json.dumps(record), flush=True)
    return 0


def main(argv=None):
    return cli(argv, __doc__, build_entries, verification_entries)


if __name__ == "__main__":
    sys.exit(main())
