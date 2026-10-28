"""Package the unchanged pinned random kernels for native host dispatch."""

import argparse
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
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host.prepare import COMMIT, require_revision
from demos.integrations.mlx.portable_host.random_layout import (
    MAX_KEY_ELEMENTS,
    MAX_NATIVE_BYTES,
    RandomOutputLayout,
)

SOURCE = "mlx/backend/metal/kernels/random.metal"
SOURCE_SHA256 = "f1a19b3f11b7b10203824890f13debc6d627959b4e7f17c219e2e9da553c1bd7"
ENTRIES = ("rbitsc", "rbits")


def dispatch_contract(key_count, bytes_per_key, execution):
    layout = RandomOutputLayout(key_count, bytes_per_key)
    if bytes_per_key < 4 or layout.native_byte_count > MAX_NATIVE_BYTES:
        raise ValueError("Random cache requires bounded native byte storage")
    launch = layout.execution
    if execution != launch or any(
        type(value) is not int for values in execution.values() for value in values
    ):
        raise ValueError("Random cache launch differs from its output layout")
    return {"keyCount": key_count, "bytesPerKey": bytes_per_key, "execution": launch}


def _checked_contract(contract):
    if not isinstance(contract, dict) or set(contract) != {
        "keyCount",
        "bytesPerKey",
        "execution",
    }:
        raise ValueError("Random package requires a complete dispatch contract")
    return dispatch_contract(
        contract["keyCount"], contract["bytesPerKey"], contract["execution"]
    )


def _region(contract):
    count = contract["execution"]["workgroupCount"]
    size = contract["execution"]["workgroupSize"]
    (region,) = plan_dispatch_regions(
        [groups * width for groups, width in zip(count, size)], size
    )
    return region.to_json()


def _condition(entry, contract):
    return {
        "dtype": "uint64",
        "minimum": contract["bytesPerKey"],
        "maximum": contract["bytesPerKey"],
        "sourceResource": {
            "schemaVersion": 1,
            "backend": "metal",
            "entryPoint": entry,
            "parameter": "bytes_per_key",
            "parameterIndex": 3,
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_source(root):
    require_revision(root)
    if hashlib.sha256((root / SOURCE).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned random source hash differs")
    changed = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain",
            "--",
            "mlx/backend/metal/kernels",
        ],
        text=True,
        timeout=30,
    )
    if changed.strip():
        raise ValueError("Pinned kernel tree has local changes")


def build_packages(root, target, output, *, contract=None):
    if target not in {"metal", "opengl", "directx"}:
        raise ValueError("Unsupported random target")
    contract = (
        _checked_contract(contract)
        if contract is not None
        else dispatch_contract(
            1, 4, {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]}
        )
    )
    verify_source(root)
    with tempfile.TemporaryDirectory(prefix=".random-audit-", dir=root) as directory:
        work = Path(directory)
        config = ProjectConfig(
            root=root,
            source_roots=("mlx/backend/metal/kernels",),
            include_patterns=(SOURCE,),
            include_dirs=(".",),
            targets=(target,),
            output_dir=f"{work.name}/out",
            entry_points={SOURCE: ENTRIES},
            workgroup_size=tuple(contract["execution"]["workgroupSize"]),
            metadata_preconditions=[
                dict(
                    source=SOURCE,
                    entryPoint=entry,
                    parameter="bytes_per_key",
                    minimum=contract["bytesPerKey"],
                    maximum=contract["bytesPerKey"],
                )
                for entry in ENTRIES
            ],
            source_options={
                "metal": {
                    "preserve_resource_origins": True,
                    "target_options": {
                        target: (
                            {}
                            if target == "metal"
                            else {"dispatch_region": _region(contract)}
                        )
                    },
                }
            },
            index_range_assertions=(
                [
                    {
                        "source": SOURCE,
                        "function": function,
                        "expression": expression,
                        "minimum": 0,
                        "maximum": MAX_KEY_ELEMENTS - 1,
                    }
                    for function, expression in (
                        ("rbits", "k1_elem"),
                        ("rbits", "k2_elem"),
                    )
                ]
                if target == "opengl"
                else ()
            ),
        )
        report = translate_project(config, format_output=False)
        report.write_json(work / "report.json")
        shutil.copytree(work, output / "translation")
        payload = report.to_json()
        if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 2:
            raise ValueError("Random translation did not produce both entries")
        entries = {
            item["path"]: item["entryPoint"]["source"] for item in payload["artifacts"]
        }
        manifest = build_runtime_artifact_manifest(work / "report.json")
        if not manifest["success"]:
            raise ValueError("Random runtime manifest is incomplete")
        write_json(work / "artifacts.json", manifest)
        package = output / "package"
        result = build_runtime_package(work / "artifacts.json", package)
        if not result["success"]:
            raise ValueError("Random runtime package is incomplete")
        loader = build_runtime_loader_manifest(package / "runtime-package.json")
        if not loader["success"]:
            raise ValueError("Random loader manifest is incomplete")
        descriptors = {}
        for unit in loader["loadUnits"]:
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=unit["id"]
            )
            descriptors[entries[descriptor["source"]["artifactPath"]]] = descriptor
        if set(descriptors) != set(ENTRIES):
            raise ValueError("Random entry coverage differs")
        write_json(
            output / "index.json",
            {
                "commit": COMMIT,
                "target": target,
                "maximumNativeByteCount": MAX_NATIVE_BYTES,
                "contract": contract,
                "descriptors": descriptors,
            },
        )
        return descriptors


def load_index(directory, target, *, expected_contract=None):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if (
        not isinstance(index, dict)
        or index.get("commit") != COMMIT
        or index.get("target") != target
        or index.get("maximumNativeByteCount") != MAX_NATIVE_BYTES
        or not isinstance(index.get("descriptors"), dict)
        or set(index.get("descriptors", {})) != set(ENTRIES)
    ):
        raise ValueError("Random packages must match the pin, target and entry set")
    contract = _checked_contract(index.get("contract"))
    if expected_contract is not None and contract != _checked_contract(
        expected_contract
    ):
        raise ValueError("Random package contract differs from the requested launch")
    loader = build_runtime_loader_manifest(directory / "package/runtime-package.json")
    if not loader["success"] or len(loader["loadUnits"]) != 2:
        raise ValueError("Random runtime package is incomplete")
    observed = {}
    for unit in loader["loadUnits"]:
        name = unit.get("entryPoint", {}).get("source")
        descriptor = build_native_loader_abi_descriptor(loader, load_unit_id=unit["id"])
        if name not in ENTRIES or index["descriptors"][name] != descriptor:
            raise ValueError("Random descriptor differs from its runtime package")
        if (
            name in observed
            or descriptor["target"] != target
            or descriptor["stage"] != "compute"
            or descriptor["source"]["path"] != SOURCE
            or descriptor["source"]["backend"] != "metal"
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(target, name)
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
        ):
            raise ValueError("Random descriptor source or entry identity differs")
        conditions = [
            binding["provenance"]["metadataPrecondition"]
            for binding in descriptor["bindings"]
            if "metadataPrecondition" in binding.get("provenance", {})
        ]
        if conditions != [_condition(name, contract)] or descriptor.get(
            "provenance", {}
        ).get("dispatchRegion") != (None if target == "metal" else _region(contract)):
            raise ValueError("Random descriptor differs from its checked contract")
        observed[name] = descriptor
    if set(observed) != set(ENTRIES):
        raise ValueError("Random package entry coverage differs")
    return observed


class RandomPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "opengl", "directx"}:
            raise ValueError("Unsupported random target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()

    def _identity(self, key_count, bytes_per_key, execution):
        contract = dispatch_contract(key_count, bytes_per_key, execution)
        verify_source(self.root)
        return {
            "schemaVersion": 1,
            "revision": COMMIT,
            "target": self.target,
            "contract": contract,
            "sourceHash": SOURCE_SHA256,
            "implementationHash": translation_implementation_hash(),
            "recipeHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _directory(self, identity):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.directory / key

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text())
        if record.get("cacheIdentity") != identity or directory != self._directory(
            identity
        ):
            raise ValueError("Random cache identity differs from the requested launch")
        descriptors = load_index(
            directory, self.target, expected_contract=identity["contract"]
        )
        return descriptors, directory / "package"

    def load(self, key_count, bytes_per_key, execution):
        with self._lock:
            identity = self._identity(key_count, bytes_per_key, execution)
            return self._load(self._directory(identity), identity)

    def get(self, key_count, bytes_per_key, execution):
        with self._lock:
            identity = self._identity(key_count, bytes_per_key, execution)
            destination = self._directory(identity)
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".random-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entries"
                    staging.mkdir()
                    build_packages(
                        self.root, self.target, staging, contract=identity["contract"]
                    )
                    if self._identity(key_count, bytes_per_key, execution) != identity:
                        raise ValueError(
                            "Random translation inputs changed during packaging"
                        )
                    index = json.loads((staging / "index.json").read_text())
                    index["cacheIdentity"] = identity
                    write_json(staging / "index.json", index)
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            return self._load(destination, identity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("metal", "opengl", "directx"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)
    build_packages(args.mlx_root.resolve(), args.target, args.output_dir.resolve())
    load_index(args.output_dir, args.target)
