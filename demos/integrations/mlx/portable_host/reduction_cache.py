"""Cache whole-array reductions with checked metadata and launch contracts."""

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
from demos.integrations.mlx.portable_host.reduction_layout import stage
from demos.integrations.mlx.portable_host.reduction_packages import ALL_ENTRIES, SOURCE

SOURCE_SHA256 = "f1e410ab635eaa940ec195461069cbb013ab88ab0d1f8d5dd8790d30b32c454a"


def verify_events(events, root, directory, index):
    from demos.integrations.mlx.portable_host.reduction_packages import variant_keys
    from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

    cache = ReductionPackageCache(root, Path(directory) / "variants", index["target"])
    allowed = variant_keys(index)
    for event in events:
        execution = {name: event[name] for name in ("workgroupCount", "workgroupSize")}
        entry, count = event["entry"], event["threads"]
        contract = dispatch_contract(entry, count, execution)
        if f'w{execution["workgroupSize"][0]}/{entry}' not in allowed or event.get(
            "reductionMetadata"
        ) != {"in_size": count, "row_size": contract["rowSize"]}:
            raise ValueError("Reduction trace differs from its declared dispatch")
        descriptor, package = cache.load(entry, count, execution)
        if event.get("packageRoot") != str(package):
            raise ValueError("Reduction cache package root differs")
        for binding in descriptor["bindings"]:
            condition = binding.get("provenance", {}).get("metadataPrecondition")
            if condition is not None:
                packet = event["inputs"][binding["name"]]
                if (
                    packet.get("dtype") != "uint64"
                    or packet.get("shape") != [1]
                    or packet.get("values") != [condition["minimum"]]
                    or type(packet["values"][0]) is not int
                ):
                    raise ValueError("Reduction trace metadata upload differs")
        verify_artifacts(
            [event],
            package.parent,
            {"target": index["target"], "descriptors": {entry: descriptor}},
            variants=False,
        )


def dispatch_contract(entry, count, execution):
    if not isinstance(entry, str) or entry not in ALL_ENTRIES:
        raise ValueError("Unknown whole-array reduction entry")
    plan = stage(count, ALL_ENTRIES[entry])
    launch = {key: plan[key] for key in ("workgroupCount", "workgroupSize")}
    if (
        execution != launch
        or not isinstance(execution, dict)
        or any(
            type(value) is not int for values in execution.values() for value in values
        )
    ):
        raise ValueError("Reduction cache launch differs from the upstream plan")
    return {
        "entry": entry,
        "count": count,
        "rowSize": plan["rowSize"],
        "execution": launch,
    }


def _region(contract):
    launch = contract["execution"]
    grid = [
        count * width
        for count, width in zip(launch["workgroupCount"], launch["workgroupSize"])
    ]
    (region,) = plan_dispatch_regions(grid, launch["workgroupSize"])
    return region.to_json()


def _conditions(contract):
    return [
        {
            "dtype": "uint64",
            "minimum": value,
            "maximum": value,
            "sourceResource": {
                "schemaVersion": 1,
                "backend": "metal",
                "entryPoint": contract["entry"],
                "parameter": name,
                "parameterIndex": index,
            },
        }
        for index, name, value in (
            (2, "in_size", contract["count"]),
            (3, "row_size", contract["rowSize"]),
        )
    ]


class ReductionPackageCache:
    def __init__(self, root, directory, target):
        if target not in {"metal", "opengl", "directx"}:
            raise ValueError("Unsupported reduction target")
        self.root, self.directory = Path(root).resolve(), Path(directory).resolve()
        self.target = target
        self._lock = threading.RLock()

    def _require_source(self):
        require_revision(self.root)
        if (
            hashlib.sha256((self.root / SOURCE).read_bytes()).hexdigest()
            != SOURCE_SHA256
        ):
            raise ValueError("Pinned reduction source hash differs")
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
            raise ValueError("Reduction translation requires unchanged pinned kernels")

    def _identity(self, entry, count, execution):
        contract = dispatch_contract(entry, count, execution)
        self._require_source()
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

    def load(self, entry, count, execution):
        with self._lock:
            identity = self._identity(entry, count, execution)
            return self._load(self._directory(identity), identity)

    def get(self, entry, count, execution):
        with self._lock:
            identity = self._identity(entry, count, execution)
            destination = self._directory(identity)
            if not destination.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=".reduction-package-", dir=self.directory
                ) as temporary:
                    staging = Path(temporary) / "entry"
                    staging.mkdir()
                    self._build(staging, identity)
                    if self._identity(entry, count, execution) != identity:
                        raise ValueError("Reduction inputs changed during translation")
                    try:
                        staging.rename(destination)
                    except OSError:
                        if not destination.is_dir():
                            raise
            return self._load(destination, identity)

    def _load(self, directory, identity):
        record = json.loads((directory / "index.json").read_text())
        if record.get("identity") != identity or directory != self._directory(identity):
            raise ValueError(
                "Reduction cache identity differs from the requested dispatch"
            )
        loader = build_runtime_loader_manifest(
            directory / "package/runtime-package.json"
        )
        if not loader["success"] or len(loader["loadUnits"]) != 1:
            raise ValueError("Reduction runtime package is incomplete")
        descriptor = build_native_loader_abi_descriptor(loader)
        contract = identity["contract"]
        conditions = [
            binding["provenance"]["metadataPrecondition"]
            for binding in descriptor["bindings"]
            if "metadataPrecondition" in binding.get("provenance", {})
        ]
        if (
            descriptor != record.get("descriptor")
            or descriptor["target"] != self.target
            or descriptor["stage"] != "compute"
            or descriptor["source"]["path"] != SOURCE
            or descriptor["source"]["backend"] != "metal"
            or descriptor["source"]["hash"] != {
                "algorithm": "sha256",
                "value": SOURCE_SHA256,
            }
            or loader["loadUnits"][0]["entryPoint"]["source"] != contract["entry"]
            or descriptor["entryPoint"]["name"]
            != {"opengl": "main", "directx": "CSMain"}.get(
                self.target, contract["entry"]
            )
            or sorted(conditions, key=lambda item: item["sourceResource"]["parameter"])
            != sorted(
                _conditions(contract),
                key=lambda item: item["sourceResource"]["parameter"],
            )
            or descriptor.get("provenance", {}).get("dispatchRegion")
            != (None if self.target == "metal" else _region(contract))
        ):
            raise ValueError("Reduction descriptor differs from its checked package")
        return descriptor, directory / "package"

    def _build(self, output, identity):
        contract = identity["contract"]
        options = {}
        if self.target != "metal":
            options.update(
                software_subgroup_width=32, dispatch_region=_region(contract)
            )
        if self.target == "directx":
            options["relative_wave_shuffle_out_of_range"] = "self"
        with tempfile.TemporaryDirectory(
            prefix=".reduction-translate-", dir=self.root
        ) as temporary:
            work = Path(temporary)
            report = translate_project(
                ProjectConfig(
                    root=self.root,
                    source_roots=("mlx/backend/metal/kernels",),
                    include_patterns=(SOURCE,),
                    include_dirs=(".",),
                    targets=(self.target,),
                    output_dir=f"{work.name}/out",
                    entry_points={SOURCE: (contract["entry"],)},
                    workgroup_size_rules={
                        SOURCE: tuple(contract["execution"]["workgroupSize"])
                    },
                    metadata_preconditions=[
                        dict(
                            source=SOURCE,
                            entryPoint=contract["entry"],
                            parameter=name,
                            minimum=value,
                            maximum=value,
                        )
                        for name, value in (
                            ("in_size", contract["count"]),
                            ("row_size", contract["rowSize"]),
                        )
                    ],
                    source_options={
                        "metal": {
                            "preserve_resource_origins": True,
                            "max_template_specializations": 128,
                            "max_template_materialization_work": 8192,
                            "target_options": {self.target: options},
                        }
                    },
                ),
                format_output=False,
            )
            report.write_json(work / "report.json")
            shutil.copytree(work, output / "translation")
            payload = report.to_json()
            if payload["summary"]["failedCount"] or len(payload["artifacts"]) != 1:
                raise ValueError(
                    f"Reduction translation failed: {payload['diagnostics']}"
                )
            manifest = build_runtime_artifact_manifest(work / "report.json")
            if not manifest["success"]:
                raise ValueError(
                    f"Reduction manifest failed: {manifest['diagnostics']}"
                )
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = build_runtime_package(work / "artifacts.json", output / "package")
            if not package["success"]:
                raise ValueError(
                    f"Reduction packaging failed: {package['diagnostics']}"
                )
            loader = build_runtime_loader_manifest(
                output / "package/runtime-package.json"
            )
            if not loader["success"] or len(loader["loadUnits"]) != 1:
                raise ValueError("Reduction loading failed")
            descriptor = build_native_loader_abi_descriptor(loader)
            (output / "index.json").write_text(
                json.dumps({"identity": identity, "descriptor": descriptor}, indent=2)
                + "\n"
            )
