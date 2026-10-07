"""Native package and unchanged-source checks shared by floating binary demos."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.kernels.test_binary_complete_opengl import (
    BINARY_OPENGL_WORKLOADS,
    MLX_BINARY_SHA256,
    MLX_BINARY_SOURCE,
    _project_config,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from tests.runtime_helpers import _validate
from tests.test_translator.test_native_loader_dispatch_integration import _executor


@dataclass(frozen=True)
class BinaryCase:
    dtype: str
    operation: str
    pairs: Sequence[Tuple[int, int]]
    expected: Sequence[int]
    provenance: Mapping[str, str]
    comparison: str

    @property
    def entry(self):
        return f"vv_{self.operation}{self.dtype}"


def _batch_workloads(cases):
    assert cases, "Native binary checks must execute at least one case"
    entries = [case.entry for case in cases]
    assert len(set(entries)) == len(entries), "Duplicate binary entry"
    assert all(
        case.provenance == cases[0].provenance for case in cases
    ), "A binary batch must use one source profile"
    workloads = {workload.entry_point: workload for workload in BINARY_OPENGL_WORKLOADS}
    return [workloads[entry] for entry in entries]


def _binary_load_units(loader, entries, target):
    assert loader["success"], loader
    units = loader["loadUnits"]
    assert len(units) == len(entries)
    assert len({unit["id"] for unit in units}) == len(entries)
    assert all(
        unit["target"] == target and unit["source"] == MLX_BINARY_SOURCE
        for unit in units
    )
    by_entry = {unit["entryPoint"]["source"]: unit for unit in units}
    assert set(by_entry) == set(entries)
    assert all(unit["validation"]["loadReady"] for unit in units)
    return by_entry


def _original_metal(work, runner, library, case, guard, compare):
    buffers = []
    for index, words in enumerate(
        (
            [a for a, _ in case.pairs],
            [b for _, b in case.pairs],
            [guard] * len(case.expected),
            [len(case.pairs)],
        )
    ):
        path = work / f"original-input-{index}.bin"
        code = "I" if case.dtype == "float32" or index == 3 else "H"
        path.write_bytes(struct.pack(f"<{len(words)}{code}", *words))
        buffers.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": [len(case.pairs), 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / "original-readback"
    _run([runner, library, case.entry, request, output], work, "original-execute")
    code = "I" if case.dtype == "float32" else "H"
    actual = list(
        struct.unpack(
            f"<{len(case.expected)}{code}", (output / "buffer-2.bin").read_bytes()
        )
    )
    comparison = compare(actual, list(case.expected), case.dtype, "metal")
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == Path(
            buffers[index]
        ).read_bytes()
    return comparison


def run_binary_cases(
    tmp_path, require_env, cases, *, request_for, guard_for, compare_for, source_control
):
    if os.environ.get(require_env) != "1":
        pytest.skip(f"set {require_env}=1 for pinned native binary checks")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"directx", "opengl", "metal"}
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "mlx/backend/metal/kernels",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert (
        hashlib.sha256((root / MLX_BINARY_SOURCE).read_bytes()).hexdigest()
        == MLX_BINARY_SHA256
    )
    runner, library = source_control(root, target)
    cases = tuple(cases)
    workloads = _batch_workloads(cases)
    entries = [case.entry for case in cases]
    for case in cases:
        assert len(case.expected) == len(case.pairs) + 8
        assert list(case.expected[-8:]) == [guard_for(case.dtype)] * 8
    with tempfile.TemporaryDirectory(prefix=".current-binary-", dir=root) as directory:
        batch_work = Path(directory)
        try:
            config_path = batch_work / "crosstl.toml"
            config_path.write_text(
                _project_config(workloads[0], entry_points=entries), encoding="utf-8"
            )
            report = translate_project(
                load_project_config(root, config_path),
                targets=(target,),
                output_dir=batch_work.name + "/out",
                format_output=False,
            )
            report.write_json(batch_work / "report.json")
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == len(cases)
                and data["summary"]["failedCount"] == 0
            ), data["diagnostics"]
            assert not data["diagnostics"]
            assert len(data["artifacts"]) == len(cases)
            artifacts = {
                artifact["entryPoint"]["source"]: artifact
                for artifact in data["artifacts"]
            }
            assert set(artifacts) == set(entries)
            manifest = build_runtime_artifact_manifest(batch_work / "report.json")
            assert manifest["success"], manifest
            (batch_work / "artifacts.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            package = batch_work / "package"
            assert build_runtime_package(batch_work / "artifacts.json", package)[
                "success"
            ]
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            (batch_work / "loader.json").write_text(
                json.dumps(loader, indent=2), encoding="utf-8"
            )
            units = _binary_load_units(loader, entries, target)
            for case in cases:
                compare = compare_for(case)
                work = batch_work / case.entry
                work.mkdir()
                artifact = artifacts[case.entry]
                for name in (
                    "binary32ComparisonProfile",
                    "binary32AdditiveProfile",
                    "binary32DivisionProfile",
                    "binary32RemainderProfile",
                    "binary16RemainderProfile",
                ):
                    assert artifact["provenance"].get(name) == case.provenance.get(name)
                descriptor = build_native_loader_abi_descriptor(
                    loader, load_unit_id=units[case.entry]["id"]
                )
                (work / "descriptor.json").write_text(
                    json.dumps(descriptor, indent=2), encoding="utf-8"
                )
                source = package / descriptor["artifact"]["packagePath"]
                assert (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    == artifact["generatedHash"]["value"]
                )
                _validate(source, work, target)
                request, inputs, outputs = request_for(
                    descriptor, package, case.dtype, target, case.pairs, case.expected
                )
                assert not request.execution_plan.diagnostics
                (work / "values.json").write_text(
                    json.dumps(
                        {"pairs": case.pairs, "inputs": inputs, "outputs": outputs},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                executor = _executor(target)
                try:
                    availability = executor.is_available(request)
                    assert availability.available, availability.reason
                    result = executor.run(request)
                finally:
                    close = getattr(executor.runtime_adapter.runtime, "close", None)
                    if close:
                        close()
                (work / "result.json").write_text(
                    json.dumps(
                        {
                            "status": result.status,
                            "outputs": result.outputs,
                            "details": result.details,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                assert result.status == "ok"
                (actual,), (wanted,) = result.outputs.values(), outputs.values()
                assert {k: v for k, v in actual.items() if k != "values"} == {
                    k: v for k, v in wanted.items() if k != "values"
                }
                comparison = compare(
                    actual["values"], wanted["values"], case.dtype, target
                )
                source_comparison = None
                if target == "metal":
                    source_comparison = _original_metal(
                        work, runner, library, case, guard_for(case.dtype), compare
                    )
                (work / "evidence.json").write_text(
                    json.dumps(
                        {
                            "commit": MLX_COMMIT,
                            "sourceSha256": MLX_BINARY_SHA256,
                            "entryPoint": case.entry,
                            "target": target,
                            "dtype": case.dtype,
                            "operation": case.operation,
                            "artifactSha256": artifact["generatedHash"]["value"],
                            "comparisonProfile": case.provenance.get(
                                "binary32ComparisonProfile"
                            ),
                            "additiveProfile": case.provenance.get(
                                "binary32AdditiveProfile"
                            ),
                            "divisionProfile": case.provenance.get(
                                "binary32DivisionProfile"
                            ),
                            "pairCount": len(case.pairs),
                            "guardCount": 8,
                            "comparison": case.comparison,
                            "comparisonDetails": comparison,
                            "sourceComparisonDetails": source_comparison,
                            "sourceControlInputsVerified": target == "metal",
                            "wholeFamilyParity": False,
                            "physicalDriverUploadBytes": False,
                            "originalLibrarySha256": (
                                hashlib.sha256(library.read_bytes()).hexdigest()
                                if library
                                else None
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        finally:
            shutil.copytree(batch_work, tmp_path / "evidence", dirs_exist_ok=True)
