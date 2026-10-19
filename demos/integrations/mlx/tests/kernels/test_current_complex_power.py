"""Execute pinned MLX complex power through generated backend artifacts."""

import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor

ROOT = Path(__file__).resolve().parents[5]
MLX_COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
SOURCE = "mlx/backend/metal/kernels/binary.metal"
ENTRY = "g1_Powercomplex64"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COMPLEX_POWER"


def _cases():
    pairs = [
        (complex(3, 4), complex(0.5, 0)),
        (complex(-2, 3), complex(2, -1)),
        (0j, 0j),
        (0j, complex(-2, 1)),
        (complex(-4, 0), complex(0.5, 0)),
        (complex(-4, -0.0), complex(0.5, 0)),
    ]
    generator = random.Random(1950)
    for _ in range(250):
        values = struct.unpack(
            "<4f", struct.pack("<4f", *(generator.uniform(-3, 3) for _ in range(4)))
        )
        pairs.append((complex(*values[:2]), complex(*values[2:])))
    return pairs


def _config(work, target):
    return f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{work.name}/out"
[project.entry_points]
"{SOURCE}" = "{ENTRY}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{ENTRY}" = [1, 1, 1]
[project.source_options.metal]
max_template_specializations = 64
max_template_materialization_work = 4096
"""


def _reference(base, exponent):
    # MLX explicitly returns zero for finite powers of a zero complex base.
    return base**exponent if base != 0j else 0j


def _check_values(values, pairs):
    assert len(values) == 2 * len(pairs)
    cases = []
    for index, (base, exponent) in enumerate(pairs):
        expected = _reference(base, exponent)
        actual = complex(*values[index * 2 : index * 2 + 2])
        limit = 5e-5 * max(1.0, abs(expected))
        cases.append(
            {
                "base": [base.real, base.imag],
                "exponent": [exponent.real, exponent.imag],
                "expected": [expected.real, expected.imag],
                "actual": [actual.real, actual.imag],
                "absoluteError": abs(actual - expected),
                "limit": limit,
                "matched": abs(actual - expected) <= limit,
            }
        )
    return cases


def _runtime(report, work, target, pairs):
    report.write_json(work / "report.json")
    artifacts = build_runtime_artifact_manifest(work / "report.json")
    assert artifacts["success"], artifacts
    (work / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2), encoding="utf-8"
    )
    package = work / "package"
    assert build_runtime_package(work / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    (work / "descriptor.json").write_text(
        json.dumps(descriptor, indent=2), encoding="utf-8"
    )
    structs = [
        binding for binding in descriptor["bindings"] if binding["kind"] == "buffer"
    ]
    assert len(structs) == 3
    for binding in structs:
        layout = binding["scalarLayout"]
        assert layout["elementStrideBytes"] == 8
        assert layout["alignmentBytes"] == 4
        assert layout["structMembers"] == [
            {"name": "real", "physicalType": "float", "offsetBytes": 0},
            {"name": "imag", "physicalType": "float", "offsetBytes": 4},
        ]
    names = (
        ("a", "b", "c") if target == "directx" else ("aBuffer", "bBuffer", "cBuffer")
    )
    suffix = "Constants" if target == "directx" else "Args"
    inputs = {}
    for index in (0, 1):
        inputs[names[index]] = {
            "dtype": "float32",
            "shape": [len(pairs), 2],
            "values": [
                component
                for pair in pairs
                for component in (pair[index].real, pair[index].imag)
            ],
        }
    for name in ("a_stride", "b_stride"):
        inputs[f"{ENTRY}_{name}_{suffix}"] = {
            "dtype": "int64",
            "shape": [1],
            "values": [1],
        }
    # Initialize every output so a missed write cannot match a zero result.
    inputs[names[2]] = {
        "dtype": "float32",
        "shape": [len(pairs), 2],
        "values": [-1234.0] * (len(pairs) * 2),
    }
    expected_values = [_reference(*pair) for pair in pairs]
    outputs = {
        names[2]: {
            "dtype": "float32",
            "shape": [len(pairs), 2],
            "values": [
                component
                for value in expected_values
                for component in (value.real, value.imag)
            ],
        }
    }
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, [len(pairs), 1, 1], expected_target=target
    )
    assert request.execution_plan.diagnostics == ()
    executor = _executor(target)
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    assert result.status == "ok"
    output = result.outputs[names[2]]
    assert output["dtype"] == "float32"
    assert output["shape"] == [len(pairs), 2]
    return {"translated": _check_values(output["values"], pairs)}


def _metal(root, work, artifact, pairs):
    runner = work / "complex-power"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "demos/integrations/mlx/tests/fixtures/complex_power_metal.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    request = work / "inputs.json"
    request.write_text(
        json.dumps(
            {
                "left": [
                    component
                    for pair in pairs
                    for component in (pair[0].real, pair[0].imag)
                ],
                "right": [
                    component
                    for pair in pairs
                    for component in (pair[1].real, pair[1].imag)
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = {}
    for label, path, entry in (
        ("source", root / SOURCE, ENTRY),
        ("translated", root / artifact["path"], artifact["entryPoint"]["target"]),
    ):
        library = _metal_library(
            path, work / f"{label}.metallib", root, upstream=label == "source"
        )
        values = json.loads(
            _run([runner, library, entry, request], work, f"{label}-execute")
        )
        evidence[label] = _check_values(values["values"], pairs)
    return evidence


def test_current_mlx_complex_power_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native complex-power gate")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"directx", "opengl", "metal"}
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    assert revision == MLX_COMMIT
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
    with tempfile.TemporaryDirectory(
        prefix=".current-complex-power-", dir=root
    ) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(_config(work, target), encoding="utf-8")
        try:
            report = translate_project(
                load_project_config(root, config),
                format_output=False,
                validate=True,
                run_toolchains=False,
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
            assert len(payload["artifacts"]) == 1
            artifact = payload["artifacts"][0]
            assert artifact["entryPoint"]["source"] == ENTRY
            pairs = _cases()
            evidence = (
                _metal(root, work, artifact, pairs)
                if target == "metal"
                else _runtime(report, work, target, pairs)
            )
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": ENTRY,
                        "target": target,
                        "results": evidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            for label, cases in evidence.items():
                assert all(case["matched"] for case in cases), (
                    label,
                    [case for case in cases if not case["matched"]],
                )
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
