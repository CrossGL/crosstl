"""Execute unchanged pinned MLX unary entries with an explicit FMA profile."""

import json
import math
import os
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
from tests.test_translator.test_mlx_current_complex_power import MLX_COMMIT
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/unary.metal"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_FMA"


def wire_value(value):
    if math.isfinite(value):
        return value
    return "nan" if math.isnan(value) else "+infinity" if value > 0 else "-infinity"


def _f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


@pytest.mark.parametrize(
    "filename", ["mlx-project-porting.yml", "mlx-portable-host.yml"]
)
def test_ci_requires_current_unary_execution(filename):
    workflow = (
        Path(__file__).resolve().parents[2] / ".github/workflows" / filename
    ).read_text()
    step = workflow.split("      - name: Validate pinned unary fused arithmetic\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "CROSTL_MLX_CURRENT_ROOT:" in step
    assert "CROSTL_MLX_CURRENT_TARGET" in step
    assert "tests/test_translator/test_mlx_current_unary_fma.py" in step
    assert "--timeout-seconds 300" in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step


@pytest.mark.parametrize("operation", ["Erf", "Expm1"])
def test_current_mlx_unary_fma_native_loader(tmp_path, operation):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned unary execution")
    values = (
        [-5, 0, 0.5, 1, 2, 10]
        if operation == "Erf"
        else [-88, -87, 0, 0.5, -0.5, 5, 87, 88, 89, 90]
    )
    values += [index / 64 for index in range(-256, 257)]
    values += [-0.0, _f32(2**-126), _f32(-(2**-126))]
    values = [_f32(value) for value in values]
    reference = math.erf if operation == "Erf" else math.expm1
    expected = [_f32(reference(value)) for value in values]
    rtol, atol = (1e-5, 1e-8) if operation == "Erf" else (1e-3, 1e-4)
    _run_current_unary(
        tmp_path, operation, values, expected, rtol, atol, "metal_fma_float"
    )


def _run_current_unary(tmp_path, operation, values, expected, rtol, atol, helper):
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
    entry = f"v_{operation}float32float32"
    with tempfile.TemporaryDirectory(
        prefix=".current-unary-fma-", dir=root
    ) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(
            f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{work.name}/out"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [1, 1, 1]
[project.source_options.metal]
binary32_fma_profile = "rne-flush"
""",
            encoding="utf-8",
        )
        try:
            report = translate_project(
                load_project_config(root, config), format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
            assert payload["project"]["sourceOptions"]["metal"] == {
                "binary32_fma_profile": "rne-flush"
            }
            assert len(payload["artifacts"]) == 1
            artifact = payload["artifacts"][0]
            assert artifact["entryPoint"]["source"] == entry
            assert helper in (root / artifact["path"]).read_text()
            manifest = build_runtime_artifact_manifest(work / "report.json")
            assert manifest["success"], manifest
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = work / "package"
            assert build_runtime_package(work / "artifacts.json", package)["success"]
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            assert loader["success"], loader
            assert len(loader["loadUnits"]) == 1
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=loader["loadUnits"][0]["id"]
            )
            (work / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
            inputs, outputs = {}, {}
            matched = set()
            output_name = None
            for binding in descriptor["bindings"]:
                layout = binding["scalarLayout"]
                member = layout.get("memberName", binding["name"])
                if target == "directx":
                    member = member.removeprefix(entry + "_")
                member = {"in_": "in", "out_": "out"}.get(member, member)
                assert member in {"in", "out", "size"} and member not in matched
                matched.add(member)
                size = member == "size"
                assert layout["elementType"] == ("uint32" if size else "float32")
                assert layout["elementStrideBytes"] == 4
                data = [len(values)] if size else values if member == "in" else expected
                value = {
                    "dtype": "uint32" if size else "float32",
                    "shape": [len(data)],
                    "values": [wire_value(v) for v in data],
                }
                if member == "out":
                    # Loader output declarations require finite values. Compare
                    # every readback, including overflow to infinity, below.
                    outputs[binding["name"]] = {**value, "values": [0.0] * len(data)}
                    inputs[binding["name"]] = {**value, "values": [-1234.0] * len(data)}
                    output_name = binding["name"]
                else:
                    inputs[binding["name"]] = value
            assert matched == {"in", "out", "size"}
            request = build_native_loader_dispatch_request(
                descriptor,
                package,
                inputs,
                outputs,
                {"workgroupCount": [len(values), 1, 1], "workgroupSize": [1, 1, 1]},
                expected_target=target,
            )
            assert not request.execution_plan.diagnostics
            executor = _executor(target)
            availability = executor.is_available(request)
            assert availability.available, availability.reason
            result = executor.run(request)
            assert result.status == "ok"
            output = result.outputs[output_name]
            assert output["dtype"] == "float32" and output["shape"] == [len(values)]
            actual = [float(value) for value in output["values"]]
            assert len(actual) == len(expected)
            cases = [
                {
                    "input": wire_value(x),
                    "expected": wire_value(want),
                    "actual": wire_value(got),
                    "matched": (
                        got == want
                        or (math.isnan(got) and math.isnan(want))
                        or (
                            math.isfinite(got)
                            and math.isfinite(want)
                            and abs(got - want) <= atol + rtol * abs(want)
                        )
                    ),
                }
                for x, want, got in zip(values, expected, actual)
            ]
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": revision,
                        "entry": entry,
                        "target": target,
                        "profile": "rne-flush",
                        "rtol": rtol,
                        "atol": atol,
                        "cases": cases,
                        "runtime": result.details,
                    },
                    indent=2,
                    allow_nan=False,
                )
            )
            assert all(case["matched"] for case in cases), [
                c for c in cases if not c["matched"]
            ]
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
    return actual
