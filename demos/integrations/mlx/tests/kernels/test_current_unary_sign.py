"""Native readbacks for unchanged MLX Sign and scalar comparison promotion."""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import replace
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
from demos.integrations.mlx.tests.kernels.test_current_complex_power import MLX_COMMIT
from demos.integrations.mlx.tests.kernels.test_current_unary_fma import (
    _run_current_unary,
)
from demos.integrations.mlx.tests.kernels.test_unary_complete_opengl import (
    MLX_UNARY_SHA256,
    MLX_UNARY_SOURCE,
    UNARY_OPENGL_WORKLOADS,
    _project_config,
)
from tests.ci_helpers import assert_paths_covered
from tests.runtime_helpers import _validate
from tests.test_translator.test_metal_boolean_promotion import _inputs
from tests.test_translator.test_metal_precise_trig import _bits, _float
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_UNARY_SIGN"
UNSIGNED_SHAPES = ("v", "vn", "v2", "gn1", "gn4large")
GUARD = 0x35555555


def _unsigned_workload(shape):
    values = [0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    values += [((i * 2654435761) & 0xFFFFFFFF) if i % 17 else 0 for i in range(1024)]
    if shape.startswith("gn"):
        indices = [
            z * 90 + y * 2 + x * 11
            for z in range(3)
            for y in range(5)
            for x in range(7)
        ]
        constants = {
            "in_shape": ("int32", [3, 5, 7]),
            "in_strides": ("int64", [90, 2, 11]),
            "ndim": ("int32", [3]),
        }
        grid = [7 if shape == "gn1" else 2, 5, 3]
    else:
        indices = list(range(len(values)))
        constants = {"size": ("int64" if shape == "v2" else "uint32", [len(values)])}
        grid = {"v": [1031, 1, 1], "vn": [516, 1, 1], "v2": [12, 43, 1]}[shape]
    return {
        "source": values + [GUARD] * 8,
        "indices": indices,
        "expected": [int(values[index] != 0) for index in indices] + [GUARD] * 8,
        "constants": constants,
        "grid": grid,
    }


@pytest.mark.parametrize("shape", UNSIGNED_SHAPES)
def test_unsigned_sign_workload_covers_boundaries_and_guards(shape):
    workload = _unsigned_workload(shape)
    indices = workload["indices"]
    assert len(indices) == (105 if shape.startswith("gn") else 1031)
    assert len(set(indices)) == len(indices)
    assert min(indices) == 0 and max(indices) < len(workload["source"]) - 8
    assert {0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF} <= set(workload["source"])
    assert set(workload["expected"][:-8]) == {0, 1}
    assert workload["expected"][-8:] == workload["source"][-8:] == [GUARD] * 8
    if shape == "vn":
        assert workload["grid"][0] * 2 == len(indices) + 1
    elif shape == "v2":
        assert workload["grid"][:2] == [12, 43]
    elif shape.startswith("gn"):
        assert workload["constants"]["in_strides"] == ("int64", [90, 2, 11])


def _unsigned_original(work, entry, workload, runner, library):
    values = [
        ("uint32", workload["source"]),
        ("uint32", [GUARD] * len(workload["expected"])),
        *workload["constants"].values(),
    ]
    payloads, paths = [], []
    for index, (dtype, words) in enumerate(values):
        code = {"uint32": "I", "int32": "i", "int64": "q"}[dtype]
        payload = struct.pack(f"<{len(words)}{code}", *words)
        path = work / f"original-input-{index}.bin"
        path.write_bytes(payload)
        payloads.append(payload)
        paths.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": paths,
                "workgroupCount": workload["grid"],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        )
    )
    output = work / "original-readback"
    _run([runner, library, entry, request, output], work, "original-execute")
    assert (output / "buffer-1.bin").read_bytes() == struct.pack(
        f"<{len(workload['expected'])}I", *workload["expected"]
    )
    for index, payload in enumerate(payloads):
        if index != 1:
            assert (output / f"buffer-{index}.bin").read_bytes() == payload


def test_current_unsigned_sign_shapes_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned Sign execution")
    root = Path(os.environ["CROSTL_MLX_CURRENT_ROOT"]).resolve()
    target = os.environ["CROSTL_MLX_CURRENT_TARGET"]
    assert target in {"metal", "opengl", "directx"}
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        == MLX_COMMIT
    )
    integrity = [
        "git",
        "-C",
        str(root),
        "diff",
        "--exit-code",
        "HEAD",
        "--",
        "mlx/backend/metal/kernels",
    ]
    subprocess.run(integrity, check=True, capture_output=True, timeout=30)
    assert (
        hashlib.sha256((root / MLX_UNARY_SOURCE).read_bytes()).hexdigest()
        == MLX_UNARY_SHA256
    )
    workloads = [
        w
        for w in UNARY_OPENGL_WORKLOADS
        if w.operator_type == "Sign" and w.input_type == "uint32_t"
    ]
    entries = [w.entry_point for w in workloads]
    assert len(entries) == 5 and {w.shape for w in workloads} == set(UNSIGNED_SHAPES)
    executor = _executor(target)
    with tempfile.TemporaryDirectory(
        prefix=".current-unsigned-sign-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(_project_config(workloads[0]))
            config = replace(
                load_project_config(root, config_path),
                entry_points={MLX_UNARY_SOURCE: entries},
                entry_workgroup_size_rules={MLX_UNARY_SOURCE: {"*": (1, 1, 1)}},
            )
            report = translate_project(
                config,
                targets=(target,),
                output_dir=work.name + "/out",
                format_output=False,
            )
            report.write_json(work / "report.json")
            data = report.to_json()
            assert not data["diagnostics"] and data["summary"]["translatedCount"] == 5
            manifest = build_runtime_artifact_manifest(work / "report.json")
            assert manifest["success"], manifest
            (work / "artifacts.json").write_text(json.dumps(manifest))
            package = work / "package"
            assert build_runtime_package(work / "artifacts.json", package)["success"]
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            assert loader["success"] and len(loader["loadUnits"]) == 5
            units = {unit["entryPoint"]["source"]: unit for unit in loader["loadUnits"]}
            assert set(units) == set(entries)
            runner = library = None
            if target == "metal":
                runner = work / "readback"
                fixture = (
                    Path(__file__).resolve().parents[5]
                    / "tests/fixtures/runtime_verification/metal_raw_buffers.swift"
                )
                _run(
                    ["xcrun", "swiftc", fixture, "-o", runner],
                    work,
                    "build-reference-runner",
                )
                library = _metal_library(
                    root / MLX_UNARY_SOURCE,
                    work / "original.metallib",
                    root,
                    upstream=True,
                )
            for selected in workloads:
                entry = selected.entry_point
                case = work / entry
                case.mkdir()
                workload = _unsigned_workload(selected.shape)
                descriptor = build_native_loader_abi_descriptor(
                    loader, load_unit_id=units[entry]["id"]
                )
                (case / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
                _validate(package / descriptor["artifact"]["packagePath"], case, target)
                values = {
                    "in": ("uint32", workload["source"]),
                    "out": ("uint32", [GUARD] * len(workload["expected"])),
                    **workload["constants"],
                }
                inputs, outputs, seen = {}, {}, set()
                for binding in descriptor["bindings"]:
                    if (
                        binding.get("provenance", {}).get("kind")
                        == "generated-execution-input"
                    ):
                        continue
                    layout = binding["scalarLayout"]
                    member = layout.get("memberName", binding["name"])
                    if member.startswith(entry + "_"):
                        member = member[len(entry) + 1 :]
                    member = {"in_": "in", "out_": "out"}.get(member, member)
                    assert member in values and member not in seen, binding
                    seen.add(member)
                    dtype, words = values[member]
                    assert layout["elementType"] == dtype
                    assert layout["elementStrideBytes"] == (
                        8 if dtype == "int64" else 4
                    )
                    inputs[binding["name"]] = {
                        "dtype": dtype,
                        "shape": [len(words)],
                        "values": words,
                    }
                    if member == "out":
                        outputs[binding["name"]] = {
                            **inputs[binding["name"]],
                            "values": workload["expected"],
                        }
                assert seen == set(values)
                (case / "workload.json").write_text(json.dumps(workload, indent=2))
                (case / "values.json").write_text(
                    json.dumps({"inputs": inputs, "outputs": outputs}, indent=2)
                )
                request = build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    inputs,
                    outputs,
                    {"workgroupCount": workload["grid"], "workgroupSize": [1, 1, 1]},
                    expected_target=target,
                )
                assert not request.execution_plan.diagnostics
                available = executor.is_available(request)
                assert available.available, available.reason
                result = executor.run(request)
                (case / "result.json").write_text(
                    json.dumps(
                        {
                            "status": result.status,
                            "outputs": result.outputs,
                            "details": result.details,
                        },
                        indent=2,
                    )
                )
                assert result.status == "ok"
                (actual,) = result.outputs.values()
                assert actual["dtype"] == "uint32" and actual["shape"] == [
                    len(workload["expected"])
                ]
                assert actual["values"] == workload["expected"]
                if runner:
                    _unsigned_original(case, entry, workload, runner, library)
        finally:
            close = getattr(executor.runtime_adapter.runtime, "close", None)
            if close:
                close()
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
            subprocess.run(integrity, check=True, capture_output=True, timeout=30)


def test_current_mlx_sign_native_loader(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned Sign execution")
    inputs = [_float(word) for word in _inputs()]
    expected = [float((value > 0) - (value < 0)) for value in inputs]
    actual = _run_current_unary(
        tmp_path,
        "Sign",
        inputs,
        expected,
        0.0,
        0.0,
        "operator_call",
    )
    assert [_bits(value) for value in actual] == [_bits(value) for value in expected]


@pytest.mark.parametrize("job_id", ["mlx-metal-porting", "portable-host", "metal-host"])
def test_ci_requires_boolean_promotion(job_id):
    from tools import ci_coverage

    text = (
        Path(__file__).resolve().parents[5]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = (
        ci_coverage.workflow_job_text(text, job_id)
        .split("      - name: Validate pinned comparison arithmetic\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert 'CROSTL_REQUIRE_METAL_BOOLEAN_PROMOTION: "1"' in step
    assert "--timeout-seconds 180" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for module in (
        "demos/integrations/mlx/tests/kernels/test_current_unary_sign.py",
        "tests/test_translator/test_metal_boolean_promotion.py",
    ):
        assert module in step
        for event in ("push", "pull_request"):
            assert_paths_covered(
                ci_coverage.workflow_event_path_filters(text, event), module
            )
    if job_id == "mlx-metal-porting":
        upload = (
            ci_coverage.workflow_job_text(text, job_id)
            .split("      - name: Upload comparison arithmetic evidence\n", 1)[1]
            .split("      - name:", 1)[0]
        )
        assert "if: always()" in upload
        assert "include-hidden-files: true" in upload
        assert "if-no-files-found: error" in upload
