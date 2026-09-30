"""Native complex-power parity across every pinned MLX binary entry shape."""

import copy
import itertools
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import (
    NativeLoaderDispatchError,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.project.directx_toolchain import dxc_compiler_arguments_for_source
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from tests.test_translator.test_mlx_current_arg_reduce import _metal_library, _run
from tests.test_translator.test_mlx_current_complex_power import (
    MLX_COMMIT,
    ROOT,
    SOURCE,
    _check_values,
    _reference,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_BINARY_SHAPES"
SHAPES = (
    "ss",
    "sv",
    "vs",
    "vv",
    "sv2",
    "vs2",
    "vv2",
    "g1",
    "g1large",
    "g2",
    "g2large",
    "g3",
    "g3large",
    "gn2",
    "gn4large",
)


def _configuration(work, target, entry):
    text = f"""[project]
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
max_template_specializations = 64
max_template_materialization_work = 4096
"""
    if target == "opengl":
        for expression in (
            "offset + i",
            "a_idx",
            "b_idx",
            "out_idx",
            "out_idx++",
            "idx.x",
            "idx.y",
        ):
            text += f"""[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = 0
maximum = 511
"""
    return text


def _workload(shape):
    constants = {}
    if shape == "ss":
        indices, grid = [(0, 0)], [1, 1, 1]
    elif shape.startswith(("sv", "vs", "vv")):
        count = 15 if shape.endswith("2") else 7
        indices = [
            (0 if shape.startswith("sv") else i, 0 if shape.startswith("vs") else i)
            for i in range(count)
        ]
        grid = [5, 3, 1] if shape.endswith("2") else [count, 1, 1]
        constants["size"] = ("int64" if shape.endswith("2") else "uint32", [count])
    elif shape.startswith("g1"):
        indices, grid = [(i * 2, i * 3) for i in range(7)], [7, 1, 1]
        constants = {"a_stride": ("int64", [2]), "b_stride": ("int64", [3])}
    else:
        if shape.startswith("g2"):
            dims, a_strides, b_strides = [3, 5], [11, 2], [0, 1]
            grid = [5, 3, 1]
        elif shape.startswith("g3"):
            dims, a_strides, b_strides = [2, 3, 5], [41, 11, 2], [23, 0, 1]
            grid = [5, 3, 2]
        elif shape in {"gn2", "gn4large"}:
            dims = [2, 2, 3, 5]
            a_strides, b_strides = [101, 43, 11, 2], [0, 41, 7, 1]
            work_per_thread = 2 if shape == "gn2" else 4
            grid = [math.ceil(5 / work_per_thread), 3, 4]
            constants["shape"] = ("int32", dims)
        else:
            raise ValueError(f"Unknown binary shape: {shape}")
        # Enumerate logical coordinates independently of the kernel's index helpers.
        indices = [
            (
                sum(x * stride for x, stride in zip(coords, a_strides)),
                sum(x * stride for x, stride in zip(coords, b_strides)),
            )
            for coords in itertools.product(*(range(size) for size in dims))
        ]
        constants.update(
            {"a_strides": ("int64", a_strides), "b_strides": ("int64", b_strides)}
        )
        if shape.startswith("gn"):
            constants["ndim"] = ("int32", [len(dims)])
    generator = random.Random(1951)

    def values(count):
        return [
            complex(
                *struct.unpack(
                    "<2f",
                    struct.pack(
                        "<2f", generator.uniform(-2, 2), generator.uniform(-2, 2)
                    ),
                )
            )
            for _ in range(count)
        ]

    a = values(max(pair[0] for pair in indices) + 1)
    b = values(max(pair[1] for pair in indices) + 1)
    return {
        "a": a,
        "b": b,
        "pairs": [(a[i], b[j]) for i, j in indices],
        "indices": indices,
        "constants": constants,
        "grid": grid,
    }


def _complex_value(values):
    return {
        "dtype": "float32",
        "shape": [len(values), 2],
        "values": [part for value in values for part in (value.real, value.imag)],
    }


def test_binary_shape_fixture_covers_exact_family():
    assert len(SHAPES) == len(set(SHAPES)) == 15
    assert sum(len(_workload(shape)["pairs"]) for shape in SHAPES) == 291


@pytest.mark.parametrize("shape", SHAPES)
def test_binary_shape_fixture_bounds_and_complete_output(shape):
    workload = _workload(shape)
    assert all(0 <= index < 512 for pair in workload["indices"] for index in pair)
    assert len(workload["pairs"]) < 512
    assert len(workload["grid"]) == 3 and all(size > 0 for size in workload["grid"])
    assert len(set(workload["indices"])) == len(workload["pairs"])
    assert workload == _workload(shape)
    if shape.startswith("gn"):
        assert list(workload["constants"]) == [
            "shape",
            "a_strides",
            "b_strides",
            "ndim",
        ]
        assert len(workload["pairs"]) == 60
        assert workload["grid"] == ([3, 3, 4] if shape == "gn2" else [2, 3, 4])


@pytest.fixture(scope="module")
def current_binary_source():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native binary-shape gate")
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
    register_default_sources()
    discovery = SOURCE_REGISTRY.get("metal").discover_entry_points(
        (root / SOURCE).read_text(encoding="utf-8"),
        file_path=str(root / SOURCE),
        include_paths=[str(root)],
    )
    assert not discovery.diagnostics
    assert {
        entry.name
        for entry in discovery.entries
        if entry.name.endswith("_Powercomplex64")
    } == {f"{shape}_Powercomplex64" for shape in SHAPES}
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_binary_source, tmp_path_factory):
    root, target = current_binary_source
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("binary-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_buffer_readback.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(root / SOURCE, work / "upstream.metallib", root)


def _validate_generated_artifact(path, work, target):
    if target == "directx":
        module = work / "kernel.dxil"
        commands = [
            (
                "dxc",
                [
                    "dxc",
                    *dxc_compiler_arguments_for_source(
                        path.read_text(encoding="utf-8")
                    ),
                    "-WX",
                    "-T",
                    "cs_6_2",
                    "-E",
                    "CSMain",
                    path,
                    "-Fo",
                    module,
                ],
            )
        ]
    else:
        module = work / "kernel.spv"
        commands = [
            (
                "glslang",
                [
                    "glslangValidator",
                    "--target-env",
                    "opengl",
                    "--target-env",
                    "spirv1.3",
                    "-S",
                    "comp",
                    path,
                    "-o",
                    module,
                ],
            ),
            ("spirv-val", ["spirv-val", "--target-env", "spv1.3", module]),
        ]
    for label, command in commands:
        _run(command, work, label)
        log = json.loads((work / f"{label}.json").read_text(encoding="utf-8"))
        assert "warning:" not in (log["stdout"] + log["stderr"]).lower(), log
    assert module.stat().st_size > 0


def _native_request(report, work, target, entry, workload):
    report.write_json(work / "report.json")
    manifest = build_runtime_artifact_manifest(work / "report.json")
    assert manifest["success"], manifest
    (work / "artifacts.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
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
    names = (
        ("a", "b", "c") if target == "directx" else ("aBuffer", "bBuffer", "cBuffer")
    )
    pairs = workload["pairs"]
    inputs = {
        names[0]: _complex_value(workload["a"]),
        names[1]: _complex_value(workload["b"]),
        names[2]: _complex_value([-1234 - 1234j] * len(pairs)),
    }
    binding_names = {binding["name"] for binding in descriptor["bindings"]}
    suffix = "Constants" if target == "directx" else "Args"
    for name, (dtype, values) in workload["constants"].items():
        candidate = name if target == "directx" else f"{name}Buffer"
        actual = candidate if candidate in binding_names else f"{entry}_{name}_{suffix}"
        assert actual in binding_names
        inputs[actual] = {"dtype": dtype, "shape": [len(values)], "values": values}
    outputs = {names[2]: _complex_value([_reference(*pair) for pair in pairs])}
    if entry.startswith(("g2_", "g2large_", "g3_", "g3large_")):
        rejections = []
        for stride_name in ("a_strides", "b_strides"):
            binding_name = (
                stride_name if target == "directx" else f"{stride_name}Buffer"
            )
            binding = next(
                b for b in descriptor["bindings"] if b["name"] == binding_name
            )
            original = inputs[binding_name]
            required_bytes = len(original["values"]) * 8
            assert binding["scalarLayout"]["minimumBindingSizeBytes"] == required_bytes
            truncated = copy.deepcopy(inputs)
            truncated[binding_name]["values"] = original["values"][:-1]
            truncated[binding_name]["shape"] = [len(original["values"]) - 1]
            with pytest.raises(NativeLoaderDispatchError) as caught:
                build_native_loader_dispatch_request(
                    descriptor,
                    package,
                    truncated,
                    outputs,
                    workload["grid"],
                    expected_target=target,
                )
            diagnostics = caught.value.details["diagnostics"]
            assert any(
                diagnostic["code"].endswith("resource-view-too-small")
                for diagnostic in diagnostics
            ), diagnostics
            rejections.append({"binding": binding_name, "diagnostics": diagnostics})
        (work / "undersized-stride-preflight.json").write_text(
            json.dumps(rejections, indent=2), encoding="utf-8"
        )
    request = build_native_loader_dispatch_request(
        descriptor, package, inputs, outputs, workload["grid"], expected_target=target
    )
    assert request.execution_plan.diagnostics == ()
    (work / "values.json").write_text(
        json.dumps(
            {"inputs": inputs, "outputs": outputs, "grid": workload["grid"]}, indent=2
        ),
        encoding="utf-8",
    )
    executor = _executor(target)
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    assert result.status == "ok"
    output = result.outputs[names[2]]
    assert output["dtype"] == "float32" and output["shape"] == [len(pairs), 2]
    return {"translated": _check_values(output["values"], pairs)}


def _metal_parity(root, work, artifact, entry, workload, reference):
    runner, original = reference
    pairs = workload["pairs"]
    values = [_complex_value(workload[name]) for name in ("a", "b")]
    values.append(_complex_value([-1234 - 1234j] * len(pairs)))
    values.extend(
        {"dtype": dtype, "values": value}
        for dtype, value in workload["constants"].values()
    )
    formats = {"float32": "f", "int32": "i", "uint32": "I", "int64": "q"}
    buffers = [
        list(
            struct.pack(
                "<" + formats[value["dtype"]] * len(value["values"]), *value["values"]
            )
        )
        for value in values
    ]
    request = work / "metal-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "grid": workload["grid"],
                "outputIndex": 2,
                "outputCount": len(pairs) * 2,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    generated = _metal_library(
        root / artifact["path"], work / "translated.metallib", root
    )
    evidence = {}
    for label, library, kernel in (
        ("source", original, entry),
        ("translated", generated, artifact["entryPoint"]["target"]),
    ):
        result = json.loads(
            _run([runner, library, kernel, request], work, f"{label}-execute")
        )
        evidence[label] = _check_values(result["values"], pairs)
    return evidence


@pytest.mark.parametrize("shape", SHAPES)
def test_current_mlx_complex_power_shape_native_parity(
    current_binary_source, metal_reference, tmp_path, shape
):
    root, target = current_binary_source
    entry = f"{shape}_Powercomplex64"
    workload = _workload(shape)
    with tempfile.TemporaryDirectory(
        prefix=".current-binary-shapes-", dir=root
    ) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(_configuration(work, target, entry), encoding="utf-8")
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
            (artifact,) = payload["artifacts"]
            assert artifact["entryPoint"]["source"] == entry
            if target != "metal":
                _validate_generated_artifact(root / artifact["path"], work, target)
            evidence = (
                _metal_parity(root, work, artifact, entry, workload, metal_reference)
                if target == "metal"
                else _native_request(report, work, target, entry, workload)
            )
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
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
