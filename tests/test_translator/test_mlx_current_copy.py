"""Execute pinned copy kernels with independent strided-coordinate references."""

import hashlib
import itertools
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
from tests.test_translator.test_mlx_current_arg_reduce import _metal_library, _run
from tests.test_translator.test_mlx_current_binary_shapes import (
    _validate_generated_artifact,
)
from tests.test_translator.test_mlx_current_complex_power import MLX_COMMIT, ROOT
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/copy.metal"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COPY"
GUARD = struct.unpack("<f", struct.pack("<I", 0x6A15BEEF))[0]
FORMATS = {"float32": "f", "uint32": "I", "int32": "i", "int64": "q"}
CASES = {
    "vector": ("v_copy", [17], [1]),
    "scalar": ("s_copy", [17], [0]),
    "stride": ("g1_copy", [17], [3]),
    "broadcast-1d": ("g1_copy", [17], [0]),
    "transpose": ("g2_copy", [5, 7], [1, 5]),
    "broadcast-row": ("g2_copy", [5, 7], [0, 2]),
    "permute-3d": ("g3_copy", [3, 5, 7], [5, 1, 15]),
    "strided-4d": ("gn2_copy", [2, 3, 4, 5], [100, 7, 21, 1]),
    "broadcast-4d": ("gn2_copy", [2, 3, 4, 5], [0, 0, 0, 2]),
    "reverse-1d": ("gg1_dynamic_copy", [17], [-2]),
    "reverse-4d": ("ggn2_dynamic_copy", [2, 3, 4, 5], [100, 7, -21, -1]),
}
DYNAMIC_LAYOUTS = {
    "reverse-1d": (38, 3, [2]),
    "reverse-4d": (80, 7, [151, 43, 9, -1]),
}


def _bytes(dtype, values):
    return struct.pack("<" + FORMATS[dtype] * len(values), *values)


def _workload(case):
    prefix, shape, strides = CASES[case]
    count = math.prod(shape)
    source_offset, destination_offset, destination_strides = DYNAMIC_LAYOUTS.get(
        case, (0, 0, None)
    )
    coordinates = list(itertools.product(*(range(n) for n in shape)))
    offsets = [
        source_offset + sum(index * stride for index, stride in zip(point, strides))
        for point in coordinates
    ]
    destinations = (
        [
            destination_offset + sum(i * s for i, s in zip(point, destination_strides))
            for point in coordinates
        ]
        if destination_strides
        else list(range(count))
    )
    assert min(offsets) >= 0 and min(destinations) >= 0
    assert len(set(destinations)) == count
    source = [((i * 37) % 1009 - 504) / 8 for i in range(max(offsets) + 1)]
    source[offsets[0]] = -0.0
    expected = [GUARD] * (max(destinations) + 1 + 32)
    for src, dst in zip(offsets, destinations):
        expected[dst] = source[src]
    grid = (
        [count, 1, 1]
        if len(shape) == 1
        else [
            (shape[-1] + 1) // 2 if "n2" in prefix else shape[-1],
            shape[-2],
            math.prod(shape[:-2]),
        ]
    )
    constants = (
        {"size": ("uint32", [count], 2)}
        if prefix in {"v_copy", "s_copy"}
        else (
            {"src_stride": ("int64", strides, 3)}
            if prefix in {"g1_copy", "gg1_dynamic_copy"}
            else {"src_strides": ("int64", strides, 3)}
        )
    )
    if "n2" in prefix:
        constants.update(
            {
                "src_shape": ("int32", shape, 2),
                "ndim": ("int32", [len(shape)], 5),
            }
        )
    if destination_strides:
        name = "dst_stride" if len(shape) == 1 else "dst_strides"
        constants.update(
            {
                name: ("int64", destination_strides, 4),
                "src_offset": ("int64", [source_offset], 6),
                "dst_offset": ("int64", [destination_offset], 7),
            }
        )
    return {
        "entry": prefix + "float32float32",
        "shape": shape,
        "strides": strides,
        "source": source,
        "offsets": offsets,
        "destinations": destinations,
        "sourceOffset": source_offset,
        "destinationOffset": destination_offset,
        "destinationStrides": destination_strides,
        "expected": expected,
        "grid": grid,
        "constants": constants,
    }


def _check(actual, expected):
    assert len(actual) == len(expected)
    assert _bytes("float32", actual) == _bytes("float32", expected)
    assert _bytes("float32", actual[-32:]) == struct.pack("<32I", *([0x6A15BEEF] * 32))


def _index_contracts(target, workload):
    if target != "opengl" or not workload["entry"].startswith("gg1_dynamic_copy"):
        return ""
    # Only final addresses are bounded; signed stride intermediates may be negative.
    return "".join(
        f"""[[project.index_range_assertions]]
source = "{SOURCE}"
expression = "{expression}"
minimum = {min(indices)}
maximum = {max(indices)}
"""
        for expression, indices in (
            ("src_idx + src_offset", workload["offsets"]),
            ("dst_idx + dst_offset", workload["destinations"]),
        )
    )


@pytest.mark.parametrize("case", CASES)
def test_copy_workload_covers_logical_coordinates(case):
    workload = _workload(case)
    assert len(workload["expected"]) == max(workload["destinations"]) + 1 + 32
    assert len(workload["offsets"]) == math.prod(workload["shape"])
    assert min(workload["offsets"]) >= 0
    assert max(workload["offsets"]) < len(workload["source"])
    assert all(value > 0 for value in workload["grid"])
    first = workload["destinations"][0]
    assert workload["expected"][first] == 0
    assert math.copysign(1, workload["expected"][first]) == -1
    if case.endswith("4d"):
        assert workload["grid"] == [3, 4, 6]
        assert workload["shape"][-1] % 2 == 1


@pytest.mark.parametrize("fault", ["value", "missing", "guard", "zero-sign"])
def test_copy_readback_rejects_corruption(fault):
    expected = _workload("transpose")["expected"]
    actual = list(expected)
    if fault == "value":
        actual[1] += 1
    elif fault == "missing":
        actual.pop()
    elif fault == "guard":
        actual[-1] = 0
    else:
        actual[0] = 0.0
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_copy_index_contracts_cover_only_final_bounded_addresses():
    workload = _workload("reverse-1d")
    contract = _index_contracts("opengl", workload)
    assert "minimum = 6\nmaximum = 38" in contract
    assert "minimum = 3\nmaximum = 35" in contract
    assert contract.count("[[project.index_range_assertions]]") == 2
    assert _index_contracts("directx", workload) == ""
    assert _index_contracts("opengl", _workload("stride")) == ""


@pytest.mark.parametrize(
    "filename",
    ["mlx-portable-host.yml", "mlx-metal-host.yml", "mlx-project-porting.yml"],
)
def test_ci_requires_native_copy_parity(filename):
    workflow = (ROOT / ".github/workflows" / filename).read_text()
    step = workflow.split("      - name: Validate pinned copy layouts\n", 1)[1].split(
        "      - name:", 1
    )[0]
    assert "if:" not in step and "continue-on-error" not in step
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT",
        "CROSTL_MLX_CURRENT_TARGET",
        "--timeout-seconds 600",
        "pytest -q -n auto",
        "--basetemp=",
        "--junitxml=",
        "tests/test_translator/test_mlx_current_copy.py",
    ):
        assert required in step
    assert (
        workflow.count('      - "tests/test_translator/test_mlx_current_copy.py"') == 2
    )
    assert "if: always()" in workflow
    assert len({item[0] for item in CASES.values()}) == 8


@pytest.fixture(scope="module")
def current_copy_source(tmp_path_factory):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for the pinned native copy gate")
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
    reference = None
    if target == "metal":
        work = tmp_path_factory.mktemp("copy-metal-reference")
        runner = work / "readback"
        _run(
            [
                "xcrun",
                "swiftc",
                ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
                "-o",
                runner,
            ],
            work,
            "build-runner",
        )
        reference = runner, _metal_library(
            root / SOURCE, work / "upstream.metallib", root, upstream=True
        )
    return root, target, reference


@pytest.fixture(scope="module")
def copy_executor(current_copy_source):
    _, target, _ = current_copy_source
    executor = _executor(target)
    try:
        yield executor
    finally:
        if target == "metal":
            executor.runtime_adapter.runtime.close()


def _original_metal(work, workload, reference):
    runner, library = reference
    bindings = {
        0: ("float32", workload["source"]),
        1: ("float32", [GUARD] * len(workload["expected"])),
    }
    bindings.update(
        {
            index: (dtype, values)
            for dtype, values, index in workload["constants"].values()
        }
    )
    files = []
    for index in range(max(bindings) + 1):
        path = work / f"original-input-{index}.bin"
        path.write_bytes(_bytes(*bindings.get(index, ("uint32", [0]))))
        files.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": workload["grid"],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            },
            indent=2,
        )
    )
    output = work / "original-readback"
    result = json.loads(
        _run(
            [runner, library, workload["entry"], request, output],
            work,
            "original-execute",
        )
    )
    raw = (output / "buffer-1.bin").read_bytes()
    actual = [value for (value,) in struct.iter_unpack("<f", raw)]
    _check(actual, workload["expected"])
    for index, filename in enumerate(files):
        if index != 1:
            assert (output / f"buffer-{index}.bin").read_bytes() == Path(
                filename
            ).read_bytes()
    return {
        "device": result,
        "moduleSha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("case", CASES)
def test_current_copy_native_parity(current_copy_source, copy_executor, tmp_path, case):
    root, target, reference = current_copy_source
    workload = _workload(case)
    entry = workload["entry"]
    with tempfile.TemporaryDirectory(prefix=".current-copy-", dir=root) as directory:
        work = Path(directory)
        config = work / "crosstl.toml"
        config.write_text(f"""[project]
source_roots = ["mlx/backend/metal/kernels"]
include = ["{SOURCE}"]
include_dirs = ["."]
targets = ["{target}"]
output_dir = "{work.name}/out"
[project.entry_points]
"{SOURCE}" = "{entry}"
[project.entry_workgroup_size_rules."{SOURCE}"]
"{entry}" = [1, 1, 1]
{_index_contracts(target, workload)}
""")
        try:
            report = translate_project(
                load_project_config(root, config), format_output=False
            )
            report.write_json(work / "report.json")
            payload = report.to_json()
            assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
            (artifact,) = payload["artifacts"]
            assert artifact["entryPoint"]["source"] == entry
            manifest = build_runtime_artifact_manifest(work / "report.json")
            assert manifest["success"], manifest
            (work / "artifacts.json").write_text(json.dumps(manifest, indent=2))
            package = work / "package"
            assert build_runtime_package(work / "artifacts.json", package)["success"]
            loader = build_runtime_loader_manifest(package / "runtime-package.json")
            assert loader["success"] and len(loader["loadUnits"]) == 1
            descriptor = build_native_loader_abi_descriptor(
                loader, load_unit_id=loader["loadUnits"][0]["id"]
            )
            (work / "descriptor.json").write_text(json.dumps(descriptor, indent=2))
            source = package / descriptor["artifact"]["packagePath"]
            if target != "metal":
                _validate_generated_artifact(source, work, target)
            else:
                _metal_library(source, work / "translated.metallib", root)
            data = {
                "src": ("float32", workload["source"]),
                "dst": ("float32", [GUARD] * len(workload["expected"])),
            }
            data.update(
                {
                    name: (dtype, values)
                    for name, (dtype, values, _) in workload["constants"].items()
                }
            )
            inputs, outputs, matched = {}, {}, set()
            for binding in descriptor["bindings"]:
                if (
                    binding.get("provenance", {}).get("kind")
                    == "generated-execution-input"
                ):
                    continue
                layout = binding["scalarLayout"]
                name = layout.get("memberName", binding["name"]).removeprefix(
                    entry + "_"
                )
                assert name in data and name not in matched
                matched.add(name)
                dtype, values = data[name]
                assert layout["elementType"] == dtype
                assert layout["elementStrideBytes"] == struct.calcsize(FORMATS[dtype])
                value = {"dtype": dtype, "shape": [len(values)], "values": values}
                inputs[binding["name"]] = value
                if name == "dst":
                    output_name = binding["name"]
                    outputs[output_name] = {**value, "values": [0.0] * len(values)}
            assert matched == set(data)
            (work / "workload.json").write_text(json.dumps(workload, indent=2))
            (work / "values.json").write_text(
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
            availability = copy_executor.is_available(request)
            assert availability.available, availability.reason
            result = copy_executor.run(request)
            (work / "result.json").write_text(
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
            output = result.outputs[output_name]
            assert output["dtype"] == "float32" and output["shape"] == [
                len(workload["expected"])
            ]
            (work / "readback.bin").write_bytes(_bytes("float32", output["values"]))
            (work / "expected.bin").write_bytes(_bytes("float32", workload["expected"]))
            _check(output["values"], workload["expected"])
            evidence = {
                "commit": MLX_COMMIT,
                "target": target,
                "case": case,
                "entry": entry,
                "values": len(workload["expected"]),
                "sourceSha256": (
                    hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
                ),
                "artifactSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            if reference:
                evidence["original"] = _original_metal(work, workload, reference)
            (work / "evidence.json").write_text(json.dumps(evidence, indent=2))
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
