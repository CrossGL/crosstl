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
    build_native_loader_dispatch_request,
    load_project_config,
    translate_project,
)
from demos.integrations.mlx.portable_host import bfloat_storage
from demos.integrations.mlx.portable_host.runtime import boolean_values, physical_dtype
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_binary_shapes import (
    _validate_generated_artifact,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from tests.runtime_helpers import _prepare_native_package
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_half_copy_identity import _widen
from tests.test_translator.test_native_loader_dispatch_integration import _executor

SOURCE = "mlx/backend/metal/kernels/copy.metal"
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_COPY"
GUARD = struct.unpack("<f", struct.pack("<I", 0x6A15BEEF))[0]
FORMATS = {
    "bool": "?",
    "float32": "f",
    "uint16": "H",
    "uint32": "I",
    "int32": "i",
    "int64": "q",
}
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


@pytest.mark.parametrize("job", ["portable-host", "metal-host", "mlx-metal-porting"])
def test_ci_requires_native_copy_parity(job):
    from tests.ci_helpers import assert_workflow_triggers
    from tools import ci_coverage

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, job, "Validate pinned copy layouts"
    )
    assert "if:" not in step and "continue-on-error" not in step
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT",
        "CROSTL_MLX_CURRENT_TARGET",
        "--timeout-seconds 600",
        "pytest -q -n auto",
        "--basetemp=",
        "--junitxml=",
        "demos/integrations/mlx/tests/kernels/test_current_copy.py",
    ):
        assert required in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_copy.py"
    )
    upload_name, upload_path = {
        "portable-host": ("Retain native execution evidence", ".mlx-portable-host"),
        "metal-host": ("Retain native host execution evidence", ".mlx-metal-host"),
        "mlx-metal-porting": ("Upload pinned copy evidence", "mlx-copy-results"),
    }[job]
    upload = ci_coverage.workflow_job_step_section(workflow, job, upload_name)
    assert "if: always()" in upload
    assert "include-hidden-files: true" in upload
    assert upload_path in upload
    assert "if-no-files-found: error" in upload
    assert "retention-days: 14" in upload
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


def _original_metal(work, workload, reference, dtype="float32", output_dtype=None):
    runner, library = reference
    output_dtype = output_dtype or dtype
    bindings = {
        0: (dtype, workload["source"]),
        1: (
            output_dtype,
            workload.get("initial", [GUARD] * len(workload["expected"])),
        ),
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
    actual = [
        value for (value,) in struct.iter_unpack("<" + FORMATS[output_dtype], raw)
    ]
    if output_dtype == "float32":
        _check(actual, workload["expected"])
    else:
        assert raw == _bytes(output_dtype, workload["expected"])
    for index, filename in enumerate(files):
        if index != 1:
            assert (output / f"buffer-{index}.bin").read_bytes() == Path(
                filename
            ).read_bytes()
    return {
        "device": result,
        "moduleSha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def _copy_package(root, target, work, workload):
    entry = workload["entry"]
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
    report = translate_project(load_project_config(root, config), format_output=False)
    report.write_json(work / "report.json")
    payload = report.to_json()
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    assert artifact["entryPoint"]["source"] == entry
    descriptor, package = _prepare_native_package(report, work)
    source = package / descriptor["artifact"]["packagePath"]
    if target != "metal":
        _validate_generated_artifact(source, work, target)
    else:
        _metal_library(source, work / "translated.metallib", root)
    return descriptor, package, source


@pytest.mark.parametrize("case", CASES)
def test_current_copy_native_parity(current_copy_source, copy_executor, tmp_path, case):
    root, target, reference = current_copy_source
    workload = _workload(case)
    entry = workload["entry"]
    with tempfile.TemporaryDirectory(prefix=".current-copy-", dir=root) as directory:
        work = Path(directory)
        try:
            descriptor, package, source = _copy_package(root, target, work, workload)
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


def _half_workload(prefix, start):
    words = list(range(start, start + 32768))
    stride = 3 if prefix == "g1_copy" else 1
    source = [0xA55A] * (len(words) * stride)
    source[::stride] = words
    return {
        "entry": prefix + "float16float16",
        "source": source,
        "expected": words + [0x3555] * 32,
        "initial": [0x3555] * (len(words) + 32),
        "grid": [len(words), 1, 1],
        "constants": (
            {"src_stride": ("int64", [stride], 3)}
            if stride != 1
            else {"size": ("uint32", [len(words)], 2)}
        ),
    }


def _bfloat_boolean_workload(prefix, start):
    workload = _half_workload(prefix, start)
    words = list(range(start, start + 32768))
    expected = [word not in (0x0000, 0x8000) for word in words]
    guards = [index % 2 == 0 for index in range(32)]
    workload.update(
        entry=prefix + "bfloat16bool_",
        expected=expected + guards,
        initial=[not value for value in expected] + guards,
    )
    return workload


def test_bfloat_boolean_copy_covers_every_payload_and_preserves_guards():
    for prefix in ("v_copy", "g1_copy"):
        observed, converted = [], []
        for start in (0, 32768):
            workload = _bfloat_boolean_workload(prefix, start)
            stride = 3 if prefix == "g1_copy" else 1
            words = workload["source"][::stride]
            count = workload["grid"][0]
            assert count == len(words) == 32768
            assert workload["expected"][:count] == [
                word not in (0x0000, 0x8000) for word in words
            ]
            assert workload["initial"][:count] == [
                not value for value in workload["expected"][:count]
            ]
            assert workload["initial"][count:] == workload["expected"][count:]
            assert workload["expected"][count:] == [i % 2 == 0 for i in range(32)]
            observed.extend(words)
            converted.extend(workload["expected"][:count])
        assert observed == list(range(65536))
        assert [index for index, value in enumerate(converted) if not value] == [
            0x0000,
            0x8000,
        ]


@pytest.mark.parametrize("prefix", ("v_copy", "g1_copy"))
@pytest.mark.parametrize("start", (0, 32768))
def test_current_bfloat_boolean_copy_preserves_all_payloads(
    current_copy_source, copy_executor, tmp_path, prefix, start
):
    root, target, reference = current_copy_source
    workload = _bfloat_boolean_workload(prefix, start)
    with tempfile.TemporaryDirectory(
        prefix=".current-bfloat-boolean-copy-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            descriptor, package, source = _copy_package(root, target, work, workload)
            encoding = bfloat_storage.encoding(target)
            output_dtype = physical_dtype("bool_", target)
            inputs = {
                "src": {
                    "dtype": physical_dtype("bfloat16", target),
                    "shape": [len(workload["source"])],
                    "values": bfloat_storage.pack(workload["source"], target),
                    **({"encoding": encoding} if encoding else {}),
                },
                "dst": {
                    "dtype": output_dtype,
                    "shape": [len(workload["initial"])],
                    "values": [
                        bool(value) if output_dtype == "bool" else int(value)
                        for value in workload["initial"]
                    ],
                },
            }
            for name, (dtype, values, _) in workload["constants"].items():
                if target == "directx":
                    name = workload["entry"] + "_" + name
                inputs[name] = {
                    "dtype": dtype,
                    "shape": [len(values)],
                    "values": values,
                }
            outputs = {
                "dst": {
                    **inputs["dst"],
                    "values": [
                        bool(value) if output_dtype == "bool" else int(value)
                        for value in workload["expected"]
                    ],
                }
            }
            (work / "workload.json").write_text(json.dumps(workload, indent=2))
            (work / "values.json").write_text(
                json.dumps({"inputs": inputs, "outputs": outputs}, indent=2)
            )
            request = _request(
                descriptor, package, inputs, outputs, workload["grid"][0]
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
            assert result.outputs == _bound_values(descriptor, outputs)
            (actual,) = result.outputs.values()
            boolean_values(actual["values"], output_dtype)
            (work / "readback.bin").write_bytes(_bytes(output_dtype, actual["values"]))
            (work / "expected.bin").write_bytes(
                _bytes(output_dtype, outputs["dst"]["values"])
            )
            evidence = {
                "commit": MLX_COMMIT,
                "target": target,
                "entry": workload["entry"],
                "firstPayload": start,
                "payloadCount": 32768,
                "guardCount": 32,
                "sourceSha256": (
                    hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
                ),
                "artifactSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            if reference:
                evidence["original"] = _original_metal(
                    work, workload, reference, "uint16", "bool"
                )
            (work / "evidence.json").write_text(json.dumps(evidence, indent=2))
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)


def _half_payload(target, words):
    return {
        "dtype": "float32" if target == "opengl" else "float16",
        "encoding": "ieee754-binary32" if target == "opengl" else "ieee754-binary16",
        "shape": [len(words)],
        "values": (
            [_widen(word) for word in words] if target == "opengl" else list(words)
        ),
    }


def test_half_copy_workloads_cover_every_payload_and_preserve_guards():
    for prefix in ("v_copy", "g1_copy"):
        observed = []
        for start in (0, 32768):
            workload = _half_workload(prefix, start)
            count = workload["grid"][0]
            stride = 3 if prefix == "g1_copy" else 1
            assert count <= 65535
            assert workload["source"][::stride] == workload["expected"][:count]
            assert workload["expected"][count:] == [0x3555] * 32
            assert workload["initial"] == [0x3555] * (count + 32)
            observed.extend(workload["expected"][:count])
        assert observed == list(range(65536))


@pytest.mark.parametrize("prefix", ("v_copy", "g1_copy"))
@pytest.mark.parametrize("start", (0, 32768))
def test_current_half_copy_preserves_all_payloads(
    current_copy_source, copy_executor, tmp_path, prefix, start
):
    root, target, reference = current_copy_source
    workload = _half_workload(prefix, start)
    with tempfile.TemporaryDirectory(
        prefix=".current-half-copy-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            descriptor, package, source = _copy_package(root, target, work, workload)
            inputs = {
                "src": _half_payload(target, workload["source"]),
                "dst": _half_payload(target, workload["initial"]),
            }
            for name, (dtype, values, _) in workload["constants"].items():
                if target == "directx":
                    name = workload["entry"] + "_" + name
                inputs[name] = {
                    "dtype": dtype,
                    "shape": [len(values)],
                    "values": values,
                }
            outputs = {"dst": _half_payload(target, workload["expected"])}
            (work / "workload.json").write_text(json.dumps(workload, indent=2))
            (work / "values.json").write_text(
                json.dumps({"inputs": inputs, "outputs": outputs}, indent=2)
            )
            request = _request(
                descriptor, package, inputs, outputs, workload["grid"][0]
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
            assert result.outputs == _bound_values(descriptor, outputs)
            dtype = "uint32" if target == "opengl" else "uint16"
            (actual,) = result.outputs.values()
            (work / "readback.bin").write_bytes(_bytes(dtype, actual["values"]))
            (work / "expected.bin").write_bytes(_bytes(dtype, outputs["dst"]["values"]))
            evidence = {
                "commit": MLX_COMMIT,
                "target": target,
                "entry": workload["entry"],
                "firstPayload": start,
                "payloadCount": 32768,
                "guardCount": 32,
                "sourceSha256": (
                    hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()
                ),
                "artifactSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            if reference:
                evidence["original"] = _original_metal(
                    work, workload, reference, "uint16"
                )
            (work / "evidence.json").write_text(json.dumps(evidence, indent=2))
        finally:
            shutil.copytree(work, tmp_path / "evidence", dirs_exist_ok=True)
