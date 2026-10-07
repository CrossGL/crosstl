"""Pinned integer remainder through reflected packages and native source controls."""

import hashlib
import itertools
import json
import os
import random
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

from crosstl.project import load_project_config, translate_project
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
from tests.runtime_helpers import _prepare_native_package, _validate
from tests.test_translator.test_boolean_buffer_runtime import (
    _bound_values,
)
from tests.test_translator.test_boolean_buffer_runtime import (
    _request as _dispatch_request,
)
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import _package
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_INTEGER_REMAINDER"
TYPES = {
    "bool": ("bool_", "bool", "?", 1),
    "int8": ("int8", "char", "b", 8),
    "uint8": ("uint8", "uchar", "B", 8),
    "int16": ("int16", "short", "h", 16),
    "uint16": ("uint16", "ushort", "H", 16),
    "int32": ("int32", "int", "i", 32),
    "int64": ("int64", "long", "q", 64),
}


def _entry(dtype):
    return "vv_Remainder" + TYPES[dtype][0]


def _range(dtype):
    bits = TYPES[dtype][3]
    return (
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        if dtype.startswith("int")
        else (0, (1 << bits) - 1)
    )


def _defined(a, b, dtype):
    return b != 0 and not (
        dtype in {"int32", "int64"} and a == _range(dtype)[0] and b == -1
    )


def _expected(a, b, dtype):
    assert _defined(a, b, dtype)
    result = a % b
    return bool(result) if dtype == "bool" else result


def _pairs(dtype):
    if dtype == "bool":
        return [(False, True), (True, True)]
    low, high = _range(dtype)
    if TYPES[dtype][3] == 8:
        candidates = itertools.product(range(low, high + 1), repeat=2)
    else:
        edges = {low, low + 1, high, high - 1, 0, 1, 2, 3, 7, 127, 128, 255}
        edges.update(
            value for value in (-1, -2, -3, -7, -127, -128, -255) if low <= value
        )
        for bit in range(8, TYPES[dtype][3] - 1):
            edges.update((1 << bit, (1 << bit) - 1))
        rng = random.Random(947 + TYPES[dtype][3])
        candidates = itertools.chain(
            itertools.product(sorted(edges), repeat=2),
            ((rng.randint(low, high), rng.randint(low, high)) for _ in range(4096)),
        )
    return [(a, b) for a, b in candidates if _defined(a, b, dtype)]


def _guard(dtype):
    return True if dtype == "bool" else 27


def _payload(descriptor, name, values):
    binding = next(
        binding
        for binding in descriptor["bindings"]
        if binding["scalarLayout"].get("memberName", binding["name"]) == name
    )
    kind = binding["scalarLayout"]["elementType"]
    return {
        "dtype": kind,
        "shape": [len(values)],
        "values": [bool(value) if kind == "bool" else int(value) for value in values],
    }


def _request(descriptor, package, dtype, target, pairs, expected):
    assert descriptor["target"] == target
    inputs = {
        name: _payload(descriptor, name, values)
        for name, values in (
            ("a", [a for a, _ in pairs]),
            ("b", [b for _, b in pairs]),
            ("c", [_guard(dtype)] * len(expected)),
        )
    }
    outputs = {"c": _payload(descriptor, "c", expected)}
    (constant,) = (
        binding
        for binding in descriptor["bindings"]
        if binding["kind"] == "constant-buffer"
    )
    size = constant["scalarLayout"].get("memberName", constant["name"])
    inputs[size] = _payload(descriptor, size, [len(pairs)])
    request = _dispatch_request(descriptor, package, inputs, outputs, len(pairs))
    return request, inputs, _bound_values(descriptor, outputs)


def _check_values(actual, expected):
    assert len(actual) == len(expected)
    mismatches = [
        {"index": index, "expected": wanted, "actual": got}
        for index, (got, wanted) in enumerate(zip(actual, expected))
        if type(got) is not type(wanted) or got != wanted
    ]
    assert not mismatches, mismatches[:20]


@pytest.mark.parametrize("dtype", TYPES)
def test_integer_remainder_domain_excludes_undefined_division(dtype):
    pairs = _pairs(dtype)
    assert 0 < len(pairs) <= 65535
    assert all(_defined(a, b, dtype) for a, b in pairs)
    assert pairs == _pairs(dtype)
    low, high = _range(dtype)
    assert all(low <= a <= high and low <= b <= high for a, b in pairs)
    if dtype in {"int8", "uint8"}:
        assert len(set(pairs)) == 256 * 255
    if dtype.startswith("int"):
        assert (-7, 3) in pairs and (7, -3) in pairs
        assert _expected(-7, 3, dtype) == 2
        assert _expected(7, -3, dtype) == -2
        overflow_pair = (_range(dtype)[0], -1)
        assert (overflow_pair in pairs) == (dtype in {"int8", "int16"})


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_integer_remainder_payloads_match_reflected_storage(tmp_path, dtype, target):
    typename = TYPES[dtype][1]
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void {_entry(dtype)}(const device {typename}* a [[buffer(0)]],
                           const device {typename}* b [[buffer(1)]],
                           device {typename}* c [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {{
    if (i < size) c[i] = a[i];
}}
"""
    _, descriptor, package = _package(
        tmp_path, target, typename, (1, 1, 1), source=source, software_subgroups=False
    )
    pairs = [(False, True)] if dtype == "bool" else [(_range(dtype)[0], 1)]
    expected = [_expected(*pairs[0], dtype)] + [_guard(dtype)] * 8
    request, inputs, outputs = _request(
        descriptor, package, dtype, target, pairs, expected
    )
    assert not request.execution_plan.diagnostics
    by_name = {value.name: value for value in request.fixture.inputs}
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"])
        if name not in {"a", "b", "c"}:
            continue
        value = by_name[binding["name"]]
        assert value.dtype == layout["elementType"]
        assert value.encoding is None
        assert list(value.values) == inputs[name]["values"]
        assert layout["elementSizeBytes"] == layout["elementStrideBytes"]
        if target == "metal":
            assert value.dtype == dtype
            assert layout["elementSizeBytes"] == max(1, TYPES[dtype][3] // 8)
        elif dtype == "bool":
            assert value.dtype == "uint32" and layout["elementSizeBytes"] == 4
        elif (
            dtype in {"int8", "uint8"}
            or target == "opengl"
            and dtype in {"int16", "uint16"}
        ):
            assert value.dtype == ("int32" if dtype.startswith("int") else "uint32")
            assert layout["elementSizeBytes"] == 4
        else:
            assert value.dtype == dtype
            assert layout["elementSizeBytes"] == TYPES[dtype][3] // 8
    assert len(outputs) == 1


@pytest.mark.parametrize(
    "actual,expected",
    [([0], [False]), ([False], [0]), ([1], [0]), ([0, 0], [0, 27]), ([0], [0, 27])],
)
def test_integer_comparison_rejects_value_type_and_guard_changes(actual, expected):
    with pytest.raises(AssertionError):
        _check_values(actual, expected)


def _original_metal(work, runner, library, dtype, pairs, expected):
    buffers = []
    for index, values in enumerate(
        (
            [a for a, _ in pairs],
            [b for _, b in pairs],
            [_guard(dtype)] * len(expected),
            [len(pairs)],
        )
    ):
        path = work / f"original-input-{index}.bin"
        code = "I" if index == 3 else TYPES[dtype][2]
        path.write_bytes(struct.pack(f"<{len(values)}{code}", *values))
        buffers.append(str(path))
    request = work / "original-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": buffers,
                "workgroupCount": [len(pairs), 1, 1],
                "workgroupSize": [1, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = work / "original-readback"
    _run([runner, library, _entry(dtype), request, output], work, "original-execute")
    actual = list(
        struct.unpack(
            f"<{len(expected)}{TYPES[dtype][2]}", (output / "buffer-2.bin").read_bytes()
        )
    )
    assert (output / "buffer-2.bin").read_bytes() == struct.pack(
        f"<{len(expected)}{TYPES[dtype][2]}", *expected
    )
    _check_values(actual, expected)
    for index in (0, 1, 3):
        assert (output / f"buffer-{index}.bin").read_bytes() == Path(
            buffers[index]
        ).read_bytes()


@pytest.mark.parametrize("dtype", TYPES)
def test_current_integer_remainder_native_parity(
    tmp_path, dtype, binary_metal_reference
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for pinned native integer remainder")
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
    runner, library = binary_metal_reference(root, target)
    entry = _entry(dtype)
    pairs = _pairs(dtype)
    expected = [_expected(a, b, dtype) for a, b in pairs] + [_guard(dtype)] * 8
    workload = next(w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == entry)
    with tempfile.TemporaryDirectory(
        prefix=".current-integer-remainder-", dir=root
    ) as directory:
        work = Path(directory)
        try:
            config_path = work / "crosstl.toml"
            config_path.write_text(_project_config(workload), encoding="utf-8")
            report = translate_project(
                load_project_config(root, config_path),
                targets=(target,),
                output_dir=work.name + "/out",
                format_output=False,
            )
            report.write_json(work / "report.json")
            data = report.to_json()
            assert (
                data["summary"]["translatedCount"] == 1
                and data["summary"]["failedCount"] == 0
            ), data["diagnostics"]
            assert not data["diagnostics"]
            (artifact,) = data["artifacts"]
            descriptor, package = _prepare_native_package(report, work)
            source = package / descriptor["artifact"]["packagePath"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest()
                == artifact["generatedHash"]["value"]
            )
            _validate(source, work, target)
            request, inputs, outputs = _request(
                descriptor, package, dtype, target, pairs, expected
            )
            assert not request.execution_plan.diagnostics
            (work / "values.json").write_text(
                json.dumps(
                    {"pairs": pairs, "inputs": inputs, "outputs": outputs}, indent=2
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
            assert result.outputs.keys() == outputs.keys()
            for name, wanted in outputs.items():
                actual = result.outputs[name]
                assert {k: v for k, v in actual.items() if k != "values"} == {
                    k: v for k, v in wanted.items() if k != "values"
                }
                _check_values(actual["values"], wanted["values"])
            if target == "metal":
                _original_metal(work, runner, library, dtype, pairs, expected)
            (work / "evidence.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "sourceSha256": MLX_BINARY_SHA256,
                        "entryPoint": entry,
                        "target": target,
                        "dtype": dtype,
                        "artifactSha256": artifact["generatedHash"]["value"],
                        "pairCount": len(pairs),
                        "guardCount": 8,
                        "comparison": "exact integer values",
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
            shutil.copytree(work, tmp_path / dtype, dirs_exist_ok=True)


def test_ci_requires_integer_remainder_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_integer_remainder.py"
    assert workflow.count(path) == 1
    assert path in step and "pytest -q -n auto" in step
    assert "--timeout-seconds 900" in step
