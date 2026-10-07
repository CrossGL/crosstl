"""Pinned minimum and maximum preserve selected operands through native packages."""

import itertools
import random
from pathlib import Path

import pytest

from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    NATIVE_DTYPE_BATCHES,
    BinaryCase,
    run_binary_cases,
)
from demos.integrations.mlx.tests.kernels.test_current_copy import _half_payload
from tests.test_translator.test_bfloat_buffer_runtime import _storage as _bfloat_payload
from tests.test_translator.test_boolean_buffer_runtime import (
    _bound_values,
)
from tests.test_translator.test_boolean_buffer_runtime import (
    _request as _dispatch_request,
)
from tests.test_translator.test_software_subgroup_product import _package
from tools import ci_coverage

ROOT = Path(__file__).resolve().parents[5]
REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_EXTREMA"
TYPES = {
    "float16": ("half", 16, 0x7C00),
    "bfloat16": ("bfloat", 16, 0x7F80),
    "float32": ("float", 32, 0x7F800000),
}
OPERATIONS = ("Minimum", "Maximum")


def _entry(dtype, operation):
    return f"vv_{operation}{dtype}"


def _pairs(dtype):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    fraction_bits = 10 if dtype == "float16" else 7 if dtype == "bfloat16" else 23
    one = (
        0x3C00 if dtype == "float16" else 0x3F80 if dtype == "bfloat16" else 0x3F800000
    )
    edges = [
        0,
        1,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        one - 1,
        one,
        one + 1,
        infinity - 1,
        infinity,
        infinity + 1,
        infinity | (1 << (fraction_bits - 1)),
        infinity | (1 << (fraction_bits - 1)) | 7,
    ]
    words = edges + [word | sign for word in edges]
    pairs = list(itertools.product(words, repeat=2))
    rng = random.Random(600 + bits)
    pairs.extend((rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(4096))
    return pairs


def _expected(a, b, dtype, operation):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    magnitude = sign - 1
    if a & magnitude > infinity:
        return a
    if b & magnitude > infinity:
        return b
    x, y = a, b
    if dtype != "float16":
        x, y = (word & sign if word & infinity == 0 else word for word in (x, y))
    x, y = (0 if word & magnitude == 0 else word for word in (x, y))
    mask = (1 << bits) - 1
    x, y = ((~word & mask) if word & sign else (word | sign) for word in (x, y))
    choose_a = x < y if operation == "Minimum" else x > y
    return a if choose_a else b


def _payload(dtype, target, words):
    if dtype == "float16":
        return _half_payload(target, words)
    if dtype == "bfloat16":
        return _bfloat_payload(target, words)
    return {
        "dtype": "float32",
        "encoding": "ieee754-binary32",
        "shape": [len(words)],
        "values": list(words),
    }


def _guard(dtype):
    return 0x35555555 if dtype == "float32" else 0x3555


def _check_words(actual, expected):
    assert len(actual) == len(expected)
    differences = [
        {"index": i, "expected": b, "actual": a}
        for i, (a, b) in enumerate(zip(actual, expected))
        if type(a) is not int or a != b
    ]
    assert not differences, differences[:20]


def _request(descriptor, package, dtype, target, pairs, expected):
    assert descriptor["target"] == target
    inputs = {
        name: _payload(dtype, target, words)
        for name, words in (
            ("a", [a for a, _ in pairs]),
            ("b", [b for _, b in pairs]),
            ("c", [_guard(dtype)] * len(expected)),
        )
    }
    (constant,) = (
        binding
        for binding in descriptor["bindings"]
        if binding["kind"] == "constant-buffer"
    )
    size = constant["scalarLayout"].get("memberName", constant["name"])
    inputs[size] = {"dtype": "uint32", "shape": [1], "values": [len(pairs)]}
    outputs = {"c": _payload(dtype, target, expected)}
    request = _dispatch_request(descriptor, package, inputs, outputs, len(pairs))
    return request, inputs, _bound_values(descriptor, outputs)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_extrema_reference_preserves_zero_sign_nan_payload_and_comparison_policy(
    dtype, operation
):
    _, bits, infinity = TYPES[dtype]
    sign = 1 << (bits - 1)
    for a, b in (
        (0, sign),
        (sign, 0),
        (infinity + 1, 0),
        (0, infinity + 1),
        (infinity | sign | 1, infinity + 7),
    ):
        expected = a if a & (sign - 1) > infinity else b
        assert _expected(a, b, dtype, operation) == expected
    # Only comparisons flush subnormals; selected storage words stay unchanged.
    expected = 0 if dtype == "float16" and operation == "Minimum" else 1
    assert _expected(0, 1, dtype, operation) == expected
    pairs = _pairs(dtype)
    assert len(pairs) == 4672 and pairs == _pairs(dtype)
    assert (0, 1) in pairs and (1, 0) in pairs


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_extrema_request_retains_operand_bits_and_physical_storage(
    tmp_path, dtype, target
):
    typename = TYPES[dtype][0]
    source = f"""#include <metal_stdlib>
using namespace metal;
kernel void selection(const device {typename}* a [[buffer(0)]],
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
    sign = 1 << (TYPES[dtype][1] - 1)
    pairs = [(sign, TYPES[dtype][2] + 1)]
    expected = [sign] + [_guard(dtype)] * 8
    request, inputs, outputs = _request(
        descriptor, package, dtype, target, pairs, expected
    )
    assert not request.execution_plan.diagnostics
    by_name = {value.name: value for value in request.fixture.inputs}
    for binding in descriptor["bindings"]:
        if binding["kind"] != "buffer":
            continue
        layout = binding["scalarLayout"]
        name = layout.get("memberName", binding["name"])
        value, payload = by_name[binding["name"]], inputs[name]
        assert value.dtype == payload["dtype"] and value.encoding == payload.get(
            "encoding"
        )
        assert list(value.values) == payload["values"]
        assert layout["elementSizeBytes"] == (
            4 if target == "opengl" else TYPES[dtype][1] // 8
        )
    (output,) = outputs.values()
    assert output == _payload(dtype, target, expected)


@pytest.mark.parametrize(
    "actual,expected",
    [
        ([0], [0x80000000]),
        ([0x7FC00001], [0x7FC00002]),
        ([0x7F800001], [0x7FC00001]),
        ([0, 0], [0, 0x3555]),
        ([False], [0]),
        ([], [0]),
    ],
)
def test_extrema_comparison_rejects_payload_value_and_guard_changes(actual, expected):
    with pytest.raises(AssertionError):
        _check_words(actual, expected)


def _cases(dtypes=TYPES):
    for dtype, operation in itertools.product(dtypes, OPERATIONS):
        pairs = _pairs(dtype)
        expected = [_expected(a, b, dtype, operation) for a, b in pairs] + [
            _guard(dtype)
        ] * 8
        yield BinaryCase(
            dtype=dtype,
            operation=operation,
            pairs=pairs,
            expected=expected,
            provenance=(
                {}
                if dtype == "float16"
                else {"binary32ComparisonProfile": "flush-subnormals"}
            ),
            comparison="exact selected operand words",
        )


def _compare_native(actual, expected, _dtype, _target):
    _check_words(actual, expected)


@pytest.mark.parametrize(
    "dtypes", NATIVE_DTYPE_BATCHES, ids=lambda types: "-".join(types)
)
def test_current_extrema_native_parity(tmp_path, dtypes, binary_metal_reference):
    run_binary_cases(
        tmp_path,
        REQUIRE_ENV,
        _cases(dtypes),
        request_for=_request,
        guard_for=_guard,
        source_control=binary_metal_reference,
        compare=_compare_native,
    )


def test_ci_requires_extrema_once_per_native_target():
    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate pinned native binary math"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    path = "demos/integrations/mlx/tests/kernels/test_current_extrema.py"
    assert workflow.count(path) == 1
    assert path in step and "pytest -q -n auto" in step
    assert "--timeout-seconds 900" in step
