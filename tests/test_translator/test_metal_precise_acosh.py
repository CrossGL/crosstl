"""Precise binary32 inverse hyperbolic cosine, including native edge cases."""

import json
import math
import os
import random
import struct
import sys
from decimal import Decimal, localcontext
from functools import lru_cache

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalPreciseMathLoweringError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import (
    GUARD,
    LANE_SIGNS,
    _bits,
    _execute,
    _float,
    _round_decimal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_ACOSH"
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
kernel void acosh_values(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    uint count = 0;
    float2 pair = metal::precise::acosh(float2(x, -x));
    float3 triple = precise::acosh(float3(x, -x, x));
    float4 quad = metal::precise::acosh(float4(record(count, x), -x, x, -x));
    results[11 * i] = as_type<uint>(metal::precise::acosh(x));
    results[11 * i + 1] = as_type<uint>(pair.x);
    results[11 * i + 2] = as_type<uint>(pair.y);
    results[11 * i + 3] = as_type<uint>(triple.x);
    results[11 * i + 4] = as_type<uint>(triple.y);
    results[11 * i + 5] = as_type<uint>(triple.z);
    results[11 * i + 6] = as_type<uint>(quad.x);
    results[11 * i + 7] = as_type<uint>(quad.y);
    results[11 * i + 8] = as_type<uint>(quad.z);
    results[11 * i + 9] = as_type<uint>(quad.w);
    results[11 * i + 10] = count;
}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "acosh.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_acosh_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_acosh_float{suffix}(" in generated
    assert "return acosh(" not in generated
    assert "double" not in generated
    _compile(generated, target, tmp_path)


def test_precise_acosh_keeps_other_modes_and_source_functions(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float acosh(float x) { return x + 3.0f; }
        float explicit_mode(float x) { return metal::precise::acosh(x); }
        float imported_mode(float x) { return precise::acosh(x); }
        float default_mode(float x) { return metal::acosh(x); }
        float fast_mode(float x) { return metal::fast::acosh(x); }
        float own_function(float x) { return ::acosh(x); }
    """,
        "crossgl",
    )
    assert generated.count("return __crossgl_metal_precise_acosh_float(x);") == 2
    assert generated.count("return acosh(x);") == 2
    assert "return acosh__metal_overload_1(x);" in generated


def test_precise_acosh_names_and_state_are_isolated(tmp_path):
    source = """
        float __crossgl_metal_precise_acosh_float(float x) { return x; }
        float __crossgl_metal_precise_acosh_log1p(float x) { return x; }
        float2 __crossgl_metal_precise_acosh_float2(float2 x) { return x; }
        float2 evaluate(float2 x) { return metal::precise::acosh(x); }
    """
    converter = MetalToCrossGLConverter()
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_acosh_float_(float value)" in generated
    assert "float __crossgl_metal_precise_acosh_log1p_(float value)" in generated
    assert "return __crossgl_metal_precise_acosh_float2_(x);" in generated
    source = "float evaluate(float x) { return metal::acosh(x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_acosh" not in generated


@pytest.mark.parametrize("operand", ["int", "double", "float8", "Payload"])
def test_precise_acosh_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} x) {{ return metal::precise::acosh(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source, "crossgl")
    assert error.value.operation == "acosh"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@lru_cache(maxsize=None)
def _oracle(word):
    value = _float(word)
    if math.isnan(value) or value < 1:
        return 0x7FC00000
    if math.isinf(value):
        return 0x7F800000
    if value == 1:
        return 0
    with localcontext() as context:
        context.prec = 90
        x = Decimal.from_float(value)
        return _round_decimal((x + (x * x - 1).sqrt()).ln())


def _inputs():
    words = set(range(0x3F800000, 0x3F802001))
    for value in (1.0, 1.0625, 13 / 12, 2.0, 4096.0):
        center = _bits(value)
        words.update(range(center - 8, center + 9))
    rng = random.Random(1994)
    for exponent in range(127, 255):
        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF, rng.getrandbits(23)):
            words.add(exponent << 23 | mantissa)
    words.update(
        (
            0,
            0x80000000,
            1,
            0x7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFC12345,
            _bits(-1),
            _bits(0.5),
        )
    )
    return sorted(words)


def _expected(inputs):
    output = []
    for word in inputs:
        output.extend(_oracle(word ^ (sign << 31)) for sign in LANE_SIGNS)
        output.append(1)
    return output + GUARD


def _check_word(got, want):
    if want & 0x7FFFFFFF > 0x7F800000:
        assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
        return 0
    if want in (0, 0x7F800000):
        assert got == want, "zero or infinity"
        return 0
    assert got >> 31 == want >> 31, "result sign"
    error = abs(got - want)
    assert error <= 4, (hex(got), hex(want), error)
    return error


def _check(actual, expected):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    maximum = 0
    for index, (got, want) in enumerate(
        zip(actual[: -len(GUARD)], expected[: -len(GUARD)])
    ):
        if index % 11 == 10:
            assert got == 1, "operand evaluation count"
        else:
            maximum = max(maximum, _check_word(got, want))
    return maximum


def test_precise_acosh_oracle_and_coverage():
    words = _inputs()
    assert len(words) < 65536
    assert set(range(0x3F800000, 0x3F802001)) <= set(words)
    for value in (1.0, 1.0002447366714478, 1.0625, 4096.0, _float(0x7F7FFFFF)):
        assert _oracle(_bits(value)) == _bits(math.acosh(value))
    assert _oracle(_bits(-1.0)) == 0x7FC00000
    assert _oracle(0x7F800000) == 0x7F800000


@pytest.mark.parametrize(
    "fault", ["size", "guard", "count", "value", "zero", "infinity", "nan"]
)
def test_precise_acosh_verifier_rejects_corruption(fault):
    expected = _expected([_bits(1.0), _bits(2.0), 0x7F800000, 0x7FC12345])
    actual = list(expected)
    if fault == "size":
        actual.pop()
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[10] = 2
    elif fault == "value":
        actual[11] += 8
    elif fault == "zero":
        actual[0] = 0x80000000
    elif fault == "infinity":
        actual[22] = 0x7F7FFFFF
    else:
        actual[33] = 0
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_precise_acosh_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native acosh execution")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected(inputs)
    (tmp_path / "inputs.bin").write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )
    generated = _translate(tmp_path, SOURCE, target)
    records = {
        "generated": _execute(
            tmp_path / "generated",
            target,
            generated,
            inputs,
            expected,
            metal_entry="acosh_values",
            check_outputs=_check,
        )
    }
    if target == "metal":
        records["original"] = _execute(
            tmp_path / "original",
            target,
            SOURCE,
            inputs,
            expected,
            metal_entry="acosh_values",
            check_outputs=_check,
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "oracle": "90-digit Decimal acosh, binary32 RNE",
                "records": records,
            },
            indent=2,
        )
    )
