"""Precise binary32 arctangent with independent native numerical checks."""

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
    _pi,
    _round_decimal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_ATAN"
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
kernel void atan_values(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float2 pair = metal::precise::atan(float2(x, negative));
    float3 triple = precise::atan(float3(x, negative, x));
    float4 quad = metal::precise::atan(float4(record(count, x), negative, x, negative));
    results[11 * i] = as_type<uint>(metal::precise::atan(x));
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
    path = tmp_path / "atan.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_atan_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_atan_float{suffix}(" in generated
    assert "return atan(" not in generated
    assert "double" not in generated
    _compile(generated, target, tmp_path)


def test_precise_atan_keeps_other_modes_and_source_functions(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float atan(float x) { return x + 3.0f; }
        float explicit_mode(float x) { return metal::precise::atan(x); }
        float imported_mode(float x) { return precise::atan(x); }
        float default_mode(float x) { return metal::atan(x); }
        float fast_mode(float x) { return metal::fast::atan(x); }
        float own_function(float x) { return ::atan(x); }
    """,
        "crossgl",
    )
    assert generated.count("return __crossgl_metal_precise_atan_float(x);") == 2
    assert generated.count("return atan(x);") == 2
    assert "return atan__metal_overload_1(x);" in generated


def test_precise_atan_names_and_state_are_isolated():
    source = """
        float __crossgl_metal_precise_atan_float(float x) { return x; }
        float2 __crossgl_metal_precise_atan_float2(float2 x) { return x; }
        float2 evaluate(float2 x) { return metal::precise::atan(x); }
    """
    converter = MetalToCrossGLConverter()
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_atan_float_(float value)" in generated
    assert "return __crossgl_metal_precise_atan_float2_(x);" in generated
    source = "float evaluate(float x) { return metal::atan(x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_atan" not in generated


@pytest.mark.parametrize("operand", ["int", "double", "half", "float8", "Payload"])
def test_precise_atan_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} x) {{ return metal::precise::atan(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source, "crossgl")
    assert error.value.operation == "atan"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@lru_cache(maxsize=None)
def _oracle(word):
    magnitude = word & 0x7FFFFFFF
    sign = word & 0x80000000
    if magnitude > 0x7F800000:
        return 0x7FC00000
    if magnitude == 0:
        return word
    with localcontext() as context:
        context.prec = 100
        if magnitude == 0x7F800000:
            return _round_decimal(_pi() / 2) | sign
        x = Decimal.from_float(_float(magnitude))
        invert = x > 1
        if invert:
            x = 1 / x
        # Two half-angle identities differ from the emitted pi/4 reduction.
        for _ in range(2):
            x = x / (1 + (1 + x * x).sqrt())
        power = total = x
        for index in range(1, 200):
            power *= -x * x
            term = power / (2 * index + 1)
            total += term
            if abs(term) < Decimal("1e-95"):
                break
        else:
            raise AssertionError("arctangent reference did not converge")
        angle = 4 * total
        if invert:
            angle = _pi() / 2 - angle
        return _round_decimal(angle) | sign


def _inputs():
    center = _bits(0.125)
    words = set(range(center - 2048, center + 2049))
    for value in (2**-12, math.sqrt(2) - 1, 1.0, math.sqrt(2) + 1, 2**26):
        center = _bits(value)
        words.update(range(center - 16, center + 17))
    rng = random.Random(1995)
    for exponent in range(255):
        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF, rng.getrandbits(23)):
            words.add(exponent << 23 | mantissa)
    words.update((0x7F800000, 0x7FC12345))
    return sorted(words | {word | 0x80000000 for word in words})


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
    assert got >> 31 == want >> 31, "result sign"
    if want & 0x7FFFFFFF < 0x00800000:
        assert got == want, "zero or subnormal bits"
        return 0
    error = abs(got - want)
    assert error <= 4, (hex(got), hex(want), error)
    return error


def _check(actual, expected, *, check_word=_check_word):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    maximum = 0
    for index, (got, want) in enumerate(
        zip(actual[: -len(GUARD)], expected[: -len(GUARD)])
    ):
        if index % 11 == 10:
            assert got == 1, "operand evaluation count"
        else:
            maximum = max(maximum, check_word(got, want))
    return maximum


def _check_original_word(got, want):
    # MSL 8.1/8.5 permits subnormal flushing with either zero sign. The
    # native atan also returns +0 for -0; generated outputs remain bit-checked.
    if want & 0x7FFFFFFF < 0x00800000 and got & 0x7FFFFFFF == 0:
        return 0
    return _check_word(got, want)


def _check_original(actual, expected):
    return _check(actual, expected, check_word=_check_original_word)


def test_original_control_does_not_relax_generated_checks():
    for want in (0x80000000, 1, 0x807FFFFF):
        assert _check_original_word(0, want) == 0
        with pytest.raises(AssertionError):
            _check_word(0, want)
    with pytest.raises(AssertionError):
        _check_original_word(_bits(0.12434358894824982), _oracle(_bits(0.125)))


def test_precise_atan_oracle_and_coverage():
    words = _inputs()
    assert len(words) < 65536
    assert all(word ^ 0x80000000 in words for word in words)
    for word in words:
        value = _float(word)
        if not math.isnan(value):
            assert _oracle(word) == _bits(math.atan(value))
    assert _oracle(_bits(0.125)) == 0x3DFEADD5


@pytest.mark.parametrize(
    "fault", ["size", "guard", "count", "value", "zero", "subnormal", "nan"]
)
def test_precise_atan_verifier_rejects_corruption(fault):
    expected = _expected([0x80000000, _bits(0.125), 1, 0x7FC12345])
    actual = list(expected)
    if fault == "size":
        actual.pop()
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[10] = 2
    elif fault == "value":
        actual[11] = _bits(0.12434358894824982)
    elif fault == "zero":
        actual[0] = 0
    elif fault == "subnormal":
        actual[22] = 0
    else:
        actual[33] = 0
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_precise_atan_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native atan execution")
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
            metal_entry="atan_values",
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
            metal_entry="atan_values",
            check_outputs=_check_original,
            metal_compile_flags=("-fno-fast-math",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "oracle": "100-digit Decimal half-angle atan, binary32 RNE",
                "records": records,
                "originalMetalControl": (
                    "Permits subnormal flushing and either zero sign; generated checks remain exact."
                ),
            },
            indent=2,
        )
    )
