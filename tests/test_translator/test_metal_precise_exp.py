"""Precise binary32 exponential, narrowing boundaries and native controls."""

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
from crosstl.translator.source_licenses import SOURCE_LICENSES
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import (
    GUARD,
    LANE_SIGNS,
    _bits,
    _execute,
    _float,
    _round_decimal,
)

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_EXP"
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
using Quad = float4;
float record(thread uint& count, float value) { count += 1; return value; }
Scalar evaluate(Scalar value) { return metal::precise::exp(value); }
kernel void exp_values(device const uint* values [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float negative = as_type<float>(values[i] ^ 0x80000000u);
    uint count = 0;
    float scalar = evaluate(x);
    float2 pair = metal::precise::exp(float2(x, negative));
    float3 triple = precise::exp(float3(x, negative, x));
    Quad quad = metal::precise::exp(Quad(record(count, x), negative, x, negative));
    results[12 * i] = as_type<uint>(scalar);
    results[12 * i + 1] = as_type<uint>(pair.x);
    results[12 * i + 2] = as_type<uint>(pair.y);
    results[12 * i + 3] = as_type<uint>(triple.x);
    results[12 * i + 4] = as_type<uint>(triple.y);
    results[12 * i + 5] = as_type<uint>(triple.z);
    results[12 * i + 6] = as_type<uint>(quad.x);
    results[12 * i + 7] = as_type<uint>(quad.y);
    results[12 * i + 8] = as_type<uint>(quad.z);
    results[12 * i + 9] = as_type<uint>(quad.w);
    results[12 * i + 10] = count;
    results[12 * i + 11] = as_type<uint>(float(bfloat(scalar)));
}
"""


def _translate(tmp_path, source, target):
    path = tmp_path / "exp.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_exp_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, SOURCE, target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_precise_exp_float{suffix}(" in generated
    assert "return exp(" not in generated
    assert "double" not in generated
    for line in SOURCE_LICENSES["fdlibm"].splitlines():
        assert line in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-std=metal3.1",))


def test_precise_exp_keeps_other_modes_and_source_functions(tmp_path):
    generated = _translate(
        tmp_path,
        """
        float exp(float x) { return x + 3.0f; }
        float explicit_mode(float x) { return metal::precise::exp(x); }
        float imported_mode(float x) { return precise::exp(x); }
        float default_mode(float x) { return metal::exp(x); }
        float fast_mode(float x) { return metal::fast::exp(x); }
        float own_function(float x) { return ::exp(x); }
    """,
        "crossgl",
    )
    assert generated.count("return __crossgl_metal_precise_exp_float(x);") == 2
    assert generated.count("return exp(x);") == 2
    assert "return exp__metal_overload_1(x);" in generated


def test_precise_exp_names_and_state_are_isolated():
    source = """
        float __crossgl_metal_precise_exp_float(float x) { return x; }
        float2 __crossgl_metal_precise_exp_float2(float2 x) { return x; }
        float2 evaluate(float2 x) { return metal::precise::exp(x); }
    """
    converter = MetalToCrossGLConverter()
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_precise_exp_float_(float value)" in generated
    assert "return __crossgl_metal_precise_exp_float2_(x);" in generated
    source = "float evaluate(float x) { return metal::exp(x); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_precise_exp" not in generated


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_precise_exp_survives_saved_intermediate(tmp_path, target):
    canonical = _translate(tmp_path, SOURCE, "crossgl")
    path = tmp_path / "saved.cgl"
    path.write_text(canonical, encoding="utf-8")
    generated = translate(str(path), backend=target, format_output=False)
    assert "metal_precise_exp_float(" in generated
    assert SOURCE_LICENSES["fdlibm"].splitlines()[-1] in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-std=metal3.1",))


@pytest.mark.parametrize("operand", ["int", "double", "half2", "float8", "Payload"])
def test_precise_exp_diagnoses_unsupported_operands(tmp_path, operand):
    source = (
        "struct Payload { float x; };\n"
        + f"{operand} f({operand} x) {{ return metal::precise::exp(x); }}"
    )
    with pytest.raises(MetalPreciseMathLoweringError) as error:
        _translate(tmp_path, source, "crossgl")
    assert error.value.operation == "exp"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@lru_cache(maxsize=None)
def _oracle(word):
    magnitude = word & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        return 0x7FC00000
    value = _float(word)
    if value > 90:
        return 0x7F800000
    if value < -110:
        return 0
    if magnitude < 0x33000000:
        return 0x3F800000
    with localcontext() as context:
        context.prec = 100
        result = Decimal.from_float(value).exp()
        overflow_midpoint = Decimal(2) ** 128 - Decimal(2) ** 103
        if result >= overflow_midpoint:
            return 0x7F800000
        if result > Decimal.from_float(_float(0x7F7FFFFF)):
            return 0x7F7FFFFF
        return _round_decimal(result)


def _narrow(word):
    if word & 0x7FFFFFFF > 0x7F800000:
        return 0x7FC00000
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) << 16


def _inputs():
    words = set(range(0x40DB0000 - 1024, 0x40DB0000 + 1025))
    for boundary in (
        0x39000000,
        0x3EB17218,
        0x3F851592,
        0x42AEAC50,
        0x42B17218,
        0x42CFF1B5,
    ):
        words.update(range(boundary - 32, boundary + 33))
    rng = random.Random(1993)
    for exponent in range(256):
        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF, rng.getrandbits(23)):
            words.add(exponent << 23 | mantissa)
    words.update(_bits(rng.uniform(0, 110)) for _ in range(4096))
    return sorted(words | {word | 0x80000000 for word in words})


def _expected(inputs):
    output = []
    for word in inputs:
        output.extend(_oracle(word ^ (sign << 31)) for sign in LANE_SIGNS)
        output.extend((1, _narrow(_oracle(word))))
    return output + GUARD


def _check(actual, expected, *, original=False):
    assert len(actual) == len(expected), "output size"
    assert actual[-len(GUARD) :] == GUARD, "output guard"
    maximum = 0
    for offset in range(0, len(actual) - len(GUARD), 12):
        for lane in range(10):
            got, want = actual[offset + lane], expected[offset + lane]
            if want > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
            elif original and 0 < want < 0x800000 and got == 0:
                continue
            elif want in {0, 0x7F800000}:
                assert got == want, "zero or infinity"
            else:
                assert got < 0x7F800000, "positive finite exponential"
                distance = abs(got - want)
                assert distance <= (4 if original else 2), (
                    offset,
                    lane,
                    hex(got),
                    hex(want),
                )
                maximum = max(maximum, distance)
        assert actual[offset + 10] == 1, "operand evaluation count"
        narrowed = actual[offset + 11]
        scalar = actual[offset]
        if scalar & 0x7FFFFFFF > 0x7F800000:
            assert narrowed & 0x7FFFFFFF > 0x7F800000, "narrow NaN classification"
        else:
            assert narrowed == _narrow(scalar), "bfloat conversion"
    return maximum


def test_precise_exp_oracle_and_coverage():
    words = _inputs()
    assert len(words) < 65536
    assert {(word >> 23) & 255 for word in words} == set(range(256))
    assert all(word ^ 0x80000000 in words for word in words)
    for word in words:
        value = _float(word)
        if math.isfinite(value) and -104 <= value <= 88:
            assert _oracle(word) == _bits(math.exp(value))
    assert _oracle(0x40DB0000) == 0x446A8001
    assert _narrow(_oracle(0x40DB0000)) == 0x446B0000
    assert _oracle(0x42B17217) == 0x7F7FFF84
    assert _oracle(0x42B17218) == 0x7F800000
    assert _oracle(0xC2CFF1B4) == 1
    assert _oracle(0xC2CFF1B5) == 0


@pytest.mark.parametrize(
    "fault", ["size", "guard", "count", "value", "infinity", "nan", "narrow", "flush"]
)
def test_precise_exp_verifier_rejects_corruption(fault):
    expected = _expected([0x40DB0000, 0x7F800000, 0x7FC12345, _bits(-88)])
    actual = list(expected)
    if fault == "size":
        actual.pop()
    elif fault == "guard":
        actual[-1] ^= 1
    elif fault == "count":
        actual[10] = 2
    elif fault == "infinity":
        actual[12] = 0x7F7FFFFF
    elif fault == "nan":
        actual[24] = 0
    elif fault == "narrow":
        actual[11] = 0x446A0000
    elif fault == "flush":
        actual[36] = 0
    else:
        actual[0] -= 6
    with pytest.raises(AssertionError):
        _check(actual, expected)


def test_precise_exp_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native exp execution")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected(inputs)
    (tmp_path / "inputs.bin").write_bytes(struct.pack(f"<{len(inputs)}I", *inputs))
    (tmp_path / "expected.bin").write_bytes(
        struct.pack(f"<{len(expected)}I", *expected)
    )

    def check_outputs(actual, reference, original=False):
        maximum = _check(actual, reference, original=original)
        offset = 12 * inputs.index(0x40DB0000)
        assert actual[offset] == 0x446A8001, "precise exponential midpoint"
        assert actual[offset + 11] == 0x446B0000, "bfloat midpoint"
        return maximum

    records = {
        "generated": _execute(
            tmp_path / "generated",
            target,
            _translate(tmp_path, SOURCE, target),
            inputs,
            expected,
            metal_entry="exp_values",
            check_outputs=check_outputs,
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        )
    }
    if target == "metal":
        records["original"] = _execute(
            tmp_path / "original",
            target,
            SOURCE,
            inputs,
            expected,
            metal_entry="exp_values",
            check_outputs=lambda actual, reference: check_outputs(
                actual, reference, True
            ),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "oracle": "100-digit Decimal exp, nearest-even binary32",
                "generatedMaxUlp": 2,
                "originalMaxUlp": 4,
                "originalMetalControl": (
                    "Permits subnormal result flushing; generated checks do not."
                ),
                "records": records,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("operand", ["half", "bfloat"])
def test_precise_exp_promotes_narrow_alias_once(tmp_path, operand):
    source = """#include <metal_stdlib>
using namespace metal;
using Narrow = $operand;
Narrow record(thread uint& count, Narrow value) { count += 1; return value; }
kernel void exp_promoted(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    Narrow value = as_type<$operand>(ushort(values[i]));
    uint count = 0;
    auto implicit_value = metal::precise::exp(record(count, value));
    auto explicit_value = metal::precise::exp(float(value));
    results[3 * i] = as_type<uint>(implicit_value);
    results[3 * i + 1] = as_type<uint>(explicit_value);
    results[3 * i + 2] = count;
}
""".replace("$operand", operand)
    canonical = _translate(tmp_path, source, "crossgl")
    assert "__crossgl_metal_precise_exp_float(record(count, value))" in canonical
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native exp promotion")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = [
        0,
        0x8000,
        1,
        0x8001,
        0x3C00,
        0x40DB,
        0x7BFF,
        0x7C00,
        0x7F80,
        0x7FC1,
        0xBC00,
        0xC0DB,
        0xFBFF,
        0xFC00,
        0xFF80,
    ]
    expected = []
    for word in inputs:
        promoted = (
            _bits(struct.unpack("<e", struct.pack("<H", word))[0])
            if operand == "half"
            else word << 16
        )
        result = _oracle(promoted)
        expected.extend((result, result, 1))
    expected += GUARD

    def check_outputs(actual, reference):
        assert len(actual) == len(reference)
        assert actual[-len(GUARD) :] == GUARD
        maximum = 0
        for offset in range(0, len(reference) - len(GUARD), 3):
            assert (
                actual[offset] == actual[offset + 1]
            ), "implicit and explicit promotion"
            assert actual[offset + 2] == 1, "operand evaluation count"
            got, want = actual[offset], reference[offset]
            if want > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000
            elif want in {0, 0x7F800000}:
                assert got == want
            else:
                distance = abs(got - want)
                assert distance <= 2
                maximum = max(maximum, distance)
        return maximum

    record = _execute(
        tmp_path / "generated",
        target,
        _translate(tmp_path, source, target),
        inputs,
        expected,
        metal_entry="exp_promoted",
        check_outputs=check_outputs,
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )
    (tmp_path / "reference.json").write_text(
        json.dumps(
            {
                "operand": operand,
                "inputs": inputs,
                "expected": expected,
                "generated": record,
            },
            indent=2,
        )
    )
