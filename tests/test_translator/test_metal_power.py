"""Metal power domains, conversion boundaries and native numerical checks."""

import json
import math
import os
import struct
import sys
from decimal import Decimal, localcontext
from functools import lru_cache, partial

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalPreciseMathLoweringError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import _bits, _float, _round_decimal

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_POWER"
GUARD = 0x35A5B6C7
FIELDS = 24
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
float pow(float a, float b) { return a; }
kernel void powers(const device uint* values [[buffer(0)]],
                   device uint* results [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    float na = as_type<float>(values[2u * i] ^ 0x80000000u);
    float nb = as_type<float>(values[2u * i + 1u] ^ 0x80000000u);
    uint count = 0;
    float value = metal::pow(record(count, a), record(count, b));
    float2 pair = metal::pow(float2(a, na), float2(b, b));
    float3 triple = precise::pow(float3(a, a, na), float3(b, nb, b));
    float4 quad = metal::precise::pow(float4(a, na, a, na), float4(b, b, nb, nb));
    float2 scalar_base = metal::pow(a, float2(b, nb));
    float3 scalar_exponent = metal::pow(float3(a, na, a), record(count, b));
    half narrow = metal::pow(half(a), half(b));
    half2 narrow_pair = metal::pow(half2(a, na), half2(b, nb));
    half converted_exponent = metal::pow(half(-1.0f), 2049);
    results[4u + 24u * i] = as_type<uint>(a);
    results[5u + 24u * i] = as_type<uint>(b);
    results[6u + 24u * i] = as_type<uint>(value);
    results[7u + 24u * i] = as_type<uint>(metal::precise::pow(a, b));
    results[8u + 24u * i] = as_type<uint>(pair.x);
    results[9u + 24u * i] = as_type<uint>(pair.y);
    results[10u + 24u * i] = as_type<uint>(triple.x);
    results[11u + 24u * i] = as_type<uint>(triple.y);
    results[12u + 24u * i] = as_type<uint>(triple.z);
    results[13u + 24u * i] = as_type<uint>(quad.x);
    results[14u + 24u * i] = as_type<uint>(quad.y);
    results[15u + 24u * i] = as_type<uint>(quad.z);
    results[16u + 24u * i] = as_type<uint>(quad.w);
    results[17u + 24u * i] = as_type<uint>(scalar_base.x);
    results[18u + 24u * i] = as_type<uint>(scalar_base.y);
    results[19u + 24u * i] = as_type<uint>(scalar_exponent.x);
    results[20u + 24u * i] = as_type<uint>(scalar_exponent.y);
    results[21u + 24u * i] = as_type<uint>(scalar_exponent.z);
    results[22u + 24u * i] = as_type<uint>(float(narrow));
    results[23u + 24u * i] = as_type<uint>(float(narrow_pair.x));
    results[24u + 24u * i] = as_type<uint>(float(narrow_pair.y));
    results[25u + 24u * i] = as_type<uint>(float(converted_exponent));
    results[26u + 24u * i] = count;
    results[27u + 24u * i] = as_type<uint>(::pow(a, b));
}
"""


def _translate(tmp_path, source=SOURCE, target="crossgl"):
    path = tmp_path / "power.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_power_domain_helpers_compile(tmp_path, target):
    generated = _translate(tmp_path, target=target)
    for suffix in ("", "2", "3", "4"):
        assert f"metal_power_float{suffix}(" in generated
    _compile(
        generated,
        target,
        tmp_path,
        metal_compile_flags=("-fno-fast-math",),
        directx_compile_flags=("-enable-16bit-types",),
    )


@pytest.mark.parametrize("target", ("directx", "opengl", "metal"))
def test_power_domain_survives_saved_crossgl(tmp_path, target):
    saved = tmp_path / "saved.cgl"
    saved.write_text(_translate(tmp_path), encoding="utf-8")
    assert translate(str(saved), backend=target, format_output=False) == _translate(
        tmp_path, target=target
    )


def test_power_keeps_fast_mode_integer_helpers_and_source_overloads(tmp_path):
    source = """
        float pow(float a, float b) { return a; }
        int pow(int a, int b) { return a + b; }
        float source_call(float a, float b) { return ::pow(a, b); }
        int integer_call(int a, int b) { return ::pow(a, b); }
        float fast_call(float a, float b) { return metal::fast::pow(a, b); }
        float precise_call(float a, float b) { return metal::precise::pow(a, b); }
        float default_call(float a, float b) { return metal::pow(a, b); }
    """
    generated = _translate(tmp_path, source)
    assert (
        generated.count("return __crossgl_metal_power_float(float(a), float(b));") == 2
    )
    assert "return pow(a, b);" in generated
    assert "return pow__metal_overload_1(a, b);" in generated
    assert "return pow__metal_overload_2(a, b);" in generated


def test_power_helpers_reset_and_avoid_source_names():
    converter = MetalToCrossGLConverter()
    source = """
        float __crossgl_metal_power_float(float a) { return a; }
        float2 __crossgl_metal_power_float2(float2 a) { return a; }
        float2 evaluate(float2 a, float2 b) { return metal::pow(a, b); }
    """
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "float __crossgl_metal_power_float_(float base, float exponent)" in generated
    assert "__crossgl_metal_power_float2_(vec2(a), vec2(b))" in generated
    source = "float evaluate(float a, float b) { return metal::fast::pow(a, b); }"
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    assert "__crossgl_metal_power" not in generated


def test_power_diagnoses_global_runtime_initialization(tmp_path):
    with pytest.raises(
        MetalPreciseMathLoweringError, match="global initializers"
    ) as error:
        _translate(tmp_path, "constant float value = metal::pow(-2.0f, 3.0f);")
    assert error.value.operation == "pow"
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-precise-math-unsupported"
    )


@pytest.mark.parametrize("namespace", ("", "precise", "fast"))
@pytest.mark.parametrize("materialized", (False, True))
def test_power_preserves_bfloat_wrapper_ownership_and_narrowing(
    namespace, materialized
):
    qualifier = "METAL_FUNC" if materialized else ""
    body = "__metal_pow(float(x), float(y))" if materialized else "float(x) + float(y)"
    source = f"""
typedef bfloat bfloat16_t;
namespace metal {{
{('namespace ' + namespace + ' {') if namespace else ''}
{qualifier} bfloat16_t pow(bfloat16_t x, bfloat16_t y) {{
    return bfloat16_t({body});
}}
{'}' if namespace else ''}
}}
bfloat16_t apply(bfloat16_t x, bfloat16_t y) {{
    return metal::{(namespace + '::') if namespace else ''}pow(x, y);
}}
"""
    converter = MetalToCrossGLConverter()
    generated = converter.generate(MetalParser(MetalLexer(source).tokenize()).parse())
    lowered = materialized and namespace != "fast"
    assert ("__crossgl_metal_power_float" in generated) == lowered
    if lowered:
        assert "bfloat16(__crossgl_metal_power_float(float(x), float(y)))" in generated
    elif materialized:
        assert "bfloat16(pow(float(x), float(y)))" in generated


@lru_cache(maxsize=None)
def _oracle(a, b):
    x, y = _float(a), _float(b)
    if y == 0 or x == 1:
        return 0x3F800000
    if math.isnan(x) or math.isnan(y):
        return 0x7FC00000
    if math.isinf(y):
        if abs(x) == 1:
            return 0x3F800000
        return 0x7F800000 if (abs(x) > 1) != (y < 0) else 0
    odd = y.is_integer() and int(y) % 2 != 0
    sign = a & 0x80000000 if odd else 0
    if x == 0 or math.isinf(x):
        return sign | (0x7F800000 if math.isinf(x) != (y < 0) else 0)
    if x < 0 and not y.is_integer():
        return 0x7FC00000
    with localcontext() as context:
        context.prec = 100
        logarithm = Decimal.from_float(abs(x)).ln() * Decimal.from_float(y)
        if logarithm > 100:
            return sign | 0x7F800000
        if logarithm < -150:
            return sign
        exact = logarithm.exp()
        if exact >= Decimal(2) ** 128 - Decimal(2) ** 103:
            return sign | 0x7F800000
        return _round_decimal(exact) | sign


def _half(word):
    try:
        return struct.unpack("<H", struct.pack("<e", _float(word)))[0]
    except OverflowError:
        return (word >> 16 & 0x8000) | 0x7C00


def _wide_half(word):
    return _bits(struct.unpack("<e", struct.pack("<H", word))[0])


def _pairs():
    domains = (
        0,
        0x3F000000,
        0x3F800000,
        0x40000000,
        0x7F800000,
        0x7FC12345,
        0x7F812345,
    )
    exponents = {_bits(value) for value in (0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 8)}
    for center in (_bits(1.0), _bits(2**23), _bits(2**24), _bits(2**31)):
        exponents.update(range(center - 2, center + 3))
    exponents.update((0x7F7FFFFF, 0x7F800000, 0x7FC12345, 0x7F812345))
    exponents |= {word | 0x80000000 for word in exponents}
    pairs = {
        (a | sign, b) for a in domains for sign in (0, 0x80000000) for b in exponents
    }
    for base in (0.125, 0.5, 1.25, 2.0, 3.0, 7.0, 9.0):
        for exponent in range(-8, 9):
            for sign in (1, -1):
                pairs.add((_bits(sign * base), _bits(exponent / 2)))
    return sorted(pairs)


def _operands(a, b):
    na, nb = a ^ 0x80000000, b ^ 0x80000000
    return (
        (a, b),
        (a, b),
        (a, b),
        (na, b),
        (a, b),
        (a, nb),
        (na, b),
        (a, b),
        (na, b),
        (a, nb),
        (na, nb),
        (a, b),
        (a, nb),
        (a, b),
        (na, b),
        (a, b),
    )


def _expected(pairs):
    words = [GUARD] * 4
    for a, b in pairs:
        words.extend((a, b))
        words.extend(_oracle(x, y) for x, y in _operands(a, b))
        for x, y in ((a, b), (a, b), (a ^ 0x80000000, b ^ 0x80000000)):
            words.append(
                _wide_half(_half(_oracle(_wide_half(_half(x)), _wide_half(_half(y)))))
            )
        words.extend((0x3F800000, 3, a))
    return words + [GUARD] * 4


def _check(actual, pairs):
    expected = _expected(pairs)
    assert len(actual) == len(expected) == FIELDS * len(pairs) + 8
    assert all(type(word) is int and 0 <= word <= 0xFFFFFFFF for word in actual)
    assert actual[:4] == actual[-4:] == [GUARD] * 4, "guards"
    maximum = 0
    for i, (a, b) in enumerate(pairs):
        row = actual[4 + FIELDS * i : 4 + FIELDS * (i + 1)]
        want = expected[4 + FIELDS * i : 4 + FIELDS * (i + 1)]
        assert row[:2] == [a, b], "operand copies"
        assert row[-2:] == [3, a], "evaluation count and source overload"
        for index, (got, reference) in enumerate(zip(row[2:-2], want[2:-2])):
            if reference & 0x7FFFFFFF > 0x7F800000:
                assert got & 0x7FFFFFFF > 0x7F800000, "NaN classification"
                continue
            assert got >> 31 == reference >> 31, "result sign"
            if reference & 0x7FFFFFFF in (0, 0x3F800000, 0x7F800000):
                assert got == reference, "exact domain result"
            narrow = index >= 16
            if narrow:
                assert _wide_half(_half(got)) == got, "half rounding boundary"
                error = abs(_half(got) - _half(reference))
            else:
                error = abs(got - reference)
            assert error <= (1 if narrow else 16), (
                i,
                index,
                hex(a),
                hex(b),
                hex(got),
                hex(reference),
                error,
            )
            maximum = max(maximum, error)
    return maximum


def test_power_reference_and_comparison_preserve_domain_contracts():
    pairs = _pairs()
    assert (0xBF800000, 0x4B7FFFFF) in pairs
    assert (0xBF800000, 0x4B800000) in pairs
    assert _oracle(0xBF800000, 0x4B7FFFFF) == 0xBF800000
    assert _oracle(0xBF800000, 0x4B800000) == 0x3F800000
    assert _oracle(0x80000000, 0xBF800000) == 0xFF800000
    assert _oracle(0x7FC12345, 0) == 0x3F800000
    expected = _expected(pairs)
    assert _check(expected, pairs) == 0
    for index in (0, 4, 6, 4 + FIELDS - 2, 4 + FIELDS - 1, len(expected) - 1):
        changed = expected.copy()
        changed[index] ^= 0x80000000
        with pytest.raises(AssertionError):
            _check(changed, pairs)


@pytest.mark.parametrize("source_control", (False, True))
def test_power_executes_native_domains(tmp_path, source_control, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native power checks")
    from tests.test_translator import test_fused_math as native_math

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    if source_control and target != "metal":
        pytest.skip("The unchanged source control requires Metal")
    monkeypatch.setattr(
        native_math,
        "_compile",
        partial(
            _compile,
            metal_compile_flags=("-fno-fast-math",),
            directx_compile_flags=("-enable-16bit-types",),
        ),
    )
    pairs = _pairs()
    expected = _expected(pairs)
    source = SOURCE if source_control else _translate(tmp_path, target=target)
    actual, evidence = native_math._dispatch(
        tmp_path,
        target,
        source,
        pairs,
        len(expected),
        entry="powers" if target == "metal" else None,
        initial_output=[GUARD] * len(expected),
    )
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    maximum = _check(actual, pairs)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                **evidence,
                "sourceControl": source_control,
                "pairCount": len(pairs),
                "valueCount": 20 * len(pairs),
                "maximumStorageUlpError": maximum,
                "guardCount": 8,
                "completeFiniteDomainVerified": False,
            },
            indent=2,
        )
    )


def test_power_executes_metal_subnormal_control(tmp_path, monkeypatch):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native power checks")
    if sys.platform != "darwin":
        pytest.skip("Subnormal source behavior requires an original Metal control")
    from tests.test_translator import test_fused_math as native_math

    monkeypatch.setattr(
        native_math,
        "_compile",
        partial(_compile, metal_compile_flags=("-fno-fast-math",)),
    )
    pairs = []
    for small in (1, 0x00010000, 0x007FFFFF):
        for base in (0, small, 0x3F800000, 0x40000000, 0x7F800000, 0x7FC12345):
            for exponent in (small, 0x00800000, 0x3F000000, 0x3F800000, 0x40000000):
                for base_sign in (0, 0x80000000):
                    for exponent_sign in (0, 0x80000000):
                        pairs.append((base | base_sign, exponent | exponent_sign))
    count = FIELDS * len(pairs) + 8
    results = {}
    receipts = {}
    for name in ("original", "generated"):
        work = tmp_path / name
        work.mkdir()
        source = SOURCE if name == "original" else _translate(work, target="metal")
        actual, receipts[name] = native_math._dispatch(
            work,
            "metal",
            source,
            pairs,
            count,
            entry="powers",
            initial_output=[GUARD] * count,
        )
        assert len(actual) == count
        assert actual[:4] == actual[-4:] == [GUARD] * 4
        for index, pair in enumerate(pairs):
            row = actual[4 + FIELDS * index : 4 + FIELDS * (index + 1)]
            assert row[:2] == list(pair)
            assert row[-2:] == [3, pair[0]]
        results[name] = actual
    for got, expected in zip(results["generated"], results["original"]):
        if expected & 0x7FFFFFFF > 0x7F800000:
            assert got & 0x7FFFFFFF > 0x7F800000
        else:
            assert got == expected
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "receipts": receipts,
                "pairCount": len(pairs),
                "valueCount": 20 * len(pairs),
                "guardCountPerPath": 8,
                "crossTargetSubnormalParityVerified": False,
            },
            indent=2,
        )
    )
