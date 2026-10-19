"""Float-to-integer-to-float round trips keep the integer's +0.0 result.

Mesa folds ``float(int(x))`` into ``trunc(x)``, which returns -0.0 for x in
(-1, -0.0]. MLX's complex remainder (``a - b * static_cast<int64_t>(a / b)``)
then changes zero signs, so the OpenGL backend converts such values through a
helper that clears the sign bit whenever the integer is zero.
"""

import json
import os
import re
import struct
import sys
from pathlib import Path

import pytest

import crosstl.translator
from crosstl import translate
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_INTEGER_FLOAT_ROUND_TRIP"
OUTPUT_COUNT = 8
SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void integer_float(device const uint* values [[buffer(0)]],
                          device uint* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    float quotient = a / b;
    float remainder = a - (b * static_cast<int64_t>(quotient));
    float truncated = static_cast<float>(static_cast<int64_t>(quotient));
    float explicit_value = static_cast<float>(static_cast<int>(a));
    float assigned = static_cast<int>(a);
    float2 pair = float2(int2(float2(a, b)));
    float unsigned_value =
        static_cast<float>(static_cast<uint>(min(max(a, -0.75f), 65536.0f)));
    float product = a * static_cast<int>(b);
    results[8u * i] = as_type<uint>(remainder);
    results[8u * i + 1u] = as_type<uint>(truncated);
    results[8u * i + 2u] = as_type<uint>(explicit_value);
    results[8u * i + 3u] = as_type<uint>(assigned);
    results[8u * i + 4u] = as_type<uint>(pair.x);
    results[8u * i + 5u] = as_type<uint>(pair.y);
    results[8u * i + 6u] = as_type<uint>(unsigned_value);
    results[8u * i + 7u] = as_type<uint>(product);
}
"""
# Power-of-two divisors keep every quotient exact on targets whose division
# is not correctly rounded, so only the zero signs can differ.
DIVIDENDS = (
    0.0,
    0.25,
    0.5,
    0.75,
    0.999,
    1.0,
    1.5,
    2.0,
    2.5,
    7.0,
    9.0,
    1000.75,
    16777215.0,
)
DIVISORS = (2.0, 0.25, 0.5, 4.0, 0.125)


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _pairs():
    return [
        (_float32(sign_a * a), _float32(sign_b * b))
        for a in DIVIDENDS
        for b in DIVISORS
        for sign_a in (1.0, -1.0)
        for sign_b in (1.0, -1.0)
    ]


def _inputs():
    return [_bits(value) for pair in _pairs() for value in pair]


def _integer_float(value):
    # Python's int() truncates toward zero and float(0) is +0.0.
    return _float32(float(int(value)))


def _expected():
    expected = []
    for a, b in _pairs():
        quotient = _float32(a / b)
        truncated = _integer_float(quotient)
        remainder = _float32(a - _float32(b * truncated))
        clamped = min(max(a, -0.75), 65536.0)
        expected.extend(
            _bits(value)
            for value in (
                remainder,
                truncated,
                _integer_float(a),
                _integer_float(a),
                _integer_float(a),
                _integer_float(b),
                _integer_float(clamped),
                _float32(a * _integer_float(b)),
            )
        )
    return expected + GUARD


def _check(actual, expected):
    assert len(actual) == len(expected)
    pairs = _pairs()
    assert actual == expected, [
        (pairs[index // OUTPUT_COUNT], index % OUTPUT_COUNT, hex(wanted), hex(found))
        for index, (found, wanted) in enumerate(zip(actual, expected))
        if found != wanted
    ][:20]
    return 0


def _generate(source):
    return GLSLCodeGen().generate(crosstl.translator.parse(source))


def _translate(tmp_path, target, source=SOURCE):
    path = tmp_path / "integer_float.metal"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def test_oracle_covers_negative_fractions_and_zero_quotients():
    expected = _expected()
    pairs = _pairs()
    remainders = expected[: len(pairs) * OUTPUT_COUNT : OUTPUT_COUNT]
    assert _bits(-0.0) in remainders and _bits(0.0) in remainders
    for lane in range(1, 7):
        assert _bits(-0.0) not in expected[lane : len(pairs) * OUTPUT_COUNT : 8]
    products = expected[7 : len(pairs) * OUTPUT_COUNT : OUTPUT_COUNT]
    assert _bits(-0.0) in products and _bits(0.0) in products
    # These quotients truncate to -0.0, the value a folded conversion keeps.
    negative_fraction_quotients = [
        (a, b) for a, b in pairs if -1.0 < _float32(a / b) < 0.0
    ]
    assert len(negative_fraction_quotients) >= 24
    for (a, b), remainder in zip(pairs, remainders):
        if a == 0.0 and b > 0.0:
            # -0.0 - (b * +0.0) keeps the dividend's sign; trunc's -0.0 would not.
            assert remainder == _bits(a)


@pytest.mark.parametrize("lane", range(OUTPUT_COUNT))
def test_oracle_rejects_trunc_folded_zero_signs(lane):
    expected = _expected()
    pairs = _pairs()
    index = next(
        index
        for index, (a, b) in enumerate(pairs)
        if expected[index * OUTPUT_COUNT + lane] in {_bits(0.0), _bits(-0.0)}
    )
    changed = list(expected)
    changed[index * OUTPUT_COUNT + lane] ^= 0x80000000
    with pytest.raises(AssertionError):
        _check(changed, expected)


def test_opengl_routes_every_round_trip_through_the_exact_helper(tmp_path):
    generated = _translate(tmp_path, "opengl")
    for helper, parameter, result, zero in (
        ("crossgl_integer_float_int64_t", "int64_t", "float", None),
        ("crossgl_integer_float_int", "int", "float", None),
        ("crossgl_integer_float_uint", "uint", "float", None),
        ("crossgl_integer_float_ivec2", "ivec2", "vec2", "equal"),
    ):
        assert f"{result} {helper}({parameter} value) {{" in generated
        assert generated.count(f"{result} {helper}(") == 1
    assert (
        "uintBitsToFloat(floatBitsToUint(float(value)) & "
        "~(uint(value == int64_t(0)) << 31u))" in generated
    )
    assert (
        "uintBitsToFloat(floatBitsToUint(vec2(value)) & "
        "~(uvec2(equal(value, ivec2(0))) << 31u))" in generated
    )
    body = generated.split("void main()", 1)[1]
    assert "(b * crossgl_integer_float_int64_t(int64_t(quotient)))" in body, body
    assert "float truncated = crossgl_integer_float_int64_t(int64_t(quotient));" in body
    assert "float explicit_value = crossgl_integer_float_int(int(a));" in body
    assert "float assigned = crossgl_integer_float_int(int(a));" in body
    assert "crossgl_integer_float_ivec2(ivec2(vec2(a, b)))" in body
    assert "crossgl_integer_float_uint(uint(min(max(a, (-0.75)), 65536.0)))" in body
    assert "(a * crossgl_integer_float_int(int(b)))" in body
    assert "float(int" not in body and "float(uint" not in body


@pytest.mark.parametrize(
    "expression",
    (
        "float(index)",
        "float(int(index))",
        "float(count)",
        "float(uint(count))",
        "vec2(ivec2(index, count))",
    ),
)
def test_opengl_keeps_integer_values_without_float_sources_plain(expression):
    generated = _generate(f"""shader Plain {{
        float evaluate(uint index, int count, float value) {{
            int truncated = int(value);
            float through_variable = float(truncated);
            return vec2({expression}).x + through_variable;
        }}
    }}""")
    assert "crossgl_integer_float" not in generated
    assert "float through_variable = float(truncated);" in generated


@pytest.mark.parametrize("target", ("float", "vec2", "vec3", "vec4"))
def test_opengl_splats_and_matches_vector_widths(target):
    width = 1 if target == "float" else int(target[-1])
    integer = "int" if width == 1 else f"ivec{width}"
    source_value = "value" if width == 1 else f"{target}(value)"
    generated = _generate(f"""shader Widths {{
        {target} scalar(float value) {{ return {target}(int(value)); }}
        {target} vector(float value) {{ return {target}({integer}({source_value})); }}
    }}""")
    splat = (
        "crossgl_integer_float_int(int(value))"
        if width == 1
        else f"{target}(crossgl_integer_float_int(int(value)))"
    )
    assert f"return {splat};" in generated
    assert (
        f"return crossgl_integer_float_{integer}({integer}({source_value}));"
        in generated
    )


def test_opengl_integer_float_helper_names_avoid_user_functions():
    generated = _generate("""shader Names {
        float crossgl_integer_float_int(float value) { return value; }
        float evaluate(float value) {
            return crossgl_integer_float_int(value) + float(int(value));
        }
    }""")
    helper = re.search(r"float (crossgl_integer_float_int_\d+)\(int value\)", generated)
    assert helper is not None, generated
    assert "float crossgl_integer_float_int(float value)" in generated
    assert (
        f"crossgl_integer_float_int(value) + {helper.group(1)}(int(value))" in generated
    )


def test_opengl_double_and_global_round_trips_keep_plain_constructors():
    generated = _generate("""shader Plain {
        const float folded = float(int(-0.5));
        double wide(double value) { return double(int(value)); }
        float evaluate(float value) { return folded + float(wide(double(value))); }
    }""")
    assert "crossgl_integer_float" not in generated
    assert "const float folded = float(int((-0.5)));" in generated
    assert "double(int(value))" in generated


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_integer_float_round_trip_compiles(tmp_path, target):
    generated = _translate(tmp_path, target)
    if target == "opengl":
        assert "crossgl_integer_float_int64_t(" in generated
    _compile(generated, target, tmp_path)


def test_integer_float_round_trip_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native round trips")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected()
    assert len(expected) == OUTPUT_COUNT * len(_pairs()) + len(GUARD)
    generated = _translate(tmp_path, target)
    records = {}
    for label, source in (("generated", generated), ("original", SOURCE)):
        if label == "original" and target != "metal":
            continue
        records[label] = _execute(
            tmp_path / label,
            target,
            source,
            inputs,
            expected,
            metal_entry="integer_float",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
            invocations=len(_pairs()),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "pairCount": len(_pairs()),
                "outputCount": len(expected),
                "oracle": "binary32 RNE; integer conversion truncates and 0 -> +0.0",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_ci_requires_integer_float_round_trips_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_integer_float_round_trip.py" in step
    assert "pytest -q -n auto" in step
