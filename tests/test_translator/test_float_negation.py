"""Binary32 negation preserves payloads and evaluates its operand once."""

import json
import os
import random
import struct
import sys
from pathlib import Path

import pytest

import crosstl.translator
from crosstl import translate
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_FLOAT_NEGATION"
OUTPUT_COUNT = 16
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Scalar = float;
struct Payload { float value; };
Scalar record(thread uint& count, Scalar value) { count += 1; return value; }
float2 negate_pair(float2 value) { return -value; }
float3 negate_triple(float3 value) { return -value; }
float4 negate_quad(float4 value) { return -value; }
kernel void negate_words(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    Scalar x = as_type<float>(values[i]);
    Scalar negative = -x;
    float2 pair = negate_pair(float2(x));
    float3 triple = negate_triple(float3(x));
    float4 quad = negate_quad(float4(x));
    Payload payload = {x};
    float items[2] = {x, x};
    uint count = 0u;
    uint index = 0u;
    results[16u * i] = as_type<uint>(negative);
    results[16u * i + 1u] = as_type<uint>(pair.x);
    results[16u * i + 2u] = as_type<uint>(pair.y);
    results[16u * i + 3u] = as_type<uint>(triple.x);
    results[16u * i + 4u] = as_type<uint>(triple.y);
    results[16u * i + 5u] = as_type<uint>(triple.z);
    results[16u * i + 6u] = as_type<uint>(quad.x);
    results[16u * i + 7u] = as_type<uint>(quad.y);
    results[16u * i + 8u] = as_type<uint>(quad.z);
    results[16u * i + 9u] = as_type<uint>(quad.w);
    results[16u * i + 10u] = as_type<uint>(-record(count, x));
    results[16u * i + 11u] = as_type<uint>(-payload.value);
    results[16u * i + 12u] = as_type<uint>(-items[index++]);
    results[16u * i + 13u] = as_type<uint>(-(-x));
    results[16u * i + 14u] = count;
    results[16u * i + 15u] = index;
}
"""


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("alias", [False, True])
def test_directx_float_negation_uses_payload_sign_bit(width, alias):
    dtype = "float" if width == 1 else f"vec{width}"
    declarations = f"typedef {dtype} Value;" if alias else ""
    declared = "Value" if alias else dtype
    source = f"""shader Negation {{
        {declarations}
        {declared} evaluate({declared} value) {{ return -value; }}
    }}"""
    generated = HLSLCodeGen().generate(crosstl.translator.parse(source))
    mask = (
        "0x80000000u"
        if width == 1
        else f"uint{width}({', '.join(['0x80000000u'] * width)})"
    )
    assert f"return asfloat(asuint(value) ^ {mask});" in generated


@pytest.mark.parametrize("dtype", ["int", "uint", "double", "half", "ivec2", "mat2"])
def test_directx_non_binary32_negation_keeps_its_type(dtype):
    source = (
        f"shader Negation {{ {dtype} evaluate({dtype} value) {{ return -value; }} }}"
    )
    generated = HLSLCodeGen().generate(crosstl.translator.parse(source))
    assert "return -value;" in generated
    assert "0x80000000u" not in generated


def test_directx_nested_float_negation_does_not_form_decrement():
    source = "shader Negation { float evaluate(float value) { return -(-value); } }"
    generated = HLSLCodeGen().generate(crosstl.translator.parse(source))
    assert (
        "return asfloat(asuint(asfloat(asuint(value) ^ 0x80000000u)) ^ 0x80000000u);"
        in generated
    )
    assert "--value" not in generated


def test_directx_negative_literal_remains_a_constant_expression():
    source = "shader Negation { float evaluate() { return -1.5; } }"
    generated = HLSLCodeGen().generate(crosstl.translator.parse(source))
    assert "return -1.5;" in generated
    assert "0x80000000u" not in generated


@pytest.mark.parametrize("dtype", ["float", "vec2", "vec3", "vec4"])
def test_directx_logical_not_is_not_classified_as_floating_negation(dtype):
    source = f"""shader Negation {{
        {dtype} evaluate({dtype} value) {{ return -(!value); }}
    }}"""
    generated = HLSLCodeGen().generate(crosstl.translator.parse(source))
    assert "return -!value;" in generated
    assert "0x80000000u" not in generated


def test_float_negation_survives_saved_intermediate(tmp_path):
    original = tmp_path / "negate.metal"
    original.write_text(SOURCE, encoding="utf-8")
    intermediate = tmp_path / "negate.cgl"
    intermediate.write_text(
        translate(str(original), backend="cgl", format_output=False), encoding="utf-8"
    )
    direct = translate(str(original), backend="directx", format_output=False)
    assert (
        translate(str(intermediate), backend="directx", format_output=False) == direct
    )
    assert "asfloat(asuint(record(count, x)) ^ 0x80000000u)" in direct
    assert direct.count("record(count, x)") == 1
    assert "asfloat(asuint(items[index++]) ^ 0x80000000u)" in direct
    assert direct.count("index++") == 1


def _inputs():
    words = {
        (exponent << 23) | fraction | sign
        for exponent in range(256)
        for fraction in (0, 1, 0x3FFFFF, 0x400000, 0x7FFFFF)
        for sign in (0, 0x80000000)
    }
    rng = random.Random(1997)
    words.update(rng.getrandbits(32) for _ in range(256))
    return sorted(words)


def _expected(inputs):
    result = []
    for word in inputs:
        result.extend([word ^ 0x80000000] * 13 + [word, 1, 1])
    return result + GUARD


def _check(actual, expected):
    assert len(actual) == len(expected)
    assert actual == expected, [
        (index, hex(wanted), hex(found))
        for index, (found, wanted) in enumerate(zip(actual, expected))
        if found != wanted
    ][:20]
    return 0


@pytest.mark.parametrize("lane", [0, 1, 3, 6, 10, 11, 12, 13, 14, 15, 16])
def test_float_negation_oracle_rejects_changed_payloads_and_counters(lane):
    expected = _expected([0x007FFFFF])
    changed = list(expected)
    changed[lane] ^= 1
    with pytest.raises(AssertionError):
        _check(changed, expected)


def test_float_negation_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native float negation")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    inputs = _inputs()
    expected = _expected(inputs)
    assert len(expected) == OUTPUT_COUNT * len(inputs) + len(GUARD)
    for name, words in (("inputs", inputs), ("expected", expected)):
        (tmp_path / f"{name}.bin").write_bytes(struct.pack(f"<{len(words)}I", *words))
    original = tmp_path / "negate.metal"
    original.write_text(SOURCE, encoding="utf-8")
    generated = translate(str(original), backend=target, format_output=False)
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
            metal_entry="negate_words",
            check_outputs=_check,
            metal_compile_flags=("-fno-fast-math",),
        )
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "inputCount": len(inputs),
                "outputCount": len(expected),
                "oracle": "Exact uint32 sign-bit toggle; no floating-point arithmetic",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_ci_requires_float_negation_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_float_negation.py" in step
    assert "pytest -q -n auto" in step
