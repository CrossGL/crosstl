"""Floating remainder independently checked with exact rational arithmetic."""

import json
import os
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_fused_math import _dispatch

REQUIRE_ENV = "CROSTL_REQUIRE_FLOATING_REMAINDER"
GUARD = 0x1937A5C3


def _oracle(a, b, bits):
    fraction_bits, exponent_bits = (23, 8) if bits == 32 else (52, 11)
    sign_mask = 1 << (bits - 1)
    infinity = ((1 << exponent_bits) - 1) << fraction_bits
    x, y = a & (sign_mask - 1), b & (sign_mask - 1)
    sign = a & sign_mask
    if x >= infinity or y > infinity or y == 0:
        return infinity | (1 << (fraction_bits - 1))
    if x < y:
        return a
    if x == y:
        return sign

    def exact(value):
        exponent = value >> fraction_bits
        significand = value & ((1 << fraction_bits) - 1)
        if exponent:
            significand |= 1 << fraction_bits
        bias = (1 << (exponent_bits - 1)) - 1
        power = (exponent or 1) - bias - fraction_bits
        return Fraction(significand) * Fraction(2) ** power

    left, right = exact(x), exact(y)
    remainder = left - (left // right) * right
    real_format, word_format = ("f", "I") if bits == 32 else ("d", "Q")
    encoded = struct.unpack(
        "<" + word_format, struct.pack("<" + real_format, float(remainder))
    )[0]
    return sign | encoded


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (0xC0B00000, 0x40000000, 0xBFC00000),
        (0x40B00000, 0xC0000000, 0x3FC00000),
        (0x7F000000, 0x3F000000, 0),
        (0xFF000000, 0x3F000000, 0x80000000),
        (0x3F800000, 0x7F800000, 0x3F800000),
        (0x80000000, 0x40000000, 0x80000000),
        (0x3F800000, 0, 0x7FC00000),
        (0x7F800000, 0x40000000, 0x7FC00000),
        (3, 2, 1),
        (0x00800000, 0x007FFFFF, 1),
        (0x80800000, 0x007FFFFF, 0x80000001),
    ],
)
def test_floating_remainder_oracle_boundaries(a, b, expected):
    assert _oracle(a, b, 32) == expected


def _pairs(bits):
    fraction_bits, exponent_bits = (23, 8) if bits == 32 else (52, 11)
    bias = (1 << (exponent_bits - 1)) - 1
    one = bias << fraction_bits
    infinity = ((1 << exponent_bits) - 1) << fraction_bits
    edges = [
        0,
        1,
        2,
        3,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        (1 << fraction_bits) + 1,
        one - 1,
        one,
        one + 1,
        one + (1 << fraction_bits),
        infinity - 1,
        infinity,
        infinity + 1,
        infinity | (1 << (fraction_bits - 1)),
    ]
    signed = [value | sign for value in edges for sign in (0, 1 << (bits - 1))]
    pairs = [(a, b) for a in signed for b in signed]
    step = 1 if bits == 32 else 17
    pairs.extend(
        ((exponent << fraction_bits) | fraction, divisor)
        for exponent in range(1, (1 << exponent_bits) - 1, step)
        for fraction in (0, 1, (1 << fraction_bits) - 1)
        for divisor in (one - 1, one + 1)
    )
    random_values = random.Random(2097 + bits)
    pairs.extend(
        (random_values.getrandbits(bits), random_values.getrandbits(bits))
        for _ in range(2048)
    )
    return pairs


def _source(bits):
    scalar = "float" if bits == 32 else "double"
    vector = "vec" if bits == 32 else "dvec"
    if bits == 32:
        load = (
            "float a = asfloat(values[2u * i]); float b = asfloat(values[2u * i + 1u]);"
        )
    else:
        load = "double a = packDouble2x32(uvec2(values[4u * i], values[4u * i + 1u])); double b = packDouble2x32(uvec2(values[4u * i + 2u], values[4u * i + 3u]));"
    expressions = [
        "scalar_result",
        "pair_result.x",
        "pair_result.y",
        "triple_result.x",
        "triple_result.y",
        "triple_result.z",
        "quad_result.x",
        "quad_result.y",
        "quad_result.z",
        "quad_result.w",
    ]
    stores = []
    stride = 10 if bits == 32 else 20
    for slot, value in enumerate(expressions):
        if bits == 32:
            stores.append(f"results[4u + {stride}u * i + {slot}u] = asuint({value});")
        else:
            stores.extend(
                [
                    f"results[4u + {stride}u * i + {2 * slot}u] = unpackDouble2x32({value}).x;",
                    f"results[4u + {stride}u * i + {2 * slot + 1}u] = unpackDouble2x32({value}).y;",
                ]
            )
    return f"""shader FloatingRemainder {{
        StructuredBuffer<uint> values @ binding(0);
        RWStructuredBuffer<uint> results @ binding(1);
        compute {{
            layout(local_size_x = 1) in;
            void main(uvec3 tid @ gl_GlobalInvocationID) @ stage_entry {{
                uint i = tid.x;
                {load}
                {scalar} scalar_result = fmod(a, b);
                {vector}2 pair_result = fmod({vector}2(a, -a), {vector}2(b, -b));
                {vector}3 triple_result = fmod({vector}3(a, -a, a), b);
                {vector}4 quad_result = fmod(a, {vector}4(b, -b, b, -b));
                {' '.join(stores)}
            }}
        }}
    }}"""


@pytest.mark.parametrize("bits", [32, 64])
def test_floating_remainder_executes_on_opengl(tmp_path, bits):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native floating remainder")
    if sys.platform != "linux":
        pytest.skip("The OpenGL execution control uses Linux/EGL")
    pairs = _pairs(bits)
    sign = 1 << (bits - 1)
    words = []
    for a, b in pairs:
        normal, negative = _oracle(a, b, bits), _oracle(a ^ sign, b, bits)
        values = [
            normal,
            normal,
            negative,
            normal,
            negative,
            normal,
            normal,
            normal,
            normal,
            normal,
        ]
        for value in values:
            words.extend([value] if bits == 32 else [value & 0xFFFFFFFF, value >> 32])
    expected = [GUARD] * 4 + words + [GUARD] * 4
    inputs = (
        pairs
        if bits == 32
        else [(a & 0xFFFFFFFF, a >> 32, b & 0xFFFFFFFF, b >> 32) for a, b in pairs]
    )
    generated = GLSLCodeGen().generate_stage(parse(_source(bits)), "compute")
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        "opengl",
        generated,
        inputs,
        len(expected),
        initial_output=[GUARD] * len(expected),
    )
    assert len(actual) == len(expected)
    assert actual[:4] == actual[-4:] == [GUARD] * 4
    magnitude_mask = (1 << (bits - 1)) - 1
    infinity = 0x7F800000 if bits == 32 else 0x7FF0000000000000
    mismatches = []
    for index in range(4, len(expected) - 4, bits // 32):
        want, got = expected[index], actual[index]
        if bits == 64:
            want |= expected[index + 1] << 32
            got |= actual[index + 1] << 32
        both_nan = want & magnitude_mask > infinity and got & magnitude_mask > infinity
        if want != got and not both_nan:
            mismatches.append({"index": index, "expected": want, "actual": got})
    evidence.update(
        bits=bits,
        pairCount=len(pairs),
        lanes=[1, 2, 3, 4],
        scalarBroadcast=True,
        oracle="exact positive rational quotient and remainder",
        nanComparison="classification-only",
        guardCount=8,
        mismatchCount=len(mismatches),
        mismatches=mismatches,
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert not mismatches, mismatches[:10]


def test_ci_requires_floating_remainder_in_existing_math_job():
    from tools import ci_coverage

    path = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    )
    workflow = path.read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate OpenGL Metal math semantics"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_floating_remainder_math.py" in step
    assert "pytest -q -n auto" in step
    assert workflow.count("tests/test_translator/test_floating_remainder_math.py") == 1
