"""Keep source functions distinct from generated arithmetic bitcasts."""

import json
import os
import struct
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from tests.test_translator.test_fused_math import REQUIRE_ENV, _dispatch
from tests.test_translator.test_metal_builtin_ownership import _compile

GUARD = 0x1937A5C3
PAIRS = [(1, 2), (2, 1), (4, 8), (8, 4), (16, 2)]
SOURCE = """#include <metal_stdlib>
using namespace metal;
using Bits = uint;
using Value = float;
Value asfloat(Bits value);
Value asfloat(Bits value) { return Value(value) + 100.0f; }
float2 asfloat(uint2 value) { return float2(value) + float2(100.0f); }
Bits asuint(Value value) { return Bits(value) + 200u; }
uint2 asuint(float2 value) { return uint2(value) + uint2(200u); }
int asint(uint value) { return int(value) + 300; }
namespace custom {
Value asfloat(Bits value) { return Value(value) + 400.0f; }
Bits asuint(Value value) { return Bits(value) + 500u; }
}
static float record(thread uint& count, float value) { count += 1; return value; }
kernel void computeMain(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    float a = float(values[2*i]);
    float b = float(values[2*i+1]);
    uint count = 0;
    float fused = fma(record(count, a), b, 1.0f);
    float2 fused_pair = metal::fma(float2(a, b), float2(b, a), float2(1.0f));
    float2 quotient = float2(a, b) / float2(b, a);
    half2 remainder = fmod(half2(a, b), half2(b, a));
    results[4 + 19*i] = as_type<uint>(fused);
    results[4 + 19*i+1] = as_type<uint>(fused_pair.x);
    results[4 + 19*i+2] = as_type<uint>(fused_pair.y);
    results[4 + 19*i+3] = as_type<uint>(a / b);
    results[4 + 19*i+4] = as_type<uint>(quotient.x);
    results[4 + 19*i+5] = as_type<uint>(quotient.y);
    results[4 + 19*i+6] = uint(as_type<ushort>(fmod(half(a), half(b))));
    results[4 + 19*i+7] = uint(as_type<ushort>(remainder.x));
    results[4 + 19*i+8] = uint(as_type<ushort>(remainder.y));
    results[4 + 19*i+9] = ::asuint(1.0f);
    results[4 + 19*i+10] = uint(::asfloat(0u));
    results[4 + 19*i+11] = uint(::asfloat(uint2(1, 2)).y);
    results[4 + 19*i+12] = ::asuint(float2(1.0f, 2.0f)).y;
    results[4 + 19*i+13] = uint(::asint(1u));
    results[4 + 19*i+14] = as_type<uint>(as_type<int>(0x80000000u));
    results[4 + 19*i+15] = uint(custom::asfloat(1u));
    results[4 + 19*i+16] = custom::asuint(1.0f);
    results[4 + 19*i+17] = count;
    results[4 + 19*i+18] = as_type<uint>(as_type<float>(0x3f800000u));
}
"""


def _translate(tmp_path, target, profile):
    path = tmp_path / "ownership.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return translate(
        str(path),
        backend=target,
        format_output=False,
        source_options={
            "binary32_fma_profile": profile,
            "binary32_division_profile": profile,
            "binary16_remainder_profile": "binary32-quotient",
        },
    )


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush"])
def test_source_bitcast_names_compile_with_arithmetic_profiles(
    tmp_path, target, profile
):
    intermediate = _translate(tmp_path, "crossgl", profile)
    for builtin in ("asfloat", "asuint", "asint"):
        assert f"{builtin}__metal_overload_" in intermediate
    generated = _translate(tmp_path, target, profile)
    _compile(
        generated,
        target,
        tmp_path,
        directx_compile_flags=("-enable-16bit-types",),
        metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
    )


def test_bitcast_transport_avoids_source_identifiers_and_resets():
    source = """
    uint asuint(float x) { return uint(x) + 200u; }
    uint asuint__metal_overload_1(float x) { return uint(x); }
    uint user(float x) {
        uint asuint__metal_overload_1_2 = 5u;
        return asuint(x) + asuint__metal_overload_1_2;
    }
    """
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    names = [function.name for function in ast.functions]
    converter = MetalToCrossGLConverter()
    generated = converter.generate(ast)
    assert "uint asuint__metal_overload_1_3(float x)" in generated
    assert "asuint__metal_overload_1_3(x)" in generated
    assert [function.name for function in ast.functions] == names
    unrelated = MetalParser(
        MetalLexer("float user(float x) { return x; }").tokenize()
    ).parse()
    assert "metal_overload" not in converter.generate(unrelated)
    assert converter.metal_source_overload_groups == {}


def _expected():
    def word(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    def half(value):
        return struct.unpack("<H", struct.pack("<e", value))[0]

    result = [GUARD] * 4
    for a, b in PAIRS:
        result.extend(
            [word(a * b + 1)] * 3
            + [word(a / b)] * 2
            + [word(b / a), half(a % b), half(a % b), half(b % a)]
            + [201, 100, 102, 202, 301, 0x80000000, 401, 501, 1, 0x3F800000]
        )
    return result + [GUARD] * 4


@pytest.mark.parametrize("profile", ["rne-gradual", "rne-flush", "source"])
def test_arithmetic_builtin_ownership_executes(tmp_path, monkeypatch, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native arithmetic ownership checks")
    original = profile == "source"
    if original and sys.platform != "darwin":
        pytest.skip("The unchanged Metal control requires macOS")
    target = {"win32": "directx", "darwin": "metal", "linux": "opengl"}[sys.platform]
    generated = SOURCE if original else _translate(tmp_path, target, profile)
    monkeypatch.setattr(
        "tests.test_translator.test_fused_math._compile",
        partial(
            _compile,
            directx_compile_flags=("-enable-16bit-types",),
            metal_compile_flags=("-std=metal3.1", "-fno-fast-math"),
        ),
    )
    expected = _expected()
    (tmp_path / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    actual, evidence = _dispatch(
        tmp_path,
        target,
        generated,
        PAIRS,
        len(expected),
        initial_output=[GUARD] * len(expected),
    )
    evidence.update(
        sourceProfile=profile,
        originalSource=original,
        valueCount=len(PAIRS) * 19,
        guardCount=8,
        numericalComparison="exact-words",
    )
    (tmp_path / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    assert actual == expected


def test_ci_requires_arithmetic_ownership_in_existing_math_job():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "mlx-metal-porting", "Validate binary32 arithmetic"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    test = "tests/test_translator/test_metal_arithmetic_ownership.py::test_arithmetic_builtin_ownership_executes"
    assert test in step
    assert workflow.count(test) == 1
