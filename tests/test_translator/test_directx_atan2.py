"""Preserve source atan2 branch selection independently of HLSL atan2."""

import json
import math
import os
import random
import re
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.project.native_runtime_drivers import DirectXComputeRuntime
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)
from tests.test_translator.test_metal_builtin_ownership import _compile, _run

REQUIRE_ENV = "CROSTL_REQUIRE_DIRECTX_ATAN2"
REQUIRE_METAL_ENV = "CROSTL_REQUIRE_METAL_ATAN2"
SOURCE = """#include <metal_stdlib>
using namespace metal;
float record(thread uint& count, float value) { count += 1; return value; }
kernel void angles(device const uint* bits [[buffer(0)]],
                   device const float* values [[buffer(1)]],
                   device uint* results [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    float y = as_type<float>(bits[2 * i]);
    float x = as_type<float>(bits[2 * i + 1]);
    uint count = 0;
    float2 pair = metal::atan2(float2(y, x), float2(x, y));
    float3 triple = metal::precise::atan2(float3(y, y, -y), float3(x));
    float4 quad = metal::atan2(float4(record(count, y), x, y, x), float4(x, y, x, y));
    results[16 * i] = as_type<uint>(y);
    results[16 * i + 1] = as_type<uint>(x);
    results[16 * i + 2] = as_type<uint>(values[2 * i]);
    results[16 * i + 3] = as_type<uint>(values[2 * i + 1]);
    results[16 * i + 4] = as_type<uint>(metal::atan2(y, x));
    results[16 * i + 5] = as_type<uint>(metal::atan2(values[2 * i], values[2 * i + 1]));
    results[16 * i + 6] = as_type<uint>(pair.x);
    results[16 * i + 7] = as_type<uint>(pair.y);
    results[16 * i + 8] = as_type<uint>(triple.x);
    results[16 * i + 9] = as_type<uint>(triple.y);
    results[16 * i + 10] = as_type<uint>(triple.z);
    results[16 * i + 11] = as_type<uint>(quad.x);
    results[16 * i + 12] = as_type<uint>(quad.y);
    results[16 * i + 13] = as_type<uint>(quad.z);
    results[16 * i + 14] = as_type<uint>(quad.w);
    results[16 * i + 15] = count;
}
"""


def _translate(tmp_path, source=SOURCE, suffix="metal"):
    path = tmp_path / f"angles.{suffix}"
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend="directx", format_output=False)


def test_atan2_scalar_and_vector_helpers_compile(tmp_path):
    generated = _translate(tmp_path)
    for suffix in ("", "2", "3", "4"):
        assert f"__crossgl_atan2_float{suffix}(" in generated
    assert not re.search(r"\batan2\s*\(", generated)
    assert "precise float reduced" in generated
    assert "precise float angle" in generated
    _compile(generated, "directx", tmp_path)


def test_hlsl_atan2_roundtrip_keeps_native_intrinsic(tmp_path):
    source = """StructuredBuffer<float> values : register(t0);
    RWStructuredBuffer<float> results : register(u0);
    [numthreads(1, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {
        results[id.x] = atan2(values[2 * id.x], values[2 * id.x + 1]);
    }"""
    generated = _translate(tmp_path, source, "hlsl")
    assert "__crossgl_atan2" not in generated
    assert "atan2(" in generated
    _compile(generated, "directx", tmp_path)


def test_atan2_user_overloads_keep_their_behavior(tmp_path):
    source = """float atan2(float y, float x) { return y + x + 10.0f; }
    kernel void angles(device float* results [[buffer(0)]]) {
        results[0] = metal::atan2(-0.0f, -4.0f);
        results[1] = ::atan2(-0.0f, -4.0f);
    }"""
    generated = _translate(tmp_path, source)
    assert "atan2__metal_overload_1(" in generated
    assert "__crossgl_atan2_float(" in generated
    _compile(generated, "directx", tmp_path)


def test_atan2_helper_avoids_source_identifiers(tmp_path):
    source = """shader S {
        float __crossgl_atan2_float(float x) { return x; }
        float result(float y, float x) {
            float __crossgl_atan2_float_ = 2.0;
            return atan2(y, x) + __crossgl_atan2_float_;
        }
    }"""
    generated = HLSLCodeGen().generate(parse(source))
    assert "float __crossgl_atan2_float__(float y, float x)" in generated


@pytest.mark.parametrize(
    "value_type,mapped",
    [("half", "float16_t"), ("float16_t", "float16_t"), ("min16float", "min16float")],
)
@pytest.mark.parametrize("suffix", ["", "2", "3", "4"])
def test_atan2_narrow_operands_promote_then_narrow_once(value_type, mapped, suffix):
    value_type += suffix
    mapped += suffix
    wide = "float" + suffix
    source = f"""shader S {{
        {value_type} f({value_type} y, {value_type} x) {{ return atan2(y, x); }}
    }}"""
    generated = HLSLCodeGen().generate(parse(source))
    assert (
        f"return {mapped}(__crossgl_atan2_{wide}({wide}(y), {wide}(x)));" in generated
    )


@pytest.mark.parametrize(
    "source,reason",
    [
        ("float f(float y) { return atan2(y); }", "atan2-invalid-arity"),
        ("float f() { return atan2(missing, 1.0); }", "atan2-operand-unresolved"),
        (
            "float f(float y, vec2 x) { return atan2(y, x); }",
            "atan2-operand-shape-mismatch",
        ),
        (
            "double f(double y, double x) { return atan2(y, x); }",
            "atan2-operand-type-unsupported",
        ),
        (
            "int f(int y, int x) { return atan2(y, x); }",
            "atan2-operand-type-unsupported",
        ),
        (
            "float asfloat; float f(float y, float x) { return atan2(y, x); }",
            "atan2-target-intrinsic-shadowed",
        ),
        (
            "float atan2; float f(float y, float x) { return atan2(y, x); }",
            "atan2-target-intrinsic-shadowed",
        ),
    ],
)
def test_atan2_rejects_unproven_contracts(source, reason):
    with pytest.raises(DirectXContextualConversionError) as error:
        HLSLCodeGen().generate(parse(f"shader S {{ {source} }}"))
    assert error.value.reason == reason


def test_atan2_helpers_reset_when_codegen_is_reused():
    generator = HLSLCodeGen()
    assert "__crossgl_atan2" in generator.generate(
        parse("shader S { float f(float y, float x) { return atan2(y, x); } }")
    )
    assert "__crossgl_atan2" not in generator.generate(
        parse("shader S { float f(float y) { return y; } }")
    )


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def _pairs():
    edges = (
        0,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x40800000,
        0xC0800000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
    )
    pairs = [(y, x) for y in edges for x in edges]
    random_source = random.Random(1973)
    pairs.extend(
        tuple(_bits(random_source.uniform(-100, 100)) for _ in range(2))
        for _ in range(128)
    )
    # Cover exponent scaling, range-reduction boundaries and all four quadrants.
    finite = (
        0x00800000,
        0x00FFFFFF,
        _bits(math.sqrt(2) - 1) - 1,
        _bits(math.sqrt(2) - 1),
        _bits(math.sqrt(2) - 1) + 1,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0x7F000000,
        0x7F7FFFFF,
    )
    pairs.extend(
        (y | y_sign, x | x_sign)
        for y in finite
        for x in finite
        for y_sign in (0, 0x80000000)
        for x_sign in (0, 0x80000000)
    )
    pairs.extend(
        tuple(
            random_source.randrange(0x00800000, 0x7F800000)
            | random_source.choice((0, 0x80000000))
            for _ in range(2)
        )
        for _ in range(512)
    )
    return pairs


def _check_outputs(pairs, outputs):
    assert len(outputs) == 16 * len(pairs)
    maximum_error = 0.0
    finite_count = axis_count = nan_count = 0
    for index, (y_bits, x_bits) in enumerate(pairs):
        row = outputs[16 * index : 16 * index + 16]
        # The float and integer uploads must retain the same bits, including -0.
        assert row[:4] == [y_bits, x_bits, y_bits, x_bits], (index, row[:4])
        assert row[15] == 1, "A source operand was evaluated more than once"
        y, x = _float(y_bits), _float(x_bits)
        operands = [
            (y, x),
            (y, x),
            (y, x),
            (x, y),
            (y, x),
            (y, x),
            (-y, x),
            (y, x),
            (x, y),
            (y, x),
            (x, y),
        ]
        for bits, (left, right) in zip(row[4:15], operands):
            expected = math.atan2(left, right)
            actual = _float(bits)
            if math.isnan(expected):
                assert math.isnan(actual)
                nan_count += 1
            elif left == 0 or right == 0 or math.isinf(left) or math.isinf(right):
                assert bits == _bits(expected), (
                    index,
                    left,
                    right,
                    bits,
                    _bits(expected),
                )
                axis_count += 1
            else:
                assert abs(actual - expected) <= 2e-6, (index, actual, expected)
                maximum_error = max(maximum_error, abs(actual - expected))
                finite_count += 1
    return {
        "pairCount": len(pairs),
        "angleCount": 11 * len(pairs),
        "finiteAngleCount": finite_count,
        "exactAxisAngleCount": axis_count,
        "nanAngleCount": nan_count,
        "maximumFiniteAbsoluteError": maximum_error,
        "finiteAbsoluteErrorBound": 2e-6,
        "inputBitsPreserved": True,
        "singleOperandEvaluation": True,
    }


def test_numerical_check_rejects_windows_finite_angle_error():
    pairs = [(_bits(1.0), _bits(1.0))]
    outputs = [_bits(1.0)] * 4 + [_bits(math.pi / 4)] * 11 + [1]
    outputs[10] = _bits(-math.pi / 4)
    evidence = _check_outputs(pairs, outputs)
    assert evidence["finiteAngleCount"] == evidence["angleCount"] == 11
    assert evidence["maximumFiniteAbsoluteError"] < 2e-6
    outputs[4] = _bits(0.7854096293449402)
    with pytest.raises(AssertionError):
        _check_outputs(pairs, outputs)


def test_directx_atan2_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native DirectX atan2")
    assert sys.platform == "win32", "The DirectX numerical gate requires Windows"
    generated = _translate(tmp_path)
    artifact, module = _compile(generated, "directx", tmp_path)
    assert module.is_file(), "DXC is required"
    pairs = _pairs()
    bits = [value for pair in pairs for value in pair]
    buffers = {}
    for name, binding, dtype, values, count in (
        ("bits", 0, "uint32", bits, len(bits)),
        ("values", 1, "float32", [_float(value) for value in bits], len(bits)),
        ("results", 2, "uint32", None, 16 * len(pairs)),
    ):
        output = values is None
        scalar = "float" if dtype == "float32" else "uint"
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=binding,
                type_name=("RW" if output else "") + f"StructuredBuffer<{scalar}>",
                access="read_write" if output else "read",
            ),
            source="expectedOutput" if output else "input",
            dtype=dtype,
            shape=(count,),
            value=values,
        )
    request = NativeRuntimeDispatchRequest(
        target="directx",
        artifact={"target": "directx"},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point="CSMain",
        dispatch=RuntimeDispatchGeometry(
            entry_point="CSMain",
            workgroup_size=(1, 1, 1),
            workgroup_count=(len(pairs), 1, 1),
        ),
    )
    state = SimpleNamespace(details={})
    (tmp_path / "input-bits.json").write_text(json.dumps(pairs), encoding="utf-8")
    try:
        output = DirectXComputeRuntime().dispatch(None, state, request)["results"][
            "values"
        ]
    except Exception as error:
        (tmp_path / "runtime-error.json").write_text(
            json.dumps(
                {
                    "message": str(error),
                    "details": getattr(error, "details", {}),
                    "runtime": state.details,
                },
                indent=2,
            )
        )
        raise
    (tmp_path / "readback-bits.json").write_text(json.dumps(output), encoding="utf-8")
    (tmp_path / "runtime.json").write_text(
        json.dumps(state.details, indent=2), encoding="utf-8"
    )
    evidence = _check_outputs(pairs, output)
    (tmp_path / "numerical-summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


def test_original_metal_atan2_executes(tmp_path):
    if os.environ.get(REQUIRE_METAL_ENV) != "1":
        pytest.skip(f"set {REQUIRE_METAL_ENV}=1 for the original Metal control")
    assert sys.platform == "darwin", "The source control requires Metal"
    source = tmp_path / "original.metal"
    source.write_text(SOURCE, encoding="utf-8")
    air, library = tmp_path / "original.air", tmp_path / "original.metallib"
    # Fast math permits ignoring signed zero and non-finite operands.
    _run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-fno-fast-math",
            "-c",
            str(source),
            "-o",
            str(air),
        ]
    )
    _run(["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(library)])
    runner = tmp_path / "readback"
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/runtime_verification/metal_uint32_buffers.swift"
    )
    _run(["swiftc", str(fixture), "-o", str(runner)])
    pairs = _pairs()
    bits = [value for pair in pairs for value in pair]
    request = tmp_path / "input-bits.json"
    request.write_text(
        json.dumps(
            {
                "inputs": [bits, bits],
                "outputCount": 16 * len(pairs),
                "threadCount": len(pairs),
            }
        ),
        encoding="utf-8",
    )
    output = _run([str(runner), str(library), "angles", str(request)])
    (tmp_path / "readback-bits.json").write_text(output, encoding="utf-8")
    evidence = _check_outputs(pairs, json.loads(output)["values"])
    (tmp_path / "numerical-summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
