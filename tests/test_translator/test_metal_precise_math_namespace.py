"""Metal ``precise::`` math keeps its meaning through CrossGL.

MLX's unary kernels call ``metal::precise::{tan,sinh,cosh,tanh,asinh,atanh,
log2,log10}`` and compile with ``-fno-fast-math``.  Under that flag the plain
binary32 overloads equal the precise ones, but the binary16 overloads are
native half implementations (``tan(-0.0h)`` is ``+0``), so a round trip that
dropped the namespace changed half results.

The Metal frontend now spells such calls ``precise::<op>`` in CrossGL.  The
parser hands every backend the plain builtin and records the namespace in an
annotation.  The Metal backend writes ``metal::precise::`` again for half
operands.  The OpenGL backend reads it for the odd functions sinh, tanh,
asinh and atanh, whose GLSL builtins Mesa evaluates through exp and log: the
sign of zero is lost and ``tanh(NaN)`` is ``-1``.  Their helpers copy the
operand's sign bit and pass NaN through.  Two further Mesa folds are kept out
of OpenGL output: every float ``fma`` is computed into a ``precise`` result,
so it stays one fused operation, and ``x - x`` of one float identifier is a
``precise`` difference, so infinities and NaN still give NaN.

Mesa also folds ``0 * x`` to +0, ``0 / x`` to +0, ``x + 0`` to ``x`` and
``0 - x`` to ``-x`` when the constant zero comes from an inlined complex
operand (MLX's ArcTan multiplies by ``i``), even for ``precise`` operations.
Inside Metal free arithmetic operators on binary32, OpenGL output therefore
computes scalar ``+ - * /`` through helpers whose integer tests settle zero
operands.  Binary32 ``atan2`` settles NaN, signed zeros and infinities the
same way (Mesa drops the sign of a zero ``y``), and a binary32 local whose
only value is a sum and that is an equality operand, such as MLX's ``log1p``
``xp1``, is declared ``precise`` so ``(1 + x) == 1`` is not rewritten to
``x == 0``.

The DirectX backend reads the annotation for sinh and tanh, whose DXIL Hsin
and Htan lose the sign of zero on WARP, which also returns NaN for tanh once
exp(2|x|) overflows and 1 - 2^-24 for tanh(20).  Their helpers return NaN and
operands below 2^-12 unchanged, saturate tanh at the first binary32 operand
that rounds to one, and give the magnitude the operand's sign.  Optimized DXC
folds ``x - x`` to zero as well, so DirectX also computes it as a ``precise``
difference.
"""

import json
import math
import os
import struct
import sys

import pytest

from crosstl import translate
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.lexer import Lexer
from crosstl.translator.math_namespaces import (
    METAL_FAST_MATH_FUNCTIONS,
    METAL_MATH_NAMESPACE_ANNOTATION,
    METAL_PRECISE_MATH_FUNCTIONS,
    metal_fast_math_operation,
    metal_math_namespace,
    metal_precise_math_operation,
)
from crosstl.translator.parser import Parser
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_PRECISE_NAMESPACE"
METAL_FLAGS = ("-std=metal3.1", "-fno-fast-math")
OPERATIONS = ("tan", "sinh", "cosh", "tanh", "asinh", "atanh", "log2", "log10")
ODD_OPERATIONS = ("sinh", "tanh", "asinh", "atanh")
OUTPUTS_PER_INPUT = 2 * len(OPERATIONS)


def _metal_source():
    """One thread per input word: every operation on its half and float view.

    Results pass through typed locals, the way MLX's unary operators return
    them (the Metal backend does not type a builtin call nested directly in
    ``as_type``).
    """
    lines = []
    for index, operation in enumerate(OPERATIONS):
        prefix = "metal::precise" if index % 2 else "precise"
        lines.append(
            f"    half h{index} = {prefix}::{operation}(h);\n"
            f"    float x{index} = {prefix}::{operation}(x);\n"
            f"    results[base + {index}u] = uint(as_type<ushort>(h{index}));\n"
            f"    results[base + {len(OPERATIONS) + index}u] = "
            f"as_type<uint>(x{index});\n"
        )
    return """#include <metal_stdlib>
using namespace metal;

kernel void precise_math(device const uint* values [[buffer(0)]],
                         device uint* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    uint word = values[i];
    half h = as_type<half>(ushort(word & 0xffffu));
    float x = as_type<float>(word);
    uint base = %(stride)du * i;
""" % {"stride": OUTPUTS_PER_INPUT} + "".join(lines) + "}\n"


IEEE_SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void ieee_math(device const uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[3u * i]);
    float b = as_type<float>(values[3u * i + 1u]);
    float c = as_type<float>(values[3u * i + 2u]);
    float fused = fma(a, b, c);
    float residual = fma(a, b, c) - c;
    float difference = a - a;
    float4 lanes = fma(float4(a), float4(b), float4(c));
    results[4u * i] = as_type<uint>(fused);
    results[4u * i + 1u] = as_type<uint>(residual);
    results[4u * i + 2u] = as_type<uint>(difference);
    results[4u * i + 3u] = as_type<uint>(lanes.w);
}
"""

CROSSGL_SOURCE = """shader PreciseMath {
    compute {
        layout(std430, binding = 0) buffer Data { float data[]; };
        [numthreads(1, 1, 1)]
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            float x = data[gid.x];
            vec4 v = vec4(x, -x, 0.5 * x, x + 1.0);
            float s = PREFIXtan(x) + PREFIXsinh(x) + PREFIXcosh(x) + PREFIXtanh(x);
            s += PREFIXasinh(x) + PREFIXatanh(0.5 * x) + PREFIXlog2(x) + PREFIXlog10(x);
            vec4 w = PREFIXsinh(v) + PREFIXtanh(v);
            data[gid.x] = s + w.x + w.w;
        }
    }
}
"""

HALF_CROSSGL_SOURCE = """shader PreciseHalf {
    compute {
        layout(std430, binding = 0) buffer Data { float data[]; };
        [numthreads(1, 1, 1)]
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            float x = data[gid.x];
            float16 h = float16(x);
            f16vec2 hv = f16vec2(h, -h);
            float16 t = precise::tan(h);
            f16vec2 u = precise::sinh(hv);
            float16 p = tan(h);
            data[gid.x] = float(t) + float(u.y) + float(p) + precise::tan(x);
        }
    }
}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _float(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _half(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def _crossgl(prefix):
    return CROSSGL_SOURCE.replace("PREFIX", prefix)


def _calls(code):
    ast = Parser(Lexer(code).get_tokens()).parse()
    return [node for node in ast.walk() if isinstance(node, FunctionCallNode)]


def _translate_text(tmp_path, text, target, name):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def test_precise_math_operation_names():
    for operation in sorted(METAL_PRECISE_MATH_FUNCTIONS):
        assert metal_precise_math_operation(f"precise::{operation}") == operation
        assert metal_precise_math_operation(f"metal::precise::{operation}") == (
            operation
        )
        assert metal_precise_math_operation(f"::metal::precise::{operation}") == (
            operation
        )
        assert metal_precise_math_operation(operation) is None
        assert metal_precise_math_operation(f"metal::{operation}") is None
        assert metal_precise_math_operation(f"fast::{operation}") is None
        assert metal_math_namespace(f"precise::{operation}") == ("precise", operation)
    for operation in sorted(METAL_FAST_MATH_FUNCTIONS):
        assert metal_math_namespace(f"fast::{operation}") == ("fast", operation)
        assert metal_fast_math_operation(f"precise::{operation}") is None
    # precise:: functions with portable helpers keep their own lowering.
    assert metal_precise_math_operation("precise::sin") is None
    assert metal_precise_math_operation("precise::tanh::inner") is None
    assert metal_math_namespace("tanh") is None
    assert metal_math_namespace("precisetanh") is None


def test_crossgl_parser_normalizes_precise_calls():
    precise = _calls(_crossgl("precise::"))
    plain = _calls(_crossgl(""))
    qualified = _calls(_crossgl("metal::precise::"))
    assert len(precise) == len(plain) == len(qualified)
    math_calls = [
        call for call in precise if call.function.name in METAL_PRECISE_MATH_FUNCTIONS
    ]
    assert len(math_calls) == 10
    for call in math_calls:
        assert type(call.function).__name__ == "IdentifierNode"
        assert call.get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) == "precise"
    for call in qualified:
        if call.function.name in METAL_PRECISE_MATH_FUNCTIONS:
            assert call.get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) == "precise"
    for call in plain:
        assert METAL_MATH_NAMESPACE_ANNOTATION not in call.annotations
    assert [call.function.name for call in precise] == [
        call.function.name for call in plain
    ]
    (unknown,) = _calls(
        "shader S { compute { void main() { float x = precise::erf(1.0); } } }"
    )
    assert unknown.function.name == "precise::erf"
    assert METAL_MATH_NAMESPACE_ANNOTATION not in unknown.annotations


@pytest.mark.parametrize(
    "target",
    ["vulkan", "cuda", "hip", "rust", "mojo", "slang", "webgpu"],
)
def test_other_targets_ignore_the_precise_namespace(tmp_path, target):
    outputs = [
        _translate_text(tmp_path, _crossgl(prefix), target, f"{label}.cgl")
        for label, prefix in (
            ("plain", ""),
            ("precise", "precise::"),
            ("qualified", "metal::precise::"),
        )
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    assert "precise::" not in outputs[0]


def test_metal_backend_restores_the_namespace_for_half_operands(tmp_path):
    generated = _translate_text(tmp_path, HALF_CROSSGL_SOURCE, "metal", "h.cgl")
    assert "metal::precise::tan(h)" in generated
    # metal::precise:: has binary32 overloads only: a half scalar converts to
    # float implicitly, a half vector does not and keeps the plain overload.
    assert "metal::precise::sinh(hv)" not in generated
    assert "sinh(hv)" in generated
    # The unannotated half call and the binary32 call stay plain: under
    # -fno-fast-math plain binary32 math is the precise implementation.
    assert "half p = tan(h);" in generated
    assert "metal::precise::tan(x)" not in generated
    assert "tan(x)" in generated
    _compile(generated, "metal", tmp_path, metal_compile_flags=METAL_FLAGS)


def test_metal_backend_keeps_binary32_precise_calls_plain(tmp_path):
    precise = _translate_text(tmp_path, _crossgl("precise::"), "metal", "p.cgl")
    plain = _translate_text(tmp_path, _crossgl(""), "metal", "plain.cgl")
    assert precise == plain
    assert "precise::" not in precise


def test_metal_frontend_marks_precise_calls_and_round_trips_them(tmp_path):
    source = _metal_source()
    crossgl = _translate_text(tmp_path, source, "crossgl", "precise.metal")
    for operation in OPERATIONS:
        assert f"precise::{operation}(h)" in crossgl
        assert f"precise::{operation}(x)" in crossgl
        assert f"metal::precise::{operation}(" not in crossgl
    generated = _translate_text(tmp_path, source, "metal", "round.metal")
    for operation in OPERATIONS:
        assert f"metal::precise::{operation}(h)" in generated
        assert f"metal::precise::{operation}(x)" not in generated
        assert f"{operation}(x)" in generated
    _compile(generated, "metal", tmp_path, metal_compile_flags=METAL_FLAGS)


def _kernel(body):
    """Wrap ``body`` in a one-buffer Metal kernel with ``float x = data[i]``."""
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "typedef bfloat bfloat16_t;\n\n"
        "kernel void probe(device float* data [[buffer(0)]],\n"
        "                  uint i [[thread_position_in_grid]]) {\n"
        "    float x = data[i];\n" + body + "}\n"
    )


ODD_KERNEL = _kernel("""    float4 v = float4(x, -x, 0.5f * x, x + 1.0f);
    float s = precise::tan(x) + precise::sinh(x) + precise::cosh(x);
    s += metal::precise::tanh(x) + precise::asinh(x) + precise::atanh(0.5f * x);
    s += precise::log2(x) + metal::precise::log10(x);
    float4 w = precise::sinh(v) + metal::precise::tanh(v);
    float p = sinh(x) + tanh(x);
    data[i] = s + w.x + w.w + p;
""")


def test_opengl_odd_precise_functions_keep_sign_and_nan(tmp_path):
    generated = _translate_text(tmp_path, ODD_KERNEL, "opengl", "odd.metal")
    plain = _translate_text(
        tmp_path,
        ODD_KERNEL.replace("metal::precise::", "").replace("precise::", ""),
        "opengl",
        "plain.metal",
    )
    for operation in ODD_OPERATIONS:
        helper = f"crossgl_precise_{operation}_float"
        assert f"float {helper}(float value) {{" in generated
        assert (
            f"uint magnitude = floatBitsToUint({operation}(value)) & 0x7fffffffu;"
            in generated
        )
        assert f"{helper}(" in generated.split("void main()", 1)[1]
        assert f"crossgl_precise_{operation}" not in plain
    for operation in ("sinh", "tanh"):
        assert f"vec4 crossgl_precise_{operation}_vec4(vec4 value) {{" in generated
        assert f"crossgl_precise_{operation}_vec4(v)" in generated
    assert "mix(result, value, isnan(value))" in generated
    assert "isnan(value) ? value : result" in generated
    assert "magnitude | (floatBitsToUint(value) & 0x80000000u)" in generated
    # Unannotated calls keep the builtins, as do even functions and functions
    # without the Mesa defect.
    assert "float p = (sinh(x) + tanh(x));" in generated
    for operation in ("tan", "cosh", "log2", "log10"):
        assert f"crossgl_precise_{operation}_" not in generated
    for operation in ("tan", "cosh", "log2"):
        assert f"{operation}(x)" in generated
    _compile(generated, "opengl", tmp_path)


def test_opengl_precise_helpers_apply_to_narrow_float_operands(tmp_path):
    source = _kernel("""    half h = half(x);
    bfloat16_t b = bfloat16_t(x);
    half t = precise::sinh(h);
    bfloat16_t u = metal::precise::tanh(b);
    data[i] = float(t) + float(u);
""")
    generated = _translate_text(tmp_path, source, "opengl", "narrow.metal")
    body = generated.split("void main()", 1)[1]
    assert "crossgl_precise_sinh_float(h)" in body
    assert "crossgl_precise_tanh_float(b)" in body
    _compile(generated, "opengl", tmp_path)


def test_opengl_fma_is_one_precise_operation(tmp_path):
    source = _kernel("""    float3 v = float3(x, x + 1.0f, x + 2.0f);
    float s = fma(x, 2.0f, x) - x;
    float3 w = metal::fma(v, v, v);
    data[i] = s + w.z;
""")
    generated = _translate_text(tmp_path, source, "opengl", "fma.metal")
    assert (
        "float crossgl_fma_float(float left, float right, float addend) {\n"
        "    precise float result = fma(left, right, addend);\n"
        "    return result;\n}\n"
    ) in generated
    assert "precise vec3 result = fma(left, right, addend);" in generated
    body = generated.split("void main()", 1)[1]
    assert "crossgl_fma_float(x, 2.0, x) - x" in body
    assert "crossgl_fma_vec3(v, v, v)" in body
    assert " fma(" not in body
    _compile(generated, "opengl", tmp_path)


def test_opengl_fma_rewrite_leaves_user_functions_alone(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;

float fused(float a, float b, float c) { return a * b + c; }

kernel void probe(device float* data [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
    float x = data[i];
    data[i] = fused(x, x, x);
}
"""
    generated = _translate_text(tmp_path, source, "opengl", "user.metal")
    assert "crossgl_fma_" not in generated
    assert "crossgl_difference_" not in generated


def test_opengl_identical_float_difference_is_precise(tmp_path):
    source = _kernel("""    float y = x + 1.0f;
    float2 v = float2(x, y);
    int k = int(i);
    float d = x - x;
    float e = x - y;
    float2 f = v - v;
    int g = k - k;
    data[i] = d + e + f.y + float(g);
""")
    generated = _translate_text(tmp_path, source, "opengl", "difference.metal")
    assert (
        "float crossgl_difference_float(float left, float right) {\n"
        "    precise float result = left - right;\n"
        "    return result;\n}\n"
    ) in generated
    body = generated.split("void main()", 1)[1]
    assert "float d = crossgl_difference_float(x, x);" in body
    assert "vec2 f = crossgl_difference_vec2(v, v);" in body
    assert "float e = (x - y);" in body
    assert "int g = (k - k);" in body
    _compile(generated, "opengl", tmp_path)


# MLX's complex_t operators (complex.h) and binary32 log1p (utils.h), applied
# to the constants that MLX's ArcTan multiplies by: i and a zero imaginary part.
FREE_OPERATOR_KERNEL = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct pair_t {
  T real;
  T imag;
  constexpr pair_t(T real, T imag) thread : real(real), imag(imag) {};
};

template <typename T>
constexpr pair_t<T> operator+(pair_t<T> a, pair_t<T> b) {
  return {a.real + b.real, a.imag + b.imag};
}

template <typename T>
constexpr pair_t<T> operator-(T a, pair_t<T> b) {
  return {a - b.real, -b.imag};
}

template <typename T>
constexpr pair_t<T> operator*(pair_t<T> a, pair_t<T> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

template <typename T>
constexpr pair_t<T> operator/(pair_t<T> a, pair_t<T> b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

inline float mlx_log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == 1.0f) {
    return x;
  }
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

kernel void ieee_zeros(device const uint* values [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[2u * i]);
    float b = as_type<float>(values[2u * i + 1u]);
    pair_t<float> z = pair_t<float>(a, b);
    pair_t<float> unit = pair_t<float>(0.0f, 1.0f);
    pair_t<float> zero = pair_t<float>(0.0f, 0.0f);
    pair_t<float> p = unit * z;
    pair_t<float> q = z / unit;
    pair_t<float> r = 0.0f - z;
    pair_t<float> s = z + zero;
    float t = metal::atan2(b, a);
    float l = mlx_log1p(a);
    float plain = a * 0.0f + b;
    uint base = 11u * i;
    results[base] = as_type<uint>(p.real);
    results[base + 1u] = as_type<uint>(p.imag);
    results[base + 2u] = as_type<uint>(q.real);
    results[base + 3u] = as_type<uint>(q.imag);
    results[base + 4u] = as_type<uint>(r.real);
    results[base + 5u] = as_type<uint>(r.imag);
    results[base + 6u] = as_type<uint>(s.real);
    results[base + 7u] = as_type<uint>(s.imag);
    results[base + 8u] = as_type<uint>(t);
    results[base + 9u] = as_type<uint>(l);
    results[base + 10u] = as_type<uint>(plain);
}
"""
FREE_OPERATOR_OUTPUTS = 11


def test_opengl_free_operators_settle_zero_operands_exactly(tmp_path):
    generated = _translate_text(
        tmp_path, FREE_OPERATOR_KERNEL, "opengl", "operators.metal"
    )
    for operation, operator in (
        ("sum", "+"),
        ("difference", "-"),
        ("product", "*"),
        ("quotient", "/"),
    ):
        helper = generated.split(
            f"float crossgl_ieee_{operation}_float(float left, float right) {{\n", 1
        )[1].split("\n}\n", 1)[0]
        # Integer tests of the operand bits survive Mesa's constant folding;
        # the remaining operands get the precise operation.
        assert "uint leftBits = floatBitsToUint(left);" in helper
        assert f"    precise float result = left {operator} right;" in helper
    # Two zeros sum to -0 only when both are -0; x - y is x + (-y).
    assert "return uintBitsToFloat(leftBits & rightBits);" in generated
    assert "return uintBitsToFloat(leftBits & (rightBits ^ 0x80000000u));" in generated
    # 0 * inf and 0 / 0 are NaN, other zero products and quotients are signed.
    assert "? 0x7fc00000u : (leftBits ^ rightBits) & 0x80000000u);" in generated
    assert (
        "crossgl_ieee_difference_float(crossgl_ieee_product_float(a.real, b.real), "
        "crossgl_ieee_product_float(a.imag, b.imag))"
    ) in generated
    assert "crossgl_ieee_difference_float(a, b.real), (-b.imag))" in generated
    assert (
        "float denom = crossgl_ieee_sum_float(crossgl_ieee_product_float(b.real, "
        "b.real), crossgl_ieee_product_float(b.imag, b.imag));"
    ) in generated
    assert (
        "crossgl_ieee_quotient_float(x, denom), crossgl_ieee_quotient_float(y, denom)"
    ) in generated
    body = generated.split("void main()", 1)[1]
    # Arithmetic outside the operators keeps the plain GLSL operators.
    assert "float plain = ((a * 0.0) + b);" in body
    assert "crossgl_ieee_" not in body
    _compile(generated, "opengl", tmp_path)


def test_opengl_exactness_test_locals_are_precise(tmp_path):
    generated = _translate_text(
        tmp_path, FREE_OPERATOR_KERNEL, "opengl", "operators.metal"
    )
    # Mesa rewrites a non-precise (1 + x) == 1 into x == 0 and (1 + x) - 1
    # into x; a precise local keeps MLX's log1p exact for tiny operands.
    assert (
        "float mlx_log1p(float x) {\n"
        "    precise float xp1 = (1.0 + x);\n"
        "    if ((xp1 == 1.0)) {\n"
        "        return x;\n"
        "    }\n"
        "    return (x * (log(xp1) / (xp1 - 1.0)));\n"
        "}\n"
    ) in generated
    source = _kernel("""    float scaled = x * 2.0f;
    float shifted = 1.0f + x;
    half narrow = half(1.0f) + half(x);
    float u = scaled == 1.0f ? x : shifted - 1.0f;
    float w = narrow == half(1.0f) ? x : 2.0f;
    float total = 0.0f;
    total = total * x + 1.0f;
    total = metal::max(total, x);
    float step = x - 1.0f;
    step += x;
    float v = total == 0.0f ? x : (step != 0.0f ? 1.0f / total : 2.0f);
    data[i] = u + w + v;
""")
    scoped = _translate_text(tmp_path, source, "opengl", "scoped.metal")
    body = scoped.split("void main()", 1)[1]
    # Only binary32 locals whose single value is a sum tested for equality;
    # a reassigned accumulator compares a later value, not the sum.
    assert "    float scaled = (x * 2.0);\n" in body
    assert "    float shifted = (1.0 + x);\n" in body
    assert "    float total = 0.0;\n" in body
    assert "    float step = (x - 1.0);\n" in body
    assert "precise" not in body
    _compile(scoped, "opengl", tmp_path)


def test_opengl_binary32_atan2_is_canonical(tmp_path):
    generated = _translate_text(
        tmp_path, FREE_OPERATOR_KERNEL, "opengl", "operators.metal"
    )
    helper = generated.split("float crossgl_atan2_float(float y, float x) {\n", 1)[
        1
    ].split("\n}\n", 1)[0]
    # NaN, signed zeros and infinities are settled by integer tests (GLSL
    # leaves atan(0, 0) undefined and Mesa drops the sign of a zero y); the
    # builtin computes every other quotient.
    assert "return uintBitsToFloat(0x7fc00000u);" in helper
    assert "return uintBitsToFloat(sign | (negativeX ? 0x40490fdbu : 0u));" in helper
    assert "return uintBitsToFloat(sign | 0x3fc90fdbu);" in helper
    assert "? (negativeX ? 0x4016cbe4u : 0x3f490fdbu) : 0x3fc90fdbu;" in helper
    assert helper.endswith("    return atan(y, x);")
    assert "float t = crossgl_atan2_float(b, a);" in generated
    source = _kernel("""    float2 tv = atan2(float2(x, 1.0f), float2(1.0f, x));
    half th = atan2(half(x), half(1.0f));
    data[i] = tv.y + float(th);
""")
    lanes = _translate_text(tmp_path, source, "opengl", "lanes.metal")
    assert (
        "vec2 crossgl_atan2_vec2(vec2 y, vec2 x) {\n"
        "    return vec2(crossgl_atan2_float(y.x, x.x), crossgl_atan2_float(y.y, x.y));\n"
        "}\n"
    ) in lanes
    body = lanes.split("void main()", 1)[1]
    assert "vec2 tv = crossgl_atan2_vec2(vec2(x, 1.0), vec2(1.0, x));" in body
    # Binary16 operands keep the builtin and the half rounding contract.
    assert "atan(crossgl_round_half1(float(x)), 1.0)" in body
    _compile(lanes, "opengl", tmp_path)


def test_atan2_helper_angles_are_the_rounded_binary32_angles():
    for bits, angle in (
        (0x40490FDB, math.pi),
        (0x3FC90FDB, math.pi / 2),
        (0x4016CBE4, 3 * math.pi / 4),
        (0x3F490FDB, math.pi / 4),
    ):
        assert _bits(angle) == bits, hex(bits)
    # C99 Annex F, which Python's atan2 follows, for every settled class.
    special = (0.0, -0.0, 1.0, -1.0, math.inf, -math.inf)
    for y in special:
        for x in special:
            if y in (1.0, -1.0) and x in (1.0, -1.0):
                continue  # the builtin's case
            expected = _bits(math.atan2(y, x))
            assert _atan2_helper_model(y, x) == expected, (y, x)
    assert _atan2_helper_model(math.nan, 0.0) == 0x7FC00000
    assert _atan2_helper_model(0.0, -math.nan) == 0x7FC00000
    assert _atan2_helper_model(2.0, 3.0) is None


def _atan2_helper_model(y, x):
    """crossgl_atan2_float's integer tests; None where the builtin decides."""
    y_bits, x_bits = _bits(y), _bits(x)
    y_magnitude, x_magnitude = y_bits & 0x7FFFFFFF, x_bits & 0x7FFFFFFF
    sign = y_bits & 0x80000000
    negative_x = bool(x_bits & 0x80000000)
    if y_magnitude > 0x7F800000 or x_magnitude > 0x7F800000:
        return 0x7FC00000
    if y_magnitude == 0:
        return sign | (0x40490FDB if negative_x else 0)
    if x_magnitude == 0:
        return sign | 0x3FC90FDB
    if y_magnitude == 0x7F800000:
        if x_magnitude == 0x7F800000:
            return sign | (0x4016CBE4 if negative_x else 0x3F490FDB)
        return sign | 0x3FC90FDB
    if x_magnitude == 0x7F800000:
        return sign | (0x40490FDB if negative_x else 0)
    return None


DIFFERENCE_KERNEL = """#include <metal_stdlib>
using namespace metal;

kernel void probe(device uint* data [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(data[5u * i]);
    float y = x + 1.0f;
    float2 v = float2(x, y);
    half h = half(x);
    int k = int(i);
    float d = x - x;
    float e = x - y;
    float2 f = v - v;
    half q = h - h;
    int g = k - k;
    // Each result is stored on its own: a fast-math consumer could still
    // assume a finite operand and drop the difference.
    data[5u * i] = as_type<uint>(d);
    data[5u * i + 1u] = as_type<uint>(e);
    data[5u * i + 2u] = as_type<uint>(f.y);
    data[5u * i + 3u] = uint(as_type<ushort>(q));
    data[5u * i + 4u] = uint(g);
}
"""


def test_directx_identical_float_difference_is_precise(tmp_path):
    generated = _translate_text(
        tmp_path, DIFFERENCE_KERNEL, "directx", "difference.metal"
    )
    # Optimized DXC folds an unqualified x - x to 0 even for inf and NaN.
    for value_type in ("float", "float2", "float16_t"):
        assert (
            f"{value_type} __crossgl_precise_difference_{value_type}("
            f"{value_type} left, {value_type} right) {{\n"
            f"    precise {value_type} result = left - right;\n"
            "    return result;\n}\n"
        ) in generated
    body = generated.split("void CSMain(", 1)[1]
    assert "__crossgl_precise_difference_float(x, x)" in body
    assert "__crossgl_precise_difference_float2(v, v)" in body
    assert "__crossgl_precise_difference_float16_t(h, h)" in body
    assert "(x - y)" in body
    assert "(k - k)" in body
    # The OpenGL rewrites stay OpenGL-only.
    ieee = _translate_text(tmp_path, IEEE_SOURCE, "directx", "ieee.metal")
    assert "crossgl_fma_" not in ieee
    assert "__crossgl_precise_difference_float(a, a)" in ieee
    _compile(generated, "directx", tmp_path)


def test_directx_precise_sinh_and_tanh_use_exact_helpers(tmp_path):
    generated = _translate_text(tmp_path, ODD_KERNEL, "directx", "odd.metal")
    plain = _translate_text(
        tmp_path,
        ODD_KERNEL.replace("metal::precise::", "").replace("precise::", ""),
        "directx",
        "plain.metal",
    )
    body = generated.split("void CSMain(", 1)[1]
    for operation in ("sinh", "tanh"):
        helper = f"__crossgl_precise_{operation}_float"
        assert f"float {helper}(float value) {{" in generated
        assert f"float4 {helper}4(float4 value) {{" in generated
        assert f"{helper}(x)" in body
        assert f"{helper}4(v)" in body
        assert helper not in plain
    # NaN and |x| < 2^-12 return the operand itself, signed zeros included;
    # the magnitude result takes the operand's sign bit.
    assert (
        "    if (magnitudeBits > 0x7f800000u || magnitudeBits < 0x39800000u) {\n"
        "        return value;\n"
        "    }\n"
    ) in generated
    assert "asuint(sinh(asfloat(magnitudeBits))) & 0x7fffffffu" in generated
    assert "magnitudeBits >= 0x41102cb4u\n        ? 0x3f800000u\n" in generated
    assert "return asfloat(resultBits | (bits & 0x80000000u));" in generated
    # Unannotated calls keep the builtins; asinh and atanh keep their exact
    # inverse-hyperbolic helpers, and the even and other functions stay plain.
    assert "float p = (sinh(x) + tanh(x));" in body
    for operation in ("asinh", "atanh"):
        assert f"__crossgl_{operation}_float(" in body
        assert f"__crossgl_{operation}_float(" in plain
        assert f"__crossgl_precise_{operation}" not in generated
    for operation in ("tan", "cosh", "log2", "log10"):
        assert f"__crossgl_precise_{operation}_float" not in generated
    _compile(generated, "directx", tmp_path)


def test_directx_precise_helpers_apply_to_narrow_float_operands(tmp_path):
    source = _kernel("""    half h = half(x);
    half2 hv = half2(h, -h);
    bfloat16_t b = bfloat16_t(x);
    half t = precise::sinh(h);
    half2 w = precise::tanh(hv);
    half2 s = precise::sinh(hv);
    bfloat16_t u = metal::precise::tanh(b);
    bfloat16_t z = metal::precise::sinh(b);
    data[i] = float(t) + float(w.y) + float(s.x) + float(u) + float(z);
""")
    generated = _translate_text(tmp_path, source, "directx", "narrow.metal")
    body = generated.split("void CSMain(", 1)[1]
    # Metal's precise:: overloads are binary32: narrow operands widen.  Metal
    # rounds a binary32 sinh past the binary16 range to infinity, but a native
    # float16_t conversion clamps it to 65504 on WARP, so sinh narrows through
    # the round-to-nearest-even helper.  |tanh| <= 1 cannot overflow.
    assert (
        "float16_t __crossgl_precise_sinh_float16_t(float16_t value) {\n"
        "    return __crossgl_round_half1("
        "__crossgl_precise_sinh_float((float)value));\n}\n"
    ) in generated
    assert (
        "__crossgl_round_half1(__crossgl_precise_sinh_float((float)value.y))"
        in generated
    )
    assert "(float16_t)__crossgl_precise_sinh_float(" not in generated
    assert "(float16_t)__crossgl_precise_tanh_float((float)value.y)" in generated
    assert (
        "    } else if (magnitude >= 0x477ff000u) {\n        result = 0x7c00u;\n"
    ) in generated
    assert "__crossgl_precise_sinh_float16_t(h)" in body
    assert "__crossgl_precise_tanh_float16_t2(hv)" in body
    assert "__crossgl_precise_sinh_float16_t2(hv)" in body
    assert "__crossgl_precise_tanh_float(__crossgl_bfloat16_to_float(" in body
    assert "__crossgl_precise_sinh_float(__crossgl_bfloat16_to_float(" in body
    _compile(generated, "directx", tmp_path)


def test_directx_tanh_saturation_bound_is_the_first_binary32_one():
    """From 0x41102cb4 on, round-to-nearest binary32 tanh is exactly one."""

    def rounded_tanh(bits):
        return _float(_bits(math.tanh(_float(bits))))

    assert rounded_tanh(0x41102CB4) == 1.0
    assert rounded_tanh(0x41102CB3) == _float(0x3F7FFFFF)
    # |x| < 2^-12 rounds sinh and tanh to x itself.
    for bits in range(0x397FF000, 0x39800000, 37):
        value = _float(bits)
        assert _float(_bits(math.sinh(value))) == value
        assert _float(_bits(math.tanh(value))) == value


# Every special class plus ordinary magnitudes, as (float32 word, half word).
SPECIAL_FLOATS = (
    0.0,
    -0.0,
    math.inf,
    -math.inf,
    math.nan,
    -math.nan,
    1.0e-30,
    -1.0e-30,
    0.25,
    -0.25,
    0.5,
    -0.5,
    0.75,
    -0.75,
    1.0,
    -1.0,
    2.0,
    -2.0,
    7.5,
    -7.5,
    20.0,
    -20.0,
    100.0,
    -100.0,
)


def _special_inputs():
    """Two words per row: the binary32 operand and the binary16 operand."""
    words = []
    for value in SPECIAL_FLOATS:
        words.append(_bits(value))
        words.append(struct.unpack("<H", struct.pack("<e", value))[0])
    return words


def _odd_oracle(operation, value):
    """Return the exact class of op(value): ('nan',), ('inf', sign) or ('finite', v)."""
    if math.isnan(value):
        return ("nan",)
    try:
        result = getattr(math, operation)(value)
    except OverflowError:
        return ("inf", math.copysign(1.0, value))
    except ValueError:
        if operation == "atanh" and abs(value) == 1.0:
            return ("inf", math.copysign(1.0, value))
        return ("nan",)
    if math.isinf(result):
        return ("inf", math.copysign(1.0, result))
    return ("finite", result)


def _check_odd_result(operation, operand, result, *, narrow):
    expected = _odd_oracle(operation, operand)
    label = (operation, "half" if narrow else "float", operand, result)
    if expected[0] == "nan":
        assert math.isnan(result), label
        return
    assert not math.isnan(result), label
    assert math.copysign(1.0, result) == math.copysign(1.0, operand), label
    if expected[0] == "inf":
        assert math.isinf(result), label
        return
    if operand == 0.0:
        assert result == 0.0, label
        return
    want = expected[1]
    limit = 65504.0 if narrow else 3.4028234663852886e38
    if abs(want) > limit:
        assert math.isinf(result), label
        return
    if 0.25 <= abs(operand) <= 20.0:
        assert abs(result - want) <= (4.0e-3 if narrow else 1.0e-4) * abs(want), label


def _gl_odd_source():
    lines = []
    for index, operation in enumerate(ODD_OPERATIONS):
        lines.append(
            f"    half h{index} = precise::{operation}(h);\n"
            f"    float x{index} = metal::precise::{operation}(x);\n"
            f"    results[base + {index}u] = uint(as_type<ushort>(h{index}));\n"
            f"    results[base + {len(ODD_OPERATIONS) + index}u] = "
            f"as_type<uint>(x{index});\n"
        )
    return """#include <metal_stdlib>
using namespace metal;

kernel void precise_odd(device const uint* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[2u * i]);
    half h = as_type<half>(ushort(values[2u * i + 1u] & 0xffffu));
    uint base = %(stride)du * i;
""" % {"stride": 2 * len(ODD_OPERATIONS)} + "".join(lines) + "}\n"


def test_odd_oracle_and_checker_reject_the_mesa_results():
    assert _odd_oracle("tanh", math.nan) == ("nan",)
    assert _odd_oracle("atanh", -1.0) == ("inf", -1.0)
    assert _odd_oracle("atanh", 2.0) == ("nan",)
    assert _odd_oracle("sinh", 1000.0) == ("inf", 1.0)
    # Mesa's builtins: sinh(-0) = +0 and tanh(NaN) = -1.
    with pytest.raises(AssertionError):
        _check_odd_result("sinh", -0.0, 0.0, narrow=False)
    with pytest.raises(AssertionError):
        _check_odd_result("tanh", math.nan, -1.0, narrow=False)
    with pytest.raises(AssertionError):
        _check_odd_result("asinh", 2.0, 1.2, narrow=False)
    _check_odd_result("sinh", -0.0, -0.0, narrow=False)
    _check_odd_result("tanh", -math.inf, -1.0, narrow=True)


def _skip_without_native(target_names):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native precise math")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    if target not in target_names:
        pytest.skip(f"{target} output does not depend on the precise namespace")
    return target


def _words(actual, expected):
    assert len(actual) == len(expected)
    assert actual[-len(GUARD) :] == GUARD
    return actual[: -len(GUARD)]


def test_round_trip_matches_original_precise_bits_for_every_half(tmp_path):
    _skip_without_native({"metal"})
    inputs = [(index << 16) | index for index in range(1 << 16)]
    expected = [0] * (OUTPUTS_PER_INPUT * len(inputs)) + GUARD
    source = _metal_source()
    generated = _translate_text(tmp_path, source, "metal", "round.metal")
    # The previous lowering: every precise:: call became the plain builtin.
    previous = generated.replace("metal::precise::", "")
    words = {}
    for label, text in (
        ("original", source),
        ("generated", generated),
        ("previous", previous),
    ):
        words[label] = _execute(
            tmp_path / label,
            "metal",
            text,
            inputs,
            expected,
            metal_entry="precise_math",
            check_outputs=_words,
            metal_compile_flags=METAL_FLAGS,
        ).pop("maxUlpError")
    mismatches = [
        index
        for index, (left, right) in enumerate(
            zip(words["generated"], words["original"])
        )
        if left != right
    ]
    assert not mismatches, [
        (OPERATIONS[(index % OUTPUTS_PER_INPUT) % len(OPERATIONS)], hex(index))
        for index in mismatches[:16]
    ]
    differing = {}
    for index, (left, right) in enumerate(zip(words["previous"], words["original"])):
        if left != right:
            slot = index % OUTPUTS_PER_INPUT
            operation = OPERATIONS[slot % len(OPERATIONS)]
            kind = "half" if slot < len(OPERATIONS) else "float"
            differing[f"{kind}:{operation}"] = differing.get(f"{kind}:{operation}", 0)
            differing[f"{kind}:{operation}"] += 1
    # Negative control: the plain half overloads are observably different, so
    # the equality above is evidence that precise:: survived.
    assert differing.get("half:tan", 0) > 0, differing
    assert not any(key.startswith("float:") for key in differing), differing
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "inputs": len(inputs),
                "outputsPerInput": OUTPUTS_PER_INPUT,
                "plainLoweringDifferences": differing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_opengl_odd_functions_keep_sign_of_zero_and_nan_natively(tmp_path):
    _skip_without_native({"opengl"})
    inputs = _special_inputs()
    rows = len(inputs) // 2
    stride = 2 * len(ODD_OPERATIONS)
    expected = [0] * (stride * rows) + GUARD
    generated = _translate_text(tmp_path, _gl_odd_source(), "opengl", "odd.metal")
    for operation in ODD_OPERATIONS:
        assert f"crossgl_precise_{operation}_float(" in generated
    actual = _execute(
        tmp_path / "generated",
        "opengl",
        generated,
        inputs,
        expected,
        check_outputs=_words,
        invocations=rows,
    ).pop("maxUlpError")
    for row in range(rows):
        float_operand = _float(inputs[2 * row])
        half_operand = _half(inputs[2 * row + 1])
        for index, operation in enumerate(ODD_OPERATIONS):
            _check_odd_result(
                operation,
                half_operand,
                _half(actual[row * stride + index]),
                narrow=True,
            )
            _check_odd_result(
                operation,
                float_operand,
                _float(actual[row * stride + len(ODD_OPERATIONS) + index]),
                narrow=False,
            )


def test_directx_odd_functions_keep_sign_nan_and_saturation_natively(tmp_path):
    _skip_without_native({"directx"})
    inputs = _special_inputs()
    rows = len(inputs) // 2
    stride = 2 * len(ODD_OPERATIONS)
    expected = [0] * (stride * rows) + GUARD
    generated = _translate_text(tmp_path, _gl_odd_source(), "directx", "odd.metal")
    for operation in ("sinh", "tanh"):
        assert f"__crossgl_precise_{operation}_float(" in generated
    actual = _execute(
        tmp_path / "generated",
        "directx",
        generated,
        inputs,
        expected,
        check_outputs=_words,
        directx_compile_flags=("-enable-16bit-types",),
        invocations=rows,
    ).pop("maxUlpError")
    tanh_index = ODD_OPERATIONS.index("tanh")
    for row in range(rows):
        float_operand = _float(inputs[2 * row])
        half_operand = _half(inputs[2 * row + 1])
        for index, operation in enumerate(ODD_OPERATIONS):
            _check_odd_result(
                operation,
                half_operand,
                _half(actual[row * stride + index]),
                narrow=True,
            )
            _check_odd_result(
                operation,
                float_operand,
                _float(actual[row * stride + len(ODD_OPERATIONS) + index]),
                narrow=False,
            )
        # Binary32 tanh is exactly +-1 from |x| = 9.0109138 (WARP's Htan
        # gives 1 - 2^-24 at 20 and NaN once exp(2|x|) overflows).
        for operand, word, narrow in (
            (half_operand, actual[row * stride + tanh_index], True),
            (
                float_operand,
                actual[row * stride + len(ODD_OPERATIONS) + tanh_index],
                False,
            ),
        ):
            if not math.isnan(operand) and abs(operand) >= 9.0109138:
                result = _half(word) if narrow else _float(word)
                assert result == math.copysign(1.0, operand), (operand, result)


def test_directx_identical_difference_stays_exact_natively(tmp_path):
    _skip_without_native({"directx"})
    inputs = [_bits(value) for row in FMA_ROWS for value in row]
    expected = [0] * (4 * len(FMA_ROWS)) + GUARD
    generated = _translate_text(tmp_path, IEEE_SOURCE, "directx", "ieee.metal")
    assert "__crossgl_precise_difference_float(a, a)" in generated
    actual = _execute(
        tmp_path / "generated",
        "directx",
        generated,
        inputs,
        expected,
        check_outputs=_words,
        invocations=len(FMA_ROWS),
    ).pop("maxUlpError")
    difference = _fma_results(actual)["difference"]
    # x - x is NaN for infinities and NaN, and +0 for finite operands.
    assert math.isnan(difference[2]) and math.isnan(difference[3]), difference
    for row in (0, 1, 4):
        assert difference[row] == 0.0, difference
        assert math.copysign(1.0, difference[row]) == 1.0, difference


FMA_ROWS = (
    # A fused product keeps 2^-46, which a split multiply-add rounds away.
    (_float(0x3F800001), _float(0x3F800001), -_float(0x3F800002)),
    # fma rounds 2^24 + 2.25 to 2^24 + 2, so fma(a, b, c) - c is 2, not a * b.
    (1.5, 1.5, 16777216.0),
    (math.inf, 1.0, 0.0),
    (math.nan, 1.0, 0.0),
    (3.0, 4.0, 5.0),
)


def _fma_results(actual):
    columns = [
        [_float(actual[4 * row + lane]) for row in range(len(FMA_ROWS))]
        for lane in range(4)
    ]
    return dict(zip(("fused", "residual", "difference", "lanes"), columns))


def _fma_contract_holds(results):
    """GLSL 4.60 8.3 makes a precise fma one operation; fusing it stays the
    device's choice, so row 0 may hold the fused 2^-46 or the split 0.  No
    surrounding arithmetic may be folded into it, and x - x is never 0 for
    infinities or NaN."""
    return (
        results["fused"][0] in (2.0**-46, 0.0)
        and results["lanes"][0] == results["fused"][0]
        and results["fused"][1] == 16777218.0
        and results["lanes"][1] == 16777218.0
        and results["residual"][1] == 2.0
        and math.isnan(results["difference"][2])
        and math.isnan(results["difference"][3])
        and results["difference"][0] == 0.0
        and results["difference"][4] == 0.0
        and results["fused"][4] == 17.0
        and results["residual"][4] == 12.0
        and results["lanes"][4] == 17.0
    )


def test_fma_contract_rejects_folded_results():
    exact = {
        "fused": [2.0**-46, 16777218.0, math.inf, math.nan, 17.0],
        "residual": [_float(0x3F800002), 2.0, math.nan, math.nan, 12.0],
        "difference": [0.0, 0.0, math.nan, math.nan, 0.0],
        "lanes": [2.0**-46, 16777218.0, math.inf, math.nan, 17.0],
    }
    assert _fma_contract_holds(exact)
    split = {key: list(values) for key, values in exact.items()}
    split["fused"][0] = split["lanes"][0] = 0.0
    assert _fma_contract_holds(split)
    for column, row, value in (
        ("fused", 0, 2.0**-40),  # neither the fused nor the split result
        ("lanes", 0, 0.0),  # scalar and vector fma disagree
        ("residual", 1, 2.25),  # fma(a, b, c) - c folded to a * b
        ("difference", 2, 0.0),  # x - x folded to 0
        ("difference", 3, 0.0),
    ):
        corrupted = {key: list(values) for key, values in exact.items()}
        corrupted[column][row] = value
        assert not _fma_contract_holds(corrupted), (column, row)


def test_opengl_fma_and_identical_difference_stay_exact_natively(tmp_path):
    _skip_without_native({"opengl"})
    inputs = [_bits(value) for row in FMA_ROWS for value in row]
    expected = [0] * (4 * len(FMA_ROWS)) + GUARD
    generated = _translate_text(tmp_path, IEEE_SOURCE, "opengl", "ieee.metal")
    assert "crossgl_fma_float(a, b, c)" in generated
    assert "crossgl_difference_float(a, a)" in generated
    records = {}
    for label, text in (
        ("generated", generated),
        # The previous lowering: native fma and x - x without precise results.
        ("previous", generated.replace("    precise ", "    ")),
    ):
        actual = _execute(
            tmp_path / label,
            "opengl",
            text,
            inputs,
            expected,
            check_outputs=_words,
            invocations=len(FMA_ROWS),
        ).pop("maxUlpError")
        records[label] = _fma_results(actual)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "rows": [[repr(value) for value in row] for row in FMA_ROWS],
                "records": {
                    label: {key: [repr(v) for v in values] for key, values in r.items()}
                    for label, r in records.items()
                },
                "deviceFusesPreciseFma": records["generated"]["fused"][0] != 0.0,
                "previousLoweringHolds": _fma_contract_holds(records["previous"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert _fma_contract_holds(records["generated"]), records


# (a, b) rows for FREE_OPERATOR_KERNEL: signed zeros, infinities, NaN, and
# operands below half an ulp of one, where MLX's log1p returns x itself.
FREE_OPERATOR_ROWS = (
    (0.0, 0.0),
    (-0.0, 0.0),
    (0.0, -0.0),
    (-0.0, -0.0),
    (-2.0, 3.0),
    (2.0, -0.0),
    (-0.0, -3.0),
    (0.5, 0.25),
    (-0.5, -0.75),
    (math.inf, 1.0),
    (-math.inf, 0.0),
    (math.inf, math.inf),
    (-math.inf, -math.inf),
    (-math.inf, math.inf),
    (3.0, -math.inf),
    (math.nan, 1.0),
    (1.0, math.nan),
    (1.0e-10, -0.0),
    (-1.0e-10, math.inf),
    (_float(0x33800000), 1.0),
)


def _ieee_binary32(operator, left, right):
    """One correctly rounded binary32 operation on binary32 operands."""
    if operator == "/":
        if right == 0.0:
            if left == 0.0 or math.isnan(left):
                return math.nan
            return math.copysign(math.inf, left) * math.copysign(1.0, right)
        value = left / right
    else:
        value = {"+": left + right, "-": left - right, "*": left * right}[operator]
    return _float(_bits(value))


def _free_operator_oracle(a, b):
    """FREE_OPERATOR_KERNEL's operator results under Metal's -fno-fast-math."""

    def product(x, y):
        return _ieee_binary32("*", x, y)

    def total(x, y):
        return _ieee_binary32("+", x, y)

    def difference(x, y):
        return _ieee_binary32("-", x, y)

    # unit * z, z / unit with unit = (0, 1), 0 - z and z + (0, 0).
    p = (
        difference(product(0.0, a), product(1.0, b)),
        total(product(0.0, b), product(1.0, a)),
    )
    denom = total(product(0.0, 0.0), product(1.0, 1.0))
    q = (
        _ieee_binary32("/", total(product(a, 0.0), product(b, 1.0)), denom),
        _ieee_binary32("/", difference(product(b, 0.0), product(a, 1.0)), denom),
    )
    r = (difference(0.0, a), -b)
    s = (total(a, 0.0), total(b, 0.0))
    return p + q + r + s


def _same_binary32(actual_word, expected):
    if math.isnan(expected):
        return math.isnan(_float(actual_word))
    return actual_word == _bits(expected)


def test_free_operator_oracle_rejects_the_mesa_folds():
    # Mesa folds 0 * x to +0, x + 0 to x and 0 - x to -x, even when precise.
    assert _bits(_free_operator_oracle(-2.0, 0.0)[0]) == 0x80000000  # -0 - 0
    assert _bits(_free_operator_oracle(-0.0, -3.0)[1]) == 0x80000000  # -0 + -0
    assert _bits(_free_operator_oracle(-0.0, 0.0)[6]) == 0  # -0 + 0 is +0
    assert _bits(_free_operator_oracle(0.0, 1.0)[4]) == 0  # 0 - 0 is +0
    assert math.isnan(_free_operator_oracle(math.inf, 1.0)[0])  # 0 * inf
    assert not _same_binary32(_bits(0.0), -0.0)
    assert _same_binary32(0x7FC00001, math.nan)


def test_opengl_free_operators_atan2_and_log1p_stay_ieee_natively(tmp_path):
    _skip_without_native({"opengl"})
    inputs = [_bits(value) for row in FREE_OPERATOR_ROWS for value in row]
    rows = len(FREE_OPERATOR_ROWS)
    expected = [0] * (FREE_OPERATOR_OUTPUTS * rows) + GUARD
    generated = _translate_text(
        tmp_path, FREE_OPERATOR_KERNEL, "opengl", "operators.metal"
    )
    assert "crossgl_ieee_product_float(" in generated
    assert "precise float xp1" in generated
    actual = _execute(
        tmp_path / "generated",
        "opengl",
        generated,
        inputs,
        expected,
        check_outputs=_words,
        invocations=rows,
    ).pop("maxUlpError")
    mismatches = []
    for row, (a, b) in enumerate(FREE_OPERATOR_ROWS):
        words = actual[row * FREE_OPERATOR_OUTPUTS : (row + 1) * FREE_OPERATOR_OUTPUTS]
        for slot, want in enumerate(_free_operator_oracle(a, b)):
            if not _same_binary32(words[slot], want):
                mismatches.append(("operator", slot, a, b, hex(words[slot])))
        settled = _atan2_helper_model(b, a)
        angle = _float(words[8])
        if settled is not None:
            if not _same_binary32(words[8], _float(settled)):
                mismatches.append(("atan2", a, b, hex(words[8])))
        elif abs(angle - math.atan2(b, a)) > 1.0e-5 * abs(math.atan2(b, a)):
            mismatches.append(("atan2", a, b, angle))
        if not math.isinf(a) and not math.isnan(a) and a > -1.0:
            if _float(_bits(1.0 + a)) == 1.0:
                # Rounding absorbed a: MLX returns the operand bit for bit.
                if words[9] != _bits(a):
                    mismatches.append(("log1p", a, hex(words[9])))
            elif abs(_float(words[9]) - math.log1p(a)) > 1.0e-5 * abs(math.log1p(a)):
                mismatches.append(("log1p", a, _float(words[9])))
    assert not mismatches, mismatches


def test_ci_requires_native_precise_namespace():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_precise_math_namespace.py" in step
