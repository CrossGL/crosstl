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
    ["directx", "dx12", "vulkan", "cuda", "hip", "rust", "mojo", "slang", "webgpu"],
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


def test_directx_output_is_unchanged_by_the_ieee_rewrites(tmp_path):
    generated = _translate_text(tmp_path, IEEE_SOURCE, "directx", "ieee.metal")
    assert "crossgl_fma_" not in generated
    assert "crossgl_difference_" not in generated
    assert "(a - a)" in generated
    _compile(generated, "directx", tmp_path)


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
