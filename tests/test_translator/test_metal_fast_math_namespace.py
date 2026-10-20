"""Metal ``fast::`` math keeps its namespace through CrossGL, and only on Metal.

MLX compiles its kernels with ``-fno-fast-math`` and selects the relaxed
implementations explicitly (``fast::exp`` in softmax, logsumexp, attention and
cross entropy; ``fast::cos``/``fast::sin`` in rope and FFT).  Lowering those
calls to the plain builtins made the round trip call the precise
implementations: results changed bits and register pressure rose, which on an
M2 Max lowered the bfloat16 D=512 vector attention pipeline's
``maxTotalThreadsPerThreadgroup`` from MLX's 1024 to 832.

CrossGL now spells such calls ``fast::<op>``.  The parser hands every backend
the plain builtin and records the namespace in an annotation that only the
Metal backend reads, so OpenGL and DirectX output is byte-identical.
"""

import json
import os
import struct
import sys
from pathlib import Path

import pytest

import crosstl.backend.Metal.MetalCrossGLCodeGen as metal_frontend
from crosstl import translate
from crosstl.translator.ast import FunctionCallNode
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.math_namespaces import (
    METAL_FAST_MATH_FUNCTIONS,
    METAL_MATH_NAMESPACE_ANNOTATION,
    metal_fast_math_operation,
)
from crosstl.translator.parser import Parser
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_precise_trig import GUARD, _execute

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_FAST_MATH"
METAL_FLAGS = ("-std=metal3.1", "-fno-fast-math")
UNARY = ("exp", "exp2", "log", "log2", "sin", "cos", "tan", "sqrt", "rsqrt")
OUTPUTS_PER_INPUT = len(UNARY) + 1 + 4 + 3
SOURCE = (
    """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

template <typename T>
inline T relaxed_exp(T x) {
  return fast::exp(x);
}

kernel void fast_math(device const uint* values [[buffer(0)]],
                      device uint* results [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float x = as_type<float>(values[i]);
    float m = abs(x);
    uint base = %(stride)du * i;
""" % {"stride": OUTPUTS_PER_INPUT}
    + "".join(
        f"    results[base + {index}u] = as_type<uint>("
        f"{'metal::fast' if index % 2 else 'fast'}::{operation}("
        f"{'m' if operation in {'log', 'log2', 'sqrt', 'rsqrt'} else 'x'}));\n"
        for index, operation in enumerate(UNARY)
    )
    + f"""    results[base + {len(UNARY)}u] = as_type<uint>(metal::fast::pow(m, 1.37f));
    float4 lanes = fast::exp(float4(x, -x, 0.5f * x, x + 1.0f));
    float2 turns = metal::fast::sin(float2(x, 3.0f * x));
    results[base + {len(UNARY) + 1}u] = as_type<uint>(lanes.x);
    results[base + {len(UNARY) + 2}u] = as_type<uint>(lanes.w);
    results[base + {len(UNARY) + 3}u] = as_type<uint>(turns.x);
    results[base + {len(UNARY) + 4}u] = as_type<uint>(turns.y);
    half h = half(clamp(x, -8.0f, 8.0f));
    bfloat16_t b = bfloat16_t(x);
    float relaxed_half = fast::exp(h);
    float relaxed_bfloat = fast::exp(b);
    results[base + {len(UNARY) + 5}u] = as_type<uint>(relaxed_half);
    results[base + {len(UNARY) + 6}u] = uint(as_type<ushort>(relaxed_exp(h)));
    results[base + {len(UNARY) + 7}u] = as_type<uint>(relaxed_bfloat);
}}
"""
)
# Arguments where Apple's relaxed and precise implementations are known to
# round differently, plus ordinary and special values.
INPUTS = (
    0.0,
    -0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.14159265,
    -3.7,
    7.25,
    10.0,
    -20.0,
    0.001,
    44.5,
    -60.25,
    100.0,
    1000.0,
    12345.678,
    -88.5,
    0.7,
    6.0e-8,
)
CROSSGL_SOURCE = """shader FastMath {
    compute {
        layout(std430, binding = 0) buffer Data { float data[]; };
        [numthreads(1, 1, 1)]
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            float x = data[gid.x];
            vec4 v = vec4(x, x + 1.0, x + 2.0, x + 3.0);
            float s = PREFIXexp(x) + PREFIXexp2(x) + PREFIXlog(x) + PREFIXlog2(x);
            s += PREFIXsin(x) + PREFIXcos(x) + PREFIXtan(x) + PREFIXpow(x, 2.5);
            s += PREFIXsqrt(x) + PREFIXrsqrt(x);
            vec4 w = PREFIXexp(v) + PREFIXsin(v) + PREFIXsqrt(v);
            data[gid.x] = s + w.x + w.w;
        }
    }
}
"""


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _crossgl(prefix):
    return CROSSGL_SOURCE.replace("PREFIX", prefix)


def _calls(code):
    ast = Parser(Lexer(code).get_tokens()).parse()
    return [node for node in ast.walk() if isinstance(node, FunctionCallNode)]


def _translate(tmp_path, target, source=SOURCE, name="fast_math.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return translate(str(path), backend=target, format_output=False)


def test_fast_math_operation_names():
    for operation in sorted(METAL_FAST_MATH_FUNCTIONS):
        assert metal_fast_math_operation(f"fast::{operation}") == operation
        assert metal_fast_math_operation(f"metal::fast::{operation}") == operation
        assert metal_fast_math_operation(f"::metal::fast::{operation}") == operation
        assert metal_fast_math_operation(operation) is None
        assert metal_fast_math_operation(f"metal::{operation}") is None
        assert metal_fast_math_operation(f"precise::{operation}") is None
    assert metal_fast_math_operation("fast::tanh") is None
    assert metal_fast_math_operation("fast::exp::inner") is None
    assert metal_fast_math_operation("fastexp") is None


def test_crossgl_parser_normalizes_fast_calls():
    fast = _calls(_crossgl("fast::"))
    plain = _calls(_crossgl(""))
    qualified = _calls(_crossgl("metal::fast::"))
    assert len(fast) == len(plain) == len(qualified)
    math = [call for call in fast if call.function.name in METAL_FAST_MATH_FUNCTIONS]
    assert len(math) == 13
    for call in math:
        assert type(call.function).__name__ == "IdentifierNode"
        assert call.get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) == "fast"
    for call in qualified:
        if call.function.name in METAL_FAST_MATH_FUNCTIONS:
            assert call.get_annotation(METAL_MATH_NAMESPACE_ANNOTATION) == "fast"
    for call in plain:
        assert METAL_MATH_NAMESPACE_ANNOTATION not in call.annotations
    assert [call.function.name for call in fast] == [
        call.function.name for call in plain
    ]
    # Unknown fast:: names keep their spelling and gain no annotation.
    (unknown,) = _calls(
        "shader S { compute { void main() { float x = fast::tanh(1.0); } } }"
    )
    assert unknown.function.name == "fast::tanh"
    assert METAL_MATH_NAMESPACE_ANNOTATION not in unknown.annotations


@pytest.mark.parametrize(
    "target",
    [
        "opengl",
        "directx",
        "dx12",
        "vulkan",
        "cuda",
        "hip",
        "rust",
        "mojo",
        "slang",
        "webgpu",
    ],
)
def test_other_targets_ignore_the_fast_namespace(tmp_path, target):
    outputs = []
    for label, prefix in (("plain", ""), ("fast", "fast::"), ("q", "metal::fast::")):
        path = tmp_path / f"{label}.cgl"
        path.write_text(_crossgl(prefix), encoding="utf-8")
        # A target that refuses an unrelated construct must refuse every
        # spelling identically; the namespace may not change either outcome.
        try:
            outputs.append(translate(str(path), backend=target, format_output=False))
        except LoweringRefused as refusal:
            outputs.append(f"refused: {refusal}")
    assert outputs[0] == outputs[1] == outputs[2]
    assert "fast::" not in outputs[0]


def test_metal_backend_restores_the_fast_namespace(tmp_path):
    path = tmp_path / "fast.cgl"
    path.write_text(_crossgl("fast::"), encoding="utf-8")
    generated = translate(str(path), backend="metal", format_output=False)
    plain_path = tmp_path / "plain.cgl"
    plain_path.write_text(_crossgl(""), encoding="utf-8")
    plain = translate(str(plain_path), backend="metal", format_output=False)
    for operation in ("exp", "exp2", "log", "log2", "sin", "cos", "tan", "sqrt"):
        assert f"metal::fast::{operation}(x)" in generated
        assert f"metal::fast::{operation}(" not in plain
    assert "metal::fast::rsqrt(x)" in generated
    assert "metal::fast::pow(x, 2.5)" in generated
    assert "metal::fast::exp(v)" in generated
    assert "metal::fast::sin(v)" in generated
    assert "metal::fast::sqrt(v)" in generated
    assert generated.replace("metal::fast::", "") == plain
    _compile(generated, "metal", tmp_path, metal_compile_flags=METAL_FLAGS)


def test_metal_backend_keeps_fast_namespace_only_for_its_overloads(tmp_path):
    path = tmp_path / "narrow.cgl"
    path.write_text(
        """shader S {
    compute {
        layout(std430, binding = 0) buffer Data { float data[]; };
        [numthreads(1, 1, 1)]
        void main(uvec3 gid @ gl_GlobalInvocationID) {
            float x = data[gid.x];
            float16 h = float16(x);
            f16vec2 hv = f16vec2(h, h);
            f16vec2 e = fast::exp(hv);
            data[gid.x] = fast::exp(h) + float(e.x) + fast::exp(x);
        }
    }
}
""",
        encoding="utf-8",
    )
    generated = translate(str(path), backend="metal", format_output=False)
    # Scalars convert to float implicitly; half vectors do not, so they keep
    # the plain overload instead of an ill-formed fast:: call.
    assert "metal::fast::exp(h)" in generated
    assert "metal::fast::exp(x)" in generated
    assert "metal::fast::exp(hv)" not in generated
    assert "exp(hv)" in generated
    _compile(generated, "metal", tmp_path, metal_compile_flags=METAL_FLAGS)


def test_round_trip_restores_every_fast_call(tmp_path):
    crossgl = _translate(tmp_path, "crossgl")
    generated = _translate(tmp_path, "metal")
    for operation in UNARY:
        argument = "m" if operation in {"log", "log2", "sqrt", "rsqrt"} else "x"
        assert f"fast::{operation}({argument})" in crossgl
        assert f"metal::fast::{operation}({argument})" in generated
    assert "metal::fast::pow(m, 1.37)" in generated
    assert "metal::fast::exp(float4(" in generated
    assert "metal::fast::sin(float2(" in generated
    assert "metal::fast::exp(h)" in generated
    assert "metal::fast::exp(b)" in generated
    # results[base + 0u] and the body of relaxed_exp<half>.
    assert generated.count("metal::fast::exp(x)") == 2
    _compile(generated, "metal", tmp_path, metal_compile_flags=METAL_FLAGS)


@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_round_trip_leaves_other_targets_byte_identical(tmp_path, monkeypatch, target):
    current = _translate(tmp_path, target)
    # The frontend without the namespace marker is the previous behaviour.
    monkeypatch.setattr(metal_frontend, "metal_fast_math_operation", lambda _: None)
    previous = _translate(tmp_path, target)
    assert previous == current
    assert "fast::" not in current
    _compile(current, target, tmp_path)


def _check(actual, expected):
    assert actual[-len(GUARD) :] == GUARD
    assert len(actual) == len(expected)
    return actual[: -len(GUARD)]


def test_fast_math_round_trip_matches_the_original_bits(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native fast math")
    if sys.platform != "darwin":
        pytest.skip("fast:: changes Metal output only; other targets are pinned")
    inputs = [_bits(value) for value in INPUTS]
    expected = [0] * (OUTPUTS_PER_INPUT * len(INPUTS)) + GUARD
    generated = _translate(tmp_path, "metal")
    # The previous lowering: every fast:: call became the precise builtin.
    precise = generated.replace("metal::fast::", "")
    records = {}
    words = {}
    for label, source in (
        ("original", SOURCE),
        ("generated", generated),
        ("precise", precise),
    ):
        records[label] = _execute(
            tmp_path / label,
            "metal",
            source,
            inputs,
            expected,
            metal_entry="fast_math",
            check_outputs=_check,
            metal_compile_flags=METAL_FLAGS,
        )
        words[label] = records[label].pop("maxUlpError")
    assert words["generated"] == words["original"]
    differing = [
        index
        for index, (left, right) in enumerate(zip(words["precise"], words["original"]))
        if left != right
    ]
    # Negative control: the precise lowering is observably different, so the
    # equality above is evidence that fast:: survived.
    assert differing, "inputs do not distinguish fast:: from precise math"
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "inputs": list(INPUTS),
                "outputsPerInput": OUTPUTS_PER_INPUT,
                "preciseLoweringDifferences": len(differing),
                "records": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def test_ci_requires_native_fast_math_without_an_additional_runner():
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("      - name: Validate Metal builtin ownership\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_fast_math_namespace.py" in step
