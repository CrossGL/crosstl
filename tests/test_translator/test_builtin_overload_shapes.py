"""A call that only a built-in accepts keeps the built-in when a user overload of its name is renamed.

``float dot(float x, float y)`` beside the library's ``dot(floatN, floatN)`` is renamed ``dot_fn`` in Metal output;
the calls with vector arguments must still reach the library, since no implicit conversion turns a vector into the
user overload's scalar parameters.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen.ast_walk import walk_ast_values
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.builtin_overload_shapes import (
    calls_only_builtins_accept,
)
from crosstl.translator.semantics.hub_types import call_name

CROSSGL = """
shader S {
    buffer vec4 values[];
    buffer float results[];
    float dot(float x, float y) { return x * y; }
    float clamp(vec2 x, float lo, float hi) { return x.x + lo + hi; }
    compute {
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec4 v = values[i];
            float f = v.x;
            float s = dot(v, vec4(1.0));
            s += dot(f, f);
            s += dot(values[i], values[i]);
            s += clamp(f, 0.0, 1.0);
            s += clamp(v.xy, 0.0, 1.0);
            results[i] = s;
        }
    }
}
"""

METAL = """
#include <metal_stdlib>
using namespace metal;
static inline float dot(float x, float y) { return x * y; }
template <typename T>
kernel void reduce_impl(device const T* a [[buffer(0)]], device float* o [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    T v = a[i];
    float s = dot(v, T(1.0f));
    s += dot(a[i], a[i]);
    o[i] = s;
}
template [[host_name("reduce_f32")]] kernel void reduce_impl<float>(device const float*, device float*, uint);
template [[host_name("reduce_f32x4")]] kernel void reduce_impl<float4>(device const float4*, device float*, uint);
"""


def _squashed(code):
    return " ".join(code.split())


def _translate(tmp_path, name, source, **options):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), format_output=False, **options)


def test_vector_calls_are_left_to_the_builtin():
    program = Parser(Lexer(CROSSGL).get_tokens()).parse()
    builtin = calls_only_builtins_accept(program, {"dot", "clamp"})
    calls = [
        value
        for value in walk_ast_values(program)
        if isinstance(value, A.FunctionCallNode)
    ]
    kept = sorted(
        (call_name(call), len(call.arguments)) for call in calls if id(call) in builtin
    )
    # dot(v, vec4(1.0)), dot(values[i], values[i]) and clamp(f, 0.0, 1.0) need another shape.
    assert kept == [("clamp", 3), ("dot", 2), ("dot", 2)]


def test_unknown_argument_types_follow_the_user_overload():
    program = Parser(Lexer("""
shader S {
    float dot(float x, float y) { return x * y; }
    float helper(float x) { return dot(mystery(x), x); }
}
""").get_tokens()).parse()
    assert calls_only_builtins_accept(program, {"dot"}) == set()


def test_metal_output_keeps_vector_calls_on_the_library(tmp_path):
    code = _squashed(_translate(tmp_path, "shapes.cgl", CROSSGL, backend="metal"))
    assert "float dot_fn(float x, float y)" in code
    assert "dot_fn(f, f)" in code
    assert "dot(v, float4(1.0))" in code
    assert "dot_fn(v" not in code
    assert "clamp_fn(v.xy, 0.0, 1.0)" in code
    assert "clamp(f, 0.0, 1.0)" in code


def test_metal_template_specializations_keep_vector_dot(tmp_path):
    code = _squashed(
        _translate(
            tmp_path, "shapes.metal", METAL, backend="metal", source_backend="metal"
        )
    )
    assert "dot_fn(v, float(1.0))" in code
    assert "dot(v, float4(1.0))" in code
    assert "dot(a[i], a[i])" in code


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="needs the Metal compiler")
@pytest.mark.parametrize(
    ("name", "source", "options"),
    [
        ("shapes.cgl", CROSSGL, {}),
        ("shapes.metal", METAL, {"source_backend": "metal"}),
    ],
)
def test_metal_output_compiles(tmp_path, name, source, options):
    code = _translate(tmp_path, name, source, backend="metal", **options)
    output = tmp_path / "out.metal"
    output.write_text(code)
    probe = subprocess.run(
        ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(output), "-o", "/dev/null"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
