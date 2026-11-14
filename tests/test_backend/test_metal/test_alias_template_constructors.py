"""Calls spelled with an alias-template instance are type constructors, not functions."""

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

PRELUDE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
using Id = T;

template <typename T>
using Pair = vec<T, 2>;

template <typename T, int N>
using Wide = vec<T, N>;

struct Plain {
    float a;
    float b;
};
"""


def convert(body: str) -> str:
    source = PRELUDE + """
kernel void fill(device float* values [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
""" + body + "\n}\n"
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)
    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    return crossgl


@pytest.mark.parametrize(
    "statement, expected",
    [
        ("float x = Id<float>(values[gid]);", "float x = float(buffer_load("),
        ("int i = Id<int>(values[gid]);", "int i = int(buffer_load("),
        ("float2 v = Pair<float>(values[gid], 1.0f);", "vec2 v = vec2(buffer_load("),
        ("half3 w = Wide<half, 3>(half(values[gid]));", "f16vec3 w = f16vec3(float16("),
        ("Plain p = Id<Plain>(Plain{values[gid], 1.0f});", "Plain p = Plain(Plain("),
    ],
)
def test_alias_template_constructor_call_uses_the_aliased_type(statement, expected):
    crossgl = convert(statement + "\nvalues[gid] = 0.0f;")

    assert expected in crossgl
    assert "_u3c" not in crossgl


def test_alias_template_constructor_argument_selects_the_vector_overload():
    source = PRELUDE + """
float total(float2 v) { return v.x + v.y; }
float total(float v) { return v; }

kernel void fill(device float* values [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    values[gid] = total(Pair<float>(values[gid], 1.0f)) + Pair<float>(2.0f).y;
}
"""
    ast = MetalParser(MetalLexer(source).tokenize()).parse()
    crossgl = MetalToCrossGLConverter().generate(ast)

    assert "total(vec2(buffer_load(values, gid), 1.0f)) + vec2(2.0f).y" in crossgl
    assert "Pair" not in crossgl.split("// Compute Shader", 1)[-1]
