"""OpenCL C import: a program-scope ``const`` variable with a constant initializer is a CrossGL constant."""

import pytest

import crosstl
from crosstl.backend.OpenCL.OpenCLCrossGLCodeGen import OpenCLToCrossGLConverter
from crosstl.backend.OpenCL.OpenCLLexer import OpenCLLexer
from crosstl.backend.OpenCL.OpenCLParser import OpenCLParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser

KERNEL = """
__kernel void k(__global int *dst) {
  dst[get_global_id(0)] = %s;
}
"""


def to_crossgl(source):
    ast = OpenCLParser(OpenCLLexer(source).tokenize()).parse()
    crossgl = OpenCLToCrossGLConverter().generate(ast)
    CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse()
    return crossgl


@pytest.mark.parametrize(
    "declaration, expected",
    [
        ("const int c = 1;", "const i32 c = 1;"),
        ("global const int c = 2;", "const i32 c = 2;"),
        ("__global const int c = 2;", "const i32 c = 2;"),
        ("static const int c = 3;", "const i32 c = 3;"),
        ("int const c = 4;", "const i32 c = 4;"),
        ("const float c = 1.5f;", "const f32 c = 1.5f;"),
        ("const uint c = 2u * 3u;", "const u32 c = (2u * 3u);"),
    ],
)
def test_program_scope_const_imports_as_crossgl_constant(declaration, expected):
    crossgl = to_crossgl(declaration + KERNEL % "c")

    assert expected in crossgl
    assert "var c" not in crossgl
    assert "var<uniform> c" not in crossgl


def test_program_scope_constant_can_name_an_earlier_constant():
    crossgl = to_crossgl(
        "const int base = 4;\nconst int twice = base * 2;" + KERNEL % "twice"
    )

    assert "const i32 base = 4;" in crossgl
    assert "const i32 twice = (base * 2);" in crossgl


def test_constant_address_space_variable_keeps_its_import():
    # __constant data the host can bind stays a uniform with its initializer.
    crossgl = to_crossgl("__constant int c = 4;" + KERNEL % "c")

    assert "var<uniform> c: i32 = 4;" in crossgl


def test_program_scope_constants_are_compile_time_constants_in_hlsl(tmp_path):
    source = tmp_path / "consts.cl"
    source.write_text(
        "const int c1 = 1;\nglobal const int c2 = 2;\nstatic const int c3 = 3;\n"
        + KERNEL % "c1 + c2 + c3"
    )

    hlsl = crosstl.translate(str(source), backend="directx")

    for name, value in (("c1", 1), ("c2", 2), ("c3", 3)):
        assert f"static const int {name} = {value};" in hlsl
    # A constant is not host data, so it takes no place in a constant buffer.
    assert "cbuffer" not in hlsl
