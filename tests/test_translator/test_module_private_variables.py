"""Module-scope CrossGL variables retain private mutable storage."""

import re

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen


def test_module_private_variable_is_private_on_source_backends():
    source = """var<private> counter: i32 = 3;

shader P {
    compute {
        @numthreads(1, 1, 1)
        void main() { counter = counter + 4; }
    }
}
"""

    ast = parse(source)
    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(parse(source))
    slang = SlangCodeGen().generate(parse(source))

    assert "static int counter = 3;" in hlsl
    assert "int counter = 3;" in glsl
    assert "uniform int counter" not in glsl
    assert "static int counter = 3;" in slang


def test_plain_module_variables_are_private_when_written_or_read_only():
    source = """var counter: i32 = 3;
var limit: i32 = 7;

shader P {
    compute {
        @numthreads(1, 1, 1)
        void main() { counter = counter + limit; }
    }
}
"""

    ast = parse(source)
    assert [variable.qualifiers for variable in ast.global_variables] == [
        ["private"],
        ["private"],
    ]

    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(parse(source))
    slang = SlangCodeGen().generate(parse(source))

    assert "static int counter = 3;" in hlsl
    assert "static int limit = 7;" in hlsl
    assert "int counter = 3;" in glsl
    assert "int limit = 7;" in glsl
    assert not re.search(
        r"^\s*(?:layout\s*\([^\n]*\)\s*)?uniform\b[^;]*\b(?:counter|limit)\b",
        glsl,
        re.MULTILINE,
    )
    assert "static int counter = 3;" in slang
    assert "static int limit = 7;" in slang
