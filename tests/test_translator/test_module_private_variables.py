"""Module-scope CrossGL variables retain private mutable storage."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen


@pytest.mark.parametrize("address_space", ["<private>", ""])
def test_module_mutable_variable_is_private_on_source_backends(address_space):
    source = f"""var{address_space} counter: i32 = 3;

shader P {{
    compute {{
        @numthreads(1, 1, 1)
        void main() {{ counter = counter + 4; }}
    }}
}}
"""

    ast = parse(source)
    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(parse(source))
    slang = SlangCodeGen().generate(parse(source))

    assert "static int counter = 3;" in hlsl
    assert "int counter = 3;" in glsl
    assert "uniform int counter" not in glsl
    assert "static int counter = 3;" in slang
