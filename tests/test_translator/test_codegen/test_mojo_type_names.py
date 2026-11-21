"""Mojo output spells sized scalar names through its type table and refuses another language's built-in types.

A type name the Mojo type table did not map was written into the output as it stands, so Metal's
``visible_function_table`` became a Mojo variable type and CrossGL's ``float16`` a ``float16`` that Mojo does not
define.
"""

import pytest

from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader S {
    struct Payload { vec3 color; };
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            %s
        }
    }
}
"""


def mojo(body):
    code = PROGRAM % body
    return MojoCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())


@pytest.mark.parametrize(
    "body, name",
    [
        ("Payload p; visible_function_table vft; vft[0](p);", "visible_function_table"),
        ("command_buffer commands;", "command_buffer"),
        ("bool2x2 m;", "bool2x2"),
    ],
)
def test_foreign_built_in_types_are_refused(body, name):
    with pytest.raises(
        MojoLoweringRefused,
        match=f"the built-in type '{name}' has no Mojo spelling",
    ):
        mojo(body)


def test_a_program_struct_named_like_a_foreign_built_in_type_is_spelled():
    code = """
    shader S {
        struct intersector { float t; };
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() { intersector hit; hit.t = 1.0; }
        }
    }
    """
    out = MojoCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())
    assert "var hit: intersector" in out or "var hit = intersector(" in out


@pytest.mark.parametrize(
    "name, mojo_type",
    [
        ("float16", "Float16"),
        ("float16_t", "Float16"),
        ("float32", "Float32"),
        ("float64_t", "Float64"),
        ("int32_t", "Int32"),
        ("uint32", "UInt32"),
        ("float1", "Float32"),
        ("int1", "Int32"),
        ("float32_t3", "SIMD[DType.float32, 4]"),
        ("int16_t2", "SIMD[DType.int16, 2]"),
        ("uint8_t4", "SIMD[DType.uint8, 4]"),
    ],
)
def test_sized_scalar_and_vector_names_use_the_mojo_type_table(name, mojo_type):
    assert MojoCodeGen().map_type(name) == mojo_type
