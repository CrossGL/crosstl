"""The Mojo generator refuses a call to another language's built-in function it has no lowering for.

A call the generator could not map was written into the Mojo output as it stands (``lit(a, a, a)``, ``fwidth(a)``,
``acosf(a)``, ``simd_sum(a)``), which names nothing in Mojo's ``math`` or ``gpu`` modules.
"""

import pytest

from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused
from crosstl.translator.codegen.written_function_names import (
    MOJO_CALLABLE_FUNCTION_NAMES,
    MOJO_NON_CALLS,
    written_foreign_call_names,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader S {
    compute {
        float helper(float x) { return x * 2.0; }
        void main() {
            float a = 1.0;
            vec3 v = vec3(a, a, a);
            float b = %s;
            float c = helper(b);
        }
    }
}
"""


def mojo(expression):
    code = PROGRAM % expression
    return MojoCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())


@pytest.mark.parametrize(
    "expression, name",
    [
        ("lit(a, a, a).y", "lit"),
        ("fwidth(a)", "fwidth"),
        ("acosf(a)", "acosf"),
        ("simd_sum(a)", "simd_sum"),
        ("msad4(a, a, a)", "msad4"),
        ("ddx(a)", "ddx"),
        ("rcp(a)", "rcp"),
    ],
)
def test_foreign_builtin_call_is_refused(expression, name):
    with pytest.raises(MojoLoweringRefused) as refused:
        mojo(expression)

    assert f"call to '{name}'" in str(refused.value)


@pytest.mark.parametrize(
    "expression, written",
    [
        ("sin(a)", "sin(a)"),
        ("floor(a)", "floor(a)"),
        ("log2(a)", "log2(a)"),
        ("atan(a, a)", "atan2(a, a)"),
        ("fma(a, a, a)", "fma(a, a, a)"),
        ("length(v)", "magnitude(v)"),
        ("helper(a)", "helper(a)"),
    ],
)
def test_mojo_library_and_program_calls_are_written(expression, written):
    assert f"var b: Float32 = {written}" in mojo(expression)


def test_mojo_comments_keywords_and_declarations_are_not_foreign_calls():
    code = (
        "fn sign(x: Float32) -> Float32:\n"
        "    return (x)  # lit(x) in a comment\n"
        "fn main():\n"
        '    var s = "fwidth(x)"\n'
        "    var c = a and (b) or not (d)\n"
        "    var t = sign(1.0)\n"
    )

    assert (
        written_foreign_call_names(code, MOJO_CALLABLE_FUNCTION_NAMES, MOJO_NON_CALLS)
        == []
    )


def test_undeclared_foreign_call_is_found_in_mojo_text():
    code = "fn main():\n    var t = sign(1.0) + fwidth(2.0)\n"

    assert written_foreign_call_names(
        code, MOJO_CALLABLE_FUNCTION_NAMES, MOJO_NON_CALLS
    ) == ["fwidth", "sign"]
