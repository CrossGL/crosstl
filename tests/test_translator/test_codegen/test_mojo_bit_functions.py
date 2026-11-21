"""GLSL's integer bit functions lower to Mojo's ``std.bit`` functions with GLSL's results.

The Mojo generator wrote ``bitCount``, ``bitfieldReverse``, ``findLSB`` and ``findMSB`` as they stand, which name
nothing in Mojo; the program is now spelled with ``pop_count``, ``bit_reverse`` and ``count_leading_zeros``.
"""

import pytest

from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader Bits {
    int msb(int v) { return findMSB(v); }
    int msbu(uint v) { return findMSB(v); }
    int lsb(int v) { return findLSB(v); }
    int count(uint v) { return bitCount(v); }
    uint reverse(uint v) { return bitfieldReverse(v); }
    ivec4 msb4(ivec4 v) { return findMSB(v); }
}
"""


def mojo(code):
    return MojoCodeGen().generate(Parser(Lexer(code).get_tokens()).parse())


def test_bit_functions_are_spelled_with_std_bit():
    generated = mojo(PROGRAM)

    assert (
        "from std.bit import bit_reverse, count_leading_zeros, pop_count" in generated
    )
    for name in ("bitCount", "bitfieldReverse", "findLSB", "findMSB"):
        assert f"{name}(" not in generated
    # A signed operand's arithmetic shift complements a negative value; 0 and -1 give -1.
    assert (
        "(31 - count_leading_zeros(((v) ^ ((v) >> 31)))).cast[DType.int32]()"
        in generated
    )
    assert "(31 - count_leading_zeros((v))).cast[DType.int32]()" in generated
    # The lowest set bit, or 0, whose 32 leading zeros give -1.
    assert (
        "(31 - count_leading_zeros(((v) & (~(v) + 1)))).cast[DType.int32]()"
        in generated
    )
    assert "pop_count((v)).cast[DType.int32]()" in generated
    assert "bit_reverse((v))" in generated


def test_bit_function_on_a_narrow_operand_is_refused():
    from crosstl.translator.codegen.mojo_refusals import MojoLoweringRefused

    with pytest.raises(MojoLoweringRefused) as refused:
        mojo("shader S { int f(int16 v) { return findMSB(v); } }")

    assert "call to 'findMSB'" in str(refused.value)
