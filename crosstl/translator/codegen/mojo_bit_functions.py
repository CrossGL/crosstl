"""GLSL's integer bit functions in Mojo output, on the ``bit`` module of the Mojo standard library.

The generator wrote ``bitCount``, ``bitfieldReverse``, ``findLSB`` and ``findMSB`` calls as they stand, which name
nothing in Mojo. (:mod:`~crosstl.translator.semantics.bit_functions` has already spelled HLSL's ``countbits``,
``reversebits``, ``firstbitlow`` and ``firstbithigh`` as these.) For 32-bit integer operands, scalar or SIMD:

- ``bitCount(x)`` is ``pop_count(x)``, converted to ``Int32`` lanes as GLSL 4.60 section 8.8 returns them;
- ``bitfieldReverse(x)`` is ``bit_reverse(x)``, of ``x``'s type;
- ``findLSB(x)`` is ``31 - count_leading_zeros(x & (~x + 1))``: ``x & -x`` keeps the lowest set bit, and for 0 it is
  0, whose 32 leading zeros give GLSL's -1;
- ``findMSB(x)`` is ``31 - count_leading_zeros(x)`` for an unsigned ``x``, and ``31 - count_leading_zeros(x ^ (x >>
  31))`` for a signed one: the arithmetic shift complements a negative value, whose most significant 0 bit GLSL
  reports, and both 0 and -1 give -1.

The functions are those of Mojo 1.x's ``std.bit`` module. A call on an operand of another width is left to the
foreign-call refusal, since the constant 31 would be wrong for it.
"""

from __future__ import annotations

__all__ = ["MOJO_BIT_FUNCTIONS", "MOJO_BIT_IMPORTS", "generate_mojo_bit_function_call"]

MOJO_BIT_FUNCTIONS = frozenset({"bitCount", "bitfieldReverse", "findLSB", "findMSB"})
# The std.bit functions the lowerings call, imported by name.
MOJO_BIT_IMPORTS = frozenset({"pop_count", "bit_reverse", "count_leading_zeros"})

_SIGNED_32 = "DType.int32"
_UNSIGNED_32 = "DType.uint32"


def generate_mojo_bit_function_call(generator, func_name, args):
    """The Mojo spelling of a GLSL bit function call on a 32-bit operand, or ``None``."""
    if func_name not in MOJO_BIT_FUNCTIONS or len(args) != 1:
        return None
    if generator.is_user_defined_function(func_name):
        return None
    dtype = generator.expression_mojo_dtype(args[0])
    if dtype not in (_SIGNED_32, _UNSIGNED_32):
        return None
    value = f"({generator.generate_expression(args[0])})"
    if func_name == "bitfieldReverse":
        generator.required_bit_imports.add("bit_reverse")
        return f"bit_reverse({value})"
    if func_name == "bitCount":
        generator.required_bit_imports.add("pop_count")
        return f"pop_count({value}).cast[DType.int32]()"
    generator.required_bit_imports.add("count_leading_zeros")
    if func_name == "findLSB":
        isolated = f"({value} & (~{value} + 1))"
    elif dtype == _SIGNED_32:
        isolated = f"({value} ^ ({value} >> 31))"
    else:
        isolated = value
    return f"(31 - count_leading_zeros({isolated})).cast[DType.int32]()"
