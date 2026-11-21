"""Keep imported 8- and 16-bit SPIR-V integer arithmetic at its declared width.

SPIR-V computes an integer instruction at the width of its result type, so an 8- or 16-bit ``OpIAdd``, ``OpISub``,
``OpIMul``, ``OpShiftLeftLogical`` or ``OpSNegate`` wraps modulo 2^N (SPIR-V 1.6 section 3.52.13: the result
"equals the low-order N bits of the correct result R, where N is the component width"). CrossGL promotes a scalar 8- or 16-bit operand to
``int`` before arithmetic, as C++ [conv.prom] does (``PROMOTING_SOURCES`` in
:mod:`crosstl.translator.semantics.narrow`), so ``v * v`` on an ``i8`` holding 13 is 169 in CrossGL and -87 in
SPIR-V.

The importer therefore wraps such a scalar result in a constructor of its own type, ``i8(v * v)``. A conversion to a
narrow integer type keeps the value modulo 2^N (GLSL 4.60 section 5.4.1 and C++ [conv.integral]), which is the
SPIR-V result whatever the promoted operation computed. ``OpNot`` of an unsigned operand is wrapped too, because the
complement of a zero-extended value is negative; the complement of a sign-extended value is already the
sign-extended complement. Vector operations are not promoted (MSL section 2.2, OpenCL C 6.3), and division,
remainder, the other bitwise operations, right shifts and comparisons cannot leave the range of their operands, so
they are left alone. ``OpSDiv`` of the minimum value by -1 is undefined in SPIR-V and needs nothing either.
"""

from .VulkanAst import FunctionCallNode

__all__ = ["spirv_narrow_integer_result"]

_NARROW_SCALARS = {"i8": True, "u8": False, "i16": True, "u16": False}
_WRAPPING_OPERATIONS = frozenset(
    {"IAdd", "ISub", "IMul", "ShiftLeftLogical", "SNegate"}
)


def spirv_narrow_integer_result(operation, result_type_id, expression, types):
    """``expression`` converted to its 8- or 16-bit scalar result type when CrossGL promotion could widen it.

    Returns ``expression`` unchanged for every other operation and result type.
    """
    info = types.get(result_type_id) or {}
    if info.get("kind") != "scalar":
        return expression
    name = info.get("name")
    signed = _NARROW_SCALARS.get(name)
    if signed is None:
        return expression
    if operation in _WRAPPING_OPERATIONS or (operation == "Not" and not signed):
        return FunctionCallNode(name, [expression])
    return expression
