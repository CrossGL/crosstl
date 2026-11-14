"""Opcode-determined integer signedness for imported SPIR-V instructions.

SPIR-V takes the signedness of an integer instruction from its opcode, not from the
types of its operands (SPIR-V 1.6 section 3.3.6 OpTypeInt: "the type of operation of
an instruction comes from the instruction's opcode, not the signedness of the
operands").  ``OpULessThan`` on two ``int`` values compares them as unsigned,
``OpShiftRightLogical`` on an ``int`` base shifts in zeros, and ``OpConvertSToF`` of a
``uint`` reads it as two's complement.

CrossGL, like GLSL, HLSL and MSL, takes the signedness of an operator from its operand
types.  An operand whose type disagrees with the opcode is therefore reinterpreted with
a same-width constructor of the opposite signedness, which preserves the bit pattern in
every target (GLSL 4.60 section 5.4.1, HLSL and MSL integral conversions modulo 2^n),
before the operation.  When the operation's type then differs from the instruction's
result type, the result is reinterpreted back.  Operands that already agree are left
alone, so output for modules whose types match their opcodes does not change.

``OpSMod`` is always lowered: its non-zero result takes the sign of the divisor
(SPIR-V 1.6 section 3.52.13), while CrossGL's ``%`` truncates and so takes the sign of
the dividend, as ``OpSRem`` does.  The truncated remainder ``r`` is corrected by adding
the divisor when ``r`` is non-zero and its sign differs from the divisor's; the sum
cannot overflow because the two have opposite signs.  The condition is computed with
bit operations, ``((r ^ y) & (r | -r)) >> (width - 1)``, so it applies per component
to vectors; ``-r`` cannot overflow because ``|r| < |y| <= 2^(width - 1)``.
"""

from .VulkanAst import BinaryOpNode, FunctionCallNode, UnaryOpNode

__all__ = [
    "spirv_integer_binary_expression",
    "spirv_integer_conversion_expression",
    "spirv_integer_type_name",
]

_SCALAR_INTEGERS = {
    "int": (True, "32"),
    "uint": (False, "32"),
    "i8": (True, "8"),
    "u8": (False, "8"),
    "i16": (True, "16"),
    "u16": (False, "16"),
    "i64": (True, "64"),
    "u64": (False, "64"),
}
_SCALAR_NAMES = {value: key for key, value in _SCALAR_INTEGERS.items()}

# Both operands are read with the opcode's signedness.
_SIGNED_OPERAND_OPS = {
    "SDiv": "/",
    "SRem": "%",
    "SMod": "%",
    "SLessThan": "<",
    "SLessThanEqual": "<=",
    "SGreaterThan": ">",
    "SGreaterThanEqual": ">=",
}
_UNSIGNED_OPERAND_OPS = {
    "UDiv": "/",
    "UMod": "%",
    "ULessThan": "<",
    "ULessThanEqual": "<=",
    "UGreaterThan": ">",
    "UGreaterThanEqual": ">=",
}
# Only the base is read with the opcode's signedness; the shift count is unsigned in
# SPIR-V and every target accepts a count of either signedness.
_RIGHT_SHIFT_OPS = {"ShiftRightArithmetic": True, "ShiftRightLogical": False}
# Two's-complement operations whose bits do not depend on signedness: operands are
# reinterpreted to the signedness of the result type so the operator is well typed.
_AGNOSTIC_OPS = {
    "IAdd": "+",
    "ISub": "-",
    "IMul": "*",
    "BitwiseAnd": "&",
    "BitwiseOr": "|",
    "BitwiseXor": "^",
}
_EQUALITY_OPS = {"IEqual": "==", "INotEqual": "!="}
_COMPARISON_OPS = {
    "SLessThan",
    "SLessThanEqual",
    "SGreaterThan",
    "SGreaterThanEqual",
    "ULessThan",
    "ULessThanEqual",
    "UGreaterThan",
    "UGreaterThanEqual",
}


def _integer_shape(type_id, types):
    """Return ``(signed, width, component_count)`` for an integer scalar or vector type."""
    info = types.get(type_id) or {}
    count = None
    if info.get("kind") == "vector":
        count = str(info.get("component_count"))
        info = types.get(info.get("component_type")) or {}
    if info.get("kind") != "scalar":
        return None
    shape = _SCALAR_INTEGERS.get(info.get("name"))
    if shape is None:
        return None
    return shape[0], shape[1], count


def _shape_name(signed, width, count, vector_type_name):
    scalar = _SCALAR_NAMES[(signed, width)]
    if count is None:
        return scalar
    return vector_type_name(scalar, count)


def spirv_integer_type_name(type_id, types, signed, vector_type_name):
    """The name of the integer type with the shape of ``type_id`` and the given signedness, else ``None``."""
    shape = _integer_shape(type_id, types)
    if shape is None:
        return None
    return _shape_name(signed, shape[1], shape[2], vector_type_name)


def _reinterpret(expression, shape, signed, vector_type_name):
    if shape is None or shape[0] == signed:
        return expression
    name = _shape_name(signed, shape[1], shape[2], vector_type_name)
    if name is None:
        return None
    return FunctionCallNode(name, [expression])


def _computed_shape(lhs_shape, signed):
    return (signed, lhs_shape[1], lhs_shape[2])


def _wrap_result(expression, computed, result_shape, vector_type_name):
    if result_shape is None or computed is None or computed[0] == result_shape[0]:
        return expression
    return _reinterpret(expression, computed, result_shape[0], vector_type_name)


def spirv_integer_binary_expression(
    operation,
    result_type_id,
    lhs,
    lhs_type_id,
    rhs,
    rhs_type_id,
    types,
    vector_type_name,
):
    """Lower a binary integer instruction whose operand types disagree with its opcode.

    Returns ``None`` when the operands already agree (the caller emits the plain
    operator) or when the operands are not integers.
    """
    lhs_shape = _integer_shape(lhs_type_id, types)
    rhs_shape = _integer_shape(rhs_type_id, types)
    result_shape = _integer_shape(result_type_id, types)
    if lhs_shape is None or rhs_shape is None:
        return None

    if operation == "SMod":
        return _signed_modulus(
            lhs, lhs_shape, rhs, rhs_shape, result_shape, vector_type_name
        )

    if operation in _SIGNED_OPERAND_OPS or operation in _UNSIGNED_OPERAND_OPS:
        signed = operation in _SIGNED_OPERAND_OPS
        operator = (_SIGNED_OPERAND_OPS if signed else _UNSIGNED_OPERAND_OPS)[operation]
        operands = (lhs_shape, rhs_shape)
        if all(shape[0] == signed for shape in operands) and (
            operation in _COMPARISON_OPS
            or result_shape is None
            or result_shape[0] == signed
        ):
            return None
        new_lhs = _reinterpret(lhs, lhs_shape, signed, vector_type_name)
        new_rhs = _reinterpret(rhs, rhs_shape, signed, vector_type_name)
        if new_lhs is None or new_rhs is None:
            return None
        expression = BinaryOpNode(new_lhs, operator, new_rhs)
        if operation in _COMPARISON_OPS:
            return expression
        return _wrap_result(
            expression,
            _computed_shape(lhs_shape, signed),
            result_shape,
            vector_type_name,
        )

    if operation in _RIGHT_SHIFT_OPS:
        signed = _RIGHT_SHIFT_OPS[operation]
        if lhs_shape[0] == signed and (
            result_shape is None or result_shape[0] == signed
        ):
            return None
        new_lhs = _reinterpret(lhs, lhs_shape, signed, vector_type_name)
        if new_lhs is None:
            return None
        return _wrap_result(
            BinaryOpNode(new_lhs, ">>", rhs),
            _computed_shape(lhs_shape, signed),
            result_shape,
            vector_type_name,
        )

    if operation == "ShiftLeftLogical" and result_shape is not None:
        if lhs_shape[0] == result_shape[0]:
            return None
        new_lhs = _reinterpret(lhs, lhs_shape, result_shape[0], vector_type_name)
        if new_lhs is None:
            return None
        return BinaryOpNode(new_lhs, "<<", rhs)

    if operation in _AGNOSTIC_OPS and result_shape is not None:
        signed = result_shape[0]
        if lhs_shape[0] == signed and rhs_shape[0] == signed:
            return None
        new_lhs = _reinterpret(lhs, lhs_shape, signed, vector_type_name)
        new_rhs = _reinterpret(rhs, rhs_shape, signed, vector_type_name)
        if new_lhs is None or new_rhs is None:
            return None
        return BinaryOpNode(new_lhs, _AGNOSTIC_OPS[operation], new_rhs)

    if operation in _EQUALITY_OPS:
        if lhs_shape[0] == rhs_shape[0]:
            return None
        new_rhs = _reinterpret(rhs, rhs_shape, lhs_shape[0], vector_type_name)
        if new_rhs is None:
            return None
        return BinaryOpNode(lhs, _EQUALITY_OPS[operation], new_rhs)

    return None


def _signed_modulus(lhs, lhs_shape, rhs, rhs_shape, result_shape, vector_type_name):
    """``OpSMod`` as the truncated remainder corrected toward the sign of the divisor."""
    dividend = _reinterpret(lhs, lhs_shape, True, vector_type_name)
    divisor = _reinterpret(rhs, rhs_shape, True, vector_type_name)
    if dividend is None or divisor is None:
        return None
    remainder = BinaryOpNode(dividend, "%", divisor)
    signs_differ = BinaryOpNode(remainder, "^", divisor)
    nonzero = BinaryOpNode(remainder, "|", UnaryOpNode("-", remainder))
    mask = BinaryOpNode(
        BinaryOpNode(signs_differ, "&", nonzero), ">>", str(int(lhs_shape[1]) - 1)
    )
    corrected = BinaryOpNode(remainder, "+", BinaryOpNode(divisor, "&", mask))
    return _wrap_result(
        corrected,
        _computed_shape(lhs_shape, True),
        result_shape,
        vector_type_name,
    )


# Conversions whose operand is read with a fixed signedness (True: signed).
_SOURCE_SIGNEDNESS = {
    "ConvertSToF": True,
    "SConvert": True,
    "ConvertUToF": False,
    "UConvert": False,
}


def spirv_integer_conversion_expression(
    operation,
    result_type_id,
    operand,
    operand_type_id,
    types,
    vector_type_name,
):
    """Lower a conversion whose source or result signedness disagrees with its opcode.

    ``OpConvertSToF``/``OpSConvert`` read a signed source and sign-extend;
    ``OpConvertUToF``/``OpUConvert`` read an unsigned source and zero-extend;
    ``OpConvertFToS`` produces a signed value whatever the result type's signedness
    (SPIR-V 1.6 section 3.52.11).  Returns ``None`` when the plain constructor of the
    result type is already exact.
    """
    result_shape = _integer_shape(result_type_id, types)
    source_signedness = _SOURCE_SIGNEDNESS.get(operation)
    if source_signedness is not None:
        # A conversion from a source of the opcode's signedness is exact as the plain
        # constructor of the result type: integral conversions sign- or zero-extend by
        # the source type and wrap modulo 2^n into the result type.
        operand_shape = _integer_shape(operand_type_id, types)
        if operand_shape is None or operand_shape[0] == source_signedness:
            return None
        source = _reinterpret(
            operand, operand_shape, source_signedness, vector_type_name
        )
        result_name = (types.get(result_type_id) or {}).get("name")
        if source is None or result_name is None:
            return None
        return FunctionCallNode(result_name, [source])

    if operation == "ConvertFToS" and result_shape is not None and not result_shape[0]:
        computed = (True, result_shape[1], result_shape[2])
        computed_name = _shape_name(*computed, vector_type_name)
        if computed_name is None:
            return None
        return _wrap_result(
            FunctionCallNode(computed_name, [operand]),
            computed,
            result_shape,
            vector_type_name,
        )
    return None
