"""WGSL operands of integer shift and bitwise operators.

HLSL, GLSL, MSL and CrossGL accept a scalar shift amount for a vector, and a signed amount. A WGSL shift instead takes
an amount of type ``u32``, or ``vecN<u32>`` when the shifted value is a ``vecN`` (WGSL, "Bit Expressions"). Both
languages use only the low bits of the amount (HLSL and DXIL mask it to the operand width, WGSL takes it modulo the
width), so converting a signed amount to ``u32`` keeps those bits.

WGSL bitwise ``&``, ``|`` and ``^`` need both operands of one type, while the C-style languages replicate a scalar
across a vector operand. A scalar whose type already matches the vector's components, or an untyped literal, is
replicated. When one operand is ``uint`` or ``uvecN`` and the other ``int`` or ``ivecN``, the signed operand is
converted to the unsigned type first, as GLSL 4.60 does for ``&``, ``|`` and ``^`` (sections 5.9 and 4.1.10) and
for the overloads of ``min``, ``max`` and ``clamp``; WGSL has no implicit conversions.
"""

from __future__ import annotations

from ..ast import LiteralNode, UnaryOpNode

__all__ = [
    "BITWISE_OPERATORS",
    "MIXED_INTEGER_OPERATORS",
    "SHIFT_OPERATORS",
    "is_abstract_integer_literal",
    "mixed_integer_call_is_unsigned",
    "mixed_integer_call_type",
    "unsigned_bitwise_operand_types",
    "wgsl_bitwise_operands",
    "wgsl_mixed_integer_call_arguments",
    "wgsl_mixed_integer_operands",
    "wgsl_shift_amount",
]

SHIFT_OPERATORS = frozenset({"<<", ">>"})
BITWISE_OPERATORS = frozenset({"&", "|", "^"})
MIXED_INTEGER_COMPARISON_OPERATORS = frozenset({"<", "<=", ">", ">=", "==", "!="})
MIXED_INTEGER_ARITHMETIC_OPERATORS = frozenset({"+", "-", "*", "/", "%"})
MIXED_INTEGER_OPERATORS = (
    MIXED_INTEGER_COMPARISON_OPERATORS | MIXED_INTEGER_ARITHMETIC_OPERATORS
)


def _type_name(codegen, vtype):
    if vtype is None:
        return None
    try:
        return codegen.type_name_string(vtype)
    except ValueError:
        return None


def _is_integer_literal(expr):
    if isinstance(expr, UnaryOpNode) and expr.operator in {"-", "+"}:
        return _is_integer_literal(expr.operand)
    return isinstance(expr, LiteralNode) and isinstance(expr.value, int)


def _is_non_negative_integer_literal(expr):
    return (
        isinstance(expr, LiteralNode)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
        and expr.value >= 0
    )


def is_abstract_integer_literal(expr):
    """True if ``expr`` is an untyped non-negative integer literal, an abstract int in WGSL."""
    if isinstance(expr, UnaryOpNode) and expr.operator == "+":
        return is_abstract_integer_literal(expr.operand)
    if not (
        isinstance(expr, LiteralNode)
        and isinstance(expr.value, int)
        and not isinstance(expr.value, bool)
        and expr.value >= 0
    ):
        return False
    literal_type = getattr(expr, "literal_type", None)
    if literal_type is not None:
        type_name = str(getattr(literal_type, "name", literal_type))
        if type_name in {"uint", "u32"}:
            return False
    return True


def _integer_component(codegen, vtype):
    """``i32``, ``u32`` (or another scalar name) for a scalar or vector type, else None."""
    shape = codegen.vector_shape(vtype)
    if shape is not None:
        return shape[0]
    return codegen.scalar_type_name(vtype)


def _unsigned_type(codegen, vtype):
    shape = codegen.vector_shape(vtype)
    return "u32" if shape is None else f"vec{shape[1]}<u32>"


def _unsigned(codegen, expr, rendered, vtype):
    """``rendered`` of signed ``expr`` converted to the unsigned type of the same shape."""
    converted = codegen.concrete_negative_integer_literal(expr, rendered)
    return f"{_unsigned_type(codegen, vtype)}({converted})"


def _mixes_signedness(codegen, arguments, argument_types):
    components = {
        _integer_component(codegen, arg_type)
        for arg, arg_type in zip(arguments, argument_types)
        if not is_abstract_integer_literal(arg)
    }
    return components >= {"i32", "u32"}


def _call_vector_size(codegen, argument_types):
    sizes = {
        shape[1]
        for shape in (codegen.vector_shape(arg_type) for arg_type in argument_types)
        if shape is not None
    }
    return sizes.pop() if len(sizes) == 1 else None


def wgsl_mixed_integer_call_arguments(codegen, arguments):
    """The WGSL spellings of arguments to min, max or clamp with mixed integer types.

    A signed argument next to an unsigned one is converted to ``u32``, or to ``vecN<u32>`` for a vector. When the
    call takes a vector, a scalar argument is also replicated to ``vecN<u32>``, since WGSL's ``min``, ``max`` and
    ``clamp`` take arguments of one type.
    """
    argument_types = [codegen.expression_type(arg) for arg in arguments]
    rendered = [codegen.generate_expression(arg) for arg in arguments]
    if not _mixes_signedness(codegen, arguments, argument_types):
        return rendered
    vector_size = _call_vector_size(codegen, argument_types)
    for index, arg in enumerate(arguments):
        arg_type = argument_types[index]
        if _integer_component(codegen, arg_type) == "i32" and not (
            is_abstract_integer_literal(arg)
        ):
            rendered[index] = _unsigned(codegen, arg, rendered[index], arg_type)
        if vector_size is not None and codegen.vector_shape(arg_type) is None:
            rendered[index] = f"vec{vector_size}<u32>({rendered[index]})"
    return rendered


def mixed_integer_call_is_unsigned(codegen, arguments):
    """Whether min, max or clamp of ``arguments`` mixes ``uint`` and ``int`` values, so its WGSL call is unsigned.

    Abstract integer literals take the other arguments' type and do not count.
    """
    argument_types = [codegen.expression_type(arg) for arg in arguments]
    return _mixes_signedness(codegen, arguments, argument_types)


def mixed_integer_call_type(codegen, arguments):
    """The WGSL type of a min, max or clamp call that mixes ``uint`` and ``int`` arguments: ``u32`` or ``vecN<u32>``."""
    vector_size = _call_vector_size(
        codegen, [codegen.expression_type(arg) for arg in arguments]
    )
    return "u32" if vector_size is None else f"vec{vector_size}<u32>"


def wgsl_mixed_integer_operands(codegen, left, right):
    """The WGSL spellings of mixed signed/unsigned integer operands."""
    arguments = (left, right)
    argument_types = [codegen.expression_type(arg) for arg in arguments]
    shapes = [codegen.vector_shape(arg_type) for arg_type in argument_types]
    scalar_types = [
        codegen.scalar_type_name(arg_type) if shape is None else shape[0]
        for arg_type, shape in zip(argument_types, shapes)
    ]
    rendered = [codegen.generate_expression(arg) for arg in arguments]

    if "u32" in scalar_types:
        for index, arg in enumerate(arguments):
            if scalar_types[index] != "i32" or is_abstract_integer_literal(arg):
                continue

            vector_shape = shapes[index]
            target_type = (
                f"vec{vector_shape[1]}<u32>" if vector_shape is not None else "u32"
            )
            converted = codegen.concrete_negative_integer_literal(arg, rendered[index])
            rendered[index] = f"{target_type}({converted})"

    return tuple(rendered)


def wgsl_shift_amount(codegen, amount, shifted_type, rendered=None):
    """The WGSL spelling of shift ``amount`` for a value of ``shifted_type``."""
    if rendered is None:
        rendered = codegen.generate_expression(amount)
    shifted_shape = codegen.vector_shape(shifted_type)
    amount_type = codegen.expression_type(amount)
    amount_shape = codegen.vector_shape(amount_type)
    if amount_shape is not None:
        if amount_shape[0] == "u32":
            return rendered
        return f"vec{amount_shape[1]}<u32>({rendered})"
    if _is_non_negative_integer_literal(amount) or _type_name(codegen, amount_type) in {
        "u32",
        None,
    }:
        scalar = rendered
    else:
        scalar = f"u32({rendered})"
    if shifted_shape is None:
        return scalar
    return f"vec{shifted_shape[1]}<u32>({scalar})"


def _replicated(codegen, scalar_expr, scalar_type, rendered, vector_type):
    shape = codegen.vector_shape(vector_type)
    if shape is None:
        return None
    if _type_name(codegen, scalar_type) != shape[0] and not _is_integer_literal(
        scalar_expr
    ):
        return None
    return f"{codegen.type_name_string(vector_type)}({rendered})"


def unsigned_bitwise_operand_types(codegen, left, right):
    """The operand types of ``left`` and ``right`` in a WGSL bitwise expression.

    A signed operand next to an unsigned one is converted (see :func:`wgsl_bitwise_operands`), so its type becomes
    the unsigned type of the same shape.
    """
    operands = (left, right)
    types = [codegen.expression_type(operand) for operand in operands]
    if not _mixes_signedness(codegen, operands, types):
        return tuple(types)
    return tuple(
        (
            _unsigned_type(codegen, operand_type)
            if _integer_component(codegen, operand_type) == "i32"
            and not is_abstract_integer_literal(operand)
            else operand_type
        )
        for operand, operand_type in zip(operands, types)
    )


def wgsl_bitwise_operands(codegen, left, right):
    """The WGSL spellings of the operands of a bitwise ``&``, ``|`` or ``^``."""
    left_rendered = codegen.generate_expression(left)
    right_rendered = codegen.generate_expression(right)
    left_type, right_type = unsigned_bitwise_operand_types(codegen, left, right)
    if left_type != codegen.expression_type(left):
        left_rendered = _unsigned(codegen, left, left_rendered, left_type)
    if right_type != codegen.expression_type(right):
        right_rendered = _unsigned(codegen, right, right_rendered, right_type)
    left_vector = codegen.vector_shape(left_type) is not None
    right_vector = codegen.vector_shape(right_type) is not None
    if left_vector and not right_vector and right_type is not None:
        replicated = _replicated(codegen, right, right_type, right_rendered, left_type)
        if replicated is not None:
            right_rendered = replicated
    elif right_vector and not left_vector and left_type is not None:
        replicated = _replicated(codegen, left, left_type, left_rendered, right_type)
        if replicated is not None:
            left_rendered = replicated
    return left_rendered, right_rendered
