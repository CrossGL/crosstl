"""Exact scalar constant conversions for bfloat values and widened carriers."""

import struct

from ..ast import (
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    UnaryOpNode,
)
from .array_utils import _UnsignedLiteralInt, evaluate_literal_int_expression


def _integer_float_bits(value, fraction_bits):
    """Round an integer directly to a format with an eight-bit exponent."""
    magnitude = abs(value)
    sign = int(value < 0) << (fraction_bits + 8)
    if not magnitude:
        return sign
    exponent = magnitude.bit_length() - 1
    shift = exponent - fraction_bits
    if shift > 0:
        significand, remainder = divmod(magnitude, 1 << shift)
        midpoint = 1 << (shift - 1)
        significand += remainder > midpoint or (
            remainder == midpoint and significand & 1
        )
    else:
        significand = magnitude << -shift
    # Adding the significand also handles a carry into the next exponent.
    return sign | ((exponent + 126) << fraction_bits) + significand


def bfloat16_constant_bits(value):
    """Round integers directly, or binary32 values, to nearest-even bfloat."""
    if isinstance(value, int):
        return _integer_float_bits(value, 7)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0x7FFFFFFF > 0x7F800000:
        return ((bits >> 16) | 0x40) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bfloat16_constant_float(value):
    return struct.unpack("<f", struct.pack("<I", bfloat16_constant_bits(value) << 16))[
        0
    ]


def _type_name(node):
    name = node if isinstance(node, str) else getattr(node, "name", "")
    return name.rsplit("::", 1)[-1]


def scalar_constant_value(expression, *, constants=None, function_names=()):
    """Evaluate literals and scalar casts without erasing conversion boundaries.

    Arithmetic, unsigned negation, narrowing integer casts and user functions
    are left to the target's checked runtime path or constant diagnostic.
    """
    if isinstance(expression, (int, float)):
        return expression
    if isinstance(expression, IdentifierNode):
        return (constants or {}).get(expression.name)
    if isinstance(expression, LiteralNode):
        name = _type_name(expression.literal_type)
        value = expression.value
        if name in {"float", "float32", "float32_t", "f32"}:
            try:
                return struct.unpack("<f", struct.pack("<f", value))[0]
            except (OverflowError, struct.error):
                return None
        if name == "bool":
            return bool(value)
        value = evaluate_literal_int_expression(expression)
        return _checked_integer(value, name)
    if isinstance(expression, UnaryOpNode) and expression.op in {"+", "-"}:
        value = scalar_constant_value(
            expression.operand, constants=constants, function_names=function_names
        )
        if value is None or isinstance(value, _UnsignedLiteralInt):
            return None
        return value if expression.op == "+" else -value
    if isinstance(expression, CastNode):
        name, arguments = _type_name(expression.target_type), [expression.expression]
    elif isinstance(expression, ConstructorNode):
        name, arguments = _type_name(expression.constructor_type), expression.arguments
    elif isinstance(expression, FunctionCallNode):
        name, arguments = _type_name(expression.function), expression.arguments
        if name in function_names:
            return None
    else:
        return None
    if len(arguments) > 1:
        return None
    value = (
        scalar_constant_value(
            arguments[0], constants=constants, function_names=function_names
        )
        if arguments
        else 0
    )
    if value is None:
        return None
    try:
        if name in {"bfloat", "bfloat16", "bfloat16_t"}:
            return bfloat16_constant_float(value)
        if name in {"float", "float32", "float32_t", "f32"}:
            if isinstance(value, int):
                return struct.unpack(
                    "<f", struct.pack("<I", _integer_float_bits(value, 23))
                )[0]
            return struct.unpack("<f", struct.pack("<f", value))[0]
        if name in {"half", "float16", "float16_t", "f16"}:
            return struct.unpack("<e", struct.pack("<e", value))[0]
    except (OverflowError, struct.error):
        return None
    return _checked_integer(value, name)


def _checked_integer(value, name):
    if not isinstance(value, int):
        return None
    for bits, signed_names, unsigned_names in (
        (8, {"char", "int8", "int8_t", "i8"}, {"uchar", "uint8", "uint8_t", "u8"}),
        (
            16,
            {"short", "int16", "int16_t", "i16"},
            {"ushort", "uint16", "uint16_t", "u16"},
        ),
        (32, {"int", "int32", "int32_t", "i32"}, {"uint", "uint32", "uint32_t", "u32"}),
        (
            64,
            {"long", "int64", "int64_t", "i64"},
            {"ulong", "uint64", "uint64_t", "u64", "size_t"},
        ),
    ):
        if name in signed_names:
            return int(value) if -(1 << (bits - 1)) <= value < 1 << (bits - 1) else None
        if name in unsigned_names:
            return _UnsignedLiteralInt(value) if 0 <= value < 1 << bits else None
    return None
