"""Fold scalar CrossGL constant expressions with the target's arithmetic.

Module-scope constants must be emitted as constant values by targets such as
SPIR-V, so an initializer like ``-INFINITY`` or ``A * 3.0 + 1.0`` has to be
evaluated during translation. The evaluation here follows the arithmetic the
shader would perform at run time:

* ``float`` values are binary32 and are rounded after every operation; ``double``
  values are binary64;
* 32- and 64-bit integers wrap, and ``int``/``uint`` conversions keep the bit
  pattern;
* operands take the usual arithmetic conversions (``bool`` < ``int`` < ``uint`` <
  ``i64`` < ``u64`` < ``float`` < ``double``).

Anything whose result the source languages leave undefined (integer division by
zero, out-of-range shifts, float-to-integer conversion of an unrepresentable
value, a math function outside its domain) is not folded: the caller sees
``None`` and must refuse the construct instead of inventing a value.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional, Union

from ..ast import (
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)

ScalarValue = Union[bool, int, float]

FLOAT_KINDS = ("float", "double")
INTEGER_KINDS = {"int": (32, True), "uint": (32, False), "i64": (64, True)}
INTEGER_KINDS["u64"] = (64, False)
SCALAR_KINDS = ("bool", "int", "uint", "i64", "u64", "float", "double")
_KIND_RANK = {kind: rank for rank, kind in enumerate(SCALAR_KINDS)}


@dataclass(frozen=True)
class ScalarConstant:
    """A folded scalar: its CrossGL scalar kind and its exact value."""

    kind: str
    value: ScalarValue


NameResolver = Callable[[str], Optional[ScalarConstant]]
KindResolver = Callable[[object], Optional[str]]


def round_to_binary32(value: float) -> float:
    """Return *value* rounded to the nearest binary32 value, ties to even."""
    if not math.isfinite(value):
        return value
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def shortest_binary32_text(value: float) -> str:
    """Return the shortest decimal text that reads back as binary32 *value*."""
    if not math.isfinite(value) or value == 0.0:
        return repr(value)
    for digits in range(1, 10):
        text = f"{value:.{digits}g}"
        if round_to_binary32(float(text)) == value:
            return text
    return repr(value)


def canonical_float_value(kind: str, value: float) -> float:
    """Return a Python float whose shortest repr reads back as *value* in *kind*."""
    if kind != "float" or not math.isfinite(value):
        return value
    return float(shortest_binary32_text(value))


def _wrap_integer(kind: str, value: int) -> int:
    width, signed = INTEGER_KINDS[kind]
    value &= (1 << width) - 1
    if signed and value >= 1 << (width - 1):
        value -= 1 << width
    return value


def _int_to_float(kind: str, value: int) -> float:
    if kind == "double" or abs(value) < 1 << 53:
        result = float(value)
        return round_to_binary32(result) if kind == "float" else result
    # Round to 24 significant bits directly, so a 64-bit value is not rounded
    # twice (first to binary64, then to binary32).
    magnitude = abs(value)
    shift = magnitude.bit_length() - 24
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return round_to_binary32(math.copysign(float(quotient << shift), value))


def convert_scalar_constant(
    constant: ScalarConstant, kind: str
) -> ScalarConstant | None:
    """Convert *constant* to *kind* as a constructor or implicit conversion would."""
    if kind not in _KIND_RANK:
        return None
    source, value = constant.kind, constant.value
    if source == kind:
        return constant
    if kind == "bool":
        return ScalarConstant("bool", bool(value != 0))
    if source == "bool":
        value = int(value)
        source = "int"
    if kind in FLOAT_KINDS:
        if source in FLOAT_KINDS:
            value = float(value)
            return ScalarConstant(
                kind, round_to_binary32(value) if kind == "float" else value
            )
        return ScalarConstant(kind, _int_to_float(kind, int(value)))
    if source in FLOAT_KINDS:
        if not math.isfinite(value):
            return None
        truncated = math.trunc(value)
        width, signed = INTEGER_KINDS[kind]
        low = -(1 << (width - 1)) if signed else 0
        high = (1 << (width - 1)) - 1 if signed else (1 << width) - 1
        if not low <= truncated <= high:
            return None
        return ScalarConstant(kind, truncated)
    return ScalarConstant(kind, _wrap_integer(kind, int(value)))


def _integer_literal(value: int, kind: str) -> ScalarConstant | None:
    if kind in FLOAT_KINDS:
        return ScalarConstant(kind, _int_to_float(kind, value))
    if kind == "bool":
        return ScalarConstant("bool", value != 0)
    if kind not in INTEGER_KINDS:
        return None
    width, _signed = INTEGER_KINDS[kind]
    # A literal that fits the width's bit pattern (such as 0xFFFFFFFF typed
    # int) keeps that pattern; anything wider is not a value of the type.
    if not -(1 << (width - 1)) <= value < 1 << width:
        return None
    return ScalarConstant(kind, _wrap_integer(kind, value))


def _normalized(constant: ScalarConstant) -> ScalarConstant | None:
    """Return *constant* with its value represented exactly in its kind."""
    kind, value = constant.kind, constant.value
    if kind == "bool":
        return ScalarConstant(kind, bool(value))
    if isinstance(value, bool):
        value = int(value)
    if kind in FLOAT_KINDS:
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return ScalarConstant(
            kind, round_to_binary32(value) if kind == "float" else value
        )
    if isinstance(value, float):
        if not value.is_integer():
            return None
        value = int(value)
    if not isinstance(value, int):
        return None
    return ScalarConstant(kind, _wrap_integer(kind, value))


def common_kind(left: str, right: str, *, arithmetic: bool = True) -> str:
    kind = max(left, right, key=_KIND_RANK.__getitem__)
    if arithmetic and kind == "bool":
        return "int"
    return kind


def _float_result(kind: str, value: float) -> ScalarConstant:
    return ScalarConstant(kind, round_to_binary32(value) if kind == "float" else value)


def _float_divide(left: float, right: float) -> float:
    if right != 0.0:
        return left / right
    if left == 0.0 or math.isnan(left):
        return math.nan
    negative = (math.copysign(1.0, left) < 0) != (math.copysign(1.0, right) < 0)
    return -math.inf if negative else math.inf


_FLOAT_BINARY = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _float_divide,
}

_LOGICAL_SPELLINGS = {"and": "&&", "or": "||"}

_COMPARISONS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _integer_binary(kind: str, operator: str, left: int, right: int):
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator in {"/", "%"}:
        if right == 0:
            return None
        width, signed = INTEGER_KINDS[kind]
        if signed and left == -(1 << (width - 1)) and right == -1:
            return None
        quotient = abs(left) // abs(right)
        if (left < 0) != (right < 0):
            quotient = -quotient
        return quotient if operator == "/" else left - quotient * right
    if operator == "&":
        return left & right
    if operator == "|":
        return left | right
    if operator == "^":
        return left ^ right
    return None


class ScalarConstantFolder:
    """Evaluate scalar constant expressions; ``None`` means "not foldable"."""

    def __init__(
        self,
        resolve_name: NameResolver,
        kind_of_type: KindResolver,
        *,
        fold_builtin: None | (
            Callable[[str, list[ScalarConstant]], ScalarConstant | None]
        ) = None,
    ):
        self.resolve_name = resolve_name
        self.kind_of_type = kind_of_type
        self.fold_builtin = fold_builtin

    def fold(self, expr) -> ScalarConstant | None:
        """Return the folded value of *expr*, or ``None`` when it is not foldable."""
        if expr is None:
            return None
        if isinstance(expr, bool):
            return ScalarConstant("bool", expr)
        if isinstance(expr, int):
            return ScalarConstant("int", _wrap_integer("int", expr))
        if isinstance(expr, float):
            return ScalarConstant("float", round_to_binary32(expr))
        if isinstance(expr, str):
            return self.fold_name(expr)
        if isinstance(expr, LiteralNode):
            return self.fold_literal(expr)
        if isinstance(expr, (IdentifierNode, VariableNode)):
            name = getattr(expr, "name", None)
            return self.fold_name(name) if isinstance(name, str) else None
        if isinstance(expr, UnaryOpNode):
            return self.fold_unary(expr)
        if isinstance(expr, BinaryOpNode):
            return self.fold_binary(expr)
        if isinstance(expr, TernaryOpNode):
            return self.fold_ternary(expr)
        if isinstance(expr, CastNode):
            operand = self.fold(expr.expression)
            kind = self.kind_of_type(expr.target_type)
            if operand is None or kind is None:
                return None
            return convert_scalar_constant(operand, kind)
        if isinstance(expr, FunctionCallNode):
            return self.fold_call(expr)
        return None

    def fold_name(self, name: str | None) -> ScalarConstant | None:
        if not name:
            return None
        if name in {"true", "false"}:
            return ScalarConstant("bool", name == "true")
        constant = self.resolve_name(name)
        if constant is None or constant.kind not in _KIND_RANK:
            return None
        return _normalized(constant)

    def fold_literal(self, node: LiteralNode) -> ScalarConstant | None:
        value = node.value
        kind = self.kind_of_type(getattr(node, "literal_type", None))
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                value = lowered == "true"
            else:
                return None
        if isinstance(value, bool):
            return convert_scalar_constant(
                ScalarConstant("bool", value), kind or "bool"
            )
        if isinstance(value, int):
            return _integer_literal(value, kind or "int")
        if isinstance(value, float):
            # The literal is read as binary64 first, then converted once.
            return convert_scalar_constant(
                ScalarConstant("double", value), kind or "float"
            )
        return None

    def fold_unary(self, node: UnaryOpNode) -> ScalarConstant | None:
        if getattr(node, "is_postfix", False):
            return None
        operand = self.fold(node.operand)
        if operand is None:
            return None
        return self.unary_value(node.operator, operand)

    def unary_value(self, operator, operand: ScalarConstant) -> ScalarConstant | None:
        """Apply unary *operator* to a folded operand."""
        if operand is None:
            return None
        kind = operand.kind
        if operator == "!":
            if kind != "bool":
                operand = convert_scalar_constant(operand, "bool")
            return ScalarConstant("bool", not operand.value)
        if kind == "bool":
            operand = convert_scalar_constant(operand, "int")
            kind = "int"
        if operator == "+":
            return operand
        if operator == "-":
            if kind in FLOAT_KINDS:
                return ScalarConstant(kind, -operand.value)
            return ScalarConstant(kind, _wrap_integer(kind, -operand.value))
        if operator == "~" and kind in INTEGER_KINDS:
            return ScalarConstant(kind, _wrap_integer(kind, ~operand.value))
        return None

    def fold_binary(self, node: BinaryOpNode) -> ScalarConstant | None:
        # Both operands must be constant expressions, even where the value of
        # && and || short-circuits.
        left = self.fold(node.left)
        if left is None:
            return None
        right = self.fold(node.right)
        if right is None:
            return None
        return self.binary_value(node.operator, left, right)

    def binary_value(
        self, operator, left: ScalarConstant, right: ScalarConstant
    ) -> ScalarConstant | None:
        """Apply binary *operator* to folded operands."""
        if left is None or right is None:
            return None
        if operator in _LOGICAL_SPELLINGS:
            operator = _LOGICAL_SPELLINGS[operator]
        if operator in {"&&", "||"}:
            left_bool = convert_scalar_constant(left, "bool").value
            right_bool = convert_scalar_constant(right, "bool").value
            if operator == "&&":
                return ScalarConstant("bool", left_bool and right_bool)
            return ScalarConstant("bool", left_bool or right_bool)
        if operator == "^^":
            left_bool = convert_scalar_constant(left, "bool").value
            right_bool = convert_scalar_constant(right, "bool").value
            return ScalarConstant("bool", left_bool != right_bool)
        if operator in {"<<", ">>"}:
            return self.fold_shift(operator, left, right)
        if operator in _COMPARISONS:
            kind = common_kind(left.kind, right.kind, arithmetic=False)
            left = convert_scalar_constant(left, kind)
            right = convert_scalar_constant(right, kind)
            if left is None or right is None:
                return None
            return ScalarConstant(
                "bool", bool(_COMPARISONS[operator](left.value, right.value))
            )
        kind = common_kind(left.kind, right.kind)
        left = convert_scalar_constant(left, kind)
        right = convert_scalar_constant(right, kind)
        if left is None or right is None:
            return None
        if kind in FLOAT_KINDS:
            if operator == "%":
                if right.value == 0.0 or not math.isfinite(left.value):
                    return None
                return _float_result(kind, math.fmod(left.value, right.value))
            function = _FLOAT_BINARY.get(operator)
            if function is None:
                return None
            return _float_result(kind, function(left.value, right.value))
        result = _integer_binary(kind, operator, left.value, right.value)
        if result is None:
            return None
        return ScalarConstant(kind, _wrap_integer(kind, result))

    def fold_shift(
        self, operator: str, left: ScalarConstant, right: ScalarConstant
    ) -> ScalarConstant | None:
        if left.kind == "bool":
            left = convert_scalar_constant(left, "int")
        if left.kind not in INTEGER_KINDS or right.kind not in INTEGER_KINDS:
            return None
        width, _signed = INTEGER_KINDS[left.kind]
        amount = right.value
        if not 0 <= amount < width:
            return None
        if operator == "<<":
            return ScalarConstant(
                left.kind, _wrap_integer(left.kind, left.value << amount)
            )
        # Python's >> is arithmetic for negative values and logical for
        # non-negative ones, which matches signed and unsigned SPIR-V shifts.
        return ScalarConstant(left.kind, left.value >> amount)

    def fold_ternary(self, node: TernaryOpNode) -> ScalarConstant | None:
        condition = self.fold(node.condition)
        true_value = self.fold(node.true_expr)
        false_value = self.fold(node.false_expr)
        if condition is None or true_value is None or false_value is None:
            return None
        return self.ternary_value(condition, true_value, false_value)

    def ternary_value(
        self,
        condition: ScalarConstant,
        true_value: ScalarConstant,
        false_value: ScalarConstant,
    ) -> ScalarConstant | None:
        """Select between folded operands as ``condition ? a : b`` would."""
        kind = common_kind(true_value.kind, false_value.kind, arithmetic=False)
        chosen = (
            true_value
            if convert_scalar_constant(condition, "bool").value
            else false_value
        )
        return convert_scalar_constant(chosen, kind)

    def fold_call(self, node: FunctionCallNode) -> ScalarConstant | None:
        callee = getattr(node, "function", None)
        name = getattr(callee, "name", callee)
        if not isinstance(name, str) or getattr(node, "generic_args", None):
            return None
        arguments = []
        for argument in getattr(node, "arguments", None) or []:
            folded = self.fold(argument)
            if folded is None:
                return None
            arguments.append(folded)
        return self.call_value(name, arguments)

    def call_value(self, name: str, arguments: list) -> ScalarConstant | None:
        """Fold a constructor or built-in call over folded *arguments*."""
        kind = self.kind_of_type(name)
        if kind is not None:
            if len(arguments) != 1:
                return None
            return convert_scalar_constant(arguments[0], kind)
        if self.fold_builtin is None or not arguments:
            return None
        return self.fold_builtin(name, arguments)


_UNARY_MATH = {
    "exp": math.exp,
    "exp2": lambda x: math.pow(2.0, x),
    "log": math.log,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "radians": math.radians,
    "degrees": math.degrees,
    "sqrt": math.sqrt,
    "inversesqrt": lambda x: 1.0 / math.sqrt(x),
    "rsqrt": lambda x: 1.0 / math.sqrt(x),
}

# Inputs for which GLSL.std.450 leaves the result undefined; such calls are
# not folded.
_UNARY_MATH_DOMAIN = {
    "log": lambda x: x > 0.0,
    "log2": lambda x: x > 0.0,
    "asin": lambda x: -1.0 <= x <= 1.0,
    "acos": lambda x: -1.0 <= x <= 1.0,
    "acosh": lambda x: x >= 1.0,
    "atanh": lambda x: -1.0 < x < 1.0,
    "sqrt": lambda x: x >= 0.0,
    "inversesqrt": lambda x: x > 0.0,
    "rsqrt": lambda x: x > 0.0,
}

_ROUNDING = {"floor": math.floor, "ceil": math.ceil, "trunc": math.trunc}


def _float_arguments(arguments: list[ScalarConstant]) -> list | None:
    kind = "float"
    for argument in arguments:
        kind = common_kind(kind, argument.kind)
    converted = [convert_scalar_constant(argument, kind) for argument in arguments]
    if any(value is None or not math.isfinite(value.value) for value in converted):
        return None
    return converted


def _fold_float_math(name: str, kind: str, values: list[float]):
    def rounded(value: float) -> float:
        return round_to_binary32(value) if kind == "float" else value

    if len(values) == 1 and name in _UNARY_MATH:
        (value,) = values
        domain = _UNARY_MATH_DOMAIN.get(name)
        if domain is not None and not domain(value):
            return None
        return rounded(_UNARY_MATH[name](value))
    if len(values) == 1 and name in _ROUNDING:
        # A zero result keeps the operand's sign: ceil(-0.5) is -0.0.
        return math.copysign(float(_ROUNDING[name](values[0])), values[0])
    if len(values) == 1 and name in {"round", "roundEven"}:
        (value,) = values
        if name == "round" and value - math.floor(value) == 0.5:
            # GLSL rounds a fraction of exactly 0.5 in a direction the
            # implementation chooses.
            return None
        # Python rounds ties to even, as roundEven does.
        return math.copysign(float(round(value)), value)
    if name == "mod" and len(values) == 2:
        x, y = values
        if y == 0.0:
            return None
        # GLSL defines mod(x, y) as x - y * floor(x / y).
        return rounded(x - rounded(y * math.floor(rounded(x / y))))
    if name == "fmod" and len(values) == 2:
        # The C remainder is exact, so it needs no rounding.
        return None if values[1] == 0.0 else math.fmod(values[0], values[1])
    if name == "fma" and len(values) == 3:
        return fused_multiply_add(kind, *values)
    if name == "fract" and len(values) == 1:
        return rounded(values[0] - math.floor(values[0]))
    if name == "saturate" and len(values) == 1:
        return min(max(values[0], 0.0), 1.0)
    if name in {"atan", "atan2"} and len(values) == 2:
        if values[0] == 0.0 and values[1] == 0.0:
            return None
        return rounded(math.atan2(values[0], values[1]))
    if name == "pow" and len(values) == 2:
        base, exponent = values
        if base < 0.0 or (base == 0.0 and exponent <= 0.0):
            return None
        return rounded(math.pow(base, exponent))
    if name == "step" and len(values) == 2:
        return 0.0 if values[1] < values[0] else 1.0
    if name == "mix" and len(values) == 3:
        x, y, a = values
        return rounded(rounded(x * rounded(1.0 - a)) + rounded(y * a))
    if name == "smoothstep" and len(values) == 3:
        edge0, edge1, x = values
        if edge0 >= edge1:
            return None
        t = rounded(rounded(x - edge0) / rounded(edge1 - edge0))
        t = min(max(t, 0.0), 1.0)
        return rounded(rounded(t * t) * rounded(3.0 - rounded(2.0 * t)))
    return None


_BINARY_FORMATS = {"float": (24, -126, 127), "double": (53, -1022, 1023)}


def round_fraction(value: Fraction, kind: str) -> float:
    """Round exact rational *value* once to the nearest *kind* value, ties to even."""
    precision, minimum_exponent, maximum_exponent = _BINARY_FORMATS[kind]
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    if Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    quantum = Fraction(2) ** (max(exponent, minimum_exponent) - precision + 1)
    steps, remainder = divmod(magnitude, quantum)
    if remainder * 2 > quantum or (remainder * 2 == quantum and steps % 2):
        steps += 1
    result = steps * quantum
    if result >= Fraction(2) ** (maximum_exponent + 1):
        return sign * math.inf
    return sign * float(result)


def fused_multiply_add(kind: str, a: float, b: float, c: float) -> float:
    """Return ``a * b + c`` rounded once, as a fused multiply-add computes it."""
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact != 0:
        return round_fraction(exact, kind)
    product_negative = math.copysign(1.0, a) * math.copysign(1.0, b) < 0
    if (a == 0.0 or b == 0.0) and c == 0.0:
        # IEEE 754: the sum of two zeros is -0 only when both are -0.
        if product_negative and math.copysign(1.0, c) < 0:
            return -0.0
    return 0.0


_SMALLEST_NORMAL = {"float": 2.0**-126, "double": 2.0**-1022}


def _fold_ldexp(
    mantissa: ScalarConstant, exponent: ScalarConstant
) -> ScalarConstant | None:
    if exponent.kind not in INTEGER_KINDS or mantissa.kind == "bool":
        return None
    kind = mantissa.kind if mantissa.kind in FLOAT_KINDS else "float"
    mantissa = convert_scalar_constant(mantissa, kind)
    if mantissa is None or not math.isfinite(mantissa.value):
        return None
    if mantissa.value == 0.0:
        return mantissa
    if not -1200 <= exponent.value <= 1200:
        return None
    result = round_fraction(
        Fraction(mantissa.value) * Fraction(2) ** exponent.value, kind
    )
    # GLSL leaves overflow undefined, and a subnormal result may be flushed.
    if not math.isfinite(result) or abs(result) < _SMALLEST_NORMAL[kind]:
        return None
    return ScalarConstant(kind, result)


# Reinterpreting casts: (operand kind or None for "any 32-bit kind", result kind).
_BIT_CASTS = {
    "floatBitsToInt": ("float", "int"),
    "floatBitsToUint": ("float", "uint"),
    "intBitsToFloat": ("int", "float"),
    "uintBitsToFloat": ("uint", "float"),
    "asint": (None, "int"),
    "asuint": (None, "uint"),
    "asfloat": (None, "float"),
}

# The only NaN a float constant literal can spell (0x1.8p+128).
_CANONICAL_NAN_BITS = 0x7FC00000


def _fold_bit_cast(name: str, argument: ScalarConstant) -> ScalarConstant | None:
    source, target = _BIT_CASTS[name]
    if source is None:
        source = argument.kind
    if source == "float":
        if argument.kind not in ("float", "int", "uint"):
            return None
    elif source not in ("int", "uint") or argument.kind not in ("int", "uint"):
        return None
    argument = convert_scalar_constant(argument, source)
    if source == "float":
        if math.isnan(argument.value):
            return None
        bits = struct.unpack("<I", struct.pack("<f", argument.value))[0]
    else:
        bits = argument.value & 0xFFFFFFFF
    if target == "float":
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        if math.isnan(value) and bits != _CANONICAL_NAN_BITS:
            return None
        return ScalarConstant("float", value)
    return ScalarConstant(target, _wrap_integer(target, bits))


def fold_math_builtin(
    name: str, arguments: list[ScalarConstant]
) -> ScalarConstant | None:
    """Fold a scalar call to a CrossGL math built-in, or return ``None``."""
    if name in _BIT_CASTS and len(arguments) == 1:
        return _fold_bit_cast(name, arguments[0])
    if name == "ldexp" and len(arguments) == 2:
        return _fold_ldexp(*arguments)
    integer = all(argument.kind in INTEGER_KINDS for argument in arguments)
    if integer and name in {"abs", "min", "max", "clamp", "sign"}:
        kind = arguments[0].kind
        for argument in arguments[1:]:
            kind = common_kind(kind, argument.kind)
        values = [convert_scalar_constant(a, kind).value for a in arguments]
        width, signed = INTEGER_KINDS[kind]
        if name == "abs" and len(values) == 1:
            if signed and values[0] == -(1 << (width - 1)):
                return None
            return ScalarConstant(kind, abs(values[0]))
        if name == "sign" and len(values) == 1:
            return ScalarConstant(kind, (values[0] > 0) - (values[0] < 0))
        if name in {"min", "max"} and len(values) == 2:
            return ScalarConstant(kind, (min if name == "min" else max)(values))
        if name == "clamp" and len(values) == 3 and values[1] <= values[2]:
            return ScalarConstant(kind, min(max(values[0], values[1]), values[2]))
        return None

    converted = _float_arguments(arguments)
    if converted is None:
        return None
    kind = converted[0].kind
    values = [argument.value for argument in converted]
    if name == "abs" and len(values) == 1:
        return ScalarConstant(kind, abs(values[0]))
    if name == "sign" and len(values) == 1:
        return ScalarConstant(kind, float((values[0] > 0) - (values[0] < 0)))
    if name in {"min", "max"} and len(values) == 2:
        return ScalarConstant(kind, (min if name == "min" else max)(values))
    if name == "clamp" and len(values) == 3:
        if values[1] > values[2]:
            return None
        return ScalarConstant(kind, min(max(values[0], values[1]), values[2]))
    try:
        result = _fold_float_math(name, kind, values)
    except (OverflowError, ValueError):
        return None
    if result is None:
        return None
    return ScalarConstant(kind, result)


__all__ = (
    "FLOAT_KINDS",
    "INTEGER_KINDS",
    "SCALAR_KINDS",
    "ScalarConstant",
    "ScalarConstantFolder",
    "canonical_float_value",
    "common_kind",
    "convert_scalar_constant",
    "fold_math_builtin",
    "fused_multiply_add",
    "round_fraction",
    "round_to_binary32",
    "shortest_binary32_text",
)
