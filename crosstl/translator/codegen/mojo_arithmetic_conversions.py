"""C and GLSL implicit scalar conversions, written out for Mojo.

C (C17 section 6.3.1.8, the usual arithmetic conversions) and GLSL (GLSL 4.60 section 4.1.10) convert the
operands of an arithmetic or relational operator to one common type, and convert a value to the type of the
variable, parameter or return value it initializes. Mojo converts neither: ``Int32 * Float32`` and
``var x: Float32 = i`` do not compile. The generator therefore writes each conversion as a constructor call of the
target type, which truncates a float toward zero and wraps an integer, as C does.

A literal operand is left alone, since Mojo gives an integer or float literal the type of the other operand; a float
literal next to an integer operand converts that operand to float instead.
"""

from ..ast import LiteralNode, UnaryOpNode
from .mojo_tables import MOJO_SCALAR_DTYPES

__all__ = ["MojoArithmeticConversionMixin", "mojo_scalar_kind"]

_SHIFT_OPERATORS = frozenset({"<<", ">>"})
_FLOAT_BITS = {"DType.float16": 16, "DType.float32": 32, "DType.float64": 64}
_INTEGER_BITS = {
    "DType.int8": (8, True),
    "DType.int16": (16, True),
    "DType.int32": (32, True),
    "DType.int64": (64, True),
    "DType.uint8": (8, False),
    "DType.uint16": (16, False),
    "DType.uint32": (32, False),
    "DType.uint64": (64, False),
}
_DTYPE_TYPE_NAMES = {
    "DType.float16": "half",
    "DType.float32": "float",
    "DType.float64": "double",
    "DType.int8": "int8_t",
    "DType.int16": "int16_t",
    "DType.int32": "int",
    "DType.int64": "int64_t",
    "DType.uint8": "uint8_t",
    "DType.uint16": "uint16_t",
    "DType.uint32": "uint",
    "DType.uint64": "uint64_t",
}


def mojo_scalar_kind(codegen, type_name):
    """The Mojo DType of a numeric scalar CrossGL type, or ``None`` for any other type."""
    if not isinstance(type_name, str):
        return None
    dtype = MOJO_SCALAR_DTYPES.get(codegen.type_name(type_name))
    if dtype in _FLOAT_BITS or dtype in _INTEGER_BITS:
        return dtype
    return None


def _is_literal(expr):
    if isinstance(expr, UnaryOpNode) and getattr(expr, "op", None) == "-":
        expr = expr.operand
    return isinstance(expr, LiteralNode) and not isinstance(expr.value, (bool, str))


def common_scalar_dtype(left, right):
    """The DType the usual arithmetic conversions give two numeric scalar DTypes."""
    if left == right:
        return left
    if left in _FLOAT_BITS or right in _FLOAT_BITS:
        float_dtypes = [dtype for dtype in (left, right) if dtype in _FLOAT_BITS]
        return max(float_dtypes, key=_FLOAT_BITS.get)
    left_bits, left_signed = _INTEGER_BITS[left]
    right_bits, right_signed = _INTEGER_BITS[right]
    # The integer promotions widen anything narrower than int to int first.
    left_bits, left_signed = (32, True) if left_bits < 32 else (left_bits, left_signed)
    right_bits, right_signed = (
        (32, True) if right_bits < 32 else (right_bits, right_signed)
    )
    if left_signed == right_signed:
        bits, signed = max(left_bits, right_bits), left_signed
    else:
        unsigned_bits = left_bits if not left_signed else right_bits
        signed_bits = left_bits if left_signed else right_bits
        if unsigned_bits >= signed_bits:
            bits, signed = unsigned_bits, False
        else:
            bits, signed = signed_bits, True
    return next(
        dtype
        for dtype, (dtype_bits, dtype_signed) in _INTEGER_BITS.items()
        if dtype_bits == bits and dtype_signed == signed
    )


class MojoArithmeticConversionMixin:
    """Implicit scalar conversion members of ``MojoCodeGen``."""

    def mojo_binary_operand_dtypes(self, expr, op):
        """The DTypes of a scalar binary operation's operands and their common DType, or ``None``."""
        left = mojo_scalar_kind(self, self.expression_result_type(expr.left))
        right = mojo_scalar_kind(self, self.expression_result_type(expr.right))
        if left is None or right is None:
            return None
        if op in _SHIFT_OPERATORS:
            # A shift has the promoted type of its left operand (C17 6.5.7p3).
            return left, right, common_scalar_dtype(left, left)
        return left, right, common_scalar_dtype(left, right)

    def mojo_converted_binary_operands(self, expr, op, left_code, right_code):
        """The operand code of a scalar binary operation, each converted to the operands' common type."""
        dtypes = self.mojo_binary_operand_dtypes(expr, op)
        if dtypes is None:
            return left_code, right_code
        left, right, common = dtypes
        return (
            self.mojo_converted_operand(expr.left, left, common, left_code),
            self.mojo_converted_operand(expr.right, right, common, right_code),
        )

    def mojo_converted_operand(self, operand, dtype, common, code):
        if dtype == common or _is_literal(operand):
            return code
        return f"{self.map_type(_DTYPE_TYPE_NAMES[common])}({code})"

    def mojo_binary_common_type(self, op, left_type, right_type):
        """The CrossGL type of a scalar arithmetic result under the usual arithmetic conversions, or ``None``."""
        left = mojo_scalar_kind(self, left_type)
        right = mojo_scalar_kind(self, right_type)
        if left is None or right is None:
            return None
        if op in _SHIFT_OPERATORS:
            right = left
        return _DTYPE_TYPE_NAMES[common_scalar_dtype(left, right)]

    def mojo_converted_to_target(self, expr, code, target_type):
        """``code`` converted to the numeric scalar ``target_type`` it initializes, when its own type differs."""
        target = mojo_scalar_kind(self, target_type)
        if target is None or _is_literal(expr):
            return code
        source = mojo_scalar_kind(self, self.expression_result_type(expr))
        if source is None or source == target:
            return code
        return f"{self.map_type(_DTYPE_TYPE_NAMES[target])}({code})"
