"""Keep imported 8- and 16-bit Slang integer arithmetic at its declared width.

Slang applies no integer promotion: an operator whose operands are both 8- or 16-bit integer scalars computes in
the usual arithmetic conversion of the two narrow types (the wider one, or the unsigned one of equal width, as C++
[expr.arith.conv] orders them), so ``int8_t * int8_t`` is an ``int8_t`` and slangc 2026.20 writes ``OpIMul %char``,
and ``int8_t < uint8_t`` compares unsigned 8-bit values (``OpULessThan`` on ``uchar``). A shift keeps the type of
its left operand, unary ``-`` and ``~`` keep their operand's type, and ``a /= b`` converts ``b`` to ``a``'s type
first. An operand of any other type (``int``, an integer literal, ``uint``, ``float``) converts the narrow operand
instead, which CrossGL's promotion computes the same way.

CrossGL promotes a narrow scalar operand to ``int`` before an operator applies (C++ [conv.prom],
``PROMOTING_SOURCES`` in :mod:`crosstl.translator.semantics.narrow`), so ``v * v`` on an ``int8_t`` holding 13 is
169 in CrossGL and -87 in Slang. The importer therefore writes:

- ``+``, ``-``, ``*`` and ``<<`` with a narrow result, unary ``-``, and ``~`` of an unsigned narrow operand, as
  ``T(e)``. A conversion to a narrow integer type keeps the value modulo 2^N (C++ [conv.integral]), which is the
  Slang result whatever the promoted operation computed. Division, remainder, the other bitwise operators, right
  shifts and comparisons cannot leave the range of the common type, and signed division of the minimum value by -1
  is undefined in Slang's SPIR-V (``OpSDiv``), so those keep their form;
- an operand of a binary or conditional operator whose signedness differs from the common narrow type, as
  ``T(operand)``, so CrossGL promotes the value Slang computes with (``-1`` read as ``uint8_t`` is 255);
- the right operand of ``/=`` and ``%=`` on a narrow scalar as ``T(b)``, Slang's conversion of it, since dividing
  by the promoted ``b`` differs from dividing by ``T(b)``.

Vector operations are not promoted by CrossGL (MSL section 2.2, OpenCL C 6.3) and are left alone.
"""

from __future__ import annotations

import re

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)

__all__ = ["SlangNarrowArithmeticMixin", "slang_narrow_scalar"]

# Slang's 8- and 16-bit integer scalars: (signed, width).
_NARROW_SCALARS = {
    "int8_t": (True, 8),
    "uint8_t": (False, 8),
    "int16_t": (True, 16),
    "uint16_t": (False, 16),
}
_NARROW_VECTOR = re.compile(r"(u?int(?:8|16)_t)[1-4]")
_BUFFER_ELEMENT = re.compile(
    r"\s*\w*(?:Buffer|Array)\w*\s*<\s*(\w+)\s*(?:,[^<>]*)?>\s*"
)
_ARITHMETIC = frozenset({"+", "-", "*", "/", "%", "&", "|", "^"})
_RELATIONAL = frozenset({"<", "<=", ">", ">=", "==", "!="})
_SHIFTS = frozenset({"<<", ">>"})
# Operators whose narrow result can leave the range of its type when CrossGL computes it promoted.
_WRAPPING_BINARY = frozenset({"+", "-", "*", "<<"})
_CONVERTED_COMPOUND = frozenset({"/=", "%="})
# Generic built-ins that return their arguments' type when every argument has it.
_SAME_TYPE_BUILTINS = frozenset({"abs", "min", "max", "clamp"})


def slang_narrow_scalar(type_name):
    """The Slang 8- or 16-bit integer scalar ``type_name`` names, else ``None``."""
    name = str(type_name or "").strip()
    return name if name in _NARROW_SCALARS else None


def _common_narrow(left, right):
    """The type Slang computes a binary operator of two narrow scalars in (C++ [expr.arith.conv] without promotion)."""
    if left == right:
        return left
    left_signed, left_width = _NARROW_SCALARS[left]
    right_signed, right_width = _NARROW_SCALARS[right]
    if left_signed == right_signed:
        return left if left_width >= right_width else right
    signed, unsigned = (left, right) if left_signed else (right, left)
    if _NARROW_SCALARS[unsigned][1] >= _NARROW_SCALARS[signed][1]:
        return unsigned
    return signed


class SlangNarrowArithmeticMixin:
    """Narrow integer arithmetic of Slang source written so that CrossGL's promotion computes Slang's result."""

    def collect_narrow_function_returns(self, functions):
        """Record the narrow scalar return type of every user function that has one return type."""
        returns = {}
        for function in functions:
            name = getattr(function, "name", None)
            if not name:
                continue
            narrow = slang_narrow_scalar(getattr(function, "return_type", None))
            returns.setdefault(name, set()).add(narrow)
        self.narrow_function_returns = {
            name: next(iter(types))
            for name, types in sorted(returns.items())
            if len(types) == 1 and None not in types
        }

    def narrow_element_type(self, type_name):
        """The narrow scalar an element access of a value of ``type_name`` reads, else ``None``."""
        text = str(type_name or "").strip()
        vector = _NARROW_VECTOR.fullmatch(text)
        if vector:
            return vector.group(1)
        buffer = _BUFFER_ELEMENT.fullmatch(text)
        if buffer:
            return slang_narrow_scalar(buffer.group(1))
        return slang_narrow_scalar(text)

    def narrow_expression_type(self, expr):
        """The Slang 8- or 16-bit integer scalar type of ``expr``, else ``None`` (also when it is unknown)."""
        if isinstance(expr, VariableNode):
            return slang_narrow_scalar(self.expression_type(expr))
        if isinstance(expr, ArrayAccessNode):
            return self.narrow_element_type(self.expression_type(expr.array))
        if isinstance(expr, MemberAccessNode):
            member_type = self.expression_type(expr)
            if member_type is not None:
                return slang_narrow_scalar(member_type)
            vector = _NARROW_VECTOR.fullmatch(
                str(self.expression_type(expr.object) or "")
            )
            if vector and re.fullmatch(r"[xyzwrgba]", str(expr.member)):
                return vector.group(1)
            return None
        if isinstance(expr, CastNode):
            return slang_narrow_scalar(expr.target_type)
        if isinstance(expr, VectorConstructorNode):
            return slang_narrow_scalar(expr.type_name)
        if isinstance(expr, FunctionCallNode):
            return self._narrow_call_type(expr)
        if isinstance(expr, AssignmentNode):
            return self.narrow_expression_type(expr.left)
        if isinstance(expr, UnaryOpNode):
            if expr.op in {"-", "+", "~"}:
                return self.narrow_expression_type(expr.operand)
            return None
        if isinstance(expr, TernaryOpNode):
            return self._narrow_pair_type(expr.true_expr, expr.false_expr)
        if isinstance(expr, BinaryOpNode):
            if expr.op in _SHIFTS:
                return self.narrow_expression_type(expr.left)
            if expr.op in _ARITHMETIC:
                return self._narrow_pair_type(expr.left, expr.right)
        return None

    def _narrow_call_type(self, expr):
        name = str(expr.name)
        constructed = slang_narrow_scalar(name)
        if constructed is not None:
            return constructed
        if name in self.user_function_names:
            return getattr(self, "narrow_function_returns", {}).get(name)
        if name in _SAME_TYPE_BUILTINS and expr.args:
            types = {self.narrow_expression_type(arg) for arg in expr.args}
            if len(types) == 1:
                return types.pop()
        return None

    def _narrow_pair_type(self, left, right):
        left_type = self.narrow_expression_type(left)
        right_type = self.narrow_expression_type(right)
        if left_type is None or right_type is None:
            return None
        return _common_narrow(left_type, right_type)

    def _narrow_operand(self, operand, rendered, common):
        """``rendered`` converted to ``common`` when its signedness differs from that of ``common``."""
        operand_type = self.narrow_expression_type(operand)
        if operand_type is None or operand_type == common:
            return rendered
        if _NARROW_SCALARS[operand_type][0] == _NARROW_SCALARS[common][0]:
            return rendered
        return f"{self.map_type(common)}({rendered})"

    def narrow_binary_operands(self, expr, left, right):
        """``(left, right)`` with each converted to the common narrow type where Slang converts its signedness."""
        if expr.op not in _ARITHMETIC and expr.op not in _RELATIONAL:
            return left, right
        common = self._narrow_pair_type(expr.left, expr.right)
        if common is None:
            return left, right
        return (
            self._narrow_operand(expr.left, left, common),
            self._narrow_operand(expr.right, right, common),
        )

    def narrow_conditional_operands(self, expr, true_value, false_value):
        """The two values of ``c ? a : b`` converted to their common narrow type where the signedness differs."""
        common = self._narrow_pair_type(expr.true_expr, expr.false_expr)
        if common is None:
            return true_value, false_value
        return (
            self._narrow_operand(expr.true_expr, true_value, common),
            self._narrow_operand(expr.false_expr, false_value, common),
        )

    def narrow_result_wrapped(self, expr):
        """Whether ``expr`` is rendered as ``T(...)`` by :meth:`narrow_binary_result` or :meth:`narrow_unary_result`."""
        if isinstance(expr, BinaryOpNode):
            return (
                expr.op in _WRAPPING_BINARY
                and self.narrow_expression_type(expr) is not None
            )
        if isinstance(expr, UnaryOpNode) and expr.op in {"-", "~"}:
            narrow = self.narrow_expression_type(expr.operand)
            return narrow is not None and (
                expr.op == "-" or not _NARROW_SCALARS[narrow][0]
            )
        return False

    def narrow_binary_result(self, expr, rendered):
        """``rendered`` as ``T(rendered)`` when Slang computes ``expr`` modulo 2^N of its narrow type ``T``."""
        if expr.op not in _WRAPPING_BINARY:
            return rendered
        narrow = self.narrow_expression_type(expr)
        if narrow is None:
            return rendered
        return f"{self.map_type(narrow)}({rendered})"

    def narrow_unary_result(self, expr, rendered):
        """``-v`` of a narrow ``v`` and ``~v`` of an unsigned narrow ``v`` as ``T(...)``, Slang's narrow result."""
        if expr.op not in {"-", "~"}:
            return rendered
        if not self.narrow_result_wrapped(expr):
            return rendered
        return f"{self.map_type(self.narrow_expression_type(expr.operand))}({rendered})"

    def narrow_compound_right(self, node, rhs):
        """The right operand of ``a /= b`` or ``a %= b`` on a narrow scalar ``a`` as ``T(b)``, Slang's conversion."""
        if node.operator not in _CONVERTED_COMPOUND:
            return rhs
        narrow = self.narrow_expression_type(node.left)
        if narrow is None or self.narrow_expression_type(node.right) == narrow:
            return rhs
        if _literal_fits(node.right, narrow):
            return rhs
        return f"{self.map_type(narrow)}({rhs})"


def _literal_fits(expr, narrow):
    """Whether ``expr`` is a decimal or hexadecimal integer literal whose value ``narrow`` represents."""
    match = (
        re.fullmatch(r"(0[xX][0-9a-fA-F]+|[0-9]+)[uU]?", str(expr).strip())
        if isinstance(expr, str)
        else None
    )
    if match is None:
        return False
    value = (
        int(match.group(1), 0) if not re.fullmatch(r"0[0-7]+", match.group(1)) else None
    )
    if value is None:
        return False
    signed, width = _NARROW_SCALARS[narrow]
    return value < (1 << (width - 1) if signed else 1 << width)
