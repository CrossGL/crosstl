"""Array extents of OpenCL C declarations.

An OpenCL C array extent is an integer constant expression (C11 6.7.6.2), often a macro product such as
``float s[N * 2]``. The HIP importer this importer derives from turned an extent that is not a name or a literal
into an identifier (``hip_array_extent_4_mul_2``) that nothing declares, and DXC rejected the HLSL with "use of
undeclared identifier". CrossGL parses an expression extent, so the OpenCL importer instead:

- folds an extent made only of integer literals to its value with C's integer arithmetic (division and remainder
  truncate toward zero), which every target accepts;
- keeps an extent that also names constants as the expression, for CrossGL to resolve;
- refuses any other extent with :class:`OpenCLArrayExtentUnsupportedError`.
"""

from __future__ import annotations

import ast
import re

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "OpenCLArrayExtentUnsupportedError",
    "fold_integer_constant_expression",
    "format_opencl_array_extent",
]

_INTEGER_LITERAL = re.compile(
    r"(?<![\w.])(?P<body>0[xX][0-9A-Fa-f]+|0[0-7]*|[1-9][0-9]*)(?P<suffix>[uUlL]*)(?![\w.])"
)
_LITERAL_EXPRESSION = re.compile(r"[\s0-9A-Fa-fxXuUlL()+\-*/%<>&|^~]+")
_NAMED_EXPRESSION = re.compile(r"[\s\w()+\-*/%<>&|^~]+")
_SIMPLE_EXTENT = re.compile(r"[A-Za-z_]\w*|(?:0[xX][0-9A-Fa-f]+|\d+)[uU]?")


class OpenCLArrayExtentUnsupportedError(TranslationDiagnostic):
    """An OpenCL C array extent the importer cannot write as a CrossGL extent."""

    code = CODE_PREFIX + "opencl-array-extent-unsupported"


def _decimal_literal(match):
    body = match.group("body")
    if body.lower().startswith("0x"):
        return str(int(body, 16))
    if len(body) > 1 and body.startswith("0"):
        return str(int(body, 8))
    return body


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.Invert):
            return ~operand
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            if right == 0:
                raise ValueError("division by zero")
            quotient = abs(left) // abs(right)
            if (left < 0) != (right < 0):
                quotient = -quotient
            return quotient if not isinstance(op, ast.Mod) else left - quotient * right
        if isinstance(op, ast.LShift) and right >= 0:
            return left << right
        if isinstance(op, ast.RShift) and right >= 0:
            return left >> right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
    raise ValueError("not an integer constant expression")


def fold_integer_constant_expression(text):
    """The value of a C integer constant expression made of literals and operators, or None."""
    if not _LITERAL_EXPRESSION.fullmatch(text):
        return None
    python_text = _INTEGER_LITERAL.sub(_decimal_literal, text)
    # C's / is truncating integer division; spell it so Python parses it as a binary operator.
    python_text = python_text.replace("/", "//")
    try:
        return _evaluate(ast.parse(python_text, mode="eval"))
    except (SyntaxError, ValueError, RecursionError):
        return None


def format_opencl_array_extent(size, fallback):
    """The CrossGL text of the OpenCL C array extent ``size``; ``fallback`` formats a name or a literal."""
    text = str(size).strip()
    if not text or _SIMPLE_EXTENT.fullmatch(text):
        return fallback(size)
    value = fold_integer_constant_expression(text)
    if value is not None:
        if value <= 0:
            raise OpenCLArrayExtentUnsupportedError(
                f"array extent '{text}' is {value}; an array extent must be positive"
            )
        return str(value)
    if _NAMED_EXPRESSION.fullmatch(text) and text.count("(") == text.count(")"):
        return text
    raise OpenCLArrayExtentUnsupportedError(
        f"array extent '{text}' is not an integer constant expression of names, literals and operators"
    )
