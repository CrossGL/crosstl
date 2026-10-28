"""Namespace-scope ``const`` and ``constexpr`` constants of CUDA and HIP sources.

A CUDA or HIP kernel can read a namespace-scope ``constexpr`` variable, and a ``const`` one of arithmetic type
with a constant initializer, without ``__device__`` or ``__constant__``. Importing such a constant as a mutable
CrossGL global (``var N: i32 = 3;``) loses that: the CUDA and HIP outputs then write a plain host variable, which
device code cannot read, and the other targets treat it as writable state. The importers write a CrossGL
``const`` instead, when its initializer is a constant expression. One that calls a function or reads a variable,
such as ``const float S = sqrtf(2.0f);``, is initialized when the program starts, so it stays a variable.
"""

from __future__ import annotations

import re

from .common_ast import (
    BinaryOpNode,
    CastNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)

# Declarations with a memory-space or linkage qualifier keep their own lowering.
_NON_CONSTANT_QUALIFIERS = frozenset(
    {"__constant__", "__device__", "__shared__", "__managed__", "extern"}
)
# A number with any suffix, a character or string literal with any prefix, or a boolean.
_LITERAL = re.compile(r"""\.?[0-9]|(u8|u|U|L)?R?["']|(true|false)$""")
# The calls that build a value from their arguments: functional casts and the vector constructors.
_CONSTRUCTOR = re.compile(
    r"(make_)?(u?char|u?short|u?int|u?long|u?longlong|float|double)[1-4]|dim3"
    r"|bool|char|short|int|long|unsigned|float|double|u?int(8|16|32|64)_t|size_t"
)
_CONSTANT_UNARY_OPERATORS = frozenset({"-", "+", "!", "~"})


def is_c_family_namespace_constant(node, constants) -> bool:
    """Return whether a namespace-scope variable declaration is a compile-time constant.

    ``constants`` holds the names of the constants declared before ``node``. Pointers and references are left
    alone: ``const char*`` is a pointer to const data, not a constant.
    """
    if node.value is None:
        return False
    qualifiers = set(node.qualifiers or [])
    if qualifiers & _NON_CONSTANT_QUALIFIERS:
        return False
    vtype = str(node.vtype or "")
    if "*" in vtype or "&" in vtype or "[" in vtype:
        return False
    if not {"const", "constexpr"} & (qualifiers | set(vtype.split())):
        return False
    return is_constant_expression(node.value, constants)


def is_constant_expression(value, constants) -> bool:
    """Return whether ``value`` is built only from literals, ``constants``, operators, casts and constructors."""
    if isinstance(value, str):
        return value in constants or _LITERAL.match(value) is not None
    if isinstance(value, BinaryOpNode):
        return is_constant_expression(value.left, constants) and is_constant_expression(
            value.right, constants
        )
    if isinstance(value, UnaryOpNode):
        return value.op in _CONSTANT_UNARY_OPERATORS and is_constant_expression(
            value.operand, constants
        )
    if isinstance(value, TernaryOpNode):
        parts = (value.condition, value.true_expr, value.false_expr)
        return all(is_constant_expression(part, constants) for part in parts)
    if isinstance(value, CastNode):
        return is_constant_expression(value.expression, constants)
    if isinstance(value, MemberAccessNode):
        return not value.is_pointer and is_constant_expression(value.object, constants)
    if isinstance(value, InitializerListNode):
        return all(
            is_constant_expression(element, constants) for element in value.elements
        )
    if isinstance(value, DesignatedInitializerNode):
        return is_constant_expression(value.value, constants)
    if isinstance(value, FunctionCallNode) and isinstance(value.name, str):
        if value.name == "sizeof":
            return True
        return _CONSTRUCTOR.fullmatch(value.name) is not None and all(
            is_constant_expression(arg, constants) for arg in value.args
        )
    return False
