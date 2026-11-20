"""Which operands of a SPIR-V operation are numeric literals that must not take the enclosing expression's type.

The SPIR-V generator types a non-negative ``int`` literal as ``uint`` when the expression it initializes or converts
is unsigned. That context belongs to the outermost expression only: a literal operand of an operator or a call takes
the other operand's type, or the parameter's, and otherwise keeps its own (GLSL 4.60 section 5.9; an operation on
``int`` operands is signed, and its result is converted afterwards). A literal under a unary sign or complement, as in
``v > -990``, is a literal operand as well.
"""

from __future__ import annotations

from ..ast import LiteralNode, UnaryOpNode

__all__ = ["is_literal_operand"]

_LITERAL_PREFIXES = frozenset({"-", "+", "~"})


def is_literal_operand(node) -> bool:
    """Whether ``node`` is a literal, possibly under unary ``-``, ``+`` or ``~``."""
    while isinstance(node, UnaryOpNode) and node.op in _LITERAL_PREFIXES:
        if getattr(node, "is_postfix", False):
            return False
        node = node.operand
    return isinstance(node, LiteralNode)
