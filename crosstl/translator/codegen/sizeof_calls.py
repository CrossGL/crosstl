"""``sizeof`` in the generators: a type operand, after generic substitution, becomes its byte count.

:mod:`..semantics.sizeof_values` folds every ``sizeof`` and ``metal_sizeof`` whose operand type is concrete before
generation. One whose type is an enclosing generic function's type parameter (``sizeof(T)``, or ``sizeof(x)`` with
``T x``, which that pass rewrites to ``sizeof(T)``) is folded here, in the specialization, where the generator knows
``T``. A call this cannot size is refused with :class:`..semantics.sizeof_values.SizeofUnsupportedError`; the
generators never print it.
"""

from __future__ import annotations

import re

from ..ast import ArrayAccessNode, IdentifierNode, LiteralNode
from ..semantics.sizeof_values import (
    SIZEOF_LAYOUTS,
    SizeofUnsupportedError,
    sizeof_type_bytes,
)
from .enum_utils import substitute_generic_type_name

__all__ = ["sizeof_call_bytes"]

_ARRAY_SPELLING = re.compile(r"(.+?)\s*\[\s*(\d+)\s*\]")


def _operand_spelling(operand):
    if isinstance(operand, IdentifierNode):
        return str(operand.name)
    if isinstance(operand, ArrayAccessNode) and isinstance(
        operand.array_expr, IdentifierNode
    ):
        index = operand.index_expr
        if isinstance(index, LiteralNode) and str(index.value).rstrip("uU").isdigit():
            return f"{operand.array_expr.name}[{str(index.value).rstrip('uU')}]"
    return None


def _substituted(spelling, substitutions):
    match = _ARRAY_SPELLING.fullmatch(spelling)
    if match:
        element = _substituted(match.group(1), substitutions)
        return f"{element}[{match.group(2)}]"
    return substitute_generic_type_name(spelling, substitutions)


def sizeof_call_bytes(generator, func_name, args, target):
    """The byte count of a ``sizeof(type)`` or ``metal_sizeof(type)`` call, ``None`` for any other call.

    Raises :class:`SizeofUnsupportedError` for an operand it cannot size.
    """
    layout = SIZEOF_LAYOUTS.get(func_name) if isinstance(func_name, str) else None
    if layout is None:
        return None
    if len(args or []) != 1:
        raise SizeofUnsupportedError(
            "", "it takes exactly one operand", target=target, function=func_name
        )
    spelling = _operand_spelling(args[0])
    if spelling is None:
        raise SizeofUnsupportedError(
            type(args[0]).__name__,
            "the operand is not a type name; sizeof of a value is folded from its declaration before generation",
            target=target,
            function=func_name,
        )
    substitutions = (
        getattr(generator, "current_generic_function_substitutions", None) or {}
    )
    concrete = _substituted(spelling, substitutions) if substitutions else spelling
    return sizeof_type_bytes(concrete, target=target, layout=layout)
