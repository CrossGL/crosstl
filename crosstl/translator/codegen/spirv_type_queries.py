"""Type queries on storage expressions that do not evaluate the expression's side effects.

Overload matching asks for an argument's storage-buffer type before the call is lowered, once per candidate search,
and the type comes from the pointer the expression names. Resolving that pointer lowers the expression, so an
argument such as ``buf.v[j++].z`` passed to an ``inout`` parameter had its ``j++`` evaluated once per query as well as
once for the call. GLSL 4.60 section 6.1.1 and HLSL evaluate each argument exactly once.

A subscript's value never changes the type of the element it selects, so the query resolves a copy of the expression
whose subscripts with side effects read element ``0`` instead, and drops the access chains and loads that resolution
emitted, which nothing else uses. Any other part with a side effect, such as a call that returns the base, makes the
type unknown to the query rather than evaluating it.
"""

from __future__ import annotations

import copy
import re

from ..ast import ArrayAccessNode, LiteralNode, MemberAccessNode, PrimitiveType

# Instructions with no effect beyond their result id, which a type query may emit and then drop.
_UNUSED_QUERY_INSTRUCTION = re.compile(
    r"^\s*%\w+ = Op(?:AccessChain|InBoundsAccessChain|PtrAccessChain|Load)\b"
)


def type_query_pointer(gen, expr):
    """The pointer ``expr`` names, for a type query; None when that needs a side effect other than a subscript's."""
    query = side_effect_free_type_query_expression(gen, expr)
    if query is None:
        return None
    if query is expr:
        return gen.variable_pointer_from_expression(expr)
    start = len(gen.code_lines)
    pointer = gen.variable_pointer_from_expression(query)
    if all(_UNUSED_QUERY_INSTRUCTION.match(line) for line in gen.code_lines[start:]):
        del gen.code_lines[start:]
    return pointer


def side_effect_free_type_query_expression(gen, expr):
    """``expr``, or a copy whose subscripts with side effects are ``0``; None when another part has a side effect."""
    if gen.side_effect_free_call_argument(expr):
        return expr
    if isinstance(expr, MemberAccessNode):
        base = side_effect_free_type_query_expression(gen, expr.object)
        if base is None:
            return None
        query = copy.copy(expr)
        query.object = query.object_expr = base
        return query
    if isinstance(expr, ArrayAccessNode):
        base = side_effect_free_type_query_expression(gen, expr.array)
        if base is None:
            return None
        index = expr.index
        if not gen.side_effect_free_call_argument(index):
            index = LiteralNode(0, PrimitiveType("int"))
        query = copy.copy(expr)
        query.array = query.array_expr = base
        query.index = query.index_expr = index
        return query
    return None
