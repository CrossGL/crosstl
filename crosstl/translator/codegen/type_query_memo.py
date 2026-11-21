"""Scoped memo for a code generator's expression type queries.

A generator's type queries (``HLSLCodeGen.expression_result_type`` and ``hlsl_source_expression_type``) call each
other and themselves on the operands of an expression, and several of their branches ask for the type of the same
operand again: a bit cast asks for its argument's source type and then for its result type, an integer operation
asks once per arithmetic contract. On nested calls such as the bit casts of wrapped integer arithmetic, the work
then grows by a constant factor with every level: one query on a SPIR-V ray generation shader with a doubly nested
``OpSMod`` made 449,214 type queries and took 7 s of CPU.

:func:`scoped_type_query` memoizes the decorated queries for the duration of the outermost query only. The
generator state the queries read (variable types, the type table, profile flags) cannot change while a query runs,
so every repeated question inside one query has the answer it had the first time. Nothing is kept between outermost
queries, so a later query sees the generator's state at that time, exactly as without the memo. The decorated
queries of one generator share one scope, keyed by query name and node. An entry keeps its node alive, so a node's
``id`` is never reused for another node while the memo exists. A query that raises leaves no entry.
"""

from __future__ import annotations

import functools

__all__ = ("scoped_type_query",)

_MEMO = "_scoped_type_query_memo"


def scoped_type_query(method):
    """Memoize a one-argument type query of the generator within its outermost call."""
    key = method.__name__

    @functools.wraps(method)
    def query(self, expression):
        memo = self.__dict__.get(_MEMO)
        if memo is None:
            self.__dict__[_MEMO] = {}
            try:
                return query(self, expression)
            finally:
                self.__dict__[_MEMO] = None
        slot = (key, id(expression))
        entry = memo.get(slot)
        if entry is not None and entry[0] is expression:
            return entry[1]
        result = method(self, expression)
        memo[slot] = (expression, result)
        return result

    return query
