"""Scopes of namespace members across reopened blocks of their namespace.

A namespace may be opened any number of times, and a member declared in one block is visible by unqualified lookup
in every later block of the same namespace ([basic.scope.namespace], [namespace.def]): ::

    namespace c10 { namespace metal { constant constexpr unsigned ILP_PER_THREAD = 4; } }
    namespace c10 { namespace metal { static_assert(ILP_PER_THREAD == 4, ""); } }

The Metal importer records a local integral constant with the brace block it is declared in as its scope, so the
assertion in the second block saw no binding and was refused as unresolved. :func:`reopened_namespace_scopes` gives
the later blocks of the namespace that a declaration's block belongs to, so the importer can record the binding in
each of them.

Inside a later block, the namespace member hides a declaration of the same name in an enclosing scope even when that
one was declared between the two blocks ([basic.lookup.unqual]). The importer resolves a name to the visible binding
declared last, so the binding recorded for a later block is treated as declared where the block opens.
"""

from __future__ import annotations

from typing import Sequence, Tuple

__all__ = ["reopened_namespace_scopes"]

Span = Tuple[int, int]


def reopened_namespace_scopes(
    namespace_spans: Sequence[tuple[int, int, str]],
    scope: Span,
    declaration_position: int,
) -> list[Span]:
    """The ``(start, end)`` bodies of the later blocks of the namespace whose block body is ``scope``.

    ``namespace_spans`` holds ``(body_start, body_end, qualified_name)`` for every namespace block, and the member
    is declared at ``declaration_position`` in ``scope``. When ``scope`` is not a namespace block the result is
    empty. A block nested in another returned block (an unnamed namespace shares its parent's name) is covered by
    it and omitted.
    """
    qualified = next(
        (name for start, end, name in namespace_spans if (start, end) == scope),
        None,
    )
    if qualified is None:
        return []
    blocks: list[Span] = []
    for start, end, name in sorted(namespace_spans):
        if name != qualified or start <= declaration_position or (start, end) == scope:
            continue
        if any(outer <= start and end <= outer_end for outer, outer_end in blocks):
            continue
        blocks.append((start, end))
    return blocks
