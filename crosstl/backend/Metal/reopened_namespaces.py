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

Headers reopen the same namespace hundreds of times, and a unit is scanned again after each template
materialization step. :class:`ReopenedNamespaceIndex` therefore indexes the blocks of each namespace once per scan
and returns only the later blocks whose text names the member: a block that never spells the name cannot look it up,
so recording the binding there changes no lookup. A redeclaration of the member is not recorded again
(:meth:`ReopenedNamespaceIndex.unrecorded_later_blocks`), since its later blocks already hold the first binding.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from typing import Sequence, Tuple

__all__ = ["ReopenedNamespaceIndex", "reopened_namespace_scopes"]

Span = Tuple[int, int]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ReopenedNamespaceIndex:
    """The blocks of every namespace of one source text, in source order, with the identifiers each block spells."""

    def __init__(self, namespace_spans: Sequence[tuple[int, int, str]], code: str = ""):
        self.code = code
        self.qualified_by_block: dict[Span, str] = {}
        self.blocks_by_name: dict[str, list[Span]] = {}
        for start, end, name in sorted(namespace_spans):
            self.qualified_by_block.setdefault((start, end), name)
            self.blocks_by_name.setdefault(name, []).append((start, end))
        self.block_starts = {
            name: [start for start, _end in blocks]
            for name, blocks in self.blocks_by_name.items()
        }
        self.identifiers: dict[Span, set[str]] = {}
        self.recorded: set[tuple[str, str]] = set()

    @property
    def has_reopened_namespace(self) -> bool:
        return any(len(blocks) > 1 for blocks in self.blocks_by_name.values())

    def spelled_in(self, block: Span) -> set[str]:
        names = self.identifiers.get(block)
        if names is None:
            names = set(_IDENTIFIER.findall(self.code, block[0], block[1]))
            self.identifiers[block] = names
        return names

    def later_blocks(
        self, scope: Span, declaration_position: int, name: str | None = None
    ) -> list[Span]:
        """The later blocks of the namespace whose block body is ``scope``; with ``name``, only those spelling it.

        When ``scope`` is not a namespace block the result is empty. A block nested in another returned block (an
        unnamed namespace shares its parent's name) is covered by it and omitted.
        """
        qualified = self.qualified_by_block.get(tuple(scope))
        if qualified is None:
            return []
        blocks = self.blocks_by_name[qualified]
        first = bisect_right(self.block_starts[qualified], declaration_position)
        later: list[Span] = []
        covered_until = -1
        for start, end in blocks[first:]:
            if (start, end) == tuple(scope) or end <= covered_until:
                continue
            covered_until = end
            if name is not None and name not in self.spelled_in((start, end)):
                continue
            later.append((start, end))
        return later

    def unrecorded_later_blocks(
        self, scope: Span, declaration_position: int, name: str
    ) -> list[Span]:
        """:meth:`later_blocks` for the first declaration of ``name`` in its namespace; empty for any redeclaration.

        The blocks after a redeclaration also follow the first declaration, so they already hold its binding, and a
        namespace member is declared at most once with a meaning of its own ([basic.def.odr]). Recording each
        redeclaration again would cost blocks times declarations: every block of ``namespace c10`` that opens
        ``namespace metal`` declares the name ``metal`` there.
        """
        qualified = self.qualified_by_block.get(tuple(scope))
        if qualified is None or (qualified, name) in self.recorded:
            return []
        self.recorded.add((qualified, name))
        return self.later_blocks(scope, declaration_position, name)


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
    return ReopenedNamespaceIndex(namespace_spans).later_blocks(
        scope, declaration_position
    )
