"""Visibility of local integral constant bindings, answered by a sweep over source positions.

The Metal importer records every local ``constexpr`` integral constant, parameter and type alias of a translation unit
as a binding with a declaration position and the lexical scope it is visible in. Template materialization then asks,
for thousands of positions, which binding each name resolves to there. Scanning every binding for every question is
quadratic in the size of the unit, and on large kernels with many instantiations it dominates translation time.

:class:`IntegralBindingTable` keeps the bindings in the order they were declared and answers the same question with a
sweep. Positions asked in nondecreasing order cost amortized time proportional to the bindings visible there:
bindings enter once their declaration is reached and leave, through a heap keyed on scope end, once their scope has
closed. Asking for an earlier position restarts the sweep. The answer is exactly the one the linear scan gives: for
every name, the visible binding with the latest declaration, the first of equal ones on a tie. When the table cannot
guarantee that, because bindings were added out of declaration order or the mapping was edited directly,
:meth:`IntegralBindingTable.visible_at` returns ``None`` and the caller scans.
"""

import heapq
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _visible(binding: Any, position: int) -> bool:
    return binding.scope_start <= position < binding.scope_end


def _latest_visible(stack: List[Any], position: int) -> Optional[Any]:
    """The visible entry of ``stack`` (in declaration order) with the latest declaration, the first of equal ones."""
    index = len(stack) - 1
    while index >= 0 and not _visible(stack[index], position):
        index -= 1
    if index < 0:
        return None
    best = stack[index]
    declaration = best.declaration_position
    index -= 1
    while index >= 0 and stack[index].declaration_position == declaration:
        if _visible(stack[index], position):
            best = stack[index]
        index -= 1
    return best


class IntegralBindingTable(dict):
    """``name -> bindings in declaration order`` (tuples), with a sweep index for :meth:`visible_at`.

    Add bindings with :meth:`add`. Any other edit of the mapping disables the index, and queries then return ``None``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._order: List[Tuple[int, Any]] = []
        self._name_index: Dict[str, int] = {}
        self._names: List[str] = []
        self._indexable = True
        self._reset_sweep()

    def _reset_sweep(self) -> None:
        self._cursor = 0
        self._position: Optional[int] = None
        self._stacks: Dict[int, List[Any]] = {}
        self._expiry: List[Tuple[int, int, int]] = []
        self._live: Set[int] = set()

    def add(self, name: str, binding: Any) -> None:
        """Append ``binding`` to ``name``'s bindings; bindings must arrive in nondecreasing declaration order."""
        if (
            self._order
            and binding.declaration_position < self._order[-1][1].declaration_position
        ):
            self._indexable = False
        index = self._name_index.get(name)
        if index is None:
            index = self._name_index[name] = len(self._names)
            self._names.append(name)
        self._order.append((index, binding))
        dict.__setitem__(self, name, dict.get(self, name, ()) + (binding,))

    def visible_at(
        self, position: int, names: Optional[Iterable[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """For each name (all, or ``names`` in sorted order), the binding visible at ``position``.

        Returns ``None`` when the index cannot answer exactly; the caller then scans the bindings itself.
        """
        if not self._indexable:
            return None
        if self._position is not None and position < self._position:
            self._reset_sweep()
        self._position = position
        self._advance(position)
        if names is None:
            selected = sorted(self._live)
        else:
            selected = []
            for name in sorted(names):
                index = self._name_index.get(name)
                if index is not None and index in self._live:
                    selected.append(index)
        visible: Dict[str, Any] = {}
        for index in selected:
            best = _latest_visible(self._stacks[index], position)
            if best is not None:
                visible[self._names[index]] = best
        return visible

    def _advance(self, position: int) -> None:
        order = self._order
        while (
            self._cursor < len(order)
            and order[self._cursor][1].declaration_position <= position
        ):
            index, binding = order[self._cursor]
            self._stacks.setdefault(index, []).append(binding)
            heapq.heappush(self._expiry, (binding.scope_end, self._cursor, index))
            self._live.add(index)
            self._cursor += 1
        while self._expiry and self._expiry[0][0] <= position:
            _end, _sequence, index = heapq.heappop(self._expiry)
            stack = self._stacks.get(index)
            # Positions only grow within a sweep, so a closed scope stays closed.
            while stack and stack[-1].scope_end <= position:
                stack.pop()
            if not stack:
                self._stacks.pop(index, None)
                self._live.discard(index)

    def _disable_index(self) -> None:
        self._indexable = False

    def __setitem__(self, key, value) -> None:
        self._disable_index()
        super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        self._disable_index()
        super().__delitem__(key)

    def setdefault(self, key, default=None):
        self._disable_index()
        return super().setdefault(key, default)

    def update(self, *args, **kwargs) -> None:
        self._disable_index()
        super().update(*args, **kwargs)

    def pop(self, *args):
        self._disable_index()
        return super().pop(*args)

    def popitem(self):
        self._disable_index()
        return super().popitem()

    def clear(self) -> None:
        self._disable_index()
        super().clear()

    def __ior__(self, other):
        self._disable_index()
        return super().__ior__(other)


def indexed_visible_bindings(
    bindings: Any, position: int, names: Optional[Iterable[str]] = None
) -> Optional[Dict[str, Any]]:
    """``bindings.visible_at(position, names)`` for an :class:`IntegralBindingTable`, otherwise ``None``."""
    if isinstance(bindings, IntegralBindingTable):
        return bindings.visible_at(position, names)
    return None
