"""Visibility of local receiver declarations, answered by a sweep over source positions.

Struct-method lowering asks, for every ``receiver.method(...)`` call site of a unit, which local declaration of
``receiver`` is visible there. The declarations of one name are kept in source order, and the answer is the last one
that ends at or before the position and whose scope contains it. Scanning every declaration of the name for every call
is quadratic: a unit with many template instantiations declares the same local names (``loader``, ``op``) in each of
them, and the scan dominated the translation of such units.

:class:`ReceiverDeclarationSweep` answers the same question for positions asked in nondecreasing order in amortized
time proportional to the declarations still open there. Declarations enter once they end at or before the position, in
the order of the list, and leave from the top of a stack once their scope has closed; positions only grow within a
sweep, so a closed scope stays closed. The stack therefore holds, in list order, every declaration the linear scan
would consider except ones that are no longer visible, and its last visible entry is exactly the scan's answer. Asking
for an earlier position, or a list that changed length, restarts the sweep.
"""

from typing import Any, Dict, List, Optional


class ReceiverDeclarationSweep:
    """The latest visible declaration in one name's declaration list (see the module docstring)."""

    __slots__ = ("candidates", "_length", "_cursor", "_position", "_stack")

    def __init__(self, candidates: List[Any]) -> None:
        self.candidates = candidates
        self._reset()

    def _reset(self) -> None:
        self._length = len(self.candidates)
        self._cursor = 0
        self._position: Optional[int] = None
        self._stack: List[Any] = []

    def visible_at(self, position: int) -> Optional[Any]:
        """The last declaration ending at or before ``position`` whose scope contains it, as the linear scan finds it.

        The linear scan stops at the first declaration of the list that ends after ``position``; the sweep stops at
        the same one.
        """
        candidates = self.candidates
        if len(candidates) != self._length or (
            self._position is not None and position < self._position
        ):
            self._reset()
        self._position = position
        stack = self._stack
        cursor = self._cursor
        count = len(candidates)
        while cursor < count and candidates[cursor].declaration_end <= position:
            while stack and stack[-1].scope_end <= position:
                stack.pop()
            stack.append(candidates[cursor])
            cursor += 1
        self._cursor = cursor
        while stack and stack[-1].scope_end <= position:
            stack.pop()
        for index in range(len(stack) - 1, -1, -1):
            candidate = stack[index]
            if candidate.scope_start <= position < candidate.scope_end:
                return candidate
        return None


def visible_receiver_declaration(
    sweeps: Dict[str, ReceiverDeclarationSweep],
    candidates: List[Any],
    receiver: str,
    position: int,
) -> Optional[Any]:
    """Answer through the sweep of ``receiver`` in ``sweeps``, starting a new one when its list was replaced."""
    sweep = sweeps.get(receiver)
    if sweep is None or sweep.candidates is not candidates:
        sweep = sweeps[receiver] = ReceiverDeclarationSweep(candidates)
    return sweep.visible_at(position)
