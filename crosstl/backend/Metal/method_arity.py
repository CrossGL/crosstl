"""Argument-count checks for Metal member functions.

``_MetalStructMethod.parameter_names`` lists only *named* parameters, so it is
not an arity: an unnamed tag parameter such as ``metal::bool_constant<B>`` is
part of the signature but has no name. Overload selection that compares
argument counts has to count every declared parameter and honour defaults.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

SplitCommas = Callable[[str], List[str]]
SplitAssignment = Callable[[str], Tuple[str, object]]


def parameter_arity(
    parameters: str,
    split_commas: SplitCommas,
    split_assignment: SplitAssignment,
) -> tuple[int, int]:
    """Return ``(required, total)`` for a parameter-list text.

    ``void`` and an empty list take no arguments. A parameter with a default
    value, and every parameter after it, is optional.
    """
    items = [item.strip() for item in split_commas(parameters) if item.strip()]
    if items == ["void"]:
        items = []
    required = len(items)
    for index, item in enumerate(items):
        _declaration, default = split_assignment(item)
        if default is not None and str(default).strip():
            required = index
            break
    return required, len(items)


def accepts_argument_count(
    parameters: str,
    count: int,
    split_commas: SplitCommas,
    split_assignment: SplitAssignment,
) -> bool:
    """Whether a call with ``count`` arguments can bind to ``parameters``."""
    if "..." in parameters:
        required, _total = parameter_arity(
            parameters.replace("...", ""), split_commas, split_assignment
        )
        return count >= max(0, required - 1)
    required, total = parameter_arity(parameters, split_commas, split_assignment)
    return required <= count <= total
