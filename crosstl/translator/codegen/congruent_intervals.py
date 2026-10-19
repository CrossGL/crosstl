"""Bound wrapped unsigned ring arithmetic by congruence at a destination width.

C-family unsigned ``+``, ``-`` and ``*`` are defined modulo ``2**width``. When
an integer destination of ``bits`` bits receives such a result, the stored
value is the mathematical value reduced modulo ``2**bits``, provided every
operation that may wrap, and every operation consuming a wrapped value, is an
unsigned ring operation at least ``bits`` wide. Callers convert the returned
mathematical interval to the destination type; the conversion is exact.

A select yields one operand converted to its common type, so it may forward
a wrapped operand when that type is at least ``bits`` wide. Signed overflow,
division, remainder, shifts, comparisons and calls never consume a wrapped
value here, so their mathematical and stored values must agree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..ast import BinaryOpNode, TernaryOpNode, UnaryOpNode

RING_BINARY_OPERATORS = frozenset({"+", "-", "*"})
RING_UNARY_OPERATORS = frozenset({"+", "-"})


class _Unproven(Exception):
    pass


def congruent_ring_interval(
    expression: Any,
    destination_bits: int,
    *,
    bounds: Callable[[Any], tuple[int, int] | None],
    domain: Callable[[Any], Any],
    operator: Callable[[Any], str],
    walk: Callable[[Any], Iterable[Any]],
) -> tuple[int, int] | None:
    """Return the mathematical interval of ``expression`` or ``None``.

    ``bounds`` returns a node's mathematical interval, computed from the
    flow facts of its operands without modelling wrap. ``domain`` returns the
    node's source integer domain with ``minimum`` and ``maximum`` attributes.
    """
    if (
        type(destination_bits) is not int
        or not 1 <= destination_bits <= 64
        or not isinstance(expression, (BinaryOpNode, TernaryOpNode, UnaryOpNode))
    ):
        return None

    def within(node):
        node_bounds, node_domain = bounds(node), domain(node)
        if node_bounds is None or node_domain is None:
            raise _Unproven
        return (
            node_domain.minimum
            <= node_bounds[0]
            <= node_bounds[1]
            <= (node_domain.maximum)
        )

    def width(node, *, unsigned):
        node_domain = domain(node)
        if node_domain is None or unsigned and node_domain.minimum != 0:
            return None
        return (node_domain.maximum + 1).bit_length() - (node_domain.minimum == 0)

    def barrier(node):
        """Require every typed arithmetic value inside ``node`` to be exact.

        Like the non-wrapping proof, untyped or unbounded nodes such as
        comparisons do not establish a wrapped value.
        """
        for inner in walk(node):
            if not isinstance(inner, (BinaryOpNode, UnaryOpNode)):
                continue
            inner_bounds, inner_domain = bounds(inner), domain(inner)
            if (
                inner_bounds is not None
                and inner_domain is not None
                and not (
                    inner_domain.minimum
                    <= inner_bounds[0]
                    <= inner_bounds[1]
                    <= inner_domain.maximum
                )
            ):
                raise _Unproven
        return False

    def analyze(node):
        """Return whether the node's value may differ from its mathematical value."""
        if isinstance(node, TernaryOpNode):
            barrier(node.condition)
            wrapped = [analyze(node.true_expr), analyze(node.false_expr)]
            if not any(wrapped) and within(node):
                return False
            common = width(node, unsigned=False)
            if common is None or common < destination_bits:
                raise _Unproven
            return True
        if isinstance(node, BinaryOpNode):
            operands, ring = (node.left, node.right), RING_BINARY_OPERATORS
        elif isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operands, ring = (node.operand,), RING_UNARY_OPERATORS
        else:
            return barrier(node)
        wrapped = [analyze(operand) for operand in operands]
        if not any(wrapped) and within(node):
            return False
        ring_width = width(node, unsigned=True)
        if (
            operator(node) not in ring
            or ring_width is None
            or ring_width < destination_bits
        ):
            raise _Unproven
        return True

    try:
        analyze(expression)
    except _Unproven:
        return None
    return bounds(expression)
