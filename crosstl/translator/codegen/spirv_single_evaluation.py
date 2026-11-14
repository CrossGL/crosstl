"""Evaluate a subscript index once when the SPIR-V emitter visits an access twice.

C, GLSL, HLSL and MSL evaluate each operand of an expression once (C++ [intro.execution], GLSL 4.60
section 5.9), so the index of ``t[k--]`` decrements ``k`` once and ``t[atomicAdd(c, 1)]`` claims one
slot. Two places in the emitter look at an access before they emit it for real:

* a local declaration first asks whether its initializer aliases a resource or a storage-buffer
  element (``S s = buffer[i]`` binds ``s`` to the element instead of copying it), which builds an
  access chain for the initializer; when it does not alias, the declaration evaluates the
  initializer again as a value;
* an assignment used as a value (``a = (t[k++] = x)``) stores through its target and then reads the
  target back.

Each visit lowers the index again, so an index that writes (``++``, ``--``, an assignment, a call)
runs its side effect twice. A :class:`SubscriptIndexMemo` records the index values computed by the
first visit and hands each one back, once, to the next visit of the same index node. Indices that
can neither write nor call are lowered again as before, so their output is unchanged.

A memo is scoped to one declaration or one assignment expression and is pushed on a stack, so a
nested declaration (for example inside an inlined function body) starts with an empty memo.
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

__all__ = [
    "SubscriptIndexMemo",
    "evaluate_subscript_index",
    "recording_subscript_indices",
    "subscript_index_scope",
]


class SubscriptIndexMemo:
    """Index values from a first visit, waiting for the second visit of the same node."""

    def __init__(self) -> None:
        self.recording = False
        self.values: Dict[int, List[object]] = {}

    def record(self, index_expr, value) -> None:
        self.values.setdefault(id(index_expr), []).append(value)

    def peek(self, index_expr) -> Optional[object]:
        pending = self.values.get(id(index_expr))
        return pending[-1] if pending else None

    def take(self, index_expr) -> Optional[object]:
        pending = self.values.get(id(index_expr))
        if not pending:
            return None
        return pending.pop(0)


def _memo_stack(generator) -> List[Optional[SubscriptIndexMemo]]:
    stack = getattr(generator, "subscript_index_memos", None)
    if stack is None:
        stack = []
        generator.subscript_index_memos = stack
    return stack


def _current_memo(generator) -> Optional[SubscriptIndexMemo]:
    stack = getattr(generator, "subscript_index_memos", None)
    return stack[-1] if stack else None


@contextmanager
def subscript_index_scope(generator) -> Iterator[SubscriptIndexMemo]:
    """Give one declaration or assignment expression its own memo."""
    memo = SubscriptIndexMemo()
    stack = _memo_stack(generator)
    stack.append(memo)
    try:
        yield memo
    finally:
        stack.pop()


@contextmanager
def recording_subscript_indices(generator) -> Iterator[None]:
    """Record the indices lowered inside the block for the next visit in the same scope."""
    memo = _current_memo(generator)
    if memo is None:
        yield
        return
    previous = memo.recording
    memo.recording = True
    try:
        yield
    finally:
        memo.recording = previous


def evaluate_subscript_index(generator, index_expr):
    """Lower ``index_expr``, reusing the value a recorded first visit already computed."""
    memo = _current_memo(generator)
    if memo is None or generator.side_effect_free_call_argument(index_expr):
        return generator.process_expression(index_expr)
    if not memo.recording:
        recorded = memo.take(index_expr)
        if recorded is not None:
            return recorded
        return generator.process_expression(index_expr)
    recorded = memo.peek(index_expr)
    if recorded is not None:
        # A second probe of the same declaration shares the first probe's value.
        return recorded
    value = generator.process_expression(index_expr)
    if value is not None:
        memo.record(index_expr, value)
    return value
