"""Whether control can reach the end of a WGSL function body, by WGSL's behavior analysis.

WGSL computes for each statement the set of ways control leaves it: ``Next``, ``Return``, ``Break`` and
``Continue`` (WGSL "Behavior Analysis"). It rejects a function with a return type whose body has the behavior
``Next``. The analysis does not evaluate conditions: a ``while`` loop, a ``for`` loop with a condition and the
``break if`` of a lowered ``do``-``while`` all have ``Break``, so the loop has ``Next`` although every iteration
of ``do { return x; } while (c);`` or ``while (true) { if (d) { return x; } }`` returns.

C-family sources end value-returning functions this way. Control that flows off the end of such a function is
undefined behavior in C++ (C++17 [stmt.return]/2), which MSL, CUDA and HIP follow, so no defined execution of the
source reaches the end. :func:`end_is_reachable` tells the WGSL emitter when to close the body with a ``return``
of the return type's zero value, which only such executions would reach.

The analysis runs on CrossGL statements as the WGSL emitter lowers them. A compound statement it does not know
is taken to have every behavior, so the answer errs towards ``Next``; an extra ``return`` after a statement that
always returns is unreachable code, which WGSL allows.
"""

from __future__ import annotations

from ..ast import (
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ForNode,
    IfNode,
    LoopNode,
    ReturnNode,
    SwitchNode,
    WhileNode,
)

__all__ = ["BREAK", "CONTINUE", "NEXT", "RETURN", "behaviors", "end_is_reachable"]

NEXT = "next"
RETURN = "return"
BREAK = "break"
CONTINUE = "continue"
EVERY = frozenset({NEXT, RETURN, BREAK, CONTINUE})
_COMPOUND_FIELDS = ("body", "statements", "cases", "arms", "then_branch")


def end_is_reachable(statements) -> bool:
    """True when the statements of a function body, in order, can have the behavior ``Next``."""
    return NEXT in sequence_behaviors(statements)


def sequence_behaviors(statements) -> frozenset:
    """``s1 s2``: ``(B1 - {Next}) | B2`` when ``Next`` is in ``B1``, else ``B1``."""
    result = {NEXT}
    for statement in statements or ():
        if NEXT not in result:
            break
        result = (result - {NEXT}) | behaviors(statement)
    return frozenset(result)


def loop_behaviors(body, exits) -> frozenset:
    """``loop { s }``; ``exits`` adds the ``Break`` of a loop condition or a ``break if``."""
    result = set(body)
    if exits:
        result.add(BREAK)
    if BREAK in result:
        return frozenset((result - {BREAK, CONTINUE}) | {NEXT})
    return frozenset(result - {NEXT, CONTINUE})


def behaviors(statement) -> frozenset:
    """The behaviors of one CrossGL statement, lowered as the WGSL emitter lowers it."""
    if statement is None:
        return frozenset({NEXT})
    if isinstance(statement, (list, tuple)):
        return sequence_behaviors(statement)
    if isinstance(statement, BlockNode):
        return sequence_behaviors(statement.statements)
    if isinstance(statement, ReturnNode):
        return frozenset({RETURN})
    if isinstance(statement, BreakNode):
        return frozenset({BREAK})
    if isinstance(statement, ContinueNode):
        return frozenset({CONTINUE})
    if isinstance(statement, IfNode):
        return _if_behaviors(statement)
    if isinstance(statement, (WhileNode, ForNode)):
        return loop_behaviors(
            behaviors(statement.body), exits=statement.condition is not None
        )
    if isinstance(statement, DoWhileNode):
        return loop_behaviors(behaviors(statement.body), exits=True)
    if isinstance(statement, LoopNode):
        return loop_behaviors(behaviors(statement.body), exits=False)
    if isinstance(statement, SwitchNode):
        return _switch_behaviors(statement)
    continuing = _continuing_loop_behaviors(statement)
    if continuing is not None:
        return continuing
    if any(getattr(statement, name, None) is not None for name in _COMPOUND_FIELDS):
        return EVERY
    return frozenset({NEXT})


def _if_behaviors(statement) -> frozenset:
    result = set(behaviors(statement.then_branch))
    for body in getattr(statement, "else_if_bodies", None) or ():
        result |= behaviors(body)
    if statement.else_branch is None:
        result.add(NEXT)
    else:
        result |= behaviors(statement.else_branch)
    return frozenset(result)


def _switch_behaviors(statement) -> frozenset:
    """A ``switch`` turns ``Break`` into ``Next``; a missing default or a clause that falls off gives ``Next``."""
    clauses = [getattr(case, "statements", case) for case in statement.cases or ()]
    default = getattr(statement, "default_case", None)
    if default is not None:
        clauses.append(getattr(default, "statements", default))
    result = set()
    for clause in clauses:
        result |= behaviors(clause)
    if default is None and not any(
        getattr(case, "value", True) is None for case in statement.cases or ()
    ):
        result.add(NEXT)
    if BREAK in result:
        result = (result - {BREAK}) | {NEXT}
    return frozenset(result)


def _continuing_loop_behaviors(statement):
    from .wgsl_loop_continuing import WGSLContinuingLoopNode

    if not isinstance(statement, WGSLContinuingLoopNode):
        return None
    body = sequence_behaviors([*statement.body, *statement.continuing])
    return loop_behaviors(body, exits=statement.break_if is not None)
