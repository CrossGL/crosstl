"""Initializer-free counter loops entered with an unknown, uniform counter.

``loop_induction.init_less_counter_loops`` proves loops that start from the
counter's literal initializer. This module covers the remaining shape, as in
``int k = n; if (c) { for (; k > 0; k--) { ... } } else { for (; k > 0;
k--) { ... } }``: a loop reached with whatever value the counter has, which a
caller's flow analysis may prove uniform. Its trips then depend only on that
value, and termination has to hold for every value of the counter's type.
"""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import (
    AssignmentNode,
    BlockNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import _UnsignedLiteralInt

_INCREMENTS = {"++", "PRE_INCREMENT", "POST_INCREMENT"}
_DECREMENTS = {"--", "PRE_DECREMENT", "POST_DECREMENT"}
_UNARY_WRITES = _INCREMENTS | _DECREMENTS | {"&"}
_LOOPS = (ForNode, WhileNode, DoWhileNode, LoopNode, ForInNode)


def counter_update(update):
    """Return ``(name, sign, operand)`` for ``c++``, ``c--``, ``c += e``, ``c -= e``.

    ``operand`` is ``None`` for the unit increments; ``sign`` is ``1`` or
    ``-1``. Any other update returns ``None``.
    """
    if isinstance(update, UnaryOpNode) and isinstance(update.operand, IdentifierNode):
        if update.operator in _INCREMENTS:
            return update.operand.name, 1, None
        if update.operator in _DECREMENTS:
            return update.operand.name, -1, None
        return None
    if isinstance(update, AssignmentNode) and isinstance(update.target, IdentifierNode):
        if update.operator == "+=":
            return update.target.name, 1, update.value
        if update.operator == "-=":
            return update.target.name, -1, update.value
    return None


def _structured_loops(statements, found, visible, redeclared):
    """Collect initializer-free ``for`` loops reached without entering a loop.

    Each loop is recorded with the declarations visible where it starts:
    ``visible`` maps a name to its innermost declaration in an enclosing
    statement list that precedes the loop. A name declared twice in one
    statement list is ill-formed in every source language and goes into
    ``redeclared``.
    """
    if statements is None:
        return
    if isinstance(statements, BlockNode):
        statements = statements.statements
    if not isinstance(statements, (list, tuple)):
        statements = [statements]
    visible = dict(visible)
    declared_here = set()
    for statement in statements:
        if isinstance(statement, VariableNode) and statement.name:
            if statement.name in declared_here:
                redeclared.add(statement.name)
            declared_here.add(statement.name)
            visible[statement.name] = statement
        elif isinstance(statement, ForNode):
            if statement.init is None:
                found.append((statement, visible))
        elif isinstance(statement, IfNode):
            _structured_loops(statement.then_branch, found, visible, redeclared)
            for body in getattr(statement, "else_if_bodies", None) or []:
                _structured_loops(body, found, visible, redeclared)
            _structured_loops(statement.else_branch, found, visible, redeclared)
        elif isinstance(statement, BlockNode):
            _structured_loops(statement, found, visible, redeclared)


def any_start_counter_loops(function, walk_ast, writes):
    """Find initializer-free loops whose counter only those loops write.

    Each loop sits outside every other loop, reached only through blocks and
    ``if`` branches, and updates its counter by ``++``, ``--``, ``+=`` or
    ``-=``. The counter is the local, declared with an initializer, that the
    name denotes where the loop starts: one declaration may serve several
    loops, and each branch of an ``if`` may declare its own, as in ``if (c) {
    uint k = s; for (; k <= e; k += 16) { ... } } else { uint k = s; for
    (...) }``. Every write to the name anywhere in the function is the update
    of such a loop, so no loop encloses another write to it, and whether its
    value at each loop is uniform is a flow fact the caller tracks.
    ``writes(node)`` names what a call or a pointer or reference declaration
    may write or alias. Returns ``{id(loop): (name, declaration, loop)}``.
    """
    body = getattr(function, "body", None)
    if body is None:
        return {}
    parameters = {
        parameter.name
        for parameter in getattr(function, "parameters", None) or []
        if getattr(parameter, "name", None)
    }
    loops, redeclared = [], set()
    _structured_loops(body, loops, {}, redeclared)
    counters = {}
    for loop, visible in loops:
        update = counter_update(loop.update)
        if update is not None and update[0] not in parameters:
            counters.setdefault(update[0], []).append((loop, visible))
    if not counters:
        return {}
    updates = {id(loop.update) for group in counters.values() for loop, _ in group}
    disqualified = set(redeclared)
    for node in walk_ast(body):
        if id(node) in updates:
            continue
        if isinstance(node, AssignmentNode) and isinstance(node.target, IdentifierNode):
            disqualified.add(node.target.name)
        elif (
            isinstance(node, UnaryOpNode)
            and node.operator in _UNARY_WRITES
            and isinstance(node.operand, IdentifierNode)
        ):
            disqualified.add(node.operand.name)
        elif isinstance(node, FunctionCallNode) or (
            isinstance(node, VariableNode)
            and isinstance(node.var_type, (PointerType, ReferenceType))
        ):
            disqualified |= set(writes(node) or ())
    result = {}
    for name in sorted(counters):
        if name in disqualified:
            continue
        for loop, visible in counters[name]:
            declaration = visible.get(name)
            if declaration is not None and declaration.initial_value is not None:
                result[id(loop)] = (name, declaration, loop)
    return result


def any_start_counter_loop_terminates(type_name, offset, stops, step, operator):
    """Prove ``for (; c + offset OP stop; c += step)`` from any start value.

    An ascending loop (``<``/``<=``, positive step) stops at most one step
    past the largest counter the condition admits; a descending one
    (``>``/``>=``, negative step) at most one step below the smallest. That
    last counter must fit the counter's type for every extreme ``stop``.
    An unknown start could overflow ``c + offset`` on the first test, so the
    offset must be zero.
    """
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return False
    # HLSL's minimum-precision integers hold at least 16 bits.
    widened = {"min16int": "short", "min16uint": "ushort"}
    shape = source_integer_shape(
        widened[type_name] if type_name in widened else type_name
    )
    if shape is None or shape[2] != 1 or shape[1] > 32:
        return False
    values = (offset, step, *stops)
    if not stops or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        return False
    if int(offset) != 0 or int(step) == 0:
        return False
    ascending = operator in {"<", "<="}
    if not ascending and operator not in {">", ">="}:
        return False
    if ascending != (int(step) > 0):
        return False
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    if unsigned and min(int(value) for value in stops) < 0:
        return False
    if not unsigned and any(isinstance(value, _UnsignedLiteralInt) for value in values):
        # A negative signed counter would compare as a huge unsigned value.
        return False
    minimum = 0 if unsigned else -(1 << (shape[1] - 1))
    maximum = (1 << (shape[1] - (not unsigned))) - 1
    step = int(step)
    for stop in stops:
        stop = int(stop)
        if not minimum <= stop <= maximum:
            return False
        if ascending:
            limit = stop - (operator == "<")
            if limit >= minimum and limit + step > maximum:
                return False
        else:
            limit = stop + (operator == ">")
            if limit <= maximum and limit + step < minimum:
                return False
    return True


def wrapping_counter_loop_uniform(type_name, offset, step, operator):
    """Prove ``for (; c + offset OP stop; c += step)`` uniform without a range.

    A 32-bit unsigned counter wraps modulo 2**32 in Metal, HLSL and GLSL
    alike, so every ``c + offset`` test and every ``c += step`` is a total
    function of the counter, the bound and the literals. With a uniform start
    and bound, every invocation runs the same trips, whether or not the loop
    terminates, and leaves it, if at all, with the same counter. This holds
    for any bound, so it needs no termination proof. Signed counters have no
    such guarantee, because signed overflow is undefined in Metal, and
    narrower or minimum-precision counters may be promoted.
    """
    if not isinstance(type_name, str) or not type_name.isidentifier():
        return False
    shape = source_integer_shape(type_name)
    if shape != (ArithmeticScalarKind.UNSIGNED_INTEGER, 32, 1):
        return False
    values = (offset, step)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return False
    if not 0 <= int(offset) < (1 << 32) or not 0 < abs(int(step)) < (1 << 32):
        return False
    return operator in {"<", "<=", ">", ">="}
