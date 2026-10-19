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


def _structured_loops(statements, found):
    """Collect initializer-free ``for`` loops reached without entering a loop."""
    if statements is None:
        return
    if isinstance(statements, BlockNode):
        statements = statements.statements
    if not isinstance(statements, (list, tuple)):
        statements = [statements]
    for statement in statements:
        if isinstance(statement, ForNode):
            if statement.init is None:
                found.append(statement)
        elif isinstance(statement, IfNode):
            _structured_loops(statement.then_branch, found)
            for body in getattr(statement, "else_if_bodies", None) or []:
                _structured_loops(body, found)
            _structured_loops(statement.else_branch, found)
        elif isinstance(statement, BlockNode):
            _structured_loops(statement, found)


def any_start_counter_loops(function, walk_ast, writes):
    """Find initializer-free loops whose counter only those loops write.

    Each loop sits outside every other loop, reached only through blocks and
    ``if`` branches, and updates its counter by ``++``, ``--``, ``+=`` or
    ``-=``. The counter is a local declared once with an initializer, and
    every write to it anywhere in the function is the update of such a loop.
    No loop encloses another write to it, so whether its value at each loop
    is uniform is a flow fact the caller tracks. ``writes(node)`` names what a
    call or a pointer or reference declaration may write or alias. Returns
    ``{id(loop): (name, declaration, loop)}``.
    """
    body = getattr(function, "body", None)
    if body is None:
        return {}
    parameters = {
        parameter.name
        for parameter in getattr(function, "parameters", None) or []
        if getattr(parameter, "name", None)
    }
    loops = []
    _structured_loops(body, loops)
    counters = {}
    for loop in loops:
        update = counter_update(loop.update)
        if update is not None and update[0] not in parameters:
            counters.setdefault(update[0], []).append(loop)
    if not counters:
        return {}
    nodes = list(walk_ast(body))
    declarations = {}
    for node in nodes:
        if isinstance(node, VariableNode) and node.name in counters:
            declarations.setdefault(node.name, []).append(node)
    updates = {id(loop.update) for group in counters.values() for loop in group}
    disqualified = set()
    for node in nodes:
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
        declared = declarations.get(name, [])
        if (
            name in disqualified
            or len(declared) != 1
            or declared[0].initial_value is None
        ):
            continue
        for loop in counters[name]:
            result[id(loop)] = (name, declared[0], loop)
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
    shape = source_integer_shape(
        {"min16int": "short", "min16uint": "ushort"}.get(type_name, type_name)
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
