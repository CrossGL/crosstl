"""Compile-time boolean locals that decide which branch of an ``if`` runs.

Template materialization often leaves guards such as
``const bool has_mask = !true; if (has_mask) { ... }``.  Every invocation takes
the same branch of such a guard, and the other branch never runs, so collective
control-flow validation may follow only the taken branch.

Guards such as ``!has_mask || bool(mask[i])`` fold too when the constant side
decides them: short-circuit evaluation never evaluates the other side.
"""

from __future__ import annotations

from typing import Any, Callable

from ..ast import (
    BinaryOpNode,
    BlockNode,
    DoWhileNode,
    ForNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)


def guard_bool_value(codegen: Any, expr: Any, constants: dict[str, bool]):
    """Return the boolean a guard always takes, or ``None``.

    Beyond ``codegen.hlsl_bool_constant_value``, ``a && b`` and ``a || b`` fold
    when both sides fold, or when the left side alone decides the result
    (``false && b``, ``true || b``) so that ``b`` is never evaluated.
    """
    value = codegen.hlsl_bool_constant_value(expr, constants)
    if value is not None:
        return value
    if isinstance(expr, UnaryOpNode) and codegen.map_operator(expr.operator) == "!":
        inner = guard_bool_value(codegen, expr.operand, constants)
        return None if inner is None else not inner
    if isinstance(expr, BinaryOpNode):
        operator = codegen.map_operator(expr.operator)
        if operator in {"&&", "||"}:
            left = guard_bool_value(codegen, expr.left, constants)
            if left is None:
                return None
            if left is (operator == "||"):
                return left
            return guard_bool_value(codegen, expr.right, constants)
    return None


def literal_guard_value(
    expr: Any, constants: dict[str, bool], map_operator: Callable[[Any], Any]
):
    """Fold a guard built only from booleans, or return ``None``.

    Operands are ``true``/``false`` literals and names ``constants`` binds to a
    boolean. ``!``, ``==`` and ``!=`` need every operand; ``&&`` and ``||``
    fold as soon as their left side decides them, because short-circuit
    evaluation never evaluates the right side then.
    """
    if isinstance(expr, LiteralNode):
        value = expr.value
        if isinstance(value, str) and value in {"true", "false"}:
            return value == "true"
        return value if isinstance(value, bool) else None
    if isinstance(expr, IdentifierNode):
        value = constants.get(expr.name)
        return value if isinstance(value, bool) else None
    if isinstance(expr, UnaryOpNode) and map_operator(expr.operator) == "!":
        inner = literal_guard_value(expr.operand, constants, map_operator)
        return None if inner is None else not inner
    if isinstance(expr, BinaryOpNode):
        operator = map_operator(expr.operator)
        if operator not in {"&&", "||", "==", "!="}:
            return None
        left = literal_guard_value(expr.left, constants, map_operator)
        if left is None:
            return None
        if operator in {"&&", "||"} and left is (operator == "||"):
            return left
        right = literal_guard_value(expr.right, constants, map_operator)
        if right is None or operator in {"&&", "||"}:
            return right
        return (left == right) is (operator == "==")
    return None


def constant_bool_locals(
    codegen: Any, function: Any, inherited: dict[str, bool]
) -> dict[str, bool]:
    """Return the boolean value of every name that is fixed throughout ``function``.

    ``inherited`` holds the constants visible on entry, without the function's
    parameters.  A local qualifies when it is declared exactly once, is
    ``const``, has an initializer that evaluates to a boolean with the values
    already known, and is never written afterwards.  Any other local hides an
    inherited constant of the same name.
    """
    body = getattr(function, "body", None)
    if body is None:
        return dict(inherited)
    declarations: dict[str, list[Any]] = {}
    order: list[Any] = []
    for node in codegen.walk_ast(body):
        if isinstance(node, VariableNode) and node.name:
            declarations.setdefault(node.name, []).append(node)
            order.append(node)
    written = codegen.hlsl_software_subgroup_written_names(body)
    constants = {
        name: value for name, value in inherited.items() if name not in declarations
    }
    for node in order:
        name = node.name
        if (
            len(declarations[name]) != 1
            or name in written
            or "const" not in (getattr(node, "qualifiers", None) or [])
        ):
            continue
        value = guard_bool_value(
            codegen, getattr(node, "initial_value", None), constants
        )
        if value is not None:
            constants[name] = value
    return constants


def _statements(body: Any) -> list[Any]:
    if isinstance(body, BlockNode) or hasattr(body, "statements"):
        return list(body.statements or [])
    if isinstance(body, list):
        return list(body)
    return [] if body is None else [body]


def lexical_constant_guards(
    function: Any,
    inherited: dict[str, Any],
    *,
    written: set[str],
    fold: Callable[[Any, dict[str, Any]], Any],
    tracked: Callable[[Any], bool] = lambda declaration: True,
) -> dict[int, bool]:
    """Return ``{id(if_node): taken}`` for guards that visible constants decide.

    Constants follow lexical scope: a local holds its value from its
    declaration to the end of its block, and any declaration of the same name
    hides it. A local qualifies when ``tracked`` accepts it, it is ``const``,
    it is not in ``written`` and ``fold(initializer, constants)`` gives its
    value. ``inherited`` holds the constants visible on entry, without the
    function's parameters. ``else if`` chains and guards inside ``switch``
    and range loops are left alone.
    """
    body = getattr(function, "body", None)
    if body is None:
        return {}
    guards: dict[int, bool] = {}

    def visit(statements: Any, constants: dict[str, Any]) -> None:
        constants = dict(constants)
        for statement in _statements(statements):
            if isinstance(statement, VariableNode):
                constants.pop(statement.name, None)
                if (
                    statement.name not in written
                    and "const" in (getattr(statement, "qualifiers", None) or [])
                    and tracked(statement)
                ):
                    value = fold(getattr(statement, "initial_value", None), constants)
                    if value is not None:
                        constants[statement.name] = value
            elif isinstance(statement, IfNode):
                if not getattr(statement, "else_if_conditions", None):
                    value = fold(statement.condition, constants)
                    if value is not None:
                        guards[id(statement)] = bool(value)
                for branch in (
                    getattr(statement, "then_branch", None),
                    *(getattr(statement, "else_if_bodies", None) or []),
                    getattr(statement, "else_branch", None),
                ):
                    visit(branch, constants)
            elif isinstance(statement, BlockNode):
                visit(statement, constants)
            elif isinstance(statement, (ForNode, WhileNode, DoWhileNode)):
                scoped = dict(constants)
                init = getattr(statement, "init", None)
                if isinstance(init, VariableNode):
                    scoped.pop(init.name, None)
                visit(statement.body, scoped)

    parameters = {
        getattr(parameter, "name", None)
        for parameter in getattr(function, "parameters", None) or []
    }
    visit(
        body,
        {name: value for name, value in inherited.items() if name not in parameters},
    )
    return guards


def literal_branch_guards(
    functions: Any,
    inherited: dict[str, bool],
    *,
    written: Callable[[Any], set[str]],
    map_operator: Callable[[Any], Any],
    tracked: Callable[[Any], bool],
) -> dict[int, bool]:
    """``lexical_constant_guards`` over ``functions`` with ``literal_guard_value``.

    ``written(function)`` names what the function may write; ``tracked``
    accepts the boolean local declarations that may hold a constant.
    """
    guards: dict[int, bool] = {}
    for function in functions:
        guards.update(
            lexical_constant_guards(
                function,
                inherited,
                written=written(function),
                fold=lambda expression, constants: literal_guard_value(
                    expression, constants, map_operator
                ),
                tracked=tracked,
            )
        )
    return guards


def guarded_dead_nodes(function: Any, guards: dict[int, bool], walk) -> set[int]:
    """Ids of every node inside a branch that a constant guard rules out."""
    dead: set[int] = set()
    for node in walk(getattr(function, "body", None)):
        if isinstance(node, IfNode) and id(node) in guards:
            skipped = (
                getattr(node, "else_branch", None)
                if guards[id(node)]
                else getattr(node, "then_branch", None)
            )
            dead.update(id(child) for child in walk(skipped))
    return dead


def constant_branch_guards(
    codegen: Any, function: Any, inherited: dict[str, bool]
) -> tuple[dict[int, bool], set[int]]:
    """DirectX guards that visible boolean constants decide.

    Returns ``(guards, short_circuited)``. ``short_circuited`` holds the guards
    that only short-circuit evaluation decides, such as ``!masked || load``:
    their remaining operands may not type-check in HLSL, so the generator
    emits only the selected branch. A guard whose every operand is constant
    keeps its condition, and only its dead branch is dropped.
    """
    written = codegen.hlsl_software_subgroup_written_names(
        getattr(function, "body", None)
    )
    guards = lexical_constant_guards(
        function,
        inherited,
        written=written,
        fold=lambda expression, constants: guard_bool_value(
            codegen, expression, constants
        ),
    )
    complete = lexical_constant_guards(
        function,
        inherited,
        written=written,
        fold=codegen.hlsl_bool_constant_value,
    )
    return guards, {
        node_id for node_id, value in guards.items() if complete.get(node_id) != value
    }
