"""Compile-time boolean locals that decide which branch of an ``if`` runs.

Template materialization often leaves guards such as
``const bool has_mask = !true; if (has_mask) { ... }``.  Every invocation takes
the same branch of such a guard, and the other branch never runs, so collective
control-flow validation may follow only the taken branch.
"""

from __future__ import annotations

from typing import Any

from ..ast import VariableNode


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
        value = codegen.hlsl_bool_constant_value(
            getattr(node, "initial_value", None), constants
        )
        if value is not None:
            constants[name] = value
    return constants
