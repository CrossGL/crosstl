"""Guards that boolean constants decide in OpenGL software-subgroup proofs.

Every invocation takes the same branch of such a guard, so the uniformity
analyses follow only that branch, and ``glsl_static_if_body`` emits only it.
Module constants that another module declaration shadows, and specialization
constants, stay unknown.
"""

from __future__ import annotations

from typing import Any

from .constant_guards import guarded_dead_nodes, literal_branch_guards


def glsl_constant_guards(
    codegen: Any, functions: Any
) -> tuple[dict[int, bool], set[int]]:
    """Return ``({id(if_node): taken}, dead node ids)`` for ``functions``."""
    shadowed: set[str] = set()
    for function in functions:
        shadowed |= codegen.glsl_software_subgroup_module_constant_shadow_names(
            getattr(function, "body", None)
        )
    constant_types = getattr(codegen, "glsl_constant_types", {})
    specialized = getattr(codegen, "glsl_specialization_constant_names", ())
    inherited = {
        name: bool(value)
        for name, value in codegen.literal_int_constants.items()
        if constant_types.get(name) == "bool"
        and type(value) is int
        and value in (0, 1)
        and name not in shadowed
        and name not in specialized
    }
    guards = literal_branch_guards(
        functions,
        inherited,
        written=lambda function: codegen.glsl_software_subgroup_assigned_names(
            getattr(function, "body", None)
        ),
        map_operator=codegen.map_operator,
        tracked=lambda declaration: codegen.type_name_string(declaration.var_type)
        == "bool",
    )
    dead: set[int] = set()
    for function in functions:
        dead |= guarded_dead_nodes(function, guards, codegen.walk_ast)
    return guards, dead
