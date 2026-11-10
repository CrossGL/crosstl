"""Fold constructor calls that constant-initialize a value into field-wise aggregates.

A ``static constexpr`` data member is constant-initialized ([dcl.constexpr], [basic.start.static]). When its
initializer calls a constructor, that constructor is ``constexpr`` and runs during constant evaluation. The importer
lowers constructors to runtime factory functions, and a call to one of those is no longer a constant expression. On
the Metal round trip the factory is not even declared yet where the struct's member uses it.

A constructor whose body is empty is fully described by its mem-initializers and the default member initializers
([class.base.init]): each member takes the value of its initializer with the parameters bound to the arguments.
That value is the CrossGL aggregate ``Owner(member0, member1, ...)``, which every generator emits as a constant
expression. The fold applies only when that reading is exact. Otherwise :func:`fold_constant_constructor_call`
returns ``None`` and the caller keeps the factory call:

* the constructor body has statements, or the owner is a union or has base classes;
* an initializer names something other than a data member (a base or a delegating constructor);
* a member is an array, or has neither an initializer nor a default member initializer;
* an initializer or default refers to a member of the owner or to ``this`` instead of to a parameter or a value
  outside the class, because the factory's member-relative scope does not exist in a constant initializer.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

from .MetalAst import VectorConstructorNode

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_SIMPLE_VALUE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fFhHuU]?")


@contextmanager
def constant_initialization(codegen) -> Iterator[None]:
    """Mark the enclosed rendering as the initializer of a constant-initialized value."""
    codegen.constant_initialization_depth = (
        getattr(codegen, "constant_initialization_depth", 0) + 1
    )
    try:
        yield
    finally:
        codegen.constant_initialization_depth -= 1


def in_constant_initialization(codegen) -> bool:
    return getattr(codegen, "constant_initialization_depth", 0) > 0


def _referenced_identifiers(values) -> set[str]:
    """Every identifier token reachable from ``values``, including those held in plain strings."""
    found: set[str] = set()
    stack = list(values)
    seen = set()
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            found.update(_IDENTIFIER.findall(value))
            continue
        if value is None or isinstance(value, (bool, int, float)):
            continue
        if isinstance(value, dict):
            stack.extend(value.values())
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        if hasattr(value, "__dict__"):
            stack.extend(
                child
                for field, child in vars(value).items()
                if field != "source_location"
            )
    return found


def _member_value_nodes(codegen, struct_node, constructor) -> list[object] | None:
    """The expression that initializes each runtime member, in declaration order, or ``None``."""
    if getattr(struct_node, "aggregate_kind", "struct") == "union":
        return None
    if getattr(struct_node, "base_types", None):
        return None
    if constructor.body is None or list(constructor.body):
        return None
    members = codegen.constructor_runtime_members(struct_node)
    member_names = {member.name for member in members}
    initializers = {}
    for initializer in constructor.initializers:
        target = str(initializer.target).strip()
        if target not in member_names or target in initializers:
            return None
        initializers[target] = initializer
    values = []
    for member in members:
        if getattr(member, "array_sizes", None):
            return None
        initializer = initializers.get(member.name)
        if initializer is None:
            value = getattr(member, "default_value", None)
            if value is None:
                return None
            values.append(value)
            continue
        member_type = codegen.resolve_type_alias(
            codegen.metal_declaration_expression_type(member)
        )
        if (
            codegen.pointer_element_type(member_type) is not None
            and len(initializer.arguments) == 1
        ):
            values.append(initializer.arguments[0])
        else:
            values.append(
                VectorConstructorNode(member.vtype, list(initializer.arguments))
            )
    return values


def _class_scope_names(struct_node) -> set[str]:
    names = {"this"}
    for member in getattr(struct_node, "members", None) or []:
        name = getattr(member, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _bound_argument(codegen, parameter_type, argument, rendered) -> str | None:
    """The value a parameter of ``parameter_type`` takes when copy-initialized from ``argument``, or ``None``.

    Copy-initializing a parameter converts the argument to the parameter type ([dcl.init.general]/16.9), so an
    ``int`` parameter bound to ``2.75f`` holds ``2``. A scalar parameter therefore binds to an explicit conversion
    unless the argument already has the parameter's type. Any other parameter type folds only for an argument of
    exactly that type, because a conversion to it would run a user-defined constructor.
    """
    if not parameter_type:
        return None
    argument_type = codegen.metal_source_overload_value_type(
        codegen.expression_metal_type(argument)
    )
    if argument_type == parameter_type:
        return rendered if _SIMPLE_VALUE.fullmatch(rendered) else f"({rendered})"
    if codegen.metal_scalar_arithmetic_type_info(parameter_type) is not None:
        return f"{codegen.map_type(parameter_type)}({rendered})"
    return None


def fold_constant_constructor_call(
    codegen,
    struct_node,
    constructor,
    arguments: list[object],
    rendered_arguments: list[str],
) -> str | None:
    """Render ``Owner(member values)`` for a constant-initializing constructor call, or ``None``.

    ``arguments`` holds one argument expression per constructor parameter, defaults included, and
    ``rendered_arguments`` their rendering under the constructor's template bindings, which the caller still has
    in effect.
    """
    values = _member_value_nodes(codegen, struct_node, constructor)
    if values is None:
        return None
    parameters = list(constructor.params or [])
    if not len(arguments) == len(rendered_arguments) == len(parameters):
        return None
    parameter_names = [getattr(parameter, "name", None) for parameter in parameters]
    if not all(isinstance(name, str) and name for name in parameter_names):
        return None
    class_names = _class_scope_names(struct_node) - set(parameter_names)
    if _referenced_identifiers(values) & class_names:
        return None

    parameter_types = {
        name: codegen.metal_source_overload_value_type(
            codegen.metal_declaration_expression_type(parameter)
        )
        for name, parameter in zip(parameter_names, parameters)
    }
    bindings = {}
    for name, argument, rendered in zip(parameter_names, arguments, rendered_arguments):
        bound = _bound_argument(codegen, parameter_types[name], argument, rendered)
        if bound is None:
            return None
        bindings[name] = bound
    saved_shadow_scopes = codegen.template_binding_shadow_scopes
    saved_variable_types = codegen.current_variable_types
    # Mem-initializers are evaluated in the class scope of the constructor, so no
    # local of an enclosing function can shadow a parameter there.
    codegen.template_binding_shadow_scopes = []
    codegen.current_variable_types = {**(saved_variable_types or {}), **parameter_types}
    codegen.template_value_bindings.append(bindings)
    try:
        rendered_values = [
            codegen.generate_expression(value, False) for value in values
        ]
    finally:
        codegen.template_value_bindings.pop()
        codegen.current_variable_types = saved_variable_types
        codegen.template_binding_shadow_scopes = saved_shadow_scopes
    owner = codegen.map_struct_name(struct_node.name)
    return f"{owner}({', '.join(rendered_values)})"
