"""Rust resource bindings: each resource reaches the functions that use it as a parameter, or is refused.

rust-gpu passes every resource to an entry point as a parameter with a ``#[spirv(descriptor_set, binding)]``
attribute; a module-level ``static`` is not a resource binding. Only a single texture or sampler that an entry point
used directly, with an explicit binding, used to become such a parameter. Every other resource was written as a
default-initialized ``static`` under a comment calling it a "compile-only placeholder": the program compiled, and
each read of the resource returned a default value instead of the bound data.

Now every resource kind, fixed-size and runtime-sized arrays of them, and stage-local resources reach the entry
points that use them, directly or through the helper functions they call, as bound parameters, with the explicit
binding or the one the generator assigns; a helper receives the resources it needs as trailing parameters. A
resource the program refers to that still cannot be bound (one with an initializer, or one referenced outside the
program's functions) is refused; a resource nothing refers to has no value to lose and is still declared.
"""

from __future__ import annotations

from ..ast import FunctionCallNode, IdentifierNode, ParameterNode
from .ast_walk import walk_ast_values

__all__ = [
    "PROMOTABLE_RESOURCE_KINDS",
    "referenced_identifier_names",
    "refuse_unbound_rust_resource",
    "resource_parameters",
]

# Every resource kind of the Rust runtime (``rust_exact_lookups.RUST_RESOURCE_KINDS``, the HLSL buffer objects and
# GLSL buffer blocks) has an entry-parameter form: rust-gpu binds images, samplers and acceleration structures in
# the UniformConstant storage class and buffers with ``#[spirv(storage_buffer, ...)]``.
PROMOTABLE_RESOURCE_KINDS = frozenset(
    {
        "acceleration_structure",
        "buffer",
        "glsl_buffer_block",
        "image",
        "sampler",
        "texture",
    }
)


def referenced_identifier_names(ast):
    """Every name an identifier expression in ``ast`` spells, scope aside.

    Scope is ignored on purpose: a local that shadows a resource only makes the resource look used, which refuses a
    program that could have been translated, never the reverse.
    """
    return frozenset(
        value.name
        for value in walk_ast_values(ast)
        if isinstance(value, IdentifierNode)
    )


def _called_names(function, user_function_names):
    """The user functions ``function`` calls by name."""
    called = set()
    for value in walk_ast_values(getattr(function, "body", None)):
        if not isinstance(value, FunctionCallNode):
            continue
        callee = getattr(value, "function", getattr(value, "name", None))
        if isinstance(callee, IdentifierNode):
            callee = callee.name
        if isinstance(callee, str) and callee in user_function_names:
            called.add(callee)
    return called


def _parameter_names(function):
    return {
        param.name
        for param in getattr(function, "parameters", getattr(function, "params", []))
    }


def _used_names(function):
    """The names ``function`` refers to that are not its parameters, counting a shadowed name as used."""
    return referenced_identifier_names(
        getattr(function, "body", None)
    ) - _parameter_names(function)


def resource_parameters(entries, helpers, resources):
    """Thread each resource to the entry points that use it, directly or through the helpers they call.

    ``entries`` maps ``id(entry)`` to an entry function, ``helpers`` lists every other function, and ``resources``
    maps a resource name to its ``{"node", "type_name"}`` record. A helper that uses a resource, or calls a helper
    that does, receives it as a trailing parameter (helpers are keyed by name, so every overload of a name takes the
    same resources), and each entry point that reaches it receives it as a bound parameter.

    A name is used wherever an identifier spells it, even under a local of the same name: the extra parameter
    such a shadowing local hides is harmless, while a missed use would leave the resource unbound.

    Returns ``(entry parameters by id, helper parameters by name)``, each list in resource declaration order.
    """
    helper_names = {getattr(helper, "name", None) for helper in helpers} - {None}
    direct = {}
    calls = {}
    for helper in helpers:
        name = getattr(helper, "name", None)
        if name is None:
            continue
        used = _used_names(helper)
        direct.setdefault(name, set()).update(set(resources) & used)
        calls.setdefault(name, set()).update(_called_names(helper, helper_names))
    reach = {name: set(names) for name, names in direct.items()}
    changed = True
    while changed:
        changed = False
        for name in sorted(reach):
            before = len(reach[name])
            for callee in calls[name]:
                reach[name] |= reach.get(callee, set())
            changed |= len(reach[name]) != before

    def ordered(names):
        return [resources[name] for name in resources if name in names]

    entry_params = {}
    for entry_id, entry in entries.items():
        used = set(resources) & _used_names(entry)
        for callee in _called_names(entry, helper_names):
            used |= reach.get(callee, set())
        params = ordered(used - _parameter_names(entry))
        if params:
            entry_params[entry_id] = params
    helper_params = {
        name: [
            ParameterNode(info["node"].name, info["type_name"])
            for info in ordered(names)
        ]
        for name, names in sorted(reach.items())
        if names
    }
    return entry_params, helper_params


def _reason(generator, node, type_name, resource_kind):
    if resource_kind not in PROMOTABLE_RESOURCE_KINDS:
        why = f"{resource_kind} resources have no rust-gpu entry-parameter lowering"
    elif getattr(node, "initial_value", getattr(node, "value", None)) is not None:
        why = "a resource with an initializer cannot be bound"
    elif generator.is_array_type_name(generator.type_name_string(type_name)):
        why = "a runtime-sized or specialization-sized resource array has no rust-gpu entry-parameter lowering"
    else:
        why = "it is referenced outside the program's functions, where no entry-point parameter reaches"
    return (
        f"resource: '{node.name}' ({type_name}) would be a module-level static with a default value, not a "
        f"rust-gpu resource binding: {why}"
    )


def refuse_unbound_rust_resource(generator, node, type_name):
    """Refuse ``node``, a resource the generator writes as a static, when the program refers to it.

    Returns the empty string, so a call can stand where the declaration's prefix is assembled.
    """
    name = getattr(node, "name", None)
    resource_kind = generator.rust_resource_kind(type_name, node=node)
    if not name or resource_kind is None:
        return ""
    if name not in generator.rust_referenced_names:
        return ""
    return generator.refuse_lowering(_reason(generator, node, type_name, resource_kind))
