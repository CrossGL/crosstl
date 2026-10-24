"""Concrete types in the body of a specialized generic function.

A generic function's specialization (:func:`.generic_function_utils._ensure_generic_function_specialization`) is a
copy whose signature takes the concrete types of the call. Its body kept the type parameters, so
``generic<T> T twice(T x) { T y = x + x; return y; }`` specialized for ``float`` declared ``T y``: the Metal and HLSL
generators emitted that undeclared type, and the GLSL and SPIR-V generators refused it.
:func:`substitute_body_types` gives every declaration, cast and constructor call in the body its concrete type.
"""

from __future__ import annotations

from ..ast import ASTNode, FunctionCallNode, IdentifierNode
from ..semantics.rewrite import child_slots, get_child
from .enum_utils import substitute_generic_type_name

__all__ = ["substitute_body_types"]

# Fields that hold a type; an alias field holding the same object is updated with it.
_TYPE_FIELDS = ("var_type", "vtype", "target_type", "const_type")


def _substituted(generator, value, substitutions):
    if value is None:
        return None
    name = generator.type_name_string(value)
    concrete = substitute_generic_type_name(name, substitutions)
    return None if concrete == name else concrete


def _walk(node):
    stack, seen = [node], set()
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, ASTNode) and id(item) not in seen:
            seen.add(id(item))
            yield item
            stack.extend(
                get_child(item, field, key) for field, key in child_slots(item)
            )


def substitute_body_types(generator, body, substitutions) -> int:
    """Replace the type parameters in ``body``'s types by ``substitutions``; returns how many fields changed."""
    if not substitutions:
        return 0
    changed = 0
    for node in _walk(body):
        replaced = {}
        for field in _TYPE_FIELDS:
            value = getattr(node, field, None)
            if value is None:
                continue
            if id(value) not in replaced:
                replaced[id(value)] = _substituted(generator, value, substitutions)
            concrete = replaced[id(value)]
            if concrete is not None:
                setattr(node, field, concrete)
                changed += 1
        if isinstance(node, FunctionCallNode):
            function = getattr(node, "function", None)
            name = function.name if isinstance(function, IdentifierNode) else None
            if name in substitutions:
                # ``T(0)`` constructs the concrete type.
                function.name = substitutions[name]
                function.identifier = substitutions[name]
                changed += 1
    return changed
