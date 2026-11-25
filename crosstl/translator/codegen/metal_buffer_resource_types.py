"""The scope type the Metal generator registers for a global it passes to a function in a buffer.

A global array such as ``uniform float3 lights[N]`` becomes the parameter ``constant float3* lights [[buffer(n)]]``.
Registering the parameter with its element type made ``lights[i]`` read as a component access on ``float3``, so the
element was typed ``float``: ``twice(lights[i])`` deduced ``T = float`` while emitting, a specialization that
collection, which reads the declaration, never created. The parameter is registered with the declaration's array
type instead, so an element access yields the element type in both passes.

A pointer global such as ``buffer float* outp`` is registered the same way (``float*``), but generic collection
only saw the globals ``metal_generic_global_types`` lists, which leaves pointers out. ``pick(outp[i])`` then had no
argument type while collecting and ``float`` while emitting, so the generator refused the call (#3764).
:func:`metal_generic_buffer_resource_types` gives collection the type emission registers for these globals.
"""

from __future__ import annotations

from ..ast import PointerType, ReferenceType

__all__ = [
    "metal_buffer_resource_scope_type",
    "metal_generic_buffer_resource_types",
]


def metal_buffer_resource_scope_type(codegen, buffer_type, array_size):
    """The type of a buffer-resource global inside a function: its element type, with ``[n]`` for an array."""
    type_name = codegen.type_name_string(buffer_type)
    if array_size is None or not type_name:
        return type_name
    return f"{type_name}[{array_size}]"


def metal_generic_buffer_resource_types(codegen, global_vars):
    """``{name: type}`` for the pointer and reference globals the Metal generator passes as buffer resources.

    Each type is the one ``register_metal_buffer_resource_parameter_scope`` gives the global inside a function that
    uses it, so generic collection deduces the same type arguments from ``outp[i]`` as emission does.
    """
    types = {}
    for node in global_vars or []:
        name = getattr(node, "name", None)
        if (
            not name
            or name in types
            or not isinstance(
                getattr(node, "var_type", None), (PointerType, ReferenceType)
            )
            or codegen.is_metal_argument_buffer_global(node)
        ):
            continue
        _, array_size, vtype, _ = codegen.metal_program_scope_variable_shape(node)
        if codegen.is_glsl_buffer_block_variable(node, vtype):
            continue
        if codegen.metal_buffer_resource_address_space(node, vtype) is None:
            continue
        type_name = metal_buffer_resource_scope_type(codegen, vtype, array_size)
        if type_name:
            types[name] = type_name
    return types
