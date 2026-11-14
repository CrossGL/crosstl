"""The scope type the Metal generator registers for a global it passes to a function in a buffer.

A global array such as ``uniform float3 lights[N]`` becomes the parameter ``constant float3* lights [[buffer(n)]]``.
Registering the parameter with its element type made ``lights[i]`` read as a component access on ``float3``, so the
element was typed ``float``: ``twice(lights[i])`` deduced ``T = float`` while emitting, a specialization that
collection, which reads the declaration, never created. The parameter is registered with the declaration's array
type instead, so an element access yields the element type in both passes.
"""

from __future__ import annotations

__all__ = ["metal_buffer_resource_scope_type"]


def metal_buffer_resource_scope_type(codegen, buffer_type, array_size):
    """The type of a buffer-resource global inside a function: its element type, with ``[n]`` for an array."""
    type_name = codegen.type_name_string(buffer_type)
    if array_size is None or not type_name:
        return type_name
    return f"{type_name}[{array_size}]"
