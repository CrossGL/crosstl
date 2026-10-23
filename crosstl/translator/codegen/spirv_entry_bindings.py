"""Explicit storage-buffer bindings of SPIR-V entry-point parameters.

An entry point that declares its resources as parameters, as Metal kernels do, can mix explicit and allocated
bindings: a ``device float *out [[buffer(0)]]`` parameter keeps binding 0, while a ``texture2d<float> source
[[texture(0)]]`` parameter has no Vulkan binding of its own and takes the next free one. MSL numbers buffers,
textures and samplers in separate index spaces (MSL 5.2.1), so both parameters can say 0. Parameters are processed
in order, so the texture used to take binding 0 before the buffer claimed it, and the buffer was then refused as a
duplicate. :func:`reserve_entry_buffer_bindings` reserves every explicit storage-buffer binding of an entry point's
parameters before any of them is allocated, so the allocated ones go around them. Two parameters that claim the same
binding are still refused when the second one is declared.
"""

from __future__ import annotations

__all__ = ["reserve_entry_buffer_bindings"]


def reserve_entry_buffer_bindings(codegen, parameters) -> int:
    """Reserve the explicit ``(set, binding)`` of each storage-buffer parameter; returns how many were reserved."""
    reserved = 0
    for parameter in parameters:
        declared = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if declared is None:
            continue
        type_name = codegen.type_name_from_value(declared)
        if not codegen.is_structured_buffer_declared_type_name(type_name):
            continue
        binding = codegen.explicit_descriptor_binding(parameter)
        if binding is None:
            continue
        key = (codegen.resource_descriptor_set(parameter), binding)
        if key not in codegen.used_resource_bindings:
            codegen.reserved_resource_bindings.add(key)
            reserved += 1
    return reserved
