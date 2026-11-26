"""WGSL output of the textures and samplers that a stage entry point takes as parameters.

A Metal entry point takes its textures and samplers as parameters (``texture2d<float> tex [[texture(0)]]``),
and the Metal importer keeps them as parameters of the CrossGL stage entry with ``@ texture(n)`` and
``@ sampler(n)``. A WGSL entry point's parameters must all be stage inputs with ``@builtin`` or
``@location`` (WGSL, "Entry Point Declaration"); textures and samplers are module-scope ``var``
declarations with ``@group`` and ``@binding`` (WGSL, "Resource Interface"). Such a parameter is declared
as the module variable of the same name, as an entry point's storage buffer parameters already are, and the
body's references resolve to it.

Metal numbers buffers, textures and samplers in separate index spaces, so ``[[buffer(0)]]``,
``[[texture(0)]]`` and ``[[sampler(0)]]`` name three resources. The binding allocator treats each
attribute as a register class of its own, as it treats HLSL's ``t``, ``s``, ``u`` and ``b`` registers: a
binding that another class already holds moves to the next free binding of the group.
"""

from __future__ import annotations

from ..ast import VariableNode

# The attribute spelling of each Metal index space, and the register class the binding allocator gives it.
METAL_BINDING_CLASSES = {
    "buffer": "metal-buffer",
    "texture": "metal-texture",
    "sampler": "metal-sampler",
}


def is_entry_opaque_resource_parameter(codegen, parameter):
    """Whether ``parameter`` of a stage entry point is a texture, storage image or sampler."""
    param_type = getattr(parameter, "param_type", None)
    if param_type is None:
        return False
    if codegen.sampled_texture_type(param_type) is not None:
        return True
    if codegen.multisampled_texture_type(param_type) is not None:
        return True
    if codegen.is_sampler_type(param_type):
        return True
    type_name = codegen.resource_type_name(param_type) or ""
    return type_name in codegen.STORAGE_TEXTURE_DIMENSION_MAP


def generate_entry_opaque_resource(codegen, parameter):
    """The module-scope declaration of an entry point's texture, storage image or sampler ``parameter``."""
    variable = VariableNode(
        parameter.name,
        parameter.param_type,
        attributes=list(getattr(parameter, "attributes", None) or []),
        qualifiers=list(getattr(parameter, "qualifiers", None) or []),
        resource_qualifiers=list(getattr(parameter, "resource_qualifiers", None) or []),
    )
    reserved_sampler = codegen._reserved_sampler_bindings.get(id(parameter))
    if reserved_sampler is not None:
        codegen._reserved_sampler_bindings[id(variable)] = reserved_sampler
    return codegen.generate_global_variable(variable)
