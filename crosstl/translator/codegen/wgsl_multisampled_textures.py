"""Multisampled textures in the WGSL generator.

GLSL's ``sampler2DMS``, ``isampler2DMS`` and ``usampler2DMS`` are read only with ``texelFetch(t, P, sample)`` and
queried with ``textureSize(t)`` and ``textureSamples(t)`` (GLSL 4.60 sections 8.9.1 and 8.9.2); they are never
sampled through a sampler. WGSL declares them as ``texture_multisampled_2d<T>`` (WGSL section 6.5.4) and reads them
with ``textureLoad(t, coords, sample_index)``, ``textureDimensions(t)`` and ``textureNumSamples(t)``, so the WGSL
emitter declares only the texture binding, without the companion sampler a sampled texture gets.

WGSL has no multisampled array texture, so ``sampler2DMSArray`` and its integer forms stay refused as resource types
WGSL cannot express.
"""

from __future__ import annotations

from ..refusals import WGSLRefusal

__all__ = ["MULTISAMPLED_TEXTURE_TYPE_MAP", "WGSLMultisampledTextureMixin"]

# CrossGL multisampled texture types and the WGSL texture each declares (WGSL section 6.5.4).
MULTISAMPLED_TEXTURE_TYPE_MAP = {
    "sampler2dms": "texture_multisampled_2d<f32>",
    "isampler2dms": "texture_multisampled_2d<i32>",
    "usampler2dms": "texture_multisampled_2d<u32>",
    "texture2dms": "texture_multisampled_2d<f32>",
}


class WGSLMultisampledTextureMixin:
    """Multisampled texture members of ``WGSLCodeGen``."""

    def multisampled_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        if type_name is None:
            return None
        return MULTISAMPLED_TEXTURE_TYPE_MAP.get(type_name)

    def generate_multisampled_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise WGSLRefusal(
                "WGSL target does not support initializers for multisampled "
                f"texture resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return (
            f"{attributes}\nvar {self.module_identifier_name(node.name)}: "
            f"{texture_type};"
        )
