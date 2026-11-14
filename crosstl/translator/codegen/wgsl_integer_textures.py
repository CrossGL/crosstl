"""Filtered texture lookups on integer samplers, which WGSL output refuses.

GLSL reads an ``isampler*`` or ``usampler*`` texture with ``texture``, ``textureLod``, ``textureGrad``,
``textureOffset`` and their offset forms, through the sampler's filter and wrap state. WGSL defines
``textureSample``, ``textureSampleLevel``, ``textureSampleGrad`` and ``textureSampleBias`` only for ``f32``
textures, and the built-ins that read ``i32`` and ``u32`` textures, ``textureLoad`` and ``textureGather``, cannot
apply that state. Writing ``textureSample`` produces a module that naga rejects, so the WGSL emitter refuses these
lookups with :class:`WGSLIntegerTextureSampleUnsupportedError` instead. ``texelFetch`` and ``textureSize`` on
integer samplers still translate, to ``textureLoad`` and ``textureDimensions``.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "FILTERED_TEXTURE_LOOKUPS",
    "WGSLIntegerTextureSampleUnsupportedError",
    "is_integer_sampler_type_name",
]

# Normalized names of the lookups the WGSL emitter writes as textureSample, textureSampleLevel, textureSampleGrad
# or textureSampleBias.
FILTERED_TEXTURE_LOOKUPS = frozenset(
    {
        "texture",
        "texturegrad",
        "texturegradoffset",
        "texturelod",
        "texturelodoffset",
        "textureoffset",
    }
)


class WGSLIntegerTextureSampleUnsupportedError(TranslationDiagnostic):
    """A filtered texture lookup on an integer sampler, which no WGSL built-in can perform."""

    def __init__(self, function_name, texture):
        self.function_name = function_name
        self.texture = texture
        self.missing_capabilities = ("wgsl.integer-texture-sampling",)
        super().__init__(
            f"WGSL target cannot lower {function_name}() on an integer sampler ({texture}): WGSL only samples f32 "
            "textures, and textureLoad and textureGather cannot apply the sampler's filter and wrap state; use "
            "texelFetch() instead",
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-integer-texture-sample-unsupported",
        )


def is_integer_sampler_type_name(type_name):
    """Whether ``type_name``, a lower-case CrossGL resource type name, is an ``isampler*`` or ``usampler*``."""
    return bool(type_name) and type_name.startswith(("isampler", "usampler"))
