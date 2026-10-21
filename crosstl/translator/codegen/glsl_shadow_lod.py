"""Level-zero depth compares on GLSL shadow samplers that have no ``textureLod`` overload.

GLSL 4.60 and GLSL ES 3.00 define ``textureLod`` and ``textureLodOffset`` on ``sampler2DShadow`` only (section 8.9.2,
"Texel Lookup Functions"); ``samplerCubeShadow`` and ``sampler2DArrayShadow`` need GL_EXT_texture_shadow_lod. Both
have ``textureGrad`` overloads, and ``sampler2DArrayShadow`` a ``textureGradOffset`` one. Explicit zero derivatives
give a scale factor of 0, so the computed level of detail is clamped to the sampler's minimum and the base level is
sampled with the magnification filter, which is what an explicit level 0 selects (OpenGL 4.6 section 8.14). An
explicit level of zero, the CrossGL form of HLSL ``SampleCmpLevelZero``, is therefore emitted as a gradient sample
with zero derivatives, as SPIRV-Cross does for these samplers.

Any other explicit level, and any level on ``samplerCubeArrayShadow`` (which has no ``textureGrad`` overload), needs
GL_EXT_texture_shadow_lod, which adds ``textureLod`` on ``sampler2DArrayShadow``, ``samplerCubeShadow`` and
``samplerCubeArrayShadow`` (``textureLod(samplerCubeArrayShadow, vec4 P, float compare, float lod)``) and
``textureLodOffset`` on ``sampler2DArrayShadow``, for OpenGL 3.0 and OpenGL ES 3.0 and later. The GLSL emitter
requires the extension when it emits one of those overloads; WebGL 2 exposes no such extension, so it refuses them.
"""

from __future__ import annotations

import re

from ..ast import LiteralNode

__all__ = [
    "GLSL_SHADOW_LOD_EXTENSION",
    "SHADOW_LOD_EXTENSION_SAMPLERS",
    "require_shadow_lod_extension",
    "is_zero_lod_literal",
    "zero_lod_compare_gradient_call",
]

GLSL_SHADOW_LOD_EXTENSION = "GL_EXT_texture_shadow_lod"
# Shadow samplers whose textureLod overloads GL_EXT_texture_shadow_lod adds, and whether it adds textureLodOffset.
SHADOW_LOD_EXTENSION_SAMPLERS = {
    "sampler2DArrayShadow": True,
    "samplerCubeShadow": False,
    "samplerCubeArrayShadow": False,
}

# Shadow sampler: (gradient vector type, whether textureGradOffset is defined for it).
_GRADIENT_SHADOWS = {
    "samplerCubeShadow": ("vec3", False),
    "sampler2DArrayShadow": ("vec2", True),
}


def is_zero_lod_literal(node) -> bool:
    """Whether ``node`` is a numeric literal equal to zero."""
    if not isinstance(node, LiteralNode) or isinstance(node.value, bool):
        return False
    value = node.value
    if isinstance(value, str):
        value = value.strip().rstrip("fFhHlLuU")
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def zero_lod_compare_gradient_call(
    texture_type, texture_name, compare_coord, lod_node, offset=None
):
    """``textureGrad[Offset]`` with zero derivatives for a level-zero compare, or ``None`` when it does not apply."""
    shape = _GRADIENT_SHADOWS.get(texture_type)
    if shape is None or not is_zero_lod_literal(lod_node):
        return None
    gradient, has_offset = shape
    zero = f"{gradient}(0.0)"
    if offset is None:
        return f"textureGrad({texture_name}, {compare_coord}, {zero}, {zero})"
    if not has_offset:
        return None
    return (
        f"textureGradOffset({texture_name}, {compare_coord}, {zero}, {zero}, {offset})"
    )


def require_shadow_lod_extension(code):
    """``code`` with ``#extension GL_EXT_texture_shadow_lod : require`` after its ``#version`` line."""
    if re.search(
        rf"^\s*#extension\s+{GLSL_SHADOW_LOD_EXTENSION}\s*:\s*(?:enable|require)\b",
        code,
        re.MULTILINE,
    ):
        return code
    line = f"#extension {GLSL_SHADOW_LOD_EXTENSION} : require\n"
    match = re.search(r"^\s*#version[^\n]*\n", code, re.MULTILINE)
    if match is None:
        return line + code
    return code[: match.end()] + line + code[match.end() :]
