"""CrossGL spellings of the Slang core-module functions that a CrossGL built-in means under another name.

Slang's core module (``hlsl.meta.slang``, Slang v2026.20) spells GLSL's unpacking functions with the result type in
the name: ``unpackUnorm4x8ToFloat`` is GLSL's ``unpackUnorm4x8``, and ``unpackUnorm4x8ToHalf`` is defined as
``half4(unpackUnorm4x8ToFloat(x))`` on every target but Metal, where it is MSL's ``unpack_unorm4x8_to_half``, the
same value. ``copysign_half``, ``copysign_float`` and ``copysign_double`` are ``copysign`` for one floating-point
type, and ``EvaluateAttributeAtCentroid`` is GLSL's ``interpolateAtCentroid`` (``InterpolateAtCentroid`` on SPIR-V).
"""

from __future__ import annotations

__all__ = [
    "SLANG_CORE_FUNCTION_IMPORTS",
    "crossgl_half_unpack_call",
]

SLANG_CORE_FUNCTION_IMPORTS = {
    "unpackHalf2x16ToFloat": "unpackHalf2x16",
    "unpackSnorm2x16ToFloat": "unpackSnorm2x16",
    "unpackSnorm4x8ToFloat": "unpackSnorm4x8",
    "unpackUnorm2x16ToFloat": "unpackUnorm2x16",
    "unpackUnorm4x8ToFloat": "unpackUnorm4x8",
    "copysign_half": "copysign",
    "copysign_float": "copysign",
    "copysign_double": "copysign",
    "EvaluateAttributeAtCentroid": "interpolateAtCentroid",
}

# Each half-precision unpacking function: the CrossGL unpacking function and the half vector it converts to.
_HALF_UNPACKS = {
    "unpackHalf2x16ToHalf": ("unpackHalf2x16", "f16vec2"),
    "unpackSnorm2x16ToHalf": ("unpackSnorm2x16", "f16vec2"),
    "unpackSnorm4x8ToHalf": ("unpackSnorm4x8", "f16vec4"),
    "unpackUnorm2x16ToHalf": ("unpackUnorm2x16", "f16vec2"),
    "unpackUnorm4x8ToHalf": ("unpackUnorm4x8", "f16vec4"),
}


def crossgl_half_unpack_call(name, rendered_args):
    """CrossGL source of a call to a Slang ``unpack...ToHalf`` function, or ``None`` for any other callee."""
    if name not in _HALF_UNPACKS or len(rendered_args) != 1:
        return None
    function, vector_type = _HALF_UNPACKS[name]
    return f"{vector_type}({function}({rendered_args[0]}))"
