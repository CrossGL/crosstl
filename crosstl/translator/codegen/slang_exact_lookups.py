"""Exact stage, semantic and call-name lookups for the Slang generator.

The generator used to look these names up with ``mapping.get(key, key)``, which writes a key the table does not
know into Slang as CrossGL spelled it. This module replaces those lookups:

- :data:`SLANG_SHADER_STAGE_NAMES` spells every CrossGL stage as Slang's ``[shader("...")]`` attribute names it
  (Slang user's guide, "Entry Points"); a name that is no stage is refused with
  :class:`~crosstl.translator.stage_utils.UnknownShaderStageError`.
- :func:`slang_semantic_spelling_is_valid` checks the text a declaration would write after a colon. Slang
  semantics are HLSL's: an identifier with an optional index, ``SV_`` system values included; other text is
  refused where it would be written. (``semantic_suffix`` also refuses a CrossGL ``gl_`` built-in the semantic
  table does not rename, which slangc would read as a user semantic no stage writes.)
- :data:`SLANG_BITCAST_INTRINSICS` lists the call names that are Slang's bit-preserving casts.
- :func:`slang_foreign_call_reason` refuses a call that would be written with a built-in function name of another
  language. Slang's core module declares the HLSL intrinsics, and the inverse hyperbolic functions
  :data:`SLANG_CORE_FUNCTION_NAMES` lists besides; a GLSL, MSL, CUDA or CrossGL built-in the generator did not
  lower or rename is neither, so a call to it would not compile or would reach a function the program declares
  for another purpose.
"""

from __future__ import annotations

from ..semantics.builtin_function_names import (
    BUILTIN_FUNCTION_NAMES,
    HLSL_FUNCTION_NAMES,
)
from ..stage_utils import requested_stage_name
from .hlsl_semantic_attributes import HLSL_SEMANTIC_NAME

__all__ = [
    "SLANG_BITCAST_INTRINSICS",
    "SLANG_CORE_FUNCTION_NAMES",
    "SLANG_SHADER_STAGE_NAMES",
    "slang_foreign_call_reason",
    "slang_shader_stage",
    "slang_semantic_spelling_is_valid",
]

SLANG_SHADER_STAGE_NAMES = {
    "vertex": "vertex",
    "fragment": "fragment",
    "geometry": "geometry",
    "tessellation_control": "hull",
    "tessellation_evaluation": "domain",
    "compute": "compute",
    "task": "amplification",
    "amplification": "amplification",
    "object": "amplification",
    "mesh": "mesh",
    "ray_generation": "raygeneration",
    "ray_intersection": "intersection",
    "ray_any_hit": "anyhit",
    "ray_closest_hit": "closesthit",
    "ray_miss": "miss",
    "ray_callable": "callable",
}

# Each call name the generator writes as one of Slang's bit-preserving casts, and the cast it becomes.
SLANG_BITCAST_INTRINSICS = {
    "asfloat": "asfloat",
    "asint": "asint",
    "asuint": "asuint",
    "floatBitsToInt": "asint",
    "floatBitsToUint": "asuint",
    "intBitsToFloat": "asfloat",
    "uintBitsToFloat": "asfloat",
}

# Functions Slang's core module declares that HLSL does not: the inverse hyperbolic functions.
SLANG_CORE_FUNCTION_NAMES = frozenset({"acosh", "asinh", "atanh"})

_SLANG_CALLABLE_FUNCTION_NAMES = HLSL_FUNCTION_NAMES | SLANG_CORE_FUNCTION_NAMES


def slang_shader_stage(stage_name):
    """The ``[shader("...")]`` spelling of the CrossGL stage ``stage_name``; ``None`` for no stage."""
    canonical = requested_stage_name(stage_name)
    if canonical is None:
        return None
    return SLANG_SHADER_STAGE_NAMES[canonical]


def slang_semantic_spelling_is_valid(semantic_text):
    """Whether ``semantic_text`` can stand after the colon of a Slang declaration."""
    return HLSL_SEMANTIC_NAME.fullmatch(semantic_text) is not None


def slang_foreign_call_reason(callee, function_map):
    """Why a call to ``callee`` has no Slang callee, or ``None`` when it has one.

    Only a built-in function name of a source language that ``function_map`` does not rename is refused: any other
    name is a function, constructor or type the program declares, which ``declared_functions`` has checked.
    """
    if not isinstance(callee, str) or callee not in BUILTIN_FUNCTION_NAMES:
        return None
    if callee in function_map or callee in _SLANG_CALLABLE_FUNCTION_NAMES:
        return None
    return (
        f"call to {callee}: {callee} is a built-in function of another language, and "
        "Slang declares no function of that name"
    )
