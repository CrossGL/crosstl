"""The desktop GLSL versions below which an enabled extension's built-ins, or a built-in overload, do not exist.

An ``#extension`` directive does not make a construct available at every version. glslang, the reference
compiler, declares the built-ins of several extensions only from a desktop version of its own, whatever the
extension specification says, so the ``#version`` line has to reach that version too. The floors below were
measured with glslangValidator: each is the lowest desktop version at which a fragment (or, for compute-only
extensions, compute) shader using the extension's built-ins compiles.

Two built-in overloads also depend on their operand types, which :mod:`.glsl_version_requirements` cannot read
from a function name alone:

- the 64-bit integer overloads of the common, integer and vector relational functions, which glslang declares
  under GL_ARB_gpu_shader_int64 only from GLSL 4.50 (the 64-bit types themselves exist from GLSL 4.00);
- ``textureOffset`` on a ``sampler2DArrayShadow``, which glslang declares from GLSL 4.30.
"""

from __future__ import annotations

import re

from .glsl_version_requirements import GLSLRequirement

__all__ = [
    "EXTENSION_VERSION_FLOORS",
    "extension_floor_requirements",
    "typed_overload_requirements",
]

# Extension -> the lowest desktop version at which glslang declares its built-ins. GLSL ES output is not
# constrained here: these extensions are desktop-only in glslang, so their GLSL ES uses are refused elsewhere.
EXTENSION_VERSION_FLOORS = {
    "GL_AMD_shader_explicit_vertex_parameter": 450,
    "GL_ARB_derivative_control": 400,
    "GL_ARB_gpu_shader_int64": 400,
    "GL_ARB_shader_ballot": 450,
    "GL_ARB_shader_clock": 450,
    "GL_ARB_shader_texture_image_samples": 430,
    "GL_ARB_sparse_texture2": 450,
    "GL_ARB_sparse_texture_clamp": 450,
    "GL_EXT_fragment_invocation_density": 450,
    "GL_EXT_fragment_shader_barycentric": 450,
    "GL_EXT_nonuniform_qualifier": 400,
    "GL_EXT_shader_atomic_float": 440,
    "GL_EXT_shader_realtime_clock": 450,
    "GL_NV_conservative_raster_underestimation": 430,
    "GL_NV_fragment_shader_barycentric": 450,
}
# Satisfied by every GLSL ES version this generator writes.
_ANY_ES = 300


def extension_floor_requirements(extensions):
    """The version requirements of the enabled ``extensions`` (names), in a stable order."""
    return [
        GLSLRequirement(
            f"the built-ins of {name}, which glslang declares from GLSL "
            f"{EXTENSION_VERSION_FLOORS[name] // 100}.{EXTENSION_VERSION_FLOORS[name] % 100 // 10}0",
            EXTENSION_VERSION_FLOORS[name],
            _ANY_ES,
        )
        for name in sorted(extensions)
        if name in EXTENSION_VERSION_FLOORS
    ]


INT64_BUILTIN_OVERLOADS = GLSLRequirement(
    "a 64-bit integer overload of a built-in function", 450, None
)
SHADOW_ARRAY_TEXTURE_OFFSET = GLSLRequirement(
    "textureOffset on a sampler2DArrayShadow", 430, _ANY_ES
)

_INT64_TYPES = r"(?:u?int64_t|[iu]64vec[234])"
_INT64_DECLARATION = re.compile(rf"\b{_INT64_TYPES}\s+([A-Za-z_]\w*)")
_INT64_VALUE = re.compile(rf"\b{_INT64_TYPES}\s*\(|\b\d+(?:[uU]?[lL])\b")
# Every function glslang overloads for 64-bit integers under GL_ARB_gpu_shader_int64 (glslang Initialize.cpp).
_INT64_FUNCTIONS = (
    "abs",
    "sign",
    "min",
    "max",
    "clamp",
    "mix",
    "lessThan",
    "lessThanEqual",
    "greaterThan",
    "greaterThanEqual",
    "packInt2x32",
    "unpackInt2x32",
    "packUint2x32",
    "unpackUint2x32",
    "doubleBitsToInt64",
    "doubleBitsToUint64",
    "int64BitsToDouble",
    "uint64BitsToDouble",
)
# Functions that exist only for 64-bit integer operands or results.
_INT64_ONLY_FUNCTIONS = frozenset(
    _INT64_FUNCTIONS[_INT64_FUNCTIONS.index("packInt2x32") :]
)
_INT64_CALL = re.compile(r"\b(" + "|".join(_INT64_FUNCTIONS) + r")\s*\(")
# A uniform or a function parameter of the type.
_SHADOW_ARRAY_DECLARATION = re.compile(r"\bsampler2DArrayShadow\s+([A-Za-z_]\w*)")
_TEXTURE_OFFSET_CALL = re.compile(r"\btextureOffset\s*\(\s*([A-Za-z_]\w*)\b")


def _call_arguments(text, start):
    """The text between the parenthesis that opens at ``start - 1`` and the one that closes it."""
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:]


def _calls_int64_overload(body):
    names = frozenset(_INT64_DECLARATION.findall(body))
    if not names and not _INT64_VALUE.search(body):
        return False
    names_pattern = (
        re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(names))) + r")\b")
        if names
        else None
    )
    for match in _INT64_CALL.finditer(body):
        if match.group(1) in _INT64_ONLY_FUNCTIONS:
            return True
        arguments = _call_arguments(body, match.end())
        if _INT64_VALUE.search(arguments) or (
            names_pattern is not None and names_pattern.search(arguments)
        ):
            return True
    return False


def typed_overload_requirements(body, extensions):
    """The requirements of the built-in overloads in comment-free program text ``body`` that depend on types.

    A call of a listed function with an argument that names a 64-bit integer variable or value counts (``equal``
    and ``notEqual`` are not listed: glslang has their 64-bit overloads from GLSL 4.00). A variable named like a
    64-bit one in another scope makes the check conservative, never permissive.
    """
    requirements = []
    if "GL_ARB_gpu_shader_int64" in extensions and _calls_int64_overload(body):
        requirements.append(INT64_BUILTIN_OVERLOADS)
    shadow_arrays = frozenset(_SHADOW_ARRAY_DECLARATION.findall(body))
    if shadow_arrays and any(
        match.group(1) in shadow_arrays for match in _TEXTURE_OFFSET_CALL.finditer(body)
    ):
        requirements.append(SHADOW_ARRAY_TEXTURE_OFFSET)
    return requirements
