"""Exact name tables for the GLSL generator, with a refusal for every name outside them.

The generator looked names up as ``mapping.get(name, name)``, which writes a name the table does not know into the
output as it stands: a call to a function neither GLSL nor the program defines, an HLSL-only ray-tracing operation,
a semantic that names no GLSL built-in, or a structure member the structure does not declare. Each of those is now
looked up exactly:

- A call classification (derivative, interpolation, component-wise, boolean reduction, read-only built-in) asks
  whether the call reaches one of a finite set of GLSL functions: :func:`call_names_resolving_to` lists the call
  names that do, which are the set's own names that ``function_map`` does not rename and the CrossGL names it
  renames into the set. A name outside that list is classified as none of them.
- Mesh and ray-tracing operations have one GLSL function each, with a fixed parameter count
  (GL_EXT_mesh_shader section 8.x, GL_EXT_ray_tracing sections 6.4 and 8.19); any other operation or argument count
  is refused.
- A call whose callee the program does not declare must name a GLSL built-in function, a constructor or an
  array's ``length()`` method (GLSL 4.60 sections 5.4, 5.9 and 8). A name the rendered program then declares is a
  helper the generator wrote; anything else is refused instead of being written as a call nothing defines.
- A semantic ``semantic_map`` does not rename is a user varying only if it is an identifier outside the reserved
  ``gl_`` and ``SV_`` namespaces; a ``gl_`` name must be a built-in variable (:mod:`..semantics.builtin_names`).
- A member or identifier name is looked up in the generator's own name map; a name the map does not hold is
  written only when it is an identifier GLSL does not reserve.

Every refusal goes through ``refuse_lowering``, so the program is reported with the generator's typed
``LOWERING_REFUSAL_TYPE`` and never returned.
"""

from __future__ import annotations

import re

from ..semantics.builtin_function_names import (
    GLSL_EXTENSION_FUNCTION_NAMES,
    GLSL_FUNCTION_NAMES,
)
from ..semantics.builtin_names import is_crossgl_gl_builtin

__all__ = [
    "GLSL_BITCAST_CANONICAL_NAMES",
    "GLSL_CALLABLE_FUNCTION_NAMES",
    "GLSL_MESH_INTRINSICS",
    "GLSL_RAY_TRACING_INTRINSICS",
    "call_names_resolving_to",
    "glsl_call_names_resolving_to",
    "glsl_mesh_intrinsic_call",
    "glsl_note_call_target",
    "glsl_ray_tracing_intrinsic_call",
    "glsl_undeclared_call_reasons",
    "glsl_unmapped_identifier",
    "glsl_unmapped_semantic_name",
]

# GL_EXT_shader_tile_image section 8.x: reads of the tile image attachments.
GLSL_TILE_IMAGE_FUNCTION_NAMES = frozenset(
    {"colorAttachmentReadEXT", "depthAttachmentReadEXT", "stencilAttachmentReadEXT"}
)

# The GLSL functions a call may name without the program declaring them.
GLSL_CALLABLE_FUNCTION_NAMES = (
    GLSL_FUNCTION_NAMES | GLSL_EXTENSION_FUNCTION_NAMES | GLSL_TILE_IMAGE_FUNCTION_NAMES
)

# CrossGL mesh operation -> (GL_EXT_mesh_shader function, parameter count):
# ``void SetMeshOutputsEXT(uint vertexCount, uint primitiveCount)`` and
# ``void EmitMeshTasksEXT(uint groupCountX, uint groupCountY, uint groupCountZ)``.
GLSL_MESH_INTRINSICS = {
    "SetMeshOutputCounts": ("SetMeshOutputsEXT", 2),
    "DispatchMesh": ("EmitMeshTasksEXT", 3),
}

# CrossGL ray-tracing operation -> (GL_EXT_ray_tracing spelling, parameter count). ``traceRayEXT`` takes the
# acceleration structure, ray flags, cull mask, SBT offset and stride, miss index, origin, Tmin, direction, Tmax
# and the payload location; ``reportIntersectionEXT(float hitT, uint hitKind)``;
# ``executeCallableEXT(uint sbtRecordIndex, int callable)``. ``terminateRayEXT`` and ``ignoreIntersectionEXT``
# are jump statements (section 6.4) and take no operands.
GLSL_RAY_TRACING_INTRINSICS = {
    "TraceRay": ("traceRayEXT", 11),
    "ReportHit": ("reportIntersectionEXT", 2),
    "CallShader": ("executeCallableEXT", 2),
    "AcceptHitAndEndSearch": ("terminateRayEXT", 0),
    "IgnoreHit": ("ignoreIntersectionEXT", 0),
}
GLSL_RAY_TRACING_JUMP_STATEMENTS = frozenset(
    {"terminateRayEXT", "ignoreIntersectionEXT"}
)

# Bitcast spelling -> the generator's canonical bitcast name (GLSL 4.60 section 8.3 and HLSL asint/asuint/asfloat).
GLSL_BITCAST_CANONICAL_NAMES = {
    "floatBitsToInt": "asint",
    "floatBitsToUint": "asuint",
    "intBitsToFloat": "asfloat",
    "uintBitsToFloat": "asfloat",
    "asint": "asint",
    "asuint": "asuint",
    "asfloat": "asfloat",
}

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A constructor of an array type: an element type followed by one or more dimensions (GLSL 4.60 section 5.4.4).
_ARRAY_CONSTRUCTOR = re.compile(r"(?P<element>[A-Za-z_][A-Za-z0-9_]*)(?:\[[^\[\]]*\])+")
# The ``length()`` method of an array, vector or matrix (GLSL 4.60 section 5.9).
_LENGTH_METHOD = re.compile(r".+\.length")
# A qualifier or return type before a declared function's name; a return type may be an array (section 6.1).
_DECLARATION_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\[\]]*\])*")
# Words that may stand before a call at the start of a line without making it a declaration.
_STATEMENT_WORDS = frozenset({"return", "else", "do", "case", "default", "goto"})


def call_names_resolving_to(mapping, names):
    """The call names ``mapping`` resolves to a member of ``names``: the members it does not rename, and its keys
    whose value is a member."""
    names = frozenset(names)
    return frozenset(name for name in names if name not in mapping) | frozenset(
        key for key, value in mapping.items() if value in names
    )


def glsl_call_names_resolving_to(generator, label, names, mapping=None):
    """:func:`call_names_resolving_to` over ``generator.function_map`` (or ``mapping``), cached under ``label``.

    ``names`` must be the same set for every call with one ``label``; the cache is rebuilt when the mapping object
    changes.
    """
    mapping = generator.function_map if mapping is None else mapping
    cache = generator.__dict__.setdefault("_glsl_call_names_resolving_to", {})
    entry = cache.get(label)
    if entry is None or entry[0] is not mapping:
        entry = (mapping, call_names_resolving_to(mapping, names))
        cache[label] = entry
    return entry[1]


def glsl_mesh_intrinsic_call(generator, operation, arguments):
    """The GLSL call for mesh operation ``operation`` with the generated ``arguments``, or a refusal."""
    entry = GLSL_MESH_INTRINSICS.get(operation)
    if entry is None:
        return generator.refuse_lowering(
            f"mesh operation '{operation}' has no GL_EXT_mesh_shader function"
        )
    function_name, arity = entry
    if len(arguments) != arity:
        return generator.refuse_lowering(
            f"mesh operation '{operation}' becomes {function_name}, which takes {arity} "
            f"arguments; the call passes {len(arguments)}"
        )
    return f"{function_name}({', '.join(arguments)})"


def glsl_ray_tracing_intrinsic_call(generator, operation, arguments):
    """The GLSL form of ray-tracing operation ``operation`` with the generated ``arguments``, or a refusal."""
    entry = GLSL_RAY_TRACING_INTRINSICS.get(operation)
    if entry is None:
        return generator.refuse_lowering(
            f"ray-tracing operation '{operation}' has no GL_EXT_ray_tracing form"
        )
    function_name, arity = entry
    if len(arguments) != arity:
        return generator.refuse_lowering(
            f"ray-tracing operation '{operation}' becomes {function_name}, which takes {arity} "
            f"operands; the call passes {len(arguments)}"
        )
    if function_name in GLSL_RAY_TRACING_JUMP_STATEMENTS:
        return function_name
    return f"{function_name}({', '.join(arguments)})"


def glsl_unmapped_semantic_name(generator, semantic):
    """The GLSL meaning of a semantic ``semantic_map`` does not rename: the semantic itself, if it may be one.

    A ``gl_`` name must be a GLSL or CrossGL built-in variable; anything else in that namespace, an ``SV_`` system
    value no table maps, and text that is not an identifier are refused rather than declared as a user varying.
    """
    text = str(semantic)
    if text.lower().startswith("gl_"):
        if is_crossgl_gl_builtin(text):
            return text
        return generator.refuse_lowering(
            f"semantic '{text}' is in the gl_ namespace GLSL reserves (GLSL 4.60 section 3.7) but names no "
            "built-in variable"
        )
    if text.lower().startswith("sv_"):
        return generator.refuse_lowering(
            f"semantic '{text}' is a Direct3D system value with no GLSL built-in variable"
        )
    if _IDENTIFIER.fullmatch(text):
        return text
    return generator.refuse_lowering(
        f"semantic '{text}' is not an identifier, so it cannot name a GLSL varying"
    )


def glsl_unmapped_identifier(generator, name, what):
    """``name`` written as itself, when no name map renames it: refused unless GLSL can spell it unchanged."""
    text = str(name)
    if _IDENTIFIER.fullmatch(text) and not generator.glsl_identifier_requires_escape(
        text
    ):
        return text
    return generator.refuse_lowering(
        f"{what} '{text}' has no GLSL name: it is not an identifier GLSL leaves to programs"
    )


def _is_constructor(generator, name):
    if generator.glsl_constructor_type(name) is not None:
        return True
    structs = getattr(generator, "structs_by_name", None) or {}
    return name in structs


def glsl_note_call_target(generator, name, argument_count):
    """Check the callee of a call to no function the program declares, before it is written as ``name(...)``.

    A GLSL built-in function, a constructor (including an array constructor) and ``length()`` pass. Another
    identifier is recorded for :func:`glsl_undeclared_call_reasons`, which accepts it only when the rendered
    program declares it. Any other callee text is refused at once: GLSL has no function values or other methods.
    """
    if not isinstance(name, str) or not name:
        return
    if name in GLSL_CALLABLE_FUNCTION_NAMES or _is_constructor(generator, name):
        return
    array_constructor = _ARRAY_CONSTRUCTOR.fullmatch(name)
    if array_constructor is not None and _is_constructor(
        generator, array_constructor.group("element")
    ):
        return
    if argument_count == 0 and _LENGTH_METHOD.fullmatch(name):
        return
    if _IDENTIFIER.fullmatch(name):
        pending = generator.__dict__.setdefault("glsl_unresolved_call_targets", {})
        pending.setdefault(name, None)
        return
    generator.refuse_lowering(
        f"call through '{name}': GLSL calls only named functions, constructors and length()"
    )


def _declares_function(code, name):
    """Whether a line of ``code`` declares or defines ``name``: qualifiers and a return type, then ``name(``."""
    for match in re.finditer(rf"(?m)^[ \t]*(.*?)\b{re.escape(name)}[ \t]*\(", code):
        prefix = match.group(1).split()
        if prefix == ["#define"]:
            return True
        if not prefix or not all(_DECLARATION_WORD.fullmatch(word) for word in prefix):
            continue
        if prefix[-1] not in _STATEMENT_WORDS:
            return True
    return False


def glsl_undeclared_call_reasons(generator, code):
    """Refusal reasons for the recorded callees that ``code`` calls but neither declares nor defines."""
    pending = generator.__dict__.get("glsl_unresolved_call_targets") or {}
    reasons = []
    for name in sorted(pending):
        if not re.search(rf"\b{re.escape(name)}[ \t]*\(", code):
            continue
        if _declares_function(code, name):
            continue
        reasons.append(
            f"call to '{name}', which is neither a GLSL built-in function nor a function the program declares"
        )
    return reasons
