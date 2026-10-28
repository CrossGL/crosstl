"""CrossGL buffers without a declared layout are laid out by std430 rules; mark them where a target's rules differ.

A CrossGL storage buffer (``buffer T* b``, ``StructuredBuffer<T>``, ``RWStructuredBuffer<T>``) that carries no
``@buffer_layout`` attribute holds its elements at the offsets GLSL's std430 rules give (GLSL 4.60 section 7.6.2.2),
the layout the GLSL, SPIR-V and WGSL generators already emit (docs/source/adr/0005-default-buffer-layout.md). The
importers mark every buffer whose source layout differs from it (``@buffer_layout(hlsl_structured)``,
``@buffer_layout(metal)``).

MSL lays a ``device`` buffer out by its own rules: ``float3`` takes 16 bytes and ``bool`` one (MSL 3.2 section 2.2
and table 2.4), so ``struct { vec3 a; float b; }`` is 32 bytes with ``b`` at byte 16, where std430 puts ``b`` at
byte 12 in a 16-byte element. The Metal generator emitted the element type unchanged, so a kernel read and wrote
other bytes than the host and every other target.

:func:`mark_default_buffer_layouts` adds ``@buffer_layout(std430)`` to each unmarked buffer, global, stage or
parameter, whose element type MSL lays out differently. The Metal generator then declares the element with packed
vectors and explicit padding at its std430 offsets, as it does for HLSL structured buffers, or raises a typed
diagnostic for an element it cannot reproduce.

An HLSL structured buffer aligns a vector to its scalar (D3D11.3 functional specification, section 5.3.7), so
``struct { float a; vec3 b; }`` puts ``b`` at byte 4 where std430 puts it at byte 16, and ``vec3 v[]`` has a
12-byte stride where std430 has 16. For the HLSL targets the marker adds the same attribute where the HLSL layout
differs, and :func:`.structured_layouts.legalize_structured_layouts` stores the buffer as 32-bit words at the std430
offsets.

A buffer whose element type has no computable layout is left to the generator unchanged, and so is one that holds a
``bool``: each target stores booleans in its own physical size, which the project package reports to the host
(``crosstl.project``), rather than in std430's 32 bits.
"""

from __future__ import annotations

from .. import ast as A
from .structured_layouts import (
    HLSL_TARGETS,
    StructuredLayoutUnsupportedError,
    holds_boolean,
    layouts_agree,
)

__all__ = ["TARGETS", "mark_default_buffer_layouts"]

# Targets whose own buffer rules differ from std430, with the rules ``structured_layouts`` names them by.
_TARGET_RULES = {
    "metal": "metal",
    **{target: "hlsl" for target in sorted(HLSL_TARGETS)},
}
TARGETS = frozenset(_TARGET_RULES)
_WRAPPERS = frozenset({"StructuredBuffer", "RWStructuredBuffer"})
_ATTRIBUTE = "buffer_layout"


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _declarations(program):
    """Global and stage variables, then the parameters of every function, in program order."""
    yield from getattr(program, "global_variables", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_variables", None) or []
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        functions.extend(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    for function in functions:
        yield from getattr(function, "parameters", None) or []


def _qualifiers(declaration):
    return {
        str(qualifier) for qualifier in getattr(declaration, "qualifiers", None) or []
    }


def _element_type(declaration):
    """The element type of a storage-buffer declaration, or ``None`` for any other declaration."""
    type_node = getattr(declaration, "var_type", None) or getattr(
        declaration, "param_type", None
    )
    if isinstance(type_node, A.PointerType):
        space = str(getattr(type_node, "address_space", None) or "")
        if "buffer" in _qualifiers(declaration) or space == "device":
            return type_node.pointee_type
        return None
    if isinstance(type_node, A.NamedType) and type_node.name in _WRAPPERS:
        arguments = list(getattr(type_node, "generic_args", None) or [])
        return arguments[0] if len(arguments) == 1 else None
    return None


def _marked(declaration):
    return any(
        str(getattr(attribute, "name", "")).lower() == _ATTRIBUTE
        for attribute in getattr(declaration, "attributes", None) or []
    )


def mark_default_buffer_layouts(program, target) -> int:
    """Mark the unmarked buffers of ``program`` whose layout ``target`` changes ``@buffer_layout(std430)``.

    Returns how many declarations were marked.
    """
    rules = _TARGET_RULES.get(str(target or "").lower())
    if rules is None or not isinstance(program, A.ShaderNode):
        return 0
    cache, marked = {}, 0
    for declaration in _declarations(program):
        if _marked(declaration):
            continue
        element = _element_type(declaration)
        if element is None:
            continue
        context = f"buffer {getattr(declaration, 'name', '')}"
        try:
            if holds_boolean(program, element, context, cache) or layouts_agree(
                program, element, "std430", rules, context, cache
            ):
                continue
        except StructuredLayoutUnsupportedError:
            continue
        declaration.attributes = list(
            getattr(declaration, "attributes", None) or []
        ) + [A.AttributeNode(_ATTRIBUTE, [A.IdentifierNode("std430")])]
        marked += 1
    return marked
