"""Metal linkage of the values a vertex function hands to a fragment function.

Metal matches each member of a fragment function's ``[[stage_in]]`` struct to a member of the vertex function's
output by its ``[[user(name)]]`` attribute when the member carries one, and by member name otherwise (MSL
specification, "Vertex Function Output Attributes" and "Fragment Function Input Attributes").
``[[attribute(n)]]`` names a vertex-buffer attribute: a vertex ``[[stage_in]]`` member needs it, a vertex output or
fragment ``[[stage_in]]`` member ignores it, and a loose fragment parameter cannot take it at all.  A user value can
reach a fragment function only through its ``[[stage_in]]`` struct; a loose vertex or fragment parameter takes only
a built-in attribute.

The source languages link by other keys:

- GLSL and SPIR-V by ``location`` (GLSL 4.60 section 4.4.1; by name only when no location is given);
- HLSL by semantic name and index, case-insensitively, with ``TEXCOORD`` meaning ``TEXCOORD0``;
- MSL by ``user`` name, else by member name.

A varying with an explicit key therefore carries it as ``[[user(KEY)]]``: ``locn<N>`` for location ``N`` (the
spelling SPIRV-Cross uses), the upper-cased semantic name followed by its index for an HLSL semantic, and the name
itself for an MSL ``user`` attribute.  A varying that occupies several slots (a matrix, or an array that Metal needs
split into one member per element) takes consecutive locations or semantic indices, as in GLSL and HLSL.  A varying
without a key keeps its name and links by name, as it does in the source.  A system-value semantic for which Metal
has no attribute is refused, never demoted to a user varying.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .stage_distance_layout import distance_component_count, distance_semantic

__all__ = [
    "METAL_VARYING_BUILTINS",
    "MetalVaryingLinkError",
    "is_metal_varying_builtin",
    "metal_varying_attribute",
    "varying_user_key",
]

# Metal attributes, as ``canonical_metal_semantic`` spells them, that a vertex output or fragment input may carry
# instead of a user key.  ``color(n)`` and ``depth(...)`` belong to fragment outputs and are matched by prefix.
METAL_VARYING_BUILTINS = frozenset(
    {
        "amplification_count",
        "amplification_id",
        "barycentric_coord",
        "base_instance",
        "base_vertex",
        "clip_distance",
        "front_facing",
        "instance_id",
        "point_coord",
        "point_size",
        "position",
        "primitive_id",
        "render_target_array_index",
        "sample_id",
        "sample_mask",
        "stencil_ref",
        "vertex_id",
        "viewport_array_index",
    }
)

_SYSTEM_VALUE = re.compile(r"(?:sv|gl)_", re.IGNORECASE)
_INDEXED_SEMANTIC = re.compile(r"([A-Za-z_][A-Za-z0-9_]*?)([0-9]*)")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class MetalVaryingLinkError(TranslationDiagnostic):
    """A vertex output or fragment input whose linkage key Metal cannot express."""

    def __init__(self, name, reason, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ("metal.stage-varying-link",)
        super().__init__(
            f"stage varying '{name}': {reason}",
            target="metal",
            code=f"{CODE_PREFIX}metal-stage-varying-link-unsupported",
        )


def is_metal_varying_builtin(canonical) -> bool:
    """Whether ``canonical`` (a Metal attribute spelling) is a built-in rather than a user varying key."""
    if canonical is None:
        return False
    canonical = str(canonical)
    if canonical.startswith(("color(", "depth(")):
        return True
    return canonical in METAL_VARYING_BUILTINS


def varying_user_key(semantic, *, location=None, user_name=None, offset=0, name="?"):
    """The ``[[user(...)]]`` key for one slot of a varying, or None when the varying links by name.

    ``location`` wins over ``user_name``, which wins over ``semantic``; ``offset`` is the slot's index within a
    varying that occupies several slots.
    """
    if location is not None:
        return f"locn{int(location) + offset}"
    if user_name is not None:
        user_name = str(user_name).strip()
        if not _IDENTIFIER.fullmatch(user_name):
            raise MetalVaryingLinkError(
                name, f"user key '{user_name}' is not an identifier"
            )
        return user_name if offset == 0 else f"{user_name}_{offset}"
    if semantic is None:
        return None
    semantic = str(semantic).strip()
    if _SYSTEM_VALUE.match(semantic):
        raise MetalVaryingLinkError(
            name,
            f"system-value semantic '{semantic}' has no Metal vertex-output or fragment-input attribute",
        )
    match = _INDEXED_SEMANTIC.fullmatch(semantic)
    if match is None:
        raise MetalVaryingLinkError(
            name, f"semantic '{semantic}' is not an HLSL-style semantic name"
        )
    base, digits = match.groups()
    return f"{base.upper()}{int(digits or 0) + offset}"


def _attribute_named(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == name:
            return attribute
    return None


def metal_varying_attribute(codegen, node, semantic, *, offset=0):
    """The linkage attribute for ``node``, a vertex output or fragment input member or parameter.

    ``semantic`` is the member's effective semantic.  Returns None when it is a Metal built-in, which the caller
    maps as before; otherwise ``" [[user(KEY)]]"``, or ``""`` for a varying that links by name.
    """
    name = getattr(node, "name", None) or "?"
    _check_metal_distance(codegen, node, semantic, name)
    _check_metal_distance(codegen, node, codegen.semantic_from_node(node), name)
    if semantic is not None and is_metal_varying_builtin(
        codegen.canonical_metal_semantic(semantic)
    ):
        return None
    location_attribute = _attribute_named(node, "location")
    location = None
    if location_attribute is not None:
        location = codegen.explicit_location_attribute_index(node)
        if location is None:
            raise MetalVaryingLinkError(
                name, "its location is not a constant non-negative integer"
            )
    user_name = None
    user_attribute = _attribute_named(node, "user")
    if location is None and user_attribute is not None:
        arguments = getattr(user_attribute, "arguments", None) or []
        if len(arguments) != 1:
            raise MetalVaryingLinkError(name, "a user attribute takes one key")
        user_name = codegen.attribute_value_to_string(arguments[0])
    raw = codegen.semantic_from_node(node)
    if raw is not None and str(raw).lower() in {"location", "user", "attribute"}:
        raw = None
    if raw is not None and is_metal_varying_builtin(
        codegen.canonical_metal_semantic(raw)
    ):
        raw = None
    key = varying_user_key(
        raw, location=location, user_name=user_name, offset=offset, name=name
    )
    return "" if key is None else f" [[user({key})]]"


def _check_metal_distance(codegen, node, semantic, name):
    """Refuse a clip or cull distance member that Metal's one ``[[clip_distance]]`` value cannot carry.

    MSL declares clip distances as one ``float`` or ``float[n]`` member with ``[[clip_distance]]`` and has no
    cull distance attribute (MSL 3.2 section 5.2.3.3). A float vector, or a second value with a higher semantic
    index (HLSL ``SV_ClipDistance1``), would need the members packed into one array.
    """
    parsed = distance_semantic(semantic)
    if parsed is None:
        return
    builtin, index = parsed
    if builtin == "gl_CullDistance":
        raise MetalVaryingLinkError(name, "Metal has no cull distance attribute")
    if index != 0:
        raise MetalVaryingLinkError(
            name,
            f"clip distance semantic index {index}: Metal carries clip distances in one "
            "[[clip_distance]] member, so only semantic index 0 maps",
        )
    array_size = getattr(node, "size", None) if hasattr(node, "element_type") else None
    type_node = (
        getattr(node, "element_type", None)
        if array_size is not None
        else getattr(node, "member_type", None) or getattr(node, "vtype", None)
    )
    shape = distance_component_count(codegen.type_name_string(type_node), array_size)
    if shape is None or shape[0] == "vector":
        raise MetalVaryingLinkError(
            name,
            "a Metal [[clip_distance]] member is a float or a float array, not "
            f"'{codegen.type_name_string(type_node)}'",
        )
