"""Slang semantics and Vulkan locations of CrossGL stage varyings placed with ``@location(N)``.

GLSL places a user varying, a vertex attribute or a fragment colour output at location ``N`` (GLSL 4.60 section
4.4.1 and 4.4.2), and the GLSL import keeps that as ``@location(N)``. Slang names a fragment colour output with
``SV_Target<N>`` and any other user varying with a user semantic; ``TEXCOORD<N>`` is the one DXC and Direct3D pair
across stages and with the vertex input layout, as the HLSL output does. A bare ``: location`` is no valid pixel
shader output (DXC rejects it) and loses ``N``.

slangc 2026.20 does not take SPIR-V locations from semantic indices. It numbers each entry point's inputs and its
outputs in declaration order, so ``SV_Target2`` declared after ``SV_Target0`` lands at Location 1, as does
``TEXCOORD2`` after ``TEXCOORD0``. ``[[vk::location(N)]]`` places a varying at Location ``N``. slangc refuses a mix
of explicit and implicit locations among an entry point's inputs or among its outputs (E39023), so a stage
structure with any placed member gives every user varying member a location, and it accepts the attribute on
structure members and entry parameters only, not on an entry point's return value (E31002).
"""

from __future__ import annotations

import re

__all__ = [
    "entry_parameter_location_prefix",
    "fragment_output_location_node_ids",
    "location_attribute_index",
    "semantic_location_index",
    "struct_member_location_prefix",
    "varying_location",
    "varying_location_semantic",
    "vk_location_prefix",
]

_NUMBERED_VARYING = re.compile(r"(?:TEXCOORD|SV_TARGET)(\d*)", re.IGNORECASE)


def location_attribute_index(attribute):
    """``N`` of a ``@location(N)`` ``attribute`` with a literal non-negative ``N``, else None."""
    if str(getattr(attribute, "name", "")).lower() != "location":
        return None
    arguments = getattr(attribute, "arguments", None) or []
    if len(arguments) != 1:
        return None
    value = getattr(arguments[0], "value", arguments[0])
    if isinstance(value, bool):
        return None
    try:
        index = int(str(value), 0)
    except ValueError:
        return None
    return index if index >= 0 else None


def varying_location(node):
    """``N`` of the ``@location(N)`` attribute on ``node``, or None when it has none."""
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == "location":
            return location_attribute_index(attribute)
    return None


def varying_location_semantic(index, fragment_output):
    """The Slang semantic of a varying at location ``index``."""
    return f"SV_Target{index}" if fragment_output else f"TEXCOORD{index}"


def semantic_location_index(semantic):
    """The index of a ``TEXCOORD<N>`` or ``SV_Target<N>`` semantic (0 without a digit), or None for any other."""
    match = _NUMBERED_VARYING.fullmatch(str(semantic or ""))
    if match is None:
        return None
    return int(match.group(1) or 0)


def vk_location_prefix(index):
    return f"[[vk::location({index})]] "


def _is_output_parameter(parameter):
    return any(
        str(qualifier).lower() in {"out", "inout"}
        for qualifier in getattr(parameter, "qualifiers", None) or []
    )


def fragment_output_location_node_ids(codegen, ast):
    """Ids of the nodes whose ``@location(N)`` names fragment colour output ``N``.

    These are the fragment entry points (for their return value), their ``out`` parameters and the members of the
    structures they return.
    """
    node_ids = set()
    for function in codegen.slang_stage_entry_functions(ast, "fragment"):
        node_ids.add(id(function))
        for parameter in (
            getattr(function, "parameters", None)
            or getattr(function, "params", None)
            or []
        ):
            if _is_output_parameter(parameter):
                node_ids.add(id(parameter))
    for struct_name in codegen.fragment_entry_output_struct_names:
        struct = codegen.user_structs_by_name.get(struct_name)
        for member in getattr(struct, "members", None) or []:
            node_ids.add(id(member))
    return node_ids


def _struct_member_locations(codegen, struct_node, default_semantics):
    """Member name to location for the user varyings of a placed stage structure, or {} when it is not placed.

    A structure is placed when a member has ``@location`` or when a fragment entry point returns it. A member whose
    ``@location`` takes the index of its ``TEXCOORD<N>`` or ``SV_Target<N>`` semantic; one with another user
    semantic, such as ``FOG``, takes the lowest location left free. System values take none.
    """
    members = getattr(struct_node, "members", None) or []
    placed = getattr(
        struct_node, "name", None
    ) in codegen.fragment_entry_output_struct_names or any(
        varying_location(member) is not None for member in members
    )
    if not placed:
        return {}
    locations = {}
    unnumbered = []
    for member in members:
        name = getattr(member, "name", None)
        location = varying_location(member)
        if location is None:
            semantic = codegen.semantic_from_node(member)
            if semantic is None:
                semantic = default_semantics.get(name)
            if semantic is None:
                continue
            mapped = str(codegen.map_semantic(semantic) or "")
            if mapped.upper().startswith("SV_") and not mapped.upper().startswith(
                "SV_TARGET"
            ):
                continue
            location = semantic_location_index(mapped)
            if location is None:
                unnumbered.append(name)
                continue
        locations[name] = location
    used = set(locations.values())
    location = 0
    for name in unnumbered:
        while location in used:
            location += 1
        locations[name] = location
        used.add(location)
    return locations


def struct_member_location_prefix(codegen, struct_node, member, default_semantics):
    """``[[vk::location(N)]] `` for a user varying member of a placed stage structure, else ``""``.

    Every user varying of such a structure is placed, so that no direction mixes explicit and implicit locations.
    """
    locations = _struct_member_locations(codegen, struct_node, default_semantics)
    location = locations.get(getattr(member, "name", None))
    return "" if location is None else vk_location_prefix(location)


def entry_parameter_location_prefix(codegen, parameter, semantic, shader_type):
    """``[[vk::location(N)]] `` for an entry parameter with ``@location(N)`` or a fragment ``SV_Target<N>`` output."""
    if shader_type is None:
        return ""
    location = varying_location(parameter)
    if location is not None:
        return vk_location_prefix(location)
    if (
        semantic is None
        or id(parameter) not in codegen.slang_fragment_output_location_ids
    ):
        return ""
    mapped = str(codegen.map_semantic(semantic, shader_type) or "")
    if not mapped.upper().startswith("SV_TARGET"):
        return ""
    location = semantic_location_index(mapped)
    return "" if location is None else vk_location_prefix(location)
