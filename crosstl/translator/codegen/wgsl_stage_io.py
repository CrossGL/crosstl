"""Location and interpolation rules for WGSL user-defined stage IO.

WGSL requires every entry-point parameter to carry either ``@builtin`` or
``@location`` (WGSL section 12.3.1, "Entry Point IO"), and every
user-defined vertex output or fragment input of integer type to be declared
``@interpolate(flat)`` (WGSL section 12.3.1.4, "Interpolation"). CrossGL lets
both stay implicit, so the WGSL generator fills them in here, and writes the
CrossGL interpolation qualifiers (``@ flat``, ``@ noperspective``,
``@ centroid``, ``@ sample`` and their HLSL spellings) as ``@interpolate``.

A system-value semantic (``gl_*`` or ``SV_*``) that has no WGSL built-in
value (WGSL section 13.3.1.1, "Built-in Inputs and Outputs") is refused
rather than demoted to a user ``@location``, which would silently turn a
pipeline-provided value into an unrelated vertex attribute or varying.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..validation import (
    INTERPOLATION_MODE_METADATA_NAMES,
    INTERPOLATION_SAMPLING_METADATA_NAMES,
)
from .stage_utils import normalize_stage_name

__all__ = [
    "WGSLStageIOUnsupportedError",
    "entry_parameter_locations",
    "interpolation_qualifiers",
    "is_system_semantic_key",
    "member_interpolation",
    "refuse_unmapped_system_semantic",
    "vertex_output_interpolation",
    "with_interpolation",
]

USER_IO_STAGES = frozenset({"vertex", "fragment"})

# Colour-target semantics are user locations, not built-in values.
_COLOR_TARGET_RE = re.compile(r"(?:sv_target\d*|gl_fragcolor\d*|gl_fragdata)")


class WGSLStageIOUnsupportedError(TranslationDiagnostic):
    """A stage input or output that WGSL entry-point IO cannot express."""

    def __init__(self, reason):
        super().__init__(
            reason,
            target="wgsl",
            code=f"{CODE_PREFIX}wgsl-stage-io-unsupported",
        )


def is_system_semantic_key(key):
    """Return whether a normalized semantic names a system value."""
    return key.startswith(("gl_", "sv_")) and not _COLOR_TARGET_RE.fullmatch(key)


def refuse_unmapped_system_semantic(codegen, node, context):
    """Refuse a system-value semantic the WGSL generator cannot map."""
    for attr in getattr(node, "attributes", []) or []:
        semantic = str(getattr(attr, "name", attr))
        key = codegen.semantic_key(semantic)
        if key in codegen.BUILTIN_SEMANTICS:
            return
        if is_system_semantic_key(key):
            raise WGSLStageIOUnsupportedError(
                f"WGSL target has no built-in value for system semantic "
                f"{semantic} on {context}"
            )


_INTEGER_IO_TYPE_RE = re.compile(r"(?:i32|u32|vec[234]<(?:i32|u32)>)")


def is_integer_io_type(wgsl_type_name):
    """Return whether a WGSL type spelling is an integer scalar or vector."""
    return bool(_INTEGER_IO_TYPE_RE.fullmatch(str(wgsl_type_name).strip()))


# The WGSL interpolation type of each interpolating CrossGL mode.
_INTERPOLATION_TYPES = {"noperspective": "linear", "smooth": "perspective"}


def interpolation_qualifiers(codegen, attributes):
    """Return the CrossGL interpolation mode and sampling named by ``attributes``.

    Either is ``None`` when absent. CrossGL validation refuses a declaration
    with two modes or two samplings, so each attribute list names at most one.
    """
    mode = sampling = None
    for attr in attributes or []:
        key = codegen.semantic_key(str(getattr(attr, "name", attr)))
        mode = INTERPOLATION_MODE_METADATA_NAMES.get(key, mode)
        sampling = INTERPOLATION_SAMPLING_METADATA_NAMES.get(key, sampling)
    return mode, sampling


def with_interpolation(attributes, wgsl_type_name, qualifiers=(None, None), context=""):
    """Add the ``@interpolate`` a user IO declaration needs to its attributes.

    ``qualifiers`` is the CrossGL interpolation mode and sampling. Integer IO is
    always flat, and refused with any other mode. WGSL's flat interpolation
    takes only the ``first`` or ``either`` sampling, and GLSL gives ``flat`` with
    ``centroid`` or ``sample`` the meaning of ``flat`` alone (GLSL 4.60 section
    4.5), so both become ``@interpolate(flat)``. Declarations without
    ``@location`` are not user IO and an explicit ``@interpolate`` wins, so
    neither is changed.
    """
    if "@location(" not in attributes or "@interpolate(" in attributes:
        return attributes
    mode, sampling = qualifiers
    if is_integer_io_type(wgsl_type_name):
        if mode not in (None, "flat"):
            raise WGSLStageIOUnsupportedError(
                f"WGSL target cannot interpolate integer {context} with {mode}; "
                "WGSL integer user IO must be @interpolate(flat)"
            )
        mode = "flat"
    if mode == "flat":
        return f"{attributes} @interpolate(flat)"
    if sampling is not None:
        interpolation_type = _INTERPOLATION_TYPES[mode or "smooth"]
        return f"{attributes} @interpolate({interpolation_type}, {sampling})"
    if mode == "noperspective":
        return f"{attributes} @interpolate(linear)"
    return attributes


def _struct_member_slots(codegen, struct):
    """Yield each located member of a stage-interface struct with its location slots."""
    struct_name = getattr(struct, "name", "")
    for member in getattr(struct, "members", []) or []:
        member_type = getattr(member, "member_type", None)
        slot_count = codegen.stage_interface_location_slot_count(member_type)
        location = codegen._stage_interface_member_locations.get(
            (struct_name, member.name)
        )
        if location is None:
            binding_kind, explicit = codegen.member_stage_binding(member)
            if binding_kind != "location" or explicit is None:
                continue
            location = explicit
        yield member, range(location, location + slot_count)


def _struct_member_locations(codegen, struct):
    """Return the location slots a stage-interface struct already occupies."""
    used = set()
    for _member, slots in _struct_member_slots(codegen, struct):
        used.update(slots)
    return used


def _refuse_unlocatable_parameter(codegen, parameter, stage_name):
    param_type = getattr(parameter, "param_type", None)
    if codegen.stage_interface_matrix_member_info_for_type(param_type) is not None:
        kind = "matrix"
    elif codegen.stage_interface_array_member_info_for_type(param_type) is not None:
        kind = "array"
    else:
        return
    raise WGSLStageIOUnsupportedError(
        f"WGSL target cannot bind {stage_name} entry-point parameter "
        f"'{parameter.name}' of {kind} type "
        f"{codegen.type_name_string(param_type)} to a @location; "
        "WGSL user IO must be a numeric scalar or vector, so pass it in a "
        "stage interface struct instead"
    )


def entry_parameter_locations(codegen, entry_point, stage_name, skip_parameter_ids):
    """Assign the lowest free ``@location`` to unbound entry parameters.

    Explicit locations, semantic-derived locations and the members of
    stage-interface struct parameters are reserved first. Unbound parameters
    then take the free slots in declaration order, the rule the WGSL
    generator already applies to unbound stage-interface struct members.
    Returns a mapping from ``id(parameter)`` to its location.
    """
    if stage_name not in USER_IO_STAGES or entry_point is None:
        return {}

    structs_by_name = codegen._stage_interface_structs_by_name
    used_locations = set()
    unbound_parameters = []
    for parameter in getattr(entry_point, "parameters", []) or []:
        if id(parameter) in skip_parameter_ids:
            continue
        param_type = getattr(parameter, "param_type", None)
        struct = structs_by_name.get(codegen.struct_type_name(param_type))
        if struct is not None:
            used_locations.update(_struct_member_locations(codegen, struct))
            continue
        binding_kind, location = codegen.member_stage_binding(parameter)
        if binding_kind == "builtin":
            continue
        refuse_unmapped_system_semantic(
            codegen,
            parameter,
            f"{stage_name} entry-point parameter '{parameter.name}'",
        )
        _refuse_unlocatable_parameter(codegen, parameter, stage_name)
        if binding_kind == "location":
            if location is not None:
                used_locations.add(location)
            continue
        if (
            codegen.scalar_type_name(param_type) is None
            and codegen.vector_shape(param_type) is None
        ):
            # Textures, samplers and pointers are not user IO.
            continue
        unbound_parameters.append(parameter)

    locations = {}
    next_location = 0
    for parameter in unbound_parameters:
        while next_location in used_locations:
            next_location += 1
        locations[id(parameter)] = next_location
        used_locations.add(next_location)
        next_location += 1
    return locations


def _fragment_input_qualifiers(codegen, entry_point):
    """Map each location a fragment entry point reads to its interpolation qualifiers."""
    structs_by_name = codegen._stage_interface_structs_by_name
    parameters = getattr(entry_point, "parameters", []) or []
    resource_ids = {
        id(parameter)
        for parameter in parameters
        if codegen.is_stage_resource_parameter(parameter)
    }
    implicit = entry_parameter_locations(
        codegen, entry_point, "fragment", resource_ids
    )
    qualifiers = {}
    for parameter in parameters:
        if id(parameter) in resource_ids:
            continue
        param_type = getattr(parameter, "param_type", None)
        struct = structs_by_name.get(codegen.struct_type_name(param_type))
        if struct is not None:
            for member, slots in _struct_member_slots(codegen, struct):
                for slot in slots:
                    qualifiers[slot] = interpolation_qualifiers(
                        codegen, member.attributes
                    )
            continue
        location = implicit.get(id(parameter))
        if location is None:
            binding_kind, location = codegen.member_stage_binding(parameter)
            if binding_kind != "location" or location is None:
                continue
        qualifiers[location] = interpolation_qualifiers(codegen, parameter.attributes)
    return qualifiers


def member_interpolation(codegen, struct_name, member, location):
    """Return the interpolation mode and sampling of a struct member slot.

    A vertex output slot that a fragment input reads takes that input's
    qualifiers (:func:`vertex_output_interpolation`); others keep their own.
    """
    mirrored = codegen._vertex_output_interpolation.get((struct_name, location))
    if mirrored is not None:
        return mirrored
    return interpolation_qualifiers(codegen, getattr(member, "attributes", None))


def vertex_output_interpolation(codegen, stage_nodes):
    """Give vertex outputs the interpolation of the fragment inputs they feed.

    WebGPU refuses a render pipeline whose vertex output and fragment input at
    one location differ in interpolation (WGSL section 13.3.1.4). GLSL, HLSL
    and MSL interpolate a varying as the fragment input declares it, so a
    vertex output takes the qualifiers of the fragment input at its location.
    Returns a mapping from ``(struct name, location)`` of each such vertex
    output struct slot to the fragment input's mode and sampling.
    """
    entry_points = {}
    for stage_node in stage_nodes:
        stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
        entry_point = codegen.stage_output_lowered_entry_point(
            stage_node, getattr(stage_node, "entry_point", None)
        )
        if stage_name in USER_IO_STAGES and entry_point is not None:
            entry_points[stage_name] = entry_point
    if set(entry_points) != USER_IO_STAGES:
        return {}

    fragment = _fragment_input_qualifiers(codegen, entry_points["fragment"])
    struct = codegen._stage_interface_structs_by_name.get(
        codegen.struct_type_name(getattr(entry_points["vertex"], "return_type", None))
    )
    if struct is None:
        return {}
    return {
        (struct.name, slot): fragment[slot]
        for _member, slots in _struct_member_slots(codegen, struct)
        for slot in slots
        if slot in fragment
    }
