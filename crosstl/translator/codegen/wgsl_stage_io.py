"""Location and interpolation rules for WGSL user-defined stage IO.

WGSL requires every entry-point parameter to carry either ``@builtin`` or
``@location`` (WGSL section 12.3.1, "Entry Point IO"), and every
user-defined vertex output or fragment input of integer type to be declared
``@interpolate(flat)`` (WGSL section 12.3.1.4, "Interpolation"). CrossGL lets
both stay implicit, so the WGSL generator fills them in here.

A system-value semantic (``gl_*`` or ``SV_*``) that has no WGSL built-in
value (WGSL section 13.3.1.1, "Built-in Inputs and Outputs") is refused
rather than demoted to a user ``@location``, which would silently turn a
pipeline-provided value into an unrelated vertex attribute or varying.
"""

from __future__ import annotations

import re

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "WGSLStageIOUnsupportedError",
    "entry_parameter_locations",
    "is_system_semantic_key",
    "refuse_unmapped_system_semantic",
    "with_flat_integer_interpolation",
]

USER_IO_STAGES = frozenset({"vertex", "fragment"})

# Colour-target semantics are user locations, not built-in values.
_COLOR_TARGET_RE = re.compile(r"(?:sv_target\d*|gl_fragcolor)")


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


def with_flat_integer_interpolation(attributes, wgsl_type_name):
    """Add ``@interpolate(flat)`` to integer user IO that lacks interpolation."""
    if "@location(" not in attributes or "@interpolate(" in attributes:
        return attributes
    if not is_integer_io_type(wgsl_type_name):
        return attributes
    return f"{attributes} @interpolate(flat)"


def _struct_member_locations(codegen, struct):
    """Return the location slots a stage-interface struct already occupies."""
    used = set()
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
        used.update(range(location, location + slot_count))
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
