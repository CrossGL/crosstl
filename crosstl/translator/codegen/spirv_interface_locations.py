"""The locations of the user varyings in a SPIR-V entry point's stage interface.

SPIR-V links a stage's outputs to the next stage's inputs by ``Location``, and the host binds vertex inputs by it, so
the emitter places them as every CrossTL generator does (:mod:`.stage_io_semantic_locations`): an explicit
``location`` wins, a classic semantic (``POSITION``, ``TEXCOORDn``, ``COLORn``, ...) takes its fixed location, and
every other user value takes the next free location, skipping the fixed ones. Built-in values take none.

Each structure the entry point reads or returns is placed as one interface, avoiding the locations earlier structures
of the same entry point took, and the entry point's other parameters then avoid every structure member's location.
The emitter used to give an unplaced member its index in the structure, counting built-ins, so a structure such as
``{POSITION, BLENDWEIGHT, NORMAL}`` put ``BLENDWEIGHT`` and ``NORMAL`` both at location 1 and was refused, and the
CrossGL spellings the HLSL importer writes (``TexCoord0``, ``Color0``) took other locations than in GLSL, Metal and
WGSL output.
"""

from __future__ import annotations

from .spirv_stage_distances import entry_point_distance_layout
from .stage_io_semantic_locations import allocate_locations

__all__ = [
    "plan_entry_input_locations",
    "plan_struct_member_locations",
]


def plan_struct_member_locations(
    codegen, members, metadata, distances, execution_model, storage_class
):
    """``member name -> location`` of the user varyings among a stage structure's ``members``."""
    names, entries = _struct_entries(
        codegen, members, metadata, distances, execution_model, storage_class
    )
    occupied = _occupied_locations(codegen, storage_class)
    return dict(zip(names, allocate_locations(entries, occupied)))


def plan_entry_input_locations(codegen, parameters, execution_model):
    """``(id(parameter), member name or None) -> location`` of the user inputs an entry point's ``parameters`` read.

    ``parameters`` are ``(parameter, value type)`` pairs of the stage inputs, in declaration order.
    """
    planned, taken, loose = {}, set(), []
    for param, value_type in parameters:
        struct_name = value_type.type.base_type
        members = codegen.current_struct_members.get(struct_name)
        if not members:
            loose.append((param, value_type))
            continue
        metadata = codegen.struct_member_metadata.get(struct_name, {})
        distances = entry_point_distance_layout(codegen, members, metadata)
        names, entries = _struct_entries(
            codegen, members, metadata, distances, execution_model, "Input"
        )
        locations = allocate_locations(entries, taken)
        for name, location, (_fixed, span) in zip(names, locations, entries):
            planned[(id(param), name)] = location
            taken.update(range(location, location + max(span, 1)))

    keys, entries = [], []
    for param, value_type in loose:
        fixed = _fixed_location(
            codegen, execution_model, "Input", param, value_type, None
        )
        if fixed is False:
            continue
        keys.append((id(param), None))
        entries.append((fixed, codegen.interface_location_span(param)))
    planned.update(zip(keys, allocate_locations(entries, taken)))
    return planned


def _struct_entries(
    codegen, members, metadata, distances, execution_model, storage_class
):
    names, entries = [], []
    for member_type, member_name in members:
        node = metadata.get(member_name, {}).get("node")
        if distances.slot(member_name) is not None:
            continue
        fixed = _fixed_location(
            codegen, execution_model, storage_class, node, member_type, member_name
        )
        if fixed is False:
            continue
        names.append(member_name)
        entries.append((fixed, codegen.struct_member_interface_location_span(node)))
    return names, entries


def _fixed_location(
    codegen, execution_model, storage_class, node, value_type, member_name
):
    """The explicit or semantic location of a value, ``None`` when it is allocated, ``False`` for a built-in."""
    semantic = codegen.entry_point_interface_semantic(
        execution_model, storage_class, node, value_type, member_name
    )
    if _is_builtin(codegen, semantic, storage_class):
        return False
    if node is not None:
        explicit = codegen.explicit_location_attribute(node)
        if explicit is not None:
            return explicit
    return codegen.entry_point_interface_location(
        semantic, storage_class, node, None, execution_model
    )


def _is_builtin(codegen, semantic, storage_class):
    if semantic is None:
        return False
    if storage_class == "Input" and codegen.compute_builtin_info(semantic) is not None:
        return True
    return codegen.stage_builtin_info(semantic) is not None


def _occupied_locations(codegen, storage_class):
    """Per-vertex locations the current entry point's interface already uses in ``storage_class``."""
    used = (
        codegen.used_input_locations
        if storage_class == "Input"
        else codegen.used_output_locations
    )
    scope = codegen.interface_location_scope()
    return {key[2] for key in used if key[0] == scope and key[1] == "per_vertex"}
