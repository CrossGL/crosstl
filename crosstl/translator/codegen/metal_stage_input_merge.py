"""One ``[[stage_in]]`` structure for a Metal vertex or fragment entry that reads several stage inputs.

A Metal vertex or fragment function takes at most one ``[[stage_in]]`` argument; Apple's compiler rejects a second
one with "cannot use the stage_in attribute on more than one parameter". A CrossGL entry can read its inputs through
several structure parameters, as an HLSL vertex shader does with one structure per vertex buffer and one per instance,
or through a structure parameter beside loose inputs that carry semantics. The emitter gave each structure parameter
its own ``[[stage_in]]`` and wrapped the loose inputs in one more, so none of those entries compiled.

When an entry has more than one stage input, every structure parameter becomes a member of the entry's generated
``[[stage_in]]`` structure, next to the loose inputs that structure already holds, and the body starts with a copy of
each member under the parameter's name, as it does for the loose inputs. Metal accepts a nested structure as a member
of a ``[[stage_in]]`` structure (:mod:`.metal_nested_stage_io`) and reads the ``[[attribute(n)]]``, ``[[user(...)]]``
and interpolation attributes of its members as it would read them on the outer structure, so the vertex attributes
the host binds, and the links of fragment inputs to the vertex outputs, stay those of the separate structures.

A vertex attribute index names one attribute of the vertex descriptor, so two inputs whose ``[[attribute(n)]]`` ranges
overlap cannot share the structure and are refused. A loose input that takes the next free index skips the indices
the structure parameters use.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "MetalStageInputMergeError",
    "merge_metal_stage_inputs",
    "metal_stage_input_struct_parameter",
    "nested_stage_input_attribute_indices",
]

_STAGES = ("vertex", "fragment")


class MetalStageInputMergeError(TranslationDiagnostic):
    """Stage inputs of one Metal entry that cannot share its single ``[[stage_in]]`` structure."""

    code = CODE_PREFIX + "metal-stage-input-merge-unsupported"

    def __init__(self, entry: str, reason: str):
        self.entry = entry
        super().__init__(
            f"{entry}: {reason}; a Metal vertex or fragment function takes one [[stage_in]] structure",
            target="metal",
        )


def metal_stage_input_struct_parameter(
    codegen, shader_type, parameter, mapped_type, parameter_attribute
) -> dict | None:
    """The stage-input record of a structure parameter the emitter marks ``[[stage_in]]``, or None."""
    if shader_type not in _STAGES:
        return None
    if str(parameter_attribute or "").strip() != "[[stage_in]]":
        return None
    struct = codegen.structs_by_name.get(
        codegen.type_name_string(
            getattr(parameter, "param_type", None) or getattr(parameter, "vtype", None)
        )
    )
    if struct is None:
        return None
    return {
        "name": parameter.name,
        "mapped_type": mapped_type,
        "attribute": "",
        "preserve_user_attribute": True,
        "nested_struct": struct,
    }


def nested_stage_input_attribute_indices(codegen, struct) -> list[int]:
    """The vertex attribute indices the members of a structure parameter declare, with each member's span."""
    defaults = codegen.metal_default_struct_member_semantics(struct)
    indices = []
    for member in getattr(struct, "members", []) or []:
        semantic = defaults.get(getattr(member, "name", None))
        if semantic is None:
            semantic = codegen.semantic_from_node(member)
        index = codegen.metal_attribute_index_from_semantic(semantic)
        if index is not None:
            span = codegen.metal_stage_io_member_attribute_span(member)
            indices.extend(range(index, index + span))
    return indices


def _attribute_owners(codegen, inputs) -> dict[int, str]:
    """Each vertex attribute index the inputs fix, mapped to the input that declares it; overlaps are refused."""
    owners: dict[int, str] = {}
    for record in inputs:
        struct = record.get("nested_struct")
        if struct is not None:
            indices = nested_stage_input_attribute_indices(codegen, struct)
        else:
            index = codegen.metal_attribute_index_from_attribute(
                record.get("attribute")
            )
            indices = [] if index is None else [index]
        for index in indices:
            previous = owners.setdefault(index, record["name"])
            if previous != record["name"]:
                raise MetalStageInputMergeError(
                    codegen.current_function_name or "entry",
                    f"stage inputs '{previous}' and '{record['name']}' both use vertex attribute {index}",
                )
    return owners


def merge_metal_stage_inputs(
    codegen,
    shader_type,
    params,
    struct_inputs,
    vertex_stage_input_parameters,
    fragment_stage_input_parameters,
):
    """Move the structure parameters of an entry with several stage inputs into its generated stage-input structure.

    ``struct_inputs`` pairs the index in ``params`` of each structure parameter declared ``[[stage_in]]`` with its
    record from :func:`metal_stage_input_struct_parameter`. Loose inputs are already in the stage's input list, which
    the signature turns into the entry's one ``[[stage_in]]`` structure and the copies at the top of its body.
    """
    if shader_type not in _STAGES or not struct_inputs:
        return
    loose = (
        vertex_stage_input_parameters
        if shader_type == "vertex"
        else fragment_stage_input_parameters
    )
    if len(struct_inputs) + (1 if loose else 0) < 2:
        return
    for position, _record in sorted(struct_inputs, key=lambda item: -item[0]):
        del params[position]
    records = [record for _position, record in struct_inputs]
    if shader_type == "vertex":
        _attribute_owners(codegen, records + list(loose))
    loose[:0] = records
