"""SPIR-V output of ``QuadReadLaneAt`` with a lane index computed at run time.

``OpGroupNonUniformQuadBroadcast`` takes its ``Index`` from a constant instruction before SPIR-V 1.5, and from 1.5
on the index must still be dynamically uniform. HLSL's
``QuadReadLaneAt``, Metal's ``quad_broadcast`` and ``quad_shuffle``, and CrossGL's ``QuadReadLaneAt`` accept a lane
that differs per invocation. Such a lane is read the way GLSL output's ``crossglQuadReadLaneAt`` reads it: every
invocation of the quad broadcasts the value of each of the four quad lanes with a constant index, and ``OpSelect``
returns the one ``lane & 3`` selects. That is the value ``OpGroupNonUniformQuadBroadcast`` reads for the lane, in
every stage where quad operations run, because the quad membership is the one the broadcast itself uses. A lane
above 3, which HLSL and Metal leave undefined, reads quad lane ``lane & 3``, as GLSL, CUDA and HIP output read it.
"""

from __future__ import annotations

from .spirv_values import SpirvId

__all__ = ["lower_spirv_quad_lane_read"]

_QUAD_LANES = 4


def lower_spirv_quad_lane_read(codegen, value: SpirvId, lane: SpirvId) -> SpirvId:
    """``QuadReadLaneAt(value, lane)`` for a lane that is not a constant, as four broadcasts and a selection."""
    uint_type = codegen.register_primitive_type("uint")
    bool_type = codegen.register_primitive_type("bool")
    result_type = codegen.ensure_registered_type(value.type)
    lane = codegen.convert_wave_lane_operand("QuadReadLaneAt", lane)
    if lane is None:
        return codegen.wave_result_default("QuadReadLaneAt", [value])

    codegen.require_group_non_uniform("GroupNonUniformQuad")
    scope = codegen.subgroup_scope_id()
    variables = []
    for index in range(_QUAD_LANES):
        constant = codegen.register_constant(index, uint_type)
        read = _emit(
            codegen,
            "OpGroupNonUniformQuadBroadcast",
            result_type,
            scope,
            value,
            constant,
        )
        # Keeping each read in a variable stops SPIRV-Cross (MoltenVK) from moving the broadcasts into the
        # branches of the C conditional it writes for the selection, where the other quad lanes do not take part.
        variable = codegen.create_variable(result_type, "Function")
        codegen.store_to_variable(variable, read)
        variables.append(variable)
    reads = [
        codegen.load_from_variable(variable, result_type) for variable in variables
    ]

    three = codegen.register_constant(_QUAD_LANES - 1, uint_type)
    quad_lane = _emit(codegen, "OpBitwiseAnd", uint_type, lane, three)
    condition_type = _condition_type(codegen, result_type, bool_type)
    selected = reads[_QUAD_LANES - 1]
    for index in range(_QUAD_LANES - 2, -1, -1):
        constant = codegen.register_constant(index, uint_type)
        condition = _emit(codegen, "OpIEqual", bool_type, quad_lane, constant)
        if condition_type is not bool_type:
            condition = codegen.splat_scalar_to_vector(condition, condition_type)
        selected = _emit(
            codegen, "OpSelect", result_type, condition, reads[index], selected
        )
    return selected


def _condition_type(codegen, result_type: SpirvId, bool_type: SpirvId) -> SpirvId:
    """The ``OpSelect`` condition type for ``result_type``: ``bool``, or a ``bool`` vector of its width."""
    vector = codegen.vector_component_type_and_count(result_type.type.base_type)
    if vector is None:
        return bool_type
    return codegen.register_vector_type(bool_type, vector[1])


def _emit(codegen, opcode: str, result_type: SpirvId, *operands: SpirvId) -> SpirvId:
    id_value = codegen.get_id()
    operand_text = " ".join(f"%{operand.id}" for operand in operands)
    codegen.emit(f"%{id_value} = {opcode} %{result_type.id} {operand_text}")
    codegen.value_types[id_value] = result_type
    return SpirvId(id_value, result_type.type)
