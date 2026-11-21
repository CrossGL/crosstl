"""SPIR-V lowering of the quad votes ``QuadAll`` and ``QuadAny``.

HLSL Shader Model 6.7's ``QuadAll`` and ``QuadAny`` (GLSL ``subgroupQuadAll`` and ``subgroupQuadAny`` from
GL_EXT_shader_quad_control) evaluate a predicate over the active invocations of the quad. SPV_KHR_quad_control
defines the matching instructions, ``OpGroupNonUniformQuadAllKHR`` and ``OpGroupNonUniformQuadAnyKHR``, under the
``QuadControlKHR`` capability; DXC and glslang lower the calls to them. The instructions take a scalar Boolean
predicate and no scope operand.

A quad vote reads the invocations of a quad as the quad swaps and broadcasts do, so an entry of a compute-like stage
that reaches one declares its derivative group (``spirv_compute_derivative_groups``).
"""

from __future__ import annotations

from .spirv_compute_derivative_groups import mark_quad_operation
from .spirv_values import SpirvId

__all__ = ["QUAD_VOTE_OPCODES", "lower_spirv_quad_vote"]

QUAD_VOTE_OPCODES = {
    "QuadAll": "OpGroupNonUniformQuadAllKHR",
    "QuadAny": "OpGroupNonUniformQuadAnyKHR",
}


def lower_spirv_quad_vote(codegen, operation, args):
    """Emit the quad vote ``operation`` on the predicate ``args[0]`` and return its Boolean result."""
    if len(args) != 1:
        codegen.refuse_lowering(f"{operation} requires exactly one bool argument")
        return codegen.wave_result_default(operation, args)
    bool_type = codegen.register_primitive_type("bool")
    if args[0].type.base_type != "bool":
        codegen.refuse_lowering(
            f"{operation} requires a scalar bool argument: SPV_KHR_quad_control's "
            f"{QUAD_VOTE_OPCODES[operation]} takes a scalar Boolean predicate"
        )
        return codegen.default_value_for_type(bool_type)
    # The instruction needs no GroupNonUniform capability, but SPV_KHR_quad_control builds on SPIR-V 1.3, which
    # the generator selects for a module that declares one.
    codegen.require_group_non_uniform()
    codegen.require_extension("SPV_KHR_quad_control")
    codegen.require_capability("QuadControlKHR")
    mark_quad_operation(codegen)
    id_value = codegen.get_id()
    codegen.emit(
        f"%{id_value} = {QUAD_VOTE_OPCODES[operation]} %{bool_type.id} %{args[0].id}"
    )
    result = SpirvId(id_value, bool_type.type)
    codegen.value_types[id_value] = bool_type
    return result
