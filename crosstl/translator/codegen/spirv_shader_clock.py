"""SPIR-V lowering of the GLSL shader clock built-ins.

``GL_ARB_shader_clock`` and ``GL_EXT_shader_realtime_clock`` read a hardware counter: ``clockARB`` and
``clockRealtimeEXT`` return a ``uint64_t``, ``clock2x32ARB`` and ``clockRealtime2x32EXT`` the same value as a
``uvec2`` (low word first). SPIR-V expresses all four with ``OpReadClockKHR`` (``SPV_KHR_shader_clock``,
capability ``ShaderClockKHR``): the ``ARB`` forms read it at ``Subgroup`` scope and the ``Realtime`` forms at
``Device`` scope, which is the instruction glslang emits for each. The 64-bit result needs the ``Int64`` capability.
"""

from __future__ import annotations

__all__ = ["SPIRV_SHADER_CLOCKS", "spirv_shader_clock_call"]

# CrossGL name -> (scope, result is the two-word form).
SPIRV_SHADER_CLOCKS = {
    "clockARB": (3, False),
    "clock2x32ARB": (3, True),
    "clockRealtimeEXT": (1, False),
    "clockRealtime2x32EXT": (1, True),
}


def spirv_shader_clock_call(codegen, function_name, arguments, spirv_id_class):
    """Emit ``OpReadClockKHR`` for the shader clock call ``function_name(arguments...)``."""
    scope, pair = SPIRV_SHADER_CLOCKS[function_name]
    uint_type = codegen.register_primitive_type("uint")
    result_type = (
        codegen.register_vector_type(uint_type, 2)
        if pair
        else codegen.register_primitive_type("u64")
    )
    if arguments:
        codegen.refuse_lowering(
            f"shader clock builtin '{function_name}' requires 0 operands; "
            f"got {len(arguments)}"
        )
        return spirv_id_class(codegen.get_id(), result_type.type)
    codegen.require_capability("ShaderClockKHR")
    codegen.require_extension("SPV_KHR_shader_clock")
    scope_id = codegen.register_constant(scope, uint_type)
    result_id = codegen.get_id()
    codegen.emit(f"%{result_id} = OpReadClockKHR %{result_type.id} %{scope_id.id}")
    return spirv_id_class(result_id, result_type.type)
