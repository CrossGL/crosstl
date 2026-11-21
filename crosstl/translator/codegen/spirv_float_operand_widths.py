"""Built-in calls whose floating operands mix ``float`` and ``double`` take the ``double`` overload.

GLSL 4.60 section 6.1 resolves ``max(0.9, x)`` with a ``double`` ``x`` to ``max(double, double)``: ``float`` converts to
``double`` implicitly (section 4.1.10) and never back, so the ``float`` overload is not viable. HLSL, C and CUDA promote
the same way. The SPIR-V lowerings type a call by its first operand, so before this pass they converted ``x`` to
``float`` and computed in binary32. Widening the ``float`` operands first is exact, and an unsuffixed literal operand
becomes the ``double`` constant of its decimal value (``spirv_float_literals``).

``refract`` decides by its vectors ``I`` and ``N`` alone. Its ratio ``eta`` is a scalar parameter that the lowering
already converts to the vectors' component type, and GLSL.std.450 Refract types it apart from the result, so a
``double`` ``eta`` does not move a ``float`` refraction into binary64.
"""

from __future__ import annotations

from .spirv_values import SpirvId

__all__ = ["WIDENED_FLOAT_BUILTINS", "widen_float_operands"]

# The built-ins (after ``spirv_builtin_call_key``) whose floating operands share one component type.
WIDENED_FLOAT_BUILTINS = frozenset(
    {
        "atan",
        "atan2",
        "clamp",
        "cross",
        "distance",
        "dot",
        "faceforward",
        "fma",
        "fmod",
        "max",
        "min",
        "mix",
        "mod",
        "pow",
        "reflect",
        "refract",
        "smoothstep",
        "step",
    }
)

_FLOAT_COMPONENTS = frozenset({"float", "double"})


def _component(codegen, value: SpirvId) -> str:
    return codegen.normalize_primitive_name(
        codegen.scalar_or_vector_component_type(value.type)
    )


def widen_float_operands(
    codegen, function_name: str, args: list[SpirvId]
) -> list[SpirvId]:
    """``args`` with each ``float`` scalar or vector converted to ``double`` when another operand is ``double``.

    Calls with a non-floating operand (``mix`` with a Boolean selector, an integer operand) keep their operands.
    """
    if function_name not in WIDENED_FLOAT_BUILTINS or len(args) < 2:
        return args
    if any(arg is None for arg in args):
        return args
    components = [_component(codegen, arg) for arg in args]
    if not all(component in _FLOAT_COMPONENTS for component in components):
        return args
    deciding = components[:2] if function_name == "refract" else components
    if "double" not in deciding or "float" not in components:
        return args
    double_type = codegen.register_primitive_type("double")
    widened = []
    for arg, component in zip(args, components):
        if component == "float":
            vector = codegen.vector_component_type_and_count(arg.type.base_type)
            target = (
                double_type
                if vector is None
                else codegen.register_vector_type(double_type, vector[1])
            )
            arg = codegen.convert_value_to_type(arg, target)
        widened.append(arg)
    return widened
