"""Polyfills for source-language math functions that GLSL.std.450 lacks.

GLSL.std.450 has ``Log`` and ``Log2`` but no base-10 logarithm, while MSL, HLSL,
CUDA, HIP and C all define ``log10``. Without a lowering the SPIR-V emitter has no
correct output for the call. The polyfill is the one DXC's SPIR-V back end emits
for HLSL ``log10``: ``Log2(x) * log10(2)``. The extra rounding of the product
stays within the precision Vulkan allows for ``Log2`` (Vulkan 1.3 specification,
"Precision and Operation of SPIR-V Instructions": 3 ULP outside [0.5, 2.0],
absolute error 2^-21 inside it).

``Log2`` accepts only 16- and 32-bit floating-point operands, so the polyfill
applies to 32-bit float scalars and vectors; other operands fall through to the
host's operand diagnostics.

The mixin relies on the host class's ``emit_glsl_std450_instruction``,
``binary_operation``, ``ensure_registered_type``, ``register_primitive_type``,
``register_constant``, ``splat_scalar_to_vector``,
``vector_component_type_and_count`` and ``float32_scalar_or_vector_operand_is_valid``
helpers.
"""

from __future__ import annotations

import math

# log10(2), rounded to binary32 by register_constant.
LOG10_OF_2 = math.log10(2.0)


class SpirvMathPolyfillMixin:
    """Lower math functions that need more than one GLSL.std.450 instruction."""

    def call_math_polyfill_function(self, function_name: str, args: list):
        """Lower a polyfilled math call, or return None if not applicable."""
        if function_name == "log10" and len(args) == 1:
            return self.emit_log10_polyfill(args[0])
        return None

    def emit_log10_polyfill(self, value):
        if not self.float32_scalar_or_vector_operand_is_valid(value):
            return None
        result_type = self.ensure_registered_type(value.type)
        log2 = self.emit_glsl_std450_instruction("Log2", result_type, [value])
        float_type = self.register_primitive_type("float")
        scale = self.register_constant(LOG10_OF_2, float_type)
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        if vector_info is not None:
            scale = self.splat_scalar_to_vector(scale, result_type)
        return self.binary_operation("*", result_type, log2, scale)
