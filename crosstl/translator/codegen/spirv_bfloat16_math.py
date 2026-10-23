"""bfloat16 operands of floating-point math builtins in SPIR-V.

The SPIR-V emitter stores ``bfloat16`` as 16-bit storage bits, because core SPIR-V
and GLSL.std.450 have no brain-float type, and it already evaluates bfloat16
arithmetic in binary32 before rounding back (``binary_operation``).

The math builtins need the same treatment. MSL defines its ``bfloat`` math
functions as the binary32 function of the widened operands, rounded to bfloat
(MSL specification section 6.5, "Math Functions"; MLX's ``bf16_math.h`` spells the
same definition out for older compilers). Every bfloat16 value is exactly
representable in binary32, so:

- value-returning functions widen each bfloat16 operand, call the binary32
  lowering and round the result to bfloat16 with round-to-nearest-even;
- the classification predicates (``isnan``, ``isinf``, ``isfinite``,
  ``signbit``) widen their operand and return the binary32 answer, which is exact.

A call that mixes bfloat16 and other floating-point operands is left to the host's
operand diagnostics, because its result type is not bfloat16 by construction.

The mixin relies on the host class's ``call_builtin_function``,
``convert_value_to_type``, ``ensure_registered_type``, ``register_primitive_type``,
``register_vector_type``, ``scalar_or_vector_component_type`` and
``vector_component_type_and_count`` helpers and on ``BFLOAT16_TYPE_NAME``.
"""

from __future__ import annotations

BFLOAT16_VALUE_FUNCTIONS = frozenset(
    {
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "ceil",
        "clamp",
        "copysign",
        "cos",
        "cosh",
        "exp",
        "exp10",
        "exp2",
        "fabs",
        "floor",
        "fma",
        "fmax",
        "fmin",
        "fmod",
        "fract",
        "inversesqrt",
        "log",
        "log10",
        "log2",
        "max",
        "min",
        "mix",
        "mod",
        "pow",
        "rint",
        "round",
        "roundEven",
        "saturate",
        "sign",
        "sin",
        "sinh",
        "smoothstep",
        "sqrt",
        "step",
        "tan",
        "tanh",
        "trunc",
    }
)
BFLOAT16_PREDICATE_FUNCTIONS = frozenset({"isfinite", "isinf", "isnan", "signbit"})
FLOATING_COMPONENT_TYPES = frozenset({"float", "double", "half"})


class SpirvBfloat16MathMixin:
    """Evaluate bfloat16 math builtins in binary32 and round the result back."""

    def call_bfloat16_math_function(self, function_name: str, args: list):
        """Lower a math builtin with bfloat16 operands, or return None if not applicable."""
        predicate = function_name in BFLOAT16_PREDICATE_FUNCTIONS
        if not predicate and function_name not in BFLOAT16_VALUE_FUNCTIONS:
            return None
        components = [self._bfloat16_math_component(arg) for arg in args]
        if self.BFLOAT16_TYPE_NAME not in components:
            return None
        if any(component in FLOATING_COMPONENT_TYPES for component in components):
            return None
        promoted = [
            (
                self._widen_bfloat16_operand(arg)
                if component == self.BFLOAT16_TYPE_NAME
                else arg
            )
            for arg, component in zip(args, components)
        ]
        result = self.call_builtin_function(function_name, promoted)
        if result is None or predicate:
            return result
        result_type = self.ensure_registered_type(result.type)
        if self.scalar_or_vector_component_type(result_type.type) != "float":
            return result
        return self.convert_value_to_type(
            result, self._bfloat16_type_like(result_type.type.base_type)
        )

    def _bfloat16_math_component(self, value) -> str | None:
        value_type = getattr(value, "type", None)
        if value_type is None:
            return None
        return self.scalar_or_vector_component_type(
            self.ensure_registered_type(value_type).type
        )

    def _widen_bfloat16_operand(self, value):
        value_type = self.ensure_registered_type(value.type)
        float_type = self.register_primitive_type("float")
        vector_info = self.vector_component_type_and_count(value_type.type.base_type)
        if vector_info is not None:
            float_type = self.register_vector_type(float_type, vector_info[1])
        return self.convert_value_to_type(value, float_type)

    def _bfloat16_type_like(self, base_type_name: str):
        bfloat_type = self.register_primitive_type(self.BFLOAT16_TYPE_NAME)
        vector_info = self.vector_component_type_and_count(base_type_name)
        if vector_info is None:
            return bfloat_type
        return self.register_vector_type(bfloat_type, vector_info[1])
