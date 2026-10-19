"""IEEE-exact floating-point classification and remainder lowering for SPIR-V.

Vulkan lets an implementation "assume that arguments and results are not NaNs or
infinities" and ignore the sign of zero unless an entry point declares the
``SignedZeroInfNanPreserve`` execution mode (Vulkan 1.3 specification, section
"Precision and Operation of SPIR-V Instructions"). Drivers act on that licence:
MoltenVK compiles shaders with Metal fast math, so ``OpIsNan`` folds to false and
``OpFRem`` loses the sign of a zero result. Declaring the execution mode needs
``SPV_KHR_float_controls`` and a device feature that Vulkan 1.0 to 1.1 targets cannot
assume.

The source languages define these operations exactly (C99 7.12.3 and F.9.7.1, MSL
section 6.5, HLSL ``isnan``/``fmod``), so this mixin lowers them through integer
operations on the value's bits, which no floating-point optimisation may change:

- ``isnan``, ``isinf`` and ``isfinite`` test the exponent and mantissa fields;
- ``fmin``/``fmax`` (C99 7.12.12: a NaN operand yields the other operand) select
  around ``FMin``/``FMax`` with those tests;
- the truncated remainder (C ``fmod`` and the C/HLSL ``%`` on floats) copies the
  dividend's sign bit onto the ``OpFRem`` result, which C99 F.9.7.1 requires for
  zero results as well.

The mixin relies on the host class's SPIR-V value helpers (``get_id``, ``emit``,
``register_primitive_type``, ``register_vector_type``, ``register_constant``,
``splat_scalar_to_vector``, ``composite_extract``, ``composite_construct``).
"""

from __future__ import annotations

FLOAT32_SIGN_MASK = 0x80000000
FLOAT32_MAGNITUDE_MASK = 0x7FFFFFFF
FLOAT32_EXPONENT_MASK = 0x7F800000
# The high word of a binary64 value holds its sign, exponent and top mantissa bits.
FLOAT64_HIGH_EXPONENT_MASK = 0x7FF00000

FLOAT_CLASSIFICATION_FUNCTIONS = ("isnan", "isinf", "isfinite")
NAN_AVOIDING_MIN_MAX_FUNCTIONS = {"fmin": "FMin", "fmax": "FMax"}


class SpirvFloatSemanticsMixin:
    """Lower IEEE-dependent float builtins without relying on driver float modes."""

    def call_float_semantics_function(self, function_name: str, args: list):
        """Lower isnan/isinf/isfinite/fmin/fmax, or return None if not applicable."""
        if function_name in FLOAT_CLASSIFICATION_FUNCTIONS and len(args) == 1:
            return self.emit_float_classification(function_name, args[0])
        if function_name in NAN_AVOIDING_MIN_MAX_FUNCTIONS and len(args) == 2:
            return self.emit_nan_avoiding_min_max(function_name, args)
        return None

    # -- shapes ------------------------------------------------------------------------

    def _float_shape(self, value) -> tuple[str, int | None] | None:
        value_type = self.ensure_registered_type(value.type)
        component = self.scalar_or_vector_component_type(value_type.type)
        if component not in {"float", "double"}:
            return None
        vector_info = self.vector_component_type_and_count(value_type.type.base_type)
        return component, (None if vector_info is None else vector_info[1])

    def _shaped_type(self, component_name: str, count: int | None):
        component = self.register_primitive_type(component_name)
        if count is None:
            return component
        return self.register_vector_type(component, count)

    def _shaped_constant(self, value, component_name: str, count: int | None):
        component = self.register_primitive_type(component_name)
        constant = self.register_constant(value, component)
        if count is None:
            return constant
        return self.splat_scalar_to_vector(
            constant, self.register_vector_type(component, count)
        )

    def _emit_typed(self, opcode: str, result_type, *operands):
        id_value = self.get_id()
        operand_text = " ".join(f"%{operand.id}" for operand in operands)
        self.emit(f"%{id_value} = {opcode} %{result_type.id} {operand_text}")
        result = self.make_spirv_value(id_value, result_type)
        return result

    def make_spirv_value(self, id_value: int, result_type):
        from .SPIRV_codegen import SpirvId

        self.value_types[id_value] = result_type
        return SpirvId(id_value, result_type.type)

    def _per_component(self, value, count: int | None, scalar_function, result_name):
        """Apply ``scalar_function`` to each component of a double vector."""
        if count is None:
            return scalar_function(value)
        element_type = self.register_primitive_type("double")
        parts = [
            scalar_function(self.composite_extract(value, element_type, index))
            for index in range(count)
        ]
        return self.composite_construct(self._shaped_type(result_name, count), parts)

    # -- classification ----------------------------------------------------------------

    def emit_float_classification(self, function_name: str, value):
        shape = self._float_shape(value)
        if shape is None:
            return None
        component, count = shape
        if component == "double":
            return self._per_component(
                value,
                count,
                lambda scalar: self._classify_float64(function_name, scalar),
                "bool",
            )
        return self._classify_float32(function_name, value, count)

    def _classify_float32(self, function_name: str, value, count: int | None):
        uint_type = self._shaped_type("uint", count)
        bool_type = self._shaped_type("bool", count)
        bits = self._emit_typed("OpBitcast", uint_type, value)
        if function_name == "isfinite":
            exponent = self._emit_typed(
                "OpBitwiseAnd",
                uint_type,
                bits,
                self._shaped_constant(FLOAT32_EXPONENT_MASK, "uint", count),
            )
            return self._emit_typed(
                "OpINotEqual",
                bool_type,
                exponent,
                self._shaped_constant(FLOAT32_EXPONENT_MASK, "uint", count),
            )
        magnitude = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            bits,
            self._shaped_constant(FLOAT32_MAGNITUDE_MASK, "uint", count),
        )
        infinity = self._shaped_constant(FLOAT32_EXPONENT_MASK, "uint", count)
        opcode = "OpUGreaterThan" if function_name == "isnan" else "OpIEqual"
        return self._emit_typed(opcode, bool_type, magnitude, infinity)

    def _float64_words(self, value):
        uint_type = self.register_primitive_type("uint")
        words_type = self.register_vector_type(uint_type, 2)
        words = self._emit_typed("OpBitcast", words_type, value)
        low = self.composite_extract(words, uint_type, 0)
        high = self.composite_extract(words, uint_type, 1)
        return words_type, low, high

    def _classify_float64(self, function_name: str, value):
        uint_type = self.register_primitive_type("uint")
        bool_type = self.register_primitive_type("bool")
        _, low, high = self._float64_words(value)
        exponent_mask = self.register_constant(FLOAT64_HIGH_EXPONENT_MASK, uint_type)
        if function_name == "isfinite":
            exponent = self._emit_typed("OpBitwiseAnd", uint_type, high, exponent_mask)
            return self._emit_typed("OpINotEqual", bool_type, exponent, exponent_mask)
        magnitude = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            high,
            self.register_constant(FLOAT32_MAGNITUDE_MASK, uint_type),
        )
        zero = self.register_constant(0, uint_type)
        at_infinity = self._emit_typed("OpIEqual", bool_type, magnitude, exponent_mask)
        low_zero = self._emit_typed("OpIEqual", bool_type, low, zero)
        if function_name == "isinf":
            return self._emit_typed("OpLogicalAnd", bool_type, at_infinity, low_zero)
        above_infinity = self._emit_typed(
            "OpUGreaterThan", bool_type, magnitude, exponent_mask
        )
        low_nonzero = self._emit_typed("OpLogicalNot", bool_type, low_zero)
        quiet_payload = self._emit_typed(
            "OpLogicalAnd", bool_type, at_infinity, low_nonzero
        )
        return self._emit_typed("OpLogicalOr", bool_type, above_infinity, quiet_payload)

    # -- fmin / fmax -------------------------------------------------------------------

    def emit_nan_avoiding_min_max(self, function_name: str, args: list):
        float_shapes = [self._float_shape(arg) for arg in args]
        if all(shape is None for shape in float_shapes):
            return None
        result_type = self.ensure_registered_type(
            args[0].type if float_shapes[0] is not None else args[1].type
        )
        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None or any(
            self._float_shape(operand) is None for operand in operands
        ):
            return None
        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')
        left, right = operands
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{NAN_AVOIDING_MIN_MAX_FUNCTIONS[function_name]} %{left.id} %{right.id}"
        )
        ordered = self.make_spirv_value(id_value, result_type)
        left_nan = self.emit_float_classification("isnan", left)
        right_nan = self.emit_float_classification("isnan", right)
        # C99 7.12.12.2/3: if exactly one operand is NaN, return the other one.
        chosen = self._emit_typed("OpSelect", result_type, right_nan, left, ordered)
        return self._emit_typed("OpSelect", result_type, left_nan, right, chosen)

    # -- truncated remainder -----------------------------------------------------------

    def emit_truncated_float_remainder(self, result_type, left, right):
        """Emit OpFRem and give the result the dividend's sign (C99 F.9.7.1)."""
        remainder = self._emit_typed("OpFRem", result_type, left, right)
        self.decorate_no_contraction_result(remainder.id, "OpFRem", result_type)
        shape = self._float_shape(remainder)
        if shape is None:
            return remainder
        component, count = shape
        if component == "double":
            if count is None:
                return self._copy_float64_sign(remainder, left)
            element_type = self.register_primitive_type("double")
            parts = [
                self._copy_float64_sign(
                    self.composite_extract(remainder, element_type, index),
                    self.composite_extract(left, element_type, index),
                )
                for index in range(count)
            ]
            return self.composite_construct(result_type, parts)
        uint_type = self._shaped_type("uint", count)
        magnitude = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            self._emit_typed("OpBitcast", uint_type, remainder),
            self._shaped_constant(FLOAT32_MAGNITUDE_MASK, "uint", count),
        )
        sign = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            self._emit_typed("OpBitcast", uint_type, left),
            self._shaped_constant(FLOAT32_SIGN_MASK, "uint", count),
        )
        signed = self._emit_typed("OpBitwiseOr", uint_type, magnitude, sign)
        return self._emit_typed("OpBitcast", result_type, signed)

    def _copy_float64_sign(self, remainder, dividend):
        uint_type = self.register_primitive_type("uint")
        double_type = self.register_primitive_type("double")
        words_type, low, high = self._float64_words(remainder)
        _, _, dividend_high = self._float64_words(dividend)
        magnitude = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            high,
            self.register_constant(FLOAT32_MAGNITUDE_MASK, uint_type),
        )
        sign = self._emit_typed(
            "OpBitwiseAnd",
            uint_type,
            dividend_high,
            self.register_constant(FLOAT32_SIGN_MASK, uint_type),
        )
        signed_high = self._emit_typed("OpBitwiseOr", uint_type, magnitude, sign)
        words = self.composite_construct(words_type, [low, signed_high])
        return self._emit_typed("OpBitcast", double_type, words)
