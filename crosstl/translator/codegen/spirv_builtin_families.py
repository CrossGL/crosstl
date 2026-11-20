"""Built-in function families of the SPIR-V generator: calls handed to the packing, bitfield, synchronization, bitcast, derivative, bfloat16, float-semantics, polyfill and matrix helpers; operand-count checks and the scalar built-ins lowered without GLSL.std.450; the geometric and interpolation built-ins; vector, struct and matrix constructors; and the remaining GLSL.std.450 extended instructions."""

import re
from typing import (
    List,
)

from ..unhandled import (
    UNHANDLED,
)
from .bitfield_builtins import (
    BITFIELD_ARITIES,
    spirv_bitfield_call,
)
from .pack_builtins import (
    PACK_FUNCTIONS,
    spirv_pack_call,
)
from .spirv_exponent_builtins import (
    spirv_ldexp_call,
)
from .spirv_matrix_constructors import (
    one_argument_matrix_columns,
)
from .spirv_short_circuit import (
    lower_logical_intrinsic,
)
from .spirv_values import (
    CALLEE_IDENTIFIER,
    INDIRECT_CALL_DETAIL,
    SpirvId,
)

# The GLSL.std.450 instruction of each built-in the extended-instruction lowering accepts.
GLSL_STD450_INSTRUCTIONS = {
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "atan2": "Atan2",
    "asin": "Asin",
    "acos": "Acos",
    "atan": "Atan",
    "sinh": "Sinh",
    "cosh": "Cosh",
    "tanh": "Tanh",
    "asinh": "Asinh",
    "acosh": "Acosh",
    "atanh": "Atanh",
    "exp": "Exp",
    "log": "Log",
    "exp2": "Exp2",
    "log2": "Log2",
    "sqrt": "Sqrt",
    "inversesqrt": "InverseSqrt",
    "abs": "FAbs",
    # Metal and C spell the floating-point absolute value fabs.
    "fabs": "FAbs",
    "floor": "Floor",
    "ceil": "Ceil",
    "fract": "Fract",
    "trunc": "Trunc",
    "round": "Round",
    "roundEven": "RoundEven",
    "degrees": "Degrees",
    "radians": "Radians",
    "length": "Length",
    "distance": "Distance",
    "cross": "Cross",
    "normalize": "Normalize",
    "reflect": "Reflect",
    "refract": "Refract",
}

# Built-ins whose GLSL.std.450 instruction takes one operand.
GLSL_STD450_UNARY_FUNCTIONS = frozenset(
    {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "exp2",
        "log2",
        "sqrt",
        "inversesqrt",
        "abs",
        "floor",
        "ceil",
        "fract",
        "trunc",
        "round",
        "roundEven",
        "degrees",
        "radians",
        "length",
        "normalize",
    }
)

# The operand count the lowering checks for each GLSL.std.450 built-in that has a fixed one.
GLSL_STD450_OPERAND_COUNTS = {name: 1 for name in sorted(GLSL_STD450_UNARY_FUNCTIONS)}
GLSL_STD450_OPERAND_COUNTS.update(
    {
        "atan2": 2,
        "distance": 2,
        "cross": 2,
        "reflect": 2,
        "refract": 3,
    }
)


class SpirvBuiltinFamilyMixin:
    """Built-in function families of ``VulkanSPIRVCodeGen.call_builtin_function``."""

    def spirv_delegated_builtin_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Hand packing, bitfield, synchronization, bitcast, derivative and other helper-owned built-ins to their helpers."""
        if function_name in PACK_FUNCTIONS:
            return spirv_pack_call(self, function_name, args)
        if function_name in BITFIELD_ARITIES:
            return spirv_bitfield_call(self, function_name, args)
        if function_name == "ldexp":
            return spirv_ldexp_call(self, args)
        synchronization_call = self.call_synchronization_function(function_name, args)
        if synchronization_call is not None:
            return synchronization_call

        bfloat16_bitcast = self.call_bfloat16_bitcast_function(function_name, args)
        if bfloat16_bitcast is not None:
            return bfloat16_bitcast

        explicit_bitcast = self.call_explicit_bitcast_function(function_name, args)
        if explicit_bitcast is not None:
            return explicit_bitcast

        scalar_constructor = self.call_scalar_constructor(function_name, args)
        if scalar_constructor is not None:
            return scalar_constructor

        derivative_call = self.call_derivative_function(function_name, args)
        if derivative_call is not None:
            return derivative_call

        bitcast_call = self.call_bitcast_builtin_function(function_name, args)
        if bitcast_call is not None:
            return bitcast_call

        integer_bit_call = self.call_integer_bit_builtin_function(function_name, args)
        if integer_bit_call is not None:
            return integer_bit_call

        bfloat16_math_call = self.call_bfloat16_math_function(function_name, args)
        if bfloat16_math_call is not None:
            return bfloat16_math_call

        float_semantics_call = self.call_float_semantics_function(function_name, args)
        if float_semantics_call is not None:
            return float_semantics_call

        math_polyfill_call = self.call_math_polyfill_function(function_name, args)
        if math_polyfill_call is not None:
            return math_polyfill_call

        matrix_builtin_call = self.call_matrix_builtin_function(function_name, args)
        if matrix_builtin_call is not None:
            return matrix_builtin_call
        return UNHANDLED

    def spirv_scalar_builtin_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Check operand counts and emit boolean mix, saturate, sign and absolute-value built-ins."""
        exact_operand_counts = {
            "min": 2,
            "max": 2,
            "clamp": 3,
            "mix": 3,
            "pow": 2,
            "fma": 3,
            "faceforward": 3,
            "refract": 3,
            "length": 1,
            "distance": 2,
            "normalize": 1,
            "reflect": 2,
            "step": 2,
            "smoothstep": 3,
        }
        expected_operand_count = exact_operand_counts.get(function_name)
        if expected_operand_count is not None and len(args) != expected_operand_count:
            operand_label = "operand" if expected_operand_count == 1 else "operands"
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires "
                f"{expected_operand_count} {operand_label}",
            )

        if function_name == "mix" and len(args) == 3:
            bool_mix = self.call_bool_mix_function(args)
            if bool_mix is not None:
                return bool_mix

        if self.glsl_std450_id is None:
            self.glsl_std450_id = self.get_id()
            self.emit(f'%{self.glsl_std450_id} = OpExtInstImport "GLSL.std.450"')

        if function_name == "saturate" and len(args) == 1:
            saturated = self.call_saturate_function(args[0])
            if saturated is not None:
                return saturated

        if function_name == "sign" and len(args) == 1:
            signed = self.call_sign_function(args[0])
            if signed is not None:
                return signed

        if function_name == "abs" and len(args) == 1:
            absolute = self.call_abs_function(args[0])
            if absolute is not None:
                return absolute
            raise self.builtin_lowering_error(
                function_name,
                "abs requires floating-point or signed integer "
                "scalar or vector operand",
            )

        # C/MSL fabs (C99 7.12.7.2, MSL 6.5) is the floating-point absolute value.
        if function_name == "fabs" and len(args) == 1:
            if self._float_shape(args[0]) is not None:
                return self.call_abs_function(args[0])
            raise self.builtin_lowering_error(
                function_name,
                "fabs requires a floating-point scalar or vector operand",
            )
        return UNHANDLED

    def spirv_geometric_builtin_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit logical, select, power, fused multiply-add, geometric, step and min/max/clamp/mix built-ins."""
        if function_name in {"and", "or"} and len(args) == 2:
            return lower_logical_intrinsic(self, function_name, args)

        # MSL select(a, b, c) (MSL 6.4) yields c ? b : a per component, like mix(a, b, bvec).
        if function_name == "select" and len(args) == 3:
            selected = self.call_bool_mix_function(args)
            if selected is not None:
                return selected
            raise self.builtin_lowering_error(
                function_name,
                "select requires matching value operands and a bool condition "
                "that is scalar or matches their component count",
            )

        if function_name == "pow" and len(args) == 2:
            power = self.call_pow_function(args)
            if power is not None:
                return power
            raise self.builtin_lowering_error(
                function_name,
                "pow requires compatible 32-bit floating-point "
                "scalar or vector operands",
            )

        if function_name == "fma" and len(args) == 3:
            fused = self.call_fma_function(args)
            if fused is not None:
                return fused
            raise self.builtin_lowering_error(
                function_name,
                "fma requires compatible floating-point scalar " "or vector operands",
            )

        if function_name == "faceforward" and len(args) == 3:
            faced = self.call_faceforward_function(args)
            if faced is not None:
                return faced
            raise self.builtin_lowering_error(
                function_name,
                "faceforward requires compatible floating-point "
                "scalar or vector operands",
            )

        if function_name == "refract" and len(args) == 3:
            refracted = self.call_refract_function(args)
            if refracted is not None:
                return refracted
            raise self.builtin_lowering_error(
                function_name,
                "refract requires matching floating-point incident "
                "and normal operands plus scalar eta",
            )

        if function_name == "length" and len(args) == 1:
            measured = self.call_length_function(args[0])
            if measured is not None:
                return measured
            raise self.builtin_lowering_error(
                function_name,
                "length requires floating-point scalar or vector operand",
            )

        if function_name == "distance" and len(args) == 2:
            measured = self.call_distance_function(args)
            if measured is not None:
                return measured
            raise self.builtin_lowering_error(
                function_name,
                "distance requires matching floating-point scalar "
                "or vector operands",
            )

        if function_name == "normalize" and len(args) == 1:
            normalized = self.call_normalize_function(args[0])
            if normalized is not None:
                return normalized
            raise self.builtin_lowering_error(
                function_name,
                "normalize requires floating-point scalar or vector operand",
            )

        if function_name == "reflect" and len(args) == 2:
            reflected = self.call_reflect_function(args)
            if reflected is not None:
                return reflected
            raise self.builtin_lowering_error(
                function_name,
                "reflect requires matching floating-point scalar " "or vector operands",
            )

        if function_name == "step" and len(args) == 2:
            stepped = self.call_step_smoothstep_function("Step", args, 1)
            if stepped is not None:
                return stepped
            raise self.builtin_lowering_error(
                function_name,
                "step requires compatible floating-point scalar " "or vector operands",
            )

        if function_name == "smoothstep" and len(args) == 3:
            smoothed = self.call_step_smoothstep_function("SmoothStep", args, 2)
            if smoothed is not None:
                return smoothed
            raise self.builtin_lowering_error(
                function_name,
                "smoothstep requires compatible floating-point "
                "scalar or vector operands",
            )

        if (
            function_name in {"min", "max"}
            and len(args) == 2
            or function_name == "clamp"
            and len(args) == 3
        ):
            min_max_clamp = self.call_min_max_clamp_function(function_name, args)
            if min_max_clamp is not None:
                return min_max_clamp
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires compatible scalar or "
                "vector numeric operands",
            )

        if function_name == "mix" and len(args) == 3:
            numeric_mix = self.call_numeric_mix_function(args)
            if numeric_mix is not None:
                return numeric_mix
            raise self.builtin_lowering_error(
                function_name,
                "mix requires compatible value operands and selector",
            )
        return UNHANDLED

    def spirv_vector_and_struct_constructor_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit vector and struct constructor calls."""
        vector_info = self.vector_component_type_and_count(function_name)
        if vector_info:
            component_type_name, component_count = vector_info
            component_type = self.register_primitive_type(component_type_name)
            vector_type = self.register_vector_type(component_type, component_count)
            if not args:
                if component_type_name == "bool":
                    zero_value = False
                    one_value = True
                elif component_type_name in {"int", "uint"}:
                    zero_value = 0
                    one_value = 1
                elif component_type_name == self.BFLOAT16_TYPE_NAME:
                    zero_value = 0
                    one_value = self.bfloat16_bits_for_float(1.0)
                else:
                    zero_value = 0.0
                    one_value = 1.0

                # Preserve old defaults: first component one, rest zero.
                component_zero = self.register_constant(zero_value, component_type)
                component_one = self.register_constant(one_value, component_type)

                default_args = [component_zero] * component_count
                if component_count > 0:
                    default_args[0] = component_one

                constructor_args = default_args
            else:
                constructor_args = self.flatten_vector_constructor_args(
                    function_name, args, component_type, component_count
                )

            return self.composite_construct(vector_type, constructor_args)

        if function_name in self.struct_types:
            struct_type = self.struct_types[function_name]
            if len(args) == 1:
                converted = self.convert_value_to_type(args[0], struct_type)
                if self.value_has_type(converted, struct_type):
                    return converted
            constructor_args = self.struct_constructor_args(function_name, args)
            return self.composite_construct(struct_type, constructor_args)
        return UNHANDLED

    def spirv_matrix_constructor_call(
        self,
        function_name: str,
        args: List[SpirvId],
        matrix_function_name,
    ):
        """Emit a matrix constructor from columns, scalars or a single diagonal value."""
        match = re.fullmatch(r"(d)?mat([234])(?:x([234]))?", matrix_function_name)
        is_double, cols, rows = match.groups()
        cols = int(cols)
        rows = int(rows or cols)

        component_type = self.register_primitive_type(
            "double" if is_double else "float"
        )
        vector_type = self.register_vector_type(component_type, rows)
        matrix_type = self.register_matrix_type(vector_type, cols)

        if not args:
            zero_value = self.register_constant(0.0, component_type)
            one_value = self.register_constant(1.0, component_type)

            col_vectors = []
            for col in range(cols):
                col_components = []
                for row in range(rows):
                    if col == row:
                        col_components.append(one_value)
                    else:
                        col_components.append(zero_value)

                col_vectors.append(
                    self.composite_construct(vector_type, col_components)
                )
        elif len(args) == 1 and (
            col_vectors := one_argument_matrix_columns(
                self,
                function_name,
                args[0],
                component_type,
                vector_type,
                cols,
                rows,
            )
        ):
            pass
        else:
            components = self.flatten_matrix_constructor_components(
                function_name, args, component_type, cols * rows
            )
            col_vectors = []
            for col in range(cols):
                start = col * rows
                col_vectors.append(
                    self.composite_construct(
                        vector_type, components[start : start + rows]
                    )
                )

        return self.composite_construct(matrix_type, col_vectors)

    def spirv_glsl_std450_call(
        self,
        function_name: str,
        args: List[SpirvId],
    ):
        """Emit the remaining built-ins as GLSL.std.450 extended instructions."""
        float_type = self.primitive_types["float"]
        std450_function_name = (
            "atan2" if function_name == "atan" and len(args) == 2 else function_name
        )
        if std450_function_name not in GLSL_STD450_INSTRUCTIONS:
            if not CALLEE_IDENTIFIER.fullmatch(function_name):
                raise self.builtin_lowering_error(
                    function_name, INDIRECT_CALL_DETAIL, reason="indirect-call"
                )
            raise self.builtin_lowering_error(
                function_name,
                "it is neither a declared function nor a built-in with a "
                "SPIR-V lowering",
                reason="unknown-function",
            )

        expected_operand_count = GLSL_STD450_OPERAND_COUNTS.get(std450_function_name)
        if expected_operand_count is not None and len(args) != expected_operand_count:
            operand_label = "operand" if expected_operand_count == 1 else "operands"
            raise self.builtin_lowering_error(
                function_name,
                f"{function_name} requires "
                f"{expected_operand_count} {operand_label}",
            )

        vector_math_diagnostic = self.vector_math_operand_diagnostic(
            std450_function_name, args
        )
        if vector_math_diagnostic is not None:
            raise self.builtin_lowering_error(function_name, vector_math_diagnostic)

        result_type = float_type.type

        if args:
            if std450_function_name in [
                "sin",
                "cos",
                "tan",
                "atan2",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "tanh",
                "asinh",
                "acosh",
                "atanh",
                "exp",
                "log",
                "exp2",
                "log2",
                "sqrt",
                "inversesqrt",
                "abs",
                "floor",
                "ceil",
                "fract",
                "trunc",
                "round",
                "roundEven",
                "degrees",
                "radians",
            ]:
                # These functions return the same type as their first argument
                result_type = args[0].type
            elif std450_function_name in ["length", "distance"]:
                # These functions return the scalar component type.
                result_type = self.metric_result_type(args).type
            elif std450_function_name in ["normalize", "reflect", "refract"]:
                # These functions return the same vector type as their first argument
                result_type = args[0].type
            elif std450_function_name in ["cross"]:
                # cross product returns a vec3
                component_type_name = "float"
                vector_info = self.vector_component_type_and_count(
                    args[0].type.base_type
                )
                if vector_info is not None:
                    component_type_name = vector_info[0]
                component_type = self.register_primitive_type(component_type_name)
                vector_type = self.register_vector_type(component_type, 3)
                result_type = vector_type.type

        result_type_id = self.ensure_registered_type(result_type)
        return self.emit_glsl_std450_instruction(
            GLSL_STD450_INSTRUCTIONS[std450_function_name], result_type_id, args
        )
