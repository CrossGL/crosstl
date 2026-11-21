"""Steps of the SPIR-V generator's function-call lowering: ray-query, byte-address, length and structured-buffer method calls; callees with dedicated lowerings, such as fences, generic struct constructors, optional payloads, debug printf, shader clocks, residency queries, output-operand and extended-arithmetic built-ins, geometry and mesh output calls, enum paths and lambdas; and argument lowering."""

from ..ast import (
    ArrayLiteralNode,
)
from ..matrix_shape import (
    crossgl_matrix_shape,
    length_method_object,
)
from ..unhandled import (
    UNHANDLED,
)
from .extended_arithmetic_builtins import (
    EXTENDED_ARITHMETIC_FUNCTIONS,
    spirv_extended_arithmetic_call,
)
from .generic_function_utils import (
    reject_unresolved_generic_member_call,
)
from .spirv_byte_address_atomics import (
    process_byte_address_atomic_call,
)
from .spirv_debug_printf import (
    DEBUG_PRINTF_FUNCTIONS,
    spirv_debug_printf_call,
)
from .spirv_exponent_builtins import (
    OUTPUT_OPERAND_FUNCTIONS,
    spirv_output_operand_call,
)
from .spirv_fragment_interlock import INTERLOCK_CALLS, spirv_interlock_call
from .spirv_geometry_streams import (
    STREAM_FUNCTIONS,
)
from .spirv_shader_clock import (
    SPIRV_SHADER_CLOCKS,
    spirv_shader_clock_call,
)
from .spirv_texture_residency import (
    spirv_texture_residency_call,
    spirv_texture_resident_call,
)
from .spirv_values import (
    SpirvId,
)
from .texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
)


class SpirvCallLoweringMixin:
    """Steps of ``SpirvCallExpressionMixin.spirv_function_call_expression``."""

    def spirv_buffer_method_call_expression(
        self,
        expr,
    ):
        """Lower ray-query, byte-address, length and structured-buffer method calls."""
        reject_unresolved_generic_member_call(expr, "Vulkan SPIR-V")
        ray_query_call = self.ray_query_call_from_function_call(expr)
        if ray_query_call is not None:
            return self.process_ray_query_operation(ray_query_call)

        handled_byte_address_call, byte_address_result = (
            self.process_byte_address_buffer_method_call(expr)
        )
        if handled_byte_address_call:
            return byte_address_result
        handled_byte_address_call, byte_address_result = (
            process_byte_address_atomic_call(self, expr)
        )
        if handled_byte_address_call:
            return byte_address_result

        handled_length_call, length_result = self.process_block_member_length_call(expr)
        if handled_length_call:
            return length_result
        length_object = length_method_object(
            getattr(expr, "function", None), getattr(expr, "args", None) or []
        )
        matrix_shape = (
            crossgl_matrix_shape(self.expression_result_type(length_object))
            if length_object is not None
            else None
        )
        if matrix_shape is not None:
            # A matrix's length() is its column count (GLSL 4.60 section 5.10).
            return self.register_constant(
                matrix_shape[1], self.register_primitive_type("int")
            )

        handled_structured_buffer_call, structured_buffer_result = (
            self.process_structured_buffer_method_call(expr)
        )
        if handled_structured_buffer_call:
            return structured_buffer_result
        return UNHANDLED

    def spirv_dedicated_callee_expression(
        self,
        expr,
        callee_name,
    ):
        """Lower calls whose callee has a dedicated lowering."""
        if self.is_canonical_atomic_thread_fence_call(callee_name):
            return self.reject_atomic_thread_fence(expr)

        instance_constructor = self.process_generic_struct_constructor_call(
            callee_name, expr
        )
        if instance_constructor is not None:
            return instance_constructor

        if isinstance(callee_name, str) and callee_name.rsplit("::", 1)[-1] == "Some":
            payload_type = self.option_or_expected_payload_type_name()
            if payload_type is not None and not expr.args:
                return self.default_value_for_type(self.map_crossgl_type(payload_type))
            if payload_type is not None:
                payload_value = self.process_expression_with_expected_type(
                    expr.args[0],
                    payload_type,
                )
                if payload_value is None:
                    return None
                return self.convert_value_to_type(
                    payload_value,
                    self.map_crossgl_type(payload_type),
                )

        if isinstance(callee_name, str) and callee_name.lower() in {
            "nonuniformext",
            "nonuniformresourceindex",
        }:
            # HLSL's NonUniformResourceIndex and GLSL's nonuniformEXT both mark a divergent descriptor index.
            return self.process_non_uniform_function_call(callee_name, expr.args)

        if callee_name in DEBUG_PRINTF_FUNCTIONS and not self.has_function_reference(
            callee_name
        ):
            return spirv_debug_printf_call(self, callee_name, expr.args, SpirvId)

        if callee_name in SPIRV_SHADER_CLOCKS and not self.has_function_reference(
            callee_name
        ):
            return spirv_shader_clock_call(self, callee_name, list(expr.args), SpirvId)

        if callee_name == TEXTURE_RESIDENCY_OPERATION and not (
            self.has_function_reference(callee_name)
        ):
            return spirv_texture_residency_call(self, list(expr.args), expr)
        if callee_name == TEXTURE_RESIDENT_OPERATION and not (
            self.has_function_reference(callee_name)
        ):
            return spirv_texture_resident_call(self, list(expr.args), expr)

        if callee_name in OUTPUT_OPERAND_FUNCTIONS and not self.has_function_reference(
            callee_name
        ):
            return spirv_output_operand_call(self, callee_name, list(expr.args))

        if (
            callee_name in EXTENDED_ARITHMETIC_FUNCTIONS
            and not self.has_function_reference(callee_name)
        ):
            return spirv_extended_arithmetic_call(self, callee_name, list(expr.args))

        if callee_name in INTERLOCK_CALLS and not self.has_function_reference(
            callee_name
        ):
            return spirv_interlock_call(self, callee_name, list(expr.args))

        if callee_name in {"SetVertex", "SetPrimitive"}:
            return self.process_mesh_output_function_call(callee_name, expr.args)

        if callee_name in {"EmitVertex", "EndPrimitive", *STREAM_FUNCTIONS}:
            return self.process_geometry_stream_function_call(callee_name, expr.args)

        if (
            isinstance(callee_name, str)
            and self.enum_path_parts(callee_name) is not None
        ):
            enum_value = self.process_enum_variant_constructor(
                callee_name,
                list(getattr(expr, "args", []) or []),
            )
            if enum_value is not None:
                return enum_value

        if callee_name == "lambda":
            return self.unsupported_lambda_default_value("lambda expression")

        if callee_name == "buffer_dimensions":
            return self.process_buffer_dimensions_function_call(expr)

        if any(self.contains_lambda_expression(arg) for arg in expr.args):
            result_type = None
            function_signature = self.resolve_function_signature(callee_name)
            if function_signature is not None:
                result_type = function_signature[0]
            return self.unsupported_lambda_default_value(
                f"call to {callee_name or 'unknown callee'}",
                result_type,
            )
        return UNHANDLED

    def spirv_lowered_call_arguments(
        self,
        call_args,
        callee_expr,
        callee_name,
    ):
        """Lower the call's arguments, with contextual parameter types for array literals, and the stage objects the callee needs."""
        args = []
        has_errors = False
        skipped_arg_indices = self.skipped_function_parameter_indices(callee_name)
        mesh_output_arg_indices = self.resolve_function_mesh_output_parameter_indices(
            callee_name
        )
        contextual_parameter_types = {}
        ambiguous_contextual_arguments = set()
        selected_function_reference = None
        temporary_array_arguments = set()
        if any(isinstance(arg, ArrayLiteralNode) for arg in call_args):
            (
                contextual_parameter_types,
                ambiguous_contextual_arguments,
                selected_function_reference,
                temporary_array_arguments,
            ) = self.contextual_call_parameter_types(
                callee_name,
                call_args,
                skipped_arg_indices,
                mesh_output_arg_indices,
            )
        for arg_index, arg in enumerate(call_args):
            if arg_index in skipped_arg_indices:
                continue
            if arg_index in mesh_output_arg_indices:
                continue
            if self.expression_contains_empty_array_literal(arg):
                if arg_index in ambiguous_contextual_arguments:
                    raise self.unsupported_call_initializer_type_error(
                        callee_name, arg_index, arg, ambiguous=True
                    )
                if arg_index not in contextual_parameter_types:
                    raise self.unsupported_call_initializer_type_error(
                        callee_name, arg_index, arg, ambiguous=False
                    )
            arg_value = self.process_call_argument(
                callee_name,
                arg,
                arg_index,
                expected_type=contextual_parameter_types.get(arg_index),
                allow_array_temporary=arg_index in temporary_array_arguments,
            )
            if arg_value is None:
                self.refuse_lowering(
                    f"Failed to evaluate argument for {callee_name or callee_expr}"
                )
                has_errors = True
                float_type = self.register_primitive_type("float")
                arg_value = self.register_constant(0.0, float_type)
            args.append(arg_value)

        for stage_object_name in self.required_function_stage_object_argument_names(
            callee_name
        ):
            stage_object = self.local_variables.get(stage_object_name)
            if stage_object is None:
                self.refuse_lowering(
                    "Failed to evaluate stage object argument "
                    f"{stage_object_name} for {callee_name or callee_expr}"
                )
                has_errors = True
                continue
            args.append(stage_object)
        return args, has_errors, selected_function_reference
