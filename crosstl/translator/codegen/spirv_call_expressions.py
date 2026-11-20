"""Function-call expressions of the SPIR-V generator: method and resource calls, user functions, built-ins and their argument lowering."""

from ..unhandled import UNHANDLED
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_definition_name,
    generic_function_value_arguments,
)
from .spirv_call_lowering import SpirvCallLoweringMixin
from .spirv_values import (
    INDIRECT_CALL_DETAIL,
    SPIRVBuiltinLoweringError,
    SpirvId,
)


class SpirvCallExpressionMixin(SpirvCallLoweringMixin):
    """Function-call member of ``VulkanSPIRVCodeGen.process_expression``."""

    def spirv_function_call_expression(
        self,
        expr,
    ):
        """Lower a function call."""
        handled = self.spirv_buffer_method_call_expression(expr)
        if handled is not UNHANDLED:
            return handled

        callee_expr = getattr(expr, "function", getattr(expr, "name", None))
        callee_name = None
        if hasattr(callee_expr, "name"):
            callee_name = callee_expr.name
        elif isinstance(callee_expr, str):
            callee_name = callee_expr

        handled = self.spirv_dedicated_callee_expression(expr, callee_name)
        if handled is not UNHANDLED:
            return handled

        inline_storage_buffer_function = self.resolve_inline_storage_buffer_function(
            callee_name, expr.args, call_node=expr
        )
        if inline_storage_buffer_function is not None:
            return self.inline_storage_buffer_function_call(
                inline_storage_buffer_function, expr.args, call_node=expr
            )

        generic_function_definition = self.generic_function_definitions.get(
            generic_function_definition_name(self, callee_name)
        )
        if (
            generic_function_definition is not None
            and not self.has_matching_concrete_function_candidate(
                callee_name, expr.args
            )
        ):
            specialized_callee_name = generic_function_call_name(
                self, callee_name, expr.args
            )
            if specialized_callee_name is None:
                if isinstance(generic_function_definition, list):
                    generic_function_definition = generic_function_definition[0]
                raise self.unsupported_generic_function_error(
                    generic_function_definition, call_node=expr
                )
            call_args = generic_function_value_arguments(
                self,
                callee_name,
                expr.args,
            )
            callee_name = specialized_callee_name
        else:
            call_args = list(expr.args)

        args, has_errors, selected_function_reference = (
            self.spirv_lowered_call_arguments(call_args, callee_expr, callee_name)
        )

        if has_errors and callee_name == "vec2":
            float_type = self.register_primitive_type("float")
            vector_type = self.register_vector_type(float_type, 2)
            id_value = self.get_id()

            while len(args) < 2:
                args.append(self.register_constant(0.0, float_type))

            arg_list = " ".join([f"%{arg.id}" for arg in args[:2]])
            self.emit(
                f"%{id_value} = OpCompositeConstruct %{vector_type.id} {arg_list}"
            )
            return SpirvId(id_value, vector_type.type)

        if callee_name is None:
            raise SPIRVBuiltinLoweringError(
                "<expression>",
                INDIRECT_CALL_DETAIL,
                reason="indirect-call",
                source_location=getattr(expr, "source_location", None),
            )

        if not self.validate_function_image_access_arguments(callee_name, expr.args):
            result_type = None
            function_signature = self.resolve_function_signature(callee_name)
            if function_signature is not None:
                result_type = function_signature[0]
            if result_type is None or result_type.type.base_type == "void":
                return None
            return self.default_value_for_type(result_type)

        if self.is_resource_type_name(callee_name):
            constructed_resource = self.call_resource_constructor(callee_name, args)
            if constructed_resource is not None:
                return constructed_resource

        if (
            callee_name in self.resource_function_names()
            and not self.has_function_reference(callee_name)
        ):
            return self.call_resource_function(callee_name, args)

        return self.call_function(
            callee_name,
            args,
            function_reference=selected_function_reference,
        )
