"""Metal rendering of CrossGL function-call expressions for ``MetalCodeGen.generate_expression``."""

from ..unhandled import UNHANDLED
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generic_function_call_name,
    generic_function_value_arguments,
    reject_unresolved_generic_function_call,
)
from .metal_builtin_call_forms import MetalBuiltinCallFormMixin


class MetalCallExpressionMixin(MetalBuiltinCallFormMixin):
    """Function-call members of ``MetalCodeGen.generate_expression``."""

    def generate_metal_function_call_expression(
        self,
        expr,
    ):
        """Render a function-call expression."""
        option_payload = self.option_constructor_payload_expression(expr)
        if option_payload is not None:
            return option_payload

        func_expr = getattr(expr, "function", None)
        if func_expr is None:
            func_expr = expr.name
        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            expr.args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call

        func_name = None
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.generate_expression(func_expr)

        if (
            func_name == "select"
            and func_name not in self.user_function_names
            and len(expr.args) == 3
            and all(
                self.is_matrix_value_type(self.expression_result_type(argument))
                for argument in expr.args[:2]
            )
        ):
            false_value, true_value, condition = expr.args
            condition_type = self.map_type(self.expression_result_type(condition))
            if condition_type != "bool":
                return self.refuse_lowering(
                    "matrix select condition must be a scalar bool",
                    self.generate_expression(false_value),
                )
            if not all(
                self.metal_expression_is_side_effect_free(value)
                for value in (false_value, true_value)
            ):
                return self.refuse_lowering(
                    "matrix select evaluates both value arguments, but a "
                    "conditional expression would skip one side-effecting argument",
                    self.generate_expression(false_value),
                )
            return (
                f"({self.generate_expression(condition)} ? "
                f"{self.generate_expression(true_value)} : "
                f"{self.generate_expression(false_value)})"
            )

        handled = self.generate_metal_numeric_builtin_call(expr, func_expr, func_name)
        if handled is not UNHANDLED:
            return handled

        handled = self.generate_metal_resource_builtin_call(expr, func_name)
        if handled is not UNHANDLED:
            return handled
        specialized_func_name = generic_function_call_name(
            self,
            func_name,
            expr.args,
        )
        argument_func_name = func_name
        call_args = expr.args
        if specialized_func_name is not None:
            # Leading type arguments, as in bump(vec2, w, w), select the specialization; they are not passed.
            call_args = generic_function_value_arguments(self, func_name, expr.args)
            callee = specialized_func_name
            func_name = specialized_func_name
        else:
            reject_unresolved_generic_function_call(self, func_name, "Metal")
        handled = self.generate_metal_math_builtin_call(expr, func_name)
        if handled is not UNHANDLED:
            return handled
        handled = self.generate_metal_type_constructor_call(expr, func_name)
        if handled is not UNHANDLED:
            return handled
        readonly_raw_buffer_call = self.readonly_raw_buffer_call_diagnostic(
            argument_func_name, call_args
        )
        if readonly_raw_buffer_call is not None:
            return readonly_raw_buffer_call
        readonly_parameter_call = self.readonly_metal_parameter_call_diagnostic(
            argument_func_name, call_args
        )
        if readonly_parameter_call is not None:
            return readonly_parameter_call
        mesh_context_call = self.metal_mesh_dispatch_context_call_diagnostic(func_name)
        if mesh_context_call is not None:
            return mesh_context_call
        self.validate_address_space_call(argument_func_name, call_args)
        wave_lane_call = self.metal_wave_lane_helper_call_diagnostic(func_name)
        if wave_lane_call is not None:
            return wave_lane_call
        readonly_mesh_payload_call = self.readonly_metal_mesh_payload_call_diagnostic(
            argument_func_name, call_args
        )
        if readonly_mesh_payload_call is not None:
            return readonly_mesh_payload_call
        self.validate_function_resource_argument_types(func_name, call_args)
        self.validate_function_image_access_arguments(func_name, call_args)
        args = self.generate_function_call_arguments(argument_func_name, call_args)
        if self.function_call_matches_known_signature(func_name, call_args):
            args.extend(self.required_function_stage_output_argument_names(func_name))
            args.extend(
                self.required_function_stage_parameter_argument_names(func_name)
            )
            args.extend(
                self.cbuffer_parameter_name(cbuffer)
                for cbuffer in self.required_function_cbuffers(func_name)
            )
            args.extend(
                self.required_function_thread_private_global_argument_names(func_name)
            )
            args.extend(self.required_function_resource_argument_names(func_name))
            args.extend(self.required_metal_mesh_dispatch_context_arguments(func_name))
            wave_lane_args = self.required_metal_wave_lane_context_arguments(func_name)
            if wave_lane_args is not None:
                args.extend(wave_lane_args)
        args = ", ".join(args)
        if callee == func_name and (
            self.metal_precise_math_call(expr, func_name)
            or self.metal_safe_math_call(expr, func_name)
        ):
            return f"metal::precise::{func_name}({args})"
        return f"{callee}({args})"
