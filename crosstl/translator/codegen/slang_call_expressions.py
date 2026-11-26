"""Slang rendering of CrossGL function-call expressions."""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    UnaryOpNode,
)
from .enum_utils import (
    generate_enum_constructor_call,
)
from .generic_function_utils import (
    generate_numeric_trait_method_call,
    generate_static_generic_numeric_call,
    generic_function_call_name,
    reject_unresolved_generic_function_call,
)
from .slang_buffer_lengths import (
    slang_glsl_buffer_block_length,
    slang_structured_buffer_length,
)
from .slang_core_functions import generate_slang_core_function_call
from .slang_shader_clocks import generate_slang_shader_clock_call
from .slang_value_resizes import slang_resize_constructor
from .slang_wave_shuffles import (
    as_type_target,
)


class SlangCallExpressionMixin:
    """Function-call member of ``SlangCodeGen.generate_expression``."""

    def generate_slang_function_call_expression(
        self,
        node,
    ):
        """Render a function call."""
        func_expr = getattr(node, "function", None)
        if func_expr is None:
            func_expr = node.name
        clock_name = getattr(func_expr, "name", func_expr)
        if isinstance(clock_name, str) and clock_name not in self.user_function_names:
            clock_call = generate_slang_shader_clock_call(
                self, clock_name, getattr(node, "args", None)
            )
            if clock_call is not None:
                return clock_call
        ray_query_call = self.slang_ray_query_call_parts(node)
        if ray_query_call is not None:
            operation, query_expr, args = ray_query_call
            return self.generate_slang_ray_query_expression(operation, query_expr, args)
        ray_tracing_call = self.slang_ray_tracing_call_parts(node)
        if ray_tracing_call is not None:
            operation, args = ray_tracing_call
            return self.generate_slang_ray_tracing_call_expression(operation, args)
        bit_cast_type = as_type_target(getattr(func_expr, "name", func_expr))
        call_arguments = getattr(node, "arguments", getattr(node, "args", [])) or []
        if bit_cast_type is not None and len(call_arguments) == 1:
            return (
                f"bit_cast<{self.convert_type(bit_cast_type)}>"
                f"({self.generate_expression(call_arguments[0])})"
            )
        if hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            callee = func_expr.name
        elif isinstance(func_expr, str):
            callee = func_expr
        else:
            callee = self.generate_expression(func_expr)
        if (
            callee == "select"
            and callee not in self.user_function_names
            and len(node.args) == 3
            and all(
                self.is_matrix_value_type(self.expression_result_type(argument))
                for argument in node.args[:2]
            )
        ):
            false_value, true_value, condition = node.args
            condition_is_matrix_comparison = (
                isinstance(condition, BinaryOpNode)
                and str(getattr(condition, "op", getattr(condition, "operator", "")))
                in {"<", ">", "<=", ">=", "==", "!="}
                and any(
                    self.is_matrix_value_type(self.expression_result_type(operand))
                    for operand in (condition.left, condition.right)
                )
            )
            if (
                self.convert_type(self.expression_result_type(condition)) != "bool"
                or condition_is_matrix_comparison
            ):
                return self.refuse_lowering(
                    "matrix select condition must be a scalar bool",
                    self.generate_expression(false_value),
                )
            if any(
                any(
                    isinstance(item, (FunctionCallNode, AssignmentNode))
                    or (
                        isinstance(item, UnaryOpNode)
                        and str(getattr(item, "op", "")) in {"++", "--"}
                    )
                    for item in self.walk_ast(value)
                )
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
        numeric_trait_call = generate_numeric_trait_method_call(
            self,
            func_expr,
            node.args,
        )
        if numeric_trait_call is not None:
            return numeric_trait_call
        static_generic_call = generate_static_generic_numeric_call(self, callee)
        if static_generic_call is not None:
            return static_generic_call
        specialized_callee = generic_function_call_name(self, callee, node.args)
        if specialized_callee is not None:
            callee = specialized_callee
        else:
            reject_unresolved_generic_function_call(self, callee, "Slang")
        synchronization_call = self.generate_slang_synchronization_call(
            callee, node.args, statement_context=False
        )
        if synchronization_call is not None:
            return synchronization_call
        if callee not in self.user_function_names:
            glsl_block_atomic_call = self.generate_glsl_buffer_block_atomic_call(
                callee, node.args
            )
            if glsl_block_atomic_call is not None:
                return glsl_block_atomic_call
            resource_call = self.generate_resource_call(
                callee,
                node.args,
                statement_context=self.is_direct_statement_expression(node),
            )
            if resource_call is not None:
                return resource_call
        if isinstance(func_expr, MemberAccessNode):
            geometry_stream_call = self.generate_slang_geometry_stream_member_call(
                func_expr,
                node.args,
                statement_context=self.is_direct_statement_expression(node),
            )
            if geometry_stream_call is not None:
                return geometry_stream_call
            block_length = slang_glsl_buffer_block_length(self, func_expr, node.args)
            if block_length is not None:
                return block_length
            buffer_length = slang_structured_buffer_length(self, func_expr, node.args)
            if buffer_length is not None:
                return buffer_length
            resource_member_call = self.generate_resource_member_call(
                func_expr,
                node.args,
                statement_context=self.is_direct_statement_expression(node),
            )
            if resource_member_call is not None:
                return resource_member_call
        if callee == "mix" and callee not in self.user_function_names:
            bool_mix = self.generate_bool_mix_call(node.args)
            if bool_mix is not None:
                return bool_mix
        if (
            callee == "mod"
            and len(node.args) == 2
            and callee not in self.user_function_names
        ):
            left = self.generate_expression(node.args[0])
            right = self.generate_expression(node.args[1])
            return self.generate_slang_mod_expression(left, right)
        if (
            callee == "select"
            and len(node.args) == 3
            and callee not in self.user_function_names
        ):
            # CrossGL select(a, b, c) is MSL's (MSL 6.4): c ? b : a per component. Slang's select is HLSL's
            # select(cond, trueValue, falseValue). Both evaluate every argument.
            false_value, true_value, condition = node.args
            args = ", ".join(
                self.generate_expression_without_expected(argument)
                for argument in (condition, true_value, false_value)
            )
            return f"select({args})"
        if callee == "inverse" and callee not in self.user_function_names:
            if len(node.args) != 1:
                fallback_type = self.current_expression_expected_type or "float"
                return self.refuse_lowering(
                    "Slang inverse: requires one argument",
                    self.zero_value_for_type(fallback_type),
                )
            return self.generate_slang_matrix_inverse_call(node.args[0])
        if (
            callee in {"matrixCompMult", "outerProduct"}
            and callee not in self.user_function_names
        ):
            return self.generate_slang_matrix_builtin_call(callee, node.args)
        if callee == "lambda" and callee not in self.user_function_names:
            lambda_expr = self.generate_lambda_expression(node.args)
            if lambda_expr is not None:
                return lambda_expr
            # Written as a call, ``lambda(...)`` would name a function no Slang module declares.
            return self.refuse_lowering(
                "lambda: Slang lambda expressions take typed parameters "
                "(`(int x) => x + 1`); this lambda's parameters or body "
                "have no Slang spelling",
                "lambda()",
            )
        enum_constructor = generate_enum_constructor_call(self, callee, node.args)
        if enum_constructor is not None:
            return enum_constructor
        if (
            isinstance(callee, str)
            and callee in self.user_struct_names
            and callee not in self.user_function_names
        ):
            args = self.generate_struct_constructor_arguments(callee, node.args)
            return f"{self.convert_type(callee)}({args})"
        bitcast_callee = self.slang_bitcast_intrinsic_callee(callee)
        if isinstance(callee, str) and callee in self.user_function_names:
            args = self.generate_user_function_arguments(callee, node.args)
        elif bitcast_callee is not None:
            args = ", ".join(
                [self.generate_expression_without_expected(arg) for arg in node.args]
            )
        else:
            # The call's own expected type says nothing about its operands.
            args = ", ".join(
                [self.generate_expression_without_expected(arg) for arg in node.args]
            )
        if isinstance(callee, str) and callee not in self.user_function_names:
            resize = slang_resize_constructor(self, callee, node.args)
            if resize is not None:
                return resize
            matrix_constructor = self.hlsl_matrix_orientation.constructor(
                callee,
                node.args,
                lambda arg: self.generate_expression_with_expected(arg, None),
            )
            if matrix_constructor is not None:
                return matrix_constructor
        callee = self.convert_type(callee)
        if (
            callee == "saturate"
            and len(node.args) == 1
            and callee not in self.user_function_names
        ):
            return f"clamp({args}, 0.0, 1.0)"
        if (
            callee == "atan"
            and len(node.args) == 2
            and callee not in self.user_function_names
        ):
            callee = "atan2"
        if callee not in self.user_function_names:
            bit_query = self.generate_slang_bit_query_call(callee, node.args, args)
            if bit_query is not None:
                return bit_query
            core_call = generate_slang_core_function_call(self, callee, node.args)
            if core_call is not None:
                return core_call
            callee = self.slang_callee_name(callee)
        return f"{callee}({args})"
