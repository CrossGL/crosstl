"""Renderers of the expression kinds that ``MetalToCrossGLConverter.generate_expression`` dispatches on: operators, calls, member and subscript access, casts, vector constructors and texture samples."""

from ....translator.math_namespaces import (
    metal_fast_math_operation,
    metal_precise_math_operation,
)
from ...comma_expressions import crossgl_comma_expression
from ..aggregate_scalar_conversions import (
    refuse_aggregate_builtin_operation,
    render_explicit_aggregate_scalar_conversion,
)
from ..implicit_constructor_conversions import convert_user_call_arguments
from ..instance_static_members import render_instance_static_struct_member
from ..matrix_columns import MetalMatrixColumnsError, matrix_columns_object
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    InitializerListNode,
    LambdaNode,
    TernaryOpNode,
    VariableNode,
)
from ..sparse_color import METAL_SPARSE_TEXTURE_METHODS, import_sparse_color_member
from ..static_member_calls import unresolved_static_member_call_owner
from ..vote_masks import render_vote_mask_conversion, render_vote_mask_method
from .support import (
    MetalDivisionProfileError,
    MetalStructMethodCallResolutionError,
    MetalWideVectorLoweringError,
)


class MetalExpressionMixin:
    """Members of ``MetalToCrossGLConverter``: one renderer per expression node kind of ``generate_expression``."""

    def generate_binary_expression(self, expr, is_main=False):
        """Render a Metal binary operator expression as CrossGL syntax."""
        comma = crossgl_comma_expression(
            expr, lambda operand: self.generate_expression(operand, is_main)
        )
        if comma is not None:
            return comma
        free_operator_call = self.generate_metal_free_operator_call(
            expr.op,
            (expr.left, expr.right),
            is_main,
            getattr(expr, "source_location", None),
        )
        if free_operator_call is not None:
            return free_operator_call
        if self.binary32_comparison_profile is not None and expr.op in {
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            profiled = self.generate_profiled_metal_comparison(expr, is_main)
            if profiled is not None:
                return profiled
        if self.floating_contraction_profile is not None and expr.op in {"+", "-"}:
            contracted = self.generate_metal_contracted_arithmetic(expr, is_main)
            if contracted is not None:
                return contracted
        cooperative_matrix_operation = {
            "*": "cooperative_matrix_multiply",
            "+": "cooperative_matrix_add",
            "-": "cooperative_matrix_subtract",
        }.get(expr.op)
        if cooperative_matrix_operation is not None and all(
            self.is_metal_cooperative_matrix_expression(operand)
            for operand in (expr.left, expr.right)
        ):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{cooperative_matrix_operation}({left}, {right})"
        wide_vector_binary = self.generate_wide_vector_binary_expression(expr, is_main)
        if wide_vector_binary is not None:
            return wide_vector_binary
        if (
            expr.op in {"/", "+", "-", "*"}
            and self.metal_arithmetic_profile(expr.op)[0] is not None
        ):
            profiled = self.generate_profiled_metal_arithmetic(expr, is_main)
            if profiled is not None:
                return profiled
        conversion_plan = self.metal_builtin_lowered_conversion_plan(
            expr.op,
            (expr.left, expr.right),
        )
        if conversion_plan is not None:
            helpers, _converted_types = conversion_plan
            operands = []
            for index, (operand, helper) in enumerate(
                zip((expr.left, expr.right), helpers)
            ):
                if helper is None:
                    operands.append(
                        self.generate_binary_operand(
                            operand,
                            expr.op,
                            index == 1,
                            is_main,
                        )
                    )
                    continue
                helper_name = self.sanitize_identifier(
                    self.function_output_name(helper)
                )
                operands.append(
                    f"{helper_name}({self.generate_expression(operand, is_main)})"
                )
            return f"{operands[0]} {expr.op} {operands[1]}"
        refuse_aggregate_builtin_operation(
            self, expr, getattr(expr, "source_location", None)
        )
        left = self.generate_binary_operand(expr.left, expr.op, False, is_main)
        right = self.generate_binary_operand(expr.right, expr.op, True, is_main)
        source_types = [
            self.expression_metal_type(operand) for operand in (expr.left, expr.right)
        ]
        if self.binary16_widening_profile is not None and expr.op in {
            "+",
            "-",
            "*",
            "/",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
        }:
            computation = (
                self.metal_binary_expression_type(expr)
                if expr.op in {"+", "-", "*", "/"}
                else self.metal_scalar_binary_result_type("+", *source_types)
            )
            left = self.render_metal_half_widening(left, source_types[0], computation)
            right = self.render_metal_half_widening(right, source_types[1], computation)
        bfloat_plan = self.metal_bfloat_arithmetic_plan(
            expr.op, *source_types, getattr(expr, "source_location", None)
        )
        if bfloat_plan is not None:
            operands = [left, right]
            for index, conversion_type in enumerate(bfloat_plan[1]):
                if conversion_type is not None:
                    operands[index] = (
                        f"{self.map_type(conversion_type)}({operands[index]})"
                    )
            return f"{operands[0]} {expr.op} {operands[1]}"
        conversion = self.metal_vector_scalar_conversion(expr.op, *source_types)
        if conversion is not None:
            index, element_type = conversion
            operands = [left, right]
            operands[index] = self.metal_vector_scalar_operand(
                operands[index], source_types[index], element_type
            )
            left, right = operands
        if expr.op in {"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>"}:
            source_types = [
                self.resolve_type_alias(source_type) for source_type in source_types
            ]
            operand_types = [
                self.normalized_metal_type(vtype) for vtype in source_types
            ]
            if (
                "bool" in operand_types
                and all(
                    self.metal_pointer_pointee_type_once(vtype) is None
                    for vtype in source_types
                )
                and all(
                    self.metal_scalar_arithmetic_type_info(vtype) is not None
                    for vtype in operand_types
                )
            ):
                # C++ scalar bool arithmetic promotes to signed int before
                # the operation, not back to bool after each subexpression.
                if operand_types[0] == "bool":
                    left = f"int({left})"
                if operand_types[1] == "bool":
                    right = f"int({right})"
        return f"{left} {expr.op} {right}"

    def generate_function_call_expression(self, expr, is_main=False):
        """Render a Metal free function or constructor call as CrossGL syntax."""
        if len(getattr(expr, "args", None) or ()) == 1:
            vote_mask = render_vote_mask_conversion(
                self, expr.name, expr.args[0], is_main
            )
            if vote_mask is not None:
                return vote_mask
        filled_matrix = self.generate_metal_make_filled_matrix(expr, is_main)
        if filled_matrix is not None:
            return filled_matrix
        profiled_constructor = self.generate_metal_half_widening_constructor(
            expr.name, expr.args, is_main
        )
        if profiled_constructor is not None:
            return profiled_constructor
        lowered_static_call = self.generate_lowered_static_struct_method_call(
            expr, is_main
        )
        if lowered_static_call is not None:
            return lowered_static_call
        lowered_method_call = self.generate_lowered_struct_method_call(expr, is_main)
        if lowered_method_call is not None:
            return lowered_method_call
        cooperative_matrix_call = self.generate_cooperative_matrix_function_call(
            expr, is_main
        )
        if cooperative_matrix_call is not None:
            return cooperative_matrix_call
        constructor_arguments = expr.args
        if (
            getattr(expr, "is_braced_constructor", False)
            and len(expr.args) == 1
            and isinstance(expr.args[0], InitializerListNode)
        ):
            constructor_arguments = expr.args[0].elements
        aggregate_conversion = render_explicit_aggregate_scalar_conversion(
            self,
            expr.name,
            constructor_arguments,
            is_main,
            getattr(expr, "source_location", None),
        )
        if aggregate_conversion is not None:
            return aggregate_conversion
        explicit_constructor = self.generate_explicit_constructor_call(
            expr.name,
            constructor_arguments,
            is_main,
            getattr(expr, "source_location", None),
        )
        if explicit_constructor is not None:
            return explicit_constructor
        wide_vector_constructor = self.generate_wide_vector_constructor(
            expr.name,
            constructor_arguments,
            is_main,
            getattr(expr, "source_location", None),
            braced=getattr(expr, "is_braced_constructor", False),
        )
        if wide_vector_constructor is not None:
            return wide_vector_constructor
        if getattr(expr, "is_braced_constructor", False) and expr.args:
            initializer = expr.args[0]
            if isinstance(initializer, InitializerListNode):
                return self.generate_initializer_list(initializer, is_main, expr.name)
        sizeof_value = self.render_metal_sizeof_expression(expr)
        if sizeof_value is not None:
            return sizeof_value
        numeric_limit = self.metal_numeric_limits_expression(expr.name, expr.args)
        if numeric_limit is not None:
            return numeric_limit
        uniform_value = self.metal_uniform_value_expression(
            expr.name, expr.args, is_main
        )
        if uniform_value is not None:
            return uniform_value
        bit_scan = self.metal_bit_scan_expression(expr.name, expr.args, is_main)
        if bit_scan is not None:
            return bit_scan
        sync_call = self.metal_synchronization_function_call(
            expr.name,
            expr.args,
            getattr(expr, "source_location", None),
        )
        if sync_call is not None:
            return sync_call
        atomic_call = self.metal_atomic_function_call(
            expr.name, expr.args, is_main, getattr(expr, "source_location", None)
        )
        if atomic_call is not None:
            return atomic_call
        callback = next((arg for arg in expr.args if isinstance(arg, LambdaNode)), None)
        if callback is not None:
            raise self.unsupported_callback_error(expr.name, callback)
        self.reject_unsupported_wide_vector_call(expr)
        constexpr_value = self.render_constexpr_helper_call(
            expr,
            is_main,
        )
        if constexpr_value is not None:
            return constexpr_value
        materialized_name = self.materialized_value_template_call_name(
            expr.name,
            expr.args,
            getattr(expr, "source_location", None),
        )
        self.validate_materialized_metal_stdlib_wrapper_call(expr)
        materialized_wave_call = self.generate_materialized_bfloat_wave_wrapper_call(
            expr, is_main
        )
        if materialized_wave_call is not None:
            return materialized_wave_call
        materialized_math_call = self.generate_materialized_bfloat_math_wrapper_call(
            expr, is_main
        )
        if materialized_math_call is not None:
            return materialized_math_call
        atan2_call = self.generate_metal_precise_atan2_call(expr, is_main)
        if atan2_call is not None:
            return atan2_call
        power_call = self.generate_metal_power_call(expr, is_main)
        if power_call is not None:
            return power_call
        log2_call = self.generate_metal_log2_call(expr, is_main)
        if log2_call is not None:
            return log2_call
        round_call = self.generate_metal_round_call(expr, is_main)
        if round_call is not None:
            return round_call
        fused_call = self.generate_metal_fma_call(expr, is_main)
        if fused_call is not None:
            return fused_call
        remainder_call = self.generate_metal_half_remainder_call(expr, is_main)
        if remainder_call is not None:
            return remainder_call
        remainder_call = self.generate_metal_remainder_call(expr, is_main)
        if remainder_call is not None:
            return remainder_call
        if self.resolve_metal_math_builtin_name(expr.name, expr.args) == "divide":
            result_type = self.metal_math_builtin_result_type(expr)
            left, right = (self.generate_expression(arg, is_main) for arg in expr.args)
            if self.metal_math_builtin_namespace_mode(expr.name) != "fast":
                if (
                    self.binary32_division_profile is not None
                    and self.current_function is None
                    and self.metal_profiled_arithmetic_type(result_type) is not None
                ):
                    raise MetalDivisionProfileError(
                        self.binary32_division_profile,
                        result_type,
                        "global constant division cannot call a runtime arithmetic helper",
                        getattr(expr, "source_location", None),
                    )
                profiled = self.render_profiled_metal_division(left, right, result_type)
                if profiled is not None:
                    return profiled
            return f"(({left}) / ({right}))"
        if self.resolve_metal_math_builtin_name(expr.name, expr.args) == "copysign":
            self.metal_math_builtin_result_type(expr)
        return self.render_metal_function_call(expr, materialized_name, is_main)

    def render_metal_function_call(self, expr, materialized_name, is_main=False):
        """Render a call that no special form claimed: its storage-wrapper and widened arguments, its transported
        overload name, and the ``fast::`` or ``precise::`` namespace of a math builtin.
        """
        wrapper_arguments = self.generate_storage_wrapper_call_arguments(
            materialized_name, expr, is_main
        )
        if self.binary16_widening_profile is not None:
            selected = self.selected_metal_callable(expr)
            if selected is None:
                candidates = self.metal_user_function_overloads(
                    materialized_name, source_offset=self.alias_source_offset(expr)
                )
                signatures = {
                    self.metal_function_source_signature(candidate)
                    for candidate in candidates
                }
                if len(signatures) == 1 and len(candidates[0].params) == len(expr.args):
                    candidate = candidates[0]
                    if all(
                        self.metal_source_overload_argument_match_rank(
                            argument,
                            self.expression_metal_type(argument),
                            parameter,
                        )
                        is not None
                        for argument, parameter in zip(expr.args, candidate.params)
                    ):
                        selected = candidate
            if selected is not None:
                for index, parameter in enumerate(selected.params):
                    if index not in wrapper_arguments and index < len(expr.args):
                        wrapper_arguments[index] = self.generate_metal_value_argument(
                            expr.args[index], parameter, is_main
                        )
        convert_user_call_arguments(
            self, materialized_name, expr, wrapper_arguments, is_main
        )
        materialized_name = self.transported_metal_source_overload_name(
            materialized_name,
            expr.args,
            getattr(expr, "source_location", None),
        )
        unbound_member = unresolved_static_member_call_owner(self, materialized_name)
        if unbound_member is not None:
            raise MetalStructMethodCallResolutionError(
                unbound_member[0],
                unbound_member[1],
                (),
                (),
                "the struct declares no materialized static member function of "
                "that name",
                getattr(expr, "source_location", None),
            )
        function_name = self.map_function_call_name(
            materialized_name,
            expr.args,
            source_offset=self.alias_source_offset(expr),
        )
        if function_name == "sampler":
            args = ", ".join(
                self.generate_sampler_constructor_arg(arg, is_main) for arg in expr.args
            )
        else:
            args = ", ".join(
                (
                    wrapper_arguments[index]
                    if index in wrapper_arguments
                    else self.generate_metal_wave_argument(
                        materialized_name, function_name, expr, index, is_main
                    )
                )
                for index in range(len(expr.args))
            )
        if metal_fast_math_operation(materialized_name) == function_name:
            # CrossGL keeps the namespace for the Metal backend; its parser
            # hands every other backend the plain builtin.
            function_name = f"fast::{function_name}"
        elif metal_precise_math_operation(materialized_name) == function_name:
            function_name = f"precise::{function_name}"
        elif function_name == "sizeof":
            # An operand this importer cannot size names a template parameter; CrossGL sizes it with MSL's
            # layout once the generators know the type.
            function_name = "metal_sizeof"
        return f"{function_name}({args})"

    def generate_call_expression(self, expr, is_main=False):
        """Render a Metal call through a callee expression as CrossGL syntax."""
        if (
            isinstance(expr.callee, FunctionCallNode)
            and str(expr.callee.name) in {"decltype", "metal::decltype"}
            and len(expr.callee.args) == 1
        ):
            inferred_type = self.expression_metal_type(expr.callee)
            if inferred_type is not None:
                constructor = self.map_type(inferred_type)
                arguments = ", ".join(
                    self.generate_expression(arg, is_main) for arg in expr.args
                )
                return f"{constructor}({arguments})"
        callee = self.generate_postfix_operand(expr.callee, is_main)
        args = ", ".join(self.generate_expression(arg, is_main) for arg in expr.args)
        return f"{callee}({args})"

    def generate_method_call_expression(self, expr, is_main=False):
        """Render a Metal method call as CrossGL syntax."""
        vote_method = render_vote_mask_method(self, expr, is_main)
        if vote_method is not None:
            return vote_method
        sparse_member = import_sparse_color_member(self, expr, is_main)
        if sparse_member is not None:
            return sparse_member
        wide_vector = self.wide_vector_expression_info(expr.object)
        if wide_vector is not None:
            raise MetalWideVectorLoweringError(
                wide_vector["source_type"],
                "the method has no semantics-preserving aggregate overload",
                getattr(expr, "source_location", None),
                operation=f"method {expr.method}",
            )
        obj = self.generate_postfix_operand(expr.object, is_main)
        args = ", ".join(self.generate_expression(arg, is_main) for arg in expr.args)
        method = expr.method
        if method in METAL_SPARSE_TEXTURE_METHODS:
            # A sparse read returns sparse_color<T>: the texel and whether its tile is resident.
            return self.refuse_import(
                f"sparse texture {method}: CrossGL has no texture read that reports tile residency "
                "(the texel and residency pair MSL returns as sparse_color<T>)"
            )
        resource_size_expr = self.resource_size_method_expression(expr, is_main)
        if resource_size_expr is not None:
            return resource_size_expr
        if method == "get_num_mip_levels":
            return f"textureQueryLevels({obj})"
        if method == "get_num_samples":
            samples_function = (
                "imageSamples"
                if self.is_storage_image_expression(expr.object)
                else "textureSamples"
            )
            return f"{samples_function}({obj})"
        descriptor = self.resource_method_descriptor(method)
        if descriptor and descriptor["sampled_texture"]:
            diagnostic = self.unsupported_storage_texture_sampled_method(
                expr.object, method, is_main
            )
            if diagnostic is not None:
                return diagnostic
        if method == "sample_compare":
            option_call = self.texture_compare_option_method_call(
                obj,
                expr.object,
                expr.args,
                is_main,
            )
            if option_call is not None:
                return option_call
        if descriptor and descriptor["storage_operation"] == "read":
            if self.is_storage_image_expression(expr.object):
                coord = self.storage_image_coordinate_expression(expr.object, expr.args)
                return f"imageLoad({obj}, {coord})" if coord else f"{obj}.read()"
            sampled_read = self.sampled_texture_read_expression(
                expr.object, expr.args, is_main
            )
            if sampled_read is not None:
                return self.metal_depth_texel_component(expr.object, sampled_read)
            return (
                f"{descriptor['function']}({obj}, {args})" if args else f"{obj}.read()"
            )
        if descriptor and descriptor["storage_operation"] == "write":
            if self.is_storage_image_expression(expr.object):
                coord = self.storage_image_coordinate_expression(
                    expr.object, expr.args[1:]
                )
                value = (
                    self.generate_expression(expr.args[0], is_main) if expr.args else ""
                )
                if coord and value:
                    value = self.storage_image_write_value(
                        expr.object, expr.args[0], value
                    )
                    return f"imageStore({obj}, {coord}, {value})"
                return f"{obj}.write({args})"
            return self.unsupported_sampled_texture_write(expr.object, method, is_main)
        if descriptor and method == "gather":
            return self.texture_gather_call(obj, expr.object, expr.args, is_main)
        if descriptor and method == "gather_compare":
            gather_compare_call = self.texture_gather_compare_call(
                obj, expr.object, expr.args, is_main
            )
            if gather_compare_call is not None:
                return gather_compare_call
        if descriptor:
            return f"{descriptor['function']}({obj}, {args})"
        return f"{obj}.{method}({args})"

    def generate_member_access_expression(self, expr, is_main=False):
        """Render a Metal member access as CrossGL syntax."""
        if expr.member == "value" and isinstance(expr.object, VariableNode):
            constant = self.render_integral_constant_binding(expr.object.name)
            if constant is not None:
                return constant
        static_member = self.render_decltype_static_struct_member(expr)
        if static_member is None:
            # `t.value` for a static member names its constant (instance_static_members.py).
            static_member = render_instance_static_struct_member(self, expr)
        if static_member is not None:
            return static_member
        if matrix_columns_object(self, expr) is not None:
            raise MetalMatrixColumnsError(getattr(expr, "source_location", None))
        obj = self.generate_postfix_operand(expr.object, is_main)
        constructor_this = (
            isinstance(expr.object, VariableNode)
            and expr.object.name == "this"
            and self.render_constructor_member_identifier("this") is not None
        )
        if getattr(expr, "is_pointer", False) and not constructor_this:
            obj = f"(*{obj})"
        wide_vector = self.wide_vector_expression_info(expr.object)
        if wide_vector is not None:
            lane = self.wide_vector_lane_index(str(expr.member), wide_vector["width"])
            if lane is None:
                raise MetalWideVectorLoweringError(
                    wide_vector["source_type"],
                    f"member selector '{expr.member}' is not a single lane",
                    getattr(expr, "source_location", None),
                    operation="member-access",
                )
            return f"{obj}.lanes[{lane}]"
        return f"{obj}.{self.render_struct_member_identifier(expr.object, expr.member)}"

    def generate_array_access_expression(self, expr, is_main=False):
        """Render a Metal subscript expression as CrossGL syntax."""
        cooperative_matrix_element = self.metal_cooperative_matrix_element_access(expr)
        if cooperative_matrix_element is not None:
            matrix, element_index = cooperative_matrix_element
            self.register_cooperative_matrix_element_contract(matrix)
            matrix_text = self.generate_expression(matrix, is_main)
            index_text = self.generate_expression(element_index, is_main)
            return f"cooperative_matrix_element({matrix_text}, {index_text})"
        matrix = matrix_columns_object(self, expr.array)
        if matrix is not None:
            matrix_text = self.generate_postfix_operand(matrix, is_main)
            return f"{matrix_text}[{self.generate_expression(expr.index, is_main)}]"
        component_info = self.small_vector_index_info(expr)
        if component_info is not None:
            return self.generate_small_vector_component_read(
                expr, component_info, is_main
            )
        if (
            not self.suppress_structured_buffer_index_lowering
            and id(expr) not in self.structured_buffer_lvalue_roots
            and self.is_structured_buffer_element_access(expr)
        ):
            return self.generate_structured_buffer_load(expr, is_main)
        wide_vector = self.wide_vector_expression_info(expr.array)
        array = self.generate_postfix_operand(expr.array, is_main)
        index = self.generate_expression(expr.index, is_main)
        if wide_vector is not None:
            return f"{array}.lanes[{index}]"
        return f"{array}[{index}]"

    def generate_unary_expression(self, expr, is_main=False):
        """Render a Metal unary operator expression as CrossGL syntax."""
        free_operator_call = self.generate_metal_free_operator_call(
            expr.op,
            (expr.operand,),
            is_main,
            getattr(expr, "source_location", None),
        )
        if free_operator_call is not None:
            return free_operator_call
        if expr.op in {"++", "--"} and isinstance(expr.operand, ArrayAccessNode):
            component_update = self.generate_small_vector_component_update(
                expr.operand, expr.op, postfix=False, is_main=is_main
            )
            if component_update is not None:
                return component_update
        if expr.op == "-" and self.is_metal_cooperative_matrix_expression(expr.operand):
            operand = self.generate_expression(expr.operand, is_main)
            return f"cooperative_matrix_negate({operand})"
        wide_vector = self.wide_vector_expression_info(expr.operand)
        if wide_vector is not None and expr.op != "&":
            raise MetalWideVectorLoweringError(
                wide_vector["source_type"],
                "the unary operator has no semantics-preserving aggregate lowering",
                getattr(expr, "source_location", None),
                operation=expr.op,
            )
        if expr.op == "&" and self.is_structured_buffer_element_access(expr.operand):
            operand = self.generate_without_structured_buffer_index_lowering(
                expr.operand, is_main
            )
        elif expr.op == "&" and isinstance(expr.operand, ArrayAccessNode):
            component_info = self.small_vector_index_info(expr.operand)
            if component_info is not None:
                array = self.generate_postfix_operand(expr.operand.array, is_main)
                index = self.generate_expression(expr.operand.index, is_main)
                operand = f"{array}[{index}]"
            else:
                operand = self.generate_expression(expr.operand, is_main)
        elif expr.op in {"++", "--"}:
            operand = self.generate_with_structured_buffer_lvalue(
                expr.operand,
                lambda: self.generate_expression(expr.operand, is_main),
            )
        else:
            operand = self.generate_expression(expr.operand, is_main)
        if expr.op == "post...":
            return operand
        if isinstance(expr.operand, (AssignmentNode, BinaryOpNode, TernaryOpNode)):
            operand = f"({operand})"
        return f"({expr.op}{operand})"

    def generate_postfix_expression(self, expr, is_main=False):
        """Render a Metal postfix increment or decrement as CrossGL syntax."""
        if expr.op in {"++", "--"} and isinstance(expr.operand, ArrayAccessNode):
            component_update = self.generate_small_vector_component_update(
                expr.operand, expr.op, postfix=True, is_main=is_main
            )
            if component_update is not None:
                return component_update
        wide_vector = self.wide_vector_expression_info(expr.operand)
        if wide_vector is not None:
            raise MetalWideVectorLoweringError(
                wide_vector["source_type"],
                "the postfix operator has no semantics-preserving aggregate lowering",
                getattr(expr, "source_location", None),
                operation=expr.op,
            )
        operand = self.generate_with_structured_buffer_lvalue(
            expr.operand,
            lambda: self.generate_postfix_operand(expr.operand, is_main),
        )
        return f"{operand}{expr.op}"

    def generate_ternary_expression(self, expr, is_main=False):
        """Render a Metal conditional expression as CrossGL syntax."""
        wide_vector = self.wide_vector_expression_info(
            expr.true_expr
        ) or self.wide_vector_expression_info(expr.false_expr)
        if wide_vector is not None:
            raise MetalWideVectorLoweringError(
                wide_vector["source_type"],
                "conditional selection has no semantics-preserving aggregate lowering",
                getattr(expr, "source_location", None),
                operation="?:",
            )
        condition = self.generate_precedence_operand(
            expr.condition,
            self.conditional_precedence,
            is_main,
            parenthesize_equal=True,
        )
        true_expr = self.generate_expression(expr.true_expr, is_main)
        false_expr = self.generate_precedence_operand(
            expr.false_expr,
            self.conditional_precedence,
            is_main,
        )
        if self.binary16_widening_profile is not None:
            result_type = self.expression_metal_type(expr)
            true_expr = self.render_metal_half_widening(
                true_expr, self.expression_metal_type(expr.true_expr), result_type
            )
            false_expr = self.render_metal_half_widening(
                false_expr, self.expression_metal_type(expr.false_expr), result_type
            )
        return f"{condition} ? {true_expr} : {false_expr}"

    def generate_cast_expression(self, expr, is_main=False):
        """Render a Metal cast expression as CrossGL syntax."""
        vote_mask = render_vote_mask_conversion(
            self, expr.target_type, expr.expression, is_main
        )
        if vote_mask is not None:
            return vote_mask
        compound_literal = self.generate_metal_compound_literal(expr, is_main)
        if compound_literal is not None:
            return compound_literal
        profiled_constructor = self.generate_metal_half_widening_constructor(
            expr.target_type, [expr.expression], is_main
        )
        if profiled_constructor is not None:
            return profiled_constructor
        cooperative_fragment = self.generate_cooperative_matrix_fragment_read(
            expr, is_main
        )
        if cooperative_fragment is not None:
            return cooperative_fragment
        aggregate_conversion = render_explicit_aggregate_scalar_conversion(
            self,
            expr.target_type,
            [expr.expression],
            is_main,
            getattr(expr, "source_location", None),
        )
        if aggregate_conversion is not None:
            return aggregate_conversion
        explicit_constructor = self.generate_explicit_constructor_call(
            expr.target_type,
            [expr.expression],
            is_main,
            getattr(expr, "source_location", None),
        )
        if explicit_constructor is not None:
            return explicit_constructor
        wide_vector_cast = self.generate_wide_vector_constructor(
            expr.target_type,
            [expr.expression],
            is_main,
            getattr(expr, "source_location", None),
        )
        if wide_vector_cast is not None:
            return wide_vector_cast
        mapped_type = self.map_type(expr.target_type)
        if (
            self.metal_pointer_pointee_type_once(
                self.resolve_type_alias(expr.target_type)
            )
            is not None
        ):
            qualifier_set = set(getattr(expr, "qualifiers", ()) or ())
            pointer_qualifiers = [
                qualifier
                for qualifier in (
                    "const",
                    "threadgroup_imageblock",
                    "threadgroup",
                    "thread",
                    "device",
                    "constant",
                )
                if qualifier in qualifier_set
            ]
            if (
                not self.preserve_pointer_pointee_const
                and "threadgroup" in pointer_qualifiers
            ):
                # Threadgroup pointer parameters and locals drop pointee
                # const under this policy, so a cast to `const threadgroup
                # T*` must too, or it cannot initialize those declarations.
                pointer_qualifiers = [
                    qualifier
                    for qualifier in pointer_qualifiers
                    if qualifier != "const"
                ]
            if pointer_qualifiers:
                mapped_type = f"{' '.join(pointer_qualifiers)} {mapped_type}"
        value = self.generate_expression(expr.expression, is_main)
        if not self.cast_uses_constructor_syntax(mapped_type):
            return (
                f"({mapped_type})"
                f"{self.generate_cast_operand(expr.expression, value)}"
            )
        return f"{self.sanitize_identifier(mapped_type)}({value})"

    def generate_vector_constructor_expression(self, expr, is_main=False):
        """Render a Metal vector constructor as CrossGL syntax."""
        if len(getattr(expr, "args", None) or ()) == 1:
            vote_mask = render_vote_mask_conversion(
                self, expr.type_name, expr.args[0], is_main
            )
            if vote_mask is not None:
                return vote_mask
        profiled_constructor = self.generate_metal_half_widening_constructor(
            expr.type_name, expr.args, is_main
        )
        if profiled_constructor is not None:
            return profiled_constructor
        size_query = self.texture_size_constructor_expression(expr, is_main)
        if size_query is not None:
            return size_query
        aggregate_conversion = render_explicit_aggregate_scalar_conversion(
            self,
            expr.type_name,
            list(expr.args),
            is_main,
            getattr(expr, "source_location", None),
        )
        if aggregate_conversion is not None:
            return aggregate_conversion
        explicit_constructor = self.generate_explicit_constructor_call(
            expr.type_name,
            list(expr.args),
            is_main,
            getattr(expr, "source_location", None),
        )
        if explicit_constructor is not None:
            return explicit_constructor
        wide_vector_constructor = self.generate_wide_vector_constructor(
            expr.type_name,
            expr.args,
            is_main,
            getattr(expr, "source_location", None),
        )
        if wide_vector_constructor is not None:
            return wide_vector_constructor
        mapped_type = self.map_type(expr.type_name)
        if mapped_type == "sampler":
            args = ", ".join(
                self.generate_sampler_constructor_arg(arg, is_main) for arg in expr.args
            )
        else:
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
        return f"{mapped_type}({args})"

    def generate_texture_sample_expression(self, expr, is_main=False):
        """Render a Metal texture sample as CrossGL syntax."""
        diagnostic = self.unsupported_storage_texture_sampled_method(
            expr.texture, "sample", is_main
        )
        if diagnostic is not None:
            return diagnostic
        texture = self.generate_expression(expr.texture, is_main)
        sampler_expr = expr.sampler
        coords_expr = expr.coordinates
        if coords_expr is None and self.is_samplerless_sample_expression(expr.texture):
            coords_expr = sampler_expr
            sampler_expr = None
        sampler = self.generate_expression(sampler_expr, is_main)
        sample_args = [texture]
        if sampler:
            sample_args.append(sampler)
        options = getattr(expr, "options", None)
        if options is None:
            options = [expr.lod] if getattr(expr, "lod", None) is not None else []
        coords, options = self.texture_sample_coordinate_and_options(
            expr.texture, coords_expr, options, is_main
        )
        sample_args.append(coords)

        if options:
            option_call = self.texture_sample_options_call(
                options, sample_args, is_main
            )
            if option_call is not None:
                return self.metal_depth_texel_component(expr.texture, option_call)

            lod_expr = self.unwrap_texture_option_argument(expr.lod, "level")
            lod = self.generate_expression(lod_expr, is_main)
            return self.metal_depth_texel_component(
                expr.texture, f"textureLod({', '.join(sample_args + [lod])})"
            )

        return self.metal_depth_texel_component(
            expr.texture, f"texture({', '.join(sample_args)})"
        )
