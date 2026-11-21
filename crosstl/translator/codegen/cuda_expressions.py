"""Operator, constructor, member-access and array-access visitors of the CUDA generator."""

from ..ast import (
    IdentifierNode,
    MemberAccessNode,
)
from ..refusals import CudaRefusal
from .c_family_operands import (
    postfix_operand,
    prefix_unary_operand,
)
from .cuda_builtin_components import compute_builtin_subscript
from .cuda_refusals import CudaLoweringRefused
from .cuda_tables import CUDA_SCALAR_CONSTRUCTOR_TYPE_ALIASES


class CudaExpressionMixin:
    """Expression members of ``CudaCodeGen``."""

    def format_swizzle_assignment_expression(self, target_node, value_node, operator):
        if not isinstance(target_node, MemberAccessNode):
            return None
        object_node = getattr(
            target_node, "object_expr", getattr(target_node, "object", None)
        )
        components = self.member_swizzle_components(target_node)
        if object_node is None or components is None or len(components) <= 1:
            return None

        object_info = self.swizzle_result_vector_info(
            self.expression_result_type(object_node)
        )
        if object_info is None:
            return None
        result_type = self.swizzle_vector_type(object_info, len(components))
        result_info = self.swizzle_result_vector_info(result_type)
        if result_info is None:
            return None

        object_expr = self.visit(object_node)
        value_expr = self.visit(value_node)
        value_info = self.swizzle_result_vector_info(
            self.expression_result_type(value_node)
        )
        temp_name = self.next_cuda_temp_variable("swizzle_value")
        if value_info is not None:
            if len(value_info["components"]) != len(components):
                return None
            temp_type = result_info["type"]
            value_parts = [f"{temp_name}.{part}" for part in result_info["components"]]
        else:
            scalar_type = self.scalar_component_type(
                self.expression_result_type(value_node)
            )
            if scalar_type is None:
                scalar_type = self.vector_scalar_parameter_type(object_info)
            temp_type = self.convert_crossgl_type_to_cuda(scalar_type)
            value_parts = None

        statements = [f"{temp_type} {temp_name} = {value_expr}"]
        if value_parts is None:
            value_parts = [temp_name] * len(components)

        compound_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        binary_op = compound_ops.get(operator)
        for component, value_part in zip(components, value_parts):
            target = f"{object_expr}.{component}"
            if operator == "=":
                statements.append(f"{target} = {value_part}")
            elif binary_op is not None:
                statements.append(f"{target} = ({target} {binary_op} {value_part})")
            else:
                return None
        return "; ".join(statements)

    def visit_BinaryOpNode(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        precise = self.precise_binary_operation(node, left, right, operator)
        if precise is not None:
            return precise
        logical = self.lower_vector_logical_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if logical is not None:
            return logical
        bitwise = self.lower_vector_bitwise_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if bitwise is not None:
            return bitwise
        comparison = self.lower_vector_comparison_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if comparison is not None:
            return comparison
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        if operator == "%":
            modulo = self.lower_scalar_modulo_operation(
                node.left,
                left,
                node.right,
                right,
            )
            if modulo is not None:
                return modulo
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.visit(node.operand)
        operator = getattr(node, "operator", getattr(node, "op", "+"))
        if operator == "not":
            operator = "!"
        if getattr(node, "is_postfix", getattr(node, "postfix", False)):
            return f"{postfix_operand(operand)}{operator}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, operator)
        if lowered is not None:
            return lowered
        return f"{operator}{prefix_unary_operand(operator, operand)}"

    def cuda_scalar_alias_constructor_expression(self, constructor_type, args):
        """Lower a sized scalar constructor such as ``f64(x)`` or ``u32(x)`` to ``static_cast``."""
        if constructor_type not in CUDA_SCALAR_CONSTRUCTOR_TYPE_ALIASES:
            return None
        if len(args) > 1:
            return None

        mapped_type = self.map_type(constructor_type)
        value = args[0] if args else "0"
        return f"static_cast<{mapped_type}>({value})"

    def cuda_vector_constructor_arguments(
        self,
        node,
        vector_info,
        positional_args,
        positional_exprs,
    ):
        named_args = dict(getattr(node, "named_arguments", {}) or {})
        if not named_args:
            return positional_args, positional_exprs

        components = vector_info["components"]
        if all(field_name in components for field_name in named_args):
            ordered_named_args = [
                named_args[component]
                for component in components
                if component in named_args
            ]
            ordered_named_exprs = [self.visit(arg) for arg in ordered_named_args]
            return (
                positional_args + ordered_named_args,
                positional_exprs + ordered_named_exprs,
            )

        ordered_named_args = list(named_args.values())
        ordered_named_exprs = [self.visit(arg) for arg in ordered_named_args]
        return (
            ordered_named_args + positional_args,
            ordered_named_exprs + positional_exprs,
        )

    def cuda_struct_constructor_brace_arguments(self, node, struct_name):
        """Render struct constructor fields and embedded resource sidecars."""
        positional_args = list(getattr(node, "arguments", []) or [])
        named_args = dict(getattr(node, "named_arguments", {}) or {})
        fields = list(self.struct_member_types.get(struct_name, {}).items())
        field_names = [field_name for field_name, _field_type in fields]

        if len(positional_args) > len(fields):
            raise CudaRefusal(
                f"Struct constructor {struct_name} expects at most {len(fields)} "
                f"arguments, got {len(positional_args)}"
            )

        unknown_names = sorted(set(named_args) - set(field_names))
        if unknown_names:
            raise CudaRefusal(
                f"Struct constructor {struct_name} has no field "
                f"{', '.join(unknown_names)}"
            )

        rendered_args = []
        metadata_members = self.struct_query_metadata_members.get(struct_name, set())
        for index, (field_name, field_type) in enumerate(fields):
            if index < len(positional_args):
                raw_arg = positional_args[index]
            else:
                raw_arg = named_args.get(field_name)

            if raw_arg is None:
                rendered_arg = self.diagnostic_zero_value_for_type(field_type)
            else:
                rendered_arg = self.generate_expression_with_expected(
                    raw_arg,
                    field_type,
                )
            rendered_args.append(rendered_arg)

            if field_name not in metadata_members:
                continue
            metadata_arg = (
                self.query_metadata_expression(raw_arg) if raw_arg is not None else None
            )
            if metadata_arg is None:
                metadata_arg = self.unavailable_query_metadata_argument(
                    field_name,
                    field_type,
                )
            rendered_args.append(metadata_arg)

        return rendered_args

    def visit_MemberAccessNode(self, node):
        """Visit member access"""
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        raw_object_name = getattr(object_node, "name", None)
        if raw_object_name is not None:
            alias_component = self.cuda_stage_builtin_alias_expression(
                raw_object_name, node.member
            )
            if alias_component is not None:
                return alias_component
            if not self.cuda_identifier_shadows_compute_builtin(raw_object_name):
                direct_builtin = self.cuda_compute_builtin_expression(
                    self.cuda_compute_builtin_role_for_name(raw_object_name),
                    node.member,
                )
                if direct_builtin is not None:
                    return direct_builtin
                raw_member_access = f"{raw_object_name}.{node.member}"
                if raw_member_access in self.builtin_map:
                    return self.builtin_map[raw_member_access]

        obj = self.visit(object_node)
        member_access = f"{obj}.{node.member}"
        if (
            member_access in self.builtin_map
            and not self.cuda_identifier_shadows_compute_builtin(member_access)
        ):
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, obj)
        if swizzle is not None:
            return swizzle

        return member_access

    def visit_PointerAccessNode(self, node):
        """Visit C-style pointer member access."""
        pointer_expr = self.visit(getattr(node, "pointer_expr", None))
        return f"{pointer_expr}->{node.member}"

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.swizzle_result_vector_info(
            self.expression_result_type(object_node)
        )
        if vector_info is None:
            return None

        components = self.member_swizzle_components(node)
        if components is None:
            return None

        if len(components) == 1:
            return f"{object_expr}.{components[0]}"

        result_type = self.swizzle_vector_type(vector_info, len(components))
        result_info = self.swizzle_result_vector_info(result_type)
        if result_info is None:
            return None

        swizzle_call = self.generate_vector_swizzle_single_eval_call(
            result_info,
            vector_info,
            object_node,
            object_expr,
            components,
        )
        if swizzle_call is not None:
            return swizzle_call

        args = [f"{object_expr}.{component}" for component in components]
        return f"{result_info['constructor']}({', '.join(args)})"

    def visit_ArrayAccessNode(self, node):
        """Visit array access"""
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        builtin_component = self.cuda_compute_builtin_index_component(node)
        if builtin_component is not None:
            return builtin_component
        if hasattr(node, "array_expr"):
            array = self.visit(node.array_expr)
        else:
            array = self.visit(node.array)

        self.subscript_index_depth += 1
        try:
            if hasattr(node, "index_expr"):
                index = self.visit(node.index_expr)
            else:
                index = self.visit(node.index)
        finally:
            self.subscript_index_depth -= 1

        return f"{array}[{index}]"

    def cuda_compute_builtin_index_component(self, node):
        """Return the component expression for a constant subscript of a compute built-in.

        ``gl_GlobalInvocationID[0]`` reads the ``.x`` component: a CUDA
        ``uint3`` has no subscript operator, so the ``make_uint3(...)[0]`` the
        generic path would emit does not compile. A user symbol with a
        built-in name shadows the built-in and keeps the generic subscript. A
        runtime subscript reads its component through a helper
        (:mod:`.cuda_builtin_components`). An out-of-range constant subscript,
        a subscript of the scalar local-invocation index, or a write to a
        built-in component is refused.
        """
        array_node = getattr(node, "array_expr", getattr(node, "array", None))
        if not isinstance(array_node, IdentifierNode):
            return None
        name = getattr(array_node, "name", None)
        if name in self.stage_builtin_aliases:
            role = self.stage_builtin_aliases[name]
        elif self.cuda_identifier_shadows_compute_builtin(name):
            return None
        else:
            role = self.cuda_compute_builtin_role_for_name(name)
        if role is None:
            return None
        if self.assignment_lhs_depth > 0 and self.subscript_index_depth == 0:
            raise CudaLoweringRefused(
                [f"cannot assign to a component of compute built-in '{name}' in CUDA"]
            )
        index_node = getattr(node, "index_expr", getattr(node, "index", None))
        return compute_builtin_subscript(
            self,
            name,
            role,
            index_node,
            self.cuda_compute_builtin_expression,
            CudaLoweringRefused,
            "CUDA",
        )

    def visit_ArrayLiteralNode(self, node):
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"
