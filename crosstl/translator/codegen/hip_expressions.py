"""Operator, constructor, member-access and array-access visitors of the HIP generator."""

from ..ast import (
    ArrayLiteralNode,
    IdentifierNode,
    MemberAccessNode,
)
from .c_family_operands import (
    postfix_operand,
    prefix_unary_operand,
)
from .enum_utils import generate_enum_constructor_expression
from .generic_struct_utils import generate_struct_constructor_expression
from .hip_refusals import HipLoweringRefused
from .hip_tables import HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES


class HipExpressionMixin:
    """Expression members of ``HipCodeGen``."""

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
        temp_name = self.next_hip_temp_variable("swizzle_value")
        value_info = self.swizzle_result_vector_info(
            self.expression_result_type(value_node)
        )
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
            temp_type = self.map_type(scalar_type)
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

    def visit_BinaryOpNode(self, node) -> str:
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

        if operator == "and":
            return f"({left} && {right})"
        elif operator == "or":
            return f"({left} || {right})"
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

    def visit_UnaryOpNode(self, node) -> str:
        operand = self.visit(node.operand)

        if node.op in {"!", "not"}:
            lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
            if lowered is not None:
                return lowered
            return f"!{prefix_unary_operand('!', operand)}"
        elif node.op in ["++", "--"]:
            if getattr(node, "is_postfix", getattr(node, "postfix", False)):
                return f"{postfix_operand(operand)}{node.op}"
            else:
                return f"{node.op}{prefix_unary_operand(node.op, operand)}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
        if lowered is not None:
            return lowered
        return f"{node.op}{prefix_unary_operand(node.op, operand)}"

    def visit_ConstructorNode(self, node) -> str:
        enum_constructor = generate_enum_constructor_expression(self, node)
        if enum_constructor is not None:
            return enum_constructor

        struct_constructor = generate_struct_constructor_expression(self, node)
        if struct_constructor is not None:
            return struct_constructor

        constructor_type = self.type_name_string(
            getattr(node, "constructor_type", getattr(node, "vtype", None))
        )
        positional_args = list(getattr(node, "arguments", []) or [])
        named_args = dict(getattr(node, "named_arguments", {}) or {})
        rendered_args = [self.visit(arg) for arg in positional_args]

        vector_info = self.vector_type_info(constructor_type)
        if vector_info:
            positional_args, rendered_args = self.hip_vector_constructor_arguments(
                vector_info,
                positional_args,
                rendered_args,
                named_args,
            )
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, positional_args, rendered_args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, positional_args, rendered_args
            )
            if constructor_call is not None:
                return constructor_call
            rendered_args = self.generate_vector_constructor_args(
                vector_info, positional_args, rendered_args
            )
            return f"{vector_info['constructor']}({', '.join(rendered_args)})"

        if named_args and constructor_type:
            fields = list(self.struct_member_types.get(constructor_type, {}).items())
            consumed = set()
            for field_name, field_type in fields[len(rendered_args) :]:
                if field_name not in named_args:
                    break
                consumed.add(field_name)
                rendered_args.append(
                    self.generate_expression_with_expected(
                        named_args[field_name],
                        field_type,
                    )
                )
            for field_name, value in named_args.items():
                if field_name not in consumed:
                    rendered_args.append(self.visit(value))

        half_constructor = self.hip_half_constructor_expression(
            constructor_type, positional_args, rendered_args
        )
        if half_constructor is not None:
            return half_constructor

        scalar_alias_constructor = self.hip_scalar_alias_constructor_expression(
            constructor_type, rendered_args
        )
        if scalar_alias_constructor is not None:
            return scalar_alias_constructor

        mapped_type = self.map_type(constructor_type) if constructor_type else ""
        if not mapped_type:
            return "{" + ", ".join(rendered_args) + "}"
        return f"{mapped_type}{{{', '.join(rendered_args)}}}"

    def hip_scalar_alias_constructor_expression(self, constructor_type, args):
        """Lower shader scalar aliases to valid HIP/C++ casts."""
        if constructor_type not in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES:
            return None
        if len(args) > 1:
            return None

        mapped_type = self.map_type(constructor_type)
        value = args[0] if args else "0"
        return f"static_cast<{mapped_type}>({value})"

    def hip_vector_constructor_arguments(
        self,
        vector_info,
        positional_args,
        positional_exprs,
        named_args,
    ):
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

    def visit_ArrayAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        builtin_component = self.hip_compute_builtin_index_component(node)
        if builtin_component is not None:
            return builtin_component
        array = self.visit(node.array)
        self.subscript_index_depth += 1
        try:
            index = self.visit(node.index)
        finally:
            self.subscript_index_depth -= 1
        return f"{array}[{index}]"

    def hip_compute_builtin_index_component(self, node):
        """Return the component expression for a constant subscript of a compute built-in.

        ``gl_GlobalInvocationID[0]`` reads the ``.x`` component: a HIP
        ``uint3`` has no subscript operator, so the ``make_uint3(...)[0]`` the
        generic path would emit does not compile. A user symbol with a
        built-in name shadows the built-in and keeps the generic subscript. A
        non-constant or out-of-range subscript, a subscript of the scalar
        local-invocation index, or a write to a built-in component has no
        faithful HIP lowering and is refused instead.
        """
        array_node = getattr(node, "array_expr", getattr(node, "array", None))
        if not isinstance(array_node, IdentifierNode):
            return None
        name = getattr(array_node, "name", None)
        if name in self.stage_builtin_aliases:
            role = self.stage_builtin_aliases[name]
        elif self.source_identifier_shadows_builtin(name):
            return None
        else:
            role = self.hip_compute_builtin_role_for_name(name)
        if role is None:
            return None
        if self.assignment_lhs_depth > 0 and self.subscript_index_depth == 0:
            raise HipLoweringRefused(
                [f"cannot assign to a component of compute built-in '{name}' in HIP"]
            )
        index_node = getattr(node, "index_expr", getattr(node, "index", None))
        index = self.literal_int_value(index_node)
        if index is None or index not in (0, 1, 2):
            raise HipLoweringRefused(
                [
                    (
                        f"cannot subscript compute built-in '{name}' with a "
                        "non-constant index in HIP; use .x, .y or .z"
                    )
                ]
            )
        component = "xyz"[index]
        expression = self.hip_compute_builtin_expression(role, component)
        if expression is None:
            raise HipLoweringRefused(
                [f"compute built-in '{name}' has no '.{component}' component in HIP"]
            )
        return expression

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_MemberAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        raw_object_name = getattr(node.object, "name", None)
        raw_object_shadows_builtin = self.source_identifier_shadows_builtin(
            raw_object_name
        )
        if raw_object_name is not None and not raw_object_shadows_builtin:
            alias_component = self.hip_stage_builtin_alias_expression(
                raw_object_name, node.member
            )
            if alias_component is not None:
                return alias_component
            ray_builtin = self.hip_ray_builtin_expression(raw_object_name)
            if ray_builtin is not None:
                return f"{ray_builtin}.{node.member}"
            direct_builtin = self.hip_compute_builtin_expression(
                self.hip_compute_builtin_role_for_name(raw_object_name), node.member
            )
            if direct_builtin is not None:
                return direct_builtin
            raw_member_access = f"{raw_object_name}.{node.member}"
            if raw_member_access in self.builtin_map:
                return self.builtin_map[raw_member_access]

        object_expr = self.visit(node.object)
        member_access = f"{object_expr}.{node.member}"
        if not raw_object_shadows_builtin and member_access in self.builtin_map:
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, object_expr)
        if swizzle is not None:
            return swizzle

        return member_access

    def visit_PointerAccessNode(self, node) -> str:
        pointer_expr = self.visit(getattr(node, "pointer_expr", None))
        return f"{pointer_expr}->{node.member}"

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.swizzle_result_vector_info(
            self.expression_result_type(object_node)
        )
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
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
