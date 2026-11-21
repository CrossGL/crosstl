"""Slang rendering of the CrossGL expression nodes that need more than one step: array accesses and literals, member accesses, constructors, and unary and ternary operators."""

from ..ast import (
    BinaryOpNode,
    IdentifierNode,
)
from .c_family_operands import (
    postfix_operand,
    prefix_unary_operand,
)
from .enum_utils import (
    generate_enum_constructor_expression,
)
from .generic_struct_utils import (
    named_struct_constructor_arguments,
)
from .slang_cull_distance import (
    slang_cull_distance_argument,
    slang_is_scalar_cull_distance_member,
)
from .slang_value_resizes import slang_resize_constructor

HIT_TRIANGLE_VERTEX_POSITIONS = "gl_HitTriangleVertexPositionsEXT"


class SlangExpressionNodeMixin:
    """Per-node members of ``SlangCodeGen.generate_expression``."""

    def generate_slang_array_access_expression(
        self,
        node,
    ):
        """Render an array access: a hull output alias, a GLSL buffer block load, a matrix subscript or a plain subscript."""
        hull_output_alias = self.slang_hull_output_array_alias(node)
        if hull_output_alias is not None:
            return hull_output_alias
        block_load = self.generate_glsl_buffer_block_array_load(node)
        if block_load is not None:
            return block_load
        array_expr = getattr(node, "array", getattr(node, "array_expr", None))
        index_expr = getattr(node, "index", getattr(node, "index_expr", None))
        matrix_access = self.hlsl_matrix_orientation.subscript(array_expr, index_expr)
        if matrix_access is not None:
            return matrix_access
        index = self.format_array_access_index(index_expr)
        if (
            isinstance(array_expr, IdentifierNode)
            and array_expr.name == HIT_TRIANGLE_VERTEX_POSITIONS
        ):
            # GL_EXT_ray_tracing_position_fetch's array; Slang reads one vertex position at a time.
            return f"HitTriangleVertexPosition({index})"
        array = postfix_operand(self.generate_expression(array_expr))
        return f"{array}[{index}]"

    def generate_slang_array_literal_expression(
        self,
        node,
    ):
        """Render an array literal."""
        expected_element_type = self.array_literal_element_expected_type(
            self.current_expression_expected_type
        )
        elements = ", ".join(
            (
                self.generate_expression_with_expected(element, expected_element_type)
                if expected_element_type is not None
                else self.generate_expression(element)
            )
            for element in node.elements
        )
        return f"{{{elements}}}"

    def generate_slang_member_access_expression(
        self,
        node,
    ):
        """Render a member access."""
        hull_output_alias = self.slang_hull_output_member_alias(node)
        if hull_output_alias is not None:
            return hull_output_alias
        block_load = self.generate_glsl_buffer_block_member_load(node)
        if block_load is not None:
            return block_load
        object_type = self.expression_result_type(node.object)
        member_info = self.slang_lowered_struct_resource_member_info(
            object_type, node.member
        )
        if member_info is not None:
            return member_info["global_name"]
        obj = postfix_operand(self.generate_expression_without_expected(node.object))
        if slang_is_scalar_cull_distance_member(self, object_type, node.member):
            return f"{obj}.{node.member}[0]"
        return f"{obj}.{node.member}"

    def generate_slang_constructor_expression(
        self,
        node,
    ):
        """Render a constructor."""
        enum_constructor = generate_enum_constructor_expression(self, node)
        if enum_constructor is not None:
            return enum_constructor
        constructor_type = self.type_name_string(node.constructor_type)
        if node.named_arguments:
            fields = list(self.struct_member_types.get(constructor_type, {}).items())
            arguments = named_struct_constructor_arguments(
                constructor_type, node, fields
            )
            args = ", ".join(
                slang_cull_distance_argument(
                    self,
                    constructor_type,
                    field_name,
                    self.generate_expression_with_expected(arg, field_type),
                )
                for (arg, field_type), (field_name, _) in zip(arguments, fields)
            )
            return f"{self.convert_type(constructor_type)}({args})"
        resize = slang_resize_constructor(self, constructor_type, node.arguments)
        if resize is not None:
            return resize
        matrix_constructor = self.hlsl_matrix_orientation.constructor(
            constructor_type,
            node.arguments,
            lambda arg: self.generate_expression_with_expected(arg, None),
        )
        if matrix_constructor is not None:
            return matrix_constructor
        callee = self.convert_type(constructor_type)
        args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
        return f"{callee}({args})"

    def generate_slang_unary_expression(
        self,
        node,
    ):
        """Render a unary operator."""
        if node.op == "*" and self.is_structured_buffer_resource_type(
            self.expression_result_type(node.operand)
        ):
            # ``*name`` of a ``buffer T* name`` is its first element: a structured buffer has no address.
            return f"{postfix_operand(self.generate_expression(node.operand))}[0]"
        operand = self.generate_expression(node.operand)
        if isinstance(node.operand, BinaryOpNode):
            operand = f"({operand})"
        if getattr(node, "is_postfix", False):
            return f"{postfix_operand(operand)}{node.op}"
        return f"{node.op}{prefix_unary_operand(node.op, operand)}"

    def generate_slang_ternary_expression(
        self,
        node,
    ):
        """Render a ternary operator."""
        bool_vector_select = self.generate_bool_vector_select_expression(
            node.condition, node.true_expr, node.false_expr
        )
        if bool_vector_select is not None:
            self.slang_precise_lowered_condition(node.condition)
            return bool_vector_select
        condition = self.slang_precise_condition(
            node.condition, self.generate_expression(node.condition)
        )
        if self.current_expression_expected_type is not None:
            true_expr = self.generate_expression_with_expected(
                node.true_expr, self.current_expression_expected_type
            )
            false_expr = self.generate_expression_with_expected(
                node.false_expr, self.current_expression_expected_type
            )
        else:
            true_expr = self.generate_expression(node.true_expr)
            false_expr = self.generate_expression(node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"
