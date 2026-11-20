"""Mojo rendering of the CrossGL expression nodes that need more than one step: raw strings, variables, binary operators, member accesses, ternary operators and identifier-like nodes."""

from .mojo_integer_promotion import (
    C_PROMOTING_BINARY_OPERATORS,
    promoted_operand,
)


class MojoExpressionNodeMixin:
    """Per-node members of ``MojoCodeGen.generate_expression``."""

    def generate_mojo_string_expression(
        self,
        expr,
        target_type,
    ):
        """Render an expression given as a raw string."""
        if expr in self.expression_identifier_replacements:
            return self.expression_identifier_replacements[expr]
        ray_builtin = self.mojo_ray_builtin_expression(expr)
        if ray_builtin is not None and expr not in self.variable_types:
            return ray_builtin
        gpu_builtin = self.mojo_gpu_builtin_expression(expr)
        if gpu_builtin is not None:
            return gpu_builtin
        builtin = self.mojo_builtin_placeholder_expression(expr)
        if builtin is not None:
            return builtin
        mapped_reference = self.map_enum_variant_reference(expr, target_type)
        if mapped_reference != expr:
            return mapped_reference
        return self.mojo_identifier(expr)

    def generate_mojo_variable_expression(
        self,
        expr,
    ):
        """Render a variable node."""
        if hasattr(expr, "vtype") and expr.vtype and expr.name:
            if expr.name in self.expression_identifier_replacements:
                return self.expression_identifier_replacements[expr.name]
            ray_builtin = self.mojo_ray_builtin_expression(expr.name)
            if ray_builtin is not None and expr.name not in self.variable_types:
                return ray_builtin
            gpu_builtin = self.mojo_gpu_builtin_expression(expr.name)
            if gpu_builtin is not None:
                return gpu_builtin
            builtin = self.mojo_builtin_placeholder_expression(expr.name)
            if builtin is not None:
                return builtin
            return self.mojo_identifier(expr.name)
        elif hasattr(expr, "name"):
            if expr.name in self.expression_identifier_replacements:
                return self.expression_identifier_replacements[expr.name]
            ray_builtin = self.mojo_ray_builtin_expression(expr.name)
            if ray_builtin is not None and expr.name not in self.variable_types:
                return ray_builtin
            gpu_builtin = self.mojo_gpu_builtin_expression(expr.name)
            if gpu_builtin is not None:
                return gpu_builtin
            builtin = self.mojo_builtin_placeholder_expression(expr.name)
            if builtin is not None:
                return builtin
            return self.mojo_identifier(expr.name)
        else:
            return str(expr)

    def generate_mojo_binary_expression(
        self,
        expr,
        target_type,
        target_context,
    ):
        """Render a binary operator."""
        comparison = self.generate_vector_comparison(expr)
        if comparison is not None:
            return comparison
        boolean_vector_logical = self.generate_boolean_vector_logical_op(expr)
        if boolean_vector_logical is not None:
            return boolean_vector_logical
        vector_binary = self.generate_vector_binary_op(expr)
        if vector_binary is not None:
            return vector_binary
        numeric_vector_binary = self.generate_numeric_vector_binary_op(
            expr, target_type, target_context
        )
        if numeric_vector_binary is not None:
            return numeric_vector_binary
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        op = self.map_operator(expr.op)
        if op in C_PROMOTING_BINARY_OPERATORS:
            left = promoted_operand(self, expr.left, left)
            right = promoted_operand(self, expr.right, right)
        return f"({left} {op} {right})"

    def generate_mojo_member_access_expression(
        self,
        expr,
    ):
        """Render a member access."""
        builtin_member = self.generate_builtin_member_access(expr)
        if builtin_member is not None:
            return builtin_member
        obj = self.generate_expression(expr.object)
        enum_variant = self.map_enum_variant_reference(f"{obj}.{expr.member}")
        if enum_variant != f"{obj}.{expr.member}":
            return enum_variant
        obj_type = self.expression_result_type(expr.object)
        obj_struct_type, obj_fields = self.struct_type_name_and_fields(obj_type)
        if obj_struct_type is not None and expr.member in obj_fields:
            return f"{obj}.{self.mojo_identifier(expr.member)}"
        swizzle_indices = self.get_swizzle_indices(expr.member)
        if swizzle_indices is not None:
            return self.generate_swizzle(
                expr.object, obj, obj_type, expr.member, swizzle_indices
            )
        return f"{obj}.{self.mojo_identifier(expr.member)}"

    def generate_mojo_ternary_expression(
        self,
        expr,
        target_type,
        target_context,
    ):
        """Render a ternary operator."""
        bool_vector_select = self.generate_bool_vector_select_expression(
            expr.condition, expr.true_expr, expr.false_expr
        )
        if bool_vector_select is not None:
            return bool_vector_select
        branch_context = target_context or "ternary expression"
        condition = self.generate_expression(expr.condition)
        self.validate_condition_scalar(expr.condition, "ternary condition")
        true_expr = self.generate_expression(
            expr.true_expr, target_type, f"{branch_context} true branch"
        )
        false_expr = self.generate_expression(
            expr.false_expr, target_type, f"{branch_context} false branch"
        )
        return f"({true_expr} if {condition} else {false_expr})"

    def generate_mojo_identifier_like_expression(
        self,
        expr,
        target_type,
    ):
        """Render a node whose class name contains ``Identifier``."""
        name = getattr(expr, "name", str(expr))
        if name in self.expression_identifier_replacements:
            return self.expression_identifier_replacements[name]
        ray_builtin = self.mojo_ray_builtin_expression(name)
        if ray_builtin is not None and name not in self.variable_types:
            return ray_builtin
        gpu_builtin = self.mojo_gpu_builtin_expression(name)
        if gpu_builtin is not None:
            return gpu_builtin
        builtin = self.mojo_builtin_placeholder_expression(name)
        if builtin is not None:
            return builtin
        mapped_reference = self.map_enum_variant_reference(name, target_type)
        if mapped_reference != name:
            return mapped_reference
        return self.mojo_identifier(name)
