"""Control flow and node visitors in the HLSL to CrossGL converter: loops, if, switch and assignment statements, and the node visitors."""

from ...common_ast import (
    BreakNode,
    ContinueNode,
)
from ...switch_clauses import ordered_switch_clauses
from ..DirectxAst import (
    IfNode,
    ReturnNode,
    VariableNode,
)


class ControlFlowMixin:
    """Control flow members of ``HLSLToCrossGLConverter``."""

    def generate_for_loop(self, node, indent, is_main):
        previous_types = dict(self.current_variable_types)
        previous_dims = dict(self.current_resource_array_dims)

        def render_initializer(initializer, include_type=True):
            if isinstance(initializer, VariableNode):
                array_suffix = self.format_array_suffixes(initializer, is_main)
                qualifier_prefix = self.format_local_storage_qualifier_prefix(
                    initializer
                ) + self.format_precise_qualifier_prefix(initializer)
                declarator = f"{self.render_identifier(initializer.name)}{array_suffix}"
                if include_type:
                    text = (
                        f"{qualifier_prefix}{self.map_variable_type(initializer)} "
                        f"{declarator}"
                    )
                else:
                    text = declarator
                if initializer.value is not None:
                    initializer_value = self.generate_expression(
                        initializer.value,
                        is_main,
                        expected_type=initializer.vtype,
                    )
                    text += f" = {initializer_value}"
                self.record_variable_type(initializer)
                return text
            return self.generate_expression(initializer, is_main)

        if isinstance(node.init, list):
            declaration_list = all(isinstance(item, VariableNode) for item in node.init)
            init = ", ".join(
                render_initializer(
                    item, include_type=not declaration_list or index == 0
                )
                for index, item in enumerate(node.init)
            )
        elif isinstance(node.init, VariableNode):
            init = render_initializer(node.init)
        elif node.init is None:
            init = ""
        else:
            init = self.generate_expression(node.init, is_main)

        condition = (
            self.generate_expression(node.condition, is_main)
            if node.condition is not None
            else ""
        )
        if isinstance(node.update, list):
            update = ", ".join(
                self.generate_expression(expr, is_main) for expr in node.update
            )
        else:
            update = (
                self.generate_expression(node.update, is_main)
                if node.update is not None
                else ""
            )

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        self.current_variable_types = previous_types
        self.current_resource_array_dims = previous_dims
        return code

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        # Keep the CrossGL do/while form: a ``continue`` in the body must still
        # reach the loop condition, which a ``while (true)`` rewrite skips.
        condition = self.generate_expression(node.condition, is_main)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + f"}} while ({condition});\n"
        return code

    def generate_if_statement(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1, is_main)
        code += "    " * indent + "}"

        if node.else_body:
            if isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent, is_main)
            else:
                code += " else {\n"
                code += self.generate_function_body(node.else_body, indent + 1, is_main)
                code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        if self.is_storage_image_texel_access(node.left):
            storage_store = self.generate_storage_image_store(
                node.left, node.right, node.operator, is_main
            )
            if storage_store is not None:
                return storage_store
        component_access = self.storage_image_component_assignment_access(node.left)
        if component_access is not None:
            access, component = component_access
            storage_store = self.generate_storage_image_component_store(
                access, component, node.right, node.operator, is_main
            )
            if storage_store is not None:
                return storage_store
        converted = self.wide_integer_compound_expression(node, is_main)
        if converted is not None:
            return converted
        matrix_assignment = self.matrix_assignment_expression(node, is_main)
        if matrix_assignment is not None:
            return matrix_assignment
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.scalar_matrix_assignment_value(node, is_main)
        if rhs is None:
            rhs = self.generate_expression(node.right, is_main)
            rhs = self.truncated_assignment_value(node, rhs)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def stage_struct_parameter_member_alias(self, expr):
        if not self.current_stage_struct_parameter_member_aliases:
            return None
        obj = getattr(expr, "object", None)
        if isinstance(obj, VariableNode):
            obj_name = getattr(obj, "name", None)
        elif isinstance(obj, str):
            obj_name = obj
        else:
            return None
        return self.current_stage_struct_parameter_member_aliases.get(
            (obj_name, getattr(expr, "member", None))
        )

    def generate_switch_statement(self, node, indent=1, is_main=False):
        previous_types = self.current_variable_types
        previous_dims = self.current_resource_array_dims
        self.current_variable_types = dict(previous_types)
        self.current_resource_array_dims = dict(previous_dims)
        expression = getattr(node, "expression", None) or getattr(
            node, "condition", None
        )
        code = (
            "    " * indent
            + f"switch ({self.generate_expression(expression, is_main)}) {{\n"
        )

        for value, statements in ordered_switch_clauses(node):
            label = (
                "default"
                if value is None
                else f"case {self.generate_expression(value, is_main)}"
            )
            code += "    " * (indent + 1) + f"{label}:\n"
            code += self.generate_function_body(
                statements, indent + 2, is_main, new_scope=False
            )

        code += "    " * indent + "}\n"
        self.current_variable_types = previous_types
        self.current_resource_array_dims = previous_dims
        return code

    def visit_BinaryOpNode(self, node):
        converted = self.wide_integer_binary_expression(node)
        if converted is not None:
            return converted
        if hasattr(node.left, "visit"):
            left = node.visit_child(self, node.left)
        else:
            left = self.generate_expression(node.left)

        if hasattr(node.right, "visit"):
            right = node.visit_child(self, node.right)
        else:
            right = self.generate_expression(node.right)

        if hasattr(node.op, "token_type"):
            if node.op.token_type in ("BITWISE_AND", "AMPERSAND", "&"):
                return f"({left} & {right})"
            elif node.op.token_type in ("BITWISE_OR", "PIPE", "|"):
                return f"({left} | {right})"
            elif node.op.token_type in ("BITWISE_XOR", "CARET", "^"):
                return f"({left} ^ {right})"
        elif hasattr(node.op, "value"):
            if node.op.value in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op.value in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op.value in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"
        elif isinstance(node.op, str):
            if node.op in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"

        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        return f"{left} {op_str} {right}"

    def visit_UnaryOpNode(self, node):
        if hasattr(node, "expr"):
            expr_node = node.expr
        else:
            expr_node = node.operand

        if hasattr(expr_node, "visit"):
            expr = node.visit_child(self, expr_node)
        else:
            expr = self.generate_expression(expr_node)

        if hasattr(node.op, "token_type") and node.op.token_type in (
            "BITWISE_NOT",
            "TILDE",
            "~",
        ):
            return f"(~{expr})"
        elif hasattr(node.op, "value") and node.op.value in (
            "~",
            "BITWISE_NOT",
            "TILDE",
        ):
            return f"(~{expr})"
        elif isinstance(node.op, str) and node.op in ("~", "BITWISE_NOT", "TILDE"):
            return f"(~{expr})"

        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        if getattr(node, "is_postfix", False):
            return f"{expr}{op_str}"
        return f"{op_str}{expr}"

    def visit_SwitchStatementNode(self, node):
        return self.visit_SwitchNode(node)

    def visit_SwitchCaseNode(self, node):
        return self.visit_CaseNode(node)

    def visit_SwitchNode(self, node):
        condition = self.generate_expression(node.condition)
        code = f"switch ({condition}) {{\n"

        for value, statements in ordered_switch_clauses(node):
            label = (
                "default"
                if value is None
                else f"case {self.generate_expression(value)}"
            )
            code += self.get_indent() + f"{label}:\n"
            self.indentation += 1
            for stmt in statements:
                code += self.get_indent() + self.generate_statement(stmt) + "\n"
            self.indentation -= 1

        code += self.get_indent() + "}\n"
        return code

    def visit_CaseNode(self, node):
        value = self.generate_expression(node.value)
        code = self.get_indent() + f"case {value}:\n"

        self.indentation += 1
        for stmt in node.body:
            code += self.get_indent() + self.generate_statement(stmt) + "\n"
        self.indentation -= 1

        return code

    def generate_statement(self, node):
        """Render one DirectX backend statement node as CrossGL source."""
        if isinstance(node, str):
            return node
        if isinstance(node, BreakNode):
            return "break;"
        if isinstance(node, ContinueNode):
            return "continue;"
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            return f"return {self.generate_expression(node.value)};"
        elif hasattr(self, f"visit_{type(node).__name__}"):
            method = getattr(self, f"visit_{type(node).__name__}")
            return method(node)
        else:
            return self.generate_expression(node)
