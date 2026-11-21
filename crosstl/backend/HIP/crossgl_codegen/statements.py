"""Statements and control flow in the HIP to CrossGL converter."""

from ...assignment_operands import grouped_operand


class HipStatementMixin:
    """Statement members of ``HipToCrossGLConverter``."""

    def visit_ReturnNode(self, node):
        if hasattr(node, "value") and node.value:
            target_type = self.return_type_by_node_id.get(id(node))
            if target_type is not None:
                self.validate_record_value_against_target(
                    node.value, target_type, "return statement", copy_semantics=True
                )
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def visit_BreakNode(self, node):
        self.emit("break;")

    def visit_ContinueNode(self, node):
        self.emit("continue;")

    def visit_BlockNode(self, node):
        """A nested ``{ ... }`` stays a CrossGL block, with its own variable and name scopes."""
        self.emit("{")
        self.indent_level += 1
        self.push_variable_type_scope()
        self.push_identifier_name_scope()
        try:
            for stmt in node.statements:
                self.emit_statement(stmt)
        finally:
            self.pop_identifier_name_scope()
            self.pop_variable_type_scope()
            self.indent_level -= 1
        self.emit("}")
        return ""

    def visit_IfNode(self, node):
        self.validate_record_scalar_context(node.condition, "if condition")
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "if_body") and node.if_body:
            if isinstance(node.if_body, list):
                for stmt in node.if_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.if_body)
        self.indent_level -= 1

        if hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            if isinstance(node.else_body, list):
                for stmt in node.else_body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        init_node = node.init if hasattr(node, "init") else None
        scoped_init = isinstance(init_node, list)
        if scoped_init:
            self.emit("{")
            self.indent_level += 1
            for stmt in init_node:
                self.emit_statement(stmt)
            init = ""
        else:
            init = self.format_statement_fragment(init_node)
        if hasattr(node, "condition") and node.condition:
            self.validate_record_scalar_context(node.condition, "for condition")
        condition = (
            self.visit(node.condition)
            if hasattr(node, "condition") and node.condition
            else ""
        )
        update = self.format_statement_fragment(
            node.update if hasattr(node, "update") else None
        )

        self.emit(f"for ({init}; {condition}; {update}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")
        if scoped_init:
            self.indent_level -= 1
            self.emit("}")

    def visit_RangeForNode(self, node):
        iterable = self.visit(node.iterable)
        self.emit(f"for {node.name} in {iterable} {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_WhileNode(self, node):
        self.validate_record_scalar_context(node.condition, "while condition")
        condition = self.visit(node.condition)
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit("}")

    def visit_DoWhileNode(self, node):
        self.validate_record_scalar_context(node.condition, "do-while condition")
        condition = self.visit(node.condition)
        self.emit("do {")

        self.indent_level += 1
        if hasattr(node, "body") and node.body:
            if isinstance(node.body, list):
                for stmt in node.body:
                    self.emit_statement(stmt)
            else:
                self.emit_statement(node.body)
        self.indent_level -= 1

        self.emit(f"}} while ({condition});")

    def visit_SwitchNode(self, node):
        self.validate_record_scalar_context(node.expression, "switch expression")
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        ordered_cases = getattr(node, "ordered_cases", None)
        if ordered_cases is not None:
            for case in ordered_cases:
                if case.value is None:
                    self.emit("default:")
                    self.indent_level += 1
                    for stmt in getattr(case, "body", []):
                        self.emit_statement(stmt)
                    self.indent_level -= 1
                else:
                    self.visit(case)
        else:
            for case in getattr(node, "cases", []):
                self.visit(case)

            if getattr(node, "default_case", None) is not None:
                self.emit("default:")
                self.indent_level += 1
                for stmt in node.default_case:
                    self.emit_statement(stmt)
                self.indent_level -= 1

        self.indent_level -= 1
        self.emit("}")

    def visit_CaseNode(self, node):
        value = self.visit(node.value)
        self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "body", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_TernaryOpNode(self, node):
        self.validate_record_scalar_context(node.condition, "ternary condition")
        condition = grouped_operand(self, node.condition)
        true_expr = grouped_operand(self, node.true_expr)
        false_expr = grouped_operand(self, node.false_expr)
        return f"({condition} ? {true_expr} : {false_expr})"
