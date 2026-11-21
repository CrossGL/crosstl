"""Statement emission of the Slang generator: statement bodies, expression preludes, assignments and loop headers."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BlockNode,
    BreakNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    IdentifierNode,
    IfNode,
    MatchNode,
    ReturnNode,
    SwitchNode,
    VariableNode,
    WhileNode,
)
from .for_initializers import join_for_initializer
from .matrix_compound_assignment import matrix_product_assignment
from .slang_workgroup_locals import slang_workgroup_local_prefix


class SlangStatementsMixin:
    """Statement members of ``SlangCodeGen``."""

    def assignment_from_statement(self, statement):
        if isinstance(statement, AssignmentNode):
            return statement
        if isinstance(statement, ExpressionStatementNode) and isinstance(
            getattr(statement, "expression", None), AssignmentNode
        ):
            return statement.expression
        return None

    def identifier_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        return None

    def emit_function_body_statement(self, statement, statement_index, rewrite):
        if (
            rewrite is not None
            and rewrite["mode"] == "direct"
            and statement_index == rewrite["statement_index"]
        ):
            value = self.generate_expression_with_expected(
                rewrite["value"], rewrite["return_type_name"]
            )
            value = self.slang_precise_returned_value(
                rewrite["value"], value, rewrite["return_type_name"]
            )
            return f"{self.indent()}return {value};"
        return self.emit_statement(statement)

    def emit_statement(self, node):
        statement = self.generate_statement(node)
        return self.indent_statement_text(statement)

    def generate_scoped_statement_body(self, body, indent):
        return "".join(
            self.indent_text(self.generate_statement(stmt), indent)
            for stmt in self.get_statements(body)
        )

    def generate_statement_body(self, body, indent):
        return self.generate_scoped_statement_body(body, indent)

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = self.indent_str * indent
        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}{self.indent_str}break;\n"
        code += f"{indent_str}}}\n"
        return code

    def indent_text(self, text, indent):
        if not text:
            return ""
        prefix = self.indent_str * indent
        return (
            "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
            + "\n"
        )

    def indent_statement_text(self, statement):
        lines = statement.splitlines()
        return "\n".join(
            self.indent() + line if line and not line[0].isspace() else line
            for line in lines
        )

    def expression_prelude_active(self):
        return bool(self.expression_prelude_stack)

    def add_expression_prelude(self, lines, result_name=None):
        if not self.expression_prelude_stack:
            return
        self.expression_prelude_stack[-1].extend(lines)
        if result_name:
            self.expression_prelude_result_stack[-1].add(result_name)

    def generate_expression_with_prelude(self, expr, expected_type=None):
        self.expression_prelude_stack.append([])
        self.expression_prelude_result_stack.append(set())
        try:
            if expected_type is None:
                value = self.generate_expression(expr)
            else:
                value = self.generate_expression_with_expected(expr, expected_type)
            prelude = self.expression_prelude_stack[-1]
            result_names = self.expression_prelude_result_stack[-1]
            return prelude, result_names, value
        finally:
            self.expression_prelude_stack.pop()
            self.expression_prelude_result_stack.pop()

    def generate_statement_expression_with_prelude(self, expr):
        self.statement_expression_node_stack.append(expr)
        try:
            return self.generate_expression_with_prelude(expr)
        finally:
            self.statement_expression_node_stack.pop()

    def is_direct_statement_expression(self, node):
        return (
            self.current_expression_expected_type is None
            and bool(self.statement_expression_node_stack)
            and self.statement_expression_node_stack[-1] is node
        )

    def statement_with_prelude(self, prelude, statement):
        if not prelude:
            return statement
        statements = list(prelude)
        if statement:
            statements.append(statement)
        return "\n".join(statements)

    def generate_statement(self, node):
        """Render a single CrossGL statement as Slang code."""
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            prelude, _results, value = self.generate_expression_with_prelude(
                node.value, self.current_function_return_type
            )
            value = self.slang_precise_returned_value(
                node.value, value, self.current_function_return_type
            )
            return self.statement_with_prelude(prelude, f"return {value};")
        elif isinstance(node, AssignmentNode):
            return self.generate_assignment_statement(node)
        elif isinstance(node, ExpressionStatementNode):
            if isinstance(node.expression, AssignmentNode):
                return self.generate_assignment_statement(node.expression)
            tail_return = self.generate_tail_expression_statement(node)
            if tail_return is not None:
                return tail_return
            synchronization_statement = self.generate_slang_synchronization_statement(
                node.expression
            )
            if synchronization_statement is not None:
                return synchronization_statement
            prelude, result_names, expr = (
                self.generate_statement_expression_with_prelude(node.expression)
            )
            statement = "" if prelude and expr in result_names else f"{expr};"
            return self.statement_with_prelude(prelude, statement)
        elif isinstance(node, VariableNode):
            return self.generate_variable_statement(node)
        elif isinstance(node, IfNode):
            return self.generate_if(node)
        elif isinstance(node, ForNode):
            return self.generate_for(node)
        elif isinstance(node, ForInNode):
            return self.generate_for_in(node)
        elif isinstance(node, WhileNode):
            return self.generate_while(node)
        elif isinstance(node, DoWhileNode):
            return self.generate_do_while(node)
        elif isinstance(node, MatchNode):
            return self.generate_match(node)
        elif isinstance(node, SwitchNode):
            return self.generate_switch(node)
        elif isinstance(node, BlockNode):
            return self.generate_block_statement(node)
        elif isinstance(node, BreakNode):
            return "break;"
        elif isinstance(node, ContinueNode):
            return "continue;"
        else:
            synchronization_statement = self.generate_slang_synchronization_statement(
                node
            )
            if synchronization_statement is not None:
                return synchronization_statement
            prelude, result_names, expr = (
                self.generate_statement_expression_with_prelude(node)
            )
            statement = "" if prelude and expr in result_names else f"{expr};"
            return self.statement_with_prelude(prelude, statement)

    def generate_block_statement(self, node):
        """A nested ``{ ... }`` block; a declaration in it ends with the block, as in HLSL and C."""
        saved_types = (dict(self.variable_types), dict(self.local_variable_types))
        result = "{\n"
        self.indent_level += 1
        try:
            for stmt in self.get_statements(node):
                result += self.emit_statement(stmt) + "\n"
        finally:
            self.indent_level -= 1
            for table, saved in zip(
                (self.variable_types, self.local_variable_types), saved_types
            ):
                table.clear()
                table.update(saved)
        return result + self.indent() + "}"

    def generate_variable_statement(self, node):
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.variable_declaration_type(node, initial_value)
        self.register_variable_type(node.name, var_type, node)
        mapped_type = self.map_resource_type_with_format(var_type, node)
        declaration = self.format_declaration(var_type, node.name, node)
        prefix = (
            # A function-scope groupshared variable is static (slang_workgroup_locals.py).
            slang_workgroup_local_prefix(self, node, initial_value)
            + self.slang_declaration_qualifier_prefix(node)
            + self.slang_resource_memory_qualifier_prefix(mapped_type, node)
        )
        declaration = (
            self.slang_precise_local_prefix(node, prefix) + prefix + declaration
        )
        self.slang_precise_local_initializer(node, initial_value)
        if initial_value is not None:
            prelude, _results, initial_expr = self.generate_expression_with_prelude(
                initial_value,
                self.initializer_expected_type(var_type),
            )
            return self.statement_with_prelude(
                prelude, f"{declaration} = {initial_expr};"
            )
        return f"{declaration};"

    def slang_matrix_product_assignment(self, node):
        # Slang *= on matrices is component-wise; CrossGL's is the product.
        return matrix_product_assignment(
            node.left,
            node.right,
            node.operator,
            type_of=self.expression_result_type,
            is_matrix=self.is_matrix_value_type,
            is_vector=self.is_vector_value_type,
            backend="slang",
        )

    def generate_assignment(self, node):
        node = self.slang_promoted_compound_assignment(node)
        matrix_product = self.slang_matrix_product_assignment(node)
        if matrix_product is not None:
            return self.generate_assignment(matrix_product)
        block_store = self.generate_glsl_buffer_block_store(
            node.left, node.right, node.operator
        )
        if block_store is not None:
            self.slang_precise_lowered_store(node)
            return block_store
        column_assignment = self.hlsl_matrix_orientation.column_assignment(
            node.left, node.right, node.operator, statement_context=True
        )
        if column_assignment is not None:
            self.slang_precise_lowered_store(node)
            return column_assignment
        left = self.slang_assignment_target_alias(
            node.left
        ) or self.generate_expression(node.left)
        right = self.generate_expression_with_expected(
            node.right, self.expression_result_type(node.left)
        )
        right = self.slang_precise_stored_value(
            node.left, node.right, right, node.operator
        )
        # Slang's floating % and %= are the truncated remainder, as CrossGL's.
        return f"{left} {node.operator} {right}"

    def generate_assignment_statement(self, node):
        node = self.slang_promoted_compound_assignment(node)
        matrix_product = self.slang_matrix_product_assignment(node)
        if matrix_product is not None:
            return self.generate_assignment_statement(matrix_product)
        block_store = self.generate_glsl_buffer_block_store(
            node.left, node.right, node.operator
        )
        if block_store is not None:
            self.slang_precise_lowered_store(node)
            return block_store
        column_assignment = self.hlsl_matrix_orientation.column_assignment(
            node.left, node.right, node.operator, statement_context=True
        )
        if column_assignment is not None:
            self.slang_precise_lowered_store(node)
            return self.statement_with_prelude([], f"{column_assignment};")
        left = self.slang_assignment_target_alias(
            node.left
        ) or self.generate_expression(node.left)
        prelude, _results, right = self.generate_expression_with_prelude(
            node.right, self.expression_result_type(node.left)
        )
        right = self.slang_precise_stored_value(
            node.left, node.right, right, node.operator
        )
        # Slang's floating % and %= are the truncated remainder, as CrossGL's.
        statement = f"{left} {node.operator} {right};"
        return self.statement_with_prelude(prelude, statement)

    def tail_expression_returns(self, node):
        if not isinstance(node, ExpressionStatementNode):
            return False
        if not getattr(node, "is_tail_expression", False):
            return False
        return_type = self.type_name_string(self.current_function_return_type)
        return bool(return_type and return_type != "void")

    def generate_tail_expression_statement(self, node):
        if not self.tail_expression_returns(node):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        prelude, _results, value = self.generate_expression_with_prelude(
            node.expression,
            return_type,
        )
        value = self.slang_precise_returned_value(node.expression, value, return_type)
        return self.statement_with_prelude(prelude, f"return {value};")

    def generate_for_header_statement(self, node, atomic_value_context=None):
        if atomic_value_context is not None:
            self.atomic_value_context_stack.append(atomic_value_context)
            try:
                return self.generate_for_header_statement(node)
            finally:
                self.atomic_value_context_stack.pop()

        if isinstance(node, list):
            return join_for_initializer(
                [self.generate_for_header_statement(item) for item in node],
                [isinstance(item, VariableNode) for item in node],
                "slang",
            )
        if isinstance(node, AssignmentNode):
            return self.generate_assignment(node)
        if isinstance(node, VariableNode):
            initial_value = getattr(node, "initial_value", getattr(node, "value", None))
            var_type = self.variable_declaration_type(node, initial_value)
            self.register_variable_type(node.name, var_type, node)
            declaration = self.format_declaration(var_type, node.name, node)
            declaration = self.slang_declaration_qualifier_prefix(node) + declaration
            if initial_value is not None:
                initial_expr = self.generate_expression_with_expected(
                    initial_value,
                    self.initializer_expected_type(var_type),
                )
                return f"{declaration} = {initial_expr}"
            return declaration
        statement_expr = (
            node.expression if isinstance(node, ExpressionStatementNode) else node
        )
        self.statement_expression_node_stack.append(statement_expr)
        try:
            return self.generate_expression(node)
        finally:
            self.statement_expression_node_stack.pop()

    def generate_loop_condition_expression(self, node, atomic_value_context):
        self.atomic_value_context_stack.append(atomic_value_context)
        try:
            return self.generate_expression(node)
        finally:
            self.atomic_value_context_stack.pop()

    def slang_assignment_target_alias(self, target):
        hull_output_alias = self.slang_hull_output_array_alias(target)
        if hull_output_alias is not None:
            return hull_output_alias

        hull_output_alias = self.slang_hull_output_member_alias(target)
        if hull_output_alias is not None:
            return hull_output_alias

        target_name = self.slang_stage_builtin_output_target_name(target)
        if target_name is None:
            return None
        return self.identifier_aliases.get(target_name)

    def slang_hull_output_array_alias(self, target):
        if self.current_hull_output_rewrite is None:
            return None
        if not isinstance(target, ArrayAccessNode):
            return None

        array = getattr(target, "array", getattr(target, "array_expr", None))
        if self.identifier_name(array) != "gl_out":
            return None

        return self.current_hull_output_rewrite["local_name"]

    def slang_hull_output_member_alias(self, target):
        if self.current_hull_output_rewrite is None:
            return None

        member_target = self.slang_hull_output_member_target(target)
        if member_target is None:
            return None

        member, _index = member_target
        return f"{self.current_hull_output_rewrite['local_name']}.{member}"

    def generate_expression_with_expected(self, expr, expected_type):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.slang_promoted_store(
                expr, self.generate_expression(expr), expected_type
            )
        finally:
            self.current_expression_expected_type = previous_expected_type

    def generate_expression_without_expected(self, expr):
        previous_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = None
        try:
            return self.generate_expression(expr)
        finally:
            self.current_expression_expected_type = previous_expected_type
