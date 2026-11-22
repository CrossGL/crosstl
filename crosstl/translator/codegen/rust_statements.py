"""Rust statements, blocks, branches and loops emitted for CrossGL statements."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    ConstructorNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReturnNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import get_array_size_from_node
from .assignment_values import is_reevaluable_place
from .rust_gpu_builtins import BARRIERS


class RustStatementMixin:
    """Statement members of ``RustCodeGen``."""

    def generate_statement(self, stmt, indent=0):
        """Render a single CrossGL statement as Rust code."""
        indent_str = "    " * indent

        if isinstance(stmt, VariableNode):
            initial_value = getattr(stmt, "initial_value", None)
            vtype = self.variable_declaration_type(stmt, initial_value)
            self.register_variable_type(stmt.name, vtype, scope="local")
            binding_keyword = self.local_let_keyword(stmt)
            binding_name = self.rust_identifier(stmt.name)
            rust_type = self.map_type(vtype)
            if self.pointer_type_parts(rust_type) is not None:
                # Dereferencing a raw pointer needs unsafe code that rust-gpu's SPIR-V pointer model does not
                # accept for a pointer into a storage buffer or a local.
                self.refuse_lowering(
                    f"pointer local: '{stmt.name}' ({vtype}) is not lowered to a rust-gpu reference"
                )
            default_array_initializer = self.local_array_default_initializer(rust_type)
            if initial_value is not None:
                increment_init = self.generate_increment_initializer_declaration(
                    stmt,
                    initial_value,
                    vtype,
                    indent,
                )
                if increment_init is not None:
                    return increment_init
                if isinstance(initial_value, MatchNode):
                    init_expr = self.generate_match_expression(
                        initial_value,
                        indent,
                        target_type=vtype,
                    )
                else:
                    init_expr = self.generate_expression_with_type(initial_value, vtype)
                    init_expr = self.normalize_assignment_rhs(
                        vtype, initial_value, init_expr, "="
                    )
                if self.expression_contains_block_node(initial_value):
                    init_expr = self.indent_multiline_expression(init_expr, indent)
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {init_expr};\n"
                )
            elif self.is_generated_struct_type(vtype):
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = Default::default();\n"
                )
            elif self.is_default_constructible_ray_runtime_type(rust_type):
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = Default::default();\n"
                )
            elif default_array_initializer is not None:
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {default_array_initializer};\n"
                )
            elif self.local_value_default_initializer(rust_type) is not None:
                return (
                    f"{indent_str}{binding_keyword} {binding_name}: "
                    f"{rust_type} = {self.local_value_default_initializer(rust_type)};\n"
                )
            else:
                return f"{indent_str}{binding_keyword} {binding_name}: {rust_type};\n"

        elif isinstance(stmt, ArrayNode):
            return self.generate_array_declaration(stmt, indent)

        elif isinstance(stmt, BlockNode):
            return (
                f"{indent_str}{self.generate_block_expression(stmt, indent=indent)}\n"
            )

        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"

        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)

        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)

        elif isinstance(stmt, ForInNode):
            return self.generate_for_in(stmt, indent)

        elif isinstance(stmt, WhileNode):
            return self.generate_while(stmt, indent)

        elif isinstance(stmt, LoopNode):
            return self.generate_loop(stmt, indent)

        elif isinstance(stmt, DoWhileNode):
            return self.generate_do_while(stmt, indent)

        elif isinstance(stmt, MatchNode):
            return self.generate_match(stmt, indent)

        elif isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)

        elif isinstance(stmt, ReturnNode):
            if hasattr(stmt, "value") and stmt.value is not None:
                # Handle both single values and lists
                if isinstance(stmt.value, list):
                    # Multiple return values (tuple)
                    values = ", ".join(
                        self.generate_expression(val) for val in stmt.value
                    )
                    return f"{indent_str}return ({values});\n"
                else:
                    # Single return value
                    return_expr = self.generate_return_expression_with_type(
                        stmt.value, self.current_return_type
                    )
                    return_expr = self.normalize_assignment_rhs(
                        self.current_return_type, stmt.value, return_expr, "="
                    )
                    if self.expression_contains_block_node(stmt.value):
                        return_expr = self.indent_multiline_expression(
                            return_expr,
                            indent,
                        )
                    if self.current_rust_stage_outputs:
                        return self.generate_rust_stage_output_return(
                            indent_str, return_expr
                        )
                    if self.current_rust_output_parameters:
                        return self.generate_rust_output_return(indent_str, return_expr)
                    return f"{indent_str}return {return_expr};\n"
            else:
                # Void return
                if self.current_rust_output_parameters:
                    return self.generate_rust_output_return(indent_str, None)
                return f"{indent_str}return;\n"

        elif isinstance(stmt, BreakNode):
            switch_label = self.active_switch_label()
            if switch_label is not None:
                return f"{indent_str}break {switch_label};\n"
            context = self.active_do_while_context()
            if context:
                break_flag = context["break_flag"]
                return f"{indent_str}{break_flag} = true;\n{indent_str}break;\n"
            return f"{indent_str}break;\n"

        elif isinstance(stmt, ContinueNode):
            label = self.loop_jump_label()
            if self.active_do_while_context():
                return f"{indent_str}break{label};\n"
            context = self.active_for_context()
            if context:
                update = context["update"]
                if str(update).strip():
                    return f"{indent_str}{update};\n{indent_str}continue{label};\n"
            return f"{indent_str}continue{label};\n"

        elif hasattr(stmt, "__class__") and "ExpressionStatement" in str(
            stmt.__class__
        ):
            if hasattr(stmt, "expression"):
                if self.is_discard_statement(stmt):
                    return f"{indent_str}spirv_std::arch::kill();\n"
                expression = self.generate_discarded_expression(stmt.expression)
                if getattr(stmt, "is_tail_expression", False):
                    return f"{indent_str}{expression}\n"
                return f"{indent_str}{expression};\n"
            else:
                return f"{indent_str}{self.generate_expression(stmt)};\n"

        elif isinstance(stmt, ArrayAccessNode):
            # ArrayAccessNode as statement - likely misclassified
            return self.refuse_lowering(
                "statement: an array element read as a statement"
            )

        elif isinstance(stmt, SyncNode):
            return self.generate_sync_statement(stmt, indent)

        elif stmt is None:
            # An empty statement.
            return ""
        else:
            # Try to generate as expression
            expr_result = self.generate_discarded_expression(stmt)
            if expr_result and expr_result.strip():
                return f"{indent_str}{expr_result};\n"
            else:
                return self.refuse_lowering(
                    f"statement: {type(stmt).__name__} has no Rust lowering"
                )

    def is_discard_statement(self, stmt):
        expression = getattr(stmt, "expression", None)
        return isinstance(expression, IdentifierNode) and expression.name == "discard"

    def generate_sync_statement(self, stmt, indent):
        """Render a SyncNode as a Rust synchronization call."""
        indent_str = "    " * indent
        sync_type = getattr(stmt, "sync_type", "")
        call = BARRIERS.get(sync_type)
        if call is None:
            return self.refuse_lowering(
                f"synchronization {sync_type or '<unnamed>'} has no rust-gpu barrier"
            )
        return f"{indent_str}{call};\n"

    def local_value_default_initializer(self, rust_type):
        scalar_initializer = self.rust_scalar_default_literal(rust_type)
        if scalar_initializer is not None:
            return scalar_initializer
        if self.is_rust_shader_pod_value_type(rust_type):
            return "Default::default()"
        return None

    def generate_wave_op_expression(self, expr):
        """Render a WaveOpNode as a Rust subgroup intrinsic call."""
        operation = getattr(expr, "operation", "")
        rust_name = self.wave_op_map.get(operation)
        if rust_name is None:
            return self.refuse_lowering(
                f"subgroup operation {operation or '<unnamed>'}", "0"
            )
        args = ", ".join(self.generate_expression(arg) for arg in expr.arguments or [])
        return f"{rust_name}({args})"

    def wave_op_result_type(self, operation, arguments):
        """Infer the CrossGL result type for a Rust subgroup helper."""
        helper_name = self.wave_op_map.get(operation)
        if helper_name is None:
            # generate_wave_op_expression refuses the operation; it has no result type.
            return None
        arg_types = [self.expression_result_type(arg) for arg in arguments or []]
        return self.wave_helper_result_type(helper_name, arg_types)

    def wave_helper_result_type(self, helper_name, arg_types):
        helper_name = str(helper_name or "")
        if helper_name in {
            "subgroup_size",
            "subgroup_invocation_id",
            "subgroup_ballot_bit_count",
            "subgroup_exclusive_ballot_bit_count",
            "subgroup_partitioned_ballot_bit_count",
        }:
            return "uint"
        if helper_name in {
            "subgroup_elect",
            "subgroup_any",
            "subgroup_all",
            "subgroup_all_equal",
            "subgroup_quad_any",
            "subgroup_quad_all",
        }:
            return "bool"
        if helper_name in {"subgroup_ballot", "subgroup_match"}:
            return "uvec4"
        if helper_name in {
            "subgroup_add",
            "subgroup_mul",
            "subgroup_min",
            "subgroup_max",
            "subgroup_and",
            "subgroup_or",
            "subgroup_xor",
            "subgroup_broadcast_first",
            "subgroup_broadcast",
            "subgroup_exclusive_add",
            "subgroup_exclusive_mul",
            "subgroup_partitioned_add",
            "subgroup_partitioned_mul",
            "subgroup_partitioned_and",
            "subgroup_partitioned_or",
            "subgroup_partitioned_xor",
            "subgroup_quad_broadcast",
            "subgroup_quad_swap_horizontal",
            "subgroup_quad_swap_vertical",
            "subgroup_quad_swap_diagonal",
            "subgroup_shuffle",
            "subgroup_shuffle_xor",
        }:
            return arg_types[0] if arg_types else None
        return None

    def local_let_keyword(self, stmt):
        """Return `let` or `let mut` for a generated local binding."""
        qualifiers = {str(q) for q in getattr(stmt, "qualifiers", []) or []}
        if (
            "mut" in qualifiers
            or getattr(stmt, "name", None) in self.current_mutated_names
        ):
            return "let mut"
        return "let"

    def generate_increment_initializer_declaration(
        self,
        stmt,
        initial_value,
        vtype,
        indent,
    ):
        if not isinstance(initial_value, UnaryOpNode):
            return None

        op = getattr(initial_value, "operator", getattr(initial_value, "op", ""))
        op = self.map_operator(op)
        if op not in {"++", "--"}:
            return None

        operand = self.generate_expression(getattr(initial_value, "operand", ""))
        assignment_op = "+=" if op == "++" else "-="
        increment = self.rust_increment_literal(
            self.expression_result_type(getattr(initial_value, "operand", None))
        )
        update = f"{'    ' * indent}{operand} {assignment_op} {increment};\n"
        declaration = (
            f"{'    ' * indent}{self.local_let_keyword(stmt)} "
            f"{self.rust_identifier(stmt.name)}: "
            f"{self.map_type(vtype)} = {operand};\n"
        )
        is_postfix = getattr(
            initial_value,
            "is_postfix",
            getattr(initial_value, "postfix", False),
        )
        if is_postfix:
            return declaration + update
        return update + declaration

    def statement_list(self, body):
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        if body is None:
            return []
        return [body]

    def block_tail_expression(self, tail):
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail,
            "is_tail_expression",
            False,
        ):
            return tail.expression
        if isinstance(
            tail,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                BinaryOpNode,
                BlockNode,
                ConstructorNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                TernaryOpNode,
                UnaryOpNode,
            ),
        ) or isinstance(tail, str):
            return tail
        return None

    def indent_multiline_expression(self, expression, indent, indent_first=False):
        if not isinstance(expression, str):
            return expression

        prefix = "    " * indent
        if "\n" not in expression:
            return f"{prefix}{expression}" if indent_first else expression

        lines = expression.split("\n")
        start_index = 0 if indent_first else 1
        for index in range(start_index, len(lines)):
            if lines[index]:
                lines[index] = f"{prefix}{lines[index]}"
        return "\n".join(lines)

    def expression_contains_block_node(self, node):
        if node is None:
            return False
        if isinstance(node, BlockNode):
            return True
        if isinstance(node, (list, tuple)):
            return any(self.expression_contains_block_node(item) for item in node)
        if isinstance(node, dict):
            return any(
                self.expression_contains_block_node(item) for item in node.values()
            )

        child_attrs = (
            "expression",
            "value",
            "initial_value",
            "target",
            "left",
            "right",
            "condition",
            "then_branch",
            "else_branch",
            "body",
            "guard",
            "operand",
            "object_expr",
            "object",
            "array_expr",
            "array",
            "index_expr",
            "index",
            "function",
            "arguments",
            "args",
            "named_arguments",
            "elements",
            "true_expr",
            "false_expr",
            "arms",
            "statements",
        )
        return any(
            self.expression_contains_block_node(getattr(node, attr, None))
            for attr in child_attrs
        )

    def generate_block_expression(
        self,
        node,
        target_type=None,
        return_context=False,
        indent=0,
    ):
        statements = self.statement_list(node)
        explicit_tail = getattr(node, "expression", None)
        if explicit_tail is not None:
            prefix_statements = statements
            tail_expression = explicit_tail
        elif statements:
            prefix_statements = statements[:-1]
            tail_expression = self.block_tail_expression(statements[-1])
        else:
            prefix_statements = []
            tail_expression = None

        indent_str = "    " * indent
        body_indent = indent + 1
        code = "{\n"

        saved_variable_types = self.variable_types.copy()
        saved_local_variable_names = self.local_variable_names.copy()
        try:
            for statement in prefix_statements:
                code += self.generate_statement(statement, body_indent)

            if tail_expression is not None:
                if return_context:
                    tail_value = self.generate_return_branch_expression_with_type(
                        tail_expression,
                        target_type,
                    )
                else:
                    tail_value = self.generate_expression_with_type(
                        tail_expression,
                        target_type,
                    )
                tail_value = self.normalize_assignment_rhs(
                    target_type,
                    tail_expression,
                    tail_value,
                    "=",
                )
                code += (
                    self.indent_multiline_expression(
                        tail_value,
                        body_indent,
                        indent_first=True,
                    )
                    + "\n"
                )
            elif statements:
                code += self.generate_statement(statements[-1], body_indent)
        finally:
            self.variable_types = saved_variable_types
            self.local_variable_names = saved_local_variable_names

        code += f"{indent_str}}}"
        return code

    def active_do_while_context(self):
        if not self.do_while_contexts:
            return None
        context = self.do_while_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def active_for_context(self):
        if not self.for_contexts:
            return None
        context = self.for_contexts[-1]
        if context["loop_depth"] == self.loop_depth:
            return context
        return None

    def get_variable_type(self, node):
        if hasattr(node, "var_type"):
            vtype = node.var_type
        elif hasattr(node, "vtype"):
            vtype = node.vtype
        else:
            return None

        if self.is_inferred_declaration_type(vtype):
            return None
        type_name = self.type_name_string(vtype)
        return self.resource_type_with_attributes(type_name, node)

    def is_inferred_declaration_type(self, type_name):
        if type_name is None:
            return True
        type_name = self.type_name_string(type_name)
        return type_name.strip() in {"", "None", "auto"}

    def variable_declaration_type(self, node, initial_value=None):
        declared_type = self.get_variable_type(node)
        if declared_type is not None:
            return declared_type

        inferred_type = self.expression_result_type(initial_value)
        if inferred_type is not None:
            return inferred_type
        if isinstance(initial_value, ArrayLiteralNode):
            # An initializer list is never a float: refuse rather than declare the binding with a wrong type.
            self.refuse_lowering(
                f"local '{getattr(node, 'name', '')}': the element type of its untyped initializer list "
                "cannot be inferred"
            )
        return "float"

    def statement_body_terminates_inner_loop(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def generate_array_declaration(self, node, indent=0):
        indent_str = "    " * indent
        element_type_name = self.convert_type_node_to_string(node.element_type)
        element_type = self.map_type(element_type_name)
        size = get_array_size_from_node(node)

        if size is None:
            rust_type = f"Vec<{element_type}>"
            source_type = f"{element_type_name}[]"
        else:
            rust_type = f"[{element_type}; {size}]"
            source_type = f"{element_type_name}[{size}]"

        self.register_variable_type(node.name, source_type, scope="local")
        initializer = self.local_array_default_initializer(rust_type)
        return (
            f"{indent_str}{self.local_let_keyword(node)} "
            f"{self.rust_identifier(node.name)}: "
            f"{rust_type} = {initializer};\n"
        )

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_condition_expression(
            node.condition if hasattr(node, "condition") else node.if_condition
        )
        code = f"{indent_str}if {condition} {{\n"

        if_body = getattr(node, "then_branch", getattr(node, "if_body", None))
        if hasattr(if_body, "statements"):
            for stmt in if_body.statements:
                code += self.generate_statement(stmt, indent + 1)
        elif isinstance(if_body, list):
            for stmt in if_body:
                code += self.generate_statement(stmt, indent + 1)

        code += f"{indent_str}}}"

        else_branch = getattr(node, "else_branch", None)
        if else_branch:
            if hasattr(else_branch, "__class__") and "If" in str(else_branch.__class__):
                # Generate else if by recursively generating the nested if with else if prefix
                elif_condition = self.generate_condition_expression(
                    else_branch.condition
                    if hasattr(else_branch, "condition")
                    else else_branch.if_condition
                )
                code += f" else if {elif_condition} {{\n"

                elif_body = getattr(
                    else_branch, "then_branch", getattr(else_branch, "if_body", None)
                )
                if hasattr(elif_body, "statements"):
                    for stmt in elif_body.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(elif_body, list):
                    for stmt in elif_body:
                        code += self.generate_statement(stmt, indent + 1)

                code += f"{indent_str}}}"

                nested_else = getattr(else_branch, "else_branch", None)
                if nested_else:
                    if hasattr(nested_else, "__class__") and "If" in str(
                        nested_else.__class__
                    ):
                        # Another else if - recursively handle
                        remaining_code = self.generate_if(nested_else, indent)
                        # Remove the "if" prefix and replace with "else if"
                        remaining_lines = remaining_code.split("\n")
                        if remaining_lines[0].strip().startswith("if "):
                            remaining_lines[0] = remaining_lines[0].replace(
                                "if ", " else if ", 1
                            )
                        code += "\n".join(
                            remaining_lines[1:]
                        )  # Skip first line as we already handled it
                    else:
                        # Final else clause
                        code += f" else {{\n"
                        if hasattr(nested_else, "statements"):
                            for stmt in nested_else.statements:
                                code += self.generate_statement(stmt, indent + 1)
                        elif isinstance(nested_else, list):
                            for stmt in nested_else:
                                code += self.generate_statement(stmt, indent + 1)
                        else:
                            code += self.generate_statement(nested_else, indent + 1)
                        code += f"{indent_str}}}"
            else:
                code += f" else {{\n"
                if hasattr(else_branch, "statements"):
                    for stmt in else_branch.statements:
                        code += self.generate_statement(stmt, indent + 1)
                elif isinstance(else_branch, list):
                    for stmt in else_branch:
                        code += self.generate_statement(stmt, indent + 1)
                else:
                    code += self.generate_statement(else_branch, indent + 1)
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # C scopes a for-init declaration to the loop, so the lowering is a block: { init; while cond { body; update; } }.
        inner = indent + 1
        inner_str = "    " * inner
        init_parts = []
        for init_stmt in self.statement_list(node.init):
            part = self.generate_statement(init_stmt, 0).strip()
            if part.endswith(";"):
                part = part[:-1]
            if part:
                init_parts.append(part)
        init = f";\n{inner_str}".join(init_parts)
        condition = (
            self.generate_condition_expression(node.condition)
            if node.condition is not None
            else ""
        )
        if not str(condition).strip():
            condition = "true"
        update = "; ".join(
            part
            for part in (
                str(self.generate_discarded_expression(item)).strip()
                for item in self.statement_list(node.update)
            )
            if part
        )

        code = f"{indent_str}{{\n"
        if init:
            code += f"{inner_str}{init};\n"
        body = ""
        self.loop_depth += 1
        self.for_contexts.append({"loop_depth": self.loop_depth, "update": update})
        label = self.enter_loop_label()
        try:
            for stmt in self.statement_list(node.body):
                body += self.generate_statement(stmt, inner + 1)
        finally:
            prefix = self.exit_loop_label(label)
            self.for_contexts.pop()
            self.loop_depth -= 1
        code += f"{inner_str}{prefix}while {condition} {{\n" + body

        if str(update).strip():
            code += f"{inner_str}    {update};\n"
        code += f"{inner_str}}}\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_for_in(self, node, indent):
        indent_str = "    " * indent
        pattern = getattr(node, "pattern", "item")
        if isinstance(pattern, str):
            pattern = self.rust_identifier(pattern)
        iterable = self.generate_for_in_iterable(getattr(node, "iterable", None))

        body = ""
        self.loop_depth += 1
        label = self.enter_loop_label()
        try:
            for stmt in self.statement_list(getattr(node, "body", [])):
                body += self.generate_statement(stmt, indent + 1)
        finally:
            prefix = self.exit_loop_label(label)
            self.loop_depth -= 1
        code = f"{indent_str}{prefix}for {pattern} in {iterable} {{\n" + body

        code += f"{indent_str}}}\n"
        return code

    def generate_for_in_iterable(self, iterable_node):
        if isinstance(iterable_node, RangeNode):
            start = self.generate_expression(iterable_node.start)
            end = self.generate_expression(iterable_node.end)
            operator = "..=" if iterable_node.inclusive else ".."
            return f"{start}{operator}{end}"

        iterable = self.generate_expression(iterable_node)
        return f"0..{iterable}"

    def generate_while(self, node, indent):
        indent_str = "    " * indent
        condition = self.generate_condition_expression(node.condition)

        body = ""
        self.loop_depth += 1
        label = self.enter_loop_label()
        try:
            for stmt in self.statement_list(node.body):
                body += self.generate_statement(stmt, indent + 1)
        finally:
            prefix = self.exit_loop_label(label)
            self.loop_depth -= 1
        code = f"{indent_str}{prefix}while {condition} {{\n" + body

        code += f"{indent_str}}}\n"
        return code

    def generate_loop(self, node, indent):
        indent_str = "    " * indent
        body = ""
        self.loop_depth += 1
        label = self.enter_loop_label()
        try:
            for stmt in self.statement_list(node.body):
                body += self.generate_statement(stmt, indent + 1)
        finally:
            prefix = self.exit_loop_label(label)
            self.loop_depth -= 1
        code = f"{indent_str}{prefix}loop {{\n" + body

        code += f"{indent_str}}}\n"
        return code

    def generate_do_while(self, node, indent):
        indent_str = "    " * indent
        break_flag = f"__cgl_do_break_{self.do_while_counter}"
        self.do_while_counter += 1
        condition = self.generate_condition_expression(node.condition)

        code = f"{indent_str}let mut {break_flag}: bool = false;\n"
        body = ""
        self.loop_depth += 1
        self.do_while_contexts.append(
            {"loop_depth": self.loop_depth, "break_flag": break_flag}
        )
        label = self.enter_loop_label()
        try:
            for stmt in self.statement_list(node.body):
                body += self.generate_statement(stmt, indent + 2)
        finally:
            prefix = self.exit_loop_label(label)
            self.do_while_contexts.pop()
            self.loop_depth -= 1
        code += f"{indent_str}loop {{\n"
        code += f"{indent_str}    {prefix}loop {{\n" + body

        if not self.statement_body_terminates_inner_loop(node.body):
            code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if {break_flag} {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}    if !({condition}) {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += f"{indent_str}}}\n"

        return code

    def generate_discarded_expression(self, expr):
        """Render an expression whose value is unused, such as a for update."""
        if isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        if isinstance(expr, UnaryOpNode):
            op = self.map_operator(getattr(expr, "operator", getattr(expr, "op", "")))
            if op in {"++", "--"}:
                return self.generate_increment_update(expr.operand, op)
        return self.generate_expression(expr)

    def generate_assignment_value(self, node):
        """Lower an assignment used as a value.

        A Rust assignment has type ``()``, so the stored value is read back
        from the target inside a block expression.
        """
        target = getattr(node, "target", getattr(node, "left", None))
        if not is_reevaluable_place(target):
            return self.refuse_lowering(
                "assignment used as a value: its target has side effects, so "
                "the stored value cannot be read back"
            )
        assignment = self.generate_assignment(node)
        return f"{{ {assignment}; {self.generate_expression(target)} }}"
