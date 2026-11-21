"""Statement and leaf-expression visitors of the CUDA generator: variables, assignments, identifiers, literals and control flow."""

from ..ast import (
    AssignmentNode,
    FunctionCallNode,
    MatchNode,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import (
    CudaRefusal,
    supersedes_pending_refusals,
)
from .assignment_values import assignment_value_expression
from .c_float_literals import c_float_literal_text
from .cuda_constant_folding import (
    device_constant_declaration,
    device_variable_declaration,
    is_device_variable,
)
from .cuda_hip_array_copies import generate_cpp_array_copy
from .for_initializers import join_for_initializer
from .gpu_device_globals import (
    at_file_scope,
    constant_in_device_memory,
    file_scope_placement,
)
from .gpu_subgroup_builtins import (
    reserved_builtin_reason,
    subgroup_builtin_expression,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)


class CudaStatementMixin:
    """Statement members of ``CudaCodeGen``."""

    def visit_VariableNode(self, node):
        var_type = self.get_variable_node_type(node)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if isinstance(initial_value, MatchNode):
            self.emit_match_expression_variable(node, initial_value, var_type)
            return None

        metadata_comment = self.cuda_resource_metadata_comment(node, var_type)
        if metadata_comment:
            self.emit(metadata_comment)
        place = file_scope_placement(self, node, metadata_comment)
        declaration = self.format_variable_declaration(node)
        if place is constant_in_device_memory and declaration != node.name:
            declaration = device_constant_declaration(
                self, node.name, var_type, initial_value, declaration, CudaRefusal
            )
        elif at_file_scope(self) and is_device_variable(node):
            declaration = device_variable_declaration(
                self, node.name, var_type, initial_value, declaration, CudaRefusal
            )
        if declaration != node.name:
            self.emit(f"{place(declaration)};")
            initial_value = getattr(
                node,
                "initial_value",
                getattr(node, "value", None),
            )
            effective_var_type = var_type
            if effective_var_type is None and initial_value is not None:
                effective_var_type = self.expression_result_type(initial_value)
            metadata_declaration = self.query_metadata_snapshot_declaration(
                node.name,
                effective_var_type,
                initial_value,
            )
            if metadata_declaration is None:
                metadata_declaration = self.query_metadata_declaration(
                    node.name,
                    effective_var_type,
                )
            if metadata_declaration:
                self.emit(f"{place(metadata_declaration)};")
            length_declaration = self.structured_buffer_length_declaration(
                node.name, var_type
            )
            if length_declaration:
                self.emit(f"{place(length_declaration)};")
            counter_declaration = self.structured_buffer_counter_declaration(
                node.name, var_type
            )
            if counter_declaration:
                self.emit(f"{place(counter_declaration)};")
            return None

        return node.name

    def emit_match_expression_variable(self, node, match_node, var_type):
        if var_type is None:
            try:
                var_type = infer_match_expression_result_type(self, match_node)
            except ValueError:
                var_type = "int"
        var_type = var_type or "auto"
        self.register_variable_type(node.name, var_type, node)

        self.emit(f"{self.format_typed_declarator(var_type, node.name)};")
        try:
            assignment = generate_match_expression_assignment(
                self,
                match_node,
                node.name,
                var_type,
                self.indent_level,
                "CUDA",
            )
        except ValueError as error:
            if supersedes_pending_refusals(error):
                raise
            assignment = self.generate_match_expression_diagnostic_assignment(
                node.name,
                var_type,
                self.cuda_match_error_reason(error),
                self.indent_level,
                "CUDA",
            )
        self.emit_generated_code(assignment)

    def visit_ExpressionStatementNode(self, node):
        if isinstance(node.expression, AssignmentNode):
            self.emit_assignment_statement(node.expression)
            return None

        expr = self.visit(node.expression)
        if expr and expr.strip():
            self.emit(f"{expr};")

    def emit_assignment_statement(self, node):
        operator = getattr(node, "operator", getattr(node, "op", "="))
        if operator == "=":
            target = getattr(node, "target", getattr(node, "left", None))
            value = getattr(node, "value", getattr(node, "right", None))
            array_copy = generate_cpp_array_copy(self, target, value)
            if array_copy is not None:
                self.emit_generated_code(array_copy)
                metadata_assignment = self.format_query_metadata_assignment(node)
                if metadata_assignment:
                    self.emit(f"{metadata_assignment};")
                return

        assignment = self.format_assignment_expression(node)
        if assignment and assignment.strip():
            self.emit(f"{assignment};")

        metadata_assignment = self.format_query_metadata_assignment(node)
        if metadata_assignment:
            self.emit(f"{metadata_assignment};")

    def format_for_clause_expression(self, node):
        """Return a for-clause expression, preserving CUDA metadata sidecars."""
        if node is None:
            return ""
        if isinstance(node, list):
            return join_for_initializer(
                [self.format_for_clause_expression(item) for item in node],
                [isinstance(item, VariableNode) for item in node],
                "cuda",
            )
        if isinstance(node, VariableNode):
            return self.format_variable_declaration(node)

        expr_node = getattr(node, "expression", node)
        if isinstance(expr_node, AssignmentNode):
            assignment = self.format_assignment_expression(expr_node)
            metadata_assignment = self.format_query_metadata_assignment(expr_node)
            parts = [
                part
                for part in (assignment, metadata_assignment)
                if part and part.strip()
            ]
            return ", ".join(parts)

        return self.visit(expr_node) or ""

    def visit_IdentifierNode(self, node):
        name = getattr(node, "name", str(node))
        builtin_alias = self.cuda_stage_builtin_alias_expression(name)
        if builtin_alias is not None:
            return builtin_alias
        if self.cuda_identifier_shadows_compute_builtin(name):
            return name
        ray_builtin = self.cuda_ray_builtin_expression(name)
        if ray_builtin is not None and name not in self.variable_types:
            return ray_builtin
        enum_constant = self.enum_variant_constants.get(name)
        if enum_constant is not None:
            return enum_constant
        return self.cuda_identifier_spelling(name)

    def cuda_identifier_spelling(self, name):
        """The CUDA spelling of an identifier: a mapped built-in, or a program name as it is declared.

        A name in the reserved ``gl_`` namespace that is neither mapped nor declared by the program is a built-in
        variable with no CUDA equivalent, and is refused (:mod:`.gpu_subgroup_builtins`).
        """
        if name in self.builtin_map:
            return self.builtin_map[name]
        if name not in self.variable_types:
            subgroup = subgroup_builtin_expression(
                name, self.cuda_warp_lane_index_expression()
            )
            if subgroup is not None:
                return subgroup
            reason = reserved_builtin_reason(name, "CUDA")
            if reason is not None:
                return self.refuse_lowering(reason, name)
        return name

    def cuda_ray_builtin_expression(self, name):
        helper_name = {
            "gl_LaunchIDEXT": "cgl_ray_launch_id",
            "gl_LaunchSizeEXT": "cgl_ray_launch_size",
            "gl_HitTEXT": "cgl_ray_hit_t",
            "gl_HitKindEXT": "cgl_ray_hit_kind",
            "gl_WorldRayOriginEXT": "cgl_ray_world_origin",
            "gl_WorldRayDirectionEXT": "cgl_ray_world_direction",
            "gl_ObjectRayOriginEXT": "cgl_ray_object_origin",
            "gl_ObjectRayDirectionEXT": "cgl_ray_object_direction",
            "gl_RayTminEXT": "cgl_ray_t_min",
            "gl_IncomingRayFlagsEXT": "cgl_ray_incoming_flags",
            "gl_InstanceCustomIndexEXT": "cgl_ray_instance_custom_index",
            "gl_GeometryIndexEXT": "cgl_ray_geometry_index",
        }.get(str(name))
        if helper_name is None:
            return None
        self.require_cuda_ray_runtime_helpers()
        return f"{helper_name}()"

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if literal_type == "float":
            float_text = c_float_literal_text(value)
            if float_text is not None:
                return float_text
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def visit_LiteralNode(self, node):
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_AssignmentNode(self, node):
        return assignment_value_expression(self.format_assignment_expression(node))

    def visit_IfNode(self, node):
        """Visit if statement"""
        condition = self.visit(node.condition)
        self.emit(f"if ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "then_branch"):
            self.emit_body(node.then_branch)
        elif hasattr(node, "if_body"):
            self.emit_body(node.if_body)

        self.indent_level -= 1

        if hasattr(node, "else_branch") and node.else_branch:
            self.emit("} else {")
            self.indent_level += 1

            self.emit_body(node.else_branch)

            self.indent_level -= 1
        elif hasattr(node, "else_body") and node.else_body:
            self.emit("} else {")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1

        self.emit("}")

    def visit_ForNode(self, node):
        """Visit for loop"""
        init_str = ""
        if node.init:
            init_str = self.format_for_clause_expression(node.init)

        condition_str = ""
        if node.condition:
            condition_str = self.visit(node.condition)

        update_str = ""
        if node.update:
            update_str = self.format_for_clause_expression(node.update)

        self.emit(f"for ({init_str}; {condition_str}; {update_str}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_ForInNode(self, node):
        """Lower CrossGL for-in loops to counted CUDA loops."""
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.visit(iterable.start)
            end = self.visit(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.visit(iterable)
            comparator = "<"

        self.emit(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern}) {{"
        )
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.emit("}")

    def visit_WhileNode(self, node):
        """Visit while loop"""
        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"while ({condition}) {{")

        self.indent_level += 1

        if hasattr(node, "body"):
            self.emit_body(node.body)

        self.indent_level -= 1
        self.emit("}")

    def visit_DoWhileNode(self, node):
        """Visit do-while loop."""
        self.emit("do {")

        self.indent_level += 1
        if hasattr(node, "body"):
            self.emit_body(node.body)
        self.indent_level -= 1

        condition = self.visit(node.condition) if node.condition else ""
        self.emit(f"}} while ({condition});")

    def visit_LoopNode(self, node):
        """Lower CrossGL infinite loops to CUDA while loops."""
        self.emit("while (true) {")

        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1

        self.emit("}")

    def visit_SwitchNode(self, node):
        """Visit switch statement"""
        expression = self.visit(node.expression)
        self.emit(f"switch ({expression}) {{")

        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1

        self.emit("}")

    def visit_MatchNode(self, node):
        """Lower CrossGL match statements to CUDA switch or if/else code."""
        try:
            if is_switch_lowerable_match(node):
                code = generate_switch_match(self, node, self.indent_level)
            else:
                code = generate_ordered_conditional_match(
                    self, node, self.indent_level, "CUDA"
                )
        except ValueError as error:
            if supersedes_pending_refusals(error):
                raise
            reason = self.cuda_match_error_reason(error)
            code = self.refuse_lowering(f"CUDA match statement: {reason}")
        self.emit_generated_code(code)

    def cuda_match_error_reason(self, error):
        reason = str(error)
        prefixes = (
            "Unsupported match expression for CUDA codegen; ",
            "Unsupported match arm for CUDA codegen; ",
        )
        for prefix in prefixes:
            if reason.startswith(prefix):
                return reason[len(prefix) :]
        return reason

    def generate_match_expression_diagnostic_assignment(
        self, target_variable, target_type, reason, indent, _target_name
    ):
        indent_str = "    " * indent
        fallback = self.refuse_lowering(
            f"CUDA match expression: {reason}",
            self.diagnostic_zero_value_for_type(target_type),
        )
        return f"{indent_str}{target_variable} = {fallback};\n"

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def statement_body_has_statements(self, body):
        return bool(self.statement_list(body))

    def generate_scoped_statement_body(self, body, indent):
        saved_output = self.output
        saved_indent = self.indent_level
        self.output = []
        self.indent_level = indent
        try:
            self.emit_body(body)
            if not self.output:
                return ""
            return "\n".join(self.output) + "\n"
        finally:
            self.output = saved_output
            self.indent_level = saved_indent

    def generate_expression(self, node):
        if node is None:
            return ""
        return self.visit(node)

    def generate_expression_with_expected(self, node, _expected_type):
        return self.generate_expression(node)

    def visit_CaseNode(self, node):
        """Visit switch case/default label"""
        if getattr(node, "value", None) is None:
            self.emit("default:")
        else:
            value = self.visit(node.value)
            self.emit(f"case {value}:")

        self.indent_level += 1
        for stmt in getattr(node, "statements", []):
            self.emit_statement(stmt)
        self.indent_level -= 1

    def visit_ReturnNode(self, node):
        """Visit return statement"""
        if self.current_function_is_kernel_entry:
            if node.value and not isinstance(node.value, MatchNode):
                if self.kernel_return_expression_may_have_side_effects(node.value):
                    self.emit_discarded_kernel_return_expression(node.value)
                else:
                    self.visit(node.value)
            self.emit("return;")
            return
        if node.value:
            if isinstance(node.value, MatchNode):
                return self.emit_match_expression_return(node.value)
            value = self.visit(node.value)
            self.emit(f"return {value};")
        else:
            self.emit("return;")

    def emit_discarded_kernel_return_expression(self, value_node):
        """Emit a non-void kernel return value when evaluating it can do work."""
        if isinstance(value_node, AssignmentNode):
            self.emit_assignment_statement(value_node)
            return

        value = self.visit(value_node)
        if isinstance(value, str) and value.strip():
            self.emit(f"{value};")

    def kernel_return_expression_may_have_side_effects(self, node):
        """Return whether discarding a kernel return expression would drop work."""
        if node is None:
            return False
        if isinstance(
            node,
            (
                AssignmentNode,
                FunctionCallNode,
                RayTracingOpNode,
                RayQueryOpNode,
            ),
        ):
            return True
        if isinstance(node, UnaryOpNode):
            operator = getattr(node, "operator", getattr(node, "op", ""))
            if operator in {"++", "--"}:
                return True

        child_attrs = (
            "left",
            "right",
            "operand",
            "condition",
            "true_expr",
            "false_expr",
            "array_expr",
            "array",
            "index_expr",
            "index",
            "object_expr",
            "object",
            "pointer_expr",
            "arguments",
            "args",
            "named_arguments",
        )
        for attr in child_attrs:
            child = getattr(node, attr, None)
            if isinstance(child, dict):
                children = child.values()
            elif isinstance(child, (list, tuple)):
                children = child
            else:
                children = (child,)
            if any(
                self.kernel_return_expression_may_have_side_effects(item)
                for item in children
            ):
                return True
        return False

    def emit_match_expression_return(self, match_node):
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            raise CudaRefusal(
                "Unsupported match expression for CUDA codegen; return context "
                "requires a concrete non-void result type"
            )

        result_name = self.next_cuda_temp_variable("match_value")
        self.emit(f"{self.format_typed_declarator(return_type, result_name)};")
        try:
            assignment = generate_match_expression_assignment(
                self,
                match_node,
                result_name,
                return_type,
                self.indent_level,
                "CUDA",
            )
        except ValueError as error:
            if supersedes_pending_refusals(error):
                raise
            assignment = self.generate_match_expression_diagnostic_assignment(
                result_name,
                return_type,
                self.cuda_match_error_reason(error),
                self.indent_level,
                "CUDA",
            )
        self.emit_generated_code(assignment)
        self.emit(f"return {result_name};")

    def visit_BreakNode(self, node):
        """Visit break statement"""
        self.emit("break;")

    def visit_ContinueNode(self, node):
        """Visit continue statement"""
        self.emit("continue;")

    def visit_BlockNode(self, node):
        """A nested block keeps its braces: a declaration in it ends with the block (C++17 [basic.scope.block])."""
        saved_types = dict(self.variable_types)
        self.emit("{")
        self.indent_level += 1
        try:
            self.emit_body(node.statements)
        finally:
            self.indent_level -= 1
            self.variable_types.clear()
            self.variable_types.update(saved_types)
        self.emit("}")
