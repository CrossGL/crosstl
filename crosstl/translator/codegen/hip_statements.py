"""Statement and leaf-expression visitors of the HIP generator: control flow, returns, literals and identifiers."""

from ..ast import (
    AssignmentNode,
    BreakNode,
    ContinueNode,
    MatchNode,
    RangeNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import HipRefusal
from .assignment_values import assignment_value_expression
from .c_float_literals import c_float_literal_text
from .enum_utils import enum_value_expression
from .for_initializers import join_for_initializer
from .gpu_subgroup_builtins import (
    reserved_builtin_reason,
    subgroup_builtin_expression,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    is_switch_lowerable_match,
)
from .resource_query import ResourceQueryMixin


class HipStatementMixin:
    """Statement members of ``HipCodeGen``."""

    def visit_list(self, node_list) -> str:
        for node in node_list:
            self.emit_statement(node)
        return ""

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        result = self.visit_discarded_expression(node)
        if isinstance(result, str) and result.strip():
            self.add_line(f"{result};")

    def visit_discarded_expression(self, node):
        """Render a node whose value is unused, such as a statement."""
        if isinstance(node, AssignmentNode):
            return self.format_assignment_expression(node)
        return self.visit(node)

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def statement_body_terminates(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def visit_IfNode(self, node) -> str:
        condition = self.visit(node.if_condition)
        self.add_line(f"if ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.if_body)
        self.indent_level -= 1
        self.add_line("}")

        if hasattr(node, "else_body") and node.else_body:
            self.add_line("else")
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1
            self.add_line("}")

        return ""

    def hip_for_clause(self, node):
        """One for-loop initializer or update clause; a list is one declaration or one comma expression."""
        if not node:
            return ""
        if isinstance(node, list):
            return join_for_initializer(
                [self.hip_for_clause(item) for item in node],
                [isinstance(item, VariableNode) for item in node],
                "hip",
            )
        if isinstance(node, VariableNode):
            return self.format_variable_declaration(node)
        if hasattr(node, "expression"):
            return self.visit_discarded_expression(node.expression)
        return self.visit_discarded_expression(node)

    def visit_ForNode(self, node) -> str:
        init = self.hip_for_clause(node.init)
        condition = self.visit(node.condition) if node.condition else ""
        update = self.hip_for_clause(node.update)

        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_ForInNode(self, node) -> str:
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

        self.add_line(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern})"
        )
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_WhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line(f"while ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_DoWhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line("do")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line(f"}} while ({condition});")

        return ""

    def visit_LoopNode(self, node) -> str:
        self.add_line("while (true)")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_SwitchNode(self, node) -> str:
        expression = self.visit(node.expression)

        self.add_line(f"switch ({expression})")
        self.add_line("{")
        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_MatchNode(self, node) -> str:
        if is_switch_lowerable_match(node):
            code = generate_switch_match(self, node, self.indent_level)
        else:
            code = generate_ordered_conditional_match(
                self, node, self.indent_level, "HIP"
            )
        self.add_generated_code(code)
        return ""

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
        saved_lines = self.code_lines
        saved_indent = self.indent_level
        self.code_lines = []
        self.indent_level = indent
        try:
            self.emit_body(body)
            if not self.code_lines:
                return ""
            return "\n".join(self.code_lines) + "\n"
        finally:
            self.code_lines = saved_lines
            self.indent_level = saved_indent

    def generate_expression(self, node):
        if node is None:
            return ""
        return self.visit(node)

    def generate_expression_with_expected(self, node, expected_type):
        saved_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(node)
        finally:
            self.current_expression_expected_type = saved_expected_type

    def visit_CaseNode(self, node) -> str:
        if getattr(node, "value", None) is None:
            self.add_line("default:")
        else:
            value = self.visit(node.value)
            self.add_line(f"case {value}:")

        self.indent_level += 1
        self.emit_body(getattr(node, "statements", []))
        self.indent_level -= 1

        return ""

    def visit_ReturnNode(self, node) -> str:
        if self.current_function_is_kernel_entry:
            if node.value and not isinstance(node.value, MatchNode):
                self.visit(node.value)
            self.add_line("return;")
            return ""
        if node.value:
            if isinstance(node.value, MatchNode):
                return self.emit_match_expression_return(node.value)
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def emit_match_expression_return(self, match_node) -> str:
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            raise HipRefusal(
                "Unsupported match expression for HIP codegen; return context "
                "requires a concrete non-void result type"
            )

        result_name = self.next_hip_temp_variable("match_value")
        self.add_line(f"{self.format_typed_declarator(return_type, result_name)};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                result_name,
                return_type,
                self.indent_level,
                "HIP",
            )
        )
        self.add_line(f"return {result_name};")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        return assignment_value_expression(self.format_assignment_expression(node))

    def visit_TernaryOpNode(self, node) -> str:
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

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

    def visit_LiteralNode(self, node) -> str:
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_IdentifierNode(self, node) -> str:
        name = getattr(node, "name", str(node))
        if name in getattr(self, "enum_variant_constants", {}):
            return enum_value_expression(self, name)
        builtin_alias = self.hip_stage_builtin_alias_expression(name)
        if builtin_alias is not None:
            return builtin_alias
        if not self.source_identifier_shadows_builtin(name):
            ray_builtin = self.hip_ray_builtin_expression(name)
            if ray_builtin is not None:
                return ray_builtin
            if name in self.builtin_map:
                return self.builtin_map[name]
            subgroup = subgroup_builtin_expression(name)
            if subgroup is not None:
                return subgroup
            reason = reserved_builtin_reason(name, "HIP")
            if reason is not None:
                return self.refuse_lowering(reason, name)
        # A program name, written as it is declared.
        return name

    def hip_ray_builtin_expression(self, name):
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
        self.require_hip_ray_runtime_helpers()
        return f"{helper_name}()"

    def visit_ExpressionStatementNode(self, node) -> str:
        expr = self.visit_discarded_expression(node.expression)
        self.add_line(f"{expr};")
        return ""

    def visit_BlockNode(self, node) -> str:
        """A nested block keeps its braces: a declaration in it ends with the block (C++17 [basic.scope.block])."""
        saved_types = dict(self.variable_types)
        self.add_line("{")
        self.indent_level += 1
        try:
            if hasattr(node, "statements"):
                self.emit_body(node.statements)
        finally:
            self.indent_level -= 1
            self.variable_types.clear()
            self.variable_types.update(saved_types)
        self.add_line("}")
        return ""

    def visit_BreakNode(self, node) -> str:
        self.add_line("break;")
        return ""

    def visit_ContinueNode(self, node) -> str:
        self.add_line("continue;")
        return ""

    def visit_EnumNode(self, node) -> str:
        self.add_line(f"enum {node.name}")
        self.add_line("{")
        self.indent_level += 1

        if hasattr(node, "variants") and node.variants:
            for i, variant in enumerate(node.variants):
                if hasattr(variant, "value") and variant.value:
                    value = self.visit(variant.value)
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name} = {value}")
                    else:
                        self.add_line(f"{variant.name} = {value},")
                else:
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name}")
                    else:
                        self.add_line(f"{variant.name},")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def get_parameter_type(self, param):
        """Return a parameter type with HIP resource access metadata applied."""
        param_type = ResourceQueryMixin.get_parameter_type(self, param)
        return self.resource_type_with_access(param_type, param)

    def get_variable_node_type(self, node):
        """Return a variable type with HIP resource access metadata applied."""
        var_type = ResourceQueryMixin.get_variable_node_type(self, node)
        return self.resource_type_with_access(var_type, node)
