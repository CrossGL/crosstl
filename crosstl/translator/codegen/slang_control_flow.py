"""Control flow of the Slang generator: lambdas, if, loops, switch and match."""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    AtomicOpNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CaseNode,
    ConstructorPatternNode,
    ContinueNode,
    ExpressionStatementNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    LiteralNode,
    LiteralPatternNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReferenceType,
    ReturnNode,
    StructPatternNode,
    TernaryOpNode,
    UnaryOpNode,
    WaveOpNode,
    WildcardPatternNode,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .slang_switch_cases import slang_case_needs_scope


class SlangControlFlowMixin:
    """Control-flow members of ``SlangCodeGen``."""

    def generate_lambda_expression(self, args):
        """Render supported CrossGL pseudo-lambdas as Slang lambda expressions."""
        if not args:
            return None

        params = []
        for arg in args[:-1]:
            param = self.generate_lambda_parameter(arg)
            if param is None:
                return None
            params.append(param)

        body = self.generate_lambda_body(args[-1])
        if body is None:
            return None
        return f"({', '.join(params)}) => {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            return None

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        if mapped_type is None:
            return None
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if not raw:
            return None
        return raw

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.generate_expression(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw or ":" in raw:
            return None
        if any(char in raw for char in "{}()"):
            return None

        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None

        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        canonical_type = (
            "".join(type_name.split())
            if "<" in type_name or ">" in type_name
            else type_name
        )
        if any(char.isspace() for char in canonical_type):
            return None
        if any(char in canonical_type for char in "{},;[]()"):
            return None

        mapped_type = self.convert_type(canonical_type)
        if "<" in canonical_type or ">" in canonical_type:
            if mapped_type == canonical_type:
                return None

        if any(char in mapped_type for char in "<>{},;[]()"):
            return None
        if not mapped_type:
            return None
        if not (mapped_type[0].isalpha() or mapped_type[0] == "_"):
            return None
        if not all(char.isalnum() or char == "_" for char in mapped_type):
            return None
        return mapped_type

    def format_array_access_index(self, index):
        if isinstance(index, BinaryOpNode):
            return self.format_array_size_expression(index)
        return self.generate_expression(index)

    def generate_if(self, node):
        condition_node = getattr(node, "condition", getattr(node, "if_condition", None))
        prelude, _results, condition = self.generate_expression_with_prelude(
            condition_node
        )
        condition = self.slang_precise_condition(condition_node, condition)
        result = f"if ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(getattr(node, "if_body", [])):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"

        else_body = getattr(node, "else_body", None)
        if else_body:
            result += "\nelse\n{\n"
            self.indent_level += 1
            for stmt in self.get_statements(else_body):
                result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
            result += self.indent() + "}"

        return self.statement_with_prelude(prelude, result)

    def generate_for(self, node):
        init = (
            self.generate_for_header_statement(
                node.init, "for-loop initializer context"
            )
            if node.init
            else ""
        )
        condition = (
            self.slang_precise_condition(
                node.condition,
                self.generate_loop_condition_expression(
                    node.condition, "for-loop condition context"
                ),
            )
            if node.condition
            else ""
        )
        update = (
            self.generate_for_header_statement(node.update, "for-loop update context")
            if node.update
            else ""
        )

        result = f"for ({init}; {condition}; {update})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_for_in(self, node):
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.generate_expression(iterable.start)
            end = self.generate_expression(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.generate_expression(iterable)
            comparator = "<"

        result = (
            f"for (int {pattern} = {start}; "
            f"{pattern} {comparator} {end}; ++{pattern})\n{{\n"
        )

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_while(self, node):
        condition = self.slang_precise_condition(
            node.condition,
            self.generate_loop_condition_expression(
                node.condition, "while-loop condition context"
            ),
        )
        result = f"while ({condition})\n{{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + "}"
        return result

    def generate_loop(self, node):
        """CrossGL's unconditional ``loop`` (also the loop a self tail call becomes) as ``for (;;)``."""
        if getattr(node, "label", None):
            return self.refuse_lowering(
                f"loop label '{node.label}': Slang output writes no loop labels"
            )
        result = "for (;;)\n{\n"
        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1
        result += self.indent() + "}"
        return result

    def generate_do_while(self, node):
        condition = self.slang_precise_condition(
            node.condition,
            self.generate_loop_condition_expression(
                node.condition, "do-while-loop condition context"
            ),
        )
        result = "do\n{\n"

        self.indent_level += 1
        for stmt in self.get_statements(node.body):
            result += self.emit_statement(stmt) + "\n"
        self.indent_level -= 1

        result += self.indent() + f"}} while ({condition});"
        return result

    def generate_switch(self, node):
        prelude, _results, expression = self.generate_expression_with_prelude(
            node.expression
        )
        expression = self.slang_precise_condition(node.expression, expression)
        result = f"switch ({expression})\n{{\n"

        self.indent_level += 1
        cases = list(getattr(node, "cases", []) or [])
        for position, case in enumerate(cases):
            if not isinstance(case, CaseNode):
                continue

            if case.value is None:
                result += self.indent() + "default:\n"
            else:
                case_value = self.generate_expression(case.value)
                result += self.indent() + f"case {case_value}:\n"

            self.indent_level += 1
            if slang_case_needs_scope(self, cases, position):
                # The case's own locals end with its braces (slang_switch_cases.py).
                block = BlockNode(list(self.get_statements(case.statements)))
                result += self.indent() + self.generate_block_statement(block) + "\n"
            else:
                for stmt in self.get_statements(case.statements):
                    result += self.emit_statement(stmt) + "\n"
            self.indent_level -= 1
        self.indent_level -= 1

        result += self.indent() + "}"
        return self.statement_with_prelude(prelude, result)

    def generate_match(self, node):
        indent = self.indent_level
        if is_switch_lowerable_match(node):
            return generate_switch_match(self, node, indent).rstrip()
        return generate_ordered_conditional_match(self, node, indent, "Slang").rstrip()

    def generate_match_expression(self, node):
        expected_type = self.type_name_string(self.current_expression_expected_type)
        if not expected_type:
            try:
                expected_type = infer_match_expression_result_type(self, node)
            except ValueError as error:
                return self.refuse_lowering(
                    "Slang match expression: "
                    f"{self.slang_match_expression_error_reason(error)}",
                    "0",
                )
        if not expected_type or expected_type in {"auto", "void"}:
            return self.refuse_lowering(
                "Slang match expression: requires typed result context", "0"
            )

        if not self.expression_prelude_active():
            return self.refuse_lowering(
                "Slang match expression: requires statement prelude context",
                self.zero_value_for_type(expected_type),
            )

        temp_name = self.unique_expression_temp_name("cgl_match_value")
        self.register_variable_type(temp_name, expected_type)
        try:
            if is_switch_lowerable_match(node):
                assignment = self.generate_switch_match_expression_assignment(
                    node, temp_name, expected_type
                )
            else:
                assignment = generate_match_expression_assignment(
                    self,
                    node,
                    temp_name,
                    expected_type,
                    self.indent_level,
                    "Slang",
                ).rstrip()
        except ValueError as error:
            return self.refuse_lowering(
                "Slang match expression: "
                f"{self.slang_match_expression_error_reason(error)}",
                self.zero_value_for_type(expected_type),
            )
        lines = [f"{self.format_declaration(expected_type, temp_name)};"]
        if assignment:
            lines.extend(assignment.splitlines())
        self.add_expression_prelude(lines, result_name=temp_name)
        return temp_name

    def slang_match_expression_error_reason(self, error):
        reason = str(error)
        prefixes = (
            "Unsupported match expression for Slang codegen; ",
            "Unsupported match arm for Slang codegen; ",
        )
        for prefix in prefixes:
            if reason.startswith(prefix):
                return reason[len(prefix) :]
        return reason

    def generate_match_expression_diagnostic_assignment(
        self, temp_name, expected_type, reason, indent, _target_name
    ):
        """Assign the zero value on the path no arm takes; refuse an arm that produces no value.

        The unmatched path is reached only by a subject the arms do not cover, which a well-formed value-position
        match excludes; Slang still needs the variable assigned on that path. An arm without a value is a malformed
        match, not an unreachable path.
        """
        indent_str = self.indent_str * indent
        if reason != "no wildcard arm handles remaining cases":
            self.refuse_lowering(f"Slang match expression: {reason}")
        return f"{indent_str}{temp_name} = {self.zero_value_for_type(expected_type)};\n"

    def generate_switch_match_expression_assignment(
        self, node, temp_name, expected_type
    ):
        indent = self.indent_level
        indent_str = self.indent_str * indent
        case_indent = self.indent_str * (indent + 1)
        body_indent = self.indent_str * (indent + 2)
        expression = self.generate_expression(getattr(node, "expression", None))
        lines = [
            f"{indent_str}switch ({expression})",
            f"{indent_str}{{",
        ]

        wildcard_body = None
        for arm in getattr(node, "arms", []) or []:
            pattern = getattr(arm, "pattern", None)
            body = getattr(arm, "body", [])
            if isinstance(pattern, WildcardPatternNode):
                wildcard_body = body
                continue

            lines.append(
                f"{case_indent}case {self.generate_expression(pattern.literal)}:"
            )
            lines.extend(
                self.generate_switch_match_expression_arm_lines(
                    body, temp_name, expected_type, indent + 2
                )
            )
            if not self.match_expression_arm_terminates(body):
                lines.append(f"{body_indent}break;")

        if wildcard_body is not None:
            lines.append(f"{case_indent}default:")
            lines.extend(
                self.generate_switch_match_expression_arm_lines(
                    wildcard_body, temp_name, expected_type, indent + 2
                )
            )
            if not self.match_expression_arm_terminates(wildcard_body):
                lines.append(f"{body_indent}break;")
        else:
            lines.append(f"{case_indent}default:")
            # Only a subject no arm covers reaches this path; see
            # generate_match_expression_diagnostic_assignment.
            lines.append(
                f"{body_indent}{temp_name} = "
                f"{self.zero_value_for_type(expected_type)};"
            )
            lines.append(f"{body_indent}break;")

        lines.append(f"{indent_str}}}")
        return "\n".join(lines)

    def generate_switch_match_expression_arm_lines(
        self, body, temp_name, expected_type, indent
    ):
        prefix, tail = self.match_expression_tail(body)
        lines = []
        for stmt in prefix:
            lines.extend(
                self.indent_text(self.generate_statement(stmt), indent).splitlines()
            )
        if tail is None:
            self.refuse_lowering("Slang match expression: arm does not produce a value")
            lines.append(
                f"{self.indent_str * indent}{temp_name} = "
                f"{self.zero_value_for_type(expected_type)};"
            )
            return lines

        prelude, _results, value = self.generate_expression_with_prelude(
            tail, expected_type
        )
        statement = self.statement_with_prelude(prelude, f"{temp_name} = {value};")
        lines.extend(self.indent_text(statement, indent).splitlines())
        return lines

    def generate_match_expression_arm_lines(self, body, temp_name, expected_type):
        prefix, tail = self.match_expression_tail(body)
        lines = []
        for stmt in prefix:
            lines.extend(self.emit_statement(stmt).splitlines())
        if tail is None:
            lines.extend(
                self.generate_match_expression_diagnostic_assignment_lines(
                    temp_name, expected_type, "arm does not produce a value"
                )
            )
            return lines

        prelude, _results, value = self.generate_expression_with_prelude(
            tail, expected_type
        )
        statement = self.statement_with_prelude(prelude, f"{temp_name} = {value};")
        lines.extend(self.indent_statement_text(statement).splitlines())
        return lines

    def generate_match_expression_diagnostic_assignment_lines(
        self, temp_name, expected_type, reason
    ):
        self.refuse_lowering(f"Slang match expression: {reason}")
        return [
            self.indent() + f"{temp_name} = {self.zero_value_for_type(expected_type)};"
        ]

    def match_expression_tail(self, body):
        if body is None:
            return [], None

        if isinstance(body, ExpressionStatementNode):
            if getattr(body, "is_tail_expression", False):
                return [], body.expression
            return [body], None

        if hasattr(body, "statements"):
            statements = list(body.statements)
            if not statements:
                return [], None

            tail = statements[-1]
            if isinstance(tail, ExpressionStatementNode) and getattr(
                tail, "is_tail_expression", False
            ):
                return statements[:-1], tail.expression
            return statements, None

        if isinstance(body, list):
            statements = body
        else:
            statements = [body]

        if not statements:
            return [], None

        tail = statements[-1]
        if isinstance(tail, ExpressionStatementNode) and getattr(
            tail, "is_tail_expression", False
        ):
            return statements[:-1], tail.expression
        if self.is_match_expression_value_node(tail):
            return statements[:-1], tail
        return statements, None

    def is_match_expression_value_node(self, node):
        return isinstance(
            node,
            (
                ArrayAccessNode,
                ArrayLiteralNode,
                AssignmentNode,
                AtomicOpNode,
                BinaryOpNode,
                FunctionCallNode,
                IdentifierNode,
                LiteralNode,
                MatchNode,
                MemberAccessNode,
                MeshOpNode,
                RayQueryOpNode,
                RayTracingOpNode,
                TernaryOpNode,
                UnaryOpNode,
                WaveOpNode,
            ),
        ) or isinstance(node, str)

    def match_expression_arm_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def is_supported_match_arm(self, arm):
        if getattr(arm, "guard", None) is not None:
            return False
        pattern = getattr(arm, "pattern", None)
        return isinstance(pattern, (LiteralPatternNode, WildcardPatternNode))

    def match_arm_rejection_reason(self, arms):
        wildcard_index = None
        for index, arm in enumerate(arms):
            if getattr(arm, "guard", None) is not None:
                return "guarded arms cannot be lowered to switch"

            pattern = getattr(arm, "pattern", None)
            if self.is_range_style_match_pattern(pattern):
                return "range-style patterns cannot be lowered to switch"

            if isinstance(pattern, WildcardPatternNode):
                if wildcard_index is not None:
                    return "multiple wildcard arms cannot be lowered to switch"
                wildcard_index = index
                continue

            if isinstance(pattern, LiteralPatternNode):
                continue

            if isinstance(pattern, IdentifierPatternNode):
                return "identifier binding patterns cannot be lowered to switch"

            if isinstance(pattern, ConstructorPatternNode):
                return "constructor patterns cannot be lowered to switch"

            if isinstance(pattern, StructPatternNode):
                return "struct destructuring patterns cannot be lowered to switch"

            return (
                "only unguarded literal patterns and a final wildcard can be "
                "lowered to switch"
            )

        if wildcard_index is not None and wildcard_index != len(arms) - 1:
            return "wildcard arm must be final"
        return None

    def pointer_pointee_type_name(self, vtype):
        if isinstance(vtype, PointerType):
            return self.type_name_string(vtype.pointee_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("*") else None

    def reference_referent_type_name(self, vtype):
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        return type_name[:-1].strip() if type_name.endswith("&") else None

    def member_lookup_type_name(self, vtype):
        return (
            self.pointer_pointee_type_name(vtype)
            or self.reference_referent_type_name(vtype)
            or vtype
        )

    def is_range_style_match_pattern(self, pattern):
        if isinstance(pattern, RangeNode):
            return True
        return isinstance(pattern, IdentifierPatternNode) and pattern.name == ".."

    def validate_match_arms(self, arms):
        return self.match_arm_rejection_reason(arms) is None

    def statement_body_terminates(self, body):
        statements = self.get_statements(body)
        if not statements:
            return False
        tail = statements[-1]
        return isinstance(tail, (BreakNode, ContinueNode, ReturnNode)) or (
            self.tail_expression_returns(tail)
        )

    def get_statements(self, body):
        if body is None:
            return []
        if hasattr(body, "statements"):
            return body.statements
        if isinstance(body, list):
            return body
        return [body]
