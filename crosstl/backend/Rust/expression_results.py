"""Rust block, if, match and loop expressions in value position, and their materialization into CrossGL statements."""

from .RustAst import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    AwaitNode,
    BinaryOpNode,
    BreakNode,
    CastNode,
    ConditionChainNode,
    ContinueNode,
    DereferenceNode,
    FunctionCallNode,
    IfNode,
    LabeledBlockNode,
    LetPatternConditionNode,
    LoopNode,
    MatchesMacroNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReferenceNode,
    ReturnNode,
    StructInitializationNode,
    TernaryOpNode,
    TryBlockNode,
    TupleNode,
    UnaryOpNode,
    VectorConstructorNode,
)


class RustExpressionResultMixin:
    """Value-position expression members of ``RustToCrossGLConverter``."""

    def generate_loop_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_loop(
            stmt.value,
            indent,
            result_target=stmt.name,
            loop_contexts=loop_contexts,
        )
        return code

    def generate_loop_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_loop(
            stmt.right,
            indent,
            result_target=target,
            loop_contexts=loop_contexts,
        )

    def generate_loop_expression_return(self, loop_node, indent, loop_contexts=None):
        return self.generate_loop(
            loop_node,
            indent,
            result_target=self.return_result_target,
            loop_contexts=loop_contexts,
        )

    def generate_labeled_block_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        target_name = self.declare_local_alias(stmt.name)
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(target_name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(stmt.vtype, target_name)};\n"
            )
        elif inferred_type:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(inferred_type, target_name)};\n"
            )
        else:
            code += f"{indent_str}auto {target_name};\n"

        code += self.generate_labeled_block_expression_result(
            stmt.value,
            indent,
            target_name,
            loop_contexts,
        )
        return code

    def generate_block_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        target_name = self.declare_local_alias(stmt.name)
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(target_name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(stmt.vtype, target_name)};\n"
            )
        elif inferred_type:
            code += (
                f"{indent_str}"
                f"{self.format_typed_declarator(inferred_type, target_name)};\n"
            )
        else:
            code += f"{indent_str}auto {target_name};\n"

        code += self.generate_block_expression_result(
            stmt.value,
            indent,
            target_name,
            loop_contexts,
        )
        return code

    def generate_block_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_block_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_block_expression_return(self, block_node, indent, loop_contexts=None):
        return self.generate_block_expression_result(
            block_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_if_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(stmt.name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_if_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_if_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_if_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_if_expression_return(self, if_node, indent, loop_contexts=None):
        return self.generate_if_expression_result(
            if_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_match_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        inferred_type = stmt.vtype or self.infer_value_type(stmt.value)
        if inferred_type:
            self.add_value_type(stmt.name, inferred_type)

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_match_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_match_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_match_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_match_expression_return(self, match_node, indent, loop_contexts=None):
        return self.generate_match_expression_result(
            match_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_matches_expression_let(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}{stmt.name};\n"

        code += self.generate_matches_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_matches_expression_assignment(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        return self.generate_matches_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_matches_expression_return(
        self, matches_node, indent, loop_contexts=None
    ):
        return self.generate_matches_expression_result(
            matches_node,
            indent,
            self.return_result_target,
            loop_contexts,
        )

    def generate_matches_expression_result(
        self,
        matches_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if result_target is self.return_result_target:
            result_name = self.next_matches_result_name()
            code = f"{indent_str}bool {result_name} = false;\n"
            code += self.generate_matches_pattern_assignment(
                matches_node,
                indent,
                result_name,
                loop_contexts,
            )
            code += f"{indent_str}return {result_name};\n"
            return code

        code = f"{indent_str}{result_target} = false;\n"
        code += self.generate_matches_pattern_assignment(
            matches_node,
            indent,
            result_target,
            loop_contexts,
        )
        return code

    def generate_matches_pattern_assignment(
        self,
        matches_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        subject, code = self.generate_let_else_subject(
            matches_node.expression,
            indent,
            loop_contexts,
        )

        def success(success_indent):
            success_indent_str = "    " * success_indent
            if matches_node.guard is None:
                return f"{success_indent_str}{result_target} = true;\n"

            guard_code, guard = self.generate_try_expression(
                matches_node.guard,
                success_indent,
                loop_contexts,
            )
            return (
                guard_code + f"{success_indent_str}if ({guard}) {{\n"
                f"{success_indent_str}    {result_target} = true;\n"
                f"{success_indent_str}}}\n"
            )

        code += self.generate_nested_pattern_match(
            subject,
            matches_node.pattern,
            indent,
            success,
        )
        return code

    def generate_matches_condition_flag(
        self,
        matches_node,
        indent,
        loop_contexts=None,
    ):
        result_name = self.next_matches_result_name()
        indent_str = "    " * indent
        code = f"{indent_str}bool {result_name} = false;\n"
        code += self.generate_matches_pattern_assignment(
            matches_node,
            indent,
            result_name,
            loop_contexts,
        )
        return result_name, code

    def generate_if_expression_result(
        self, if_node, indent, result_target, loop_contexts=None
    ):
        if isinstance(if_node.condition, ConditionChainNode):
            return self.generate_condition_chain_if_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        if isinstance(if_node.condition, MatchesMacroNode):
            return self.generate_matches_if_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        if isinstance(if_node.condition, LetPatternConditionNode):
            return self.generate_if_let_expression_result(
                if_node,
                indent,
                result_target,
                loop_contexts,
            )

        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            if_node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_result_branch(
            if_node.if_body, indent + 1, result_target, loop_contexts
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_result_branch(
                if_node.else_body, indent + 1, result_target, loop_contexts
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_condition_chain_if_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        def success(success_indent):
            return self.generate_result_branch(
                if_node.if_body,
                success_indent,
                result_target,
                loop_contexts,
            )

        def failure(failure_indent):
            return self.generate_result_branch(
                if_node.else_body,
                failure_indent,
                result_target,
                loop_contexts,
            )

        return self.generate_condition_chain_branch(
            if_node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_if_let_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        def success(success_indent):
            return self.generate_result_branch(
                if_node.if_body,
                success_indent,
                result_target,
                loop_contexts,
            )

        def failure(failure_indent):
            return self.generate_result_branch(
                if_node.else_body,
                failure_indent,
                result_target,
                loop_contexts,
            )

        return self.generate_let_pattern_condition_branch(
            if_node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_matches_if_expression_result(
        self,
        if_node,
        indent,
        result_target,
        loop_contexts=None,
    ):
        condition, code = self.generate_matches_condition_flag(
            if_node.condition,
            indent,
            loop_contexts,
        )
        indent_str = "    " * indent

        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_result_branch(
            if_node.if_body,
            indent + 1,
            result_target,
            loop_contexts,
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_result_branch(
                if_node.else_body,
                indent + 1,
                result_target,
                loop_contexts,
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_result_branch(self, branch, indent, result_target, loop_contexts=None):
        if self.is_block_expression_node(branch):
            return self.generate_block_expression_result(
                branch, indent, result_target, loop_contexts
            )
        if isinstance(branch, list):
            return self.generate_scoped_function_body(
                branch,
                indent,
                loop_contexts,
                allow_implicit_final_return=(
                    result_target is self.return_result_target
                ),
            )
        if branch is not None:
            return self.generate_expression_result(
                branch, indent, result_target, loop_contexts
            )
        return ""

    def generate_match_expression_result(
        self, match_node, indent, result_target, loop_contexts=None
    ):
        if self.match_requires_if_chain(match_node):
            return self.generate_match_if_chain(
                match_node,
                indent,
                lambda body, branch_indent, contexts: self.generate_result_branch(
                    body,
                    branch_indent,
                    result_target,
                    contexts,
                ),
                loop_contexts,
            )

        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            match_node.expression,
            indent,
            loop_contexts,
        )
        expression = self.generate_match_subject_expression(subject)
        switch_break_flag = self.create_switch_break_flag(match_node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        code += self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in match_node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_result_branch(
                arm.body,
                indent + 2,
                result_target,
                arm_loop_contexts,
            )
            if self.match_arm_needs_switch_terminator(arm.body):
                code += f"{indent_str}        break;\n"

        code += f"{indent_str}}}\n"
        code += self.generate_switch_break_propagation(
            switch_break_flag,
            loop_contexts,
            indent,
        )
        return code

    def generate_block_expression_result(
        self, block_node, indent, result_target, loop_contexts=None
    ):
        block_node = self.get_block_expression_node(block_node)
        indent_str = "    " * indent
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(
                block_node.statements, indent, loop_contexts
            )
            expression = self.get_block_expression(block_node)

            if isinstance(expression, LoopNode):
                code += self.generate_loop(
                    expression,
                    indent,
                    result_target=result_target,
                    loop_contexts=loop_contexts,
                )
            elif isinstance(expression, IfNode):
                code += self.generate_if_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif isinstance(expression, MatchNode):
                code += self.generate_match_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif expression is not None:
                code += self.generate_expression_result(
                    expression,
                    indent,
                    result_target,
                    loop_contexts,
                )
            elif (
                result_target is self.return_result_target
                and not self.branch_guarantees_control_transfer(block_node)
            ):
                code += f"{indent_str}return;\n"

            return code
        finally:
            self.pop_local_callable_scope()

    def generate_labeled_block_expression_result(
        self, node, indent, result_target, loop_contexts=None
    ):
        block_node = self.get_block_expression_node(node.body)
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(
            loop_contexts,
            node,
            result_target=result_target,
        )
        loop_context = nested_contexts[-1]

        self.push_local_callable_scope()
        try:
            code = self.generate_labeled_control_declarations(loop_context, indent)
            code += f"{indent_str}while (true) {{\n"
            code += self.generate_function_body(
                block_node.statements,
                indent + 1,
                nested_contexts,
            )

            expression = self.get_block_expression(block_node)
            if expression is not None:
                code += self.generate_expression_result(
                    expression,
                    indent + 1,
                    result_target,
                    nested_contexts,
                )
                if not self.branch_guarantees_control_transfer(expression):
                    code += f"{indent_str}    break;\n"
            elif not self.branch_guarantees_control_transfer(block_node):
                code += f"{indent_str}    break;\n"

            code += f"{indent_str}}}\n"
            return code
        finally:
            self.pop_local_callable_scope()

    def generate_expression_result(
        self, expression, indent, result_target, loop_contexts=None
    ):
        indent_str = "    " * indent

        if isinstance(expression, ReturnNode):
            if expression.value is None:
                return f"{indent_str}return;\n"
            return self.generate_expression_result(
                expression.value,
                indent,
                self.return_result_target,
                loop_contexts,
            )
        if isinstance(expression, BreakNode):
            return self.generate_break_statement(expression, indent, loop_contexts)
        if isinstance(expression, ContinueNode):
            return self.generate_continue_statement(expression, indent, loop_contexts)
        if isinstance(expression, LabeledBlockNode):
            return self.generate_labeled_block_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, LoopNode):
            return self.generate_loop(
                expression,
                indent,
                result_target=result_target,
                loop_contexts=loop_contexts,
            )
        if isinstance(expression, TryBlockNode):
            return self.generate_try_block_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, IfNode):
            return self.generate_if_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, MatchNode):
            return self.generate_match_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if self.is_block_expression_node(expression):
            return self.generate_block_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )
        if isinstance(expression, AssignmentNode):
            return self.generate_assignment_expression_result(
                expression,
                indent,
                result_target,
                loop_contexts,
            )

        if result_target is self.return_result_target:
            option_value = self.generate_option_unwrap_return_expression(expression)
            if option_value is not None:
                return f"{indent_str}return {option_value};\n"

        if self.expression_contains_try(expression):
            prelude, value = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            if result_target is self.return_result_target:
                return prelude + f"{indent_str}return {value};\n"
            if result_target:
                return prelude + f"{indent_str}{result_target} = {value};\n"
            if value:
                return prelude + f"{indent_str}{value};\n"
            return prelude

        if self.expression_contains_inline_result_expression(expression):
            prelude, value = self.generate_materialized_expression(
                expression,
                indent,
                loop_contexts,
            )
            if result_target is self.return_result_target:
                return prelude + f"{indent_str}return {value};\n"
            if result_target:
                return prelude + f"{indent_str}{result_target} = {value};\n"
            if value:
                return prelude + f"{indent_str}{value};\n"
            return prelude

        if result_target is self.return_result_target or result_target:
            value = self.generate_reparseable_expression(expression)
        else:
            value = self.generate_expression(expression)
        if result_target is self.return_result_target:
            return f"{indent_str}return {value};\n"
        if result_target:
            return f"{indent_str}{result_target} = {value};\n"
        if value:
            return f"{indent_str}{value};\n"
        return ""

    def is_inline_result_expression(self, expression):
        if isinstance(
            expression,
            (LoopNode, LabeledBlockNode, IfNode, MatchNode, AssignmentNode),
        ):
            return True

        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            if block_node is None:
                return False
            if block_node.statements:
                return True
            return self.expression_contains_inline_result_expression(
                self.get_block_expression(block_node)
            )

        return False

    def expression_contains_inline_result_expression(self, expression):
        if expression is None:
            return False

        if self.is_inline_result_expression(expression):
            return True

        if isinstance(expression, BinaryOpNode):
            return self.expression_contains_inline_result_expression(
                expression.left
            ) or self.expression_contains_inline_result_expression(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_contains_inline_result_expression(expression.operand)
        if isinstance(expression, FunctionCallNode):
            return self.expression_contains_inline_result_expression(
                expression.name
            ) or any(
                self.expression_contains_inline_result_expression(arg)
                for arg in expression.args
            )
        if isinstance(expression, MemberAccessNode):
            return self.expression_contains_inline_result_expression(expression.object)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_contains_inline_result_expression(
                expression.array
            ) or self.expression_contains_inline_result_expression(expression.index)
        if isinstance(expression, RangeNode):
            return self.expression_contains_inline_result_expression(
                expression.start
            ) or self.expression_contains_inline_result_expression(expression.end)
        if isinstance(expression, VectorConstructorNode):
            return any(
                self.expression_contains_inline_result_expression(arg)
                for arg in expression.args
            )
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_contains_inline_result_expression(expression.condition)
                or self.expression_contains_inline_result_expression(
                    expression.true_expr
                )
                or self.expression_contains_inline_result_expression(
                    expression.false_expr
                )
            )
        if isinstance(expression, AwaitNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, CastNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, ReferenceNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, DereferenceNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, TupleNode):
            return any(
                self.expression_contains_inline_result_expression(elem)
                for elem in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_contains_inline_result_expression(elem)
                for elem in expression.elements
            ) or self.expression_contains_inline_result_expression(expression.size)
        if isinstance(expression, StructInitializationNode):
            fields = (
                expression.fields.items()
                if isinstance(expression.fields, dict)
                else expression.fields
            )
            return any(
                self.expression_contains_inline_result_expression(field_value)
                for _, field_value in fields
            )
        if isinstance(expression, MatchesMacroNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            ) or self.expression_contains_inline_result_expression(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_contains_inline_result_expression(
                expression.expression
            )
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_contains_inline_result_expression(operand)
                for operand in expression.operands
            )

        return False

    def generate_materialized_expression(self, expression, indent, loop_contexts=None):
        if self.is_inline_result_expression(expression):
            return self.materialize_inline_result_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, BinaryOpNode):
            if expression.op in {"&&", "||"}:
                return self.generate_materialized_logical_expression(
                    expression,
                    indent,
                    loop_contexts,
                )
            left_code, left = self.generate_materialized_expression(
                expression.left,
                indent,
                loop_contexts,
            )
            right_code, right = self.generate_materialized_expression(
                expression.right,
                indent,
                loop_contexts,
            )
            return left_code + right_code, f"({left} {expression.op} {right})"

        if isinstance(expression, UnaryOpNode):
            code, operand = self.generate_materialized_expression(
                expression.operand,
                indent,
                loop_contexts,
            )
            return code, f"({expression.op}{operand})"

        if isinstance(expression, FunctionCallNode):
            return self.generate_materialized_function_call(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, MemberAccessNode):
            code, obj = self.generate_materialized_expression(
                expression.object,
                indent,
                loop_contexts,
            )
            member_name = self.resolve_member_access_name(obj, expression.member)
            return code, f"{obj}.{member_name}"

        if isinstance(expression, ArrayAccessNode):
            array_code, array = self.generate_materialized_expression(
                expression.array,
                indent,
                loop_contexts,
            )
            index_code, index = self.generate_materialized_expression(
                expression.index,
                indent,
                loop_contexts,
            )
            return array_code + index_code, f"{array}[{index}]"

        if isinstance(expression, RangeNode):
            start_code, start = self.generate_materialized_optional_expression(
                expression.start,
                indent,
                loop_contexts,
            )
            end_code, end = self.generate_materialized_optional_expression(
                expression.end,
                indent,
                loop_contexts,
            )
            return start_code + end_code, self.format_range_expression(
                start,
                end,
                expression.inclusive,
            )

        if isinstance(expression, VectorConstructorNode):
            code, args = self.generate_materialized_expression_list(
                expression.args,
                indent,
                loop_contexts,
            )
            type_name = self.map_type(expression.type_name)
            return code, f"{type_name}({', '.join(args)})"

        if isinstance(expression, TernaryOpNode):
            return self.generate_materialized_ternary_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, AwaitNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, CastNode):
            code, value = self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )
            return code, self.format_cast_expression(expression.target_type, value)

        if isinstance(expression, ReferenceNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, DereferenceNode):
            return self.generate_materialized_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, TupleNode):
            code, elements = self.generate_materialized_expression_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"({', '.join(elements)})"

        if isinstance(expression, ArrayNode):
            code, elements = self.generate_materialized_expression_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"{{{', '.join(elements)}}}"

        if isinstance(expression, StructInitializationNode):
            code = ""
            fields = []
            field_items = (
                expression.fields.items()
                if isinstance(expression.fields, dict)
                else expression.fields
            )
            for field_name, field_value in field_items:
                field_code, value = self.generate_materialized_expression(
                    field_value,
                    indent,
                    loop_contexts,
                )
                code += field_code
                fields.append(f"{field_name}: {value}")
            struct_name = self.format_struct_initialization_name(expression.struct_name)
            return code, f"{struct_name} {{ {', '.join(fields)} }}"

        return "", self.generate_expression(expression)

    def generate_materialized_optional_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        if expression is None:
            return "", ""
        return self.generate_materialized_expression(expression, indent, loop_contexts)

    def generate_materialized_expression_list(
        self,
        expressions,
        indent,
        loop_contexts=None,
    ):
        code = ""
        values = []
        for expression in expressions:
            expression_code, value = self.generate_materialized_expression(
                expression,
                indent,
                loop_contexts,
            )
            code += expression_code
            values.append(value)
        return code, values

    def materialize_inline_result_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            if not block_node.statements:
                block_expression = self.get_block_expression(block_node)
                if block_expression is None:
                    return "", ""
                return self.generate_materialized_expression(
                    block_expression,
                    indent,
                    loop_contexts,
                )

        value_name = self.next_inline_expression_value_name()
        code = f"{indent_str}auto {value_name};\n"
        code += self.generate_expression_result(
            expression,
            indent,
            value_name,
            loop_contexts,
        )
        return code, value_name

    def generate_materialized_logical_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        left_code, left = self.generate_materialized_expression(
            expression.left,
            indent,
            loop_contexts,
        )
        result_name = self.next_inline_expression_value_name()
        code = left_code
        code += f"{indent_str}auto {result_name};\n"

        if expression.op == "||":
            code += f"{indent_str}if ({left}) {{\n"
            code += f"{indent_str}    {result_name} = true;\n"
            code += f"{indent_str}}} else {{\n"
        else:
            code += f"{indent_str}if (!{left}) {{\n"
            code += f"{indent_str}    {result_name} = false;\n"
            code += f"{indent_str}}} else {{\n"

        right_code, right = self.generate_materialized_expression(
            expression.right,
            indent + 1,
            loop_contexts,
        )
        code += right_code
        code += f"{indent_str}    {result_name} = {right};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_materialized_ternary_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        condition_code, condition = self.generate_materialized_expression(
            expression.condition,
            indent,
            loop_contexts,
        )
        result_name = self.next_inline_expression_value_name()
        code = condition_code
        code += f"{indent_str}auto {result_name};\n"
        code += f"{indent_str}if ({condition}) {{\n"
        true_code, true_expr = self.generate_materialized_expression(
            expression.true_expr,
            indent + 1,
            loop_contexts,
        )
        code += true_code
        code += f"{indent_str}    {result_name} = {true_expr};\n"
        code += f"{indent_str}}} else {{\n"
        false_code, false_expr = self.generate_materialized_expression(
            expression.false_expr,
            indent + 1,
            loop_contexts,
        )
        code += false_code
        code += f"{indent_str}    {result_name} = {false_expr};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_materialized_function_call(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        if isinstance(expression.name, MemberAccessNode):
            method_name = expression.name.member
            if method_name == "unwrap_or" and len(expression.args) == 1:
                option_unwrap = self.generate_materialized_option_unwrap_or_call(
                    expression,
                    indent,
                    loop_contexts,
                )
                if option_unwrap is not None:
                    return option_unwrap

        args_code, args = self.generate_materialized_expression_list(
            expression.args,
            indent,
            loop_contexts,
        )

        if isinstance(expression.name, MemberAccessNode):
            obj_code, obj = self.generate_materialized_expression(
                expression.name.object,
                indent,
                loop_contexts,
            )
            method_call = self.format_method_call_parts(
                expression.name.member,
                obj,
                args,
                expression.args,
                self.infer_value_type(expression.name.object),
            )
            if method_call is not None:
                return obj_code + args_code, method_call
            return (
                obj_code + args_code,
                f"{obj}.{expression.name.member}({', '.join(args)})",
            )

        if isinstance(expression.name, str):
            constructor = self.format_path_constructor_call_parts(
                expression.name,
                args,
                expression.args,
            )
            if constructor is not None:
                return args_code, constructor
            associated_call = self.format_associated_impl_function_call(
                expression.name,
                args,
            )
            if associated_call is not None:
                return args_code, associated_call
            primitive_bitcast_call = (
                self.format_primitive_float_bitcast_associated_call(
                    expression.name,
                    args,
                )
            )
            if primitive_bitcast_call is not None:
                return args_code, primitive_bitcast_call
            expression_call = self.format_unary_expression_intrinsic_call(
                expression.name,
                args,
            )
            if expression_call is not None:
                return args_code, expression_call
            func_name = self.map_function(expression.name)
            return args_code, f"{func_name}({', '.join(args)})"

        name_code, func_name = self.generate_materialized_expression(
            expression.name,
            indent,
            loop_contexts,
        )
        return name_code + args_code, f"{func_name}({', '.join(args)})"

    def generate_materialized_option_unwrap_or_call(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        receiver = expression.name.object
        if not isinstance(receiver, FunctionCallNode):
            return None
        if not isinstance(receiver.name, str):
            return None

        helper_name = self.ensure_option_unwrap_or_helper(receiver.name)
        if helper_name is None:
            return None

        receiver_code, receiver_args = self.generate_materialized_expression_list(
            receiver.args,
            indent,
            loop_contexts,
        )
        fallback_code, fallback = self.generate_materialized_expression(
            expression.args[0],
            indent,
            loop_contexts,
        )
        args = receiver_args + [fallback]
        return receiver_code + fallback_code, f"{helper_name}({', '.join(args)})"
