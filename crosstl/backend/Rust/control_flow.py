"""Rust if, for, while and loop statements, including let chains and iterator ranges, as CrossGL."""

import re

from .RustAst import (
    ConditionChainNode,
    FunctionCallNode,
    IfNode,
    LetPatternConditionNode,
    MatchesMacroNode,
    MemberAccessNode,
    RangeNode,
    TryNode,
    TupleNode,
)


class RustControlFlowMixin:
    """Branch and loop statement members of ``RustToCrossGLConverter``."""

    def get_block_expression(self, block_node):
        return getattr(
            block_node, "expression", getattr(block_node, "returns_value", None)
        )

    def generate_assignment(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_reparseable_expression(node.right)
        return f"{left} {node.operator} {right}"

    def generate_if_statement(self, node, indent, loop_contexts=None):
        if isinstance(node.condition, ConditionChainNode):
            return self.generate_condition_chain_if_statement(
                node,
                indent,
                loop_contexts,
            )

        if isinstance(node.condition, MatchesMacroNode):
            return self.generate_matches_if_statement(node, indent, loop_contexts)

        if isinstance(node.condition, LetPatternConditionNode):
            return self.generate_if_let_statement(node, indent, loop_contexts)

        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.if_body, indent + 1, loop_contexts
        )
        code += f"{indent_str}}}"

        if node.else_body:
            if (
                isinstance(node.else_body, list)
                and len(node.else_body) == 1
                and isinstance(node.else_body[0], IfNode)
            ):
                code += " else "
                code += self.generate_if_statement(
                    node.else_body[0], 0, loop_contexts
                ).lstrip()
            else:
                code += " else {\n"
                if isinstance(node.else_body, list):
                    code += self.generate_scoped_function_body(
                        node.else_body, indent + 1, loop_contexts
                    )
                else:
                    code += self.generate_scoped_function_body(
                        [node.else_body], indent + 1, loop_contexts
                    )
                code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_condition_chain_if_statement(self, node, indent, loop_contexts=None):
        def success(success_indent):
            return self.generate_scoped_function_body(
                node.if_body,
                success_indent,
                loop_contexts,
            )

        failure = None
        if node.else_body is not None:

            def failure(failure_indent):
                if isinstance(node.else_body, list):
                    return self.generate_scoped_function_body(
                        node.else_body,
                        failure_indent,
                        loop_contexts,
                    )
                return self.generate_scoped_function_body(
                    [node.else_body],
                    failure_indent,
                    loop_contexts,
                )

        return self.generate_condition_chain_branch(
            node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_matches_if_statement(self, node, indent, loop_contexts=None):
        condition, code = self.generate_matches_condition_flag(
            node.condition,
            indent,
            loop_contexts,
        )
        indent_str = "    " * indent

        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.if_body, indent + 1, loop_contexts
        )
        code += f"{indent_str}}}"

        if node.else_body is not None:
            code += " else {\n"
            if isinstance(node.else_body, list):
                code += self.generate_scoped_function_body(
                    node.else_body,
                    indent + 1,
                    loop_contexts,
                )
            else:
                code += self.generate_scoped_function_body(
                    [node.else_body],
                    indent + 1,
                    loop_contexts,
                )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_if_let_statement(self, node, indent, loop_contexts=None):
        def success(success_indent):
            return self.generate_scoped_function_body(
                node.if_body,
                success_indent,
                loop_contexts,
            )

        failure = None
        if node.else_body is not None:

            def failure(failure_indent):
                if isinstance(node.else_body, list):
                    return self.generate_scoped_function_body(
                        node.else_body,
                        failure_indent,
                        loop_contexts,
                    )
                return self.generate_scoped_function_body(
                    [node.else_body],
                    failure_indent,
                    loop_contexts,
                )

        return self.generate_let_pattern_condition_branch(
            node.condition,
            indent,
            success,
            failure,
            loop_contexts,
        )

    def generate_let_pattern_condition_branch(
        self,
        condition,
        indent,
        success,
        failure=None,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            condition.expression,
            indent,
            loop_contexts,
        )

        if failure is None:
            code += self.generate_nested_pattern_match(
                subject,
                condition.pattern,
                indent,
                success,
            )
            return code

        matched_flag = self.next_match_chain_flag_name()
        code += f"{indent_str}bool {matched_flag} = false;\n"

        def matched_success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n" + success(
                success_indent
            )

        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_nested_pattern_match(
            subject,
            condition.pattern,
            indent + 1,
            matched_success,
        )
        code += f"{indent_str}}}\n"
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += failure(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_condition_chain_branch(
        self,
        chain,
        indent,
        success,
        failure=None,
        loop_contexts=None,
    ):
        if failure is None:
            return self.generate_condition_chain_operands(
                chain.operands,
                0,
                indent,
                success,
                loop_contexts,
            )

        matched_flag = self.next_match_chain_flag_name()
        indent_str = "    " * indent
        code = f"{indent_str}bool {matched_flag} = false;\n"

        def matched_success(success_indent):
            success_indent_str = "    " * success_indent
            return f"{success_indent_str}{matched_flag} = true;\n" + success(
                success_indent
            )

        code += self.generate_condition_chain_operands(
            chain.operands,
            0,
            indent,
            matched_success,
            loop_contexts,
        )
        code += f"{indent_str}if (!{matched_flag}) {{\n"
        code += failure(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_condition_chain_operands(
        self,
        operands,
        index,
        indent,
        success,
        loop_contexts=None,
    ):
        if index >= len(operands):
            return success(indent)

        operand = operands[index]

        def continue_chain(next_indent):
            return self.generate_condition_chain_operands(
                operands,
                index + 1,
                next_indent,
                success,
                loop_contexts,
            )

        return self.generate_condition_chain_operand(
            operand,
            indent,
            continue_chain,
            loop_contexts,
        )

    def generate_condition_chain_operand(
        self,
        operand,
        indent,
        success,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        if isinstance(operand, LetPatternConditionNode):
            subject, code = self.generate_match_subject(
                operand.expression,
                indent,
                loop_contexts,
            )
            code += self.generate_nested_pattern_match(
                subject,
                operand.pattern,
                indent,
                success,
            )
            return code

        if isinstance(operand, MatchesMacroNode):
            condition, code = self.generate_matches_condition_flag(
                operand,
                indent,
                loop_contexts,
            )
            code += f"{indent_str}if ({condition}) {{\n"
            code += success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        condition_code, condition = self.generate_try_expression(
            operand,
            indent,
            loop_contexts,
        )
        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += success(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_for_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        # Convert Rust for-in loop to C-style for loop
        code = self.generate_labeled_control_declarations(loop_context, indent)
        enumerate_loop = self.generate_enumerate_for_loop(
            node,
            indent,
            loop_contexts,
            nested_contexts,
        )
        if enumerate_loop is not None:
            return code + enumerate_loop

        pattern = self.generate_for_loop_pattern(node.pattern)
        code += self.generate_for_loop_header(
            pattern,
            node.iterable,
            indent,
            loop_contexts,
        )
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_for_loop_pattern(self, pattern):
        if isinstance(pattern, str) and not self.is_discard_pattern(pattern):
            return pattern
        return self.next_for_loop_index_name()

    def next_for_loop_index_name(self):
        name = f"_for_index_{self.for_loop_index_counter}"
        self.for_loop_index_counter += 1
        return name

    def next_for_loop_step_name(self):
        name = f"_for_step_{self.for_loop_step_counter}"
        self.for_loop_step_counter += 1
        return name

    def next_for_loop_bound_name(self):
        name = f"_for_bound_{self.for_loop_bound_counter}"
        self.for_loop_bound_counter += 1
        return name

    def next_for_loop_iterable_name(self):
        name = f"_for_iterable_{self.for_loop_iterable_counter}"
        self.for_loop_iterable_counter += 1
        return name

    def generate_enumerate_for_loop(
        self,
        node,
        indent,
        loop_contexts=None,
        nested_contexts=None,
    ):
        enumerate_parts = self.parse_for_enumerate_iterable(node.pattern, node.iterable)
        if enumerate_parts is None:
            return None

        index_pattern, value_pattern, collection = enumerate_parts
        indent_str = "    " * indent

        collection_setup, collection_expr = self.generate_for_loop_iterable_expression(
            collection,
            indent,
            loop_contexts,
        )
        index_name = self.generate_enumerate_index_name(index_pattern)

        code = collection_setup
        code += (
            f"{indent_str}for (int {index_name} = 0; "
            f"{index_name} < {collection_expr}.length(); {index_name}++) {{\n"
        )
        code += self.generate_enumerate_value_binding(
            value_pattern,
            collection_expr,
            index_name,
            indent + 1,
        )
        code += self.generate_scoped_function_body(
            node.body,
            indent + 1,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"
        return code

    def parse_for_enumerate_iterable(self, pattern, iterable):
        if not (
            isinstance(pattern, TupleNode)
            and len(pattern.elements) == 2
            and isinstance(iterable, FunctionCallNode)
            and isinstance(iterable.name, MemberAccessNode)
            and iterable.name.member == "enumerate"
            and not iterable.args
        ):
            return None

        iterator = iterable.name.object
        if not (
            isinstance(iterator, FunctionCallNode)
            and isinstance(iterator.name, MemberAccessNode)
            and iterator.name.member in {"iter", "iter_mut", "into_iter"}
            and not iterator.args
        ):
            return None

        return pattern.elements[0], pattern.elements[1], iterator.name.object

    def generate_for_loop_iterable_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        setup, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )

        if setup:
            if isinstance(expression, TryNode):
                return setup, value
            name = self.next_for_loop_iterable_name()
            return setup + f"{indent_str}auto {name} = {value};\n", name

        if self.expression_has_side_effects(expression):
            name = self.next_for_loop_iterable_name()
            return f"{indent_str}auto {name} = {value};\n", name

        return "", value

    def generate_enumerate_index_name(self, index_pattern):
        if isinstance(index_pattern, str) and not self.is_discard_pattern(
            index_pattern
        ):
            return index_pattern
        return self.next_for_loop_index_name()

    def generate_enumerate_value_binding(
        self,
        value_pattern,
        collection_expr,
        index_name,
        indent,
    ):
        if not isinstance(value_pattern, str) or self.is_discard_pattern(value_pattern):
            return ""

        indent_str = "    " * indent
        return f"{indent_str}auto {value_pattern} = {collection_expr}[{index_name}];\n"

    def generate_for_loop_header(self, pattern, iterable, indent, loop_contexts=None):
        indent_str = "    " * indent
        (
            range_node,
            step_expression,
            is_reverse,
            align_reverse_step,
        ) = self.parse_for_range_iterable(iterable)

        if range_node is not None:
            start_setup, start = self.generate_for_loop_setup_expression(
                range_node.start if range_node.start is not None else "0",
                indent,
                self.next_for_loop_bound_name,
                loop_contexts,
            )
            setup = start_setup
            condition = ""
            step = None

            if range_node.end is not None:
                end_setup, end = self.generate_for_loop_setup_expression(
                    range_node.end,
                    indent,
                    self.next_for_loop_bound_name,
                    loop_contexts,
                )
                setup += end_setup
                if step_expression is not None:
                    step_setup, step = self.generate_for_loop_setup_expression(
                        step_expression,
                        indent,
                        self.next_for_loop_step_name,
                        loop_contexts,
                    )
                    setup += step_setup

                if is_reverse:
                    if step is not None and align_reverse_step:
                        initial_value = self.format_reverse_stepped_range_start(
                            start,
                            end,
                            step,
                            range_node.inclusive,
                        )
                    else:
                        initial_value = (
                            end
                            if range_node.inclusive
                            else self.format_reverse_exclusive_range_start(end)
                        )
                    condition = f"{pattern} >= {start}"
                else:
                    initial_value = start
                    comparison = "<=" if range_node.inclusive else "<"
                    condition = f"{pattern} {comparison} {end}"
            else:
                initial_value = start
                if step_expression is not None:
                    step_setup, step = self.generate_for_loop_setup_expression(
                        step_expression,
                        indent,
                        self.next_for_loop_step_name,
                        loop_contexts,
                    )
                    setup += step_setup

            if step is None:
                increment = f"{pattern}--" if is_reverse else f"{pattern}++"
            else:
                operator = "-=" if is_reverse else "+="
                increment = f"{pattern} {operator} {step}"

            return (
                setup + f"{indent_str}for (int {pattern} = {initial_value}; "
                f"{condition}; {increment}) {{\n"
            )

        setup, iterable_expr = self.generate_for_loop_setup_expression(
            iterable,
            indent,
            self.next_for_loop_bound_name,
            loop_contexts,
        )

        return (
            setup + f"{indent_str}for (int {pattern} = 0; "
            f"{pattern} < {iterable_expr}; {pattern}++) {{\n"
        )

    def generate_for_loop_setup_expression(
        self,
        expression,
        indent,
        name_factory,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        setup, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )

        if setup:
            if not isinstance(expression, TryNode):
                name = name_factory()
                setup += f"{indent_str}int {name} = {value};\n"
                return setup, name
            return setup, value

        if self.expression_has_side_effects(expression):
            name = name_factory()
            return f"{indent_str}int {name} = {value};\n", name

        return "", value

    def format_reverse_exclusive_range_start(self, end):
        if isinstance(end, str) and re.fullmatch(r"-?\d+", end):
            return str(int(end) - 1)
        return f"({end} - 1)"

    def format_reverse_stepped_range_start(self, start, end, step, inclusive):
        start_value = self.parse_integer_literal(start)
        end_value = self.parse_integer_literal(end)
        step_value = self.parse_integer_literal(step)

        if (
            start_value is not None
            and end_value is not None
            and step_value is not None
            and step_value != 0
        ):
            final_value = end_value if inclusive else end_value - 1
            step_count = (final_value - start_value) // step_value
            return str(start_value + (step_count * step_value))

        final_expr = (
            end if inclusive else self.format_reverse_exclusive_range_start(end)
        )
        if start == "0":
            return f"({final_expr} - ({final_expr} % {step}))"
        return f"({start} + ((({final_expr} - {start}) / {step}) * {step}))"

    def parse_integer_literal(self, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
            return int(value)
        return None

    def parse_for_range_iterable(self, iterable):
        if isinstance(iterable, RangeNode):
            return iterable, None, False, False

        if not isinstance(iterable, FunctionCallNode):
            return None, None, False, False

        callee = iterable.name
        if (
            isinstance(callee, MemberAccessNode)
            and callee.member == "rev"
            and not iterable.args
        ):
            range_node, step_expression, _, align_reverse_step = (
                self.parse_for_range_iterable(callee.object)
            )
            if range_node is not None and range_node.end is not None:
                return (
                    range_node,
                    step_expression,
                    True,
                    step_expression is not None or align_reverse_step,
                )

        if (
            isinstance(callee, MemberAccessNode)
            and callee.member == "step_by"
            and len(iterable.args) == 1
        ):
            range_node, _, is_reverse, align_reverse_step = (
                self.parse_for_range_iterable(callee.object)
            )
            if range_node is not None and is_reverse:
                return range_node, iterable.args[0], True, align_reverse_step
            if isinstance(callee.object, RangeNode):
                return callee.object, iterable.args[0], False, False

        return None, None, False, False

    def generate_while_loop(self, node, indent, loop_contexts=None):
        if isinstance(node.condition, ConditionChainNode):
            return self.generate_condition_chain_while_loop(
                node,
                indent,
                loop_contexts,
            )

        if isinstance(node.condition, MatchesMacroNode):
            return self.generate_matches_while_loop(node, indent, loop_contexts)

        if isinstance(node.condition, LetPatternConditionNode):
            return self.generate_while_let_loop(node, indent, loop_contexts)

        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        if self.expression_contains_try(node.condition):
            code += f"{indent_str}while (true) {{\n"
            condition_code, condition = self.generate_try_expression(
                node.condition,
                indent + 1,
                nested_contexts,
            )
            code += condition_code
            code += f"{indent_str}    if (!{condition}) {{\n"
            code += f"{indent_str}        break;\n"
            code += f"{indent_str}    }}\n"
            code += self.generate_scoped_function_body(
                node.body, indent + 1, nested_contexts
            )
            code += f"{indent_str}}}\n"
            return code

        condition = self.generate_expression(node.condition)
        code += f"{indent_str}while ({condition}) {{\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_condition_chain_while_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"

        def success(success_indent):
            return self.generate_scoped_function_body(
                node.body,
                success_indent,
                nested_contexts,
            )

        def failure(failure_indent):
            failure_indent_str = "    " * failure_indent
            return f"{failure_indent_str}break;\n"

        code += self.generate_condition_chain_branch(
            node.condition,
            indent + 1,
            success,
            failure,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_matches_while_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"
        condition, condition_code = self.generate_matches_condition_flag(
            node.condition,
            indent + 1,
            nested_contexts,
        )
        code += condition_code
        code += f"{indent_str}    if (!{condition}) {{\n"
        code += f"{indent_str}        break;\n"
        code += f"{indent_str}    }}\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_while_let_loop(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(loop_contexts, node)
        loop_context = nested_contexts[-1]

        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"

        def success(success_indent):
            return self.generate_scoped_function_body(
                node.body,
                success_indent,
                nested_contexts,
            )

        def failure(failure_indent):
            failure_indent_str = "    " * failure_indent
            return f"{failure_indent_str}break;\n"

        code += self.generate_let_pattern_condition_branch(
            node.condition,
            indent + 1,
            success,
            failure,
            nested_contexts,
        )
        code += f"{indent_str}}}\n"

        return code

    def generate_loop(self, node, indent, result_target=None, loop_contexts=None):
        indent_str = "    " * indent
        nested_contexts = self.extend_loop_contexts(
            loop_contexts, node, result_target=result_target
        )
        loop_context = nested_contexts[-1]

        # Convert Rust infinite loop to while(true)
        code = self.generate_labeled_control_declarations(loop_context, indent)
        code += f"{indent_str}while (true) {{\n"
        code += self.generate_scoped_function_body(
            node.body, indent + 1, nested_contexts
        )
        code += f"{indent_str}}}\n"

        return code
