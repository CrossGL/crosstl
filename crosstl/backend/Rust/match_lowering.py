"""Rust match statements and patterns lowered to CrossGL switch statements and if chains."""

import re

from .RustAst import (
    ArrayNode,
    FunctionCallNode,
    MatchArmNode,
    MatchBindingPatternNode,
    MatchOrPatternNode,
    MatchRestPatternNode,
    MatchStructPatternNode,
    RangeNode,
    ReferenceNode,
    TryNode,
    TupleNode,
)


class RustMatchMixin:
    """Match and pattern members of ``RustToCrossGLConverter``."""

    def generate_match_statement(self, node, indent, loop_contexts=None):
        if self.match_requires_if_chain(node):
            return self.generate_match_if_chain(
                node,
                indent,
                self.generate_match_statement_branch,
                loop_contexts,
            )

        indent_str = "    " * indent
        subject, code = self.generate_match_subject(
            node.expression,
            indent,
            loop_contexts,
        )
        expression = self.generate_match_subject_expression(subject)
        switch_break_flag = self.create_switch_break_flag(node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        # Convert Rust match to switch statement
        code += self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_match_statement_branch(
                arm.body,
                indent + 2,
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

    def generate_match_statement_branch(self, body, indent, loop_contexts=None):
        if isinstance(body, list):
            return self.generate_scoped_function_body(body, indent, loop_contexts)
        return self.generate_result_branch(body, indent, None, loop_contexts)

    def generate_match_case_label(self, pattern, indent):
        indent_str = "    " * indent
        if pattern == "_":
            return f"{indent_str}default:\n"

        return f"{indent_str}case {self.generate_expression(pattern)}:\n"

    def match_requires_if_chain(self, match_node):
        return any(
            isinstance(arm.pattern, MatchOrPatternNode)
            or arm.guard is not None
            or not self.is_switch_case_pattern(arm.pattern)
            for arm in match_node.arms
        )

    def is_switch_case_pattern(self, pattern):
        return pattern == "_" or isinstance(pattern, str)

    def next_match_chain_flag_name(self):
        name = f"_rust_match_matched_{self.match_chain_counter}"
        self.match_chain_counter += 1
        return name

    def next_match_subject_name(self):
        name = f"_rust_match_subject_{self.match_subject_counter}"
        self.match_subject_counter += 1
        return name

    def next_match_payload_name(self):
        name = f"_rust_match_payload_{self.match_payload_counter}"
        self.match_payload_counter += 1
        return name

    def next_match_or_flag_name(self):
        name = f"_rust_match_or_matched_{self.match_or_counter}"
        self.match_or_counter += 1
        return name

    def next_match_tuple_name(self):
        name = f"_rust_match_tuple_{self.match_tuple_counter}"
        self.match_tuple_counter += 1
        return name

    def next_match_array_name(self):
        name = f"_rust_match_array_{self.match_array_counter}"
        self.match_array_counter += 1
        return name

    def next_matches_result_name(self):
        name = f"_rust_matches_result_{self.matches_result_counter}"
        self.matches_result_counter += 1
        return name

    def next_closure_param_name(self):
        name = f"_rust_closure_arg_{self.closure_param_counter}"
        self.closure_param_counter += 1
        return name

    def next_try_subject_name(self):
        name = f"_rust_try_subject_{self.try_subject_counter}"
        self.try_subject_counter += 1
        return name

    def next_try_value_name(self):
        name = f"_rust_try_value_{self.try_value_counter}"
        self.try_value_counter += 1
        return name

    def next_try_block_value_name(self):
        name = f"_rust_try_block_value_{self.try_block_value_counter}"
        self.try_block_value_counter += 1
        return name

    def next_transparent_block_value_name(self):
        name = f"_rust_block_value_{self.transparent_block_value_counter}"
        self.transparent_block_value_counter += 1
        return name

    def next_inline_expression_value_name(self):
        name = f"_rust_expr_value_{self.inline_expression_value_counter}"
        self.inline_expression_value_counter += 1
        return name

    def generate_match_if_chain(
        self,
        match_node,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        subject, code = self.generate_match_subject(
            match_node.expression,
            indent,
            loop_contexts,
        )

        matched_flag = self.next_match_chain_flag_name()
        indent_str = "    " * indent
        code += f"{indent_str}bool {matched_flag} = false;\n"

        for arm in match_node.arms:
            code += self.generate_match_if_chain_arm(
                subject,
                arm,
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        return code

    def generate_match_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if isinstance(expression, TupleNode):
            return self.generate_match_tuple_subject(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, ArrayNode):
            return self.generate_match_array_subject(
                expression,
                indent,
                loop_contexts,
            )

        if self.match_subject_needs_statement_result(expression):
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        if self.expression_contains_try(expression):
            try_code, subject = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            if not isinstance(expression, TryNode) and self.expression_has_side_effects(
                expression
            ):
                subject_name = self.next_match_subject_name()
                try_code += f"{indent_str}auto {subject_name} = {subject};\n"
                subject = subject_name
            return subject, try_code

        subject = self.generate_expression(expression)
        if self.expression_has_side_effects(expression):
            subject_name = self.next_match_subject_name()
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            subject = subject_name

        return subject, code

    def match_subject_needs_statement_result(self, expression):
        return self.is_block_expression_node(expression) and (
            self.expression_contains_try(expression)
            or self.expression_has_side_effects(expression)
        )

    def generate_match_tuple_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        elements = []
        code = ""

        for element in expression.elements:
            if isinstance(element, TupleNode):
                nested_subject, nested_code = self.generate_match_tuple_subject(
                    element,
                    indent,
                    loop_contexts,
                )
                code += nested_code
                elements.append(nested_subject)
                continue

            element_name = self.next_match_tuple_name()
            element_code, element_value = self.generate_try_expression(
                element,
                indent,
                loop_contexts,
            )
            code += element_code
            code += f"{indent_str}auto {element_name} = " f"{element_value};\n"
            elements.append(element_name)

        return TupleNode(elements), code

    def generate_match_array_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        elements = []
        code = ""

        if expression.size is not None:
            subject_code, subject = self.generate_try_expression(
                expression,
                indent,
                loop_contexts,
            )
            subject_name = self.next_match_array_name()
            code += subject_code
            code += f"{indent_str}auto {subject_name} = {subject};\n"
            return subject_name, code

        for element in expression.elements:
            if isinstance(element, ArrayNode):
                nested_subject, nested_code = self.generate_match_array_subject(
                    element,
                    indent,
                )
                code += nested_code
                elements.append(nested_subject)
                continue

            element_name = self.next_match_array_name()
            element_code, element_value = self.generate_try_expression(
                element,
                indent,
                loop_contexts,
            )
            code += element_code
            code += f"{indent_str}auto {element_name} = " f"{element_value};\n"
            elements.append(element_name)

        return ArrayNode(elements), code

    def generate_match_if_chain_arm(
        self,
        subject,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        if isinstance(arm.pattern, ReferenceNode):
            return self.generate_match_if_chain_arm(
                subject,
                MatchArmNode(arm.pattern.expression, arm.guard, arm.body),
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        if isinstance(arm.pattern, MatchBindingPatternNode) and isinstance(
            arm.pattern.pattern, MatchOrPatternNode
        ):
            code = ""
            for pattern in arm.pattern.pattern.patterns:
                code += self.generate_match_if_chain_arm(
                    subject,
                    MatchArmNode(
                        MatchBindingPatternNode(arm.pattern.name, pattern),
                        arm.guard,
                        arm.body,
                    ),
                    matched_flag,
                    indent,
                    branch_generator,
                    loop_contexts,
                )
            return code

        if isinstance(arm.pattern, MatchOrPatternNode):
            code = ""
            for pattern in arm.pattern.patterns:
                code += self.generate_match_if_chain_arm(
                    subject,
                    MatchArmNode(pattern, arm.guard, arm.body),
                    matched_flag,
                    indent,
                    branch_generator,
                    loop_contexts,
                )
            return code

        if self.pattern_has_nested_constructor(arm.pattern):
            return self.generate_nested_constructor_match_arm(
                subject,
                arm,
                matched_flag,
                indent,
                branch_generator,
                loop_contexts,
            )

        indent_str = "    " * indent
        pattern_condition = self.generate_match_pattern_condition(
            subject,
            arm.pattern,
        )
        if pattern_condition is None:
            arm_condition = f"!{matched_flag}"
        else:
            arm_condition = f"!{matched_flag} && {pattern_condition}"

        code = f"{indent_str}if ({arm_condition}) {{\n"
        code += self.generate_match_pattern_bindings(
            subject,
            arm.pattern,
            indent + 1,
        )

        if arm.guard is not None:
            guard_code, guard = self.generate_try_expression(
                arm.guard,
                indent + 1,
                loop_contexts,
            )
            code += guard_code
            code += f"{indent_str}    if ({guard}) {{\n"
            code += f"{indent_str}        {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 2, loop_contexts)
            code += f"{indent_str}    }}\n"
        else:
            code += f"{indent_str}    {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 1, loop_contexts)

        code += f"{indent_str}}}\n"
        return code

    def pattern_has_nested_constructor(self, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.pattern_has_nested_constructor(pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.pattern_has_nested_constructor(pattern.expression)

        if isinstance(pattern, TupleNode):
            return True

        if isinstance(pattern, ArrayNode):
            return True

        if isinstance(pattern, MatchStructPatternNode):
            return True

        if isinstance(pattern, MatchOrPatternNode):
            return any(
                isinstance(alternative, FunctionCallNode)
                or self.pattern_has_nested_constructor(alternative)
                or isinstance(alternative, TupleNode)
                or isinstance(alternative, ArrayNode)
                or isinstance(alternative, MatchStructPatternNode)
                for alternative in pattern.patterns
            )

        if not isinstance(pattern, FunctionCallNode):
            return False

        return any(
            isinstance(arg, FunctionCallNode)
            or self.pattern_has_nested_constructor(arg)
            for arg in pattern.args
        )

    def generate_nested_constructor_match_arm(
        self,
        subject,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        indent_str = "    " * indent

        def success(success_indent):
            return self.generate_match_arm_success(
                arm,
                matched_flag,
                success_indent,
                branch_generator,
                loop_contexts,
            )

        code = f"{indent_str}if (!{matched_flag}) {{\n"
        code += self.generate_nested_pattern_match(
            subject,
            arm.pattern,
            indent + 1,
            success,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        if isinstance(pattern, MatchBindingPatternNode):

            def binding_success(success_indent):
                success_indent_str = "    " * success_indent
                binding = ""
                if not self.is_discard_pattern(pattern.name):
                    binding = (
                        f"{success_indent_str}auto {pattern.name} = "
                        f"{self.generate_match_subject_expression(subject)};\n"
                    )
                return binding + success(success_indent)

            return self.generate_nested_pattern_match(
                subject,
                pattern.pattern,
                indent,
                binding_success,
            )

        if isinstance(pattern, TupleNode):
            return self.generate_nested_tuple_pattern_match(
                subject,
                pattern,
                0,
                indent,
                success,
            )

        if isinstance(pattern, ArrayNode):
            return self.generate_nested_array_pattern_match(
                subject,
                pattern,
                indent,
                success,
            )

        if isinstance(pattern, MatchStructPatternNode):
            return self.generate_nested_struct_pattern_match(
                subject,
                pattern,
                indent,
                success,
            )

        if isinstance(pattern, ReferenceNode):
            return self.generate_nested_pattern_match(
                subject,
                pattern.expression,
                indent,
                success,
            )

        if isinstance(pattern, MatchOrPatternNode):
            matched_flag = self.next_match_or_flag_name()
            code = f"{indent_str}bool {matched_flag} = false;\n"
            for alternative in pattern.patterns:

                def alternative_success(success_indent, matched_flag=matched_flag):
                    success_indent_str = "    " * success_indent
                    return f"{success_indent_str}{matched_flag} = true;\n" + success(
                        success_indent
                    )

                code += f"{indent_str}if (!{matched_flag}) {{\n"
                code += self.generate_nested_pattern_match(
                    subject,
                    alternative,
                    indent + 1,
                    alternative_success,
                )
                code += f"{indent_str}}}\n"
            return code

        if not isinstance(pattern, FunctionCallNode):
            condition = self.generate_match_pattern_condition(subject, pattern)
            if condition is None:
                code = self.generate_match_pattern_bindings(subject, pattern, indent)
                code += success(indent)
                return code

            code = f"{indent_str}if ({condition}) {{\n"
            code += self.generate_match_pattern_bindings(subject, pattern, indent + 1)
            code += success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        if isinstance(pattern, MatchRestPatternNode):
            return success(indent)

        subject_expr = self.generate_match_subject_expression(subject)
        constructor = self.generate_constructor_helper_name(pattern.name)
        code = f"{indent_str}if (is_{constructor}({subject_expr})) {{\n"
        code += self.generate_nested_constructor_args(
            subject_expr,
            constructor,
            pattern.args,
            0,
            indent + 1,
            success,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_struct_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        subject_expr = self.generate_match_subject_expression(subject)

        def fields_success(success_indent):
            return self.generate_nested_struct_pattern_fields(
                subject_expr,
                pattern,
                0,
                success_indent,
                success,
            )

        if self.is_record_variant_pattern(pattern):
            helper = self.generate_constructor_helper_name(pattern.name)
            code = f"{indent_str}if (is_{helper}({subject_expr})) {{\n"
            code += fields_success(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        return fields_success(indent)

    def generate_nested_struct_pattern_fields(
        self,
        subject,
        pattern,
        index,
        indent,
        success,
    ):
        if index >= len(pattern.fields):
            return success(indent)

        field_name, field_pattern = pattern.fields[index]
        field_subject = self.generate_struct_field_accessor(
            subject,
            pattern,
            field_name,
        )

        def continue_fields(next_indent):
            return self.generate_nested_struct_pattern_fields(
                subject,
                pattern,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            field_subject,
            field_pattern,
            indent,
            continue_fields,
        )

    def generate_nested_tuple_pattern_match(
        self,
        subject,
        pattern,
        index,
        indent,
        success,
    ):
        if index >= len(pattern.elements):
            return success(indent)

        if isinstance(subject, TupleNode) and index >= len(subject.elements):
            return ""

        element_subject = self.generate_tuple_element_accessor(subject, index)

        def continue_elements(next_indent):
            return self.generate_nested_tuple_pattern_match(
                subject,
                pattern,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            element_subject,
            pattern.elements[index],
            indent,
            continue_elements,
        )

    def generate_nested_array_pattern_match(self, subject, pattern, indent, success):
        indent_str = "    " * indent
        rest_index = self.find_array_rest_index(pattern)
        length_condition = self.generate_array_length_condition(
            subject,
            pattern,
            rest_index,
        )

        def length_success(success_indent):
            return self.generate_nested_array_elements(
                subject,
                pattern,
                rest_index,
                success_indent,
                success,
            )

        if length_condition is None:
            return length_success(indent)

        code = f"{indent_str}if ({length_condition}) {{\n"
        code += length_success(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_nested_array_elements(
        self,
        subject,
        pattern,
        rest_index,
        indent,
        success,
    ):
        entries = self.get_array_pattern_entries(subject, pattern, rest_index)

        def rest_success(success_indent):
            code = self.generate_array_rest_binding(
                subject,
                pattern,
                rest_index,
                success_indent,
            )
            code += success(success_indent)
            return code

        return self.generate_nested_array_element_entries(
            entries,
            0,
            indent,
            rest_success,
        )

    def generate_nested_array_element_entries(
        self,
        entries,
        index,
        indent,
        success,
    ):
        if index >= len(entries):
            return success(indent)

        element_pattern, element_subject = entries[index]

        def continue_entries(next_indent):
            return self.generate_nested_array_element_entries(
                entries,
                index + 1,
                next_indent,
                success,
            )

        return self.generate_nested_pattern_match(
            element_subject,
            element_pattern,
            indent,
            continue_entries,
        )

    def find_array_rest_index(self, pattern):
        for index, element in enumerate(pattern.elements):
            if self.is_array_rest_pattern(element):
                return index
        return None

    def is_array_rest_pattern(self, pattern):
        if isinstance(pattern, MatchRestPatternNode):
            return True
        return isinstance(pattern, MatchBindingPatternNode) and isinstance(
            pattern.pattern, MatchRestPatternNode
        )

    def generate_array_length_condition(self, subject, pattern, rest_index):
        fixed_count = len(pattern.elements) - (1 if rest_index is not None else 0)
        subject_len = self.get_static_array_subject_length(subject)

        if rest_index is None:
            if subject_len is not None:
                return None if subject_len == fixed_count else "false"
            return (
                f"({self.generate_array_length_expression(subject)} == {fixed_count})"
            )

        if subject_len is not None:
            return None if subject_len >= fixed_count else "false"
        return f"({self.generate_array_length_expression(subject)} >= {fixed_count})"

    def get_static_array_subject_length(self, subject):
        if isinstance(subject, ArrayNode):
            return len(subject.elements)
        return None

    def get_array_pattern_entries(self, subject, pattern, rest_index):
        entries = []
        elements = pattern.elements

        if rest_index is None:
            for index, element in enumerate(elements):
                entries.append(
                    (element, self.generate_array_element_accessor(subject, index))
                )
            return entries

        for index, element in enumerate(elements[:rest_index]):
            entries.append(
                (element, self.generate_array_element_accessor(subject, index))
            )

        suffix = elements[rest_index + 1 :]
        for offset, element in enumerate(suffix):
            entries.append(
                (
                    element,
                    self.generate_array_suffix_element_accessor(
                        subject,
                        len(suffix) - offset,
                    ),
                )
            )

        return entries

    def generate_array_rest_binding(self, subject, pattern, rest_index, indent):
        if rest_index is None:
            return ""

        rest_pattern = pattern.elements[rest_index]
        if not isinstance(rest_pattern, MatchBindingPatternNode):
            return ""
        if self.is_discard_pattern(rest_pattern.name):
            return ""

        indent_str = "    " * indent
        suffix_count = len(pattern.elements) - rest_index - 1
        rest_length = self.generate_array_rest_length_expression(
            subject,
            rest_index,
            suffix_count,
        )
        rest_slice = self.generate_array_slice_expression(
            subject,
            rest_index,
            rest_length,
        )
        return f"{indent_str}auto {rest_pattern.name} = {rest_slice};\n"

    def generate_array_rest_length_expression(self, subject, rest_index, suffix_count):
        subject_len = self.get_static_array_subject_length(subject)
        fixed_count = rest_index + suffix_count
        if subject_len is not None:
            return str(subject_len - fixed_count)

        length_expr = self.generate_array_length_expression(subject)
        if fixed_count == 0:
            return length_expr
        return f"({length_expr} - {fixed_count})"

    def generate_array_length_expression(self, subject):
        subject_len = self.get_static_array_subject_length(subject)
        if subject_len is not None:
            return str(subject_len)
        return f"length({self.generate_match_subject_expression(subject)})"

    def generate_array_slice_expression(self, subject, start, length):
        return (
            f"_rust_slice({self.generate_match_subject_expression(subject)}, "
            f"{start}, {length})"
        )

    def generate_nested_constructor_args(
        self,
        subject,
        constructor,
        args,
        index,
        indent,
        success,
    ):
        if index >= len(args):
            return success(indent)

        indent_str = "    " * indent
        arg = args[index]
        payload = self.generate_constructor_payload_accessor(
            subject,
            constructor,
            index,
            len(args),
        )

        def continue_args(next_indent):
            return self.generate_nested_constructor_args(
                subject,
                constructor,
                args,
                index + 1,
                next_indent,
                success,
            )

        if isinstance(arg, MatchOrPatternNode):
            payload_name = self.next_match_payload_name()
            code = f"{indent_str}auto {payload_name} = {payload};\n"
            code += self.generate_nested_pattern_match(
                payload_name,
                arg,
                indent,
                continue_args,
            )
            return code

        if isinstance(arg, MatchBindingPatternNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, TupleNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, ArrayNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, MatchStructPatternNode):
            return self.generate_nested_pattern_match(
                payload,
                arg,
                indent,
                continue_args,
            )

        if isinstance(arg, ReferenceNode):
            return self.generate_nested_pattern_match(
                payload,
                arg.expression,
                indent,
                continue_args,
            )

        if self.is_discard_pattern(arg):
            return continue_args(indent)

        if self.is_simple_pattern_binding(arg):
            code = f"{indent_str}auto {arg} = {payload};\n"
            code += continue_args(indent)
            return code

        if isinstance(arg, FunctionCallNode):
            payload_name = self.next_match_payload_name()
            code = f"{indent_str}auto {payload_name} = {payload};\n"
            code += self.generate_nested_pattern_match(
                payload_name,
                arg,
                indent,
                continue_args,
            )
            return code

        if isinstance(arg, RangeNode):
            condition = self.generate_match_pattern_condition(payload, arg)
            code = f"{indent_str}if ({condition}) {{\n"
            code += continue_args(indent + 1)
            code += f"{indent_str}}}\n"
            return code

        condition = f"({payload} == {self.generate_expression(arg)})"
        code = f"{indent_str}if ({condition}) {{\n"
        code += continue_args(indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_match_arm_success(
        self,
        arm,
        matched_flag,
        indent,
        branch_generator,
        loop_contexts=None,
    ):
        indent_str = "    " * indent
        if arm.guard is not None:
            guard = self.generate_expression(arm.guard)
            code = f"{indent_str}if ({guard}) {{\n"
            code += f"{indent_str}    {matched_flag} = true;\n"
            code += branch_generator(arm.body, indent + 1, loop_contexts)
            code += f"{indent_str}}}\n"
            return code

        code = f"{indent_str}{matched_flag} = true;\n"
        code += branch_generator(arm.body, indent, loop_contexts)
        return code

    def generate_match_pattern_condition(self, subject, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.generate_match_pattern_condition(subject, pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.generate_match_pattern_condition(subject, pattern.expression)

        if isinstance(pattern, MatchOrPatternNode):
            alternatives = []
            for alternative in pattern.patterns:
                condition = self.generate_match_pattern_condition(subject, alternative)
                alternatives.append(condition or "true")
            return " || ".join(f"({condition})" for condition in alternatives)

        conditions = self.generate_pattern_conditions(subject, pattern)
        if not conditions:
            return None

        return " && ".join(conditions)

    def generate_pattern_conditions(self, subject, pattern):
        if isinstance(pattern, MatchBindingPatternNode):
            return self.generate_pattern_conditions(subject, pattern.pattern)

        if isinstance(pattern, ReferenceNode):
            return self.generate_pattern_conditions(subject, pattern.expression)

        if isinstance(pattern, TupleNode):
            if isinstance(subject, TupleNode) and len(pattern.elements) != len(
                subject.elements
            ):
                return ["false"]

            conditions = []
            for index, element in enumerate(pattern.elements):
                conditions.extend(
                    self.generate_pattern_conditions(
                        self.generate_tuple_element_accessor(subject, index),
                        element,
                    )
                )
            return conditions

        if isinstance(pattern, ArrayNode):
            rest_index = self.find_array_rest_index(pattern)
            length_condition = self.generate_array_length_condition(
                subject,
                pattern,
                rest_index,
            )
            conditions = [length_condition] if length_condition is not None else []
            for element, element_subject in self.get_array_pattern_entries(
                subject,
                pattern,
                rest_index,
            ):
                conditions.extend(
                    self.generate_pattern_conditions(element_subject, element)
                )
            return conditions

        if isinstance(pattern, MatchStructPatternNode):
            subject_expr = self.generate_match_subject_expression(subject)
            conditions = []
            if self.is_record_variant_pattern(pattern):
                helper = self.generate_constructor_helper_name(pattern.name)
                conditions.append(f"is_{helper}({subject_expr})")

            for field_name, field_pattern in pattern.fields:
                conditions.extend(
                    self.generate_pattern_conditions(
                        self.generate_struct_field_accessor(
                            subject_expr,
                            pattern,
                            field_name,
                        ),
                        field_pattern,
                    )
                )
            return conditions

        if isinstance(pattern, MatchRestPatternNode):
            return []

        if isinstance(pattern, MatchOrPatternNode):
            condition = self.generate_match_pattern_condition(subject, pattern)
            return [f"({condition})"] if condition else []

        if isinstance(pattern, RangeNode):
            subject_expr = self.generate_match_subject_expression(subject)
            start = self.generate_expression(pattern.start)
            end = self.generate_expression(pattern.end)
            end_operator = "<=" if pattern.inclusive else "<"
            return [
                f"({subject_expr} >= {start})",
                f"({subject_expr} {end_operator} {end})",
            ]

        if self.is_discard_pattern(pattern) or self.is_simple_pattern_binding(pattern):
            return []

        if isinstance(pattern, FunctionCallNode):
            subject_expr = self.generate_match_subject_expression(subject)
            constructor = self.generate_constructor_helper_name(pattern.name)
            conditions = [f"is_{constructor}({subject_expr})"]
            arg_count = len(pattern.args)
            for index, arg in enumerate(pattern.args):
                payload = self.generate_constructor_payload_accessor(
                    subject_expr,
                    constructor,
                    index,
                    arg_count,
                )
                conditions.extend(self.generate_pattern_conditions(payload, arg))
            return conditions

        return [
            f"({self.generate_match_subject_expression(subject)} == "
            f"{self.generate_expression(pattern)})"
        ]

    def generate_match_pattern_bindings(self, subject, pattern, indent):
        if isinstance(pattern, MatchBindingPatternNode):
            indent_str = "    " * indent
            binding = ""
            if not self.is_discard_pattern(pattern.name):
                binding += (
                    f"{indent_str}auto {pattern.name} = "
                    f"{self.generate_match_subject_expression(subject)};\n"
                )
            binding += self.generate_match_pattern_bindings(
                subject,
                pattern.pattern,
                indent,
            )
            return binding

        if isinstance(pattern, ReferenceNode):
            return self.generate_match_pattern_bindings(
                subject,
                pattern.expression,
                indent,
            )

        if isinstance(pattern, MatchOrPatternNode):
            return ""

        indent_str = "    " * indent
        if self.is_simple_pattern_binding(pattern):
            return (
                f"{indent_str}auto {pattern} = "
                f"{self.generate_match_subject_expression(subject)};\n"
            )

        if isinstance(pattern, TupleNode):
            code = ""
            for index, element in enumerate(pattern.elements):
                code += self.generate_match_pattern_bindings(
                    self.generate_tuple_element_accessor(subject, index),
                    element,
                    indent,
                )
            return code

        if isinstance(pattern, ArrayNode):
            code = ""
            rest_index = self.find_array_rest_index(pattern)
            for element, element_subject in self.get_array_pattern_entries(
                subject,
                pattern,
                rest_index,
            ):
                code += self.generate_match_pattern_bindings(
                    element_subject,
                    element,
                    indent,
                )
            code += self.generate_array_rest_binding(
                subject,
                pattern,
                rest_index,
                indent,
            )
            return code

        if isinstance(pattern, MatchStructPatternNode):
            subject_expr = self.generate_match_subject_expression(subject)
            code = ""
            for field_name, field_pattern in pattern.fields:
                code += self.generate_match_pattern_bindings(
                    self.generate_struct_field_accessor(
                        subject_expr,
                        pattern,
                        field_name,
                    ),
                    field_pattern,
                    indent,
                )
            return code

        if isinstance(pattern, MatchRestPatternNode):
            return ""

        if not isinstance(pattern, FunctionCallNode):
            return ""

        subject_expr = self.generate_match_subject_expression(subject)
        constructor = self.generate_constructor_helper_name(pattern.name)
        arg_count = len(pattern.args)
        code = ""
        for index, arg in enumerate(pattern.args):
            payload = self.generate_constructor_payload_accessor(
                subject_expr,
                constructor,
                index,
                arg_count,
            )
            code += self.generate_match_pattern_bindings(payload, arg, indent)
        return code

    def generate_match_subject_expression(self, subject):
        if isinstance(subject, TupleNode):
            return self.generate_expression(subject)
        if isinstance(subject, ArrayNode):
            return self.generate_expression(subject)
        if not isinstance(subject, str):
            return self.generate_expression(subject)
        return str(subject)

    def generate_tuple_element_accessor(self, subject, index):
        if isinstance(subject, TupleNode):
            element = subject.elements[index]
            if isinstance(element, TupleNode):
                return element
            return self.generate_expression(element)

        return f"_rust_tuple_{index}({self.generate_match_subject_expression(subject)})"

    def generate_array_element_accessor(self, subject, index):
        if isinstance(subject, ArrayNode):
            element = subject.elements[index]
            if isinstance(element, ArrayNode):
                return element
            return self.generate_expression(element)

        return f"{self.generate_match_subject_expression(subject)}[{index}]"

    def generate_array_suffix_element_accessor(self, subject, reverse_index):
        subject_len = self.get_static_array_subject_length(subject)
        if subject_len is not None:
            return self.generate_array_element_accessor(
                subject,
                subject_len - reverse_index,
            )

        length_expr = self.generate_array_length_expression(subject)
        if reverse_index == 1:
            index_expr = f"({length_expr} - 1)"
        else:
            index_expr = f"({length_expr} - {reverse_index})"
        return f"{self.generate_match_subject_expression(subject)}[{index_expr}]"

    def is_record_variant_pattern(self, pattern):
        return isinstance(pattern.name, str) and "::" in pattern.name

    def generate_struct_field_accessor(self, subject, pattern, field_name):
        subject_expr = self.generate_match_subject_expression(subject)
        if self.is_record_variant_pattern(pattern):
            helper = self.generate_constructor_helper_name(pattern.name)
            return f"unwrap_{helper}_{field_name}({subject_expr})"

        return f"{subject_expr}.{field_name}"

    def generate_constructor_payload_accessor(
        self,
        subject,
        constructor,
        index,
        arg_count,
    ):
        if arg_count == 1:
            return f"unwrap_{constructor}({subject})"
        return f"unwrap_{constructor}_{index}({subject})"

    def generate_constructor_helper_name(self, constructor):
        if isinstance(constructor, str):
            name = constructor
        else:
            name = self.generate_expression(constructor)

        helper_name = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
        return helper_name or "variant"

    def is_simple_pattern_binding(self, pattern):
        if not isinstance(pattern, str):
            return False
        if pattern == "_" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern):
            return False
        if pattern in {"true", "false", "None", "Some", "Ok", "Err"}:
            return False
        return pattern[0].islower()

    def match_arm_needs_switch_terminator(self, body):
        return not self.branch_guarantees_control_transfer(body)
