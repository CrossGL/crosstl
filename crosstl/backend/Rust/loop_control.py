"""Rust break, continue and labeled control flow carried through CrossGL switches and loops."""

from .RustAst import (
    BreakNode,
    ContinueNode,
    ForNode,
    IfNode,
    LabeledBlockNode,
    LoopNode,
    MatchNode,
    ReturnNode,
    WhileNode,
)


class RustLoopControlMixin:
    """Break, continue and labeled-control members of ``RustToCrossGLConverter``."""

    def branch_guarantees_control_transfer(self, body):
        if isinstance(body, (BreakNode, ContinueNode, ReturnNode)):
            return True
        if isinstance(body, list):
            if not body:
                return False
            return self.branch_guarantees_control_transfer(body[-1])
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            if block_node.statements:
                return self.branch_guarantees_control_transfer(
                    block_node.statements[-1]
                )
            return self.branch_guarantees_control_transfer(
                self.get_block_expression(block_node)
            )
        if isinstance(body, LabeledBlockNode):
            return self.branch_guarantees_control_transfer(body.body)
        if isinstance(body, IfNode):
            if body.else_body is None:
                return False
            return self.branch_guarantees_control_transfer(
                body.if_body
            ) and self.branch_guarantees_control_transfer(body.else_body)
        if isinstance(body, MatchNode):
            return bool(body.arms) and all(
                self.branch_guarantees_control_transfer(arm.body) for arm in body.arms
            )
        return False

    def create_switch_break_flag(self, match_node, loop_contexts=None):
        if not self.match_break_targets_current_loop(match_node, loop_contexts):
            return None

        flag = f"_rust_switch_break_{self.switch_break_counter}"
        self.switch_break_counter += 1
        return flag

    def match_break_targets_current_loop(self, match_node, loop_contexts=None):
        contexts = list(loop_contexts or [])
        if not contexts:
            return False

        current_loop = contexts[-1]
        current_label = current_loop.get("label")
        return any(
            self.contains_current_loop_break_in_switch(
                arm.body,
                current_label,
                nested_loop_depth=0,
            )
            for arm in match_node.arms
        )

    def contains_current_loop_break_in_switch(
        self, node, current_label, nested_loop_depth=0
    ):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_current_loop_break_in_switch(
                    child,
                    current_label,
                    nested_loop_depth,
                )
                for child in node
            )
        if isinstance(node, BreakNode):
            if nested_loop_depth != 0:
                return False
            return node.label is None or node.label == current_label
        if isinstance(node, LabeledBlockNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, LoopNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, WhileNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, ForNode):
            return self.contains_current_loop_break_in_switch(
                node.body,
                current_label,
                nested_loop_depth + 1,
            )
        if isinstance(node, IfNode):
            return self.contains_current_loop_break_in_switch(
                node.if_body,
                current_label,
                nested_loop_depth,
            ) or self.contains_current_loop_break_in_switch(
                node.else_body,
                current_label,
                nested_loop_depth,
            )
        if isinstance(node, MatchNode):
            return any(
                self.contains_current_loop_break_in_switch(
                    arm.body,
                    current_label,
                    nested_loop_depth,
                )
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_current_loop_break_in_switch(
                block_node.statements,
                current_label,
                nested_loop_depth,
            ) or self.contains_current_loop_break_in_switch(
                self.get_block_expression(block_node),
                current_label,
                nested_loop_depth,
            )
        return False

    def with_switch_break_flag(self, loop_contexts=None, switch_break_flag=None):
        contexts = list(loop_contexts or [])
        if not switch_break_flag or not contexts:
            return contexts

        contexts[-1] = dict(contexts[-1])
        contexts[-1]["switch_break_flag"] = switch_break_flag
        return contexts

    def current_switch_break_flag(self, loop_contexts=None):
        contexts = list(loop_contexts or [])
        if not contexts:
            return None
        return contexts[-1].get("switch_break_flag")

    def generate_switch_break_declaration(self, switch_break_flag, indent):
        if not switch_break_flag:
            return ""
        indent_str = "    " * indent
        return f"{indent_str}bool {switch_break_flag} = false;\n"

    def generate_switch_break_propagation(
        self, switch_break_flag, loop_contexts=None, indent=1
    ):
        if not switch_break_flag:
            return ""

        indent_str = "    " * indent
        outer_switch_break_flag = self.current_switch_break_flag(loop_contexts)
        if outer_switch_break_flag:
            return (
                f"{indent_str}if ({switch_break_flag}) {{\n"
                f"{indent_str}    {outer_switch_break_flag} = true;\n"
                f"{indent_str}    break;\n"
                f"{indent_str}}}\n"
            )

        return (
            f"{indent_str}if ({switch_break_flag}) {{\n"
            f"{indent_str}    break;\n"
            f"{indent_str}}}\n"
        )

    def generate_break_statement(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""
        labeled_context = self.resolve_labeled_loop_context(node.label, loop_contexts)
        if labeled_context is not None and not self.is_innermost_loop_context(
            labeled_context,
            loop_contexts,
        ):
            target = labeled_context["result_target"]

            if target is self.return_result_target and node.value is not None:
                return self.generate_expression_result(
                    node.value,
                    indent,
                    self.return_result_target,
                    loop_contexts,
                )

            if target and node.value is not None:
                code += self.generate_expression_result(
                    node.value,
                    indent,
                    target,
                    loop_contexts,
                )
            elif node.value is not None:
                code += self.generate_discarded_expression(
                    node.value,
                    indent,
                    loop_contexts,
                )

            if labeled_context.get("break_flag"):
                code += f"{indent_str}{labeled_context['break_flag']} = true;\n"
            code += f"{indent_str}break;\n"
            return code

        switch_break_flag = self.current_switch_break_flag(loop_contexts)
        if switch_break_flag and (
            node.label is None
            or self.is_innermost_loop_context(labeled_context, loop_contexts)
        ):
            target = self.resolve_break_value_target(node, loop_contexts)

            if target is self.return_result_target and node.value is not None:
                return self.generate_expression_result(
                    node.value,
                    indent,
                    self.return_result_target,
                    loop_contexts,
                )

            if target and node.value is not None:
                code += self.generate_expression_result(
                    node.value,
                    indent,
                    target,
                    loop_contexts,
                )
            elif node.value is not None:
                code += self.generate_discarded_expression(
                    node.value,
                    indent,
                    loop_contexts,
                )

            code += f"{indent_str}{switch_break_flag} = true;\n"
            code += f"{indent_str}break;\n"
            return code

        target = self.resolve_break_value_target(node, loop_contexts)

        if target is self.return_result_target and node.value is not None:
            return self.generate_expression_result(
                node.value,
                indent,
                self.return_result_target,
                loop_contexts,
            )

        if target and node.value is not None:
            code += self.generate_expression_result(
                node.value,
                indent,
                target,
                loop_contexts,
            )

        if target is None and node.value is not None:
            code += self.generate_discarded_expression(
                node.value,
                indent,
                loop_contexts,
            )

        code += f"{indent_str}break;\n"
        return code

    def generate_continue_statement(self, node, indent, loop_contexts=None):
        indent_str = "    " * indent
        labeled_context = self.resolve_labeled_loop_context(node.label, loop_contexts)

        if labeled_context is None or self.is_innermost_loop_context(
            labeled_context,
            loop_contexts,
        ):
            return f"{indent_str}continue;\n"

        code = ""
        if labeled_context.get("continue_flag"):
            code += f"{indent_str}{labeled_context['continue_flag']} = true;\n"
        code += f"{indent_str}break;\n"
        return code

    def extend_loop_contexts(self, loop_contexts, node, result_target=None):
        contexts = list(loop_contexts or [])
        label = getattr(node, "label", None)
        break_flag = None
        continue_flag = None
        if label and self.loop_uses_labeled_control_flags(node):
            suffix = self.make_labeled_control_suffix(label)
            break_flag = f"_rust_break_{suffix}"
            continue_flag = f"_rust_continue_{suffix}"

        contexts.append(
            {
                "label": label,
                "result_target": result_target,
                "break_flag": break_flag,
                "continue_flag": continue_flag,
                "switch_break_flag": None,
            }
        )
        return contexts

    def loop_uses_labeled_control_flags(self, loop_node):
        label = getattr(loop_node, "label", None)
        if not label:
            return False
        return self.contains_labeled_control_target(
            getattr(loop_node, "body", []),
            label,
            nested_loop_depth=0,
        )

    def contains_labeled_control_target(self, node, label, nested_loop_depth=0):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_labeled_control_target(child, label, nested_loop_depth)
                for child in node
            )
        if isinstance(node, (BreakNode, ContinueNode)):
            return node.label == label and nested_loop_depth > 0
        if isinstance(node, LabeledBlockNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, LoopNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, WhileNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, ForNode):
            return self.contains_labeled_control_target(
                node.body,
                label,
                nested_loop_depth + 1,
            )
        if isinstance(node, IfNode):
            return self.contains_labeled_control_target(
                node.if_body,
                label,
                nested_loop_depth,
            ) or self.contains_labeled_control_target(
                node.else_body,
                label,
                nested_loop_depth,
            )
        if isinstance(node, MatchNode):
            return any(
                self.contains_labeled_control_target(
                    arm.body,
                    label,
                    nested_loop_depth,
                )
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_labeled_control_target(
                block_node.statements,
                label,
                nested_loop_depth,
            ) or self.contains_labeled_control_target(
                self.get_block_expression(block_node),
                label,
                nested_loop_depth,
            )
        return False

    def make_labeled_control_suffix(self, label):
        clean = "".join(ch if ch.isalnum() else "_" for ch in str(label).strip("'"))
        clean = clean.strip("_") or "label"
        suffix = f"{clean}_{self.labeled_control_counter}"
        self.labeled_control_counter += 1
        return suffix

    def generate_labeled_control_declarations(self, context, indent):
        if not context.get("break_flag"):
            return ""

        indent_str = "    " * indent
        return (
            f"{indent_str}bool {context['break_flag']} = false;\n"
            f"{indent_str}bool {context['continue_flag']} = false;\n"
        )

    def generate_labeled_control_propagation(self, loop_contexts=None, indent=1):
        contexts = list(loop_contexts or [])
        if not contexts:
            return ""

        indent_str = "    " * indent
        code = ""
        innermost_index = len(contexts) - 1

        for index, context in enumerate(contexts):
            break_flag = context.get("break_flag")
            continue_flag = context.get("continue_flag")
            if not break_flag:
                continue

            if index == innermost_index:
                code += (
                    f"{indent_str}if ({continue_flag}) {{\n"
                    f"{indent_str}    {continue_flag} = false;\n"
                    f"{indent_str}    continue;\n"
                    f"{indent_str}}}\n"
                    f"{indent_str}if ({break_flag}) {{\n"
                    f"{indent_str}    break;\n"
                    f"{indent_str}}}\n"
                )
            else:
                code += (
                    f"{indent_str}if ({continue_flag} || {break_flag}) {{\n"
                    f"{indent_str}    break;\n"
                    f"{indent_str}}}\n"
                )

        return code

    def statement_requires_labeled_control_propagation(self, stmt, loop_contexts=None):
        contexts = list(loop_contexts or [])
        innermost_index = len(contexts) - 1

        for index, context in enumerate(contexts):
            if not context.get("break_flag"):
                continue

            if index == innermost_index:
                if self.contains_labeled_control_target(
                    stmt,
                    context["label"],
                    nested_loop_depth=0,
                ):
                    return True
            elif self.contains_labeled_control_reference(stmt, context["label"]):
                return True

        return False

    def statement_needs_labeled_control_propagation(self, stmt, loop_contexts=None):
        if isinstance(stmt, (BreakNode, ContinueNode, ReturnNode)):
            return False

        if not self.statement_requires_labeled_control_propagation(
            stmt,
            loop_contexts,
        ):
            return False

        if (
            self.is_block_expression_node(stmt)
            and self.branch_guarantees_control_transfer(stmt)
            and not self.contains_switch_match(stmt)
        ):
            return False

        return True

    def contains_switch_match(self, node):
        if node is None:
            return False
        if isinstance(node, list):
            return any(self.contains_switch_match(child) for child in node)
        if isinstance(node, MatchNode):
            return not self.match_requires_if_chain(node) or any(
                self.contains_switch_match(arm.body) for arm in node.arms
            )
        if isinstance(node, LabeledBlockNode):
            return self.contains_switch_match(node.body)
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_switch_match(
                block_node.statements,
            ) or self.contains_switch_match(self.get_block_expression(block_node))
        if isinstance(node, IfNode):
            return self.contains_switch_match(
                node.if_body,
            ) or self.contains_switch_match(node.else_body)
        if isinstance(node, LoopNode):
            return self.contains_switch_match(node.body)
        if isinstance(node, WhileNode):
            return self.contains_switch_match(node.body)
        if isinstance(node, ForNode):
            return self.contains_switch_match(node.body)
        return False

    def contains_labeled_control_reference(self, node, label):
        if node is None:
            return False
        if isinstance(node, list):
            return any(
                self.contains_labeled_control_reference(child, label) for child in node
            )
        if isinstance(node, (BreakNode, ContinueNode)):
            return node.label == label
        if isinstance(node, LabeledBlockNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, LoopNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, WhileNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, ForNode):
            return self.contains_labeled_control_reference(node.body, label)
        if isinstance(node, IfNode):
            return self.contains_labeled_control_reference(
                node.if_body,
                label,
            ) or self.contains_labeled_control_reference(node.else_body, label)
        if isinstance(node, MatchNode):
            return any(
                self.contains_labeled_control_reference(arm.body, label)
                for arm in node.arms
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.contains_labeled_control_reference(
                block_node.statements,
                label,
            ) or self.contains_labeled_control_reference(
                self.get_block_expression(block_node),
                label,
            )
        return False

    def resolve_labeled_loop_context(self, label, loop_contexts=None):
        if not label:
            return None

        for context in reversed(list(loop_contexts or [])):
            if context["label"] == label:
                return context
        return None

    def is_innermost_loop_context(self, context, loop_contexts=None):
        contexts = list(loop_contexts or [])
        return bool(contexts) and context is contexts[-1]

    def resolve_break_value_target(self, node, loop_contexts=None):
        if node.value is None:
            return None

        contexts = list(loop_contexts or [])
        if not contexts:
            return None

        if node.label:
            for context in reversed(contexts):
                if context["label"] == node.label:
                    return context["result_target"]
            return None

        return contexts[-1]["result_target"]
