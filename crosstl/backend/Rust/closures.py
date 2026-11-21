"""Rust closures lowered to CrossGL helper functions or inline expressions."""

import re

from .RustAst import (
    ArrayAccessNode,
    ArrayNode,
    AssignmentNode,
    AwaitNode,
    BinaryOpNode,
    CastNode,
    ClosureNode,
    ConditionChainNode,
    DereferenceNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    LetNode,
    LetPatternConditionNode,
    LoopNode,
    MatchBindingPatternNode,
    MatchesMacroNode,
    MatchNode,
    MatchStructPatternNode,
    MemberAccessNode,
    ReferenceNode,
    ReturnNode,
    StructInitializationNode,
    TernaryOpNode,
    TryBlockNode,
    TryNode,
    TupleNode,
    UnaryOpNode,
    VectorConstructorNode,
    WhileNode,
)


class RustClosureMixin:
    """Closure members of ``RustToCrossGLConverter``."""

    def try_generate_closure_helper(self, closure, context_name=None):
        if self.current_closure_helpers is None:
            return None
        if self.closure_helper_generation_depth:
            return None
        if not self.can_generate_closure_helper(closure):
            return None
        if self.closure_has_captures(closure):
            return None

        helper_name = self.next_closure_helper_name(context_name)
        helper_code = self.generate_closure_helper_function(helper_name, closure)
        self.current_closure_helpers.append(helper_code)
        self.closure_helper_names.add(helper_name)
        self.user_function_names.add(helper_name)
        return helper_name

    def can_generate_closure_helper(self, closure):
        if not getattr(closure, "return_type", None):
            return False
        if self.expression_contains_try(closure.body):
            return False
        if self.closure_body_contains_unsupported_helper_node(closure.body):
            return False

        for param in closure.params:
            if not param.param_type:
                return False
            if not self.is_helper_compatible_parameter_type(param.param_type):
                return False
            if not self.is_helper_supported_closure_parameter_pattern(param.pattern):
                return False
        return True

    def is_helper_compatible_parameter_type(self, param_type):
        mapped_type = self.map_type(param_type)
        if not mapped_type:
            return False
        if mapped_type.startswith("(") or "->" in mapped_type:
            return False
        return True

    def is_helper_supported_closure_parameter_pattern(self, pattern):
        if self.is_discard_pattern(pattern):
            return True
        if isinstance(pattern, str):
            return self.is_plain_identifier(pattern)
        if isinstance(pattern, ReferenceNode):
            return self.is_helper_supported_closure_parameter_pattern(
                pattern.expression
            )
        if isinstance(pattern, MatchBindingPatternNode):
            return self.is_plain_identifier(
                pattern.name
            ) and self.is_helper_supported_closure_parameter_pattern(pattern.pattern)
        if isinstance(pattern, TupleNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(element)
                for element in pattern.elements
            )
        if isinstance(pattern, MatchStructPatternNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(field_pattern)
                for _, field_pattern in pattern.fields
            )
        if isinstance(pattern, FunctionCallNode):
            return all(
                self.is_helper_supported_closure_parameter_pattern(arg)
                for arg in pattern.args
            )
        return False

    def closure_body_contains_unsupported_helper_node(self, node):
        unsupported_types = (
            ClosureNode,
            TryNode,
            TryBlockNode,
            AwaitNode,
            LoopNode,
            ForNode,
            WhileNode,
            MatchNode,
            MatchesMacroNode,
            LetPatternConditionNode,
            ConditionChainNode,
        )
        if node is None:
            return False
        if isinstance(node, unsupported_types):
            return True
        if isinstance(node, (str, int, float, bool)):
            return False
        if isinstance(node, list):
            return any(
                self.closure_body_contains_unsupported_helper_node(item)
                for item in node
            )
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            return self.closure_body_contains_unsupported_helper_node(
                block_node.statements
            ) or self.closure_body_contains_unsupported_helper_node(
                self.get_block_expression(block_node)
            )
        if isinstance(node, LetNode):
            return self.closure_body_contains_unsupported_helper_node(node.value)
        if isinstance(node, ReturnNode):
            return self.closure_body_contains_unsupported_helper_node(node.value)
        if isinstance(node, IfNode):
            return (
                self.closure_body_contains_unsupported_helper_node(node.condition)
                or self.closure_body_contains_unsupported_helper_node(node.if_body)
                or self.closure_body_contains_unsupported_helper_node(node.else_body)
            )
        if isinstance(node, BinaryOpNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.left
            ) or self.closure_body_contains_unsupported_helper_node(node.right)
        if isinstance(node, UnaryOpNode):
            return self.closure_body_contains_unsupported_helper_node(node.operand)
        if isinstance(node, FunctionCallNode):
            return self.closure_body_contains_unsupported_helper_node(node.name) or any(
                self.closure_body_contains_unsupported_helper_node(arg)
                for arg in node.args
            )
        if isinstance(node, MemberAccessNode):
            return self.closure_body_contains_unsupported_helper_node(node.object)
        if isinstance(node, AssignmentNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.left
            ) or self.closure_body_contains_unsupported_helper_node(node.right)
        if isinstance(node, CastNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, ReferenceNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, DereferenceNode):
            return self.closure_body_contains_unsupported_helper_node(node.expression)
        if isinstance(node, ArrayAccessNode):
            return self.closure_body_contains_unsupported_helper_node(
                node.array
            ) or self.closure_body_contains_unsupported_helper_node(node.index)
        if isinstance(node, VectorConstructorNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(arg)
                for arg in node.args
            )
        if isinstance(node, TupleNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(element)
                for element in node.elements
            )
        if isinstance(node, ArrayNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(element)
                for element in node.elements
            ) or self.closure_body_contains_unsupported_helper_node(node.size)
        if isinstance(node, StructInitializationNode):
            return any(
                self.closure_body_contains_unsupported_helper_node(value)
                for _, value in node.fields
            )
        if isinstance(node, TernaryOpNode):
            return (
                self.closure_body_contains_unsupported_helper_node(node.condition)
                or self.closure_body_contains_unsupported_helper_node(node.true_expr)
                or self.closure_body_contains_unsupported_helper_node(node.false_expr)
            )
        return False

    def closure_has_captures(self, closure):
        local_names = set()
        for param in closure.params:
            self.add_closure_local_bindings(param.pattern, local_names)
        captures = set()
        self.collect_closure_identifier_captures(
            closure.body,
            local_names,
            captures,
        )
        return bool(captures)

    def collect_closure_identifier_captures(self, node, local_names, captures):
        if node is None or isinstance(node, (int, float, bool)):
            return
        if isinstance(node, str):
            if self.is_capturable_identifier(node) and node not in local_names:
                captures.add(node)
            return
        if isinstance(node, list):
            for item in node:
                self.collect_closure_identifier_captures(item, local_names, captures)
            return
        if self.is_block_expression_node(node):
            block_node = self.get_block_expression_node(node)
            block_locals = set(local_names)
            for statement in block_node.statements:
                self.collect_closure_identifier_captures(
                    statement,
                    block_locals,
                    captures,
                )
            self.collect_closure_identifier_captures(
                self.get_block_expression(block_node),
                block_locals,
                captures,
            )
            return
        if isinstance(node, LetNode):
            self.collect_closure_identifier_captures(
                node.value,
                local_names,
                captures,
            )
            self.add_closure_local_bindings(node.name, local_names)
            return
        if isinstance(node, ReturnNode):
            self.collect_closure_identifier_captures(
                node.value,
                local_names,
                captures,
            )
            return
        if isinstance(node, IfNode):
            self.collect_closure_identifier_captures(
                node.condition,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.if_body,
                set(local_names),
                captures,
            )
            self.collect_closure_identifier_captures(
                node.else_body,
                set(local_names),
                captures,
            )
            return
        if isinstance(node, BinaryOpNode):
            self.collect_closure_identifier_captures(node.left, local_names, captures)
            self.collect_closure_identifier_captures(node.right, local_names, captures)
            return
        if isinstance(node, UnaryOpNode):
            self.collect_closure_identifier_captures(
                node.operand, local_names, captures
            )
            return
        if isinstance(node, FunctionCallNode):
            if isinstance(node.name, MemberAccessNode):
                self.collect_closure_identifier_captures(
                    node.name,
                    local_names,
                    captures,
                )
            elif isinstance(node.name, str):
                if (
                    node.name in local_names
                    or not self.is_closure_allowed_callable(node.name)
                ) and self.is_capturable_identifier(node.name):
                    if node.name not in local_names:
                        captures.add(node.name)
            else:
                self.collect_closure_identifier_captures(
                    node.name,
                    local_names,
                    captures,
                )
            for arg in node.args:
                self.collect_closure_identifier_captures(arg, local_names, captures)
            return
        if isinstance(node, MemberAccessNode):
            self.collect_closure_identifier_captures(node.object, local_names, captures)
            return
        if isinstance(node, AssignmentNode):
            self.collect_closure_identifier_captures(node.left, local_names, captures)
            self.collect_closure_identifier_captures(node.right, local_names, captures)
            return
        if isinstance(node, CastNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, ReferenceNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, DereferenceNode):
            self.collect_closure_identifier_captures(
                node.expression,
                local_names,
                captures,
            )
            return
        if isinstance(node, ArrayAccessNode):
            self.collect_closure_identifier_captures(node.array, local_names, captures)
            self.collect_closure_identifier_captures(node.index, local_names, captures)
            return
        if isinstance(node, VectorConstructorNode):
            for arg in node.args:
                self.collect_closure_identifier_captures(arg, local_names, captures)
            return
        if isinstance(node, TupleNode):
            for element in node.elements:
                self.collect_closure_identifier_captures(
                    element,
                    local_names,
                    captures,
                )
            return
        if isinstance(node, ArrayNode):
            for element in node.elements:
                self.collect_closure_identifier_captures(
                    element,
                    local_names,
                    captures,
                )
            self.collect_closure_identifier_captures(node.size, local_names, captures)
            return
        if isinstance(node, StructInitializationNode):
            for _, value in node.fields:
                self.collect_closure_identifier_captures(value, local_names, captures)
            return
        if isinstance(node, TernaryOpNode):
            self.collect_closure_identifier_captures(
                node.condition,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.true_expr,
                local_names,
                captures,
            )
            self.collect_closure_identifier_captures(
                node.false_expr,
                local_names,
                captures,
            )

    def add_closure_local_bindings(self, pattern, local_names):
        if isinstance(pattern, str):
            if self.is_plain_identifier(pattern) and not self.is_discard_pattern(
                pattern
            ):
                local_names.add(pattern)
            return

        for name in self.collect_pattern_binding_names(pattern):
            local_names.add(name)

    def is_capturable_identifier(self, value):
        if not isinstance(value, str):
            return False
        if "::" in value:
            return False
        if value in {"true", "false", "None", "Some", "Ok", "Err"}:
            return False
        return self.is_plain_identifier(value)

    def is_closure_allowed_callable(self, name):
        if not isinstance(name, str):
            return False
        base_name = name.rsplit("::", 1)[-1]
        if base_name in self.user_function_names:
            return True
        if base_name in self.closure_helper_names:
            return True
        if base_name in self.type_map:
            return True
        if base_name in {"Some", "Ok", "Err"}:
            return True
        return base_name[:1].isupper()

    def is_plain_identifier(self, value):
        return isinstance(value, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value)

    def next_closure_helper_name(self, context_name=None):
        context = self.sanitize_closure_helper_context(context_name)
        while True:
            if context:
                name = f"_rust_closure_{context}_{self.closure_helper_counter}"
            else:
                name = f"_rust_closure_{self.closure_helper_counter}"
            self.closure_helper_counter += 1
            if (
                name not in self.user_function_names
                and name not in self.closure_helper_names
            ):
                return name

    def next_local_function_item_name(self, context_name=None):
        context = self.sanitize_closure_helper_context(context_name)
        while True:
            if context:
                name = f"_rust_local_{context}_{self.local_function_item_counter}"
            else:
                name = f"_rust_local_{self.local_function_item_counter}"
            self.local_function_item_counter += 1
            if (
                name not in self.user_function_names
                and name not in self.closure_helper_names
                and name not in self.local_function_item_names
            ):
                self.local_function_item_names.add(name)
                return name

    def sanitize_closure_helper_context(self, context_name):
        if not context_name:
            return ""
        context = re.sub(r"[^A-Za-z0-9_]+", "_", str(context_name)).strip("_")
        if not context:
            return ""
        if context[0].isdigit():
            context = f"_{context}"
        return context

    def generate_closure_helper_function(self, helper_name, closure):
        parameter_infos = [
            self.prepare_closure_parameter(param) for param in closure.params
        ]
        params = [info["declarator"] for info in parameter_infos]
        param_types = [
            (info["subject"], param.param_type)
            for info, param in zip(parameter_infos, closure.params)
        ]
        pattern_params = [
            (info["subject"], info["pattern"])
            for info in parameter_infos
            if info["pattern"] is not None
        ]
        return_type = self.map_type(closure.return_type)
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = closure.return_type
        self.closure_helper_generation_depth += 1
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        try:
            if pattern_params:
                body = self.generate_closure_pattern_parameter_body(
                    pattern_params,
                    0,
                    2,
                    closure.body,
                )
            else:
                body = self.generate_closure_body_code(closure.body, indent=2)
        finally:
            self.pop_local_callable_scope()
            self.pop_value_type_scope()
            self.closure_helper_generation_depth -= 1
            self.current_function_return_type = previous_return_type

        return (
            f"    {return_type} {helper_name}({', '.join(params)}) {{\n"
            f"{body}"
            "    }\n\n"
        )

    def generate_closure_expression(self, closure):
        parameter_infos = [
            self.prepare_closure_parameter(param) for param in closure.params
        ]
        params = ", ".join(info["declarator"] for info in parameter_infos)
        param_types = [
            (info["subject"], param.param_type)
            for info, param in zip(parameter_infos, closure.params)
        ]
        self.push_value_type_scope(param_types)
        self.push_local_callable_scope()
        try:
            body = self.generate_closure_body(
                closure.body,
                [
                    (info["subject"], info["pattern"])
                    for info in parameter_infos
                    if info["pattern"] is not None
                ],
                closure.return_type,
            )
        finally:
            self.pop_local_callable_scope()
            self.pop_value_type_scope()
        if not params:
            if isinstance(body, str) and not body.strip().startswith("{"):
                return f"lambda({{ return {body}; }})"
            return f"lambda({body})"
        return f"lambda({params}, {body})"

    def prepare_closure_parameter(self, param):
        if self.is_simple_closure_parameter(param):
            name = param.pattern
            return {
                "declarator": self.generate_closure_parameter_declarator(
                    name,
                    param.param_type,
                ),
                "subject": name,
                "pattern": None,
            }

        name = self.next_closure_param_name()
        pattern = None if self.is_discard_pattern(param.pattern) else param.pattern
        return {
            "declarator": self.generate_closure_parameter_declarator(
                name,
                param.param_type,
            ),
            "subject": name,
            "pattern": pattern,
        }

    def is_simple_closure_parameter(self, param):
        return (
            isinstance(param.pattern, str)
            and not self.is_discard_pattern(param.pattern)
            and self.is_simple_pattern_binding(param.pattern)
        )

    def generate_closure_parameter_declarator(self, name, param_type=None):
        if param_type:
            return self.format_typed_declarator(param_type, name)
        return name

    def generate_closure_body(self, body, pattern_params=None, return_type=None):
        pattern_params = pattern_params or []
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = self.get_closure_return_type(
            body,
            return_type,
            previous_return_type,
        )
        try:
            if pattern_params:
                code = self.generate_closure_pattern_parameter_body(
                    pattern_params,
                    0,
                    0,
                    body,
                )
                return f"{{ {self.compact_generated_block(code)} }}"

            if self.is_block_expression_node(body) or self.expression_contains_try(
                body
            ):
                code = self.generate_closure_body_code(body, indent=0)
                return f"{{ {self.compact_generated_block(code)} }}"
            return self.generate_expression(body)
        finally:
            self.current_function_return_type = previous_return_type

    def get_closure_return_type(self, body, return_type, fallback_return_type):
        if return_type:
            return return_type

        inferred = self.infer_closure_return_type(body)
        if inferred is not None:
            return inferred

        return fallback_return_type

    def infer_closure_return_type(self, body):
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            expression = self.get_block_expression(block_node)
            inferred = self.infer_closure_return_type_from_expression(expression)
            if inferred is not None:
                return inferred

            for statement in reversed(block_node.statements):
                inferred = self.infer_closure_return_type_from_expression(statement)
                if inferred is not None:
                    return inferred
            return None

        return self.infer_closure_return_type_from_expression(body)

    def infer_closure_return_type_from_expression(self, expression):
        if expression is None:
            return None

        if isinstance(expression, ReturnNode):
            return self.infer_closure_return_type_from_expression(expression.value)

        if self.is_block_expression_node(expression):
            return self.infer_closure_return_type(expression)

        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name,
            str,
        ):
            constructor = expression.name.rsplit("::", 1)[-1]
            if constructor in {"Ok", "Err"}:
                return "Result"
            if constructor == "Some":
                return "Option"

        if isinstance(expression, str) and expression.rsplit("::", 1)[-1] == "None":
            return "Option"

        if isinstance(expression, TernaryOpNode):
            return self.common_inferred_closure_return_type(
                self.infer_closure_return_type_from_expression(expression.true_expr),
                self.infer_closure_return_type_from_expression(expression.false_expr),
            )

        if isinstance(expression, IfNode):
            return self.common_inferred_closure_return_type(
                self.infer_closure_return_type_from_expression(expression.if_body),
                self.infer_closure_return_type_from_expression(expression.else_body),
            )

        if isinstance(expression, MatchNode):
            inferred = None
            has_inferred_arm = False
            for arm in expression.arms:
                arm_type = self.infer_closure_return_type_from_expression(arm.body)
                if arm_type is None:
                    continue
                if not has_inferred_arm:
                    inferred = arm_type
                    has_inferred_arm = True
                    continue
                if inferred != arm_type:
                    return None
            return inferred

        if isinstance(expression, list):
            for statement in reversed(expression):
                inferred = self.infer_closure_return_type_from_expression(statement)
                if inferred is not None:
                    return inferred

        return None

    def common_inferred_closure_return_type(self, left, right):
        if left is None:
            return right
        if right is None:
            return left
        if left == right:
            return left
        return None

    def generate_closure_pattern_parameter_body(
        self,
        pattern_params,
        index,
        indent,
        body,
    ):
        if index >= len(pattern_params):
            return self.generate_closure_body_code(body, indent)

        subject, pattern = pattern_params[index]

        def success(success_indent):
            return self.generate_closure_pattern_parameter_body(
                pattern_params,
                index + 1,
                success_indent,
                body,
            )

        return self.generate_nested_pattern_match(
            subject,
            pattern,
            indent,
            success,
        )

    def generate_closure_body_code(self, body, indent=0):
        if self.is_block_expression_node(body):
            block_node = self.get_block_expression_node(body)
            self.push_local_callable_scope()
            try:
                code = self.generate_function_body(block_node.statements, indent=indent)
                expression = self.get_block_expression(block_node)
                if expression is not None:
                    code += self.generate_expression_result(
                        expression,
                        indent=indent,
                        result_target=self.return_result_target,
                    )
                return code
            finally:
                self.pop_local_callable_scope()

        return self.generate_expression_result(
            body,
            indent=indent,
            result_target=self.return_result_target,
        )

    def compact_generated_block(self, code):
        lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("//"):
                stripped = f"/* {stripped[2:].strip()} */"
            lines.append(stripped)
        return " ".join(lines)
