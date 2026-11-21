"""The Rust ``?`` operator and try blocks lowered to CrossGL control flow."""

from .import_tables import BLOCK_EXPRESSION_NODE_TYPES
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
    FunctionCallNode,
    IfNode,
    LetNode,
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
    TryNode,
    TupleNode,
    UnaryOpNode,
    VectorConstructorNode,
)


class RustTryOperatorMixin:
    """``?`` operator members of ``RustToCrossGLConverter``."""

    def generate_try_let_statement(self, stmt, indent, loop_contexts=None):
        indent_str = "    " * indent
        code = ""

        if stmt.vtype:
            code += (
                f"{indent_str}{self.format_typed_declarator(stmt.vtype, stmt.name)};\n"
            )
        else:
            code += f"{indent_str}auto {stmt.name};\n"

        code += self.generate_expression_result(
            stmt.value,
            indent,
            stmt.name,
            loop_contexts,
        )
        return code

    def generate_option_unwrap_return_expression(self, expression):
        fallback_name = self.current_option_unwrap_fallback_name
        if fallback_name is None:
            return None

        if isinstance(expression, str):
            value = self.resolve_imported_module_path(expression)
            if value.rsplit("::", 1)[-1] == "None":
                return fallback_name
            return None

        if not isinstance(expression, FunctionCallNode):
            return None

        name = expression.name
        if isinstance(name, MemberAccessNode):
            name = self.generate_expression(name)
        if not isinstance(name, str):
            return None
        if name.rsplit("::", 1)[-1] != "Some":
            return None

        if not expression.args:
            return fallback_name
        return self.generate_reparseable_expression(expression.args[0])

    def generate_try_assignment_statement(self, stmt, indent, loop_contexts=None):
        target = self.generate_expression(stmt.left)
        if stmt.operator != "=":
            prelude, value = self.generate_try_expression(
                stmt.right,
                indent,
                loop_contexts,
            )
            indent_str = "    " * indent
            return prelude + f"{indent_str}{target} {stmt.operator} {value};\n"
        return self.generate_expression_result(
            stmt.right,
            indent,
            target,
            loop_contexts,
        )

    def generate_try_expression(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent

        if isinstance(expression, TryNode):
            return self.generate_try_node_expression(
                expression,
                indent,
                loop_contexts,
            )

        if self.is_transparent_block_expression_node(expression):
            value_name = self.next_transparent_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_block_expression_result(
                expression,
                indent,
                value_name,
                loop_contexts,
            )
            return code, value_name

        if isinstance(expression, TryBlockNode):
            value_name = self.next_try_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_try_block_expression_result(
                expression,
                indent,
                value_name,
                loop_contexts,
            )
            return code, value_name

        if not self.expression_contains_try(expression):
            return "", self.generate_expression(expression)

        if isinstance(expression, BinaryOpNode):
            left_code, left = self.generate_try_expression(
                expression.left,
                indent,
                loop_contexts,
            )
            right_code, right = self.generate_try_expression(
                expression.right,
                indent,
                loop_contexts,
            )
            return left_code + right_code, self.format_crossgl_binary(
                expression, left, right
            )

        if isinstance(expression, UnaryOpNode):
            code, operand = self.generate_try_expression(
                expression.operand,
                indent,
                loop_contexts,
            )
            return code, f"({self.crossgl_unary_operator(expression)}{operand})"

        if isinstance(expression, FunctionCallNode):
            return self.generate_try_function_call_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, MemberAccessNode):
            code, obj = self.generate_try_expression(
                expression.object,
                indent,
                loop_contexts,
            )
            member_name = self.resolve_member_access_name(obj, expression.member)
            return code, f"{obj}.{member_name}"

        if isinstance(expression, AwaitNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, ArrayAccessNode):
            array_code, array = self.generate_try_expression(
                expression.array,
                indent,
                loop_contexts,
            )
            index_code, index = self.generate_try_expression(
                expression.index,
                indent,
                loop_contexts,
            )
            return array_code + index_code, f"{array}[{index}]"

        if isinstance(expression, VectorConstructorNode):
            code, args = self.generate_try_argument_list(
                expression.args,
                indent,
                loop_contexts,
            )
            type_name = self.map_type(expression.type_name)
            return code, f"{type_name}({', '.join(args)})"

        if isinstance(expression, TernaryOpNode):
            return self.generate_try_ternary_expression(
                expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, CastNode):
            code, value = self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )
            return code, self.format_cast_expression(expression.target_type, value)

        if isinstance(expression, ReferenceNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, DereferenceNode):
            return self.generate_try_expression(
                expression.expression,
                indent,
                loop_contexts,
            )

        if isinstance(expression, TupleNode):
            code, elements = self.generate_try_argument_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            return code, f"({', '.join(elements)})"

        if isinstance(expression, ArrayNode):
            code, elements = self.generate_try_argument_list(
                expression.elements,
                indent,
                loop_contexts,
            )
            if expression.size is not None and len(elements) == 1:
                size_code, size = self.generate_try_expression(
                    expression.size,
                    indent,
                    loop_contexts,
                )
                return code + size_code, f"_rust_repeat({elements[0]}, {size})"
            return code, "{" + ", ".join(elements) + "}"

        if isinstance(expression, RangeNode):
            start_code, start = self.generate_try_range_bound(
                expression.start,
                indent,
                loop_contexts,
            )
            end_code, end = self.generate_try_range_bound(
                expression.end,
                indent,
                loop_contexts,
            )
            return start_code + end_code, self.format_range_expression(
                start,
                end,
                expression.inclusive,
            )

        if isinstance(expression, StructInitializationNode):
            return self.generate_try_struct_initialization_expression(
                expression,
                indent,
                loop_contexts,
            )
        return "", self.generate_expression(expression)

    def generate_try_range_bound(self, bound, indent, loop_contexts=None):
        if bound is None:
            return "", ""
        return self.generate_try_expression(bound, indent, loop_contexts)

    def format_range_expression(self, start, end, inclusive=False):
        operator = "..=" if inclusive else ".."
        return f"{start or ''}{operator}{end or ''}"

    def generate_try_node_expression(self, try_node, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject, code = self.generate_try_subject(
            try_node.expression,
            indent,
            loop_contexts,
        )
        code += self.generate_try_propagation(subject, indent)

        value_name = self.next_try_value_name()
        code += (
            f"{indent_str}auto {value_name} = {self.generate_try_unwrap(subject)};\n"
        )
        return code, value_name

    def generate_try_subject(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        subject_name = self.next_try_subject_name()

        if isinstance(expression, TryBlockNode):
            code = f"{indent_str}auto {subject_name};\n"
            code += self.generate_try_block_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        if isinstance(
            expression,
            (
                LoopNode,
                IfNode,
                MatchNode,
                MatchesMacroNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            code = f"{indent_str}auto {subject_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                subject_name,
                loop_contexts,
            )
            return subject_name, code

        prelude, value = self.generate_try_expression(
            expression,
            indent,
            loop_contexts,
        )
        code = prelude + f"{indent_str}auto {subject_name} = {value};\n"
        return subject_name, code

    def generate_try_propagation(self, subject, indent):
        indent_str = "    " * indent
        if self.current_try_kind() == "option":
            return (
                f"{indent_str}if (is_None({subject})) {{\n"
                f"{indent_str}    return None;\n"
                f"{indent_str}}}\n"
            )

        return (
            f"{indent_str}if (is_Err({subject})) {{\n"
            f"{indent_str}    return Err(unwrap_Err({subject}));\n"
            f"{indent_str}}}\n"
        )

    def generate_try_unwrap(self, subject):
        if self.current_try_kind() == "option":
            return f"unwrap_Some({subject})"
        return f"unwrap_Ok({subject})"

    def current_try_kind(self):
        return_type = self.current_function_return_type or ""
        return_type = self.strip_reference_type(return_type)
        generic = self.parse_generic_type(return_type)
        base_name = generic[0] if generic is not None else return_type
        base_name = base_name.rsplit("::", 1)[-1]

        if base_name == "Option":
            return "option"
        return "result"

    def generate_try_argument_list(self, args, indent, loop_contexts=None):
        code = ""
        values = []

        for arg in args:
            arg_code, arg_value = self.generate_try_expression(
                arg,
                indent,
                loop_contexts,
            )
            code += arg_code
            values.append(arg_value)

        return code, values

    def generate_try_function_call_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        if isinstance(expression.name, MemberAccessNode):
            object_code, obj = self.generate_try_expression(
                expression.name.object,
                indent,
                loop_contexts,
            )
            args_code, args = self.generate_try_argument_list(
                expression.args,
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
                return object_code + args_code, method_call
            return (
                object_code + args_code,
                f"{obj}.{expression.name.member}({', '.join(args)})",
            )

        args_code, args = self.generate_try_argument_list(
            expression.args,
            indent,
            loop_contexts,
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
            default_value = self.format_default_value_call(expression.name, args)
            if default_value is not None:
                return args_code, default_value
            primitive_bitcast_call = (
                self.format_primitive_float_bitcast_associated_call(
                    expression.name,
                    args,
                )
            )
            if primitive_bitcast_call is not None:
                return args_code, primitive_bitcast_call
            unresolved_associated_call = self.format_unresolved_associated_call(
                expression.name,
                args,
            )
            if unresolved_associated_call is not None:
                return args_code, unresolved_associated_call
            return args_code, f"{self.map_function(expression.name)}({', '.join(args)})"

        name_code, name = self.generate_try_expression(
            expression.name,
            indent,
            loop_contexts,
        )
        return name_code + args_code, f"{name}({', '.join(args)})"

    def generate_try_ternary_expression(self, expression, indent, loop_contexts=None):
        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            expression.condition,
            indent,
            loop_contexts,
        )
        result_name = self.next_try_value_name()
        code = condition_code
        code += f"{indent_str}auto {result_name};\n"
        code += f"{indent_str}if ({condition}) {{\n"
        true_code, true_value = self.generate_try_expression(
            expression.true_expr,
            indent + 1,
            loop_contexts,
        )
        code += true_code
        code += f"{indent_str}    {result_name} = {true_value};\n"
        code += f"{indent_str}}} else {{\n"
        false_code, false_value = self.generate_try_expression(
            expression.false_expr,
            indent + 1,
            loop_contexts,
        )
        code += false_code
        code += f"{indent_str}    {result_name} = {false_value};\n"
        code += f"{indent_str}}}\n"
        return code, result_name

    def generate_try_struct_initialization_expression(
        self,
        expression,
        indent,
        loop_contexts=None,
    ):
        code = ""
        fields = []

        for field_name, field_expression in expression.fields:
            field_code, field_value = self.generate_try_expression(
                field_expression,
                indent,
                loop_contexts,
            )
            code += field_code
            fields.append(f"{field_name}: {field_value}")

        struct_name = self.format_struct_initialization_name(expression.struct_name)
        return code, f"{struct_name} {{ {', '.join(fields)} }}"

    def generate_try_block_expression(self, try_block):
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = previous_return_type or "Result"
        try:
            code = self.generate_try_block_body_code(try_block.block, indent=0)
        finally:
            self.current_function_return_type = previous_return_type

        return f"lambda({{ {self.compact_generated_block(code)} }})()"

    def generate_try_block_expression_result(
        self,
        try_block,
        indent,
        result_target,
        loop_contexts=None,
    ):
        if result_target is self.return_result_target:
            previous_return_type = self.current_function_return_type
            self.current_function_return_type = previous_return_type or "Result"
            try:
                return self.generate_try_block_body_code(try_block.block, indent)
            finally:
                self.current_function_return_type = previous_return_type

        if not result_target:
            result_target = self.next_try_block_value_name()

        indent_str = "    " * indent
        previous_return_type = self.current_function_return_type
        self.current_function_return_type = previous_return_type or "Result"
        try:
            block_code = self.generate_try_block_body_code(try_block.block, indent + 1)
        finally:
            self.current_function_return_type = previous_return_type

        code = f"{indent_str}do {{\n"
        code += self.rewrite_try_block_returns_to_target(block_code, result_target)
        code += f"{indent_str}}} while (false);\n"
        return code

    def rewrite_try_block_returns_to_target(self, code, result_target):
        rewritten = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("return ") and stripped.endswith(";"):
                indent = line[: len(line) - len(line.lstrip())]
                value = stripped[len("return ") : -1]
                rewritten.append(f"{indent}{result_target} = {value};")
                rewritten.append(f"{indent}continue;")
            else:
                rewritten.append(line)
        return "\n".join(rewritten) + ("\n" if code.endswith("\n") else "")

    def generate_try_block_body_code(self, block_node, indent=0):
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(block_node.statements, indent=indent)
            expression = self.get_block_expression(block_node)
            code += self.generate_try_block_success_return(expression, indent)
            return code
        finally:
            self.pop_local_callable_scope()

    def generate_try_block_success_return(self, expression, indent):
        indent_str = "    " * indent

        if expression is None:
            return f"{indent_str}return {self.generate_try_success_value('()')};\n"

        if isinstance(
            expression,
            (
                LoopNode,
                IfNode,
                MatchNode,
            )
            + BLOCK_EXPRESSION_NODE_TYPES,
        ):
            value_name = self.next_try_block_value_name()
            code = f"{indent_str}auto {value_name};\n"
            code += self.generate_expression_result(
                expression,
                indent,
                value_name,
            )
            code += (
                f"{indent_str}return "
                f"{self.generate_try_success_value(value_name)};\n"
            )
            return code

        prelude, value = self.generate_try_expression(expression, indent)
        return (
            prelude + f"{indent_str}return {self.generate_try_success_value(value)};\n"
        )

    def generate_try_success_value(self, value):
        if self.current_try_kind() == "option":
            return f"Some({value})"
        return f"Ok({value})"

    def expression_contains_try(self, expression):
        if expression is None:
            return False
        if isinstance(expression, TryNode):
            return True
        if isinstance(expression, TryBlockNode):
            return True
        if isinstance(expression, BinaryOpNode):
            return self.expression_contains_try(
                expression.left
            ) or self.expression_contains_try(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_contains_try(expression.operand)
        if isinstance(expression, FunctionCallNode):
            return self.expression_contains_try(expression.name) or any(
                self.expression_contains_try(arg) for arg in expression.args
            )
        if isinstance(expression, MemberAccessNode):
            return self.expression_contains_try(expression.object)
        if isinstance(expression, AwaitNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_contains_try(
                expression.array
            ) or self.expression_contains_try(expression.index)
        if isinstance(expression, VectorConstructorNode):
            return any(self.expression_contains_try(arg) for arg in expression.args)
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_contains_try(expression.condition)
                or self.expression_contains_try(expression.true_expr)
                or self.expression_contains_try(expression.false_expr)
            )
        if isinstance(expression, CastNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ReferenceNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, DereferenceNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, TupleNode):
            return any(
                self.expression_contains_try(elem) for elem in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_contains_try(elem) for elem in expression.elements
            ) or self.expression_contains_try(expression.size)
        if isinstance(expression, RangeNode):
            return self.expression_contains_try(
                expression.start
            ) or self.expression_contains_try(expression.end)
        if isinstance(expression, StructInitializationNode):
            return any(
                self.expression_contains_try(field_expression)
                for _, field_expression in expression.fields
            )
        if isinstance(expression, MatchesMacroNode):
            return self.expression_contains_try(
                expression.expression
            ) or self.expression_contains_try(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_contains_try(expression.expression)
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_contains_try(operand) for operand in expression.operands
            )
        if isinstance(expression, ClosureNode):
            return False
        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            return any(
                self.expression_contains_try(statement)
                for statement in block_node.statements
            ) or self.expression_contains_try(self.get_block_expression(block_node))
        if isinstance(expression, IfNode):
            return (
                self.expression_contains_try(expression.condition)
                or self.expression_contains_try(expression.if_body)
                or self.expression_contains_try(expression.else_body)
            )
        if isinstance(expression, MatchNode):
            return self.expression_contains_try(expression.expression) or any(
                self.expression_contains_try(arm.guard)
                or self.expression_contains_try(arm.body)
                for arm in expression.arms
            )
        if isinstance(expression, list):
            return any(self.expression_contains_try(item) for item in expression)
        if isinstance(expression, ReturnNode):
            return self.expression_contains_try(expression.value)
        if isinstance(expression, AssignmentNode):
            return self.expression_contains_try(
                expression.left
            ) or self.expression_contains_try(expression.right)
        if isinstance(expression, LetNode):
            return self.expression_contains_try(expression.value)
        return False
