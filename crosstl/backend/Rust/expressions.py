"""Rust expressions, expression statements, macros and literals as CrossGL expressions."""

import re

from ..comma_expressions import crossgl_tuple_literal
from .import_tables import (
    RUST_BYTE_CHAR_RE,
    RUST_BYTE_RAW_STRING_RE,
    RUST_BYTE_STRING_RE,
    RUST_C_RAW_STRING_RE,
    RUST_C_STRING_RE,
    RUST_CHAR_RE,
    RUST_NUMERIC_LITERAL_RE,
    RUST_RAW_STRING_RE,
    RUST_STRING_RE,
)
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
    LabeledBlockNode,
    LetPatternConditionNode,
    LoopNode,
    MatchesMacroNode,
    MatchNode,
    MemberAccessNode,
    RangeNode,
    ReferenceNode,
    StructInitializationNode,
    TernaryOpNode,
    TryBlockNode,
    TryNode,
    TupleNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
)


class RustExpressionMixin:
    """Expression, macro and literal members of ``RustToCrossGLConverter``."""

    def generate_discarded_expression(self, expression, indent, loop_contexts=None):
        if isinstance(expression, IfNode):
            return self.generate_discarded_if_expression(
                expression,
                indent,
                loop_contexts,
            )
        if isinstance(expression, MatchNode):
            return self.generate_discarded_match_expression(
                expression,
                indent,
                loop_contexts,
            )
        if self.is_block_expression_node(expression):
            return self.generate_discarded_block_expression(
                expression,
                indent,
                loop_contexts,
            )
        if isinstance(expression, LabeledBlockNode):
            return self.generate_labeled_block_expression_result(
                expression,
                indent,
                None,
                loop_contexts,
            )
        if isinstance(expression, LoopNode):
            return self.generate_loop(
                expression,
                indent,
                result_target=None,
                loop_contexts=loop_contexts,
            )
        if self.expression_has_side_effects(expression):
            return self.generate_expression_result(
                expression,
                indent,
                None,
                loop_contexts,
            )
        return ""

    def generate_discarded_if_expression(self, if_node, indent, loop_contexts=None):
        indent_str = "    " * indent
        condition_code, condition = self.generate_try_expression(
            if_node.condition,
            indent,
            loop_contexts,
        )

        code = condition_code
        code += f"{indent_str}if ({condition}) {{\n"
        code += self.generate_discarded_result_branch(
            if_node.if_body,
            indent + 1,
            loop_contexts,
        )
        code += f"{indent_str}}}"

        if if_node.else_body is not None:
            code += " else {\n"
            code += self.generate_discarded_result_branch(
                if_node.else_body,
                indent + 1,
                loop_contexts,
            )
            code += f"{indent_str}}}"

        code += "\n"
        return code

    def generate_discarded_match_expression(
        self, match_node, indent, loop_contexts=None
    ):
        if self.match_requires_if_chain(match_node):
            return self.generate_match_if_chain(
                match_node,
                indent,
                lambda body, branch_indent, contexts: self.generate_discarded_result_branch(
                    body,
                    branch_indent,
                    contexts,
                ),
                loop_contexts,
            )

        indent_str = "    " * indent
        expression = self.generate_expression(match_node.expression)
        switch_break_flag = self.create_switch_break_flag(match_node, loop_contexts)
        arm_loop_contexts = self.with_switch_break_flag(
            loop_contexts,
            switch_break_flag,
        )

        code = self.generate_switch_break_declaration(switch_break_flag, indent)
        code += f"{indent_str}switch ({expression}) {{\n"

        for arm in match_node.arms:
            code += self.generate_match_case_label(arm.pattern, indent + 1)
            code += self.generate_discarded_result_branch(
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

    def generate_discarded_result_branch(self, branch, indent, loop_contexts=None):
        if self.is_block_expression_node(branch):
            return self.generate_discarded_block_expression(
                branch,
                indent,
                loop_contexts,
            )
        if isinstance(branch, list):
            return self.generate_scoped_function_body(branch, indent, loop_contexts)
        if branch is not None:
            return self.generate_discarded_expression(
                branch,
                indent,
                loop_contexts,
            )
        return ""

    def generate_discarded_block_expression(
        self, block_node, indent, loop_contexts=None
    ):
        block_node = self.get_block_expression_node(block_node)
        self.push_local_callable_scope()
        try:
            code = self.generate_function_body(
                block_node.statements, indent, loop_contexts
            )
            expression = self.get_block_expression(block_node)
            if expression is not None:
                code += self.generate_discarded_expression(
                    expression,
                    indent,
                    loop_contexts,
                )
            return code
        finally:
            self.pop_local_callable_scope()

    def expression_has_side_effects(self, expression):
        if isinstance(expression, FunctionCallNode):
            if self.is_pure_method_call(expression):
                return self.method_call_has_side_effects(expression)
            return True
        if isinstance(expression, TryNode):
            return True
        if isinstance(expression, TryBlockNode):
            return True
        if isinstance(expression, AwaitNode):
            return self.expression_has_side_effects(expression.expression)
        if self.is_block_expression_node(expression):
            block_node = self.get_block_expression_node(expression)
            return bool(block_node.statements) or self.expression_has_side_effects(
                self.get_block_expression(block_node)
            )
        if isinstance(expression, AssignmentNode):
            return True
        if isinstance(expression, IfNode):
            return True
        if isinstance(expression, MatchNode):
            return True
        if isinstance(expression, MatchesMacroNode):
            return self.expression_has_side_effects(
                expression.expression
            ) or self.expression_has_side_effects(expression.guard)
        if isinstance(expression, LetPatternConditionNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, ConditionChainNode):
            return any(
                self.expression_has_side_effects(operand)
                for operand in expression.operands
            )
        if isinstance(expression, ClosureNode):
            return False
        if isinstance(expression, LoopNode):
            return True
        if isinstance(expression, BinaryOpNode):
            return self.expression_has_side_effects(
                expression.left
            ) or self.expression_has_side_effects(expression.right)
        if isinstance(expression, UnaryOpNode):
            return self.expression_has_side_effects(expression.operand)
        if isinstance(expression, CastNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, ReferenceNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, DereferenceNode):
            return self.expression_has_side_effects(expression.expression)
        if isinstance(expression, TupleNode):
            return any(
                self.expression_has_side_effects(element)
                for element in expression.elements
            )
        if isinstance(expression, ArrayNode):
            return any(
                self.expression_has_side_effects(element)
                for element in expression.elements
            ) or self.expression_has_side_effects(expression.size)
        if isinstance(expression, ArrayAccessNode):
            return self.expression_has_side_effects(
                expression.array
            ) or self.expression_has_side_effects(expression.index)
        if isinstance(expression, MemberAccessNode):
            return self.expression_has_side_effects(expression.object)
        if isinstance(expression, VectorConstructorNode):
            return any(self.expression_has_side_effects(arg) for arg in expression.args)
        if isinstance(expression, StructInitializationNode):
            return any(
                self.expression_has_side_effects(field_expression)
                for _, field_expression in expression.fields
            )
        if isinstance(expression, TernaryOpNode):
            return (
                self.expression_has_side_effects(expression.condition)
                or self.expression_has_side_effects(expression.true_expr)
                or self.expression_has_side_effects(expression.false_expr)
            )
        return False

    def is_pure_method_call(self, expression):
        if not (
            isinstance(expression, FunctionCallNode)
            and isinstance(expression.name, MemberAccessNode)
        ):
            return False

        method_name = expression.name.member
        if method_name in {
            "len",
            "length",
            "normalize",
            "max_element",
            "min_element",
        }:
            return not expression.args
        if method_name in {"dot", "cross"}:
            return len(expression.args) == 1
        return not expression.args and self.is_swizzle_member(method_name)

    def method_call_has_side_effects(self, expression):
        return self.expression_has_side_effects(expression.name.object) or any(
            self.expression_has_side_effects(arg) for arg in expression.args
        )

    def is_swizzle_member(self, member):
        if not isinstance(member, str) or not 1 <= len(member) <= 4:
            return False
        return (
            all(char in "xyzw" for char in member)
            or all(char in "rgba" for char in member)
            or all(char in "stpq" for char in member)
        )

    def generate_expression(self, expr):
        """Render a Rust backend expression node as CrossGL syntax."""
        if isinstance(expr, str):
            value = self.resolve_imported_module_path(self.normalize_rust_literal(expr))
            associated_const = self.format_impl_associated_const(value)
            if associated_const is not None:
                return associated_const
            scalar_constant = self.format_scalar_associated_constant(value)
            if scalar_constant is not None:
                return scalar_constant
            vector_constant = self.format_vector_associated_constant(value)
            if vector_constant is not None:
                return vector_constant
            return self.resolve_name_alias(value)
        elif isinstance(expr, VariableNode):
            return self.resolve_name_alias(expr.name)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"({expr.op}{operand})"
        elif isinstance(expr, FunctionCallNode):
            method_call = self.format_method_call(expr)
            if method_call is not None:
                return method_call

            if isinstance(expr.name, str) and expr.name.endswith("!"):
                func_name = self.map_function(expr.name)
                special_macro = self.generate_reparseable_macro_call(func_name, expr)
                if special_macro is not None:
                    return special_macro

                if (
                    self.reparseable_macro_expression_depth
                    and not self.should_preserve_value_macro_bang(func_name)
                ):
                    func_name = func_name[:-1]
                if func_name.endswith("!"):
                    return self.refuse_rust_macro(func_name)
                args = ", ".join(self.generate_macro_argument(arg) for arg in expr.args)
                return f"{func_name}({args})"

            if isinstance(expr.name, str):
                constructor = self.format_path_constructor_call(expr.name, expr.args)
                if constructor is not None:
                    return constructor
                arg_values = [self.generate_expression(arg) for arg in expr.args]
                associated_call = self.format_associated_impl_function_call(
                    expr.name,
                    arg_values,
                )
                if associated_call is not None:
                    return associated_call
                primitive_bitcast_call = (
                    self.format_primitive_float_bitcast_associated_call(
                        expr.name,
                        arg_values,
                    )
                )
                if primitive_bitcast_call is not None:
                    return primitive_bitcast_call
                expression_call = self.format_unary_expression_intrinsic_call(
                    expr.name,
                    arg_values,
                )
                if expression_call is not None:
                    return expression_call
                unresolved_associated_call = self.format_unresolved_associated_call(
                    expr.name,
                    arg_values,
                )
                if unresolved_associated_call is not None:
                    return unresolved_associated_call
                func_name = self.map_function(expr.name)
                args = ", ".join(arg_values)
            else:
                func_name = self.generate_expression(expr.name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            member_name = self.resolve_member_access_name(obj, expr.member)
            return f"{obj}.{member_name}"
        elif isinstance(expr, AwaitNode):
            return self.generate_expression(expr.expression)
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, RangeNode):
            start = "" if expr.start is None else self.generate_expression(expr.start)
            end = "" if expr.end is None else self.generate_expression(expr.end)
            return self.format_range_expression(start, end, expr.inclusive)
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            type_name = self.map_type(expr.type_name)
            return f"{type_name}({args})"
        elif isinstance(expr, TernaryOpNode):
            condition = self.generate_expression(expr.condition)
            true_expr = self.generate_expression(expr.true_expr)
            false_expr = self.generate_expression(expr.false_expr)
            return f"({condition} ? {true_expr} : {false_expr})"
        elif isinstance(
            expr,
            (LoopNode, LabeledBlockNode, IfNode, MatchNode, AssignmentNode),
        ):
            return self.generate_inline_result_expression(expr)
        elif self.is_block_expression_node(expr):
            return self.generate_inline_block_expression(expr)
        elif isinstance(expr, MatchesMacroNode):
            return self.generate_matches_inline_condition(expr)
        elif isinstance(expr, ConditionChainNode):
            return " && ".join(
                f"({self.generate_expression(operand)})"
                for operand in expr.operands
                if not isinstance(operand, LetPatternConditionNode)
            )
        elif isinstance(expr, ClosureNode):
            return self.generate_closure_expression(expr)
        elif isinstance(expr, TryBlockNode):
            return self.generate_try_block_expression(expr)
        elif isinstance(expr, TryNode):
            return self.generate_try_unwrap(self.generate_expression(expr.expression))
        elif isinstance(expr, CastNode):
            expression = self.generate_expression(expr.expression)
            return self.format_cast_expression(expr.target_type, expression)
        elif isinstance(expr, ReferenceNode):
            # References are handled differently in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, DereferenceNode):
            # Dereferences are typically not needed in CrossGL
            return self.generate_expression(expr.expression)
        elif isinstance(expr, TupleNode):
            # ``(a, b,)``: without the trailing comma CrossGL reads the comma operator.
            return crossgl_tuple_literal(
                self.generate_expression(elem) for elem in expr.elements
            )
        elif isinstance(expr, ArrayNode):
            if expr.size is not None and len(expr.elements) == 1:
                repeated = self.expand_repeated_array_literal(
                    expr.elements[0], expr.size
                )
                if repeated is not None:
                    return repeated

            elements = ", ".join(
                self.generate_expression(elem) for elem in expr.elements
            )
            return f"{{{elements}}}"
        elif isinstance(expr, StructInitializationNode):
            fields = ", ".join(
                f"{field_name}: {self.generate_expression(field_value)}"
                for field_name, field_value in expr.fields
            )
            struct_name = self.format_struct_initialization_name(expr.struct_name)
            return f"{struct_name} {{ {fields} }}"
        elif isinstance(expr, (int, float, bool)):
            return str(expr).lower() if isinstance(expr, bool) else str(expr)
        else:
            return str(expr)

    def generate_macro_argument(self, arg):
        if isinstance(arg, str):
            return self.normalize_macro_body(arg)
        return self.generate_expression(arg)

    def format_struct_initialization_name(self, struct_name):
        if struct_name == "Self":
            return struct_name
        return self.map_type(struct_name)

    def generate_global_initializer(self, value):
        if self.is_reparseable_global_macro(value):
            return self.generate_reparseable_global_macro(value)
        return self.generate_reparseable_expression(value)

    def generate_reparseable_macro_call(self, func_name, expr):
        base_name = self.rust_macro_base_name(func_name)

        if base_name in self.VALUE_PASSTHROUGH_MACROS and expr.args:
            return self.normalize_reparseable_macro_body(str(expr.args[0]))

        if base_name in self.OPAQUE_VALUE_MACROS:
            return f"{self.reparseable_macro_function_name(func_name)}()"

        if self.should_preserve_value_macro_bang(func_name):
            return None

        if base_name in self.REPARSEABLE_STATEMENT_MACROS:
            if any(self.macro_argument_needs_opaque_reparse(arg) for arg in expr.args):
                return f"{self.reparseable_macro_function_name(func_name)}()"
            args = ", ".join(
                self.generate_reparseable_global_macro_argument(arg)
                for arg in expr.args
            )
            return f"{self.reparseable_macro_function_name(func_name)}({args})"

        return None

    def macro_argument_needs_opaque_reparse(self, arg):
        if isinstance(arg, str):
            return self.raw_macro_argument_needs_opaque_reparse(arg)
        if isinstance(arg, StructInitializationNode):
            return True
        if isinstance(arg, FunctionCallNode):
            return self.macro_argument_needs_opaque_reparse(arg.name) or any(
                self.macro_argument_needs_opaque_reparse(a) for a in arg.args
            )
        if isinstance(arg, MemberAccessNode):
            return self.macro_argument_needs_opaque_reparse(arg.object)
        if isinstance(arg, ArrayAccessNode):
            return self.macro_argument_needs_opaque_reparse(
                arg.array
            ) or self.macro_argument_needs_opaque_reparse(arg.index)
        if isinstance(arg, ArrayNode):
            return any(
                self.macro_argument_needs_opaque_reparse(element)
                for element in arg.elements
            ) or self.macro_argument_needs_opaque_reparse(arg.size)
        if isinstance(arg, TupleNode):
            return any(
                self.macro_argument_needs_opaque_reparse(element)
                for element in arg.elements
            )
        if isinstance(arg, BinaryOpNode):
            return self.macro_argument_needs_opaque_reparse(
                arg.left
            ) or self.macro_argument_needs_opaque_reparse(arg.right)
        if isinstance(arg, UnaryOpNode):
            return self.macro_argument_needs_opaque_reparse(arg.operand)
        if isinstance(arg, CastNode):
            return self.macro_argument_needs_opaque_reparse(arg.expression)
        if isinstance(arg, ReferenceNode):
            return self.macro_argument_needs_opaque_reparse(arg.expression)
        if isinstance(arg, DereferenceNode):
            return self.macro_argument_needs_opaque_reparse(arg.expression)
        if isinstance(arg, TernaryOpNode):
            return (
                self.macro_argument_needs_opaque_reparse(arg.condition)
                or self.macro_argument_needs_opaque_reparse(arg.true_expr)
                or self.macro_argument_needs_opaque_reparse(arg.false_expr)
            )
        return False

    def raw_macro_argument_needs_opaque_reparse(self, arg):
        return bool(re.search(r"\b[A-Za-z_][A-Za-z0-9_:]*\s*\{[^{}]*:", arg))

    def reparseable_macro_function_name(self, func_name):
        if isinstance(func_name, str) and func_name.endswith("!"):
            return func_name[:-1]
        return func_name

    def rust_macro_base_name(self, func_name):
        if not isinstance(func_name, str):
            return ""
        return self.reparseable_macro_function_name(func_name).rsplit("::", 1)[-1]

    INLINE_ASSEMBLY_MACROS = frozenset({"asm", "global_asm", "naked_asm"})

    def refuse_rust_macro(self, func_name):
        """Refuse a macro call that has no CrossGL form; writing ``name!(...)`` would not be CrossGL."""
        base_name = self.rust_macro_base_name(func_name)
        if base_name in self.INLINE_ASSEMBLY_MACROS:
            reason = (
                f"inline assembly '{base_name}!' is written for one instruction set, and CrossGL has no "
                "inline assembly"
            )
        else:
            reason = (
                f"the macro '{func_name}' has no CrossGL translation: its expansion is not known to the "
                "importer"
            )
        return self.refuse_import(reason, placeholder="0")

    def generate_reparseable_expression(self, expr):
        self.reparseable_macro_expression_depth += 1
        try:
            return self.generate_expression(expr)
        finally:
            self.reparseable_macro_expression_depth -= 1

    def should_preserve_value_macro_bang(self, func_name):
        if not isinstance(func_name, str) or not func_name.endswith("!"):
            return False

        base_name = func_name[:-1].rsplit("::", 1)[-1]
        return base_name in self.SOURCE_VALUE_MACROS

    def is_reparseable_global_macro(self, value):
        if not isinstance(value, FunctionCallNode) or not isinstance(value.name, str):
            return False

        name = value.name[:-1] if value.name.endswith("!") else value.name
        return name.rsplit("::", 1)[-1] in self.GLOBAL_INITIALIZER_MACROS

    def generate_reparseable_global_macro(self, value):
        name = self.map_function(value.name)
        if isinstance(name, str) and name.endswith("!"):
            name = name[:-1]

        args = ", ".join(
            self.generate_reparseable_global_macro_argument(arg) for arg in value.args
        )
        return f"{name}({args})"

    def generate_reparseable_global_macro_argument(self, arg):
        if isinstance(arg, str):
            return self.normalize_reparseable_macro_body(arg)
        return self.generate_global_initializer(arg)

    def format_cast_expression(self, target_type, expression):
        if target_type == "_":
            return expression

        mapped_type = self.map_type(target_type)
        if mapped_type == "_":
            return expression
        return f"{mapped_type}({expression})"

    def normalize_macro_body(self, body):
        literal = self.normalize_macro_literal_body(body)
        if literal is not None:
            return literal
        return body.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    def normalize_reparseable_macro_body(self, body):
        literal = self.normalize_macro_literal_body(body)
        if literal is not None:
            return literal
        body = self.normalize_macro_body(body)
        return re.sub(
            r"\b([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)!\s*(?=[({\[])",
            r"\1",
            body,
        )

    def normalize_macro_literal_body(self, body):
        stripped = body.strip()
        if not self.is_rust_literal(stripped):
            return None
        return self.normalize_rust_literal(stripped)

    def is_rust_literal(self, value):
        return any(
            pattern.match(value)
            for pattern in (
                RUST_BYTE_RAW_STRING_RE,
                RUST_C_RAW_STRING_RE,
                RUST_RAW_STRING_RE,
                RUST_STRING_RE,
                RUST_BYTE_STRING_RE,
                RUST_C_STRING_RE,
                RUST_BYTE_CHAR_RE,
                RUST_CHAR_RE,
                RUST_NUMERIC_LITERAL_RE,
            )
        )

    def generate_inline_result_expression(self, expression):
        value_name = self.next_inline_expression_value_name()
        code = f"auto {value_name};\n"
        code += self.generate_expression_result(expression, 0, value_name)
        code += f"return {value_name};\n"
        return f"lambda({{ {self.compact_generated_block(code)} }})()"

    def generate_inline_block_expression(self, expression):
        block_node = self.get_block_expression_node(expression)
        block_expression = self.get_block_expression(block_node)

        if not block_node.statements:
            if block_expression is None:
                return ""
            return self.generate_expression(block_expression)

        return self.generate_inline_result_expression(expression)

    def generate_matches_inline_condition(self, matches_node):
        condition = self.generate_match_pattern_condition(
            matches_node.expression,
            matches_node.pattern,
        )
        condition = condition or "true"

        if matches_node.guard is None:
            return f"({condition})"

        if self.collect_pattern_binding_names(matches_node.pattern):
            return f"({condition})"

        guard = self.generate_expression(matches_node.guard)
        return f"(({condition}) && ({guard}))"

    def normalize_rust_literal(self, value):
        byte_raw_string = RUST_BYTE_RAW_STRING_RE.match(value)
        if byte_raw_string:
            return self.normalize_raw_string_literal(byte_raw_string.group(2))

        c_raw_string = RUST_C_RAW_STRING_RE.match(value)
        if c_raw_string:
            return self.normalize_raw_string_literal(c_raw_string.group(2))

        raw_string = RUST_RAW_STRING_RE.match(value)
        if raw_string:
            return self.normalize_raw_string_literal(raw_string.group(2))

        string = RUST_STRING_RE.match(value)
        if string:
            return self.normalize_string_literal(string.group(1))

        byte_string = RUST_BYTE_STRING_RE.match(value)
        if byte_string:
            return self.normalize_byte_string_literal(byte_string.group(1))

        c_string = RUST_C_STRING_RE.match(value)
        if c_string:
            return self.normalize_string_literal(c_string.group(1))

        byte_char = RUST_BYTE_CHAR_RE.match(value)
        if byte_char:
            return self.normalize_byte_char_literal(byte_char.group(1))

        char = RUST_CHAR_RE.match(value)
        if char:
            return self.normalize_char_literal(char.group(1))

        return self.normalize_numeric_literal(value)

    def normalize_char_literal(self, body):
        if body.startswith("\\x"):
            return str(int(body[2:], 16))
        if body.startswith("\\u{") and body.endswith("}"):
            return str(int(body[3:-1].replace("_", ""), 16))
        return f"'{body}'"

    def normalize_numeric_literal(self, value):
        match = RUST_NUMERIC_LITERAL_RE.match(value)
        if not match:
            return value

        body = match.group("body").replace("_", "")
        if body.startswith(("0x", "0X")):
            return str(int(body, 16))
        if body.startswith(("0b", "0B")):
            return str(int(body, 2))
        if body.startswith(("0o", "0O")):
            return str(int(body, 8))
        return body

    def normalize_raw_string_literal(self, content):
        escaped = (
            content.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        return f'"{escaped}"'

    def normalize_string_literal(self, content):
        escaped = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{escaped}"'

    def normalize_byte_string_literal(self, content):
        return self.normalize_string_literal(content)

    def normalize_byte_char_literal(self, content):
        if len(content) == 1:
            return str(ord(content))

        escape_values = {
            r"\0": 0,
            r"\n": 10,
            r"\r": 13,
            r"\t": 9,
            r"\\": 92,
            r"\'": 39,
            r"\"": 34,
        }
        if content in escape_values:
            return str(escape_values[content])

        if len(content) == 4 and content.startswith(r"\x"):
            return str(int(content[2:], 16))

        return f"b'{content}'"
