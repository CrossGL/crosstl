"""Expression precedence levels, postfix expressions, template suffixes, function calls, kernel launches and primary expressions in the HIP parser."""

from ..HipAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    HipBuiltinNode,
    KernelLaunchNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)


class HipExpressionMixin:
    """Expression members of ``HipParser``."""

    def parse_expression(self, allow_comma=False):
        expr = self.parse_assignment_expression()
        if not allow_comma:
            return expr

        self.skip_newlines()
        while self.match("COMMA"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_assignment_expression()
            expr = BinaryOpNode(expr, op, right)
            self.skip_newlines()

        return expr

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()
        self.skip_newlines()

        if self.match(
            "ASSIGN",
            "PLUS_ASSIGN",
            "MINUS_ASSIGN",
            "MULTIPLY_ASSIGN",
            "DIVIDE_ASSIGN",
            "STAR_ASSIGN",
            "SLASH_ASSIGN",
            "PERCENT_ASSIGN",
            "AND_ASSIGN",
            "OR_ASSIGN",
            "XOR_ASSIGN",
            "LSHIFT_ASSIGN",
            "RSHIFT_ASSIGN",
        ):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_ternary_expression(self):
        expr = self.parse_logical_or_expression()
        self.skip_newlines()

        if self.match("QUESTION"):
            self.advance()
            self.skip_newlines()
            true_expr = self.parse_expression()
            self.skip_newlines()
            self.consume("COLON")
            self.skip_newlines()
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()
        self.skip_newlines()

        while self.match("LOGICAL_OR", "OR"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()
        self.skip_newlines()

        while self.match("LOGICAL_AND", "AND"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()
        self.skip_newlines()

        while self.match("BITWISE_OR", "PIPE"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()
        self.skip_newlines()

        while self.match("BITWISE_XOR", "XOR"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()
        self.skip_newlines()

        while self.match("BITWISE_AND", "AMPERSAND"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()
        self.skip_newlines()

        while self.match("EQ", "NE"):
            op = self.normalized_operator_value()
            self.advance()
            self.skip_newlines()
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()
        self.skip_newlines()

        while self.match("LT", "LE", "GT", "GE"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_shift_expression(self):
        left = self.parse_additive_expression()
        self.skip_newlines()

        while self.match("SHIFT_LEFT", "SHIFT_RIGHT", "LSHIFT", "RSHIFT"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_additive_expression(self):
        left = self.parse_multiplicative_expression()
        self.skip_newlines()

        while self.match("PLUS", "MINUS"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_unary_expression()
        self.skip_newlines()

        while self.match("MULTIPLY", "STAR", "DIVIDE", "SLASH", "MODULO", "PERCENT"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_unary_expression(self):
        if self.match(
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
            "TILDE",
            "INCREMENT",
            "DECREMENT",
            "STAR",
            "AMPERSAND",
        ):
            op = self.normalized_operator_value()
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        expr = self.parse_primary_expression()

        while True:
            self.skip_newlines()
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                index = self.parse_expression()
                self.consume("RBRACKET")
                expr = ArrayAccessNode(expr, index)
            elif self.match("SCOPE"):
                self.consume("SCOPE")
                member = self.consume_qualified_name_member()
                expr = self.append_qualified_name(expr, "::", member)
            elif self.match("LT") and self.is_template_suffix():
                expr = self.append_template_suffix(expr)
            elif self.match("DOT"):
                self.consume("DOT")
                member = self.parse_member_name()
                expr = MemberAccessNode(expr, member, False)
            elif self.match("ARROW"):
                self.consume("ARROW")
                member = self.parse_member_name()
                expr = MemberAccessNode(expr, member, True)
            elif self.match("LPAREN"):
                self.consume("LPAREN")
                self.skip_newlines()
                if expr == "sizeof" and self.is_sizeof_type_operand():
                    args = [self.parse_type()]
                    self.skip_newlines()
                else:
                    args = self.parse_argument_list()
                self.consume("RPAREN")
                expr = self.parse_function_call_node(expr, args)
            elif self.match("LBRACE"):
                initializer = self.parse_initializer_list()
                expr = self.parse_function_call_node(expr, initializer.elements)
            elif self.match("KERNEL_LAUNCH_START"):
                expr = self.parse_kernel_launch(expr)
            elif self.match("ELLIPSIS"):
                self.advance()
                expr = FunctionCallNode(self.PACK_EXPANSION_FUNCTION_NAME, [expr])
            elif self.match("INCREMENT", "DECREMENT"):
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

    def parse_member_name(self):
        if self.match("TEMPLATE"):
            self.advance()
            return self.consume_function_name()
        if self.match(*self.MEMBER_NAME_TOKENS):
            member = self.current_token.value
            self.advance()
            return member
        self.error("Expected member name")

    def consume_qualified_name_member(self):
        if self.match("TEMPLATE"):
            self.advance()
            return self.consume_function_name()
        if self.match("IDENTIFIER") or self.is_type_token(allow_identifier=False):
            member = self.current_token.value
            self.advance()
            return member
        self.error("Expected qualified name member")

    def is_sizeof_type_operand(self):
        saved_pos = self.pos
        try:
            self.skip_newlines()
            saw_integral_sign = False
            while self.match(*self.TYPE_QUALIFIER_TOKENS):
                if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                    saw_integral_sign = True
                self.advance()

            if saw_integral_sign and self.match("RPAREN"):
                return True

            if not self.is_type_token(allow_identifier=False) and not (
                self.match("IDENTIFIER")
                and self.current_token.value in self.type_aliases
            ):
                return False

            token_type = self.current_token.type
            self.advance()
            if token_type in self.ELABORATED_TYPE_TOKENS:
                if token_type == "ENUM" and self.match("CLASS", "STRUCT"):
                    self.advance()
                if not self.match("IDENTIFIER"):
                    return False
                self.advance()
            if token_type == "LONG" and self.match("LONG"):
                self.advance()
            if token_type == "LONG" and self.match("INT"):
                self.advance()

            while self.match("ASTERISK", "STAR"):
                self.advance()

            while self.match("LBRACKET"):
                self.advance()
                while self.current_token and not self.match("RBRACKET"):
                    self.advance()
                if not self.current_token:
                    return False
                self.advance()

            self.skip_newlines()
            return self.match("RPAREN")
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

    def append_qualified_name(self, base, separator, member):
        if isinstance(base, str):
            return f"{base}{separator}{member}"
        return f"{self.expression_to_text(base)}{separator}{member}"

    def is_template_suffix(self):
        index = self.pos
        if self.tokens[index].type != "LT":
            return False

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LT":
                depth += 1
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
            elif token_type == "RSHIFT":
                depth -= 2
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
                if depth < 0:
                    return False
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
                if depth < 0:
                    return False
            elif token_type in {"SEMICOLON", "ASSIGN"}:
                return False
            index += 1

        return False

    def append_template_suffix(self, base):
        suffix = self.parse_template_suffix()
        if isinstance(base, str):
            return f"{base}{suffix}"
        return f"{self.expression_to_text(base)}{suffix}"

    def parse_template_suffix(self):
        self.consume("LT")
        parts = []
        depth = 1

        while depth > 0:
            if self.current_token is None or self.current_token.type == "EOF":
                # The input ended inside the argument list.
                raise self.syntax_error(
                    "Unterminated template argument list: expected '>'"
                )
            token_type = self.current_token.type
            token_value = self.current_token.value
            if token_type == "LT" and self.is_nested_template_lt_at_pos(self.pos):
                depth += 1
                parts.append(token_value)
                self.consume("LT")
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    self.consume("GT")
                    break
                parts.append(token_value)
                self.consume("GT")
            elif token_type == "RSHIFT":
                self.consume("RSHIFT")
                for _ in range(2):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            elif token_type == "KERNEL_LAUNCH_END":
                self.consume("KERNEL_LAUNCH_END")
                for _ in range(3):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            else:
                parts.append(token_value)
                self.consume(token_type)

        return f"<{self.format_template_parts(parts)}>"

    def is_nested_template_lt_at_pos(self, index):
        previous_index = index - 1
        while previous_index >= 0 and self.tokens[previous_index].type == "NEWLINE":
            previous_index -= 1

        if previous_index < 0:
            return True

        previous_type = self.tokens[previous_index].type
        previous_value = self.tokens[previous_index].value
        if previous_type == "IDENTIFIER" and previous_value == "sizeof":
            return False
        return previous_type in {
            "IDENTIFIER",
            *self.BUILTIN_TYPE_TOKENS,
            *self.VECTOR_TYPE_TOKENS,
            *self.RESOURCE_TYPE_TOKENS,
            *self.ELABORATED_TYPE_TOKENS,
            "TYPENAME",
            "TEMPLATE",
            "GT",
            "RSHIFT",
            "KERNEL_LAUNCH_END",
        }

    def format_template_parts(self, parts):
        formatted = []
        previous = None

        for part in parts:
            if part == "\n":
                continue
            if part == ",":
                formatted.append(", ")
            else:
                if self.needs_template_part_space(previous, part):
                    formatted.append(" ")
                formatted.append(part)
            previous = part

        return "".join(formatted).strip()

    def needs_template_part_space(self, previous, current):
        if previous is None:
            return False
        if previous in {"<", "[", "(", "::", ","}:
            return False
        if current in {">", "]", ")", ",", "::", "<", "[", "(", "*", "&", "&&"}:
            return False
        if previous in {"*", "&", "&&"}:
            return False
        return self.is_template_word(previous) and self.is_template_word(current)

    def is_template_word(self, part):
        return part.replace("_", "").isalnum()

    def parse_function_call_node(self, function_name, args):
        named_cast = self.parse_cpp_named_cast_call(function_name, args)
        if named_cast is not None:
            return named_cast

        if (
            function_name in self.ATOMIC_FUNCTION_NAMES
            and function_name not in self.user_function_names
        ):
            return AtomicOperationNode(function_name, args)

        if (
            function_name == "hipLaunchKernelGGL"
            and len(args) >= 5
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_hip_kernel_function_arg(args[0]),
                args[1],
                args[2],
                args[3],
                args[4],
                args[5:],
            )

        if (
            function_name == "hipLaunchKernel"
            and len(args) == 6
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_hip_kernel_function_arg(args[0]),
                args[1],
                args[2],
                args[4],
                args[5],
                [args[3]],
            )

        return FunctionCallNode(function_name, args)

    def parse_cpp_named_cast_call(self, function_name, args):
        if not isinstance(function_name, str) or len(args) != 1:
            return None

        for cast_name in self.CPP_NAMED_CASTS:
            prefix = f"{cast_name}<"
            if function_name.startswith(prefix) and function_name.endswith(">"):
                return CastNode(function_name[len(prefix) : -1], args[0])

        return None

    def unwrap_hip_kernel_function_arg(self, function_arg):
        if isinstance(function_arg, CastNode):
            return self.unwrap_hip_kernel_function_arg(function_arg.expression)
        if (
            isinstance(function_arg, FunctionCallNode)
            and function_arg.name == "HIP_KERNEL_NAME"
            and len(function_arg.args) == 1
        ):
            return function_arg.args[0]
        return function_arg

    def parse_kernel_launch(self, kernel_name):
        kernel_name = self.unwrap_hip_kernel_function_arg(kernel_name)
        self.consume("KERNEL_LAUNCH_START")

        self.skip_newlines()
        blocks = self.parse_expression()
        self.skip_newlines()
        self.consume("COMMA")
        self.skip_newlines()
        threads = self.parse_expression()
        self.skip_newlines()

        shared_mem = None
        stream = None

        if self.match("COMMA"):
            self.advance()
            self.skip_newlines()
            shared_mem = self.parse_expression()
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
                stream = self.parse_expression()
                self.skip_newlines()

        self.consume("KERNEL_LAUNCH_END")

        self.skip_newlines()
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")

        return KernelLaunchNode(kernel_name, blocks, threads, shared_mem, stream, args)

    def parse_primary_expression(self):
        if self.match("LBRACKET") and self.is_lambda_expression_start():
            return self.parse_lambda_expression()

        if self.match("IDENTIFIER", *self.CONTEXTUAL_IDENTIFIER_TOKENS):
            if self.current_token.value == "new":
                return self.parse_new_expression()

            name = self.current_token.value
            self.advance()

            if name in ["threadIdx", "blockIdx", "blockDim", "gridDim"]:
                component = None
                if self.match("DOT"):
                    self.advance()
                    if self.match("IDENTIFIER"):
                        component = self.current_token.value
                        self.advance()
                return HipBuiltinNode(name, component)

            return name

        elif self.match("THREADIDX", "BLOCKIDX", "BLOCKDIM", "GRIDDIM"):
            name = self.current_token.value
            self.advance()

            component = None
            if self.match("DOT"):
                self.advance()
                if self.match("IDENTIFIER"):
                    component = self.current_token.value
                    self.advance()

            return HipBuiltinNode(name, component)

        elif self.match(*self.FLAT_BUILTIN_TOKEN_MAP):
            return self.parse_flat_builtin_node()

        elif self.match("WARPSIZE"):
            self.advance()
            return HipBuiltinNode("warpSize")

        elif self.match(*self.ATOMIC_FUNCTION_TOKENS):
            name = self.current_token.value
            self.advance()
            return name

        elif self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            return f"::{member}"

        elif self.is_type_constructor_expression_start():
            return self.parse_type_without_array_suffix()

        elif self.match("STRING"):
            return self.parse_string_literal_sequence()

        elif self.match("INTEGER", "FLOAT_NUM", "FLOAT"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("TRUE", "FALSE", "NULL", "NULLPTR", "HIPSUCCESS"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("CHAR_LIT"):
            return self.parse_character_literal()

        elif self.match("SYNCTHREADS", "SYNCWARP"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("LPAREN"):
            if self.is_cast_expression():
                self.consume("LPAREN")
                self.skip_newlines()
                target_type = self.parse_type()
                self.skip_newlines()
                self.consume("RPAREN")
                self.skip_newlines()
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.consume("LPAREN")
            self.skip_newlines()
            expr = self.parse_expression(allow_comma=True)
            self.consume("RPAREN")
            return expr
        elif self.match("LBRACE"):
            return self.parse_initializer_list()

        else:
            self.error(
                f"Unexpected token in expression: {self.current_token.type if self.current_token else 'EOF'}"
            )
