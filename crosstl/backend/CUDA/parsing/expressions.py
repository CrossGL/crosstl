"""Expression precedence levels, postfix expressions, template suffixes and declarations, function calls, kernel launches, primary expressions and requires expressions in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import (
    ArrayAccessNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    CudaBuiltinNode,
    FunctionCallNode,
    KernelLaunchNode,
    MemberAccessNode,
    TernaryOpNode,
    UnaryOpNode,
)


class CudaExpressionMixin:
    """Expression members of ``CudaParser``."""

    def parse_expression(self, allow_comma=False):
        expr = self.parse_assignment_expression()
        if allow_comma and self.current_token[0] == "COMMA":
            return self.parse_comma_expression(expr)
        return expr

    def parse_comma_expression(self, first_expr):
        expressions = [first_expr]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_assignment_expression())
        return FunctionCallNode("cuda_comma_expression", expressions)

    def parse_ternary_expression(self):
        expr = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.normalized_operator_value()
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.normalized_operator_value()
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()

        while self.current_token[0] == "BITWISE_OR":
            op = self.normalized_operator_value()
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()

        while self.current_token[0] == "BITWISE_XOR":
            op = self.normalized_operator_value()
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()

        while self.current_token[0] == "BITWISE_AND":
            op = self.normalized_operator_value()
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.normalized_operator_value()
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_shift_expression(self):
        left = self.parse_additive_expression()

        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MODULO"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        if self.current_token[0] in [
            "PLUS",
            "MINUS",
            "LOGICAL_NOT",
            "BITWISE_NOT",
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            op = self.normalized_operator_value()
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)
        elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_postfix_expression()
            return UnaryOpNode(op, operand)
        else:
            return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()

        while True:
            if self.is_ellipsis_at_current_index():
                self.consume_ellipsis()
                left = UnaryOpNode("post...", left)
            elif self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.parse_member_name()
                left = MemberAccessNode(left, member, False)
            elif self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                member = self.parse_qualified_name_member()
                left = self.append_qualified_name(left, "::", member)
            elif self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                left = self.append_template_suffix(left)
            elif self.is_arrow_star_lambda_start():
                self.eat("ARROW")
                self.eat("MULTIPLY")
                lambda_node = self.parse_lambda_expression()
                left = FunctionCallNode("operator->*", [left, lambda_node])
            elif self.current_token[0] == "ARROW":
                self.eat("ARROW")
                member = self.parse_member_name()
                left = MemberAccessNode(left, member, True)
            elif self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                left = ArrayAccessNode(left, index)
            elif self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                args = []
                if left == "sizeof" and self.is_sizeof_type_operand():
                    args.append(self.parse_type())
                else:
                    self.parse_expression_arguments(args, "RPAREN")
                self.eat("RPAREN")

                if (
                    isinstance(left, str)
                    and left in self.ATOMIC_FUNCTION_NAMES
                    and left not in self.user_function_names
                ):
                    left = AtomicOperationNode(left, args)
                else:
                    left = self.parse_function_call_node(left, args)
            elif self.current_token[0] == "LBRACE" and self.is_braced_constructor(left):
                initializer = self.parse_initializer_list()
                left = FunctionCallNode(left, initializer.elements)
            elif self.current_token[0] == "KERNEL_LAUNCH_START":
                return self.parse_kernel_launch(left)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(f"post{op}", left)
            else:
                break

        return left

    def is_arrow_star_lambda_start(self):
        return (
            self.current_token[0] == "ARROW"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "MULTIPLY"
            and self.tokens[self.current_index + 2][0] == "LBRACKET"
        )

    def parse_member_name(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            member = self.consume_function_name()
            if self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                member += self.parse_template_suffix()
            return member
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            return self.parse_name_component()
        raise CudaSyntaxError(f"Expected IDENTIFIER, got {self.current_token[0]}")

    def parse_qualified_name_member(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            member = self.consume_function_name()
            if self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                member += self.parse_template_suffix()
            return member
        return self.parse_name_component()

    def parse_name_component(self):
        if self.current_token[0] not in self.NAME_COMPONENT_TOKENS:
            raise CudaSyntaxError(f"Expected IDENTIFIER, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def is_braced_constructor(self, callee):
        return isinstance(callee, str)

    def append_qualified_name(self, base, separator, member):
        if isinstance(base, str):
            return f"{base}{separator}{member}"
        return f"{self.expression_to_text(base)}{separator}{member}"

    def is_template_suffix(self):
        index = self.current_index
        if self.tokens[index][0] != "LESS_THAN":
            return False

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
                    }
            elif token_type == "SHIFT_RIGHT":
                depth -= 2
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
                    }
                if depth < 0:
                    return False
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
                    }
                if depth < 0:
                    return False
            elif token_type in {"SEMICOLON", "ASSIGN", "EOF"}:
                return False
            index += 1

        return False

    def append_template_suffix(self, base):
        suffix = self.parse_template_suffix()
        if isinstance(base, str):
            return f"{base}{suffix}"
        if isinstance(base, MemberAccessNode):
            # ``x.f<T>`` names member ``f<T>`` of ``x``.
            return MemberAccessNode(
                base.object, f"{base.member}{suffix}", base.is_pointer
            )
        return f"{self.expression_to_text(base)}{suffix}"

    def parse_template_suffix(self):
        self.eat("LESS_THAN")
        parts = []
        depth = 1

        while depth > 0:
            token_type, token_value = self.current_token
            if token_type == "LESS_THAN":
                depth += 1
                parts.append(token_value)
                self.eat("LESS_THAN")
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(token_value)
                self.eat("GREATER_THAN")
            elif token_type == "SHIFT_RIGHT":
                self.eat("SHIFT_RIGHT")
                for _ in range(2):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            elif token_type == "KERNEL_LAUNCH_END":
                self.eat("KERNEL_LAUNCH_END")
                for _ in range(3):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            else:
                parts.append(token_value)
                self.eat(token_type)

        return f"<{self.format_template_parts(parts)}>"

    def parse_template_declaration(self):
        self.eat("TEMPLATE")
        if self.current_token[0] == "LESS_THAN":
            header = self.parse_template_suffix()
            self.skip_leading_requires_clause()
            # The function that starts here takes the header (``take_template_header``).
            self.pending_template_header = (header, self.current_index)
            return

        self.collect_balanced_tokens_until("SEMICOLON")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_leading_requires_clause(self):
        if not self.is_identifier_value("requires"):
            return

        self.eat("IDENTIFIER")
        if self.is_requires_expression_start():
            self.parse_requires_expression()
        elif self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()

    def format_template_parts(self, parts):
        formatted = []
        previous = None

        for part in parts:
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
            function_name in {"cudaLaunchKernel", "cudaLaunchCooperativeKernel"}
            and len(args) == 6
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_cuda_kernel_function_arg(args[0]),
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

    def unwrap_cuda_kernel_function_arg(self, function_arg):
        if isinstance(function_arg, CastNode):
            return self.unwrap_cuda_kernel_function_arg(function_arg.expression)
        if isinstance(function_arg, UnaryOpNode) and function_arg.op == "&":
            return function_arg.operand
        return function_arg

    def is_sizeof_type_operand(self):
        type_end = self.skip_type_at_index(self.current_index)
        return (
            type_end is not None
            and type_end < len(self.tokens)
            and self.tokens[type_end][0] == "RPAREN"
        )

    def parse_kernel_launch(self, kernel_name):
        self.eat("KERNEL_LAUNCH_START")

        blocks = self.parse_expression()
        threads = None

        shared_mem = None
        stream = None

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            threads = self.parse_expression()

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                shared_mem = self.parse_expression()

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    stream = self.parse_expression()

        self.eat("KERNEL_LAUNCH_END")

        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")

        return KernelLaunchNode(kernel_name, blocks, threads, shared_mem, stream, args)

    def parse_primary_expression(self):
        if self.current_token[0] == "LBRACKET" and self.is_lambda_expression_start():
            return self.parse_lambda_expression()
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            if self.current_token[0] == "IDENTIFIER" and self.current_token[
                1
            ].startswith("_"):
                value += self.current_token[1]
                self.eat("IDENTIFIER")
            return value
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            while self.current_token[0] == "STRING":
                next_value = self.current_token[1]
                if value.endswith('"') and next_value.startswith('"'):
                    value = value[:-1] + next_value[1:]
                else:
                    value += next_value
                self.eat("STRING")
            return value
        elif self.current_token[0] == "CHAR_LIT":
            value = self.current_token[1]
            self.eat("CHAR_LIT")
            return value
        elif self.current_token[0] in ["TRUE", "FALSE"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif self.current_token[0] in ["NULL", "NULLPTR"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif (
            self.current_token[0] in self.TYPE_TOKENS
            and self.current_token[0] != "IDENTIFIER"
        ):
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif (
            self.current_token[0] in {"SIGNED", "UNSIGNED"}
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        ):
            # `unsigned(x)`: a lone sign keyword is a simple-type-specifier naming
            # int, so it can start a functional cast (C++ [expr.type.conv]).
            target_type = (
                "unsigned int" if self.current_token[0] == "UNSIGNED" else "int"
            )
            self.eat(self.current_token[0])
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return CastNode(target_type, expr)
        elif self.current_token[0] == "IDENTIFIER":
            if self.is_requires_expression_start():
                return self.parse_requires_expression()
            if self.current_token[1] == "new":
                return self.parse_new_expression()

            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name
        elif self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            name = self.parse_name_component()
            return f"::{name}"
        elif self.current_token[0] in self.ATOMIC_FUNCTION_TOKENS:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            return name
        elif self.current_token[0] in [
            "THREADIDX",
            "BLOCKIDX",
            "GRIDDIM",
            "BLOCKDIM",
            "WARPSIZE",
        ]:
            builtin_name = self.current_token[1]
            self.eat(self.current_token[0])

            if self.current_token[0] == "DOT":
                self.eat("DOT")
                component = self.eat("IDENTIFIER")[1]
                return CudaBuiltinNode(builtin_name, component)
            else:
                return CudaBuiltinNode(builtin_name)
        elif self.current_token[0] == "LPAREN":
            if self.is_parenthesized_fold_expression_start():
                return self.parse_parenthesized_fold_expression()
            if self.is_cast_expression():
                self.eat("LPAREN")
                target_type = self.parse_type()
                self.eat("RPAREN")
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.eat("LPAREN")
            expr = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        else:
            raise CudaSyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def is_requires_expression_start(self):
        if (
            self.current_token[0] != "IDENTIFIER"
            or self.current_token[1] != "requires"
            or self.current_index + 1 >= len(self.tokens)
        ):
            return False

        next_type = self.tokens[self.current_index + 1][0]
        if next_type == "LBRACE":
            return True
        if next_type != "LPAREN":
            return False

        body_index = self.skip_balanced_tokens_at_index(
            self.current_index + 1, "LPAREN", "RPAREN"
        )
        return body_index < len(self.tokens) and self.tokens[body_index][0] == "LBRACE"

    def parse_requires_expression(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        return "true"
