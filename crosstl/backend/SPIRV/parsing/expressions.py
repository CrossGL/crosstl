"""Expressions in the GLSL-style text form accepted by the SPIR-V parser: member access, calls, postfix suffixes, primary expressions, casts, initializer lists and the precedence levels."""

from crosstl.translator.refusals import SpirvSyntaxError

from ...common_ast import InitializerListNode
from ..VulkanAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    MemberAccessNode,
    MethodCallNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..VulkanLexer import VALID_DATA_TYPES


class VulkanExpressionMixin:
    """Expression members of ``VulkanParser``."""

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SpirvSyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_function_call(self, name):
        args = self.parse_call_arguments()
        return FunctionCallNode(name, args)

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def parse_function_call_or_identifier(self):
        func_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACKET" and self.looks_like_array_constructor():
            func_name += self.parse_array_suffixes_as_text()

        if self.current_token[0] == "LPAREN":
            node = self.parse_function_call(func_name)
        else:
            node = VariableNode("", func_name)
        return self.parse_postfix_suffixes(node)

    def looks_like_array_constructor(self):
        index = self.pos
        if self.tokens[index][0] != "LBRACKET":
            return False

        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            depth = 1
            index += 1
            while index < len(self.tokens) and depth:
                token_type = self.tokens[index][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                index += 1
            if depth:
                return False

        return index < len(self.tokens) and self.tokens[index][0] == "LPAREN"

    def parse_postfix_suffixes(self, node):
        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "LPAREN":
                    node = MethodCallNode(node, member, self.parse_call_arguments())
                else:
                    node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue

            return node

    def is_swizzle_member(self, member):
        return (
            isinstance(member, str)
            and 1 <= len(member) <= 4
            and any(
                set(member) <= component_set
                for component_set in self.SWIZZLE_COMPONENT_SETS
            )
        )

    def parse_numeric_postfix_suffixes(self, value):
        if value.lower().endswith("hf"):
            value = value[:-2]
        elif value[-1:] in {"u", "U", "f", "F"}:
            value = value[:-1]

        if (
            value.endswith(".")
            and self.current_token[0] == "IDENTIFIER"
            and self.is_swizzle_member(self.current_token[1])
        ):
            member = self.current_token[1]
            self.eat("IDENTIFIER")
            node = MemberAccessNode(value[:-1] or "0", member)
            return self.parse_postfix_suffixes(node)

        return self.parse_postfix_suffixes(value)

    def looks_like_member_call_statement(self):
        index = self.pos
        if self.tokens[index][0] != "IDENTIFIER":
            return False

        while index + 2 < len(self.tokens):
            if self.tokens[index + 1][0] != "DOT":
                return False
            if self.tokens[index + 2][0] != "IDENTIFIER":
                return False
            index += 2
            if index + 1 < len(self.tokens) and self.tokens[index + 1][0] == "LPAREN":
                return True

        return False

    def parse_array_suffixes_as_text(self):
        suffix = ""
        while self.current_token[0] == "LBRACKET":
            suffix += "["
            self.eat("LBRACKET")
            while self.current_token[0] != "RBRACKET":
                if self.current_token[0] == "EOF":
                    raise SpirvSyntaxError("Unterminated array suffix")
                suffix += str(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("RBRACKET")
            suffix += "]"
        return suffix

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()
            return UnaryOpNode("-", value)

        if (
            self.current_token[0] == "BITWISE_NOT"
            or self.current_token[0] == "BINARY_NOT"
        ):
            self.eat(self.current_token[0])
            value = self.parse_primary()
            return UnaryOpNode("~", value)

        if (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[1] in VALID_DATA_TYPES
        ):
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return self.parse_numeric_postfix_suffixes(value)
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            return value
        elif self.current_token[0] == "LPAREN":
            if self.looks_like_c_style_cast():
                return self.parse_c_style_cast()
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return self.parse_postfix_suffixes(expr)
        elif self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        else:
            raise SpirvSyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def looks_like_c_style_cast(self):
        return (
            self.current_token[0] == "LPAREN"
            and self.peek(1)
            in {
                "IDENTIFIER",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "VEC2",
                "VEC3",
                "VEC4",
                "MAT2",
                "MAT3",
                "MAT4",
            }
            and self.peek(2) == "RPAREN"
            and self.peek(3)
            in {
                "IDENTIFIER",
                "NUMBER",
                "STRING",
                "LPAREN",
                "LBRACE",
                "MINUS",
                "PLUS",
                "NOT",
                "BITWISE_NOT",
                "BINARY_NOT",
            }
        )

    def parse_c_style_cast(self):
        self.eat("LPAREN")
        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        self.eat("RPAREN")
        return self.parse_postfix_suffixes(
            FunctionCallNode(type_name, [self.parse_unary()])
        )

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SpirvSyntaxError("Unterminated initializer list")
            elements.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
                continue
            break
        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            token_type = self.current_token[0]
            op = self.current_token[1]
            self.eat(token_type)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment_or_function_call(
        self,
        terminators=None,
        consume_terminator=True,
    ):
        self.skip_hlsl_attributes()
        terminators = terminators or {"SEMICOLON"}
        type_name = ""
        qualifiers = self.parse_declaration_qualifiers()

        if qualifiers:
            if (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[1] in VALID_DATA_TYPES
            ):
                parsed_type = self.parse_data_type(allow_identifier=True)
                type_name = " ".join([*qualifiers, parsed_type])
            else:
                raise SpirvSyntaxError(
                    f"Unexpected token after declaration qualifier: {self.current_token[0]}"
                )
        elif self.current_token[0] == "IDENTIFIER" and self.peek(1) in [
            "POST_INCREMENT",
            "POST_DECREMENT",
        ]:
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_INCREMENT", VariableNode("", name))
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_DECREMENT", VariableNode("", name))
            else:
                raise SpirvSyntaxError(
                    f"Unexpected token after identifier: {self.current_token[0]}"
                )
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER":
            type_name = self.parse_data_type(allow_identifier=True)
        elif self.current_token[1] in VALID_DATA_TYPES:
            type_name = self.parse_data_type()
        if self.current_token[0] == "IDENTIFIER":
            if type_name and "COMMA" not in terminators:
                return self.parse_variable_declaration(
                    type_name,
                    terminators=terminators,
                    consume_terminator=consume_terminator,
                )
            return self.parse_variable(
                type_name,
                terminators=terminators,
                consume_terminator=consume_terminator,
            )

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()
        if self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op_name)
        return left

    def parse_ternary_expression(self):
        left = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()
        while self.current_token[0] == "OR":
            op_symbol = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()
        while self.current_token[0] == "AND":
            op_symbol = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()
        while self.current_token[0] == "BINARY_OR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()
        while self.current_token[0] == "BINARY_XOR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()
        while self.current_token[0] == "BINARY_AND":
            op_symbol = self.current_token[1]
            self.eat("BINARY_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_shift_expression(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_additive()
            op_symbol = "<<" if op == "BITWISE_SHIFT_LEFT" else ">>"
            left = BinaryOpNode(left, op_symbol, right)

        return left

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_unary())
        if self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_unary())
        return self.parse_primary()
