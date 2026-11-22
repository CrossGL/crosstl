"""Expression precedence levels, unary expressions and cast detection in the Metal parser."""

from ..MetalAst import (
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    FunctionCallNode,
    TernaryOpNode,
    UnaryOpNode,
)
from .parser_tables import (
    CONSTRUCTOR_TYPE_TOKENS,
    QUALIFIER_TOKENS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    SIGNED_TYPE_PREFIXES,
    TYPE_TOKENS,
    UNARY_KEYWORDS,
)


class MetalExpressionParsingMixin:
    """Expression members of ``MetalParser``."""

    def parse_expression(self, allow_comma=False):
        if allow_comma:
            return self.parse_comma_expression()
        return self.parse_assignment()

    def parse_comma_expression(self):
        left = self.parse_assignment()
        while self.current_token[0] == "COMMA":
            op = self.current_token[1]
            self.eat("COMMA")
            right = self.parse_assignment()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment(self):
        left = self.parse_conditional()
        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "ASSIGN_MOD",
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment()
            return AssignmentNode(left, right, op)
        return left

    def parse_conditional(self):
        left = self.parse_logical_or()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(left, true_expr, false_expr)
        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.current_token[0] == "OR":
            op = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_logical_and(self):
        left = self.parse_bitwise_or()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_or(self):
        left = self.parse_bitwise_xor()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_xor(self):
        left = self.parse_bitwise_and()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_and(self):
        left = self.parse_equality()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_relational(self):
        left = self.parse_shift()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_shift(self):
        left = self.parse_additive()
        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_unary(self):
        if self.current_token[0] in UNARY_KEYWORDS:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "LPAREN" and self.is_type_in_parens():
                self.eat("LPAREN")
                type_name, _quals = self.parse_type_specifier()
                self.eat("RPAREN")
                return FunctionCallNode(op, [type_name])
            operand = self.parse_unary()
            return FunctionCallNode(op, [operand])
        if self.current_token[0] == "LPAREN" and self.is_type_in_parens(
            require_cast_operand=True
        ):
            self.eat("LPAREN")
            type_name, qualifiers = self.parse_type_specifier()
            self.eat("RPAREN")
            operand = self.parse_unary()
            return CastNode(type_name, operand, qualifiers=qualifiers)
        if self.current_token[0] in [
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
            "INCREMENT",
            "DECREMENT",
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_postfix()

    def is_type_in_parens(self, *, require_cast_operand=False):
        if self.current_token[0] != "LPAREN":
            return False
        idx = self.pos + 1
        saw_qualifier = False
        while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
            saw_qualifier = True
            idx += 1
        if idx >= len(self.tokens):
            return False
        tok_type = self.tokens[idx][0]
        typename_prefix = tok_type == "IDENTIFIER" and self.tokens[idx][1] == "typename"
        if typename_prefix:
            idx += 1
            if idx >= len(self.tokens):
                return False
            tok_type = self.tokens[idx][0]
        if tok_type not in TYPE_TOKENS:
            return False
        type_reference_start = idx
        type_name_evidence = (
            typename_prefix or saw_qualifier or tok_type != "IDENTIFIER"
        )
        if tok_type == "IDENTIFIER":
            name = self.tokens[idx][1]
            type_name_evidence = type_name_evidence or name in self.known_types
            next_type = self.tokens[idx + 1][0] if idx + 1 < len(self.tokens) else "EOF"
            if name in SIGNED_TYPE_PREFIXES and next_type in TYPE_TOKENS:
                type_name_evidence = True
                idx += 2
            else:
                if (
                    not typename_prefix
                    and name not in self.known_types
                    and next_type not in {"SCOPE", "LESS_THAN"}
                    and not (
                        saw_qualifier
                        and next_type in {"MULTIPLY", "BITWISE_AND", "RPAREN"}
                    )
                    and not (
                        next_type == "RPAREN" and self.is_cast_operand_start_at(idx + 2)
                    )
                ):
                    return False
                idx += 1
        else:
            idx += 1
        idx = self.skip_type_reference_suffix_at(idx)
        if any(
            token_type == "SCOPE"
            for token_type, _token_value in self.tokens[type_reference_start:idx]
        ):
            tail_name = self.scoped_type_tail_name_at(type_reference_start, idx)
            type_name_evidence = (
                typename_prefix or saw_qualifier or tail_name in self.known_types
            )
        saw_pointer_suffix = False
        while idx < len(self.tokens) and self.tokens[idx][0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            saw_pointer_suffix = True
            idx += 1
        type_name_evidence = type_name_evidence or saw_pointer_suffix
        if idx >= len(self.tokens) or self.tokens[idx][0] != "RPAREN":
            return False
        if require_cast_operand:
            return self.is_cast_operand_start_at(
                idx + 1,
                allow_unary=type_name_evidence,
            )
        return True

    def scoped_type_tail_name_at(self, start, end):
        tail_name = self.tokens[start][1]
        angle_depth = 0
        expect_scoped_part = False
        for token_type, token_value in self.tokens[start + 1 : end]:
            if token_type == "LESS_THAN":
                angle_depth += 1
                continue
            if token_type == "GREATER_THAN" and angle_depth > 0:
                angle_depth -= 1
                continue
            if token_type == "SHIFT_RIGHT" and angle_depth > 0:
                angle_depth = max(0, angle_depth - 2)
                continue
            if angle_depth > 0:
                continue
            if token_type == "SCOPE":
                expect_scoped_part = True
                continue
            if not expect_scoped_part:
                continue
            if (token_type, token_value) == ("IDENTIFIER", "template"):
                continue
            if token_type in SCOPED_IDENTIFIER_PART_TOKENS:
                tail_name = token_value
            expect_scoped_part = False
        return tail_name

    def is_cast_operand_start_at(self, idx, *, allow_unary=False):
        if idx >= len(self.tokens):
            return False
        token_type = self.tokens[idx][0]
        if token_type in CONSTRUCTOR_TYPE_TOKENS or token_type in {
            "IDENTIFIER",
            "METAL",
            "SCOPE",
            "NUMBER",
            "TRUE",
            "FALSE",
            "CHAR_LITERAL",
            "STRING",
            "LPAREN",
            "LBRACE",
        }:
            return True
        return allow_unary and token_type in {
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
            "INCREMENT",
            "DECREMENT",
            "MULTIPLY",
            "BITWISE_AND",
        }
