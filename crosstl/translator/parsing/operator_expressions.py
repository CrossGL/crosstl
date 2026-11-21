"""Operator expressions in the CrossGL parser: ranges, assignments, the binary precedence levels, unary operators and C-style casts."""

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    PointerReinterpretNode,
    PointerType,
    RangeNode,
    TernaryOpNode,
    UnaryOpNode,
)
from ..refusals import CrossGLSyntaxError
from .vocabulary import RESOURCE_ADDRESS_SPACE_NAMES


class ParserOperatorExpressionMixin:
    """Operator expression members of ``Parser``."""

    def parse_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse an expression using the highest-precedence entry point."""
        return self.parse_assignment_expression(
            stop_at_generic_close, reject_logical_at_top_level
        )

    def parse_range_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse range and inclusive-range expressions."""
        left = self.parse_ternary_expression(
            stop_at_generic_close, reject_logical_at_top_level
        )

        if self.current_token[0] in ["RANGE", "RANGE_INCLUSIVE"]:
            inclusive = self.current_token[0] == "RANGE_INCLUSIVE"
            self.eat(self.current_token[0])
            right = self.parse_ternary_expression(
                stop_at_generic_close, reject_logical_at_top_level
            )
            return RangeNode(left, right, inclusive=inclusive)

        return left

    def parse_assignment_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse assignment and compound-assignment expressions."""
        left = self.parse_range_expression(
            stop_at_generic_close, reject_logical_at_top_level
        )

        if self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "ASSIGN_MOD",
            "ASSIGN_XOR",
            "ASSIGN_OR",
            "ASSIGN_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression(
                stop_at_generic_close, reject_logical_at_top_level
            )
            if op == "=":
                self.constrain_cooperative_matrix_result(
                    right,
                    self.cooperative_matrix_expression_type(left),
                    "assignment target",
                )
            return AssignmentNode(left, right, op)

        return left

    def parse_ternary_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse a ternary conditional expression."""
        condition = self.parse_logical_or_expression(
            stop_at_generic_close, reject_logical_at_top_level
        )

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression(
                stop_at_generic_close, reject_logical_at_top_level
            )
            self.eat("COLON")
            false_expr = self.parse_ternary_expression(
                stop_at_generic_close, reject_logical_at_top_level
            )
            return TernaryOpNode(condition, true_expr, false_expr)

        return condition

    def parse_logical_or_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse logical OR expressions."""
        left = self.parse_logical_and_expression(
            stop_at_generic_close, reject_logical_at_top_level
        )

        while self.current_token[0] == "LOGICAL_OR":
            if reject_logical_at_top_level:
                raise CrossGLSyntaxError(
                    "Top-level '||' is not allowed in a generic argument list in expression context"
                )
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression(
                stop_at_generic_close, reject_logical_at_top_level
            )
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(
        self, stop_at_generic_close=False, reject_logical_at_top_level=False
    ):
        """Parse logical AND expressions."""
        left = self.parse_bitwise_or_expression(stop_at_generic_close)

        while self.current_token[0] == "LOGICAL_AND":
            if reject_logical_at_top_level:
                raise CrossGLSyntaxError(
                    "Top-level '&&' is not allowed in a generic argument list in expression context"
                )
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self, stop_at_generic_close=False):
        """Parse bitwise OR expressions."""
        left = self.parse_bitwise_xor_expression(stop_at_generic_close)

        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self, stop_at_generic_close=False):
        """Parse bitwise XOR expressions."""
        left = self.parse_bitwise_and_expression(stop_at_generic_close)

        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self, stop_at_generic_close=False):
        """Parse bitwise AND expressions."""
        left = self.parse_equality_expression(stop_at_generic_close)

        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self, stop_at_generic_close=False):
        """Parse equality and inequality expressions."""
        left = self.parse_relational_expression(stop_at_generic_close)

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self, stop_at_generic_close=False):
        """Parse relational comparison expressions."""
        left = self.parse_shift_expression(stop_at_generic_close)

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ] and not (stop_at_generic_close and self.current_token[0] == "GREATER_THAN"):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_shift_expression(self, stop_at_generic_close=False):
        """Parse bit-shift expressions."""
        left = self.parse_additive_expression(stop_at_generic_close)

        while self.current_token[0] in [
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ] and not (
            stop_at_generic_close and self.current_token[0] == "BITWISE_SHIFT_RIGHT"
        ):
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self, stop_at_generic_close=False):
        """Parse addition and subtraction expressions."""
        left = self.parse_multiplicative_expression(stop_at_generic_close)

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self, stop_at_generic_close=False):
        """Parse multiplication, division, and modulo expressions."""
        left = self.parse_power_expression(stop_at_generic_close)

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_power_expression(stop_at_generic_close)
            left = BinaryOpNode(left, op, right)

        return left

    def parse_power_expression(self, stop_at_generic_close=False):
        """Parse exponentiation syntax into the canonical ``pow`` intrinsic."""
        left = self.parse_unary_expression(stop_at_generic_close)

        if self.current_token[0] == "POWER":
            self.eat("POWER")
            right = self.parse_power_expression(stop_at_generic_close)
            return FunctionCallNode(IdentifierNode("pow"), [left, right])

        return left

    def parse_unary_expression(self, stop_at_generic_close=False):
        """Parse prefix unary expressions."""
        cast_expression = self.try_parse_c_style_cast(stop_at_generic_close)
        if cast_expression is not None:
            return cast_expression

        if self.current_token[0] in [
            "NOT",
            "MINUS",
            "PLUS",
            "MULTIPLY",
            "BITWISE_NOT",
            "BITWISE_AND",
            "INCREMENT",
            "DECREMENT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression(stop_at_generic_close)
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def current_token_is_keyword_spelled_operand(self):
        """Whether the current keyword token is a name used as a cast operand.

        Source languages without CrossGL's keywords use names such as ``from``,
        which the expression parser reads as identifiers: ``(uint8)from``.
        """
        token_type = self.current_token[0]
        return (
            token_type != "IDENTIFIER"
            and token_type not in self.EXPRESSION_CONTINUATION_KEYWORDS
            and self.current_token_is_binding_identifier()
        )

    def try_parse_c_style_cast(self, stop_at_generic_close=False):
        """Parse a C-style cast ``(type)operand`` into a constructor call.

        The Metal/HLSL/GLSL reverse code generators emit C-style casts such as
        ``(int64)tid``. These are lowered to the equivalent function-style
        constructor call ``int64(tid)`` (a ``FunctionCallNode`` with the type
        name as callee) so the existing scalar/vector cast handling applies.

        Pointer targets and type keywords are unambiguous casts. Identifier-led
        scalar targets remain speculative so ordinary parenthesized expressions
        like ``(a) * b`` are never misparsed.
        """
        if self.current_token[0] != "LPAREN":
            return None
        if not (
            self.next_token_starts_type()
            or self.token_is_c_style_cast_address_space(self.peek())
            or self.peek()[0] == "CONST"
        ):
            return None

        saved_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()

        def restore():
            self.undo_token_edits(token_mark)
            self.pos = saved_pos
            self.current_token = saved_token

        try:
            self.eat("LPAREN")
            address_spaces = []
            is_const = False
            while self.current_token[0] == "CONST" or (
                self.token_is_c_style_cast_address_space(self.current_token)
            ):
                if self.current_token[0] == "CONST":
                    if is_const:
                        raise CrossGLSyntaxError(
                            "Duplicate const qualifier in C-style cast target"
                        )
                    is_const = True
                else:
                    address_spaces.append(str(self.current_token[1]).lower())
                self.eat(self.current_token[0])

            if not self.is_type_token():
                restore()
                return None
            # Type names such as ``int64``/``uint64`` and user structs are lexed
            # as identifiers, which are ambiguous with variables. Permit a
            # complete generic type-id to enter the speculative type parser;
            # malformed generic syntax and comparisons restore the token stream.
            identifier_type = self.current_token[0] == "IDENTIFIER"
            if identifier_type and self.peek()[0] not in {
                "RPAREN",
                "MULTIPLY",
                "LESS_THAN",
            }:
                restore()
                return None
            target_type = self.parse_type()
            if self.current_token[0] != "RPAREN":
                restore()
                return None
            self.eat("RPAREN")
        except Exception:
            restore()
            return None

        unambiguous_target = (
            not identifier_type
            or isinstance(target_type, PointerType)
            or bool(address_spaces)
            or is_const
        )

        if address_spaces or is_const:
            if not isinstance(target_type, PointerType):
                raise CrossGLSyntaxError(
                    "Qualified C-style cast target must be a pointer type"
                )
            unique_address_spaces = set(address_spaces)
            if len(unique_address_spaces) > 1:
                raise CrossGLSyntaxError(
                    "Conflicting address spaces in C-style cast target: "
                    + ", ".join(address_spaces)
                )
            if address_spaces:
                target_type.address_space = address_spaces[0]
            if is_const:
                target_type.is_mutable = False

        # Pointer targets and keyword types accept prefix operators in the
        # operand. Identifier scalar casts still require a primary start so
        # ``(a) * b`` remains multiplication.
        allowed_operand_starts = (
            self.CAST_OPERAND_START_TOKENS
            if unambiguous_target
            else self.UNAMBIGUOUS_CAST_OPERAND_START_TOKENS
        )
        if (
            self.current_token[0] not in allowed_operand_starts
            and not self.is_type_token()
            and not self.current_token_is_keyword_spelled_operand()
        ):
            if not unambiguous_target:
                restore()
                return None

            token_type, token_value = self.current_token
            raise CrossGLSyntaxError(
                "Expected expression after C-style cast, "
                f"got {token_type} {token_value!r}"
            )

        try:
            operand = self.parse_unary_expression(stop_at_generic_close)
        except Exception:
            if unambiguous_target:
                raise
            restore()
            return None

        if isinstance(target_type, PointerType):
            return PointerReinterpretNode(operand, target_type)

        type_name = self.format_type_argument(target_type)
        return FunctionCallNode(IdentifierNode(type_name), [operand])

    def next_token_starts_type(self):
        """Whether the token following the current one can begin a type.

        This is a cheap gate for cast detection; ``try_parse_c_style_cast``
        performs the precise disambiguation (identifier types must be closed by
        ``)`` and followed by an unambiguous operand).
        """
        next_token = self.peek()
        saved_token = self.current_token
        self.current_token = next_token
        try:
            return self.is_type_token()
        finally:
            self.current_token = saved_token

    @staticmethod
    def token_is_c_style_cast_address_space(token):
        """Return whether ``token`` is a resource address-space qualifier."""
        return str(token[1]).lower() in RESOURCE_ADDRESS_SPACE_NAMES

    def current_token_is_tuple_field_number(self):
        """Return whether the current token is ``.N``, a float token that follows an operand as a tuple field."""
        token_type, token_value = self.current_token
        if token_type != "FLOAT_NUMBER":
            return False
        digits = str(token_value)[1:]
        return (
            str(token_value).startswith(".") and digits.isascii() and digits.isdigit()
        )
