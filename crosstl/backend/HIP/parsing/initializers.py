"""new expressions, initializer lists, string, character and user-defined literals, designated initializers, casts and argument lists in the HIP parser."""

from ..HipAst import (
    BinaryOpNode,
    DesignatedInitializerNode,
    HipBuiltinNode,
    InitializerListNode,
    NewNode,
    UnaryOpNode,
)


class HipInitializerMixin:
    """Initializer members of ``HipParser``."""

    def parse_new_expression(self):
        self.advance()
        placement_args = None
        if self.match("LPAREN"):
            placement_args = self.parse_parenthesized_argument_list()
            self.skip_newlines()

        target_type = self.parse_type_without_array_suffix()

        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            size = None
            if not self.match("RBRACKET"):
                size = self.parse_expression()
            self.consume("RBRACKET")
            node = NewNode(target_type, size=size, is_array=True)
            node.placement_args = placement_args
            return node

        args = []
        if self.match("LPAREN"):
            self.consume("LPAREN")
            args = self.parse_argument_list()
            self.consume("RPAREN")

        node = NewNode(target_type, args=args)
        node.placement_args = placement_args
        return node

    def parse_initializer_list(self):
        self.consume("LBRACE")
        elements = []

        self.skip_newlines()
        while self.current_token and not self.match("RBRACE"):
            elements.append(self.parse_initializer_element())
            self.skip_newlines()
            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
                if self.match("RBRACE"):
                    break
            else:
                break

        self.consume("RBRACE")
        return InitializerListNode(elements)

    def parse_initializer_element(self):
        if self.match("LBRACKET", "DOT"):
            return self.parse_designated_initializer()
        return self.parse_expression()

    def parse_string_literal_sequence(self):
        token = self.consume("STRING")
        value = token.value + self.parse_user_defined_literal_suffix(token)
        last_value = token.value
        self.skip_newlines()

        while self.match("STRING"):
            next_token = self.consume("STRING")
            next_value = next_token.value + self.parse_user_defined_literal_suffix(
                next_token
            )
            if (
                value.endswith('"')
                and next_value.startswith('"')
                and not self.is_cpp_raw_string_literal(last_value)
            ):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
            last_value = next_token.value
            self.skip_newlines()

        return value

    def is_cpp_raw_string_literal(self, value):
        if not isinstance(value, str):
            return False

        text = value.lstrip()
        for prefix in ("u8", "u", "U", "L"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break

        return text.startswith('R"')

    def parse_character_literal(self):
        token = self.consume("CHAR_LIT")
        return token.value + self.parse_user_defined_literal_suffix(token)

    def parse_user_defined_literal_suffix(self, literal_token):
        if not self.match("IDENTIFIER"):
            return ""
        if not self.is_adjacent_to_literal_token(literal_token, self.current_token):
            return ""

        suffix = self.current_token.value
        self.advance()
        return suffix

    def is_adjacent_to_literal_token(self, literal_token, next_token):
        newline_count = literal_token.value.count("\n")
        if newline_count:
            end_line = literal_token.line + newline_count
            end_column = len(literal_token.value.rsplit("\n", 1)[1]) + 1
        else:
            end_line = literal_token.line
            end_column = literal_token.column + len(literal_token.value)

        return next_token.line == end_line and next_token.column == end_column

    def parse_designated_initializer(self):
        designators = []

        while self.match("LBRACKET", "DOT"):
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                index = self.parse_expression()
                self.consume("RBRACKET")
                designators.append(("index", index))
            else:
                self.consume("DOT")
                field = self.consume("IDENTIFIER").value
                designators.append(("field", field))
            self.skip_newlines()

        self.consume("ASSIGN")
        self.skip_newlines()
        value = self.parse_expression()
        return DesignatedInitializerNode(designators, value)

    def expression_to_text(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, HipBuiltinNode):
            if expr.component:
                return f"{expr.builtin_name}.{expr.component}"
            return expr.builtin_name
        if isinstance(expr, BinaryOpNode):
            left = self.expression_to_text(expr.left)
            right = self.expression_to_text(expr.right)
            return f"({left} {expr.op} {right})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.op}{self.expression_to_text(expr.operand)}"
        return str(expr)

    def is_cast_expression(self):
        if not self.match("LPAREN"):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        type_end = self.skip_type_at_pos(
            type_start,
            allow_unknown_identifier_pointers=True,
        )
        if (
            type_end is not None
            and type_end + 1 < len(self.tokens)
            and self.tokens[type_end].type == "RPAREN"
            and self.tokens[type_end + 1].type
            in {"COLON", "COMMA", "RPAREN", "SEMICOLON"}
        ):
            return False
        if self.is_unknown_identifier_cast_followed_by_ambiguous_unary(
            type_start, type_end
        ):
            return False
        return (
            type_end is not None
            and type_end < len(self.tokens)
            and self.tokens[type_end].type == "RPAREN"
            and self.is_cast_type_sequence(type_start, type_end)
        )

    def is_unknown_identifier_cast_followed_by_ambiguous_unary(self, start, end):
        if end is None or end >= len(self.tokens):
            return False

        type_tokens = [
            token for token in self.tokens[start:end] if token.type != "NEWLINE"
        ]
        if len(type_tokens) != 1 or type_tokens[0].type != "IDENTIFIER":
            return False
        if self.is_identifier_type_name(type_tokens[0].value):
            return False

        operand_index = self.skip_newlines_at_pos(end + 1)
        return operand_index < len(self.tokens) and self.tokens[operand_index].type in {
            "STAR",
            "AMPERSAND",
        }

    def is_cast_type_sequence(self, start, end):
        index = start
        saw_integral_sign = False
        while index < end:
            if self.tokens[index].type == "NEWLINE":
                index += 1
                continue
            if self.tokens[index].type not in self.TYPE_QUALIFIER_TOKENS:
                break
            if self.tokens[index].type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        while index < end and self.tokens[index].type == "NEWLINE":
            index += 1

        if index >= end:
            return saw_integral_sign

        token = self.tokens[index]
        if saw_integral_sign and token.type in {"STAR", "AMPERSAND"}:
            # `(unsigned *)p`: the sign keyword alone names int.
            return True
        if token.type in self.ELABORATED_TYPE_TOKENS:
            return True
        if self.is_decltype_type_start_at_pos(index):
            return True
        if self.is_type_token(token, allow_identifier=False):
            return True

        if token.type != "IDENTIFIER":
            return False

        type_tokens = self.tokens[index:end]
        return (
            self.is_identifier_type_name(token.value)
            or self.is_probable_identifier_type_name(token.value)
            or any(item.type == "SCOPE" for item in type_tokens)
            or any(
                item.type in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}
                for item in type_tokens
            )
        )

    def parse_argument_list(self):
        args = []

        self.skip_newlines()
        if self.match("RPAREN"):
            return args

        while True:
            self.skip_newlines()
            if self.is_type_argument():
                arg = self.parse_type()
            else:
                arg = self.parse_expression()
            args.append(arg)
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()
            else:
                break

        return args

    def is_type_argument(self):
        if not self.is_type_argument_start_at_pos(self.pos):
            return False

        type_end = self.skip_type_at_pos(self.pos)
        if type_end is None:
            return False

        type_end = self.skip_newlines_at_pos(type_end)
        return type_end < len(self.tokens) and self.tokens[type_end].type in {
            "COMMA",
            "RPAREN",
        }

    def is_type_argument_start_at_pos(self, index):
        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)

        while index < len(self.tokens) and self.tokens[index].type in {
            *self.TYPE_PREFIX_TOKENS,
            *self.TYPE_QUALIFIER_TOKENS,
        }:
            index += 1

        index = self.skip_cpp_attributes_at_pos(index)
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        if self.is_type_token(token, allow_identifier=False):
            return True
        return token.type == "IDENTIFIER" and self.is_identifier_type_name(token.value)
