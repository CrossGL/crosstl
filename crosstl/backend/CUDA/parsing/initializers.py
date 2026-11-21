"""new expressions, initializer lists, designated initializers, cast detection and assignment expressions in the CUDA parser."""

from ..CudaAst import (
    AssignmentNode,
    BinaryOpNode,
    CudaBuiltinNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    NewNode,
    UnaryOpNode,
)


class CudaInitializerMixin:
    """Initializer members of ``CudaParser``."""

    def parse_new_expression(self):
        self.eat("IDENTIFIER")

        placement_args = []
        if self.current_token[0] == "LPAREN":
            placement_args = self.parse_argument_list()

        target_type = self.parse_type_without_array_suffix()

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            node = NewNode(target_type, size=size, is_array=True)
            if placement_args:
                node.placement_args = placement_args
            return node

        args = []
        if self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()

        node = NewNode(target_type, args=args)
        if placement_args:
            node.placement_args = placement_args
        return node

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []

        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_initializer_element())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
            else:
                break

        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_initializer_element(self):
        if self.current_token[0] in ["LBRACKET", "DOT"]:
            return self.parse_designated_initializer()
        return self.parse_expression()

    def parse_designated_initializer(self):
        designators = []

        while self.current_token[0] in ["LBRACKET", "DOT"]:
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                designators.append(("index", index))
            else:
                self.eat("DOT")
                field = self.eat("IDENTIFIER")[1]
                designators.append(("field", field))

        self.eat("ASSIGN")
        value = self.parse_expression()
        return DesignatedInitializerNode(designators, value)

    def is_cast_expression(self):
        if self.current_token[0] != "LPAREN":
            return False

        saved_index = self.current_index
        try:
            self.eat("LPAREN")
            saw_integral_sign = False
            while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
                if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                    saw_integral_sign = True
                self.eat(self.current_token[0])

            if self.is_implicit_int_current_type(saw_integral_sign):
                # `(unsigned)x`, `(signed)x`: a sign keyword alone names int (C++ [dcl.type.simple]).
                pass
            elif self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
                self.parse_elaborated_type_name()
            else:
                if self.current_token[0] not in self.TYPE_TOKENS or (
                    self.current_token[0] == "IDENTIFIER"
                    and not self.is_identifier_type_name(self.current_token[1])
                    and not self.is_identifier_cast_target_at_current_index()
                    and not self.is_qualified_identifier_cast_target_at_current_index()
                ):
                    return False

                token_type = self.current_token[0]
                self.eat(token_type)
                self.consume_composite_scalar_type_suffix(token_type)

                while True:
                    if self.current_token[0] == "LESS_THAN":
                        self.parse_template_suffix()
                        continue
                    if self.current_token[0] == "SCOPE":
                        self.eat("SCOPE")
                        self.parse_name_component()
                        continue
                    break

            while self.current_token[0] in {
                "MULTIPLY",
                *self.TYPE_REFERENCE_TOKENS,
                *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
            }:
                self.eat(self.current_token[0])

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                while self.current_token[0] not in ["RBRACKET", "EOF"]:
                    self.eat(self.current_token[0])
                self.eat("RBRACKET")

            return self.current_token[0] == "RPAREN"
        finally:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]

    def consume_composite_scalar_type_suffix(self, token_type):
        if token_type == "LONG":
            saw_long_long = False
            if self.current_token[0] == "LONG":
                self.eat("LONG")
                saw_long_long = True

            if saw_long_long and self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                self.eat(self.current_token[0])

            if self.current_token[0] == "INT":
                self.eat("INT")
            elif not saw_long_long and self.current_token[0] == "DOUBLE":
                self.eat("DOUBLE")
        elif token_type == "SHORT" and self.current_token[0] == "INT":
            self.eat("INT")

    def is_identifier_cast_target_at_current_index(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        identifier_name = self.current_token[1]
        close_index = self.current_index + 1
        saw_declarator_marker = False
        while close_index < len(self.tokens) and self.tokens[close_index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            if self.tokens[close_index][0] in {
                "MULTIPLY",
                *self.TYPE_REFERENCE_TOKENS,
            }:
                saw_declarator_marker = True
            close_index += 1

        if close_index >= len(self.tokens) or self.tokens[close_index][0] != "RPAREN":
            return False

        operand_index = close_index + 1
        if (
            operand_index < len(self.tokens)
            and self.tokens[operand_index][0] == "LPAREN"
            and not self.is_identifier_type_name(identifier_name)
            and not self.is_probable_identifier_type_name(identifier_name)
        ):
            return False

        # ``(b) * r``, ``(b) & r``, ``(b) + r`` and ``(b) - r`` are binary operations unless ``b`` names a type.
        if (
            not self.is_identifier_type_name(identifier_name)
            and not saw_declarator_marker
            and operand_index < len(self.tokens)
            and self.tokens[operand_index][0]
            in {"MULTIPLY", "BITWISE_AND", "PLUS", "MINUS"}
        ):
            return False

        return self.is_cast_operand_start_at_index(operand_index)

    def is_qualified_identifier_cast_target_at_current_index(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        index = self.current_index + 1
        saw_scope = False
        saw_template = False
        final_token = self.current_token
        while True:
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False
                saw_template = True
                continue
            if (
                index + 1 < len(self.tokens)
                and self.tokens[index][0] == "SCOPE"
                and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
            ):
                saw_scope = True
                final_token = self.tokens[index + 1]
                index += 2
                continue
            break

        if not saw_scope and not saw_template:
            return False

        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1

        if index >= len(self.tokens) or self.tokens[index][0] != "RPAREN":
            return False

        operand_index = index + 1
        if (
            operand_index < len(self.tokens)
            and self.tokens[operand_index][0] == "LPAREN"
            and not self.is_probable_cast_target_name(final_token)
        ):
            return False

        return self.is_cast_operand_start_at_index(operand_index)

    def is_probable_cast_target_name(self, token):
        token_type, token_value = token
        if token_type != "IDENTIFIER":
            return token_type in self.TYPE_TOKENS
        return self.is_identifier_type_name(
            token_value
        ) or self.is_probable_identifier_type_name(token_value)

    def is_cast_operand_start_at_index(self, index):
        if index >= len(self.tokens):
            return False

        return self.tokens[index][0] in {
            "IDENTIFIER",
            "NUMBER",
            "STRING",
            "CHAR_LIT",
            "TRUE",
            "FALSE",
            "NULL",
            "NULLPTR",
            "LPAREN",
            "LBRACKET",
            "PLUS",
            "MINUS",
            "LOGICAL_NOT",
            "BITWISE_NOT",
            "MULTIPLY",
            "BITWISE_AND",
            "INCREMENT",
            "DECREMENT",
            *self.ATOMIC_FUNCTION_TOKENS,
            "THREADIDX",
            "BLOCKIDX",
            "GRIDDIM",
            "BLOCKDIM",
            "WARPSIZE",
        }

    def expression_to_text(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, CudaBuiltinNode):
            if expr.component:
                return f"{expr.builtin_name}.{expr.component}"
            return expr.builtin_name
        if isinstance(expr, BinaryOpNode):
            left = self.expression_to_text(expr.left)
            right = self.expression_to_text(expr.right)
            return f"({left} {expr.op} {right})"
        if isinstance(expr, UnaryOpNode):
            if expr.op.startswith("post"):
                return f"{self.expression_to_text(expr.operand)}{expr.op[4:]}"
            return f"{expr.op}{self.expression_to_text(expr.operand)}"
        if isinstance(expr, FunctionCallNode):
            name = self.expression_to_text(expr.name)
            args = ", ".join(self.expression_to_text(arg) for arg in expr.args)
            return f"{name}({args})"
        return str(expr)

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()

        if self.current_token[0] in [
            "ASSIGN",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "MODULO_EQUALS",
            "AND_EQUALS",
            "OR_EQUALS",
            "XOR_EQUALS",
            "SHIFT_LEFT_EQUALS",
            "SHIFT_RIGHT_EQUALS",
        ]:
            op = self.normalized_operator_value()
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def normalized_operator_value(self):
        alternative_operators = {
            "and": "&&",
            "and_eq": "&=",
            "bitand": "&",
            "bitor": "|",
            "compl": "~",
            "not": "!",
            "not_eq": "!=",
            "or": "||",
            "or_eq": "|=",
            "xor": "^",
            "xor_eq": "^=",
        }
        value = self.current_token[1]
        # An ISO C++ alternative token (``and``, ``not_eq``) is spelled as its operator; any other operator token
        # is already the operator.
        if value in alternative_operators:
            return alternative_operators[value]
        return value
