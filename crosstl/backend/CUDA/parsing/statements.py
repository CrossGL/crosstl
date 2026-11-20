"""Blocks, comma and fold expression statements, try, throw, delete, label and goto statements, and structured bindings in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import (
    BlockNode,
    DeleteNode,
    FunctionCallNode,
    VariableNode,
)


class CudaStatementMixin:
    """Statement members of ``CudaParser``."""

    def parse_scoped_statement(self):
        """A statement of a block or a ``case``; a nested ``{ ... }`` keeps its own scope as a ``BlockNode``."""
        if self.current_token[0] == "LBRACE":
            return BlockNode(self.parse_block())
        return self.parse_statement()

    def parse_block(self):
        self.eat("LBRACE")
        statements = []

        self.block_depth += 1
        try:
            while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
                stmt = self.parse_scoped_statement()
                if stmt:
                    if isinstance(stmt, list):
                        statements.extend(stmt)
                    else:
                        statements.append(stmt)
        finally:
            self.block_depth -= 1

        self.eat("RBRACE")
        return statements

    def parse_comma_expression_statement(self, first_expr):
        expressions = [first_expr]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_assignment_expression())
        return expressions

    def is_parenthesized_comma_expression_statement(self):
        if self.current_token[0] != "LPAREN":
            return False

        depth = 0
        saw_comma = False
        index = self.current_index
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    next_index = index + 1
                    return (
                        saw_comma
                        and next_index < len(self.tokens)
                        and self.tokens[next_index][0] == "SEMICOLON"
                    )
            elif token_type == "COMMA" and depth == 1:
                saw_comma = True
            elif token_type == "EOF":
                return False
            index += 1

        return False

    def is_fold_expression_statement_start(self):
        return self.current_token[0] == "LPAREN" and self.is_ellipsis_at_index(
            self.current_index + 1
        )

    def parse_fold_expression_statement(self):
        raw_tokens = self.collect_balanced_raw_tokens_until("SEMICOLON")
        self.eat("SEMICOLON")
        raw_expression = self.format_lambda_raw_tokens(raw_tokens)
        return FunctionCallNode("cuda_fold_expression", [raw_expression])

    def is_parenthesized_fold_expression_start(self):
        if self.current_token[0] != "LPAREN":
            return False

        depth = 0
        index = self.current_index
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return False
            elif depth == 1 and self.is_ellipsis_at_index(index):
                return True
            elif token_type == "EOF":
                return False
            index += 1
        return False

    def parse_parenthesized_fold_expression(self):
        raw_tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            raw_tokens.append(self.current_token)
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                self.eat("RPAREN")
                if depth == 0:
                    break
                continue
            self.eat(token_type)

        raw_expression = self.format_lambda_raw_tokens(raw_tokens)
        return FunctionCallNode("cuda_fold_expression", [raw_expression])

    def parse_parenthesized_comma_expression_statement(self):
        self.eat("LPAREN")
        first_expr = self.parse_assignment_expression()
        expressions = self.parse_comma_expression_statement(first_expr)
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return expressions

    def parse_try_statement(self):
        self.eat("IDENTIFIER")
        try_body = self.parse_statement()

        while self.is_identifier_value("catch"):
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            self.parse_statement()

        return try_body

    def parse_throw_statement(self):
        self.eat("IDENTIFIER")
        args = []
        if self.current_token[0] != "SEMICOLON":
            args.append(self.parse_expression())
        self.eat("SEMICOLON")
        return FunctionCallNode("throw", args)

    def is_identifier_value(self, value):
        return self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value

    def is_structured_binding_declaration(self):
        index = self.current_index
        index = self.skip_declaration_prefixes_at_index(index)
        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_structured_binding_type_at_index(index)

        if index is None:
            return False

        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_structured_binding_names_at_index(index)
        return (
            index is not None
            and index < len(self.tokens)
            and self.tokens[index][0] in {"ASSIGN", "LPAREN", "LBRACE"}
        )

    def skip_structured_binding_type_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] in {
            "CONST",
            "VOLATILE",
        }:
            index += 1

        if (
            index >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or self.tokens[index][1] != "auto"
        ):
            return None

        index += 1
        index = self.skip_postfix_type_qualifiers_at_index(index)

        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_REFERENCE_TOKENS
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return index

    def skip_structured_binding_names_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LBRACKET":
            return None

        index += 1
        expect_name = True
        saw_name = False

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if expect_name:
                if token_type not in self.NAME_COMPONENT_TOKENS:
                    return None
                saw_name = True
                expect_name = False
                index += 1
                continue

            if token_type == "COMMA":
                expect_name = True
                index += 1
                continue
            if token_type == "RBRACKET":
                return index + 1 if saw_name and not expect_name else None
            return None

        return None

    def parse_structured_binding_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()

        vtype = self.parse_structured_binding_type()
        self.parse_alignment_attributes()
        name = self.parse_structured_binding_name()
        value = self.parse_variable_initializer(vtype)
        return VariableNode(vtype, name, value, qualifiers)

    def parse_structured_binding_type(self):
        type_parts = []

        while self.current_token[0] in {"CONST", "VOLATILE"}:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        type_parts.append(self.eat("IDENTIFIER")[1])
        self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def parse_structured_binding_name(self):
        names = []
        self.eat("LBRACKET")
        while self.current_token[0] != "RBRACKET":
            names.append(self.parse_name_component())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RBRACKET":
                raise CudaSyntaxError(
                    f"Expected COMMA or RBRACKET, got {self.current_token[0]}"
                )
        self.eat("RBRACKET")
        return f"[{', '.join(names)}]"

    def parse_delete_statement(self):
        self.eat("IDENTIFIER")
        is_array = False

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            is_array = True

        expression = self.parse_unary_expression()
        return DeleteNode(expression, is_array)

    def is_label_statement_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "COLON"
        )

    def parse_label_statement(self):
        self.eat("IDENTIFIER")
        self.eat("COLON")

        if self.current_token[0] in {"RBRACE", "EOF"}:
            return None
        return self.parse_statement()

    def parse_goto_statement(self):
        self.eat("GOTO")

        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            self.eat(self.current_token[0])
        else:
            while self.current_token[0] not in {"SEMICOLON", "EOF"}:
                self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return None
