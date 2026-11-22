"""Declarators in the Metal parser: numeric identifier errors, pointer and parenthesized declarators, declarator metadata, member names and array sizes."""

from crosstl.translator.refusals import MetalSyntaxError

from .parser_tables import (
    KEYWORD_IDENTIFIER_TOKENS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    STAGE_TOKENS,
    TYPE_IDENTIFIER_TOKENS,
)


class MetalDeclaratorParsingMixin:
    """Declarator members of ``MetalParser``."""

    def parse_declarator(self):
        name = ""
        type_suffix = ""
        grouped_suffix = False
        if self.is_parenthesized_declarator_start():
            name, type_suffix = self.parse_parenthesized_declarator()
            grouped_suffix = True
        else:
            while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
                type_suffix += self.declarator_pointer_token_suffix(
                    self.current_token[0]
                )
                self.eat(self.current_token[0])
        if not name and self.is_current_name_token():
            name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.template_argument_list_followed_by_call(
                follow_token_types={"SCOPE"}
            ):
                template_args = self.parse_template_argument_suffix()
                name += f"<{self.format_generic_type_tokens(template_args)}>"
            while self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                    raise MetalSyntaxError(
                        f"Expected identifier after '::', got {self.current_token[0]}"
                    )
                name += f"::{self.current_token[1]}"
                self.eat(self.current_token[0])
        if not name and self.current_token[0] == "NUMBER":
            raise self.invalid_numeric_identifier_error(self.current_token)
        return name, self.parse_declarator_array_sizes(), type_suffix, grouped_suffix

    def invalid_numeric_identifier_error(self, token):
        fragment = str(token[1])
        if self.peek(1)[0] == "IDENTIFIER":
            fragment += str(self.peek(1)[1])
        return self.syntax_error(
            "Invalid Metal identifier "
            f"'{fragment}': generated helper names must start with a letter "
            "or underscore; underscore-separated numeric segments are only "
            "valid after the first character",
            token=token,
        )

    def invalid_adjacent_numeric_suffix_error(self, token):
        fragment = self.adjacent_numeric_suffix_fragment()
        return self.syntax_error(
            "Invalid adjacent numeric suffix fragment "
            f"'{fragment}': generated numeric suffixes must be part of "
            "the identifier or a valid member name, not a standalone "
            "numeric token",
            token=token,
        )

    def adjacent_numeric_suffix_fragment(self):
        fragment = str(self.current_token[1])
        idx = self.pos + 1
        while idx < len(self.tokens) and self.tokens[idx][0] in {
            "IDENTIFIER",
            "NUMBER",
        }:
            value = str(self.tokens[idx][1])
            if not value.startswith("_"):
                break
            fragment += value
            idx += 1
        return fragment

    def eat_statement_semicolon(self):
        if self.current_token[0] == "NUMBER":
            raise self.invalid_adjacent_numeric_suffix_error(self.current_token)
        self.eat("SEMICOLON")

    def declarator_pointer_token_suffix(self, token_type):
        return "*" if token_type == "MULTIPLY" else "&"

    def is_parenthesized_declarator_start(self):
        if self.current_token[0] != "LPAREN":
            return False
        idx = self.pos + 1
        saw_pointer_or_reference = False
        while idx < len(self.tokens) and self.tokens[idx][0] in {
            "MULTIPLY",
            "BITWISE_AND",
        }:
            saw_pointer_or_reference = True
            idx += 1
        return (
            saw_pointer_or_reference
            and idx + 1 < len(self.tokens)
            and self.is_name_token_at(idx)
            and self.tokens[idx + 1][0] == "RPAREN"
        )

    def parse_parenthesized_declarator(self):
        self.eat("LPAREN")
        type_suffix = ""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            type_suffix += self.declarator_pointer_token_suffix(self.current_token[0])
            self.eat(self.current_token[0])
        if not self.is_current_name_token():
            raise MetalSyntaxError(
                f"Expected declarator name, got {self.current_token[0]}"
            )
        name = self.current_token[1]
        self.eat(self.current_token[0])
        self.eat("RPAREN")
        return name, type_suffix

    def apply_declarator_type_suffix(self, vtype, type_suffix):
        return f"{vtype}{type_suffix}" if type_suffix else vtype

    def base_type_for_remaining_declarators(self, vtype):
        while vtype.endswith(("*", "&")):
            vtype = vtype[:-1]
        return vtype

    def apply_declarator_metadata(self, var_node, type_suffix, grouped_suffix):
        if not type_suffix:
            return
        var_node.declarator_type_suffix = type_suffix
        var_node.declarator_type_suffix_grouped = grouped_suffix

    def is_name_token_at(self, idx):
        return (
            self.tokens[idx][0] == "IDENTIFIER"
            or self.tokens[idx][0] in STAGE_TOKENS
            or self.tokens[idx][0] in KEYWORD_IDENTIFIER_TOKENS
            or self.tokens[idx][0] in TYPE_IDENTIFIER_TOKENS
            or self.tokens[idx][0] == "COMPUTE"
        )

    def is_current_name_token(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[0] in STAGE_TOKENS
            or self.current_token[0] in KEYWORD_IDENTIFIER_TOKENS
            or self.current_token[0] in TYPE_IDENTIFIER_TOKENS
            or self.current_token[0] == "COMPUTE"
        )

    def is_current_member_name_token(self):
        return self.is_current_name_token() or self.current_token[0] in {
            "READ",
            "WRITE",
            "READ_WRITE",
        }

    def parse_member_name(self, context):
        if not self.is_current_member_name_token():
            raise MetalSyntaxError(
                f"Expected identifier after {context}, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat(self.current_token[0])
        return member

    def parse_declarator_array_sizes(self):
        array_sizes = []
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            array_sizes.append(size)

        return array_sizes
