"""Declaration lookahead in the Metal parser: operator and conversion-operator functions, decltype instantiations, function-definition detection, template argument list and type suffix skipping, and qualifier tokens."""

from .parser_tables import (
    IDENTIFIER_PREFIX_TYPE_QUALIFIERS,
    IDENTIFIER_TYPE_QUALIFIERS,
    OPERATOR_OVERLOAD_TOKENS,
    QUALIFIER_TOKENS,
    RAYTRACING_TYPE_QUALIFIERS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    SIGNED_TYPE_PREFIXES,
    STAGE_TOKENS,
    TYPE_QUALIFIER_FUNCTIONS,
    TYPE_TOKENS,
)


class MetalDeclarationLookaheadParsingMixin:
    """Declaration lookahead members of ``MetalParser``."""

    def is_operator_function_definition(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return False

        idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        operator_end = self.skip_operator_name_at(idx)
        return (
            operator_end > idx
            and operator_end < len(self.tokens)
            and self.tokens[operator_end][0] == "LPAREN"
        )

    def skip_operator_name_at(self, idx):
        if idx + 1 >= len(self.tokens):
            return idx
        token_type, token_value = self.tokens[idx]
        if token_type != "IDENTIFIER" or token_value != "operator":
            return idx

        operator_token = self.tokens[idx + 1][0]
        if operator_token not in OPERATOR_OVERLOAD_TOKENS:
            return idx
        if operator_token == "LBRACKET":
            return (
                idx + 3
                if (
                    idx + 2 < len(self.tokens) and self.tokens[idx + 2][0] == "RBRACKET"
                )
                else idx
            )
        if operator_token == "LPAREN":
            return (
                idx + 3
                if (idx + 2 < len(self.tokens) and self.tokens[idx + 2][0] == "RPAREN")
                else idx
            )
        return idx + 2

    def skip_operator_function_definition(self):
        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_block()

    def is_conversion_operator_fragment(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)

        while idx < len(self.tokens) and self.tokens[idx] == (
            "IDENTIFIER",
            "explicit",
        ):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)

        if idx >= len(self.tokens) or self.tokens[idx] != ("IDENTIFIER", "operator"):
            return False

        idx = self.skip_conversion_operator_type_at(idx + 1)
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return False

        idx = self.skip_balanced_tokens_at(idx, "LPAREN", "RPAREN")
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type in {"CONST", "VOLATILE", "BITWISE_AND", "AND"}:
                idx += 1
                continue
            if self.is_qualifier_token_at(idx) or self.is_gnu_attribute_start_at(idx):
                idx += 1
                continue
            break

        return idx < len(self.tokens) and self.tokens[idx][0] in {
            "LBRACE",
            "SEMICOLON",
        }

    def skip_conversion_operator_type_at(self, idx):
        idx = self.skip_leading_attribute_tokens_at(idx)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)

        if idx >= len(self.tokens):
            return idx
        if self.tokens[idx][0] in {"STRUCT", "ENUM"}:
            idx += 1
            if idx < len(self.tokens) and self.is_name_token_at(idx):
                idx += 1
        elif self.tokens[idx][0] in SCOPED_IDENTIFIER_PART_TOKENS:
            idx += 1
        else:
            return idx

        idx = self.skip_type_reference_suffix_at(idx)
        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break
        return idx

    def is_decltype_template_instantiation_declaration(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        if not self.is_decltype_start_at(idx):
            return False
        idx = self.skip_decltype_type_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return False
        idx += 1

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LESS_THAN":
            return False
        idx = self.skip_template_argument_list_at(idx)
        return idx < len(self.tokens) and self.tokens[idx][0] == "SEMICOLON"

    def skip_decltype_template_instantiation_declaration(self):
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_decltype_start_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "decltype")
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def skip_decltype_type_at(self, idx):
        if not self.is_decltype_start_at(idx):
            return idx
        idx += 2
        depth = 1
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            idx += 1
        return idx

    def is_function_definition(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens):
            return False

        tok_type = self.tokens[idx][0]
        if tok_type in STAGE_TOKENS:
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
            while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
                idx += 1
                idx = self.skip_leading_attribute_tokens_at(idx)
            if idx >= len(self.tokens):
                return False
            tok_type = self.tokens[idx][0]

        if tok_type == "IDENTIFIER" and self.tokens[idx][1] in SIGNED_TYPE_PREFIXES:
            idx = self.skip_signed_type_prefix_at(idx)
            type_name = "int"
        elif self.is_decltype_start_at(idx):
            # ``decltype(e) f(...)``: parse_type_specifier reads the decltype return type.
            type_name = "decltype"
            idx = self.skip_decltype_type_at(idx)
        elif (
            tok_type in {"STRUCT", "ENUM"}
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] == "IDENTIFIER"
        ):
            type_name = f"{self.tokens[idx][1]} {self.tokens[idx + 1][1]}"
            idx += 2
        elif (
            tok_type == "IDENTIFIER"
            and self.tokens[idx][1] == "typename"
            and idx + 1 < len(self.tokens)
            and (
                self.tokens[idx + 1][0] in {"METAL", "SCOPE"}
                or (
                    self.tokens[idx + 1][0] == "IDENTIFIER"
                    and idx + 2 < len(self.tokens)
                    and self.tokens[idx + 2][0] == "SCOPE"
                )
            )
        ):
            idx += 1
            type_name = self.tokens[idx][1]
            idx = self.skip_scoped_type_name_at(idx)
        elif tok_type not in TYPE_TOKENS:
            return False
        else:
            type_name = self.tokens[idx][1]
            idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)

        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_scoped_type_suffix_at(idx)

        while idx < len(self.tokens) and self.tokens[idx][0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            idx += 1

        idx = self.skip_leading_attribute_tokens_at(idx)

        if (
            idx < len(self.tokens)
            and self.tokens[idx][0] in STAGE_TOKENS
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] != "LPAREN"
        ):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)

        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return False
        idx += 1
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens) or not self.is_name_token_at(idx):
                return False
            idx += 1

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            if (
                idx < len(self.tokens)
                and self.tokens[idx][0] == "LESS_THAN"
                and self.template_argument_list_at_followed_by(idx, {"LPAREN"})
            ):
                idx = self.skip_template_argument_list_at(idx)
            else:
                return False

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return False
        if type_name == "sampler" and self.parenthesized_list_ends_with_semicolon(idx):
            return False
        return True

    def template_argument_list_at_followed_by(self, idx, follow_token_types):
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LESS_THAN":
            return False
        end = self.skip_template_argument_list_at(idx)
        return (
            end > idx
            and end < len(self.tokens)
            and self.tokens[end][0] in follow_token_types
        )

    def skip_template_argument_list_at(self, idx):
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "SHIFT_RIGHT" and depth >= 2:
                depth -= 2
                if depth == 0:
                    return idx + 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif depth > 0 and token_type in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return idx
            idx += 1
        return idx

    def skip_scoped_type_suffix_at(self, idx):
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
        return idx

    def skip_type_reference_suffix_at(self, idx):
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if (
                idx < len(self.tokens)
                and self.tokens[idx] == ("IDENTIFIER", "template")
                and idx + 1 < len(self.tokens)
                and self.tokens[idx + 1][0] in SCOPED_IDENTIFIER_PART_TOKENS
            ):
                idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
        return idx

    def parenthesized_list_ends_with_semicolon(self, idx):
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] == "SEMICOLON"
                    )
            idx += 1
        return False

    def is_contextual_access_identifier_at(self, idx):
        if idx >= len(self.tokens) or self.tokens[idx][0] not in {
            "READ",
            "WRITE",
            "READ_WRITE",
        }:
            return False
        next_type = self.tokens[idx + 1][0] if idx + 1 < len(self.tokens) else "EOF"
        return next_type in {
            "SEMICOLON",
            "COMMA",
            "EQUALS",
            "LPAREN",
            "LBRACE",
            "LBRACKET",
            "LESS_THAN",
            "SCOPE",
            "RPAREN",
            "GREATER_THAN",
        }

    def is_qualifier_token_at(self, idx):
        if idx >= len(self.tokens):
            return False
        if self.is_contextual_access_identifier_at(idx):
            return False
        tok_type, value = self.tokens[idx]
        if tok_type in QUALIFIER_TOKENS or (
            tok_type == "IDENTIFIER"
            and value in IDENTIFIER_TYPE_QUALIFIERS | RAYTRACING_TYPE_QUALIFIERS
        ):
            return True
        if tok_type == "IDENTIFIER" and value in IDENTIFIER_PREFIX_TYPE_QUALIFIERS:
            return self.prefix_type_qualifier_followed_by_type_at(idx)
        return tok_type == "IDENTIFIER" and value in TYPE_QUALIFIER_FUNCTIONS
