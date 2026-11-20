"""Macro block and statement invocations, concept declarations and statement directive markers in the CUDA parser."""


class CudaMacroMixin:
    """Macro members of ``CudaParser``."""

    def is_macro_block_invocation_start(self):
        return (
            self.is_macro_block_invocation_at_index(self.current_index)
            or self.is_bare_macro_before_block_invocation()
            or self.is_object_like_macro_before_block_invocation()
        )

    def is_macro_block_invocation_at_index(self, index):
        if (
            index + 1 >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or not self.is_macro_like_identifier(self.tokens[index][1])
            or self.tokens[index + 1][0] != "LPAREN"
        ):
            return False

        block_index = self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")
        return (
            block_index < len(self.tokens) and self.tokens[block_index][0] == "LBRACE"
        )

    def is_bare_macro_before_block_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.is_macro_like_identifier(self.current_token[1])
            and self.current_index + 1 < len(self.tokens)
            and self.is_macro_block_invocation_at_index(self.current_index + 1)
        )

    def is_object_like_macro_before_block_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.is_macro_like_identifier(self.current_token[1])
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LBRACE"
        )

    def is_macro_like_identifier(self, name):
        return any(character.isupper() for character in name) and all(
            character.isupper() or character.isdigit() or character == "_"
            for character in name
        )

    def is_statement_macro_invocation_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1]
            in self.CUDA_STATEMENT_MACRO_INVOCATION_IDENTIFIERS
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        )

    def skip_statement_macro_invocation(self):
        self.eat("IDENTIFIER")
        self.skip_balanced_parentheses()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_top_level_macro_statement_start(self):
        end_index = self.top_level_macro_statement_end_index(self.current_index)
        return end_index is not None

    def is_concept_declaration_start(self):
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)

        if index + 2 >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type != "IDENTIFIER":
            return False

        if token_value == "concept":
            name_index = index + 1
        elif token_value == "_CCCL_CONCEPT" or token_value.endswith("_CONCEPT"):
            name_index = index + 1
        else:
            return False

        return (
            name_index + 1 < len(self.tokens)
            and self.tokens[name_index][0] in self.NAME_COMPONENT_TOKENS
            and self.tokens[name_index + 1][0] == "ASSIGN"
        )

    def skip_concept_declaration(self):
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                self.eat("SEMICOLON")
                break

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1

            self.eat(token_type)

    def top_level_macro_statement_end_index(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or not self.is_macro_like_identifier(self.tokens[index][1])
        ):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
            index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")

        if index < len(self.tokens) and self.tokens[index][0] == "SEMICOLON":
            return index + 1
        if index >= len(self.tokens):
            return index

        follow_type, follow_value = self.tokens[index]
        if follow_type == "IDENTIFIER" and self.is_macro_like_identifier(follow_value):
            return index
        if follow_type in self.TOP_LEVEL_MACRO_STATEMENT_FOLLOW_TOKENS:
            return index
        return None

    def skip_top_level_macro_statement(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_statement_directive_marker_start(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        marker_end = self.statement_directive_marker_end_index(self.current_index)
        return (
            marker_end is not None
            and marker_end < len(self.tokens)
            and self.tokens[marker_end][0] in self.STATEMENT_DIRECTIVE_FOLLOW_TOKENS
        )

    def statement_directive_marker_end_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "IDENTIFIER":
            return None

        name = self.tokens[index][1]
        if name == "_Pragma":
            if index + 1 >= len(self.tokens) or self.tokens[index + 1][0] != "LPAREN":
                return None
            return self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")

        if not self.is_macro_like_identifier(name):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
            index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")
        return index

    def skip_statement_directive_marker(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()

    def skip_macro_block_invocation(self):
        if self.is_bare_macro_before_block_invocation():
            self.eat("IDENTIFIER")

        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
