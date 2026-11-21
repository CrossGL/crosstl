"""Lambda expressions in the HIP parser."""

from ..HipAst import (
    FunctionCallNode,
    VariableNode,
)


class HipLambdaMixin:
    """Lambda members of ``HipParser``."""

    def is_lambda_expression_start(self):
        if not self.match("LBRACKET"):
            return False

        index = self.pos
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            elif token_type == "EOF":
                return False
            index += 1
        else:
            return False

        index = self.skip_lambda_specifiers_at_pos(index)
        index = self.skip_lambda_template_parameters_at_pos(index)
        index = self.skip_lambda_specifiers_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type in {
            "LPAREN",
            "LBRACE",
        }

    def skip_lambda_specifiers_at_pos(self, index):
        while index < len(self.tokens):
            token = self.tokens[index]
            if token.type == "NEWLINE":
                index += 1
                continue
            if token.type in self.LAMBDA_SPECIFIER_TOKENS or (
                token.type == "IDENTIFIER"
                and token.value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                index += 1
                if (
                    token.value == "noexcept"
                    and index < len(self.tokens)
                    and self.tokens[index].type == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_pos(index, "LPAREN", "RPAREN")
                continue
            break
        return index

    def skip_lambda_template_parameters_at_pos(self, index):
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            skipped = self.skip_template_at_pos(index)
            if skipped is not None:
                return skipped
        return index

    def skip_balanced_tokens_at_pos(self, index, open_token, close_token):
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return index
            index += 1
        return index

    def parse_lambda_expression(self):
        self.consume_balanced_lambda_tokens("LBRACKET", "RBRACKET")
        self.skip_newlines()
        self.skip_lambda_specifiers()
        self.skip_lambda_template_parameters()
        self.skip_lambda_specifiers()

        args = []
        if self.match("LPAREN"):
            self.consume("LPAREN")
            self.skip_newlines()
            while not self.match("RPAREN"):
                args.append(self.parse_lambda_parameter())
                self.skip_newlines()
                if self.match("COMMA"):
                    self.advance()
                    self.skip_newlines()
                elif not self.match("RPAREN"):
                    self.error(
                        f"Expected COMMA or RPAREN in lambda parameters, got {self.current_token.type if self.current_token else 'EOF'}"
                    )
            self.consume("RPAREN")

        self.skip_newlines()
        self.skip_lambda_specifiers()
        if self.match("ARROW"):
            self.skip_lambda_trailing_return_type()
            self.skip_newlines()
            self.skip_lambda_specifiers()

        args.append(self.parse_lambda_block_body())
        return FunctionCallNode("lambda", args)

    def skip_lambda_specifiers(self):
        while self.current_token:
            if self.match("NEWLINE"):
                self.advance()
                continue
            if self.current_token.type in self.LAMBDA_SPECIFIER_TOKENS or (
                self.current_token.type == "IDENTIFIER"
                and self.current_token.value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                token_value = self.current_token.value
                self.advance()
                if token_value == "noexcept" and self.match("LPAREN"):
                    self.consume_balanced_lambda_tokens("LPAREN", "RPAREN")
                continue
            break

    def skip_lambda_template_parameters(self):
        if self.match("LT"):
            self.parse_template_suffix()
            self.skip_newlines()

    def skip_lambda_trailing_return_type(self):
        self.consume("ARROW")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type == "LBRACE"
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.advance()

    def parse_lambda_parameter(self):
        saved_pos = self.pos
        try:
            param_type = self.parse_type()
            if not self.match("IDENTIFIER"):
                self.error("Expected lambda parameter name")
            param_name = self.consume("IDENTIFIER").value
            param_type += self.parse_array_suffix()
            self.skip_lambda_parameter_default()
            return VariableNode(param_type, param_name)
        except SyntaxError:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )
            raw = self.collect_lambda_parameter_raw()
            return VariableNode("", raw)

    def skip_lambda_parameter_default(self):
        if not self.match("ASSIGN"):
            return

        self.advance()
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
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
            self.advance()

    def collect_lambda_parameter_raw(self):
        tokens = []
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0

        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break

            tokens.append(self.current_token)
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.advance()

        raw = self.format_lambda_raw_tokens(tokens).strip()
        if not raw:
            self.error("Expected lambda parameter")
        return raw

    def parse_lambda_block_body(self):
        expression_body = self.try_parse_lambda_return_expression()
        if expression_body is not None:
            return expression_body

        return self.parse_raw_lambda_block_body()

    def try_parse_lambda_return_expression(self):
        saved_pos = self.pos
        completed = False
        try:
            self.consume("LBRACE")
            self.skip_newlines()
            if not self.match("RETURN"):
                return None
            self.consume("RETURN")
            self.skip_newlines()
            if self.match("SEMICOLON"):
                return None
            value = self.parse_expression()
            self.skip_newlines()
            if not self.match("SEMICOLON"):
                return None
            self.consume("SEMICOLON")
            self.skip_newlines()
            if not self.match("RBRACE"):
                return None
            self.consume("RBRACE")
            completed = True
            return value
        except SyntaxError:
            return None
        finally:
            if not completed:
                self.pos = saved_pos
                self.current_token = self.tokens[self.pos]

    def parse_raw_lambda_block_body(self):
        tokens = []
        depth = 0
        while self.current_token:
            token = self.current_token
            token_type = token.type
            tokens.append(token)
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                self.consume("RBRACE")
                if depth == 0:
                    return self.format_lambda_raw_tokens(tokens)
                continue
            self.advance()

        raise self.syntax_error("Unterminated lambda block body")

    def consume_balanced_lambda_tokens(self, open_token, close_token):
        depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                self.consume(close_token)
                if depth == 0:
                    return
                continue
            self.advance()

        raise self.syntax_error(f"Unterminated lambda {open_token}")

    def format_lambda_raw_tokens(self, tokens):
        text = ""
        previous = None
        for token in tokens:
            token_type = token.type
            value = str(token.value)
            if token_type == "NEWLINE":
                continue
            if not text:
                text = "{ " if value == "{" else value
            elif value in {")", "]", ";", ",", ":"}:
                text = text.rstrip() + value
                if value in {";", ","}:
                    text += " "
            elif value == "}":
                if not text.endswith((" ", "{")):
                    text += " "
                text += value
            elif value in {"(", "[", ".", "::", "->"}:
                text = text.rstrip() + value
            elif value == "{":
                if not text.endswith(" "):
                    text += " "
                text += value + " "
            elif previous and previous.value in {"(", "[", ".", "::", "->"}:
                text += value
            elif text.endswith((" ", "{")):
                text += value
            else:
                text += " " + value
            previous = token
        return text.strip()
