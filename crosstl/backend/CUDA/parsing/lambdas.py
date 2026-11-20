"""Lambda expressions in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import (
    FunctionCallNode,
    VariableNode,
)


class CudaLambdaMixin:
    """Lambda members of ``CudaParser``."""

    def is_lambda_expression_start(self):
        if self.current_token[0] != "LBRACKET":
            return False

        index = self.current_index
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
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

        index = self.skip_lambda_specifiers_at_index(index)
        index = self.skip_lambda_template_parameters_at_index(index)
        index = self.skip_lambda_specifiers_at_index(index)
        return index < len(self.tokens) and self.tokens[index][0] in {
            "LPAREN",
            "LBRACE",
        }

    def skip_lambda_specifiers_at_index(self, index):
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.LAMBDA_SPECIFIER_TOKENS or (
                token_type == "IDENTIFIER"
                and token_value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                index += 1
                if (
                    token_value == "noexcept"
                    and index < len(self.tokens)
                    and self.tokens[index][0] == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_index(
                        index, "LPAREN", "RPAREN"
                    )
                continue
            break
        return index

    def skip_lambda_template_parameters_at_index(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index)
            if skipped is not None:
                return skipped
        return index

    def skip_balanced_tokens_at_index(self, index, open_token, close_token):
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
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
        self.skip_lambda_specifiers()
        self.skip_lambda_template_parameters()
        self.skip_lambda_specifiers()

        args = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                args.append(self.parse_lambda_parameter())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.current_token[0] != "RPAREN":
                    raise CudaSyntaxError(
                        f"Expected COMMA or RPAREN in lambda parameters, got {self.current_token[0]}"
                    )
            self.eat("RPAREN")

        self.skip_lambda_specifiers()
        if self.current_token[0] == "ARROW":
            self.skip_lambda_trailing_return_type()
            self.skip_lambda_specifiers()

        args.append(self.parse_lambda_block_body())
        return FunctionCallNode("lambda", args)

    def skip_lambda_specifiers(self):
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in self.LAMBDA_SPECIFIER_TOKENS or (
                token_type == "IDENTIFIER"
                and token_value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                self.eat(token_type)
                if token_value == "noexcept" and self.current_token[0] == "LPAREN":
                    self.consume_balanced_lambda_tokens("LPAREN", "RPAREN")
                continue
            break

    def skip_lambda_template_parameters(self):
        if self.current_token[0] == "LESS_THAN":
            self.parse_template_suffix()

    def skip_lambda_trailing_return_type(self):
        self.eat("ARROW")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "LBRACE"
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break
            if token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.eat(token_type)

    def parse_lambda_parameter(self):
        saved_index = self.current_index
        try:
            param_type = self.parse_type()
            if self.current_token[0] != "IDENTIFIER":
                raise CudaSyntaxError("Expected lambda parameter name")
            param_name = self.eat("IDENTIFIER")[1]
            param_type += self.parse_array_suffix()
            self.skip_parameter_default()
            return VariableNode(param_type, param_name)
        except SyntaxError:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]
            raw = self.collect_lambda_parameter_raw()
            return VariableNode("", raw)

    def skip_parameter_default(self):
        if self.current_token[0] != "ASSIGN":
            return

        self.eat("ASSIGN")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break
            if token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth > 0:
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
            self.eat(token_type)

    def collect_lambda_parameter_raw(self):
        tokens = []
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break

            tokens.append(self.current_token)
            if token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.eat(token_type)

        raw = self.format_lambda_raw_tokens(tokens).strip()
        if not raw:
            raise CudaSyntaxError("Expected lambda parameter")
        return raw

    def parse_lambda_block_body(self):
        expression_body = self.try_parse_lambda_return_expression()
        if expression_body is not None:
            return expression_body

        return self.parse_raw_lambda_block_body()

    def try_parse_lambda_return_expression(self):
        saved_index = self.current_index
        completed = False
        try:
            self.eat("LBRACE")
            if self.current_token[0] != "RETURN":
                return None
            self.eat("RETURN")
            if self.current_token[0] == "SEMICOLON":
                return None
            value = self.parse_expression()
            if self.current_token[0] != "SEMICOLON":
                return None
            self.eat("SEMICOLON")
            if self.current_token[0] != "RBRACE":
                return None
            self.eat("RBRACE")
            completed = True
            return value
        except SyntaxError:
            return None
        finally:
            if not completed:
                self.current_index = saved_index
                self.current_token = self.tokens[self.current_index]

    def parse_raw_lambda_block_body(self):
        tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token = self.current_token
            token_type = token[0]
            tokens.append(token)
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                self.eat("RBRACE")
                if depth == 0:
                    return self.format_lambda_raw_tokens(tokens)
                continue
            self.eat(token_type)

        raise CudaSyntaxError("Unterminated lambda block body")

    def consume_balanced_lambda_tokens(self, open_token, close_token):
        depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                self.eat(close_token)
                if depth == 0:
                    return
                continue
            self.eat(token_type)

        raise CudaSyntaxError(f"Unterminated lambda {open_token}")

    def format_lambda_raw_tokens(self, tokens):
        text = ""
        previous = None
        for token_type, value in tokens:
            value = str(value)
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
            elif previous and previous[1] in {"(", "[", ".", "::", "->"}:
                text += value
            elif text.endswith((" ", "{")):
                text += value
            else:
                text += " " + value
            previous = (token_type, value)
        return text.strip()
