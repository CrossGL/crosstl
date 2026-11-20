"""Function definitions in the CUDA parser: trailing return types, post-function qualifiers, requires clauses, attributes, launch bounds and parameters."""

from ..CudaAst import (
    FunctionNode,
    KernelNode,
    VariableNode,
)


class CudaFunctionMixin:
    """Function members of ``CudaParser``."""

    def parse_function(self):
        qualifiers = []
        attributes = []
        linkage = None

        while (
            self.current_token[0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier()
        ):
            if self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                attributes.append(self.parse_function_attribute())
                continue

            qualifier = self.current_token[1]
            qualifiers.append(qualifier)
            self.eat(self.current_token[0])
            if qualifier == "extern" and self.current_token[0] == "STRING":
                linkage = self.current_token[1].strip('"')
                self.eat("STRING")

        return_type = self.parse_type()
        while (
            self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.current_token[0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier()
        ):
            if self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                attributes.append(self.parse_function_attribute())
                continue

            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        name = self.consume_function_name()
        if self.current_token[0] == "LESS_THAN":
            name += self.parse_template_suffix()
        self.user_function_names.add(name)
        params = self.parse_parameters()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers()
        body = None
        if self.is_identifier_value("try"):
            body = self.parse_try_statement()
        elif self.current_token[0] == "LBRACE":
            body = self.parse_block()
        elif self.current_token[0] == "ASSIGN":
            self.skip_function_declaration_assignment_suffix()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        else:
            self.eat("LBRACE")

        if "__global__" in qualifiers or "__tile_global__" in qualifiers:
            return KernelNode(
                return_type,
                name,
                params,
                body,
                attributes,
                qualifiers=qualifiers,
                linkage=linkage,
            )

        function = FunctionNode(return_type, name, params, body, qualifiers, attributes)
        function.linkage = linkage
        return function

    def parse_trailing_return_type(self, return_type):
        if self.current_token[0] != "ARROW":
            return return_type

        self.eat("ARROW")
        return self.parse_type()

    def skip_post_function_qualifiers(self):
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in {"CONST", "VOLATILE", *self.TYPE_REFERENCE_TOKENS}:
                self.eat(token_type)
                continue
            if token_type == "IDENTIFIER" and token_value in {
                "noexcept",
                "override",
                "final",
            }:
                self.eat("IDENTIFIER")
                if token_value == "noexcept" and self.current_token[0] == "LPAREN":
                    self.skip_balanced_parentheses()
                continue
            if token_type == "IDENTIFIER" and token_value == "requires":
                self.skip_requires_clause()
                continue
            break

    def skip_function_declaration_assignment_suffix(self):
        self.eat("ASSIGN")

        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            elif self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()
            elif self.current_token[0] == "LBRACE":
                self.skip_balanced_brace_block()
            else:
                self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_requires_clause(self):
        self.eat("IDENTIFIER")
        if self.is_requires_expression_start():
            self.parse_requires_expression()
            return

        paren_depth = 0
        bracket_depth = 0
        template_depth = 0
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                paren_depth == 0
                and bracket_depth == 0
                and template_depth == 0
                and (
                    token_type in {"LBRACE", "SEMICOLON"}
                    or (token_type == "IDENTIFIER" and token_value == "try")
                )
            ):
                break

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LESS_THAN":
                template_depth += 1
            elif token_type == "GREATER_THAN" and template_depth > 0:
                template_depth -= 1
            elif token_type == "SHIFT_RIGHT" and template_depth > 0:
                template_depth = max(0, template_depth - 2)

            self.eat(token_type)

    def parse_function_attribute(self):
        attribute_name = self.current_token[1]
        self.eat(self.current_token[0])
        values = []

        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            depth = 1
            while self.current_token[0] != "EOF" and depth > 0:
                token_type, token_value = self.current_token
                if token_type == "LPAREN":
                    depth += 1
                    values.append(token_value)
                    self.eat("LPAREN")
                elif token_type == "RPAREN":
                    depth -= 1
                    if depth == 0:
                        self.eat("RPAREN")
                        break
                    values.append(token_value)
                    self.eat("RPAREN")
                else:
                    values.append(token_value)
                    self.eat(token_type)

        return f"{attribute_name}({self.format_attribute_tokens(values)})"

    def parse_launch_bounds_attribute(self):
        return self.parse_function_attribute()

    def format_attribute_tokens(self, values):
        text = "".join(values).strip()
        return text.replace(",", ", ")

    def parse_parameters(self):
        self.eat("LPAREN")
        params = []

        if (
            self.current_token[0] == "VOID"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "RPAREN"
        ):
            self.eat("VOID")
            self.eat("RPAREN")
            return params

        if self.current_token[0] != "RPAREN":
            params.append(self.parse_parameter())

            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.parse_parameter())

        self.eat("RPAREN")
        return params

    def parse_parameter(self):
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        self.skip_cuda_parameter_attributes()

        if self.is_ellipsis_at_current_index():
            self.consume_ellipsis()
            return VariableNode("...", "")

        param_type = self.parse_type()
        param_type = self.parse_declarator_prefix(param_type)
        if self.is_function_pointer_parameter_declarator():
            return self.parse_function_pointer_parameter(param_type)

        if self.is_ellipsis_at_current_index():
            self.consume_ellipsis()
            param_type = f"{param_type} ...".strip()

        if self.current_token[0] in {"COMMA", "RPAREN"}:
            self.skip_parameter_default()
            return VariableNode(param_type, "")

        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        self.skip_cuda_parameter_attributes()

        param_name = self.parse_name_component()
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        param_type += self.parse_array_suffix()
        self.skip_parameter_default()
        return VariableNode(param_type, param_name)

    def skip_cuda_parameter_attributes(self):
        while self.current_token[0] in self.CUDA_PARAMETER_ATTRIBUTE_TOKENS or (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in self.CUDA_PARAMETER_ATTRIBUTE_IDENTIFIERS
        ):
            self.eat(self.current_token[0])

    def is_ellipsis_at_current_index(self):
        return self.is_ellipsis_at_index(self.current_index)

    def is_ellipsis_at_index(self, index):
        return (
            index + 2 < len(self.tokens)
            and self.tokens[index][0] == "DOT"
            and self.tokens[index + 1][0] == "DOT"
            and self.tokens[index + 2][0] == "DOT"
        )

    def consume_ellipsis(self):
        self.eat("DOT")
        self.eat("DOT")
        self.eat("DOT")

    def is_function_pointer_parameter_declarator(self):
        return (
            self.current_index + 2 < len(self.tokens)
            and self.current_token[0] == "LPAREN"
            and self.tokens[self.current_index + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[self.current_index + 2][0] == "IDENTIFIER"
        )

    def parse_function_pointer_parameter(self, return_type):
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        name = self.eat("IDENTIFIER")[1]
        self.eat("RPAREN")

        if self.current_token[0] == "LPAREN":
            self.parse_parameters()
        array_suffix = self.parse_array_suffix()

        if array_suffix:
            param_type = f"{return_type} {pointer_prefix} {array_suffix}".strip()
        else:
            param_type = f"{return_type} ({pointer_prefix})".strip()
        self.skip_parameter_default()
        return VariableNode(param_type, name)
