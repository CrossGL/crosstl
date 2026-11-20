"""Function definitions in the HIP parser: qualifiers, launch bounds, qualified constructors, trailing return types, requires clauses and constructor initializer lists."""

from ..HipAst import (
    FunctionNode,
    KernelNode,
)


class HipFunctionMixin:
    """Function members of ``HipParser``."""

    def parse_function_with_qualifier(self):
        qualifiers = []
        attributes = []

        while True:
            self.skip_newlines()
            self.skip_cpp_attributes()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                attributes.extend(parsed_attributes)
                continue
            if (
                not self.match(
                    *self.FUNCTION_SPECIFIER_TOKENS,
                    "__DEVICE__",
                    "__HOST__",
                    "__GLOBAL__",
                    "__FORCEINLINE__",
                    "__NOINLINE__",
                    "__LAUNCH_BOUNDS__",
                    "CONSTEXPR",
                )
                and not self.is_identifier_function_specifier_token()
            ):
                break
            if self.match("__LAUNCH_BOUNDS__"):
                attributes.append(self.parse_launch_bounds_attribute())
            else:
                is_linkage_specifier = self.current_token.type == "EXTERN"
                qualifiers.append(self.current_token.value)
                self.advance()
                if is_linkage_specifier and self.match("STRING"):
                    self.advance()

        return_type = self.parse_type()
        self.skip_newlines()

        while self.match("__LAUNCH_BOUNDS__"):
            attributes.append(self.parse_launch_bounds_attribute())
            self.skip_newlines()

        while self.is_interleaved_function_specifier_token():
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        function = FunctionNode(return_type, name, params, body, qualifiers, attributes)

        if any(item in self.KERNEL_FUNCTION_SPECIFIER_VALUES for item in qualifiers):
            return KernelNode(return_type, name, params, body, attributes)

        return function

    def parse_launch_bounds_attribute(self):
        self.consume("__LAUNCH_BOUNDS__")
        values = []

        if self.match("LPAREN"):
            self.advance()
            depth = 1
            while self.current_token and depth > 0:
                if self.match("LPAREN"):
                    depth += 1
                    values.append(self.current_token.value)
                    self.advance()
                elif self.match("RPAREN"):
                    depth -= 1
                    if depth == 0:
                        self.advance()
                        break
                    values.append(self.current_token.value)
                    self.advance()
                else:
                    values.append(self.current_token.value)
                    self.advance()

        return f"__launch_bounds__({self.format_attribute_tokens(values)})"

    def format_attribute_tokens(self, values):
        text = "".join(values).strip()
        return text.replace(",", ", ")

    def parse_simple_function(self):
        qualifiers = []
        attributes = []
        self.skip_cpp_attributes()
        while (
            self.match(*self.FUNCTION_DECLARATION_SPECIFIER_TOKENS)
            or self.is_identifier_function_specifier_token()
        ):
            is_linkage_specifier = self.current_token.type == "EXTERN"
            qualifiers.append(self.current_token.value)
            self.advance()
            if is_linkage_specifier and self.match("STRING"):
                self.advance()
            self.skip_cpp_attributes()

        return_type = self.parse_type()
        self.skip_newlines()
        while self.is_interleaved_function_specifier_token():
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode(return_type, name, params, body, qualifiers, attributes)

    def is_qualified_constructor_definition(self):
        index = self.skip_newlines_at_pos(self.pos)
        if (
            index + 4 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index + 1].type == "SCOPE"
            and self.tokens[index + 2].type == "TILDE"
            and self.tokens[index + 3].type == "IDENTIFIER"
            and self.tokens[index].value == self.tokens[index + 3].value
            and self.tokens[index + 4].type == "LPAREN"
        ):
            return True

        return (
            index + 3 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index + 1].type == "SCOPE"
            and self.tokens[index + 2].type == "IDENTIFIER"
            and self.tokens[index].value == self.tokens[index + 2].value
            and self.tokens[index + 3].type == "LPAREN"
        )

    def parse_qualified_constructor_definition(self):
        class_name = self.consume("IDENTIFIER").value
        self.consume("SCOPE")
        destructor = False
        if self.match("TILDE"):
            destructor = True
            self.advance()
        constructor_name = self.consume("IDENTIFIER").value
        name = (
            f"{class_name}::~{constructor_name}"
            if destructor
            else f"{class_name}::{constructor_name}"
        )

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        self.skip_constructor_initializer_list()

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode("", name, params, body, [])

    def parse_trailing_return_type(self, return_type):
        self.skip_newlines()
        if not self.match("ARROW"):
            return return_type
        self.advance()
        self.skip_newlines()
        return self.parse_type()

    def skip_post_function_qualifiers(self, attributes=None):
        attributes = attributes if attributes is not None else []
        while True:
            self.skip_newlines()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                attributes.extend(parsed_attributes)
                continue
            if self.match("CONST", "VOLATILE"):
                self.advance()
                continue
            if self.match("IDENTIFIER") and self.current_token.value in {
                "noexcept",
                "override",
                "final",
            }:
                self.advance()
                if self.match("LPAREN"):
                    self.skip_balanced_parentheses()
                continue
            if self.match("IDENTIFIER") and self.current_token.value == "requires":
                self.skip_requires_clause()
                continue
            break

    def skip_requires_clause(self):
        if not (self.match("IDENTIFIER") and self.current_token.value == "requires"):
            return
        self.advance()
        self.skip_newlines()
        if self.match("IDENTIFIER") and self.current_token.value == "requires":
            self.advance()
            self.skip_newlines()
            if self.match("LPAREN"):
                self.skip_balanced_parentheses()
                self.skip_newlines()
            if self.match("LBRACE"):
                self.skip_balanced_brace_block()
                self.skip_newlines()
            return
        if self.match("LPAREN"):
            self.skip_balanced_parentheses()

    def skip_balanced_parentheses(self):
        if not self.match("LPAREN"):
            return

        depth = 0
        while self.current_token:
            if self.match("LPAREN"):
                depth += 1
            elif self.match("RPAREN"):
                depth -= 1
                if depth == 0:
                    self.advance()
                    return
            self.advance()

    def skip_constructor_initializer_list(self):
        self.skip_newlines()
        if not self.match("COLON"):
            return

        self.advance()
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0
        previous_significant_token = None

        while self.current_token:
            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("LBRACE")
            ):
                if (
                    previous_significant_token is not None
                    and previous_significant_token.type in {
                        "IDENTIFIER",
                        "THREADIDX",
                        "BLOCKIDX",
                        "BLOCKDIM",
                        "GRIDDIM",
                    }
                ):
                    self.skip_balanced_brace_block()
                    previous_significant_token = self.tokens[self.pos - 1]
                    self.skip_newlines()
                    continue
                return

            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("SEMICOLON")
            ):
                return

            if self.match("LPAREN"):
                paren_depth += 1
            elif self.match("RPAREN"):
                paren_depth = max(0, paren_depth - 1)
            elif self.match("LBRACE"):
                brace_depth += 1
            elif self.match("RBRACE"):
                if brace_depth == 0:
                    return
                brace_depth -= 1
            elif self.match("LBRACKET"):
                bracket_depth += 1
            elif self.match("RBRACKET"):
                bracket_depth = max(0, bracket_depth - 1)

            if not self.match("NEWLINE"):
                previous_significant_token = self.current_token
            self.advance()
            self.skip_newlines()
