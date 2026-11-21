"""Types in the HIP parser: type names, decltype, elaborated types and parameter lists."""

from ...CUDA.parameter_defaults import UnparsedParameterDefault


class HipTypeMixin:
    """Type members of ``HipParser``."""

    def parse_type(self):
        type_parts = []
        saw_integral_sign = False

        self.skip_cpp_attributes()
        while self.match(*self.TYPE_PREFIX_TOKENS):
            self.advance()

        for qualifier in self.parse_line_split_type_qualifiers():
            if qualifier in {"signed", "unsigned"}:
                saw_integral_sign = True
            type_parts.append(qualifier)

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        self.skip_newlines()
        self.skip_cpp_attributes()
        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
            or self.match(*self.ELABORATED_TYPE_TOKENS)
            or self.match("IDENTIFIER")
        ):
            type_parts.append(self.parse_type_name())
        else:
            type_parts.append("int")  # Default type

        self.parse_postfix_type_qualifiers(type_parts)
        while self.match("ASTERISK", "STAR"):
            type_parts.append("*")
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        while self.match(*self.TYPE_REFERENCE_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        if not (
            self.is_auto_type_parts(type_parts)
            and self.is_structured_binding_declarator_at_pos(self.pos)
        ):
            array_suffix = self.parse_array_suffix()
            if array_suffix:
                type_parts.append(array_suffix)

        return " ".join(type_parts)

    def is_auto_type_parts(self, type_parts):
        return "auto" in type_parts

    def parse_type_without_array_suffix(self):
        type_parts = []
        saw_integral_sign = False

        self.skip_cpp_attributes()
        while self.match(*self.TYPE_PREFIX_TOKENS):
            self.advance()

        for qualifier in self.parse_line_split_type_qualifiers():
            if qualifier in {"signed", "unsigned"}:
                saw_integral_sign = True
            type_parts.append(qualifier)

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        self.skip_newlines()
        self.skip_cpp_attributes()
        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
            or self.match(*self.ELABORATED_TYPE_TOKENS)
            or self.match("IDENTIFIER")
        ):
            type_parts.append(self.parse_type_name())
        else:
            type_parts.append("int")

        self.parse_postfix_type_qualifiers(type_parts)
        while self.match("ASTERISK", "STAR"):
            type_parts.append("*")
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        while self.match(*self.TYPE_REFERENCE_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def is_implicit_int_current_type(self, saw_integral_sign):
        return self.is_implicit_int_type_at_pos(self.pos, saw_integral_sign)

    def is_decltype_type_start(self):
        return self.is_decltype_type_start_at_pos(self.pos)

    def is_decltype_type_start_at_pos(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index].value == "decltype"
            and self.tokens[index + 1].type == "LPAREN"
        )

    def skip_decltype_type_at_pos(self, index):
        if not self.is_decltype_type_start_at_pos(index):
            return None

        index += 1
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return None
            index += 1

        return None

    def parse_decltype_type_name(self):
        self.consume("IDENTIFIER")
        self.consume("LPAREN")
        parts = []
        depth = 1

        while depth > 0:
            if self.current_token is None:
                raise self.syntax_error("Unterminated decltype type")

            token_type = self.current_token.type
            token_value = self.current_token.value
            if token_type == "EOF":
                raise self.syntax_error("Unterminated decltype type")
            if token_type == "LPAREN":
                depth += 1
                parts.append(token_value)
                self.consume("LPAREN")
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.consume("RPAREN")
                    break
                parts.append(token_value)
                self.consume("RPAREN")
            else:
                parts.append(token_value)
                self.consume(token_type)

        return f"decltype({self.format_template_parts(parts)})"

    def parse_postfix_type_qualifiers(self, type_parts):
        self.skip_newlines()
        while self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

    def parse_type_name(self):
        if self.match(*self.ELABORATED_TYPE_TOKENS):
            return self.parse_elaborated_type_name()

        type_name = self.current_token.value
        token_type = self.current_token.type
        self.advance()

        if token_type in {"CHAR", "SHORT", "INT"}:
            integral_specifiers = [type_name]
            while self.match("SIGNED", "UNSIGNED", "SHORT", "INT"):
                integral_specifiers.append(self.current_token.value)
                self.advance()
            if "char" in integral_specifiers:
                integral_base = "char"
            elif "short" in integral_specifiers:
                integral_base = "short"
            else:
                integral_base = "int"
            if "unsigned" in integral_specifiers:
                type_name = f"unsigned {integral_base}"
            elif "signed" in integral_specifiers:
                type_name = f"signed {integral_base}"
            else:
                type_name = integral_base

        if token_type == "LONG" and self.match("LONG"):
            type_name += " long"
            self.advance()
            if self.match("SIGNED", "UNSIGNED"):
                type_name += f" {self.current_token.value}"
                self.advance()
                if self.match("INT"):
                    type_name += " int"
                    self.advance()
            elif self.match("INT"):
                self.advance()
        elif token_type == "LONG" and self.match("INT"):
            self.advance()
        elif token_type == "SHORT" and self.match("INT"):
            self.advance()

        if self.match("LT"):
            type_name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            type_name += f"::{member}"
            if self.match("LT"):
                type_name += self.parse_template_suffix()

        return type_name

    def parse_elaborated_type_name(self):
        type_name = self.current_token.value
        token_type = self.current_token.type
        self.advance()
        self.skip_newlines()

        if token_type == "ENUM" and self.match("CLASS", "STRUCT"):
            type_name += f" {self.current_token.value}"
            self.advance()
            self.skip_newlines()

        member = self.consume_qualified_name_member()
        type_name += f" {member}"
        if self.match("LT"):
            type_name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            type_name += f"::{member}"
            if self.match("LT"):
                type_name += self.parse_template_suffix()

        return type_name

    def parse_parameter_list(self):
        params = []
        previous_base_type = None

        self.skip_newlines()
        if self.match("RPAREN"):
            return params
        if self.match("VOID") and self.peek() and self.peek().type == "RPAREN":
            self.advance()
            return params

        while True:
            self.skip_newlines()
            if self.match("RPAREN"):
                break

            self.skip_cpp_attributes()
            if previous_base_type and self.is_parameter_declarator_continuation():
                param_type = self.parse_declarator_prefix(previous_base_type)
            else:
                param_type = self.parse_type()
                previous_base_type = self.strip_declarator_markers(param_type)
            self.skip_newlines()

            param_name = ""
            reference_array_name = self.parse_reference_array_parameter_declarator()
            function_pointer_name = (
                None
                if reference_array_name is not None
                else self.parse_function_pointer_parameter_declarator()
            )
            if reference_array_name is not None:
                param_type += " & "
                param_name = reference_array_name
            elif function_pointer_name is not None:
                param_type += " (*)"
                param_name = function_pointer_name
            elif self.is_declarator_name_token():
                param_name = self.current_token.value
                self.advance()
                self.skip_cpp_attributes()

            if self.match("ELLIPSIS"):
                param_type += " ..."
                self.advance()
                self.skip_newlines()

            if not param_name and self.is_declarator_name_token():
                param_name = self.current_token.value
                self.advance()
                self.skip_cpp_attributes()

            param_type += self.parse_array_suffix()
            param = {"type": param_type, "name": param_name}
            default = self.parse_default_parameter_value()
            if default is not None:
                param["default"] = default
            params.append(param)

            self.skip_newlines()
            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
            else:
                break

        return params

    def is_parameter_declarator_continuation(self):
        if self.match("ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS):
            return True
        return (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type in {"ASTERISK", "STAR"}
        )

    def parse_reference_array_parameter_declarator(self):
        """Consume the ``(&name)`` of a reference-to-array parameter ``T (&name)[N]`` and return the name."""
        name_token = self.peek(2)
        if not (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type == "AMPERSAND"
            and name_token
            and name_token.type in self.DECLARATOR_NAME_TOKENS
            and self.peek(3)
            and self.peek(3).type == "RPAREN"
            and self.peek(4)
            and self.peek(4).type == "LBRACKET"
        ):
            return None
        self.consume("LPAREN")
        self.advance()
        name = self.current_token.value
        self.advance()
        self.consume("RPAREN")
        return name

    def parse_function_pointer_parameter_declarator(self):
        if not (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type in {"ASTERISK", "STAR"}
        ):
            return None

        self.consume("LPAREN")
        self.advance()

        name = ""
        if self.is_declarator_name_token():
            name = self.current_token.value
            self.advance()

        self.consume("RPAREN")

        if self.match("LPAREN"):
            self.skip_balanced_parentheses()

        return name

    def parse_default_parameter_value(self):
        """Parse a parameter's ``= value`` default argument, or return ``None`` when it has none.

        A default the expression parser cannot read up to the next ``,`` or ``)`` is skipped and returned as an
        :class:`UnparsedParameterDefault` that keeps its text, so the importer refuses it instead of dropping it.
        """
        self.skip_newlines()
        if not self.match("ASSIGN"):
            return None
        start = self.pos
        self.advance()
        self.skip_newlines()
        try:
            value = self.parse_assignment_expression()
            self.skip_newlines()
        except SyntaxError:
            value = None
        if value is not None and self.match("COMMA", "RPAREN"):
            return value
        self.pos = start
        self.current_token = self.tokens[start]
        self.skip_default_parameter_value()
        text = " ".join(str(token.value) for token in self.tokens[start + 1 : self.pos])
        return UnparsedParameterDefault(text)

    def skip_default_parameter_value(self):
        self.skip_newlines()
        if not self.match("ASSIGN"):
            return

        self.advance()
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0

        while self.current_token:
            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("COMMA", "RPAREN")
            ):
                return

            if self.match("LPAREN"):
                paren_depth += 1
            elif self.match("RPAREN"):
                if paren_depth == 0:
                    return
                paren_depth -= 1
            elif self.match("LBRACE"):
                brace_depth += 1
            elif self.match("RBRACE"):
                if brace_depth == 0:
                    return
                brace_depth -= 1
            elif self.match("LBRACKET"):
                bracket_depth += 1
            elif self.match("RBRACKET"):
                if bracket_depth == 0:
                    return
                bracket_depth -= 1

            self.advance()
