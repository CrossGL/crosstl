"""Function and declarator names in the HIP parser: user function names, function specifiers, operator function names and variable declarator names."""


class HipDeclaratorNameMixin:
    """Declarator name members of ``HipParser``."""

    def collect_user_function_names(self):
        names = set()
        depth = 0
        index = 0

        while index < len(self.tokens):
            linkage_end = self.skip_linkage_specifier_at_pos(index)
            if depth == 0 and linkage_end is not None:
                if (
                    linkage_end < len(self.tokens)
                    and self.tokens[linkage_end].type == "LBRACE"
                ):
                    index = linkage_end + 1
                    continue
                index = linkage_end
                continue

            token_type = self.tokens[index].type
            if token_type == "LBRACE":
                depth += 1
                index += 1
                continue
            if token_type == "RBRACE":
                depth = max(0, depth - 1)
                index += 1
                continue

            if depth == 0:
                name_index = self.function_name_index_at(index)
                if name_index is not None:
                    names.add(self.tokens[name_index].value)
                    index = name_index + 1
                    continue

            index += 1

        return names

    def function_name_index_at(self, index):
        linkage_end = self.skip_linkage_specifier_at_pos(index)
        if linkage_end is not None:
            index = linkage_end

        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        function_qualifiers = {
            *self.FUNCTION_DECLARATION_SPECIFIER_TOKENS,
            "__LAUNCH_BOUNDS__",
        }
        while index < len(self.tokens) and (
            self.tokens[index].type in function_qualifiers
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            if self.tokens[index].type == "__LAUNCH_BOUNDS__":
                index = self.skip_launch_bounds_at_pos(index)
            else:
                is_linkage_specifier = self.tokens[index].type == "EXTERN"
                index += 1
                if is_linkage_specifier:
                    index = self.skip_linkage_language_string_at_pos(index)
            index = self.skip_newlines_at_pos(index)
            index = self.skip_type_attribute_prefixes_at_pos(index)

        index = self.skip_newlines_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        index = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if index is None:
            return None

        index = self.skip_newlines_at_pos(index)
        while (
            index < len(self.tokens) and self.tokens[index].type == "__LAUNCH_BOUNDS__"
        ):
            index = self.skip_launch_bounds_at_pos(index)
            index = self.skip_newlines_at_pos(index)

        while index < len(self.tokens) and self.is_interleaved_function_specifier_token(
            self.tokens[index]
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)

        parameter_list_start = self.skip_function_name_at(index)
        if (
            parameter_list_start is not None
            and self.is_plausible_function_parameter_list_at_pos(parameter_list_start)
        ):
            return index

        return None

    def is_leading_type_attribute_qualified_function(self):
        attribute_end = self.skip_type_attribute_prefixes_at_pos(self.pos)
        if attribute_end == self.pos:
            return False

        attribute_end = self.skip_newlines_at_pos(attribute_end)
        return (
            self.is_function_qualifier_token_at_pos(attribute_end)
            and self.function_name_index_at(self.pos) is not None
        )

    def is_function_qualifier_token_at_pos(self, index):
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        return token.type in {
            *self.FUNCTION_SPECIFIER_TOKENS,
            "__DEVICE__",
            "__HOST__",
            "__GLOBAL__",
            "__FORCEINLINE__",
            "__NOINLINE__",
            "__LAUNCH_BOUNDS__",
            "CONSTEXPR",
        } or self.is_identifier_function_specifier_token(token)

    def is_interleaved_function_specifier_token(self, token=None):
        token = token or self.current_token
        return token is not None and (
            token.type in {
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
            }
            or self.is_identifier_function_specifier_token(token)
        )

    def is_identifier_function_specifier_token(self, token=None):
        token = token or self.current_token
        return bool(
            token
            and token.type == "IDENTIFIER"
            and token.value in self.IDENTIFIER_FUNCTION_SPECIFIER_VALUES
        )

    def is_class_member_function_specifier_token(self, token=None):
        token = token or self.current_token
        return bool(
            token
            and (
                token.type in self.CLASS_MEMBER_FUNCTION_SPECIFIER_TOKENS
                or self.is_identifier_function_specifier_token(token)
                or (
                    token.type == "IDENTIFIER"
                    and token.value in self.CLASS_MEMBER_FUNCTION_SPECIFIER_VALUES
                )
            )
        )

    def consume_class_member_function_specifiers(self):
        qualifiers = []

        while self.current_token:
            self.skip_newlines()
            self.skip_cpp_attributes()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                qualifiers.extend(parsed_attributes)
                continue
            if not self.is_class_member_function_specifier_token():
                break
            qualifiers.append(self.current_token.value)
            self.advance()

        return qualifiers

    def parse_class_member_function_specifier_prefix(self, record_name=None):
        saved_pos = self.pos
        saved_token = self.current_token
        qualifiers = self.consume_class_member_function_specifiers()

        if not qualifiers:
            return []

        if (
            self.is_conversion_operator_declaration()
            or (
                record_name
                and self.is_class_constructor_declaration_at_pos(self.pos, record_name)
            )
            or self.is_function_declaration()
        ):
            return qualifiers

        self.pos = saved_pos
        self.current_token = saved_token
        return []

    def is_function_name_token(self, token=None):
        token = token or self.current_token
        return bool(token and token.type in self.FUNCTION_NAME_TOKENS)

    def consume_function_name(self):
        if not self.is_function_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected function name, got {token_type}")

        name = self.current_token.value
        self.advance()
        name += self.consume_operator_function_suffix(name)
        if self.match("LT"):
            name += self.parse_template_suffix()
        while self.match("SCOPE"):
            self.advance()
            if self.match("TILDE"):
                self.advance()
                member = self.consume("IDENTIFIER").value
                name += f"::~{member}"
                continue
            if not self.is_function_name_token():
                token_type = self.current_token.type if self.current_token else "EOF"
                self.error(f"Expected function name after scope, got {token_type}")
            name += f"::{self.current_token.value}"
            member = self.current_token.value
            self.advance()
            name += self.consume_operator_function_suffix(member)
            if self.match("LT"):
                name += self.parse_template_suffix()
        return name

    def consume_operator_function_suffix(self, name):
        if name != "operator":
            return ""

        if self.match("LBRACKET"):
            self.advance()
            self.consume("RBRACKET")
            return "[]"
        if self.match("LPAREN"):
            self.advance()
            self.consume("RPAREN")
            return "()"
        if self.match(*self.OVERLOADABLE_OPERATOR_TOKENS):
            operator_value = self.normalized_operator_value()
            self.advance()
            return operator_value
        return ""

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
        value = self.current_token.value
        # An ISO C++ alternative token (``and``, ``not_eq``) is spelled as its operator; any other operator token
        # is already the operator.
        if value in alternative_operators:
            return alternative_operators[value]
        return value

    def is_declarator_name_token(self):
        return self.match(*self.DECLARATOR_NAME_TOKENS)

    def is_declarator_name_token_at(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index].type in self.DECLARATOR_NAME_TOKENS
        )

    def consume_declarator_name(self):
        if not self.is_declarator_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected declarator name, got {token_type}")

        name = self.current_token.value
        self.advance()
        return name

    def consume_variable_declarator_name(self):
        if not self.is_declarator_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected declarator name, got {token_type}")

        name = self.current_token.value
        self.advance()
        if self.match("LT"):
            name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.advance()
            if not self.is_declarator_name_token():
                token_type = self.current_token.type if self.current_token else "EOF"
                self.error(f"Expected declarator name after scope, got {token_type}")
            name += f"::{self.current_token.value}"
            self.advance()
            if self.match("LT"):
                name += self.parse_template_suffix()

        return name
