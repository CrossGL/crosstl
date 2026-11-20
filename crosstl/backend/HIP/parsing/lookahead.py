"""Declaration lookahead in the HIP parser: function and variable declaration detection and the token skippers it uses."""


class HipLookaheadMixin:
    """Lookahead members of ``HipParser``."""

    def is_function_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier followed by (
        index = self.pos

        index = self.skip_cpp_attributes_at_pos(index)
        while index < len(self.tokens) and (
            self.tokens[index].type in self.FUNCTION_DECLARATION_SPECIFIER_TOKENS
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            is_linkage_specifier = self.tokens[index].type == "EXTERN"
            index += 1
            if is_linkage_specifier:
                index = self.skip_linkage_language_string_at_pos(index)
            index = self.skip_cpp_attributes_at_pos(index)

        index = self.skip_type_at_pos(
            index, allow_unknown_identifier_pointers=self.block_depth == 0
        )
        if index is not None:
            index = self.skip_newlines_at_pos(index)
            while index < len(
                self.tokens
            ) and self.is_interleaved_function_specifier_token(self.tokens[index]):
                index += 1
                index = self.skip_newlines_at_pos(index)
            parameter_list_start = self.skip_function_name_at(index)
            if (
                parameter_list_start is not None
                and self.is_plausible_function_parameter_list_at_pos(
                    parameter_list_start
                )
            ):
                return True

        return False

    def is_plausible_function_parameter_list_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return False

        index += 1
        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        if index >= len(self.tokens):
            return False

        token_type = self.tokens[index].type
        if token_type == "RPAREN":
            return True
        if (
            token_type == "VOID"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RPAREN"
        ):
            return True
        if token_type == "ELLIPSIS":
            return True
        if self.is_function_pointer_parameter_start_at_pos(index):
            return True

        return self.is_plausible_function_parameter_at_pos(index)

    def is_plausible_function_parameter_at_pos(self, index):
        type_end = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if type_end is None:
            return False

        index = self.skip_newlines_at_pos(type_end)
        index = self.skip_cpp_attributes_at_pos(index)
        while index < len(self.tokens) and self.tokens[index].type in {
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)
            index = self.skip_newlines_at_pos(index)

        if index < len(self.tokens) and self.tokens[index].type == "ELLIPSIS":
            index += 1
            index = self.skip_newlines_at_pos(index)

        if index >= len(self.tokens):
            return False

        if self.tokens[index].type in {"COMMA", "RPAREN", "ASSIGN"}:
            return True

        if self.is_declarator_name_token_at(index):
            index = self.skip_variable_declarator_name_at_pos(index)
            if index is None:
                return False
            index = self.skip_newlines_at_pos(index)
            index = self.skip_cpp_attributes_at_pos(index)
            index = self.skip_array_suffix_at_pos(index)
            index = self.skip_newlines_at_pos(index)
            return index < len(self.tokens) and self.tokens[index].type in {
                "COMMA",
                "RPAREN",
                "ASSIGN",
            }

        return False

    def is_function_pointer_parameter_start_at_pos(self, index):
        type_end = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        return (
            type_end is not None
            and type_end + 2 < len(self.tokens)
            and self.tokens[type_end].type == "LPAREN"
            and self.tokens[type_end + 1].type
            in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}
            and self.is_declarator_name_token_at(type_end + 2)
        )

    def skip_function_name_at(self, index):
        if index >= len(self.tokens) or not self.is_function_name_token(
            self.tokens[index]
        ):
            return None

        name = self.tokens[index].value
        index += 1
        index = self.skip_operator_function_suffix_at(index, name)
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while index < len(self.tokens) and self.tokens[index].type == "SCOPE":
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "TILDE":
                index += 1
            if index >= len(self.tokens) or not self.is_function_name_token(
                self.tokens[index]
            ):
                return None
            name = self.tokens[index].value
            index += 1
            index = self.skip_operator_function_suffix_at(index, name)
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        if index < len(self.tokens) and self.tokens[index].type == "LPAREN":
            return index
        return None

    def skip_operator_function_suffix_at(self, index, name):
        if name != "operator" or index >= len(self.tokens):
            return index

        if (
            self.tokens[index].type == "LBRACKET"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RBRACKET"
        ):
            return index + 2
        if (
            self.tokens[index].type == "LPAREN"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RPAREN"
        ):
            return index + 2
        if self.tokens[index].type in self.OVERLOADABLE_OPERATOR_TOKENS:
            return index + 1
        return index

    def is_variable_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier not followed by (
        index = self.pos

        while index < len(self.tokens) and self.tokens[index].type in {
            "__SHARED__",
            "__CONSTANT__",
            "__MANAGED__",
            "__DEVICE__",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
            "CONST",
            "VOLATILE",
            "REGISTER",
        }:
            index += 1

        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        index = self.skip_type_at_pos(index)
        if index is not None:
            while (
                index < len(self.tokens)
                and self.tokens[index].type in self.DECLARATION_QUALIFIER_TOKENS
            ):
                index += 1
            index = self.skip_newlines_at_pos(index)
            structured_binding_end = self.skip_structured_binding_declarator_at_pos(
                index
            )
            if structured_binding_end is not None:
                return True

            declarator_name_end = self.skip_variable_declarator_name_at_pos(index)
            if declarator_name_end is not None:
                index = declarator_name_end
                index = self.skip_newlines_at_pos(index)
                index = self.skip_array_suffix_at_pos(index)
                index = self.skip_newlines_at_pos(index)
                index = self.skip_type_attribute_prefixes_at_pos(index)
                index = self.skip_newlines_at_pos(index)
                if index < len(self.tokens) and self.tokens[index].type in {
                    "SEMICOLON",
                    "ASSIGN",
                    "LBRACKET",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                }:
                    return True

        return False

    def skip_variable_declarator_name_at_pos(self, index):
        if not self.is_declarator_name_token_at(index):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while index < len(self.tokens) and self.tokens[index].type == "SCOPE":
            index += 1
            if not self.is_declarator_name_token_at(index):
                return None
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        return index

    def skip_type_attribute_prefixes_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index].value in self.TYPE_ATTRIBUTE_IDENTIFIERS
        ):
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "LPAREN":
                index = self.skip_balanced_tokens_at_pos(index, "LPAREN", "RPAREN")
            index = self.skip_newlines_at_pos(index)
        return index

    def skip_cpp_attributes_at_pos(self, index):
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "LBRACKET"
            and self.tokens[index + 1].type == "LBRACKET"
        ):
            index += 2
            depth = 1

            while index < len(self.tokens) and depth > 0:
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index].type == "LBRACKET"
                    and self.tokens[index + 1].type == "LBRACKET"
                ):
                    index += 2
                    depth += 1
                    continue
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index].type == "RBRACKET"
                    and self.tokens[index + 1].type == "RBRACKET"
                ):
                    index += 2
                    depth -= 1
                    continue
                index += 1

            index = self.skip_newlines_at_pos(index)

        return index

    def skip_type_at_pos(self, index, allow_unknown_identifier_pointers=False):
        saw_integral_sign = False
        index = self.line_split_type_qualifiers_end(index)
        index = self.skip_cpp_attributes_at_pos(index)
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_PREFIX_TOKENS
        ):
            index += 1

        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            if self.tokens[index].type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        if index >= len(self.tokens):
            return None

        if self.is_implicit_int_type_at_pos(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.is_decltype_type_start_at_pos(index):
            type_token = "IDENTIFIER"
            type_value = "decltype"
            index = self.skip_decltype_type_at_pos(index)
            if index is None:
                return None
        elif self.is_type_token(self.tokens[index]):
            type_token = self.tokens[index].type
            type_value = self.tokens[index].value
            index += 1
            if type_token in self.ELABORATED_TYPE_TOKENS:
                index = self.skip_elaborated_type_name_at_pos(
                    index, keyword_token_type=type_token, allow_scoped_suffix=True
                )
                if index is None:
                    return None
            if type_token in {"CHAR", "SHORT", "INT"}:
                while index < len(self.tokens) and self.tokens[index].type in {
                    "SIGNED",
                    "UNSIGNED",
                    "SHORT",
                    "INT",
                }:
                    index += 1
            if type_token == "LONG" and index < len(self.tokens):
                saw_long_long = False
                if self.tokens[index].type == "LONG":
                    index += 1
                    saw_long_long = True
                if (
                    saw_long_long
                    and index < len(self.tokens)
                    and self.tokens[index].type in {"SIGNED", "UNSIGNED"}
                ):
                    index += 1
                if index < len(self.tokens) and self.tokens[index].type == "INT":
                    index += 1
            elif (
                type_token == "SHORT"
                and index < len(self.tokens)
                and self.tokens[index].type == "INT"
            ):
                index += 1
        else:
            return None

        has_qualified_suffix = False

        if index < len(self.tokens) and self.tokens[index].type == "LT":
            has_qualified_suffix = True
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "SCOPE"
            and self.is_qualified_type_member_token_at(index + 1)
        ):
            has_qualified_suffix = True
            index += 2
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        index = self.skip_postfix_type_qualifiers_at_pos(index)
        can_have_pointer_suffix = (
            allow_unknown_identifier_pointers
            or type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or type_value == "decltype"
            or type_token in self.ELABORATED_TYPE_TOKENS
            or self.is_identifier_type_name(type_value)
            or self.is_probable_identifier_type_name(type_value)
        )
        while (
            can_have_pointer_suffix
            and index < len(self.tokens)
            and self.tokens[index].type
            in {
                "ASTERISK",
                "STAR",
                *self.TYPE_REFERENCE_TOKENS,
            }
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)

        if type_value == "auto" and self.is_structured_binding_declarator_at_pos(index):
            return index

        return self.skip_array_suffix_at_pos(index)

    def skip_elaborated_type_name_at_pos(
        self, index, keyword_token_type=None, allow_scoped_suffix=False
    ):
        index = self.skip_newlines_at_pos(index)
        if (
            index < len(self.tokens)
            and keyword_token_type == "ENUM"
            and self.tokens[index].type in {"CLASS", "STRUCT"}
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)

        if index >= len(self.tokens) or not self.is_qualified_type_member_token_at(
            index
        ):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        if allow_scoped_suffix:
            while (
                index + 1 < len(self.tokens)
                and self.tokens[index].type == "SCOPE"
                and self.is_qualified_type_member_token_at(index + 1)
            ):
                index += 2
                if index < len(self.tokens) and self.tokens[index].type == "LT":
                    index = self.skip_template_at_pos(index)
                    if index is None:
                        return None

        return index

    def is_qualified_type_member_token_at(self, index):
        return index < len(self.tokens) and (
            self.tokens[index].type == "IDENTIFIER"
            or self.is_type_token(self.tokens[index], allow_identifier=False)
        )

    def skip_structured_binding_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index].type != "LBRACKET":
            return None

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type in {"SEMICOLON", "LBRACE", "RBRACE"} and depth == 0:
                return None
            index += 1

        return None

    def is_structured_binding_declarator_at_pos(self, index):
        return self.skip_structured_binding_declarator_at_pos(index) is not None

    def skip_structured_binding_declarator_at_pos(self, index):
        binding_end = self.skip_structured_binding_at_pos(index)
        if binding_end is None:
            return None

        binding_end = self.skip_newlines_at_pos(binding_end)
        if binding_end < len(self.tokens) and self.tokens[binding_end].type in {
            "ASSIGN",
            "SEMICOLON",
        }:
            return binding_end
        return None

    def is_implicit_int_type_at_pos(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index].type
        if token_type in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}:
            return True

        if token_type in {
            "SEMICOLON",
            "ASSIGN",
            "LBRACKET",
            "LPAREN",
            "LBRACE",
            "COMMA",
            "RPAREN",
            "COLON",
        }:
            return True

        if token_type != "IDENTIFIER":
            return False

        if self.is_identifier_type_name(self.tokens[index].value):
            return False

        next_type = (
            self.tokens[index + 1].type if index + 1 < len(self.tokens) else "EOF"
        )
        return next_type in {
            "SEMICOLON",
            "ASSIGN",
            "LBRACKET",
            "LPAREN",
            "LBRACE",
            "COMMA",
            "RPAREN",
            "COLON",
        }

    def is_hip_opaque_handle_type(self, type_name):
        return (
            isinstance(type_name, str)
            and type_name.startswith("hip")
            and type_name.endswith("_t")
        )

    def is_identifier_type_name(self, type_name):
        return (
            type_name in self.type_aliases
            or type_name in self.HIP_IDENTIFIER_TYPE_NAMES
            or self.is_hip_opaque_handle_type(type_name)
        )

    def is_probable_identifier_type_name(self, type_name):
        return isinstance(type_name, str) and type_name[:1].isupper()

    def skip_postfix_type_qualifiers_at_pos(self, index):
        index = self.skip_newlines_at_pos(index)
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.POSTFIX_TYPE_QUALIFIER_TOKENS
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)
        return index

    def skip_linkage_specifier_at_pos(self, index):
        if (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "EXTERN"
            and self.tokens[index + 1].type == "STRING"
        ):
            return index + 2
        return None

    def skip_linkage_language_string_at_pos(self, index):
        if index < len(self.tokens) and self.tokens[index].type == "STRING":
            return index + 1
        return index

    def skip_launch_bounds_at_pos(self, index):
        index += 1
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return index

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type in {"SEMICOLON", "LBRACE", "RBRACE"}:
                return index
            index += 1

        return index

    def skip_template_at_pos(self, index):
        depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LT" and (
                depth == 0 or self.is_nested_template_lt_at_pos(index)
            ):
                depth += 1
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "RSHIFT":
                depth -= 2
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
            elif token_type in {"SEMICOLON", "ASSIGN"}:
                return None
            index += 1

        return None

    def skip_array_suffix_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index].type == "LBRACKET":
            index += 1
            while index < len(self.tokens) and self.tokens[index].type != "RBRACKET":
                index += 1
            if index < len(self.tokens) and self.tokens[index].type == "RBRACKET":
                index += 1
            else:
                break

        return index
