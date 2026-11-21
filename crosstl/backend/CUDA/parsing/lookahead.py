"""Declaration lookahead in the CUDA parser: struct and user function names, function and variable declaration detection and the token skippers it uses."""


class CudaLookaheadMixin:
    """Lookahead members of ``CudaParser``."""

    def collect_struct_names(self):
        names = set()
        for index, token in enumerate(self.tokens[:-1]):
            if (
                token[0] in {"CLASS", "STRUCT"}
                and self.tokens[index + 1][0] == "IDENTIFIER"
            ):
                names.add(self.tokens[index + 1][1])
        return names

    def collect_user_function_names(self):
        names = set()
        depth = 0
        index = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
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
                    names.add(self.tokens[name_index][1])
                    index = name_index + 1
                    continue

            index += 1

        return names

    def function_name_index_at(self, index):
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        while index < len(self.tokens) and (
            self.tokens[index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(index)
        ):
            if self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if (
                self.tokens[index][0] == "EXTERN"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "STRING"
            ):
                index += 2
                continue
            index += 1

        index = self.skip_cpp_attribute_specifiers_at_index(index)

        index = self.skip_type_at_index(index, allow_unknown_identifier_declarator=True)
        if index is None:
            return None

        while index < len(self.tokens) and (
            self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.tokens[index][0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(index)
        ):
            if self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
            else:
                index += 1

        name_end_index = self.function_name_end_index_at(index)
        if name_end_index is not None:
            suffix_index = self.skip_optional_function_template_suffix_at_index(
                name_end_index
            )
            if (
                suffix_index < len(self.tokens)
                and self.tokens[suffix_index][0] == "LPAREN"
                and self.is_plausible_function_parameter_list_at_index(suffix_index)
            ):
                return index

        return None

    def peek_function(self):
        saved_index = self.current_index

        saved_index = self.skip_cpp_attribute_specifiers_at_index(saved_index)
        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
                continue
            if (
                self.tokens[saved_index][0] == "EXTERN"
                and saved_index + 1 < len(self.tokens)
                and self.tokens[saved_index + 1][0] == "STRING"
            ):
                saved_index += 2
                continue
            saved_index += 1

        saved_index = self.skip_cpp_attribute_specifiers_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        while (
            saved_index is not None
            and saved_index < len(self.tokens)
            and (
                self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
                or self.tokens[saved_index][0]
                in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
                or self.is_function_identifier_specifier_at_index(saved_index)
            )
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
            else:
                saved_index += 1

        if saved_index is not None:
            name_end_index = self.function_name_end_index_at(saved_index)
            if name_end_index is not None:
                suffix_index = self.skip_optional_function_template_suffix_at_index(
                    name_end_index
                )
                return (
                    suffix_index < len(self.tokens)
                    and self.tokens[suffix_index][0] == "LPAREN"
                    and self.is_plausible_function_parameter_list_at_index(suffix_index)
                )

        return False

    def skip_optional_function_template_suffix_at_index(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index)
            if skipped is not None:
                return skipped
        return index

    def is_function_pointer_declaration_start(self):
        saved_index = self.current_index

        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
                continue
            saved_index += 1

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )
        if saved_index is None:
            return False

        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
            else:
                saved_index += 1

        return (
            saved_index + 3 < len(self.tokens)
            and self.tokens[saved_index][0] == "LPAREN"
            and self.tokens[saved_index + 1][0] == "MULTIPLY"
            and self.tokens[saved_index + 2][0] in self.NAME_COMPONENT_TOKENS
            and self.tokens[saved_index + 3][0] in {"RPAREN", "LPAREN"}
        )

    def peek_variable(self):
        saved_index = self.current_index

        saved_index = self.skip_declaration_prefixes_at_index(saved_index)
        saved_index = self.skip_alignment_attributes_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        if saved_index is not None:
            saved_index = self.skip_interleaved_declaration_qualifiers_at_index(
                saved_index
            )
            saved_index = self.skip_alignment_attributes_at_index(saved_index)
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] in self.NAME_COMPONENT_TOKENS
            ):
                saved_index += 1
                saved_index = self.skip_array_suffix_at_index(saved_index)

                if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
                    "SEMICOLON",
                    "ASSIGN",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                ]:
                    return True

        return False

    def skip_declaration_prefixes_at_index(self, index):
        while index < len(self.tokens):
            next_index = self.skip_cuda_declaration_attributes_at_index(index)
            if next_index != index:
                index = next_index
                continue

            token_type = self.tokens[index][0]
            if token_type in self.DECLARATION_PREFIX_TOKENS:
                index += 1
                continue
            if (
                token_type == "VOLATILE"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] in self.CUDA_STORAGE_QUALIFIER_TOKENS
            ):
                index += 1
                continue
            break
        return index

    def skip_interleaved_declaration_qualifiers_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.DECLARATION_PREFIX_TOKENS
        ):
            index += 1
        return index

    def skip_alignment_attributes_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "ALIGNAS":
            index += 1
            if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
                continue

            depth = 0
            while index < len(self.tokens):
                token_type = self.tokens[index][0]
                if token_type == "LPAREN":
                    depth += 1
                elif token_type == "RPAREN":
                    depth -= 1
                    if depth == 0:
                        index += 1
                        break
                elif token_type == "EOF":
                    return index
                index += 1

        return index

    def skip_gnu_attributes_at_index(self, index):
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index] == ("IDENTIFIER", "__attribute__")
            and self.tokens[index + 1][0] == "LPAREN"
        ):
            index = self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")
        return index

    def skip_type_at_index(self, index, allow_unknown_identifier_declarator=False):
        saw_integral_sign = False
        saw_type_qualifier = False
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            saw_type_qualifier = True
            if self.tokens[index][0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        if index >= len(self.tokens):
            return None

        has_qualified_suffix = False

        if self.is_implicit_int_type_at_index(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.is_decltype_type_start_at_index(index):
            type_token = "IDENTIFIER"
            type_value = "decltype"
            index = self.skip_decltype_type_at_index(index)
            if index is None:
                return None
        elif self.tokens[index][0] in self.ELABORATED_TYPE_TOKENS:
            type_token = self.tokens[index][0]
            type_value = self.tokens[index][1]
            index += 1
            if (
                index < len(self.tokens)
                and self.tokens[index][0] in self.NAME_COMPONENT_TOKENS
            ):
                type_value += f" {self.tokens[index][1]}"
                index += 1
            else:
                return None
        elif self.tokens[index][0] in self.TYPE_TOKENS:
            type_token = self.tokens[index][0]
            type_value = self.tokens[index][1]
            index += 1
            index = self.skip_composite_scalar_type_suffix_at_index(index, type_token)
        elif self.is_global_qualified_name_start_at_index(index):
            type_token = "IDENTIFIER"
            type_value = f"::{self.tokens[index + 1][1]}"
            has_qualified_suffix = True
            index += 2
        else:
            return None

        while True:
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                has_qualified_suffix = True
                index = self.skip_template_at_index(index)
                if index is None:
                    return None
                continue
            if self.is_qualified_type_member_at_index(index):
                has_qualified_suffix = True
                index = self.skip_qualified_type_member_at_index(index)
                continue
            break

        index = self.skip_postfix_type_qualifiers_at_index(index)
        can_have_pointer_suffix = (
            type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or type_value == "decltype"
            or self.is_identifier_type_name(type_value)
            or type_token in self.ELABORATED_TYPE_TOKENS
            or saw_type_qualifier
            or (
                allow_unknown_identifier_declarator
                and self.is_probable_identifier_type_name(type_value)
                and self.has_unknown_identifier_declarator_suffix_at_index(index)
            )
        )
        while (
            can_have_pointer_suffix
            and index < len(self.tokens)
            and self.tokens[index][0] in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return self.skip_array_suffix_at_index(index)

    def is_probable_identifier_type_name(self, name):
        return (
            name in self.type_aliases
            or name in self.struct_names
            or name in self.CUDA_IDENTIFIER_TYPE_NAMES
            or name.endswith("_t")
            or any(character.isupper() for character in name)
        )

    def has_unknown_identifier_declarator_suffix_at_index(self, index):
        suffix_index = index
        saw_declarator_marker = False

        while suffix_index < len(self.tokens):
            token_type = self.tokens[suffix_index][0]
            if token_type in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}:
                saw_declarator_marker = True
                suffix_index += 1
                suffix_index = self.skip_postfix_type_qualifiers_at_index(suffix_index)
                continue
            if token_type in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
                suffix_index += 1
                continue
            break

        if (
            not saw_declarator_marker
            or suffix_index >= len(self.tokens)
            or self.tokens[suffix_index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return False

        suffix_index += 1
        suffix_index = self.skip_array_suffix_at_index(suffix_index)
        return suffix_index < len(self.tokens) and self.tokens[suffix_index][0] in {
            "SEMICOLON",
            "ASSIGN",
            "LPAREN",
            "LBRACE",
            "COMMA",
        }

    def is_function_identifier_specifier_at_index(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] in self.FUNCTION_IDENTIFIER_SPECIFIERS
        )

    def is_function_identifier_specifier(self):
        return self.is_function_identifier_specifier_at_index(self.current_index)

    def is_plausible_function_parameter_list_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
            return False

        index += 1
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        index = self.skip_cuda_parameter_attributes_at_index(index)
        if index >= len(self.tokens):
            return False

        token_type = self.tokens[index][0]
        if token_type == "RPAREN":
            return True
        if (
            token_type == "VOID"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1][0] == "RPAREN"
        ):
            return True
        if self.is_ellipsis_at_index(index):
            return True
        if self.is_function_pointer_parameter_start_at_index(index):
            return True

        return self.is_plausible_function_parameter_at_index(index)

    def is_plausible_function_parameter_at_index(self, index):
        index = self.skip_cuda_parameter_attributes_at_index(index)
        type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if type_end is None:
            return False

        index = self.skip_cpp_attribute_specifiers_at_index(type_end)
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        if self.is_ellipsis_at_index(index):
            index += 3

        if index >= len(self.tokens):
            return False

        if self.tokens[index][0] in {"COMMA", "RPAREN", "ASSIGN"}:
            return True

        if self.tokens[index][0] in self.NAME_COMPONENT_TOKENS:
            index += 1
            index = self.skip_array_suffix_at_index(index)
            index = self.skip_cpp_attribute_specifiers_at_index(index)
            return index < len(self.tokens) and self.tokens[index][0] in {
                "COMMA",
                "RPAREN",
                "ASSIGN",
            }

        return False

    def skip_cuda_parameter_attributes_at_index(self, index):
        while index < len(self.tokens) and (
            self.tokens[index][0] in self.CUDA_PARAMETER_ATTRIBUTE_TOKENS
            or (
                self.tokens[index][0] == "IDENTIFIER"
                and self.tokens[index][1] in self.CUDA_PARAMETER_ATTRIBUTE_IDENTIFIERS
            )
        ):
            index += 1
        return index

    def is_function_pointer_parameter_start_at_index(self, index):
        type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        return (
            type_end is not None
            and type_end + 2 < len(self.tokens)
            and self.tokens[type_end][0] == "LPAREN"
            and self.tokens[type_end + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[type_end + 2][0] in self.NAME_COMPONENT_TOKENS
        )

    def is_implicit_int_type_at_index(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index][0]
        if token_type == "RPAREN":
            return True

        if token_type in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}:
            return True

        if token_type != "IDENTIFIER":
            return False

        next_type = self.tokens[index + 1][0] if index + 1 < len(self.tokens) else "EOF"
        return next_type in {
            "SEMICOLON",
            "ASSIGN",
            "LBRACKET",
            "LPAREN",
            "LBRACE",
            "COMMA",
            "RPAREN",
        }

    def is_global_qualified_name_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
        )

    def is_qualified_type_member_at_index(self, index):
        if index + 1 >= len(self.tokens) or self.tokens[index][0] != "SCOPE":
            return False
        if self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS:
            return True
        return (
            index + 2 < len(self.tokens)
            and self.tokens[index + 1][0] == "TEMPLATE"
            and self.tokens[index + 2][0] in self.NAME_COMPONENT_TOKENS
        )

    def skip_qualified_type_member_at_index(self, index):
        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "TEMPLATE":
            index += 1
        return index + 1

    def skip_composite_scalar_type_suffix_at_index(self, index, type_token):
        if type_token == "LONG":
            saw_long_long = False
            if index < len(self.tokens) and self.tokens[index][0] == "LONG":
                index += 1
                saw_long_long = True

            if (
                saw_long_long
                and index < len(self.tokens)
                and self.tokens[index][0] in {"SIGNED", "UNSIGNED"}
            ):
                index += 1

            if index < len(self.tokens) and self.tokens[index][0] == "INT":
                return index + 1
            if (
                not saw_long_long
                and index < len(self.tokens)
                and self.tokens[index][0] == "DOUBLE"
            ):
                return index + 1
        elif (
            type_token == "SHORT"
            and index < len(self.tokens)
            and self.tokens[index][0] == "INT"
        ):
            return index + 1
        return index

    def skip_postfix_type_qualifiers_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS
        ):
            index += 1
        return index

    def skip_function_attribute_at_index(self, index):
        index += 1
        if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
            return index

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return index
            index += 1

        return index

    def skip_launch_bounds_at_index(self, index):
        return self.skip_function_attribute_at_index(index)

    def skip_template_at_index(self, index, allow_assignment=False):
        depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "SHIFT_RIGHT":
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
            elif token_type in {"SEMICOLON", "EOF"} or (
                token_type == "ASSIGN" and not allow_assignment
            ):
                return None
            index += 1

        return None

    def skip_array_suffix_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            index += 1
            while index < len(self.tokens) and self.tokens[index][0] != "RBRACKET":
                index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "RBRACKET":
                index += 1
            else:
                break

        return index
