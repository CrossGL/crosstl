"""Function and operator function names, preprocessor lines and linkage blocks in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import PreprocessorNode


class CudaFunctionNameMixin:
    """Function name members of ``CudaParser``."""

    def is_function_name_token(self, token=None):
        token = token or self.current_token
        return bool(token and token[0] in self.FUNCTION_NAME_TOKENS)

    def function_name_end_index_at(self, index):
        if index >= len(self.tokens) or not self.is_function_name_token(
            self.tokens[index]
        ):
            return None

        token_type, token_value = self.tokens[index]
        if token_type == "IDENTIFIER" and token_value == "operator":
            return self.operator_function_name_end_index_at(index)

        return index + 1

    def operator_function_name_end_index_at(self, index):
        index += 1
        if index >= len(self.tokens):
            return None

        token_type = self.tokens[index][0]
        if token_type == "LBRACKET":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "RBRACKET"
                else None
            )
        if token_type == "LPAREN":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "RPAREN"
                else None
            )
        if token_type in self.OVERLOADABLE_OPERATOR_TOKENS:
            return index + 1

        return None

    def is_identifier_type_name(self, name):
        return (
            name in self.type_aliases
            or name in self.struct_names
            or name in self.CUDA_IDENTIFIER_TYPE_NAMES
        )

    def consume_function_name(self):
        if not self.is_function_name_token():
            token_type = self.current_token[0] if self.current_token else "EOF"
            raise CudaSyntaxError(f"Expected function name, got {token_type}")

        if self.current_token == ("IDENTIFIER", "operator"):
            return self.consume_operator_function_name()

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def consume_operator_function_name(self):
        name = self.eat("IDENTIFIER")[1]
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            return f"{name}[]"
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            self.eat("RPAREN")
            return f"{name}()"
        if self.current_token[0] not in self.OVERLOADABLE_OPERATOR_TOKENS:
            raise CudaSyntaxError(
                f"Expected overloaded operator, got {self.current_token[0]}"
            )

        operator_value = self.current_token[1]
        self.eat(self.current_token[0])
        return f"{name}{operator_value}"

    def parse_preprocessor(self):
        directive_token = self.eat("PREPROCESSOR")
        directive_text = directive_token[1].strip()

        if directive_text.startswith("#include"):
            content = directive_text[8:].strip()
            return PreprocessorNode("include", content)
        elif directive_text.startswith("#define"):
            content = directive_text[7:].strip()
            return PreprocessorNode("define", content)
        else:
            return PreprocessorNode("other", directive_text)

    def is_linkage_block_start(self):
        return (
            self.current_token[0] == "EXTERN"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "STRING"
            and self.tokens[self.current_index + 2][0] == "LBRACE"
        )

    def parse_linkage_block(self):
        self.eat("EXTERN")
        language = self.eat("STRING")[1].strip('"')
        self.eat("LBRACE")
        items = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                items.append(self.parse_preprocessor())
            elif self.is_cpp_attribute_specifier_start():
                self.skip_cpp_attribute_specifiers()
            elif self.current_token[0] == "TEMPLATE":
                self.parse_template_declaration()
            elif self.is_namespace_alias_start():
                self.parse_namespace_alias()
            elif self.is_type_alias_start():
                aliases = self.parse_type_alias()
                if isinstance(aliases, list):
                    items.extend(aliases)
                elif aliases is not None:
                    items.append(aliases)
            elif self.is_struct_or_class_declaration_start():
                items.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                items.append(self.parse_enum())
            elif self.is_out_of_class_member_definition_start():
                self.skip_out_of_class_member_definition()
            elif self.is_function_pointer_declaration_start():
                self.skip_out_of_class_member_definition()
            elif self.peek_function():
                func = self.parse_function()
                if func is not None:
                    items.append(func)
            elif self.current_token[0] in self.CUDA_FUNCTION_ENTRY_TOKENS:
                if self.current_token[0] == "DEVICE" and self.peek_variable():
                    self.append_parsed_global_variable(
                        items, self.parse_global_variable()
                    )
                elif self.is_function_pointer_declaration_start():
                    self.skip_out_of_class_member_definition()
                else:
                    func = self.parse_function()
                    if func is not None:
                        items.append(func)
            elif (
                self.current_token[0] in ["CONSTANT", "SHARED"] or self.peek_variable()
            ):
                self.append_parsed_global_variable(items, self.parse_global_variable())
            else:
                self.eat(self.current_token[0])

        self.eat("RBRACE")
        self.apply_linkage_to_items(items, language)
        return items

    def apply_linkage_to_items(self, items, language):
        for item in items:
            if hasattr(item, "qualifiers"):
                qualifiers = getattr(item, "qualifiers", []) or []
                if "extern" not in qualifiers:
                    qualifiers.insert(0, "extern")
                item.qualifiers = qualifiers
                item.linkage = language
