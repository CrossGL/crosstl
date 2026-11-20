"""typedef and using aliases, function-pointer and member-pointer alias declarators, alignment, GNU, CUDA and C++ attributes, and namespace aliases in the CUDA parser."""

from crosstl.translator.refusals import CudaSyntaxError

from ..CudaAst import (
    EnumNode,
    StructNode,
    TypeAliasNode,
)


class CudaAliasMixin:
    """Alias members of ``CudaParser``."""

    def is_type_alias_start(self):
        return self.current_token[0] == "TYPEDEF" or self.is_identifier_value("using")

    def parse_type_alias(self):
        if self.current_token[0] == "TYPEDEF":
            return self.parse_typedef_alias()
        return self.parse_using_alias()

    def parse_typedef_alias(self):
        self.eat("TYPEDEF")
        self.parse_cuda_declaration_attributes()
        if self.current_token[0] in {"CLASS", "STRUCT", "UNION"}:
            return self.parse_typedef_struct_alias()
        if self.current_token[0] == "ENUM":
            return self.parse_typedef_enum_alias()

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        aliases = [self.parse_type_alias_declarator(first_type, allow_prefix=False)]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            aliases.append(
                self.parse_type_alias_declarator(base_type, allow_prefix=True)
            )

        self.eat("SEMICOLON")
        return aliases

    def parse_type_alias_declarator(self, base_type, allow_prefix):
        alias_type = base_type
        if allow_prefix:
            alias_type = self.parse_declarator_prefix(alias_type)

        if self.current_token[0] == "LPAREN":
            return self.parse_function_pointer_type_alias_declarator(alias_type)

        name = self.parse_name_component()
        if self.current_token[0] == "LPAREN":
            return self.parse_function_type_alias_declarator(alias_type, name)

        alias_type += self.parse_array_suffix()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_function_type_alias_declarator(self, return_type, name):
        params = self.parse_parameters()
        alias_type = f"{return_type} ()".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def parse_function_pointer_type_alias_declarator(self, return_type):
        self.eat("LPAREN")
        member_pointer_scope = self.parse_member_pointer_scope_prefix()
        pointer_prefix = self.parse_declarator_prefix("")
        if member_pointer_scope:
            pointer_prefix = f"{member_pointer_scope}{pointer_prefix}"
        name = self.eat("IDENTIFIER")[1]
        self.eat("RPAREN")
        params = self.parse_parameters()

        alias_type = f"{return_type} ({pointer_prefix})".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def parse_member_pointer_scope_prefix(self):
        parts = []
        while self.is_member_pointer_scope_component_start(self.current_index):
            part = self.parse_name_component()
            if self.current_token[0] == "LESS_THAN":
                part += self.parse_template_suffix()
            parts.append(part)
            self.eat("SCOPE")
            if self.current_token[0] in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}:
                break

        return f"{'::'.join(parts)}::" if parts else ""

    def is_member_pointer_scope_component_start(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return False

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            index = self.skip_template_at_index(index)
            if index is None:
                return False

        return index < len(self.tokens) and self.tokens[index][0] == "SCOPE"

    def parse_typedef_struct_alias(self):
        aggregate_type = self.eat(self.current_token[0])[1]
        attributes = self.parse_cuda_struct_attribute_prefix()

        tag_name = None
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            tag_name = self.parse_name_component()

        if self.current_token[0] != "LBRACE":
            alias = self.parse_type_alias_declarator(
                f"{aggregate_type} {tag_name}", allow_prefix=True
            )
            self.eat("SEMICOLON")
            return alias

        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        alias_name = tag_name
        if self.current_token[0] == "IDENTIFIER":
            alias_name = self.eat("IDENTIFIER")[1]
        if not alias_name:
            raise CudaSyntaxError("Expected typedef struct alias name")

        self.eat("SEMICOLON")
        self.type_aliases.add(alias_name)
        self.struct_names.add(alias_name)
        return self.mark_union(
            StructNode(alias_name, members, attributes=attributes), aggregate_type
        )

    def parse_typedef_enum_alias(self):
        self.eat("ENUM")
        is_scoped = False

        if self.current_token[0] in {"CLASS", "STRUCT"}:
            is_scoped = True
            self.eat(self.current_token[0])

        tag_name = None
        if self.current_token[0] == "IDENTIFIER":
            tag_name = self.eat("IDENTIFIER")[1]

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        if self.current_token[0] != "LBRACE":
            alias = self.parse_type_alias_declarator(
                f"enum {tag_name}", allow_prefix=True
            )
            self.eat("SEMICOLON")
            return alias

        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")

        alias_name = tag_name
        if self.current_token[0] == "IDENTIFIER":
            alias_name = self.eat("IDENTIFIER")[1]
        if not alias_name:
            raise CudaSyntaxError("Expected typedef enum alias name")

        self.eat("SEMICOLON")
        for name in {tag_name, alias_name}:
            if name:
                self.type_aliases.add(name)
                self.struct_names.add(name)

        enum_node = EnumNode(alias_name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        enum_node.tag_name = tag_name
        return enum_node

    def parse_alignment_attributes(self):
        attributes = []
        while self.current_token[0] == "ALIGNAS":
            name = self.eat("ALIGNAS")[1]
            self.eat("LPAREN")
            args = self.collect_balanced_tokens_until("RPAREN")
            self.eat("RPAREN")
            attributes.append(f"{name}({args})")
        return attributes

    def skip_gnu_attributes(self):
        while (
            self.current_token == ("IDENTIFIER", "__attribute__")
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        ):
            self.eat("IDENTIFIER")
            self.skip_balanced_parentheses()

    def parse_cuda_declaration_attributes(self):
        while self.is_cuda_declaration_attribute_identifier():
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()

    def parse_cuda_struct_attribute_prefix(self):
        attributes = []
        while True:
            before_index = self.current_index
            self.parse_cuda_declaration_attributes()
            attributes.extend(self.parse_alignment_attributes())
            if self.current_index == before_index:
                return attributes

    def is_cuda_declaration_attribute_identifier_at_index(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] in self.CUDA_DECLARATION_ATTRIBUTE_IDENTIFIERS
        )

    def is_cuda_declaration_attribute_identifier(self):
        return self.is_cuda_declaration_attribute_identifier_at_index(
            self.current_index
        )

    def skip_cuda_declaration_attributes_at_index(self, index):
        while self.is_cuda_declaration_attribute_identifier_at_index(index):
            index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
                index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")
        return index

    def skip_cuda_struct_attribute_prefix_at_index(self, index):
        while True:
            before_index = index
            index = self.skip_cuda_declaration_attributes_at_index(index)
            index = self.skip_alignment_attributes_at_index(index)
            if index == before_index:
                return index

    def is_cpp_attribute_specifier_start(self):
        return self.is_cpp_attribute_specifier_start_at_index(self.current_index)

    def is_cpp_attribute_specifier_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "LBRACKET"
            and self.tokens[index + 1][0] == "LBRACKET"
        )

    def skip_cpp_attribute_specifiers(self):
        while self.is_cpp_attribute_specifier_start():
            self.eat("LBRACKET")
            self.eat("LBRACKET")
            depth = 1

            while self.current_token[0] != "EOF" and depth > 0:
                if self.is_cpp_attribute_specifier_start():
                    self.eat("LBRACKET")
                    self.eat("LBRACKET")
                    depth += 1
                    continue
                if (
                    self.current_index + 1 < len(self.tokens)
                    and self.current_token[0] == "RBRACKET"
                    and self.tokens[self.current_index + 1][0] == "RBRACKET"
                ):
                    self.eat("RBRACKET")
                    self.eat("RBRACKET")
                    depth -= 1
                    continue
                self.eat(self.current_token[0])

    def skip_cpp_attribute_specifiers_at_index(self, index):
        while self.is_cpp_attribute_specifier_start_at_index(index):
            index += 2
            depth = 1

            while index < len(self.tokens) and depth > 0:
                if self.is_cpp_attribute_specifier_start_at_index(index):
                    index += 2
                    depth += 1
                    continue
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index][0] == "RBRACKET"
                    and self.tokens[index + 1][0] == "RBRACKET"
                ):
                    index += 2
                    depth -= 1
                    continue
                index += 1

        return index

    def collect_balanced_tokens_until(self, terminator):
        tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                token_type == terminator
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break

            tokens.append(token_value)
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

        return " ".join(tokens)

    def collect_balanced_raw_tokens_until(self, terminator):
        tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                token_type == terminator
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break

            tokens.append(self.current_token)
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

        return tokens

    def parse_using_alias(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "NAMESPACE":
            self.skip_until_semicolon()
            return None

        name = self.eat("IDENTIFIER")[1]
        if self.current_token[0] == "SCOPE":
            self.skip_until_semicolon()
            return None

        self.parse_cuda_declaration_attributes()
        self.eat("ASSIGN")
        self.skip_alias_type_annotations()
        if (
            self.is_decltype_alias_type_start()
            or self.is_dependent_decltype_alias_type_start()
        ):
            alias_type = self.parse_alias_type_expression_until_semicolon()
            self.eat("SEMICOLON")
            self.type_aliases.add(name)
            return TypeAliasNode(alias_type, name)

        alias_type = self.parse_type()
        if self.is_unnamed_function_pointer_alias_declarator():
            return self.parse_using_function_pointer_alias_declarator(name, alias_type)

        self.eat("SEMICOLON")
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def skip_alias_type_annotations(self):
        while self.is_alias_type_annotation_start():
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()

    def is_alias_type_annotation_start(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        if not self.is_macro_like_identifier(self.current_token[1]):
            return False

        if self.current_index + 1 >= len(self.tokens):
            return False

        next_type = self.tokens[self.current_index + 1][0]
        return next_type in {
            *self.TYPE_QUALIFIER_TOKENS,
            *self.TYPE_TOKENS,
            *self.ELABORATED_TYPE_TOKENS,
            *self.NAME_COMPONENT_TOKENS,
            "SCOPE",
        }

    def is_decltype_alias_type_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "decltype"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        )

    def is_dependent_decltype_alias_type_start(self):
        return (
            self.current_token[0] == "TYPENAME"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1] == ("IDENTIFIER", "decltype")
            and self.tokens[self.current_index + 2][0] == "LPAREN"
        )

    def parse_alias_type_expression_until_semicolon(self):
        parts = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        template_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and template_depth == 0
            ):
                break

            parts.append(token_value)
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
            elif token_type == "LESS_THAN":
                template_depth += 1
            elif token_type == "GREATER_THAN" and template_depth > 0:
                template_depth -= 1
            elif token_type == "SHIFT_RIGHT" and template_depth > 0:
                template_depth = max(0, template_depth - 2)
            self.eat(token_type)

        return self.format_template_parts(parts)

    def is_unnamed_function_pointer_alias_declarator(self):
        return (
            self.current_index + 2 < len(self.tokens)
            and self.current_token[0] == "LPAREN"
            and self.tokens[self.current_index + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[self.current_index + 2][0] == "RPAREN"
        )

    def parse_using_function_pointer_alias_declarator(self, name, return_type):
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        self.eat("RPAREN")
        params = self.parse_parameters() if self.current_token[0] == "LPAREN" else []
        self.eat("SEMICOLON")

        alias_type = f"{return_type} ({pointer_prefix})".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def is_namespace_alias_start(self):
        return (
            self.current_token[0] == "NAMESPACE"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "IDENTIFIER"
            and self.tokens[self.current_index + 2][0] == "ASSIGN"
        )

    def parse_namespace_alias(self):
        self.eat("NAMESPACE")
        alias = self.eat("IDENTIFIER")[1]
        self.eat("ASSIGN")
        target_parts = []
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            token_type, token_value = self.current_token
            target_parts.append(token_value)
            self.eat(token_type)

        target = "".join(target_parts).strip()
        if alias and target:
            self.namespace_aliases[alias] = target
        return None
