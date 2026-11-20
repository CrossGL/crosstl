"""Struct, union and enum declarations, shared aggregates, out-of-class member definitions, struct methods and balanced-group skipping in the CUDA parser."""

from ..CudaAst import (
    EnumNode,
    StructNode,
    VariableNode,
)


class CudaRecordMixin:
    """Record members of ``CudaParser``."""

    def is_struct_or_class_declaration_start(self):
        if self.current_token[0] not in {"CLASS", "STRUCT", "UNION"}:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(self.current_index + 1)
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

        return index < len(self.tokens) and self.tokens[index][0] in {
            "COLON",
            "LBRACE",
            "SEMICOLON",
        }

    def is_shared_aggregate_declaration_start(self):
        index = self.current_index
        index, _ = self.skip_aggregate_declaration_prefixes_at_index(index)

        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_gnu_attributes_at_index(index)
        if index >= len(self.tokens) or self.tokens[index][0] not in {
            "STRUCT",
            "UNION",
        }:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(index + 1)
        if (
            index < len(self.tokens)
            and self.tokens[index][0] in self.NAME_COMPONENT_TOKENS
        ):
            index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False

        return index < len(self.tokens) and self.tokens[index][0] == "LBRACE"

    def skip_aggregate_declaration_prefixes_at_index(self, index):
        qualifiers = []
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.DECLARATION_PREFIX_TOKENS or token_type in {
                "CONST",
                "VOLATILE",
            }:
                qualifiers.append(token_value)
                index += 1
                continue
            break
        return index, qualifiers

    def parse_aggregate_declaration_prefixes(self):
        qualifiers = []
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in self.DECLARATION_PREFIX_TOKENS or token_type in {
                "CONST",
                "VOLATILE",
            }:
                qualifiers.append(token_value)
                self.eat(token_type)
                continue
            break
        return qualifiers

    def parse_shared_aggregate_declaration(self):
        qualifiers = self.parse_aggregate_declaration_prefixes()
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()

        aggregate_type = self.eat(self.current_token[0])[1]
        attributes = self.parse_cuda_struct_attribute_prefix()

        tag_name = None
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            tag_name = self.parse_name_component()
            if self.current_token[0] == "LESS_THAN":
                tag_name += self.parse_template_suffix()

        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        declarators = []
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            name = self.parse_name_component()
            array_suffix = self.parse_array_suffix()
            declarators.append((name, array_suffix))
            if self.current_token[0] == "ASSIGN":
                self.eat("ASSIGN")
                self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                self.parse_initializer_list()
            if self.current_token[0] != "COMMA":
                break

        self.eat("SEMICOLON")

        if not tag_name:
            first_name = declarators[0][0] if declarators else "union"
            prefix = "__anonymous_shared" if "__shared__" in qualifiers else "anonymous"
            tag_name = f"{prefix}_{first_name}_layout"

        self.struct_names.add(tag_name)
        nodes = [
            self.mark_union(
                StructNode(tag_name, members, attributes=attributes), aggregate_type
            )
        ]
        for name, array_suffix in declarators:
            vtype = f"{tag_name}{array_suffix}"
            if "__shared__" in qualifiers:
                nodes.append(self.create_shared_memory_node(vtype, name, qualifiers))
            else:
                nodes.append(
                    VariableNode(
                        vtype,
                        name,
                        qualifiers=qualifiers,
                    )
                )
        return nodes if len(nodes) > 1 else nodes[0]

    def skip_until_semicolon(self):
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    @staticmethod
    def mark_union(node, aggregate_type):
        """``node`` with ``is_union`` set when it was declared with ``union``, whose members share storage."""
        node.is_union = str(aggregate_type) == "union"
        return node

    def parse_struct(self):
        aggregate_type = self.eat(self.current_token[0])[1]
        attributes = self.parse_cuda_struct_attribute_prefix()
        name = self.parse_name_component()
        if self.current_token[0] == "LESS_THAN":
            name += self.parse_template_suffix()
        self.struct_names.add(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return self.mark_union(
                StructNode(name, [], attributes=attributes), aggregate_type
            )
        if self.current_token[0] == "COLON":
            self.skip_struct_inheritance_clause()
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return self.mark_union(
            StructNode(name, members, attributes=attributes), aggregate_type
        )

    def skip_struct_inheritance_clause(self):
        self.eat("COLON")
        while self.current_token[0] not in {"LBRACE", "EOF"}:
            self.eat(self.current_token[0])

    def is_out_of_class_member_definition_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_struct_method_specifiers_at_index(index)

        if self.is_scoped_member_name_at_index(index):
            return True

        return_type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if return_type_end is None:
            return False

        return_type_end = self.skip_return_pointer_suffix_at_index(return_type_end)
        return_type_end = self.skip_struct_method_specifiers_at_index(return_type_end)
        return self.is_scoped_member_name_at_index(return_type_end)

    def is_scoped_member_name_at_index(self, index):
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

        saw_scope = False
        while index < len(self.tokens) and self.tokens[index][0] == "SCOPE":
            member_end = self.scoped_member_name_component_end_index_at(index + 1)
            if member_end is None:
                return False

            saw_scope = True
            index = member_end
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False

        return (
            saw_scope and index < len(self.tokens) and self.tokens[index][0] == "LPAREN"
        )

    def scoped_member_name_component_end_index_at(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "TEMPLATE":
            index += 1

        if index >= len(self.tokens):
            return None

        token_type, token_value = self.tokens[index]
        if token_type == "IDENTIFIER" and token_value == "operator":
            return self.operator_function_name_end_index_at(index)

        if token_type == "BITWISE_NOT":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
                else None
            )

        if token_type in self.NAME_COMPONENT_TOKENS:
            return index + 1

        return None

    def skip_out_of_class_member_definition(self):
        while self.current_token[0] not in {"LPAREN", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()

        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            elif self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()
            else:
                self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_enum(self):
        self.eat("ENUM")
        is_scoped = False

        if self.current_token[0] in {"CLASS", "STRUCT"}:
            is_scoped = True
            self.eat(self.current_token[0])

        name = None
        if self.current_token[0] == "IDENTIFIER":
            name = self.eat("IDENTIFIER")[1]
            self.type_aliases.add(name)
            self.struct_names.add(name)

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            enum_node = EnumNode(name, [])
            enum_node.underlying_type = underlying_type
            enum_node.is_scoped = is_scoped
            return enum_node

        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        enum_node = EnumNode(name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        return enum_node

    def parse_enum_members(self):
        members = []

        while self.current_token[0] not in {"RBRACE", "EOF"}:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            member_name = self.current_token[1]
            self.eat(self.current_token[0])

            member_value = None
            if self.current_token[0] == "ASSIGN":
                self.eat("ASSIGN")
                member_value = self.parse_expression()

            members.append((member_name, member_value))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        return members

    def parse_struct_members(self):
        members = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            if self.is_struct_access_label():
                self.eat(self.current_token[0])
                self.eat("COLON")
                continue
            if self.is_friend_type_declaration_start():
                self.skip_until_semicolon()
                continue
            if self.is_identifier_value("friend"):
                if self.is_struct_method_start():
                    self.skip_struct_method()
                else:
                    self.skip_until_semicolon()
                continue
            if self.is_templated_type_alias_start():
                self.parse_template_declaration()
                self.parse_type_alias()
                continue
            if self.is_templated_struct_or_class_declaration_start():
                self.parse_template_declaration()
                self.parse_struct()
                continue
            if self.is_anonymous_aggregate_member_start():
                members.extend(self.parse_anonymous_aggregate_member())
                continue
            if self.is_struct_or_class_declaration_start():
                self.parse_struct()
                continue
            if self.current_token[0] == "ENUM":
                self.parse_enum()
                continue
            if self.is_struct_method_start():
                self.skip_struct_method()
                continue
            if self.is_type_alias_start():
                self.parse_type_alias()
                continue

            member_declarations = self.parse_variable_declaration_list()
            members.extend(member_declarations)
            self.eat("SEMICOLON")
        return members

    def is_templated_type_alias_start(self):
        if self.current_token[0] != "TEMPLATE":
            return False
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        return index < len(self.tokens) and self.tokens[index] == (
            "IDENTIFIER",
            "using",
        )

    def is_templated_struct_or_class_declaration_start(self):
        if self.current_token[0] != "TEMPLATE":
            return False
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        return index < len(self.tokens) and self.tokens[index][0] in {"STRUCT", "CLASS"}

    def is_friend_type_declaration_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if self.is_function_identifier_specifier_at_index(index):
                index += 1
                continue
            if token_value == "friend":
                break
            return False

        return (
            index + 1 < len(self.tokens)
            and self.tokens[index] == ("IDENTIFIER", "friend")
            and self.tokens[index + 1][0] in {"CLASS", "STRUCT"}
        )

    def is_struct_access_label(self):
        return (
            self.current_token[0] in {"PUBLIC", "PRIVATE", "PROTECTED"}
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "COLON"
        )

    def is_struct_method_start(self):
        index = self.current_index
        while True:
            next_index = self.skip_optional_template_declaration_at_index(index)
            next_index = self.skip_cpp_attribute_specifiers_at_index(next_index)
            next_index = self.skip_struct_method_specifiers_at_index(next_index)
            next_index = self.skip_cpp_attribute_specifiers_at_index(next_index)
            if next_index == index:
                break
            index = next_index

        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_value == "operator":
            return True
        if (
            token_type == "BITWISE_NOT"
            and index + 2 < len(self.tokens)
            and self.tokens[index + 1][0] == "IDENTIFIER"
            and self.tokens[index + 2][0] == "LPAREN"
        ):
            return True
        if (
            token_type == "IDENTIFIER"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1][0] == "LPAREN"
        ):
            return True

        name_index = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if name_index is None or name_index >= len(self.tokens):
            return False
        name_index = self.skip_return_pointer_suffix_at_index(name_index)

        while name_index < len(self.tokens) and (
            self.tokens[name_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.tokens[name_index][0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(name_index)
        ):
            if self.tokens[name_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                name_index = self.skip_function_attribute_at_index(name_index)
            else:
                name_index += 1

        token_type, token_value = self.tokens[name_index]
        if token_value == "operator":
            return True
        qualified_name_end = self.qualified_function_name_end_index_at(name_index)
        if qualified_name_end is not None:
            return (
                qualified_name_end < len(self.tokens)
                and self.tokens[qualified_name_end][0] == "LPAREN"
            )
        return (
            self.is_function_name_token(self.tokens[name_index])
            and name_index + 1 < len(self.tokens)
            and self.tokens[name_index + 1][0] == "LPAREN"
        )

    def qualified_function_name_end_index_at(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "SCOPE":
            index += 1

        if (
            index >= len(self.tokens)
            or self.tokens[index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return None

        index += 1
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
        ):
            index += 2
        return index

    def skip_return_pointer_suffix_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)
        return index

    def skip_optional_template_declaration_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "TEMPLATE":
            return index

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index, allow_assignment=True)
            if skipped is not None:
                return skipped
        return index

    def skip_struct_method_specifiers_at_index(self, index):
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if (
                token_type in self.FUNCTION_SPECIFIER_TOKENS
                or token_type == "VIRTUAL"
                or (
                    token_type == "IDENTIFIER"
                    and self.tokens[index][1]
                    in self.STRUCT_METHOD_IDENTIFIER_SPECIFIERS
                )
                or self.is_function_identifier_specifier_at_index(index)
            ):
                if (
                    token_type == "EXTERN"
                    and index + 1 < len(self.tokens)
                    and self.tokens[index + 1][0] == "STRING"
                ):
                    index += 2
                    continue
                if (
                    token_type == "IDENTIFIER"
                    and self.tokens[index][1] == "explicit"
                    and index + 1 < len(self.tokens)
                    and self.tokens[index + 1][0] == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_index(
                        index + 1, "LPAREN", "RPAREN"
                    )
                    continue
                index += 1
                continue
            break
        return index

    def is_anonymous_aggregate_member_start(self):
        if self.current_token[0] not in {"STRUCT", "UNION"}:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(self.current_index + 1)
        return index < len(self.tokens) and self.tokens[index][0] == "LBRACE"

    def parse_anonymous_aggregate_member(self):
        aggregate_type = self.current_token[1]
        self.eat(self.current_token[0])
        attributes = self.parse_cuda_struct_attribute_prefix()
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            name = self.parse_name_component()
            vtype = aggregate_type + self.parse_array_suffix()
            self.eat("SEMICOLON")
            return [VariableNode(vtype, name, attributes=attributes)]

        self.eat("SEMICOLON")
        return members

    def skip_struct_method(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            if self.current_token[0] == "LESS_THAN":
                self.parse_template_suffix()
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()

        while self.current_token[0] not in {"SEMICOLON", "RBRACE", "EOF"}:
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            elif self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()
            elif self.current_token[0] == "LBRACE":
                self.skip_balanced_brace_block()
                if self.current_token[0] not in {"COMMA", "LBRACE"}:
                    break
            else:
                self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_balanced_brace_block(self):
        self.eat("LBRACE")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACE")
                    break
            self.eat(token_type)

    def skip_balanced_parentheses(self):
        self.eat("LPAREN")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)

    def skip_balanced_brackets(self):
        self.eat("LBRACKET")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACKET")
                    break
            self.eat(token_type)
