"""Struct, union, enum and class declarations, record members, constructors and conversion operators in the HIP parser."""

from ..HipAst import (
    EnumNode,
    FunctionNode,
    StructNode,
    VariableNode,
)


class HipRecordMixin:
    """Record members of ``HipParser``."""

    def parse_struct(self):
        self.consume("STRUCT")
        self.skip_newlines()
        self.skip_cpp_attributes()
        attributes = self.parse_type_attribute_prefixes()
        self.skip_newlines()

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)
            if self.match("LT"):
                name += self.parse_template_suffix()

        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        self.skip_newlines()
        is_final = self.consume_record_final_specifier()
        base_classes = self.parse_class_inheritance_clause()
        self.skip_newlines()
        members = []
        has_body = self.match("LBRACE")
        if has_body:
            self.consume("LBRACE")
            members = self.parse_struct_members(name)
            self.consume("RBRACE")

        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        struct_node = StructNode(
            name, members, self.current_record_layout_attributes(attributes)
        )
        struct_node.is_final = is_final
        struct_node.has_body = has_body
        struct_node.base_classes = base_classes
        self.skip_newlines()
        declarations = self.parse_record_trailing_declarators("struct", name)
        if declarations:
            # ``struct Inner inner;`` names an existing record; only a body declares one.
            return (
                [struct_node, *declarations] if has_body or not name else declarations
            )

        if self.match("SEMICOLON"):
            self.advance()

        return struct_node

    def parse_union(self):
        self.consume("UNION")
        self.skip_newlines()
        self.skip_cpp_attributes()
        attributes = self.parse_type_attribute_prefixes()
        self.skip_newlines()

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)
            if self.match("LT"):
                name += self.parse_template_suffix()

        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        self.skip_newlines()
        members = []
        has_body = self.match("LBRACE")
        if has_body:
            self.consume("LBRACE")
            members = self.parse_struct_members(name)
            self.consume("RBRACE")

        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        union_node = StructNode(
            name, members, self.current_record_layout_attributes(attributes)
        )
        union_node.is_union = True
        union_node.has_body = has_body

        self.skip_newlines()
        declarations = self.parse_record_trailing_declarators("union", name)
        if declarations:
            return [union_node, *declarations] if has_body or not name else declarations

        if self.match("SEMICOLON"):
            self.advance()

        return union_node

    def parse_record_trailing_declarators(self, keyword, name):
        if not (
            self.is_declarator_name_token()
            or self.match("ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS)
        ):
            return []

        record_type = f"{keyword} {name}" if name else f"{keyword} <anonymous>"
        declarations = [
            self.parse_variable_declarator(record_type, [], allow_prefix=True)
        ]

        while self.match("COMMA"):
            self.advance()
            declarations.append(
                self.parse_variable_declarator(record_type, [], allow_prefix=True)
            )

        self.skip_newlines()
        if self.match("SEMICOLON"):
            self.advance()

        return declarations

    def parse_enum(self):
        self.consume("ENUM")
        is_scoped = False
        self.skip_newlines()

        if self.match("CLASS", "STRUCT"):
            is_scoped = True
            self.advance()
            self.skip_newlines()

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)

        underlying_type = None
        self.skip_newlines()
        if self.match("COLON"):
            self.advance()
            self.skip_newlines()
            underlying_type = self.parse_type()

        self.skip_newlines()
        self.consume("LBRACE")
        members = self.parse_enum_members()
        self.consume("RBRACE")

        enum_node = EnumNode(name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        self.skip_newlines()
        declarations = self.parse_record_trailing_declarators("enum", name)
        if declarations:
            return [enum_node, *declarations]

        if self.match("SEMICOLON"):
            self.advance()
        return enum_node

    def parse_enum_members(self):
        members = []

        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "COMMA"):
                self.advance()
                continue

            member_name = self.current_token.value
            self.advance()

            member_value = None
            self.skip_newlines()
            if self.match("ASSIGN"):
                self.advance()
                self.skip_newlines()
                member_value = self.parse_expression()

            members.append((member_name, member_value))
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()

        return members

    def parse_struct_members(self, record_name=None):
        members = []
        skip_member = getattr(self, "skip_un" + "supported_struct_member")
        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            if self.is_cpp_attribute_start():
                self.skip_cpp_attributes()
                continue

            if self.match("PUBLIC", "PRIVATE", "PROTECTED"):
                self.advance()
                if self.match("COLON"):
                    self.advance()
                continue

            if self.match("STRUCT"):
                member = self.parse_struct()
                members.extend(member if isinstance(member, list) else [member])
                continue

            if self.match("UNION"):
                member = self.parse_union()
                members.extend(member if isinstance(member, list) else [member])
                continue

            if self.match("CLASS"):
                member = self.parse_class()
                members.extend(member if isinstance(member, list) else [member])
                continue

            if self.match("ENUM"):
                member = self.parse_enum()
                members.extend(member if isinstance(member, list) else [member])
                continue

            if self.is_type_alias_start():
                alias = self.parse_type_alias(preserve_using_declaration=True)
                if alias is not None:
                    members.extend(alias if isinstance(alias, list) else [alias])
                continue

            if self.match("TEMPLATE"):
                saved_pos = self.pos
                try:
                    member = self.parse_template_prefixed_declaration()
                    if member:
                        members.extend(member if isinstance(member, list) else [member])
                    continue
                except Exception:
                    self.pos = saved_pos
                    self.current_token = self.tokens[self.pos]
                    skip_member()
                    continue

            member_qualifiers = self.parse_class_member_function_specifier_prefix(
                record_name
            )

            if self.is_conversion_operator_declaration():
                members.append(self.parse_conversion_operator_marker(member_qualifiers))
                continue

            if record_name and self.is_class_constructor_declaration(record_name):
                members.append(
                    self.parse_class_constructor(record_name, member_qualifiers)
                )
                continue

            if self.is_function_declaration():
                member_function = self.parse_simple_function()
                member_function.qualifiers = [
                    *member_qualifiers,
                    *member_function.qualifiers,
                ]
                members.append(member_function)
                continue

            if member_qualifiers:
                skip_member()
                continue

            saved_pos = self.pos
            try:
                declarations = self.parse_variable_declaration_list(
                    consume_semicolon=True
                )
                members.extend(declarations)
            except Exception:
                self.pos = saved_pos
                self.current_token = self.tokens[self.pos]
                self.skip_unsupported_struct_member()

        return members

    def skip_unsupported_struct_member(self):
        paren_depth = 0
        bracket_depth = 0

        while self.current_token:
            if self.match("LPAREN"):
                paren_depth += 1
                self.advance()
                continue
            if self.match("RPAREN"):
                paren_depth = max(0, paren_depth - 1)
                self.advance()
                continue
            if self.match("LBRACKET"):
                bracket_depth += 1
                self.advance()
                continue
            if self.match("RBRACKET"):
                bracket_depth = max(0, bracket_depth - 1)
                self.advance()
                continue
            if self.match("LBRACE") and paren_depth == 0 and bracket_depth == 0:
                self.skip_balanced_brace_block()
                return
            if self.match("SEMICOLON") and paren_depth == 0 and bracket_depth == 0:
                self.advance()
                return
            if self.match("RBRACE") and paren_depth == 0 and bracket_depth == 0:
                return
            self.advance()

    def skip_balanced_brace_block(self):
        depth = 0
        while self.current_token:
            if self.match("LBRACE"):
                depth += 1
            elif self.match("RBRACE"):
                depth -= 1

            self.advance()
            if depth == 0:
                return

    def parse_class(self):
        self.consume("CLASS")
        self.skip_newlines()
        self.skip_cpp_attributes()
        attributes = self.parse_type_attribute_prefixes()
        self.skip_newlines()

        name = self.consume("IDENTIFIER").value
        self.type_aliases.add(name)
        if self.match("LT"):
            name += self.parse_template_suffix()

        members = []
        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        self.skip_newlines()
        is_final = self.consume_record_final_specifier()
        base_classes = self.parse_class_inheritance_clause()
        self.skip_newlines()
        has_body = self.match("LBRACE")
        if has_body:
            self.consume("LBRACE")
            members = self.parse_struct_members(name)
            self.consume("RBRACE")

        self.skip_newlines()
        attributes.extend(self.parse_type_attribute_prefixes())
        class_node = StructNode(
            name, members, self.current_record_layout_attributes(attributes)
        )
        class_node.is_class = True
        class_node.is_final = is_final
        class_node.has_body = has_body
        class_node.base_classes = base_classes
        self.skip_newlines()
        declarations = self.parse_record_trailing_declarators("class", name)
        if declarations:
            return [class_node, *declarations] if has_body else declarations

        if self.match("SEMICOLON"):
            self.advance()

        return class_node

    def consume_record_final_specifier(self):
        if self.match("FINAL") or (
            self.match("IDENTIFIER") and self.current_token.value == "final"
        ):
            self.advance()
            self.skip_newlines()
            return True
        return False

    def parse_class_inheritance_clause(self):
        if not self.match("COLON"):
            return []

        self.advance()
        base_classes = []
        while self.current_token and not self.match("LBRACE", "SEMICOLON"):
            self.skip_newlines()
            access = None
            is_virtual = False
            while self.match("PUBLIC", "PRIVATE", "PROTECTED", "VIRTUAL"):
                if self.match("VIRTUAL"):
                    is_virtual = True
                else:
                    access = self.current_token.value
                self.advance()
                self.skip_newlines()

            base_type = self.parse_type()
            base_classes.append(
                {
                    "type": base_type,
                    "access": access,
                    "is_virtual": is_virtual,
                }
            )
            self.skip_newlines()
            if not self.match("COMMA"):
                break
            self.advance()

        return base_classes

    def is_class_constructor_declaration(self, class_name):
        return self.is_class_constructor_declaration_at_pos(self.pos, class_name)

    def is_class_constructor_declaration_at_pos(self, index, class_name):
        index = self.skip_newlines_at_pos(index)
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        if token.type == "TILDE":
            return (
                index + 2 < len(self.tokens)
                and self.tokens[index + 1].type == "IDENTIFIER"
                and self.tokens[index + 1].value == class_name
                and self.tokens[index + 2].type == "LPAREN"
            )

        return (
            token.type == "IDENTIFIER"
            and token.value == class_name
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "LPAREN"
        )

    def parse_class_constructor(self, class_name, qualifiers=None):
        qualifiers = qualifiers or []
        destructor = False
        if self.match("TILDE"):
            destructor = True
            self.advance()

        self.consume("IDENTIFIER")
        function_name = f"~{class_name}" if destructor else class_name

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        self.skip_post_function_qualifiers()
        self.skip_constructor_initializer_list()

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        constructor = FunctionNode("", function_name, params, body, list(qualifiers))
        constructor.record_owner = class_name
        constructor.is_record_constructor = not destructor
        constructor.is_record_destructor = destructor
        return constructor

    def is_conversion_operator_declaration(self):
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value == "operator"
            and self.peek_non_newline()
        ):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        if (
            type_start >= len(self.tokens)
            or self.tokens[type_start].type in self.OVERLOADABLE_OPERATOR_TOKENS
        ):
            return False

        type_end = self.skip_type_at_pos(
            type_start, allow_unknown_identifier_pointers=True
        )
        if type_end is None:
            return False

        type_end = self.skip_newlines_at_pos(type_end)
        return type_end < len(self.tokens) and self.tokens[type_end].type == "LPAREN"

    def parse_conversion_operator_marker(self, qualifiers=None):
        self.consume("IDENTIFIER")
        self.skip_newlines()
        target_type = self.parse_type()
        marker = FunctionNode(
            target_type,
            f"operator {target_type}",
            [],
            None,
            list(qualifiers or []),
        )
        marker.is_conversion_operator = True
        marker.conversion_target_type = target_type
        self.skip_unsupported_struct_member()
        return marker

    def parse_struct_member(self):
        saved_pos = self.pos
        try:
            member_type = self.parse_type()
            name = self.consume_declarator_name()
            member_type += self.parse_array_suffix()

            if self.match("SEMICOLON"):
                self.advance()

            return VariableNode(member_type, name)
        except Exception:
            self.pos = saved_pos
            self.current_token = self.tokens[self.pos]
            self.skip_unsupported_struct_member()
            return None
