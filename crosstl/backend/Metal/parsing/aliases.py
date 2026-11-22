"""Preprocessor directives, using declarations and aliases, enums, typedefs, decltype types, string literal runs, static assertions and pragmas in the Metal parser."""

from crosstl.translator.refusals import MetalSyntaxError

from ..MetalAst import (
    CallableTypeAliasNode,
    EnumNode,
    PreprocessorNode,
    StaticAssertNode,
    StructNode,
    TypeAliasNode,
)
from ..string_literals import concatenate_string_literals


class MetalAliasParsingMixin:
    """Alias members of ``MetalParser``."""

    def parse_preprocessor_directive(self):
        text = self.current_token[1] or ""
        self.eat("PREPROCESSOR")
        stripped = text.lstrip("#").strip()
        if stripped:
            parts = stripped.split(None, 1)
            directive = f"#{parts[0]}"
            content = parts[1] if len(parts) > 1 else ""
            return PreprocessorNode(directive, content)

        directive = text
        content = ""
        if self.current_token[0] == "LESS_THAN":
            self.eat("LESS_THAN")
            parts = []
            while self.current_token[0] != "GREATER_THAN":
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("GREATER_THAN")
            content = "<" + "".join(parts) + ">"
        elif self.current_token[0] == "STRING":
            content = self.current_token[1]
            self.eat("STRING")
        elif self.current_token[0] not in ["EOF", "PREPROCESSOR"]:
            content = str(self.current_token[1])
            self.eat(self.current_token[0])
        return PreprocessorNode(directive, content)

    def parse_using_statement(self):
        start_token = self.current_token
        self.eat("USING")
        if self.current_token[0] == "NAMESPACE":
            self.eat("NAMESPACE")
            target_name = None
            if self.current_token[0] in {"IDENTIFIER", "METAL"}:
                target_name = self.parse_scoped_identifier()
            self.eat("SEMICOLON")
            if target_name:
                self.using_namespace_directives.append(
                    {
                        "namespace": self.current_namespace(),
                        "target": target_name,
                        "source_location": self.source_span_from_tokens(
                            start_token, self.tokens[self.pos - 1]
                        ),
                        "declaration_context": self.current_declaration_context(),
                    }
                )
            return None
        if self.is_using_declaration_start():
            target_name = self.parse_using_declaration()
            self.using_declarations.append(
                {"namespace": self.current_namespace(), "target": target_name}
            )
            return None
        alias_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("EQUALS")
        if self.is_union_alias_start():
            return self.parse_using_union_alias(alias_name)
        globally_qualified = self.current_token[0] == "SCOPE"
        alias_type, qualifiers = self.parse_type_specifier()
        qualifier_contract = dict(self.last_type_specifier_qualifier_contract)
        if self.current_token[0] == "LPAREN":
            indirection = self.parse_callable_alias_abstract_indirection()
            parameters = self.parse_callable_alias_parameters()
            self.eat("SEMICOLON")
            self.register_known_type(alias_name)
            return self.annotate_declaration_scope(
                CallableTypeAliasNode(
                    alias_type,
                    alias_name,
                    parameters,
                    indirection=indirection,
                    qualifiers=qualifiers,
                    source_location=self.source_span_from_tokens(
                        start_token, self.tokens[self.pos - 1]
                    ),
                )
            )
        self.eat("SEMICOLON")
        self.register_known_type(alias_name)
        alias = TypeAliasNode(
            alias_type,
            alias_name,
            qualifiers=qualifiers,
            source_location=self.source_span_from_tokens(
                start_token, self.tokens[self.pos - 1]
            ),
        )
        alias.pointee_qualifiers = list(qualifier_contract["pointee"])
        # The parser drops a leading ``::`` from type names. Alias-template expansion needs it, because a target
        # such as ``::metal::array<T, N>`` declared in a namespace that contains its own ``metal`` namespace
        # names the global one ([basic.lookup.qual]).
        alias.alias_type_globally_qualified = globally_qualified
        return self.annotate_declaration_scope(alias)

    def is_using_declaration_start(self):
        return self.current_token[0] in {"IDENTIFIER", "METAL", "SCOPE"} and not (
            self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "EQUALS"
        )

    def parse_using_declaration(self):
        target_name = self.parse_scoped_identifier()
        self.eat("SEMICOLON")
        return target_name

    def is_union_alias_start(self):
        return self.current_token == ("IDENTIFIER", "union")

    def parse_using_union_alias(self, alias_name):
        self.eat("IDENTIFIER")
        if self.current_token[0] != "LBRACE":
            raise MetalSyntaxError("Expected anonymous union body in using alias")
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")
        self.eat("SEMICOLON")

        self.register_known_type(alias_name)
        union_node = self.annotate_declaration_scope(StructNode(alias_name, members))
        union_node.aggregate_kind = "union"
        union_node.using_alias = True
        union_node.type_aliases = list(self.last_struct_type_aliases)
        return union_node

    def parse_enum(self):
        start_token = self.current_token
        name, is_scoped, underlying_type = self.parse_enum_header()
        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        enum = self.annotate_declaration_scope(EnumNode(name, members))
        enum.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        enum.is_scoped = is_scoped
        enum.underlying_type = underlying_type
        return enum

    def parse_enum_header(self):
        self.eat("ENUM")
        is_scoped = False
        if self.current_token[0] == "CLASS":
            is_scoped = True
            self.eat("CLASS")
        name = None
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.register_known_type(name)
        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type, _qualifiers = self.parse_type_specifier()
        return name, is_scoped, underlying_type

    def parse_enum_members(self):
        members = []
        while self.current_token[0] != "RBRACE":
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                member_value = self.parse_expression()
            members.append((member_name, member_value))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] == "RBRACE":
                break
            else:
                raise MetalSyntaxError(
                    f"Expected comma or closing brace in enum, got {self.current_token[0]}"
                )
        return members

    def parse_typedef(self):
        start_token = self.current_token
        self.eat("TYPEDEF")
        if self.current_token[0] == "STRUCT":
            declaration = self.parse_typedef_struct()
            if isinstance(declaration, StructNode):
                self.annotate_declaration_scope(declaration)
                declaration.declaration_source_location = self.source_span_from_tokens(
                    start_token, self.tokens[self.pos - 1]
                )
            return declaration
        if self.is_union_alias_start():
            declaration = self.parse_typedef_union()
            if isinstance(declaration, StructNode):
                self.annotate_declaration_scope(declaration)
                declaration.declaration_source_location = self.source_span_from_tokens(
                    start_token, self.tokens[self.pos - 1]
                )
            return declaration
        if self.current_token[0] == "ENUM":
            return self.parse_typedef_enum()
        qualifiers = []
        qualifier_contract = {"pointee": ()}
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "decltype"
        ):
            alias_type = self.parse_decltype_member_type()
        else:
            alias_type, qualifiers = self.parse_type_specifier()
            qualifier_contract = dict(self.last_type_specifier_qualifier_contract)
        if self.current_token[0] == "LPAREN":
            alias_name, indirection = self.parse_function_typedef_declarator()
            if self.current_token[0] != "LPAREN":
                self.eat("SEMICOLON")
                self.register_known_type(alias_name)
                return TypeAliasNode(
                    self.apply_declarator_type_suffix(alias_type, indirection),
                    alias_name,
                    qualifiers=qualifiers,
                    declarator_type_suffix=indirection,
                    declarator_type_suffix_grouped=bool(indirection),
                    source_location=self.source_span_from_tokens(
                        start_token, self.tokens[self.pos - 1]
                    ),
                )
            parameters = self.parse_callable_alias_parameters()
            self.eat("SEMICOLON")
            self.register_known_type(alias_name)
            return CallableTypeAliasNode(
                alias_type,
                alias_name,
                parameters,
                indirection=indirection,
                qualifiers=qualifiers,
                declarator_type_suffix_grouped=bool(indirection),
                source_location=self.source_span_from_tokens(
                    start_token, self.tokens[self.pos - 1]
                ),
            )
        alias_name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
        if self.current_token[0] == "LPAREN":
            parameters = self.parse_callable_alias_parameters()
            self.eat("SEMICOLON")
            self.register_known_type(alias_name)
            return CallableTypeAliasNode(
                alias_type,
                alias_name,
                parameters,
                indirection=type_suffix,
                qualifiers=qualifiers,
                array_sizes=array_sizes,
                declarator_type_suffix_grouped=grouped_suffix,
                source_location=self.source_span_from_tokens(
                    start_token, self.tokens[self.pos - 1]
                ),
            )
        self.eat("SEMICOLON")
        self.register_known_type(alias_name)
        alias = TypeAliasNode(
            alias_type,
            alias_name,
            qualifiers=qualifiers,
            array_sizes=array_sizes,
            declarator_type_suffix=type_suffix,
            declarator_type_suffix_grouped=grouped_suffix,
            source_location=self.source_span_from_tokens(
                start_token, self.tokens[self.pos - 1]
            ),
        )
        alias.pointee_qualifiers = list(qualifier_contract["pointee"])
        return alias

    def parse_typedef_enum(self):
        tag_name, is_scoped, underlying_type = self.parse_enum_header()

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_enum_members()
            self.eat("RBRACE")
            (
                alias_name,
                _array_sizes,
                _type_suffix,
                _grouped_suffix,
            ) = self.parse_declarator()
            self.eat("SEMICOLON")
            enum_name = alias_name or tag_name
            if not enum_name:
                raise MetalSyntaxError("Expected typedef enum name")
            self.register_known_type(enum_name)
            enum = EnumNode(enum_name, members)
            enum.typedef_tag = tag_name
            enum.is_scoped = is_scoped
            enum.underlying_type = underlying_type
            return enum

        if not tag_name:
            raise MetalSyntaxError("Expected typedef enum body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.register_known_type(alias_name)
        return TypeAliasNode(f"enum {tag_name}", alias_name)

    def parse_typedef_union(self):
        self.eat("IDENTIFIER")
        tag_name = None
        if self.is_current_name_token():
            tag_name = self.current_token[1]
            self.eat(self.current_token[0])
            self.register_known_type(tag_name)

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_struct_members()
            nested_structs = list(self.last_struct_nested_aggregates)
            self.eat("RBRACE")
            alias_name, array_sizes, type_suffix, grouped_suffix = (
                self.parse_declarator()
            )
            self.eat("SEMICOLON")
            union_name = alias_name or tag_name
            if not union_name:
                raise MetalSyntaxError("Expected typedef union name")
            self.register_known_type(union_name)
            union_node = StructNode(union_name, members)
            union_node.typedef_tag = tag_name
            union_node.aggregate_kind = "union"
            union_node.nested_structs = nested_structs
            union_node.is_typedef_aggregate = True
            union_node.typedef_array_sizes = array_sizes
            union_node.typedef_declarator_type_suffix = type_suffix
            union_node.typedef_declarator_type_suffix_grouped = grouped_suffix
            return union_node

        if not tag_name:
            raise MetalSyntaxError("Expected typedef union body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.register_known_type(alias_name)
        return TypeAliasNode(f"union {tag_name}", alias_name)

    def parse_function_typedef_declarator(self):
        self.eat("LPAREN")
        indirection = ""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            indirection += self.declarator_pointer_token_suffix(self.current_token[0])
            self.eat(self.current_token[0])
        alias_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("RPAREN")
        return alias_name, indirection

    def parse_callable_alias_abstract_indirection(self):
        if self.current_token[0] != "LPAREN" or self.peek(1)[0] not in {
            "MULTIPLY",
            "BITWISE_AND",
        }:
            return ""

        self.eat("LPAREN")
        indirection = ""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            indirection += self.declarator_pointer_token_suffix(self.current_token[0])
            self.eat(self.current_token[0])
        self.eat("RPAREN")
        return indirection

    def parse_callable_alias_parameters(self):
        self.eat("LPAREN")
        parameters = self.parse_parameters()
        self.eat("RPAREN")
        return parameters

    def parse_parenthesized_parameter_tokens(self):
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError("Unterminated function typedef parameter list")

    def parse_decltype_type(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        parts = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            parts.append(token_value)
            self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError("Unterminated decltype typedef")
        return f"decltype({self.format_generic_type_tokens(parts)})"

    def parse_decltype_member_type(self):
        """``decltype(e)`` optionally followed by a nested member such as ``::elem_type`` ([dcl.type.decltype])."""
        text = self.parse_decltype_type()
        while self.current_token[0] == "SCOPE" and self.peek(1)[0] == "IDENTIFIER":
            self.eat("SCOPE")
            text += f"::{self.current_token[1]}"
            self.eat("IDENTIFIER")
        return text

    def parse_typedef_struct(self):
        self.eat("STRUCT")
        struct_attributes = self.parse_attributes()
        alignas_specs = self.parse_alignas_specifiers()
        tag_name = None
        if self.is_current_name_token():
            tag_name = self.current_token[1]
            self.eat(self.current_token[0])
            self.register_known_type(tag_name)

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_struct_members()
            # Aggregates defined among the members, such as an unnamed union, are declared with the struct.
            nested_structs = list(self.last_struct_nested_aggregates)
            self.eat("RBRACE")
            struct_attributes.extend(self.parse_attributes())
            alias_name, array_sizes, type_suffix, grouped_suffix = (
                self.parse_declarator()
            )
            self.eat("SEMICOLON")
            struct_name = alias_name or tag_name
            if not struct_name:
                raise MetalSyntaxError("Expected typedef struct name")
            self.register_known_type(struct_name)
            struct_node = StructNode(struct_name, members)
            struct_node.typedef_tag = tag_name
            struct_node.nested_structs = nested_structs
            struct_node.is_typedef_aggregate = True
            struct_node.attributes = struct_attributes
            struct_node.alignas = alignas_specs
            struct_node.typedef_array_sizes = array_sizes
            struct_node.typedef_declarator_type_suffix = type_suffix
            struct_node.typedef_declarator_type_suffix_grouped = grouped_suffix
            return struct_node

        if not tag_name:
            raise MetalSyntaxError("Expected typedef struct body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.register_known_type(alias_name)
        return TypeAliasNode(f"struct {tag_name}", alias_name)

    def parse_string_literal_run(self):
        """Parse adjacent string literals as the one literal C++ joins them into."""
        spellings = []
        while self.current_token[0] == "STRING":
            spellings.append(self.current_token[1])
            self.eat("STRING")
        try:
            return concatenate_string_literals(spellings)
        except ValueError as exc:
            raise MetalSyntaxError(
                f"Invalid string literal concatenation: {exc}"
            ) from exc

    def parse_static_assert(self):
        self.eat("STATIC_ASSERT")
        self.eat("LPAREN")
        condition = self.parse_expression()
        message = None
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] == "STRING":
                message = self.parse_string_literal_run()
            else:
                message = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return StaticAssertNode(condition, message)

    def parse_pragma_statement(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError("Unterminated _Pragma statement")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
