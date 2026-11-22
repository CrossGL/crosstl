"""Template-prefixed declarations, explicit instantiations, typedef and using aliases, C++ attributes and namespaces in the HIP parser."""

from ...CUDA.function_templates import parse_template_parameters
from ..HipAst import EnumNode, FunctionNode, StructNode, TypeAliasNode


class HipAliasMixin:
    """Alias members of ``HipParser``."""

    def parse_template_prefixed_declaration(self):
        self.consume("TEMPLATE")
        self.skip_newlines()
        if not self.match("LT"):
            return self.parse_explicit_template_instantiation()

        parameters = parse_template_parameters(self.parse_template_suffix())
        declaration = self.parse_template_prefixed_declaration_body()
        if isinstance(declaration, FunctionNode) and parameters:
            declaration.template_parameters = parameters
        return declaration

    def parse_template_prefixed_declaration_body(self):
        """The declaration after a ``template <...>`` header."""
        self.skip_newlines()
        self.skip_requires_clause()
        self.skip_newlines()
        self.skip_cpp_attributes()

        if self.is_leading_type_attribute_qualified_function():
            return self.parse_function_with_qualifier()

        if (
            self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
                "__LAUNCH_BOUNDS__",
            )
            or self.is_identifier_function_specifier_token()
        ):
            if self.function_name_index_at(self.pos) is None:
                declarations = self.parse_variable_declaration_list()
                return declarations if len(declarations) > 1 else declarations[0]
            return self.parse_function_with_qualifier()
        if self.match("STRUCT"):
            return self.parse_struct()
        if self.match("UNION"):
            return self.parse_union()
        if self.match("ENUM"):
            return self.parse_enum()
        if self.match("CLASS"):
            return self.parse_class()
        if self.is_type_alias_start():
            return self.parse_type_alias()
        if self.is_function_declaration():
            return self.parse_simple_function()
        if self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]

        token_type = (
            self.current_token.type
            if self.current_token is not None
            else "end of input"
        )
        raise self.syntax_error(
            f"Expected declaration after template prefix, got {token_type}"
        )

    def parse_explicit_template_instantiation(self):
        qualifiers = ["template"]
        attributes = []

        while (
            self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
            )
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        return_type = self.parse_type()
        self.skip_newlines()
        name = self.consume_function_name()
        if self.match("LT"):
            name += self.parse_template_suffix()

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        if self.match("SEMICOLON"):
            self.advance()

        self.user_function_names.add(name)
        return FunctionNode(return_type, name, params, None, qualifiers, attributes)

    def is_identifier_value(self, value):
        return self.match("IDENTIFIER") and self.current_token.value == value

    def is_type_alias_start(self):
        return self.match("TYPEDEF") or self.is_identifier_value("using")

    def parse_type_alias(self, preserve_using_declaration=False):
        if self.match("TYPEDEF"):
            return self.parse_typedef_alias()
        return self.parse_using_alias(preserve_using_declaration)

    def parse_typedef_alias(self):
        self.consume("TYPEDEF")
        if self.match("STRUCT"):
            return self.parse_typedef_struct_alias()
        if self.match("ENUM"):
            return self.parse_typedef_enum_alias()

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        aliases = [self.parse_type_alias_declarator(first_type, allow_prefix=False)]

        while self.match("COMMA"):
            self.advance()
            aliases.append(
                self.parse_type_alias_declarator(base_type, allow_prefix=True)
            )

        if self.match("SEMICOLON"):
            self.advance()
        return aliases

    def parse_type_alias_declarator(self, base_type, allow_prefix):
        alias_type = base_type
        if allow_prefix:
            alias_type = self.parse_declarator_prefix(alias_type)

        function_pointer_name = self.parse_function_pointer_parameter_declarator()
        if function_pointer_name is not None:
            alias_type += " (*)"
            self.type_aliases.add(function_pointer_name)
            return TypeAliasNode(alias_type, function_pointer_name)

        name = self.consume("IDENTIFIER").value
        self.skip_newlines()
        if self.match("LPAREN"):
            self.consume_function_type_parameter_list()
            self.type_aliases.add(name)
            return TypeAliasNode(alias_type, name)

        alias_type += self.parse_array_suffix()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def consume_function_type_parameter_list(self):
        self.consume("LPAREN")
        self.parse_parameter_list()
        self.consume("RPAREN")

    def parse_typedef_struct_alias(self):
        self.consume("STRUCT")
        self.skip_newlines()
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()
        self.skip_newlines()

        tag_name = None
        if self.match("IDENTIFIER"):
            tag_name = self.current_token.value
            self.advance()

        self.skip_newlines()
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()
        self.skip_newlines()
        if not self.match("LBRACE"):
            alias = self.parse_type_alias_declarator(
                f"struct {tag_name}", allow_prefix=True
            )
            if self.match("SEMICOLON"):
                self.advance()
            return alias

        self.consume("LBRACE")
        members = self.parse_struct_members(tag_name)
        self.consume("RBRACE")

        alias_base_type = f"struct {tag_name}" if tag_name else "struct <anonymous>"
        aliases = []
        self.skip_newlines()
        while self.current_token and not self.match("SEMICOLON"):
            aliases.append(
                self.parse_type_alias_declarator(alias_base_type, allow_prefix=True)
            )
            self.skip_newlines()
            if not self.match("COMMA"):
                break
            self.advance()
            self.skip_newlines()

        alias_name = aliases[0].name if aliases else tag_name
        if not alias_name:
            self.error("Expected typedef struct alias name")

        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(alias_name)

        extra_aliases = []
        for alias in aliases[1:]:
            alias.alias_type = self.replace_typedef_record_base_type(
                alias.alias_type, alias_base_type, alias_name
            )
            extra_aliases.append(alias)

        struct_node = StructNode(alias_name, members)
        return [struct_node, *extra_aliases] if extra_aliases else struct_node

    def replace_typedef_record_base_type(self, alias_type, original_base, alias_name):
        if not isinstance(alias_type, str):
            return alias_type
        if alias_type.startswith(original_base):
            return alias_name + alias_type[len(original_base) :]
        return alias_type

    def parse_typedef_enum_alias(self):
        self.consume("ENUM")
        is_scoped = False
        self.skip_newlines()

        if self.match("CLASS", "STRUCT"):
            is_scoped = True
            self.advance()
            self.skip_newlines()

        tag_name = None
        if self.match("IDENTIFIER"):
            tag_name = self.current_token.value
            self.type_aliases.add(tag_name)
            self.advance()

        underlying_type = None
        self.skip_newlines()
        if self.match("COLON"):
            self.advance()
            self.skip_newlines()
            underlying_type = self.parse_type()

        self.skip_newlines()
        if not self.match("LBRACE"):
            alias = self.parse_type_alias_declarator(
                f"enum {tag_name}", allow_prefix=True
            )
            if self.match("SEMICOLON"):
                self.advance()
            return alias

        self.consume("LBRACE")
        members = self.parse_enum_members()
        self.consume("RBRACE")

        alias_name = tag_name
        self.skip_newlines()
        if self.match("IDENTIFIER"):
            alias_name = self.current_token.value
            self.advance()
        if not alias_name:
            self.error("Expected typedef enum alias name")

        if self.match("SEMICOLON"):
            self.advance()

        self.type_aliases.add(alias_name)
        enum_node = EnumNode(alias_name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        enum_node.enum_tag = tag_name
        return enum_node

    def parse_using_alias(self, preserve_declaration=False):
        self.advance()
        if self.match("NAMESPACE"):
            self.skip_until_semicolon()
            return None
        if self.match("SCOPE"):
            self.skip_until_semicolon()
            return None

        if not self.match("IDENTIFIER"):
            self.skip_until_semicolon()
            return None

        name = self.consume("IDENTIFIER").value
        self.parse_type_attribute_prefixes()
        if not self.match("ASSIGN"):
            if not preserve_declaration:
                self.skip_until_semicolon()
                return None
            declaration_tokens = [name]
            while self.current_token and not self.match("SEMICOLON"):
                if not self.match("NEWLINE"):
                    declaration_tokens.append(self.current_token.value)
                self.advance()
            if self.match("SEMICOLON"):
                self.advance()

            declaration_name = "".join(declaration_tokens)
            marker = FunctionNode("", declaration_name, [], None)
            marker.is_using_declaration = True
            parts = [part for part in declaration_name.split("::") if part]
            marker.is_inherited_constructor = bool(
                len(parts) >= 2 and parts[-1] == parts[-2]
            )
            return marker

        self.consume("ASSIGN")
        self.skip_newlines()
        attributes = self.parse_type_attribute_prefixes()
        alias_type = self.parse_type()
        if attributes:
            alias_type = " ".join([*attributes, alias_type])
        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_type_attribute_prefixes(self):
        attributes = []

        while (
            self.match("IDENTIFIER")
            and self.current_token.value in self.TYPE_ATTRIBUTE_IDENTIFIERS
        ):
            attribute_name = self.current_token.value
            self.advance()
            attribute_args = ""
            if self.match("LPAREN"):
                attribute_args = "".join(
                    self.consume_balanced_token_values("LPAREN", "RPAREN")
                )
            attributes.append(f"{attribute_name}{attribute_args}")
            self.skip_newlines()

        return attributes

    def is_cpp_attribute_start(self):
        return (
            self.match("LBRACKET")
            and self.peek() is not None
            and self.peek().type == "LBRACKET"
        )

    def skip_cpp_attributes(self):
        while self.is_cpp_attribute_start():
            self.advance()
            self.advance()
            depth = 1

            while self.current_token and depth > 0:
                if self.is_cpp_attribute_start():
                    self.advance()
                    self.advance()
                    depth += 1
                    continue
                if (
                    self.match("RBRACKET")
                    and self.peek() is not None
                    and self.peek().type == "RBRACKET"
                ):
                    self.advance()
                    self.advance()
                    depth -= 1
                    continue
                self.advance()

            self.skip_newlines()

    def consume_balanced_token_values(self, open_token, close_token):
        values = []
        depth = 0

        while self.current_token:
            token = self.current_token
            values.append(token.value)

            if token.type == open_token:
                depth += 1
            elif token.type == close_token:
                depth -= 1
                self.advance()
                if depth == 0:
                    return values
                continue

            self.advance()

        self.error(f"Unterminated {open_token}")

    def parse_namespace_block(self):
        self.consume("NAMESPACE")
        self.skip_newlines()

        name_parts = []
        while self.match("IDENTIFIER", "SCOPE"):
            name_parts.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        if self.match("ASSIGN"):
            self.consume("ASSIGN")
            target_parts = []
            while self.current_token and not self.match("SEMICOLON"):
                target_parts.append(self.current_token.value)
                self.advance()
            alias = "".join(name_parts).strip()
            target = "".join(target_parts).strip()
            if alias and target and self.namespace_depth == 0 and self.block_depth == 0:
                self.namespace_aliases[alias] = target
            if self.match("SEMICOLON"):
                self.advance()
            return None

        self.consume("LBRACE")
        self.namespace_depth += 1
        statements = []
        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    statements.extend(stmt)
                else:
                    statements.append(stmt)

        self.consume("RBRACE")
        self.namespace_depth -= 1
        if self.match("SEMICOLON"):
            self.advance()
        return statements

    def skip_until_semicolon(self):
        while self.current_token and not self.match("SEMICOLON"):
            self.advance()
        if self.match("SEMICOLON"):
            self.advance()
