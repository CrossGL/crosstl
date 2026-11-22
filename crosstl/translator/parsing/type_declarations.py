"""Type declarations in the CrossGL parser: structs and their members, enums, constant buffers, generic declarations and traits."""

from copy import deepcopy

from ..ast import (
    AttributeNode,
    EnumNode,
    EnumVariantNode,
    NamedType,
    StructMemberNode,
    StructNode,
    VariableNode,
)
from ..refusals import CrossGLSyntaxError
from ..static_assertion_syntax import mark_template_assertions


class ParserTypeDeclarationMixin:
    """Type declaration members of ``Parser``."""

    def parse_struct(self, leading_attributes=None):
        """Parse a struct declaration and its member list."""
        if self.current_token[0] == "EOF":
            return None

        attributes = list(leading_attributes or [])

        self.eat("STRUCT")

        if self.current_token[0] != "IDENTIFIER":
            raise CrossGLSyntaxError(
                "Expected a structure name after 'struct', got "
                f"{self.current_token[0]} {self.current_token[1]!r}"
            )

        name = self.parse_binding_identifier()

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        attributes.extend(self.parse_attribute_annotations(allow_single_square=False))

        if self.current_token[0] != "LBRACE":
            if self.current_token[0] == "SEMICOLON" and not attributes:
                # ``struct S;`` declares the name only; the definition carries the members.
                self.eat("SEMICOLON")
                return None
            self.reject_unexpected_token(f"structure {name!r} before its '{{'")

        self.eat("LBRACE")
        members = []
        methods = []
        first_assertion = len(self.static_assertions)

        previous_scope = self.enter_generic_parameter_scope(generic_params)
        try:
            while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
                if self.is_function_declaration():
                    # A member function reads its receiver's members by name.
                    static_member = self.current_token[0] == "STATIC"
                    method = self.parse_function()
                    if static_member and "static" not in method.qualifiers:
                        method.qualifiers.append("static")
                    methods.append(method)
                    continue
                member = self.parse_struct_member()
                self.append_parsed_nodes(members, member)
        finally:
            self.restore_generic_parameter_scope(previous_scope)
        mark_template_assertions(
            self.static_assertions[first_assertion:], name, generic_params
        )

        self.reject_end_of_input(f"structure {name!r}")
        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return StructNode(
            name=name,
            members=members,
            generic_params=generic_params,
            attributes=attributes,
            methods=methods,
        )

    def parse_struct_member(self):
        """Parse one struct member declaration."""
        if self.current_token[0] == "EOF":
            return None

        if self.parse_scope_static_assertion():
            return None

        if self.current_token[0] == "PREPROCESSOR":
            return self.parse_preprocessor_directive()

        if self.current_token[0] == "PRECISION":
            return self.parse_precision_statement()

        if self.current_token[0] == "ENUM":
            return self.parse_enum()

        if self.current_token[0] == "SEMICOLON":
            # An empty member declaration, which C++ [class.mem] allows.
            self.eat("SEMICOLON")
            return None

        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] in [
            "generic",
            "trait",
        ]:
            raise CrossGLSyntaxError(
                f"CrossGL cannot declare a {self.current_token[1]} inside a structure"
            )

        leading_attributes = []
        while True:
            if self.current_token[0] == "LAYOUT":
                leading_attributes.extend(self.parse_layout_attributes())
                continue

            attributes = self.parse_attribute_annotations()
            if attributes:
                leading_attributes.extend(attributes)
                continue

            break

        if self.current_token_is_binding_identifier() and self.peek()[0] == "COLON":
            name = self.parse_binding_identifier()
            self.eat("COLON")
            member_type = self.parse_type()

            attributes = list(leading_attributes)
            attributes.extend(self.parse_post_declaration_attributes())

            if self.current_token[0] not in ["SEMICOLON", "COMMA", "RBRACE"]:
                self.reject_unexpected_token(f"structure member {name!r}")

            if self.current_token[0] in ["SEMICOLON", "COMMA"]:
                self.eat(self.current_token[0])
            return StructMemberNode(
                name=name, member_type=member_type, attributes=attributes
            )

        qualifier_attributes = []
        resource_qualifiers = []
        declaration_qualifiers = []
        if self.current_token_is_variable_qualifier():
            parsed_qualifiers = self.parse_variable_qualifiers()
            declaration_qualifiers, resource_qualifiers = (
                self.partition_resource_qualifiers(parsed_qualifiers)
            )
            qualifier_attributes = [
                AttributeNode(name=qualifier) for qualifier in declaration_qualifiers
            ]

        member_type = self.parse_type()
        member_type = self.parse_array_suffixes(member_type)

        if not self.current_token_is_binding_identifier():
            self.reject_unexpected_token(
                "a structure member declaration, before its name"
            )

        members = []
        while True:
            if not self.current_token_is_binding_identifier():
                self.reject_unexpected_token(
                    "a structure member declaration, before its name"
                )

            name = self.parse_binding_identifier()
            declarator_type = self.parse_array_suffixes(deepcopy(member_type))
            declaration_resource_qualifiers = self.apply_pointer_resource_contract(
                declarator_type, declaration_qualifiers, resource_qualifiers
            )

            attributes = list(leading_attributes)
            attributes.extend(qualifier_attributes)
            attributes.extend(self.parse_post_declaration_attributes())

            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()

            member = StructMemberNode(
                name=name,
                member_type=declarator_type,
                default_value=default_value,
                attributes=attributes,
                resource_qualifiers=declaration_resource_qualifiers,
            )

            if self.current_token[0] == "COMMA":
                members.append(member)
                self.eat("COMMA")
                continue

            if self.current_token[0] == "SEMICOLON":
                members.append(member)
                self.eat("SEMICOLON")
                return members[0] if len(members) == 1 else members

            if self.current_token[0] == "LPAREN":
                # Skipping to the next ';' would drop a member function silently.
                raise CrossGLSyntaxError(
                    f"Struct member '{name}' is declared like a function that CrossGL "
                    "cannot read as a member function"
                )
            self.reject_unexpected_token(f"structure member {name!r}")

    def parse_enum(self):
        """Parse an enum declaration and its variants."""
        self.eat("ENUM")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        self.eat("LBRACE")
        variants = []

        while self.current_token[0] != "RBRACE":
            variant = self.parse_enum_variant()
            variants.append(variant)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RBRACE")

        return EnumNode(name=name, variants=variants, underlying_type=underlying_type)

    def parse_enum_variant(self):
        """Parse one enum variant, including tuple or struct payloads."""
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        value = None
        variant_data = None

        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            # Parse variant data/parameters
            variant_params = []
            while self.current_token[0] != "RPAREN":
                if self.current_token[0] == "IDENTIFIER":
                    # Type parameter
                    param_type = self.parse_type()
                    variant_params.append(param_type)
                else:
                    # Parse as expression
                    param_expr = self.parse_expression()
                    variant_params.append(param_expr)

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.current_token[0] != "RPAREN":
                    break
            self.eat("RPAREN")
            variant_data = variant_params

        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

        elif self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            struct_members = []
            while self.current_token[0] != "RBRACE":
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.peek()[0] == "COLON":
                    member_name = self.current_token[1]
                    self.eat(self.current_token[0])
                    self.eat("COLON")
                    member_type = self.parse_type()
                    struct_members.append((member_name, member_type))

                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                elif self.current_token[0] != "RBRACE":
                    self.reject_unexpected_token(f"enum variant {name!r}")
                else:
                    break
            self.eat("RBRACE")
            variant_data = struct_members

        variant_node = EnumVariantNode(name=name, value=value)
        if variant_data:
            variant_node.data = variant_data

        return variant_node

    def parse_cbuffer_as_struct(self, leading_attributes=None):
        """Parse a constant/uniform buffer declaration as a struct node."""
        attributes = list(leading_attributes or [])

        if self.current_token[0] == "CBUFFER":
            self.eat("CBUFFER")
        else:
            self.eat("UNIFORM")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        attributes.extend(self.parse_post_declaration_attributes())

        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            self.reject_end_of_input(f"cbuffer {name!r}")
            member = self.parse_struct_member()
            self.append_parsed_nodes(members, member)

        self.eat("RBRACE")

        if self.current_token_is_declaration_binding_identifier() and self.peek()[
            0
        ] in (
            "SEMICOLON",
            "LBRACKET",
            "AT",
            "ATTRIBUTE",
        ):
            # Without ';' or '[' next, the name is the type of the next declaration.
            return self.parse_constant_block_instance(name, members, attributes)

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        node = StructNode(name=name, members=members, attributes=attributes)
        node.is_cbuffer = True
        return node

    def parse_constant_block_instance(self, block_name, members, attributes):
        """Desugar ``cbuffer Block { ... } instance[N];`` into ``struct Block`` and ``ConstantBuffer<Block>``.

        An instance name scopes the block's members, which are then read as ``instance.member`` (GLSL's named
        uniform blocks and arrays of them). The block becomes a plain struct and the instance a uniform
        ``ConstantBuffer`` of it that carries the block's attributes, the spelling every generator lowers.
        """
        instance_name = self.parse_binding_identifier()
        sizes = []
        while self.current_token[
            0
        ] == "LBRACKET" and not self.current_token_starts_square_attribute(
            allow_single_square=False
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            sizes.append(size)
        attributes.extend(self.parse_post_declaration_attributes())
        self.eat("SEMICOLON")

        struct_node = StructNode(name=block_name, members=members)
        variable_node = VariableNode(
            name=instance_name,
            var_type=self.wrap_array_dimensions(
                NamedType("ConstantBuffer", [NamedType(block_name)]), sizes
            ),
            qualifiers=["uniform"],
            attributes=attributes,
        )
        return [struct_node, variable_node]

    def append_constant_block(self, parsed, cbuffers, structs, variables):
        """Route a parsed constant block, or the struct and variable of an instance-named one."""
        if isinstance(parsed, list):
            struct_node, variable_node = parsed
            structs.append(struct_node)
            variables.append(variable_node)
        elif parsed:
            cbuffers.append(parsed)

    # Helper methods
    def is_cbuffer_declaration(self):
        """Return whether the current token begins a cbuffer declaration."""
        if self.current_token[0] == "CBUFFER":
            return True
        if self.current_token[0] != "UNIFORM" or self.peek()[0] != "IDENTIFIER":
            return False

        offset = 2
        while self.peek(offset)[0] in ["AT", "ATTRIBUTE"]:
            if self.peek(offset)[0] == "ATTRIBUTE":
                offset += 1
            else:
                if self.peek(offset + 1)[0] != "IDENTIFIER":
                    return False
                offset += 2

            if self.peek(offset)[0] == "LPAREN":
                depth = 1
                offset += 1
                while depth and self.peek(offset)[0] != "EOF":
                    if self.peek(offset)[0] == "LPAREN":
                        depth += 1
                    elif self.peek(offset)[0] == "RPAREN":
                        depth -= 1
                    offset += 1

        return self.peek(offset)[0] == "LBRACE"

    def parse_generic_declaration(self):
        """Parse a ``generic<...>`` wrapper around a struct, enum, function or trait.

        A malformed wrapper raises ``SyntaxError``; skipping it would drop the declaration from the program.
        """
        self.eat("IDENTIFIER")

        if self.current_token[0] != "LESS_THAN":
            raise CrossGLSyntaxError(
                f"Expected '<' after 'generic', got {self.current_token[1]!r}"
            )

        generic_params = self.parse_generic_parameters()

        self.reject_end_of_input("generic declaration")
        previous_scope = self.enter_generic_parameter_scope(generic_params)
        try:
            if self.current_token[0] == "STRUCT":
                first_assertion = len(self.static_assertions)
                struct_node = self.parse_struct()
                if struct_node:
                    struct_node.generic_params = generic_params
                    mark_template_assertions(
                        self.static_assertions[first_assertion:],
                        struct_node.name,
                        generic_params,
                    )
                return struct_node
            if self.current_token[0] == "ENUM":
                enum_node = self.parse_enum()
                if enum_node:
                    enum_node.generic_params = generic_params
                return enum_node
            if self.current_token[0] == "FUNCTION" or self.is_function_declaration():
                func_node = self.parse_function()
                if func_node:
                    func_node.generic_params = generic_params
                return func_node
            if self.current_token[0] == "TRAIT":
                trait_node = self.parse_trait()
                if trait_node:
                    trait_node.generic_params = generic_params
                return trait_node
            raise CrossGLSyntaxError(
                "Expected a struct, enum, function or trait after the generic parameters, got "
                f"{self.current_token[1]!r}"
            )
        finally:
            self.restore_generic_parameter_scope(previous_scope)

    def parse_trait(self):
        """Parse a trait-like declaration into the current AST shape."""
        if self.current_token[0] == "EOF":
            return None

        self.eat("TRAIT")

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        generic_params = []
        if self.current_token[0] == "LESS_THAN":
            generic_params = self.parse_generic_parameters()

        self.eat("LBRACE")

        methods = []
        previous_scope = self.enter_generic_parameter_scope(generic_params)
        try:
            while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
                if self.is_function_declaration():
                    func = self.parse_function()
                    if func:
                        methods.append(func)
                else:
                    self.reject_unexpected_token(f"trait {name!r}")
        finally:
            self.restore_generic_parameter_scope(previous_scope)

        self.reject_end_of_input(f"trait {name!r}")
        self.eat("RBRACE")

        # Traits are represented as marked structs to avoid a broader AST migration.
        trait_node = StructNode(
            name=name, members=methods, attributes=[], generic_params=generic_params
        )
        trait_node.is_trait = True
        return trait_node
