"""Variable declarations in the CrossGL parser: address-space declarations, declarators, array suffixes, variable qualifiers, declaration lookahead and constants."""

from copy import deepcopy

from ..ast import (
    ArrayType,
    ConstantNode,
    VariableNode,
)
from .vocabulary import (
    PARAMETER_QUALIFIER_TOKEN_TYPES,
    RESOURCE_MEMORY_QUALIFIER_NAMES,
    VARIABLE_QUALIFIER_NAMES,
    VARIABLE_QUALIFIER_TOKEN_TYPES,
)


class ParserVariableDeclarationMixin:
    """Variable declaration members of ``Parser``."""

    def current_token_starts_var_address_space_declaration(self):
        """Return whether the current token starts ``var<address-space>`` syntax."""
        return self.current_token[0] == "VAR" and self.peek()[0] == "LESS_THAN"

    def parse_var_address_space_qualifiers(self):
        """Parse the address-space qualifier list in ``var<...>`` declarations."""
        qualifiers = []
        self.eat("VAR")
        self.eat("LESS_THAN")

        while (
            not self.current_token_is_type_greater_than()
            and self.current_token[0] != "EOF"
        ):
            token_type, token_value = self.current_token
            if (
                token_type != "COMMA"
                and isinstance(token_value, str)
                and token_value.isidentifier()
            ):
                qualifiers.append(token_value.lower())
            self.eat(token_type)

        self.eat_type_greater_than()
        return qualifiers

    def parse_var_address_space_declaration(self, attributes):
        """Parse ``var<workgroup> name: Type`` resource/shared declarations."""
        qualifiers = self.parse_var_address_space_qualifiers()
        name = self.parse_binding_identifier()
        self.eat("COLON")
        var_type = self.parse_resource_parameter_type()
        sizes = self.parse_array_suffix_sizes()

        attributes.extend(self.parse_post_declaration_attributes())
        sizes += self.parse_array_suffix_sizes()
        var_type = self.wrap_array_dimensions(var_type, sizes)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        self.eat("SEMICOLON")

        node = VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=qualifiers,
            attributes=attributes,
            is_mutable="const" not in qualifiers,
        )
        node.is_var_address_space = True
        node.address_space_qualifiers = list(qualifiers)
        return self.finish_variable_contract(node)

    def parse_variable_declaration(self, leading_attributes=None):
        """Parse a variable declaration, including qualifiers and attributes."""
        attributes = list(leading_attributes or [])
        attributes.extend(self.parse_attribute_annotations())
        if self.current_token[0] == "LET":
            # ``@ deprecated("...") let pi = 3.14;``: annotations apply to a ``let`` binding as well.
            return self.parse_let_declaration(attributes)
        is_type_alias = self.current_token_starts_type_alias_declaration()
        if is_type_alias:
            self.eat("IDENTIFIER")

        if self.current_token_starts_var_address_space_declaration():
            return self.parse_var_address_space_declaration(attributes)

        is_colon_style_var = (
            self.current_token[0] == "VAR"
            and self.peek()[0] != "LESS_THAN"
            and self.peek(2)[0] in {"COLON", "EQUALS"}
        )

        parsed_qualifiers = (
            []
            if is_colon_style_var
            else self.parse_variable_qualifiers(layout_attributes=attributes)
        )
        qualifiers, resource_qualifiers = self.partition_resource_qualifiers(
            parsed_qualifiers
        )

        if is_colon_style_var:
            self.eat("VAR")
            name = self.parse_binding_identifier()
            var_type = None
            sizes = []
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                var_type = self.parse_type()
                sizes = self.parse_array_suffix_sizes()

            attributes.extend(self.parse_post_declaration_attributes())
            if var_type is not None:
                sizes += self.parse_array_suffix_sizes()
                var_type = self.wrap_array_dimensions(var_type, sizes)

            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression()

            self.eat("SEMICOLON")

            return self.finish_variable_contract(
                VariableNode(
                    name=name,
                    var_type=var_type,
                    initial_value=initial_value,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    is_mutable="const" not in qualifiers,
                    is_type_alias=is_type_alias,
                )
            )

        else:
            var_type = self.parse_type()
            declarations = []
            while True:
                declarations.append(
                    self.parse_variable_declarator(
                        var_type,
                        qualifiers,
                        attributes,
                        resource_qualifiers,
                        is_type_alias=is_type_alias,
                    )
                )
                if self.current_token[0] != "COMMA":
                    break
                self.eat("COMMA")

            self.eat("SEMICOLON")
            return declarations[0] if len(declarations) == 1 else declarations

    def parse_variable_declarator(
        self,
        base_type,
        qualifiers,
        declaration_attributes,
        resource_qualifiers=None,
        *,
        is_type_alias=False,
    ):
        """Parse one declarator after a shared C-style variable type."""
        var_type = self.parse_array_suffixes(deepcopy(base_type))
        name = self.parse_binding_identifier()
        generic_params = self.parse_declaration_square_generic_parameters()
        sizes = self.parse_array_suffix_sizes()

        attributes = list(declaration_attributes)
        attributes.extend(self.parse_post_declaration_attributes())
        sizes += self.parse_array_suffix_sizes()
        var_type = self.wrap_array_dimensions(var_type, sizes)

        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()

        declaration_resource_qualifiers = self.apply_pointer_resource_contract(
            var_type, qualifiers, resource_qualifiers or []
        )
        variable = VariableNode(
            name=name,
            var_type=var_type,
            initial_value=initial_value,
            qualifiers=list(qualifiers),
            resource_qualifiers=declaration_resource_qualifiers,
            attributes=attributes,
            is_mutable="const" not in qualifiers,
            is_type_alias=is_type_alias,
        )
        variable.generic_params = generic_params
        return self.finish_variable_contract(variable)

    def parse_declaration_square_generic_parameters(self):
        """Parse declaration-level square generic parameters after a binding name."""
        if (
            self.current_token[0] == "LBRACKET"
            and not self.current_token_starts_square_attribute(
                allow_single_square=False
            )
            and self.square_brackets_form_type_arguments()
        ):
            return self.parse_square_generic_arguments()
        return []

    def parse_array_suffix_sizes(self):
        """Parse the sizes of consecutive non-attribute array suffixes, in source order (``None`` when unsized)."""
        sizes = []
        while (
            self.current_token[0] == "LBRACKET"
            and not self.current_token_starts_square_attribute(
                allow_single_square=False
            )
            and not self.square_brackets_form_type_arguments()
        ):
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            sizes.append(size)
        return sizes

    @staticmethod
    def wrap_array_dimensions(type_node, sizes):
        """``T`` with suffixes ``[a][b]`` as an array of ``a`` arrays of ``b`` ``T``: the first suffix is outermost.

        A declarator's suffixes enclose its type's own (GLSL 4.60 section 4.1.9: ``vec4[2] a[3]`` is an array of 3
        ``vec4[2]``), so they wrap ``type_node`` whole.
        """
        for size in reversed(sizes):
            type_node = ArrayType(type_node, size)
        return type_node

    def parse_array_suffixes(self, type_node):
        """Parse one or more non-attribute array suffixes after a type/name."""
        return self.wrap_array_dimensions(type_node, self.parse_array_suffix_sizes())

    def parse_variable_qualifiers(self, layout_attributes=None):
        """Parse declaration qualifiers that precede a variable type.

        GLSL 4.20 accepts the qualifiers of a declaration in any order, so ``uniform layout(rgba16f) image2D x;``
        is the same declaration as ``layout(rgba16f) uniform image2D x;``. When ``layout_attributes`` is a list, a
        ``layout(...)`` qualifier that follows another qualifier is parsed into it instead of ending the sequence.
        """
        qualifiers = []
        while True:
            if (
                layout_attributes is not None
                and qualifiers
                and self.current_token[0] == "LAYOUT"
                and self.peek()[0] == "LPAREN"
            ):
                layout_attributes.extend(self.parse_layout_attributes())
                continue
            if not self.current_token_is_variable_qualifier():
                break
            token_type, token_value = self.current_token
            if str(token_value).lower() in RESOURCE_MEMORY_QUALIFIER_NAMES:
                qualifiers.append(self.parse_resource_memory_qualifier())
                continue
            qualifiers.append(str(token_value).lower())
            self.eat(token_type)
        return qualifiers

    def current_token_is_variable_qualifier(self):
        """Return whether the current token is a variable declaration qualifier."""
        token_type, token_value = self.current_token
        return self.token_is_variable_qualifier(token_type, token_value)

    def token_is_variable_qualifier(self, token_type, token_value):
        """Return whether a token is a variable declaration qualifier."""
        if token_type in VARIABLE_QUALIFIER_TOKEN_TYPES:
            return True
        return (
            token_type == "IDENTIFIER" and str(token_value) in VARIABLE_QUALIFIER_NAMES
        )

    def current_token_is_parameter_qualifier(self):
        """Return whether the current token is a parameter qualifier."""
        token_type, token_value = self.current_token
        if token_type in PARAMETER_QUALIFIER_TOKEN_TYPES:
            return True
        return (
            token_type == "IDENTIFIER" and str(token_value) in VARIABLE_QUALIFIER_NAMES
        )

    def is_variable_declaration(self):
        """
        Lookahead check for variable declarations.
        Handles complex cases and distinguishes from function calls and member access.
        """
        saved_pos = self.pos
        saved_token = self.current_token
        token_mark = self.mark_token_edits()

        try:
            self.parse_attribute_annotations()

            if self.current_token[0] == "LET":
                return True

            if self.current_token_starts_type_alias_declaration():
                self.eat("IDENTIFIER")

            if (
                self.current_token[0] == "VAR"
                and self.peek()[0] != "LESS_THAN"
                and self.peek(2)[0] in {"COLON", "EQUALS"}
            ):
                return True

            if self.current_token_starts_var_address_space_declaration():
                return True

            self.parse_variable_qualifiers(layout_attributes=[])

            if not self.is_type_token():
                return False

            self.advance_over_type()

            if (
                self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}
                and not self.pointer_suffix_starts_declaration()
            ):
                return False

            self.advance_over_pointer_suffix()

            if not self.current_token_is_declaration_binding_identifier():
                return False

            self.parse_binding_identifier()

            next_token = self.current_token[0]

            if next_token == "LBRACKET":
                return True

            if next_token == "EQUALS":
                return True

            if next_token == "SEMICOLON":
                return True

            if next_token == "COMMA":
                return True

            if next_token in ["RPAREN", "COMMA"] and self.in_parameter_context():
                return True

            if next_token in ["DOT", "LPAREN"]:
                return False

            if next_token in [
                "PLUS",
                "MINUS",
                "MULTIPLY",
                "DIVIDE",
                "MOD",
                "EQUAL",
                "NOT_EQUAL",
                "LESS_THAN",
                "GREATER_THAN",
                "LOGICAL_AND",
                "LOGICAL_OR",
                "BITWISE_AND",
                "BITWISE_OR",
                "ASSIGN_ADD",
                "ASSIGN_SUB",
                "ASSIGN_MUL",
                "ASSIGN_DIV",
                "ASSIGN_MOD",
                "ASSIGN_XOR",
                "ASSIGN_OR",
                "ASSIGN_AND",
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
            ]:
                return False

            return True

        except Exception:
            return False
        finally:
            self.undo_token_edits(token_mark)
            self.pos = saved_pos
            self.current_token = saved_token

    def current_token_starts_type_alias_declaration(self):
        """Return whether the current token starts a C-style type alias."""
        token_type, token_value = self.current_token
        return token_type == "IDENTIFIER" and token_value == "typedef"

    def pointer_suffix_starts_declaration(self):
        """Return whether pointer/reference suffix lookahead forms a declaration."""
        offset = 0
        while self.peek(offset)[0] in {"MULTIPLY", "BITWISE_AND"}:
            token_type = self.peek(offset)[0]
            offset += 1
            if token_type == "BITWISE_AND" and self.peek(offset)[0] == "MUT":
                offset += 1

        declarator_type, declarator_value = self.peek(offset)
        if declarator_type != "IDENTIFIER" and not (
            # ``device float* local = p;``: a contextual keyword names the variable, as it may without the pointer.
            isinstance(declarator_value, str)
            and declarator_value.isidentifier()
            and self.peek(offset + 1)[0] in {"EQUALS", "SEMICOLON", "COMMA", "LBRACKET"}
        ):
            return False

        next_token = self.peek(offset + 1)[0]
        # ``buffer float* data @binding(0);``: a post-declaration attribute follows the declarator.
        if next_token in {"LBRACKET", "EQUALS", "SEMICOLON", "COMMA", "ATTRIBUTE"}:
            return True
        return next_token == "RPAREN" and self.in_parameter_context()

    def advance_over_pointer_suffix(self):
        """Advance over pointer/reference suffix tokens in type lookahead."""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            token_type = self.current_token[0]
            self.eat(token_type)
            if token_type == "BITWISE_AND" and self.current_token[0] == "MUT":
                self.eat("MUT")

    def in_parameter_context(self):
        """Check if we're currently parsing function parameters."""
        for i in range(max(0, self.pos - 10), self.pos):
            if i < len(self.tokens) and self.tokens[i][0] == "LPAREN":
                return True
        return False

    def parse_constant(self):
        """Parse a constant declaration."""
        self.eat("CONST")
        const_type = self.parse_type()
        name = self.parse_binding_identifier()
        const_type = self.parse_array_suffixes(const_type)
        attributes = self.parse_post_declaration_attributes()

        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")

        return ConstantNode(
            name=name, const_type=const_type, value=value, attributes=attributes
        )
