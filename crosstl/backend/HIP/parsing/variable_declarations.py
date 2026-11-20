"""Variable declarations in the HIP parser: qualifiers, declarators, bit-fields, structured bindings, memory spaces, initializers and array suffixes."""

from ...CUDA.local_memory_spaces import refuse_local_memory_space
from ..HipAst import (
    FunctionCallNode,
    VariableNode,
)


class HipVariableDeclarationMixin:
    """Variable declaration members of ``HipParser``."""

    def parse_variable_declaration(self, consume_semicolon=True):
        qualifiers = self.parse_declaration_qualifiers()
        attributes = []
        var_type = self.parse_variable_declaration_type(qualifiers, attributes)
        self.skip_newlines()
        name = self.consume_variable_declarator_name()
        var_type += self.parse_array_suffix()
        attributes.extend(self.skip_declarator_attribute_suffixes())
        self.skip_newlines()

        value = None
        if self.match("ASSIGN"):
            self.advance()
            value = self.parse_expression()
        elif self.match("LPAREN"):
            value = FunctionCallNode(
                self.constructor_initializer_name(var_type),
                self.parse_parenthesized_argument_list(),
            )
        elif self.match("LBRACE"):
            value = self.parse_initializer_list()

        if consume_semicolon:
            self.skip_newlines()
            self.consume("SEMICOLON")

        return VariableNode(
            var_type,
            name,
            value,
            qualifiers,
            attributes=attributes,
            is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
            is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                var_type, qualifiers
            ),
        )

    def parse_variable_declaration_list(self, consume_semicolon=True):
        qualifiers = self.parse_declaration_qualifiers()
        attributes = []
        first_type = self.parse_variable_declaration_type(qualifiers, attributes)
        base_type = self.strip_declarator_markers(first_type)
        declarations = [
            self.parse_variable_declarator(
                first_type, qualifiers, allow_prefix=False, attributes=attributes
            )
        ]

        while self.match("COMMA"):
            self.advance()
            declarations.append(
                self.parse_variable_declarator(
                    base_type, qualifiers, allow_prefix=True, attributes=attributes
                )
            )

        if consume_semicolon:
            self.skip_newlines()
            self.consume("SEMICOLON")

        return declarations

    def parse_declaration_qualifiers(self):
        qualifiers = []

        while self.match(*self.DECLARATION_QUALIFIER_TOKENS):
            qualifiers.append(self.current_token.value)
            self.advance()

        return qualifiers

    def parse_variable_declaration_type(self, qualifiers, attributes=None):
        attributes = attributes if attributes is not None else []
        saved_pos = self.pos
        saved_token = self.current_token
        type_prefixes = []
        saw_integral_sign = False

        self.skip_newlines()
        self.skip_cpp_attributes()
        parsed_attributes = self.parse_type_attribute_prefixes()

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_prefixes.append(self.current_token.value)
            self.advance()

        self.skip_cpp_attributes()
        interleaved_qualifiers = self.parse_declaration_qualifiers()
        if interleaved_qualifiers:
            qualifiers.extend(interleaved_qualifiers)
            if self.is_implicit_int_current_type(saw_integral_sign):
                base_type = "int"
            else:
                base_type = self.parse_type()
            qualifiers.extend(self.parse_declaration_qualifiers())
            attributes.extend(parsed_attributes)
            return " ".join([*type_prefixes, base_type]).strip()

        self.pos = saved_pos
        self.current_token = saved_token
        self.skip_newlines()
        self.skip_cpp_attributes()
        attributes.extend(self.parse_type_attribute_prefixes())
        base_type = self.parse_type()
        qualifiers.extend(self.parse_declaration_qualifiers())
        return base_type

    def reject_disallowed_local_memory_space(self, qualifiers, name):
        """Refuse a function-body variable that HIP allows in its memory space only with static storage."""
        refuse_local_memory_space(qualifiers, name)

    def parse_variable_declarator(
        self, base_type, qualifiers, allow_prefix, attributes=None
    ):
        attributes = list(attributes or [])
        var_type = base_type
        if allow_prefix:
            var_type = self.parse_declarator_prefix(var_type)

        self.skip_newlines()
        if self.match("LBRACKET"):
            name = self.parse_structured_binding_name()
            self.skip_newlines()
            value = self.parse_variable_initializer(var_type)
            return VariableNode(
                var_type,
                name,
                value,
                list(qualifiers),
                attributes=attributes,
                is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
                is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                    var_type, qualifiers
                ),
            )

        name = "" if self.match("COLON") else self.consume_variable_declarator_name()
        var_type += self.parse_array_suffix()
        attributes.extend(self.skip_declarator_attribute_suffixes())
        self.skip_newlines()
        bitfield_width = self.parse_bitfield_width()
        value = self.parse_variable_initializer(var_type)

        if self.block_depth > 0:
            self.reject_disallowed_local_memory_space(qualifiers, name)

        declaration = VariableNode(
            var_type,
            name,
            value,
            list(qualifiers),
            attributes=attributes,
            is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
            is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                var_type, qualifiers
            ),
        )
        if bitfield_width is not None:
            declaration.bitfield_width = bitfield_width
        return declaration

    def parse_bitfield_width(self):
        self.skip_newlines()
        if not self.match("COLON"):
            return None

        self.advance()
        self.skip_newlines()
        if self.match("COMMA", "SEMICOLON"):
            self.error("Expected HIP record bitfield width")
        return self.parse_expression()

    def parse_structured_binding_name(self):
        tokens = []
        self.consume("LBRACKET")
        self.skip_newlines()

        while self.current_token and not self.match("RBRACKET"):
            if not self.match("NEWLINE"):
                tokens.append(self.current_token.value)
            self.advance()

        self.consume("RBRACKET")
        return f"[{self.format_structured_binding_tokens(tokens)}]"

    def format_structured_binding_tokens(self, tokens):
        text = ""
        previous = None
        for token in tokens:
            if token == ",":
                text = text.rstrip() + ", "
            elif token in {"&", "*", "&&"}:
                text += token
            elif previous in {"&", "*", "&&"} or not text or text.endswith(" "):
                text += token
            else:
                text += " " + token
            previous = token
        return text.strip()

    def is_extern_shared_memory(self, qualifiers):
        return "__shared__" in qualifiers and "extern" in qualifiers

    def is_dynamic_shared_memory(self, var_type, qualifiers):
        return (
            self.is_extern_shared_memory(qualifiers)
            and isinstance(var_type, str)
            and var_type.endswith("[]")
        )

    def parse_declarator_prefix(self, base_type):
        parts = [base_type] if base_type else []

        while self.match(
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        ):
            if self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
                parts.append(self.current_token.value)
                self.advance()
                continue

            parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(parts)

        return " ".join(parts)

    def parse_variable_initializer(self, var_type):
        self.skip_newlines()
        self.skip_declarator_attribute_suffixes()
        if self.match("ASSIGN"):
            self.advance()
            self.skip_newlines()
            return self.parse_expression()
        if self.match("LPAREN"):
            return FunctionCallNode(
                self.constructor_initializer_name(var_type),
                self.parse_parenthesized_argument_list(),
            )
        if self.match("LBRACE"):
            return self.parse_initializer_list()
        return None

    def skip_declarator_attribute_suffixes(self):
        attributes = []
        previous_pos = -1
        while self.current_token and self.pos != previous_pos:
            previous_pos = self.pos
            self.skip_newlines()
            attributes.extend(self.parse_type_attribute_prefixes())
        return attributes

    def constructor_initializer_name(self, var_type):
        return " ".join(
            part
            for part in str(var_type).split()
            if part not in {
                "const",
                "volatile",
                "__restrict__",
                "__restrict",
                "restrict",
                "&",
                "&&",
            }
        )

    def strip_declarator_markers(self, var_type):
        parts = str(var_type).split()
        while parts and parts[-1] in {"*", "&", "&&", "__restrict__", "restrict"}:
            parts.pop()
        return " ".join(parts)

    def parse_array_suffix(self):
        suffixes = []

        while self.match("LBRACKET"):
            self.consume("LBRACKET")
            if not self.match("RBRACKET"):
                size = self.parse_expression()
                suffixes.append(f"[{self.expression_to_text(size)}]")
            else:
                suffixes.append("[]")
            self.consume("RBRACKET")

        return "".join(suffixes)

    def parse_parenthesized_argument_list(self):
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")
        return args

    def parse_line_split_type_qualifiers(self):
        """Read qualifiers that a line break separates from the rest of a type.

        In `const` / `float* in` or `__restrict__ const` / `float* in`, the
        base type reader stops at the line break; read the qualifiers up to the
        last break here and leave the rest of the type, and qualifiers on one
        line with it, to the rest of :meth:`parse_type`.
        """
        end = self.line_split_type_qualifiers_end(self.pos)
        qualifiers = []
        while self.current_token and self.pos < end:
            if not self.match("NEWLINE"):
                qualifiers.append(self.current_token.value)
            self.advance()
        return qualifiers

    def line_split_type_qualifiers_end(self, index):
        """Index after the last line break that follows a type qualifier at ``index``, or ``index`` if none does.

        A sign word whose next line starts the declarator, as in `unsigned` /
        `n`, stays with the type reader, which reads it as `unsigned int`.
        """
        end = index
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            next_index = self.skip_newlines_at_pos(index + 1)
            implicit_int = self.tokens[index].type in {
                "SIGNED",
                "UNSIGNED",
            } and self.is_implicit_int_type_at_pos(next_index, True)
            if next_index != index + 1 and not implicit_int:
                end = next_index
            index = next_index
        return end
