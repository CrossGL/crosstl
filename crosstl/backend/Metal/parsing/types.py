"""Global variables and type specifiers in the Metal parser: type qualifiers, signed prefixes, balanced token text, generic type tokens and alignas specifiers."""

from crosstl.translator.refusals import MetalSyntaxError

from ..MetalAst import (
    AssignmentNode,
    VariableNode,
)
from ..signed_type_spellings import fold_signed_type_spellings
from .parser_tables import (
    IDENTIFIER_PREFIX_TYPE_QUALIFIERS,
    IDENTIFIER_TYPE_QUALIFIERS,
    QUALIFIER_TOKENS,
    RAYTRACING_TYPE_QUALIFIERS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    SIGNED_PREFIX_TYPE_TOKENS,
    SIGNED_TYPE_PREFIXES,
    TYPE_QUALIFIER_FUNCTIONS,
    TYPE_TOKENS,
    UNSIGNED_INTEGER_TYPES,
)


class MetalTypeParsingMixin:
    """Type members of ``MetalParser``."""

    def parse_global_variable(self, pre_alignas=None):
        start_token = self.current_token
        attributes = self.parse_attributes()
        alignas_specs = pre_alignas or self.parse_alignas_specifiers()
        vtype, qualifiers = self.parse_type_specifier(attributes=attributes)
        declaration_attributes = list(attributes)
        base_vtype = self.base_type_for_remaining_declarators(vtype)
        name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
        vtype = self.apply_declarator_type_suffix(vtype, type_suffix)
        var_attributes = self.parse_attributes()
        attributes.extend(var_attributes)

        var_node = VariableNode(
            vtype, name, qualifiers=qualifiers, attributes=attributes
        )
        var_node.array_sizes = array_sizes
        self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
        var_node.alignas = alignas_specs
        var_node.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        self.register_local_variable_name(name)
        if "const" in qualifiers or "constexpr" in qualifiers:
            var_node.is_const = True

        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()
            node = AssignmentNode(var_node, value)
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_global_variable_declarations(
                    base_vtype,
                    qualifiers,
                    declaration_attributes,
                    alignas_specs,
                    [node],
                )
            self.eat_statement_semicolon()
            return node

        if self.current_token[0] == "LPAREN":
            node = AssignmentNode(
                var_node,
                self.parse_parenthesized_declaration_initializer(vtype),
            )
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_global_variable_declarations(
                    base_vtype,
                    qualifiers,
                    declaration_attributes,
                    alignas_specs,
                    [node],
                )
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "LBRACE":
            value = self.parse_braced_declaration_initializer(vtype)
            node = AssignmentNode(var_node, value)
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_global_variable_declarations(
                    base_vtype,
                    qualifiers,
                    declaration_attributes,
                    alignas_specs,
                    [node],
                )
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "COMMA":
            return self.parse_remaining_global_variable_declarations(
                base_vtype,
                qualifiers,
                declaration_attributes,
                alignas_specs,
                [var_node],
            )

        self.eat_statement_semicolon()
        return var_node

    def parse_remaining_global_variable_declarations(
        self, vtype, qualifiers, declaration_attributes, alignas_specs, nodes
    ):
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            decl_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes = list(declaration_attributes)
            attributes.extend(self.parse_attributes())
            var_node = VariableNode(
                decl_type, name, qualifiers=list(qualifiers), attributes=attributes
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = list(alignas_specs)
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                nodes.append(AssignmentNode(var_node, self.parse_expression()))
            elif self.current_token[0] == "LPAREN":
                nodes.append(
                    AssignmentNode(
                        var_node,
                        self.parse_parenthesized_declaration_initializer(decl_type),
                    )
                )
            elif self.current_token[0] == "LBRACE":
                nodes.append(
                    AssignmentNode(
                        var_node,
                        self.parse_braced_declaration_initializer(decl_type),
                    )
                )
            else:
                nodes.append(var_node)

        self.eat("SEMICOLON")
        return nodes

    def parse_type_specifier(self, attributes=None):
        qualifiers = []
        pointee_qualifiers = []
        indirection_qualifiers = []
        while (
            self.is_type_qualifier_start()
            or self.current_token[0] == "ATTRIBUTE"
            or self.is_gnu_attribute_start()
        ):
            if self.is_type_qualifier_start():
                parsed_qualifiers = self.parse_type_qualifier()
                qualifiers.extend(parsed_qualifiers)
                pointee_qualifiers.extend(parsed_qualifiers)
                continue
            parsed_attributes = self.parse_attributes()
            if attributes is not None:
                attributes.extend(parsed_attributes)

        if (
            self.current_token[0] in {"STRUCT", "ENUM"}
            and self.peek(1)[0] == "IDENTIFIER"
        ):
            tag_kind = self.current_token[1]
            self.eat(self.current_token[0])
            base_type = f"{tag_kind} {self.current_token[1]}"
            self.eat("IDENTIFIER")
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "decltype"
        ):
            base_type = self.parse_decltype_member_type()
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "typename"
            and self.peek(1) == ("IDENTIFIER", "decltype")
            and self.peek(2)[0] == "LPAREN"
        ):
            self.eat("IDENTIFIER")
            base_type = self.parse_decltype_member_type()
        elif self.current_token[0] in {"METAL", "SCOPE"} or (
            self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "SCOPE"
        ):
            base_type = self.parse_scoped_identifier()
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "typename"
            and self.peek(1)[0] in TYPE_TOKENS | {"SCOPE"}
        ):
            self.eat("IDENTIFIER")
            if self.current_token[0] in {"METAL", "SCOPE"} or (
                self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "SCOPE"
            ):
                base_type = self.parse_scoped_identifier()
            else:
                base_type = self.current_token[1]
                self.eat(self.current_token[0])
        elif self.current_token[0] == "MESH" and self.peek(1)[0] == "LESS_THAN":
            # ``mesh<V, P, NV, NP, topology>`` after ``using namespace metal``: the
            # lexer reads ``mesh`` as the stage keyword of ``[[mesh]]``.
            base_type = self.current_token[1]
            self.eat("MESH")
        elif self.current_token[0] not in TYPE_TOKENS:
            raise MetalSyntaxError(f"Expected type, got {self.current_token[0]}")
        elif (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in SIGNED_TYPE_PREFIXES
        ):
            signed_prefix = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] in SIGNED_PREFIX_TYPE_TOKENS:
                base_type = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                base_type = "int"
            if signed_prefix == "unsigned":
                base_type = UNSIGNED_INTEGER_TYPES[base_type]
        else:
            base_type = self.current_token[1]
            self.eat(self.current_token[0])

        if self.current_token[0] == "LESS_THAN":
            inner = self.parse_template_argument_suffix()
            base_type = f"{base_type}<{self.format_generic_type_tokens(inner)}>"

        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                raise MetalSyntaxError(
                    f"Expected identifier after '::', got {self.current_token[0]}"
                )
            base_type += f"::{self.current_token[1]}"
            self.eat(self.current_token[0])

        pointer_suffix = ""
        saw_indirection = False
        while self.is_post_type_qualifier_start() or self.current_token[0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            if self.is_post_type_qualifier_start():
                parsed_qualifiers = self.parse_type_qualifier()
                qualifiers.extend(parsed_qualifiers)
                if saw_indirection:
                    indirection_qualifiers.extend(parsed_qualifiers)
                else:
                    pointee_qualifiers.extend(parsed_qualifiers)
                continue
            pointer_suffix += "*" if self.current_token[0] == "MULTIPLY" else "&"
            saw_indirection = True
            self.eat(self.current_token[0])

        if self.current_token[0] == "ELLIPSIS":
            pointer_suffix += "..."
            self.eat("ELLIPSIS")

        self.last_type_specifier_qualifier_contract = {
            "pointee": tuple(dict.fromkeys(pointee_qualifiers)),
            "indirection": tuple(dict.fromkeys(indirection_qualifiers)),
        }
        return base_type + pointer_suffix, qualifiers

    def is_post_type_qualifier_start(self):
        if not self.is_type_qualifier_start():
            return False
        if self.current_token[0] == "RESTRICT" and self.peek(1)[0] in {
            "SEMICOLON",
            "COMMA",
            "EQUALS",
            "LPAREN",
            "LBRACE",
            "LBRACKET",
        }:
            return False
        return True

    def skip_signed_type_prefix_at(self, idx):
        if (
            idx >= len(self.tokens)
            or self.tokens[idx][0] != "IDENTIFIER"
            or self.tokens[idx][1] not in SIGNED_TYPE_PREFIXES
        ):
            return idx
        idx += 1
        if idx < len(self.tokens) and self.tokens[idx][0] in SIGNED_PREFIX_TYPE_TOKENS:
            idx += 1
        return idx

    def is_type_qualifier_start(self):
        if self.is_contextual_access_identifier_at(self.pos):
            return False
        if self.current_token[0] in QUALIFIER_TOKENS:
            return True
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1]
            in IDENTIFIER_TYPE_QUALIFIERS | RAYTRACING_TYPE_QUALIFIERS
        ):
            return True
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in IDENTIFIER_PREFIX_TYPE_QUALIFIERS
        ):
            return self.prefix_type_qualifier_followed_by_type_at(self.pos)
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in TYPE_QUALIFIER_FUNCTIONS
        )

    def parse_type_qualifier(self):
        if self.current_token[0] in QUALIFIER_TOKENS or (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1]
            in (
                IDENTIFIER_TYPE_QUALIFIERS
                | RAYTRACING_TYPE_QUALIFIERS
                | IDENTIFIER_PREFIX_TYPE_QUALIFIERS
            )
        ):
            qualifier = self.current_token[1]
            self.eat(self.current_token[0])
            return (qualifier,)

        qualifier_name = self.current_token[1]
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            args = self.parse_balanced_token_text("LPAREN", "RPAREN")
            return (f"{qualifier_name}({args})",)
        return (qualifier_name,)

    def parse_balanced_token_text(
        self, open_token, close_token, error_message="Unterminated type qualifier"
    ):
        self.eat(open_token)
        depth = 1
        tokens = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == open_token:
                depth += 1
                tokens.append(token_value)
                self.eat(token_type)
            elif token_type == close_token:
                depth -= 1
                if depth == 0:
                    self.eat(close_token)
                    break
                tokens.append(token_value)
                self.eat(token_type)
            else:
                tokens.append(token_value)
                self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError(error_message)
        return self.format_generic_type_tokens(tokens)

    def prefix_type_qualifier_followed_by_type_at(self, idx):
        next_idx = idx + 1
        if next_idx >= len(self.tokens):
            return False
        token_type, token_value = self.tokens[next_idx]
        if token_type != "IDENTIFIER":
            return token_type in TYPE_TOKENS
        return token_value in self.known_types or self.identifier_type_name_at(next_idx)

    def identifier_type_name_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx][0] == "IDENTIFIER"
            and self.tokens[idx + 1][0]
            in {"IDENTIFIER", "MULTIPLY", "BITWISE_AND", "LESS_THAN", "SCOPE"}
        )

    def format_generic_type_tokens(self, tokens):
        text = ""
        previous = ""
        compact_before = {">", ",", "]", ")", "*", "&", "::"}
        compact_after = {"<", ",", "[", "(", "::"}
        for token in fold_signed_type_spellings(tokens):
            token = str(token)
            if (
                text
                and token not in compact_before
                and previous not in compact_after
                and self.generic_type_token_needs_space(previous, token)
            ):
                text += " "
            text += token
            previous = token
        return text

    def generic_type_token_needs_space(self, previous, current):
        if not previous or not current:
            return False
        return (
            previous[-1].isalnum() or previous[-1] == "_" or previous[-1] == ">"
        ) and (current[0].isalnum() or current[0] == "_")

    def parse_alignas_specifiers(self):
        specs = []
        while self.current_token[0] == "ALIGNAS":
            self.eat("ALIGNAS")
            self.eat("LPAREN")
            if self.is_type_start():
                type_name, _quals = self.parse_type_specifier()
                specs.append(("type", type_name))
            else:
                expr = self.parse_expression()
                specs.append(expr)
            self.eat("RPAREN")
        return specs

    def is_type_start(self):
        if self.is_type_qualifier_start():
            return True
        if self.current_token[0] in TYPE_TOKENS:
            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                if name in self.known_types:
                    return True
                next_tok = self.peek(1)[0]
                if next_tok == "SCOPE":
                    return True
                return False
            return True
        return False
