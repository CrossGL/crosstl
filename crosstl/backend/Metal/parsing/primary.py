"""Primary expressions in the Metal parser: lambdas, initializer lists, designated initializers, scoped identifiers, vector constructors, calls and texture sampling."""

import re

from crosstl.translator.refusals import MetalSyntaxError

from ..MetalAst import (
    CallNode,
    DesignatedInitializerNode,
    DiscardNode,
    FunctionCallNode,
    InitializerListNode,
    LambdaNode,
    MemberAccessNode,
    MethodCallNode,
    TextureSampleNode,
    VariableNode,
    VectorConstructorNode,
)
from .parser_tables import (
    CONSTRUCTOR_TYPE_TOKENS,
    SCOPED_IDENTIFIER_PART_TOKENS,
    TYPE_TOKENS,
)


class MetalPrimaryParsingMixin:
    """Primary members of ``MetalParser``."""

    def parse_primary(self):
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        if self.current_token[0] in ["TRUE", "FALSE"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        if self.current_token[0] == "CHAR_LITERAL":
            value = self.current_token[1]
            self.eat("CHAR_LITERAL")
            return value
        if self.current_token[0] == "STRING":
            return self.parse_string_literal_run()
        if self.current_token[0] == "LPAREN":
            open_token = self.current_token
            self.eat("LPAREN")
            expr = self.parse_expression(allow_comma=True)
            close_token = self.current_token
            self.eat("RPAREN")
            group_location = self.source_span_from_tokens(open_token, close_token)
            if (
                group_location is not None
                and hasattr(expr, "__dict__")
                and self.expression_contains_scoped_reference(expr)
            ):
                group_locations = list(
                    getattr(expr, "group_source_locations", ()) or ()
                )
                group_locations.append(group_location)
                expr.group_source_locations = group_locations
                expr.group_source_location = group_location
            return expr
        if self.current_token[0] == "LBRACKET":
            return self.parse_lambda_expression()
        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        if self.current_token[0] in CONSTRUCTOR_TYPE_TOKENS:
            if self.is_current_name_token() and self.peek(1)[0] not in {
                "LPAREN",
                "LBRACE",
            }:
                start_token = self.current_token
                name = self.parse_scoped_identifier()
                node = VariableNode("", name)
                node.globally_qualified = start_token[0] == "SCOPE"
                if "::" in name:
                    node.source_location = self.source_span_from_tokens(
                        start_token,
                        self.tokens[self.pos - 1],
                    )
                return node
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "LPAREN":
                return self.parse_vector_constructor(type_name)
            if self.current_token[0] == "LBRACE":
                initializer = self.parse_initializer_list()
                node = FunctionCallNode(type_name, [initializer])
                node.is_braced_constructor = True
                return node
            raise MetalSyntaxError(f"Unexpected type in expression: {type_name}")
        if self.current_token[0] in {"METAL", "SCOPE"} or self.is_current_name_token():
            start_token = self.current_token
            name = self.parse_scoped_identifier()
            node = VariableNode("", name)
            node.globally_qualified = start_token[0] == "SCOPE"
            if "::" in name:
                node.source_location = self.source_span_from_tokens(
                    start_token,
                    self.tokens[self.pos - 1],
                )
            return node
        raise MetalSyntaxError(
            f"Unexpected token in expression: {self.current_token[0]}"
        )

    def parse_lambda_expression(self):
        start_token = self.current_token
        capture = self.parse_lambda_capture()
        params = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            if self.current_token[0] != "RPAREN":
                params = self.parse_parameters()
            self.eat("RPAREN")

        return_type = None
        specifier_tokens = []
        while self.current_token[0] not in {"LBRACE", "EOF"}:
            if self.current_token[0] == "ARROW":
                self.eat("ARROW")
                return_type, _qualifiers = self.parse_type_specifier()
                continue
            specifier_tokens.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.current_token[0] != "LBRACE":
            raise MetalSyntaxError("Expected lambda body")

        self.pending_block_scope_names.append(
            {param.name for param in params if getattr(param, "name", None)}
        )
        body = self.parse_block()
        node = LambdaNode(
            capture,
            params,
            body,
            return_type,
            self.format_generic_type_tokens(specifier_tokens).split(),
        )
        node.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        return node

    def parse_lambda_capture(self):
        self.eat("LBRACKET")
        depth = 1
        parts = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LBRACKET":
                depth += 1
                parts.append(token_value)
                self.eat(token_type)
                continue
            if token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACKET")
                    break
                parts.append(token_value)
                self.eat(token_type)
                continue
            parts.append(token_value)
            self.eat(token_type)
        if depth != 0:
            raise MetalSyntaxError("Unterminated lambda capture list")
        return self.format_generic_type_tokens(parts)

    def parse_initializer_list(self):
        start_token = self.current_token
        self.eat("LBRACE")
        elements = []

        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_initializer_element())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
                continue
            break

        self.eat("RBRACE")
        node = InitializerListNode(elements)
        node.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        return node

    def parse_initializer_element(self):
        if self.current_token[0] in ("DOT", "LBRACKET"):
            return self.parse_designated_initializer()
        return self.parse_expression()

    def parse_designated_initializer(self):
        designators = []

        while self.current_token[0] in ("DOT", "LBRACKET"):
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                field = self.current_token[1]
                self.eat("IDENTIFIER")
                designators.append(("field", field))
                continue
            self.eat("LBRACKET")
            index = self.parse_designator_bound_expression()
            if self.current_token[0] == "ELLIPSIS":
                self.eat("ELLIPSIS")
                end = self.parse_designator_bound_expression()
                designators.append(("range", index, end))
                self.eat("RBRACKET")
                continue
            self.eat("RBRACKET")
            designators.append(("index", index))

        self.eat("EQUALS")
        value = self.parse_expression()
        return DesignatedInitializerNode(designators, value)

    def parse_designator_bound_expression(self):
        depth = 0
        parts = []
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if depth == 0:
                if token_type == "RBRACKET":
                    break
                if token_type == "ELLIPSIS":
                    break
            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"}:
                depth -= 1
            parts.append(token_value)
            self.eat(token_type)
        if not parts:
            raise MetalSyntaxError("Expected designator expression")
        return self.format_generic_type_tokens(parts)

    def parse_scoped_identifier(self):
        parts = []
        if self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
        if self.current_token[0] == "METAL":
            parts.append("metal")
            self.eat("METAL")
        else:
            parts.append(self.current_token[1])
            if not self.is_current_name_token():
                raise MetalSyntaxError(
                    f"Expected identifier, got {self.current_token[0]}"
                )
            self.eat(self.current_token[0])
        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if (
                self.current_token == ("IDENTIFIER", "template")
                and self.peek(1)[0] in TYPE_TOKENS
            ):
                self.eat("IDENTIFIER")
            if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                raise MetalSyntaxError(
                    f"Expected identifier after '::', got {self.current_token[0]}"
                )
            if self.current_token[0] == "METAL":
                parts.append("metal")
                self.eat("METAL")
            else:
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
        if parts and parts[0] in self.namespace_aliases:
            parts = self.namespace_aliases[parts[0]].split("::") + parts[1:]
        return "::".join(parts)

    def parse_vector_constructor(self, type_name):
        start_token = self.current_token
        args = self.parse_parenthesized_arguments()
        node = VectorConstructorNode(type_name, args)
        node.source_location = self.source_span_from_tokens(
            start_token, self.tokens[self.pos - 1]
        )
        return node

    @staticmethod
    def declaration_type_is_reference(type_name):
        return bool(re.search(r"&{1,2}\s*$", str(type_name)))

    def parse_parenthesized_declaration_initializer(self, type_name):
        initializer = self.parse_vector_constructor(type_name)
        if self.declaration_type_is_reference(type_name) and len(initializer.args) == 1:
            return initializer.args[0]
        return initializer

    def parse_braced_declaration_initializer(self, type_name):
        initializer = self.parse_initializer_list()
        if (
            self.declaration_type_is_reference(type_name)
            and len(initializer.elements) == 1
        ):
            return initializer.elements[0]
        return initializer

    def parse_parenthesized_arguments(self):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return args

    def parse_call(self, callee):
        open_token = self.current_token
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        close_token = self.current_token
        self.eat("RPAREN")

        if isinstance(callee, MemberAccessNode):
            if callee.member == "sample":
                return self.build_texture_sample(callee.object, args)
            return MethodCallNode(callee.object, callee.member, args)
        if isinstance(callee, VariableNode):
            if callee.name == "discard_fragment" and not args:
                return DiscardNode()
            call_name = callee.name
            if getattr(callee, "globally_qualified", False):
                call_name = f"::{call_name}"
            node = FunctionCallNode(call_name, args)
            call_location = self.source_span_from_tokens(open_token, close_token)
            callee_location = getattr(callee, "source_location", None)
            if callee_location is not None:
                call_location = dict(callee_location)
                closing_location = self.source_span_from_tokens(open_token, close_token)
                if closing_location is not None:
                    for key in ("end_line", "end_column", "end_offset"):
                        if key in closing_location:
                            call_location[key] = closing_location[key]
                    if "offset" in call_location and "end_offset" in call_location:
                        call_location["length"] = (
                            call_location["end_offset"] - call_location["offset"]
                        )
            node.source_location = call_location
            return node
        return CallNode(callee, args)

    def build_texture_sample(self, texture, args):
        sampler = args[0] if len(args) > 0 else None
        coords = args[1] if len(args) > 1 else None
        options = args[2:]
        if options:
            node = TextureSampleNode(texture, sampler, coords, options[0])
            node.options = options
            return node
        return TextureSampleNode(texture, sampler, coords)

    def parse_texture_sample_args(self, texture):
        self.eat("LPAREN")
        sampler = self.parse_expression()
        self.eat("COMMA")
        coordinates = self.parse_expression()

        # Preserve optional sample modifiers such as level(), bias(), gradients, and offsets.
        options = []
        if self.current_token[0] == "COMMA":
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                options.append(self.parse_expression())

        self.eat("RPAREN")

        if options:
            node = TextureSampleNode(texture, sampler, coordinates, options[0])
            node.options = options
            return node
        return TextureSampleNode(texture, sampler, coordinates)

    def parse_texture_sample(self):
        texture = self.parse_expression()
        self.eat("DOT")
        self.eat("IDENTIFIER")  # 'sample' method
        self.eat("LPAREN")
        sampler = self.parse_expression()
        self.eat("COMMA")
        coordinates = self.parse_expression()
        options = []
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            options.append(self.parse_expression())
        self.eat("RPAREN")
        if options:
            node = TextureSampleNode(texture, sampler, coordinates, options[0])
            node.options = options
            return node
        return TextureSampleNode(texture, sampler, coordinates)
