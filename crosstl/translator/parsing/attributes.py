"""Attributes and layouts in the CrossGL parser: square and colon attributes, layout qualifiers, execution configurations, GLSL buffer blocks, preprocessor directives, precision statements and attributed declarations."""

from ..anonymous_blocks import AnonymousBufferBlock
from ..ast import (
    AttributeNode,
    BinaryOpNode,
    IdentifierNode,
    LayoutQualifierNode,
    LiteralNode,
    NamedType,
    PreprocessorNode,
    PrimitiveType,
    StructNode,
    UnaryOpNode,
    VariableNode,
)
from ..host_workgroup_size import (
    explicit_workgroup_size,
    padded_workgroup_size_arguments,
)
from ..refusals import CrossGLSyntaxError
from ..vulkan_memory_scope import VULKAN_MEMORY_SCOPE_QUALIFIERS


class ParserAttributeMixin:
    """Attribute members of ``Parser``."""

    def parse_square_attributes(self, allow_single_square=True):
        """Parse HLSL-style ``[attr]`` or Metal-style ``[[attr]]`` metadata."""
        attributes = []

        while self.current_token_starts_square_attribute(allow_single_square):
            double_bracket = self.peek()[0] == "LBRACKET"
            self.eat("LBRACKET")
            if double_bracket:
                self.eat("LBRACKET")

            while self.current_token[0] != "RBRACKET":
                name = str(self.current_token[1])
                self.eat(self.current_token[0])

                arguments = []
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    while self.current_token[0] != "RPAREN":
                        arguments.append(self.parse_expression())
                        if self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                            continue
                        break
                    self.eat("RPAREN")

                attributes.append(self.source_attribute(name, arguments))

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break

            self.eat("RBRACKET")
            if double_bracket:
                self.eat("RBRACKET")

        return attributes

    def current_token_starts_square_attribute(self, allow_single_square=True):
        """Return whether the current bracketed group is metadata."""
        if self.current_token[0] != "LBRACKET":
            return False

        offset = 1
        double_bracket = self.peek()[0] == "LBRACKET"
        if double_bracket:
            offset = 2
        elif not allow_single_square:
            return False

        name_token_type, name_token_value = self.peek(offset)
        if name_token_type in {"RBRACKET", "EOF"}:
            return False
        if not isinstance(name_token_value, str) or not name_token_value.isidentifier():
            return False

        return self.peek(offset + 1)[0] in {"LPAREN", "COMMA", "RBRACKET"}

    def parse_attribute_annotations(self, allow_single_square=True):
        """Parse consecutive ``@``, HLSL ``[]``, or Metal ``[[]]`` attributes."""
        attributes = []
        while self.current_token[0] in [
            "AT",
            "ATTRIBUTE",
        ] or self.current_token_starts_square_attribute(allow_single_square):
            if self.current_token[0] in ["AT", "ATTRIBUTE"]:
                attributes.extend(self.parse_attributes())
            else:
                attributes.extend(self.parse_square_attributes(allow_single_square))
        return attributes

    def parse_post_declaration_attributes(self):
        """Parse declaration metadata that appears after a name or parameter list."""
        attributes = []
        while (
            self.current_token[0] == "COLON"
            or self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute(allow_single_square=False)
        ):
            if self.current_token[0] == "COLON":
                attributes.extend(self.parse_colon_semantic_attributes())
            else:
                attributes.extend(
                    self.parse_attribute_annotations(allow_single_square=False)
                )
        return attributes

    def parse_colon_semantic_attributes(self):
        """Parse HLSL-style ``: SEMANTIC`` metadata into attributes."""
        if self.current_token[0] != "COLON":
            return []

        self.eat("COLON")
        name = str(self.current_token[1])
        self.eat(self.current_token[0])

        arguments = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                arguments.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break
            self.eat("RPAREN")

        return [self.source_attribute(name, arguments)]

    def source_attribute(self, name, arguments):
        """An attribute as CrossGL source spells it, with a workgroup size's omitted dimensions made 1."""
        return AttributeNode(
            name=name,
            arguments=padded_workgroup_size_arguments(
                name, arguments, lambda: LiteralNode(1, PrimitiveType("int"))
            ),
        )

    def parse_compute_layout(self):
        """Parse compute local-size layout metadata inside a stage block."""
        return self.execution_config_from_layout(self.parse_stage_layout_qualifier())

    def parse_layout_attributes(self):
        """Parse a ``layout(...)`` qualifier into attributes."""
        attributes = []
        self.eat("LAYOUT")
        self.eat("LPAREN")

        while self.current_token[0] != "RPAREN":
            name = self.current_token[1]
            self.eat(self.current_token[0])

            arguments = []
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                arguments.append(self.parse_expression())

            attributes.append(AttributeNode(name=name, arguments=arguments))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")
        return attributes

    def parse_stage_layout_qualifier(self):
        """Parse stage-level ``layout(...) in/out;`` metadata."""
        entries = []
        self.eat("LAYOUT")
        self.eat("LPAREN")

        while self.current_token[0] != "RPAREN":
            key = self.current_token[1]
            self.eat(self.current_token[0])

            arguments = []
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                arguments.append(self.parse_expression())

            entries.append(AttributeNode(name=key, arguments=arguments))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        direction = None
        if self.current_token[0] in {"IN", "IDENTIFIER"} and self.current_token[1] in {
            "in",
            "out",
            "patch",
        }:
            direction = self.current_token[1]
            self.eat(self.current_token[0])
        self.eat("SEMICOLON")
        return LayoutQualifierNode(entries=entries, direction=direction)

    def execution_config_from_layout(self, layout):
        """Return execution metadata implied by a stage layout qualifier."""
        if getattr(layout, "direction", None) != "in":
            return {}

        execution_config = {}
        for entry in getattr(layout, "entries", []) or []:
            # local_size_{x,y,z}_id name the specialization constants that set a dimension (GLSL 4.60 4.4.1.4).
            if entry.name not in {
                "local_size_x",
                "local_size_y",
                "local_size_z",
                "local_size_x_id",
                "local_size_y_id",
                "local_size_z_id",
            }:
                continue
            if len(entry.arguments) != 1:
                continue
            execution_config[entry.name] = self.layout_argument_to_string(
                entry.arguments[0]
            )
        return execution_config

    def execution_config_from_function_attributes(self, function):
        """Return execution metadata implied by stage function attributes."""
        size = explicit_workgroup_size(
            getattr(function, "attributes", []) or [], self.layout_argument_to_string
        )
        return {} if size is None else {"numthreads": size}

    def layout_argument_to_string(self, argument):
        """Return the source-like value for simple layout argument expressions."""
        if isinstance(argument, BinaryOpNode):
            left = self.layout_argument_to_string(argument.left)
            right = self.layout_argument_to_string(argument.right)
            return f"{left} {argument.op} {right}"
        if isinstance(argument, UnaryOpNode):
            operand = self.layout_argument_to_string(argument.operand)
            if argument.is_postfix:
                return f"{operand}{argument.op}"
            return f"{argument.op}{operand}"
        if hasattr(argument, "value"):
            return str(argument.value)
        if hasattr(argument, "name"):
            return str(argument.name)
        return str(argument)

    def is_stage_layout_qualifier(self):
        """Return whether the current layout is a standalone stage qualifier."""
        if self.current_token[0] != "LAYOUT" or self.peek(1)[0] != "LPAREN":
            return False

        depth = 1
        offset = 2
        while depth > 0:
            token_type, _token_value = self.peek(offset)
            if token_type == "EOF":
                return False
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
            offset += 1

        direction_type, direction_value = self.peek(offset)
        if direction_type == "IN":
            direction_value = "in"
        if direction_type != "IDENTIFIER" and direction_type != "IN":
            return False
        if direction_value not in {"in", "out", "patch"}:
            return False
        return self.peek(offset + 1)[0] == "SEMICOLON"

    def is_layout_buffer_block_declaration(self):
        """Return whether current tokens start a GLSL-style buffer block."""
        offset = 0
        while self.token_is_layout_buffer_block_qualifier(self.peek(offset)):
            offset += 1
        return (
            self.peek(offset)[0] == "BUFFER"
            and self.peek(offset + 1)[0] == "IDENTIFIER"
            and self.peek(offset + 2)[0] == "LBRACE"
        )

    def parse_layout_buffer_block(self, layout_attributes):
        """Parse ``layout(...) [readonly] buffer Block { ... } instance;``."""
        qualifiers = []
        while self.token_is_layout_buffer_block_qualifier(self.current_token):
            token_type, token_value = self.current_token
            qualifiers.append(str(token_value).lower())
            self.eat(token_type)

        self.eat("BUFFER")
        block_name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            self.reject_end_of_input(f"buffer block {block_name!r}")
            member = self.parse_struct_member()
            self.append_parsed_nodes(members, member)
        self.eat("RBRACE")

        variable_name = self.default_layout_buffer_instance_name(block_name)
        variable_type = NamedType(block_name)
        anonymous = self.current_token[0] != "IDENTIFIER"
        if not anonymous:
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")

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
        variable_type = self.wrap_array_dimensions(variable_type, sizes)

        self.eat("SEMICOLON")

        struct_node = StructNode(name=block_name, members=members)
        variable_node = VariableNode(
            name=variable_name,
            var_type=variable_type,
            qualifiers=qualifiers + ["buffer"],
            attributes=self.glsl_buffer_block_attributes(layout_attributes),
        )
        if anonymous:
            self.anonymous_buffer_blocks.append(
                AnonymousBufferBlock(
                    variable_node,
                    [getattr(member, "name", None) for member in members],
                )
            )
        return struct_node, variable_node

    def glsl_buffer_block_attributes(self, layout_attributes):
        """Convert GLSL layout metadata into CrossGL buffer-block attributes."""
        attributes = []
        block_layout = self.glsl_buffer_block_layout_attribute(layout_attributes)
        if block_layout is not None:
            attributes.append(
                AttributeNode(
                    "glsl_buffer_block",
                    [IdentifierNode(block_layout)],
                )
            )
        for attr in layout_attributes:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name != block_layout:
                attributes.append(attr)
        return attributes

    def glsl_buffer_block_layout_attribute(self, layout_attributes):
        """Return the GLSL memory layout name from layout attributes."""
        for attr in layout_attributes:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def token_is_layout_buffer_block_qualifier(self, token):
        """Return whether token is a qualifier before a GLSL buffer block."""
        token_type, token_value = token
        if token_type == "BUFFER":
            return False
        if token_type == "IDENTIFIER" and token_value in VULKAN_MEMORY_SCOPE_QUALIFIERS:
            return True
        return self.token_is_variable_qualifier(token_type, token_value)

    def default_layout_buffer_instance_name(self, block_name):
        """Return a stable fallback instance name for anonymous buffer blocks."""
        if not block_name:
            return "bufferBlock"
        return block_name[0].lower() + block_name[1:]

    def parse_preprocessor_directive(self):
        """Parse a preprocessor token into a structured directive node."""
        text = self.current_token[1] or ""
        self.eat("PREPROCESSOR")
        stripped = text.lstrip("#").strip()
        if not stripped:
            return PreprocessorNode("", "")
        parts = stripped.split(None, 1)
        directive = parts[0]
        content = parts[1] if len(parts) > 1 else ""
        return PreprocessorNode(directive, content)

    def parse_precision_statement(self):
        """Parse a GLSL-style precision statement as a preprocessor node."""
        self.eat("PRECISION")
        parts = []
        while self.current_token[0] not in ["SEMICOLON", "EOF"]:
            parts.append(str(self.current_token[1]))
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        content = " ".join(parts).strip()
        return PreprocessorNode("precision", content)

    def parse_attributed_cbuffer(self, leading_attributes=None):
        attributes = list(leading_attributes or [])
        attributes.extend(self.parse_attribute_annotations())
        return self.parse_cbuffer_as_struct(attributes)

    def current_token_starts_attributed_cbuffer_declaration(self):
        """Return whether leading attributes are followed by a cbuffer."""
        if not (
            self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute()
        ):
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        try:
            attributes = self.parse_attribute_annotations()
            return bool(attributes and self.is_cbuffer_declaration())
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

    def parse_attributed_struct(self):
        attributes = self.parse_attribute_annotations()
        if self.is_generic_declaration():
            struct_node = self.parse_generic_declaration()
            if not isinstance(struct_node, StructNode):
                raise CrossGLSyntaxError(
                    "Attributes before a generic declaration must precede a struct"
                )
            struct_node.attributes = attributes + list(struct_node.attributes or [])
            return struct_node
        return self.parse_struct(attributes)

    def current_token_starts_attributed_struct_declaration(self):
        """Return whether leading attributes are followed by ``struct`` or ``generic<...> struct``."""
        if not (
            self.current_token[0] in ["AT", "ATTRIBUTE"]
            or self.current_token_starts_square_attribute()
        ):
            return False

        saved_pos = self.pos
        saved_token = self.current_token
        try:
            attributes = self.parse_attribute_annotations()
            if attributes and self.is_generic_declaration():
                self.eat("IDENTIFIER")
                if self.current_token[0] != "LESS_THAN":
                    return False
                self.parse_generic_parameters()
            return bool(attributes and self.current_token[0] == "STRUCT")
        finally:
            self.pos = saved_pos
            self.current_token = saved_token

    def parse_attributes(self):
        """Parse one or more ``@`` attribute annotations."""
        attributes = []

        while self.current_token[0] in ["AT", "ATTRIBUTE"]:
            if self.current_token[0] == "ATTRIBUTE":
                name = self.current_token[1][1:]
                self.eat("ATTRIBUTE")
            else:
                self.eat("AT")
                name = self.current_token[1]
                self.eat("IDENTIFIER")

            arguments = []
            if name in {"gl_FragData"} and self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    arguments.append(self.parse_expression())
                self.eat("RBRACKET")
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                while self.current_token[0] != "RPAREN":
                    arguments.append(self.parse_expression())
                    if self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                self.eat("RPAREN")

            attributes.append(self.source_attribute(name, arguments))

        return attributes
