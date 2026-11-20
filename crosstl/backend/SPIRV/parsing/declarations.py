"""Declarations in the GLSL-style text form accepted by the SPIR-V parser: precision, layouts, data types, qualifiers, push constants, descriptor sets, structs, functions and parameters."""

from crosstl.translator.refusals import SpirvSyntaxError

from ..VulkanAst import (
    DescriptorSetNode,
    FunctionNode,
    LayoutNode,
    PushConstantNode,
    StructNode,
    VariableNode,
)
from ..VulkanLexer import VALID_DATA_TYPES


class VulkanDeclarationMixin:
    """Declaration members of ``VulkanParser``."""

    def parse_precision_declaration(self):
        self.eat("PRECISION")
        if self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
            self.eat(self.current_token[0])

        if self.current_token[1] not in VALID_DATA_TYPES:
            raise SpirvSyntaxError(
                f"Unexpected precision type: {self.current_token[1]}"
            )
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []
        push_constant = False
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "PUSH_CONSTANT":
                push_constant = True
                self.eat("PUSH_CONSTANT")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                continue

            binding_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                binding_value = self.parse_layout_qualifier_value()
                bindings.append((binding_name, binding_value))
            else:
                bindings.append((binding_name, None))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        declaration_qualifiers = self.parse_layout_declaration_qualifiers()
        if (
            self.has_specialization_constant_qualifier(bindings)
            and self.current_token[0] == "CONST"
        ):
            declaration = self.parse_assignment_or_function_call()
            return LayoutNode(
                bindings,
                declaration=declaration,
                layout_type="CONST",
                declaration_qualifiers=declaration_qualifiers,
            )

        layout_type = None
        block_name = None
        if self.current_token[0] in ["IN", "OUT", "UNIFORM", "BUFFER"]:
            layout_type = self.current_token[0]
            self.eat(layout_type)
            declaration_qualifiers.extend(self.parse_layout_declaration_qualifiers())
            if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "LBRACE":
                block_name = self.current_token[1]
                self.eat(self.current_token[0])

        data_type = None
        struct_fields = None
        if layout_type in ["UNIFORM", "BUFFER"]:
            if self.current_token[0] == "LBRACE":
                struct_fields = self.parse_layout_block_fields()
                data_type = "struct"
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SpirvSyntaxError(
                    "Expected structured data block after 'uniform' or 'buffer'"
                )
        else:
            if layout_type in ["IN", "OUT"] and self.current_token[0] == "SEMICOLON":
                pass
            elif (
                layout_type in ["IN", "OUT"]
                and block_name
                and self.current_token[0] == "LBRACE"
            ):
                struct_fields = self.parse_layout_block_fields()
                data_type = "struct"
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SpirvSyntaxError(f"Unexpected type: {self.current_token[1]}")

        variable_name = None
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")
            variable_name += self.parse_array_suffixes_as_text()

        self.eat("SEMICOLON")
        return LayoutNode(
            bindings,
            push_constant=push_constant,
            layout_type=layout_type,
            data_type=data_type,
            variable_name=variable_name,
            struct_fields=struct_fields,
            block_name=block_name,
            declaration_qualifiers=declaration_qualifiers,
        )

    def parse_layout_qualifier_value(self):
        parts = []
        depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if depth == 0 and token_type in {"COMMA", "RPAREN"}:
                break

            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1

            parts.append(self.current_token[1])
            self.eat(token_type)

        if not parts:
            raise SpirvSyntaxError(
                f"Expected layout qualifier value, got {self.current_token[0]}"
            )
        if depth:
            raise SpirvSyntaxError("Unterminated layout qualifier value")

        return self.format_layout_qualifier_value(parts)

    def format_layout_qualifier_value(self, parts):
        text = " ".join(str(part) for part in parts)
        return (
            text.replace("( ", "(")
            .replace(" )", ")")
            .replace("[ ", "[")
            .replace(" ]", "]")
            .replace(" . ", ".")
            .replace(" ,", ",")
            .replace(", ", ", ")
        )

    def has_specialization_constant_qualifier(self, qualifiers):
        return any(str(name).lower() == "constant_id" for name, _ in qualifiers)

    def is_data_type_token(self, allow_identifier=False):
        return self.current_token[1] in VALID_DATA_TYPES or (
            allow_identifier and self.current_token[0] == "IDENTIFIER"
        )

    def is_data_type_token_at(self, index, allow_identifier=False):
        if index >= len(self.tokens):
            return False
        token_type, token_value = self.tokens[index]
        return token_value in VALID_DATA_TYPES or (
            allow_identifier and token_type == "IDENTIFIER"
        )

    def parse_data_type(self, allow_identifier=False, error_message=None):
        if not self.is_data_type_token(allow_identifier=allow_identifier):
            raise SpirvSyntaxError(
                error_message or f"Unexpected type: {self.current_token[1]}"
            )

        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        type_name += self.parse_type_template_suffix()
        type_name += self.parse_array_suffixes_as_text()
        return type_name

    def parse_type_template_suffix(self):
        if self.current_token[0] != "LESS_THAN":
            return ""

        self.eat("LESS_THAN")
        parts = []
        depth = 1
        while depth:
            if self.current_token[0] == "EOF":
                raise SpirvSyntaxError("Unterminated type template argument list")

            if self.current_token[0] == "LESS_THAN":
                depth += 1
                parts.append("<")
                self.eat("LESS_THAN")
                continue

            if self.current_token[0] == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(">")
                self.eat("GREATER_THAN")
                continue

            parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        return f"<{self.format_type_template_parts(parts)}>"

    def format_type_template_parts(self, parts):
        text = " ".join(str(part) for part in parts)
        return (
            text.replace(" ,", ",")
            .replace(", ", ", ")
            .replace("< ", "<")
            .replace(" >", ">")
        )

    def skip_type_template_suffix_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LESS_THAN":
            return index

        depth = 1
        index += 1
        while index < len(self.tokens) and depth:
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
            index += 1
        return index if depth == 0 else len(self.tokens)

    def skip_array_suffixes_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            depth = 1
            index += 1
            while index < len(self.tokens) and depth:
                token_type = self.tokens[index][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                index += 1
            if depth:
                return len(self.tokens)
        return index

    def skip_type_suffixes_at_pos(self, index):
        index = self.skip_type_template_suffix_at_pos(index)
        return self.skip_array_suffixes_at_pos(index)

    def is_geometry_input_primitive_qualifier_at(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][1] not in self.GEOMETRY_INPUT_PRIMITIVE_QUALIFIERS
        ):
            return False

        type_index = index + 1
        if not self.is_data_type_token_at(type_index, allow_identifier=True):
            return False

        declarator_index = self.skip_type_suffixes_at_pos(type_index + 1)
        return (
            declarator_index < len(self.tokens)
            and self.tokens[declarator_index][0] == "IDENTIFIER"
        )

    def parse_layout_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_parameter_qualifiers(self):
        qualifiers = []
        while (
            self.current_token[0] in self.PARAMETER_QUALIFIER_TOKENS
            or self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS
            or self.is_geometry_input_primitive_qualifier_at(self.pos)
        ):
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_layout_block_fields(self):
        self.eat("LBRACE")
        struct_fields = []

        while self.current_token[0] != "RBRACE":
            self.skip_hlsl_attributes()
            while self.current_token[0] == "LAYOUT":
                self.skip_layout_annotation()
            self.parse_layout_declaration_qualifiers()
            field_type = self.parse_data_type(
                allow_identifier=True,
                error_message="Expected some data type before an identifier",
            )
            field_name = self.current_token[1]
            self.eat("IDENTIFIER")
            field_name += self.parse_array_suffixes_as_text()
            self.parse_optional_hlsl_semantic()
            self.eat("SEMICOLON")
            struct_fields.append((field_type, field_name))

        self.eat("RBRACE")
        return struct_fields

    def skip_layout_annotation(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        depth = 1
        while depth and self.current_token[0] != "EOF":
            if self.current_token[0] == "LPAREN":
                depth += 1
            elif self.current_token[0] == "RPAREN":
                depth -= 1
            self.eat(self.current_token[0])
        if depth:
            raise SpirvSyntaxError("Unterminated layout annotation")

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            self.skip_hlsl_attributes()
            while self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
                self.eat(self.current_token[0])
            if self.current_token[0] in [
                "VEC2",
                "VEC3",
                "VEC4",
                "IVEC2",
                "IVEC3",
                "IVEC4",
                "UVEC2",
                "UVEC3",
                "UVEC4",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "MAT2",
                "MAT3",
                "MAT4",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[1] in VALID_DATA_TYPES:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[0] == "IDENTIFIER":
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SpirvSyntaxError(
                    f"Unexpected token in struct member: {self.current_token}"
                )
            type_name += self.parse_type_template_suffix()
            type_name += self.parse_array_suffixes_as_text()

            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_name += self.parse_array_suffixes_as_text()
            semantic = self.parse_optional_hlsl_semantic()

            self.eat("SEMICOLON")

            members.append(VariableNode(type_name, member_name, semantic=semantic))

        self.eat("RBRACE")
        while self.current_token[0] == "IDENTIFIER":
            self.eat("IDENTIFIER")
            self.parse_array_suffixes_as_text()
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        qualifiers = self.parse_declaration_qualifiers()
        return_type = self.parse_data_type(allow_identifier=True)
        func_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        semantic = self.parse_optional_hlsl_semantic()
        body = self.parse_block()
        return FunctionNode(
            return_type,
            func_name,
            params,
            body,
            qualifiers=qualifiers,
            semantic=semantic,
        )

    def parse_parameters(self):
        params = []
        if self.current_token[0] == "VOID" and self.peek(1) == "RPAREN":
            self.eat("VOID")
            return params

        while self.current_token[0] != "RPAREN":
            self.skip_hlsl_attributes()
            qualifiers = self.parse_parameter_qualifiers()
            vtype = self.parse_data_type(allow_identifier=True)
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            name += self.parse_array_suffixes_as_text()
            semantic = self.parse_optional_hlsl_semantic()
            params.append(
                VariableNode(vtype, name, qualifiers=qualifiers, semantic=semantic)
            )
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params
