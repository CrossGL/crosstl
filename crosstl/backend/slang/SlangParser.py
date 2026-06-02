"""Parser for Slang source AST construction."""

from .SlangAst import *
from .SlangLexer import *


class SlangParser:
    """Parse Slang tokens into the Slang backend AST."""

    DECLARATION_TYPE_TOKENS = {"FLOAT", "FVECTOR", "INT", "UINT", "BOOL", "MATRIX"}
    RESOURCE_TYPE_TOKENS = {"TEXTURE2D", "SAMPLER_STATE"}
    QUALIFIER_TOKENS = {"CONST", "CONSTEXPR", "INLINE", "STATIC"}
    IDENTIFIER_QUALIFIERS = {
        "uniform",
        "in",
        "out",
        "inout",
        "public",
        "internal",
        "private",
        "__global",
        "__extern_cpp",
        "groupshared",
        "globallycoherent",
        "flat",
        "highp",
        "mediump",
        "lowp",
        "no_diff",
        "nointerpolation",
        "noperspective",
        "override",
        "__prefix",
        "__postfix",
        "__ref",
        "__func_extension",
        "expand",
        "each",
    }
    OPERATOR_SYMBOL_TOKENS = {
        "PLUS",
        "MINUS",
        "MULTIPLY",
        "DIVIDE",
        "MOD",
        "EQUAL",
        "NOT_EQUAL",
        "LESS_THAN",
        "LESS_EQUAL",
        "GREATER_THAN",
        "GREATER_EQUAL",
        "AND",
        "OR",
        "NOT",
        "BITWISE_NOT",
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "ASSIGN_MOD",
        "ASSIGN_AND",
        "ASSIGN_OR",
        "ASSIGN_XOR",
        "BITWISE_SHIFT_LEFT",
        "ASSIGN_SHIFT_LEFT",
        "BITWISE_SHIFT_RIGHT",
        "ASSIGN_SHIFT_RIGHT",
        "INCREMENT",
        "DECREMENT",
        "BITWISE_AND",
        "BITWISE_OR",
        "BITWISE_XOR",
        "COMMA",
    }
    GEOMETRY_INPUT_PRIMITIVE_QUALIFIERS = {
        "point",
        "line",
        "triangle",
        "lineadj",
        "triangleadj",
    }
    MESH_OUTPUT_PARAMETER_QUALIFIERS = {"indices", "vertices", "primitives"}
    BUILTIN_IDENTIFIER_TYPES = {
        "double",
        "double2",
        "double3",
        "double4",
        "half",
        "half2",
        "half3",
        "half4",
        "int2",
        "int3",
        "int4",
        "uint2",
        "uint3",
        "uint4",
        "bool2",
        "bool3",
        "bool4",
    }
    TYPE_NAME_TOKENS = (
        DECLARATION_TYPE_TOKENS | RESOURCE_TYPE_TOKENS | {"IDENTIFIER", "VOID"}
    )
    TOP_LEVEL_DECLARATION_TOKENS = (
        TYPE_NAME_TOKENS | QUALIFIER_TOKENS | {"GENERIC", "TYPEALIAS"}
    )
    ASSIGNMENT_TOKENS = (
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "ASSIGN_MOD",
        "ASSIGN_AND",
        "ASSIGN_OR",
        "ASSIGN_XOR",
        "ASSIGN_SHIFT_LEFT",
        "ASSIGN_SHIFT_RIGHT",
    )

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        imports = []
        exports = []
        includes = []
        modules = []
        implementing_modules = []
        functions = []
        structs = []
        interfaces = []
        extensions = []
        typedefs = []
        enums = []
        cbuffers = []
        global_variables = []
        while self.current_token[0] != "EOF":
            pending_attributes = []
            while self.current_token[0] == "LBRACKET":
                pending_attributes.extend(self.parse_attribute_list())
            if self.current_token[0] == "EOF":
                break

            declaration_token = self.peek_declaration_token_type()

            if self.current_token[0] == "IMPORT":
                imports.append(self.parse_import())
            elif self.current_token[0] == "INCLUDE":
                includes.append(self.parse_include())
            elif self.current_token[0] == "MODULE":
                modules.append(self.parse_module_declaration())
            elif self.current_token[0] == "IMPLEMENTING":
                implementing_modules.append(self.parse_implementing_declaration())
            elif self.is_glsl_precision_declaration_start():
                self.parse_glsl_precision_declaration()
            elif self.current_token[0] == "EXPORT":
                exports.append(self.parse_export(attributes=pending_attributes))
            elif declaration_token == "INTERFACE":
                interfaces.append(self.parse_interface())
            elif declaration_token == "STRUCT":
                structs.append(self.parse_struct())
            elif declaration_token == "EXTENSION":
                extensions.append(self.parse_extension())
            elif declaration_token == "CBUFFER":
                cbuffers.append(self.parse_cbuffer(attributes=pending_attributes))
            elif self.is_glsl_uniform_block_declaration_start():
                cbuffers.append(
                    self.parse_glsl_uniform_block(attributes=pending_attributes)
                )
            elif declaration_token == "ENUM":
                enums.append(self.parse_enum())
            elif declaration_token in {"TYPEDEF", "TYPEALIAS"}:
                typedefs.append(self.parse_typedef())
            elif self.is_namespace_declaration_start():
                self.skip_namespace_declaration()
            elif self.current_token[0] in self.TOP_LEVEL_DECLARATION_TOKENS:
                if self.is_function():
                    functions.append(
                        self.parse_function(
                            attributes=pending_attributes,
                            allow_signature=True,
                        )
                    )
                else:
                    self.append_parsed_statement(
                        global_variables,
                        self.parse_global_variable(attributes=pending_attributes),
                    )
            else:
                self.eat(self.current_token[0])

        return ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            imports=imports,
            exports=exports,
            typedefs=typedefs,
            cbuffers=cbuffers,
            interfaces=interfaces,
            extensions=extensions,
            enums=enums,
            includes=includes,
            modules=modules,
            implementing_modules=implementing_modules,
        )

    def is_function(self):
        if self.is_function_extension_declaration_start():
            return True

        current_pos = self.skip_declaration_prefix_tokens(
            self.pos, include_generic=True
        )
        if current_pos >= len(self.tokens) or self.tokens[current_pos][0] == "EOF":
            return False

        current_pos += 1
        current_pos = self.skip_generic_type_suffix_tokens(current_pos)
        current_pos = self.skip_qualified_type_suffix_tokens(current_pos)
        current_pos = self.skip_type_array_suffix_tokens(current_pos)
        current_pos = self.skip_pointer_declarator_tokens(current_pos)
        if self.is_operator_identifier_at(current_pos):
            current_pos = self.skip_operator_function_name_tokens(current_pos)
        else:
            if self.tokens[current_pos][0] != "IDENTIFIER":
                return False
            current_pos += 1
        current_pos = self.skip_generic_type_suffix_tokens(current_pos)
        return self.tokens[current_pos][0] == "LPAREN"

    def is_function_extension_declaration_start(self):
        current_pos = self.skip_declaration_prefix_tokens(
            self.pos, include_generic=True
        )
        if current_pos + 1 >= len(self.tokens):
            return False
        return (
            self.tokens[current_pos][0] == "IDENTIFIER"
            and self.tokens[current_pos][1] in {"__apply", "fwd_diff", "bwd_diff"}
            and self.tokens[current_pos + 1][0] == "LPAREN"
        )

    def is_operator_identifier_at(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] == "operator"
        )

    def skip_operator_function_name_tokens(self, current_pos):
        if not self.is_operator_identifier_at(current_pos):
            return current_pos

        current_pos += 1
        if current_pos >= len(self.tokens):
            return current_pos

        token_type = self.tokens[current_pos][0]
        if (
            token_type == "LPAREN"
            and current_pos + 2 < len(self.tokens)
            and self.tokens[current_pos + 1][0] == "RPAREN"
            and self.tokens[current_pos + 2][0] == "LPAREN"
        ):
            return current_pos + 2
        if (
            token_type == "LBRACKET"
            and current_pos + 2 < len(self.tokens)
            and self.tokens[current_pos + 1][0] == "RBRACKET"
            and self.tokens[current_pos + 2][0] == "LPAREN"
        ):
            return current_pos + 2
        if token_type in self.OPERATOR_SYMBOL_TOKENS:
            return current_pos + 1
        return current_pos

    def is_namespace_declaration_start(self):
        return (
            self.current_token == ("IDENTIFIER", "namespace")
            and self.pos + 2 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "IDENTIFIER"
            and self.tokens[self.pos + 2][0] == "LBRACE"
        )

    def skip_namespace_declaration(self):
        self.eat("IDENTIFIER")
        self.eat("IDENTIFIER")
        self.skip_balanced_block()

    def is_glsl_precision_declaration_start(self):
        return (
            self.current_token == ("IDENTIFIER", "precision")
            and self.pos + 2 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "IDENTIFIER"
            and self.tokens[self.pos + 1][1] in {"highp", "mediump", "lowp"}
            and self.tokens[self.pos + 2][0] in self.TYPE_NAME_TOKENS
        )

    def parse_glsl_precision_declaration(self):
        self.eat("IDENTIFIER")
        self.eat("IDENTIFIER")
        self.parse_type_name()
        self.eat("SEMICOLON")

    def skip_balanced_block(self):
        self.eat("LBRACE")
        depth = 1
        while depth:
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated namespace block")
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1
            self.eat(self.current_token[0])

    def skip_declaration_prefix_tokens(self, current_pos, include_generic=False):
        while current_pos < len(self.tokens):
            if self.is_layout_qualifier_at(current_pos):
                current_pos = self.skip_layout_qualifier_tokens(current_pos)
                continue
            if self.is_qualifier_token_at(current_pos):
                current_pos += 1
            elif include_generic and self.tokens[current_pos][0] == "GENERIC":
                current_pos += 1
            else:
                break
            if (
                include_generic
                and current_pos < len(self.tokens)
                and self.tokens[current_pos][0] == "LESS_THAN"
            ):
                current_pos = self.skip_generic_type_suffix_tokens(current_pos)
        return current_pos

    def is_qualifier_token_at(self, index):
        token_type, token_value = self.tokens[index]
        return token_type in self.QUALIFIER_TOKENS or (
            token_type == "IDENTIFIER" and token_value in self.IDENTIFIER_QUALIFIERS
        )

    def is_layout_qualifier_at(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index] == ("IDENTIFIER", "layout")
            and self.tokens[index + 1][0] == "LPAREN"
        )

    def skip_layout_qualifier_tokens(self, current_pos):
        current_pos += 1
        if current_pos >= len(self.tokens) or self.tokens[current_pos][0] != "LPAREN":
            return current_pos

        depth = 0
        while current_pos < len(self.tokens):
            token_type = self.tokens[current_pos][0]
            if token_type == "EOF":
                raise SyntaxError("Unterminated layout qualifier")
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return current_pos + 1
            current_pos += 1

        raise SyntaxError("Unterminated layout qualifier")

    def peek_declaration_token_type(self):
        current_pos = self.skip_declaration_prefix_tokens(self.pos)
        if current_pos >= len(self.tokens):
            return "EOF"
        return self.tokens[current_pos][0]

    def parse_qualifiers(self):
        qualifiers = []
        while self.is_layout_qualifier_at(self.pos) or self.is_qualifier_token_at(
            self.pos
        ):
            if self.is_layout_qualifier_at(self.pos):
                qualifiers.append(self.parse_layout_qualifier())
            else:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])
        return qualifiers

    def parse_layout_qualifier(self):
        qualifier = [self.current_token[1]]
        self.eat("IDENTIFIER")
        qualifier.extend(self.parse_balanced_parenthesized_tokens("layout qualifier"))
        return "".join(qualifier)

    def parse_balanced_parenthesized_tokens(self, context):
        parts = []
        self.eat("LPAREN")
        parts.append("(")
        depth = 1
        while depth:
            token_type, token_value = self.current_token
            if token_type == "EOF":
                raise SyntaxError(f"Unterminated {context}")
            if token_type == "LPAREN":
                depth += 1
                parts.append("(")
                self.eat("LPAREN")
                continue
            if token_type == "RPAREN":
                depth -= 1
                parts.append(")")
                self.eat("RPAREN")
                continue
            parts.append(str(token_value))
            self.eat(token_type)
        return parts

    def is_geometry_input_primitive_qualifier_at(self, index):
        return self.is_parameter_role_qualifier_at(
            index, self.GEOMETRY_INPUT_PRIMITIVE_QUALIFIERS
        )

    def is_mesh_output_parameter_qualifier_at(self, index):
        return self.is_parameter_role_qualifier_at(
            index, self.MESH_OUTPUT_PARAMETER_QUALIFIERS
        )

    def is_parameter_role_qualifier_at(self, index, qualifier_names):
        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type != "IDENTIFIER" or token_value not in qualifier_names:
            return False

        type_pos = index + 1
        if (
            type_pos >= len(self.tokens)
            or self.tokens[type_pos][0] not in self.TYPE_NAME_TOKENS
        ):
            return False

        name_pos = type_pos + 1
        if name_pos >= len(self.tokens):
            return False
        name_pos = self.skip_generic_type_suffix_tokens(name_pos)
        name_pos = self.skip_type_array_suffix_tokens(name_pos)
        name_pos = self.skip_pointer_declarator_tokens(name_pos)
        return name_pos < len(self.tokens) and self.tokens[name_pos][0] == "IDENTIFIER"

    def parse_parameter_qualifiers(self):
        qualifiers = self.parse_qualifiers()
        while self.is_geometry_input_primitive_qualifier_at(self.pos):
            qualifiers.append(self.current_token[1])
            self.eat("IDENTIFIER")
        while self.is_mesh_output_parameter_qualifier_at(self.pos):
            qualifiers.append(self.current_token[1])
            self.eat("IDENTIFIER")
        return qualifiers

    def parse_declaration_prefixes(self):
        qualifiers = []
        is_generic = False
        while (
            self.current_token[0] == "GENERIC"
            or self.is_layout_qualifier_at(self.pos)
            or self.is_qualifier_token_at(self.pos)
        ):
            if self.current_token[0] == "GENERIC":
                is_generic = True
                self.eat("GENERIC")
                if self.current_token[0] == "LESS_THAN":
                    self.parse_generic_type_suffix()
            elif self.is_layout_qualifier_at(self.pos):
                qualifiers.append(self.parse_layout_qualifier())
            else:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])
        return qualifiers, is_generic

    def skip_generic_type_suffix_tokens(self, current_pos):
        if self.tokens[current_pos][0] != "LESS_THAN":
            return current_pos

        depth = 0
        while self.tokens[current_pos][0] != "EOF":
            token_type = self.tokens[current_pos][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return current_pos + 1
            elif token_type == "BITWISE_SHIFT_RIGHT":
                depth -= 2
                if depth == 0:
                    return current_pos + 1
                if depth < 0:
                    raise SyntaxError("Unterminated generic type suffix")
            current_pos += 1
        raise SyntaxError("Unterminated generic type suffix")

    def skip_type_array_suffix_tokens(self, current_pos):
        while (
            current_pos < len(self.tokens) and self.tokens[current_pos][0] == "LBRACKET"
        ):
            depth = 0
            while (
                current_pos < len(self.tokens) and self.tokens[current_pos][0] != "EOF"
            ):
                token_type = self.tokens[current_pos][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                    if depth == 0:
                        current_pos += 1
                        break
                current_pos += 1
            else:
                raise SyntaxError("Unterminated array type suffix")
        return current_pos

    def skip_qualified_type_suffix_tokens(self, current_pos):
        while current_pos < len(self.tokens):
            if self.tokens[current_pos][0] == "DOT":
                delimiter_width = 1
            elif (
                self.tokens[current_pos][0] == "COLON"
                and current_pos + 1 < len(self.tokens)
                and self.tokens[current_pos + 1][0] == "COLON"
            ):
                delimiter_width = 2
            else:
                break

            segment_pos = current_pos + delimiter_width
            if (
                segment_pos >= len(self.tokens)
                or self.tokens[segment_pos][0] not in self.TYPE_NAME_TOKENS
            ):
                break
            current_pos = self.skip_generic_type_suffix_tokens(segment_pos + 1)

        return current_pos

    def skip_pointer_declarator_tokens(self, current_pos):
        while (
            current_pos < len(self.tokens) and self.tokens[current_pos][0] == "MULTIPLY"
        ):
            current_pos += 1
        return current_pos

    def parse_attribute_list(self):
        attributes = []
        self.eat("LBRACKET")

        if self.current_token[0] == "LBRACKET":
            while self.current_token[0] == "LBRACKET":
                attributes.extend(self.parse_attribute_list())
            self.eat("RBRACKET")
            return attributes

        while self.current_token[0] != "RBRACKET":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated attribute list")

            if self.current_token[0] not in {"IDENTIFIER", "SHADER"}:
                self.eat(self.current_token[0])
                continue

            name = self.parse_attribute_name()
            arguments = []
            if self.current_token[0] == "LPAREN":
                arguments = self.parse_attribute_arguments()
            attributes.append({"name": name, "arguments": arguments})

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            while self.current_token[0] not in {"COMMA", "RBRACKET"}:
                if self.current_token[0] == "EOF":
                    raise SyntaxError("Unterminated attribute list")
                self.eat(self.current_token[0])

        self.eat("RBRACKET")
        return attributes

    def parse_attribute_name(self):
        name_tokens = {"IDENTIFIER", "SHADER"}
        if self.current_token[0] not in name_tokens:
            raise SyntaxError(f"Expected attribute name, got {self.current_token[0]}")

        parts = [self.current_token[1]]
        self.eat(self.current_token[0])
        while (
            self.pos + 2 < len(self.tokens)
            and self.current_token[0] == "COLON"
            and self.tokens[self.pos + 1][0] == "COLON"
            and self.tokens[self.pos + 2][0] in name_tokens
        ):
            self.eat("COLON")
            self.eat("COLON")
            parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        return "::".join(parts)

    def parse_attribute_arguments(self):
        arguments = []
        current_argument = []
        depth = 0

        self.eat("LPAREN")
        while not (self.current_token[0] == "RPAREN" and depth == 0):
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated attribute argument list")

            token_type, token_value = self.current_token
            if token_type == "COMMA" and depth == 0:
                arguments.append("".join(current_argument).strip())
                current_argument = []
                self.eat("COMMA")
                continue

            if token_type in {"LPAREN", "LBRACKET"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET"}:
                depth -= 1

            current_argument.append(str(token_value))
            self.eat(token_type)

        if current_argument:
            arguments.append("".join(current_argument).strip())
        self.eat("RPAREN")
        return arguments

    def get_numthreads_attribute(self, attributes):
        for attribute in attributes:
            if str(attribute.get("name", "")).lower() == "numthreads":
                arguments = attribute.get("arguments", [])
                if arguments:
                    normalized_arguments = list(arguments[:3])
                    while len(normalized_arguments) < 3:
                        normalized_arguments.append("1")
                    return tuple(normalized_arguments)
        return None

    def get_shader_attribute(self, attributes):
        for attribute in attributes:
            if str(attribute.get("name", "")).lower() == "shader":
                arguments = attribute.get("arguments", [])
                if arguments:
                    return str(arguments[0]).strip().strip("\"'")
        return None

    def filter_function_attributes(self, attributes):
        return [
            attribute
            for attribute in attributes
            if str(attribute.get("name", "")).lower() != "shader"
        ]

    def parse_type_name(self, allow_array_suffix=False):
        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        if self.current_token[0] == "LESS_THAN":
            type_name += self.parse_generic_type_suffix()
        type_name += self.parse_qualified_type_suffix()
        if allow_array_suffix and self.current_token[0] == "LBRACKET":
            type_name += self.parse_type_array_suffixes()
        return type_name

    def parse_qualified_type_suffix(self):
        suffix = ""
        while True:
            if self.current_token[0] == "DOT":
                delimiter = "."
                self.eat("DOT")
            elif (
                self.current_token[0] == "COLON"
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "COLON"
            ):
                delimiter = "::"
                self.eat("COLON")
                self.eat("COLON")
            else:
                break

            if self.current_token[0] not in self.TYPE_NAME_TOKENS:
                raise SyntaxError(
                    f"Expected qualified type segment, got {self.current_token[0]}"
                )
            suffix += f"{delimiter}{self.current_token[1]}"
            self.eat(self.current_token[0])
            if self.current_token[0] == "LESS_THAN":
                suffix += self.parse_generic_type_suffix()

        return suffix

    def parse_type_array_suffixes(self):
        suffix = ""
        while self.current_token[0] == "LBRACKET":
            suffix += "["
            self.eat("LBRACKET")
            depth = 1
            while depth > 0:
                token_type, token_value = self.current_token
                if token_type == "EOF":
                    raise SyntaxError("Unterminated array type suffix")
                if token_type == "LBRACKET":
                    depth += 1
                    suffix += "["
                    self.eat("LBRACKET")
                    continue
                if token_type == "RBRACKET":
                    depth -= 1
                    suffix += "]"
                    self.eat("RBRACKET")
                    continue
                suffix += str(token_value)
                self.eat(token_type)
        return suffix

    def parse_pointer_suffix(self):
        suffix = ""
        while self.current_token[0] == "MULTIPLY":
            suffix += "*"
            self.eat("MULTIPLY")
        return suffix

    def parse_generic_type_suffix(self):
        parts = ["<"]
        self.eat("LESS_THAN")
        depth = 1
        while depth > 0:
            token_type, token_value = self.current_token
            if token_type == "EOF":
                raise SyntaxError("Unterminated generic type suffix")
            if token_type == "LESS_THAN":
                depth += 1
                parts.append("<")
            elif token_type == "GREATER_THAN":
                depth -= 1
                parts.append(">")
            elif token_type == "BITWISE_SHIFT_RIGHT":
                if depth < 2:
                    raise SyntaxError("Unterminated generic type suffix")
                depth -= 2
                parts.append(">>")
            elif token_type == "COMMA":
                parts.append(", ")
            else:
                parts.append(str(token_value))
            self.eat(token_type)
        return "".join(parts)

    def parse_register_annotation(self):
        if self.current_token[0] != "COLON":
            return None
        if (
            self.pos + 1 >= len(self.tokens)
            or self.tokens[self.pos + 1][0] != "REGISTER"
        ):
            return None

        self.eat("COLON")
        self.eat("REGISTER")
        self.eat("LPAREN")
        register_parts = []
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated register annotation")
            register_parts.append(str(self.current_token[1]))
            self.eat(self.current_token[0])
        self.eat("RPAREN")
        return "".join(register_parts)

    def parse_semantic_annotations(self):
        annotations = []
        while self.current_token[0] == "COLON":
            self.eat("COLON")
            annotations.append(self.parse_semantic_annotation())
        if not annotations:
            return None
        return ":".join(annotations)

    def parse_semantic_annotation(self):
        if self.current_token[0] not in {"IDENTIFIER", "SHADER"}:
            raise SyntaxError(f"Expected semantic name, got {self.current_token[0]}")

        annotation = [str(self.current_token[1])]
        self.eat(self.current_token[0])
        if self.current_token[0] != "LPAREN":
            return "".join(annotation)

        depth = 0
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            annotation.append(str(token_value))
            self.eat(token_type)
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return "".join(annotation)

        raise SyntaxError("Unterminated semantic argument list")

    def parse_array_suffixes(self):
        sizes = []
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "RBRACKET":
                sizes.append(None)
            else:
                sizes.append(self.parse_expression())
            self.eat("RBRACKET")
        return sizes

    def parse_cbuffer(self, attributes=None):
        attributes = attributes or []
        qualifiers = self.parse_qualifiers()
        self.eat("CBUFFER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        register_name = self.parse_register_annotation()
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.extend(self.parse_struct_field_members())
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        node = StructNode(name, members)
        node.register = register_name
        node.attributes = attributes
        node.qualifiers = qualifiers
        return node

    def is_glsl_uniform_block_declaration_start(self):
        current_pos = self.pos
        saw_uniform = False
        while current_pos < len(self.tokens):
            if self.is_layout_qualifier_at(current_pos):
                current_pos = self.skip_layout_qualifier_tokens(current_pos)
                continue
            token_type, token_value = self.tokens[current_pos]
            if token_type == "IDENTIFIER" and token_value == "uniform":
                saw_uniform = True
                current_pos += 1
                continue
            if self.is_qualifier_token_at(current_pos):
                current_pos += 1
                continue
            break

        if not saw_uniform or current_pos + 1 >= len(self.tokens):
            return False
        if self.tokens[current_pos][0] not in self.TYPE_NAME_TOKENS:
            return False
        return self.tokens[current_pos + 1][0] == "LBRACE"

    def parse_glsl_uniform_block(self, attributes=None):
        attributes = attributes or []
        qualifiers = self.parse_qualifiers()
        name = self.parse_type_name()
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.extend(self.parse_struct_field_members())
        self.eat("RBRACE")

        instances = []
        while self.current_token[0] == "IDENTIFIER":
            instance_name = self.current_token[1]
            self.eat("IDENTIFIER")
            instances.append(
                VariableNode(
                    name,
                    instance_name,
                    qualifiers=qualifiers,
                    array_sizes=self.parse_array_suffixes(),
                    attributes=attributes,
                )
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        node = StructNode(name, members)
        node.qualifiers = qualifiers
        node.attributes = attributes
        node.instances = instances
        return node

    def parse_global_variable(self, attributes=None):
        attributes = attributes or []
        qualifiers = self.parse_qualifiers()
        var_type = self.parse_type_name(allow_array_suffix=True)
        var_type += self.parse_pointer_suffix()
        declarations = []

        while True:
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            register_name = self.parse_register_annotation()
            semantic = None
            if self.current_token[0] == "COLON":
                semantic = self.parse_semantic_annotations()
            variable = VariableNode(
                var_type,
                var_name,
                qualifiers=qualifiers,
                array_sizes=array_sizes,
                attributes=attributes,
                register=register_name,
                semantic=semantic,
            )
            if self.current_token[0] == "EQUALS":
                op = self.current_token[1]
                self.eat("EQUALS")
                value = self.parse_expression()
                declarations.append(AssignmentNode(variable, value, op))
            else:
                declarations.append(variable)

            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        self.eat("SEMICOLON")
        if len(declarations) == 1:
            return declarations[0]
        return declarations

    def parse_import(self):
        self.eat("IMPORT")
        module_path = self.parse_module_path()
        self.eat("SEMICOLON")
        return ImportNode(module_path)

    def parse_include(self):
        self.eat("INCLUDE")
        include_path = self.parse_module_path()
        self.eat("SEMICOLON")
        return include_path

    def parse_module_declaration(self):
        self.eat("MODULE")
        module_path = self.parse_module_path(allow_string=False)
        self.eat("SEMICOLON")
        return module_path

    def parse_implementing_declaration(self):
        self.eat("IMPLEMENTING")
        module_path = self.parse_module_path(allow_string=False)
        self.eat("SEMICOLON")
        return module_path

    def parse_module_path(self, allow_string=True):
        if allow_string and self.current_token[0] == "STRING":
            module_path = self.current_token[1][1:-1]
            self.eat("STRING")
            return module_path

        parts = [self.current_token[1]]
        self.eat("IDENTIFIER")
        while self.current_token[0] == "DOT":
            self.eat("DOT")
            parts.append(self.current_token[1])
            self.eat("IDENTIFIER")
        return ".".join(parts)

    def parse_export(self, attributes=None):
        self.eat("EXPORT")
        declaration_token = self.peek_declaration_token_type()
        if declaration_token == "INTERFACE":
            exported_item = self.parse_interface()
        elif declaration_token == "STRUCT":
            exported_item = self.parse_struct()
        elif declaration_token == "EXTENSION":
            exported_item = self.parse_extension()
        elif self.is_function():
            exported_item = self.parse_function(attributes=attributes)
        elif self.is_variable_declaration_start():
            exported_item = self.parse_global_variable(attributes=attributes)
        else:
            raise SyntaxError(
                f"Expected export declaration, got {self.current_token[0]}"
            )
        return ExportNode(exported_item)

    def parse_interface(self):
        qualifiers = self.parse_qualifiers()
        self.eat("INTERFACE")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        generic_parameters = None
        if self.current_token[0] == "LESS_THAN":
            generic_parameters = self.parse_generic_type_suffix()
        conformances = self.parse_conformance_clause()
        generic_constraints = []
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()

        self.eat("LBRACE")
        methods = []
        associated_types = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated interface declaration")
            pending_attributes = []
            while self.current_token[0] == "LBRACKET":
                pending_attributes.extend(self.parse_attribute_list())
            if self.current_token[0] == "RBRACE":
                break
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            declaration_token = self.peek_declaration_token_type()
            if declaration_token == "ASSOCIATEDTYPE":
                associated_types.append(self.parse_associated_type())
                continue
            if not self.is_function():
                raise SyntaxError(
                    f"Unsupported interface member: {self.current_token[0]}"
                )
            methods.append(
                self.parse_function(attributes=pending_attributes, allow_signature=True)
            )
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        node = InterfaceNode(
            name,
            methods,
            generic_parameters=generic_parameters,
            associated_types=associated_types,
        )
        node.qualifiers = qualifiers
        node.conformances = conformances
        node.generic_constraints = generic_constraints
        return node

    def parse_struct(self):
        qualifiers = self.parse_qualifiers()
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        generic_parameters = None
        if self.current_token[0] == "LESS_THAN":
            generic_parameters = self.parse_generic_type_suffix()
        conformances = self.parse_conformance_clause()
        generic_constraints = []
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()
        self.eat("LBRACE")
        members = []
        methods = []
        typedefs = []
        enums = []
        structs = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated struct declaration")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            pending_attributes = []
            while self.current_token[0] == "LBRACKET":
                pending_attributes.extend(self.parse_attribute_list())
            if self.current_token[0] == "RBRACE":
                break
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            declaration_token = self.peek_declaration_token_type()
            if declaration_token in {"TYPEDEF", "TYPEALIAS"}:
                typedefs.append(self.parse_typedef())
                continue
            if declaration_token == "ENUM":
                enums.append(self.parse_enum())
                continue
            if declaration_token == "STRUCT":
                structs.append(self.parse_struct())
                continue
            if self.is_constructor():
                methods.append(
                    self.parse_constructor(
                        attributes=pending_attributes,
                        allow_signature=True,
                    )
                )
                continue
            if self.is_function():
                methods.append(
                    self.parse_function(
                        attributes=pending_attributes,
                        allow_signature=True,
                    )
                )
                continue
            members.extend(
                self.parse_struct_field_members(attributes=pending_attributes)
            )
        self.eat("RBRACE")
        node = StructNode(name, members)
        node.methods = methods
        node.typedefs = typedefs
        node.enums = enums
        node.structs = structs
        node.generic_parameters = generic_parameters
        node.generic_constraints = generic_constraints
        node.conformances = conformances
        node.qualifiers = qualifiers
        return node

    def parse_struct_field_members(self, attributes=None):
        attributes = attributes or []
        members = []
        qualifiers = self.parse_qualifiers()
        vtype = self.parse_type_name(allow_array_suffix=True)
        vtype += self.parse_pointer_suffix()
        while True:
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            semantic = None
            if self.current_token[0] == "COLON":
                semantic = self.parse_semantic_annotations()
            value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_expression()
            members.append(
                VariableNode(
                    vtype,
                    var_name,
                    value=value,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    array_sizes=array_sizes,
                    semantic=semantic,
                )
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        self.eat("SEMICOLON")
        return members

    def is_constructor(self):
        current_pos = self.skip_declaration_prefix_tokens(self.pos)
        if current_pos + 1 >= len(self.tokens):
            return False
        return (
            self.tokens[current_pos][0] == "IDENTIFIER"
            and self.tokens[current_pos][1] == "__init"
            and self.tokens[current_pos + 1][0] == "LPAREN"
        )

    def parse_constructor(self, attributes=None, allow_signature=False):
        attributes = attributes or []
        qualifiers, is_generic = self.parse_declaration_prefixes()
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        generic_constraints = []
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()

        if self.current_token[0] == "SEMICOLON":
            if not allow_signature:
                raise SyntaxError("Expected constructor body, got SEMICOLON")
            self.eat("SEMICOLON")
            body = []
            is_declaration = True
        else:
            body = self.parse_block()
            is_declaration = False

        return FunctionNode(
            "void",
            name,
            params,
            body,
            qualifiers=qualifiers,
            is_generic=is_generic,
            generic_constraints=generic_constraints,
            is_declaration=is_declaration,
            attributes=attributes,
        )

    def parse_extension(self):
        qualifiers = self.parse_qualifiers()
        self.eat("EXTENSION")
        generic_parameters = None
        generic_constraints = []
        if self.current_token[0] == "LESS_THAN":
            generic_parameters = self.parse_generic_type_suffix()
        extended_type = self.parse_type_name()
        conformances = self.parse_conformance_clause()
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()
        self.eat("LBRACE")
        methods = []
        typedefs = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated extension declaration")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            pending_attributes = []
            while self.current_token[0] == "LBRACKET":
                pending_attributes.extend(self.parse_attribute_list())
            if self.current_token[0] == "RBRACE":
                break
            declaration_token = self.peek_declaration_token_type()
            if declaration_token in {"TYPEDEF", "TYPEALIAS"}:
                typedefs.append(self.parse_typedef())
                continue
            if not self.is_function():
                raise SyntaxError(
                    f"Unsupported extension member: {self.current_token[0]}"
                )
            methods.append(
                self.parse_function(attributes=pending_attributes, allow_signature=True)
            )
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        node = ExtensionNode(extended_type, methods, conformances=conformances)
        node.qualifiers = qualifiers
        node.typedefs = typedefs
        node.generic_parameters = generic_parameters
        node.generic_constraints = generic_constraints
        node.is_generic = generic_parameters is not None
        return node

    def parse_conformance_clause(self):
        conformances = []
        if self.current_token[0] != "COLON":
            return conformances

        self.eat("COLON")
        while True:
            conformances.append(self.parse_type_name())
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        return conformances

    def parse_enum(self):
        self.eat("ENUM")
        enum_kind = None
        if self.current_token == ("IDENTIFIER", "class"):
            enum_kind = "class"
            self.eat("IDENTIFIER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated enum declaration")
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                member_value = self.parse_expression()
            members.append((member_name, member_value))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RBRACE":
                raise SyntaxError(
                    f"Expected comma or closing brace in enum, got {self.current_token[0]}"
                )
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        enum = EnumNode(name, members)
        enum.kind = enum_kind
        return enum

    def parse_typedef(self):
        qualifiers = self.parse_qualifiers()
        if self.current_token[0] == "TYPEDEF":
            self.eat("TYPEDEF")
            original_type = self.parse_type_name(allow_array_suffix=True)
            original_type += self.parse_pointer_suffix()
            new_type = self.current_token[1]
            self.eat("IDENTIFIER")
        else:
            self.eat("TYPEALIAS")
            new_type = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            original_type = self.parse_type_name(allow_array_suffix=True)
            original_type += self.parse_pointer_suffix()
        self.eat("SEMICOLON")
        node = TypedefNode(original_type, new_type)
        node.qualifiers = qualifiers
        return node

    def parse_associated_type(self):
        qualifiers = self.parse_qualifiers()
        self.eat("ASSOCIATEDTYPE")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        constraint_type = None
        target_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            constraint_type = self.parse_type_name()
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            target_type = self.parse_type_name(allow_array_suffix=True)
            target_type += self.parse_pointer_suffix()
        self.eat("SEMICOLON")
        return AssociatedTypeNode(
            name,
            constraint_type=constraint_type,
            target_type=target_type,
            qualifiers=qualifiers,
        )

    def parse_function(self, shader_type=None, attributes=None, allow_signature=False):
        if self.is_function_extension_declaration_start():
            return self.parse_function_extension(
                shader_type=shader_type,
                attributes=attributes,
                allow_signature=allow_signature,
            )

        attributes = attributes or []
        shader_type = shader_type or self.get_shader_attribute(attributes)
        attributes = self.filter_function_attributes(attributes)
        qualifiers, is_generic = self.parse_declaration_prefixes()
        return_type = self.parse_type_name(allow_array_suffix=True)
        return_type += self.parse_pointer_suffix()
        name = self.parse_function_name()
        generic_parameters = None
        if self.current_token[0] == "LESS_THAN":
            generic_parameters = self.parse_generic_type_suffix()
            is_generic = True
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        generic_constraints = []
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()
        semantic = None
        if self.current_token[0] == "COLON":
            semantic = self.parse_semantic_annotations()
        if self.current_token[0] == "WHERE":
            generic_constraints.extend(self.parse_generic_constraints())
        if self.current_token[0] == "SEMICOLON":
            if not allow_signature:
                raise SyntaxError("Expected function body, got SEMICOLON")
            self.eat("SEMICOLON")
            body = []
            is_declaration = True
        else:
            body = self.parse_block()
            is_declaration = False
        return FunctionNode(
            return_type,
            name,
            params,
            body,
            qualifiers=qualifiers,
            qualifier=shader_type,
            semantic=semantic,
            is_generic=is_generic,
            generic_parameters=generic_parameters,
            generic_constraints=generic_constraints,
            is_declaration=is_declaration,
            attributes=attributes,
            numthreads=self.get_numthreads_attribute(attributes),
        )

    def parse_function_extension(
        self, shader_type=None, attributes=None, allow_signature=False
    ):
        attributes = attributes or []
        shader_type = shader_type or self.get_shader_attribute(attributes)
        attributes = self.filter_function_attributes(attributes)
        qualifiers, is_generic, generic_parameters = (
            self.parse_function_extension_prefixes()
        )
        name = self.parse_function_extension_name()
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        self.eat("MINUS")
        self.eat("GREATER_THAN")
        return_type = self.parse_type_name(allow_array_suffix=True)
        return_type += self.parse_pointer_suffix()
        generic_constraints = []
        if self.current_token[0] == "WHERE":
            generic_constraints = self.parse_generic_constraints()
        if self.current_token[0] == "SEMICOLON":
            if not allow_signature:
                raise SyntaxError("Expected function body, got SEMICOLON")
            self.eat("SEMICOLON")
            body = []
            is_declaration = True
        else:
            body = self.parse_block()
            is_declaration = False
        return FunctionNode(
            return_type,
            name,
            params,
            body,
            qualifiers=qualifiers,
            qualifier=shader_type,
            is_generic=is_generic,
            generic_parameters=generic_parameters,
            generic_constraints=generic_constraints,
            is_declaration=is_declaration,
            attributes=attributes,
            numthreads=self.get_numthreads_attribute(attributes),
        )

    def parse_function_extension_prefixes(self):
        qualifiers = []
        is_generic = False
        generic_parameters = None
        while self.current_token[0] == "GENERIC" or self.is_qualifier_token_at(
            self.pos
        ):
            if self.current_token[0] == "GENERIC":
                is_generic = True
                self.eat("GENERIC")
                if self.current_token[0] == "LESS_THAN":
                    generic_parameters = self.parse_generic_type_suffix()
                continue

            qualifier = self.current_token[1]
            qualifiers.append(qualifier)
            self.eat(self.current_token[0])
            if qualifier == "__func_extension" and self.current_token[0] == "LESS_THAN":
                is_generic = True
                generic_parameters = self.parse_generic_type_suffix()

        return qualifiers, is_generic, generic_parameters

    def parse_function_extension_name(self):
        extension_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        target_name = self.parse_function_extension_target_name()
        self.eat("RPAREN")
        return f"{extension_name}({target_name})"

    def parse_function_extension_target_name(self):
        tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "RPAREN" and depth == 0:
                break
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
            tokens.append(str(token_value))
            self.eat(token_type)

        if not tokens:
            raise SyntaxError("Expected function extension target")
        return "".join(tokens)

    def parse_function_name(self):
        if self.current_token == ("IDENTIFIER", "operator"):
            return self.parse_operator_function_name()

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        return name

    def parse_operator_function_name(self):
        self.eat("IDENTIFIER")

        if (
            self.current_token[0] == "LPAREN"
            and self.pos + 2 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "RPAREN"
            and self.tokens[self.pos + 2][0] == "LPAREN"
        ):
            self.eat("LPAREN")
            self.eat("RPAREN")
            return "operator()"

        if (
            self.current_token[0] == "LBRACKET"
            and self.pos + 2 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "RBRACKET"
            and self.tokens[self.pos + 2][0] == "LPAREN"
        ):
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            return "operator[]"

        if self.current_token[0] in self.OPERATOR_SYMBOL_TOKENS:
            operator_name = f"operator{self.current_token[1]}"
            self.eat(self.current_token[0])
            return operator_name

        return "operator"

    def parse_generic_constraints(self):
        constraints = []
        self.eat("WHERE")
        while True:
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected generic parameter name, got {self.current_token[0]}"
                )
            parameter = self.parse_type_name()
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                relation = ":"
            elif self.current_token[0] == "EQUAL":
                self.eat("EQUAL")
                relation = "=="
            else:
                raise SyntaxError(
                    "Only simple generic conformance and equality constraints are supported"
                )
            constraint_type = self.parse_type_name()
            constraint = GenericConstraintNode(parameter, constraint_type)
            constraint.relation = relation
            constraints.append(constraint)
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        return constraints

    def parse_parameters(self):
        params = []
        if self.current_token[0] == "VOID" and self.tokens[self.pos + 1][0] == "RPAREN":
            self.eat("VOID")
            return params

        while self.current_token[0] != "RPAREN":
            struct_def = ""
            qualifiers = self.parse_parameter_qualifiers()
            vtype = self.parse_type_name(allow_array_suffix=True)
            vtype += self.parse_pointer_suffix()
            name = ""
            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            semantic = None
            if name and self.current_token[0] == "IDENTIFIER":
                struct_def = f" {self.current_token[1]}"
                self.eat("IDENTIFIER")
            if self.current_token[0] == "COLON":
                semantic = self.parse_semantic_annotations()
            params.append(
                VariableNode(
                    vtype + struct_def,
                    name,
                    qualifiers=qualifiers,
                    semantic=semantic,
                    array_sizes=array_sizes,
                )
            )
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        statements = []
        self.eat("LBRACE")
        while self.current_token[0] != "RBRACE":
            self.append_parsed_statement(statements, self.parse_statement())
        self.eat("RBRACE")
        return statements

    def parse_statement_or_block(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        return self.as_statement_list(self.parse_statement())

    def as_statement_list(self, statement):
        if isinstance(statement, list):
            return statement
        return [statement]

    def append_parsed_statement(self, statements, statement):
        statements.extend(self.as_statement_list(statement))

    def parse_statement(self):
        while self.current_token[0] == "LBRACKET":
            self.parse_attribute_list()

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return []
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        if self.current_token[0] in {"TYPEDEF", "TYPEALIAS"}:
            return self.parse_typedef()
        if self.is_variable_declaration_start():
            return self.parse_variable_declaration_or_assignment()
        if self.current_token[0] == "IDENTIFIER" and self.tokens[self.pos + 1][0] in {
            "LPAREN",
            "LBRACKET",
            "DOT",
            "COLON",
        }:
            return self.parse_expression_statement()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "DO":
            return self.parse_do_while_statement()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        elif self.current_token[0] == "BREAK":
            return self.parse_break_statement()
        elif self.current_token[0] == "CONTINUE":
            return self.parse_continue_statement()
        elif self.current_token[0] == "DISCARD":
            return self.parse_discard_statement()
        else:
            return self.parse_expression_statement()

    def is_variable_declaration_start(self):
        current_pos = self.skip_declaration_prefix_tokens(self.pos)
        if current_pos >= len(self.tokens):
            return False

        token_type = self.tokens[current_pos][0]
        if token_type not in (
            self.DECLARATION_TYPE_TOKENS | self.RESOURCE_TYPE_TOKENS | {"IDENTIFIER"}
        ):
            return False
        if token_type == "IDENTIFIER" and self.tokens[current_pos + 1][0] not in {
            "IDENTIFIER",
            "LBRACKET",
            "LESS_THAN",
            "DOT",
            "COLON",
        }:
            return False

        next_pos = self.skip_generic_type_suffix_tokens(current_pos + 1)
        next_pos = self.skip_qualified_type_suffix_tokens(next_pos)
        next_pos = self.skip_type_array_suffix_tokens(next_pos)
        next_pos = self.skip_pointer_declarator_tokens(next_pos)
        return next_pos < len(self.tokens) and self.tokens[next_pos][0] == "IDENTIFIER"

    def parse_variable_declaration_or_assignment(self):
        qualifiers = self.parse_qualifiers()
        var_type = self.parse_type_name(allow_array_suffix=True)
        var_type += self.parse_pointer_suffix()
        declarations = []

        while True:
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            variable = VariableNode(
                var_type,
                name,
                qualifiers=qualifiers,
                array_sizes=array_sizes,
            )

            if self.current_token[0] in self.ASSIGNMENT_TOKENS:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                value = self.parse_expression()
                declarations.append(AssignmentNode(variable, value, op))
            else:
                declarations.append(variable)

            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            if len(declarations) == 1:
                return declarations[0]
            return declarations

        raise SyntaxError(f"Unexpected token in declaration: {self.current_token[0]}")

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_or_block()
        elif self.current_token[0] == "ELSE_IF":
            else_body = self.parse_else_if_statement()
        return IfNode(condition, if_body, else_body)

    def parse_else_if_statement(self):
        self.eat("ELSE_IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_or_block()
        elif self.current_token[0] == "ELSE_IF":
            else_body = self.parse_else_if_statement()
        return IfNode(condition, if_body, else_body)

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        init = self.parse_for_initializer()

        condition = self.parse_for_condition()

        update = self.parse_for_update()
        self.eat("RPAREN")

        body = self.parse_statement_or_block()

        return ForNode(init, condition, update, body)

    def parse_for_initializer(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        if self.is_variable_declaration_start():
            return self.parse_variable_declaration_or_assignment()

        init = self.parse_expression()
        self.eat("SEMICOLON")
        return init

    def parse_for_condition(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        condition = self.parse_expression()
        self.eat("SEMICOLON")
        return condition

    def parse_for_update(self):
        if self.current_token[0] == "RPAREN":
            return None
        return self.parse_expression()

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement_or_block()
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_statement_or_block()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        ordered_cases = []
        default_case = None
        seen_default = False
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue

            if self.current_token[0] == "CASE":
                self.eat("CASE")
                value = self.parse_expression()
                self.eat("COLON")
                body = []
                while self.current_token[0] not in {"CASE", "DEFAULT", "RBRACE"}:
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")
                        continue
                    self.append_parsed_statement(body, self.parse_statement())
                case = CaseNode(value, body)
                cases.append(case)
                ordered_cases.append(case)
                continue

            if self.current_token[0] == "DEFAULT":
                if seen_default:
                    raise SyntaxError("duplicate default label in switch")
                seen_default = True
                self.eat("DEFAULT")
                self.eat("COLON")
                default_case = []
                while self.current_token[0] not in {"CASE", "RBRACE"}:
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")
                        continue
                    self.append_parsed_statement(default_case, self.parse_statement())
                ordered_cases.append(CaseNode(None, default_case))
                continue

            raise SyntaxError(f"Unexpected token in switch: {self.current_token[0]}")

        self.eat("RBRACE")
        switch = SwitchNode(expression, cases, default_case)
        switch.ordered_cases = ordered_cases
        return switch

    def parse_return_statement(self):
        self.eat("RETURN")
        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_break_statement(self):
        self.eat("BREAK")
        self.eat("SEMICOLON")
        return BreakNode()

    def parse_continue_statement(self):
        self.eat("CONTINUE")
        self.eat("SEMICOLON")
        return ContinueNode()

    def parse_discard_statement(self):
        self.eat("DISCARD")
        self.eat("SEMICOLON")
        return DiscardNode()

    def parse_expression_statement(self):
        expressions = [self.parse_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_expression())
        self.eat("SEMICOLON")
        return expressions if len(expressions) > 1 else expressions[0]

    def parse_expression(self):
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_ternary()
        if self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment()
            return AssignmentNode(left, right, op)
        return left

    def parse_ternary(self):
        left = self.parse_logical_or()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)
        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.current_token[0] == "OR":
            op = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_logical_and(self):
        left = self.parse_bitwise_or()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_or(self):
        left = self.parse_bitwise_xor()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_xor(self):
        left = self.parse_bitwise_and()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_and(self):
        left = self.parse_equality()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_relational(self):
        left = self.parse_shift()
        while self.current_token[0] in {
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        } or self.current_token in {("IDENTIFIER", "is"), ("IDENTIFIER", "as")}:
            if self.current_token[0] == "IDENTIFIER":
                op = self.current_token[1]
                self.eat("IDENTIFIER")
                right = self.parse_type_name()
            else:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                right = self.parse_shift()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_shift(self):
        left = self.parse_additive()
        while self.current_token[0] in ["BITWISE_SHIFT_LEFT", "BITWISE_SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_unary(self):
        if self.current_token[0] == "INCREMENT":
            self.eat("INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_unary())
        if self.current_token[0] == "DECREMENT":
            self.eat("DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_unary())
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] == "MULTIPLY":
            op = self.current_token[1]
            self.eat("MULTIPLY")
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token == ("IDENTIFIER", "no_diff"):
            self.eat("IDENTIFIER")
            return self.parse_unary()
        if self.current_token[0] == "LPAREN" and self.is_c_style_cast_start():
            return self.parse_c_style_cast()
        return self.parse_primary()

    def is_c_style_cast_start(self):
        if self.current_token[0] != "LPAREN":
            return False

        type_pos = self.pos + 1
        if type_pos >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[type_pos]
        if token_type == "IDENTIFIER":
            is_builtin_identifier_type = token_value in self.BUILTIN_IDENTIFIER_TYPES
            is_generic_identifier_type = self.tokens[type_pos + 1][0] == "LESS_THAN"
            if not is_builtin_identifier_type and not is_generic_identifier_type:
                close_pos = self.skip_pointer_declarator_tokens(type_pos + 1)
                if (
                    close_pos >= len(self.tokens)
                    or self.tokens[close_pos][0] != "RPAREN"
                ):
                    return False
                operand_pos = close_pos + 1
                if operand_pos >= len(self.tokens):
                    return False
                return self.tokens[operand_pos][0] in {"NUMBER", "LBRACE"}
        elif token_type not in (
            self.DECLARATION_TYPE_TOKENS | self.RESOURCE_TYPE_TOKENS | {"VOID"}
        ):
            return False

        try:
            close_pos = self.skip_generic_type_suffix_tokens(type_pos + 1)
        except SyntaxError:
            return False
        close_pos = self.skip_pointer_declarator_tokens(close_pos)
        if close_pos >= len(self.tokens) or self.tokens[close_pos][0] != "RPAREN":
            return False

        operand_pos = close_pos + 1
        if operand_pos >= len(self.tokens):
            return False

        return self.tokens[operand_pos][0] in {
            "IDENTIFIER",
            "NUMBER",
            "LPAREN",
            "LBRACE",
            "INT",
            "FLOAT",
            "FVECTOR",
            "MATRIX",
            "UINT",
            "BOOL",
            "INCREMENT",
            "DECREMENT",
            "PLUS",
            "MINUS",
            "BITWISE_NOT",
            "NOT",
        }

    def parse_c_style_cast(self):
        self.eat("LPAREN")
        target_type = self.parse_type_name()
        target_type += self.parse_pointer_suffix()
        self.eat("RPAREN")
        return CastNode(target_type, self.parse_unary())

    def parse_primary(self):
        if self.current_token[0] in [
            "IDENTIFIER",
            "INT",
            "FLOAT",
            "FVECTOR",
            "MATRIX",
            "UINT",
            "BOOL",
            "GENERIC",
        ]:
            if self.current_token[0] in [
                "INT",
                "FLOAT",
                "FVECTOR",
                "MATRIX",
                "UINT",
                "BOOL",
                "GENERIC",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
                if self.current_token[0] == "LBRACKET":
                    type_name += self.parse_type_array_suffixes()
                if self.current_token[0] == "IDENTIFIER":
                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    if self.current_token[0] == "LBRAKET":
                        self.eat("LBRAKET")
                        if self.current_token[0] == "NUMBER":
                            index = self.current_token[1]
                        else:
                            index = self.current_token[1]
                            self.eat("IDENTIFIER")
                        self.eat("RBRAKET")
                        return VariableNode(type_name, f"{var_name}[{index}]")
                    return VariableNode(type_name, var_name)
                elif self.current_token[0] == "LPAREN":
                    return self.parse_vector_constructor(type_name)
                elif self.current_token[0] == "LBRACE":
                    return self.parse_vector_constructor(
                        type_name,
                        open_token="LBRACE",
                        close_token="RBRACE",
                    )
                elif self.current_token[0] == "DOT":
                    return self.parse_postfix_suffixes(VariableNode("", type_name))
            return self.parse_function_call_or_identifier()

        if self.current_token[0] == "LBRAKET":
            self.eat("LBRAKET")
            expr = self.parse_expression()
            self.eat("RBRAKET")
            return expr

        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()

        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return self.parse_postfix_suffixes(value)
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            return value
        elif self.current_token[0] == "LPAREN":
            if self.is_lambda_expression_start():
                return self.parse_lambda_expression()
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return self.parse_postfix_suffixes(expr)
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_initializer_list(self):
        elements = []
        self.eat("LBRACE")
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated initializer list")
            elements.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RBRACE":
                raise SyntaxError(
                    f"Expected COMMA or RBRACE in initializer list, got {self.current_token[0]}"
                )
        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_vector_constructor(
        self,
        type_name,
        open_token="LPAREN",
        close_token="RPAREN",
    ):
        self.eat(open_token)
        args = []
        while self.current_token[0] != close_token:
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != close_token:
                raise SyntaxError(
                    f"Expected COMMA or {close_token}, got {self.current_token[0]}"
                )
        self.eat(close_token)
        return VectorConstructorNode(type_name, args)

    def is_lambda_expression_start(self):
        if self.current_token[0] != "LPAREN":
            return False

        depth = 0
        index = self.pos
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    next_index = index + 1
                    return (
                        next_index < len(self.tokens)
                        and self.tokens[next_index][0] == "FAT_ARROW"
                    )
            elif token_type == "EOF":
                return False
            index += 1

        return False

    def parse_lambda_expression(self):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_lambda_parameter())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN in lambda parameters, got {self.current_token[0]}"
                )
        self.eat("RPAREN")
        self.eat("FAT_ARROW")

        if self.current_token[0] == "LBRACE":
            args.append(self.parse_lambda_block_body())
        else:
            args.append(self.parse_expression())
        return FunctionCallNode("lambda", args)

    def parse_lambda_parameter(self):
        tokens = []
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break

            if token_type == "LESS_THAN":
                angle_depth += 1
            elif token_type == "GREATER_THAN" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1

            tokens.append(self.current_token)
            self.eat(token_type)

        raw = self.format_lambda_raw_tokens(tokens).strip()
        if not raw:
            raise SyntaxError("Expected lambda parameter")

        parts = raw.rsplit(None, 1)
        if len(parts) == 1:
            return VariableNode("", raw)
        return VariableNode(parts[0], parts[1])

    def parse_lambda_block_body(self):
        tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token = self.current_token
            token_type = token[0]
            tokens.append(token)
            self.eat(token_type)
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                if depth == 0:
                    return self.format_lambda_raw_tokens(tokens)

        raise SyntaxError("Unterminated lambda block body")

    def format_lambda_raw_tokens(self, tokens):
        text = ""
        previous = None
        for token_type, value in tokens:
            value = str(value)
            if not text:
                text = "{ " if value == "{" else value
            elif value in {")", "]", ";", ",", ":"}:
                text = text.rstrip() + value
                if value in {";", ","}:
                    text += " "
            elif value == "}":
                if not text.endswith((" ", "{")):
                    text += " "
                text += value
            elif value in {"(", "[", "."}:
                text = text.rstrip() + value
            elif value == "{":
                if not text.endswith(" "):
                    text += " "
                text += value + " "
            elif previous and previous[1] in {"(", "[", "."}:
                text += value
            elif text.endswith((" ", "{")):
                text += value
            else:
                text += " " + value
            previous = (token_type, value)
        return text.strip()

    def parse_function_call_or_identifier(self):
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        while True:
            if (
                self.current_token[0] == "LESS_THAN"
                and self.is_generic_expression_suffix()
            ):
                name += self.parse_generic_type_suffix()
                continue
            if (
                self.current_token[0] == "COLON"
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "COLON"
                and self.pos + 2 < len(self.tokens)
                and self.tokens[self.pos + 2][0] == "IDENTIFIER"
            ):
                self.eat("COLON")
                self.eat("COLON")
                name += f"::{self.current_token[1]}"
                self.eat("IDENTIFIER")
                continue
            break
        if self.current_token[0] == "LBRACKET" and self.is_array_constructor_suffix():
            name += self.parse_type_array_suffixes()
            return self.parse_vector_constructor(name)
        if self.current_token[0] == "LPAREN":
            node = self.parse_function_call(name)
        else:
            node = VariableNode("", name)
        return self.parse_postfix_suffixes(node)

    def is_array_constructor_suffix(self):
        index = self.pos
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    return (
                        index + 1 < len(self.tokens)
                        and self.tokens[index + 1][0] == "LPAREN"
                    )
            elif token_type == "EOF":
                return False
            index += 1
        return False

    def is_generic_expression_suffix(self):
        try:
            suffix_end = self.skip_generic_type_suffix_tokens(self.pos)
        except SyntaxError:
            return False
        if suffix_end >= len(self.tokens):
            return False
        if self.tokens[suffix_end][0] == "COLON":
            return (
                suffix_end + 1 < len(self.tokens)
                and self.tokens[suffix_end + 1][0] == "COLON"
            )
        return self.tokens[suffix_end][0] in {
            "COMMA",
            "DOT",
            "LPAREN",
            "RPAREN",
            "SEMICOLON",
        }

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                )
        self.eat("RPAREN")
        return args

    def parse_call(self, callee):
        args = self.parse_call_arguments()
        if isinstance(callee, VariableNode):
            return FunctionCallNode(callee.name, args)
        if isinstance(callee, MemberAccessNode):
            return MethodCallNode(callee.object, callee.member, args)
        return CallNode(callee, args)

    def parse_postfix_suffixes(self, node):
        while True:
            if self.current_token[0] == "LPAREN":
                node = self.parse_call(node)
                continue

            if self.current_token[0] == "DOT":
                self.eat("DOT")
                if self.current_token[0] != "IDENTIFIER":
                    raise SyntaxError(
                        f"Expected identifier after dot, got {self.current_token[0]}"
                    )
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                if (
                    self.current_token[0] == "LESS_THAN"
                    and self.is_generic_expression_suffix()
                ):
                    member += self.parse_generic_type_suffix()
                node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue

            if self.current_token[0] == "INCREMENT":
                self.eat("INCREMENT")
                return UnaryOpNode("POST_INCREMENT", self.valid_postfix_update(node))

            if self.current_token[0] == "DECREMENT":
                self.eat("DECREMENT")
                return UnaryOpNode("POST_DECREMENT", self.valid_postfix_update(node))

            return node

    def valid_postfix_update(self, node):
        if isinstance(node, (VariableNode, MemberAccessNode, ArrayAccessNode)):
            return node
        raise SyntaxError(f"Invalid postfix update target: {type(node).__name__}")

    def parse_function_call(self, name):
        return FunctionCallNode(name, self.parse_call_arguments())

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)
