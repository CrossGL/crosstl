"""Parser for DirectX HLSL source AST construction."""

import re

from ..common_ast import (
    ArrayAccessNode,
    AttributeNode,
    BreakNode,
    CastNode,
    ContinueNode,
    InitializerListNode,
    PreprocessorNode,
    TextureSampleNode,
)
from .DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    CaseNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    IncludeNode,
    MemberAccessNode,
    PragmaNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)

TYPE_TOKENS = {
    "FLOAT",
    "HALF",
    "DOUBLE",
    "INT",
    "UINT",
    "BOOL",
    "VOID",
    "DWORD",
    "MIN16FLOAT",
    "MIN10FLOAT",
    "MIN16INT",
    "MIN12INT",
    "MIN16UINT",
    "INT64_T",
    "UINT64_T",
    "FVECTOR",
    "IVECTOR",
    "UVECTOR",
    "BVECTOR",
    "MATRIX",
    "TEXTURE1D",
    "TEXTURE1DARRAY",
    "TEXTURE2D",
    "TEXTURE2DARRAY",
    "TEXTURE2DMS",
    "TEXTURE2DMSARRAY",
    "TEXTURE3D",
    "TEXTURECUBE",
    "TEXTURECUBEARRAY",
    "FEEDBACKTEXTURE2D",
    "FEEDBACKTEXTURE2DARRAY",
    "RWTEXTURE1D",
    "RWTEXTURE1DARRAY",
    "RWTEXTURE2D",
    "RWTEXTURE2DARRAY",
    "RWTEXTURE2DMS",
    "RWTEXTURE2DMSARRAY",
    "RWTEXTURE3D",
    "RWTEXTURECUBE",
    "RWTEXTURECUBEARRAY",
    "RASTERIZERORDEREDTEXTURE1D",
    "RASTERIZERORDEREDTEXTURE1DARRAY",
    "RASTERIZERORDEREDTEXTURE2D",
    "RASTERIZERORDEREDTEXTURE2DARRAY",
    "RASTERIZERORDEREDTEXTURE3D",
    "RASTERIZERORDEREDBUFFER",
    "RASTERIZERORDEREDSTRUCTUREDBUFFER",
    "RASTERIZERORDEREDBYTEADDRESSBUFFER",
    "STRUCTUREDBUFFER",
    "RWSTRUCTUREDBUFFER",
    "APPENDSTRUCTUREDBUFFER",
    "CONSUMESTRUCTUREDBUFFER",
    "BYTEADDRESSBUFFER",
    "RWBYTEADDRESSBUFFER",
    "RAYTRACING_ACCELERATION_STRUCTURE",
    "RAYQUERY",
    "BUFFER",
    "RWBUFFER",
    "SAMPLER_STATE",
    "SAMPLER_COMPARISON_STATE",
    "INPUTPATCH",
    "OUTPUTPATCH",
    "POINTSTREAM",
    "LINESTREAM",
    "TRIANGLESTREAM",
    "IDENTIFIER",
}

QUALIFIER_TOKENS = {
    "STATIC",
    "CONST",
    "INLINE",
    "EXTERN",
    "EXPORT",
    "VOLATILE",
    "PRECISE",
    "GLOBALLYCOHERENT",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "NOINTERPOLATION",
    "LINEAR",
    "CENTROID",
    "SAMPLE",
    "IN",
    "OUT",
    "INOUT",
    "UNIFORM",
    "GROUPSHARED",
}

ASSIGNMENT_TOKENS = {
    "EQUALS",
    "PLUS_EQUALS",
    "MINUS_EQUALS",
    "MULTIPLY_EQUALS",
    "DIVIDE_EQUALS",
    "MOD_EQUALS",
    "ASSIGN_AND",
    "ASSIGN_OR",
    "ASSIGN_XOR",
    "ASSIGN_SHIFT_LEFT",
    "ASSIGN_SHIFT_RIGHT",
}


class HLSLParser:
    """Parse HLSL tokens into the DirectX backend shader AST."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else ("EOF", "")
        self.synthetic_structs = []

    def parse(self):
        structs = []
        functions = []
        global_variables = []
        cbuffers = []
        enums = []
        typedefs = []

        while self.current_token[0] != "EOF":
            if self.current_token_is_keyword("USING", "using"):
                self.parse_using_directive()
                continue

            if self.current_token_is_keyword("NAMESPACE", "namespace"):
                namespace_ast = self.parse_namespace_block()
                structs.extend(namespace_ast.structs)
                functions.extend(namespace_ast.functions)
                global_variables.extend(namespace_ast.global_variables)
                cbuffers.extend(getattr(namespace_ast, "cbuffers", []) or [])
                enums.extend(getattr(namespace_ast, "enums", []) or [])
                typedefs.extend(getattr(namespace_ast, "typedefs", []) or [])
                continue

            if self.current_token[0] == "PREPROCESSOR":
                directive = self.parse_preprocessor_directive()
                if directive is not None:
                    structs.append(directive)
                continue

            if self.current_token[0] == "STRUCT":
                synthetic_start = len(self.synthetic_structs)
                struct = self.parse_struct()
                structs.extend(self.synthetic_structs[synthetic_start:])
                structs.append(struct)
                continue

            if self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
                continue

            if self.current_token[0] == "TYPEDEF":
                typedefs.append(self.parse_typedef())
                continue

            attributes = self.parse_attribute_list()
            if self.current_token[0] in {"CBUFFER", "TBUFFER"}:
                cbuffers.append(self.parse_cbuffer(attributes=attributes))
                continue

            qualifiers = self.parse_qualifiers()

            if not self.is_type_token(self.current_token[0]):
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                else:
                    self.eat(self.current_token[0])
                continue

            return_type = self.parse_type()
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected identifier after type, got {self.current_token[0]}"
                )

            name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "LPAREN":
                func = self.parse_function(return_type, name, qualifiers, attributes)
                functions.append(func)
            else:
                declarations = self.parse_variable_declaration_list_rest(
                    return_type,
                    name,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                    consume_semicolon=True,
                )
                global_variables.extend(declarations)

        return ShaderNode(
            includes=[],
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            cbuffers=cbuffers,
            enums=enums,
            typedefs=typedefs,
        )

    def eat(self, expected_type):
        if self.current_token[0] == expected_type:
            token = self.current_token
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")
            return token
        raise SyntaxError(f"Expected {expected_type}, got {self.current_token[0]}")

    def peek(self, offset=1):
        idx = self.current_index + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return ("EOF", "")

    def is_type_token(self, token_type):
        return token_type in TYPE_TOKENS

    def current_token_is_keyword(self, token_type, value):
        return self.current_token[0] == token_type or (
            self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value
        )

    def current_token_is_double_colon(self):
        return self.current_token[0] == "COLON" and self.peek()[0] == "COLON"

    def eat_keyword(self, token_type, value):
        if self.current_token[0] == token_type:
            return self.eat(token_type)
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value:
            return self.eat("IDENTIFIER")
        raise SyntaxError(f"Expected {value}, got {self.current_token[0]}")

    def eat_double_colon(self):
        self.eat("COLON")
        self.eat("COLON")

    def parse_using_directive(self):
        self.eat_keyword("USING", "using")
        if self.current_token_is_keyword("NAMESPACE", "namespace"):
            self.eat_keyword("NAMESPACE", "namespace")
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            if self.current_token_is_double_colon():
                self.eat_double_colon()
            else:
                self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_namespace_block(self):
        self.eat_keyword("NAMESPACE", "namespace")
        if self.current_token[0] == "IDENTIFIER":
            self.eat("IDENTIFIER")
            while self.current_token_is_double_colon():
                self.eat_double_colon()
                self.eat("IDENTIFIER")

        self.eat("LBRACE")
        namespace_tokens = []
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                depth += 1
                namespace_tokens.append(self.current_token)
                self.eat("LBRACE")
                continue
            if self.current_token[0] == "RBRACE":
                depth -= 1
                if depth == 0:
                    break
                namespace_tokens.append(self.current_token)
                self.eat("RBRACE")
                continue
            namespace_tokens.append(self.current_token)
            self.eat(self.current_token[0])

        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        parser = HLSLParser(namespace_tokens + [("EOF", "")])
        return parser.parse()

    def parse_preprocessor_directive(self):
        token = self.eat("PREPROCESSOR")
        text = token[1].strip()

        if text.startswith("#pragma"):
            parts = text.split(None, 2)
            directive = parts[1] if len(parts) > 1 else ""
            value = parts[2] if len(parts) > 2 else None
            return PragmaNode(directive, value)

        if text.startswith("#include"):
            match = re.search(r"#include\s*([<\"])([^>\"]+)[>\"]", text)
            if match:
                path = match.group(2)
                is_system = match.group(1) == "<"
                return IncludeNode(path, is_system)
            path = text[len("#include") :].strip()
            return IncludeNode(path, False)

        directive = text[1:].split(None, 1)[0] if text.startswith("#") else text
        content = text[len(directive) + 1 :].strip() if directive else text
        return PreprocessorNode(directive, content)

    def parse_attribute_list(self):
        attributes = []
        while self.current_token[0] == "LBRACKET":
            is_double_bracket = self.peek()[0] == "LBRACKET"
            self.eat("LBRACKET")

            if is_double_bracket:
                self.eat("LBRACKET")

            while self.current_token[0] != "EOF" and not self.is_attribute_list_end(
                is_double_bracket
            ):
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue

                if self.current_token[0] != "IDENTIFIER":
                    self.eat(self.current_token[0])
                    continue

                name = self.parse_attribute_name()
                args = []
                if self.current_token[0] == "LPAREN":
                    self.eat("LPAREN")
                    if self.current_token[0] != "RPAREN":
                        args.append(self.parse_expression())
                        while self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                            args.append(self.parse_expression())
                    self.eat("RPAREN")

                attributes.append(AttributeNode(name, args))

                while not self.is_attribute_list_end(
                    is_double_bracket
                ) and self.current_token[0] not in {"COMMA", "EOF"}:
                    self.eat(self.current_token[0])

            if self.current_token[0] != "RBRACKET":
                break
            self.eat("RBRACKET")
            if is_double_bracket:
                self.eat("RBRACKET")

        return attributes

    def is_attribute_list_end(self, is_double_bracket):
        if self.current_token[0] != "RBRACKET":
            return False
        return not is_double_bracket or self.peek()[0] == "RBRACKET"

    def parse_attribute_name(self):
        parts = [self.current_token[1]]
        self.eat("IDENTIFIER")

        while (
            self.current_token[0] == "COLON"
            and self.peek()[0] == "COLON"
            and self.peek(2)[0] == "IDENTIFIER"
        ):
            self.eat("COLON")
            self.eat("COLON")
            parts.append(self.current_token[1])
            self.eat("IDENTIFIER")

        return "::".join(parts)

    def parse_qualifiers(self):
        qualifiers = []
        while self.current_token[0] in QUALIFIER_TOKENS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_type(self):
        if not self.is_type_token(self.current_token[0]):
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")

        base = self.current_token[1]
        self.eat(self.current_token[0])
        type_name = base

        if self.current_token[0] == "LESS_THAN":
            args = self.parse_generic_arguments()
            type_name = f"{base}<{', '.join(args)}>"

        return type_name

    def parse_generic_arguments(self):
        args = []
        self.eat("LESS_THAN")
        while (
            self.current_token[0] != "GREATER_THAN" and self.current_token[0] != "EOF"
        ):
            if self.is_type_token(self.current_token[0]):
                args.append(self.parse_type())
            else:
                args.append(self.current_token[1])
                self.eat(self.current_token[0])
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("GREATER_THAN")
        return args

    def parse_array_suffixes(self):
        sizes = []
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] != "RBRACKET":
                sizes.append(self.parse_expression())
            else:
                sizes.append(None)
            self.eat("RBRACKET")
        return sizes

    def parse_semantic_or_register(self):
        semantic = None
        register = None
        packoffset = None

        if self.current_token[0] != "COLON":
            return semantic, register, packoffset

        self.eat("COLON")
        if self.current_token[0] == "REGISTER":
            register = self.parse_register_binding("REGISTER")
            return semantic, register, packoffset
        if self.current_token[0] == "PACKOFFSET":
            packoffset = self.parse_register_binding("PACKOFFSET")
            return semantic, register, packoffset

        semantic = self.current_token[1]
        self.eat("IDENTIFIER")
        return semantic, register, packoffset

    def parse_register_binding(self, token_type):
        self.eat(token_type)
        self.eat("LPAREN")
        parts = []
        while self.current_token[0] != "RPAREN" and self.current_token[0] != "EOF":
            if self.current_token[0] == "COMMA":
                parts.append(", ")
                self.eat("COMMA")
                continue
            parts.append(self.current_token[1])
            self.eat(self.current_token[0])
        self.eat("RPAREN")
        return "".join(str(part) for part in parts).strip()

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        semantic = None
        if self.current_token[0] == "COLON":
            semantic, _, _ = self.parse_semantic_or_register()

        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            attributes = self.parse_attribute_list()
            qualifiers = self.parse_qualifiers()
            if self.current_token[0] == "STRUCT":
                declarations = self.parse_nested_struct_member(
                    name,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                )
                members.extend(self.ensure_statement_list(declarations))
                continue
            if not self.is_type_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected type in struct member, got {self.current_token[0]}"
                )
            declarations = self.parse_variable_declaration(
                qualifiers=qualifiers,
                attributes=attributes,
                allow_semantic=True,
                consume_semicolon=True,
            )
            members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")

        variables = []
        if self.current_token[0] == "IDENTIFIER":
            variables.append(self.current_token[1])
            self.eat("IDENTIFIER")
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                variables.append(self.current_token[1])
                self.eat("IDENTIFIER")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        struct_node = StructNode(name, members)
        struct_node.variables = variables
        struct_node.semantic = semantic
        return struct_node

    def parse_nested_struct_member(
        self,
        parent_name,
        qualifiers=None,
        attributes=None,
        allow_semantic=True,
    ):
        self.eat("STRUCT")
        nested_name = None
        if self.current_token[0] == "IDENTIFIER" and self.peek()[0] == "LBRACE":
            nested_name = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("LBRACE")
        nested_members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            member_attributes = self.parse_attribute_list()
            member_qualifiers = self.parse_qualifiers()
            if self.current_token[0] == "STRUCT":
                declarations = self.parse_nested_struct_member(
                    nested_name or parent_name,
                    qualifiers=member_qualifiers,
                    attributes=member_attributes,
                    allow_semantic=True,
                )
                nested_members.extend(self.ensure_statement_list(declarations))
                continue
            if not self.is_type_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected type in struct member, got {self.current_token[0]}"
                )
            declarations = self.parse_variable_declaration(
                qualifiers=member_qualifiers,
                attributes=member_attributes,
                allow_semantic=True,
                consume_semicolon=True,
            )
            nested_members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")

        if self.current_token[0] != "IDENTIFIER":
            if nested_name is None:
                raise SyntaxError("Expected identifier after anonymous struct member")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            self.synthetic_structs.append(StructNode(nested_name, nested_members))
            return []

        first_name = self.current_token[1]
        self.eat("IDENTIFIER")
        struct_type = nested_name or self.synthetic_struct_type_name(
            parent_name, first_name
        )
        self.synthetic_structs.append(StructNode(struct_type, nested_members))
        return self.parse_variable_declaration_list_rest(
            struct_type,
            first_name,
            qualifiers=qualifiers,
            attributes=attributes,
            allow_semantic=allow_semantic,
            consume_semicolon=True,
        )

    def synthetic_struct_type_name(self, parent_name, member_name):
        base = re.sub(r"\W+", "_", f"{parent_name}_{member_name}").strip("_")
        if not base:
            base = "AnonymousStruct"
        existing = {getattr(struct, "name", None) for struct in self.synthetic_structs}
        if base not in existing:
            return base
        index = 1
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def parse_enum(self):
        self.eat("ENUM")
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "class":
            self.eat("IDENTIFIER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
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
                raise SyntaxError(
                    f"Expected comma or closing brace in enum, got {self.current_token[0]}"
                )
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return EnumNode(name, members)

    def parse_typedef(self):
        self.eat("TYPEDEF")
        alias_type = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return TypeAliasNode(alias_type, name)

    def parse_cbuffer(self, attributes=None):
        buffer_kind = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "COLON":
            _, cbuffer_register, cbuffer_packoffset = self.parse_semantic_or_register()
        else:
            cbuffer_register = None
            cbuffer_packoffset = None

        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            qualifiers = self.parse_qualifiers()
            declarations = self.parse_variable_declaration(
                qualifiers=qualifiers,
                attributes=[],
                allow_semantic=True,
                consume_semicolon=True,
            )
            members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        cbuffer_node = StructNode(name, members)
        cbuffer_node.is_cbuffer = True
        cbuffer_node.buffer_kind = buffer_kind
        cbuffer_node.is_tbuffer = buffer_kind == "tbuffer"
        cbuffer_node.register = cbuffer_register
        cbuffer_node.packoffset = cbuffer_packoffset
        cbuffer_node.attributes = attributes or []
        return cbuffer_node

    def parse_function(self, return_type, name, qualifiers, attributes):
        params = self.parse_parameters()

        semantic = None
        if self.current_token[0] == "COLON":
            semantic, _, _ = self.parse_semantic_or_register()

        is_prototype = False
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            body = []
            is_prototype = True
        else:
            body = self.parse_block()

        qualifier = self.infer_function_qualifier(name, attributes, params, semantic)

        function = FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=qualifiers + ([qualifier] if qualifier else []),
            attributes=attributes,
            qualifier=qualifier,
            semantic=semantic,
        )
        function.is_prototype = is_prototype
        return function

    def infer_function_qualifier(self, name, attributes, params, semantic):
        for attr in attributes:
            if attr.name.lower() == "shader" and attr.args:
                raw = attr.args[0]
                stage_name = raw
                if isinstance(raw, str):
                    stage_name = raw.strip().strip("\"'").lower()
                else:
                    stage_name = str(raw).lower()
                stage_map = {
                    "vertex": "vertex",
                    "pixel": "fragment",
                    "fragment": "fragment",
                    "compute": "compute",
                    "geometry": "geometry",
                    "hull": "tessellation_control",
                    "domain": "tessellation_evaluation",
                    "mesh": "mesh",
                    "amplification": "task",
                    "task": "task",
                    "raygeneration": "ray_generation",
                    "intersection": "ray_intersection",
                    "closesthit": "ray_closest_hit",
                    "anyhit": "ray_any_hit",
                    "miss": "ray_miss",
                    "callable": "ray_callable",
                }
                if stage_name in stage_map:
                    return stage_map[stage_name]
        name_lower = name.lower()
        if name_lower.startswith("vs"):
            return "vertex"
        if name_lower.startswith("ps"):
            return "fragment"
        if name_lower.startswith("cs"):
            return "compute"
        if name_lower.startswith("gs"):
            return "geometry"
        if name_lower.startswith("hs"):
            return "tessellation_control"
        if name_lower.startswith("ds"):
            return "tessellation_evaluation"
        if name_lower.startswith("ms"):
            return "mesh"
        if name_lower.startswith("as"):
            return "task"
        if any(attr.name == "numthreads" for attr in attributes):
            return "compute"
        if semantic:
            semantic_upper = semantic.upper()
            if semantic_upper.startswith("SV_TARGET"):
                return "fragment"
            if semantic_upper == "SV_POSITION":
                return "vertex"
        for param in params:
            if getattr(param, "semantic", None) == "SV_DispatchThreadID":
                return "compute"
        return None

    def parse_parameters(self):
        self.eat("LPAREN")
        params = []
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        mesh_parameter_roles = {
            "payload": "mesh_payload",
            "vertices": "vertices",
            "indices": "indices",
            "primitives": "primitives",
        }

        if self.current_token[0] != "RPAREN":
            while True:
                attributes = self.parse_attribute_list()
                qualifiers = self.parse_qualifiers()
                if (
                    self.current_token[0] == "IDENTIFIER"
                    and self.current_token[1] in primitive_qualifiers
                ):
                    attributes.append(
                        AttributeNode("primitive", [self.current_token[1]])
                    )
                    self.eat("IDENTIFIER")
                has_direction_qualifier = any(
                    str(qualifier).lower() in {"in", "out", "inout"}
                    for qualifier in qualifiers
                )
                while (
                    has_direction_qualifier
                    and self.current_token[0] == "IDENTIFIER"
                    and self.current_token[1].lower() in mesh_parameter_roles
                ):
                    role = self.current_token[1].lower()
                    attributes.append(AttributeNode(mesh_parameter_roles[role]))
                    self.eat("IDENTIFIER")
                if not self.is_type_token(self.current_token[0]):
                    raise SyntaxError(
                        f"Unexpected token in parameter list: {self.current_token[0]}"
                    )
                param_type = self.parse_type()

                if self.current_token[0] == "IDENTIFIER":
                    name = self.current_token[1]
                    self.eat("IDENTIFIER")
                else:
                    name = ""

                array_sizes = self.parse_array_suffixes()
                semantic, _, _ = self.parse_semantic_or_register()
                value = None
                if self.current_token[0] == "EQUALS":
                    self.eat("EQUALS")
                    value = self.parse_expression()

                param = VariableNode(
                    param_type,
                    name,
                    value=value,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    is_const="const" in qualifiers,
                    semantic=semantic,
                )
                param.array_sizes = array_sizes
                params.append(param)

                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                break

        self.eat("RPAREN")
        return params

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            stmt = self.parse_statement()
            if stmt is None:
                continue
            if isinstance(stmt, list):
                statements.extend(stmt)
            else:
                statements.append(stmt)
        self.eat("RBRACE")
        return statements

    def parse_statement(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        attributes = self.parse_attribute_list()

        if self.current_token[0] == "RETURN":
            self.eat("RETURN")
            value = None
            if self.current_token[0] != "SEMICOLON":
                value = self.parse_expression()
            self.eat("SEMICOLON")
            return self.attach_attributes(ReturnNode(value), attributes)

        if self.current_token[0] == "IF":
            return self.attach_attributes(self.parse_if_statement(), attributes)

        if self.current_token[0] == "FOR":
            return self.attach_attributes(self.parse_for_loop(), attributes)

        if self.current_token[0] == "WHILE":
            return self.attach_attributes(self.parse_while_loop(), attributes)

        if self.current_token[0] == "DO":
            return self.attach_attributes(self.parse_do_while_loop(), attributes)

        if self.current_token[0] == "SWITCH":
            return self.attach_attributes(self.parse_switch_statement(), attributes)

        if self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return self.attach_attributes(BreakNode(), attributes)

        if self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return self.attach_attributes(ContinueNode(), attributes)

        if self.current_token[0] == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return "discard"

        if self.current_token[0] == "LBRACE":
            return self.parse_block()

        if self.current_token[0] == "PREPROCESSOR":
            self.parse_preprocessor_directive()
            return None

        if self.looks_like_declaration():
            qualifiers = self.parse_qualifiers()
            var = self.parse_variable_declaration(
                qualifiers=qualifiers,
                attributes=attributes,
                allow_semantic=False,
                consume_semicolon=True,
            )
            return var

        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return self.attach_attributes(expr, attributes)

    def attach_attributes(self, node, attributes):
        if node is not None and attributes:
            existing = getattr(node, "attributes", []) or []
            node.attributes = existing + attributes
        return node

    def ensure_statement_list(self, stmt):
        if stmt is None:
            return []
        if isinstance(stmt, list):
            return stmt
        return [stmt]

    def looks_like_declaration(self):
        idx = self.current_index
        while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return False
        idx += 1
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            depth = 0
            while idx < len(self.tokens):
                if self.tokens[idx][0] == "LESS_THAN":
                    depth += 1
                elif self.tokens[idx][0] == "GREATER_THAN":
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] != "IDENTIFIER":
            return False
        return True

    def parse_variable_declaration(
        self,
        qualifiers=None,
        attributes=None,
        allow_semantic=True,
        consume_semicolon=True,
    ):
        qualifiers = qualifiers or []
        attributes = attributes or []
        vtype = self.parse_type()
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        declarations = self.parse_variable_declaration_list_rest(
            vtype,
            name,
            qualifiers=qualifiers,
            attributes=attributes,
            allow_semantic=allow_semantic,
            consume_semicolon=consume_semicolon,
        )
        return declarations[0] if len(declarations) == 1 else declarations

    def parse_variable_declaration_list_rest(
        self,
        vtype,
        first_name,
        qualifiers=None,
        attributes=None,
        allow_semantic=True,
        consume_semicolon=True,
    ):
        declarations = []
        name = first_name
        while True:
            declarations.append(
                self.parse_variable_declaration_rest(
                    vtype,
                    name,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    allow_semantic=allow_semantic,
                    consume_semicolon=False,
                )
            )
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
            name = self.current_token[1]
            self.eat("IDENTIFIER")

        if consume_semicolon:
            self.eat("SEMICOLON")

        return declarations

    def parse_variable_declaration_rest(
        self,
        vtype,
        name,
        qualifiers=None,
        attributes=None,
        allow_semantic=True,
        consume_semicolon=True,
    ):
        qualifiers = qualifiers or []
        attributes = attributes or []

        array_sizes = self.parse_array_suffixes()
        semantic = None
        register = None
        packoffset = None

        if allow_semantic:
            semantic, register, packoffset = self.parse_semantic_or_register()

        value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

        sampler_state = None
        if self.is_sampler_state_type(vtype) and self.current_token[0] == "LBRACE":
            sampler_state = self.parse_sampler_state_block()

        if consume_semicolon:
            self.eat("SEMICOLON")

        var = VariableNode(
            vtype,
            name,
            value=value,
            qualifiers=qualifiers,
            attributes=attributes,
            is_const="const" in qualifiers,
            semantic=semantic,
        )
        var.array_sizes = array_sizes
        var.register = register
        var.packoffset = packoffset
        if sampler_state is not None:
            var.sampler_state = sampler_state
        return var

    def is_sampler_state_type(self, vtype):
        return str(vtype).split("<", 1)[0] in {
            "SamplerState",
            "SamplerComparisonState",
        }

    def parse_sampler_state_block(self):
        self.eat("LBRACE")
        state = []
        while self.current_token[0] != "RBRACE":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            value = self.parse_expression()
            self.eat("SEMICOLON")
            state.append((name, value))
        self.eat("RBRACE")
        return state

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")

        if_body = self.parse_statement_or_block()

        else_body = None
        if self.current_token[0] in ["ELSE", "ELSE_IF"]:
            if self.current_token[0] == "ELSE":
                self.eat("ELSE")
            else:
                else_body = self.parse_else_if_statement()
                return IfNode(condition, if_body, else_body)

            if self.current_token[0] == "IF":
                else_body = self.parse_if_statement()
            else:
                else_body = self.parse_statement_or_block()

        return IfNode(condition, if_body, else_body)

    def parse_else_if_statement(self):
        self.eat("ELSE_IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_statement_or_block()

        else_body = None
        if self.current_token[0] == "ELSE_IF":
            else_body = self.parse_else_if_statement()
        elif self.current_token[0] == "ELSE":
            self.eat("ELSE")
            if self.current_token[0] == "IF":
                else_body = self.parse_if_statement()
            else:
                else_body = self.parse_statement_or_block()

        return IfNode(condition, if_body, else_body)

    def parse_statement_or_block(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        stmt = self.parse_statement()
        return [stmt] if stmt is not None else []

    def parse_for_loop(self):
        self.eat("FOR")
        self.eat("LPAREN")

        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.looks_like_declaration():
                qualifiers = self.parse_qualifiers()
                init = self.parse_variable_declaration(
                    qualifiers=qualifiers,
                    attributes=[],
                    allow_semantic=False,
                    consume_semicolon=False,
                )
            else:
                init = self.parse_expression_sequence()
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression_sequence()
        self.eat("RPAREN")

        body = self.parse_statement_or_block()
        return ForNode(init, condition, update, body)

    def parse_expression_sequence(self):
        expressions = [self.parse_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_expression())
        return expressions[0] if len(expressions) == 1 else expressions

    def parse_while_loop(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement_or_block()
        return WhileNode(condition, body)

    def parse_do_while_loop(self):
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
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default_body = None

        while self.current_token[0] in ("CASE", "DEFAULT"):
            if self.current_token[0] == "CASE":
                cases.append(self.parse_switch_case())
            else:
                self.eat("DEFAULT")
                self.eat("COLON")
                default_body = []
                while self.current_token[0] not in [
                    "CASE",
                    "DEFAULT",
                    "RBRACE",
                    "EOF",
                ]:
                    stmt = self.parse_statement()
                    if stmt is not None:
                        if isinstance(stmt, list):
                            default_body.extend(stmt)
                        else:
                            default_body.append(stmt)

        self.eat("RBRACE")
        return SwitchNode(condition, cases, default_body)

    def parse_switch_case(self):
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        body = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            stmt = self.parse_statement()
            if stmt is None:
                continue
            if isinstance(stmt, list):
                body.extend(stmt)
            else:
                body.append(stmt)

        return CaseNode(value, body)

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_conditional_expression()
        if self.current_token[0] in ASSIGNMENT_TOKENS:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)
        return left

    def parse_conditional_expression(self):
        expr = self.parse_logical_or_expression()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_conditional_expression()
            return TernaryOpNode(expr, true_expr, false_expr)
        return expr

    def parse_logical_or_expression(self):
        expr = self.parse_logical_and_expression()
        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_logical_and_expression(self):
        expr = self.parse_bitwise_or_expression()
        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_or_expression(self):
        expr = self.parse_bitwise_xor_expression()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_xor_expression(self):
        expr = self.parse_bitwise_and_expression()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_bitwise_and_expression(self):
        expr = self.parse_equality_expression()
        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_equality_expression(self):
        expr = self.parse_relational_expression()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_relational_expression(self):
        expr = self.parse_shift_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_shift_expression(self):
        expr = self.parse_additive_expression()
        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_additive_expression(self):
        expr = self.parse_multiplicative_expression()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_multiplicative_expression(self):
        expr = self.parse_unary_expression()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_unary_expression(self):
        if self.current_token[0] == "LPAREN" and self.looks_like_cast():
            self.eat("LPAREN")
            target_type = self.parse_type()
            self.eat("RPAREN")
            operand = self.parse_unary_expression()
            return CastNode(target_type, operand)

        if self.current_token[0] in [
            "PLUS",
            "MINUS",
            "LOGICAL_NOT",
            "BITWISE_NOT",
            "INCREMENT",
            "DECREMENT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def looks_like_cast(self):
        if self.current_token[0] != "LPAREN":
            return False
        if not self.is_type_token(self.peek()[0]):
            return False
        idx = self.current_index + 1
        if idx < len(self.tokens) and self.tokens[idx][0] in TYPE_TOKENS:
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                depth = 0
                while idx < len(self.tokens):
                    if self.tokens[idx][0] == "LESS_THAN":
                        depth += 1
                    elif self.tokens[idx][0] == "GREATER_THAN":
                        depth -= 1
                        if depth == 0:
                            idx += 1
                            break
                    idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "RPAREN":
                next_token = (
                    self.tokens[idx + 1] if idx + 1 < len(self.tokens) else ("EOF", "")
                )
                return next_token[0] in {
                    "IDENTIFIER",
                    "NUMBER",
                    "HEX_NUMBER",
                    "BINARY_NUMBER",
                    "OCT_NUMBER",
                    "TRUE",
                    "FALSE",
                    "STRING",
                    "CHAR_LITERAL",
                    "LPAREN",
                    "LBRACE",
                    "PLUS",
                    "MINUS",
                    "LOGICAL_NOT",
                    "BITWISE_NOT",
                    "INCREMENT",
                    "DECREMENT",
                    *TYPE_TOKENS,
                }
        return False

    def parse_postfix_expression(self):
        expr = self.parse_primary_expression()
        while True:
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                expr = ArrayAccessNode(expr, index)
            elif self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                expr = MemberAccessNode(expr, member)
            elif self.current_token_is_double_colon():
                expr = self.parse_scoped_name(expr)
            elif self.current_token[0] == "LPAREN":
                args = self.parse_call_arguments()

                if isinstance(expr, MemberAccessNode) and isinstance(expr.member, str):
                    if expr.member == "Sample" and len(args) == 2:
                        expr = TextureSampleNode(expr.object, args[0], args[1])
                        continue
                    if expr.member == "SampleLevel" and len(args) == 3:
                        expr = TextureSampleNode(expr.object, args[0], args[1], args[2])
                        continue
                expr = FunctionCallNode(expr, args)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                node = UnaryOpNode(op, expr)
                node.is_postfix = True
                expr = node
            else:
                break
        return expr

    def parse_scoped_name(self, prefix):
        if not isinstance(prefix, str):
            raise SyntaxError("Expected identifier before scoped name separator")

        scoped_name = prefix
        while self.current_token_is_double_colon():
            self.eat_double_colon()
            member = self.current_token[1]
            self.eat("IDENTIFIER")
            scoped_name = f"{scoped_name}::{member}"
        return scoped_name

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def is_template_type_constructor_start(self):
        if self.current_token[0] != "IDENTIFIER":
            return False
        if self.current_token[1] not in {"vector", "matrix"}:
            return False
        if self.peek()[0] != "LESS_THAN":
            return False

        depth = 0
        idx = self.current_index + 1
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    next_token = (
                        self.tokens[idx + 1]
                        if idx + 1 < len(self.tokens)
                        else ("EOF", "")
                    )
                    return next_token[0] == "LPAREN"
            elif token_type == "EOF":
                return False
            idx += 1
        return False

    def parse_primary_expression(self):
        token_type, value = self.current_token
        if self.is_template_type_constructor_start():
            type_name = self.parse_type()
            args = self.parse_call_arguments()
            return VectorConstructorNode(type_name, args)
        if token_type == "IDENTIFIER":
            self.eat("IDENTIFIER")
            return value
        if token_type == "CLIP":
            self.eat("CLIP")
            return value
        if token_type in ["NUMBER", "HEX_NUMBER", "BINARY_NUMBER", "OCT_NUMBER"]:
            self.eat(token_type)
            return self.parse_numeric_literal(token_type, value)
        if token_type in ["TRUE", "FALSE"]:
            self.eat(token_type)
            return token_type == "TRUE"
        if token_type == "STRING":
            return self.parse_string_literal()
        if token_type == "CHAR_LITERAL":
            self.eat("CHAR_LITERAL")
            return value
        if token_type == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        if token_type == "LBRACE":
            return self.parse_initializer_list()
        if token_type in [
            "FLOAT",
            "HALF",
            "DOUBLE",
            "INT",
            "UINT",
            "BOOL",
            "FVECTOR",
            "IVECTOR",
            "UVECTOR",
            "BVECTOR",
            "MATRIX",
        ]:
            type_name = value
            self.eat(token_type)
            if self.current_token[0] == "LPAREN":
                args = self.parse_call_arguments()
                return VectorConstructorNode(type_name, args)
            return type_name

        raise SyntaxError(
            f"Unexpected token in primary expression: {self.current_token}"
        )

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []
        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
                continue
            break
        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_string_literal(self):
        value = self.current_token[1]
        self.eat("STRING")
        while self.current_token[0] == "STRING":
            next_value = self.current_token[1]
            value = self.concatenate_string_literals(value, next_value)
            self.eat("STRING")
        return value

    def concatenate_string_literals(self, left, right):
        if (
            len(left) >= 2
            and len(right) >= 2
            and left[0] == left[-1] == '"'
            and right[0] == right[-1] == '"'
        ):
            return left[:-1] + right[1:]
        return left + right

    def parse_numeric_literal(self, token_type, value):
        if token_type == "HEX_NUMBER":
            stripped = re.sub(r"[uUlL]+$", "", value)
            return int(stripped, 16)
        if token_type == "BINARY_NUMBER":
            stripped = re.sub(r"[uUlL]+$", "", value)
            return int(stripped, 2)
        if token_type == "OCT_NUMBER":
            stripped = re.sub(r"[uUlL]+$", "", value)
            return int(stripped, 8)
        stripped = re.sub(r"[fFhHuUlL]+$", "", value)
        if "." in stripped or "e" in stripped or "E" in stripped:
            return float(stripped)
        return int(stripped)
