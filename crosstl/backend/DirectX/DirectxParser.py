import re

from .DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    WhileNode,
    DoWhileNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    EnumNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    PragmaNode,
    IncludeNode,
    SwitchNode,
    CaseNode,
    TernaryOpNode,
)
from ..common_ast import (
    BreakNode,
    ContinueNode,
    ArrayAccessNode,
    CastNode,
    AttributeNode,
    PreprocessorNode,
    TextureSampleNode,
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
    "VOLATILE",
    "PRECISE",
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
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else ("EOF", "")

    def parse(self):
        structs = []
        functions = []
        global_variables = []
        cbuffers = []
        enums = []
        typedefs = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                directive = self.parse_preprocessor_directive()
                if directive is not None:
                    structs.append(directive)
                continue

            if self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
                continue

            if self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
                continue

            if self.current_token[0] == "TYPEDEF":
                typedefs.append(self.parse_typedef())
                continue

            if self.current_token[0] == "CBUFFER":
                cbuffers.append(self.parse_cbuffer())
                continue

            attributes = self.parse_attribute_list()
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
                var = self.parse_variable_declaration_rest(
                    return_type,
                    name,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                    consume_semicolon=True,
                )
                global_variables.append(var)

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
            self.eat("LBRACKET")
            if self.current_token[0] != "IDENTIFIER":
                while self.current_token[0] != "RBRACKET" and self.current_token[0] != "EOF":
                    self.eat(self.current_token[0])
                self.eat("RBRACKET")
                continue

            name = self.current_token[1]
            self.eat("IDENTIFIER")
            args = []
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")

            while self.current_token[0] != "RBRACKET" and self.current_token[0] != "EOF":
                self.eat(self.current_token[0])
            self.eat("RBRACKET")
            attributes.append(AttributeNode(name, args))

        return attributes

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
        while self.current_token[0] != "GREATER_THAN" and self.current_token[0] != "EOF":
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
            if not self.is_type_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected type in struct member, got {self.current_token[0]}"
                )
            member_type = self.parse_type()
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            member_semantic, _, _ = self.parse_semantic_or_register()
            self.eat("SEMICOLON")

            member = VariableNode(
                member_type,
                member_name,
                qualifiers=qualifiers,
                attributes=attributes,
                semantic=member_semantic,
            )
            member.array_sizes = array_sizes
            members.append(member)

        self.eat("RBRACE")

        # Variable declarations after struct
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

    def parse_cbuffer(self):
        self.eat("CBUFFER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "COLON":
            _, register, packoffset = self.parse_semantic_or_register()
        else:
            register = None
            packoffset = None

        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            qualifiers = self.parse_qualifiers()
            member_type = self.parse_type()
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            array_sizes = self.parse_array_suffixes()
            semantic = None
            register = None
            packoffset = None
            if self.current_token[0] == "COLON":
                semantic, register, packoffset = self.parse_semantic_or_register()
            self.eat("SEMICOLON")

            member = VariableNode(
                member_type,
                member_name,
                qualifiers=qualifiers,
                semantic=semantic,
            )
            member.register = register
            member.packoffset = packoffset
            member.array_sizes = array_sizes
            members.append(member)

        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        cbuffer_node = StructNode(name, members)
        cbuffer_node.is_cbuffer = True
        cbuffer_node.register = register
        cbuffer_node.packoffset = packoffset
        return cbuffer_node
    def parse_function(self, return_type, name, qualifiers, attributes):
        params = self.parse_parameters()

        semantic = None
        if self.current_token[0] == "COLON":
            semantic, _, _ = self.parse_semantic_or_register()

        body = self.parse_block()

        qualifier = self.infer_function_qualifier(name, attributes, params, semantic)

        return FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=qualifiers + ([qualifier] if qualifier else []),
            attributes=attributes,
            qualifier=qualifier,
            semantic=semantic,
        )

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

        if self.current_token[0] != "RPAREN":
            while True:
                attributes = self.parse_attribute_list()
                qualifiers = self.parse_qualifiers()
                if (
                    self.current_token[0] == "IDENTIFIER"
                    and self.current_token[1] in primitive_qualifiers
                ):
                    attributes.append(AttributeNode("primitive", [self.current_token[1]]))
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

                param = VariableNode(
                    param_type,
                    name,
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

        if self.current_token[0] == "RETURN":
            self.eat("RETURN")
            value = None
            if self.current_token[0] != "SEMICOLON":
                value = self.parse_expression()
            self.eat("SEMICOLON")
            return ReturnNode(value)

        if self.current_token[0] == "IF":
            return self.parse_if_statement()

        if self.current_token[0] == "FOR":
            return self.parse_for_loop()

        if self.current_token[0] == "WHILE":
            return self.parse_while_loop()

        if self.current_token[0] == "DO":
            return self.parse_do_while_loop()

        if self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()

        if self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()

        if self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()

        if self.current_token[0] == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return "discard"

        if self.current_token[0] == "PREPROCESSOR":
            self.parse_preprocessor_directive()
            return None

        if self.looks_like_declaration():
            attributes = self.parse_attribute_list()
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
        return expr

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
        return self.parse_variable_declaration_rest(
            vtype,
            name,
            qualifiers=qualifiers,
            attributes=attributes,
            allow_semantic=allow_semantic,
            consume_semicolon=consume_semicolon,
        )

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
        return var

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
                self.eat("ELSE_IF")
                self.eat("LPAREN")
                else_condition = self.parse_expression()
                self.eat("RPAREN")
                else_body = IfNode(
                    else_condition,
                    self.parse_statement_or_block(),
                    None,
                )
                return IfNode(condition, if_body, else_body)

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
                init = self.parse_expression()
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()
        self.eat("RPAREN")

        body = self.parse_statement_or_block()
        return ForNode(init, condition, update, body)

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
                return True
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
            elif self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                args = []
                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")

                if isinstance(expr, MemberAccessNode) and isinstance(expr.member, str):
                    if expr.member in ["Sample", "SampleLevel"] and len(args) >= 2:
                        lod = args[2] if expr.member == "SampleLevel" and len(args) > 2 else None
                        expr = TextureSampleNode(expr.object, args[0], args[1], lod)
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

    def parse_primary_expression(self):
        token_type, value = self.current_token
        if token_type == "IDENTIFIER":
            self.eat("IDENTIFIER")
            return value
        if token_type in ["NUMBER", "HEX_NUMBER", "BINARY_NUMBER", "OCT_NUMBER"]:
            self.eat(token_type)
            return self.parse_numeric_literal(token_type, value)
        if token_type in ["TRUE", "FALSE"]:
            self.eat(token_type)
            return token_type == "TRUE"
        if token_type in ["STRING", "CHAR_LITERAL"]:
            self.eat(token_type)
            return value
        if token_type == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
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
                self.eat("LPAREN")
                args = []
                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")
                return VectorConstructorNode(type_name, args)
            return type_name

        raise SyntaxError(
            f"Unexpected token in primary expression: {self.current_token}"
        )

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
