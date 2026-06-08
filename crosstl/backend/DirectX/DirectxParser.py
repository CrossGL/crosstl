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
    "FIXED",
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
    "FLOAT16_T",
    "FLOAT32_T",
    "FLOAT64_T",
    "INT16_T",
    "INT32_T",
    "UINT16_T",
    "UINT32_T",
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
    "REORDERCOHERENT",
    "ROW_MAJOR",
    "COLUMN_MAJOR",
    "NOINTERPOLATION",
    "NOPERSPECTIVE",
    "LINEAR",
    "CENTROID",
    "SAMPLE",
    "IN",
    "OUT",
    "INOUT",
    "UNIFORM",
    "GROUPSHARED",
}

INTERPOLATION_QUALIFIER_TOKENS = {
    "NOINTERPOLATION",
    "NOPERSPECTIVE",
    "LINEAR",
    "CENTROID",
    "SAMPLE",
}

CONTEXTUAL_QUALIFIER_IDENTIFIERS = {"center", "shared", "snorm", "unorm"}
CONTEXTUAL_RESOURCE_IDENTIFIER_TOKENS = {
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
}
CONTEXTUAL_IDENTIFIER_TOKENS = {
    "CLIP",
    "GLOBALLYCOHERENT",
    "PRECISE",
    "REORDERCOHERENT",
    "SAMPLE",
} | CONTEXTUAL_RESOURCE_IDENTIFIER_TOKENS
COMPOSITE_TYPE_PREFIXES = {"signed", "unsigned"}
COMPOSITE_TYPE_TOKENS = {"INT", "UINT", "DWORD", "IVECTOR", "UVECTOR", "MATRIX"}

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

OPERATOR_OVERLOAD_TOKENS = {
    "ASSIGN_SHIFT_LEFT",
    "ASSIGN_SHIFT_RIGHT",
    "PLUS_EQUALS",
    "MINUS_EQUALS",
    "MULTIPLY_EQUALS",
    "DIVIDE_EQUALS",
    "MOD_EQUALS",
    "ASSIGN_AND",
    "ASSIGN_OR",
    "ASSIGN_XOR",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "EQUAL",
    "NOT_EQUAL",
    "LOGICAL_AND",
    "LOGICAL_OR",
    "LOGICAL_NOT",
    "BITWISE_NOT",
    "BITWISE_XOR",
    "BITWISE_OR",
    "BITWISE_AND",
    "INCREMENT",
    "DECREMENT",
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "MOD",
    "EQUALS",
    "LESS_THAN",
    "GREATER_THAN",
}

EFFECT_BLOCK_KEYWORDS = {
    "fxgroup",
    "pass",
    "program",
    "state",
    "technique",
    "technique10",
    "technique11",
}

STATEMENT_START_TOKENS = {
    "RETURN",
    "IF",
    "FOR",
    "WHILE",
    "DO",
    "SWITCH",
    "BREAK",
    "CONTINUE",
    "DISCARD",
    "LBRACE",
}

UNEXPANDED_STATEMENT_ATTRIBUTE_MACROS = {
    "UNITY_BRANCH": "branch",
    "UNITY_FLATTEN": "flatten",
    "UNITY_LOOP": "loop",
    "UNITY_UNROLL": "unroll",
}

SCALAR_CONSTRUCTOR_TOKENS = {
    "FLOAT",
    "HALF",
    "FIXED",
    "DOUBLE",
    "INT",
    "UINT",
    "BOOL",
    "DWORD",
    "MIN16FLOAT",
    "MIN10FLOAT",
    "MIN16INT",
    "MIN12INT",
    "MIN16UINT",
    "FLOAT16_T",
    "FLOAT32_T",
    "FLOAT64_T",
    "INT16_T",
    "INT32_T",
    "UINT16_T",
    "UINT32_T",
    "INT64_T",
    "UINT64_T",
}

COMPOSITE_DECLARATOR_IDENTIFIER_TOKENS = {
    "FVECTOR",
    "IVECTOR",
    "UVECTOR",
    "BVECTOR",
    "MATRIX",
}


class HLSLParser:
    """Parse HLSL tokens into the DirectX backend shader AST."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else ("EOF", "")
        self.synthetic_structs = []
        self.synthetic_cbuffer_names = set()
        self.synthetic_enum_count = 0

    def parse(self):
        structs = []
        functions = []
        global_variables = []
        cbuffers = []
        enums = []
        typedefs = []

        while self.current_token[0] != "EOF":
            if self.current_token_is_keyword("USING", "using"):
                alias = self.parse_using_directive()
                if alias is not None:
                    typedefs.append(alias)
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

            if (
                self.current_token[0] == "STRUCT"
                and not self.looks_like_external_declaration()
            ):
                if self.is_struct_forward_declaration():
                    structs.append(self.parse_struct_forward_declaration())
                    continue
                synthetic_start = len(self.synthetic_structs)
                struct = self.parse_struct()
                structs.extend(self.synthetic_structs[synthetic_start:])
                structs.append(struct)
                continue

            if self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
                continue

            if self.current_token[0] == "TYPEDEF":
                synthetic_start = len(self.synthetic_structs)
                typedefs.append(self.parse_typedef())
                structs.extend(self.synthetic_structs[synthetic_start:])
                continue

            if self.is_class_declaration_prefix():
                self.parse_class_declaration()
                continue

            if self.is_template_declaration_prefix():
                self.parse_template_declaration_prefix()
                continue

            if self.is_effect_metadata_block():
                self.parse_effect_metadata_block()
                continue

            attributes = self.parse_attribute_list()
            if self.is_class_declaration_prefix():
                self.parse_class_declaration()
                continue

            if self.is_qualified_struct_definition_start():
                qualifiers = self.parse_qualifiers()
                synthetic_start = len(self.synthetic_structs)
                struct = self.parse_struct(
                    declaration_qualifiers=qualifiers,
                    declaration_attributes=attributes,
                )
                structs.extend(self.synthetic_structs[synthetic_start:])
                structs.append(struct)
                continue

            if self.current_token[0] in {"CBUFFER", "TBUFFER"}:
                synthetic_start = len(self.synthetic_structs)
                cbuffers.append(self.parse_cbuffer(attributes=attributes))
                structs.extend(self.synthetic_structs[synthetic_start:])
                continue

            if not self.looks_like_external_declaration():
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                else:
                    self.eat(self.current_token[0])
                continue

            qualifiers = self.parse_qualifiers()
            attributes.extend(self.parse_function_modifier_attributes())
            qualifiers.extend(self.parse_qualifiers())
            return_type = self.parse_type()
            qualifiers.extend(self.parse_post_type_qualifiers())
            attributes.extend(self.parse_attribute_list())
            if not self.is_declarator_identifier_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected identifier after type, got {self.current_token[0]}"
                )

            name = self.parse_function_declarator_name()

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

    def is_effect_metadata_block(self):
        if self.current_token[0] != "IDENTIFIER":
            return False
        if str(self.current_token[1]).lower() not in EFFECT_BLOCK_KEYWORDS:
            return False

        idx = self.current_index + 1
        while idx < len(self.tokens) and self.tokens[idx][0] not in {
            "LBRACE",
            "LESS_THAN",
            "SEMICOLON",
            "EOF",
        }:
            idx += 1

        while idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_angle_list_at(idx)
            if idx is None:
                return False

        return idx < len(self.tokens) and self.tokens[idx][0] == "LBRACE"

    def parse_effect_metadata_block(self):
        self.eat("IDENTIFIER")
        while self.current_token[0] not in {
            "LBRACE",
            "LESS_THAN",
            "SEMICOLON",
            "EOF",
        }:
            self.eat(self.current_token[0])

        self.skip_effect_declaration_suffixes()

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

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
        return token_type == "STRUCT" or token_type in TYPE_TOKENS

    def is_identifier_token(self, token_type):
        return token_type == "IDENTIFIER" or token_type in CONTEXTUAL_IDENTIFIER_TOKENS

    def is_identifier_token_at(self, index):
        return index < len(self.tokens) and self.is_identifier_token(
            self.tokens[index][0]
        )

    def is_declarator_identifier_token(self, token_type):
        return (
            self.is_identifier_token(token_type)
            or token_type in COMPOSITE_DECLARATOR_IDENTIFIER_TOKENS
        )

    def is_declarator_identifier_token_at(self, index):
        return index < len(self.tokens) and self.is_declarator_identifier_token(
            self.tokens[index][0]
        )

    def is_macro_like_identifier(self, token):
        token_type, value = token
        if token_type != "IDENTIFIER":
            return False
        text = str(value)
        return any(char.isalpha() for char in text) and text.upper() == text

    def skip_unexpanded_member_macro(self):
        if not self.is_macro_like_identifier(self.current_token):
            return False
        if self.peek()[0] not in {"LPAREN", "SEMICOLON", "RBRACE"}:
            return False

        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_delimiter_block("LPAREN", "RPAREN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return True

    def parse_identifier(self):
        if self.is_identifier_token(self.current_token[0]):
            token = self.current_token
            self.eat(token[0])
            return token[1]
        raise SyntaxError(f"Expected identifier, got {self.current_token[0]}")

    def parse_declarator_identifier(self):
        if self.is_declarator_identifier_token(self.current_token[0]):
            token = self.current_token
            self.eat(token[0])
            return token[1]
        raise SyntaxError(
            f"Expected declarator identifier, got {self.current_token[0]}"
        )

    def parse_typedef_name(self):
        if self.is_identifier_token(self.current_token[0]) or self.current_token[0] in {
            "FVECTOR",
            "IVECTOR",
            "UVECTOR",
            "BVECTOR",
            "MATRIX",
        }:
            token = self.current_token
            self.eat(token[0])
            return token[1]
        raise SyntaxError(f"Expected typedef name, got {self.current_token[0]}")

    def current_token_is_keyword(self, token_type, value):
        return self.current_token[0] == token_type or (
            self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value
        )

    def current_token_is_double_colon(self):
        return self.current_token[0] == "COLON" and self.peek()[0] == "COLON"

    def token_at_is_double_colon(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "COLON"
            and self.tokens[index + 1][0] == "COLON"
        )

    def is_type_start_at(self, index):
        if self.token_at_is_double_colon(index):
            index += 2
        return index < len(self.tokens) and self.is_type_token(self.tokens[index][0])

    def is_template_declaration_prefix(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "template"
            and self.peek()[0] == "LESS_THAN"
        )

    def parse_template_declaration_prefix(self):
        self.eat("IDENTIFIER")
        self.skip_template_argument_list()

    def parse_template_declaration_prefixes(self):
        while self.is_template_declaration_prefix():
            self.parse_template_declaration_prefix()

    def skip_template_argument_list(self):
        self.eat("LESS_THAN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    return
            elif token_type == "SHIFT_RIGHT" and depth > 1:
                depth -= 2
                if depth == 0:
                    self.eat("SHIFT_RIGHT")
                    return
            self.eat(token_type)

        raise SyntaxError("Unterminated template argument list")

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
            return None

        if (
            self.is_identifier_token(self.current_token[0])
            and self.peek()[0] == "EQUALS"
        ):
            name = self.parse_identifier()
            self.eat("EQUALS")
            qualifiers = self.parse_qualifiers()
            alias_type = self.parse_type()
            qualifiers.extend(self.parse_post_type_qualifiers())
            attributes = []
            if self.current_token[0] == "LBRACKET" and self.peek()[0] == "LBRACKET":
                attributes = self.parse_attribute_list()
            array_sizes = self.parse_array_suffixes()
            self.eat("SEMICOLON")

            alias = TypeAliasNode(alias_type, name)
            alias.qualifiers = qualifiers
            alias.attributes = attributes
            alias.array_sizes = array_sizes
            return alias

        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            if self.current_token_is_double_colon():
                self.eat_double_colon()
            else:
                self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return None

    def is_class_declaration_prefix(self):
        return self.current_token_is_keyword(
            "CLASS", "class"
        ) or self.current_token_is_keyword("INTERFACE", "interface")

    def is_qualified_struct_definition_start(self):
        idx = self.current_index
        saw_qualifier = False
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            saw_qualifier = True
            idx += 1

        if (
            not saw_qualifier
            or idx >= len(self.tokens)
            or self.tokens[idx][0] != "STRUCT"
        ):
            return False

        idx += 1
        if idx < len(self.tokens) and self.tokens[idx][0] == "LBRACE":
            return True
        if idx < len(self.tokens) and self.is_identifier_token_at(idx):
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_angle_list_at(idx)
                if idx is None:
                    return False
            return idx < len(self.tokens) and self.tokens[idx][0] == "LBRACE"
        return False

    def parse_class_declaration(self):
        if self.current_token_is_keyword("CLASS", "class"):
            self.eat_keyword("CLASS", "class")
        else:
            self.eat_keyword("INTERFACE", "interface")

        if self.is_identifier_token(self.current_token[0]):
            self.parse_identifier()

        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()

        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_balanced_brace_block(self):
        self.eat("LBRACE")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
            self.eat(token_type)

        if depth != 0:
            raise SyntaxError("Unterminated brace block")

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
        parser.synthetic_enum_count = self.synthetic_enum_count
        namespace_ast = parser.parse()
        self.synthetic_enum_count = parser.synthetic_enum_count
        return namespace_ast

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
        while self.is_qualifier_token_at(self.current_index):
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_function_modifier_attributes(self):
        attributes = []
        while self.is_clipplanes_modifier_at(self.current_index):
            attributes.append(self.parse_clipplanes_modifier())
        return attributes

    def parse_clipplanes_modifier(self):
        self.eat("IDENTIFIER")
        args = self.parse_call_arguments()
        return AttributeNode("clipplanes", args)

    def parse_post_type_qualifiers(self):
        qualifiers = []
        while self.is_post_type_qualifier_token_at(self.current_index):
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def is_post_type_qualifier_token_at(self, index):
        if index >= len(self.tokens):
            return False
        return str(self.tokens[index][1]).lower() == "const"

    def is_qualifier_token_at(self, index):
        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type in QUALIFIER_TOKENS:
            return True
        if (
            token_type == "IDENTIFIER"
            and token_value in CONTEXTUAL_QUALIFIER_IDENTIFIERS
        ):
            next_token_type = (
                self.tokens[index + 1][0] if index + 1 < len(self.tokens) else None
            )
            return next_token_type in QUALIFIER_TOKENS or next_token_type in TYPE_TOKENS
        return False

    def parse_type(self):
        if self.current_token_is_double_colon():
            self.eat_double_colon()

        if not self.is_type_token(self.current_token[0]):
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")

        if self.current_token[0] == "STRUCT":
            self.eat("STRUCT")
            if not self.is_identifier_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected struct type name, got {self.current_token[0]}"
                )
            return self.parse_identifier()

        base = self.current_token[1]
        self.eat(self.current_token[0])
        type_name = base
        if (
            base in COMPOSITE_TYPE_PREFIXES
            and self.current_token[0] in COMPOSITE_TYPE_TOKENS
        ):
            type_name = f"{type_name} {self.current_token[1]}"
            self.eat(self.current_token[0])

        while self.current_token[0] == "COLON" and self.peek()[0] == "COLON":
            self.eat("COLON")
            self.eat("COLON")
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected identifier after scoped type, got {self.current_token[0]}"
                )
            type_name += f"::{self.current_token[1]}"
            self.eat("IDENTIFIER")

        if self.current_token[0] == "LESS_THAN":
            args = self.parse_generic_arguments()
            type_name = f"{type_name}<{', '.join(args)}>"

        return type_name

    def parse_generic_arguments(self):
        args = []
        self.eat("LESS_THAN")
        current = []
        depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "GREATER_THAN" and depth == 0:
                if current:
                    args.append(self.format_generic_argument_tokens(current))
                self.eat("GREATER_THAN")
                return args
            if token_type == "SHIFT_RIGHT" and depth > 0:
                current.append(("GREATER_THAN", ">"))
                if depth == 1:
                    args.append(self.format_generic_argument_tokens(current))
                    self.eat("SHIFT_RIGHT")
                    return args
                current.append(("GREATER_THAN", ">"))
                depth -= 2
                self.eat("SHIFT_RIGHT")
                continue
            if token_type == "COMMA" and depth == 0:
                args.append(self.format_generic_argument_tokens(current))
                current = []
                self.eat("COMMA")
                continue

            current.append(self.current_token)
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
            self.eat(token_type)

        raise SyntaxError("Unterminated generic argument list")

    def format_generic_argument_tokens(self, tokens):
        result = []
        previous_kind = None
        previous_text = None
        index = 0
        while index < len(tokens):
            token_type, token_value = tokens[index]
            if (
                token_type == "COLON"
                and index + 1 < len(tokens)
                and tokens[index + 1][0] == "COLON"
            ):
                text = "::"
                current_kind = "scope"
                index += 2
            else:
                text = str(token_value)
                current_kind = self.generic_argument_token_kind(
                    token_type, previous_kind, previous_text
                )
                index += 1

            if result and self.generic_argument_needs_space(
                previous_kind, current_kind, previous_text, text
            ):
                result.append(" ")
            result.append(text)
            previous_kind = current_kind
            previous_text = text

        return "".join(result).strip()

    def generic_argument_token_kind(self, token_type, previous_kind, previous_text):
        if token_type in {"PLUS", "MINUS"} and (
            previous_kind is None
            or previous_kind in {"binary_operator", "unary_operator"}
            or previous_text in {"(", "[", "<", ","}
        ):
            return "unary_operator"
        if token_type in {
            "PLUS",
            "MINUS",
            "MULTIPLY",
            "DIVIDE",
            "MOD",
            "BITWISE_AND",
            "BITWISE_OR",
            "BITWISE_XOR",
            "SHIFT_LEFT",
            "SHIFT_RIGHT",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "LOGICAL_AND",
            "LOGICAL_OR",
        }:
            return "binary_operator"
        return token_type

    def generic_argument_needs_space(
        self, previous_kind, current_kind, previous_text, current_text
    ):
        if previous_kind is None:
            return False
        if current_text in {",", ">", ")", "]", ".", "::"}:
            return False
        if previous_text in {"<", "(", "[", ".", "::"}:
            return False
        if current_text in {"<", "(", "["}:
            return False
        if previous_text == ",":
            return True
        if previous_kind == "unary_operator" or current_kind == "unary_operator":
            return False
        if previous_kind == "binary_operator" or current_kind == "binary_operator":
            return True
        return bool(re.match(r"[A-Za-z0-9_]", str(previous_text)[-1:])) and bool(
            re.match(r"[A-Za-z0-9_]", str(current_text)[:1])
        )

    def parse_array_suffixes(self):
        sizes = []
        while self.current_token[0] == "LBRACKET" and self.peek()[0] != "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] != "RBRACKET":
                sizes.append(self.parse_expression())
            else:
                sizes.append(None)
            self.eat("RBRACKET")
        return sizes

    def format_array_suffixes_for_type(self, sizes):
        suffixes = []
        for size in sizes:
            if size is None:
                suffixes.append("[]")
            else:
                suffixes.append(f"[{size}]")
        return "".join(suffixes)

    def parse_semantic_or_register(self):
        semantic = None
        register = None
        packoffset = None

        semantics = []
        while self.current_token[0] == "COLON":
            self.eat("COLON")
            if self.current_token[0] == "REGISTER":
                register = self.parse_register_binding("REGISTER")
                continue
            if self.current_token[0] == "PACKOFFSET":
                packoffset = self.parse_register_binding("PACKOFFSET")
                continue

            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected semantic identifier, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                args = self.parse_semantic_argument_list()
                name = f"{name}({args})"
            semantics.append(name)

        if semantics:
            semantic = ": ".join(semantics)
        return semantic, register, packoffset

    def parse_parameter_semantic_and_interpolation(self):
        semantic = None
        register = None
        packoffset = None
        interpolation_qualifiers = []

        semantics = []
        while self.current_token[0] == "COLON":
            self.eat("COLON")
            if self.current_token[0] == "REGISTER":
                register = self.parse_register_binding("REGISTER")
                continue
            if self.current_token[0] == "PACKOFFSET":
                packoffset = self.parse_register_binding("PACKOFFSET")
                continue
            if self.current_token[0] in INTERPOLATION_QUALIFIER_TOKENS:
                interpolation_qualifiers.extend(self.parse_interpolation_qualifiers())
                continue

            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError(
                    f"Expected semantic identifier, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                args = self.parse_semantic_argument_list()
                name = f"{name}({args})"
            semantics.append(name)

        interpolation_qualifiers.extend(self.parse_interpolation_qualifiers())

        if semantics:
            semantic = ": ".join(semantics)
        return semantic, register, packoffset, interpolation_qualifiers

    def parse_interpolation_qualifiers(self):
        qualifiers = []
        while self.current_token[0] in INTERPOLATION_QUALIFIER_TOKENS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_semantic_argument_list(self):
        self.eat("LPAREN")
        parts = []
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            if token_type == "COMMA":
                parts.append(", ")
            else:
                parts.append(str(token_value))
            self.eat(token_type)
        if depth != 0:
            raise SyntaxError("Unterminated semantic argument list")
        return "".join(parts).strip()

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

    def parse_struct_base_list(self):
        self.eat("COLON")
        base_classes = []
        while True:
            if not self.is_identifier_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected struct base identifier, got {self.current_token[0]}"
                )
            base_name = self.parse_identifier()
            if self.current_token_is_double_colon():
                base_name = self.parse_scoped_name(base_name)
            base_classes.append(base_name)
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")
        return base_classes

    def is_struct_forward_declaration(self):
        if self.current_token[0] != "STRUCT":
            return False

        idx = self.current_index + 1
        if idx >= len(self.tokens) or not self.is_identifier_token_at(idx):
            return False
        idx += 1

        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_angle_list_at(idx)
            if idx is None:
                return False

        return idx < len(self.tokens) and self.tokens[idx][0] == "SEMICOLON"

    def skip_angle_list_at(self, idx):
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif token_type == "SHIFT_RIGHT" and depth > 1:
                depth -= 2
                if depth == 0:
                    return idx + 1
            elif token_type == "EOF":
                return None
            idx += 1
        return None

    def parse_struct_forward_declaration(self):
        self.eat("STRUCT")
        name = self.parse_identifier()
        if self.current_token[0] == "LESS_THAN":
            self.skip_template_argument_list()
        self.eat("SEMICOLON")

        struct_node = StructNode(name, [])
        struct_node.is_forward_declaration = True
        return struct_node

    def parse_struct(self, declaration_qualifiers=None, declaration_attributes=None):
        declaration_qualifiers = list(declaration_qualifiers or [])
        declaration_attributes = list(declaration_attributes or [])
        self.eat("STRUCT")
        struct_attributes = self.parse_attribute_list()
        name = None
        if self.is_identifier_token(self.current_token[0]):
            name = self.parse_identifier()
            if self.current_token[0] == "LESS_THAN":
                self.skip_template_argument_list()

        semantic = None
        base_classes = []
        if name is not None and self.current_token[0] == "COLON":
            base_classes = self.parse_struct_base_list()

        self.eat("LBRACE")
        members = []
        methods = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            self.parse_template_declaration_prefixes()
            attributes = self.parse_attribute_list()
            self.parse_template_declaration_prefixes()
            qualifiers = self.parse_qualifiers()
            if self.skip_unexpanded_member_macro():
                continue
            if self.current_token_is_keyword("USING", "using"):
                self.parse_using_directive()
                continue
            if self.is_constructor_or_destructor_member(name):
                methods.append(
                    self.parse_constructor_or_destructor_member(
                        name, qualifiers=qualifiers, attributes=attributes
                    )
                )
                continue
            if self.current_token[0] == "ENUM":
                members.append(self.parse_enum())
                continue
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
            return_type = self.parse_type()
            qualifiers.extend(self.parse_post_type_qualifiers())
            member_name = self.parse_member_declarator_name()
            if self.current_token[0] == "LPAREN":
                methods.append(
                    self.parse_function(
                        return_type,
                        member_name,
                        qualifiers=qualifiers,
                        attributes=attributes,
                    )
                )
            else:
                declarations = self.parse_variable_declaration_list_rest(
                    return_type,
                    member_name,
                    qualifiers=qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                    consume_semicolon=True,
                )
                members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")

        variables = []
        variable_declarations = []
        if self.is_declarator_identifier_token(self.current_token[0]):
            first_name = self.parse_declarator_identifier()
            struct_type = name or self.synthetic_struct_type_name(
                "AnonymousStruct", first_name
            )
            if name is None:
                name = struct_type
            variable_declarations = self.parse_variable_declaration_list_rest(
                struct_type,
                first_name,
                qualifiers=declaration_qualifiers,
                attributes=declaration_attributes,
                allow_semantic=True,
                consume_semicolon=True,
            )
            variables = [declaration.name for declaration in variable_declarations]
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        struct_node = StructNode(name, members)
        struct_node.attributes = struct_attributes
        struct_node.variables = variables
        struct_node.variable_declarations = variable_declarations
        struct_node.semantic = semantic
        struct_node.base_classes = base_classes
        struct_node.methods = methods
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
        if (
            self.is_identifier_token(self.current_token[0])
            and self.peek()[0] == "LBRACE"
        ):
            nested_name = self.parse_identifier()

        self.eat("LBRACE")
        nested_members = []
        nested_methods = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            self.parse_template_declaration_prefixes()
            member_attributes = self.parse_attribute_list()
            self.parse_template_declaration_prefixes()
            member_qualifiers = self.parse_qualifiers()
            if self.skip_unexpanded_member_macro():
                continue
            if self.current_token_is_keyword("USING", "using"):
                self.parse_using_directive()
                continue
            if self.is_constructor_or_destructor_member(nested_name):
                nested_methods.append(
                    self.parse_constructor_or_destructor_member(
                        nested_name,
                        qualifiers=member_qualifiers,
                        attributes=member_attributes,
                    )
                )
                continue
            if self.current_token[0] == "ENUM":
                nested_members.append(self.parse_enum())
                continue
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
            return_type = self.parse_type()
            member_qualifiers.extend(self.parse_post_type_qualifiers())
            member_name = self.parse_member_declarator_name()
            if self.current_token[0] == "LPAREN":
                nested_methods.append(
                    self.parse_function(
                        return_type,
                        member_name,
                        qualifiers=member_qualifiers,
                        attributes=member_attributes,
                    )
                )
            else:
                declarations = self.parse_variable_declaration_list_rest(
                    return_type,
                    member_name,
                    qualifiers=member_qualifiers,
                    attributes=member_attributes,
                    allow_semantic=True,
                    consume_semicolon=True,
                )
                nested_members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")

        if not self.is_declarator_identifier_token(self.current_token[0]):
            if nested_name is None:
                raise SyntaxError("Expected identifier after anonymous struct member")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            nested_struct = StructNode(nested_name, nested_members)
            nested_struct.methods = nested_methods
            self.synthetic_structs.append(nested_struct)
            return []

        first_name = self.parse_declarator_identifier()
        struct_type = nested_name or self.synthetic_struct_type_name(
            parent_name, first_name
        )
        nested_struct = StructNode(struct_type, nested_members)
        nested_struct.methods = nested_methods
        self.synthetic_structs.append(nested_struct)
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

    def synthetic_enum_name(self):
        self.synthetic_enum_count += 1
        return f"AnonymousEnum_{self.synthetic_enum_count}"

    def parse_operator_overload_name(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "operator"
        ):
            self.eat("IDENTIFIER")
            if self.current_token[0] in OPERATOR_OVERLOAD_TOKENS:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                return f"operator{op}"
            if self.current_token[0] == "LBRACKET" and self.peek()[0] == "RBRACKET":
                self.eat("LBRACKET")
                self.eat("RBRACKET")
                return "operator[]"
            if self.current_token[0] == "LPAREN" and self.peek()[0] == "RPAREN":
                self.eat("LPAREN")
                self.eat("RPAREN")
                return "operator()"
            if self.is_type_token(self.current_token[0]):
                return f"operator {self.parse_type()}"
            raise SyntaxError(
                f"Expected overloaded operator name, got {self.current_token[0]}"
            )
        raise SyntaxError(
            f"Expected overloaded operator name, got {self.current_token[0]}"
        )

    def parse_member_declarator_name(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "operator"
        ):
            return self.parse_operator_overload_name()
        return self.parse_declarator_identifier()

    def parse_function_declarator_name(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "operator"
        ):
            name = self.parse_operator_overload_name()
        else:
            name = self.parse_declarator_identifier()
        while self.current_token_is_double_colon():
            self.eat_double_colon()
            if (
                self.current_token[0] == "IDENTIFIER"
                and self.current_token[1] == "operator"
            ):
                member = self.parse_operator_overload_name()
            else:
                member = self.parse_identifier()
            name = f"{name}::{member}"
        return name

    def parse_enum(self):
        self.eat("ENUM")
        is_scoped = False
        if self.current_token[0] == "IDENTIFIER" and self.current_token[1] == "class":
            self.eat("IDENTIFIER")
            is_scoped = True
            name = self.current_token[1]
            self.eat("IDENTIFIER")
        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
        else:
            name = self.synthetic_enum_name()

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

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
        enum = EnumNode(name, members)
        enum.is_scoped = is_scoped
        enum.underlying_type = underlying_type
        return enum

    def parse_typedef(self):
        self.eat("TYPEDEF")
        qualifiers = self.parse_qualifiers()

        if self.is_struct_typedef_with_body():
            return self.parse_struct_typedef(qualifiers)

        alias_type = self.parse_type()
        qualifiers.extend(self.parse_post_type_qualifiers())
        attributes = self.parse_attribute_list()
        name = self.parse_typedef_name()
        array_sizes = self.parse_array_suffixes()
        self.eat("SEMICOLON")
        alias = TypeAliasNode(alias_type, name)
        alias.qualifiers = qualifiers
        alias.array_sizes = array_sizes
        alias.attributes = attributes
        return alias

    def is_struct_typedef_with_body(self):
        if self.current_token[0] != "STRUCT":
            return False
        return self.peek()[0] == "LBRACE" or (
            self.is_identifier_token(self.peek()[0]) and self.peek(2)[0] == "LBRACE"
        )

    def parse_struct_typedef(self, qualifiers):
        self.eat("STRUCT")
        struct_attributes = self.parse_attribute_list()

        explicit_name = None
        if (
            self.is_identifier_token(self.current_token[0])
            and self.peek()[0] == "LBRACE"
        ):
            explicit_name = self.parse_identifier()

        self.eat("LBRACE")
        members = []
        methods = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            self.parse_template_declaration_prefixes()
            attributes = self.parse_attribute_list()
            self.parse_template_declaration_prefixes()
            member_qualifiers = self.parse_qualifiers()
            if self.skip_unexpanded_member_macro():
                continue
            if self.current_token_is_keyword("USING", "using"):
                self.parse_using_directive()
                continue
            if self.is_constructor_or_destructor_member(explicit_name):
                methods.append(
                    self.parse_constructor_or_destructor_member(
                        explicit_name,
                        qualifiers=member_qualifiers,
                        attributes=attributes,
                    )
                )
                continue
            if self.current_token[0] == "ENUM":
                members.append(self.parse_enum())
                continue
            if self.current_token[0] == "STRUCT":
                declarations = self.parse_nested_struct_member(
                    explicit_name or "AnonymousStruct",
                    qualifiers=member_qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                )
                members.extend(self.ensure_statement_list(declarations))
                continue
            if not self.is_type_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected type in typedef struct member, got {self.current_token[0]}"
                )
            return_type = self.parse_type()
            member_qualifiers.extend(self.parse_post_type_qualifiers())
            member_name = self.parse_member_declarator_name()
            if self.current_token[0] == "LPAREN":
                methods.append(
                    self.parse_function(
                        return_type,
                        member_name,
                        qualifiers=member_qualifiers,
                        attributes=attributes,
                    )
                )
            else:
                declarations = self.parse_variable_declaration_list_rest(
                    return_type,
                    member_name,
                    qualifiers=member_qualifiers,
                    attributes=attributes,
                    allow_semantic=True,
                    consume_semicolon=True,
                )
                members.extend(self.ensure_statement_list(declarations))

        self.eat("RBRACE")
        alias_name = self.parse_identifier()
        array_sizes = self.parse_array_suffixes()
        self.eat("SEMICOLON")

        struct_name = explicit_name or self.synthetic_struct_type_name(
            "AnonymousStruct", alias_name
        )
        struct_node = StructNode(struct_name, members)
        struct_node.attributes = struct_attributes
        struct_node.methods = methods
        self.synthetic_structs.append(struct_node)

        alias = TypeAliasNode(struct_name, alias_name)
        alias.qualifiers = qualifiers
        alias.array_sizes = array_sizes
        return alias

    def is_constructor_or_destructor_member(self, struct_name):
        if not struct_name:
            return False
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == struct_name
            and self.peek()[0] == "LPAREN"
        ):
            return True
        return (
            self.current_token[0] == "BITWISE_NOT"
            and self.peek()[0] == "IDENTIFIER"
            and self.peek()[1] == struct_name
            and self.peek(2)[0] == "LPAREN"
        )

    def parse_constructor_or_destructor_member(
        self, struct_name, qualifiers=None, attributes=None
    ):
        qualifiers = qualifiers or []
        attributes = attributes or []
        is_destructor = self.current_token[0] == "BITWISE_NOT"

        if is_destructor:
            self.eat("BITWISE_NOT")
            name = f"~{self.parse_identifier()}"
        else:
            name = self.parse_identifier()

        params = self.parse_parameters()
        qualifiers = qualifiers + self.parse_trailing_function_qualifiers()
        if self.current_token[0] == "COLON":
            self.skip_constructor_initializer_list()

        is_prototype = False
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            body = []
            is_prototype = True
        else:
            body = self.parse_block()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

        function = FunctionNode(
            return_type="",
            name=name,
            params=params,
            body=body,
            qualifiers=qualifiers,
            attributes=attributes,
        )
        function.is_constructor = not is_destructor
        function.is_destructor = is_destructor
        function.is_prototype = is_prototype
        return function

    def skip_constructor_initializer_list(self):
        self.eat("COLON")
        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.skip_constructor_initializer_entry()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            break

    def skip_constructor_initializer_entry(self):
        while self.current_token[0] not in {
            "LPAREN",
            "LBRACE",
            "COMMA",
            "SEMICOLON",
            "EOF",
        }:
            if self.current_token[0] == "LESS_THAN":
                self.skip_template_argument_list()
                continue
            self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            self.skip_balanced_delimiter_block("LPAREN", "RPAREN")
        elif self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()

    def skip_balanced_delimiter_block(self, open_token, close_token):
        self.eat(open_token)
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
            self.eat(token_type)

        if depth != 0:
            raise SyntaxError(f"Unterminated {open_token} block")

    def parse_cbuffer(self, attributes=None):
        buffer_kind = self.current_token[1]
        self.eat(self.current_token[0])
        name = None
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

        if self.current_token[0] == "COLON":
            _, cbuffer_register, cbuffer_packoffset = self.parse_semantic_or_register()
        else:
            cbuffer_register = None
            cbuffer_packoffset = None

        if name is None:
            name = self.synthetic_cbuffer_name(cbuffer_register)

        self.eat("LBRACE")
        members = []
        methods = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] in {"CBUFFER", "TBUFFER"}:
                members.append(self.parse_cbuffer(attributes=[]))
                continue
            member_attributes = self.parse_attribute_list()
            qualifiers = self.parse_qualifiers()
            if self.skip_unexpanded_member_macro():
                continue
            if self.current_token[0] == "STRUCT":
                declarations = self.parse_nested_struct_member(
                    name,
                    qualifiers=qualifiers,
                    attributes=member_attributes,
                    allow_semantic=True,
                )
                members.extend(self.ensure_statement_list(declarations))
                continue

            if not self.is_type_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected type in cbuffer member, got {self.current_token[0]}"
                )

            member_type = self.parse_type()
            qualifiers.extend(self.parse_post_type_qualifiers())
            member_attributes.extend(self.parse_attribute_list())
            member_name = self.parse_member_declarator_name()
            if self.current_token[0] == "LPAREN":
                methods.append(
                    self.parse_function(
                        member_type,
                        member_name,
                        qualifiers=qualifiers,
                        attributes=member_attributes,
                    )
                )
                continue

            declarations = self.parse_variable_declaration_list_rest(
                member_type,
                member_name,
                qualifiers=qualifiers,
                attributes=member_attributes,
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
        cbuffer_node.methods = methods
        return cbuffer_node

    def synthetic_cbuffer_name(self, cbuffer_register):
        suffix = ""
        if cbuffer_register:
            suffix = re.sub(r"\W+", "_", cbuffer_register).strip("_")
        base = f"AnonymousCBuffer_{suffix}" if suffix else "AnonymousCBuffer"
        name = base
        index = 1
        while name in self.synthetic_cbuffer_names:
            name = f"{base}_{index}"
            index += 1
        self.synthetic_cbuffer_names.add(name)
        return name

    def parse_function(self, return_type, name, qualifiers, attributes):
        params = self.parse_parameters()
        qualifiers = qualifiers + self.parse_trailing_function_qualifiers()
        return_array_sizes = self.parse_array_suffixes()

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
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

        qualifier = self.infer_function_qualifier(
            name, attributes, params, semantic, body
        )

        function = FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=qualifiers + ([qualifier] if qualifier else []),
            attributes=attributes,
            qualifier=qualifier,
            semantic=semantic,
            array_sizes=return_array_sizes,
        )
        function.is_prototype = is_prototype
        return function

    def parse_trailing_function_qualifiers(self):
        qualifiers = []
        while self.current_token[0] == "CONST":
            qualifiers.append(self.current_token[1])
            self.eat("CONST")
        return qualifiers

    def infer_function_qualifier(self, name, attributes, params, semantic, body=None):
        attribute_names = {str(attr.name).lower() for attr in attributes}
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
        has_mesh_output_parameter = any(
            any(
                str(attr.name).lower() in {"vertices", "indices", "primitives"}
                for attr in getattr(param, "attributes", []) or []
            )
            for param in params
        )
        if "outputtopology" in attribute_names and (
            "numthreads" in attribute_names or has_mesh_output_parameter
        ):
            return "mesh"
        has_geometry_parameter = any(
            self.is_geometry_stream_type(getattr(param, "vtype", None))
            or self.parameter_has_geometry_primitive(param)
            for param in params
        )
        if "maxvertexcount" in attribute_names or has_geometry_parameter:
            return "geometry"
        if (
            "outputcontrolpoints" in attribute_names
            or "patchconstantfunc" in attribute_names
        ):
            return "tessellation_control"
        if "domain" in attribute_names:
            return "tessellation_evaluation"
        if "numthreads" in attribute_names and self.contains_function_call(
            body, "DispatchMesh"
        ):
            return "task"
        if "numthreads" in attribute_names:
            return "compute"
        if semantic:
            semantic_upper = semantic.upper()
            if semantic_upper.startswith("SV_TARGET"):
                return "fragment"
            if semantic_upper == "SV_POSITION":
                return "vertex"
        for param in params:
            if not getattr(param, "semantic", None):
                continue

            semantic_names = self.semantic_names_for_stage_inference(param.semantic)
            if self.parameter_has_output_qualifier(param):
                if any(
                    self.is_fragment_output_semantic(name) for name in semantic_names
                ):
                    return "fragment"
                if any(self.is_vertex_output_semantic(name) for name in semantic_names):
                    return "vertex"
            if any(self.is_vertex_input_semantic(name) for name in semantic_names):
                return "vertex"
        for param in params:
            if str(getattr(param, "semantic", "")).upper() == "SV_DISPATCHTHREADID":
                return "compute"
        return None

    def contains_function_call(self, node, function_name):
        target_name = str(function_name).lower()
        visited = set()

        def walk(current):
            if current is None or isinstance(current, (str, int, float, bool)):
                return False

            current_id = id(current)
            if current_id in visited:
                return False
            visited.add(current_id)

            if isinstance(current, FunctionCallNode):
                callee = getattr(current, "name", None)
                if isinstance(callee, str) and callee.lower() == target_name:
                    return True

            if isinstance(current, dict):
                return any(walk(value) for value in current.values())

            if isinstance(current, (list, tuple, set)):
                return any(walk(value) for value in current)

            if hasattr(current, "__dict__"):
                return any(walk(value) for value in vars(current).values())

            return False

        return walk(node)

    def is_geometry_stream_type(self, param_type):
        base = str(param_type or "").split("<", 1)[0].strip()
        return base in {"PointStream", "LineStream", "TriangleStream"}

    def parameter_has_geometry_primitive(self, param):
        return any(
            str(getattr(attr, "name", "")).lower() == "primitive"
            for attr in getattr(param, "attributes", []) or []
        )

    def semantic_names_for_stage_inference(self, semantic):
        return [
            part.strip().upper() for part in str(semantic).split(":") if part.strip()
        ]

    def parameter_has_output_qualifier(self, param):
        return any(
            str(qualifier).lower() in {"out", "inout"}
            for qualifier in getattr(param, "qualifiers", []) or []
        )

    def is_fragment_output_semantic(self, semantic):
        return (
            semantic.startswith("SV_TARGET")
            or semantic.startswith("SV_DEPTH")
            or semantic == "SV_COVERAGE"
        )

    def is_vertex_output_semantic(self, semantic):
        return semantic == "SV_POSITION" or semantic.startswith(
            ("SV_CLIPDISTANCE", "SV_CULLDISTANCE")
        )

    def is_vertex_input_semantic(self, semantic):
        return semantic in {"SV_VERTEXID", "SV_INSTANCEID"}

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
                qualifiers = self.parse_parameter_modifiers(
                    attributes, primitive_qualifiers, mesh_parameter_roles
                )
                if not self.is_type_token(self.current_token[0]):
                    raise SyntaxError(
                        f"Unexpected token in parameter list: {self.current_token[0]}"
                    )
                param_type = self.parse_type()
                qualifiers.extend(self.parse_post_type_qualifiers())
                attributes.extend(self.parse_attribute_list())

                if self.is_declarator_identifier_token(self.current_token[0]):
                    name = self.parse_declarator_identifier()
                else:
                    name = ""

                array_sizes = self.parse_array_suffixes()
                (
                    semantic,
                    _,
                    _,
                    interpolation_qualifiers,
                ) = self.parse_parameter_semantic_and_interpolation()
                qualifiers.extend(interpolation_qualifiers)
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

    def parse_parameter_modifiers(
        self, attributes, primitive_qualifiers, mesh_parameter_roles
    ):
        qualifiers = []
        while True:
            before_index = self.current_index
            qualifiers.extend(self.parse_qualifiers())

            while self.is_unexpanded_parameter_modifier_macro_at(self.current_index):
                self.eat("IDENTIFIER")
                qualifiers.extend(self.parse_qualifiers())

            if (
                self.current_token[0] == "IDENTIFIER"
                and self.current_token[1] in primitive_qualifiers
            ):
                attributes.append(AttributeNode("primitive", [self.current_token[1]]))
                self.eat("IDENTIFIER")
                continue

            has_direction_qualifier = any(
                str(qualifier).lower() in {"in", "out", "inout"}
                for qualifier in qualifiers
            )
            while has_direction_qualifier and self.is_parameter_role_token_at(
                self.current_index, mesh_parameter_roles
            ):
                role = self.current_token[1].lower()
                attributes.append(AttributeNode(mesh_parameter_roles[role]))
                self.eat("IDENTIFIER")

            if self.current_index == before_index:
                break
        return qualifiers

    def is_unexpanded_parameter_modifier_macro_at(self, index):
        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type != "IDENTIFIER":
            return False
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(token_value)):
            return False

        type_start = index + 1
        while type_start < len(self.tokens) and self.is_qualifier_token_at(type_start):
            type_start += 1

        type_end = self.skip_type_name_at(type_start)
        return type_end is not None and self.is_declarator_identifier_token_at(type_end)

    def is_parameter_role_token_at(self, index, roles):
        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type != "IDENTIFIER" or token_value.lower() not in roles:
            return False

        type_end = self.skip_type_name_at(index + 1)
        return type_end is not None and self.is_declarator_identifier_token_at(type_end)

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
        attributes.extend(self.parse_unexpanded_statement_modifier_macros())
        attributes.extend(self.parse_attribute_list())

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

        if self.current_token_is_keyword("USING", "using"):
            self.parse_using_directive()
            return None

        if self.current_token[0] == "TYPEDEF":
            self.parse_typedef()
            return None

        if self.is_class_declaration_prefix():
            self.parse_class_declaration()
            return None

        if self.skip_unexpanded_statement_macro():
            return None

        if self.looks_like_function_prototype():
            self.parse_local_function_prototype(attributes)
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

        if self.current_token[0] == "STRUCT":
            return self.parse_struct()

        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return self.attach_attributes(expr, attributes)

    def looks_like_function_prototype(self):
        idx = self.current_index
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        idx = self.skip_function_modifier_attributes_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        idx = self.skip_type_name_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_post_type_qualifier_token_at(idx):
            idx += 1
        idx = self.skip_attribute_lists_at(idx)
        if idx is None or not self.is_declarator_identifier_token_at(idx):
            return False

        idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return False
        idx = self.skip_parenthesized_list_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_post_type_qualifier_token_at(idx):
            idx += 1
        idx = self.skip_array_suffixes_at(idx)
        if idx is None or idx >= len(self.tokens):
            return False

        return self.tokens[idx][0] == "SEMICOLON"

    def skip_unexpanded_statement_macro(self):
        if not self.is_macro_like_identifier(self.current_token):
            return False
        if self.peek()[0] != "LPAREN":
            return False

        end_index = self.skip_parenthesized_list_at(self.current_index + 1)
        if end_index is None or end_index >= len(self.tokens):
            return False
        if self.tokens[end_index][0] == "SEMICOLON":
            return False
        if self.tokens[end_index][0] in {
            "COMMA",
            "RPAREN",
            "RBRACKET",
            "PLUS",
            "MINUS",
            "MULTIPLY",
            "DIVIDE",
            "MOD",
            "BITWISE_AND",
            "BITWISE_OR",
            "BITWISE_XOR",
            "SHIFT_LEFT",
            "SHIFT_RIGHT",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "LOGICAL_AND",
            "LOGICAL_OR",
            *ASSIGNMENT_TOKENS,
        }:
            return False

        self.eat("IDENTIFIER")
        self.skip_balanced_delimiter_block("LPAREN", "RPAREN")
        return True

    def parse_unexpanded_statement_modifier_macros(self):
        attributes = []
        while (
            self.is_macro_like_identifier(self.current_token)
            and self.peek()[0] in STATEMENT_START_TOKENS
        ):
            macro_name = str(self.current_token[1])
            attribute_name = UNEXPANDED_STATEMENT_ATTRIBUTE_MACROS.get(macro_name)
            if attribute_name:
                attributes.append(AttributeNode(attribute_name))
            self.eat("IDENTIFIER")
        return attributes

    def parse_local_function_prototype(self, attributes=None):
        qualifiers = self.parse_qualifiers()
        attributes = list(attributes or [])
        attributes.extend(self.parse_function_modifier_attributes())
        qualifiers.extend(self.parse_qualifiers())
        return_type = self.parse_type()
        qualifiers.extend(self.parse_post_type_qualifiers())
        attributes.extend(self.parse_attribute_list())
        name = self.parse_function_declarator_name()
        return self.parse_function(return_type, name, qualifiers, attributes)

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
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        idx = self.skip_type_name_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_post_type_qualifier_token_at(idx):
            idx += 1

        idx = self.skip_attribute_lists_at(idx)
        if idx is None:
            return False

        if idx >= len(self.tokens) or not self.is_declarator_identifier_token_at(idx):
            return False
        idx += 1
        idx = self.skip_array_suffixes_at(idx)
        if idx is None or idx >= len(self.tokens):
            return False

        return self.tokens[idx][0] in {
            "SEMICOLON",
            "EQUALS",
            "LBRACE",
            "COLON",
            "COMMA",
        }

    def looks_like_external_declaration(self):
        idx = self.current_index
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        idx = self.skip_function_modifier_attributes_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        idx = self.skip_type_name_at(idx)
        if idx is None:
            return False
        while idx < len(self.tokens) and self.is_post_type_qualifier_token_at(idx):
            idx += 1
        idx = self.skip_attribute_lists_at(idx)
        if idx is None:
            return False
        if not self.is_declarator_identifier_token_at(idx):
            return False

        idx += 1
        idx = self.skip_array_suffixes_at(idx)
        if idx is None or idx >= len(self.tokens):
            return False

        token_type = self.tokens[idx][0]
        if token_type in {"LPAREN", "SEMICOLON", "EQUALS", "LBRACE", "COLON"}:
            return True

        while token_type == "COMMA":
            idx += 1
            if not self.is_declarator_identifier_token_at(idx):
                return False
            idx += 1
            idx = self.skip_array_suffixes_at(idx)
            if idx is None or idx >= len(self.tokens):
                return False
            token_type = self.tokens[idx][0]
            if token_type in {"EQUALS", "LBRACE", "SEMICOLON", "COLON"}:
                return True

        return False

    def skip_function_modifier_attributes_at(self, idx):
        while self.is_clipplanes_modifier_at(idx):
            idx = self.skip_parenthesized_list_at(idx + 1)
            if idx is None:
                return None
        return idx

    def is_clipplanes_modifier_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx][0] == "IDENTIFIER"
            and str(self.tokens[idx][1]).lower() == "clipplanes"
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def skip_parenthesized_list_at(self, idx):
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return None

        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif token_type == "EOF":
                return None
            idx += 1
        return None

    def skip_attribute_lists_at(self, idx):
        while (
            idx + 1 < len(self.tokens)
            and self.tokens[idx][0] == "LBRACKET"
            and self.tokens[idx + 1][0] == "LBRACKET"
        ):
            idx += 2
            depth = 1
            while idx < len(self.tokens):
                token_type = self.tokens[idx][0]
                if token_type == "EOF":
                    return None
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
            else:
                return None

            if idx >= len(self.tokens) or self.tokens[idx][0] != "RBRACKET":
                return None
            idx += 1

        return idx

    def skip_array_suffixes_at(self, idx):
        while idx < len(self.tokens) and self.tokens[idx][0] == "LBRACKET":
            depth = 0
            while idx < len(self.tokens):
                token_type = self.tokens[idx][0]
                if token_type == "LBRACKET":
                    depth += 1
                elif token_type == "RBRACKET":
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                elif token_type == "EOF":
                    return None
                idx += 1
            else:
                return None
        return idx

    def skip_type_name_at(self, idx):
        if self.token_at_is_double_colon(idx):
            idx += 2

        if idx >= len(self.tokens) or not self.is_type_token(self.tokens[idx][0]):
            return None

        if self.tokens[idx][0] == "STRUCT":
            idx += 1
            if not self.is_identifier_token_at(idx):
                return None
            return idx + 1

        base = self.tokens[idx][1]
        idx += 1
        if (
            base in COMPOSITE_TYPE_PREFIXES
            and idx < len(self.tokens)
            and self.tokens[idx][0] in COMPOSITE_TYPE_TOKENS
        ):
            idx += 1

        while (
            idx + 2 < len(self.tokens)
            and self.tokens[idx][0] == "COLON"
            and self.tokens[idx + 1][0] == "COLON"
            and self.tokens[idx + 2][0] == "IDENTIFIER"
        ):
            idx += 3

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
                elif self.tokens[idx][0] == "SHIFT_RIGHT" and depth > 1:
                    depth -= 2
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
            else:
                return None

        return idx

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
        qualifiers.extend(self.parse_post_type_qualifiers())
        attributes.extend(self.parse_attribute_list())
        name = self.parse_declarator_identifier()
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
            name = self.parse_declarator_identifier()

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
        qualifiers = list(qualifiers or [])
        attributes = list(attributes or [])

        array_sizes = self.parse_array_suffixes()
        attributes.extend(self.parse_attribute_list())
        bit_width = self.parse_bitfield_width()
        attributes.extend(self.parse_attribute_list())
        semantic = None
        register = None
        packoffset = None

        if allow_semantic:
            semantic, register, packoffset = self.parse_semantic_or_register()

        self.skip_effect_declaration_suffixes()

        value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

        self.skip_effect_declaration_suffixes()

        sampler_state = None
        if self.is_sampler_state_type(vtype) and self.current_token[0] == "LBRACE":
            sampler_state = self.parse_sampler_state_block()
            if self.is_sampler_state_initializer(value):
                value = None
        elif self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()

        self.skip_effect_declaration_suffixes()

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
        var.bit_width = bit_width
        var.register = register
        var.packoffset = packoffset
        if sampler_state is not None:
            var.sampler_state = sampler_state
        return var

    def parse_bitfield_width(self):
        if self.current_token[0] != "COLON":
            return None
        if self.peek()[0] not in {
            "NUMBER",
            "HEX_NUMBER",
            "BINARY_NUMBER",
            "OCT_NUMBER",
        }:
            return None

        self.eat("COLON")
        return self.parse_expression()

    def skip_effect_declaration_suffixes(self):
        while self.current_token[0] == "LESS_THAN":
            self.skip_effect_annotation_block()

    def skip_effect_annotation_block(self):
        self.eat("LESS_THAN")
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "GREATER_THAN" and self.peek()[0] in {
                "LBRACE",
                "COMMA",
                "SEMICOLON",
                "EQUALS",
                "RBRACE",
                "EOF",
            }:
                self.eat("GREATER_THAN")
                return
            self.eat(self.current_token[0])

        raise SyntaxError("Unterminated effect annotation block")

    def is_sampler_state_type(self, vtype):
        return str(vtype).split("<", 1)[0] in {
            "SamplerState",
            "SamplerComparisonState",
            "sampler",
            "sampler_state",
        }

    def is_sampler_state_initializer(self, value):
        return isinstance(value, str) and value.lower() == "sampler_state"

    def parse_sampler_state_block(self):
        self.eat("LBRACE")
        state = []
        while self.current_token[0] != "RBRACE":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            value = self.parse_sampler_state_value()
            self.eat("SEMICOLON")
            state.append((name, value))
        self.eat("RBRACE")
        return state

    def parse_sampler_state_value(self):
        if self.current_token[0] == "LESS_THAN":
            return self.parse_sampler_state_angle_value()
        return self.parse_expression()

    def parse_sampler_state_angle_value(self):
        self.eat("LESS_THAN")
        tokens = []
        depth = 1
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LESS_THAN":
                depth += 1
                tokens.append(self.current_token)
                self.eat("LESS_THAN")
                continue
            if token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    return self.format_generic_argument_tokens(tokens)
                tokens.append(self.current_token)
                self.eat("GREATER_THAN")
                continue
            tokens.append(self.current_token)
            self.eat(token_type)

        raise SyntaxError("Unterminated sampler_state angle-bracket value")

    def parse_condition_expression(self):
        if self.looks_like_declaration():
            qualifiers = self.parse_qualifiers()
            return self.parse_variable_declaration(
                qualifiers=qualifiers,
                attributes=[],
                allow_semantic=False,
                consume_semicolon=False,
            )
        return self.parse_expression()

    def parse_hlsl_condition_header(self, keyword):
        self.eat(keyword)
        if self.current_token[0] == "COLON":
            self.eat("COLON")
        self.eat("LPAREN")
        condition = self.parse_condition_expression()
        self.eat("RPAREN")
        return condition

    def parse_if_statement(self):
        condition = self.parse_hlsl_condition_header("IF")

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
        condition = self.parse_hlsl_condition_header("ELSE_IF")
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
        condition = self.parse_condition_expression()
        self.eat("RPAREN")
        body = self.parse_statement_or_block()
        return WhileNode(condition, body)

    def parse_do_while_loop(self):
        self.eat("DO")
        body = self.parse_statement_or_block()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_condition_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        condition = self.parse_condition_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default_body = None

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "CASE":
                case = self.parse_switch_case()
                cases.append(case)
                cases.extend(getattr(case, "additional_cases", []))
                nested_default = getattr(case, "additional_default_body", None)
                if nested_default is not None:
                    default_body = nested_default
            elif self.current_token[0] == "DEFAULT":
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
            else:
                if default_body is None:
                    default_body = []
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
        additional_cases = []
        additional_default_body = None
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            if self.current_token[0] == "LBRACE":
                (
                    block_body,
                    block_cases,
                    block_default_body,
                ) = self.parse_switch_scoped_block()
                body.extend(block_body)
                additional_cases.extend(block_cases)
                if block_default_body is not None:
                    additional_default_body = block_default_body
                continue

            stmt = self.parse_statement()
            if stmt is None:
                continue
            if isinstance(stmt, list):
                body.extend(stmt)
            else:
                body.append(stmt)

        case = CaseNode(value, body)
        case.additional_cases = additional_cases
        case.additional_default_body = additional_default_body
        return case

    def parse_switch_scoped_block(self):
        self.eat("LBRACE")
        body = []
        cases = []
        default_body = None

        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "CASE":
                case = self.parse_switch_case()
                body.extend(case.body)
                cases.append(case)
                cases.extend(getattr(case, "additional_cases", []))
                nested_default = getattr(case, "additional_default_body", None)
                if nested_default is not None:
                    default_body = nested_default
                continue

            if self.current_token[0] == "DEFAULT":
                self.eat("DEFAULT")
                self.eat("COLON")
                default_body = []
                while self.current_token[0] not in [
                    "CASE",
                    "DEFAULT",
                    "RBRACE",
                    "EOF",
                ]:
                    if self.current_token[0] == "LBRACE":
                        (
                            block_body,
                            block_cases,
                            block_default_body,
                        ) = self.parse_switch_scoped_block()
                        default_body.extend(block_body)
                        body.extend(block_body)
                        cases.extend(block_cases)
                        if block_default_body is not None:
                            default_body = block_default_body
                        continue

                    stmt = self.parse_statement()
                    if stmt is None:
                        continue
                    if isinstance(stmt, list):
                        default_body.extend(stmt)
                        body.extend(stmt)
                    else:
                        default_body.append(stmt)
                        body.append(stmt)
                continue

            stmt = self.parse_statement()
            if stmt is None:
                continue
            if isinstance(stmt, list):
                body.extend(stmt)
            else:
                body.append(stmt)

        self.eat("RBRACE")
        return body, cases, default_body

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
            target_type = self.parse_cast_target_type()
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

    def parse_cast_target_type(self):
        self.parse_qualifiers()
        target_type = self.parse_type()
        self.parse_post_type_qualifiers()
        target_type += self.format_array_suffixes_for_type(self.parse_array_suffixes())
        return target_type

    def looks_like_cast(self):
        if self.current_token[0] != "LPAREN":
            return False

        idx = self.skip_cast_target_type_at(self.current_index + 1)
        if idx is None:
            return False
        if idx >= len(self.tokens) or self.tokens[idx][0] != "RPAREN":
            return False

        next_token = self.tokens[idx + 1] if idx + 1 < len(self.tokens) else ("EOF", "")
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

    def skip_cast_target_type_at(self, idx):
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        idx = self.skip_type_name_at(idx)
        if idx is None:
            return None

        while idx < len(self.tokens) and self.is_post_type_qualifier_token_at(idx):
            idx += 1

        return self.skip_array_suffixes_at(idx)

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
                member = self.parse_identifier()
                if self.looks_like_template_call_arguments():
                    member = self.format_templated_name(
                        member, self.parse_generic_arguments()
                    )
                expr = MemberAccessNode(expr, member)
            elif self.current_token_is_double_colon():
                expr = self.parse_scoped_name(expr)
            elif self.looks_like_template_call_arguments() and isinstance(expr, str):
                expr = self.format_templated_name(expr, self.parse_generic_arguments())
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

    def format_templated_name(self, name, args):
        return f"{name}<{', '.join(args)}>"

    def looks_like_template_call_arguments(self):
        if self.current_token[0] != "LESS_THAN":
            return False

        depth = 0
        idx = self.current_index
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
            elif token_type == "SHIFT_RIGHT" and depth > 1:
                depth -= 2
                if depth == 0:
                    next_token = (
                        self.tokens[idx + 1]
                        if idx + 1 < len(self.tokens)
                        else ("EOF", "")
                    )
                    return next_token[0] == "LPAREN"
            elif depth == 1 and token_type in {"SEMICOLON", "RPAREN", "RBRACE"}:
                return False
            elif token_type == "EOF":
                return False
            idx += 1
        return False

    def parse_scoped_name(self, prefix):
        if not isinstance(prefix, str):
            raise SyntaxError("Expected identifier before scoped name separator")

        scoped_name = prefix
        while self.current_token_is_double_colon():
            self.eat_double_colon()
            member = self.parse_identifier()
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
        if self.peek()[0] == "LPAREN":
            return True
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
            elif token_type == "SHIFT_RIGHT" and depth > 1:
                depth -= 2
                if depth == 0:
                    next_token = (
                        self.tokens[idx + 1]
                        if idx + 1 < len(self.tokens)
                        else ("EOF", "")
                    )
                    return next_token[0] == "LPAREN"
            elif depth == 1 and token_type in {"SEMICOLON", "RPAREN", "RBRACE"}:
                return False
            elif token_type == "EOF":
                return False
            idx += 1
        return False

    def parse_primary_expression(self):
        token_type, value = self.current_token
        if self.current_token_is_double_colon():
            self.eat_double_colon()
            if not self.is_identifier_token(self.current_token[0]):
                raise SyntaxError(
                    f"Expected identifier after global scope qualifier, got {self.current_token[0]}"
                )
            scoped_name = self.parse_identifier()
            if self.current_token_is_double_colon():
                scoped_name = self.parse_scoped_name(scoped_name)
            return scoped_name
        if self.is_template_type_constructor_start():
            type_name = self.parse_type()
            args = self.parse_call_arguments()
            return VectorConstructorNode(type_name, args)
        if self.is_effect_compile_expression_start():
            return self.parse_effect_compile_expression()
        if self.is_identifier_token(token_type):
            self.eat(token_type)
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
            if self.current_token[0] == "COMMA":
                elements = [expr]
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    elements.append(self.parse_expression())
                self.eat("RPAREN")
                return InitializerListNode(elements)
            self.eat("RPAREN")
            return expr
        if token_type == "LBRACE":
            return self.parse_initializer_list()
        if token_type in [
            *SCALAR_CONSTRUCTOR_TOKENS,
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

    def is_effect_compile_expression_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and str(self.current_token[1]).lower() == "compile"
            and self.peek()[0] == "IDENTIFIER"
        )

    def parse_effect_compile_expression(self):
        self.eat("IDENTIFIER")
        profile = self.parse_identifier()
        entry = self.parse_postfix_expression()
        return FunctionCallNode("compile", [profile, entry])

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
        if "#INF" in value.upper():
            return float("inf")
        if "#" in value:
            return float("nan")
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
