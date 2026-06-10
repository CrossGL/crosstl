"""Parser for Metal source AST construction."""

import sys

from .MetalAst import *
from .MetalLexer import *

MIN_METAL_PARSE_RECURSION_LIMIT = 10000


def ensure_metal_parse_recursion_limit():
    # Dawn/Tint generated MSL can contain deeply nested braced constructor chains.
    if sys.getrecursionlimit() < MIN_METAL_PARSE_RECURSION_LIMIT:
        sys.setrecursionlimit(MIN_METAL_PARSE_RECURSION_LIMIT)


# Token groups for parsing
QUALIFIER_TOKENS = {
    "CONSTANT",
    "DEVICE",
    "THREADGROUP",
    "THREADGROUP_IMAGEBLOCK",
    "THREAD",
    "CONST",
    "CONSTEXPR",
    "STATIC",
    "INLINE",
    "VOLATILE",
    "RESTRICT",
    "READ",
    "WRITE",
    "READ_WRITE",
}

TYPE_TOKENS = {
    "VOID",
    "FLOAT",
    "HALF",
    "DOUBLE",
    "INT",
    "UINT",
    "LONG",
    "ULONG",
    "SHORT",
    "USHORT",
    "CHAR",
    "UCHAR",
    "BOOL",
    "SIZE_T",
    "PTRDIFF_T",
    "INT64_T",
    "UINT64_T",
    "INT8_T",
    "UINT8_T",
    "INT16_T",
    "UINT16_T",
    "INT32_T",
    "UINT32_T",
    "VECTOR",
    "PACKED_VECTOR",
    "SIMD_VECTOR",
    "MATRIX",
    "SIMD_MATRIX",
    "ATOMIC_INT",
    "ATOMIC_UINT",
    "ATOMIC_BOOL",
    "TEXTURE1D",
    "TEXTURE1D_ARRAY",
    "TEXTURE2D",
    "TEXTURE2D_MS",
    "TEXTURE2D_MS_ARRAY",
    "TEXTURE3D",
    "TEXTURECUBE",
    "TEXTURECUBE_ARRAY",
    "TEXTURE2D_ARRAY",
    "TEXTUREBUFFER",
    "DEPTH2D",
    "DEPTH2D_ARRAY",
    "DEPTHCUBE",
    "DEPTHCUBE_ARRAY",
    "DEPTH2D_MS",
    "DEPTH2D_MS_ARRAY",
    "ACCELERATION_STRUCTURE",
    "INTERSECTION_FUNCTION_TABLE",
    "VISIBLE_FUNCTION_TABLE",
    "INDIRECT_COMMAND_BUFFER",
    "SAMPLER",
    "IDENTIFIER",
    "METAL",
    "ENUM",
    "TYPEDEF",
}

STAGE_TOKENS = {
    "VERTEX",
    "FRAGMENT",
    "KERNEL",
    "INTERSECTION",
    "ANYHIT",
    "CLOSESTHIT",
    "MISS",
    "CALLABLE",
    "MESH",
    "OBJECT",
    "AMPLIFICATION",
}
SCOPED_IDENTIFIER_PART_TOKENS = (
    TYPE_TOKENS
    | STAGE_TOKENS
    | {
        "METAL",
        "READ",
        "WRITE",
        "READ_WRITE",
    }
)

UNARY_KEYWORDS = {"SIZEOF", "ALIGNOF"}
MACRO_QUALIFIERS = {"METAL_FUNC", "STEEL_CONST", "C10_METAL_CONSTEXPR"}
MACRO_QUALIFIER_ALIASES = {
    "STEEL_CONST": ("constant",),
    "C10_METAL_CONSTEXPR": ("constant", "constexpr"),
}
IDENTIFIER_TYPE_QUALIFIERS = MACRO_QUALIFIERS | {
    "__restrict",
    "__restrict__",
    "object_data",
}
IDENTIFIER_PREFIX_TYPE_QUALIFIERS = {"out", "inout"}
RAYTRACING_TYPE_QUALIFIERS = {"ray_data"}
TYPE_QUALIFIER_FUNCTIONS = {"coherent"}
SIGNED_TYPE_PREFIXES = {"signed", "unsigned"}
SIGNED_PREFIX_TYPE_TOKENS = {"CHAR", "SHORT", "INT", "LONG"}
KEYWORD_IDENTIFIER_TOKENS = {"BUFFER", "SAMPLER", "METAL", "RESTRICT"}
TYPE_IDENTIFIER_TOKENS = {"PACKED_VECTOR"}
GNU_EXTENSION_PREFIXES = {"__extension__"}
STRUCT_METHOD_PREFIXES = {"virtual"}
AGGREGATE_ALIGNMENT_MACROS = {"mittens_DEFAULT_ALIGN"}
OPERATOR_OVERLOAD_TOKENS = {
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "MOD",
    "EQUAL",
    "NOT_EQUAL",
    "LESS_THAN",
    "GREATER_THAN",
    "LESS_EQUAL",
    "GREATER_EQUAL",
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
    "AND",
    "OR",
    "NOT",
    "BITWISE_NOT",
    "BITWISE_AND",
    "BITWISE_OR",
    "BITWISE_XOR",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "INCREMENT",
    "DECREMENT",
    "EQUALS",
    "LBRACKET",
    "LPAREN",
}
TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS = {
    "SEMICOLON",
    "COMMA",
    "RPAREN",
    "RBRACKET",
    "RBRACE",
    "QUESTION",
    "COLON",
    "AND",
    "OR",
    "BITWISE_OR",
    "BITWISE_XOR",
    "BITWISE_AND",
    "EQUAL",
    "NOT_EQUAL",
    "LESS_THAN",
    "GREATER_THAN",
    "LESS_EQUAL",
    "GREATER_EQUAL",
    "SHIFT_LEFT",
    "SHIFT_RIGHT",
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "MOD",
}
TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS = TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS | {
    "LPAREN",
    "SCOPE",
    "LBRACE",
}
CONSTRUCTOR_TYPE_TOKENS = TYPE_TOKENS - {
    "VOID",
    "IDENTIFIER",
    "METAL",
    "ENUM",
    "TYPEDEF",
}
STAGE_ATTRIBUTE_NAMES = {
    "vertex",
    "fragment",
    "kernel",
    "intersection",
    "anyhit",
    "closesthit",
    "miss",
    "callable",
    "mesh",
    "object",
    "amplification",
}


class MetalParser:
    """Parse Metal tokens into the Metal backend shader AST."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )
        self.skip_comments()
        self.known_types = {
            "void",
            "bool",
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "float",
            "half",
            "double",
            "size_t",
            "ptrdiff_t",
            "int64_t",
            "uint64_t",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "sampler",
            "texture1d",
            "texture1d_array",
            "texture2d",
            "texture2d_array",
            "texture2d_ms",
            "texture2d_ms_array",
            "texture3d",
            "texturecube",
            "texturecube_array",
            "texture_buffer",
            "depth2d",
            "depth2d_array",
            "depth2d_ms",
            "depth2d_ms_array",
            "depthcube",
            "depthcube_array",
            "acceleration_structure",
            "intersection_function_table",
            "visible_function_table",
            "indirect_command_buffer",
            "atomic_int",
            "atomic_uint",
            "atomic_bool",
            "enum",
            "ray",
            "ray_data",
            "intersection_result",
            "intersection_params",
            "triangle_intersection_params",
            "intersector",
            "packed_float2",
            "packed_float3",
            "packed_float4",
            "packed_half2",
            "packed_half3",
            "packed_half4",
            "packed_int2",
            "packed_int3",
            "packed_int4",
            "packed_uint2",
            "packed_uint3",
            "packed_uint4",
            "simd_float2",
            "simd_float3",
            "simd_float4",
            "simd_float2x2",
            "simd_float3x3",
            "simd_float4x4",
            "simd_int2",
            "simd_int3",
            "simd_int4",
            "simd_uint2",
            "simd_uint3",
            "simd_uint4",
            "vector_float2",
            "vector_float3",
            "vector_float4",
            "matrix_float3x3",
            "matrix_float4x4",
        }
        self.local_variable_scopes = []
        self.pending_block_scope_names = []
        self.known_variable_templates = set()
        self.known_function_templates = set()
        self.namespace_aliases = {}

    def skip_comments(self):
        while self.pos < len(self.tokens) and self.current_token[0] in [
            "COMMENT_SINGLE",
            "COMMENT_MULTI",
        ]:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def peek(self, offset=1):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return ("EOF", None)

    def parse(self):
        ensure_metal_parse_recursion_limit()
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        functions = []
        preprocessors = []
        structs = []
        enums = []
        typedefs = []
        constants = []
        global_variables = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                directive = self.parse_preprocessor_directive()
                if directive is not None:
                    preprocessors.append(directive)
            elif self.current_token[0] == "NAMESPACE":
                self.parse_namespace_start()
            elif self.current_token[0] == "USING":
                alias = self.parse_using_statement()
                if isinstance(alias, StructNode):
                    structs.append(alias)
                elif alias is not None:
                    typedefs.append(alias)
            elif self.is_union_declaration_start():
                union = self.parse_union()
                if union is not None:
                    structs.append(union)
                    global_variables.extend(getattr(union, "trailing_declarations", []))
            elif self.is_bare_macro_invocation():
                self.skip_bare_macro_invocation()
            elif self.is_top_level_expression_statement_start():
                self.parse_expression_statement()
            elif self.is_template_declaration_start():
                declaration = self.parse_template_declaration()
                if isinstance(declaration, FunctionNode):
                    functions.append(declaration)
                elif isinstance(declaration, StructNode):
                    structs.append(declaration)
            elif self.current_token[0] == "STRUCT":
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    struct = self.parse_struct()
                    if struct is not None:
                        structs.append(struct)
                        global_variables.extend(
                            getattr(struct, "trailing_declarations", [])
                        )
            elif self.current_token[0] == "CLASS":
                class_node = self.parse_class()
                if class_node is not None:
                    structs.append(class_node)
                    global_variables.extend(
                        getattr(class_node, "trailing_declarations", [])
                    )
            elif self.current_token[0] == "ALIGNAS":
                alignas_specs = self.parse_alignas_specifiers()
                if self.current_token[0] in {"STRUCT", "CLASS"}:
                    struct = (
                        self.parse_struct(alignas_specs)
                        if self.current_token[0] == "STRUCT"
                        else self.parse_class(alignas_specs)
                    )
                    if struct is not None:
                        structs.append(struct)
                        global_variables.extend(
                            getattr(struct, "trailing_declarations", [])
                        )
                else:
                    self.add_global_declaration(
                        global_variables,
                        self.parse_global_variable(pre_alignas=alignas_specs),
                    )
            elif self.current_token[0] == "ENUM":
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    enums.append(self.parse_enum())
            elif self.current_token[0] == "TYPEDEF":
                typedef = self.parse_typedef()
                if isinstance(typedef, StructNode):
                    structs.append(typedef)
                elif isinstance(typedef, EnumNode):
                    enums.append(typedef)
                elif typedef is not None:
                    typedefs.append(typedef)
            elif self.current_token[0] == "STATIC_ASSERT":
                global_variables.append(self.parse_static_assert())
            elif self.is_operator_function_definition():
                self.skip_operator_function_definition()
            elif self.is_decltype_template_instantiation_declaration():
                self.skip_decltype_template_instantiation_declaration()
            elif self.is_top_level_parameter_fragment_start():
                self.parse_top_level_parameter_fragment()
            elif self.is_qualified_aggregate_declaration_start():
                aggregate = self.parse_qualified_aggregate_declaration()
                if aggregate is not None:
                    structs.append(aggregate)
                    global_variables.extend(
                        getattr(aggregate, "trailing_declarations", [])
                    )
            elif self.current_token[0] == "CONSTANT":
                if self.is_constant_buffer():
                    constants.append(self.parse_constant_buffer())
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            elif self.current_token[0] == "ATTRIBUTE" or self.is_gnu_attribute_start():
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            elif self.current_token[0] in STAGE_TOKENS or (
                self.current_token[0] in TYPE_TOKENS
                or self.current_token[0] in QUALIFIER_TOKENS
            ):
                if self.is_function_definition():
                    function = self.parse_function()
                    if function is not None:
                        functions.append(function)
                else:
                    self.add_global_declaration(
                        global_variables, self.parse_global_variable()
                    )
            else:
                self.eat(self.current_token[0])

        return ShaderNode(
            includes=preprocessors,
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            constant=constants,
            enums=enums,
            typedefs=typedefs,
        )

    def add_global_declaration(self, global_variables, declaration):
        if isinstance(declaration, list):
            global_variables.extend(declaration)
        else:
            global_variables.append(declaration)

    def is_bare_macro_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.peek(1)[0] == "LPAREN"
            and not self.is_gnu_attribute_start()
        )

    def is_top_level_expression_statement_start(self):
        return self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "DOT"

    def is_top_level_parameter_fragment_start(self):
        if not (
            self.current_token[0] in TYPE_TOKENS
            or self.current_token[0] in QUALIFIER_TOKENS
            or self.is_type_qualifier_start()
        ):
            return False

        idx = self.pos
        depth = 0
        saw_comma = False
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "EOF":
                return saw_comma
            if depth == 0 and token_type in {"SEMICOLON", "LBRACE"}:
                return False
            if depth == 0 and token_type == "COMMA":
                saw_comma = True
            if token_type in {"LPAREN", "LBRACKET", "LESS_THAN"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "GREATER_THAN"} and depth > 0:
                depth -= 1
            idx += 1
        return False

    def parse_top_level_parameter_fragment(self):
        while self.current_token[0] != "EOF":
            self.parse_attributes()
            self.parse_type_specifier()
            self.parse_declarator()
            self.parse_attributes()
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                self.parse_expression()
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                break
            if self.current_token[0] == "EOF":
                break
            raise SyntaxError(
                f"Expected comma or end of parameter fragment, got {self.current_token[0]}"
            )

    def skip_bare_macro_invocation(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise SyntaxError("Unterminated macro invocation")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_namespace_start(self):
        self.eat("NAMESPACE")
        namespace_name = None
        if self.current_token[0] in {"IDENTIFIER", "METAL"}:
            namespace_name = self.parse_scoped_identifier()
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            target_name = self.parse_scoped_identifier()
            if namespace_name:
                self.namespace_aliases[namespace_name] = target_name
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_constant_buffer(self):
        if self.current_token[0] != "CONSTANT":
            return False
        next_tok = self.peek(1)
        next_next = self.peek(2)
        return next_tok[0] == "IDENTIFIER" and next_next[0] == "LBRACE"

    def is_template_declaration_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "template"
        )

    def is_union_declaration_start(self):
        idx = self.skip_gnu_extension_prefix_tokens_at(self.pos)
        return (
            idx < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "union")
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] in {"IDENTIFIER", "LBRACE"}
        )

    def is_qualified_aggregate_declaration_start(self):
        idx = self.pos
        saw_qualifier = False
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            saw_qualifier = True
            idx += 1
        if not saw_qualifier or idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] in {"STRUCT", "CLASS"}:
            idx += 1
            idx = self.skip_alignas_specifier_tokens_at(idx)
            if idx >= len(self.tokens) or not self.is_name_token_at(idx):
                return False
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_alignas_specifier_tokens_at(idx)
            return idx < len(self.tokens) and self.tokens[idx][0] in {
                "LBRACE",
                "COLON",
            }
        return (
            self.tokens[idx] == ("IDENTIFIER", "union")
            and idx + 2 < len(self.tokens)
            and self.is_name_token_at(idx + 1)
            and self.tokens[idx + 2][0] == "LBRACE"
        )

    def parse_qualified_aggregate_declaration(self):
        qualifiers = []
        while self.is_type_qualifier_start():
            qualifiers.extend(self.parse_type_qualifier())

        if self.current_token[0] == "STRUCT":
            aggregate = self.parse_struct()
        elif self.current_token[0] == "CLASS":
            aggregate = self.parse_class()
        elif self.is_union_declaration_start():
            aggregate = self.parse_union()
        else:
            raise SyntaxError(
                f"Expected aggregate declaration, got {self.current_token[0]}"
            )

        for declaration in getattr(aggregate, "trailing_declarations", []) or []:
            var_node = (
                declaration.left
                if isinstance(declaration, AssignmentNode)
                and isinstance(declaration.left, VariableNode)
                else declaration
            )
            if isinstance(var_node, VariableNode):
                var_node.qualifiers = list(qualifiers) + list(
                    getattr(var_node, "qualifiers", []) or []
                )
        return aggregate

    def parse_template_declaration(self):
        template_parameters = self.parse_template_prefix()
        template_type_names = {
            name
            for kind, name in template_parameters
            if kind in {"typename", "class"} and name
        }
        added_type_names = template_type_names - self.known_types
        self.known_types.update(added_type_names)

        try:
            if self.current_token[0] in {"STRUCT", "CLASS"}:
                struct = (
                    self.parse_struct()
                    if self.current_token[0] == "STRUCT"
                    else self.parse_class()
                )
                if struct is not None:
                    struct.generics = [
                        name for _kind, name in template_parameters if name
                    ]
                    struct.template_parameters = template_parameters
                return struct

            if not self.is_function_definition():
                variable_template_name = self.template_variable_declaration_name_at(
                    self.pos
                )
                if variable_template_name:
                    self.known_variable_templates.add(variable_template_name)
                self.skip_template_declaration()
                return None

            function = self.parse_function()
            if function is not None:
                self.known_function_templates.add(function.name)
                function.generics = [
                    name for _kind, name in template_parameters if name
                ]
                function.template_parameters = template_parameters
            return function
        finally:
            self.known_types.difference_update(added_type_names)

    def parse_template_prefix(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] != "LESS_THAN":
            return []

        parameters = []
        parameter_tokens = []
        depth = 1
        grouped_depth = 0
        self.eat("LESS_THAN")

        while depth > 0 and self.current_token[0] != "EOF":
            token = self.current_token
            if token[0] in {"LPAREN", "LBRACKET", "LBRACE"}:
                grouped_depth += 1
                parameter_tokens.append(token)
                self.eat(token[0])
            elif token[0] in {"RPAREN", "RBRACKET", "RBRACE"} and grouped_depth > 0:
                grouped_depth -= 1
                parameter_tokens.append(token)
                self.eat(token[0])
            elif grouped_depth == 0 and token[0] == "LESS_THAN":
                depth += 1
                parameter_tokens.append(token)
                self.eat("LESS_THAN")
            elif grouped_depth == 0 and token[0] == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parameter_tokens.append(token)
                self.eat("GREATER_THAN")
            elif grouped_depth == 0 and token[0] == "SHIFT_RIGHT" and depth >= 2:
                closers_to_keep = min(2, depth - 1)
                parameter_tokens.extend(
                    ("GREATER_THAN", ">") for _ in range(closers_to_keep)
                )
                depth -= 2
                self.eat("SHIFT_RIGHT")
                if depth == 0:
                    break
            elif grouped_depth == 0 and token[0] == "COMMA" and depth == 1:
                parsed = self.parse_template_parameter_tokens(parameter_tokens)
                if parsed is not None:
                    parameters.append(parsed)
                parameter_tokens = []
                self.eat("COMMA")
            else:
                parameter_tokens.append(token)
                self.eat(token[0])

        parsed = self.parse_template_parameter_tokens(parameter_tokens)
        if parsed is not None:
            parameters.append(parsed)
        return parameters

    def parse_template_parameter_tokens(self, tokens):
        for idx, (token_type, value) in enumerate(tokens):
            if value == "typename" or token_type == "CLASS":
                name_idx = idx + 1
                is_variadic = False
                if name_idx < len(tokens) and tokens[name_idx][0] == "ELLIPSIS":
                    is_variadic = True
                    name_idx += 1
                if name_idx < len(tokens) and tokens[name_idx][0] == "IDENTIFIER":
                    kind = f"{value}..." if is_variadic else value
                    return (kind, tokens[name_idx][1])
        if len(tokens) >= 2 and tokens[-1][0] == "IDENTIFIER":
            return ("value", tokens[-1][1])
        return None

    def template_variable_declaration_name_at(self, idx):
        idx = self.skip_leading_attribute_tokens_at(idx)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return None

        idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return None

        name = self.tokens[idx][1]
        if idx + 1 < len(self.tokens) and self.tokens[idx + 1][0] == "LPAREN":
            return None
        return name

    def skip_template_declaration(self):
        paren_depth = 0
        brace_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN":
                paren_depth = max(0, paren_depth - 1)
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE":
                if brace_depth > 0:
                    brace_depth -= 1
                    self.eat("RBRACE")
                    if brace_depth == 0 and paren_depth == 0:
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                        return
                    continue
                self.eat("RBRACE")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                return
            elif token_type == "SEMICOLON" and paren_depth == 0 and brace_depth == 0:
                self.eat("SEMICOLON")
                return
            self.eat(token_type)

    def is_operator_function_definition(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        if idx >= len(self.tokens) or self.tokens[idx][0] not in TYPE_TOKENS:
            return False

        idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        operator_end = self.skip_operator_name_at(idx)
        return (
            operator_end > idx
            and operator_end < len(self.tokens)
            and self.tokens[operator_end][0] == "LPAREN"
        )

    def skip_operator_name_at(self, idx):
        if idx + 1 >= len(self.tokens):
            return idx
        token_type, token_value = self.tokens[idx]
        if token_type != "IDENTIFIER" or token_value != "operator":
            return idx

        operator_token = self.tokens[idx + 1][0]
        if operator_token not in OPERATOR_OVERLOAD_TOKENS:
            return idx
        if operator_token == "LBRACKET":
            return (
                idx + 3
                if (
                    idx + 2 < len(self.tokens) and self.tokens[idx + 2][0] == "RBRACKET"
                )
                else idx
            )
        if operator_token == "LPAREN":
            return (
                idx + 3
                if (idx + 2 < len(self.tokens) and self.tokens[idx + 2][0] == "RPAREN")
                else idx
            )
        return idx + 2

    def skip_operator_function_definition(self):
        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_block()

    def is_decltype_template_instantiation_declaration(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1

        if not self.is_decltype_start_at(idx):
            return False
        idx = self.skip_decltype_type_at(idx)

        while idx < len(self.tokens):
            if self.is_qualifier_token_at(idx):
                idx += 1
                continue
            if self.tokens[idx][0] in {"MULTIPLY", "BITWISE_AND"}:
                idx += 1
                continue
            break

        idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return False
        idx += 1

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LESS_THAN":
            return False
        idx = self.skip_template_argument_list_at(idx)
        return idx < len(self.tokens) and self.tokens[idx][0] == "SEMICOLON"

    def skip_decltype_template_instantiation_declaration(self):
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_decltype_start_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "decltype")
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def skip_decltype_type_at(self, idx):
        if not self.is_decltype_start_at(idx):
            return idx
        idx += 2
        depth = 1
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            idx += 1
        return idx

    def is_function_definition(self):
        idx = self.skip_leading_attribute_tokens_at(self.pos)
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
        if idx >= len(self.tokens):
            return False

        tok_type = self.tokens[idx][0]
        if tok_type in STAGE_TOKENS:
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)
            while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
                idx += 1
                idx = self.skip_leading_attribute_tokens_at(idx)
            if idx >= len(self.tokens):
                return False
            tok_type = self.tokens[idx][0]

        if tok_type == "IDENTIFIER" and self.tokens[idx][1] in SIGNED_TYPE_PREFIXES:
            idx = self.skip_signed_type_prefix_at(idx)
            type_name = "int"
        elif (
            tok_type in {"STRUCT", "ENUM"}
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] == "IDENTIFIER"
        ):
            type_name = f"{self.tokens[idx][1]} {self.tokens[idx + 1][1]}"
            idx += 2
        elif (
            tok_type == "IDENTIFIER"
            and self.tokens[idx][1] == "typename"
            and idx + 1 < len(self.tokens)
            and (
                self.tokens[idx + 1][0] in {"METAL", "SCOPE"}
                or (
                    self.tokens[idx + 1][0] == "IDENTIFIER"
                    and idx + 2 < len(self.tokens)
                    and self.tokens[idx + 2][0] == "SCOPE"
                )
            )
        ):
            idx += 1
            type_name = self.tokens[idx][1]
            idx = self.skip_scoped_type_name_at(idx)
        elif tok_type not in TYPE_TOKENS:
            return False
        else:
            type_name = self.tokens[idx][1]
            idx += 1
        idx = self.skip_scoped_type_suffix_at(idx)

        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_scoped_type_suffix_at(idx)

        while idx < len(self.tokens) and self.tokens[idx][0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            idx += 1

        idx = self.skip_leading_attribute_tokens_at(idx)

        if (
            idx < len(self.tokens)
            and self.tokens[idx][0] in STAGE_TOKENS
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] != "LPAREN"
        ):
            idx += 1
            idx = self.skip_leading_attribute_tokens_at(idx)

        if idx >= len(self.tokens) or not self.is_name_token_at(idx):
            return False
        idx += 1
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens) or not self.is_name_token_at(idx):
                return False
            idx += 1

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            if (
                idx < len(self.tokens)
                and self.tokens[idx][0] == "LESS_THAN"
                and self.template_argument_list_at_followed_by(idx, {"LPAREN"})
            ):
                idx = self.skip_template_argument_list_at(idx)
            else:
                return False

        if idx >= len(self.tokens) or self.tokens[idx][0] != "LPAREN":
            return False
        if type_name == "sampler" and self.parenthesized_list_ends_with_semicolon(idx):
            return False
        return True

    def template_argument_list_at_followed_by(self, idx, follow_token_types):
        if idx >= len(self.tokens) or self.tokens[idx][0] != "LESS_THAN":
            return False
        end = self.skip_template_argument_list_at(idx)
        return (
            end > idx
            and end < len(self.tokens)
            and self.tokens[end][0] in follow_token_types
        )

    def skip_template_argument_list_at(self, idx):
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "SHIFT_RIGHT" and depth >= 2:
                depth -= 2
                if depth == 0:
                    return idx + 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif depth > 0 and token_type in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return idx
            idx += 1
        return idx

    def skip_scoped_type_suffix_at(self, idx):
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
        return idx

    def skip_type_reference_suffix_at(self, idx):
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if (
                idx < len(self.tokens)
                and self.tokens[idx] == ("IDENTIFIER", "template")
                and idx + 1 < len(self.tokens)
                and self.tokens[idx + 1][0] in SCOPED_IDENTIFIER_PART_TOKENS
            ):
                idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
        return idx

    def parenthesized_list_ends_with_semicolon(self, idx):
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] == "SEMICOLON"
                    )
            idx += 1
        return False

    def is_qualifier_token_at(self, idx):
        if idx >= len(self.tokens):
            return False
        tok_type, value = self.tokens[idx]
        if tok_type in QUALIFIER_TOKENS or (
            tok_type == "IDENTIFIER"
            and value in IDENTIFIER_TYPE_QUALIFIERS | RAYTRACING_TYPE_QUALIFIERS
        ):
            return True
        if tok_type == "IDENTIFIER" and value in IDENTIFIER_PREFIX_TYPE_QUALIFIERS:
            return self.prefix_type_qualifier_followed_by_type_at(idx)
        return tok_type == "IDENTIFIER" and value in TYPE_QUALIFIER_FUNCTIONS

    def parse_preprocessor_directive(self):
        text = self.current_token[1] or ""
        self.eat("PREPROCESSOR")
        stripped = text.lstrip("#").strip()
        if stripped:
            parts = stripped.split(None, 1)
            directive = f"#{parts[0]}"
            content = parts[1] if len(parts) > 1 else ""
            return PreprocessorNode(directive, content)

        directive = text
        content = ""
        if self.current_token[0] == "LESS_THAN":
            self.eat("LESS_THAN")
            parts = []
            while self.current_token[0] != "GREATER_THAN":
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("GREATER_THAN")
            content = "<" + "".join(parts) + ">"
        elif self.current_token[0] == "STRING":
            content = self.current_token[1]
            self.eat("STRING")
        elif self.current_token[0] not in ["EOF", "PREPROCESSOR"]:
            content = str(self.current_token[1])
            self.eat(self.current_token[0])
        return PreprocessorNode(directive, content)

    def parse_using_statement(self):
        self.eat("USING")
        if self.current_token[0] == "NAMESPACE":
            self.eat("NAMESPACE")
            if self.current_token[0] in {"IDENTIFIER", "METAL"}:
                self.parse_scoped_identifier()
            self.eat("SEMICOLON")
            return None
        if self.is_using_declaration_start():
            self.parse_using_declaration()
            return None
        alias_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("EQUALS")
        if self.is_union_alias_start():
            return self.parse_using_union_alias(alias_name)
        alias_type, _qualifiers = self.parse_type_specifier()
        if self.current_token[0] == "LPAREN":
            self.parse_parenthesized_parameter_tokens()
            self.eat("SEMICOLON")
            self.known_types.add(alias_name)
            alias = TypeAliasNode(alias_type, alias_name)
            alias.is_function_type = True
            return alias
        self.eat("SEMICOLON")
        self.known_types.add(alias_name)
        return TypeAliasNode(alias_type, alias_name)

    def is_using_declaration_start(self):
        return self.current_token[0] in {"IDENTIFIER", "METAL", "SCOPE"} and not (
            self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "EQUALS"
        )

    def parse_using_declaration(self):
        self.parse_scoped_identifier()
        self.eat("SEMICOLON")

    def is_union_alias_start(self):
        return self.current_token == ("IDENTIFIER", "union")

    def parse_using_union_alias(self, alias_name):
        self.eat("IDENTIFIER")
        if self.current_token[0] != "LBRACE":
            raise SyntaxError("Expected anonymous union body in using alias")
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")
        self.eat("SEMICOLON")

        self.known_types.add(alias_name)
        union_node = StructNode(alias_name, members)
        union_node.aggregate_kind = "union"
        union_node.using_alias = True
        return union_node

    def parse_enum(self):
        name, is_scoped, underlying_type = self.parse_enum_header()
        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        enum = EnumNode(name, members)
        enum.is_scoped = is_scoped
        enum.underlying_type = underlying_type
        return enum

    def parse_enum_header(self):
        self.eat("ENUM")
        is_scoped = False
        if self.current_token[0] == "CLASS":
            is_scoped = True
            self.eat("CLASS")
        name = None
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.known_types.add(name)
        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type, _qualifiers = self.parse_type_specifier()
        return name, is_scoped, underlying_type

    def parse_enum_members(self):
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
        return members

    def parse_typedef(self):
        self.eat("TYPEDEF")
        if self.current_token[0] == "STRUCT":
            return self.parse_typedef_struct()
        if self.is_union_alias_start():
            return self.parse_typedef_union()
        if self.current_token[0] == "ENUM":
            return self.parse_typedef_enum()
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "decltype"
        ):
            alias_type = self.parse_decltype_type()
        else:
            alias_type, _qualifiers = self.parse_type_specifier()
        if self.current_token[0] == "LPAREN":
            alias_name = self.parse_function_typedef_declarator(alias_type)
            self.eat("SEMICOLON")
            self.known_types.add(alias_name)
            return TypeAliasNode(alias_type, alias_name)
        alias_name, _array, _type_suffix, _grouped_suffix = self.parse_declarator()
        if self.current_token[0] == "LPAREN":
            self.parse_parenthesized_parameter_tokens()
            self.eat("SEMICOLON")
            self.known_types.add(alias_name)
            alias = TypeAliasNode(alias_type, alias_name)
            alias.is_function_type = True
            return alias
        self.eat("SEMICOLON")
        self.known_types.add(alias_name)
        return TypeAliasNode(alias_type, alias_name)

    def parse_typedef_enum(self):
        tag_name, is_scoped, underlying_type = self.parse_enum_header()

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_enum_members()
            self.eat("RBRACE")
            (
                alias_name,
                _array_sizes,
                _type_suffix,
                _grouped_suffix,
            ) = self.parse_declarator()
            self.eat("SEMICOLON")
            enum_name = alias_name or tag_name
            if not enum_name:
                raise SyntaxError("Expected typedef enum name")
            self.known_types.add(enum_name)
            enum = EnumNode(enum_name, members)
            enum.typedef_tag = tag_name
            enum.is_scoped = is_scoped
            enum.underlying_type = underlying_type
            return enum

        if not tag_name:
            raise SyntaxError("Expected typedef enum body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.known_types.add(alias_name)
        return TypeAliasNode(f"enum {tag_name}", alias_name)

    def parse_typedef_union(self):
        self.eat("IDENTIFIER")
        tag_name = None
        if self.is_current_name_token():
            tag_name = self.current_token[1]
            self.eat(self.current_token[0])
            self.known_types.add(tag_name)

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_struct_members()
            self.eat("RBRACE")
            (
                alias_name,
                _array_sizes,
                _type_suffix,
                _grouped_suffix,
            ) = self.parse_declarator()
            self.eat("SEMICOLON")
            union_name = alias_name or tag_name
            if not union_name:
                raise SyntaxError("Expected typedef union name")
            self.known_types.add(union_name)
            union_node = StructNode(union_name, members)
            union_node.typedef_tag = tag_name
            union_node.aggregate_kind = "union"
            return union_node

        if not tag_name:
            raise SyntaxError("Expected typedef union body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.known_types.add(alias_name)
        return TypeAliasNode(f"union {tag_name}", alias_name)

    def parse_function_typedef_declarator(self, return_type):
        self.eat("LPAREN")
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            self.eat(self.current_token[0])
        alias_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("RPAREN")
        if self.current_token[0] == "LPAREN":
            self.parse_parenthesized_parameter_tokens()
        return alias_name

    def parse_parenthesized_parameter_tokens(self):
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise SyntaxError("Unterminated function typedef parameter list")

    def parse_decltype_type(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        parts = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            parts.append(token_value)
            self.eat(token_type)
        if depth != 0:
            raise SyntaxError("Unterminated decltype typedef")
        return f"decltype({self.format_generic_type_tokens(parts)})"

    def parse_typedef_struct(self):
        self.eat("STRUCT")
        struct_attributes = self.parse_attributes()
        alignas_specs = self.parse_alignas_specifiers()
        self.skip_aggregate_alignment_macros()
        tag_name = None
        if self.is_current_name_token():
            tag_name = self.current_token[1]
            self.eat(self.current_token[0])
            self.known_types.add(tag_name)

        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            members = self.parse_struct_members()
            self.eat("RBRACE")
            struct_attributes.extend(self.parse_attributes())
            (
                alias_name,
                _array_sizes,
                _type_suffix,
                _grouped_suffix,
            ) = self.parse_declarator()
            self.eat("SEMICOLON")
            struct_name = alias_name or tag_name
            if not struct_name:
                raise SyntaxError("Expected typedef struct name")
            self.known_types.add(struct_name)
            struct_node = StructNode(struct_name, members)
            struct_node.typedef_tag = tag_name
            struct_node.attributes = struct_attributes
            struct_node.alignas = alignas_specs
            return struct_node

        if not tag_name:
            raise SyntaxError("Expected typedef struct body or tag name")
        alias_name, _array_sizes, _type_suffix, _grouped_suffix = (
            self.parse_declarator()
        )
        self.eat("SEMICOLON")
        if alias_name == tag_name:
            return None
        self.known_types.add(alias_name)
        return TypeAliasNode(f"struct {tag_name}", alias_name)

    def parse_static_assert(self):
        self.eat("STATIC_ASSERT")
        self.eat("LPAREN")
        condition = self.parse_expression()
        message = None
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] == "STRING":
                message = self.current_token[1]
                self.eat("STRING")
            else:
                message = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return StaticAssertNode(condition, message)

    def parse_pragma_statement(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)
        if depth != 0:
            raise SyntaxError("Unterminated _Pragma statement")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_global_variable(self, pre_alignas=None):
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
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "LPAREN":
            args = self.parse_parenthesized_arguments()
            node = AssignmentNode(var_node, VectorConstructorNode(vtype, args))
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
            value = self.parse_initializer_list()
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

        self.eat("SEMICOLON")
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
                args = self.parse_parenthesized_arguments()
                nodes.append(
                    AssignmentNode(var_node, VectorConstructorNode(decl_type, args))
                )
            elif self.current_token[0] == "LBRACE":
                nodes.append(AssignmentNode(var_node, self.parse_initializer_list()))
            else:
                nodes.append(var_node)

        self.eat("SEMICOLON")
        return nodes

    def parse_type_specifier(self, attributes=None):
        qualifiers = []
        while (
            self.is_type_qualifier_start()
            or self.current_token[0] == "ATTRIBUTE"
            or self.is_gnu_attribute_start()
        ):
            if self.is_type_qualifier_start():
                qualifiers.extend(self.parse_type_qualifier())
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
            base_type = self.parse_decltype_type()
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
        elif self.current_token[0] not in TYPE_TOKENS:
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")
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
                base_type = {
                    "char": "uchar",
                    "short": "ushort",
                    "int": "uint",
                    "long": "ulong",
                }.get(base_type, base_type)
        else:
            base_type = self.current_token[1]
            self.eat(self.current_token[0])

        if self.current_token[0] == "LESS_THAN":
            inner = self.parse_template_argument_suffix()
            base_type = f"{base_type}<{self.format_generic_type_tokens(inner)}>"

        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                raise SyntaxError(
                    f"Expected identifier after '::', got {self.current_token[0]}"
                )
            base_type += f"::{self.current_token[1]}"
            self.eat(self.current_token[0])

        pointer_suffix = ""
        while self.is_post_type_qualifier_start() or self.current_token[0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            if self.is_post_type_qualifier_start():
                qualifiers.extend(self.parse_type_qualifier())
                continue
            pointer_suffix += "*" if self.current_token[0] == "MULTIPLY" else "&"
            self.eat(self.current_token[0])

        if self.current_token[0] == "ELLIPSIS":
            pointer_suffix += "..."
            self.eat("ELLIPSIS")

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
            return MACRO_QUALIFIER_ALIASES.get(qualifier, (qualifier,))

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
            raise SyntaxError(error_message)
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
        for token in tokens:
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

    def parse_declarator(self):
        name = ""
        type_suffix = ""
        grouped_suffix = False
        if self.is_parenthesized_declarator_start():
            name, type_suffix = self.parse_parenthesized_declarator()
            grouped_suffix = True
        else:
            while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
                type_suffix += self.declarator_pointer_token_suffix(
                    self.current_token[0]
                )
                self.eat(self.current_token[0])
        if not name and self.is_current_name_token():
            name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.template_argument_list_followed_by_call(
                follow_token_types={"SCOPE"}
            ):
                template_args = self.parse_template_argument_suffix()
                name += f"<{self.format_generic_type_tokens(template_args)}>"
            while self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                    raise SyntaxError(
                        f"Expected identifier after '::', got {self.current_token[0]}"
                    )
                name += f"::{self.current_token[1]}"
                self.eat(self.current_token[0])
        return name, self.parse_declarator_array_sizes(), type_suffix, grouped_suffix

    def declarator_pointer_token_suffix(self, token_type):
        return "*" if token_type == "MULTIPLY" else "&"

    def is_parenthesized_declarator_start(self):
        if self.current_token[0] != "LPAREN":
            return False
        idx = self.pos + 1
        saw_pointer_or_reference = False
        while idx < len(self.tokens) and self.tokens[idx][0] in {
            "MULTIPLY",
            "BITWISE_AND",
        }:
            saw_pointer_or_reference = True
            idx += 1
        return (
            saw_pointer_or_reference
            and idx + 1 < len(self.tokens)
            and self.is_name_token_at(idx)
            and self.tokens[idx + 1][0] == "RPAREN"
        )

    def parse_parenthesized_declarator(self):
        self.eat("LPAREN")
        type_suffix = ""
        while self.current_token[0] in {"MULTIPLY", "BITWISE_AND"}:
            type_suffix += self.declarator_pointer_token_suffix(self.current_token[0])
            self.eat(self.current_token[0])
        if not self.is_current_name_token():
            raise SyntaxError(f"Expected declarator name, got {self.current_token[0]}")
        name = self.current_token[1]
        self.eat(self.current_token[0])
        self.eat("RPAREN")
        return name, type_suffix

    def apply_declarator_type_suffix(self, vtype, type_suffix):
        return f"{vtype}{type_suffix}" if type_suffix else vtype

    def base_type_for_remaining_declarators(self, vtype):
        while vtype.endswith(("*", "&")):
            vtype = vtype[:-1]
        return vtype

    def apply_declarator_metadata(self, var_node, type_suffix, grouped_suffix):
        if not type_suffix:
            return
        var_node.declarator_type_suffix = type_suffix
        var_node.declarator_type_suffix_grouped = grouped_suffix

    def is_name_token_at(self, idx):
        return (
            self.tokens[idx][0] == "IDENTIFIER"
            or self.tokens[idx][0] in STAGE_TOKENS
            or self.tokens[idx][0] in KEYWORD_IDENTIFIER_TOKENS
            or self.tokens[idx][0] in TYPE_IDENTIFIER_TOKENS
            or self.tokens[idx][0] == "COMPUTE"
        )

    def is_current_name_token(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[0] in STAGE_TOKENS
            or self.current_token[0] in KEYWORD_IDENTIFIER_TOKENS
            or self.current_token[0] in TYPE_IDENTIFIER_TOKENS
            or self.current_token[0] == "COMPUTE"
        )

    def is_current_member_name_token(self):
        return self.is_current_name_token() or self.current_token[0] in {
            "READ",
            "WRITE",
            "READ_WRITE",
        }

    def parse_member_name(self, context):
        if not self.is_current_member_name_token():
            raise SyntaxError(
                f"Expected identifier after {context}, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat(self.current_token[0])
        return member

    def parse_declarator_array_sizes(self):
        array_sizes = []
        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            array_sizes.append(size)

        return array_sizes

    def parse_struct(self, pre_alignas=None):
        alignas_specs = (
            list(pre_alignas)
            if pre_alignas is not None
            else self.parse_alignas_specifiers()
        )
        self.eat("STRUCT")
        struct_attributes = self.parse_attributes()
        alignas_specs.extend(self.parse_alignas_specifiers())
        self.skip_aggregate_alignment_macros()
        if not self.is_current_name_token():
            raise SyntaxError(f"Expected identifier, got {self.current_token[0]}")
        name = self.current_token[1]
        self.eat(self.current_token[0])
        self.known_types.add(name)
        if self.current_token[0] == "LESS_THAN":
            template_args = self.parse_template_argument_suffix()
            name = f"{name}<{self.format_generic_type_tokens(template_args)}>"
            self.known_types.add(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        base_types = []
        if self.current_token[0] == "COLON":
            base_types = self.parse_struct_base_clause()
        self.eat("LBRACE")

        members = self.parse_struct_members()

        struct_node = StructNode(name, members)
        struct_node.attributes = struct_attributes
        struct_node.alignas = alignas_specs
        struct_node.base_types = base_types
        self.eat("RBRACE")
        struct_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        return struct_node

    def parse_class(self, pre_alignas=None):
        alignas_specs = (
            list(pre_alignas)
            if pre_alignas is not None
            else self.parse_alignas_specifiers()
        )
        self.eat("CLASS")
        class_attributes = self.parse_attributes()
        alignas_specs.extend(self.parse_alignas_specifiers())
        self.skip_aggregate_alignment_macros()
        if not self.is_current_name_token():
            raise SyntaxError(f"Expected identifier, got {self.current_token[0]}")
        name = self.current_token[1]
        self.eat(self.current_token[0])
        self.known_types.add(name)
        if self.current_token[0] == "LESS_THAN":
            template_args = self.parse_template_argument_suffix()
            name = f"{name}<{self.format_generic_type_tokens(template_args)}>"
            self.known_types.add(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        base_types = []
        if self.current_token[0] == "COLON":
            base_types = self.parse_struct_base_clause()
        self.eat("LBRACE")

        members = self.parse_struct_members()

        class_node = StructNode(name, members)
        class_node.attributes = class_attributes
        class_node.alignas = alignas_specs
        class_node.base_types = base_types
        class_node.aggregate_kind = "class"
        self.eat("RBRACE")
        class_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        return class_node

    def parse_struct_base_clause(self):
        self.eat("COLON")
        bases = []
        parts = []
        depth = 0
        paired_closers = {
            "LESS_THAN": "GREATER_THAN",
            "LPAREN": "RPAREN",
            "LBRACKET": "RBRACKET",
        }
        open_tokens = set(paired_closers)
        close_tokens = set(paired_closers.values())

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type == "LBRACE" or (depth == 0 and token_type == "COMMA"):
                if parts:
                    bases.append(self.format_generic_type_tokens(parts))
                    parts = []
                if token_type == "COMMA":
                    self.eat("COMMA")
                    continue
                break
            if token_type in open_tokens:
                depth += 1
            elif token_type == "SHIFT_RIGHT" and depth > 0:
                depth = max(0, depth - 2)
            elif token_type in close_tokens and depth > 0:
                depth -= 1
            parts.append(token_value)
            self.eat(token_type)

        if parts:
            bases.append(self.format_generic_type_tokens(parts))
        return bases

    def parse_union(self):
        self.skip_gnu_extension_prefix_tokens()
        self.eat("IDENTIFIER")
        name = self.current_token[1] if self.is_current_name_token() else None
        if name:
            self.eat(self.current_token[0])
            self.known_types.add(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        if not name:
            raise SyntaxError("Expected union name")
        self.eat("LBRACE")

        members = self.parse_struct_members()

        union_node = StructNode(name, members)
        union_node.aggregate_kind = "union"
        self.eat("RBRACE")
        union_node.trailing_declarations = self.parse_trailing_aggregate_declarations(
            name
        )
        return union_node

    def parse_trailing_aggregate_declarations(self, aggregate_type):
        declarations = []
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return declarations

        while True:
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            decl_type = self.apply_declarator_type_suffix(aggregate_type, type_suffix)
            attributes = self.parse_attributes()
            var_node = VariableNode(decl_type, name, attributes=attributes)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                declarations.append(AssignmentNode(var_node, self.parse_expression()))
            elif self.current_token[0] == "LPAREN":
                args = self.parse_parenthesized_arguments()
                declarations.append(
                    AssignmentNode(var_node, VectorConstructorNode(decl_type, args))
                )
            elif self.current_token[0] == "LBRACE":
                declarations.append(
                    AssignmentNode(var_node, self.parse_initializer_list())
                )
            else:
                declarations.append(var_node)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            self.eat("SEMICOLON")
            return declarations

    def parse_struct_members(self):
        members = []
        while self.current_token[0] != "RBRACE":
            if self.is_access_specifier_label():
                self.parse_access_specifier_label()
                continue
            if self.is_nested_aggregate_declaration_start():
                self.skip_nested_aggregate_declaration()
                continue
            if self.current_token == ("IDENTIFIER", "friend"):
                self.skip_struct_method()
                continue
            if (
                self.current_token[0] == "IDENTIFIER"
                and self.current_token[1] in STRUCT_METHOD_PREFIXES
            ):
                self.skip_struct_method()
                continue
            if self.current_token[0] == "STATIC_ASSERT":
                members.append(self.parse_static_assert())
                continue
            if self.current_token[0] == "USING":
                self.parse_using_statement()
                continue
            if self.current_token[0] == "TYPEDEF":
                self.parse_typedef()
                continue
            if self.is_template_declaration_start():
                self.skip_template_declaration()
                continue
            member_alignas = self.parse_alignas_specifiers()
            if self.is_nested_aggregate_declaration_start():
                self.skip_nested_aggregate_declaration()
                continue
            if self.is_struct_conversion_operator_start():
                self.skip_struct_method()
                continue
            if self.is_struct_destructor_start():
                self.skip_struct_method()
                continue
            vtype, qualifiers = self.parse_type_specifier()
            if self.current_token[0] == "OPERATOR":
                self.skip_struct_method()
                continue
            var_name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            if var_name == "operator":
                self.skip_struct_method()
                continue
            if self.current_token[0] == "LPAREN":
                self.skip_struct_method()
                continue
            attributes = self.parse_attributes()
            array_sizes.extend(self.parse_declarator_array_sizes())
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                default_value = self.parse_initializer_list()
            var_node = VariableNode(
                member_type, var_name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = member_alignas
            var_node.default_value = default_value
            members.append(var_node)
            if self.current_token[0] == "COMMA":
                members.extend(
                    self.parse_remaining_struct_member_declarations(
                        self.base_type_for_remaining_declarators(vtype),
                        qualifiers,
                        member_alignas,
                    )
                )
                continue
            self.eat("SEMICOLON")
        return members

    def parse_remaining_struct_member_declarations(
        self, vtype, qualifiers, alignas_specs
    ):
        members = []
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes = self.parse_attributes()
            array_sizes.extend(self.parse_declarator_array_sizes())
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                default_value = self.parse_initializer_list()

            var_node = VariableNode(
                member_type,
                name,
                qualifiers=list(qualifiers),
                attributes=attributes,
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = list(alignas_specs)
            var_node.default_value = default_value
            members.append(var_node)

        self.eat("SEMICOLON")
        return members

    def is_nested_aggregate_declaration_start(self):
        idx = self.skip_gnu_extension_prefix_tokens_at(self.pos)
        if idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"STRUCT", "CLASS"} and self.tokens[idx] != (
            "IDENTIFIER",
            "union",
        ):
            return False

        idx += 1
        idx = self.skip_alignas_specifier_tokens_at(idx)
        if idx < len(self.tokens) and self.is_name_token_at(idx):
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_template_argument_list_at(idx)
            idx = self.skip_alignas_specifier_tokens_at(idx)

        if idx >= len(self.tokens):
            return False
        return self.tokens[idx][0] in {"LBRACE", "COLON", "SEMICOLON"}

    def skip_gnu_extension_prefix_tokens_at(self, idx):
        while (
            idx < len(self.tokens)
            and self.tokens[idx][0] == "IDENTIFIER"
            and self.tokens[idx][1] in GNU_EXTENSION_PREFIXES
        ):
            idx += 1
        return idx

    def skip_gnu_extension_prefix_tokens(self):
        while (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in GNU_EXTENSION_PREFIXES
        ):
            self.eat("IDENTIFIER")

    def skip_aggregate_alignment_macros(self):
        while (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in AGGREGATE_ALIGNMENT_MACROS
        ):
            self.eat("IDENTIFIER")

    def skip_alignas_specifier_tokens_at(self, idx):
        while idx < len(self.tokens) and self.tokens[idx][0] == "ALIGNAS":
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN":
                idx = self.skip_balanced_tokens_at(idx, "LPAREN", "RPAREN")
        return idx

    def skip_nested_aggregate_declaration(self):
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0
        angle_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and angle_depth == 0
            ):
                self.eat("SEMICOLON")
                return

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LESS_THAN" and brace_depth == 0:
                angle_depth += 1
            elif token_type == "SHIFT_RIGHT" and brace_depth == 0 and angle_depth >= 2:
                angle_depth -= 2
            elif token_type == "GREATER_THAN" and brace_depth == 0 and angle_depth > 0:
                angle_depth -= 1

            self.eat(token_type)

        raise SyntaxError("Unterminated nested aggregate declaration")

    def is_struct_conversion_operator_start(self):
        idx = self.pos
        while idx < len(self.tokens) and self.is_qualifier_token_at(idx):
            idx += 1
        while idx < len(self.tokens) and self.tokens[idx] == ("IDENTIFIER", "explicit"):
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx] == ("IDENTIFIER", "operator")

    def is_struct_destructor_start(self):
        return (
            self.current_token[0] == "BITWISE_NOT"
            and self.pos + 2 < len(self.tokens)
            and self.is_name_token_at(self.pos + 1)
            and self.tokens[self.pos + 2][0] == "LPAREN"
        )

    def is_access_specifier_label(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in {"public", "private", "protected"}
            and self.peek(1)[0] == "COLON"
        )

    def parse_access_specifier_label(self):
        self.eat("IDENTIFIER")
        self.eat("COLON")

    def skip_struct_method(self):
        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_block()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

    def skip_balanced_block(self):
        self.eat("LBRACE")
        depth = 1
        while depth > 0 and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACE")
                    break
            self.eat(self.current_token[0])

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            member_alignas = self.parse_alignas_specifiers()
            vtype, qualifiers = self.parse_type_specifier()
            var_name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            member_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            self.eat("SEMICOLON")
            var_node = VariableNode(member_type, var_name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.alignas = member_alignas
            members.append(var_node)

        self.eat("RBRACE")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return ConstantBufferNode(name, members)

    def parse_function(self):
        attributes = self.parse_attributes()

        qualifier = None
        if self.current_token[0] in STAGE_TOKENS:
            qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        return_type, _return_qualifiers = self.parse_type_specifier(
            attributes=attributes
        )

        if self.current_token[0] in STAGE_TOKENS and self.peek(1)[0] != "LPAREN":
            if qualifier is None:
                qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        attributes.extend(self.parse_attributes())

        name = self.current_token[1]
        if not self.is_current_name_token():
            raise SyntaxError(f"Expected function name, got {self.current_token[0]}")
        self.eat(self.current_token[0])
        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if not self.is_current_name_token():
                raise SyntaxError(
                    f"Expected function name after '::', got {self.current_token[0]}"
                )
            name += f"::{self.current_token[1]}"
            self.eat(self.current_token[0])
        name = self.parse_function_operator_suffix(name)
        if self.template_argument_list_followed_by_call(follow_token_types={"LPAREN"}):
            template_args = self.parse_template_argument_suffix()
            name += f"<{self.format_generic_type_tokens(template_args)}>"

        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)
        self.parse_function_method_qualifiers()
        trailing_return_type = self.parse_optional_trailing_return_type(attributes)
        if trailing_return_type is not None:
            return_type = trailing_return_type
            attributes.extend(self.parse_attributes())
            self.parse_function_method_qualifiers()
        attribute_qualifier, attributes = self.extract_stage_attributes(attributes)
        if qualifier is None:
            qualifier = attribute_qualifier

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        if self.is_defaulted_or_deleted_function_declaration():
            self.skip_defaulted_or_deleted_function_declaration()
            return None
        self.pending_block_scope_names.append(
            {param.name for param in params if getattr(param, "name", None)}
        )
        body = self.parse_block()

        return FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=[qualifier] if qualifier else [],
            attributes=attributes,
            qualifier=qualifier,  # Also store as single qualifier for backward compatibility
        )

    def is_defaulted_or_deleted_function_declaration(self):
        return (
            self.current_token[0] == "EQUALS"
            and self.peek(1)[0] in {"DEFAULT", "IDENTIFIER"}
            and self.peek(1)[1] in {"default", "delete"}
            and self.peek(2)[0] == "SEMICOLON"
        )

    def skip_defaulted_or_deleted_function_declaration(self):
        self.eat("EQUALS")
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_function_method_qualifiers(self):
        while self.current_token[0] in {"CONST", "VOLATILE", "BITWISE_AND", "AND"}:
            self.eat(self.current_token[0])

        if self.current_token == ("IDENTIFIER", "noexcept"):
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.parse_balanced_token_text(
                    "LPAREN",
                    "RPAREN",
                    error_message="Unterminated noexcept qualifier",
                )

        while self.current_token == (
            "IDENTIFIER",
            "override",
        ) or self.current_token == (
            "IDENTIFIER",
            "final",
        ):
            self.eat("IDENTIFIER")

    def parse_function_operator_suffix(self, name):
        if (
            name.split("::")[-1] != "operator"
            or self.current_token[0] != "LPAREN"
            or self.peek(1)[0] != "RPAREN"
        ):
            return name
        self.eat("LPAREN")
        self.eat("RPAREN")
        return f"{name}()"

    def parse_optional_trailing_return_type(self, attributes):
        if self.current_token[0] != "ARROW":
            return None
        self.eat("ARROW")
        trailing_attributes = self.parse_attributes()
        return_type, _return_qualifiers = self.parse_type_specifier(
            attributes=trailing_attributes
        )
        attributes.extend(trailing_attributes)
        return return_type

    def extract_stage_attributes(self, attributes):
        qualifier = None
        remaining = []
        for attr in attributes:
            attr_name = getattr(attr, "name", None)
            if attr_name in STAGE_ATTRIBUTE_NAMES and not getattr(attr, "args", []):
                if qualifier is None:
                    qualifier = attr_name
                continue
            remaining.append(attr)
        return qualifier, remaining

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            if self.current_token[0] == "ELLIPSIS":
                self.eat("ELLIPSIS")
                params.append(VariableNode("...", ""))
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue
                if self.current_token[0] == "RPAREN":
                    break
                raise SyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )
            attributes = self.parse_attributes()
            vtype, qualifiers = self.parse_type_specifier(attributes=attributes)
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            if grouped_suffix and self.current_token[0] == "LPAREN":
                self.parse_function_pointer_parameter_suffix()
            param_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            default_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                default_value = self.parse_expression()
            param_attributes = self.parse_attributes()
            attributes.extend(param_attributes)

            var_node = VariableNode(
                param_type, name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            var_node.default_value = default_value
            params.append(var_node)

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] == "RPAREN":
                break
            else:
                raise SyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )
        return params

    def parse_function_pointer_parameter_suffix(self):
        self.parse_balanced_token_text(
            "LPAREN",
            "RPAREN",
            error_message="Unterminated function pointer parameter",
        )

    def is_gnu_attribute_start(self):
        return (
            self.current_token == ("IDENTIFIER", "__attribute__")
            and self.peek(1)[0] == "LPAREN"
        )

    def is_gnu_attribute_start_at(self, idx):
        return (
            idx + 1 < len(self.tokens)
            and self.tokens[idx] == ("IDENTIFIER", "__attribute__")
            and self.tokens[idx + 1][0] == "LPAREN"
        )

    def skip_balanced_tokens_at(self, idx, open_token, close_token):
        if idx >= len(self.tokens) or self.tokens[idx][0] != open_token:
            return idx
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                if depth == 0:
                    return idx + 1
            elif (
                open_token == "LESS_THAN"
                and close_token == "GREATER_THAN"
                and token_type == "SHIFT_RIGHT"
                and depth >= 2
            ):
                depth -= 2
                if depth == 0:
                    return idx + 1
            idx += 1
        return idx

    def skip_leading_attribute_tokens_at(self, idx):
        while idx < len(self.tokens):
            if self.tokens[idx][0] == "ATTRIBUTE":
                idx += 1
            elif self.is_gnu_attribute_start_at(idx):
                idx = self.skip_balanced_tokens_at(idx + 1, "LPAREN", "RPAREN")
            else:
                break
        return idx

    def parse_attributes(self):
        def split_top_level(text):
            parts = []
            buf = ""
            depth = 0
            for ch in text:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                if ch == "," and depth == 0:
                    if buf.strip():
                        parts.append(buf.strip())
                    buf = ""
                    continue
                buf += ch
            if buf.strip():
                parts.append(buf.strip())
            return parts

        def append_attributes(attr_content):
            for part in split_top_level(attr_content):
                name = part
                args = []
                if "(" in part and part.endswith(")"):
                    name, arg_str = part.split("(", 1)
                    arg_str = arg_str[:-1]  # remove trailing )
                    args = [arg.strip() for arg in split_top_level(arg_str)]
                    name = name.strip()
                attributes.append(AttributeNode(name.strip(), args))

        attributes = []
        while self.current_token[0] == "ATTRIBUTE" or self.is_gnu_attribute_start():
            if self.current_token[0] == "ATTRIBUTE":
                attr_content = self.current_token[1][2:-2].strip()  # Remove [[ and ]]
                append_attributes(attr_content)
                self.eat("ATTRIBUTE")
                continue

            self.eat("IDENTIFIER")
            attr_content = self.parse_balanced_token_text(
                "LPAREN", "RPAREN", "Unterminated GNU attribute"
            ).strip()
            if attr_content.startswith("(") and attr_content.endswith(")"):
                attr_content = attr_content[1:-1].strip()
            append_attributes(attr_content)
        return attributes

    def parse_block(self):
        statements = []
        initial_scope_names = (
            self.pending_block_scope_names.pop()
            if self.pending_block_scope_names
            else set()
        )
        self.local_variable_scopes.append(set(initial_scope_names))
        self.eat("LBRACE")
        try:
            while self.current_token[0] != "RBRACE":
                statement = self.parse_statement()
                if isinstance(statement, list):
                    statements.extend(statement)
                elif statement is not None:
                    statements.append(statement)
            self.eat("RBRACE")
            return statements
        finally:
            self.local_variable_scopes.pop()

    def parse_statement_body(self):
        if self.current_token[0] == "LBRACE":
            return self.parse_block()
        statement = self.parse_statement()
        if isinstance(statement, list):
            return statement
        return [] if statement is None else [statement]

    def is_declaration_start(self):
        return self.is_declaration_start_at(
            self.skip_leading_attribute_tokens_at(self.pos)
        )

    def is_declaration_start_at(self, idx):
        if idx >= len(self.tokens):
            return False
        token_type, token_value = self.tokens[idx]
        if self.is_scoped_call_expression_start_at(idx):
            return False
        if self.is_templated_call_expression_start_at(idx):
            return False
        if self.is_known_local_variable_name(token_value):
            return False
        if self.is_scoped_type_declaration_start_at(idx):
            return True
        if token_type == "ALIGNAS":
            return True
        if token_type == "STRUCT":
            return (
                idx + 2 < len(self.tokens)
                and self.tokens[idx + 1][0] == "IDENTIFIER"
                and self.tokens[idx + 2][0]
                in {
                    "IDENTIFIER",
                    "MULTIPLY",
                    "BITWISE_AND",
                    "LBRACKET",
                }
            )
        if self.is_qualifier_token_at(idx):
            return True
        if token_type in TYPE_TOKENS:
            if token_type == "IDENTIFIER":
                if self.is_decltype_start_at(idx):
                    return True
                if token_value in SIGNED_TYPE_PREFIXES:
                    return True
                if (
                    token_value
                    in IDENTIFIER_TYPE_QUALIFIERS | RAYTRACING_TYPE_QUALIFIERS
                ):
                    return True
                next_idx = idx + 1
                next_tok = (
                    self.tokens[next_idx][0] if next_idx < len(self.tokens) else "EOF"
                )
                if next_tok in [
                    "IDENTIFIER",
                    "SCOPE",
                    "LESS_THAN",
                    "MULTIPLY",
                    "BITWISE_AND",
                ]:
                    return True
                while next_idx < len(self.tokens) and self.is_qualifier_token_at(
                    next_idx
                ):
                    next_idx += 1
                if next_idx != idx + 1:
                    next_tok = (
                        self.tokens[next_idx][0]
                        if next_idx < len(self.tokens)
                        else "EOF"
                    )
                    return next_tok in {
                        "IDENTIFIER",
                        "MULTIPLY",
                        "BITWISE_AND",
                        "LBRACKET",
                    }
                return token_value in self.known_types
            return True
        return False

    def is_templated_call_expression_start_at(self, idx):
        if idx + 1 >= len(self.tokens) or self.tokens[idx][0] != "IDENTIFIER":
            return False
        if self.tokens[idx + 1][0] != "LESS_THAN":
            return False
        end = self.skip_template_argument_list_at(idx + 1)
        return end < len(self.tokens) and self.tokens[end][0] == "LPAREN"

    def is_scoped_type_declaration_start_at(self, idx):
        if idx >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"SCOPE", "METAL"} and not (
            self.tokens[idx][0] == "IDENTIFIER"
            and idx + 1 < len(self.tokens)
            and self.tokens[idx + 1][0] == "SCOPE"
        ):
            return False

        idx = self.skip_scoped_type_name_at(idx)
        if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
            idx = self.skip_template_argument_list_at(idx)
        while idx < len(self.tokens) and self.tokens[idx][0] in {
            "MULTIPLY",
            "BITWISE_AND",
        }:
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx][0] in {
            "IDENTIFIER",
            "LBRACKET",
        }

    def skip_scoped_type_name_at(self, idx):
        if idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
        if idx >= len(self.tokens):
            return idx
        if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
            return idx
        idx += 1
        while idx < len(self.tokens) and self.tokens[idx][0] == "SCOPE":
            idx += 1
            if idx >= len(self.tokens):
                return idx
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return idx
            idx += 1
        return idx

    def is_known_local_variable_name(self, name):
        return any(name in scope for scope in reversed(self.local_variable_scopes))

    def register_local_variable_name(self, name):
        if self.local_variable_scopes and name:
            self.local_variable_scopes[-1].add(name)

    def is_scoped_call_expression_start_at(self, idx):
        if idx + 2 >= len(self.tokens):
            return False
        if self.tokens[idx][0] not in {"IDENTIFIER", "METAL"}:
            return False
        if self.tokens[idx + 1][0] != "SCOPE":
            return False

        idx += 2
        while idx < len(self.tokens):
            if self.tokens[idx][0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                return False
            idx += 1
            if idx < len(self.tokens) and self.tokens[idx][0] == "LESS_THAN":
                idx = self.skip_balanced_tokens_at(idx, "LESS_THAN", "GREATER_THAN")
            if idx >= len(self.tokens) or self.tokens[idx][0] != "SCOPE":
                break
            idx += 1

        return idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN"

    def parse_statement(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None
        if self.is_for_loop_macro_prefix():
            self.eat("IDENTIFIER")
            return None
        if self.current_token[0] == "PREPROCESSOR":
            self.parse_preprocessor_directive()
            return None
        if self.current_token == ("IDENTIFIER", "_Pragma"):
            self.parse_pragma_statement()
            return None
        if self.current_token[0] == "USING":
            self.parse_using_statement()
            return None
        if self.current_token[0] == "TYPEDEF":
            self.parse_typedef()
            return None
        if self.is_nested_aggregate_declaration_start():
            self.skip_nested_aggregate_declaration()
            return None
        if self.current_token[0] == "LBRACE":
            return BlockNode(self.parse_block())
        if self.is_statement_expression_block_start():
            return self.parse_statement_expression_block()
        if self.current_token[0] == "ENUM":
            return self.parse_enum()
        if self.is_declaration_start():
            return self.parse_variable_declaration_or_assignment()
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
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif self.current_token[0] == "CONTINUE":
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif self.current_token[0] == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return DiscardNode()
        elif self.current_token[0] == "STATIC_ASSERT":
            return self.parse_static_assert()
        else:
            return self.parse_expression_statement()

    def is_statement_expression_block_start(self):
        return self.current_token[0] == "LPAREN" and self.peek(1)[0] == "LBRACE"

    def parse_statement_expression_block(self):
        self.eat("LPAREN")
        block = BlockNode(self.parse_block())
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return block

    def is_for_loop_macro_prefix(self):
        if self.current_token[0] != "IDENTIFIER" or self.peek(1)[0] != "FOR":
            return False
        name = self.current_token[1]
        return bool(name) and name == name.upper()

    def parse_variable_declaration_or_assignment(self):
        attributes = self.parse_attributes()
        alignas_specs = self.parse_alignas_specifiers()
        attributes.extend(self.parse_attributes())
        vtype, qualifiers = self.parse_type_specifier(attributes=attributes)
        base_vtype = self.base_type_for_remaining_declarators(vtype)
        name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
        vtype = self.apply_declarator_type_suffix(vtype, type_suffix)
        attributes.extend(self.parse_attributes())

        var_node = VariableNode(
            vtype, name, qualifiers=qualifiers, attributes=attributes
        )
        var_node.array_sizes = array_sizes
        self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
        var_node.alignas = alignas_specs
        self.register_local_variable_name(name)
        if "const" in qualifiers or "constexpr" in qualifiers:
            var_node.is_const = True

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return var_node
        if self.current_token[0] == "COMMA":
            return self.parse_remaining_variable_declarations(
                base_vtype, qualifiers, [var_node]
            )

        if self.current_token[0] in [
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
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            node = AssignmentNode(var_node, value, op)
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "LPAREN":
            args = self.parse_parenthesized_arguments()
            node = AssignmentNode(var_node, VectorConstructorNode(vtype, args))
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat("SEMICOLON")
            return node

        if self.current_token[0] == "LBRACE":
            node = AssignmentNode(var_node, self.parse_initializer_list())
            if self.current_token[0] == "COMMA":
                return self.parse_remaining_variable_declarations(
                    base_vtype, qualifiers, [node]
                )
            self.eat("SEMICOLON")
            return node

        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_remaining_variable_declarations(self, vtype, qualifiers, nodes):
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            decl_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            attributes = self.parse_attributes()
            var_node = VariableNode(
                decl_type, name, qualifiers=list(qualifiers), attributes=attributes
            )
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True

            if self.current_token[0] in [
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
            ]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                nodes.append(AssignmentNode(var_node, self.parse_expression(), op))
            elif self.current_token[0] == "LPAREN":
                args = self.parse_parenthesized_arguments()
                nodes.append(
                    AssignmentNode(var_node, VectorConstructorNode(vtype, args))
                )
            elif self.current_token[0] == "LBRACE":
                nodes.append(AssignmentNode(var_node, self.parse_initializer_list()))
            else:
                nodes.append(var_node)

        self.eat("SEMICOLON")
        return nodes

    def parse_if_statement(self):
        if_chain = []
        else_if_chain = []
        else_body = None
        while self.current_token[0] == "IF":
            self.eat("IF")
            self.parse_optional_if_constexpr()
            self.eat("LPAREN")
            condition = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
            self.parse_control_statement_attributes()
            body = self.parse_statement_body()
            if_chain.append((condition, body))
        while self.current_token[0] == "ELSE_IF":
            self.eat("ELSE_IF")
            self.parse_optional_if_constexpr()
            self.eat("LPAREN")
            condition = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
            self.parse_control_statement_attributes()
            body = self.parse_statement_body()
            else_if_chain.append((condition, body))

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement_body()

        return IfNode(
            if_chain=if_chain, else_if_chain=else_if_chain, else_body=else_body
        )

    def parse_optional_if_constexpr(self):
        if self.current_token[0] == "CONSTEXPR":
            self.eat("CONSTEXPR")
        elif self.current_token == ("IDENTIFIER", "IF_CONSTEXPR"):
            self.eat("IDENTIFIER")

    def parse_control_statement_attributes(self):
        self.parse_attributes()

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if self.current_token[0] != "SEMICOLON":
            init = self.parse_for_init()
            if self.current_token[0] == "COMMA":
                init = self.parse_for_tail_items(init, "SEMICOLON")
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression(allow_comma=True)
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression(allow_comma=True)
            if self.current_token[0] == "COMMA":
                update = self.parse_for_tail_items(update, "RPAREN")
        self.eat("RPAREN")

        self.parse_control_statement_attributes()
        body = self.parse_statement_body()

        return ForNode(init, condition, update, body)

    def parse_for_tail_items(self, first_item, terminator):
        items = [first_item]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if self.current_token[0] == terminator:
                break
            items.append(
                self.parse_for_init()
                if terminator == "SEMICOLON"
                else self.parse_expression()
            )
        return items

    def is_range_for_statement(self):
        idx = self.pos
        depth = 0
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if token_type == "RPAREN" and depth == 0:
                return False
            if token_type == "SEMICOLON" and depth == 0:
                return False
            if token_type == "COLON" and depth == 0:
                return True
            if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                depth += 1
            elif token_type in {"RPAREN", "RBRACKET", "RBRACE"} and depth > 0:
                depth -= 1
            idx += 1
        return False

    def parse_range_for_statement(self):
        vtype, _qualifiers = self.parse_type_specifier()
        name, _array_sizes, _type_suffix, _grouped_suffix = self.parse_declarator()
        self.eat("COLON")
        iterable = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")

        body = self.parse_statement_body()

        return RangeForNode(vtype, name, iterable, body)

    def parse_for_init(self):
        if self.is_declaration_start():
            vtype, qualifiers = self.parse_type_specifier()
            name, array_sizes, type_suffix, grouped_suffix = self.parse_declarator()
            init_type = self.apply_declarator_type_suffix(vtype, type_suffix)
            var_node = VariableNode(init_type, name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
            self.apply_declarator_metadata(var_node, type_suffix, grouped_suffix)
            self.register_local_variable_name(name)
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                init_value = self.parse_expression()
                return AssignmentNode(var_node, init_value)
            return var_node
        return self.parse_expression(allow_comma=True)

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode(None)
        value = self.parse_expression(allow_comma=True)
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")
        self.parse_control_statement_attributes()
        body = self.parse_statement_body()
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_statement_body()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression(allow_comma=True)
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_expression_statement(self):
        expr = self.parse_expression(allow_comma=True)
        self.eat("SEMICOLON")
        return expr

    def parse_expression(self, allow_comma=False):
        if allow_comma:
            return self.parse_comma_expression()
        return self.parse_assignment()

    def parse_comma_expression(self):
        left = self.parse_assignment()
        while self.current_token[0] == "COMMA":
            op = self.current_token[1]
            self.eat("COMMA")
            right = self.parse_assignment()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment(self):
        left = self.parse_conditional()
        if self.current_token[0] in [
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
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment()
            return AssignmentNode(left, right, op)
        return left

    def parse_conditional(self):
        left = self.parse_logical_or()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(left, true_expr, false_expr)
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
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_shift(self):
        left = self.parse_additive()
        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
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
        if self.current_token[0] in UNARY_KEYWORDS:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "LPAREN" and self.is_type_in_parens():
                self.eat("LPAREN")
                type_name, _quals = self.parse_type_specifier()
                self.eat("RPAREN")
                return FunctionCallNode(op, [type_name])
            operand = self.parse_unary()
            return FunctionCallNode(op, [operand])
        if self.current_token[0] == "LPAREN" and self.is_type_in_parens():
            self.eat("LPAREN")
            type_name, _quals = self.parse_type_specifier()
            self.eat("RPAREN")
            operand = self.parse_unary()
            return CastNode(type_name, operand)
        if self.current_token[0] in [
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
            "INCREMENT",
            "DECREMENT",
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_postfix()

    def is_type_in_parens(self):
        if self.current_token[0] != "LPAREN":
            return False
        idx = self.pos + 1
        saw_qualifier = False
        while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
            saw_qualifier = True
            idx += 1
        if idx >= len(self.tokens):
            return False
        tok_type = self.tokens[idx][0]
        typename_prefix = tok_type == "IDENTIFIER" and self.tokens[idx][1] == "typename"
        if typename_prefix:
            idx += 1
            if idx >= len(self.tokens):
                return False
            tok_type = self.tokens[idx][0]
        if tok_type not in TYPE_TOKENS:
            return False
        if tok_type == "IDENTIFIER":
            name = self.tokens[idx][1]
            next_type = self.tokens[idx + 1][0] if idx + 1 < len(self.tokens) else "EOF"
            if name in SIGNED_TYPE_PREFIXES and next_type in TYPE_TOKENS:
                idx += 2
            else:
                if (
                    not typename_prefix
                    and name not in self.known_types
                    and next_type not in {"SCOPE", "LESS_THAN"}
                    and not (
                        saw_qualifier
                        and next_type in {"MULTIPLY", "BITWISE_AND", "RPAREN"}
                    )
                    and not (
                        next_type == "RPAREN" and self.is_cast_operand_start_at(idx + 2)
                    )
                ):
                    return False
                idx += 1
        else:
            idx += 1
        idx = self.skip_type_reference_suffix_at(idx)
        while idx < len(self.tokens) and self.tokens[idx][0] in [
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx][0] == "RPAREN"

    def is_cast_operand_start_at(self, idx):
        if idx >= len(self.tokens):
            return False
        token_type = self.tokens[idx][0]
        return token_type in CONSTRUCTOR_TYPE_TOKENS or token_type in {
            "IDENTIFIER",
            "METAL",
            "SCOPE",
            "NUMBER",
            "TRUE",
            "FALSE",
            "CHAR_LITERAL",
            "STRING",
            "LPAREN",
            "LBRACE",
        }

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.is_static_cast_template_call(node):
                template_args = self.parse_template_argument_suffix()
                target_type = self.format_generic_type_tokens(template_args)
                self.eat("LPAREN")
                expression = self.parse_expression()
                self.eat("RPAREN")
                node = CastNode(target_type, expression)
                continue
            if self.is_as_type_template_call(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{''.join(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.is_template_identifier_suffix(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{self.format_generic_type_tokens(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.is_template_variable_suffix(node):
                template_args = self.parse_template_argument_suffix()
                suffix = f"<{self.format_generic_type_tokens(template_args)}>"
                if isinstance(node, VariableNode):
                    node.name += suffix
                else:
                    node.member += suffix
                continue
            if self.current_token[0] == "LPAREN":
                node = self.parse_call(node)
                continue
            if self.current_token[0] == "ELLIPSIS":
                self.eat("ELLIPSIS")
                node = UnaryOpNode("post...", node)
                continue
            if self.current_token[0] == "LBRACE" and isinstance(node, VariableNode):
                initializer = self.parse_initializer_list()
                node = FunctionCallNode(node.name, [initializer])
                node.is_braced_constructor = True
                continue
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                self.skip_template_disambiguator()
                if self.is_conversion_operator_call_start():
                    node = self.parse_conversion_operator_call(node)
                    continue
                member = self.parse_member_name("dot")
                node = MemberAccessNode(node, member)
                continue
            if self.current_token[0] == "ARROW":
                self.eat("ARROW")
                self.skip_template_disambiguator()
                if self.is_conversion_operator_call_start():
                    node = self.parse_conversion_operator_call(node)
                    continue
                member = self.parse_member_name("arrow")
                node = MemberAccessNode(node, member, True)
                continue
            if self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                member = self.parse_member_name("scope")
                if isinstance(node, VariableNode):
                    node.name += f"::{member}"
                elif isinstance(node, MemberAccessNode):
                    node.member += f"::{member}"
                else:
                    node = MemberAccessNode(node, member)
                continue
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = None
                if self.current_token[0] != "RBRACKET":
                    index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue
            if self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                node = PostfixOpNode(node, op)
                continue
            break
        return node

    def is_conversion_operator_call_start(self):
        if self.current_token != ("IDENTIFIER", "operator"):
            return False
        next_type = self.peek(1)[0]
        return next_type not in OPERATOR_OVERLOAD_TOKENS

    def parse_conversion_operator_call(self, node):
        self.eat("IDENTIFIER")
        target_type, _qualifiers = self.parse_type_specifier()
        self.eat("LPAREN")
        self.eat("RPAREN")
        return CastNode(target_type, node)

    def skip_template_disambiguator(self):
        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "template"
            and self.peek(1)[0] in {"IDENTIFIER", "READ", "WRITE", "READ_WRITE"}
        ):
            self.eat("IDENTIFIER")

    def is_static_cast_template_call(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if isinstance(node, VariableNode):
            callee_name = node.name
        elif isinstance(node, MemberAccessNode):
            callee_name = node.member
        else:
            return False
        if callee_name.split("::")[-1] not in {
            "static_cast",
            "reinterpret_cast",
            "const_cast",
        }:
            return False
        return self.template_argument_list_followed_by_call(
            require_type_like_argument=False
        )

    def is_as_type_template_call(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if isinstance(node, VariableNode):
            callee_name = node.name
        elif isinstance(node, MemberAccessNode):
            callee_name = node.member
        else:
            return False
        if callee_name.split("::")[-1] != "as_type":
            return False
        return self.template_argument_list_followed_by_call()

    def is_template_identifier_suffix(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if not isinstance(node, (VariableNode, MemberAccessNode)):
            return False
        name = node.name if isinstance(node, VariableNode) else node.member
        if name in self.known_function_templates:
            return self.template_argument_list_followed_by_call(
                follow_token_types=TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
                require_type_like_argument=False,
            )
        return self.template_argument_list_followed_by_call(
            follow_token_types=TEMPLATE_IDENTIFIER_SUFFIX_FOLLOW_TOKENS,
            require_type_like_argument=True,
        )

    def is_template_variable_suffix(self, node):
        if self.current_token[0] != "LESS_THAN":
            return False
        if not isinstance(node, (VariableNode, MemberAccessNode)):
            return False
        name = node.name if isinstance(node, VariableNode) else node.member
        if not (
            self.is_variable_template_name(name)
            or name in self.known_variable_templates
        ):
            return False
        return self.template_argument_list_followed_by_call(
            follow_token_types=TEMPLATE_VARIABLE_SUFFIX_FOLLOW_TOKENS,
            require_type_like_argument=True,
        )

    def is_variable_template_name(self, name):
        if not isinstance(name, str):
            return False
        return name.split("::")[-1].endswith("_v")

    def template_argument_list_followed_by_call(
        self, follow_token_types=None, require_type_like_argument=False
    ):
        if self.current_token[0] != "LESS_THAN":
            return False
        follow_token_types = follow_token_types or {"LPAREN"}
        idx = self.pos
        depth = 0
        saw_type_like = False
        while idx < len(self.tokens):
            token_type = self.tokens[idx][0]
            if depth > 0 and token_type in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
                return False
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "SHIFT_RIGHT" and depth >= 2:
                depth -= 2
                if depth == 0:
                    if require_type_like_argument and not saw_type_like:
                        return False
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] in follow_token_types
                    )
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    if require_type_like_argument and not saw_type_like:
                        return False
                    return (
                        idx + 1 < len(self.tokens)
                        and self.tokens[idx + 1][0] in follow_token_types
                    )
            elif depth > 0 and self.is_template_type_argument_token(idx):
                saw_type_like = True
            idx += 1
        return False

    def is_template_type_argument_token(self, idx):
        token_type, token_value = self.tokens[idx]
        if token_type in TYPE_TOKENS - {"IDENTIFIER", "TYPEDEF"}:
            return True
        if token_type == "IDENTIFIER":
            next_type = self.tokens[idx + 1][0] if idx + 1 < len(self.tokens) else "EOF"
            return token_value in self.known_types or next_type in {
                "SCOPE",
                "LESS_THAN",
                "COMMA",
                "GREATER_THAN",
            }
        return False

    def parse_template_argument_suffix(self):
        self.eat("LESS_THAN")
        depth = 1
        grouped_depth = 0
        parts = []
        while depth > 0 and self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if grouped_depth == 0 and token_type == "LESS_THAN":
                depth += 1
                parts.append(token_value)
                self.eat("LESS_THAN")
            elif grouped_depth == 0 and token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(token_value)
                self.eat("GREATER_THAN")
            elif grouped_depth == 0 and token_type == "SHIFT_RIGHT" and depth >= 2:
                parts.extend(">" for _ in range(min(2, depth - 1)))
                depth -= 2
                self.eat("SHIFT_RIGHT")
                if depth == 0:
                    break
            else:
                if token_type in {"LPAREN", "LBRACKET", "LBRACE"}:
                    grouped_depth += 1
                elif (
                    token_type in {"RPAREN", "RBRACKET", "RBRACE"} and grouped_depth > 0
                ):
                    grouped_depth -= 1
                parts.append(token_value)
                self.eat(token_type)
        return parts

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
            value = self.current_token[1]
            self.eat("STRING")
            while self.current_token[0] == "STRING":
                value += self.current_token[1]
                self.eat("STRING")
            return value
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
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
                name = self.parse_scoped_identifier()
                return VariableNode("", name)
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "LPAREN":
                return self.parse_vector_constructor(type_name)
            if self.current_token[0] == "LBRACE":
                initializer = self.parse_initializer_list()
                node = FunctionCallNode(type_name, [initializer])
                node.is_braced_constructor = True
                return node
            raise SyntaxError(f"Unexpected type in expression: {type_name}")
        if self.current_token[0] in {"METAL", "SCOPE"} or self.is_current_name_token():
            name = self.parse_scoped_identifier()
            return VariableNode("", name)
        raise SyntaxError(f"Unexpected token in expression: {self.current_token[0]}")

    def parse_lambda_expression(self):
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
            raise SyntaxError("Expected lambda body")

        self.pending_block_scope_names.append(
            {param.name for param in params if getattr(param, "name", None)}
        )
        body = self.parse_block()
        return LambdaNode(
            capture,
            params,
            body,
            return_type,
            self.format_generic_type_tokens(specifier_tokens).split(),
        )

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
            raise SyntaxError("Unterminated lambda capture list")
        return self.format_generic_type_tokens(parts)

    def parse_initializer_list(self):
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
        return InitializerListNode(elements)

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
            raise SyntaxError("Expected designator expression")
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
                raise SyntaxError(f"Expected identifier, got {self.current_token[0]}")
            self.eat(self.current_token[0])
        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if (
                self.current_token == ("IDENTIFIER", "template")
                and self.peek(1)[0] in TYPE_TOKENS
            ):
                self.eat("IDENTIFIER")
            if self.current_token[0] not in SCOPED_IDENTIFIER_PART_TOKENS:
                raise SyntaxError(
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
        args = self.parse_parenthesized_arguments()
        return VectorConstructorNode(type_name, args)

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
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")

        if isinstance(callee, MemberAccessNode):
            if callee.member == "sample":
                return self.build_texture_sample(callee.object, args)
            return MethodCallNode(callee.object, callee.member, args)
        if isinstance(callee, VariableNode):
            if callee.name == "discard_fragment" and not args:
                return DiscardNode()
            return FunctionCallNode(callee.name, args)
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

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default = None

        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "CASE":
                cases.append(self.parse_case_statement())
            elif self.current_token[0] == "DEFAULT":
                self.eat("DEFAULT")
                self.eat("COLON")
                default_statements = []

                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
                    if self.current_token[0] == "BREAK":
                        self.eat("BREAK")
                        self.eat("SEMICOLON")
                        break
                    else:
                        statement = self.parse_statement()
                        if isinstance(statement, list):
                            default_statements.extend(statement)
                        elif statement is not None:
                            default_statements.append(statement)

                default = default_statements
            else:
                raise SyntaxError(
                    f"Unexpected token in switch statement: {self.current_token[0]}"
                )

        self.eat("RBRACE")
        return SwitchNode(expression, cases, default)

    def parse_case_statement(self):
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []

        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            if self.current_token[0] == "BREAK":
                self.eat("BREAK")
                self.eat("SEMICOLON")
                break
            else:
                statement = self.parse_statement()
                if isinstance(statement, list):
                    statements.extend(statement)
                elif statement is not None:
                    statements.append(statement)

        return CaseNode(value, statements)
