from .MetalLexer import *
from .MetalAst import *

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

UNARY_KEYWORDS = {"SIZEOF", "ALIGNOF"}


class MetalParser:
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
        }

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
            elif self.current_token[0] == "USING":
                alias = self.parse_using_statement()
                if alias is not None:
                    typedefs.append(alias)
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "ALIGNAS":
                alignas_specs = self.parse_alignas_specifiers()
                if self.current_token[0] == "STRUCT":
                    structs.append(self.parse_struct(alignas_specs))
                else:
                    # Treat as global variable with alignas
                    global_variables.append(
                        self.parse_global_variable(pre_alignas=alignas_specs)
                    )
            elif self.current_token[0] == "ENUM":
                enums.append(self.parse_enum())
            elif self.current_token[0] == "TYPEDEF":
                typedefs.append(self.parse_typedef())
            elif self.current_token[0] == "STATIC_ASSERT":
                global_variables.append(self.parse_static_assert())
            elif self.current_token[0] == "CONSTANT":
                if self.is_constant_buffer():
                    constants.append(self.parse_constant_buffer())
                else:
                    global_variables.append(self.parse_global_variable())
            elif self.current_token[0] in STAGE_TOKENS or (
                self.current_token[0] in TYPE_TOKENS
                or self.current_token[0] in QUALIFIER_TOKENS
            ):
                if self.is_function_definition():
                    functions.append(self.parse_function())
                else:
                    global_variables.append(self.parse_global_variable())
            else:
                self.eat(self.current_token[0])  # Skip unknown tokens

        # Create ShaderNode with proper keyword arguments for compatibility
        return ShaderNode(
            includes=preprocessors,
            functions=functions,
            structs=structs,
            global_variables=global_variables,
            constant=constants,
            enums=enums,
            typedefs=typedefs,
        )

    def is_constant_buffer(self):
        if self.current_token[0] != "CONSTANT":
            return False
        next_tok = self.peek(1)
        next_next = self.peek(2)
        return next_tok[0] == "IDENTIFIER" and next_next[0] == "LBRACE"

    def is_function_definition(self):
        idx = self.pos
        # Skip attributes
        while idx < len(self.tokens) and self.tokens[idx][0] == "ATTRIBUTE":
            idx += 1
        # Skip qualifiers
        while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
            idx += 1
        if idx >= len(self.tokens):
            return False

        tok_type = self.tokens[idx][0]
        if tok_type in STAGE_TOKENS:
            idx += 1
            while idx < len(self.tokens) and self.tokens[idx][0] == "ATTRIBUTE":
                idx += 1
            while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
                idx += 1
            if idx >= len(self.tokens):
                return False
            tok_type = self.tokens[idx][0]

        if tok_type not in TYPE_TOKENS:
            return False
        idx += 1

        # Skip generic type parameters
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

        while idx < len(self.tokens) and self.tokens[idx][0] in ["MULTIPLY", "BITWISE_AND"]:
            idx += 1

        if idx >= len(self.tokens) or self.tokens[idx][0] != "IDENTIFIER":
            return False
        idx += 1

        return idx < len(self.tokens) and self.tokens[idx][0] == "LPAREN"

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
            # Best-effort capture of directive argument
            content = str(self.current_token[1])
            self.eat(self.current_token[0])
        return PreprocessorNode(directive, content)

    def parse_using_statement(self):
        self.eat("USING")
        if self.current_token[0] == "NAMESPACE":
            self.eat("NAMESPACE")
            self.eat("METAL")
            self.eat("SEMICOLON")
            return None
        # Handle using alias: using Alias = Type;
        alias_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("EQUALS")
        alias_type, _qualifiers = self.parse_type_specifier()
        self.eat("SEMICOLON")
        self.known_types.add(alias_name)
        return TypeAliasNode(alias_type, alias_name)

    def parse_enum(self):
        self.eat("ENUM")
        if self.current_token[0] == "CLASS":
            self.eat("CLASS")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.known_types.add(name)
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
        alias_type, _qualifiers = self.parse_type_specifier()
        alias_name, _array = self.parse_declarator()
        self.eat("SEMICOLON")
        self.known_types.add(alias_name)
        return TypeAliasNode(alias_type, alias_name)

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

    def parse_global_variable(self, pre_alignas=None):
        attributes = self.parse_attributes()
        alignas_specs = pre_alignas or self.parse_alignas_specifiers()
        vtype, qualifiers = self.parse_type_specifier()
        name, array_sizes = self.parse_declarator()
        var_attributes = self.parse_attributes()
        attributes.extend(var_attributes)

        var_node = VariableNode(
            vtype, name, qualifiers=qualifiers, attributes=attributes
        )
        var_node.array_sizes = array_sizes
        var_node.alignas = alignas_specs
        if "const" in qualifiers or "constexpr" in qualifiers:
            var_node.is_const = True

        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()
            self.eat("SEMICOLON")
            return AssignmentNode(var_node, value)

        self.eat("SEMICOLON")
        return var_node

    def parse_type_specifier(self):
        qualifiers = []
        while self.current_token[0] in QUALIFIER_TOKENS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.current_token[0] not in TYPE_TOKENS:
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")

        if self.current_token[0] == "METAL" or (
            self.current_token[0] == "IDENTIFIER" and self.peek(1)[0] == "SCOPE"
        ):
            base_type = self.parse_scoped_identifier()
        else:
            base_type = self.current_token[1]
            self.eat(self.current_token[0])

        # Handle generic types like texture2d<float, access::read_write>
        if self.current_token[0] == "LESS_THAN":
            depth = 0
            inner = []
            self.eat("LESS_THAN")
            depth += 1
            while depth > 0 and self.current_token[0] != "EOF":
                if self.current_token[0] == "LESS_THAN":
                    depth += 1
                    inner.append(self.current_token[1])
                    self.eat("LESS_THAN")
                elif self.current_token[0] == "GREATER_THAN":
                    depth -= 1
                    if depth == 0:
                        self.eat("GREATER_THAN")
                        break
                    inner.append(self.current_token[1])
                    self.eat("GREATER_THAN")
                else:
                    inner.append(self.current_token[1])
                    self.eat(self.current_token[0])
            base_type = f"{base_type}<{''.join(inner)}>"

        # Handle pointers/references
        pointer_suffix = ""
        while self.current_token[0] in ["MULTIPLY", "BITWISE_AND"]:
            pointer_suffix += "*" if self.current_token[0] == "MULTIPLY" else "&"
            self.eat(self.current_token[0])

        return base_type + pointer_suffix, qualifiers

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
        if self.current_token[0] in QUALIFIER_TOKENS:
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
        array_sizes = []
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            array_sizes.append(size)

        return name, array_sizes

    def parse_struct(self, pre_alignas=None):
        alignas_specs = pre_alignas or self.parse_alignas_specifiers()
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.known_types.add(name)
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            member_alignas = self.parse_alignas_specifiers()
            vtype, qualifiers = self.parse_type_specifier()
            var_name, array_sizes = self.parse_declarator()
            attributes = self.parse_attributes()
            self.eat("SEMICOLON")
            var_node = VariableNode(
                vtype, var_name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.array_sizes = array_sizes
            var_node.alignas = member_alignas
            members.append(var_node)

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        struct_node = StructNode(name, members)
        struct_node.alignas = alignas_specs
        return struct_node

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            member_alignas = self.parse_alignas_specifiers()
            vtype, qualifiers = self.parse_type_specifier()
            var_name, array_sizes = self.parse_declarator()
            self.eat("SEMICOLON")
            var_node = VariableNode(vtype, var_name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
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

        return_type, _return_qualifiers = self.parse_type_specifier()

        # Handle potential second qualifier after return type
        if self.current_token[0] in STAGE_TOKENS:
            if qualifier is None:
                qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        # Handle possible attribute after parameters
        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)

        body = self.parse_block()

        # Create FunctionNode with proper argument order matching common_ast
        return FunctionNode(
            return_type=return_type,
            name=name,
            params=params,
            body=body,
            qualifiers=[qualifier] if qualifier else [],
            attributes=attributes,
            qualifier=qualifier,  # Also store as single qualifier for backward compatibility
        )

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            attributes = self.parse_attributes()
            vtype, qualifiers = self.parse_type_specifier()
            name, array_sizes = self.parse_declarator()
            param_attributes = self.parse_attributes()
            attributes.extend(param_attributes)

            var_node = VariableNode(
                vtype, name, qualifiers=qualifiers, attributes=attributes
            )
            var_node.array_sizes = array_sizes
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

    def parse_attributes(self):
        attributes = []
        while self.current_token[0] == "ATTRIBUTE":
            attr_content = self.current_token[1][2:-2].strip()  # Remove [[ and ]]

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

            for part in split_top_level(attr_content):
                name = part
                args = []
                if "(" in part and part.endswith(")"):
                    name, arg_str = part.split("(", 1)
                    arg_str = arg_str[:-1]  # remove trailing )
                    args = [arg.strip() for arg in split_top_level(arg_str)]
                    name = name.strip()
                attributes.append(AttributeNode(name.strip(), args))
            self.eat("ATTRIBUTE")
        return attributes

    def parse_block(self):
        statements = []
        self.eat("LBRACE")
        while self.current_token[0] != "RBRACE":
            statements.append(self.parse_statement())
        self.eat("RBRACE")
        return statements

    def is_declaration_start(self):
        if self.current_token[0] == "ALIGNAS":
            return True
        if self.current_token[0] in QUALIFIER_TOKENS:
            return True
        if self.current_token[0] in TYPE_TOKENS:
            if self.current_token[0] == "IDENTIFIER":
                next_tok = self.peek(1)[0]
                if next_tok in ["IDENTIFIER", "SCOPE", "LESS_THAN", "MULTIPLY", "BITWISE_AND"]:
                    return True
                return self.current_token[1] in self.known_types
            return True
        return False

    def parse_statement(self):
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

    def parse_variable_declaration_or_assignment(self):
        alignas_specs = self.parse_alignas_specifiers()
        vtype, qualifiers = self.parse_type_specifier()
        name, array_sizes = self.parse_declarator()
        attributes = self.parse_attributes()

        var_node = VariableNode(
            vtype, name, qualifiers=qualifiers, attributes=attributes
        )
        var_node.array_sizes = array_sizes
        var_node.alignas = alignas_specs
        if "const" in qualifiers or "constexpr" in qualifiers:
            var_node.is_const = True

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return var_node

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
            self.eat("SEMICOLON")
            return AssignmentNode(var_node, value, op)

        # Fallback to expression statement if not a declaration
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_if_statement(self):
        if_chain = []
        else_if_chain = []
        else_body = None
        while self.current_token[0] == "IF":
            self.eat("IF")
            self.eat("LPAREN")
            condition = self.parse_expression()
            self.eat("RPAREN")
            body = self.parse_block()
            if_chain.append((condition, body))
        while self.current_token[0] == "ELSE_IF":
            self.eat("ELSE_IF")
            self.eat("LPAREN")
            condition = self.parse_expression()
            self.eat("RPAREN")
            body = self.parse_block()
            else_if_chain.append((condition, body))

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block()

        return IfNode(if_chain=if_chain, else_if_chain=else_if_chain, else_body=else_body)

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        init = None
        if self.current_token[0] != "SEMICOLON":
            init = self.parse_for_init()
        self.eat("SEMICOLON")

        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()
        self.eat("RPAREN")

        body = self.parse_block()

        return ForNode(init, condition, update, body)

    def parse_for_init(self):
        if self.is_declaration_start():
            vtype, qualifiers = self.parse_type_specifier()
            name, array_sizes = self.parse_declarator()
            var_node = VariableNode(vtype, name, qualifiers=qualifiers)
            var_node.array_sizes = array_sizes
            if "const" in qualifiers or "constexpr" in qualifiers:
                var_node.is_const = True
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                init_value = self.parse_expression()
                return AssignmentNode(var_node, init_value)
            return var_node
        return self.parse_expression()

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode(None)
        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_block()
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_block()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_expression(self):
        return self.parse_assignment()

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
        # Skip qualifiers
        while idx < len(self.tokens) and self.tokens[idx][0] in QUALIFIER_TOKENS:
            idx += 1
        if idx >= len(self.tokens):
            return False
        tok_type = self.tokens[idx][0]
        if tok_type not in TYPE_TOKENS:
            return False
        idx += 1
        # Skip generic type parameters
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
        # Pointer/reference
        while idx < len(self.tokens) and self.tokens[idx][0] in ["MULTIPLY", "BITWISE_AND"]:
            idx += 1
        return idx < len(self.tokens) and self.tokens[idx][0] == "RPAREN"

    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.current_token[0] == "LPAREN":
                node = self.parse_call(node)
                continue
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                if self.current_token[0] not in ["IDENTIFIER", "READ", "WRITE", "READ_WRITE"]:
                    raise SyntaxError(
                        f"Expected identifier after dot, got {self.current_token[0]}"
                    )
                member = self.current_token[1]
                self.eat(self.current_token[0])
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

    def parse_primary(self):
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        if self.current_token[0] in ["TRUE", "FALSE"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        if self.current_token[0] in ["VECTOR", "MATRIX", "SIMD_MATRIX", "PACKED_VECTOR", "SIMD_VECTOR", "FLOAT", "HALF", "DOUBLE", "INT", "UINT", "BOOL"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            if self.current_token[0] == "LPAREN":
                return self.parse_vector_constructor(type_name)
            raise SyntaxError(f"Unexpected type in expression: {type_name}")
        if self.current_token[0] in ["IDENTIFIER", "METAL"]:
            name = self.parse_scoped_identifier()
            return VariableNode("", name)
        raise SyntaxError(f"Unexpected token in expression: {self.current_token[0]}")

    def parse_scoped_identifier(self):
        parts = []
        if self.current_token[0] == "METAL":
            parts.append("metal")
            self.eat("METAL")
        else:
            parts.append(self.current_token[1])
            self.eat("IDENTIFIER")
        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            if self.current_token[0] not in TYPE_TOKENS and self.current_token[0] != "METAL":
                raise SyntaxError(
                    f"Expected identifier after '::', got {self.current_token[0]}"
                )
            if self.current_token[0] == "METAL":
                parts.append("metal")
                self.eat("METAL")
            else:
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
        return "::".join(parts)

    def parse_vector_constructor(self, type_name):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return VectorConstructorNode(type_name, args)

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
            return FunctionCallNode(callee.name, args)
        return CallNode(callee, args)

    def build_texture_sample(self, texture, args):
        sampler = args[0] if len(args) > 0 else None
        coords = args[1] if len(args) > 1 else None
        lod = args[2] if len(args) > 2 else None
        if lod is not None:
            return TextureSampleNode(texture, sampler, coords, lod)
        return TextureSampleNode(texture, sampler, coords)

    def parse_texture_sample_args(self, texture):
        self.eat("LPAREN")
        sampler = self.parse_expression()
        self.eat("COMMA")
        coordinates = self.parse_expression()

        # Support for optional LOD parameter
        lod = None
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            lod = self.parse_expression()

        self.eat("RPAREN")

        if lod is not None:
            return TextureSampleNode(texture, sampler, coordinates, lod)
        return TextureSampleNode(texture, sampler, coordinates)

    def parse_texture_sample(self):
        texture = self.parse_expression()
        self.eat("DOT")
        self.eat("IDENTIFIER")  # 'sample' method
        self.eat("LPAREN")
        sampler = self.parse_expression()
        self.eat("COMMA")
        coordinates = self.parse_expression()
        self.eat("RPAREN")
        return TextureSampleNode(texture, sampler, coordinates)

    def parse_switch_statement(self):
        """Parse a switch statement

        This method parses a switch statement in Metal shader code.

        Returns:
            SwitchNode: A node representing the switch statement
        """
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

                # Parse statements until we hit a case, default, or end of switch
                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
                    if self.current_token[0] == "BREAK":
                        self.eat("BREAK")
                        self.eat("SEMICOLON")
                        break
                    else:
                        default_statements.append(self.parse_statement())

                default = default_statements
            else:
                raise SyntaxError(
                    f"Unexpected token in switch statement: {self.current_token[0]}"
                )

        self.eat("RBRACE")
        return SwitchNode(expression, cases, default)

    def parse_case_statement(self):
        """Parse a case statement

        This method parses a case statement in Metal shader code.

        Returns:
            CaseNode: A node representing the case statement
        """
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []

        # Parse statements until we hit a case, default, break, or end of switch
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            if self.current_token[0] == "BREAK":
                self.eat("BREAK")
                self.eat("SEMICOLON")
                break
            else:
                statements.append(self.parse_statement())

        return CaseNode(value, statements)
