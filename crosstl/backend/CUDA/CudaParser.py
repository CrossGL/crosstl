"""CUDA Parser Implementation"""

from .CudaAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CastNode,
    ConstantMemoryNode,
    ContinueNode,
    CudaAsmNode,
    CudaAsmOperandNode,
    CudaBuiltinNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    InitializerListNode,
    KernelLaunchNode,
    KernelNode,
    MemberAccessNode,
    NewNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    SharedMemoryNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)


class CudaParser:
    """Parse CUDA tokens into the CUDA backend shader AST."""

    TYPE_QUALIFIER_TOKENS = {
        "CONST",
        "VOLATILE",
        "UNSIGNED",
        "SIGNED",
        "RESTRICT",
        "GRID_CONSTANT",
        "TYPENAME",
    }
    POSTFIX_TYPE_QUALIFIER_TOKENS = {"CONST", "RESTRICT"}
    TYPE_REFERENCE_TOKENS = {"BITWISE_AND", "LOGICAL_AND"}
    CPP_NAMED_CASTS = {"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"}
    ATOMIC_FUNCTION_TOKENS = {
        "ATOMICADD",
        "ATOMICSUB",
        "ATOMICMAX",
        "ATOMICMIN",
        "ATOMICEXCH",
        "ATOMICCAS",
        "ATOMICAND",
        "ATOMICOR",
        "ATOMICXOR",
        "ATOMICINC",
        "ATOMICDEC",
    }
    ATOMIC_FUNCTION_NAMES = {
        "atomicAdd",
        "atomicSub",
        "atomicMax",
        "atomicMin",
        "atomicExch",
        "atomicCAS",
        "atomicAnd",
        "atomicOr",
        "atomicXor",
        "atomicInc",
        "atomicDec",
    }
    FUNCTION_NAME_TOKENS = {"IDENTIFIER", *ATOMIC_FUNCTION_TOKENS}
    LAMBDA_SPECIFIER_TOKENS = {
        "DEVICE",
        "HOST",
        "MUTABLE",
        "INLINE",
        "STATIC",
        "FORCEINLINE",
        "NOINLINE",
    }
    LAMBDA_IDENTIFIER_SPECIFIERS = {
        "constexpr",
        "consteval",
        "noexcept",
        "__noexcept",
    }
    DECLARATION_QUALIFIER_TOKENS = {
        "CONST",
        "VOLATILE",
        "STATIC",
        "EXTERN",
        "CONSTEXPR",
        "SHARED",
        "CONSTANT",
        "DEVICE",
        "MANAGED",
    }
    FUNCTION_ATTRIBUTE_TOKENS = {"LAUNCH_BOUNDS", "CLUSTER_DIMS", "BLOCK_SIZE"}
    FUNCTION_SPECIFIER_TOKENS = {
        "GLOBAL",
        "DEVICE",
        "HOST",
        "INLINE",
        "STATIC",
        "EXTERN",
        "CONSTEXPR",
        "FORCEINLINE",
        "NOINLINE",
        *FUNCTION_ATTRIBUTE_TOKENS,
    }
    TYPE_TOKENS = {
        "VOID",
        "CHAR",
        "SHORT",
        "INT",
        "LONG",
        "FLOAT",
        "DOUBLE",
        "BOOL",
        "FLOAT2",
        "FLOAT3",
        "FLOAT4",
        "INT2",
        "INT3",
        "INT4",
        "DOUBLE2",
        "DOUBLE3",
        "DOUBLE4",
        "UINT2",
        "UINT3",
        "UINT4",
        "CHAR2",
        "CHAR3",
        "CHAR4",
        "UCHAR2",
        "UCHAR3",
        "UCHAR4",
        "SHORT2",
        "SHORT3",
        "SHORT4",
        "USHORT2",
        "USHORT3",
        "USHORT4",
        "LONG2",
        "LONG3",
        "LONG4",
        "ULONG2",
        "ULONG3",
        "ULONG4",
        "LONGLONG2",
        "LONGLONG3",
        "LONGLONG4",
        "ULONGLONG2",
        "ULONGLONG3",
        "ULONGLONG4",
        "SIZE_T",
        "TEXTURE",
        "SURFACE",
        "CUDAARRAY",
        "CUDAARRAYT",
        "IDENTIFIER",
    }
    CUDA_IDENTIFIER_TYPE_NAMES = {
        "cudaTextureObject_t",
        "cudaSurfaceObject_t",
        "half",
        "__half",
        "half2",
        "__half2",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
    }

    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None
        self.type_aliases = set()
        self.struct_names = self.collect_struct_names()
        self.user_function_names = self.collect_user_function_names()

    def collect_struct_names(self):
        names = set()
        for index, token in enumerate(self.tokens[:-1]):
            if token[0] == "STRUCT" and self.tokens[index + 1][0] == "IDENTIFIER":
                names.add(self.tokens[index + 1][1])
        return names

    def collect_user_function_names(self):
        names = set()
        depth = 0
        index = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACE":
                depth += 1
                index += 1
                continue
            if token_type == "RBRACE":
                depth = max(0, depth - 1)
                index += 1
                continue

            if depth == 0:
                name_index = self.function_name_index_at(index)
                if name_index is not None:
                    names.add(self.tokens[name_index][1])
                    index = name_index + 1
                    continue

            index += 1

        return names

    def function_name_index_at(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.FUNCTION_SPECIFIER_TOKENS
        ):
            if self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if (
                self.tokens[index][0] == "EXTERN"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "STRING"
            ):
                index += 2
                continue
            index += 1

        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        index = self.skip_type_at_index(index)
        if index is None:
            return None

        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
        ):
            index = self.skip_function_attribute_at_index(index)

        if (
            index + 1 < len(self.tokens)
            and self.is_function_name_token(self.tokens[index])
            and self.tokens[index + 1][0] == "LPAREN"
        ):
            return index

        return None

    def parse(self):
        includes = []
        functions = []
        structs = []
        global_variables = []
        kernels = []
        typedefs = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                includes.append(self.parse_preprocessor())
            elif self.current_token[0] == "TEMPLATE":
                self.parse_template_declaration()
            elif self.is_type_alias_start():
                aliases = self.parse_type_alias()
                if isinstance(aliases, list):
                    for alias in aliases:
                        if isinstance(alias, StructNode):
                            structs.append(alias)
                        elif alias is not None:
                            typedefs.append(alias)
                elif isinstance(aliases, StructNode):
                    structs.append(aliases)
                elif aliases is not None:
                    typedefs.append(aliases)
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                structs.append(self.parse_enum())
            elif self.is_linkage_block_start():
                for item in self.parse_linkage_block():
                    if isinstance(item, KernelNode):
                        kernels.append(item)
                    elif isinstance(item, FunctionNode):
                        functions.append(item)
                    elif isinstance(item, (StructNode, EnumNode)):
                        structs.append(item)
                    elif isinstance(item, TypeAliasNode):
                        typedefs.append(item)
                    else:
                        global_variables.append(item)
            elif self.peek_function():
                func = self.parse_function()
                if isinstance(func, KernelNode):
                    kernels.append(func)
                else:
                    functions.append(func)
            elif self.current_token[0] in ["GLOBAL", "DEVICE", "HOST"]:
                if self.current_token[0] == "DEVICE" and self.peek_variable():
                    global_variables.append(self.parse_global_variable())
                    continue

                func = self.parse_function()
                if isinstance(func, KernelNode):
                    kernels.append(func)
                else:
                    functions.append(func)
            elif (
                self.current_token[0] in ["CONSTANT", "SHARED"] or self.peek_variable()
            ):
                global_variables.append(self.parse_global_variable())
            else:
                self.eat(self.current_token[0])

        return ShaderNode(
            includes, functions, structs, global_variables, kernels, typedefs=typedefs
        )

    def peek_function(self):
        saved_index = self.current_index

        while (
            saved_index < len(self.tokens)
            and self.tokens[saved_index][0] in self.FUNCTION_SPECIFIER_TOKENS
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
                continue
            if (
                self.tokens[saved_index][0] == "EXTERN"
                and saved_index + 1 < len(self.tokens)
                and self.tokens[saved_index + 1][0] == "STRING"
            ):
                saved_index += 2
                continue
            saved_index += 1

        while (
            saved_index < len(self.tokens)
            and self.tokens[saved_index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            saved_index += 1

        saved_index = self.skip_type_at_index(saved_index)

        while (
            saved_index is not None
            and saved_index < len(self.tokens)
            and self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
        ):
            saved_index = self.skip_function_attribute_at_index(saved_index)

        if saved_index is not None:
            if (
                saved_index < len(self.tokens) - 1
                and self.is_function_name_token(self.tokens[saved_index])
                and self.tokens[saved_index + 1][0] == "LPAREN"
            ):
                return True

        return False

    def peek_variable(self):
        saved_index = self.current_index

        while (
            saved_index < len(self.tokens)
            and self.tokens[saved_index][0] in self.DECLARATION_QUALIFIER_TOKENS
        ):
            saved_index += 1

        saved_index = self.skip_type_at_index(saved_index)

        if saved_index is not None:
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] == "IDENTIFIER"
            ):
                saved_index += 1
                saved_index = self.skip_array_suffix_at_index(saved_index)

                if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
                    "SEMICOLON",
                    "ASSIGN",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                ]:
                    return True

        return False

    def skip_type_at_index(self, index):
        saw_integral_sign = False
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            if self.tokens[index][0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        if index >= len(self.tokens):
            return None

        if self.is_implicit_int_type_at_index(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.tokens[index][0] in self.TYPE_TOKENS:
            type_token = self.tokens[index][0]
            type_value = self.tokens[index][1]
            index += 1
            index = self.skip_long_long_suffix_at_index(index, type_token)
        else:
            return None

        has_qualified_suffix = False

        while (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] == "IDENTIFIER"
        ):
            has_qualified_suffix = True
            index += 2

        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            has_qualified_suffix = True
            index = self.skip_template_at_index(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_index(index)
        can_have_pointer_suffix = (
            type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or self.is_identifier_type_name(type_value)
        )
        while (
            can_have_pointer_suffix
            and index < len(self.tokens)
            and self.tokens[index][0] in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return self.skip_array_suffix_at_index(index)

    def is_implicit_int_type_at_index(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index][0]
        if token_type in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}:
            return True

        if token_type != "IDENTIFIER":
            return False

        next_type = self.tokens[index + 1][0] if index + 1 < len(self.tokens) else "EOF"
        return next_type in {
            "SEMICOLON",
            "ASSIGN",
            "LBRACKET",
            "LPAREN",
            "LBRACE",
            "COMMA",
        }

    def skip_long_long_suffix_at_index(self, index, type_token):
        if (
            type_token == "LONG"
            and index < len(self.tokens)
            and self.tokens[index][0] == "LONG"
        ):
            return index + 1
        return index

    def skip_postfix_type_qualifiers_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS
        ):
            index += 1
        return index

    def skip_function_attribute_at_index(self, index):
        index += 1
        if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
            return index

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return index
            index += 1

        return index

    def skip_launch_bounds_at_index(self, index):
        return self.skip_function_attribute_at_index(index)

    def skip_template_at_index(self, index):
        depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "SHIFT_RIGHT":
                depth -= 2
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
            elif token_type in {"SEMICOLON", "ASSIGN", "EOF"}:
                return None
            index += 1

        return None

    def skip_array_suffix_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "LBRACKET":
            index += 1
            while index < len(self.tokens) and self.tokens[index][0] != "RBRACKET":
                index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "RBRACKET":
                index += 1
            else:
                break

        return index

    def eat(self, expected_type):
        if self.current_token[0] == expected_type:
            token = self.current_token
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")
            return token
        else:
            raise SyntaxError(f"Expected {expected_type}, got {self.current_token[0]}")

    def is_function_name_token(self, token=None):
        token = token or self.current_token
        return bool(token and token[0] in self.FUNCTION_NAME_TOKENS)

    def is_identifier_type_name(self, name):
        return (
            name in self.type_aliases
            or name in self.struct_names
            or name in self.CUDA_IDENTIFIER_TYPE_NAMES
        )

    def consume_function_name(self):
        if not self.is_function_name_token():
            token_type = self.current_token[0] if self.current_token else "EOF"
            raise SyntaxError(f"Expected function name, got {token_type}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def parse_preprocessor(self):
        directive_token = self.eat("PREPROCESSOR")
        directive_text = directive_token[1].strip()

        if directive_text.startswith("#include"):
            content = directive_text[8:].strip()
            return PreprocessorNode("include", content)
        elif directive_text.startswith("#define"):
            content = directive_text[7:].strip()
            return PreprocessorNode("define", content)
        else:
            return PreprocessorNode("other", directive_text)

    def is_linkage_block_start(self):
        return (
            self.current_token[0] == "EXTERN"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "STRING"
            and self.tokens[self.current_index + 2][0] == "LBRACE"
        )

    def parse_linkage_block(self):
        self.eat("EXTERN")
        language = self.eat("STRING")[1].strip('"')
        self.eat("LBRACE")
        items = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                items.append(self.parse_preprocessor())
            elif self.current_token[0] == "TEMPLATE":
                self.parse_template_declaration()
            elif self.is_type_alias_start():
                aliases = self.parse_type_alias()
                if isinstance(aliases, list):
                    items.extend(aliases)
                elif aliases is not None:
                    items.append(aliases)
            elif self.current_token[0] == "STRUCT":
                items.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                items.append(self.parse_enum())
            elif self.peek_function():
                items.append(self.parse_function())
            elif self.current_token[0] in ["GLOBAL", "DEVICE", "HOST"]:
                if self.current_token[0] == "DEVICE" and self.peek_variable():
                    items.append(self.parse_global_variable())
                else:
                    items.append(self.parse_function())
            elif (
                self.current_token[0] in ["CONSTANT", "SHARED"] or self.peek_variable()
            ):
                items.append(self.parse_global_variable())
            else:
                self.eat(self.current_token[0])

        self.eat("RBRACE")
        self.apply_linkage_to_items(items, language)
        return items

    def apply_linkage_to_items(self, items, language):
        for item in items:
            if hasattr(item, "qualifiers"):
                qualifiers = getattr(item, "qualifiers", []) or []
                if "extern" not in qualifiers:
                    qualifiers.insert(0, "extern")
                item.qualifiers = qualifiers
                item.linkage = language

    def is_type_alias_start(self):
        return self.current_token[0] == "TYPEDEF" or self.is_identifier_value("using")

    def parse_type_alias(self):
        if self.current_token[0] == "TYPEDEF":
            return self.parse_typedef_alias()
        return self.parse_using_alias()

    def parse_typedef_alias(self):
        self.eat("TYPEDEF")
        if self.current_token[0] == "STRUCT":
            return self.parse_typedef_struct_alias()

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        aliases = [self.parse_type_alias_declarator(first_type, allow_prefix=False)]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            aliases.append(
                self.parse_type_alias_declarator(base_type, allow_prefix=True)
            )

        self.eat("SEMICOLON")
        return aliases

    def parse_type_alias_declarator(self, base_type, allow_prefix):
        alias_type = base_type
        if allow_prefix:
            alias_type = self.parse_declarator_prefix(alias_type)

        name = self.eat("IDENTIFIER")[1]
        alias_type += self.parse_array_suffix()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_typedef_struct_alias(self):
        self.eat("STRUCT")
        attributes = self.parse_alignment_attributes()

        tag_name = None
        if self.current_token[0] == "IDENTIFIER":
            tag_name = self.eat("IDENTIFIER")[1]

        if self.current_token[0] != "LBRACE":
            alias = self.parse_type_alias_declarator(
                f"struct {tag_name}", allow_prefix=True
            )
            self.eat("SEMICOLON")
            return alias

        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        alias_name = tag_name
        if self.current_token[0] == "IDENTIFIER":
            alias_name = self.eat("IDENTIFIER")[1]
        if not alias_name:
            raise SyntaxError("Expected typedef struct alias name")

        self.eat("SEMICOLON")
        self.type_aliases.add(alias_name)
        self.struct_names.add(alias_name)
        return StructNode(alias_name, members, attributes=attributes)

    def parse_alignment_attributes(self):
        attributes = []
        while self.current_token[0] == "ALIGNAS":
            name = self.eat("ALIGNAS")[1]
            self.eat("LPAREN")
            args = self.collect_balanced_tokens_until("RPAREN")
            self.eat("RPAREN")
            attributes.append(f"{name}({args})")
        return attributes

    def collect_balanced_tokens_until(self, terminator):
        tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                token_type == terminator
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break

            tokens.append(token_value)
            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1
            self.eat(token_type)

        return " ".join(tokens)

    def parse_using_alias(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "NAMESPACE":
            self.skip_until_semicolon()
            return None

        name = self.eat("IDENTIFIER")[1]
        if self.current_token[0] == "SCOPE":
            self.skip_until_semicolon()
            return None

        self.eat("ASSIGN")
        alias_type = self.parse_type()
        self.eat("SEMICOLON")
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def skip_until_semicolon(self):
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_struct(self):
        self.eat("STRUCT")
        attributes = self.parse_alignment_attributes()
        name = self.eat("IDENTIFIER")[1]
        if self.current_token[0] == "LESS_THAN":
            name += self.parse_template_suffix()
        self.struct_names.add(name)
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members, attributes=attributes)

    def parse_enum(self):
        self.eat("ENUM")
        is_scoped = False

        if self.current_token[0] in {"CLASS", "STRUCT"}:
            is_scoped = True
            self.eat(self.current_token[0])

        name = None
        if self.current_token[0] == "IDENTIFIER":
            name = self.eat("IDENTIFIER")[1]
            self.type_aliases.add(name)
            self.struct_names.add(name)

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        enum_node = EnumNode(name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        return enum_node

    def parse_enum_members(self):
        members = []

        while self.current_token[0] not in {"RBRACE", "EOF"}:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            member_name = self.current_token[1]
            self.eat(self.current_token[0])

            member_value = None
            if self.current_token[0] == "ASSIGN":
                self.eat("ASSIGN")
                member_value = self.parse_expression()

            members.append((member_name, member_value))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        return members

    def parse_struct_members(self):
        members = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                continue
            if self.is_struct_access_label():
                self.eat(self.current_token[0])
                self.eat("COLON")
                continue
            if self.is_struct_method_start():
                self.skip_struct_method()
                continue

            member_declarations = self.parse_variable_declaration_list()
            members.extend(member_declarations)
            self.eat("SEMICOLON")
        return members

    def is_struct_access_label(self):
        return (
            self.current_token[0] in {"PUBLIC", "PRIVATE", "PROTECTED"}
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "COLON"
        )

    def is_struct_method_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_struct_method_specifiers_at_index(index)

        if index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_value == "operator":
            return True
        if (
            token_type == "BITWISE_NOT"
            and index + 2 < len(self.tokens)
            and self.tokens[index + 1][0] == "IDENTIFIER"
            and self.tokens[index + 2][0] == "LPAREN"
        ):
            return True
        if (
            token_type == "IDENTIFIER"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1][0] == "LPAREN"
        ):
            return True

        name_index = self.skip_type_at_index(index)
        if name_index is None or name_index >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[name_index]
        if token_value == "operator":
            return True
        return (
            self.is_function_name_token(self.tokens[name_index])
            and name_index + 1 < len(self.tokens)
            and self.tokens[name_index + 1][0] == "LPAREN"
        )

    def skip_optional_template_declaration_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "TEMPLATE":
            return index

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index)
            if skipped is not None:
                return skipped
        return index

    def skip_struct_method_specifiers_at_index(self, index):
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if token_type in self.FUNCTION_SPECIFIER_TOKENS or token_type == "VIRTUAL":
                if (
                    token_type == "EXTERN"
                    and index + 1 < len(self.tokens)
                    and self.tokens[index + 1][0] == "STRING"
                ):
                    index += 2
                    continue
                index += 1
                continue
            break
        return index

    def skip_struct_method(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            if self.current_token[0] == "LESS_THAN":
                self.parse_template_suffix()

        while self.current_token[0] not in {"SEMICOLON", "LBRACE", "RBRACE", "EOF"}:
            self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_balanced_brace_block(self):
        self.eat("LBRACE")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACE")
                    break
            self.eat(token_type)

    def parse_function(self):
        qualifiers = []
        attributes = []
        linkage = None

        while self.current_token[0] in self.FUNCTION_SPECIFIER_TOKENS:
            if self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                attributes.append(self.parse_function_attribute())
                continue

            qualifier = self.current_token[1]
            qualifiers.append(qualifier)
            self.eat(self.current_token[0])
            if qualifier == "extern" and self.current_token[0] == "STRING":
                linkage = self.current_token[1].strip('"')
                self.eat("STRING")

        return_type = self.parse_type()
        while self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS:
            attributes.append(self.parse_function_attribute())

        name = self.consume_function_name()
        self.user_function_names.add(name)
        params = self.parse_parameters()
        body = None
        if self.current_token[0] == "LBRACE":
            body = self.parse_block()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        else:
            self.eat("LBRACE")

        if "__global__" in qualifiers:
            return KernelNode(
                return_type,
                name,
                params,
                body,
                attributes,
                qualifiers=qualifiers,
                linkage=linkage,
            )

        function = FunctionNode(return_type, name, params, body, qualifiers, attributes)
        function.linkage = linkage
        return function

    def parse_function_attribute(self):
        attribute_name = self.current_token[1]
        self.eat(self.current_token[0])
        values = []

        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            depth = 1
            while self.current_token[0] != "EOF" and depth > 0:
                token_type, token_value = self.current_token
                if token_type == "LPAREN":
                    depth += 1
                    values.append(token_value)
                    self.eat("LPAREN")
                elif token_type == "RPAREN":
                    depth -= 1
                    if depth == 0:
                        self.eat("RPAREN")
                        break
                    values.append(token_value)
                    self.eat("RPAREN")
                else:
                    values.append(token_value)
                    self.eat(token_type)

        return f"{attribute_name}({self.format_attribute_tokens(values)})"

    def parse_launch_bounds_attribute(self):
        return self.parse_function_attribute()

    def format_attribute_tokens(self, values):
        text = "".join(values).strip()
        return text.replace(",", ", ")

    def parse_parameters(self):
        self.eat("LPAREN")
        params = []

        if (
            self.current_token[0] == "VOID"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "RPAREN"
        ):
            self.eat("VOID")
            self.eat("RPAREN")
            return params

        if self.current_token[0] != "RPAREN":
            params.append(self.parse_parameter())

            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.parse_parameter())

        self.eat("RPAREN")
        return params

    def parse_parameter(self):
        param_type = self.parse_type()
        param_name = self.eat("IDENTIFIER")[1]
        param_type += self.parse_array_suffix()
        return VariableNode(param_type, param_name)

    def parse_type(self):
        type_parts = []
        saw_integral_sign = False

        while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.current_token[0] in self.TYPE_TOKENS:
            type_parts.append(self.parse_type_name())

        self.parse_postfix_type_qualifiers(type_parts)
        while self.current_token[0] == "MULTIPLY":
            type_parts.append("*")
            self.eat("MULTIPLY")
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                type_parts.append(f"[{self.current_token[1]}]")
                self.eat("NUMBER")
            else:
                type_parts.append("[]")
            self.eat("RBRACKET")

        return " ".join(type_parts)

    def parse_type_without_array_suffix(self):
        type_parts = []
        saw_integral_sign = False

        while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.current_token[0] in self.TYPE_TOKENS:
            type_parts.append(self.parse_type_name())

        self.parse_postfix_type_qualifiers(type_parts)
        while self.current_token[0] == "MULTIPLY":
            type_parts.append("*")
            self.eat("MULTIPLY")
            self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def is_implicit_int_current_type(self, saw_integral_sign):
        return self.is_implicit_int_type_at_index(self.current_index, saw_integral_sign)

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.current_token[0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

    def parse_type_name(self):
        token_type = self.current_token[0]
        type_name = self.current_token[1]
        self.eat(token_type)
        if token_type == "LONG" and self.current_token[0] == "LONG":
            type_name += f" {self.current_token[1]}"
            self.eat("LONG")

        while self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            member = self.eat("IDENTIFIER")[1]
            type_name += f"::{member}"

        if self.current_token[0] == "LESS_THAN":
            type_name += self.parse_template_suffix()

        return type_name

    def parse_global_variable(self):
        qualifiers = []

        while self.current_token[0] in ["CONSTANT", "SHARED", "DEVICE", "MANAGED"]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        var = self.parse_variable_declaration()
        var.qualifiers = qualifiers
        self.eat("SEMICOLON")

        if "__constant__" in qualifiers:
            return ConstantMemoryNode(var.vtype, var.name, var.value)
        elif "__shared__" in qualifiers:
            return self.create_shared_memory_node(var.vtype, var.name, qualifiers)
        else:
            return var

    def parse_variable_declaration(self):
        qualifiers = []

        while self.current_token[0] in [
            "SHARED",
            "CONSTANT",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
            "DEVICE",
            "MANAGED",
        ]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        vtype = self.parse_type()
        name = self.eat("IDENTIFIER")[1]
        vtype += self.parse_array_suffix()

        value = None
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            value = self.parse_expression()
        elif self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()
            value = FunctionCallNode(vtype, args)
        elif self.current_token[0] == "LBRACE":
            value = self.parse_initializer_list()

        var = VariableNode(vtype, name, value, qualifiers)

        if "__shared__" in qualifiers:
            return self.create_shared_memory_node(vtype, name, qualifiers)
        elif "__constant__" in qualifiers:
            return ConstantMemoryNode(vtype, name, value)
        else:
            return var

    def parse_variable_declaration_list(self):
        qualifiers = []

        while self.current_token[0] in [
            "SHARED",
            "CONSTANT",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
            "DEVICE",
            "MANAGED",
        ]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        declarations = [
            self.parse_variable_declarator(first_type, qualifiers, allow_prefix=False)
        ]

        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            declarations.append(
                self.parse_variable_declarator(base_type, qualifiers, allow_prefix=True)
            )

        return declarations

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        vtype = base_type
        if allow_prefix:
            vtype = self.parse_declarator_prefix(vtype)

        name = self.eat("IDENTIFIER")[1]
        vtype += self.parse_array_suffix()
        value = self.parse_variable_initializer(vtype)

        if "__shared__" in qualifiers:
            return self.create_shared_memory_node(vtype, name, qualifiers)
        if "__constant__" in qualifiers:
            return ConstantMemoryNode(vtype, name, value)
        return VariableNode(vtype, name, value, list(qualifiers))

    def create_shared_memory_node(self, vtype, name, qualifiers):
        is_extern = "extern" in qualifiers
        is_dynamic = is_extern and isinstance(vtype, str) and vtype.endswith("[]")
        return SharedMemoryNode(
            vtype,
            name,
            is_extern=is_extern,
            is_dynamic=is_dynamic,
        )

    def parse_declarator_prefix(self, base_type):
        parts = [base_type] if base_type else []

        while self.current_token[0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            if self.current_token[0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
                parts.append(self.current_token[1])
                self.eat(self.current_token[0])
                continue

            parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(parts)

        return " ".join(parts)

    def parse_variable_initializer(self, vtype):
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            return self.parse_expression()
        if self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()
            return FunctionCallNode(vtype, args)
        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        return None

    def strip_declarator_markers(self, vtype):
        parts = str(vtype).split()
        declarator_markers = {
            "*",
            "&",
            "&&",
            "__restrict__",
            "__restrict",
            "restrict",
        }
        while parts and parts[-1] in declarator_markers:
            parts.pop()
        return " ".join(parts)

    def parse_array_suffix(self):
        suffixes = []

        while self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
                suffixes.append(f"[{self.expression_to_text(size)}]")
            else:
                suffixes.append("[]")
            self.eat("RBRACKET")

        return "".join(suffixes)

    def parse_argument_list(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def parse_block(self):
        self.eat("LBRACE")
        statements = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            stmt = self.parse_statement()
            if stmt:
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
        if self.current_token[0] == "PREPROCESSOR":
            return self.parse_preprocessor()
        if self.current_token[0] == "IF":
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
        elif self.current_token[0] in ["SYNCTHREADS", "SYNCWARP"]:
            return self.parse_sync_statement()
        elif self.current_token[0] == "ASM":
            return self.parse_asm_statement()
        elif self.current_token[0] == "LBRACE":
            return self.parse_block()
        elif self.is_type_alias_start():
            return self.parse_type_alias()
        elif self.is_identifier_value("delete"):
            stmt = self.parse_delete_statement()
            self.eat("SEMICOLON")
            return stmt
        elif self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            self.eat("SEMICOLON")
            return declarations if len(declarations) > 1 else declarations[0]
        else:
            expr = self.parse_assignment_expression()
            self.eat("SEMICOLON")
            return expr

    def is_identifier_value(self, value):
        return self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value

    def parse_delete_statement(self):
        self.eat("IDENTIFIER")
        is_array = False

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            is_array = True

        expression = self.parse_unary_expression()
        return DeleteNode(expression, is_array)

    def is_variable_declaration(self):
        saved_index = self.current_index

        while (
            saved_index < len(self.tokens)
            and self.tokens[saved_index][0] in self.DECLARATION_QUALIFIER_TOKENS
        ):
            saved_index += 1

        saved_index = self.skip_type_at_index(saved_index)

        if saved_index is not None:
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] == "IDENTIFIER"
            ):
                saved_index += 1
                if saved_index >= len(self.tokens):
                    return True
                return self.tokens[saved_index][0] in [
                    "SEMICOLON",
                    "ASSIGN",
                    "LBRACKET",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                ]
        return False

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")

        if_body = self.parse_statement()

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement()

        return IfNode(condition, if_body, else_body)

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list()
                init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
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
            update = self.parse_for_update_expression()
        self.eat("RPAREN")

        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_for_update_expression(self):
        updates = [self.parse_assignment_expression()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            updates.append(self.parse_assignment_expression())
        return updates if len(updates) > 1 else updates[0]

    def is_range_for_statement(self):
        index = self.skip_range_for_type_at_index(self.current_index)
        if index is None:
            return False

        if index >= len(self.tokens) or self.tokens[index][0] != "IDENTIFIER":
            return False

        index += 1
        index = self.skip_array_suffix_at_index(index)
        return index < len(self.tokens) and self.tokens[index][0] == "COLON"

    def skip_range_for_type_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        if index >= len(self.tokens) or self.tokens[index][0] not in self.TYPE_TOKENS:
            return None

        type_token = self.tokens[index][0]
        index += 1
        index = self.skip_long_long_suffix_at_index(index, type_token)
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] == "IDENTIFIER"
        ):
            index += 2

        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            index = self.skip_template_at_index(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_index(index)
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return self.skip_array_suffix_at_index(index)

    def parse_range_for_statement(self):
        vtype = self.parse_type()
        name = self.eat("IDENTIFIER")[1]
        self.eat("COLON")
        iterable = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement()

        return RangeForNode(vtype, name, iterable, body)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_statement()
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
            if self.current_token[0] == "CASE":
                self.eat("CASE")
                value = self.parse_expression()
                self.eat("COLON")
                body = []
                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE"]:
                    body.append(self.parse_statement())
                case = CaseNode(value, body)
                cases.append(case)
                ordered_cases.append(case)
            elif self.current_token[0] == "DEFAULT":
                if seen_default:
                    raise SyntaxError("duplicate default label in switch")
                seen_default = True
                self.eat("DEFAULT")
                self.eat("COLON")
                default_case = []
                while self.current_token[0] not in ["CASE", "RBRACE"]:
                    default_case.append(self.parse_statement())
                ordered_cases.append(CaseNode(None, default_case))
            else:
                self.eat(self.current_token[0])

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

    def parse_sync_statement(self):
        sync_type = self.current_token[1]
        self.eat(self.current_token[0])

        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        self.eat("SEMICOLON")

        return SyncNode(sync_type, args)

    def parse_asm_statement(self):
        self.eat("ASM")
        is_volatile = False
        if self.current_token[0] == "VOLATILE":
            is_volatile = True
            self.eat("VOLATILE")

        self.eat("LPAREN")
        template = self.parse_asm_string_literal()
        outputs = []
        inputs = []
        clobbers = []

        section_index = self.consume_asm_section_delimiters()
        if section_index == 1:
            outputs = self.parse_asm_operands()
        elif section_index == 2:
            inputs = self.parse_asm_operands()
        elif section_index >= 3:
            clobbers = self.parse_asm_clobbers()

        while self.current_token[0] in {"COLON", "SCOPE"}:
            section_index += self.consume_asm_section_delimiters()
            if section_index == 2:
                inputs = self.parse_asm_operands()
            elif section_index >= 3:
                clobbers = self.parse_asm_clobbers()
                break

        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return CudaAsmNode(template, outputs, inputs, clobbers, is_volatile)

    def consume_asm_section_delimiters(self):
        delimiter_count = 0
        while self.current_token[0] in {"COLON", "SCOPE"}:
            if self.current_token[0] == "COLON":
                delimiter_count += 1
                self.eat("COLON")
            else:
                delimiter_count += 2
                self.eat("SCOPE")
        return delimiter_count

    def parse_asm_string_literal(self):
        value = self.eat("STRING")[1]
        while self.current_token[0] == "STRING":
            next_value = self.eat("STRING")[1]
            if value.endswith('"') and next_value.startswith('"'):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
        return value

    def parse_asm_operands(self):
        operands = []

        while self.current_token[0] not in {"COLON", "SCOPE", "RPAREN", "EOF"}:
            symbolic_name = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                symbolic_name = self.eat("IDENTIFIER")[1]
                self.eat("RBRACKET")

            constraint = self.eat("STRING")[1]
            expression = None
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                if self.current_token[0] != "RPAREN":
                    expression = self.parse_expression()
                self.eat("RPAREN")

            operands.append(CudaAsmOperandNode(constraint, expression, symbolic_name))
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        return operands

    def parse_asm_clobbers(self):
        clobbers = []

        while self.current_token[0] not in {"RPAREN", "EOF"}:
            clobbers.append(self.eat("STRING")[1])
            if self.current_token[0] != "COMMA":
                break
            self.eat("COMMA")

        return clobbers

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_ternary_expression(self):
        expr = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()

        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()

        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()

        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_shift_expression(self):
        left = self.parse_additive_expression()

        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MODULO"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        if self.current_token[0] in [
            "PLUS",
            "MINUS",
            "LOGICAL_NOT",
            "BITWISE_NOT",
            "MULTIPLY",
            "BITWISE_AND",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)
        elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_postfix_expression()
            return UnaryOpNode(op, operand)
        else:
            return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        left = self.parse_primary_expression()

        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.eat("IDENTIFIER")[1]
                left = MemberAccessNode(left, member, False)
            elif self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                member = self.eat("IDENTIFIER")[1]
                left = self.append_qualified_name(left, "::", member)
            elif self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                left = self.append_template_suffix(left)
            elif self.current_token[0] == "ARROW":
                self.eat("ARROW")
                member = self.eat("IDENTIFIER")[1]
                left = MemberAccessNode(left, member, True)
            elif self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                left = ArrayAccessNode(left, index)
            elif self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                args = []
                if left == "sizeof" and self.is_sizeof_type_operand():
                    args.append(self.parse_type())
                elif self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")

                if (
                    isinstance(left, str)
                    and left in self.ATOMIC_FUNCTION_NAMES
                    and left not in self.user_function_names
                ):
                    left = AtomicOperationNode(left, args)
                else:
                    left = self.parse_function_call_node(left, args)
            elif self.current_token[0] == "KERNEL_LAUNCH_START":
                return self.parse_kernel_launch(left)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(f"post{op}", left)
            else:
                break

        return left

    def append_qualified_name(self, base, separator, member):
        if isinstance(base, str):
            return f"{base}{separator}{member}"
        return f"{self.expression_to_text(base)}{separator}{member}"

    def is_template_suffix(self):
        index = self.current_index
        if self.tokens[index][0] != "LESS_THAN":
            return False

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LESS_THAN":
                depth += 1
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                    }
            elif token_type == "SHIFT_RIGHT":
                depth -= 2
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                    }
                if depth < 0:
                    return False
            elif token_type in {"SEMICOLON", "ASSIGN", "EOF"}:
                return False
            index += 1

        return False

    def append_template_suffix(self, base):
        suffix = self.parse_template_suffix()
        if isinstance(base, str):
            return f"{base}{suffix}"
        return f"{self.expression_to_text(base)}{suffix}"

    def parse_template_suffix(self):
        self.eat("LESS_THAN")
        parts = []
        depth = 1

        while depth > 0:
            token_type, token_value = self.current_token
            if token_type == "LESS_THAN":
                depth += 1
                parts.append(token_value)
                self.eat("LESS_THAN")
            elif token_type == "GREATER_THAN":
                depth -= 1
                if depth == 0:
                    self.eat("GREATER_THAN")
                    break
                parts.append(token_value)
                self.eat("GREATER_THAN")
            elif token_type == "SHIFT_RIGHT":
                self.eat("SHIFT_RIGHT")
                for _ in range(2):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            else:
                parts.append(token_value)
                self.eat(token_type)

        return f"<{self.format_template_parts(parts)}>"

    def parse_template_declaration(self):
        self.eat("TEMPLATE")
        if self.current_token[0] == "LESS_THAN":
            self.parse_template_suffix()
            return

        self.collect_balanced_tokens_until("SEMICOLON")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def format_template_parts(self, parts):
        formatted = []
        previous = None

        for part in parts:
            if part == ",":
                formatted.append(", ")
            else:
                if self.needs_template_part_space(previous, part):
                    formatted.append(" ")
                formatted.append(part)
            previous = part

        return "".join(formatted).strip()

    def needs_template_part_space(self, previous, current):
        if previous is None:
            return False
        if previous in {"<", "[", "(", "::", ","}:
            return False
        if current in {">", "]", ")", ",", "::", "<", "[", "(", "*", "&", "&&"}:
            return False
        if previous in {"*", "&", "&&"}:
            return False
        return self.is_template_word(previous) and self.is_template_word(current)

    def is_template_word(self, part):
        return part.replace("_", "").isalnum()

    def parse_function_call_node(self, function_name, args):
        named_cast = self.parse_cpp_named_cast_call(function_name, args)
        if named_cast is not None:
            return named_cast

        if (
            function_name == "cudaLaunchKernel"
            and len(args) == 6
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_cuda_kernel_function_arg(args[0]),
                args[1],
                args[2],
                args[4],
                args[5],
                [args[3]],
            )

        return FunctionCallNode(function_name, args)

    def parse_cpp_named_cast_call(self, function_name, args):
        if not isinstance(function_name, str) or len(args) != 1:
            return None

        for cast_name in self.CPP_NAMED_CASTS:
            prefix = f"{cast_name}<"
            if function_name.startswith(prefix) and function_name.endswith(">"):
                return CastNode(function_name[len(prefix) : -1], args[0])

        return None

    def unwrap_cuda_kernel_function_arg(self, function_arg):
        if isinstance(function_arg, CastNode):
            return function_arg.expression
        return function_arg

    def is_sizeof_type_operand(self):
        saved_index = self.current_index
        try:
            while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
                self.eat(self.current_token[0])

            if (
                self.current_token[0] not in self.TYPE_TOKENS
                or self.current_token[0] == "IDENTIFIER"
            ):
                return False

            token_type = self.current_token[0]
            self.eat(token_type)
            if token_type == "LONG" and self.current_token[0] == "LONG":
                self.eat("LONG")
            while self.current_token[0] == "MULTIPLY":
                self.eat("MULTIPLY")

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                while self.current_token[0] not in ["RBRACKET", "EOF"]:
                    self.eat(self.current_token[0])
                self.eat("RBRACKET")

            return self.current_token[0] == "RPAREN"
        finally:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]

    def parse_kernel_launch(self, kernel_name):
        self.eat("KERNEL_LAUNCH_START")

        blocks = self.parse_expression()
        self.eat("COMMA")

        threads = self.parse_expression()

        shared_mem = None
        stream = None

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            shared_mem = self.parse_expression()

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                stream = self.parse_expression()

        self.eat("KERNEL_LAUNCH_END")

        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")

        return KernelLaunchNode(kernel_name, blocks, threads, shared_mem, stream, args)

    def parse_primary_expression(self):
        if self.current_token[0] == "LBRACKET" and self.is_lambda_expression_start():
            return self.parse_lambda_expression()
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
            self.eat("STRING")
            while self.current_token[0] == "STRING":
                next_value = self.current_token[1]
                if value.endswith('"') and next_value.startswith('"'):
                    value = value[:-1] + next_value[1:]
                else:
                    value += next_value
                self.eat("STRING")
            return value
        elif self.current_token[0] == "CHAR_LIT":
            value = self.current_token[1]
            self.eat("CHAR_LIT")
            return value
        elif self.current_token[0] in ["TRUE", "FALSE"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif self.current_token[0] in ["NULL", "NULLPTR"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif (
            self.current_token[0] in self.TYPE_TOKENS
            and self.current_token[0] != "IDENTIFIER"
        ):
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif self.current_token[0] == "IDENTIFIER":
            if self.current_token[1] == "new":
                return self.parse_new_expression()

            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name
        elif self.current_token[0] in self.ATOMIC_FUNCTION_TOKENS:
            name = self.current_token[1]
            self.eat(self.current_token[0])
            return name
        elif self.current_token[0] in [
            "THREADIDX",
            "BLOCKIDX",
            "GRIDDIM",
            "BLOCKDIM",
            "WARPSIZE",
        ]:
            builtin_name = self.current_token[1]
            self.eat(self.current_token[0])

            if self.current_token[0] == "DOT":
                self.eat("DOT")
                component = self.eat("IDENTIFIER")[1]
                return CudaBuiltinNode(builtin_name, component)
            else:
                return CudaBuiltinNode(builtin_name)
        elif self.current_token[0] == "LPAREN":
            if self.is_cast_expression():
                self.eat("LPAREN")
                target_type = self.parse_type()
                self.eat("RPAREN")
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def is_lambda_expression_start(self):
        if self.current_token[0] != "LBRACKET":
            return False

        index = self.current_index
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    index += 1
                    break
            elif token_type == "EOF":
                return False
            index += 1
        else:
            return False

        index = self.skip_lambda_specifiers_at_index(index)
        return index < len(self.tokens) and self.tokens[index][0] in {
            "LPAREN",
            "LBRACE",
        }

    def skip_lambda_specifiers_at_index(self, index):
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.LAMBDA_SPECIFIER_TOKENS or (
                token_type == "IDENTIFIER"
                and token_value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                index += 1
                if (
                    token_value == "noexcept"
                    and index < len(self.tokens)
                    and self.tokens[index][0] == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_index(
                        index, "LPAREN", "RPAREN"
                    )
                continue
            break
        return index

    def skip_balanced_tokens_at_index(self, index, open_token, close_token):
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "EOF":
                return index
            index += 1
        return index

    def parse_lambda_expression(self):
        self.consume_balanced_lambda_tokens("LBRACKET", "RBRACKET")
        self.skip_lambda_specifiers()

        args = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                args.append(self.parse_lambda_parameter())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                elif self.current_token[0] != "RPAREN":
                    raise SyntaxError(
                        f"Expected COMMA or RPAREN in lambda parameters, got {self.current_token[0]}"
                    )
            self.eat("RPAREN")

        self.skip_lambda_specifiers()
        if self.current_token[0] == "ARROW":
            self.skip_lambda_trailing_return_type()
            self.skip_lambda_specifiers()

        args.append(self.parse_lambda_block_body())
        return FunctionCallNode("lambda", args)

    def skip_lambda_specifiers(self):
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in self.LAMBDA_SPECIFIER_TOKENS or (
                token_type == "IDENTIFIER"
                and token_value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                self.eat(token_type)
                if token_value == "noexcept" and self.current_token[0] == "LPAREN":
                    self.consume_balanced_lambda_tokens("LPAREN", "RPAREN")
                continue
            break

    def skip_lambda_trailing_return_type(self):
        self.eat("ARROW")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "LBRACE"
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
            self.eat(token_type)

    def parse_lambda_parameter(self):
        saved_index = self.current_index
        try:
            param_type = self.parse_type()
            if self.current_token[0] != "IDENTIFIER":
                raise SyntaxError("Expected lambda parameter name")
            param_name = self.eat("IDENTIFIER")[1]
            param_type += self.parse_array_suffix()
            self.skip_lambda_parameter_default()
            return VariableNode(param_type, param_name)
        except SyntaxError:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]
            raw = self.collect_lambda_parameter_raw()
            return VariableNode("", raw)

    def skip_lambda_parameter_default(self):
        if self.current_token[0] != "ASSIGN":
            return

        self.eat("ASSIGN")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
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
            elif token_type == "LBRACE":
                brace_depth += 1
            elif token_type == "RBRACE" and brace_depth > 0:
                brace_depth -= 1
            self.eat(token_type)

    def collect_lambda_parameter_raw(self):
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

            tokens.append(self.current_token)
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
            self.eat(token_type)

        raw = self.format_lambda_raw_tokens(tokens).strip()
        if not raw:
            raise SyntaxError("Expected lambda parameter")
        return raw

    def parse_lambda_block_body(self):
        expression_body = self.try_parse_lambda_return_expression()
        if expression_body is not None:
            return expression_body

        return self.parse_raw_lambda_block_body()

    def try_parse_lambda_return_expression(self):
        saved_index = self.current_index
        completed = False
        try:
            self.eat("LBRACE")
            if self.current_token[0] != "RETURN":
                return None
            self.eat("RETURN")
            if self.current_token[0] == "SEMICOLON":
                return None
            value = self.parse_expression()
            if self.current_token[0] != "SEMICOLON":
                return None
            self.eat("SEMICOLON")
            if self.current_token[0] != "RBRACE":
                return None
            self.eat("RBRACE")
            completed = True
            return value
        except SyntaxError:
            return None
        finally:
            if not completed:
                self.current_index = saved_index
                self.current_token = self.tokens[self.current_index]

    def parse_raw_lambda_block_body(self):
        tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token = self.current_token
            token_type = token[0]
            tokens.append(token)
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                self.eat("RBRACE")
                if depth == 0:
                    return self.format_lambda_raw_tokens(tokens)
                continue
            self.eat(token_type)

        raise SyntaxError("Unterminated lambda block body")

    def consume_balanced_lambda_tokens(self, open_token, close_token):
        depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                self.eat(close_token)
                if depth == 0:
                    return
                continue
            self.eat(token_type)

        raise SyntaxError(f"Unterminated lambda {open_token}")

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
            elif value in {"(", "[", ".", "::", "->"}:
                text = text.rstrip() + value
            elif value == "{":
                if not text.endswith(" "):
                    text += " "
                text += value + " "
            elif previous and previous[1] in {"(", "[", ".", "::", "->"}:
                text += value
            elif text.endswith((" ", "{")):
                text += value
            else:
                text += " " + value
            previous = (token_type, value)
        return text.strip()

    def parse_new_expression(self):
        self.eat("IDENTIFIER")
        target_type = self.parse_type_without_array_suffix()

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            return NewNode(target_type, size=size, is_array=True)

        args = []
        if self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()

        return NewNode(target_type, args=args)

    def parse_initializer_list(self):
        self.eat("LBRACE")
        elements = []

        while self.current_token[0] != "RBRACE":
            elements.append(self.parse_initializer_element())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RBRACE":
                    break
            else:
                break

        self.eat("RBRACE")
        return InitializerListNode(elements)

    def parse_initializer_element(self):
        if self.current_token[0] in ["LBRACKET", "DOT"]:
            return self.parse_designated_initializer()
        return self.parse_expression()

    def parse_designated_initializer(self):
        designators = []

        while self.current_token[0] in ["LBRACKET", "DOT"]:
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                designators.append(("index", index))
            else:
                self.eat("DOT")
                field = self.eat("IDENTIFIER")[1]
                designators.append(("field", field))

        self.eat("ASSIGN")
        value = self.parse_expression()
        return DesignatedInitializerNode(designators, value)

    def is_cast_expression(self):
        if self.current_token[0] != "LPAREN":
            return False

        saved_index = self.current_index
        try:
            self.eat("LPAREN")
            while self.current_token[0] in self.TYPE_QUALIFIER_TOKENS:
                self.eat(self.current_token[0])

            if self.current_token[0] not in self.TYPE_TOKENS or (
                self.current_token[0] == "IDENTIFIER"
                and not self.is_identifier_type_name(self.current_token[1])
                and not self.is_identifier_cast_target_at_current_index()
            ):
                return False

            token_type = self.current_token[0]
            self.eat(token_type)
            if token_type == "LONG" and self.current_token[0] == "LONG":
                self.eat("LONG")
            while self.current_token[0] == "MULTIPLY":
                self.eat("MULTIPLY")

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                while self.current_token[0] not in ["RBRACKET", "EOF"]:
                    self.eat(self.current_token[0])
                self.eat("RBRACKET")

            return self.current_token[0] == "RPAREN"
        finally:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]

    def is_identifier_cast_target_at_current_index(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        close_index = self.current_index + 1
        while close_index < len(self.tokens) and self.tokens[close_index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            close_index += 1

        if close_index >= len(self.tokens) or self.tokens[close_index][0] != "RPAREN":
            return False

        operand_index = close_index + 1
        if operand_index >= len(self.tokens):
            return False

        return self.tokens[operand_index][0] in {
            "IDENTIFIER",
            "NUMBER",
            "STRING",
            "CHAR_LIT",
            "TRUE",
            "FALSE",
            "NULL",
            "NULLPTR",
            "LPAREN",
            "LBRACKET",
            "PLUS",
            "MINUS",
            "LOGICAL_NOT",
            "BITWISE_NOT",
            "MULTIPLY",
            "BITWISE_AND",
            "INCREMENT",
            "DECREMENT",
            *self.ATOMIC_FUNCTION_TOKENS,
            "THREADIDX",
            "BLOCKIDX",
            "GRIDDIM",
            "BLOCKDIM",
            "WARPSIZE",
        }

    def expression_to_text(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, CudaBuiltinNode):
            if expr.component:
                return f"{expr.builtin_name}.{expr.component}"
            return expr.builtin_name
        if isinstance(expr, BinaryOpNode):
            left = self.expression_to_text(expr.left)
            right = self.expression_to_text(expr.right)
            return f"({left} {expr.op} {right})"
        if isinstance(expr, UnaryOpNode):
            return f"{expr.op}{self.expression_to_text(expr.operand)}"
        return str(expr)

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()

        if self.current_token[0] in [
            "ASSIGN",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
            "MODULO_EQUALS",
            "AND_EQUALS",
            "OR_EQUALS",
            "XOR_EQUALS",
            "SHIFT_LEFT_EQUALS",
            "SHIFT_RIGHT_EQUALS",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left
