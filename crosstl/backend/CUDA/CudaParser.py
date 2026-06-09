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
    POSTFIX_TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "RESTRICT"}
    TYPE_REFERENCE_TOKENS = {"BITWISE_AND", "LOGICAL_AND"}
    ELABORATED_TYPE_TOKENS = {"CLASS", "STRUCT", "UNION", "ENUM"}
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
        "atomicAdd_block",
        "atomicAdd_system",
        "atomicSub",
        "atomicSub_block",
        "atomicSub_system",
        "atomicMax",
        "atomicMax_block",
        "atomicMax_system",
        "atomicMin",
        "atomicMin_block",
        "atomicMin_system",
        "atomicExch",
        "atomicExch_block",
        "atomicExch_system",
        "atomicCAS",
        "atomicCAS_block",
        "atomicCAS_system",
        "atomicAnd",
        "atomicAnd_block",
        "atomicAnd_system",
        "atomicOr",
        "atomicOr_block",
        "atomicOr_system",
        "atomicXor",
        "atomicXor_block",
        "atomicXor_system",
        "atomicInc",
        "atomicInc_block",
        "atomicInc_system",
        "atomicDec",
        "atomicDec_block",
        "atomicDec_system",
    }
    FUNCTION_NAME_TOKENS = {"IDENTIFIER", *ATOMIC_FUNCTION_TOKENS}
    OVERLOADABLE_OPERATOR_TOKENS = {
        "PLUS",
        "MINUS",
        "MULTIPLY",
        "DIVIDE",
        "MODULO",
        "BITWISE_AND",
        "BITWISE_OR",
        "BITWISE_XOR",
        "BITWISE_NOT",
        "LOGICAL_NOT",
        "ASSIGN",
        "LESS_THAN",
        "GREATER_THAN",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "MODULO_EQUALS",
        "AND_EQUALS",
        "OR_EQUALS",
        "XOR_EQUALS",
        "EQUAL",
        "NOT_EQUAL",
        "SPACESHIP",
        "LESS_EQUAL",
        "GREATER_EQUAL",
        "LOGICAL_AND",
        "LOGICAL_OR",
        "SHIFT_LEFT",
        "SHIFT_RIGHT",
        "SHIFT_LEFT_EQUALS",
        "SHIFT_RIGHT_EQUALS",
        "INCREMENT",
        "DECREMENT",
        "LBRACKET",
        "LPAREN",
    }
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
        "_CCCL_DEVICE",
        "constexpr",
        "consteval",
        "noexcept",
        "__noexcept",
    }
    STRUCT_METHOD_IDENTIFIER_SPECIFIERS = {
        "explicit",
        "friend",
    }
    STATEMENT_DIRECTIVE_FOLLOW_TOKENS = {
        "FOR",
        "IF",
        "WHILE",
        "DO",
        "SWITCH",
        "LBRACE",
    }
    TOP_LEVEL_MACRO_STATEMENT_FOLLOW_TOKENS = {
        "EOF",
        "PREPROCESSOR",
        "TEMPLATE",
        "NAMESPACE",
        "STRUCT",
        "CLASS",
        "UNION",
        "ENUM",
        "TYPEDEF",
        "USING",
        "EXTERN",
    }
    FUNCTION_IDENTIFIER_SPECIFIERS = {
        "APIENTRY",
        "CALLBACK",
        "CUTE_HOST_DEVICE",
        "CUDARTAPI",
        "_CCCL_API",
        "_CCCL_DEVICE",
        "_CCCL_DEVICE_API",
        "_CCCL_FORCEINLINE",
        "_CCCL_HOST_API",
        "_CCCL_HOST_DEVICE",
        "_CCCL_HOST_DEVICE_API",
        "_CCCL_KERNEL_ATTRIBUTES",
        "_CCCL_NODEBUG_HOST_API",
        "VKAPI_ATTR",
        "VKAPI_CALL",
        "WINAPI",
        "consteval",
    }
    DECLARATION_QUALIFIER_TOKENS = {
        "CONST",
        "VOLATILE",
        "STATIC",
        "EXTERN",
        "CONSTEXPR",
        "REGISTER",
        "SHARED",
        "CONSTANT",
        "DEVICE",
        "MANAGED",
    }
    DECLARATION_PREFIX_TOKENS = {
        "STATIC",
        "EXTERN",
        "CONSTEXPR",
        "REGISTER",
        "MUTABLE",
        "SHARED",
        "CONSTANT",
        "DEVICE",
        "MANAGED",
    }
    CUDA_DECLARATION_ATTRIBUTE_IDENTIFIERS = {
        "__device_builtin__",
        "__device_builtin_",
        "__cudart_builtin__",
        "__VECTOR_TYPE_DEPRECATED__",
    }
    CUDA_STORAGE_QUALIFIER_TOKENS = {
        "SHARED",
        "CONSTANT",
        "DEVICE",
        "MANAGED",
    }
    CUDA_PARAMETER_ATTRIBUTE_TOKENS = set()
    CUDA_PARAMETER_ATTRIBUTE_IDENTIFIERS = {"_CCCL_GRID_CONSTANT"}
    CUDA_STATEMENT_MACRO_INVOCATION_IDENTIFIERS = {"NV_IF_TARGET"}
    CUDA_FUNCTION_ENTRY_TOKENS = {"GLOBAL", "TILE_GLOBAL", "TILE", "DEVICE", "HOST"}
    FUNCTION_ATTRIBUTE_TOKENS = {"LAUNCH_BOUNDS", "CLUSTER_DIMS", "BLOCK_SIZE"}
    FUNCTION_SPECIFIER_TOKENS = {
        "GLOBAL",
        "TILE_GLOBAL",
        "TILE",
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
    POST_RETURN_FUNCTION_SPECIFIER_TOKENS = {
        "GLOBAL",
        "TILE_GLOBAL",
        "TILE",
        "DEVICE",
        "HOST",
        "INLINE",
        "FORCEINLINE",
        "NOINLINE",
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
    NAME_COMPONENT_TOKENS = {
        "IDENTIFIER",
        "THREADIDX",
        "BLOCKIDX",
        "GRIDDIM",
        "BLOCKDIM",
        "WARPSIZE",
        *TYPE_TOKENS,
        *ATOMIC_FUNCTION_TOKENS,
    }
    CUDA_IDENTIFIER_TYPE_NAMES = {
        "cudaTextureObject_t",
        "cudaSurfaceObject_t",
        "uint",
        "half",
        "__half",
        "half2",
        "__half2",
        "__nv_bfloat16",
        "__nv_bfloat162",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
    }
    COMPOSITE_SCALAR_TYPE_SUFFIX_TOKENS = {
        "LONG": {"LONG", "INT", "DOUBLE"},
        "SHORT": {"INT"},
    }

    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None
        self.type_aliases = set()
        self.namespace_aliases = {}
        self.struct_names = self.collect_struct_names()
        self.user_function_names = self.collect_user_function_names()

    def collect_struct_names(self):
        names = set()
        for index, token in enumerate(self.tokens[:-1]):
            if (
                token[0] in {"CLASS", "STRUCT"}
                and self.tokens[index + 1][0] == "IDENTIFIER"
            ):
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
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        while index < len(self.tokens) and (
            self.tokens[index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(index)
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

        index = self.skip_cpp_attribute_specifiers_at_index(index)

        index = self.skip_type_at_index(index, allow_unknown_identifier_declarator=True)
        if index is None:
            return None

        while index < len(self.tokens) and (
            self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.tokens[index][0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(index)
        ):
            if self.tokens[index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
            else:
                index += 1

        name_end_index = self.function_name_end_index_at(index)
        if name_end_index is not None:
            suffix_index = self.skip_optional_function_template_suffix_at_index(
                name_end_index
            )
            if (
                suffix_index < len(self.tokens)
                and self.tokens[suffix_index][0] == "LPAREN"
                and self.is_plausible_function_parameter_list_at_index(suffix_index)
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
            elif self.is_cpp_attribute_specifier_start():
                self.skip_cpp_attribute_specifiers()
            elif self.is_concept_declaration_start():
                self.skip_concept_declaration()
            elif self.current_token[0] == "TEMPLATE":
                self.parse_template_declaration()
            elif self.is_namespace_alias_start():
                self.parse_namespace_alias()
            elif self.is_type_alias_start():
                aliases = self.parse_type_alias()
                if isinstance(aliases, list):
                    for alias in aliases:
                        if isinstance(alias, (StructNode, EnumNode)):
                            structs.append(alias)
                        elif alias is not None:
                            typedefs.append(alias)
                elif isinstance(aliases, (StructNode, EnumNode)):
                    structs.append(aliases)
                elif aliases is not None:
                    typedefs.append(aliases)
            elif self.is_struct_or_class_declaration_start():
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
                        self.append_parsed_global_variable(global_variables, item)
            elif self.is_out_of_class_member_definition_start():
                self.skip_out_of_class_member_definition()
            elif self.is_function_pointer_declaration_start():
                self.skip_out_of_class_member_definition()
            elif self.is_top_level_macro_statement_start():
                self.skip_top_level_macro_statement()
            elif self.is_macro_block_invocation_start():
                self.skip_macro_block_invocation()
            elif self.peek_function():
                func = self.parse_function()
                if func is None:
                    continue
                if isinstance(func, KernelNode):
                    kernels.append(func)
                else:
                    functions.append(func)
            elif self.current_token[0] in self.CUDA_FUNCTION_ENTRY_TOKENS:
                if self.current_token[0] == "DEVICE" and self.peek_variable():
                    self.append_parsed_global_variable(
                        global_variables, self.parse_global_variable()
                    )
                    continue
                if self.is_function_pointer_declaration_start():
                    self.skip_out_of_class_member_definition()
                    continue

                func = self.parse_function()
                if func is None:
                    continue
                if isinstance(func, KernelNode):
                    kernels.append(func)
                else:
                    functions.append(func)
            elif (
                self.current_token[0] in ["CONSTANT", "SHARED"] or self.peek_variable()
            ):
                self.append_parsed_global_variable(
                    global_variables, self.parse_global_variable()
                )
            else:
                self.eat(self.current_token[0])

        shader = ShaderNode(
            includes, functions, structs, global_variables, kernels, typedefs=typedefs
        )
        shader.namespace_aliases = dict(self.namespace_aliases)
        return shader

    def peek_function(self):
        saved_index = self.current_index

        saved_index = self.skip_cpp_attribute_specifiers_at_index(saved_index)
        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
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

        saved_index = self.skip_cpp_attribute_specifiers_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        while (
            saved_index is not None
            and saved_index < len(self.tokens)
            and (
                self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
                or self.tokens[saved_index][0]
                in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
                or self.is_function_identifier_specifier_at_index(saved_index)
            )
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
            else:
                saved_index += 1

        if saved_index is not None:
            name_end_index = self.function_name_end_index_at(saved_index)
            if name_end_index is not None:
                suffix_index = self.skip_optional_function_template_suffix_at_index(
                    name_end_index
                )
                return (
                    suffix_index < len(self.tokens)
                    and self.tokens[suffix_index][0] == "LPAREN"
                    and self.is_plausible_function_parameter_list_at_index(suffix_index)
                )

        return False

    def skip_optional_function_template_suffix_at_index(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index)
            if skipped is not None:
                return skipped
        return index

    def is_function_pointer_declaration_start(self):
        saved_index = self.current_index

        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
                continue
            saved_index += 1

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )
        if saved_index is None:
            return False

        while saved_index < len(self.tokens) and (
            self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.is_function_identifier_specifier_at_index(saved_index)
        ):
            if self.tokens[saved_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                saved_index = self.skip_function_attribute_at_index(saved_index)
            else:
                saved_index += 1

        return (
            saved_index + 3 < len(self.tokens)
            and self.tokens[saved_index][0] == "LPAREN"
            and self.tokens[saved_index + 1][0] == "MULTIPLY"
            and self.tokens[saved_index + 2][0] in self.NAME_COMPONENT_TOKENS
            and self.tokens[saved_index + 3][0] in {"RPAREN", "LPAREN"}
        )

    def peek_variable(self):
        saved_index = self.current_index

        saved_index = self.skip_declaration_prefixes_at_index(saved_index)
        saved_index = self.skip_alignment_attributes_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        if saved_index is not None:
            saved_index = self.skip_interleaved_declaration_qualifiers_at_index(
                saved_index
            )
            saved_index = self.skip_alignment_attributes_at_index(saved_index)
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] in self.NAME_COMPONENT_TOKENS
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

    def skip_declaration_prefixes_at_index(self, index):
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in self.DECLARATION_PREFIX_TOKENS:
                index += 1
                continue
            if (
                token_type == "VOLATILE"
                and index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] in self.CUDA_STORAGE_QUALIFIER_TOKENS
            ):
                index += 1
                continue
            break
        return index

    def skip_interleaved_declaration_qualifiers_at_index(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.DECLARATION_PREFIX_TOKENS
        ):
            index += 1
        return index

    def skip_alignment_attributes_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] == "ALIGNAS":
            index += 1
            if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
                continue

            depth = 0
            while index < len(self.tokens):
                token_type = self.tokens[index][0]
                if token_type == "LPAREN":
                    depth += 1
                elif token_type == "RPAREN":
                    depth -= 1
                    if depth == 0:
                        index += 1
                        break
                elif token_type == "EOF":
                    return index
                index += 1

        return index

    def skip_gnu_attributes_at_index(self, index):
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index] == ("IDENTIFIER", "__attribute__")
            and self.tokens[index + 1][0] == "LPAREN"
        ):
            index = self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")
        return index

    def skip_type_at_index(self, index, allow_unknown_identifier_declarator=False):
        saw_integral_sign = False
        saw_type_qualifier = False
        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_QUALIFIER_TOKENS
        ):
            saw_type_qualifier = True
            if self.tokens[index][0] in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        if index >= len(self.tokens):
            return None

        has_qualified_suffix = False

        if self.is_implicit_int_type_at_index(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.is_decltype_type_start_at_index(index):
            type_token = "IDENTIFIER"
            type_value = "decltype"
            index = self.skip_decltype_type_at_index(index)
            if index is None:
                return None
        elif self.tokens[index][0] in self.ELABORATED_TYPE_TOKENS:
            type_token = self.tokens[index][0]
            type_value = self.tokens[index][1]
            index += 1
            if (
                index < len(self.tokens)
                and self.tokens[index][0] in self.NAME_COMPONENT_TOKENS
            ):
                type_value += f" {self.tokens[index][1]}"
                index += 1
            else:
                return None
        elif self.tokens[index][0] in self.TYPE_TOKENS:
            type_token = self.tokens[index][0]
            type_value = self.tokens[index][1]
            index += 1
            index = self.skip_composite_scalar_type_suffix_at_index(index, type_token)
        elif self.is_global_qualified_name_start_at_index(index):
            type_token = "IDENTIFIER"
            type_value = f"::{self.tokens[index + 1][1]}"
            has_qualified_suffix = True
            index += 2
        else:
            return None

        while True:
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                has_qualified_suffix = True
                index = self.skip_template_at_index(index)
                if index is None:
                    return None
                continue
            if self.is_qualified_type_member_at_index(index):
                has_qualified_suffix = True
                index = self.skip_qualified_type_member_at_index(index)
                continue
            break

        index = self.skip_postfix_type_qualifiers_at_index(index)
        can_have_pointer_suffix = (
            type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or type_value == "decltype"
            or self.is_identifier_type_name(type_value)
            or type_token in self.ELABORATED_TYPE_TOKENS
            or saw_type_qualifier
            or (
                allow_unknown_identifier_declarator
                and self.is_probable_identifier_type_name(type_value)
                and self.has_unknown_identifier_declarator_suffix_at_index(index)
            )
        )
        while (
            can_have_pointer_suffix
            and index < len(self.tokens)
            and self.tokens[index][0] in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return self.skip_array_suffix_at_index(index)

    def is_probable_identifier_type_name(self, name):
        return (
            name in self.type_aliases
            or name in self.struct_names
            or name in self.CUDA_IDENTIFIER_TYPE_NAMES
            or name.endswith("_t")
            or any(character.isupper() for character in name)
        )

    def has_unknown_identifier_declarator_suffix_at_index(self, index):
        suffix_index = index
        saw_declarator_marker = False

        while suffix_index < len(self.tokens):
            token_type = self.tokens[suffix_index][0]
            if token_type in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}:
                saw_declarator_marker = True
                suffix_index += 1
                suffix_index = self.skip_postfix_type_qualifiers_at_index(suffix_index)
                continue
            if token_type in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
                suffix_index += 1
                continue
            break

        if (
            not saw_declarator_marker
            or suffix_index >= len(self.tokens)
            or self.tokens[suffix_index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return False

        suffix_index += 1
        suffix_index = self.skip_array_suffix_at_index(suffix_index)
        return suffix_index < len(self.tokens) and self.tokens[suffix_index][0] in {
            "SEMICOLON",
            "ASSIGN",
            "LPAREN",
            "LBRACE",
            "COMMA",
        }

    def is_function_identifier_specifier_at_index(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] in self.FUNCTION_IDENTIFIER_SPECIFIERS
        )

    def is_function_identifier_specifier(self):
        return self.is_function_identifier_specifier_at_index(self.current_index)

    def is_plausible_function_parameter_list_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LPAREN":
            return False

        index += 1
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        index = self.skip_cuda_parameter_attributes_at_index(index)
        if index >= len(self.tokens):
            return False

        token_type = self.tokens[index][0]
        if token_type == "RPAREN":
            return True
        if (
            token_type == "VOID"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1][0] == "RPAREN"
        ):
            return True
        if self.is_ellipsis_at_index(index):
            return True
        if self.is_function_pointer_parameter_start_at_index(index):
            return True

        return self.is_plausible_function_parameter_at_index(index)

    def is_plausible_function_parameter_at_index(self, index):
        index = self.skip_cuda_parameter_attributes_at_index(index)
        type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if type_end is None:
            return False

        index = self.skip_cpp_attribute_specifiers_at_index(type_end)
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        if self.is_ellipsis_at_index(index):
            index += 3

        if index >= len(self.tokens):
            return False

        if self.tokens[index][0] in {"COMMA", "RPAREN", "ASSIGN"}:
            return True

        if self.tokens[index][0] in self.NAME_COMPONENT_TOKENS:
            index += 1
            index = self.skip_array_suffix_at_index(index)
            index = self.skip_cpp_attribute_specifiers_at_index(index)
            return index < len(self.tokens) and self.tokens[index][0] in {
                "COMMA",
                "RPAREN",
                "ASSIGN",
            }

        return False

    def skip_cuda_parameter_attributes_at_index(self, index):
        while index < len(self.tokens) and (
            self.tokens[index][0] in self.CUDA_PARAMETER_ATTRIBUTE_TOKENS
            or (
                self.tokens[index][0] == "IDENTIFIER"
                and self.tokens[index][1] in self.CUDA_PARAMETER_ATTRIBUTE_IDENTIFIERS
            )
        ):
            index += 1
        return index

    def is_function_pointer_parameter_start_at_index(self, index):
        type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        return (
            type_end is not None
            and type_end + 2 < len(self.tokens)
            and self.tokens[type_end][0] == "LPAREN"
            and self.tokens[type_end + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[type_end + 2][0] in self.NAME_COMPONENT_TOKENS
        )

    def is_implicit_int_type_at_index(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index][0]
        if token_type == "RPAREN":
            return True

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
            "RPAREN",
        }

    def is_global_qualified_name_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
        )

    def is_qualified_type_member_at_index(self, index):
        if index + 1 >= len(self.tokens) or self.tokens[index][0] != "SCOPE":
            return False
        if self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS:
            return True
        return (
            index + 2 < len(self.tokens)
            and self.tokens[index + 1][0] == "TEMPLATE"
            and self.tokens[index + 2][0] in self.NAME_COMPONENT_TOKENS
        )

    def skip_qualified_type_member_at_index(self, index):
        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "TEMPLATE":
            index += 1
        return index + 1

    def skip_composite_scalar_type_suffix_at_index(self, index, type_token):
        if type_token == "LONG":
            saw_long_long = False
            if index < len(self.tokens) and self.tokens[index][0] == "LONG":
                index += 1
                saw_long_long = True

            if (
                saw_long_long
                and index < len(self.tokens)
                and self.tokens[index][0] in {"SIGNED", "UNSIGNED"}
            ):
                index += 1

            if index < len(self.tokens) and self.tokens[index][0] == "INT":
                return index + 1
            if (
                not saw_long_long
                and index < len(self.tokens)
                and self.tokens[index][0] == "DOUBLE"
            ):
                return index + 1
        elif (
            type_token == "SHORT"
            and index < len(self.tokens)
            and self.tokens[index][0] == "INT"
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

    def skip_template_at_index(self, index, allow_assignment=False):
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
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    return index + 1
                if depth < 0:
                    return None
            elif token_type in {"SEMICOLON", "EOF"} or (
                token_type == "ASSIGN" and not allow_assignment
            ):
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

    def function_name_end_index_at(self, index):
        if index >= len(self.tokens) or not self.is_function_name_token(
            self.tokens[index]
        ):
            return None

        token_type, token_value = self.tokens[index]
        if token_type == "IDENTIFIER" and token_value == "operator":
            return self.operator_function_name_end_index_at(index)

        return index + 1

    def operator_function_name_end_index_at(self, index):
        index += 1
        if index >= len(self.tokens):
            return None

        token_type = self.tokens[index][0]
        if token_type == "LBRACKET":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "RBRACKET"
                else None
            )
        if token_type == "LPAREN":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] == "RPAREN"
                else None
            )
        if token_type in self.OVERLOADABLE_OPERATOR_TOKENS:
            return index + 1

        return None

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

        if self.current_token == ("IDENTIFIER", "operator"):
            return self.consume_operator_function_name()

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def consume_operator_function_name(self):
        name = self.eat("IDENTIFIER")[1]
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            return f"{name}[]"
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            self.eat("RPAREN")
            return f"{name}()"
        if self.current_token[0] not in self.OVERLOADABLE_OPERATOR_TOKENS:
            raise SyntaxError(
                f"Expected overloaded operator, got {self.current_token[0]}"
            )

        operator_value = self.current_token[1]
        self.eat(self.current_token[0])
        return f"{name}{operator_value}"

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
            elif self.is_cpp_attribute_specifier_start():
                self.skip_cpp_attribute_specifiers()
            elif self.current_token[0] == "TEMPLATE":
                self.parse_template_declaration()
            elif self.is_namespace_alias_start():
                self.parse_namespace_alias()
            elif self.is_type_alias_start():
                aliases = self.parse_type_alias()
                if isinstance(aliases, list):
                    items.extend(aliases)
                elif aliases is not None:
                    items.append(aliases)
            elif self.is_struct_or_class_declaration_start():
                items.append(self.parse_struct())
            elif self.current_token[0] == "ENUM":
                items.append(self.parse_enum())
            elif self.is_out_of_class_member_definition_start():
                self.skip_out_of_class_member_definition()
            elif self.is_function_pointer_declaration_start():
                self.skip_out_of_class_member_definition()
            elif self.peek_function():
                func = self.parse_function()
                if func is not None:
                    items.append(func)
            elif self.current_token[0] in self.CUDA_FUNCTION_ENTRY_TOKENS:
                if self.current_token[0] == "DEVICE" and self.peek_variable():
                    self.append_parsed_global_variable(
                        items, self.parse_global_variable()
                    )
                elif self.is_function_pointer_declaration_start():
                    self.skip_out_of_class_member_definition()
                else:
                    func = self.parse_function()
                    if func is not None:
                        items.append(func)
            elif (
                self.current_token[0] in ["CONSTANT", "SHARED"] or self.peek_variable()
            ):
                self.append_parsed_global_variable(items, self.parse_global_variable())
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
        self.parse_cuda_declaration_attributes()
        if self.current_token[0] == "STRUCT":
            return self.parse_typedef_struct_alias()
        if self.current_token[0] == "ENUM":
            return self.parse_typedef_enum_alias()

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

        if self.current_token[0] == "LPAREN":
            return self.parse_function_pointer_type_alias_declarator(alias_type)

        name = self.parse_name_component()
        if self.current_token[0] == "LPAREN":
            return self.parse_function_type_alias_declarator(alias_type, name)

        alias_type += self.parse_array_suffix()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_function_type_alias_declarator(self, return_type, name):
        params = self.parse_parameters()
        alias_type = f"{return_type} ()".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def parse_function_pointer_type_alias_declarator(self, return_type):
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        name = self.eat("IDENTIFIER")[1]
        self.eat("RPAREN")
        params = self.parse_parameters()

        alias_type = f"{return_type} ({pointer_prefix})".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def parse_typedef_struct_alias(self):
        self.eat("STRUCT")
        attributes = self.parse_cuda_struct_attribute_prefix()

        tag_name = None
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            tag_name = self.parse_name_component()

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

    def parse_typedef_enum_alias(self):
        self.eat("ENUM")
        is_scoped = False

        if self.current_token[0] in {"CLASS", "STRUCT"}:
            is_scoped = True
            self.eat(self.current_token[0])

        tag_name = None
        if self.current_token[0] == "IDENTIFIER":
            tag_name = self.eat("IDENTIFIER")[1]

        underlying_type = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            underlying_type = self.parse_type()

        if self.current_token[0] != "LBRACE":
            alias = self.parse_type_alias_declarator(
                f"enum {tag_name}", allow_prefix=True
            )
            self.eat("SEMICOLON")
            return alias

        self.eat("LBRACE")
        members = self.parse_enum_members()
        self.eat("RBRACE")

        alias_name = tag_name
        if self.current_token[0] == "IDENTIFIER":
            alias_name = self.eat("IDENTIFIER")[1]
        if not alias_name:
            raise SyntaxError("Expected typedef enum alias name")

        self.eat("SEMICOLON")
        for name in {tag_name, alias_name}:
            if name:
                self.type_aliases.add(name)
                self.struct_names.add(name)

        enum_node = EnumNode(alias_name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        enum_node.tag_name = tag_name
        return enum_node

    def parse_alignment_attributes(self):
        attributes = []
        while self.current_token[0] == "ALIGNAS":
            name = self.eat("ALIGNAS")[1]
            self.eat("LPAREN")
            args = self.collect_balanced_tokens_until("RPAREN")
            self.eat("RPAREN")
            attributes.append(f"{name}({args})")
        return attributes

    def skip_gnu_attributes(self):
        while (
            self.current_token == ("IDENTIFIER", "__attribute__")
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        ):
            self.eat("IDENTIFIER")
            self.skip_balanced_parentheses()

    def parse_cuda_declaration_attributes(self):
        while self.is_cuda_declaration_attribute_identifier():
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()

    def parse_cuda_struct_attribute_prefix(self):
        attributes = []
        while True:
            before_index = self.current_index
            self.parse_cuda_declaration_attributes()
            attributes.extend(self.parse_alignment_attributes())
            if self.current_index == before_index:
                return attributes

    def is_cuda_declaration_attribute_identifier_at_index(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] in self.CUDA_DECLARATION_ATTRIBUTE_IDENTIFIERS
        )

    def is_cuda_declaration_attribute_identifier(self):
        return self.is_cuda_declaration_attribute_identifier_at_index(
            self.current_index
        )

    def skip_cuda_declaration_attributes_at_index(self, index):
        while self.is_cuda_declaration_attribute_identifier_at_index(index):
            index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
                index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")
        return index

    def skip_cuda_struct_attribute_prefix_at_index(self, index):
        while True:
            before_index = index
            index = self.skip_cuda_declaration_attributes_at_index(index)
            index = self.skip_alignment_attributes_at_index(index)
            if index == before_index:
                return index

    def is_cpp_attribute_specifier_start(self):
        return self.is_cpp_attribute_specifier_start_at_index(self.current_index)

    def is_cpp_attribute_specifier_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "LBRACKET"
            and self.tokens[index + 1][0] == "LBRACKET"
        )

    def skip_cpp_attribute_specifiers(self):
        while self.is_cpp_attribute_specifier_start():
            self.eat("LBRACKET")
            self.eat("LBRACKET")
            depth = 1

            while self.current_token[0] != "EOF" and depth > 0:
                if self.is_cpp_attribute_specifier_start():
                    self.eat("LBRACKET")
                    self.eat("LBRACKET")
                    depth += 1
                    continue
                if (
                    self.current_index + 1 < len(self.tokens)
                    and self.current_token[0] == "RBRACKET"
                    and self.tokens[self.current_index + 1][0] == "RBRACKET"
                ):
                    self.eat("RBRACKET")
                    self.eat("RBRACKET")
                    depth -= 1
                    continue
                self.eat(self.current_token[0])

    def skip_cpp_attribute_specifiers_at_index(self, index):
        while self.is_cpp_attribute_specifier_start_at_index(index):
            index += 2
            depth = 1

            while index < len(self.tokens) and depth > 0:
                if self.is_cpp_attribute_specifier_start_at_index(index):
                    index += 2
                    depth += 1
                    continue
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index][0] == "RBRACKET"
                    and self.tokens[index + 1][0] == "RBRACKET"
                ):
                    index += 2
                    depth -= 1
                    continue
                index += 1

        return index

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

    def collect_balanced_raw_tokens_until(self, terminator):
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

            tokens.append(self.current_token)
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

        return tokens

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
        if (
            self.is_decltype_alias_type_start()
            or self.is_dependent_decltype_alias_type_start()
        ):
            alias_type = self.parse_alias_type_expression_until_semicolon()
            self.eat("SEMICOLON")
            self.type_aliases.add(name)
            return TypeAliasNode(alias_type, name)

        alias_type = self.parse_type()
        if self.is_unnamed_function_pointer_alias_declarator():
            return self.parse_using_function_pointer_alias_declarator(name, alias_type)

        self.eat("SEMICOLON")
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def is_decltype_alias_type_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "decltype"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        )

    def is_dependent_decltype_alias_type_start(self):
        return (
            self.current_token[0] == "TYPENAME"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1] == ("IDENTIFIER", "decltype")
            and self.tokens[self.current_index + 2][0] == "LPAREN"
        )

    def parse_alias_type_expression_until_semicolon(self):
        parts = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        template_depth = 0

        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
                and template_depth == 0
            ):
                break

            parts.append(token_value)
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
            elif token_type == "LESS_THAN":
                template_depth += 1
            elif token_type == "GREATER_THAN" and template_depth > 0:
                template_depth -= 1
            elif token_type == "SHIFT_RIGHT" and template_depth > 0:
                template_depth = max(0, template_depth - 2)
            self.eat(token_type)

        return self.format_template_parts(parts)

    def is_unnamed_function_pointer_alias_declarator(self):
        return (
            self.current_index + 2 < len(self.tokens)
            and self.current_token[0] == "LPAREN"
            and self.tokens[self.current_index + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[self.current_index + 2][0] == "RPAREN"
        )

    def parse_using_function_pointer_alias_declarator(self, name, return_type):
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        self.eat("RPAREN")
        params = self.parse_parameters() if self.current_token[0] == "LPAREN" else []
        self.eat("SEMICOLON")

        alias_type = f"{return_type} ({pointer_prefix})".strip()
        alias = TypeAliasNode(alias_type, name)
        alias.params = params
        self.type_aliases.add(name)
        return alias

    def is_namespace_alias_start(self):
        return (
            self.current_token[0] == "NAMESPACE"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "IDENTIFIER"
            and self.tokens[self.current_index + 2][0] == "ASSIGN"
        )

    def parse_namespace_alias(self):
        self.eat("NAMESPACE")
        alias = self.eat("IDENTIFIER")[1]
        self.eat("ASSIGN")
        target_parts = []
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            token_type, token_value = self.current_token
            target_parts.append(token_value)
            self.eat(token_type)

        target = "".join(target_parts).strip()
        if alias and target:
            self.namespace_aliases[alias] = target
        return None

    def is_struct_or_class_declaration_start(self):
        if self.current_token[0] not in {"CLASS", "STRUCT", "UNION"}:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(self.current_index + 1)
        if (
            index >= len(self.tokens)
            or self.tokens[index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return False

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            index = self.skip_template_at_index(index)
            if index is None:
                return False

        return index < len(self.tokens) and self.tokens[index][0] in {
            "COLON",
            "LBRACE",
            "SEMICOLON",
        }

    def is_shared_aggregate_declaration_start(self):
        index = self.current_index
        index, _ = self.skip_aggregate_declaration_prefixes_at_index(index)

        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_gnu_attributes_at_index(index)
        if index >= len(self.tokens) or self.tokens[index][0] not in {
            "STRUCT",
            "UNION",
        }:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(index + 1)
        if (
            index < len(self.tokens)
            and self.tokens[index][0] in self.NAME_COMPONENT_TOKENS
        ):
            index += 1
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False

        return index < len(self.tokens) and self.tokens[index][0] == "LBRACE"

    def skip_aggregate_declaration_prefixes_at_index(self, index):
        qualifiers = []
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.DECLARATION_PREFIX_TOKENS or token_type in {
                "CONST",
                "VOLATILE",
            }:
                qualifiers.append(token_value)
                index += 1
                continue
            break
        return index, qualifiers

    def parse_aggregate_declaration_prefixes(self):
        qualifiers = []
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in self.DECLARATION_PREFIX_TOKENS or token_type in {
                "CONST",
                "VOLATILE",
            }:
                qualifiers.append(token_value)
                self.eat(token_type)
                continue
            break
        return qualifiers

    def parse_shared_aggregate_declaration(self):
        qualifiers = self.parse_aggregate_declaration_prefixes()
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()

        self.eat(self.current_token[0])
        attributes = self.parse_cuda_struct_attribute_prefix()

        tag_name = None
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            tag_name = self.parse_name_component()
            if self.current_token[0] == "LESS_THAN":
                tag_name += self.parse_template_suffix()

        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        declarators = []
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue
            name = self.parse_name_component()
            array_suffix = self.parse_array_suffix()
            declarators.append((name, array_suffix))
            if self.current_token[0] == "ASSIGN":
                self.eat("ASSIGN")
                self.parse_expression()
            elif self.current_token[0] == "LBRACE":
                self.parse_initializer_list()
            if self.current_token[0] != "COMMA":
                break

        self.eat("SEMICOLON")

        if not tag_name:
            first_name = declarators[0][0] if declarators else "union"
            prefix = "__anonymous_shared" if "__shared__" in qualifiers else "anonymous"
            tag_name = f"{prefix}_{first_name}_layout"

        self.struct_names.add(tag_name)
        nodes = [StructNode(tag_name, members, attributes=attributes)]
        for name, array_suffix in declarators:
            vtype = f"{tag_name}{array_suffix}"
            if "__shared__" in qualifiers:
                nodes.append(self.create_shared_memory_node(vtype, name, qualifiers))
            else:
                nodes.append(
                    VariableNode(
                        vtype,
                        name,
                        qualifiers=qualifiers,
                    )
                )
        return nodes if len(nodes) > 1 else nodes[0]

    def skip_until_semicolon(self):
        while self.current_token[0] not in {"SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def parse_struct(self):
        self.eat(self.current_token[0])
        attributes = self.parse_cuda_struct_attribute_prefix()
        name = self.parse_name_component()
        if self.current_token[0] == "LESS_THAN":
            name += self.parse_template_suffix()
        self.struct_names.add(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return StructNode(name, [], attributes=attributes)
        if self.current_token[0] == "COLON":
            self.skip_struct_inheritance_clause()
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members, attributes=attributes)

    def skip_struct_inheritance_clause(self):
        self.eat("COLON")
        while self.current_token[0] not in {"LBRACE", "EOF"}:
            self.eat(self.current_token[0])

    def is_out_of_class_member_definition_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_struct_method_specifiers_at_index(index)

        if self.is_scoped_member_name_at_index(index):
            return True

        return_type_end = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if return_type_end is None:
            return False

        return_type_end = self.skip_return_pointer_suffix_at_index(return_type_end)
        return_type_end = self.skip_struct_method_specifiers_at_index(return_type_end)
        return self.is_scoped_member_name_at_index(return_type_end)

    def is_scoped_member_name_at_index(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return False

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            index = self.skip_template_at_index(index)
            if index is None:
                return False

        saw_scope = False
        while index < len(self.tokens) and self.tokens[index][0] == "SCOPE":
            member_end = self.scoped_member_name_component_end_index_at(index + 1)
            if member_end is None:
                return False

            saw_scope = True
            index = member_end
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False

        return (
            saw_scope and index < len(self.tokens) and self.tokens[index][0] == "LPAREN"
        )

    def scoped_member_name_component_end_index_at(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "TEMPLATE":
            index += 1

        if index >= len(self.tokens):
            return None

        token_type, token_value = self.tokens[index]
        if token_type == "IDENTIFIER" and token_value == "operator":
            return self.operator_function_name_end_index_at(index)

        if token_type == "BITWISE_NOT":
            return (
                index + 2
                if index + 1 < len(self.tokens)
                and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
                else None
            )

        if token_type in self.NAME_COMPONENT_TOKENS:
            return index + 1

        return None

    def skip_out_of_class_member_definition(self):
        while self.current_token[0] not in {"LPAREN", "SEMICOLON", "EOF"}:
            self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()

        while self.current_token[0] not in {"LBRACE", "SEMICOLON", "EOF"}:
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            elif self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()
            else:
                self.eat(self.current_token[0])

        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

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

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            enum_node = EnumNode(name, [])
            enum_node.underlying_type = underlying_type
            enum_node.is_scoped = is_scoped
            return enum_node

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
            if self.is_friend_type_declaration_start():
                self.skip_until_semicolon()
                continue
            if self.is_identifier_value("friend"):
                if self.is_struct_method_start():
                    self.skip_struct_method()
                else:
                    self.skip_until_semicolon()
                continue
            if self.is_templated_type_alias_start():
                self.parse_template_declaration()
                self.parse_type_alias()
                continue
            if self.is_templated_struct_or_class_declaration_start():
                self.parse_template_declaration()
                self.parse_struct()
                continue
            if self.is_anonymous_aggregate_member_start():
                members.extend(self.parse_anonymous_aggregate_member())
                continue
            if self.is_struct_or_class_declaration_start():
                self.parse_struct()
                continue
            if self.current_token[0] == "ENUM":
                self.parse_enum()
                continue
            if self.is_struct_method_start():
                self.skip_struct_method()
                continue
            if self.is_type_alias_start():
                self.parse_type_alias()
                continue

            member_declarations = self.parse_variable_declaration_list()
            members.extend(member_declarations)
            self.eat("SEMICOLON")
        return members

    def is_templated_type_alias_start(self):
        if self.current_token[0] != "TEMPLATE":
            return False
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        return index < len(self.tokens) and self.tokens[index] == (
            "IDENTIFIER",
            "using",
        )

    def is_templated_struct_or_class_declaration_start(self):
        if self.current_token[0] != "TEMPLATE":
            return False
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        return index < len(self.tokens) and self.tokens[index][0] in {"STRUCT", "CLASS"}

    def is_friend_type_declaration_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        while index < len(self.tokens):
            token_type, token_value = self.tokens[index]
            if token_type in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if self.is_function_identifier_specifier_at_index(index):
                index += 1
                continue
            if token_value == "friend":
                break
            return False

        return (
            index + 1 < len(self.tokens)
            and self.tokens[index] == ("IDENTIFIER", "friend")
            and self.tokens[index + 1][0] in {"CLASS", "STRUCT"}
        )

    def is_struct_access_label(self):
        return (
            self.current_token[0] in {"PUBLIC", "PRIVATE", "PROTECTED"}
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "COLON"
        )

    def is_struct_method_start(self):
        index = self.current_index
        index = self.skip_optional_template_declaration_at_index(index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)
        index = self.skip_struct_method_specifiers_at_index(index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)

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

        name_index = self.skip_type_at_index(
            index, allow_unknown_identifier_declarator=True
        )
        if name_index is None or name_index >= len(self.tokens):
            return False
        name_index = self.skip_return_pointer_suffix_at_index(name_index)

        while name_index < len(self.tokens) and (
            self.tokens[name_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.tokens[name_index][0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier_at_index(name_index)
        ):
            if self.tokens[name_index][0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                name_index = self.skip_function_attribute_at_index(name_index)
            else:
                name_index += 1

        token_type, token_value = self.tokens[name_index]
        if token_value == "operator":
            return True
        qualified_name_end = self.qualified_function_name_end_index_at(name_index)
        if qualified_name_end is not None:
            return (
                qualified_name_end < len(self.tokens)
                and self.tokens[qualified_name_end][0] == "LPAREN"
            )
        return (
            self.is_function_name_token(self.tokens[name_index])
            and name_index + 1 < len(self.tokens)
            and self.tokens[name_index + 1][0] == "LPAREN"
        )

    def qualified_function_name_end_index_at(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "SCOPE":
            index += 1

        if (
            index >= len(self.tokens)
            or self.tokens[index][0] not in self.NAME_COMPONENT_TOKENS
        ):
            return None

        index += 1
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "SCOPE"
            and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
        ):
            index += 2
        return index

    def is_macro_block_invocation_start(self):
        return (
            self.is_macro_block_invocation_at_index(self.current_index)
            or self.is_bare_macro_before_block_invocation()
            or self.is_object_like_macro_before_block_invocation()
        )

    def is_macro_block_invocation_at_index(self, index):
        if (
            index + 1 >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or not self.is_macro_like_identifier(self.tokens[index][1])
            or self.tokens[index + 1][0] != "LPAREN"
        ):
            return False

        block_index = self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")
        return (
            block_index < len(self.tokens) and self.tokens[block_index][0] == "LBRACE"
        )

    def is_bare_macro_before_block_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.is_macro_like_identifier(self.current_token[1])
            and self.current_index + 1 < len(self.tokens)
            and self.is_macro_block_invocation_at_index(self.current_index + 1)
        )

    def is_object_like_macro_before_block_invocation(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.is_macro_like_identifier(self.current_token[1])
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LBRACE"
        )

    def is_macro_like_identifier(self, name):
        return any(character.isupper() for character in name) and all(
            character.isupper() or character.isdigit() or character == "_"
            for character in name
        )

    def is_statement_macro_invocation_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1]
            in self.CUDA_STATEMENT_MACRO_INVOCATION_IDENTIFIERS
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "LPAREN"
        )

    def skip_statement_macro_invocation(self):
        self.eat("IDENTIFIER")
        self.skip_balanced_parentheses()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_top_level_macro_statement_start(self):
        end_index = self.top_level_macro_statement_end_index(self.current_index)
        return end_index is not None

    def is_concept_declaration_start(self):
        index = self.skip_optional_template_declaration_at_index(self.current_index)
        index = self.skip_cpp_attribute_specifiers_at_index(index)

        if index + 2 >= len(self.tokens):
            return False

        token_type, token_value = self.tokens[index]
        if token_type != "IDENTIFIER":
            return False

        if token_value == "concept":
            name_index = index + 1
        elif token_value == "_CCCL_CONCEPT" or token_value.endswith("_CONCEPT"):
            name_index = index + 1
        else:
            return False

        return (
            name_index + 1 < len(self.tokens)
            and self.tokens[name_index][0] in self.NAME_COMPONENT_TOKENS
            and self.tokens[name_index + 1][0] == "ASSIGN"
        )

    def skip_concept_declaration(self):
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if (
                token_type == "SEMICOLON"
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                self.eat("SEMICOLON")
                break

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

    def top_level_macro_statement_end_index(self, index):
        if (
            index >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or not self.is_macro_like_identifier(self.tokens[index][1])
        ):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
            index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")

        if index < len(self.tokens) and self.tokens[index][0] == "SEMICOLON":
            return index + 1
        if index >= len(self.tokens):
            return index

        follow_type, follow_value = self.tokens[index]
        if follow_type == "IDENTIFIER" and self.is_macro_like_identifier(follow_value):
            return index
        if follow_type in self.TOP_LEVEL_MACRO_STATEMENT_FOLLOW_TOKENS:
            return index
        return None

    def skip_top_level_macro_statement(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def is_statement_directive_marker_start(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        marker_end = self.statement_directive_marker_end_index(self.current_index)
        return (
            marker_end is not None
            and marker_end < len(self.tokens)
            and self.tokens[marker_end][0] in self.STATEMENT_DIRECTIVE_FOLLOW_TOKENS
        )

    def statement_directive_marker_end_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "IDENTIFIER":
            return None

        name = self.tokens[index][1]
        if name == "_Pragma":
            if index + 1 >= len(self.tokens) or self.tokens[index + 1][0] != "LPAREN":
                return None
            return self.skip_balanced_tokens_at_index(index + 1, "LPAREN", "RPAREN")

        if not self.is_macro_like_identifier(name):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LPAREN":
            index = self.skip_balanced_tokens_at_index(index, "LPAREN", "RPAREN")
        return index

    def skip_statement_directive_marker(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()

    def skip_macro_block_invocation(self):
        if self.is_bare_macro_before_block_invocation():
            self.eat("IDENTIFIER")

        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

    def skip_return_pointer_suffix_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)
        return index

    def skip_optional_template_declaration_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "TEMPLATE":
            return index

        index += 1
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index, allow_assignment=True)
            if skipped is not None:
                return skipped
        return index

    def skip_struct_method_specifiers_at_index(self, index):
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type in self.FUNCTION_ATTRIBUTE_TOKENS:
                index = self.skip_function_attribute_at_index(index)
                continue
            if (
                token_type in self.FUNCTION_SPECIFIER_TOKENS
                or token_type == "VIRTUAL"
                or (
                    token_type == "IDENTIFIER"
                    and self.tokens[index][1]
                    in self.STRUCT_METHOD_IDENTIFIER_SPECIFIERS
                )
                or self.is_function_identifier_specifier_at_index(index)
            ):
                if (
                    token_type == "EXTERN"
                    and index + 1 < len(self.tokens)
                    and self.tokens[index + 1][0] == "STRING"
                ):
                    index += 2
                    continue
                if (
                    token_type == "IDENTIFIER"
                    and self.tokens[index][1] == "explicit"
                    and index + 1 < len(self.tokens)
                    and self.tokens[index + 1][0] == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_index(
                        index + 1, "LPAREN", "RPAREN"
                    )
                    continue
                index += 1
                continue
            break
        return index

    def is_anonymous_aggregate_member_start(self):
        if self.current_token[0] not in {"STRUCT", "UNION"}:
            return False

        index = self.skip_cuda_struct_attribute_prefix_at_index(self.current_index + 1)
        return index < len(self.tokens) and self.tokens[index][0] == "LBRACE"

    def parse_anonymous_aggregate_member(self):
        aggregate_type = self.current_token[1]
        self.eat(self.current_token[0])
        attributes = self.parse_cuda_struct_attribute_prefix()
        self.eat("LBRACE")
        members = self.parse_struct_members()
        self.eat("RBRACE")

        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            name = self.parse_name_component()
            vtype = aggregate_type + self.parse_array_suffix()
            self.eat("SEMICOLON")
            return [VariableNode(vtype, name, attributes=attributes)]

        self.eat("SEMICOLON")
        return members

    def skip_struct_method(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            if self.current_token[0] == "LESS_THAN":
                self.parse_template_suffix()
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()

        while self.current_token[0] not in {"SEMICOLON", "RBRACE", "EOF"}:
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            elif self.current_token[0] == "LBRACKET":
                self.skip_balanced_brackets()
            elif self.current_token[0] == "LBRACE":
                self.skip_balanced_brace_block()
                if self.current_token[0] not in {"COMMA", "LBRACE"}:
                    break
            else:
                self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
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

    def skip_balanced_parentheses(self):
        self.eat("LPAREN")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
            self.eat(token_type)

    def skip_balanced_brackets(self):
        self.eat("LBRACKET")
        depth = 1
        while self.current_token[0] != "EOF" and depth > 0:
            token_type = self.current_token[0]
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    self.eat("RBRACKET")
                    break
            self.eat(token_type)

    def parse_function(self):
        qualifiers = []
        attributes = []
        linkage = None

        while (
            self.current_token[0] in self.FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier()
        ):
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
        while (
            self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS
            or self.current_token[0] in self.POST_RETURN_FUNCTION_SPECIFIER_TOKENS
            or self.is_function_identifier_specifier()
        ):
            if self.current_token[0] in self.FUNCTION_ATTRIBUTE_TOKENS:
                attributes.append(self.parse_function_attribute())
                continue

            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        name = self.consume_function_name()
        if self.current_token[0] == "LESS_THAN":
            name += self.parse_template_suffix()
        self.user_function_names.add(name)
        params = self.parse_parameters()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers()
        body = None
        if self.is_identifier_value("try"):
            body = self.parse_try_statement()
        elif self.current_token[0] == "LBRACE":
            body = self.parse_block()
        elif self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        else:
            self.eat("LBRACE")

        if "__global__" in qualifiers or "__tile_global__" in qualifiers:
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

    def parse_trailing_return_type(self, return_type):
        if self.current_token[0] != "ARROW":
            return return_type

        self.eat("ARROW")
        return self.parse_type()

    def skip_post_function_qualifiers(self):
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if token_type in {"CONST", "VOLATILE", *self.TYPE_REFERENCE_TOKENS}:
                self.eat(token_type)
                continue
            if token_type == "IDENTIFIER" and token_value in {
                "noexcept",
                "override",
                "final",
            }:
                self.eat("IDENTIFIER")
                if token_value == "noexcept" and self.current_token[0] == "LPAREN":
                    self.skip_balanced_parentheses()
                continue
            if token_type == "IDENTIFIER" and token_value == "requires":
                self.skip_requires_clause()
                continue
            break

    def skip_requires_clause(self):
        self.eat("IDENTIFIER")
        paren_depth = 0
        bracket_depth = 0
        template_depth = 0
        while self.current_token[0] != "EOF":
            token_type, token_value = self.current_token
            if (
                paren_depth == 0
                and bracket_depth == 0
                and template_depth == 0
                and (
                    token_type in {"LBRACE", "SEMICOLON"}
                    or (token_type == "IDENTIFIER" and token_value == "try")
                )
            ):
                break

            if token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            elif token_type == "LESS_THAN":
                template_depth += 1
            elif token_type == "GREATER_THAN" and template_depth > 0:
                template_depth -= 1
            elif token_type == "SHIFT_RIGHT" and template_depth > 0:
                template_depth = max(0, template_depth - 2)

            self.eat(token_type)

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
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        self.skip_cuda_parameter_attributes()

        if self.is_ellipsis_at_current_index():
            self.consume_ellipsis()
            return VariableNode("...", "")

        param_type = self.parse_type()
        param_type = self.parse_declarator_prefix(param_type)
        if self.is_function_pointer_parameter_declarator():
            return self.parse_function_pointer_parameter(param_type)

        if self.is_ellipsis_at_current_index():
            self.consume_ellipsis()
            param_type = f"{param_type} ...".strip()

        if self.current_token[0] in {"COMMA", "RPAREN"}:
            self.skip_parameter_default()
            return VariableNode(param_type, "")

        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        self.skip_cuda_parameter_attributes()

        param_name = self.parse_name_component()
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
        param_type += self.parse_array_suffix()
        self.skip_parameter_default()
        return VariableNode(param_type, param_name)

    def skip_cuda_parameter_attributes(self):
        while self.current_token[0] in self.CUDA_PARAMETER_ATTRIBUTE_TOKENS or (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] in self.CUDA_PARAMETER_ATTRIBUTE_IDENTIFIERS
        ):
            self.eat(self.current_token[0])

    def is_ellipsis_at_current_index(self):
        return self.is_ellipsis_at_index(self.current_index)

    def is_ellipsis_at_index(self, index):
        return (
            index + 2 < len(self.tokens)
            and self.tokens[index][0] == "DOT"
            and self.tokens[index + 1][0] == "DOT"
            and self.tokens[index + 2][0] == "DOT"
        )

    def consume_ellipsis(self):
        self.eat("DOT")
        self.eat("DOT")
        self.eat("DOT")

    def is_function_pointer_parameter_declarator(self):
        return (
            self.current_index + 2 < len(self.tokens)
            and self.current_token[0] == "LPAREN"
            and self.tokens[self.current_index + 1][0]
            in {"MULTIPLY", *self.TYPE_REFERENCE_TOKENS}
            and self.tokens[self.current_index + 2][0] == "IDENTIFIER"
        )

    def parse_function_pointer_parameter(self, return_type):
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        name = self.eat("IDENTIFIER")[1]
        self.eat("RPAREN")

        if self.current_token[0] == "LPAREN":
            self.parse_parameters()
        array_suffix = self.parse_array_suffix()

        if array_suffix:
            param_type = f"{return_type} {pointer_prefix} {array_suffix}".strip()
        else:
            param_type = f"{return_type} ({pointer_prefix})".strip()
        self.skip_parameter_default()
        return VariableNode(param_type, name)

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
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
            type_parts.append(self.parse_elaborated_type_name())
        elif self.is_global_qualified_name_start_at_index(self.current_index):
            type_parts.append(self.parse_type_name())
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
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
            type_parts.append(self.parse_elaborated_type_name())
        elif self.is_global_qualified_name_start_at_index(self.current_index):
            type_parts.append(self.parse_type_name())
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

    def is_decltype_type_start(self):
        return self.is_decltype_type_start_at_index(self.current_index)

    def is_decltype_type_start_at_index(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index][0] == "IDENTIFIER"
            and self.tokens[index][1] == "decltype"
            and self.tokens[index + 1][0] == "LPAREN"
        )

    def skip_decltype_type_at_index(self, index):
        if not self.is_decltype_type_start_at_index(index):
            return None

        index += 1
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
                return None
            index += 1

        return None

    def parse_decltype_type_name(self):
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        parts = []
        depth = 1

        while depth > 0:
            token_type, token_value = self.current_token
            if token_type == "EOF":
                raise SyntaxError("Unterminated decltype type")
            if token_type == "LPAREN":
                depth += 1
                parts.append(token_value)
                self.eat("LPAREN")
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.eat("RPAREN")
                    break
                parts.append(token_value)
                self.eat("RPAREN")
            else:
                parts.append(token_value)
                self.eat(token_type)

        return f"decltype({self.format_template_parts(parts)})"

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.current_token[0] in self.POSTFIX_TYPE_QUALIFIER_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

    def parse_type_name(self):
        token_type = self.current_token[0]
        if token_type == "SCOPE":
            self.eat("SCOPE")
            type_name = f"::{self.parse_name_component()}"
        else:
            type_name = self.current_token[1]
            self.eat(token_type)

        if token_type == "LONG" and self.current_token[0] == "LONG":
            type_name += f" {self.current_token[1]}"
            self.eat("LONG")
            if self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                type_name += f" {self.current_token[1]}"
                self.eat(self.current_token[0])
            if self.current_token[0] == "INT":
                type_name += f" {self.current_token[1]}"
                self.eat("INT")
        elif (
            token_type in self.COMPOSITE_SCALAR_TYPE_SUFFIX_TOKENS
            and self.current_token[0]
            in self.COMPOSITE_SCALAR_TYPE_SUFFIX_TOKENS[token_type]
        ):
            type_name += f" {self.current_token[1]}"
            self.eat(self.current_token[0])

        while True:
            if self.current_token[0] == "LESS_THAN":
                type_name += self.parse_template_suffix()
                continue
            if self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                if self.current_token[0] == "TEMPLATE":
                    self.eat("TEMPLATE")
                member = self.parse_name_component()
                type_name += f"::{member}"
                continue
            break

        return type_name

    def parse_elaborated_type_name(self):
        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        member = self.parse_name_component()
        return f"{type_name} {member}"

    def parse_global_variable(self):
        declarations = self.parse_variable_declaration_list()
        self.eat("SEMICOLON")
        return declarations if len(declarations) > 1 else declarations[0]

    def append_parsed_global_variable(self, target, variable):
        if isinstance(variable, list):
            target.extend(variable)
        else:
            target.append(variable)

    def parse_variable_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()

        vtype = self.parse_type()
        self.parse_interleaved_declaration_qualifiers(qualifiers)
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()
        name = self.parse_name_component()
        vtype += self.parse_array_suffix()
        self.skip_bitfield_width()

        value = None
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            value = self.parse_expression()
        elif self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()
            value = FunctionCallNode(self.constructor_initializer_name(vtype), args)
        elif self.current_token[0] == "LBRACE":
            value = self.parse_initializer_list()

        var = VariableNode(vtype, name, value, qualifiers)

        if "__shared__" in qualifiers:
            return self.create_shared_memory_node(vtype, name, qualifiers)
        elif "__constant__" in qualifiers:
            return ConstantMemoryNode(vtype, name, value, qualifiers)
        else:
            return var

    def parse_variable_declaration_list(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()
        self.skip_gnu_attributes()

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

    def parse_declaration_prefixes(self):
        qualifiers = []
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            if token_type in self.DECLARATION_PREFIX_TOKENS:
                qualifiers.append(self.current_token[1])
                self.eat(token_type)
                continue
            if (
                token_type == "VOLATILE"
                and self.current_index + 1 < len(self.tokens)
                and self.tokens[self.current_index + 1][0]
                in self.CUDA_STORAGE_QUALIFIER_TOKENS
            ):
                qualifiers.append(self.current_token[1])
                self.eat(token_type)
                continue
            break
        return qualifiers

    def parse_interleaved_declaration_qualifiers(self, qualifiers):
        while self.current_token[0] in self.DECLARATION_PREFIX_TOKENS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        vtype = base_type
        if allow_prefix:
            vtype = self.parse_declarator_prefix(vtype)

        declarator_qualifiers = list(qualifiers)
        self.parse_interleaved_declaration_qualifiers(declarator_qualifiers)
        self.parse_alignment_attributes()
        name = self.parse_name_component()
        vtype += self.parse_array_suffix()
        self.skip_bitfield_width()
        value = self.parse_variable_initializer(vtype)

        if "__shared__" in declarator_qualifiers:
            return self.create_shared_memory_node(vtype, name, declarator_qualifiers)
        if "__constant__" in declarator_qualifiers:
            return ConstantMemoryNode(vtype, name, value, declarator_qualifiers)
        return VariableNode(vtype, name, value, declarator_qualifiers)

    def create_shared_memory_node(self, vtype, name, qualifiers):
        is_extern = "extern" in qualifiers
        is_dynamic = (
            is_extern
            and isinstance(vtype, str)
            and self.has_unsized_array_dimension(vtype)
        )
        return SharedMemoryNode(
            vtype,
            name,
            is_extern=is_extern,
            is_dynamic=is_dynamic,
            qualifiers=qualifiers,
        )

    def has_unsized_array_dimension(self, vtype):
        return "[]" in vtype

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
            return FunctionCallNode(self.constructor_initializer_name(vtype), args)
        if self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        return None

    def skip_bitfield_width(self):
        if self.current_token[0] != "COLON":
            return

        self.eat("COLON")
        if self.current_token[0] not in {"COMMA", "SEMICOLON", "EOF"}:
            self.parse_expression()

    def constructor_initializer_name(self, vtype):
        return " ".join(
            part
            for part in str(vtype).split()
            if part
            not in {
                "const",
                "volatile",
                "__restrict__",
                "__restrict",
                "restrict",
                "&",
                "&&",
            }
        )

    def parse_function_pointer_variable_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()

        return_type = self.parse_type()
        self.eat("LPAREN")
        pointer_prefix = self.parse_declarator_prefix("")
        name = self.parse_name_component()
        self.eat("RPAREN")

        params = []
        if self.current_token[0] == "LPAREN":
            params = self.parse_parameters()

        vtype = f"{return_type} ({pointer_prefix})".strip()
        value = self.parse_variable_initializer(vtype)
        variable = VariableNode(vtype, name, value, qualifiers)
        variable.params = params
        return variable

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
        self.parse_expression_arguments(args, "RPAREN")
        self.eat("RPAREN")
        return args

    def parse_expression_arguments(self, args, terminator):
        while self.current_token[0] != terminator:
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                continue

            args.append(self.parse_expression())

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != terminator:
                break

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
        if self.is_cpp_attribute_specifier_start():
            self.skip_cpp_attribute_specifiers()
            return self.parse_statement()
        if self.is_namespace_alias_start():
            return self.parse_namespace_alias()
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
        elif self.current_token[0] == "GOTO":
            return self.parse_goto_statement()
        elif self.current_token[0] in ["SYNCTHREADS", "SYNCWARP"]:
            return self.parse_sync_statement()
        elif self.current_token[0] == "ASM":
            return self.parse_asm_statement()
        elif self.current_token[0] == "LBRACE":
            return self.parse_block()
        elif self.is_label_statement_start():
            return self.parse_label_statement()
        elif self.is_type_alias_start():
            return self.parse_type_alias()
        elif self.is_identifier_value("try"):
            return self.parse_try_statement()
        elif self.is_identifier_value("throw"):
            return self.parse_throw_statement()
        elif self.is_identifier_value("delete"):
            stmt = self.parse_delete_statement()
            self.eat("SEMICOLON")
            return stmt
        elif self.is_structured_binding_declaration():
            declaration = self.parse_structured_binding_declaration()
            self.eat("SEMICOLON")
            return declaration
        elif self.is_function_pointer_declaration_start():
            declaration = self.parse_function_pointer_variable_declaration()
            self.eat("SEMICOLON")
            return declaration
        elif self.is_shared_aggregate_declaration_start():
            return self.parse_shared_aggregate_declaration()
        elif self.is_struct_or_class_declaration_start():
            return self.parse_struct()
        elif self.is_macro_block_invocation_start():
            self.skip_macro_block_invocation()
            return None
        elif self.is_statement_macro_invocation_start():
            self.skip_statement_macro_invocation()
            return None
        elif self.is_statement_directive_marker_start():
            self.skip_statement_directive_marker()
            return self.parse_statement()
        elif self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            self.eat("SEMICOLON")
            return declarations if len(declarations) > 1 else declarations[0]
        elif self.current_token[0] == "ENUM":
            return self.parse_enum()
        elif self.is_parenthesized_comma_expression_statement():
            if self.is_fold_expression_statement_start():
                return self.parse_fold_expression_statement()
            return self.parse_parenthesized_comma_expression_statement()
        else:
            expr = self.parse_assignment_expression()
            if self.current_token[0] == "COMMA":
                expr = self.parse_comma_expression_statement(expr)
            self.eat("SEMICOLON")
            return expr

    def parse_comma_expression_statement(self, first_expr):
        expressions = [first_expr]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_assignment_expression())
        return expressions

    def is_parenthesized_comma_expression_statement(self):
        if self.current_token[0] != "LPAREN":
            return False

        depth = 0
        saw_comma = False
        index = self.current_index
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    next_index = index + 1
                    return (
                        saw_comma
                        and next_index < len(self.tokens)
                        and self.tokens[next_index][0] == "SEMICOLON"
                    )
            elif token_type == "COMMA" and depth == 1:
                saw_comma = True
            elif token_type == "EOF":
                return False
            index += 1

        return False

    def is_fold_expression_statement_start(self):
        return self.current_token[0] == "LPAREN" and self.is_ellipsis_at_index(
            self.current_index + 1
        )

    def parse_fold_expression_statement(self):
        raw_tokens = self.collect_balanced_raw_tokens_until("SEMICOLON")
        self.eat("SEMICOLON")
        raw_expression = self.format_lambda_raw_tokens(raw_tokens)
        return FunctionCallNode("cuda_fold_expression", [raw_expression])

    def is_parenthesized_fold_expression_start(self):
        if self.current_token[0] != "LPAREN":
            return False

        depth = 0
        index = self.current_index
        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return False
            elif depth == 1 and self.is_ellipsis_at_index(index):
                return True
            elif token_type == "EOF":
                return False
            index += 1
        return False

    def parse_parenthesized_fold_expression(self):
        raw_tokens = []
        depth = 0
        while self.current_token[0] != "EOF":
            token_type = self.current_token[0]
            raw_tokens.append(self.current_token)
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                self.eat("RPAREN")
                if depth == 0:
                    break
                continue
            self.eat(token_type)

        raw_expression = self.format_lambda_raw_tokens(raw_tokens)
        return FunctionCallNode("cuda_fold_expression", [raw_expression])

    def parse_parenthesized_comma_expression_statement(self):
        self.eat("LPAREN")
        first_expr = self.parse_assignment_expression()
        expressions = self.parse_comma_expression_statement(first_expr)
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return expressions

    def parse_try_statement(self):
        self.eat("IDENTIFIER")
        try_body = self.parse_statement()

        while self.is_identifier_value("catch"):
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LPAREN":
                self.skip_balanced_parentheses()
            self.parse_statement()

        return try_body

    def parse_throw_statement(self):
        self.eat("IDENTIFIER")
        args = []
        if self.current_token[0] != "SEMICOLON":
            args.append(self.parse_expression())
        self.eat("SEMICOLON")
        return FunctionCallNode("throw", args)

    def is_identifier_value(self, value):
        return self.current_token[0] == "IDENTIFIER" and self.current_token[1] == value

    def is_structured_binding_declaration(self):
        index = self.current_index
        index = self.skip_declaration_prefixes_at_index(index)
        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_structured_binding_type_at_index(index)

        if index is None:
            return False

        index = self.skip_alignment_attributes_at_index(index)
        index = self.skip_structured_binding_names_at_index(index)
        return (
            index is not None
            and index < len(self.tokens)
            and self.tokens[index][0] in {"ASSIGN", "LPAREN", "LBRACE"}
        )

    def skip_structured_binding_type_at_index(self, index):
        while index < len(self.tokens) and self.tokens[index][0] in {
            "CONST",
            "VOLATILE",
        }:
            index += 1

        if (
            index >= len(self.tokens)
            or self.tokens[index][0] != "IDENTIFIER"
            or self.tokens[index][1] != "auto"
        ):
            return None

        index += 1
        index = self.skip_postfix_type_qualifiers_at_index(index)

        while (
            index < len(self.tokens)
            and self.tokens[index][0] in self.TYPE_REFERENCE_TOKENS
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_index(index)

        return index

    def skip_structured_binding_names_at_index(self, index):
        if index >= len(self.tokens) or self.tokens[index][0] != "LBRACKET":
            return None

        index += 1
        expect_name = True
        saw_name = False

        while index < len(self.tokens):
            token_type = self.tokens[index][0]
            if expect_name:
                if token_type not in self.NAME_COMPONENT_TOKENS:
                    return None
                saw_name = True
                expect_name = False
                index += 1
                continue

            if token_type == "COMMA":
                expect_name = True
                index += 1
                continue
            if token_type == "RBRACKET":
                return index + 1 if saw_name and not expect_name else None
            return None

        return None

    def parse_structured_binding_declaration(self):
        qualifiers = self.parse_declaration_prefixes()
        self.parse_alignment_attributes()

        vtype = self.parse_structured_binding_type()
        self.parse_alignment_attributes()
        name = self.parse_structured_binding_name()
        value = self.parse_variable_initializer(vtype)
        return VariableNode(vtype, name, value, qualifiers)

    def parse_structured_binding_type(self):
        type_parts = []

        while self.current_token[0] in {"CONST", "VOLATILE"}:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        type_parts.append(self.eat("IDENTIFIER")[1])
        self.parse_postfix_type_qualifiers(type_parts)

        while self.current_token[0] in self.TYPE_REFERENCE_TOKENS:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def parse_structured_binding_name(self):
        names = []
        self.eat("LBRACKET")
        while self.current_token[0] != "RBRACKET":
            names.append(self.parse_name_component())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] != "RBRACKET":
                raise SyntaxError(
                    f"Expected COMMA or RBRACKET, got {self.current_token[0]}"
                )
        self.eat("RBRACKET")
        return f"[{', '.join(names)}]"

    def parse_delete_statement(self):
        self.eat("IDENTIFIER")
        is_array = False

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            self.eat("RBRACKET")
            is_array = True

        expression = self.parse_unary_expression()
        return DeleteNode(expression, is_array)

    def is_label_statement_start(self):
        return (
            self.current_token[0] == "IDENTIFIER"
            and self.current_index + 1 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "COLON"
        )

    def parse_label_statement(self):
        self.eat("IDENTIFIER")
        self.eat("COLON")

        if self.current_token[0] in {"RBRACE", "EOF"}:
            return None
        return self.parse_statement()

    def parse_goto_statement(self):
        self.eat("GOTO")

        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            self.eat(self.current_token[0])
        else:
            while self.current_token[0] not in {"SEMICOLON", "EOF"}:
                self.eat(self.current_token[0])

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return None

    def is_variable_declaration(self):
        saved_index = self.current_index

        saved_index = self.skip_declaration_prefixes_at_index(saved_index)
        saved_index = self.skip_alignment_attributes_at_index(saved_index)
        saved_index = self.skip_gnu_attributes_at_index(saved_index)

        saved_index = self.skip_type_at_index(
            saved_index, allow_unknown_identifier_declarator=True
        )

        if saved_index is not None:
            saved_index = self.skip_interleaved_declaration_qualifiers_at_index(
                saved_index
            )
            saved_index = self.skip_alignment_attributes_at_index(saved_index)
            saved_index = self.skip_gnu_attributes_at_index(saved_index)
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] in self.NAME_COMPONENT_TOKENS
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
        is_constexpr = False
        if self.current_token[0] == "CONSTEXPR":
            is_constexpr = True
            self.eat("CONSTEXPR")
        self.eat("LPAREN")

        condition_or_init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list()
                condition_or_init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                condition_or_init = self.parse_expression()

        init = None
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            init = condition_or_init
            condition = self.parse_expression()
        else:
            condition = condition_or_init

        self.eat("RPAREN")

        if_body = self.parse_statement()

        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_statement()

        node = IfNode(condition, if_body, else_body)
        node.is_constexpr = is_constexpr
        if init is not None:
            return [*init, node] if isinstance(init, list) else [init, node]
        return node

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
                init = self.parse_for_update_expression()
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
        index = self.skip_composite_scalar_type_suffix_at_index(index, type_token)
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

    def parse_expression(self, allow_comma=False):
        expr = self.parse_assignment_expression()
        if allow_comma and self.current_token[0] == "COMMA":
            return self.parse_comma_expression(expr)
        return expr

    def parse_comma_expression(self, first_expr):
        expressions = [first_expr]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            expressions.append(self.parse_assignment_expression())
        return FunctionCallNode("cuda_comma_expression", expressions)

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
            op = self.normalized_operator_value()
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.normalized_operator_value()
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()

        while self.current_token[0] == "BITWISE_OR":
            op = self.normalized_operator_value()
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()

        while self.current_token[0] == "BITWISE_XOR":
            op = self.normalized_operator_value()
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()

        while self.current_token[0] == "BITWISE_AND":
            op = self.normalized_operator_value()
            self.eat("BITWISE_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.normalized_operator_value()
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
            op = self.normalized_operator_value()
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
            if self.is_ellipsis_at_current_index():
                self.consume_ellipsis()
                left = UnaryOpNode("post...", left)
            elif self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.parse_member_name()
                left = MemberAccessNode(left, member, False)
            elif self.current_token[0] == "SCOPE":
                self.eat("SCOPE")
                member = self.parse_qualified_name_member()
                left = self.append_qualified_name(left, "::", member)
            elif self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                left = self.append_template_suffix(left)
            elif self.is_arrow_star_lambda_start():
                self.eat("ARROW")
                self.eat("MULTIPLY")
                lambda_node = self.parse_lambda_expression()
                left = FunctionCallNode("operator->*", [left, lambda_node])
            elif self.current_token[0] == "ARROW":
                self.eat("ARROW")
                member = self.parse_member_name()
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
                else:
                    self.parse_expression_arguments(args, "RPAREN")
                self.eat("RPAREN")

                if (
                    isinstance(left, str)
                    and left in self.ATOMIC_FUNCTION_NAMES
                    and left not in self.user_function_names
                ):
                    left = AtomicOperationNode(left, args)
                else:
                    left = self.parse_function_call_node(left, args)
            elif self.current_token[0] == "LBRACE" and self.is_braced_constructor(left):
                initializer = self.parse_initializer_list()
                left = FunctionCallNode(left, initializer.elements)
            elif self.current_token[0] == "KERNEL_LAUNCH_START":
                return self.parse_kernel_launch(left)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(f"post{op}", left)
            else:
                break

        return left

    def is_arrow_star_lambda_start(self):
        return (
            self.current_token[0] == "ARROW"
            and self.current_index + 2 < len(self.tokens)
            and self.tokens[self.current_index + 1][0] == "MULTIPLY"
            and self.tokens[self.current_index + 2][0] == "LBRACKET"
        )

    def parse_member_name(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            member = self.consume_function_name()
            if self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                member += self.parse_template_suffix()
            return member
        if self.current_token[0] in self.NAME_COMPONENT_TOKENS:
            return self.parse_name_component()
        raise SyntaxError(f"Expected IDENTIFIER, got {self.current_token[0]}")

    def parse_qualified_name_member(self):
        if self.current_token[0] == "TEMPLATE":
            self.eat("TEMPLATE")
            member = self.consume_function_name()
            if self.current_token[0] == "LESS_THAN" and self.is_template_suffix():
                member += self.parse_template_suffix()
            return member
        return self.parse_name_component()

    def parse_name_component(self):
        if self.current_token[0] not in self.NAME_COMPONENT_TOKENS:
            raise SyntaxError(f"Expected IDENTIFIER, got {self.current_token[0]}")

        name = self.current_token[1]
        self.eat(self.current_token[0])
        return name

    def is_braced_constructor(self, callee):
        return isinstance(callee, str)

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
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
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
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
                    }
                if depth < 0:
                    return False
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1][0]
                        if index + 1 < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in {
                        "LPAREN",
                        "LBRACE",
                        "LESS_THAN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                        "RBRACKET",
                        "SEMICOLON",
                        "LOGICAL_AND",
                        "LOGICAL_OR",
                        "QUESTION",
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
            elif token_type == "KERNEL_LAUNCH_END":
                self.eat("KERNEL_LAUNCH_END")
                for _ in range(3):
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
            function_name in {"cudaLaunchKernel", "cudaLaunchCooperativeKernel"}
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
            return self.unwrap_cuda_kernel_function_arg(function_arg.expression)
        if isinstance(function_arg, UnaryOpNode) and function_arg.op == "&":
            return function_arg.operand
        return function_arg

    def is_sizeof_type_operand(self):
        type_end = self.skip_type_at_index(self.current_index)
        return (
            type_end is not None
            and type_end < len(self.tokens)
            and self.tokens[type_end][0] == "RPAREN"
        )

    def parse_kernel_launch(self, kernel_name):
        self.eat("KERNEL_LAUNCH_START")

        blocks = self.parse_expression()
        threads = None

        shared_mem = None
        stream = None

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            threads = self.parse_expression()

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
            if self.current_token[0] == "IDENTIFIER" and self.current_token[
                1
            ].startswith("_"):
                value += self.current_token[1]
                self.eat("IDENTIFIER")
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
            if self.is_requires_expression_start():
                return self.parse_requires_expression()
            if self.current_token[1] == "new":
                return self.parse_new_expression()

            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name
        elif self.current_token[0] == "SCOPE":
            self.eat("SCOPE")
            name = self.parse_name_component()
            return f"::{name}"
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
            if self.is_parenthesized_fold_expression_start():
                return self.parse_parenthesized_fold_expression()
            if self.is_cast_expression():
                self.eat("LPAREN")
                target_type = self.parse_type()
                self.eat("RPAREN")
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.eat("LPAREN")
            expr = self.parse_expression(allow_comma=True)
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] == "LBRACE":
            return self.parse_initializer_list()
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def is_requires_expression_start(self):
        if (
            self.current_token[0] != "IDENTIFIER"
            or self.current_token[1] != "requires"
            or self.current_index + 1 >= len(self.tokens)
        ):
            return False

        next_type = self.tokens[self.current_index + 1][0]
        if next_type == "LBRACE":
            return True
        if next_type != "LPAREN":
            return False

        body_index = self.skip_balanced_tokens_at_index(
            self.current_index + 1, "LPAREN", "RPAREN"
        )
        return body_index < len(self.tokens) and self.tokens[body_index][0] == "LBRACE"

    def parse_requires_expression(self):
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            self.skip_balanced_parentheses()
        if self.current_token[0] == "LBRACE":
            self.skip_balanced_brace_block()
        return "true"

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
        index = self.skip_lambda_template_parameters_at_index(index)
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

    def skip_lambda_template_parameters_at_index(self, index):
        if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
            skipped = self.skip_template_at_index(index)
            if skipped is not None:
                return skipped
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
        self.skip_lambda_template_parameters()
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

    def skip_lambda_template_parameters(self):
        if self.current_token[0] == "LESS_THAN":
            self.parse_template_suffix()

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
            self.skip_parameter_default()
            return VariableNode(param_type, param_name)
        except SyntaxError:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]
            raw = self.collect_lambda_parameter_raw()
            return VariableNode("", raw)

    def skip_parameter_default(self):
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

        placement_args = []
        if self.current_token[0] == "LPAREN":
            placement_args = self.parse_argument_list()

        target_type = self.parse_type_without_array_suffix()

        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            size = None
            if self.current_token[0] != "RBRACKET":
                size = self.parse_expression()
            self.eat("RBRACKET")
            node = NewNode(target_type, size=size, is_array=True)
            if placement_args:
                node.placement_args = placement_args
            return node

        args = []
        if self.current_token[0] == "LPAREN":
            args = self.parse_argument_list()

        node = NewNode(target_type, args=args)
        if placement_args:
            node.placement_args = placement_args
        return node

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

            if self.current_token[0] in self.ELABORATED_TYPE_TOKENS:
                self.parse_elaborated_type_name()
            else:
                if self.current_token[0] not in self.TYPE_TOKENS or (
                    self.current_token[0] == "IDENTIFIER"
                    and not self.is_identifier_type_name(self.current_token[1])
                    and not self.is_identifier_cast_target_at_current_index()
                    and not self.is_qualified_identifier_cast_target_at_current_index()
                ):
                    return False

                token_type = self.current_token[0]
                self.eat(token_type)
                self.consume_composite_scalar_type_suffix(token_type)

                while True:
                    if self.current_token[0] == "LESS_THAN":
                        self.parse_template_suffix()
                        continue
                    if self.current_token[0] == "SCOPE":
                        self.eat("SCOPE")
                        self.parse_name_component()
                        continue
                    break

            while self.current_token[0] in {
                "MULTIPLY",
                *self.TYPE_REFERENCE_TOKENS,
                *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
            }:
                self.eat(self.current_token[0])

            while self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                while self.current_token[0] not in ["RBRACKET", "EOF"]:
                    self.eat(self.current_token[0])
                self.eat("RBRACKET")

            return self.current_token[0] == "RPAREN"
        finally:
            self.current_index = saved_index
            self.current_token = self.tokens[self.current_index]

    def consume_composite_scalar_type_suffix(self, token_type):
        if token_type == "LONG":
            saw_long_long = False
            if self.current_token[0] == "LONG":
                self.eat("LONG")
                saw_long_long = True

            if saw_long_long and self.current_token[0] in {"SIGNED", "UNSIGNED"}:
                self.eat(self.current_token[0])

            if self.current_token[0] == "INT":
                self.eat("INT")
            elif not saw_long_long and self.current_token[0] == "DOUBLE":
                self.eat("DOUBLE")
        elif token_type == "SHORT" and self.current_token[0] == "INT":
            self.eat("INT")

    def is_identifier_cast_target_at_current_index(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        identifier_name = self.current_token[1]
        close_index = self.current_index + 1
        saw_declarator_marker = False
        while close_index < len(self.tokens) and self.tokens[close_index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            if self.tokens[close_index][0] in {
                "MULTIPLY",
                *self.TYPE_REFERENCE_TOKENS,
            }:
                saw_declarator_marker = True
            close_index += 1

        if close_index >= len(self.tokens) or self.tokens[close_index][0] != "RPAREN":
            return False

        operand_index = close_index + 1
        if (
            operand_index < len(self.tokens)
            and self.tokens[operand_index][0] == "LPAREN"
            and not self.is_identifier_type_name(identifier_name)
            and not self.is_probable_identifier_type_name(identifier_name)
        ):
            return False

        if (
            not self.is_identifier_type_name(identifier_name)
            and not saw_declarator_marker
            and operand_index < len(self.tokens)
            and self.tokens[operand_index][0] in {"MULTIPLY", "BITWISE_AND"}
        ):
            return False

        return self.is_cast_operand_start_at_index(operand_index)

    def is_qualified_identifier_cast_target_at_current_index(self):
        if self.current_token[0] != "IDENTIFIER":
            return False

        index = self.current_index + 1
        saw_scope = False
        saw_template = False
        final_token = self.current_token
        while True:
            if index < len(self.tokens) and self.tokens[index][0] == "LESS_THAN":
                index = self.skip_template_at_index(index)
                if index is None:
                    return False
                saw_template = True
                continue
            if (
                index + 1 < len(self.tokens)
                and self.tokens[index][0] == "SCOPE"
                and self.tokens[index + 1][0] in self.NAME_COMPONENT_TOKENS
            ):
                saw_scope = True
                final_token = self.tokens[index + 1]
                index += 2
                continue
            break

        if not saw_scope and not saw_template:
            return False

        while index < len(self.tokens) and self.tokens[index][0] in {
            "MULTIPLY",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1

        if index >= len(self.tokens) or self.tokens[index][0] != "RPAREN":
            return False

        operand_index = index + 1
        if (
            operand_index < len(self.tokens)
            and self.tokens[operand_index][0] == "LPAREN"
            and not self.is_probable_cast_target_name(final_token)
        ):
            return False

        return self.is_cast_operand_start_at_index(operand_index)

    def is_probable_cast_target_name(self, token):
        token_type, token_value = token
        if token_type != "IDENTIFIER":
            return token_type in self.TYPE_TOKENS
        return self.is_identifier_type_name(
            token_value
        ) or self.is_probable_identifier_type_name(token_value)

    def is_cast_operand_start_at_index(self, index):
        if index >= len(self.tokens):
            return False

        return self.tokens[index][0] in {
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
            if expr.op.startswith("post"):
                return f"{self.expression_to_text(expr.operand)}{expr.op[4:]}"
            return f"{expr.op}{self.expression_to_text(expr.operand)}"
        if isinstance(expr, FunctionCallNode):
            name = self.expression_to_text(expr.name)
            args = ", ".join(self.expression_to_text(arg) for arg in expr.args)
            return f"{name}({args})"
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
            op = self.normalized_operator_value()
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def normalized_operator_value(self):
        alternative_operators = {
            "and": "&&",
            "and_eq": "&=",
            "bitand": "&",
            "bitor": "|",
            "compl": "~",
            "not": "!",
            "not_eq": "!=",
            "or": "||",
            "or_eq": "|=",
            "xor": "^",
            "xor_eq": "^=",
        }
        return alternative_operators.get(self.current_token[1], self.current_token[1])
