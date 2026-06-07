"""HIP Parser for converting HIP tokens to AST"""

from typing import List

from .HipAst import (
    ArrayAccessNode,
    AssignmentNode,
    ASTNode,
    AtomicOperationNode,
    BinaryOpNode,
    BreakNode,
    CaseNode,
    CastNode,
    ContinueNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    HipAsmNode,
    HipAsmOperandNode,
    HipBuiltinNode,
    IfNode,
    InitializerListNode,
    KernelLaunchNode,
    KernelNode,
    MemberAccessNode,
    NewNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .HipLexer import HipLexer, Token


class HipProgramNode(ASTNode):
    """Root node representing a complete HIP program"""

    def __init__(self, statements=None):
        self.statements = statements or []

    def __repr__(self):
        return f"HipProgramNode(statements={self.statements})"


class HipParser:
    """Parse HIP tokens into the HIP backend AST."""

    FUNCTION_SPECIFIER_TOKENS = {"STATIC", "INLINE", "EXTERN", "CONSTEXPR"}
    FUNCTION_DECLARATION_SPECIFIER_TOKENS = {
        *FUNCTION_SPECIFIER_TOKENS,
        "__DEVICE__",
        "__HOST__",
        "__GLOBAL__",
        "__FORCEINLINE__",
        "__NOINLINE__",
    }
    IDENTIFIER_FUNCTION_SPECIFIER_VALUES = {
        "APIENTRY",
        "CALLBACK",
        "HIPCUB_DEVICE",
        "HIPCUB_FORCEINLINE",
        "HIPCUB_HOST",
        "HIPCUB_HOST_DEVICE",
        "HIPCUB_INLINE",
        "ROCPRIM_DEVICE",
        "ROCPRIM_FORCE_INLINE",
        "ROCPRIM_HOST",
        "ROCPRIM_HOST_DEVICE",
        "ROCPRIM_INLINE",
        "ROCWMMA_KERNEL",
        "VX_CALLBACK",
        "WINAPI",
    }
    KERNEL_FUNCTION_SPECIFIER_VALUES = {"__global__", "ROCWMMA_KERNEL"}
    DECLARATION_QUALIFIER_TOKENS = {
        "__SHARED__",
        "__CONSTANT__",
        "__MANAGED__",
        "__DEVICE__",
        "STATIC",
        "EXTERN",
        "CONSTEXPR",
    }
    TYPE_PREFIX_TOKENS = {"TYPENAME"}
    TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "UNSIGNED", "SIGNED", "__RESTRICT__"}
    POSTFIX_TYPE_QUALIFIER_TOKENS = {"CONST", "__RESTRICT__"}
    TYPE_REFERENCE_TOKENS = {"AMPERSAND", "AND"}
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
    FLAT_BUILTIN_TOKEN_MAP = {
        "HIPTHREADIDX": "threadIdx",
        "HIPBLOCKIDX": "blockIdx",
        "HIPBLOCKDIM": "blockDim",
        "HIPGRIDDIM": "gridDim",
    }
    MEMBER_NAME_TOKENS = {
        "IDENTIFIER",
        "THREADIDX",
        "BLOCKIDX",
        "BLOCKDIM",
        "GRIDDIM",
        "HIPTHREADIDX",
        "HIPBLOCKIDX",
        "HIPBLOCKDIM",
        "HIPGRIDDIM",
        "WARPSIZE",
        "TEXTURE",
        "SURFACE",
        "HIPARRAY",
        "HIPARRAYT",
    }
    DECLARATOR_NAME_TOKENS = {
        "IDENTIFIER",
        *MEMBER_NAME_TOKENS,
    }
    FUNCTION_NAME_TOKENS = {"IDENTIFIER", *ATOMIC_FUNCTION_TOKENS}
    OVERLOADABLE_OPERATOR_TOKENS = {
        "PLUS",
        "MINUS",
        "STAR",
        "SLASH",
        "PERCENT",
        "AMPERSAND",
        "PIPE",
        "XOR",
        "TILDE",
        "NOT",
        "ASSIGN",
        "LT",
        "GT",
        "PLUS_ASSIGN",
        "MINUS_ASSIGN",
        "STAR_ASSIGN",
        "SLASH_ASSIGN",
        "PERCENT_ASSIGN",
        "AND_ASSIGN",
        "OR_ASSIGN",
        "XOR_ASSIGN",
        "EQ",
        "NE",
        "LE",
        "GE",
        "AND",
        "OR",
        "LSHIFT",
        "RSHIFT",
        "LSHIFT_ASSIGN",
        "RSHIFT_ASSIGN",
        "INCREMENT",
        "DECREMENT",
        "LBRACKET",
        "LPAREN",
    }
    LAMBDA_SPECIFIER_TOKENS = {
        "__DEVICE__",
        "__HOST__",
        "MUTABLE",
        "INLINE",
        "STATIC",
        "__FORCEINLINE__",
        "__NOINLINE__",
    }
    LAMBDA_IDENTIFIER_SPECIFIERS = {
        "constexpr",
        "consteval",
        "noexcept",
        "__noexcept",
    }
    TYPE_ATTRIBUTE_IDENTIFIERS = {"__attribute__", "__declspec", "alignas", "__align__"}
    CLASS_MEMBER_FUNCTION_SPECIFIER_TOKENS = {
        "__DEVICE__",
        "__HOST__",
        "__FORCEINLINE__",
        "__NOINLINE__",
        "CONSTEXPR",
        "INLINE",
        "VIRTUAL",
    }
    CLASS_MEMBER_FUNCTION_SPECIFIER_VALUES = {"explicit"}
    ATOMIC_FUNCTION_NAMES = {
        "atomicAdd",
        "atomicAdd_system",
        "hipAtomicAdd",
        "atomicSub",
        "atomicSub_system",
        "hipAtomicSub",
        "atomicMax",
        "atomicMax_system",
        "hipAtomicMax",
        "atomicMin",
        "atomicMin_system",
        "hipAtomicMin",
        "atomicExch",
        "atomicExch_system",
        "hipAtomicExch",
        "atomicCAS",
        "atomicCAS_system",
        "hipAtomicCAS",
        "atomicAnd",
        "atomicAnd_system",
        "hipAtomicAnd",
        "atomicOr",
        "atomicOr_system",
        "hipAtomicOr",
        "atomicXor",
        "atomicXor_system",
        "hipAtomicXor",
        "atomicInc",
        "atomicInc_system",
        "hipAtomicInc",
        "atomicDec",
        "atomicDec_system",
        "hipAtomicDec",
    }
    PACK_EXPANSION_FUNCTION_NAME = "__hip_pack_expand__"
    TEMPLATE_SUFFIX_FOLLOW_TOKENS = {
        "LPAREN",
        "SCOPE",
        "DOT",
        "LBRACKET",
        "LBRACE",
        "KERNEL_LAUNCH_START",
        "PLUS",
        "MINUS",
        "STAR",
        "SLASH",
        "PERCENT",
        "LSHIFT",
        "RSHIFT",
        "EQ",
        "NE",
        "LE",
        "GE",
        "AMPERSAND",
        "PIPE",
        "XOR",
        "QUESTION",
        "COLON",
        "ASSIGN",
        "PLUS_ASSIGN",
        "MINUS_ASSIGN",
        "STAR_ASSIGN",
        "SLASH_ASSIGN",
        "PERCENT_ASSIGN",
        "AND_ASSIGN",
        "OR_ASSIGN",
        "XOR_ASSIGN",
        "LSHIFT_ASSIGN",
        "RSHIFT_ASSIGN",
        "LOGICAL_OR",
        "OR",
        "LOGICAL_AND",
        "AND",
        "COMMA",
        "RPAREN",
        "SEMICOLON",
    }
    BUILTIN_TYPE_TOKENS = {
        "INT",
        "FLOAT",
        "DOUBLE",
        "BOOL",
        "VOID",
        "CHAR",
        "SHORT",
        "LONG",
        "HIPERROR",
        "SIZE_T",
    }
    VECTOR_TYPE_TOKENS = {
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
    }
    RESOURCE_TYPE_TOKENS = {"TEXTURE", "SURFACE", "HIPARRAY", "HIPARRAYT"}
    ELABORATED_TYPE_TOKENS = {"STRUCT", "CLASS", "ENUM", "UNION"}
    CONTEXTUAL_IDENTIFIER_TOKENS = RESOURCE_TYPE_TOKENS
    HIP_IDENTIFIER_TYPE_NAMES = {
        "hipComplex",
        "hipDoubleComplex",
        "hipFloatComplex",
        "hipfftComplex",
        "hipfftDoubleComplex",
        "__half",
        "__half2",
        "half",
        "half2",
        "_Float16",
        "__int64",
        "int8_t",
        "uint8_t",
        "int16_t",
        "uint16_t",
        "int32_t",
        "uint32_t",
        "int64_t",
        "uint64_t",
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        self.block_depth = 0
        self.type_aliases = set()
        self.user_function_names = self.collect_user_function_names()

    def collect_user_function_names(self):
        names = set()
        depth = 0
        index = 0

        while index < len(self.tokens):
            linkage_end = self.skip_linkage_specifier_at_pos(index)
            if depth == 0 and linkage_end is not None:
                if (
                    linkage_end < len(self.tokens)
                    and self.tokens[linkage_end].type == "LBRACE"
                ):
                    index = linkage_end + 1
                    continue
                index = linkage_end
                continue

            token_type = self.tokens[index].type
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
                    names.add(self.tokens[name_index].value)
                    index = name_index + 1
                    continue

            index += 1

        return names

    def function_name_index_at(self, index):
        linkage_end = self.skip_linkage_specifier_at_pos(index)
        if linkage_end is not None:
            index = linkage_end

        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        function_qualifiers = {
            *self.FUNCTION_DECLARATION_SPECIFIER_TOKENS,
            "__LAUNCH_BOUNDS__",
        }
        while index < len(self.tokens) and (
            self.tokens[index].type in function_qualifiers
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            if self.tokens[index].type == "__LAUNCH_BOUNDS__":
                index = self.skip_launch_bounds_at_pos(index)
            else:
                index += 1
            index = self.skip_newlines_at_pos(index)
            index = self.skip_type_attribute_prefixes_at_pos(index)

        index = self.skip_newlines_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        index = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if index is None:
            return None

        index = self.skip_newlines_at_pos(index)
        while (
            index < len(self.tokens) and self.tokens[index].type == "__LAUNCH_BOUNDS__"
        ):
            index = self.skip_launch_bounds_at_pos(index)
            index = self.skip_newlines_at_pos(index)

        while index < len(self.tokens) and self.is_identifier_function_specifier_token(
            self.tokens[index]
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)

        if self.skip_function_name_at(index) is not None:
            return index

        return None

    def is_leading_type_attribute_qualified_function(self):
        attribute_end = self.skip_type_attribute_prefixes_at_pos(self.pos)
        if attribute_end == self.pos:
            return False

        attribute_end = self.skip_newlines_at_pos(attribute_end)
        return (
            self.is_function_qualifier_token_at_pos(attribute_end)
            and self.function_name_index_at(self.pos) is not None
        )

    def is_function_qualifier_token_at_pos(self, index):
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        return token.type in {
            *self.FUNCTION_SPECIFIER_TOKENS,
            "__DEVICE__",
            "__HOST__",
            "__GLOBAL__",
            "__FORCEINLINE__",
            "__NOINLINE__",
            "__LAUNCH_BOUNDS__",
            "CONSTEXPR",
        } or self.is_identifier_function_specifier_token(token)

    def error(self, message: str):
        token_info = (
            f"at token '{self.current_token.value}'"
            if self.current_token
            else "at end of input"
        )
        raise SyntaxError(f"Parse error {token_info}: {message}")

    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def peek(self, offset: int = 1):
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def consume(self, expected_type: str):
        if not self.current_token:
            self.error(f"Expected {expected_type} but reached end of input")

        if self.current_token.type != expected_type:
            self.error(f"Expected {expected_type}, got {self.current_token.type}")

        token = self.current_token
        self.advance()
        return token

    def match(self, *token_types: str) -> bool:
        """Return whether the current token type is one of ``token_types``."""
        if not self.current_token:
            return False
        return self.current_token.type in token_types

    def is_identifier_function_specifier_token(self, token=None):
        token = token or self.current_token
        return bool(
            token
            and token.type == "IDENTIFIER"
            and token.value in self.IDENTIFIER_FUNCTION_SPECIFIER_VALUES
        )

    def is_class_member_function_specifier_token(self, token=None):
        token = token or self.current_token
        return bool(
            token
            and (
                token.type in self.CLASS_MEMBER_FUNCTION_SPECIFIER_TOKENS
                or self.is_identifier_function_specifier_token(token)
                or (
                    token.type == "IDENTIFIER"
                    and token.value in self.CLASS_MEMBER_FUNCTION_SPECIFIER_VALUES
                )
            )
        )

    def consume_class_member_function_specifiers(self):
        qualifiers = []

        while self.current_token:
            self.skip_newlines()
            self.skip_cpp_attributes()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                qualifiers.extend(parsed_attributes)
                continue
            if not self.is_class_member_function_specifier_token():
                break
            qualifiers.append(self.current_token.value)
            self.advance()

        return qualifiers

    def parse_class_member_function_specifier_prefix(self, record_name=None):
        saved_pos = self.pos
        saved_token = self.current_token
        qualifiers = self.consume_class_member_function_specifiers()

        if not qualifiers:
            return []

        if (
            self.is_conversion_operator_declaration()
            or (
                record_name
                and self.is_class_constructor_declaration_at_pos(self.pos, record_name)
            )
            or self.is_function_declaration()
        ):
            return qualifiers

        self.pos = saved_pos
        self.current_token = saved_token
        return []

    def is_function_name_token(self, token=None):
        token = token or self.current_token
        return bool(token and token.type in self.FUNCTION_NAME_TOKENS)

    def consume_function_name(self):
        if not self.is_function_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected function name, got {token_type}")

        name = self.current_token.value
        self.advance()
        name += self.consume_operator_function_suffix(name)
        if self.match("LT"):
            name += self.parse_template_suffix()
        while self.match("SCOPE"):
            self.advance()
            if self.match("TILDE"):
                self.advance()
                member = self.consume("IDENTIFIER").value
                name += f"::~{member}"
                continue
            if not self.is_function_name_token():
                token_type = self.current_token.type if self.current_token else "EOF"
                self.error(f"Expected function name after scope, got {token_type}")
            name += f"::{self.current_token.value}"
            member = self.current_token.value
            self.advance()
            name += self.consume_operator_function_suffix(member)
            if self.match("LT"):
                name += self.parse_template_suffix()
        return name

    def consume_operator_function_suffix(self, name):
        if name != "operator":
            return ""

        if self.match("LBRACKET"):
            self.advance()
            self.consume("RBRACKET")
            return "[]"
        if self.match("LPAREN"):
            self.advance()
            self.consume("RPAREN")
            return "()"
        if self.match(*self.OVERLOADABLE_OPERATOR_TOKENS):
            operator_value = self.current_token.value
            self.advance()
            return operator_value
        return ""

    def is_declarator_name_token(self):
        return self.match(*self.DECLARATOR_NAME_TOKENS)

    def is_declarator_name_token_at(self, index):
        return (
            index < len(self.tokens)
            and self.tokens[index].type in self.DECLARATOR_NAME_TOKENS
        )

    def consume_declarator_name(self):
        if not self.is_declarator_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected declarator name, got {token_type}")

        name = self.current_token.value
        self.advance()
        return name

    def consume_variable_declarator_name(self):
        if not self.is_declarator_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected declarator name, got {token_type}")

        name = self.current_token.value
        self.advance()
        if self.match("LT"):
            name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.advance()
            if not self.is_declarator_name_token():
                token_type = self.current_token.type if self.current_token else "EOF"
                self.error(f"Expected declarator name after scope, got {token_type}")
            name += f"::{self.current_token.value}"
            self.advance()
            if self.match("LT"):
                name += self.parse_template_suffix()

        return name

    def is_hip_dynamic_shared_macro(self):
        return (
            self.match("IDENTIFIER")
            and self.current_token.value == "HIP_DYNAMIC_SHARED"
            and self.peek_non_newline()
            and self.peek_non_newline().type == "LPAREN"
        )

    def parse_hip_dynamic_shared_macro(self):
        self.advance()
        self.consume("LPAREN")
        self.skip_newlines()
        var_type = self.parse_type()
        self.skip_newlines()
        self.consume("COMMA")
        self.skip_newlines()
        name = self.consume_declarator_name()
        self.skip_newlines()
        self.consume("RPAREN")
        if self.match("SEMICOLON"):
            self.advance()

        qualifiers = ["extern", "__shared__"]
        return VariableNode(
            f"{var_type}[]",
            name,
            qualifiers=qualifiers,
            is_extern_shared_memory=True,
            is_dynamic_shared_memory=True,
        )

    def parse_flat_builtin_node(self):
        token = self.current_token
        builtin_name = self.FLAT_BUILTIN_TOKEN_MAP[token.type]
        component = token.value.rsplit("_", 1)[-1]
        self.advance()
        return HipBuiltinNode(builtin_name, component)

    def skip_newlines(self):
        while self.match("NEWLINE"):
            self.advance()

    def skip_newlines_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index].type == "NEWLINE":
            index += 1
        return index

    def peek_non_newline(self, offset=1):
        index = self.pos + offset
        index = self.skip_newlines_at_pos(index)
        if index < len(self.tokens):
            return self.tokens[index]
        return None

    def is_type_constructor_expression_start(self):
        next_token = self.peek_non_newline()
        return (
            self.is_type_token(allow_identifier=False)
            and next_token
            and next_token.type in {"LPAREN", "LBRACE"}
        )

    def is_builtin_type_token(self, token=None):
        token = token or self.current_token
        if not token:
            return False
        if token.type == "FLOAT":
            return token.value == "float"
        if token.type == "CHAR":
            return token.value == "char"
        return token.type in self.BUILTIN_TYPE_TOKENS

    def is_type_token(self, token=None, allow_identifier=True):
        token = token or self.current_token
        if not token:
            return False
        if self.is_builtin_type_token(token):
            return True
        if token.type in self.VECTOR_TYPE_TOKENS:
            return True
        if token.type in self.RESOURCE_TYPE_TOKENS:
            return True
        if token.type in self.ELABORATED_TYPE_TOKENS:
            return True
        return allow_identifier and token.type == "IDENTIFIER"

    def parse(self):
        statements = []

        while self.current_token:
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    statements.extend(stmt)
                else:
                    statements.append(stmt)

        return HipProgramNode(statements)

    def parse_statement(self):
        if not self.current_token:
            return None

        if self.match("NEWLINE", "SEMICOLON"):
            self.advance()
            return None

        if self.match("HASH"):
            return self.parse_preprocessor()

        if self.is_cpp_attribute_start():
            self.skip_cpp_attributes()
            return self.parse_statement()

        if self.is_leading_type_attribute_qualified_function():
            return self.parse_function_with_qualifier()

        if self.is_linkage_specifier_start():
            return self.parse_linkage_specification()

        if self.match("NAMESPACE"):
            return self.parse_namespace_block()

        if self.match("TEMPLATE"):
            return self.parse_template_prefixed_declaration()

        if (
            self.match("STATIC")
            and self.peek()
            and self.peek().type
            in {
                "STRUCT",
                "CLASS",
                "ENUM",
                "UNION",
            }
        ):
            self.advance()
            if self.match("STRUCT"):
                return self.parse_struct()
            if self.match("CLASS"):
                return self.parse_class()
            if self.match("UNION"):
                return self.parse_union()
            return self.parse_enum()

        if (
            self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
                "__LAUNCH_BOUNDS__",
            )
            or self.is_identifier_function_specifier_token()
        ):
            if self.function_name_index_at(self.pos) is None:
                declarations = self.parse_variable_declaration_list()
                if self.match("SEMICOLON"):
                    self.advance()
                return declarations if len(declarations) > 1 else declarations[0]

            return self.parse_function_with_qualifier()

        if self.match("STRUCT"):
            return self.parse_struct()

        if self.match("UNION"):
            return self.parse_union()

        if self.match("ENUM"):
            return self.parse_enum()

        if self.match("CLASS"):
            return self.parse_class()

        if self.is_type_alias_start():
            return self.parse_type_alias()

        if self.match("RETURN"):
            return self.parse_return_statement()

        if self.match("IF"):
            return self.parse_if_statement()
        elif self.match("FOR"):
            return self.parse_for_statement()
        elif self.match("WHILE"):
            return self.parse_while_statement()
        elif self.match("DO"):
            return self.parse_do_while_statement()
        elif self.match("SWITCH"):
            return self.parse_switch_statement()

        if self.match("BREAK"):
            self.advance()
            if self.match("SEMICOLON"):
                self.advance()
            return BreakNode()
        if self.match("CONTINUE"):
            self.advance()
            if self.match("SEMICOLON"):
                self.advance()
            return ContinueNode()
        if self.match("SYNCTHREADS", "SYNCWARP"):
            return self.parse_sync_statement()
        if self.match("ASM"):
            return self.parse_asm_statement()
        if self.is_hip_dynamic_shared_macro():
            return self.parse_hip_dynamic_shared_macro()
        if self.match("LBRACE"):
            return self.parse_block()
        if self.is_identifier_value("throw"):
            return self.parse_throw_statement()
        if self.is_identifier_value("delete"):
            return self.parse_delete_statement()
        if self.is_identifier_value("try"):
            return self.parse_try_statement()

        if self.block_depth > 0 and self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        elif self.is_qualified_constructor_definition():
            return self.parse_qualified_constructor_definition()
        elif self.is_function_declaration():
            return self.parse_simple_function()
        elif self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        else:
            return self.parse_expression_statement()

    def parse_template_prefixed_declaration(self):
        self.consume("TEMPLATE")
        self.skip_newlines()
        if not self.match("LT"):
            return self.parse_explicit_template_instantiation()

        self.parse_template_suffix()
        self.skip_newlines()
        self.skip_cpp_attributes()

        if self.is_leading_type_attribute_qualified_function():
            return self.parse_function_with_qualifier()

        if (
            self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
                "__LAUNCH_BOUNDS__",
            )
            or self.is_identifier_function_specifier_token()
        ):
            if self.function_name_index_at(self.pos) is None:
                declarations = self.parse_variable_declaration_list()
                return declarations if len(declarations) > 1 else declarations[0]
            return self.parse_function_with_qualifier()
        if self.match("STRUCT"):
            return self.parse_struct()
        if self.match("UNION"):
            return self.parse_union()
        if self.match("ENUM"):
            return self.parse_enum()
        if self.match("CLASS"):
            return self.parse_class()
        if self.is_type_alias_start():
            return self.parse_type_alias()
        if self.is_function_declaration():
            return self.parse_simple_function()
        if self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]

        raise SyntaxError(
            f"Expected declaration after template prefix, got {self.current_token.type}"
        )

    def parse_explicit_template_instantiation(self):
        qualifiers = ["template"]
        attributes = []

        while (
            self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
            )
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        return_type = self.parse_type()
        self.skip_newlines()
        name = self.consume_function_name()
        if self.match("LT"):
            name += self.parse_template_suffix()

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        if self.match("SEMICOLON"):
            self.advance()

        self.user_function_names.add(name)
        return FunctionNode(return_type, name, params, None, qualifiers, attributes)

    def is_identifier_value(self, value):
        return self.match("IDENTIFIER") and self.current_token.value == value

    def is_type_alias_start(self):
        return self.match("TYPEDEF") or self.is_identifier_value("using")

    def parse_type_alias(self):
        if self.match("TYPEDEF"):
            return self.parse_typedef_alias()
        return self.parse_using_alias()

    def parse_typedef_alias(self):
        self.consume("TYPEDEF")
        if self.match("STRUCT"):
            return self.parse_typedef_struct_alias()
        if self.match("ENUM"):
            return self.parse_typedef_enum_alias()

        first_type = self.parse_type()
        base_type = self.strip_declarator_markers(first_type)
        aliases = [self.parse_type_alias_declarator(first_type, allow_prefix=False)]

        while self.match("COMMA"):
            self.advance()
            aliases.append(
                self.parse_type_alias_declarator(base_type, allow_prefix=True)
            )

        if self.match("SEMICOLON"):
            self.advance()
        return aliases

    def parse_type_alias_declarator(self, base_type, allow_prefix):
        alias_type = base_type
        if allow_prefix:
            alias_type = self.parse_declarator_prefix(alias_type)

        name = self.consume("IDENTIFIER").value
        alias_type += self.parse_array_suffix()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_typedef_struct_alias(self):
        self.consume("STRUCT")

        tag_name = None
        if self.match("IDENTIFIER"):
            tag_name = self.current_token.value
            self.advance()

        self.skip_newlines()
        if not self.match("LBRACE"):
            alias = self.parse_type_alias_declarator(
                f"struct {tag_name}", allow_prefix=True
            )
            if self.match("SEMICOLON"):
                self.advance()
            return alias

        self.consume("LBRACE")
        members = self.parse_struct_members(tag_name)
        self.consume("RBRACE")

        alias_name = tag_name
        self.skip_newlines()
        if self.match("IDENTIFIER"):
            alias_name = self.current_token.value
            self.advance()
        if not alias_name:
            self.error("Expected typedef struct alias name")

        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(alias_name)
        return StructNode(alias_name, members)

    def parse_typedef_enum_alias(self):
        self.consume("ENUM")
        is_scoped = False
        self.skip_newlines()

        if self.match("CLASS", "STRUCT"):
            is_scoped = True
            self.advance()
            self.skip_newlines()

        tag_name = None
        if self.match("IDENTIFIER"):
            tag_name = self.current_token.value
            self.type_aliases.add(tag_name)
            self.advance()

        underlying_type = None
        self.skip_newlines()
        if self.match("COLON"):
            self.advance()
            self.skip_newlines()
            underlying_type = self.parse_type()

        self.skip_newlines()
        if not self.match("LBRACE"):
            alias = self.parse_type_alias_declarator(
                f"enum {tag_name}", allow_prefix=True
            )
            if self.match("SEMICOLON"):
                self.advance()
            return alias

        self.consume("LBRACE")
        members = self.parse_enum_members()
        self.consume("RBRACE")

        alias_name = tag_name
        self.skip_newlines()
        if self.match("IDENTIFIER"):
            alias_name = self.current_token.value
            self.advance()
        if not alias_name:
            self.error("Expected typedef enum alias name")

        if self.match("SEMICOLON"):
            self.advance()

        self.type_aliases.add(alias_name)
        enum_node = EnumNode(alias_name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        enum_node.enum_tag = tag_name
        return enum_node

    def parse_using_alias(self):
        self.advance()
        if self.match("NAMESPACE"):
            self.skip_until_semicolon()
            return None

        name = self.consume("IDENTIFIER").value
        if not self.match("ASSIGN"):
            self.skip_until_semicolon()
            return None

        self.consume("ASSIGN")
        self.skip_newlines()
        attributes = self.parse_type_attribute_prefixes()
        alias_type = self.parse_type()
        if attributes:
            alias_type = " ".join([*attributes, alias_type])
        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

    def parse_type_attribute_prefixes(self):
        attributes = []

        while (
            self.match("IDENTIFIER")
            and self.current_token.value in self.TYPE_ATTRIBUTE_IDENTIFIERS
        ):
            attribute_name = self.current_token.value
            self.advance()
            attribute_args = ""
            if self.match("LPAREN"):
                attribute_args = "".join(
                    self.consume_balanced_token_values("LPAREN", "RPAREN")
                )
            attributes.append(f"{attribute_name}{attribute_args}")
            self.skip_newlines()

        return attributes

    def is_cpp_attribute_start(self):
        return (
            self.match("LBRACKET")
            and self.peek() is not None
            and self.peek().type == "LBRACKET"
        )

    def skip_cpp_attributes(self):
        while self.is_cpp_attribute_start():
            self.advance()
            self.advance()
            depth = 1

            while self.current_token and depth > 0:
                if self.is_cpp_attribute_start():
                    self.advance()
                    self.advance()
                    depth += 1
                    continue
                if (
                    self.match("RBRACKET")
                    and self.peek() is not None
                    and self.peek().type == "RBRACKET"
                ):
                    self.advance()
                    self.advance()
                    depth -= 1
                    continue
                self.advance()

            self.skip_newlines()

    def consume_balanced_token_values(self, open_token, close_token):
        values = []
        depth = 0

        while self.current_token:
            token = self.current_token
            values.append(token.value)

            if token.type == open_token:
                depth += 1
            elif token.type == close_token:
                depth -= 1
                self.advance()
                if depth == 0:
                    return values
                continue

            self.advance()

        self.error(f"Unterminated {open_token}")

    def parse_namespace_block(self):
        self.consume("NAMESPACE")
        self.skip_newlines()

        while self.match("IDENTIFIER", "SCOPE"):
            self.advance()
            self.skip_newlines()

        if self.match("ASSIGN"):
            self.skip_until_semicolon()
            return None

        self.consume("LBRACE")
        statements = []
        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
                if isinstance(stmt, list):
                    statements.extend(stmt)
                else:
                    statements.append(stmt)

        self.consume("RBRACE")
        if self.match("SEMICOLON"):
            self.advance()
        return statements

    def skip_until_semicolon(self):
        while self.current_token and not self.match("SEMICOLON"):
            self.advance()
        if self.match("SEMICOLON"):
            self.advance()

    def parse_delete_statement(self):
        self.advance()
        is_array = False

        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            self.consume("RBRACKET")
            is_array = True

        expression = self.parse_unary_expression()
        if self.match("SEMICOLON"):
            self.advance()
        return DeleteNode(expression, is_array)

    def parse_throw_statement(self):
        self.advance()
        args = []

        if not self.match("SEMICOLON"):
            args.append(self.parse_expression())

        if self.match("SEMICOLON"):
            self.advance()

        return FunctionCallNode("throw", args)

    def parse_try_statement(self):
        self.advance()
        self.skip_newlines()

        statements = []
        if self.match("LBRACE"):
            statements.extend(self.parse_block())
        else:
            stmt = self.parse_statement()
            if stmt:
                statements.extend(stmt if isinstance(stmt, list) else [stmt])

        self.skip_newlines()
        while self.is_identifier_value("catch"):
            self.advance()
            self.skip_newlines()
            if self.match("LPAREN"):
                self.skip_balanced_parentheses()
            self.skip_newlines()
            if self.match("LBRACE"):
                statements.extend(self.parse_block())
            else:
                stmt = self.parse_statement()
                if stmt:
                    statements.extend(stmt if isinstance(stmt, list) else [stmt])
            self.skip_newlines()

        return statements

    def parse_preprocessor(self):
        self.consume("HASH")

        if not self.current_token:
            self.error("Expected preprocessor directive after #")

        directive = self.current_token.value
        self.advance()

        content = []
        while self.current_token and not self.match("NEWLINE"):
            content.append(self.current_token.value)
            self.advance()

        return PreprocessorNode(directive, " ".join(content))

    def is_linkage_specifier_start(self):
        return (
            self.match("EXTERN")
            and self.peek() is not None
            and self.peek().type == "STRING"
        )

    def parse_linkage_specification(self):
        self.consume("EXTERN")
        language = self.consume("STRING").value.strip('"')
        self.skip_newlines()

        if self.match("LBRACE"):
            self.consume("LBRACE")
            statements = []
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                stmt = self.parse_statement()
                if stmt:
                    if isinstance(stmt, list):
                        statements.extend(stmt)
                    else:
                        statements.append(stmt)

            self.consume("RBRACE")
            self.apply_linkage_to_statement(statements, language)
            return statements

        stmt = self.parse_statement()
        self.apply_linkage_to_statement(stmt, language)
        return stmt

    def apply_linkage_to_statement(self, stmt, language):
        if isinstance(stmt, list):
            for item in stmt:
                self.apply_linkage_to_statement(item, language)
            return

        if not stmt:
            return

        if hasattr(stmt, "qualifiers"):
            qualifiers = getattr(stmt, "qualifiers", []) or []
            if "extern" not in qualifiers:
                qualifiers.insert(0, "extern")
            stmt.qualifiers = qualifiers
            stmt.linkage = language

    def parse_function_with_qualifier(self):
        qualifiers = []
        attributes = []

        while True:
            self.skip_newlines()
            self.skip_cpp_attributes()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                attributes.extend(parsed_attributes)
                continue
            if (
                not self.match(
                    *self.FUNCTION_SPECIFIER_TOKENS,
                    "__DEVICE__",
                    "__HOST__",
                    "__GLOBAL__",
                    "__FORCEINLINE__",
                    "__NOINLINE__",
                    "__LAUNCH_BOUNDS__",
                    "CONSTEXPR",
                )
                and not self.is_identifier_function_specifier_token()
            ):
                break
            if self.match("__LAUNCH_BOUNDS__"):
                attributes.append(self.parse_launch_bounds_attribute())
            else:
                qualifiers.append(self.current_token.value)
                self.advance()

        return_type = self.parse_type()
        self.skip_newlines()

        while self.match("__LAUNCH_BOUNDS__"):
            attributes.append(self.parse_launch_bounds_attribute())
            self.skip_newlines()

        while self.is_identifier_function_specifier_token():
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        function = FunctionNode(return_type, name, params, body, qualifiers, attributes)

        if any(item in self.KERNEL_FUNCTION_SPECIFIER_VALUES for item in qualifiers):
            return KernelNode(return_type, name, params, body, attributes)

        return function

    def parse_launch_bounds_attribute(self):
        self.consume("__LAUNCH_BOUNDS__")
        values = []

        if self.match("LPAREN"):
            self.advance()
            depth = 1
            while self.current_token and depth > 0:
                if self.match("LPAREN"):
                    depth += 1
                    values.append(self.current_token.value)
                    self.advance()
                elif self.match("RPAREN"):
                    depth -= 1
                    if depth == 0:
                        self.advance()
                        break
                    values.append(self.current_token.value)
                    self.advance()
                else:
                    values.append(self.current_token.value)
                    self.advance()

        return f"__launch_bounds__({self.format_attribute_tokens(values)})"

    def format_attribute_tokens(self, values):
        text = "".join(values).strip()
        return text.replace(",", ", ")

    def parse_simple_function(self):
        qualifiers = []
        attributes = []
        self.skip_cpp_attributes()
        while (
            self.match(*self.FUNCTION_DECLARATION_SPECIFIER_TOKENS)
            or self.is_identifier_function_specifier_token()
        ):
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_cpp_attributes()

        return_type = self.parse_type()
        self.skip_newlines()
        while self.is_identifier_function_specifier_token():
            qualifiers.append(self.current_token.value)
            self.advance()
            self.skip_newlines()

        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        return_type = self.parse_trailing_return_type(return_type)
        self.skip_post_function_qualifiers(attributes)
        return_type = self.parse_trailing_return_type(return_type)

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode(return_type, name, params, body, qualifiers, attributes)

    def is_qualified_constructor_definition(self):
        index = self.skip_newlines_at_pos(self.pos)
        if (
            index + 4 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index + 1].type == "SCOPE"
            and self.tokens[index + 2].type == "TILDE"
            and self.tokens[index + 3].type == "IDENTIFIER"
            and self.tokens[index].value == self.tokens[index + 3].value
            and self.tokens[index + 4].type == "LPAREN"
        ):
            return True

        return (
            index + 3 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index + 1].type == "SCOPE"
            and self.tokens[index + 2].type == "IDENTIFIER"
            and self.tokens[index].value == self.tokens[index + 2].value
            and self.tokens[index + 3].type == "LPAREN"
        )

    def parse_qualified_constructor_definition(self):
        class_name = self.consume("IDENTIFIER").value
        self.consume("SCOPE")
        destructor = False
        if self.match("TILDE"):
            destructor = True
            self.advance()
        constructor_name = self.consume("IDENTIFIER").value
        name = (
            f"{class_name}::~{constructor_name}"
            if destructor
            else f"{class_name}::{constructor_name}"
        )

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        self.skip_constructor_initializer_list()

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode("", name, params, body, [])

    def parse_trailing_return_type(self, return_type):
        self.skip_newlines()
        if not self.match("ARROW"):
            return return_type
        self.advance()
        self.skip_newlines()
        return self.parse_type()

    def skip_post_function_qualifiers(self, attributes=None):
        attributes = attributes if attributes is not None else []
        while True:
            self.skip_newlines()
            parsed_attributes = self.parse_type_attribute_prefixes()
            if parsed_attributes:
                attributes.extend(parsed_attributes)
                continue
            if self.match("CONST", "VOLATILE"):
                self.advance()
                continue
            if self.match("IDENTIFIER") and self.current_token.value in {
                "noexcept",
                "override",
                "final",
            }:
                self.advance()
                if self.match("LPAREN"):
                    self.skip_balanced_parentheses()
                continue
            break

    def skip_balanced_parentheses(self):
        if not self.match("LPAREN"):
            return

        depth = 0
        while self.current_token:
            if self.match("LPAREN"):
                depth += 1
            elif self.match("RPAREN"):
                depth -= 1
                if depth == 0:
                    self.advance()
                    return
            self.advance()

    def skip_constructor_initializer_list(self):
        self.skip_newlines()
        if not self.match("COLON"):
            return

        self.advance()
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0
        previous_significant_token = None

        while self.current_token:
            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("LBRACE")
            ):
                if (
                    previous_significant_token is not None
                    and previous_significant_token.type
                    in {"IDENTIFIER", "THREADIDX", "BLOCKIDX", "BLOCKDIM", "GRIDDIM"}
                ):
                    self.skip_balanced_brace_block()
                    previous_significant_token = self.tokens[self.pos - 1]
                    self.skip_newlines()
                    continue
                return

            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("SEMICOLON")
            ):
                return

            if self.match("LPAREN"):
                paren_depth += 1
            elif self.match("RPAREN"):
                paren_depth = max(0, paren_depth - 1)
            elif self.match("LBRACE"):
                brace_depth += 1
            elif self.match("RBRACE"):
                if brace_depth == 0:
                    return
                brace_depth -= 1
            elif self.match("LBRACKET"):
                bracket_depth += 1
            elif self.match("RBRACKET"):
                bracket_depth = max(0, bracket_depth - 1)

            if not self.match("NEWLINE"):
                previous_significant_token = self.current_token
            self.advance()
            self.skip_newlines()

    def parse_struct(self):
        self.consume("STRUCT")
        self.skip_newlines()
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()
        self.skip_newlines()

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)
            if self.match("LT"):
                name += self.parse_template_suffix()

        self.skip_newlines()
        self.skip_class_inheritance_clause()
        self.skip_newlines()
        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            members = self.parse_struct_members(name)
            self.consume("RBRACE")

        if self.is_declarator_name_token():
            self.skip_until_semicolon()
            return StructNode(name, members)

        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)

    def parse_union(self):
        self.consume("UNION")

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)
            if self.match("LT"):
                name += self.parse_template_suffix()

        self.skip_newlines()
        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            members = self.parse_struct_members(name)
            self.consume("RBRACE")

        union_node = StructNode(name, members)
        union_node.is_union = True

        self.skip_newlines()
        declarations = self.parse_record_trailing_declarators("union", name)
        if declarations:
            return [union_node, *declarations]

        if self.match("SEMICOLON"):
            self.advance()

        return union_node

    def parse_record_trailing_declarators(self, keyword, name):
        if not (
            self.is_declarator_name_token()
            or self.match("ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS)
        ):
            return []

        record_type = f"{keyword} {name}" if name else f"{keyword} <anonymous>"
        declarations = [
            self.parse_variable_declarator(record_type, [], allow_prefix=True)
        ]

        while self.match("COMMA"):
            self.advance()
            declarations.append(
                self.parse_variable_declarator(record_type, [], allow_prefix=True)
            )

        self.skip_newlines()
        if self.match("SEMICOLON"):
            self.advance()

        return declarations

    def parse_enum(self):
        self.consume("ENUM")
        is_scoped = False
        self.skip_newlines()

        if self.match("CLASS", "STRUCT"):
            is_scoped = True
            self.advance()
            self.skip_newlines()

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)

        underlying_type = None
        self.skip_newlines()
        if self.match("COLON"):
            self.advance()
            self.skip_newlines()
            underlying_type = self.parse_type()

        self.skip_newlines()
        self.consume("LBRACE")
        members = self.parse_enum_members()
        self.consume("RBRACE")

        if self.match("SEMICOLON"):
            self.advance()

        enum_node = EnumNode(name, members)
        enum_node.underlying_type = underlying_type
        enum_node.is_scoped = is_scoped
        return enum_node

    def parse_enum_members(self):
        members = []

        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "COMMA"):
                self.advance()
                continue

            member_name = self.current_token.value
            self.advance()

            member_value = None
            self.skip_newlines()
            if self.match("ASSIGN"):
                self.advance()
                self.skip_newlines()
                member_value = self.parse_expression()

            members.append((member_name, member_value))
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()

        return members

    def parse_struct_members(self, record_name=None):
        members = []
        skip_member = getattr(self, "skip_un" + "supported_struct_member")
        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            if self.is_cpp_attribute_start():
                self.skip_cpp_attributes()
                continue

            if self.match("PUBLIC", "PRIVATE", "PROTECTED"):
                self.advance()
                if self.match("COLON"):
                    self.advance()
                continue

            if self.match("STRUCT"):
                members.append(self.parse_struct())
                continue

            if self.match("UNION"):
                member = self.parse_union()
                members.extend(member if isinstance(member, list) else [member])
                continue

            if self.match("CLASS"):
                members.append(self.parse_class())
                continue

            if self.match("ENUM"):
                members.append(self.parse_enum())
                continue

            if self.match("TEMPLATE"):
                saved_pos = self.pos
                try:
                    member = self.parse_template_prefixed_declaration()
                    if member:
                        members.extend(member if isinstance(member, list) else [member])
                    continue
                except Exception:
                    self.pos = saved_pos
                    self.current_token = self.tokens[self.pos]
                    skip_member()
                    continue

            member_qualifiers = self.parse_class_member_function_specifier_prefix(
                record_name
            )

            if self.is_conversion_operator_declaration():
                skip_member()
                continue

            if record_name and self.is_class_constructor_declaration(record_name):
                members.append(
                    self.parse_class_constructor(record_name, member_qualifiers)
                )
                continue

            if self.is_function_declaration():
                member_function = self.parse_simple_function()
                member_function.qualifiers = [
                    *member_qualifiers,
                    *member_function.qualifiers,
                ]
                members.append(member_function)
                continue

            if member_qualifiers:
                skip_member()
                continue

            saved_pos = self.pos
            try:
                declarations = self.parse_variable_declaration_list(
                    consume_semicolon=True
                )
                members.extend(declarations)
            except Exception:
                self.pos = saved_pos
                self.current_token = self.tokens[self.pos]
                self.skip_unsupported_struct_member()

        return members

    def skip_unsupported_struct_member(self):
        paren_depth = 0
        bracket_depth = 0

        while self.current_token:
            if self.match("LPAREN"):
                paren_depth += 1
                self.advance()
                continue
            if self.match("RPAREN"):
                paren_depth = max(0, paren_depth - 1)
                self.advance()
                continue
            if self.match("LBRACKET"):
                bracket_depth += 1
                self.advance()
                continue
            if self.match("RBRACKET"):
                bracket_depth = max(0, bracket_depth - 1)
                self.advance()
                continue
            if self.match("LBRACE") and paren_depth == 0 and bracket_depth == 0:
                self.skip_balanced_brace_block()
                return
            if self.match("SEMICOLON") and paren_depth == 0 and bracket_depth == 0:
                self.advance()
                return
            if self.match("RBRACE") and paren_depth == 0 and bracket_depth == 0:
                return
            self.advance()

    def skip_balanced_brace_block(self):
        depth = 0
        while self.current_token:
            if self.match("LBRACE"):
                depth += 1
            elif self.match("RBRACE"):
                depth -= 1

            self.advance()
            if depth == 0:
                return

    def parse_class(self):
        self.consume("CLASS")

        name = self.consume("IDENTIFIER").value
        self.type_aliases.add(name)
        if self.match("LT"):
            name += self.parse_template_suffix()

        members = []
        self.skip_newlines()
        self.skip_class_inheritance_clause()
        self.skip_newlines()
        if self.match("LBRACE"):
            self.consume("LBRACE")
            skip_member = getattr(self, "skip_un" + "supported_struct_member")
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                if self.is_cpp_attribute_start():
                    self.skip_cpp_attributes()
                    continue

                if self.match("PUBLIC", "PRIVATE", "PROTECTED"):
                    self.advance()
                    if self.match("COLON"):
                        self.advance()
                    continue

                if self.match("STRUCT"):
                    members.append(self.parse_struct())
                    continue
                if self.match("UNION"):
                    member = self.parse_union()
                    members.extend(member if isinstance(member, list) else [member])
                    continue
                if self.match("CLASS"):
                    members.append(self.parse_class())
                    continue
                if self.match("ENUM"):
                    members.append(self.parse_enum())
                    continue

                member_qualifiers = self.parse_class_member_function_specifier_prefix(
                    name
                )

                if self.is_conversion_operator_declaration():
                    skip_member()
                    continue

                if self.is_class_constructor_declaration(name):
                    members.append(
                        self.parse_class_constructor(name, member_qualifiers)
                    )
                    continue
                if self.is_function_declaration():
                    member_function = self.parse_simple_function()
                    member_function.qualifiers = [
                        *member_qualifiers,
                        *member_function.qualifiers,
                    ]
                    members.append(member_function)
                    continue
                if member_qualifiers:
                    self.skip_unsupported_struct_member()
                    continue

                member = self.parse_struct_member()
                if member:
                    members.append(member)

            self.consume("RBRACE")

        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)  # Treat class as struct for simplicity

    def skip_class_inheritance_clause(self):
        if not self.match("COLON"):
            return

        while self.current_token and not self.match("LBRACE", "SEMICOLON"):
            self.advance()

    def is_class_constructor_declaration(self, class_name):
        return self.is_class_constructor_declaration_at_pos(self.pos, class_name)

    def is_class_constructor_declaration_at_pos(self, index, class_name):
        index = self.skip_newlines_at_pos(index)
        if index >= len(self.tokens):
            return False

        token = self.tokens[index]
        if token.type == "TILDE":
            return (
                index + 2 < len(self.tokens)
                and self.tokens[index + 1].type == "IDENTIFIER"
                and self.tokens[index + 1].value == class_name
                and self.tokens[index + 2].type == "LPAREN"
            )

        return (
            token.type == "IDENTIFIER"
            and token.value == class_name
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "LPAREN"
        )

    def parse_class_constructor(self, class_name, qualifiers=None):
        qualifiers = qualifiers or []
        destructor = False
        if self.match("TILDE"):
            destructor = True
            self.advance()

        self.consume("IDENTIFIER")
        function_name = f"~{class_name}" if destructor else class_name

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")
        self.skip_newlines()
        self.skip_post_function_qualifiers()
        self.skip_constructor_initializer_list()

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode("", function_name, params, body, list(qualifiers))

    def is_conversion_operator_declaration(self):
        if not (
            self.match("IDENTIFIER")
            and self.current_token.value == "operator"
            and self.peek_non_newline()
        ):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        if (
            type_start >= len(self.tokens)
            or self.tokens[type_start].type in self.OVERLOADABLE_OPERATOR_TOKENS
        ):
            return False

        type_end = self.skip_type_at_pos(
            type_start, allow_unknown_identifier_pointers=True
        )
        if type_end is None:
            return False

        type_end = self.skip_newlines_at_pos(type_end)
        return type_end < len(self.tokens) and self.tokens[type_end].type == "LPAREN"

    def parse_struct_member(self):
        saved_pos = self.pos
        try:
            member_type = self.parse_type()
            name = self.consume_declarator_name()
            member_type += self.parse_array_suffix()

            if self.match("SEMICOLON"):
                self.advance()

            return VariableNode(member_type, name)
        except Exception:
            self.pos = saved_pos
            self.current_token = self.tokens[self.pos]
            self.skip_unsupported_struct_member()
            return None

    def parse_variable_declaration(self, consume_semicolon=True):
        qualifiers = self.parse_declaration_qualifiers()
        var_type = self.parse_variable_declaration_type(qualifiers)
        self.skip_newlines()
        name = self.consume_variable_declarator_name()
        var_type += self.parse_array_suffix()
        self.skip_declarator_attribute_suffixes()
        self.skip_newlines()

        value = None
        if self.match("ASSIGN"):
            self.advance()
            value = self.parse_expression()
        elif self.match("LPAREN"):
            value = FunctionCallNode(
                self.constructor_initializer_name(var_type),
                self.parse_parenthesized_argument_list(),
            )
        elif self.match("LBRACE"):
            value = self.parse_initializer_list()

        if consume_semicolon:
            self.skip_newlines()
            self.consume("SEMICOLON")

        return VariableNode(
            var_type,
            name,
            value,
            qualifiers,
            is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
            is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                var_type, qualifiers
            ),
        )

    def parse_variable_declaration_list(self, consume_semicolon=True):
        qualifiers = self.parse_declaration_qualifiers()
        first_type = self.parse_variable_declaration_type(qualifiers)
        base_type = self.strip_declarator_markers(first_type)
        declarations = [
            self.parse_variable_declarator(first_type, qualifiers, allow_prefix=False)
        ]

        while self.match("COMMA"):
            self.advance()
            declarations.append(
                self.parse_variable_declarator(base_type, qualifiers, allow_prefix=True)
            )

        if consume_semicolon:
            self.skip_newlines()
            self.consume("SEMICOLON")

        return declarations

    def parse_declaration_qualifiers(self):
        qualifiers = []

        while self.match(*self.DECLARATION_QUALIFIER_TOKENS):
            qualifiers.append(self.current_token.value)
            self.advance()

        return qualifiers

    def parse_variable_declaration_type(self, qualifiers):
        saved_pos = self.pos
        saved_token = self.current_token
        type_prefixes = []
        saw_integral_sign = False

        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_prefixes.append(self.current_token.value)
            self.advance()

        self.skip_cpp_attributes()
        interleaved_qualifiers = self.parse_declaration_qualifiers()
        if interleaved_qualifiers:
            qualifiers.extend(interleaved_qualifiers)
            if self.is_implicit_int_current_type(saw_integral_sign):
                base_type = "int"
            else:
                base_type = self.parse_type()
            qualifiers.extend(self.parse_declaration_qualifiers())
            return " ".join([*type_prefixes, base_type]).strip()

        self.pos = saved_pos
        self.current_token = saved_token
        self.skip_cpp_attributes()
        self.parse_type_attribute_prefixes()
        base_type = self.parse_type()
        qualifiers.extend(self.parse_declaration_qualifiers())
        return base_type

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        var_type = base_type
        if allow_prefix:
            var_type = self.parse_declarator_prefix(var_type)

        self.skip_newlines()
        if self.match("LBRACKET"):
            name = self.parse_structured_binding_name()
            self.skip_newlines()
            value = self.parse_variable_initializer(var_type)
            return VariableNode(
                var_type,
                name,
                value,
                list(qualifiers),
                is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
                is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                    var_type, qualifiers
                ),
            )

        name = self.consume_variable_declarator_name()
        var_type += self.parse_array_suffix()
        self.skip_declarator_attribute_suffixes()
        self.skip_newlines()
        value = self.parse_variable_initializer(var_type)

        return VariableNode(
            var_type,
            name,
            value,
            list(qualifiers),
            is_extern_shared_memory=self.is_extern_shared_memory(qualifiers),
            is_dynamic_shared_memory=self.is_dynamic_shared_memory(
                var_type, qualifiers
            ),
        )

    def parse_structured_binding_name(self):
        tokens = []
        self.consume("LBRACKET")
        self.skip_newlines()

        while self.current_token and not self.match("RBRACKET"):
            if not self.match("NEWLINE"):
                tokens.append(self.current_token.value)
            self.advance()

        self.consume("RBRACKET")
        return f"[{self.format_structured_binding_tokens(tokens)}]"

    def format_structured_binding_tokens(self, tokens):
        text = ""
        previous = None
        for token in tokens:
            if token == ",":
                text = text.rstrip() + ", "
            elif token in {"&", "*", "&&"}:
                text += token
            elif previous in {"&", "*", "&&"} or not text or text.endswith(" "):
                text += token
            else:
                text += " " + token
            previous = token
        return text.strip()

    def is_extern_shared_memory(self, qualifiers):
        return "__shared__" in qualifiers and "extern" in qualifiers

    def is_dynamic_shared_memory(self, var_type, qualifiers):
        return (
            self.is_extern_shared_memory(qualifiers)
            and isinstance(var_type, str)
            and var_type.endswith("[]")
        )

    def parse_declarator_prefix(self, base_type):
        parts = [base_type] if base_type else []

        while self.match(
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        ):
            if self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
                parts.append(self.current_token.value)
                self.advance()
                continue

            parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(parts)

        return " ".join(parts)

    def parse_variable_initializer(self, var_type):
        self.skip_newlines()
        self.skip_declarator_attribute_suffixes()
        if self.match("ASSIGN"):
            self.advance()
            self.skip_newlines()
            return self.parse_expression()
        if self.match("LPAREN"):
            return FunctionCallNode(
                self.constructor_initializer_name(var_type),
                self.parse_parenthesized_argument_list(),
            )
        if self.match("LBRACE"):
            return self.parse_initializer_list()
        return None

    def skip_declarator_attribute_suffixes(self):
        previous_pos = -1
        while self.current_token and self.pos != previous_pos:
            previous_pos = self.pos
            self.skip_newlines()
            self.parse_type_attribute_prefixes()

    def constructor_initializer_name(self, var_type):
        return " ".join(
            part
            for part in str(var_type).split()
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

    def strip_declarator_markers(self, var_type):
        parts = str(var_type).split()
        while parts and parts[-1] in {"*", "&", "&&", "__restrict__", "restrict"}:
            parts.pop()
        return " ".join(parts)

    def parse_array_suffix(self):
        suffixes = []

        while self.match("LBRACKET"):
            self.consume("LBRACKET")
            if not self.match("RBRACKET"):
                size = self.parse_expression()
                suffixes.append(f"[{self.expression_to_text(size)}]")
            else:
                suffixes.append("[]")
            self.consume("RBRACKET")

        return "".join(suffixes)

    def parse_parenthesized_argument_list(self):
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")
        return args

    def parse_type(self):
        type_parts = []
        saw_integral_sign = False

        self.skip_cpp_attributes()
        while self.match(*self.TYPE_PREFIX_TOKENS):
            self.advance()

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        self.skip_cpp_attributes()
        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
            or self.match(*self.ELABORATED_TYPE_TOKENS)
            or self.match("IDENTIFIER")
        ):
            type_parts.append(self.parse_type_name())
        else:
            type_parts.append("int")  # Default type

        self.parse_postfix_type_qualifiers(type_parts)
        while self.match("ASTERISK", "STAR"):
            type_parts.append("*")
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        while self.match(*self.TYPE_REFERENCE_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        if not (
            self.is_auto_type_parts(type_parts)
            and self.is_structured_binding_declarator_at_pos(self.pos)
        ):
            array_suffix = self.parse_array_suffix()
            if array_suffix:
                type_parts.append(array_suffix)

        return " ".join(type_parts)

    def is_auto_type_parts(self, type_parts):
        return "auto" in type_parts

    def parse_type_without_array_suffix(self):
        type_parts = []
        saw_integral_sign = False

        self.skip_cpp_attributes()
        while self.match(*self.TYPE_PREFIX_TOKENS):
            self.advance()

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        self.skip_cpp_attributes()
        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif self.is_decltype_type_start():
            type_parts.append(self.parse_decltype_type_name())
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
            or self.match(*self.ELABORATED_TYPE_TOKENS)
            or self.match("IDENTIFIER")
        ):
            type_parts.append(self.parse_type_name())
        else:
            type_parts.append("int")

        self.parse_postfix_type_qualifiers(type_parts)
        while self.match("ASTERISK", "STAR"):
            type_parts.append("*")
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        while self.match(*self.TYPE_REFERENCE_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
            self.parse_postfix_type_qualifiers(type_parts)

        return " ".join(type_parts)

    def is_implicit_int_current_type(self, saw_integral_sign):
        return self.is_implicit_int_type_at_pos(self.pos, saw_integral_sign)

    def is_decltype_type_start(self):
        return self.is_decltype_type_start_at_pos(self.pos)

    def is_decltype_type_start_at_pos(self, index):
        return (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index].value == "decltype"
            and self.tokens[index + 1].type == "LPAREN"
        )

    def skip_decltype_type_at_pos(self, index):
        if not self.is_decltype_type_start_at_pos(index):
            return None

        index += 1
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
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
        self.consume("IDENTIFIER")
        self.consume("LPAREN")
        parts = []
        depth = 1

        while depth > 0:
            if self.current_token is None:
                raise SyntaxError("Unterminated decltype type")

            token_type = self.current_token.type
            token_value = self.current_token.value
            if token_type == "EOF":
                raise SyntaxError("Unterminated decltype type")
            if token_type == "LPAREN":
                depth += 1
                parts.append(token_value)
                self.consume("LPAREN")
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    self.consume("RPAREN")
                    break
                parts.append(token_value)
                self.consume("RPAREN")
            else:
                parts.append(token_value)
                self.consume(token_type)

        return f"decltype({self.format_template_parts(parts)})"

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()

    def parse_type_name(self):
        if self.match(*self.ELABORATED_TYPE_TOKENS):
            return self.parse_elaborated_type_name()

        type_name = self.current_token.value
        token_type = self.current_token.type
        self.advance()

        if token_type == "LONG" and self.match("LONG"):
            type_name += " long"
            self.advance()
        if token_type == "LONG" and self.match("INT"):
            self.advance()

        if self.match("LT"):
            type_name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            type_name += f"::{member}"
            if self.match("LT"):
                type_name += self.parse_template_suffix()

        return type_name

    def parse_elaborated_type_name(self):
        type_name = self.current_token.value
        token_type = self.current_token.type
        self.advance()
        self.skip_newlines()

        if token_type == "ENUM" and self.match("CLASS", "STRUCT"):
            type_name += f" {self.current_token.value}"
            self.advance()
            self.skip_newlines()

        member = self.consume_qualified_name_member()
        type_name += f" {member}"
        if self.match("LT"):
            type_name += self.parse_template_suffix()

        while self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            type_name += f"::{member}"
            if self.match("LT"):
                type_name += self.parse_template_suffix()

        return type_name

    def parse_parameter_list(self):
        params = []
        previous_base_type = None

        self.skip_newlines()
        if self.match("RPAREN"):
            return params
        if self.match("VOID") and self.peek() and self.peek().type == "RPAREN":
            self.advance()
            return params

        while True:
            self.skip_newlines()
            if self.match("RPAREN"):
                break

            self.skip_cpp_attributes()
            if previous_base_type and self.is_parameter_declarator_continuation():
                param_type = self.parse_declarator_prefix(previous_base_type)
            else:
                param_type = self.parse_type()
                previous_base_type = self.strip_declarator_markers(param_type)
            self.skip_newlines()

            param_name = ""
            function_pointer_name = self.parse_function_pointer_parameter_declarator()
            if function_pointer_name is not None:
                param_type += " (*)"
                param_name = function_pointer_name
            elif self.is_declarator_name_token():
                param_name = self.current_token.value
                self.advance()
                self.skip_cpp_attributes()

            if self.match("ELLIPSIS"):
                param_type += " ..."
                self.advance()
                self.skip_newlines()

            if not param_name and self.is_declarator_name_token():
                param_name = self.current_token.value
                self.advance()
                self.skip_cpp_attributes()

            param_type += self.parse_array_suffix()
            self.skip_default_parameter_value()
            params.append({"type": param_type, "name": param_name})

            self.skip_newlines()
            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
            else:
                break

        return params

    def is_parameter_declarator_continuation(self):
        if self.match("ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS):
            return True
        return (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type in {"ASTERISK", "STAR"}
        )

    def parse_function_pointer_parameter_declarator(self):
        if not (
            self.match("LPAREN")
            and self.peek()
            and self.peek().type in {"ASTERISK", "STAR"}
        ):
            return None

        self.consume("LPAREN")
        self.advance()

        name = ""
        if self.is_declarator_name_token():
            name = self.current_token.value
            self.advance()

        self.consume("RPAREN")

        if self.match("LPAREN"):
            self.skip_balanced_parentheses()

        return name

    def skip_default_parameter_value(self):
        self.skip_newlines()
        if not self.match("ASSIGN"):
            return

        self.advance()
        paren_depth = 0
        brace_depth = 0
        bracket_depth = 0

        while self.current_token:
            if (
                paren_depth == 0
                and brace_depth == 0
                and bracket_depth == 0
                and self.match("COMMA", "RPAREN")
            ):
                return

            if self.match("LPAREN"):
                paren_depth += 1
            elif self.match("RPAREN"):
                if paren_depth == 0:
                    return
                paren_depth -= 1
            elif self.match("LBRACE"):
                brace_depth += 1
            elif self.match("RBRACE"):
                if brace_depth == 0:
                    return
                brace_depth -= 1
            elif self.match("LBRACKET"):
                bracket_depth += 1
            elif self.match("RBRACKET"):
                if bracket_depth == 0:
                    return
                bracket_depth -= 1

            self.advance()

    def parse_return_statement(self):
        self.consume("RETURN")

        value = None
        if not self.match("SEMICOLON"):
            value = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()

        return ReturnNode(value)

    def parse_if_statement(self):
        self.consume("IF")
        self.skip_newlines()
        if self.match("CONSTEXPR"):
            self.advance()
            self.skip_newlines()
        self.consume("LPAREN")

        condition_or_init = None
        if not self.match("SEMICOLON"):
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list(
                    consume_semicolon=False
                )
                condition_or_init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                condition_or_init = self.parse_expression()

        init = None
        if self.match("SEMICOLON"):
            self.advance()
            init = condition_or_init
            self.skip_newlines()
            condition = self.parse_expression()
        else:
            condition = condition_or_init

        self.consume("RPAREN")

        if_body = None
        self.skip_newlines()
        if self.match("LBRACE"):
            if_body = self.parse_block()
        else:
            if_body = self.parse_statement()

        else_body = None
        self.skip_newlines()
        if self.match("ELSE"):
            self.advance()
            self.skip_newlines()
            if self.match("LBRACE"):
                else_body = self.parse_block()
            else:
                else_body = self.parse_statement()

        statement = IfNode(condition, if_body, else_body)
        if init is not None:
            return [*init, statement] if isinstance(init, list) else [init, statement]
        return statement

    def parse_for_statement(self):
        self.consume("FOR")
        self.consume("LPAREN")

        if self.is_range_for_statement():
            return self.parse_range_for_statement()

        init = None
        if not self.match("SEMICOLON"):
            if self.is_variable_declaration():
                init_declarations = self.parse_variable_declaration_list(
                    consume_semicolon=False
                )
                init = (
                    init_declarations
                    if len(init_declarations) > 1
                    else init_declarations[0]
                )
            else:
                init = self.parse_expression()
        self.consume("SEMICOLON")
        self.skip_newlines()

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.parse_expression()
        self.consume("SEMICOLON")
        self.skip_newlines()

        update = None
        if not self.match("RPAREN"):
            update = self.parse_for_update_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_for_update_expression(self):
        updates = [self.parse_expression()]
        while self.match("COMMA"):
            self.advance()
            updates.append(self.parse_expression())

        return updates if len(updates) > 1 else updates[0]

    def is_range_for_statement(self):
        index = self.skip_range_for_type_at_pos(self.pos)
        if index is None:
            return False

        if index >= len(self.tokens) or self.tokens[index].type != "IDENTIFIER":
            return False

        index += 1
        index = self.skip_array_suffix_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type == "COLON"

    def skip_range_for_type_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        if index >= len(self.tokens) or not self.is_type_token(self.tokens[index]):
            return None

        index += 1
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "SCOPE"
            and self.tokens[index + 1].type == "IDENTIFIER"
        ):
            index += 2

        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_pos(index)
        while index < len(self.tokens) and self.tokens[index].type in {
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)

        return self.skip_array_suffix_at_pos(index)

    def parse_range_for_statement(self):
        vtype = self.parse_type()
        name = self.consume("IDENTIFIER").value
        self.consume("COLON")
        iterable = self.parse_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return RangeForNode(vtype, name, iterable, body)

    def parse_while_statement(self):
        self.consume("WHILE")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.consume("DO")
        self.skip_newlines()
        body = self.parse_statement()
        self.skip_newlines()
        self.consume("WHILE")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        if self.match("SEMICOLON"):
            self.advance()

        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.consume("SWITCH")
        self.consume("LPAREN")
        expression = self.parse_expression()
        self.consume("RPAREN")
        self.skip_newlines()
        self.consume("LBRACE")

        cases = []
        ordered_cases = []
        default_case = None
        seen_default = False

        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            if self.match("CASE"):
                self.advance()
                value = self.parse_expression()
                self.consume("COLON")
                body = []
                while self.current_token and not self.match(
                    "CASE", "DEFAULT", "RBRACE"
                ):
                    stmt = self.parse_statement()
                    if stmt:
                        body.append(stmt)
                case = CaseNode(value, body)
                cases.append(case)
                ordered_cases.append(case)
            elif self.match("DEFAULT"):
                if seen_default:
                    raise SyntaxError("duplicate default label in switch")
                seen_default = True
                self.advance()
                self.consume("COLON")
                default_case = []
                while self.current_token and not self.match("CASE", "RBRACE"):
                    stmt = self.parse_statement()
                    if stmt:
                        default_case.append(stmt)
                ordered_cases.append(CaseNode(None, default_case))
            else:
                self.advance()

        self.consume("RBRACE")
        switch = SwitchNode(expression, cases, default_case)
        switch.ordered_cases = ordered_cases
        return switch

    def parse_sync_statement(self):
        sync_type = self.current_token.value
        self.advance()

        self.consume("LPAREN")
        args = []
        if not self.match("RPAREN"):
            args.append(self.parse_expression())
            while self.match("COMMA"):
                self.advance()
                args.append(self.parse_expression())
        self.consume("RPAREN")

        if self.match("SEMICOLON"):
            self.advance()

        return SyncNode(sync_type, args)

    def parse_asm_statement(self):
        self.consume("ASM")
        is_volatile = False
        if self.match("VOLATILE"):
            is_volatile = True
            self.advance()

        self.consume("LPAREN")
        self.skip_newlines()
        template = self.parse_asm_string_literal()
        outputs = []
        inputs = []
        clobbers = []

        self.skip_newlines()
        section_index = self.consume_asm_section_delimiters()
        if section_index == 1:
            outputs = self.parse_asm_operands()
        elif section_index == 2:
            inputs = self.parse_asm_operands()
        elif section_index >= 3:
            clobbers = self.parse_asm_clobbers()

        self.skip_newlines()
        while self.match("COLON", "SCOPE"):
            section_index += self.consume_asm_section_delimiters()
            if section_index == 2:
                inputs = self.parse_asm_operands()
            elif section_index >= 3:
                clobbers = self.parse_asm_clobbers()
                break
            self.skip_newlines()

        self.skip_newlines()
        self.consume("RPAREN")
        if self.match("SEMICOLON"):
            self.advance()
        return HipAsmNode(template, outputs, inputs, clobbers, is_volatile)

    def consume_asm_section_delimiters(self):
        delimiter_count = 0
        self.skip_newlines()
        while self.match("COLON", "SCOPE"):
            if self.match("COLON"):
                delimiter_count += 1
                self.advance()
            else:
                delimiter_count += 2
                self.advance()
            self.skip_newlines()
        return delimiter_count

    def parse_asm_string_literal(self):
        value = self.consume("STRING").value
        self.skip_newlines()
        while self.match("STRING"):
            next_value = self.consume("STRING").value
            if value.endswith('"') and next_value.startswith('"'):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
            self.skip_newlines()
        return value

    def parse_asm_operands(self):
        operands = []

        self.skip_newlines()
        while self.current_token and not self.match("COLON", "SCOPE", "RPAREN"):
            symbolic_name = None
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                symbolic_name = self.consume("IDENTIFIER").value
                self.consume("RBRACKET")

            constraint = self.consume("STRING").value
            expression = None
            if self.match("LPAREN"):
                self.consume("LPAREN")
                if not self.match("RPAREN"):
                    expression = self.parse_expression()
                self.consume("RPAREN")

            operands.append(HipAsmOperandNode(constraint, expression, symbolic_name))
            if not self.match("COMMA"):
                break
            self.advance()
            self.skip_newlines()

        return operands

    def parse_asm_clobbers(self):
        clobbers = []

        self.skip_newlines()
        while self.current_token and not self.match("RPAREN"):
            clobbers.append(self.consume("STRING").value)
            if not self.match("COMMA"):
                break
            self.advance()
            self.skip_newlines()

        return clobbers

    def parse_block(self):
        self.consume("LBRACE")
        statements = []

        self.block_depth += 1
        try:
            while self.current_token and not self.match("RBRACE"):
                stmt = self.parse_statement()
                if stmt:
                    if isinstance(stmt, list):
                        statements.extend(stmt)
                    else:
                        statements.append(stmt)
        finally:
            self.block_depth -= 1

        self.consume("RBRACE")
        return statements

    def parse_expression_statement(self):
        expr = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()
            return expr
        if self.is_optional_semicolon_macro_statement(expr):
            self.skip_newlines()
            if self.match("SEMICOLON"):
                self.advance()
            return expr

        self.skip_newlines()
        self.consume("SEMICOLON")

        return expr

    def is_optional_semicolon_macro_statement(self, expr):
        if not self.is_expression_statement_boundary():
            return False
        if not isinstance(expr, FunctionCallNode):
            return False
        name = expr.name
        return (
            isinstance(name, str)
            and name.isupper()
            and any(char.isalpha() for char in name)
        )

    def is_expression_statement_boundary(self):
        if self.match("NEWLINE", "RBRACE"):
            return True
        if self.pos > 0 and self.tokens[self.pos - 1].type == "NEWLINE":
            return True
        return False

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()
        self.skip_newlines()

        if self.match(
            "ASSIGN",
            "PLUS_ASSIGN",
            "MINUS_ASSIGN",
            "MULTIPLY_ASSIGN",
            "DIVIDE_ASSIGN",
            "STAR_ASSIGN",
            "SLASH_ASSIGN",
            "PERCENT_ASSIGN",
            "AND_ASSIGN",
            "OR_ASSIGN",
            "XOR_ASSIGN",
            "LSHIFT_ASSIGN",
            "RSHIFT_ASSIGN",
        ):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_ternary_expression(self):
        expr = self.parse_logical_or_expression()
        self.skip_newlines()

        if self.match("QUESTION"):
            self.advance()
            self.skip_newlines()
            true_expr = self.parse_expression()
            self.skip_newlines()
            self.consume("COLON")
            self.skip_newlines()
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()
        self.skip_newlines()

        while self.match("LOGICAL_OR", "OR"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()
        self.skip_newlines()

        while self.match("LOGICAL_AND", "AND"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()
        self.skip_newlines()

        while self.match("BITWISE_OR", "PIPE"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()
        self.skip_newlines()

        while self.match("BITWISE_XOR", "XOR"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()
        self.skip_newlines()

        while self.match("BITWISE_AND", "AMPERSAND"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()
        self.skip_newlines()

        while self.match("EQ", "NE"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()
        self.skip_newlines()

        while self.match("LT", "LE", "GT", "GE"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_shift_expression(self):
        left = self.parse_additive_expression()
        self.skip_newlines()

        while self.match("SHIFT_LEFT", "SHIFT_RIGHT", "LSHIFT", "RSHIFT"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_additive_expression(self):
        left = self.parse_multiplicative_expression()
        self.skip_newlines()

        while self.match("PLUS", "MINUS"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_unary_expression()
        self.skip_newlines()

        while self.match("MULTIPLY", "STAR", "DIVIDE", "SLASH", "MODULO", "PERCENT"):
            op = self.current_token.value
            self.advance()
            self.skip_newlines()
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)
            self.skip_newlines()

        return left

    def parse_unary_expression(self):
        if self.match(
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
            "TILDE",
            "INCREMENT",
            "DECREMENT",
            "STAR",
            "AMPERSAND",
        ):
            op = self.current_token.value
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        expr = self.parse_primary_expression()

        while True:
            self.skip_newlines()
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                index = self.parse_expression()
                self.consume("RBRACKET")
                expr = ArrayAccessNode(expr, index)
            elif self.match("SCOPE"):
                self.consume("SCOPE")
                member = self.consume_qualified_name_member()
                expr = self.append_qualified_name(expr, "::", member)
            elif self.match("LT") and self.is_template_suffix():
                expr = self.append_template_suffix(expr)
            elif self.match("DOT"):
                self.consume("DOT")
                member = self.parse_member_name()
                expr = MemberAccessNode(expr, member, False)
            elif self.match("ARROW"):
                self.consume("ARROW")
                member = self.parse_member_name()
                expr = MemberAccessNode(expr, member, True)
            elif self.match("LPAREN"):
                self.consume("LPAREN")
                self.skip_newlines()
                if expr == "sizeof" and self.is_sizeof_type_operand():
                    args = [self.parse_type()]
                    self.skip_newlines()
                else:
                    args = self.parse_argument_list()
                self.consume("RPAREN")
                expr = self.parse_function_call_node(expr, args)
            elif self.match("LBRACE"):
                initializer = self.parse_initializer_list()
                expr = self.parse_function_call_node(expr, initializer.elements)
            elif self.match("KERNEL_LAUNCH_START"):
                expr = self.parse_kernel_launch(expr)
            elif self.match("ELLIPSIS"):
                self.advance()
                expr = FunctionCallNode(self.PACK_EXPANSION_FUNCTION_NAME, [expr])
            elif self.match("INCREMENT", "DECREMENT"):
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

    def parse_member_name(self):
        if self.match("TEMPLATE"):
            self.advance()
            return self.consume_function_name()
        if self.match(*self.MEMBER_NAME_TOKENS):
            member = self.current_token.value
            self.advance()
            return member
        self.error("Expected member name")

    def consume_qualified_name_member(self):
        if self.match("TEMPLATE"):
            self.advance()
            return self.consume_function_name()
        if self.match("IDENTIFIER") or self.is_type_token(allow_identifier=False):
            member = self.current_token.value
            self.advance()
            return member
        self.error("Expected qualified name member")

    def is_sizeof_type_operand(self):
        saved_pos = self.pos
        try:
            self.skip_newlines()
            saw_integral_sign = False
            while self.match(*self.TYPE_QUALIFIER_TOKENS):
                if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                    saw_integral_sign = True
                self.advance()

            if saw_integral_sign and self.match("RPAREN"):
                return True

            if not self.is_type_token(allow_identifier=False) and not (
                self.match("IDENTIFIER")
                and self.current_token.value in self.type_aliases
            ):
                return False

            token_type = self.current_token.type
            self.advance()
            if token_type in self.ELABORATED_TYPE_TOKENS:
                if token_type == "ENUM" and self.match("CLASS", "STRUCT"):
                    self.advance()
                if not self.match("IDENTIFIER"):
                    return False
                self.advance()
            if token_type == "LONG" and self.match("LONG"):
                self.advance()

            while self.match("ASTERISK", "STAR"):
                self.advance()

            while self.match("LBRACKET"):
                self.advance()
                while self.current_token and not self.match("RBRACKET"):
                    self.advance()
                if not self.current_token:
                    return False
                self.advance()

            self.skip_newlines()
            return self.match("RPAREN")
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

    def append_qualified_name(self, base, separator, member):
        if isinstance(base, str):
            return f"{base}{separator}{member}"
        return f"{self.expression_to_text(base)}{separator}{member}"

    def is_template_suffix(self):
        index = self.pos
        if self.tokens[index].type != "LT":
            return False

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LT":
                depth += 1
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
            elif token_type == "RSHIFT":
                depth -= 2
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
                if depth < 0:
                    return False
            elif token_type == "KERNEL_LAUNCH_END":
                depth -= 3
                if depth == 0:
                    next_index = index + 1
                    while (
                        next_index < len(self.tokens)
                        and self.tokens[next_index].type == "NEWLINE"
                    ):
                        next_index += 1
                    next_type = (
                        self.tokens[next_index].type
                        if next_index < len(self.tokens)
                        else "EOF"
                    )
                    return next_type in self.TEMPLATE_SUFFIX_FOLLOW_TOKENS
                if depth < 0:
                    return False
            elif token_type in {"SEMICOLON", "ASSIGN"}:
                return False
            index += 1

        return False

    def append_template_suffix(self, base):
        suffix = self.parse_template_suffix()
        if isinstance(base, str):
            return f"{base}{suffix}"
        return f"{self.expression_to_text(base)}{suffix}"

    def parse_template_suffix(self):
        self.consume("LT")
        parts = []
        depth = 1

        while depth > 0:
            token_type = self.current_token.type
            token_value = self.current_token.value
            if token_type == "LT":
                depth += 1
                parts.append(token_value)
                self.consume("LT")
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    self.consume("GT")
                    break
                parts.append(token_value)
                self.consume("GT")
            elif token_type == "RSHIFT":
                self.consume("RSHIFT")
                for _ in range(2):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            elif token_type == "KERNEL_LAUNCH_END":
                self.consume("KERNEL_LAUNCH_END")
                for _ in range(3):
                    depth -= 1
                    if depth == 0:
                        break
                    parts.append(">")
            else:
                parts.append(token_value)
                self.consume(token_type)

        return f"<{self.format_template_parts(parts)}>"

    def format_template_parts(self, parts):
        formatted = []
        previous = None

        for part in parts:
            if part == "\n":
                continue
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
            function_name in self.ATOMIC_FUNCTION_NAMES
            and function_name not in self.user_function_names
        ):
            return AtomicOperationNode(function_name, args)

        if (
            function_name == "hipLaunchKernelGGL"
            and len(args) >= 5
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_hip_kernel_function_arg(args[0]),
                args[1],
                args[2],
                args[3],
                args[4],
                args[5:],
            )

        if (
            function_name == "hipLaunchKernel"
            and len(args) == 6
            and function_name not in self.user_function_names
        ):
            return KernelLaunchNode(
                self.unwrap_hip_kernel_function_arg(args[0]),
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

    def unwrap_hip_kernel_function_arg(self, function_arg):
        if isinstance(function_arg, CastNode):
            return self.unwrap_hip_kernel_function_arg(function_arg.expression)
        if (
            isinstance(function_arg, FunctionCallNode)
            and function_arg.name == "HIP_KERNEL_NAME"
            and len(function_arg.args) == 1
        ):
            return function_arg.args[0]
        return function_arg

    def parse_kernel_launch(self, kernel_name):
        kernel_name = self.unwrap_hip_kernel_function_arg(kernel_name)
        self.consume("KERNEL_LAUNCH_START")

        self.skip_newlines()
        blocks = self.parse_expression()
        self.skip_newlines()
        self.consume("COMMA")
        self.skip_newlines()
        threads = self.parse_expression()
        self.skip_newlines()

        shared_mem = None
        stream = None

        if self.match("COMMA"):
            self.advance()
            self.skip_newlines()
            shared_mem = self.parse_expression()
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
                stream = self.parse_expression()
                self.skip_newlines()

        self.consume("KERNEL_LAUNCH_END")

        self.skip_newlines()
        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")

        return KernelLaunchNode(kernel_name, blocks, threads, shared_mem, stream, args)

    def parse_primary_expression(self):
        if self.match("LBRACKET") and self.is_lambda_expression_start():
            return self.parse_lambda_expression()

        if self.match("IDENTIFIER", *self.CONTEXTUAL_IDENTIFIER_TOKENS):
            if self.current_token.value == "new":
                return self.parse_new_expression()

            name = self.current_token.value
            self.advance()

            if name in ["threadIdx", "blockIdx", "blockDim", "gridDim"]:
                component = None
                if self.match("DOT"):
                    self.advance()
                    if self.match("IDENTIFIER"):
                        component = self.current_token.value
                        self.advance()
                return HipBuiltinNode(name, component)

            return name

        elif self.match("THREADIDX", "BLOCKIDX", "BLOCKDIM", "GRIDDIM"):
            name = self.current_token.value
            self.advance()

            component = None
            if self.match("DOT"):
                self.advance()
                if self.match("IDENTIFIER"):
                    component = self.current_token.value
                    self.advance()

            return HipBuiltinNode(name, component)

        elif self.match(*self.FLAT_BUILTIN_TOKEN_MAP):
            return self.parse_flat_builtin_node()

        elif self.match("WARPSIZE"):
            self.advance()
            return HipBuiltinNode("warpSize")

        elif self.match(*self.ATOMIC_FUNCTION_TOKENS):
            name = self.current_token.value
            self.advance()
            return name

        elif self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume_qualified_name_member()
            return f"::{member}"

        elif self.is_type_constructor_expression_start():
            return self.parse_type_without_array_suffix()

        elif self.match("STRING"):
            return self.parse_string_literal_sequence()

        elif self.match("INTEGER", "FLOAT_NUM", "FLOAT"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("TRUE", "FALSE", "NULL", "NULLPTR", "HIPSUCCESS"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("CHAR_LIT"):
            return self.parse_character_literal()

        elif self.match("SYNCTHREADS", "SYNCWARP"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("LPAREN"):
            if self.is_cast_expression():
                self.consume("LPAREN")
                target_type = self.parse_type()
                self.consume("RPAREN")
                self.skip_newlines()
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.consume("LPAREN")
            self.skip_newlines()
            expr = self.parse_expression()
            self.consume("RPAREN")
            return expr
        elif self.match("LBRACE"):
            return self.parse_initializer_list()

        else:
            self.error(
                f"Unexpected token in expression: {self.current_token.type if self.current_token else 'EOF'}"
            )

    def is_lambda_expression_start(self):
        if not self.match("LBRACKET"):
            return False

        index = self.pos
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
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

        index = self.skip_lambda_specifiers_at_pos(index)
        index = self.skip_lambda_template_parameters_at_pos(index)
        index = self.skip_lambda_specifiers_at_pos(index)
        return index < len(self.tokens) and self.tokens[index].type in {
            "LPAREN",
            "LBRACE",
        }

    def skip_lambda_specifiers_at_pos(self, index):
        while index < len(self.tokens):
            token = self.tokens[index]
            if token.type == "NEWLINE":
                index += 1
                continue
            if token.type in self.LAMBDA_SPECIFIER_TOKENS or (
                token.type == "IDENTIFIER"
                and token.value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                index += 1
                if (
                    token.value == "noexcept"
                    and index < len(self.tokens)
                    and self.tokens[index].type == "LPAREN"
                ):
                    index = self.skip_balanced_tokens_at_pos(index, "LPAREN", "RPAREN")
                continue
            break
        return index

    def skip_lambda_template_parameters_at_pos(self, index):
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            skipped = self.skip_template_at_pos(index)
            if skipped is not None:
                return skipped
        return index

    def skip_balanced_tokens_at_pos(self, index, open_token, close_token):
        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
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
        self.skip_newlines()
        self.skip_lambda_specifiers()
        self.skip_lambda_template_parameters()
        self.skip_lambda_specifiers()

        args = []
        if self.match("LPAREN"):
            self.consume("LPAREN")
            self.skip_newlines()
            while not self.match("RPAREN"):
                args.append(self.parse_lambda_parameter())
                self.skip_newlines()
                if self.match("COMMA"):
                    self.advance()
                    self.skip_newlines()
                elif not self.match("RPAREN"):
                    self.error(
                        f"Expected COMMA or RPAREN in lambda parameters, got {self.current_token.type if self.current_token else 'EOF'}"
                    )
            self.consume("RPAREN")

        self.skip_newlines()
        self.skip_lambda_specifiers()
        if self.match("ARROW"):
            self.skip_lambda_trailing_return_type()
            self.skip_newlines()
            self.skip_lambda_specifiers()

        args.append(self.parse_lambda_block_body())
        return FunctionCallNode("lambda", args)

    def skip_lambda_specifiers(self):
        while self.current_token:
            if self.match("NEWLINE"):
                self.advance()
                continue
            if self.current_token.type in self.LAMBDA_SPECIFIER_TOKENS or (
                self.current_token.type == "IDENTIFIER"
                and self.current_token.value in self.LAMBDA_IDENTIFIER_SPECIFIERS
            ):
                token_value = self.current_token.value
                self.advance()
                if token_value == "noexcept" and self.match("LPAREN"):
                    self.consume_balanced_lambda_tokens("LPAREN", "RPAREN")
                continue
            break

    def skip_lambda_template_parameters(self):
        if self.match("LT"):
            self.parse_template_suffix()
            self.skip_newlines()

    def skip_lambda_trailing_return_type(self):
        self.consume("ARROW")
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type == "LBRACE"
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.advance()

    def parse_lambda_parameter(self):
        saved_pos = self.pos
        try:
            param_type = self.parse_type()
            if not self.match("IDENTIFIER"):
                self.error("Expected lambda parameter name")
            param_name = self.consume("IDENTIFIER").value
            param_type += self.parse_array_suffix()
            self.skip_lambda_parameter_default()
            return VariableNode(param_type, param_name)
        except SyntaxError:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )
            raw = self.collect_lambda_parameter_raw()
            return VariableNode("", raw)

    def skip_lambda_parameter_default(self):
        if not self.match("ASSIGN"):
            return

        self.advance()
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
                and brace_depth == 0
            ):
                break
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
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
            self.advance()

    def collect_lambda_parameter_raw(self):
        tokens = []
        angle_depth = 0
        paren_depth = 0
        bracket_depth = 0

        while self.current_token:
            token_type = self.current_token.type
            if (
                token_type in {"COMMA", "RPAREN"}
                and angle_depth == 0
                and paren_depth == 0
                and bracket_depth == 0
            ):
                break

            tokens.append(self.current_token)
            if token_type == "LT":
                angle_depth += 1
            elif token_type == "GT" and angle_depth > 0:
                angle_depth -= 1
            elif token_type == "LPAREN":
                paren_depth += 1
            elif token_type == "RPAREN" and paren_depth > 0:
                paren_depth -= 1
            elif token_type == "LBRACKET":
                bracket_depth += 1
            elif token_type == "RBRACKET" and bracket_depth > 0:
                bracket_depth -= 1
            self.advance()

        raw = self.format_lambda_raw_tokens(tokens).strip()
        if not raw:
            self.error("Expected lambda parameter")
        return raw

    def parse_lambda_block_body(self):
        expression_body = self.try_parse_lambda_return_expression()
        if expression_body is not None:
            return expression_body

        return self.parse_raw_lambda_block_body()

    def try_parse_lambda_return_expression(self):
        saved_pos = self.pos
        completed = False
        try:
            self.consume("LBRACE")
            self.skip_newlines()
            if not self.match("RETURN"):
                return None
            self.consume("RETURN")
            self.skip_newlines()
            if self.match("SEMICOLON"):
                return None
            value = self.parse_expression()
            self.skip_newlines()
            if not self.match("SEMICOLON"):
                return None
            self.consume("SEMICOLON")
            self.skip_newlines()
            if not self.match("RBRACE"):
                return None
            self.consume("RBRACE")
            completed = True
            return value
        except SyntaxError:
            return None
        finally:
            if not completed:
                self.pos = saved_pos
                self.current_token = self.tokens[self.pos]

    def parse_raw_lambda_block_body(self):
        tokens = []
        depth = 0
        while self.current_token:
            token = self.current_token
            token_type = token.type
            tokens.append(token)
            if token_type == "LBRACE":
                depth += 1
            elif token_type == "RBRACE":
                depth -= 1
                self.consume("RBRACE")
                if depth == 0:
                    return self.format_lambda_raw_tokens(tokens)
                continue
            self.advance()

        raise SyntaxError("Unterminated lambda block body")

    def consume_balanced_lambda_tokens(self, open_token, close_token):
        depth = 0
        while self.current_token:
            token_type = self.current_token.type
            if token_type == open_token:
                depth += 1
            elif token_type == close_token:
                depth -= 1
                self.consume(close_token)
                if depth == 0:
                    return
                continue
            self.advance()

        raise SyntaxError(f"Unterminated lambda {open_token}")

    def format_lambda_raw_tokens(self, tokens):
        text = ""
        previous = None
        for token in tokens:
            token_type = token.type
            value = str(token.value)
            if token_type == "NEWLINE":
                continue
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
            elif previous and previous.value in {"(", "[", ".", "::", "->"}:
                text += value
            elif text.endswith((" ", "{")):
                text += value
            else:
                text += " " + value
            previous = token
        return text.strip()

    def parse_new_expression(self):
        self.advance()
        placement_args = None
        if self.match("LPAREN"):
            placement_args = self.parse_parenthesized_argument_list()
            self.skip_newlines()

        target_type = self.parse_type_without_array_suffix()

        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            size = None
            if not self.match("RBRACKET"):
                size = self.parse_expression()
            self.consume("RBRACKET")
            node = NewNode(target_type, size=size, is_array=True)
            node.placement_args = placement_args
            return node

        args = []
        if self.match("LPAREN"):
            self.consume("LPAREN")
            args = self.parse_argument_list()
            self.consume("RPAREN")

        node = NewNode(target_type, args=args)
        node.placement_args = placement_args
        return node

    def parse_initializer_list(self):
        self.consume("LBRACE")
        elements = []

        self.skip_newlines()
        while self.current_token and not self.match("RBRACE"):
            elements.append(self.parse_initializer_element())
            self.skip_newlines()
            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
                if self.match("RBRACE"):
                    break
            else:
                break

        self.consume("RBRACE")
        return InitializerListNode(elements)

    def parse_initializer_element(self):
        if self.match("LBRACKET", "DOT"):
            return self.parse_designated_initializer()
        return self.parse_expression()

    def parse_string_literal_sequence(self):
        token = self.consume("STRING")
        value = token.value + self.parse_user_defined_literal_suffix(token)
        self.skip_newlines()

        while self.match("STRING"):
            next_token = self.consume("STRING")
            next_value = next_token.value + self.parse_user_defined_literal_suffix(
                next_token
            )
            if value.endswith('"') and next_value.startswith('"'):
                value = value[:-1] + next_value[1:]
            else:
                value += next_value
            self.skip_newlines()

        return value

    def parse_character_literal(self):
        token = self.consume("CHAR_LIT")
        return token.value + self.parse_user_defined_literal_suffix(token)

    def parse_user_defined_literal_suffix(self, literal_token):
        if not self.match("IDENTIFIER"):
            return ""
        if not self.is_adjacent_to_literal_token(literal_token, self.current_token):
            return ""

        suffix = self.current_token.value
        self.advance()
        return suffix

    def is_adjacent_to_literal_token(self, literal_token, next_token):
        newline_count = literal_token.value.count("\n")
        if newline_count:
            end_line = literal_token.line + newline_count
            end_column = len(literal_token.value.rsplit("\n", 1)[1]) + 1
        else:
            end_line = literal_token.line
            end_column = literal_token.column + len(literal_token.value)

        return next_token.line == end_line and next_token.column == end_column

    def parse_designated_initializer(self):
        designators = []

        while self.match("LBRACKET", "DOT"):
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                index = self.parse_expression()
                self.consume("RBRACKET")
                designators.append(("index", index))
            else:
                self.consume("DOT")
                field = self.consume("IDENTIFIER").value
                designators.append(("field", field))
            self.skip_newlines()

        self.consume("ASSIGN")
        self.skip_newlines()
        value = self.parse_expression()
        return DesignatedInitializerNode(designators, value)

    def expression_to_text(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, HipBuiltinNode):
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

    def is_cast_expression(self):
        if not self.match("LPAREN"):
            return False

        type_start = self.skip_newlines_at_pos(self.pos + 1)
        type_end = self.skip_type_at_pos(
            type_start,
            allow_unknown_identifier_pointers=True,
        )
        if (
            type_end is not None
            and type_end + 1 < len(self.tokens)
            and self.tokens[type_end].type == "RPAREN"
            and self.tokens[type_end + 1].type in {"COMMA", "RPAREN", "SEMICOLON"}
        ):
            return False
        if self.is_unknown_identifier_cast_followed_by_ambiguous_unary(
            type_start, type_end
        ):
            return False
        return (
            type_end is not None
            and type_end < len(self.tokens)
            and self.tokens[type_end].type == "RPAREN"
            and self.is_cast_type_sequence(type_start, type_end)
        )

    def is_unknown_identifier_cast_followed_by_ambiguous_unary(self, start, end):
        if end is None or end >= len(self.tokens):
            return False

        type_tokens = [
            token for token in self.tokens[start:end] if token.type != "NEWLINE"
        ]
        if len(type_tokens) != 1 or type_tokens[0].type != "IDENTIFIER":
            return False
        if self.is_identifier_type_name(type_tokens[0].value):
            return False

        operand_index = self.skip_newlines_at_pos(end + 1)
        return operand_index < len(self.tokens) and self.tokens[operand_index].type in {
            "STAR",
            "AMPERSAND",
        }

    def is_cast_type_sequence(self, start, end):
        index = start
        while index < end and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS:
            index += 1

        if index >= end:
            return False

        token = self.tokens[index]
        if token.type in self.ELABORATED_TYPE_TOKENS:
            return True
        if self.is_decltype_type_start_at_pos(index):
            return True
        if self.is_type_token(token, allow_identifier=False):
            return True

        if token.type != "IDENTIFIER":
            return False

        type_tokens = self.tokens[index:end]
        return (
            self.is_identifier_type_name(token.value)
            or self.is_probable_identifier_type_name(token.value)
            or any(item.type == "SCOPE" for item in type_tokens)
            or any(
                item.type in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}
                for item in type_tokens
            )
        )

    def parse_argument_list(self):
        args = []

        self.skip_newlines()
        if self.match("RPAREN"):
            return args

        while True:
            self.skip_newlines()
            arg = self.parse_expression()
            args.append(arg)
            self.skip_newlines()

            if self.match("COMMA"):
                self.advance()
            else:
                break

        return args

    def is_function_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier followed by (
        index = self.pos

        index = self.skip_cpp_attributes_at_pos(index)
        while index < len(self.tokens) and (
            self.tokens[index].type in self.FUNCTION_DECLARATION_SPECIFIER_TOKENS
            or self.is_identifier_function_specifier_token(self.tokens[index])
        ):
            index += 1
            index = self.skip_cpp_attributes_at_pos(index)

        index = self.skip_type_at_pos(
            index, allow_unknown_identifier_pointers=self.block_depth == 0
        )
        if index is not None:
            index = self.skip_newlines_at_pos(index)
            while index < len(
                self.tokens
            ) and self.is_identifier_function_specifier_token(self.tokens[index]):
                index += 1
                index = self.skip_newlines_at_pos(index)
            parameter_list_start = self.skip_function_name_at(index)
            if (
                parameter_list_start is not None
                and self.is_plausible_function_parameter_list_at_pos(
                    parameter_list_start
                )
            ):
                return True

        return False

    def is_plausible_function_parameter_list_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return False

        index += 1
        index = self.skip_newlines_at_pos(index)
        index = self.skip_cpp_attributes_at_pos(index)
        if index >= len(self.tokens):
            return False

        token_type = self.tokens[index].type
        if token_type == "RPAREN":
            return True
        if (
            token_type == "VOID"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RPAREN"
        ):
            return True
        if token_type == "ELLIPSIS":
            return True
        if self.is_function_pointer_parameter_start_at_pos(index):
            return True

        return self.is_plausible_function_parameter_at_pos(index)

    def is_plausible_function_parameter_at_pos(self, index):
        type_end = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if type_end is None:
            return False

        index = self.skip_newlines_at_pos(type_end)
        index = self.skip_cpp_attributes_at_pos(index)
        while index < len(self.tokens) and self.tokens[index].type in {
            "ASTERISK",
            "STAR",
            *self.TYPE_REFERENCE_TOKENS,
            *self.POSTFIX_TYPE_QUALIFIER_TOKENS,
        }:
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)
            index = self.skip_newlines_at_pos(index)

        if index < len(self.tokens) and self.tokens[index].type == "ELLIPSIS":
            index += 1
            index = self.skip_newlines_at_pos(index)

        if index >= len(self.tokens):
            return False

        if self.tokens[index].type in {"COMMA", "RPAREN", "ASSIGN"}:
            return True

        if self.is_declarator_name_token_at(index):
            index = self.skip_variable_declarator_name_at_pos(index)
            if index is None:
                return False
            index = self.skip_newlines_at_pos(index)
            index = self.skip_cpp_attributes_at_pos(index)
            index = self.skip_array_suffix_at_pos(index)
            index = self.skip_newlines_at_pos(index)
            return index < len(self.tokens) and self.tokens[index].type in {
                "COMMA",
                "RPAREN",
                "ASSIGN",
            }

        return False

    def is_function_pointer_parameter_start_at_pos(self, index):
        type_end = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        return (
            type_end is not None
            and type_end + 2 < len(self.tokens)
            and self.tokens[type_end].type == "LPAREN"
            and self.tokens[type_end + 1].type
            in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}
            and self.is_declarator_name_token_at(type_end + 2)
        )

    def skip_function_name_at(self, index):
        if index >= len(self.tokens) or not self.is_function_name_token(
            self.tokens[index]
        ):
            return None

        name = self.tokens[index].value
        index += 1
        index = self.skip_operator_function_suffix_at(index, name)
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while index < len(self.tokens) and self.tokens[index].type == "SCOPE":
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "TILDE":
                index += 1
            if index >= len(self.tokens) or not self.is_function_name_token(
                self.tokens[index]
            ):
                return None
            name = self.tokens[index].value
            index += 1
            index = self.skip_operator_function_suffix_at(index, name)
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        if index < len(self.tokens) and self.tokens[index].type == "LPAREN":
            return index
        return None

    def skip_operator_function_suffix_at(self, index, name):
        if name != "operator" or index >= len(self.tokens):
            return index

        if (
            self.tokens[index].type == "LBRACKET"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RBRACKET"
        ):
            return index + 2
        if (
            self.tokens[index].type == "LPAREN"
            and index + 1 < len(self.tokens)
            and self.tokens[index + 1].type == "RPAREN"
        ):
            return index + 2
        if self.tokens[index].type in self.OVERLOADABLE_OPERATOR_TOKENS:
            return index + 1
        return index

    def is_variable_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier not followed by (
        index = self.pos

        while index < len(self.tokens) and self.tokens[index].type in {
            "__SHARED__",
            "__CONSTANT__",
            "__MANAGED__",
            "__DEVICE__",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
            "CONST",
            "VOLATILE",
        }:
            index += 1

        index = self.skip_cpp_attributes_at_pos(index)
        index = self.skip_type_attribute_prefixes_at_pos(index)
        index = self.skip_type_at_pos(index)
        if index is not None:
            while (
                index < len(self.tokens)
                and self.tokens[index].type in self.DECLARATION_QUALIFIER_TOKENS
            ):
                index += 1
            index = self.skip_newlines_at_pos(index)
            structured_binding_end = self.skip_structured_binding_declarator_at_pos(
                index
            )
            if structured_binding_end is not None:
                return True

            declarator_name_end = self.skip_variable_declarator_name_at_pos(index)
            if declarator_name_end is not None:
                index = declarator_name_end
                index = self.skip_newlines_at_pos(index)
                index = self.skip_array_suffix_at_pos(index)
                index = self.skip_newlines_at_pos(index)
                index = self.skip_type_attribute_prefixes_at_pos(index)
                index = self.skip_newlines_at_pos(index)
                if index < len(self.tokens) and self.tokens[index].type in {
                    "SEMICOLON",
                    "ASSIGN",
                    "LBRACKET",
                    "LPAREN",
                    "LBRACE",
                    "COMMA",
                }:
                    return True

        return False

    def skip_variable_declarator_name_at_pos(self, index):
        if not self.is_declarator_name_token_at(index):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while index < len(self.tokens) and self.tokens[index].type == "SCOPE":
            index += 1
            if not self.is_declarator_name_token_at(index):
                return None
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        return index

    def skip_type_attribute_prefixes_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type == "IDENTIFIER"
            and self.tokens[index].value in self.TYPE_ATTRIBUTE_IDENTIFIERS
        ):
            index += 1
            if index < len(self.tokens) and self.tokens[index].type == "LPAREN":
                index = self.skip_balanced_tokens_at_pos(index, "LPAREN", "RPAREN")
            index = self.skip_newlines_at_pos(index)
        return index

    def skip_cpp_attributes_at_pos(self, index):
        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "LBRACKET"
            and self.tokens[index + 1].type == "LBRACKET"
        ):
            index += 2
            depth = 1

            while index < len(self.tokens) and depth > 0:
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index].type == "LBRACKET"
                    and self.tokens[index + 1].type == "LBRACKET"
                ):
                    index += 2
                    depth += 1
                    continue
                if (
                    index + 1 < len(self.tokens)
                    and self.tokens[index].type == "RBRACKET"
                    and self.tokens[index + 1].type == "RBRACKET"
                ):
                    index += 2
                    depth -= 1
                    continue
                index += 1

            index = self.skip_newlines_at_pos(index)

        return index

    def skip_type_at_pos(self, index, allow_unknown_identifier_pointers=False):
        saw_integral_sign = False
        index = self.skip_cpp_attributes_at_pos(index)
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_PREFIX_TOKENS
        ):
            index += 1

        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            if self.tokens[index].type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        index = self.skip_cpp_attributes_at_pos(index)
        if index >= len(self.tokens):
            return None

        if self.is_implicit_int_type_at_pos(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.is_decltype_type_start_at_pos(index):
            type_token = "IDENTIFIER"
            type_value = "decltype"
            index = self.skip_decltype_type_at_pos(index)
            if index is None:
                return None
        elif self.is_type_token(self.tokens[index]):
            type_token = self.tokens[index].type
            type_value = self.tokens[index].value
            index += 1
            if type_token in self.ELABORATED_TYPE_TOKENS:
                index = self.skip_elaborated_type_name_at_pos(
                    index, keyword_token_type=type_token, allow_scoped_suffix=True
                )
                if index is None:
                    return None
            if type_token == "LONG" and index < len(self.tokens):
                if self.tokens[index].type == "LONG":
                    index += 1
                if index < len(self.tokens) and self.tokens[index].type == "INT":
                    index += 1
        else:
            return None

        has_qualified_suffix = False

        if index < len(self.tokens) and self.tokens[index].type == "LT":
            has_qualified_suffix = True
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "SCOPE"
            and self.is_qualified_type_member_token_at(index + 1)
        ):
            has_qualified_suffix = True
            index += 2
            if index < len(self.tokens) and self.tokens[index].type == "LT":
                index = self.skip_template_at_pos(index)
                if index is None:
                    return None

        index = self.skip_postfix_type_qualifiers_at_pos(index)
        can_have_pointer_suffix = (
            allow_unknown_identifier_pointers
            or type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or type_value == "decltype"
            or type_token in self.ELABORATED_TYPE_TOKENS
            or self.is_identifier_type_name(type_value)
            or self.is_probable_identifier_type_name(type_value)
        )
        while (
            can_have_pointer_suffix
            and index < len(self.tokens)
            and self.tokens[index].type
            in {
                "ASTERISK",
                "STAR",
                *self.TYPE_REFERENCE_TOKENS,
            }
        ):
            index += 1
            index = self.skip_postfix_type_qualifiers_at_pos(index)

        if type_value == "auto" and self.is_structured_binding_declarator_at_pos(index):
            return index

        return self.skip_array_suffix_at_pos(index)

    def skip_elaborated_type_name_at_pos(
        self, index, keyword_token_type=None, allow_scoped_suffix=False
    ):
        index = self.skip_newlines_at_pos(index)
        if (
            index < len(self.tokens)
            and keyword_token_type == "ENUM"
            and self.tokens[index].type in {"CLASS", "STRUCT"}
        ):
            index += 1
            index = self.skip_newlines_at_pos(index)

        if index >= len(self.tokens) or not self.is_qualified_type_member_token_at(
            index
        ):
            return None

        index += 1
        if index < len(self.tokens) and self.tokens[index].type == "LT":
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        if allow_scoped_suffix:
            while (
                index + 1 < len(self.tokens)
                and self.tokens[index].type == "SCOPE"
                and self.is_qualified_type_member_token_at(index + 1)
            ):
                index += 2
                if index < len(self.tokens) and self.tokens[index].type == "LT":
                    index = self.skip_template_at_pos(index)
                    if index is None:
                        return None

        return index

    def is_qualified_type_member_token_at(self, index):
        return index < len(self.tokens) and (
            self.tokens[index].type == "IDENTIFIER"
            or self.is_type_token(self.tokens[index], allow_identifier=False)
        )

    def skip_structured_binding_at_pos(self, index):
        if index >= len(self.tokens) or self.tokens[index].type != "LBRACKET":
            return None

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LBRACKET":
                depth += 1
            elif token_type == "RBRACKET":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type in {"SEMICOLON", "LBRACE", "RBRACE"} and depth == 0:
                return None
            index += 1

        return None

    def is_structured_binding_declarator_at_pos(self, index):
        return self.skip_structured_binding_declarator_at_pos(index) is not None

    def skip_structured_binding_declarator_at_pos(self, index):
        binding_end = self.skip_structured_binding_at_pos(index)
        if binding_end is None:
            return None

        binding_end = self.skip_newlines_at_pos(binding_end)
        if binding_end < len(self.tokens) and self.tokens[binding_end].type in {
            "ASSIGN",
            "SEMICOLON",
        }:
            return binding_end
        return None

    def is_implicit_int_type_at_pos(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index].type
        if token_type in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}:
            return True

        if token_type != "IDENTIFIER":
            return False

        if self.is_identifier_type_name(self.tokens[index].value):
            return False

        next_type = (
            self.tokens[index + 1].type if index + 1 < len(self.tokens) else "EOF"
        )
        return next_type in {
            "SEMICOLON",
            "ASSIGN",
            "LBRACKET",
            "LPAREN",
            "LBRACE",
            "COMMA",
        }

    def is_hip_opaque_handle_type(self, type_name):
        return (
            isinstance(type_name, str)
            and type_name.startswith("hip")
            and type_name.endswith("_t")
        )

    def is_identifier_type_name(self, type_name):
        return (
            type_name in self.type_aliases
            or type_name in self.HIP_IDENTIFIER_TYPE_NAMES
            or self.is_hip_opaque_handle_type(type_name)
        )

    def is_probable_identifier_type_name(self, type_name):
        return isinstance(type_name, str) and type_name[:1].isupper()

    def skip_postfix_type_qualifiers_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.POSTFIX_TYPE_QUALIFIER_TOKENS
        ):
            index += 1
        return index

    def skip_linkage_specifier_at_pos(self, index):
        if (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "EXTERN"
            and self.tokens[index + 1].type == "STRING"
        ):
            return index + 2
        return None

    def skip_launch_bounds_at_pos(self, index):
        index += 1
        if index >= len(self.tokens) or self.tokens[index].type != "LPAREN":
            return index

        depth = 0
        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LPAREN":
                depth += 1
            elif token_type == "RPAREN":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type in {"SEMICOLON", "LBRACE", "RBRACE"}:
                return index
            index += 1

        return index

    def skip_template_at_pos(self, index):
        depth = 0

        while index < len(self.tokens):
            token_type = self.tokens[index].type
            if token_type == "LT":
                depth += 1
            elif token_type == "GT":
                depth -= 1
                if depth == 0:
                    return index + 1
            elif token_type == "RSHIFT":
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
            elif token_type in {"SEMICOLON", "ASSIGN"}:
                return None
            index += 1

        return None

    def skip_array_suffix_at_pos(self, index):
        while index < len(self.tokens) and self.tokens[index].type == "LBRACKET":
            index += 1
            while index < len(self.tokens) and self.tokens[index].type != "RBRACKET":
                index += 1
            if index < len(self.tokens) and self.tokens[index].type == "RBRACKET":
                index += 1
            else:
                break

        return index


def parse_hip_code(code: str) -> HipProgramNode:
    """Parse HIP source text and return the backend AST."""
    lexer = HipLexer(code)
    tokens = lexer.tokenize()
    parser = HipParser(tokens)
    return parser.parse()
