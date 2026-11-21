"""HIP Parser for converting HIP tokens to AST"""

from typing import List

from crosstl.translator.refusals import HipSyntaxError

from .HipAst import ASTNode, BreakNode, ContinueNode
from .HipLexer import HipLexer, Token
from .parsing.aliases import HipAliasMixin
from .parsing.control_flow import HipControlFlowMixin
from .parsing.declarator_names import HipDeclaratorNameMixin
from .parsing.expressions import HipExpressionMixin
from .parsing.functions import HipFunctionMixin
from .parsing.initializers import HipInitializerMixin
from .parsing.lambdas import HipLambdaMixin
from .parsing.lookahead import HipLookaheadMixin
from .parsing.macros import HipMacroMixin
from .parsing.records import HipRecordMixin
from .parsing.special_statements import HipSpecialStatementMixin
from .parsing.types import HipTypeMixin
from .parsing.variable_declarations import HipVariableDeclarationMixin


class HipProgramNode(ASTNode):
    """Root node representing a complete HIP program"""

    def __init__(self, statements=None):
        self.statements = statements or []
        self.namespace_aliases = {}

    def __repr__(self):
        return f"HipProgramNode(statements={self.statements})"


class HipParser(
    HipDeclaratorNameMixin,
    HipMacroMixin,
    HipAliasMixin,
    HipSpecialStatementMixin,
    HipFunctionMixin,
    HipRecordMixin,
    HipVariableDeclarationMixin,
    HipTypeMixin,
    HipControlFlowMixin,
    HipExpressionMixin,
    HipLambdaMixin,
    HipInitializerMixin,
    HipLookaheadMixin,
):
    """Parse HIP tokens into the HIP backend AST."""

    # The typed syntax error of the source language; the OpenCL parser, which extends this one, names its own.
    syntax_error = HipSyntaxError

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
        "_CCCL_API",
        "_CCCL_CONSTEXPR_CXX20",
        "_CCCL_EXEC_CHECK_DISABLE",
        "_CCCL_HIDE_FROM_ABI",
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
        "ROCWMMA_DEVICE",
        "ROCWMMA_FORCE_INLINE",
        "ROCWMMA_HOST",
        "ROCWMMA_HOST_DEVICE",
        "ROCWMMA_INLINE",
        "ROCWMMA_KERNEL",
        "THRUST_DEVICE",
        "THRUST_DEVICE_FUNCTION",
        "THRUST_FORCEINLINE",
        "THRUST_FUNCTION",
        "THRUST_HIP_DEVICE_FUNCTION",
        "THRUST_HIP_FUNCTION",
        "THRUST_HIP_RUNTIME_FUNCTION",
        "THRUST_HOST",
        "THRUST_HOST_DEVICE",
        "THRUST_NODISCARD",
        "THRUST_NODISCARD_FRIEND",
        "THRUST_RUNTIME_FUNCTION",
        "VX_CALLBACK",
        "WINAPI",
        "consteval",
    }
    KERNEL_FUNCTION_SPECIFIER_VALUES = {"__global__", "ROCWMMA_KERNEL"}
    DECLARATION_QUALIFIER_TOKENS = {
        "__SHARED__",
        "__CONSTANT__",
        "__MANAGED__",
        "__DEVICE__",
        "STATIC",
        "THREAD_LOCAL",
        "EXTERN",
        "CONSTEXPR",
        "INLINE",
        "REGISTER",
    }
    TYPE_PREFIX_TOKENS = {"TYPENAME"}
    TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "UNSIGNED", "SIGNED", "__RESTRICT__"}
    POSTFIX_TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "__RESTRICT__"}
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
        "SPACESHIP",
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
        "ARROW",
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
    CATCH_TEST_BLOCK_MACRO_NAMES = {
        "HIP_TEST_CASE",
        "HIP_TEMPLATE_TEST_CASE",
        "TEST_CASE",
        "TEMPLATE_LIST_TEST_CASE",
        "TEMPLATE_PRODUCT_TEST_CASE",
        "TEMPLATE_TEST_CASE",
        "TEMPLATE_TEST_CASE_SIG",
    }
    CATCH_NESTED_BLOCK_MACRO_NAMES = {
        "AND_THEN",
        "AND_WHEN",
        "DYNAMIC_SECTION",
        "GIVEN",
        "SECTION",
        "THEN",
        "WHEN",
    }
    GTEST_TEST_BLOCK_MACRO_NAMES = {
        "TEST",
        "TEST_F",
        "TEST_P",
        "TYPED_TEST",
        "TYPED_TEST_P",
    }
    CATCH_TEST_BLOCK_MACRO_NAMES |= GTEST_TEST_BLOCK_MACRO_NAMES
    BARE_NESTED_BLOCK_MACRO_NAMES = {
        "HIP_TEST_ATOMIC_BACKWARD_COMPAT_MEMORY",
    }
    STANDALONE_STATEMENT_MACRO_NAMES = {
        "BEGIN_ROCPRIM_NAMESPACE",
        "CHECK_IMAGE_SUPPORT",
        "END_ROCPRIM_NAMESPACE",
        "_CCCL_BEGIN_NAMESPACE_CUDA_STD",
        "_CCCL_BEGIN_NAMESPACE_STD",
        "_CCCL_END_NAMESPACE_CUDA_STD",
        "_CCCL_END_NAMESPACE_STD",
        "_CCCL_EXEC_CHECK_DISABLE",
        "THRUST_EXEC_CHECK_DISABLE",
        "THRUST_NAMESPACE_BEGIN",
        "THRUST_NAMESPACE_END",
    }
    OPTIONAL_SEMICOLON_MACRO_NAMES = {
        "INSTANTIATE_TESTS",
        "INSTANTIATE_TEST_CATEGORIES",
        "INSTANTIATE_TEST_CASE_P",
        "INSTANTIATE_TEST_SUITE_P",
    }
    TYPE_ATTRIBUTE_IDENTIFIERS = {
        "_CCCL_NODEBUG_ALIAS",
        "_CCCL_TYPE_VISIBILITY_DEFAULT",
        "__attribute__",
        "__declspec",
        "alignas",
        "__align__",
    }
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
        "coalesced_group",
        "cluster_group",
        "__half",
        "__half2",
        "half",
        "half2",
        "grid_group",
        "multi_grid_group",
        "thread_block",
        "float16_t",
        "float32_t",
        "float64_t",
        "thread_group",
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
        self.namespace_depth = 0
        self.type_aliases = set()
        self.namespace_aliases = {}
        self.pragma_pack_alignment = None
        self.pragma_pack_stack = []
        self.user_function_names = self.collect_user_function_names()

    def error(self, message: str):
        token_info = (
            f"at token '{self.current_token.value}'"
            if self.current_token
            else "at end of input"
        )
        raise self.syntax_error(f"Parse error {token_info}: {message}")

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

        program = HipProgramNode(statements)
        program.namespace_aliases = dict(self.namespace_aliases)
        return program

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

        if self.is_standalone_statement_macro():
            return self.parse_standalone_statement_macro()

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
        if self.match("GOTO"):
            return self.parse_goto_statement()
        if self.match("SYNCTHREADS", "SYNCWARP"):
            return self.parse_sync_statement()
        if self.match("ASM"):
            return self.parse_asm_statement()
        if self.is_hip_dynamic_shared_macro():
            return self.parse_hip_dynamic_shared_macro()
        if self.is_catch_test_block_macro():
            return self.parse_catch_test_block_macro()
        if self.is_catch_nested_block_macro():
            return self.parse_catch_nested_block_macro()
        if self.is_bare_nested_block_macro():
            return self.parse_bare_nested_block_macro()
        if self.match("LBRACE"):
            return self.parse_block()
        if self.is_label_statement_start():
            return self.parse_label_statement()
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


def parse_hip_code(code: str) -> HipProgramNode:
    """Parse HIP source text and return the backend AST."""
    lexer = HipLexer(code)
    tokens = lexer.tokenize()
    parser = HipParser(tokens)
    return parser.parse()
