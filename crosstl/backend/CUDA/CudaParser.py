"""CUDA Parser Implementation"""

from crosstl.translator.refusals import CudaSyntaxError

from .CudaAst import (
    BreakNode,
    ContinueNode,
    EnumNode,
    FunctionNode,
    KernelNode,
    ShaderNode,
    StructNode,
    TypeAliasNode,
)
from .parsing.aliases import CudaAliasMixin
from .parsing.control_flow import CudaControlFlowMixin
from .parsing.expressions import CudaExpressionMixin
from .parsing.function_names import CudaFunctionNameMixin
from .parsing.functions import CudaFunctionMixin
from .parsing.initializers import CudaInitializerMixin
from .parsing.lambdas import CudaLambdaMixin
from .parsing.lookahead import CudaLookaheadMixin
from .parsing.macros import CudaMacroMixin
from .parsing.records import CudaRecordMixin
from .parsing.statements import CudaStatementMixin
from .parsing.struct_methods import CudaStructMethodMixin
from .parsing.types import CudaTypeMixin
from .parsing.variable_declarations import CudaVariableDeclarationMixin


class CudaParser(
    CudaLookaheadMixin,
    CudaFunctionNameMixin,
    CudaAliasMixin,
    CudaRecordMixin,
    CudaStructMethodMixin,
    CudaMacroMixin,
    CudaFunctionMixin,
    CudaTypeMixin,
    CudaVariableDeclarationMixin,
    CudaStatementMixin,
    CudaControlFlowMixin,
    CudaExpressionMixin,
    CudaLambdaMixin,
    CudaInitializerMixin,
):
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
        "_CCCL_CONSTEXPR_CXX20",
        "_CCCL_DEVICE",
        "_CCCL_DEVICE_API",
        "_CCCL_EXEC_CHECK_DISABLE",
        "_CCCL_FORCEINLINE",
        "_CCCL_HIDE_FROM_ABI",
        "_CCCL_HOST_API",
        "_CCCL_HOST_DEVICE",
        "_CCCL_HOST_DEVICE_API",
        "_CCCL_KERNEL_ATTRIBUTES",
        "_CCCL_NODEBUG_ALIAS",
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
        "_CCCL_NO_UNIQUE_ADDRESS",
        "_CCCL_TYPE_VISIBILITY_DEFAULT",
        "_CCCL_NODEBUG_ALIAS",
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
        "CONSTEXPR",
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
        self.block_depth = 0

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
                # Where the struct's member functions go among the free functions.
                structs[-1].function_index = len(functions)
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
        shader.struct_method_names = sorted(self.struct_method_names)
        return shader

    def eat(self, expected_type):
        # The end of input is never consumed: a construct that reads past it is unterminated.
        if expected_type == "EOF" and self.current_token[0] == "EOF":
            raise CudaSyntaxError("Unexpected end of input")
        if self.current_token[0] == expected_type:
            token = self.current_token
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")
            return token
        else:
            raise CudaSyntaxError(
                f"Expected {expected_type}, got {self.current_token[0]}"
            )

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
