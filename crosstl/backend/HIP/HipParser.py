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
    }
    FUNCTION_NAME_TOKENS = {"IDENTIFIER", *ATOMIC_FUNCTION_TOKENS}
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
    ATOMIC_FUNCTION_NAMES = {
        "atomicAdd",
        "hipAtomicAdd",
        "atomicSub",
        "hipAtomicSub",
        "atomicMax",
        "hipAtomicMax",
        "atomicMin",
        "hipAtomicMin",
        "atomicExch",
        "hipAtomicExch",
        "atomicCAS",
        "hipAtomicCAS",
        "atomicAnd",
        "hipAtomicAnd",
        "atomicOr",
        "hipAtomicOr",
        "atomicXor",
        "hipAtomicXor",
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
    HIP_IDENTIFIER_TYPE_NAMES = {
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
        function_qualifiers = {
            *self.FUNCTION_SPECIFIER_TOKENS,
            "__DEVICE__",
            "__HOST__",
            "__GLOBAL__",
            "__FORCEINLINE__",
            "__NOINLINE__",
            "__LAUNCH_BOUNDS__",
        }
        while (
            index < len(self.tokens) and self.tokens[index].type in function_qualifiers
        ):
            if self.tokens[index].type == "__LAUNCH_BOUNDS__":
                index = self.skip_launch_bounds_at_pos(index)
            else:
                index += 1
            index = self.skip_newlines_at_pos(index)

        index = self.skip_newlines_at_pos(index)
        index = self.skip_type_at_pos(index, allow_unknown_identifier_pointers=True)
        if index is None:
            return None

        index = self.skip_newlines_at_pos(index)
        while (
            index < len(self.tokens) and self.tokens[index].type == "__LAUNCH_BOUNDS__"
        ):
            index = self.skip_launch_bounds_at_pos(index)
            index = self.skip_newlines_at_pos(index)

        if (
            index + 1 < len(self.tokens)
            and self.is_function_name_token(self.tokens[index])
            and self.tokens[index + 1].type == "LPAREN"
        ):
            return index

        return None

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

    def is_function_name_token(self, token=None):
        token = token or self.current_token
        return bool(token and token.type in self.FUNCTION_NAME_TOKENS)

    def consume_function_name(self):
        if not self.is_function_name_token():
            token_type = self.current_token.type if self.current_token else "EOF"
            self.error(f"Expected function name, got {token_type}")

        name = self.current_token.value
        self.advance()
        return name

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

        if self.is_linkage_specifier_start():
            return self.parse_linkage_specification()

        if self.match("TEMPLATE"):
            return self.parse_template_prefixed_declaration()

        if self.match(
            "__DEVICE__",
            "__HOST__",
            "__GLOBAL__",
            "__FORCEINLINE__",
            "__NOINLINE__",
            "__LAUNCH_BOUNDS__",
        ):
            if self.function_name_index_at(self.pos) is None:
                declarations = self.parse_variable_declaration_list()
                if self.match("SEMICOLON"):
                    self.advance()
                return declarations if len(declarations) > 1 else declarations[0]

            return self.parse_function_with_qualifier()

        if self.match("STRUCT"):
            return self.parse_struct()

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
        if self.match("LBRACE"):
            return self.parse_block()
        if self.is_identifier_value("delete"):
            return self.parse_delete_statement()

        if self.block_depth > 0 and self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        elif self.is_function_declaration():
            return self.parse_simple_function()
        elif self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        else:
            return self.parse_expression_statement()

    def parse_template_prefixed_declaration(self):
        self.consume("TEMPLATE")
        self.parse_template_suffix()
        self.skip_newlines()

        if self.match("__DEVICE__", "__HOST__", "__GLOBAL__", "__LAUNCH_BOUNDS__"):
            return self.parse_function_with_qualifier()
        if self.match("STRUCT"):
            return self.parse_struct()
        if self.match("ENUM"):
            return self.parse_enum()
        if self.match("CLASS"):
            return self.parse_class()
        if self.is_function_declaration():
            return self.parse_simple_function()

        raise SyntaxError(
            f"Expected declaration after template prefix, got {self.current_token.type}"
        )

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

        if not self.match("LBRACE"):
            alias = self.parse_type_alias_declarator(
                f"struct {tag_name}", allow_prefix=True
            )
            if self.match("SEMICOLON"):
                self.advance()
            return alias

        self.consume("LBRACE")
        members = self.parse_struct_members()
        self.consume("RBRACE")

        alias_name = tag_name
        if self.match("IDENTIFIER"):
            alias_name = self.current_token.value
            self.advance()
        if not alias_name:
            self.error("Expected typedef struct alias name")

        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(alias_name)
        return StructNode(alias_name, members)

    def parse_using_alias(self):
        self.advance()
        if self.match("NAMESPACE"):
            self.skip_until_semicolon()
            return None

        name = self.consume("IDENTIFIER").value
        self.consume("ASSIGN")
        self.skip_newlines()
        alias_type = self.parse_type()
        if self.match("SEMICOLON"):
            self.advance()
        self.type_aliases.add(name)
        return TypeAliasNode(alias_type, name)

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
            if not self.match(
                *self.FUNCTION_SPECIFIER_TOKENS,
                "__DEVICE__",
                "__HOST__",
                "__GLOBAL__",
                "__FORCEINLINE__",
                "__NOINLINE__",
                "__LAUNCH_BOUNDS__",
                "CONSTEXPR",
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

        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        function = FunctionNode(return_type, name, params, body, qualifiers, attributes)

        if "__global__" in qualifiers:
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
        while self.match(*self.FUNCTION_SPECIFIER_TOKENS):
            qualifiers.append(self.current_token.value)
            self.advance()

        return_type = self.parse_type()
        self.skip_newlines()
        name = self.consume_function_name()
        self.user_function_names.add(name)

        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")

        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode(return_type, name, params, body, qualifiers)

    def parse_struct(self):
        self.consume("STRUCT")

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()
            self.type_aliases.add(name)

        self.skip_newlines()
        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            members = self.parse_struct_members()
            self.consume("RBRACE")

        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)

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

    def parse_struct_members(self):
        members = []
        while self.current_token and not self.match("RBRACE"):
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
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

        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                if self.match("PUBLIC", "PRIVATE", "PROTECTED"):
                    self.advance()
                    if self.match("COLON"):
                        self.advance()
                    continue

                member = self.parse_struct_member()
                if member:
                    members.append(member)

            self.consume("RBRACE")

        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)  # Treat class as struct for simplicity

    def parse_struct_member(self):
        try:
            member_type = self.parse_type()
            name = self.consume("IDENTIFIER").value
            member_type += self.parse_array_suffix()

            if self.match("SEMICOLON"):
                self.advance()

            return VariableNode(member_type, name)
        except Exception:
            while self.current_token and not self.match("SEMICOLON", "RBRACE"):
                self.advance()
            if self.match("SEMICOLON"):
                self.advance()
            return None

    def parse_variable_declaration(self, consume_semicolon=True):
        qualifiers = []

        while self.match(
            "__SHARED__",
            "__CONSTANT__",
            "__MANAGED__",
            "__DEVICE__",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        var_type = self.parse_type()
        name = self.consume("IDENTIFIER").value
        var_type += self.parse_array_suffix()
        self.skip_newlines()

        value = None
        if self.match("ASSIGN"):
            self.advance()
            value = self.parse_expression()
        elif self.match("LPAREN"):
            value = FunctionCallNode(var_type, self.parse_parenthesized_argument_list())
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
        qualifiers = []

        while self.match(
            "__SHARED__",
            "__CONSTANT__",
            "__MANAGED__",
            "__DEVICE__",
            "STATIC",
            "EXTERN",
            "CONSTEXPR",
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        first_type = self.parse_type()
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

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        var_type = base_type
        if allow_prefix:
            var_type = self.parse_declarator_prefix(var_type)

        name = self.consume("IDENTIFIER").value
        var_type += self.parse_array_suffix()
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
        if self.match("ASSIGN"):
            self.advance()
            self.skip_newlines()
            return self.parse_expression()
        if self.match("LPAREN"):
            return FunctionCallNode(var_type, self.parse_parenthesized_argument_list())
        if self.match("LBRACE"):
            return self.parse_initializer_list()
        return None

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

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
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

        array_suffix = self.parse_array_suffix()
        if array_suffix:
            type_parts.append(array_suffix)

        return " ".join(type_parts)

    def parse_type_without_array_suffix(self):
        type_parts = []
        saw_integral_sign = False

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            if self.current_token.type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            type_parts.append(self.current_token.value)
            self.advance()

        if self.is_implicit_int_current_type(saw_integral_sign):
            type_parts.append("int")
        elif (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
            or self.match(*self.RESOURCE_TYPE_TOKENS)
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

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()

    def parse_type_name(self):
        type_name = self.current_token.value
        token_type = self.current_token.type
        self.advance()

        if token_type == "LONG" and self.match("LONG"):
            type_name += " long"
            self.advance()
        if token_type == "LONG" and self.match("INT"):
            self.advance()

        while self.match("SCOPE"):
            self.consume("SCOPE")
            member = self.consume("IDENTIFIER").value
            type_name += f"::{member}"

        if self.match("LT"):
            type_name += self.parse_template_suffix()

        return type_name

    def parse_parameter_list(self):
        params = []

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

            param_type = self.parse_type()

            param_name = ""
            if self.match("IDENTIFIER"):
                param_name = self.current_token.value
                self.advance()

            param_type += self.parse_array_suffix()
            params.append({"type": param_type, "name": param_name})

            self.skip_newlines()
            if self.match("COMMA"):
                self.advance()
                self.skip_newlines()
            else:
                break

        return params

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

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.parse_expression()
        self.consume("SEMICOLON")

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

        while self.match("LOGICAL_OR", "OR"):
            op = self.current_token.value
            self.advance()
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()

        while self.match("LOGICAL_AND", "AND"):
            op = self.current_token.value
            self.advance()
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()

        while self.match("BITWISE_OR", "PIPE"):
            op = self.current_token.value
            self.advance()
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()

        while self.match("BITWISE_XOR", "XOR"):
            op = self.current_token.value
            self.advance()
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()

        while self.match("BITWISE_AND", "AMPERSAND"):
            op = self.current_token.value
            self.advance()
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()

        while self.match("EQ", "NE"):
            op = self.current_token.value
            self.advance()
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()

        while self.match("LT", "LE", "GT", "GE"):
            op = self.current_token.value
            self.advance()
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op, right)

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

        while self.match("PLUS", "MINUS"):
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        left = self.parse_unary_expression()

        while self.match("MULTIPLY", "STAR", "DIVIDE", "SLASH", "MODULO", "PERCENT"):
            op = self.current_token.value
            self.advance()
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        if self.match(
            "PLUS",
            "MINUS",
            "NOT",
            "BITWISE_NOT",
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
                member = self.consume("IDENTIFIER").value
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
                if expr == "sizeof" and self.is_sizeof_type_operand():
                    args = [self.parse_type()]
                else:
                    args = self.parse_argument_list()
                self.consume("RPAREN")
                expr = self.parse_function_call_node(expr, args)
            elif self.match("LBRACE"):
                initializer = self.parse_initializer_list()
                expr = self.parse_function_call_node(expr, initializer.elements)
            elif self.match("KERNEL_LAUNCH_START"):
                expr = self.parse_kernel_launch(expr)
            elif self.match("INCREMENT", "DECREMENT"):
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

    def parse_member_name(self):
        if self.match(*self.MEMBER_NAME_TOKENS):
            member = self.current_token.value
            self.advance()
            return member
        self.error("Expected member name")

    def is_sizeof_type_operand(self):
        saved_pos = self.pos
        try:
            while self.match(*self.TYPE_QUALIFIER_TOKENS):
                self.advance()

            if not self.is_type_token(allow_identifier=False) and not (
                self.match("IDENTIFIER")
                and self.current_token.value in self.type_aliases
            ):
                return False

            token_type = self.current_token.type
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
                    return next_type in {
                        "LPAREN",
                        "SCOPE",
                        "DOT",
                        "KERNEL_LAUNCH_START",
                        "COMMA",
                        "RPAREN",
                    }
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
            else:
                parts.append(token_value)
                self.consume(token_type)

        return f"<{self.format_template_parts(parts)}>"

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
                args[0],
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
            return function_arg.expression
        return function_arg

    def parse_kernel_launch(self, kernel_name):
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

        self.consume("LPAREN")
        args = self.parse_argument_list()
        self.consume("RPAREN")

        return KernelLaunchNode(kernel_name, blocks, threads, shared_mem, stream, args)

    def parse_primary_expression(self):
        if self.match("LBRACKET") and self.is_lambda_expression_start():
            return self.parse_lambda_expression()

        if self.match("IDENTIFIER"):
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
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("SYNCTHREADS", "SYNCWARP"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("LPAREN"):
            if self.is_cast_expression():
                self.consume("LPAREN")
                target_type = self.parse_type()
                self.consume("RPAREN")
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)

            self.consume("LPAREN")
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
        target_type = self.parse_type_without_array_suffix()

        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            size = None
            if not self.match("RBRACKET"):
                size = self.parse_expression()
            self.consume("RBRACKET")
            return NewNode(target_type, size=size, is_array=True)

        args = []
        if self.match("LPAREN"):
            self.consume("LPAREN")
            args = self.parse_argument_list()
            self.consume("RPAREN")

        return NewNode(target_type, args=args)

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

        saved_pos = self.pos
        try:
            self.advance()
            while self.match(*self.TYPE_QUALIFIER_TOKENS):
                self.advance()

            if not self.is_type_token(allow_identifier=False) and not (
                self.match("IDENTIFIER")
                and self.current_token.value in self.type_aliases
            ):
                return False

            token_type = self.current_token.type
            self.advance()
            if token_type == "LONG" and self.match("LONG"):
                self.advance()
            while self.match("ASTERISK", "STAR"):
                self.advance()

            while self.match("LBRACKET"):
                self.advance()
                while self.current_token and not self.match("RBRACKET"):
                    self.advance()
                self.consume("RBRACKET")

            return self.match("RPAREN")
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
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

        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.FUNCTION_SPECIFIER_TOKENS
        ):
            index += 1

        index = self.skip_type_at_pos(
            index, allow_unknown_identifier_pointers=self.block_depth == 0
        )
        if index is not None:
            index = self.skip_newlines_at_pos(index)
            if (
                index + 1 < len(self.tokens)
                and self.is_function_name_token(self.tokens[index])
                and self.tokens[index + 1].type == "LPAREN"
            ):
                return True

        return False

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

        index = self.skip_type_at_pos(index)
        if index is not None:
            if index < len(self.tokens) and self.tokens[index].type == "IDENTIFIER":
                index += 1
                while index < len(self.tokens) and self.tokens[index].type == "NEWLINE":
                    index += 1
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

    def skip_type_at_pos(self, index, allow_unknown_identifier_pointers=False):
        saw_integral_sign = False
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            if self.tokens[index].type in {"SIGNED", "UNSIGNED"}:
                saw_integral_sign = True
            index += 1

        if index >= len(self.tokens):
            return None

        if self.is_implicit_int_type_at_pos(index, saw_integral_sign):
            type_token = "INT"
            type_value = "int"
        elif self.is_type_token(self.tokens[index]):
            type_token = self.tokens[index].type
            type_value = self.tokens[index].value
            index += 1
            if type_token == "LONG" and index < len(self.tokens):
                if self.tokens[index].type == "LONG":
                    index += 1
                if index < len(self.tokens) and self.tokens[index].type == "INT":
                    index += 1
        else:
            return None

        has_qualified_suffix = False

        while (
            index + 1 < len(self.tokens)
            and self.tokens[index].type == "SCOPE"
            and self.tokens[index + 1].type == "IDENTIFIER"
        ):
            has_qualified_suffix = True
            index += 2

        if index < len(self.tokens) and self.tokens[index].type == "LT":
            has_qualified_suffix = True
            index = self.skip_template_at_pos(index)
            if index is None:
                return None

        index = self.skip_postfix_type_qualifiers_at_pos(index)
        can_have_pointer_suffix = (
            allow_unknown_identifier_pointers
            or type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
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

        return self.skip_array_suffix_at_pos(index)

    def is_implicit_int_type_at_pos(self, index, saw_integral_sign):
        if not saw_integral_sign or index >= len(self.tokens):
            return False

        token_type = self.tokens[index].type
        if token_type in {"ASTERISK", "STAR", *self.TYPE_REFERENCE_TOKENS}:
            return True

        if token_type != "IDENTIFIER":
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
