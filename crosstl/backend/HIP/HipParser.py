"""HIP Parser for converting HIP tokens to AST"""

from typing import List
from .HipLexer import HipLexer, Token
from .HipAst import (
    ASTNode,
    ShaderNode,
    FunctionNode,
    KernelNode,
    StructNode,
    VariableNode,
    AssignmentNode,
    BinaryOpNode,
    UnaryOpNode,
    FunctionCallNode,
    AtomicOperationNode,
    KernelLaunchNode,
    CastNode,
    CaseNode,
    DesignatedInitializerNode,
    DeleteNode,
    DoWhileNode,
    InitializerListNode,
    SyncNode,
    MemberAccessNode,
    NewNode,
    ArrayAccessNode,
    IfNode,
    ForNode,
    RangeForNode,
    WhileNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    PreprocessorNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    HipBuiltinNode,
)


class HipProgramNode(ASTNode):
    """Root node representing a complete HIP program"""

    def __init__(self, statements=None):
        """Initialize the program node with top-level statements."""
        self.statements = statements or []

    def __repr__(self):
        """Return a developer-readable program representation."""
        return f"HipProgramNode(statements={self.statements})"


class HipParser:
    """Parse HIP tokens into the HIP backend AST."""

    FUNCTION_SPECIFIER_TOKENS = {"STATIC", "INLINE", "EXTERN"}
    TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "UNSIGNED", "SIGNED", "__RESTRICT__"}
    POSTFIX_TYPE_QUALIFIER_TOKENS = {"__RESTRICT__"}
    TYPE_REFERENCE_TOKENS = {"AMPERSAND", "AND"}
    CPP_NAMED_CASTS = {"static_cast", "reinterpret_cast", "const_cast", "dynamic_cast"}
    ATOMIC_FUNCTION_TOKENS = {
        "ATOMICADD",
        "ATOMICSUB",
        "ATOMICMAX",
        "ATOMICMIN",
        "ATOMICEXCH",
        "ATOMICCAS",
    }
    FLAT_BUILTIN_TOKEN_MAP = {
        "HIPTHREADIDX": "threadIdx",
        "HIPBLOCKIDX": "blockIdx",
        "HIPBLOCKDIM": "blockDim",
        "HIPGRIDDIM": "gridDim",
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

    def __init__(self, tokens: List[Token]):
        """Initialize the parser with a token stream from ``HipLexer``."""
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
        function_qualifiers = {
            *self.FUNCTION_SPECIFIER_TOKENS,
            "__DEVICE__",
            "__HOST__",
            "__GLOBAL__",
            "__FORCEINLINE__",
            "__NOINLINE__",
        }
        while (
            index < len(self.tokens) and self.tokens[index].type in function_qualifiers
        ):
            index += 1

        index = self.skip_type_at_pos(index)
        if index is None:
            return None

        if (
            index + 1 < len(self.tokens)
            and self.is_function_name_token(self.tokens[index])
            and self.tokens[index + 1].type == "LPAREN"
        ):
            return index

        return None

    def error(self, message: str):
        """Raise a syntax error annotated with the current token."""
        token_info = (
            f"at token '{self.current_token.value}'"
            if self.current_token
            else "at end of input"
        )
        raise SyntaxError(f"Parse error {token_info}: {message}")

    def advance(self):
        """Advance to the next token or mark the parser as finished."""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def peek(self, offset: int = 1):
        """Look ahead at the next token without advancing"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def consume(self, expected_type: str):
        """Consume and return the current token when its type matches."""
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
        """Advance past newline tokens between declarations/statements."""
        while self.match("NEWLINE"):
            self.advance()

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
        return allow_identifier and token.type == "IDENTIFIER"

    def parse(self):
        """Parse the entire HIP program into a ``HipProgramNode``."""
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
        """Parse a single statement"""
        if not self.current_token:
            return None

        # Skip newlines and semicolons
        if self.match("NEWLINE", "SEMICOLON"):
            self.advance()
            return None

        # Parse preprocessor directives
        if self.match("HASH"):
            return self.parse_preprocessor()

        if self.match("TEMPLATE"):
            return self.parse_template_prefixed_declaration()

        # Parse device/host/global qualifiers
        if self.match("__DEVICE__", "__HOST__", "__GLOBAL__"):
            return self.parse_function_with_qualifier()

        # Parse struct definitions
        if self.match("STRUCT"):
            return self.parse_struct()

        # Parse class definitions
        if self.match("CLASS"):
            return self.parse_class()

        if self.is_type_alias_start():
            return self.parse_type_alias()

        # Parse return statements
        if self.match("RETURN"):
            return self.parse_return_statement()

        # Parse control flow statements
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
        if self.match("LBRACE"):
            return self.parse_block()
        if self.is_identifier_value("delete"):
            return self.parse_delete_statement()

        # Try to parse function or variable declaration
        if self.block_depth > 0 and self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        elif self.is_function_declaration():
            return self.parse_simple_function()
        elif self.is_variable_declaration():
            declarations = self.parse_variable_declaration_list()
            return declarations if len(declarations) > 1 else declarations[0]
        else:
            # Parse expression statement
            return self.parse_expression_statement()

    def parse_template_prefixed_declaration(self):
        self.consume("TEMPLATE")
        self.parse_template_suffix()
        self.skip_newlines()

        if self.match("__DEVICE__", "__HOST__", "__GLOBAL__"):
            return self.parse_function_with_qualifier()
        if self.match("STRUCT"):
            return self.parse_struct()
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
        """Parse preprocessor directives"""
        self.consume("HASH")

        if not self.current_token:
            self.error("Expected preprocessor directive after #")

        directive = self.current_token.value
        self.advance()

        # Parse the rest of the line
        content = []
        while self.current_token and not self.match("NEWLINE"):
            content.append(self.current_token.value)
            self.advance()

        return PreprocessorNode(directive, " ".join(content))

    def parse_function_with_qualifier(self):
        qualifiers = []

        while self.match(
            "__DEVICE__", "__HOST__", "__GLOBAL__", "__FORCEINLINE__", "__NOINLINE__"
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        return_type = self.parse_type()
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

        function = FunctionNode(return_type, name, params, body, qualifiers)

        # __global__ qualifier marks a kernel
        if "__global__" in qualifiers:
            return KernelNode(return_type, name, params, body)

        return function

    def parse_simple_function(self):
        qualifiers = []
        while self.match(*self.FUNCTION_SPECIFIER_TOKENS):
            qualifiers.append(self.current_token.value)
            self.advance()

        return_type = self.parse_type()
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

        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                member = self.parse_struct_member()
                if member:
                    members.append(member)

            self.consume("RBRACE")

        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)

    def parse_class(self):
        """Parse class definitions (treat similar to struct for now)"""
        self.consume("CLASS")

        name = self.consume("IDENTIFIER").value

        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                # Skip access specifiers
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
            "__SHARED__", "__CONSTANT__", "__DEVICE__", "STATIC", "EXTERN"
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        var_type = self.parse_type()
        name = self.consume("IDENTIFIER").value
        var_type += self.parse_array_suffix()

        value = None
        if self.match("ASSIGN"):
            self.advance()
            value = self.parse_expression()
        elif self.match("LPAREN"):
            value = FunctionCallNode(var_type, self.parse_parenthesized_argument_list())
        elif self.match("LBRACE"):
            value = self.parse_initializer_list()

        if consume_semicolon:
            self.consume("SEMICOLON")

        return VariableNode(var_type, name, value, qualifiers)

    def parse_variable_declaration_list(self, consume_semicolon=True):
        qualifiers = []

        while self.match(
            "__SHARED__", "__CONSTANT__", "__DEVICE__", "STATIC", "EXTERN"
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
            self.consume("SEMICOLON")

        return declarations

    def parse_variable_declarator(self, base_type, qualifiers, allow_prefix):
        var_type = base_type
        if allow_prefix:
            var_type = self.parse_declarator_prefix(var_type)

        name = self.consume("IDENTIFIER").value
        var_type += self.parse_array_suffix()
        value = self.parse_variable_initializer(var_type)

        return VariableNode(var_type, name, value, list(qualifiers))

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

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()

        if (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
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

        while self.match(*self.TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()

        if (
            self.is_builtin_type_token()
            or self.match(*self.VECTOR_TYPE_TOKENS)
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

    def parse_postfix_type_qualifiers(self, type_parts):
        while self.match(*self.POSTFIX_TYPE_QUALIFIER_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()

    def parse_type_name(self):
        type_name = self.current_token.value
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
        condition = self.parse_expression()
        self.consume("RPAREN")

        if_body = None
        self.skip_newlines()
        if self.match("LBRACE"):
            if_body = self.parse_block()
        else:
            if_body = self.parse_statement()

        else_body = None
        if self.match("ELSE"):
            self.advance()
            self.skip_newlines()
            if self.match("LBRACE"):
                else_body = self.parse_block()
            else:
                else_body = self.parse_statement()

        return IfNode(condition, if_body, else_body)

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
        default_case = None

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
                cases.append(CaseNode(value, body))
            elif self.match("DEFAULT"):
                self.advance()
                self.consume("COLON")
                default_case = []
                while self.current_token and not self.match("CASE", "RBRACE"):
                    stmt = self.parse_statement()
                    if stmt:
                        default_case.append(stmt)
            else:
                self.advance()

        self.consume("RBRACE")
        return SwitchNode(expression, cases, default_case)

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

        self.consume("SEMICOLON")

        return expr

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()

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
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_ternary_expression(self):
        expr = self.parse_logical_or_expression()

        if self.match("QUESTION"):
            self.advance()
            true_expr = self.parse_expression()
            self.consume("COLON")
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

        while self.match("SHIFT_LEFT", "SHIFT_RIGHT", "LSHIFT", "RSHIFT"):
            op = self.current_token.value
            self.advance()
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

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
                member = self.consume("IDENTIFIER").value
                expr = MemberAccessNode(expr, member, False)
            elif self.match("ARROW"):
                self.consume("ARROW")
                member = self.consume("IDENTIFIER").value
                expr = MemberAccessNode(expr, member, True)
            elif self.match("LPAREN"):
                self.consume("LPAREN")
                args = self.parse_argument_list()
                self.consume("RPAREN")
                expr = self.parse_function_call_node(expr, args)
            elif self.match("KERNEL_LAUNCH_START"):
                expr = self.parse_kernel_launch(expr)
            elif self.match("INCREMENT", "DECREMENT"):
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

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
                    next_type = (
                        self.tokens[index + 1].type
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
            elif token_type == "RSHIFT":
                depth -= 2
                if depth == 0:
                    next_type = (
                        self.tokens[index + 1].type
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

            # Check for HIP built-in variables
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

        elif self.match("INTEGER", "FLOAT_NUM", "FLOAT", "STRING"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("TRUE", "FALSE", "NULL", "NULLPTR", "HIPSUCCESS"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("CHAR") and self.current_token.value != "char":
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

        index = self.skip_type_at_pos(index)
        if index is not None:
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
            "__DEVICE__",
            "STATIC",
            "EXTERN",
            "CONST",
            "VOLATILE",
            "UNSIGNED",
            "SIGNED",
        }:
            index += 1

        index = self.skip_type_at_pos(index)
        if index is not None:
            if index < len(self.tokens) and self.tokens[index].type == "IDENTIFIER":
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

    def skip_type_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.TYPE_QUALIFIER_TOKENS
        ):
            index += 1

        if index >= len(self.tokens) or not self.is_type_token(self.tokens[index]):
            return None

        type_token = self.tokens[index].type
        type_value = self.tokens[index].value
        has_qualified_suffix = False
        index += 1

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
            type_token != "IDENTIFIER"
            or has_qualified_suffix
            or type_value == "auto"
            or type_value in self.type_aliases
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

    def skip_postfix_type_qualifiers_at_pos(self, index):
        while (
            index < len(self.tokens)
            and self.tokens[index].type in self.POSTFIX_TYPE_QUALIFIER_TOKENS
        ):
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
