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
    CastNode,
    CaseNode,
    DoWhileNode,
    InitializerListNode,
    SyncNode,
    MemberAccessNode,
    ArrayAccessNode,
    IfNode,
    ForNode,
    WhileNode,
    ReturnNode,
    BreakNode,
    ContinueNode,
    PreprocessorNode,
    SwitchNode,
    TernaryOpNode,
    HipBuiltinNode,
)


class HipProgramNode(ASTNode):
    """Root node representing a complete HIP program"""

    def __init__(self, statements=None):
        self.statements = statements or []

    def __repr__(self):
        return f"HipProgramNode(statements={self.statements})"


class HipParser:
    """Parser for HIP language constructs"""

    FUNCTION_SPECIFIER_TOKENS = {"STATIC", "INLINE", "EXTERN"}
    TYPE_QUALIFIER_TOKENS = {"CONST", "VOLATILE", "UNSIGNED", "SIGNED"}
    BUILTIN_TYPE_TOKENS = {
        "INT",
        "FLOAT",
        "DOUBLE",
        "BOOL",
        "VOID",
        "CHAR",
        "SHORT",
        "LONG",
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
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        self.block_depth = 0

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
        """Look ahead at the next token without advancing"""
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
        if not self.current_token:
            return False
        return self.current_token.type in token_types

    def skip_newlines(self):
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
        """Parse the entire HIP program"""
        statements = []

        while self.current_token:
            if self.match("NEWLINE", "SEMICOLON"):
                self.advance()
                continue

            stmt = self.parse_statement()
            if stmt:
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

        # Parse device/host/global qualifiers
        if self.match("__DEVICE__", "__HOST__", "__GLOBAL__"):
            return self.parse_function_with_qualifier()

        # Parse struct definitions
        if self.match("STRUCT"):
            return self.parse_struct()

        # Parse class definitions
        if self.match("CLASS"):
            return self.parse_class()

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

        # Try to parse function or variable declaration
        if self.block_depth > 0 and self.is_variable_declaration():
            return self.parse_variable_declaration()
        elif self.is_function_declaration():
            return self.parse_simple_function()
        elif self.is_variable_declaration():
            return self.parse_variable_declaration()
        else:
            # Parse expression statement
            return self.parse_expression_statement()

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
        name = self.consume("IDENTIFIER").value

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
        name = self.consume("IDENTIFIER").value

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

        if consume_semicolon and self.match("SEMICOLON"):
            self.advance()

        return VariableNode(var_type, name, value, qualifiers)

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

        if self.is_builtin_type_token():
            type_parts.append(self.current_token.value)
            self.advance()
        elif self.match(*self.VECTOR_TYPE_TOKENS):
            type_parts.append(self.current_token.value)
            self.advance()
        elif self.match("IDENTIFIER"):
            type_parts.append(self.current_token.value)
            self.advance()
        else:
            type_parts.append("int")  # Default type

        while self.match("ASTERISK", "STAR"):
            type_parts.append("*")
            self.advance()

        return " ".join(type_parts)

    def parse_parameter_list(self):
        params = []

        if self.match("RPAREN"):
            return params

        while True:
            param_type = self.parse_type()

            param_name = ""
            if self.match("IDENTIFIER"):
                param_name = self.current_token.value
                self.advance()

            param_type += self.parse_array_suffix()
            params.append({"type": param_type, "name": param_name})

            if self.match("COMMA"):
                self.advance()
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

        init = None
        if not self.match("SEMICOLON"):
            if self.is_variable_declaration():
                init = self.parse_variable_declaration(consume_semicolon=False)
            else:
                init = self.parse_expression()
        self.consume("SEMICOLON")

        condition = None
        if not self.match("SEMICOLON"):
            condition = self.parse_expression()
        self.consume("SEMICOLON")

        update = None
        if not self.match("RPAREN"):
            update = self.parse_expression()
        self.consume("RPAREN")

        self.skip_newlines()
        body = self.parse_statement()

        return ForNode(init, condition, update, body)

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
                expr = FunctionCallNode(expr, args)
            elif self.match("INCREMENT", "DECREMENT"):
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

    def parse_primary_expression(self):
        if self.match("IDENTIFIER"):
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

        elif self.match("INTEGER", "FLOAT_NUM", "FLOAT", "STRING"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("TRUE", "FALSE", "NULL", "NULLPTR"):
            value = self.current_token.value
            self.advance()
            return value

        elif self.match("CHAR") and self.current_token.value != "char":
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

    def parse_initializer_list(self):
        self.consume("LBRACE")
        elements = []

        while self.current_token and not self.match("RBRACE"):
            elements.append(self.parse_expression())
            if self.match("COMMA"):
                self.advance()
                if self.match("RBRACE"):
                    break
            else:
                break

        self.consume("RBRACE")
        return InitializerListNode(elements)

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

            if not self.is_type_token(allow_identifier=False):
                return False

            self.advance()
            while self.match("ASTERISK", "STAR"):
                self.advance()

            return self.match("RPAREN")
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

    def parse_argument_list(self):
        args = []

        if self.match("RPAREN"):
            return args

        while True:
            arg = self.parse_expression()
            args.append(arg)

            if self.match("COMMA"):
                self.advance()
            else:
                break

        return args

    def is_function_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier followed by (
        saved_pos = self.pos
        try:
            while self.match(*self.FUNCTION_SPECIFIER_TOKENS):
                self.advance()

            while self.match(*self.TYPE_QUALIFIER_TOKENS):
                self.advance()

            if self.is_type_token():
                self.advance()
                while self.match("ASTERISK", "STAR"):
                    self.advance()
                if self.match("IDENTIFIER"):
                    self.advance()
                    if self.match("LPAREN"):
                        return True
        except Exception:
            pass
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

        return False

    def is_variable_declaration(self) -> bool:
        # Simple heuristic: type followed by identifier not followed by (
        saved_pos = self.pos
        try:
            while self.match(
                "__SHARED__",
                "__CONSTANT__",
                "__DEVICE__",
                "STATIC",
                "EXTERN",
                "CONST",
                "VOLATILE",
                "UNSIGNED",
                "SIGNED",
            ):
                self.advance()

            if self.is_type_token():
                type_token = self.current_token.type
                self.advance()
                if type_token != "IDENTIFIER":
                    while self.match("ASTERISK", "STAR"):
                        self.advance()
                if self.match("IDENTIFIER"):
                    self.advance()
                    if self.match("SEMICOLON", "ASSIGN", "LBRACKET", "LPAREN"):
                        return True
        except Exception:
            pass
        finally:
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

        return False


def parse_hip_code(code: str) -> HipProgramNode:
    """Parse HIP code and return AST"""
    lexer = HipLexer(code)
    tokens = lexer.tokenize()
    parser = HipParser(tokens)
    return parser.parse()
