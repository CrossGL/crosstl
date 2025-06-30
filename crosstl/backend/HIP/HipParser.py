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

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None

    def error(self, message: str):
        """Raise a parsing error with context"""
        token_info = (
            f"at token '{self.current_token.value}'"
            if self.current_token
            else "at end of input"
        )
        raise SyntaxError(f"Parse error {token_info}: {message}")

    def advance(self):
        """Move to the next token"""
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
        """Consume a token of the expected type"""
        if not self.current_token:
            self.error(f"Expected {expected_type} but reached end of input")

        if self.current_token.type != expected_type:
            self.error(f"Expected {expected_type}, got {self.current_token.type}")

        token = self.current_token
        self.advance()
        return token

    def match(self, *token_types: str) -> bool:
        """Check if current token matches any of the given types"""
        if not self.current_token:
            return False
        return self.current_token.type in token_types

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

        # Try to parse function or variable declaration
        if self.is_function_declaration():
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
        """Parse function with device/host qualifiers"""
        qualifiers = []

        # Parse all qualifiers
        while self.match(
            "__DEVICE__", "__HOST__", "__GLOBAL__", "__FORCEINLINE__", "__NOINLINE__"
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        # Parse return type
        return_type = self.parse_type()

        # Parse function name
        name = self.consume("IDENTIFIER").value

        # Parse parameters
        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")

        # Parse function body or semicolon for declaration
        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        function = FunctionNode(return_type, name, params, body, qualifiers)

        # If it has __global__ qualifier, it's a kernel
        if "__global__" in qualifiers:
            return KernelNode(return_type, name, params, body)

        return function

    def parse_simple_function(self):
        """Parse a simple function without qualifiers"""
        return_type = self.parse_type()
        name = self.consume("IDENTIFIER").value

        # Parse parameters
        self.consume("LPAREN")
        params = self.parse_parameter_list()
        self.consume("RPAREN")

        # Parse function body or semicolon for declaration
        body = None
        if self.match("LBRACE"):
            body = self.parse_block()
        elif self.match("SEMICOLON"):
            self.advance()

        return FunctionNode(return_type, name, params, body)

    def parse_struct(self):
        """Parse struct definitions"""
        self.consume("STRUCT")

        name = None
        if self.match("IDENTIFIER"):
            name = self.current_token.value
            self.advance()

        # Parse struct body
        members = []
        if self.match("LBRACE"):
            self.consume("LBRACE")
            while self.current_token and not self.match("RBRACE"):
                if self.match("NEWLINE", "SEMICOLON"):
                    self.advance()
                    continue

                # Parse struct member
                member = self.parse_struct_member()
                if member:
                    members.append(member)

            self.consume("RBRACE")

        # Optional semicolon
        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)

    def parse_class(self):
        """Parse class definitions (treat similar to struct for now)"""
        self.consume("CLASS")

        name = self.consume("IDENTIFIER").value

        # Parse class body (simplified)
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

                # Parse class member
                member = self.parse_struct_member()
                if member:
                    members.append(member)

            self.consume("RBRACE")

        # Optional semicolon
        if self.match("SEMICOLON"):
            self.advance()

        return StructNode(name, members)  # Treat class as struct for simplicity

    def parse_struct_member(self):
        """Parse struct member"""
        try:
            # Parse type
            member_type = self.parse_type()

            # Parse member name
            name = self.consume("IDENTIFIER").value

            # Handle array declarations
            if self.match("LBRACKET"):
                self.consume("LBRACKET")
                # Skip array size for now
                while self.current_token and not self.match("RBRACKET"):
                    self.advance()
                self.consume("RBRACKET")

            # Consume semicolon
            if self.match("SEMICOLON"):
                self.advance()

            return VariableNode(member_type, name)
        except:
            # Skip problematic member
            while self.current_token and not self.match("SEMICOLON", "RBRACE"):
                self.advance()
            if self.match("SEMICOLON"):
                self.advance()
            return None

    def parse_variable_declaration(self):
        """Parse variable declarations"""
        qualifiers = []

        # Parse storage qualifiers
        while self.match(
            "__SHARED__", "__CONSTANT__", "__DEVICE__", "STATIC", "EXTERN"
        ):
            qualifiers.append(self.current_token.value)
            self.advance()

        # Parse type
        var_type = self.parse_type()

        # Parse variable name
        name = self.consume("IDENTIFIER").value

        # Parse initializer (optional)
        value = None
        if self.match("ASSIGN"):
            self.advance()
            value = self.parse_expression()

        # Handle array declarations
        if self.match("LBRACKET"):
            self.consume("LBRACKET")
            # Skip array size for now
            while self.current_token and not self.match("RBRACKET"):
                self.advance()
            self.consume("RBRACKET")

        # Consume semicolon
        if self.match("SEMICOLON"):
            self.advance()

        return VariableNode(var_type, name, value, qualifiers)

    def parse_type(self):
        """Parse type specification"""
        type_name = ""

        # Handle basic types and HIP vector types
        if self.match(
            "INT",
            "FLOAT",
            "DOUBLE",
            "BOOL",
            "VOID",
            "CHAR",
            "SHORT",
            "LONG",
            "UNSIGNED",
            "SIGNED",
        ):
            type_name = self.current_token.value
            self.advance()
        elif self.match(
            "FLOAT2",
            "FLOAT3",
            "FLOAT4",
            "INT2",
            "INT3",
            "INT4",
            "DOUBLE2",
            "DOUBLE3",
            "DOUBLE4",
        ):
            type_name = self.current_token.value
            self.advance()
        elif self.match("IDENTIFIER"):
            type_name = self.current_token.value
            self.advance()
        else:
            type_name = "int"  # Default type

        # Handle pointer types
        while self.match("ASTERISK", "STAR"):
            type_name += "*"
            self.advance()

        return type_name

    def parse_parameter_list(self):
        """Parse function parameter list"""
        params = []

        if self.match("RPAREN"):
            return params

        while True:
            # Parse parameter type
            param_type = self.parse_type()

            # Parse parameter name (optional)
            param_name = ""
            if self.match("IDENTIFIER"):
                param_name = self.current_token.value
                self.advance()

            params.append({"type": param_type, "name": param_name})

            if self.match("COMMA"):
                self.advance()
            else:
                break

        return params

    def parse_return_statement(self):
        """Parse return statement"""
        self.consume("RETURN")

        value = None
        if not self.match("SEMICOLON"):
            value = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()

        return ReturnNode(value)

    def parse_if_statement(self):
        """Parse if statement"""
        self.consume("IF")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        # Parse if body - could be a single statement or block
        if_body = None
        if self.match("LBRACE"):
            if_body = self.parse_block()
        else:
            if_body = self.parse_statement()

        else_body = None
        if self.match("ELSE"):
            self.advance()
            if self.match("LBRACE"):
                else_body = self.parse_block()
            else:
                else_body = self.parse_statement()

        return IfNode(condition, if_body, else_body)

    def parse_for_statement(self):
        """Parse for statement"""
        self.consume("FOR")
        self.consume("LPAREN")

        init = None
        if not self.match("SEMICOLON"):
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

        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_while_statement(self):
        """Parse while statement"""
        self.consume("WHILE")
        self.consume("LPAREN")
        condition = self.parse_expression()
        self.consume("RPAREN")

        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_block(self):
        """Parse a block of statements"""
        self.consume("LBRACE")
        statements = []

        while self.current_token and not self.match("RBRACE"):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        self.consume("RBRACE")
        return statements

    def parse_expression_statement(self):
        """Parse expression statement"""
        expr = self.parse_expression()

        if self.match("SEMICOLON"):
            self.advance()

        return expr

    def parse_expression(self):
        """Parse expressions (simplified)"""
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        """Parse assignment expressions"""
        left = self.parse_logical_or_expression()

        if self.match(
            "ASSIGN", "PLUS_ASSIGN", "MINUS_ASSIGN", "MULTIPLY_ASSIGN", "DIVIDE_ASSIGN"
        ):
            op = self.current_token.value
            self.advance()
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op)

        return left

    def parse_logical_or_expression(self):
        """Parse logical OR expressions"""
        left = self.parse_logical_and_expression()

        while self.match("LOGICAL_OR"):
            op = self.current_token.value
            self.advance()
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        """Parse logical AND expressions"""
        left = self.parse_equality_expression()

        while self.match("LOGICAL_AND"):
            op = self.current_token.value
            self.advance()
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        """Parse equality expressions"""
        left = self.parse_relational_expression()

        while self.match("EQ", "NE"):
            op = self.current_token.value
            self.advance()
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        """Parse relational expressions"""
        left = self.parse_additive_expression()

        while self.match("LT", "LE", "GT", "GE"):
            op = self.current_token.value
            self.advance()
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        """Parse additive expressions"""
        left = self.parse_multiplicative_expression()

        while self.match("PLUS", "MINUS"):
            op = self.current_token.value
            self.advance()
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        """Parse multiplicative expressions"""
        left = self.parse_unary_expression()

        while self.match("MULTIPLY", "STAR", "DIVIDE", "SLASH", "MODULO", "PERCENT"):
            op = self.current_token.value
            self.advance()
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        """Parse unary expressions"""
        if self.match(
            "PLUS", "MINUS", "NOT", "BITWISE_NOT", "INCREMENT", "DECREMENT", "STAR"
        ):
            op = self.current_token.value
            self.advance()
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        """Parse postfix expressions"""
        expr = self.parse_primary_expression()

        while True:
            if self.match("LBRACKET"):  # Array access
                self.consume("LBRACKET")
                index = self.parse_expression()
                self.consume("RBRACKET")
                expr = ArrayAccessNode(expr, index)
            elif self.match("DOT"):  # Member access
                self.consume("DOT")
                member = self.consume("IDENTIFIER").value
                expr = MemberAccessNode(expr, member, False)
            elif self.match("ARROW"):  # Pointer member access
                self.consume("ARROW")
                member = self.consume("IDENTIFIER").value
                expr = MemberAccessNode(expr, member, True)
            elif self.match("LPAREN"):  # Function call
                self.consume("LPAREN")
                args = self.parse_argument_list()
                self.consume("RPAREN")
                expr = FunctionCallNode(expr, args)
            elif self.match("INCREMENT", "DECREMENT"):  # Postfix increment/decrement
                op = self.current_token.value
                self.advance()
                expr = UnaryOpNode(op + "_POST", expr)
            else:
                break

        return expr

    def parse_primary_expression(self):
        """Parse primary expressions"""
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
            # Handle built-in variables as specific tokens
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

        elif self.match("LPAREN"):
            self.consume("LPAREN")
            expr = self.parse_expression()
            self.consume("RPAREN")
            return expr

        else:
            self.error(
                f"Unexpected token in expression: {self.current_token.type if self.current_token else 'EOF'}"
            )

    def parse_argument_list(self):
        """Parse function call argument list"""
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
        """Check if current position is a function declaration"""
        # Simple heuristic: type followed by identifier followed by (
        saved_pos = self.pos
        try:
            # Skip type
            if self.match(
                "IDENTIFIER", "INT", "FLOAT", "DOUBLE", "VOID", "BOOL", "CHAR"
            ):
                self.advance()
                if self.match("IDENTIFIER"):
                    self.advance()
                    if self.match("LPAREN"):
                        return True
        except:
            pass
        finally:
            # Restore position
            self.pos = saved_pos
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else None
            )

        return False

    def is_variable_declaration(self) -> bool:
        """Check if current position is a variable declaration"""
        # Simple heuristic: type followed by identifier (not followed by ()
        saved_pos = self.pos
        try:
            # Skip qualifiers
            while self.match(
                "__SHARED__", "__CONSTANT__", "__DEVICE__", "STATIC", "EXTERN"
            ):
                self.advance()

            # Check type
            if self.match(
                "IDENTIFIER",
                "INT",
                "FLOAT",
                "DOUBLE",
                "VOID",
                "BOOL",
                "CHAR",
                "FLOAT2",
                "FLOAT3",
                "FLOAT4",
            ):
                self.advance()
                if self.match("IDENTIFIER"):
                    self.advance()
                    # Not a function if not followed by (
                    if not self.match("LPAREN"):
                        return True
        except:
            pass
        finally:
            # Restore position
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
