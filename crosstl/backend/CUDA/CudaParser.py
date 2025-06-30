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
    CudaBuiltinNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    KernelLaunchNode,
    KernelNode,
    MemberAccessNode,
    PreprocessorNode,
    ReturnNode,
    ShaderNode,
    SharedMemoryNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .CudaLexer import CudaLexer


class CudaParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None

    def parse(self):
        """Parse the entire CUDA program"""
        includes = []
        functions = []
        structs = []
        global_variables = []
        kernels = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                includes.append(self.parse_preprocessor())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif (
                self.current_token[0] in ["GLOBAL", "DEVICE", "HOST"]
                or self.peek_function()
            ):
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
                # Skip unexpected tokens
                self.eat(self.current_token[0])

        return ShaderNode(includes, functions, structs, global_variables, kernels)

    def peek_function(self):
        """Check if the next tokens form a function declaration"""
        # Look ahead for function pattern: [qualifiers] type name (
        saved_index = self.current_index

        # Skip qualifiers and type
        while saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
            "CONST",
            "STATIC",
            "INLINE",
            "EXTERN",
        ]:
            saved_index += 1

        # Skip type (could be multiple tokens)
        if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
            "VOID",
            "INT",
            "FLOAT",
            "DOUBLE",
            "CHAR",
            "SHORT",
            "LONG",
            "BOOL",
            "IDENTIFIER",
            "UNSIGNED",
            "SIGNED",
        ]:
            saved_index += 1

        # Check for function name followed by parenthesis
        if (
            saved_index < len(self.tokens) - 1
            and self.tokens[saved_index][0] == "IDENTIFIER"
            and self.tokens[saved_index + 1][0] == "LPAREN"
        ):
            return True

        return False

    def peek_variable(self):
        """Check if the next tokens form a variable declaration"""
        # Look for type followed by identifier and semicolon or assignment
        saved_index = self.current_index

        # Skip qualifiers
        while saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
            "CONST",
            "STATIC",
            "EXTERN",
        ]:
            saved_index += 1

        # Check for type
        if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
            "INT",
            "FLOAT",
            "DOUBLE",
            "CHAR",
            "SHORT",
            "LONG",
            "BOOL",
            "IDENTIFIER",
            "UNSIGNED",
            "SIGNED",
            "FLOAT2",
            "FLOAT3",
            "FLOAT4",
        ]:
            saved_index += 1
            # Check for identifier
            if (
                saved_index < len(self.tokens)
                and self.tokens[saved_index][0] == "IDENTIFIER"
            ):
                saved_index += 1
                # Check for semicolon or assignment
                if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
                    "SEMICOLON",
                    "ASSIGN",
                ]:
                    return True

        return False

    def eat(self, expected_type):
        """Consume a token of the expected type"""
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

    def parse_preprocessor(self):
        """Parse preprocessor directives"""
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

    def parse_struct(self):
        """Parse struct declaration"""
        self.eat("STRUCT")
        name = self.eat("IDENTIFIER")[1]
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            member = self.parse_variable_declaration()
            members.append(member)
            self.eat("SEMICOLON")

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        """Parse function declaration including kernels"""
        qualifiers = []

        # Parse CUDA qualifiers
        while self.current_token[0] in ["GLOBAL", "DEVICE", "HOST", "INLINE", "STATIC"]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        # Parse return type
        return_type = self.parse_type()

        # Parse function name
        name = self.eat("IDENTIFIER")[1]

        # Parse parameters
        params = self.parse_parameters()

        # Parse function body
        body = self.parse_block()

        # Check if it's a kernel function
        if "__global__" in qualifiers:
            return KernelNode(return_type, name, params, body)
        else:
            return FunctionNode(return_type, name, params, body, qualifiers)

    def parse_parameters(self):
        """Parse function parameters"""
        self.eat("LPAREN")
        params = []

        if self.current_token[0] != "RPAREN":
            params.append(self.parse_parameter())

            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.parse_parameter())

        self.eat("RPAREN")
        return params

    def parse_parameter(self):
        """Parse a single parameter"""
        param_type = self.parse_type()
        param_name = self.eat("IDENTIFIER")[1]
        return VariableNode(param_type, param_name)

    def parse_type(self):
        """Parse type specification"""
        type_parts = []

        # Handle qualifiers
        while self.current_token[0] in ["CONST", "VOLATILE", "UNSIGNED", "SIGNED"]:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])

        # Handle basic types
        if self.current_token[0] in [
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
            "SIZE_T",
        ]:
            type_parts.append(self.current_token[1])
            self.eat(self.current_token[0])
        elif self.current_token[0] == "IDENTIFIER":
            type_parts.append(self.current_token[1])
            self.eat("IDENTIFIER")

        # Handle pointer/reference
        while self.current_token[0] == "MULTIPLY":
            type_parts.append("*")
            self.eat("MULTIPLY")

        # Handle array brackets
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                type_parts.append(f"[{self.current_token[1]}]")
                self.eat("NUMBER")
            else:
                type_parts.append("[]")
            self.eat("RBRACKET")

        return " ".join(type_parts)

    def parse_global_variable(self):
        """Parse global variable declaration"""
        qualifiers = []

        # Parse CUDA memory qualifiers
        while self.current_token[0] in ["CONSTANT", "SHARED", "DEVICE", "MANAGED"]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        var = self.parse_variable_declaration()
        var.qualifiers = qualifiers
        self.eat("SEMICOLON")

        # Return specialized nodes for CUDA memory types
        if "__constant__" in qualifiers:
            return ConstantMemoryNode(var.vtype, var.name, var.value)
        elif "__shared__" in qualifiers:
            return SharedMemoryNode(var.vtype, var.name)
        else:
            return var

    def parse_variable_declaration(self):
        """Parse variable declaration"""
        qualifiers = []

        # Parse CUDA memory qualifiers
        while self.current_token[0] in ["SHARED", "CONSTANT"]:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])

        vtype = self.parse_type()
        name = self.eat("IDENTIFIER")[1]

        # Handle array declarations
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                size = self.current_token[1]
                vtype += f"[{size}]"
                self.eat("NUMBER")
            else:
                vtype += "[]"
            self.eat("RBRACKET")

        value = None
        if self.current_token[0] == "ASSIGN":
            self.eat("ASSIGN")
            value = self.parse_expression()

        var = VariableNode(vtype, name, value, qualifiers)

        # Return specialized nodes for CUDA memory types
        if "__shared__" in qualifiers:
            return SharedMemoryNode(vtype, name)
        elif "__constant__" in qualifiers:
            return ConstantMemoryNode(vtype, name, value)
        else:
            return var

    def parse_block(self):
        """Parse a block of statements"""
        self.eat("LBRACE")
        statements = []

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

        self.eat("RBRACE")
        return statements

    def parse_statement(self):
        """Parse a single statement"""
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
        elif self.current_token[0] == "LBRACE":
            return self.parse_block()
        elif self.is_variable_declaration():
            var = self.parse_variable_declaration()
            self.eat("SEMICOLON")
            return var
        else:
            # Expression statement or assignment
            expr = self.parse_assignment_expression()
            self.eat("SEMICOLON")
            return expr

    def is_variable_declaration(self):
        """Check if current position is a variable declaration"""
        # Simple heuristic: type followed by identifier
        if self.current_token[0] in [
            "INT",
            "FLOAT",
            "DOUBLE",
            "CHAR",
            "BOOL",
            "VOID",
            "FLOAT2",
            "FLOAT3",
            "FLOAT4",
            "IDENTIFIER",
            "SHARED",
            "CONSTANT",
        ]:
            # Look ahead for identifier
            saved_index = self.current_index

            # Skip qualifiers
            while saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
                "SHARED",
                "CONSTANT",
            ]:
                saved_index += 1

            # Check for type
            if saved_index < len(self.tokens) and self.tokens[saved_index][0] in [
                "INT",
                "FLOAT",
                "DOUBLE",
                "CHAR",
                "BOOL",
                "VOID",
                "FLOAT2",
                "FLOAT3",
                "FLOAT4",
                "IDENTIFIER",
            ]:
                saved_index += 1

                # Check for identifier
                if (
                    saved_index < len(self.tokens)
                    and self.tokens[saved_index][0] == "IDENTIFIER"
                ):
                    return True
        return False

    def parse_if_statement(self):
        """Parse if statement"""
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
        """Parse for loop"""
        self.eat("FOR")
        self.eat("LPAREN")

        # Parse initialization
        init = None
        if self.current_token[0] != "SEMICOLON":
            if self.is_variable_declaration():
                init = self.parse_variable_declaration()
            else:
                init = self.parse_expression()
        self.eat("SEMICOLON")

        # Parse condition
        condition = None
        if self.current_token[0] != "SEMICOLON":
            condition = self.parse_expression()
        self.eat("SEMICOLON")

        # Parse update
        update = None
        if self.current_token[0] != "RPAREN":
            update = self.parse_expression()
        self.eat("RPAREN")

        # Parse body
        body = self.parse_statement()

        return ForNode(init, condition, update, body)

    def parse_while_statement(self):
        """Parse while loop"""
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_statement()

        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        """Parse do-while loop"""
        self.eat("DO")
        body = self.parse_statement()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")

        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        """Parse switch statement"""
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default_case = None

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "CASE":
                self.eat("CASE")
                value = self.parse_expression()
                self.eat("COLON")
                body = []
                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE"]:
                    body.append(self.parse_statement())
                cases.append(CaseNode(value, body))
            elif self.current_token[0] == "DEFAULT":
                self.eat("DEFAULT")
                self.eat("COLON")
                default_case = []
                while self.current_token[0] not in ["CASE", "RBRACE"]:
                    default_case.append(self.parse_statement())
            else:
                self.eat(self.current_token[0])  # Skip unexpected tokens

        self.eat("RBRACE")
        return SwitchNode(expression, cases, default_case)

    def parse_return_statement(self):
        """Parse return statement"""
        self.eat("RETURN")

        value = None
        if self.current_token[0] != "SEMICOLON":
            value = self.parse_expression()

        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_sync_statement(self):
        """Parse CUDA synchronization statements"""
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

    def parse_expression(self):
        """Parse expression with precedence"""
        return self.parse_ternary_expression()

    def parse_ternary_expression(self):
        """Parse ternary conditional operator"""
        expr = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        """Parse logical OR expression"""
        left = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and_expression(self):
        """Parse logical AND expression"""
        left = self.parse_equality_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_equality_expression(self):
        """Parse equality expression"""
        left = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_relational_expression(self):
        """Parse relational expression"""
        left = self.parse_additive_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_additive_expression(self):
        """Parse additive expression"""
        left = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_multiplicative_expression(self):
        """Parse multiplicative expression"""
        left = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MODULO"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary_expression(self):
        """Parse unary expression"""
        if self.current_token[0] in ["PLUS", "MINUS", "LOGICAL_NOT", "BITWISE_NOT"]:
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
        """Parse postfix expression"""
        left = self.parse_primary_expression()

        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.eat("IDENTIFIER")[1]
                left = MemberAccessNode(left, member, False)
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
                # Function call
                self.eat("LPAREN")
                args = []
                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")

                # Check for atomic operations
                if isinstance(left, str) and left.startswith("atomic"):
                    left = AtomicOperationNode(left, args)
                else:
                    left = FunctionCallNode(left, args)
            elif self.current_token[0] == "KERNEL_LAUNCH_START":
                # Kernel launch: kernel<<<blocks, threads>>>(args)
                return self.parse_kernel_launch(left)
            elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                left = UnaryOpNode(f"post{op}", left)
            else:
                break

        return left

    def parse_kernel_launch(self, kernel_name):
        """Parse CUDA kernel launch syntax"""
        self.eat("KERNEL_LAUNCH_START")

        # Parse blocks
        blocks = self.parse_expression()
        self.eat("COMMA")

        # Parse threads
        threads = self.parse_expression()

        # Parse optional shared memory and stream
        shared_mem = None
        stream = None

        if self.current_token[0] == "COMMA":
            self.eat("COMMA")
            shared_mem = self.parse_expression()

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                stream = self.parse_expression()

        self.eat("KERNEL_LAUNCH_END")

        # Parse arguments
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
        """Parse primary expression"""
        if self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        elif self.current_token[0] == "STRING":
            value = self.current_token[1]
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
        elif self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name
        elif self.current_token[0] in [
            "ATOMICADD",
            "ATOMICSUB",
            "ATOMICMAX",
            "ATOMICMIN",
            "ATOMICEXCH",
            "ATOMICCAS",
        ]:
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

            # Check for component access (.x, .y, .z)
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                component = self.eat("IDENTIFIER")[1]
                return CudaBuiltinNode(builtin_name, component)
            else:
                return CudaBuiltinNode(builtin_name)
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            if self.current_token[0] in ["INT", "FLOAT", "DOUBLE", "CHAR"]:
                # Type cast
                target_type = self.parse_type()
                self.eat("RPAREN")
                expr = self.parse_unary_expression()
                return CastNode(target_type, expr)
            else:
                # Parenthesized expression
                expr = self.parse_expression()
                self.eat("RPAREN")
                return expr
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def parse_assignment_expression(self):
        """Parse assignment expression"""
        left = self.parse_expression()

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
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression()
            return AssignmentNode(left, right, op)

        return left
