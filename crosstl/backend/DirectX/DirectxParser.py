from .DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    WhileNode,
    DoWhileNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    PragmaNode,
    IncludeNode,
    SwitchNode,
    CaseNode,
    SwitchStatementNode,
    SwitchCaseNode,
    TernaryOpNode,
)
from .DirectxLexer import HLSLLexer, Lexer, TokenType


class HLSLParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = tokens[0] if tokens else None
        print(f"Initializing parser with {len(tokens)} tokens")

    def parse(self):
        structs = []
        functions = []
        global_variables = []
        cbuffers = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PRAGMA":
                # Handle pragma directive
                self.eat("PRAGMA")
                directive_text = self.current_token[1]
                self.eat("IDENTIFIER")
                value = []
                while self.current_token[0] != "SEMICOLON":
                    value.append(self.current_token[1])
                    self.eat(self.current_token[0])
                self.eat("SEMICOLON")
                directive = PragmaNode(directive_text, " ".join(value))
                structs.append(directive)
            elif self.current_token[0] == "INCLUDE":
                # Handle include directive
                self.eat("INCLUDE")
                path = self.current_token[1]
                self.eat("STRING")
                structs.append(IncludeNode(path))
            elif self.current_token[0] == "STRUCT":
                # Handle struct definition
                s = self.parse_struct()
                structs.append(s)
            elif self.current_token[0] == "CBUFFER":
                # Handle cbuffer
                c = self.parse_cbuffer()
                cbuffers.append(c)
            elif self.current_token[0] in [
                "VOID",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "FVECTOR",
                "MATRIX",
            ]:
                # Function or variable declaration
                return_type = self.current_token[1]
                self.eat(self.current_token[0])
                name = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "LPAREN":
                    # Function declaration
                    f = self.parse_function(return_type, name)
                    functions.append(f)
                else:
                    # Global variable
                    semantic = None
                    if self.current_token[0] == "COLON":
                        self.eat("COLON")
                        semantic = self.current_token[1]
                        self.eat("IDENTIFIER")

                    self.eat("SEMICOLON")
                    global_variables.append(VariableNode(return_type, name, semantic))
            elif self.current_token[0] == "HALF":
                # Handle half type
                return_type = self.current_token[1]
                self.eat("HALF")
                name = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "LPAREN":
                    # Function declaration
                    f = self.parse_function(return_type, name)
                    functions.append(f)
                else:
                    # Global variable
                    semantic = None
                    if self.current_token[0] == "COLON":
                        self.eat("COLON")
                        semantic = self.current_token[1]
                        self.eat("IDENTIFIER")

                    self.eat("SEMICOLON")
                    global_variables.append(VariableNode(return_type, name, semantic))
            elif (
                self.current_token[0] == "TEXTURE2D"
                or self.current_token[0] == "SAMPLER_STATE"
            ):
                # Handle texture and sampler state declarations
                var_type = self.current_token[1]
                self.eat(self.current_token[0])
                name = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                    global_variables.append(VariableNode(var_type, name))
                elif self.current_token[0] == "COLON":
                    self.eat("COLON")
                    semantic = self.current_token[1]
                    self.eat("IDENTIFIER")
                    self.eat("SEMICOLON")
                    global_variables.append(VariableNode(var_type, name, semantic))
                else:
                    # This handles texture declarations with register
                    self.skip_until("SEMICOLON")
                    self.eat("SEMICOLON")
                    global_variables.append(VariableNode(var_type, name))
            elif self.current_token[0] == "IDENTIFIER":
                # Handle custom type function or variable
                return_type = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "IDENTIFIER":
                    # Function or variable with custom type
                    name = self.current_token[1]
                    self.eat("IDENTIFIER")

                    if self.current_token[0] == "LPAREN":
                        # Function declaration
                        f = self.parse_function(return_type, name)
                        functions.append(f)
                    else:
                        # Global variable
                        semantic = None
                        if self.current_token[0] == "COLON":
                            self.eat("COLON")
                            semantic = self.current_token[1]
                            self.eat("IDENTIFIER")

                        self.eat("SEMICOLON")
                        global_variables.append(
                            VariableNode(return_type, name, semantic)
                        )
                else:
                    print(f"Skipping token: {self.current_token}")
                    self.eat(self.current_token[0])
            else:
                print(f"Skipping token: {self.current_token}")
                self.eat(self.current_token[0])

        return ShaderNode(structs, functions, global_variables, cbuffers)

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

    def skip_until(self, token_type):
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.current_index += 1
            if self.current_index < len(self.tokens):
                self.current_token = self.tokens[self.current_index]
            else:
                self.current_token = ("EOF", "")

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        semantic = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            semantic = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            # Parse member type
            member_type = self.current_token[1]
            self.eat(self.current_token[0])  # eat the type token

            # Parse member name
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Check for semantic
            member_semantic = None
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                member_semantic = self.current_token[1]
                self.eat("IDENTIFIER")

            self.eat("SEMICOLON")

            members.append(VariableNode(member_type, member_name, member_semantic))

        self.eat("RBRACE")

        # Check for variable declarations
        variables = []
        if self.current_token[0] == "IDENTIFIER":
            variables.append(self.current_token[1])
            self.eat("IDENTIFIER")

            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                variables.append(self.current_token[1])
                self.eat("IDENTIFIER")

            self.eat("SEMICOLON")

        return StructNode(name, members, variables, semantic)

    def parse_cbuffer(self):
        self.eat("CBUFFER")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Skip register assignment if present
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            self.eat("REGISTER")
            self.eat("LPAREN")
            self.eat("IDENTIFIER")
            self.eat("RPAREN")

        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            # Parse member type
            member_type = self.current_token[1]
            self.eat(self.current_token[0])  # eat the type token

            # Parse member name
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Skip array declaration if present
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                self.eat("NUMBER")
                self.eat("RBRACKET")

            self.eat("SEMICOLON")

            members.append(VariableNode(member_type, member_name))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self, return_type, name):
        params = self.parse_parameters()

        # Check for function semantic
        semantic = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            semantic = self.current_token[1]
            self.eat("IDENTIFIER")

        # Determine function qualifier based on name
        qualifier = None
        if name.startswith("VS"):
            qualifier = "vertex"
        elif name.startswith("PS"):
            qualifier = "fragment"
        elif name.startswith("CS"):
            qualifier = "compute"

        # Parse function body
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return FunctionNode(return_type, name, params, body, qualifier, semantic)

    def parse_parameters(self):
        self.eat("LPAREN")

        params = []
        if self.current_token[0] != "RPAREN":
            # Parse first parameter
            param_type = self.current_token[1]
            self.eat(self.current_token[0])  # eat the type token

            param_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Check for semantic
            param_semantic = None
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                param_semantic = self.current_token[1]
                self.eat("IDENTIFIER")

            params.append(VariableNode(param_type, param_name, param_semantic))

            # Parse additional parameters
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")

                param_type = self.current_token[1]
                self.eat(self.current_token[0])  # eat the type token

                param_name = self.current_token[1]
                self.eat("IDENTIFIER")

                # Check for semantic
                param_semantic = None
                if self.current_token[0] == "COLON":
                    self.eat("COLON")
                    param_semantic = self.current_token[1]
                    self.eat("IDENTIFIER")

                params.append(VariableNode(param_type, param_name, param_semantic))

        self.eat("RPAREN")
        return params

    def parse_block(self):
        statements = []

        while self.current_token[0] != "RBRACE":
            try:
                # Variable declaration
                if self.current_token[0] in [
                    "FLOAT",
                    "INT",
                    "UINT",
                    "BOOL",
                    "FVECTOR",
                    "MATRIX",
                    "DOUBLE",
                    "HALF",
                ]:
                    var_type = self.current_token[1]
                    self.eat(self.current_token[0])

                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")

                    if self.current_token[0] == "SEMICOLON":
                        # Simple variable declaration
                        self.eat("SEMICOLON")
                        statements.append(VariableNode(var_type, var_name))
                    elif self.current_token[0] == "EQUALS":
                        # Variable initialization
                        self.eat("EQUALS")
                        value = self.parse_expression()
                        self.eat("SEMICOLON")
                        var = VariableNode(var_type, var_name)
                        statements.append(AssignmentNode(var, value))
                    else:
                        # Skip complex declarations for now
                        self.skip_until("SEMICOLON")
                        self.eat("SEMICOLON")
                        statements.append(VariableNode(var_type, var_name))
                # Assignment or function call
                elif self.current_token[0] == "IDENTIFIER":
                    left = self.parse_expression()

                    if self.current_token[0] in [
                        "EQUALS",
                        "PLUS_EQUALS",
                        "MINUS_EQUALS",
                        "MULTIPLY_EQUALS",
                        "DIVIDE_EQUALS",
                    ]:
                        op = self.current_token[1]
                        self.eat(self.current_token[0])
                        right = self.parse_expression()
                        self.eat("SEMICOLON")
                        statements.append(AssignmentNode(left, right, op))
                    else:
                        # Function call or other expression
                        if self.current_token[0] != "SEMICOLON":
                            # If not followed by semicolon, there might be comments or other tokens to skip
                            self.skip_until("SEMICOLON")
                        self.eat("SEMICOLON")
                        statements.append(left)
                # Return statement
                elif self.current_token[0] == "RETURN":
                    self.eat("RETURN")
                    value = self.parse_expression()
                    if self.current_token[0] != "SEMICOLON":
                        # Skip any comments until semicolon
                        self.skip_until("SEMICOLON")
                    self.eat("SEMICOLON")
                    statements.append(ReturnNode(value))
                # If statement
                elif self.current_token[0] == "IF":
                    statements.append(self.parse_if_statement())
                # Else-if statement - should not appear directly in a block, this is handled in parse_if_statement
                elif self.current_token[0] == "ELSE_IF":
                    # Handle "else if" as a separate token
                    self.eat("ELSE_IF")
                    self.eat("LPAREN")
                    condition = self.parse_expression()
                    self.eat("RPAREN")

                    self.eat("LBRACE")
                    if_body = self.parse_block()
                    self.eat("RBRACE")

                    # Check for else
                    else_body = None
                    if self.current_token[0] == "ELSE":
                        self.eat("ELSE")
                        self.eat("LBRACE")
                        else_body = self.parse_block()
                        self.eat("RBRACE")

                    statements.append(IfNode(condition, if_body, else_body))
                # For loop
                elif self.current_token[0] == "FOR":
                    statements.append(self.parse_for_loop())
                # While loop
                elif self.current_token[0] == "WHILE":
                    statements.append(self.parse_while_loop())
                # Do-while loop
                elif self.current_token[0] == "DO":
                    statements.append(self.parse_do_while_loop())
                # Switch statement
                elif self.current_token[0] == "SWITCH":
                    statements.append(self.parse_switch_statement())
                elif self.current_token[0] == "BREAK":
                    self.eat("BREAK")
                    self.eat("SEMICOLON")
                    statements.append(
                        "break"
                    )  # Just return a string for break statements
                else:
                    print(f"Skipping unexpected token in block: {self.current_token}")
                    self.eat(self.current_token[0])
            except SyntaxError as e:
                # Handle parsing errors gracefully
                print(f"Error parsing block: {e}")
                # Skip to next semicolon or end of block
                self.skip_until("SEMICOLON")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")

        return statements

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")

        # Parse if body
        self.eat("LBRACE")
        if_body = self.parse_block()
        self.eat("RBRACE")

        # Check for else
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")

            # Check if this is an "else if" construct
            if self.current_token[0] == "IF":
                # Else-if statement handled as nested if
                else_body = self.parse_if_statement()
            elif self.current_token[0] == "ELSE_IF":
                # Direct "else if" token (some lexers may combine these)
                self.eat("ELSE_IF")
                self.eat("LPAREN")
                else_if_condition = self.parse_expression()
                self.eat("RPAREN")

                self.eat("LBRACE")
                else_if_body = self.parse_block()
                self.eat("RBRACE")

                # Check for additional else
                else_if_else = None
                if self.current_token[0] == "ELSE":
                    self.eat("ELSE")
                    self.eat("LBRACE")
                    else_if_else = self.parse_block()
                    self.eat("RBRACE")

                else_body = IfNode(else_if_condition, else_if_body, else_if_else)
            else:
                # Regular else block
                self.eat("LBRACE")
                else_body = self.parse_block()
                self.eat("RBRACE")

        return IfNode(condition, if_body, else_body)

    def parse_for_loop(self):
        self.eat("FOR")
        self.eat("LPAREN")

        # Initialization
        if self.current_token[0] in ["FLOAT", "INT", "UINT", "BOOL"]:
            var_type = self.current_token[1]
            self.eat(self.current_token[0])

            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            self.eat("EQUALS")
            init_value = self.parse_expression()
            init = AssignmentNode(VariableNode(var_type, var_name), init_value)
        else:
            init = self.parse_expression()

        self.eat("SEMICOLON")

        # Condition
        condition = self.parse_expression()
        self.eat("SEMICOLON")

        # Update
        update = self.parse_expression()
        self.eat("RPAREN")

        # Body
        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return ForNode(init, condition, update, body)

    def parse_while_loop(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        return WhileNode(condition, body)

    def parse_do_while_loop(self):
        self.eat("DO")

        self.eat("LBRACE")
        body = self.parse_block()
        self.eat("RBRACE")

        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")

        return DoWhileNode(condition, body)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")

        self.eat("LBRACE")

        cases = []
        default_body = None

        while self.current_token[0] in ("CASE", "DEFAULT"):
            if self.current_token[0] == "CASE":
                cases.append(self.parse_switch_case())
            else:  # DEFAULT case
                self.eat("DEFAULT")
                self.eat("COLON")

                default_stmts = []
                while (
                    self.current_token[0] != "CASE"
                    and self.current_token[0] != "DEFAULT"
                    and self.current_token[0] != "RBRACE"
                ):
                    default_stmts.append(self.parse_statement())

                default_body = default_stmts

        self.eat("RBRACE")

        return SwitchNode(condition, cases, default_body)

    def parse_switch_case(self):
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        body = []
        while (
            self.current_token[0] != "CASE"
            and self.current_token[0] != "DEFAULT"
            and self.current_token[0] != "RBRACE"
        ):
            stmt = self.parse_statement()
            if stmt:  # Some statements might return None
                body.append(stmt)

        return CaseNode(value, body)

    def parse_statement(self):
        if self.current_token[0] in [
            "FLOAT",
            "INT",
            "UINT",
            "BOOL",
            "FVECTOR",
            "MATRIX",
            "DOUBLE",
            "HALF",
        ]:
            var_type = self.current_token[1]
            self.eat(self.current_token[0])

            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "SEMICOLON":
                # Simple variable declaration
                self.eat("SEMICOLON")
                return VariableNode(var_type, var_name)
            elif self.current_token[0] == "EQUALS":
                # Variable initialization
                self.eat("EQUALS")
                value = self.parse_expression()
                self.eat("SEMICOLON")
                var = VariableNode(var_type, var_name)
                return AssignmentNode(var, value)
            else:
                # Skip complex declarations for now
                self.skip_until("SEMICOLON")
                self.eat("SEMICOLON")
                return VariableNode(var_type, var_name)
        elif self.current_token[0] == "IDENTIFIER":
            left = self.parse_expression()

            if self.current_token[0] in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
            ]:
                op = self.current_token[1]
                self.eat(self.current_token[0])
                right = self.parse_expression()
                self.eat("SEMICOLON")
                return AssignmentNode(left, right, op)
            else:
                # Function call or other expression
                self.eat("SEMICOLON")
                return left
        elif self.current_token[0] == "RETURN":
            self.eat("RETURN")
            value = self.parse_expression()
            self.eat("SEMICOLON")
            return ReturnNode(value)
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_loop()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_loop()
        elif self.current_token[0] == "DO":
            return self.parse_do_while_loop()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return "break"  # Just return a string for break statements
        else:
            print(f"Skipping unexpected token in statement: {self.current_token}")
            self.eat(self.current_token[0])
            return None

    def parse_expression(self):
        return self.parse_conditional_expression()

    def parse_conditional_expression(self):
        expr = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_conditional_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

    def parse_logical_or_expression(self):
        expr = self.parse_logical_and_expression()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_logical_and_expression(self):
        expr = self.parse_equality_expression()

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_equality_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_equality_expression(self):
        expr = self.parse_relational_expression()

        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_relational_expression(self):
        expr = self.parse_additive_expression()

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_additive_expression(self):
        expr = self.parse_multiplicative_expression()

        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_multiplicative_expression(self):
        expr = self.parse_unary_expression()

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr

    def parse_unary_expression(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary_expression()
            return UnaryOpNode(op, operand)

        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        expr = self.parse_primary_expression()

        while self.current_token[0] in ["LBRACKET", "DOT", "LPAREN"]:
            if self.current_token[0] == "LBRACKET":
                # Array indexing
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                expr = BinaryOpNode(expr, "[]", index)
            elif self.current_token[0] == "DOT":
                # Member access
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                expr = MemberAccessNode(expr, member)
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
                expr = FunctionCallNode(expr, args)

        return expr

    def parse_primary_expression(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return name
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return float(value) if "." in value else int(value)
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] in ["FVECTOR", "MATRIX"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])

            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
                args = []

                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())

                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())

                self.eat("RPAREN")
                return VectorConstructorNode(type_name, args)
            else:
                return type_name
        else:
            raise SyntaxError(
                f"Unexpected token in primary expression: {self.current_token}"
            )

    def parse_bitwise_expression(self):
        expr = self.parse_additive_expression()

        while self.current_token[0] in [
            "BITWISE_AND",
            "BITWISE_OR",
            "BITWISE_XOR",
            "AMPERSAND",
            "PIPE",
            "CARET",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive_expression()
            expr = BinaryOpNode(expr, op, right)

        return expr
