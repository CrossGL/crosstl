from .DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
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
)
from .DirectxLexer import HLSLLexer


class HLSLParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.skip_comments()  # Skip any initial comments

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()  # Skip comments after eating a token
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        input_struct = None
        output_struct = None
        functions = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "STRUCT":
                struct = self.parse_struct()
                if struct.name == "VS_INPUT":
                    input_struct = struct
                elif struct.name == "VS_OUTPUT":
                    output_struct = struct
            elif self.current_token[0] in ["VOID", "FLOAT", "FVECTOR", "IDENTIFIER"]:
                functions.append(self.parse_function())
            else:
                self.eat(self.current_token[0])  # Skip unknown tokens

        return ShaderNode(input_struct, output_struct, functions)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])  # Eat the type (FVECTOR, FLOAT, etc.)
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            semantic = None
            if self.current_token[0] == "SEMANTIC":
                semantic = self.current_token[1]
                self.eat("SEMANTIC")
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name, semantic))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        return_type = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        if self.current_token[0] == "SEMANTIC":
            self.eat("SEMANTIC")
        body = self.parse_block()
        return FunctionNode(return_type, name, params, body)

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "SEMANTIC":
                self.eat("SEMANTIC")
            params.append(VariableNode(vtype, name))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        statements = []
        self.eat("LBRACE")
        while self.current_token[0] != "RBRACE":
            statements.append(self.parse_statement())
        self.eat("RBRACE")
        return statements

    def parse_statement(self):
        if self.current_token[0] in [
            "FLOAT",
            "FVECTOR",
            "INT",
            "UINT",
            "BOOL",
            "IDENTIFIER",
        ]:
            return self.parse_variable_declaration_or_assignment()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        else:
            return self.parse_expression_statement()

    def parse_variable_declaration_or_assignment(self):
        if self.current_token[0] in [
            "FLOAT",
            "FVECTOR",
            "INT",
            "UINT",
            "BOOL",
            "IDENTIFIER",
        ]:
            # This could be a type name or a variable name
            first_token = self.current_token
            self.eat(self.current_token[0])

            if self.current_token[0] == "IDENTIFIER":
                # This is a variable declaration
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "SEMICOLON":
                    # Variable declaration without initialization
                    self.eat("SEMICOLON")
                    return VariableNode(first_token[1], name)
                elif self.current_token[0] == "EQUALS":
                    # Variable declaration with initialization
                    self.eat("EQUALS")
                    value = self.parse_expression()
                    self.eat("SEMICOLON")
                    return AssignmentNode(VariableNode(first_token[1], name), value)
            else:
                # This is an assignment or a more complex expression
                left = self.parse_member_access(first_token[1])
                if self.current_token[0] == "EQUALS":
                    self.eat("EQUALS")
                    right = self.parse_expression()
                    self.eat("SEMICOLON")
                    return AssignmentNode(left, right)
                else:
                    self.eat("SEMICOLON")
                    return left
        else:
            # This is an expression statement
            expr = self.parse_expression()
            self.eat("SEMICOLON")
            return expr

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_block()
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block()
        return IfNode(condition, if_body, else_body)

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")

        # Parse initialization
        if self.current_token[0] in ["INT", "FLOAT", "FVECTOR"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            init_value = self.parse_expression()
            init = VariableNode(type_name, var_name, init_value)
        else:
            init = self.parse_expression()
        self.eat("SEMICOLON")

        # Parse condition
        condition = self.parse_expression()
        self.eat("SEMICOLON")

        # Parse update
        update = self.parse_expression()
        self.eat("RPAREN")

        # Parse body
        body = self.parse_block()

        return ForNode(init, condition, update, body)

    def parse_return_statement(self):
        self.eat("RETURN")
        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_expression(self):
        left = self.parse_logical_or()
        while self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_logical_or()
            left = AssignmentNode(left, right, op)
        return left

    def parse_assignment(self):
        left = self.parse_logical_or()
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            right = self.parse_assignment()
            return AssignmentNode(left, right)
        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()
        while self.current_token[0] == "OR":
            op = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_logical_and(self):
        left = self.parse_equality()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_equality()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_relational(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_primary()

    def parse_primary(self):
        if self.current_token[0] in ["IDENTIFIER", "INT", "FLOAT", "FVECTOR"]:
            if self.current_token[0] in ["INT", "FLOAT", "FVECTOR"]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
                if self.current_token[0] == "IDENTIFIER":
                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    return VariableNode(type_name, var_name)
                elif self.current_token[0] == "LPAREN":
                    # Handle vector constructor
                    return self.parse_vector_constructor(type_name)
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_vector_constructor(self, type_name):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return VectorConstructorNode(type_name, args)

    def parse_function_call_or_identifier(self):
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if self.current_token[0] == "LPAREN":
            return self.parse_function_call(name)
        elif self.current_token[0] == "DOT":
            return self.parse_member_access(name)
        return VariableNode("", name)

    def parse_function_call(self, name):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return FunctionCallNode(name, args)

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        # Check if there's another dot after this member access
        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)
