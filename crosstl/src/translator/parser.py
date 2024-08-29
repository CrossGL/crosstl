# compiler/parser.py

from .ast import (
    ShaderNode,
    FunctionNode,
    AssignmentNode,
    IfNode,
    ForNode,
    ReturnNode,
    FunctionCallNode,
    BinaryOpNode,
    MemberAccessNode,
    VariableNode,
    UniformNode,
    UnaryOpNode,
    TernaryOpNode,
    VERTEXShaderNode,
    FRAGMENTShaderNode,
)

from .lexer import Lexer


class Parser:
    """A simple parser for the shader language

    This parser generates an abstract syntax tree (AST) from a list of tokens.

    Attributes:
        tokens (list): A list of tokens generated from the input code

    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def skip_comments(self):
        """Skip comments in the token list

        This method skips comments in the token list by incrementing the position
        until the current token is not a comment token.

        """

        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        """Consume the current token if it matches the expected token type

        This method consumes the current token if it matches the expected token type.
        If the current token does not match the expected token type, a SyntaxError is raised.

        Args:
            token_type (str): The expected token type

        Raises:
            SyntaxError: If the current token does not match the expected token type

        """
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()  # Skip comments after eating a token
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse_uniforms(self):
        """Parse uniform declarations

        This method parses uniform declarations in the shader code.
        It consumes the "UNIFORM" token and then parses the uniform type and name.
        The method returns a list of UniformNode objects.

        Returns:
            list: A list of UniformNode objects

        Raises:
            SyntaxError: If the current token is not "UNIFORM"

        """

        uniforms = []
        while self.current_token[0] == "UNIFORM":
            self.eat("UNIFORM")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "SAMPLER2D",
                "MATRIX",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, DOUBLE, UINT, INT or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            uniforms.append(UniformNode(vtype, name))
        return uniforms

    def parse(self):
        """Parse the shader code

        This method parses the shader code and generates an abstract syntax tree (AST).

        Returns:
            ShaderNode: The root node of the AST

        """
        return self.parse_shader()

    def parse_shader(self):
        """Parse the shader code

        This method parses the shader code and generates a ShaderNode object.

        Returns:
            ShaderNode: The root node of the AST

        Raises:
            SyntaxError: If the current token is not "SHADER"

        """
        self.eat("SHADER")
        self.skip_comments()  # Skip comments after eating SHADER
        if self.current_token[0] in ("IDENTIFIER", "MAIN"):
            shader_name = self.current_token[1]
            self.eat(self.current_token[0])
            self.skip_comments()
        else:
            raise SyntaxError(
                f"Expected IDENTIFIER or MAIN, got {self.current_token[0]}"
            )
        self.eat("LBRACE")

        global_inputs = self.parse_inputs()
        self.parse_uniforms()
        global_outputs = self.parse_outputs()

        global_functions = []

        vertex_section = None
        fragment_section = None

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "VERTEX":
                vertex_section = self.parse_shader_section("VERTEX")
                self.skip_comments()  # Skip comments while parsing functions
            elif self.current_token[0] == "FRAGMENT":
                fragment_section = self.parse_shader_section("FRAGMENT")
                self.skip_comments()  # Skip comments while parsing functions
            elif self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "VOID",
                "MATRIX",
            ]:
                global_functions.append(self.parse_function())
                self.skip_comments()  # Skip comments while parsing functions
            else:
                raise SyntaxError(f"Unexpected token: {self.current_token[0]}")

        self.eat("RBRACE")
        return ShaderNode(
            shader_name,
            global_inputs,
            global_outputs,
            global_functions,
            vertex_section,
            fragment_section,
        )

    def parse_shader_section(self, section_type):
        """Parse a shader section

        This method parses a shader section (VERTEX or FRAGMENT) and generates a VERTEXShaderNode or FRAGMENTShaderNode object.

        Args:
            section_type (str): The type of shader section (VERTEX or FRAGMENT)

        Returns:
            VERTEXShaderNode or FRAGMENTShaderNode: The root node of the AST for the shader section

        Raises:
            SyntaxError: If the current token is not "VERTEX" or "FRAGMENT"

        """
        self.eat(section_type)
        self.eat("LBRACE")
        functions = []
        intermidiate = []
        inputs = []
        outputs = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "UNIFORM":
                self.parse_uniforms()
            elif self.current_token[0] == "INPUT":
                inputs.extend(self.parse_inputs())
            elif self.current_token[0] == "OUTPUT":
                outputs.extend(self.parse_outputs())
            elif (
                self.current_token[0]
                in ["VECTOR", "FLOAT", "DOUBLE", "UINT", "INT", "VOID", "MATRIX"]
                and self.peak(2)[0] == "LPAREN"
            ):
                functions.append(self.parse_function())
            elif self.current_token[0] == "FOR":
                intermidiate.append(self.parse_for_loop())
            elif self.current_token[0] == "IF":
                intermidiate.append(self.parse_if_statement())
            elif self.current_token[0] in [
                "VECTOR",
                "IDENTIFIER",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "MATRIX",
            ]:
                intermidiate.append(self.parse_assignment_or_function_call())
            else:
                raise SyntaxError(f"Unexpected token {self.current_token[0]}")
        self.eat("RBRACE")
        if section_type == "VERTEX":
            return VERTEXShaderNode(inputs, outputs, functions, intermidiate)
        else:
            return FRAGMENTShaderNode(inputs, outputs, functions, intermidiate)

    def parse_inputs(self):
        """Parse input declarations

        This method parses input declarations in the shader code.

        Returns:

            list: A list of input declarations


        Raises:

            SyntaxError: If the current token is not "INPUT"


        """
        inputs = []
        while self.current_token[0] == "INPUT":
            self.eat("INPUT")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "MATRIX",
                "SAMPLER2D",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, DOUBLE, UINT, INT, MATRIX, or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            inputs.append((vtype, name))
        return inputs

    def parse_outputs(self):
        """Parse output declarations

        This method parses output declarations in the shader code.

        Returns:

            list: A list of output declarations


        Raises:

            SyntaxError: If the current token is not "OUTPUT"

        """
        outputs = []
        while self.current_token[0] == "OUTPUT":
            self.eat("OUTPUT")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "MATRIX",
                "SAMPLER2D",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, DOUBLE, UINT, INT, MATRIX, or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            outputs.append((vtype, name))
        return outputs

    def parse_function(self):
        """Parse a function declaration

        This method parses a function declaration in the shader code.

        Returns:

                FunctionNode: A FunctionNode object representing the function declaration

        Raises:

            SyntaxError: If the current token is not a valid function declaration

        """
        return_type = self.parse_type()
        if self.current_token[0] == "MAIN":
            fname = self.current_token[1]
            self.eat("MAIN")
        elif self.current_token[0] == "IDENTIFIER":
            fname = self.current_token[1]
            self.eat("IDENTIFIER")
        else:
            raise SyntaxError(
                f"Expected MAIN or IDENTIFIER, got {self.current_token[0]}"
            )
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = self.parse_body()
        self.eat("RBRACE")
        return FunctionNode(return_type, fname, params, body)

    def parse_parameters(self):
        """Parse function parameters

        This method parses function parameters in the shader code.

        Returns:

            list: A list of function parameters

        """

        params = []
        if self.current_token[0] != "RPAREN":
            params.append(self.parse_parameter())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.parse_parameter())
        return params

    def parse_parameter(self):
        """Parse a function parameter

        This method parses a function parameter in the shader code.

        Returns:

            tuple: A tuple containing the parameter type and name

        """
        param_type = self.parse_type()
        param_name = self.current_token[1]

        self.eat("IDENTIFIER")
        return (param_type, param_name)

    def parse_type(self):
        """Parse a type declaration

        This method parses a type declaration in the shader code.

        Returns:

            str: The type name

        Raises:

            SyntaxError: If the current token is not a valid type declaration

        """
        if self.current_token[0] == "VOID":
            self.eat("VOID")
            return "void"
        elif self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
            "SAMPLER2D",
        ]:
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            return vtype
        elif self.current_token[0] == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if type_name in ["int", "uint", "float", "double"]:
                return type_name
            return type_name
        else:
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")

    def parse_body(self):
        """Parse a function body

        This method parses a function body in the shader code.

        Returns:

            list: A list of statements in the function body

        """
        body = []
        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "IF":
                body.append(self.parse_if_statement())
            elif self.current_token[0] == "FOR":
                body.append(self.parse_for_loop())
            elif self.current_token[0] == "RETURN":
                body.append(self.parse_return_statement())
            elif self.current_token[0] in [
                "VECTOR",
                "IDENTIFIER",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
            ]:
                body.append(self.parse_assignment_or_function_call())
            else:
                raise SyntaxError(f"Unexpected token {self.current_token[0]}")
        return body

    def parse_if_statement(self):
        """Parse an if statement

        This method parses an if statement in the shader code.

        Returns:

            IfNode: An IfNode object representing the if statement

        """
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        if_body = self.parse_body()
        self.eat("RBRACE")
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("LBRACE")
            else_body = self.parse_body()
            self.eat("RBRACE")
        return IfNode(condition, if_body, else_body)

    def peak(self, n):
        """Peek ahead in the token list

        This method returns the nth token ahead in the token list.

        Args:

            n (int): The number of tokens to peek ahead

        Returns:

                tuple: The nth token ahead in the token list

        """

        return self.tokens[self.pos + n]

    def parse_for_loop(self):
        """Parse a for loop

        This method parses a for loop in the shader code.

        Returns:

            ForNode: A ForNode object representing the for loop

        """

        self.eat("FOR")
        self.eat("LPAREN")

        init = self.parse_assignment_or_function_call()

        condition = self.parse_assignment_or_function_call()

        if self.peak(2)[0] == "RPAREN":
            update = self.parse_update()
        else:
            update = self.parse_assignment_or_function_call(update_condition=True)

        self.eat("RPAREN")
        self.eat("LBRACE")

        body = self.parse_body()
        self.eat("RBRACE")

        return ForNode(init, condition, update, body)

    def parse_update(self):
        """Parse an update statement

        This method parses an update statement in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the update statement

        """
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "INCREMENT":
                op_name = self.current_token[1]
                self.eat("INCREMENT")
                return VariableNode("", name + op_name)
            elif self.current_token[0] == "DECREMENT":
                op_name = self.current_token[1]
                self.eat("DECREMENT")
                return VariableNode("", name + op_name)
            else:
                raise SyntaxError(
                    f"Expected INCREMENT or DECREMENT, got {self.current_token[0]}"
                )
        elif self.current_token[0] in ["INCREMENT", "DECREMENT"]:
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                return VariableNode("", op_name + name)
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_return_statement(self):
        """Parse a return statement

        This method parses a return statement in the shader code.

        Returns:

            ReturnNode: A ReturnNode object representing the return statement

        """
        self.eat("RETURN")
        return_value = []
        return_value.append(self.parse_expression())
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            return_value.append(self.parse_expression())
        self.eat("SEMICOLON")
        return ReturnNode(return_value)

    def parse_assignment_or_function_call(self, update_condition=False):
        """Parse an assignment or function call

        This method parses an assignment or function call in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the assignment or function call

        """
        type_name = ""
        inc_dec = False
        if self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
        ]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable_declaration(type_name, update_condition)
        if self.current_token[0] in ["INCREMENT", "DECREMENT"]:
            inc_dec = True
            inc_dec_op = self.current_token[1]
            self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if inc_dec:
            name = VariableNode(type_name, VariableNode("", inc_dec_op + name))
        if self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
        ]:
            return self.parse_assignment(name)
        elif self.current_token[0] == "INCREMENT":
            self.eat("INCREMENT")
            op_name = self.current_token[1]
            return VariableNode(type_name, VariableNode("", name + op_name))
        elif self.current_token[0] == "DECREMENT":
            self.eat("DECREMENT")
            op_name = self.current_token[1]
            return VariableNode(type_name, VariableNode("", name + op_name))
        elif self.current_token[0] == "LPAREN":
            return self.parse_function_call(name)
        else:
            raise SyntaxError(
                f"Unexpected token after identifier: {self.current_token[0]}"
            )

    def parse_variable_declaration(self, type_name, update_condition=False):
        """Parse a variable declaration

        This method parses a variable declaration in the shader code.

        Args:

            type_name (str): The type of the variable
            update_condition (bool): A flag indicating if the variable is from for loop update statement
        Returns:

            VariableNode: A VariableNode object representing the variable declaration

        Raises:

            SyntaxError: If the current token is not a valid variable declaration

        """
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if self.current_token[0] == "DOT":
            name = self.parse_member_access(name)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

        elif self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            if self.current_token[0] == "DOT":
                value = self.parse_member_access(value)
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op, value)

            else:
                if update_condition:
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                else:
                    raise SyntaxError(
                        f"Expected ';' after variable assignment, found: {self.current_token[0]}"
                    )

        elif self.current_token[0] in (
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "EQUALS",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "EQUAL",
        ):
            op = self.current_token[0]
            self.eat(op)
            value = self.parse_expression()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op, value)
            else:
                raise SyntaxError(
                    f"Expected ';' after compound assignment, found: {self.current_token[0]}"
                )

        else:
            raise SyntaxError(
                f"Unexpected token in variable declaration: {self.current_token[0]}"
            )

    def parse_assignment(self, name):
        """Parse an assignment statement

        This method parses an assignment statement in the shader code.

        Args:

            name (str): The name of the variable being assigned

        Returns:

            AssignmentNode: An AssignmentNode object representing the assignment statement

        Raises:

            SyntaxError: If the current token is not a valid assignment statement

        """

        if self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
        ]:
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return BinaryOpNode(name, op_name, value)
        else:
            raise SyntaxError(
                f"Expected assignment operator, found: {self.current_token[0]}"
            )

    def parse_additive(self):
        """Parse an additive expression

        This method parses an additive expression in the shader code.

        Returns:

                ASTNode: An ASTNode object representing the additive expression

        """
        expr = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_multiplicative()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_multiplicative(self):
        """Parse a multiplicative expression

        This method parses a multiplicative expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the multiplicative expression

        """
        expr = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_unary()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_unary(self):
        """Parse a unary expression

        This method parses a unary expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the unary expression

        """
        if self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[0]
            self.eat(op)
            expr = self.parse_unary()
            return UnaryOpNode(op, expr)
        return self.parse_primary()

    def parse_primary(self):
        """Parse a primary expression

        This method parses a primary expression in the shader code.

        Returns:


            ASTNode: An ASTNode object representing the primary expression

        Raises:

            SyntaxError: If the current token is not a valid primary expression

        """
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] in ["NUMBER", "FLOAT_NUMBER"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        elif self.current_token[0] in [
            "IDENTIFIER",
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
        ]:
            return self.parse_function_call_or_identifier()
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_function_call(self, name):
        """Parse a function call

        This method parses a function call in the shader code.

        Args:

            name (str): The name of the function being called

        Returns:

            FunctionCallNode: A FunctionCallNode object representing the function call

        """
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return FunctionCallNode(name, args)

    def parse_expression(self):
        """Parse an expression

        This method parses an expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the expression

        """
        expr = self.parse_ternary()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "LOGICAL_AND",
            "LOGICAL_OR",
            "PLUS",
            "MINUS",
            "MULTIPLY",
            "DIVIDE",
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_ternary()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_ternary(self):
        """Parse a ternary expression

        This method parses a ternary expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the ternary expression

        """
        expr = self.parse_additive()
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            expr = TernaryOpNode(expr, true_expr, false_expr)
        return expr

    def parse_function_call_or_identifier(self):
        """Parse a function call or identifier

        This method parses a function call or identifier in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the function call or identifier

        """
        if self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
        ]:
            func_name = self.current_token[1]
            self.eat(self.current_token[0])
        else:
            func_name = self.current_token[1]
            self.eat("IDENTIFIER")

        if self.current_token[0] == "LPAREN":
            return self.parse_function_call(func_name)
        elif self.current_token[0] == "DOT":
            return self.parse_member_access(func_name)
        return VariableNode("", func_name)

    def parse_member_access(self, object):
        """Parse a member access

        This method parses a member access in the shader code.

        Args:

                object (str): The object being accessed

        Returns:

                MemberAccessNode: A MemberAccessNode object representing the member access

        Raises:

            SyntaxError: If the current token is not a valid member access

        """
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
