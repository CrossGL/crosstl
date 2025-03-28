from .OpenglAst import *
from .OpenglLexer import *


class GLSLParser:
    def __init__(self, tokens, shader_type="vertex"):
        self.tokens = tokens
        self.shader_type = shader_type
        self.pos = 0
        self.main_count = 0
        self.current_token = self.tokens[self.pos]
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):
        if self.current_token[1] == "main" and self.shader_type == "vertex":
            self.shader_type = "fragment"
            # If we encounter the second 'main', switch shader_type to 'fragment'
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        io_variables = []
        constant = []
        uniforms = []
        global_variables = []
        functions = []
        structs = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "VERSION":
                self.parse_version_directive()
            elif self.current_token[0] == "LAYOUT":
                io_variables.append(self.parse_layout())
            elif self.current_token[0] == "IN" or self.current_token[0] == "OUT":
                io_variables.append(self.parse_in_out())
            elif self.current_token[0] == "CONSTANT":
                constant.append(self.parse_constant())
            elif self.current_token[0] == "UNIFORM":
                uniforms.append(self.parse_uniform())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] in [
                "VOID",
                "FLOAT",
                "INT",
                "DOUBLE",
                "VECTOR",
                "MATRIX",
                "IDENTIFIER",
            ]:
                if self.is_function():
                    functions.append(self.parse_function())
                else:
                    global_variables.append(self.parse_global_variable())
            else:
                self.eat(self.current_token[0])  # Skip token

        return ShaderNode(
            io_variables,
            constant,
            uniforms,
            global_variables,
            functions,
            self.shader_type,
            structs,
        )

    def is_function(self):
        current_pos = self.pos
        while self.tokens[current_pos][0] != "EOF":
            if self.tokens[current_pos][0] == "LPAREN":
                return True
            if self.tokens[current_pos][0] == "SEMICOLON":
                return False
            current_pos += 1

    def parse_constant(self):
        self.eat("CONSTANT")
        vtype = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("EQUALS")
        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ConstantNode(vtype, name, value)

    def parse_version_directive(self):
        if self.current_token[0] == "VERSION":
            self.eat("VERSION")

            if self.current_token[0] == "NUMBER":
                self.current_token[1]
                self.eat("NUMBER")

                # Handle any additional tokens after NUMBER (like 'core')
                if self.current_token[0] == "CORE":
                    self.current_token[1]
                    self.eat("CORE")
            else:
                raise SyntaxError(
                    f"Expected NUMBER after VERSION, got {self.current_token[0]}"
                )
        else:
            raise SyntaxError(
                f"Expected VERSION directive, got {self.current_token[0]}"
            )

    def parse_global_variable(self):
        type_name = ""
        if self.current_token[0] in ["FLOAT", "INT", "DOUBLE", "MATRIX", "VECTOR"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return VariableNode(type_name, name)
            elif self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_expression()
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode(type_name, name), value)
            else:
                raise SyntaxError(
                    f"Unexpected token in global variable declaration: {self.current_token[0]}"
                )
        else:
            raise SyntaxError(
                f"Expected IDENTIFIER after type name, got {self.current_token[0]}"
            )

    def parse_in_out(self):
        if self.current_token[0] == "IN":
            io_type = "IN"
        else:
            io_type = "OUT"
        self.eat(io_type)
        dtype = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return VariableNode(dtype, name, f"{self.shader_type}_{io_type}")

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")

        location_number = None
        dtype = None
        name = None
        io_type = None

        while self.current_token[0] != "SEMICOLON":
            if self.current_token[0] == "IDENTIFIER":
                self.eat("IDENTIFIER")
                self.eat("EQUALS")
                location_number = self.current_token[1]
                self.eat("NUMBER")
                self.eat("RPAREN")
            elif self.current_token[0] == "IN" or self.current_token[0] == "OUT":
                io_type = self.current_token[0]
                self.eat(self.current_token[0])
                dtype = self.current_token[1]
                self.eat(self.current_token[0])
                name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SyntaxError(
                    f"Unexpected token in layout: {self.current_token[0]}"
                )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("SEMICOLON")
        return LayoutNode(
            location_number,
            dtype,
            name,
            f"{self.shader_type}_{io_type}",
            f"layout(location = {location_number})",
        )

    def parse_uniform(self):
        self.eat("UNIFORM")
        dtype = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return VariableNode(dtype, name)

    def parse_function(self):
        return_type = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        qualifier = None
        if name == "main":
            qualifier = self.shader_type
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = self.parse_body()
        self.eat("RBRACE")
        return FunctionNode(return_type, name, params, body, qualifier)

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
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        if_body = self.parse_body()
        self.eat("RBRACE")
        else_if_condition = []
        else_if_body = []
        else_body = None

        while self.current_token[0] == "ELSE" and self.peak(1)[0] == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition.append(self.parse_expression())
            self.eat("RPAREN")
            self.eat("LBRACE")
            else_if_body.append(self.parse_body())
            self.eat("RBRACE")

        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("LBRACE")
            else_body = self.parse_body()
            self.eat("RBRACE")
        return IfNode(if_condition, if_body, else_if_condition, else_if_body, else_body)

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
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "ASSIGN_SHIFT_LEFT" "ASSIGN_SHIFT_RIGHT",
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
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "ASSIGN_SHIFT_LEFT" "ASSIGN_SHIFT_RIGHT",
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
            "BITWISE_XOR",
            "EQUAL",
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "ASSIGN_SHIFT_RIGHT",
            "ASSIGN_SHIFT_LEFT",
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
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "ASSIGN_SHIFT_RIGHT",
            "ASSIGN_SHIFT_LEFT",
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
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
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
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
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
            "ASSIGN_AND",
            "ASSIGN_OR",
            "ASSIGN_XOR",
            "ASSIGN_MOD",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "ASSIGN_SHIFT_RIGHT",
            "ASSIGN_SHIFT_LEFT",
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

    def parse_struct(self):
        """Parse a struct declaration in GLSL.

        Returns:
            StructNode: A node representing the struct declaration
        """
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        fields = []
        while self.current_token[0] != "RBRACE":
            field_type = self.current_token[1]
            self.eat(self.current_token[0])
            field_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            fields.append({"type": field_type, "name": field_name})

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, fields)
