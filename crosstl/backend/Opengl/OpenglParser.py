from .OpenglAst import (
    ShaderNode,
    VariableNode,
    AssignmentNode,
    FunctionNode,
    ArrayAccessNode,
    BinaryOpNode,
    UnaryOpNode,
    ReturnNode,
    FunctionCallNode,
    IfNode,
    ForNode,
    VectorConstructorNode,
    LayoutNode,
    ConstantNode,
    MemberAccessNode,
    TernaryOpNode,
    StructNode,
    SwitchNode,
    CaseNode,
    BlockNode,
    NumberNode,
)
from .OpenglLexer import GLSLLexer


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
        """Match the current token type with the expected token type

        Args:

            token_type: The token type to match with the current token type

        Raises:

            SyntaxError: If the current token type does not match the expected

                token type

        """
        if self.current_token[0] == token_type:
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", "")
        else:
            raise SyntaxError(
                f"Syntax error: expected {token_type}, got {self.current_token[0]}"
            )

    def skip_until(self, token_types):
        """Skip tokens until one of the specified token types is found

        This method is useful for error recovery, allowing the parser to
        skip ahead to a known synchronization point.

        Args:
            token_types (list): A list of token types to stop at

        Returns:
            bool: True if a token in token_types was found, False if EOF was reached
        """
        while self.current_token[0] != "EOF":
            if self.current_token[0] in token_types:
                return True
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", "")
                return False
        return False

    def parse(self):
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        io_variables = []
        constant = []
        uniforms = []
        functions = []
        global_variables = []
        structs = []

        while self.current_token[0] != "EOF":
            try:
                if self.current_token[0] == "UNIFORM":
                    uniforms.append(self.parse_global_variable())
                elif self.current_token[0] in ["ATTRIBUTE", "VARYING", "IN", "OUT"]:
                    io_variables.append(self.parse_global_variable())
                elif self.current_token[0] == "CONST":
                    constant.append(self.parse_global_variable())
                elif self.current_token[0] == "STRUCT":
                    structs.append(self.parse_struct())
                elif self.current_token[0] in [
                    "FLOAT",
                    "INT",
                    "DOUBLE",
                    "VOID",
                    "VECTOR",
                    "MATRIX",
                    "SAMPLER2D",
                    "IDENTIFIER",
                ]:
                    # Check if this is a function or a global variable
                    if (
                        self.pos + 2 < len(self.tokens)
                        and self.tokens[self.pos + 1][0] == "IDENTIFIER"
                        and self.tokens[self.pos + 2][0] == "LPAREN"
                    ):
                        functions.append(self.parse_function())
                    else:
                        global_variables.append(self.parse_global_variable())
                else:
                    # Skip unrecognized tokens
                    self.pos += 1
                    if self.pos < len(self.tokens):
                        self.current_token = self.tokens[self.pos]
                    else:
                        self.current_token = ("EOF", "")
            except SyntaxError as e:
                # Log the error but continue parsing
                print(
                    f"Warning: {e} at position {self.pos}, token {self.current_token}"
                )
                # Skip to the next semicolon to try to recover
                while (
                    self.current_token[0] != "SEMICOLON"
                    and self.current_token[0] != "EOF"
                ):
                    self.pos += 1
                    if self.pos < len(self.tokens):
                        self.current_token = self.tokens[self.pos]
                    else:
                        self.current_token = ("EOF", "")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")  # Consume the semicolon and continue

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
        """Parse a global variable declaration.
        This includes uniform, attribute, varying, const, and other global variables.

        Returns:
            ASTNode: A node representing the global variable
        """
        type_name = ""
        qualifier = None
        vector_type = None

        # Check if this is a qualified variable (attribute, varying, etc.)
        if self.current_token[0] in ["ATTRIBUTE", "VARYING", "UNIFORM", "IN", "OUT"]:
            qualifier = self.current_token[
                0
            ].lower()  # Store as lowercase for consistency
            self.eat(self.current_token[0])

        # Handle type specification
        if self.current_token[0] in [
            "FLOAT",
            "INT",
            "DOUBLE",
            "MATRIX",
            "VECTOR",
            "SAMPLER2D",
        ]:
            type_name = self.current_token[1]
            vector_type = self.current_token[0]  # Store the token type for vectors
            self.eat(self.current_token[0])

        # Process variables (possibly multiple in a comma-separated list)
        variables = []

        # Check if we have a valid identifier
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Handle array declaration
            array_size = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] == "NUMBER":
                    array_size = int(self.current_token[1])
                    self.eat("NUMBER")
                self.eat("RBRACKET")

            # Create variable node
            var_node = VariableNode(type_name, name, qualifier, array_size=array_size)

            # Handle initialization
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                value = self.parse_expression()
                variables.append(AssignmentNode(var_node, value))
            else:
                variables.append(var_node)

            # Continue if there are more variables
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                # Process more variables in this declaration later
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            else:
                # Try to recover if we don't see expected tokens
                print(
                    f"Warning: Expected comma or semicolon after variable declaration, got {self.current_token[0]}"
                )

        # Return the first variable that was successfully parsed
        if variables:
            return variables[0]

        # If we couldn't parse a variable, skip to the next semicolon
        while self.current_token[0] != "SEMICOLON" and self.current_token[0] != "EOF":
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", "")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        # Create a placeholder variable as fallback
        return VariableNode("", "placeholder")

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
        """Parse a uniform declaration

        This method parses a uniform declaration in the shader code.

        Returns:

            VariableNode: A VariableNode object representing the uniform declaration

        """
        self.eat("UNIFORM")
        dtype = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Check if this is an array declaration
        array_size = None
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            array_size = int(self.current_token[1])
            self.eat("NUMBER")
            self.eat("RBRACKET")

        self.eat("SEMICOLON")
        return VariableNode(dtype, name, array_size=array_size)

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
            elif self.current_token[0] == "SWITCH":
                body.append(self.parse_switch_statement())
            elif self.current_token[0] == "RETURN":
                body.append(self.parse_return_statement())
            elif self.current_token[0] == "BREAK":
                # Handle break statements
                self.eat("BREAK")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                # No need to add anything to body as break is handled implicitly
            elif self.current_token[0] in [
                "VECTOR",
                "IDENTIFIER",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
            ]:
                body.append(self.parse_assignment_or_function_call())
            elif self.current_token[0] in [
                "PLUS",
                "MINUS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
            ]:
                # Handle standalone operators like += in the body
                left = VariableNode("", "")  # Create an empty variable node
                op = self.current_token[1]
                self.eat(self.current_token[0])
                right = self.parse_expression()
                body.append(BinaryOpNode(left, op, right))
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
            elif self.current_token[0] == "QUESTION":
                # Handle ternary operator expression statements
                condition = body.pop()  # Pop the last expression as condition
                self.eat("QUESTION")
                true_expr = self.parse_expression()
                self.eat("COLON")
                false_expr = self.parse_expression()
                body.append(TernaryOpNode(condition, true_expr, false_expr))
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
            elif self.current_token[0] == "LBRACKET":
                # Handle array access within the body
                index_expr = None
                self.eat("LBRACKET")
                if self.current_token[0] != "RBRACKET":
                    index_expr = self.parse_expression()
                self.eat("RBRACKET")

                # Create an array access node with the last element as the array
                if body:
                    array_node = body.pop()
                    array_access = ArrayAccessNode(array_node, index_expr)
                    body.append(array_access)
                else:
                    # If there's no previous element, create an empty variable node
                    body.append(ArrayAccessNode(VariableNode("", ""), index_expr))

                # Handle possible assignment
                if self.current_token[0] == "EQUALS":
                    self.eat("EQUALS")
                    value = self.parse_expression()
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")
                    array_node = body.pop()
                    body.append(BinaryOpNode(array_node, "=", value))
            elif self.current_token[0] == "SEMICOLON":
                # Skip standalone semicolons
                self.eat("SEMICOLON")
            else:
                # Try to recover from unexpected tokens
                print(f"Warning: Unexpected token in body: {self.current_token[0]}")
                self.pos += 1
                if self.pos < len(self.tokens):
                    self.current_token = self.tokens[self.pos]
                else:
                    self.current_token = ("EOF", "")
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

        # Make sure we're at the closing parenthesis
        if self.current_token[0] != "RPAREN":
            # Try to handle complex conditions like (value & mask) for bitwise ops
            while self.current_token[0] != "RPAREN" and self.current_token[0] != "EOF":
                # If we see a bitwise operator, handle it
                if self.current_token[0] in [
                    "BITWISE_AND",
                    "BITWISE_OR",
                    "BITWISE_XOR",
                ]:
                    op = self.current_token[1]
                    self.eat(self.current_token[0])
                    right = self.parse_expression()
                    if_condition = BinaryOpNode(if_condition, op, right)
                else:
                    # Skip unknown tokens to recover
                    self.pos += 1
                    if self.pos < len(self.tokens):
                        self.current_token = self.tokens[self.pos]
                    else:
                        self.current_token = ("EOF", "")

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

        # Parse initialization (e.g. int i = 0)
        init_type = ""
        if self.current_token[0] in ["INT", "FLOAT", "DOUBLE", "VECTOR", "MATRIX"]:
            init_type = self.current_token[1]
            self.eat(self.current_token[0])

        init = self.parse_variable_declaration(init_type)

        # Parse condition (e.g. i < 10)
        condition = self.parse_expression()
        self.eat("SEMICOLON")

        # Parse update (e.g. i++)
        # Extract identifier for handling i++, i+=1, etc.
        if self.current_token[0] == "IDENTIFIER":
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "PLUS" and self.peak(1)[0] == "PLUS":
                # Handle i++
                self.eat("PLUS")
                self.eat("PLUS")
                update = BinaryOpNode(VariableNode("", var_name), "+=", 1)
            elif self.current_token[0] == "MINUS" and self.peak(1)[0] == "MINUS":
                # Handle i--
                self.eat("MINUS")
                self.eat("MINUS")
                update = BinaryOpNode(VariableNode("", var_name), "-=", 1)
            elif self.current_token[0] == "PLUS_EQUALS":
                # Handle i+=1
                self.eat("PLUS_EQUALS")
                value = self.parse_expression()
                update = BinaryOpNode(VariableNode("", var_name), "+=", value)
            elif self.current_token[0] == "MINUS_EQUALS":
                # Handle i-=1
                self.eat("MINUS_EQUALS")
                value = self.parse_expression()
                update = BinaryOpNode(VariableNode("", var_name), "-=", value)
            elif self.current_token[0] == "EQUALS":
                # Handle i=i+1
                self.eat("EQUALS")
                value = self.parse_expression()
                update = BinaryOpNode(VariableNode("", var_name), "=", value)
            else:
                # Reset position and try the general approach
                self.pos -= 1  # Go back to the identifier
                self.current_token = self.tokens[self.pos]
                update = self.parse_assignment_or_function_call(update_condition=True)
        else:
            update = self.parse_assignment_or_function_call(update_condition=True)

        self.eat("RPAREN")
        self.eat("LBRACE")
        body = self.parse_body()
        self.eat("RBRACE")
        return ForNode(init, condition, update, body)

    def parse_switch_statement(self):
        """Parse a switch statement

        This method parses a switch statement in the shader code.

        Returns:
            SwitchNode: A SwitchNode object representing the switch statement
        """
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")

        cases = []
        default = None

        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "CASE":
                cases.append(self.parse_case_statement())
            elif self.current_token[0] == "DEFAULT":
                self.eat("DEFAULT")
                self.eat("COLON")
                default_statements = []

                # Parse statements until we hit a case, default, or end of switch
                while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
                    if self.current_token[0] == "IF":
                        default_statements.append(self.parse_if_statement())
                    elif self.current_token[0] == "FOR":
                        default_statements.append(self.parse_for_loop())
                    elif self.current_token[0] == "RETURN":
                        default_statements.append(self.parse_return_statement())
                    elif self.current_token[0] == "BREAK":
                        # Handle break statement in default case
                        self.eat("BREAK")
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                        # Break is implicit, no need to add to statements
                    elif self.current_token[0] in [
                        "VECTOR",
                        "IDENTIFIER",
                        "FLOAT",
                        "DOUBLE",
                        "UINT",
                        "INT",
                    ]:
                        default_statements.append(
                            self.parse_assignment_or_function_call()
                        )
                    else:
                        raise SyntaxError(
                            f"Unexpected token in default case: {self.current_token[0]}"
                        )

                default = default_statements
            else:
                raise SyntaxError(
                    f"Unexpected token in switch statement: {self.current_token[0]}"
                )

        self.eat("RBRACE")
        return SwitchNode(expression, cases, default)

    def parse_case_statement(self):
        """Parse a case statement within a switch

        This method parses a case statement in the shader code.

        Returns:
            CaseNode: A CaseNode object representing the case statement
        """
        self.eat("CASE")
        value = self.parse_expression()
        self.eat("COLON")

        statements = []

        # Parse statements until we hit a case, default, or end of switch
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            if self.current_token[0] == "IF":
                statements.append(self.parse_if_statement())
            elif self.current_token[0] == "FOR":
                statements.append(self.parse_for_loop())
            elif self.current_token[0] == "RETURN":
                statements.append(self.parse_return_statement())
            elif self.current_token[0] == "BREAK":
                self.eat("BREAK")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                # Add a break node or just continue as break is implicit in case statements
            elif self.current_token[0] in [
                "VECTOR",
                "IDENTIFIER",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
            ]:
                statements.append(self.parse_assignment_or_function_call())
            else:
                raise SyntaxError(
                    f"Unexpected token in case statement: {self.current_token[0]}"
                )

        return CaseNode(value, statements)

    def parse_return_statement(self):
        """Parse a return statement

        This method parses a return statement in the shader code.

        Returns:

            ReturnNode: A ReturnNode object representing the return statement

        """
        self.eat("RETURN")
        return_value = []

        # Empty return statement
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode(return_value)

        # Return with expression
        return_value.append(self.parse_expression())
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            return_value.append(self.parse_expression())

        # Allow optional semicolon at the end of function (for cases where we hit RBRACE)
        if self.current_token[0] == "SEMICOLON":
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
        inc_dec_op = ""  # Initialize to empty string

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
        if self.current_token[0] in (
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
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
        ):
            op = self.current_token[1]
            self.eat(self.current_token[0])

            # Check if the next expression involves array access
            if self.current_token[0] == "IDENTIFIER":
                identifier = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "LBRACKET":
                    # We have an array access
                    self.eat("LBRACKET")
                    index = self.parse_expression()
                    self.eat("RBRACKET")
                    value = ArrayAccessNode(identifier, index)
                else:
                    # Go back to just after the operator
                    self.pos -= 1
                    self.current_token = self.tokens[self.pos]
                    value = self.parse_expression()
            else:
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

        # Handle function calls - create a special case to recognize nested function calls
        # Need to handle gl_Position = vec4(position, 1.0) case
        if self.current_token[0] == "LPAREN":
            # This is a function call
            args = []
            self.eat("LPAREN")
            if self.current_token[0] != "RPAREN":
                args.append(self.parse_expression())
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    args.append(self.parse_expression())
            self.eat("RPAREN")

            # Optional semicolon at end of statement
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            return FunctionCallNode(name, args)

        # Handle array access after variable name
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            index = self.parse_expression()
            self.eat("RBRACKET")
            name = ArrayAccessNode(name, index)

        if self.current_token[0] == "DOT":
            name = self.parse_member_access(name)

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

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
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
        ):
            op = self.current_token[1]
            self.eat(self.current_token[0])

            # Check if the next expression involves array access
            if self.current_token[0] == "IDENTIFIER":
                identifier = self.current_token[1]
                self.eat("IDENTIFIER")

                if self.current_token[0] == "LBRACKET":
                    # We have an array access
                    self.eat("LBRACKET")
                    index = self.parse_expression()
                    self.eat("RBRACKET")
                    value = ArrayAccessNode(identifier, index)
                elif self.current_token[0] == "LPAREN":
                    # We have a function call
                    args = []
                    self.eat("LPAREN")
                    if self.current_token[0] != "RPAREN":
                        args.append(self.parse_expression())
                        while self.current_token[0] == "COMMA":
                            self.eat("COMMA")
                            args.append(self.parse_expression())
                    self.eat("RPAREN")
                    value = FunctionCallNode(identifier, args)
                else:
                    # Go back to just after the operator
                    self.pos -= 1
                    self.current_token = self.tokens[self.pos]
                    value = self.parse_expression()
            else:
                value = self.parse_expression()

            if self.current_token[0] == "DOT":
                value = self.parse_member_access(value)
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op, value)
            else:
                if update_condition:
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                elif self.current_token[0] in ["IF", "FOR", "WHILE", "SWITCH"]:
                    # Just return the binary operation without requiring semicolon
                    # This is needed for cases like: float value = compute() if (value > 0) {...}
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                elif self.current_token[0] == "RBRACE":
                    # Allow missing semicolon at the end of a block
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                else:
                    # This could be a problem of the current parsing approach
                    # In complex expressions like a = b = c = 5, we need to be more lenient
                    return BinaryOpNode(VariableNode(type_name, name), op, value)

        else:
            # Try more lenient parsing for function calls, member access, etc.
            return VariableNode(type_name, name)

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

    def parse_expression(self):
        """Parse an expression

        This method parses an expression in the shader code.

        Returns:

            Node: A Node object representing the expression

        Raises:

            SyntaxError: If the current token is not a valid expression

        """
        left = self.parse_logical_or()

        # Handle ternary operator: condition ? true_expr : false_expr
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            return TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_logical_or(self):
        left = self.parse_logical_and()

        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_logical_and(self):
        left = self.parse_bitwise_or()  # Changed from parse_equality

        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_bitwise_or()  # Changed from parse_equality
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_or(self):
        """Parse a bitwise OR expression"""
        left = self.parse_bitwise_xor()

        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_xor(self):
        """Parse a bitwise XOR expression"""
        left = self.parse_bitwise_and()

        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_bitwise_and(self):
        """Parse a bitwise AND expression"""
        left = self.parse_equality()

        while self.current_token[0] == "BITWISE_AND":
            op = self.current_token[1]
            self.eat("BITWISE_AND")
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
        left = self.parse_shift()  # Use parse_shift instead of parse_additive

        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift()  # Use parse_shift instead of parse_additive
            left = BinaryOpNode(left, op, right)

        return left

    def parse_shift(self):
        """Parse bitwise shift expressions (<<, >>)"""
        left = self.parse_additive()

        # Handle bit shift operations
        while True:
            if (
                self.current_token[0] == "LESS_THAN"
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "LESS_THAN"
            ):
                # Left shift <<
                self.eat("LESS_THAN")
                self.eat("LESS_THAN")
                op = "<<"
                right = self.parse_additive()
                left = BinaryOpNode(left, op, right)
            elif (
                self.current_token[0] == "GREATER_THAN"
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "GREATER_THAN"
            ):
                # Right shift >>
                self.eat("GREATER_THAN")
                self.eat("GREATER_THAN")
                op = ">>"
                right = self.parse_additive()
                left = BinaryOpNode(left, op, right)
            else:
                break

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

        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)

        return left

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_primary()

    def parse_primary(self):
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return NumberNode(value)
        elif self.current_token[0] == "IDENTIFIER" or self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
        ]:
            return self.parse_function_call_or_identifier()
        # Handle bitwise operators and other special tokens that could be used in expressions
        elif self.current_token[0] in ["BITWISE_NOT", "PLUS", "MINUS"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        elif self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "EQUALS",
            "BITWISE_XOR",
            "BITWISE_AND",
            "BITWISE_OR",
            "EQUAL",
            "NOT_EQUAL",
        ]:
            # These tokens should be handled as part of binary operations, not as primary expressions
            # Create a placeholder variable to start the binary operation
            left = VariableNode("", "")
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_expression()
            return BinaryOpNode(left, op, right)
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

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

        # Check if this is followed by a semicolon (typically for statements)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return FunctionCallNode(name, args)

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
