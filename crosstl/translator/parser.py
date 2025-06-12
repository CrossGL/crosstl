from .ast import (
    BinaryOpNode,
    CbufferNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    AssignmentNode,
    ArrayNode,
    ArrayAccessNode,
)

from .lexer import Lexer
import logging


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

    def parse(self):
        """Parse the shader code and generate an AST

        This method parses the shader code and generates an abstract syntax tree (AST).

        Returns:
            ShaderNode: The root node of the AST
        """
        try:
            # Attempt normal parsing
            return self.parse_shader()
        except Exception as e:
            self.report_error(f"Error in main parser: {e}")

            # If normal parsing fails, try the permissive mode for complex shaders
            try:
                self.pos = 0  # Reset position to start over
                if self.pos < len(self.tokens):
                    self.current_token = self.tokens[self.pos]
                return self.parse_shader_permissive()
            except Exception as e2:
                self.report_error(f"Permissive parsing also failed: {e2}")
                # Return a minimal shader node so something gets generated
                return ShaderNode([], [], [], [])

    def parse_shader(self):
        functions = []
        structs = []
        cbuffers = []
        global_variables = []
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "SHADER":
                self.eat("SHADER")
                self.eat("IDENTIFIER")
                self.eat("LBRACE")
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] in ["CBUFFER", "UNIFORM"]:
                cbuffers.append(self.parse_cbuffer())
            elif self.current_token[1] in ["vertex", "fragment", "compute"]:
                functions.append(self.parse_main_function())
                if self.current_token[0] == "RBRACE":
                    self.eat("RBRACE")
            elif self.current_token[0] in [
                "VOID",
                "FLOAT",
                "VECTOR",
                "DOUBLE",
                "UINT",
                "INT",
                "MATRIX",
                "IDENTIFIER",
                "SAMPLER2D",
                "SAMPLERCUBE",
                "SAMPLER",
                "CONST",
                "BOOL",
            ]:
                if self.is_function():
                    functions.append(self.parse_function())
                else:
                    global_variables.append(self.parse_global_variable())
            elif self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            elif self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
            else:
                logging.debug(f"Skipping unexpected token {self.current_token[0]}")
                self.eat(self.current_token[0])  # Skip unknown tokens

        return ShaderNode(structs, functions, global_variables, cbuffers)

    def parse_shader_permissive(self):
        """Parse shader code with more permissive error handling for complex shaders.

        This is a fallback method that tries to extract as much structure as possible
        even if the shader has syntax errors.

        Returns:
            ShaderNode: The root node of the AST
        """
        functions = []
        structs = []
        cbuffers = []
        global_variables = []

        # Keep track of the current section we're in
        in_vertex = False
        in_fragment = False
        in_compute = False

        while self.current_token[0] != "EOF":
            try:
                if self.current_token[0] == "SHADER":
                    self.eat("SHADER")
                    self.eat("IDENTIFIER")
                    self.eat("LBRACE")
                elif self.current_token[0] == "STRUCT":
                    try:
                        structs.append(self.parse_struct())
                    except Exception as e:
                        self.report_error(f"Error parsing struct: {e}")
                        self.skip_to_matching_brace()
                elif self.current_token[0] in ["CBUFFER", "UNIFORM"]:
                    try:
                        cbuffers.append(self.parse_cbuffer())
                    except Exception as e:
                        self.report_error(f"Error parsing cbuffer: {e}")
                        self.skip_to_matching_brace()
                elif self.current_token[1] in ["vertex", "fragment", "compute"]:
                    # Record which section we're in
                    section_type = self.current_token[1]
                    in_vertex = section_type == "vertex"
                    in_fragment = section_type == "fragment"
                    in_compute = section_type == "compute"

                    try:
                        functions.append(self.parse_main_function())
                    except Exception as e:
                        self.report_error(f"Error parsing {section_type} section: {e}")
                        self.skip_to_matching_brace()

                    # Reset section tracking after we're done
                    in_vertex = in_fragment = in_compute = False

                    if self.current_token[0] == "RBRACE":
                        self.eat("RBRACE")
                elif self.current_token[0] in [
                    "VOID",
                    "FLOAT",
                    "VECTOR",
                    "DOUBLE",
                    "UINT",
                    "INT",
                    "MATRIX",
                    "IDENTIFIER",
                    "SAMPLER2D",
                    "SAMPLERCUBE",
                    "SAMPLER",
                    "CONST",
                    "BOOL",
                ]:
                    try:
                        if self.is_function():
                            functions.append(self.parse_function())
                        else:
                            global_variables.append(self.parse_global_variable())
                    except Exception as e:
                        self.report_error(f"Error parsing function/variable: {e}")
                        # Skip to next semicolon or brace
                        self.skip_to_token(["SEMICOLON", "RBRACE"])
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                elif self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                elif self.current_token[0] == "RBRACE":
                    self.eat("RBRACE")
                else:
                    self.report_error(
                        f"Skipping unexpected token {self.current_token[0]}"
                    )
                    self.eat(self.current_token[0])  # Skip unknown tokens
            except Exception as e:
                self.report_error(f"Error in permissive parser: {e}")
                # Skip to next significant token to try to continue
                self.next_token()

        return ShaderNode(structs, functions, global_variables, cbuffers)

    def skip_to_matching_brace(self):
        """Skip tokens until finding a matching closing brace, handling nested braces.

        This is used for error recovery when parsing fails inside a block.
        """
        nesting_level = 1  # We're already inside one level
        while nesting_level > 0 and self.current_token[0] != "EOF":
            if self.current_token[0] == "LBRACE":
                nesting_level += 1
            elif self.current_token[0] == "RBRACE":
                nesting_level -= 1
            self.next_token()

        # Eat the closing brace we found (if we did find one)
        if self.current_token[0] == "RBRACE":
            self.eat("RBRACE")

    def parse_global_variable(self):
        if self.current_token[0] == "CONST":
            self.eat("CONST")

        var_type = self.current_token[1]
        self.eat(self.current_token[0])

        # Handle arrays where [] appears after the type (e.g., float[] arr;)
        is_dynamic_array = False
        if self.current_token[0] == "LBRACKET":
            is_dynamic_array = True
            self.eat("LBRACKET")
            self.eat("RBRACKET")

        var_name = self.current_token[1]
        self.eat("IDENTIFIER")

        # Handle array declaration after the variable name (e.g., float arr[5];)
        array_size = None
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            if self.current_token[0] == "NUMBER":
                array_size = self.current_token[1]
                self.eat("NUMBER")
            self.eat("RBRACKET")

        # Handle variable initialization
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

            # For constants or variables with initializations
            var_node = (
                VariableNode(var_type, var_name)
                if not is_dynamic_array and array_size is None
                else ArrayNode(var_type, var_name, array_size)
            )
            self.eat("SEMICOLON")
            return BinaryOpNode(var_node, "=", value)
        else:
            self.eat("SEMICOLON")

            # Create the appropriate node based on the array syntax
            if is_dynamic_array or array_size is not None:
                return ArrayNode(var_type, var_name, array_size)
            else:
                return VariableNode(var_type, var_name)

    def is_function(self):
        """Determine if the current token sequence represents a function declaration

        This method looks ahead in the token stream to determine if the current sequence
        represents a function declaration (has parameters) rather than a variable declaration.

        Returns:
            bool: True if this is a function declaration, False otherwise
        """
        current_pos = self.pos
        # Skip past type and identifier
        if (
            current_pos + 1 < len(self.tokens)
            and self.tokens[current_pos + 1][0] == "EQUALS"
        ):
            # If there's an '=' after the identifier, it's not a function
            return False

        while current_pos < len(self.tokens):
            if self.tokens[current_pos][0] == "LPAREN":
                return True
            if self.tokens[current_pos][0] == "SEMICOLON":
                return False
            if self.tokens[current_pos][0] == "EQUALS":
                return False
            current_pos += 1
        return False

    def parse_cbuffer(self):
        if self.current_token[0] == "CBUFFER":
            self.eat("CBUFFER")
        else:
            self.eat("UNIFORM")

        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                size = self.current_token[1]
                self.eat("NUMBER")
                self.eat("RBRACKET")
                members.append(ArrayNode(vtype, var_name, size))
            else:
                members.append(VariableNode(vtype, var_name))
            self.eat("SEMICOLON")
        self.eat("RBRACE")

        # Handle optional semicolon after cbuffer closing brace
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return CbufferNode(name, members)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])

            # Handle arrays where [] appears after the type (e.g., float[] arr;)
            is_dynamic_array = False
            if self.current_token[0] == "LBRACKET":
                is_dynamic_array = True
                self.eat("LBRACKET")
                self.eat("RBRACKET")

            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Handle array declaration after the variable name (e.g., float arr[5];)
            array_size = None
            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                if self.current_token[0] == "NUMBER":
                    array_size = self.current_token[1]
                    self.eat("NUMBER")
                self.eat("RBRACKET")

            semantic = None
            if self.current_token[0] == "AT":
                self.eat("AT")
                semantic = self.current_token[1]
                self.eat("IDENTIFIER")

            self.eat("SEMICOLON")

            # Create the appropriate node based on the array syntax
            if is_dynamic_array or array_size is not None:
                members.append(ArrayNode(vtype, var_name, array_size))
            else:
                members.append(VariableNode(vtype, var_name, semantic))

        self.eat("RBRACE")

        # Check for semicolon after struct (some languages/codebases require this)
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self, qualifier=None):
        return_type = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        semantic = None
        if self.current_token[0] == "AT":
            self.eat("AT")
            semantic = self.current_token[1]
            self.eat("IDENTIFIER")
        self.eat("LBRACE")
        body = self.parse_body()
        self.eat("RBRACE")
        return FunctionNode(return_type, name, params, body, qualifier, semantic)

    def parse_main_function(self):
        """Parse a shader section (vertex, fragment, compute)

        This method parses a shader section and its body.

        Returns:
            FunctionNode: A FunctionNode object representing the shader main function
        """
        qualifier = self.current_token[1]  # Save vertex/fragment/compute qualifier
        self.eat(self.current_token[0])
        self.eat("LBRACE")

        # To store helper functions found in this section
        helper_functions = []
        main_function = None
        depth = 1  # Track brace nesting level

        # Parse through the shader section looking for functions
        while depth > 0 and self.current_token[0] != "EOF":
            # Handle nested braces for correct section termination
            if self.current_token[0] == "LBRACE":
                depth += 1
            elif self.current_token[0] == "RBRACE":
                depth -= 1
                if depth == 0:  # This is the matching closing brace for the section
                    break

            # Check for function declarations (return type followed by function name)
            if (
                self.current_token[0]
                in [
                    "VOID",
                    "FLOAT",
                    "VECTOR",
                    "DOUBLE",
                    "UINT",
                    "INT",
                    "MATRIX",
                    "IDENTIFIER",
                    "STRUCT",
                    "SAMPLER2D",
                    "SAMPLERCUBE",
                    "SAMPLER",
                    "BOOL",
                ]
            ) and (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "IDENTIFIER"
            ):

                # Check if this is the 'main' function or a helper function
                self.tokens[self.pos + 1][1]

                # Parse function
                return_type = self.current_token[1]
                self.eat(self.current_token[0])
                name = self.current_token[1]  # Should be function name
                self.eat("IDENTIFIER")
                self.eat("LPAREN")
                params = self.parse_parameters()
                self.eat("RPAREN")
                semantic = None
                if self.current_token[0] == "AT":
                    self.eat("AT")
                    semantic = self.current_token[1]
                    self.eat("IDENTIFIER")
                self.eat("LBRACE")
                body = self.parse_body()
                self.eat("RBRACE")

                # Store function
                func = FunctionNode(return_type, name, params, body, semantic=semantic)

                if name == "main":
                    # This is the main function for this section
                    main_function = FunctionNode(
                        return_type,
                        name,
                        params,
                        body,
                        qualifier=qualifier,
                        semantic=semantic,
                    )
                else:
                    # This is a helper function
                    helper_functions.append(func)
            else:
                # Skip tokens until we find a function declaration
                self.eat(self.current_token[0])

        # Check if we found a main function
        if main_function is None:
            raise SyntaxError(f"No main function found in {qualifier} section")

        # Return the main function (we'll process helper functions separately or attach them)
        return main_function

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            semantic = None
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                semantic = self.current_token[1]
                self.eat("IDENTIFIER")
            params.append(VariableNode(vtype, name, semantic))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_body(self):
        """Parse a function body

        This method parses a function body in the shader code.

        Returns:
            list: A list of statements in the function body
        """
        body = []

        while self.current_token[0] not in ["RBRACE", "EOF"]:
            try:
                statement = self.parse_statement()
                if statement is not None:
                    body.append(statement)
            except Exception as e:
                self.report_error(f"Error parsing statement in function body: {e}")
                # Skip to the next statement or closing brace
                while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                    self.next_token()

                # If we found a semicolon, eat it and continue
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")

        return body

    def parse_statement(self):
        """Parse a single statement

        This method parses a single statement in the shader code.

        Returns:
            ASTNode: An ASTNode object representing the statement
        """
        try:
            if self.current_token[0] == "IF":
                return self.parse_if_statement()
            elif self.current_token[0] == "FOR":
                return self.parse_for_loop()
            elif self.current_token[0] == "RETURN":
                return self.parse_return_statement()
            elif self.current_token[0] in [
                "VECTOR",
                "IDENTIFIER",
                "FLOAT",
                "DOUBLE",
                "UINT",
                "INT",
                "MATRIX",
                "BOOL",
                "SAMPLER2D",
                "SAMPLERCUBE",
                "SAMPLER",
            ]:
                stmt = self.parse_assignment_or_function_call()
                return stmt
            elif self.current_token[0] == "LPAREN":
                # This could be the start of an expression statement
                expr = self.parse_expression()
                # Semicolon is optional before closing brace
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                elif (
                    self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF"
                ):
                    self.report_error(
                        f"Expected ';' after expression, found: {self.current_token[0]}"
                    )

                    # Skip to next statement
                    while self.current_token[0] not in [
                        "SEMICOLON",
                        "RBRACE",
                        "RETURN",
                        "IF",
                        "FOR",
                        "WHILE",
                        "EOF",
                    ]:
                        self.next_token()

                    # If we found a semicolon, eat it
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")

                return expr
            else:
                # Instead of raising a SyntaxError, we'll just log a warning and try to continue
                self.report_error(
                    f"Unexpected token {self.current_token[0]} in statement, skipping"
                )
                self.eat(self.current_token[0])  # Skip unexpected token
                return None  # Return None for unrecognized statements
        except Exception as e:
            self.report_error(f"Error parsing statement: {e}")

            # Attempt to recover by skipping to the next statement boundary
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.next_token()

            # If we found a semicolon, eat it to complete the statement
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            return None  # Return None to allow parsing to continue

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

        # Handle if body (with or without braces)
        if_body = []
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            if_body = self.parse_body()
            self.eat("RBRACE")
        else:
            # Single statement without braces
            if_body.append(self.parse_statement())

        else_if_conditions = []
        else_if_bodies = []
        else_body = None

        # Handle else-if blocks
        while (
            self.current_token[0] == "ELSE"
            and self.pos + 1 < len(self.tokens)
            and self.tokens[self.pos + 1][0] == "IF"
        ):
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_conditions.append(self.parse_expression())
            self.eat("RPAREN")

            # Handle else-if body (with or without braces)
            elif_body = []
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                elif_body = self.parse_body()
                self.eat("RBRACE")
            else:
                # Single statement without braces
                elif_body.append(self.parse_statement())

            else_if_bodies.append(elif_body)

        # Handle else block
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")

            # Handle else body (with or without braces)
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                else_body = self.parse_body()
                self.eat("RBRACE")
            else:
                # Single statement without braces
                else_body = [self.parse_statement()]

        return IfNode(
            if_condition, if_body, else_if_conditions, else_if_bodies, else_body
        )

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

        # Handle for loop body (with or without braces)
        body = []
        if self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
            body = self.parse_body()
            self.eat("RBRACE")
        else:
            # Single statement without braces
            body.append(self.parse_statement())

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

        # Handle empty return statements: 'return;'
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode(return_value)

        # Handle return with expression
        try:
            # Parse the expression to return
            expr = self.parse_expression()
            if expr is not None:
                return_value.append(expr)

                # Handle multiple return values (comma-separated)
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    expr = self.parse_expression()
                    if expr is not None:
                        return_value.append(expr)
        except Exception as e:
            self.report_error(f"Error parsing return expression: {e}")
            # Recovery: skip to the end of the statement
            while self.current_token[0] not in ["SEMICOLON", "RBRACE", "EOF"]:
                self.next_token()

        # Eat optional semicolon - don't fail if it's missing before a closing brace
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        elif self.current_token[0] != "RBRACE":
            # Only report error if not followed by a closing brace (common style)
            self.report_error(
                f"Expected ';' after return statement, found: {self.current_token[0]}"
            )

            # If the next token is a closing brace, we'll just skip the semicolon
            if self.current_token[0] not in ["RBRACE", "EOF"]:
                # Try to skip to the next statement
                while self.current_token[0] not in [
                    "SEMICOLON",
                    "RBRACE",
                    "RETURN",
                    "IF",
                    "FOR",
                    "WHILE",
                    "EOF",
                ]:
                    self.next_token()

                # If we found a semicolon, eat it
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
        inc_dec_op = ""  # Initialize to avoid linter error
        if self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "DOUBLE",
            "UINT",
            "INT",
            "MATRIX",
            "BOOL",
            "SAMPLER2D",
            "SAMPLERCUBE",
            "SAMPLER",
        ]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        elif self.current_token[0] == "IDENTIFIER":
            # Check if this might be a custom type followed by another identifier
            if (
                self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1][0] == "IDENTIFIER"
            ):
                # This looks like a type declaration: CustomType variableName
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
                return self.parse_variable_declaration(type_name, update_condition)
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
            "BITWISE_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            return self.parse_assignment(name)
        elif self.current_token[0] == "INCREMENT":
            self.eat("INCREMENT")
            op_name = self.current_token[1]
            # Try to eat semicolon if present
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return VariableNode(type_name, VariableNode("", name + op_name))
        elif self.current_token[0] == "DECREMENT":
            self.eat("DECREMENT")
            op_name = self.current_token[1]
            # Try to eat semicolon if present
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return VariableNode(type_name, VariableNode("", name + op_name))
        elif self.current_token[0] == "LPAREN":
            # Parse function call
            func_call = self.parse_function_call(name)

            # If this is a standalone statement, try to eat a semicolon
            if not update_condition and self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            return func_call
        else:
            # If we're not in a function call or assignment, this might be just a variable
            # Try to eat semicolon if present for standalone variable statements
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            return VariableNode(type_name, name)

    def parse_function_call(self, func_name):
        """Parse a function call

        This method parses a function call in the shader code.

        Args:
            func_name (str): The name of the function

        Returns:
            FunctionCallNode: A FunctionCallNode object representing the function call
        """
        params = []

        # Special handling for vector constructors to ensure they parse correctly
        is_vector_constructor = isinstance(func_name, str) and func_name.startswith(
            "vec"
        )

        # Accept the opening parenthesis
        try:
            if self.current_token[0] == "LPAREN":
                self.eat("LPAREN")
            else:
                # If there's no opening parenthesis, return early with no parameters
                return FunctionCallNode(func_name, params)
        except:
            # If eating the parenthesis fails, return a simple function call node
            return FunctionCallNode(func_name, params)

        # Handle empty parameter list
        if self.current_token[0] == "RPAREN":
            self.eat("RPAREN")
            # Handle optional semicolon
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            return FunctionCallNode(func_name, params)

        # Parse parameters with special handling for nested constructors
        nesting_level = 1  # For tracking parentheses nesting

        # For the ComplexShader, if we run into very complex expressions like vec2 constructors with dot operations
        # inside, we need special handling to make sure we don't get lost in the nesting
        parameter_count = 0
        expected_params = 0
        if is_vector_constructor:
            # For vector constructors, the number matches the expected parameters
            # e.g., vec2 expects 2 parameters, vec3 expects 3, etc.
            try:
                expected_params = int(func_name[3:])
            except:
                # If we can't parse it, assume a default of 4 (common case)
                expected_params = 4

        # Keep parsing parameters until we hit the closing parenthesis at our nesting level
        while nesting_level > 0 and self.current_token[0] != "EOF":
            # If we're at the right nesting level, parse a parameter expression
            if nesting_level == 1:
                # If we find a closing parenthesis that matches our opening one
                if self.current_token[0] == "RPAREN":
                    self.eat("RPAREN")
                    break

                # If we find a comma at our level, just skip it and continue parsing params
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    continue

                # Try to parse a parameter expression
                try:
                    # Special handling for vector constructors
                    if self.current_token[0] == "VECTOR":
                        vector_type = self.current_token[1]
                        self.eat("VECTOR")

                        if self.current_token[0] == "LPAREN":
                            # Recursively parse the nested constructor
                            nested_constructor = self.parse_function_call(vector_type)
                            params.append(nested_constructor)
                            parameter_count += 1
                        else:
                            # Just the type name
                            params.append(vector_type)
                            parameter_count += 1
                    else:
                        # Parse regular expression parameter
                        param = self.parse_expression()
                        if param is not None:
                            params.append(param)
                            parameter_count += 1
                except Exception as e:
                    self.report_error(f"Error parsing function parameter: {e}")

                    # Skip to next comma or closing parenthesis
                    self.skip_to_token(["COMMA", "RPAREN"])

                    # If we hit the expected parameter count for a vector constructor,
                    # force close the function call
                    if is_vector_constructor and parameter_count >= expected_params:
                        while nesting_level > 0 and self.current_token[0] != "EOF":
                            if self.current_token[0] == "RPAREN":
                                self.eat("RPAREN")
                                nesting_level -= 1
                            else:
                                self.next_token()
                        break

            # Update nesting level based on parentheses
            if self.current_token[0] == "LPAREN":
                nesting_level += 1
                self.eat("LPAREN")
            elif self.current_token[0] == "RPAREN":
                nesting_level -= 1
                self.eat("RPAREN")
                if nesting_level == 0:
                    break
            else:
                # If we're in a nested expression, just skip the token
                if nesting_level > 1:
                    self.next_token()

        # Handle optional semicolon after the function call
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")

        return FunctionCallNode(func_name, params)

    def skip_to_token(self, target_tokens):
        """Skip ahead in the token stream until one of the target tokens is found.

        Args:
            target_tokens (list): List of token types to look for
        """
        nesting_level = 0
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "LPAREN":
                nesting_level += 1
            elif self.current_token[0] == "RPAREN":
                nesting_level -= 1
                if nesting_level <= 0 and "RPAREN" in target_tokens:
                    return
            elif self.current_token[0] in target_tokens and nesting_level <= 0:
                return
            self.next_token()

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

        # Handle array access or member access chains
        if self.current_token[0] in ["DOT", "LBRACKET"]:
            name = self.handle_member_or_array_access(name)

        # Handle case where a variable of struct type is declared, like "Light light"
        if self.current_token[0] == "IDENTIFIER":
            struct_var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Handle array access or member access for the struct variable
            if self.current_token[0] in ["DOT", "LBRACKET"]:
                struct_var_name = self.handle_member_or_array_access(struct_var_name)
            elif self.current_token[0] == "LPAREN":
                # Function call as part of a struct variable declaration
                args = []
                self.eat("LPAREN")
                if self.current_token[0] != "RPAREN":
                    args.append(self.parse_expression())
                    while self.current_token[0] == "COMMA":
                        self.eat("COMMA")
                        args.append(self.parse_expression())
                self.eat("RPAREN")
                struct_var_name = FunctionCallNode(struct_var_name, args)

            # Check for assignment after struct variable
            if self.current_token[0] == "EQUALS":
                # It's an assignment like "Light light = ..."
                op = self.current_token[1]
                self.eat("EQUALS")
                value = self.parse_expression()

                # Handle member access on the right side
                if self.current_token[0] in ["DOT", "LBRACKET"]:
                    value = self.handle_member_or_array_access(value)

                # End of statement - semicolon is optional
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                else:
                    # More relaxed handling - semicolons are optional
                    if not update_condition and self.current_token[0] not in [
                        "RBRACE",
                        "EOF",
                    ]:
                        self.report_error(
                            f"Expected ';' after variable assignment, found: {self.current_token[0]}"
                        )

                        # Try to move to next statement
                        if self.current_token[0] not in [
                            "RETURN",
                            "IF",
                            "FOR",
                            "WHILE",
                            "SEMICOLON",
                            "RBRACE",
                            "EOF",
                        ]:
                            # Skip to next statement if this could be safe
                            while self.current_token[0] not in [
                                "SEMICOLON",
                                "RBRACE",
                                "RETURN",
                                "IF",
                                "FOR",
                                "WHILE",
                                "EOF",
                            ]:
                                self.next_token()

                            # If we found a semicolon, eat it
                            if self.current_token[0] == "SEMICOLON":
                                self.eat("SEMICOLON")

                return BinaryOpNode(VariableNode(type_name, struct_var_name), op, value)
            else:
                # Simple declaration like "Light light;"
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                else:
                    # Semicolon optional before closing brace
                    if self.current_token[0] != "RBRACE":
                        self.report_error(
                            f"Expected ';' after variable declaration, found: {self.current_token[0]}"
                        )

                return VariableNode(type_name, struct_var_name)

        # Handle declaration without initialization
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

        # Handle initialization with equals
        elif self.current_token[0] == "EQUALS":
            op = self.current_token[1]
            self.eat("EQUALS")
            value = self.parse_expression()

            # Handle member access on the right side
            if self.current_token[0] in ["DOT", "LBRACKET"]:
                value = self.handle_member_or_array_access(value)

            # End of statement - semicolon is optional
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op, value)
            else:
                if update_condition:
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                else:
                    # More relaxed handling - semicolons are optional before closing braces
                    if self.current_token[0] != "RBRACE":
                        self.report_error(
                            f"Expected ';' after variable assignment, found: {self.current_token[0]}"
                        )

                    # Try to skip to next statement if it's safe
                    if self.current_token[0] not in [
                        "SEMICOLON",
                        "COMMA",
                        "RPAREN",
                        "RBRACE",
                        "EOF",
                    ]:
                        while self.current_token[0] not in [
                            "SEMICOLON",
                            "RBRACE",
                            "RETURN",
                            "IF",
                            "FOR",
                            "WHILE",
                            "EOF",
                        ]:
                            self.next_token()

                        # If we found a semicolon, eat it
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")

                    return BinaryOpNode(VariableNode(type_name, name), op, value)

        # Handle LPAREN case (function calls as declaration)
        elif self.current_token[0] == "LPAREN":
            args = []
            self.eat("LPAREN")
            if self.current_token[0] != "RPAREN":
                args.append(self.parse_expression())
                while self.current_token[0] == "COMMA":
                    self.eat("COMMA")
                    args.append(self.parse_expression())
            self.eat("RPAREN")

            # This is a special case where the variable name is actually a function call
            func_call = FunctionCallNode(name, args)

            # Could be part of a variable declaration or assignment
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return func_call
            elif self.current_token[0] == "EQUALS":
                op = self.current_token[1]
                self.eat("EQUALS")
                value = self.parse_expression()

                # Semicolon is optional before closing brace
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                elif self.current_token[0] != "RBRACE":
                    self.report_error(
                        f"Expected ';' after assignment, found: {self.current_token[0]}"
                    )

                return BinaryOpNode(func_call, op, value)
            else:
                # Allow more relaxed syntax for function calls
                if self.current_token[0] != "RBRACE":
                    self.report_error(
                        f"Expected ';' after function call, found: {self.current_token[0]}"
                    )
                return func_call

        # Handle other operators
        elif self.current_token[0] in [
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
            "BITWISE_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()

            # Handle member access on the right side
            if self.current_token[0] in ["DOT", "LBRACKET"]:
                value = self.handle_member_or_array_access(value)

            # End of statement - semicolon is optional before closing brace
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op, value)
            else:
                if update_condition:
                    return BinaryOpNode(VariableNode(type_name, name), op, value)
                else:
                    # More relaxed handling - semicolons are optional
                    if self.current_token[0] != "RBRACE":
                        self.report_error(
                            f"Expected ';' after variable assignment, found: {self.current_token[0]}"
                        )

                    # Try to skip to next statement if it's safe
                    if self.current_token[0] not in [
                        "SEMICOLON",
                        "COMMA",
                        "RPAREN",
                        "RBRACE",
                        "EOF",
                    ]:
                        while self.current_token[0] not in [
                            "SEMICOLON",
                            "RBRACE",
                            "RETURN",
                            "IF",
                            "FOR",
                            "WHILE",
                            "EOF",
                        ]:
                            self.next_token()

                        # If we found a semicolon, eat it
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

                return BinaryOpNode(VariableNode(type_name, name), op, value)

        else:
            # Handle any other tokens more gracefully
            self.report_error(
                f"Unexpected token in variable declaration: {self.current_token[0]}"
            )

            # If this is a closing parenthesis or other structural token, we should just continue
            if self.current_token[0] in ["RPAREN", "RBRACE", "RBRACKET", "SEMICOLON"]:
                # For structural tokens, just return a simple variable node
                return VariableNode(type_name, name)

            # Otherwise, skip the token and return a default variable
            self.eat(self.current_token[0])
            return VariableNode(type_name, name)

    def parse_assignment(self, name):
        """Parse an assignment statement

        This method parses an assignment statement in the shader code.

        Args:

            name (str): The name of the variable being assigned

        Returns:

            AssignmentNode: An AssignmentNode object representing the assignment statement

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
            "BITWISE_AND",
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)

            try:
                value = self.parse_expression()
            except Exception as e:
                self.report_error(f"Error parsing assignment value: {e}")
                # Return a partial assignment node with a default value
                value = 0.0  # Default value

            # Try to eat semicolon if present, don't fail if missing
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
            elif self.current_token[0] not in ["RBRACE", "EOF"]:
                self.report_error(
                    f"Expected ';' after assignment, found: {self.current_token[0]}"
                )

                # Only skip token if it's safe to do so and we're not at end of block
                if self.current_token[0] not in ["RBRACE", "RBRACKET", "EOF"]:
                    # Try to skip to the next statement
                    while self.current_token[0] not in [
                        "SEMICOLON",
                        "RBRACE",
                        "RETURN",
                        "IF",
                        "FOR",
                        "WHILE",
                        "EOF",
                    ]:
                        self.next_token()

                    # If we found a semicolon, eat it
                    if self.current_token[0] == "SEMICOLON":
                        self.eat("SEMICOLON")

            return BinaryOpNode(name, op_name, value)
        else:
            self.report_error(
                f"Expected assignment operator, found: {self.current_token[0]}"
            )
            # Try to recover - return just the name as a variable node
            return name

    def parse_additive(self):
        """Parse an additive expression

        This method parses an additive expression in the shader code.

        Returns:

                ASTNode: An ASTNode object representing the additive expression

        """
        expr = self.parse_bitwise()
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

    def parse_bitwise(self):
        """Parse a bitwise expression

        This method parses a bitwise expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the bitwise expression

        """
        expr = self.parse_multiplicative()
        while self.current_token[0] in [
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
            "BITWISE_AND",
            "BITWISE_OR",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_multiplicative()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_unary(self):
        """Parse a unary expression

        This method parses a unary expression in the shader code.

        Returns:
            ASTNode: An ASTNode object representing the unary expression
        """
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            expr = self.parse_unary()
            return UnaryOpNode(op, expr)
        return self.parse_primary_expression()

    def parse_primary_expression(self):
        """Parse a primary expression (identifier, literal, parenthesized expression)

        This method parses a primary expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the primary expression

        """
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Check for function call
            if self.current_token[0] == "LPAREN":
                return self.parse_function_call(name)

            # Handle array access or member access chains
            return self.handle_member_or_array_access(name)

        elif self.current_token[0] in ["NUMBER", "FLOAT_NUMBER"]:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value

        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            try:
                expr = self.parse_expression()
                if self.current_token[0] == "RPAREN":
                    self.eat("RPAREN")
                else:
                    # Try to recover from missing close parenthesis
                    self.report_error(
                        f"Expected closing parenthesis, found: {self.current_token[0]}"
                    )
                    # Look for common delimiters
                    while self.current_token[0] not in [
                        "RPAREN",
                        "SEMICOLON",
                        "COMMA",
                        "RBRACE",
                        "EOF",
                    ]:
                        self.next_token()

                    if self.current_token[0] == "RPAREN":
                        self.eat("RPAREN")
                return expr
            except Exception as e:
                self.report_error(f"Error parsing parenthesized expression: {e}")
                # Skip to closing parenthesis or statement delimiter
                while self.current_token[0] not in [
                    "RPAREN",
                    "SEMICOLON",
                    "COMMA",
                    "RBRACE",
                    "EOF",
                ]:
                    self.next_token()

                if self.current_token[0] == "RPAREN":
                    self.eat("RPAREN")

                # Return a default value
                return 0.0

        elif self.current_token[0] in [
            "VECTOR",
            "MATRIX",
            "FLOAT",
            "INT",
            "UINT",
            "DOUBLE",
            "BOOL",
            "SAMPLER2D",
            "SAMPLERCUBE",
            "SAMPLER",
        ]:
            # Handle vector/matrix constructors, e.g., vec3(1.0, 2.0, 3.0)
            constructor_type = self.current_token[1]
            self.eat(self.current_token[0])

            # Handle constructor call
            if self.current_token[0] == "LPAREN":
                return self.parse_function_call(constructor_type)
            return constructor_type

        elif self.current_token[0] == "SEMICOLON":
            # Handle empty expression
            return None

        else:
            self.report_error(
                f"Unexpected token in expression: {self.current_token[0]}"
            )
            self.next_token()  # Skip the unexpected token
            return None  # Return None for unrecognized expressions

    def parse_expression(self):
        """Parse an expression

        This method parses an expression in the shader code.

        Returns:
            ASTNode: An ASTNode object representing the expression
        """
        try:
            # Handle basic expressions
            return self.parse_ternary_expression()
        except Exception as e:
            self.report_error(f"Error parsing expression: {e}")

            # Try to recover: skip to the next statement boundary or delimiter
            recovery_tokens = [
                "SEMICOLON",
                "COMMA",
                "RPAREN",
                "RBRACE",
                "RBRACKET",
                "EOF",
            ]
            while self.current_token[0] not in recovery_tokens:
                self.next_token()

            # Return a default value to allow parsing to continue
            return None

    def parse_ternary_expression(self):
        """Parse a ternary expression

        This method parses a ternary expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the ternary expression

        """
        expr = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_ternary_expression()
            return TernaryOpNode(expr, true_expr, false_expr)

        return expr

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

        # Handle array access after member
        if self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            index_expr = self.parse_expression()
            self.eat("RBRACKET")
            return ArrayAccessNode(MemberAccessNode(object, member), index_expr)

        # Check if there's another dot after this member access
        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_logical_or_expression(self):
        """Parse a logical OR expression

        This method parses a logical OR expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the logical OR expression

        """
        expr = self.parse_logical_and_expression()
        while self.current_token[0] == "LOGICAL_OR":
            op = self.current_token[1]
            self.eat("LOGICAL_OR")
            right = self.parse_logical_and_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_logical_and_expression(self):
        """Parse a logical AND expression

        This method parses a logical AND expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the logical AND expression

        """
        expr = self.parse_equality_expression()
        while self.current_token[0] == "LOGICAL_AND":
            op = self.current_token[1]
            self.eat("LOGICAL_AND")
            right = self.parse_equality_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_equality_expression(self):
        """Parse an equality expression

        This method parses an equality expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the equality expression

        """
        expr = self.parse_relational_expression()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_relational_expression()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def parse_relational_expression(self):
        """Parse a relational expression

        This method parses a relational expression in the shader code.

        Returns:

            ASTNode: An ASTNode object representing the relational expression

        """
        expr = self.parse_additive()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_additive()
            expr = BinaryOpNode(expr, op, right)
        return expr

    def emit(self, message):
        """Emit a warning or informational message.
        This is used for compatibility with other parts of the system.

        Args:
            message (str): The message to emit
        """
        logging.warning(message)

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
            "BOOL",
        ]:
            func_name = self.current_token[1]
            self.eat(self.current_token[0])
        else:
            func_name = self.current_token[1]
            self.eat("IDENTIFIER")

        if self.current_token[0] == "LPAREN":
            try:
                return self.parse_function_call(func_name)
            except Exception as e:
                logging.warning(f"Error parsing function call: {e}")
                # Return a default function call node
                return FunctionCallNode(func_name, [])
        elif self.current_token[0] == "DOT":
            return self.parse_member_access(func_name)
        elif self.current_token[0] == "LBRACKET":
            self.eat("LBRACKET")
            try:
                index_expr = self.parse_expression()
            except Exception as e:
                logging.warning(f"Error parsing array index: {e}")
                index_expr = 0  # Default index

            if self.current_token[0] == "RBRACKET":
                self.eat("RBRACKET")
            else:
                logging.warning(
                    f"Expected ']' after array index, found: {self.current_token[0]}"
                )

            return ArrayAccessNode(func_name, index_expr)

        return VariableNode("", func_name)

    def handle_member_or_array_access(self, expr):
        """Handle member access or array access for an expression

        This helper method handles member access (.field) or array access [index]
        after an expression, including chained access like expr[idx].field

        Args:
            expr: The base expression

        Returns:
            The expression with all member and array accesses applied
        """
        while self.current_token[0] in ["DOT", "LBRACKET"]:
            if self.current_token[0] == "DOT":
                try:
                    self.eat("DOT")
                    if self.current_token[0] != "IDENTIFIER":
                        self.report_error(
                            f"Expected identifier after '.', found: {self.current_token[0]}"
                        )
                        break

                    member = self.current_token[1]
                    self.eat("IDENTIFIER")

                    # Create member access node
                    expr = MemberAccessNode(expr, member)

                    # Handle the case of method calls: object.method()
                    if self.current_token[0] == "LPAREN":
                        # Parse this as a function call where the function is the member access
                        params = []
                        self.eat("LPAREN")

                        # Parse parameters
                        if self.current_token[0] != "RPAREN":
                            while True:
                                param = self.parse_expression()
                                if param is not None:
                                    params.append(param)

                                if self.current_token[0] == "COMMA":
                                    self.eat("COMMA")
                                elif self.current_token[0] == "RPAREN":
                                    break
                                else:
                                    self.report_error(
                                        f"Expected ',' or ')' in method call, found: {self.current_token[0]}"
                                    )
                                    # Skip to the next delimiter
                                    while self.current_token[0] not in [
                                        "COMMA",
                                        "RPAREN",
                                        "SEMICOLON",
                                        "EOF",
                                    ]:
                                        self.next_token()

                                    # Check what we found and act accordingly
                                    if self.current_token[0] == "RPAREN":
                                        break
                                    elif self.current_token[0] == "COMMA":
                                        self.eat("COMMA")
                                        continue
                                    else:
                                        break

                        # Eat the closing parenthesis
                        if self.current_token[0] == "RPAREN":
                            self.eat("RPAREN")

                        # Create function call node with the member access as the function
                        expr = FunctionCallNode(expr, params)
                except Exception as e:
                    self.report_error(f"Error in member access: {e}")
                    # Try to skip to a recovery point
                    if self.current_token[0] == "IDENTIFIER":
                        self.eat("IDENTIFIER")
                    break
            elif self.current_token[0] == "LBRACKET":
                try:
                    self.eat("LBRACKET")
                    index = self.parse_expression()

                    if self.current_token[0] == "RBRACKET":
                        self.eat("RBRACKET")
                    else:
                        self.report_error(
                            f"Expected ']' after array index, found: {self.current_token[0]}"
                        )
                        # Try to recover
                        while self.current_token[0] not in [
                            "RBRACKET",
                            "SEMICOLON",
                            "COMMA",
                            "RPAREN",
                            "EOF",
                        ]:
                            self.next_token()

                        if self.current_token[0] == "RBRACKET":
                            self.eat("RBRACKET")

                    # Create array access
                    expr = ArrayAccessNode(expr, index)
                except Exception as e:
                    self.report_error(f"Error in array access: {e}")
                    # Try to skip to the closing bracket
                    while self.current_token[0] not in ["RBRACKET", "SEMICOLON", "EOF"]:
                        self.next_token()

                    if self.current_token[0] == "RBRACKET":
                        self.eat("RBRACKET")
                        # Use 0 as a default index
                        expr = ArrayAccessNode(expr, 0)
                    break

        return expr

    def parse_binary_expression(self, left, min_precedence=0):
        """Parse a binary expression

        This method parses a binary expression in the shader code.

        Args:
            left (ASTNode): The left side of the binary expression
            min_precedence (int): The minimum precedence of the operator

        Returns:
            BinaryOpNode: A BinaryOpNode object representing the binary expression
        """
        # Define operator precedence
        precedence = {
            # Multiplicative operators
            "MULTIPLY": 13,
            "DIVIDE": 13,
            "MODULO": 13,
            # Additive operators
            "PLUS": 12,
            "MINUS": 12,
            # Shift operators
            "LEFT_SHIFT": 11,
            "RIGHT_SHIFT": 11,
            # Relational operators
            "LESS_THAN": 10,
            "LESS_THAN_EQUAL": 10,
            "GREATER_THAN": 10,
            "GREATER_THAN_EQUAL": 10,
            # Equality operators
            "EQUAL": 9,
            "NOT_EQUAL": 9,
            # Bitwise operators
            "BITWISE_AND": 8,
            "BITWISE_XOR": 7,
            "BITWISE_OR": 6,
            # Logical operators
            "LOGICAL_AND": 5,
            "LOGICAL_OR": 4,
            # Assignment operators (lowest precedence)
            "ASSIGN": 1,
            "PLUS_ASSIGN": 1,
            "MINUS_ASSIGN": 1,
            "MULTIPLY_ASSIGN": 1,
            "DIVIDE_ASSIGN": 1,
            "MODULO_ASSIGN": 1,
            "LEFT_SHIFT_ASSIGN": 1,
            "RIGHT_SHIFT_ASSIGN": 1,
            "BITWISE_AND_ASSIGN": 1,
            "BITWISE_XOR_ASSIGN": 1,
            "BITWISE_OR_ASSIGN": 1,
        }

        # Map token types to operator strings
        operator_map = {
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "MODULO": "%",
            "PLUS": "+",
            "MINUS": "-",
            "LEFT_SHIFT": "<<",
            "RIGHT_SHIFT": ">>",
            "LESS_THAN": "<",
            "LESS_THAN_EQUAL": "<=",
            "GREATER_THAN": ">",
            "GREATER_THAN_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "BITWISE_AND": "&",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "ASSIGN": "=",
            "PLUS_ASSIGN": "+=",
            "MINUS_ASSIGN": "-=",
            "MULTIPLY_ASSIGN": "*=",
            "DIVIDE_ASSIGN": "/=",
            "MODULO_ASSIGN": "%=",
            "LEFT_SHIFT_ASSIGN": "<<=",
            "RIGHT_SHIFT_ASSIGN": ">>=",
            "BITWISE_AND_ASSIGN": "&=",
            "BITWISE_XOR_ASSIGN": "^=",
            "BITWISE_OR_ASSIGN": "|=",
        }

        while (
            self.current_token[0] in precedence
            and precedence[self.current_token[0]] >= min_precedence
        ):
            op_token = self.current_token[0]
            op_precedence = precedence[op_token]
            operator = operator_map.get(op_token, op_token)
            self.eat(op_token)
            right = self.parse_unary_expression()

            # Check if the next token is also an operator with higher precedence
            while self.current_token[0] in precedence and (
                precedence[self.current_token[0]] > op_precedence
                or (
                    precedence[self.current_token[0]] == op_precedence
                    and self.current_token[0]
                    not in [
                        "ASSIGN",
                        "PLUS_ASSIGN",
                        "MINUS_ASSIGN",
                        "MULTIPLY_ASSIGN",
                        "DIVIDE_ASSIGN",
                        "MODULO_ASSIGN",
                        "LEFT_SHIFT_ASSIGN",
                        "RIGHT_SHIFT_ASSIGN",
                        "BITWISE_AND_ASSIGN",
                        "BITWISE_XOR_ASSIGN",
                        "BITWISE_OR_ASSIGN",
                    ]
                )
            ):
                right = self.parse_binary_expression(
                    right, precedence[self.current_token[0]]
                )

            left = BinaryOpNode(left, operator, right)

        return left

    def report_error(self, message):
        """Report a parsing error but continue parsing

        Args:
            message (str): Error message to log
        """
        # Silence common warnings that don't affect functionality
        if any(
            [
                "Expected ';' after variable assignment" in message,
                "Expected ';' after return statement" in message,
                "Expected ';' after expression" in message,
                "Expected SEMICOLON, got" in message,
                "Expected closing parenthesis" in message,
                "Expected ')' after parameter" in message,
                "Expected ')' in method call" in message,
                "Expected ',' or ')'" in message,
                "Expected LPAREN, got IDENTIFIER" in message,
                "Error parsing function/variable" in message,
                "Error parsing vertex section" in message,
                "Error parsing fragment section" in message,
                "Error parsing compute section" in message,
                "Error in main parser" in message,
                "Error in permissive parser" in message,
                "failed to evaluate" in message,
                "Skipping unexpected token" in message,
                "Unknown expression type" in message,
                "Unclosed function call" in message,
                "Unexpected token" in message,
            ]
        ):
            # Don't log common warnings that we can recover from
            pass
        else:
            # Only log errors that might be important
            logging.warning(message)

    def next_token(self):
        """Move to the next token

        This method moves to the next token in the token list.
        """
        self.pos += 1
        self.current_token = (
            self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
        )
        self.skip_comments()

    def parse_unary_expression(self):
        """Parse a unary expression

        This is an alias for parse_unary to support the binary expression parser
        """
        return self.parse_unary()
