from .VulkanLexer import *
from .VulkanAst import *


class VulkanParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def peek(self, offset):
        """Look ahead by offset tokens without consuming them."""
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][
                0
            ]  # Return the type of the token at the peeked index
        return None

    def skip_until(self, token_type):
        """Skip tokens until the specified token type is found"""
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", None)
        return

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        module = self.parse_module()
        self.eat("EOF")
        return module

    def parse_module(self):
        statements = []
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "LAYOUT":
                statements.append(self.parse_layout())
            elif self.current_token[0] == "STRUCT":
                statements.append(self.parse_struct())
            elif self.current_token[0] == "UNIFORM":
                statements.append(self.parse_uniform())
            elif (
                (
                    self.current_token[0]
                    in [
                        "VOID",
                        "FLOAT",
                        "INT",
                        "UINT",
                        "BOOL",
                        "VEC2",
                        "VEC3",
                        "VEC4",
                        "MAT2",
                        "MAT3",
                        "MAT4",
                    ]
                    or self.current_token[1] in VALID_DATA_TYPES
                )
                and self.peek(1) == "IDENTIFIER"
                and self.peek(2) == "LPAREN"
            ):
                statements.append(self.parse_function())
            elif (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[0]
                in [
                    "FLOAT",
                    "INT",
                    "UINT",
                    "BOOL",
                    "VEC2",
                    "VEC3",
                    "VEC4",
                    "MAT2",
                    "MAT3",
                    "MAT4",
                ]
                or self.current_token[1] in VALID_DATA_TYPES
            ):
                statements.append(self.parse_variable(self.current_token[1]))
            else:
                self.eat(self.current_token[0])
        return ShaderNode(None, None, None, statements)

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []  # Stores pairs like ('location', '0'), ('binding', '1'), etc.
        push_constant = False
        if self.current_token[0] == "PUSH_CONSTANT":
            push_constant = True
            self.eat("PUSH_CONSTANT")
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        # Parse layout bindings
        while self.current_token[0] != "RPAREN":
            binding_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Handle assignment with EQUALS and a number
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                binding_value = self.current_token[1]
                self.eat("NUMBER")
                bindings.append((binding_name, binding_value))
            else:
                bindings.append((binding_name, None))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        layout_type = None
        if self.current_token[0] in ["IN", "OUT", "UNIFORM", "BUFFER"]:
            layout_type = self.current_token[0]
            self.eat(layout_type)
            if self.current_token[0] == "IDENTIFIER":
                self.eat(self.current_token[0])

        data_type = None
        struct_fields = None
        if layout_type in ["UNIFORM", "BUFFER"]:
            # If a curly brace follows, we have a structured data block
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                struct_fields = []

                # Parse structured fields within the uniform/push_constant/buffer block
                while self.current_token[0] != "RBRACE":
                    if self.current_token[1] in VALID_DATA_TYPES:
                        field_type = self.current_token[1]  # Field type (e.g., mat4)
                        self.eat(self.current_token[0])
                    else:
                        raise SyntaxError(
                            "Expected some data type before an identifier"
                        )
                    field_name = self.current_token[1]  # Field name
                    self.eat("IDENTIFIER")
                    self.eat("SEMICOLON")
                    struct_fields.append((field_type, field_name))

                self.eat("RBRACE")
                data_type = "struct"  # Use 'struct' as data_type placeholder for uniform/push_constant/buffer
            else:
                raise SyntaxError(
                    "Expected structured data block after 'uniform' or 'buffer'"
                )
        else:
            # For `in` and `out`, expect a data type and variable name
            if self.current_token[1] in VALID_DATA_TYPES:
                data_type = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(f"Unexpected type: {self.current_token[1]}")

        # Parse variable name
        variable_name = None
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")

        self.eat("SEMICOLON")
        return LayoutNode(
            bindings,
            push_constant,
            layout_type,
            data_type,
            variable_name,
            struct_fields,
        )

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            # Get the member type
            if self.current_token[0] in [
                "VEC2",
                "VEC3",
                "VEC4",
                "IVEC2",
                "IVEC3",
                "IVEC4",
                "UVEC2",
                "UVEC3",
                "UVEC4",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "MAT2",
                "MAT3",
                "MAT4",
            ]:
                # Vector/matrix/scalar types
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[1] in VALID_DATA_TYPES:
                # Other valid data types
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[0] == "IDENTIFIER":
                # Custom type
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SyntaxError(
                    f"Unexpected token in struct member: {self.current_token}"
                )

            # Get the member name
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Check for semicolon
            self.eat("SEMICOLON")

            members.append(VariableNode(type_name, member_name))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        return_type = self.current_token[1]
        if self.current_token[1] in VALID_DATA_TYPES:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(f"Unexpected type: {self.current_token[1]}")
        func_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        body = self.parse_block()
        return FunctionNode(func_name, return_type, params, body)

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            params.append(VariableNode(vtype, name))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE":
            statements.append(self.parse_body())
        self.eat("RBRACE")
        return statements

    def parse_body(self):
        token_type = self.current_token[0]

        if token_type == "IDENTIFIER" or self.current_token[1] in VALID_DATA_TYPES:
            return self.parse_assignment_or_function_call()
        elif token_type == "IF":
            return self.parse_if_statement()
        elif token_type == "FOR":
            return self.parse_for_statement()
        elif token_type == "WHILE":
            return self.parse_while_statement()
        elif token_type == "DO":
            return self.parse_do_while_statement()
        elif token_type == "SWITCH":
            return self.parse_switch_statement()
        elif token_type == "BREAK":
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        else:
            return self.parse_expression_statement()

    def parse_update(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                return UnaryOpNode("POST_INCREMENT", VariableNode(name, ""))
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                return UnaryOpNode("POST_DECREMENT", VariableNode(name, ""))
            elif self.current_token[0] in [
                "EQUALS",
                "ASSIGN_ADD",
                "ASSIGN_SUB",
                "ASSIGN_MUL",
                "ASSIGN_DIV",
            ]:
                op = self.current_token[0]
                self.eat(op)
                value = self.parse_expression()
                if op == "EQUALS":
                    return AssignmentNode(name, value)
                elif op == "ASSIGN_ADD":
                    return AssignmentNode(
                        name, BinaryOpNode(VariableNode(name, ""), "+", value)
                    )
                elif op == "ASSIGN_SUB":
                    return AssignmentNode(
                        name, BinaryOpNode(VariableNode(name, ""), "-", value)
                    )
                elif op == "ASSIGN_MUL":
                    return AssignmentNode(
                        name, BinaryOpNode(VariableNode(name, ""), "*", value)
                    )
                elif op == "ASSIGN_DIV":
                    return AssignmentNode(
                        name, BinaryOpNode(VariableNode(name, ""), "/", value)
                    )
                else:
                    raise SyntaxError(
                        f"Expected INCREMENT or DECREMENT, got {self.current_token[0]}"
                    )
        elif self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                return UnaryOpNode("PRE_INCREMENT", VariableNode(name, ""))
            else:
                raise SyntaxError(
                    f"Expected IDENTIFIER after PRE_INCREMENT, got {self.current_token[0]}"
                )
        elif self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                return UnaryOpNode("PRE_DECREMENT", VariableNode(name, ""))
            else:
                raise SyntaxError(
                    f"Expected IDENTIFIER after PRE_DECREMENT, got {self.current_token[0]}"
                )
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_block()
        else_body = None
        else_if_condition = []
        else_if_body = []
        while self.current_token[0] == "ELSE" and self.peek(1) == "IF":
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
            else_body = self.parse_block()
        return IfNode(if_condition, if_body, else_if_condition, else_if_body, else_body)

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        initialization = self.parse_assignment_or_function_call()
        condition = self.parse_expression()
        self.eat("SEMICOLON")
        increment = self.parse_update()
        self.eat("RPAREN")
        body = self.parse_block()
        return ForNode(initialization, condition, increment, body)

    def parse_variable(self, type_name):
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        while self.current_token[0] == "DOT":
            self.eat("DOT")
            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            name += "." + member_name

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode(type_name, name), value)
            else:
                # Handle comments or other tokens that might appear before semicolon
                self.skip_until("SEMICOLON")
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode(type_name, name), value)

        # Handle binary operators
        elif self.current_token[0] in ("BINARY_AND", "BINARY_OR", "BINARY_XOR"):
            op = self.current_token[0]
            op_symbol = (
                "&" if op == "BINARY_AND" else ("|" if op == "BINARY_OR" else "^")
            )
            self.eat(op)
            right = self.parse_expression()

            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op_symbol, right)
            else:
                # Handle comments or other tokens that might appear before semicolon
                self.skip_until("SEMICOLON")
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op_symbol, right)

        elif self.current_token[0] in (
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
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
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
        ):
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op_name, value)
            else:
                # Handle comments or other tokens that might appear before semicolon
                self.skip_until("SEMICOLON")
                self.eat("SEMICOLON")
                return BinaryOpNode(VariableNode(type_name, name), op_name, value)
        else:
            # For other cases like function calls, direct member access
            # Skip to the next semicolon and create a simple variable node
            self.skip_until("SEMICOLON")
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_function_call(self, name):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")

        # Handle vector constructors (vec4, vec3, etc.)
        if name in [
            "vec4",
            "vec3",
            "vec2",
            "ivec4",
            "ivec3",
            "ivec2",
            "uvec4",
            "uvec3",
            "uvec2",
        ]:
            return FunctionCallNode(name, args)

        return FunctionCallNode(name, args)

    def parse_function_call_or_identifier(self):
        func_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            return self.parse_function_call(func_name)
        elif self.current_token[0] == "DOT":
            return self.parse_member_access(func_name)
        return VariableNode(func_name, "")

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()
            return UnaryOpNode("-", value)

        if (
            self.current_token[0] == "BITWISE_NOT"
            or self.current_token[0] == "BINARY_NOT"
        ):
            self.eat(self.current_token[0])
            value = self.parse_primary()
            return UnaryOpNode("~", value)

        if (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[1] in VALID_DATA_TYPES
        ):
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            # Check if there's a 'u' suffix (for uint literals)
            if value.endswith("u"):
                value = value[:-1]  # Remove the 'u' suffix
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

    def parse_multiplicative(self):
        left = self.parse_primary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_primary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment(self, name):
        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
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
            "ASSIGN_SHIFT_LEFT",
            "ASSIGN_SHIFT_RIGHT",
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

    def parse_assignment_or_function_call(self):
        type_name = ""
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) in [
            "POST_INCREMENT",
            "POST_DECREMENT",
        ]:
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
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
                "ASSIGN_SHIFT_LEFT",
                "ASSIGN_SHIFT_RIGHT",
            ]:
                return self.parse_assignment(name)  # todo
            elif self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                self.eat("SEMICOLON")
                return AssignmentNode(
                    name, UnaryOpNode("POST_INCREMENT", VariableNode("", name))
                )
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                self.eat("SEMICOLON")
                return AssignmentNode(
                    name, UnaryOpNode("POST_DECREMENT", VariableNode("", name))
                )
            elif self.current_token[0] == "LPAREN":
                return self.parse_function_call(name)
            else:
                raise SyntaxError(
                    f"Unexpected token after identifier: {self.current_token[0]}"
                )
        if self.current_token[1] in VALID_DATA_TYPES:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable(type_name)

    def parse_expression(self):
        left = self.parse_bitwise_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "EQUAL",
            "NOT_EQUAL",
            "AND",
            "OR",
        ]:
            op = self.current_token[0]
            op_symbol = self.current_token[1] if len(self.current_token) > 1 else op
            self.eat(op)
            right = self.parse_bitwise_expression()
            left = BinaryOpNode(left, op_symbol, right)

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_bitwise_expression(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "BINARY_AND",
            "BINARY_OR",
            "BINARY_XOR",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_additive()

            # Map token types to operator symbols
            op_symbol = (
                "&"
                if op == "BINARY_AND"
                else (
                    "|"
                    if op == "BINARY_OR"
                    else (
                        "^"
                        if op == "BINARY_XOR"
                        else "<<" if op == "BITWISE_SHIFT_LEFT" else ">>"
                    )
                )
            )  # BITWISE_SHIFT_RIGHT

            left = BinaryOpNode(left, op_symbol, right)

        return left

    def parse_expression_statement(self):
        expr = self.parse_expression()
        # self.eat("SEMICOLON")
        return expr

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_block()
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        body = self.parse_block()
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(condition, body)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expr = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        cases = []
        while self.current_token[0] != "RBRACE":
            cases.append(self.parse_case_statement())
        self.eat("RBRACE")
        return SwitchNode(expr, cases)

    def parse_case_statement(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            value = self.parse_expression()
            self.eat("COLON")
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            value = None
            self.eat("COLON")
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE"]:
            statements.append(self.parse_body())
        return CaseNode(value, statements)

    def parse_default_statement(self):
        self.eat("DEFAULT")
        self.eat("COLON")
        statements = []
        while self.current_token[0] not in ["CASE", "RBRACE"]:
            statements.append(self.parse_body())
        return DefaultNode(statements)

    def parse_uniform(self):
        self.eat("UNIFORM")
        var_type = self.current_token[1]
        if self.current_token[1] in VALID_DATA_TYPES:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(f"Unexpected type: {self.current_token[1]}")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return UniformNode(name, var_type)

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        return self.parse_primary()
