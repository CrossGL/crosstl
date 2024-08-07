from .MetalLexer import *
from .MetalAst import *


class MetalParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

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
        shader = self.parse_shader()
        self.eat("EOF")
        return shader

    def parse_shader(self):
        functions = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PREPROCESSOR":
                self.parse_preprocessor_directive()
            elif self.current_token[0] == "USING":
                self.parse_using_statement()
            elif self.current_token[0] == "STRUCT":
                functions.append(self.parse_struct())
            elif self.current_token[0] == "CONSTANT":
                functions.append(self.parse_constant_buffer())
            elif self.current_token[0] in [
                "VERTEX",
                "FRAGMENT",
                "KERNEL",
                "VOID",
                "FLOAT",
                "HALF",
                "INT",
                "UINT",
                "BOOL",
                "VECTOR",
                "IDENTIFIER",
            ]:
                functions.append(self.parse_function())
            else:
                self.eat(self.current_token[0])  # Skip unknown tokens

        return ShaderNode(functions)

    def parse_preprocessor_directive(self):
        self.eat("PREPROCESSOR")
        if self.current_token[0] == "LESS_THAN":
            self.eat("LESS_THAN")
            while self.current_token[0] != "GREATER_THAN":
                self.eat(self.current_token[0])
            self.eat("GREATER_THAN")
        elif self.current_token[0] == "STRING":
            self.eat("STRING")

    def parse_using_statement(self):
        self.eat("USING")
        self.eat("NAMESPACE")
        self.eat("METAL")
        self.eat("SEMICOLON")

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])  # Eat the type
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            attributes = self.parse_attributes()
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name, attributes))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])  # Eat the type
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name))

        self.eat("RBRACE")

        return ConstantBufferNode(name, members)

    def parse_function(self):
        attributes = self.parse_attributes()

        qualifier = None
        if self.current_token[0] in ["VERTEX", "FRAGMENT", "KERNEL"]:
            qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        return_type = self.current_token[1]
        self.eat(self.current_token[0])

        # Handle potential second qualifier after return type
        if self.current_token[0] in ["VERTEX", "FRAGMENT", "KERNEL"]:
            if qualifier is None:
                qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        # Handle possible attribute after parameters
        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)

        body = self.parse_block()

        return FunctionNode(qualifier, return_type, name, params, body, attributes)

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            attributes = self.parse_attributes()
            if self.current_token[0] in [
                "FLOAT",
                "HALF",
                "INT",
                "UINT",
                "BOOL",
                "VECTOR",
                "IDENTIFIER",
                "TEXTURE2D",
                "SAMPLER",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])

                if self.current_token[0] == "LESS_THAN":
                    self.eat("LESS_THAN")
                    if self.current_token[0] in ["FLOAT", "INT", "UINT", "VECTOR"]:
                        vtype += f"<{self.current_token[1]}>"
                        self.eat(self.current_token[0])
                    self.eat("GREATER_THAN")

                if self.current_token[0] == "IDENTIFIER":
                    name = self.current_token[1]
                    self.eat("IDENTIFIER")
                else:
                    name = ""  # Handle case where there's no explicit parameter name

                param_attributes = self.parse_attributes()
                attributes.extend(param_attributes)

                params.append(VariableNode(vtype, name, attributes))
            else:
                # Handle unexpected token
                raise SyntaxError(
                    f"Unexpected token in parameter list: {self.current_token[0]}"
                )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
            elif self.current_token[0] == "RPAREN":
                break
            else:
                # Handle unexpected token
                raise SyntaxError(
                    f"Expected comma or closing parenthesis, got {self.current_token[0]}"
                )
        return params

    def parse_attributes(self):
        attributes = []
        while self.current_token[0] == "ATTRIBUTE":
            attr_content = self.current_token[1][2:-2]  # Remove [[ and ]]
            attr_parts = attr_content.split("(")
            if len(attr_parts) > 1:
                name = attr_parts[0]
                args = attr_parts[1][:-1].split(",")  # Remove closing ) and split args
            else:
                name = attr_content
                args = []
            attributes.append(AttributeNode(name, args))
            self.eat("ATTRIBUTE")
        return attributes

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
            "HALF",
            "INT",
            "UINT",
            "BOOL",
            "VECTOR",
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
            first_token = self.current_token
            self.eat(self.current_token[0])

            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                    return VariableNode(first_token[1], name)
                elif self.current_token[0] == "EQUALS":
                    self.eat("EQUALS")
                    value = self.parse_expression()
                    self.eat("SEMICOLON")
                    return AssignmentNode(VariableNode(first_token[1], name), value)
            elif self.current_token[0] == "EQUALS":
                # This handles cases like "test = float3(1.0, 1.0, 1.0);"
                self.eat("EQUALS")
                value = self.parse_expression()
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode("", first_token[1]), value)
            elif self.current_token[0] == "DOT":
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
                if self.current_token[0] in [
                    "PLUS_EQUALS",
                    "MINUS_EQUALS",
                    "MULTIPLY_EQUALS",
                    "DIVIDE_EQUALS",
                    "EQUAL",
                ]:
                    op = self.current_token[1]
                    self.eat(self.current_token[0])
                    expr = self.parse_expression()
                    self.eat("SEMICOLON")
                    return BinaryOpNode(VariableNode("", first_token[1]), op, expr)
                    # This handles cases like "float3(1.0, 1.0, 1.0);"
                else:
                    expr = self.parse_expression()
                    self.eat("SEMICOLON")
                    return expr
        else:
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

        if self.current_token[0] in ["INT", "UINT"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            init_value = self.parse_expression()
            init = VariableNode(type_name, var_name)
            init = AssignmentNode(init, init_value)
        else:
            init = self.parse_expression()
        self.eat("SEMICOLON")

        condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = self.parse_expression()
        self.eat("RPAREN")

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
        return self.parse_assignment()

    def parse_assignment(self):
        left = self.parse_logical_or()
        if self.current_token[0] in [
            "EQUALS",
            "PLUS_EQUALS",
            "MINUS_EQUALS",
            "MULTIPLY_EQUALS",
            "DIVIDE_EQUALS",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment()
            return AssignmentNode(left, right, op)
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)
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
        if self.current_token[0] in [
            "IDENTIFIER",
            "INT",
            "UINT",
            "FLOAT",
            "HALF",
            "BOOL",
            "VECTOR",
        ]:
            if self.current_token[0] in [
                "INT",
                "UINT",
                "FLOAT",
                "HALF",
                "BOOL",
                "VECTOR",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
                if self.current_token[0] == "IDENTIFIER":
                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    return VariableNode(type_name, var_name)
                elif self.current_token[0] == "LPAREN":
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

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_texture_sample(self):
        texture = self.parse_expression()
        self.eat("DOT")
        self.eat("IDENTIFIER")  # 'sample' method
        self.eat("LPAREN")
        sampler = self.parse_expression()
        self.eat("COMMA")
        coordinates = self.parse_expression()
        self.eat("RPAREN")
        return TextureSampleNode(texture, sampler, coordinates)
