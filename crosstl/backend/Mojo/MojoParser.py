from .MojoLexer import *
from .MojoAst import *


class MojoParser:
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
        module = self.parse_module()
        self.eat("EOF")
        return module

    def parse_module(self):
        statements = []

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "IMPORT":
                statements.append(self.parse_import_statement())
            elif self.current_token[0] == "STRUCT":
                statements.append(self.parse_struct())
            elif self.current_token[0] == "CLASS":
                statements.append(self.parse_class())
            elif self.current_token[0] == "CONSTANT":
                statements.append(self.parse_constant_buffer())
            elif self.current_token[0] == "FN":
                statements.append(self.parse_function())
            elif self.current_token[0] in ["LET", "VAR"]:
                statements.append(self.parse_variable_declaration_or_assignment())
            elif self.current_token[0] == "DECORATOR":
                statements.append(self.parse_decorator())
            else:
                self.eat(self.current_token[0])

        return ShaderNode(statements)

    def parse_import_statement(self):
        self.eat("IMPORT")
        module_name = self.current_token[1]
        self.eat("IDENTIFIER")
        alias = None
        if self.current_token[0] == "AS":
            self.eat("AS")
            alias = self.current_token[1]
            self.eat("IDENTIFIER")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return ImportNode(module_name, alias)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("COLON")

        members = []
        while (
            self.current_token[0] != "EOF"
            and self.current_token[0] != "FN"
            and self.current_token[0] != "STRUCT"
            and self.current_token[0] != "CLASS"
        ):
            vtype = None
            if self.current_token[0] == "LET" or self.current_token[0] == "VAR":
                vtype = self.current_token[1]
                self.eat(self.current_token[0])

            var_name = self.current_token[1]
            self.eat("IDENTIFIER")

            # Handle the colon followed by the type
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                vtype = self.current_token[1]
                self.eat(self.current_token[0])

            attributes = self.parse_attributes()

            members.append(VariableNode(vtype, var_name, attributes))

        return StructNode(name, members)

    def parse_class(self):
        self.eat("CLASS")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        base_classes = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                base_classes.append(self.current_token[1])
                self.eat("IDENTIFIER")
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            self.eat("RPAREN")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            if self.current_token[0] == "FN":
                members.append(self.parse_function())
            elif self.current_token[0] in ["LET", "VAR"]:
                members.append(self.parse_variable_declaration_or_assignment())
            elif self.current_token[0] == "CLASS":
                members.append(self.parse_class())
            else:
                self.eat(self.current_token[0])  # Skip unk

        self.eat("RBRACE")
        return ClassNode(name, base_classes, members)

    def parse_constant_buffer(self):
        self.eat("CONSTANT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")

        members = []
        while self.current_token[0] != "RBRACE":
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            var_name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name))

        self.eat("RBRACE")

        return ConstantBufferNode(name, members)

    def parse_function(self):
        attributes = self.parse_attributes()

        qualifier = None
        if self.current_token[0] == "FN":
            qualifier = self.current_token[1]
            self.eat(self.current_token[0])

        return_type = None
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")

        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            if self.current_token[0] == "GREATER_THAN":
                self.eat("GREATER_THAN")
                return_type = self.current_token[1]
                self.eat(self.current_token[0])

        post_attributes = self.parse_attributes()
        attributes.extend(post_attributes)

        body = self.parse_block()

        return FunctionNode(qualifier, return_type, name, params, body, attributes)

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            attributes = self.parse_attributes()

            if self.current_token[0] in ["FLOAT", "INT", "UINT", "BOOL", "IDENTIFIER"]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])

                # Handle the case where there's a colon indicating a type annotation
                if self.current_token[0] == "COLON":
                    self.eat("COLON")
                    vtype = self.current_token[
                        1
                    ]  # Update the type to the one after colon
                    self.eat(self.current_token[0])

                name = ""
                if self.current_token[0] == "IDENTIFIER":
                    name = self.current_token[1]
                    self.eat("IDENTIFIER")

                param_attributes = self.parse_attributes()
                attributes.extend(param_attributes)

                params.append(VariableNode(vtype, name, attributes))

            else:
                raise SyntaxError(
                    f"Unexpected token in parameter list: {self.current_token[0]}"
                )

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
                if self.current_token[0] == "RPAREN":
                    raise SyntaxError("Trailing comma in parameter list is not allowed")
            elif self.current_token[0] == "RPAREN":
                break
            else:
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
                args = attr_parts[1][:-1].split(",")
            else:
                name = attr_content
                args = []
            attributes.append(AttributeNode(name, args))
            self.eat("ATTRIBUTE")
        return attributes

    def parse_block(self):
        if self.current_token[0] == "COLON":
            self.eat("COLON")
        elif self.current_token[0] == "LBRACE":
            self.eat("LBRACE")
        else:
            raise SyntaxError(f"Expected COLON or LBRACE, got {self.current_token[0]}")

        statements = []
        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            statements.append(self.parse_statement())

        if self.current_token[0] == "RBRACE":
            self.eat("RBRACE")

        return statements

    def parse_statement(self):
        if self.current_token[0] in [
            "FLOAT",
            "INT",
            "UINT",
            "BOOL",
            "IDENTIFIER",
            "LET",
            "VAR",
        ]:
            return self.parse_variable_declaration_or_assignment()

        elif self.current_token[0] == "FN":
            return self.parse_function()
        elif self.current_token[0] == "IF":
            return self.parse_if_statement()
        elif self.current_token[0] == "FOR":
            return self.parse_for_statement()
        elif self.current_token[0] == "WHILE":
            return self.parse_while_statement()
        elif self.current_token[0] == "RETURN":
            return self.parse_return_statement()
        elif self.current_token[0] == "SWITCH":
            return self.parse_switch_statement()
        elif self.current_token[0] == "STRUCT":
            return self.parse_struct()
        else:
            return self.parse_expression_statement()

    def parse_variable_declaration_or_assignment(self):
        if self.current_token[0] in ["LET", "VAR"]:
            var_type = self.current_token[0]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "COLON":
                self.eat("COLON")
                self.current_token[1]
                self.eat(self.current_token[0])

            initial_value = None
            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                initial_value = self.parse_expression()

            # Only eat semicolon if it's actually present
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")

            return VariableDeclarationNode(var_type, name, initial_value)
        else:
            return self.parse_assignment()

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

        init = self.parse_variable_declaration_or_assignment()
        self.eat("SEMICOLON")
        condition = self.parse_expression()
        self.eat("SEMICOLON")
        update = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_block()

        return ForNode(init, condition, update, body)

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        body = self.parse_block()
        return WhileNode(condition, body)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expression = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        cases = []
        while self.current_token[0] != "RBRACE":
            cases.append(self.parse_switch_case())
        self.eat("RBRACE")
        return SwitchNode(expression, cases)

    def parse_switch_case(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            condition = self.parse_expression()
            self.eat("COLON")
            body = self.parse_block()
            return SwitchCaseNode(condition, body)
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            self.eat("COLON")
            body = self.parse_block()
            return SwitchCaseNode(None, body)
        else:
            raise SyntaxError(
                f"Unexpected token in switch case: {self.current_token[0]}"
            )

    def parse_return_statement(self):
        self.eat("RETURN")
        value = self.parse_expression()

        if self.current_token[0] == "SEMICOLON":
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
            "ASSIGN_XOR",
            "ASSIGN_OR",
            "ASSIGN_AND",
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
        left = self.parse_bitwise_or()
        while self.current_token[0] == "AND":
            op = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_or(self):
        left = self.parse_bitwise_xor()
        while self.current_token[0] == "BITWISE_OR":
            op = self.current_token[1]
            self.eat("BITWISE_OR")
            right = self.parse_bitwise_xor()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_xor(self):
        left = self.parse_bitwise_and()
        while self.current_token[0] == "BITWISE_XOR":
            op = self.current_token[1]
            self.eat("BITWISE_XOR")
            right = self.parse_bitwise_and()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_bitwise_and(self):
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
        left = self.parse_shift()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_shift()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_shift(self):
        left = self.parse_additive()
        while self.current_token[0] in ["SHIFT_LEFT", "SHIFT_RIGHT"]:
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
        if self.current_token[0] in [
            "IDENTIFIER",
            "NUMBER",
            "INT",
            "FLOAT",
            "BOOL",
            "STRING",
        ]:
            if self.current_token[0] in ["INT", "FLOAT", "BOOL", "STRING"]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
                if self.current_token[0] == "IDENTIFIER":
                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    if self.current_token[0] == "COLON":
                        self.eat("COLON")
                        type_annotation = self.current_token[1]
                        self.eat(self.current_token[0])
                        return VariableNode(type_annotation, var_name)
                    return VariableNode(type_name, var_name)
                elif self.current_token[0] == "LPAREN":
                    return self.parse_function_call_or_identifier()
            elif self.current_token[0] == "NUMBER":
                value = self.current_token[1]
                self.eat("NUMBER")
                return value

            return self.parse_function_call_or_identifier()

        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr

        # Handle top-level keywords
        elif self.current_token[0] in ["FN", "STRUCT", "CLASS", "LET", "VAR"]:
            raise SyntaxError(f"Unexpected top-level keyword: {self.current_token[0]}")

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
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "LPAREN":
                return self.parse_function_call(name)
            elif self.current_token[0] == "DOT":
                return self.parse_member_access(name)
            elif self.current_token[0] == "COLON":
                self.eat("COLON")
                type_annotation = self.current_token[1]
                self.eat(self.current_token[0])
                return VariableNode(type_annotation, name)
            return VariableNode("", name)

        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value

        else:
            raise SyntaxError(
                f"Expected IDENTIFIER or NUMBER, got {self.current_token[0]}"
            )

    def parse_function_call(self, name):
        self.eat("LPAREN")
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")  # Continue parsing the next argument
            elif self.current_token[0] != "RPAREN":
                raise SyntaxError(
                    f"Expected COMMA or RPAREN, got {self.current_token[0]}"
                )
        self.eat("RPAREN")
        return FunctionCallNode(name, args)

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected IDENTIFIER after dot, got {self.current_token[0]}"
            )

        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "LPAREN":
            return self.parse_function_call(member)

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_decorator(self):
        self.eat("DECORATOR")
        name = self.current_token[1]
        args = []
        if self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            while self.current_token[0] != "RPAREN":
                args.append(self.parse_expression())
                if self.current_token[0] == "COMMA":
                    self.eat("COMMA")
            self.eat("RPAREN")
        return DecoratorNode(name, args)
