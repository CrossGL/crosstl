from .SlangLexer import *
from .SlangAst import *


class SlangParser:
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
        imports = []
        exports = []
        functions = []
        structs = []
        typedefs = []
        cbuffers = []
        global_variables = []
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "IMPORT":
                imports.append(self.parse_import())
            elif self.current_token[0] == "EXPORT":
                exports.append(self.parse_export())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "CBUFFER":
                cbuffers.append(self.parse_cbuffer())
            elif self.current_token[0] == "TYPEDEF":
                typedefs.append(self.parse_typedef())
            elif self.current_token[0] == "TYPE_SHADER":
                type_shader = self.current_token[1].split('"')[1]
                self.eat("TYPE_SHADER")
                functions.append(self.parse_function(type_shader))
            elif self.current_token[0] in [
                "VOID",
                "FLOAT",
                "FVECTOR",
                "IDENTIFIER",
                "GENERIC",
                "TEXTURE2D",
                "SAMPLER_STATE",
            ]:
                if self.is_function():
                    functions.append(self.parse_function())
                else:
                    global_variables.append(self.parse_global_variable())
            else:
                self.eat(self.current_token[0])  # Skip unknown tokens

        return ShaderNode(
            imports,
            exports,
            structs,
            typedefs,
            functions,
            global_variables,
            cbuffers,
        )

    def is_function(self):
        # Look ahead to check if there's a left parenthesis after the identifier
        current_pos = self.pos
        while self.tokens[current_pos][0] != "EOF":
            if self.tokens[current_pos][0] == "LPAREN":
                return True
            if self.tokens[current_pos][0] == "SEMICOLON":
                return False
            current_pos += 1
        return False

    def parse_cbuffer(self):
        self.eat("CBUFFER")
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
                var_name += f"[{size}]"
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name))
        self.eat("RBRACE")
        return StructNode(name, members)

    def parse_global_variable(self):
        var_type = self.current_token[1]
        self.eat(self.current_token[0])
        var_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return VariableNode(var_type, var_name)

    def parse_import(self):
        self.eat("IMPORT")
        module_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return ImportNode(module_name)

    def parse_export(self):
        self.eat("EXPORT")
        exported_item = (
            self.parse_function()
            if self.current_token[0] in ["VOID", "FLOAT", "FVECTOR", "IDENTIFIER"]
            else self.parse_struct()
        )
        return ExportNode(exported_item)

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
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                semantic = self.current_token[1]
                self.eat(self.current_token[0])
            self.eat("SEMICOLON")
            members.append(VariableNode(vtype, var_name, semantic))
        self.eat("RBRACE")
        return StructNode(name, members)

    def parse_typedef(self):
        self.eat("TYPEDEF")
        original_type = self.current_token[1]
        self.eat(self.current_token[0])
        new_type = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return TypedefNode(original_type, new_type)

    def parse_function(self, shader_type=None):
        is_generic = False
        if self.current_token[0] == "GENERIC":
            is_generic = True
            self.eat("GENERIC")
        return_type = self.current_token[1]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        semantic = None
        if self.current_token[0] == "COLON":
            self.eat("COLON")
            semantic = self.current_token[1]
            self.eat(self.current_token[0])
        body = self.parse_block()
        return FunctionNode(
            return_type, name, params, body, is_generic, shader_type, semantic
        )

    def parse_parameters(self):
        params = []
        while self.current_token[0] != "RPAREN":
            struct_def = " "
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            semantic = None
            if self.current_token[0] == "IDENTIFIER":
                struct_def = self.current_token[1]
                self.eat("IDENTIFIER")
            if self.current_token[0] == "COLON":
                self.eat("COLON")
                semantic = self.current_token[1]
                self.eat(self.current_token[0])
            params.append(VariableNode(vtype + struct_def, name, semantic))
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
            first_token = self.current_token
            self.eat(self.current_token[0])

            if self.current_token[0] == "IDENTIFIER":
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "SEMICOLON":
                    self.eat("SEMICOLON")
                    return VariableNode(first_token[1], name)
                elif self.current_token[0] in [
                    "EQUALS",
                    "PLUS_EQUALS",
                    "MINUS_EQUALS",
                    "MULTIPLY_EQUALS",
                    "DIVIDE_EQUALS",
                ]:
                    self.eat(self.current_token[0])
                    value = self.parse_expression()
                    if self.current_token[0] == "LPAREN":
                        # This handles cases like "float3 test = float3(1.0, 1.0, 1.0);"
                        self.eat("LPAREN")
                        args = []
                        while self.current_token[0] != "RPAREN":
                            args.append(self.parse_expression())
                            if self.current_token[0] == "COMMA":
                                self.eat("COMMA")
                        self.eat("RPAREN")
                        self.eat("SEMICOLON")
                        return AssignmentNode(
                            VariableNode(first_token[1], name),
                            VectorConstructorNode(first_token[1], args),
                        )
                    elif self.current_token[0] == "LBRACKET":
                        self.eat("LBRACKET")
                        index = self.current_token[1]
                        self.eat("IDENTIFIER")
                        self.eat("RBRACKET")
                        if self.current_token[0] == "SEMICOLON":
                            self.eat("SEMICOLON")
                        return AssignmentNode(
                            VariableNode(first_token[1], f"{name}[{index}]"), value
                        )
                    self.eat("SEMICOLON")
                    return AssignmentNode(VariableNode(first_token[1], name), value)
            elif self.current_token[0] in [
                "EQUALS",
                "PLUS_EQUALS",
                "MINUS_EQUALS",
                "MULTIPLY_EQUALS",
                "DIVIDE_EQUALS",
            ]:
                # This handles cases like "test = float3(1.0, 1.0, 1.0);"
                self.eat(self.current_token[0])
                value = self.parse_expression()
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode("", first_token[1]), value)
            elif self.current_token[0] == "DOT":
                left = self.parse_member_access(first_token[1])
                if self.current_token[0] in [
                    "EQUALS",
                    "PLUS_EQUALS",
                    "MINUS_EQUALS",
                    "MULTIPLY_EQUALS",
                    "DIVIDE_EQUALS",
                ]:
                    self.eat(self.current_token[0])
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
        elif self.current_token[0] == "ELSE_IF":
            else_body = self.parse_else_if_statement()
        return IfNode(condition, if_body, else_body)

    def parse_else_if_statement(self):
        self.eat("ELSE_IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_block()
        else_body = None
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block()
        elif self.current_token[0] == "ELSE_IF":
            else_body = self.parse_else_if_statement()
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
            init = VariableNode(type_name, var_name)
            init = AssignmentNode(init, init_value)
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
        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)
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
            "INT",
            "FLOAT",
            "FVECTOR",
            "GENERIC",
        ]:
            if self.current_token[0] in ["INT", "FLOAT", "FVECTOR", "GENERIC"]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
                if self.current_token[0] == "IDENTIFIER":
                    var_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    if self.current_token[0] == "LBRAKET":
                        self.eat("LBRAKET")
                        if self.current_token[0] == "NUMBER":
                            index = self.current_token[1]
                        else:
                            index = self.current_token[1]
                            self.eat("IDENTIFIER")
                        self.eat("RBRAKET")
                        return VariableNode(type_name, f"{var_name}[{index}]")
                    return VariableNode(type_name, var_name)
                elif self.current_token[0] == "LPAREN":
                    return self.parse_vector_constructor(type_name)
            return self.parse_function_call_or_identifier()

        if self.current_token[0] == "LBRAKET":
            self.eat("LBRAKET")
            expr = self.parse_expression()
            self.eat("RBRAKET")
            return expr

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
