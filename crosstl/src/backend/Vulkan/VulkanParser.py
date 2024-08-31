from VulkanLexer import *
from VulkanAst import *


class VulkanParser:
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
            if self.current_token[0] == "LAYOUT":
                statements.append(self.parse_layout())
            elif self.current_token[0] == "STRUCT":
                statements.append(self.parse_struct())
            elif self.current_token[0] == "UNIFORM":
                statements.append(self.parse_uniform())
            elif self.current_token[0] == "IDENTIFIER" and self.peek(1) == "LPAREN":
                statements.append(self.parse_function())
            elif self.current_token[0] in ["IDENTIFIER"]:
                statements.append(self.parse_variable_declaration_or_assignment())
            else:
                self.eat(self.current_token[0])
        return ShaderNode(None, None, None, statements)

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []
        while self.current_token[0] != "RPAREN":
            bindings.append(self.current_token[1])
            self.eat("IDENTIFIER")
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        self.eat("IDENTIFIER") 
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("SEMICOLON")
        return LayoutNode(bindings, name)

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable_declaration_or_assignment())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable_declaration_or_assignment())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable_declaration_or_assignment())
        self.eat("RBRACE")
        return StructNode(name, members)

    def parse_function(self):
        return_type = self.current_token[1]
        self.eat("IDENTIFIER")
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
            statements.append(self.parse_statement())
        self.eat("RBRACE")
        return statements

    def parse_statement(self):
        token_type = self.current_token[0]

        if token_type == "IDENTIFIER":
            if self.peek(1)[0] == "LPAREN":
                return self.parse_function()
            else:
                return self.parse_variable_declaration_or_assignment()
        elif token_type in ["IF", "FOR"]:
            if token_type == "IF":
                return self.parse_if_statement()
            elif token_type == "FOR":
                return self.parse_for_statement()
        else:
            return self.parse_expression_statement()
        
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
        initialization = self.parse_expression_statement()  
        condition = self.parse_expression()
        self.eat("SEMICOLON")  
        increment = self.parse_expression()
        self.eat("RPAREN") 
        body = self.parse_block()
        return ForNode(initialization, condition, increment, body)

    def parse_variable_declaration_or_assignment(self):
        var_type = self.current_token[0]
        self.eat(self.current_token[0])
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        initial_value = None
        if self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            initial_value = self.parse_expression()
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
        return VariableDeclarationNode(var_type, name, initial_value)

    def parse_expression(self):
        return self.parse_primary()

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_primary(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            return VariableNode("", name)
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            return value
        else:
            raise SyntaxError(f"Unexpected token: {self.current_token[0]}")