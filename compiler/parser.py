from ast import ShaderNode, FunctionNode, VariableNode, AssignmentNode

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
        else:
            raise SyntaxError(f'Expected {token_type}, got {self.current_token[0]}')

    def parse(self):
        return self.parse_shader()

    def parse_shader(self):
        self.eat('SHADER')
        name = self.current_token[1]
        self.eat('IDENTIFIER')
        self.eat('LBRACE')
        inputs = self.parse_inputs()
        outputs = self.parse_outputs()
        main_function = self.parse_function()
        self.eat('RBRACE')
        return ShaderNode(name, inputs, outputs, main_function)

    def parse_inputs(self):
        inputs = []
        while self.current_token[0] == 'INPUT':
            self.eat('INPUT')
            vtype = self.current_token[1]
            self.eat('VECTOR')
            name = self.current_token[1]
            self.eat('IDENTIFIER')
            self.eat('SEMICOLON')
            inputs.append((vtype, name))
        return inputs

    def parse_outputs(self):
        outputs = []
        while self.current_token[0] == 'OUTPUT':
            self.eat('OUTPUT')
            vtype = self.current_token[1]
            self.eat('VECTOR')
            name = self.current_token[1]
            self.eat('IDENTIFIER')
            self.eat('SEMICOLON')
            outputs.append((vtype, name))
        return outputs

    def parse_function(self):
        self.eat('VOID')
        fname = self.current_token[1]
        self.eat('MAIN')
        self.eat('LPAREN')
        self.eat('RPAREN')
        self.eat('LBRACE')
        body = self.parse_body()
        self.eat('RBRACE')
        return FunctionNode(fname, body)

    def parse_body(self):
        body = []
        while self.current_token[0] != 'RBRACE':
            if self.current_token[0] == 'IDENTIFIER':
                body.append(self.parse_assignment())
            else:
                raise SyntaxError(f'Unexpected token {self.current_token[0]}')
        return body

    def parse_assignment(self):
        name = self.current_token[1]
        self.eat('IDENTIFIER')
        self.eat('EQUALS')
        value = self.parse_expression()
        self.eat('SEMICOLON')
        return AssignmentNode(name, value)

    def parse_expression(self):
        if self.current_token[0] == 'IDENTIFIER':
            value = self.current_token[1]
            self.eat('IDENTIFIER')
        elif self.current_token[0] == 'NUMBER':
            value = self.current_token[1]
            self.eat('NUMBER')
        else:
            raise SyntaxError(f'Unexpected token in expression {self.current_token[0]}')
        return value

