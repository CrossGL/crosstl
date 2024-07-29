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

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def eat(self, token_type):

        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()  # Skip comments after eating a token
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse_uniforms(self):
        uniforms = []
        while self.current_token[0] == "UNIFORM":
            self.eat("UNIFORM")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "INT",
                "SAMPLER2D",
                "MATRIX",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, INT, or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            uniforms.append(UniformNode(vtype, name))
        return uniforms

    def parse(self):
        return self.parse_shader()

    def parse_shader(self):
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
            elif self.current_token[0] in ["VECTOR", "FLOAT", "INT", "VOID"]:
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
        self.eat(section_type)
        self.eat("LBRACE")
        inputs = self.parse_inputs()
        self.parse_uniforms()
        outputs = self.parse_outputs()

        functions = []
        while self.current_token[0] != "RBRACE":
            functions.append(self.parse_function())
        self.eat("RBRACE")
        if section_type == "VERTEX":
            return VERTEXShaderNode(inputs, outputs, functions)
        else:
            return FRAGMENTShaderNode(inputs, outputs, functions)

    def parse_inputs(self):
        inputs = []
        while self.current_token[0] == "INPUT":
            self.eat("INPUT")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "INT",
                "MATRIX",
                "SAMPLER2D",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, INT, MATRIX, or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            inputs.append((vtype, name))
        return inputs

    def parse_outputs(self):
        outputs = []
        while self.current_token[0] == "OUTPUT":
            self.eat("OUTPUT")
            if self.current_token[0] in [
                "VECTOR",
                "FLOAT",
                "INT",
                "MATRIX",
                "SAMPLER2D",
            ]:
                vtype = self.current_token[1]
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Expected VECTOR, FLOAT, INT, MATRIX, or SAMPLER2D, got {self.current_token[0]}"
                )
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            outputs.append((vtype, name))
        return outputs

    def parse_function(self):
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
        params = []
        if self.current_token[0] != "RPAREN":
            params.append(self.parse_parameter())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                params.append(self.parse_parameter())
        return params

    def parse_parameter(self):
        param_type = self.parse_type()
        param_name = self.current_token[1]

        self.eat("IDENTIFIER")
        return (param_type, param_name)

    def parse_type(self):
        if self.current_token[0] == "VOID":
            self.eat("VOID")
            return "void"
        elif self.current_token[0] in ["VECTOR", "FLOAT", "INT", "MATRIX", "SAMPLER2D"]:
            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            return vtype
        elif self.current_token[0] == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if type_name in ["int", "float"]:
                return type_name
            return type_name
        else:
            raise SyntaxError(f"Expected type, got {self.current_token[0]}")

    def parse_body(self):
        body = []
        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "IF":
                body.append(self.parse_if_statement())
            elif self.current_token[0] == "FOR":
                body.append(self.parse_for_loop())
            elif self.current_token[0] == "RETURN":
                body.append(self.parse_return_statement())
            elif self.current_token[0] in ["VECTOR", "IDENTIFIER", "FLOAT", "INT"]:
                body.append(self.parse_assignment_or_function_call())
            else:
                raise SyntaxError(f"Unexpected token {self.current_token[0]}")
        return body

    def parse_if_statement(self):
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

    def parse_for_loop(self):
        self.eat("FOR")
        self.eat("LPAREN")

        init = self.parse_assignment_or_function_call()

        condition = self.parse_expression()
        self.eat("SEMICOLON")

        update = self.parse_update()

        self.eat("RPAREN")
        self.eat("LBRACE")

        body = self.parse_body()
        self.eat("RBRACE")

        return ForNode(init, condition, update, body)

    def parse_update(self):
        if self.current_token[0] == "IDENTIFIER":
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "INCREMENT":
                self.eat("INCREMENT")
                return AssignmentNode(name, UnaryOpNode("++", VariableNode("", name)))
            elif self.current_token[0] == "DECREMENT":
                self.eat("DECREMENT")
                return AssignmentNode(name, UnaryOpNode("--", VariableNode("", name)))
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
                    name, BinaryOpNode(VariableNode("", name), "+", value)
                )
            elif op == "ASSIGN_SUB":
                return AssignmentNode(
                    name, BinaryOpNode(VariableNode("", name), "-", value)
                )
            elif op == "ASSIGN_MUL":
                return AssignmentNode(
                    name, BinaryOpNode(VariableNode("", name), "*", value)
                )
            elif op == "ASSIGN_DIV":
                return AssignmentNode(
                    name, BinaryOpNode(VariableNode("", name), "/", value)
                )
            else:
                raise SyntaxError(
                    f"Expected INCREMENT or DECREMENT, got {self.current_token[0]}"
                )
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_return_statement(self):
        self.eat("RETURN")
        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_assignment_or_function_call(self):
        type_name = ""
        if self.current_token[0] in ["VECTOR", "FLOAT", "INT", "MATRIX"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable_declaration(type_name)

        name = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] in [
            "EQUALS",
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
        ]:
            return self.parse_assignment(name)
        elif self.current_token[0] == "INCREMENT":
            self.eat("INCREMENT")
            return AssignmentNode(name, UnaryOpNode("++", VariableNode("", name)))
        elif self.current_token[0] == "DECREMENT":
            self.eat("DECREMENT")
            return AssignmentNode(name, UnaryOpNode("--", VariableNode("", name)))
        elif self.current_token[0] == "LPAREN":
            return self.parse_function_call(name)
        else:
            raise SyntaxError(
                f"Unexpected token after identifier: {self.current_token[0]}"
            )

    def parse_variable_declaration(self, type_name):
        name = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return VariableNode(type_name, name)

        elif self.current_token[0] == "EQUALS":
            self.eat("EQUALS")
            value = self.parse_expression()

            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode(type_name, name), value)

            # if self.current_token[0] == "RPAREN":
            #    self.eat("RPAREN")

            else:
                raise SyntaxError(
                    f"Expected ';' after variable assignment, found: {self.current_token[0]}"
                )

        elif self.current_token[0] in (
            "ASSIGN_ADD",
            "ASSIGN_SUB",
            "ASSIGN_MUL",
            "ASSIGN_DIV",
        ):
            op = self.current_token[0]
            self.eat(op)
            value = self.parse_expression()
            if self.current_token[0] == "SEMICOLON":
                self.eat("SEMICOLON")
                return AssignmentNode(VariableNode(type_name, name), value)
            else:
                raise SyntaxError(
                    f"Expected ';' after compound assignment, found: {self.current_token[0]}"
                )

        else:
            raise SyntaxError(
                f"Unexpected token in variable declaration: {self.current_token[0]}"
            )

    def parse_assignment(self, name):
        self.eat("EQUALS")
        value = self.parse_expression()
        if self.current_token[0] != "SEMICOLON":
            raise SyntaxError(
                f"Expected ';' after assignment, found: {self.current_token[0]}"
            )
        self.eat("SEMICOLON")
        return AssignmentNode(name, value)

    # def parse_expression(self):
    #     return self.parse_additive()

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_multiplicative(self):
        left = self.parse_primary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_primary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()  # Handle the negation as a unary operation
            return UnaryOpNode("-", value)

        if self.current_token[0] in ("IDENTIFIER", "VECTOR", "FLOAT"):
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

    def parse_function_call(self, name):
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
        left = self.parse_additive()
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
            self.eat(op)
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_function_call_or_identifier(self):
        if self.current_token[0] == "VECTOR":
            func_name = self.current_token[1]
            self.eat("VECTOR")
        else:
            func_name = self.current_token[1]
            self.eat("IDENTIFIER")

        if self.current_token[0] == "LPAREN":
            return self.parse_function_call(func_name)
        elif self.current_token[0] == "DOT":
            return self.parse_member_access(func_name)
        return VariableNode("", func_name)

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


# Usage example
if __name__ == "__main__":
    code = """shader main {
        
                input vec3 position;
                            input vec2 texCoord;
                            input mat2 depth;
                            output vec4 fragColor;
                            output float depth;
                            vec3 customFunction(vec3 random, float factor) {
                                return random * factor;
                            }

                            void main() {
                                vec3 color = vec3(position.x,position.y, 0.0);
                                float factor = 1.0;

                                if (texCoord.x > 0.5) {
                                    color = vec3(1.0, 0.0, 0.0);
                                } else {
                                    color = vec3(0.0, 1.0, 0.0);
                                }

                                for (int i = 0; i < 3; i = i + 1) {
                                    factor = factor * 0.5;
                                    color = customFunction(color, factor);
                                }

                                if (length(color) > 1.0) {
                                    color = normalize(color);
                                }

                                fragColor = vec4(color, 1.0);
                            }
        
                vertex {
                            input vec3 position;
                            input vec2 texCoord;
                            input mat2 depth;
                            output vec4 fragColor;
                            output float depth;
                            vec3 customFunction(vec3 random, float factor) {
                                return random * factor;
                            }

                            void main() {
                                vec3 color = vec3(position.x,position.y, 0.0);
                                float factor = 1.0;

                                if (texCoord.x > 0.5) {
                                    color = vec3(1.0, 0.0, 0.0);
                                } else {
                                    color = vec3(0.0, 1.0, 0.0);
                                }

                                for (int i = 0; i < 3; i = i + 1) {
                                    factor = factor * 0.5;
                                    color = customFunction(color, factor);
                                }

                                if (length(color) > 1.0) {
                                    color = normalize(color);
                                }

                                fragColor = vec4(color, 1.0);
                            }
                            }
                            fragment {
                                input vec4 fragColor;
                                output vec4 finalColor;
                                
                                void main() {
                                    finalColor = fragColor;
                                }
                                
                            }
                        }"""
    lexer = Lexer(code)
    print("Tokens:")
    parser = Parser(lexer.tokens)
    print(parser.parse())
