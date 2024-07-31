from .OpenglAst import (
    LayoutNode,
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
    VersionDirectiveNode,
)
from .OpenglLexer import Lexer


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_MULTI"]:
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

    def parse(self):
        self.skip_comments()
        version_node = self.parse_version_directive()  # Handle version directive
        shader_node = self.parse_shader(version_node)
        return shader_node

    def parse_version_directive(self):
        if self.current_token[0] == "VERSION":
            self.eat("VERSION")

            if self.current_token[0] == "NUMBER":
                number = self.current_token[1]
                self.eat("NUMBER")

                version_identifier = None
                # Handle any additional tokens after NUMBER (like 'core')
                if self.current_token[0] == "CORE":
                    version_identifier = self.current_token[1]
                    self.eat("CORE")
                return VersionDirectiveNode(number, version_identifier)
            else:
                raise SyntaxError(
                    f"Expected NUMBER after VERSION, got {self.current_token[0]}"
                )
        else:
            raise SyntaxError(
                f"Expected VERSION directive, got {self.current_token[0]}"
            )

    def parse_layout(self, current_section):
        self.eat("LAYOUT")
        self.eat("LPAREN")

        if (
            self.current_token[0] == "IDENTIFIER"
            and self.current_token[1] == "location"
        ):
            self.eat("IDENTIFIER")
            self.eat("EQUALS")
            location_number = self.current_token[1]
            self.eat("NUMBER")

            self.eat("RPAREN")
            self.skip_comments()

            if self.current_token[0] == "IN":
                self.eat("IN")
                dtype = self.parse_type()
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                self.eat("SEMICOLON")
                return LayoutNode(
                    section=current_section,
                    location_number=location_number,
                    dtype=dtype,
                    name=name,
                )
            elif self.current_token[0] == "OUT":
                self.eat("OUT")
                dtype = self.parse_type()
                name = self.current_token[1]
                self.eat("IDENTIFIER")
                self.eat("SEMICOLON")
                return LayoutNode(
                    section=current_section,
                    location_number=location_number,
                    dtype=dtype,
                    name=name,
                )
            else:
                raise SyntaxError("Expected 'IN' or 'OUT' after location in LAYOUT")
        else:
            raise SyntaxError("Expected IDENTIFIER 'location' in LAYOUT")

    def parse_shader(self, version_node):
        global_inputs = []
        global_outputs = []
        uniforms = []
        vertex_section = VERTEXShaderNode([], [], [], [], [])
        fragment_section = FRAGMENTShaderNode([], [], [], [], [])
        current_section = None

        while self.current_token[0] != "EOF":
            if self.current_token[0] == "COMMENT_SINGLE":
                comment_content = (
                    self.current_token[1].strip().lower()
                )  # Normalize content
                if "vertex shader" in comment_content:
                    current_section = "VERTEX"
                elif "fragment shader" in comment_content:
                    current_section = "FRAGMENT"
                else:
                    current_section = "VERTEX"

                self.eat("COMMENT_SINGLE")

            if self.current_token[0] == "LAYOUT":
                self.skip_comments()
                layout_node = self.parse_layout(current_section)
                if current_section == "VERTEX":
                    vertex_section.layout_qualifiers.append(layout_node)
                elif current_section == "FRAGMENT":
                    fragment_section.layout_qualifiers.append(layout_node)

            elif self.current_token[0] == "IN":
                self.skip_comments()
                inputs = self.parse_inputs()
                if current_section == "VERTEX":
                    vertex_section.inputs.extend(inputs)
                elif current_section == "FRAGMENT":
                    fragment_section.inputs.extend(inputs)
                else:
                    global_inputs.extend(inputs)

            elif self.current_token[0] == "OUT":
                self.skip_comments()
                outputs = self.parse_outputs()
                if current_section == "VERTEX":
                    vertex_section.outputs.extend(outputs)
                elif current_section == "FRAGMENT":
                    fragment_section.outputs.extend(outputs)
                else:
                    global_outputs.extend(outputs)

            elif self.current_token[0] == "UNIFORM":
                self.skip_comments()
                uniforms.extend(self.parse_uniforms())

            elif self.current_token[0] == "VERSION":
                self.parse_version_directive()

            elif self.current_token[0] in ["VOID", "FLOAT", "VECTOR"]:
                self.skip_comments()
                if current_section:
                    function_node = self.parse_function()
                    if current_section == "VERTEX":
                        vertex_section.functions.append(function_node)
                    elif current_section == "FRAGMENT":
                        fragment_section.functions.append(function_node)
                else:
                    raise SyntaxError("Function found outside of shader section")
            elif self.current_token[0] == "LBRACE":
                if current_section is None:
                    raise SyntaxError("LBRACE encountered outside of shader section")
                self.eat("LBRACE")
                section_content = self.parse_shader_section()
                if current_section == "VERTEX":
                    vertex_section.inputs.extend(section_content[0])
                    vertex_section.outputs.extend(section_content[1])
                    vertex_section.uniforms.extend(section_content[2])
                    vertex_section.layout_qualifiers.extend(section_content[3])
                    vertex_section.functions.extend(section_content[4])
                elif current_section == "FRAGMENT":
                    fragment_section.inputs.extend(section_content[0])
                    fragment_section.outputs.extend(section_content[1])
                    fragment_section.uniforms.extend(section_content[2])
                    fragment_section.layout_qualifiers.extend(section_content[3])
                    fragment_section.functions.extend(section_content[4])
                self.eat("RBRACE")
                current_section = None
            else:
                raise SyntaxError(f"Unexpected token {self.current_token[0]}")

        # print(f"Final vertex section: {vertex_section}")
        # print(f"Final fragment section: {fragment_section}")

        return ShaderNode(
            version=version_node,
            global_inputs=global_inputs,
            global_outputs=global_outputs,
            uniforms=uniforms,
            vertex_section=vertex_section,
            fragment_section=fragment_section,
            functions=[],
        )

    def parse_shader_section(self, current_section):
        inputs = []
        outputs = []
        uniforms = []
        functions = []
        layout_qualifiers = []

        self.eat("LBRACE")

        while self.current_token[0] != "RBRACE" and self.current_token[0] != "EOF":
            if self.current_token[0] == "LAYOUT":
                self.skip_comments()
                layout_node = self.parse_layout(current_section)
                layout_qualifiers.append(layout_node)

            elif self.current_token[0] == "IN":
                self.skip_comments()
                inputs.extend(self.parse_inputs())
                # print(f"Inputs collected: {inputs}")

            elif self.current_token[0] == "OUT":
                self.skip_comments()
                outputs.extend(self.parse_outputs())
                # print(f"Outputs collected: {outputs}")

            elif self.current_token[0] == "UNIFORM":
                self.skip_comments()
                uniforms.extend(self.parse_uniforms())
                # print(f"Uniforms collected: {uniforms}")

            elif self.current_token[0] in ["VOID", "FLOAT", "VECTOR"]:
                self.skip_comments()
                functions.append(self.parse_function())
                # print(f"Functions collected: {functions}")

            elif self.current_token[0] == "RBRACE":
                self.eat("RBRACE")
                return (inputs, outputs, uniforms, layout_qualifiers, functions)

            else:
                raise SyntaxError(
                    f"Unexpected token {self.current_token[0]} in shader section"
                )

        raise SyntaxError("Unexpected end of input in shader section")

    def parse_inputs(self):
        inputs = []
        while self.current_token[0] == "IN":
            self.eat("IN")
            vtype = self.parse_type()
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            inputs.append((vtype, name))
        return inputs

    def parse_outputs(self):
        outputs = []
        while self.current_token[0] == "OUT":
            self.eat("OUT")
            vtype = self.parse_type()
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            outputs.append((vtype, name))
        return outputs

    def parse_uniforms(self):
        uniforms = []
        while self.current_token[0] == "UNIFORM":
            self.eat("UNIFORM")
            vtype = self.parse_type()
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            self.eat("SEMICOLON")
            uniforms.append(UniformNode(vtype, name))
        return uniforms

    def parse_variable(self, type_name):
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

    def parse_assignment_or_function_call(self):
        type_name = ""
        if self.current_token[0] in ["VECTOR", "FLOAT", "INT", "MATRIX"]:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable(type_name)

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

    def parse_body(self):
        body = []
        while self.current_token[0] not in ["RBRACE", "EOF"]:
            if self.current_token[0] == "IF":
                body.append(self.parse_if())
            elif self.current_token[0] == "FOR":
                body.append(self.parse_for())
            elif self.current_token[0] == "RETURN":
                body.append(self.parse_return())
            elif self.current_token[0] in ["VECTOR", "IDENTIFIER", "FLOAT", "INT"]:
                body.append(self.parse_assignment_or_function_call())
            else:
                raise SyntaxError(f"Unexpected token {self.current_token[0]}")
        return body

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
        elif self.current_token[0] in [
            "VECTOR",
            "FLOAT",
            "INT",
            "MATRIX",
            "BOOLEAN",
            "SAMPLER2D",
        ]:
            dtype = self.current_token[1]
            self.eat(self.current_token[0])
            return dtype
        elif self.current_token[0] == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
            if type_name in ["int", "float"]:
                return type_name
            return type_name
        else:
            raise SyntaxError(f"Unexpected type token: {self.current_token[0]}")

    def parse_arguments(self):
        args = []
        while self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return args

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

    def parse_assignment(self):
        var_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("EQUALS")
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return AssignmentNode(var_name, expr)

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

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_multiplicative()
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

    def parse_multiplicative(self):
        left = self.parse_primary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE"]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_primary()
            left = BinaryOpNode(left, op, right)
        return left

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

    def parse_return(self):
        self.eat("RETURN")
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(expr)

    def parse_if(self):
        self.eat("IF")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = self.parse_body()
        self.eat("RBRACE")
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            self.eat("LBRACE")
            else_body = self.parse_body()
            self.eat("RBRACE")
            return IfNode(condition, body, else_body)
        else:
            return IfNode(condition, body)

    def parse_for(self):
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
