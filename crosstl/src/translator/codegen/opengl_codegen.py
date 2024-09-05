from ..ast import (
    ShaderNode,
    AssignmentNode,
    FunctionNode,
    ReturnNode,
    BinaryOpNode,
    UnaryOpNode,
    IfNode,
    ForNode,
    VariableNode,
    FunctionCallNode,
    MemberAccessNode,
    VERTEXShaderNode,
    FRAGMENTShaderNode,
    TernaryOpNode,
)


class GLSLCodeGen:
    def __init__(self):
        self.current_shader = None
        self.vertex_item = None
        self.fragment_item = None

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            self.current_shader = ast
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        self.shader_inputs = node.global_inputs
        self.shader_outputs = node.global_outputs
        code = "#version 450\n\n"

        # Generate global inputs and outputs
        for i, (vtype, name) in enumerate(self.shader_inputs):
            code += f"layout(location = {i}) in {self.map_type(vtype)} {name};\n"
        for i, (vtype, name) in enumerate(self.shader_outputs):
            code += f"layout(location = {i}) out {self.map_type(vtype)} {name};\n"
        code += "\n"

        # Generate functions
        code += "// Vertex shader\n\n"
        for function in node.global_functions:
            code += self.generate_function(function, "global") + "\n"

        # Generate vertex shader section
        self.vertex_item = node.vertex_section
        if isinstance(self.vertex_item, VERTEXShaderNode):
            shader_type = "vertex"
            if self.vertex_item.inputs:
                for i, (vtype, name) in enumerate(self.vertex_item.inputs):
                    code += (
                        f"layout(location = {i}) in {self.map_type(vtype)} {name};\n"
                    )

            if self.vertex_item.outputs:
                for i, (vtype, name) in enumerate(self.vertex_item.outputs):
                    if i == 0:
                        code += f"out {self.map_type(vtype)} {name};\n"
                    else:
                        code += f"layout(location = {i-1}) out {self.map_type(vtype)} {name};\n"
                code += "\n"

            if self.vertex_item.functions:
                code += f"{self.generate_function(self.vertex_item.functions, shader_type)}\n"

            if self.vertex_item.intermidiate:
                code += f"{self.generate_intermidiate(self.vertex_item.intermidiate, shader_type)}\n"

            if self.vertex_item.functions:
                for function_node in self.vertex_item.functions:
                    if function_node.name == "main":
                        code += f"{self.generate_main(function_node, shader_type)}\n"

        # Generate fragment shader section
        self.fragment_item = node.fragment_section
        if isinstance(self.fragment_item, FRAGMENTShaderNode):
            code += "\n// Fragment shader\n\n"
            shader_type = "fragment"
            if self.fragment_item.inputs:
                for i, (vtype, name) in enumerate(self.fragment_item.inputs):
                    if i == 0:
                        code += f"in {self.map_type(vtype)} {name};\n"
                    else:
                        code += f"layout(location = {i-1}) in {self.map_type(vtype)} {name};\n"
            if self.fragment_item.outputs:
                for i, (vtype, name) in enumerate(self.fragment_item.outputs):
                    code += (
                        f"layout(location = {i}) out {self.map_type(vtype)} {name};\n"
                    )
                code += "\n"

            if self.fragment_item.functions:
                code += f"{self.generate_function(self.fragment_item.functions, shader_type)}\n"
            if self.fragment_item.intermidiate:
                code += f"{self.generate_intermidiate(self.fragment_item.intermidiate, shader_type)}\n"
            if self.fragment_item.functions:
                for function_node in self.fragment_item.functions:
                    if function_node.name == "main":
                        code += f"{self.generate_main(function_node, shader_type)}\n"

        return code

    def generate_function(self, node, shader_type):
        code = ""
        if shader_type in ["vertex", "fragment"]:
            for function_node in node:
                if function_node.name != "main":
                    params = ", ".join(
                        f"{self.map_type(param[0])} {param[1]}"
                        for param in function_node.params
                    )
                    code += f"{self.map_type(function_node.return_type)} {function_node.name}({params}) {{\n"
                    for stmt in function_node.body:
                        code += self.generate_statement(
                            stmt, 1, shader_type=shader_type
                        )
                    code += "}\n"
        elif shader_type == "global":
            params = ", ".join(
                f"{self.map_type(param[0])} {param[1]}" for param in node.params
            )
            code = f"{self.map_type(node.return_type)} {node.name}({params}) {{\n"
            for stmt in node.body:
                code += self.generate_statement(stmt, 1, shader_type=shader_type)
            code += "}\n"
        return code

    def generate_main(self, node, shader_type):
        code = ""
        params = ", ".join(
            f"{self.map_type(param[0])} {param[1]}" for param in node.params
        )
        code += f"{self.map_type(node.return_type)} {node.name}({params}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, 1, shader_type=shader_type)
        code += "}\n"
        return code

    def generate_statement(self, stmt, indent=0, shader_type=None):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
            return f"{indent_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt, shader_type)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent, shader_type)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent, shader_type)
        elif isinstance(stmt, ReturnNode):
            code = ""
            for i, return_stmt in enumerate(stmt.value):
                code += f"{self.generate_expression(return_stmt, shader_type)}"
                if i < len(stmt.value) - 1:
                    code += ", "
            return f"{indent_str}return {code};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt, shader_type)};\n"

    def generate_intermidiate(self, node, shader_type):
        code = ""
        for stmt in node:
            code += self.generate_statement(stmt, 0, shader_type=shader_type)
        return code

    def generate_assignment(self, node, shader_type=None):
        if isinstance(node.name, VariableNode) and node.name.vtype:
            return f"{self.map_type(node.name.vtype)} {node.name.name} = {self.generate_expression(node.value, shader_type)}"
        else:
            lhs = self.generate_expression(node.name, shader_type)
            return f"{lhs} = {self.generate_expression(node.value, shader_type)}"

    def generate_if(self, node, indent, shader_type=None):
        indent_str = "    " * indent
        code = f"{indent_str}if ({self.generate_expression(node.if_condition, shader_type)}) {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1, shader_type)
        code += f"{indent_str}}}"

        for else_if_condition, else_if_body in zip(
            node.else_if_conditions, node.else_if_bodies
        ):
            code += f" else if ({self.generate_expression(else_if_condition, shader_type)}) {{\n"
            for stmt in else_if_body:
                code += self.generate_statement(stmt, indent + 1, shader_type)
            code += f"{indent_str}}}"

        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1, shader_type)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node, indent, shader_type=None):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0, shader_type).strip()[
            :-1
        ]  # Remove trailing semicolon

        condition = self.generate_statement(node.condition, 0, shader_type).strip()[
            :-1
        ]  # Remove trailing semicolon

        update = self.generate_statement(node.update, 0, shader_type).strip()[:-1]

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1, shader_type)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr, shader_type=None):
        if isinstance(expr, str):
            return self.translate_expression(expr, shader_type)
        elif isinstance(expr, VariableNode):
            if isinstance(expr.name, str):
                return f"{self.map_type(expr.vtype)} {self.translate_expression(expr.name, shader_type)}"
            else:
                return f"{self.map_type(expr.vtype)} {self.generate_expression(expr.name, shader_type)}"
        elif isinstance(expr, BinaryOpNode):
            return f"{self.generate_expression(expr.left, shader_type)} {self.map_operator(expr.op)} {self.generate_expression(expr.right, shader_type)}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, shader_type) for arg in expr.args
            )
            func_name = self.translate_expression(expr.name, shader_type)
            return f"{func_name}({args})"
        elif isinstance(expr, UnaryOpNode):
            return f"{self.map_operator(expr.op)}{self.generate_expression(expr.operand, shader_type)}"

        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, shader_type)} ? {self.generate_expression(expr.true_expr, shader_type)} : {self.generate_expression(expr.false_expr, shader_type)}"
        elif isinstance(expr, MemberAccessNode):
            return f"{self.generate_expression(expr.object, shader_type)}.{expr.member}"
        else:
            return str(expr)

    def translate_expression(self, expr, shader_type):
        if shader_type == "vertex":
            if self.vertex_item and self.vertex_item.inputs:
                for _, input_name in self.vertex_item.inputs:
                    if expr == input_name:
                        return input_name
            if self.vertex_item and self.vertex_item.outputs:
                for _, output_name in self.vertex_item.outputs:
                    if expr == output_name:
                        return output_name
        elif shader_type == "fragment":
            if self.fragment_item and self.fragment_item.inputs:
                for _, input_name in self.fragment_item.inputs:
                    if expr == input_name:
                        return input_name
            if self.fragment_item and self.fragment_item.outputs:
                for _, output_name in self.fragment_item.outputs:
                    if expr == output_name:
                        return output_name

        return expr

    def map_type(self, vtype):
        return vtype

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "LESS_THAN": "<",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "GREATER_THAN": ">",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
        }
        return op_map.get(op, op)
