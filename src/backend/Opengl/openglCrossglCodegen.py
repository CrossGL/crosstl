from .OpenglAst import *
from .OpenglParser import *
from .OpenglLexer import *


class CrossglCodeGen:
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
        # Set up the shader
        self.shader_inputs = node.global_inputs
        self.shader_outputs = node.global_outputs
        self.uniforms = node.uniforms

        code = "shader main {\n"

        # Generate vertex shader section
        self.vertex_item = node.vertex_section
        if self.vertex_item:
            code += "    vertex {\n"
            # Print the vertex section to check its content
            code += self.generate_layouts(self.vertex_item.layout_qualifiers)
            # Process inputs and outputs
            for vtype, name in self.shader_inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.shader_outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
            for vtype, name in self.vertex_item.inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.vertex_item.outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
                code += "\n"
            code += self.generate_uniforms() + "\n"
            code += "\n"

            # Print the functions to check if they're there
            # print(f"Vertex functions: {self.vertex_item.functions}")

            # Generate functions
            code += self.generate_functions(self.vertex_item.functions, "vertex")
            code += "    }\n"
        else:
            print("No vertex shader section to generate.")

        # Generate fragment shader section if present
        self.fragment_item = node.fragment_section
        if self.fragment_item and (
            self.fragment_item.layout_qualifiers or self.fragment_item.functions
        ):
            code += "    fragment {\n"
            code += self.generate_layouts(self.fragment_item.layout_qualifiers)
            # Process inputs and outputs
            for vtype, name in self.fragment_item.inputs:
                code += f"        input {self.map_type(vtype)} {name};\n"
            for vtype, name in self.fragment_item.outputs:
                code += f"        output {self.map_type(vtype)} {name};\n"
                code += "\n"
            code += self.generate_uniforms() + "\n"
            code += "\n"

            # Print the functions to check if they're there
            # print(f"Fragment functions: {self.fragment_item.functions}")

            # Generate functions
            code += self.generate_functions(self.fragment_item.functions, "fragment")
            code += "    }\n"
        else:
            print("No fragment shader section to generate.")

        code += "}\n"

        return code

    def generate_uniforms(self):
        uniform_lines = []
        for uniform in self.uniforms:
            uniform_lines.append(f"        {uniform};")
        return "\n".join(uniform_lines)

    def generate_layouts(self, layouts):
        code = ""
        for layout in layouts:
            code += f"        input {layout.dtype} {layout.name};\n"
        return code

    def generate_functions(self, functions, shader_type):
        code = ""

        if shader_type in ["vertex", "fragment"]:
            for function_node in functions:
                # Generate parameter list
                params = ", ".join(
                    f"{self.map_type(param[0])} {param[1]}"
                    for param in function_node.params
                )
                # Generate function header
                code += f"        {self.map_type(function_node.return_type)} {function_node.name}({params}) {{\n"
                # Generate function body
                for stmt in function_node.body:
                    code += self.generate_statement(stmt, 2)
                # Close function definition
                code += "        }\n"

        return code

    def generate_statement(
        self,
        stmt,
        indent=0,
    ):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
            return f"{indent_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            return f"{indent_str}return {self.generate_expression(stmt.value)};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_assignment(self, node):
        lhs = self.generate_expression(node.name)
        rhs = self.generate_expression(node.value)
        return f"{lhs} = {rhs}"

    def generate_if(self, node: IfNode, indent):
        indent_str = "    " * indent
        code = f"{indent_str}if ({self.generate_expression(node.condition)}) {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}"
        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node: ForNode, indent):
        indent_str = "    " * indent

        init = self.generate_statement(node.init, 0).strip()
        condition = self.generate_expression(node.condition)
        update = self.generate_statement(node.update, 0).strip()

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return self.translate_expression(expr)
        elif isinstance(expr, VariableNode):
            return self.translate_expression(expr.name)
        elif isinstance(expr, BinaryOpNode):
            return f"({self.generate_expression(expr.left)} {self.map_operator(expr.op)} {self.generate_expression(expr.right)})"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            func_name = self.translate_expression(expr.name)
            return f"{func_name}({args})"

        elif isinstance(expr, MemberAccessNode):
            return f"{self.generate_expression(expr.object)}.{expr.member}"
        else:
            return str(expr)

    def translate_expression(self, expr):
        return expr

    def map_type(self, vtype):
        type_map = {
            "vec3": "vec3",
            "vec4": "vec4",
            "float": "float",
            "int": "int",
            "bool": "bool",
        }
        return type_map.get(vtype, vtype)

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
        }
        return op_map.get(op, op)
