from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *


class HLSLToCrossGLConverter:
    def __init__(self):
        self.vertex_inputs = []
        self.vertex_outputs = []
        self.fragment_inputs = []
        self.fragment_outputs = []
        self.type_map = {
            "float": "float",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "int": "int",
        }

    def convert(self, ast):
        self.process_structs(ast)

        code = "shader main {\n"

        # Generate vertex shader
        code += "    // Vertex Shader\n"
        code += "    vertex {\n"
        code += self.generate_io_declarations("vertex")
        code += "\n"
        code += self.generate_vertex_main(
            next(f for f in ast.functions if f.name == "VSMain")
        )
        code += "    }\n\n"

        # Generate custom functions
        for func in ast.functions:
            if func.name not in ["VSMain", "PSMain"]:
                code += self.generate_function(func)

        # Generate fragment shader
        code += "    // Fragment Shader\n"
        code += "    fragment {\n"
        code += self.generate_io_declarations("fragment")
        code += "\n"
        code += self.generate_fragment_main(
            next(f for f in ast.functions if f.name == "PSMain")
        )
        code += "    }\n"

        code += "}\n"
        return code

    def process_structs(self, ast):
        if ast.input_struct and ast.input_struct.name == "Vertex_INPUT":
            for member in ast.input_struct.members:
                self.vertex_inputs.append((member.vtype, member.name))
        if ast.output_struct and ast.output_struct.name == "Vertex_OUTPUT":
            for member in ast.output_struct.members:
                if member.name != "position":
                    self.vertex_outputs.append((member.vtype, member.name))
                    self.fragment_inputs.append((member.vtype, member.name))
        if ast.output_struct and ast.output_struct.name == "Fragment_OUTPUT":
            for member in ast.output_struct.members:
                self.fragment_outputs.append((member.vtype, member.name))

    def generate_io_declarations(self, shader_type):
        code = ""
        if shader_type == "vertex":
            for type, name in self.vertex_inputs:
                code += f"        input {self.map_type(type)} {name};\n"
            for type, name in self.vertex_outputs:
                code += f"        output {self.map_type(type)} {name};\n"
        elif shader_type == "fragment":
            for type, name in self.fragment_inputs:
                code += f"        input {self.map_type(type)} {name};\n"
            for type, name in self.fragment_outputs:
                code += f"        output {self.map_type(type)} {name};\n"
        return code.rstrip()

    def generate_function(self, func):
        params = ", ".join(f"{self.map_type(p.vtype)} {p.name}" for p in func.params)
        code = f"    {self.map_type(func.return_type)} {func.name}({params}) {{\n"
        code += self.generate_function_body(func.body, indent=2)
        code += "    }\n\n"
        return code

    def generate_vertex_main(self, func):
        code = "        void main() {\n"
        code += self.generate_function_body(func.body, indent=3, is_main=True)
        code += "        }\n"
        return code

    def generate_fragment_main(self, func):
        code = "        void main() {\n"
        code += self.generate_function_body(func.body, indent=3, is_main=True)
        code += "        }\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                if not is_main:
                    code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    code += f"return {self.generate_expression(stmt.value, is_main)};\n"
        return code

    def generate_assignment(self, node, is_main):
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        if (
            is_main
            and isinstance(node.left, MemberAccessNode)
            and node.left.object == "output"
        ):
            if node.left.member == "position":
                return f"gl_Position = {rhs}"
            return f"{node.left.member} = {rhs}"
        return f"{lhs} = {rhs}"

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            return f"{expr.vtype} {expr.name}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"({left} {expr.op} {right})"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"({expr.operator}{operand})"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            if obj == "output" or obj == "input":
                return expr.member
            return f"{obj}.{expr.member}"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        else:
            return str(expr)

    def map_type(self, hlsl_type):
        return self.type_map.get(hlsl_type, hlsl_type)
