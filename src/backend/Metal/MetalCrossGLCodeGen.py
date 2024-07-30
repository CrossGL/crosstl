from .MetalAst import *
from .MetalParser import *
from .MetalLexer import *


class MetalToCrossGLConverter:
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

    def generate(self, ast):
        self.process_structs(ast)

        code = "shader main {\n"

        # Generate vertex shader
        vertex_func = next(
            f
            for f in ast.functions
            if isinstance(f, FunctionNode) and f.qualifier == "vertex"
        )
        code += "    // Vertex Shader\n"
        code += "    vertex {\n"
        code += self.generate_io_declarations("vertex")
        code += "\n"
        code += self.generate_main_function(vertex_func)
        code += "    }\n\n"

        # Generate custom functions
        code += " \n"
        for func in ast.functions:
            if isinstance(func, FunctionNode) and func.qualifier is None:
                code += self.generate_function(func)

        # Generate fragment shader
        fragment_func = next(
            f
            for f in ast.functions
            if isinstance(f, FunctionNode) and f.qualifier == "fragment"
        )
        code += "    // Fragment Shader\n"
        code += "    fragment {\n"
        code += self.generate_io_declarations("fragment")
        code += "\n"
        code += self.generate_main_function(fragment_func)
        code += "    }\n"

        code += "}\n"
        return code

    def process_structs(self, ast):
        for node in ast.functions:
            if isinstance(node, StructNode):
                if node.name == "Vertex_INPUT":
                    for member in node.members:
                        self.vertex_inputs.append(
                            (self.map_type(member.vtype), member.name)
                        )
                elif node.name == "Vertex_OUTPUT":
                    for member in node.members:
                        if member.name != "position":
                            self.vertex_outputs.append(
                                (self.map_type(member.vtype), member.name)
                            )
                elif node.name == "Fragment_INPUT":
                    for member in node.members:
                        self.fragment_inputs.append(
                            (self.map_type(member.vtype), member.name)
                        )
                elif node.name == "Fragment_OUTPUT":
                    for member in node.members:
                        self.fragment_outputs.append(
                            (self.map_type(member.vtype), member.name)
                        )

    def generate_io_declarations(self, shader_type):
        code = ""
        if shader_type == "vertex":
            for type, name in self.vertex_inputs:
                code += f"        input {type} {name};\n"
            for type, name in self.vertex_outputs:
                code += f"        output {type} {name};\n"
        elif shader_type == "fragment":
            for type, name in self.fragment_inputs:
                code += f"        input {type} {name};\n"
            for type, name in self.fragment_outputs:
                code += f"        output {type} {name};\n"
        return code.rstrip()

    def generate_function(self, func):
        params = ", ".join(f"{self.map_type(p.vtype)} {p.name}" for p in func.params)
        code = f"    {self.map_type(func.return_type)} {func.name}({params}) {{\n"
        code += self.generate_function_body(func.body, indent=2)
        code += "    }\n\n"
        return code

    def generate_main_function(self, func):
        code = "        void main() {\n"
        code += self.generate_function_body(func.body, indent=3, is_main=True)
        code += "        }\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            if isinstance(stmt, VariableNode) and stmt.name in ["output", "input"]:
                continue
            code += "    " * indent
            if isinstance(stmt, VariableNode):

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
            and node.left.member == "position"
        ):
            return f"gl_Position = {rhs}"
        if (
            is_main
            and isinstance(node.left, MemberAccessNode)
            and node.left.object == "output"
        ):
            return f"{node.left.member} = {rhs}"
        return f"{lhs} = {rhs}"

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            return f"{expr.vtype} {expr.name}"
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.op} {right}"
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

    def map_type(self, metal_type):
        return self.type_map.get(metal_type, metal_type)
