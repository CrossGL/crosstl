from ..ast import ShaderNode, AssignmentNode


class GLSLCodeGen:
    def generate(self, ast):
        code = ""
        if isinstance(ast, ShaderNode):
            code += self.generate_shader(ast)
        return code

    def generate_assignment(self, node):
        return f"{node.name} = {node.value};"

    def generate_shader(self, node):
        code = "#version 450\n\n"
        for i, (vtype, name) in enumerate(node.inputs):
            code += f"layout(location = {i}) in {self.map_type(vtype)} {name};\n"
        for i, (vtype, name) in enumerate(node.outputs):
            code += f"layout(location = {i}) out {self.map_type(vtype)} {name};\n"
        code += f"\n{self.generate_function(node.main_function)}\n"
        return code

    def generate_function(self, node):
        code = f"void {node.name}() {{\n"
        for stmt in node.body:
            if isinstance(stmt, AssignmentNode):
                code += f"    {self.generate_assignment(stmt)}\n"
        code += "}"
        return code

    def map_type(self, vtype):
        return vtype
