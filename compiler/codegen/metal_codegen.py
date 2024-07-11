from ..ast import ShaderNode, AssignmentNode


class MetalCodeGen:
    def generate(self, ast):
        code = ""
        if isinstance(ast, ShaderNode):
            code += self.generate_shader(ast)
        return code

    def generate_assignment(self, node):
        return f"{node.name} = {node.value};"

    def generate_shader(self, node):
        code = "#include <metal_stdlib>\nusing namespace metal;\n\n"
        code += "struct VertexInput {\n"
        for i, (vtype, name) in enumerate(node.inputs):
            code += f"    {self.map_type(vtype)} {name} [[attribute({i})]];\n"
        code += "};\n\n"
        code += "struct FragmentOutput {\n"
        for i, (vtype, name) in enumerate(node.outputs):
            code += f"    {self.map_type(vtype)} {name} [[color({i})]];\n"
        code += "};\n\n"
        code += f"{self.generate_function(node.main_function)}\n"
        return code

    def generate_function(self, node):
        code = (
            f"fragment FragmentOutput {node.name}(VertexInput input [[stage_in]]) {{\n"
        )
        code += "    FragmentOutput output;\n"
        for stmt in node.body:
            if isinstance(stmt, AssignmentNode):
                code += f"    {self.generate_assignment(stmt)}\n"
        code += "    return output;\n"
        code += "}"
        return code

    def map_type(self, vtype):
        type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
        }
        return type_mapping.get(vtype, vtype)
