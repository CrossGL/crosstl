from ..ast import ShaderNode, AssignmentNode


class HLSLCodeGen:
    def generate(self, ast):
        code = ""
        if isinstance(ast, ShaderNode):
            code += self.generate_shader(ast)
        return code

    def generate_shader(self, node):
        code = "struct VS_INPUT {\n"
        for vtype, name in node.inputs:
            code += f"    {self.map_type(vtype)} {name} : POSITION;\n"
        code += "};\n\n"

        code += "struct PS_OUTPUT {\n"
        for vtype, name in node.outputs:
            code += f"    {self.map_type(vtype)} {name} : SV_TARGET;\n"
        code += "};\n\n"

        code += f"{self.generate_function(node.main_function)}\n"
        return code

    def generate_function(self, node):
        code = f"{self.map_type('void')} {node.name}(VS_INPUT input) {{\n"
        for stmt in node.body:
            if isinstance(stmt, AssignmentNode):
                code += f"    {self.generate_assignment(stmt)}\n"
        code += "    return;\n"
        code += "}"
        return code

    def generate_assignment(self, node):
        return f"{node.name} = {node.value};"

    def map_type(self, vtype):
        type_mapping = {
            'void': 'void',
            'vec2': 'float2',
            'vec3': 'float3',
            'vec4': 'float4',
            'mat2': 'float2x2',
            'mat3': 'float3x3',
            'mat4': 'float4x4',
            'int': 'int',
            'ivec2': 'int2',
            'ivec3': 'int3',
            'ivec4': 'int4',
            'uint': 'uint',
            'uvec2': 'uint2',
            'uvec3': 'uint3',
            'uvec4': 'uint4',
            'bool': 'bool',
            'bvec2': 'bool2',
            'bvec3': 'bool3',
            'bvec4': 'bool4',
            'float': 'float',
            'double': 'double',
            'sampler2D': 'Texture2D',
            'samplerCube': 'TextureCube',
        }
        return type_mapping.get(vtype, vtype)
