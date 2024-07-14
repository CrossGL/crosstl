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
            # Scalar Types
            "void": "void",
            "char": "int",
            "signed char": "int",
            "unsigned char": "uint",
            "short": "int",
            "signed short": "int",
            "unsigned short": "uint",
            "int": "int",
            "signed int": "int",
            "unsigned int": "uint",
            "long": "int64_t",
            "signed long": "int64_t",
            "unsigned long": "uint64_t",
            "float": "float",
            "half": "half",
            "bool": "bool",

            # Vector Types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "char2": "int2",
            "char3": "int3",
            "char4": "int4",
            "uchar2": "uint2",
            "uchar3": "uint3",
            "uchar4": "uint4",
            "short2": "int2",
            "short3": "int3",
            "short4": "int4",
            "ushort2": "uint2",
            "ushort3": "uint3",
            "ushort4": "uint4",
            "int2": "int2",
            "int3": "int3",
            "int4": "int4",
            "uint2": "uint2",
            "uint3": "uint3",
            "uint4": "uint4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "float2": "float2",
            "float3": "float3",
            "float4": "float4",
            "half2": "half2",
            "half3": "half3",
            "half4": "half4",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "bool2": "bool2",
            "bool3": "bool3",
            "bool4": "bool4",

            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",

            # Matrix Types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "half2x2": "half2x2",
            "half3x3": "half3x3",
            "half4x4": "half4x4"
        }
        return type_mapping.get(vtype, vtype)
    

 # Usage example
if __name__ == "__main__":
    from compiler.lexer import Lexer
    from compiler.parser import Parser

    lexer = Lexer("shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }")
    parser = Parser(lexer.tokens)
    ast = parser.parse()

    codegen = MetalCodeGen()
    metal_code = codegen.generate(ast)
    print(metal_code)
