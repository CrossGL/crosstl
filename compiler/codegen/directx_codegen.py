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
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
        }
        return type_mapping.get(vtype, vtype)


# Usage example
if __name__ == "__main__":
    from compiler.lexer import Lexer
    from compiler.parser import Parser

    code = "shader main { input vec3 position; output vec4 color; void main() { color = vec4(position, 1.0); } }"
    lexer = Lexer(code)
    parser = Parser(lexer.tokens)
    ast = parser.parse()

    codegen = HLSLCodeGen()
    hlsl_code = codegen.generate(ast)
    print(hlsl_code)
