from ..ast import (
    ShaderNode,
    AssignmentNode,
    FunctionNode,
    ReturnNode,
    BinaryOpNode,
    IfNode,
    ForNode,
    VariableNode,
    FunctionCallNode,
    MemberAccessNode,
    UnaryOpNode,
)


class HLSLCodeGen:
    def __init__(self):
        self.current_shader = None

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            self.current_shader = ast
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        self.shader_inputs = node.inputs
        self.shader_outputs = node.outputs

        code = "struct VS_INPUT {\n"
        for i, (vtype, name) in enumerate(node.inputs):

            if i >= 1:
                code += f"    {self.map_type(vtype)} {name} : TEXCOORD{i};\n"
            else:
                code += f"    {self.map_type(vtype)} {name} : POSITION{i};\n"
        code += "};\n\n"

        code += "struct PS_OUTPUT {\n"
        for i, (vtype, name) in enumerate(node.outputs):
            code += f"    {self.map_type(vtype)} {name} : SV_TARGET{i};\n"
        code += "};\n\n"

        for function in node.functions:
            code += self.generate_function(function) + "\n"
        return code

    def generate_function(self, node):
        if node.name == "main":
            params = "VS_INPUT input"
            return_type = "PS_OUTPUT"
            is_vs_input = True
        else:
            params = ", ".join(
                f"{self.map_type(param[0])} {param[1]}" for param in node.params
            )
            return_type = self.map_type(node.return_type)
            is_vs_input = "VS_INPUT" in params

        code = f"{return_type} {node.name}({params}) {{\n"
        if node.name == "main":
            code += "    PS_OUTPUT output;\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, 1, is_vs_input=is_vs_input)
        if node.name == "main":
            code += "    return output;\n"
        code += "}\n"
        return code

    def generate_statement(self, stmt, indent=0, is_vs_input=False):
        indent_str = "    " * indent
        if isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt, is_vs_input)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent, is_vs_input)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent, is_vs_input)
        elif isinstance(stmt, ReturnNode):
            return f"{indent_str}return {self.generate_expression(stmt.value, is_vs_input)};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt, is_vs_input)};\n"

    def generate_assignment(self, node, is_vs_input=False):
        if isinstance(node.name, VariableNode) and node.name.vtype:
            return f"{self.map_type(node.name.vtype)} {node.name.name} = {self.generate_expression(node.value, is_vs_input)}"
        else:
            lhs = self.generate_expression(node.name, is_vs_input)
            if lhs in [var[1] for var in self.shader_outputs]:
                lhs = f"output.{lhs}"
            return f"{lhs} = {self.generate_expression(node.value, is_vs_input)}"

    def generate_if(self, node, indent, is_vs_input=False):
        indent_str = "    " * indent
        code = f"{indent_str}if ({self.generate_expression(node.condition, is_vs_input)}) {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1, is_vs_input)
        code += f"{indent_str}}}"
        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1, is_vs_input)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node, indent, is_vs_input=False):
        indent_str = "    " * indent

        if isinstance(node.init, AssignmentNode) and isinstance(
            node.init.name, VariableNode
        ):
            init = f"{self.map_type(node.init.name.vtype)} {node.init.name.name} = {self.generate_expression(node.init.value, is_vs_input)}"
        else:
            init = self.generate_statement(node.init, 0, is_vs_input).strip()[
                :-1
            ]  # Remove trailing semicolon

        condition = self.generate_expression(node.condition, is_vs_input)

        if isinstance(node.update, AssignmentNode) and isinstance(
            node.update.value, UnaryOpNode
        ):
            update = f"{node.update.value.operand.name}++"
        else:
            update = self.generate_statement(node.update, 0, is_vs_input).strip()[:-1]

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1, is_vs_input)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr, is_vs_input=False):
        if isinstance(expr, str):
            return self.translate_expression(expr, is_vs_input)
        elif isinstance(expr, VariableNode):
            return self.translate_expression(expr.name, is_vs_input)
        elif isinstance(expr, BinaryOpNode):
            return f"({self.generate_expression(expr.left, is_vs_input)} {self.map_operator(expr.op)} {self.generate_expression(expr.right, is_vs_input)})"
        elif isinstance(expr, UnaryOpNode):
            return f"{self.map_operator(expr.op)}{self.generate_expression(expr.operand, is_vs_input)}"
        elif isinstance(expr, FunctionCallNode) and expr.args is not None:
            args = ", ".join(
                self.generate_expression(arg, is_vs_input) for arg in expr.args
            )
            return f"{self.translate_expression(expr.name, is_vs_input)}({args})"
        elif isinstance(expr, MemberAccessNode):
            return f"{self.generate_expression(expr.object, is_vs_input)}.{expr.member}"
        else:
            return str(expr)

    def translate_expression(self, expr, is_vs_input):
        translations = {"vec2": "float2", "vec3": "float3", "vec4": "float4"}
        if is_vs_input:
            for _, input_name in self.shader_inputs:
                translations[input_name] = f"input.{input_name}"

        for cglsl, hlsl in translations.items():
            expr = expr.replace(cglsl, hlsl)
        return expr

    def map_type(self, vtype):
        type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "int": "int",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "uint": "uint",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "bool": "bool",
            "bvec2": "bool2",
            "bvec3": "bool3",
            "bvec4": "bool4",
            "float": "float",
            "double": "double",
            "sampler2D": "Texture2D",
            "samplerCube": "TextureCube",
        }
        return type_mapping.get(vtype, vtype)

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


# Usage example
if __name__ == "__main__":
    from compiler.lexer import Lexer
    from compiler.parser import Parser

    code = """shader main {
                            input vec3 position;
                            input vec2 texCoord;
                            input mat2 depth;
                            output vec4 fragColor;
                            output float depth;
                            vec3 customFunction(vec3 random, float factor) {
                                return random * factor;
                            }

                            void main() {
                                vec3 color = vec3(position.x,position.y, 0.0);
                                float factor = 1.0;

                                if (texCoord.x > 0.5) {
                                    color = vec3(1.0, 0.0, 0.0);
                                } else {
                                    color = vec3(0.0, 1.0, 0.0);
                                }

                                for (int i = 0; i < 3; i = i + 1) {
                                    factor = factor * 0.5;
                                    color = customFunction(color, factor);
                                }

                                if (length(color) > 1.0) {
                                    color = normalize(color);
                                }

                                fragColor = vec4(color, 1.0);
                            }
                        }"""
    lexer = Lexer(code)
    parser = Parser(lexer.tokens)
    ast = parser.parse()

    codegen = HLSLCodeGen()
    hlsl_code = codegen.generate(ast)
    print(hlsl_code)
