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
)


class MetalCodeGen:
    def __init__(self):
        self.current_shader = None
        self.shader_inputs = []
        self.shader_outputs = []

    def generate(self, ast):
        if isinstance(ast, ShaderNode):
            self.current_shader = ast
            return self.generate_shader(ast)
        return ""

    def generate_shader(self, node):
        self.shader_inputs = node.inputs
        self.shader_outputs = node.outputs

        code = "#include <metal_stdlib>\nusing namespace metal;\n\n"
        code += "struct VertexInput {\n"
        for i, (vtype, name) in enumerate(node.inputs):
            code += f"    {self.map_type(vtype)} {name} [[attribute({i})]];\n"
        code += "};\n\n"

        code += "struct FragmentOutput {\n"
        for i, (vtype, name) in enumerate(node.outputs):
            code += f"    {self.map_type(vtype)} {name} [[color({i})]];\n"
        code += "};\n\n"

        for function in node.functions:
            code += self.generate_function(function) + "\n"
        return code

    def generate_function(self, node):
        if node.name == "main":
            params = "VertexInput input [[stage_in]]"
            return_type = "fragment FragmentOutput"
            is_main = True
        else:
            params = ", ".join(
                f"{self.map_type(param[0])} {param[1]}" for param in node.params
            )
            return_type = self.map_type(node.return_type)
            is_main = False

        if node.name == "main":
            code = f"{return_type} fr{node.name}({params}) {{\n"
        else:
            code = f"{return_type} {node.name}({params}) {{\n"
        if node.name == "main":
            code += "    FragmentOutput output;\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, 1, is_main=is_main)
        if node.name == "main":
            code += "    return output;\n"
        code += "}\n"
        return code

    def generate_statement(self, stmt, indent=0, is_main=False):
        indent_str = "    " * indent
        if isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt, is_main)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent, is_main)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent, is_main)
        elif isinstance(stmt, ReturnNode):
            return (
                f"{indent_str}return {self.generate_expression(stmt.value, is_main)};\n"
            )
        else:
            return f"{indent_str}{self.generate_expression(stmt, is_main)};\n"

    def generate_assignment(self, node, is_main=False):
        if isinstance(node.name, VariableNode) and node.name.vtype:
            return f"{self.map_type(node.name.vtype)} {node.name.name} = {self.generate_expression(node.value, is_main)}"
        else:
            lhs = self.generate_expression(node.name, is_main)
            if lhs in [var[1] for var in self.shader_outputs]:
                lhs = f"output.{lhs}"
            return f"{lhs} = {self.generate_expression(node.value, is_main)}"

    def generate_if(self, node, indent, is_main=False):
        indent_str = "    " * indent
        code = (
            f"{indent_str}if ({self.generate_expression(node.condition, is_main)}) {{\n"
        )
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1, is_main)
        code += f"{indent_str}}}"
        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1, is_main)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node, indent, is_main=False):
        indent_str = "    " * indent

        if isinstance(node.init, AssignmentNode) and isinstance(
            node.init.name, VariableNode
        ):
            init = f"{self.map_type(node.init.name.vtype)} {node.init.name.name} = {self.generate_expression(node.init.value, is_main)}"
        else:
            init = self.generate_statement(node.init, 0, is_main).strip()[
                :-1
            ]  # Remove trailing semicolon

        condition = self.generate_expression(node.condition, is_main)
        update = self.generate_statement(node.update, 0, is_main).strip()[
            :-1
        ]  # Remove trailing semicolon

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1, is_main)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return self.translate_expression(expr, is_main)
        elif isinstance(expr, VariableNode):
            return self.translate_expression(expr.name, is_main)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"({left} {self.map_operator(expr.op)} {right})"
        elif isinstance(expr, FunctionCallNode):
            func_name = self.translate_expression(expr.name, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            return f"{self.generate_expression(expr.object, is_main)}.{expr.member}"
        else:
            return str(expr)

    def translate_expression(self, expr, is_main):
        translations = {
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
        }
        if is_main:
            for _, input_name in self.shader_inputs:
                translations[input_name] = f"input.{input_name}"

        # First, replace vector constructors
        for glsl, metal in translations.items():
            expr = expr.replace(f"{glsl}(", f"{metal}(")

        # Then, replace any remaining exact matches
        words = expr.split()
        translated_words = [translations.get(word, word) for word in words]
        return " ".join(translated_words)

    def map_type(self, vtype):
        type_mapping = {
            "void": "void",
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "sampler2D": "texture2d<float>",
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
                            output vec4 fragColor;
                            vec3 customFunction(vec3 random, float factor) {
                                return random * factor;
                            }

                            void main() {
                                vec3 color = vec3(position.x, position.y, 0.0);
                                float factor = 1.0;

                                if (texCoord.x > 0.5) {
                                    color = vec3(1.0, 0.0, 0.0);
                                } else {
                                    color = vec3(0.0, 1.0, 0.0);
                                }

                                for (int i = 0; i < 3; i= i + 1) {
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

    codegen = MetalCodeGen()
    metal_code = codegen.generate(ast)
    print(metal_code)
