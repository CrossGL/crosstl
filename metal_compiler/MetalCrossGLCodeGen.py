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
        self.process_functions(ast)

        code = "shader main {\n"

        # Generate custom functions
        for func in ast.functions:
            if isinstance(func, FunctionNode) and func.qualifier is None:
                code += self.generate_function(func)

        # Generate vertex shader
        vertex_func = next(
            f
            for f in ast.functions
            if isinstance(f, FunctionNode) and f.qualifier == "vertex"
        )
        code += "vertex {\n"
        code += self.generate_io_declarations("vertex")
        code += self.generate_main_function(vertex_func)
        code += "}\n"

        # Generate fragment shader
        fragment_func = next(
            f
            for f in ast.functions
            if isinstance(f, FunctionNode) and f.qualifier == "fragment"
        )
        code += "fragment {\n"
        code += self.generate_io_declarations("fragment")
        code += self.generate_main_function(fragment_func)
        code += "}\n"

        code += "}\n"
        return code

    def process_structs(self, ast):
        for node in ast.functions:
            if isinstance(node, StructNode):
                if node.name == "VertexInput":
                    for member in node.members:
                        self.vertex_inputs.append(
                            (self.map_type(member.vtype), member.name)
                        )
                elif node.name == "FragmentInput":
                    for member in node.members:
                        if member.name != "position":
                            self.vertex_outputs.append(
                                (self.map_type(member.vtype), member.name)
                            )
                            self.fragment_inputs.append(
                                (self.map_type(member.vtype), member.name)
                            )

    def process_functions(self, ast):
        for func in ast.functions:
            if isinstance(func, FunctionNode):
                if func.qualifier == "fragment":
                    self.fragment_outputs.append(
                        (self.map_type(func.return_type), "cgl_FragColor")
                    )

    def generate_io_declarations(self, shader_type):
        code = ""
        if shader_type == "vertex":
            for type, name in self.vertex_inputs:
                code += f"input {type} {name};\n"
            for type, name in self.vertex_outputs:
                code += f"output {type} {name};\n"
        elif shader_type == "fragment":
            for type, name in self.fragment_inputs:
                code += f"input {type} {name};\n"
            for type, name in self.fragment_outputs:
                code += f"output {type} {name};\n"
        return code

    def generate_function(self, func):
        params = ", ".join(f"{self.map_type(p.vtype)} {p.name}" for p in func.params)
        code = f"{self.map_type(func.return_type)} {func.name}({params}) {{\n"
        code += self.generate_function_body(func.body, indent=1)
        code += "}\n\n"
        return code

    def generate_main_function(self, func):
        code = "void main() {\n"
        code += self.generate_function_body(func.body, indent=1, is_main=True)
        code += "}\n"
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
            elif isinstance(stmt, IfNode):
                code += self.generate_if(stmt, indent, is_main)
            elif isinstance(stmt, ForNode):
                code += self.generate_for(stmt, indent, is_main)
            elif isinstance(stmt, ReturnNode):
                if is_main:
                    if any(
                        output[1] == "cgl_FragColor" for output in self.fragment_outputs
                    ):
                        code += f"cgl_FragColor = {self.generate_expression(stmt.value, is_main)};\n"
                    else:
                        # Handle vertex shader output
                        code += f"// Return statement converted to assignments\n"
                        for type, name in self.vertex_outputs:
                            code += f"{name} = {self.generate_expression(stmt.value, is_main)}.{name};\n"
                        code += f"cgl_position = {self.generate_expression(stmt.value, is_main)}.position;\n"
                else:
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
            return f"cgl_position = {rhs}"
        return f"{lhs} = {rhs}"

    def generate_if(self, node, indent, is_main):
        code = f"if ({self.generate_expression(node.condition, is_main)}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1, is_main)
        code += "    " * indent + "}"
        if node.else_body:
            code += " else {\n"
            code += self.generate_function_body(node.else_body, indent + 1, is_main)
            code += "    " * indent + "}"
        code += "\n"
        return code

    def generate_for(self, node, indent, is_main):
        init = self.generate_expression(node.init, is_main)
        condition = self.generate_expression(node.condition, is_main)
        update = self.generate_expression(node.update, is_main)
        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            return expr.name
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
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
                return f"{expr.member}"
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


# Usage
if __name__ == "__main__":
    code = """
    // This is a single-line comment
    /* This is a
       multi-line comment */
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
        float2 texCoord [[attribute(1)]];
    };

    struct FragmentInput {
        float4 position [[position]];
        float2 texCoord;
    };

    float4 calculateNormal(float3 position, float3 normal)
    {
        float4 result = float4(normal, 1.0);
        return result;
    }

    vertex FragmentInput vertexShader(VertexInput input [[stage_in]])
    {
        FragmentInput output;
        output.texCoord = input.texCoord;
        for (int i = 0; i < 10; i = i + 1) {
            output.position = float4(input.position, 1.0);
        }
        
        if (output.position.x > 0.0) {
            output.position.y = 0.0;
        } else {
            output.position.y = 1.0;
        }
        
        output.position = float4(input.position, 1.0);
        return output;
    }

    fragment float4 fragmentShader(FragmentInput input [[stage_in]])
    {
        float4 color = input.texCoord;
        return float4(color.rgb, 1.0);
    }

    """

    lexer = MetalLexer(code)
    parser = MetalParser(lexer.tokens)
    ast = parser.parse()
    print("Parsing completed successfully!")
    print(ast)
    codegen = MetalToCrossGLConverter()
    metal_code = codegen.generate(ast)
    print(metal_code)
