from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *


class HLSLToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            "void": "void",
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            "float2x2": "mat2",
            "float3x3": "mat3",
            "float4x4": "mat4",
            "int": "int",
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            "uint": "uint",
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            "bool": "bool",
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            "float": "float",
            "double": "double",
            "Texture2D": "sampler2D",
            "TextureCube": "samplerCube",
        }
        self.semantic_map = {
            # Vertex inputs instance
            "FRONT_FACE": "gl_IsFrontFace",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "InstanceID",
            "VERTEX_ID": "VertexID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_VertexID": "gl_VertexID",
            # Vertex outputs
            "SV_POSITION": "gl_Position",
            # Fragment inputs
            "SV_TARGET": "gl_FragColor",
            "SV_TARGET0": "gl_FragColor0",
            "SV_TARGET1": "gl_FragColor1",
            "SV_TARGET2": "gl_FragColor2",
            "SV_TARGET3": "gl_FragColor3",
            "SV_TARGET4": "gl_FragColor4",
            "SV_TARGET5": "gl_FragColor5",
            "SV_TARGET6": "gl_FragColor6",
            "SV_TARGET7": "gl_FragColor7",
            "SV_DEPTH": "gl_FragDepth",
            "SV_DEPTH0": "gl_FragDepth0",
            "SV_DEPTH1": "gl_FragDepth1",
            "SV_DEPTH2": "gl_FragDepth2",
            "SV_DEPTH3": "gl_FragDepth3",
            "SV_DEPTH4": "gl_FragDepth4",
            "SV_DEPTH5": "gl_FragDepth5",
            "SV_DEPTH6": "gl_FragDepth6",
            "SV_DEPTH7": "gl_FragDepth7",
        }

    def generate(self, ast):
        code = "shader main {\n"
        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"    struct {node.name} {{\n"
                for member in node.members:
                    code += f"        {self.map_type(member.vtype)} {member.name} {self.map_semantic(member.semantic)};\n"
                code += "    }\n"
        # Generate global variables
        for node in ast.global_variables:
            code += f"    {self.map_type(node.vtype)} {node.name};\n"
        # Generate cbuffers
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate custom functions
        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "    // Vertex Shader\n"
                code += "    vertex {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            elif func.qualifier == "fragment":
                code += "    // Fragment Shader\n"
                code += "    fragment {\n"
                code += self.generate_function(func)
                code += "    }\n\n"

            elif func.qualifier == "compute":
                code += "    // Compute Shader\n"
                code += "    compute {\n"
                code += self.generate_function(func)
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += "}\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                code += f"    cbuffer {node.name} {{\n"
                for member in node.members:
                    code += f"        {self.map_type(member.vtype)} {member.name};\n"
                code += "    }\n"
        return code

    def generate_function(self, func, indent=1):
        code = " "
        code += "  " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name} {self.map_semantic(p.semantic)}"
            for p in func.params
        )
        code += f"    {self.map_type(func.return_type)} {func.name}({params}) {self.map_semantic(func.semantic)} {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                code += f"{self.map_type(stmt.vtype)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"

            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    code += f"return {self.generate_expression(stmt.value, is_main)};\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
        return code

    def generate_for_loop(self, node, indent, is_main):
        init = self.generate_expression(node.init, is_main)
        condition = self.generate_expression(node.condition, is_main)
        update = self.generate_expression(node.update, is_main)

        code = f"for ({init}; {condition}; {update}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"while ({condition}) {{\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "}\n"
        return code

    def generate_do_while_loop(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = "do {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * indent + "} "
        code += f"while ({condition});"
        return code

    def generate_if_statement(self, node, indent, is_main):
        condition = self.generate_expression(node.condition, is_main)

        code = f"if ({condition}) {{\n"
        code += self.generate_function_body(node.if_body, indent + 1, is_main)
        code += "    " * indent + "}"

        if node.else_body:
            if isinstance(node.else_body, IfNode):
                code += " else "
                code += self.generate_if_statement(node.else_body, indent, is_main)
            else:
                code += " else {\n"
                code += self.generate_function_body(node.else_body, indent + 1, is_main)
                code += "    " * indent + "}"

        code += "\n"
        return code

    def generate_assignment(self, node, is_main):
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def generate_expression(self, expr, is_main=False):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            return f"{self.map_type(expr.vtype)} {expr.name}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.op} {right}"

        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.operator} {right}"

        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"

        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"

        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        else:
            return str(expr)

    def map_type(self, hlsl_type):
        if hlsl_type:
            return self.type_map.get(hlsl_type, hlsl_type)
        return hlsl_type

    def map_semantic(self, semantic):
        if semantic is not None:
            return f"@ {self.semantic_map.get(semantic, semantic)}"
        else:
            return ""
