from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *


class HLSLToCrossGLConverter:
    def __init__(self):
        self.vertex_inputs = []
        self.vertex_outputs = []
        self.fragment_inputs = []
        self.fragment_outputs = []
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
            # Vertex inputs position
            "POSITION": "in_Position",
            "POSITION0": "in_Position0",
            "POSITION1": "in_Position1",
            "POSITION2": "in_Position2",
            "POSITION3": "in_Position3",
            "POSITION4": "in_Position4",
            "POSITION5": "in_Position5",
            "POSITION6": "in_Position6",
            "POSITION7": "in_Position7",
            # Vertex inputs normal
            "NORMAL": "in_Normal",
            "NORMAL0": "in_Normal0",
            "NORMAL1": "in_Normal1",
            "NORMAL2": "in_Normal2",
            "NORMAL3": "in_Normal3",
            "NORMAL4": "in_Normal4",
            "NORMAL5": "in_Normal5",
            "NORMAL6": "in_Normal6",
            "NORMAL7": "in_Normal7",
            # Vertex inputs tangent
            "TANGENT": "in_Tangent",
            "TANGENT0": "in_Tangent0",
            "TANGENT1": "in_Tangent1",
            "TANGENT2": "in_Tangent2",
            "TANGENT3": "in_Tangent3",
            "TANGENT4": "in_Tangent4",
            "TANGENT5": "in_Tangent5",
            "TANGENT6": "in_Tangent6",
            "TANGENT7": "in_Tangent7",
            # Vertex inputs binormal
            "BINORMAL": "in_Binormal",
            "BINORMAL0": "in_Binormal0",
            "BINORMAL1": "in_Binormal1",
            "BINORMAL2": "in_Binormal2",
            "BINORMAL3": "in_Binormal3",
            "BINORMAL4": "in_Binormal4",
            "BINORMAL5": "in_Binormal5",
            "BINORMAL6": "in_Binormal6",
            "BINORMAL7": "in_Binormal7",
            # Vertex inputs color
            "COLOR": "Color",
            "COLOR0": "Color0",
            "COLOR1": "Color1",
            "COLOR2": "Color2",
            "COLOR3": "Color3",
            "COLOR4": "Color4",
            "COLOR5": "Color5",
            "COLOR6": "Color6",
            "COLOR7": "Color7",
            # Vertex inputs texcoord
            "TEXCOORD": "TexCoord",
            "TEXCOORD0": "TexCoord0",
            "TEXCOORD1": "TexCoord1",
            "TEXCOORD2": "TexCoord2",
            "TEXCOORD3": "TexCoord3",
            "TEXCOORD4": "TexCoord4",
            "TEXCOORD5": "TexCoord5",
            "TEXCOORD6": "TexCoord6",
            # Vertex inputs instance
            "FRONT_FACE": "SV_IsFrontFace",
            "PRIMITIVE_ID": "SV_PrimitiveID",
            "INSTANCE_ID": "SV_InstanceID",
            "VERTEX_ID": "SV_VertexID",
            # Vertex outputs
            "SV_Position": "Out_Position",
            "SV_Position0": "Out_Position0",
            "SV_Position1": "Out_Position1",
            "SV_Position2": "Out_Position2",
            "SV_Position3": "Out_Position3",
            "SV_Position4": "Out_Position4",
            "SV_Position5": "Out_Position5",
            "SV_Position6": "Out_Position6",
            "SV_Position7": "Out_Position7",
            # Fragment inputs
            "SV_Target": "Out_Color",
            "SV_Target0": "Out_Color0",
            "SV_Target1": "Out_Color1",
            "SV_Target2": "Out_Color2",
            "SV_Target3": "Out_Color3",
            "SV_Target4": "Out_Color4",
            "SV_Target5": "Out_Color5",
            "SV_Target6": "Out_Color6",
            "SV_Target7": "Out_Color7",
            "SV_Depth": "Out_Depth",
            "SV_Depth0": "Out_Depth0",
            "SV_Depth1": "Out_Depth1",
            "SV_Depth2": "Out_Depth2",
            "SV_Depth3": "Out_Depth3",
            "SV_Depth4": "Out_Depth4",
            "SV_Depth5": "Out_Depth5",
            "SV_Depth6": "Out_Depth6",
            "SV_Depth7": "Out_Depth7",
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
