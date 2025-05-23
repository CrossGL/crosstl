from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    ArrayAccessNode,
    ArrayNode,
)
from .array_utils import parse_array_type, format_array_type, get_array_size_from_node


class HLSLCodeGen:
    def __init__(self):
        self.type_mapping = {
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
            "sampler": "SamplerState",
        }

        self.semantic_map = {
            "gl_VertexID": "SV_VertexID",
            "gl_InstanceID": "SV_InstanceID",
            "gl_IsFrontFace": "FRONT_FACE",
            "gl_PrimitiveID": "PRIMITIVE_ID",
            "InstanceID": "INSTANCE_ID",
            "VertexID": "VERTEX_ID",
            "gl_Position": "SV_POSITION",
            "gl_PointSize": "SV_POINTSIZE",
            "gl_ClipDistance": "SV_ClipDistance",
            "gl_CullDistance": "SV_CullDistance",
            "gl_FragColor": "SV_TARGET",
            "gl_FragColor0": "SV_TARGET0",
            "gl_FragColor1": "SV_TARGET1",
            "gl_FragColor2": "SV_TARGET2",
            "gl_FragColor3": "SV_TARGET3",
            "gl_FragColor4": "SV_TARGET4",
            "gl_FragColor5": "SV_TARGET5",
            "gl_FragColor6": "SV_TARGET6",
            "gl_FragColor7": "SV_TARGET7",
            "gl_FragDepth": "SV_DEPTH",
        }

    def generate(self, ast):
        code = "\n"
        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"struct {node.name} {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    {self.map_type(member.element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in HLSL
                            code += f"    {self.map_type(member.element_type)}[] {member.name};\n"
                    else:
                        code += f"    {self.map_type(member.vtype)} {member.name}{self.map_semantic(member.semantic)};\n"
                code += "};\n"

        # Generate global variables
        for i, node in enumerate(ast.global_variables):
            if node.vtype in ["sampler2D", "samplerCube"]:
                code += "// Texture Samplers\n"
                code += f"{self.map_type(node.vtype)} {node.name} :register(t{i});\n"
            elif node.vtype in ["sampler"]:
                code += "// Sampler States\n"
                code += f"{self.map_type(node.vtype)} {node.name} :register(s{i});\n"
            else:
                code += f"{self.map_type(node.vtype)} {node.name};\n"

        # Generate cbuffers
        if ast.cbuffers:
            code += "// Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        # Generate custom functions
        for func in ast.functions:
            if func.qualifier == "vertex":
                code += "// Vertex Shader\n"
                code += self.generate_function(func, shader_type="vertex")
            elif func.qualifier == "fragment":
                code += "// Fragment Shader\n"
                code += self.generate_function(func, shader_type="fragment")
            elif func.qualifier == "compute":
                code += "// Compute Shader\n"
                code += self.generate_function(func, shader_type="compute")
            else:
                code += self.generate_function(func)

        return code

    def generate_cbuffers(self, ast):
        code = ""
        for i, node in enumerate(ast.cbuffers):
            if isinstance(node, StructNode):
                code += f"cbuffer {node.name} : register(b{i}) {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    {self.map_type(member.element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported, so we'll make it fixed size
                            code += f"    {self.map_type(member.element_type)} {member.name}[1];\n"
                    else:
                        code += f"    {self.map_type(member.vtype)} {member.name};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # Generic cbuffer handling
                code += f"cbuffer {node.name} : register(b{i}) {{\n"
                for member in node.members:
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += f"    {self.map_type(member.element_type)} {member.name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in cbuffers usually not supported
                            code += f"    {self.map_type(member.element_type)} {member.name}[1];\n"
                    else:
                        code += f"    {self.map_type(member.vtype)} {member.name};\n"
                code += "};\n"
        return code

    def generate_function(self, func, indent=0, shader_type=None):
        code = ""
        code += "  " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name} {self.map_semantic(p.semantic)}"
            for p in func.params
        )
        shader_map = {"vertex": "VSMain", "fragment": "PSMain", "compute": "CSMain"}

        if func.qualifier in shader_map:
            code += f"// {func.qualifier.capitalize()} Shader\n"
            code += f"{self.map_type(func.return_type)} {shader_map[func.qualifier]}({params}) {{\n"
        else:
            code += f"{self.map_type(func.return_type)} {func.name}({params}) {{\n"

        for stmt in func.body:
            code += self.generate_statement(stmt, indent + 1)
        code += "  " * indent + "}\n\n"
        return code

    def generate_statement(self, stmt, indent=0):
        indent_str = "    " * indent
        if isinstance(stmt, VariableNode):
            return f"{indent_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
        elif isinstance(stmt, ArrayNode):
            # Improved array node handling
            element_type = self.map_type(stmt.element_type)
            size = get_array_size_from_node(stmt)

            if size is None:
                # HLSL dynamic arrays need a size, but can be accessed with buffer types
                # For basic shaders, use a fixed size as fallback
                return f"{indent_str}{element_type}[1024] {stmt.name};\n"
            else:
                return f"{indent_str}{element_type}[{size}] {stmt.name};\n"
        elif isinstance(stmt, AssignmentNode):
            return f"{indent_str}{self.generate_assignment(stmt)};\n"
        elif isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        elif isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        elif isinstance(stmt, ReturnNode):
            code = ""
            for i, return_stmt in enumerate(stmt.value):
                code += f"{self.generate_expression(return_stmt)}"
                if i < len(stmt.value) - 1:
                    code += ", "
            return f"{indent_str}return {code};\n"
        else:
            return f"{indent_str}{self.generate_expression(stmt)};\n"

    def generate_assignment(self, node):
        lhs = self.generate_expression(node.left)
        rhs = self.generate_expression(node.right)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def generate_if(self, node, indent):
        indent_str = "    " * indent
        code = f"{indent_str}if ({self.generate_expression(node.if_condition)}) {{\n"
        for stmt in node.if_body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}"

        for else_if_condition, else_if_body in zip(
            node.else_if_conditions, node.else_if_bodies
        ):
            code += f" else if ({self.generate_expression(else_if_condition)}) {{\n"
            for stmt in else_if_body:
                code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"

        if node.else_body:
            code += " else {\n"
            for stmt in node.else_body:
                code += self.generate_statement(stmt, indent + 1)
            code += f"{indent_str}}}"
        code += "\n"
        return code

    def generate_for(self, node, indent):
        indent_str = "    " * indent

        # Extract and remove the trailing semicolon from init, condition, and update expressions
        init = self.generate_statement(node.init, 0).strip().rstrip(";")
        condition = self.generate_statement(node.condition, 0).strip().rstrip(";")
        update = self.generate_statement(node.update, 0).strip().rstrip(";")

        code = f"{indent_str}for ({init}; {condition}; {update}) {{\n"
        for stmt in node.body:
            code += self.generate_statement(stmt, indent + 1)
        code += f"{indent_str}}}\n"
        return code

    def generate_expression(self, expr):
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            name = self.generate_expression(expr.name)
            return f"{self.map_type(expr.vtype)} {name}"
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"{left} {self.map_operator(expr.op)} {right}"
        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left)
            right = self.generate_expression(expr.right)
            return f"{left} {self.map_operator(expr.operator)} {right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            return f"{self.map_operator(expr.op)}{operand}"
        elif isinstance(expr, ArrayAccessNode):
            # Handle array access
            array = self.generate_expression(expr.array)
            index = self.generate_expression(expr.index)
            return f"{array}[{index}]"
        elif isinstance(expr, FunctionCallNode):
            # Handle special vector constructor calls
            if expr.name in ["vec2", "vec3", "vec4"]:
                mapped_type = self.map_type(expr.name)
                args = ", ".join(self.generate_expression(arg) for arg in expr.args)
                return f"{mapped_type}({args})"
            # Standard function call
            args = ", ".join(self.generate_expression(arg) for arg in expr.args)
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition)} ? {self.generate_expression(expr.true_expr)} : {self.generate_expression(expr.false_expr)}"
        else:
            return str(expr)

    def map_type(self, vtype):
        if vtype:
            # Handle array types with a more robust approach
            if "[" in vtype and "]" in vtype:
                base_type, size = parse_array_type(vtype)
                base_mapped = self.type_mapping.get(base_type, base_type)
                return format_array_type(base_mapped, size, "hlsl")

            # Use the regular type mapping for non-array types
            return self.type_mapping.get(vtype, vtype)
        return vtype

    def map_operator(self, op):
        op_map = {
            "PLUS": "+",
            "MINUS": "-",
            "MULTIPLY": "*",
            "DIVIDE": "/",
            "BITWISE_XOR": "^",
            "BITWISE_OR": "|",
            "BITWISE_AND": "&",
            "LESS_THAN": "<",
            "GREATER_THAN": ">",
            "ASSIGN_ADD": "+=",
            "ASSIGN_SUB": "-=",
            "ASSIGN_MUL": "*=",
            "ASSIGN_DIV": "/=",
            "ASSIGN_MOD": "%=",
            "LESS_EQUAL": "<=",
            "GREATER_EQUAL": ">=",
            "EQUAL": "==",
            "NOT_EQUAL": "!=",
            "AND": "&&",
            "OR": "||",
            "EQUALS": "=",
            "ASSIGN_SHIFT_LEFT": "<<=",
            "ASSIGN_SHIFT_RIGHT": ">>=",
            "ASSIGN_AND": "&=",
            "ASSIGN_OR": "|=",
            "ASSIGN_XOR": "^=",
            "LOGICAL_AND": "&&",
            "LOGICAL_OR": "||",
            "BITWISE_SHIFT_RIGHT": ">>",
            "BITWISE_SHIFT_LEFT": "<<",
            "MOD": "%",
        }
        return op_map.get(op, op)

    def map_semantic(self, semantic):
        if semantic:
            return f": {self.semantic_map.get(semantic, semantic)}"
        else:
            return ""  # Handle None by returning an empty string
