from .MetalAst import *
from .MetalParser import *
from .MetalLexer import *


class MetalToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "char": "int8",
            "uchar": "uint8",
            "short": "int16",
            "ushort": "uint16",
            "int": "int",
            "uint": "uint",
            "long": "int64",
            "ulong": "uint64",
            "int64_t": "int64",
            "uint64_t": "uint64",
            "float": "float",
            "half": "float16",
            "double": "double",
            "size_t": "uint64",
            "ptrdiff_t": "int64",
            # Vector Types - float
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            # Vector Types - half
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            # Vector Types - int
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            # Vector Types - uint
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
            # Vector Types - short
            "short2": "i16vec2",
            "short3": "i16vec3",
            "short4": "i16vec4",
            # Vector Types - ushort
            "ushort2": "u16vec2",
            "ushort3": "u16vec3",
            "ushort4": "u16vec4",
            # Vector Types - char
            "char2": "i8vec2",
            "char3": "i8vec3",
            "char4": "i8vec4",
            # Vector Types - uchar
            "uchar2": "u8vec2",
            "uchar3": "u8vec3",
            "uchar4": "u8vec4",
            # Vector Types - bool
            "bool2": "bvec2",
            "bool3": "bvec3",
            "bool4": "bvec4",
            # Matrix Types - float
            "float2x2": "mat2",
            "float2x3": "mat2x3",
            "float2x4": "mat2x4",
            "float3x2": "mat3x2",
            "float3x3": "mat3",
            "float3x4": "mat3x4",
            "float4x2": "mat4x2",
            "float4x3": "mat4x3",
            "float4x4": "mat4",
            # Matrix Types - half
            "half2x2": "f16mat2",
            "half2x3": "f16mat2x3",
            "half2x4": "f16mat2x4",
            "half3x2": "f16mat3x2",
            "half3x3": "f16mat3",
            "half3x4": "f16mat3x4",
            "half4x2": "f16mat4x2",
            "half4x3": "f16mat4x3",
            "half4x4": "f16mat4",
            # Texture Types
            "texture1d": "sampler1D",
            "texture1d<float>": "sampler1D",
            "texture1d<half>": "sampler1D",
            "texture1d<int>": "isampler1D",
            "texture1d<uint>": "usampler1D",
            "texture2d": "sampler2D",
            "texture2d<float>": "sampler2D",
            "texture2d<half>": "sampler2D",
            "texture2d<int>": "isampler2D",
            "texture2d<uint>": "usampler2D",
            "texture3d": "sampler3D",
            "texture3d<float>": "sampler3D",
            "texture3d<half>": "sampler3D",
            "texture3d<int>": "isampler3D",
            "texture3d<uint>": "usampler3D",
            "texturecube": "samplerCube",
            "texturecube<float>": "samplerCube",
            "texturecube<half>": "samplerCube",
            "texturecube<int>": "isamplerCube",
            "texturecube<uint>": "usamplerCube",
            "TextureCube": "samplerCube",
            "texture2d_array": "sampler2DArray",
            "texture2d_array<float>": "sampler2DArray",
            "texture2d_array<half>": "sampler2DArray",
            "texture2d_array<int>": "isampler2DArray",
            "texture2d_array<uint>": "usampler2DArray",
            "depth2d": "sampler2DShadow",
            "depth2d<float>": "sampler2DShadow",
            # Sampler type
            "sampler": "sampler",
        }

        self.map_semantics = {
            # Vertex attributes
            "attribute(0)": "Position",
            "attribute(1)": "Normal",
            "attribute(2)": "Tangent",
            "attribute(3)": "Binormal",
            "attribute(4)": "TexCoord",
            "attribute(5)": "TexCoord0",
            "attribute(6)": "TexCoord1",
            "attribute(7)": "TexCoord2",
            "attribute(8)": "TexCoord3",
            "attribute(9)": "Color",
            "attribute(10)": "Color0",
            "attribute(11)": "Color1",
            "vertex_id": "gl_VertexID",
            "instance_id": "gl_InstanceID",
            "base_vertex": "gl_BaseVertex",
            "base_instance": "gl_BaseInstance",
            "position": "gl_Position",
            "point_size": "gl_PointSize",
            "clip_distance": "gl_ClipDistance",
            "front_facing": "gl_IsFrontFace",
            "point_coord": "gl_PointCoord",
            "color(0)": "gl_FragColor",
            "color(1)": "gl_FragColor1",
            "color(2)": "gl_FragColor2",
            "color(3)": "gl_FragColor3",
            "color(4)": "gl_FragColor4",
            "depth(any)": "gl_FragDepth",
            "stage_in": "gl_FragColor",
        }

    def generate(self, ast):
        code = "shader main {\n"
        # Generate custom functions
        code += "\n"
        self.constant_struct_name = []

        # Get constants - support both 'constant' and 'constants' attributes
        constants = getattr(ast, "constant", []) or getattr(ast, "constants", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                self.process_constant_struct(ast)

        # Get structs - support both 'struct' and 'structs' attributes
        structs = getattr(ast, "structs", []) or getattr(ast, "struct", []) or []
        for struct_node in structs:
            if isinstance(struct_node, StructNode):
                if struct_node.name in self.constant_struct_name:
                    code += "    // cbuffers\n"
                    code += f"    cbuffer {struct_node.name} {{\n"
                else:
                    code += "    // Structs\n"
                    code += f"    struct {struct_node.name} {{\n"
                for member in struct_node.members:
                    semantic = self.map_semantic(getattr(member, "attributes", None))
                    code += f"        {self.map_type(member.vtype)} {member.name} {semantic};\n"
                code += "    }\n\n"

        # Get functions
        functions = getattr(ast, "functions", []) or []
        for f in functions:
            qualifier = getattr(f, "qualifier", None)
            if qualifier == "vertex":
                code += "    // Vertex Shader\n"
                code += "    vertex {\n"
                code += self.generate_function(f)
                code += "    }\n\n"
            elif qualifier == "fragment":
                code += "    // Fragment Shader\n"
                code += "    fragment {\n"
                code += self.generate_function(f)
                code += "    }\n\n"
            elif qualifier == "kernel":
                code += "    // Compute Shader\n"
                code += "    compute {\n"
                code += self.generate_function(f)
                code += "    }\n\n"
            else:
                code += self.generate_function(f)

        code += "}\n"
        return code

    def process_constant_struct(self, node):
        constants = (
            getattr(node, "constant", []) or getattr(node, "constants", []) or []
        )
        structs = getattr(node, "structs", []) or getattr(node, "struct", []) or []
        for constant in constants:
            if isinstance(constant, ConstantBufferNode):
                # Iterate over all structs and append the ones matching the constant name
                self.constant_struct_name.extend(
                    struct.name for struct in structs if struct.name == constant.name
                )

    def generate_function(self, func, indent=2):
        code = ""
        code += "    " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name} {self.map_semantic(p.attributes)}"
            for p in func.params
        )
        code += f"{self.map_type(func.return_type)} {func.name}({params})  {self.map_semantic(func.attributes)} {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                const_str = (
                    "const " if hasattr(stmt, "is_const") and stmt.is_const else ""
                )
                code += f"{const_str}{self.map_type(stmt.vtype)} {stmt.name};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    code += f"return {self.generate_expression(stmt.value, is_main)};\n"
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif isinstance(stmt, FunctionCallNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                # For any unhandled statement type, attempt to generate something useful
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
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
        code = ""
        if node.if_chain:
            # Handle the if chain
            for condition, body in node.if_chain:
                code += f"if ({self.generate_expression(condition, is_main)}) {{\n"
                code += self.generate_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"
        # Handling the else if chain
        if node.else_if_chain:
            for condition, body in node.else_if_chain:
                code += (
                    f" else if ({self.generate_expression(condition, is_main)}) {{\n"
                )
                code += self.generate_function_body(body, indent + 1, is_main)
                code += "    " * indent + "}"

        # Handling the else condition
        if node.else_body:
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
        if expr is None:
            return ""
        elif isinstance(expr, str):
            return expr
        elif isinstance(expr, VariableNode):
            if expr.vtype:
                const_str = (
                    "const " if hasattr(expr, "is_const") and expr.is_const else ""
                )
                return f"{const_str}{self.map_type(expr.vtype)} {expr.name}"
            else:
                return expr.name
        elif isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr, is_main)
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            return f"{left} {expr.op} {right}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{expr.name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"({expr.op}{operand})"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, TextureSampleNode):
            texture = self.generate_expression(expr.texture, is_main)
            self.generate_expression(expr.sampler, is_main)
            coords = self.generate_expression(expr.coordinates, is_main)

            # Handle LOD parameter if present
            if hasattr(expr, "lod") and expr.lod is not None:
                lod = self.generate_expression(expr.lod, is_main)
                # In CrossGL, texture sampling with LOD is done with textureLod(sampler, coordinates, lod)
                return f"textureLod({texture}, {coords}, {lod})"

            # In CrossGL, texture sampling is done with texture(sampler, coordinates)
            return f"texture({texture}, {coords})"
        elif isinstance(expr, float) or isinstance(expr, int) or isinstance(expr, bool):
            return str(expr)
        else:
            # For any unhandled expression type, return a placeholder
            return f"/* Unhandled expression: {type(expr).__name__} */"

    def map_type(self, metal_type):
        if metal_type:
            # Special case for generic types not explicitly defined in the map
            if (
                "<" in metal_type
                and ">" in metal_type
                and metal_type not in self.type_map
            ):
                base_type, inner_type = metal_type.split("<", 1)
                inner_type = inner_type.rstrip(">")
                if base_type == "texture2d":
                    if inner_type in ["float", "half"]:
                        return "sampler2D"
                    elif inner_type == "int":
                        return "isampler2D"
                    elif inner_type == "uint":
                        return "usampler2D"
                elif base_type == "texturecube":
                    if inner_type in ["float", "half"]:
                        return "samplerCube"
                    elif inner_type == "int":
                        return "isamplerCube"
                    elif inner_type == "uint":
                        return "usamplerCube"

            return self.type_map.get(metal_type, metal_type)
        return metal_type

    def map_semantic(self, semantic):
        if semantic:
            for semantic in semantic:
                if isinstance(semantic, AttributeNode):
                    name = semantic.name
                    args = semantic.args
                    if args:
                        out = self.map_semantics.get(
                            f"{name}({args[0]})", f"{name}({args[0]})"
                        )
                        return f"@{out}"
                    else:
                        out = self.map_semantics.get(f"{name}", f"{name}")
                        return f"@{out}"
                else:
                    return ""
        else:
            return ""

    def generate_switch_statement(self, node, indent, is_main):
        """Generate CrossGL code for a switch statement

        Args:
            node: SwitchNode representing a Metal switch statement
            indent: Current indentation level
            is_main: Whether this is within the main function

        Returns:
            str: The CrossGL switch statement
        """
        expression = self.generate_expression(node.expression, is_main)
        code = f"switch ({expression}) {{\n"

        # Generate case statements
        for case in node.cases:
            case_value = self.generate_expression(case.value, is_main)
            code += "    " * (indent + 1) + f"case {case_value}:\n"

            # Generate case body
            for stmt in case.statements:
                code += "    " * (indent + 2)
                if isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent + 2, is_main)
                else:
                    code += self.generate_expression(stmt, is_main) + ";\n"

            # Add implicit break if not present
            code += "    " * (indent + 2) + "break;\n"

        # Generate default case if present
        if node.default:
            code += "    " * (indent + 1) + "default:\n"

            for stmt in node.default:
                code += "    " * (indent + 2)
                if isinstance(stmt, SwitchNode):
                    code += self.generate_switch_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, IfNode):
                    code += self.generate_if_statement(stmt, indent + 2, is_main)
                elif isinstance(stmt, ForNode):
                    code += self.generate_for_loop(stmt, indent + 2, is_main)
                else:
                    code += self.generate_expression(stmt, is_main) + ";\n"

            # Add implicit break if not present
            code += "    " * (indent + 2) + "break;\n"

        code += "    " * indent + "}\n"
        return code
