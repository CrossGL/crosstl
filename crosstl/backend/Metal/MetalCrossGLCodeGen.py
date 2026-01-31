from .MetalAst import *
from .MetalParser import *
from .MetalLexer import *


class MetalToCrossGLConverter:
    def __init__(self):
        self.rt_qualifiers = {
            "intersection",
            "anyhit",
            "closesthit",
            "miss",
            "callable",
            "mesh",
            "object",
            "amplification",
        }
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "char": "int8",
            "uchar": "uint8",
            "short": "int16",
            "ushort": "uint16",
            "int8_t": "int8",
            "uint8_t": "uint8",
            "int16_t": "int16",
            "uint16_t": "uint16",
            "int32_t": "int",
            "uint32_t": "uint",
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
            "atomic_int": "atomic_int",
            "atomic_uint": "atomic_uint",
            "atomic_bool": "atomic_bool",
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
            # Packed vector types
            "packed_float2": "vec2",
            "packed_float3": "vec3",
            "packed_float4": "vec4",
            "packed_half2": "f16vec2",
            "packed_half3": "f16vec3",
            "packed_half4": "f16vec4",
            "packed_int2": "ivec2",
            "packed_int3": "ivec3",
            "packed_int4": "ivec4",
            "packed_uint2": "uvec2",
            "packed_uint3": "uvec3",
            "packed_uint4": "uvec4",
            # SIMD types
            "simd_float2": "vec2",
            "simd_float3": "vec3",
            "simd_float4": "vec4",
            "simd_float2x2": "mat2",
            "simd_float3x3": "mat3",
            "simd_float4x4": "mat4",
            "simd_int2": "ivec2",
            "simd_int3": "ivec3",
            "simd_int4": "ivec4",
            "simd_uint2": "uvec2",
            "simd_uint3": "uvec3",
            "simd_uint4": "uvec4",
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
            "texture1d_array": "sampler1DArray",
            "texture1d_array<float>": "sampler1DArray",
            "texture1d_array<half>": "sampler1DArray",
            "texture1d_array<int>": "isampler1DArray",
            "texture1d_array<uint>": "usampler1DArray",
            "texture2d": "sampler2D",
            "texture2d<float>": "sampler2D",
            "texture2d<half>": "sampler2D",
            "texture2d<int>": "isampler2D",
            "texture2d<uint>": "usampler2D",
            "texture2d_ms": "sampler2DMS",
            "texture2d_ms<float>": "sampler2DMS",
            "texture2d_ms<half>": "sampler2DMS",
            "texture2d_ms<int>": "isampler2DMS",
            "texture2d_ms<uint>": "usampler2DMS",
            "texture2d_ms_array": "sampler2DMSArray",
            "texture2d_ms_array<float>": "sampler2DMSArray",
            "texture2d_ms_array<half>": "sampler2DMSArray",
            "texture2d_ms_array<int>": "isampler2DMSArray",
            "texture2d_ms_array<uint>": "usampler2DMSArray",
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
            "texturecube_array": "samplerCubeArray",
            "texturecube_array<float>": "samplerCubeArray",
            "texturecube_array<half>": "samplerCubeArray",
            "texturecube_array<int>": "isamplerCubeArray",
            "texturecube_array<uint>": "usamplerCubeArray",
            "texture2d_array": "sampler2DArray",
            "texture2d_array<float>": "sampler2DArray",
            "texture2d_array<half>": "sampler2DArray",
            "texture2d_array<int>": "isampler2DArray",
            "texture2d_array<uint>": "usampler2DArray",
            "texture_buffer": "samplerBuffer",
            "texture_buffer<float>": "samplerBuffer",
            "texture_buffer<half>": "samplerBuffer",
            "texture_buffer<int>": "isamplerBuffer",
            "texture_buffer<uint>": "usamplerBuffer",
            "depth2d": "sampler2DShadow",
            "depth2d<float>": "sampler2DShadow",
            "depth2d_array": "sampler2DArrayShadow",
            "depth2d_array<float>": "sampler2DArrayShadow",
            "depthcube": "samplerCubeShadow",
            "depthcube<float>": "samplerCubeShadow",
            "depthcube_array": "samplerCubeArrayShadow",
            "depthcube_array<float>": "samplerCubeArrayShadow",
            "depth2d_ms": "sampler2DMS",
            "depth2d_ms<float>": "sampler2DMS",
            "depth2d_ms_array": "sampler2DMSArray",
            "depth2d_ms_array<float>": "sampler2DMSArray",
            "acceleration_structure": "acceleration_structure",
            "intersection_function_table": "intersection_function_table",
            "visible_function_table": "visible_function_table",
            "indirect_command_buffer": "indirect_command_buffer",
            "ray": "ray",
            "ray_data": "ray_data",
            "intersection_result": "intersection_result",
            "intersection_params": "intersection_params",
            "triangle_intersection_params": "triangle_intersection_params",
            "intersector": "intersector",
            # Sampler type
            "sampler": "sampler",
        }

        self.map_semantics = {
            # Vertex attributes
            "attribute(0)": "POSITION",
            "attribute(1)": "NORMAL",
            "attribute(2)": "TANGENT",
            "attribute(3)": "BINORMAL",
            "attribute(4)": "TEXCOORD",
            "attribute(5)": "TEXCOORD0",
            "attribute(6)": "TEXCOORD1",
            "attribute(7)": "TEXCOORD2",
            "attribute(8)": "TEXCOORD3",
            "attribute(9)": "COLOR",
            "attribute(10)": "COLOR0",
            "attribute(11)": "COLOR1",
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
            "sample_id": "gl_SampleID",
            "sample_mask": "gl_SampleMask",
            "primitive_id": "gl_PrimitiveID",
            "viewport_array_index": "gl_ViewportIndex",
            "render_target_array_index": "gl_Layer",
            "thread_position_in_grid": "gl_GlobalInvocationID",
            "thread_position_in_threadgroup": "gl_LocalInvocationID",
            "threadgroup_position_in_grid": "gl_WorkGroupID",
            "thread_index_in_threadgroup": "gl_LocalInvocationIndex",
            "stage_in": "",
        }

    def generate(self, ast):
        code = ""
        includes = getattr(ast, "includes", []) or []
        for inc in includes:
            if isinstance(inc, PreprocessorNode):
                line = f"{inc.directive} {inc.content}".strip()
            else:
                line = str(inc).strip()
            if line:
                code += f"{line}\n"
        if includes:
            code += "\n"
        code += "shader main {\n"
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
        enums = getattr(ast, "enums", []) or []
        typedefs = getattr(ast, "typedefs", []) or []

        if typedefs:
            code += "    // Typedefs\n"
            for alias in typedefs:
                if isinstance(alias, TypeAliasNode):
                    code += (
                        f"    typedef {self.map_type(alias.alias_type)} {alias.name};\n"
                    )
            code += "\n"

        if enums:
            code += "    // Enums\n"
            for enum in enums:
                if isinstance(enum, EnumNode):
                    code += f"    enum {enum.name} {{\n"
                    for member_name, member_value in enum.members:
                        if member_value is not None:
                            value = self.generate_expression(member_value, False)
                            code += f"        {member_name} = {value},\n"
                        else:
                            code += f"        {member_name},\n"
                    code += "    };\n\n"
        for struct_node in structs:
            if isinstance(struct_node, StructNode):
                if struct_node.name in self.constant_struct_name:
                    code += "    // cbuffers\n"
                    code += f"    cbuffer {struct_node.name} {{\n"
                else:
                    code += "    // Structs\n"
                    struct_alignas = ""
                    if hasattr(struct_node, "alignas") and struct_node.alignas:
                        parts = []
                        for item in struct_node.alignas:
                            if isinstance(item, tuple) and item[0] == "type":
                                parts.append(f"alignas({self.map_type(item[1])})")
                            else:
                                parts.append(
                                    f"alignas({self.generate_expression(item, False)})"
                                )
                        struct_alignas = " ".join(parts) + " "
                    code += f"    {struct_alignas}struct {struct_node.name} {{\n"
                for member in struct_node.members:
                    decl = self.format_decl(member, include_semantic=True)
                    code += f"        {decl};\n"
                code += "    }\n\n"

        globals_list = getattr(ast, "global_variables", []) or getattr(
            ast, "global_vars", []
        )
        if globals_list:
            code += "    // Globals\n"
            for glob in globals_list:
                if isinstance(glob, StaticAssertNode):
                    cond = self.generate_expression(glob.condition, False)
                    if glob.message is not None:
                        msg = (
                            glob.message
                            if isinstance(glob.message, str)
                            else self.generate_expression(glob.message, False)
                        )
                        code += f"    static_assert({cond}, {msg});\n"
                    else:
                        code += f"    static_assert({cond});\n"
                    continue
                if isinstance(glob, AssignmentNode):
                    left = self.generate_expression(glob.left, False)
                    right = self.generate_expression(glob.right, False)
                    code += f"    {left} {glob.operator} {right};\n"
                elif isinstance(glob, VariableNode):
                    decl = self.format_decl(glob, include_semantic=True)
                    code += f"    {decl};\n"
            code += "\n"

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
            elif qualifier in self.rt_qualifiers:
                code += f"    // {qualifier} function\n"
                code += self.generate_function(f)
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

    def format_array_suffix(self, var):
        if not hasattr(var, "array_sizes") or not var.array_sizes:
            return ""
        suffix = ""
        for size in var.array_sizes:
            if size is None:
                suffix += "[]"
            else:
                suffix += f"[{self.generate_expression(size, False)}]"
        return suffix

    def format_decl(self, var, include_semantic=False):
        alignas_prefix = ""
        if hasattr(var, "alignas") and var.alignas:
            parts = []
            for item in var.alignas:
                if isinstance(item, tuple) and item[0] == "type":
                    parts.append(f"alignas({self.map_type(item[1])})")
                else:
                    parts.append(f"alignas({self.generate_expression(item, False)})")
            alignas_prefix = " ".join(parts) + " "
        type_str = f"{self.map_type(var.vtype)}{self.format_array_suffix(var)}"
        const_str = "const " if hasattr(var, "is_const") and var.is_const else ""
        semantic = (
            self.map_semantic(getattr(var, "attributes", None))
            if include_semantic
            else ""
        )
        parts = [alignas_prefix + const_str + type_str, var.name]
        if semantic:
            parts.append(semantic)
        return " ".join(part for part in parts if part)

    def generate_function(self, func, indent=2):
        code = ""
        code += "    " * indent
        params = ", ".join(
            self.format_decl(p, include_semantic=True) for p in func.params
        )
        fn_semantic = self.map_semantic(func.attributes)
        suffix = f" {fn_semantic}" if fn_semantic else ""
        code += f"{self.map_type(func.return_type)} {func.name}({params}){suffix} {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    }\n\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                decl = self.format_decl(stmt, include_semantic=False)
                code += f"{decl};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"
            elif isinstance(stmt, ReturnNode):
                if not is_main:
                    if stmt.value is None:
                        code += "return;\n"
                    else:
                        code += (
                            f"return {self.generate_expression(stmt.value, is_main)};\n"
                        )
            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, ForNode):
                code += self.generate_for_loop(stmt, indent, is_main)
            elif isinstance(stmt, WhileNode):
                code += self.generate_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, DoWhileNode):
                code += self.generate_do_while_loop(stmt, indent, is_main)
            elif isinstance(stmt, IfNode):
                code += self.generate_if_statement(stmt, indent, is_main)
            elif isinstance(stmt, SwitchNode):
                code += self.generate_switch_statement(stmt, indent, is_main)
            elif (
                isinstance(stmt, FunctionCallNode)
                or isinstance(stmt, MethodCallNode)
                or isinstance(stmt, CallNode)
            ):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, PostfixOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, DiscardNode):
                code += "discard;\n"
            elif isinstance(stmt, StaticAssertNode):
                cond = self.generate_expression(stmt.condition, is_main)
                if stmt.message is not None:
                    msg = (
                        stmt.message
                        if isinstance(stmt.message, str)
                        else self.generate_expression(stmt.message, is_main)
                    )
                    code += f"static_assert({cond}, {msg});\n"
                else:
                    code += f"static_assert({cond});\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                expr = self.generate_expression(stmt, is_main)
                if expr:
                    code += f"{expr};\n"
                else:
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
        code += "    " * indent + f"}} while ({condition});\n"
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
                return f"{const_str}{self.map_type(expr.vtype)}{self.format_array_suffix(expr)} {expr.name}"
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
        elif isinstance(expr, CallNode):
            callee = self.generate_expression(expr.callee, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{callee}({args})"
        elif isinstance(expr, MethodCallNode):
            obj = self.generate_expression(expr.object, is_main)
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            method = expr.method
            if method == "read":
                return f"textureLoad({obj}, {args})" if args else f"{obj}.read()"
            if method == "write":
                return f"textureStore({obj}, {args})" if args else f"{obj}.write()"
            if method == "sample_compare":
                return f"textureCompare({obj}, {args})"
            if method == "sample_compare_level":
                return f"textureCompareLod({obj}, {args})"
            if method == "gather":
                return f"textureGather({obj}, {args})"
            if method == "gather_compare":
                return f"textureGatherCompare({obj}, {args})"
            return f"{obj}.{method}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"({expr.op}{operand})"
        elif isinstance(expr, PostfixOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            return f"{operand}{expr.op}"
        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"
        elif isinstance(expr, CastNode):
            return f"({self.map_type(expr.target_type)}){self.generate_expression(expr.expression, is_main)}"
        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, TextureSampleNode):
            texture = self.generate_expression(expr.texture, is_main)
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
        if not metal_type:
            return metal_type

        base = metal_type.strip()
        if base.startswith("metal::"):
            base = base.split("metal::", 1)[1]
        if base.startswith("raytracing::"):
            base = base.split("raytracing::", 1)[1]
        suffix = ""
        while base.endswith("*") or base.endswith("&"):
            suffix = base[-1] + suffix
            base = base[:-1].strip()

        # Normalize generic access qualifiers: texture2d<float, access::read_write>
        if "<" in base and ">" in base:
            base_name, inner = base.split("<", 1)
            inner = inner.rstrip(">")
            if "," in inner:
                inner = inner.split(",", 1)[0].strip()
            base = f"{base_name}<{inner.strip()}>"

        mapped = self.type_map.get(base, base)
        return f"{mapped}{suffix}"

    def map_semantic(self, semantic):
        if not semantic:
            return ""

        outputs = []
        for attr in semantic:
            if not isinstance(attr, AttributeNode):
                continue
            name = attr.name
            args = [str(a).strip() for a in attr.args] if attr.args else []
            key = f"{name}({args[0]})" if args else name
            out = self.map_semantics.get(key, self.map_semantics.get(name, None))
            if out is None:
                if args:
                    out = f"{name}({', '.join(args)})"
                else:
                    out = name
            if out:
                outputs.append(f"@{out}")
        return " ".join(outputs)

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
