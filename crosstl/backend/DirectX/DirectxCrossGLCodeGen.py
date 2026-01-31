from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *
from ..common_ast import ArrayAccessNode, BreakNode, CastNode, ContinueNode, TextureSampleNode


class HLSLToCrossGLConverter:
    def __init__(self):
        self.type_map = {
            # Scalar Types
            "void": "void",
            "bool": "bool",
            "int": "int",
            "uint": "uint",
            "dword": "uint",
            "float": "float",
            "half": "float16",
            "double": "double",
            "min16float": "float16",
            "min10float": "float16",
            "min16int": "int16",
            "min12int": "int16",
            "min16uint": "uint16",
            "int64_t": "int64",
            "uint64_t": "uint64",
            # Vector Types - float
            "float2": "vec2",
            "float3": "vec3",
            "float4": "vec4",
            # Vector Types - half
            "half2": "f16vec2",
            "half3": "f16vec3",
            "half4": "f16vec4",
            # Vector Types - double
            "double2": "dvec2",
            "double3": "dvec3",
            "double4": "dvec4",
            # Vector Types - int
            "int2": "ivec2",
            "int3": "ivec3",
            "int4": "ivec4",
            # Vector Types - uint
            "uint2": "uvec2",
            "uint3": "uvec3",
            "uint4": "uvec4",
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
            # Matrix Types - double
            "double2x2": "dmat2",
            "double2x3": "dmat2x3",
            "double2x4": "dmat2x4",
            "double3x2": "dmat3x2",
            "double3x3": "dmat3",
            "double3x4": "dmat3x4",
            "double4x2": "dmat4x2",
            "double4x3": "dmat4x3",
            "double4x4": "dmat4",
            # Texture Types
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "Texture2DArray": "sampler2DArray",
            "TextureCubeArray": "samplerCubeArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "FeedbackTexture2D": "feedbackTexture2D",
            "FeedbackTexture2DArray": "feedbackTexture2DArray",
            # RW Texture Types (for compute shaders)
            "RWTexture1D": "image1D",
            "RWTexture1DArray": "image1DArray",
            "RWTexture2D": "image2D",
            "RWTexture2DArray": "image2DArray",
            "RWTexture2DMS": "image2DMS",
            "RWTexture2DMSArray": "image2DMSArray",
            "RWTexture3D": "image3D",
            "RWTextureCube": "imageCube",
            "RWTextureCubeArray": "imageCubeArray",
            # Buffer Types
            "Buffer": "samplerBuffer",
            "RWBuffer": "imageBuffer",
            "StructuredBuffer": "buffer",
            "RWStructuredBuffer": "buffer",
            "ByteAddressBuffer": "buffer",
            "RWByteAddressBuffer": "buffer",
            "AppendStructuredBuffer": "buffer",
            "ConsumeStructuredBuffer": "buffer",
            "RaytracingAccelerationStructure": "accelerationStructure",
            "RayQuery": "rayQuery",
            "InputPatch": "inputPatch",
            "OutputPatch": "outputPatch",
            "PointStream": "pointStream",
            "LineStream": "lineStream",
            "TriangleStream": "triangleStream",
            # Sampler Types
            "SamplerState": "sampler",
            "SamplerComparisonState": "samplerShadow",
        }
        self.function_map = {
            "lerp": "mix",
            "rsqrt": "inverseSqrt",
        }
        self.interlocked_map = {
            "InterlockedAdd": "atomicAdd",
            "InterlockedAnd": "atomicAnd",
            "InterlockedOr": "atomicOr",
            "InterlockedXor": "atomicXor",
            "InterlockedMin": "atomicMin",
            "InterlockedMax": "atomicMax",
            "InterlockedExchange": "atomicExchange",
            "InterlockedCompareExchange": "atomicCompareExchange",
        }
        self.texture_method_map = {
            "Sample": "texture_sample",
            "SampleLevel": "texture_sample_level",
            "SampleGrad": "texture_sample_grad",
            "SampleBias": "texture_sample_bias",
            "SampleCmp": "texture_sample_cmp",
            "SampleCmpLevelZero": "texture_sample_cmp_level_zero",
            "Load": "texture_load",
            "Gather": "texture_gather",
            "GatherRed": "texture_gather_red",
            "GatherGreen": "texture_gather_green",
            "GatherBlue": "texture_gather_blue",
            "GatherAlpha": "texture_gather_alpha",
            "GetDimensions": "texture_dimensions",
        }
        self.buffer_method_map = {
            "Load": "buffer_load",
            "Store": "buffer_store",
            "Append": "buffer_append",
            "Consume": "buffer_consume",
            "GetDimensions": "buffer_dimensions",
        }
        self.semantic_map = {
            # System-value semantics - Vertex inputs
            "SV_VertexID": "gl_VertexID",
            "SV_InstanceID": "gl_InstanceID",
            "SV_PrimitiveID": "gl_PrimitiveID",
            # System-value semantics - Vertex outputs
            "SV_POSITION": "gl_Position",
            "SV_Position": "gl_Position",
            "SV_ClipDistance": "gl_ClipDistance",
            "SV_CullDistance": "gl_CullDistance",
            # System-value semantics - Fragment inputs
            "SV_IsFrontFace": "gl_FrontFacing",
            "SV_SampleIndex": "gl_SampleID",
            "SV_Coverage": "gl_SampleMask",
            # System-value semantics - Fragment outputs
            "SV_TARGET": "gl_FragColor",
            "SV_Target": "gl_FragColor",
            "SV_TARGET0": "gl_FragData[0]",
            "SV_Target0": "gl_FragData[0]",
            "SV_TARGET1": "gl_FragData[1]",
            "SV_Target1": "gl_FragData[1]",
            "SV_TARGET2": "gl_FragData[2]",
            "SV_Target2": "gl_FragData[2]",
            "SV_TARGET3": "gl_FragData[3]",
            "SV_Target3": "gl_FragData[3]",
            "SV_TARGET4": "gl_FragData[4]",
            "SV_Target4": "gl_FragData[4]",
            "SV_TARGET5": "gl_FragData[5]",
            "SV_Target5": "gl_FragData[5]",
            "SV_TARGET6": "gl_FragData[6]",
            "SV_Target6": "gl_FragData[6]",
            "SV_TARGET7": "gl_FragData[7]",
            "SV_Target7": "gl_FragData[7]",
            "SV_DEPTH": "gl_FragDepth",
            "SV_Depth": "gl_FragDepth",
            "SV_DepthGreaterEqual": "gl_FragDepth",
            "SV_DepthLessEqual": "gl_FragDepth",
            # System-value semantics - Compute shader
            "SV_GroupID": "gl_WorkGroupID",
            "SV_GroupThreadID": "gl_LocalInvocationID",
            "SV_DispatchThreadID": "gl_GlobalInvocationID",
            "SV_GroupIndex": "gl_LocalInvocationIndex",
            # Geometry shader semantics
            "SV_GSInstanceID": "gl_InvocationID",
            "SV_RenderTargetArrayIndex": "gl_Layer",
            "SV_ViewportArrayIndex": "gl_ViewportIndex",
            # Tessellation semantics
            "SV_OutputControlPointID": "gl_InvocationID",
            "SV_TessFactor": "gl_TessLevelOuter",
            "SV_InsideTessFactor": "gl_TessLevelInner",
            "SV_DomainLocation": "gl_TessCoord",
            # Mesh/Task semantics
            "SV_ViewID": "gl_ViewID",
            "SV_DispatchMeshID": "mesh_DispatchMeshID",
            # Raytracing semantics
            "SV_RayFlags": "rt_RayFlags",
            "SV_CullMask": "rt_CullMask",
            "SV_ObjectRayOrigin": "rt_ObjectRayOrigin",
            "SV_ObjectRayDirection": "rt_ObjectRayDirection",
            "SV_WorldRayOrigin": "rt_WorldRayOrigin",
            "SV_WorldRayDirection": "rt_WorldRayDirection",
            "SV_RayTMin": "rt_RayTMin",
            "SV_RayTCurrent": "rt_RayTCurrent",
            "SV_HitKind": "rt_HitKind",
            "SV_InstanceIndex": "rt_InstanceIndex",
            "SV_PrimitiveIndex": "rt_PrimitiveIndex",
            "SV_GeometryIndex": "rt_GeometryIndex",
            "SV_RayContributionToHitGroupIndex": "rt_RayContributionToHitGroupIndex",
            "SV_ShaderIndex": "rt_ShaderIndex",
            # Legacy semantics
            "FRONT_FACE": "gl_FrontFacing",
            "PRIMITIVE_ID": "gl_PrimitiveID",
            "INSTANCE_ID": "gl_InstanceID",
            "VERTEX_ID": "gl_VertexID",
            # User-defined semantics
            "POSITION": "Position",
            "POSITION0": "Position",
            "NORMAL": "Normal",
            "NORMAL0": "Normal",
            "TANGENT": "Tangent",
            "TANGENT0": "Tangent",
            "BINORMAL": "Binormal",
            "BINORMAL0": "Binormal",
            "TEXCOORD": "TexCoord",
            "TEXCOORD0": "TexCoord0",
            "TEXCOORD1": "TexCoord1",
            "TEXCOORD2": "TexCoord2",
            "TEXCOORD3": "TexCoord3",
            "TEXCOORD4": "TexCoord4",
            "TEXCOORD5": "TexCoord5",
            "TEXCOORD6": "TexCoord6",
            "TEXCOORD7": "TexCoord7",
            "COLOR": "Color",
            "COLOR0": "Color0",
            "COLOR1": "Color1",
            "BLENDWEIGHT": "BlendWeight",
            "BLENDINDICES": "BlendIndices",
            "PSIZE": "PointSize",
            "FOG": "Fog",
        }
        self.bitwise_op_map = {
            "&": "bitAnd",
            "|": "bitOr",
            "^": "bitXor",
            "~": "bitNot",
            "<<": "bitShiftLeft",
            ">>": "bitShiftRight",
        }
        self.indentation = 0
        self.code = []

    def get_indent(self):
        return "    " * self.indentation

    def format_array_suffixes(self, node, is_main=False):
        sizes = getattr(node, "array_sizes", None)
        if not sizes:
            return ""
        parts = []
        for size in sizes:
            if size is None:
                parts.append("[]")
            else:
                parts.append(f"[{self.generate_expression(size, is_main)}]")
        return "".join(parts)

    def format_attributes(self, attributes, indent):
        if not attributes:
            return ""
        lines = ""
        for attr in attributes:
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                rendered_args = ", ".join(self.generate_expression(arg) for arg in args)
                lines += "    " * indent + f"@ {attr.name}({rendered_args})\n"
            else:
                lines += "    " * indent + f"@ {attr.name}\n"
        return lines

    def format_binding_attributes(self, node, indent):
        lines = ""
        register = getattr(node, "register", None)
        packoffset = getattr(node, "packoffset", None)
        if register:
            parts = [part.strip() for part in str(register).split(",") if part.strip()]
            rendered = ", ".join(parts)
            lines += "    " * indent + f"@ register({rendered})\n"
        if packoffset:
            lines += "    " * indent + f"@ packoffset({packoffset})\n"
        return lines

    def visit(self, node):
        # Special case for SwitchStatementNode and SwitchCaseNode
        if isinstance(node, SwitchStatementNode):
            return self.visit_SwitchStatementNode(node)
        elif isinstance(node, SwitchCaseNode):
            return self.visit_SwitchCaseNode(node)
        elif isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)

        # For other node types, use existing methods
        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        code = "shader main {\n"
        typedefs = getattr(ast, "typedefs", []) or []
        enums = getattr(ast, "enums", []) or []
        if typedefs:
            for alias in typedefs:
                alias_type = getattr(alias, "alias_type", None) or getattr(
                    alias, "original_type", None
                )
                if alias_type is not None:
                    code += f"    typedef {self.map_type(alias_type)} {alias.name};\n"
        if enums:
            for enum in enums:
                if isinstance(enum, EnumNode):
                    code += f"    enum {enum.name} {{\n"
                    for member_name, member_value in enum.members:
                        if member_value is None:
                            code += f"        {member_name},\n"
                        else:
                            code += (
                                f"        {member_name} = "
                                f"{self.generate_expression(member_value)},\n"
                            )
                    code += "    }\n"
        # Generate structs
        for node in ast.structs:
            if isinstance(node, StructNode):
                code += f"    struct {node.name} {{\n"
                for member in node.members:
                    array_suffix = self.format_array_suffixes(member)
                    semantic = self.map_semantic(member.semantic)
                    semantic = f" {semantic}" if semantic else ""
                    code += (
                        f"        {self.map_type(member.vtype)} "
                        f"{member.name}{array_suffix}{semantic};\n"
                    )
                code += "    }\n"
            elif isinstance(node, PragmaNode):
                code += f"    #pragma {node.directive} {node.value};\n"
            elif isinstance(node, IncludeNode):
                code += f"    #include {node.path}\n"
        # Generate global variables
        for node in ast.global_variables:
            code += self.format_attributes(getattr(node, "attributes", []), 1)
            code += self.format_binding_attributes(node, 1)
            array_suffix = self.format_array_suffixes(node)
            code += f"    {self.map_type(node.vtype)} {node.name}{array_suffix};\n"
        # Generate cbuffers
        if ast.cbuffers:
            code += "    // Constant Buffers\n"
            code += self.generate_cbuffers(ast)

        stage_map = {
            "vertex": "vertex",
            "fragment": "fragment",
            "compute": "compute",
            "geometry": "geometry",
            "tessellation_control": "tessellation_control",
            "tessellation_evaluation": "tessellation_evaluation",
            "mesh": "mesh",
            "task": "task",
            "ray_generation": "ray_generation",
            "ray_intersection": "ray_intersection",
            "ray_closest_hit": "ray_closest_hit",
            "ray_miss": "ray_miss",
            "ray_any_hit": "ray_any_hit",
            "ray_callable": "ray_callable",
        }
        # Generate custom functions
        for func in ast.functions:
            stage_name = stage_map.get(func.qualifier)
            if stage_name:
                code += f"    // {stage_name} Shader\n"
                code += f"    {stage_name} {{\n"
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
                code += self.format_binding_attributes(node, 1)
                code += f"    cbuffer {node.name} {{\n"
                for member in node.members:
                    array_suffix = self.format_array_suffixes(member)
                    code += (
                        f"        {self.map_type(member.vtype)} "
                        f"{member.name}{array_suffix};\n"
                    )
                code += "    }\n"
        return code

    def generate_function(self, func, indent=1):
        code = self.format_attributes(getattr(func, "attributes", []), indent)
        code += "    " * indent
        params = ", ".join(
            f"{self.map_type(p.vtype)} {p.name}{self.format_array_suffixes(p)}"
            f"{(' ' + self.map_semantic(p.semantic)) if self.map_semantic(p.semantic) else ''}"
            for p in func.params
        )
        semantic = self.map_semantic(func.semantic)
        semantic = f" {semantic}" if semantic else ""
        code += f"{self.map_type(func.return_type)} {func.name}({params}){semantic} {{\n"
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    " * indent + "}\n\n"
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                array_suffix = self.format_array_suffixes(stmt, is_main)
                if stmt.value is not None:
                    value = self.generate_expression(stmt.value, is_main)
                    code += (
                        f"{self.map_type(stmt.vtype)} {stmt.name}{array_suffix} = "
                        f"{value};\n"
                    )
                else:
                    code += f"{self.map_type(stmt.vtype)} {stmt.name}{array_suffix};\n"
            elif isinstance(stmt, AssignmentNode):
                code += self.generate_assignment(stmt, is_main) + ";\n"

            elif isinstance(stmt, BinaryOpNode):
                code += f"{self.generate_expression(stmt.left, is_main)} {stmt.op} {self.generate_expression(stmt.right, is_main)};\n"
            elif isinstance(stmt, UnaryOpNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, ReturnNode):
                if stmt.value is None:
                    code += "return;\n"
                elif not is_main:
                    code += f"return {self.generate_expression(stmt.value, is_main)};\n"
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
            elif isinstance(stmt, BreakNode):
                code += "break;\n"
            elif isinstance(stmt, ContinueNode):
                code += "continue;\n"
            elif isinstance(stmt, FunctionCallNode):
                code += f"{self.generate_expression(stmt, is_main)};\n"
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                # For any unhandled statement type
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def format_float(self, value: float) -> str:
        text = format(value, ".10f")
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-0"):
            text = "0"
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text

    def maybe_parenthesize(self, expr, rendered: str) -> str:
        if isinstance(expr, (BinaryOpNode, TernaryOpNode, AssignmentNode)):
            return f"({rendered})"
        return rendered

    def generate_for_loop(self, node, indent, is_main):
        if isinstance(node.init, VariableNode):
            array_suffix = self.format_array_suffixes(node.init, is_main)
            init = f"{self.map_type(node.init.vtype)} {node.init.name}{array_suffix}"
            if node.init.value is not None:
                init += f" = {self.generate_expression(node.init.value, is_main)}"
        elif node.init is None:
            init = ""
        else:
            init = self.generate_expression(node.init, is_main)

        condition = (
            self.generate_expression(node.condition, is_main)
            if node.condition is not None
            else ""
        )
        update = (
            self.generate_expression(node.update, is_main) if node.update is not None else ""
        )

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

        code = "while (true) {\n"
        code += self.generate_function_body(node.body, indent + 1, is_main)
        code += "    " * (indent + 1) + f"if (!({condition})) {{\n"
        code += "    " * (indent + 2) + "break;\n"
        code += "    " * (indent + 1) + "}\n"
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
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            left = self.maybe_parenthesize(expr.left, left)
            right = self.maybe_parenthesize(expr.right, right)
            return f"{left} {expr.op} {right}"

        elif isinstance(expr, AssignmentNode):
            left = self.generate_expression(expr.left, is_main)
            right = self.generate_expression(expr.right, is_main)
            op = expr.operator
            return f"{left} {op} {right}"

        elif isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand, is_main)
            operand = self.maybe_parenthesize(expr.operand, operand)
            if getattr(expr, "is_postfix", False):
                return f"{operand}{expr.op}"
            return f"{expr.op}{operand}"
        elif isinstance(expr, FunctionCallNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            if isinstance(expr.name, MemberAccessNode):
                obj = self.generate_expression(expr.name.object, is_main)
                member = expr.name.member
                if member == "Load":
                    if len(expr.args) <= 1:
                        return f"{self.buffer_method_map['Load']}({obj}, {args})"
                    return f"{self.texture_method_map['Load']}({obj}, {args})"
                if member == "GetDimensions":
                    if len(expr.args) <= 1:
                        return f"{self.buffer_method_map['GetDimensions']}({obj}, {args})"
                    return f"{self.texture_method_map['GetDimensions']}({obj}, {args})"
                if member in self.texture_method_map:
                    return f"{self.texture_method_map[member]}({obj}, {args})"
                if member in self.buffer_method_map:
                    return f"{self.buffer_method_map[member]}({obj}, {args})"
                return f"{obj}.{member}({args})"

            func_name = (
                expr.name
                if isinstance(expr.name, str)
                else self.generate_expression(expr.name, is_main)
            )
            if func_name == "saturate":
                if expr.args:
                    return f"clamp({self.generate_expression(expr.args[0], is_main)}, 0.0, 1.0)"
                return "clamp(0.0, 0.0, 1.0)"
            func_name = self.function_map.get(func_name, func_name)
            func_name = self.interlocked_map.get(func_name, func_name)
            return f"{func_name}({args})"
        elif isinstance(expr, MemberAccessNode):
            obj = self.generate_expression(expr.object, is_main)
            return f"{obj}.{expr.member}"
        elif isinstance(expr, ArrayAccessNode):
            array = self.generate_expression(expr.array, is_main)
            index = self.generate_expression(expr.index, is_main)
            return f"{array}[{index}]"
        elif isinstance(expr, CastNode):
            target_type = self.map_type(expr.target_type)
            expression = self.generate_expression(expr.expression, is_main)
            return f"{target_type}({expression})"
        elif isinstance(expr, TextureSampleNode):
            texture = self.generate_expression(expr.texture, is_main)
            sampler = self.generate_expression(expr.sampler, is_main)
            coords = self.generate_expression(expr.coordinates, is_main)
            if getattr(expr, "lod", None) is not None:
                lod = self.generate_expression(expr.lod, is_main)
                return f"texture_sample_level({texture}, {sampler}, {coords}, {lod})"
            return f"texture_sample({texture}, {sampler}, {coords})"

        elif isinstance(expr, TernaryOpNode):
            return f"{self.generate_expression(expr.condition, is_main)} ? {self.generate_expression(expr.true_expr, is_main)} : {self.generate_expression(expr.false_expr, is_main)}"

        elif isinstance(expr, VectorConstructorNode):
            args = ", ".join(
                self.generate_expression(arg, is_main) for arg in expr.args
            )
            return f"{self.map_type(expr.type_name)}({args})"
        elif isinstance(expr, bool):
            return "true" if expr else "false"
        elif isinstance(expr, float):
            return self.format_float(expr)
        elif isinstance(expr, int):
            return str(expr)
        else:
            return str(expr)

    def map_type(self, hlsl_type):
        if not hlsl_type:
            return hlsl_type
        type_name = hlsl_type
        if "<" in type_name and type_name.endswith(">"):
            base, _ = type_name.split("<", 1)
            type_name = base
        return self.type_map.get(type_name, type_name)

    def map_semantic(self, semantic):
        if not semantic:
            return ""
        mapped = self.semantic_map.get(semantic)
        if mapped is None and isinstance(semantic, str):
            mapped = self.semantic_map.get(semantic.upper())
        mapped = mapped or semantic
        return f"@ {mapped}"

    def generate_switch_statement(self, node, indent=1, is_main=False):
        # Support both 'condition' and 'expression' attributes for compatibility
        expression = getattr(node, "expression", None) or getattr(
            node, "condition", None
        )
        code = (
            "    " * indent
            + f"switch ({self.generate_expression(expression, is_main)}) {{\n"
        )

        for case in node.cases:
            code += (
                "    " * (indent + 1)
                + f"case {self.generate_expression(case.value, is_main)}:\n"
            )
            # Support both 'body' and 'statements' attributes
            case_body = getattr(case, "body", None) or getattr(case, "statements", [])
            code += self.generate_function_body(case_body, indent + 2, is_main)

        # Support multiple attribute names for default case
        default_body = (
            getattr(node, "default_body", None)
            or getattr(node, "default_case", None)
            or getattr(node, "default", None)
        )
        if default_body:
            code += "    " * (indent + 1) + "default:\n"
            code += self.generate_function_body(default_body, indent + 2, is_main)

        code += "    " * indent + "}\n"
        return code

    def visit_BinaryOpNode(self, node):
        if hasattr(node.left, "visit"):
            left = node.visit_child(self, node.left)
        else:
            left = self.generate_expression(node.left)

        if hasattr(node.right, "visit"):
            right = node.visit_child(self, node.right)
        else:
            right = self.generate_expression(node.right)

        # Handle bitwise operations based on token value
        if hasattr(node.op, "token_type"):
            if node.op.token_type in ("BITWISE_AND", "AMPERSAND", "&"):
                return f"({left} & {right})"
            elif node.op.token_type in ("BITWISE_OR", "PIPE", "|"):
                return f"({left} | {right})"
            elif node.op.token_type in ("BITWISE_XOR", "CARET", "^"):
                return f"({left} ^ {right})"
        elif hasattr(node.op, "value"):
            # Handle string values
            if node.op.value in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op.value in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op.value in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"
        elif isinstance(node.op, str):
            # Direct string comparison
            if node.op in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"

        # Falls back to string representation of the operator
        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        return f"{left} {op_str} {right}"

    def visit_UnaryOpNode(self, node):
        if hasattr(node, "expr"):
            expr_node = node.expr
        else:
            expr_node = node.operand

        if hasattr(expr_node, "visit"):
            expr = node.visit_child(self, expr_node)
        else:
            expr = self.generate_expression(expr_node)

        # Handle bitwise NOT based on token type or value
        if hasattr(node.op, "token_type") and node.op.token_type in (
            "BITWISE_NOT",
            "TILDE",
            "~",
        ):
            return f"(~{expr})"
        elif hasattr(node.op, "value") and node.op.value in (
            "~",
            "BITWISE_NOT",
            "TILDE",
        ):
            return f"(~{expr})"
        elif isinstance(node.op, str) and node.op in ("~", "BITWISE_NOT", "TILDE"):
            return f"(~{expr})"

        # Falls back to string representation of the operator
        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        if getattr(node, "is_postfix", False):
            return f"{expr}{op_str}"
        return f"{op_str}{expr}"

    def visit_SwitchStatementNode(self, node):
        # Handle the alternative SwitchStatementNode type if needed
        return self.visit_SwitchNode(node)

    def visit_SwitchCaseNode(self, node):
        # Handle the alternative SwitchCaseNode type if needed
        return self.visit_CaseNode(node)

    def visit_StructNode(self, node):
        # Generate code for a struct definition
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = ""
            if member.semantic:
                semantic = f" {self.map_semantic(member.semantic)}"

            array_suffix = self.format_array_suffixes(member)
            code += (
                self.get_indent()
                + f"{self.map_type(member.vtype)} {member.name}{array_suffix}{semantic};\n"
            )

        self.indentation -= 1
        code += self.get_indent() + "}\n"
        return code

    def visit_SwitchNode(self, node):
        # Generate the switch statement code
        condition = self.generate_expression(node.condition)
        code = f"switch ({condition}) {{\n"

        # Generate case statements
        for case in node.cases:
            code += self.visit_CaseNode(case)

        # Generate default case if exists
        if node.default_body:
            code += self.get_indent() + "default:\n"
            self.indentation += 1
            for stmt in node.default_body:
                code += self.get_indent() + self.generate_statement(stmt) + "\n"
            self.indentation -= 1

        code += self.get_indent() + "}\n"
        return code

    def visit_CaseNode(self, node):
        # Generate a case statement
        value = self.generate_expression(node.value)
        code = self.get_indent() + f"case {value}:\n"

        # Generate the case body
        self.indentation += 1
        for stmt in node.body:
            code += self.get_indent() + self.generate_statement(stmt) + "\n"
        self.indentation -= 1

        return code

    def generate_statement(self, node):
        """Generate a statement in CrossGL syntax"""
        if isinstance(node, str):
            return node
        if isinstance(node, BreakNode):
            return "break;"
        if isinstance(node, ContinueNode):
            return "continue;"
        if isinstance(node, ReturnNode):
            if node.value is None:
                return "return;"
            return f"return {self.generate_expression(node.value)};"
        elif hasattr(self, f"visit_{type(node).__name__}"):
            method = getattr(self, f"visit_{type(node).__name__}")
            return method(node)
        else:
            return self.generate_expression(node)
