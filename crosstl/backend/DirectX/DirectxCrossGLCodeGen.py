"""Reverse code generator that emits CrossGL from HLSL AST nodes."""

from .DirectxAst import *
from .DirectxParser import *
from .DirectxLexer import *
from ..common_ast import (
    ArrayAccessNode,
    BreakNode,
    CastNode,
    ContinueNode,
    TextureSampleNode,
)


class HLSLToCrossGLConverter:
    """Serialize DirectX backend AST nodes back into CrossGL source."""

    def __init__(self):
        """Initialize HLSL-to-CrossGL type, function, and semantic mappings."""
        self.structured_buffer_types = {
            "Buffer",
            "RWBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }
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
            "RasterizerOrderedTexture1D": "image1D",
            "RasterizerOrderedTexture1DArray": "image1DArray",
            "RasterizerOrderedTexture2D": "image2D",
            "RasterizerOrderedTexture2DArray": "image2DArray",
            "RasterizerOrderedTexture3D": "image3D",
            # Buffer Types
            "Buffer": "samplerBuffer",
            "RWBuffer": "imageBuffer",
            "StructuredBuffer": "buffer",
            "RWStructuredBuffer": "buffer",
            "ByteAddressBuffer": "buffer",
            "RWByteAddressBuffer": "buffer",
            "RasterizerOrderedByteAddressBuffer": "RWByteAddressBuffer",
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
            "SamplerComparisonState": "sampler",
        }
        self.type_map.update(self.minimum_precision_type_map())
        self.shadow_texture_type_map = {
            "Texture2D": "sampler2DShadow",
            "Texture2DArray": "sampler2DArrayShadow",
            "TextureCube": "samplerCubeShadow",
            "TextureCubeArray": "samplerCubeArrayShadow",
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
            "Sample": "texture",
            "SampleLevel": "textureLod",
            "SampleGrad": "textureGrad",
            "SampleBias": "texture",
            "SampleCmp": "textureCompare",
            "SampleCmpLevelZero": "textureCompare",
            "Load": "texelFetch",
            "Gather": "textureGather",
            "GetDimensions": "texture_dimensions",
        }
        self.texture_gather_component_map = {
            "GatherRed": "0",
            "GatherGreen": "1",
            "GatherBlue": "2",
            "GatherAlpha": "3",
        }
        self.buffer_method_map = {
            "Load": "buffer_load",
            "Load2": "buffer_load2",
            "Load3": "buffer_load3",
            "Load4": "buffer_load4",
            "Store": "buffer_store",
            "Store2": "buffer_store2",
            "Store3": "buffer_store3",
            "Store4": "buffer_store4",
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
        self.shadow_texture_names = set()
        self.shadow_texture_declaration_ids = set()
        self.global_variable_types = {}
        self.global_resource_array_dims = {}
        self.current_variable_types = {}
        self.current_resource_array_dims = {}
        self.suppress_storage_image_index_lowering = False

    @staticmethod
    def minimum_precision_type_map():
        type_map = {}
        vector_aliases = {
            "min16float": "f16vec",
            "min10float": "f16vec",
            "min16int": "i16vec",
            "min12int": "i16vec",
            "min16uint": "u16vec",
        }
        for hlsl_prefix, crossgl_prefix in vector_aliases.items():
            for width in range(2, 5):
                type_map[f"{hlsl_prefix}{width}"] = f"{crossgl_prefix}{width}"

        for hlsl_prefix in ("min16float", "min10float"):
            for columns in range(2, 5):
                for rows in range(2, 5):
                    suffix = str(columns) if columns == rows else f"{columns}x{rows}"
                    type_map[f"{hlsl_prefix}{columns}x{rows}"] = f"f16mat{suffix}"

        return type_map

    def texture_method_descriptor(self, member, arg_count=None):
        if member in {"Load", "GetDimensions"}:
            texture_function = self.texture_method_map[member]
            buffer_function = self.buffer_method_map[member]
            use_buffer = arg_count is not None and arg_count <= 1
            return {
                "member": member,
                "function": buffer_function if use_buffer else texture_function,
                "texture_function": texture_function,
                "buffer_function": buffer_function,
                "component": None,
                "usage": "regular" if member == "Load" else None,
                "buffer_when_max_args": 1,
            }
        if member in self.texture_gather_component_map:
            return {
                "member": member,
                "function": self.texture_method_map["Gather"],
                "texture_function": self.texture_method_map["Gather"],
                "buffer_function": None,
                "component": self.texture_gather_component_map[member],
                "usage": "regular",
                "buffer_when_max_args": None,
            }
        if member in self.texture_method_map:
            texture_function = self.texture_method_map[member]
            if member == "Sample" and arg_count == 3:
                texture_function = "textureOffset"
            elif member == "SampleLevel" and arg_count == 4:
                texture_function = "textureLodOffset"
            elif member == "SampleGrad" and arg_count == 5:
                texture_function = "textureGradOffset"
            if member == "SampleBias" and arg_count is not None and arg_count >= 4:
                texture_function = "textureOffset"
            usage = (
                "comparison"
                if member in {"SampleCmp", "SampleCmpLevelZero"}
                else "regular"
            )
            return {
                "member": member,
                "function": texture_function,
                "texture_function": texture_function,
                "buffer_function": None,
                "component": None,
                "usage": usage,
                "buffer_when_max_args": None,
            }
        return None

    def resource_method_descriptor(self, member, arg_count=None):
        texture_descriptor = self.texture_method_descriptor(member, arg_count)
        if texture_descriptor:
            descriptor = dict(texture_descriptor)
            uses_buffer = (
                descriptor["buffer_function"] is not None
                and descriptor["function"] == descriptor["buffer_function"]
            )
            descriptor["resource"] = "buffer" if uses_buffer else "texture"
            descriptor["operation"] = {
                "Load": "load",
                "GetDimensions": "dimensions",
                "Sample": "sample",
                "SampleLevel": "sample_lod",
                "SampleGrad": "sample_grad",
                "SampleBias": "sample_bias",
                "SampleCmp": "sample_compare",
                "SampleCmpLevelZero": "sample_compare",
                "Gather": "gather",
                "GatherRed": "gather",
                "GatherGreen": "gather",
                "GatherBlue": "gather",
                "GatherAlpha": "gather",
            }.get(member)
            return descriptor

        if member in self.buffer_method_map:
            return {
                "member": member,
                "function": self.buffer_method_map[member],
                "texture_function": None,
                "buffer_function": self.buffer_method_map[member],
                "component": None,
                "usage": None,
                "buffer_when_max_args": None,
                "resource": "buffer",
                "operation": (
                    {
                        "Store": "store",
                        "Append": "append",
                        "Consume": "consume",
                        "Load": "load",
                        "GetDimensions": "dimensions",
                    }.get(member)
                ),
            }
        return None

    def resource_method_arguments(self, obj, member, rendered_args, descriptor):
        if (
            member == "SampleBias"
            and descriptor["function"] == "textureOffset"
            and len(rendered_args) >= 4
        ):
            sampler, coords, bias, offset = rendered_args[:4]
            method_args = [obj, sampler, coords, offset, bias]
        else:
            method_args = [obj, *rendered_args]

        if descriptor["component"] is not None:
            method_args.append(descriptor["component"])
        return method_args

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

    def format_attributes(self, attributes, indent, skip_names=None):
        if not attributes:
            return ""
        skip_names = {str(name).lower() for name in skip_names or ()}
        lines = ""
        for attr in attributes:
            attr_name = str(getattr(attr, "name", ""))
            if attr_name.lower() in skip_names:
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                rendered_args = ", ".join(self.generate_expression(arg) for arg in args)
                lines += "    " * indent + f"@ {attr_name}({rendered_args})\n"
            else:
                lines += "    " * indent + f"@ {attr_name}\n"
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

    def is_uav_resource_type(self, hlsl_type):
        if not hlsl_type:
            return False
        type_name = str(hlsl_type)
        if "<" in type_name:
            type_name = type_name.split("<", 1)[0]
        return type_name.startswith(
            ("RWTexture", "RWBuffer", "RasterizerOrdered")
        ) or type_name in {
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def format_resource_qualifier_attributes(self, node, indent):
        if not self.is_uav_resource_type(getattr(node, "vtype", None)):
            return ""

        lines = ""
        if self.is_rasterizer_ordered_resource_type(getattr(node, "vtype", None)):
            lines += "    " * indent + "@ rasterizer_ordered\n"

        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "globallycoherent" in qualifiers:
            lines += "    " * indent + "@ globallycoherent\n"
        return lines

    def record_variable_type(self, node, type_map=None, array_dim_map=None):
        name = getattr(node, "name", None)
        if not name:
            return
        if type_map is None:
            type_map = self.current_variable_types
        if array_dim_map is None:
            array_dim_map = self.current_resource_array_dims
        type_map[name] = getattr(node, "vtype", None)
        array_dim_map[name] = len(getattr(node, "array_sizes", None) or [])

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.expression_base_name(expr.object)
        return None

    def expression_raw_type(self, expr):
        name = self.expression_base_name(expr)
        if not name:
            return None
        raw_type = self.current_variable_types.get(name)
        if raw_type is None:
            raw_type = self.global_variable_types.get(name)
        return raw_type

    def expression_resource_array_dims(self, expr):
        name = self.expression_base_name(expr)
        if not name:
            return 0
        if name in self.current_resource_array_dims:
            return self.current_resource_array_dims[name]
        return self.global_resource_array_dims.get(name, 0)

    def raw_type_base(self, type_name):
        if not type_name:
            return ""
        base = str(type_name).strip()
        if "<" in base and base.endswith(">"):
            base = base.split("<", 1)[0]
        return base

    def is_rasterizer_ordered_resource_type(self, type_name):
        return self.raw_type_base(type_name).startswith("RasterizerOrdered")

    def is_rw_texture_type(self, type_name):
        return self.raw_type_base(type_name).startswith(
            ("RWTexture", "RasterizerOrderedTexture")
        )

    def is_rw_typed_buffer_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "RWBuffer",
            "RWStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
        }

    def array_access_depth(self, expr):
        depth = 0
        while isinstance(expr, ArrayAccessNode):
            depth += 1
            expr = expr.array
        return depth

    def is_storage_image_texel_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not self.is_rw_texture_type(self.expression_raw_type(expr)):
            return False
        return self.array_access_depth(expr) > self.expression_resource_array_dims(expr)

    def generate_without_storage_index_lowering(self, expr, is_main=False):
        previous = self.suppress_storage_image_index_lowering
        self.suppress_storage_image_index_lowering = True
        try:
            return self.generate_expression(expr, is_main)
        finally:
            self.suppress_storage_image_index_lowering = previous

    def generate_storage_image_access_parts(self, expr, is_main=False):
        image = self.generate_without_storage_index_lowering(expr.array, is_main)
        coord = self.generate_expression(expr.index, is_main)
        return image, coord

    def generate_storage_image_load(self, expr, is_main=False):
        image, coord = self.generate_storage_image_access_parts(expr, is_main)
        return f"imageLoad({image}, {coord})"

    def generate_storage_image_store(self, access, value, operator, is_main=False):
        image, coord = self.generate_storage_image_access_parts(access, is_main)
        rendered_value = self.generate_expression(value, is_main)
        if operator != "=":
            compound_ops = {
                "+=": "+",
                "-=": "-",
                "*=": "*",
                "/=": "/",
                "%=": "%",
                "&=": "&",
                "|=": "|",
                "^=": "^",
                "<<=": "<<",
                ">>=": ">>",
            }
            binary_op = compound_ops.get(operator)
            if binary_op is None:
                return None
            current_value = self.generate_storage_image_load(access, is_main)
            rendered_value = f"{current_value} {binary_op} {rendered_value}"
        return f"imageStore({image}, {coord}, {rendered_value})"

    def storage_image_component_type(self, access):
        raw_type = self.expression_raw_type(access)
        if raw_type is None:
            return None
        type_name = str(raw_type)
        if "<" not in type_name or ">" not in type_name:
            return None
        return type_name.split("<", 1)[1].rsplit(">", 1)[0].strip()

    def generate_storage_image_atomic_value(self, arg, component_type, is_main):
        if (
            component_type == "uint"
            and isinstance(arg, int)
            and not isinstance(arg, bool)
            and arg >= 0
        ):
            return f"{arg}u"
        return self.generate_expression(arg, is_main)

    def typed_buffer_component_type(self, access):
        raw_type = self.expression_raw_type(access)
        if raw_type is None:
            return None
        type_name = str(raw_type)
        if "<" not in type_name or ">" not in type_name:
            return None
        return type_name.split("<", 1)[1].rsplit(">", 1)[0].strip()

    def is_typed_buffer_element_access(self, expr):
        if not isinstance(expr, ArrayAccessNode):
            return False
        if not self.is_rw_typed_buffer_type(self.expression_raw_type(expr)):
            return False
        return self.array_access_depth(expr) > self.expression_resource_array_dims(expr)

    def generate_typed_buffer_atomic_value(self, arg, component_type, is_main):
        if (
            component_type == "uint"
            and isinstance(arg, int)
            and not isinstance(arg, bool)
            and arg >= 0
        ):
            return f"{arg}u"
        return self.generate_expression(arg, is_main)

    def interlocked_storage_image_atomic_expression(self, func_name, args, is_main):
        operation_map = {
            "InterlockedAdd": "imageAtomicAdd",
            "InterlockedAnd": "imageAtomicAnd",
            "InterlockedOr": "imageAtomicOr",
            "InterlockedXor": "imageAtomicXor",
            "InterlockedMin": "imageAtomicMin",
            "InterlockedMax": "imageAtomicMax",
            "InterlockedExchange": "imageAtomicExchange",
            "InterlockedCompareExchange": "imageAtomicCompSwap",
        }
        operation = operation_map.get(func_name)
        if (
            operation is None
            or not args
            or not self.is_storage_image_texel_access(args[0])
        ):
            return None

        expected_min_args = 3 if func_name == "InterlockedCompareExchange" else 2
        if len(args) < expected_min_args:
            return None

        image, coord = self.generate_storage_image_access_parts(args[0], is_main)
        component_type = self.storage_image_component_type(args[0])
        if func_name == "InterlockedCompareExchange":
            value_args = [
                self.generate_storage_image_atomic_value(
                    args[1], component_type, is_main
                ),
                self.generate_storage_image_atomic_value(
                    args[2], component_type, is_main
                ),
            ]
            original_index = 3
        else:
            value_args = [
                self.generate_storage_image_atomic_value(
                    args[1], component_type, is_main
                )
            ]
            original_index = 2

        atomic_call = f"{operation}({image}, {coord}, {', '.join(value_args)})"
        if len(args) > original_index:
            original = self.generate_without_storage_index_lowering(
                args[original_index], is_main
            )
            return f"{original} = {atomic_call}"
        return atomic_call

    def interlocked_typed_buffer_atomic_expression(self, func_name, args, is_main):
        operation_map = {
            "InterlockedAdd": "atomicAdd",
            "InterlockedAnd": "atomicAnd",
            "InterlockedOr": "atomicOr",
            "InterlockedXor": "atomicXor",
            "InterlockedMin": "atomicMin",
            "InterlockedMax": "atomicMax",
            "InterlockedExchange": "atomicExchange",
            "InterlockedCompareExchange": "atomicCompareExchange",
        }
        operation = operation_map.get(func_name)
        if (
            operation is None
            or not args
            or not self.is_typed_buffer_element_access(args[0])
        ):
            return None

        expected_min_args = 3 if func_name == "InterlockedCompareExchange" else 2
        if len(args) < expected_min_args:
            return None

        target = self.generate_without_storage_index_lowering(args[0], is_main)
        component_type = self.typed_buffer_component_type(args[0])
        if func_name == "InterlockedCompareExchange":
            value_args = [
                self.generate_typed_buffer_atomic_value(
                    args[1], component_type, is_main
                ),
                self.generate_typed_buffer_atomic_value(
                    args[2], component_type, is_main
                ),
            ]
            original_index = 3
        else:
            value_args = [
                self.generate_typed_buffer_atomic_value(
                    args[1], component_type, is_main
                )
            ]
            original_index = 2

        atomic_call = f"{operation}({target}, {', '.join(value_args)})"
        if len(args) > original_index:
            original = self.generate_without_storage_index_lowering(
                args[original_index], is_main
            )
            return f"{original} = {atomic_call}"
        return atomic_call

    def iter_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            for value in node.values():
                yield value
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                yield value
            return
        for value in getattr(node, "__dict__", {}).values():
            yield value

    def collect_direct_texture_method_usage_names(self, root):
        comparison_names = set()
        regular_names = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, TextureSampleNode):
                texture_name = self.expression_base_name(node.texture)
                if texture_name:
                    regular_names.add(texture_name)
            if isinstance(node, FunctionCallNode) and isinstance(
                node.name, MemberAccessNode
            ):
                texture_name = self.expression_base_name(node.name.object)
                descriptor = self.resource_method_descriptor(
                    node.name.member, len(getattr(node, "args", []) or [])
                )
                usage = (
                    descriptor["usage"]
                    if descriptor and descriptor["resource"] == "texture"
                    else None
                )
                if usage == "comparison":
                    if texture_name:
                        comparison_names.add(texture_name)
                elif usage == "regular":
                    if texture_name:
                        regular_names.add(texture_name)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return comparison_names, regular_names

    def collect_function_texture_parameter_usage(self, root):
        function_usage = {}
        for func in getattr(root, "functions", []) or []:
            param_by_name = {
                getattr(param, "name", None): index
                for index, param in enumerate(getattr(func, "params", []) or [])
            }
            param_by_name.pop(None, None)
            comparison_names, regular_names = (
                self.collect_direct_texture_method_usage_names(
                    getattr(func, "body", [])
                )
            )
            function_usage[getattr(func, "name", None)] = {
                "comparison": {
                    param_by_name[name]
                    for name in comparison_names
                    if name in param_by_name
                },
                "regular": {
                    param_by_name[name]
                    for name in regular_names
                    if name in param_by_name
                },
            }
        function_usage.pop(None, None)
        return function_usage

    def collect_nonparameter_texture_usage_names(self, root):
        comparison_names = set()
        regular_names = set()

        for node in getattr(root, "global_variables", []) or []:
            direct_comparison, direct_regular = (
                self.collect_direct_texture_method_usage_names(node)
            )
            comparison_names.update(direct_comparison)
            regular_names.update(direct_regular)

        for func in getattr(root, "functions", []) or []:
            param_names = {
                getattr(param, "name", None)
                for param in getattr(func, "params", []) or []
            }
            param_names.discard(None)
            direct_comparison, direct_regular = (
                self.collect_direct_texture_method_usage_names(
                    getattr(func, "body", [])
                )
            )
            comparison_names.update(direct_comparison - param_names)
            regular_names.update(direct_regular - param_names)

        return comparison_names, regular_names

    def propagate_function_texture_usage(
        self, root, function_usage, comparison_names, regular_names
    ):
        changed = False

        def apply_call_usage(node, caller_function_name, caller_param_by_name):
            nonlocal changed
            if not isinstance(node, FunctionCallNode) or not isinstance(node.name, str):
                return
            callee_usage = function_usage.get(node.name)
            if not callee_usage:
                return
            for usage_kind, usage_names in (
                ("comparison", comparison_names),
                ("regular", regular_names),
            ):
                for param_index in callee_usage[usage_kind]:
                    if param_index >= len(node.args):
                        continue
                    arg_name = self.expression_base_name(node.args[param_index])
                    if not arg_name:
                        continue
                    caller_param_index = caller_param_by_name.get(arg_name)
                    if caller_param_index is not None:
                        caller_usage = function_usage.get(caller_function_name)
                        if (
                            caller_usage is not None
                            and caller_param_index not in caller_usage[usage_kind]
                        ):
                            caller_usage[usage_kind].add(caller_param_index)
                            changed = True
                    elif arg_name not in usage_names:
                        usage_names.add(arg_name)
                        changed = True

        def visit(node, caller_function_name, caller_param_by_name):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            apply_call_usage(node, caller_function_name, caller_param_by_name)
            for child in self.iter_ast_children(node):
                visit(child, caller_function_name, caller_param_by_name)

        for func in getattr(root, "functions", []) or []:
            param_by_name = {
                getattr(param, "name", None): index
                for index, param in enumerate(getattr(func, "params", []) or [])
            }
            param_by_name.pop(None, None)
            visit(getattr(func, "body", []), getattr(func, "name", None), param_by_name)

        return changed

    def collect_shadow_texture_names(self, root):
        comparison_names, regular_names = self.collect_nonparameter_texture_usage_names(
            root
        )
        function_usage = self.collect_function_texture_parameter_usage(root)
        while self.propagate_function_texture_usage(
            root, function_usage, comparison_names, regular_names
        ):
            pass

        shadow_names = comparison_names - regular_names
        self.shadow_texture_declaration_ids = set()
        for node in getattr(root, "global_variables", []) or []:
            if getattr(node, "name", None) in shadow_names:
                self.shadow_texture_declaration_ids.add(id(node))

        for func in getattr(root, "functions", []) or []:
            usage = function_usage.get(getattr(func, "name", None), {})
            shadow_indices = usage.get("comparison", set()) - usage.get(
                "regular", set()
            )
            for index, param in enumerate(getattr(func, "params", []) or []):
                if index in shadow_indices:
                    self.shadow_texture_declaration_ids.add(id(param))

        return shadow_names

    def map_variable_type(self, node):
        hlsl_type = getattr(node, "vtype", None)
        if id(node) not in self.shadow_texture_declaration_ids:
            return self.map_type(hlsl_type)
        type_name = hlsl_type
        if type_name and "<" in type_name and type_name.endswith(">"):
            type_name = type_name.split("<", 1)[0]
        return self.shadow_texture_type_map.get(type_name, self.map_type(hlsl_type))

    def visit(self, node):
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

        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        """Generate a complete CrossGL shader from a parsed HLSL AST."""
        self.shadow_texture_names = self.collect_shadow_texture_names(ast)
        self.global_variable_types = {}
        self.global_resource_array_dims = {}
        self.current_variable_types = {}
        self.current_resource_array_dims = {}
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
                        f"        {self.map_variable_type(member)} "
                        f"{member.name}{array_suffix}{semantic};\n"
                    )
                code += "    }\n"
            elif isinstance(node, PragmaNode):
                code += f"    #pragma {node.directive} {node.value};\n"
            elif isinstance(node, IncludeNode):
                code += f"    #include {node.path}\n"
        for node in ast.global_variables:
            self.record_variable_type(
                node, self.global_variable_types, self.global_resource_array_dims
            )
            code += self.format_attributes(getattr(node, "attributes", []), 1)
            code += self.format_resource_qualifier_attributes(node, 1)
            code += self.format_binding_attributes(node, 1)
            array_suffix = self.format_array_suffixes(node)
            code += f"    {self.map_variable_type(node)} {node.name}{array_suffix};\n"
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
        for func in ast.functions:
            stage_name = stage_map.get(func.qualifier)
            if stage_name:
                code += f"    // {stage_name} Shader\n"
                code += f"    {stage_name} {{\n"
                code += self.generate_function(func, skip_attribute_names={"shader"})
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
                        f"        {self.map_variable_type(member)} "
                        f"{member.name}{array_suffix};\n"
                    )
                code += "    }\n"
        return code

    def generate_function(self, func, indent=1, skip_attribute_names=None):
        """Render one HLSL function node as a CrossGL function block."""
        code = self.format_attributes(
            getattr(func, "attributes", []), indent, skip_attribute_names
        )
        code += "    " * indent
        previous_variable_types = self.current_variable_types
        previous_resource_array_dims = self.current_resource_array_dims
        self.current_variable_types = dict(self.global_variable_types)
        self.current_resource_array_dims = dict(self.global_resource_array_dims)
        for param in func.params:
            self.record_variable_type(param)
        params = ", ".join(
            f"{self.map_variable_type(p)} {p.name}{self.format_array_suffixes(p)}"
            f"{(' ' + self.map_semantic(p.semantic)) if self.map_semantic(p.semantic) else ''}"
            for p in func.params
        )
        semantic = self.map_semantic(func.semantic)
        semantic = f" {semantic}" if semantic else ""
        code += (
            f"{self.map_type(func.return_type)} {func.name}({params}){semantic} {{\n"
        )
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    " * indent + "}\n\n"
        self.current_variable_types = previous_variable_types
        self.current_resource_array_dims = previous_resource_array_dims
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            code += "    " * indent
            if isinstance(stmt, VariableNode):
                array_suffix = self.format_array_suffixes(stmt, is_main)
                if stmt.value is not None:
                    value = self.generate_expression(stmt.value, is_main)
                    self.record_variable_type(stmt)
                    code += (
                        f"{self.map_variable_type(stmt)} {stmt.name}{array_suffix} = "
                        f"{value};\n"
                    )
                else:
                    self.record_variable_type(stmt)
                    code += (
                        f"{self.map_variable_type(stmt)} "
                        f"{stmt.name}{array_suffix};\n"
                    )
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
            init = (
                f"{self.map_variable_type(node.init)} "
                f"{node.init.name}{array_suffix}"
            )
            if node.init.value is not None:
                init += f" = {self.generate_expression(node.init.value, is_main)}"
            self.record_variable_type(node.init)
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
            self.generate_expression(node.update, is_main)
            if node.update is not None
            else ""
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
        if self.is_storage_image_texel_access(node.left):
            storage_store = self.generate_storage_image_store(
                node.left, node.right, node.operator, is_main
            )
            if storage_store is not None:
                return storage_store
        lhs = self.generate_expression(node.left, is_main)
        rhs = self.generate_expression(node.right, is_main)
        op = node.operator
        return f"{lhs} {op} {rhs}"

    def generate_expression(self, expr, is_main=False):
        """Render a DirectX backend expression node as CrossGL syntax."""
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
            if isinstance(expr.name, MemberAccessNode):
                obj = self.generate_expression(expr.name.object, is_main)
                member = expr.name.member
                rendered_args = [
                    self.generate_expression(arg, is_main) for arg in expr.args
                ]
                args = ", ".join(rendered_args)
                descriptor = self.resource_method_descriptor(member, len(expr.args))
                if descriptor:
                    method_args = self.resource_method_arguments(
                        obj, member, rendered_args, descriptor
                    )
                    return f"{descriptor['function']}({', '.join(method_args)})"
                return f"{obj}.{member}({args})"

            func_name = (
                expr.name
                if isinstance(expr.name, str)
                else self.generate_expression(expr.name, is_main)
            )
            interlocked_image_atomic = self.interlocked_storage_image_atomic_expression(
                func_name, expr.args, is_main
            )
            if interlocked_image_atomic is not None:
                return interlocked_image_atomic
            interlocked_buffer_atomic = self.interlocked_typed_buffer_atomic_expression(
                func_name, expr.args, is_main
            )
            if interlocked_buffer_atomic is not None:
                return interlocked_buffer_atomic
            if func_name in self.interlocked_map:
                rendered_args = [
                    self.generate_without_storage_index_lowering(arg, is_main)
                    for arg in expr.args
                ]
            else:
                rendered_args = [
                    self.generate_expression(arg, is_main) for arg in expr.args
                ]
            args = ", ".join(rendered_args)
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
            if (
                not self.suppress_storage_image_index_lowering
                and self.is_storage_image_texel_access(expr)
            ):
                return self.generate_storage_image_load(expr, is_main)
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
                return f"textureLod({texture}, {sampler}, {coords}, {lod})"
            return f"texture({texture}, {sampler}, {coords})"

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
        """Map an HLSL type name to the closest CrossGL type name."""
        if not hlsl_type:
            return hlsl_type
        type_name = str(hlsl_type)
        if "<" in type_name and type_name.endswith(">"):
            base, generic_args = type_name.split("<", 1)
            generic_type = generic_args[:-1].strip()
            rasterizer_buffer_type = self.map_rasterizer_ordered_buffer_type(
                base, generic_type
            )
            if rasterizer_buffer_type:
                return rasterizer_buffer_type
            if base in self.structured_buffer_types:
                return type_name
            storage_image_type = self.map_rw_texture_type(base, generic_type)
            if storage_image_type:
                return storage_image_type
            type_name = base
        return self.type_map.get(type_name, type_name)

    def map_rasterizer_ordered_buffer_type(self, base_type, element_type):
        buffer_type = {
            "RasterizerOrderedBuffer": "RWBuffer",
            "RasterizerOrderedStructuredBuffer": "RWStructuredBuffer",
        }.get(base_type)
        if buffer_type is None:
            return None
        return f"{buffer_type}<{element_type}>"

    def map_rw_texture_type(self, base_type, element_type):
        image_type = {
            "RWTexture1D": "image1D",
            "RWTexture1DArray": "image1DArray",
            "RWTexture2D": "image2D",
            "RWTexture2DArray": "image2DArray",
            "RWTexture2DMS": "image2DMS",
            "RWTexture2DMSArray": "image2DMSArray",
            "RWTexture3D": "image3D",
            "RWTextureCube": "imageCube",
            "RWTextureCubeArray": "imageCubeArray",
            "RasterizerOrderedTexture1D": "image1D",
            "RasterizerOrderedTexture1DArray": "image1DArray",
            "RasterizerOrderedTexture2D": "image2D",
            "RasterizerOrderedTexture2DArray": "image2DArray",
            "RasterizerOrderedTexture3D": "image3D",
        }.get(base_type)
        if image_type is None:
            return None

        element = element_type.strip()
        if element.startswith("uint"):
            return f"u{image_type}"
        if element.startswith("int"):
            return f"i{image_type}"
        return image_type

    def map_semantic(self, semantic):
        """Map an HLSL semantic to CrossGL semantic annotation syntax."""
        if not semantic:
            return ""
        mapped = self.semantic_map.get(semantic)
        if mapped is None and isinstance(semantic, str):
            mapped = self.semantic_map.get(semantic.upper())
        mapped = mapped or semantic
        return f"@ {mapped}"

    def generate_switch_statement(self, node, indent=1, is_main=False):
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
            case_body = getattr(case, "body", None) or getattr(case, "statements", [])
            code += self.generate_function_body(case_body, indent + 2, is_main)

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

        if hasattr(node.op, "token_type"):
            if node.op.token_type in ("BITWISE_AND", "AMPERSAND", "&"):
                return f"({left} & {right})"
            elif node.op.token_type in ("BITWISE_OR", "PIPE", "|"):
                return f"({left} | {right})"
            elif node.op.token_type in ("BITWISE_XOR", "CARET", "^"):
                return f"({left} ^ {right})"
        elif hasattr(node.op, "value"):
            if node.op.value in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op.value in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op.value in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"
        elif isinstance(node.op, str):
            if node.op in ("&", "BITWISE_AND", "AMPERSAND"):
                return f"({left} & {right})"
            elif node.op in ("|", "BITWISE_OR", "PIPE"):
                return f"({left} | {right})"
            elif node.op in ("^", "BITWISE_XOR", "CARET"):
                return f"({left} ^ {right})"

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

        op_str = node.op.value if hasattr(node.op, "value") else str(node.op)
        if getattr(node, "is_postfix", False):
            return f"{expr}{op_str}"
        return f"{op_str}{expr}"

    def visit_SwitchStatementNode(self, node):
        return self.visit_SwitchNode(node)

    def visit_SwitchCaseNode(self, node):
        return self.visit_CaseNode(node)

    def visit_StructNode(self, node):
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = ""
            if member.semantic:
                semantic = f" {self.map_semantic(member.semantic)}"

            array_suffix = self.format_array_suffixes(member)
            code += (
                self.get_indent() + f"{self.map_variable_type(member)} "
                f"{member.name}{array_suffix}{semantic};\n"
            )

        self.indentation -= 1
        code += self.get_indent() + "}\n"
        return code

    def visit_SwitchNode(self, node):
        condition = self.generate_expression(node.condition)
        code = f"switch ({condition}) {{\n"

        for case in node.cases:
            code += self.visit_CaseNode(case)

        if node.default_body:
            code += self.get_indent() + "default:\n"
            self.indentation += 1
            for stmt in node.default_body:
                code += self.get_indent() + self.generate_statement(stmt) + "\n"
            self.indentation -= 1

        code += self.get_indent() + "}\n"
        return code

    def visit_CaseNode(self, node):
        value = self.generate_expression(node.value)
        code = self.get_indent() + f"case {value}:\n"

        self.indentation += 1
        for stmt in node.body:
            code += self.get_indent() + self.generate_statement(stmt) + "\n"
        self.indentation -= 1

        return code

    def generate_statement(self, node):
        """Render one DirectX backend statement node as CrossGL source."""
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
