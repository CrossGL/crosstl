"""Reverse code generator that emits CrossGL from HLSL AST nodes."""

import re

from ..common_ast import (
    ArrayAccessNode,
    BreakNode,
    CastNode,
    ContinueNode,
    InitializerListNode,
    TextureSampleNode,
)
from .DirectxAst import *
from .DirectxLexer import *
from .DirectxParser import *


class HLSLToCrossGLConverter:
    """Serialize DirectX backend AST nodes back into CrossGL source."""

    def __init__(self):
        self.structured_buffer_types = {
            "Buffer",
            "ConstantBuffer",
            "RWBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "TextureBuffer",
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
            "ByteAddressBuffer": "ByteAddressBuffer",
            "RWByteAddressBuffer": "RWByteAddressBuffer",
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
            "EvaluateAttributeAtSample": "interpolateAtSample",
            "EvaluateAttributeSnapped": "interpolateAtOffset",
            "EvaluateAttributeCentroid": "interpolateAtCentroid",
            "GroupMemoryBarrierWithGroupSync": "workgroupBarrier",
            "GroupMemoryBarrier": "groupMemoryBarrier",
            "DeviceMemoryBarrier": "deviceMemoryBarrier",
            "AllMemoryBarrier": "allMemoryBarrier",
        }
        self.function_statement_sequences = {
            "DeviceMemoryBarrierWithGroupSync": (
                "deviceMemoryBarrier",
                "workgroupBarrier",
            ),
            "AllMemoryBarrierWithGroupSync": (
                "allMemoryBarrier",
                "workgroupBarrier",
            ),
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
            "SampleCmpLevel": "textureCompareLod",
            "SampleCmpGrad": "textureCompareGrad",
            "SampleCmpBias": "textureCompare",
            "SampleCmpLevelZero": "textureCompare",
            "Load": "texelFetch",
            "Gather": "textureGather",
            "GatherCmp": "textureGatherCompare",
            "GetDimensions": "texture_dimensions",
        }
        self.feedback_method_map = {
            "WriteSamplerFeedback": "write_sampler_feedback",
            "WriteSamplerFeedbackBias": "write_sampler_feedback_bias",
            "WriteSamplerFeedbackGrad": "write_sampler_feedback_grad",
            "WriteSamplerFeedbackLevel": "write_sampler_feedback_level",
        }
        self.texture_gather_component_map = {
            "GatherRed": "0",
            "GatherGreen": "1",
            "GatherBlue": "2",
            "GatherAlpha": "3",
        }
        self.texture_gather_compare_component_map = {
            "GatherCmpRed": "0",
            "GatherCmpGreen": "1",
            "GatherCmpBlue": "2",
            "GatherCmpAlpha": "3",
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
            "IncrementCounter": "buffer_increment_counter",
            "DecrementCounter": "buffer_decrement_counter",
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
            "PSIZE": "gl_PointSize",
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
        self.struct_member_types = {}
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

    def texture_method_descriptor(self, member, arg_count=None, resource_type=None):
        def with_dropped_parameters(descriptor, parameters):
            if parameters:
                descriptor["drop_trailing_args"] = len(parameters)
                descriptor["dropped_parameters"] = list(parameters)
            return descriptor

        if member in self.feedback_method_map:
            resource_base = self.raw_type_base(resource_type)
            if resource_base.startswith("FeedbackTexture"):
                return {
                    "member": member,
                    "function": self.feedback_method_map[member],
                    "texture_function": self.feedback_method_map[member],
                    "buffer_function": None,
                    "component": None,
                    "usage": None,
                    "buffer_when_max_args": None,
                    "resource": "feedback_texture",
                    "resource_type": resource_type,
                    "operation": {
                        "WriteSamplerFeedback": "write_sampler_feedback",
                        "WriteSamplerFeedbackBias": "write_sampler_feedback_bias",
                        "WriteSamplerFeedbackGrad": "write_sampler_feedback_grad",
                        "WriteSamplerFeedbackLevel": "write_sampler_feedback_level",
                    }[member],
                }

        cube_family_resource = self.raw_type_base(resource_type) in {
            "TextureCube",
            "TextureCubeArray",
        }

        if member == "Load":
            texture_function = self.texture_method_map[member]
            buffer_function = self.buffer_method_map[member]
            resource_base = self.raw_type_base(resource_type)
            dropped_parameters = []
            is_multisample = resource_base in {
                "Texture2DMS",
                "Texture2DMSArray",
                "RWTexture2DMS",
                "RWTexture2DMSArray",
                "RasterizerOrderedTexture2DMS",
                "RasterizerOrderedTexture2DMSArray",
            }
            if self.is_rw_texture_type(resource_type):
                function = "imageLoad"
                if arg_count == (3 if is_multisample else 2):
                    dropped_parameters.append("status output")
            elif resource_base.startswith(("Texture", "FeedbackTexture")):
                has_offset = arg_count is not None and arg_count >= (
                    3 if is_multisample else 2
                )
                function = "texelFetchOffset" if has_offset else texture_function
                if arg_count == (4 if is_multisample else 3):
                    dropped_parameters.append("status output")
            else:
                function = (
                    buffer_function
                    if arg_count is not None and arg_count <= 1
                    else texture_function
                )
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": function,
                    "texture_function": texture_function,
                    "buffer_function": buffer_function,
                    "component": None,
                    "usage": "regular",
                    "buffer_when_max_args": 1,
                    "resource_type": resource_type,
                    "diagnostic_kind": "tiled_resource_status",
                },
                dropped_parameters,
            )
        if member == "GetDimensions":
            texture_function = self.texture_method_map[member]
            buffer_function = self.buffer_method_map[member]
            use_buffer = self.is_buffer_resource_type(resource_type)
            if resource_type is None:
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
        if member in {"CalculateLevelOfDetail", "CalculateLevelOfDetailUnclamped"}:
            resource_base = self.raw_type_base(resource_type)
            diagnostic_reason = None
            if arg_count != 2:
                diagnostic_reason = "expected sampler and coordinate arguments"
            elif resource_base and self.is_multisample_texture_type(resource_type):
                diagnostic_reason = f"{member} is unavailable for multisample textures"
            elif resource_base and (
                self.is_rw_texture_type(resource_type)
                or self.is_buffer_resource_type(resource_type)
                or not resource_base.startswith("Texture")
            ):
                diagnostic_reason = f"{member} is only available on sampled textures"

            descriptor = {
                "member": member,
                "function": "textureQueryLod",
                "texture_function": "textureQueryLod",
                "buffer_function": None,
                "component": None,
                "usage": None if diagnostic_reason else "regular",
                "buffer_when_max_args": None,
                "result_component": (
                    ".y" if member == "CalculateLevelOfDetailUnclamped" else ".x"
                ),
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if diagnostic_reason:
                descriptor["diagnostic_reason"] = diagnostic_reason
                descriptor["fallback_expression"] = "0.0"
            return descriptor
        if member == "GetSamplePosition":
            resource_base = self.raw_type_base(resource_type)
            diagnostic_reason = None
            if arg_count != 1:
                diagnostic_reason = "expected sample-index argument"
            elif resource_base and resource_base not in {
                "Texture2DMS",
                "Texture2DMSArray",
            }:
                diagnostic_reason = (
                    "GetSamplePosition is only available on sampled multisample "
                    "textures"
                )

            descriptor = {
                "member": member,
                "function": "textureSamplePosition",
                "texture_function": "textureSamplePosition",
                "buffer_function": None,
                "component": None,
                "usage": None if diagnostic_reason else "regular",
                "buffer_when_max_args": None,
                "diagnostic_label": "texture sample-position query",
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if diagnostic_reason:
                descriptor["diagnostic_reason"] = diagnostic_reason
                descriptor["fallback_expression"] = "vec2(0.0, 0.0)"
            return descriptor
        if member in self.texture_gather_component_map:
            dropped_parameters = []
            if cube_family_resource and arg_count == 3:
                texture_function = "textureGather"
                dropped_parameters.append("status output")
            elif arg_count in {3, 4}:
                texture_function = "textureGatherOffset"
                if arg_count == 4:
                    dropped_parameters.append("status output")
            elif arg_count in {6, 7}:
                texture_function = "textureGatherOffsets"
                if arg_count == 7:
                    dropped_parameters.append("status output")
            else:
                texture_function = "textureGather"
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": texture_function,
                    "texture_function": texture_function,
                    "buffer_function": None,
                    "component": self.texture_gather_component_map[member],
                    "usage": "regular",
                    "buffer_when_max_args": None,
                },
                dropped_parameters,
            )
        if member in self.texture_gather_compare_component_map:
            component = self.texture_gather_compare_component_map[member]
            descriptor = {
                "member": member,
                "function": "textureGatherCompare",
                "texture_function": "textureGatherCompare",
                "buffer_function": None,
                "component": None,
                "usage": "comparison",
                "buffer_when_max_args": None,
                "diagnostic_label": "texture gather-compare component",
            }
            if resource_type is not None:
                descriptor["resource_type"] = resource_type
            if component != "0":
                descriptor["diagnostic_reason"] = (
                    f"{member} requires a compare-gather component selector; "
                    "CrossGL only represents the default/red compare gather"
                )
                descriptor["fallback_expression"] = "vec4(0.0)"
                return descriptor
            if arg_count in {7, 8}:
                descriptor["diagnostic_reason"] = (
                    "GatherCmpRed four-offset overload has no CrossGL helper yet"
                )
                descriptor["fallback_expression"] = "vec4(0.0)"
                return descriptor

            dropped_parameters = []
            resource_base = self.raw_type_base(resource_type)
            if resource_base in {"TextureCube", "TextureCubeArray"} and arg_count == 4:
                dropped_parameters.append("status output")
            elif arg_count in {4, 5}:
                descriptor["function"] = "textureGatherCompareOffset"
                descriptor["texture_function"] = "textureGatherCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            return with_dropped_parameters(descriptor, dropped_parameters)
        if member in self.texture_method_map:
            texture_function = self.texture_method_map[member]
            dropped_parameters = []
            if member == "Sample" and cube_family_resource and arg_count in {3, 4}:
                texture_function = "texture"
                if arg_count == 3:
                    dropped_parameters.append("LOD clamp")
                else:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "Sample" and arg_count in {3, 4, 5}:
                texture_function = "textureOffset"
                if arg_count == 4:
                    dropped_parameters.append("LOD clamp")
                elif arg_count == 5:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleLevel" and cube_family_resource and arg_count == 4:
                texture_function = "textureLod"
                dropped_parameters.append("status output")
            elif member == "SampleLevel" and arg_count in {4, 5}:
                texture_function = "textureLodOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            elif (
                member == "SampleGrad"
                and cube_family_resource
                and arg_count
                in {
                    5,
                    6,
                }
            ):
                texture_function = "textureGrad"
                if arg_count == 5:
                    dropped_parameters.append("LOD clamp")
                else:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleGrad" and arg_count in {5, 6, 7}:
                texture_function = "textureGradOffset"
                if arg_count == 6:
                    dropped_parameters.append("LOD clamp")
                elif arg_count == 7:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif (
                member == "SampleBias"
                and cube_family_resource
                and arg_count
                in {
                    4,
                    5,
                }
            ):
                texture_function = "texture"
                if arg_count == 4:
                    dropped_parameters.append("LOD clamp")
                else:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleBias" and arg_count in {4, 5, 6}:
                texture_function = "textureOffset"
                if arg_count == 5:
                    dropped_parameters.append("LOD clamp")
                elif arg_count == 6:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif (
                member == "SampleCmp"
                and cube_family_resource
                and arg_count
                in {
                    4,
                    5,
                }
            ):
                texture_function = "textureCompare"
                if arg_count == 4:
                    dropped_parameters.append("LOD clamp")
                else:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleCmp" and arg_count in {4, 5, 6}:
                texture_function = "textureCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("LOD clamp")
                elif arg_count == 6:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleCmpLevel" and cube_family_resource and arg_count == 5:
                texture_function = "textureCompareLod"
                dropped_parameters.append("status output")
            elif member == "SampleCmpLevel" and arg_count in {5, 6}:
                texture_function = "textureCompareLodOffset"
                if arg_count == 6:
                    dropped_parameters.append("status output")
            elif (
                member == "SampleCmpGrad"
                and cube_family_resource
                and arg_count
                in {
                    6,
                    7,
                }
            ):
                texture_function = "textureCompareGrad"
                if arg_count == 6:
                    dropped_parameters.append("LOD clamp")
                else:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif member == "SampleCmpGrad" and arg_count in {6, 7, 8}:
                texture_function = "textureCompareGradOffset"
                if arg_count == 7:
                    dropped_parameters.append("LOD clamp")
                elif arg_count == 8:
                    dropped_parameters.extend(["LOD clamp", "status output"])
            elif (
                member == "SampleCmpBias"
                and cube_family_resource
                and arg_count
                in {
                    4,
                    5,
                    6,
                }
            ):
                texture_function = "textureCompare"
                if arg_count == 4:
                    dropped_parameters.append("LOD bias")
                elif arg_count == 5:
                    dropped_parameters.extend(["LOD bias", "LOD clamp"])
                else:
                    dropped_parameters.extend(
                        ["LOD bias", "LOD clamp", "status output"]
                    )
            elif member == "SampleCmpBias" and arg_count in {4, 5, 6, 7}:
                if arg_count == 4:
                    texture_function = "textureCompare"
                    dropped_parameters.append("LOD bias")
                else:
                    texture_function = "textureCompareOffset"
                    if arg_count == 5:
                        dropped_parameters.append("LOD bias")
                    elif arg_count == 6:
                        dropped_parameters.extend(["LOD bias", "LOD clamp"])
                    else:
                        dropped_parameters.extend(
                            ["LOD bias", "LOD clamp", "status output"]
                        )
            elif (
                member == "SampleCmpLevelZero"
                and cube_family_resource
                and arg_count == 4
            ):
                texture_function = "textureCompare"
                dropped_parameters.append("status output")
            elif member == "SampleCmpLevelZero" and arg_count in {4, 5}:
                texture_function = "textureCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            elif member == "Gather" and cube_family_resource and arg_count == 3:
                texture_function = "textureGather"
                dropped_parameters.append("status output")
            elif member == "Gather" and arg_count in {3, 4}:
                texture_function = "textureGatherOffset"
                if arg_count == 4:
                    dropped_parameters.append("status output")
            elif member == "GatherCmp" and cube_family_resource and arg_count == 4:
                texture_function = "textureGatherCompare"
                dropped_parameters.append("status output")
            elif member == "GatherCmp" and arg_count in {4, 5}:
                texture_function = "textureGatherCompareOffset"
                if arg_count == 5:
                    dropped_parameters.append("status output")
            usage = (
                "comparison"
                if member
                in {
                    "SampleCmp",
                    "SampleCmpLevel",
                    "SampleCmpGrad",
                    "SampleCmpBias",
                    "SampleCmpLevelZero",
                    "GatherCmp",
                }
                else "regular"
            )
            return with_dropped_parameters(
                {
                    "member": member,
                    "function": texture_function,
                    "texture_function": texture_function,
                    "buffer_function": None,
                    "component": None,
                    "usage": usage,
                    "buffer_when_max_args": None,
                },
                dropped_parameters,
            )
        return None

    def resource_method_descriptor(self, member, arg_count=None, resource_type=None):
        member = self.templated_method_base(member)
        texture_descriptor = self.texture_method_descriptor(
            member, arg_count, resource_type
        )
        if texture_descriptor:
            descriptor = dict(texture_descriptor)
            uses_buffer = (
                descriptor["buffer_function"] is not None
                and descriptor["function"] == descriptor["buffer_function"]
            )
            if descriptor["function"] == "imageLoad":
                descriptor["resource"] = "image"
            else:
                descriptor["resource"] = "buffer" if uses_buffer else "texture"
            descriptor["operation"] = {
                "Load": "load",
                "GetDimensions": "dimensions",
                "Sample": "sample",
                "SampleLevel": "sample_lod",
                "SampleGrad": "sample_grad",
                "SampleBias": "sample_bias",
                "SampleCmp": "sample_compare",
                "SampleCmpLevel": "sample_compare_lod",
                "SampleCmpGrad": "sample_compare_grad",
                "SampleCmpBias": "sample_compare_bias",
                "SampleCmpLevelZero": "sample_compare",
                "CalculateLevelOfDetail": "query_lod",
                "CalculateLevelOfDetailUnclamped": "query_lod",
                "GetSamplePosition": "query_sample_position",
                "Gather": "gather",
                "GatherRed": "gather",
                "GatherGreen": "gather",
                "GatherBlue": "gather",
                "GatherAlpha": "gather",
                "GatherCmp": "gather_compare",
                "GatherCmpRed": "gather_compare",
                "GatherCmpGreen": "gather_compare",
                "GatherCmpBlue": "gather_compare",
                "GatherCmpAlpha": "gather_compare",
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
                        "IncrementCounter": "increment_counter",
                        "DecrementCounter": "decrement_counter",
                        "Load": "load",
                        "GetDimensions": "dimensions",
                    }.get(member)
                ),
            }
        return None

    @staticmethod
    def templated_method_base(member):
        if isinstance(member, str) and "<" in member:
            return member.split("<", 1)[0]
        return member

    def resource_method_arguments(
        self, obj, member, rendered_args, descriptor, raw_args=None, is_main=False
    ):
        member = self.templated_method_base(member)
        drop_trailing_args = descriptor.get("drop_trailing_args", 0)
        if drop_trailing_args and member != "SampleCmpBias":
            rendered_args = rendered_args[:-drop_trailing_args]
            if raw_args is not None:
                raw_args = raw_args[:-drop_trailing_args]

        if member == "Load" and descriptor["function"] in {
            "texelFetch",
            "texelFetchOffset",
        }:
            return self.texture_load_method_arguments(
                obj, rendered_args, descriptor, raw_args or [], is_main
            )

        if (
            member == "SampleBias"
            and descriptor["function"] == "textureOffset"
            and len(rendered_args) >= 4
        ):
            sampler, coords, bias, offset = rendered_args[:4]
            method_args = [obj, sampler, coords, offset, bias]
        elif member == "SampleCmpBias" and descriptor["function"] == "textureCompare":
            sampler, coords, compare = rendered_args[:3]
            method_args = [obj, sampler, coords, compare]
        elif (
            member == "SampleCmpBias"
            and descriptor["function"] == "textureCompareOffset"
            and len(rendered_args) >= 5
        ):
            sampler, coords, compare, _bias, offset = rendered_args[:5]
            method_args = [obj, sampler, coords, compare, offset]
        else:
            method_args = [obj, *rendered_args]

        if descriptor["component"] is not None:
            method_args.append(descriptor["component"])
        return method_args

    def texture_gather_compare_red_offsets_expression(self, obj, rendered_args):
        sampler, coord, compare, *offset_args = rendered_args[:7]
        components = ("x", "y", "z", "w")
        values = []
        for offset, component in zip(offset_args, components):
            gather = (
                f"textureGatherCompareOffset({obj}, {sampler}, {coord}, "
                f"{compare}, {offset})"
            )
            values.append(f"{gather}.{component}")
        return f"vec4({', '.join(values)})"

    def refine_texture_load_status_descriptor(self, member, args, descriptor):
        if (
            member != "Load"
            or descriptor.get("diagnostic_kind") != "tiled_resource_status"
        ):
            return descriptor

        resource_type = descriptor.get("resource_type")
        if self.is_rw_texture_type(resource_type):
            return descriptor

        resource_base = self.raw_type_base(resource_type)
        if not resource_base.startswith(("Texture", "FeedbackTexture")):
            return descriptor

        status_arg_count = 3 if self.is_multisample_texture_type(resource_type) else 2
        if len(args) != status_arg_count:
            return descriptor

        status_type = self.raw_type_base(self.expression_raw_type(args[-1]))
        if status_type != "uint":
            return descriptor

        descriptor = dict(descriptor)
        descriptor["function"] = descriptor["texture_function"]
        descriptor["drop_trailing_args"] = 1
        descriptor["dropped_parameters"] = ["status output"]
        return descriptor

    def texture_load_method_arguments(
        self, obj, rendered_args, descriptor, raw_args, is_main=False
    ):
        resource_base = self.raw_type_base(descriptor.get("resource_type"))
        is_multisample = resource_base in {"Texture2DMS", "Texture2DMSArray"}
        if is_multisample:
            return [obj, *rendered_args]

        split_location = None
        if raw_args:
            split_location = self.split_texture_load_location(
                raw_args[0], resource_base, is_main
            )

        if split_location:
            coord, lod = split_location
            if descriptor["function"] == "texelFetchOffset" and len(rendered_args) >= 2:
                return [obj, coord, lod, rendered_args[1]]
            return [obj, coord, lod]

        if descriptor["function"] == "texelFetchOffset":
            descriptor["function"] = "texelFetch"
        if descriptor["function"] == "texelFetch" and len(rendered_args) == 1:
            return [obj, rendered_args[0], "0"]
        return [obj, *rendered_args]

    def split_texture_load_location(self, location_arg, resource_base, is_main=False):
        coord_components = {
            "Texture1D": 1,
            "Texture1DArray": 2,
            "Texture2D": 2,
            "Texture2DArray": 3,
            "Texture3D": 3,
        }.get(resource_base)
        if coord_components is None:
            return None

        if not isinstance(location_arg, VectorConstructorNode):
            return None

        ctor_args = getattr(location_arg, "args", None)
        if not ctor_args:
            return None

        if len(ctor_args) == 2 and coord_components > 1:
            return (
                self.generate_expression(ctor_args[0], is_main),
                self.generate_expression(ctor_args[1], is_main),
            )

        if len(ctor_args) != coord_components + 1:
            return None

        lod = self.generate_expression(ctor_args[-1], is_main)
        coord_args = [self.generate_expression(arg, is_main) for arg in ctor_args[:-1]]
        if coord_components == 1:
            coord = coord_args[0]
        else:
            coord = f"ivec{coord_components}({', '.join(coord_args)})"
        return coord, lod

    def resource_method_diagnostic(self, member, descriptor):
        diagnostic_reason = descriptor.get("diagnostic_reason")
        if diagnostic_reason:
            resource = self.raw_type_base(descriptor.get("resource_type")) or "resource"
            label = descriptor.get("diagnostic_label", "texture LOD query")
            return (
                f"/* unsupported DirectX {label} for {resource}: "
                f"{diagnostic_reason} */"
            )
        dropped_parameters = descriptor.get("dropped_parameters")
        if not dropped_parameters:
            return None
        parameters = ", ".join(dropped_parameters)
        if descriptor.get("diagnostic_kind") == "tiled_resource_status":
            return (
                f"/* unsupported DirectX tiled-resource status for {member}: "
                f"dropped {parameters} */"
            )
        return (
            f"/* unsupported DirectX texture overload extras for {member}: "
            f"dropped {parameters} */"
        )

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

    def format_storage_qualifier_prefix(self, node, allowed_qualifiers):
        qualifiers = []
        allowed = {str(qualifier).lower() for qualifier in allowed_qualifiers}
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name in allowed:
                qualifiers.append(qualifier_name)
        return f"{' '.join(qualifiers)} " if qualifiers else ""

    def format_global_storage_qualifier_prefix(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        ordered = []
        if "shared" in qualifiers:
            ordered.append("shared")
        if "groupshared" in qualifiers:
            ordered.append("groupshared")
        if "static" in qualifiers:
            ordered.append("static")
            if "const" in qualifiers:
                ordered.append("const")
        return f"{' '.join(ordered)} " if ordered else ""

    def format_interpolation_attributes(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        attributes = []

        if "nointerpolation" in qualifiers:
            attributes.append("flat")
        elif "noperspective" in qualifiers:
            attributes.append("noperspective")
        elif "linear" in qualifiers:
            attributes.append("smooth")

        if "centroid" in qualifiers:
            attributes.append("centroid")
        elif "sample" in qualifiers:
            attributes.append("sample")

        return " ".join(f"@ {attribute}" for attribute in attributes)

    def format_semantic_and_interpolation_attributes(self, node, semantic):
        attributes = []
        mapped_semantic = self.map_semantic(semantic)
        if mapped_semantic:
            attributes.append(mapped_semantic)
        interpolation_attributes = self.format_interpolation_attributes(node)
        if interpolation_attributes:
            attributes.append(interpolation_attributes)
        return f" {' '.join(attributes)}" if attributes else ""

    def format_inline_parameter_attributes(self, parameter):
        mesh_role_attributes = {"mesh_payload", "vertices", "indices", "primitives"}
        rendered = []
        for attr in getattr(parameter, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", ""))
            if attr_name.lower() not in mesh_role_attributes:
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                rendered_args = ", ".join(self.generate_expression(arg) for arg in args)
                rendered.append(f"@ {attr_name}({rendered_args})")
            else:
                rendered.append(f"@ {attr_name}")
        return " ".join(rendered)

    def infer_ray_parameter_semantic(self, function_qualifier, parameter, index):
        if getattr(parameter, "semantic", None):
            return parameter.semantic

        qualifier = str(function_qualifier or "").lower()
        if qualifier == "ray_miss" and index == 0:
            return "payload"
        if qualifier in {"ray_closest_hit", "ray_any_hit"}:
            if (
                str(getattr(parameter, "vtype", ""))
                == "BuiltInTriangleIntersectionAttributes"
            ):
                return "hit_attribute"
            if index == 0:
                return "payload"
            if index == 1:
                return "hit_attribute"
        if qualifier == "ray_callable" and index == 0:
            return "callable_data"
        return None

    def format_parameter(
        self, parameter, function_qualifier=None, parameter_index=None
    ):
        prefixes = []
        attributes = self.format_inline_parameter_attributes(parameter)
        if attributes:
            prefixes.append(attributes)
            qualifier_prefix = self.format_storage_qualifier_prefix(
                parameter, {"in", "out", "inout", "const"}
            ).strip()
            if qualifier_prefix:
                prefixes.append(qualifier_prefix)
        parameter_name = parameter.name or f"_param{parameter_index}"
        parameter_text = (
            f"{self.map_variable_type(parameter)} {parameter_name}"
            f"{self.format_array_suffixes(parameter)}"
        )
        semantic = self.map_semantic(
            self.infer_ray_parameter_semantic(
                function_qualifier, parameter, parameter_index
            )
        )
        interpolation_attributes = self.format_interpolation_attributes(parameter)
        if semantic or interpolation_attributes:
            parameter_text += " " + " ".join(
                attribute
                for attribute in (semantic, interpolation_attributes)
                if attribute
            )
        prefixes.append(parameter_text)
        return " ".join(prefixes)

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

    def collect_struct_member_types(self, structs):
        member_types = {}
        for struct in structs or []:
            if not isinstance(struct, StructNode):
                continue
            member_types[struct.name] = {
                member.name: getattr(member, "vtype", None)
                for member in getattr(struct, "members", []) or []
            }
        return member_types

    def vector_swizzle_raw_type(self, type_name, swizzle):
        if not type_name or not isinstance(swizzle, str):
            return None
        component_sets = ("xyzw", "rgba")
        component_positions = None
        for components in component_sets:
            if all(component in components for component in swizzle):
                component_positions = [
                    components.index(component) for component in swizzle
                ]
                break
        if component_positions is None:
            return None

        base = str(type_name).strip()
        scalar_prefixes = (
            "min16float",
            "min10float",
            "min16uint",
            "min16int",
            "min12int",
            "double",
            "float",
            "uint",
            "int",
            "bool",
        )
        for scalar in scalar_prefixes:
            for width in range(2, 5):
                if base != f"{scalar}{width}":
                    continue
                if max(component_positions) >= width:
                    continue
                if len(swizzle) == 1:
                    return scalar
                if len(swizzle) <= 4:
                    return f"{scalar}{len(swizzle)}"
        return None

    def member_access_raw_type(self, object_type, member):
        if not object_type:
            return None

        object_base = str(object_type).strip()
        if "[" in object_base:
            object_base = object_base.split("[", 1)[0]

        struct_members = self.struct_member_types.get(object_base)
        if struct_members and member in struct_members:
            return struct_members[member]

        return self.vector_swizzle_raw_type(object_base, member)

    def expression_raw_type(self, expr):
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_raw_type(expr.object)
            member_type = self.member_access_raw_type(object_type, expr.member)
            return member_type if member_type is not None else object_type
        if isinstance(expr, ArrayAccessNode):
            return self.expression_raw_type(expr.array)

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

    def is_multisample_texture_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "Texture2DMS",
            "Texture2DMSArray",
            "RWTexture2DMS",
            "RWTexture2DMSArray",
            "RasterizerOrderedTexture2DMS",
            "RasterizerOrderedTexture2DMSArray",
        }

    def is_buffer_resource_type(self, type_name):
        return self.raw_type_base(type_name) in {
            "Buffer",
            "RWBuffer",
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
            "ByteAddressBuffer",
            "RWByteAddressBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RasterizerOrderedByteAddressBuffer",
        }

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
            yield from node.values()
            return
        if isinstance(node, (list, tuple, set)):
            yield from node
            return
        yield from getattr(node, "__dict__", {}).values()

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

    def collect_variable_types_from_nodes(self, nodes, type_map):
        for node in nodes or []:
            if isinstance(node, VariableNode) and getattr(node, "name", None):
                type_map[node.name] = getattr(node, "vtype", None)
            for child in self.iter_ast_children(node):
                self.collect_variable_types_from_nodes([child], type_map)

    def expression_raw_type_from_map(self, expr, variable_types):
        if isinstance(expr, MemberAccessNode):
            object_type = self.expression_raw_type_from_map(expr.object, variable_types)
            member_type = self.member_access_raw_type(object_type, expr.member)
            return member_type if member_type is not None else object_type
        if isinstance(expr, ArrayAccessNode):
            return self.expression_raw_type_from_map(expr.array, variable_types)
        if isinstance(expr, VariableNode):
            return variable_types.get(expr.name)
        if isinstance(expr, str):
            return variable_types.get(expr)
        return None

    def expression_struct_member_key(self, expr, variable_types):
        while isinstance(expr, ArrayAccessNode):
            expr = expr.array
        if not isinstance(expr, MemberAccessNode):
            return None
        owner_type = self.expression_raw_type_from_map(expr.object, variable_types)
        owner_base = self.raw_type_base(owner_type)
        if owner_base not in self.struct_member_types:
            return None
        return owner_base, expr.member

    def collect_direct_texture_member_usage_keys(self, root, variable_types):
        comparison_keys = set()
        regular_keys = set()

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, FunctionCallNode) and isinstance(
                node.name, MemberAccessNode
            ):
                resource_type = self.expression_raw_type_from_map(
                    node.name.object, variable_types
                )
                descriptor = self.resource_method_descriptor(
                    node.name.member,
                    len(getattr(node, "args", []) or []),
                    resource_type,
                )
                usage = (
                    descriptor["usage"]
                    if descriptor and descriptor["resource"] == "texture"
                    else None
                )
                member_key = self.expression_struct_member_key(
                    node.name.object, variable_types
                )
                if member_key and usage == "comparison":
                    comparison_keys.add(member_key)
                elif member_key and usage == "regular":
                    regular_keys.add(member_key)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return comparison_keys, regular_keys

    def collect_texture_member_usage_keys(self, root):
        comparison_keys = set()
        regular_keys = set()
        global_variable_types = {
            getattr(node, "name", None): getattr(node, "vtype", None)
            for node in getattr(root, "global_variables", []) or []
        }
        global_variable_types.pop(None, None)

        for node in getattr(root, "global_variables", []) or []:
            direct_comparison, direct_regular = (
                self.collect_direct_texture_member_usage_keys(
                    node, global_variable_types
                )
            )
            comparison_keys.update(direct_comparison)
            regular_keys.update(direct_regular)

        for func in getattr(root, "functions", []) or []:
            function_variable_types = dict(global_variable_types)
            for param in getattr(func, "params", []) or []:
                if getattr(param, "name", None):
                    function_variable_types[param.name] = getattr(param, "vtype", None)
            self.collect_variable_types_from_nodes(
                getattr(func, "body", []), function_variable_types
            )
            direct_comparison, direct_regular = (
                self.collect_direct_texture_member_usage_keys(
                    getattr(func, "body", []), function_variable_types
                )
            )
            comparison_keys.update(direct_comparison)
            regular_keys.update(direct_regular)

        return comparison_keys, regular_keys

    def function_variable_type_maps(self, root):
        global_variable_types = {
            getattr(node, "name", None): getattr(node, "vtype", None)
            for node in getattr(root, "global_variables", []) or []
        }
        global_variable_types.pop(None, None)
        type_maps = {}

        for func in getattr(root, "functions", []) or []:
            function_variable_types = dict(global_variable_types)
            for param in getattr(func, "params", []) or []:
                if getattr(param, "name", None):
                    function_variable_types[param.name] = getattr(param, "vtype", None)
            self.collect_variable_types_from_nodes(
                getattr(func, "body", []), function_variable_types
            )
            type_maps[getattr(func, "name", None)] = function_variable_types

        type_maps.pop(None, None)
        return type_maps

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

    def propagate_function_texture_member_usage(
        self, root, function_usage, member_comparison_keys, member_regular_keys
    ):
        type_maps = self.function_variable_type_maps(root)
        changed = False

        def apply_call_usage(node, caller_function_name):
            nonlocal changed
            if not isinstance(node, FunctionCallNode) or not isinstance(node.name, str):
                return
            callee_usage = function_usage.get(node.name)
            if not callee_usage:
                return
            variable_types = type_maps.get(caller_function_name, {})
            for usage_kind, usage_keys in (
                ("comparison", member_comparison_keys),
                ("regular", member_regular_keys),
            ):
                for param_index in callee_usage[usage_kind]:
                    if param_index >= len(node.args):
                        continue
                    member_key = self.expression_struct_member_key(
                        node.args[param_index], variable_types
                    )
                    if member_key and member_key not in usage_keys:
                        usage_keys.add(member_key)
                        changed = True

        def visit(node, caller_function_name):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            apply_call_usage(node, caller_function_name)
            for child in self.iter_ast_children(node):
                visit(child, caller_function_name)

        for func in getattr(root, "functions", []) or []:
            visit(getattr(func, "body", []), getattr(func, "name", None))

        return changed

    def collect_shadow_texture_names(self, root):
        comparison_names, regular_names = self.collect_nonparameter_texture_usage_names(
            root
        )
        member_comparison_keys, member_regular_keys = (
            self.collect_texture_member_usage_keys(root)
        )
        function_usage = self.collect_function_texture_parameter_usage(root)
        while self.propagate_function_texture_usage(
            root, function_usage, comparison_names, regular_names
        ):
            pass
        while self.propagate_function_texture_member_usage(
            root, function_usage, member_comparison_keys, member_regular_keys
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

        shadow_member_keys = member_comparison_keys - member_regular_keys
        for struct in getattr(root, "structs", []) or []:
            if not isinstance(struct, StructNode):
                continue
            for member in getattr(struct, "members", []) or []:
                if (struct.name, getattr(member, "name", None)) in shadow_member_keys:
                    self.shadow_texture_declaration_ids.add(id(member))

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
        self.struct_member_types = self.collect_struct_member_types(ast.structs)
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
                    code += f"    type {alias.name} = {self.map_type(alias_type)};\n"
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
                    attributes = self.format_semantic_and_interpolation_attributes(
                        member, member.semantic
                    )
                    code += (
                        f"        {self.map_variable_type(member)} "
                        f"{member.name}{array_suffix}{attributes};\n"
                    )
                code += "    }\n"
            elif isinstance(node, PragmaNode):
                value = f" {node.value}" if node.value else ""
                code += f"    #pragma {node.directive}{value}\n"
            elif isinstance(node, IncludeNode):
                code += f"    #include {node.path}\n"
        for node in ast.global_variables:
            self.record_variable_type(
                node, self.global_variable_types, self.global_resource_array_dims
            )
            code += self.format_attributes(getattr(node, "attributes", []), 1)
            code += self.format_resource_qualifier_attributes(node, 1)
            code += self.format_binding_attributes(node, 1)
            storage_prefix = self.format_global_storage_qualifier_prefix(node)
            array_suffix = self.format_array_suffixes(node)
            initializer = ""
            if getattr(node, "value", None) is not None:
                initializer = f" = {self.generate_expression(node.value)}"
            code += (
                f"    {storage_prefix}{self.map_variable_type(node)} "
                f"{node.name}{array_suffix}{initializer};\n"
            )
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
                entry_name = "main" if stage_name.startswith("ray_") else None
                code += self.generate_function(
                    func,
                    skip_attribute_names={"shader"},
                    entry_name=entry_name,
                )
                code += "    }\n\n"
            else:
                code += self.generate_function(func)

        code += "}\n"
        return code

    def generate_cbuffers(self, ast):
        code = ""
        for node in ast.cbuffers:
            if isinstance(node, StructNode):
                attributes = list(getattr(node, "attributes", []) or [])
                if getattr(node, "is_tbuffer", False):
                    attributes = [AttributeNode("tbuffer")] + attributes
                code += self.format_attributes(attributes, 1)
                code += self.format_binding_attributes(node, 1)
                code += f"    cbuffer {node.name} {{\n"
                for member in node.members:
                    for qualifier in getattr(member, "qualifiers", []) or []:
                        if qualifier in {"row_major", "column_major"}:
                            code += f"        @ {qualifier}\n"
                    code += self.format_attributes(getattr(member, "attributes", []), 2)
                    code += self.format_binding_attributes(member, 2)
                    array_suffix = self.format_array_suffixes(member)
                    code += (
                        f"        {self.map_variable_type(member)} "
                        f"{member.name}{array_suffix};\n"
                    )
                code += "    }\n"
        return code

    def generate_function(
        self, func, indent=1, skip_attribute_names=None, entry_name=None
    ):
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
        qualifier = getattr(func, "qualifier", None)
        params = ", ".join(
            self.format_parameter(p, qualifier, index)
            for index, p in enumerate(func.params)
        )
        semantic = self.map_semantic(func.semantic)
        semantic = f" {semantic}" if semantic else ""
        function_name = entry_name or func.name
        code += (
            f"{self.map_type(func.return_type)} "
            f"{function_name}({params}){semantic} {{\n"
        )
        code += self.generate_function_body(func.body, indent=indent + 1)
        code += "    " * indent + "}\n\n"
        self.current_variable_types = previous_variable_types
        self.current_resource_array_dims = previous_resource_array_dims
        return code

    def generate_function_body(self, body, indent=0, is_main=False):
        code = ""
        for stmt in body:
            if isinstance(stmt, FunctionCallNode):
                code += self.generate_function_call_statement(stmt, indent, is_main)
                continue
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
            elif isinstance(stmt, str):
                code += f"{stmt};\n"
            else:
                code += f"// Unhandled statement type: {type(stmt).__name__}\n"
        return code

    def generate_function_call_statement(self, stmt, indent=0, is_main=False):
        clip_statement = self.generate_clip_statement(stmt, indent, is_main)
        if clip_statement is not None:
            return clip_statement
        lowered_sequence = self.generate_function_call_statement_sequence(stmt, indent)
        if lowered_sequence is not None:
            return lowered_sequence
        lowered_get_dimensions = self.generate_get_dimensions_statement(
            stmt, indent, is_main
        )
        if lowered_get_dimensions is not None:
            return lowered_get_dimensions
        return "    " * indent + f"{self.generate_expression(stmt, is_main)};\n"

    def generate_clip_statement(self, stmt, indent=0, is_main=False):
        if not isinstance(stmt.name, str) or stmt.name != "clip" or not stmt.args:
            return None
        condition = self.generate_clip_condition(stmt.args[0], is_main)
        indent_text = "    " * indent
        return (
            f"{indent_text}if ({condition}) {{\n"
            f"{indent_text}    discard;\n"
            f"{indent_text}}}\n"
        )

    def generate_clip_condition(self, operand_expr, is_main=False):
        operand = self.generate_expression(operand_expr, is_main)
        operand = self.maybe_parenthesize(operand_expr, operand)
        component_count = self.clip_operand_component_count(operand_expr)
        if component_count and component_count > 1:
            return f"any(lessThan({operand}, vec{component_count}(0.0)))"
        return f"{operand} < 0.0"

    def clip_operand_component_count(self, operand_expr):
        raw_type = self.expression_raw_type(operand_expr)
        if raw_type is None:
            return None
        mapped_type = self.map_type(raw_type)
        for prefix in ("vec", "ivec", "uvec", "bvec", "dvec", "f16vec"):
            if mapped_type.startswith(prefix):
                suffix = mapped_type[len(prefix) :]
                if suffix.isdigit():
                    return int(suffix)
        return None

    def generate_function_call_statement_sequence(self, stmt, indent=0):
        if not isinstance(stmt.name, str):
            return None
        sequence = self.function_statement_sequences.get(stmt.name)
        if sequence is None or stmt.args:
            return None
        indent_text = "    " * indent
        return "".join(f"{indent_text}{func_name}();\n" for func_name in sequence)

    def generate_get_dimensions_statement(self, stmt, indent=0, is_main=False):
        if not isinstance(stmt.name, MemberAccessNode):
            return None
        member = self.templated_method_base(stmt.name.member)
        if member != "GetDimensions":
            return None

        resource_type = self.expression_raw_type(stmt.name.object)
        resource_base = self.raw_type_base(resource_type)
        if not resource_base:
            return None

        obj = self.generate_expression(stmt.name.object, is_main)
        rendered_args = [self.generate_expression(arg, is_main) for arg in stmt.args]
        indent_text = "    " * indent

        if self.is_buffer_resource_type(resource_type):
            return (
                indent_text
                + f"buffer_dimensions({', '.join([obj, *rendered_args])});\n"
            )

        assignments = self.get_dimensions_assignments(
            resource_base, obj, stmt.args, rendered_args, is_main
        )
        if assignments is None:
            descriptor = self.resource_method_descriptor(
                member, len(stmt.args), resource_type
            )
            function = descriptor["function"] if descriptor else "texture_dimensions"
            diagnostic = (
                f"/* unsupported DirectX GetDimensions overload for {resource_base}: "
                "preserved dimension helper call */"
            )
            return (
                indent_text
                + f"{diagnostic} {function}({', '.join([obj, *rendered_args])});\n"
            )
        return "".join(indent_text + assignment + ";\n" for assignment in assignments)

    def get_dimensions_assignments(
        self, resource_base, obj, raw_args, rendered_args, is_main=False
    ):
        layout = self.get_dimensions_layout(resource_base, obj, rendered_args)
        if layout is None:
            return None

        size_expr = layout["size_expr"]
        component_suffixes = layout["component_suffixes"]
        assignments = []
        for target, suffix, arg_index in zip(
            layout["dimension_targets"],
            component_suffixes,
            layout["dimension_indices"],
        ):
            value = f"{size_expr}{suffix}"
            assignments.append(
                f"{target} = {self.cast_get_dimensions_value(value, raw_args[arg_index])}"
            )

        if layout.get("levels_target") is not None:
            value = f"textureQueryLevels({obj})"
            assignments.append(
                f"{layout['levels_target']} = "
                f"{self.cast_get_dimensions_value(value, raw_args[layout['levels_index']])}"
            )
        if layout.get("samples_target") is not None:
            value = f"textureSamples({obj})"
            assignments.append(
                f"{layout['samples_target']} = "
                f"{self.cast_get_dimensions_value(value, raw_args[layout['samples_index']])}"
            )
        return assignments

    def get_dimensions_layout(self, resource_base, obj, rendered_args):
        args_count = len(rendered_args)
        if args_count == 0:
            return None

        texture_dimensions = {
            "Texture1D": 1,
            "Texture1DArray": 2,
            "Texture2D": 2,
            "Texture2DArray": 3,
            "Texture3D": 3,
            "TextureCube": 2,
            "TextureCubeArray": 3,
            "FeedbackTexture2D": 2,
            "FeedbackTexture2DArray": 3,
        }
        image_dimensions = {
            "RWTexture1D": 1,
            "RWTexture1DArray": 2,
            "RWTexture2D": 2,
            "RWTexture2DArray": 3,
            "RWTexture3D": 3,
            "RWTextureCube": 2,
            "RWTextureCubeArray": 3,
            "RasterizerOrderedTexture1D": 1,
            "RasterizerOrderedTexture1DArray": 2,
            "RasterizerOrderedTexture2D": 2,
            "RasterizerOrderedTexture2DArray": 3,
            "RasterizerOrderedTexture3D": 3,
        }
        multisample_dimensions = {
            "Texture2DMS": 2,
            "Texture2DMSArray": 3,
            "RWTexture2DMS": 2,
            "RWTexture2DMSArray": 3,
            "RasterizerOrderedTexture2DMS": 2,
            "RasterizerOrderedTexture2DMSArray": 3,
        }

        if resource_base in multisample_dimensions:
            dimensions = multisample_dimensions[resource_base]
            if args_count not in {dimensions, dimensions + 1}:
                return None
            size_function = (
                "imageSize"
                if resource_base.startswith(("RWTexture", "RasterizerOrderedTexture"))
                else "textureSize"
            )
            return {
                "size_expr": f"{size_function}({obj})",
                "dimension_targets": rendered_args[:dimensions],
                "dimension_indices": list(range(dimensions)),
                "component_suffixes": self.dimension_component_suffixes(dimensions),
                "samples_target": (
                    rendered_args[dimensions] if args_count == dimensions + 1 else None
                ),
                "samples_index": dimensions if args_count == dimensions + 1 else None,
            }

        if resource_base in image_dimensions:
            dimensions = image_dimensions[resource_base]
            if args_count != dimensions:
                return None
            return {
                "size_expr": f"imageSize({obj})",
                "dimension_targets": rendered_args,
                "dimension_indices": list(range(dimensions)),
                "component_suffixes": self.dimension_component_suffixes(dimensions),
            }

        dimensions = texture_dimensions.get(resource_base)
        if dimensions is None:
            return None

        lod = "0"
        out_start = 0
        levels_index = None
        if args_count == dimensions:
            pass
        elif args_count == dimensions + 1:
            levels_index = dimensions
        elif args_count == dimensions + 2:
            lod = rendered_args[0]
            out_start = 1
            levels_index = dimensions + 1
        else:
            return None

        dimension_targets = rendered_args[out_start : out_start + dimensions]
        if len(dimension_targets) != dimensions:
            return None
        return {
            "size_expr": f"textureSize({obj}, {lod})",
            "dimension_targets": dimension_targets,
            "dimension_indices": list(range(out_start, out_start + dimensions)),
            "component_suffixes": self.dimension_component_suffixes(dimensions),
            "levels_target": (
                rendered_args[levels_index] if levels_index is not None else None
            ),
            "levels_index": levels_index,
        }

    def dimension_component_suffixes(self, dimensions):
        if dimensions == 1:
            return [""]
        return [f".{component}" for component in ("x", "y", "z")[:dimensions]]

    def cast_get_dimensions_value(self, value, target_arg):
        if self.get_dimensions_target_scalar_type(target_arg) == "uint":
            return f"uint({value})"
        return value

    def get_dimensions_target_scalar_type(self, target_arg):
        if target_arg is None:
            return None

        mapped_type = self.map_type(self.expression_raw_type(target_arg))
        if mapped_type == "uint":
            return "uint"

        if not isinstance(target_arg, MemberAccessNode):
            return mapped_type

        if target_arg.member not in {"x", "y", "z", "w", "r", "g", "b", "a"}:
            return mapped_type

        if mapped_type in {"uvec2", "uvec3", "uvec4"}:
            return "uint"
        return mapped_type

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
        def render_initializer(initializer):
            if isinstance(initializer, VariableNode):
                array_suffix = self.format_array_suffixes(initializer, is_main)
                text = (
                    f"{self.map_variable_type(initializer)} "
                    f"{initializer.name}{array_suffix}"
                )
                if initializer.value is not None:
                    text += f" = {self.generate_expression(initializer.value, is_main)}"
                self.record_variable_type(initializer)
                return text
            return self.generate_expression(initializer, is_main)

        if isinstance(node.init, list):
            init = ", ".join(render_initializer(item) for item in node.init)
        elif isinstance(node.init, VariableNode):
            init = render_initializer(node.init)
        elif node.init is None:
            init = ""
        else:
            init = self.generate_expression(node.init, is_main)

        condition = (
            self.generate_expression(node.condition, is_main)
            if node.condition is not None
            else ""
        )
        if isinstance(node.update, list):
            update = ", ".join(
                self.generate_expression(expr, is_main) for expr in node.update
            )
        else:
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
                method_member = self.templated_method_base(member)
                rendered_args = [
                    self.generate_expression(arg, is_main) for arg in expr.args
                ]
                args = ", ".join(rendered_args)
                resource_type = self.expression_raw_type(expr.name.object)
                resource_base = self.raw_type_base(resource_type)
                if (
                    method_member == "GatherCmpRed"
                    and len(rendered_args) in {7, 8}
                    and resource_base in {"Texture2D", "Texture2DArray"}
                ):
                    call = self.texture_gather_compare_red_offsets_expression(
                        obj, rendered_args[:7]
                    )
                    if len(rendered_args) == 8:
                        diagnostic = self.resource_method_diagnostic(
                            method_member, {"dropped_parameters": ["status output"]}
                        )
                        return f"{diagnostic} {call}"
                    return call
                descriptor = self.resource_method_descriptor(
                    method_member, len(expr.args), resource_type
                )
                if descriptor:
                    descriptor = self.refine_texture_load_status_descriptor(
                        method_member, expr.args, descriptor
                    )
                    if descriptor.get("fallback_expression") is not None:
                        call = descriptor["fallback_expression"]
                    else:
                        method_args = self.resource_method_arguments(
                            obj,
                            method_member,
                            rendered_args,
                            descriptor,
                            expr.args,
                            is_main,
                        )
                        call = f"{descriptor['function']}({', '.join(method_args)})"
                        result_component = descriptor.get("result_component")
                        if result_component:
                            call = f"{call}{result_component}"
                    diagnostic = self.resource_method_diagnostic(
                        method_member, descriptor
                    )
                    if diagnostic:
                        return f"{diagnostic} {call}"
                    return call
                return f"{obj}.{member}({args})"

            func_name = (
                expr.name
                if isinstance(expr.name, str)
                else self.generate_expression(expr.name, is_main)
            )
            if func_name == "CheckAccessFullyMapped":
                return (
                    "/* unsupported DirectX tiled-resource status check: "
                    "CheckAccessFullyMapped assumed fully mapped */ true"
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
            if func_name == "mul" and len(expr.args) == 2:
                left = self.maybe_parenthesize(expr.args[0], rendered_args[0])
                right = self.maybe_parenthesize(expr.args[1], rendered_args[1])
                return f"({left} * {right})"
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
        elif isinstance(expr, InitializerListNode):
            elements = ", ".join(
                self.generate_expression(element, is_main) for element in expr.elements
            )
            return f"{{{elements}}}"
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
            base = base.strip()
            generic_type = generic_args[:-1].strip()
            vector_or_matrix_type = self.map_template_vector_or_matrix_type(
                base, generic_type
            )
            if vector_or_matrix_type:
                return vector_or_matrix_type
            rasterizer_buffer_type = self.map_rasterizer_ordered_buffer_type(
                base, generic_type
            )
            if rasterizer_buffer_type:
                return rasterizer_buffer_type
            feedback_texture_type = self.map_feedback_texture_type(base, generic_type)
            if feedback_texture_type:
                return feedback_texture_type
            if base in self.structured_buffer_types:
                mapped_args = ", ".join(
                    self.sanitize_type_name(arg)
                    for arg in self.split_generic_arguments(generic_type)
                )
                return f"{self.sanitize_type_name(base)}<{mapped_args}>"
            storage_image_type = self.map_rw_texture_type(base, generic_type)
            if storage_image_type:
                return storage_image_type
            type_name = base
        default_template_type = {
            "vector": "vec4",
            "matrix": "mat4",
        }.get(type_name)
        if default_template_type:
            return default_template_type
        return self.type_map.get(type_name, self.sanitize_type_name(type_name))

    def sanitize_type_name(self, type_name):
        """Convert HLSL scoped type paths into CrossGL identifier-safe names."""
        return str(type_name).replace("::", "_")

    def map_template_vector_or_matrix_type(self, base_type, generic_type):
        args = self.split_generic_arguments(generic_type)
        if base_type == "vector":
            scalar_type = args[0] if args else "float"
            components = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
            return self.map_template_vector_type(scalar_type, components)
        if base_type == "matrix":
            scalar_type = args[0] if args else "float"
            rows = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
            cols = self.parse_template_dimension(args[2]) if len(args) > 2 else 4
            return self.map_template_matrix_type(scalar_type, rows, cols)
        return None

    def split_generic_arguments(self, generic_type):
        args = []
        depth = 0
        current = []
        for char in generic_type:
            if char == "<":
                depth += 1
            elif char == ">" and depth:
                depth -= 1
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current or generic_type.strip():
            args.append("".join(current).strip())
        return args

    def parse_template_dimension(self, value):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None

    def map_template_vector_type(self, scalar_type, components):
        if components is None or components < 1 or components > 4:
            return None
        scalar = str(scalar_type).strip()
        if components == 1:
            return self.map_type(scalar)
        prefixes = {
            "float": "vec",
            "half": "f16vec",
            "min16float": "f16vec",
            "min10float": "f16vec",
            "float16_t": "f16vec",
            "double": "dvec",
            "int": "ivec",
            "min16int": "i16vec",
            "min12int": "i16vec",
            "int16_t": "i16vec",
            "uint": "uvec",
            "min16uint": "u16vec",
            "uint16_t": "u16vec",
            "bool": "bvec",
        }
        prefix = prefixes.get(scalar)
        if prefix is None:
            return None
        return f"{prefix}{components}"

    def map_template_matrix_type(self, scalar_type, rows, cols):
        if rows is None or cols is None or rows < 2 or rows > 4 or cols < 2 or cols > 4:
            return None
        prefixes = {
            "float": "mat",
            "half": "f16mat",
            "min16float": "f16mat",
            "min10float": "f16mat",
            "float16_t": "f16mat",
            "double": "dmat",
        }
        prefix = prefixes.get(str(scalar_type).strip())
        if prefix is None:
            return None
        suffix = str(rows) if rows == cols else f"{rows}x{cols}"
        return f"{prefix}{suffix}"

    def map_rasterizer_ordered_buffer_type(self, base_type, element_type):
        buffer_type = {
            "RasterizerOrderedBuffer": "RWBuffer",
            "RasterizerOrderedStructuredBuffer": "RWStructuredBuffer",
        }.get(base_type)
        if buffer_type is None:
            return None
        return f"{buffer_type}<{element_type}>"

    def map_feedback_texture_type(self, base_type, feedback_type):
        texture_type = {
            "FeedbackTexture2D": "feedbackTexture2D",
            "FeedbackTexture2DArray": "feedbackTexture2DArray",
        }.get(base_type)
        if texture_type is None:
            return None
        return f"{texture_type}<{feedback_type}>"

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
        payload_access = []
        if isinstance(semantic, str):
            for access_name, access_args in re.findall(
                r"\b(read|write)\s*\(([^)]*)\)", semantic
            ):
                args = ", ".join(
                    arg.strip() for arg in access_args.split(",") if arg.strip()
                )
                payload_access.append(f"@ hlsl_{access_name}({args})")
        if payload_access:
            return " ".join(payload_access)
        mapped = self.semantic_map.get(semantic)
        if mapped is None and isinstance(semantic, str):
            semantic_upper = semantic.upper()
            mapped = self.semantic_map.get(semantic_upper)
            if mapped is None:
                for hlsl_prefix, crossgl_semantic in (
                    ("SV_CLIPDISTANCE", "gl_ClipDistance"),
                    ("SV_CULLDISTANCE", "gl_CullDistance"),
                ):
                    if not semantic_upper.startswith(hlsl_prefix):
                        continue
                    suffix = semantic_upper[len(hlsl_prefix) :]
                    if not suffix or suffix.isdigit():
                        mapped = crossgl_semantic
                        break
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
            attributes = self.format_semantic_and_interpolation_attributes(
                member, member.semantic
            )

            array_suffix = self.format_array_suffixes(member)
            code += (
                self.get_indent() + f"{self.map_variable_type(member)} "
                f"{member.name}{array_suffix}{attributes};\n"
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
