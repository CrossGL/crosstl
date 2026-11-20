"""Constant HLSL import tables: shadow texture types, intrinsic and method maps, semantics and operators.

Each ``HLSLToCrossGLConverter`` instance copies these tables in ``__init__``, so per-instance changes stay local.
"""

HLSL_SHADOW_TEXTURE_TYPES = {
    "Texture1D": "texture1DShadow",
    "Texture1DArray": "texture1DArrayShadow",
    "Texture2D": "texture2DShadow",
    "Texture2DArray": "texture2DArrayShadow",
    "TextureCube": "textureCubeShadow",
    "TextureCubeArray": "textureCubeArrayShadow",
}


HLSL_FUNCTION_MAP = {
    "countbits": "bitCount",
    "ddx": "dFdx",
    "ddx_coarse": "dFdxCoarse",
    "ddx_fine": "dFdxFine",
    "ddy": "dFdy",
    "ddy_coarse": "dFdyCoarse",
    "ddy_fine": "dFdyFine",
    "firstbithigh": "findMSB",
    "firstbitlow": "findLSB",
    "frac": "fract",
    "fmod": "fmod",
    "lerp": "mix",
    "reversebits": "bitfieldReverse",
    # HLSL round rounds halfway cases to the nearest even integer (DXC lowers it to Round_ne); CrossGL round
    # leaves them implementation-defined, as GLSL does.
    "round": "roundEven",
    "rsqrt": "inverseSqrt",
    "atan2": "atan",
    "EvaluateAttributeAtSample": "interpolateAtSample",
    "EvaluateAttributeSnapped": "interpolateAtOffset",
    "EvaluateAttributeCentroid": "interpolateAtCentroid",
    "GroupMemoryBarrierWithGroupSync": "workgroupBarrier",
    # GroupMemoryBarrier orders groupshared memory only: GLSL's memoryBarrierShared, not groupMemoryBarrier,
    # which orders buffer and image accesses as well.
    "GroupMemoryBarrier": "memoryBarrierShared",
    "DeviceMemoryBarrier": "deviceMemoryBarrier",
    "AllMemoryBarrier": "allMemoryBarrier",
}


# A barrier with group sync that orders device memory releases the writes before it and acquires them
# after it: a fence on each side of the execution barrier. DeviceMemoryBarrierWithGroupSync does not order
# groupshared memory, so its execution barrier is workgroupExecutionBarrier.
HLSL_FUNCTION_STATEMENT_SEQUENCES = {
    "DeviceMemoryBarrierWithGroupSync": (
        "deviceMemoryBarrier",
        "workgroupExecutionBarrier",
        "deviceMemoryBarrier",
    ),
    "AllMemoryBarrierWithGroupSync": (
        "allMemoryBarrier",
        "workgroupBarrier",
        "allMemoryBarrier",
    ),
}


HLSL_INTERLOCKED_FUNCTIONS = {
    "InterlockedAdd": "atomicAdd",
    "InterlockedAnd": "atomicAnd",
    "InterlockedOr": "atomicOr",
    "InterlockedXor": "atomicXor",
    "InterlockedMin": "atomicMin",
    "InterlockedMax": "atomicMax",
    "InterlockedExchange": "atomicExchange",
    "InterlockedCompareExchange": "atomicCompareExchange",
}


HLSL_BYTE_ADDRESS_INTERLOCKED_METHODS = {
    "InterlockedAdd": "atomicAdd",
    "InterlockedAnd": "atomicAnd",
    "InterlockedOr": "atomicOr",
    "InterlockedXor": "atomicXor",
    "InterlockedMin": "atomicMin",
    "InterlockedMax": "atomicMax",
    "InterlockedExchange": "atomicExchange",
    "InterlockedCompareExchange": "atomicCompareExchange",
    "InterlockedCompareStore": "atomicCompareExchange",
}


HLSL_TEXTURE_METHODS = {
    "Sample": "texture",
    "SampleLevel": "textureLod",
    "SampleGrad": "textureGrad",
    "SampleBias": "texture",
    "SampleCmp": "textureCompare",
    "SampleCmpLevel": "textureCompareLod",
    "SampleCmpGrad": "textureCompareGrad",
    "SampleCmpBias": "textureCompare",
    # SampleCmpLevelZero compares at mip level 0, with no implicit derivatives.
    "SampleCmpLevelZero": "textureCompareLod",
    "Load": "texelFetch",
    "Gather": "textureGather",
    "GatherCmp": "textureGatherCompare",
    "GetDimensions": "texture_dimensions",
}


HLSL_FEEDBACK_METHODS = {
    "WriteSamplerFeedback": "write_sampler_feedback",
    "WriteSamplerFeedbackBias": "write_sampler_feedback_bias",
    "WriteSamplerFeedbackGrad": "write_sampler_feedback_grad",
    "WriteSamplerFeedbackLevel": "write_sampler_feedback_level",
}


HLSL_TEXTURE_GATHER_COMPONENTS = {
    "GatherRed": "0",
    "GatherGreen": "1",
    "GatherBlue": "2",
    "GatherAlpha": "3",
}


HLSL_TEXTURE_GATHER_COMPARE_COMPONENTS = {
    "GatherCmpRed": "0",
    "GatherCmpGreen": "1",
    "GatherCmpBlue": "2",
    "GatherCmpAlpha": "3",
}


HLSL_LEGACY_TEXTURE_FUNCTION_SAMPLER_TYPES = {
    "tex1D": {"sampler", "sampler1D"},
    "tex2D": {"sampler2D", "sampler2D_half", "sampler2D_float"},
    "tex3D": {"sampler3D"},
    "texCUBE": {"samplerCUBE", "samplerCube"},
    "tex2Dlod": {"sampler2D", "sampler2D_half", "sampler2D_float"},
    "tex2Dbias": {"sampler2D", "sampler2D_half", "sampler2D_float"},
    "tex2Dgrad": {"sampler2D", "sampler2D_half", "sampler2D_float"},
}


HLSL_BUFFER_METHODS = {
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


HLSL_SEMANTICS = {
    # System-value semantics - Vertex inputs. SV_VertexID excludes the
    # draw's base vertex, which gl_VertexID includes; see
    # crosstl.translator.semantics.relative_vertex_ids.
    "SV_VertexID": "gl_RelativeVertexID",
    "SV_InstanceID": "gl_InstanceID",
    # Shader Model 6.8 draw parameters; DXC lowers them to SPIR-V's
    # BuiltIn BaseVertex and BaseInstance.
    "SV_StartVertexLocation": "gl_BaseVertex",
    "SV_StartInstanceLocation": "gl_BaseInstance",
    "SV_PrimitiveID": "gl_PrimitiveID",
    # A mesh primitive's cull flag (GL_EXT_mesh_shader's built-in).
    "SV_CullPrimitive": "gl_CullPrimitiveEXT",
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
    "SV_StencilRef": "gl_FragStencilRefEXT",
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
    "SV_ViewID": "gl_ViewIndex",
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
    "VERTEX_ID": "gl_RelativeVertexID",
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


HLSL_BITWISE_OPERATORS = {
    "&": "bitAnd",
    "|": "bitOr",
    "^": "bitXor",
    "~": "bitNot",
    "<<": "bitShiftLeft",
    ">>": "bitShiftRight",
}


HLSL_STRUCTURED_BUFFER_TYPES = {
    "Buffer",
    "ConstantBuffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "TextureBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
}


HLSL_GENERIC_PASSTHROUGH_TYPES = {
    "DispatchNodeInputRecord",
    "RWDispatchNodeInputRecord",
    "GroupNodeInputRecords",
    "RWGroupNodeInputRecords",
    "ThreadNodeInputRecord",
    "RWThreadNodeInputRecord",
    "NodeOutput",
    "NodeOutputArray",
    "ThreadNodeOutputRecords",
    "GroupNodeOutputRecords",
    "NodeOutputRecords",
}
