"""Static tables of the Metal to CrossGL converter: stage qualifiers, type names, texture methods,
function attributes and stage-attribute semantics. ``MetalToCrossGLConverter`` keeps a per-instance copy of each.
"""

from ..vote_masks import MASK_CROSSGL_TYPES

# Metal function qualifiers of ray-tracing, mesh and object stages.
METAL_RAY_TRACING_STAGE_QUALIFIERS = {
    "intersection",
    "anyhit",
    "closesthit",
    "miss",
    "callable",
    "mesh",
    "object",
    "amplification",
}

# Metal type names and their CrossGL spellings.
METAL_TO_CROSSGL_TYPES = {
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
    "xhalf": "float16",
    "bfloat": "bfloat16",
    "double": "double",
    "size_t": "uint64",
    "ptrdiff_t": "int64",
    "atomic_int": "atomic_int",
    "atomic_uint": "atomic_uint",
    "atomic_bool": "atomic_bool",
    "atomic_ulong": "atomic_ulong",
    "atomic_float": "atomic_float",
    # Vector Types - float
    "float2": "vec2",
    "float3": "vec3",
    "float4": "vec4",
    # Vector Types - half
    "half2": "f16vec2",
    "half3": "f16vec3",
    "half4": "f16vec4",
    "xhalf2": "f16vec2",
    "xhalf3": "f16vec3",
    "xhalf4": "f16vec4",
    # Vector Types - bfloat
    "bfloat2": "bfloat16vec2",
    "bfloat3": "bfloat16vec3",
    "bfloat4": "bfloat16vec4",
    # Vector Types - int
    "int2": "ivec2",
    "int3": "ivec3",
    "int4": "ivec4",
    # Vector Types - uint
    "uint2": "uvec2",
    "uint3": "uvec3",
    "uint4": "uvec4",
    # SIMD-group and quad ballot masks and their vote_t integers (see vote_masks.py)
    **MASK_CROSSGL_TYPES,
    # Vector Types - 64-bit int
    "long2": "i64vec2",
    "long3": "i64vec3",
    "long4": "i64vec4",
    "ulong2": "u64vec2",
    "ulong3": "u64vec3",
    "ulong4": "u64vec4",
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
    "packed_char2": "i8vec2",
    "packed_char3": "i8vec3",
    "packed_char4": "i8vec4",
    "packed_uchar2": "u8vec2",
    "packed_uchar3": "u8vec3",
    "packed_uchar4": "u8vec4",
    "packed_short2": "i16vec2",
    "packed_short3": "i16vec3",
    "packed_short4": "i16vec4",
    "packed_ushort2": "u16vec2",
    "packed_ushort3": "u16vec3",
    "packed_ushort4": "u16vec4",
    "packed_int2": "ivec2",
    "packed_int3": "ivec3",
    "packed_int4": "ivec4",
    "packed_uint2": "uvec2",
    "packed_uint3": "uvec3",
    "packed_uint4": "uvec4",
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
    "xhalf2x2": "f16mat2",
    "xhalf2x3": "f16mat2x3",
    "xhalf2x4": "f16mat2x4",
    "xhalf3x2": "f16mat3x2",
    "xhalf3x3": "f16mat3",
    "xhalf3x4": "f16mat3x4",
    "xhalf4x2": "f16mat4x2",
    "xhalf4x3": "f16mat4x3",
    "xhalf4x4": "f16mat4",
    # Texture Types
    "texture1d": "texture1D",
    "texture1d<float>": "texture1D",
    "texture1d<half>": "texture1D",
    "texture1d<int>": "itexture1D",
    "texture1d<uint>": "utexture1D",
    "texture1d_array": "texture1DArray",
    "texture1d_array<float>": "texture1DArray",
    "texture1d_array<half>": "texture1DArray",
    "texture1d_array<int>": "itexture1DArray",
    "texture1d_array<uint>": "utexture1DArray",
    "texture2d": "texture2D",
    "texture2d<float>": "texture2D",
    "texture2d<half>": "texture2D",
    "texture2d<int>": "itexture2D",
    "texture2d<uint>": "utexture2D",
    "texture2d_ms": "texture2DMS",
    "texture2d_ms<float>": "texture2DMS",
    "texture2d_ms<half>": "texture2DMS",
    "texture2d_ms<int>": "itexture2DMS",
    "texture2d_ms<uint>": "utexture2DMS",
    "texture2d_ms_array": "texture2DMSArray",
    "texture2d_ms_array<float>": "texture2DMSArray",
    "texture2d_ms_array<half>": "texture2DMSArray",
    "texture2d_ms_array<int>": "itexture2DMSArray",
    "texture2d_ms_array<uint>": "utexture2DMSArray",
    "texture3d": "texture3D",
    "texture3d<float>": "texture3D",
    "texture3d<half>": "texture3D",
    "texture3d<int>": "itexture3D",
    "texture3d<uint>": "utexture3D",
    "texturecube": "textureCube",
    "texturecube<float>": "textureCube",
    "texturecube<half>": "textureCube",
    "texturecube<int>": "itextureCube",
    "texturecube<uint>": "utextureCube",
    "TextureCube": "textureCube",
    "texturecube_array": "textureCubeArray",
    "texturecube_array<float>": "textureCubeArray",
    "texturecube_array<half>": "textureCubeArray",
    "texturecube_array<int>": "itextureCubeArray",
    "texturecube_array<uint>": "utextureCubeArray",
    "texture2d_array": "texture2DArray",
    "texture2d_array<float>": "texture2DArray",
    "texture2d_array<half>": "texture2DArray",
    "texture2d_array<int>": "itexture2DArray",
    "texture2d_array<uint>": "utexture2DArray",
    "texture_buffer": "samplerBuffer",
    "texture_buffer<float>": "samplerBuffer",
    "texture_buffer<half>": "samplerBuffer",
    "texture_buffer<int>": "isamplerBuffer",
    "texture_buffer<uint>": "usamplerBuffer",
    "depth2d": "texture2DShadow",
    "depth2d<float>": "texture2DShadow",
    "depth2d_array": "texture2DArrayShadow",
    "depth2d_array<float>": "texture2DArrayShadow",
    "depthcube": "textureCubeShadow",
    "depthcube<float>": "textureCubeShadow",
    "depthcube_array": "textureCubeArrayShadow",
    "depthcube_array<float>": "textureCubeArrayShadow",
    "depth2d_ms": "texture2DMS",
    "depth2d_ms<float>": "texture2DMS",
    "depth2d_ms_array": "texture2DMSArray",
    "depth2d_ms_array<float>": "texture2DMSArray",
    # SwiftUI layer-effect shaders expose the rendered layer as a
    # sampler-like object with sample(coord) syntax.
    "SwiftUI::Layer": "sampler2D",
    # RealityKit custom material shaders use framework-scoped opaque
    # parameter types with no direct CrossGL namespace syntax.
    "realitykit::surface_parameters": "surface_parameters",
    "realitykit::geometry_parameters": "geometry_parameters",
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

# Metal texture methods and the CrossGL functions that express them.
TEXTURE_METHOD_FUNCTIONS = {
    "read": "textureLoad",
    "write": "textureStore",
    "sample_compare": "textureCompare",
    "sample_compare_level": "textureCompareLod",
    "gather": "textureGather",
    "gather_compare": "textureGatherCompare",
}

# Texture methods that read or write storage textures.
TEXTURE_METHOD_STORAGE_OPERATIONS = {
    "read": "read",
    "write": "write",
}

# Texture methods that take a sampler.
SAMPLED_TEXTURE_METHODS = {
    "sample_compare",
    "sample_compare_level",
    "gather",
    "gather_compare",
}

# Texture methods and the operation names recorded for them.
TEXTURE_METHOD_OPERATIONS = {
    "read": "load",
    "write": "store",
    "sample_compare": "sample_compare",
    "sample_compare_level": "sample_compare_lod",
    "gather": "gather",
    "gather_compare": "gather_compare",
}

# ``component::`` names of ``gather`` and the component index each selects.
TEXTURE_GATHER_COMPONENTS = {
    "x": "0",
    "y": "1",
    "z": "2",
    "w": "3",
}

# Texture methods that query a size.
TEXTURE_SIZE_QUERY_METHODS = {
    "get_width",
    "get_height",
    "get_depth",
    "get_array_size",
}

# CrossGL qualifiers of parameters the callee writes.
PARAMETER_DIRECTION_QUALIFIERS = ("inout", "out")

# Metal function attributes that set fragment execution modes.
FRAGMENT_EXECUTION_ATTRIBUTE_NAMES = {
    "early_fragment_tests",
}

# Metal function attributes that carry metadata only.
FUNCTION_METADATA_ATTRIBUTE_NAMES = {
    "host_name",
}

# Metal stage attributes and the CrossGL built-ins they bind.
METAL_ATTRIBUTE_SEMANTICS = {
    "vertex_id": "gl_VertexID",
    "instance_id": "gl_InstanceIndex",
    "base_vertex": "gl_BaseVertex",
    "base_instance": "gl_BaseInstance",
    "position": "gl_Position",
    "point_size": "gl_PointSize",
    "clip_distance": "gl_ClipDistance",
    "front_facing": "gl_FrontFacing",
    "point_coord": "gl_PointCoord",
    "color(0)": "gl_FragColor",
    "color(1)": "gl_FragColor1",
    "color(2)": "gl_FragColor2",
    "color(3)": "gl_FragColor3",
    "color(4)": "gl_FragColor4",
    "depth(any)": "gl_FragDepth",
    "stencil": "gl_FragStencilRefEXT",
    "sample_id": "gl_SampleID",
    "sample_mask": "gl_SampleMask",
    "primitive_id": "gl_PrimitiveID",
    "viewport_array_index": "gl_ViewportIndex",
    "render_target_array_index": "gl_Layer",
    "thread_position_in_grid": "gl_GlobalInvocationID",
    "thread_position_in_threadgroup": "gl_LocalInvocationID",
    "threadgroup_position_in_grid": "gl_WorkGroupID",
    "thread_index_in_threadgroup": "gl_LocalInvocationIndex",
    "quadgroup_index_in_threadgroup": "gl_QuadGroupID",
    "thread_index_in_quadgroup": "gl_QuadGroupInvocationID",
    "thread_index_in_simdgroup": "gl_SubgroupInvocationID",
    "simdgroup_index_in_threadgroup": "gl_SubgroupID",
    "simdgroups_per_threadgroup": "gl_NumSubgroups",
    "threads_per_threadgroup": "gl_WorkGroupSize",
    "threads_per_simdgroup": "gl_SubgroupSize",
    "threadgroups_per_grid": "gl_NumWorkGroups",
    "stage_in": "",
}
