"""Plain type names that need no declaration in a CrossGL program.

CrossGL's own scalar, vector, matrix and opaque types, and the spellings of the built-in types of the languages the
importers read (GLSL, HLSL, MSL, SPIR-V, CUDA, HIP, Slang and Mojo), which an importer may carry into the IR as a
plain name for the generators to map. :mod:`.declared_types` refuses any other plain type name a program does not
declare. The set is built from families so that every width, component count and dimension is covered alike.
"""

from __future__ import annotations

__all__ = ["BUILTIN_TYPE_NAMES"]

_COUNTS = (2, 3, 4)
_SHAPES = tuple(f"{c}x{r}" for c in _COUNTS for r in _COUNTS)

# Scalars of every language: plain, sized (``int8``, ``uint16_t``) and HLSL minimum-precision spellings, and CrossGL's
# ``Self`` (the implementing type inside a trait or ``impl`` block) and ``str`` (an enum payload's string slice).
_SCALARS = (
    "Self",
    "str",
    "void",
    "bool",
    "int",
    "uint",
    "float",
    "double",
    "half",
    "bfloat",
    "bfloat16",
    "bfloat16_t",
    "char",
    "uchar",
    "short",
    "ushort",
    "long",
    "ulong",
    "size_t",
    "ptrdiff_t",
    "dword",
    "unsigned",
    "signed",
    "auto",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "float16_t",
    "float32_t",
    "float64_t",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f16",
    "f32",
    "f64",
)

# GLSL-style vector and matrix prefixes: ``vec3``, ``i64vec2``, ``f16mat4x3``, ``bfloat16vec4``.
_GLSL_VECTOR_PREFIXES = (
    "vec",
    "ivec",
    "uvec",
    "bvec",
    "dvec",
    "hvec",
    "i8vec",
    "u8vec",
    "i16vec",
    "u16vec",
    "i32vec",
    "u32vec",
    "i64vec",
    "u64vec",
    "f16vec",
    "f32vec",
    "f64vec",
    "bfloat16vec",
)
_GLSL_MATRIX_PREFIXES = ("mat", "dmat", "f16mat", "f32mat", "f64mat")

# HLSL- and MSL-style element names that take a component count or a matrix shape: ``float4``, ``uint3x2``.
_SHAPED_ELEMENTS = (
    "bool",
    "int",
    "uint",
    "float",
    "double",
    "half",
    "bfloat",
    "char",
    "uchar",
    "short",
    "ushort",
    "long",
    "ulong",
    "dword",
    "min16float",
    "min10float",
    "min16int",
    "min12int",
    "min16uint",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "float16_t",
    "float32_t",
    "float64_t",
)
_PACKED_ELEMENTS = (
    "float",
    "half",
    "bfloat",
    "int",
    "uint",
    "short",
    "ushort",
    "char",
    "uchar",
    "long",
    "ulong",
    "bool",
)

# GLSL opaque types. Samplers and textures come in float, ``i`` and ``u`` forms, shadow samplers in float only.
_GLSL_DIMENSIONS = (
    "1D",
    "2D",
    "3D",
    "Cube",
    "2DRect",
    "Buffer",
    "1DArray",
    "2DArray",
    "CubeArray",
    "2DMS",
    "2DMSArray",
)
_GLSL_SHADOW_DIMENSIONS = (
    "1D",
    "2D",
    "Cube",
    "2DRect",
    "1DArray",
    "2DArray",
    "CubeArray",
)
_GLSL_OPAQUE = (
    "sampler",
    "samplerShadow",
    "comparison_sampler",
    "atomic_uint",
    "subpassInput",
    "isubpassInput",
    "usubpassInput",
    "subpassInputMS",
    "isubpassInputMS",
    "usubpassInputMS",
    "accelerationStructureEXT",
    "accelerationStructureNV",
    "rayQueryEXT",
    "hitObjectNV",
    "samplerExternalOES",
)

# HLSL resources, system structures and stage stream types, as plain names.
_HLSL_TYPES = (
    "SamplerState",
    "SamplerComparisonState",
    "sampler_state",
    "Buffer",
    "RWBuffer",
    "ByteAddressBuffer",
    "RWByteAddressBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
    "ConstantBuffer",
    "TextureBuffer",
    "RaytracingAccelerationStructure",
    "RayDesc",
    "RayQuery",
    "BuiltInTriangleIntersectionAttributes",
    "HitObject",
    "InputPatch",
    "OutputPatch",
    "PointStream",
    "LineStream",
    "TriangleStream",
    "FeedbackTexture2D",
    "FeedbackTexture2DArray",
)
_HLSL_TEXTURE_DIMENSIONS = (
    "1D",
    "1DArray",
    "2D",
    "2DArray",
    "2DMS",
    "2DMSArray",
    "3D",
    "Cube",
    "CubeArray",
)

# MSL opaque and library types, as plain names (``texture2d<float>`` carries arguments and is not checked).
_MSL_TEXTURES = (
    "texture1d",
    "texture1d_array",
    "texture2d",
    "texture2d_array",
    "texture2d_ms",
    "texture2d_ms_array",
    "texture3d",
    "texturecube",
    "texturecube_array",
    "texture_buffer",
    "depth2d",
    "depth2d_array",
    "depth2d_ms",
    "depth2d_ms_array",
    "depthcube",
    "depthcube_array",
)
_MSL_TYPES = (
    "sampler",
    "atomic_int",
    "atomic_uint",
    "atomic_bool",
    "atomic_float",
    "atomic_ulong",
    "acceleration_structure",
    "instance_acceleration_structure",
    "primitive_acceleration_structure",
    "intersection_function_table",
    "visible_function_table",
    "intersector",
    "intersection_query",
    "rasterization_rate_map_data",
    "rasterization_rate_map_decoder",
    "render_grid",
    "command_buffer",
    "compute_command",
    "render_command",
)

# Mojo numeric, SIMD and pointer types, and CUDA/HIP built-in types; ``int1``..``double4`` are added below.
_MOJO_TYPES = (
    "Int",
    "UInt",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float16",
    "BFloat16",
    "Float32",
    "Float64",
    "Bool",
    "DType",
    "SIMD",
    "Scalar",
    "UnsafePointer",
    "DTypePointer",
    "LayoutTensor",
    "NDBuffer",
)
_CUDA_TYPES = (
    "dim3",
    "__half",
    "__half2",
    "__nv_bfloat16",
    "__nv_bfloat162",
    "__hip_bfloat16",
    "__hip_bfloat162",
    "cudaTextureObject_t",
    "cudaSurfaceObject_t",
    "hipTextureObject_t",
    "hipSurfaceObject_t",
    "longlong",
    "ulonglong",
)
_CUDA_VECTOR_ELEMENTS = (
    "char",
    "uchar",
    "short",
    "ushort",
    "int",
    "uint",
    "long",
    "ulong",
    "longlong",
    "ulonglong",
    "float",
    "double",
)


def _build():
    names = (
        set(_SCALARS)
        | set(_GLSL_OPAQUE)
        | set(_HLSL_TYPES)
        | set(_MSL_TYPES)
        | set(_MSL_TEXTURES)
    )
    names |= set(_MOJO_TYPES) | set(_CUDA_TYPES)
    names.update(
        f"{element}{count}"
        for element in _CUDA_VECTOR_ELEMENTS
        for count in (1, *_COUNTS)
    )
    for prefix in _GLSL_VECTOR_PREFIXES:
        names.update(f"{prefix}{count}" for count in _COUNTS)
    for prefix in _GLSL_MATRIX_PREFIXES:
        names.update(f"{prefix}{count}" for count in _COUNTS)
        names.update(f"{prefix}{shape}" for shape in _SHAPES)
    for element in _SHAPED_ELEMENTS:
        names.update(f"{element}{count}" for count in _COUNTS)
        names.update(f"{element}{shape}" for shape in _SHAPES)
    for element in _PACKED_ELEMENTS:
        names.update(f"packed_{element}{count}" for count in _COUNTS)
        names.update(f"packed_{element}{shape}" for shape in _SHAPES)
    for kind in ("sampler", "texture", "image"):
        for prefix in ("", "i", "u"):
            names.update(f"{prefix}{kind}{dimension}" for dimension in _GLSL_DIMENSIONS)
    for kind in ("sampler", "texture"):
        names.update(
            f"{kind}{dimension}Shadow" for dimension in _GLSL_SHADOW_DIMENSIONS
        )
    for prefix in ("Texture", "RWTexture", "RasterizerOrderedTexture"):
        names.update(f"{prefix}{dimension}" for dimension in _HLSL_TEXTURE_DIMENSIONS)
    for prefix in (
        "RasterizerOrderedBuffer",
        "RasterizerOrderedByteAddressBuffer",
        "RasterizerOrderedStructuredBuffer",
    ):
        names.add(prefix)
    return frozenset(names)


BUILTIN_TYPE_NAMES = _build()
