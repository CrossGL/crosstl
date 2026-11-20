"""Type, resource, built-in and helper tables of the Mojo generator."""


class MojoScalarConversionError(ValueError):
    """Raised when a source vector reaches a place that needs a scalar."""

    project_diagnostic_code = "project.translate.mojo-scalar-conversion-unsupported"
    missing_capabilities = ("mojo.scalar-conversion-lowering",)

    def __init__(
        self,
        message,
        *,
        source_type=None,
        target_type=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.source_type = source_type
        self.target_type = target_type
        self.reason = reason
        self.source_location = source_location


def _matrix_aliases(prefix, dtype, *, rows_first):
    aliases = {}
    for first in (2, 3, 4):
        for second in (2, 3, 4):
            columns, rows = (second, first) if rows_first else (first, second)
            aliases[f"{prefix}{first}x{second}"] = (dtype, columns, rows)
    return aliases


def _square_matrix_aliases(prefix, dtype):
    return {f"{prefix}{size}": (dtype, size, size) for size in (2, 3, 4)}


MOJO_VECTOR_TYPES = {
    "vec2": ("DType.float32", 2, 2, None),
    "vec3": ("DType.float32", 3, 4, "0.0"),
    "vec4": ("DType.float32", 4, 4, None),
    "float2": ("DType.float32", 2, 2, None),
    "float3": ("DType.float32", 3, 4, "0.0"),
    "float4": ("DType.float32", 4, 4, None),
    "packed_float2": ("DType.float32", 2, 2, None),
    "packed_float3": ("DType.float32", 3, 4, "0.0"),
    "packed_float4": ("DType.float32", 4, 4, None),
    "simd_float2": ("DType.float32", 2, 2, None),
    "simd_float3": ("DType.float32", 3, 4, "0.0"),
    "simd_float4": ("DType.float32", 4, 4, None),
    "vec2<f32>": ("DType.float32", 2, 2, None),
    "vec3<f32>": ("DType.float32", 3, 4, "0.0"),
    "vec4<f32>": ("DType.float32", 4, 4, None),
    "vec2<f16>": ("DType.float16", 2, 2, None),
    "vec3<f16>": ("DType.float16", 3, 4, "0.0"),
    "vec4<f16>": ("DType.float16", 4, 4, None),
    "half2": ("DType.float16", 2, 2, None),
    "half3": ("DType.float16", 3, 4, "0.0"),
    "half4": ("DType.float16", 4, 4, None),
    "packed_half2": ("DType.float16", 2, 2, None),
    "packed_half3": ("DType.float16", 3, 4, "0.0"),
    "packed_half4": ("DType.float16", 4, 4, None),
    "simd_half2": ("DType.float16", 2, 2, None),
    "simd_half3": ("DType.float16", 3, 4, "0.0"),
    "simd_half4": ("DType.float16", 4, 4, None),
    "f16vec2": ("DType.float16", 2, 2, None),
    "f16vec3": ("DType.float16", 3, 4, "0.0"),
    "f16vec4": ("DType.float16", 4, 4, None),
    "min16float2": ("DType.float16", 2, 2, None),
    "min16float3": ("DType.float16", 3, 4, "0.0"),
    "min16float4": ("DType.float16", 4, 4, None),
    "min10float2": ("DType.float16", 2, 2, None),
    "min10float3": ("DType.float16", 3, 4, "0.0"),
    "min10float4": ("DType.float16", 4, 4, None),
    "vec2<f64>": ("DType.float64", 2, 2, None),
    "vec3<f64>": ("DType.float64", 3, 4, "0.0"),
    "vec4<f64>": ("DType.float64", 4, 4, None),
    "double2": ("DType.float64", 2, 2, None),
    "double3": ("DType.float64", 3, 4, "0.0"),
    "double4": ("DType.float64", 4, 4, None),
    "simd_double2": ("DType.float64", 2, 2, None),
    "simd_double3": ("DType.float64", 3, 4, "0.0"),
    "simd_double4": ("DType.float64", 4, 4, None),
    "vec2<i8>": ("DType.int8", 2, 2, None),
    "vec3<i8>": ("DType.int8", 3, 4, "0"),
    "vec4<i8>": ("DType.int8", 4, 4, None),
    "vec2<i32>": ("DType.int32", 2, 2, None),
    "vec3<i32>": ("DType.int32", 3, 4, "0"),
    "vec4<i32>": ("DType.int32", 4, 4, None),
    "vec2<i16>": ("DType.int16", 2, 2, None),
    "vec3<i16>": ("DType.int16", 3, 4, "0"),
    "vec4<i16>": ("DType.int16", 4, 4, None),
    "int2": ("DType.int32", 2, 2, None),
    "int3": ("DType.int32", 3, 4, "0"),
    "int4": ("DType.int32", 4, 4, None),
    "packed_int2": ("DType.int32", 2, 2, None),
    "packed_int3": ("DType.int32", 3, 4, "0"),
    "packed_int4": ("DType.int32", 4, 4, None),
    "simd_int2": ("DType.int32", 2, 2, None),
    "simd_int3": ("DType.int32", 3, 4, "0"),
    "simd_int4": ("DType.int32", 4, 4, None),
    "short2": ("DType.int16", 2, 2, None),
    "short3": ("DType.int16", 3, 4, "0"),
    "short4": ("DType.int16", 4, 4, None),
    "packed_short2": ("DType.int16", 2, 2, None),
    "packed_short3": ("DType.int16", 3, 4, "0"),
    "packed_short4": ("DType.int16", 4, 4, None),
    "simd_short2": ("DType.int16", 2, 2, None),
    "simd_short3": ("DType.int16", 3, 4, "0"),
    "simd_short4": ("DType.int16", 4, 4, None),
    "i8vec2": ("DType.int8", 2, 2, None),
    "i8vec3": ("DType.int8", 3, 4, "0"),
    "i8vec4": ("DType.int8", 4, 4, None),
    "i16vec2": ("DType.int16", 2, 2, None),
    "i16vec3": ("DType.int16", 3, 4, "0"),
    "i16vec4": ("DType.int16", 4, 4, None),
    "min16int2": ("DType.int16", 2, 2, None),
    "min16int3": ("DType.int16", 3, 4, "0"),
    "min16int4": ("DType.int16", 4, 4, None),
    "min12int2": ("DType.int16", 2, 2, None),
    "min12int3": ("DType.int16", 3, 4, "0"),
    "min12int4": ("DType.int16", 4, 4, None),
    "vec2<u32>": ("DType.uint32", 2, 2, None),
    "vec3<u32>": ("DType.uint32", 3, 4, "0"),
    "vec4<u32>": ("DType.uint32", 4, 4, None),
    "vec2<u16>": ("DType.uint16", 2, 2, None),
    "vec3<u16>": ("DType.uint16", 3, 4, "0"),
    "vec4<u16>": ("DType.uint16", 4, 4, None),
    "vec2<u8>": ("DType.uint8", 2, 2, None),
    "vec3<u8>": ("DType.uint8", 3, 4, "0"),
    "vec4<u8>": ("DType.uint8", 4, 4, None),
    "uint2": ("DType.uint32", 2, 2, None),
    "uint3": ("DType.uint32", 3, 4, "0"),
    "uint4": ("DType.uint32", 4, 4, None),
    "packed_uint2": ("DType.uint32", 2, 2, None),
    "packed_uint3": ("DType.uint32", 3, 4, "0"),
    "packed_uint4": ("DType.uint32", 4, 4, None),
    "simd_uint2": ("DType.uint32", 2, 2, None),
    "simd_uint3": ("DType.uint32", 3, 4, "0"),
    "simd_uint4": ("DType.uint32", 4, 4, None),
    "ushort2": ("DType.uint16", 2, 2, None),
    "ushort3": ("DType.uint16", 3, 4, "0"),
    "ushort4": ("DType.uint16", 4, 4, None),
    "packed_ushort2": ("DType.uint16", 2, 2, None),
    "packed_ushort3": ("DType.uint16", 3, 4, "0"),
    "packed_ushort4": ("DType.uint16", 4, 4, None),
    "simd_ushort2": ("DType.uint16", 2, 2, None),
    "simd_ushort3": ("DType.uint16", 3, 4, "0"),
    "simd_ushort4": ("DType.uint16", 4, 4, None),
    "u8vec2": ("DType.uint8", 2, 2, None),
    "u8vec3": ("DType.uint8", 3, 4, "0"),
    "u8vec4": ("DType.uint8", 4, 4, None),
    "u16vec2": ("DType.uint16", 2, 2, None),
    "u16vec3": ("DType.uint16", 3, 4, "0"),
    "u16vec4": ("DType.uint16", 4, 4, None),
    "min16uint2": ("DType.uint16", 2, 2, None),
    "min16uint3": ("DType.uint16", 3, 4, "0"),
    "min16uint4": ("DType.uint16", 4, 4, None),
    "vec2<bool>": ("DType.bool", 2, 2, None),
    "vec3<bool>": ("DType.bool", 3, 4, "False"),
    "vec4<bool>": ("DType.bool", 4, 4, None),
    "ivec2": ("DType.int32", 2, 2, None),
    "ivec3": ("DType.int32", 3, 4, "0"),
    "ivec4": ("DType.int32", 4, 4, None),
    "uvec2": ("DType.uint32", 2, 2, None),
    "uvec3": ("DType.uint32", 3, 4, "0"),
    "uvec4": ("DType.uint32", 4, 4, None),
    "dvec2": ("DType.float64", 2, 2, None),
    "dvec3": ("DType.float64", 3, 4, "0.0"),
    "dvec4": ("DType.float64", 4, 4, None),
    "bvec2": ("DType.bool", 2, 2, None),
    "bvec3": ("DType.bool", 3, 4, "False"),
    "bvec4": ("DType.bool", 4, 4, None),
    "bool2": ("DType.bool", 2, 2, None),
    "bool3": ("DType.bool", 3, 4, "False"),
    "bool4": ("DType.bool", 4, 4, None),
}
MOJO_MATRIX_TYPES = {
    "mat2": ("DType.float32", 2, 2),
    "mat3": ("DType.float32", 3, 3),
    "mat4": ("DType.float32", 4, 4),
    "mat2x2": ("DType.float32", 2, 2),
    "mat2x3": ("DType.float32", 2, 3),
    "mat2x4": ("DType.float32", 2, 4),
    "mat3x2": ("DType.float32", 3, 2),
    "mat3x3": ("DType.float32", 3, 3),
    "mat3x4": ("DType.float32", 3, 4),
    "mat4x2": ("DType.float32", 4, 2),
    "mat4x3": ("DType.float32", 4, 3),
    "mat4x4": ("DType.float32", 4, 4),
    "dmat2": ("DType.float64", 2, 2),
    "dmat3": ("DType.float64", 3, 3),
    "dmat4": ("DType.float64", 4, 4),
    "dmat2x2": ("DType.float64", 2, 2),
    "dmat2x3": ("DType.float64", 2, 3),
    "dmat2x4": ("DType.float64", 2, 4),
    "dmat3x2": ("DType.float64", 3, 2),
    "dmat3x3": ("DType.float64", 3, 3),
    "dmat3x4": ("DType.float64", 3, 4),
    "dmat4x2": ("DType.float64", 4, 2),
    "dmat4x3": ("DType.float64", 4, 3),
    "dmat4x4": ("DType.float64", 4, 4),
    **_matrix_aliases("float", "DType.float32", rows_first=True),
    **_matrix_aliases("simd_float", "DType.float32", rows_first=True),
    **_matrix_aliases("double", "DType.float64", rows_first=True),
    **_matrix_aliases("simd_double", "DType.float64", rows_first=True),
    **_matrix_aliases("half", "DType.float16", rows_first=True),
    **_matrix_aliases("simd_half", "DType.float16", rows_first=True),
    **_matrix_aliases("min16float", "DType.float16", rows_first=True),
    **_matrix_aliases("min10float", "DType.float16", rows_first=True),
    **_matrix_aliases("int", "DType.int32", rows_first=True),
    **_matrix_aliases("simd_int", "DType.int32", rows_first=True),
    **_matrix_aliases("uint", "DType.uint32", rows_first=True),
    **_matrix_aliases("simd_uint", "DType.uint32", rows_first=True),
    **_matrix_aliases("short", "DType.int16", rows_first=True),
    **_matrix_aliases("simd_short", "DType.int16", rows_first=True),
    **_matrix_aliases("min16int", "DType.int16", rows_first=True),
    **_matrix_aliases("min12int", "DType.int16", rows_first=True),
    **_matrix_aliases("ushort", "DType.uint16", rows_first=True),
    **_matrix_aliases("simd_ushort", "DType.uint16", rows_first=True),
    **_matrix_aliases("min16uint", "DType.uint16", rows_first=True),
    **_square_matrix_aliases("f16mat", "DType.float16"),
    **_matrix_aliases("f16mat", "DType.float16", rows_first=False),
}
SWIZZLE_SETS = {
    "xyzw": {"x": 0, "y": 1, "z": 2, "w": 3},
    "rgba": {"r": 0, "g": 1, "b": 2, "a": 3},
}
MOJO_DTYPE_INFO = {
    "DType.float16": ("half", "half", "0.0"),
    "DType.float32": ("float", "vec", "0.0"),
    "DType.float64": ("double", "dvec", "0.0"),
    "DType.int8": ("char", "i8vec", "0"),
    "DType.int16": ("short", "i16vec", "0"),
    "DType.int32": ("int", "ivec", "0"),
    "DType.uint8": ("uchar", "u8vec", "0"),
    "DType.uint16": ("ushort", "u16vec", "0"),
    "DType.uint32": ("uint", "uvec", "0"),
    "DType.bool": ("bool", "bvec", "False"),
}
MOJO_DTYPE_SUFFIX = {
    "DType.float16": "f16",
    "DType.float32": "f32",
    "DType.float64": "f64",
    "DType.int8": "i8",
    "DType.int16": "i16",
    "DType.int32": "i32",
    "DType.uint8": "u8",
    "DType.uint16": "u16",
    "DType.uint32": "u32",
    "DType.bool": "bool",
}
MOJO_SCALAR_DTYPES = {
    "f16": "DType.float16",
    "half": "DType.float16",
    "min10float": "DType.float16",
    "min16float": "DType.float16",
    "float": "DType.float32",
    "double": "DType.float64",
    "i8": "DType.int8",
    "i16": "DType.int16",
    "i32": "DType.int32",
    "int8": "DType.int8",
    "int8_t": "DType.int8",
    "int16": "DType.int16",
    "int16_t": "DType.int16",
    "int": "DType.int32",
    "min12int": "DType.int16",
    "min16int": "DType.int16",
    "short": "DType.int16",
    "u8": "DType.uint8",
    "u16": "DType.uint16",
    "u32": "DType.uint32",
    "uint": "DType.uint32",
    "uint8": "DType.uint8",
    "uint8_t": "DType.uint8",
    "uint16": "DType.uint16",
    "uint16_t": "DType.uint16",
    "min16uint": "DType.uint16",
    "ushort": "DType.uint16",
    "bool": "DType.bool",
}
MOJO_RESOURCE_TYPE_MAPPING = {
    "sampler1D": "Texture1D",
    "sampler1DArray": "Texture1DArray",
    "sampler1DArrayShadow": "Texture1DArrayShadow",
    "sampler1DShadow": "Texture1DShadow",
    "sampler2D": "Texture2D",
    "sampler2DArray": "Texture2DArray",
    "sampler2DArrayShadow": "Texture2DArrayShadow",
    "sampler2DShadow": "Texture2DShadow",
    "sampler2DMS": "Texture2DMS",
    "sampler2DMSArray": "Texture2DMSArray",
    "sampler3D": "Texture3D",
    "samplerCube": "TextureCube",
    "samplerCubeArray": "TextureCubeArray",
    "samplerCubeArrayShadow": "TextureCubeArrayShadow",
    "samplerCubeShadow": "TextureCubeShadow",
    "isampler1D": "Texture1DInt",
    "isampler1DArray": "Texture1DArrayInt",
    "isampler2D": "Texture2DInt",
    "isampler2DArray": "Texture2DArrayInt",
    "isampler2DMS": "Texture2DMSInt",
    "isampler2DMSArray": "Texture2DMSArrayInt",
    "isampler3D": "Texture3DInt",
    "isamplerCube": "TextureCubeInt",
    "isamplerCubeArray": "TextureCubeArrayInt",
    "usampler1D": "Texture1DUInt",
    "usampler1DArray": "Texture1DArrayUInt",
    "usampler2D": "Texture2DUInt",
    "usampler2DArray": "Texture2DArrayUInt",
    "usampler2DMS": "Texture2DMSUInt",
    "usampler2DMSArray": "Texture2DMSArrayUInt",
    "usampler3D": "Texture3DUInt",
    "usamplerCube": "TextureCubeUInt",
    "usamplerCubeArray": "TextureCubeArrayUInt",
    "sampler": "Sampler",
    "comparison_sampler": "Sampler",
    "SamplerState": "Sampler",
    "SamplerComparisonState": "Sampler",
    "image1D": "Image1D",
    "image1DArray": "Image1DArray",
    "image2D": "Image2D",
    "image2DArray": "Image2DArray",
    "image2DMS": "Image2DMS",
    "image2DMSArray": "Image2DMSArray",
    "image3D": "Image3D",
    "imageCube": "ImageCube",
    "iimage1D": "IImage1D",
    "iimage1DArray": "IImage1DArray",
    "iimage2D": "IImage2D",
    "iimage2DArray": "IImage2DArray",
    "iimage2DMS": "IImage2DMS",
    "iimage2DMSArray": "IImage2DMSArray",
    "iimage3D": "IImage3D",
    "uimage1D": "UImage1D",
    "uimage1DArray": "UImage1DArray",
    "uimage2D": "UImage2D",
    "uimage2DArray": "UImage2DArray",
    "uimage2DMS": "UImage2DMS",
    "uimage2DMSArray": "UImage2DMSArray",
    "uimage3D": "UImage3D",
    "accelerationStructureEXT": "RayTracingAccelerationStructure",
    "AccelerationStructure": "RayTracingAccelerationStructure",
    "acceleration_structure": "RayTracingAccelerationStructure",
    "RaytracingAccelerationStructure": "RayTracingAccelerationStructure",
    "RayTracingAccelerationStructure": "RayTracingAccelerationStructure",
}
MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES = frozenset({"Texture2D"})
MOJO_CONCRETE_RESOURCE_TYPES = frozenset(
    {"Sampler", *MOJO_CONCRETE_SAMPLED_TEXTURE_TYPES}
)
MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING = {
    "RWTexture1D": ("Image1D", "IImage1D", "UImage1D"),
    "RWTexture1DArray": ("Image1DArray", "IImage1DArray", "UImage1DArray"),
    "RWTexture2D": ("Image2D", "IImage2D", "UImage2D"),
    "RWTexture2DArray": ("Image2DArray", "IImage2DArray", "UImage2DArray"),
    "RWTexture2DMS": ("Image2DMS", "IImage2DMS", "UImage2DMS"),
    "RWTexture2DMSArray": (
        "Image2DMSArray",
        "IImage2DMSArray",
        "UImage2DMSArray",
    ),
    "RWTexture3D": ("Image3D", "IImage3D", "UImage3D"),
    "RasterizerOrderedTexture1D": ("Image1D", "IImage1D", "UImage1D"),
    "RasterizerOrderedTexture1DArray": (
        "Image1DArray",
        "IImage1DArray",
        "UImage1DArray",
    ),
    "RasterizerOrderedTexture2D": ("Image2D", "IImage2D", "UImage2D"),
    "RasterizerOrderedTexture2DArray": (
        "Image2DArray",
        "IImage2DArray",
        "UImage2DArray",
    ),
    "RasterizerOrderedTexture2DMS": ("Image2DMS", "IImage2DMS", "UImage2DMS"),
    "RasterizerOrderedTexture2DMSArray": (
        "Image2DMSArray",
        "IImage2DMSArray",
        "UImage2DMSArray",
    ),
    "RasterizerOrderedTexture3D": ("Image3D", "IImage3D", "UImage3D"),
}
MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES = set()
MOJO_TYPED_IMAGE_DTYPE_SUFFIX = {
    "DType.float16": "Half",
    "DType.float32": "Float",
    "DType.float64": "Double",
    "DType.int16": "Int16",
    "DType.int32": "Int",
    "DType.uint16": "UInt16",
    "DType.uint32": "UInt",
}
MOJO_TYPED_IMAGE_SUFFIX_INFO = {
    suffix if width == 1 else f"{suffix}{width}": (dtype, width)
    for dtype, suffix in MOJO_TYPED_IMAGE_DTYPE_SUFFIX.items()
    for width in (1, 2, 3, 4)
}
MOJO_TYPED_TEXTURE_DTYPE_SUFFIX = {
    "DType.int32": "Int",
    "DType.uint32": "UInt",
}
MOJO_TYPED_TEXTURE_SUFFIX_INFO = {
    suffix if width == 4 else f"{suffix}{width}": (dtype, width)
    for dtype, suffix in MOJO_TYPED_TEXTURE_DTYPE_SUFFIX.items()
    for width in (1, 2, 3, 4)
}
MOJO_IMAGE_RESOURCE_BASE_TYPES = tuple(
    sorted(
        {
            resource_type
            for resource_type in MOJO_RESOURCE_TYPE_MAPPING.values()
            if resource_type.startswith(("Image", "IImage", "UImage"))
        },
        key=len,
        reverse=True,
    )
)
MOJO_TEXTURE_RESOURCE_BASE_TYPES = tuple(
    sorted(
        {
            resource_type
            for resource_type in MOJO_RESOURCE_TYPE_MAPPING.values()
            if resource_type.startswith("Texture")
            and not any(
                resource_type.endswith(suffix)
                for suffix in MOJO_TYPED_TEXTURE_SUFFIX_INFO
            )
        },
        key=len,
        reverse=True,
    )
)
MOJO_RESOURCE_SAMPLE_COORDS = {
    "Texture1D": "Float32",
    "Texture1DArray": "SIMD[DType.float32, 2]",
    "Texture1DArrayShadow": "SIMD[DType.float32, 4]",
    "Texture1DShadow": "SIMD[DType.float32, 2]",
    "Texture2D": "SIMD[DType.float32, 2]",
    "Texture2DArray": "SIMD[DType.float32, 4]",
    "Texture2DArrayShadow": "SIMD[DType.float32, 4]",
    "Texture2DShadow": "SIMD[DType.float32, 4]",
    "Texture3D": "SIMD[DType.float32, 4]",
    "TextureCube": "SIMD[DType.float32, 4]",
    "TextureCubeArray": "SIMD[DType.float32, 4]",
    "TextureCubeArrayShadow": "SIMD[DType.float32, 4]",
    "TextureCubeShadow": "SIMD[DType.float32, 4]",
}
MOJO_RESOURCE_TEXEL_COORDS = {
    "Texture1D": "Int32",
    "Texture1DArray": "SIMD[DType.int32, 2]",
    "Texture1DArrayShadow": "SIMD[DType.int32, 2]",
    "Texture1DShadow": "Int32",
    "Texture2D": "SIMD[DType.int32, 2]",
    "Texture2DArray": "SIMD[DType.int32, 4]",
    "Texture2DArrayShadow": "SIMD[DType.int32, 4]",
    "Texture2DShadow": "SIMD[DType.int32, 2]",
    "Texture2DMS": "SIMD[DType.int32, 2]",
    "Texture2DMSArray": "SIMD[DType.int32, 4]",
    "Texture3D": "SIMD[DType.int32, 4]",
    "TextureCube": "SIMD[DType.int32, 4]",
    "TextureCubeArray": "SIMD[DType.int32, 4]",
    "TextureCubeArrayShadow": "SIMD[DType.int32, 4]",
    "TextureCubeShadow": "SIMD[DType.int32, 4]",
    "Image1D": "Int32",
    "Image1DArray": "SIMD[DType.int32, 2]",
    "Image2D": "SIMD[DType.int32, 2]",
    "Image2DArray": "SIMD[DType.int32, 4]",
    "Image2DMS": "SIMD[DType.int32, 2]",
    "Image2DMSArray": "SIMD[DType.int32, 4]",
    "Image3D": "SIMD[DType.int32, 4]",
    "ImageCube": "SIMD[DType.int32, 4]",
    "IImage1D": "Int32",
    "IImage1DArray": "SIMD[DType.int32, 2]",
    "IImage2D": "SIMD[DType.int32, 2]",
    "IImage2DArray": "SIMD[DType.int32, 4]",
    "IImage2DMS": "SIMD[DType.int32, 2]",
    "IImage2DMSArray": "SIMD[DType.int32, 4]",
    "IImage3D": "SIMD[DType.int32, 4]",
    "UImage1D": "Int32",
    "UImage1DArray": "SIMD[DType.int32, 2]",
    "UImage2D": "SIMD[DType.int32, 2]",
    "UImage2DArray": "SIMD[DType.int32, 4]",
    "UImage2DMS": "SIMD[DType.int32, 2]",
    "UImage2DMSArray": "SIMD[DType.int32, 4]",
    "UImage3D": "SIMD[DType.int32, 4]",
}
MOJO_RESOURCE_OFFSET_COORDS = {
    "Texture1D": "Int32",
    "Texture1DArray": "Int32",
    "Texture1DArrayShadow": "Int32",
    "Texture1DShadow": "Int32",
    "Texture2D": "SIMD[DType.int32, 2]",
    "Texture2DArray": "SIMD[DType.int32, 2]",
    "Texture2DArrayShadow": "SIMD[DType.int32, 2]",
    "Texture2DShadow": "SIMD[DType.int32, 2]",
    "Texture3D": "SIMD[DType.int32, 4]",
}
MOJO_RESOURCE_COMPARE_COORDS = {
    "Texture1DArrayShadow": "SIMD[DType.float32, 2]",
    "Texture1DShadow": "Float32",
    "Texture2DArrayShadow": "SIMD[DType.float32, 4]",
    "Texture2DShadow": "SIMD[DType.float32, 2]",
    "TextureCubeArrayShadow": "SIMD[DType.float32, 4]",
    "TextureCubeShadow": "SIMD[DType.float32, 4]",
}
MOJO_RESOURCE_PROJECTED_COORDS = {
    "Texture1D": "SIMD[DType.float32, 2]",
    "Texture1DShadow": "SIMD[DType.float32, 2]",
    "Texture2D": "SIMD[DType.float32, 4]",
    "Texture2DShadow": "SIMD[DType.float32, 4]",
    "Texture3D": "SIMD[DType.float32, 4]",
}
MOJO_RESOURCE_SIZE_RETURNS = {
    "Texture1D": "Int32",
    "Texture1DArray": "SIMD[DType.int32, 2]",
    "Texture1DArrayShadow": "SIMD[DType.int32, 2]",
    "Texture1DShadow": "Int32",
    "Texture2D": "SIMD[DType.int32, 2]",
    "Texture2DArray": "SIMD[DType.int32, 4]",
    "Texture2DArrayShadow": "SIMD[DType.int32, 4]",
    "Texture2DShadow": "SIMD[DType.int32, 2]",
    "Texture2DMS": "SIMD[DType.int32, 2]",
    "Texture2DMSArray": "SIMD[DType.int32, 4]",
    "Texture3D": "SIMD[DType.int32, 4]",
    "TextureCube": "SIMD[DType.int32, 2]",
    "TextureCubeArray": "SIMD[DType.int32, 4]",
    "TextureCubeArrayShadow": "SIMD[DType.int32, 4]",
    "TextureCubeShadow": "SIMD[DType.int32, 2]",
    "Image1D": "Int32",
    "Image1DArray": "SIMD[DType.int32, 2]",
    "Image2D": "SIMD[DType.int32, 2]",
    "Image2DArray": "SIMD[DType.int32, 4]",
    "Image2DMS": "SIMD[DType.int32, 2]",
    "Image2DMSArray": "SIMD[DType.int32, 4]",
    "Image3D": "SIMD[DType.int32, 4]",
    "ImageCube": "SIMD[DType.int32, 2]",
    "IImage1D": "Int32",
    "IImage1DArray": "SIMD[DType.int32, 2]",
    "IImage2D": "SIMD[DType.int32, 2]",
    "IImage2DArray": "SIMD[DType.int32, 4]",
    "IImage2DMS": "SIMD[DType.int32, 2]",
    "IImage2DMSArray": "SIMD[DType.int32, 4]",
    "IImage3D": "SIMD[DType.int32, 4]",
    "UImage1D": "Int32",
    "UImage1DArray": "SIMD[DType.int32, 2]",
    "UImage2D": "SIMD[DType.int32, 2]",
    "UImage2DArray": "SIMD[DType.int32, 4]",
    "UImage2DMS": "SIMD[DType.int32, 2]",
    "UImage2DMSArray": "SIMD[DType.int32, 4]",
    "UImage3D": "SIMD[DType.int32, 4]",
}
MOJO_GENERIC_TEXTURE_BUILTINS = {
    "textureOffset": ("sample_offset", "vec4"),
    "textureLodOffset": ("sample_lod_offset", "vec4"),
    "textureGradOffset": ("sample_grad_offset", "vec4"),
    "textureGather": ("texture_gather", "vec4"),
    "textureGatherOffset": ("texture_gather_offset", "vec4"),
    "textureGatherOffsets": ("texture_gather_offsets", "vec4"),
    "textureProj": ("sample_proj", "vec4"),
    "textureProjOffset": ("sample_proj_offset", "vec4"),
    "textureProjLod": ("sample_proj_lod", "vec4"),
    "textureProjLodOffset": ("sample_proj_lod_offset", "vec4"),
    "textureProjGrad": ("sample_proj_grad", "vec4"),
    "textureProjGradOffset": ("sample_proj_grad_offset", "vec4"),
    "textureQueryLod": ("texture_query_lod", "vec2"),
    "textureCompare": ("texture_compare", "float"),
    "textureCompareOffset": ("texture_compare_offset", "float"),
    "textureCompareLod": ("texture_compare_lod", "float"),
    "textureCompareLodOffset": ("texture_compare_lod_offset", "float"),
    "textureCompareGrad": ("texture_compare_grad", "float"),
    "textureCompareGradOffset": ("texture_compare_grad_offset", "float"),
    "textureCompareProj": ("texture_compare_proj", "float"),
    "textureCompareProjOffset": ("texture_compare_proj_offset", "float"),
    "textureCompareProjLod": ("texture_compare_proj_lod", "float"),
    "textureCompareProjLodOffset": ("texture_compare_proj_lod_offset", "float"),
    "textureCompareProjGrad": ("texture_compare_proj_grad", "float"),
    "textureCompareProjGradOffset": ("texture_compare_proj_grad_offset", "float"),
    "textureGatherCompare": ("texture_gather_compare", "vec4"),
    "textureGatherCompareOffset": ("texture_gather_compare_offset", "vec4"),
    "CalculateLevelOfDetail": ("calculate_level_of_detail", "float"),
    "CalculateLevelOfDetailUnclamped": (
        "calculate_level_of_detail_unclamped",
        "float",
    ),
}
MOJO_IMAGE_ATOMIC_BUILTINS = {
    "imageAtomicAdd": "image_atomic_add",
    "imageAtomicMin": "image_atomic_min",
    "imageAtomicMax": "image_atomic_max",
    "imageAtomicAnd": "image_atomic_and",
    "imageAtomicOr": "image_atomic_or",
    "imageAtomicXor": "image_atomic_xor",
    "imageAtomicExchange": "image_atomic_exchange",
    "imageAtomicCompSwap": "image_atomic_comp_swap",
}
MOJO_ATOMIC_OP_ALIASES = {
    **MOJO_IMAGE_ATOMIC_BUILTINS,
    "imageAtomicCompareExchange": "image_atomic_comp_swap",
    "atomicAdd": "image_atomic_add",
    "atomicMin": "image_atomic_min",
    "atomicMax": "image_atomic_max",
    "atomicAnd": "image_atomic_and",
    "atomicOr": "image_atomic_or",
    "atomicXor": "image_atomic_xor",
    "atomicExchange": "image_atomic_exchange",
    "atomicCompSwap": "image_atomic_comp_swap",
    "atomicCompareExchange": "image_atomic_comp_swap",
    "Add": "image_atomic_add",
    "Min": "image_atomic_min",
    "Max": "image_atomic_max",
    "And": "image_atomic_and",
    "Or": "image_atomic_or",
    "Xor": "image_atomic_xor",
    "Exchange": "image_atomic_exchange",
    "CompareExchange": "image_atomic_comp_swap",
    "CompSwap": "image_atomic_comp_swap",
    "InterlockedAdd": "image_atomic_add",
    "InterlockedMin": "image_atomic_min",
    "InterlockedMax": "image_atomic_max",
    "InterlockedAnd": "image_atomic_and",
    "InterlockedOr": "image_atomic_or",
    "InterlockedXor": "image_atomic_xor",
    "InterlockedExchange": "image_atomic_exchange",
    "InterlockedCompareExchange": "image_atomic_comp_swap",
}
MOJO_SCALAR_ATOMIC_OPS = {
    "atomicAdd": "add",
    "atomicMin": "min",
    "atomicMax": "max",
    "atomicAnd": "and",
    "atomicOr": "or",
    "atomicXor": "xor",
    "atomicExchange": "exchange",
    "atomicCompSwap": "compare_exchange",
    "atomicCompareExchange": "compare_exchange",
    "InterlockedAdd": "add",
    "InterlockedMin": "min",
    "InterlockedMax": "max",
    "InterlockedAnd": "and",
    "InterlockedOr": "or",
    "InterlockedXor": "xor",
    "InterlockedExchange": "exchange",
    "InterlockedCompareExchange": "compare_exchange",
}
MOJO_SCALAR_ATOMIC_TYPES = {
    "int",
    "uint",
    "short",
    "ushort",
    "long",
    "ulong",
}
MOJO_TYPED_BUFFER_RESOURCE_TYPES = {
    "Buffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "AppendStructuredBuffer",
    "ConsumeStructuredBuffer",
}
MOJO_BYTE_ADDRESS_BUFFER_TYPES = {
    "ByteAddressBuffer",
    "RWByteAddressBuffer",
}
MOJO_HLSL_BUFFER_TYPE_ALIASES = {
    "RasterizerOrderedBuffer": "RWBuffer",
    "RasterizerOrderedStructuredBuffer": "RWStructuredBuffer",
    "RasterizerOrderedByteAddressBuffer": "RWByteAddressBuffer",
}
MOJO_BUFFER_RESOURCE_TYPES = (
    MOJO_TYPED_BUFFER_RESOURCE_TYPES | MOJO_BYTE_ADDRESS_BUFFER_TYPES
)
MOJO_RESERVED_RESOURCE_TYPE_NAMES = frozenset(
    set(MOJO_RESOURCE_TYPE_MAPPING)
    | set(MOJO_RESOURCE_TYPE_MAPPING.values())
    | set(MOJO_HLSL_WRITABLE_TEXTURE_TYPE_MAPPING)
    | MOJO_HLSL_UNSUPPORTED_WRITABLE_TEXTURE_TYPES
    | set(MOJO_HLSL_BUFFER_TYPE_ALIASES)
    | MOJO_BUFFER_RESOURCE_TYPES
)
MOJO_BUFFER_STORE_RESOURCE_TYPES = {
    "RWBuffer",
    "RWStructuredBuffer",
    "RWByteAddressBuffer",
}
MOJO_BUFFER_LOAD_RESOURCE_TYPES = {
    "Buffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
    "ByteAddressBuffer",
    "RWByteAddressBuffer",
}
MOJO_TYPED_BUFFER_LOAD_RESOURCE_TYPES = {
    "Buffer",
    "RWBuffer",
    "StructuredBuffer",
    "RWStructuredBuffer",
}
MOJO_REINTERPRET_BUILTINS = {
    "asfloat": "Float32",
    "asint": "Int32",
    "asuint": "UInt32",
}
MOJO_REINTERPRET_ALIAS_MAP = {
    "floatBitsToInt": "asint",
    "floatBitsToUint": "asuint",
    "intBitsToFloat": "asfloat",
    "uintBitsToFloat": "asfloat",
}
MOJO_REINTERPRET_TARGET_TYPES = {
    "asfloat": ("float", "vec"),
    "asint": ("int", "ivec"),
    "asuint": ("uint", "uvec"),
}
MOJO_BYTE_ADDRESS_LOAD_METHODS = {
    "Load": 1,
    "Load2": 2,
    "Load3": 3,
    "Load4": 4,
}
MOJO_BYTE_ADDRESS_STORE_METHODS = {
    "Store": 1,
    "Store2": 2,
    "Store3": 3,
    "Store4": 4,
}
MOJO_BYTE_ADDRESS_ATOMIC_METHODS = {
    "InterlockedAdd",
    "InterlockedAnd",
    "InterlockedCompareExchange",
    "InterlockedCompareStore",
    "InterlockedExchange",
    "InterlockedMax",
    "InterlockedMin",
    "InterlockedOr",
    "InterlockedXor",
}
MOJO_BUFFER_COUNTER_METHODS = {
    "DecrementCounter",
    "IncrementCounter",
}
MOJO_BUFFER_OP_ALIASES = {
    "Append": "Append",
    "append": "Append",
    "buffer_append": "Append",
    "Consume": "Consume",
    "consume": "Consume",
    "buffer_consume": "Consume",
    "GetDimensions": "GetDimensions",
    "getDimensions": "GetDimensions",
    "dimensions": "GetDimensions",
    "buffer_dimensions": "GetDimensions",
    "Load": "Load",
    "load": "Load",
    "buffer_load": "Load",
    "Store": "Store",
    "store": "Store",
    "buffer_store": "Store",
    **{name: name for name in MOJO_BYTE_ADDRESS_LOAD_METHODS},
    **{name: name for name in MOJO_BYTE_ADDRESS_STORE_METHODS},
}
MOJO_RESOURCE_QUERY_MEMBER_REQUIREMENTS = {
    "GetDimensions": ("GetDimensions", "texture, image, or buffer resource receiver"),
    "textureSize": ("texture_size", "texture resource receiver"),
    "textureQueryLevels": ("texture_query_levels", "texture resource receiver"),
    "textureSamples": ("texture_samples", "texture resource receiver"),
    "imageSamples": ("image_samples", "image resource receiver"),
}
MOJO_RESOURCE_OPERATION_MEMBER_REQUIREMENTS = {
    "Load": ("Load", "texture, image, or buffer resource receiver"),
    "Store": ("Store", "image or buffer resource receiver"),
    **{
        member: (
            helper_name,
            (
                "image or byte-address buffer resource receiver"
                if member in MOJO_BYTE_ADDRESS_ATOMIC_METHODS
                else "image resource receiver"
            ),
        )
        for member, helper_name in MOJO_ATOMIC_OP_ALIASES.items()
    },
    **{
        member: (member, "byte-address buffer resource receiver")
        for member in MOJO_BYTE_ADDRESS_ATOMIC_METHODS
        if member not in MOJO_ATOMIC_OP_ALIASES
    },
}
MOJO_INTEGER_DTYPES = {
    "DType.int8",
    "DType.int16",
    "DType.int32",
    "DType.uint8",
    "DType.uint16",
    "DType.uint32",
}
# Element types the round_even helper has overloads for, with their scalar types.
ROUND_EVEN_FLOAT_DTYPES = {
    "DType.float16": "Float16",
    "DType.float32": "Float32",
    "DType.float64": "Float64",
}
MOJO_FLOAT_DTYPES = {
    "DType.float32",
    "DType.float64",
}
MOJO_INTEGER_INDEX_TYPES = {
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "i8",
    "i16",
    "i32",
    "int8",
    "int8_t",
    "int",
    "int16",
    "int16_t",
    "long",
    "min12int",
    "min16int",
    "min16uint",
    "short",
    "u8",
    "u16",
    "u32",
    "uint",
    "uint8",
    "uint8_t",
    "uint16",
    "uint16_t",
    "ulong",
    "ushort",
}
MOJO_MAPPED_INTEGER_TYPES = {
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
}
MOJO_RAY_STAGE_TYPES = {
    "ray_any_hit",
    "ray_callable",
    "ray_closest_hit",
    "ray_generation",
    "ray_intersection",
    "ray_miss",
}
MOJO_MESH_STAGE_TYPES = {"task", "amplification", "object", "mesh"}
MOJO_SUPPORTED_STAGE_TYPES = (
    {
        "vertex",
        "fragment",
        "compute",
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
    }
    | MOJO_RAY_STAGE_TYPES
    | MOJO_MESH_STAGE_TYPES
)
MOJO_SYNC_BUILTINS = {
    "barrier": "_crossgl_workgroup_barrier",
    "workgroupBarrier": "_crossgl_workgroup_barrier",
    "memoryBarrier": "_crossgl_memory_barrier",
    "memoryBarrierShared": "_crossgl_memory_barrier_shared",
    "memoryBarrierBuffer": "_crossgl_memory_barrier_buffer",
    "memoryBarrierImage": "_crossgl_memory_barrier_image",
    "groupMemoryBarrier": "_crossgl_group_memory_barrier",
    "allMemoryBarrier": "_crossgl_all_memory_barrier",
    "deviceMemoryBarrier": "_crossgl_device_memory_barrier",
}
MOJO_WAVE_OP_ARITIES = {
    "WaveGetLaneCount": 0,
    "WaveGetLaneIndex": 0,
    "WaveIsFirstLane": 0,
    "WaveActiveSum": 1,
    "WaveActiveProduct": 1,
    "WaveActiveBitAnd": 1,
    "WaveActiveBitOr": 1,
    "WaveActiveBitXor": 1,
    "WaveActiveMin": 1,
    "WaveActiveMax": 1,
    "WaveActiveAllTrue": 1,
    "WaveActiveAnyTrue": 1,
    "WaveActiveAllEqual": 1,
    "WaveActiveBallot": 1,
    "WaveActiveCountBits": 1,
    "WaveReadLaneAt": 2,
    "WaveReadLaneFirst": 1,
    "WavePrefixSum": 1,
    "WavePrefixProduct": 1,
    "WavePrefixCountBits": 1,
    "QuadReadAcrossX": 1,
    "QuadReadAcrossY": 1,
    "QuadReadAcrossDiagonal": 1,
    "QuadReadLaneAt": 2,
    "WaveMatch": 1,
    "WaveMultiPrefixSum": 2,
    "WaveMultiPrefixCountBits": 2,
    "WaveMultiPrefixProduct": 2,
    "WaveMultiPrefixBitAnd": 2,
    "WaveMultiPrefixBitOr": 2,
    "WaveMultiPrefixBitXor": 2,
}
MOJO_WAVE_UINT_RESULT_OPS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}
MOJO_WAVE_BOOL_RESULT_OPS = {
    "WaveIsFirstLane",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
}
MOJO_WAVE_UVEC4_RESULT_OPS = {
    "WaveActiveBallot",
    "WaveMatch",
}
MOJO_RESOURCE_ACCESS_QUALIFIERS = {
    "access::read": "readonly",
    "access::read_write": "readwrite",
    "access::write": "writeonly",
    "read": "readonly",
    "read_write": "readwrite",
    "readonly": "readonly",
    "readwrite": "readwrite",
    "write": "writeonly",
    "writeonly": "writeonly",
}
MOJO_RESOURCE_MEMORY_QUALIFIERS = {
    "coherent",
    "globallycoherent",
    "restrict",
    "volatile",
}
MOJO_NON_SEMANTIC_ATTRIBUTES = {
    "access",
    "binding",
    "buffer",
    "builtin",
    "compute",
    "domain",
    "fragment",
    "glsl_buffer_block",
    "group",
    "layout",
    "maxvertexcount",
    "numthreads",
    "outputcontrolpoints",
    "outputtopology",
    "partitioning",
    "patchconstantfunc",
    "register",
    "sampler",
    "set",
    "space",
    "std140",
    "std430",
    "texture",
    "threadgroup_size",
    "tessellation_control",
    "tessellation_evaluation",
    "uav",
    "vertex",
}
MOJO_INPUT_ONLY_RETURN_SEMANTICS = {
    "gl_fragcoord",
    "gl_frontfacing",
    "gl_globalinvocationid",
    "gl_instanceid",
    "gl_localinvocationid",
    "gl_pointcoord",
    "gl_vertexid",
    "gl_workgroupid",
    "sv_dispatchthreadid",
    "sv_groupid",
    "sv_groupindex",
    "sv_groupthreadid",
    "sv_instanceid",
    "sv_isfrontface",
    "sv_vertexid",
}
MOJO_BUILTIN_PLACEHOLDERS = {
    "gl_GlobalInvocationID": ("global_invocation_id", "uvec3"),
    "gl_LocalInvocationID": ("local_invocation_id", "uvec3"),
    "gl_WorkGroupID": ("workgroup_id", "uvec3"),
    "gl_NumWorkGroups": ("num_workgroups", "uvec3"),
    "gl_WorkGroupSize": ("workgroup_size", "uvec3"),
    "gl_LocalInvocationIndex": ("local_invocation_index", "uint"),
    "SV_DispatchThreadID": ("global_invocation_id", "uvec3"),
    "SV_GroupID": ("workgroup_id", "uvec3"),
    "SV_GroupThreadID": ("local_invocation_id", "uvec3"),
    "SV_GroupIndex": ("local_invocation_index", "uint"),
    "SV_VertexID": ("vertex_id", "uint"),
    "SV_InstanceID": ("instance_id", "uint"),
    "gl_VertexID": ("vertex_id", "int"),
    "gl_InstanceID": ("instance_id", "int"),
    "gl_FrontFacing": ("front_facing", "bool"),
    "SV_IsFrontFace": ("front_facing", "bool"),
    "gl_FragCoord": ("position", "vec4"),
    "gl_PointCoord": ("point_coord", "vec2"),
}
MOJO_GPU_BUILTIN_ALIASES = {
    "gl_GlobalInvocationID": ("global_idx_uint", "uvec3"),
    "SV_DispatchThreadID": ("global_idx_uint", "uvec3"),
    "gl_LocalInvocationID": ("thread_idx_uint", "uvec3"),
    "SV_GroupThreadID": ("thread_idx_uint", "uvec3"),
    "gl_WorkGroupID": ("block_idx_uint", "uvec3"),
    "SV_GroupID": ("block_idx_uint", "uvec3"),
    "gl_NumWorkGroups": ("grid_dim_uint", "uvec3"),
    "gl_WorkGroupSize": ("block_dim_uint", "uvec3"),
}
MOJO_GPU_BUILTIN_PARAMETER_ORDER = (
    "global_idx_uint",
    "thread_idx_uint",
    "block_idx_uint",
    "grid_dim_uint",
    "block_dim_uint",
)
MOJO_VECTOR_COMPARISON_METHODS = {
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "==": "eq",
    "!=": "ne",
}
MOJO_VECTOR_ARITHMETIC_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
}
MOJO_MATH_HELPERS = {
    "clamp",
    "cross_product",
    "dot_product",
    "fmod",
    "lerp",
    "magnitude",
    "max",
    "min",
    "normalize",
    "power",
    "reflect",
    "refract",
    "round_even",
    "smoothstep",
    "step",
}
# The traits a CrossGL value struct conforms to in Mojo 1.2: it is copied implicitly on assignment and argument
# passing, as a CrossGL struct value is.
MOJO_VALUE_STRUCT_TRAITS = "ImplicitlyCopyable, Movable"
