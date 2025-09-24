"""
Centralized Type Mappings for CrossTL.
Eliminates duplication across backends.
"""

# Universal type mappings that all backends should support
UNIVERSAL_TYPE_MAPPINGS = {
    # Basic types
    "void": "void",
    "bool": "bool",
    "int": "int",
    "uint": "uint", 
    "float": "float",
    "double": "double",
    
    # Vector types
    "vec2": "vec2",
    "vec3": "vec3",
    "vec4": "vec4",
    "ivec2": "ivec2",
    "ivec3": "ivec3",
    "ivec4": "ivec4",
    "uvec2": "uvec2",
    "uvec3": "uvec3",
    "uvec4": "uvec4",
    "bvec2": "bvec2",
    "bvec3": "bvec3", 
    "bvec4": "bvec4",
    
    # Matrix types
    "mat2": "mat2",
    "mat3": "mat3",
    "mat4": "mat4",
    "mat2x2": "mat2x2",
    "mat3x3": "mat3x3",
    "mat4x4": "mat4x4",
    
    # Texture types
    "sampler2D": "sampler2D",
    "samplerCube": "samplerCube",
    "sampler3D": "sampler3D",
}

# Backend-specific type mappings
BACKEND_TYPE_MAPPINGS = {
    "cuda": {
        # CUDA-specific mappings
        "vec2": "float2",
        "vec3": "float3", 
        "vec4": "float4",
        "ivec2": "int2",
        "ivec3": "int3",
        "ivec4": "int4",
        "uvec2": "uint2",
        "uvec3": "uint3",
        "uvec4": "uint4",
        "mat2": "float2x2",
        "mat3": "float3x3",
        "mat4": "float4x4",
        "sampler2D": "texture<float4, 2>",
        "samplerCube": "textureCube<float4>",
    },
    
    "metal": {
        # Metal-specific mappings
        "vec2": "float2",
        "vec3": "float3",
        "vec4": "float4", 
        "ivec2": "int2",
        "ivec3": "int3",
        "ivec4": "int4",
        "uvec2": "uint2",
        "uvec3": "uint3",
        "uvec4": "uint4",
        "mat2": "float2x2",
        "mat3": "float3x3",
        "mat4": "float4x4",
        "sampler2D": "texture2d<float>",
        "samplerCube": "texturecube<float>",
    },
    
    "directx": {
        # DirectX/HLSL-specific mappings
        "vec2": "float2",
        "vec3": "float3",
        "vec4": "float4",
        "ivec2": "int2", 
        "ivec3": "int3",
        "ivec4": "int4",
        "uvec2": "uint2",
        "uvec3": "uint3",
        "uvec4": "uint4",
        "mat2": "float2x2",
        "mat3": "float3x3",
        "mat4": "float4x4",
        "sampler2D": "Texture2D",
        "samplerCube": "TextureCube",
    },
    
    "opengl": {
        # OpenGL/GLSL uses CrossGL types mostly as-is
        # No overrides needed for most types
    },
    
    "vulkan": {
        # Vulkan/SPIR-V uses GLSL-like types
        # No overrides needed for most types
    },
    
    "rust": {
        # Rust-specific mappings
        "void": "()",
        "bool": "bool",
        "int": "i32",
        "uint": "u32",
        "float": "f32",
        "double": "f64",
        "vec2": "Vec2<f32>",
        "vec3": "Vec3<f32>",
        "vec4": "Vec4<f32>",
        "ivec2": "Vec2<i32>",
        "ivec3": "Vec3<i32>",
        "ivec4": "Vec4<i32>",
        "uvec2": "Vec2<u32>",
        "uvec3": "Vec3<u32>",
        "uvec4": "Vec4<u32>",
        "mat2": "Mat2<f32>",
        "mat3": "Mat3<f32>", 
        "mat4": "Mat4<f32>",
    },
    
    "mojo": {
        # Mojo-specific mappings
        "void": "None",
        "bool": "Bool",
        "int": "Int32",
        "uint": "UInt32",
        "float": "Float32",
        "double": "Float64",
        "vec2": "SIMD[DType.float32, 2]",
        "vec3": "SIMD[DType.float32, 3]",
        "vec4": "SIMD[DType.float32, 4]",
        "ivec2": "SIMD[DType.int32, 2]",
        "ivec3": "SIMD[DType.int32, 3]",
        "ivec4": "SIMD[DType.int32, 4]",
        "mat2": "Matrix[DType.float32, 2, 2]",
        "mat3": "Matrix[DType.float32, 3, 3]",
        "mat4": "Matrix[DType.float32, 4, 4]",
    },
    
    "hip": {
        # HIP uses CUDA-like types
        "vec2": "float2",
        "vec3": "float3",
        "vec4": "float4",
        "ivec2": "int2",
        "ivec3": "int3", 
        "ivec4": "int4",
        "uvec2": "uint2",
        "uvec3": "uint3",
        "uvec4": "uint4",
        "mat2": "float2x2",
        "mat3": "float3x3",
        "mat4": "float4x4",
    },
    
    "slang": {
        # Slang uses DirectX-like types
        "vec2": "float2",
        "vec3": "float3",
        "vec4": "float4",
        "ivec2": "int2",
        "ivec3": "int3",
        "ivec4": "int4",
        "uvec2": "uint2",
        "uvec3": "uint3",
        "uvec4": "uint4",
        "mat2": "float2x2",
        "mat3": "float3x3",
        "mat4": "float4x4",
        "sampler2D": "Texture2D",
        "samplerCube": "TextureCube",
    }
}


def get_type_mapping(backend: str) -> dict:
    """Get comprehensive type mapping for a backend."""
    mapping = UNIVERSAL_TYPE_MAPPINGS.copy()
    
    # Apply backend-specific overrides
    backend_overrides = BACKEND_TYPE_MAPPINGS.get(backend.lower(), {})
    mapping.update(backend_overrides)
    
    return mapping


def map_type(type_name: str, backend: str) -> str:
    """Map a type to backend-specific representation."""
    mapping = get_type_mapping(backend)
    return mapping.get(type_name, type_name)


# Universal function mappings
UNIVERSAL_FUNCTION_MAPPINGS = {
    # Math functions
    "sqrt": "sqrt",
    "pow": "pow", 
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "abs": "abs",
    "min": "min",
    "max": "max",
    "floor": "floor",
    "ceil": "ceil",
    
    # Vector functions
    "dot": "dot",
    "cross": "cross",
    "length": "length",
    "normalize": "normalize",
    "distance": "distance",
    "reflect": "reflect",
    
    # Atomic operations
    "atomicAdd": "atomicAdd",
    "atomicSub": "atomicSub",
    "atomicMax": "atomicMax",
    "atomicMin": "atomicMin",
    "atomicExchange": "atomicExchange",
    "atomicCompareExchange": "atomicCompareExchange",
}

# Backend-specific function mappings
BACKEND_FUNCTION_MAPPINGS = {
    "cuda": {
        # CUDA uses 'f' suffix for float functions
        "sqrt": "sqrtf",
        "pow": "powf",
        "sin": "sinf",
        "cos": "cosf",
        "tan": "tanf",
        "abs": "fabsf",
        "min": "fminf",
        "max": "fmaxf",
        "floor": "floorf",
        "ceil": "ceilf",
        "atomicExchange": "atomicExch",
        "atomicCompareExchange": "atomicCAS",
    },
    
    "metal": {
        # Metal uses standard names mostly
        "texture": "sample",
    },
    
    "directx": {
        # DirectX specific function mappings
        "fract": "frac",
        "mix": "lerp",
    },
    
    "opengl": {
        # OpenGL specific
        "frac": "fract",
        "lerp": "mix",
    },
    
    "rust": {
        # Rust specific
        "log": "ln",  # Natural log
    },
    
    "mojo": {
        # Mojo specific
        "sqrt": "math.sqrt",
        "pow": "math.pow", 
        "sin": "math.sin",
        "cos": "math.cos",
        "abs": "math.abs",
    }
}


def get_function_mapping(backend: str) -> dict:
    """Get comprehensive function mapping for a backend."""
    mapping = UNIVERSAL_FUNCTION_MAPPINGS.copy()
    
    # Apply backend-specific overrides
    backend_overrides = BACKEND_FUNCTION_MAPPINGS.get(backend.lower(), {})
    mapping.update(backend_overrides)
    
    return mapping


def map_function(function_name: str, backend: str) -> str:
    """Map a function to backend-specific representation."""
    mapping = get_function_mapping(backend)
    return mapping.get(function_name, function_name)
