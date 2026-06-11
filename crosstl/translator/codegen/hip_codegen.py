"""
CrossGL to HIP Code Generator

This module provides code generation functionality to convert CrossGL AST to HIP source code.
HIP (Heterogeneous-Compute Interface for Portability) is AMD's CUDA-compatible runtime API
for GPU programming.
"""

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    ASTNode,
    BreakNode,
    CbufferNode,
    ConstantNode,
    ConstructorNode,
    ContinueNode,
    EnumNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    MeshOpNode,
    PointerAccessNode,
    PrimitiveType,
    RangeNode,
    RayQueryOpNode,
    RayTracingOpNode,
    ReturnNode,
    ShaderNode,
    StructNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from .array_utils import parse_array_type, split_array_type_suffix
from .enum_utils import (
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_struct_fields,
    enum_value_expression,
    generate_enum_constants,
    generate_enum_constructor_call,
    generate_enum_constructor_expression,
    generate_enum_constructor_functions,
    generate_enum_structs,
    generate_generic_enum_constants,
    generate_generic_enum_constructor_functions,
    generate_generic_enum_structs,
    generic_enum_specialized_fields,
    generic_enum_specialized_type_name,
    infer_enum_constructor_type,
)
from .generic_function_utils import (
    reject_unsupported_generic_functions as reject_generic_functions_for_target,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generate_generic_structs,
    generate_struct_constructor_expression,
    generic_struct_member_type_name,
    generic_struct_specialized_fields,
    generic_struct_specialized_type_name,
    infer_struct_constructor_type,
)
from .match_utils import (
    generate_match_expression_assignment,
    generate_ordered_conditional_match,
    generate_switch_match,
    infer_match_expression_result_type,
    is_switch_lowerable_match,
)
from .resource_arrays import format_array_declarator
from .resource_diagnostics import ResourceDiagnosticMixin
from .resource_query import ResourceQueryMixin
from .stage_utils import (
    is_fragment_output_parameter,
    normalize_stage_name,
    stage_matches,
)
from .vector_arithmetic import VectorArithmeticMixin

HIP_WAVE_OP_ARITIES = {
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

HIP_WAVE_PREDICATE_ARGUMENT_OPS = {
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

HIP_WAVE_UINT_RESULT_OPS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}

HIP_WAVE_BOOL_RESULT_OPS = {
    "WaveIsFirstLane",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
}

HIP_WAVE_UVEC4_RESULT_OPS = {
    "WaveActiveBallot",
    "WaveMatch",
}

HIP_WARP_SYNC_BUILTIN_ARITIES = {
    "__shfl_down_sync": (3, 4),
}

HIP_BITCAST_FUNCTION_TARGETS = {
    "floatBitsToInt": "int",
    "floatBitsToUint": "uint",
    "intBitsToFloat": "float",
    "uintBitsToFloat": "float",
    "asfloat": "float",
    "asint": "int",
    "asuint": "uint",
}

HIP_INTEGER_BIT_FUNCTION_ALIASES = {
    "countbits": "bitCount",
    "reversebits": "bitfieldReverse",
    "firstbitlow": "findLSB",
    "firstbithigh": "findMSB",
}

HIP_UNSUPPORTED_FP16_VECTOR_TYPES = {
    "vec4<f16>",
    "vec4<float16>",
    "vec4<half>",
    "f16vec4",
    "float16vec4",
    "half4",
}

HIP_FP16_VEC3_TYPES = {
    "vec3<f16>",
    "vec3<float16>",
    "vec3<half>",
    "f16vec3",
    "float16vec3",
    "half3",
}

HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES = {
    "uint",
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "i64",
    "u64",
    "f32",
    "f64",
}


class HipCodeGen(VectorArithmeticMixin, ResourceQueryMixin, ResourceDiagnosticMixin):
    """Emit HIP source from the shared CrossGL translator AST."""

    resource_diagnostic_backend = "HIP"
    struct_constructor_uses_braces = True
    synchronization_builtins = {
        "barrier",
        "groupMemoryBarrier",
        "memoryBarrier",
        "memoryBarrierShared",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "allMemoryBarrier",
        "deviceMemoryBarrier",
        "workgroupBarrier",
    }
    sampled_resource_type_aliases = {
        "Texture1D": "sampler1D",
        "Texture1DArray": "sampler1DArray",
        "Texture2D": "sampler2D",
        "Texture2DArray": "sampler2DArray",
        "Texture2DMS": "sampler2DMS",
        "Texture2DMSArray": "sampler2DMSArray",
        "Texture3D": "sampler3D",
        "TextureCube": "samplerCube",
        "TextureCubeArray": "samplerCubeArray",
    }
    storage_resource_type_aliases = {
        "RWTexture1D": "image1D",
        "RWTexture1DArray": "image1DArray",
        "RWTexture2D": "image2D",
        "RWTexture2DArray": "image2DArray",
        "RWTexture2DMS": "image2DMS",
        "RWTexture2DMSArray": "image2DMSArray",
        "RWTexture3D": "image3D",
        "RWTextureCube": "imageCube",
        "RWTextureCubeArray": "imageCubeArray",
    }
    sampler_state_type_aliases = {
        "SamplerState": "sampler",
        "SamplerComparisonState": "sampler",
    }

    def __init__(self):
        """Initialize HIP type maps and per-generation visitor state."""
        self.indent_level = 0
        self.code_lines = []
        self.current_function = None
        self.current_function_return_type = None
        self.variable_counter = 0
        self.match_temp_variable_index = 0
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = {}
        self.helper_functions = {}
        self.query_resource_names = set()
        self.query_metadata_function_params = {}
        self.query_functions_by_name = {}
        self.structured_buffer_length_names = set()
        self.structured_buffer_length_function_params = {}
        self.current_structured_buffer_length_parameters = {}
        self.current_function_name = None
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False
        self.current_expression_expected_type = None
        self.structs_by_name = {}
        self.generic_enum_struct_definitions = {}
        self.generic_enum_specializations = {}
        self.generic_struct_definitions = {}
        self.generic_struct_specializations = {}
        self.plain_enums = []
        self.struct_payload_enums = []
        self.enum_type_names = set()
        self.enum_struct_type_names = set()
        self.enum_struct_variant_fields = {}
        self.enum_variant_constructors = {}
        self.enum_variant_constructor_fields = {}
        self.enum_variant_constants = {}

        # CrossGL to HIP type mapping
        self.type_map = {
            # Basic types
            "int": "int",
            "float": "float",
            "double": "double",
            "bool": "bool",
            "void": "void",
            "i8": "char",
            "u8": "unsigned char",
            "i16": "short",
            "u16": "unsigned short",
            "i32": "int",
            "u32": "unsigned int",
            "i64": "long long",
            "u64": "unsigned long long",
            "f16": "half",
            "f32": "float",
            "f64": "double",
            "uint": "unsigned int",
            "float16": "half",
            "half": "half",
            "str": "int",
            # Vector types
            "vec2": "float2",
            "vec3": "float3",
            "vec4": "float4",
            "vec2<f16>": "half2",
            "vec2<i8>": "char2",
            "vec3<i8>": "char3",
            "vec4<i8>": "char4",
            "vec2<u8>": "uchar2",
            "vec3<u8>": "uchar3",
            "vec4<u8>": "uchar4",
            "vec2<i16>": "short2",
            "vec3<i16>": "short3",
            "vec4<i16>": "short4",
            "vec2<u16>": "ushort2",
            "vec3<u16>": "ushort3",
            "vec4<u16>": "ushort4",
            "vec2<f32>": "float2",
            "vec3<f32>": "float3",
            "vec4<f32>": "float4",
            "ivec2": "int2",
            "ivec3": "int3",
            "ivec4": "int4",
            "vec2<i32>": "int2",
            "vec3<i32>": "int3",
            "vec4<i32>": "int4",
            "uvec2": "uint2",
            "uvec3": "uint3",
            "uvec4": "uint4",
            "vec2<u32>": "uint2",
            "vec3<u32>": "uint3",
            "vec4<u32>": "uint4",
            "dvec2": "double2",
            "dvec3": "double3",
            "dvec4": "double4",
            "vec2<f64>": "double2",
            "vec3<f64>": "double3",
            "vec4<f64>": "double4",
            "f16vec2": "half2",
            "f16vec3": "cgl_half3",
            "float16vec3": "cgl_half3",
            "half3": "cgl_half3",
            "half2": "half2",
            "vec3<f16>": "cgl_half3",
            "vec3<float16>": "cgl_half3",
            "vec3<half>": "cgl_half3",
            "bvec2": "uchar2",
            "bvec3": "uchar3",
            "bvec4": "uchar4",
            "vec2<bool>": "uchar2",
            "vec3<bool>": "uchar3",
            "vec4<bool>": "uchar4",
            "bool2": "uchar2",
            "bool3": "uchar3",
            "bool4": "uchar4",
            # Matrix types
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Texture/resource types
            "sampler": "hipTextureObject_t",
            "sampler1D": "hipTextureObject_t",
            "sampler1DArray": "hipTextureObject_t",
            "sampler2D": "hipTextureObject_t",
            "sampler3D": "hipTextureObject_t",
            "samplerCube": "hipTextureObject_t",
            "sampler2DArray": "hipTextureObject_t",
            "sampler2DShadow": "hipTextureObject_t",
            "sampler2DArrayShadow": "hipTextureObject_t",
            "samplerCubeShadow": "hipTextureObject_t",
            "samplerCubeArray": "hipTextureObject_t",
            "samplerCubeArrayShadow": "hipTextureObject_t",
            "sampler2DMS": "hipTextureObject_t",
            "sampler2DMSArray": "hipTextureObject_t",
            "isampler1D": "hipTextureObject_t",
            "isampler1DArray": "hipTextureObject_t",
            "isampler2D": "hipTextureObject_t",
            "isampler3D": "hipTextureObject_t",
            "isamplerCube": "hipTextureObject_t",
            "isampler2DArray": "hipTextureObject_t",
            "isamplerCubeArray": "hipTextureObject_t",
            "isampler2DMS": "hipTextureObject_t",
            "isampler2DMSArray": "hipTextureObject_t",
            "usampler1D": "hipTextureObject_t",
            "usampler1DArray": "hipTextureObject_t",
            "usampler2D": "hipTextureObject_t",
            "usampler3D": "hipTextureObject_t",
            "usamplerCube": "hipTextureObject_t",
            "usampler2DArray": "hipTextureObject_t",
            "usamplerCubeArray": "hipTextureObject_t",
            "usampler2DMS": "hipTextureObject_t",
            "usampler2DMSArray": "hipTextureObject_t",
            "image1D": "hipSurfaceObject_t",
            "image1DArray": "hipSurfaceObject_t",
            "image2D": "hipSurfaceObject_t",
            "image3D": "hipSurfaceObject_t",
            "imageCube": "hipSurfaceObject_t",
            "imageCubeArray": "hipSurfaceObject_t",
            "image2DArray": "hipSurfaceObject_t",
            "image2DMS": "hipSurfaceObject_t",
            "image2DMSArray": "hipSurfaceObject_t",
            "iimage1D": "hipSurfaceObject_t",
            "iimage1DArray": "hipSurfaceObject_t",
            "iimage2D": "hipSurfaceObject_t",
            "iimage3D": "hipSurfaceObject_t",
            "iimageCube": "hipSurfaceObject_t",
            "iimageCubeArray": "hipSurfaceObject_t",
            "iimage2DArray": "hipSurfaceObject_t",
            "iimage2DMS": "hipSurfaceObject_t",
            "iimage2DMSArray": "hipSurfaceObject_t",
            "uimage1D": "hipSurfaceObject_t",
            "uimage1DArray": "hipSurfaceObject_t",
            "uimage2D": "hipSurfaceObject_t",
            "uimage3D": "hipSurfaceObject_t",
            "uimageCube": "hipSurfaceObject_t",
            "uimageCubeArray": "hipSurfaceObject_t",
            "uimage2DArray": "hipSurfaceObject_t",
            "uimage2DMS": "hipSurfaceObject_t",
            "uimage2DMSArray": "hipSurfaceObject_t",
            "buffer": "hipDeviceptr_t",
            "accelerationStructureEXT": "CglRayTracingAccelerationStructure",
            "AccelerationStructure": "CglRayTracingAccelerationStructure",
            "acceleration_structure": "CglRayTracingAccelerationStructure",
            "RaytracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayTracingAccelerationStructure": "CglRayTracingAccelerationStructure",
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
        }

        # CrossGL to HIP function mapping
        self.function_map = {
            # Math functions
            "sin": "sinf",
            "cos": "cosf",
            "tan": "tanf",
            "asin": "asinf",
            "acos": "acosf",
            "atan": "atanf",
            "atan2": "atan2f",
            "sinh": "sinhf",
            "cosh": "coshf",
            "tanh": "tanhf",
            "exp": "expf",
            "exp2": "exp2f",
            "log": "logf",
            "log2": "log2f",
            "sqrt": "sqrtf",
            "inversesqrt": "rsqrtf",
            "inverseSqrt": "rsqrtf",
            "rsqrt": "rsqrtf",
            "pow": "powf",
            "fma": "fmaf",
            "mad": "fmaf",
            "abs": "fabsf",
            "floor": "floorf",
            "ceil": "ceilf",
            "round": "roundf",
            "trunc": "truncf",
            "mod": "fmodf",
            "min": "fminf",
            "max": "fmaxf",
            "clamp": "fmaxf(fminf",  # Special handling needed
            "step": "step",
            "smoothstep": "smoothstep",
            # Vector functions
            "length": "length",
            "distance": "distance",
            "dot": "dot",
            "cross": "cross",
            "normalize": "normalize",
            "reflect": "reflect",
            "refract": "refract",
            # Geometric functions
            "faceforward": "faceforward",
            # Vector constructors
            "vec2": "make_float2",
            "vec3": "make_float3",
            "vec4": "make_float4",
            "float2": "make_float2",
            "float3": "make_float3",
            "float4": "make_float4",
            "f16vec3": "cgl_make_half3",
            "float16vec3": "cgl_make_half3",
            "half3": "cgl_make_half3",
            "vec3<f16>": "cgl_make_half3",
            "vec3<float16>": "cgl_make_half3",
            "vec3<half>": "cgl_make_half3",
            "vec2<f32>": "make_float2",
            "vec3<f32>": "make_float3",
            "vec4<f32>": "make_float4",
            "vec2<i8>": "make_char2",
            "vec3<i8>": "make_char3",
            "vec4<i8>": "make_char4",
            "vec2<u8>": "make_uchar2",
            "vec3<u8>": "make_uchar3",
            "vec4<u8>": "make_uchar4",
            "vec2<i16>": "make_short2",
            "vec3<i16>": "make_short3",
            "vec4<i16>": "make_short4",
            "vec2<u16>": "make_ushort2",
            "vec3<u16>": "make_ushort3",
            "vec4<u16>": "make_ushort4",
            "ivec2": "make_int2",
            "ivec3": "make_int3",
            "ivec4": "make_int4",
            "int2": "make_int2",
            "int3": "make_int3",
            "int4": "make_int4",
            "vec2<i32>": "make_int2",
            "vec3<i32>": "make_int3",
            "vec4<i32>": "make_int4",
            "uvec2": "make_uint2",
            "uvec3": "make_uint3",
            "uvec4": "make_uint4",
            "uint2": "make_uint2",
            "uint3": "make_uint3",
            "uint4": "make_uint4",
            "vec2<u32>": "make_uint2",
            "vec3<u32>": "make_uint3",
            "vec4<u32>": "make_uint4",
            "dvec2": "make_double2",
            "dvec3": "make_double3",
            "dvec4": "make_double4",
            "double2": "make_double2",
            "double3": "make_double3",
            "double4": "make_double4",
            "vec2<f64>": "make_double2",
            "vec3<f64>": "make_double3",
            "vec4<f64>": "make_double4",
            "bvec2": "make_uchar2",
            "bvec3": "make_uchar3",
            "bvec4": "make_uchar4",
            "uchar2": "make_uchar2",
            "uchar3": "make_uchar3",
            "uchar4": "make_uchar4",
            "vec2<bool>": "make_uchar2",
            "vec3<bool>": "make_uchar3",
            "vec4<bool>": "make_uchar4",
            "bool2": "make_uchar2",
            "bool3": "make_uchar3",
            "bool4": "make_uchar4",
            # Matrix constructors
            "mat2": "float2x2",
            "mat3": "float3x3",
            "mat4": "float4x4",
            "mat2x2": "float2x2",
            "mat2x3": "float2x3",
            "mat2x4": "float2x4",
            "mat3x2": "float3x2",
            "mat3x3": "float3x3",
            "mat3x4": "float3x4",
            "mat4x2": "float4x2",
            "mat4x3": "float4x3",
            "mat4x4": "float4x4",
            "dmat2": "double2x2",
            "dmat3": "double3x3",
            "dmat4": "double4x4",
            "dmat2x2": "double2x2",
            "dmat2x3": "double2x3",
            "dmat2x4": "double2x4",
            "dmat3x2": "double3x2",
            "dmat3x3": "double3x3",
            "dmat3x4": "double3x4",
            "dmat4x2": "double4x2",
            "dmat4x3": "double4x3",
            "dmat4x4": "double4x4",
            # Atomic operations
            "atomicAdd": "atomicAdd",
            "atomicSub": "atomicSub",
            "atomicMin": "atomicMin",
            "atomicMax": "atomicMax",
            "atomicAnd": "atomicAnd",
            "atomicOr": "atomicOr",
            "atomicXor": "atomicXor",
            "atomicExchange": "atomicExch",
            "atomicCompareExchange": "atomicCAS",
            "atomicCompSwap": "atomicCAS",
            # Synchronization
            "barrier": "__syncthreads",
            "groupMemoryBarrier": "__threadfence_block",
            "memoryBarrier": "__threadfence",
            "memoryBarrierShared": "__threadfence_block",
            "memoryBarrierBuffer": "__threadfence",
            "memoryBarrierImage": "__threadfence",
            "allMemoryBarrier": "__threadfence",
            "deviceMemoryBarrier": "__threadfence",
            "workgroupBarrier": "__syncthreads",
            # Ray tracing placeholders
            "RayDesc": "CglRayDesc",
            "RayQuery": "CglRayQuery",
            "BuiltInTriangleIntersectionAttributes": (
                "CglBuiltInTriangleIntersectionAttributes"
            ),
            "TraceRay": "cgl_trace_ray",
            "CallShader": "cgl_call_shader",
            "ReportHit": "cgl_report_hit",
            "IgnoreHit": "cgl_ignore_hit",
            "AcceptHitAndEndSearch": "cgl_accept_hit_and_end_search",
            # Texture functions
            "texture": "tex2D",
            "textureLod": "tex2DLod",
            "textureGrad": "tex2DGrad",
        }

        # Built-in variable mappings
        self.builtin_map = {
            "gl_LocalInvocationID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "gl_LocalInvocationID.x": "threadIdx.x",
            "gl_LocalInvocationID.y": "threadIdx.y",
            "gl_LocalInvocationID.z": "threadIdx.z",
            "gl_WorkGroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "gl_WorkGroupID.x": "blockIdx.x",
            "gl_WorkGroupID.y": "blockIdx.y",
            "gl_WorkGroupID.z": "blockIdx.z",
            "gl_WorkGroupSize": "make_uint3(blockDim.x, blockDim.y, blockDim.z)",
            "gl_WorkGroupSize.x": "blockDim.x",
            "gl_WorkGroupSize.y": "blockDim.y",
            "gl_WorkGroupSize.z": "blockDim.z",
            "gl_NumWorkGroups": "make_uint3(gridDim.x, gridDim.y, gridDim.z)",
            "gl_NumWorkGroups.x": "gridDim.x",
            "gl_NumWorkGroups.y": "gridDim.y",
            "gl_NumWorkGroups.z": "gridDim.z",
            "gl_GlobalInvocationID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "gl_GlobalInvocationID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "gl_GlobalInvocationID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "gl_GlobalInvocationID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "gl_LocalInvocationIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GroupThreadID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GroupThreadID.x": "threadIdx.x",
            "SV_GroupThreadID.y": "threadIdx.y",
            "SV_GroupThreadID.z": "threadIdx.z",
            "SV_GROUPTHREADID": "make_uint3(threadIdx.x, threadIdx.y, threadIdx.z)",
            "SV_GROUPTHREADID.x": "threadIdx.x",
            "SV_GROUPTHREADID.y": "threadIdx.y",
            "SV_GROUPTHREADID.z": "threadIdx.z",
            "SV_GroupID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GroupID.x": "blockIdx.x",
            "SV_GroupID.y": "blockIdx.y",
            "SV_GroupID.z": "blockIdx.z",
            "SV_GROUPID": "make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)",
            "SV_GROUPID.x": "blockIdx.x",
            "SV_GROUPID.y": "blockIdx.y",
            "SV_GROUPID.z": "blockIdx.z",
            "SV_DispatchThreadID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DispatchThreadID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DispatchThreadID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DispatchThreadID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_DISPATCHTHREADID": (
                "make_uint3((blockIdx.x * blockDim.x + threadIdx.x), "
                "(blockIdx.y * blockDim.y + threadIdx.y), "
                "(blockIdx.z * blockDim.z + threadIdx.z))"
            ),
            "SV_DISPATCHTHREADID.x": "(blockIdx.x * blockDim.x + threadIdx.x)",
            "SV_DISPATCHTHREADID.y": "(blockIdx.y * blockDim.y + threadIdx.y)",
            "SV_DISPATCHTHREADID.z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            "SV_GroupIndex": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
            "SV_GROUPINDEX": (
                "(threadIdx.z * blockDim.y * blockDim.x + "
                "threadIdx.y * blockDim.x + threadIdx.x)"
            ),
        }

    def generate(self, node: ASTNode) -> str:
        """Generate complete HIP source for a CrossGL AST."""
        self.code_lines = []
        self.indent_level = 0
        self.validate_supported_stage_types(node)
        self.reject_unsupported_generic_functions(node)
        self.variable_types = {}
        self.image_resource_accesses = {}
        self.buffer_resource_accesses = {}
        self.glsl_buffer_block_accesses = {}
        self.glsl_buffer_block_layouts = {}
        self.hip_resource_binding_cursors = {}
        self.hip_used_resource_bindings = {}
        self.current_function_return_type = None
        self.match_temp_variable_index = 0
        self.struct_member_types = {}
        self.struct_member_semantics = {}
        self.function_return_types = self.collect_function_return_types(node)
        self.helper_functions = {}
        self.resource_query_info_required = False
        self.assignment_lhs_depth = 0
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False
        self.current_stage_name = None
        self.hip_function_capture_params = {}
        self.current_expression_expected_type = None
        self.setup_enum_and_generic_metadata(node)
        (
            self.query_resource_names,
            self.query_metadata_function_params,
        ) = self.collect_resource_query_requirements(node)
        (
            self.structured_buffer_length_names,
            self.structured_buffer_length_function_params,
        ) = self.collect_structured_buffer_length_requirements(node)
        self.query_functions_by_name = {
            getattr(func, "name", None): func
            for func in self.query_collect_functions(node)
        }
        self.query_functions_by_name = {
            name: func for name, func in self.query_functions_by_name.items() if name
        }
        self.reserve_explicit_hip_resource_bindings(node)

        self.add_includes()
        self.add_generated_code(self.generate_hip_matrix_type_helpers())
        self.add_line()
        if self.has_geometry_stage(node):
            self.add_generated_code(self.generate_hip_geometry_stream_helpers())
            self.add_line()
        if self.has_tessellation_stage(node):
            self.add_generated_code(self.generate_hip_tessellation_patch_helpers())
            self.add_line()
        if self.has_mesh_task_stage(node):
            self.add_generated_code(self.generate_hip_mesh_task_helpers())
            self.add_line()
        self.visit(node)
        self.insert_helper_functions()

        return "\n".join(self.code_lines)

    def fused_multiply_add_result_type(self, raw_args):
        if len(raw_args) != 3:
            return None

        scalar_result_type = None
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return arg_type
            component_type = self.scalar_component_type(arg_type)
            if component_type == "double":
                scalar_result_type = "double"
            elif component_type == "float" and scalar_result_type is None:
                scalar_result_type = "float"
        return scalar_result_type

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions before emitting non-compilable HIP code."""
        reject_generic_functions_for_target(ast_node, "HIP")

    def setup_enum_and_generic_metadata(self, ast_node):
        """Collect enum and concrete generic type metadata used by HIP lowering."""
        structs = list(getattr(ast_node, "structs", []) or [])
        self.structs_by_name = {
            node.name: node
            for node in structs
            if isinstance(node, StructNode) and getattr(node, "name", None)
        }
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast_node,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )
        self.generic_struct_specializations = collect_generic_struct_specializations(
            ast_node,
            self.generic_struct_definitions,
            self.type_name_string,
        )
        self.plain_enums = collect_plain_enums(structs)
        self.struct_payload_enums = collect_struct_payload_enums(structs)
        self.enum_type_names = collect_enum_type_names(self.plain_enums)
        self.enum_struct_type_names = (
            collect_enum_type_names(self.struct_payload_enums)
            | set(self.generic_enum_struct_definitions)
            | {
                specialization["struct_name"]
                for specialization in self.generic_enum_specializations.values()
            }
        )
        self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constructors = collect_enum_variant_constructors(
            self.struct_payload_enums
        )
        self.enum_variant_constructor_fields = collect_enum_variant_constructor_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constants = {
            **collect_enum_variant_constants(
                self.plain_enums + self.struct_payload_enums
            ),
            **collect_generic_enum_variant_constants(
                self.generic_enum_struct_definitions
            ),
        }
        self.struct_member_types.update(
            self.generic_struct_definition_member_types(self.generic_struct_definitions)
        )
        self.struct_member_types.update(
            self.enum_payload_struct_member_types(self.struct_payload_enums)
        )
        self.struct_member_types.update(
            collect_generic_enum_specialization_member_types(
                self,
                self.generic_enum_specializations,
            )
        )
        self.struct_member_types.update(
            collect_generic_struct_specialization_member_types(
                self,
                self.generic_struct_specializations,
            )
        )

    def enum_payload_struct_member_types(self, enums):
        member_types = {}
        for enum in enums or []:
            fields = {"variant": "int"}
            for field_name, field_type in enum_struct_fields(enum) or []:
                fields[field_name] = self.type_name_string(field_type)
            member_types[enum.name] = fields
        return member_types

    def generic_struct_definition_member_types(self, definitions):
        member_types = {}
        for name, definition in (definitions or {}).items():
            member_types[name] = {
                member.name: generic_struct_member_type_name(
                    member,
                    self.type_name_string,
                )
                for member in definition["members"]
            }
        return member_types

    def unsupported_stage_types(self):
        return set()

    def hip_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def hip_ray_stage_metadata_name(self, stage_name):
        return {
            "ray_any_hit": "any_hit",
            "ray_callable": "callable",
            "ray_closest_hit": "closest_hit",
            "ray_generation": "ray_generation",
            "ray_intersection": "intersection",
            "ray_miss": "miss",
        }.get(stage_name, stage_name)

    def has_geometry_stage(self, ast_node, target_stage=None):
        if not stage_matches(target_stage, "geometry"):
            return False

        return self.has_stage(ast_node, "geometry")

    def has_tessellation_stage(self, ast_node, target_stage=None):
        if not (
            stage_matches(target_stage, "tessellation_control")
            or stage_matches(target_stage, "tessellation_evaluation")
        ):
            return False

        return self.has_stage(ast_node, "tessellation_control") or self.has_stage(
            ast_node, "tessellation_evaluation"
        )

    def hip_mesh_task_stage_names(self):
        return {"mesh", "task", "object", "amplification"}

    def has_mesh_task_stage(self, ast_node, target_stage=None):
        stage_names = self.hip_mesh_task_stage_names()
        if target_stage is not None and not any(
            stage_matches(target_stage, stage_name) for stage_name in stage_names
        ):
            return False

        return any(self.has_stage(ast_node, stage_name) for stage_name in stage_names)

    def has_stage(self, ast_node, stage_name):
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == stage_name:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == stage_name
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_hip_geometry_stream_helpers(self):
        return (
            "template <typename T>\n"
            "struct CglHipPointStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipLineStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipTriangleStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n"
        )

    def generate_hip_tessellation_patch_helpers(self):
        return (
            "template <typename T, int N>\n"
            "struct CglHipInputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n\n"
            "template <typename T, int N>\n"
            "struct CglHipOutputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n"
        )

    def generate_hip_mesh_task_helpers(self):
        return (
            "__device__ inline void cgl_hip_set_mesh_output_counts(\n"
            "    unsigned int vertex_count, unsigned int primitive_count)\n"
            "{\n"
            "    (void)vertex_count;\n"
            "    (void)primitive_count;\n"
            "}\n\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "}\n\n"
            "template <typename Payload>\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z,\n"
            "    const Payload& payload)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "    (void)payload;\n"
            "}\n"
        )

    def generate_hip_matrix_type_helpers(self):
        components = ("x", "y", "z", "w")

        def matrix_type(scalar, columns, rows):
            return f"{scalar}{columns}x{rows}"

        def vector_type(scalar, size):
            return f"{scalar}{size}"

        def vector_constructor(scalar, size, args):
            return f"make_{scalar}{size}({', '.join(args)})"

        lines = ["// CrossGL matrix value helpers"]
        for scalar in ("float", "double"):
            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    count = columns * rows
                    params = ", ".join(f"{scalar} v{i}" for i in range(count))
                    assignments = " ".join(f"m[{i}] = v{i};" for i in range(count))
                    zero = f"{scalar}(0)"
                    add_args = ", ".join(f"m[{i}] + rhs.m[{i}]" for i in range(count))
                    sub_args = ", ".join(f"m[{i}] - rhs.m[{i}]" for i in range(count))
                    neg_args = ", ".join(f"-m[{i}]" for i in range(count))
                    mul_args = ", ".join(f"m[{i}] * rhs" for i in range(count))
                    div_args = ", ".join(f"m[{i}] / rhs" for i in range(count))
                    column_params = ", ".join(
                        f"{vector_type(scalar, rows)} c{column}"
                        for column in range(columns)
                    )
                    column_assignments = " ".join(
                        (f"m[{column * rows + row}] = " f"c{column}.{components[row]};")
                        for column in range(columns)
                        for row in range(rows)
                    )
                    one = f"{scalar}(1)"
                    lines.extend(
                        [
                            f"struct {type_name} {{",
                            f"    {scalar} m[{count}];",
                            f"    static const int CGL_COLUMNS = {columns};",
                            f"    static const int CGL_ROWS = {rows};",
                            f"    __host__ __device__ {type_name}() {{ }}",
                            f"    __host__ __device__ explicit {type_name}({scalar} diagonal) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                        ]
                    )
                    for index in range(min(columns, rows)):
                        lines.append(f"        m[{index * rows + index}] = diagonal;")
                    lines.extend(
                        [
                            "    }",
                            f"    __host__ __device__ {type_name}({column_params}) {{ {column_assignments} }}",
                            (
                                "    template <typename Matrix, "
                                "typename = decltype(Matrix::CGL_COLUMNS), "
                                "typename = decltype(Matrix::CGL_ROWS)>"
                            ),
                            f"    __host__ __device__ explicit {type_name}(const Matrix& source) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                            f"        for (int column = 0; column < {columns}; ++column) {{",
                            f"            for (int row = 0; row < {rows}; ++row) {{",
                            "                if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {",
                            f"                    m[column * {rows} + row] = source.m[column * Matrix::CGL_ROWS + row];",
                            "                } else if (column == row) {",
                            f"                    m[column * {rows} + row] = {one};",
                            "                }",
                            "            }",
                            "        }",
                            "    }",
                            f"    __host__ __device__ {type_name}({params}) {{ {assignments} }}",
                            f"    __host__ __device__ {scalar}& operator[](int index) {{ return m[index]; }}",
                            f"    __host__ __device__ const {scalar}& operator[](int index) const {{ return m[index]; }}",
                            f"    __host__ __device__ {type_name} operator+(const {type_name}& rhs) const {{",
                            f"        return {type_name}({add_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-(const {type_name}& rhs) const {{",
                            f"        return {type_name}({sub_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-() const {{",
                            f"        return {type_name}({neg_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator*({scalar} rhs) const {{",
                            f"        return {type_name}({mul_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator/({scalar} rhs) const {{",
                            f"        return {type_name}({div_args});",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator+=(const {type_name}& rhs) {{",
                            "        *this = *this + rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator-=(const {type_name}& rhs) {{",
                            "        *this = *this - rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator*=({scalar} rhs) {{",
                            "        *this = *this * rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator/=({scalar} rhs) {{",
                            "        *this = *this / rhs;",
                            "        return *this;",
                            "    }",
                            "};",
                            "",
                            f"__host__ __device__ inline {type_name} operator*({scalar} lhs, const {type_name}& rhs) {{",
                            "    return rhs * lhs;",
                            "}",
                            "",
                        ]
                    )

            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    result_type = matrix_type(scalar, rows, columns)
                    transpose_values = [
                        f"value.m[{row * rows + column}]"
                        for column in range(rows)
                        for row in range(columns)
                    ]
                    lines.extend(
                        [
                            f"__host__ __device__ inline {result_type} transpose(const {type_name}& value) {{",
                            f"    return {result_type}({', '.join(transpose_values)});",
                            "}",
                            "",
                        ]
                    )

            for size in range(2, 5):
                type_name = matrix_type(scalar, size, size)
                zero = f"{scalar}(0)"
                one = f"{scalar}(1)"
                lines.extend(
                    [
                        f"__host__ __device__ inline {type_name} inverse(const {type_name}& value) {{",
                        f"    {type_name} a(value);",
                        f"    {type_name} result({one});",
                        f"    for (int column = 0; column < {size}; ++column) {{",
                        "        int pivot_row = column;",
                        f"        {scalar} pivot_value = a.m[column * {size} + column];",
                        (
                            f"        {scalar} pivot_abs = pivot_value < {zero} ? "
                            "-pivot_value : pivot_value;"
                        ),
                        f"        for (int row = column + 1; row < {size}; ++row) {{",
                        f"            {scalar} candidate_value = a.m[column * {size} + row];",
                        (
                            f"            {scalar} candidate_abs = "
                            f"candidate_value < {zero} ? "
                            "-candidate_value : candidate_value;"
                        ),
                        "            if (candidate_abs > pivot_abs) {",
                        "                pivot_abs = candidate_abs;",
                        "                pivot_row = row;",
                        "            }",
                        "        }",
                        f"        if (pivot_abs == {zero}) {{",
                        "            return result;",
                        "        }",
                        "        if (pivot_row != column) {",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                {scalar} tmp = a.m[c * {size} + column];",
                        f"                a.m[c * {size} + column] = a.m[c * {size} + pivot_row];",
                        f"                a.m[c * {size} + pivot_row] = tmp;",
                        f"                tmp = result.m[c * {size} + column];",
                        f"                result.m[c * {size} + column] = result.m[c * {size} + pivot_row];",
                        f"                result.m[c * {size} + pivot_row] = tmp;",
                        "            }",
                        "        }",
                        f"        {scalar} pivot = a.m[column * {size} + column];",
                        f"        for (int c = 0; c < {size}; ++c) {{",
                        f"            a.m[c * {size} + column] /= pivot;",
                        f"            result.m[c * {size} + column] /= pivot;",
                        "        }",
                        f"        for (int row = 0; row < {size}; ++row) {{",
                        "            if (row == column) {",
                        "                continue;",
                        "            }",
                        f"            {scalar} factor = a.m[column * {size} + row];",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                a.m[c * {size} + row] -= factor * a.m[c * {size} + column];",
                        f"                result.m[c * {size} + row] -= factor * result.m[c * {size} + column];",
                        "            }",
                        "        }",
                        "    }",
                        "    return result;",
                        "}",
                        "",
                    ]
                )

            for left_columns in range(2, 5):
                for left_rows in range(2, 5):
                    left_type = matrix_type(scalar, left_columns, left_rows)
                    in_vector = vector_type(scalar, left_columns)
                    out_vector = vector_type(scalar, left_rows)
                    values = []
                    for row in range(left_rows):
                        terms = [
                            f"lhs.m[{column * left_rows + row}] * rhs.{components[column]}"
                            for column in range(left_columns)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {left_type}& lhs, const {in_vector}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_rows, values)};",
                            "}",
                            "",
                        ]
                    )

                    in_vector = vector_type(scalar, left_rows)
                    out_vector = vector_type(scalar, left_columns)
                    values = []
                    for column in range(left_columns):
                        terms = [
                            f"lhs.{components[row]} * rhs.m[{column * left_rows + row}]"
                            for row in range(left_rows)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {in_vector}& lhs, const {left_type}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_columns, values)};",
                            "}",
                            "",
                        ]
                    )

                    for right_columns in range(2, 5):
                        right_type = matrix_type(scalar, right_columns, left_columns)
                        result_type = matrix_type(scalar, right_columns, left_rows)
                        values = []
                        for column in range(right_columns):
                            for row in range(left_rows):
                                terms = [
                                    (
                                        f"lhs.m[{shared * left_rows + row}] * "
                                        f"rhs.m[{column * left_columns + shared}]"
                                    )
                                    for shared in range(left_columns)
                                ]
                                values.append(" + ".join(terms))
                        lines.extend(
                            [
                                f"__host__ __device__ inline {result_type} operator*(const {left_type}& lhs, const {right_type}& rhs) {{",
                                f"    return {result_type}({', '.join(values)});",
                                "}",
                                "",
                            ]
                        )
        return "\n".join(lines)

    def validate_supported_stage_types(self, ast_node, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast_node, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise ValueError(f"HIP output does not support stage type(s): {stage_list}")

    def add_includes(self):
        """Emit the standard HIP runtime include block."""
        self.code_lines.extend(
            [
                "#include <hip/hip_runtime.h>",
                "#include <hip/hip_runtime_api.h>",
                "#include <hip/math_functions.h>",
                "#include <hip/device_functions.h>",
                "#include <hip/hip_fp16.h>",
                "",
            ]
        )

    def indent(self) -> str:
        """Return whitespace for the current indentation level."""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Append one HIP output line using the current indentation level."""
        if line:
            self.code_lines.append(self.indent() + line)
        else:
            self.code_lines.append("")

    def add_generated_code(self, code: str):
        """Append pre-indented generated code to the output stream."""
        for line in code.rstrip("\n").splitlines():
            self.code_lines.append(line.rstrip())

    def visit(self, node: ASTNode) -> str:
        """Dispatch an AST node to its HIP visitor method."""
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> str:
        """Raise a clear error for unsupported AST nodes."""
        raise NotImplementedError(
            f"Code generation not implemented for {type(node).__name__}"
        )

    def ordered_generic_struct_specializations(self):
        specializations = self.generic_struct_specializations or {}
        if not specializations:
            return {}

        type_text_by_struct_name = {
            specialization["struct_name"]: type_text
            for type_text, specialization in specializations.items()
        }
        dependencies = {type_text: set() for type_text in specializations}
        for type_text, specialization in specializations.items():
            for _field_name, field_type in generic_struct_specialized_fields(
                self.type_name_string,
                specialization,
            ):
                dependency_name = generic_struct_specialized_type_name(
                    self,
                    field_type,
                )
                dependency_type = type_text_by_struct_name.get(dependency_name)
                if dependency_type and dependency_type != type_text:
                    dependencies[type_text].add(dependency_type)

        ordered = {}
        visiting = set()
        visited = set()
        type_order = {
            type_text: index for index, type_text in enumerate(specializations)
        }

        def visit(type_text):
            if type_text in visited or type_text in visiting:
                return
            visiting.add(type_text)
            for dependency_type in sorted(
                dependencies[type_text],
                key=lambda item: type_order[item],
            ):
                visit(dependency_type)
            visiting.remove(type_text)
            visited.add(type_text)
            ordered[type_text] = specializations[type_text]

        for type_text in specializations:
            visit(type_text)
        return ordered

    def mapped_type_dependency_name(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, _array_suffix = split_array_type_suffix(str(mapped_type or ""))
        base_type = base_type.strip()
        if base_type in {"int", "unsigned int", "float", "double", "bool", "void"}:
            return None
        return base_type or None

    def struct_node_dependency_names(self, node):
        dependencies = []
        for member in getattr(node, "members", []) or []:
            if isinstance(member, ArrayNode):
                member_type = getattr(
                    member,
                    "element_type",
                    getattr(member, "vtype", "float"),
                )
            elif hasattr(member, "member_type"):
                member_type = self.type_name_string(member.member_type)
            else:
                member_type = getattr(member, "vtype", "float")
            dependency = self.mapped_type_dependency_name(member_type)
            if dependency is not None:
                dependencies.append(dependency)
        return dependencies

    def order_struct_declaration_entries(self, entries):
        by_name = {}
        for entry in entries:
            by_name.setdefault(entry["name"], entry)

        order_index = {entry["name"]: index for index, entry in enumerate(entries)}
        dependencies = {
            entry["name"]: [
                dependency
                for dependency in entry["dependencies"]
                if dependency in by_name and dependency != entry["name"]
            ]
            for entry in entries
        }

        ordered = []
        visiting = set()
        visited = set()

        def visit(name):
            if name in visited or name in visiting:
                return
            visiting.add(name)
            for dependency in sorted(
                dependencies.get(name, []),
                key=lambda item: order_index[item],
            ):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(by_name[name])

        for entry in entries:
            visit(entry["name"])
        return ordered

    def generate_struct_declaration_code(self, node):
        saved_lines = self.code_lines
        saved_indent = self.indent_level
        self.code_lines = []
        self.indent_level = 0
        try:
            self.visit_StructNode(node)
            return "\n".join(self.code_lines).rstrip() + "\n\n"
        finally:
            self.code_lines = saved_lines
            self.indent_level = saved_indent

    def generate_ordered_data_struct_declarations(self, data_struct_nodes):
        entries = []
        for node in data_struct_nodes:
            entries.append(
                {
                    "name": node.name,
                    "dependencies": self.struct_node_dependency_names(node),
                    "code": self.generate_struct_declaration_code(node),
                }
            )

        for specialization in self.ordered_generic_struct_specializations().values():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in (
                            generic_struct_specialized_fields(
                                self.type_name_string,
                                specialization,
                            )
                        )
                    ],
                    "code": generate_generic_structs(
                        self,
                        {specialization["type_name"]: specialization},
                    ),
                }
            )

        for enum in self.struct_payload_enums:
            fields = enum_struct_fields(enum) or []
            entries.append(
                {
                    "name": enum.name,
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in fields
                    ],
                    "code": generate_enum_structs(self, [enum]),
                }
            )

        for type_text, specialization in self.generic_enum_specializations.items():
            entries.append(
                {
                    "name": specialization["struct_name"],
                    "dependencies": [
                        self.mapped_type_dependency_name(field_type)
                        for _field_name, field_type in generic_enum_specialized_fields(
                            self,
                            specialization,
                        )
                    ],
                    "code": generate_generic_enum_structs(
                        self,
                        {type_text: specialization},
                    ),
                }
            )

        code = ""
        for entry in self.order_struct_declaration_entries(entries):
            code += entry["code"]
        return code

    def default_value_expression_for_type(self, type_value):
        mapped_type = self.map_type(type_value)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type or ""))
        if array_suffix:
            return None

        vector_info = self.vector_type_info(mapped_type)
        if vector_info is not None:
            scalar = (
                "0.0" if vector_info["component_type"] in {"float", "double"} else "0"
            )
            args = ", ".join([scalar] * len(vector_info["components"]))
            return f"{vector_info['constructor']}({args})"

        if base_type in self.struct_member_types:
            defaults = [
                self.default_value_expression_for_type(field_type)
                or self.primitive_default_value_expression(field_type)
                for field_type in self.struct_member_types[base_type].values()
            ]
            return f"{base_type}{{{', '.join(defaults)}}}"

        return None

    def primitive_default_value_expression(self, type_value):
        mapped_type = self.map_type(type_value)
        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "double"}:
            return "0.0"
        if mapped_type in {"int", "unsigned int"}:
            return "0"
        return f"{mapped_type}{{}}"

    def is_hip_data_struct_node(self, node):
        if getattr(node, "is_cbuffer", False):
            return False
        for member in getattr(node, "members", []) or []:
            if isinstance(member, (EnumNode, FunctionNode, StructNode)):
                return False
        return True

    def visit_ShaderNode(self, node: ShaderNode) -> str:
        """Render a full shader/program AST as a HIP translation unit."""
        structs = getattr(node, "structs", [])
        self.add_generated_code(
            generate_enum_constants(
                self,
                self.plain_enums + self.struct_payload_enums,
                qualifier="static const",
            )
        )
        self.add_generated_code(
            generate_generic_enum_constants(
                self,
                self.generic_enum_struct_definitions,
                qualifier="static const",
            )
        )

        data_struct_nodes = []
        for struct in structs:
            if isinstance(struct, EnumNode):
                if struct in self.plain_enums:
                    self.visit(struct)
                continue
            if not isinstance(struct, StructNode):
                self.visit(struct)
                continue
            if struct.name in self.generic_enum_struct_definitions:
                continue
            if struct.name in self.generic_struct_definitions:
                continue
            if self.is_hip_data_struct_node(struct):
                data_struct_nodes.append(struct)
            elif getattr(struct, "is_cbuffer", False):
                self.visit(struct)

        self.add_generated_code(
            self.generate_ordered_data_struct_declarations(data_struct_nodes)
        )
        self.add_generated_code(
            generate_enum_constructor_functions(self, self.struct_payload_enums)
        )
        self.add_generated_code(
            generate_generic_enum_constructor_functions(
                self,
                self.generic_enum_specializations,
            )
        )

        for constant in getattr(node, "constants", []) or []:
            self.visit(constant)

        global_vars = getattr(node, "global_variables", [])
        for var in global_vars:
            self.visit(var)

        emitted_stage_local_variables = {
            getattr(var, "name", None): self.type_name_string(
                self.get_variable_node_type(var)
            )
            for var in global_vars
            if getattr(var, "name", None)
        }
        for stage in (getattr(node, "stages", {}) or {}).values():
            for var in getattr(stage, "local_variables", []) or []:
                if self.is_hip_shared_variable(var):
                    continue
                var_name = getattr(var, "name", None)
                var_type = self.type_name_string(self.get_variable_node_type(var))
                if not var_name:
                    continue
                previous_type = emitted_stage_local_variables.get(var_name)
                if previous_type is not None:
                    if previous_type != var_type:
                        raise ValueError(
                            "Conflicting HIP stage-local declaration for "
                            f"'{var_name}': {previous_type} differs from {var_type}"
                        )
                    self.register_variable_type(var_name, var_type, var)
                    continue
                self.visit(var)
                emitted_stage_local_variables[var_name] = var_type

        cbuffers = getattr(node, "cbuffers", [])
        for cbuffer in cbuffers:
            self.visit(cbuffer)

        functions = getattr(node, "functions", [])
        for func in functions:
            self.visit(func)

        # Handle shader stages (new AST structure)
        if hasattr(node, "stages") and node.stages:
            emitted_local_functions = set()
            stage_entry_name_counts = self.stage_entry_name_counts(node.stages)
            for stage_type, stage in node.stages.items():
                if hasattr(stage, "entry_point"):
                    # Set the stage type context for proper qualifier handling
                    stage_name = (
                        str(stage_type).split(".")[-1].lower()
                        if hasattr(stage_type, "name")
                        else str(stage_type).lower()
                    )

                    # Temporarily set qualifier for compute stages
                    if stage_name == "compute" or "compute" in stage_name:
                        # Set the function qualifier to compute for proper __global__ generation
                        if hasattr(stage.entry_point, "qualifiers"):
                            if "compute" not in stage.entry_point.qualifiers:
                                stage.entry_point.qualifiers.append("compute")
                        else:
                            stage.entry_point.qualifiers = ["compute"]

                    local_captures = self.collect_stage_local_function_captures(stage)
                    self.hip_function_capture_params.update(local_captures)
                    saved_stage_name = self.current_stage_name
                    saved_override = getattr(
                        self, "current_stage_entry_function_name", None
                    )
                    saved_stage_local_variables = getattr(
                        self, "current_stage_local_variables", []
                    )
                    try:
                        self.current_stage_name = saved_stage_name
                        for func in getattr(stage, "local_functions", []):
                            if id(func) in emitted_local_functions:
                                continue
                            self.visit(func)
                            emitted_local_functions.add(id(func))

                        self.current_stage_name = stage_name
                        self.current_stage_entry_function_name = (
                            self.stage_entry_function_name(
                                stage_name, stage.entry_point, stage_entry_name_counts
                            )
                        )
                        self.current_stage_local_variables = (
                            getattr(stage, "local_variables", []) or []
                        )
                        self.visit(stage.entry_point)
                    finally:
                        self.current_stage_entry_function_name = saved_override
                        self.current_stage_name = saved_stage_name
                        self.current_stage_local_variables = saved_stage_local_variables
                        for captured_name in local_captures:
                            self.hip_function_capture_params.pop(captured_name, None)

        return ""

    def stage_entry_name_counts(self, stages):
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if normalize_stage_name(stage_name) == "compute" and name == "main":
            return "compute_main"
        if stage_name in self.hip_ray_stage_names() and name == "main":
            return f"{stage_name}_{name}"
        if (
            stage_name
            in {"geometry", "tessellation_control", "tessellation_evaluation"}
            | self.hip_mesh_task_stage_names()
            and name == "main"
        ):
            return f"{stage_name}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name

    def collect_stage_local_function_captures(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        entry_params = list(
            getattr(entry_point, "parameters", getattr(entry_point, "params", []))
        )
        entry_params_by_name = {
            getattr(param, "name", None): param for param in entry_params
        }
        entry_params_by_name = {
            name: param for name, param in entry_params_by_name.items() if name
        }
        if not entry_params_by_name:
            return {}

        captures = {}
        for function in getattr(stage, "local_functions", []) or []:
            if getattr(function, "body", None) is None:
                continue
            used_names = self.collect_function_free_identifier_names(function)
            captured_params = [
                entry_params_by_name[name]
                for name in entry_params_by_name
                if name in used_names
            ]
            if captured_params:
                captures[getattr(function, "name", None)] = captured_params
        return {name: params for name, params in captures.items() if name}

    def is_hip_shared_variable(self, node):
        """Return whether a stage-local variable maps to HIP shared memory."""
        return any(
            "shared" in qualifier_name or "workgroup" in qualifier_name
            for qualifier_name in (
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            )
        )

    def collect_function_free_identifier_names(self, function):
        params = getattr(function, "parameters", getattr(function, "params", []))
        bound_names = {getattr(param, "name", None) for param in params}
        bound_names = {name for name in bound_names if name}
        body = getattr(function, "body", None)
        local_names = {
            getattr(node, "name", None)
            for node in self.query_walk_nodes(body)
            if isinstance(node, VariableNode)
        }
        bound_names.update(name for name in local_names if name)
        return {
            node.name
            for node in self.query_walk_nodes(body)
            if isinstance(node, IdentifierNode) and node.name not in bound_names
        }

    def visit_FunctionNode(self, node: FunctionNode) -> str:
        """Render a CrossGL function or compute entry point as HIP code."""
        saved_variable_types = self.variable_types.copy()
        saved_image_resource_accesses = self.image_resource_accesses.copy()
        saved_buffer_resource_accesses = self.buffer_resource_accesses.copy()
        saved_glsl_buffer_block_accesses = self.glsl_buffer_block_accesses.copy()
        saved_glsl_buffer_block_layouts = self.glsl_buffer_block_layouts.copy()
        self.current_function = node.name
        saved_current_function_return_type = self.current_function_return_type
        saved_current_function_name = self.current_function_name
        saved_stage_builtin_aliases = self.stage_builtin_aliases
        saved_current_function_is_kernel_entry = self.current_function_is_kernel_entry
        saved_structured_buffer_length_parameters = (
            self.current_structured_buffer_length_parameters
        )
        self.current_function_name = node.name
        self.current_structured_buffer_length_parameters = {}
        self.stage_builtin_aliases = {}
        self.current_function_is_kernel_entry = False

        qualifiers = []
        source_qualifiers = list(getattr(node, "qualifiers", []) or [])
        stage_attribute_names = {
            "compute",
            "fragment",
            "geometry",
            "kernel",
            "task",
            "mesh",
            "vertex",
        } | self.hip_ray_stage_names()
        source_qualifiers.extend(
            attribute_name
            for attribute_name in (
                getattr(attribute, "name", None)
                for attribute in getattr(node, "attributes", []) or []
            )
            if normalize_stage_name(attribute_name) in stage_attribute_names
        )
        source_qualifiers = [
            qualifier for qualifier in source_qualifiers if qualifier is not None
        ]
        if source_qualifiers:
            for qualifier in source_qualifiers:
                qualifier_name = normalize_stage_name(qualifier)
                if qualifier_name in {"kernel", "compute"}:
                    qualifiers.append("__global__")
                elif qualifier_name == "device":
                    qualifiers.append("__device__")
                else:
                    qualifiers.append("__device__")
        elif hasattr(node, "qualifier") and node.qualifier:
            qualifier_name = normalize_stage_name(node.qualifier)
            if qualifier_name in {"kernel", "compute"}:
                qualifiers.append("__global__")
            elif qualifier_name == "device":
                qualifiers.append("__device__")
            else:
                qualifiers.append("__device__")
        else:
            qualifiers.append("__device__")

        stage_name = self.function_stage_name(node)
        is_kernel_entry = "__global__" in qualifiers
        self.current_function_is_kernel_entry = is_kernel_entry

        if hasattr(node, "return_type"):
            self.current_function_return_type = node.return_type
            return_type = self.map_type(node.return_type)
        else:
            self.current_function_return_type = "void"
            return_type = "void"

        param_list = list(getattr(node, "parameters", getattr(node, "params", [])))
        param_list.extend(self.hip_function_capture_params.get(node.name, []))
        return_semantic = self.semantic_from_node(node)
        fragment_output_param = None
        fragment_output_params = [
            param
            for param in param_list
            if is_fragment_output_parameter(
                stage_name,
                param,
                self.semantic_from_node(param),
            )
        ]
        if (
            fragment_output_params
            and return_semantic is None
            and self.map_type(self.current_function_return_type) == "void"
        ):
            fragment_output_param = fragment_output_params[0]
            self.current_function_return_type = self.get_parameter_type(
                fragment_output_param
            )
            return_type = self.map_type(self.current_function_return_type)
            return_semantic = self.semantic_from_node(fragment_output_param)
            param_list = [
                param for param in param_list if param is not fragment_output_param
            ]

        self.validate_hip_return_semantic(
            stage_name, self.current_function_return_type, return_semantic
        )
        self.validate_hip_struct_return_semantics(
            stage_name, self.current_function_return_type
        )
        if is_kernel_entry:
            self.current_function_return_type = "void"
            return_type = "void"

        self.validate_hip_stage_parameter_semantics(stage_name, param_list)
        if stage_name == "geometry":
            self.validate_hip_geometry_stage(node, param_list)
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.validate_hip_tessellation_stage(node, stage_name)
        if stage_name in self.hip_mesh_task_stage_names():
            self.validate_hip_mesh_task_stage(node, stage_name)
        param_declarations = []
        for param in param_list:
            param_type = self.get_parameter_type(param)
            param_name = getattr(param, "name", getattr(param, "param_name", None))
            builtin_role = self.hip_compute_builtin_parameter_role(stage_name, param)
            if param_name and builtin_role is not None:
                self.register_variable_type(param_name, param_type, param)
                self.stage_builtin_aliases[param_name] = builtin_role
                continue

            param_declarations.append(self.visit_parameter(param))
            metadata_param = self.query_metadata_parameter(param_name, param_type)
            if metadata_param:
                param_declarations.append(metadata_param)
            length_param = self.structured_buffer_length_parameter(
                node.name, param_name, param_type
            )
            if length_param:
                self.current_structured_buffer_length_parameters[param_name] = (
                    self.structured_buffer_length_name(param_name)
                )
                param_declarations.append(length_param)
            counter_param = self.structured_buffer_counter_parameter(
                param_name, param_type
            )
            if counter_param:
                param_declarations.append(counter_param)
        params = ", ".join(param_declarations)

        qualifier_str = " ".join(qualifiers)
        function_name = getattr(self, "current_stage_entry_function_name", None)
        if not function_name:
            function_name = node.name
        if is_kernel_entry and function_name == "main":
            function_name = "compute_main"
        launch_bounds = self.hip_compute_launch_bounds_attribute(
            node, stage_name, is_kernel_entry
        )
        if launch_bounds is not None:
            signature = (
                f"{qualifier_str} {return_type} {launch_bounds} "
                f"{function_name}({params})"
            )
        else:
            signature = f"{qualifier_str} {return_type} {function_name}({params})"
        body = getattr(node, "body", None)

        if stage_name in self.hip_ray_stage_names():
            self.add_line(
                "// CrossGL ray stage: "
                f"{self.hip_ray_stage_metadata_name(stage_name)}"
            )
        if stage_name == "geometry":
            self.add_generated_code(
                self.generate_hip_geometry_stage_comments(node, param_list)
            )
        if stage_name in {"tessellation_control", "tessellation_evaluation"}:
            self.add_generated_code(
                self.generate_hip_tessellation_stage_comments(node, stage_name)
            )
        if stage_name in self.hip_mesh_task_stage_names():
            self.add_generated_code(
                self.generate_hip_mesh_task_stage_comments(node, stage_name)
            )
        if return_semantic:
            self.add_line(
                f"// CrossGL return semantic: {self.map_semantic(return_semantic)}"
            )
        if stage_name:
            for param in param_list:
                param_semantic = self.semantic_from_node(param)
                if param_semantic:
                    param_name = getattr(
                        param, "name", getattr(param, "param_name", "param")
                    )
                    self.add_line(
                        f"// CrossGL parameter semantic: {param_name}: "
                        f"{self.map_semantic(param_semantic)}"
                    )
        if body is None:
            self.add_line(f"{signature};")
        else:
            self.add_line(signature)
            self.add_line("{")
            self.indent_level += 1
            fragment_output_name = None
            if fragment_output_param is not None:
                fragment_output_name = getattr(
                    fragment_output_param,
                    "name",
                    getattr(fragment_output_param, "param_name", None),
                )
                fragment_output_type = self.resource_type_with_access(
                    self.get_parameter_type(fragment_output_param),
                    fragment_output_param,
                )
                self.register_variable_type(
                    fragment_output_name,
                    fragment_output_type,
                    fragment_output_param,
                )
                self.add_line(
                    f"{self.format_typed_declarator(fragment_output_type, fragment_output_name)};"
                )
            for local_var in getattr(self, "current_stage_local_variables", []) or []:
                if self.is_hip_shared_variable(local_var):
                    self.visit(local_var)
            self.emit_body(body)
            if fragment_output_name is not None and not self.statement_body_terminates(
                body
            ):
                self.add_line(f"return {fragment_output_name};")

            self.indent_level -= 1
            self.add_line("}")

        self.add_line()
        self.current_function = None
        self.current_function_return_type = saved_current_function_return_type
        self.variable_types = saved_variable_types
        self.image_resource_accesses = saved_image_resource_accesses
        self.buffer_resource_accesses = saved_buffer_resource_accesses
        self.glsl_buffer_block_accesses = saved_glsl_buffer_block_accesses
        self.glsl_buffer_block_layouts = saved_glsl_buffer_block_layouts
        self.current_function_name = saved_current_function_name
        self.stage_builtin_aliases = saved_stage_builtin_aliases
        self.current_function_is_kernel_entry = saved_current_function_is_kernel_entry
        self.current_structured_buffer_length_parameters = (
            saved_structured_buffer_length_parameters
        )
        return ""

    def visit_parameter(self, param) -> str:
        if isinstance(param, dict):
            param_type = param.get("type", "int")
            param_name = param.get("name", "param")
        else:
            if hasattr(param, "param_type"):
                param_type = param.param_type
            elif hasattr(param, "vtype"):
                param_type = param.vtype
            else:
                param_type = "int"

            param_name = getattr(param, "name", "param")

        param_type = self.resource_type_with_access(param_type, param)
        geometry_stream_declaration = self.format_hip_geometry_stream_parameter(
            param_type,
            param_name,
        )
        if geometry_stream_declaration is not None:
            self.register_variable_type(
                param_name,
                self.hip_geometry_stream_mapped_type(param_type),
                param,
            )
            return geometry_stream_declaration

        self.register_variable_type(param_name, param_type, param)
        declaration = self.format_typed_declarator(param_type, param_name)
        return self.apply_resource_memory_qualifiers_to_declaration(
            declaration, param_name, param
        )

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if self.is_semantic_name(attr_name):
                return attr_name
        return None

    def is_semantic_name(self, name):
        if name is None:
            return False
        name = str(name)
        upper_name = name.upper()
        if name.startswith("gl_"):
            return True
        if upper_name in {
            "BINORMAL",
            "CALLABLE_DATA",
            "CALLABLEDATAEXT",
            "CALLABLEDATAINEXT",
            "COLOR",
            "HIT_ATTRIBUTE",
            "HITATTRIBUTEEXT",
            "NORMAL",
            "PAYLOAD",
            "POSITION",
            "RAYPAYLOADEXT",
            "RAYPAYLOADINEXT",
            "SV_DEPTH",
            "SV_DISPATCHRAYSDIMENSIONS",
            "SV_DISPATCHRAYSINDEX",
            "SV_DISPATCHTHREADID",
            "SV_DRAWID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_POSITION",
            "SV_PRIMITIVEID",
            "SV_SAMPLEINDEX",
            "SV_STARTINSTANCELOCATION",
            "SV_STARTVERTEXLOCATION",
            "SV_TARGET",
            "SV_VERTEXID",
            "TANGENT",
            "TEXCOORD",
        }:
            return True
        for prefix in ("COLOR", "SV_TARGET", "TEXCOORD"):
            if upper_name.startswith(prefix) and upper_name[len(prefix) :].isdigit():
                return True
        return False

    def map_semantic(self, semantic):
        if semantic is None:
            return ""
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth(any)"
        if lower_name == "gl_fragcolor":
            return "target(0)"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "SV_TARGET":
            return "target(0)"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "TEXCOORD":
            return "texcoord"
        if (
            upper_name.startswith("TEXCOORD")
            and upper_name[len("TEXCOORD") :].isdigit()
        ):
            return f"texcoord({upper_name[len('TEXCOORD'):]})"
        if upper_name == "COLOR":
            return "color"
        if upper_name.startswith("COLOR") and upper_name[len("COLOR") :].isdigit():
            return f"color({upper_name[len('COLOR'):]})"
        generic_semantics = {
            "BINORMAL": "binormal",
            "NORMAL": "normal",
            "POSITION": "position",
            "TANGENT": "tangent",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_BaseVertex": "start_vertex_location",
            "gl_BaseInstance": "start_instance_location",
            "gl_DrawID": "draw_id",
            "gl_WorkGroupID": "workgroup_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "gl_WorkGroupSize": "workgroup_size",
            "gl_NumWorkGroups": "num_workgroups",
            "payload": "ray_payload",
            "rayPayloadEXT": "ray_payload",
            "rayPayloadInEXT": "ray_payload",
            "hit_attribute": "hit_attribute",
            "hitAttributeEXT": "hit_attribute",
            "callable_data": "callable_data",
            "callableDataEXT": "callable_data",
            "callableDataInEXT": "callable_data",
            "gl_LaunchIDEXT": "launch_id",
            "gl_LaunchSizeEXT": "launch_size",
            "gl_HitTEXT": "hit_t",
            "gl_HitKindEXT": "hit_kind",
            "SV_DispatchRaysIndex": "launch_id",
            "SV_DispatchRaysDimensions": "launch_size",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DrawID": "draw_id",
            "SV_GroupID": "workgroup_id",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_InstanceID": "instance_id",
            "SV_IsFrontFace": "front_facing",
            "SV_PrimitiveID": "primitive_id",
            "SV_SampleIndex": "sample_index",
            "SV_StartInstanceLocation": "start_instance_location",
            "SV_StartVertexLocation": "start_vertex_location",
            "SV_VertexID": "vertex_id",
        }
        return generic_semantics.get(semantic_name, semantic_name)

    def hip_semantic_output_kind(self, semantic):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name in {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_numworkgroups",
            "gl_pointcoord",
            "gl_vertexid",
            "gl_workgroupsize",
            "gl_workgroupid",
        } or upper_name in {
            "SV_DISPATCHTHREADID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
            return "input_only"
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def is_hip_float4_type(self, type_name):
        return self.map_type(type_name) == "float4"

    def is_hip_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "float"

    def validate_hip_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.hip_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return
        if kind in {"position", "color"}:
            if self.is_hip_float4_type(type_name):
                return
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "expected vec4-compatible type"
            )
        if kind == "depth" and not self.is_hip_float_scalar_type(type_name):
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "expected float type"
            )

    def validate_hip_output_semantic_stage(self, stage_name, semantic, context):
        kind = self.hip_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if stage_name is None:
            return
        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if stage_name not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise ValueError(
                f"Unsupported {semantic} {context} for HIP {stage_name} stage; "
                f"valid stage is {allowed}"
            )

    def function_stage_name(self, node):
        if self.current_stage_name:
            return self.current_stage_name
        qualifiers = list(getattr(node, "qualifiers", []) or [])
        qualifier = getattr(node, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)
        qualifiers.extend(
            getattr(attribute, "name", None)
            for attribute in getattr(node, "attributes", []) or []
        )
        supported_stage_names = (
            {
                "compute",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "vertex",
            }
            | self.hip_ray_stage_names()
            | self.hip_mesh_task_stage_names()
        )
        for qualifier in qualifiers:
            qualifier_name = normalize_stage_name(qualifier)
            if qualifier_name in supported_stage_names:
                return qualifier_name
        return None

    def hip_stage_attribute_arguments(self, func, attribute_name):
        requested = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == requested:
                return self.attribute_arguments(attr)
        return []

    def literal_int_value(self, node):
        if isinstance(node, LiteralNode):
            value = node.value
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                literal_type = getattr(
                    getattr(node, "literal_type", None), "name", None
                )
                if literal_type in {"char", "string"}:
                    return None
                try:
                    return int(value, 0)
                except ValueError:
                    return None
            return None

        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator not in {"+", "-"}:
                return None
            value = self.literal_int_value(node.operand)
            if value is None:
                return None
            if operator == "-":
                return -value
            return value

        return None

    def hip_compute_launch_bounds_attribute(self, func, stage_name, is_kernel_entry):
        """Return a HIP launch-bounds attribute for compute @numthreads metadata."""
        if not is_kernel_entry or stage_name != "compute":
            return None

        for attribute_name in ("numthreads", "local_size", "workgroup_size"):
            arguments = self.hip_stage_attribute_arguments(func, attribute_name)
            if not arguments:
                continue
            if len(arguments) != 3:
                raise ValueError(
                    f"HIP compute stage {attribute_name} requires exactly "
                    "three arguments"
                )

            values = []
            literal_product = 1
            all_literal = True
            for axis, argument in zip("xyz", arguments):
                literal_value = self.literal_int_value(argument)
                if literal_value is not None:
                    if literal_value <= 0:
                        raise ValueError(
                            f"HIP compute stage {attribute_name} {axis} "
                            f"dimension ({literal_value}) must be positive"
                        )
                    literal_product *= literal_value
                else:
                    all_literal = False
                value_text = self.attribute_value_to_string(argument)
                values.append(str(value_text))

            max_threads = (
                str(literal_product)
                if all_literal
                else " * ".join(f"({value})" for value in values)
            )
            if attribute_name == "workgroup_size" and max_threads == "1":
                continue
            return f"__launch_bounds__({max_threads})"

        return None

    def hip_geometry_maxvertexcount(self, func):
        arguments = self.hip_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise ValueError("HIP geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise ValueError(
                "HIP geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                f"HIP geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def hip_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.hip_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise ValueError(
                f"HIP {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def hip_tessellation_output_control_points(self, func):
        arguments = self.hip_stage_attribute_arguments(func, "outputcontrolpoints")
        if not arguments:
            raise ValueError(
                "HIP tessellation_control stage requires outputcontrolpoints "
                "attribute"
            )
        if len(arguments) != 1:
            raise ValueError(
                "HIP tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise ValueError(
                "HIP tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        return value_text

    def canonical_hip_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def hip_tessellation_domain(self, func, stage_name, required=False):
        value = self.hip_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise ValueError(f"HIP {stage_name} stage requires domain attribute")
            return None
        canonical = self.canonical_hip_tessellation_domain(value)
        if canonical is None:
            raise ValueError(
                f"HIP {stage_name} stage domain '{value}' must be tri, quad, "
                "or isoline"
            )
        return canonical

    def hip_tessellation_partitioning(self, func, stage_name):
        value = self.hip_stage_attribute_value(func, "partitioning", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        valid_partitioning = {
            "fractional_even",
            "fractional_odd",
            "integer",
            "pow2",
        }
        if normalized not in valid_partitioning:
            valid = ", ".join(sorted(valid_partitioning))
            raise ValueError(
                f"HIP {stage_name} stage partitioning '{value}' must be one of: "
                f"{valid}"
            )
        return normalized

    def validate_hip_tessellation_stage(self, func, stage_name):
        if stage_name == "tessellation_control":
            self.hip_tessellation_output_control_points(func)
            self.hip_tessellation_domain(func, stage_name)
        elif stage_name == "tessellation_evaluation":
            self.hip_tessellation_domain(func, stage_name, required=True)
        self.hip_tessellation_partitioning(func, stage_name)

    def generate_hip_tessellation_stage_comments(self, func, stage_name):
        lines = []
        if stage_name == "tessellation_control":
            output_points = self.hip_tessellation_output_control_points(func)
            lines.append(
                "// CrossGL tessellation control stage: "
                f"outputcontrolpoints={output_points}\n"
            )
        else:
            domain = self.hip_tessellation_domain(func, stage_name, required=True)
            lines.append(
                "// CrossGL tessellation evaluation stage: " f"domain={domain}\n"
            )

        domain = self.hip_tessellation_domain(
            func, stage_name, required=stage_name == "tessellation_evaluation"
        )
        if stage_name == "tessellation_control" and domain is not None:
            lines.append(f"// CrossGL tessellation domain: {domain}\n")

        partitioning = self.hip_tessellation_partitioning(func, stage_name)
        if partitioning is not None:
            lines.append(f"// CrossGL tessellation partitioning: {partitioning}\n")

        output_topology = self.hip_stage_attribute_value(
            func, "outputtopology", stage_name
        )
        if output_topology is not None:
            lines.append(
                "// CrossGL tessellation output topology: " f"{output_topology}\n"
            )

        patch_constant_function = self.hip_stage_attribute_value(
            func, "patchconstantfunc", stage_name
        )
        if patch_constant_function is not None:
            lines.append(
                "// CrossGL tessellation patch constant function: "
                f"{patch_constant_function}\n"
            )

        return "".join(lines)

    def hip_mesh_task_stage_label(self, stage_name):
        return {
            "amplification": "amplification/task",
            "object": "object/task",
        }.get(stage_name, stage_name)

    def hip_mesh_task_positive_attribute(self, func, attribute_name, stage_name):
        value = self.hip_stage_attribute_value(func, attribute_name, stage_name)
        if value is None:
            return None
        value_int = self.literal_int_value(
            self.hip_stage_attribute_arguments(func, attribute_name)[0]
        )
        if value_int is not None and value_int <= 0:
            raise ValueError(
                f"HIP {stage_name} stage {attribute_name} ({value_int}) "
                "must be positive"
            )
        return value

    def hip_mesh_task_numthreads(self, func, stage_name):
        for attribute_name in ("numthreads", "local_size", "workgroup_size"):
            arguments = self.hip_stage_attribute_arguments(func, attribute_name)
            if not arguments:
                continue
            if len(arguments) != 3:
                raise ValueError(
                    f"HIP {stage_name} stage {attribute_name} requires exactly "
                    "three arguments"
                )
            values = []
            for axis, argument in zip("xyz", arguments):
                value = self.literal_int_value(argument)
                if value is not None and value <= 0:
                    raise ValueError(
                        f"HIP {stage_name} stage {attribute_name} {axis} "
                        f"dimension ({value}) must be positive"
                    )
                values.append(self.attribute_value_to_string(argument))
            return ", ".join(values)
        return None

    def hip_mesh_task_output_topology(self, func, stage_name):
        value = self.hip_stage_attribute_value(func, "outputtopology", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        topology_aliases = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        topology = topology_aliases.get(normalized)
        if topology is None:
            raise ValueError(
                f"HIP {stage_name} stage outputtopology '{value}' must be "
                "point, line, or triangle"
            )
        return topology

    def validate_hip_mesh_task_stage(self, func, stage_name):
        self.hip_mesh_task_numthreads(func, stage_name)
        if stage_name == "mesh":
            self.hip_mesh_task_output_topology(func, stage_name)
            self.hip_mesh_task_positive_attribute(func, "max_vertices", stage_name)
            self.hip_mesh_task_positive_attribute(func, "max_primitives", stage_name)

    def generate_hip_mesh_task_stage_comments(self, func, stage_name):
        label = self.hip_mesh_task_stage_label(stage_name)
        lines = [f"// CrossGL {label} stage\n"]

        numthreads = self.hip_mesh_task_numthreads(func, stage_name)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {numthreads}\n")

        if stage_name == "mesh":
            topology = self.hip_mesh_task_output_topology(func, stage_name)
            if topology is not None:
                lines.append(f"// CrossGL mesh output topology: {topology}\n")
            for attribute_name in ("max_vertices", "max_primitives"):
                value = self.hip_mesh_task_positive_attribute(
                    func, attribute_name, stage_name
                )
                if value is not None:
                    lines.append(f"// CrossGL mesh {attribute_name}: {value}\n")

        return "".join(lines)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def hip_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def hip_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.hip_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def hip_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def hip_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            if isinstance(size, int):
                return size
            return self.literal_int_value(size)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        return array_size

    def validate_hip_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.hip_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.hip_parameter_is_array(parameter):
                raise ValueError(
                    "HIP geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.hip_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise ValueError(
                    "HIP geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_hip_geometry_stage(self, func, parameters):
        self.hip_geometry_maxvertexcount(func)

        if not any(
            self.hip_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise ValueError(
                "HIP geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.hip_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise ValueError(
                "HIP geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_hip_geometry_input_primitive_arity(parameters)

    def generate_hip_geometry_stage_comments(self, func, parameters):
        maxvertexcount = self.hip_geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.hip_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.hip_geometry_stream_info(
                self.parameter_raw_type(parameter)
            )
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"// CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                "// CrossGL geometry input primitive: "
                f"{', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                "// CrossGL geometry output stream: "
                f"{', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def validate_hip_return_semantic(self, stage_name, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "void":
            raise ValueError(
                f"Unsupported {semantic} return semantic for HIP codegen; "
                "void return type"
            )
        self.validate_hip_output_semantic_stage(stage_name, semantic, "return semantic")
        self.validate_hip_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_hip_struct_return_semantics(self, stage_name, return_type):
        if stage_name is None:
            return
        base_type = self.type_name_string(return_type)
        if not base_type:
            return
        base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_hip_output_semantic_stage(stage_name, semantic, context)
            self.validate_hip_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def hip_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def hip_stage_parameter_semantic_rules(self):
        ray_stages = self.hip_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "unsigned int", {"vertex"}),
            "gl_instanceid": ("instance_id", "unsigned int", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "int", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "unsigned int", {"vertex"}),
            "gl_drawid": ("draw_id", "unsigned int", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "unsigned int", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "unsigned int", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "int", {"vertex"}),
            "SV_STARTINSTANCELOCATION": (
                "start_instance_location",
                "unsigned int",
                {"vertex"},
            ),
            "SV_DRAWID": ("draw_id", "unsigned int", {"vertex"}),
            "gl_position": ("position", "float4", {"fragment"}),
            "gl_fragcoord": ("position", "float4", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "float2", {"fragment"}),
            "gl_primitiveid": (
                "primitive_id",
                "unsigned int",
                {"fragment"} | ray_hit_stages,
            ),
            "gl_sampleid": ("sample_index", "unsigned int", {"fragment"}),
            "SV_POSITION": ("position", "float4", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": (
                "primitive_id",
                "unsigned int",
                {"fragment"} | ray_hit_stages,
            ),
            "SV_SAMPLEINDEX": ("sample_index", "unsigned int", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "uint3", {"compute"}),
            "gl_localinvocationid": ("local_invocation_id", "uint3", {"compute"}),
            "gl_globalinvocationid": ("global_invocation_id", "uint3", {"compute"}),
            "gl_localinvocationindex": (
                "local_invocation_index",
                "unsigned int",
                {"compute"},
            ),
            "gl_workgroupsize": ("workgroup_size", "uint3", {"compute"}),
            "gl_numworkgroups": ("num_workgroups", "uint3", {"compute"}),
            "SV_GROUPID": ("workgroup_id", "uint3", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "uint3", {"compute"}),
            "SV_DISPATCHTHREADID": ("global_invocation_id", "uint3", {"compute"}),
            "SV_GROUPINDEX": ("local_invocation_index", "unsigned int", {"compute"}),
            "gl_launchidext": ("launch_id", "uint3", ray_stages),
            "gl_launchsizeext": ("launch_size", "uint3", ray_stages),
            "gl_hittext": ("hit_t", "float", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "unsigned int", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "uint3", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": ("launch_size", "uint3", ray_stages),
        }

    def hip_compute_builtin_parameter_role(self, stage_name, param):
        """Return the compute built-in role for a semantic kernel parameter."""
        if stage_name != "compute":
            return None

        semantic = self.semantic_from_node(param)
        if semantic is None:
            return None

        rule = self.hip_stage_parameter_semantic_rules().get(
            self.hip_stage_parameter_semantic_key(semantic)
        )
        if rule is None:
            return None

        role, _expected_type, allowed_stages = rule
        if "compute" not in allowed_stages:
            return None
        if role in {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "num_workgroups",
            "workgroup_size",
            "workgroup_id",
        }:
            return role
        return None

    def hip_compute_builtin_expression(self, role, component=None):
        """Return the HIP expression for a CrossGL compute built-in role."""
        local_index = (
            "(threadIdx.z * blockDim.y * blockDim.x + "
            "threadIdx.y * blockDim.x + threadIdx.x)"
        )
        component_expressions = {
            "global_invocation_id": {
                "x": "(blockIdx.x * blockDim.x + threadIdx.x)",
                "y": "(blockIdx.y * blockDim.y + threadIdx.y)",
                "z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            },
            "local_invocation_id": {
                "x": "threadIdx.x",
                "y": "threadIdx.y",
                "z": "threadIdx.z",
            },
            "workgroup_id": {
                "x": "blockIdx.x",
                "y": "blockIdx.y",
                "z": "blockIdx.z",
            },
            "workgroup_size": {
                "x": "blockDim.x",
                "y": "blockDim.y",
                "z": "blockDim.z",
            },
            "num_workgroups": {
                "x": "gridDim.x",
                "y": "gridDim.y",
                "z": "gridDim.z",
            },
        }

        if role == "local_invocation_index":
            return local_index if component is None else None

        role_components = component_expressions.get(role)
        if role_components is None:
            return None
        if component is not None:
            if component in role_components:
                return role_components[component]
            swizzle_components = tuple(str(component))
            if all(part in role_components for part in swizzle_components):
                constructor = {
                    2: "make_uint2",
                    3: "make_uint3",
                    4: "make_uint4",
                }.get(len(swizzle_components))
                if constructor is not None:
                    values = ", ".join(
                        role_components[part] for part in swizzle_components
                    )
                    return f"{constructor}({values})"
            return None
        return "make_uint3({x}, {y}, {z})".format(**role_components)

    def hip_compute_builtin_role_for_name(self, name):
        """Return the compute built-in role represented by a source identifier."""
        return {
            "gl_GlobalInvocationID": "global_invocation_id",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DISPATCHTHREADID": "global_invocation_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_GROUPTHREADID": "local_invocation_id",
            "gl_WorkGroupID": "workgroup_id",
            "SV_GroupID": "workgroup_id",
            "SV_GROUPID": "workgroup_id",
            "gl_WorkGroupSize": "workgroup_size",
            "gl_NumWorkGroups": "num_workgroups",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GROUPINDEX": "local_invocation_index",
        }.get(name)

    def hip_compute_builtin_type(self, name):
        """Return the HIP value type for a compute built-in identifier."""
        role = self.hip_compute_builtin_role_for_name(name)
        if role is None:
            return None
        if role == "local_invocation_index":
            return "uint"
        return "uint3"

    def hip_compute_builtin_member_type(self, name, member):
        """Return the type of a direct compute built-in member/swizzle."""
        root_type = self.hip_compute_builtin_type(name)
        vector_info = self.vector_type_info(root_type)
        if vector_info is None:
            return None
        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        components = [component_aliases.get(component) for component in str(member)]
        if not components or any(component is None for component in components):
            return None
        if any(component not in vector_info["components"] for component in components):
            return None
        if len(components) == 1:
            return vector_info["component_type"]
        return self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )

    def hip_stage_builtin_alias_expression(self, name, component=None):
        role = self.stage_builtin_aliases.get(name)
        if role is None:
            return None
        return self.hip_compute_builtin_expression(role, component)

    def source_identifier_shadows_builtin(self, name):
        """Return whether a source variable should block builtin-name lowering."""
        return (
            name is not None
            and name in self.variable_types
            and name not in self.stage_builtin_aliases
        )

    def validate_hip_stage_parameter_semantic_type(
        self, param, semantic, expected_type
    ):
        actual_type = self.map_type(self.get_parameter_type(param))
        if actual_type == expected_type:
            return
        raise ValueError(
            f"Unsupported {semantic} stage parameter semantic for HIP codegen; "
            f"expected {expected_type} type"
        )

    def validate_hip_stage_parameter_semantics(self, stage_name, parameters):
        if stage_name is None:
            return

        rules = self.hip_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for param in parameters or []:
            semantic = self.semantic_from_node(param)
            if semantic is None:
                continue

            semantic_key = self.hip_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if stage_name not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise ValueError(
                        f"Unsupported {semantic} stage parameter semantic for HIP "
                        f"{stage_name} stage; valid stage is {allowed}"
                    )
                param_name = getattr(param, "name", getattr(param, "param_name", None))
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise ValueError(
                        f"Duplicate HIP stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{param_name}'"
                    )
                seen_system_semantics[mapped_semantic] = param_name
                self.validate_hip_stage_parameter_semantic_type(
                    param, semantic, expected_type
                )
                continue

            kind = self.hip_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise ValueError(
                    f"Unsupported {semantic} stage parameter semantic for HIP "
                    f"{stage_name} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def visit_StructNode(self, node: StructNode) -> str:
        if getattr(node, "is_cbuffer", False):
            return self.emit_cbuffer_members_as_constant_memory(node)
        self.add_line(f"struct {node.name}")
        self.add_line("{")
        self.indent_level += 1

        members = getattr(node, "members", [])
        member_types = {}
        member_semantics = {}
        for member in members:
            if hasattr(member, "member_type"):
                member_type = member.member_type
            elif hasattr(member, "vtype"):
                member_type = member.vtype
            elif hasattr(member, "var_type"):
                member_type = member.var_type
            else:
                member_type = "float"

            member_type = self.resource_type_with_access(member_type, member)
            member_types[member.name] = member_type
            semantic = self.semantic_from_node(member)
            if semantic:
                member_semantics[member.name] = semantic
                self.validate_hip_builtin_semantic_type(
                    semantic, member_type, "struct member semantic"
                )
            semantic_comment = (
                f" // CrossGL semantic: {self.map_semantic(semantic)}"
                if semantic
                else ""
            )
            declaration = self.format_typed_declarator(member_type, member.name)
            declaration = self.apply_resource_memory_qualifiers_to_declaration(
                declaration, member.name, member
            )
            self.add_line(f"{declaration};{semantic_comment}")

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def cbuffer_member_type(self, member):
        if hasattr(member, "member_type"):
            return member.member_type
        if hasattr(member, "vtype"):
            return member.vtype
        if hasattr(member, "var_type"):
            return member.var_type
        return "int"

    def emit_cbuffer_members_as_constant_memory(self, node) -> str:
        metadata_comment = self.hip_resource_metadata_comment(
            node, getattr(node, "name", None), kind="cbuffer"
        )
        if metadata_comment:
            self.add_line(metadata_comment)
        self.add_line(f"// Constant buffer: {node.name}")
        for member in getattr(node, "members", []) or []:
            member_type = self.cbuffer_member_type(member)
            declaration = self.format_typed_declarator(member_type, member.name)
            self.add_line(f"__constant__ {declaration};")
        self.add_line()
        return ""

    def visit_VariableNode(self, node: VariableNode) -> str:
        var_type = self.get_variable_node_type(node)
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        if isinstance(initial_value, MatchNode):
            self.emit_match_expression_variable(node, initial_value, var_type)
            return ""

        metadata_comment = self.hip_resource_metadata_comment(node, var_type)
        if metadata_comment:
            self.add_line(metadata_comment)
        self.add_line(f"{self.format_variable_declaration(node)};")
        metadata_declaration = self.query_metadata_declaration(node.name, var_type)
        if metadata_declaration:
            self.add_line(f"{metadata_declaration};")
        length_declaration = self.structured_buffer_length_declaration(
            node.name, var_type
        )
        if length_declaration:
            self.add_line(f"{length_declaration};")
        counter_declaration = self.structured_buffer_counter_declaration(
            node.name, var_type
        )
        if counter_declaration:
            self.add_line(f"{counter_declaration};")
        return ""

    def visit_ConstantNode(self, node: ConstantNode) -> str:
        const_type = self.type_name_string(getattr(node, "const_type", "auto"))
        value = getattr(node, "value", None)
        self.register_variable_type(node.name, const_type, node, value)
        declaration_type = self.glsl_buffer_block_declaration_type(const_type, node)
        declaration = self.format_typed_declarator(declaration_type, node.name)
        if value is not None:
            declaration += f" = {self.visit(value)}"
        self.add_line(f"const {declaration};")
        return ""

    def format_variable_declaration(self, node: VariableNode) -> str:
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        var_type = self.get_variable_node_type(node)

        if var_type is None and initial_value is not None:
            inferred_type = self.expression_result_type(initial_value)
            self.register_variable_type(node.name, inferred_type, node, initial_value)
            declaration = self.format_typed_declarator(
                inferred_type or "auto", node.name
            )
        else:
            var_type = var_type or "int"
            self.register_variable_type(node.name, var_type, node, initial_value)
            var_type = self.glsl_buffer_block_declaration_type(var_type, node)
            declaration = self.format_typed_declarator(var_type, node.name)

        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )

        qualifiers = self.variable_memory_qualifiers(node)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        if initial_value is not None:
            declaration += f" = {self.visit(initial_value)}"

        return declaration

    def variable_memory_qualifiers(self, node: VariableNode):
        qualifiers = []
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if "workgroup" in qualifier_name or "shared" in qualifier_name:
                qualifiers.append("__shared__")
            elif "uniform" in qualifier_name:
                qualifiers.append("__constant__")
        return qualifiers

    def declaration_qualifier_names(self, node):
        """Return normalized declaration qualifier and attribute names."""
        names = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name:
                names.add(attr_name)
        return names

    def apply_resource_memory_qualifiers_to_declaration(self, declaration, name, node):
        """Apply HIP-native pointer qualifiers for resource-like declarations."""
        if "*" not in declaration or not name:
            return declaration

        names = self.declaration_qualifier_names(node)
        if not names:
            return declaration

        if (
            names & {"const", "constant", "readonly"}
            and "writeonly" not in names
            and not declaration.startswith("const ")
        ):
            declaration = f"const {declaration}"

        if "volatile" in names:
            if declaration.startswith("const "):
                declaration = f"const volatile {declaration[len('const '):]}"
            elif not declaration.startswith("volatile "):
                declaration = f"volatile {declaration}"

        if "restrict" in names and "__restrict__" not in declaration:
            name_token = f" {name}"
            if name_token in declaration:
                declaration = declaration.replace(
                    name_token, f" __restrict__{name_token}", 1
                )

        return declaration

    def emit_match_expression_variable(self, node, match_node, var_type):
        if var_type is None:
            var_type = infer_match_expression_result_type(self, match_node)
        var_type = var_type or "auto"
        self.register_variable_type(node.name, var_type, node)

        declaration = self.format_typed_declarator(var_type, node.name)
        declaration = self.apply_resource_memory_qualifiers_to_declaration(
            declaration, node.name, node
        )
        qualifiers = self.variable_memory_qualifiers(node)
        if qualifiers:
            declaration = f"{' '.join(qualifiers)} {declaration}"

        self.add_line(f"{declaration};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                node.name,
                var_type,
                self.indent_level,
                "HIP",
            )
        )

    def visit_CbufferNode(self, node: CbufferNode) -> str:
        return self.emit_cbuffer_members_as_constant_memory(node)

    def visit_list(self, node_list) -> str:
        for node in node_list:
            self.emit_statement(node)
        return ""

    def emit_statement(self, node):
        """Render and append one statement node when it produces code."""
        if node is None:
            return

        result = self.visit(node)
        if isinstance(result, str) and result.strip():
            self.add_line(f"{result};")

    def emit_body(self, body):
        """Render a list-like or block-like function body."""
        if isinstance(body, list):
            for stmt in body:
                self.emit_statement(stmt)
        elif hasattr(body, "statements"):
            for stmt in body.statements:
                self.emit_statement(stmt)
        else:
            self.emit_statement(body)

    def statement_list(self, body):
        if body is None:
            return []
        if isinstance(body, list):
            return body
        if hasattr(body, "statements"):
            return body.statements
        return [body]

    def statement_body_terminates(self, body):
        statements = self.statement_list(body)
        if not statements:
            return False
        return isinstance(statements[-1], (BreakNode, ContinueNode, ReturnNode))

    def visit_IfNode(self, node) -> str:
        condition = self.visit(node.if_condition)
        self.add_line(f"if ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.if_body)
        self.indent_level -= 1
        self.add_line("}")

        if hasattr(node, "else_body") and node.else_body:
            self.add_line("else")
            self.add_line("{")
            self.indent_level += 1
            self.emit_body(node.else_body)
            self.indent_level -= 1
            self.add_line("}")

        return ""

    def visit_ForNode(self, node) -> str:
        if isinstance(node.init, VariableNode):
            init = self.format_variable_declaration(node.init)
        elif hasattr(node.init, "expression"):
            init = self.visit(node.init.expression)
        else:
            init = self.visit(node.init) if node.init else ""
        condition = self.visit(node.condition) if node.condition else ""
        update = self.visit(node.update) if node.update else ""

        self.add_line(f"for ({init}; {condition}; {update})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_ForInNode(self, node) -> str:
        pattern = getattr(node, "pattern", "item")
        iterable = getattr(node, "iterable", None)

        if isinstance(iterable, RangeNode):
            start = self.visit(iterable.start)
            end = self.visit(iterable.end)
            comparator = "<=" if iterable.inclusive else "<"
        else:
            start = "0"
            end = self.visit(iterable)
            comparator = "<"

        self.add_line(
            f"for (int {pattern} = {start}; {pattern} {comparator} {end}; ++{pattern})"
        )
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(getattr(node, "body", []))
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_WhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line(f"while ({condition})")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_DoWhileNode(self, node) -> str:
        condition = self.visit(node.condition) if node.condition else ""

        self.add_line("do")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line(f"}} while ({condition});")

        return ""

    def visit_LoopNode(self, node) -> str:
        self.add_line("while (true)")
        self.add_line("{")
        self.indent_level += 1
        self.emit_body(node.body)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_SwitchNode(self, node) -> str:
        expression = self.visit(node.expression)

        self.add_line(f"switch ({expression})")
        self.add_line("{")
        self.indent_level += 1
        for case in getattr(node, "cases", []):
            self.visit(case)
        self.indent_level -= 1
        self.add_line("}")

        return ""

    def visit_MatchNode(self, node) -> str:
        if is_switch_lowerable_match(node):
            code = generate_switch_match(self, node, self.indent_level)
        else:
            code = generate_ordered_conditional_match(
                self, node, self.indent_level, "HIP"
            )
        self.add_generated_code(code)
        return ""

    def generate_switch_case(self, label, body, indent, auto_break=False):
        indent_str = "    " * indent
        if not auto_break and not self.statement_body_has_statements(body):
            return f"{indent_str}{label}:\n"

        code = f"{indent_str}{label}: {{\n"
        code += self.generate_scoped_statement_body(body, indent + 1)
        if auto_break and not self.statement_body_terminates(body):
            code += f"{indent_str}    break;\n"
        code += f"{indent_str}}}\n"
        return code

    def statement_body_has_statements(self, body):
        return bool(self.statement_list(body))

    def generate_scoped_statement_body(self, body, indent):
        saved_lines = self.code_lines
        saved_indent = self.indent_level
        self.code_lines = []
        self.indent_level = indent
        try:
            self.emit_body(body)
            if not self.code_lines:
                return ""
            return "\n".join(self.code_lines) + "\n"
        finally:
            self.code_lines = saved_lines
            self.indent_level = saved_indent

    def generate_expression(self, node):
        if node is None:
            return ""
        return self.visit(node)

    def generate_expression_with_expected(self, node, expected_type):
        saved_expected_type = self.current_expression_expected_type
        self.current_expression_expected_type = self.type_name_string(expected_type)
        try:
            return self.generate_expression(node)
        finally:
            self.current_expression_expected_type = saved_expected_type

    def visit_CaseNode(self, node) -> str:
        if getattr(node, "value", None) is None:
            self.add_line("default:")
        else:
            value = self.visit(node.value)
            self.add_line(f"case {value}:")

        self.indent_level += 1
        self.emit_body(getattr(node, "statements", []))
        self.indent_level -= 1

        return ""

    def visit_ReturnNode(self, node) -> str:
        if self.current_function_is_kernel_entry:
            if node.value and not isinstance(node.value, MatchNode):
                self.visit(node.value)
            self.add_line("return;")
            return ""
        if node.value:
            if isinstance(node.value, MatchNode):
                return self.emit_match_expression_return(node.value)
            value = self.visit(node.value)
            self.add_line(f"return {value};")
        else:
            self.add_line("return;")
        return ""

    def emit_match_expression_return(self, match_node) -> str:
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            raise ValueError(
                "Unsupported match expression for HIP codegen; return context "
                "requires a concrete non-void result type"
            )

        result_name = self.next_hip_temp_variable("match_value")
        self.add_line(f"{self.format_typed_declarator(return_type, result_name)};")
        self.add_generated_code(
            generate_match_expression_assignment(
                self,
                match_node,
                result_name,
                return_type,
                self.indent_level,
                "HIP",
            )
        )
        self.add_line(f"return {result_name};")
        return ""

    def visit_AssignmentNode(self, node) -> str:
        operator = getattr(node, "operator", getattr(node, "op", "="))
        diagnostic = self.glsl_buffer_block_write_diagnostic(node.left, "assignment")
        if diagnostic is not None:
            return diagnostic
        if operator == "=":
            diagnostic = self.structured_buffer_element_write_diagnostic(
                node.left, "assignment"
            )
        else:
            diagnostic = self.structured_buffer_element_read_write_diagnostic(
                node.left, "assignment"
            )
        if diagnostic is not None:
            return diagnostic
        swizzle_assignment = self.format_swizzle_assignment_expression(
            node.left,
            node.right,
            operator,
        )
        if swizzle_assignment is not None:
            return swizzle_assignment
        self.assignment_lhs_depth += 1
        try:
            left = self.visit(node.left)
        finally:
            self.assignment_lhs_depth -= 1
        right = self.visit(node.right)
        compound_binary_ops = {
            "+=": "+",
            "-=": "-",
            "*=": "*",
            "/=": "/",
            "%=": "%",
        }
        if operator in compound_binary_ops:
            lowered_right = self.lower_vector_binary_operation(
                node.left,
                left,
                node.right,
                right,
                compound_binary_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
            if operator == "%=":
                modulo = self.lower_scalar_modulo_operation(
                    node.left,
                    left,
                    node.right,
                    right,
                )
                if modulo is not None:
                    return f"{left} = {modulo}"
        compound_bitwise_ops = {
            "&=": "&",
            "|=": "|",
            "^=": "^",
            "<<=": "<<",
            ">>=": ">>",
        }
        if operator in compound_bitwise_ops:
            lowered_right = self.lower_vector_bitwise_operation(
                node.left,
                left,
                node.right,
                right,
                compound_bitwise_ops[operator],
            )
            if lowered_right is not None:
                return f"{left} = {lowered_right}"
        return f"{left} {operator} {right}"

    def format_swizzle_assignment_expression(self, target_node, value_node, operator):
        if not isinstance(target_node, MemberAccessNode):
            return None
        object_node = getattr(
            target_node, "object_expr", getattr(target_node, "object", None)
        )
        components = self.member_swizzle_components(target_node)
        if object_node is None or components is None or len(components) <= 1:
            return None

        object_info = self.vector_type_info(self.expression_result_type(object_node))
        if object_info is None:
            return None
        result_type = self.vector_type_for_components(
            object_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        object_expr = self.visit(object_node)
        value_expr = self.visit(value_node)
        temp_name = self.next_hip_temp_variable("swizzle_value")
        value_info = self.vector_type_info(self.expression_result_type(value_node))
        if value_info is not None:
            if len(value_info["components"]) != len(components):
                return None
            temp_type = result_info["type"]
            value_parts = [f"{temp_name}.{part}" for part in result_info["components"]]
        else:
            scalar_type = self.scalar_component_type(
                self.expression_result_type(value_node)
            )
            if scalar_type is None:
                scalar_type = self.vector_scalar_parameter_type(object_info)
            temp_type = self.map_type(scalar_type)
            value_parts = None

        statements = [f"{temp_type} {temp_name} = {value_expr}"]
        if value_parts is None:
            value_parts = [temp_name] * len(components)

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
        for component, value_part in zip(components, value_parts):
            target = f"{object_expr}.{component}"
            if operator == "=":
                statements.append(f"{target} = {value_part}")
            elif binary_op is not None:
                statements.append(f"{target} = ({target} {binary_op} {value_part})")
            else:
                return None
        return "; ".join(statements)

    def visit_BinaryOpNode(self, node) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = getattr(node, "operator", getattr(node, "op", "+"))

        logical = self.lower_vector_logical_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if logical is not None:
            return logical

        bitwise = self.lower_vector_bitwise_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if bitwise is not None:
            return bitwise

        if operator == "and":
            return f"({left} && {right})"
        elif operator == "or":
            return f"({left} || {right})"
        comparison = self.lower_vector_comparison_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if comparison is not None:
            return comparison
        lowered = self.lower_vector_binary_operation(
            node.left,
            left,
            node.right,
            right,
            operator,
        )
        if lowered is not None:
            return lowered
        if operator == "%":
            modulo = self.lower_scalar_modulo_operation(
                node.left,
                left,
                node.right,
                right,
            )
            if modulo is not None:
                return modulo
        return f"({left} {operator} {right})"

    def visit_UnaryOpNode(self, node) -> str:
        operand = self.visit(node.operand)

        if node.op in {"!", "not"}:
            lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
            if lowered is not None:
                return lowered
            return f"!{operand}"
        elif node.op in ["++", "--"]:
            if getattr(node, "is_postfix", getattr(node, "postfix", False)):
                return f"{operand}{node.op}"
            else:
                return f"{node.op}{operand}"
        lowered = self.lower_vector_unary_operation(node.operand, operand, node.op)
        if lowered is not None:
            return lowered
        return f"{node.op}{operand}"

    def visit_ConstructorNode(self, node) -> str:
        enum_constructor = generate_enum_constructor_expression(self, node)
        if enum_constructor is not None:
            return enum_constructor

        struct_constructor = generate_struct_constructor_expression(self, node)
        if struct_constructor is not None:
            return struct_constructor

        constructor_type = self.type_name_string(
            getattr(node, "constructor_type", getattr(node, "vtype", None))
        )
        positional_args = list(getattr(node, "arguments", []) or [])
        named_args = dict(getattr(node, "named_arguments", {}) or {})
        rendered_args = [self.visit(arg) for arg in positional_args]

        vector_info = self.vector_type_info(constructor_type)
        if vector_info:
            positional_args, rendered_args = self.hip_vector_constructor_arguments(
                vector_info,
                positional_args,
                rendered_args,
                named_args,
            )
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, positional_args, rendered_args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, positional_args, rendered_args
            )
            if constructor_call is not None:
                return constructor_call
            rendered_args = self.generate_vector_constructor_args(
                vector_info, positional_args, rendered_args
            )
            return f"{vector_info['constructor']}({', '.join(rendered_args)})"

        if named_args and constructor_type:
            fields = list(self.struct_member_types.get(constructor_type, {}).items())
            consumed = set()
            for field_name, field_type in fields[len(rendered_args) :]:
                if field_name not in named_args:
                    break
                consumed.add(field_name)
                rendered_args.append(
                    self.generate_expression_with_expected(
                        named_args[field_name],
                        field_type,
                    )
                )
            for field_name, value in named_args.items():
                if field_name not in consumed:
                    rendered_args.append(self.visit(value))

        half_constructor = self.hip_half_constructor_expression(
            constructor_type, positional_args, rendered_args
        )
        if half_constructor is not None:
            return half_constructor

        scalar_alias_constructor = self.hip_scalar_alias_constructor_expression(
            constructor_type, rendered_args
        )
        if scalar_alias_constructor is not None:
            return scalar_alias_constructor

        mapped_type = self.map_type(constructor_type) if constructor_type else ""
        if not mapped_type:
            return "{" + ", ".join(rendered_args) + "}"
        return f"{mapped_type}{{{', '.join(rendered_args)}}}"

    def hip_scalar_alias_constructor_expression(self, constructor_type, args):
        """Lower shader scalar aliases to valid HIP/C++ casts."""
        if constructor_type not in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES:
            return None
        if len(args) > 1:
            return None

        mapped_type = self.map_type(constructor_type)
        value = args[0] if args else "0"
        return f"static_cast<{mapped_type}>({value})"

    def hip_half_constructor_expression(self, constructor_type, raw_args, args):
        """Lower CrossGL FP16 constructors to HIP's documented half intrinsics."""
        unsupported_type = self.hip_unsupported_fp16_vector_type(constructor_type)
        if unsupported_type is not None:
            self.raise_unsupported_hip_fp16_vector_type(unsupported_type)

        if constructor_type in {"f16", "half", "float16"}:
            if not args:
                return "half{}"
            return f"__float2half({args[0]})"

        if constructor_type not in {"vec2<f16>", "half2", "f16vec2"}:
            return None

        if not args:
            return "__float2half2_rn(0.0f)"

        if len(args) == 1:
            arg_type = self.type_name_string(self.expression_result_type(raw_args[0]))
            if arg_type in {"vec2<f16>", "half2", "f16vec2"}:
                return args[0]
            return f"__float2half2_rn({args[0]})"

        return f"__floats2half2_rn({args[0]}, {args[1]})"

    def hip_vector_constructor_arguments(
        self,
        vector_info,
        positional_args,
        positional_exprs,
        named_args,
    ):
        if not named_args:
            return positional_args, positional_exprs

        components = vector_info["components"]
        if all(field_name in components for field_name in named_args):
            ordered_named_args = [
                named_args[component]
                for component in components
                if component in named_args
            ]
            ordered_named_exprs = [self.visit(arg) for arg in ordered_named_args]
            return (
                positional_args + ordered_named_args,
                positional_exprs + ordered_named_exprs,
            )

        ordered_named_args = list(named_args.values())
        ordered_named_exprs = [self.visit(arg) for arg in ordered_named_args]
        return (
            ordered_named_args + positional_args,
            ordered_named_exprs + positional_exprs,
        )

    def visit_FunctionCallNode(self, node) -> str:
        func_expr = (
            node.function if hasattr(node, "function") else getattr(node, "name", None)
        )
        func_name = None
        if isinstance(func_expr, MemberAccessNode):
            callee = self.visit(func_expr)
            func_name = callee
        elif hasattr(func_expr, "name") and getattr(func_expr, "name", None):
            func_name = func_expr.name
            callee = func_name
        elif isinstance(func_expr, str):
            func_name = func_expr
            callee = func_expr
        else:
            callee = self.visit(func_expr)
        raw_args = getattr(node, "args", getattr(node, "arguments", []))

        enum_constructor = generate_enum_constructor_call(self, func_name, raw_args)
        if enum_constructor is not None:
            return enum_constructor

        args = [self.visit(arg) for arg in raw_args]

        half_constructor = self.hip_half_constructor_expression(
            func_name, raw_args, args
        )
        if half_constructor is not None:
            return half_constructor

        if func_name == "lambda":
            return self.generate_lambda_expression(raw_args)
        if func_name in HIP_WAVE_OP_ARITIES:
            return self.generate_wave_operation(func_name, raw_args, args)
        if func_name in {"SetMeshOutputCounts", "DispatchMesh"}:
            return self.generate_mesh_task_call_expression(func_name, raw_args, args)

        is_user_function = self.is_user_defined_function(func_name)
        if not is_user_function:
            scalar_alias_constructor = self.hip_scalar_alias_constructor_expression(
                func_name, args
            )
            if scalar_alias_constructor is not None:
                return scalar_alias_constructor
        if not is_user_function:
            warp_sync_builtin_call = self.generate_warp_sync_builtin_call(
                func_name, raw_args, args
            )
            if warp_sync_builtin_call is not None:
                return warp_sync_builtin_call
        if not is_user_function:
            ray_tracing_call = self.generate_ray_tracing_call_expression(
                func_name, raw_args
            )
            if ray_tracing_call is not None:
                return ray_tracing_call
        if not is_user_function:
            buffer_call = self.generate_buffer_call(
                func_expr, func_name, raw_args, args
            )
            if buffer_call is not None:
                return buffer_call

            structured_atomic_call = self.generate_structured_buffer_atomic_call(
                func_name, raw_args, args
            )
            if structured_atomic_call is not None:
                return structured_atomic_call

            plain_atomic_call = self.generate_plain_atomic_call(
                func_name, raw_args, args
            )
            if plain_atomic_call is not None:
                return plain_atomic_call

            resource_call = self.generate_resource_call(func_name, raw_args, args)
            if resource_call is not None:
                return resource_call

            bitcast_call = self.generate_hip_bitcast_call(func_name, raw_args, args)
            if bitcast_call is not None:
                return bitcast_call

            integer_bit_call = self.generate_hip_integer_bit_call(
                func_name, raw_args, args
            )
            if integer_bit_call is not None:
                return integer_bit_call

        if is_user_function:
            args = self.hip_user_function_call_arguments(func_name, raw_args, args)
            return f"{callee}({', '.join(args)})"

        if func_name in self.synchronization_builtins and raw_args:
            raise ValueError(
                f"HIP synchronization builtin '{func_name}' requires 0 "
                f"arguments; got {len(raw_args)}"
            )

        args = self.query_metadata_call_arguments(func_name, raw_args, args)

        mapped_name = self.function_map.get(func_name, func_name)

        if func_name == "abs":
            if len(args) == 1:
                abs_call = self.generate_abs_call(raw_args, args)
                if abs_call is not None:
                    return abs_call
        elif func_name == "sign":
            if len(args) == 1:
                sign_call = self.generate_sign_call(raw_args, args)
                if sign_call is not None:
                    return sign_call
        elif func_name == "mod":
            if len(args) == 2:
                mod_call = self.generate_mod_call(raw_args, args)
                if mod_call is not None:
                    return mod_call
        elif func_name in {"fract", "frac"}:
            if len(args) == 1:
                return self.generate_fract_call(raw_args, args)
        elif func_name in {"min", "max"}:
            if len(args) == 2:
                min_max_call = self.generate_min_max_call(func_name, raw_args, args)
                if min_max_call is not None:
                    return min_max_call
        elif func_name in {"atan", "atan2"}:
            if len(args) == 2:
                atan2_call = self.generate_atan2_call(raw_args, args)
                if atan2_call is not None:
                    return atan2_call
                if func_name == "atan":
                    scalar_atan2_call = self.generate_scalar_atan2_call(raw_args, args)
                    if scalar_atan2_call is not None:
                        return scalar_atan2_call
        elif func_name in {"fma", "mad"}:
            if len(args) == 3:
                result_type = self.fused_multiply_add_result_type(raw_args)
                fma_call = self.generate_fused_multiply_add_call(raw_args, args)
                if fma_call is not None:
                    if result_type is not None:
                        node.expression_type = result_type
                    return fma_call
        elif func_name in {"dot", "cross", "length", "normalize"}:
            geometric_call = self.generate_vector_geometric_call(
                func_name,
                raw_args,
                args,
            )
            if geometric_call is not None:
                return geometric_call
        elif func_name == "clamp":
            if len(args) == 3:
                return self.generate_clamp_call(raw_args, args)
        elif func_name in {"mix", "lerp"}:
            if len(args) == 3:
                mix_call = self.generate_mix_call(raw_args, args)
                if mix_call is not None:
                    return mix_call
        elif func_name == "saturate":
            if len(args) == 1:
                saturate_call = self.generate_saturate_call(raw_args, args)
                if saturate_call is not None:
                    return saturate_call
        elif func_name == "step":
            if len(args) == 2:
                step_call = self.generate_step_call(raw_args, args)
                if step_call is not None:
                    return step_call
        elif func_name == "smoothstep":
            if len(args) == 3:
                smoothstep_call = self.generate_smoothstep_call(raw_args, args)
                if smoothstep_call is not None:
                    return smoothstep_call
        elif func_name in ["texture", "tex2D"]:
            if len(args) >= 2:
                return (
                    f"tex2D<float4>({args[0]}, "
                    f"{self.coord_component(args[1], 'x')}, "
                    f"{self.coord_component(args[1], 'y')})"
                )

        math_func_name = self.hip_math_function_name(func_name)
        vector_math_call = self.generate_vector_scalar_math_call(
            math_func_name,
            raw_args,
            args,
        )
        if vector_math_call is not None:
            return vector_math_call

        vector_info = self.vector_type_info(func_name)
        if vector_info:
            splat_call = self.generate_vector_scalar_splat_call(
                vector_info, raw_args, args
            )
            if splat_call is not None:
                return splat_call
            constructor_call = self.generate_vector_constructor_single_eval_call(
                vector_info, raw_args, args
            )
            if constructor_call is not None:
                return constructor_call
            args = self.generate_vector_constructor_args(vector_info, raw_args, args)

        scalar_math_call = self.generate_scalar_math_call(
            math_func_name, raw_args, args
        )
        if scalar_math_call is not None:
            return scalar_math_call

        args_str = ", ".join(args)
        target = mapped_name if mapped_name is not None else callee
        return f"{target}({args_str})"

    def hip_math_function_name(self, func_name):
        return {"inverseSqrt": "inversesqrt", "rsqrt": "inversesqrt"}.get(
            func_name, func_name
        )

    def normalize_hip_integer_bit_function(self, func_name):
        return HIP_INTEGER_BIT_FUNCTION_ALIASES.get(func_name, func_name)

    def hip_bitcast_result_type(self, func_name, raw_args):
        target_component = HIP_BITCAST_FUNCTION_TARGETS.get(func_name)
        if (
            target_component is None
            or self.is_user_defined_function(func_name)
            or len(raw_args or []) != 1
        ):
            return None

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            if source_info["component_type"] not in {"float", "int", "uint"}:
                return None
            return self.vector_type_for_components(
                target_component,
                len(source_info["components"]),
            )

        source_component = self.scalar_component_type(source_type)
        if source_component in {"float", "int", "uint"}:
            return target_component
        return None

    def generate_hip_bitcast_call(self, func_name, raw_args, args):
        result_type = self.hip_bitcast_result_type(func_name, raw_args)
        if result_type is None:
            return None

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        result_info = self.vector_type_info(result_type)
        if source_info is not None and result_info is not None:
            helper_name = self.require_hip_vector_bitcast_helper(
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        target_component = HIP_BITCAST_FUNCTION_TARGETS[func_name]
        source_component = self.scalar_component_type(source_type)
        return self.format_hip_scalar_bitcast(
            source_component,
            target_component,
            args[0],
        )

    def hip_integer_bit_result_type(self, func_name, raw_args):
        original_func_name = func_name
        func_name = self.normalize_hip_integer_bit_function(func_name)
        if func_name not in {
            "bitCount",
            "bitfieldReverse",
            "findLSB",
            "findMSB",
        } or self.is_user_defined_function(original_func_name):
            return None

        if len(raw_args or []) != 1:
            return "uint"

        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        if source_info is not None:
            if source_info["component_type"] not in {"int", "uint"}:
                return self.vector_type_for_components(
                    "uint",
                    len(source_info["components"]),
                )
            if func_name == "bitCount":
                return self.vector_type_for_components(
                    "uint",
                    len(source_info["components"]),
                )
            return source_type

        source_component = self.scalar_component_type(source_type)
        if source_component in {"int", "uint"}:
            if func_name == "bitCount":
                return "uint"
            if func_name == "bitfieldReverse":
                return source_type or source_component
            return source_component
        return "uint"

    def generate_hip_integer_bit_call(self, func_name, raw_args, args):
        result_type = self.hip_integer_bit_result_type(func_name, raw_args)
        if result_type is None:
            return None
        if len(raw_args or []) != 1 or len(args) != 1:
            return None

        operation = self.normalize_hip_integer_bit_function(func_name)
        source_type = self.expression_result_type(raw_args[0])
        source_info = self.vector_type_info(source_type)
        result_info = self.vector_type_info(result_type)
        if source_info is not None:
            if result_info is None or source_info["component_type"] not in {
                "int",
                "uint",
            }:
                return None
            helper_name = self.require_hip_integer_bit_vector_helper(
                operation,
                source_info,
                result_info,
            )
            return f"{helper_name}({args[0]})"

        source_component = self.scalar_component_type(source_type)
        if source_component not in {"int", "uint", None}:
            return None
        return self.format_hip_integer_bit_component(
            operation,
            args[0],
            source_component,
            self.scalar_component_type(result_type),
            source_type,
            result_type,
        )

    def require_hip_integer_bit_vector_helper(
        self,
        operation,
        source_info,
        result_info,
    ):
        helper_name = self.sanitize_helper_name(
            f"cgl_{source_info['type']}_{operation}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_hip_integer_bit_component(
                operation,
                f"value.{component}",
                source_info["component_type"],
                result_info["component_type"],
            )
            for component in source_info["components"]
        ]
        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({source_info['type']} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_hip_integer_bit_component(
        self,
        operation,
        value,
        source_component=None,
        result_component=None,
        source_type=None,
        result_type=None,
    ):
        bit_width = self.hip_integer_bit_width(source_type)
        unsigned_cast_type = "unsigned long long" if bit_width == 64 else "unsigned int"
        signed_cast_type = "long long" if bit_width == 64 else "int"
        popc_intrinsic = "__popcll" if bit_width == 64 else "__popc"
        ffs_intrinsic = "__ffsll" if bit_width == 64 else "__ffs"
        clz_intrinsic = "__clzll" if bit_width == 64 else "__clz"
        high_bit_index = 63 if bit_width == 64 else 31
        unsigned_value = (
            f"static_cast<{unsigned_cast_type}>({value})"
            if source_component == "int"
            else value
        )
        if operation == "bitCount":
            return f"{popc_intrinsic}({unsigned_value})"
        if operation == "bitfieldReverse":
            intrinsic_name = "__brevll" if bit_width == 64 else "__brev"
            reversed_value = f"{intrinsic_name}({unsigned_value})"
            if result_component == "int":
                return f"static_cast<{signed_cast_type}>({reversed_value})"
            return reversed_value
        if operation == "findLSB":
            return f"({ffs_intrinsic}({value}) - 1)"
        if operation == "findMSB":
            clz_value = (
                f"static_cast<{unsigned_cast_type}>({value})"
                if bit_width == 64 and source_component == "int"
                else value
            )
            if source_component == "int":
                inverted_value = (
                    f"static_cast<{unsigned_cast_type}>(~({value}))"
                    if bit_width == 64
                    else f"~({value})"
                )
                return (
                    f"(({value}) < 0 ? ({high_bit_index} - "
                    f"{clz_intrinsic}({inverted_value})) : "
                    f"(({value}) == 0 ? -1 : ({high_bit_index} - "
                    f"{clz_intrinsic}({clz_value}))))"
                )
            return (
                f"(({value}) == 0 ? -1 : "
                f"({high_bit_index} - {clz_intrinsic}({value})))"
            )
        return f"{operation}({value})"

    def hip_integer_bit_width(self, type_name):
        if type_name is None:
            return 32
        type_text = self.type_name_string(type_name)
        mapped_type = self.map_type(type_text)
        if type_text in {"i64", "int64", "int64_t", "u64", "uint64", "uint64_t"}:
            return 64
        if mapped_type in {
            "long long",
            "long long int",
            "unsigned long long",
            "unsigned long long int",
        }:
            return 64
        return 32

    def require_hip_vector_bitcast_helper(self, source_info, result_info):
        helper_name = self.sanitize_helper_name(
            f"cgl_{source_info['type']}_to_{result_info['type']}_bitcast"
        )
        if helper_name in self.helper_functions:
            return helper_name

        components = [
            self.format_hip_scalar_bitcast(
                source_info["component_type"],
                result_info["component_type"],
                f"value.{component}",
            )
            for component in source_info["components"]
        ]
        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({source_info['type']} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_hip_scalar_bitcast(self, source_component, target_component, value):
        if source_component == target_component:
            return value
        if source_component == "float" and target_component == "int":
            return f"__float_as_int({value})"
        if source_component == "float" and target_component == "uint":
            return f"__float_as_uint({value})"
        if source_component == "int" and target_component == "float":
            return f"__int_as_float({value})"
        if source_component == "uint" and target_component == "float":
            return f"__uint_as_float({value})"
        if source_component == "int" and target_component == "uint":
            return f"static_cast<unsigned int>({value})"
        if source_component == "uint" and target_component == "int":
            return f"static_cast<int>({value})"
        return value

    def generate_warp_sync_builtin_call(self, func_name, raw_args, args):
        expected_counts = HIP_WARP_SYNC_BUILTIN_ARITIES.get(func_name)
        if expected_counts is None:
            return None
        if len(raw_args) not in expected_counts:
            expected_list = " or ".join(str(count) for count in expected_counts)
            raise ValueError(
                f"HIP warp sync builtin {func_name} requires {expected_list} "
                f"arguments, got {len(raw_args)}"
            )
        if func_name == "__shfl_down_sync":
            helper_name = self.require_hip_shfl_down_sync_helper(
                raw_args[1],
                has_width=len(raw_args) == 4,
            )
            return f"{helper_name}({', '.join(args)})"
        return None

    def require_hip_shfl_down_sync_helper(self, value_arg, has_width=False):
        value_type = self.map_type(self.expression_result_type(value_arg) or "uint")
        width_suffix = "_width" if has_width else ""
        helper_name = (
            f"cgl_hip_shfl_down_sync{width_suffix}_"
            f"{self.wave_type_suffix(value_type)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        width_param = ", int width" if has_width else ""
        width_arg = ", width" if has_width else ""
        width_ignore = "    (void)width;\n" if has_width else ""
        helper = (
            f"__device__ inline {value_type} {helper_name}"
            f"(unsigned long long mask, {value_type} value, int offset{width_param})\n"
            "{\n"
            "#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_ENABLE_WARP_SYNC_BUILTINS)\n"
            f"    return __shfl_down_sync(mask, value, offset{width_arg});\n"
            "#else\n"
            "    (void)mask;\n"
            "    (void)offset;\n"
            f"{width_ignore}"
            "    return value;\n"
            "#endif\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def visit_MeshOpNode(self, node: MeshOpNode) -> str:
        operation = getattr(node, "operation", "")
        raw_args = getattr(node, "arguments", []) or []
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_mesh_task_call_expression(operation, raw_args, args)

    def generate_mesh_task_call_expression(self, operation, raw_args, args):
        stage_name = getattr(self, "current_stage_name", None)
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise ValueError("HIP SetMeshOutputCounts is only valid in mesh stages")
            if len(raw_args) != 2:
                raise ValueError(
                    "HIP SetMeshOutputCounts requires exactly two arguments"
                )
            return f"cgl_hip_set_mesh_output_counts({', '.join(args)})"

        if operation == "DispatchMesh":
            if stage_name not in {"task", "object", "amplification"}:
                raise ValueError(
                    "HIP DispatchMesh is only valid in task, object, or "
                    "amplification stages"
                )
            if len(raw_args) not in {3, 4}:
                raise ValueError(
                    "HIP DispatchMesh requires exactly three arguments, or four "
                    "with a task payload"
                )
            return f"cgl_hip_dispatch_mesh({', '.join(args)})"

        return None

    def visit_RayTracingOpNode(self, node: RayTracingOpNode) -> str:
        operation = getattr(node, "operation", "")
        if operation in self.function_return_types:
            args = ", ".join(
                self.visit(arg) for arg in getattr(node, "arguments", []) or []
            )
            return f"{operation}({args})"
        return self.generate_ray_tracing_call_expression(
            operation,
            getattr(node, "arguments", []),
        )

    def generate_ray_tracing_call_expression(self, operation, arguments):
        if operation not in {
            "TraceRay",
            "CallShader",
            "ReportHit",
            "IgnoreHit",
            "AcceptHitAndEndSearch",
        }:
            return None

        self.require_hip_ray_runtime_helpers()
        generated_args = [self.visit(arg) for arg in arguments or []]
        if operation == "TraceRay" and len(generated_args) == 11:
            ray_desc = (
                "CglRayDesc("
                f"{generated_args[6]}, {generated_args[7]}, "
                f"{generated_args[8]}, {generated_args[9]}"
                ")"
            )
            generated_args = generated_args[:6] + [ray_desc, generated_args[10]]
        elif operation == "ReportHit" and len(generated_args) == 2:
            generated_args.append("CglBuiltInTriangleIntersectionAttributes{}")

        helper_name = self.function_map.get(operation, operation)
        return f"{helper_name}({', '.join(generated_args)})"

    def visit_RayQueryOpNode(self, node: RayQueryOpNode) -> str:
        operation = getattr(node, "operation", "")
        helper_name = self.require_hip_ray_query_helper(operation)
        args = [self.visit(getattr(node, "query_expr", None))]
        args.extend(self.visit(arg) for arg in getattr(node, "arguments", []) or [])
        return f"{helper_name}({', '.join(args)})"

    def visit_WaveOpNode(self, node: WaveOpNode) -> str:
        raw_args = list(getattr(node, "arguments", []) or [])
        args = [self.visit(arg) for arg in raw_args]
        return self.generate_wave_operation(
            getattr(node, "operation", "WaveOp"), raw_args, args
        )

    def generate_wave_operation(self, operation, raw_args, args):
        expected_count = HIP_WAVE_OP_ARITIES.get(operation)
        if expected_count is None:
            raise ValueError(f"Unsupported HIP wave intrinsic {operation}")
        if len(raw_args) != expected_count:
            raise ValueError(
                f"HIP wave intrinsic {operation} requires {expected_count} "
                f"argument{'s' if expected_count != 1 else ''}, got {len(raw_args)}"
            )

        if operation == "WaveGetLaneCount":
            return "warpSize"
        if operation == "WaveGetLaneIndex":
            return "(threadIdx.x & (warpSize - 1))"
        if operation == "WaveIsFirstLane":
            return "((threadIdx.x & (warpSize - 1)) == 0)"

        diagnostic = self.wave_operation_diagnostic(operation, raw_args)
        if diagnostic is not None:
            return diagnostic

        helper_name = self.require_wave_helper(operation, raw_args)
        return f"{helper_name}({', '.join(args)})"

    def wave_operation_diagnostic(self, operation, raw_args):
        reason = self.unsupported_wave_operation_reason(operation, raw_args)
        if reason is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.wave_result_type(operation, raw_args)
        )
        return (
            f"/* unsupported {self.resource_backend_name()} wave intrinsic: "
            f"{operation} {reason} */ {fallback}"
        )

    def unsupported_wave_operation_reason(self, operation, raw_args):
        if not raw_args:
            return None

        if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
            predicate_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_is_scalar_component_type(predicate_type, {"bool"}):
                return f"predicate must be bool, got {predicate_type}"

        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"} and len(raw_args) > 1:
            lane_type = self.wave_diagnostic_type_name(raw_args[1])
            if not self.wave_is_scalar_component_type(lane_type, {"int", "uint"}):
                return f"lane index must be scalar int or uint, got {lane_type}"

        if operation.startswith("WaveMultiPrefix") and len(raw_args) > 1:
            mask_type = self.wave_diagnostic_type_name(raw_args[1])
            if self.map_type(mask_type) != "uint4":
                return f"partition mask must be uvec4, got {mask_type}"

        if operation in {
            "WaveActiveSum",
            "WaveActiveProduct",
            "WaveActiveMin",
            "WaveActiveMax",
            "WavePrefixSum",
            "WavePrefixProduct",
            "WaveMultiPrefixSum",
            "WaveMultiPrefixProduct",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"float", "double", "int", "uint"}
            ):
                return "value must be numeric scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveMultiPrefixBitAnd",
            "WaveMultiPrefixBitOr",
            "WaveMultiPrefixBitXor",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"int", "uint"}
            ):
                return "value must be integer scalar or vector, " f"got {value_type}"

        if operation in {
            "WaveActiveAllEqual",
            "WaveMatch",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "QuadReadLaneAt",
            "QuadReadAcrossX",
            "QuadReadAcrossY",
            "QuadReadAcrossDiagonal",
        }:
            value_type = self.wave_diagnostic_type_name(raw_args[0])
            if not self.wave_has_scalar_or_vector_component(
                value_type, {"bool", "float", "double", "int", "uint"}
            ):
                return "value must be scalar or vector, " f"got {value_type}"

        return None

    def wave_diagnostic_type_name(self, arg):
        type_name = self.expression_result_type(arg)
        if type_name is None:
            return "unknown"
        return self.map_type(type_name)

    def wave_is_scalar_component_type(self, type_name, components):
        if self.vector_type_info(type_name) is not None:
            return False
        return self.scalar_component_type(type_name) in components

    def wave_has_scalar_or_vector_component(self, type_name, components):
        vector_info = self.vector_type_info(type_name)
        if vector_info is not None:
            return vector_info["component_type"] in components
        return self.scalar_component_type(type_name) in components

    def require_wave_helper(self, operation, raw_args):
        result_type = self.map_type(self.wave_result_type(operation, raw_args))
        arg_types = [
            self.wave_argument_type(operation, index, arg)
            for index, arg in enumerate(raw_args)
        ]
        helper_name = self.wave_helper_name(operation, result_type, arg_types)
        if helper_name in self.helper_functions:
            return helper_name

        parameter_names = self.wave_helper_parameter_names(operation, len(arg_types))
        params = [
            f"{arg_type} {parameter_name}"
            for parameter_name, arg_type in zip(parameter_names, arg_types)
        ]
        helper = (
            f"__device__ inline {result_type} {helper_name}({', '.join(params)})\n"
            "{\n"
            f"    return {self.wave_helper_return_expression(operation, result_type)};\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def wave_result_type(self, operation, raw_args):
        if operation in HIP_WAVE_UINT_RESULT_OPS:
            return "uint"
        if operation in HIP_WAVE_BOOL_RESULT_OPS:
            return "bool"
        if operation in HIP_WAVE_UVEC4_RESULT_OPS:
            return "uvec4"
        if raw_args:
            return self.expression_result_type(raw_args[0]) or "uint"
        return "uint"

    def wave_argument_type(self, operation, index, arg):
        if index == 0 and operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
            return self.map_type("bool")
        if index == 1 and operation.startswith("WaveMultiPrefix"):
            return self.map_type("uvec4")
        if index == 1 and operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return self.map_type("uint")
        return self.map_type(self.expression_result_type(arg) or "uint")

    def wave_helper_name(self, operation, result_type, arg_types):
        suffix_parts = [self.wave_type_suffix(result_type)]
        suffix_parts.extend(self.wave_type_suffix(arg_type) for arg_type in arg_types)
        return (
            f"cgl_hip_{self.wave_operation_suffix(operation)}_{'_'.join(suffix_parts)}"
        )

    def wave_operation_suffix(self, operation):
        name = operation[0].lower() + operation[1:]
        suffix = []
        for char in name:
            if char.isupper():
                suffix.append("_")
                suffix.append(char.lower())
            else:
                suffix.append(char)
        return "".join(suffix)

    def wave_type_suffix(self, type_name):
        suffix = type_name.replace("unsigned int", "uint")
        suffix = suffix.replace(" ", "_").replace("*", "_ptr")
        suffix = suffix.replace("&", "_ref").replace("::", "_")
        return "".join(char if char.isalnum() else "_" for char in suffix).strip("_")

    def wave_helper_parameter_names(self, operation, arg_count):
        if arg_count == 1:
            if operation in HIP_WAVE_PREDICATE_ARGUMENT_OPS:
                return ["predicate"]
            return ["value"]
        if operation in {"WaveReadLaneAt", "QuadReadLaneAt"}:
            return ["value", "lane"]
        if operation == "WaveMultiPrefixCountBits":
            return ["predicate", "mask"]
        return ["value", "mask"]

    def wave_helper_return_expression(self, operation, result_type):
        if operation in {"WaveActiveAllTrue", "WaveActiveAnyTrue"}:
            return "predicate"
        if operation == "WaveActiveAllEqual":
            return "true"
        if operation in {
            "WaveActiveCountBits",
            "WavePrefixCountBits",
            "WaveMultiPrefixCountBits",
        }:
            return "(predicate ? 1u : 0u)"
        if operation == "WaveActiveBallot":
            return "make_uint4((predicate ? 1u : 0u), 0u, 0u, 0u)"
        if operation == "WaveMatch":
            return "make_uint4(0u, 0u, 0u, 0u)"
        if result_type == "bool":
            return "false"
        return "value"

    def generate_lambda_expression(self, args):
        """Render CrossGL's pseudo-lambda as a HIP device lambda."""
        if not args:
            return "[&] __device__ () {}"

        params = ", ".join(self.generate_lambda_parameter(arg) for arg in args[:-1])
        body = self.generate_lambda_body(args[-1])
        return f"[&] __device__ ({params}) {body}"

    def generate_lambda_parameter(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        typed_param = self.split_lambda_typed_parameter(raw)
        if typed_param is None:
            param_name = self.lambda_fallback_parameter_name(raw)
            return f"auto {param_name}" if param_name else "auto"

        type_name, param_name = typed_param
        mapped_type = self.lambda_parameter_type(type_name)
        return f"{mapped_type} {param_name}"

    def generate_lambda_body(self, arg):
        raw = self.lambda_raw_argument_text(arg).strip()
        if raw.startswith("{") and raw.endswith("}"):
            return raw
        if raw:
            return f"{{ return {raw}; }}"
        return "{}"

    def lambda_raw_argument_text(self, arg):
        if isinstance(arg, IdentifierNode):
            return arg.name
        if isinstance(arg, str):
            return arg
        return self.visit(arg)

    def split_lambda_typed_parameter(self, raw):
        if not raw:
            return None
        if any(char in raw for char in "{}()"):
            return None
        parts = raw.rsplit(None, 1)
        if len(parts) != 2:
            return None
        type_name, param_name = parts
        if not param_name.isidentifier():
            return None
        return type_name, param_name

    def lambda_parameter_type(self, type_name):
        if "<" in type_name or ">" in type_name:
            return "auto"
        return self.map_type(type_name)

    def lambda_fallback_parameter_name(self, raw):
        if not raw:
            return ""
        candidate = raw.rsplit(None, 1)[-1]
        if candidate.isidentifier():
            return candidate
        return raw

    def generate_vector_constructor_args(self, vector_info, raw_args, args):
        """Flatten vector arguments passed to HIP make_* constructors."""
        if len(args) == 1:
            arg_type = self.expression_result_type(raw_args[0])
            if arg_type is not None and not self.vector_type_info(arg_type):
                return args * len(vector_info["components"])

        generated_args = []
        for raw_arg, arg_expr in zip(raw_args, args):
            swizzle_components = self.member_swizzle_components(raw_arg)
            if swizzle_components is not None:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        vector_info,
                    )
                )
                if len(generated_args) >= len(vector_info["components"]):
                    return generated_args[: len(vector_info["components"])]
                continue

            arg_info = self.vector_type_info(self.expression_result_type(raw_arg))
            if arg_info is None:
                generated_args.append(arg_expr)
            else:
                generated_args.extend(
                    self.vector_argument_lane_expressions(
                        raw_arg,
                        arg_expr,
                        arg_info,
                    )
                )

            if len(generated_args) >= len(vector_info["components"]):
                return generated_args[: len(vector_info["components"])]

        return generated_args

    def vector_argument_lane_expressions(self, raw_arg, arg_expr, arg_info):
        swizzle_components = self.member_swizzle_components(raw_arg)
        if swizzle_components is not None:
            object_node = getattr(
                raw_arg,
                "object_expr",
                getattr(raw_arg, "object", None),
            )
            role = self.hip_compute_builtin_role_for_name(
                getattr(object_node, "name", None)
            )
            if role is not None:
                lanes = [
                    self.hip_compute_builtin_expression(role, component)
                    for component in swizzle_components
                ]
                if all(lane is not None for lane in lanes):
                    return lanes
            object_expr = self.visit(object_node)
            return [f"{object_expr}.{component}" for component in swizzle_components]

        return [f"{arg_expr}.{component}" for component in arg_info["components"]]

    def member_swizzle_components(self, node):
        if not isinstance(node, MemberAccessNode):
            return None

        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        return components

    def generate_mod_call(self, raw_args, args):
        """Lower CrossGL/GLSL mod with GLSL floor semantics."""
        if len(raw_args) != 2 or len(args) != 2:
            return None

        vector_call = self.lower_vector_binary_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            "%",
        )
        if vector_call is not None:
            return vector_call

        return self.lower_scalar_modulo_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
        )

    def lower_scalar_modulo_operation(
        self,
        left_node,
        left_expr,
        right_node,
        right_expr,
    ):
        """Lower floating-point scalar modulo with GLSL floor semantics."""
        scalar_type = self.hip_mod_scalar_type(left_node, right_node)
        if scalar_type is None:
            return None

        helper_name = self.require_hip_scalar_mod_helper(scalar_type)
        return f"{helper_name}({left_expr}, {right_expr})"

    def format_vector_binary_component(self, component_type, operator, left, right):
        if operator == "%" and component_type in {"float", "double"}:
            helper_name = self.require_hip_scalar_mod_helper(component_type)
            return f"{helper_name}({left}, {right})"
        return super().format_vector_binary_component(
            component_type,
            operator,
            left,
            right,
        )

    def hip_mod_scalar_type(self, left_node, right_node):
        component_types = []
        for node in (left_node, right_node):
            arg_type = self.expression_result_type(node)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type is not None:
                component_types.append(component_type)

        if "double" in component_types:
            return "double"
        if "float" in component_types:
            return "float"
        return None

    def require_hip_scalar_mod_helper(self, scalar_type):
        helper_name = f"cgl_mod_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}"
            f"({scalar_type} lhs, {scalar_type} rhs)\n"
            "{\n"
            f"    return lhs - rhs * {floor_name}(lhs / rhs);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_fract_call(self, raw_args, args):
        arg_type = self.expression_result_type(raw_args[0])
        vector_info = self.vector_type_info(arg_type)
        if vector_info is not None:
            helper_name = self.require_vector_fract_helper(vector_info)
            if helper_name is not None:
                return f"{helper_name}({args[0]})"

        scalar_type = self.fract_scalar_type(arg_type)
        helper_name = self.require_scalar_fract_helper(scalar_type)
        return f"{helper_name}({args[0]})"

    def fract_scalar_type(self, type_name):
        if type_name is not None and not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name) if type_name is not None else None
        if mapped_type == "double" or type_name in {"double", "f64"}:
            return "double"
        return "float"

    def require_scalar_fract_helper(self, scalar_type):
        helper_name = f"cgl_fract_{scalar_type}"
        if helper_name in self.helper_functions:
            return helper_name

        floor_name = "floor" if scalar_type == "double" else "floorf"
        helper = (
            f"__device__ inline {scalar_type} {helper_name}({scalar_type} value)\n"
            "{\n"
            f"    return value - {floor_name}(value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_vector_fract_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_fract"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_helper_name = self.require_scalar_fract_helper(component_type)
        components = [
            f"{scalar_helper_name}(value.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}({vector_type} value)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def generate_clamp_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        if not value_info:
            scalar_type = self.clamp_scalar_type(value_type)
            scalar_call = self.generate_scalar_clamp_single_eval_call(
                scalar_type,
                raw_args,
                args,
            )
            if scalar_call is not None:
                return scalar_call
            return self.format_clamp_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        min_info = self.vector_type_info(self.expression_result_type(raw_args[1]))
        max_info = self.vector_type_info(self.expression_result_type(raw_args[2]))
        if min_info and len(min_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        if max_info and len(max_info["components"]) != len(value_info["components"]):
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"

        helper_name = self.require_vector_clamp_helper(
            value_info,
            min_is_vector=min_info is not None,
            max_is_vector=max_info is not None,
        )
        if helper_name is None:
            return f"fmaxf({args[1]}, fminf({args[2]}, {args[0]}))"
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def generate_saturate_call(self, raw_args, args):
        value_type = self.expression_result_type(raw_args[0])
        value_info = self.vector_type_info(value_type)
        component_type = (
            value_info["component_type"]
            if value_info is not None
            else self.scalar_component_type(value_type)
        )
        if component_type not in {"float", "double"}:
            return None

        float_type = PrimitiveType("float")
        return self.generate_clamp_call(
            [
                raw_args[0],
                LiteralNode(0.0, float_type),
                LiteralNode(1.0, float_type),
            ],
            [args[0], "0.0", "1.0"],
        )

    def generate_step_call(self, raw_args, args):
        edge_type = self.expression_result_type(raw_args[0])
        value_type = self.expression_result_type(raw_args[1])
        edge_info = self.vector_type_info(edge_type)
        value_info = self.vector_type_info(value_type)

        if edge_info is None and value_info is None:
            scalar_type = self.step_scalar_type(raw_args)
            if scalar_type is None:
                return None
            return self.format_step_component(scalar_type, args[0], args[1])

        result_info = value_info or edge_info
        if result_info["component_type"] not in {"float", "double"}:
            return None
        if not self.compatible_step_operand(edge_info, result_info):
            return None
        if not self.compatible_step_operand(value_info, result_info):
            return None
        if edge_info is None and not self.compatible_step_scalar(edge_type):
            return None
        if value_info is None and not self.compatible_step_scalar(value_type):
            return None

        helper_name = self.require_vector_step_helper(
            result_info,
            edge_is_vector=edge_info is not None,
            value_is_vector=value_info is not None,
        )
        return f"{helper_name}({args[0]}, {args[1]})"

    def step_scalar_type(self, raw_args):
        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)
        return "double" if "double" in component_types else "float"

    def compatible_step_operand(self, operand_info, result_info):
        if operand_info is None:
            return True
        return operand_info["component_type"] == result_info["component_type"] and len(
            operand_info["components"]
        ) == len(result_info["components"])

    def compatible_step_scalar(self, type_name):
        component_type = self.scalar_component_type(type_name)
        return component_type in {"float", "double", None}

    def require_vector_step_helper(self, result_info, edge_is_vector, value_is_vector):
        edge_shape = "vector" if edge_is_vector else "scalar"
        value_shape = "vector" if value_is_vector else "scalar"
        helper_name = (
            f"cgl_{result_info['type']}_step_{edge_shape}_edge_{value_shape}_value"
        )
        if helper_name in self.helper_functions:
            return helper_name

        scalar_type = self.vector_scalar_parameter_type(result_info)
        edge_type = result_info["type"] if edge_is_vector else scalar_type
        value_type = result_info["type"] if value_is_vector else scalar_type
        components = []
        for component in result_info["components"]:
            edge = f"edge.{component}" if edge_is_vector else "edge"
            value = f"value.{component}" if value_is_vector else "value"
            components.append(
                self.format_step_component(result_info["component_type"], edge, value)
            )

        helper = (
            f"__device__ inline {result_info['type']} {helper_name}"
            f"({edge_type} edge, {value_type} value)\n"
            "{\n"
            f"    return {result_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_step_component(self, scalar_type, edge, value):
        if scalar_type == "double":
            zero = "0.0"
            one = "1.0"
        else:
            zero = "0.0f"
            one = "1.0f"
        return f"(({value}) < ({edge}) ? {zero} : {one})"

    def generate_mix_call(self, raw_args, args):
        left_type = self.expression_result_type(raw_args[0])
        right_type = self.expression_result_type(raw_args[1])
        factor_type = self.expression_result_type(raw_args[2])
        left_info = self.vector_type_info(left_type)
        right_info = self.vector_type_info(right_type)
        factor_info = self.vector_type_info(factor_type)

        if not left_info and not right_info and not factor_info:
            scalar_bool_mix = self.lower_bool_scalar_mix_operation(
                raw_args[0],
                args[0],
                raw_args[1],
                args[1],
                raw_args[2],
                args[2],
            )
            if scalar_bool_mix is not None:
                return scalar_bool_mix
            scalar_mix_call = self.generate_scalar_mix_single_eval_call(
                raw_args,
                args,
            )
            if scalar_mix_call is not None:
                return scalar_mix_call
            return self.format_mix_component(args[0], args[1], args[2])

        if left_info is None or right_info is None:
            return None
        if (
            len(left_info["components"]) != len(right_info["components"])
            or left_info["component_type"] != right_info["component_type"]
        ):
            return None

        bool_mix = self.lower_bool_vector_mix_operation(
            raw_args[0],
            args[0],
            raw_args[1],
            args[1],
            raw_args[2],
            args[2],
            left_info,
            right_info,
            factor_info,
        )
        if bool_mix is not None:
            return bool_mix

        if factor_info is not None and (
            len(factor_info["components"]) != len(left_info["components"])
            or factor_info["component_type"] != left_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_mix_helper(
            left_info,
            factor_is_vector=factor_info is not None,
        )
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"

    def require_vector_mix_helper(self, vector_info, factor_is_vector):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        factor_shape = "vector" if factor_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_mix_{factor_shape}"
        if helper_name in self.helper_functions:
            return helper_name

        factor_type = (
            vector_type
            if factor_is_vector
            else self.vector_scalar_parameter_type(vector_info)
        )
        components = []
        for component in vector_info["components"]:
            factor_component = f"a.{component}" if factor_is_vector else "a"
            components.append(
                self.format_mix_component(
                    f"x.{component}",
                    f"y.{component}",
                    factor_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} x, {vector_type} y, {factor_type} a)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_mix_component(self, left, right, factor):
        return f"({left} + (({right} - {left}) * {factor}))"

    def generate_fused_multiply_add_call(self, raw_args, args):
        vector_infos = [
            self.vector_type_info(self.expression_result_type(raw_arg))
            for raw_arg in raw_args
        ]
        vector_info = next((info for info in vector_infos if info is not None), None)
        if vector_info is None:
            scalar_type = self.fused_multiply_add_scalar_type(raw_args)
            if scalar_type is None:
                return None
            return self.format_fused_multiply_add_component(
                scalar_type,
                args[0],
                args[1],
                args[2],
            )

        if vector_info["component_type"] not in {"float", "double"}:
            return None
        for info in vector_infos:
            if info is None:
                continue
            if (
                len(info["components"]) != len(vector_info["components"])
                or info["component_type"] != vector_info["component_type"]
            ):
                return None

        component_count = len(vector_info["components"])
        pieces = []
        for raw_arg, arg_expr, info in zip(raw_args, args, vector_infos):
            if info is None and not self.compatible_fma_scalar(
                raw_arg, vector_info["component_type"]
            ):
                return None
            pieces.append(
                self.vector_operation_piece(
                    raw_arg,
                    arg_expr,
                    info,
                    component_count,
                    vector_info["component_type"],
                )
            )

        helper_name = self.require_vector_fused_multiply_add_helper(
            vector_info,
            pieces,
        )
        return f"{helper_name}({', '.join(piece['arg_expr'] for piece in pieces)})"

    def fused_multiply_add_scalar_type(self, raw_args):
        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)
        return "double" if "double" in component_types else "float"

    def compatible_fma_scalar(self, raw_arg, vector_component_type):
        component_type = self.scalar_component_type(
            self.expression_result_type(raw_arg)
        )
        if component_type is None:
            return True
        if component_type in {"int", "uint"}:
            return True
        return component_type == vector_component_type

    def require_vector_fused_multiply_add_helper(self, vector_info, pieces):
        signature = "_".join(
            self.vector_constructor_piece_signature(piece) for piece in pieces
        )
        helper_name = self.sanitize_helper_name(
            f"cgl_{vector_info['type']}_fma_{signature}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        params = [
            f"{piece['param_type']} arg{index}" for index, piece in enumerate(pieces)
        ]
        component_args = []
        for component in vector_info["components"]:
            multiply_left = self.vector_operation_piece_param_expr(
                pieces[0], 0, component
            )
            multiply_right = self.vector_operation_piece_param_expr(
                pieces[1], 1, component
            )
            addend = self.vector_operation_piece_param_expr(pieces[2], 2, component)
            component_args.append(
                self.format_fused_multiply_add_component(
                    vector_info["component_type"],
                    multiply_left,
                    multiply_right,
                    addend,
                )
            )

        helper = (
            f"__device__ inline {vector_info['type']} {helper_name}"
            f"({', '.join(params)})\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(component_args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_fused_multiply_add_component(self, scalar_type, left, right, addend):
        target = "fma" if scalar_type == "double" else "fmaf"
        return f"{target}({left}, {right}, {addend})"

    def generate_atan2_call(self, raw_args, args):
        y_type = self.expression_result_type(raw_args[0])
        x_type = self.expression_result_type(raw_args[1])
        y_info = self.vector_type_info(y_type)
        x_info = self.vector_type_info(x_type)

        if y_info is None and x_info is None:
            return None
        if y_info is None or x_info is None:
            return None
        if (
            len(y_info["components"]) != len(x_info["components"])
            or y_info["component_type"] != x_info["component_type"]
        ):
            return None

        helper_name = self.require_vector_atan2_helper(y_info)
        if helper_name is None:
            return None
        return f"{helper_name}({args[0]}, {args[1]})"

    def generate_scalar_atan2_call(self, raw_args, args):
        component_types = []
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return None
            component_type = self.scalar_component_type(arg_type)
            if component_type not in {"float", "double", None}:
                return None
            component_types.append(component_type)

        target = "atan2" if "double" in component_types else "atan2f"
        return f"{target}({', '.join(args)})"

    def require_vector_atan2_helper(self, vector_info):
        component_type = vector_info["component_type"]
        if component_type not in {"float", "double"}:
            return None

        vector_type = vector_info["type"]
        helper_name = f"cgl_{vector_type}_atan2"
        if helper_name in self.helper_functions:
            return helper_name

        scalar_func = "atan2" if component_type == "double" else "atan2f"
        components = [
            f"{scalar_func}(y.{component}, x.{component})"
            for component in vector_info["components"]
        ]
        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} y, {vector_type} x)\n"
            "{\n"
            f"    return {vector_info['constructor']}({', '.join(components)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def clamp_scalar_type(self, type_name):
        if type_name is None:
            return "float"
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        mapped_type = self.map_type(type_name)
        if mapped_type in {"float", "double"}:
            return mapped_type
        if mapped_type in {
            "bool",
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "unsigned int",
            "long long",
            "unsigned long long",
        }:
            return mapped_type
        return "float"

    def require_vector_clamp_helper(self, vector_info, min_is_vector, max_is_vector):
        if vector_info["component_type"] == "bool":
            return None

        vector_type = vector_info["type"]
        scalar_type = self.vector_scalar_parameter_type(vector_info)
        min_shape = "vector" if min_is_vector else "scalar"
        max_shape = "vector" if max_is_vector else "scalar"
        helper_name = f"cgl_{vector_type}_clamp"
        if min_shape != "vector" or max_shape != "vector":
            helper_name += f"_{min_shape}_min_{max_shape}_max"
        if helper_name in self.helper_functions:
            return helper_name

        min_type = vector_type if min_is_vector else scalar_type
        max_type = vector_type if max_is_vector else scalar_type
        components = vector_info["components"]
        constructor = vector_info["constructor"]
        args = []
        for component in components:
            value_component = f"value.{component}"
            min_component = f"min_value.{component}" if min_is_vector else "min_value"
            max_component = f"max_value.{component}" if max_is_vector else "max_value"
            args.append(
                self.format_clamp_component(
                    vector_info["component_type"],
                    value_component,
                    min_component,
                    max_component,
                )
            )

        helper = (
            f"__device__ inline {vector_type} {helper_name}"
            f"({vector_type} value, {min_type} min_value, {max_type} max_value)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def format_clamp_component(self, component_type, value, min_value, max_value):
        if component_type == "float":
            return f"fmaxf({min_value}, fminf({max_value}, {value}))"
        if component_type == "double":
            return f"fmax({min_value}, fmin({max_value}, {value}))"
        return (
            f"(({value}) < ({min_value}) ? ({min_value}) : "
            f"(({value}) > ({max_value}) ? ({max_value}) : ({value})))"
        )

    def insert_helper_functions(self):
        if not self.helper_functions:
            return
        helpers = []
        if self.resource_query_info_required:
            helpers.extend(
                [
                    "struct CglResourceQueryInfo {",
                    "    int width;",
                    "    int height;",
                    "    int depth;",
                    "    int elements;",
                    "    int levels;",
                    "    int samples;",
                    "};",
                    "",
                ]
            )
        for helper in self.helper_functions.values():
            helpers.extend(helper.splitlines())
            helpers.append("")
        self.code_lines[5:5] = helpers

    def require_hip_ray_runtime_helpers(self):
        helper_name = "cgl_ray_runtime_helpers"
        if helper_name in self.helper_functions:
            return

        self.helper_functions[helper_name] = (
            "struct CglRayTracingAccelerationStructure\n"
            "{\n"
            "    unsigned long long handle;\n"
            "};\n"
            "\n"
            "struct CglRayDesc\n"
            "{\n"
            "    float3 origin;\n"
            "    float t_min;\n"
            "    float3 direction;\n"
            "    float t_max;\n"
            "\n"
            "    __host__ __device__ CglRayDesc()\n"
            "        : origin(make_float3(0.0f, 0.0f, 0.0f)),\n"
            "          t_min(0.0f),\n"
            "          direction(make_float3(0.0f, 0.0f, 1.0f)),\n"
            "          t_max(0.0f)\n"
            "    {\n"
            "    }\n"
            "\n"
            "    __host__ __device__ CglRayDesc(\n"
            "        float3 origin_value,\n"
            "        float t_min_value,\n"
            "        float3 direction_value,\n"
            "        float t_max_value)\n"
            "        : origin(origin_value),\n"
            "          t_min(t_min_value),\n"
            "          direction(direction_value),\n"
            "          t_max(t_max_value)\n"
            "    {\n"
            "    }\n"
            "};\n"
            "\n"
            "struct CglRayQuery\n"
            "{\n"
            "    unsigned int state;\n"
            "};\n"
            "\n"
            "struct CglBuiltInTriangleIntersectionAttributes\n"
            "{\n"
            "    float2 barycentrics;\n"
            "};\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_id()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline uint3 cgl_ray_launch_size()\n"
            "{\n"
            "    return make_uint3(0u, 0u, 0u);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_hit_t()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_hit_kind()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_world_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_origin()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float3 cgl_ray_object_direction()\n"
            "{\n"
            "    return make_float3(0.0f, 0.0f, 0.0f);\n"
            "}\n"
            "\n"
            "__device__ inline float cgl_ray_t_min()\n"
            "{\n"
            "    return 0.0f;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_incoming_flags()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_instance_custom_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "__device__ inline unsigned int cgl_ray_geometry_index()\n"
            "{\n"
            "    return 0u;\n"
            "}\n"
            "\n"
            "template <typename AS, typename Flags, typename Mask, "
            "typename HitGroup, typename Multiplier, typename Miss, "
            "typename Ray, typename Payload>\n"
            "__device__ inline void cgl_trace_ray(\n"
            "    const AS&,\n"
            "    const Flags&,\n"
            "    const Mask&,\n"
            "    const HitGroup&,\n"
            "    const Multiplier&,\n"
            "    const Miss&,\n"
            "    const Ray&,\n"
            "    const Payload&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Index, typename Data>\n"
            "__device__ inline void cgl_call_shader(const Index&, const Data&)\n"
            "{\n"
            "}\n"
            "\n"
            "template <typename Distance, typename Kind, typename Attributes>\n"
            "__device__ inline bool cgl_report_hit(\n"
            "    const Distance&,\n"
            "    const Kind&,\n"
            "    const Attributes&)\n"
            "{\n"
            "    return false;\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_ignore_hit()\n"
            "{\n"
            "}\n"
            "\n"
            "__device__ inline void cgl_accept_hit_and_end_search()\n"
            "{\n"
            "}\n"
        )

    def require_hip_ray_query_helper(self, operation):
        self.require_hip_ray_runtime_helpers()
        helper_name = f"cgl_ray_query_{self.hip_snake_case_name(operation)}"
        if helper_name in self.helper_functions:
            return helper_name

        return_type = "bool" if operation == "Proceed" else "unsigned int"
        return_value = "false" if return_type == "bool" else "0u"
        self.helper_functions[helper_name] = (
            "template <typename Query, typename... Args>\n"
            f"__device__ inline {return_type} {helper_name}(Query&, Args&&...)\n"
            "{\n"
            f"    return {return_value};\n"
            "}"
        )
        return helper_name

    def hip_snake_case_name(self, name):
        text = str(name)
        result = []
        for index, char in enumerate(text):
            if char.isupper() and index > 0:
                previous = text[index - 1]
                next_char = text[index + 1] if index + 1 < len(text) else ""
                if previous.islower() or previous.isdigit() or next_char.islower():
                    result.append("_")
            result.append(char.lower())
        return "".join(result).replace("__", "_").strip("_")

    def register_variable_type(self, name, type_name, node=None, source_node=None):
        if not name or type_name is None:
            return
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        self.variable_types[name] = type_name
        if self.is_glsl_buffer_block_node(node):
            access = self.explicit_resource_access(node) or "readwrite"
            self.glsl_buffer_block_accesses[name] = access
            self.glsl_buffer_block_layouts[name] = self.glsl_buffer_block_layout(node)
        else:
            self.glsl_buffer_block_accesses.pop(name, None)
            self.glsl_buffer_block_layouts.pop(name, None)
        if self.is_buffer_resource_type(type_name):
            access = self.explicit_resource_access(node)
            if access is None and source_node is not None:
                access = self.buffer_resource_access(source_node)
            if access is None:
                self.buffer_resource_accesses.pop(name, None)
            else:
                self.buffer_resource_accesses[name] = access
        else:
            self.buffer_resource_accesses.pop(name, None)
        if not self.is_storage_image_type(type_name):
            self.image_resource_accesses.pop(name, None)
            return

        access = self.explicit_resource_access(node)
        if access is None and source_node is not None:
            access = self.image_resource_access(source_node)
        if access is None:
            self.image_resource_accesses.pop(name, None)
        else:
            self.image_resource_accesses[name] = access

    @property
    def local_variable_types(self):
        return self.variable_types

    @local_variable_types.setter
    def local_variable_types(self, value):
        self.variable_types = value

    def next_hip_temp_variable(self, prefix):
        index = self.match_temp_variable_index
        self.match_temp_variable_index += 1
        return f"__crossgl_{prefix}_{index}"

    def get_expression_name(self, node):
        if isinstance(node, IdentifierNode):
            return node.name
        if isinstance(node, VariableNode):
            return node.name
        if isinstance(node, str):
            return node
        if isinstance(node, ArrayAccessNode):
            array_node = getattr(node, "array", getattr(node, "array_expr", None))
            return self.get_expression_name(array_node)
        return None

    def get_expression_type(self, node):
        name = self.get_expression_name(node)
        if name is None:
            return None
        return self.variable_types.get(name)

    def resource_expression_type(self, node):
        type_name = self.get_expression_type(node)
        if type_name is None:
            type_name = self.expression_result_type(node)
        if type_name is not None and not isinstance(type_name, str):
            return self.convert_type_node_to_string(type_name)
        return type_name

    def is_buffer_resource_type(self, type_name):
        """Return whether a type is a structured or byte-address buffer resource."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return False
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def buffer_resource_access(self, buffer_arg):
        """Return explicit access metadata for a HIP buffer resource expression."""
        if isinstance(buffer_arg, ArrayAccessNode):
            array_node = getattr(
                buffer_arg, "array", getattr(buffer_arg, "array_expr", None)
            )
            return self.buffer_resource_access(array_node)

        buffer_name = self.get_expression_name(buffer_arg)
        if not buffer_name:
            return None
        return self.buffer_resource_accesses.get(buffer_name)

    def map_vector_arithmetic_type(self, type_name):
        return self.map_type(type_name)

    def vector_type_info(self, type_name):
        half3_info = self.hip_half3_vector_type_info(type_name)
        if half3_info is not None:
            return half3_info
        narrow_info = self.hip_narrow_integer_vector_type_info(type_name)
        if narrow_info is not None:
            return narrow_info
        return super().vector_type_info(type_name)

    def hip_half3_vector_type_info(self, type_name):
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None
        compact_type = "".join(str(type_text).split())
        if compact_type not in HIP_FP16_VEC3_TYPES and compact_type != "cgl_half3":
            return None
        self.require_hip_half3_helper()
        return {
            "type": "cgl_half3",
            "constructor": "cgl_make_half3",
            "component_type": "half",
            "components": ("x", "y", "z"),
        }

    def hip_narrow_integer_vector_type_info(self, type_name):
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None

        compact_type = "".join(str(type_text).split())
        if (
            len(compact_type) < 8
            or not compact_type.startswith("vec")
            or compact_type[3] not in {"2", "3", "4"}
            or compact_type[4] != "<"
            or compact_type[-1] != ">"
        ):
            return None

        scalar_type = compact_type[5:-1]
        native_prefix, component_type = {
            "i8": ("char", "int"),
            "u8": ("uchar", "uint"),
            "i16": ("short", "int"),
            "u16": ("ushort", "uint"),
        }.get(scalar_type, (None, None))
        if native_prefix is None:
            return None

        size = int(compact_type[3])
        components = ("x", "y", "z", "w")[:size]
        vector_type = f"{native_prefix}{size}"
        return {
            "type": vector_type,
            "constructor": f"make_{vector_type}",
            "component_type": component_type,
            "components": components,
        }

    def hip_unsupported_fp16_vector_type(self, type_name):
        type_text = self.type_name_string(type_name)
        if type_text is None:
            return None
        compact_type = "".join(str(type_text).split())
        if compact_type in HIP_UNSUPPORTED_FP16_VECTOR_TYPES:
            return compact_type
        return None

    def raise_unsupported_hip_fp16_vector_type(self, type_name):
        raise ValueError(
            "HIP does not support FP16 vector type "
            f"{type_name}; supported FP16 HIP aliases are f16/half "
            "and vec2<f16>/half2; vec3<f16>/half3 lowers to cgl_half3"
        )

    def require_hip_half3_helper(self):
        helper_name = "cgl_half3_type"
        if helper_name in self.helper_functions:
            return
        self.helper_functions[helper_name] = (
            "struct cgl_half3\n"
            "{\n"
            "    half x;\n"
            "    half y;\n"
            "    half z;\n"
            "\n"
            "    __host__ __device__ cgl_half3()\n"
            "        : x(__float2half(0.0f)), y(__float2half(0.0f)), z(__float2half(0.0f))\n"
            "    {\n"
            "    }\n"
            "\n"
            "    __host__ __device__ cgl_half3(half x_value, half y_value, half z_value)\n"
            "        : x(x_value), y(y_value), z(z_value)\n"
            "    {\n"
            "    }\n"
            "};\n"
            "\n"
            "__host__ __device__ inline half cgl_to_half(half value)\n"
            "{\n"
            "    return value;\n"
            "}\n"
            "\n"
            "template <typename T>\n"
            "__host__ __device__ inline half cgl_to_half(T value)\n"
            "{\n"
            "    return __float2half(static_cast<float>(value));\n"
            "}\n"
            "\n"
            "__host__ __device__ inline cgl_half3 cgl_make_half3()\n"
            "{\n"
            "    return cgl_half3();\n"
            "}\n"
            "\n"
            "template <typename X, typename Y, typename Z>\n"
            "__host__ __device__ inline cgl_half3 cgl_make_half3(X x, Y y, Z z)\n"
            "{\n"
            "    return cgl_half3(cgl_to_half(x), cgl_to_half(y), cgl_to_half(z));\n"
            "}"
        )

    def require_surface_read_helper(self, helper_name):
        helpers = {
            "cgl_surf1Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf1Dread(hipSurfaceObject_t surfObj, int x)\n"
                "{\n"
                "    T value;\n"
                "    surf1Dread(&value, surfObj, x);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf1DLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surf1DLayeredread(hipSurfaceObject_t surfObj, int x, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surf1DLayeredread(&value, surfObj, x, layer);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf2Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2Dread(hipSurfaceObject_t surfObj, int x, int y)\n"
                "{\n"
                "    T value;\n"
                "    surf2Dread(&value, surfObj, x, y);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf3Dread": (
                "template <typename T>\n"
                "__device__ T cgl_surf3Dread(hipSurfaceObject_t surfObj, int x, int y, int z)\n"
                "{\n"
                "    T value;\n"
                "    surf3Dread(&value, surfObj, x, y, z);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surf2DLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surf2DLayeredread(hipSurfaceObject_t surfObj, int x, int y, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surf2DLayeredread(&value, surfObj, x, y, layer);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surfCubemapread": (
                "template <typename T>\n"
                "__device__ T cgl_surfCubemapread(hipSurfaceObject_t surfObj, int x, int y, int face)\n"
                "{\n"
                "    T value;\n"
                "    surfCubemapread(&value, surfObj, x, y, face);\n"
                "    return value;\n"
                "}"
            ),
            "cgl_surfCubemapLayeredread": (
                "template <typename T>\n"
                "__device__ T cgl_surfCubemapLayeredread(hipSurfaceObject_t surfObj, int x, int y, int face, int layer)\n"
                "{\n"
                "    T value;\n"
                "    surfCubemapLayeredread(&value, surfObj, x, y, face, layer);\n"
                "    return value;\n"
                "}"
            ),
        }
        self.require_helper_function(helper_name, helpers[helper_name])

    def vector_component_count_for_type(self, type_name):
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        scalar_types = {
            "bool",
            "double",
            "f16",
            "f32",
            "f64",
            "float",
            "half",
            "i32",
            "int",
            "u32",
            "uint",
        }
        if base_type in scalar_types:
            return 1

        vector_prefixes = (
            "bvec",
            "dvec",
            "ivec",
            "uvec",
            "vec",
            "bool",
            "double",
            "float",
            "half",
            "int",
            "uint",
        )
        for prefix in vector_prefixes:
            if not base_type.startswith(prefix):
                continue
            suffix = base_type[len(prefix) :]
            if suffix and suffix[0] in {"2", "3", "4"}:
                return int(suffix[0])
        return None

    def texel_fetch_coordinate_count(self, raw_coord):
        coord_type = self.get_expression_type(raw_coord)
        if coord_type is None:
            coord_type = self.expression_result_type(raw_coord)
        return self.vector_component_count_for_type(coord_type)

    def sampled_texture_shape_type(self, texture_type):
        """Return the float sampler shape spelling for typed sampled textures."""
        texture_type = self.resource_base_type(texture_type)
        if not isinstance(texture_type, str):
            return None
        for prefix in ("isampler", "usampler"):
            if texture_type.startswith(prefix):
                return f"sampler{texture_type[len(prefix):]}"
        return texture_type

    def sampled_texture_value_type(self, texture_type):
        """Return the HIP vector type read from a sampled texture."""
        texture_type = self.resource_base_type(texture_type)
        if isinstance(texture_type, str) and texture_type.startswith("isampler"):
            return "int4"
        if isinstance(texture_type, str) and texture_type.startswith("usampler"):
            return "uint4"
        return "float4"

    def is_sampled_texture_family_type(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        return (
            isinstance(texture_type, str)
            and texture_type != "sampler"
            and texture_type.startswith(("sampler", "isampler", "usampler"))
        )

    def expected_texel_fetch_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
        }.get(texture_type)

    def expected_texture_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 2,
            "sampler2D": 2,
            "sampler2DArray": 3,
            "sampler3D": 3,
            "samplerCube": 3,
            "samplerCubeArray": 4,
        }.get(texture_type)

    def texture_coordinate_rank_diagnostic(self, func_name, texture_type, raw_coord):
        expected_coord_count = self.expected_texture_coordinate_count(texture_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def expected_texture_gradient_coordinate_counts(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": {1},
            "sampler1DArray": {1},
            "sampler2D": {2},
            "sampler2DArray": {2},
            "sampler3D": {3, 4},
            "samplerCube": {3, 4},
            "samplerCubeArray": {3, 4},
        }.get(texture_type)

    def texture_gradient_rank_diagnostic(self, texture_type, *raw_gradients):
        expected_gradient_counts = self.expected_texture_gradient_coordinate_counts(
            texture_type
        )
        if expected_gradient_counts is None:
            return None

        for raw_gradient in raw_gradients:
            actual_gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
            if (
                actual_gradient_count is not None
                and actual_gradient_count not in expected_gradient_counts
            ):
                return self.unsupported_sampled_resource_call(
                    "textureGrad derivative rank",
                    texture_type,
                    [],
                )
        return None

    def expected_projected_texture_coordinate_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 2,
            "sampler2D": 3,
            "sampler3D": 4,
            "samplerCube": 4,
        }.get(texture_type)

    def projected_texture_coordinate_rank_diagnostic(
        self, func_name, texture_type, raw_coord
    ):
        expected_coord_count = self.expected_projected_texture_coordinate_count(
            texture_type
        )
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_sampled_resource_call(
                f"{func_name} coordinate rank",
                texture_type,
                [],
            )
        return None

    def projected_coord_args(self, texture_type, texture_name, coord):
        texture_type = self.sampled_texture_shape_type(texture_type)
        projection_component = {
            "sampler1D": "y",
            "sampler2D": "z",
            "sampler3D": "w",
            "samplerCube": "w",
        }.get(texture_type)
        if projection_component is None:
            return None

        projection = self.coord_component(coord, projection_component)
        components = ["x"]
        if texture_type in {"sampler2D", "sampler3D", "samplerCube"}:
            components.append("y")
        if texture_type in {"sampler3D", "samplerCube"}:
            components.append("z")
        projected_components = [
            f"({self.coord_component(coord, component)} / {projection})"
            for component in components
        ]
        return ", ".join([texture_name, *projected_components])

    def expected_texel_fetch_offset_count(self, texture_type):
        texture_type = self.sampled_texture_shape_type(texture_type)
        return {
            "sampler1D": 1,
            "sampler1DArray": 1,
            "sampler2D": 2,
            "sampler2DArray": 2,
            "sampler3D": 3,
        }.get(texture_type)

    def texel_fetch_offset_rank_diagnostic(self, texture_type, raw_offset):
        expected_offset_count = self.expected_texel_fetch_offset_count(texture_type)
        actual_offset_count = self.texel_fetch_coordinate_count(raw_offset)
        if (
            expected_offset_count is not None
            and actual_offset_count is not None
            and actual_offset_count != expected_offset_count
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset offset rank",
                texture_type,
                [],
            )
        return None

    def offset_coord_component(self, coord, offset, component):
        return (
            f"({self.coord_component(coord, component)} + "
            f"{self.coord_component(offset, component)})"
        )

    def hip_texture_gradient_argument(self, texture_type, raw_gradient, gradient):
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type not in {"sampler3D", "samplerCube", "samplerCubeArray"}:
            return gradient

        gradient_count = self.texel_fetch_coordinate_count(raw_gradient)
        if gradient_count == 3:
            return (
                f"make_float4({self.coord_component(gradient, 'x')}, "
                f"{self.coord_component(gradient, 'y')}, "
                f"{self.coord_component(gradient, 'z')}, 0.0f)"
            )
        return gradient

    def expected_image_coordinate_count(self, image_type):
        image_type = self.resource_base_type(image_type)
        if not isinstance(image_type, str):
            return None
        if "1DArray" in image_type:
            return 2
        if "1D" in image_type:
            return 1
        if "CubeArray" in image_type:
            return 4
        if "3D" in image_type:
            return 3
        if "Cube" in image_type:
            return 3
        if "Array" in image_type:
            return 3
        if "2D" in image_type:
            return 2
        return None

    def unsupported_image_coordinate_rank_call(self, func_name, image_type):
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported {self.resource_backend_name()} image resource call: "
            f"{func_name} coordinate rank on {image_type} */ {fallback}"
        )

    def image_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_coordinate_rank_call(func_name, image_type)
        return None

    def image_resource_access(self, image_arg):
        if isinstance(image_arg, ArrayAccessNode):
            array_node = getattr(
                image_arg, "array", getattr(image_arg, "array_expr", None)
            )
            return self.image_resource_access(array_node)

        image_name = self.get_expression_name(image_arg)
        if not image_name:
            return None
        return self.image_resource_accesses.get(image_name)

    def unsupported_image_access_call(self, func_name, image_type, reason):
        image_type = image_type or "unknown resource"
        fallback = "((void)0)"
        if func_name == "imageLoad":
            fallback = self.zero_value_for_type(self.image_value_type(image_type))
        return (
            f"/* unsupported {self.resource_backend_name()} image access: "
            f"{func_name} {reason} on {image_type} */ {fallback}"
        )

    def image_access_diagnostic(self, func_name, image_type, raw_image):
        access = self.image_resource_access(raw_image)
        if func_name == "imageLoad" and access == "writeonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires readable image resource",
            )
        if func_name == "imageStore" and access == "readonly":
            return self.unsupported_image_access_call(
                func_name,
                image_type,
                "requires writable image resource",
            )
        return None

    def unsupported_image_atomic_coordinate_rank_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} image atomic resource call: "
            f"{func_name} coordinate rank on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_coordinate_rank_diagnostic(self, func_name, image_type, raw_coord):
        expected_coord_count = self.expected_image_coordinate_count(image_type)
        actual_coord_count = self.texel_fetch_coordinate_count(raw_coord)
        if (
            expected_coord_count is not None
            and actual_coord_count is not None
            and actual_coord_count != expected_coord_count
        ):
            return self.unsupported_image_atomic_coordinate_rank_call(
                func_name, image_type
            )
        return None

    def unsupported_image_atomic_access_call(self, func_name, image_type):
        image_type = image_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} image atomic resource call: "
            f"{func_name} requires readwrite image resource on {image_type} */ "
            f"{self.image_atomic_zero_value(image_type)}"
        )

    def image_atomic_access_diagnostic(self, func_name, image_type, raw_image):
        if self.image_resource_access(raw_image) in {"readonly", "writeonly"}:
            return self.unsupported_image_atomic_access_call(func_name, image_type)
        return None

    def image_surface_x_offset(self, coord, value_type, image_type):
        if "1D" in image_type and "Array" not in image_type:
            return f"{coord} * sizeof({value_type})"
        return self.surface_x_offset(coord, value_type)

    def is_storage_image_type(self, type_name):
        if not isinstance(type_name, str):
            type_name = self.convert_type_node_to_string(type_name)
        base_type = self.resource_base_type(type_name)
        if not isinstance(base_type, str):
            return False
        image_shapes = {
            "image1D",
            "image1DArray",
            "image2D",
            "image2DArray",
            "image3D",
            "imageCube",
            "imageCubeArray",
            "image2DMS",
            "image2DMSArray",
        }
        return base_type in image_shapes or any(
            base_type == f"{prefix}{shape}"
            for prefix in ("i", "u")
            for shape in image_shapes
        )

    def unsupported_scalar_resource_query_call(self, func_name, resource_type):
        resource_type = resource_type or "unknown resource"
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ 0"
        )

    def unsupported_dimension_resource_query_call(self, func_name, resource_type):
        spec = self.dimension_query_spec(resource_type)
        if spec is None:
            return None
        return_type = self.query_return_type(spec["dimensions"])
        fallback = self.query_constructor(
            return_type,
            ["0"] * len(spec["dimensions"]),
        )
        return (
            f"/* unsupported {self.resource_backend_name()} resource query: "
            f"{func_name} on {resource_type} */ {fallback}"
        )

    def generate_dimension_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        if func_name == "textureSize" and not self.is_sampled_resource_type(
            resource_type
        ):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        if func_name == "imageSize" and not self.is_storage_image_type(resource_type):
            return self.unsupported_dimension_resource_query_call(
                func_name, resource_type
            )
        return ResourceQueryMixin.generate_dimension_query(
            self, func_name, raw_args, args
        )

    def generate_sample_count_query(self, func_name, raw_args, args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None

        expected_resource = (
            self.is_sampled_resource_type(resource_type)
            if func_name == "textureSamples"
            else self.is_storage_image_type(resource_type)
        )
        sample_count_query = None
        if expected_resource:
            sample_count_query = ResourceQueryMixin.generate_sample_count_query(
                self, func_name, raw_args, args
            )
        if sample_count_query is not None:
            return sample_count_query
        return self.unsupported_scalar_resource_query_call(func_name, resource_type)

    def generate_texture_query_levels(self, raw_args):
        if not raw_args:
            return None

        resource_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if resource_type is None:
            return None
        spec = self.dimension_query_spec(resource_type)
        if (
            not self.is_sampled_resource_type(resource_type)
            or spec is None
            or not spec["mip"]
        ):
            return self.unsupported_scalar_resource_query_call(
                "textureQueryLevels", resource_type
            )
        return ResourceQueryMixin.generate_texture_query_levels(self, raw_args)

    def is_sampler_state_resource_type(self, type_name):
        return self.resource_base_type(type_name) == "sampler"

    def has_explicit_sampler_argument(self, raw_args):
        if len(raw_args) < 3:
            return False
        return self.is_sampler_state_resource_type(
            self.resource_expression_type(raw_args[1])
        )

    def normalize_split_sampler_texture_args(self, func_name, raw_args, args):
        split_sampler_functions = {
            "texture",
            "textureLod",
            "textureGrad",
            "textureGather",
            "textureGatherOffset",
            "textureGatherOffsets",
            "textureCompare",
            "textureCompareLod",
            "textureCompareGrad",
            "textureCompareOffset",
            "textureCompareLodOffset",
            "textureCompareGradOffset",
            "textureCompareProj",
            "textureCompareProjOffset",
            "textureCompareProjLod",
            "textureCompareProjLodOffset",
            "textureCompareProjGrad",
            "textureCompareProjGradOffset",
            "textureGatherCompare",
            "textureGatherCompareOffset",
            "textureOffset",
            "textureLodOffset",
            "textureGradOffset",
            "textureProj",
            "textureProjOffset",
            "textureProjLod",
            "textureProjLodOffset",
            "textureProjGrad",
            "textureProjGradOffset",
            "textureQueryLod",
        }
        if (
            func_name not in split_sampler_functions
            or not self.has_explicit_sampler_argument(raw_args)
        ):
            return raw_args, args
        return [raw_args[0], *raw_args[2:]], [args[0], *args[2:]]

    def generate_resource_call(self, func_name, raw_args, args):
        if func_name in {"textureSize", "imageSize"}:
            return self.generate_dimension_query(func_name, raw_args, args)

        if func_name in {"textureSamples", "imageSamples"}:
            return self.generate_sample_count_query(func_name, raw_args, args)

        if func_name == "textureQueryLevels":
            return self.generate_texture_query_levels(raw_args)

        raw_args, args = self.normalize_split_sampler_texture_args(
            func_name, raw_args, args
        )

        if func_name == "textureQueryLod" and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_resource_query_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "texture",
                "textureLod",
                "textureGrad",
                "textureGather",
                "textureCompare",
                "textureCompareLod",
                "textureCompareGrad",
                "textureCompareOffset",
                "textureCompareLodOffset",
                "textureCompareGradOffset",
                "textureCompareProj",
                "textureCompareProjOffset",
                "textureCompareProjLod",
                "textureCompareProjLodOffset",
                "textureCompareProjGrad",
                "textureCompareProjGradOffset",
                "textureGatherCompare",
                "textureGatherCompareOffset",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )

        if func_name == "textureGather" and len(args) >= 2:
            texture_gather = self.generate_texture_gather_call(raw_args, args)
            if texture_gather is not None:
                return texture_gather

        if (
            func_name
            in {
                "textureGather",
                "textureGatherOffset",
                "textureGatherOffsets",
            }
            and len(args) >= 2
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if (
            func_name
            in {
                "textureProj",
                "textureProjLod",
                "textureProjGrad",
            }
            and len(args) >= 2
        ):
            projected_texture = self.generate_projected_texture_call(
                func_name, raw_args, args
            )
            if projected_texture is not None:
                return projected_texture

        if func_name == "texelFetchOffset" and len(args) >= 4:
            texel_fetch_offset = self.generate_texel_fetch_offset_call(raw_args, args)
            if texel_fetch_offset is not None:
                return texel_fetch_offset

        if (
            func_name
            in {
                "textureOffset",
                "textureLodOffset",
                "textureGradOffset",
                "textureProj",
                "textureProjOffset",
                "textureProjLod",
                "textureProjLodOffset",
                "textureProjGrad",
                "textureProjGradOffset",
                "texelFetchOffset",
            }
            and raw_args
        ):
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None:
                if self.is_shadow_resource_type(texture_type):
                    return self.unsupported_shadow_resource_call(
                        func_name, texture_type, args
                    )
                if self.is_multisample_resource_type(texture_type):
                    return self.unsupported_multisample_resource_call(
                        func_name, texture_type, args
                    )
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

        if func_name in {
            "imageAtomicAdd",
            "imageAtomicMin",
            "imageAtomicMax",
            "imageAtomicAnd",
            "imageAtomicOr",
            "imageAtomicXor",
            "imageAtomicExchange",
            "imageAtomicCompSwap",
        }:
            image_type = None
            if raw_args:
                image_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
                access_diagnostic = self.image_atomic_access_diagnostic(
                    func_name, image_type, raw_args[0]
                )
                if access_diagnostic is not None:
                    return access_diagnostic
            if len(raw_args) >= 2:
                coordinate_diagnostic = self.image_atomic_coordinate_rank_diagnostic(
                    func_name, image_type, raw_args[1]
                )
                if coordinate_diagnostic is not None:
                    return coordinate_diagnostic
            return self.unsupported_image_atomic_resource_call(
                func_name, image_type, args
            )

        if func_name in {"texture", "textureLod", "textureGrad"} and len(args) >= 2:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
                func_name, texture_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            grad_x = None
            grad_y = None
            if func_name == "textureGrad" and len(args) >= 4:
                gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                    texture_type,
                    raw_args[2],
                    raw_args[3],
                )
                if gradient_diagnostic is not None:
                    return gradient_diagnostic
                grad_x = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[2],
                    args[2],
                )
                grad_y = self.hip_texture_gradient_argument(
                    texture_type,
                    raw_args[3],
                    args[3],
                )

            texture_name = args[0]
            coord = args[1]
            if texture_type == "sampler1D":
                if func_name == "texture":
                    return f"tex1D<float4>({texture_name}, {coord})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLod<float4>({texture_name}, {coord}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DGrad<float4>"
                        f"({texture_name}, {coord}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex2D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex2DGrad<float4>" f"({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "sampler1DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}"
                )
                if func_name == "texture":
                    return f"tex1DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex1DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex1DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler2DArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex2DLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex2DLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"tex2DLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "sampler3D":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"tex3D<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"tex3DLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

            if texture_type == "samplerCube":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}"
                )
                if func_name == "texture":
                    return f"texCubemap<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapGrad<float4>" f"({coord_args}, {grad_x}, {grad_y})"
                    )

            if texture_type == "samplerCubeArray":
                coord_args = (
                    f"{texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')}, "
                    f"{self.coord_component(coord, 'w')}"
                )
                if func_name == "texture":
                    return f"texCubemapLayered<float4>({coord_args})"
                if func_name == "textureLod" and len(args) >= 3:
                    return f"texCubemapLayeredLod<float4>({coord_args}, {args[2]})"
                if func_name == "textureGrad" and len(args) >= 4:
                    return (
                        f"texCubemapLayeredGrad<float4>"
                        f"({coord_args}, {grad_x}, {grad_y})"
                    )

        if func_name == "texelFetch" and len(args) >= 3:
            texture_type = self.resource_base_type(
                self.get_expression_type(raw_args[0])
            )
            if texture_type is not None and not self.is_sampled_texture_family_type(
                texture_type
            ):
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )
            if self.is_shadow_resource_type(texture_type):
                return self.unsupported_shadow_resource_call(
                    func_name, texture_type, args
                )
            if self.is_multisample_resource_type(texture_type):
                return self.unsupported_multisample_resource_call(
                    func_name, texture_type, args
                )
            texture_shape = self.sampled_texture_shape_type(texture_type)
            if texture_shape in {"samplerCube", "samplerCubeArray"}:
                return self.unsupported_sampled_resource_call(
                    func_name, texture_type, args
                )

            texture_name = args[0]
            coord = args[1]
            value_type = self.sampled_texture_value_type(texture_type)
            expected_coord_count = self.expected_texel_fetch_coordinate_count(
                texture_shape
            )
            actual_coord_count = self.texel_fetch_coordinate_count(raw_args[1])
            if (
                expected_coord_count is not None
                and actual_coord_count is not None
                and actual_coord_count != expected_coord_count
            ):
                return self.unsupported_sampled_resource_call(
                    "texelFetch coordinate rank",
                    texture_type,
                    args,
                )
            if texture_shape == "sampler1D":
                return f"tex1Dfetch<{value_type}>({texture_name}, {coord})"
            if texture_shape == "sampler1DArray":
                return (
                    f"tex1DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2D":
                return (
                    f"tex2D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')})"
                )
            if texture_shape == "sampler2DArray":
                return (
                    f"tex2DLayered<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )
            if texture_shape == "sampler3D":
                return (
                    f"tex3D<{value_type}>({texture_name}, "
                    f"{self.coord_component(coord, 'x')}, "
                    f"{self.coord_component(coord, 'y')}, "
                    f"{self.coord_component(coord, 'z')})"
                )

        if func_name == "imageLoad" and len(args) >= 2:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                self.require_surface_read_helper("cgl_surf1DLayeredread")
                layer = self.coord_component(coord, "y")
                return (
                    f"cgl_surf1DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {layer})"
                )
            if "1D" in image_type:
                self.require_surface_read_helper("cgl_surf1Dread")
                return f"cgl_surf1Dread<{value_type}>({image_name}, {x})"
            if "CubeArray" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapLayeredread")
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"cgl_surfCubemapLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                self.require_surface_read_helper("cgl_surfCubemapread")
                face = self.coord_component(coord, "z")
                return (
                    f"cgl_surfCubemapread<{value_type}>"
                    f"({image_name}, {x}, {y}, {face})"
                )
            if "3D" in image_type:
                self.require_surface_read_helper("cgl_surf3Dread")
                z = self.coord_component(coord, "z")
                return f"cgl_surf3Dread<{value_type}>({image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                self.require_surface_read_helper("cgl_surf2DLayeredread")
                layer = self.coord_component(coord, "z")
                return (
                    f"cgl_surf2DLayeredread<{value_type}>"
                    f"({image_name}, {x}, {y}, {layer})"
                )
            if "2D" in image_type:
                self.require_surface_read_helper("cgl_surf2Dread")
                return f"cgl_surf2Dread<{value_type}>({image_name}, {x}, {y})"

        if func_name == "imageStore" and len(args) >= 3:
            image_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
            if image_type is None:
                return None
            if self.is_multisample_resource_type(image_type):
                return self.unsupported_multisample_resource_call(
                    func_name, image_type, args
                )
            access_diagnostic = self.image_access_diagnostic(
                func_name, image_type, raw_args[0]
            )
            if access_diagnostic is not None:
                return access_diagnostic

            image_name = args[0]
            coord = args[1]
            value = args[2]
            coordinate_diagnostic = self.image_coordinate_rank_diagnostic(
                func_name, image_type, raw_args[1]
            )
            if coordinate_diagnostic is not None:
                return coordinate_diagnostic
            value_type = self.image_value_type(image_type)
            x = self.image_surface_x_offset(coord, value_type, image_type)
            y = self.coord_component(coord, "y")

            if "1DArray" in image_type:
                layer = self.coord_component(coord, "y")
                return f"surf1DLayeredwrite({value}, {image_name}, {x}, {layer})"
            if "1D" in image_type:
                return f"surf1Dwrite({value}, {image_name}, {x})"
            if "CubeArray" in image_type:
                face = self.coord_component(coord, "z")
                layer = self.coord_component(coord, "w")
                return (
                    f"surfCubemapLayeredwrite"
                    f"({value}, {image_name}, {x}, {y}, {face}, {layer})"
                )
            if "Cube" in image_type:
                face = self.coord_component(coord, "z")
                return f"surfCubemapwrite({value}, {image_name}, {x}, {y}, {face})"
            if "3D" in image_type:
                z = self.coord_component(coord, "z")
                return f"surf3Dwrite({value}, {image_name}, {x}, {y}, {z})"
            if "Array" in image_type:
                layer = self.coord_component(coord, "z")
                return f"surf2DLayeredwrite({value}, {image_name}, {x}, {y}, {layer})"
            if "2D" in image_type:
                return f"surf2Dwrite({value}, {image_name}, {x}, {y})"

        return None

    def generate_texel_fetch_offset_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is not None and not self.is_sampled_texture_family_type(
            texture_type
        ):
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(
                "texelFetchOffset", texture_type, args
            )
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                "texelFetchOffset", texture_type, args
            )
        texture_shape = self.sampled_texture_shape_type(texture_type)
        if texture_shape in {"samplerCube", "samplerCubeArray"}:
            return self.unsupported_sampled_resource_call(
                "texelFetchOffset", texture_type, args
            )

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "texelFetchOffset",
            texture_type,
            raw_args[1],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        offset_diagnostic = self.texel_fetch_offset_rank_diagnostic(
            texture_type,
            raw_args[3],
        )
        if offset_diagnostic is not None:
            return offset_diagnostic

        texture_name = args[0]
        coord = args[1]
        offset = args[3]
        value_type = self.sampled_texture_value_type(texture_type)
        if texture_shape == "sampler1D":
            return f"tex1Dfetch<{value_type}>({texture_name}, ({coord} + {offset}))"
        if texture_shape == "sampler1DArray":
            return (
                f"tex1DLayered<{value_type}>({texture_name}, "
                f"({self.coord_component(coord, 'x')} + {offset}), "
                f"{self.coord_component(coord, 'y')})"
            )
        if texture_shape == "sampler2D":
            return (
                f"tex2D<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')})"
            )
        if texture_shape == "sampler2DArray":
            return (
                f"tex2DLayered<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.coord_component(coord, 'z')})"
            )
        if texture_shape == "sampler3D":
            return (
                f"tex3D<{value_type}>({texture_name}, "
                f"{self.offset_coord_component(coord, offset, 'x')}, "
                f"{self.offset_coord_component(coord, offset, 'y')}, "
                f"{self.offset_coord_component(coord, offset, 'z')})"
            )
        return None

    def generate_projected_texture_call(self, func_name, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type is None:
            return None
        if self.is_shadow_resource_type(texture_type):
            return self.unsupported_shadow_resource_call(func_name, texture_type, args)
        if self.is_multisample_resource_type(texture_type):
            return self.unsupported_multisample_resource_call(
                func_name, texture_type, args
            )

        coord = args[1]
        coordinate_diagnostic = self.projected_texture_coordinate_rank_diagnostic(
            func_name, texture_type, raw_args[1]
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        grad_x = None
        grad_y = None
        if func_name == "textureProjGrad" and len(args) >= 4:
            gradient_diagnostic = self.texture_gradient_rank_diagnostic(
                texture_type,
                raw_args[2],
                raw_args[3],
            )
            if gradient_diagnostic is not None:
                return gradient_diagnostic
            grad_x = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[2],
                args[2],
            )
            grad_y = self.hip_texture_gradient_argument(
                texture_type,
                raw_args[3],
                args[3],
            )

        texture_name = args[0]
        coord_args = self.projected_coord_args(texture_type, texture_name, coord)
        if coord_args is None:
            return self.unsupported_sampled_resource_call(func_name, texture_type, args)

        if texture_type == "sampler1D":
            if func_name == "textureProj":
                return f"tex1D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex1DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex1DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler2D":
            if func_name == "textureProj":
                return f"tex2D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex2DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex2DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "sampler3D":
            if func_name == "textureProj":
                return f"tex3D<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"tex3DLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"tex3DGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        if texture_type == "samplerCube":
            if func_name == "textureProj":
                return f"texCubemap<float4>({coord_args})"
            if func_name == "textureProjLod" and len(args) >= 3:
                return f"texCubemapLod<float4>({coord_args}, {args[2]})"
            if func_name == "textureProjGrad" and grad_x is not None:
                return f"texCubemapGrad<float4>({coord_args}, {grad_x}, {grad_y})"

        return self.unsupported_sampled_resource_call(func_name, texture_type, args)

    def generate_texture_gather_call(self, raw_args, args):
        texture_type = self.resource_base_type(self.get_expression_type(raw_args[0]))
        if texture_type != "sampler2D":
            return None

        coordinate_index = 1
        component = None
        component_arg = None
        if len(args) == 2:
            pass
        elif len(args) == 3:
            component = args[2]
            component_arg = raw_args[2]
        else:
            return None

        coordinate_diagnostic = self.texture_coordinate_rank_diagnostic(
            "textureGather",
            texture_type,
            raw_args[coordinate_index],
        )
        if coordinate_diagnostic is not None:
            return coordinate_diagnostic

        component_diagnostic = self.texture_gather_component_diagnostic(
            texture_type, component_arg
        )
        if component_diagnostic is not None:
            return component_diagnostic

        texture_name = args[0]
        coord = args[coordinate_index]
        gather_args = (
            f"{texture_name}, "
            f"{self.coord_component(coord, 'x')}, "
            f"{self.coord_component(coord, 'y')}"
        )
        if component is not None:
            return f"tex2Dgather<float4>({gather_args}, {component})"
        return f"tex2Dgather<float4>({gather_args})"

    def texture_gather_component_diagnostic(self, texture_type, component_arg):
        if not self.is_texture_gather_literal_component(component_arg):
            return None
        component_value = self.literal_int_value(component_arg)
        if component_value in {0, 1, 2, 3}:
            return None
        return self.unsupported_sampled_resource_call(
            "textureGather component literal must be 0, 1, 2, or 3",
            texture_type,
            [],
        )

    def is_texture_gather_literal_component(self, node):
        if isinstance(node, LiteralNode):
            return True
        if isinstance(node, UnaryOpNode) and not getattr(node, "is_postfix", False):
            operator = getattr(node, "operator", getattr(node, "op", None))
            if operator in {"+", "-"}:
                return self.is_texture_gather_literal_component(node.operand)
        return False

    def generate_buffer_call(self, function_expr, func_name, raw_args, args):
        """Lower structured and byte-address buffer calls to HIP pointer forms."""
        byte_address_call = self.generate_byte_address_buffer_call(
            function_expr, func_name, raw_args, args
        )
        if byte_address_call is not None:
            return byte_address_call

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "Append" and args:
                return self.generate_structured_buffer_append(
                    buffer_expr, buffer_type, args[0], operation
                )
            if operation == "Consume":
                return self.generate_structured_buffer_consume(
                    buffer_expr, buffer_type, operation
                )
            if operation == "GetDimensions":
                return self.generate_structured_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )
            access = self.format_structured_buffer_access(buffer_expr, raw_args, args)
            if access is None:
                return None
            if operation == "Load":
                parts = self.structured_buffer_type_parts(buffer_type)
                fallback = self.diagnostic_zero_value_for_type(
                    parts[1] if parts is not None else None
                )
                diagnostic = self.structured_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                return access
            if operation == "Store":
                diagnostic = self.structured_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.structured_buffer_is_writable(buffer_type) and len(args) >= 2:
                    return f"{access} = {args[1]}"
                return self.structured_buffer_diagnostic_call(
                    "Store", buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is None:
                return None
            fallback = self.diagnostic_zero_value_for_type(parts[1])
            diagnostic = self.structured_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, fallback
            )
            if diagnostic is not None:
                return diagnostic
            return f"{args[0]}[{args[1]}]"

        if func_name == "buffer_append" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_append(
                raw_args[0], buffer_type, args[1], func_name
            )

        if func_name == "buffer_consume" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_consume(
                raw_args[0], buffer_type, func_name
            )

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            return self.generate_structured_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.structured_buffer_type_parts(buffer_type) is None:
                return None
            diagnostic = self.structured_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.structured_buffer_is_writable(buffer_type):
                return f"{args[0]}[{args[1]}] = {args[2]}"
            return self.structured_buffer_diagnostic_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_byte_address_buffer_call(
        self, function_expr, func_name, raw_args, args
    ):
        """Lower byte-address buffer loads and stores to typed HIP helpers."""
        member_call = self.byte_address_buffer_member_call(function_expr)
        if member_call is not None:
            buffer_expr, operation, buffer_type = member_call
            if operation == "GetDimensions":
                return self.generate_byte_address_buffer_dimensions(
                    buffer_expr, buffer_type, raw_args, args, operation
                )

            if operation in self.byte_address_buffer_atomic_operations():
                return self.generate_byte_address_buffer_atomic(
                    buffer_expr, buffer_type, operation, args
                )

            component_count = self.byte_address_buffer_component_count(operation)
            if component_count is None:
                return None

            buffer_name = self.visit(buffer_expr)
            if operation.startswith("Load") and len(args) >= 1:
                fallback = self.diagnostic_zero_value_for_type(
                    self.byte_address_buffer_value_type(component_count)
                )
                diagnostic = self.byte_address_buffer_read_diagnostic(
                    buffer_expr, buffer_type, operation, fallback
                )
                if diagnostic is not None:
                    return diagnostic
                helper_name = self.require_byte_address_load_helper(component_count)
                return f"{helper_name}({buffer_name}, {args[0]})"

            if operation.startswith("Store") and len(args) >= 2:
                diagnostic = self.byte_address_buffer_write_diagnostic(
                    buffer_expr, buffer_type, operation
                )
                if diagnostic is not None:
                    return diagnostic
                if self.byte_address_buffer_is_writable(buffer_type):
                    helper_name = self.require_byte_address_store_helper(
                        component_count
                    )
                    return f"{helper_name}({buffer_name}, {args[0]}, {args[1]})"
                return self.byte_address_buffer_diagnostic_call(
                    operation, buffer_type, "((void)0)"
                )
            return None

        if func_name == "buffer_load" and len(args) >= 2:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_read_diagnostic(
                raw_args[0], buffer_type, func_name, "0u"
            )
            if diagnostic is not None:
                return diagnostic
            helper_name = self.require_byte_address_load_helper(1)
            return f"{helper_name}({args[0]}, {args[1]})"

        if func_name == "buffer_dimensions" and args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            return self.generate_byte_address_buffer_dimensions(
                raw_args[0], buffer_type, raw_args[1:], args[1:], func_name
            )

        if func_name == "buffer_store" and len(args) >= 3:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is None:
                return None
            diagnostic = self.byte_address_buffer_write_diagnostic(
                raw_args[0], buffer_type, func_name
            )
            if diagnostic is not None:
                return diagnostic
            if self.byte_address_buffer_is_writable(buffer_type):
                helper_name = self.require_byte_address_store_helper(1)
                return f"{helper_name}({args[0]}, {args[1]}, {args[2]})"
            return self.byte_address_buffer_diagnostic_call(
                "buffer_store", buffer_type, "((void)0)"
            )

        return None

    def generate_structured_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower structured-buffer dimensions through an explicit length sidecar."""
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* HIP structured buffer dimensions "
                "requires explicit length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def generate_structured_buffer_append(
        self, buffer_expr, buffer_type, value, operation
    ):
        """Lower AppendStructuredBuffer writes through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "AppendStructuredBuffer":
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, "((void)0)"
            )

        helper_name = self.require_structured_buffer_append_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter}, {value})"

    def generate_structured_buffer_consume(self, buffer_expr, buffer_type, operation):
        """Lower ConsumeStructuredBuffer reads through an explicit counter."""
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None or parts[0] != "ConsumeStructuredBuffer":
            fallback = self.diagnostic_zero_value_for_type(
                parts[1] if parts is not None else None
            )
            return self.structured_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )

        counter = self.structured_buffer_counter_expression(buffer_expr)
        if counter is None:
            return self.structured_buffer_diagnostic_call(
                operation,
                buffer_type,
                self.diagnostic_zero_value_for_type(parts[1]),
            )

        helper_name = self.require_structured_buffer_consume_helper()
        buffer_name = self.visit(buffer_expr)
        return f"{helper_name}({buffer_name}, {counter})"

    def require_structured_buffer_append_helper(self):
        """Register the HIP helper for AppendStructuredBuffer operations."""
        helper_name = "cgl_append_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline void "
            f"{helper_name}(T* buffer, uint* counter, const T& value)\n"
            "{\n"
            "    uint index = atomicAdd(counter, 1u);\n"
            "    buffer[index] = value;\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_structured_buffer_consume_helper(self):
        """Register the HIP helper for ConsumeStructuredBuffer operations."""
        helper_name = "cgl_consume_structured_buffer"
        if helper_name in self.helper_functions:
            return helper_name

        helper = (
            "template <typename T>\n"
            "__device__ inline T "
            f"{helper_name}(const T* buffer, uint* counter)\n"
            "{\n"
            "    uint index = atomicSub(counter, 1u) - 1u;\n"
            "    return buffer[index];\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def structured_buffer_atomic_operations(self):
        """Return generic atomic mappings for RWStructuredBuffer element targets."""
        integer_kinds = {"int", "uint"}
        add_exchange_kinds = {*integer_kinds, "float"}
        return {
            "atomicAdd": ("atomicAdd", 2, add_exchange_kinds, "int/uint/float"),
            "atomicSub": ("atomicSub", 2, integer_kinds, "int/uint"),
            "atomicMin": ("atomicMin", 2, integer_kinds, "int/uint"),
            "atomicMax": ("atomicMax", 2, integer_kinds, "int/uint"),
            "atomicAnd": ("atomicAnd", 2, integer_kinds, "int/uint"),
            "atomicOr": ("atomicOr", 2, integer_kinds, "int/uint"),
            "atomicXor": ("atomicXor", 2, integer_kinds, "int/uint"),
            "atomicInc": ("atomicInc", 2, {"uint"}, "uint"),
            "atomicDec": ("atomicDec", 2, {"uint"}, "uint"),
            "atomicExchange": ("atomicExch", 2, add_exchange_kinds, "int/uint/float"),
            "atomicCompareExchange": ("atomicCAS", 3, integer_kinds, "int/uint"),
            "atomicCompSwap": ("atomicCAS", 3, integer_kinds, "int/uint"),
        }

    def generate_structured_buffer_atomic_call(self, func_name, raw_args, args):
        """Lower atomics on RWStructuredBuffer element lvalues to HIP atomics."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None or not raw_args:
            return None

        target = self.structured_buffer_atomic_target(raw_args[0])
        if target is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        target_type = target["target_type"]
        fallback = self.diagnostic_zero_value_for_type(target_type)

        if len(args) != required_arg_count:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires {required_arg_count} argument(s)",
                fallback,
            )

        buffer_base_type, _ = self.structured_buffer_type_parts(target["buffer_type"])
        access_diagnostic = self.structured_buffer_read_write_diagnostic(
            target["buffer_expr"],
            target["buffer_type"],
            func_name,
            fallback,
        )
        if access_diagnostic is not None:
            return access_diagnostic
        if buffer_base_type != "RWStructuredBuffer":
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                "requires RWStructuredBuffer target",
                fallback,
            )

        scalar_kind = self.hip_atomic_scalar_kind(target_type)
        if scalar_kind not in supported_kinds:
            return self.unsupported_structured_buffer_atomic_call(
                func_name,
                target["buffer_type"],
                f"requires supported scalar {supported_kinds_label} target",
                fallback,
            )

        target_expr = args[0]
        value_args = ", ".join(args[1:])
        return f"{intrinsic}(&{target_expr}, {value_args})"

    def generate_plain_atomic_call(self, func_name, raw_args, args):
        """Lower HIP atomics on ordinary scalar lvalues to pointer operands."""
        operation = self.structured_buffer_atomic_operations().get(func_name)
        if operation is None:
            return None

        intrinsic, required_arg_count, supported_kinds, supported_kinds_label = (
            operation
        )
        raw_target = raw_args[0] if raw_args else None
        target_expr = (
            self.strip_address_of_expression(raw_target) if raw_target else None
        )
        address_taken = target_expr is not raw_target
        target_type = self.expression_result_type(target_expr) if target_expr else None
        indirect_info = self.hip_indirect_type_info(target_type)
        pointee_type = (
            indirect_info["pointee_type"] if indirect_info is not None else None
        )
        atomic_type = pointee_type or target_type
        fallback = self.diagnostic_zero_value_for_type(atomic_type)

        if len(args) != required_arg_count:
            return self.unsupported_plain_atomic_call(
                func_name,
                f"requires {required_arg_count} argument(s)",
                fallback,
            )
        if target_expr is None or (
            pointee_type is None and not self.is_plain_atomic_lvalue(target_expr)
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                "requires assignable scalar target",
                fallback,
            )

        if (
            address_taken
            and indirect_info is not None
            and indirect_info["kind"] == "pointer"
        ):
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {indirect_info['type_label']} requires pointer target, "
                "not address of pointer",
                fallback,
            )

        readonly_reason = self.plain_atomic_readonly_target_reason(
            target_expr, indirect_info
        )
        if readonly_reason is not None:
            return self.unsupported_plain_atomic_call(
                func_name,
                readonly_reason,
                fallback,
            )

        access_diagnostic = self.glsl_buffer_block_read_write_diagnostic(
            target_expr, func_name, fallback
        )
        if access_diagnostic is not None:
            return access_diagnostic

        scalar_kind = self.hip_atomic_scalar_kind(atomic_type)
        if scalar_kind not in supported_kinds:
            type_label = (
                indirect_info["type_label"]
                if indirect_info is not None
                else self.type_name_string(atomic_type) or "unknown target"
            )
            return self.unsupported_plain_atomic_call(
                func_name,
                f"on {type_label} requires supported scalar "
                f"{supported_kinds_label} target",
                fallback,
            )

        target_code = self.visit(target_expr)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            target_pointer = target_code
        else:
            target_pointer = f"&{target_code}"
        value_args = ", ".join(args[1:])
        return f"{intrinsic}({target_pointer}, {value_args})"

    def is_plain_atomic_lvalue(self, expr):
        """Return whether an expression can be addressed for a HIP atomic."""
        return isinstance(
            expr, (IdentifierNode, ArrayAccessNode, MemberAccessNode, PointerAccessNode)
        )

    def hip_indirect_type_info(self, type_name):
        """Return pointer/reference metadata for HIP indirect type spellings."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        if type_name.startswith("ptr<") and type_name.endswith(">"):
            pointee = type_name[4:-1].strip()
            return {
                "kind": "pointer",
                "pointee_type": pointee,
                "readonly": False,
                "type_label": f"{pointee}*",
            }

        mapped_type = self.map_type(type_name)
        for candidate in (type_name, mapped_type):
            info = self.hip_pointer_or_reference_type_info(candidate)
            if info is not None:
                return info
        return None

    def hip_pointer_or_reference_type_info(self, type_name):
        """Parse a HIP pointer/reference spelling into target metadata."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        stripped = type_name.strip()
        for suffix, kind in (("*", "pointer"), ("&", "reference")):
            if not stripped.endswith(suffix):
                continue
            pointee = stripped[: -len(suffix)].strip()
            readonly = pointee.startswith("const ")
            if readonly:
                pointee = pointee[len("const ") :].strip()
            return {
                "kind": kind,
                "pointee_type": pointee,
                "readonly": readonly,
                "type_label": stripped,
            }
        return None

    def plain_atomic_readonly_target_reason(self, target_expr, indirect_info):
        """Return a diagnostic reason for readonly plain atomic targets."""
        if indirect_info is not None and indirect_info["readonly"]:
            if indirect_info["kind"] == "reference":
                return (
                    f"on {indirect_info['type_label']} requires mutable "
                    "reference target"
                )
            return f"on {indirect_info['type_label']} requires writable pointer target"

        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            array_type = self.expression_result_type(array_expr)
            array_info = self.hip_indirect_type_info(array_type)
            if array_info is not None and array_info["readonly"]:
                if array_info["kind"] == "reference":
                    return (
                        f"on {array_info['type_label']} requires mutable "
                        "reference target"
                    )
                return f"on {array_info['type_label']} requires writable pointer target"
        return None

    def strip_address_of_expression(self, expr):
        """Return the lvalue inside an address-of expression."""
        if isinstance(expr, UnaryOpNode):
            operator = getattr(expr, "operator", getattr(expr, "op", None))
            if operator == "&":
                return getattr(expr, "operand", expr)
        return expr

    def unsupported_plain_atomic_call(self, operation, reason, fallback):
        """Return diagnostic code for unsupported ordinary HIP atomics."""
        return (
            f"/* unsupported {self.resource_backend_name()} atomic: "
            f"{operation} {reason} */ {fallback}"
        )

    def structured_buffer_atomic_target(self, target_expr):
        """Return RWStructuredBuffer target metadata for an atomic lvalue."""
        element_access = self.structured_buffer_element_access(target_expr)
        if element_access is None:
            return None

        array_expr = getattr(
            element_access, "array_expr", getattr(element_access, "array", None)
        )
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        target_type = self.expression_result_type(target_expr) or parts[1]
        return {
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "target_type": target_type,
        }

    def structured_buffer_element_access(self, target_expr):
        """Return the structured-buffer element access inside an atomic lvalue."""
        if isinstance(target_expr, ArrayAccessNode):
            array_expr = getattr(
                target_expr, "array_expr", getattr(target_expr, "array", None)
            )
            buffer_type = self.expression_result_type(array_expr)
            if self.structured_buffer_type_parts(buffer_type) is not None:
                return target_expr

        if isinstance(target_expr, MemberAccessNode):
            object_expr = getattr(
                target_expr, "object_expr", getattr(target_expr, "object", None)
            )
            return self.structured_buffer_element_access(object_expr)

        return None

    def structured_buffer_data_access_info(self, expr):
        """Return direct structured-buffer element access metadata, if any."""
        if not isinstance(expr, ArrayAccessNode):
            return None

        array_expr = getattr(expr, "array_expr", getattr(expr, "array", None))
        buffer_type = self.expression_result_type(array_expr)
        parts = self.structured_buffer_type_parts(buffer_type)
        if parts is None:
            return None

        result_type = self.expression_result_type(expr)
        if self.structured_buffer_type_parts(result_type) is not None:
            return None

        return {
            "buffer_expr": array_expr,
            "buffer_type": buffer_type,
            "element_type": result_type or parts[1],
        }

    def structured_buffer_element_read_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def structured_buffer_element_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        return self.structured_buffer_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation
        )

    def structured_buffer_element_read_write_diagnostic(self, expr, operation):
        info = self.structured_buffer_data_access_info(expr)
        if info is None:
            return None
        fallback = self.diagnostic_zero_value_for_type(info["element_type"])
        return self.structured_buffer_read_write_diagnostic(
            info["buffer_expr"], info["buffer_type"], operation, fallback
        )

    def hip_atomic_scalar_kind(self, type_name):
        """Return the HIP atomic scalar kind supported for structured buffers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        mapped_type = self.map_type(type_name)
        if type_name in {"uint", "u32"} or mapped_type in {"uint", "unsigned int"}:
            return "uint"
        if type_name in {"int", "i32"} or mapped_type == "int":
            return "int"
        if type_name in {"float", "f32"} or mapped_type == "float":
            return "float"
        return None

    def unsupported_structured_buffer_atomic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for unsupported structured-buffer atomics."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer atomic: "
            f"{operation} on {buffer_type} {reason} */ {fallback}"
        )

    def generate_byte_address_buffer_dimensions(
        self, buffer_expr, buffer_type, raw_dimension_args, dimension_args, operation
    ):
        """Lower byte-address buffer dimensions through a byte-length sidecar."""
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        length_expr = self.structured_buffer_length_expression(buffer_expr)
        if length_expr is None:
            length_expr = (
                "0 /* HIP byte-address buffer dimensions "
                "requires explicit byte-length sidecar */"
            )

        if dimension_args:
            return f"{dimension_args[0]} = {length_expr}"
        return length_expr

    def structured_buffer_member_call(self, function_expr):
        """Return structured-buffer member-call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.structured_buffer_type_parts(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def format_structured_buffer_access(self, buffer_expr, raw_args, args):
        """Format one HIP pointer access for a structured-buffer operation."""
        if not raw_args or not args:
            return None
        buffer_name = self.visit(buffer_expr)
        return f"{buffer_name}[{args[0]}]"

    def structured_buffer_diagnostic_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for rejected structured-buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_access_diagnostic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for rejected structured-buffer access metadata."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} structured buffer "
            f"access: {operation} {reason} on {buffer_type} */ {fallback}"
        )

    def structured_buffer_read_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access == "writeonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readable buffer access",
                fallback,
            )
        return None

    def structured_buffer_write_diagnostic(self, buffer_expr, buffer_type, operation):
        access = self.buffer_resource_access(buffer_expr)
        if access == "readonly":
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires writable buffer access",
                "((void)0)",
            )
        return None

    def structured_buffer_read_write_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access in {"readonly", "writeonly"}:
            return self.structured_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readwrite buffer access",
                fallback,
            )
        return None

    def diagnostic_zero_value_for_type(self, type_name):
        """Return a HIP fallback expression for rejected value-producing calls."""
        if type_name is None:
            return "0"

        mapped_type = self.map_type(type_name)
        vector_info = self.vector_type_info(mapped_type) or self.vector_type_info(
            self.type_name_string(type_name)
        )
        if vector_info is not None:
            component_type = vector_info["component_type"]
            if component_type == "uint":
                zero = "0u"
            elif component_type == "double":
                zero = "0.0"
            elif component_type == "float":
                zero = "0.0f"
            elif component_type == "bool":
                zero = "false"
            else:
                zero = "0"
            values = ", ".join([zero] * len(vector_info["components"]))
            return f"{vector_info['constructor']}({values})"

        if mapped_type == "bool":
            return "false"
        if mapped_type in {"float", "half"}:
            return "0.0f"
        if mapped_type == "double":
            return "0.0"
        if mapped_type in {"uint", "unsigned int", "u32"}:
            return "0u"
        if mapped_type in {
            "int",
            "short",
            "char",
            "long long",
            "unsigned char",
            "unsigned short",
            "unsigned long long",
        }:
            return "0"
        return f"{mapped_type}{{}}"

    def collect_structured_buffer_length_requirements(self, root):
        """Collect buffers that need explicit length sidecar parameters."""
        functions = self.query_collect_functions(root)
        functions_by_name = {getattr(func, "name", None): func for func in functions}
        functions_by_name = {
            name: func for name, func in functions_by_name.items() if name
        }
        param_names = {
            func_name: {
                getattr(param, "name", None)
                for param in getattr(func, "parameters", getattr(func, "params", []))
            }
            for func_name, func in functions_by_name.items()
        }
        param_names = {
            func_name: {name for name in names if name}
            for func_name, names in param_names.items()
        }

        global_length_names = set()
        function_param_length_names = {
            func_name: set() for func_name in functions_by_name
        }

        def mark_resource_name(func_name, resource_name):
            if not resource_name:
                return False
            if resource_name in param_names.get(func_name, set()):
                before = len(function_param_length_names[func_name])
                function_param_length_names[func_name].add(resource_name)
                return len(function_param_length_names[func_name]) != before
            before = len(global_length_names)
            global_length_names.add(resource_name)
            return len(global_length_names) != before

        for func_name, func in functions_by_name.items():
            for call in self.query_walk_nodes(getattr(func, "body", [])):
                if not isinstance(call, FunctionCallNode):
                    continue
                buffer_expr = self.structured_buffer_dimensions_target(call)
                if buffer_expr is None:
                    continue
                mark_resource_name(func_name, self.get_expression_name(buffer_expr))

        changed = True
        while changed:
            changed = False
            for caller_name, caller in functions_by_name.items():
                caller_params = param_names.get(caller_name, set())
                for call in self.query_walk_nodes(getattr(caller, "body", [])):
                    if not isinstance(call, FunctionCallNode):
                        continue
                    callee_name = self.raw_function_call_name(call)
                    callee = functions_by_name.get(callee_name)
                    if callee is None:
                        continue

                    callee_required = function_param_length_names.get(
                        callee_name, set()
                    )
                    if not callee_required:
                        continue

                    callee_params = getattr(
                        callee, "parameters", getattr(callee, "params", [])
                    )
                    raw_args = getattr(call, "arguments", getattr(call, "args", []))
                    for index, param in enumerate(callee_params):
                        if index >= len(raw_args):
                            continue
                        param_name = getattr(param, "name", None)
                        if param_name not in callee_required:
                            continue

                        arg_name = self.get_expression_name(raw_args[index])
                        if not arg_name:
                            continue
                        if arg_name in caller_params:
                            before = len(function_param_length_names[caller_name])
                            function_param_length_names[caller_name].add(arg_name)
                            changed = (
                                changed
                                or len(function_param_length_names[caller_name])
                                != before
                            )
                        else:
                            before = len(global_length_names)
                            global_length_names.add(arg_name)
                            changed = changed or len(global_length_names) != before

        return (
            global_length_names,
            {
                func_name: names
                for func_name, names in function_param_length_names.items()
                if names
            },
        )

    def structured_buffer_dimensions_target(self, call):
        """Return the buffer expression queried by a dimensions call."""
        func_name = self.raw_function_call_name(call)
        raw_args = getattr(call, "arguments", getattr(call, "args", []))
        if func_name == "buffer_dimensions" and raw_args:
            return raw_args[0]

        function_expr = getattr(call, "function", getattr(call, "name", None))
        if isinstance(function_expr, MemberAccessNode):
            member = getattr(function_expr, "member", None)
            if member == "GetDimensions":
                return getattr(
                    function_expr,
                    "object_expr",
                    getattr(function_expr, "object", None),
                )
        return None

    def structured_buffer_length_name(self, name):
        """Return the sidecar length parameter/declaration name for a buffer."""
        return f"{name}_length"

    def structured_buffer_requires_length(self, name):
        """Return whether a global buffer needs a length sidecar."""
        return bool(name and name in self.structured_buffer_length_names)

    def structured_buffer_parameter_requires_length(
        self, func_name, name, type_name=None
    ):
        """Return whether a function parameter needs a length sidecar."""
        if not name:
            return False
        if name not in self.structured_buffer_length_function_params.get(
            func_name, set()
        ):
            return False
        return type_name is None or self.buffer_type_supports_length(type_name)

    def buffer_type_supports_length(self, type_name):
        """Return whether a resource type can use a HIP length sidecar."""
        return (
            self.structured_buffer_type_parts(type_name) is not None
            or self.byte_address_buffer_base_type(type_name) is not None
        )

    def structured_buffer_length_declaration(self, name, type_name):
        """Format a global/local sidecar length declaration when required."""
        if not self.structured_buffer_requires_length(name):
            return None
        if not self.buffer_type_supports_length(type_name):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def structured_buffer_length_parameter(self, func_name, name, type_name):
        """Format a sidecar length parameter when required."""
        if not self.structured_buffer_parameter_requires_length(
            func_name, name, type_name
        ):
            return None
        return self.format_structured_buffer_length_declarator(type_name, name)

    def format_structured_buffer_length_declarator(self, type_name, name):
        """Format the HIP uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        length_name = self.structured_buffer_length_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"const uint* {length_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("const uint*", length_name, array_suffix)

    def structured_buffer_length_data_expression(self, buffer_expr):
        """Return the sidecar length pointer paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_length = self.structured_buffer_length_data_expression(array_expr)
            if base_length is None:
                return None
            return f"{base_length}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None

        length_parameter = self.current_structured_buffer_length_parameters.get(name)
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(name):
            return self.structured_buffer_length_name(name)
        return None

    def structured_buffer_length_expression(self, buffer_expr):
        """Return the scalar length expression for a buffer resource."""
        length_data = self.structured_buffer_length_data_expression(buffer_expr)
        if length_data is None:
            return None
        return f"{length_data}[0]"

    def structured_buffer_counter_name(self, name):
        """Return the sidecar counter parameter/declaration name for a buffer."""
        return f"{name}_counter"

    def structured_buffer_counter_declaration(self, name, type_name):
        """Format a global/local sidecar counter declaration when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def structured_buffer_counter_parameter(self, name, type_name):
        """Format a sidecar counter parameter when required."""
        if not self.structured_buffer_requires_counter(type_name):
            return None
        return self.format_structured_buffer_counter_declarator(type_name, name)

    def format_structured_buffer_counter_declarator(self, type_name, name):
        """Format the HIP uint pointer sidecar matching a buffer declarator."""
        type_name = self.type_name_string(type_name)
        counter_name = self.structured_buffer_counter_name(name)
        if "[" not in type_name or "]" not in type_name:
            return f"uint* {counter_name}"

        array_suffix = type_name[type_name.find("[") :]
        return format_array_declarator("uint*", counter_name, array_suffix)

    def structured_buffer_counter_expression(self, buffer_expr):
        """Return the sidecar counter expression paired with a buffer expression."""
        if isinstance(buffer_expr, ArrayAccessNode):
            array_expr = getattr(
                buffer_expr, "array_expr", getattr(buffer_expr, "array", None)
            )
            index_expr = getattr(
                buffer_expr, "index_expr", getattr(buffer_expr, "index", None)
            )
            base_counter = self.structured_buffer_counter_expression(array_expr)
            if base_counter is None:
                return None
            return f"{base_counter}[{self.visit(index_expr)}]"

        name = self.get_expression_name(buffer_expr)
        if not name:
            return None
        return self.structured_buffer_counter_name(name)

    def hip_user_function_call_arguments(self, func_name, raw_args, args):
        """Expand user calls with HIP sidecar resource arguments."""
        callee = self.query_functions_by_name.get(func_name)
        if callee is None:
            return args + self.hip_captured_function_call_args(func_name)

        params = getattr(callee, "parameters", getattr(callee, "params", []))
        query_params = self.query_metadata_function_params.get(func_name, set())
        expanded_args = []
        for index, arg in enumerate(args):
            expanded_args.append(arg)
            if index >= len(params) or index >= len(raw_args):
                continue

            param = params[index]
            param_name = getattr(param, "name", None)
            if param_name in query_params:
                metadata_arg = self.query_metadata_expression(raw_args[index])
                if metadata_arg:
                    expanded_args.append(metadata_arg)

            param_type = self.get_parameter_type(param)
            if self.structured_buffer_parameter_requires_length(
                func_name, param_name, param_type
            ):
                length_arg = self.structured_buffer_length_data_expression(
                    raw_args[index]
                )
                if length_arg:
                    expanded_args.append(length_arg)

            if self.structured_buffer_requires_counter(param_type):
                counter_arg = self.structured_buffer_counter_expression(raw_args[index])
                if counter_arg:
                    expanded_args.append(counter_arg)

        expanded_args.extend(self.hip_captured_function_call_args(func_name))
        return expanded_args

    def hip_captured_function_call_args(self, func_name):
        return [
            self.visit(IdentifierNode(param.name))
            for param in self.hip_function_capture_params.get(func_name, [])
        ]

    def byte_address_buffer_member_call(self, function_expr):
        """Return byte-address buffer member-call pieces, if applicable."""
        if not isinstance(function_expr, MemberAccessNode):
            return None

        buffer_expr = getattr(
            function_expr, "object_expr", getattr(function_expr, "object", None)
        )
        buffer_type = self.expression_result_type(buffer_expr)
        if self.byte_address_buffer_base_type(buffer_type) is None:
            return None

        return buffer_expr, getattr(function_expr, "member", ""), buffer_type

    def byte_address_buffer_component_count(self, operation):
        """Return the uint lane count for byte-address load/store operations."""
        operation_counts = {
            "Load": 1,
            "Load2": 2,
            "Load3": 3,
            "Load4": 4,
            "Store": 1,
            "Store2": 2,
            "Store3": 3,
            "Store4": 4,
        }
        return operation_counts.get(operation)

    def byte_address_buffer_atomic_operations(self):
        """Return supported RWByteAddressBuffer atomic method mappings."""
        return {
            "InterlockedAdd": ("add", "atomicAdd", 2),
            "InterlockedMin": ("min", "atomicMin", 2),
            "InterlockedMax": ("max", "atomicMax", 2),
            "InterlockedAnd": ("and", "atomicAnd", 2),
            "InterlockedOr": ("or", "atomicOr", 2),
            "InterlockedXor": ("xor", "atomicXor", 2),
            "InterlockedExchange": ("exchange", "atomicExch", 2),
            "InterlockedCompareExchange": ("compare_exchange", "atomicCAS", 3),
        }

    def generate_byte_address_buffer_atomic(
        self, buffer_expr, buffer_type, operation, args
    ):
        """Lower RWByteAddressBuffer Interlocked* methods to HIP atomics."""
        operation_info = self.byte_address_buffer_atomic_operations().get(operation)
        if operation_info is None:
            return None

        operation_name, intrinsic, required_args = operation_info
        has_out_arg = len(args) == required_args + 1
        fallback = "((void)0)" if has_out_arg else "0u"
        if len(args) < required_args or len(args) > required_args + 1:
            return self.byte_address_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )
        diagnostic = self.byte_address_buffer_read_write_diagnostic(
            buffer_expr, buffer_type, operation, fallback
        )
        if diagnostic is not None:
            return diagnostic
        if not self.byte_address_buffer_is_writable(buffer_type):
            return self.byte_address_buffer_diagnostic_call(
                operation, buffer_type, fallback
            )

        helper_name = self.require_byte_address_atomic_helper(operation_name, intrinsic)
        buffer_name = self.visit(buffer_expr)
        helper_args = [buffer_name, *args[:required_args]]
        call = f"{helper_name}({', '.join(helper_args)})"
        if has_out_arg:
            return f"{args[required_args]} = {call}"
        return call

    def byte_address_atomic_helper_name(self, operation):
        """Return a stable HIP helper name for a byte-address atomic operation."""
        return f"cgl_byte_address_atomic_{operation}_uint"

    def require_byte_address_atomic_helper(self, operation, intrinsic):
        """Register a HIP byte-address atomic helper and return its name."""
        helper_name = self.byte_address_atomic_helper_name(operation)
        if helper_name in self.helper_functions:
            return helper_name

        pointer_expr = "reinterpret_cast<unsigned int*>(buffer + offset)"
        if operation == "compare_exchange":
            helper = (
                "__device__ inline uint "
                f"{helper_name}(unsigned char* buffer, uint offset, "
                "uint compare_value, uint value)\n"
                "{\n"
                f"    return {intrinsic}({pointer_expr}, compare_value, value);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        helper = (
            "__device__ inline uint "
            f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
            "{\n"
            f"    return {intrinsic}({pointer_expr}, value);\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_value_type(self, component_count):
        """Return the HIP uint vector type for a byte-address operation."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def byte_address_helper_suffix(self, component_count):
        """Return a stable helper-name suffix for byte-address operations."""
        if component_count == 1:
            return "uint"
        return f"uint{component_count}"

    def require_byte_address_load_helper(self, component_count):
        """Register a HIP byte-address load helper and return its name."""
        helper_name = (
            f"cgl_byte_address_load_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        if component_count == 1:
            helper = (
                "__device__ inline uint "
                f"{helper_name}(const unsigned char* buffer, uint offset)\n"
                "{\n"
                "    return *reinterpret_cast<const uint*>(buffer + offset);\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_load_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        args = [
            f"{scalar_helper}(buffer, offset + {index * 4}u)"
            for index, _ in enumerate(components)
        ]
        value_type = self.byte_address_buffer_value_type(component_count)
        constructor = self.function_map.get(value_type, value_type)
        helper = (
            f"__device__ inline {value_type} "
            f"{helper_name}(const unsigned char* buffer, uint offset)\n"
            "{\n"
            f"    return {constructor}({', '.join(args)});\n"
            "}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def require_byte_address_store_helper(self, component_count):
        """Register a HIP byte-address store helper and return its name."""
        helper_name = (
            f"cgl_byte_address_store_{self.byte_address_helper_suffix(component_count)}"
        )
        if helper_name in self.helper_functions:
            return helper_name

        value_type = self.byte_address_buffer_value_type(component_count)
        if component_count == 1:
            helper = (
                "__device__ inline void "
                f"{helper_name}(unsigned char* buffer, uint offset, uint value)\n"
                "{\n"
                "    *reinterpret_cast<uint*>(buffer + offset) = value;\n"
                "}"
            )
            self.helper_functions[helper_name] = helper
            return helper_name

        scalar_helper = self.require_byte_address_store_helper(1)
        components = ("x", "y", "z", "w")[:component_count]
        lines = [
            f"    {scalar_helper}(buffer, offset + {index * 4}u, value.{component});"
            for index, component in enumerate(components)
        ]
        helper = (
            f"__device__ inline void {helper_name}"
            f"(unsigned char* buffer, uint offset, {value_type} value)\n"
            "{\n" + "\n".join(lines) + "\n}"
        )
        self.helper_functions[helper_name] = helper
        return helper_name

    def byte_address_buffer_diagnostic_call(self, operation, buffer_type, fallback):
        """Return diagnostic code for rejected byte-address buffer operations."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer call: "
            f"{operation} on {buffer_type} */ {fallback}"
        )

    def byte_address_buffer_access_diagnostic_call(
        self, operation, buffer_type, reason, fallback
    ):
        """Return diagnostic code for rejected byte-address access metadata."""
        buffer_type = self.type_name_string(buffer_type) or "unknown buffer"
        return (
            f"/* unsupported {self.resource_backend_name()} byte-address buffer "
            f"access: {operation} {reason} on {buffer_type} */ {fallback}"
        )

    def byte_address_buffer_read_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access == "writeonly":
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readable buffer access",
                fallback,
            )
        return None

    def byte_address_buffer_write_diagnostic(self, buffer_expr, buffer_type, operation):
        access = self.buffer_resource_access(buffer_expr)
        if access == "readonly":
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires writable buffer access",
                "((void)0)",
            )
        return None

    def byte_address_buffer_read_write_diagnostic(
        self, buffer_expr, buffer_type, operation, fallback
    ):
        access = self.buffer_resource_access(buffer_expr)
        if access in {"readonly", "writeonly"}:
            return self.byte_address_buffer_access_diagnostic_call(
                operation,
                buffer_type,
                "requires readwrite buffer access",
                fallback,
            )
        return None

    def visit_str(self, node) -> str:
        return str(node)

    def visit_int(self, node) -> str:
        return str(node)

    def visit_float(self, node) -> str:
        return str(node)

    def visit_ArrayAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
            diagnostic = self.structured_buffer_element_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        array = self.visit(node.array)
        index = self.visit(node.index)
        return f"{array}[{index}]"

    def visit_ArrayLiteralNode(self, node: ArrayLiteralNode) -> str:
        elements = ", ".join(self.visit(element) for element in node.elements)
        return f"{{{elements}}}"

    def visit_MemberAccessNode(self, node) -> str:
        if self.assignment_lhs_depth == 0:
            diagnostic = self.glsl_buffer_block_read_diagnostic(node, "load")
            if diagnostic is not None:
                return diagnostic
        raw_object_name = getattr(node.object, "name", None)
        raw_object_shadows_builtin = self.source_identifier_shadows_builtin(
            raw_object_name
        )
        if raw_object_name is not None and not raw_object_shadows_builtin:
            alias_component = self.hip_stage_builtin_alias_expression(
                raw_object_name, node.member
            )
            if alias_component is not None:
                return alias_component
            ray_builtin = self.hip_ray_builtin_expression(raw_object_name)
            if ray_builtin is not None:
                return f"{ray_builtin}.{node.member}"
            direct_builtin = self.hip_compute_builtin_expression(
                self.hip_compute_builtin_role_for_name(raw_object_name), node.member
            )
            if direct_builtin is not None:
                return direct_builtin
            raw_member_access = f"{raw_object_name}.{node.member}"
            if raw_member_access in self.builtin_map:
                return self.builtin_map[raw_member_access]

        object_expr = self.visit(node.object)
        member_access = f"{object_expr}.{node.member}"
        if not raw_object_shadows_builtin and member_access in self.builtin_map:
            return self.builtin_map[member_access]

        swizzle = self.generate_vector_swizzle(node, object_expr)
        if swizzle is not None:
            return swizzle

        return member_access

    def visit_PointerAccessNode(self, node) -> str:
        pointer_expr = self.visit(getattr(node, "pointer_expr", None))
        return f"{pointer_expr}->{node.member}"

    def generate_vector_swizzle(self, node, object_expr):
        object_node = getattr(node, "object_expr", getattr(node, "object", None))
        vector_info = self.vector_type_info(self.expression_result_type(object_node))
        if vector_info is None:
            return None

        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        member = getattr(node, "member", "")
        components = [component_aliases.get(component) for component in member]
        if not components or any(component is None for component in components):
            return None

        available_components = vector_info["components"]
        if any(component not in available_components for component in components):
            return None

        if len(components) == 1:
            return f"{object_expr}.{components[0]}"

        result_type = self.vector_type_for_components(
            vector_info["component_type"], len(components)
        )
        result_info = self.vector_type_info(result_type)
        if result_info is None:
            return None

        swizzle_call = self.generate_vector_swizzle_single_eval_call(
            result_info,
            vector_info,
            object_node,
            object_expr,
            components,
        )
        if swizzle_call is not None:
            return swizzle_call

        args = [f"{object_expr}.{component}" for component in components]
        return f"{result_info['constructor']}({', '.join(args)})"

    def visit_TernaryOpNode(self, node) -> str:
        condition = self.visit(node.condition)
        true_expr = self.visit(node.true_expr)
        false_expr = self.visit(node.false_expr)
        lowered = self.lower_vector_ternary_operation(
            node.condition,
            condition,
            node.true_expr,
            true_expr,
            node.false_expr,
            false_expr,
        )
        if lowered is not None:
            return lowered
        return f"({condition} ? {true_expr} : {false_expr})"

    def format_literal(self, value, literal_type=None):
        if isinstance(value, bool):
            return "true" if value else "false"
        if literal_type == "bool" and isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "false"}:
                return lower_value
        if literal_type == "char":
            escaped = self.escape_literal(value, quote="'")
            return f"'{escaped}'"
        if (
            literal_type == "uint"
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return f"{value}u"
        if isinstance(value, str):
            escaped = self.escape_literal(value, quote='"')
            return f'"{escaped}"'
        return str(value)

    def escape_literal(self, value, quote):
        text = str(value)
        escaped = []
        for index, char in enumerate(text):
            if char == "\n":
                escaped.append("\\n")
            elif char == "\r":
                escaped.append("\\r")
            elif char == "\t":
                escaped.append("\\t")
            elif char == quote and (index == 0 or text[index - 1] != "\\"):
                escaped.append("\\" + char)
            else:
                escaped.append(char)
        return "".join(escaped)

    def visit_LiteralNode(self, node) -> str:
        literal_type = getattr(getattr(node, "literal_type", None), "name", None)
        return self.format_literal(node.value, literal_type)

    def visit_IdentifierNode(self, node) -> str:
        name = getattr(node, "name", str(node))
        if name in getattr(self, "enum_variant_constants", {}):
            return enum_value_expression(self, name)
        builtin_alias = self.hip_stage_builtin_alias_expression(name)
        if builtin_alias is not None:
            return builtin_alias
        if not self.source_identifier_shadows_builtin(name):
            ray_builtin = self.hip_ray_builtin_expression(name)
            if ray_builtin is not None:
                return ray_builtin
            return self.builtin_map.get(name, name)
        return name

    def hip_ray_builtin_expression(self, name):
        helper_name = {
            "gl_LaunchIDEXT": "cgl_ray_launch_id",
            "gl_LaunchSizeEXT": "cgl_ray_launch_size",
            "gl_HitTEXT": "cgl_ray_hit_t",
            "gl_HitKindEXT": "cgl_ray_hit_kind",
            "gl_WorldRayOriginEXT": "cgl_ray_world_origin",
            "gl_WorldRayDirectionEXT": "cgl_ray_world_direction",
            "gl_ObjectRayOriginEXT": "cgl_ray_object_origin",
            "gl_ObjectRayDirectionEXT": "cgl_ray_object_direction",
            "gl_RayTminEXT": "cgl_ray_t_min",
            "gl_IncomingRayFlagsEXT": "cgl_ray_incoming_flags",
            "gl_InstanceCustomIndexEXT": "cgl_ray_instance_custom_index",
            "gl_GeometryIndexEXT": "cgl_ray_geometry_index",
        }.get(str(name))
        if helper_name is None:
            return None
        self.require_hip_ray_runtime_helpers()
        return f"{helper_name}()"

    def visit_ExpressionStatementNode(self, node) -> str:
        expr = self.visit(node.expression)
        self.add_line(f"{expr};")
        return ""

    def visit_BlockNode(self, node) -> str:
        if hasattr(node, "statements"):
            self.emit_body(node.statements)
        return ""

    def visit_BreakNode(self, node) -> str:
        self.add_line("break;")
        return ""

    def visit_ContinueNode(self, node) -> str:
        self.add_line("continue;")
        return ""

    def visit_EnumNode(self, node) -> str:
        self.add_line(f"enum {node.name}")
        self.add_line("{")
        self.indent_level += 1

        if hasattr(node, "variants") and node.variants:
            for i, variant in enumerate(node.variants):
                if hasattr(variant, "value") and variant.value:
                    value = self.visit(variant.value)
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name} = {value}")
                    else:
                        self.add_line(f"{variant.name} = {value},")
                else:
                    if i == len(node.variants) - 1:
                        self.add_line(f"{variant.name}")
                    else:
                        self.add_line(f"{variant.name},")

        self.indent_level -= 1
        self.add_line("};")
        self.add_line()
        return ""

    def type_name_string(self, type_name):
        """Return a stable string spelling for TypeNode or legacy type values."""
        if type_name is None:
            return None
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            return self.convert_type_node_to_string(type_name)
        return str(type_name)

    def get_parameter_type(self, param):
        """Return a parameter type with HIP resource access metadata applied."""
        param_type = ResourceQueryMixin.get_parameter_type(self, param)
        return self.resource_type_with_access(param_type, param)

    def get_variable_node_type(self, node):
        """Return a variable type with HIP resource access metadata applied."""
        var_type = ResourceQueryMixin.get_variable_node_type(self, node)
        return self.resource_type_with_access(var_type, node)

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "pointee_type"):
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            prefix = "" if getattr(type_node, "is_mutable", True) else "const "
            return f"{prefix}{pointee_type}*"
        if hasattr(type_node, "referenced_type"):
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            prefix = "" if getattr(type_node, "is_mutable", False) else "const "
            return f"{prefix}{referenced_type}&"
        if hasattr(type_node, "name"):
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                converted_args = []
                for arg in generic_args:
                    if (
                        hasattr(arg, "name")
                        or hasattr(arg, "element_type")
                        or hasattr(arg, "pointee_type")
                        or hasattr(arg, "referenced_type")
                    ):
                        converted = self.convert_type_node_to_string(arg)
                    elif hasattr(arg, "value"):
                        converted = str(arg.value)
                    else:
                        converted = str(arg)
                    if converted is None:
                        if hasattr(arg, "value"):
                            converted = str(arg.value)
                        else:
                            converted = str(arg)
                    converted_args.append(converted)
                args = ", ".join(converted_args)
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type"):
            if hasattr(type_node, "rows"):
                element_type = self.convert_type_node_to_string(type_node.element_type)
                prefix = "dmat" if element_type == "double" else "mat"
                return f"{prefix}{type_node.rows}x{type_node.cols}"
            elif not hasattr(type_node, "size"):
                return str(type_node)
            elif str(type(type_node)).find("ArrayType") != -1:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                if type_node.size is not None:
                    return f"{element_type}[{self.format_array_size(type_node.size)}]"
                else:
                    return f"{element_type}[]"
            else:
                element_type = self.convert_type_node_to_string(type_node.element_type)
                size = type_node.size
                if element_type == "float":
                    return f"float{size}"
                elif element_type == "f16":
                    return f"vec{size}<f16>"
                elif element_type == "int":
                    return f"int{size}"
                elif element_type in {"i8", "u8", "i16", "u16"}:
                    return f"vec{size}<{element_type}>"
                else:
                    return f"{element_type}{size}"
        else:
            return str(type_node)

    def map_type(self, type_name) -> str:
        """Map a CrossGL type name or type node to a HIP type string."""
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_str = self.convert_type_node_to_string(type_name)
        else:
            type_str = str(type_name)

        unsupported_type = self.hip_unsupported_fp16_vector_type(type_str)
        if unsupported_type is not None:
            self.raise_unsupported_hip_fp16_vector_type(unsupported_type)

        generic_enum_type = generic_enum_specialized_type_name(self, type_str)
        if generic_enum_type is not None:
            return generic_enum_type

        generic_struct_type = generic_struct_specialized_type_name(self, type_str)
        if generic_struct_type is not None:
            return generic_struct_type

        if type_str in getattr(self, "enum_type_names", set()):
            return "int"

        if type_str in getattr(self, "enum_struct_type_names", set()):
            return type_str

        geometry_stream_type = self.hip_geometry_stream_mapped_type(type_str)
        if geometry_stream_type is not None:
            return geometry_stream_type

        tessellation_patch_type = self.hip_tessellation_patch_mapped_type(type_str)
        if tessellation_patch_type is not None:
            return tessellation_patch_type

        if "[" in type_str and "]" in type_str:
            base_type = type_str.split("[")[0]
            array_part = type_str[type_str.find("[") :]
            canonical_base = self.canonical_resource_type(base_type)
            mapped_base = self.map_type(canonical_base or base_type)
            return f"{mapped_base}{array_part}"

        structured_buffer_type = self.hip_structured_buffer_type(type_str)
        if structured_buffer_type is not None:
            return structured_buffer_type

        byte_address_buffer_type = self.hip_byte_address_buffer_type(type_str)
        if byte_address_buffer_type is not None:
            return byte_address_buffer_type

        canonical_type = self.canonical_resource_type(type_str)
        mapped_type = self.type_map.get(canonical_type or type_str, type_str)
        return self.hip_mapped_type_result(mapped_type)

    def hip_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, generic_arg = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None
        generic_arg = generic_arg[:-1].strip()
        if not generic_arg:
            return None
        return base_name, self.map_type(generic_arg)

    def hip_geometry_stream_mapped_type(self, type_name):
        stream_info = self.hip_geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        return f"CglHip{stream_name}<{output_type}>"

    def format_hip_geometry_stream_parameter(self, raw_param_type, name):
        stream_type = self.hip_geometry_stream_mapped_type(raw_param_type)
        if stream_type is None:
            return None
        return f"{stream_type}& {name}"

    def hip_tessellation_patch_type_info(self, type_name):
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None
        base_name, args = parts
        if base_name not in {"InputPatch", "OutputPatch"} or len(args) != 2:
            return None
        element_type, point_count = args
        if not element_type or not point_count:
            return None
        mapped_element_type = self.map_type(element_type)
        return base_name, mapped_element_type, point_count

    def hip_tessellation_patch_mapped_type(self, type_name):
        patch_info = self.hip_tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        base_name, element_type, point_count = patch_info
        helper_name = {
            "InputPatch": "CglHipInputPatch",
            "OutputPatch": "CglHipOutputPatch",
        }[base_name]
        return f"{helper_name}<{element_type}, {point_count}>"

    def hip_mapped_type_result(self, mapped_type):
        base_type = str(mapped_type).split("[", 1)[0].strip()
        if base_type == "cgl_half3":
            self.require_hip_half3_helper()
        if base_type in {
            "CglRayTracingAccelerationStructure",
            "CglRayDesc",
            "CglRayQuery",
            "CglBuiltInTriangleIntersectionAttributes",
        }:
            self.require_hip_ray_runtime_helpers()
        return mapped_type

    def attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "name") and value.name is not None:
            return str(value.name)
        if hasattr(value, "value") and value.value is not None:
            return str(value.value).strip('"')
        return str(value)

    def attribute_arguments(self, attr):
        return getattr(attr, "arguments", getattr(attr, "args", [])) or []

    def resource_binding_index_value(self, value, prefixes=()):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        for prefix in prefixes:
            if raw_value.startswith(prefix) and raw_value[len(prefix) :].isdigit():
                return int(raw_value[len(prefix) :])
        return None

    def resource_register_space_value(self, value):
        raw_value = self.attribute_value_to_string(value)
        if raw_value is None:
            return None
        raw_value = str(raw_value).strip().lower()
        if raw_value.isdigit():
            return int(raw_value)
        if raw_value.startswith("space") and raw_value[5:].isdigit():
            return int(raw_value[5:])
        return None

    def explicit_hip_resource_binding_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return binding
        return None

    def explicit_hip_resource_set_index(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if attr_name in {"set", "group"} and arguments:
                set_index = self.resource_binding_index_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "space" and arguments:
                set_index = self.resource_register_space_value(arguments[0])
                if set_index is not None:
                    return set_index
            if attr_name == "register":
                for argument in arguments[1:]:
                    set_index = self.resource_register_space_value(argument)
                    if set_index is not None:
                        return set_index
        return 0

    def hip_resource_register_metadata(self, node):
        for attr in getattr(node, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "register":
                continue
            values = [
                self.attribute_value_to_string(argument)
                for argument in self.attribute_arguments(attr)
            ]
            values = [value for value in values if value]
            if values:
                return ",".join(values)
        return None

    def hip_explicit_resource_binding_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            if attr_name in {"binding", "buffer", "sampler", "texture", "uav"}:
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            elif attr_name == "register":
                binding = self.resource_binding_index_value(
                    arguments[0], ("b", "s", "t", "u")
                )
            else:
                binding = None
            if binding is not None:
                return attr_name
        return None

    def hip_resource_base_type_and_count(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name, 1
        if "[" not in type_name or "]" not in type_name:
            return type_name, 1
        base_type = type_name.split("[", 1)[0].strip()
        suffix = type_name[type_name.find("[") :]
        count = 1
        while suffix.startswith("["):
            close_bracket = suffix.find("]")
            if close_bracket < 0:
                break
            size_text = suffix[1:close_bracket].strip()
            if size_text.isdigit():
                count *= int(size_text)
            suffix = suffix[close_bracket + 1 :]
        return base_type, count

    def hip_resource_kind(self, type_name, node=None, forced_kind=None):
        if forced_kind is not None:
            return forced_kind
        if node is not None and self.is_glsl_buffer_block_node(node):
            return "glsl_buffer_block"

        base_type, _count = self.hip_resource_base_type_and_count(type_name)
        if not base_type:
            return None
        generic_parts = self.generic_type_parts(base_type)
        base_name = generic_parts[0] if generic_parts is not None else base_type
        base_name = base_name.rsplit("::", 1)[-1]

        if (
            self.structured_buffer_type_parts(base_type) is not None
            or self.byte_address_buffer_base_type(base_type) is not None
        ):
            return "buffer"

        canonical = self.canonical_resource_type(base_name) or base_name
        canonical = canonical.rsplit("::", 1)[-1]
        mapped_type = self.type_map.get(canonical, canonical)
        mapped_base = mapped_type.split("<", 1)[0].rsplit("::", 1)[-1]
        if mapped_base == "CglRayTracingAccelerationStructure":
            return "acceleration_structure"
        if canonical in {"sampler", "SamplerState"}:
            return "sampler"
        if canonical.startswith("sampler"):
            return "texture"
        if canonical.startswith(("image", "iimage", "uimage")):
            return "image"
        return None

    def hip_resource_binding_namespace(self, kind):
        if kind in {"cbuffer", "glsl_buffer_block"}:
            return "buffer"
        return kind

    def hip_resource_binding_validation_count(self, node, resource_kind, count):
        if (
            resource_kind in {"image", "sampler", "texture"}
            and count != 1
            and self.hip_explicit_resource_binding_attribute(node) == "binding"
        ):
            return 1
        return count

    def hip_resource_binding_range_conflicts(self, key, binding, count):
        end = binding + count - 1
        for used_start, used_end, _used_name in self.hip_used_resource_bindings.get(
            key, []
        ):
            if binding <= used_end and used_start <= end:
                return True
        return False

    def next_available_hip_resource_binding(self, namespace, set_index, count):
        key = (namespace, set_index)
        binding = self.hip_resource_binding_cursors.get(key, 0)
        while self.hip_resource_binding_range_conflicts(key, binding, count):
            binding += 1
        self.hip_resource_binding_cursors[key] = binding + count
        return binding

    def reserve_hip_resource_binding(self, namespace, set_index, binding, count, name):
        key = (namespace, set_index)
        end = binding + count - 1
        ranges = self.hip_used_resource_bindings.setdefault(key, [])
        for used_start, used_end, used_name in ranges:
            if binding <= used_end and used_start <= end:
                if used_start == binding and used_end == end and used_name == name:
                    return
                raise ValueError(
                    "Conflicting HIP resource binding for "
                    f"'{name}': {namespace} set {set_index} binding "
                    f"{binding}-{end} overlaps '{used_name}' binding "
                    f"{used_start}-{used_end}"
                )
        ranges.append((binding, end, name))
        self.hip_resource_binding_cursors[key] = max(
            self.hip_resource_binding_cursors.get(key, 0), end + 1
        )

    def hip_resource_metadata_comment(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        if not name or resource_kind is None:
            return ""

        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        validation_count = self.hip_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        binding = self.explicit_hip_resource_binding_index(node)
        if binding is None:
            binding = self.next_available_hip_resource_binding(
                namespace, set_index, validation_count
            )
            binding_source = "automatic"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, validation_count, name
            )
        else:
            binding_source = "explicit"
            self.reserve_hip_resource_binding(
                namespace, set_index, binding, validation_count, name
            )

        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            f"kind={resource_kind}",
        ]
        if resource_kind == "glsl_buffer_block":
            layout = self.glsl_buffer_block_layout(node)
            if layout:
                parts.append(f"layout={layout}")
            access = self.explicit_resource_access(node)
            if access:
                parts.append(f"access={access}")
        parts.extend(
            [
                f"set={set_index}",
                f"binding={binding}",
                f"binding_source={binding_source}",
            ]
        )
        if count != 1:
            parts.append(f"count={count}")
        register_metadata = self.hip_resource_register_metadata(node)
        if register_metadata:
            parts.append(f"register={register_metadata}")
        return " ".join(parts)

    def reserve_explicit_hip_resource_binding(self, node, type_name, kind=None):
        name = getattr(node, "name", None)
        resource_kind = self.hip_resource_kind(type_name, node=node, forced_kind=kind)
        binding = self.explicit_hip_resource_binding_index(node)
        if not name or resource_kind is None or binding is None:
            return
        _base_type, count = self.hip_resource_base_type_and_count(type_name)
        validation_count = self.hip_resource_binding_validation_count(
            node, resource_kind, count
        )
        namespace = self.hip_resource_binding_namespace(resource_kind)
        set_index = self.explicit_hip_resource_set_index(node)
        self.reserve_hip_resource_binding(
            namespace, set_index, binding, validation_count, name
        )

    def reserve_explicit_hip_resource_bindings(self, ast):
        for node in getattr(ast, "global_variables", []) or []:
            type_name = self.get_variable_node_type(node) or "float"
            self.reserve_explicit_hip_resource_binding(node, type_name)
        for cbuffer in getattr(ast, "cbuffers", []) or []:
            self.reserve_explicit_hip_resource_binding(
                cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
            )
        stages = getattr(ast, "stages", {}) or {}
        for stage in stages.values():
            for node in getattr(stage, "local_variables", []) or []:
                type_name = self.get_variable_node_type(node) or "float"
                self.reserve_explicit_hip_resource_binding(node, type_name)
            for cbuffer in getattr(stage, "local_cbuffers", []) or []:
                self.reserve_explicit_hip_resource_binding(
                    cbuffer, getattr(cbuffer, "name", None), kind="cbuffer"
                )

    def explicit_resource_access(self, node):
        if node is None:
            return None

        access_names = {
            "read": "readonly",
            "readonly": "readonly",
            "write": "writeonly",
            "writeonly": "writeonly",
            "read_write": "readwrite",
            "readwrite": "readwrite",
            "access::read": "readonly",
            "access::write": "writeonly",
            "access::read_write": "readwrite",
        }
        for qualifier in getattr(node, "qualifiers", []) or []:
            access = access_names.get(str(qualifier).lower())
            if access is not None:
                return access

        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "access":
                arguments = getattr(attr, "arguments", []) or []
                if not arguments:
                    continue
                raw_access = self.attribute_value_to_string(arguments[0])
                access = access_names.get(str(raw_access).lower())
            else:
                access = access_names.get(attr_name)
            if access is not None:
                return access
        return None

    def is_glsl_buffer_block_node(self, node):
        attributes = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(node, "attributes", []) or []
        }
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "glsl_buffer_block" in attributes:
            return True
        return "buffer" in qualifiers and bool(
            attributes & {"std140", "std430", "scalar"}
        )

    def glsl_buffer_block_layout(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", "")).lower()
            if attr_name == "glsl_buffer_block":
                arguments = getattr(attr, "arguments", []) or []
                if arguments:
                    return self.attribute_value_to_string(arguments[0])
            if attr_name in {"std140", "std430", "scalar"}:
                return attr_name
        return None

    def glsl_buffer_block_declaration_type(self, type_name, node):
        type_name = self.type_name_string(type_name)
        if not self.is_glsl_buffer_block_node(node):
            return type_name
        if not type_name or "[" not in type_name or "]" not in type_name:
            return type_name
        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket].strip()
        array_suffix = type_name[open_bracket:]
        if array_suffix != "[]":
            return type_name
        if self.explicit_resource_access(node) == "readonly":
            return f"const {base_type}{array_suffix}"
        return type_name

    def glsl_buffer_block_metadata_comment(self, node, type_name):
        name = getattr(node, "name", None)
        if not name or not self.is_glsl_buffer_block_node(node):
            return ""
        parts = [
            "// CrossGL resource metadata:",
            f"name={name}",
            "kind=glsl_buffer_block",
        ]
        layout = self.glsl_buffer_block_layout(node)
        if layout:
            parts.append(f"layout={layout}")
        access = self.explicit_resource_access(node)
        if access:
            parts.append(f"access={access}")
        return " ".join(parts)

    def glsl_buffer_block_root_name(self, expr):
        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.glsl_buffer_block_root_name(array_expr)
        if isinstance(expr, MemberAccessNode):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.glsl_buffer_block_root_name(object_expr)
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, str):
            return expr
        return getattr(expr, "name", None)

    def glsl_buffer_block_access(self, expr):
        root_name = self.glsl_buffer_block_root_name(expr)
        if root_name is None:
            return None, None
        return self.glsl_buffer_block_accesses.get(root_name), root_name

    def glsl_buffer_block_diagnostic(self, operation, expr, reason, fallback):
        _, resource_name = self.glsl_buffer_block_access(expr)
        resource_name = resource_name or "unknown"
        return (
            f"/* unsupported {self.resource_backend_name()} GLSL buffer block "
            f"{operation}: resource '{resource_name}' {reason} */ {fallback}"
        )

    def glsl_buffer_block_read_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "writeonly":
            return None
        fallback = self.diagnostic_zero_value_for_type(
            self.expression_result_type(expr)
        )
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is writeonly", fallback
        )

    def glsl_buffer_block_write_diagnostic(self, expr, operation):
        access, _ = self.glsl_buffer_block_access(expr)
        if access != "readonly":
            return None
        return self.glsl_buffer_block_diagnostic(
            operation, expr, "is readonly", "((void)0)"
        )

    def glsl_buffer_block_read_write_diagnostic(self, expr, operation, fallback):
        access, _ = self.glsl_buffer_block_access(expr)
        if access == "readonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is readonly", fallback
            )
        if access == "writeonly":
            return self.glsl_buffer_block_diagnostic(
                operation, expr, "is writeonly", fallback
            )
        return None

    def apply_readonly_qualifier_to_type(self, type_name, node):
        """Apply readonly declaration qualifiers to pointer/reference types."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return type_name

        if self.hip_pointer_or_reference_type_info(type_name) is None:
            return type_name

        if not (
            self.declaration_qualifier_names(node) & {"const", "readonly", "constant"}
        ):
            return type_name

        if type_name.startswith("const "):
            return type_name
        return f"const {type_name}"

    def resource_type_with_access(self, type_name, node):
        type_name = self.apply_readonly_qualifier_to_type(type_name, node)
        if not type_name:
            return type_name

        access = self.explicit_resource_access(node)
        if access is None:
            return type_name

        base_type = type_name
        array_suffix = ""
        if "[" in type_name and "]" in type_name:
            open_bracket = type_name.find("[")
            base_type = type_name[:open_bracket]
            array_suffix = type_name[open_bracket:]

        parts = self.structured_buffer_type_parts(base_type)
        if parts is not None:
            base_name, element_type = parts
            if base_name in {"StructuredBuffer", "RWStructuredBuffer"}:
                mapped_base = (
                    "StructuredBuffer" if access == "readonly" else "RWStructuredBuffer"
                )
                return f"{mapped_base}<{element_type}>{array_suffix}"

        byte_base = self.byte_address_buffer_base_type(base_type)
        if byte_base in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            mapped_base = (
                "ByteAddressBuffer" if access == "readonly" else "RWByteAddressBuffer"
            )
            return f"{mapped_base}{array_suffix}"

        return type_name

    def generic_type_parts(self, type_name):
        """Split a generic type name into base name and top-level arguments."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None

        base_type = type_name.split("[", 1)[0].strip()
        generic_start = base_type.find("<")
        generic_end = base_type.rfind(">")
        if generic_start == -1 or generic_end < generic_start:
            return None

        base_name = base_type[:generic_start].strip()
        args_text = base_type[generic_start + 1 : generic_end].strip()
        args = []
        depth = 0
        current = []
        for char in args_text:
            if char == "<":
                depth += 1
                current.append(char)
            elif char == ">":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        trailing_arg = "".join(current).strip()
        if trailing_arg:
            args.append(trailing_arg)
        return base_name, args

    def structured_buffer_type_parts(self, type_name):
        """Return structured-buffer base and element type, if applicable."""
        parts = self.generic_type_parts(type_name)
        if parts is None:
            return None

        base_name, args = parts
        if (
            base_name
            not in {
                "StructuredBuffer",
                "RWStructuredBuffer",
                "AppendStructuredBuffer",
                "ConsumeStructuredBuffer",
            }
            or not args
        ):
            return None
        return base_name, args[0]

    def structured_buffer_is_writable(self, type_name):
        """Return whether a structured-buffer type permits writes."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] == "RWStructuredBuffer"

    def structured_buffer_requires_counter(self, type_name):
        """Return whether a structured-buffer type needs an explicit counter."""
        parts = self.structured_buffer_type_parts(type_name)
        return parts is not None and parts[0] in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def hip_structured_buffer_type(self, type_name):
        """Map structured-buffer resources to HIP pointer types."""
        parts = self.structured_buffer_type_parts(type_name)
        if parts is None:
            return None

        base_name, element_type = parts
        hip_element_type = self.map_type(element_type)
        if base_name in {"StructuredBuffer", "ConsumeStructuredBuffer"}:
            return f"const {hip_element_type}*"
        return f"{hip_element_type}*"

    def byte_address_buffer_base_type(self, type_name):
        """Return the byte-address buffer base type, if applicable."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        base_type = type_name.split("[", 1)[0].strip()
        if base_type in {"ByteAddressBuffer", "RWByteAddressBuffer"}:
            return base_type
        return None

    def byte_address_buffer_is_writable(self, type_name):
        """Return whether a byte-address buffer type permits writes."""
        return self.byte_address_buffer_base_type(type_name) == "RWByteAddressBuffer"

    def hip_byte_address_buffer_type(self, type_name):
        """Map ByteAddressBuffer and RWByteAddressBuffer to HIP byte pointers."""
        base_type = self.byte_address_buffer_base_type(type_name)
        if base_type == "ByteAddressBuffer":
            return "const unsigned char*"
        if base_type == "RWByteAddressBuffer":
            return "unsigned char*"
        return None

    def array_access_element_type(self, type_name):
        """Return the element type for HIP arrays and structured buffers."""
        array_element_type = super().array_access_element_type(type_name)
        if array_element_type is not None:
            return array_element_type

        indirect_info = self.hip_indirect_type_info(type_name)
        if indirect_info is not None and indirect_info["kind"] == "pointer":
            return indirect_info["pointee_type"]

        parts = self.structured_buffer_type_parts(type_name)
        if parts is not None:
            return parts[1]
        return None

    def resource_call_result_type(self, func_name, raw_args):
        if func_name in {"texelFetch", "texelFetchOffset"} and raw_args:
            resource_type = self.resource_base_type(
                self.resource_expression_type(raw_args[0])
            )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            if self.is_sampled_texture_family_type(resource_type):
                return self.sampled_texture_value_type(resource_type)

        if func_name in {
            "textureProj",
            "textureProjLod",
            "textureProjGrad",
            "texelFetchOffset",
        }:
            resource_type = None
            if raw_args:
                resource_type = self.resource_base_type(
                    self.get_expression_type(raw_args[0])
                )
            if self.is_shadow_resource_type(resource_type):
                return "float"
            return "float4"
        return super().resource_call_result_type(func_name, raw_args)

    def expression_result_type(self, node):
        """Infer expression result types with HIP buffer operations."""
        if isinstance(node, (IdentifierNode, VariableNode)):
            name = getattr(node, "name", None)
            if not self.source_identifier_shadows_builtin(name):
                builtin_type = self.hip_compute_builtin_type(name)
                if builtin_type is not None:
                    return builtin_type
        if isinstance(node, WaveOpNode):
            return self.wave_result_type(
                getattr(node, "operation", ""), getattr(node, "arguments", []) or []
            )
        if isinstance(node, RayTracingOpNode):
            if getattr(node, "operation", "") == "ReportHit":
                return "bool"
            return None
        if isinstance(node, RayQueryOpNode):
            operation = getattr(node, "operation", "")
            if operation == "Proceed":
                return "bool"
            if operation in {"CandidateRayT", "CommittedRayT"}:
                return "float"
            return "uint"
        if isinstance(node, FunctionCallNode):
            function_expr = getattr(node, "function", getattr(node, "name", None))
            func_name = getattr(function_expr, "name", function_expr)
            if self.is_user_defined_function(func_name):
                return self.function_return_types.get(func_name)
            raw_args = getattr(node, "arguments", getattr(node, "args", [])) or []
            if func_name in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES and len(raw_args) <= 1:
                return func_name
            if func_name in HIP_WAVE_OP_ARITIES:
                return self.wave_result_type(
                    func_name,
                    raw_args,
                )
            if func_name == "ReportHit":
                return "bool"
            if (
                func_name in {"inverseSqrt", "rsqrt"}
                and raw_args
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0])
            bitcast_result_type = self.hip_bitcast_result_type(func_name, raw_args)
            if bitcast_result_type is not None:
                return bitcast_result_type
            integer_bit_result_type = self.hip_integer_bit_result_type(
                func_name,
                raw_args,
            )
            if integer_bit_result_type is not None:
                return integer_bit_result_type
            if func_name in {"fma", "mad"}:
                cached_type = getattr(node, "expression_type", None) or getattr(
                    node, "vtype", None
                )
                if cached_type is not None:
                    return cached_type
                return self.fused_multiply_add_result_type(raw_args)
            if (
                func_name == "lerp"
                and len(raw_args) == 3
                and not self.is_user_defined_function(func_name)
            ):
                return self.expression_result_type(raw_args[0]) or (
                    self.expression_result_type(raw_args[1])
                    if len(raw_args) > 1
                    else None
                )
            if func_name == "step" and len(raw_args) == 2:
                return self.step_result_type(raw_args)
            buffer_result_type = self.buffer_call_result_type(node)
            if buffer_result_type is not None:
                return buffer_result_type
        if isinstance(node, MemberAccessNode):
            object_node = getattr(
                node,
                "object_expr",
                getattr(node, "object", None),
            )
            object_name = getattr(object_node, "name", None)
            if not self.source_identifier_shadows_builtin(object_name):
                builtin_member_type = self.hip_compute_builtin_member_type(
                    object_name,
                    getattr(node, "member", ""),
                )
                if builtin_member_type is not None:
                    return builtin_member_type
            member_type = self.member_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, PointerAccessNode):
            member_type = self.pointer_access_member_type(node)
            if member_type is not None:
                return member_type
        if isinstance(node, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(node, "constructor_type", getattr(node, "vtype", None))
            )
            return (
                infer_enum_constructor_type(self, node)
                or infer_struct_constructor_type(self, node)
                or (
                    constructor_type
                    if constructor_type in HIP_SCALAR_CONSTRUCTOR_TYPE_ALIASES
                    else None
                )
            )
        if isinstance(node, MatchNode):
            return infer_match_expression_result_type(self, node)
        return super().expression_result_type(node)

    def step_result_type(self, raw_args):
        edge_type = self.expression_result_type(raw_args[0])
        value_type = self.expression_result_type(raw_args[1])
        value_info = self.vector_type_info(value_type)
        if value_info is not None:
            return value_type
        edge_info = self.vector_type_info(edge_type)
        if edge_info is not None:
            return edge_type
        return self.step_scalar_type(raw_args)

    def struct_member_lookup_type(self, type_name):
        """Return the struct key after HIP pointer/reference wrappers."""
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        indirect_info = self.hip_indirect_type_info(type_name)
        if indirect_info is not None:
            return indirect_info["pointee_type"]
        return type_name

    def member_access_member_type(self, node):
        """Return the member type for object.field expressions."""
        object_node = getattr(
            node,
            "object_expr",
            getattr(node, "object", None),
        )
        object_type = self.struct_member_lookup_type(
            self.expression_result_type(object_node)
        )
        return self.struct_member_types.get(object_type, {}).get(
            getattr(node, "member", "")
        )

    def pointer_access_member_type(self, node):
        """Return the member type for ptr->field expressions."""
        pointer_expr = getattr(node, "pointer_expr", None)
        pointer_info = self.hip_indirect_type_info(
            self.expression_result_type(pointer_expr)
        )
        if pointer_info is None:
            return None

        struct_type = self.struct_member_lookup_type(pointer_info["pointee_type"])
        return self.struct_member_types.get(struct_type, {}).get(
            getattr(node, "member", "")
        )

    def buffer_call_result_type(self, node):
        """Infer result type for structured and byte-address buffer read calls."""
        function_expr = getattr(node, "function", getattr(node, "name", None))
        raw_args = getattr(node, "arguments", getattr(node, "args", []))

        byte_member_call = self.byte_address_buffer_member_call(function_expr)
        if byte_member_call is not None:
            _, operation, _ = byte_member_call
            if operation == "GetDimensions":
                return "uint"
            if operation in self.byte_address_buffer_atomic_operations():
                return "uint"
            if operation.startswith("Load") and raw_args:
                component_count = self.byte_address_buffer_component_count(operation)
                if component_count is not None:
                    return self.byte_address_buffer_value_type(component_count)
            return None

        member_call = self.structured_buffer_member_call(function_expr)
        if member_call is not None:
            _, operation, buffer_type = member_call
            if operation == "Load" and raw_args:
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "Consume":
                return self.structured_buffer_type_parts(buffer_type)[1]
            if operation == "GetDimensions" and not raw_args:
                return "uint"
            return None

        func_name = getattr(function_expr, "name", function_expr)
        if func_name == "buffer_load" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.byte_address_buffer_base_type(buffer_type) is not None:
                return "uint"
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_consume" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            parts = self.structured_buffer_type_parts(buffer_type)
            if parts is not None:
                return parts[1]
        if func_name == "buffer_dimensions" and raw_args:
            buffer_type = self.expression_result_type(raw_args[0])
            if self.buffer_type_supports_length(buffer_type):
                return "uint"
        if func_name in self.structured_buffer_atomic_operations() and raw_args:
            target = self.structured_buffer_atomic_target(raw_args[0])
            if target is not None:
                return target["target_type"]
            target_type = self.expression_result_type(raw_args[0])
            if self.hip_atomic_scalar_kind(target_type) is not None:
                return target_type
        return None

    def canonical_sampled_resource_type(self, type_name):
        """Return the sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return the image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def canonical_resource_type(self, type_name):
        """Return the canonical sampler/image resource type for an alias."""
        if isinstance(type_name, str):
            base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
            if base_type in {
                "accelerationStructureEXT",
                "AccelerationStructure",
                "acceleration_structure",
                "RaytracingAccelerationStructure",
                "RayTracingAccelerationStructure",
            }:
                return "RayTracingAccelerationStructure"
            sampler_type = self.sampler_state_type_aliases.get(base_type)
            if sampler_type is not None:
                return sampler_type
        return self.canonical_sampled_resource_type(
            type_name
        ) or self.canonical_storage_resource_type(type_name)

    def dimension_query_spec(self, type_name):
        """Return HIP-specific resource metadata dimensions before shared specs."""
        base_type = self.resource_base_type(type_name)
        hip_specs = {
            "image1D": (("width",), False, False),
            "iimage1D": (("width",), False, False),
            "uimage1D": (("width",), False, False),
            "image1DArray": (("width", "elements"), False, False),
            "iimage1DArray": (("width", "elements"), False, False),
            "uimage1DArray": (("width", "elements"), False, False),
            "iimageCube": (("width", "height"), False, False),
            "uimageCube": (("width", "height"), False, False),
            "imageCubeArray": (("width", "height", "elements"), False, False),
            "iimageCubeArray": (("width", "height", "elements"), False, False),
            "uimageCubeArray": (("width", "height", "elements"), False, False),
        }
        spec = hip_specs.get(base_type)
        if spec is None:
            return super().dimension_query_spec(base_type)
        dimensions, mip, samples = spec
        return {"dimensions": dimensions, "mip": mip, "samples": samples}

    def resource_base_type(self, type_name):
        """Normalize resource aliases before resource dispatch decisions."""
        base_type = ResourceDiagnosticMixin.resource_base_type(self, type_name)
        return self.canonical_resource_type(base_type) or base_type

    def format_typed_declarator(self, type_name, name, dynamic_array_as_pointer=True):
        if (
            hasattr(type_name, "name")
            or hasattr(type_name, "element_type")
            or hasattr(type_name, "pointee_type")
            or hasattr(type_name, "referenced_type")
        ):
            type_name = self.convert_type_node_to_string(type_name)
        else:
            type_name = str(type_name)

        if "[" not in type_name or "]" not in type_name:
            return f"{self.map_type(type_name)} {name}"

        open_bracket = type_name.find("[")
        base_type = type_name[:open_bracket]
        array_suffix = type_name[open_bracket:]
        mapped_base = self.map_type(base_type)

        return format_array_declarator(
            mapped_base,
            name,
            array_suffix,
            dynamic_array_as_pointer=dynamic_array_as_pointer,
        )

    def format_array_size(self, size):
        if size is None:
            return ""
        if isinstance(size, int):
            return str(size)
        return self.visit(size)

    def generate_kernel_wrapper(self, kernel_node: FunctionNode) -> str:
        """Generate a host-side HIP launch wrapper for a kernel node."""
        wrapper_lines = []

        wrapper_name = f"launch_{kernel_node.name}"
        params = []
        args = []

        for param in kernel_node.parameters:
            param_type = self.map_type(param.param_type)
            params.append(f"{param_type} {param.name}")
            args.append(param.name)

        params.extend(["dim3 gridSize", "dim3 blockSize", "hipStream_t stream = 0"])

        wrapper_lines.extend(
            [
                f"void {wrapper_name}({', '.join(params)})",
                "{",
                f"    hipLaunchKernelGGL({kernel_node.name}, gridSize, blockSize, 0, stream, {', '.join(args)});",
                "}",
            ]
        )

        return "\n".join(wrapper_lines)


def generate_hip_code(ast: ShaderNode) -> str:
    """Generate HIP source from a CrossGL shader AST."""
    generator = HipCodeGen()
    return generator.generate(ast)
