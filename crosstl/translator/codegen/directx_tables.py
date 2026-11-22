"""Class-level tables of the HLSL generator: minimum-precision and native 16-bit types, interpolation modifiers, derivative and wave intrinsic sets, software subgroup operations, reserved identifiers and non-uniform resource index names."""

from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from .barrier_peephole import HLSL_EXECUTION_BARRIER_PLACEHOLDER
from .hlsl_refusals import DirectXLoweringRefused
from .software_quad_reductions import (
    QUAD_BUILTIN_OPERATORS,
    SOFTWARE_QUAD_REDUCTIONS,
)


class DirectXTableMixin:
    """Table members of ``HLSLCodeGen``."""

    LOWERING_REFUSAL_TYPE = DirectXLoweringRefused
    # The type names the program being rendered declares, and the foreign built-in type names its rendering has
    # written so far (``None`` outside a rendering); see :func:`written_type_name`.
    program_type_names = frozenset()
    written_foreign_type_names = None

    DIRECTX_COOPERATIVE_MATRIX_COMPONENT_TYPES = {
        "float": "float",
        "f32": "float",
        "int": "int",
        "i32": "int",
        "int32_t": "int",
        "uint": "uint",
        "u32": "uint",
        "uint32_t": "uint",
    }
    DIRECTX_COOPERATIVE_MATRIX_LANE_LOCAL_OPERATIONS = frozenset(
        {
            "element",
            "negate",
            "elementwise_add",
            "elementwise_subtract",
            "elementwise_multiply",
        }
    )
    DIRECTX_COOPERATIVE_MATRIX_CONTRACT_FIELDS = (
        "component_type",
        "rows",
        "columns",
        "scope",
        "use",
        "layout",
        "fragment_layout",
        "subgroup_size",
        "elements_per_lane",
        "fragment_provenance",
        "fragment_mapping",
        "fragment_mapping_provenance",
    )

    HLSL_CLANG_TRAILING_ZERO_BUILTINS = frozenset(
        {"__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll"}
    )
    HLSL_CLANG_TRAILING_ZERO_WIDTHS = {
        "__builtin_ctz": 32,
        "__builtin_ctzl": 64,
        "__builtin_ctzll": 64,
    }
    HLSL_NARROW_INTEGER_TYPE_NAMES = frozenset(
        {
            "char",
            "signed char",
            "unsigned char",
            "uchar",
            "i8",
            "u8",
            "int8",
            "uint8",
            "int8_t",
            "uint8_t",
            "short",
            "signed short",
            "unsigned short",
            "ushort",
            "i16",
            "u16",
            "int16",
            "uint16",
            "int16_t",
            "uint16_t",
            "min12int",
            "min16int",
            "min16uint",
        }
    )
    HLSL_MINIMUM_PRECISION_INTEGER_BASE_TYPES = frozenset(
        {"min12int", "min16int", "min16uint"}
    )
    HLSL_NATIVE_16_BIT_ARITHMETIC_BASE_TYPES = frozenset(
        {"float16_t", "int16_t", "uint16_t"}
    )
    HLSL_INTEGER_ARITHMETIC_BASE_TYPES = (
        "uint64_t",
        "int64_t",
        "uint16_t",
        "int16_t",
        "min16uint",
        "min16int",
        "min12int",
        "uint",
        "int",
    )
    # Exact binary32 conversion helpers for 64-bit integer lanes.
    HLSL_WIDE_INTEGER_FLOAT_HELPER_STEMS = {
        "uint64_t": "__crossgl_uint64_to_float",
        "int64_t": "__crossgl_int64_to_float",
    }
    HLSL_PATCH_CONTROL_POINT_LIMIT = 32
    METAL_ONLY_SYSTEM_INCLUDES = frozenset(
        {
            "metal_atomic",
            "metal_atomic.h",
            "metal_integer",
            "metal_integer.h",
            "metal_math",
            "metal_math.h",
            "metal_stdlib",
            "metal_stdlib.h",
            # Apple's platform-conditional macros, which shared Metal and host headers include.
            "targetconditionals.h",
        }
    )
    METAL_ADDRESS_SPACE_METADATA_ATTRIBUTES = frozenset(
        {"constant", "device", "thread", "threadgroup"}
    )
    METAL_TYPE_ALIAS_GLOBALS = frozenset({"bfloat16_t", "float16_t"})
    HLSL_SOURCE_BFLOAT16_TYPES = frozenset({"bfloat", "bfloat16", "bfloat16_t"})
    HLSL_BFLOAT16_BUILTIN_CONTRACTS = {
        "abs": ("abs", 1, "bfloat"),
        "acos": ("acos", 1, "bfloat"),
        "asin": ("asin", 1, "bfloat"),
        "atan": ("atan", 1, "bfloat"),
        "atan2": ("atan2", 2, "bfloat"),
        "ceil": ("ceil", 1, "bfloat"),
        "cos": ("cos", 1, "bfloat"),
        "cosh": ("cosh", 1, "bfloat"),
        "exp": ("exp", 1, "bfloat"),
        "fabs": ("abs", 1, "bfloat"),
        "floor": ("floor", 1, "bfloat"),
        "fmod": ("fmod", 2, "bfloat"),
        "isfinite": ("isfinite", 1, "bool"),
        "isinf": ("isinf", 1, "bool"),
        "isnan": ("isnan", 1, "bool"),
        "log": ("log", 1, "bfloat"),
        "log10": ("log10", 1, "bfloat"),
        "log2": ("log2", 1, "bfloat"),
        "max": ("max", 2, "bfloat"),
        "min": ("min", 2, "bfloat"),
        "pow": ("pow", 2, "bfloat"),
        "rint": ("round", 1, "bfloat"),
        "rsqrt": ("rsqrt", 1, "bfloat"),
        "sin": ("sin", 1, "bfloat"),
        "sinh": ("sinh", 1, "bfloat"),
        "sqrt": ("sqrt", 1, "bfloat"),
        "tan": ("tan", 1, "bfloat"),
        "tanh": ("tanh", 1, "bfloat"),
        "trunc": ("trunc", 1, "bfloat"),
    }
    HLSL_NATIVE_16_BIT_STORAGE_RESOURCES = frozenset(
        {
            "AppendStructuredBuffer",
            "Buffer",
            "ConsumeStructuredBuffer",
            "RasterizerOrderedBuffer",
            "RasterizerOrderedStructuredBuffer",
            "RWBuffer",
            "RWStructuredBuffer",
            "StructuredBuffer",
        }
    )
    HLSL_CANONICAL_NARROW_FLOAT_CONSTRUCTORS = frozenset({"bfloat", "bfloat16"})
    HLSL_COPYSIGN_NAMES = frozenset(
        {"copysign", "metal::copysign", "metal_u3a_u3acopysign"}
    )
    HLSL_SIGNBIT_NAMES = frozenset(
        {"signbit", "metal::signbit", "metal_u3a_u3asignbit"}
    )
    HLSL_COMPONENTWISE_FLOAT_UNARY_FUNCTIONS = frozenset(
        {
            "abs",
            "acos",
            "asin",
            "atan",
            "ceil",
            "cos",
            "cosh",
            "degrees",
            "exp",
            "exp2",
            "floor",
            "frac",
            "log",
            "log2",
            "radians",
            "round",
            "rsqrt",
            "saturate",
            "sign",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "trunc",
        }
    )
    HLSL_INVERSE_HYPERBOLIC_NAMES = frozenset({"acosh", "asinh", "atanh"})
    HLSL_INVERSE_HYPERBOLIC_ALIASES = {
        "acosh": "acosh",
        "asinh": "asinh",
        "atanh": "atanh",
        "metal::acosh": "acosh",
        "metal::asinh": "asinh",
        "metal::atanh": "atanh",
        "metal_u3a_u3aacosh": "acosh",
        "metal_u3a_u3aasinh": "asinh",
        "metal_u3a_u3aatanh": "atanh",
    }
    HLSL_INVERSE_HYPERBOLIC_FLOAT_TYPES = frozenset(
        {"float", "float2", "float3", "float4"}
    )
    HLSL_INVERSE_HYPERBOLIC_NARROW_TYPES = frozenset(
        {
            "half",
            "half2",
            "half3",
            "half4",
            "float16_t",
            "float16_t2",
            "float16_t3",
            "float16_t4",
            "min16float",
            "min16float2",
            "min16float3",
            "min16float4",
        }
    )
    HLSL_CANONICAL_64_BIT_VECTOR_CONSTRUCTORS = frozenset(
        {
            "i64vec2",
            "i64vec3",
            "i64vec4",
            "u64vec2",
            "u64vec3",
            "u64vec4",
        }
    )
    HLSL_INTERPOLATION_MODE_MODIFIERS = {
        "flat": "nointerpolation",
        "nointerpolation": "nointerpolation",
        "noperspective": "noperspective",
        "linear_noperspective": "noperspective",
        "linear_noperspective_centroid": "noperspective",
    }
    HLSL_INTERPOLATION_SAMPLING_MODIFIERS = {
        "centroid": "centroid",
        "linear_centroid": "centroid",
        "linear_noperspective_centroid": "centroid",
        "sample": "sample",
        "linear_sample": "sample",
    }
    HLSL_RESERVED_IDENTIFIER_NAMES = {
        "centroid",
        "column_major",
        "globallycoherent",
        "line",
        "lineadj",
        "linear",
        "nointerpolation",
        "noperspective",
        "point",
        "precise",
        "row_major",
        "sample",
        "snorm",
        "triangle",
        "triangleadj",
        "uniform",
        "unorm",
    }
    HLSL_FEEDBACK_WRITE_HELPERS = {
        "write_sampler_feedback": ("WriteSamplerFeedback", {4, 5}),
        "write_sampler_feedback_bias": ("WriteSamplerFeedbackBias", {5, 6}),
        "write_sampler_feedback_grad": ("WriteSamplerFeedbackGrad", {6, 7}),
        "write_sampler_feedback_level": ("WriteSamplerFeedbackLevel", {5}),
    }
    HLSL_PIXEL_ONLY_FEEDBACK_WRITE_HELPERS = {
        "write_sampler_feedback": "WriteSamplerFeedback",
        "write_sampler_feedback_bias": "WriteSamplerFeedbackBias",
    }
    HLSL_INTERPOLATION_INTRINSICS = {
        "interpolateAtSample": "EvaluateAttributeAtSample",
        "interpolateAtOffset": "EvaluateAttributeSnapped",
        "interpolateAtCentroid": "EvaluateAttributeCentroid",
    }
    HLSL_SYNCHRONIZATION_INTRINSICS = {
        "atomicThreadFence": "GroupMemoryBarrier",
        "barrier": "GroupMemoryBarrierWithGroupSync",
        "workgroupBarrier": "GroupMemoryBarrierWithGroupSync",
        # GroupMemoryBarrierWithGroupSync once barrier_peephole has merged adjacent barriers.
        "workgroupExecutionBarrier": HLSL_EXECUTION_BARRIER_PLACEHOLDER,
        # CrossGL's groupMemoryBarrier orders buffer and image accesses as well as shared memory (GLSL 4.60
        # section 8.16); GroupMemoryBarrier orders groupshared memory only, so the device-scope fence covers it.
        "groupMemoryBarrier": "AllMemoryBarrier",
        "memoryBarrierShared": "GroupMemoryBarrier",
        "deviceMemoryBarrier": "DeviceMemoryBarrier",
        "memoryBarrierBuffer": "DeviceMemoryBarrier",
        "memoryBarrierImage": "DeviceMemoryBarrier",
        "memoryBarrier": "AllMemoryBarrier",
        "allMemoryBarrier": "AllMemoryBarrier",
        # HLSL has no wave-scope fence; the group- and device-scope fences order the same memory for every
        # thread of the group, which includes the wave.
        "subgroupMemoryBarrier": "AllMemoryBarrier",
        "subgroupMemoryBarrierBuffer": "DeviceMemoryBarrier",
        "subgroupMemoryBarrierShared": "GroupMemoryBarrier",
        "subgroupMemoryBarrierImage": "DeviceMemoryBarrier",
    }
    HLSL_ATOMIC_FENCE_MEMORY_FLAGS = frozenset(
        {
            "mem_none",
            "mem_device",
            "mem_threadgroup",
            "mem_texture",
            "mem_threadgroup_imageblock",
            "mem_object_data",
        }
    )
    HLSL_ATOMIC_FENCE_MEMORY_ORDERS = frozenset(
        {
            "memory_order_relaxed",
            "memory_order_acquire",
            "memory_order_release",
            "memory_order_acq_rel",
            "memory_order_seq_cst",
        }
    )
    HLSL_ATOMIC_FENCE_THREAD_SCOPES = frozenset(
        {
            "thread_scope_thread",
            "thread_scope_simdgroup",
            "thread_scope_threadgroup",
            "thread_scope_device",
            "thread_scope_system",
        }
    )
    HLSL_DERIVATIVE_INTRINSICS = {
        "dFdx": "ddx",
        "dFdy": "ddy",
        "dFdxCoarse": "ddx_coarse",
        "dFdxFine": "ddx_fine",
        "dFdyCoarse": "ddy_coarse",
        "dFdyFine": "ddy_fine",
        "ddx": "ddx",
        "ddy": "ddy",
        "ddx_coarse": "ddx_coarse",
        "ddx_fine": "ddx_fine",
        "ddy_coarse": "ddy_coarse",
        "ddy_fine": "ddy_fine",
        "fwidth": "fwidth",
        "fwidthCoarse": "ddx_coarse/ddy_coarse",
        "fwidthFine": "ddx_fine/ddy_fine",
        "fwidth_coarse": "ddx_coarse/ddy_coarse",
        "fwidth_fine": "ddx_fine/ddy_fine",
    }
    HLSL_RAY_FLAG_VALUES = {
        "RAY_FLAG_NONE": 0x00,
        "RAY_FLAG_FORCE_OPAQUE": 0x01,
        "RAY_FLAG_FORCE_NON_OPAQUE": 0x02,
        "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH": 0x04,
        "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER": 0x08,
        "RAY_FLAG_CULL_BACK_FACING_TRIANGLES": 0x10,
        "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES": 0x20,
        "RAY_FLAG_CULL_OPAQUE": 0x40,
        "RAY_FLAG_CULL_NON_OPAQUE": 0x80,
        "RAY_FLAG_SKIP_TRIANGLES": 0x100,
        "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES": 0x200,
    }
    HLSL_RAY_FLAG_KNOWN_MASK = 0x3FF
    HLSL_RAY_FLAG_MUTUALLY_EXCLUSIVE_GROUPS = (
        (
            "RAY_FLAG_FORCE_OPAQUE",
            "RAY_FLAG_FORCE_NON_OPAQUE",
            "RAY_FLAG_CULL_OPAQUE",
            "RAY_FLAG_CULL_NON_OPAQUE",
        ),
        (
            "RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
            "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES",
            "RAY_FLAG_SKIP_TRIANGLES",
        ),
        (
            "RAY_FLAG_SKIP_TRIANGLES",
            "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES",
        ),
    )
    HLSL_WAVE_INTRINSIC_ARITIES = {
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
        "WaveActiveCountBits": 1,
        "WaveActiveBallot": 1,
        "WaveReadLaneAt": 2,
        "WaveReadLaneFirst": 1,
        "WaveShuffleDown": 2,
        "WaveShuffleUp": 2,
        "WaveShuffleAndFillUp": 3,
        "WaveShuffleXor": 2,
        "WavePrefixSum": 1,
        "WavePrefixProduct": 1,
        "WavePrefixInclusiveSum": 1,
        "WavePrefixInclusiveProduct": 1,
        "WavePrefixCountBits": 1,
        "WaveMatch": 1,
        "WaveMultiPrefixSum": 2,
        "WaveMultiPrefixCountBits": 2,
        "WaveMultiPrefixProduct": 2,
        "WaveMultiPrefixBitAnd": 2,
        "WaveMultiPrefixBitOr": 2,
        "WaveMultiPrefixBitXor": 2,
        "QuadReadAcrossX": 1,
        "QuadReadAcrossY": 1,
        "QuadReadAcrossDiagonal": 1,
        "QuadReadLaneAt": 2,
        "QuadAny": 1,
        "QuadAll": 1,
        "QuadActiveSum": 1,
    }
    HLSL_WAVE_INTRINSIC_ALTERNATE_ARITIES = {
        "WaveShuffleAndFillUp": (3, 4),
    }
    HLSL_WAVE_BOOL_ARGUMENT_INTRINSICS = {
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveBallot",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveMultiPrefixCountBits",
        "QuadAny",
        "QuadAll",
    }
    HLSL_WAVE_NUMERIC_VALUE_INTRINSICS = {
        "WaveActiveSum",
        "WaveActiveProduct",
        "WaveActiveMin",
        "WaveActiveMax",
        "WavePrefixSum",
        "WavePrefixProduct",
        "WavePrefixInclusiveSum",
        "WavePrefixInclusiveProduct",
        "WaveMultiPrefixSum",
        "WaveMultiPrefixProduct",
        "QuadActiveSum",
    }
    # HLSL has no native inclusive scan; an inclusive scan is the exclusive scan
    # combined with the lane's own value. Maps the canonical inclusive op to the
    # exclusive HLSL builtin and the combining operator.
    HLSL_WAVE_INCLUSIVE_PREFIX_OPERATORS = {
        "WavePrefixInclusiveSum": ("WavePrefixSum", "+"),
        "WavePrefixInclusiveProduct": ("WavePrefixProduct", "*"),
    }
    # Bitwise wave reductions require integer operands in HLSL. bf16_math.h
    # invokes them on floats (e.g. __metal_simd_xor(float(data))) to reduce the
    # bit pattern, so a float operand is bitcast through uint and back.
    HLSL_WAVE_BITWISE_VALUE_INTRINSICS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
    }
    HLSL_WAVE_FLOAT_BITCAST_COMPONENT_TYPES = {"float"}
    HLSL_WAVE_INTEGER_VALUE_INTRINSICS = {
        "WaveActiveBitAnd",
        "WaveActiveBitOr",
        "WaveActiveBitXor",
        "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixBitOr",
        "WaveMultiPrefixBitXor",
    }
    HLSL_WAVE_UINT_RESULT_INTRINSICS = {
        "WaveGetLaneCount",
        "WaveGetLaneIndex",
        "WaveActiveCountBits",
        "WavePrefixCountBits",
        "WaveMultiPrefixCountBits",
    }
    HLSL_WAVE_BOOL_RESULT_INTRINSICS = {
        "WaveIsFirstLane",
        "WaveActiveAllTrue",
        "WaveActiveAnyTrue",
        "WaveActiveAllEqual",
        "QuadAny",
        "QuadAll",
    }
    HLSL_WAVE_UINT4_RESULT_INTRINSICS = {
        "WaveActiveBallot",
        "WaveMatch",
    }
    HLSL_WAVE_LANE_READ_VALUE_INTRINSICS = {
        "WaveReadLaneAt",
        "WaveReadLaneFirst",
        "WaveShuffleDown",
        "WaveShuffleUp",
        "WaveShuffleXor",
        "QuadReadAcrossX",
        "QuadReadAcrossY",
        "QuadReadAcrossDiagonal",
        "QuadReadLaneAt",
    }
    HLSL_WAVE_SHUFFLE_AND_FILL_INTRINSICS = {"WaveShuffleAndFillUp"}
    HLSL_WAVE_VALUE_RESULT_INTRINSICS = (
        HLSL_WAVE_NUMERIC_VALUE_INTRINSICS
        | HLSL_WAVE_INTEGER_VALUE_INTRINSICS
        | HLSL_WAVE_LANE_READ_VALUE_INTRINSICS
        | HLSL_WAVE_SHUFFLE_AND_FILL_INTRINSICS
    )
    HLSL_WAVE_NUMERIC_COMPONENT_TYPES = {
        "float",
        "float16_t",
        "half",
        "min16float",
        "min10float",
        "double",
        "int",
        "min16int",
        "min12int",
        "int64_t",
        "int16_t",
        "uint",
        "uint16_t",
        "min16uint",
        "uint64_t",
    }
    HLSL_WAVE_INTEGER_COMPONENT_TYPES = {
        "int",
        "min16int",
        "min12int",
        "int64_t",
        "int16_t",
        "uint",
        "uint16_t",
        "min16uint",
        "uint64_t",
    }
    HLSL_WAVE_BASIC_COMPONENT_TYPES = HLSL_WAVE_NUMERIC_COMPONENT_TYPES | {"bool"}
    HLSL_WAVE_SIZE_LANE_COUNTS = {4, 8, 16, 32, 64, 128}
    HLSL_RESOURCE_POINTER_UNSIGNED_DELTA_TYPES = {
        "uint",
        "uint16_t",
        "uint64_t",
        "min16uint",
    }
    # Subgroup/wave builtins (Metal [[thread_index_in_simdgroup]] /
    # [[threads_per_simdgroup]], canonicalised to gl_Subgroup* in CrossGL) have no
    # HLSL system-value semantic. Invocation ID and size map to wave intrinsics;
    # gl_NumSubgroups derives from the compile-time thread-group size.
    #
    # gl_SubgroupID is intentionally absent from this expression table. Direct3D
    # does not guarantee that flattened SV_GroupIndex values are contiguous within
    # a physical wave, so dividing by WaveGetLaneCount() is valid only when a fixed
    # WaveSize proves the whole workgroup occupies one wave. Other entries route it
    # through the synchronized, collision-safe physical-wave allocator.
    HLSL_WAVE_GROUP_SIZE_PLACEHOLDER = "__CROSSGL_GROUP_SIZE__"
    HLSL_WAVE_BUILTIN_INTRINSICS = {
        "gl_SubgroupInvocationID": "WaveGetLaneIndex()",
        "gl_SubgroupSize": "WaveGetLaneCount()",
        "gl_NumSubgroups": (
            f"(({HLSL_WAVE_GROUP_SIZE_PLACEHOLDER} + WaveGetLaneCount() - 1u) "
            "/ WaveGetLaneCount())"
        ),
    }
    # GLSL compute/subgroup system-value builtins, canonicalised from Metal
    # [[thread_position_in_grid]] / [[simdgroup_index_in_threadgroup]] / etc. These
    # are only meaningful as HLSL system-value semantics (or wave intrinsics) on a
    # true stage entry point. On an ordinary (non-entry) helper function they have
    # no HLSL semantic form and must not leak into a parameter ``: gl_*`` semantic;
    # the value flows in as a plain parameter, exactly as GLSL/SPIR-V emit it.
    HLSL_GLSL_COMPUTE_BUILTIN_SEMANTICS = frozenset(
        {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
            "gl_SubgroupID",
            "gl_NumSubgroups",
            *QUAD_BUILTIN_OPERATORS,
            # Metal [[threads_per_grid]] has no GLSL canonical form (it is kept as
            # the raw Metal name) and no HLSL system value; on a non-entry helper it
            # must likewise be emitted as a plain parameter, never a ": ..." semantic.
            "threads_per_grid",
        }
    )
    # Compute builtins that, even on a true stage entry point, have NO valid HLSL
    # parameter system value: they are either compile-time constants or
    # uniform/derived expressions. Emitting them as ``: <builtin>`` parameter
    # semantics produces invalid HLSL (and ``gl_WorkGroupSize`` additionally trips
    # the ``\bgl_[A-Z]`` unresolved-construct guardrail). Such a parameter is instead
    # lowered to a body-local initialised from the matching HLSL expression, or — when
    # the expression is unavailable and the parameter is unused — dropped entirely.
    #   gl_WorkGroupSize (Metal threads_per_threadgroup) -> uint3(numthreads...)
    #   gl_NumWorkGroups (Metal threadgroups_per_grid)   -> num-workgroups uniform
    #   threads_per_grid (raw Metal)                     -> num_workgroups * numthreads
    HLSL_COMPUTE_VALUE_BUILTIN_SEMANTICS = frozenset(
        {
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "threads_per_grid",
        }
    )
    HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES = HLSL_RESERVED_IDENTIFIER_NAMES | {
        # GLSL atan(y, x) lowers to HLSL atan2(y, x); keep local symbols from
        # capturing that intrinsic because HLSL does not provide a namespace escape.
        "atan2",
        # GLSL bitCount lowers to HLSL countbits.
        "countbits",
        # GLSL bitfieldReverse lowers to HLSL reversebits.
        "reversebits",
        # GLSL findLSB/findMSB lower to HLSL first-bit intrinsics.
        "firstbithigh",
        "firstbitlow",
        # OpenCL/GLSL fma lowers to HLSL mad.
        "mad",
        # Software subgroup barriers must retain the target synchronization call.
        "GroupMemoryBarrierWithGroupSync",
        # Resource queries become helper functions of these names; a local of the same name would hide them.
        "imageSamples",
        "imageSize",
        "textureQueryLevels",
        "textureSamples",
        "textureSize",
    }
    HLSL_BITCAST_FUNCTION_TARGETS = {
        "floatBitsToInt": "int",
        "floatBitsToUint": "uint",
        "intBitsToFloat": "float",
        "uintBitsToFloat": "float",
        "asint": "int",
        "asuint": "uint",
        "asfloat": "float",
    }
    # The HLSL intrinsic each bitcast spelling becomes (HLSL reference, asint/asuint/asfloat).
    HLSL_BITCAST_INTRINSICS = {
        "floatBitsToInt": "asint",
        "floatBitsToUint": "asuint",
        "intBitsToFloat": "asfloat",
        "uintBitsToFloat": "asfloat",
        "asint": "asint",
        "asuint": "asuint",
        "asfloat": "asfloat",
    }
    HLSL_NONUNIFORM_RESOURCE_INDEX_NAMES = {
        "NonUniformResourceIndex",
        "nonuniform",
        "nonuniformEXT",
    }
    HLSL_GLSL_BUILTIN_INT_CONSTANTS = GLSL_BUILTIN_INT_LIMITS
    HLSL_SPECIALIZATION_CONSTANT_TYPES = frozenset({"bool", "int", "uint", "float"})
    HLSL_RELATIVE_WAVE_SHUFFLE_OUT_OF_RANGE_POLICIES = frozenset({"undefined", "self"})
    HLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH = 32
    HLSL_SOFTWARE_SUBGROUP_REDUCTIONS = {
        "WaveActiveSum": "sum",
        "WaveActiveProduct": "product",
        "WaveActiveMin": "min",
        "WaveActiveMax": "max",
        "WaveActiveAllTrue": "all",
        "WaveActiveAnyTrue": "any",
    }
    HLSL_SOFTWARE_SUBGROUP_VOTES = frozenset({"WaveActiveAllTrue", "WaveActiveAnyTrue"})
    HLSL_SOFTWARE_SUBGROUP_SCANS = {
        "WavePrefixSum": ("+", False),
        "WavePrefixProduct": ("*", False),
        "WavePrefixInclusiveSum": ("+", True),
        "WavePrefixInclusiveProduct": ("*", True),
    }
    # Subgroup barriers that software subgroups lower to the workgroup barrier.
    HLSL_SOFTWARE_SUBGROUP_BARRIERS = frozenset(
        {"subgroupExecutionBarrier", "subgroupBarrier"}
    )
    HLSL_SOFTWARE_SUBGROUP_OPERATIONS = frozenset(
        {
            "WaveShuffleDown",
            "WaveShuffleUp",
            "WaveShuffleXor",
            "WaveReadLaneAt",
            "WaveShuffleAndFillUp",
            *HLSL_SOFTWARE_SUBGROUP_BARRIERS,
            *HLSL_SOFTWARE_SUBGROUP_REDUCTIONS,
            *HLSL_SOFTWARE_SUBGROUP_SCANS,
            *SOFTWARE_QUAD_REDUCTIONS,
        }
    )
    # Relative shuffles: helper name, source-lane validity and source lane.
    HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES = {
        "WaveShuffleDown": (
            "shuffle_down",
            "delta < ({active} - lane)",
            "lane + delta",
        ),
        "WaveShuffleUp": ("shuffle_up", "delta <= lane", "lane - delta"),
        "WaveShuffleXor": ("shuffle_xor", "(lane ^ delta) < {active}", "lane ^ delta"),
    }
    HLSL_SOFTWARE_SUBGROUP_VALUE_TYPES = frozenset({"float", "int", "uint"})
    HLSL_SOFTWARE_SUBGROUP_SHUFFLE_TYPES = HLSL_SOFTWARE_SUBGROUP_VALUE_TYPES | {
        "uint2",
        "int16_t",
        "uint16_t",
        "int64_t",
        "uint64_t",
    }

    # HLSL has no native relative-shuffle wave ops, so shuffle-down/up/xor are
    # emulated by reading the lane relative to the current one via
    # `WaveReadLaneAt`. Out-of-range lanes remain undefined by default; projects
    # may explicitly select a typed calling-lane fallback for deterministic
    # reduction artifacts without changing in-range reads.
    HLSL_WAVE_RELATIVE_SHUFFLE_OPERATORS = {
        "WaveShuffleDown": "+",
        "WaveShuffleUp": "-",
        "WaveShuffleXor": "^",
    }
    HLSL_WAVE_SHUFFLE_VALUE_INTRINSICS = {
        "WaveReadLaneAt",
        "WaveReadLaneFirst",
        *HLSL_WAVE_RELATIVE_SHUFFLE_OPERATORS,
        *HLSL_WAVE_SHUFFLE_AND_FILL_INTRINSICS,
    }
