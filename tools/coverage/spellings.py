"""How CrossGL names and calls the built-in functions of each backend.

A coverage probe calls a backend built-in through CrossGL. Most have the same name and arguments in both, so the
probe uses the backend's spelling. The tables here list the exceptions: the functions CrossGL spells as GLSL does
(:data:`RENAMES`), those it has under a name GLSL lacks (:data:`WAVES`, :data:`OTHER_NAMES`) and those it writes with
other arguments (:data:`COUNTERPARTS`). Rust's entries, for spirv-std and glam, are in :mod:`.rust_spellings`. The
feature lists use the tables to give each feature its CrossGL spelling, and issue reconciliation uses them to
recognise that spelling in a diagnostic.
"""

from __future__ import annotations

import re

from . import rust_spellings

# The CrossGL spelling of a function that reads a vector's element at an index, which CrossGL writes as ``v[i]``.
SUBSCRIPT = "[]"
# The CrossGL spelling of a function that returns a vector with the element at an index replaced, which CrossGL writes
# as an assignment, ``v[i] = x``, to a copy of the vector.
ELEMENT = "[]="

# Built-in functions CrossGL spells as GLSL does rather than as the backend does. A rename applies only when the GLSL
# feature list has the new name; any other backend function is probed under its own spelling.
RENAMES = {
    "hlsl": {
        "lerp": "mix",
        "frac": "fract",
        "rsqrt": "inversesqrt",
        "mad": "fma",
        "countbits": "bitCount",
        "firstbithigh": "findMSB",
        "firstbitlow": "findLSB",
        "reversebits": "bitfieldReverse",
        "asint": "floatBitsToInt",
        "asuint": "floatBitsToUint",
        "atan2": "atan",
        "ddx": "dFdx",
        "ddy": "dFdy",
        "ddx_coarse": "dFdxCoarse",
        "ddx_fine": "dFdxFine",
        "ddy_coarse": "dFdyCoarse",
        "ddy_fine": "dFdyFine",
        "GroupMemoryBarrierWithGroupSync": "barrier",
        "GroupMemoryBarrier": "memoryBarrierShared",
        "AllMemoryBarrier": "memoryBarrier",
        "DeviceMemoryBarrier": "memoryBarrierBuffer",
        "InterlockedAdd": "atomicAdd",
        "InterlockedMin": "atomicMin",
        "InterlockedMax": "atomicMax",
        "InterlockedAnd": "atomicAnd",
        "InterlockedOr": "atomicOr",
        "InterlockedXor": "atomicXor",
        "WaveActiveAllTrue": "allInvocations",
        "WaveActiveAnyTrue": "anyInvocation",
        "WaveActiveAllEqual": "allInvocationsEqual",
    },
    # CUDA after :mod:`.features.cuda` drops the single-precision suffix: ``fabsf`` arrives here as ``fabs``. CrossGL's
    # fmin and fmax return the operand that is not NaN, as C's, MSL's and SPIR-V's NMin and NMax do, and its min and
    # max need not, so CUDA's and Metal's fmin and fmax keep their names.
    "cuda": {
        "fabs": "abs",
        "rsqrt": "inversesqrt",
        "atan2": "atan",
        "rint": "roundEven",
        "__popc": "bitCount",
        "__brev": "bitfieldReverse",
        "__float_as_int": "floatBitsToInt",
        "__float_as_uint": "floatBitsToUint",
        "__int_as_float": "intBitsToFloat",
        "__uint_as_float": "uintBitsToFloat",
        "__isnanf": "isnan",
        "__isinff": "isinf",
        "__syncthreads": "barrier",
        "__threadfence": "memoryBarrier",
        "__threadfence_block": "memoryBarrierShared",
        "atomicExch": "atomicExchange",
        "atomicCAS": "atomicCompSwap",
    },
    "wgsl": {
        "atan2": "atan",
        "inverseSqrt": "inversesqrt",
        "faceForward": "faceforward",
        "countOneBits": "bitCount",
        "reverseBits": "bitfieldReverse",
        "firstLeadingBit": "findMSB",
        "firstTrailingBit": "findLSB",
        "extractBits": "bitfieldExtract",
        "insertBits": "bitfieldInsert",
        "bitcast": "floatBitsToUint",
        "select": "mix",
        "dpdx": "dFdx",
        "dpdy": "dFdy",
        "dpdxCoarse": "dFdxCoarse",
        "dpdxFine": "dFdxFine",
        "dpdyCoarse": "dFdyCoarse",
        "dpdyFine": "dFdyFine",
        "pack4x8snorm": "packSnorm4x8",
        "pack4x8unorm": "packUnorm4x8",
        "pack2x16snorm": "packSnorm2x16",
        "pack2x16unorm": "packUnorm2x16",
        "pack2x16float": "packHalf2x16",
        "unpack4x8snorm": "unpackSnorm4x8",
        "unpack4x8unorm": "unpackUnorm4x8",
        "unpack2x16snorm": "unpackSnorm2x16",
        "unpack2x16unorm": "unpackUnorm2x16",
        "unpack2x16float": "unpackHalf2x16",
        "workgroupBarrier": "barrier",
        "storageBarrier": "memoryBarrierBuffer",
        "textureBarrier": "memoryBarrierImage",
        "subgroupAll": "allInvocations",
        "subgroupAny": "anyInvocation",
    },
    "metal": {
        "atan2": "atan",
        "fabs": "abs",
        "rsqrt": "inversesqrt",
        "rint": "roundEven",
        "select": "mix",
        "popcount": "bitCount",
        "reverse_bits": "bitfieldReverse",
        "extract_bits": "bitfieldExtract",
        "insert_bits": "bitfieldInsert",
        "dfdx": "dFdx",
        "dfdy": "dFdy",
        "discard_fragment": "discard",
        "pack_float_to_snorm4x8": "packSnorm4x8",
        "pack_float_to_unorm4x8": "packUnorm4x8",
        "pack_float_to_snorm2x16": "packSnorm2x16",
        "pack_float_to_unorm2x16": "packUnorm2x16",
        "unpack_snorm4x8_to_float": "unpackSnorm4x8",
        "unpack_unorm4x8_to_float": "unpackUnorm4x8",
        "unpack_snorm2x16_to_float": "unpackSnorm2x16",
        "unpack_unorm2x16_to_float": "unpackUnorm2x16",
        "simd_all": "allInvocations",
        "simd_any": "anyInvocation",
    },
    "mojo": {
        "atan2": "atan",
        "rsqrt": "inversesqrt",
        "pop_count": "bitCount",
        "bit_reverse": "bitfieldReverse",
    },
    "rust": rust_spellings.RENAMES,
}
# HIP spells its device functions the way CUDA does. Slang spells its built-in functions the way HLSL does, and adds
# the coarse and fine fwidth that HLSL lacks, centroid interpolation under its own name, and unpacking functions that
# name the result's type, of which the ones that return float vectors are GLSL's.
RENAMES["hip"] = RENAMES["cuda"]
RENAMES["slang"] = RENAMES["hlsl"] | {
    "fwidth_coarse": "fwidthCoarse",
    "fwidth_fine": "fwidthFine",
    "EvaluateAttributeAtCentroid": "interpolateAtCentroid",
    "unpackHalf2x16ToFloat": "unpackHalf2x16",
    "unpackSnorm2x16ToFloat": "unpackSnorm2x16",
    "unpackSnorm4x8ToFloat": "unpackSnorm4x8",
    "unpackUnorm2x16ToFloat": "unpackUnorm2x16",
    "unpackUnorm4x8ToFloat": "unpackUnorm4x8",
}

# Subgroup functions, which CrossGL spells with the HLSL wave names (``crosstl/translator/wave_operations.py``) and
# the GLSL feature list, built from the core GLSL specification, lacks.
WAVES = {"rust": rust_spellings.WAVES}


# Functions CrossGL writes with other arguments than the backend, as GLSL prototypes. WGSL's texture functions take a
# texture and a sampler, where CrossGL, like GLSL, takes one combined sampler; ``const`` marks an argument the probe
# passes as the constant 0, for a WGSL function that always reads level 0.
COUNTERPARTS = {
    "wgsl": {
        "textureDimensions": "ivec2 textureSize(sampler2D, const int)",
        "textureGather": "vec4 textureGather(sampler2D, vec2)",
        "textureGatherCompare": "vec4 textureGather(sampler2DShadow, vec2, float)",
        "textureLoad": "vec4 texelFetch(sampler2D, ivec2, int)",
        "textureNumLayers": "ivec3 textureSize(sampler2DArray, const int)",
        "textureNumLevels": "int textureQueryLevels(sampler2D)",
        "textureNumSamples": "int textureSamples(sampler2DMS)",
        "textureSample": "vec4 texture(sampler2D, vec2)",
        "textureSampleBias": "vec4 texture(sampler2D, vec2, float)",
        "textureSampleCompare": "float texture(sampler2DShadow, vec3)",
        "textureSampleCompareLevel": (
            "float textureLod(sampler2DShadow, vec3, const float)"
        ),
        "textureSampleGrad": "vec4 textureGrad(sampler2D, vec2, vec2, vec2)",
        "textureSampleLevel": "vec4 textureLod(sampler2D, vec2, float)",
        # CrossGL writes this one as an expression (:data:`OTHER_NAMES`), which takes these arguments.
        "textureSampleBaseClampToEdge": "vec4 textureLod(sampler2D, vec2)",
        "textureStore": "void imageStore(image2D, ivec2, vec4)",
        # WGSL returns a result struct where CrossGL, like GLSL, returns one part and writes the other to an output
        # parameter. Its only compare-exchange is the weak one, and CrossGL's is atomicCompSwap.
        "frexp": "vec3 frexp(vec3, out ivec3)",
        "modf": "vec3 modf(vec3, out vec3)",
        "atomicCompareExchangeWeak": "int atomicCompSwap(inout int, int, int)",
        # CrossGL asks a buffer for its element count with buffer_dimensions, as the SPIR-V importer writes
        # OpArrayLength.
        "arrayLength": "uint buffer_dimensions(buffer uint)",
    },
    # Metal's barrier takes memory flags, which the probe passes as mem_none: an execution barrier only.
    "metal": {"threadgroup_barrier": "void barrier()"},
    # Mojo returns both parts of frexp and modf as a tuple. std.atomic's static load and store are CrossGL's relaxed
    # atomicLoad and atomicStore, and its max and min return nothing. Atomic.fetch_sub takes an Atomic value, where
    # CrossGL's atomicAdd, which it is of the negated operand (:data:`OTHER_NAMES`), takes the buffer element.
    "mojo": {
        "frexp": "float frexp(float, out int)",
        "modf": "float modf(float, out float)",
        "Atomic.load": "int atomicLoad(inout int)",
        "Atomic.store": "void atomicStore(inout int, int)",
        "Atomic.max": "void atomicMax(inout int, int)",
        "Atomic.min": "void atomicMin(inout int, int)",
        "Atomic.fetch_sub": "int atomicAdd(inout int, int)",
    },
    "rust": rust_spellings.COUNTERPARTS,
}
# Metal functions whose CrossGL operation takes other arguments. Compare-exchange takes the expected value through a
# pointer and returns whether it swapped, where CrossGL's atomicCompSwap, like GLSL's, takes it by value and returns
# the old value; a ballot returns a simd_vote mask, where CrossGL's returns a uvec4.
COUNTERPARTS["metal"].update(
    {
        "atomic_compare_exchange_weak": "int atomicCompSwap(inout int, int, int)",
        "atomic_compare_exchange_weak_explicit": (
            "int atomicCompSwap(inout int, int, int)"
        ),
        "simd_ballot": "uvec4 subgroupBallot(bool)",
    }
)
# Metal's atomic functions, in forms with and without a memory order. CrossGL's, like GLSL's, are relaxed and take
# none, so the probe passes memory_order_relaxed to the Metal function.
_METAL_ATOMICS = {
    "exchange": "atomicExchange",
    "fetch_add": "atomicAdd",
    "fetch_and": "atomicAnd",
    "fetch_max": "atomicMax",
    "fetch_min": "atomicMin",
    "fetch_or": "atomicOr",
    "fetch_xor": "atomicXor",
}
RENAMES["metal"].update({f"atomic_{op}": name for op, name in _METAL_ATOMICS.items()})
# OpenCL C has no generator, so none of its probes is written in CrossGL, but issue reconciliation still needs the
# CrossGL names of its atomic functions to find them in a diagnostic. OpenCL C 2.0's are Metal's, with and without a
# memory order; the older ones of OpenCL C 1.1 and the cl_khr_*_atomics extensions start with atomic_ or atom_.
_OPENCL_ATOMICS = {
    "add": "atomicAdd",
    "and": "atomicAnd",
    "max": "atomicMax",
    "min": "atomicMin",
    "or": "atomicOr",
    "xor": "atomicXor",
    "xchg": "atomicExchange",
    "cmpxchg": "atomicCompSwap",
}
RENAMES["opencl"] = {
    f"{prefix}_{op}": name
    for prefix in ("atomic", "atom")
    for op, name in _OPENCL_ATOMICS.items()
} | {
    f"atomic_{op}{suffix}": name
    for op, name in _METAL_ATOMICS.items()
    for suffix in ("", "_explicit")
}
COUNTERPARTS["metal"].update(
    {
        f"atomic_{op}_explicit": f"int {name}(inout int, int)"
        for op, name in _METAL_ATOMICS.items()
    }
)
# CrossGL's relaxed atomicLoad and atomicStore, which GLSL lacks, are the ones the Metal importer writes for
# atomic_load_explicit and atomic_store_explicit. Metal's atomic_max and atomic_min are its 64-bit ones, which return
# nothing.
COUNTERPARTS["metal"].update(
    {
        f"atomic_{op}{suffix}": prototype
        for op, prototype in (
            ("load", "int atomicLoad(inout int)"),
            ("store", "void atomicStore(inout int, int)"),
            ("max", "void atomicMax(inout uint64_t, uint64_t)"),
            ("min", "void atomicMin(inout uint64_t, uint64_t)"),
        )
        for suffix in ("", "_explicit")
    }
)
# Metal functions that take memory flags, which CrossGL writes as the operations the Metal importer makes of them. The
# probe passes mem_none and memory_order_relaxed. A simdgroup barrier with mem_none is CrossGL's execution-only
# subgroupExecutionBarrier, and a fence keeps its flags, order and scope as CrossGL constants (:data:`OTHER_NAMES`).
# CrossGL has no atomic subtraction, so it adds the negated operand, which gives the same wrapped result, for
# atomic_fetch_sub as for atomic_fetch_sub_explicit.
COUNTERPARTS["metal"].update(
    {
        "simdgroup_barrier": "void subgroupExecutionBarrier()",
        "atomic_thread_fence": "void atomicThreadFence()",
        "atomic_fetch_sub": "int atomicAdd(inout int, int)",
        "atomic_fetch_sub_explicit": "int atomicAdd(inout int, int)",
    }
)


# Built-in functions CrossGL has under a name the GLSL feature list lacks, with the same arguments and result: a
# CrossGL wave operation, a GLSL extension function, a function of another language that CrossGL defines, or a
# built-in value. A ``gl_`` name is a built-in value (GLSL 4.60 section 3.7 reserves the prefix), which the probe reads
# instead of calling, with a ``{}`` for each argument that indexes it.
OTHER_NAMES = {
    # HLSL's and Slang's select take the condition first, select(c, t, f), where CrossGL's, like Metal's, takes it
    # last: select(f, t, c). The HLSL importer reorders the arguments, and the HLSL generator orders them back.
    "hlsl": {"select": "select({2}, {1}, {0})"},
    "slang": {
        "select": "select({2}, {1}, {0})",
        # Slang's WavePrefixMax, WavePrefixMin and bitwise prefix operations are exclusive scans, as its
        # WavePrefixSum and HLSL's are; CrossGL names those that HLSL lacks after the GLSL scan.
        "WavePrefixMax": "WavePrefixExclusiveMax",
        "WavePrefixMin": "WavePrefixExclusiveMin",
        "WavePrefixBitAnd": "WavePrefixExclusiveBitAnd",
        "WavePrefixBitOr": "WavePrefixExclusiveBitOr",
        "WavePrefixBitXor": "WavePrefixExclusiveBitXor",
        "WaveShuffle": "WaveReadLaneAt",
        # A lane given as a constant broadcasts, as GL_KHR_shader_subgroup_ballot's subgroupBroadcast does.
        "WaveBroadcastLaneAt": "subgroupBroadcast",
        "WaveGetLaneEqMask": "gl_SubgroupEqMask",
        "WaveGetLaneGeMask": "gl_SubgroupGeMask",
        "WaveGetLaneGtMask": "gl_SubgroupGtMask",
        "WaveGetLaneLeMask": "gl_SubgroupLeMask",
        "WaveGetLaneLtMask": "gl_SubgroupLtMask",
        "WaveGetNumWaves": "gl_NumSubgroups",
        "WaveGetWaveIndex": "gl_SubgroupID",
        "WorkgroupCount": "gl_NumWorkGroups",
        # CUDA's threadIdx, blockIdx and blockDim, which Slang reads for CUDA only.
        "cudaThreadIdx": "gl_LocalInvocationID",
        "cudaBlockIdx": "gl_WorkGroupID",
        "cudaBlockDim": "gl_WorkGroupSize",
        "HitTriangleVertexPosition": "gl_HitTriangleVertexPositionsEXT[{}]",
        # GL_EXT_shader_realtime_clock's device-scope clock, as two 32-bit halves.
        "getRealtimeClock": "clockRealtime2x32EXT",
        # Slang's copysign calls these for each element type.
        "copysign_float": "copysign",
        "copysign_half": "copysign",
        "copysign_double": "copysign",
        # Slang's HLSL pack_s8, pack_u8 and unpack_* functions, which CrossGL has, call these
        # (hlsl.meta.slang, "Pack/Unpack Math Intrinsics").
        "packInt4x8": "pack_s8",
        "packUint4x8": "pack_u8",
        "packInt4x8Clamp": "pack_clamp_s8",
        "packUint4x8Clamp": "pack_clamp_u8",
        "unpackInt4x8ToInt16": "unpack_s8s16",
        "unpackInt4x8ToInt32": "unpack_s8s32",
        "unpackUint4x8ToUint16": "unpack_u8u16",
        "unpackUint4x8ToUint32": "unpack_u8u32",
        # Slang defines HLSL's SM 6.5 multi-prefix scans as these exclusive scans.
        "WaveMultiPrefixExclusiveSum": "WaveMultiPrefixSum",
        "WaveMultiPrefixExclusiveProduct": "WaveMultiPrefixProduct",
        "WaveMultiPrefixExclusiveBitAnd": "WaveMultiPrefixBitAnd",
        "WaveMultiPrefixExclusiveBitOr": "WaveMultiPrefixBitOr",
        "WaveMultiPrefixExclusiveBitXor": "WaveMultiPrefixBitXor",
        # The rest are written as Slang writes them for HLSL or GLSL: an inclusive scan as the exclusive one
        # combined with the lane's own value, and the ballot of every converged lane as WaveActiveBallot(true).
        "WaveMultiPrefixInclusiveSum": "(WaveMultiPrefixSum({0}, {1}) + {0})",
        "WaveMultiPrefixInclusiveProduct": "(WaveMultiPrefixProduct({0}, {1}) * {0})",
        "WaveMultiPrefixInclusiveBitAnd": "(WaveMultiPrefixBitAnd({0}, {1}) & {0})",
        "WaveMultiPrefixInclusiveBitOr": "(WaveMultiPrefixBitOr({0}, {1}) | {0})",
        "WaveMultiPrefixInclusiveBitXor": "(WaveMultiPrefixBitXor({0}, {1}) ^ {0})",
        "WaveGetConvergedMulti": "WaveActiveBallot(true)",
        "WaveGetActiveMulti": "WaveActiveBallot(true)",
        "ReorderThread": "MaybeReorderThread",
        # A rotation reads the value of the lane delta places up, wrapping within the wave or the cluster, as Slang
        # writes it for CUDA (hlsl.meta.slang, WaveRotate, and slang-cuda-prelude.h, _slang_waveClusteredRotate).
        "WaveRotate": (
            "WaveReadLaneAt({0}, (WaveGetLaneIndex() + {1}) % WaveGetLaneCount())"
        ),
        "WaveClusteredRotate": (
            "WaveReadLaneAt({0}, WaveGetLaneIndex() / {2} * {2} + (WaveGetLaneIndex() % {2} + {1}) % {2})"
        ),
        "getRealtimeClockLow": "clockRealtime2x32EXT().x",
        "WorkgroupSize": "ivec3(gl_WorkGroupSize)",
        "unpackHalf2x16ToHalf": "f16vec2(unpackHalf2x16({}))",
        "unpackSnorm2x16ToHalf": "f16vec2(unpackSnorm2x16({}))",
        "unpackUnorm2x16ToHalf": "f16vec2(unpackUnorm2x16({}))",
        "unpackSnorm4x8ToHalf": "f16vec4(unpackSnorm4x8({}))",
        "unpackUnorm4x8ToHalf": "f16vec4(unpackUnorm4x8({}))",
    },
    # WGSL's quad operations are HLSL's. Its leading and trailing zero counts are Metal's clz and ctz, which count
    # every bit of a zero operand, as WGSL's do (MSL 4.1, "Integer Functions"). Mojo's are the same functions.
    "wgsl": {
        "quadBroadcast": "QuadReadLaneAt",
        "quadSwapX": "QuadReadAcrossX",
        "quadSwapY": "QuadReadAcrossY",
        "quadSwapDiagonal": "QuadReadAcrossDiagonal",
        "countLeadingZeros": "clz",
        "countTrailingZeros": "ctz",
        # WGSL's 8-bit integer packing is HLSL's: component i in bits 8i to 8i + 7, unpacked with sign or zero
        # extension (WGSL, "Data Packing Built-in Functions"). pack4xU8Clamp clamps unsigned components to
        # [0, 255], where pack_clamp_u8 clamps signed ones, so it packs the components min() has clamped.
        "pack4xI8": "pack_s8",
        "pack4xU8": "pack_u8",
        "pack4xI8Clamp": "pack_clamp_s8",
        "pack4xU8Clamp": "pack_u8(min({0}, uvec4(255u)))",
        "unpack4xI8": "unpack_s8s32",
        "unpack4xU8": "unpack_u8u32",
        # WGSL's packed dot products are HLSL's with a zero accumulator (WGSL, "Dot Product Built-in Functions";
        # HLSL Shader Model 6.4, dot4add_i8packed and dot4add_u8packed).
        "dot4I8Packed": "dot4add_i8packed({0}, {1}, 0)",
        "dot4U8Packed": "dot4add_u8packed({0}, {1}, 0u)",
        # quantizeToF16 rounds each component to binary16 and back, which for vec2<f32> is
        # unpack2x16float(pack2x16float(e)) (WGSL, "quantizeToF16"): HLSL's f16tof32 of its f32tof16.
        "quantizeToF16": "f16tof32(f32tof16({0}))",
        # textureSampleBaseClampToEdge samples level 0 with its coordinates clamped to [half_texel, 1 - half_texel],
        # where half_texel is vec2(0.5) / vec2<f32>(textureDimensions(t, 0)) (WGSL, "textureSampleBaseClampToEdge").
        "textureSampleBaseClampToEdge": (
            "textureLod({0}, clamp({1}, vec2(0.5) / vec2(textureSize({0}, 0)), "
            "vec2(1.0) - vec2(0.5) / vec2(textureSize({0}, 0))), 0.0)"
        ),
    },
    # Metal's fence takes its scope as a third argument that defaults to thread_scope_device (MSL 4.1, "Fence
    # Functions"). Its atomic_fetch_sub and atomic_fetch_sub_explicit are an atomicAdd of the negated operand, as the
    # Metal importer writes them.
    "metal": {
        "atomic_thread_fence": (
            "atomicThreadFence(mem_none, memory_order_relaxed, thread_scope_device)"
        ),
        "atomic_fetch_sub": "atomicAdd({0}, -({1}))",
        "atomic_fetch_sub_explicit": "atomicAdd({0}, -({1}))",
    },
    # CUDA's fdivide and, without fast math, fdividef return a / b (__clang_cuda_math.h).
    "cuda": {"fdivide": "({0} / {1})"},
    # HIP's wall_clock64 reads the device's constant-rate counter (amd_device_functions.h), which is the device-scope
    # clock GL_EXT_shader_realtime_clock's clockRealtimeEXT reads.
    "hip": {"wall_clock64": "int64(clockRealtimeEXT())"},
    "mojo": {
        "count_leading_zeros": "clz",
        "count_trailing_zeros": "ctz",
        # Mojo's gamma is the C library's tgamma.
        "gamma": "tgamma",
        # std.bit as Mojo 1.2.0 writes it (std/bit/bit.mojo), for the Int32 a probe passes. For a signed value,
        # bit_width counts the bits of the value or, when it is negative, of its complement, which is GLSL's findMSB
        # plus one (GLSL 4.60 section 8.8). log2_floor is -1 for zero and for a negative value, and the powers of two
        # are 1 shifted by a bit width. rotate_bits_left and rotate_bits_right are funnel shifts of the value with
        # itself, by the shift of 1 the probe passes.
        "bit_not": "(~{0})",
        "bit_width": "(findMSB({0}) + 1)",
        "log2_floor": "({0} < 0 ? -1 : findMSB({0}))",
        "log2_ceil": "({0} <= 1 ? 0 : findMSB({0} - 1) + 1)",
        "next_power_of_two": "({0} > 1 ? 1 << (findMSB({0} - 1) + 1) : 1)",
        "prev_power_of_two": "({0} > 0 ? 1 << findMSB({0}) : 0)",
        "rotate_bits_left": "(({0} << 1) | int(uint({0}) >> 31u))",
        "rotate_bits_right": "(int(uint({0}) >> 1u) | ({0} << 31))",
        "byte_swap": (
            "int((uint({0}) << 24u) | ((uint({0}) & 0xFF00u) << 8u) | ((uint({0}) >> 8u) & 0xFF00u)"
            " | (uint({0}) >> 24u))"
        ),
        # std.math (std/math/math.mojo): recip is 1.0 / x, inf the float32 infinity, and iota of width 1 its offset,
        # 0 by default. mulwide widens both factors before it multiplies. Mojo's // rounds towards negative
        # infinity, so align_down and align_up, which divide with it, correct CrossGL's truncating division when the
        # quotient is inexact and, for align_down, negative or, for align_up, positive. ceildiv of a Float32 is the
        # ceiling of the quotient.
        "recip": "rcp",
        "inf": "(INFINITY)",
        "iota": "(0.0)",
        # MAX's warp functions (max/gpu/primitives/warp.mojo), which a probe calls through their module, are
        # GL_KHR_shader_subgroup's. shuffle_idx reads the lane it names, and prefix_sum is inclusive unless asked to
        # be exclusive. A lane-group reduction over the whole warp, as the probe makes it, is the warp reduction.
        # broadcast copies lane 0's value, and vote returns the 32-bit ballot of a 32-lane warp, the first component
        # of CrossGL's. match_any is CUDA's __match_any_sync over the full warp (match.any.sync.b32 with the mask
        # 0xFFFFFFFF). max.gpu.sync's syncwarp waits for the warp's lanes and orders their memory accesses, as
        # subgroupBarrier does.
        "warp.shuffle_idx": "subgroupShuffle",
        "warp.shuffle_up": "subgroupShuffleUp",
        "warp.shuffle_down": "subgroupShuffleDown",
        "warp.shuffle_xor": "subgroupShuffleXor",
        "warp.sum": "subgroupAdd",
        "warp.max": "subgroupMax",
        "warp.min": "subgroupMin",
        "warp.lane_group_sum": "subgroupAdd",
        "warp.lane_group_max": "subgroupMax",
        "warp.lane_group_min": "subgroupMin",
        "warp.prefix_sum": "subgroupInclusiveAdd",
        "warp.broadcast": "subgroupBroadcast({0}, 0u)",
        "warp.vote": "subgroupBallot({0}).x",
        "warp.match_any": "__match_any_sync(4294967295u, {0})",
        "syncwarp": "subgroupBarrier",
        # max.gpu's lane_id is the invocation's index in its warp, and warp_id divides the x index in the block by the
        # warp size (max/gpu/primitives/id.mojo), both as an Int. Atomic.fetch_add returns the value it replaced, and
        # Atomic.fetch_sub, which CrossGL has no operation of its own for, is an atomicAdd of the negated operand,
        # which wraps to the same result.
        "lane_id": "int64(gl_SubgroupInvocationID)",
        "warp_id": "int64(gl_LocalInvocationID.x / gl_SubgroupSize)",
        "Atomic.fetch_add": "atomicAdd",
        "Atomic.fetch_sub": "atomicAdd({0}, -({1}))",
        # std.atomic's fence is sequentially consistent by default (atomic.mojo, _DEFAULT_MEMORY_ORDERING), as CUDA's
        # __threadfence is.
        "fence": "memoryBarrier",
        "mulwide": "(uint64({0}) * uint64({1}))",
        "align_down": (
            "(({0} / {1} - ({0} != {0} / {1} * {1} && ({0} < 0) != ({1} < 0) ? 1 : 0)) * {1})"
        ),
        "align_up": (
            "(({0} / {1} + ({0} != {0} / {1} * {1} && ({0} < 0) == ({1} < 0) ? 1 : 0)) * {1})"
        ),
        "ceildiv": "ceil({0} / {1})",
        # isclose with the defaults the probe keeps (symmetrical, atol 1e-08, rtol 1e-05, equal_nan false): equal
        # values, or two finite values whose distance is within the larger of atol and rtol times the larger
        # magnitude. Mojo converts both Float64 tolerances to the Float32 of the arguments, which rounds them to
        # the float32 literals written here.
        "isclose": (
            "(({0} == {1}) || (isfinite({0}) && isfinite({1})"
            " && abs({0} - {1}) <= max(1e-08, 1e-05 * max(abs({0}), abs({1})))))"
        ),
        # ulp as std/math/math.mojo computes it: NaN for NaN, the magnitude of an infinity, and otherwise the gap
        # from the magnitude to the next float up or, when that is infinite, down.
        "ulp": (
            "(isnan({0}) ? {0} : (isinf(abs({0})) ? abs({0}) : (isinf(nextafter(abs({0}), INFINITY))"
            " ? abs({0}) - nextafter(abs({0}), -INFINITY) : nextafter(abs({0}), INFINITY) - abs({0}))))"
        ),
    },
    "rust": rust_spellings.OTHER_NAMES,
}


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    if name in WAVES.get(language, {}):
        return WAVES[language][name]
    if name in OTHER_NAMES.get(language, {}):
        return OTHER_NAMES[language][name]
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name


# A CrossGL scalar conversion or vector constructor, which converts rather than names an operation.
_CONSTRUCTOR = re.compile(r"(?:[biud]|f16)?vec[234]|float|double|u?int(?:64)?|bool")
_NAMED = re.compile(r"\bgl_\w+|\b[A-Za-z_]\w*(?=\()")


def is_expression(spelled: str) -> bool:
    """Whether a CrossGL spelling is an expression, such as ``WaveActiveBallot(true)``, rather than one name."""
    return "(" in spelled or "{" in spelled


def named(spelled: str) -> list[str]:
    """The built-in functions an expression spelling calls and the built-in values it reads, in order."""
    return [n for n in _NAMED.findall(spelled) if not _CONSTRUCTOR.fullmatch(n)]
