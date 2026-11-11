"""How CrossGL names and calls the built-in functions of each backend.

A coverage probe calls a backend built-in through CrossGL. Most have the same name and arguments in both, so the
probe uses the backend's spelling. The tables here list the exceptions: the functions CrossGL spells as GLSL does
(:data:`RENAMES`), those it has under a name GLSL lacks (:data:`WAVES`, :data:`OTHER_NAMES`) and those it writes with
other arguments (:data:`COUNTERPARTS`). The feature lists use them to give
each feature its CrossGL spelling, and issue reconciliation uses them to recognise that spelling in a diagnostic.
"""

from __future__ import annotations

import re

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
    # Rust names a function by its trait or type, as ``Float::ln`` or ``Vec3::lerp``; only the ones that compute what
    # the GLSL function does are renamed.
    "rust": {
        "Float::ln": "log",
        "Float::powf": "pow",
        "Float::mul_add": "fma",
        "Float::to_degrees": "degrees",
        "Float::to_radians": "radians",
        "Float::is_nan": "isnan",
        "Float::is_infinite": "isinf",
        "Float::atan2": "atan",
        "Vec3::ln": "log",
        "Vec3::powf": "pow",
        "Vec3::mul_add": "fma",
        "Vec3::lerp": "mix",
        "Vec3::fract_gl": "fract",
        "Vec3::is_nan_mask": "isnan",
        "Vec3::cmpeq": "equal",
        "Vec3::cmpne": "notEqual",
        "Vec3::cmplt": "lessThan",
        "Vec3::cmple": "lessThanEqual",
        "Vec3::cmpgt": "greaterThan",
        "Vec3::cmpge": "greaterThanEqual",
        "Derivative::dfdx": "dFdx",
        "Derivative::dfdy": "dFdy",
        "Derivative::fwidth": "fwidth",
        "Derivative::dfdx_fine": "dFdxFine",
        "Derivative::dfdy_fine": "dFdyFine",
        "Derivative::fwidth_fine": "fwidthFine",
        "Derivative::dfdx_coarse": "dFdxCoarse",
        "Derivative::dfdy_coarse": "dFdyCoarse",
        "Derivative::fwidth_coarse": "fwidthCoarse",
        "workgroup_memory_barrier": "memoryBarrierShared",
        "workgroup_memory_barrier_with_group_sync": "barrier",
        "device_memory_barrier": "memoryBarrierBuffer",
        "all_memory_barrier": "memoryBarrier",
        # spirv-std computes these with GLSL.std.450 SMax, SMin, UMax and UMin; CrossGL's operand types select them.
        "signed_max": "max",
        "signed_min": "min",
        "unsigned_max": "max",
        "unsigned_min": "min",
        "atomic_i_add": "atomicAdd",
        "atomic_f_add": "atomicAdd",
        "atomic_s_min": "atomicMin",
        "atomic_u_min": "atomicMin",
        "atomic_f_min": "atomicMin",
        "atomic_s_max": "atomicMax",
        "atomic_u_max": "atomicMax",
        "atomic_f_max": "atomicMax",
        "atomic_and": "atomicAnd",
        "atomic_or": "atomicOr",
        "atomic_xor": "atomicXor",
        "atomic_exchange": "atomicExchange",
        "subgroup_all": "allInvocations",
        "subgroup_any": "anyInvocation",
        "subgroup_all_equal": "allInvocationsEqual",
        "emit_vertex": "EmitVertex",
        "end_primitive": "EndPrimitive",
    },
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
# the GLSL feature list, built from the core GLSL specification, lacks. rust-gpu names one function for each operation
# and operand kind, as ``subgroup_i_add`` and ``subgroup_f_add``; in CrossGL the operand's type selects the
# instruction, and ``bool`` operands make the bitwise operations logical. Each operator maps to its wave name and the
# operand kinds rust-gpu spells it for, with their types.
_INTEGER = {"i_": "uint", "f_": "float"}
_ORDERED = {"s_": "int", "u_": "uint", "f_": "float"}
_BITWISE = {"": "uint", "logical_": "bool"}
_RUST_SUBGROUP_OPERATORS = {
    "add": ("Sum", _INTEGER),
    "mul": ("Product", _INTEGER),
    "min": ("Min", _ORDERED),
    "max": ("Max", _ORDERED),
    "and": ("BitAnd", _BITWISE),
    "or": ("BitOr", _BITWISE),
    "xor": ("BitXor", _BITWISE),
}
WAVES = {
    "rust": (
        {
            f"subgroup_{scan}{kind}{operator}": prefix + wave
            for operator, (wave, kinds) in _RUST_SUBGROUP_OPERATORS.items()
            for kind in kinds
            for scan, prefix in (
                ("", "WaveActive"),
                ("inclusive_", "WavePrefixInclusive"),
                (
                    "exclusive_",
                    (
                        "WavePrefix"
                        if operator in ("add", "mul")
                        else "WavePrefixExclusive"
                    ),
                ),
            )
        }
        | {
            "subgroup_elect": "WaveIsFirstLane",
            "subgroup_broadcast_first": "WaveReadLaneFirst",
            "subgroup_shuffle": "WaveReadLaneAt",
            "subgroup_shuffle_xor": "WaveShuffleXor",
            "subgroup_shuffle_up": "WaveShuffleUp",
            "subgroup_shuffle_down": "WaveShuffleDown",
            "subgroup_quad_broadcast": "QuadReadLaneAt",
            "subgroup_ballot": "WaveActiveBallot",
            "subgroup_inverse_ballot": "WaveInverseBallot",
            "subgroup_ballot_bit_extract": "WaveBallotBitExtract",
            "subgroup_ballot_bit_count": "WaveBallotBitCount",
            "subgroup_ballot_inclusive_bit_count": "WaveBallotInclusiveBitCount",
            "subgroup_ballot_exclusive_bit_count": "WaveBallotExclusiveBitCount",
            "subgroup_ballot_find_lsb": "WaveBallotFindLSB",
            "subgroup_ballot_find_msb": "WaveBallotFindMSB",
            # HLSL has no subgroup barriers, so CrossGL spells them as GL_KHR_shader_subgroup does.
            "subgroup_barrier": "subgroupBarrier",
            "subgroup_memory_barrier": "subgroupMemoryBarrier",
            "subgroup_memory_barrier_buffer": "subgroupMemoryBarrierBuffer",
            "subgroup_memory_barrier_image": "subgroupMemoryBarrierImage",
            "subgroup_memory_barrier_shared": "subgroupMemoryBarrierShared",
        }
    )
}


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
    # Mojo returns both parts of frexp and modf as a tuple.
    "mojo": {
        "frexp": "float frexp(float, out int)",
        "modf": "float modf(float, out float)",
    },
    # Rust returns the sine and the cosine as a tuple, where CrossGL, like HLSL, writes them to output parameters. Rust
    # names a geometry stream with a constant generic parameter, which CrossGL, like GLSL, passes as an argument.
    # glam's Vec3A is a 3-vector aligned to 16 bytes, as CrossGL's vec3 is in a buffer, so CrossGL converts to it
    # with the vec3 constructor, which also builds a vector from an array's or a slice's first three elements. glam's
    # extend, truncate and as_dvec3 are vector constructors in CrossGL, as in GLSL (GLSL 4.60 section 5.4.2): vec4
    # appends an element, vec2 keeps the first two and dvec3 widens each one.
    "rust": {
        "Float::sin_cos": "void sincos(float, out float, out float)",
        "Vec3::sin_cos": "void sincos(vec3, out vec3, out vec3)",
        "Vec3::to_vec3a": "vec3 vec3(vec3)",
        "Vec3::from_array": "vec3 vec3(float, float, float)",
        "Vec3::from_slice": "vec3 vec3(float, float, float)",
        "Vec3::extend": "vec4 vec4(vec3, float)",
        "Vec3::truncate": "vec2 vec2(vec3)",
        "Vec3::as_dvec3": "dvec3 dvec3(vec3)",
        "emit_stream_vertex": "void EmitStreamVertex(const int)",
        "end_stream_primitive": "void EndStreamPrimitive(const int)",
    },
}
# rust-gpu takes a clustered reduction's cluster size as a constant generic parameter, where CrossGL takes it as a
# constant argument.
COUNTERPARTS["rust"].update(
    {
        f"subgroup_clustered_{kind}{operator}": (
            f"{scalar} WaveClustered{wave}({scalar}, const uint)"
        )
        for operator, (wave, kinds) in _RUST_SUBGROUP_OPERATORS.items()
        for kind, scalar in kinds.items()
    }
)
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


# Built-in functions CrossGL has under a name the GLSL feature list lacks, with the same arguments and result: a
# CrossGL wave operation, a GLSL extension function, a function of another language that CrossGL defines, or a
# built-in value. A ``gl_`` name is a built-in value (GLSL 4.60 section 3.7 reserves the prefix), which the probe reads
# instead of calling, with a ``{}`` for each argument that indexes it.
OTHER_NAMES = {
    "slang": {
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
        # extension (WGSL, "Data Packing Built-in Functions"). pack4xU8Clamp clamps unsigned components, where
        # pack_clamp_u8 clamps signed ones, so it keeps its own name.
        "pack4xI8": "pack_s8",
        "pack4xU8": "pack_u8",
        "pack4xI8Clamp": "pack_clamp_s8",
        "unpack4xI8": "unpack_s8s32",
        "unpack4xU8": "unpack_u8u32",
    },
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
        "inf": "infinity()",
        "iota": "(0.0)",
        "mulwide": "(uint64({0}) * uint64({1}))",
        "align_down": (
            "(({0} / {1} - ({0} != {0} / {1} * {1} && ({0} < 0) != ({1} < 0) ? 1 : 0)) * {1})"
        ),
        "align_up": (
            "(({0} / {1} + ({0} != {0} / {1} * {1} && ({0} < 0) == ({1} < 0) ? 1 : 0)) * {1})"
        ),
        "ceildiv": "ceil({0} / {1})",
    },
    "rust": {
        # The C math functions of CUDA's device library and MSL compute what these methods of a scalar do. glam's
        # Vec3::is_finite tests every element at once, and is_finite_mask each one.
        "Float::exp_m1": "expm1",
        "Float::ln_1p": "log1p",
        "Float::is_finite": "isfinite",
        "Float::is_normal": "isnormal",
        "Float::is_sign_negative": "signbit",
        "Vec3::is_finite_mask": "isfinite",
        # spirv-std documents these as exact implementations of the HLSL barriers.
        "all_memory_barrier_with_group_sync": "AllMemoryBarrierWithGroupSync",
        "device_memory_barrier_with_group_sync": "DeviceMemoryBarrierWithGroupSync",
        # The probe passes memory_barrier subgroup scope and acquire-release semantics on uniform memory
        # (features/rust.py), the OpMemoryBarrier that GL_KHR_shader_subgroup's subgroupMemoryBarrierBuffer is.
        "memory_barrier": "subgroupMemoryBarrierBuffer",
        # OpKill, and the ray tracing and mesh shading instructions GL_EXT_ray_tracing and GL_EXT_mesh_shader name.
        "kill": "discard",
        "report_intersection": "reportIntersectionEXT",
        "ignore_intersection": "ignoreIntersectionEXT",
        "terminate_ray": "terminateRayEXT",
        "emit_mesh_tasks_ext": "EmitMeshTasksEXT",
        "set_mesh_outputs_ext": "SetMeshOutputsEXT",
        # OpEmitMeshTasksEXT with a payload operand, which CrossGL writes as DXR's DispatchMesh with the payload last.
        "emit_mesh_tasks_ext_payload": "DispatchMesh",
        # OpExecuteCallableKHR, which CrossGL calls by DXR's name, CallShader, with the shader index first. The probe
        # passes execute_callable the index 0 as its constant generic parameter.
        "execute_callable": "CallShader(0u, {0})",
        # A lane given as a constant broadcasts, as for Slang's WaveBroadcastLaneAt. subgroup_broadcast_const takes
        # the lane as a constant generic parameter, which the probe sets to 0.
        "subgroup_broadcast": "subgroupBroadcast",
        "subgroup_broadcast_const": "subgroupBroadcast({0}, 0u)",
        # The probe swaps in direction 0, OpGroupNonUniformQuadSwap's horizontal swap, which is HLSL's
        # QuadReadAcrossX.
        "subgroup_quad_swap": "QuadReadAcrossX",
        # spirv-std passes the new value before the comparator; CrossGL's atomicCompSwap, like GLSL's, takes the
        # comparator first. Both return the original value.
        "atomic_compare_exchange": "atomicCompSwap({0}, {2}, {1})",
        # OpIsHelperInvocationEXT, and OpReadClockKHR at the subgroup scope the probe passes, as a 64-bit integer or
        # as two 32-bit halves (GL_ARB_shader_clock).
        "is_helper_invocation": "IsHelperLane",
        "read_clock_khr": "clockARB",
        "read_clock_uvec2_khr": "clock2x32ARB",
        # num-traits computes recip as 1 / x and glam's Vec3::recip divides each element the same way, as HLSL's
        # rcp does. rust-gpu lowers powi to GLSL.std.450 Pow, CrossGL's pow, which takes a float exponent.
        "Float::recip": "rcp",
        "Vec3::recip": "rcp",
        "Float::powi": "pow({0}, float({1}))",
        # Rust's fract truncates, where CrossGL's fract floors, and num-traits' log takes a base: num-traits computes
        # them as self - trunc(self) and self.ln() / base.ln() (src/float.rs), and glam's Vec3::fract as
        # self - self.trunc() (src/f32/vec3.rs).
        "Float::fract": "({0} - trunc({0}))",
        "Vec3::fract": "({0} - trunc({0}))",
        "Float::log": "(log({0}) / log({1}))",
        # core's sign tests and constants for f32, which num-traits' Float returns: is_sign_positive is the
        # negation of is_sign_negative, a subnormal is a non-zero value smaller in magnitude than MIN_POSITIVE,
        # and signum is NaN for NaN and otherwise 1.0 with the operand's sign.
        "Float::is_sign_positive": "!signbit({0})",
        "Float::is_subnormal": "({0} != 0.0 && abs({0}) < 1.17549435e-38)",
        "Float::signum": "(isnan({0}) ? {0} : copysign(1.0, {0}))",
        "Float::max_value": "(3.40282347e38)",
        "Float::min_value": "(-3.40282347e38)",
        "Float::min_positive_value": "(1.17549435e-38)",
        "Float::neg_infinity": "(-infinity())",
        "Float::neg_zero": "(-0.0)",
        # glam's Vec3 methods that glam writes in Rust arithmetic (glam 0.34, src/f32/vec3.rs), written with the same
        # CrossGL operations. glam's length_squared is dot(v, v). Its max_element and min_element compare with > and
        # <, which pick the same element as max and min unless an element is NaN.
        "Vec3::element_sum": "({0}.x + {0}.y + {0}.z)",
        "Vec3::element_product": "({0}.x * {0}.y * {0}.z)",
        "Vec3::max_element": "max({0}.x, max({0}.y, {0}.z))",
        "Vec3::min_element": "min({0}.x, min({0}.y, {0}.z))",
        # max_position and min_position keep the first of equal elements: glam replaces its candidate only when a
        # later element compares greater or less.
        "Vec3::max_position": (
            "({0}.z > ({0}.y > {0}.x ? {0}.y : {0}.x) ? 2u : ({0}.y > {0}.x ? 1u : 0u))"
        ),
        "Vec3::min_position": (
            "({0}.z < ({0}.y < {0}.x ? {0}.y : {0}.x) ? 2u : ({0}.y < {0}.x ? 1u : 0u))"
        ),
        "Vec3::with_x": "vec3({1}, {0}.y, {0}.z)",
        "Vec3::with_y": "vec3({0}.x, {1}, {0}.z)",
        "Vec3::with_z": "vec3({0}.x, {0}.y, {1})",
        "Vec3::midpoint": "(({0} + {1}) * 0.5)",
        "Vec3::lerp_monotonic": "({0} + ({1} - {0}) * {2})",
        "Vec3::length_recip": "(1.0 / length({0}))",
        "Vec3::dot_into_vec": "vec3(dot({0}, {1}))",
        "Vec3::to_homogeneous": "vec4({0}, 1.0)",
        "Vec3::from_homogeneous": "({0}.xyz / {0}.w)",
        "Vec3::as_ivec3": "ivec3({0})",
        "Vec3::as_uvec3": "uvec3({0})",
        "Vec3::is_finite": "all(isfinite({0}))",
        "Vec3::is_nan": "any(isnan({0}))",
        "Vec3::is_normalized": "(abs(dot({0}, {0}) - 1.0) <= 0.0002)",
        "Vec3::is_negative_mask": (
            "bvec3(signbit({0}.x), signbit({0}.y), signbit({0}.z))"
        ),
        "Vec3::is_negative_bitmask": (
            "(uint(signbit({0}.x)) | (uint(signbit({0}.y)) << 1u) | (uint(signbit({0}.z)) << 2u))"
        ),
        "Vec3::signum": (
            "vec3(isnan({0}.x) ? {0}.x : copysign(1.0, {0}.x), isnan({0}.y) ? {0}.y : copysign(1.0, {0}.y), "
            "isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z))"
        ),
        "Vec3::abs_diff_eq": "all(lessThanEqual(abs({0} - {1}), vec3({2})))",
        "Vec3::project_onto": "({1} * (dot({0}, {1}) / dot({1}, {1})))",
        "Vec3::project_onto_normalized": "({1} * dot({0}, {1}))",
        "Vec3::reject_from": "({0} - {1} * (dot({0}, {1}) / dot({1}, {1})))",
        "Vec3::reject_from_normalized": "({0} - {1} * dot({0}, {1}))",
        "Vec3::normalize_or": (
            "(isfinite(1.0 / length({0})) && 1.0 / length({0}) > 0.0 ? {0} * (1.0 / length({0})) : {1})"
        ),
        "Vec3::normalize_or_zero": (
            "(isfinite(1.0 / length({0})) && 1.0 / length({0}) > 0.0 ? {0} * (1.0 / length({0})) : vec3(0.0))"
        ),
        "Vec3::clamp_length_max": (
            "(dot({0}, {0}) > {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) : {0})"
        ),
        "Vec3::clamp_length_min": (
            "(dot({0}, {0}) < {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) : {0})"
        ),
        "Vec3::clamp_length": (
            "(dot({0}, {0}) < {1} * {1} ? {1} * ({0} / sqrt(dot({0}, {0}))) "
            ": (dot({0}, {0}) > {2} * {2} ? {2} * ({0} / sqrt(dot({0}, {0}))) : {0}))"
        ),
        "Vec3::move_towards": (
            "(length({1} - {0}) <= {2} || length({1} - {0}) <= 0.0001 ? {1} "
            ": {0} + ({1} - {0}) / length({1} - {0}) * {2})"
        ),
        "Vec3::angle_to": "atan(dot(cross({0}, {1}), {2}), dot({0}, {1}))",
        "Vec3::any_orthogonal_vector": (
            "(abs({0}.x) > abs({0}.y) ? vec3(-{0}.z, 0.0, {0}.x) : vec3(0.0, {0}.z, -{0}.y))"
        ),
        # glam's no_std build, the one rust-gpu compiles, divides and takes the remainder per element with Rust's
        # truncating %, CrossGL's fmod: div_euclid corrects trunc(a / b) by one away from b when the remainder is
        # negative, and rem_euclid adds |b| to a negative remainder (glam src/f32/math.rs, libm_math).
        "Vec3::div_euclid": (
            "vec3("
            + ", ".join(
                f"(fmod({{0}}.{c}, {{1}}.{c}) < 0.0 ? trunc({{0}}.{c} / {{1}}.{c}) + ({{1}}.{c} > 0.0 ? -1.0 : 1.0)"
                f" : trunc({{0}}.{c} / {{1}}.{c}))"
                for c in "xyz"
            )
            + ")"
        ),
        "Vec3::rem_euclid": (
            "vec3("
            + ", ".join(
                f"(fmod({{0}}.{c}, {{1}}.{c}) < 0.0 ? fmod({{0}}.{c}, {{1}}.{c}) + abs({{1}}.{c})"
                f" : fmod({{0}}.{c}, {{1}}.{c}))"
                for c in "xyz"
            )
            + ")"
        ),
        # Pixar's orthonormal basis, as glam writes it: with s = signum(z) and a = -1 / (s + z), the vector is
        # (x * y * a, s + y * y * a, -y).
        "Vec3::any_orthonormal_vector": (
            "vec3({0}.x * {0}.y * (-1.0 / ((isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.z)), "
            "(isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.y * {0}.y "
            "* (-1.0 / ((isnan({0}.z) ? {0}.z : copysign(1.0, {0}.z)) + {0}.z)), -{0}.y)"
        ),
        "Vec3::rotate_x": (
            "vec3({0}.x, {0}.y * cos({1}) - {0}.z * sin({1}), {0}.y * sin({1}) + {0}.z * cos({1}))"
        ),
        "Vec3::rotate_y": (
            "vec3({0}.x * cos({1}) + {0}.z * sin({1}), {0}.y, {0}.x * -sin({1}) + {0}.z * cos({1}))"
        ),
        "Vec3::rotate_z": (
            "vec3({0}.x * cos({1}) - {0}.y * sin({1}), {0}.x * sin({1}) + {0}.y * cos({1}), {0}.z)"
        ),
    },
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
