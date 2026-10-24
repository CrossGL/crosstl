"""How CrossGL names and calls the built-in functions of each backend.

A coverage probe calls a backend built-in through CrossGL. Most have the same name and arguments in both, so the
probe uses the backend's spelling. The tables here list the exceptions: the functions CrossGL spells as GLSL does
(:data:`RENAMES`) and those it writes with other arguments (:data:`COUNTERPARTS`). The feature lists use them to give
each feature its CrossGL spelling, and issue reconciliation uses them to recognise that spelling in a diagnostic.
"""

from __future__ import annotations

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
    # CUDA after :mod:`.features.cuda` drops the single-precision suffix: ``fabsf`` arrives here as ``fabs``.
    "cuda": {
        "fabs": "abs",
        "fmin": "min",
        "fmax": "max",
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
        "fmax": "max",
        "fmin": "min",
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
# HIP spells its device functions the way CUDA does, and Slang its built-in functions the way HLSL does.
RENAMES["hip"] = RENAMES["cuda"]
RENAMES["slang"] = RENAMES["hlsl"]


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
    # with the vec3 constructor, which also builds a vector from an array's or a slice's first three elements.
    "rust": {
        "Float::sin_cos": "void sincos(float, out float, out float)",
        "Vec3::sin_cos": "void sincos(vec3, out vec3, out vec3)",
        "Vec3::to_vec3a": "vec3 vec3(vec3)",
        "Vec3::from_array": "vec3 vec3(float, float, float)",
        "Vec3::from_slice": "vec3 vec3(float, float, float)",
        "emit_stream_vertex": "void EmitStreamVertex(const int)",
        "end_stream_primitive": "void EndStreamPrimitive(const int)",
    },
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
COUNTERPARTS["metal"].update(
    {
        f"atomic_{op}_explicit": f"int {name}(inout int, int)"
        for op, name in _METAL_ATOMICS.items()
    }
)


def spelling(language: str, name: str, glsl_names: set[str] | frozenset[str]) -> str:
    """The CrossGL name of ``language``'s built-in ``name``."""
    renamed = RENAMES.get(language, {}).get(name, name)
    return renamed if renamed in glsl_names else name
