"""Wave, warp, bit-cast, constructor and half-vector tables of the CUDA generator."""

CUDA_WAVE_OP_ARITIES = {
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
CUDA_WAVE_PREDICATE_ARGUMENT_OPS = {
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveBallot",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}
CUDA_WAVE_UINT_RESULT_OPS = {
    "WaveGetLaneCount",
    "WaveGetLaneIndex",
    "WaveActiveCountBits",
    "WavePrefixCountBits",
    "WaveMultiPrefixCountBits",
}
CUDA_WAVE_BOOL_RESULT_OPS = {
    "WaveIsFirstLane",
    "WaveActiveAllTrue",
    "WaveActiveAnyTrue",
    "WaveActiveAllEqual",
}
CUDA_WAVE_UVEC4_RESULT_OPS = {
    "WaveActiveBallot",
    "WaveMatch",
}
CUDA_WARP_SYNC_BUILTIN_ARITIES = {
    "__shfl_down_sync": (3, 4),
}
CUDA_BITCAST_FUNCTION_TARGETS = {
    "floatBitsToInt": "int",
    "floatBitsToUint": "uint",
    "intBitsToFloat": "float",
    "uintBitsToFloat": "float",
    "asfloat": "float",
    "asint": "int",
    "asuint": "uint",
}
CUDA_64BIT_BITCAST_FUNCTIONS = {
    "doubleBitsToLong": ("__double_as_longlong", "long long"),
    "longBitsToDouble": ("__longlong_as_double", "double"),
}
CUDA_INTEGER_BIT_FUNCTION_ALIASES = {
    "countbits": "bitCount",
    "reversebits": "bitfieldReverse",
    "firstbitlow": "findLSB",
    "firstbithigh": "findMSB",
}
# CrossGL's sized scalar constructors have no CUDA spelling, and several map to multi-word C++ types
# (``unsigned int``, ``long long``) that cannot take the functional-cast form ``T(x)``, so they lower to
# ``static_cast<T>(x)`` as the HIP generator's do.
CUDA_SCALAR_CONSTRUCTOR_TYPE_ALIASES = {
    "i8",
    "u8",
    "i16",
    "u16",
    "i32",
    "u32",
    "i64",
    "u64",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int64",
    "uint64",
    "f32",
    "f64",
}
CUDA_FP16_VECTOR_TYPES = {
    "cgl_half3": 3,
    "cgl_half4": 4,
    "vec3<f16>": 3,
    "vec3<float16>": 3,
    "vec3<half>": 3,
    "f16vec3": 3,
    "float16vec3": 3,
    "half3": 3,
    "vec4<f16>": 4,
    "vec4<float16>": 4,
    "vec4<half>": 4,
    "f16vec4": 4,
    "float16vec4": 4,
    "half4": 4,
}
