"""Wave, warp, bit-cast, half-vector and constructor tables of the HIP generator."""

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
HIP_64BIT_BITCAST_FUNCTIONS = {
    "doubleBitsToLong": ("__double_as_longlong", "long long"),
    "longBitsToDouble": ("__longlong_as_double", "double"),
}
HIP_INTEGER_BIT_FUNCTION_ALIASES = {
    "countbits": "bitCount",
    "reversebits": "bitfieldReverse",
    "firstbitlow": "findLSB",
    "firstbithigh": "findMSB",
}
HIP_FP16_VECTOR_TYPES = {
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
    "int64",
    "uint64",
    "f32",
    "f64",
}
