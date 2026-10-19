
float16_t __crossgl_round_half1(float value) {
    uint bits = asuint(value);
    uint sign = (bits >> 16u) & 0x8000u;
    uint magnitude = bits & 0x7fffffffu;
    uint result;
    if (magnitude >= 0x7f800000u) {
        result = magnitude == 0x7f800000u ? 0x7c00u : 0x7e00u;
    } else if (magnitude >= 0x477ff000u) {
        result = 0x7c00u;
    } else if (magnitude >= 0x38800000u) {
        result = (magnitude - 0x38000000u + 0xfffu + ((magnitude >> 13u) & 1u)) >> 13u;
    } else if (magnitude < 0x33000000u) {
        result = 0u;
    } else {
        // Round the subnormal significand, including the smallest normal carry.
        uint shift = 126u - (magnitude >> 23u);
        uint significand = (magnitude & 0x7fffffu) | 0x800000u;
        result = significand >> shift;
        uint remainder = significand & ((1u << shift) - 1u);
        uint midpoint = 1u << (shift - 1u);
        if (remainder > midpoint || (remainder == midpoint && (result & 1u) != 0u)) {
            result += 1u;
        }
    }
    return asfloat16(uint16_t(sign | result));
}
float16_t3 __crossgl_round_half3(float3 value) {
    return float16_t3(__crossgl_round_half1(value.x), __crossgl_round_half1(value.y), __crossgl_round_half1(value.z));
}

struct Camera {
    float4x4 invViewMatrix: TEXCOORD0;
};
struct Input {
    float3 position: POSITION;
};
struct Output {
    float16_t3 viewDir: TEXCOORD0;
};
ConstantBuffer<Camera> camera : register(b0);
// Vertex Shader
Output VSMain(Input in_) {
    Output out_;
    out_.viewDir = __crossgl_round_half3(float3(normalize((camera.invViewMatrix[3].xyz - in_.position))));
    return out_;
}
