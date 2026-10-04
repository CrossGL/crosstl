#version 450 core
in vec3 position;
out vec3 viewDir;
layout(std140, binding = 0) uniform Camera {
    mat4 invViewMatrix;
} camera;
float crossgl_round_half1(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        return uintBitsToFloat(sign | (magnitude == 0x7f800000u ? 0x7f800000u : 0x7fc00000u));
    }
    if (magnitude >= 0x477ff000u) {
        return uintBitsToFloat(sign | 0x7f800000u);
    }
    if (magnitude >= 0x38800000u) {
        uint rounded = (magnitude + 0xfffu + ((magnitude >> 13u) & 1u)) & 0xffffe000u;
        return uintBitsToFloat(sign | rounded);
    }
    if (magnitude < 0x33000000u) {
        return uintBitsToFloat(sign);
    }
    // Round the subnormal significand before rebuilding its exact float32 value.
    uint shift = 126u - (magnitude >> 23u);
    uint significand = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded = significand >> shift;
    uint remainder = significand & ((1u << shift) - 1u);
    uint midpoint = 1u << (shift - 1u);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1u) != 0u)) {
        rounded += 1u;
    }
    if (rounded == 0u) {
        return uintBitsToFloat(sign);
    }
    uint leading = uint(findMSB(rounded));
    uint result = ((leading + 103u) << 23u) | ((rounded << (23u - leading)) & 0x7fffffu);
    return uintBitsToFloat(sign | result);
}

vec3 crossgl_round_half3(vec3 value) {
    return vec3(crossgl_round_half1(value.x), crossgl_round_half1(value.y), crossgl_round_half1(value.z));
}

// Vertex Shader
void main() {
    viewDir = crossgl_round_half3(vec3(normalize((camera.invViewMatrix[3].xyz - position))));
    return;
}
