#version 450

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer InputValues {
    uint values[];
} input_values;
layout(std430, binding = 1) buffer OutputValues {
    uint values[];
} output_values;
layout(constant_id = 7) const uint MULTIPLIER = 3u;

void main() {
    uint index = gl_GlobalInvocationID.x;
    output_values.values[index] = input_values.values[index] * MULTIPLIER;
}
