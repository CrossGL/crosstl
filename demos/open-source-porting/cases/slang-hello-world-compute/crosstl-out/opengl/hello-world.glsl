
#version 450 core
layout(std430, binding = 0) readonly buffer buffer0Buffer { float buffer0[]; };
layout(std430, binding = 1) readonly buffer buffer1Buffer { float buffer1[]; };
layout(std430, binding = 2) buffer resultBuffer { float result[]; };
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint index = gl_GlobalInvocationID.x;
    result[index] = (buffer0[index] + buffer1[index]);
}
