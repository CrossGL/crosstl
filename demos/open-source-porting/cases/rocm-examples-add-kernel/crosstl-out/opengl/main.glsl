#version 450 core
layout(std430, binding = 0) buffer aBuffer { float a[]; };
layout(std430, binding = 1) readonly buffer bBuffer { float b[]; };
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    int global_idx = int((gl_LocalInvocationID.x + (gl_WorkGroupID.x * gl_WorkGroupSize.x)));
    a[global_idx] += b[global_idx];
}
