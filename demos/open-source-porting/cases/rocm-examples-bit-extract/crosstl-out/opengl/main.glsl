#version 450 core
#extension GL_ARB_gpu_shader_int64 : require
layout(std430, binding = 0) buffer d_outputBuffer { uint d_output[]; };
layout(std430, binding = 1) readonly buffer d_inputBuffer { uint d_input[]; };
// Constant Buffers
layout(std140, binding = 2) uniform bit_extract_kernel_size_Args {
    uint64_t size;
};
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint64_t offset = uint64_t(gl_GlobalInvocationID.x);
    uint64_t stride = uint64_t((gl_WorkGroupSize.x * gl_NumWorkGroups.x));
    for (uint64_t i = offset; (i < size); i += stride) {
        d_output[uint(i)] = ((d_input[uint(i)] >> 8) & 15u);
    }
}
