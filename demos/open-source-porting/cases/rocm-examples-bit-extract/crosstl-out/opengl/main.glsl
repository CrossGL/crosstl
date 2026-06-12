#version 450 core
layout(std430, binding = 0) buffer d_outputBuffer { uint d_output[]; };
layout(std430, binding = 1) readonly buffer d_inputBuffer { uint d_input[]; };
// Constant Buffers
layout(std140, binding = 2) uniform bit_extract_kernel_size_Args {
    uint size;
};
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint offset = ((gl_WorkGroupID.x * gl_WorkGroupSize.x) + gl_LocalInvocationID.x);
    uint stride = (gl_WorkGroupSize.x * gl_NumWorkGroups.x);
    for (uint i = offset; (i < size); i += stride) {
        d_output[i] = ((d_input[i] >> 8) & 15u);
    }
}
