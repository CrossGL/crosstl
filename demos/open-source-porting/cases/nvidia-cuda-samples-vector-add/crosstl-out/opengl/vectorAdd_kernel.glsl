#version 450 core
layout(std430, binding = 0) buffer ABuffer { float A[]; };
layout(std430, binding = 1) buffer BBuffer { float B[]; };
layout(std430, binding = 2) buffer CBuffer { float C[]; };
// Constant Buffers
layout(std140, binding = 0) uniform vectorAdd_numElements_Args {
    int vectorAdd_numElements_Args_numElements;
};
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uvec3 thread_id = gl_GlobalInvocationID;
    uvec3 block_id = gl_WorkGroupID;
    uvec3 thread_local_id = gl_LocalInvocationID;
    uvec3 block_dim = gl_WorkGroupSize;
    int i = int(((gl_WorkGroupSize.x * gl_WorkGroupID.x) + gl_LocalInvocationID.x));
    if ((i < vectorAdd_numElements_Args_numElements)) {
        C[i] = (A[i] + B[i]);
    }
}
