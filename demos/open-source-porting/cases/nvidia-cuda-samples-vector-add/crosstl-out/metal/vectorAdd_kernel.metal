
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct vectorAdd_Args {
    int numElements;
};
// Compute Shader
kernel void kernel_vectorAdd(uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 threads_per_threadgroup [[threads_per_threadgroup]], constant vectorAdd_Args& vectorAdd_Args [[buffer(3)]], device float* A [[buffer(0)]], device float* B [[buffer(1)]], device float* C [[buffer(2)]]) {
    int i = threads_per_threadgroup.x * threadgroup_position_in_grid.x + thread_position_in_threadgroup.x;
    if (i < vectorAdd_Args.numElements) {
        C[i] = A[i] + B[i];
    }
}
