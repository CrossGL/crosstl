
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct vectorAdd_Args {
    int numElements;
};
// Compute Shader
kernel void kernel_vectorAdd(uint3 thread_position_in_grid [[thread_position_in_grid]], constant vectorAdd_Args& vectorAdd_Args [[buffer(3)]], device float* A [[buffer(0)]], device float* B [[buffer(1)]], device float* C [[buffer(2)]]) {
    int i = thread_position_in_grid.x;
    if (i < vectorAdd_Args.numElements) {
        C[i] = A[i] + B[i];
    }
}
