
#include <metal_stdlib>
using namespace metal;

void computeMain(uint3 threadId [[thread_position_in_grid]], const device float* buffer0, const device float* buffer1, device float* result)  {
    uint index = threadId.x;
    result[index] = buffer0[index] + buffer1[index];
}

// Compute Shader
kernel void kernel_main(const device float* buffer0 [[buffer(0)]], const device float* buffer1 [[buffer(1)]], device float* result [[buffer(2)]]) {
}

