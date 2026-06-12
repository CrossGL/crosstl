
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct saxpy_Args {
    float a;
};
// Compute Shader
kernel void kernel_saxpy(uint3 thread_position_in_grid [[thread_position_in_grid]], constant saxpy_Args& saxpy_Args [[buffer(2)]], device float* x [[buffer(0)]], device float* y [[buffer(1)]]) {
    int gid = thread_position_in_grid.x;
    y[gid] = fma(saxpy_Args.a, x[gid], y[gid]);
}
