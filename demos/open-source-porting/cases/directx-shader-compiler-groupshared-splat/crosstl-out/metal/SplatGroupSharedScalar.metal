
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void kernel_main() {
    __attribute__((unused)) threadgroup int a;
    a = 123;
    __attribute__((unused)) int4 x = int4(a);
}
