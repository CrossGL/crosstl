
#include <metal_stdlib>
using namespace metal;

__attribute__((unused)) constant uint BUFFER_ELEMENTS = 32;

uint fibonacci(uint n)  {
    if (n <= 1) {
        return n;
    }
    uint curr = 1;
    uint prev = 1;
    for (uint i = 2; i < n; ++i) {
        uint temp = curr;
        curr += prev;
        prev = temp;
    }
    return curr;
}

// Compute Shader
kernel void kernel_main(uint3 thread_position_in_grid [[thread_position_in_grid]], device uchar* pos [[buffer(0)]]) {
    uint index = thread_position_in_grid.x;
    if (index >= BUFFER_ELEMENTS) {
        return;
    }
    (*reinterpret_cast<device uint*>(pos + (index * 4))) = fibonacci((*reinterpret_cast<const device uint*>(pos + (index * 4))));
}
