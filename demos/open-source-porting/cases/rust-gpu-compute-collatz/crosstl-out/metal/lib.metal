
#include <metal_stdlib>
using namespace metal;

uint collatz(uint n)  {
    int i = 0;
    if (n == 0) {
        return uint(0);
    }
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            if (n >= 1431655765) {
                return uint(0);
            }
            n = 3 * n + 1;
        }
        i += 1;
    }
    return i;
}

uint collatz_unwrap_or(uint n, uint _rust_option_fallback)  {
    int i = 0;
    if (n == 0) {
        return _rust_option_fallback;
    }
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            if (n >= 1431655765) {
                return _rust_option_fallback;
            }
            n = 3 * n + 1;
        }
        i += 1;
    }
    return i;
}

// Compute Shader
kernel void kernel_main_cs(uint3 id [[thread_position_in_grid]], device uint* prime_indices [[buffer(0)]]) {
    uint index = uint(id.x);
    prime_indices[index] = collatz_unwrap_or(prime_indices[index], 4294967295u);
}

