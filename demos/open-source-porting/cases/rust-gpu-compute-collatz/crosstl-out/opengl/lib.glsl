#version 450 core
const int Option_Some = 0;
const int Option_None = 1;

struct Option_u32 {
    int variant;
    uint Some_0;
};

Option_u32 Option_u32_Some_make(uint payload0) {
    Option_u32 result;
    result.variant = Option_Some;
    result.Some_0 = payload0;
    return result;
}

Option_u32 Option_u32_None_make() {
    Option_u32 result;
    result.variant = Option_None;
    result.Some_0 = uint(0);
    return result;
}

layout(std430, binding = 0) buffer prime_indicesBuffer { uint prime_indices[]; };
Option_u32 collatz(uint n) {
    int i = 0;
    if ((n == 0)) {
        return Option_u32_None_make();
    }
    while ((n != 1)) {
        if (((n % 2) == 0)) {
            n = (n / 2);
        } else {
            if ((n >= 1431655765)) {
                return Option_u32_None_make();
            }
            n = ((3 * n) + 1);
        }
        i += 1;
    }
    return Option_u32_Some_make(i);
}

uint collatz_unwrap_or(uint n, uint _rust_option_fallback) {
    int i = 0;
    if ((n == 0)) {
        return _rust_option_fallback;
    }
    while ((n != 1)) {
        if (((n % 2) == 0)) {
            n = (n / 2);
        } else {
            if ((n >= 1431655765)) {
                return _rust_option_fallback;
            }
            n = ((3 * n) + 1);
        }
        i += 1;
    }
    return i;
}

// Compute Shader
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint index = uint(gl_GlobalInvocationID.x);
    prime_indices[index] = collatz_unwrap_or(prime_indices[index], 4294967295u);
}
