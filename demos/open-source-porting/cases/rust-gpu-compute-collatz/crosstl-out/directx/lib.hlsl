
static const int Option_Some = 0;
static const int Option_Absent = 1;

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

Option_u32 Option_u32_Absent_make() {
    Option_u32 result;
    result.variant = Option_Absent;
    result.Some_0 = uint(0);
    return result;
}

RWStructuredBuffer<uint> prime_indices : register(u0);
Option_u32 collatz(uint n);

uint collatz_unwrap_or(uint n, uint _rust_option_fallback);

Option_u32 collatz(uint n) {
    int i = 0;
    if (n == 0) {
        return Option_u32_Absent_make();
    }
    while ((n != 1)) {
        if ((n % 2) == 0) {
            n = (n / 2);
        } else {
            if (n >= 1431655765) {
                return Option_u32_Absent_make();
            }
            n = ((3 * n) + 1);
        }
        i += 1;
    }
    return Option_u32_Some_make(i);
}

uint collatz_unwrap_or(uint n, uint _rust_option_fallback) {
    int i = 0;
    if (n == 0) {
        return _rust_option_fallback;
    }
    while ((n != 1)) {
        if ((n % 2) == 0) {
            n = (n / 2);
        } else {
            if (n >= 1431655765) {
                return _rust_option_fallback;
            }
            n = ((3 * n) + 1);
        }
        i += 1;
    }
    return i;
}

// Compute Shader
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint index = uint(id.x);
    prime_indices[index] = collatz_unwrap_or(prime_indices[index], 4294967295u);
}
