
static const uint BUFFER_ELEMENTS = 32;

RWByteAddressBuffer pos : register(u0);
uint fibonacci(uint n) {
    if ((n <= 1)) {
        return n;
    }
    uint curr = 1;
    uint prev = 1;
    for (uint i = 2; (i < n); ++i) {
        uint temp = curr;
        curr += prev;
        prev = temp;
    }
    return curr;
}

// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint index = gl_GlobalInvocationID.x;
    if ((index >= BUFFER_ELEMENTS)) {
        return;
    }
    pos.Store((index * 4), fibonacci(pos.Load((index * 4))));
}
