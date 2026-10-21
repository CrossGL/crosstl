// CUDA layout: float3 is 12 bytes with 4-byte alignment and float2 is 8-byte aligned, so S is 24 bytes with b at
// byte 16; bytes 12-15 are padding. HLSL structured buffers put b at 12 in 20-byte elements, std430 and MSL's
// natural layout put b at 16 in 32-byte elements.
struct S {
    float3 a;
    float2 b;
};
__global__ void hazard(const S *src, S *dst) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    S s = src[i];
    S t;
    t.a = make_float3(s.a.z + s.b.x, s.a.y, s.a.x - s.b.y);
    t.b = make_float2(s.a.x * 2.0f, s.b.y + 0.5f);
    dst[i] = t;
    dst[i].b.x += 1.0f;
}
