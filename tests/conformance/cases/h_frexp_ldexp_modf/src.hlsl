// frexp, modf and ldexp with HLSL's types: the frexp and ldexp exponents are floats, and modf writes its whole
// part to an int. HLSL defines ldexp(x, e) as x * 2^e, so a fractional exponent scales by a fractional power of
// two, which exp2 computes within the precision Vulkan and D3D allow; every other column is exact.
StructuredBuffer<float> a : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float x = a[i];
    float e;
    float significand = frexp(x, e);
    int whole;
    float fraction = modf(x, whole);
    float3 e3;
    float3 significand3 = frexp(float3(x, 2.0 * x, 0.5), e3);
    o[6 * i] = significand;
    o[6 * i + 1] = e;
    o[6 * i + 2] = fraction;
    o[6 * i + 3] = float(whole);
    o[6 * i + 4] = significand3.y + e3.y - e3.z;
    o[6 * i + 5] = ldexp(x, float(i % 9u) * 0.5 - 2.0);
}
