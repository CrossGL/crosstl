// Reading A (D3D11.3 section 3.1.3.3): 32-bit denormals are flushed to sign-preserved zero.
StructuredBuffer<float> a : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = a[id.x] * 0.5f;
}
