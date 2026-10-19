// Reading A (D3D11.3 section 5.3.4): out-of-bounds structured-buffer reads return 0. The buffer has 64 elements.
StructuredBuffer<float> a : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = a[id.x + 60];
}
