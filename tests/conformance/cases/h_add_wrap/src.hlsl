// Hazard I1: signed addition overflow. HLSL/DXIL integer arithmetic wraps (two's complement).
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int x = a[id.x];
    o[id.x] = (x + 1 > x) ? 1 : 0;
}
