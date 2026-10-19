// HLSL StructuredBuffer<float3>: 12-byte elements.
StructuredBuffer<float3> a : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = a[id.x].y;
}
