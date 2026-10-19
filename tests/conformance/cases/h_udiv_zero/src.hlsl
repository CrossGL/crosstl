// Hazard H3: D3D unsigned division by zero yields 0xFFFFFFFF.
StructuredBuffer<uint> ab : register(t0);
RWStructuredBuffer<uint> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = ab[2 * id.x] / ab[2 * id.x + 1];
}
