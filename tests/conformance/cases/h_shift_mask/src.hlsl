// Hazard H2: HLSL/D3D shifts use only the low 5 bits of the shift count (x << 33 == x << 1).
StructuredBuffer<uint> xs : register(t0);
RWStructuredBuffer<uint> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = xs[2 * id.x];
    uint s = xs[2 * id.x + 1];
    o[id.x] = x << s;
}
