// Hazard H6: D3D min returns the non-NaN operand.
StructuredBuffer<float> xy : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = min(xy[2 * id.x], xy[2 * id.x + 1]);
}
