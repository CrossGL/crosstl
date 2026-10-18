// Hazard F1: `precise` forbids contracting a*b - c into a fused multiply-add (HLSL precise keyword).
// With c = fl(a*b), the unfused result is exactly 0; a fused result equals the rounding error of a*b.
StructuredBuffer<float> abc : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float a = abc[3 * id.x];
    float b = abc[3 * id.x + 1];
    float c = abc[3 * id.x + 2];
    precise float r = a * b - c;
    o[id.x] = r;
}
