// HLSL structured buffers place every 32-bit member at a multiple of four bytes: P is 28 bytes, with b at
// byte 4, c at 16 and d at 20. std430 would align b to 16, d to 8 and the element to 16 (48 bytes).
struct P {
    float a;
    float3 b;
    float c;
    float2 d;
};
StructuredBuffer<P> src : register(t0);
RWStructuredBuffer<P> dst : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    P p = src[id.x];
    P q;
    q.a = p.d.y;
    q.b = p.b.zyx;
    q.c = p.a + p.c;
    q.d = float2(p.b.x, p.c);
    dst[id.x] = q;
    dst[id.x].b.y += 0.5;
}
