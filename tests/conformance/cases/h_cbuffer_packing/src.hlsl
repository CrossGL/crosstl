// HLSL constant-buffer packing: a member never straddles a 16-byte register and every array element starts one,
// but the last element is not padded, so tail packs at 56 next to offsets[2]; the column-major float3x3 takes the
// xyz of three registers from 64, so k packs at 108; weights starts a register at 128.
cbuffer Params : register(b2) {
    float3 scale;
    float bias;
    float2 offsets[3];
    float2 tail;
    float3x3 rot;
    float k;
    int mode;
    float weights[2];
};
StructuredBuffer<float> x : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float v = x[i];
    float3 r = mul(rot, float3(v, 1.0, 2.0));
    o[8 * i + 0] = v * scale.x + bias;
    o[8 * i + 1] = v * scale.y + scale.z;
    o[8 * i + 2] = offsets[i % 3].x;
    o[8 * i + 3] = offsets[i % 3].y + tail.y;
    o[8 * i + 4] = r[i % 3];
    o[8 * i + 5] = float(mode) + k;
    o[8 * i + 6] = weights[i % 2];
    o[8 * i + 7] = tail.x;
}
