
RWStructuredBuffer<float> a : register(u0);
StructuredBuffer<float> b : register(t1);
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    int global_idx = (groupThreadID.x + (groupID.x * uint3(1, 1, 1).x));
    a[global_idx] += b[global_idx];
}
