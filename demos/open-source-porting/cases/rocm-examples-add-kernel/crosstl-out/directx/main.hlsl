
RWStructuredBuffer<float> a : register(u0);
StructuredBuffer<float> b : register(t1);
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int global_idx = dispatchThreadID.x;
    a[global_idx] += b[global_idx];
}
