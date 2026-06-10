
StructuredBuffer<float> buffer0 : register(t0);
StructuredBuffer<float> buffer1 : register(t1);
RWStructuredBuffer<float> result : register(u0);
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 threadId : SV_DispatchThreadID) {
    uint index = threadId.x;
    result[index] = (buffer0[index] + buffer1[index]);
}
