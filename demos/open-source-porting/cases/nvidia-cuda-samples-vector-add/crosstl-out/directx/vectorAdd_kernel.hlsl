
RWStructuredBuffer<float> A : register(u0);
RWStructuredBuffer<float> B : register(u1);
RWStructuredBuffer<float> C : register(u2);
// Constant Buffers
cbuffer vectorAdd_Args : register(b3) {
    int numElements;
};
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    int i = ((uint3(1, 1, 1).x * groupID.x) + groupThreadID.x);
    if (i < numElements) {
        C[i] = (A[i] + B[i]);
    }
}
