
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> y : register(u1);
// Constant Buffers
cbuffer saxpy_Args : register(b2) {
    float a;
};
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    int gid = dispatchThreadID.x;
    y[gid] = fma(a, x[gid], y[gid]);
}
