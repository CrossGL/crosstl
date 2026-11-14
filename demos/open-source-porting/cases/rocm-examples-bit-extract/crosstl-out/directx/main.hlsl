
RWStructuredBuffer<uint> d_output : register(u0);
StructuredBuffer<uint> d_input : register(t1);
// Constant Buffers
cbuffer bit_extract_kernel_Args : register(b2) {
    uint size;
};
cbuffer CrossGLDispatchInfo : register(b3) {
    uint3 crossglNumWorkGroups;
};
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint offset = dispatchThreadID.x;
    uint stride = (uint3(1, 1, 1).x * crossglNumWorkGroups.x);
    for (uint i = offset; (i < size); i += stride) {
        d_output[i] = ((d_input[i] >> 8) & 15u);
    }
}
