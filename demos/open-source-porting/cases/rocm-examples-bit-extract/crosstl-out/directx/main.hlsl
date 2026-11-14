
RWStructuredBuffer<uint> d_output : register(u0);
StructuredBuffer<uint> d_input : register(t1);
// Constant Buffers
cbuffer bit_extract_kernel_Args : register(b2) {
    uint64_t size;
};
cbuffer CrossGLDispatchInfo : register(b3) {
    uint3 crossglNumWorkGroups;
};
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint64_t offset = dispatchThreadID.x;
    uint64_t stride = (uint3(1, 1, 1).x * crossglNumWorkGroups.x);
    for (uint64_t i = offset; (i < size); i += stride) {
        d_output[uint(i)] = ((d_input[uint(i)] >> 8) & 15u);
    }
}
