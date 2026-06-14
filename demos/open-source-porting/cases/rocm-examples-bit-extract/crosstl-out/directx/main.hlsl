
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
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint offset = ((groupID.x * uint3(1, 1, 1).x) + groupThreadID.x);
    uint stride = (uint3(1, 1, 1).x * crossglNumWorkGroups.x);
    for (uint i = offset; (i < size); i += stride) {
        d_output[i] = ((d_input[i] >> 8) & 15u);
    }
}
