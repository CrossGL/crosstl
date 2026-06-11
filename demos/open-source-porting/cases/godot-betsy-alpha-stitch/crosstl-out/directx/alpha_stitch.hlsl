
Texture2D<uint4> srcRGB : register(t0);
Texture2D<uint4> srcAlpha : register(t1);
RWTexture2D<uint4> dstTexture : register(u2);
// Compute Shader
[numthreads(8, 8, 1)]
void CSMain(uint3 gl_GlobalInvocationID : SV_DispatchThreadID) {
    uint2 rgbBlock = srcRGB.Load(int3(int2(gl_GlobalInvocationID.xy), 0)).xy;
    uint2 alphaBlock = srcAlpha.Load(int3(int2(gl_GlobalInvocationID.xy), 0)).xy;
    dstTexture[int2(gl_GlobalInvocationID.xy)] = uint4(rgbBlock.xy, alphaBlock.xy);
}
