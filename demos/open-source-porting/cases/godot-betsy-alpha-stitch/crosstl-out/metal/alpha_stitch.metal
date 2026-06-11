
#include <metal_stdlib>
using namespace metal;

// Compute Shader
kernel void kernel_main(uint3 thread_position_in_grid [[thread_position_in_grid]], texture2d<uint> srcRGB [[texture(0)]], texture2d<uint> srcAlpha [[texture(1)]], texture2d<uint, access::write> dstTexture [[texture(2)]]) {
    uint2 rgbBlock = srcRGB.read(uint2(int2(thread_position_in_grid.xy)), uint(0)).xy;
    uint2 alphaBlock = srcAlpha.read(uint2(int2(thread_position_in_grid.xy)), uint(0)).xy;
    dstTexture.write(uint4(rgbBlock.xy, alphaBlock.xy), uint2(int2(thread_position_in_grid.xy)));
}
