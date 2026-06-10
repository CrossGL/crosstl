
#version 450
layout(binding = 0) uniform usampler2D srcRGB;
layout(binding = 1) uniform usampler2D srcAlpha;
layout(rgba32ui, binding = 2) restrict writeonly uniform uimage2D dstTexture;
// Compute Shader
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    uvec2 rgbBlock = texelFetch(srcRGB, ivec2(gl_GlobalInvocationID.xy), 0).xy;
    uvec2 alphaBlock = texelFetch(srcAlpha, ivec2(gl_GlobalInvocationID.xy), 0).xy;
    imageStore(dstTexture, ivec2(gl_GlobalInvocationID.xy), uvec4(rgbBlock.xy, alphaBlock.xy));
}
