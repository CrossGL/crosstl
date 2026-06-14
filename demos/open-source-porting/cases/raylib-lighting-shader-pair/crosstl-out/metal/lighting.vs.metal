
#include <metal_stdlib>
using namespace metal;

struct VertexInput {
    float3 vertexPosition [[attribute(0)]];
    float2 vertexTexCoord [[attribute(1)]];
    float3 vertexNormal [[attribute(2)]];
    float4 vertexColor [[attribute(3)]];
};
struct VertexOutput {
    float3 fragPosition;
    float2 fragTexCoord;
    float4 fragColor;
    float3 fragNormal;
    float4 gl_Position [[position]];
};
// Constant Buffers
struct Uniforms {
    float4x4 mvp;
    float4x4 matModel;
    float4x4 matNormal;
};
// Vertex Shader
vertex VertexOutput vertex_main(VertexInput input [[stage_in]], constant Uniforms& uniforms [[buffer(0)]]) {
    VertexOutput output;
    output.fragPosition = float3(uniforms.matModel * float4(input.vertexPosition, 1.0));
    output.fragTexCoord = input.vertexTexCoord;
    output.fragColor = input.vertexColor;
    output.fragNormal = normalize(float3(uniforms.matNormal * float4(input.vertexNormal, 1.0)));
    output.gl_Position = uniforms.mvp * float4(input.vertexPosition, 1.0);
    return output;
}
