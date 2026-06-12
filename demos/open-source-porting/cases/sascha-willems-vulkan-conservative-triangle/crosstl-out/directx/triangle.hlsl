
struct UBO {
    float4x4 projection;
    float4x4 model;
};
struct gl_PerVertex {
    float4 gl_Position;
};
struct VertexInput {
    float3 inPos: TEXCOORD0;
    float3 inColor: TEXCOORD1;
};
struct VertexOutput {
    float3 outColor: TEXCOORD0;
    float4 gl_Position: SV_Position;
};
// Constant Buffers
cbuffer Uniforms : register(b0) {
    UBO ubo;
};
// Vertex Shader
VertexOutput VSMain(VertexInput input) {
    VertexOutput output;
    output.outColor = input.inColor;
    output.gl_Position = mul(mul(ubo.projection, ubo.model), float4(input.inPos, 1.0));
    return output;
}
