
struct VertexOutput {
    float4 color: TEXCOORD0;
    float4 gl_Position: SV_POSITION;
};
// Constant Buffers
cbuffer Material : register(b0) {
    int kind;
    float fa[3];
};
// Vertex Shader
VertexOutput VSMain() {
    VertexOutput output;
    switch (kind) {
        case 1: {
            output.color = float4(0.2, 0.2, 0.2, 0.2);
            break;
        }
        case 2: {
            output.color = float4(0.5, 0.5, 0.5, 0.5);
            break;
        }
        default: {
            output.color = float4(0.0, 0.0, 0.0, 0.0);
            break;
        }
    }
    return output;
}
