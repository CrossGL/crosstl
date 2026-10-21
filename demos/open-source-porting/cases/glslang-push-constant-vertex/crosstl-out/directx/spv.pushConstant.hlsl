
struct VertexOutput {
    float4 color: TEXCOORD0;
};
// Constant Buffers
cbuffer Material : register(b0)
{
    uint4 __crossgl_Material_registers[1];
};
static int kind;
static float fa[3];
uint __crossgl_Material_word(int word)
{
    return __crossgl_Material_registers[word >> 2][word & 3];
}
void __crossgl_load_Material()
{
    kind = asint(__crossgl_Material_registers[0].x);
    for (int __crossgl_Material_i0 = 0; __crossgl_Material_i0 < 3; ++__crossgl_Material_i0)
    {
        fa[__crossgl_Material_i0] = asfloat(__crossgl_Material_word((__crossgl_Material_i0) + 1));
    }
}
// Vertex Shader
VertexOutput VSMain() {
    __crossgl_load_Material();
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
