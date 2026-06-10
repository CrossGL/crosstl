
struct Camera {
    float4x4 invViewMatrix: TEXCOORD0;
};
struct Input {
    float3 position: POSITION;
};
struct Output {
    half3 viewDir: TEXCOORD0;
};
// Vertex Shader
Output VSMain(Input in_, ConstantBuffer<Camera> camera) {
    return Output(half3(0));
}
