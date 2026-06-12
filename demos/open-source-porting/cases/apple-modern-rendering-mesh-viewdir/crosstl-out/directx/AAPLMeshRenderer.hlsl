
struct Camera {
    float4x4 invViewMatrix: TEXCOORD0;
};
struct Input {
    float3 position: POSITION;
};
struct Output {
    half3 viewDir: TEXCOORD0;
};
ConstantBuffer<Camera> camera : register(b0);
// Vertex Shader
Output VSMain(Input in_) {
    Output out_;
    out_.viewDir = half3(normalize((camera.invViewMatrix[3].xyz - in_.position)));
    return out_;
}
