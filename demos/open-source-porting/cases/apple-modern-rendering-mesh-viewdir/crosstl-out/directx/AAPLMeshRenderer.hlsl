
struct Camera {
    float4x4 invViewMatrix;
};
struct Input {
    float3 position: POSITION;
};
struct Output {
    half3 viewDir: TEXCOORD0;
};
// Vertex Shader
Output VSMain(Input in_, ReferenceType(referenced_type=NamedType(name=Camera, generic_args= camera[]), is_mutable=False)) {
    return Output(half3(0));
}
