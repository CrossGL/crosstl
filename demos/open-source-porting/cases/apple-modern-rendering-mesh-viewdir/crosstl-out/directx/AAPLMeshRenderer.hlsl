
struct Camera {
    float4x4 invViewMatrix;
};
struct Input {
    float3 position: POSITION;
};
struct Output {
    xhalf3 viewDir: TEXCOORD0;
};
// Vertex Shader
Output VSMain(Input in_, ReferenceType(referenced_type=NamedType(name=Camera, generic_args= camera[]), is_mutable=False)) {
    Output out_;
    out_.viewDir = xhalf3(normalize((camera.invViewMatrix[3].xyz - in_.position)));
    return out_;
}
