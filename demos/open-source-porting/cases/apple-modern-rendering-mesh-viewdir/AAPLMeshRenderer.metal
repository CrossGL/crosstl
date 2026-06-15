struct Camera { float4x4 invViewMatrix; };
struct Input { float3 position; };
struct Output { xhalf3 viewDir; };

vertex Output main_vertex(Input in [[stage_in]],
                          constant Camera& camera [[buffer(0)]]) {
    Output out;
    out.viewDir = (xhalf3)normalize(camera.invViewMatrix[3].xyz - in.position);
    return out;
}
