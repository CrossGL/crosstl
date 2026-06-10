
#include <metal_stdlib>
using namespace metal;

struct Camera {
  float4x4 invViewMatrix;
};
struct Input {
  float3 position [[attribute(0)]];
};
struct Output {
  xhalf3 viewDir;
};
// Vertex Shader
vertex Output main_vertex(Input in_ [[stage_in]],
                          constant Camera &camera [[buffer(0)]]) {
  Output out_;
  out_.viewDir = xhalf3(normalize(camera.invViewMatrix[3].xyz - in_.position));
  return out_;
}
