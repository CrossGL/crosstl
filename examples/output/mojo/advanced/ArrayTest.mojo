# Generated Mojo Shader Code
from math import *
from gpu import *

# CrossGL vector helpers
fn _crossgl_vec3_mul_f32_vs(v: SIMD[DType.float32, 4], s: Float32) -> SIMD[DType.float32, 4]:
    return SIMD[DType.float32, 4](v[0] * s, v[1] * s, v[2] * s, 0.0)


@value
struct VertexInput:
    var position: SIMD[DType.float32, 4]
    var texCoord: SIMD[DType.float32, 2]

@value
struct VertexOutput:
    var uv: SIMD[DType.float32, 2]
    var position: SIMD[DType.float32, 4]

@value
struct FragmentInput:
    var uv: SIMD[DType.float32, 2]

@value
struct FragmentOutput:
    var color: SIMD[DType.float32, 4]

# Constant Buffers
# CrossGL resource metadata: name=TestBuffer kind=cbuffer set=0 binding=0 binding_source=automatic
@value
struct TestBuffer:
    var values: InlineArray[Float32, 4]
    var colors: InlineArray[SIMD[DType.float32, 4], 2]

var values = InlineArray[Float32, 4](unsafe_uninitialized=True)
var colors = InlineArray[SIMD[DType.float32, 4], 2](unsafe_uninitialized=True)
# Vertex Shader
# CrossGL shader stage: vertex
fn vertex_main(input: VertexInput) -> VertexOutput:
    var output = VertexOutput(SIMD[DType.float32, 2](0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    output.uv = input.texCoord
    var scale: Float32 = (values[0] + values[1])
    var position: SIMD[DType.float32, 4] = _crossgl_vec3_mul_f32_vs(input.position, scale)
    output.position = SIMD[DType.float32, 4](position[0], position[1], position[2], 1.0)
    return output

# Fragment Shader
# CrossGL shader stage: fragment
fn fragment_main(input: FragmentInput) -> FragmentOutput:
    var output = FragmentOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    var color: SIMD[DType.float32, 4] = colors[0]
    if (input.uv[0] > 0.5):
        color = colors[1]
    output.color = SIMD[DType.float32, 4](color[0], color[1], color[2], 1.0)
    return output

