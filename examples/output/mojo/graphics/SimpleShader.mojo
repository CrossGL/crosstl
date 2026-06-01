# Generated Mojo Shader Code
from math import *
from gpu import *

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

# Vertex Shader
# CrossGL shader stage: vertex
fn vertex_main(input: VertexInput) -> VertexOutput:
    var output = VertexOutput(SIMD[DType.float32, 2](0.0, 0.0), SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    output.uv = input.texCoord
    output.position = SIMD[DType.float32, 4](input.position[0], input.position[1], input.position[2], 1.0)
    return output

# Fragment Shader
# CrossGL shader stage: fragment
fn fragment_main(input: FragmentInput) -> FragmentOutput:
    var output = FragmentOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    var r: Float32 = input.uv[0]
    var g: Float32 = input.uv[1]
    var b: Float32 = 0.5
    output.color = SIMD[DType.float32, 4](r, g, b, 1.0)
    return output

