# Generated Mojo Shader Code
from math import *
from gpu import *

# CrossGL math helpers
fn dot_product(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> Float32:
    return a[0] * b[0] + a[1] * b[1]

fn dot_product(a: SIMD[DType.float32, 4], b: SIMD[DType.float32, 4]) -> Float32:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

fn _crossgl_fract_f32(x: Float32) -> Float32:
    return x - floor(x)



@value
struct VertexInput:
    var position: SIMD[DType.float32, 4]

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
    output.uv = (SIMD[DType.float32, 2](input.position[0], input.position[1]) * 10.0)
    output.position = SIMD[DType.float32, 4](input.position[0], input.position[1], input.position[2], 1.0)
    return output

# Fragment Shader
fn perlinNoise(p: SIMD[DType.float32, 2]) -> Float32:
    return _crossgl_fract_f32((sin(dot_product(p, SIMD[DType.float32, 2](12.9898, 78.233))) * 43758.5453))

# CrossGL shader stage: fragment
fn fragment_main(input: FragmentInput) -> FragmentOutput:
    var output = FragmentOutput(SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    var noise: Float32 = perlinNoise(input.uv)
    var height: Float32 = (noise * 10.0)
    var color: SIMD[DType.float32, 4] = SIMD[DType.float32, 4]((height / 10.0), (1.0 - (height / 10.0)), 0.0, 0.0)
    output.color = SIMD[DType.float32, 4](color[0], color[1], color[2], 1.0)
    return output

