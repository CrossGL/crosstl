# Generated Mojo Shader Code
from math import *
from simd import *
from gpu import *

@value
struct VertexInput:
    var position: SIMD[DType.float32, 3]

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
@vertex_shader
fn main(input: VertexInput) -> VertexOutput:
    var output: VertexOutput
    output.uv = (input.position.xy * 10.0)
    output.position = SIMD[DType.float32, 4](input.position, 1.0)
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output: FragmentOutput
    var noise: Float32 = perlinNoise(input.uv)
    var height: Float32 = (noise * 10.0)
    var color: SIMD[DType.float32, 3] = SIMD[DType.float32, 3]((height / 10.0), (1.0 - (height / 10.0)), 0.0)
    output.color = SIMD[DType.float32, 4](color, 1.0)
    return output

fn perlinNoise(p: SIMD[DType.float32, 2]) -> Float32:
    return fract((sin(dot_product(p, SIMD[DType.float32, 2](12.9898, 78.233))) * 43758.5453))

