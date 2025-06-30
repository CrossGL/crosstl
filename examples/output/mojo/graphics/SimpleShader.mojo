# Generated Mojo Shader Code
from math import *
from simd import *
from gpu import *

@value
struct VertexInput:
    var position: SIMD[DType.float32, 3]
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
@vertex_shader
fn main(input: VertexInput) -> VertexOutput:
    var output: VertexOutput
    output.uv = input.texCoord
    output.position = SIMD[DType.float32, 4](input.position, 1.0)
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output: FragmentOutput
    var r: Float32 = input.uv.x
    var g: Float32 = input.uv.y
    var b: Float32 = 0.5
    output.color = SIMD[DType.float32, 4](r, g, b, 1.0)
    return output

