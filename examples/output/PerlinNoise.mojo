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
    var output
    (MemberAccessNode(object=output, member=uv) = (input.position.xy * 10.0))
    (MemberAccessNode(object=output, member=position) = SIMD[DType.float32, 4](input.position, 1.0))
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output
    (noise = perlinNoise(input.uv))
    (color = SIMD[DType.float32, 3]((height / 10.0), (1.0 - (height / 10.0)), 0.0))
    return output

