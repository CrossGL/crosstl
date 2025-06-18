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
    (MemberAccessNode(object=output, member=uv) = input.texCoord)
    (MemberAccessNode(object=output, member=position) = SIMD[DType.float32, 4](input.position, 1.0))
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output: FragmentOutput
    (r = input.uv.x)
    (g = input.uv.y)
    (b = 0.5)
    (MemberAccessNode(object=output, member=color) = SIMD[DType.float32, 4](r, g, b, 1.0))
    return output

