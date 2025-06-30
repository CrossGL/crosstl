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

# Constant Buffers
@value
struct TestBuffer:
    var values: StaticTuple[Float32, 4]
    var colors: StaticTuple[SIMD[DType.float32, 3], 2]

# Vertex Shader
@vertex_shader
fn main(input: VertexInput) -> VertexOutput:
    var output
    (MemberAccessNode(object=output, member=uv) = input.texCoord)
    (scale = (values[0] + values[1]))
    (position = (input.position * scale))
    (MemberAccessNode(object=output, member=position) = SIMD[DType.float32, 4](position, 1.0))
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output
    (color = colors[0])
    if (input.uv.x > 0.5):
        (color = colors[1])
    (MemberAccessNode(object=output, member=color) = SIMD[DType.float32, 4](color, 1.0))
    return output

