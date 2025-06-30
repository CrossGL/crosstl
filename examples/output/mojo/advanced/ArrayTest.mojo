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

var values: ArrayType(element_type=PrimitiveType(name=float, size_bits=None), size=LiteralNode(value=4, literal_type=PrimitiveType(name=int, size_bits=None)))
var colors: ArrayType(element_type=VectorType(element_type=PrimitiveType(name=float, size_bits=None), size=3), size=LiteralNode(value=2, literal_type=PrimitiveType(name=int, size_bits=None)))
# Vertex Shader
@vertex_shader
fn main(input: VertexInput) -> VertexOutput:
    var output: VertexOutput
    output.uv = input.texCoord
    var scale: Float32 = (values[0] + values[1])
    var position: SIMD[DType.float32, 3] = (input.position * scale)
    output.position = SIMD[DType.float32, 4](position, 1.0)
    return output

# Fragment Shader
@fragment_shader
fn main(input: FragmentInput) -> FragmentOutput:
    var output: FragmentOutput
    var color: SIMD[DType.float32, 3] = colors[0]
    if (input.uv.x > 0.5):
        color = colors[1]
    output.color = SIMD[DType.float32, 4](color, 1.0)
    return output

