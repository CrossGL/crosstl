# Generated Mojo Shader Code
from math import *
from simd import *
from gpu import *

@value
struct Matrix32:
    var data: vecLiteralNode(value=32, literal_type=PrimitiveType(name=int, size_bits=None))LiteralNode(value=32, literal_type=PrimitiveType(name=int, size_bits=None))

@value
struct Matrix64:
    var data: vecLiteralNode(value=64, literal_type=PrimitiveType(name=int, size_bits=None))LiteralNode(value=64, literal_type=PrimitiveType(name=int, size_bits=None))

@value
struct Matrix128:
    var data: vecLiteralNode(value=128, literal_type=PrimitiveType(name=int, size_bits=None))LiteralNode(value=128, literal_type=PrimitiveType(name=int, size_bits=None))

@value
struct MatrixDimensions:
    var rows_A: Int32
    var cols_A: Int32
    var rows_B: Int32
    var cols_B: Int32
    var rows_C: Int32
    var cols_C: Int32

@value
struct KernelParams:
    var block_size: Int32
    var grid_size_x: Int32
    var grid_size_y: Int32
    var alpha: Float32
    var beta: Float32

fn getMatrixElement(matrix: buffer_PrimitiveType(name=float, size_bits=None), row: Int32, col: Int32, stride: Int32) -> Float32:
    return matrix[((row * stride) + col)]

fn setMatrixElement(matrix: buffer_PrimitiveType(name=float, size_bits=None), row: Int32, col: Int32, stride: Int32, value: Float32) -> None:
    matrix[((row * stride) + col)] = value

# Compute Shader
@compute_shader
fn matmul_warp_optimized() -> None:
    var warp_id: Int32 = int(gl_LocalInvocationID.x)
    var global_id: Int32 = int(gl_GlobalInvocationID.x)
    var elements_per_warp: Int32 = ((matrix_size * matrix_size) / (int(gl_NumWorkGroups.x) * 32))
    var elem: Int32 = 0
    while (elem < elements_per_warp):
        var linear_id: Int32 = (global_id + ((elem * int(gl_NumWorkGroups.x)) * 32))
        if (linear_id >= (matrix_size * matrix_size)):
        var row: Int32 = (linear_id / matrix_size)
        var col: Int32 = (linear_id % matrix_size)
        var result: Float32 = 0.0
        var k: Int32 = 0
        while (k < matrix_size):
            var a_val: Float32 = (A[(((row * matrix_size) + k) + warp_id)] if ((k + warp_id) < matrix_size) else 0.0)
            var b_val: Float32 = (B[(((k + warp_id) * matrix_size) + col)] if ((k + warp_id) < matrix_size) else 0.0)
            var partial_sum: Float32 = (a_val * b_val)
            var offset: Int32 = 16
            while (offset > 0):
                partial_sum += __shfl_down_sync(4294967295, partial_sum, offset)
                offset /= 2
            if (warp_id == 0):
                result += partial_sum
            k += WARP_SIZE
        if (warp_id == 0):
            C[linear_id] = ((alpha * result) + (beta * C[linear_id]))
        (++elem)

