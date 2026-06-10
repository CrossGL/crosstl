# Generated Mojo Shader Code
from math import *
from gpu import *

# CrossGL GPU builtin placeholders
@value
struct _CrossGLGpuBuiltinU32Vec3:
    var x: UInt32
    var y: UInt32
    var z: UInt32

var block_idx_uint = _CrossGLGpuBuiltinU32Vec3(0, 0, 0)
var global_idx_uint = _CrossGLGpuBuiltinU32Vec3(0, 0, 0)
var grid_dim_uint = _CrossGLGpuBuiltinU32Vec3(0, 0, 0)
var thread_idx_uint = _CrossGLGpuBuiltinU32Vec3(0, 0, 0)

# CrossGL synchronization placeholders
fn _crossgl_workgroup_barrier():
    pass


# CrossGL wave/subgroup placeholders
fn _crossgl_cuda_shfl_down_sync(mask: Int, value: Float32, delta: Int32) -> Float32:
    return value



@value
struct Matrix32:
    var data: InlineArray[InlineArray[Float32, 32], 32]

@value
struct Matrix64:
    var data: InlineArray[InlineArray[Float32, 64], 64]

@value
struct Matrix128:
    var data: InlineArray[InlineArray[Float32, 128], 128]

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

alias TILE_SIZE = 16
alias MAX_SHARED_MEMORY = 1024
alias WARP_SIZE = 32

fn getMatrixElement(matrix: UnsafePointer[Float32], row: Int32, col: Int32, stride: Int32) -> Float32:
    return matrix[int(((row * stride) + col))]

fn setMatrixElement(owned matrix: UnsafePointer[Float32], row: Int32, col: Int32, stride: Int32, value: Float32) -> None:
    matrix[int(((row * stride) + col))] = value

# Compute Shader
var dims = MatrixDimensions(0, 0, 0, 0, 0, 0)
var params = KernelParams(0, 0, 0, 0.0, 0.0)
var matrix_A: UnsafePointer[Float32] = UnsafePointer[Float32]()
var matrix_B: UnsafePointer[Float32] = UnsafePointer[Float32]()
var matrix_C: UnsafePointer[Float32] = UnsafePointer[Float32]()
# CrossGL shader stage: compute
fn matmul_tiled() -> None:
    var tile_A = InlineArray[InlineArray[Float32, TILE_SIZE], TILE_SIZE](unsafe_uninitialized=True)
    var tile_B = InlineArray[InlineArray[Float32, TILE_SIZE], TILE_SIZE](unsafe_uninitialized=True)
    var tx: Int32 = Int32(thread_idx_uint.x)
    var ty: Int32 = Int32(thread_idx_uint.y)
    var bx: Int32 = Int32(block_idx_uint.x)
    var by: Int32 = Int32(block_idx_uint.y)
    var row: Int32 = ((by * TILE_SIZE) + ty)
    var col: Int32 = ((bx * TILE_SIZE) + tx)
    var result: Float32 = 0.0
    var num_tiles: Int32 = (((dims.cols_A + TILE_SIZE) - 1) / TILE_SIZE)
    var tile: Int32 = 0
    while (tile < num_tiles):
        var a_row: Int32 = row
        var a_col: Int32 = ((tile * TILE_SIZE) + tx)
        if ((a_row < dims.rows_A) and (a_col < dims.cols_A)):
            tile_A[int(ty)][int(tx)] = getMatrixElement(matrix_A, a_row, a_col, dims.cols_A)
        else:
            tile_A[int(ty)][int(tx)] = 0.0
        var b_row: Int32 = ((tile * TILE_SIZE) + ty)
        var b_col: Int32 = col
        if ((b_row < dims.rows_B) and (b_col < dims.cols_B)):
            tile_B[int(ty)][int(tx)] = getMatrixElement(matrix_B, b_row, b_col, dims.cols_B)
        else:
            tile_B[int(ty)][int(tx)] = 0.0
        _crossgl_workgroup_barrier()
        var k: Int32 = 0
        while (k < TILE_SIZE):
            result += (tile_A[int(ty)][int(k)] * tile_B[int(k)][int(tx)])
            k += 1
        _crossgl_workgroup_barrier()
        tile += 1
    if ((row < dims.rows_C) and (col < dims.cols_C)):
        var existing_value: Float32 = getMatrixElement(matrix_C, row, col, dims.cols_C)
        var final_result: Float32 = ((params.alpha * result) + (params.beta * existing_value))
        setMatrixElement(matrix_C, row, col, dims.cols_C, final_result)

# Compute Shader
var matrix_size: Int32 = 0
var alpha: Float32 = 0.0
var beta: Float32 = 0.0
var A: UnsafePointer[Float32] = UnsafePointer[Float32]()
var B: UnsafePointer[Float32] = UnsafePointer[Float32]()
var C: UnsafePointer[Float32] = UnsafePointer[Float32]()
# CrossGL shader stage: compute
fn matmul_square() -> None:
    var shared_A = InlineArray[InlineArray[Float32, 16], 16](unsafe_uninitialized=True)
    var shared_B = InlineArray[InlineArray[Float32, 16], 16](unsafe_uninitialized=True)
    var tx: Int32 = Int32(thread_idx_uint.x)
    var ty: Int32 = Int32(thread_idx_uint.y)
    var bx: Int32 = Int32(block_idx_uint.x)
    var by: Int32 = Int32(block_idx_uint.y)
    var row: Int32 = ((by * 16) + ty)
    var col: Int32 = ((bx * 16) + tx)
    var sum: Float32 = 0.0
    var tile: Int32 = 0
    while (tile < ((matrix_size + 15) / 16)):
        var a_col: Int32 = ((tile * 16) + tx)
        var b_row: Int32 = ((tile * 16) + ty)
        shared_A[int(ty)][int(tx)] = (A[int(((row * matrix_size) + a_col))] if ((row < matrix_size) and (a_col < matrix_size)) else 0.0)
        shared_B[int(ty)][int(tx)] = (B[int(((b_row * matrix_size) + col))] if ((b_row < matrix_size) and (col < matrix_size)) else 0.0)
        _crossgl_workgroup_barrier()
        var k: Int32 = 0
        while (k < 16):
            sum += (shared_A[int(ty)][int(k)] * shared_B[int(k)][int(tx)])
            k += 1
        _crossgl_workgroup_barrier()
        tile += 1
    if ((row < matrix_size) and (col < matrix_size)):
        C[int(((row * matrix_size) + col))] = ((alpha * sum) + (beta * C[int(((row * matrix_size) + col))]))

# Compute Shader
var matrix_A_vec: UnsafePointer[SIMD[DType.float32, 4]] = UnsafePointer[SIMD[DType.float32, 4]]()
var matrix_B_vec: UnsafePointer[SIMD[DType.float32, 4]] = UnsafePointer[SIMD[DType.float32, 4]]()
var matrix_C_vec: UnsafePointer[SIMD[DType.float32, 4]] = UnsafePointer[SIMD[DType.float32, 4]]()
# CrossGL shader stage: compute
fn matmul_vectorized() -> None:
    var shared_A_vec = InlineArray[InlineArray[SIMD[DType.float32, 4], 8], 8](unsafe_uninitialized=True)
    var shared_B_vec = InlineArray[InlineArray[SIMD[DType.float32, 4], 8], 8](unsafe_uninitialized=True)
    var tx: Int32 = Int32(thread_idx_uint.x)
    var ty: Int32 = Int32(thread_idx_uint.y)
    var bx: Int32 = Int32(block_idx_uint.x)
    var by: Int32 = Int32(block_idx_uint.y)
    var row: Int32 = ((by * 8) + ty)
    var col: Int32 = ((bx * 8) + tx)
    var result: SIMD[DType.float32, 4] = SIMD[DType.float32, 4](0.0)
    var tile: Int32 = 0
    while (tile < ((dims.cols_A + 31) / 32)):
        var a_col_vec: Int32 = ((tile * 8) + tx)
        var b_row_vec: Int32 = ((tile * 8) + ty)
        if ((row < (dims.rows_A / 4)) and (a_col_vec < (dims.cols_A / 4))):
            shared_A_vec[int(ty)][int(tx)] = matrix_A_vec[int(((row * (dims.cols_A / 4)) + a_col_vec))]
        else:
            shared_A_vec[int(ty)][int(tx)] = SIMD[DType.float32, 4](0.0)
        if ((b_row_vec < (dims.rows_B / 4)) and (col < (dims.cols_B / 4))):
            shared_B_vec[int(ty)][int(tx)] = matrix_B_vec[int(((b_row_vec * (dims.cols_B / 4)) + col))]
        else:
            shared_B_vec[int(ty)][int(tx)] = SIMD[DType.float32, 4](0.0)
        _crossgl_workgroup_barrier()
        var k: Int32 = 0
        while (k < 8):
            result += (shared_A_vec[int(ty)][int(k)] * shared_B_vec[int(k)][int(tx)])
            k += 1
        _crossgl_workgroup_barrier()
        tile += 1
    if ((row < (dims.rows_C / 4)) and (col < (dims.cols_C / 4))):
        var existing: SIMD[DType.float32, 4] = matrix_C_vec[int(((row * (dims.cols_C / 4)) + col))]
        matrix_C_vec[int(((row * (dims.cols_C / 4)) + col))] = ((params.alpha * result) + (params.beta * existing))

# Compute Shader
var batch_size: Int32 = 0
var batch_A: UnsafePointer[Float32] = UnsafePointer[Float32]()
var batch_B: UnsafePointer[Float32] = UnsafePointer[Float32]()
var batch_C: UnsafePointer[Float32] = UnsafePointer[Float32]()
# CrossGL shader stage: compute
fn matmul_batched() -> None:
    var tile_A = InlineArray[InlineArray[Float32, 16], 16](unsafe_uninitialized=True)
    var tile_B = InlineArray[InlineArray[Float32, 16], 16](unsafe_uninitialized=True)
    var tx: Int32 = Int32(thread_idx_uint.x)
    var ty: Int32 = Int32(thread_idx_uint.y)
    var bx: Int32 = Int32(block_idx_uint.x)
    var by: Int32 = Int32(block_idx_uint.y)
    var bz: Int32 = Int32(block_idx_uint.z)
    var batch_id: Int32 = bz
    if (batch_id >= batch_size):
        return None
    var matrix_offset: Int32 = ((batch_id * matrix_size) * matrix_size)
    var row: Int32 = ((by * 16) + ty)
    var col: Int32 = ((bx * 16) + tx)
    var result: Float32 = 0.0
    var tile: Int32 = 0
    while (tile < ((matrix_size + 15) / 16)):
        var a_col: Int32 = ((tile * 16) + tx)
        var b_row: Int32 = ((tile * 16) + ty)
        if ((row < matrix_size) and (a_col < matrix_size)):
            tile_A[int(ty)][int(tx)] = batch_A[int(((matrix_offset + (row * matrix_size)) + a_col))]
        else:
            tile_A[int(ty)][int(tx)] = 0.0
        if ((b_row < matrix_size) and (col < matrix_size)):
            tile_B[int(ty)][int(tx)] = batch_B[int(((matrix_offset + (b_row * matrix_size)) + col))]
        else:
            tile_B[int(ty)][int(tx)] = 0.0
        _crossgl_workgroup_barrier()
        var k: Int32 = 0
        while (k < 16):
            result += (tile_A[int(ty)][int(k)] * tile_B[int(k)][int(tx)])
            k += 1
        _crossgl_workgroup_barrier()
        tile += 1
    if ((row < matrix_size) and (col < matrix_size)):
        var c_index: Int32 = ((matrix_offset + (row * matrix_size)) + col)
        batch_C[int(c_index)] = ((alpha * result) + (beta * batch_C[int(c_index)]))

# Compute Shader
# CrossGL shader stage: compute
fn matmul_warp_optimized() -> None:
    var warp_id: Int32 = Int32(thread_idx_uint.x)
    var global_id: Int32 = Int32(global_idx_uint.x)
    var elements_per_warp: Int32 = ((matrix_size * matrix_size) / (Int32(grid_dim_uint.x) * 32))
    var elem: Int32 = 0
    while (elem < elements_per_warp):
        var linear_id: Int32 = (global_id + ((elem * Int32(grid_dim_uint.x)) * 32))
        if (linear_id >= (matrix_size * matrix_size)):
            break
        var row: Int32 = (linear_id / matrix_size)
        var col: Int32 = (linear_id % matrix_size)
        var result: Float32 = 0.0
        var k: Int32 = 0
        while (k < matrix_size):
            var a_val: Float32 = (A[int((((row * matrix_size) + k) + warp_id))] if ((k + warp_id) < matrix_size) else 0.0)
            var b_val: Float32 = (B[int((((k + warp_id) * matrix_size) + col))] if ((k + warp_id) < matrix_size) else 0.0)
            var partial_sum: Float32 = (a_val * b_val)
            var offset: Int32 = 16
            while (offset > 0):
                partial_sum += _crossgl_cuda_shfl_down_sync(4294967295, partial_sum, offset)
                offset /= 2
            if (warp_id == 0):
                result += partial_sum
            k += WARP_SIZE
        if (warp_id == 0):
            C[int(linear_id)] = ((alpha * result) + (beta * C[int(linear_id)]))
        elem += 1

