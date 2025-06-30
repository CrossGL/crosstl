// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix32 {
  pub data : vecLiteralNode(value = 32,
                            literal_type = PrimitiveType(name = int,
                                                         size_bits = None))
                 LiteralNode(value = 32,
                             literal_type = PrimitiveType(name = int,
                                                          size_bits = None)),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix64 {
  pub data : vecLiteralNode(value = 64,
                            literal_type = PrimitiveType(name = int,
                                                         size_bits = None))
                 LiteralNode(value = 64,
                             literal_type = PrimitiveType(name = int,
                                                          size_bits = None)),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix128 {
  pub data : vecLiteralNode(value = 128,
                            literal_type = PrimitiveType(name = int,
                                                         size_bits = None))
                 LiteralNode(value = 128,
                             literal_type = PrimitiveType(name = int,
                                                          size_bits = None)),
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MatrixDimensions {
  pub rows_A : i32,
               pub cols_A : i32,
                            pub rows_B : i32,
                                         pub cols_B : i32,
                                                      pub rows_C : i32,
                                                                   pub cols_C
      : i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct KernelParams {
  pub block_size : i32,
                   pub grid_size_x : i32,
                                     pub grid_size_y : i32,
                                                       pub alpha : f32,
                                                                   pub beta
      : f32,
}

// Constant Buffers
pub fn
getMatrixElement(matrix : buffer_PrimitiveType(name = float, size_bits = None),
                 row : i32, col : i32, stride : i32) -> f32 {
  return matrix[((row * stride) + col)];
}

pub fn setMatrixElement(matrix : buffer_PrimitiveType(name = float,
                                                      size_bits = None),
                        row : i32, col : i32, stride : i32, value : f32) -> () {
  matrix[((row * stride) + col)] = value;
}

// Compute Shader
#[compute_shader]
pub fn matmul_warp_optimized() -> () {
  let mut warp_id : i32 = int(gl_LocalInvocationID.x);
  let mut global_id : i32 = int(gl_GlobalInvocationID.x);
  let mut elements_per_warp
      : i32 = ((matrix_size * matrix_size) / (int(gl_NumWorkGroups.x) * 32));
  let mut elem : i32 = 0;
  ;
  while (elem < elements_per_warp) {
    let mut linear_id
        : i32 = (global_id + ((elem * int(gl_NumWorkGroups.x)) * 32));
    if (linear_id >= (matrix_size * matrix_size)) {
    }
    let mut row : i32 = (linear_id / matrix_size);
    let mut col : i32 = (linear_id % matrix_size);
    let mut result : f32 = 0.0;
    let mut k : i32 = 0;
    ;
    while (k < matrix_size) {
      let mut a_val : f32 = (if ((k + warp_id) < matrix_size) {
        A[(((row * matrix_size) + k) + warp_id)]
      } else {0.0});
      let mut b_val : f32 = (if ((k + warp_id) < matrix_size) {
        B[(((k + warp_id) * matrix_size) + col)]
      } else {0.0});
      let mut partial_sum : f32 = (a_val * b_val);
      let mut offset : i32 = 16;
      ;
      while (offset > 0) {
        partial_sum += __shfl_down_sync(4294967295, partial_sum, offset);
        offset /= 2;
      }
      if (warp_id == 0) {
        result += partial_sum;
      }
      k += WARP_SIZE;
    }
    if (warp_id == 0) {
      C[linear_id] = ((alpha * result) + (beta * C[linear_id]));
    }
    (++elem);
  }
}
