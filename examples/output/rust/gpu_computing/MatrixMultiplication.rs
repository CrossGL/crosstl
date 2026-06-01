// Generated Rust GPU Shader Code
use gpu::*;
use math::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Matrix32 {
  pub data : [[f32; 32]; 32],
}

impl Matrix32 {
  pub fn new (data : [[f32; 32]; 32])->Self {
    Self { data }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix64 {
  pub data : [[f32; 64]; 64],
}

impl Matrix64 {
  pub fn new (data : [[f32; 64]; 64])->Self {
    Self { data }
  }
}

impl Default for Matrix64 {
  fn default()->Self {
    Self {
    data:
      unsafe { std::mem::zeroed() }
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Matrix128 {
  pub data : [[f32; 128]; 128],
}

impl Matrix128 {
  pub fn new (data : [[f32; 128]; 128])->Self {
    Self { data }
  }
}

impl Default for Matrix128 {
  fn default()->Self {
    Self {
    data:
      unsafe { std::mem::zeroed() }
    }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixDimensions {
  pub rows_A : i32, pub cols_A : i32, pub rows_B : i32, pub cols_B : i32,
      pub rows_C : i32, pub cols_C : i32,
}

impl MatrixDimensions {
  pub fn new (rows_A : i32, cols_A : i32, rows_B : i32, cols_B : i32,
              rows_C : i32, cols_C : i32)
      ->Self {
    Self { rows_A, cols_A, rows_B, cols_B, rows_C, cols_C }
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct KernelParams {
  pub block_size : i32, pub grid_size_x : i32, pub grid_size_y : i32,
      pub alpha : f32, pub beta : f32,
}

impl KernelParams {
  pub fn new (block_size : i32, grid_size_x : i32, grid_size_y : i32,
              alpha_value : f32, beta_value : f32)
      ->Self {
    Self {
      block_size, grid_size_x, grid_size_y, alpha : alpha_value,
                                                    beta : beta_value
    }
  }
}

const TILE_SIZE : i32 = 16;
const MAX_SHARED_MEMORY : i32 = 1024;
const WARP_SIZE : i32 = 32;

// Constant Buffers
pub fn getMatrixElement(matrix : *mut f32, row : i32, col : i32, stride : i32)
    -> f32 {
  return unsafe{*(matrix).add(((row * stride) + col) as usize)};
}

pub fn setMatrixElement(mut matrix : *mut f32, row : i32, col : i32,
                        stride : i32, value : f32) -> () {
  unsafe{*(matrix).add(((row * stride) + col) as usize) = value};
}

static DIMS : std::sync::LazyLock<MatrixDimensions> =
                  std::sync::LazyLock::new (|| Default::default());
static PARAMS : std::sync::LazyLock<KernelParams> =
                    std::sync::LazyLock::new (|| Default::default());
static MATRIX_A : usize = 0;
static MATRIX_B : usize = 0;
static MATRIX_C : usize = 0;
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn matmul_tiled() -> () {
  let mut tile_A : Vec<Vec<f32>> = Vec::new ();
  let mut tile_B : Vec<Vec<f32>> = Vec::new ();
  let tx : i32 = (local_invocation_id().x as i32);
  let ty : i32 = (local_invocation_id().y as i32);
  let bx : i32 = (workgroup_id().x as i32);
  let by : i32 = (workgroup_id().y as i32);
  let row : i32 = ((by * TILE_SIZE) + ty);
  let col : i32 = ((bx * TILE_SIZE) + tx);
  let mut result : f32 = 0.0;
  let num_tiles : i32 = ((((*DIMS).cols_A + TILE_SIZE) - 1) / TILE_SIZE);
  let mut tile : i32 = 0;
  while (tile < num_tiles) {
    let a_row : i32 = row;
    let a_col : i32 = ((tile * TILE_SIZE) + tx);
    if ((a_row < (*DIMS).rows_A) && (a_col < (*DIMS).cols_A)) {
      tile_A[ty as usize][tx as usize] = getMatrixElement(
          (MATRIX_A as * mut f32), a_row, a_col, (*DIMS).cols_A);
    } else {
      tile_A[ty as usize][tx as usize] = 0.0;
    }
    let b_row : i32 = ((tile * TILE_SIZE) + ty);
    let b_col : i32 = col;
    if ((b_row < (*DIMS).rows_B) && (b_col < (*DIMS).cols_B)) {
      tile_B[ty as usize][tx as usize] = getMatrixElement(
          (MATRIX_B as * mut f32), b_row, b_col, (*DIMS).cols_B);
    } else {
      tile_B[ty as usize][tx as usize] = 0.0;
    }
    workgroup_barrier();
    let mut k : i32 = 0;
    while (k < TILE_SIZE) {
      result +=
          (tile_A[ty as usize][k as usize] * tile_B[k as usize][tx as usize]);
      k += 1;
    }
    workgroup_barrier();
    tile += 1;
  }
  if ((row < (*DIMS).rows_C) && (col < (*DIMS).cols_C)) {
    let existing_value : f32 = getMatrixElement((MATRIX_C as * mut f32), row,
                                                col, (*DIMS).cols_C);
    let final_result
        : f32 =
              (((*PARAMS).alpha * result) + ((*PARAMS).beta * existing_value));
    setMatrixElement((MATRIX_C as * mut f32), row, col, (*DIMS).cols_C,
                     final_result);
  }
}

static MATRIX_SIZE : i32 = 0;
static ALPHA : f32 = 0.0;
static BETA : f32 = 0.0;
static A : usize = 0;
static B : usize = 0;
static C : usize = 0;
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn matmul_square() -> () {
  let mut shared_A : [[f32; 16]; 16] =
                         std::array::from_fn(| _ | Default::default());
  let mut shared_B : [[f32; 16]; 16] =
                         std::array::from_fn(| _ | Default::default());
  let tx : i32 = (local_invocation_id().x as i32);
  let ty : i32 = (local_invocation_id().y as i32);
  let bx : i32 = (workgroup_id().x as i32);
  let by : i32 = (workgroup_id().y as i32);
  let row : i32 = ((by * 16) + ty);
  let col : i32 = ((bx * 16) + tx);
  let mut sum : f32 = 0.0;
  let mut tile : i32 = 0;
  while (tile < ((MATRIX_SIZE + 15) / 16)) {
    let a_col : i32 = ((tile * 16) + tx);
    let b_row : i32 = ((tile * 16) + ty);
    shared_A[ty as usize][tx as usize] =
        (if ((row < MATRIX_SIZE) && (a_col < MATRIX_SIZE)) {
          unsafe {
            *((A as * mut f32)).add(((row * MATRIX_SIZE) + a_col) as usize)
          }
        } else {0.0});
    shared_B[ty as usize][tx as usize] =
        (if ((b_row < MATRIX_SIZE) && (col < MATRIX_SIZE)) {
          unsafe {
            *((B as * mut f32)).add(((b_row * MATRIX_SIZE) + col) as usize)
          }
        } else {0.0});
    workgroup_barrier();
    let mut k : i32 = 0;
    while (k < 16) {
      sum += (shared_A[ty as usize][k as usize] *
              shared_B[k as usize][tx as usize]);
      k += 1;
    }
    workgroup_barrier();
    tile += 1;
  }
  if ((row < MATRIX_SIZE) && (col < MATRIX_SIZE)) {
    unsafe{
        *((C as * mut f32)).add(((row * MATRIX_SIZE) + col) as usize) =
            ((ALPHA * sum) +
             (BETA * unsafe{*((C as * mut f32))
                                 .add(((row * MATRIX_SIZE) + col) as usize)}))};
  }
}

static MATRIX_A_VEC : usize = 0;
static MATRIX_B_VEC : usize = 0;
static MATRIX_C_VEC : usize = 0;
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn matmul_vectorized() -> () {
  let mut shared_A_vec : [[Vec4<f32>; 8]; 8] =
                             std::array::from_fn(| _ | Default::default());
  let mut shared_B_vec : [[Vec4<f32>; 8]; 8] =
                             std::array::from_fn(| _ | Default::default());
  let tx : i32 = (local_invocation_id().x as i32);
  let ty : i32 = (local_invocation_id().y as i32);
  let bx : i32 = (workgroup_id().x as i32);
  let by : i32 = (workgroup_id().y as i32);
  let row : i32 = ((by * 8) + ty);
  let col : i32 = ((bx * 8) + tx);
  let mut result : Vec4<f32> = Vec4::<f32>::new (0.0, 0.0, 0.0, 0.0);
  let mut tile : i32 = 0;
  while (tile < (((*DIMS).cols_A + 31) / 32)) {
    let a_col_vec : i32 = ((tile * 8) + tx);
    let b_row_vec : i32 = ((tile * 8) + ty);
    if ((row < ((*DIMS).rows_A / 4)) && (a_col_vec < ((*DIMS).cols_A / 4))) {
      shared_A_vec[ty as usize][tx as usize] = unsafe{
          *((MATRIX_A_VEC as * mut Vec4<f32>))
               .add(((row * ((*DIMS).cols_A / 4)) + a_col_vec) as usize)};
    } else {
      shared_A_vec[ty as usize][tx as usize] =
          Vec4::<f32>::new (0.0, 0.0, 0.0, 0.0);
    }
    if ((b_row_vec < ((*DIMS).rows_B / 4)) && (col < ((*DIMS).cols_B / 4))) {
      shared_B_vec[ty as usize][tx as usize] = unsafe{
          *((MATRIX_B_VEC as * mut Vec4<f32>))
               .add(((b_row_vec * ((*DIMS).cols_B / 4)) + col) as usize)};
    } else {
      shared_B_vec[ty as usize][tx as usize] =
          Vec4::<f32>::new (0.0, 0.0, 0.0, 0.0);
    }
    workgroup_barrier();
    let mut k : i32 = 0;
    while (k < 8) {
      result = (result +
                Vec4::<f32>::new ((shared_A_vec[ty as usize][k as usize].x *
                                   shared_B_vec[k as usize][tx as usize].x),
                                  (shared_A_vec[ty as usize][k as usize].y *
                                   shared_B_vec[k as usize][tx as usize].y),
                                  (shared_A_vec[ty as usize][k as usize].z *
                                   shared_B_vec[k as usize][tx as usize].z),
                                  (shared_A_vec[ty as usize][k as usize].w *
                                   shared_B_vec[k as usize][tx as usize].w)));
      k += 1;
    }
    workgroup_barrier();
    tile += 1;
  }
  if ((row < ((*DIMS).rows_C / 4)) && (col < ((*DIMS).cols_C / 4))) {
    let existing
        : Vec4<f32> =
              unsafe{*((MATRIX_C_VEC as * mut Vec4<f32>))
                          .add(((row * ((*DIMS).cols_C / 4)) + col) as usize)};
    unsafe{*((MATRIX_C_VEC as * mut Vec4<f32>))
                .add(((row * ((*DIMS).cols_C / 4)) + col) as usize) =
               (Vec4::<f32>::new (((*PARAMS).alpha * result.x),
                                  ((*PARAMS).alpha * result.y),
                                  ((*PARAMS).alpha * result.z),
                                  ((*PARAMS).alpha * result.w)) +
                Vec4::<f32>::new (((*PARAMS).beta * existing.x),
                                  ((*PARAMS).beta * existing.y),
                                  ((*PARAMS).beta * existing.z),
                                  ((*PARAMS).beta * existing.w)))};
  }
}

static BATCH_SIZE : i32 = 0;
static BATCH_A : usize = 0;
static BATCH_B : usize = 0;
static BATCH_C : usize = 0;
// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn matmul_batched() -> () {
  let mut tile_A : [[f32; 16]; 16] =
                       std::array::from_fn(| _ | Default::default());
  let mut tile_B : [[f32; 16]; 16] =
                       std::array::from_fn(| _ | Default::default());
  let tx : i32 = (local_invocation_id().x as i32);
  let ty : i32 = (local_invocation_id().y as i32);
  let bx : i32 = (workgroup_id().x as i32);
  let by : i32 = (workgroup_id().y as i32);
  let bz : i32 = (workgroup_id().z as i32);
  let batch_id : i32 = bz;
  if (batch_id >= BATCH_SIZE) {
  }
  let matrix_offset : i32 = ((batch_id * MATRIX_SIZE) * MATRIX_SIZE);
  let row : i32 = ((by * 16) + ty);
  let col : i32 = ((bx * 16) + tx);
  let mut result : f32 = 0.0;
  let mut tile : i32 = 0;
  while (tile < ((MATRIX_SIZE + 15) / 16)) {
    let a_col : i32 = ((tile * 16) + tx);
    let b_row : i32 = ((tile * 16) + ty);
    if ((row < MATRIX_SIZE) && (a_col < MATRIX_SIZE)) {
      tile_A[ty as usize][tx as usize] = unsafe{
          *((BATCH_A as * mut f32))
               .add(((matrix_offset + (row * MATRIX_SIZE)) + a_col) as usize)};
    } else {
      tile_A[ty as usize][tx as usize] = 0.0;
    }
    if ((b_row < MATRIX_SIZE) && (col < MATRIX_SIZE)) {
      tile_B[ty as usize][tx as usize] = unsafe{
          *((BATCH_B as * mut f32))
               .add(((matrix_offset + (b_row * MATRIX_SIZE)) + col) as usize)};
    } else {
      tile_B[ty as usize][tx as usize] = 0.0;
    }
    workgroup_barrier();
    let mut k : i32 = 0;
    while (k < 16) {
      result +=
          (tile_A[ty as usize][k as usize] * tile_B[k as usize][tx as usize]);
      k += 1;
    }
    workgroup_barrier();
    tile += 1;
  }
  if ((row < MATRIX_SIZE) && (col < MATRIX_SIZE)) {
    let c_index : i32 = ((matrix_offset + (row * MATRIX_SIZE)) + col);
    unsafe{
        *((BATCH_C as * mut f32)).add(c_index as usize) =
            ((ALPHA * result) +
             (BETA * unsafe{*((BATCH_C as * mut f32)).add(c_index as usize)}))};
  }
}

// Compute Shader
#[cfg_attr(feature = "crossgl_gpu", compute_shader)]
pub fn matmul_warp_optimized() -> () {
  let warp_id : i32 = (local_invocation_id().x as i32);
  let global_id : i32 = (global_invocation_id().x as i32);
  let elements_per_warp
      : i32 =
            ((MATRIX_SIZE * MATRIX_SIZE) / ((num_workgroups().x as i32) * 32));
  let mut elem : i32 = 0;
  while (elem < elements_per_warp) {
    let linear_id
        : i32 = (global_id + ((elem * (num_workgroups().x as i32)) * 32));
    if (linear_id >= (MATRIX_SIZE * MATRIX_SIZE)) {
    }
    let row : i32 = (linear_id / MATRIX_SIZE);
    let col : i32 = (linear_id % MATRIX_SIZE);
    let mut result : f32 = 0.0;
    let mut k : i32 = 0;
    while (k < MATRIX_SIZE) {
      let a_val : f32 = (if ((k + warp_id) < MATRIX_SIZE) {
        unsafe {
          *((A as * mut f32))
               .add((((row * MATRIX_SIZE) + k) + warp_id) as usize)
        }
      } else {0.0});
      let b_val : f32 = (if ((k + warp_id) < MATRIX_SIZE) {
        unsafe {
          *((B as * mut f32))
               .add((((k + warp_id) * MATRIX_SIZE) + col) as usize)
        }
      } else {0.0});
      let mut partial_sum : f32 = (a_val * b_val);
      let mut offset : i32 = 16;
      while (offset > 0) {
        partial_sum += subgroup_shuffle_down(partial_sum, offset);
        offset /= 2;
      }
      if (warp_id == 0) {
        result += partial_sum;
      }
      k += WARP_SIZE;
    }
    if (warp_id == 0) {
      unsafe{
          *((C as * mut f32)).add(linear_id as usize) =
              ((ALPHA * result) +
               (BETA * unsafe{*((C as * mut f32)).add(linear_id as usize)}))};
    }
    elem += 1;
  }
}
