#include <hip/device_functions.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/math_functions.h>

struct Matrix32 {
  float[LiteralNode(value = 32,
                    literal_type = PrimitiveType(name = int, size_bits = None))]
       [LiteralNode(value = 32, literal_type = PrimitiveType(
                                    name = int, size_bits = None))] data;
};

struct Matrix64 {
  float[LiteralNode(value = 64,
                    literal_type = PrimitiveType(name = int, size_bits = None))]
       [LiteralNode(value = 64, literal_type = PrimitiveType(
                                    name = int, size_bits = None))] data;
};

struct Matrix128 {
  float[LiteralNode(value = 128,
                    literal_type = PrimitiveType(name = int, size_bits = None))]
       [LiteralNode(value = 128, literal_type = PrimitiveType(
                                     name = int, size_bits = None))] data;
};

struct MatrixDimensions {
  int rows_A;
  int cols_A;
  int rows_B;
  int cols_B;
  int rows_C;
  int cols_C;
};

struct KernelParams {
  int block_size;
  int grid_size_x;
  int grid_size_y;
  float alpha;
  float beta;
};

__device__ float getMatrixElement(buffer_PrimitiveType(name = float,
                                                       size_bits = None) matrix,
                                  int row, int col, int stride) {
  return matrix[((row * stride) + col)];
}

__device__ void setMatrixElement(buffer_PrimitiveType(name = float,
                                                      size_bits = None) matrix,
                                 int row, int col, int stride, float value) {
  matrix[((row * stride) + col)] = value;
}

__global__ void matmul_warp_optimized() {
  int warp_id = IdentifierNode(name = int)(gl_LocalInvocationID.x);
  int global_id = IdentifierNode(name = int)(gl_GlobalInvocationID.x);
  int elements_per_warp =
      ((matrix_size * matrix_size) /
       (IdentifierNode(name = int)(gl_NumWorkGroups.x) * 32));
  int elem = 0;
  for (; (elem < elements_per_warp); ++elem) {
    int linear_id =
        (global_id +
         ((elem * IdentifierNode(name = int)(gl_NumWorkGroups.x)) * 32));
    if ((linear_id >= (matrix_size * matrix_size))) {
      break;
    }
    int row = (linear_id / matrix_size);
    int col = (linear_id % matrix_size);
    float result = 0.0;
    int k = 0;
    for (; (k < matrix_size); k = WARP_SIZE) {
      float a_val = (((k + warp_id) < matrix_size)
                         ? A[(((row * matrix_size) + k) + warp_id)]
                         : 0.0);
      float b_val = (((k + warp_id) < matrix_size)
                         ? B[(((k + warp_id) * matrix_size) + col)]
                         : 0.0);
      float partial_sum = (a_val * b_val);
      int offset = 16;
      for (; (offset > 0); offset = 2) {
        partial_sum = IdentifierNode(name = __shfl_down_sync)(
            4294967295, partial_sum, offset);
      }
      if ((warp_id == 0)) {
        result = partial_sum;
      }
    }
    if ((warp_id == 0)) {
      C[linear_id] = ((alpha * result) + (beta * C[linear_id]));
    }
  }
}
