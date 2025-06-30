#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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
  // CUDA built-in variables
  int3 threadIdx = {threadIdx.x, threadIdx.y, threadIdx.z};
  int3 blockIdx = {blockIdx.x, blockIdx.y, blockIdx.z};
  int3 blockDim = {blockDim.x, blockDim.y, blockDim.z};
  int3 gridDim = {gridDim.x, gridDim.y, gridDim.z};

  int warp_id;
  int global_id;
  int elements_per_warp;
  int elem;
  for (None; (elem < elements_per_warp); (++elem)) {
    int linear_id;
    if ((linear_id >= (matrix_size * matrix_size))) {
    }
    int row;
    int col;
    float result;
    int k;
    k += WARP_SIZE;
    for (None; (k < matrix_size); None) {
      float a_val;
      float b_val;
      float partial_sum;
      int offset;
      offset /= 2;
      for (None; (offset > 0); None) {
        partial_sum += __shfl_down_sync(4294967295, partial_sum, offset);
      }
      if ((warp_id == 0)) {
        result += partial_sum;
      }
    }
    if ((warp_id == 0)) {
      C[linear_id] = ((alpha * result) + (beta * C[linear_id]));
    }
  }
}
