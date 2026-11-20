#include <cuda_bf16.h>
__global__ void k(const __nv_bfloat16* a, float* o, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) o[i] = __bfloat162float(a[i]); }
