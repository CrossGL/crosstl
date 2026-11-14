#include <hip/hip_runtime.h>
__global__ void k(int* o) { o[threadIdx.x] = warpSize; }
