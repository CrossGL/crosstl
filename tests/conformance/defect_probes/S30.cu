__device__ int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
__global__ void k(int* o) { o[threadIdx.x] = fib(threadIdx.x); }
