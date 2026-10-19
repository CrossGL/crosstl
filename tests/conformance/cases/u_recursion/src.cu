// CUDA allows recursive __device__ functions.
__device__ int fib(int n) {
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fib(inp[i]);
}
