// CUDA fmodf (C99): x - trunc(x / y) * y computed exactly; the result has the sign of x.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fmodf(inp[2 * i], inp[2 * i + 1]);
}
