// CUDA Math API: fminf treats a NaN argument as missing data and returns the other argument.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fminf(inp[2 * i], inp[2 * i + 1]);
}
