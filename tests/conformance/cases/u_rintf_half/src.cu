// CUDA Math API rintf (C11 7.12.9.4): halfway cases round to even in the round-to-nearest mode.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = rintf(inp[i]);
}
