// CUDA Math API roundf (C11 7.12.9.6): halfway cases round away from zero, and -0.25f gives -0.0f.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = roundf(inp[i]);
}
