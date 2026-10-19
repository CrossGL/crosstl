// nvcc defaults to -ftz=false: single-precision subnormal operands and results are preserved.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = inp[i] * 0.5f;
}
