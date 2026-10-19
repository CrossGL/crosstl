// CUDA Math API: __fmul_rn and __fadd_rn are never merged into a multiply-add.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a = inp[3 * i];
    float b = inp[3 * i + 1];
    float c = inp[3 * i + 2];
    outp[i] = __fadd_rn(__fmul_rn(a, b), -c);
}
