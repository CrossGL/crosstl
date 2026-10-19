// CUDA Math API: atan2f(+-0, -0) = +-pi, atan2f(+-0, +0) = +-0, atan2f(+-0, x<0) = +-pi, atan2f(+-0, x>0) = +-0.
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = atan2f(inp[2 * i], inp[2 * i + 1]);
}
