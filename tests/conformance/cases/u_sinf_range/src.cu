// CUDA Math API: sinf has a maximum error of 2 ulp over the full range (without -use_fast_math).
__global__ void hazard(const float *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = sinf(inp[i]);
}
