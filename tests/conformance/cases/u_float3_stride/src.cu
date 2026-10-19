// CUDA float3 is 12 bytes with 4-byte alignment.
__global__ void hazard(const float3 *inp, float *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = inp[i].y;
}
