// 64-bit unsigned product; the high word depends on all 64 bits.
__global__ void hazard(const unsigned int *inp, unsigned int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long p = (unsigned long long)inp[i] * 0xFFFFFFFFull;
    outp[i] = (unsigned int)(p >> 32);
}
