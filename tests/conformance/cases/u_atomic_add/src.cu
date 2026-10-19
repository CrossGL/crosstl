// Sanity: every thread atomically adds its input to one counter.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&outp[0], inp[i]);
}
