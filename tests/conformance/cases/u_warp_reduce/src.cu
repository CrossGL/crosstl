// warpSize is 32: a full-warp shuffle-down reduction leaves the warp's sum in lane 0.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = inp[i];
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if ((threadIdx.x & 31u) == 0u) {
        outp[i >> 5] = v;
    }
}
