// C++ pointer arithmetic on a device pointer.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int *p = inp + 3;
    outp[i] = p[i];
}
