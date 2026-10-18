// C++: unsigned char arithmetic is promoted to int and truncated modulo 256 on the store back.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char v = (unsigned char)inp[i];
    v = v + 1;
    outp[i] = (int)v;
}
