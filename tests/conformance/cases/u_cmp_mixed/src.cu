// C++ usual arithmetic conversions: the int operand converts to unsigned int.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = inp[i];
    unsigned int four = 4u;
    outp[i] = (x < four) ? 1 : 0;
}
