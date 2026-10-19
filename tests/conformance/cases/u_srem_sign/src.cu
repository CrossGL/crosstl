// C++ [expr.mul]/4: the quotient truncates toward zero and (a/b)*b + a%b == a.
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int a = inp[2 * i];
    int b = inp[2 * i + 1];
    outp[i] = a % b;
}
