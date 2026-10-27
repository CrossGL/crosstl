// C++ [dcl.init.ref] and [expr.call]: a non-const lvalue reference parameter binds the caller's object, so every write
// the callee makes through it reaches the caller, for a scalar and for a structure; a const reference only reads it.
struct Pair {
    int a;
    int b;
};
__device__ void bump(int& v, int by) { v += by; }
__device__ void swap_pair(Pair& p) {
    int t = p.a;
    p.a = p.b;
    p.b = t;
}
__device__ int twice(const int& c) { return c * 2; }
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = inp[i];
    bump(x, 5);
    Pair p;
    p.a = x;
    p.b = (int)i;
    swap_pair(p);
    outp[i] = x * 7 + p.a * 3 + p.b + twice(x);
}
