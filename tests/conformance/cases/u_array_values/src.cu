// C++ [dcl.array] and [dcl.fct]: a local or member T a[N] holds N elements, an array of arrays has its first extent
// outermost, and a parameter declared const T v[N] reads the caller's array.
struct Window {
    int taps[3];
};
__device__ int sum3(const int v[3]) { return v[0] + v[1] + v[2]; }
__device__ int corner(const int g[2][3]) { return g[1][2] - g[0][0]; }
__global__ void hazard(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = inp[i];
    int a[3];
    a[0] = x;
    a[1] = x + 1;
    a[2] = x + 2;
    Window w;
    for (int k = 0; k < 3; ++k) {
        w.taps[k] = a[k] * (k + 1);
    }
    int g[2][3];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            g[r][c] = x * r + c;
        }
    }
    outp[i] = sum3(a) + sum3(w.taps) * 2 + corner(g);
}
