
#include <metal_stdlib>
using namespace metal;

struct MatMulParams {
    uint row_dim_x;
    uint col_dim_x;
    uint inner_dim;
};
// Compute Shader
kernel void mat_mul_simple1(device float* A, device float* B, device float* X, constant MatMulParams& params, uint2 id [[thread_position_in_grid]]) {
    const uint row_dim_x = params.row_dim_x;
    const uint col_dim_x = params.col_dim_x;
    const uint inner_dim = params.inner_dim;
    if (id.x < col_dim_x && id.y < row_dim_x) {
        const uint index = id.y * col_dim_x + id.x;
        float sum = 0;
        for (uint k = 0; k < inner_dim; ++k) {
            const uint index_A = id.y * inner_dim + k;
            const uint index_B = k * col_dim_x + id.x;
            sum += A[index_A] * B[index_B];
        }
        X[index] = sum;
    }
}
