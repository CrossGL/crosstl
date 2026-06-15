#version 450 core
layout(std430, binding = 0) readonly buffer ABuffer { float A[]; };
layout(std430, binding = 1) readonly buffer BBuffer { float B[]; };
layout(std430, binding = 2) buffer XBuffer { float X[]; };
layout(std140, binding = 3) uniform MatMulParams {
    uint row_dim_x;
    uint col_dim_x;
    uint inner_dim;
} params;
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uvec2 id = uvec2(gl_GlobalInvocationID.xy);
    const uint row_dim_x = params.row_dim_x;
    const uint col_dim_x = params.col_dim_x;
    const uint inner_dim = params.inner_dim;
    if (((id.x < col_dim_x) && (id.y < row_dim_x))) {
        const uint index = ((id.y * col_dim_x) + id.x);
        float sum = 0;
        for (uint k = 0; (k < inner_dim); (++k)) {
            const uint index_A = ((id.y * inner_dim) + k);
            const uint index_B = ((k * col_dim_x) + id.x);
            sum += (A[index_A] * B[index_B]);
        }
        X[index] = sum;
    }
}
