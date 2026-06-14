
struct MatMulParams {
    uint row_dim_x;
    uint col_dim_x;
    uint inner_dim;
};
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> X : register(u2);
ConstantBuffer<MatMulParams> params : register(b3);
// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 id_dispatchThreadID : SV_DispatchThreadID) {
    uint2 id = id_dispatchThreadID.xy;
    const uint row_dim_x = params.row_dim_x;
    const uint col_dim_x = params.col_dim_x;
    const uint inner_dim = params.inner_dim;
    if (((id.x < col_dim_x) && (id.y < row_dim_x))) {
        const uint index = ((id.y * col_dim_x) + id.x);
        float sum = 0;
        for (uint k = 0; (k < inner_dim); ++k) {
            const uint index_A = ((id.y * inner_dim) + k);
            const uint index_B = ((k * col_dim_x) + id.x);
            sum += (A.Load(index_A) * B.Load(index_B));
        }
        X[index] = sum;
    }
}
