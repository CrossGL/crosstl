// HLSL: the signed operand is converted to the unsigned type (working draft 5.1 [Expr.conv]).
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int x = a[id.x];
    uint four = 4u;
    o[id.x] = (x < four) ? 1 : 0;
}
