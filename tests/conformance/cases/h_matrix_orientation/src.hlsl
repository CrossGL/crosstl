// S11 (HLSL): a floatRxC has R rows and C columns. Constructors and initializer lists fill rows, m[i] is row i,
// m._mRC and m._RC name element (R, C), mul(m, v) takes a column vector and mul(v, m) a row vector, * of two
// matrices is componentwise, (floatRxC) of a matrix keeps the upper-left elements and of a scalar splats it.
StructuredBuffer<float> a : register(t0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    uint b = i * 6;
    uint w = i * 24;
    int k = int(i & 1);
    float2x3 m = float2x3(a[b], a[b + 1], a[b + 2], a[b + 3], a[b + 4], a[b + 5]);
    float2 mv = mul(m, float3(1.0, 10.0, 100.0));
    float3 vm = mul(float2(1.0, 10.0), m);
    float3 r1 = m[1];
    float3 rk = m[k];
    o[w + 0] = mv.x;
    o[w + 1] = mv.y;
    o[w + 2] = vm.x;
    o[w + 3] = vm.y;
    o[w + 4] = vm.z;
    o[w + 5] = r1.x;
    o[w + 6] = r1.z;
    o[w + 7] = m[0][2];
    o[w + 8] = rk.y;
    o[w + 9] = m._m12 + m._11 * 10.0;
    float2 sw = m._m02_m10;
    o[w + 10] = sw.x + sw.y * 100.0;
    m[1] = m[0] + r1;
    m[0][1] = 7.0;
    o[w + 11] = m[1].x;
    o[w + 12] = m[1].z;
    o[w + 13] = m[0].y;
    m[k] = float3(a[b + 5], 2.0, 3.0);
    o[w + 14] = m[1].x + m[0].y * 10.0;
    m[k] += float3(1.0, 20.0, 300.0);
    o[w + 15] = m[k].x + m[k].z;
    float2x3 cm = m * m;
    o[w + 16] = cm[1][2];
    float3x2 t = transpose(m);
    o[w + 17] = mul(m, t)[1][0];
    o[w + 18] = mul(t, m)[2][0];
    float2x2 tr = (float2x2)m;
    o[w + 19] = tr[1][1] + tr._m01 * 10.0;
    float2x2 il = {a[b], 2.0, 3.0, a[b + 1]};
    o[w + 20] = il[0][1] + il[1][0] * 10.0 + il._22 * 100.0;
    float3x3 z = (float3x3)a[b];
    o[w + 21] = z[2][1] + z[0][0] * 10.0;
    m._m01 = 9.0;
    o[w + 22] = m[0].y;
    float2x2 fm = float2x2(float4(1.0, 2.0, 3.0, a[b]));
    o[w + 23] = fm[1][0] + fm[0][1] * 10.0 + fm[1][1] * 100.0;
}
