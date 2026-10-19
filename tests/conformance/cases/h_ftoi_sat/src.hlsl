// Hazard H5: D3D float-to-int conversion maps NaN to 0 and clamps out-of-range values.
StructuredBuffer<float> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[id.x] = (int)a[id.x];
}
