// Hazard I2: signed multiplication overflow followed by division.
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int x = a[id.x];
    int y = x * 2;
    o[id.x] = (y / 2 == x) ? 1 : 0;
}
