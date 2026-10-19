// Hazard I3: loop bound computed with signed overflow; wrapping makes the loop run zero times.
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int start = a[id.x];
    int count = 0;
    for (int k = start; k < start + 4; ++k) { count += 1; }
    o[id.x] = count;
}
