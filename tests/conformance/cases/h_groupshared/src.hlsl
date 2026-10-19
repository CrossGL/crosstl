// groupshared memory is one copy per thread group: each thread reads the value its mirror thread wrote.
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
groupshared int tile[64];
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint l : SV_GroupIndex) {
    tile[l] = a[id.x];
    GroupMemoryBarrierWithGroupSync();
    o[id.x] = tile[63u - l];
}
