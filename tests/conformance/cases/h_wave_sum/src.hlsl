// WaveActiveSum reduces over the active lanes of the wave; the wave size is reported to the host.
StructuredBuffer<int> a : register(t0);
RWStructuredBuffer<int> o : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int s = WaveActiveSum(a[id.x]);
    o[2 * id.x] = s;
    o[2 * id.x + 1] = (int)WaveGetLaneCount();
}
