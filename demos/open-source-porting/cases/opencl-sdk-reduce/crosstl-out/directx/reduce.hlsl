
RWStructuredBuffer<int> front : register(u0);
RWStructuredBuffer<int> back : register(u1);
groupshared int shared_[1024];
// Constant Buffers
cbuffer reduce_Args : register(b2) {
    uint64_t length;
    int zero_elem;
};
cbuffer CrossGLDispatchInfo : register(b3) {
    uint3 crossglNumWorkGroups;
};
int op(int lhs, int rhs) {
    return min(lhs, rhs);
}

int read_local(int shared_[1024], uint count, int zero, uint i) {
    return ((i < count) ? shared_[i] : zero);
}

uint zmin(uint a, uint b) {
    return ((a < b) ? a : b);
}

// Compute Shader
[numthreads(1, 1, 1)]
void CSMain(uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID) {
    uint lid = groupThreadID.x;
    uint lsi = uint3(1, 1, 1).x;
    uint wid = groupID.x;
    uint wsi = crossglNumWorkGroups.x;
    uint wg_stride = (lsi * 2);
    uint valid_count = zmin(wg_stride, (uint(length) - (wid * wg_stride)));
    if (lid < valid_count) {
        shared_[lid] = front[((wid * wg_stride) + lid)];
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = lsi; (i != 0); i /= 2) {
        if (lid < i) {
            shared_[lid] = op(read_local(shared_, valid_count, zero_elem, lid), read_local(shared_, valid_count, zero_elem, (lid + i)));
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0) {
        back[wid] = shared_[0];
    }
}
