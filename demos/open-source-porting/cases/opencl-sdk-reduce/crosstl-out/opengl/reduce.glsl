#version 450 core
layout(std430, binding = 0) buffer frontBuffer { int front[]; };
layout(std430, binding = 1) buffer backBuffer { int back[]; };
shared int shared_[1024];
// Constant Buffers
layout(std140, binding = 2) uniform reduce_Args {
    uint length;
    int zero_elem;
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
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint lid = gl_LocalInvocationID.x;
    uint lsi = gl_WorkGroupSize.x;
    uint wid = gl_WorkGroupID.x;
    uint wsi = gl_NumWorkGroups.x;
    uint wg_stride = (lsi * 2);
    uint valid_count = zmin(wg_stride, (uint(length) - (wid * wg_stride)));
    if ((lid < valid_count)) {
        shared_[lid] = front[((wid * wg_stride) + lid)];
    }
    barrier();
    for (uint i = lsi; (i != 0); i /= 2) {
        if ((lid < i)) {
            shared_[lid] = op(read_local(shared_, valid_count, zero_elem, lid), read_local(shared_, valid_count, zero_elem, (lid + i)));
        }
        barrier();
    }
    if ((lid == 0)) {
        back[wid] = shared_[0];
    }
}
