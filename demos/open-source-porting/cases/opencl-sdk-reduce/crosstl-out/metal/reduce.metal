
#include <metal_stdlib>
using namespace metal;

// Constant Buffers
struct reduce_Args {
    uint64_t length;
    int zero_elem;
};
int op(int lhs, int rhs) {
    return min(lhs, rhs);
}

int read_local(threadgroup int shared_[1024], uint count, int zero, uint i) {
    return i < count ? shared_[i] : zero;
}

uint zmin(uint a, uint b) {
    return a < b ? a : b;
}

// Compute Shader
kernel void kernel_reduce(uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], uint3 threads_per_threadgroup [[threads_per_threadgroup]], uint3 threadgroups_per_grid [[threadgroups_per_grid]], constant reduce_Args& reduce_Args [[buffer(2)]], device int* front [[buffer(0)]], device int* back [[buffer(1)]]) {
    __attribute__((unused)) threadgroup int shared_[1024];
    uint lid = thread_position_in_threadgroup.x;
    uint lsi = threads_per_threadgroup.x;
    uint wid = threadgroup_position_in_grid.x;
    __attribute__((unused)) uint wsi = threadgroups_per_grid.x;
    uint wg_stride = lsi * 2;
    uint valid_count = zmin(wg_stride, uint(reduce_Args.length) - wid * wg_stride);
    if (lid < valid_count) {
        shared_[lid] = front[wid * wg_stride + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lsi; i != 0; i /= 2) {
        if (lid < i) {
            shared_[lid] = op(read_local(shared_, valid_count, reduce_Args.zero_elem, lid), read_local(shared_, valid_count, reduce_Args.zero_elem, lid + i));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        back[wid] = shared_[0];
    }
}
