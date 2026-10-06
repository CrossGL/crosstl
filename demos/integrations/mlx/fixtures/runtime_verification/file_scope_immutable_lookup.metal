#include <metal_stdlib>
using namespace metal;

constexpr constant static uint lookup_table[2][4] = {
    {3u, 5u, 7u, 11u},
    {13u, 17u, 19u, 23u}
};

kernel void mlx_file_scope_immutable_lookup_u32(
    device uint* output [[buffer(0)]],
    uint index [[thread_position_in_grid]]
) {
    uint row = index & 1u;
    uint column = (index + 1u) & 3u;
    output[index] = lookup_table[row][column];
}
