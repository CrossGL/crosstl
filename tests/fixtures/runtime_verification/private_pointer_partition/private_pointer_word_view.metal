#include <metal_stdlib>
using namespace metal;

struct WordBlock {
    uint words[2];
};

uint sum_bytes(const thread uint8_t* bytes) {
    uint total = 0;
    if constexpr (4 == 4) {
        for (uint index = 0; index < 8; ++index) {
            total += uint(bytes[index]) * (index + 1);
        }
    } else {
        for (uint index = 0; index < 16; ++index) {
            total += uint(bytes[index]) * (index + 1);
        }
    }
    return total;
}

kernel void private_pointer_word_view(
    device uint* output [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    thread WordBlock block;
    block.words[0] = 0x04030201u;
    block.words[1] = 0x08070605u;
    output[tid] = sum_bytes((const thread uint8_t*)&block);
}
