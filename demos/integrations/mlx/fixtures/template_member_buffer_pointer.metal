#include <metal_stdlib>
using namespace metal;

struct ReducedMMAFrag {
  template <typename SrcPtrType>
  static float load(SrcPtrType src, const int stride) {
    return static_cast<float>(src[stride]);
  }
};

struct ReducedMMATile {
  float value;

  template <typename U>
  void load(const device U* src, const int index) {
    value = ReducedMMAFrag::load(&(src[index]), 1);
  }
};

kernel void template_member_buffer_pointer(
    const device float* src [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  ReducedMMATile tile;
  tile.load(src, int(gid));
  out[gid] = tile.value;
}
