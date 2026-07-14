#include <metal_stdlib>
using namespace metal;

struct ReferenceAccessorTile {
  static constexpr short width = 2;
  float val_frags[4];

  constexpr thread float& frag_at(const short i, const short j) {
    return val_frags[i * width + j];
  }
};

kernel void reference_accessor_write(device float* out [[buffer(0)]]) {
  ReferenceAccessorTile tile;
  tile.frag_at(1, 1) = 73.25f;
  out[0] = tile.val_frags[(1 * 2) + 1];
}
