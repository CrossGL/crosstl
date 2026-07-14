#include <metal_stdlib>
using namespace metal;

struct ReferenceAccessorOps {
  static void store(const thread float& value, device float* out) {
    out[1] = value;
  }
};

struct ReferenceAccessorTile {
  static constexpr short width = 2;
  float val_frags[4];

  constexpr thread float& frag_at(const short i, const short j) {
    return val_frags[i * width + j];
  }

  constexpr const thread float& frag_at(
      const short i,
      const short j) const {
    return val_frags[i * width + j];
  }

  void store(device float* out, const short i, const short j) const {
    ReferenceAccessorOps::store(frag_at(i, j), out);
  }
};

kernel void reference_accessor_write(device float* out [[buffer(0)]]) {
  using tile_t = ReferenceAccessorTile;
  tile_t tile;
  tile.frag_at(1, 1) = 73.25f;
  out[0] = tile.val_frags[(1 * 2) + 1];
  tile.store(out, 1, 1);
}
