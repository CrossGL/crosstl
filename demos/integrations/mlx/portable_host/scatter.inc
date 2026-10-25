// Ports mlx/backend/metal/indexing.cpp: Scatter::eval_gpu and the scatter JIT
// launch it builds. crosstl_backend.cpp includes this file after its dispatch
// helpers, so the port shares that translation unit's copy, span and entry
// checks.
//
// Storage follows the general gather's bound. One launch carries at most
// 65535 index rows along y, so larger index arrays split into rectangular
// slabs that rebase the index and update views; each slab takes MLX's
// work-per-thread choice for its own rows.
//
// Native atomics apply contending updates in an unspecified order. A float32
// sum that some order would round is therefore issued as successive launches
// whose rows never share a destination, so every destination receives its
// updates in index order, as the CPU reference applies them. Every other
// contention keeps MLX's single launch.

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlx::core {
namespace {

constexpr uint64_t max_scatter_storage = (uint64_t(1) << 28) - 33;
constexpr uint64_t max_scatter_rows = 65535;

// A launch covers ``extent`` index rows along ``split`` and every row after
// it, starting at flat row ``first``.
struct ScatterTile {
  uint64_t first;
  size_t split;
  uint64_t extent;
};

int scatter_work(int index_rank, uint64_t rows, uint64_t output_size) {
  const auto ratio = rows / output_size;
  return index_rank <= 1 || ratio < 1 ? 1
      : ratio <= 4                    ? 4
      : ratio < 16                    ? 8
      : ratio < 32                    ? 16
                                      : 32;
}

// The kernel's offset_neg_idx(indices[elem_to_loc(row)], extent).
int64_t scatter_index(const array& index, uint64_t row, int64_t extent) {
  int64_t location = 0;
  for (size_t axis = index.ndim(); axis-- > 0;) {
    location += int64_t(row % index.shape(axis)) * index.strides()[axis];
    row /= index.shape(axis);
  }
  switch (index.dtype()) {
    case int32: {
      const int64_t value = index.data<int32_t>()[location];
      return value < 0 ? value + extent : value;
    }
    case int64: {
      const int64_t value = index.data<int64_t>()[location];
      return value < 0 ? value + extent : value;
    }
    case uint32:
      return index.data<uint32_t>()[location];
    default:
      return static_cast<int64_t>(index.data<uint64_t>()[location]);
  }
}

// Whether a float32 sum is exact in every order. Every finite float32 is an
// integer multiple of 2^-149; when the terms share a coarser power-of-two step
// and their magnitudes sum to fewer than 2^24 steps, no partial sum rounds.
bool exact_in_every_order(const std::vector<float>& terms) {
  std::vector<std::pair<uint32_t, int>> parts;
  int step = std::numeric_limits<int>::max();
  for (const float value : terms) {
    if (!std::isfinite(value)) {
      return false;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t field = (bits >> 23) & 0xff;
    const uint32_t mantissa = field ? (bits & 0x7fffff) | 0x800000 : bits & 0x7fffff;
    if (mantissa) {
      const int exponent = field ? int(field) - 150 : -149;
      step = std::min(step, exponent + std::countr_zero(mantissa));
      parts.emplace_back(mantissa, exponent);
    }
  }
  uint64_t total = 0;
  for (const auto& [mantissa, exponent] : parts) {
    const int shift = exponent - step;
    if (shift >= 24) {
      return false;
    }
    total += shift >= 0 ? uint64_t(mantissa) << shift : mantissa >> -shift;
    if (total >= (uint64_t(1) << 24)) {
      return false;
    }
  }
  return true;
}

// Destination element of update slice element ``element`` for index ``row``.
class ScatterDestinations {
 public:
  ScatterDestinations(
      const std::vector<array>& indices,
      const std::vector<int>& axes,
      const array& out,
      const std::vector<int32_t>& slices)
      : indices_(indices), axes_(axes), out_(out), slices_(slices) {}

  int64_t base(uint64_t row) const {
    int64_t location = 0;
    for (size_t i = 0; i < indices_.size(); ++i) {
      const int axis = axes_[i];
      location += scatter_index(indices_[i], row, out_.shape(axis)) * out_.strides()[axis];
    }
    return location;
  }

  int64_t offset(uint64_t element) const {
    int64_t location = 0;
    for (size_t axis = slices_.size(); axis-- > 0;) {
      location += int64_t(element % slices_[axis]) * out_.strides()[axis];
      element /= slices_[axis];
    }
    return location;
  }

 private:
  const std::vector<array>& indices_;
  const std::vector<int>& axes_;
  const array& out_;
  const std::vector<int32_t>& slices_;
};

// Whether native contention could land a float32 sum differently from the
// index-order reference.
bool scatter_order_dependent(
    const ScatterDestinations& destinations,
    const array& updates,
    const array& out,
    uint64_t rows,
    uint64_t slice_size) {
  std::unordered_map<int64_t, std::vector<float>> terms;
  bool contended = false;
  const float* initial = out.data<float>();
  const float* data = updates.data<float>();
  for (uint64_t row = 0; row < rows; ++row) {
    const int64_t base = destinations.base(row);
    for (uint64_t element = 0; element < slice_size; ++element) {
      const int64_t destination = base + destinations.offset(element);
      if (destination < 0 || uint64_t(destination) >= out.size()) {
        // The runtime rejects the index; keep MLX's launch for that check.
        return false;
      }
      auto [slot, inserted] = terms.try_emplace(destination);
      if (inserted) {
        slot->second.push_back(initial[destination]);
      }
      // elem_to_loc of the flat update element over the update view.
      uint64_t flat = row * slice_size + element;
      int64_t location = 0;
      for (size_t axis = updates.ndim(); axis-- > 0;) {
        location += int64_t(flat % updates.shape(axis)) * updates.strides()[axis];
        flat /= updates.shape(axis);
      }
      slot->second.push_back(data[location]);
      contended = contended || slot->second.size() > 2;
    }
  }
  if (!contended) {
    return false;
  }
  for (const auto& [destination, sums] : terms) {
    if (sums.size() > 2 && !exact_in_every_order(sums)) {
      return true;
    }
  }
  return false;
}

// Rectangular slabs of at most max_scatter_rows rows.
std::vector<ScatterTile> scatter_slabs(const std::vector<int>& shape, uint64_t rows) {
  if (shape.empty()) {
    return {{0, 0, 1}};
  }
  size_t split = 0;
  uint64_t suffix = rows / shape[0];
  while (suffix > max_scatter_rows) {
    ++split;
    suffix /= shape[split];
  }
  const uint64_t limit = max_scatter_rows / suffix;
  std::vector<ScatterTile> tiles;
  for (uint64_t first = 0; first < rows;) {
    const uint64_t coordinate = (first / suffix) % shape[split];
    const uint64_t extent = std::min<uint64_t>(limit, shape[split] - coordinate);
    tiles.push_back({first, split, extent});
    first += extent * suffix;
  }
  return tiles;
}

// Runs of rows along the last index axis whose destinations are disjoint.
std::vector<ScatterTile> scatter_runs(
    const ScatterDestinations& destinations,
    const std::vector<int>& shape,
    uint64_t rows,
    uint64_t slice_size) {
  const size_t split = shape.size() - 1;
  const uint64_t line = shape[split];
  std::vector<ScatterTile> tiles;
  std::unordered_set<int64_t> touched;
  uint64_t first = 0;
  for (uint64_t row = 0; row < rows; ++row) {
    const int64_t base = destinations.base(row);
    bool shared = row - first == max_scatter_rows || row % line == 0;
    for (uint64_t element = 0; !shared && element < slice_size; ++element) {
      shared = touched.count(base + destinations.offset(element)) != 0;
    }
    if (row > first && shared) {
      tiles.push_back({first, split, row - first});
      touched.clear();
      first = row;
    }
    for (uint64_t element = 0; element < slice_size; ++element) {
      touched.insert(base + destinations.offset(element));
    }
  }
  tiles.push_back({first, split, rows - first});
  return tiles;
}

} // namespace

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() < 3 || inputs.size() > 12 || axes_.size() != inputs.size() - 2 ||
      (out.dtype() != int32 && out.dtype() != uint32 && out.dtype() != float32) ||
      inputs[0].dtype() != out.dtype() || inputs.back().dtype() != out.dtype() ||
      inputs[0].shape() != out.shape() || out.ndim() == 0 || out.ndim() > 64 ||
      out.size() > max_scatter_storage || inputs.back().size() > max_scatter_storage ||
      (reduce_type_ != None && reduce_type_ != Sum && reduce_type_ != Prod &&
       reduce_type_ != Min && reduce_type_ != Max)) {
    throw std::invalid_argument("CrossTL scatter requires bounded int32, uint32 or float32 indexed updates.");
  }
  const auto index_dtype = inputs[1].dtype();
  if (index_dtype != int32 && index_dtype != uint32 &&
      index_dtype != int64 && index_dtype != uint64) {
    throw std::invalid_argument("CrossTL scatter requires 32-bit or 64-bit integer indices.");
  }
  const int32_t index_rank = static_cast<int32_t>(inputs[1].ndim());
  if (index_rank > 64 || inputs.back().ndim() > 64 ||
      inputs.back().ndim() != index_rank + out.ndim() ||
      inputs[1].size() > max_scatter_storage) {
    throw std::invalid_argument("CrossTL scatter update and index ranks do not match.");
  }
  for (size_t i = 0; i < axes_.size(); ++i) {
    if (axes_[i] < 0 || axes_[i] >= out.ndim() ||
        inputs[i + 1].dtype() != index_dtype || inputs[i + 1].shape() != inputs[1].shape()) {
      throw std::invalid_argument("CrossTL scatter indices require matching shapes and types.");
    }
  }
  // Copy before updating so aliased source and update views remain unchanged.
  dispatch_copy(inputs[0], out);
  if (inputs.back().size() == 0) {
    return;
  }
  if (!out.size()) {
    throw std::invalid_argument("CrossTL scatter cannot update an empty output.");
  }
  auto upd = gather_input(inputs.back());
  const auto update_span = gather_span(upd, max_scatter_storage);
  const uint64_t idx_size = inputs[1].size();
  uint64_t upd_ndim = upd.ndim(), out_ndim = out.ndim();
  uint64_t upd_size = 1;
  for (size_t i = index_rank; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  // The slice runs along x; only index rows split into launches.
  if (upd_size > max_scatter_rows) {
    throw std::invalid_argument("CrossTL scatter launch exceeds per-axis workgroup limits.");
  }
  const char* operation = reduce_type_ == None ? "none" :
      reduce_type_ == Sum ? "sum" : reduce_type_ == Prod ? "prod" :
      reduce_type_ == Min ? "min" : "max";
  const auto count = axes_.size();
  std::vector<array> indices;
  std::vector<uint64_t> spans;
  std::vector<int32_t> shapes;
  std::vector<int64_t> strides;
  std::vector<uint8_t> contiguous;
  std::vector<std::string> names;
  for (size_t i = 0; i < count; ++i) {
    indices.push_back(gather_input(inputs[i + 1]));
    spans.push_back(gather_span(indices.back(), max_scatter_storage));
    shapes.insert(shapes.end(), indices.back().shape().begin(), indices.back().shape().end());
    strides.insert(strides.end(), indices.back().strides().begin(), indices.back().strides().end());
    contiguous.push_back(indices.back().flags().row_contiguous);
    names.push_back("idx" + std::to_string(i));
  }
  if (index_rank == 0) {
    shapes.push_back(0);
    strides.push_back(0);
    contiguous.push_back(false);
  }
  std::vector<int32_t> update_shape(upd.shape().begin(), upd.shape().end());
  std::vector<int64_t> update_strides(upd.strides().begin(), upd.strides().end());
  std::vector<int32_t> output_shape(out.shape().begin(), out.shape().end());
  std::vector<int64_t> output_strides(out.strides().begin(), out.strides().end());
  std::vector<int32_t> axes(axes_.begin(), axes_.end());
  const std::vector<int> index_shape(inputs[1].shape().begin(), inputs[1].shape().end());
  const std::vector<int32_t> slices(update_shape.begin() + index_rank, update_shape.end());
  const ScatterDestinations destinations(indices, axes_, out, slices);
  const bool ordered = index_rank > 0 && idx_size > 1 && out.dtype() == float32 &&
      reduce_type_ == Sum &&
      scatter_order_dependent(destinations, upd, out, idx_size, upd_size);
  const auto tiles = ordered ? scatter_runs(destinations, index_shape, idx_size, upd_size)
                             : scatter_slabs(index_shape, idx_size);
  int32_t idx_ndim = index_rank;
  uint64_t tile_size = idx_size;
  std::vector<CrosstlMlxBuffer> buffers = {
      {"updates", storage_type(upd.dtype()), const_cast<void*>(upd.data<void>()), update_span, 0},
      {"out", storage_type(out.dtype()), out.data<void>(), out.size(), 1},
      {"upd_shape", "int32", update_shape.data(), update_shape.size(), 0},
      {"upd_strides", "int64", update_strides.data(), update_strides.size(), 0},
      {"upd_ndim", "uint64", &upd_ndim, 1, 0},
      {"upd_size", "uint64", &upd_size, 1, 0},
      {"out_shape", "int32", output_shape.data(), output_shape.size(), 0},
      {"out_strides", "int64", output_strides.data(), output_strides.size(), 0},
      {"out_ndim", "uint64", &out_ndim, 1, 0},
      {"axes", "int32", axes.data(), count, 0},
      {"idx_shapes", "int32", shapes.data(), shapes.size(), 0},
      {"idx_strides", "int64", strides.data(), strides.size(), 0},
      {"idx_contigs", "bool_", contiguous.data(), contiguous.size(), 0},
      {"idx_ndim", "int32", &idx_ndim, 1, 0},
      {"idx_size", "uint64", &tile_size, 1, 0},
  };
  for (size_t i = 0; i < count; ++i) {
    buffers.push_back({names[i].c_str(), storage_type(index_dtype),
                      const_cast<void*>(indices[i].data<void>()), spans[i], 0});
  }
  for (const auto& tile : tiles) {
    std::vector<int32_t> tile_shape(index_shape.begin(), index_shape.end());
    std::vector<uint64_t> coordinates(index_rank);
    for (size_t axis = index_rank, remaining = tile.first; axis-- > 0;) {
      coordinates[axis] = remaining % tile_shape[axis];
      remaining /= tile_shape[axis];
    }
    if (index_rank) {
      std::fill(tile_shape.begin(), tile_shape.begin() + tile.split, 1);
      tile_shape[tile.split] = static_cast<int32_t>(tile.extent);
    }
    tile_size = 1;
    for (const auto extent : tile_shape) {
      tile_size *= extent;
    }
    if (tiles.size() > 1) {
      // Rebase each index view to the slab, which stays dense when its rows are.
      for (size_t i = 0; i < count; ++i) {
        uint64_t offset = 0, span = 1, dense_stride = 1;
        bool dense = true;
        for (size_t axis = index_rank; axis-- > 0;) {
          const uint64_t stride = indices[i].strides()[axis];
          offset += coordinates[axis] * stride;
          span += uint64_t(tile_shape[axis] - 1) * stride;
          dense = dense && (tile_shape[axis] == 1 || stride == dense_stride);
          dense_stride *= tile_shape[axis];
          shapes[i * index_rank + axis] = tile_shape[axis];
        }
        if (offset > spans[i] || span > spans[i] - offset) {
          throw std::invalid_argument("CrossTL scatter tile exceeds its index allocation.");
        }
        contiguous[i] = dense;
        buffers[15 + i].data = const_cast<uint8_t*>(indices[i].data<uint8_t>()) +
            offset * indices[i].itemsize();
        buffers[15 + i].count = span;
      }
      // The slab's updates keep every slice axis and the slab's index rows.
      uint64_t offset = 0, span = 1;
      for (size_t axis = 0; axis < upd.ndim(); ++axis) {
        const uint64_t stride = upd.strides()[axis];
        if (axis < size_t(index_rank)) {
          offset += coordinates[axis] * stride;
          update_shape[axis] = tile_shape[axis];
        }
        span += uint64_t(update_shape[axis] - 1) * stride;
      }
      if (offset > update_span || span > update_span - offset) {
        throw std::invalid_argument("CrossTL scatter tile exceeds its update allocation.");
      }
      buffers[0].data = const_cast<uint8_t*>(upd.data<uint8_t>()) + offset * upd.itemsize();
      buffers[0].count = span;
    }
    const int work = scatter_work(index_rank, tile_size, out.size());
    const std::string entry = std::string("scatter") + storage_type(out.dtype()) +
        storage_type(index_dtype) + "_" + operation + "_" + std::to_string(count) +
        "_updc_" + (upd.flags().row_contiguous ? "true" : "false") +
        "_nwork" + std::to_string(work) + "_int";
    require_entry(entry);
    const CrosstlMlxLaunch launch{
        {static_cast<uint32_t>(upd_size), static_cast<uint32_t>((tile_size + work - 1) / work), 1},
        {1, 1, 1}};
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
        out.size(), &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native scatter failed: ") + error);
    }
  }
}

} // namespace mlx::core
