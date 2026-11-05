// Ports mlx/backend/metal/indexing.cpp: Scatter::eval_gpu and the scatter JIT
// launch it builds, ScatterAxis::eval_gpu and MaskedScatter::eval_gpu.
// crosstl_backend.cpp includes this file after its dispatch helpers, so the
// port shares that translation unit's copy, span, entry and axis-launch
// checks.
//
// Storage follows the general gather's bound. One launch carries at most
// 65535 index rows along y, so larger index arrays split into rectangular
// slabs that rebase the index and update views; each slab takes MLX's
// work-per-thread choice for its own rows.
//
// The 8- and 16-bit element types update one lane of a packed atomic word,
// as MLX's mlx_atomic<T> does, so the host carries their destinations as
// those words.
//
// Native atomics apply contending updates in an unspecified order. A float32,
// float16 or bfloat16 sum that some order would round is therefore issued as
// successive launches whose rows never share a destination, so every
// destination receives its updates in index order, as the CPU reference
// applies them. Every other contention keeps MLX's single launch. An axis
// scatter's float32 sum does the same, one leading position per launch.

#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlx::core {
namespace {

constexpr uint64_t max_scatter_storage = (uint64_t(1) << 28) - 33;
constexpr uint64_t max_scatter_rows = 65535;
// A masked scatter runs one launch of one thread per mask element along x,
// after a scan that the host bounds by the same extent.
constexpr uint64_t max_masked_scatter_elements = 65535;

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

// Whether a sum in a binary floating type of ``precision`` significand bits
// is exact in every order. Every term is a finite value of that type held
// exactly as a float32, so it is an integer multiple of the type's smallest
// step; when the terms share a coarser power-of-two step and their magnitudes
// sum to fewer than 2^precision steps, no partial sum rounds. No partial sum
// overflows either: such a sum is at most the type's largest finite value,
// because no term's step exceeds the step of that value.
bool exact_in_every_order(const std::vector<float>& terms, int precision = 24) {
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
    if (shift >= precision) {
      return false;
    }
    total += shift >= 0 ? uint64_t(mantissa) << shift : mantissa >> -shift;
    if (total >= (uint64_t(1) << precision)) {
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

// Whether native contention could land a sum of T differently from the
// index-order reference.
template <typename T>
bool scatter_order_dependent(
    const ScatterDestinations& destinations,
    const array& updates,
    const array& out,
    uint64_t rows,
    uint64_t slice_size,
    int precision) {
  std::unordered_map<int64_t, std::vector<float>> terms;
  bool contended = false;
  const T* initial = out.data<T>();
  const T* data = updates.data<T>();
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
        slot->second.push_back(static_cast<float>(initial[destination]));
      }
      // elem_to_loc of the flat update element over the update view.
      uint64_t flat = row * slice_size + element;
      int64_t location = 0;
      for (size_t axis = updates.ndim(); axis-- > 0;) {
        location += int64_t(flat % updates.shape(axis)) * updates.strides()[axis];
        flat /= updates.shape(axis);
      }
      slot->second.push_back(static_cast<float>(data[location]));
      contended = contended || slot->second.size() > 2;
    }
  }
  if (!contended) {
    return false;
  }
  for (const auto& [destination, sums] : terms) {
    if (sums.size() > 2 && !exact_in_every_order(sums, precision)) {
      return true;
    }
  }
  return false;
}

// Whether a floating sum must run in index order; integer sums wrap and
// every other operation is independent of the order.
bool scatter_sum_order_dependent(
    const ScatterDestinations& destinations,
    const array& updates,
    const array& out,
    uint64_t rows,
    uint64_t slice_size) {
  switch (out.dtype()) {
    case float32:
      return scatter_order_dependent<float>(destinations, updates, out, rows, slice_size, 24);
    case float16:
      return scatter_order_dependent<float16_t>(destinations, updates, out, rows, slice_size, 11);
    case bfloat16:
      return scatter_order_dependent<bfloat16_t>(destinations, updates, out, rows, slice_size, 8);
    default:
      return false;
  }
}

// Whether ``dtype`` is an element type the host's scatter specializations carry.
bool scatter_value_type(Dtype dtype) {
  switch (dtype) {
    case int32:
    case uint32:
    case float32:
    case uint8:
    case int8:
    case uint16:
    case int16:
    case float16:
    case bfloat16:
      return true;
    default:
      return false;
  }
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

// Destination element in ``out`` of flat element ``element`` of an axis
// scatter's index view, or -1 for an index outside the scattered axis.
int64_t axis_destination(const array& idx, const array& out, int axis, uint64_t element) {
  const int64_t extent = out.shape(axis);
  const int64_t value = scatter_index(idx, element, extent);
  if (value < 0 || value >= extent) {
    return -1;
  }
  int64_t location = 0;
  for (size_t i = idx.ndim(); i-- > 0;) {
    const int64_t coordinate = i == size_t(axis) ? value : int64_t(element % idx.shape(i));
    location += coordinate * out.strides()[i];
    element /= idx.shape(i);
  }
  return location;
}

// Whether native contention could land a float32 axis sum differently from
// the index-order reference: some destination receives two or more updates
// whose sum some order would round.
bool axis_sum_order_dependent(const array& idx, const array& upd, const array& out, int axis) {
  std::unordered_map<int64_t, std::vector<float>> terms;
  bool contended = false;
  const float* initial = out.data<float>();
  const float* data = upd.data<float>();
  for (uint64_t element = 0; element < idx.size(); ++element) {
    const int64_t destination = axis_destination(idx, out, axis, element);
    if (destination < 0) {
      // The runtime rejects the index; keep MLX's launches for that check.
      return false;
    }
    auto [slot, inserted] = terms.try_emplace(destination);
    if (inserted) {
      slot->second.push_back(initial[destination]);
    }
    // elem_to_loc of the flat element over the update view.
    uint64_t flat = element;
    int64_t location = 0;
    for (size_t i = upd.ndim(); i-- > 0;) {
      location += int64_t(flat % upd.shape(i)) * upd.strides()[i];
      flat /= upd.shape(i);
    }
    slot->second.push_back(data[location]);
    contended = contended || slot->second.size() > 2;
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

// Splits each launch of one leading position into maximal runs of
// consecutive index rows whose destinations are disjoint. Destinations of one
// row differ in their trailing coordinates, so only rows can share one, and
// every destination receives its updates in index order.
std::vector<AxisLaunchTile> axis_sum_runs(
    const std::vector<AxisLaunchTile>& tiles,
    const array& idx,
    const array& out,
    int axis,
    uint64_t after) {
  const uint64_t rows = idx.shape(axis);
  std::vector<AxisLaunchTile> runs;
  std::unordered_set<int64_t> touched;
  for (const auto& tile : tiles) {
    AxisLaunchTile run = tile;
    const uint64_t end = tile.first_row + tile.rows;
    uint64_t first = tile.first_row;
    touched.clear();
    for (uint64_t row = tile.first_row; row < end; ++row) {
      const uint64_t base = (tile.first_pre * rows + row) * after;
      bool shared = false;
      for (uint64_t x = 0; !shared && x < after; ++x) {
        shared = touched.count(axis_destination(idx, out, axis, base + x)) != 0;
      }
      if (row > first && shared) {
        run.first_row = first;
        run.rows = row - first;
        runs.push_back(run);
        touched.clear();
        first = row;
      }
      for (uint64_t x = 0; x < after; ++x) {
        touched.insert(axis_destination(idx, out, axis, base + x));
      }
    }
    run.first_row = first;
    run.rows = end - first;
    runs.push_back(run);
  }
  return runs;
}

} // namespace

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() < 3 || inputs.size() > 12 || axes_.size() != inputs.size() - 2 ||
      !scatter_value_type(out.dtype()) ||
      inputs[0].dtype() != out.dtype() || inputs.back().dtype() != out.dtype() ||
      inputs[0].shape() != out.shape() || out.ndim() == 0 || out.ndim() > 64 ||
      out.size() > max_scatter_storage || inputs.back().size() > max_scatter_storage ||
      (reduce_type_ != None && reduce_type_ != Sum && reduce_type_ != Prod &&
       reduce_type_ != Min && reduce_type_ != Max)) {
    throw std::invalid_argument(
        "CrossTL scatter requires bounded indexed updates of 8-, 16- or 32-bit "
        "integers, float16, bfloat16 or float32.");
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
  const bool ordered = index_rank > 0 && idx_size > 1 && reduce_type_ == Sum &&
      scatter_sum_order_dependent(destinations, upd, out, idx_size, upd_size);
  const auto tiles = ordered ? scatter_runs(destinations, index_shape, idx_size, upd_size)
                             : scatter_slabs(index_shape, idx_size);
  int32_t idx_ndim = index_rank;
  uint64_t tile_size = idx_size;
  std::vector<CrosstlMlxBuffer> buffers = {
      {"updates", cast_type(upd.dtype()), const_cast<void*>(upd.data<void>()), update_span, 0},
      {"out", cast_type(out.dtype()), out.data<void>(), out.size(), 1},
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
    const std::string entry = std::string("scatter") + cast_type(out.dtype()) +
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

// Ports indexing.cpp ScatterAxis::eval_gpu.
void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  constexpr uint64_t max_storage_elements = (uint64_t(1) << 28) - 33;
  // The 32-bit types whose atomics every target provides.
  if (inputs.size() != 3 ||
      (out.dtype() != int32 && out.dtype() != uint32 && out.dtype() != float32) ||
      inputs[0].dtype() != out.dtype() || inputs[2].dtype() != out.dtype() ||
      inputs[0].shape() != out.shape() || out.ndim() == 0 || out.ndim() > 64 ||
      inputs[1].ndim() != out.ndim() || inputs[2].shape() != inputs[1].shape() ||
      axis_ < 0 || axis_ >= out.ndim() || out.size() > max_storage_elements ||
      inputs[1].size() > max_storage_elements ||
      (reduce_type_ != None && reduce_type_ != Sum)) {
    throw std::invalid_argument("CrossTL axis scatter requires bounded int32, uint32 or float32 arrays.");
  }
  const auto index_dtype = inputs[1].dtype();
  if (index_dtype != int32 && index_dtype != uint32 &&
      index_dtype != int64 && index_dtype != uint64) {
    throw std::invalid_argument("CrossTL axis scatter requires 32-bit or 64-bit integer indices.");
  }
  uint64_t after = 1;
  for (size_t i = 0; i < out.ndim(); ++i) {
    if (i != axis_ && inputs[1].shape(i) != out.shape(i)) {
      throw std::invalid_argument("CrossTL axis scatter requires broadcast input shapes.");
    }
    if (i > axis_) {
      after *= out.shape(i);
    }
  }
  if (inputs[1].size() && after > 65535) {
    throw std::invalid_argument("CrossTL axis scatter launch exceeds per-axis workgroup limits.");
  }
  // Preserve the source allocation, including when updates alias a source view.
  dispatch_copy(inputs[0], out);
  if (inputs[1].size() == 0) {
    return;
  }
  auto upd = gather_input(inputs[2]);
  auto idx = gather_input(inputs[1]);
  const auto update_span = gather_span(upd, max_storage_elements);
  const auto index_span = gather_span(idx, max_storage_elements);
  const std::string entry = std::string("scatter_axis") + storage_type(out.dtype()) +
      storage_type(index_dtype) + (reduce_type_ == None ? "_none_int" : "_sum_int") +
      (upd.flags().row_contiguous ? "c" : "nc") + (idx.flags().row_contiguous ? "c" : "nc");
  require_entry(entry);
  std::vector<int32_t> shape, pre_shape;
  std::vector<int64_t> update_strides, index_strides;
  for (size_t i = 0; i < out.ndim(); ++i) {
    if (i == axis_) {
      continue;
    }
    shape.push_back(idx.shape(i));
    update_strides.push_back(upd.strides()[i]);
    index_strides.push_back(idx.strides()[i]);
    if (i < axis_) {
      pre_shape.push_back(idx.shape(i));
    }
  }
  uint64_t ndim = out.ndim() - 1;
  if (shape.empty()) {
    shape.push_back(1);
    update_strides.push_back(0);
    index_strides.push_back(0);
  }
  const auto full_shape = shape;
  int32_t axis = axis_, axis_size = out.shape(axis_);
  uint64_t update_step = upd.strides()[axis_], index_step = idx.strides()[axis_];
  auto* const update_data = static_cast<uint8_t*>(const_cast<void*>(upd.data<void>()));
  auto* const index_data = static_cast<uint8_t*>(const_cast<void*>(idx.data<void>()));
  CrosstlMlxBuffer buffers[] = {
      {"upd", storage_type(upd.dtype()), update_data, update_span, 0},
      {"indices", storage_type(idx.dtype()), index_data, index_span, 0},
      {"out", storage_type(out.dtype()), out.data<void>(), out.size(), 1},
      {"shape", "int32", shape.data(), shape.size(), 0},
      {"upd_strides", "int64", update_strides.data(), update_strides.size(), 0},
      {"idx_strides", "int64", index_strides.data(), index_strides.size(), 0},
      {"ndim", "uint64", &ndim, 1, 0},
      {"axis", "int32", &axis, 1, 0},
      {"out_axis_size", "int32", &axis_size, 1, 0},
      {"upd_ax_stride", "uint64", &update_step, 1, 0},
      {"idx_ax_stride", "uint64", &index_step, 1, 0},
  };
  // A float32 sum that some order would round runs as index-order launches.
  const bool ordered = reduce_type_ == Sum && out.dtype() == float32 &&
      axis_sum_order_dependent(idx, upd, out, axis_);
  auto tiles = axis_launch_tiles(pre_shape, idx.shape(axis_), ordered);
  if (ordered) {
    tiles = axis_sum_runs(tiles, idx, out, axis_, after);
  }
  for (const auto& tile : tiles) {
    std::copy(tile.pre_shape.begin(), tile.pre_shape.end(), shape.begin());
    // Each view starts at the tile's first element and spans only the tile.
    uint64_t update_offset = tile.first_row * update_step;
    uint64_t index_offset = tile.first_row * index_step;
    uint64_t update_extent = 1 + (tile.rows - 1) * update_step;
    uint64_t index_extent = 1 + (tile.rows - 1) * index_step;
    for (size_t i = 0; i < full_shape.size(); ++i) {
      if (i < tile.pre_start.size()) {
        update_offset += tile.pre_start[i] * uint64_t(update_strides[i]);
        index_offset += tile.pre_start[i] * uint64_t(index_strides[i]);
      }
      update_extent += uint64_t(shape[i] - 1) * uint64_t(update_strides[i]);
      index_extent += uint64_t(shape[i] - 1) * uint64_t(index_strides[i]);
    }
    if (update_offset > update_span || update_extent > update_span - update_offset ||
        index_offset > index_span || index_extent > index_span - index_offset) {
      throw std::invalid_argument("CrossTL axis scatter tile exceeds its input allocation.");
    }
    const uint64_t outputs = tile.pre_count * after * uint64_t(axis_size);
    buffers[0].data = update_data + update_offset * upd.itemsize();
    buffers[0].count = update_extent;
    buffers[1].data = index_data + index_offset * idx.itemsize();
    buffers[1].count = index_extent;
    buffers[2].data = out.data<uint8_t>() + tile.first_pre * after * axis_size * out.itemsize();
    buffers[2].count = outputs;
    const CrosstlMlxLaunch launch{
        {static_cast<uint32_t>(after), static_cast<uint32_t>(tile.rows),
         static_cast<uint32_t>(tile.pre_count)}, {1, 1, 1}};
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers, 11, outputs, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native axis scatter failed: ") + error);
    }
  }
}

// Ports indexing.cpp MaskedScatter::eval_gpu. The output starts as a copy of
// the destination, an exclusive Sum scan of the mask, flattened to one row per
// batch, numbers its set elements in uint32, and the translated
// masked_assign kernel MLX generates for the value type and the source's row
// contiguity writes source element ``batch * src_batch_size + offset`` to each
// set element. The kernel reads only thread_position_in_grid, so it runs as
// one-thread workgroups over indexing.cpp's exact grid of one thread per mask
// element: Direct3D and OpenGL cannot clip the partial workgroup that
// dispatch_threads leaves.
void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 3) {
    throw std::invalid_argument(
        "CrossTL masked scatter requires a destination, a mask and a source.");
  }
  const auto& dst = inputs[0];
  const auto& mask = inputs[1];
  const auto& src = inputs[2];
  const char* value_type = cast_type(out.dtype());
  // MLX broadcasts the mask to the destination and gives every operand a
  // leading batch axis, so each mask element names one output element.
  if (!value_type || dst.dtype() != out.dtype() || src.dtype() != out.dtype() ||
      mask.dtype() != bool_ || dst.shape() != out.shape() || mask.size() != out.size() ||
      mask.ndim() < 1 || mask.ndim() > out.ndim() || src.ndim() < 1 || src.ndim() > 64 ||
      out.size() > max_masked_scatter_elements || src.size() > max_scatter_storage) {
    throw std::invalid_argument(
        "CrossTL masked scatter requires a Boolean mask over the bounded destination "
        "and a source of its type.");
  }
  dispatch_copy(dst, out);
  const uint64_t total = mask.size();
  if (total == 0) {
    return;
  }
  if (src.size() == 0 || mask.shape(0) != src.shape(0)) {
    throw std::invalid_argument(
        "CrossTL masked scatter requires one source batch per mask batch.");
  }
  // flatten_in_eval(mask, 1, -1) of the row-contiguous mask.
  const array dense = dense_input(mask);
  const int batches = mask.shape(0);
  array mask_flat({batches, static_cast<int>(total / batches)}, bool_, nullptr, {});
  mask_flat.copy_shared_buffer(
      dense, {static_cast<int64_t>(total / batches), 1}, dense.flags(), dense.data_size());
  array scatter_offsets(mask_flat.shape(), uint32, nullptr, {});
  dispatch_scan(mask_flat, scatter_offsets, Scan::Sum, 1, false, false);
  const bool contiguous = src.flags().row_contiguous;
  const std::string entry = std::string("masked_assign_") + value_type + "_" +
      (contiguous ? "true" : "false");
  require_entry(entry);
  const uint64_t span = gather_span(src, max_scatter_storage);
  std::vector<int32_t> src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> src_strides(src.strides().begin(), src.strides().end());
  int32_t src_ndim = static_cast<int32_t>(src.ndim());
  int64_t src_batch_size = src.size() / src.shape(0);
  int64_t mask_batch_size = mask_flat.size() / mask.shape(0);
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  CrosstlMlxBuffer buffers[] = {
      {"mask", "bool_", data(mask_flat), total, 0},
      {"scatter_offsets", "uint32", data(scatter_offsets), total, 0},
      {"src", value_type, data(src), span, 0},
      {"out", value_type, out.data<void>(), out.size(), 1},
      {"src_shapes", "int32", src_shape.data(), src_shape.size(), 0},
      {"src_strides", "int64", src_strides.data(), src_strides.size(), 0},
      {"src_ndim", "int32", &src_ndim, 1, 0},
      {"src_batch_size", "int64", &src_batch_size, 1, 0},
      {"mask_batch_size", "int64", &mask_batch_size, 1, 0},
  };
  const CrosstlMlxLaunch launch{{static_cast<uint32_t>(total), 1, 1}, {1, 1, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, 9, total, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native masked scatter failed: ") + error);
  }
}

} // namespace mlx::core
