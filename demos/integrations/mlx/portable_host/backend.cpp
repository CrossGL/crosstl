#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "mlx/allocator.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/no_gpu/crosstl_dispatch.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace {
std::atomic<CrosstlMlxDispatch> dispatch_callback{nullptr};
std::atomic<CrosstlMlxEntryAvailable> entry_available_callback{nullptr};
std::mutex registration_mutex;
constexpr uint64_t max_slice_update_elements = (uint64_t(1) << 28) - 33;

void require_runtime() {
  if (!dispatch_callback.load()) {
    throw std::runtime_error("CrossTL host runtime is not registered.");
  }
}

void require_entry(const std::string& entry) {
  require_runtime();
  const auto available = entry_available_callback.load();
  if (!available || available(entry.c_str()) != 1) {
    throw std::invalid_argument("No translated package for " + entry);
  }
}

CrosstlMlxLaunch elementwise_launch(uint64_t count) {
  if (count == 0 || count > 65535) {
    throw std::invalid_argument("CrossTL elementwise launch exceeds its bounds.");
  }
  return {{static_cast<uint32_t>(count), 1, 1}, {1, 1, 1}};
}

template <typename T>
T scalar_cast(double value) {
  if constexpr (std::is_integral_v<T>) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Integral Arange parameters must be finite.");
    }
    using Unsigned = std::make_unsigned_t<T>;
    const long double modulus =
        std::ldexp(1.0L, std::numeric_limits<Unsigned>::digits);
    const long double residue = std::fmod(
        std::fabs(std::trunc(static_cast<long double>(value))), modulus);
    Unsigned bits = static_cast<Unsigned>(residue);
    if (value < 0) {
      bits = Unsigned(0) - bits;
    }
    return std::bit_cast<T>(bits);
  } else {
    return static_cast<T>(value);
  }
}

template <typename T>
void dispatch_arange(
    mlx::core::array& out,
    double start_value,
    double step_value,
    const char* entry,
    const char* dtype) {
  T start = scalar_cast<T>(start_value);
  T next = scalar_cast<T>(start_value + step_value);
  T step;
  if constexpr (std::is_integral_v<T>) {
    using Unsigned = std::make_unsigned_t<T>;
    step = std::bit_cast<T>(
        static_cast<Unsigned>(next) - static_cast<Unsigned>(start));
  } else {
    step = next - start;
  }
  CrosstlMlxBuffer buffers[] = {
      {"start", dtype, &start, 1, 0},
      {"step", dtype, &step, 1, 0},
      {"out", dtype, out.data<T>(), out.size(), 1},
  };
  char error[2048] = {};
  const auto launch = elementwise_launch(out.size());
  int status = dispatch_callback.load()(
      entry, buffers, 3, out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(
        std::string("CrossTL native dispatch failed: ") + error);
  }
}

void dispatch_copy(const mlx::core::array& in, mlx::core::array& out);
const char* storage_type(mlx::core::Dtype dtype);

void dispatch_unary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation) {
  require_runtime();
  const bool logical = std::string(operation) == "LogicalNot";
  const bool invert = std::string(operation) == "BitwiseInvert";
  const bool absolute = std::string(operation) == "Abs" && inputs.size() == 1 &&
      (inputs[0].dtype() == mlx::core::float16 ||
       inputs[0].dtype() == mlx::core::bfloat16 ||
       inputs[0].dtype() == mlx::core::int32 ||
       inputs[0].dtype() == mlx::core::uint32 ||
       inputs[0].dtype() == mlx::core::int64 ||
       inputs[0].dtype() == mlx::core::uint64 ||
       inputs[0].dtype() == mlx::core::bool_);
  if (invert &&
      (inputs.size() != 1 ||
       (inputs[0].dtype() != mlx::core::int32 &&
        inputs[0].dtype() != mlx::core::uint32) ||
       out.dtype() != inputs[0].dtype())) {
    throw std::invalid_argument(
        "CrossTL BitwiseInvert requires matching int32 or uint32 arrays.");
  }
  const auto type = (invert || absolute) ? inputs[0].dtype()
      : logical                         ? mlx::core::bool_
                                        : mlx::core::float32;
  const char* dtype = type == mlx::core::float16 ? "float16" :
      type == mlx::core::bfloat16 ? "bfloat16" : storage_type(type);
  if (inputs.size() != 1 || inputs[0].dtype() != type || out.dtype() != type) {
    throw std::invalid_argument(
        "CrossTL unary dispatch requires float32 arrays, or bool for LogicalNot.");
  }
  auto in = inputs[0];
  if (in.shape() != out.shape()) {
    throw std::invalid_argument("CrossTL unary input must match output shape.");
  }
  if (in.size() == 0) {
    mlx::core::set_unary_output_data(in, out);
    return;
  }
  if (!in.flags().contiguous) {
    mlx::core::array dense(in.shape(), in.dtype(), nullptr, {});
    dispatch_copy(in, dense);
    in = std::move(dense);
  }
  if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
      in.data_size() >
          (in.buffer_size() - uint64_t(in.offset())) / in.itemsize()) {
    throw std::invalid_argument("CrossTL unary input exceeds its allocation.");
  }
  std::string entry = std::string("v_") + operation + dtype + dtype;
  require_entry(entry);
  mlx::core::set_unary_output_data(in, out);
  // Contiguous broadcasts and column-major arrays retain their stored layout.
  for (uint64_t first = 0; first < in.data_size(); first += 65535) {
    uint32_t size = static_cast<uint32_t>(
        std::min<uint64_t>(65535, in.data_size() - first));
    CrosstlMlxBuffer buffers[] = {
        {"in", dtype, in.data<char>() + first * in.itemsize(), size, 0},
        {"out", dtype, out.data<char>() + first * out.itemsize(), size, 1},
        {"size", "uint32", &size, 1, 0},
    };
    char error[2048] = {};
    const auto launch = elementwise_launch(size);
    int status = dispatch_callback.load()(
        entry.c_str(), buffers, 3, size, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(
          std::string("CrossTL native dispatch failed: ") + error);
    }
  }
}

// One exact MLX copy launch: a sub-shape of the copy, the element offset of
// its origin from the source view, and its destination offset.
struct CopyRegion {
  std::vector<int32_t> shape;
  int64_t source_origin;
  int64_t destination_offset;
};

// The copy kernel guards only its X axis, so a copy whose launch would exceed
// the portable 65535 groups on an axis runs as exact sub-copies along the
// array axes, in ascending order: the last axis in chunks of 65535 element
// pairs, then rows, then the leading axes that fold into Z.
void append_copy_regions(
    CopyRegion region,
    const std::vector<int64_t>& src_strides,
    const std::vector<int64_t>& dst_strides,
    std::vector<CopyRegion>& regions) {
  const size_t ndim = region.shape.size();
  size_t axis = ndim - 1;
  int64_t chunk = 2 * 65535;
  if ((uint64_t(region.shape[ndim - 1]) + 1) / 2 <= 65535) {
    axis = ndim - 2;
    chunk = 65535;
    if (uint64_t(region.shape[ndim - 2]) <= 65535) {
      uint64_t slices = 1;
      for (size_t leading = 0; leading + 2 < ndim; ++leading) {
        slices *= uint64_t(region.shape[leading]);
      }
      if (slices <= 65535) {
        regions.push_back(std::move(region));
        return;
      }
      axis = 0;
      while (region.shape[axis] == 1) {
        ++axis;
      }
      uint64_t inner = 1;
      for (size_t next = axis + 1; next + 2 < ndim; ++next) {
        inner *= uint64_t(region.shape[next]);
      }
      chunk = inner <= 65535 ? int64_t(65535 / inner) : 1;
    }
  }
  const int64_t extent = region.shape[axis];
  for (int64_t first = 0; first < extent; first += chunk) {
    CopyRegion part = region;
    part.shape[axis] = static_cast<int32_t>(std::min(chunk, extent - first));
    part.source_origin += first * src_strides[axis];
    part.destination_offset += first * dst_strides[axis];
    append_copy_regions(std::move(part), src_strides, dst_strides, regions);
  }
}

void launch_copy_region(
    const mlx::core::array& in,
    mlx::core::array& out,
    const CopyRegion& region,
    std::vector<int64_t> src_strides,
    std::vector<int64_t> dst_strides,
    const char* dtype,
    uint32_t destination_mode) {
  std::vector<int32_t> shape = region.shape;
  int64_t low = region.source_origin, high = region.source_origin;
  uint64_t count = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const int64_t extent = (int64_t(shape[axis]) - 1) * src_strides[axis];
    low += std::min<int64_t>(extent, 0);
    high += std::max<int64_t>(extent, 0);
    count *= uint64_t(shape[axis]);
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int64_t src_offset = region.source_origin - low;
  int64_t dst_offset = region.destination_offset;
  CrosstlMlxBuffer buffers[] = {
      {"src", dtype, const_cast<uint8_t*>(in.data<uint8_t>() + low * in.itemsize()),
       uint64_t(high - low + 1), 0},
      {"dst", dtype, out.data<void>(), out.size(), destination_mode},
      {"src_shape", "int32", shape.data(), uint64_t(ndim), 0},
      {"src_strides", "int64", src_strides.data(), uint64_t(ndim), 0},
      {"dst_strides", "int64", dst_strides.data(), uint64_t(ndim), 0},
      {"ndim", "int32", &ndim, 1, 0},
      {"src_offset", "int64", &src_offset, 1, 0},
      {"dst_offset", "int64", &dst_offset, 1, 0},
  };
  uint64_t slices = 1;
  for (size_t axis = 0; axis + 2 < shape.size(); ++axis) {
    slices *= uint64_t(shape[axis]);
  }
  const uint64_t columns = (uint64_t(shape.back()) + 1) / 2;
  const uint64_t rows = shape[shape.size() - 2];
  if (columns > 65535 || rows > 65535 || slices > 65535) {
    throw std::logic_error("CrossTL copy region exceeds 65535 groups per axis.");
  }
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(columns), static_cast<uint32_t>(rows),
       static_cast<uint32_t>(slices)},
      {1, 1, 1}};
  const std::string entry = std::string("ggn2_dynamic_copy") + dtype + dtype;
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, 8, count, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native copy failed: ") + error);
  }
}

void dispatch_copy_into(
    const mlx::core::array& in,
    mlx::core::array& out,
    std::vector<int64_t> dst_strides,
    int64_t dst_offset,
    bool preserve) {
  require_runtime();
  if (in.dtype() != out.dtype() ||
      (in.dtype() != mlx::core::float32 && in.dtype() != mlx::core::int32 &&
       in.dtype() != mlx::core::float16 &&
       in.dtype() != mlx::core::bfloat16 &&
       in.dtype() != mlx::core::uint32 && in.dtype() != mlx::core::bool_ &&
       in.dtype() != mlx::core::int64 && in.dtype() != mlx::core::uint64)) {
    throw std::invalid_argument(
        "CrossTL copying layouts require matching float16, bfloat16, float32, int32, uint32, int64, uint64 or bool arrays.");
  }
  constexpr int64_t max_index = std::numeric_limits<int32_t>::max();
  if (in.size() > max_index || in.ndim() > 64 ||
      dst_strides.size() != in.ndim() ||
      out.size() > max_index) {
    throw std::invalid_argument(
        "CrossTL copy requires signed 32-bit element counts and at most 64 axes.");
  }
  if (in.size() == 0) {
    return;
  }
  std::vector<int32_t> shape(in.shape().begin(), in.shape().end());
  std::vector<int64_t> src_strides(in.strides().begin(), in.strides().end());
  while (shape.size() < 2) {
    shape.insert(shape.begin(), 1);
    src_strides.insert(src_strides.begin(), 0);
    dst_strides.insert(dst_strides.begin(), 0);
  }
  int64_t low = 0, high = 0, source_extent = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] == 1) {
      src_strides[axis] = 0;
    }
    if (src_strides[axis] < -max_index || src_strides[axis] > max_index) {
      throw std::invalid_argument("CrossTL copy source stride exceeds signed index bounds.");
    }
    const int64_t extent = (int64_t(shape[axis]) - 1) * src_strides[axis];
    if (std::abs(extent) > max_index - 1 - source_extent) {
      throw std::invalid_argument("CrossTL copy source span exceeds signed index bounds.");
    }
    source_extent += std::abs(extent);
    low += std::min<int64_t>(extent, 0);
    high += std::max<int64_t>(extent, 0);
  }
  const int64_t item_size = in.itemsize();
  if (in.offset() < 0 || in.offset() % item_size != 0) {
    throw std::invalid_argument("CrossTL copy source span exceeds its bounds.");
  }
  const uint64_t origin = in.offset() / item_size;
  const uint64_t capacity = in.buffer_size() / item_size;
  if (origin >= capacity || uint64_t(-low) > origin ||
      uint64_t(high) >= capacity - origin) {
    throw std::invalid_argument("CrossTL copy source view exceeds its allocation.");
  }
  int64_t destination_low = dst_offset, destination_high = dst_offset;
  int64_t destination_extent = 0;
  if (dst_offset < 0 || dst_offset > max_index) {
    throw std::invalid_argument("CrossTL copy destination offset exceeds signed index bounds.");
  }
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (dst_strides[axis] < -int64_t(std::numeric_limits<int32_t>::max()) ||
        dst_strides[axis] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("CrossTL copy destination stride is invalid.");
    }
    const int64_t extent = (int64_t(shape[axis]) - 1) * dst_strides[axis];
    if (std::abs(extent) > max_index - 1 - destination_extent) {
      throw std::invalid_argument("CrossTL copy destination span exceeds signed index bounds.");
    }
    destination_extent += std::abs(extent);
    destination_low += std::min<int64_t>(extent, 0);
    destination_high += std::max<int64_t>(extent, 0);
  }
  if (dst_offset < 0 || destination_low < 0 || destination_high >= out.size() ||
      out.offset() < 0 || uint64_t(out.offset()) > out.buffer_size() ||
      out.nbytes() > out.buffer_size() - uint64_t(out.offset())) {
    throw std::invalid_argument("CrossTL copy destination exceeds its allocation.");
  }
  // Copy storage words to preserve NaN payloads, subnormals and signed zero.
  const bool boolean = in.dtype() == mlx::core::bool_;
  const bool wide = in.dtype() == mlx::core::int64 || in.dtype() == mlx::core::uint64;
  const char* dtype = in.dtype() == mlx::core::float16 ? "float16" :
      in.dtype() == mlx::core::bfloat16 ? "bfloat16" :
      wide ? storage_type(in.dtype()) : boolean ? "bool_" : "uint32";
  std::vector<CopyRegion> regions;
  append_copy_regions({shape, 0, dst_offset}, src_strides, dst_strides, regions);
  for (size_t index = 0; index < regions.size(); ++index) {
    // Later regions keep the elements that earlier regions already wrote.
    launch_copy_region(
        in, out, regions[index], src_strides, dst_strides, dtype,
        preserve || index > 0 ? CROSTL_MLX_BUFFER_INOUT : uint32_t(1));
  }
}

void dispatch_copy(const mlx::core::array& in, mlx::core::array& out) {
  if (in.size() != out.size()) {
    throw std::invalid_argument("CrossTL copy requires equal logical sizes.");
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  std::vector<int64_t> strides(in.ndim());
  int64_t stride = 1;
  for (size_t axis = in.ndim(); axis-- > 0;) {
    strides[axis] = stride;
    stride *= in.shape(axis);
  }
  dispatch_copy_into(in, out, std::move(strides), 0, false);
}

void reshape_view(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out) {
  require_runtime();
  if (inputs.size() != 1) {
    throw std::invalid_argument("CrossTL reshape requires one input.");
  }
  const auto& in = inputs[0];
  auto [copy_required, strides] = mlx::core::prepare_reshape(in, out);
  if (copy_required) {
    dispatch_copy(in, out);
    return;
  }
  mlx::core::shared_buffer_reshape(in, strides, out);
}

mlx::core::array dense_input(const mlx::core::array& in) {
  if (in.flags().row_contiguous && in.data_size() == in.size()) {
    const uint64_t bytes = in.nbytes();
    if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
        bytes > in.buffer_size() - uint64_t(in.offset())) {
      throw std::invalid_argument("CrossTL dense input exceeds its allocation.");
    }
    return in;
  }
  mlx::core::array dense(in.shape(), in.dtype(), nullptr, {});
  dispatch_copy(in, dense);
  return dense;
}

uint64_t gather_span(const mlx::core::array& in, uint64_t maximum = 65535) {
  if (in.ndim() > 64 || in.size() == 0 || in.size() > maximum) {
    throw std::invalid_argument("CrossTL gather input shape exceeds its bounds.");
  }
  uint64_t span = 1;
  for (size_t axis = 0; axis < in.ndim(); ++axis) {
    const auto stride = in.strides()[axis];
    const auto extent = in.shape(axis);
    if (stride < 0 || uint64_t(stride) > maximum || extent < 1 ||
        (stride && uint64_t(extent - 1) > (maximum - span) / uint64_t(stride))) {
      throw std::invalid_argument("CrossTL gather requires bounded nonnegative storage strides.");
    }
    span += uint64_t(extent - 1) * uint64_t(stride);
  }
  if (in.offset() < 0 ||
      in.offset() % in.itemsize() != 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      uint64_t(span) * in.itemsize() > in.buffer_size() - uint64_t(in.offset())) {
    throw std::invalid_argument("CrossTL gather input view exceeds its allocation.");
  }
  return static_cast<uint64_t>(span);
}

mlx::core::array gather_input(const mlx::core::array& in) {
  if (std::any_of(in.strides().begin(), in.strides().end(),
                  [](auto stride) { return stride < 0; })) {
    return dense_input(in);
  }
  return in;
}

void dispatch_slice_update(
    const mlx::core::array& updates,
    mlx::core::array& out,
    const std::string& entry,
    std::vector<int64_t> output_strides,
    int64_t output_offset) {
  if (updates.size() == 0) {
    return;
  }
  if (updates.size() > max_slice_update_elements || out.size() > max_slice_update_elements ||
      out.offset() < 0 || uint64_t(out.offset()) > out.buffer_size() ||
      out.nbytes() > out.buffer_size() - uint64_t(out.offset())) {
    throw std::invalid_argument("CrossTL slice update storage exceeds its bounds.");
  }
  auto dense = dense_input(updates);
  std::vector<int32_t> shape(dense.shape().begin(), dense.shape().end());
  if (shape.empty()) {
    shape = {1};
    output_strides = {1};
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  const char* dtype = storage_type(out.dtype());
  // Each rectangular slab is dense in update storage and retains destination strides.
  size_t split_axis = 0;
  uint64_t suffix = dense.size() / shape.front();
  while (suffix > 65535) {
    suffix /= shape[++split_axis];
  }
  const uint64_t axis_limit = 65535 / suffix;
  for (uint64_t first = 0; first < dense.size();) {
    auto tile_shape = shape;
    std::vector<uint64_t> coordinates(ndim);
    uint64_t remaining = first;
    int64_t tile_offset = output_offset;
    for (size_t axis = ndim; axis-- > 0;) {
      coordinates[axis] = remaining % shape[axis];
      remaining /= shape[axis];
      tile_offset += int64_t(coordinates[axis]) * output_strides[axis];
    }
    for (size_t axis = 0; axis < split_axis; ++axis) {
      tile_shape[axis] = 1;
    }
    const uint64_t extent = std::min<uint64_t>(
        axis_limit, shape[split_axis] - coordinates[split_axis]);
    tile_shape[split_axis] = static_cast<int32_t>(extent);
    int64_t size = extent * suffix;
    std::vector<int64_t> strides(ndim);
    int64_t stride = 1;
    for (size_t axis = ndim; axis-- > 0;) {
      strides[axis] = stride;
      stride *= tile_shape[axis];
    }
    CrosstlMlxBuffer buffers[] = {
        {"updates", dtype, dense.data<char>() + first * dense.itemsize(), uint64_t(size), 0},
        {"out", dtype, out.data<void>(), out.size(), CROSTL_MLX_BUFFER_INOUT},
        {"update_shape", "int32", tile_shape.data(), uint64_t(ndim), 0},
        {"update_strides", "int64", strides.data(), uint64_t(ndim), 0},
        {"update_ndim", "int32", &ndim, 1, 0},
        {"update_size", "int64", &size, 1, 0},
        {"output_strides", "int64", output_strides.data(), uint64_t(ndim), 0},
        {"output_offset", "int64", &tile_offset, 1, 0},
    };
    char error[2048] = {};
    const auto launch = elementwise_launch(size);
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers, 8, size, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native slice update failed: ") + error);
    }
    first += size;
  }
}

uint32_t all_reduce_rows(const mlx::core::array& in) {
  if (in.size() <= 4096) {
    return 1;
  }
  return in.size() <= (uint64_t(1) << 26) / in.itemsize() ? 128 : 4096;
}

void dispatch_all_reduce(
    const mlx::core::array& in,
    mlx::core::array& out,
    const std::string& operation,
    const char* dtype) {
  uint64_t size = in.size();
  const uint32_t rows = all_reduce_rows(in);
  uint64_t row_size = (size + rows - 1) / rows;
  const uint32_t width = static_cast<uint32_t>(
      std::min<uint64_t>(1024, ((row_size + 127) / 128) * 32));
  if (size == 0 || size > 2147483647 || out.size() != rows ||
      in.data_size() != size || in.offset() < 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      size > (in.buffer_size() - uint64_t(in.offset())) / in.itemsize()) {
    throw std::invalid_argument("CrossTL reduction storage does not match its pass.");
  }
  out.set_data(mlx::core::allocator::malloc(std::max<size_t>(out.nbytes(), 4)));
  std::string entry = "all_reduce_" + operation + dtype;
  CrosstlMlxBuffer buffers[] = {
      {"in", dtype, const_cast<void*>(in.data<void>()), size, 0},
      {"out", dtype, out.data<void>(), rows, 1},
      {"in_size", "uint64", &size, 1, 0},
      {"row_size", "uint64", &row_size, 1, 0},
  };
  const CrosstlMlxLaunch launch{{1, rows, 1}, {width, 1, 1}};
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, 4, size, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native reduction failed: ") + error);
  }
}

void dispatch_row_reduce(
    const mlx::core::array& in,
    mlx::core::array& out,
    const mlx::core::ReductionPlan& plan,
    const std::vector<int>& axes,
    const std::string& operation,
    const char* dtype) {
  using namespace mlx::core;
  int64_t row_size = plan.shape.back();
  const bool small = row_size <= 64;
  uint32_t width = row_size <= 512 ? 32 : row_size <= 1024 ? 128 :
      static_cast<uint32_t>(std::min<int64_t>(1024, ((row_size + 127) / 128) * 32));
  auto reduce_shape = plan.shape;
  auto reduce_strides = plan.strides;
  reduce_shape.pop_back();
  reduce_strides.pop_back();
  int32_t reduce_ndim = static_cast<int32_t>(reduce_shape.size());
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
  int32_t ndim = static_cast<int32_t>(shape.size());
  int64_t non_rows = 1;
  for (auto size : reduce_shape) {
    non_rows *= size;
  }
  int64_t span = row_size;
  auto accumulate_span = [&](const auto& sizes, const auto& steps) {
    for (size_t axis = 0; axis < sizes.size(); ++axis) {
      if (sizes[axis] < 1 || sizes[axis] > 65535 || steps[axis] < 0 || steps[axis] > 65535) {
        throw std::invalid_argument("CrossTL row reduction shape or stride exceeds its bounds.");
      }
      span += (int64_t(sizes[axis]) - 1) * steps[axis];
    }
  };
  accumulate_span(shape, strides);
  accumulate_span(reduce_shape, reduce_strides);
  if (ndim > 64 || reduce_ndim > 64 || span < 1 || span > 65535 ||
      in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
      uint64_t(span) * in.itemsize() > in.buffer_size() - uint64_t(in.offset()) ||
      uint64_t(row_size * non_rows) * out.size() != in.size()) {
    throw std::invalid_argument("CrossTL row reduction source view exceeds its bounds.");
  }
  const bool simple = !small && plan.type == ContiguousReduce && reduce_ndim == 0 &&
      in.size() / row_size >= 32;
  out.set_data(allocator::malloc(std::max<size_t>(out.nbytes(), 4)));
  char error[2048] = {};
  int status;
  if (simple) {
    uint64_t reduction_size = row_size;
    int64_t out_size = out.size();
    std::string entry = "row_reduce_simple_" + operation + dtype;
    CrosstlMlxBuffer buffers[] = {
        {"in", dtype, const_cast<void*>(in.data<void>()), uint64_t(span), 0},
        {"out", dtype, out.data<void>(), out.size(), 1},
        {"reduction_size", "uint64", &reduction_size, 1, 0},
        {"out_size", "int64", &out_size, 1, 0},
    };
    const CrosstlMlxLaunch launch{{1, uint32_t((out_size + 3) / 4), 1}, {width, 1, 1}};
    status = dispatch_callback.load()(entry.c_str(), buffers, 4, in.size(), &launch, error, sizeof(error));
  } else {
    const int dimension = reduce_ndim <= 1 ? 1 : reduce_ndim == 2 ? 2 : 5;
    std::string entry = std::string(small ? "row_reduce_small_" : "row_reduce_looped_") +
        std::to_string(dimension) + "_reduce_" + operation + dtype;
    if (ndim == 0) { shape.push_back(0); strides.push_back(0); }
    if (reduce_ndim == 0) { reduce_shape.push_back(0); reduce_strides.push_back(0); }
    CrosstlMlxBuffer buffers[] = {
        {"in", dtype, const_cast<void*>(in.data<void>()), uint64_t(span), 0},
        {"out", dtype, out.data<void>(), out.size(), 1},
        {"row_size", "int64", &row_size, 1, 0},
        {"non_row_reductions", "int64", &non_rows, 1, 0},
        {"shape", "int32", shape.data(), shape.size(), 0},
        {"strides", "int64", strides.data(), strides.size(), 0},
        {"ndim", "int32", &ndim, 1, 0},
        {"reduce_shape", "int32", reduce_shape.data(), reduce_shape.size(), 0},
        {"reduce_strides", "int64", reduce_strides.data(), reduce_strides.size(), 0},
        {"reduce_ndim", "int32", &reduce_ndim, 1, 0},
    };
    CrosstlMlxLaunch launch{{1, uint32_t(out.size()), 1}, {width, 1, 1}, {0, 0, 0}};
    if (small) {
      // The current bounded output fits upstream get_2d_grid_dims' first axis.
      const uint32_t rows = static_cast<uint32_t>(out.size());
      const bool scalar = (non_rows < 32 && row_size <= 8) || non_rows <= 8;
      if (scalar) {
        width = std::min<uint32_t>(rows, 1024);
        launch = {{(rows + width - 1) / width, 1, 1}, {width, 1, 1}, {rows, 1, 1}};
      } else {
        launch = {{1, rows, 1}, {32, 1, 1}, {32, rows, 1}};
      }
    }
    status = dispatch_callback.load()(entry.c_str(), buffers, 10, in.size(), &launch, error, sizeof(error));
  }
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native row reduction failed: ") + error);
  }
}

uint64_t column_shape_size(const mlx::core::Shape& shape) {
  uint64_t size = 1;
  for (auto dimension : shape) {
    if (dimension < 1 || uint64_t(dimension) > 65535 / size) {
      throw std::invalid_argument("CrossTL column reduction shape exceeds its bounds.");
    }
    size *= dimension;
  }
  return size;
}

enum class ColumnPass { Small, Looped, TwoPass };

// The largest input or output a column pass binds, as for matrix products; far
// below the INT32_MAX at which reduce.cpp selects its 64-bit index kernels.
constexpr uint64_t max_column_elements = (uint64_t(1) << 28) - 1;

void dispatch_column_pass(
    const mlx::core::array& in,
    mlx::core::array& out,
    mlx::core::Shape shape,
    mlx::core::Strides strides,
    mlx::core::Shape reduce_shape,
    mlx::core::Strides reduce_strides,
    ColumnPass pass,
    const std::string& operation,
    const char* dtype) {
  const bool two_pass = pass == ColumnPass::TwoPass;
  if (shape.size() > 64 || shape.size() != strides.size() ||
      reduce_shape.empty() || reduce_shape.size() > 64 ||
      reduce_shape.size() != reduce_strides.size()) {
    throw std::invalid_argument("CrossTL column reduction ranks do not match.");
  }
  uint64_t total = column_shape_size(reduce_shape);
  uint64_t outer = column_shape_size(shape);
  uint64_t reduction_size = reduce_shape.back();
  int64_t reduction_stride = reduce_strides.back();
  uint64_t non_columns = total / reduction_size;
  if (reduction_stride < 1 || uint64_t(reduction_stride) > max_column_elements ||
      outer * reduction_stride * total != in.size() ||
      outer * reduction_stride * (two_pass ? 32 : 1) != out.size() ||
      in.size() > max_column_elements || out.size() > max_column_elements ||
      outer * (two_pass ? 32 : 1) > 65535) {
    throw std::invalid_argument("CrossTL column reduction storage does not match its pass.");
  }
  uint64_t span = reduction_stride;
  auto accumulate_span = [&](const auto& sizes, const auto& steps) {
    for (size_t axis = 0; axis < sizes.size(); ++axis) {
      if (steps[axis] < 0 || uint64_t(steps[axis]) > max_column_elements) {
        throw std::invalid_argument("CrossTL column reduction stride exceeds its bounds.");
      }
      span += (uint64_t(sizes[axis]) - 1) * steps[axis];
    }
  };
  accumulate_span(shape, strides);
  accumulate_span(reduce_shape, reduce_strides);
  if (span > max_column_elements || in.offset() < 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      span * in.itemsize() > in.buffer_size() - uint64_t(in.offset())) {
    throw std::invalid_argument("CrossTL column reduction source view exceeds its bounds.");
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int32_t reduce_ndim = static_cast<int32_t>(reduce_shape.size());
  const int dimension = reduce_ndim == 1 ? 1 : reduce_ndim == 2 ? 2 : 5;
  std::string entry = pass == ColumnPass::Small
      ? "col_reduce_small_" + std::to_string(dimension) + "_reduce_" + operation + dtype
      : std::string(two_pass ? "col_reduce_2pass_" : "col_reduce_looped_") +
          std::to_string(dimension) + "_32_32_reduce_" + operation + dtype;
  if (ndim == 0) { shape.push_back(0); strides.push_back(0); }
  out.set_data(mlx::core::allocator::malloc(std::max<size_t>(out.nbytes(), 4)));
  CrosstlMlxBuffer buffers[] = {
      {"in", dtype, const_cast<void*>(in.data<void>()), span, 0},
      {"out", dtype, out.data<void>(), out.size(), 1},
      {"reduction_size", "uint64", &reduction_size, 1, 0},
      {"reduction_stride", "int64", &reduction_stride, 1, 0},
      {"shape", "int32", shape.data(), shape.size(), 0},
      {"strides", "int64", strides.data(), strides.size(), 0},
      {"ndim", "int32", &ndim, 1, 0},
      {"reduce_shape", "int32", reduce_shape.data(), reduce_shape.size(), 0},
      {"reduce_strides", "int64", reduce_strides.data(), reduce_strides.size(), 0},
      {"reduce_ndim", "int32", &reduce_ndim, 1, 0},
      {"non_col_reductions", "uint64", &non_columns, 1, 0},
      {"out_size", "uint64", &outer, 1, 0},
  };
  // reduce.cpp strided_reduce_small: lanes of four columns, at most eight rows
  // (the kernels allow 1024 threads); the looped passes use 32 x 32 tiles.
  const uint64_t blocks = (reduction_stride + 3) / 4;
  const uint32_t lanes = uint32_t(std::min<uint64_t>(blocks, 32));
  // Each pass covers the reduction stride along the launch's first axis.
  const uint64_t column_groups = pass == ColumnPass::Small
      ? (blocks + lanes - 1) / lanes
      : (uint64_t(reduction_stride) + 31) / 32;
  if (column_groups > 65535) {
    throw std::invalid_argument("CrossTL column reduction launch exceeds 65535 workgroups.");
  }
  const CrosstlMlxLaunch launch = pass == ColumnPass::Small
      ? CrosstlMlxLaunch{
            {uint32_t((blocks + lanes - 1) / lanes), uint32_t(outer), 1},
            {lanes, uint32_t(std::min<uint64_t>({8, 1024 / lanes, total})), 1}}
      : CrosstlMlxLaunch{
            {uint32_t((reduction_stride + 31) / 32), uint32_t(outer * (two_pass ? 32 : 1)), 1},
            {256, 1, 1}};
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, two_pass ? 12 : 11, in.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native column reduction failed: ") + error);
  }
}

void dispatch_column_reduce(
    const mlx::core::array& in,
    mlx::core::array& out,
    const mlx::core::ReductionPlan& plan,
    const std::vector<int>& axes,
    const std::string& operation,
    const char* dtype) {
  using namespace mlx::core;
  if (plan.shape.empty() || plan.shape.size() != plan.strides.size()) {
    throw std::invalid_argument("CrossTL column reduction plan is empty or inconsistent.");
  }
  const uint64_t total = column_shape_size(plan.shape);
  const int64_t stride = plan.strides.back();
  if (stride < 32 && total >= 1024) {
    throw std::invalid_argument("CrossTL long-column reduction plans are not implemented.");
  }
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  // Follow upstream's shape product, including broadcast views with zero strides.
  uint64_t inner = 1;
  while (!shape.empty() && int64_t(inner) < stride) {
    if (shape.back() < 1 || shape.back() > 65535 ||
        uint64_t(shape.back()) > max_column_elements / inner) {
      throw std::invalid_argument("CrossTL column reduction inner shape exceeds its bounds.");
    }
    inner *= shape.back();
    shape.pop_back();
    strides.pop_back();
  }
  if (stride < 1 || inner != uint64_t(stride)) {
    throw std::invalid_argument("CrossTL column reduction output does not match its contiguous span.");
  }
  std::tie(shape, strides) = collapse_contiguous_dims(shape, strides);
  if (total < 32) {
    dispatch_column_pass(
        in, out, shape, strides, plan.shape, plan.strides, ColumnPass::Small, operation, dtype);
    return;
  }
  const bool two_pass = total > 256 && out.size() / 32 < 1024;
  if (!two_pass) {
    dispatch_column_pass(
        in, out, shape, strides, plan.shape, plan.strides, ColumnPass::Looped, operation, dtype);
    return;
  }
  Shape intermediate_shape{32};
  intermediate_shape.insert(intermediate_shape.end(), out.shape().begin(), out.shape().end());
  array intermediate(std::move(intermediate_shape), out.dtype(), nullptr, {});
  dispatch_column_pass(
      in, intermediate, shape, strides, plan.shape, plan.strides, ColumnPass::TwoPass, operation, dtype);
  // Consume the completed native partials, preserving the upstream second pass.
  dispatch_column_pass(
      intermediate, out, {}, {}, {32}, {int64_t(out.size())}, ColumnPass::Looped, operation, dtype);
}

const char* storage_type(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::bool_) {
    return "bool_";
  }
  if (dtype == mlx::core::float32) {
    return "float32";
  }
  if (dtype == mlx::core::int32) {
    return "int32";
  }
  if (dtype == mlx::core::uint32) {
    return "uint32";
  }
  if (dtype == mlx::core::int64) {
    return "int64";
  }
  if (dtype == mlx::core::uint64) {
    return "uint64";
  }
  return nullptr;
}

const char* cast_type(mlx::core::Dtype dtype) {
  return dtype == mlx::core::float16 ? "float16" :
      dtype == mlx::core::bfloat16 ? "bfloat16" : storage_type(dtype);
}

// Converts the first `count` elements of `in`'s data into `out`'s data.
void cast_elements(
    const mlx::core::array& in,
    mlx::core::array& out,
    uint64_t count) {
  const char* source_type = cast_type(in.dtype());
  const char* destination_type = cast_type(out.dtype());
  std::string entry = std::string("v_copy") + source_type + destination_type;
  require_entry(entry);
  // v_copy specializes one independent conversion per invocation.
  for (uint64_t first = 0; first < count;) {
    uint32_t size = static_cast<uint32_t>(
        std::min<uint64_t>(count - first, 65535));
    CrosstlMlxBuffer buffers[] = {
        {"src", source_type, const_cast<uint8_t*>(in.data<uint8_t>()) + first * in.itemsize(), size, 0},
        {"dst", destination_type, out.data<uint8_t>() + first * out.itemsize(), size, 1},
        {"size", "uint32", &size, 1, 0},
    };
    char error[2048] = {};
    const auto launch = elementwise_launch(size);
    int status = dispatch_callback.load()(
        entry.c_str(), buffers, 3, size, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native cast failed: ") + error);
    }
    first += size;
  }
}

// Returns true when `inputs` names a castable pair; same-dtype casts share.
bool prepare_cast(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL cast input must match output shape.");
  }
  if (!cast_type(inputs[0].dtype()) || !cast_type(out.dtype())) {
    throw std::invalid_argument("CrossTL casts require float16, bfloat16, float32, int32, uint32, int64, uint64 or bool arrays.");
  }
  if (inputs[0].dtype() == out.dtype()) {
    out.copy_shared_buffer(inputs[0]);
    return false;
  }
  return true;
}

// A row-contiguous conversion of `inputs[0]`, densifying strided views.
void dispatch_cast(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) {
  if (!prepare_cast(inputs, out)) {
    return;
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  cast_elements(dense_input(inputs[0]), out, out.size());
}

// gpu/primitives.cpp AsType: a contiguous input converts as a vector copy
// whose output keeps the input's strides and flags (common/copy.h
// set_copy_output_data); any other input converts through a dense copy.
void dispatch_astype(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) {
  if (!prepare_cast(inputs, out)) {
    return;
  }
  const auto& in = inputs[0];
  if (!in.flags().contiguous) {
    dispatch_cast(inputs, out);
    return;
  }
  const uint64_t elements = in.data_size();
  if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
      elements > (in.buffer_size() - uint64_t(in.offset())) / in.itemsize()) {
    throw std::invalid_argument("CrossTL cast input exceeds its allocation.");
  }
  out.set_data(
      mlx::core::allocator::malloc(elements * out.itemsize()),
      elements,
      in.strides(),
      in.flags());
  if (elements > 0) {
    cast_elements(in, out, elements);
  }
}

void dispatch_empty_reduce(mlx::core::array& out, const char* operation) {
  if (out.size() > 65535) {
    throw std::invalid_argument("CrossTL reduction initialization supports at most 65535 outputs.");
  }
  if (out.size() == 0) {
    out.set_data(mlx::core::allocator::malloc(0));
    return;
  }
  const char* dtype = storage_type(out.dtype());
  const bool logical = std::string(operation) == "and" || std::string(operation) == "or";
  if (!dtype || out.dtype() == mlx::core::int64 || out.dtype() == mlx::core::uint64 ||
      logical != (out.dtype() == mlx::core::bool_) ||
      (!logical && std::string(operation) != "sum" && std::string(operation) != "prod")) {
    throw std::invalid_argument("CrossTL reduction initialization requires float32/int32/uint32 sum/product or Boolean all/any.");
  }
  out.set_data(mlx::core::allocator::malloc(std::max<size_t>(out.nbytes(), 4)));
  std::string entry = std::string("init_reduce_") + operation + dtype;
  CrosstlMlxBuffer buffer{"out", dtype, out.data<void>(), out.size(), 1};
  // init_reduce observes only the global ID, so one-thread groups preserve its writes.
  const auto launch = elementwise_launch(out.size());
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), &buffer, 1, out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native reduction initialization failed: ") + error);
  }
}

int64_t remainder_operand(const mlx::core::array& value, size_t index) {
  switch (value.dtype()) {
    case mlx::core::int8:
      return value.data<int8_t>()[index];
    case mlx::core::int16:
      return value.data<int16_t>()[index];
    case mlx::core::uint8:
      return value.data<uint8_t>()[index];
    case mlx::core::uint16:
      return value.data<uint16_t>()[index];
    case mlx::core::int32:
      return value.data<int32_t>()[index];
    case mlx::core::uint32:
      return value.data<uint32_t>()[index];
    case mlx::core::int64:
      return value.data<int64_t>()[index];
    case mlx::core::uint64:
      // Only zero matters for unsigned divisors; keep the value nonnegative.
      return value.data<uint64_t>()[index] != 0;
    default:
      throw std::invalid_argument("CrossTL integer remainders require integer arrays.");
  }
}

int64_t remainder_minimum(mlx::core::Dtype dtype) {
  return dtype == mlx::core::int64 ? std::numeric_limits<int64_t>::min()
      : dtype == mlx::core::int16  ? std::numeric_limits<int16_t>::min()
      : dtype == mlx::core::int8   ? std::numeric_limits<int8_t>::min()
                                   : std::numeric_limits<int32_t>::min();
}

void dispatch_binary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation,
    bool comparison = false,
    bool bitwise = false) {
  require_runtime();
  if (inputs.size() != 2 || inputs[0].dtype() != inputs[1].dtype() ||
      out.dtype() != (comparison ? mlx::core::bool_ : inputs[0].dtype()) ||
      inputs[0].shape() != out.shape() ||
      inputs[1].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL binary inputs must match output shape and dtype.");
  }
  const char* dtype = inputs[0].dtype() == mlx::core::float16 ?
      "float16" : inputs[0].dtype() == mlx::core::bfloat16 ?
      "bfloat16" : storage_type(inputs[0].dtype());
  if (!dtype || (!comparison && !bitwise && inputs[0].dtype() == mlx::core::bool_) ||
      (bitwise && inputs[0].dtype() != mlx::core::int32 &&
       inputs[0].dtype() != mlx::core::uint32 && inputs[0].dtype() != mlx::core::bool_)) {
    throw std::invalid_argument("CrossTL binary dispatch requires a supported dtype.");
  }
  if (std::string(operation) == "Divide" && out.dtype() != mlx::core::float32 &&
      out.dtype() != mlx::core::float16 && out.dtype() != mlx::core::bfloat16) {
    throw std::invalid_argument("CrossTL division requires float16, bfloat16 or float32 arrays.");
  }
  if (std::string(operation) == "LogAddExp" && out.dtype() != mlx::core::float32 &&
      out.dtype() != mlx::core::float16 && out.dtype() != mlx::core::bfloat16) {
    throw std::invalid_argument("CrossTL LogAddExp requires float16, bfloat16 or float32 arrays.");
  }
  if ((std::string(operation) == "LogicalAnd" || std::string(operation) == "LogicalOr") &&
      inputs[0].dtype() != mlx::core::bool_) {
    throw std::invalid_argument("CrossTL logical operations require bool arrays.");
  }
  if (std::string(operation) == "NaNEqual" && inputs[0].dtype() != mlx::core::float32 &&
      inputs[0].dtype() != mlx::core::float16 && inputs[0].dtype() != mlx::core::bfloat16) {
    operation = "Equal";
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto a = dense_input(inputs[0]);
  auto b = dense_input(inputs[1]);
  if (std::string(operation) == "Remainder" &&
      mlx::core::issubdtype(b.dtype(), mlx::core::integer)) {
    // Integer % by zero, and INT_MIN % -1, are undefined in Metal, HLSL and GLSL.
    for (size_t i = 0; i < b.size(); ++i) {
      const int64_t divisor = remainder_operand(b, i);
      const int64_t dividend = remainder_operand(a, i);
      if (divisor == 0 || (mlx::core::issubdtype(b.dtype(), mlx::core::signedinteger) &&
                           divisor == -1 && dividend == remainder_minimum(b.dtype()))) {
        throw std::invalid_argument(
            "CrossTL integer remainders require nonzero divisors without signed overflow.");
      }
    }
  }
  if (bitwise && (std::string(operation) == "LeftShift" || std::string(operation) == "RightShift")) {
    for (size_t i = 0; i < b.size(); ++i) {
      const int64_t shift = b.dtype() == mlx::core::int32
          ? b.data<int32_t>()[i] : b.data<uint32_t>()[i];
      if (shift < 0 || shift >= 32) {
        throw std::invalid_argument("CrossTL 32-bit shifts require counts in [0, 31].");
      }
    }
  }
  std::string entry = std::string("vv_") + operation + dtype;
  require_entry(entry);
  // vv_ specializes one independent element per invocation, including comparisons.
  for (uint64_t first = 0; first < out.size();) {
    uint32_t size = static_cast<uint32_t>(
        std::min<uint64_t>(out.size() - first, 65535));
    CrosstlMlxBuffer buffers[] = {
        {"a", dtype, a.data<uint8_t>() + first * a.itemsize(), size, 0},
        {"b", dtype, b.data<uint8_t>() + first * b.itemsize(), size, 0},
        {"c", comparison ? "bool_" : dtype, out.data<uint8_t>() + first * out.itemsize(), size, 1},
        {"size", "uint32", &size, 1, 0},
    };
    char error[2048] = {};
    const auto launch = elementwise_launch(size);
    int status = dispatch_callback.load()(
        entry.c_str(), buffers, 4, size, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native binary failed: ") + error);
    }
    first += size;
  }
}

const char* divmod_type(mlx::core::Dtype dtype) {
  switch (dtype) {
    case mlx::core::float16:
      return "float16";
    case mlx::core::bfloat16:
      return "bfloat16";
    case mlx::core::float32:
      return "float32";
    case mlx::core::uint8:
      return "uint8";
    case mlx::core::uint16:
      return "uint16";
    case mlx::core::uint32:
      return "uint32";
    case mlx::core::uint64:
      return "uint64";
    case mlx::core::int8:
      return "int8";
    case mlx::core::int16:
      return "int16";
    case mlx::core::int32:
      return "int32";
    case mlx::core::int64:
      return "int64";
    default:
      return nullptr;
  }
}

int divmod_launch(
    const std::string& entry,
    CrosstlMlxBuffer* buffers,
    uint32_t count,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, count, threads, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native DivMod failed: ") + error);
  }
  return status;
}

// Ports binary_op_gpu's DivMod kernel choice onto row-contiguous outputs:
// scalar and dense operands use the ss/sv/vs/vv entries, every other layout
// the collapsed general g1/g2/g3/gn2 entries with nonnegative strides.
void dispatch_divmod(
    const std::vector<mlx::core::array>& inputs,
    std::vector<mlx::core::array>& outputs) {
  require_runtime();
  if (inputs.size() != 2 || outputs.size() != 2 ||
      inputs[0].dtype() != inputs[1].dtype() ||
      inputs[0].shape() != inputs[1].shape()) {
    throw std::invalid_argument("CrossTL DivMod requires two matching operands.");
  }
  for (const auto& out : outputs) {
    if (out.dtype() != inputs[0].dtype() || out.shape() != inputs[0].shape()) {
      throw std::invalid_argument("CrossTL DivMod outputs must match their operands.");
    }
  }
  const char* dtype = divmod_type(inputs[0].dtype());
  if (!dtype) {
    throw std::invalid_argument("CrossTL DivMod requires real integer or floating arrays.");
  }
  constexpr uint64_t max_elements = uint64_t(1) << 20;
  if (outputs[0].size() > max_elements) {
    throw std::invalid_argument("CrossTL DivMod output exceeds its bounds.");
  }
  for (auto& out : outputs) {
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  }
  auto& c = outputs[0];
  auto& d = outputs[1];
  if (c.size() == 0) {
    return;
  }
  // Negative strides would index before the bound storage; copy them densely.
  auto bounded = [](const mlx::core::array& in) {
    for (auto stride : in.strides()) {
      if (stride < 0) {
        return dense_input(in);
      }
    }
    return in;
  };
  auto a = bounded(inputs[0]);
  auto b = bounded(inputs[1]);
  if (mlx::core::issubdtype(a.dtype(), mlx::core::integer)) {
    // Integer division by zero, and MIN / -1, are undefined on every target.
    const bool is_signed = mlx::core::issubdtype(a.dtype(), mlx::core::signedinteger);
    for (int64_t i = 0; i < int64_t(c.size()); ++i) {
      const int64_t divisor =
          remainder_operand(b, mlx::core::elem_to_loc(i, b.shape(), b.strides()));
      const int64_t dividend =
          remainder_operand(a, mlx::core::elem_to_loc(i, a.shape(), a.strides()));
      if (divisor == 0 ||
          (is_signed && divisor == -1 && dividend == remainder_minimum(a.dtype()))) {
        throw std::invalid_argument(
            "CrossTL DivMod requires nonzero divisors without signed overflow.");
      }
    }
  }
  auto dense = [](const mlx::core::array& x) {
    return x.flags().row_contiguous && x.data_size() == x.size();
  };
  const bool a_scalar = a.data_size() == 1;
  const bool b_scalar = b.data_size() == 1;
  const char* vector_kernel = a_scalar && b_scalar ? "ss"
      : a_scalar && dense(b)                       ? "sv"
      : dense(a) && b_scalar                       ? "vs"
      : dense(a) && dense(b)                       ? "vv"
                                                   : nullptr;
  if (vector_kernel) {
    const std::string entry = std::string(vector_kernel) + "_DivMod" + dtype;
    require_entry(entry);
    for (uint64_t first = 0; first < c.size();) {
      uint32_t size = static_cast<uint32_t>(std::min<uint64_t>(c.size() - first, 65535));
      const uint64_t a_first = a_scalar ? 0 : first;
      const uint64_t b_first = b_scalar ? 0 : first;
      CrosstlMlxBuffer buffers[] = {
          {"a", dtype, a.data<uint8_t>() + a_first * a.itemsize(), a_scalar ? 1 : size, 0},
          {"b", dtype, b.data<uint8_t>() + b_first * b.itemsize(), b_scalar ? 1 : size, 0},
          {"c", dtype, c.data<uint8_t>() + first * c.itemsize(), size, 1},
          {"d", dtype, d.data<uint8_t>() + first * d.itemsize(), size, 1},
          {"size", "uint32", &size, 1, 0},
      };
      const bool scalar_scalar = std::string(vector_kernel) == "ss";
      divmod_launch(entry, buffers, scalar_scalar ? 4 : 5, size, elementwise_launch(size));
      first += size;
    }
    return;
  }
  auto [shape, strides] =
      mlx::core::collapse_contiguous_dims(a.shape(), {a.strides(), b.strides(), c.strides()});
  const size_t ndim = shape.size();
  if (ndim == 0 || ndim > 64) {
    throw std::invalid_argument("CrossTL DivMod general layout rank exceeds its bounds.");
  }
  uint64_t spans[2] = {1, 1};
  for (size_t operand = 0; operand < 2; ++operand) {
    for (size_t axis = 0; axis < ndim; ++axis) {
      spans[operand] += uint64_t(shape[axis] - 1) * uint64_t(strides[operand][axis]);
    }
    if (spans[operand] > max_elements) {
      throw std::invalid_argument("CrossTL DivMod operand span exceeds its bounds.");
    }
  }
  uint64_t dim0 = shape[ndim - 1];
  const uint64_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
  const uint64_t rest = c.size() / (dim0 * dim1);
  const std::string kernel = ndim <= 3 ? "g" + std::to_string(ndim) : "gn2";
  if (ndim > 3) {
    dim0 = (dim0 + 1) / 2;
  }
  if (dim0 > 65535 || dim1 > 65535 || rest > 65535) {
    throw std::invalid_argument("CrossTL DivMod general grid exceeds its bounds.");
  }
  const std::string entry = kernel + "_DivMod" + dtype;
  require_entry(entry);
  int32_t rank = static_cast<int32_t>(ndim);
  std::vector<CrosstlMlxBuffer> buffers = {
      {"a", dtype, a.data<uint8_t>(), spans[0], 0},
      {"b", dtype, b.data<uint8_t>(), spans[1], 0},
      {"c", dtype, c.data<uint8_t>(), c.size(), 1},
      {"d", dtype, d.data<uint8_t>(), d.size(), 1},
  };
  if (ndim == 1) {
    buffers.push_back({"a_stride", "int64", strides[0].data(), 1, 0});
    buffers.push_back({"b_stride", "int64", strides[1].data(), 1, 0});
  } else {
    if (ndim > 3) {
      buffers.push_back({"shape", "int32", shape.data(), ndim, 0});
      buffers.push_back({"ndim", "int32", &rank, 1, 0});
    }
    buffers.push_back({"a_strides", "int64", strides[0].data(), ndim, 0});
    buffers.push_back({"b_strides", "int64", strides[1].data(), ndim, 0});
  }
  const CrosstlMlxLaunch launch = {
      {static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1), static_cast<uint32_t>(rest)},
      {1, 1, 1}};
  divmod_launch(
      entry, buffers.data(), static_cast<uint32_t>(buffers.size()), c.size(), launch);
}
} // namespace

extern "C" MLX_API int crosstl_mlx_register_dispatch(
    uint32_t version,
    CrosstlMlxDispatch callback) {
  if (version != CROSTL_MLX_DISPATCH_VERSION || !callback) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(registration_mutex);
  CrosstlMlxDispatch expected = nullptr;
  return dispatch_callback.compare_exchange_strong(expected, callback) ? 0 : 2;
}

extern "C" MLX_API int crosstl_mlx_register_runtime(
    uint32_t version,
    CrosstlMlxDispatch callback,
    CrosstlMlxEntryAvailable available) {
  if (version != CROSTL_MLX_DISPATCH_VERSION || !callback || !available) {
    return 1;
  }
  std::lock_guard<std::mutex> lock(registration_mutex);
  if (dispatch_callback.load()) {
    return 2;
  }
  entry_available_callback.store(available);
  dispatch_callback.store(callback);
  return 0;
}

namespace mlx::core::gpu {
void init() {}
void new_stream(Stream) {
  require_runtime();
}
void new_thread_unsafe_stream(Stream) {
  require_runtime();
}
void clear_streams() {}

bool is_available() {
  return dispatch_callback.load() != nullptr;
}
int device_count() {
  return 1;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int index) {
  require_runtime();
  if (index != 0) {
    throw std::invalid_argument("CrossTL host runtime has one device.");
  }
  static const std::
      unordered_map<std::string, std::variant<std::string, size_t>>
          info{{"device_name", std::string("CrossTL native host dispatch")}};
  return info;
}

void eval(array& value) {
  require_runtime();
  auto outputs = value.outputs();
  value.primitive().eval_gpu(value.inputs(), outputs);
}

// The callback returns only after device readback, so no work remains queued.
void finalize(Stream) {
  require_runtime();
}
void synchronize(Stream) {
  require_runtime();
}
} // namespace mlx::core::gpu

namespace mlx::core {
#define CROSSTL_SHARED_VIEW_GPU(Primitive)                                 \
  void Primitive::eval_gpu(const std::vector<array>& inputs, array& out) { \
    require_runtime();                                                    \
    eval(inputs, out);                                                     \
  }

CROSSTL_SHARED_VIEW_GPU(AsStrided)
CROSSTL_SHARED_VIEW_GPU(Broadcast)
CROSSTL_SHARED_VIEW_GPU(BroadcastAxes)
CROSSTL_SHARED_VIEW_GPU(Copy)
CROSSTL_SHARED_VIEW_GPU(ExpandDims)
CROSSTL_SHARED_VIEW_GPU(Squeeze)
CROSSTL_SHARED_VIEW_GPU(StopGradient)
CROSSTL_SHARED_VIEW_GPU(Transpose)

#undef CROSSTL_SHARED_VIEW_GPU

#define CROSSTL_SHARED_OUTPUTS_GPU(Primitive)        \
  void Primitive::eval_gpu(                         \
      const std::vector<array>& inputs,             \
      std::vector<array>& outputs) {                \
    require_runtime();                             \
    eval(inputs, outputs);                         \
  }

CROSSTL_SHARED_OUTPUTS_GPU(CustomTransforms)
CROSSTL_SHARED_OUTPUTS_GPU(Depends)
CROSSTL_SHARED_OUTPUTS_GPU(Split)

#undef CROSSTL_SHARED_OUTPUTS_GPU

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1) {
    throw std::invalid_argument("CrossTL slice requires one input.");
  }
  slice(inputs[0], out, start_indices_, strides_);
}

void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_astype(inputs, out);
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].shape() != out.shape() ||
      axis_ < 0 || axis_ >= inputs[0].ndim() || out.size() > 65535) {
    throw std::invalid_argument("CrossTL scan requires matching bounded array shapes.");
  }
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  auto in = inputs[0];
  if (!in.flags().contiguous || in.strides()[axis_] == 0) {
    if (in.dtype() != out.dtype()) {
      array converted(in.shape(), out.dtype(), nullptr, {});
      dispatch_cast({in}, converted);
      in = std::move(converted);
    }
    in = dense_input(in);
  }
  auto scan_type = [](Dtype dtype) {
    return dtype == float16 ? "float16" : dtype == bfloat16 ? "bfloat16" : storage_type(dtype);
  };
  const char* source_type = scan_type(in.dtype());
  const char* output_type = scan_type(out.dtype());
  if (!source_type || !output_type ||
      (out.dtype() != in.dtype() &&
       !(reduce_type_ == Sum && in.dtype() == bool_ && out.dtype() == int32))) {
    throw std::invalid_argument("CrossTL scan requires supported matching scalar types or Boolean sum promotion.");
  }
  if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
      in.data_size() > (in.buffer_size() - uint64_t(in.offset())) / in.itemsize()) {
    throw std::invalid_argument("CrossTL scan input exceeds its allocation.");
  }
  uint64_t axis_size = in.shape(axis_);
  const auto source_stride = in.strides()[axis_];
  if (source_stride < 1 || source_stride > 65535 ||
      !axis_size || !in.data_size() || in.data_size() > 65535) {
    throw std::invalid_argument("CrossTL scan storage metadata exceeds its bounds.");
  }
  uint64_t stride = source_stride;
  if (in.data_size() % (axis_size * stride)) {
    throw std::invalid_argument("CrossTL scan axis and stride do not partition its storage.");
  }
  const bool contiguous = stride == 1;
  uint64_t stride_blocks = (stride + 31) / 32;
  const uint64_t rows = (in.data_size() / (axis_size * stride)) * stride_blocks;
  if (!rows || rows > 65535) {
    throw std::invalid_argument("CrossTL scan launch exceeds its native group limit.");
  }
  const uint32_t reads = in.itemsize() <= 4 ? 4 : 2;
  uint32_t width = contiguous ? 1024 : 32 * (32 / reads);
  if (contiguous && axis_size <= reads * 2048) {
    const uint64_t reduced = axis_size <= reads * 1024 ? axis_size : axis_size / 2;
    width = uint32_t((reduced + reads * 32 - 1) / (reads * 32)) * 32;
  }
  const char* operation = reduce_type_ == Sum ? "sum" :
      reduce_type_ == Prod ? "prod" : reduce_type_ == Min ? "min" :
      reduce_type_ == Max ? "max" : "logaddexp";
  const std::string entry = std::string(contiguous ? "contig_scan_" : "strided_scan_") +
      (reverse_ ? "reverse_" : "") + (inclusive_ ? "inclusive_" : "exclusive_") +
      operation + "_" + source_type + "_" + output_type;
  require_entry(entry);
  if (in.is_donatable() && in.itemsize() == out.itemsize()) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc(in.data_size() * out.itemsize()),
                 in.data_size(), in.strides(), in.flags());
  }
  CrosstlMlxBuffer buffers[] = {
      {"in", source_type, const_cast<void*>(in.data<void>()), in.data_size(), 0},
      {"out", output_type, out.data<void>(), out.data_size(), 1},
      {"axis_size", "uint64", &axis_size, 1, 0},
      {"stride", "uint64", &stride, 1, 0},
      {"stride_blocks", "uint64", &stride_blocks, 1, 0},
  };
  const CrosstlMlxLaunch launch{{1, uint32_t(rows), 1}, {width, 1, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, contiguous ? 3 : 5, in.data_size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native scan failed: ") + error);
  }
}

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].dtype() != uint32 ||
      inputs[0].ndim() < 1 || inputs[0].ndim() > 64 || inputs[0].shape(-1) != 2 ||
      (out.dtype() != uint8 && out.dtype() != uint16 && out.dtype() != uint32)) {
    throw std::invalid_argument("CrossTL random requires uint32 key pairs and unsigned byte, halfword or word output.");
  }
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  const uint64_t key_count = inputs[0].size() / 2;
  constexpr uint64_t max_native_bytes = std::numeric_limits<int32_t>::max() - 17;
  if (key_count == 0 || inputs[0].size() > 65535 || out.nbytes() > max_native_bytes ||
      out.size() % key_count != 0) {
    throw std::invalid_argument("CrossTL random output or key count exceeds its bounds.");
  }
  const uint64_t bytes_per_key = out.nbytes() / key_count;
  if (std::max<uint64_t>(4, bytes_per_key) > max_native_bytes / key_count) {
    throw std::invalid_argument("CrossTL random native allocation exceeds its bounds.");
  }
  const uint64_t words = (bytes_per_key + 3) / 4;
  const uint64_t columns = (words + 1) / 2;
  // The kernel keeps no group-shared state: split the word-pair axis into the
  // smallest exact group height that keeps every workgroup count portable.
  uint64_t height = (columns + 65534) / 65535;
  while (height <= 1024 && columns % height != 0) {
    ++height;
  }
  if (key_count > 65535 || height > 1024) {
    throw std::invalid_argument("CrossTL random launch exceeds the portable workgroup limits.");
  }
  auto keys = gather_input(inputs[0]);
  const auto span = gather_span(keys);
  const char* entry = keys.flags().row_contiguous ? "rbitsc" : "rbits";
  require_entry(entry);
  const int32_t rank = static_cast<int32_t>(keys.ndim());
  std::vector<int32_t> shape(keys.shape().begin(), keys.shape().end());
  std::vector<int64_t> strides(keys.strides().begin(), keys.strides().end());
  out.set_data(allocator::malloc(out.nbytes()));
  CrosstlMlxBuffer buffers[] = {
      {"keys", "uint32", keys.data<void>(), span, 0},
      {"out", "int8", out.data<void>(), out.nbytes(), 1},
      {"bytes_per_key", "uint64", const_cast<uint64_t*>(&bytes_per_key), 1, 0},
      {"ndim", "int32", const_cast<int32_t*>(&rank), 1, 0},
      {"key_shape", "int32", shape.data(), shape.size(), 0},
      {"key_strides", "int64", strides.data(), strides.size(), 0},
  };
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(key_count), static_cast<uint32_t>(columns / height), 1},
      {1, static_cast<uint32_t>(height), 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry, buffers, 6, out.nbytes(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native random failed: ") + error);
  }
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  constexpr uint64_t max_storage_elements = (uint64_t(1) << 28) - 33;
  if (inputs.size() < 2 || inputs.size() > 11 ||
      axes_.size() != inputs.size() - 1 || out.size() > max_storage_elements ||
      slice_sizes_.size() != inputs[0].ndim() ||
      !storage_type(out.dtype()) || out.dtype() != inputs[0].dtype()) {
    throw std::invalid_argument("CrossTL gather specialization exceeds its supported layout.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto src = gather_input(inputs[0]);
  const auto src_span = gather_span(src, max_storage_elements);
  const int count = static_cast<int>(inputs.size() - 1);
  const auto index_dtype = inputs[1].dtype();
  if (index_dtype != int32 && index_dtype != uint32 &&
      index_dtype != int64 && index_dtype != uint64) {
    throw std::invalid_argument("CrossTL gather indices require int32, uint32, int64 or uint64.");
  }
  const int32_t idx_ndim = static_cast<int32_t>(inputs[1].ndim());
  if (idx_ndim > 64) {
    throw std::invalid_argument("CrossTL gather index rank exceeds 64.");
  }
  const std::string entry = std::string("gather") + storage_type(out.dtype()) +
      storage_type(index_dtype) + "_" + std::to_string(count) + "_" +
      std::to_string(idx_ndim) + "_int";
  require_entry(entry);
  std::vector<array> indices;
  std::vector<uint64_t> spans;
  std::vector<int32_t> shapes;
  std::vector<int64_t> strides;
  std::vector<uint8_t> contiguous;
  std::vector<std::string> names;
  for (int i = 0; i < count; ++i) {
    const auto& input = inputs[i + 1];
    if (input.dtype() != index_dtype || input.shape() != inputs[1].shape()) {
      throw std::invalid_argument("CrossTL gather index arrays must have matching shapes and types.");
    }
    indices.push_back(gather_input(input));
    spans.push_back(gather_span(indices.back(), max_storage_elements));
    shapes.insert(shapes.end(), indices.back().shape().begin(), indices.back().shape().end());
    strides.insert(strides.end(), indices.back().strides().begin(), indices.back().strides().end());
    contiguous.push_back(indices.back().flags().row_contiguous);
    names.push_back("idx" + std::to_string(i));
  }
  if (shapes.empty()) {
    shapes.push_back(1);
    strides.push_back(0);
  }
  std::vector<int32_t> src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> src_strides(src.strides().begin(), src.strides().end());
  uint64_t ndim = src.ndim();
  int32_t index_rank = idx_ndim;
  std::vector<int32_t> slices(slice_sizes_.begin(), slice_sizes_.end());
  std::vector<int32_t> axes(axes_.begin(), axes_.end());
  std::vector<CrosstlMlxBuffer> buffers = {
      {"src", storage_type(src.dtype()), const_cast<void*>(src.data<void>()), src_span, 0},
      {"out", storage_type(out.dtype()), out.data<void>(), out.size(), 1},
      {"src_shape", "int32", src_shape.data(), ndim, 0},
      {"src_strides", "int64", src_strides.data(), ndim, 0},
      {"src_ndim", "uint64", &ndim, 1, 0},
      {"slice_sizes", "int32", slices.data(), ndim, 0},
      {"axes", "int32", axes.data(), uint64_t(count), 0},
      {"idx_shapes", "int32", shapes.data(), shapes.size(), 0},
      {"idx_strides", "int64", strides.data(), strides.size(), 0},
      {"idx_contigs", "bool_", contiguous.data(), contiguous.size(), 0},
      {"idx_ndim", "int32", &index_rank, 1, 0},
  };
  for (int i = 0; i < count; ++i) {
    buffers.push_back({names[i].c_str(), storage_type(index_dtype),
                      const_cast<void*>(indices[i].data<void>()), spans[i], 0});
  }
  uint64_t slice_size = 1;
  for (auto size : slices) {
    if (size < 1 || slice_size > 65535 / uint64_t(size)) {
      throw std::invalid_argument("CrossTL gather slice size exceeds its bounds.");
    }
    slice_size *= size;
  }
  const uint64_t index_size = inputs[1].size();
  if (index_size == 0 || index_size > max_storage_elements / slice_size ||
      out.size() != index_size * slice_size) {
    throw std::invalid_argument("CrossTL gather output does not match index and slice sizes.");
  }
  // Rebase exact rectangular index slabs; source indexing and slice order are unchanged.
  size_t split_axis = 0;
  uint64_t suffix = idx_ndim ? index_size / inputs[1].shape(0) : 1;
  while (suffix > 65535) {
    ++split_axis;
    suffix /= inputs[1].shape(split_axis);
  }
  const uint64_t axis_limit = split_axis == 0 ? 65535 : 65535 / suffix;
  for (uint64_t first = 0; first < index_size;) {
    std::vector<int32_t> tile_shape(inputs[1].shape().begin(), inputs[1].shape().end());
    std::vector<uint64_t> coordinates(idx_ndim);
    uint64_t remaining = first;
    for (size_t axis = idx_ndim; axis-- > 0;) {
      coordinates[axis] = remaining % tile_shape[axis];
      remaining /= tile_shape[axis];
    }
    const uint64_t extent = idx_ndim
        ? std::min<uint64_t>(axis_limit, tile_shape[split_axis] - coordinates[split_axis]) : 1;
    if (idx_ndim) {
      for (size_t axis = 0; axis < split_axis; ++axis) {
        tile_shape[axis] = 1;
      }
      tile_shape[split_axis] = static_cast<int32_t>(extent);
    }
    const uint64_t tile_count = extent * suffix;
    for (int i = 0; i < count; ++i) {
      uint64_t offset = 0, span = 1, dense_stride = 1;
      bool dense = true;
      for (size_t axis = idx_ndim; axis-- > 0;) {
        const uint64_t stride = indices[i].strides()[axis];
        offset += coordinates[axis] * stride;
        span += uint64_t(tile_shape[axis] - 1) * stride;
        dense = dense && (tile_shape[axis] == 1 || stride == dense_stride);
        dense_stride *= tile_shape[axis];
        shapes[i * idx_ndim + axis] = tile_shape[axis];
      }
      if (offset > spans[i] || span > spans[i] - offset) {
        throw std::invalid_argument("CrossTL gather tile exceeds its index allocation.");
      }
      contiguous[i] = dense;
      buffers[11 + i].data = const_cast<uint8_t*>(indices[i].data<uint8_t>()) +
          offset * indices[i].itemsize();
      buffers[11 + i].count = span;
    }
    const uint64_t elements = tile_count * slice_size;
    buffers[1].data = out.data<uint8_t>() + first * slice_size * out.itemsize();
    buffers[1].count = elements;
    const uint64_t dim0 = idx_ndim ? tile_shape[0] : 1;
    const CrosstlMlxLaunch launch{
        {static_cast<uint32_t>(dim0), static_cast<uint32_t>(tile_count / dim0),
         static_cast<uint32_t>(slice_size)}, {1, 1, 1}};
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
        elements, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native gather failed: ") + error);
    }
    first += tile_count;
  }
}

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 2 || !storage_type(out.dtype()) ||
      inputs[0].dtype() != out.dtype() || inputs[0].ndim() == 0 ||
      inputs[0].ndim() > 64 || inputs[0].ndim() != inputs[1].ndim() ||
      axis_ < 0 || axis_ >= inputs[0].ndim() ||
      out.shape() != inputs[1].shape() || out.size() > 65535) {
    throw std::invalid_argument("CrossTL axis gather requires matching bounded array layouts.");
  }
  const auto index_dtype = inputs[1].dtype();
  if (index_dtype != int32 && index_dtype != uint32 &&
      index_dtype != int64 && index_dtype != uint64) {
    throw std::invalid_argument("CrossTL axis gather requires 32-bit or 64-bit integer indices.");
  }
  for (size_t i = 0; i < inputs[0].ndim(); ++i) {
    if (i != axis_ && inputs[0].shape(i) != inputs[1].shape(i)) {
      throw std::invalid_argument("CrossTL axis gather requires broadcast input shapes.");
    }
  }
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto src = gather_input(inputs[0]);
  auto idx = gather_input(inputs[1]);
  const auto source_span = gather_span(src);
  const auto index_span = gather_span(idx);
  const std::string entry = std::string("gather_axis") + storage_type(out.dtype()) +
      storage_type(index_dtype) + "_int" + (src.flags().row_contiguous ? "c" : "nc") +
      (idx.flags().row_contiguous ? "c" : "nc");
  require_entry(entry);
  std::vector<int32_t> shape;
  std::vector<int64_t> source_strides, index_strides;
  uint64_t before = 1, after = 1;
  for (size_t i = 0; i < src.ndim(); ++i) {
    if (i == axis_) {
      continue;
    }
    shape.push_back(idx.shape(i));
    source_strides.push_back(src.strides()[i]);
    index_strides.push_back(idx.strides()[i]);
    (i < axis_ ? before : after) *= idx.shape(i);
  }
  uint64_t ndim = src.ndim() - 1;
  if (shape.empty()) {
    shape.push_back(1);
    source_strides.push_back(0);
    index_strides.push_back(0);
  }
  int32_t axis = axis_, axis_size = src.shape(axis_);
  uint64_t source_step = src.strides()[axis_], index_step = idx.strides()[axis_];
  CrosstlMlxBuffer buffers[] = {
      {"src", storage_type(src.dtype()), const_cast<void*>(src.data<void>()), source_span, 0},
      {"indices", storage_type(idx.dtype()), const_cast<void*>(idx.data<void>()), index_span, 0},
      {"out", storage_type(out.dtype()), out.data<void>(), out.size(), 1},
      {"shape", "int32", shape.data(), shape.size(), 0},
      {"src_strides", "int64", source_strides.data(), source_strides.size(), 0},
      {"idx_strides", "int64", index_strides.data(), index_strides.size(), 0},
      {"ndim", "uint64", &ndim, 1, 0},
      {"axis", "int32", &axis, 1, 0},
      {"axis_size", "int32", &axis_size, 1, 0},
      {"src_ax_stride", "uint64", &source_step, 1, 0},
      {"idx_ax_stride", "uint64", &index_step, 1, 0},
  };
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(after), static_cast<uint32_t>(idx.shape(axis_)),
       static_cast<uint32_t>(before)}, {1, 1, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, 11, out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native axis gather failed: ") + error);
  }
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() < 3 || inputs.size() > 12 || axes_.size() != inputs.size() - 2 ||
      (out.dtype() != int32 && out.dtype() != uint32 && out.dtype() != float32) ||
      inputs[0].dtype() != out.dtype() || inputs.back().dtype() != out.dtype() ||
      inputs[0].shape() != out.shape() || out.ndim() == 0 || out.ndim() > 64 ||
      out.size() > 65535 || inputs.back().size() > 65535 ||
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
      inputs.back().ndim() != index_rank + out.ndim() || inputs[1].size() > 65535) {
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
  const auto update_span = gather_span(upd);
  uint64_t idx_size = inputs[1].size(), upd_ndim = upd.ndim(), out_ndim = out.ndim();
  uint64_t upd_size = 1;
  for (size_t i = index_rank; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }
  const auto ratio = idx_size / out.size();
  const int work = index_rank <= 1 || ratio < 1 ? 1 :
      ratio <= 4 ? 4 : ratio < 16 ? 8 : ratio < 32 ? 16 : 32;
  const char* operation = reduce_type_ == None ? "none" :
      reduce_type_ == Sum ? "sum" : reduce_type_ == Prod ? "prod" :
      reduce_type_ == Min ? "min" : "max";
  const auto count = axes_.size();
  const std::string entry = std::string("scatter") + storage_type(out.dtype()) +
      storage_type(index_dtype) + "_" + operation + "_" + std::to_string(count) +
      "_updc_" + (upd.flags().row_contiguous ? "true" : "false") +
      "_nwork" + std::to_string(work) + "_int";
  require_entry(entry);
  std::vector<array> indices;
  std::vector<uint64_t> spans;
  std::vector<int32_t> shapes;
  std::vector<int64_t> strides;
  std::vector<uint8_t> contiguous;
  std::vector<std::string> names;
  for (size_t i = 0; i < count; ++i) {
    indices.push_back(gather_input(inputs[i + 1]));
    spans.push_back(gather_span(indices.back()));
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
  int32_t idx_ndim = index_rank;
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
      {"idx_size", "uint64", &idx_size, 1, 0},
  };
  for (size_t i = 0; i < count; ++i) {
    buffers.push_back({names[i].c_str(), storage_type(index_dtype),
                      const_cast<void*>(indices[i].data<void>()), spans[i], 0});
  }
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(upd_size), static_cast<uint32_t>((idx_size + work - 1) / work), 1},
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

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 3 || (out.dtype() != int32 && out.dtype() != uint32) ||
      inputs[0].dtype() != out.dtype() || inputs[2].dtype() != out.dtype() ||
      inputs[0].shape() != out.shape() || out.ndim() == 0 || out.ndim() > 64 ||
      inputs[1].ndim() != out.ndim() || inputs[2].shape() != inputs[1].shape() ||
      axis_ < 0 || axis_ >= out.ndim() || out.size() > 65535 ||
      inputs[1].size() > 65535 || (reduce_type_ != None && reduce_type_ != Sum)) {
    throw std::invalid_argument("CrossTL axis scatter requires bounded int32 or uint32 arrays.");
  }
  const auto index_dtype = inputs[1].dtype();
  if (index_dtype != int32 && index_dtype != uint32 &&
      index_dtype != int64 && index_dtype != uint64) {
    throw std::invalid_argument("CrossTL axis scatter requires 32-bit or 64-bit integer indices.");
  }
  for (size_t i = 0; i < out.ndim(); ++i) {
    if (i != axis_ && inputs[1].shape(i) != out.shape(i)) {
      throw std::invalid_argument("CrossTL axis scatter requires broadcast input shapes.");
    }
  }
  // Preserve the source allocation, including when updates alias a source view.
  dispatch_copy(inputs[0], out);
  if (inputs[1].size() == 0) {
    return;
  }
  auto upd = gather_input(inputs[2]);
  auto idx = gather_input(inputs[1]);
  const auto update_span = gather_span(upd), index_span = gather_span(idx);
  const std::string entry = std::string("scatter_axis") + storage_type(out.dtype()) +
      storage_type(index_dtype) + (reduce_type_ == None ? "_none_int" : "_sum_int") +
      (upd.flags().row_contiguous ? "c" : "nc") + (idx.flags().row_contiguous ? "c" : "nc");
  require_entry(entry);
  std::vector<int32_t> shape;
  std::vector<int64_t> update_strides, index_strides;
  uint64_t before = 1, after = 1;
  for (size_t i = 0; i < out.ndim(); ++i) {
    if (i == axis_) {
      continue;
    }
    shape.push_back(idx.shape(i));
    update_strides.push_back(upd.strides()[i]);
    index_strides.push_back(idx.strides()[i]);
    (i < axis_ ? before : after) *= idx.shape(i);
  }
  uint64_t ndim = out.ndim() - 1;
  if (shape.empty()) {
    shape.push_back(1);
    update_strides.push_back(0);
    index_strides.push_back(0);
  }
  int32_t axis = axis_, axis_size = out.shape(axis_);
  uint64_t update_step = upd.strides()[axis_], index_step = idx.strides()[axis_];
  CrosstlMlxBuffer buffers[] = {
      {"upd", storage_type(upd.dtype()), const_cast<void*>(upd.data<void>()), update_span, 0},
      {"indices", storage_type(idx.dtype()), const_cast<void*>(idx.data<void>()), index_span, 0},
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
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(after), static_cast<uint32_t>(idx.shape(axis_)),
       static_cast<uint32_t>(before)}, {1, 1, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, 11, out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native axis scatter failed: ") + error);
  }
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1 || axes_.empty()) {
    throw std::invalid_argument("CrossTL reduction requires one input and nonempty axes.");
  }
  array in = inputs[0];
  if (in.size() > 0 && out.size() == in.size()) {
    array identity(out.shape(), in.dtype(), nullptr, {});
    reshape_view(inputs, identity);
    dispatch_cast({identity}, out);
    return;
  }
  // Extrema select an input element, so 16-bit float results need no rounding.
  const bool half_extrema = (in.dtype() == bfloat16 || in.dtype() == float16) &&
      (reduce_type_ == Min || reduce_type_ == Max);
  const char* dtype = half_extrema ? (in.dtype() == bfloat16 ? "bfloat16" : "float16")
                                   : storage_type(in.dtype());
  const bool boolean = in.dtype() == bool_;
  const bool logical = reduce_type_ == And || reduce_type_ == Or;
  const char* operation = nullptr;
  switch (reduce_type_) {
    case And: operation = "and"; break;
    case Or: operation = "or"; break;
    case Sum: operation = "sum"; break;
    case Prod: operation = "prod"; break;
    case Min: operation = boolean ? "and" : "min"; break;
    case Max: operation = boolean ? "or" : "max"; break;
  }
  if (in.size() == 0) {
    dispatch_empty_reduce(out, operation);
    return;
  }
  if (!dtype || in.dtype() == int64 || in.dtype() == uint64 || in.dtype() != out.dtype() ||
      (boolean ? (reduce_type_ == Sum || reduce_type_ == Prod) : logical)) {
    throw std::invalid_argument(
        "CrossTL reductions require matching float32/int32/uint32 numeric arrays or Boolean all/any.");
  }
  auto plan = get_reduction_plan(in, axes_);
  // Column passes bound their own views; every other plan that is not a
  // whole-array reduction binds at most 65535 elements, checked before a
  // general plan's dense copy.
  const bool column_plan =
      plan.type == ContiguousStridedReduce || plan.type == GeneralStridedReduce;
  if (out.size() != 1 && in.size() > 65535 && !column_plan) {
    throw std::invalid_argument("CrossTL row reductions support at most 65535 elements.");
  }
  if (plan.type == GeneralReduce) {
    in = dense_input(in);
    plan = get_reduction_plan(in, axes_);
  }
  if (half_extrema && plan.type != ContiguousAllReduce &&
      !((plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) &&
        plan.shape.back() <= 64)) {
    throw std::invalid_argument(
        "CrossTL 16-bit float extrema require whole-array or small-row reduction packages.");
  }
  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    dispatch_row_reduce(in, out, plan, axes_, operation, dtype);
    return;
  }
  if (plan.type == ContiguousStridedReduce || plan.type == GeneralStridedReduce) {
    dispatch_column_reduce(in, out, plan, axes_, operation, dtype);
    return;
  }
  if (plan.type != ContiguousAllReduce) {
    throw std::invalid_argument("CrossTL reduction plan is not implemented.");
  }
  if (in.size() <= 4096) {
    dispatch_all_reduce(in, out, operation, dtype);
  } else {
    // Retain upstream's two passes; each callback completes before its input expires.
    array intermediate({static_cast<int>(all_reduce_rows(in))}, out.dtype(), nullptr, {});
    dispatch_all_reduce(in, intermediate, operation, dtype);
    dispatch_all_reduce(intermediate, out, operation, dtype);
  }
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL Full input must match output shape.");
  }
  // MLX has already broadcast and cast the value; materialize it on the device.
  dispatch_copy(inputs[0], out);
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 2 || inputs[0].ndim() != out.ndim() ||
      inputs[1].size() != 1 || inputs[0].dtype() != out.dtype() ||
      inputs[1].dtype() != out.dtype() || !storage_type(out.dtype()) ||
      out.size() > 65535 || out.ndim() > 64 ||
      axes_.size() != low_pad_size_.size() || axes_.size() != high_pad_size_.size()) {
    throw std::invalid_argument(
        "CrossTL padding requires matching supported arrays, one fill value and at most 65535 output elements.");
  }
  const auto& in = inputs[0];
  Shape expected = in.shape();
  int64_t offset = 0;
  for (size_t i = 0; i < axes_.size(); ++i) {
    const int axis = axes_[i] < 0 ? out.ndim() + axes_[i] : axes_[i];
    if (axis < 0 || axis >= out.ndim() || low_pad_size_[i] < 0 || high_pad_size_[i] < 0 ||
        int64_t(expected[axis]) + low_pad_size_[i] + high_pad_size_[i] > 65535) {
      throw std::invalid_argument("CrossTL padding axes or extents are invalid.");
    }
    expected[axis] += low_pad_size_[i] + high_pad_size_[i];
    offset += out.strides()[axis] * low_pad_size_[i];
  }
  if (expected != out.shape()) {
    throw std::invalid_argument("CrossTL padding output shape does not match.");
  }
  // A broadcast view lets the unchanged copy kernel fill every output element.
  array fill(out.shape(), out.dtype(), nullptr, {});
  fill.copy_shared_buffer(
      inputs[1], Strides(out.ndim(), 0), {true, out.size() <= 1, out.size() <= 1}, 1);
  dispatch_copy(fill, out);
  dispatch_copy_into(
      in, out, std::vector<int64_t>(out.strides().begin(), out.strides().end()),
      offset, true);
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 2 || inputs[0].shape() != out.shape() ||
      inputs[1].ndim() != out.ndim() || inputs[0].dtype() != out.dtype() ||
      inputs[1].dtype() != out.dtype() ||
      (!storage_type(out.dtype()) && !(out.dtype() == bfloat16 && reduce_type_ == None)) ||
      out.size() > max_slice_update_elements || out.ndim() > 64 ||
      start_indices_.size() != out.ndim() || end_indices_.size() != out.ndim() ||
      strides_.size() != out.ndim()) {
    throw std::invalid_argument(
        "CrossTL slice updates require matching supported arrays within the storage capacity.");
  }
  const auto& update = inputs[1];
  int64_t offset = 0;
  std::vector<int64_t> destination_strides(out.ndim());
  for (int axis = 0; axis < out.ndim(); ++axis) {
    const int64_t step = strides_[axis], start = start_indices_[axis];
    // MLX normalizes starts and update shapes, but retains unnormalized stops.
    if (!step) {
      throw std::invalid_argument("CrossTL slice update strides must be nonzero.");
    }
    if (update.size()) {
      const int64_t last = start + (int64_t(update.shape(axis)) - 1) * step;
      if (start < 0 || start >= out.shape(axis) || last < 0 || last >= out.shape(axis)) {
        throw std::invalid_argument("CrossTL slice update exceeds the destination.");
      }
    }
    offset += start * out.strides()[axis];
    destination_strides[axis] = step * out.strides()[axis];
  }
  // Keep the base and update allocations alive and distinct, including aliasing views.
  std::string entry;
  if (reduce_type_ != None && update.size()) {
    const char* operation = reduce_type_ == Sum ? "sum"
        : reduce_type_ == Prod ? "prod"
        : reduce_type_ == Min ? "min"
        : reduce_type_ == Max ? "max" : nullptr;
    if (!operation) {
      throw std::invalid_argument("Unknown CrossTL slice update reduction.");
    }
    entry = std::string("slice_update_") + operation + storage_type(out.dtype());
    require_entry(entry);
  }
  dispatch_copy(inputs[0], out);
  if (reduce_type_ == None) {
    dispatch_copy_into(update, out, std::move(destination_strides), offset, true);
  } else {
    dispatch_slice_update(update, out, entry, std::move(destination_strides), offset);
  }
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape_view(inputs, out);
}

void Unflatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape_view(inputs, out);
}

void Flatten::eval_gpu(const std::vector<array>& inputs, array& out) {
  reshape_view(inputs, out);
}

// Ports mlx/backend/gpu/primitives.cpp View::eval_gpu; the general copy
// runs through the translated copy kernels.
void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1) {
    throw std::invalid_argument("CrossTL view requires one input.");
  }
  const auto& in = inputs[0];
  const auto ibytes = size_of(in.dtype());
  const auto obytes = size_of(out.dtype());
  // Upstream's disjunction; row contiguity is tested first so a scalar never
  // reads the last stride of an empty stride vector.
  if (in.flags().row_contiguous || ibytes == obytes ||
      (obytes < ibytes && in.strides().back() == 1)) {
    auto strides = in.strides();
    for (int i = 0; i < static_cast<int>(strides.size()) - 1; ++i) {
      strides[i] *= ibytes;
      strides[i] /= obytes;
    }
    out.copy_shared_buffer(
        in, strides, in.flags(), in.data_size() * ibytes / obytes);
    return;
  }
  // Upstream copies the input elements densely and reinterprets the result.
  // Narrow elements that pack into whole 32-bit words (or whole output
  // elements, when those are narrower) along the contiguous last axis are
  // copied as those words instead: the bytes are identical, and no 8- or
  // 16-bit element is loaded as a value, so non-canonical Boolean bytes from
  // an earlier view survive. A scalar is row contiguous, so ndim >= 1 here.
  array source = in;
  const int64_t word = std::min<int64_t>(obytes, 4);
  if (int64_t(ibytes) < word && in.strides().back() == 1 &&
      in.offset() % word == 0) {
    const int64_t ratio = word / int64_t(ibytes);
    bool grouped = in.shape().back() % ratio == 0;
    auto strides = in.strides();
    for (int i = 0; grouped && i < static_cast<int>(strides.size()) - 1; ++i) {
      grouped = strides[i] % ratio == 0;
      strides[i] /= ratio;
    }
    if (grouped) {
      auto shape = in.shape();
      shape.back() /= ratio;
      source = array(
          shape, word == int64_t(obytes) ? out.dtype() : uint32, nullptr, {});
      source.copy_shared_buffer(
          in, strides, in.flags(), in.data_size() * ibytes / word);
    }
  }
  array tmp(source.shape(), source.dtype(), nullptr, {});
  dispatch_copy(source, tmp);
  auto flags = out.flags();
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
  flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
  out.copy_shared_buffer(tmp, out.strides(), flags, out.size());
}

void Contiguous::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1) {
    throw std::invalid_argument("CrossTL contiguous requires one input.");
  }
  const auto& in = inputs[0];
  if (in.buffer_size() <= out.nbytes() + 16384 &&
      (in.flags().row_contiguous || (allow_col_major_ && in.flags().col_contiguous))) {
    out.copy_shared_buffer(in);
  } else {
    dispatch_copy(in, out);
  }
}

#define CROSSTL_UNARY_GPU(Primitive)                                       \
  void Primitive::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_unary(inputs, out, name());                                   \
  }

CROSSTL_UNARY_GPU(Abs)
CROSSTL_UNARY_GPU(ArcCos)
CROSSTL_UNARY_GPU(ArcCosh)
CROSSTL_UNARY_GPU(ArcSin)
CROSSTL_UNARY_GPU(ArcSinh)
CROSSTL_UNARY_GPU(ArcTan)
CROSSTL_UNARY_GPU(ArcTanh)
CROSSTL_UNARY_GPU(Ceil)
CROSSTL_UNARY_GPU(Cos)
CROSSTL_UNARY_GPU(Cosh)
CROSSTL_UNARY_GPU(Exp)
CROSSTL_UNARY_GPU(Expm1)
CROSSTL_UNARY_GPU(Floor)
CROSSTL_UNARY_GPU(Log)
CROSSTL_UNARY_GPU(Log1p)
CROSSTL_UNARY_GPU(Negative)
CROSSTL_UNARY_GPU(Sigmoid)
CROSSTL_UNARY_GPU(Erf)
CROSSTL_UNARY_GPU(ErfInv)
CROSSTL_UNARY_GPU(Sign)
CROSSTL_UNARY_GPU(Sin)
CROSSTL_UNARY_GPU(Sinh)
CROSSTL_UNARY_GPU(Square)
CROSSTL_UNARY_GPU(Sqrt)
CROSSTL_UNARY_GPU(Tan)
CROSSTL_UNARY_GPU(Tanh)
CROSSTL_UNARY_GPU(Round)
CROSSTL_UNARY_GPU(LogicalNot)
CROSSTL_UNARY_GPU(BitwiseInvert)

#undef CROSSTL_UNARY_GPU

#define CROSSTL_BINARY_GPU(Primitive)                                    \
  void Primitive::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_binary(inputs, out, name());                                 \
  }

CROSSTL_BINARY_GPU(Add)
CROSSTL_BINARY_GPU(Subtract)
CROSSTL_BINARY_GPU(Multiply)
CROSSTL_BINARY_GPU(Minimum)
CROSSTL_BINARY_GPU(Maximum)
CROSSTL_BINARY_GPU(Divide)
CROSSTL_BINARY_GPU(Remainder)
CROSSTL_BINARY_GPU(LogAddExp)

#undef CROSSTL_BINARY_GPU

void DivMod::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  dispatch_divmod(inputs, outputs);
}

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_binary(inputs, out, name(), false, true);
}

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const char* dtype = storage_type(out.dtype());
  if (inputs.size() != 3 || !dtype || out.dtype() == int64 || out.dtype() == uint64 ||
      inputs[0].dtype() != bool_ ||
      inputs[1].dtype() != out.dtype() || inputs[2].dtype() != out.dtype()) {
    throw std::invalid_argument(
        "CrossTL selection requires Boolean conditions and matching float32/int32/uint32/bool values.");
  }
  if (out.size() > 65535) {
    throw std::invalid_argument("CrossTL selection supports at most 65535 elements.");
  }
  for (const auto& in : inputs) {
    if (in.shape() != out.shape()) {
      throw std::invalid_argument(
          "CrossTL selection inputs must match the output shape.");
    }
  }
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // MLX has already cast and broadcast the operands; materialize their layouts.
  auto condition = dense_input(inputs[0]);
  auto left = dense_input(inputs[1]);
  auto right = dense_input(inputs[2]);
  uint32_t size = static_cast<uint32_t>(out.size());
  std::string entry = std::string("v_Select") + dtype;
  CrosstlMlxBuffer buffers[] = {
      {"a", "bool_", condition.data<void>(), size, 0},
      {"b", dtype, left.data<void>(), size, 0},
      {"c", dtype, right.data<void>(), size, 0},
      {"d", dtype, out.data<void>(), size, 1},
      {"size", "uint32", &size, 1, 0},
  };
  const auto launch = elementwise_launch(size);
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, 5, size, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(
        std::string("CrossTL native selection failed: ") + error);
  }
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.empty() || axis_ < 0 || axis_ >= out.ndim() || out.ndim() > 64 ||
      out.size() > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("CrossTL concatenate shape exceeds its bounds.");
  }
  int64_t axis_size = 0;
  for (const auto& in : inputs) {
    if (in.dtype() != out.dtype() ||
        (in.dtype() != float32 && in.dtype() != int32 &&
         in.dtype() != uint32 && in.dtype() != bool_)) {
      throw std::invalid_argument(
          "CrossTL concatenate requires matching float32/int32/uint32/bool arrays.");
    }
    if (in.ndim() != out.ndim() || in.size() > 65535) {
      throw std::invalid_argument(
          "CrossTL concatenate supports at most 65535 elements per input.");
    }
    for (int axis = 0; axis < out.ndim(); ++axis) {
      if (axis != axis_ && in.shape(axis) != out.shape(axis)) {
        throw std::invalid_argument("CrossTL concatenate input shapes do not match.");
      }
    }
    axis_size += in.shape(axis_);
  }
  if (axis_size != out.shape(axis_)) {
    throw std::invalid_argument("CrossTL concatenate output shape does not match.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  std::vector<int64_t> strides(out.strides().begin(), out.strides().end());
  int64_t offset = 0;
  bool preserve = false;
  for (const auto& in : inputs) {
    if (in.size()) {
      dispatch_copy_into(in, out, strides, offset, preserve);
      preserve = true;
    }
    offset += int64_t(in.shape(axis_)) * strides[axis_];
  }
}

#define CROSSTL_COMPARISON_GPU(Primitive)                                 \
  void Primitive::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_binary(inputs, out, name(), true);                            \
  }

CROSSTL_COMPARISON_GPU(Equal)
CROSSTL_COMPARISON_GPU(NotEqual)
CROSSTL_COMPARISON_GPU(Less)
CROSSTL_COMPARISON_GPU(LessEqual)
CROSSTL_COMPARISON_GPU(Greater)
CROSSTL_COMPARISON_GPU(GreaterEqual)
CROSSTL_COMPARISON_GPU(LogicalAnd)
CROSSTL_COMPARISON_GPU(LogicalOr)

#undef CROSSTL_COMPARISON_GPU

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (!inputs.empty()) {
    throw std::invalid_argument("Arange does not accept input arrays.");
  }
  if (out.dtype() != float32 && out.dtype() != int32 && out.dtype() != uint32 &&
      out.dtype() != int64 && out.dtype() != uint64) {
    throw std::invalid_argument("CrossTL Arange does not support this dtype.");
  }
  if (out.size() > 65535) {
    throw std::invalid_argument(
        "CrossTL Arange currently supports at most 65535 elements.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  if (out.dtype() == float32) {
    dispatch_arange<float>(out, start_, step_, "arangefloat32", "float32");
  } else if (out.dtype() == int32) {
    dispatch_arange<int32_t>(out, start_, step_, "arangeint32", "int32");
  } else if (out.dtype() == uint32) {
    dispatch_arange<uint32_t>(out, start_, step_, "arangeuint32", "uint32");
  } else if (out.dtype() == int64) {
    dispatch_arange<int64_t>(out, start_, step_, "arangeint64", "int64");
  } else {
    dispatch_arange<uint64_t>(out, start_, step_, "arangeuint64", "uint64");
  }
}

namespace {
constexpr uint64_t max_matmul_elements = (uint64_t(1) << 28) - 1;

// quantized.cpp ensure_row_contiguous_matrix, plus nonnegative batch strides.
array matmul_operand(const array& in) {
  const auto ndim = in.ndim();
  bool dense = in.flags().row_contiguous ||
      (ndim < 2 ? in.strides()[0] == 1
                : in.strides()[ndim - 2] == in.shape(-1) && in.strides()[ndim - 1] == 1);
  for (size_t axis = 0; dense && axis + 2 < ndim; ++axis) {
    dense = in.strides()[axis] >= 0;
  }
  return dense ? in : dense_input(in);
}

// Elements from the view's first element through its last.
uint64_t matmul_span(const array& in) {
  uint64_t span = 1;
  for (size_t axis = 0; axis < in.ndim(); ++axis) {
    const auto stride = in.strides()[axis];
    const uint64_t extent = in.shape(axis);
    if (stride < 0 || extent == 0 ||
        (stride && extent - 1 > (max_matmul_elements - span) / uint64_t(stride))) {
      throw std::invalid_argument("CrossTL matmul operand exceeds its bounds.");
    }
    span += (extent - 1) * uint64_t(stride);
  }
  if (in.offset() < 0 || in.offset() % in.itemsize() != 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      span * in.itemsize() > in.buffer_size() - uint64_t(in.offset())) {
    throw std::invalid_argument("CrossTL matmul operand exceeds its allocation.");
  }
  return span;
}

const char* matmul_storage(Dtype dtype) {
  return dtype == float32 ? "float32" : dtype == float16 ? "float16" : "bfloat16";
}

const char* quantized_source(Dtype dtype) {
  return dtype == float32 ? "float" : dtype == float16 ? "float16_t" : "bfloat16_t";
}

// add_strides_and_shapes' batch parameters, kept alive through the dispatch.
// Kernels without biases bind no bias strides.
struct QuantizedBatchViews {
  int32_t x_batch_ndims;
  Shape x_shape;
  Strides x_strides;
  int32_t w_batch_ndims;
  Shape w_shape;
  Strides w_strides;
  Strides s_strides;
  Strides b_strides;
  bool biased;

  QuantizedBatchViews(
      const array& x, const array& w, const array& scales, const array& biases)
      : QuantizedBatchViews(x, w, scales) {
    b_strides = biases.strides();
    biased = true;
  }

  QuantizedBatchViews(const array& x, const array& w, const array& scales)
      : x_batch_ndims(int32_t(x.ndim()) - 2),
        x_shape(x.shape()),
        x_strides(x.strides()),
        w_batch_ndims(int32_t(w.ndim()) - 2),
        w_shape(w.shape()),
        w_strides(w.strides()),
        s_strides(scales.strides()),
        biased(false) {}

  void bind(std::vector<CrosstlMlxBuffer>& buffers) {
    auto vector = [](const char* name, const char* type, auto& values) {
      return CrosstlMlxBuffer{name, type, values.data(), values.size(), 0};
    };
    buffers.push_back({"x_batch_ndims", "int32", &x_batch_ndims, 1, 0});
    buffers.push_back(vector("x_shape", "int32", x_shape));
    buffers.push_back(vector("x_strides", "int64", x_strides));
    buffers.push_back({"w_batch_ndims", "int32", &w_batch_ndims, 1, 0});
    buffers.push_back(vector("w_shape", "int32", w_shape));
    buffers.push_back(vector("w_strides", "int64", w_strides));
    buffers.push_back(vector("s_strides", "int64", s_strides));
    if (biased) {
      buffers.push_back(vector("b_strides", "int64", b_strides));
    }
  }
};

void quantized_dispatch(
    const std::string& entry,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch,
    const char* label) {
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads,
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native ") + label + " failed: " + error);
  }
}

// quantized.cpp qmm_splitk's partitions: about 512 threadgroups, each
// partition whole BK-wide tiles and whole groups; 1 falls back to qmm.
int qmm_split_k(int M, int N, int K, int group_size) {
  const int64_t tiles = int64_t((N + 31) / 32) * ((M + 31) / 32);
  int split_k = int(std::max<int64_t>(1, 512 / tiles));
  const int k_align = group_size > 32 ? group_size : 32;
  split_k = std::min(split_k, K / k_align);
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  return split_k;
}

// quantized.cpp qmm: (32, 2, 2) threadgroups over 32 by 32 output tiles,
// batch parameters bound only for more than one matrix. Affine kernels bind
// biases; the mxfp4, mxfp8 and nvfp4 kernels bind uint8 scales and no
// biases, and qmm leaves fp_qmm_t's global scale slot unbound.
void dispatch_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    const char* mode,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K) {
  const uint64_t B = out.size() / M / N;
  const std::string entry = std::string(mode) + (transpose ? "_qmm_t_" : "_qmm_n_") +
      quantized_source(out.dtype()) + "_gs_" + std::to_string(group_size) + "_b_" +
      std::to_string(bits) + (transpose ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "") +
      (B > 1 ? "_batch_1" : "_batch_0");
  require_entry(entry);
  const uint32_t tiles_n = (uint32_t(N) + 31) / 32;
  const uint32_t tiles_m = (uint32_t(M) + 31) / 32;
  if (out.size() > max_matmul_elements || B > 65535 || tiles_n > 65535 || tiles_m > 65535) {
    throw std::invalid_argument("CrossTL quantized matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const char* storage = matmul_storage(out.dtype());
  int32_t k = K, n = N, m = M;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", biases ? storage : "uint8", data(scales), matmul_span(scales), 0},
  };
  if (biases) {
    buffers.push_back({"biases", storage, data(*biases), matmul_span(*biases), 0});
  }
  buffers.insert(
      buffers.end(),
      {
          {"x", storage, data(x), matmul_span(x), 0},
          {"y", storage, out.data<void>(), out.size(), 1},
          {"K", "int32", &k, 1, 0},
          {"N", "int32", &n, 1, 0},
          {"M", "int32", &m, 1, 0},
      });
  QuantizedBatchViews views =
      biases ? QuantizedBatchViews(x, w, scales, *biases) : QuantizedBatchViews(x, w, scales);
  if (B > 1) {
    views.bind(buffers);
  }
  const CrosstlMlxLaunch launch{{tiles_n, tiles_m, uint32_t(B)}, {32, 2, 2}};
  quantized_dispatch(entry, buffers, out.size(), launch, "quantized matrix product");
}

// quantized.cpp qmm_splitk: K in split_k partitions whose partial products
// land on a leading partition axis, then a strided sum over that axis.
void dispatch_qmm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    const char* mode,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    int split_k) {
  const char* storage = matmul_storage(out.dtype());
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, out.dtype(), nullptr, {});
  const std::string entry = std::string(mode) + "_qmm_t_splitk_" +
      quantized_source(out.dtype()) + "_gs_" + std::to_string(group_size) + "_b_" +
      std::to_string(bits) + (N % 32 == 0 ? "_alN_true" : "_alN_false");
  require_entry(entry);
  const uint32_t tiles_n = (uint32_t(N) + 31) / 32;
  const uint32_t tiles_m = (uint32_t(M) + 31) / 32;
  if (intermediate.size() > max_matmul_elements || tiles_n > 65535 || tiles_m > 65535 ||
      split_k > 65535) {
    throw std::invalid_argument("CrossTL quantized matmul launch exceeds its bounds.");
  }
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  int32_t k = K, n = N, m = M;
  int32_t k_partition_size = K / split_k;
  int32_t split_k_partition_stride = M * N;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", biases ? storage : "uint8", data(scales), matmul_span(scales), 0},
  };
  if (biases) {
    buffers.push_back({"biases", storage, data(*biases), matmul_span(*biases), 0});
  }
  buffers.insert(
      buffers.end(),
      {
          {"x", storage, data(x), matmul_span(x), 0},
          {"y", storage, intermediate.data<void>(), intermediate.size(), 1},
          {"K", "int32", &k, 1, 0},
          {"N", "int32", &n, 1, 0},
          {"M", "int32", &m, 1, 0},
          {"k_partition_size", "int32", &k_partition_size, 1, 0},
          {"split_k_partition_stride", "int32", &split_k_partition_stride, 1, 0},
      });
  const CrosstlMlxLaunch launch{{tiles_n, tiles_m, uint32_t(split_k)}, {32, 2, 2}};
  quantized_dispatch(entry, buffers, intermediate.size(), launch, "split-K quantized matrix product");
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(0)},
      {intermediate.strides(0)});
  dispatch_column_reduce(intermediate, out, plan, {0}, "sum", storage);
}
} // namespace

// quantized.cpp qvm_split_k: the K axis in 8 partitions (32 above 8192) whose
// partial products land on a partition axis before the matrices, then a
// strided sum over that axis. Affine kernels bind biases; the mxfp4, mxfp8
// and nvfp4 kernels bind uint8 scales and no biases.
void dispatch_qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    const char* mode,
    int group_size,
    int bits,
    int M,
    int N,
    int K) {
  const int split_k = K > 8192 ? 32 : 8;
  const char* storage = matmul_storage(out.dtype());
  // adjust_matrix_offsets moves a single weight batch axis by tid.z without
  // wrapping it, so 2-D weights (the partition axis alone) under batched x
  // read past the last partition in quantized.h.
  if (w.ndim() == 2 && out.size() / M / N > 1) {
    throw std::invalid_argument(
        "CrossTL split-K vector-matrix products of batched inputs and 2-D weights are not "
        "routed: qvm_split_k offsets the weights past their partitions.");
  }
  const int split_d = (K + split_k - 1) / split_k;
  const uint64_t B = out.size() / M / N * split_k;
  const int bn = std::min(group_size, 32) * 2;
  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }
  const int x_ndim = static_cast<int>(x_shape.size());
  int32_t x_batch_ndims = x_ndim - 2 + 1;
  int32_t w_batch_ndims = int32_t(w.ndim()) - 2 + 1;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_d);
  x_strides[x_ndim - 1] = split_d;
  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, int64_t(split_d) * w.shape(-1));
  s_strides.insert(s_strides.end() - 2, int64_t(split_d) * scales.shape(-1));
  Strides b_strides;
  if (biases) {
    b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, int64_t(split_d) * biases->shape(-1));
  }
  int32_t final_block_size = K - (split_k - 1) * split_d;
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, out.dtype(), nullptr, {});
  const std::string entry = std::string(mode) + "_qvm_split_k_" +
      quantized_source(out.dtype()) + "_gs_" + std::to_string(group_size) + "_b_" +
      std::to_string(bits) + "_spk_" + std::to_string(split_k);
  require_entry(entry);
  if (intermediate.size() > max_matmul_elements || B > 65535 || uint32_t(M) > 65535) {
    throw std::invalid_argument("CrossTL quantized matmul launch exceeds its bounds.");
  }
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  int32_t in_vec_size = split_d;
  int32_t out_vec_size = N;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  auto vector = [](const char* name, const char* type, const auto& values) {
    return CrosstlMlxBuffer{
        name, type, const_cast<void*>(static_cast<const void*>(values.data())),
        values.size(), 0};
  };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", biases ? storage : "uint8", data(scales), matmul_span(scales), 0},
  };
  if (biases) {
    buffers.push_back({"biases", storage, data(*biases), matmul_span(*biases), 0});
  }
  buffers.insert(
      buffers.end(),
      {
          {"x", storage, data(x), matmul_span(x), 0},
          {"y", storage, intermediate.data<void>(), intermediate.size(), 1},
          {"in_vec_size", "int32", &in_vec_size, 1, 0},
          {"out_vec_size", "int32", &out_vec_size, 1, 0},
          {"x_batch_ndims", "int32", &x_batch_ndims, 1, 0},
          vector("x_shape", "int32", x_shape),
          vector("x_strides", "int64", x_strides),
          {"w_batch_ndims", "int32", &w_batch_ndims, 1, 0},
          vector("w_shape", "int32", w_shape),
          vector("w_strides", "int64", w_strides),
          vector("s_strides", "int64", s_strides),
      });
  if (biases) {
    buffers.push_back(vector("b_strides", "int64", b_strides));
  }
  buffers.push_back({"final_block_size", "int32", &final_block_size, 1, 0});
  const CrosstlMlxLaunch launch{
      {uint32_t(M), uint32_t((N + bn - 1) / bn), uint32_t(B)}, {32, 2, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
      intermediate.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native quantized split-K failed: ") + error);
  }
  const int axis = int(intermediate.ndim()) - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  dispatch_column_reduce(intermediate, out, plan, {axis}, "sum", storage);
}

// quantized.cpp dispatch_qmv, qvm and qvm_split_k for mxfp4, mxfp8 and nvfp4
// weights: fp_quantized.metal's kernels, with uint8 scales and no biases.
// QuantizedMatmul passes no global scale, so qmv_wide serves two or more
// vectors on every device. Products of at least the vector limit's rows take
// dispatch_qmm and dispatch_qmm_split_k instead.
void dispatch_fp_quantized_matvec(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    const char* mode,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K) {
  if (!transpose && K >= 1024) {
    dispatch_qvm_split_k(x, w, scales, std::nullopt, out, mode, group_size, bits, M, N, K);
    return;
  }
  const uint64_t B = out.size() / M / N;
  const bool quad = transpose && (K == 64 || K == 128);
  const bool wide = transpose && !quad && M >= 2;
  const bool fast = N % 8 == 0 && K % ((32 / bits) * 2 * 32) == 0;
  // qmv_wide's vecs_per_tg: the fewest tiles of at most 5 vectors.
  const int vector_tiles = (M + 4) / 5;
  const int vectors = (M + vector_tiles - 1) / vector_tiles;
  const char* kernel = !transpose ? "_qvm_"
      : quad                      ? "_qmv_quad_"
      : wide                      ? "_qmv_wide_"
      : fast                      ? "_qmv_fast_"
                                  : "_qmv_";
  const std::string entry = std::string(mode) + kernel + quantized_source(out.dtype()) +
      "_gs_" + std::to_string(group_size) + "_b_" + std::to_string(bits) +
      (quad ? "_d_" + std::to_string(K) : "") +
      (wide ? "_nv_" + std::to_string(vectors) + "_kl_16" : "") +
      (B > 1 ? "_batch_1" : "_batch_0");
  require_entry(entry);
  // qvm: 2 simdgroups of min(group_size, 32) columns; qmv_quad: 8 quads of 8
  // rows; qmv_wide: 2 simdgroups of 2 rows; qmv: 2 simdgroups of 4 rows.
  const int rows = !transpose ? 2 * std::min(group_size, 32) : quad ? 64 : wide ? 4 : 8;
  const uint32_t tiles = (uint32_t(N) + rows - 1) / rows;
  const uint32_t columns = wide ? uint32_t(vector_tiles) : uint32_t(M);
  if (out.size() > max_matmul_elements || B > 65535 || columns > 65535) {
    throw std::invalid_argument("CrossTL quantized matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const char* storage = matmul_storage(out.dtype());
  int32_t in_vec_size = K;
  int32_t out_vec_size = N;
  int32_t vector_count = M;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", "uint8", data(scales), matmul_span(scales), 0},
      {"x", storage, data(x), matmul_span(x), 0},
      {"y", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_vec_size, 1, 0},
      {"out_vec_size", "int32", &out_vec_size, 1, 0},
  };
  if (wide) {
    buffers.push_back({"M", "int32", &vector_count, 1, 0});
  }
  QuantizedBatchViews views(x, w, scales);
  if (B > 1) {
    views.bind(buffers);
  }
  const CrosstlMlxLaunch launch{{columns, tiles, uint32_t(B)}, {32, quad ? 1u : 2u, 1}};
  quantized_dispatch(
      entry, buffers, out.size(), launch, "floating-point quantized matmul");
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const bool affine = mode_ == QuantizationMode::Affine;
  // fp_quantized.metal's modes and their group sizes and bits.
  const char* fp_mode = mode_ == QuantizationMode::Mxfp4 ? "mxfp4"
      : mode_ == QuantizationMode::Mxfp8                 ? "mxfp8"
      : mode_ == QuantizationMode::Nvfp4                 ? "nvfp4"
                                                         : nullptr;
  if (affine ? inputs.size() != 4
             : fp_mode == nullptr || inputs.size() != 3 ||
                 group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
                 bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)) {
    throw std::invalid_argument(
        "CrossTL quantized matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != uint32 ||
      inputs[2].dtype() != (affine ? dtype : uint8) ||
      (affine && inputs[3].dtype() != dtype)) {
    throw std::invalid_argument("CrossTL quantized matmul requires matching floating types.");
  }
  array x = matmul_operand(inputs[0]);
  array w = matmul_operand(inputs[1]);
  array scales = matmul_operand(inputs[2]);
  // quantized.cpp QuantizedMatmul::eval_gpu's shapes.
  const bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  if (x.ndim() < 1 || w.ndim() < 2 || (!non_batched && x.ndim() < 2)) {
    throw std::invalid_argument("CrossTL quantized matmul operand ranks are invalid.");
  }
  const int K = x.shape(-1);
  const int M = non_batched ? x.size() / K : x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0 || out.size() % (uint64_t(M) * N) != 0) {
    throw std::invalid_argument("CrossTL quantized matmul shape is empty or inconsistent.");
  }
  const uint64_t B = out.size() / M / N;
  // get_qmv_batch_limit for a device without an Apple GPU generation.
  const int vector_limit = !transpose_ ? 4
      : K <= 2048 && N <= 2048        ? 18
      : K <= 4096 && N <= 4096        ? 12
                                      : 10;
  if (M >= vector_limit) {
    // Every mode's matrix-matrix route: qmm_splitk for single matrices with
    // transposed weights, which falls back to qmm when K admits no split;
    // there is no NAX unit for qmm_nax.
    std::optional<array> qmm_biases;
    if (affine) {
      qmm_biases = matmul_operand(inputs[3]);
    }
    const char* mode = affine ? "affine" : fp_mode;
    const int split_k = transpose_ && B == 1 ? qmm_split_k(M, N, K, group_size_) : 1;
    if (split_k > 1) {
      dispatch_qmm_split_k(
          x, w, scales, qmm_biases, out, mode, group_size_, bits_, M, N, K, split_k);
    } else {
      dispatch_qmm(
          x, w, scales, qmm_biases, out, mode, transpose_, group_size_, bits_, M, N, K);
    }
    return;
  }
  if (!affine) {
    dispatch_fp_quantized_matvec(
        x, w, scales, out, fp_mode, transpose_, group_size_, bits_, M, N, K);
    return;
  }
  array biases = matmul_operand(inputs[3]);
  if (!transpose_ && K >= 1024) {
    dispatch_qvm_split_k(x, w, scales, biases, out, "affine", group_size_, bits_, M, N, K);
    return;
  }
  // dispatch_qmv without qmv_wide, which MLX selects from generation 15, and
  // qvm for untransposed weights.
  const bool power_of_two = bits_ == 2 || bits_ == 4 || bits_ == 8;
  const bool quad = transpose_ && (K == 64 || K == 128) && power_of_two;
  const int pack_factor = bits_ == 3 || bits_ == 5 ? 8 : bits_ == 6 ? 4 : 32 / bits_;
  const bool fast = N % 8 == 0 && K % (pack_factor * (bits_ == 2 ? 1 : 2) * 32) == 0;
  const char* kernel = !transpose_ ? "affine_qvm_"
      : quad                       ? "affine_qmv_quad_"
      : fast                       ? "affine_qmv_fast_"
                                   : "affine_qmv_";
  const std::string entry = std::string(kernel) + quantized_source(dtype) + "_gs_" +
      std::to_string(group_size_) + "_b_" + std::to_string(bits_) +
      (quad ? "_d_" + std::to_string(K) : "") + (B > 1 ? "_batch_1" : "_batch_0");
  require_entry(entry);
  if (out.size() > max_matmul_elements || B > 65535 || uint32_t(M) > 65535) {
    throw std::invalid_argument("CrossTL quantized matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  // qvm covers 2 simdgroups of min(group_size, 32) columns per threadgroup.
  const int rows = !transpose_ ? 2 * std::min(group_size_, 32) : quad ? 64 : 8;
  const uint32_t tiles = (uint32_t(N) + rows - 1) / rows;
  const char* storage = matmul_storage(dtype);
  int32_t in_vec_size = K;
  int32_t out_vec_size = N;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", storage, data(scales), matmul_span(scales), 0},
      {"biases", storage, data(biases), matmul_span(biases), 0},
      {"x", storage, data(x), matmul_span(x), 0},
      {"y", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_vec_size, 1, 0},
      {"out_vec_size", "int32", &out_vec_size, 1, 0},
  };
  // add_strides_and_shapes binds these only when B > 1.
  QuantizedBatchViews views(x, w, scales, biases);
  if (B > 1) {
    views.bind(buffers);
  }
  const CrosstlMlxLaunch launch{
      {uint32_t(M), tiles, uint32_t(B)}, {32, quad ? 1u : 2u, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), out.size(),
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native quantized matmul failed: ") + error);
  }
}

namespace {
// A row-major copy, as contiguous_copy_gpu makes.
array contiguous_copy(const array& in) {
  array copy(in.shape(), in.dtype(), nullptr, {});
  dispatch_copy(in, copy);
  return copy;
}

// matmul.cpp check_transpose. The host also copies views with negative
// strides, which it never binds.
std::tuple<bool, int64_t, array> gemv_operand(const array& in, bool is_vector) {
  const auto stx = in.strides()[in.ndim() - 2];
  const auto sty = in.strides()[in.ndim() - 1];
  const bool nonnegative = std::all_of(
      in.strides().begin(), in.strides().end(), [](int64_t stride) { return stride >= 0; });
  if (nonnegative && sty == 1 && (!is_vector || stx == in.shape(-1))) {
    return std::make_tuple(false, stx, in);
  }
  if (nonnegative && stx == 1 && (!is_vector || sty == in.shape(-2))) {
    return std::make_tuple(true, sty, in);
  }
  return std::make_tuple(false, int64_t(in.shape(-1)), contiguous_copy(in));
}

// Run one translated split-K kernel; ``threads`` is its output extent.
void dispatch_splitk_kernel(
    const std::string& entry,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads, &launch,
      error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native split-K matmul failed: ") + error);
  }
}

// matmul.cpp steel_gemm_splitk_axpby without C: gemm_splitk writes one float32
// partial product per K partition into C_split and gemm_splitk_accum sums them
// into the output. matmul.cpp launches the accumulation with dispatch_threads
// over exactly N x M threads; the kernel reads only thread_position_in_grid,
// so one-thread workgroups cover the same threads.
void dispatch_steel_gemm_splitk(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int64_t lda,
    int64_t ldb,
    bool transpose_a,
    bool transpose_b) {
  const int _tm = (M + 32 - 1) / 32;
  const int _tn = (N + 32 - 1) / 32;
  const int _tk = K / 16;
  const int bm = M < 40 ? 16 : 32;
  const int bn = N < 40 ? 16 : 32;
  constexpr int bk = 16, wm = 2, wn = 2;
  const int split_k_partitions = std::min(std::max(2, next_power_of_2(_tk / (_tm * _tn))), 32);
  const int64_t split_k_partition_stride = int64_t(M) * N;
  const int gemm_k_iterations = (K / bk) / split_k_partitions;
  const int split_k_partition_size = gemm_k_iterations * bk;
  const int tn = (N + bn - 1) / bn;
  const int tm = (M + bm - 1) / bm;
  if (split_k_partition_stride * split_k_partitions > int64_t(max_matmul_elements) ||
      lda < 0 || lda > int64_t(max_matmul_elements) || ldb < 0 ||
      ldb > int64_t(max_matmul_elements) || N > 65535 || M > 65535) {
    throw std::invalid_argument("CrossTL split-K matmul launch exceeds its bounds.");
  }
  const bool mn_aligned = M % bm == 0 && N % bn == 0;
  const bool k_aligned = K % bk == 0;
  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("steel_gemm_splitk_") + (transpose_a ? "t" : "n") +
      (transpose_b ? "t" : "n") + "_" + storage + "_float32_bm" + std::to_string(bm) + "_bn" +
      std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" +
      std::to_string(wn) + "_MN_" + (mn_aligned ? "t" : "n") + "aligned_K_" +
      (k_aligned ? "t" : "n") + "aligned";
  const std::string accum = std::string("steel_gemm_splitk_accum_") + storage + "_float32";
  require_entry(entry);
  require_entry(accum);
  array C_split({split_k_partitions, M, N}, float32, nullptr, {});
  C_split.set_data(allocator::malloc(C_split.nbytes()));
  out.set_data(allocator::malloc(out.nbytes()));
  // GEMMSpiltKParams in params.h order.
  std::vector<int32_t> params = {
      M,
      N,
      K,
      int32_t(lda),
      int32_t(ldb),
      N,
      tn,
      tm,
      split_k_partitions,
      int32_t(split_k_partition_stride),
      split_k_partition_size,
      0,
      gemm_k_iterations};
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(a), matmul_span(a), 0},
      {"B", storage, data(b), matmul_span(b), 0},
      {"C", "float32", C_split.data<void>(), C_split.size(), 1},
      {"params", "int32", params.data(), params.size(), 0},
  };
  dispatch_splitk_kernel(
      entry, buffers, C_split.size(),
      CrosstlMlxLaunch{
          {uint32_t(tn), uint32_t(tm), uint32_t(split_k_partitions)},
          {32, uint32_t(wn), uint32_t(wm)}});
  int32_t k_partitions = split_k_partitions;
  int32_t partition_stride = int32_t(split_k_partition_stride);
  int32_t ldd = N;
  std::vector<CrosstlMlxBuffer> accum_buffers = {
      {"C_split", "float32", C_split.data<void>(), C_split.size(), 0},
      {"D", storage, out.data<void>(), out.size(), 1},
      {"k_partitions", "int32", &k_partitions, 1, 0},
      {"partition_stride", "int32", &partition_stride, 1, 0},
      {"ldd", "int32", &ldd, 1, 0},
  };
  dispatch_splitk_kernel(
      accum, accum_buffers, out.size(),
      CrosstlMlxLaunch{{uint32_t(N), uint32_t(M), 1}, {1, 1, 1}});
}

// matmul.cpp steel_matmul_axpby without C. The host reports no Apple GPU
// architecture and no NAX unit, so GEMM_TPARAM_MACRO takes its medium-device
// tile and the split-K case uses the threshold of 1024 tiles.
void dispatch_steel_gemm(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    size_t batch_size_out,
    int64_t lda,
    int64_t ldb,
    bool transpose_a,
    bool transpose_b,
    Shape batch_shape,
    const Strides& A_batch_stride,
    const Strides& B_batch_stride) {
  const int64_t split_tiles = int64_t((M + 16 - 1) / 16) * ((N + 16 - 1) / 16);
  if (batch_size_out == 1 && split_tiles <= 1024 && K / 16 >= 8 && K >= std::max(M, N)) {
    dispatch_steel_gemm_splitk(a, b, out, M, N, K, lda, ldb, transpose_a, transpose_b);
    return;
  }
  constexpr int bm = 64, bn = 64, bk = 16, wm = 2, wn = 2;
  const bool has_batch = batch_shape.size() > 1;
  auto flag = [](bool value) { return value ? "t" : "n"; };
  const char* storage = matmul_storage(out.dtype());
  // steel_matmul_regular_axpby's hash name: the kernel and its constants.
  const std::string entry = std::string("steel_gemm_fused_") + (transpose_a ? "t" : "n") +
      (transpose_b ? "t" : "n") + "_" + storage + "_" + storage + "_bm" + std::to_string(bm) +
      "_bn" + std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" + std::to_string(wm) +
      "_wn" + std::to_string(wn) + "_has_batch_" + flag(has_batch) +
      "_use_out_source_n_do_axpby_n_align_M_" + flag(M % bm == 0) + "_align_N_" +
      flag(N % bn == 0) + "_align_K_" + flag(K % bk == 0);
  require_entry(entry);
  const uint64_t tiles_n = (uint64_t(N) + bn - 1) / bn;
  const uint64_t tiles_m = (uint64_t(M) + bm - 1) / bm;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || tiles_n > 65535 ||
      tiles_m > 65535 || lda < 0 || lda > int64_t(max_matmul_elements) || ldb < 0 ||
      ldb > int64_t(max_matmul_elements) || batch_shape.empty() || batch_shape.size() > 16 ||
      A_batch_stride.size() != batch_shape.size() ||
      B_batch_stride.size() != batch_shape.size()) {
    throw std::invalid_argument("CrossTL matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  Strides batch_strides = A_batch_stride;
  batch_strides.insert(batch_strides.end(), B_batch_stride.begin(), B_batch_stride.end());
  // GEMMParams in params.h order, each widened to int64.
  std::vector<int64_t> params = {
      M,
      N,
      K,
      lda,
      ldb,
      N,
      int64_t(tiles_n),
      int64_t(tiles_m),
      A_batch_stride.back(),
      B_batch_stride.back(),
      int64_t(M) * N,
      0,
      K / bk,
      int64_t(batch_shape.size())};
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  // matmul.cpp binds batch_shape and batch_strides only with has_batch; the
  // kernel reads neither otherwise.
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(a), matmul_span(a), 0},
      {"B", storage, data(b), matmul_span(b), 0},
      {"D", storage, out.data<void>(), out.size(), 1},
      {"params", "int64", params.data(), params.size(), 0},
      {"batch_shape", "int32", batch_shape.data(), batch_shape.size(), 0},
      {"batch_strides", "int64", batch_strides.data(), batch_strides.size(), 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(tiles_n), uint32_t(tiles_m), uint32_t(batch_size_out)},
      {32, uint32_t(wn), uint32_t(wm)}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), out.size(),
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native matmul failed: ") + error);
  }
}
} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const auto dtype = out.dtype();
  if (inputs.size() != 2 || (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != dtype) {
    throw std::invalid_argument("CrossTL matmul supports float32, float16 and bfloat16 only.");
  }
  const auto& a_pre = inputs[0];
  const auto& b_pre = inputs[1];
  if (a_pre.ndim() < 2 || a_pre.ndim() != b_pre.ndim() || out.ndim() != a_pre.ndim()) {
    throw std::invalid_argument("CrossTL matmul operand ranks are invalid.");
  }
  // matmul.cpp fills the output with zeros when either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    if (out.size() == 0) {
      out.set_data(allocator::malloc(out.nbytes()));
      return;
    }
    array zero(0, dtype);
    array zeros(out.shape(), dtype, nullptr, {});
    const bool single = out.size() == 1;
    zeros.copy_shared_buffer(zero, Strides(out.ndim(), 0), {single, single, single}, 1);
    dispatch_copy(zeros, out);
    return;
  }
  int M = a_pre.shape(-2);
  const int N = b_pre.shape(-1);
  const int K = a_pre.shape(-1);
  auto [a_transposed, a_cols, a] = gemv_operand(a_pre, M == 1);
  auto [b_transposed, b_cols, b] = gemv_operand(b_pre, N == 1);
  auto [batch_shape, A_batch_stride, B_batch_stride] = collapse_batches(a, b);
  auto batch_size_out = out.size() / (size_t(M) * size_t(N));
  // Collapse batches into M as matmul.cpp does.
  if (batch_size_out > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && A_batch_stride.back() == M * K &&
      B_batch_stride.back() == 0) {
    M *= batch_shape.back();
    batch_size_out = 1;
    A_batch_stride = {0};
    B_batch_stride = {0};
    batch_shape = {1};
  }
  if (M == 1 && N == 1 && batch_size_out == 1 && a.flags().row_contiguous &&
      b.flags().row_contiguous) {
    throw std::invalid_argument("CrossTL matmul does not route single dot products yet.");
  }
  // MLX takes gemv_wide only from Apple GPU generation 15, which the host
  // does not report.
  if (std::min(M, N) != 1) {
    dispatch_steel_gemm(
        a, b, out, M, N, K, batch_size_out, a_cols, b_cols, a_transposed, b_transposed,
        batch_shape, A_batch_stride, B_batch_stride);
    return;
  }
  // gemv_axbpy without axpby.
  const bool is_b_matrix = N != 1;
  const array& mat = is_b_matrix ? b : a;
  const array& vec = is_b_matrix ? a : b;
  const bool transpose_mat = is_b_matrix ? !b_transposed : a_transposed;
  const int out_vector_len = is_b_matrix ? N : M;
  const int64_t mat_ld = is_b_matrix ? b_cols : a_cols;
  auto& vector_batch_stride = is_b_matrix ? A_batch_stride : B_batch_stride;
  auto& matrix_batch_stride = is_b_matrix ? B_batch_stride : A_batch_stride;
  int tm = 4, tn = 4, sm = 1, sn = 32, bm = 1, bn = 1, n_out_per_tgp;
  if (transpose_mat) {
    if (K >= 8192 && out_vector_len >= 2048) {
      sm = 4;
      sn = 8;
    } else {
      sm = 8;
      sn = 4;
    }
    bn = out_vector_len >= 2048 ? 16 : out_vector_len >= 512 ? 4 : 2;
    tn = out_vector_len < tn ? 1 : tn;
    n_out_per_tgp = bn * sn * tn;
  } else {
    bm = out_vector_len >= 4096 ? 8 : 4;
    if (K <= 64) {
      bm = 1;
      sm = 8;
      sn = 4;
    } else if (K >= 16 * out_vector_len) {
      bm = 1;
      bn = 8;
    }
    tm = out_vector_len < tm ? 1 : tm;
    n_out_per_tgp = bm * sm * tm;
  }
  const char* storage = matmul_storage(dtype);
  const std::string entry = std::string(transpose_mat ? "gemv_t_" : "gemv_") + storage +
      "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_sm" + std::to_string(sm) +
      "_sn" + std::to_string(sn) + "_tm" + std::to_string(tm) + "_tn" + std::to_string(tn) +
      "_nc" + (batch_shape.size() == 1 ? "0" : "1") + "_axpby0";
  require_entry(entry);
  const uint64_t n_tgp = (uint64_t(out_vector_len) + n_out_per_tgp - 1) / n_out_per_tgp;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || n_tgp > 65535 ||
      mat_ld < 0 || mat_ld > int64_t(max_matmul_elements) || batch_shape.size() > 16) {
    throw std::invalid_argument("CrossTL matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  int32_t in_vec_size = K;
  int32_t out_vec_size = out_vector_len;
  int32_t marix_ld = int32_t(mat_ld);
  int32_t batch_ndim = int32_t(batch_shape.size());
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"mat", storage, data(mat), matmul_span(mat), 0},
      {"in_vec", storage, data(vec), matmul_span(vec), 0},
      {"out_vec", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_vec_size, 1, 0},
      {"out_vec_size", "int32", &out_vec_size, 1, 0},
      {"marix_ld", "int32", &marix_ld, 1, 0},
      {"batch_ndim", "int32", &batch_ndim, 1, 0},
      {"batch_shape", "int32", batch_shape.data(), batch_shape.size(), 0},
      {"vector_batch_stride", "int64", vector_batch_stride.data(),
       vector_batch_stride.size(), 0},
      {"matrix_batch_stride", "int64", matrix_batch_stride.data(),
       matrix_batch_stride.size(), 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(n_tgp), 1, uint32_t(batch_size_out)}, {32, uint32_t(bn), uint32_t(bm)}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), out.size(),
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native matmul failed: ") + error);
  }
}
} // namespace mlx::core

namespace mlx::core::fast {

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  const bool affine = mode_ == QuantizationMode::Affine;
  const bool nvfp4 = mode_ == QuantizationMode::Nvfp4;
  const bool block = mode_ == QuantizationMode::Mxfp4 ||
      mode_ == QuantizationMode::Mxfp8 || nvfp4;
  const bool global_scale = !affine && inputs.size() == (dequantize_ ? 3 : 2);
  const bool parameters = affine
      ? (group_size_ == 32 || group_size_ == 64 || group_size_ == 128) &&
          (bits_ == 2 || bits_ == 3 || bits_ == 4 || bits_ == 5 || bits_ == 6 || bits_ == 8)
      : block && group_size_ == (nvfp4 ? 16 : 32) &&
          bits_ == (mode_ == QuantizationMode::Mxfp8 ? 8 : 4);
  const size_t input_count = dequantize_ ? (affine ? 3 : 2) : 1;
  if (!parameters || (global_scale && !nvfp4) ||
      inputs.size() != input_count + size_t(global_scale) ||
      outputs.size() != (dequantize_ ? 1 : affine ? 3 : 2)) {
    throw std::invalid_argument("CrossTL quantization parameters do not match a pinned entry.");
  }
  const auto dtype = dequantize_ ? outputs[0].dtype() : inputs[0].dtype();
  if (dtype != float32 && dtype != float16 && dtype != bfloat16) {
    throw std::invalid_argument("CrossTL quantization requires float32, float16 or bfloat16.");
  }
  const auto elements = dequantize_ ? outputs[0].size() : inputs[0].size();
  if (elements % group_size_ != 0) {
    throw std::invalid_argument("CrossTL quantization requires complete groups.");
  }
  const uint64_t groups = elements / group_size_;
  const uint64_t packed_per_group = uint64_t(group_size_) * bits_ / 8;
  const auto& packed = dequantize_ ? inputs[0] : outputs[0];
  const auto& scales = dequantize_ ? inputs[1] : outputs[1];
  if (packed.dtype() != uint32 || packed.size() != groups * packed_per_group / 4 ||
      scales.dtype() != (affine ? dtype : uint8) || scales.size() != groups) {
    throw std::invalid_argument("CrossTL quantization buffer shapes or types differ.");
  }
  if (affine) {
    const auto& biases = dequantize_ ? inputs[2] : outputs[2];
    if (biases.dtype() != dtype || biases.size() != groups || scales.shape() != biases.shape()) {
      throw std::invalid_argument("CrossTL affine quantization bias layout differs.");
    }
  }
  if (global_scale && (inputs.back().dtype() != float32 || inputs.back().size() != 1)) {
    throw std::invalid_argument("CrossTL block quantization requires one float32 global scale.");
  }
  const char* storage = dtype == float32 ? "float32" : dtype == float16 ? "float16" : "bfloat16";
  const char* source = dtype == float32 ? "float" : dtype == float16 ? "float16_t" : "bfloat16_t";
  const char* mode = affine ? "affine" : nvfp4 ? "nvfp4" :
      mode_ == QuantizationMode::Mxfp8 ? "mxfp8" : "mxfp4";
  const std::string entry = std::string(mode) + (dequantize_ ? "_dequantize_" : "_quantize_") +
      source + "_gs_" + std::to_string(group_size_) + "_b_" + std::to_string(bits_) +
      (affine ? "" : global_scale ? "_hgs_true" : "_hgs_false");
  require_entry(entry);
  for (auto& output : outputs) {
    output.set_data(allocator::malloc(output.nbytes()));
  }
  if (elements == 0) {
    return;
  }
  std::vector<array> dense;
  dense.reserve(inputs.size());
  for (const auto& input : inputs) {
    dense.push_back(dense_input(input));
  }
  const uint64_t itemsize = dequantize_ ? outputs[0].itemsize() : inputs[0].itemsize();
  const int pack_factor = bits_ == 3 || bits_ == 5 ? 8 : bits_ == 6 ? 4 : 8 / bits_;
  const uint32_t local_size = dequantize_ ? group_size_ / pack_factor : std::min(group_size_, 32);
  // Each complete group is independent. Batch without adding out-of-range lanes.
  for (uint64_t first = 0; first < groups;) {
    const uint64_t count = std::min<uint64_t>(groups - first, 65535);
    auto pointer = [](const array& value, uint64_t offset) {
      return const_cast<uint8_t*>(value.data<uint8_t>()) + offset;
    };
    std::vector<CrosstlMlxBuffer> buffers = {
        {"w", dequantize_ ? "uint8" : storage,
         pointer(dense[0], first * (dequantize_ ? packed_per_group : group_size_ * itemsize)),
         count * (dequantize_ ? packed_per_group : group_size_), 0},
        {"out", dequantize_ ? storage : "uint8",
         pointer(outputs[0], first * (dequantize_ ? group_size_ * itemsize : packed_per_group)),
         count * (dequantize_ ? group_size_ : packed_per_group), 1},
        {"scales", affine ? storage : "uint8",
         pointer(dequantize_ ? dense[1] : outputs[1], first * (affine ? itemsize : 1)), count,
         dequantize_ ? uint32_t(0) : uint32_t(1)},
    };
    if (affine) {
      buffers.push_back({"biases", storage,
          pointer(dequantize_ ? dense[2] : outputs[2], first * itemsize), count,
          dequantize_ ? uint32_t(0) : uint32_t(1)});
    } else if (global_scale) {
      buffers.push_back({"global_scale", "float32", pointer(dense.back(), 0), 1, 0});
    }
    const CrosstlMlxLaunch launch{{static_cast<uint32_t>(count), 1, 1}, {local_size, 1, 1}};
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
        count * group_size_, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native quantization failed: ") + error);
    }
    first += count;
  }
}

} // namespace mlx::core::fast
