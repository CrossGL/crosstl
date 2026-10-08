#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "mlx/allocator.h"
#include "mlx/backend/common/reduce.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/no_gpu/crosstl_dispatch.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

namespace {
std::atomic<CrosstlMlxDispatch> dispatch_callback{nullptr};
std::atomic<CrosstlMlxEntryAvailable> entry_available_callback{nullptr};
std::mutex registration_mutex;

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
  const char* dtype = type == mlx::core::float16 ? "float16" : storage_type(type);
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
  if (in.data_size() > 65535) {
    throw std::invalid_argument(
        "CrossTL unary dispatch supports at most 65535 stored elements.");
  }
  const uint64_t bytes = in.data_size() * in.itemsize();
  if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
      bytes > in.buffer_size() - uint64_t(in.offset())) {
    throw std::invalid_argument("CrossTL unary input exceeds its allocation.");
  }
  mlx::core::set_unary_output_data(in, out);
  uint32_t size = static_cast<uint32_t>(in.data_size());
  std::string entry = std::string("v_") + operation + dtype + dtype;
  CrosstlMlxBuffer buffers[] = {
      {"in", dtype, in.data<void>(), size, 0},
      {"out", dtype, out.data<void>(), size, 1},
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
       in.dtype() != mlx::core::uint32 && in.dtype() != mlx::core::bool_ &&
       in.dtype() != mlx::core::int64 && in.dtype() != mlx::core::uint64)) {
    throw std::invalid_argument(
        "CrossTL copying layouts require matching float16, float32, int32, uint32, int64, uint64 or bool arrays.");
  }
  if (in.size() > 65535 || in.ndim() > 64 ||
      dst_strides.size() != in.ndim() ||
      out.size() > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument(
        "CrossTL copy supports up to 65535 input elements and 64 axes.");
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
  int64_t low = 0, high = 0;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] == 1) {
      src_strides[axis] = 0;
    }
    if (src_strides[axis] < -65535 || src_strides[axis] > 65535) {
      throw std::invalid_argument("CrossTL copy source stride exceeds 65535.");
    }
    const int64_t extent = (int64_t(shape[axis]) - 1) * src_strides[axis];
    low += std::min<int64_t>(extent, 0);
    high += std::max<int64_t>(extent, 0);
  }
  const int64_t span = high - low + 1;
  const int64_t item_size = in.itemsize();
  if (span > 65535 || in.offset() < 0 || in.offset() % item_size != 0) {
    throw std::invalid_argument("CrossTL copy source span exceeds its bounds.");
  }
  const uint64_t origin = in.offset() / item_size;
  const uint64_t capacity = in.buffer_size() / item_size;
  if (origin >= capacity || uint64_t(-low) > origin ||
      uint64_t(high) >= capacity - origin) {
    throw std::invalid_argument("CrossTL copy source view exceeds its allocation.");
  }
  int64_t destination_low = dst_offset, destination_high = dst_offset;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (dst_strides[axis] < -int64_t(std::numeric_limits<int32_t>::max()) ||
        dst_strides[axis] > std::numeric_limits<int32_t>::max()) {
      throw std::invalid_argument("CrossTL copy destination stride is invalid.");
    }
    const int64_t extent = (int64_t(shape[axis]) - 1) * dst_strides[axis];
    destination_low += std::min<int64_t>(extent, 0);
    destination_high += std::max<int64_t>(extent, 0);
  }
  if (dst_offset < 0 || destination_low < 0 || destination_high >= out.size() ||
      out.offset() < 0 || uint64_t(out.offset()) > out.buffer_size() ||
      out.nbytes() > out.buffer_size() - uint64_t(out.offset())) {
    throw std::invalid_argument("CrossTL copy destination exceeds its allocation.");
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int64_t src_offset = -low;
  // Copy storage words to preserve NaN payloads, subnormals and signed zero.
  const bool boolean = in.dtype() == mlx::core::bool_;
  const bool wide = in.dtype() == mlx::core::int64 || in.dtype() == mlx::core::uint64;
  const char* dtype = in.dtype() == mlx::core::float16 ? "float16" :
      wide ? storage_type(in.dtype()) : boolean ? "bool_" : "uint32";
  CrosstlMlxBuffer buffers[] = {
      {"src", dtype, const_cast<uint8_t*>(in.data<uint8_t>() + low * item_size), uint64_t(span), 0},
      {"dst", dtype, out.data<void>(), out.size(),
       preserve ? CROSTL_MLX_BUFFER_INOUT : uint32_t(1)},
      {"src_shape", "int32", shape.data(), uint64_t(ndim), 0},
      {"src_strides", "int64", src_strides.data(), uint64_t(ndim), 0},
      {"dst_strides", "int64", dst_strides.data(), uint64_t(ndim), 0},
      {"ndim", "int32", &ndim, 1, 0},
      {"src_offset", "int64", &src_offset, 1, 0},
      {"dst_offset", "int64", &dst_offset, 1, 0},
  };
  char error[2048] = {};
  uint32_t slices = 1;
  for (size_t axis = 0; axis + 2 < shape.size(); ++axis) {
    slices *= static_cast<uint32_t>(shape[axis]);
  }
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>((shape.back() + 1) / 2),
       static_cast<uint32_t>(shape[shape.size() - 2]), slices},
      {1, 1, 1}};
  const std::string entry = std::string("ggn2_dynamic_copy") + dtype + dtype;
  int status = dispatch_callback.load()(
      entry.c_str(),
      buffers, 8, in.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native copy failed: ") + error);
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

uint64_t gather_span(const mlx::core::array& in) {
  if (in.ndim() > 64 || in.size() == 0 || in.size() > 65535) {
    throw std::invalid_argument("CrossTL gather input shape exceeds its bounds.");
  }
  int64_t span = 1;
  for (size_t axis = 0; axis < in.ndim(); ++axis) {
    const auto stride = in.strides()[axis];
    if (stride < 0 || stride > 65535) {
      throw std::invalid_argument("CrossTL gather requires bounded nonnegative storage strides.");
    }
    span += (int64_t(in.shape(axis)) - 1) * stride;
  }
  if (span > 65535 || in.offset() < 0 ||
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
  auto dense = dense_input(updates);
  std::vector<int32_t> shape(dense.shape().begin(), dense.shape().end());
  std::vector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (size_t axis = shape.size(); axis-- > 0;) {
    strides[axis] = stride;
    stride *= shape[axis];
  }
  if (shape.empty()) {
    shape = {1};
    strides = {1};
    output_strides = {1};
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int64_t size = dense.size();
  const char* dtype = storage_type(out.dtype());
  CrosstlMlxBuffer buffers[] = {
      {"updates", dtype, dense.data<void>(), dense.size(), 0},
      {"out", dtype, out.data<void>(), out.size(), CROSTL_MLX_BUFFER_INOUT},
      {"update_shape", "int32", shape.data(), uint64_t(ndim), 0},
      {"update_strides", "int64", strides.data(), uint64_t(ndim), 0},
      {"update_ndim", "int32", &ndim, 1, 0},
      {"update_size", "int64", &size, 1, 0},
      {"output_strides", "int64", output_strides.data(), uint64_t(ndim), 0},
      {"output_offset", "int64", &output_offset, 1, 0},
  };
  char error[2048] = {};
  const auto launch = elementwise_launch(dense.size());
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, 8, dense.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native slice update failed: ") + error);
  }
}

void dispatch_all_reduce(
    const mlx::core::array& in,
    mlx::core::array& out,
    const std::string& operation,
    const char* dtype) {
  uint64_t size = in.size();
  const uint32_t rows = size <= 4096 ? 1 : 128;
  uint64_t row_size = (size + rows - 1) / rows;
  const uint32_t width = static_cast<uint32_t>(
      std::min<uint64_t>(1024, ((row_size + 127) / 128) * 32));
  if (out.size() != rows || in.data_size() != size || in.offset() < 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      in.nbytes() > in.buffer_size() - uint64_t(in.offset())) {
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

void dispatch_column_pass(
    const mlx::core::array& in,
    mlx::core::array& out,
    mlx::core::Shape shape,
    mlx::core::Strides strides,
    mlx::core::Shape reduce_shape,
    mlx::core::Strides reduce_strides,
    bool two_pass,
    const std::string& operation,
    const char* dtype) {
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
  if (reduction_stride < 1 || reduction_stride > 65535 ||
      outer * reduction_stride * total != in.size() ||
      outer * reduction_stride * (two_pass ? 32 : 1) != out.size() ||
      in.size() > 65535 || out.size() > 65535) {
    throw std::invalid_argument("CrossTL column reduction storage does not match its pass.");
  }
  uint64_t span = reduction_stride;
  auto accumulate_span = [&](const auto& sizes, const auto& steps) {
    for (size_t axis = 0; axis < sizes.size(); ++axis) {
      if (steps[axis] < 0 || steps[axis] > 65535) {
        throw std::invalid_argument("CrossTL column reduction stride exceeds its bounds.");
      }
      span += (uint64_t(sizes[axis]) - 1) * steps[axis];
    }
  };
  accumulate_span(shape, strides);
  accumulate_span(reduce_shape, reduce_strides);
  if (span > 65535 || in.offset() < 0 ||
      uint64_t(in.offset()) > in.buffer_size() ||
      span * in.itemsize() > in.buffer_size() - uint64_t(in.offset())) {
    throw std::invalid_argument("CrossTL column reduction source view exceeds its bounds.");
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int32_t reduce_ndim = static_cast<int32_t>(reduce_shape.size());
  const int dimension = reduce_ndim == 1 ? 1 : reduce_ndim == 2 ? 2 : 5;
  std::string entry = std::string(two_pass ? "col_reduce_2pass_" : "col_reduce_looped_") +
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
  const CrosstlMlxLaunch launch{
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
  if (total < 32 || (stride < 32 && total >= 1024)) {
    throw std::invalid_argument("CrossTL small-column and long-column reduction plans are not implemented.");
  }
  auto [shape, strides] = shapes_without_reduction_axes(in, axes);
  // Follow upstream's shape product, including broadcast views with zero strides.
  uint64_t inner = 1;
  while (!shape.empty() && int64_t(inner) < stride) {
    if (shape.back() < 1 || uint64_t(shape.back()) > 65535 / inner) {
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
  const bool two_pass = total > 256 && out.size() / 32 < 1024;
  if (!two_pass) {
    dispatch_column_pass(in, out, shape, strides, plan.shape, plan.strides, false, operation, dtype);
    return;
  }
  Shape intermediate_shape{32};
  intermediate_shape.insert(intermediate_shape.end(), out.shape().begin(), out.shape().end());
  array intermediate(std::move(intermediate_shape), out.dtype(), nullptr, {});
  dispatch_column_pass(in, intermediate, shape, strides, plan.shape, plan.strides, true, operation, dtype);
  // Consume the completed native partials, preserving the upstream second pass.
  dispatch_column_pass(intermediate, out, {}, {}, {32}, {int64_t(out.size())}, false, operation, dtype);
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

void dispatch_cast(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL cast input must match output shape.");
  }
  const char* source_type = inputs[0].dtype() == mlx::core::float16 ?
      "float16" : storage_type(inputs[0].dtype());
  const char* destination_type = out.dtype() == mlx::core::float16 ?
      "float16" : storage_type(out.dtype());
  if (!source_type || !destination_type) {
    throw std::invalid_argument("CrossTL casts require float16, float32, int32, uint32, int64, uint64 or bool arrays.");
  }
  if (out.size() > 65535) {
    throw std::invalid_argument("CrossTL cast supports at most 65535 elements.");
  }
  if (inputs[0].dtype() == out.dtype()) {
    out.copy_shared_buffer(inputs[0]);
    return;
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto in = dense_input(inputs[0]);
  uint32_t size = static_cast<uint32_t>(out.size());
  std::string entry = std::string("v_copy") + source_type + destination_type;
  CrosstlMlxBuffer buffers[] = {
      {"src", source_type, in.data<void>(), size, 0},
      {"dst", destination_type, out.data<void>(), size, 1},
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
      "float16" : storage_type(inputs[0].dtype());
  if (!dtype || (!comparison && !bitwise && inputs[0].dtype() == mlx::core::bool_) ||
      (bitwise && inputs[0].dtype() != mlx::core::int32 &&
       inputs[0].dtype() != mlx::core::uint32 && inputs[0].dtype() != mlx::core::bool_)) {
    throw std::invalid_argument("CrossTL binary dispatch requires a supported dtype.");
  }
  if (std::string(operation) == "Divide" && out.dtype() != mlx::core::float32 &&
      out.dtype() != mlx::core::float16) {
    throw std::invalid_argument("CrossTL division requires float16 or float32 arrays.");
  }
  if ((std::string(operation) == "LogicalAnd" || std::string(operation) == "LogicalOr") &&
      inputs[0].dtype() != mlx::core::bool_) {
    throw std::invalid_argument("CrossTL logical operations require bool arrays.");
  }
  if (std::string(operation) == "NaNEqual" && inputs[0].dtype() != mlx::core::float32 &&
      inputs[0].dtype() != mlx::core::float16) {
    operation = "Equal";
  }
  if (out.size() > 65535) {
    throw std::invalid_argument("CrossTL binary dispatch supports at most 65535 elements.");
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto a = dense_input(inputs[0]);
  auto b = dense_input(inputs[1]);
  if (bitwise && (std::string(operation) == "LeftShift" || std::string(operation) == "RightShift")) {
    for (size_t i = 0; i < b.size(); ++i) {
      const int64_t shift = b.dtype() == mlx::core::int32
          ? b.data<int32_t>()[i] : b.data<uint32_t>()[i];
      if (shift < 0 || shift >= 32) {
        throw std::invalid_argument("CrossTL 32-bit shifts require counts in [0, 31].");
      }
    }
  }
  uint32_t size = static_cast<uint32_t>(out.size());
  std::string entry = std::string("vv_") + operation + dtype;
  CrosstlMlxBuffer buffers[] = {
      {"a", dtype, a.data<void>(), size, 0},
      {"b", dtype, b.data<void>(), size, 0},
      {"c", comparison ? "bool_" : dtype, out.data<void>(), size, 1},
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
  dispatch_cast(inputs, out);
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
  if (key_count > 65535 || columns > 65535) {
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
      {static_cast<uint32_t>(key_count), static_cast<uint32_t>(columns), 1},
      {1, 1, 1}};
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
  if (inputs.size() < 2 || inputs.size() > 11 ||
      axes_.size() != inputs.size() - 1 || out.size() > 65535 ||
      slice_sizes_.size() != inputs[0].ndim() ||
      !storage_type(out.dtype()) || out.dtype() != inputs[0].dtype()) {
    throw std::invalid_argument("CrossTL gather specialization exceeds its supported layout.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto src = gather_input(inputs[0]);
  const auto src_span = gather_span(src);
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
    spans.push_back(gather_span(indices.back()));
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
  const auto dim0 = idx_ndim ? inputs[1].shape(0) : 1;
  const auto dim1 = idx_ndim >= 2 ? inputs[1].size() / dim0 : 1;
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1), static_cast<uint32_t>(slice_size)},
      {1, 1, 1}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
      out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native gather failed: ") + error);
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
  if (in.size() > 65535) {
    throw std::invalid_argument("CrossTL reduction supports at most 65535 elements.");
  }
  if (in.size() > 0 && out.size() == in.size()) {
    array identity(out.shape(), in.dtype(), nullptr, {});
    reshape_view(inputs, identity);
    dispatch_cast({identity}, out);
    return;
  }
  const char* dtype = storage_type(in.dtype());
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
  if (plan.type == GeneralReduce) {
    in = dense_input(in);
    plan = get_reduction_plan(in, axes_);
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
    array intermediate({128}, out.dtype(), nullptr, {});
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
      inputs[1].dtype() != out.dtype() || !storage_type(out.dtype()) ||
      out.size() > 65535 || out.ndim() > 64 ||
      start_indices_.size() != out.ndim() || end_indices_.size() != out.ndim() ||
      strides_.size() != out.ndim()) {
    throw std::invalid_argument(
        "CrossTL slice updates require matching supported arrays and at most 65535 output elements.");
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

#undef CROSSTL_BINARY_GPU

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
} // namespace mlx::core

namespace mlx::core::fast {

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (mode_ != QuantizationMode::Affine ||
      (group_size_ != 32 && group_size_ != 64 && group_size_ != 128) ||
      (bits_ != 2 && bits_ != 3 && bits_ != 4 && bits_ != 5 && bits_ != 6 && bits_ != 8) ||
      inputs.size() != (dequantize_ ? 3 : 1) ||
      outputs.size() != (dequantize_ ? 1 : 3)) {
    throw std::invalid_argument("CrossTL affine quantization parameters do not match a pinned entry.");
  }
  const auto dtype = dequantize_ ? outputs[0].dtype() : inputs[0].dtype();
  if (dtype != float32 && dtype != float16 && dtype != bfloat16) {
    throw std::invalid_argument("CrossTL affine quantization requires float32, float16 or bfloat16.");
  }
  const auto elements = dequantize_ ? outputs[0].size() : inputs[0].size();
  if (elements % group_size_ != 0) {
    throw std::invalid_argument("CrossTL affine quantization requires complete groups.");
  }
  const uint64_t groups = elements / group_size_;
  const uint64_t packed_per_group = uint64_t(group_size_) * bits_ / 8;
  const auto& packed = dequantize_ ? inputs[0] : outputs[0];
  const auto& scales = dequantize_ ? inputs[1] : outputs[1];
  const auto& biases = dequantize_ ? inputs[2] : outputs[2];
  if (packed.dtype() != uint32 || packed.size() != groups * packed_per_group / 4 ||
      scales.dtype() != dtype || biases.dtype() != dtype ||
      scales.size() != groups || biases.size() != groups || scales.shape() != biases.shape()) {
    throw std::invalid_argument("CrossTL affine quantization buffer shapes or types differ.");
  }
  const char* storage = dtype == float32 ? "float32" : dtype == float16 ? "float16" : "bfloat16";
  const char* source = dtype == float32 ? "float" : dtype == float16 ? "float16_t" : "bfloat16_t";
  const std::string entry = std::string("affine_") + (dequantize_ ? "dequantize_" : "quantize_") +
      source + "_gs_" + std::to_string(group_size_) + "_b_" + std::to_string(bits_);
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
  const uint32_t local_size = dequantize_ ? group_size_ / pack_factor : 32;
  // Each complete group is independent. Batch without adding out-of-range lanes.
  for (uint64_t first = 0; first < groups;) {
    const uint64_t count = std::min<uint64_t>(groups - first, 65535);
    auto pointer = [](const array& value, uint64_t offset) {
      return const_cast<uint8_t*>(value.data<uint8_t>()) + offset;
    };
    CrosstlMlxBuffer buffers[] = {
        {"w", dequantize_ ? "uint8" : storage,
         pointer(dense[0], first * (dequantize_ ? packed_per_group : group_size_ * itemsize)),
         count * (dequantize_ ? packed_per_group : group_size_), 0},
        {"out", dequantize_ ? storage : "uint8",
         pointer(outputs[0], first * (dequantize_ ? group_size_ * itemsize : packed_per_group)),
         count * (dequantize_ ? group_size_ : packed_per_group), 1},
        {"scales", storage, pointer(dequantize_ ? dense[1] : outputs[1], first * itemsize), count,
         dequantize_ ? uint32_t(0) : uint32_t(1)},
        {"biases", storage, pointer(dequantize_ ? dense[2] : outputs[2], first * itemsize), count,
         dequantize_ ? uint32_t(0) : uint32_t(1)},
    };
    const CrosstlMlxLaunch launch{{static_cast<uint32_t>(count), 1, 1}, {local_size, 1, 1}};
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        entry.c_str(), buffers, 4, count * group_size_, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native affine quantization failed: ") + error);
    }
    first += count;
  }
}

} // namespace mlx::core::fast
