#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
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
#include "mlx/primitives.h"

namespace {
std::atomic<CrosstlMlxDispatch> dispatch_callback{nullptr};

void require_runtime() {
  if (!dispatch_callback.load()) {
    throw std::runtime_error("CrossTL host runtime is not registered.");
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

void dispatch_unary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation) {
  require_runtime();
  const bool logical = std::string(operation) == "LogicalNot";
  const auto type = logical ? mlx::core::bool_ : mlx::core::float32;
  const char* dtype = logical ? "bool_" : "float32";
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

void dispatch_copy(const mlx::core::array& in, mlx::core::array& out) {
  require_runtime();
  if (in.dtype() != out.dtype() ||
      (in.dtype() != mlx::core::float32 && in.dtype() != mlx::core::int32 &&
       in.dtype() != mlx::core::uint32 && in.dtype() != mlx::core::bool_)) {
    throw std::invalid_argument(
        "CrossTL copying layouts require matching float32, int32, uint32 or bool arrays.");
  }
  if (in.size() != out.size() || in.size() > 65535 || in.ndim() > 64) {
    throw std::invalid_argument(
        "CrossTL copy supports equal sizes up to 65535 elements and 64 axes.");
  }
  if (in.size() == 0) {
    out.set_data(mlx::core::allocator::malloc(0));
    return;
  }
  std::vector<int32_t> shape(in.shape().begin(), in.shape().end());
  std::vector<int64_t> src_strides(in.strides().begin(), in.strides().end());
  while (shape.size() < 2) {
    shape.insert(shape.begin(), 1);
    src_strides.insert(src_strides.begin(), 0);
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
  std::vector<int64_t> dst_strides(shape.size());
  int64_t stride = 1;
  for (size_t axis = shape.size(); axis-- > 0;) {
    dst_strides[axis] = stride;
    stride *= shape[axis];
  }
  int32_t ndim = static_cast<int32_t>(shape.size());
  int64_t src_offset = -low, dst_offset = 0;
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  // Copy storage words to preserve NaN payloads, subnormals and signed zero.
  const bool boolean = in.dtype() == mlx::core::bool_;
  const char* dtype = boolean ? "bool_" : "uint32";
  CrosstlMlxBuffer buffers[] = {
      {"src", dtype, const_cast<uint8_t*>(in.data<uint8_t>() + low * item_size), uint64_t(span), 0},
      {"dst", dtype, out.data<void>(), out.size(), 1},
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
  int status = dispatch_callback.load()(
      boolean ? "ggn2_dynamic_copybool_bool_" : "ggn2_dynamic_copyuint32uint32",
      buffers, 8, out.size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native copy failed: ") + error);
  }
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
  return nullptr;
}

void dispatch_cast(const std::vector<mlx::core::array>& inputs, mlx::core::array& out) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL cast input must match output shape.");
  }
  const char* source_type = storage_type(inputs[0].dtype());
  const char* destination_type = storage_type(out.dtype());
  if (!source_type || !destination_type) {
    throw std::invalid_argument("CrossTL casts require float32, int32, uint32 or bool arrays.");
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

void dispatch_binary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation,
    bool comparison = false) {
  require_runtime();
  if (inputs.size() != 2 || inputs[0].dtype() != inputs[1].dtype() ||
      out.dtype() != (comparison ? mlx::core::bool_ : inputs[0].dtype()) ||
      inputs[0].shape() != out.shape() ||
      inputs[1].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL binary inputs must match output shape and dtype.");
  }
  const char* dtype = storage_type(inputs[0].dtype());
  if (!dtype || (!comparison && inputs[0].dtype() == mlx::core::bool_)) {
    throw std::invalid_argument("CrossTL binary dispatch requires a supported 32-bit dtype.");
  }
  if (std::string(operation) == "Divide" && out.dtype() != mlx::core::float32) {
    throw std::invalid_argument("CrossTL division requires float32 arrays.");
  }
  if ((std::string(operation) == "LogicalAnd" || std::string(operation) == "LogicalOr") &&
      inputs[0].dtype() != mlx::core::bool_) {
    throw std::invalid_argument("CrossTL logical operations require bool arrays.");
  }
  if (std::string(operation) == "NaNEqual" && inputs[0].dtype() != mlx::core::float32) {
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
  CrosstlMlxDispatch expected = nullptr;
  return dispatch_callback.compare_exchange_strong(expected, callback) ? 0 : 2;
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
  if (!dtype || in.dtype() != out.dtype() ||
      (boolean ? (reduce_type_ == Sum || reduce_type_ == Prod) : logical)) {
    throw std::invalid_argument(
        "CrossTL reductions require matching float32/int32/uint32 numeric arrays or Boolean all/any.");
  }
  if (in.size() == 0) {
    throw std::invalid_argument("CrossTL empty reduction initialization is not implemented.");
  }
  auto plan = get_reduction_plan(in, axes_);
  if (plan.type == GeneralReduce) {
    in = dense_input(in);
    plan = get_reduction_plan(in, axes_);
  }
  const char* operation = nullptr;
  switch (reduce_type_) {
    case And: operation = "and"; break;
    case Or: operation = "or"; break;
    case Sum: operation = "sum"; break;
    case Prod: operation = "prod"; break;
    case Min: operation = boolean ? "and" : "min"; break;
    case Max: operation = boolean ? "or" : "max"; break;
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
