#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mlx/allocator.h"
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
  int status = dispatch_callback.load()(
      entry, buffers, 3, out.size(), error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(
        std::string("CrossTL native dispatch failed: ") + error);
  }
}

void dispatch_unary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation) {
  require_runtime();
  if (inputs.size() != 1 || inputs[0].dtype() != mlx::core::float32 ||
      out.dtype() != mlx::core::float32) {
    throw std::invalid_argument(
        "CrossTL unary dispatch requires float32 arrays.");
  }
  const auto& in = inputs[0];
  if (in.size() == 0) {
    mlx::core::set_unary_output_data(in, out);
    return;
  }
  if (!in.flags().contiguous || in.shape() != out.shape()) {
    throw std::invalid_argument(
        "CrossTL unary dispatch requires contiguous input.");
  }
  if (in.data_size() > 65535) {
    throw std::invalid_argument(
        "CrossTL unary dispatch supports at most 65535 stored elements.");
  }
  mlx::core::set_unary_output_data(in, out);
  uint32_t size = static_cast<uint32_t>(in.data_size());
  std::string entry = std::string("v_") + operation + "float32float32";
  CrosstlMlxBuffer buffers[] = {
      {"in", "float32", const_cast<float*>(in.data<float>()), size, 0},
      {"out", "float32", out.data<float>(), size, 1},
      {"size", "uint32", &size, 1, 0},
  };
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, 3, size, error, sizeof(error));
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
       in.dtype() != mlx::core::uint32)) {
    throw std::invalid_argument(
        "CrossTL copying layouts require matching float32, int32 or uint32 arrays.");
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
  if (span > 65535 || in.offset() < 0 || in.offset() % sizeof(uint32_t) != 0) {
    throw std::invalid_argument("CrossTL copy source span exceeds its bounds.");
  }
  const uint64_t origin = in.offset() / sizeof(uint32_t);
  const uint64_t capacity = in.buffer_size() / sizeof(uint32_t);
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
  CrosstlMlxBuffer buffers[] = {
      {"src", "uint32", const_cast<uint32_t*>(in.data<uint32_t>() + low), uint64_t(span), 0},
      {"dst", "uint32", out.data<uint32_t>(), out.size(), 1},
      {"src_shape", "int32", shape.data(), uint64_t(ndim), 0},
      {"src_strides", "int64", src_strides.data(), uint64_t(ndim), 0},
      {"dst_strides", "int64", dst_strides.data(), uint64_t(ndim), 0},
      {"ndim", "int32", &ndim, 1, 0},
      {"src_offset", "int64", &src_offset, 1, 0},
      {"dst_offset", "int64", &dst_offset, 1, 0},
  };
  char error[2048] = {};
  int status = dispatch_callback.load()(
      "ggn2_dynamic_copyuint32uint32", buffers, 8, out.size(), error, sizeof(error));
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

mlx::core::array binary_input(const mlx::core::array& in) {
  if (in.flags().row_contiguous && in.data_size() == in.size()) {
    const uint64_t bytes = in.nbytes();
    if (in.offset() < 0 || uint64_t(in.offset()) > in.buffer_size() ||
        bytes > in.buffer_size() - uint64_t(in.offset())) {
      throw std::invalid_argument("CrossTL binary input exceeds its allocation.");
    }
    return in;
  }
  mlx::core::array dense(in.shape(), in.dtype(), nullptr, {});
  dispatch_copy(in, dense);
  return dense;
}

void dispatch_binary(
    const std::vector<mlx::core::array>& inputs,
    mlx::core::array& out,
    const char* operation) {
  require_runtime();
  if (inputs.size() != 2 || inputs[0].dtype() != out.dtype() ||
      inputs[1].dtype() != out.dtype() || inputs[0].shape() != out.shape() ||
      inputs[1].shape() != out.shape()) {
    throw std::invalid_argument("CrossTL binary inputs must match output shape and dtype.");
  }
  const char* dtype;
  if (out.dtype() == mlx::core::float32) {
    dtype = "float32";
  } else if (out.dtype() == mlx::core::int32) {
    dtype = "int32";
  } else if (out.dtype() == mlx::core::uint32) {
    dtype = "uint32";
  } else {
    throw std::invalid_argument("CrossTL binary dispatch requires a supported 32-bit dtype.");
  }
  if (std::string(operation) == "Divide" && out.dtype() != mlx::core::float32) {
    throw std::invalid_argument("CrossTL division requires float32 arrays.");
  }
  if (out.size() > 65535) {
    throw std::invalid_argument("CrossTL binary dispatch supports at most 65535 elements.");
  }
  out.set_data(mlx::core::allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto a = binary_input(inputs[0]);
  auto b = binary_input(inputs[1]);
  uint32_t size = static_cast<uint32_t>(out.size());
  std::string entry = std::string("vv_") + operation + dtype;
  CrosstlMlxBuffer buffers[] = {
      {"a", dtype, a.data<void>(), size, 0},
      {"b", dtype, b.data<void>(), size, 0},
      {"c", dtype, out.data<void>(), size, 1},
      {"size", "uint32", &size, 1, 0},
  };
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers, 4, size, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native binary failed: ") + error);
  }
}
} // namespace

extern "C" MLX_API int crosstl_mlx_register_dispatch(
    uint32_t version,
    CrosstlMlxDispatch callback) {
  if (version != 1 || !callback) {
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
