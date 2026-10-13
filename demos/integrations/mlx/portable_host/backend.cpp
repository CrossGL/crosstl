#include <atomic>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "mlx/allocator.h"
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
