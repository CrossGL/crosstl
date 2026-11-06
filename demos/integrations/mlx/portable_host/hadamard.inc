// Ports mlx/backend/metal/hadamard.cpp: Hadamard::eval_gpu splits a
// transform of n = m * 2^k points into at most three kernels that
// hadamard_mn_contiguous generates for one size and type, and the runtime
// translates each on first use (hadamard_packages). The host launches every
// stage in slices of whole rows, or of whole m * 2^k batches for the m stage,
// so that no axis carries more than 65535 workgroups; a slice binds its rows
// from their first element, which leaves the kernels' indices unchanged.
// crosstl_backend.cpp includes this file at its end, so the port shares that
// translation unit's dispatch helpers.

#include "mlx/backend/common/hadamard.h"

namespace mlx::core {
namespace {

constexpr uint64_t hadamard_max_groups = 65535;
constexpr uint64_t hadamard_max_elements = (uint64_t(1) << 28) - 1;
// MAX_HADAMARD_THREADS_PER_GROUP.
constexpr int hadamard_max_threads = 256;

const char* hadamard_type(Dtype dtype) {
  return dtype == float32 ? "float32" : dtype == float16 ? "float16" :
      dtype == bfloat16 ? "bfloat16" : nullptr;
}

// One generated kernel: each slice holds whole units of `unit` elements,
// each unit `unit_groups` workgroup rows of `width` threads, `depth` deep.
struct HadamardStage {
  std::string entry;
  uint64_t unit;
  uint64_t unit_groups;
  uint32_t width;
  uint32_t depth;
  float scale;
};

void require_hadamard_extent(const array& x) {
  if (x.offset() < 0 || uint64_t(x.offset()) > x.buffer_size() ||
      x.nbytes() > x.buffer_size() - uint64_t(x.offset())) {
    throw std::invalid_argument("CrossTL Hadamard operand exceeds its allocation.");
  }
}

void dispatch_hadamard_stage(
    const HadamardStage& stage, const array& in, array& out, const char* dtype) {
  const uint64_t total = out.size();
  if (stage.unit == 0 || total % stage.unit != 0 || in.size() != total) {
    throw std::invalid_argument("CrossTL Hadamard stage does not cover whole transforms.");
  }
  const uint64_t per_launch = std::min(
      hadamard_max_groups / stage.unit_groups, hadamard_max_elements / stage.unit);
  if (per_launch == 0) {
    throw std::invalid_argument(
        "CrossTL Hadamard transforms exceed the portable group limit.");
  }
  require_hadamard_extent(in);
  require_hadamard_extent(out);
  char* source = const_cast<char*>(in.data<char>());
  char* destination = out.data<char>();
  const uint64_t units = total / stage.unit;
  for (uint64_t first = 0; first < units; first += per_launch) {
    const uint64_t slice = std::min(per_launch, units - first);
    const uint64_t count = slice * stage.unit;
    const uint64_t offset = first * stage.unit * out.itemsize();
    float scale = stage.scale;
    CrosstlMlxBuffer buffers[] = {
        {"in", dtype, source + offset, count, 0},
        {"out", dtype, destination + offset, count, 1},
        {"scale", "float32", &scale, 1, 0},
    };
    const uint32_t rows = static_cast<uint32_t>(slice * stage.unit_groups);
    const CrosstlMlxLaunch launch{
        {1, rows, stage.depth}, {stage.width, 1, 1}, {0, 0, 0}};
    const uint64_t threads = uint64_t(stage.width) * rows * stage.depth;
    char error[2048] = {};
    const int status = dispatch_callback.load()(
        stage.entry.c_str(), buffers, 3, threads, &launch, error, sizeof(error));
    error[sizeof(error) - 1] = '\0';
    if (status != 0) {
      throw std::runtime_error(std::string("CrossTL native Hadamard failed: ") + error);
    }
  }
}

// hadamard_mn_contiguous: the strided n1 pass, the contiguous n2 pass and
// the m pass, each reading the previous pass's output in place.
void hadamard_contiguous(
    const array& x, array& y, int m, int n1, int n2, float scale, const char* dtype) {
  const int n = n1 * n2;
  const int read_width_m = (n == 1) ? 1 : ((n == 2 || m == 28) ? 2 : 4);
  const int max_radix_1 = std::min(n1, 16);
  const int max_radix_2 = std::min(n2, 16);
  const float scale_n2 = (m == 1) ? scale : 1.0f;
  const std::string name =
      "hadamard_" + std::to_string(n * m) + "_" + std::string(dtype);
  if (n1 > 1) {
    dispatch_hadamard_stage(
        {"n1" + name, uint64_t(n), 1, uint32_t(n1 / max_radix_1), uint32_t(n2), 1.0f},
        x, y, dtype);
  }
  if (n2 > 1) {
    dispatch_hadamard_stage(
        {"n2" + name, uint64_t(n2), 1, uint32_t(n2 / max_radix_2), 1, scale_n2},
        n1 > 1 ? y : x, y, dtype);
  }
  if (m > 1) {
    const int width = std::min(n / read_width_m, hadamard_max_threads);
    dispatch_hadamard_stage(
        {"m" + name, uint64_t(m) * uint64_t(n),
         uint64_t(n / read_width_m / width), uint32_t(width), 1, scale},
        n > 1 ? y : x, y, dtype);
  }
}

} // namespace

void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  if (inputs.size() != 1) {
    throw std::invalid_argument("CrossTL Hadamard requires one input.");
  }
  auto& in = inputs[0];
  const char* dtype = hadamard_type(in.dtype());
  if (!dtype || out.dtype() != in.dtype() || in.ndim() == 0) {
    throw std::invalid_argument(
        "CrossTL Hadamard supports float32, float16 and bfloat16 arrays.");
  }
  auto [n, m] = decompose_hadamard(in.shape().back());
  // decompose_hadamard does not check that n / m is a power of two.
  if (n < 1 || (n & (n - 1)) != 0) {
    throw std::invalid_argument("CrossTL Hadamard transforms need n = m * 2^k.");
  }
  int n1 = 1, n2 = n;
  if (n > 8192) {
    for (n2 = 2; n2 * n2 < n; n2 *= 2) {
    }
    n1 = n / n2;
  }
  // The host reads every input word before it writes an output, so a
  // donated input may share the output.
  if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
    hadamard_contiguous(in, out, m, n1, n2, scale_, dtype);
  } else {
    dispatch_copy(in, out);
    hadamard_contiguous(out, out, m, n1, n2, scale_, dtype);
  }
}

} // namespace mlx::core
