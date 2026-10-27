// Ports mlx/backend/metal/fft.cpp: FFT::eval_gpu runs every axis of an N-D
// transform as fft.cpp's nd_fft_op does, and each 1-D transform plans its
// size as fft.cpp's plan_fft does. Plans that fit one threadgroup's Stockham
// decomposition run the translated fft kernel with fft.cpp's buffers,
// constants, threadgroups and grid: complex-to-complex, real-to-complex (two
// real rows per transform) and complex-to-real. Plans with one Rader factor
// run the translated rader_fft kernel the same way, with the tables
// compute_raders_constants builds on the CPU, and Bluestein plans run the
// translated bluestein_fft kernel with the tables compute_bluestein_constants
// builds. Four-step plans and Bluestein's twiddle-table plans are refused by
// name before any allocation or launch.
// fft_layout checks each launch against the same plan; the host names each
// pipeline by fft.cpp's hash name, whose translated entry has the function
// constants that name implies frozen. crosstl_backend.cpp includes this file
// after the copy ports, so it shares that translation unit's strided copy.

#include <complex>
#include <numeric>
#include <set>
#include <sstream>

#include "mlx/3rdparty/pocketfft.h"

namespace mlx::core {
namespace {

constexpr int fft_max_stockham_size = 4096;
constexpr int fft_max_rader_size = 2048;
constexpr int fft_max_bluestein_size = 2048;
constexpr int fft_min_threadgroup_mem_size = 256;

[[noreturn]] void fft_unported(const char* route) {
  throw std::invalid_argument(
      std::string("CrossTL FFT does not port fft.cpp's ") + route + " yet.");
}

// fft.cpp FFTPlan.
struct HostFFTPlan {
  int n = 0;
  std::vector<int> stockham;
  std::vector<int> rader;
  int rader_n = 1;
  int bluestein_n = -1;
  bool four_step = false;
  int n1 = 0;
  int n2 = 0;
};

// fft.cpp supported_radices, ordered by preference in decomposition.
const std::vector<int>& fft_radices() {
  static const std::vector<int> radices = {13, 11, 8, 7, 6, 5, 4, 3, 2};
  return radices;
}

std::vector<int> fft_prime_factors(int n) {
  int z = 2;
  std::vector<int> factors;
  while (z * z <= n) {
    if (n % z == 0) {
      factors.push_back(z);
      n /= z;
    } else {
      z++;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  return factors;
}

std::vector<int> fft_plan_stockham(int n) {
  const auto& radices = fft_radices();
  std::vector<int> plan(radices.size(), 0);
  const int orig_n = n;
  if (n == 1) {
    return plan;
  }
  for (size_t i = 0; i < radices.size(); i++) {
    const int radix = radices[i];
    // Manually tuned radices for powers of 2.
    if (is_power_of_2(orig_n) && orig_n < 512 && radix > 4) {
      continue;
    }
    while (n % radix == 0) {
      plan[i] += 1;
      n /= radix;
      if (n == 1) {
        return plan;
      }
    }
  }
  throw std::runtime_error("Unplannable");
}

HostFFTPlan fft_plan(int n) {
  const auto& radices = fft_radices();
  const std::set<int> radices_set(radices.begin(), radices.end());
  HostFFTPlan plan;
  plan.n = n;
  plan.rader = std::vector<int>(radices.size(), 0);
  const auto factors = fft_prime_factors(n);
  int remaining_n = n;
  auto bluestein = [&]() {
    plan.four_step = n > fft_max_bluestein_size;
    plan.bluestein_n = next_power_of_2(2 * n - 1);
    plan.stockham = fft_plan_stockham(plan.bluestein_n);
    plan.rader = std::vector<int>(radices.size(), 0);
    return plan;
  };
  if (n > fft_max_stockham_size && is_power_of_2(n)) {
    plan.four_step = true;
    plan.n2 = n > 65536 ? 1024 : 64;
    plan.n1 = n / plan.n2;
    if (plan.n1 > fft_max_stockham_size) {
      plan.n1 = fft_max_stockham_size;
      plan.n2 = n / plan.n1;
    }
    if (plan.n2 > fft_max_stockham_size) {
      std::ostringstream msg;
      msg << "[FFT] GPU FFT is not supported for size " << n
          << ", the largest supported size is "
          << fft_max_stockham_size * fft_max_stockham_size
          << ". Run the FFT on the CPU stream instead.";
      throw std::runtime_error(msg.str());
    }
    return plan;
  } else if (n > fft_max_stockham_size) {
    plan.four_step = true;
    plan.bluestein_n = next_power_of_2(2 * n - 1);
    return plan;
  }
  for (int factor : factors) {
    if (radices_set.find(factor) != radices_set.end()) {
      continue;
    }
    if (plan.rader_n > 1 || n > fft_max_rader_size) {
      return bluestein();
    }
    for (int rf : fft_prime_factors(factor - 1)) {
      if (radices_set.find(rf) == radices_set.end()) {
        return bluestein();
      }
    }
    plan.rader = fft_plan_stockham(factor - 1);
    plan.rader_n = factor;
    remaining_n /= factor;
  }
  plan.stockham = fft_plan_stockham(remaining_n);
  return plan;
}

int fft_elems_per_thread(const HostFFTPlan& plan) {
  const auto n = plan.n;
  std::vector<int> steps;
  const auto& radices = fft_radices();
  steps.insert(steps.end(), plan.stockham.begin(), plan.stockham.end());
  steps.insert(steps.end(), plan.rader.begin(), plan.rader.end());
  std::set<int> used_radices;
  for (size_t i = 0; i < steps.size(); i++) {
    if (steps[i] > 0) {
      used_radices.insert(radices[i % radices.size()]);
    }
  }
  auto used = [&](int radix) { return used_radices.count(radix) > 0; };
  if (used(7) && (used(11) || used(13))) {
    return 7;
  } else if (used(11) && used(13)) {
    return 11;
  }
  if (n == 3159) return 13;
  if (n == 3645) return 5;
  if (n == 3969) return 7;
  if (n == 1982) return 5;
  if (used_radices.size() == 1) {
    return *(used_radices.begin());
  }
  std::vector<int> radix_vec(used_radices.begin(), used_radices.end());
  if (used_radices.size() == 2 && (used(11) || used(13))) {
    return std::accumulate(radix_vec.begin(), radix_vec.end(), 0) / 2;
  }
  return radix_vec[1];
}

// fft.cpp's Rader helpers: mod_exp, primitive_root and the tables
// compute_raders_constants returns, as host vectors.
int fft_mod_exp(int x, int y, int n) {
  int out = 1;
  while (y) {
    if (y & 1) {
      out = out * x % n;
    }
    y >>= 1;
    x = x * x % n;
  }
  return out;
}

int fft_primitive_root(int n) {
  auto factors = fft_prime_factors(n - 1);
  for (int r = 2; r < n - 1; r++) {
    bool found = true;
    for (int factor : factors) {
      if (fft_mod_exp(r, (n - 1) / factor, n) == 1) {
        found = false;
        break;
      }
    }
    if (found) {
      return r;
    }
  }
  return -1;
}

struct FFTRaderTables {
  std::vector<std::complex<float>> b_q;
  std::vector<int16_t> g_q;
  std::vector<int16_t> g_minus_q;
};

FFTRaderTables fft_rader_tables(int rader_n) {
  int proot = fft_primitive_root(rader_n);
  // Fermat's little theorem
  int inv = fft_mod_exp(proot, rader_n - 2, rader_n);
  FFTRaderTables tables;
  tables.g_q.resize(rader_n - 1);
  tables.g_minus_q.resize(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    tables.g_q[i] = fft_mod_exp(proot, i, rader_n);
    tables.g_minus_q[i] = fft_mod_exp(inv, i, rader_n);
  }
  std::vector<std::complex<float>> b_q(rader_n - 1);
  for (int i = 0; i < rader_n - 1; i++) {
    float pi_i = (float)tables.g_minus_q[i] * -2.0 * M_PI / rader_n;
    b_q[i] = std::exp(std::complex<float>(0, pi_i));
  }
  tables.b_q.resize(rader_n - 1);
  std::ptrdiff_t item_size = sizeof(std::complex<float>);
  size_t fft_size = rader_n - 1;
  // fft.cpp runs this small FFT on the CPU too.
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ b_q.data(),
      /* data_out= */ tables.b_q.data(),
      /* scale= */ 1.0f);
  return tables;
}

// fft.cpp's twiddle-table selection for Bluestein plans.
constexpr int fft_bluestein_twiddle_table_fft_size = 4096;
constexpr int fft_min_bluestein_twiddle_table_batch = 1024;

// fft.cpp compute_bluestein_constants without the radix twiddle table:
// w_k = exp(-i pi k^2 / n) and w_q = fft(1 / w_k) over 1 - n .. n - 1,
// zero padded to bluestein_n, in double precision and then float32.
struct FFTBluesteinTables {
  std::vector<std::complex<float>> w_k;
  std::vector<std::complex<float>> w_q;
};

FFTBluesteinTables fft_bluestein_tables(int n, int bluestein_n) {
  FFTBluesteinTables tables;
  tables.w_k.resize(n);
  std::vector<std::complex<float>> w_q_vec(bluestein_n, 0);
  for (int i = -n + 1; i < n; i++) {
    double theta = pow(i, 2) * M_PI / (double)n;
    w_q_vec[i + n - 1] = std::exp(std::complex<double>(0, theta));
    if (i >= 0) {
      tables.w_k[i] = std::exp(std::complex<double>(0, -theta));
    }
  }
  tables.w_q.resize(bluestein_n);
  std::ptrdiff_t item_size = sizeof(std::complex<float>);
  size_t fft_size = bluestein_n;
  pocketfft::c2c(
      /* shape= */ {fft_size},
      /* stride_in= */ {item_size},
      /* stride_out= */ {item_size},
      /* axes= */ {0},
      /* forward= */ true,
      /* data_in= */ w_q_vec.data(),
      /* data_out= */ tables.w_q.data(),
      /* scale= */ 1.0f);
  return tables;
}

void fft_launch(
    const std::string& entry,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads,
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native FFT failed: ") + error);
  }
}

// fft.cpp check_input: stride one along the transform axis and a row- or
// column-contiguous layout, or a GeneralGeneral copy with the axis innermost.
array fft_input(const array& x, size_t axis) {
  if (x.strides()[axis] == 1 &&
      (x.flags().row_contiguous || x.flags().col_contiguous)) {
    return x;
  }
  array x_copy(x.shape(), x.dtype(), nullptr, {});
  Strides strides;
  int64_t cur_stride = x.shape(axis);
  for (int a = 0; a < x.ndim(); a++) {
    if (a == int(axis)) {
      strides.push_back(1);
    } else {
      strides.push_back(cur_stride);
      cur_stride *= x.shape(a);
    }
  }
  auto flags = x.flags();
  auto [data_size, is_row_contiguous, is_col_contiguous] =
      check_contiguity(x.shape(), strides);
  flags.col_contiguous = is_col_contiguous;
  flags.row_contiguous = is_row_contiguous;
  flags.contiguous = data_size == x_copy.size();
  x_copy.set_data(allocator::malloc(x.nbytes()), data_size, strides, flags);
  dispatch_copy_into(
      x, x_copy, std::vector<int64_t>(strides.begin(), strides.end()), 0, false);
  return x_copy;
}

// fft.cpp fft_op without four-step parameters.
void fft_op(
    const array& in,
    array& out,
    size_t axis,
    bool inverse,
    bool real,
    bool inplace) {
  require_runtime();
  const size_t n = out.dtype() == float32 ? out.shape(axis) : in.shape(axis);
  if (n == 1) {
    out.copy_shared_buffer(in);
    return;
  }
  const bool real_input = in.dtype() == float32;
  const bool real_output = out.dtype() == float32;
  if ((in.dtype() != complex64 && !real_input) ||
      (out.dtype() != complex64 && !real_output) ||
      (real_input && (!real || inverse)) || (real_output && (!real || !inverse)) ||
      (real && real_input == real_output)) {
    throw std::invalid_argument(
        "CrossTL FFT requires fft.cpp's complex64 and float32 transform types.");
  }
  const array in_contiguous = fft_input(in, axis);
  // real to complex: n -> (n/2)+1; complex to real: (n/2)+1 -> n.
  auto out_strides = in_contiguous.strides();
  size_t out_data_size = in_contiguous.data_size();
  if (in.shape(axis) != out.shape(axis)) {
    for (size_t i = 0; i < out_strides.size(); i++) {
      if (out_strides[i] != 1) {
        out_strides[i] = out_strides[i] / in.shape(axis) * out.shape(axis);
      }
    }
    out_data_size = out_data_size / in.shape(axis) * out.shape(axis);
  }
  const auto plan = fft_plan(int(n));
  if (plan.four_step) {
    fft_unported(plan.bluestein_n > 0 ? "multi-upload Bluestein plans" : "four-step plans");
  }
  const bool bluestein = plan.bluestein_n > 0;
  const size_t size = real_output ? out.size() : in.size();
  if (size % n != 0 || size > size_t(std::numeric_limits<int32_t>::max()) ||
      in.size() > size_t(std::numeric_limits<int32_t>::max()) ||
      out.size() > size_t(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("CrossTL FFT requires signed 32-bit transform extents.");
  }
  if (!inplace) {
    out.set_data(
        allocator::malloc(out.nbytes()), out_data_size, out_strides,
        in_contiguous.flags());
  }
  const int fft_size = bluestein ? plan.bluestein_n : int(n);
  const int elems_per_thread = fft_elems_per_thread(plan);
  const int total_batch_size = int(size / n);
  static const std::vector<int> bluestein_twiddle_table_plan = {
      0, 0, 4, 0, 0, 0, 0, 0, 0};
  if (bluestein && !real &&
      plan.bluestein_n == fft_bluestein_twiddle_table_fft_size &&
      plan.stockham == bluestein_twiddle_table_plan &&
      total_batch_size >= fft_min_bluestein_twiddle_table_batch) {
    fft_unported("Bluestein twiddle-table plans");
  }
  const int threads_per_fft = (fft_size + elems_per_thread - 1) / elems_per_thread;
  const int threadgroup_batch_size =
      std::max(fft_min_threadgroup_mem_size / fft_size, 1);
  const int threadgroup_mem_size = next_power_of_2(threadgroup_batch_size * fft_size);
  int batch_size = (total_batch_size + threadgroup_batch_size - 1) / threadgroup_batch_size;
  if (real) {
    // Two real transforms share one complex transform.
    batch_size = (batch_size + 2 - 1) / 2;
  }
  if (threadgroup_mem_size > fft_max_stockham_size || batch_size < 1 ||
      batch_size > 65535) {
    throw std::invalid_argument("CrossTL FFT launch exceeds 65535 threadgroups.");
  }
  const char* in_type_str = real_input ? "float" : "float2";
  const char* out_type_str = real_output ? "float" : "float2";
  const bool rader = !bluestein && plan.rader_n > 1;
  std::ostringstream kname;
  kname << (bluestein ? "bluestein_fft_mem_" : rader ? "rader_fft_mem_" : "fft_mem_")
        << threadgroup_mem_size << "_"
        << in_type_str << "_" << out_type_str << "_n" << fft_size << "_inv_"
        << inverse;
  const std::string entry = kname.str();
  require_entry(entry);
  int32_t n_value = fft_size;
  int32_t batch_value = total_batch_size;
  const uint64_t in_count = uint64_t(total_batch_size) * in_contiguous.shape(axis);
  const uint64_t out_count = uint64_t(total_batch_size) * out.shape(axis);
  if (in_contiguous.data_size() != in_count || out.data_size() != out_count) {
    throw std::invalid_argument("CrossTL FFT operands are not dense transform batches.");
  }
  std::vector<CrosstlMlxBuffer> buffers = {
      {"in", real_input ? "float32" : "complex64",
       const_cast<void*>(in_contiguous.data<void>()), in_count, 0},
      {"out", real_output ? "float32" : "complex64", out.data<void>(), out_count, 1},
      {"n", "int32", &n_value, 1, 0},
      {"batch_size", "int32", &batch_value, 1, 0},
  };
  FFTRaderTables tables;
  int32_t rader_value = plan.rader_n;
  if (rader) {
    // fft.cpp binds b_q, g_q and g_minus_q at 2 to 4 and rader_n last.
    tables = fft_rader_tables(plan.rader_n);
    const uint64_t table_count = uint64_t(plan.rader_n - 1);
    buffers.insert(
        buffers.begin() + 2,
        {{"raders_b_q", "complex64", tables.b_q.data(), table_count, 0},
         {"raders_g_q", "int16", tables.g_q.data(), table_count, 0},
         {"raders_g_minus_q", "int16", tables.g_minus_q.data(), table_count, 0}});
    buffers.push_back({"rader_n", "int32", &rader_value, 1, 0});
  }
  FFTBluesteinTables bluestein_tables;
  int32_t length_value = int32_t(n);
  if (bluestein) {
    // fft.cpp binds w_q and w_k at 2 and 3, then n, bluestein_n and
    // total_batch_size, which bluestein_fft names length, n and batch_size.
    bluestein_tables = fft_bluestein_tables(int(n), plan.bluestein_n);
    buffers.insert(
        buffers.begin() + 2,
        {{"w_q", "complex64", bluestein_tables.w_q.data(),
          uint64_t(plan.bluestein_n), 0},
         {"w_k", "complex64", bluestein_tables.w_k.data(), uint64_t(n), 0},
         {"length", "int32", &length_value, 1, 0}});
  }
  const CrosstlMlxLaunch launch{
      {uint32_t(batch_size), 1, 1},
      {1, uint32_t(threadgroup_batch_size), uint32_t(threads_per_fft)}};
  fft_launch(entry, buffers, out_count, launch);
}

// fft.cpp nd_fft_op: the N-D transform as a series of 1-D transforms.
void nd_fft_op(
    const array& in,
    array& out,
    const std::vector<size_t>& axes,
    bool inverse,
    bool real) {
  auto temp_shape = inverse ? in.shape() : out.shape();
  std::vector<array> temp_arrs;
  temp_arrs.emplace_back(temp_shape, complex64, nullptr, std::vector<array>{});
  if (axes.size() > 2) {
    temp_arrs.emplace_back(temp_shape, complex64, nullptr, std::vector<array>{});
  }
  for (int i = int(axes.size()) - 1; i >= 0; i--) {
    const int reverse_index = int(axes.size()) - i - 1;
    // For 5D and above, the two temporary arrays are reused.
    const bool inplace = reverse_index >= 3 && i != 0;
    // Opposite order for fft vs ifft.
    const int index = inverse ? reverse_index : i;
    const size_t axis = axes[index];
    // Mirror np.fft.(i)rfftn: only the final axis is a real transform.
    const bool step_real = real && index == int(axes.size()) - 1;
    const array& in_arr = i == int(axes.size()) - 1 ? in : temp_arrs[i % 2];
    array& out_arr = i == 0 ? out : temp_arrs[1 - i % 2];
    fft_op(in_arr, out_arr, axis, inverse, step_real, inplace);
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  if (axes_.size() > 1) {
    nd_fft_op(in, out, axes_, inverse_, real_);
  } else {
    fft_op(in, out, axes_[0], inverse_, real_, /*inplace=*/false);
  }
}

} // namespace mlx::core
