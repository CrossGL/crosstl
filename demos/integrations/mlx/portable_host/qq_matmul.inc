// Ports mlx/backend/metal/quantized.cpp: QQMatmul::eval_gpu, which quantizes
// both operands of a floating-point quantized product. crosstl_backend.cpp
// includes this file after its dispatch helpers, so the port shares that
// translation unit's quantization and matrix-vector dispatches.
//
// As quantized.cpp does, the weights are quantized with fp_quantized.metal's
// quantize kernel unless they arrive quantized, the input is quantized and
// dequantized in place of itself with the quantize_dequantize kernel, and the
// product runs through dispatch_qmv; an nvfp4 global scale of the weights
// takes the _hgs kernels. Every launch covers complete quantization groups,
// one workgroup each, as the host's quantize launches do.

namespace mlx::core {
namespace {

// fp_quantized.metal's instantiate_quantize_dequantize kernels: each thread
// rounds one value through the block format of its group and back.
void dispatch_block_quantize_dequantize(
    const array& in,
    const std::optional<array>& global_scale,
    array& out,
    const char* mode,
    int group_size,
    int bits) {
  const auto dtype = in.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) || out.dtype() != dtype ||
      out.size() != in.size() || in.size() % group_size != 0) {
    throw std::invalid_argument(
        "CrossTL quantize-dequantize requires complete groups of float32, "
        "float16 or bfloat16 values.");
  }
  const std::string entry = std::string(mode) + "_quantize_dequantize_" +
      quantized_source(dtype) + "_gs_" + std::to_string(group_size) + "_b_" +
      std::to_string(bits) + (global_scale ? "_hgs_true" : "_hgs_false");
  require_entry(entry);
  const char* storage = dtype == float32 ? "float32" : dtype == float16 ? "float16" : "bfloat16";
  const uint64_t groups = in.size() / group_size;
  const uint64_t itemsize = in.itemsize();
  for (uint64_t first = 0; first < groups;) {
    const uint64_t count = std::min<uint64_t>(groups - first, 65535);
    std::vector<CrosstlMlxBuffer> buffers = {
        {"w", storage,
         const_cast<uint8_t*>(in.data<uint8_t>()) + first * group_size * itemsize,
         count * group_size, 0},
        {"out", storage, out.data<uint8_t>() + first * group_size * itemsize,
         count * group_size, 1},
    };
    if (global_scale) {
      buffers.push_back(
          {"global_scale", "float32", const_cast<void*>(global_scale->data<void>()), 1, 0});
    }
    const CrosstlMlxLaunch launch{
        {static_cast<uint32_t>(count), 1, 1},
        {static_cast<uint32_t>(std::min(group_size, 32)), 1, 1}};
    quantized_dispatch(entry, buffers, count * group_size, launch, "quantize-dequantize");
    first += count;
  }
}

// quantized.cpp quantize_dequantize_input: a donatable row-contiguous input
// holds its own rounded values.
array qq_rounded_input(
    const array& x_pre,
    const std::optional<array>& global_scale,
    const char* mode,
    int group_size,
    int bits) {
  bool donate = x_pre.is_donatable();
  array x = dense_input(x_pre);
  donate = donate || x.id() != x_pre.id();
  array rounded = donate ? x : array(x.shape(), x.dtype(), nullptr, {});
  if (!donate) {
    rounded.set_data(allocator::malloc(rounded.nbytes()));
  }
  dispatch_block_quantize_dequantize(x, global_scale, rounded, mode, group_size, bits);
  return rounded;
}

// quantized.cpp quantize_input: float weights go through the host's port of
// the Quantize primitive's kernels, quantized weights and their scales are
// bound as they arrive.
std::pair<array, array> qq_weights(
    const Stream& stream,
    const array& w_pre,
    const std::optional<array>& scales,
    const std::optional<array>& global_scale,
    QuantizationMode mode,
    int group_size,
    int bits) {
  if (scales) {
    return {matmul_operand(w_pre), matmul_operand(*scales)};
  }
  const array w = dense_input(w_pre);
  auto packed_shape = w.shape();
  packed_shape.back() = w.shape(-1) * bits / 32;
  auto scales_shape = w.shape();
  scales_shape.back() = w.shape(-1) / group_size;
  std::vector<array> quantize_inputs{w};
  if (global_scale) {
    quantize_inputs.push_back(*global_scale);
  }
  std::vector<array> quantized{
      array(packed_shape, uint32, nullptr, {}), array(scales_shape, uint8, nullptr, {})};
  fast::Quantize(stream, {}, group_size, bits, mode, false).eval_gpu(quantize_inputs, quantized);
  return {quantized[0], quantized[1]};
}

// get_qmv_batch_limit for a device without an Apple GPU generation, as
// QuantizedMatmul::eval_gpu's port uses it.
int qq_vector_limit(int K, int N) {
  return K <= 2048 && N <= 2048 ? 18 : K <= 4096 && N <= 4096 ? 12 : 10;
}

} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const char* mode = mode_ == QuantizationMode::Mxfp4 ? "mxfp4"
      : mode_ == QuantizationMode::Mxfp8                ? "mxfp8"
      : mode_ == QuantizationMode::Nvfp4                ? "nvfp4"
                                                        : nullptr;
  const bool w_quantized = inputs.size() > 1 && inputs[1].dtype() == uint32;
  const size_t base = w_quantized ? 3 : 2;
  const bool global = mode_ == QuantizationMode::Nvfp4 && inputs.size() == base + 2;
  if (mode == nullptr || (inputs.size() != base && !global) ||
      group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
      bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)) {
    throw std::invalid_argument(
        "CrossTL quantized-input matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || (!w_quantized && inputs[1].dtype() != dtype) ||
      (w_quantized && inputs[2].dtype() != uint8)) {
    throw std::invalid_argument(
        "CrossTL quantized-input matmul requires matching floating types.");
  }
  if (global && w_quantized && inputs[1].ndim() == 2 && dtype == bfloat16 &&
      inputs[0].ndim() >= 1) {
    // quantized.cpp's qmm with the weights' global scale, refused before the
    // operands are quantized: quantization keeps the shapes that choose it.
    const int K = inputs[0].shape(-1);
    const int N = out.shape(-1);
    const int64_t M = K > 0 ? int64_t(inputs[0].size() / K) : 0;
    if (K % 32 == 0 && M >= qq_vector_limit(K, N)) {
      throw std::invalid_argument(
          "CrossTL quantized-input matmul does not yet route globally scaled "
          "bfloat16 matrix products.");
    }
  }
  std::optional<array> global_x, global_w;
  if (global) {
    global_x = dense_input(inputs[inputs.size() - 2]);
    global_w = dense_input(inputs[inputs.size() - 1]);
    if (global_x->dtype() != float32 || global_x->size() != 1 ||
        global_w->dtype() != float32 || global_w->size() != 1) {
      throw std::invalid_argument(
          "CrossTL quantized-input matmul requires float32 global scales.");
    }
  }
  std::optional<array> scales;
  if (w_quantized) {
    scales = inputs[2];
  }
  const auto [w_q, scales_w] =
      qq_weights(stream(), inputs[1], scales, global_w, mode_, group_size_, bits_);
  const array x = qq_rounded_input(inputs[0], global_x, mode, group_size_, bits_);
  const bool non_batched = w_q.ndim() == 2;
  if (x.ndim() < 1 || w_q.ndim() < 2 || (!non_batched && x.ndim() < 2)) {
    throw std::invalid_argument("CrossTL quantized-input matmul operand ranks are invalid.");
  }
  const int K = x.shape(-1);
  const int M = non_batched ? x.size() / K : x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0 || out.size() % (uint64_t(M) * N) != 0) {
    throw std::invalid_argument(
        "CrossTL quantized-input matmul shape is empty or inconsistent.");
  }
  dispatch_fp_quantized_matvec(
      x, w_q, scales_w, out, mode, true, group_size_, bits_, M, N, K, global_w, true);
}

} // namespace mlx::core
