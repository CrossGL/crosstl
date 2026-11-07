// Ports mlx/backend/metal/quantized.cpp: GatherQQMM::eval_gpu, which
// quantizes both operands of a gathered floating-point quantized product,
// and the gather_qmv launches it reaches. crosstl_backend.cpp includes this
// file after qq_matmul.inc and gather_mm.inc, so the port shares their
// quantization, index and dispatch helpers.
//
// As quantized.cpp does, the weights are quantized unless they arrive
// quantized, the input is quantized and dequantized, and an nvfp4 global
// scale of the weights is broadcast to one scale per weight matrix, which the
// _hgs kernels read at the gathered matrix. Every product goes to gather_qmv
// except the globally scaled bfloat16 matrix products quantized.cpp sends to
// gather_qmm and gather_qmm_rhs, which the port refuses with a typed error.

namespace mlx::core {
namespace {

// quantized.cpp gather_qmv for mxfp4, mxfp8 and nvfp4 weights: (32, 2, 1)
// threadgroups over (M, N / 8, B), add_strides_and_shapes' operand views and
// add_gather_strides_and_shapes' collapsed index views. Scalar indices bind
// a one-element shape and zero strides the kernels never read. The ``route``
// buffer names the primitive whose routing the launch follows.
void dispatch_gather_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    const char* mode,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    const char* route) {
  const uint64_t B = out.size() / M / N;
  const bool fast = N % 8 == 0 && K % ((32 / bits) * 2 * 32) == 0;
  const std::string entry = std::string(mode) +
      (fast ? "_gather_qmv_fast_" : "_gather_qmv_") + quantized_source(out.dtype()) +
      "_gs_" + std::to_string(group_size) + "_b_" + std::to_string(bits) +
      (global_scale ? "_hgs" : "");
  require_entry(entry);
  const uint32_t tiles = (uint32_t(N) + 7) / 8;
  if (out.size() > max_matmul_elements || B > 65535 || uint32_t(M) > 65535 || tiles > 65535) {
    throw std::invalid_argument("CrossTL gathered quantized matmul launch exceeds its bounds.");
  }
  if (lhs_indices.shape() != rhs_indices.shape() || x.ndim() < 2 || w.ndim() < 2 ||
      x.ndim() > 18 || w.ndim() > 18) {
    throw std::invalid_argument("CrossTL gathered quantized matmul views are invalid.");
  }
  auto [batch_shape, index_strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int32_t batch_ndims = batch_shape.size();
  Strides lhs_strides = index_strides[0];
  Strides rhs_strides = index_strides[1];
  if (batch_ndims == 0) {
    batch_shape = {1};
    lhs_strides = {0};
    rhs_strides = {0};
  }
  if (batch_ndims > 16) {
    throw std::invalid_argument("CrossTL gathered quantized matmul index rank exceeds 16.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const char* storage = matmul_storage(out.dtype());
  int32_t in_vec_size = K;
  int32_t out_vec_size = N;
  int32_t x_batch_ndims = int32_t(x.ndim()) - 2;
  int32_t w_batch_ndims = int32_t(w.ndim()) - 2;
  Shape x_shape = x.shape();
  Strides x_strides = x.strides();
  Shape w_shape = w.shape();
  Strides w_strides = w.strides();
  Strides s_strides = scales.strides();
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  auto vector = [](const char* name, const char* type, auto& values) {
    return CrosstlMlxBuffer{name, type, values.data(), values.size(), 0};
  };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", "uint8", data(scales), matmul_span(scales), 0},
      {"x", storage, data(x), matmul_span(x), 0},
      {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
      {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
      {"y", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_vec_size, 1, 0},
      {"out_vec_size", "int32", &out_vec_size, 1, 0},
      {"x_batch_ndims", "int32", &x_batch_ndims, 1, 0},
      vector("x_shape", "int32", x_shape),
      vector("x_strides", "int64", x_strides),
      {"w_batch_ndims", "int32", &w_batch_ndims, 1, 0},
      vector("w_shape", "int32", w_shape),
      vector("w_strides", "int64", w_strides),
      vector("s_strides", "int64", s_strides),
      {"batch_ndims", "int32", &batch_ndims, 1, 0},
      vector("batch_shape", "int32", batch_shape),
      vector("lhs_strides", "int64", lhs_strides),
      vector("rhs_strides", "int64", rhs_strides),
      {"route", "utf8", const_cast<char*>(route), std::char_traits<char>::length(route), 0},
  };
  if (global_scale) {
    buffers.push_back(
        {"global_scale", "float32", data(*global_scale), matmul_span(*global_scale), 0});
  }
  const CrosstlMlxLaunch launch{{uint32_t(M), tiles, uint32_t(B)}, {32, 2, 1}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gathered quantized matmul");
}

} // namespace

void GatherQQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const char* mode = mode_ == QuantizationMode::Mxfp4 ? "mxfp4"
      : mode_ == QuantizationMode::Mxfp8                ? "mxfp8"
      : mode_ == QuantizationMode::Nvfp4                ? "nvfp4"
                                                        : nullptr;
  // x, w, lhs_indices, rhs_indices, the scales of quantized weights, then the
  // two nvfp4 global scales when both are present.
  const bool w_quantized = inputs.size() > 1 && inputs[1].dtype() == uint32;
  const size_t base = w_quantized ? 5 : 4;
  const bool global = mode_ == QuantizationMode::Nvfp4 && inputs.size() == base + 2;
  if (mode == nullptr || (inputs.size() != base && !global) ||
      group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
      bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)) {
    throw std::invalid_argument(
        "CrossTL gathered quantized-input matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || (!w_quantized && inputs[1].dtype() != dtype) ||
      (w_quantized && inputs[4].dtype() != uint8)) {
    throw std::invalid_argument(
        "CrossTL gathered quantized-input matmul requires matching floating types.");
  }
  // An empty index batch launches no workgroup.
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }
  std::optional<array> global_x, global_w;
  if (global) {
    global_x = dense_input(inputs[inputs.size() - 2]);
    global_w = dense_input(inputs[inputs.size() - 1]);
    if (global_x->dtype() != float32 || global_x->size() != 1 ||
        global_w->dtype() != float32 || global_w->ndim() > 1) {
      throw std::invalid_argument(
          "CrossTL gathered quantized-input matmul requires float32 global scales.");
    }
  }
  std::optional<array> scales;
  if (w_quantized) {
    scales = inputs[4];
  }
  const auto [w_q, scales_w] =
      qq_weights(stream(), inputs[1], scales, global_w, mode_, group_size_, bits_);
  const array x = qq_rounded_input(inputs[0], global_x, mode, group_size_, bits_);
  if (x.ndim() < 2 || w_q.ndim() < 2) {
    throw std::invalid_argument(
        "CrossTL gathered quantized-input matmul operand ranks are invalid.");
  }
  const bool non_batched = w_q.ndim() == 2;
  const int K = x.shape(-1);
  const int M = non_batched ? x.size() / K : x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0 || out.size() % (uint64_t(M) * N) != 0) {
    throw std::invalid_argument(
        "CrossTL gathered quantized-input matmul shape is empty or inconsistent.");
  }
  const int E = w_q.size() / w_q.shape(-1) / w_q.shape(-2);
  if (global_w) {
    // quantized.cpp broadcasts the weights' global scale to one per matrix.
    if (global_w->size() != 1 && global_w->size() != uint64_t(E)) {
      throw std::invalid_argument(
          "CrossTL gathered quantized-input matmul requires one global scale per "
          "weight matrix.");
    }
    array per_matrix(Shape{E}, float32, nullptr, {});
    broadcast(*global_w, per_matrix);
    global_w = dense_input(per_matrix);
  }
  const uint64_t B = out.size() / M / N;
  const bool matrix_kernels =
      global && w_quantized && dtype == bfloat16 && w_q.ndim() == 3 && K % 32 == 0;
  if (matrix_kernels &&
      ((M == 1 && B >= 16 && right_sorted_ && B / E >= 4) || M >= qq_vector_limit(K, N))) {
    // quantized.cpp's gather_qmm_rhs and gather_qmm with the weights' global
    // scales.
    throw std::invalid_argument(
        "CrossTL gathered quantized-input matmul does not yet route globally "
        "scaled bfloat16 matrix products.");
  }
  dispatch_gather_fp_qmv(
      x, w_q, scales_w, global_w, gather_indices(inputs[2]), gather_indices(inputs[3]), out,
      mode, group_size_, bits_, M, N, K, "gather_qqmm");
}

} // namespace mlx::core
