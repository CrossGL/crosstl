// Ports mlx/backend/metal/quantized.cpp: GatherQMM::eval_gpu and
// GatherQQMM::eval_gpu, and the gather_qmv, gather_qvm and gather_qmm
// launches they reach. crosstl_backend.cpp includes this file after
// qq_matmul.inc and gather_mm.inc, so the port shares their quantization,
// index and dispatch helpers.
//
// GatherQMM routes as quantized.cpp does on a device without NAX: products
// of at least the vector limit's rows go to gather_qmm, transposed vector
// products to gather_qmv and untransposed ones to gather_qvm, for affine,
// mxfp4, mxfp8 and nvfp4 weights and nvfp4 global scales. Sorted right-hand
// indices of single rows, which quantized.cpp sends to gather_qmm_rhs, are
// refused with a typed error.
//
// GatherQQMM quantizes its weights unless they arrive quantized, quantizes
// and dequantizes its input, and broadcasts an nvfp4 global scale of the
// weights to one scale per weight matrix, which the _hgs kernels read at the
// gathered matrix. Every product goes to gather_qmv except the globally
// scaled bfloat16 matrix products quantized.cpp sends to gather_qmm and
// gather_qmm_rhs, which the port refuses before it launches anything.

namespace mlx::core {
namespace {

// add_strides_and_shapes' operand views and add_gather_strides_and_shapes'
// collapsed index views, kept alive through the dispatch. Scalar indices
// bind a one-element shape and zero strides the kernels never read. Only
// kernels with biases bind their strides.
struct GatheredViews {
  int32_t x_batch_ndims;
  Shape x_shape;
  Strides x_strides;
  int32_t w_batch_ndims;
  Shape w_shape;
  Strides w_strides;
  Strides s_strides;
  Strides b_strides;
  bool biased;
  int32_t batch_ndims;
  Shape batch_shape;
  Strides lhs_strides;
  Strides rhs_strides;

  GatheredViews(
      const array& x,
      const array& w,
      const array& scales,
      const std::optional<array>& biases,
      const array& lhs_indices,
      const array& rhs_indices)
      : x_batch_ndims(int32_t(x.ndim()) - 2),
        x_shape(x.shape()),
        x_strides(x.strides()),
        w_batch_ndims(int32_t(w.ndim()) - 2),
        w_shape(w.shape()),
        w_strides(w.strides()),
        s_strides(scales.strides()),
        biased(biases.has_value()) {
    if (lhs_indices.shape() != rhs_indices.shape() || x.ndim() < 2 || w.ndim() < 2 ||
        x.ndim() > 18 || w.ndim() > 18) {
      throw std::invalid_argument("CrossTL gathered quantized matmul views are invalid.");
    }
    if (biases) {
      b_strides = biases->strides();
    }
    auto [shape, strides] = collapse_contiguous_dims(
        lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
    batch_ndims = shape.size();
    batch_shape = shape;
    lhs_strides = strides[0];
    rhs_strides = strides[1];
    if (batch_ndims == 0) {
      batch_shape = {1};
      lhs_strides = {0};
      rhs_strides = {0};
    }
    if (batch_ndims > 16) {
      throw std::invalid_argument("CrossTL gathered quantized matmul index rank exceeds 16.");
    }
  }

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
    buffers.push_back({"batch_ndims", "int32", &batch_ndims, 1, 0});
    buffers.push_back(vector("batch_shape", "int32", batch_shape));
    buffers.push_back(vector("lhs_strides", "int64", lhs_strides));
    buffers.push_back(vector("rhs_strides", "int64", rhs_strides));
  }
};

// The operand buffers both gathered launches bind first, as quantized.cpp
// binds them: affine scales and biases in the element type, block-scaled
// uint8 scales.
std::vector<CrosstlMlxBuffer> gathered_operands(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out) {
  const char* storage = matmul_storage(out.dtype());
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
          {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
          {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
          {"y", storage, out.data<void>(), out.size(), 1},
      });
  return buffers;
}

// The ``route`` buffer names the primitive whose routing the launch follows;
// an nvfp4 global scale goes last.
void finish_gathered(
    std::vector<CrosstlMlxBuffer>& buffers,
    const std::optional<array>& global_scale,
    const char* route) {
  buffers.push_back(
      {"route", "utf8", const_cast<char*>(route), std::char_traits<char>::length(route), 0});
  if (global_scale) {
    buffers.push_back(
        {"global_scale", "float32", const_cast<void*>(global_scale->data<void>()),
         matmul_span(*global_scale), 0});
  }
}

std::string gathered_stem(
    const char* mode, const char* kernel, Dtype dtype, int group_size, int bits) {
  return std::string(mode) + kernel + quantized_source(dtype) + "_gs_" +
      std::to_string(group_size) + "_b_" + std::to_string(bits);
}

// quantized.cpp gather_qmv and gather_qvm: (32, 2, 1) threadgroups over
// (M, ceil(N / bn), B), where bn is gather_qmv's two simdgroups of four rows
// or gather_qvm's two simdgroups of min(group_size, 32) columns.
void dispatch_gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    const char* mode,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    const char* route) {
  const uint64_t B = out.size() / M / N;
  // qmv_fast_k_alignment: get_pack_factor<bits, 32> values per word.
  const int pack_factor = bits == 3 || bits == 5 ? 8 : bits == 6 ? 4 : 32 / bits;
  const bool fast = N % 8 == 0 && K % (pack_factor * (bits == 2 ? 1 : 2) * 32) == 0;
  const char* kernel = !transpose ? "_gather_qvm_"
      : fast                      ? "_gather_qmv_fast_"
                                  : "_gather_qmv_";
  const std::string entry = gathered_stem(mode, kernel, out.dtype(), group_size, bits) +
      (global_scale ? "_hgs" : "");
  require_entry(entry);
  const int rows = transpose ? 8 : 2 * std::min(group_size, 32);
  const uint32_t tiles = (uint32_t(N) + rows - 1) / rows;
  if (out.size() > max_matmul_elements || B > 65535 || uint32_t(M) > 65535 || tiles > 65535) {
    throw std::invalid_argument("CrossTL gathered quantized matmul launch exceeds its bounds.");
  }
  GatheredViews views(x, w, scales, biases, lhs_indices, rhs_indices);
  out.set_data(allocator::malloc(out.nbytes()));
  int32_t in_vec_size = K;
  int32_t out_vec_size = N;
  auto buffers = gathered_operands(x, w, scales, biases, lhs_indices, rhs_indices, out);
  buffers.push_back({"in_vec_size", "int32", &in_vec_size, 1, 0});
  buffers.push_back({"out_vec_size", "int32", &out_vec_size, 1, 0});
  views.bind(buffers);
  finish_gathered(buffers, global_scale, route);
  const CrosstlMlxLaunch launch{{uint32_t(M), tiles, uint32_t(B)}, {32, 2, 1}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gathered quantized matmul");
}

// quantized.cpp gather_qmm without NAX: (32, 2, 2) threadgroups over 32 by
// 32 output tiles of each gathered batch; gather_qmm_t names whether N
// fills whole tiles.
void dispatch_gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& global_scale,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    const char* mode,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    const char* route) {
  const uint64_t B = out.size() / M / N;
  const std::string entry =
      gathered_stem(
          mode, transpose ? "_gather_qmm_t_" : "_gather_qmm_n_", out.dtype(), group_size,
          bits) +
      (transpose ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "") +
      (global_scale ? "_hgs" : "");
  require_entry(entry);
  const uint32_t tiles_n = (uint32_t(N) + 31) / 32;
  const uint32_t tiles_m = (uint32_t(M) + 31) / 32;
  if (out.size() > max_matmul_elements || B > 65535 || tiles_n > 65535 || tiles_m > 65535) {
    throw std::invalid_argument("CrossTL gathered quantized matmul launch exceeds its bounds.");
  }
  GatheredViews views(x, w, scales, biases, lhs_indices, rhs_indices);
  out.set_data(allocator::malloc(out.nbytes()));
  int32_t k = K, n = N, m = M;
  auto buffers = gathered_operands(x, w, scales, biases, lhs_indices, rhs_indices, out);
  buffers.push_back({"K", "int32", &k, 1, 0});
  buffers.push_back({"N", "int32", &n, 1, 0});
  buffers.push_back({"M", "int32", &m, 1, 0});
  views.bind(buffers);
  finish_gathered(buffers, global_scale, route);
  const CrosstlMlxLaunch launch{{tiles_n, tiles_m, uint32_t(B)}, {32, 2, 2}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gathered quantized matmul");
}

bool affine_parameters(int group_size, int bits) {
  return (group_size == 32 || group_size == 64 || group_size == 128) &&
      (bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8);
}

} // namespace

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const bool affine = mode_ == QuantizationMode::Affine;
  const char* mode = affine                           ? "affine"
      : mode_ == QuantizationMode::Mxfp4 ? "mxfp4"
      : mode_ == QuantizationMode::Mxfp8 ? "mxfp8"
      : mode_ == QuantizationMode::Nvfp4 ? "nvfp4"
                                         : nullptr;
  // x, w, scales, then the affine biases or an nvfp4 global scale, then the
  // two index arrays.
  const bool global = !affine && inputs.size() == 6;
  if (mode == nullptr || inputs.size() != (affine || global ? 6 : 5) ||
      (global && mode_ != QuantizationMode::Nvfp4) ||
      (affine ? !affine_parameters(group_size_, bits_)
              : group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
                bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4))) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != uint32 ||
      inputs[2].dtype() != (affine ? dtype : uint8) ||
      (affine && inputs[3].dtype() != dtype) || (global && inputs[3].dtype() != float32)) {
    throw std::invalid_argument("CrossTL gathered quantized matmul requires matching types.");
  }
  // An empty index batch launches no workgroup.
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }
  // ensure_row_contiguous_matrix for the operands, ensure_row_contiguous for
  // the global scale; the indices are read by their strides.
  const array x = matmul_operand(inputs[0]);
  const array w = matmul_operand(inputs[1]);
  const array scales = matmul_operand(inputs[2]);
  std::optional<array> biases, global_scale;
  if (affine) {
    biases = matmul_operand(inputs[3]);
  } else if (global) {
    global_scale = dense_input(inputs[3]);
  }
  const array lhs_indices = gather_indices(inputs[inputs.size() - 2]);
  const array rhs_indices = gather_indices(inputs[inputs.size() - 1]);
  if (x.ndim() < 2 || w.ndim() < 2) {
    throw std::invalid_argument("CrossTL gathered quantized matmul operand ranks are invalid.");
  }
  const int K = x.shape(-1);
  const int M = x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0 || out.size() % (uint64_t(M) * N) != 0) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul shape is empty or inconsistent.");
  }
  const int64_t B = out.size() / M / N;
  const int64_t E = w.size() / w.shape(-1) / w.shape(-2);
  if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {
    // quantized.cpp's gather_qmm_rhs.
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul does not yet route sorted right-hand "
        "indices (gather_qmm_rhs).");
  }
  // get_qmv_batch_limit for a device without an Apple GPU generation.
  const int vector_limit = transpose_ ? qq_vector_limit(K, N) : 4;
  if (M >= vector_limit) {
    dispatch_gather_qmm(
        x, w, scales, biases, global_scale, lhs_indices, rhs_indices, out, mode,
        transpose_, group_size_, bits_, M, N, K, "gather_qmm");
    return;
  }
  dispatch_gather_qmv(
      x, w, scales, biases, global_scale, lhs_indices, rhs_indices, out, mode, transpose_,
      group_size_, bits_, M, N, K, "gather_qmm");
}

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
  if (global && w_quantized && dtype == bfloat16 && inputs[1].ndim() == 3 &&
      inputs[0].ndim() >= 2) {
    // quantized.cpp's gather_qmm_rhs and gather_qmm with the weights' global
    // scales, refused before the operands are quantized: quantization keeps
    // the shapes that choose them.
    const array& w = inputs[1];
    const int K = inputs[0].shape(-1);
    const int M = inputs[0].shape(-2);
    const int N = out.shape(-1);
    const int64_t E = w.shape(0);
    const int64_t B = M > 0 && N > 0 ? int64_t(out.size() / M / N) : 0;
    if (K % 32 == 0 && E > 0 &&
        ((M == 1 && B >= 16 && right_sorted_ && B / E >= 4) ||
         M >= qq_vector_limit(K, N))) {
      throw std::invalid_argument(
          "CrossTL gathered quantized-input matmul does not yet route globally "
          "scaled bfloat16 matrix products.");
    }
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
  dispatch_gather_qmv(
      x, w_q, scales_w, std::nullopt, global_w, gather_indices(inputs[2]),
      gather_indices(inputs[3]), out, mode, true, group_size_, bits_, M, N, K,
      "gather_qqmm");
}

} // namespace mlx::core
