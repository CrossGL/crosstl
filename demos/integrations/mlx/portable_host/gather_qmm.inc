// Ports mlx/backend/metal/quantized.cpp: GatherQMM::eval_gpu with the
// gather_qmm, gather_qmv and gather_qvm launches it reaches on a device without
// NAX kernels, for affine weights and for the mxfp4, mxfp8 and nvfp4 modes,
// with nvfp4's optional global scale per expert. The host reports no Apple GPU
// generation, so get_qmv_batch_limit takes its default branch, as
// QuantizedMatmul's port does. Each launch binds the buffers and values
// quantized.cpp binds, in its order; gather_qmm_layout and fp_gather_qmm_layout
// check them against the same routing and replay adjust_matrix_offsets before
// any translated kernel runs. Sorted single-row batches, which MLX sends to
// gather_qmm_rhs, are refused before any launch. crosstl_backend.cpp includes
// this file after crosstl_gather_mm.inc, so the port shares that translation
// unit's dispatch helpers and gather_indices.

namespace mlx::core {
namespace {

// quantized.cpp add_gather_strides_and_shapes: the index arrays' shape and
// strides, collapsed over their contiguous dimensions and kept alive through
// the dispatch.
struct GatherIndexViews {
  int32_t ndims;
  Shape shape;
  Strides lhs_strides;
  Strides rhs_strides;

  GatherIndexViews(const array& lhs_indices, const array& rhs_indices) {
    auto [collapsed, strides] = collapse_contiguous_dims(
        lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
    shape = collapsed;
    lhs_strides = strides[0];
    rhs_strides = strides[1];
    ndims = int32_t(shape.size());
  }

  void bind(std::vector<CrosstlMlxBuffer>& buffers) {
    auto vector = [](const char* name, const char* type, auto& values) {
      return CrosstlMlxBuffer{name, type, values.data(), values.size(), 0};
    };
    buffers.push_back({"batch_ndims", "int32", &ndims, 1, 0});
    buffers.push_back(vector("batch_shape", "int32", shape));
    buffers.push_back(vector("lhs_strides", "int64", lhs_strides));
    buffers.push_back(vector("rhs_strides", "int64", rhs_strides));
  }
};

} // namespace

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const bool affine = mode_ == QuantizationMode::Affine;
  // fp_quantized.metal's modes and their group sizes and bits.
  const char* fp_mode = mode_ == QuantizationMode::Mxfp4 ? "mxfp4"
      : mode_ == QuantizationMode::Mxfp8                 ? "mxfp8"
      : mode_ == QuantizationMode::Nvfp4                 ? "nvfp4"
                                                         : nullptr;
  // ops.cpp gather_qmm: affine weights bring biases at index 3; nvfp4 weights
  // may bring one float32 global scale per expert there instead.
  const bool global = !affine && inputs.size() == 6;
  if (affine ? inputs.size() != 6
             : fp_mode == nullptr || (inputs.size() != 5 && !global) ||
                 (global && mode_ != QuantizationMode::Nvfp4) ||
                 group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
                 bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)) {
    throw std::invalid_argument(
        "CrossTL gather quantized matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != uint32 ||
      inputs[2].dtype() != (affine ? dtype : uint8) ||
      (affine && inputs[3].dtype() != dtype) || (global && inputs[3].dtype() != float32)) {
    throw std::invalid_argument(
        "CrossTL gather quantized matmul requires matching floating types.");
  }
  const array& lhs_input = inputs[inputs.size() - 2];
  const array& rhs_input = inputs[inputs.size() - 1];
  if (inputs[0].ndim() < 2 || inputs[1].ndim() < 2 || out.ndim() < 2 ||
      lhs_input.shape() != rhs_input.shape()) {
    throw std::invalid_argument("CrossTL gather quantized matmul operand ranks are invalid.");
  }
  // quantized.cpp's ensure_row_contiguous_matrix, and ensure_row_contiguous
  // for the global scale; the kernels read the index arrays by their strides,
  // which gather_indices keeps nonnegative.
  array x = matmul_operand(inputs[0]);
  array w = matmul_operand(inputs[1]);
  array scales = matmul_operand(inputs[2]);
  std::optional<array> biases;
  std::optional<array> global_scale;
  if (affine) {
    biases = matmul_operand(inputs[3]);
  } else if (global) {
    global_scale = inputs[3].flags().row_contiguous ? inputs[3] : contiguous_copy(inputs[3]);
  }
  const array lhs_indices = gather_indices(lhs_input);
  const array rhs_indices = gather_indices(rhs_input);
  const int K = x.shape(-1);
  const int M = x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0) {
    throw std::invalid_argument("CrossTL gather quantized matmul shape is empty.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  // An empty index batch launches no threadgroup.
  if (out.size() == 0) {
    return;
  }
  const uint64_t B = out.size() / M / N;
  const uint64_t E = w.size() / w.shape(-1) / w.shape(-2);
  if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {
    throw std::invalid_argument(
        "CrossTL does not translate the sorted gather_qmm_rhs route yet.");
  }
  // ops.cpp shapes the global scale as w's batch: one value per expert.
  if (global_scale && global_scale->size() != E) {
    throw std::invalid_argument(
        "CrossTL gather quantized matmul global scale does not match the experts.");
  }
  // get_qmv_batch_limit for a device without an Apple GPU generation.
  const int vector_limit = !transpose_ ? 4
      : K <= 2048 && N <= 2048        ? 18
      : K <= 4096 && N <= 4096        ? 12
                                      : 10;
  const bool matrix = M >= vector_limit;
  const std::string mode = affine ? "affine" : fp_mode;
  const std::string stem = quantized_source(dtype) + std::string("_gs_") +
      std::to_string(group_size_) + "_b_" + std::to_string(bits_);
  const std::string suffix = global_scale ? "_hgs" : "";
  // qmv_fast_k_alignment(bits): get_pack_factor(bits, 32), times 2 packs per
  // thread except for 2 bits, times 32 lanes.
  const int pack_factor = bits_ == 3 || bits_ == 5 ? 8 : bits_ == 6 ? 4 : 32 / bits_;
  const bool fast = N % 8 == 0 && K % (pack_factor * (bits_ == 2 ? 1 : 2) * 32) == 0;
  std::string entry;
  uint32_t columns;
  uint32_t tiles;
  if (matrix) {
    entry = mode + (transpose_ ? "_gather_qmm_t_" : "_gather_qmm_n_") + stem +
        (transpose_ ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "") + suffix;
    columns = (uint32_t(N) + 31) / 32;
    tiles = (uint32_t(M) + 31) / 32;
  } else {
    // gather_qmv: 2 simdgroups of 4 rows; gather_qvm: 2 simdgroups of
    // min(group_size, 32) columns.
    const int rows = transpose_ ? 8 : 2 * std::min(group_size_, 32);
    entry = mode +
        (!transpose_ ? "_gather_qvm_"
             : fast  ? "_gather_qmv_fast_"
                     : "_gather_qmv_") +
        stem + suffix;
    columns = uint32_t(M);
    tiles = (uint32_t(N) + rows - 1) / rows;
  }
  require_entry(entry);
  GatherIndexViews index_views(lhs_indices, rhs_indices);
  if (out.size() > max_matmul_elements || B > 65535 || columns > 65535 || tiles > 65535 ||
      index_views.ndims < 1 || index_views.ndims > 16 || x.ndim() > 18 || w.ndim() > 18) {
    throw std::invalid_argument("CrossTL gather quantized matmul launch exceeds its bounds.");
  }
  const char* storage = matmul_storage(dtype);
  int32_t k = K, n = N, m = M;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  // quantized.cpp binds the biases or the global scale at slot 2, and
  // leaves it unbound for block-scaled weights without a global scale.
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", affine ? storage : "uint8", data(scales), matmul_span(scales), 0},
  };
  if (biases) {
    buffers.push_back({"biases", storage, data(*biases), matmul_span(*biases), 0});
  } else if (global_scale) {
    buffers.push_back(
        {"global_scale", "float32", data(*global_scale), matmul_span(*global_scale), 0});
  }
  buffers.insert(
      buffers.end(),
      {{"x", storage, data(x), matmul_span(x), 0},
       {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
       {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
       {"y", storage, out.data<void>(), out.size(), 1}});
  if (matrix) {
    buffers.insert(
        buffers.end(),
        {{"K", "int32", &k, 1, 0}, {"N", "int32", &n, 1, 0}, {"M", "int32", &m, 1, 0}});
  } else {
    buffers.insert(
        buffers.end(),
        {{"in_vec_size", "int32", &k, 1, 0}, {"out_vec_size", "int32", &n, 1, 0}});
  }
  QuantizedBatchViews views =
      biases ? QuantizedBatchViews(x, w, scales, *biases) : QuantizedBatchViews(x, w, scales);
  views.bind(buffers);
  index_views.bind(buffers);
  const CrosstlMlxLaunch launch = matrix
      ? CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 2}}
      : CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 1}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gather quantized matmul");
}

} // namespace mlx::core
