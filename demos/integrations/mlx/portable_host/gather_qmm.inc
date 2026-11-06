// Ports mlx/backend/metal/quantized.cpp: GatherQMM::eval_gpu with the
// gather_qmm, gather_qmv and gather_qvm launches it reaches for affine weights
// on a device without NAX kernels. The host reports no Apple GPU generation,
// so get_qmv_batch_limit takes its default branch, as QuantizedMatmul's port
// does. Each launch binds the buffers and values quantized.cpp binds, in its
// order; gather_qmm_layout checks them against the same routing and replays
// adjust_matrix_offsets before any translated kernel runs. Sorted single-row
// batches, which MLX sends to gather_qmm_rhs, and the mxfp4, mxfp8 and nvfp4
// modes are refused before any launch. crosstl_backend.cpp includes this file
// after crosstl_gather_mm.inc, so the port shares that translation unit's
// dispatch helpers and gather_indices.

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
  if (mode_ != QuantizationMode::Affine) {
    throw std::invalid_argument(
        "CrossTL gather quantized matmul translates affine weights only.");
  }
  const auto dtype = out.dtype();
  if (inputs.size() != 6 || (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != uint32 ||
      inputs[2].dtype() != dtype || inputs[3].dtype() != dtype) {
    throw std::invalid_argument(
        "CrossTL gather quantized matmul requires matching floating types.");
  }
  if (inputs[0].ndim() < 2 || inputs[1].ndim() < 2 || out.ndim() < 2 ||
      inputs[4].shape() != inputs[5].shape()) {
    throw std::invalid_argument("CrossTL gather quantized matmul operand ranks are invalid.");
  }
  // quantized.cpp's ensure_row_contiguous_matrix; the kernels read the index
  // arrays by their strides, which gather_indices keeps nonnegative.
  array x = matmul_operand(inputs[0]);
  array w = matmul_operand(inputs[1]);
  array scales = matmul_operand(inputs[2]);
  array biases = matmul_operand(inputs[3]);
  const array lhs_indices = gather_indices(inputs[4]);
  const array rhs_indices = gather_indices(inputs[5]);
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
  // get_qmv_batch_limit for a device without an Apple GPU generation.
  const int vector_limit = !transpose_ ? 4
      : K <= 2048 && N <= 2048        ? 18
      : K <= 4096 && N <= 4096        ? 12
                                      : 10;
  const bool matrix = M >= vector_limit;
  const std::string stem = quantized_source(dtype) + std::string("_gs_") +
      std::to_string(group_size_) + "_b_" + std::to_string(bits_);
  // qmv_fast_k_alignment(bits): get_pack_factor(bits, 32), times 2 packs per
  // thread except for 2 bits, times 32 lanes.
  const int pack_factor = bits_ == 3 || bits_ == 5 ? 8 : bits_ == 6 ? 4 : 32 / bits_;
  const bool fast = N % 8 == 0 && K % (pack_factor * (bits_ == 2 ? 1 : 2) * 32) == 0;
  std::string entry;
  uint32_t columns;
  uint32_t tiles;
  if (matrix) {
    entry = std::string(transpose_ ? "affine_gather_qmm_t_" : "affine_gather_qmm_n_") +
        stem + (transpose_ ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "");
    columns = (uint32_t(N) + 31) / 32;
    tiles = (uint32_t(M) + 31) / 32;
  } else {
    // gather_qmv: 2 simdgroups of 4 rows; gather_qvm: 2 simdgroups of
    // min(group_size, 32) columns.
    const int rows = transpose_ ? 8 : 2 * std::min(group_size_, 32);
    entry = std::string(
                !transpose_ ? "affine_gather_qvm_"
                    : fast  ? "affine_gather_qmv_fast_"
                            : "affine_gather_qmv_") +
        stem;
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
  std::vector<CrosstlMlxBuffer> buffers = {
      {"w", "uint32", data(w), matmul_span(w), 0},
      {"scales", storage, data(scales), matmul_span(scales), 0},
      {"biases", storage, data(biases), matmul_span(biases), 0},
      {"x", storage, data(x), matmul_span(x), 0},
      {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
      {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
      {"y", storage, out.data<void>(), out.size(), 1},
  };
  if (matrix) {
    buffers.insert(
        buffers.end(),
        {{"K", "int32", &k, 1, 0}, {"N", "int32", &n, 1, 0}, {"M", "int32", &m, 1, 0}});
  } else {
    buffers.insert(
        buffers.end(),
        {{"in_vec_size", "int32", &k, 1, 0}, {"out_vec_size", "int32", &n, 1, 0}});
  }
  QuantizedBatchViews views(x, w, scales, biases);
  views.bind(buffers);
  index_views.bind(buffers);
  const CrosstlMlxLaunch launch = matrix
      ? CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 2}}
      : CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 1}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gather quantized matmul");
}

} // namespace mlx::core
