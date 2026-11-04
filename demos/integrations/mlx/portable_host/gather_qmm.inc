// Ports mlx/backend/metal/quantized.cpp: GatherQMM::eval_gpu with the
// gather_qmm, gather_qmv and gather_qvm launches it reaches on a device
// without NAX, for every quantization mode. Each launch binds the buffers and
// values quantized.cpp binds: add_strides_and_shapes' batch views, which
// GatherQMM never skips, and add_gather_strides_and_shapes' collapsed index
// views. gather_qmm_layout checks them against the same routing and reads
// both index arrays before any translated kernel runs. Sorted single-row
// products that quantized.cpp sends to gather_qmm_rhs are refused.
// crosstl_backend.cpp includes this file at its end, after
// crosstl_gather_mm.inc, so the port shares that translation unit's
// quantized and gather helpers.

namespace mlx::core {
namespace {

// add_gather_strides_and_shapes: the index arrays' shape and strides,
// collapsed together. Scalar indices bind one inert axis the kernels never
// read: with no batch axes, elem_to_loc_broadcast reads no shape or stride.
struct GatherIndexViews {
  int32_t ndim;
  Shape shape;
  Strides lhs_strides;
  Strides rhs_strides;

  GatherIndexViews(const array& lhs_indices, const array& rhs_indices) {
    auto [collapsed, strides] = collapse_contiguous_dims(
        lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
    ndim = int32_t(collapsed.size());
    shape = ndim ? collapsed : Shape{1};
    lhs_strides = ndim ? strides[0] : Strides{0};
    rhs_strides = ndim ? strides[1] : Strides{0};
  }

  void bind(std::vector<CrosstlMlxBuffer>& buffers) {
    auto vector = [](const char* name, const char* type, auto& values) {
      return CrosstlMlxBuffer{name, type, values.data(), values.size(), 0};
    };
    buffers.push_back({"batch_ndims", "int32", &ndim, 1, 0});
    buffers.push_back(vector("batch_shape", "int32", shape));
    buffers.push_back(vector("lhs_strides", "int64", lhs_strides));
    buffers.push_back(vector("rhs_strides", "int64", rhs_strides));
  }
};

const char* gather_quantized_mode(QuantizationMode mode) {
  return mode == QuantizationMode::Affine ? "affine"
      : mode == QuantizationMode::Mxfp4   ? "mxfp4"
      : mode == QuantizationMode::Mxfp8   ? "mxfp8"
      : mode == QuantizationMode::Nvfp4   ? "nvfp4"
                                          : nullptr;
}

// quantized.cpp qmv_fast_k_alignment: get_pack_factor(bits, 32) words of
// codes per lane, two packs per lane except at 2 bits, 32 lanes.
int gather_qmv_fast_alignment(int bits) {
  const int pack_factor = bits == 3 || bits == 5 ? 8 : bits == 6 ? 4 : 32 / bits;
  return pack_factor * (bits == 2 ? 1 : 2) * 32;
}

} // namespace

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const char* mode = gather_quantized_mode(mode_);
  const bool affine = mode_ == QuantizationMode::Affine;
  // Affine gets biases at index 3, the block-scaled modes an optional
  // global scale per weight matrix there.
  const bool global = !affine && inputs.size() == 6;
  if (mode == nullptr || inputs.size() != (affine || global ? 6 : 5) ||
      (!affine &&
       (group_size_ != (mode_ == QuantizationMode::Nvfp4 ? 16 : 32) ||
        bits_ != (mode_ == QuantizationMode::Mxfp8 ? 8 : 4)))) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul parameters do not match a pinned mode.");
  }
  const auto dtype = out.dtype();
  if ((dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != uint32 ||
      inputs[2].dtype() != (affine ? dtype : uint8) ||
      (affine && inputs[3].dtype() != dtype) || (global && inputs[3].dtype() != float32) ||
      inputs[0].ndim() < 2 || inputs[1].ndim() < 2 || out.ndim() < 2) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul operand types or ranks are invalid.");
  }
  array x = matmul_operand(inputs[0]);
  array w = matmul_operand(inputs[1]);
  array scales = matmul_operand(inputs[2]);
  std::optional<array> biases;
  std::optional<array> global_scale;
  if (affine) {
    biases = matmul_operand(inputs[3]);
  } else if (global) {
    // quantized.cpp ensure_row_contiguous: one scale per weight matrix.
    global_scale = inputs[3].flags().row_contiguous ? inputs[3] : dense_input(inputs[3]);
  }
  array lhs_indices = gather_indices(inputs[inputs.size() - 2]);
  array rhs_indices = gather_indices(inputs[inputs.size() - 1]);
  const int K = x.shape(-1);
  const int M = x.shape(-2);
  const int N = out.shape(-1);
  if (K <= 0 || M <= 0 || N <= 0 || out.size() % (uint64_t(M) * N) != 0 ||
      out.shape(-2) != M) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul shape is empty or inconsistent.");
  }
  const uint64_t B = out.size() / M / N;
  if (lhs_indices.shape() != rhs_indices.shape() || lhs_indices.size() != B) {
    throw std::invalid_argument(
        "CrossTL gathered quantized matmul indices do not match the output batch.");
  }
  if (B == 0) {
    // quantized.cpp launches no threadgroups for an empty batch.
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }
  const uint64_t E = w.size() / w.shape(-1) / w.shape(-2);
  if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {
    throw std::invalid_argument(
        "CrossTL does not route sorted single-row gathered products (gather_qmm_rhs).");
  }
  // get_qmv_batch_limit for a device without an Apple GPU generation.
  const int vector_limit = !transpose_ ? 4
      : K <= 2048 && N <= 2048        ? 18
      : K <= 4096 && N <= 4096        ? 12
                                      : 10;
  const bool matrix = M >= vector_limit;
  const bool fast = N % 8 == 0 && K % gather_qmv_fast_alignment(bits_) == 0;
  const char* kernel = matrix ? (transpose_ ? "_gather_qmm_t_" : "_gather_qmm_n_")
      : !transpose_           ? "_gather_qvm_"
      : fast                  ? "_gather_qmv_fast_"
                              : "_gather_qmv_";
  const std::string entry = std::string(mode) + kernel + quantized_source(dtype) + "_gs_" +
      std::to_string(group_size_) + "_b_" + std::to_string(bits_) +
      (matrix && transpose_ ? (N % 32 == 0 ? "_alN_true" : "_alN_false") : "") +
      (global ? "_hgs" : "");
  require_entry(entry);
  // qmm's 32 by 32 tiles; qvm's 2 simdgroups of min(group_size, 32)
  // columns; qmv's 2 simdgroups of 4 rows.
  const uint64_t columns = matrix ? 32 : !transpose_ ? 2 * std::min(group_size_, 32) : 8;
  const uint64_t tiles = (uint64_t(N) + columns - 1) / columns;
  const uint64_t rows = matrix ? (uint64_t(M) + 31) / 32 : uint64_t(M);
  if (out.size() > max_matmul_elements || B > 65535 || tiles > 65535 || rows > 65535) {
    throw std::invalid_argument("CrossTL gathered quantized matmul launch exceeds its bounds.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const char* storage = matmul_storage(dtype);
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
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
  int32_t k = K, n = N, m = M;
  buffers.insert(
      buffers.end(),
      {
          {"x", storage, data(x), matmul_span(x), 0},
          {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
          {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
          {"y", storage, out.data<void>(), out.size(), 1},
      });
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
  GatherIndexViews index_views(lhs_indices, rhs_indices);
  index_views.bind(buffers);
  const CrosstlMlxLaunch launch = matrix
      ? CrosstlMlxLaunch{{uint32_t(tiles), uint32_t(rows), uint32_t(B)}, {32, 2, 2}}
      : CrosstlMlxLaunch{{uint32_t(rows), uint32_t(tiles), uint32_t(B)}, {32, 2, 1}};
  quantized_dispatch(entry, buffers, out.size(), launch, "gathered quantized matmul");
}

} // namespace mlx::core
