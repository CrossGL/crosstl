// Ports mlx/backend/metal/matmul.cpp: SegmentedMM::eval_gpu and the
// segmented_mm launch it reaches on a device without NAX. The host reports no
// Apple GPU generation, so GEMM_TPARAM_MACRO keeps its medium-device 64x64x16
// tile with 2x2 simdgroups. The launch binds the buffers and values
// matmul.cpp binds; segmented_mm_layout checks them against the same routing
// before the translated kernel runs. crosstl_backend.cpp includes this file
// after crosstl_gather_mm.inc, so the port shares that translation unit's
// dispatch helpers.

namespace mlx::core {
namespace {

// matmul.cpp check_segments_layout. ``true`` means the kernel reads one
// (k_start, k_end) pair per output batch; ``false`` that it reads boundaries,
// batch z spanning words z and z + 1, which needs both trailing strides 1.
// MLX compares each batch stride with the next stride times the extent of
// the axis itself; the host also requires the strides a batch boundary walk
// actually takes, and copies every other view, so it never binds segments
// whose batches the kernel would misread.
std::tuple<bool, array> segmented_segments(const array& x) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(true, x);
  }
  const int ndim = x.ndim();
  bool rc = true;
  for (int i = 0; i < ndim - 2; i++) {
    rc &= (x.strides(i + 1) * x.shape(i) == x.strides(i)) || (x.shape(i) == 1);
  }
  rc &= x.strides(ndim - 1) == 1;
  if (ndim > 1) {
    rc &= x.strides(ndim - 2) == 1;
  }
  // Batch z's first word is word z only if each batch axis steps over the
  // batches of the axes after it.
  int64_t batches = 1;
  for (int i = ndim - 3; rc && i >= 0; i--) {
    batches *= x.shape(i + 1);
    rc = x.shape(i) == 1 || x.strides(i) == batches;
  }
  if (rc) {
    return std::make_tuple(false, x);
  }
  return std::make_tuple(true, contiguous_copy(x));
}

// The segments of an empty contraction: every pair must be empty, since the
// kernel would otherwise read A and B beyond their zero columns.
void require_empty_segments(const array& segments_) {
  const array segments =
      segments_.flags().row_contiguous ? segments_ : contiguous_copy(segments_);
  const auto* words = segments.data<uint32_t>();
  for (size_t pair = 0; 2 * pair + 1 < segments.size(); ++pair) {
    if (words[2 * pair + 1] > words[2 * pair]) {
      throw std::invalid_argument(
          "CrossTL segmented matmul segment exceeds the contraction.");
    }
  }
}

// matmul.cpp segmented_mm: one steel GEMM tile grid per segment, each over
// the inner range the segments give for its output batch.
void dispatch_segmented_mm(
    const array& a_,
    const array& b_,
    const array& segments_,
    array& out,
    int M,
    int N,
    int K) {
  auto [transpose_a, lda, a] = gemv_operand(a_, false);
  auto [transpose_b, ldb, b] = gemv_operand(b_, false);
  auto [segments_contiguous, segments] = segmented_segments(segments_);
  constexpr int bm = 64, bn = 64, bk = 16, wm = 2, wn = 2;
  const uint64_t batch_size_out = out.size() / M / N;
  auto flag = [](bool value) { return value ? "t" : "n"; };
  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("steel_segmented_mm_") + flag(transpose_a) +
      flag(transpose_b) + "_" + storage + "_" + storage + "_bm" + std::to_string(bm) + "_bn" +
      std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" +
      std::to_string(wn) + "_segments_contiguous_" + flag(segments_contiguous) +
      "_align_M_" + flag(M % bm == 0) + "_align_N_" + flag(N % bn == 0);
  require_entry(entry);
  const int64_t tiles_n = (int64_t(N) + bn - 1) / bn;
  const int64_t tiles_m = (int64_t(M) + bm - 1) / bm;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || tiles_n > 65535 ||
      tiles_m > 65535 || lda < 0 || lda > int64_t(max_matmul_elements) || ldb < 0 ||
      ldb > int64_t(max_matmul_elements)) {
    throw std::invalid_argument("CrossTL segmented matmul launch exceeds its bounds.");
  }
  // GEMMParams in params.h order, each widened to int64.
  std::vector<int64_t> params = {
      M, N, K, lda, ldb, N, tiles_n, tiles_m, 0, 0, int64_t(M) * N, 0, 0, 0};
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(a), matmul_span(a), 0},
      {"B", storage, data(b), matmul_span(b), 0},
      {"segments", "uint32", data(segments), matmul_span(segments), 0},
      {"C", storage, out.data<void>(), out.size(), 1},
      {"params", "int64", params.data(), params.size(), 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(tiles_n), uint32_t(tiles_m), uint32_t(batch_size_out)},
      {32, uint32_t(wn), uint32_t(wm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

} // namespace

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const auto dtype = out.dtype();
  if (inputs.size() != 3 || (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != dtype ||
      inputs[2].dtype() != uint32 || inputs[0].ndim() != 2 || inputs[1].ndim() != 2 ||
      inputs[2].ndim() < 1 || out.ndim() < 2) {
    throw std::invalid_argument(
        "CrossTL segmented matmul supports float32, float16 and bfloat16 only.");
  }
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  const auto& segments = inputs[2];
  out.set_data(allocator::malloc(out.nbytes()));
  // No segment, row or column launches no workgroup.
  if (out.size() == 0) {
    return;
  }
  const int M = a.shape(-2);
  const int N = b.shape(-1);
  const int K = a.shape(-1);
  // An empty contraction reads nothing; each batch's accumulators stay zero.
  if (K == 0) {
    require_empty_segments(segments);
    array zero(0, dtype);
    array zeros(out.shape(), dtype, nullptr, {});
    const bool single = out.size() == 1;
    zeros.copy_shared_buffer(zero, Strides(out.ndim(), 0), {single, single, single}, 1);
    dispatch_copy(zeros, out);
    return;
  }
  dispatch_segmented_mm(a, b, segments, out, M, N, K);
}

} // namespace mlx::core
