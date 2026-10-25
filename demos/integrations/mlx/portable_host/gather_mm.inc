// Ports mlx/backend/metal/matmul.cpp: GatherMM::eval_gpu with the gather_mv,
// gather_mm, gather_mm_rhs and gather_mm_offsets launches it reaches on a
// device without NAX. The host reports no Apple GPU generation, so
// gemv_wide_config declines and gather_mm_wide never dispatches;
// GEMM_TPARAM_MACRO keeps its medium-device 64x64x16 tile. Each launch binds
// the buffers and values matmul.cpp binds; gather_mv_layout and
// gather_mm_layout check them against the same routing before any translated
// kernel runs. crosstl_backend.cpp includes this file at its end, so the port
// shares that translation unit's dispatch helpers.

#include "mlx/backend/common/broadcasting.h"

namespace mlx::core {
namespace {

void dispatch_gather_entry(
    const std::string& entry,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads, &launch,
      error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native gather matmul failed: ") + error);
  }
}

// The kernels read an index array by its strides. The host binds only views
// with nonnegative strides; a dense copy holds the same indices.
array gather_indices(const array& indices) {
  if (indices.dtype() != uint32) {
    throw std::invalid_argument("CrossTL gather matmul indices must be uint32.");
  }
  const bool nonnegative = std::all_of(
      indices.strides().begin(), indices.strides().end(),
      [](int64_t stride) { return stride >= 0; });
  return nonnegative ? indices : contiguous_copy(indices);
}

bool gather_dims_fit(const Shape& shape) {
  return shape.size() <= 18 &&
      std::all_of(shape.begin(), shape.end(), [](int extent) { return extent > 0; });
}

// matmul.cpp ensure_batch_contiguous, which also copies negative strides.
std::tuple<bool, int64_t, array> gather_batch_contiguous(const array& x) {
  const bool nonnegative = std::all_of(
      x.strides().begin(), x.strides().end(), [](int64_t stride) { return stride >= 0; });
  if (nonnegative && x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides()[x.ndim() - 2], x);
  }
  bool rc = nonnegative;
  for (int i = 0; rc && i + 3 < int(x.ndim()); i++) {
    rc = x.strides()[i + 1] * x.shape(i) == x.strides()[i];
  }
  if (rc) {
    const auto stx = x.strides()[x.ndim() - 2];
    const auto sty = x.strides()[x.ndim() - 1];
    const auto K = x.shape(-2);
    const auto N = x.shape(-1);
    if (sty == 1 && (N != 1 || stx == N)) {
      return std::make_tuple(false, stx, x);
    }
    if (stx == 1 && (N != 1 || sty == K)) {
      return std::make_tuple(true, sty, x);
    }
  }
  array copy = contiguous_copy(x);
  return std::make_tuple(false, copy.strides()[copy.ndim() - 2], copy);
}

// matmul.cpp gather_mm_offsets: one thread per group binary-searches the
// sorted indices. MLX launches exactly num_groups threads; one-thread
// workgroups cover them without partial groups.
array dispatch_gather_mm_offsets(const array& indices, int num_groups, int M) {
  const std::string entry = "gather_mm_offsets";
  require_entry(entry);
  if (num_groups < 1 || num_groups > 65535 || M < 1 || uint64_t(M) > indices.size() ||
      !indices.flags().row_contiguous) {
    throw std::invalid_argument("CrossTL gather matmul offsets exceed their bounds.");
  }
  array offsets({num_groups}, int32, nullptr, {});
  offsets.set_data(allocator::malloc(offsets.nbytes()));
  int32_t m = M;
  std::vector<CrosstlMlxBuffer> buffers = {
      {"indices", "uint32", const_cast<void*>(indices.data<void>()), matmul_span(indices), 0},
      {"offsets", "int32", offsets.data<void>(), uint64_t(num_groups), 1},
      {"M", "int32", &m, 1, 0},
  };
  const CrosstlMlxLaunch launch{{uint32_t(num_groups), 1, 1}, {1, 1, 1}};
  dispatch_gather_entry(entry, buffers, uint64_t(num_groups), launch);
  return offsets;
}

// matmul.cpp gather_mm_rhs: rows sorted by their right index, each group of
// rows multiplied by its own matrix of b.
void dispatch_gather_mm_rhs(
    const array& a_, const array& b_, const array& indices_, array& out) {
  array indices = indices_.flags().row_contiguous ? indices_ : contiguous_copy(indices_);
  auto [transpose_b, ldb, b] = gather_batch_contiguous(b_);
  // lhs_indices were not provided, so a broadcasts with the right indices.
  array a = a_;
  if (a_.size() / a_.shape(-2) / a_.shape(-1) != indices.size()) {
    Shape shape = indices.shape();
    shape.push_back(a_.shape(-2));
    shape.push_back(a_.shape(-1));
    array broadcast_a(std::move(shape), a_.dtype(), nullptr, {});
    broadcast(a_, broadcast_a);
    a = broadcast_a;
  }
  if (!a.flags().row_contiguous) {
    a = contiguous_copy(a);
  }
  const int K = a.shape(-1);
  const int M = a.size() / K;
  const int N = b.shape(-1);
  const int num_groups = b.size() / (b.shape(-2) * b.shape(-1));
  constexpr int bm = 16, bn = 64, bk = 16, wm = 1, wn = 2;
  auto flag = [](bool value) { return value ? "t" : "n"; };
  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("steel_gather_mm_rhs_n") + flag(transpose_b) + "_" +
      storage + "_" + storage + "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) +
      "_bk" + std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" + std::to_string(wn) +
      "_align_N_" + flag(N % bn == 0) + "_align_K_" + flag(K % bk == 0);
  require_entry(entry);
  const int64_t batch_stride_b =
      b.ndim() > 2 ? b.strides()[b.ndim() - 3] : int64_t(b.size());
  const int64_t tiles_n = (int64_t(N) + bn - 1) / bn;
  const int64_t tiles_m = std::min<int64_t>(M, (int64_t(M) + bm - 1) / bm + num_groups - 1);
  if (out.size() != uint64_t(M) * N || out.size() > max_matmul_elements || tiles_n > 65535 ||
      tiles_m > 65535 || ldb < 0 || ldb > int64_t(max_matmul_elements) ||
      batch_stride_b < 0) {
    throw std::invalid_argument("CrossTL gather matmul launch exceeds its bounds.");
  }
  array offsets = dispatch_gather_mm_offsets(indices, num_groups, M);
  // GEMMParams in params.h order, each widened to int64.
  std::vector<int64_t> params = {
      M, N, K, K, ldb, N, tiles_n, tiles_m, 0, batch_stride_b, 0, 0, K / bk, 0};
  int32_t groups = num_groups;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(a), matmul_span(a), 0},
      {"B", storage, data(b), matmul_span(b), 0},
      {"offsets", "int32", data(offsets), uint64_t(num_groups), 0},
      {"C", storage, out.data<void>(), out.size(), 1},
      {"params", "int64", params.data(), params.size(), 0},
      {"num_groups", "int32", &groups, 1, 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(tiles_n), uint32_t(tiles_m), 1}, {32, uint32_t(wn), uint32_t(wm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

// matmul.cpp gather_mv: gemv_gather, or gemv_t_gather for transposed
// matrices, with the vector and the matrix of each output batch gathered.
void dispatch_gather_mv(
    const array& mat_,
    const array& vec_,
    const array& mat_indices,
    const array& vec_indices,
    array& out,
    int N,
    int K,
    bool is_mv) {
  auto [transpose_mat, mat_cols, mat] = gemv_operand(mat_, N == 1);
  auto vec = std::get<2>(gemv_operand(vec_, true));
  // m @ v = v @ m.T with v one-dimensional.
  transpose_mat = (!is_mv) ^ transpose_mat;
  const int32_t in_vector_len = K, out_vector_len = N;
  const uint64_t batch_size_out = out.size() / N;
  const int32_t batch_ndim = out.ndim() - 2;
  const int32_t batch_ndim_mat = mat.ndim() - 2;
  const int32_t batch_ndim_vec = vec.ndim() - 2;
  Strides index_strides = vec_indices.strides();
  index_strides.insert(
      index_strides.end(), mat_indices.strides().begin(), mat_indices.strides().end());
  if (index_strides.empty()) {
    // Scalar indices: the kernels still load a stride, scaled by tid.z == 0.
    index_strides = {0, 0};
  }
  int tm = 4, tn = 4, sm = 1, sn = 32, bm = 1, bn = 1, n_out_per_tgp;
  std::string kind;
  if (transpose_mat) {
    if (in_vector_len >= 8192 && out_vector_len >= 2048) {
      sm = 4;
      sn = 8;
    } else {
      sm = 8;
      sn = 4;
    }
    bn = out_vector_len >= 2048 ? 16 : out_vector_len >= 512 ? 4 : 2;
    tn = out_vector_len < tn ? 1 : tn;
    n_out_per_tgp = bn * sn * tn;
    kind = "gemv_t_gather_";
  } else {
    bm = out_vector_len >= 4096 ? 8 : 4;
    sn = 32;
    tm = out_vector_len < tm ? 1 : tm;
    n_out_per_tgp = bm * sm * tm;
    kind = "gemv_gather_";
  }
  const std::string entry = kind + matmul_storage(out.dtype()) + "_bm" + std::to_string(bm) +
      "_bn" + std::to_string(bn) + "_sm" + std::to_string(sm) + "_sn" + std::to_string(sn) +
      "_tm" + std::to_string(tm) + "_tn" + std::to_string(tn);
  require_entry(entry);
  const uint64_t n_tgp = (uint64_t(out_vector_len) + n_out_per_tgp - 1) / n_out_per_tgp;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || n_tgp > 65535 ||
      mat_cols < 0 || mat_cols > int64_t(max_matmul_elements) ||
      !gather_dims_fit(out.shape()) || !gather_dims_fit(mat.shape()) ||
      !gather_dims_fit(vec.shape()) || index_strides.size() > 32) {
    throw std::invalid_argument("CrossTL gather matmul launch exceeds its bounds.");
  }
  int32_t mat_ld = int32_t(mat_cols);
  Shape out_shape = out.shape();
  Shape vec_shape = vec.shape();
  Strides vec_strides = vec.strides();
  Shape mat_shape = mat.shape();
  Strides mat_strides = mat.strides();
  int32_t in_size = in_vector_len, out_size = out_vector_len;
  int32_t ndim = batch_ndim, ndim_vec = batch_ndim_vec, ndim_mat = batch_ndim_mat;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  const char* storage = matmul_storage(out.dtype());
  std::vector<CrosstlMlxBuffer> buffers = {
      {"mat", storage, data(mat), matmul_span(mat), 0},
      {"in_vec", storage, data(vec), matmul_span(vec), 0},
      {"out_vec", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_size, 1, 0},
      {"out_vec_size", "int32", &out_size, 1, 0},
      {"marix_ld", "int32", &mat_ld, 1, 0},
      {"batch_ndim", "int32", &ndim, 1, 0},
      {"batch_shape", "int32", out_shape.data(), out_shape.size(), 0},
      {"index_batch_strides", "int64", index_strides.data(), index_strides.size(), 0},
      {"vector_batch_ndim", "int32", &ndim_vec, 1, 0},
      {"vector_batch_shape", "int32", vec_shape.data(), vec_shape.size(), 0},
      {"vector_batch_stride", "int64", vec_strides.data(), vec_strides.size(), 0},
      {"matrix_batch_ndim", "int32", &ndim_mat, 1, 0},
      {"matrix_batch_shape", "int32", mat_shape.data(), mat_shape.size(), 0},
      {"matrix_batch_stride", "int64", mat_strides.data(), mat_strides.size(), 0},
      {"vec_indices", "uint32", data(vec_indices), matmul_span(vec_indices), 0},
      {"mat_indices", "uint32", data(mat_indices), matmul_span(mat_indices), 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(n_tgp), 1, uint32_t(batch_size_out)}, {32, uint32_t(bn), uint32_t(bm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

// matmul.cpp gather_mm: one steel GEMM tile grid per output batch, with A and
// B moved to the matrices the indices select.
void dispatch_gather_mm(
    const array& a_,
    const array& b_,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int M,
    int N,
    int K) {
  auto [transpose_a, lda, a] = gemv_operand(a_, false);
  auto [transpose_b, ldb, b] = gemv_operand(b_, false);
  constexpr int bm = 64, bn = 64, bk = 16, wm = 2, wn = 2;
  const uint64_t batch_size_out = out.size() / M / N;
  const int32_t batch_ndim = out.ndim() - 2;
  int32_t batch_ndim_a = a.ndim() - 2;
  int32_t batch_ndim_b = b.ndim() - 2;
  auto flag = [](bool value) { return value ? "t" : "n"; };
  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("steel_gather_mm_") + flag(transpose_a) +
      flag(transpose_b) + "_" + storage + "_" + storage + "_bm" + std::to_string(bm) + "_bn" +
      std::to_string(bn) + "_bk" + std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" +
      std::to_string(wn) + "_has_batch_" + flag(batch_ndim > 1) + "_align_M_" +
      flag(M % bm == 0) + "_align_N_" + flag(N % bn == 0) + "_align_K_" + flag(K % bk == 0);
  require_entry(entry);
  const int64_t tiles_n = (int64_t(N) + bn - 1) / bn;
  const int64_t tiles_m = (int64_t(M) + bm - 1) / bm;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || tiles_n > 65535 ||
      tiles_m > 65535 || lda < 0 || lda > int64_t(max_matmul_elements) || ldb < 0 ||
      ldb > int64_t(max_matmul_elements) || batch_ndim > 16 ||
      !gather_dims_fit(a.shape()) || !gather_dims_fit(b.shape())) {
    throw std::invalid_argument("CrossTL gather matmul launch exceeds its bounds.");
  }
  // GEMMParams in params.h order, each widened to int64.
  std::vector<int64_t> params = {
      M,
      N,
      K,
      lda,
      ldb,
      N,
      tiles_n,
      tiles_m,
      batch_ndim > 0 ? lhs_indices.strides()[0] : 0,
      batch_ndim > 0 ? rhs_indices.strides()[0] : 0,
      int64_t(M) * N,
      0,
      K / bk,
      batch_ndim};
  Shape indices_shape = lhs_indices.shape();
  Strides lhs_strides = lhs_indices.strides();
  Strides rhs_strides = rhs_indices.strides();
  if (batch_ndim == 0) {
    // Scalar indices bind one-element vectors the kernel never reads.
    indices_shape = {1};
    lhs_strides = {0};
    rhs_strides = {0};
  }
  Shape a_shape = a.shape();
  Strides a_strides = a.strides();
  Shape b_shape = b.shape();
  Strides b_strides = b.strides();
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(a), matmul_span(a), 0},
      {"B", storage, data(b), matmul_span(b), 0},
      {"lhs_indices", "uint32", data(lhs_indices), matmul_span(lhs_indices), 0},
      {"rhs_indices", "uint32", data(rhs_indices), matmul_span(rhs_indices), 0},
      {"C", storage, out.data<void>(), out.size(), 1},
      {"params", "int64", params.data(), params.size(), 0},
      {"indices_shape", "int32", indices_shape.data(), indices_shape.size(), 0},
      {"lhs_strides", "int64", lhs_strides.data(), lhs_strides.size(), 0},
      {"rhs_strides", "int64", rhs_strides.data(), rhs_strides.size(), 0},
      {"batch_ndim_a", "int32", &batch_ndim_a, 1, 0},
      {"batch_shape_a", "int32", a_shape.data(), a_shape.size(), 0},
      {"batch_strides_a", "int64", a_strides.data(), a_strides.size(), 0},
      {"batch_ndim_b", "int32", &batch_ndim_b, 1, 0},
      {"batch_shape_b", "int32", b_shape.data(), b_shape.size(), 0},
      {"batch_strides_b", "int64", b_strides.data(), b_strides.size(), 0},
  };
  const CrosstlMlxLaunch launch{
      {uint32_t(tiles_n), uint32_t(tiles_m), uint32_t(batch_size_out)},
      {32, uint32_t(wn), uint32_t(wm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

} // namespace

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const auto dtype = out.dtype();
  if (inputs.size() != 4 || (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != dtype || out.ndim() < 2 ||
      inputs[0].ndim() < 2 || inputs[1].ndim() < 2) {
    throw std::invalid_argument(
        "CrossTL gather matmul supports float32, float16 and bfloat16 only.");
  }
  const auto& a = inputs[0];
  const auto& b = inputs[1];
  // matmul.cpp returns zeros when either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    if (out.size() == 0) {
      out.set_data(allocator::malloc(out.nbytes()));
      return;
    }
    array zero(0, dtype);
    array zeros(out.shape(), dtype, nullptr, {});
    const bool single = out.size() == 1;
    zeros.copy_shared_buffer(zero, Strides(out.ndim(), 0), {single, single, single}, 1);
    dispatch_copy(zeros, out);
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  // An empty index batch launches no workgroup.
  if (out.size() == 0) {
    return;
  }
  const int M = a.shape(-2);
  const int N = b.shape(-1);
  const int K = a.shape(-1);
  const array lhs_indices = gather_indices(inputs[2]);
  const array rhs_indices = gather_indices(inputs[3]);
  // Walk a and b in order, batching the products that share a matrix.
  if (M == 1 && right_sorted_) {
    dispatch_gather_mm_rhs(a, b, rhs_indices, out);
    return;
  }
  // Route to the gathered gemv if a or b is a vector.
  if (M == 1) {
    dispatch_gather_mv(b, a, rhs_indices, lhs_indices, out, N, K, false);
    return;
  }
  if (N == 1) {
    dispatch_gather_mv(a, b, lhs_indices, rhs_indices, out, M, K, true);
    return;
  }
  dispatch_gather_mm(a, b, lhs_indices, rhs_indices, out, M, N, K);
}

} // namespace mlx::core
