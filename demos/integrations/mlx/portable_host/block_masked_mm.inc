// Ports mlx/backend/metal/matmul.cpp: BlockMaskedMM::eval_gpu with the
// gemv_masked, gemv_t_masked and steel block_masked_gemm launches it reaches.
// Each launch binds the buffers and values matmul.cpp binds, with the same
// kernel name; block_masked_mm_layout checks them against the same routing
// before any translated kernel runs. Absent masks are not bound, as in
// matmul.cpp; the dispatcher fills those bindings with words the kernels
// never read. crosstl_backend.cpp includes this file after
// crosstl_segmented_mm.inc, so the port shares that translation unit's
// dispatch helpers.
//
// Two layout choices differ from matmul.cpp without changing a value:
// operands and masks with negative strides are copied first, as gemv_operand
// copies them for Matmul, and the batch strides come from the arrays the
// launch binds. matmul.cpp takes them from the inputs before check_transpose
// copies an operand, which would misread a copied batched operand.

namespace mlx::core {
namespace {

// The view a launch binds for a mask: itself, or a row-major copy when one
// of its strides is negative.
array masked_mask(const array& mask) {
  const bool nonnegative = std::all_of(
      mask.strides().begin(), mask.strides().end(),
      [](int64_t stride) { return stride >= 0; });
  return nonnegative ? mask : contiguous_copy(mask);
}

int32_t masked_int(int64_t value) {
  if (value < 0 || value > int64_t(max_matmul_elements)) {
    throw std::invalid_argument("CrossTL block masked matmul stride exceeds its bounds.");
  }
  return int32_t(value);
}

const char* masked_mask_dtype(const array& mask, Dtype dtype) {
  if (mask.dtype() == bool_) {
    return "bool_";
  }
  if (mask.dtype() != dtype) {
    throw std::invalid_argument(
        "CrossTL block masked matmul masks must be boolean or of the output type.");
  }
  return matmul_storage(dtype);
}

std::string masked_name(const array* mask, Dtype dtype) {
  if (mask == nullptr) {
    return "nomask";
  }
  return masked_mask_dtype(*mask, dtype);
}

// The trailing (matrix) strides of one bound array, as matmul.cpp pushes
// them: ``first`` then ``second``, each counted from the end.
void push_mask_strides(std::vector<int32_t>& strides, const array& mask, int first, int second) {
  strides.push_back(masked_int(mask.strides()[mask.ndim() + first]));
  strides.push_back(masked_int(mask.strides()[mask.ndim() + second]));
}

Strides batch_strides_of(const array& value, size_t batch_ndim) {
  if (value.ndim() != batch_ndim + 2) {
    throw std::invalid_argument("CrossTL block masked matmul batch ranks differ.");
  }
  return Strides(value.strides().begin(), value.strides().end() - 2);
}

struct MaskedOperands {
  array a;
  array b;
  bool transpose_a;
  bool transpose_b;
  int64_t a_cols;
  int64_t b_cols;
  std::optional<array> out_mask;
  std::optional<array> lhs_mask;
  std::optional<array> rhs_mask;
};

// matmul.cpp's gemv specialization: gemv_masked or gemv_t_masked with the
// output, matrix and vector masks of each batch.
void dispatch_masked_gemv(
    const MaskedOperands& ops,
    array& out,
    int block_size,
    int M,
    int N,
    int K,
    const Shape& batch_shape,
    const Strides& A_batch_stride,
    const Strides& B_batch_stride,
    const Strides& outmask_bstride,
    const Strides& Amask_bstride,
    const Strides& Bmask_bstride) {
  const bool is_b_matrix = N != 1;
  const array& mat = is_b_matrix ? ops.b : ops.a;
  const array& vec = is_b_matrix ? ops.a : ops.b;
  const bool transpose_mat = is_b_matrix ? !ops.transpose_b : ops.transpose_a;
  const int in_vector_len = K;
  const int out_vector_len = is_b_matrix ? N : M;
  const int64_t mat_ld = is_b_matrix ? ops.b_cols : ops.a_cols;
  const Strides& batch_strides_mat = is_b_matrix ? B_batch_stride : A_batch_stride;
  const Strides& batch_strides_vec = is_b_matrix ? A_batch_stride : B_batch_stride;
  const Strides& mask_bstrides_mat = is_b_matrix ? Bmask_bstride : Amask_bstride;
  const Strides& mask_bstrides_vec = is_b_matrix ? Amask_bstride : Bmask_bstride;
  const auto& mat_mask = is_b_matrix ? ops.rhs_mask : ops.lhs_mask;
  const auto& vec_mask = is_b_matrix ? ops.lhs_mask : ops.rhs_mask;
  const bool contiguous_kernel = batch_shape.size() == 1;
  const int32_t batch_ndim = batch_shape.size();
  int tm = 4, tn = 4, sm = 1, sn = 32, bm = 1, bn = 1;
  int n_out_per_tgp;
  std::string kname;
  if (transpose_mat) {
    sm = 8;
    sn = 4;
    bm = 1;
    bn = (block_size == 64 && out_vector_len >= 2048) ? 4 : 2;
    tm = block_size == 32 ? 4 : 8;
    tn = out_vector_len < 4 ? 1 : 4;
    n_out_per_tgp = bn * sn * tn;
    kname = "gemv_t";
  } else {
    if (block_size == 32) {
      sm = 4;
      sn = 8;
      bm = 2;
    } else {
      sm = 2;
      sn = 16;
      bm = out_vector_len >= 512 ? 4 : 2;
    }
    tm = out_vector_len < tm ? 1 : tm;
    n_out_per_tgp = bm * sm * tm;
    kname = "gemv";
  }
  const Dtype dtype = out.dtype();
  const array* out_mask = ops.out_mask ? &*ops.out_mask : nullptr;
  const array* op_mask = mat_mask ? &*mat_mask : nullptr;
  const std::string entry = kname + "_outmask_" + masked_name(out_mask, dtype) + "_opmask_" +
      masked_name(op_mask, dtype) + "_" + matmul_storage(dtype) + "_bm" + std::to_string(bm) +
      "_bn" + std::to_string(bn) + "_sm" + std::to_string(sm) + "_sn" + std::to_string(sn) +
      "_tm" + std::to_string(tm) + "_tn" + std::to_string(tn) + "_nc" +
      std::to_string(int(!contiguous_kernel));
  require_entry(entry);
  const uint64_t batch_size_out = out.size() / uint64_t(out_vector_len);
  const int64_t n_tgp = (int64_t(out_vector_len) + n_out_per_tgp - 1) / n_out_per_tgp;
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || n_tgp > 65535 ||
      mat_ld < 0 || mat_ld > int64_t(max_matmul_elements)) {
    throw std::invalid_argument("CrossTL block masked matmul launch exceeds its bounds.");
  }
  std::vector<int32_t> mask_strides;
  Strides mask_batch_strides;
  if (out_mask) {
    if (transpose_mat) {
      push_mask_strides(mask_strides, *out_mask, out.shape(-2) == 1 ? -1 : -2,
                        out.shape(-2) == 1 ? -2 : -1);
    } else {
      push_mask_strides(mask_strides, *out_mask, out.shape(-1) == 1 ? -1 : -2,
                        out.shape(-1) == 1 ? -2 : -1);
    }
    mask_batch_strides.insert(
        mask_batch_strides.end(), outmask_bstride.begin(), outmask_bstride.end());
  }
  if (op_mask) {
    if (transpose_mat) {
      push_mask_strides(mask_strides, *mat_mask, !is_b_matrix ? -2 : -1,
                        !is_b_matrix ? -1 : -2);
    } else {
      push_mask_strides(mask_strides, *mat_mask, is_b_matrix ? -2 : -1,
                        is_b_matrix ? -1 : -2);
    }
    mask_batch_strides.insert(
        mask_batch_strides.end(), mask_bstrides_mat.begin(), mask_bstrides_mat.end());
    if (transpose_mat) {
      push_mask_strides(mask_strides, *vec_mask, vec.shape(-2) == 1 ? -1 : -2,
                        vec.shape(-2) == 1 ? -2 : -1);
    } else {
      push_mask_strides(mask_strides, *vec_mask, vec.shape(-1) == 1 ? -1 : -2,
                        vec.shape(-1) == 1 ? -2 : -1);
    }
    mask_batch_strides.insert(
        mask_batch_strides.end(), mask_bstrides_vec.begin(), mask_bstrides_vec.end());
  }
  int32_t in_size = in_vector_len, out_size = out_vector_len, ld = int32_t(mat_ld);
  int32_t ndim = batch_ndim;
  std::vector<int32_t> shape(batch_shape.begin(), batch_shape.end());
  Strides vector_strides = batch_strides_vec, matrix_strides = batch_strides_mat;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  const char* storage = matmul_storage(dtype);
  std::vector<CrosstlMlxBuffer> buffers = {
      {"mat", storage, data(mat), matmul_span(mat), 0},
      {"in_vec", storage, data(vec), matmul_span(vec), 0},
      {"out_vec", storage, out.data<void>(), out.size(), 1},
      {"in_vec_size", "int32", &in_size, 1, 0},
      {"out_vec_size", "int32", &out_size, 1, 0},
      {"marix_ld", "int32", &ld, 1, 0},
      {"batch_ndim", "int32", &ndim, 1, 0},
      {"batch_shape", "int32", shape.data(), shape.size(), 0},
      {"vector_batch_stride", "int64", vector_strides.data(), vector_strides.size(), 0},
      {"matrix_batch_stride", "int64", matrix_strides.data(), matrix_strides.size(), 0},
      {"mask_strides", "int32", mask_strides.data(), mask_strides.size(), 0},
      {"mask_batch_strides", "int64", mask_batch_strides.data(), mask_batch_strides.size(), 0},
  };
  if (out_mask) {
    buffers.push_back({"out_mask", masked_mask_dtype(*out_mask, dtype), data(*out_mask),
                       matmul_span(*out_mask), 0});
  }
  if (op_mask) {
    buffers.push_back({"mat_mask", masked_mask_dtype(*mat_mask, dtype), data(*mat_mask),
                       matmul_span(*mat_mask), 0});
    buffers.push_back({"vec_mask", masked_mask_dtype(*vec_mask, dtype), data(*vec_mask),
                       matmul_span(*vec_mask), 0});
  }
  const CrosstlMlxLaunch launch{
      {uint32_t(n_tgp), 1, uint32_t(batch_size_out)}, {32, uint32_t(bn), uint32_t(bm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

// matmul.cpp's regular kernel dispatch: steel block_masked_gemm with
// block_size square tiles and 16-element K steps.
void dispatch_masked_gemm(
    const MaskedOperands& ops,
    array& out,
    int block_size,
    int M,
    int N,
    int K,
    const Shape& batch_shape,
    const Strides& batch_strides,
    int64_t A_batch_str,
    int64_t B_batch_str) {
  const int bm = block_size, bn = block_size, bk = 16, wm = 2, wn = 2;
  const bool mn_aligned = M % bm == 0 && N % bn == 0;
  const bool k_aligned = K % bk == 0;
  const Dtype dtype = out.dtype();
  const char* storage = matmul_storage(dtype);
  const array* out_mask = ops.out_mask ? &*ops.out_mask : nullptr;
  const array* op_mask = ops.lhs_mask ? &*ops.lhs_mask : nullptr;
  const std::string entry = std::string("steel_gemm_block_outmask_") +
      masked_name(out_mask, dtype) + "_opmask_" + masked_name(op_mask, dtype) + "_" +
      (ops.transpose_a ? 't' : 'n') + (ops.transpose_b ? 't' : 'n') + "_" + storage + "_" +
      storage + "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_bk" +
      std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" + std::to_string(wn) +
      "_MN_" + (mn_aligned ? "t" : "n") + "aligned_K_" + (k_aligned ? "t" : "n") + "aligned";
  require_entry(entry);
  const int64_t tn = (int64_t(N) + bn - 1) / bn;
  const int64_t tm = (int64_t(M) + bm - 1) / bm;
  const uint64_t batch_size_out = out.size() / (uint64_t(M) * N);
  if (out.size() > max_matmul_elements || batch_size_out > 65535 || tn > 65535 || tm > 65535 ||
      ops.a_cols < 0 || ops.a_cols > int64_t(max_matmul_elements) || ops.b_cols < 0 ||
      ops.b_cols > int64_t(max_matmul_elements)) {
    throw std::invalid_argument("CrossTL block masked matmul launch exceeds its bounds.");
  }
  // GEMMParams in params.h order, each widened to int64; swizzle_log is 0.
  std::vector<int64_t> params = {
      M, N, K, ops.a_cols, ops.b_cols, N, tn, tm, A_batch_str, B_batch_str,
      int64_t(M) * N, 0, K / bk, int64_t(batch_shape.size())};
  std::vector<int32_t> mask_strides;
  if (out_mask) {
    push_mask_strides(mask_strides, *out_mask, -1, -2);
  }
  if (op_mask) {
    push_mask_strides(mask_strides, *ops.lhs_mask, -1, -2);
    push_mask_strides(mask_strides, *ops.rhs_mask, -1, -2);
  }
  std::vector<int32_t> shape(batch_shape.begin(), batch_shape.end());
  Strides strides = batch_strides;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(ops.a), matmul_span(ops.a), 0},
      {"B", storage, data(ops.b), matmul_span(ops.b), 0},
      {"D", storage, out.data<void>(), out.size(), 1},
      {"params", "int64", params.data(), params.size(), 0},
      {"batch_shape", "int32", shape.data(), shape.size(), 0},
      {"batch_strides", "int64", strides.data(), strides.size(), 0},
      {"mask_strides", "int32", mask_strides.data(), mask_strides.size(), 0},
  };
  if (out_mask) {
    buffers.push_back({"out_mask", masked_mask_dtype(*out_mask, dtype), data(*out_mask),
                       matmul_span(*out_mask), 0});
  }
  if (op_mask) {
    buffers.push_back({"lhs_mask", masked_mask_dtype(*ops.lhs_mask, dtype),
                       data(*ops.lhs_mask), matmul_span(*ops.lhs_mask), 0});
    buffers.push_back({"rhs_mask", masked_mask_dtype(*ops.rhs_mask, dtype),
                       data(*ops.rhs_mask), matmul_span(*ops.rhs_mask), 0});
  }
  const CrosstlMlxLaunch launch{
      {uint32_t(tn), uint32_t(tm), uint32_t(batch_size_out)},
      {32, uint32_t(wn), uint32_t(wm)}};
  dispatch_gather_entry(entry, buffers, out.size(), launch);
}

} // namespace

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const auto dtype = out.dtype();
  if (inputs.size() < 3 || inputs.size() > 5 ||
      (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != dtype || inputs[0].ndim() < 2 ||
      inputs[1].ndim() < 2 || (block_size_ != 32 && block_size_ != 64)) {
    throw std::invalid_argument(
        "CrossTL block masked matmul supports float32, float16 and bfloat16 only.");
  }
  const auto& a_pre = inputs[0];
  const auto& b_pre = inputs[1];
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    if (out.size() == 0) {
      return;
    }
    array zero(0, a_pre.dtype());
    array zeros(out.shape(), dtype, nullptr, {});
    const bool single = out.size() == 1;
    zeros.copy_shared_buffer(zero, Strides(out.ndim(), 0), {single, single, single}, 1);
    dispatch_copy(zeros, out);
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  const int M = a_pre.shape(-2);
  const int N = b_pre.shape(-1);
  const int K = a_pre.shape(-1);
  auto [transpose_a, a_cols, a] = gemv_operand(a_pre, M == 1);
  auto [transpose_b, b_cols, b] = gemv_operand(b_pre, N == 1);
  const bool has_op_mask = inputs.size() > 3;
  const bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;
  MaskedOperands ops{a, b, transpose_a, transpose_b, a_cols, b_cols, {}, {}, {}};
  if (has_out_mask) {
    ops.out_mask = masked_mask(inputs[2]);
    masked_mask_dtype(*ops.out_mask, dtype);
  }
  if (has_op_mask) {
    ops.lhs_mask = masked_mask(inputs[2 + has_out_mask]);
    ops.rhs_mask = masked_mask(inputs[3 + has_out_mask]);
    if (ops.lhs_mask->dtype() != ops.rhs_mask->dtype()) {
      throw std::invalid_argument("CrossTL block masked matmul operand masks differ.");
    }
    masked_mask_dtype(*ops.lhs_mask, dtype);
  }
  // Check and collapse batch dimensions, from the arrays the launch binds.
  std::vector<const array*> bound = {&ops.a, &ops.b};
  if (ops.out_mask) {
    bound.push_back(&*ops.out_mask);
  }
  if (ops.lhs_mask) {
    bound.push_back(&*ops.lhs_mask);
    bound.push_back(&*ops.rhs_mask);
  }
  Shape batch_shape{1};
  Strides A_batch_stride{0}, B_batch_stride{0};
  Strides outmask_bstride{0}, Amask_bstride{0}, Bmask_bstride{0};
  int64_t A_batch_str = 0, B_batch_str = 0;
  Strides batch_strides;
  if (out.ndim() > 2) {
    const size_t batch_ndim = out.ndim() - 2;
    batch_shape = Shape(out.shape().begin(), out.shape().end() - 2);
    std::vector<Strides> bstrides;
    for (const auto* value : bound) {
      bstrides.push_back(batch_strides_of(*value, batch_ndim));
    }
    A_batch_str = bstrides[0].back();
    B_batch_str = bstrides[1].back();
    for (const auto& bstr : bstrides) {
      batch_strides.insert(batch_strides.end(), bstr.begin(), bstr.end());
    }
    A_batch_stride = bstrides[0];
    B_batch_stride = bstrides[1];
    if (has_out_mask) {
      outmask_bstride = bstrides[2];
    }
    if (has_op_mask) {
      Amask_bstride = bstrides[has_out_mask + 2];
      Bmask_bstride = bstrides[has_out_mask + 3];
    }
  } else {
    batch_strides = Strides(bound.size(), 0);
  }
  if (std::min(M, N) == 1) {
    dispatch_masked_gemv(
        ops, out, block_size_, M, N, K, batch_shape, A_batch_stride, B_batch_stride,
        outmask_bstride, Amask_bstride, Bmask_bstride);
    return;
  }
  dispatch_masked_gemm(
      ops, out, block_size_, M, N, K, batch_shape, batch_strides, A_batch_str, B_batch_str);
}

} // namespace mlx::core
