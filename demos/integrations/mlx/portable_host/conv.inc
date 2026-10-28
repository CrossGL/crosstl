// Ports mlx/backend/metal/conv.cpp: Convolution::eval_gpu routes every
// convolution as conv.cpp does. It runs the explicit GEMM routes, which unfold
// the input with naive_unfold_nd (groups == 1) or naive_unfold_transpose_nd
// and multiply the unfolded rows with the steel GEMM, the separable
// depthwise_conv_1d kernel, the small-kernel depthwise_conv_2d kernel, the
// implicit GEMM routes, which run steel_conv's implicit_gemm_conv_2d and
// implicit_gemm_conv_3d, and the general implicit GEMM route, which runs
// steel_conv_general's implicit_gemm_conv_2d_general. Every other
// route conv.cpp takes is refused by name before any launch. Each launch binds
// the buffers and values conv.cpp binds; conv_layout checks the unfold,
// depthwise and implicit GEMM launches against the same routing and
// gemm_layout the products. crosstl_backend.cpp includes this file after the
// matrix ports, so it shares that translation unit's steel GEMM and copy
// helpers.
//
// Two launch choices differ from conv.cpp without changing a value. The unfold
// and depthwise kernels read only thread_position_in_grid and
// threads_per_grid, so they run as one-thread workgroups over conv.cpp's exact
// thread grid: Direct3D and OpenGL cannot clip the partial workgroups that
// conv.cpp's dispatch_threads leaves. The host has no maxBufferLength, so an
// unfold tile holds at most the elements every translated launch is bounded
// by and at most 65535 rows, the Z workgroups of one launch; conv.cpp's
// MLX_CONV_UNFOLD_TILE_ROWS still caps it further. depthwise_conv_2d and the
// implicit GEMM kernels run with conv.cpp's threadgroups and parameters; the
// host names each depthwise_conv_2d and implicit_gemm_conv_2d_general
// pipeline by conv.cpp's hash name, whose translated entry has those function
// constants frozen.

#include <numeric>

#include "mlx/backend/metal/kernels/steel/conv/params.h"

namespace mlx::core {
namespace {

[[noreturn]] void conv_unported(const char* route) {
  throw std::invalid_argument(
      std::string("CrossTL convolution does not port conv.cpp's ") + route + " yet.");
}

// conv.cpp ensure_row_contiguous.
array conv_row_contiguous(const array& x) {
  if (!x.flags().row_contiguous) {
    return contiguous_copy(x);
  }
  // MLX leaves the stride of a size-one axis of a row-contiguous array
  // arbitrary, and conv.cpp binds it as it is. Only coordinate zero of such
  // an axis is ever read, so the dense strides name the same elements; the
  // port binds them, and every route sees conv.cpp's contiguous layout.
  Strides dense(x.ndim(), 1);
  for (int i = int(x.ndim()) - 2; i >= 0; --i) {
    dense[i] = dense[i + 1] * x.shape(i + 1);
  }
  if (dense == x.strides()) {
    return x;
  }
  array view(x.shape(), x.dtype(), nullptr, {});
  view.copy_shared_buffer(x, dense, x.flags(), x.data_size());
  return view;
}

uint32_t conv_groups(int64_t value) {
  if (value < 1 || value > 65535) {
    throw std::invalid_argument("CrossTL convolution launch exceeds 65535 workgroups.");
  }
  return uint32_t(value);
}

void conv_launch(
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
    throw std::runtime_error(std::string("CrossTL native convolution failed: ") + error);
  }
}

int conv_int(int64_t value) {
  if (value < 0 || value > int64_t(max_matmul_elements)) {
    throw std::invalid_argument("CrossTL convolution exceeds its bounds.");
  }
  return int(value);
}

// conv.cpp max_unfold_rows, with the host's bounds in place of maxBufferLength.
int conv_unfold_rows(int64_t row_elements, int total_rows) {
  uint64_t max_rows = row_elements == 0 ? uint64_t(total_rows)
                                        : max_matmul_elements / uint64_t(row_elements);
  max_rows = std::min<uint64_t>(max_rows, 65535);
  if (int forced = env::get_var("MLX_CONV_UNFOLD_TILE_ROWS", 0); forced > 0) {
    max_rows = std::min(max_rows, static_cast<uint64_t>(forced));
  }
  if (max_rows == 0) {
    throw std::invalid_argument("CrossTL convolution unfold row exceeds its bounds.");
  }
  return static_cast<int>(std::min(max_rows, static_cast<uint64_t>(total_rows)));
}

// One naive_unfold launch over conv.cpp's (C, prod(wS), tile_rows) threads.
template <int N>
void dispatch_conv_unfold(
    const char* kernel,
    const array& in,
    const array& tile,
    const MLXConvParams<N>& conv_params,
    int row_offset,
    int tile_rows) {
  int64_t kernel_points = 1;
  for (int i = 0; i < N; ++i) {
    kernel_points *= conv_params.wS[i];
  }
  const char* storage = matmul_storage(in.dtype());
  const std::string entry =
      std::string(kernel) + storage + "_" + std::to_string(N);
  require_entry(entry);
  MLXConvParams<N> params = conv_params;
  int32_t offset = row_offset;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"in", storage, data(in), matmul_span(in), 0},
      {"out", storage, data(tile), tile.size(), 1},
      {"params", "uint8", &params, sizeof(params), 0},
      {"row_offset", "int32", &offset, 1, 0},
  };
  const CrosstlMlxLaunch launch{
      {conv_groups(conv_params.C), conv_groups(kernel_points), conv_groups(tile_rows)},
      {1, 1, 1}};
  conv_launch(entry, buffers, tile.size(), launch);
}

// conv.cpp explicit_gemm_conv_ND_gpu.
template <int N>
void conv_explicit_gemm(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<N>& conv_params) {
  const int implicit_M = conv_int(out.size() / conv_params.O);
  const int implicit_K = conv_int(wt.size() / conv_params.O);
  const int implicit_N = conv_params.O;

  // The weights as a column-major K x N matrix.
  array wt_reshaped({implicit_K, implicit_N}, wt.dtype(), nullptr, {});
  auto wt_flags = wt.flags();
  wt_flags.row_contiguous = false;
  wt_flags.col_contiguous = true;
  wt_reshaped.copy_shared_buffer(wt, {1, implicit_K}, wt_flags, wt.data_size());

  // 2D view of the output; each tile writes a row window of it.
  Strides out_2d_strides{out.strides(-2), out.strides(-1)};
  array out_2d({implicit_M, implicit_N}, out.dtype(), nullptr, {});
  out_2d.copy_shared_buffer(out, out_2d_strides, out.flags(), out.data_size());

  const int max_rows = conv_unfold_rows(implicit_K, implicit_M);
  array in_unfolded({max_rows, implicit_K}, in.dtype(), nullptr, {});
  in_unfolded.set_data(allocator::malloc(in_unfolded.nbytes()));

  for (int row_offset = 0; row_offset < implicit_M; row_offset += max_rows) {
    const int tile_rows = std::min(max_rows, implicit_M - row_offset);
    array in_tile({tile_rows, implicit_K}, in.dtype(), nullptr, {});
    in_tile.copy_shared_buffer(
        in_unfolded, in_unfolded.strides(), in_unfolded.flags(), in_tile.size());
    dispatch_conv_unfold("naive_unfold_nd_", in, in_tile, conv_params, row_offset, tile_rows);

    array out_tile({tile_rows, implicit_N}, out.dtype(), nullptr, {});
    out_tile.copy_shared_buffer(
        out_2d, out_2d_strides, out_2d.flags(), out_tile.size(),
        static_cast<int64_t>(row_offset) * out_2d_strides[0]);
    // steel_matmul collapses two-dimensional operands to one batch.
    dispatch_steel_gemm(
        in_tile, wt_reshaped, out_tile, tile_rows, implicit_N, implicit_K, 1, implicit_K,
        implicit_K, false, true, Shape{1}, Strides{0}, Strides{0});
  }
}

// conv.cpp explicit_gemm_conv_group_ND_gpu.
template <int N>
void conv_explicit_gemm_group(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<N>& conv_params) {
  const int groups = conv_params.groups;
  const int C_per_group = conv_params.C / conv_params.groups;
  const int O_per_group = conv_params.O / conv_params.groups;
  const int implicit_M = conv_int(out.size() / conv_params.O);
  const int implicit_K = conv_int(wt.size() / conv_params.O);
  const int implicit_N = O_per_group;

  int kernel_size = 1;
  for (int i = 0; i < N; ++i) {
    kernel_size *= conv_params.wS[i];
  }

  // Transpose the weights so that each group's rows are contiguous.
  array wt_view({wt.shape(0), C_per_group, kernel_size}, wt.dtype(), nullptr, {});
  wt_view.copy_shared_buffer(wt, {wt.strides(0), 1, C_per_group}, wt.flags(), wt.size());
  array wt_transpose = contiguous_copy(wt_view);

  Strides out_2d_strides{out.strides(-2), out.strides(-1)};
  array out_2d({implicit_M, conv_params.O}, out.dtype(), nullptr, {});
  out_2d.copy_shared_buffer(out, out_2d_strides, out.flags(), out.data_size());

  const int max_rows = conv_unfold_rows(int64_t(implicit_K) * groups, implicit_M);
  array in_unfolded({max_rows, conv_int(int64_t(implicit_K) * groups)}, in.dtype(), nullptr, {});
  in_unfolded.set_data(allocator::malloc(in_unfolded.nbytes()));

  for (int row_offset = 0; row_offset < implicit_M; row_offset += max_rows) {
    const int tile_rows = std::min(max_rows, implicit_M - row_offset);
    array in_tile({tile_rows, implicit_K * groups}, in.dtype(), nullptr, {});
    in_tile.copy_shared_buffer(
        in_unfolded, in_unfolded.strides(), in_unfolded.flags(), in_tile.size());
    dispatch_conv_unfold(
        "naive_unfold_transpose_nd_", in, in_tile, conv_params, row_offset, tile_rows);

    array out_tile({tile_rows, conv_params.O}, out.dtype(), nullptr, {});
    out_tile.copy_shared_buffer(
        out_2d, out_2d_strides, out_2d.flags(), out_tile.size(),
        static_cast<int64_t>(row_offset) * out_2d_strides[0]);
    // conv.cpp passes batch strides {0}; the host passes A's and B's, which
    // the kernel reads only with has_batch.
    dispatch_steel_gemm_regular(
        in_tile, wt_transpose, out_tile, tile_rows, implicit_N, implicit_K, groups,
        int64_t(implicit_K) * groups, implicit_K, int64_t(implicit_N) * groups, false, true,
        Shape{1}, Strides{0, 0}, int64_t(implicit_K), int64_t(implicit_N) * implicit_K,
        int64_t(implicit_N));
  }
}

template <int N>
void conv_explicit(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<N>& conv_params) {
  if (conv_params.groups > 1) {
    conv_explicit_gemm_group(in, wt, out, conv_params);
  } else {
    conv_explicit_gemm(in, wt, out, conv_params);
  }
}

// conv.cpp depthwise_conv_1D_gpu. Every host buffer stays below 2^31
// elements, so the int64 ``_large`` kernel is never selected.
void dispatch_depthwise_conv_1d(const array& in, const array& wt, array& out) {
  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("depthwise_conv_1d_") + storage;
  require_entry(entry);
  const int64_t B = in.shape(0);
  const int64_t Tout = out.shape(1);
  const int64_t D = in.shape(2);
  int32_t K = wt.shape(1);
  int32_t strides[3] = {
      conv_int(in.strides(0)), conv_int(in.strides(1)), conv_int(in.strides(2))};
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"in", storage, data(in), matmul_span(in), 0},
      {"w", storage, data(wt), matmul_span(wt), 0},
      {"out", storage, out.data<void>(), out.size(), 1},
      {"strides", "int32", strides, 3, 0},
      {"kernel_size", "int32", &K, 1, 0},
  };
  const CrosstlMlxLaunch launch{
      {conv_groups(D), conv_groups(Tout), conv_groups(B)}, {1, 1, 1}};
  conv_launch(entry, buffers, out.size(), launch);
}

// conv.cpp's int helper parameters, refused outside int range.
int conv_signed(int64_t value) {
  if (value < -int64_t(INT32_MAX) || value > int64_t(INT32_MAX)) {
    throw std::invalid_argument("CrossTL convolution exceeds its bounds.");
  }
  return int(value);
}

// implicit_gemm_conv_2D_gpu's ImplicitGemmConv2DParams helper values.
steel::ImplicitGemmConv2DParams conv_implicit_params(
    const MLXConvParams<2>& p, int M, int N, int K, int k_iters, int tn, int tm, int bk) {
  const int64_t sign = p.flip ? -1 : 1;
  const int64_t ijw = conv_signed(p.in_strides[2] * p.kdil[1]);
  const int64_t ijh = conv_signed(p.in_strides[1] * p.kdil[0]);
  return steel::ImplicitGemmConv2DParams{
      M,
      N,
      K,
      k_iters,
      conv_signed(sign * ijw),
      conv_signed(sign * (ijh - (p.wS[1] - 1) * ijw)),
      conv_signed(bk - sign * (p.wS[0] - 1) * ijh - sign * (p.wS[1] - 1) * ijw),
      tn,
      tm,
      0};
}

// implicit_gemm_conv_3D_gpu's ImplicitGemmConv3DParams helper values.
steel::ImplicitGemmConv3DParams conv_implicit_params(
    const MLXConvParams<3>& p, int M, int N, int K, int k_iters, int tn, int tm, int bk) {
  const int64_t sign = p.flip ? -1 : 1;
  const int64_t ijw = conv_signed(p.in_strides[3] * p.kdil[2]);
  const int64_t ijh = conv_signed(p.in_strides[2] * p.kdil[1]);
  const int64_t ijd = conv_signed(p.in_strides[1] * p.kdil[0]);
  return steel::ImplicitGemmConv3DParams{
      M,
      N,
      K,
      k_iters,
      conv_signed(sign * ijw),
      conv_signed(sign * (ijh - (p.wS[2] - 1) * ijw)),
      conv_signed(sign * (ijd - (p.wS[1] - 1) * ijh - (p.wS[2] - 1) * ijw)),
      conv_signed(
          bk - sign * (p.wS[0] - 1) * ijd - sign * (p.wS[1] - 1) * ijh -
          sign * (p.wS[2] - 1) * ijw),
      tn,
      tm,
      0};
}

// conv.cpp implicit_gemm_conv_2D_gpu and implicit_gemm_conv_3D_gpu: one
// steel_conv launch of conv.cpp's tiles, with swizzle_log 0. The last weight
// tile of the last group reads rows past wt, which feed only output columns
// the kernel never stores; conv_layout binds them as poisoned storage.
template <int N>
void conv_implicit_gemm(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<N>& conv_params) {
  static_assert(N == 2 || N == 3, "steel_conv instantiates 2-D and 3-D kernels");
  const int groups = conv_params.groups;
  const int C_per_group = conv_params.C / groups;
  const int O_per_group = conv_params.O / groups;
  int64_t rows = conv_params.N;
  int64_t taps = 1;
  for (int i = 0; i < N; ++i) {
    rows *= conv_params.oS[i];
    taps *= conv_params.wS[i];
  }
  const int implicit_M = conv_int(rows);
  const int implicit_N = O_per_group;
  const int implicit_K = conv_int(taps * C_per_group);

  int wm = 2, wn = 2;
  int bm = implicit_M >= 8192 && C_per_group >= 64 ? 64 : 32;
  int bn = (bm == 64 || implicit_N >= 64) ? 64 : 32;
  const int bk = 16;
  if (implicit_N <= 16) {
    bn = 8;
    wm = 4;
    wn = 1;
  }
  const int tn = (implicit_N + bn - 1) / bn;
  const int tm = (implicit_M + bm - 1) / bm;

  // steel_conv.metal's small-channel loaders serve 2-D convolutions only.
  int n_channel_specialization = 0;
  int gemm_k_iters = conv_int(taps * ((C_per_group + bk - 1) / bk));
  if (N == 2 && C_per_group <= 2) {
    gemm_k_iters = (implicit_K + bk - 1) / bk;
    n_channel_specialization = C_per_group;
  } else if (N == 2 && C_per_group <= 4) {
    gemm_k_iters = conv_int((taps * 4 + bk - 1) / bk);
    n_channel_specialization = C_per_group;
  }
  bool small_filter = !n_channel_specialization;
  for (int i = 0; i < N; ++i) {
    small_filter = small_filter && conv_params.wS[i] <= 16;
  }
  auto gemm_params = conv_implicit_params(
      conv_params, implicit_M, implicit_N, implicit_K, gemm_k_iters, tn, tm, bk);

  const char* storage = matmul_storage(out.dtype());
  std::string entry = std::string("implicit_gemm_conv_") + std::to_string(N) + "d_" +
      storage + "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_bk" +
      std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" + std::to_string(wn);
  if (N == 2) {
    entry += std::string("_channel_") +
        (n_channel_specialization ? std::to_string(n_channel_specialization) : "l");
  }
  entry += small_filter ? "_filter_s" : "_filter_l";
  require_entry(entry);

  MLXConvParams<N> params = conv_params;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(in), matmul_span(in), 0},
      {"B", storage, data(wt), matmul_span(wt), 0},
      {"C", storage, out.data<void>(), out.size(), 1},
      {"params", "uint8", &params, sizeof(params), 0},
      {"gemm_params", "int32", &gemm_params, sizeof(gemm_params) / sizeof(int32_t), 0},
  };
  const CrosstlMlxLaunch launch{
      {conv_groups(tn), conv_groups(tm), conv_groups(groups)},
      {32, uint32_t(wn), uint32_t(wm)}};
  conv_launch(entry, buffers, out.size(), launch);
}

// conv.cpp depthwise_conv_2D_gpu: (tc, tw, th) threadgroups, the function
// constants named by conv.cpp's pipeline hash name.
void dispatch_depthwise_conv_2d(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params) {
  const char* storage = matmul_storage(out.dtype());
  const int tc = 8;
  const int tw = 8;
  const int th = 4;
  std::string entry = std::string("depthwise_conv_2d_") + storage + "_ker_h_" +
      std::to_string(conv_params.wS[0]) + "_ker_w_" + std::to_string(conv_params.wS[1]) +
      "_str_h_" + std::to_string(conv_params.str[0]) + "_str_w_" +
      std::to_string(conv_params.str[1]) + "_tgp_h_" + std::to_string(th) + "_tgp_w_" +
      std::to_string(tw) + "_do_flip_" + (conv_params.flip ? "t" : "n");
  require_entry(entry);
  MLXConvParams<2> params = conv_params;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"in", storage, data(in), matmul_span(in), 0},
      {"wt", storage, data(wt), matmul_span(wt), 0},
      {"out", storage, out.data<void>(), out.size(), 1},
      {"params", "uint8", &params, sizeof(params), 0},
  };
  const CrosstlMlxLaunch launch{
      {conv_groups(conv_params.C / tc), conv_groups((conv_params.oS[1] + tw - 1) / tw),
       conv_groups(
           ((static_cast<int64_t>(conv_params.oS[0]) + th - 1) / th) * conv_params.N)},
      {uint32_t(tc), uint32_t(tw), uint32_t(th)}};
  conv_launch(entry, buffers, out.size(), launch);
}

// conv.cpp dispatch_conv_2D_gpu. The host has no recommendedMaxWorkingSetSize
// for winograd_batch_step, so it refuses every product conv.cpp would send to
// winograd once the working set allowed.
// conv.cpp pad_in_channels_conv_2D_gpu: zero-pad the input and weight
// channels to a multiple of 16 through Pad and run the implicit GEMM on the
// padded parameters. Only the input channels grow, so the kernel writes
// straight into out. conv.cpp routes only ungrouped convolutions here.
void conv_pad_in_channels(
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const MLXConvParams<2>& conv_params) {
  const int extra_c = ((conv_params.C + 15) / 16) * 16 - conv_params.C;
  const Stream s = out.primitive().stream();
  auto pad_array = [&](const array& x) {
    auto shape = x.shape();
    shape.back() += extra_c;
    array padded(shape, x.dtype(), nullptr, {});
    array zero(0, x.dtype());
    Pad(s, {-1}, {0}, {extra_c}).eval_gpu({x, zero}, padded);
    return padded;
  };
  const array in = pad_array(in_pre);
  const array wt = pad_array(wt_pre);
  conv_implicit_gemm(
      in, wt, out, MLXConvParams<2>::with_padded_channels(conv_params, 0, extra_c));
}

// conv.cpp implicit_gemm_conv_2D_general_gpu: one steel_conv_general launch
// whose grid z takes one output row and column residue. base_h and base_w
// hold each residue's first filter tap and tap count, and the align_C
// function constant is frozen into the entry conv.cpp's hash name selects.
void conv_implicit_gemm_general(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params) {
  const int implicit_M = conv_int(
      static_cast<int64_t>(conv_params.N) * conv_params.oS[0] * conv_params.oS[1]);
  const int implicit_N = conv_params.O;
  const int implicit_K =
      conv_int(static_cast<int64_t>(conv_params.wS[0]) * conv_params.wS[1] * conv_params.C);
  const int wm = 2, wn = 2;

  const int f_wgt_jump_h =
      std::lcm(conv_params.idil[0], conv_params.kdil[0]) / conv_params.kdil[0];
  const int f_wgt_jump_w =
      std::lcm(conv_params.idil[1], conv_params.kdil[1]) / conv_params.kdil[1];
  const int f_out_jump_h =
      std::lcm(conv_params.idil[0], conv_params.str[0]) / conv_params.str[0];
  const int f_out_jump_w =
      std::lcm(conv_params.idil[1], conv_params.str[1]) / conv_params.str[1];
  const int adj_out_h = (conv_params.oS[0] + f_out_jump_h - 1) / f_out_jump_h;
  const int adj_out_w = (conv_params.oS[1] + f_out_jump_w - 1) / f_out_jump_w;
  const int adj_out_hw = conv_int(static_cast<int64_t>(adj_out_h) * adj_out_w);
  const int adj_implicit_m = conv_int(static_cast<int64_t>(conv_params.N) * adj_out_hw);
  steel::Conv2DGeneralJumpParams jump_params{
      f_wgt_jump_h,
      f_wgt_jump_w,
      f_out_jump_h,
      f_out_jump_w,
      adj_out_h,
      adj_out_w,
      adj_out_hw,
      adj_implicit_m};

  // conv.cpp's base_h and base_w loops: each residue's first tap on an
  // input pixel and the taps it steps through.
  auto bases = [&](int axis, int residues, int step) {
    std::vector<steel::Conv2DGeneralBaseInfo> table(residues);
    const int jump = conv_params.flip ? -conv_params.kdil[axis] : conv_params.kdil[axis];
    const int init =
        conv_params.flip ? (conv_params.wS[axis] - 1) * conv_params.kdil[axis] : 0;
    for (int i = 0; i < residues; ++i) {
      int loop = i * conv_params.str[axis] - conv_params.pad[axis] + init;
      int base = 0;
      while (base < conv_params.wS[axis] && loop % conv_params.idil[axis] != 0) {
        base++;
        loop += jump;
      }
      table[i] = {base, ((conv_params.wS[axis] - base) + step - 1) / step};
    }
    return table;
  };
  const auto base_h = bases(0, f_out_jump_h, f_wgt_jump_h);
  const auto base_w = bases(1, f_out_jump_w, f_wgt_jump_w);

  const int bm = adj_implicit_m >= 8192 && conv_params.C >= 64 ? 64 : 32;
  const int bn = (bm == 64 && implicit_N >= 64) ? 64 : 32;
  const int bk = 16;
  const int tn = (implicit_N + bn - 1) / bn;
  const int tm = (adj_implicit_m + bm - 1) / bm;
  const int gemm_k_iters = (conv_params.C + bk - 1) / bk;
  const bool align_C = conv_params.C % bk == 0;
  auto gemm_params = conv_implicit_params(
      conv_params, implicit_M, implicit_N, implicit_K, gemm_k_iters, tn, tm, bk);

  const char* storage = matmul_storage(out.dtype());
  const std::string entry = std::string("implicit_gemm_conv_2d_general_") + storage +
      "_bm" + std::to_string(bm) + "_bn" + std::to_string(bn) + "_bk" +
      std::to_string(bk) + "_wm" + std::to_string(wm) + "_wn" + std::to_string(wn) +
      "_alC_" + std::to_string(align_C);
  require_entry(entry);

  MLXConvParams<2> params = conv_params;
  auto data = [](const array& value) { return const_cast<void*>(value.data<void>()); };
  auto words = [](const auto& value) { return sizeof(value) / sizeof(int32_t); };
  std::vector<CrosstlMlxBuffer> buffers = {
      {"A", storage, data(in), matmul_span(in), 0},
      {"B", storage, data(wt), matmul_span(wt), 0},
      {"C", storage, out.data<void>(), out.size(), 1},
      {"params", "uint8", &params, sizeof(params), 0},
      {"gemm_params", "int32", &gemm_params, words(gemm_params), 0},
      {"jump_params", "int32", &jump_params, words(jump_params), 0},
      {"base_h", "int32", const_cast<steel::Conv2DGeneralBaseInfo*>(base_h.data()),
       base_h.size() * words(base_h[0]), 0},
      {"base_w", "int32", const_cast<steel::Conv2DGeneralBaseInfo*>(base_w.data()),
       base_w.size() * words(base_w[0]), 0},
  };
  const CrosstlMlxLaunch launch{
      {conv_groups(tn), conv_groups(tm),
       conv_groups(static_cast<int64_t>(f_out_jump_h) * f_out_jump_w)},
      {32, uint32_t(wn), uint32_t(wm)}};
  conv_launch(entry, buffers, out.size(), launch);
}

void conv_dispatch_2d(
    const array& in,
    const array& wt,
    array& out,
    const MLXConvParams<2>& conv_params) {
  const bool is_stride_one = conv_params.str[0] == 1 && conv_params.str[1] == 1;
  const bool is_kdil_one = conv_params.kdil[0] == 1 && conv_params.kdil[1] == 1;
  const bool is_idil_one = conv_params.idil[0] == 1 && conv_params.idil[1] == 1;
  if (conv_params.groups > 1) {
    const int C_per_group = conv_params.C / conv_params.groups;
    const int O_per_group = conv_params.O / conv_params.groups;
    if (!is_idil_one) {
      return conv_explicit_gemm_group(in, wt, out, conv_params);
    }
    if (C_per_group == 1 && O_per_group == 1 && is_kdil_one && conv_params.wS[0] <= 7 &&
        conv_params.wS[1] <= 7 && conv_params.str[0] <= 2 && conv_params.str[1] <= 2 &&
        conv_params.wt_strides[1] == conv_params.wS[1] && conv_params.C % 16 == 0 &&
        conv_params.C == conv_params.O) {
      return dispatch_depthwise_conv_2d(in, wt, out, conv_params);
    }
    if ((C_per_group <= 4 || C_per_group % 16 == 0) &&
        (O_per_group <= 16 || O_per_group % 16 == 0)) {
      return conv_implicit_gemm(in, wt, out, conv_params);
    }
    return conv_explicit_gemm_group(in, wt, out, conv_params);
  }
  const bool inp_large =
      (static_cast<int64_t>(conv_params.N) * conv_params.iS[0] * conv_params.iS[1]) >= 4096;
  const bool channels_large = (conv_params.C + conv_params.O) >= 256;
  const bool out_large =
      (static_cast<int64_t>(conv_params.N) * conv_params.oS[0] * conv_params.oS[1]) >= 256;
  if (!conv_params.flip && is_stride_one && is_kdil_one && is_idil_one &&
      conv_params.wS[0] == 3 && conv_params.wS[1] == 3 && conv_params.C % 32 == 0 &&
      conv_params.O % 32 == 0 && inp_large && channels_large) {
    conv_unported("winograd_conv_2D_gpu");
  }
  const bool specialized_channels = (conv_params.C <= 4 || conv_params.C % 16 == 0) &&
      (conv_params.O <= 16 || conv_params.O % 16 == 0);
  const bool out_channels_aligned = conv_params.O <= 16 || conv_params.O % 16 == 0;
  if (is_idil_one && is_stride_one && out_large && !specialized_channels &&
      out_channels_aligned && (conv_params.wS[0] * conv_params.wS[1]) >= 9) {
    return conv_pad_in_channels(in, wt, out, conv_params);
  }
  if (is_idil_one && specialized_channels) {
    return conv_implicit_gemm(in, wt, out, conv_params);
  }
  if ((conv_params.C % 16 == 0 && conv_params.O % 16 == 0) || out_large) {
    return conv_implicit_gemm_general(in, wt, out, conv_params);
  }
  conv_explicit_gemm(in, wt, out, conv_params);
}

// conv.cpp dispatch_conv_3D_gpu.
void conv_dispatch_3d(
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const MLXConvParams<3>& p) {
  const bool is_idil_one = p.idil[0] == 1 && p.idil[1] == 1 && p.idil[2] == 1;
  const int C_per_group = p.C / p.groups;
  const int O_per_group = p.O / p.groups;
  const bool mod16_channels =
      C_per_group % 16 == 0 && (O_per_group <= 16 || O_per_group % 16 == 0);
  if (is_idil_one && !mod16_channels && p.groups == 1) {
    conv_unported("pad_and_slice_conv_3D_gpu");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = conv_row_contiguous(in_pre);
  auto wt = conv_row_contiguous(wt_pre);
  if (p.groups == 1 && p.flip && p.str[0] == 1 && p.str[1] == 1 && p.str[2] == 1 &&
      p.idil[0] == 2 && p.idil[1] == 2 && p.idil[2] == 2 && p.kdil[0] == 1 &&
      p.kdil[1] == 1 && p.kdil[2] == 1 && p.wS[0] == 2 && p.wS[1] == 2 && p.wS[2] == 2 &&
      p.pad[0] == 1 && p.pad[1] == 1 && p.pad[2] == 1 &&
      static_cast<int64_t>(p.oS[0]) == 2LL * p.iS[0] &&
      static_cast<int64_t>(p.oS[1]) == 2LL * p.iS[1] &&
      static_cast<int64_t>(p.oS[2]) == 2LL * p.iS[2]) {
    conv_unported("stride_two_conv_transpose_3D_gpu");
  }
  constexpr int kSmallKdLimit3D = 7;
  if (is_idil_one && mod16_channels && p.groups == 1 && p.N == 1 &&
      p.wS[0] <= kSmallKdLimit3D && p.str[0] == 1 && p.kdil[0] == 1 && p.pad[0] == 0) {
    conv_unported("small_kd_conv_3D_gpu");
  }
  if (is_idil_one && mod16_channels) {
    return conv_implicit_gemm(in, wt, out, p);
  }
  conv_explicit(in, wt, out, p);
}

// conv.cpp conv_1D_gpu.
void conv_1d(
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    int groups,
    bool flip) {
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = conv_row_contiguous(in_pre);
  auto wt = conv_row_contiguous(wt_pre);
  const bool is_idil_one = in_dilation[0] == 1;
  const int C = in.shape(2);
  const int O = wt.shape(0);
  if (is_idil_one && groups == C && groups == O && wt_strides[0] == 1 &&
      wt_dilation[0] == 1 && padding[0] == 0 && !flip) {
    return dispatch_depthwise_conv_1d(in, wt, out);
  }
  const int C_per_group = C / groups;
  const int O_per_group = O / groups;
  if (is_idil_one && (C_per_group <= 4 || C_per_group % 16 == 0) &&
      (O_per_group <= 16 || O_per_group % 16 == 0)) {
    MLXConvParams<2> conv_params{
        static_cast<int>(in.shape(0)),
        C,
        O,
        {static_cast<int>(in.shape(1)), 1},
        {static_cast<int>(wt.shape(1)), 1},
        {static_cast<int>(out.shape(1)), 1},
        {wt_strides[0], 1},
        {padding[0], 0},
        {wt_dilation[0], 1},
        {in_dilation[0], 1},
        {in.strides()[0], in.strides()[1], 0, in.strides()[2]},
        {wt.strides()[0], wt.strides()[1], 0, wt.strides()[2]},
        {out.strides()[0], out.strides()[1], 0, out.strides()[2]},
        groups,
        flip};
    return conv_dispatch_2d(in, wt, out, conv_params);
  }
  MLXConvParams<1> conv_params{
      static_cast<int>(in.shape(0)),
      static_cast<int>(in.shape(2)),
      static_cast<int>(wt.shape(0)),
      {static_cast<int>(in.shape(1))},
      {static_cast<int>(wt.shape(1))},
      {static_cast<int>(out.shape(1))},
      {wt_strides[0]},
      {padding[0]},
      {wt_dilation[0]},
      {in_dilation[0]},
      {in.strides()[0], in.strides()[1], in.strides()[2]},
      {wt.strides()[0], wt.strides()[1], wt.strides()[2]},
      {out.strides()[0], out.strides()[1], out.strides()[2]},
      groups,
      flip};
  conv_explicit(in, wt, out, conv_params);
}

// conv.cpp conv_2D_gpu.
void conv_2d(
    const array& in_pre,
    const array& wt_pre,
    array& out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    int groups,
    bool flip) {
  out.set_data(allocator::malloc(out.nbytes()));
  auto in = conv_row_contiguous(in_pre);
  auto wt = conv_row_contiguous(wt_pre);
  MLXConvParams<2> conv_params{
      static_cast<int>(in.shape(0)),
      static_cast<int>(in.shape(3)),
      static_cast<int>(wt.shape(0)),
      {static_cast<int>(in.shape(1)), static_cast<int>(in.shape(2))},
      {static_cast<int>(wt.shape(1)), static_cast<int>(wt.shape(2))},
      {static_cast<int>(out.shape(1)), static_cast<int>(out.shape(2))},
      {wt_strides[0], wt_strides[1]},
      {padding[0], padding[1]},
      {wt_dilation[0], wt_dilation[1]},
      {in_dilation[0], in_dilation[1]},
      {in.strides(0), in.strides(1), in.strides(2), in.strides(3)},
      {wt.strides(0), wt.strides(1), wt.strides(2), wt.strides(3)},
      {out.strides(0), out.strides(1), out.strides(2), out.strides(3)},
      groups,
      flip,
  };
  conv_dispatch_2d(in, wt, out, conv_params);
}

// conv.cpp conv_3D_gpu: the parameters carry the contiguous strides that
// dispatch_conv_3D_gpu's row-contiguous operands have.
void conv_3d(
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& padding,
    const std::vector<int>& wt_strides,
    const std::vector<int>& wt_dilation,
    const std::vector<int>& in_dilation,
    int groups,
    bool flip) {
  constexpr int NDIM = 3;
  int64_t in_arr_strides[NDIM + 2];
  int64_t wt_arr_strides[NDIM + 2];
  in_arr_strides[NDIM + 1] = wt_arr_strides[NDIM + 1] = 1;
  for (int i = NDIM; i >= 0; i--) {
    in_arr_strides[i] = in_arr_strides[i + 1] * in.shape(i + 1);
    wt_arr_strides[i] = wt_arr_strides[i + 1] * wt.shape(i + 1);
  }
  MLXConvParams<3> conv_params{
      static_cast<int>(in.shape(0)),
      static_cast<int>(in.shape(4)),
      static_cast<int>(wt.shape(0)),
      {static_cast<int>(in.shape(1)), static_cast<int>(in.shape(2)),
       static_cast<int>(in.shape(3))},
      {static_cast<int>(wt.shape(1)), static_cast<int>(wt.shape(2)),
       static_cast<int>(wt.shape(3))},
      {static_cast<int>(out.shape(1)), static_cast<int>(out.shape(2)),
       static_cast<int>(out.shape(3))},
      {wt_strides[0], wt_strides[1], wt_strides[2]},
      {padding[0], padding[1], padding[2]},
      {wt_dilation[0], wt_dilation[1], wt_dilation[2]},
      {in_dilation[0], in_dilation[1], in_dilation[2]},
      {in_arr_strides[0], in_arr_strides[1], in_arr_strides[2], in_arr_strides[3],
       in_arr_strides[4]},
      {wt_arr_strides[0], wt_arr_strides[1], wt_arr_strides[2], wt_arr_strides[3],
       wt_arr_strides[4]},
      {out.strides(0), out.strides(1), out.strides(2), out.strides(3), out.strides(4)},
      groups,
      flip,
  };
  conv_dispatch_3d(in, wt, out, conv_params);
}

} // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  require_runtime();
  const auto dtype = out.dtype();
  // conv.metal instantiates its kernels for float, half and bfloat16_t.
  if (inputs.size() != 2 || (dtype != float32 && dtype != float16 && dtype != bfloat16) ||
      inputs[0].dtype() != dtype || inputs[1].dtype() != dtype) {
    throw std::invalid_argument(
        "CrossTL convolution supports float32, float16 and bfloat16 only.");
  }
  const array& in = inputs[0];
  const array& wt = inputs[1];
  if (out.ndim() < 3 || out.ndim() > 5 || in.ndim() != out.ndim() ||
      wt.ndim() != out.ndim()) {
    throw std::invalid_argument(
        "[Convolution::eval_gpu] Only supports 1D, 2D or 3D convolutions.");
  }
  // No output element launches no thread on any route.
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }
  if (in.size() == 0 || wt.size() == 0) {
    throw std::invalid_argument("CrossTL convolution requires nonempty operands.");
  }
  if (out.ndim() == 5) {
    conv_3d(
        in, wt, out, padding_lo_, kernel_strides_, kernel_dilation_, input_dilation_, groups_,
        flip_);
  } else if (out.ndim() == 4) {
    conv_2d(
        in, wt, out, padding_lo_, kernel_strides_, kernel_dilation_, input_dilation_, groups_,
        flip_);
  } else {
    conv_1d(
        in, wt, out, padding_lo_, kernel_strides_, kernel_dilation_, input_dilation_, groups_,
        flip_);
  }
}

} // namespace mlx::core
