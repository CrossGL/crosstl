// Ports mlx/backend/metal/scaled_dot_product_attention.cpp's
// ScaledDotProductAttention::use_fallback and supports_bool_mask, and the
// forward paths of its eval_gpu: queries of at most eight positions run the
// translated sdpa_vector kernel, or sdpa_vector_2pass_1 (its grouped-query
// variant for long single-token queries) and sdpa_vector_2pass_2, and longer
// queries run the translated steel_attention kernel, with
// scaled_dot_product_attention.cpp's buffers, function constants,
// threadgroups and grids. The host reports no Apple GPU architecture and no
// NAX, so the device-class rules that choose the two-pass kernels, their
// block count and the full-attention tiles take the base device's branch.
// The blocked head-dim-512 attention is refused until it is ported.
// ScaledDotProductAttentionVJP's use_fallback and eval_gpu run the
// translated sdpa_vjp_odo, sdpa_vjp_ds and sdpa_vjp_reduce kernels around the
// steel matrix products of sdpa_vjp_blocked, and sdpa_vjp_layout checks each
// of their launches.
// sdpa_layout and steel_attention_layout check each launch against the same
// addressing; the host names each pipeline by its hash name, whose translated
// entry has the function constants that name implies frozen.
// crosstl_backend.cpp includes this file after the copy ports, so it shares
// that translation unit's copy.

#include <cstring>
#include <deque>

namespace mlx::core::fast {
namespace {

constexpr uint64_t sdpa_max_offset = (uint64_t(1) << 31) - 1;
// The host device has no neural accelerators.
constexpr bool sdpa_nax_available = false;

// get_type_string's names for the instantiated attention types.
const char* sdpa_type_string(Dtype dtype) {
  return dtype == float32 ? "float" : dtype == float16 ? "float16_t" :
      dtype == bfloat16 ? "bfloat16_t" : nullptr;
}

const char* sdpa_storage_type(Dtype dtype) {
  return dtype == float32 ? "float32" : dtype == float16 ? "float16" :
      dtype == bfloat16 ? "bfloat16" : dtype == bool_ ? "bool_" : nullptr;
}

bool sdpa_use_blocked_d512(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool do_causal,
    bool has_sinks,
    bool output_logsumexp,
    Stream s) {
  return s.device == Device::gpu && !sdpa_nax_available &&
      (q.dtype() == float16 || q.dtype() == bfloat16) && q.shape(-1) == 512 &&
      v.shape(-1) == 512 && q.shape(2) >= 1024 && q.shape(2) <= k.shape(2) &&
      do_causal && !has_arr_mask && !has_sinks && !output_logsumexp;
}

// scaled_dot_product_attention.cpp has_fused_kernel.
std::tuple<bool, std::string> sdpa_has_fused_kernel(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp,
    Stream s) {
  if (s.device != Device::gpu) {
    return {false, "the fused kernels require a GPU (Metal) stream."};
  }

  if (output_logsumexp && has_arr_mask) {
    return {
        false,
        "the backward pass does not support an array mask; only causal masking "
        "is implemented."};
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = k.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;

  std::ostringstream msg;
  if (query_sequence_length > 8) {
    const bool supports_d512 = sdpa_nax_available &&
        (env::enable_tf32() || q.dtype() != float32);
    const bool asymmetric = query_head_dim == 96 && value_head_dim == 64;
    const bool supported_head_dim = asymmetric ||
        (query_head_dim == value_head_dim &&
         (query_head_dim == 64 || query_head_dim == 72 ||
          query_head_dim == 80 || query_head_dim == 96 ||
          query_head_dim == 128 || query_head_dim == 192 ||
          query_head_dim == 256 || (query_head_dim == 512 && supports_d512)));
    if (!supported_head_dim) {
      msg << "the full attention kernel supports head dims "
          << "{64, 72, 80, 96, 128, 192, 256} with matching query/value dims, "
          << "or (query, value) head dims (96, 64), "
          << "plus head dim 512 on NAX GPUs (float32 also requires TF32); got "
          << "query head dim " << query_head_dim << " and value head dim "
          << value_head_dim << ".";
      return {false, msg.str()};
    }
    if (has_mask && !has_arr_mask &&
        !(query_sequence_length <= key_sequence_length && do_causal)) {
      msg << "the full attention kernel with a causal mask requires the "
          << "query sequence to be no longer than the key sequence; got "
          << "query length " << query_sequence_length << " and key length "
          << key_sequence_length << ".";
      return {false, msg.str()};
    }
  } else {
    const bool supported_head_dim =
        (query_head_dim == value_head_dim &&
         (query_head_dim == 64 || query_head_dim == 96 ||
          query_head_dim == 128 || query_head_dim == 192 ||
          query_head_dim == 256 || query_head_dim == 512)) ||
        (query_head_dim == 192 && value_head_dim == 128) ||
        (query_head_dim == 96 && value_head_dim == 64);
    if (!supported_head_dim) {
      msg << "the vector attention kernel supports head dims "
          << "{64, 96, 128, 192, 256, 512} with matching query/value head "
          << "dims, or (query, value) head dims (96, 64) or (192, 128); got "
          << "query head dim " << query_head_dim << " and value head dim "
          << value_head_dim << ".";
      return {false, msg.str()};
    }
    if (query_sequence_length > key_sequence_length) {
      msg << "the vector attention kernel requires the query sequence to be "
          << "no longer than the key sequence; got query length "
          << query_sequence_length << " and key length " << key_sequence_length
          << ".";
      return {false, msg.str()};
    }
    if (query_sequence_length * gqa_factor > 32) {
      msg << "the vector attention kernel requires the query length times "
          << "the GQA factor to be at most 32; got query length "
          << query_sequence_length << " and GQA factor " << gqa_factor << ".";
      return {false, msg.str()};
    }
    // Head dim 512 uses the generic vector kernel, reading K/V is its main
    // cost. By default, use it only for one query, GQA factor 8, and no array
    // mask.
    if (query_head_dim == 512) {
      int min_key_sequence_length = env::get_var("MLX_SDPA_D512_MIN_KL", 1024);
      bool always = (min_key_sequence_length == 0);
      if (!always && query_sequence_length != 1) {
        msg << "the vector attention kernel for head dim 512 defaults to "
               "single-token queries; got query length "
            << query_sequence_length << ".";
        return {false, msg.str()};
      }
      if (!always && gqa_factor != 8) {
        msg << "the vector attention kernel for head dim 512 defaults to "
               "GQA factor 8; got GQA factor "
            << gqa_factor << ".";
        return {false, msg.str()};
      }
      if (!always && has_arr_mask) {
        msg << "the vector attention kernel for head dim 512 defaults to "
               "calls without array masks.";
        return {false, msg.str()};
      }
      if (key_sequence_length < min_key_sequence_length) {
        msg << "the vector attention kernel for head dim 512 requires at "
            << "least " << min_key_sequence_length << " keys; got key length "
            << key_sequence_length << ".";
        return {false, msg.str()};
      }
    }
  }
  return {true, ""};
}

// An operand's data must hold the elements the kernels address, inside its
// buffer and below a signed 32-bit element offset.
void sdpa_require_span(const array& x, uint64_t span) {
  if (span == 0 || span > sdpa_max_offset || x.offset() < 0 ||
      uint64_t(x.offset()) > x.buffer_size() ||
      span > (x.buffer_size() - uint64_t(x.offset())) / x.itemsize()) {
    throw std::invalid_argument("CrossTL attention operand exceeds its bounds.");
  }
}

bool sdpa_nonnegative(const array& x) {
  return std::all_of(x.strides().begin(), x.strides().end(), [](int64_t stride) {
    return stride >= 0;
  });
}

int32_t sdpa_int32(int64_t value) {
  if (value < 0 || value > int64_t(sdpa_max_offset)) {
    throw std::invalid_argument("CrossTL attention stride exceeds its bounds.");
  }
  return int32_t(value);
}

void sdpa_launch(
    const std::string& entry,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  require_entry(entry);
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads,
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native attention failed: ") + error);
  }
}

uint32_t sdpa_groups(int64_t value) {
  if (value < 1 || value > 65535) {
    throw std::invalid_argument("CrossTL attention grid exceeds the portable group limit.");
  }
  return uint32_t(value);
}

// The scalars and spans every vector kernel binds for q, k, v, the mask and
// the sinks, with scaled_dot_product_attention.cpp's strides.
struct SdpaOperands {
  int32_t N = 0;
  uint64_t k_head_stride = 0;
  uint64_t k_seq_stride = 0;
  uint64_t v_head_stride = 0;
  uint64_t v_seq_stride = 0;
  float scale = 0;
  int32_t mask_kv_seq_stride = 0;
  int32_t mask_q_seq_stride = 0;
  int32_t mask_head_stride = 0;
  int32_t num_q_heads = 0;
  uint64_t q_span = 0;
  uint64_t k_span = 0;
  uint64_t v_span = 0;
  uint64_t mask_span = 0;
};

SdpaOperands sdpa_operands(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  SdpaOperands ops;
  ops.N = k.shape(2);
  ops.k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  ops.k_seq_stride = k.strides()[2];
  ops.v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  ops.v_seq_stride = v.strides()[2];
  ops.scale = scale;
  const uint64_t heads = uint64_t(q.shape(0)) * q.shape(1);
  const uint64_t kv_heads = uint64_t(k.shape(0)) * k.shape(1);
  const uint64_t keys = uint64_t(k.shape(2));
  const uint64_t length = uint64_t(q.shape(2));
  ops.q_span = heads * length * q.shape(3);
  ops.k_span = (kv_heads - 1) * ops.k_head_stride + (keys - 1) * ops.k_seq_stride +
      k.shape(3);
  ops.v_span = (kv_heads - 1) * ops.v_head_stride + (keys - 1) * ops.v_seq_stride +
      v.shape(3);
  sdpa_require_span(q, ops.q_span);
  sdpa_require_span(k, ops.k_span);
  sdpa_require_span(v, ops.v_span);
  if (mask) {
    auto& m = *mask;
    ops.mask_kv_seq_stride = sdpa_int32(m.shape(3) > 1 ? m.strides(3) : 0);
    ops.mask_q_seq_stride = sdpa_int32(m.shape(2) > 1 ? m.strides(2) : 0);
    ops.mask_head_stride = sdpa_int32(
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0));
    ops.mask_span = (heads - 1) * uint64_t(ops.mask_head_stride) +
        (keys - 1) * uint64_t(ops.mask_kv_seq_stride) +
        (length - 1) * uint64_t(ops.mask_q_seq_stride) + 1;
    sdpa_require_span(m, ops.mask_span);
  }
  if (sinks) {
    ops.num_q_heads = q.shape(1);
    sdpa_require_span(*sinks, uint64_t(ops.num_q_heads));
  }
  return ops;
}

void sdpa_bind_operands(
    std::vector<CrosstlMlxBuffer>& buffers,
    const array& q,
    const array& k,
    const array& v,
    SdpaOperands& ops) {
  const char* dtype = sdpa_storage_type(q.dtype());
  buffers.push_back({"queries", dtype, const_cast<void*>(q.data<void>()), ops.q_span, 0});
  buffers.push_back({"keys", dtype, const_cast<void*>(k.data<void>()), ops.k_span, 0});
  buffers.push_back({"values", dtype, const_cast<void*>(v.data<void>()), ops.v_span, 0});
}

void sdpa_bind_strides(std::vector<CrosstlMlxBuffer>& buffers, SdpaOperands& ops) {
  buffers.push_back({"N", "int32", &ops.N, 1, 0});
  buffers.push_back({"k_head_stride", "uint64", &ops.k_head_stride, 1, 0});
  buffers.push_back({"k_seq_stride", "uint64", &ops.k_seq_stride, 1, 0});
  buffers.push_back({"v_head_stride", "uint64", &ops.v_head_stride, 1, 0});
  buffers.push_back({"v_seq_stride", "uint64", &ops.v_seq_stride, 1, 0});
  buffers.push_back({"scale", "float32", &ops.scale, 1, 0});
}

void sdpa_bind_mask(
    std::vector<CrosstlMlxBuffer>& buffers,
    const std::optional<array>& mask,
    SdpaOperands& ops) {
  if (!mask) {
    return;
  }
  auto& m = *mask;
  const bool bool_mask = m.dtype() == bool_;
  buffers.push_back(
      {bool_mask ? "bmask" : "fmask", sdpa_storage_type(m.dtype()),
       const_cast<void*>(m.data<void>()), ops.mask_span, 0});
  buffers.push_back({"mask_kv_seq_stride", "int32", &ops.mask_kv_seq_stride, 1, 0});
  buffers.push_back({"mask_q_seq_stride", "int32", &ops.mask_q_seq_stride, 1, 0});
  buffers.push_back({"mask_head_stride", "int32", &ops.mask_head_stride, 1, 0});
}

std::string sdpa_constants_name(
    const std::optional<array>& mask,
    bool query_transposed,
    bool do_causal) {
  const bool has_mask = mask.has_value();
  const bool bool_mask = has_mask && (*mask).dtype() == bool_;
  std::string name =
      has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  name += query_transposed ? "_qt" : "_qnt";
  name += do_causal ? "_c" : "_nc";
  return name;
}

// scaled_dot_product_attention.cpp sdpa_vector.
void sdpa_vector_host(
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  std::string kname = "sdpa_vector_";
  kname += sdpa_type_string(q.dtype());
  kname += "_" + std::to_string(q.shape(-1)) + "_" + std::to_string(v.shape(-1));

  int32_t gqa_factor = q.shape(1) / k.shape(1);
  SdpaOperands ops = sdpa_operands(q, k, v, scale, mask, sinks);
  const bool query_transposed = !q.flags().row_contiguous;
  std::string hash_name = kname + sdpa_constants_name(mask, query_transposed, do_causal);
  hash_name += sinks ? "_sinks" : "_nosinks";

  // The kernel's buffer order: queries, keys, values, out, gqa_factor, N,
  // the strides and the scale, then the mask and the sinks.
  std::vector<CrosstlMlxBuffer> buffers;
  sdpa_bind_operands(buffers, q, k, v, ops);
  buffers.push_back(
      {"out", sdpa_storage_type(out.dtype()), out.data<void>(), out.size(), 1});
  buffers.push_back({"gqa_factor", "int32", &gqa_factor, 1, 0});
  sdpa_bind_strides(buffers, ops);
  sdpa_bind_mask(buffers, mask, ops);
  if (sinks) {
    buffers.push_back(
        {"sinks", sdpa_storage_type(sinks->dtype()),
         const_cast<void*>(sinks->data<void>()), uint64_t(ops.num_q_heads), 0});
    buffers.push_back({"num_q_heads", "int32", &ops.num_q_heads, 1, 0});
  }
  const CrosstlMlxLaunch launch{
      {sdpa_groups(int64_t(q.shape(0)) * q.shape(1)), sdpa_groups(q.shape(2)), 1},
      {1024, 1, 1}};
  sdpa_launch(hash_name, buffers, out.size(), launch);
}

// scaled_dot_product_attention.cpp sdpa_vector_2pass.
void sdpa_vector_2pass_host(
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks) {
  std::string kname = "sdpa_vector_2pass_1";
  int gqa_factor = q.shape(1) / k.shape(1);
  bool gqa_dims =
      (gqa_factor == 8 && (q.shape(-1) == 64 || q.shape(-1) == 128)) ||
      ((gqa_factor == 12 || gqa_factor == 16) && q.shape(-1) == 128);
  if (!mask && !sinks && q.shape(2) == 1 &&
      q.shape(1) == gqa_factor * k.shape(1) && q.shape(-1) == v.shape(-1) &&
      gqa_dims && k.shape(2) >= 8192) {
    kname += "_gqa_";
    kname += std::to_string(gqa_factor);
  }
  kname += "_";
  kname += sdpa_type_string(q.dtype());
  kname += "_" + std::to_string(q.shape(-1)) + "_" + std::to_string(v.shape(-1));

  // The base device's block counts: the host reports no 's' or 'd' class.
  int n_simds = gqa_factor * q.shape(2);
  int blocks = n_simds >= 4 ? 64 : 32;
  if (int blocks_env = env::get_var("MLX_SDPA_BLOCKS", 0); blocks_env > 0) {
    // The 2-pass reduction consumes the partials in simd-width (32) chunks
    // and silently drops the tail otherwise, so round up to a multiple of 32.
    blocks = ((blocks_env + 31) / 32) * 32;
  }

  SdpaOperands ops = sdpa_operands(q, k, v, scale, mask, sinks);
  const uint64_t partials = uint64_t(q.shape(0)) * q.shape(1) * q.shape(2) * blocks;
  Shape intermediate_shape(out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, q.dtype(), nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  if (sums.size() != partials || intermediate.size() != partials * v.shape(-1) ||
      intermediate.size() > sdpa_max_offset) {
    throw std::invalid_argument("CrossTL attention partials exceed their bounds.");
  }

  const bool query_transposed = !q.flags().row_contiguous;
  std::string hash_name = kname + sdpa_constants_name(mask, query_transposed, do_causal);
  hash_name += sinks ? "_sinks_" : "_nosinks_";
  hash_name += std::to_string(blocks);
  // The host entry also names the threadgroup, which the GQA factor and the
  // query length set.
  hash_name += "@32x" + std::to_string(gqa_factor) + "x" + std::to_string(q.shape(2));

  const char* dtype = sdpa_storage_type(q.dtype());
  std::vector<CrosstlMlxBuffer> buffers;
  sdpa_bind_operands(buffers, q, k, v, ops);
  buffers.push_back({"out", dtype, intermediate.data<void>(), intermediate.size(), 1});
  buffers.push_back({"sums", "float32", sums.data<void>(), sums.size(), 1});
  buffers.push_back({"maxs", "float32", maxs.data<void>(), maxs.size(), 1});
  sdpa_bind_strides(buffers, ops);
  sdpa_bind_mask(buffers, mask, ops);
  if (sinks) {
    buffers.push_back(
        {"sinks", sdpa_storage_type(sinks->dtype()),
         const_cast<void*>(sinks->data<void>()), uint64_t(ops.num_q_heads), 0});
  }
  const CrosstlMlxLaunch first{
      {sdpa_groups(k.shape(1)), sdpa_groups(q.shape(0)), sdpa_groups(blocks)},
      {32, uint32_t(gqa_factor), uint32_t(q.shape(2))}};
  sdpa_launch(hash_name, buffers, intermediate.size(), first);

  // Final pass.
  std::string second = "sdpa_vector_2pass_2_";
  second += sdpa_type_string(q.dtype());
  second += "_" + std::to_string(v.shape(-1));
  int32_t blocks_value = blocks;
  std::vector<CrosstlMlxBuffer> combine = {
      {"partials", dtype, intermediate.data<void>(), intermediate.size(), 0},
      {"sums", "float32", sums.data<void>(), sums.size(), 0},
      {"maxs", "float32", maxs.data<void>(), maxs.size(), 0},
      {"out", sdpa_storage_type(out.dtype()), out.data<void>(), out.size(), 1},
      {"blocks", "int32", &blocks_value, 1, 0},
  };
  const CrosstlMlxLaunch final_launch{
      {sdpa_groups(int64_t(q.shape(0)) * q.shape(1)), sdpa_groups(q.shape(2)), 1},
      {1024, 1, 1}};
  sdpa_launch(second, combine, out.size(), final_launch);
}

uint64_t sdpa_steel_span(const array& x, int64_t batch, int64_t heads, int64_t rows, int64_t row) {
  return uint64_t(batch - 1) * x.strides(0) + uint64_t(heads - 1) * x.strides(1) +
      uint64_t(rows - 1) * x.strides(2) + uint64_t(row);
}

// scaled_dot_product_attention.cpp sdpa_full_self_attention_metal. The host
// has no NAX, so neither the NAX kernels nor the padded head dims apply, and
// it has no 'd' architecture, so head dim 256 takes 16-row key blocks.
void sdpa_full_attention_host(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask,
    const std::optional<array>& sinks,
    array* lse) {
  const int64_t B = q.shape(0);
  const int64_t H = q.shape(1);
  const int64_t D = q.shape(3);
  const int64_t qL = q.shape(2);
  const int64_t kL = k.shape(2);
  const int64_t gqa_factor = q.shape(1) / k.shape(1);
  const int bd = q.shape(-1);
  const int bv = v.shape(-1);
  const int bq = 32;
  const int bk = bd < 128 ? 32 : 16;
  const int wm = 4;
  const int wn = bd == 256 ? 2 : 1;

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = mask.has_value();
  const bool has_sinks = sinks.has_value();
  const bool save_lse = lse != nullptr;
  if (has_mask && mask->dtype() != bool_ && mask->dtype() != q.dtype()) {
    throw std::invalid_argument("CrossTL attention mask type has no kernel.");
  }
  auto flag = [](bool value) { return value ? "t" : "n"; };
  std::string hash_name = std::string("steel_attention_") +
      sdpa_storage_type(q.dtype()) + "_bq" + std::to_string(bq) + "_bk" +
      std::to_string(bk) + "_bd" + std::to_string(bd) + "_bv" + std::to_string(bv) +
      "_wm" + std::to_string(wm) + "_wn" + std::to_string(wn) + "_mask" +
      sdpa_storage_type(has_mask ? mask->dtype() : q.dtype()) + "_align_Q_" +
      flag(align_Q) + "_align_K_" + flag(align_K) + "_has_mask_" + flag(has_mask) +
      "_do_causal_" + flag(do_causal) + "_has_sinks_" + flag(has_sinks) +
      "_save_lse_" + flag(save_lse);

  const int64_t NQ = (qL + bq - 1) / bq;
  const int64_t NK = (kL + bk - 1) / bk;
  const int64_t NQ_aligned = qL / bq;
  const int64_t NK_aligned = kL / bk;
  uint32_t scale_bits = 0;
  std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
  // AttnParams' fields in params.h order, the scale as its binary32 bits.
  std::vector<int64_t> params = {
      B, H, D, qL, kL, gqa_factor, int64_t(scale_bits), NQ, NK, NQ_aligned,
      NK_aligned, qL - NQ_aligned * bq, kL - NK_aligned * bk, kL - qL};
  for (const array* x : {&q, &k, &v, static_cast<const array*>(&o)}) {
    for (int axis = 0; axis < 3; ++axis) {
      params.push_back(x->strides(axis));
    }
  }

  const uint64_t q_span = sdpa_steel_span(q, B, H, qL, D);
  const uint64_t k_span = sdpa_steel_span(k, B, k.shape(1), kL, D);
  const uint64_t v_span = sdpa_steel_span(v, B, v.shape(1), kL, bv);
  sdpa_require_span(q, q_span);
  sdpa_require_span(k, k_span);
  sdpa_require_span(v, v_span);
  const char* dtype = sdpa_storage_type(q.dtype());
  std::vector<CrosstlMlxBuffer> buffers;
  buffers.push_back({"Q", dtype, const_cast<void*>(q.data<void>()), q_span, 0});
  buffers.push_back({"K", dtype, const_cast<void*>(k.data<void>()), k_span, 0});
  buffers.push_back({"V", dtype, const_cast<void*>(v.data<void>()), v_span, 0});
  buffers.push_back({"O", dtype, o.data<void>(), o.size(), 1});
  buffers.push_back({"params", "int64", params.data(), params.size(), 0});
  std::vector<int64_t> mask_params;
  if (has_mask) {
    auto& m = *mask;
    mask_params = {m.strides(0), m.strides(1), m.strides(2)};
    const uint64_t mask_span = sdpa_steel_span(m, B, H, qL, kL);
    sdpa_require_span(m, mask_span);
    buffers.push_back(
        {"mask_params", "int64", mask_params.data(), mask_params.size(), 0});
    buffers.push_back(
        {"mask", sdpa_storage_type(m.dtype()), const_cast<void*>(m.data<void>()),
         mask_span, 0});
  }
  if (has_sinks) {
    sdpa_require_span(*sinks, uint64_t(H));
    buffers.push_back(
        {"sinks", sdpa_storage_type(sinks->dtype()),
         const_cast<void*>(sinks->data<void>()), uint64_t(H), 0});
  }
  if (save_lse) {
    buffers.push_back({"lse", "float32", lse->data<void>(), lse->size(), 1});
  }
  const CrosstlMlxLaunch launch{
      {sdpa_groups(NQ), sdpa_groups(H), sdpa_groups(B)},
      {32, uint32_t(wm), uint32_t(wn)}};
  sdpa_launch(hash_name, buffers, o.size(), launch);
}

// scaled_dot_product_attention.cpp ensure_row_contiguous, vjp_alloc,
// vjp_view, vjp_row_slice and sdpa_vjp_blocks. The host keeps each
// temporary alive in its scope instead of handing it to a command encoder.
array sdpa_vjp_row_contiguous(const array& x) {
  return x.flags().row_contiguous ? x : contiguous_copy(x);
}

array sdpa_vjp_alloc(Shape shape, Dtype dt) {
  array a(std::move(shape), dt, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

array sdpa_vjp_view(const array& base, Shape shape) {
  array v(shape, base.dtype(), nullptr, {});
  Strides st(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    st[i] = acc;
    acc *= shape[i];
  }
  array::Flags f{1, 1, shape.size() <= 1};
  v.copy_shared_buffer(base, st, f, static_cast<size_t>(acc), 0);
  return v;
}

array sdpa_vjp_row_slice(const array& x, int t0, int len) {
  const auto& shp = x.shape();
  int64_t Dx = shp[3];
  int64_t Tx = shp[2];
  Shape ns = {shp[0], shp[1], len, static_cast<int>(Dx)};
  Strides st = {shp[1] * Tx * Dx, Tx * Dx, Dx, 1};
  array v(ns, x.dtype(), nullptr, {});
  array::Flags f{1, len == shp[2], false};
  size_t base = static_cast<size_t>(x.offset()) / x.itemsize();
  v.copy_shared_buffer(
      x,
      st,
      f,
      x.data_size(),
      base + static_cast<size_t>(t0) * static_cast<size_t>(Dx));
  return v;
}

std::pair<int, int> sdpa_vjp_blocks(int B, int H, int qL, int kL, Dtype ctype) {
  constexpr int kDefaultBlock = 1024;
  int blk = env::get_var("MLX_SDPA_VJP_BLOCK", kDefaultBlock);
  int bq = std::min(blk, qL);
  int bk = std::min(blk, kL);
  if (int e = env::get_var("MLX_SDPA_VJP_BQ", 0); e > 0) {
    bq = std::min(e, qL);
  }
  if (int e = env::get_var("MLX_SDPA_VJP_BK", 0); e > 0) {
    bk = std::min(e, kL);
  }
  return {bq, bk};
}

// The threadgroups that cover dispatch_threads(grid, group): every VJP
// kernel returns for the threads outside its grid.
CrosstlMlxLaunch sdpa_vjp_launch(
    int64_t x,
    int64_t y,
    int64_t z,
    uint32_t group_x,
    uint32_t group_y) {
  return {
      {sdpa_groups((x + group_x - 1) / group_x),
       sdpa_groups((y + group_y - 1) / group_y),
       sdpa_groups(z)},
      {group_x, group_y, 1}};
}

// scaled_dot_product_attention.cpp sdpa_vjp_blocked. The score gradients
// take a buffer of their own rather than overwriting the scores, since the
// host refuses outputs that overlap inputs; each element of dS depends only
// on the same element of S, so every product is unchanged.
void sdpa_vjp_blocked_host(
    const array& q,
    const array& k,
    const array& v,
    const array& odo,
    const array& lse,
    const array& cot_o,
    array& dq,
    array& dk,
    array& dv,
    float scale,
    bool causal) {
  const int B = q.shape(0);
  const int H = q.shape(1);
  const int qL = q.shape(2);
  const int D = q.shape(3);
  const int Hk = k.shape(1);
  const int kL = k.shape(2);
  const int Dv = v.shape(3);
  const int G = H / Hk;
  const int BH = B * H;
  const int BHk = B * Hk;
  const int diag_off = kL - qL;

  auto [BQ, BK] = sdpa_vjp_blocks(B, H, qL, kL, q.dtype());
  Dtype ctype = q.dtype();
  const std::string tname = sdpa_type_string(ctype);
  const char* storage = sdpa_storage_type(ctype);
  const std::string ds_kernel = "sdpa_vjp_ds_" + tname;
  const std::string red_add = "sdpa_vjp_reduce_add_" + tname;
  const std::string red_set = "sdpa_vjp_reduce_set_" + tname;

  array s_buf = sdpa_vjp_alloc({BH, BQ, BK}, ctype);
  array ds_buf = sdpa_vjp_alloc({BH, BQ, BK}, ctype);
  array dp_buf = sdpa_vjp_alloc({BH, BQ, BK}, ctype);
  array p_buf = sdpa_vjp_alloc({BH, BQ, BK}, ctype);
  array dq_tile = sdpa_vjp_alloc({BH, BQ, D}, ctype);
  array dk_tile = sdpa_vjp_alloc({BH, BK, D}, ctype);
  array dv_tile = sdpa_vjp_alloc({BH, BK, Dv}, ctype);
  array dq_acc = sdpa_vjp_alloc({B, H, qL, D}, float32);
  array dk_acc = sdpa_vjp_alloc({B, Hk, kL, D}, float32);
  array dv_acc = sdpa_vjp_alloc({B, Hk, kL, Dv}, float32);

  auto data = [](const array& x) { return const_cast<void*>(x.data<void>()); };
  auto run_reduce = [&](const array& src,
                        array& acc,
                        array& out,
                        int rows,
                        int dim,
                        int group,
                        int acc_rows,
                        int row_off,
                        int nbh,
                        bool accum) {
    int32_t scalars[] = {rows, dim, group, acc_rows, row_off};
    std::vector<CrosstlMlxBuffer> buffers = {
        {"src", storage, data(src), src.size(), 0},
        {"acc", "float32", acc.data<void>(), acc.size(), CROSTL_MLX_BUFFER_INOUT},
        {"out", storage, out.data<void>(), out.size(), CROSTL_MLX_BUFFER_INOUT},
        {"rows", "int32", &scalars[0], 1, 0},
        {"dim", "int32", &scalars[1], 1, 0},
        {"group", "int32", &scalars[2], 1, 0},
        {"acc_rows", "int32", &scalars[3], 1, 0},
        {"row_off", "int32", &scalars[4], 1, 0}};
    sdpa_launch(
        accum ? red_add : red_set,
        buffers,
        out.size(),
        sdpa_vjp_launch(dim, rows, nbh, uint32_t(std::min(dim, 32)), 8));
  };

  const int n_i = (qL + BQ - 1) / BQ;
  const Strides q_bs = {H * int64_t(qL) * D, G * int64_t(qL) * D, int64_t(qL) * D};
  const Strides o_bs = {
      H * int64_t(qL) * Dv, G * int64_t(qL) * Dv, int64_t(qL) * Dv};
  const Strides k_bs = {Hk * int64_t(kL) * D, int64_t(kL) * D, 0};
  const Strides v_bs = {Hk * int64_t(kL) * Dv, int64_t(kL) * Dv, 0};
  const Shape bshape = {B, Hk, G};

  for (int j0 = 0; j0 < kL; j0 += BK) {
    const int bk_len = std::min(BK, kL - j0);
    array k_sl = sdpa_vjp_row_slice(k, j0, bk_len);
    array v_sl = sdpa_vjp_row_slice(v, j0, bk_len);
    bool kv_accum = false;

    for (int ii = 0; ii < n_i; ++ii) {
      const int i0 = ii * BQ;
      const int bq_len = std::min(BQ, qL - i0);
      if (causal && j0 > i0 + bq_len - 1 + diag_off) {
        continue;
      }
      array q_sl = sdpa_vjp_row_slice(q, i0, bq_len);
      array o_sl = sdpa_vjp_row_slice(cot_o, i0, bq_len);

      // S = Q @ K.T
      array s_v = sdpa_vjp_view(s_buf, {BH, bq_len, bk_len});
      dispatch_steel_gemm(
          q_sl, k_sl, s_v, bq_len, bk_len, D, BH, D, D, false, true, bshape, q_bs,
          k_bs);
      // dP = dO @ V.T
      array dp_v = sdpa_vjp_view(dp_buf, {BH, bq_len, bk_len});
      dispatch_steel_gemm(
          o_sl, v_sl, dp_v, bq_len, bk_len, Dv, BH, Dv, Dv, false, true, bshape,
          o_bs, v_bs);

      // P = exp(scale * S - lse), dS = P * (dP - delta) * scale
      array ds_v = sdpa_vjp_view(ds_buf, {BH, bq_len, bk_len});
      array p_v = sdpa_vjp_view(p_buf, {BH, bq_len, bk_len});
      uint32_t scale_bits = 0;
      std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
      // SDPAVJPTileParams' fields in order, the scale as its binary32 bits.
      std::vector<int64_t> params = {
          bq_len, bk_len, qL, i0, j0, int64_t(scale_bits), diag_off, causal ? 1 : 0};
      std::vector<CrosstlMlxBuffer> buffers = {
          {"S", storage, s_v.data<void>(), s_v.size(), 0},
          {"dP", storage, dp_v.data<void>(), dp_v.size(), 0},
          {"lse", "float32", data(lse), lse.size(), 0},
          {"odo", "float32", data(odo), odo.size(), 0},
          {"dS", storage, ds_v.data<void>(), ds_v.size(), 1},
          {"P", storage, p_v.data<void>(), p_v.size(), 1},
          {"p", "int64", params.data(), params.size(), 0}};
      sdpa_launch(
          ds_kernel, buffers, p_v.size(), sdpa_vjp_launch(bk_len, bq_len, BH, 32, 8));

      int64_t ss2 = static_cast<int64_t>(bq_len) * bk_len;
      Strides sc_bs = {H * ss2, G * ss2, ss2};
      // dQ = dS @ K
      array dq_v = sdpa_vjp_view(dq_tile, {BH, bq_len, D});
      dispatch_steel_gemm(
          ds_v, k_sl, dq_v, bq_len, D, bk_len, BH, bk_len, D, false, false, bshape,
          sc_bs, k_bs);
      // dK = dS.T @ Q
      array dk_v = sdpa_vjp_view(dk_tile, {BH, bk_len, D});
      dispatch_steel_gemm(
          ds_v, q_sl, dk_v, bk_len, D, bq_len, BH, bk_len, D, true, false, bshape,
          sc_bs, q_bs);
      // dV = P.T @ dO
      array dv_v = sdpa_vjp_view(dv_tile, {BH, bk_len, Dv});
      dispatch_steel_gemm(
          p_v, o_sl, dv_v, bk_len, Dv, bq_len, BH, bk_len, Dv, true, false, bshape,
          sc_bs, o_bs);

      // dQ[i0] += dQ_tile, dK[j0] += sum_G dK_tile, dV[j0] += sum_G dV_tile
      run_reduce(dq_v, dq_acc, dq, bq_len, D, 1, qL, i0, BH, j0 != 0);
      run_reduce(dk_v, dk_acc, dk, bk_len, D, G, kL, j0, BHk, kv_accum);
      run_reduce(dv_v, dv_acc, dv, bk_len, Dv, G, kL, j0, BHk, kv_accum);
      kv_accum = true;
    }
  }
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool has_sinks,
    bool is_training,
    bool output_logsumexp,
    bool force_fused,
    Stream s) {
  auto [has_fused, reason] = sdpa_has_fused_kernel(
      q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp, s);
  if (force_fused) {
    if (!has_fused) {
      std::ostringstream msg;
      msg << "[scaled_dot_product_attention] force_fused=True but no fused "
             "kernel is available: "
          << reason;
      throw std::invalid_argument(msg.str());
    }
    return false;
  }
  if (sdpa_use_blocked_d512(
          q, k, v, has_arr_mask, do_causal, has_sinks, output_logsumexp, s)) {
    return false;
  }
  if (!has_fused) {
    return true;
  }

  // The logsumexp only comes out of the fused forward
  if (output_logsumexp) {
    return false;
  }

  const int query_sequence_length = q.shape(2);
  const int query_head_dim = q.shape(-1);
  const int value_head_dim = v.shape(-1);

  if (query_head_dim == 512 && query_sequence_length > 8) {
    // Without NAX the D512 full-attention kernel is never eligible.
    return true;
  }

  if (!sdpa_nax_available && q.dtype() != float32 && do_causal &&
      query_head_dim == 256 && query_sequence_length >= 2048 &&
      query_sequence_length == k.shape(2)) {
    return false;
  }

  // Unfused path is faster for following shapes.
  if (query_sequence_length > 8) {
    return query_head_dim == 192 || query_head_dim == 256;
  } else {
    return query_head_dim == value_head_dim && query_head_dim == 192;
  }
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return true;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = outputs[0];
  if (!sdpa_type_string(q_pre.dtype()) || k_pre.dtype() != q_pre.dtype() ||
      v_pre.dtype() != q_pre.dtype() || o.dtype() != q_pre.dtype()) {
    throw std::invalid_argument(
        "CrossTL attention requires float32, float16 or bfloat16 operands.");
  }

  // A deque keeps every copy's address while later copies are added. The
  // host also copies views with negative strides, which it never binds.
  std::deque<array> copies;
  auto copy_unless = [&copies](auto predicate, const array& arr) -> const array& {
    if (!predicate(arr) || !sdpa_nonnegative(arr)) {
      copies.push_back(row_kernel_copy(arr));
      return copies.back();
    }
    return arr;
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (has_sinks_) {
    sinks = copy_unless(is_matrix_contiguous, inputs.back());
  }
  bool has_arr_mask = inputs.size() > (3 + has_sinks_);

  // Full attention mode
  if (q_pre.shape(2) > 8) {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto mask = has_arr_mask
        ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
        : std::nullopt;

    array* lse = nullptr;
    if (outputs.size() > 1) {
      outputs[1].set_data(allocator::malloc(outputs[1].nbytes()));
      lse = &outputs[1];
    }

    if (sdpa_use_blocked_d512(
            q,
            k,
            v,
            has_arr_mask,
            do_causal_,
            sinks.has_value(),
            lse != nullptr,
            stream())) {
      throw std::invalid_argument(
          "CrossTL does not run scaled_dot_product_attention.cpp's blocked "
          "head-dim-512 attention yet.");
    }
    sdpa_full_attention_host(q, k, v, scale_, o, do_causal_, mask, sinks, lse);
    return;
  }
  if (outputs.size() > 1) {
    throw std::runtime_error(
        "[scaled_dot_product_attention] the vector kernels do not produce a "
        "logsumexp; the VJP path requires the full-attention kernel.");
  }

  auto q_copy_unless = [](const array& arr) {
    if (arr.flags().row_contiguous) {
      return true;
    }
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (shape[0] == 1 || shape[1] == 1) {
      // If either the batch or head dimension is a singleton, the other can
      // be transposed with the sequence dimension
      auto bidx = shape[0] == 1 ? 1 : 0;
      return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
          (strides[bidx] == shape[3]);
    }
    return false;
  };

  auto kv_copy_unless = [](const array& arr) {
    // keys and values should be copied if:
    // - the last dimension is not contiguous
    // - the batch and head dim are not contiguous
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    if (strides.back() != 1) {
      return false;
    }
    if (shape[0] == 1 || shape[1] == 1) {
      return true;
    }
    return (strides[0] == strides[1] * shape[1]);
  };

  const array& q = copy_unless(q_copy_unless, q_pre);
  const auto& k = copy_unless(kv_copy_unless, k_pre);
  const auto& v = copy_unless(kv_copy_unless, v_pre);

  // scaled_dot_product_attention.cpp donates the query when it can. The host
  // dispatch refuses an output that overlaps an input, so the output always
  // gets its own buffer.
  o.set_data(allocator::malloc(o.nbytes()));

  auto mask_copy_unless = [&q](const array& arr) {
    auto& strides = arr.strides();
    auto& shape = arr.shape();
    return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
        (strides[0] == strides[1] * shape[1]);
  };

  auto mask = has_arr_mask
      ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
      : std::nullopt;

  // We route to the 2 pass fused attention if
  // - The device is large and the sequence length long
  // - The sequence length is even longer and we have gqa
  // The host device is neither large class, so only the second rule applies.
  bool do_causal = do_causal_ && q.shape(2) > 1;
  if (k.shape(1) < q.shape(1) && k.shape(2) >= 4096) {
    sdpa_vector_2pass_host(q, k, v, o, scale_, do_causal, mask, sinks);
  } else {
    sdpa_vector_host(q, k, v, o, scale_, do_causal, mask, sinks);
  }
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  if (s.device != Device::gpu) {
    return true;
  }
  if (env::get_var("MLX_SDPA_VJP_FALLBACK", 0) != 0) {
    return true;
  }
  auto dt = q.dtype();
  return !(dt == float32 || dt == float16 || dt == bfloat16);
}

// scaled_dot_product_attention.cpp ScaledDotProductAttentionVJP::eval_gpu.
void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (has_sinks_) {
    throw std::runtime_error(
        "[ScaledDotProductAttentionVJP] NYI: attention sinks");
  }

  auto q = sdpa_vjp_row_contiguous(inputs[0]);
  auto k = sdpa_vjp_row_contiguous(inputs[1]);
  auto v = sdpa_vjp_row_contiguous(inputs[2]);
  const int n_in = static_cast<int>(inputs.size());
  auto o = sdpa_vjp_row_contiguous(inputs[n_in - 3]);
  auto lse = sdpa_vjp_row_contiguous(inputs[n_in - 2]);
  auto cot_o = sdpa_vjp_row_contiguous(inputs[n_in - 1]);

  const int B = q.shape(0);
  const int H = q.shape(1);
  const int qL = q.shape(2);
  const int Dv = v.shape(3);

  auto& dq = outputs[0];
  auto& dk = outputs[1];
  auto& dv = outputs[2];
  dq.set_data(allocator::malloc(dq.nbytes()));
  dk.set_data(allocator::malloc(dk.nbytes()));
  dv.set_data(allocator::malloc(dv.nbytes()));

  array odo({B, H, qL}, float32, nullptr, {});
  odo.set_data(allocator::malloc(odo.nbytes()));

  const char* storage = sdpa_storage_type(q.dtype());
  const std::string odo_name = std::string("sdpa_vjp_odo_") +
      sdpa_type_string(q.dtype()) + "_" + std::to_string(Dv);
  int32_t query_length = qL;
  std::vector<CrosstlMlxBuffer> buffers = {
      {"o", storage, const_cast<void*>(o.data<void>()), o.size(), 0},
      {"cot_o", storage, const_cast<void*>(cot_o.data<void>()), cot_o.size(), 0},
      {"odo", "float32", odo.data<void>(), odo.size(), 1},
      {"qL", "int32", &query_length, 1, 0}};
  sdpa_launch(odo_name, buffers, odo.size(), sdpa_vjp_launch(32, qL, B * H, 32, 1));

  sdpa_vjp_blocked_host(q, k, v, odo, lse, cot_o, dq, dk, dv, scale_, do_causal_);
}

} // namespace mlx::core::fast
