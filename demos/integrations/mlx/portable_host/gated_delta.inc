// Ports mlx/backend/metal/gated_delta_update.cpp: GatedDeltaUpdate and
// GatedDeltaUpdateVJP use_fallback and eval_gpu. The translated targets have
// no NAX, so gated_delta_chunk_size() takes the sequential kernel up to eight
// steps and the eight-step chunk kernel beyond, and the backward runs the
// save pass and the sequential backward with checkpoint interval 1. The
// runtime translates each pipeline on first use (gated_delta_packages).
// crosstl_backend.cpp includes this file at its end, so the port shares that
// translation unit's dispatch helpers and copies.

namespace mlx::core::fast {
namespace {

constexpr uint64_t gated_delta_max_elements = (uint64_t(1) << 28) - 1;

// metal::is_nax_available(): no translated target has NAX.
constexpr bool gated_delta_nax = false;

int gated_delta_chunk_size(int T) {
  const int C =
      env::get_var("GATED_DELTA_CHUNK", T > 8 ? (gated_delta_nax ? 16 : 8) : 1);
  return (C == 16 && !gated_delta_nax) ? 8 : C;
}

int gated_delta_chunk_size_vjp() {
  return gated_delta_nax ? env::get_var("GATED_DELTA_CHUNK_VJP", 16) : 1;
}

int gated_delta_ckpt(int chunk) {
  int c = env::get_var("GATED_DELTA_CKPT", 16);
  if (c != 1 && c != 4 && c != 8 && c != 16) {
    c = 16;
  }
  return (chunk == 16) ? c : 1;
}

bool supported_gated_delta_shape(int Hk, int Dk, int Hv, int Dv) {
  if (Dk != 128 || Dv != 128) {
    return false;
  }
  return (Hk == 24 && Hv == 24) || (Hk == 32 && Hv == 32) ||
      (Hk == 16 && Hv == 32) || (Hk == 16 && Hv == 48) ||
      (Hk == 16 && Hv == 16) || (Hk == 16 && Hv == 64);
}

// get_type_string: the C++ spelling the pipeline names carry.
const char* gated_delta_type_string(Dtype dtype) {
  return dtype == float32 ? "float" : dtype == float16 ? "float16_t" :
      dtype == bfloat16 ? "bfloat16_t" : nullptr;
}

// ensure_row_contiguous; the copy stays alive as an operand of the launch.
array gated_delta_rows(const array& x) {
  return x.flags().row_contiguous ? x : contiguous_copy(x);
}

array gated_delta_alloc(Shape shape, Dtype dtype) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

// fill_gpu(array(0, dtype), out): a broadcast zero copied into every element.
void gated_delta_zero(array& out) {
  array zero(0, out.dtype());
  array fill(out.shape(), out.dtype(), nullptr, {});
  fill.copy_shared_buffer(zero, Strides(out.ndim(), 0), {true, false, false}, 1);
  dispatch_copy(fill, out);
}

// An operand's data must lie inside its buffer before its words are bound.
CrosstlMlxBuffer gated_delta_buffer(
    const char* name, const array& value, uint32_t direction) {
  if (value.offset() < 0 || uint64_t(value.offset()) > value.buffer_size() ||
      value.data_size() == 0 || value.data_size() != value.size() ||
      value.data_size() >
          (value.buffer_size() - uint64_t(value.offset())) / value.itemsize() ||
      value.data_size() > gated_delta_max_elements) {
    throw std::invalid_argument("CrossTL gated-delta operand exceeds its bounds.");
  }
  const char* dtype = value.dtype() == int32 ? "int32" : row_kernel_type(value.dtype());
  if (!dtype) {
    throw std::invalid_argument("CrossTL gated-delta operand type is unsupported.");
  }
  return {name, dtype, const_cast<void*>(value.data<void>()), value.data_size(),
          direction};
}

// dispatch_threads(grid, (32, 4, 1)): every grid gated_delta_update.cpp
// launches is a whole number of these threadgroups.
void gated_delta_launch(
    const std::string& entry, std::vector<CrosstlMlxBuffer>& buffers,
    int columns, int rows) {
  if (rows < 1 || rows > 65535 || columns < 4 || columns % 4 != 0) {
    throw std::invalid_argument(
        "CrossTL gated-delta grid exceeds the portable group limit.");
  }
  require_entry(entry);
  const CrosstlMlxLaunch launch{
      {1, uint32_t(columns / 4), uint32_t(rows)}, {32, 4, 1}, {0, 0, 0}};
  const uint64_t threads = uint64_t(32) * uint64_t(columns) * uint64_t(rows);
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()), threads,
      &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native gated delta failed: ") + error);
  }
}

std::string gated_delta_suffix(const array& q, int Dk, int Dv, int Hk, int Hv) {
  const char* type = gated_delta_type_string(q.dtype());
  if (!type) {
    throw std::invalid_argument(
        "CrossTL gated delta supports float32, float16 and bfloat16 inputs.");
  }
  return std::string(type) + "_" + std::to_string(Dk) + "_" + std::to_string(Dv) +
      "_" + std::to_string(Hk) + "_" + std::to_string(Hv);
}

// The forward's buffers in kernel order: q, k, v, state_in, g, beta, y,
// state_out, T and, for the save pass, state_cache.
std::vector<CrosstlMlxBuffer> gated_delta_forward_buffers(
    const array& q, const array& k, const array& v, const array& h0,
    const array& g, const array& beta, const array& y, uint32_t y_direction,
    const array& state, int32_t* steps) {
  return {
      gated_delta_buffer("q", q, 0),
      gated_delta_buffer("k", k, 0),
      gated_delta_buffer("v", v, 0),
      gated_delta_buffer("state_in", h0, 0),
      gated_delta_buffer("g", g, 0),
      gated_delta_buffer("beta", beta, 0),
      gated_delta_buffer("y", y, y_direction),
      gated_delta_buffer("state_out", state, 1),
      {"T", "int32", steps, 1, 0},
  };
}

} // namespace

bool GatedDeltaUpdate::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    const bool has_mask,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (has_mask) {
    return true;
  }
  return !supported_gated_delta_shape(Hk, Dk, Hv, Dv);
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 6 || outputs.size() != 2) {
    throw std::invalid_argument("CrossTL gated delta requires six inputs.");
  }
  auto q = gated_delta_rows(inputs[0]);
  auto k = gated_delta_rows(inputs[1]);
  auto v = gated_delta_rows(inputs[2]);
  auto g = gated_delta_rows(inputs[3]);
  auto beta = gated_delta_rows(inputs[4]);
  auto h0 = gated_delta_rows(inputs[5]);

  auto& out = outputs[0];
  auto& hf = outputs[1];

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);
  if (T < 1) {
    throw std::invalid_argument("CrossTL gated delta requires at least one step.");
  }

  int C = gated_delta_chunk_size(T);
  const int ckpt = gated_delta_ckpt(C);
  const std::string suffix = gated_delta_suffix(q, Dk, Dv, Hk, Hv);

  // The inference pipeline clears save_state, so state_cache stays unbound.
  gated_delta_zero(out);
  hf.set_data(allocator::malloc(hf.nbytes()));
  int32_t steps = T;
  auto buffers = gated_delta_forward_buffers(
      q, k, v, h0, g, beta, out, CROSTL_MLX_BUFFER_INOUT, hf, &steps);

  switch (C) {
    case 8:
      gated_delta_launch(
          "gated_delta_fused_chunk_" + suffix + "_" + std::to_string(C), buffers,
          Dv / 8, B * Hv);
      break;
    case 1:
    case 0:
      gated_delta_launch(
          "seq_gated_delta_" + suffix + "_" + std::to_string(ckpt), buffers, Dv,
          B * Hv);
      break;
    default:
      throw std::runtime_error(
          "NYI: Only sequential and chunk size 8,16 are supported");
  }
}

bool GatedDeltaUpdateVJP::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }
  if (env::get_var("GATED_DELTA_VJP_FALLBACK", 0) != 0) {
    return true;
  }
  return !supported_gated_delta_shape(Hk, Dk, Hv, Dv);
}

void GatedDeltaUpdateVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  // Inputs are q, k, v, g, b, h0, cot_o, cot_h; outputs dq, dk, dv, dg, db, dh.
  if (inputs.size() != 8 || outputs.size() != 6) {
    throw std::invalid_argument("CrossTL gated-delta gradients require eight inputs.");
  }
  auto q = gated_delta_rows(inputs[0]);
  auto k = gated_delta_rows(inputs[1]);
  auto v = gated_delta_rows(inputs[2]);
  auto g = gated_delta_rows(inputs[3]);
  auto beta = gated_delta_rows(inputs[4]);
  auto h0 = gated_delta_rows(inputs[5]);
  auto cot_o = gated_delta_rows(inputs[6]);
  auto cot_h = gated_delta_rows(inputs[7]);

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);
  if (T < 1) {
    throw std::invalid_argument("CrossTL gated delta requires at least one step.");
  }

  // 16 = chunked NAX, anything else = sequential.
  const int C = gated_delta_chunk_size_vjp();
  if (C == 16) {
    throw std::runtime_error("CrossTL gated delta has no NAX backward.");
  }
  const int ckpt = gated_delta_ckpt(C);
  const int n_states = T;

  auto& dq = outputs[0];
  auto& dk = outputs[1];
  auto& dv = outputs[2];
  auto& dg = outputs[3];
  auto& db = outputs[4];
  auto& dh = outputs[5];

  // The kernels accumulate their gradients as fp32, so we may need a cast.
  const bool stage_fp32 = (q.dtype() != float32);
  auto accumulator = [&](array& gradient) {
    return stage_fp32 ? array(gradient.shape(), float32, nullptr, {}) : gradient;
  };
  array dq_acc = accumulator(dq);
  array dk_acc = accumulator(dk);
  array dv_acc = accumulator(dv);
  array dg_acc = accumulator(dg);
  array db_acc = accumulator(db);
  gated_delta_zero(dq_acc);
  gated_delta_zero(dk_acc);
  gated_delta_zero(dv_acc);
  gated_delta_zero(dg_acc);
  gated_delta_zero(db_acc);
  gated_delta_zero(dh);

  array state_cache = gated_delta_alloc({B, Hv, n_states, Dv, Dk}, float32);
  array y_scratch = gated_delta_alloc({B, T, Hv, Dv}, q.dtype());
  array hf_scratch = gated_delta_alloc({B, Hv, Dv, Dk}, float32);

  const std::string suffix = gated_delta_suffix(q, Dk, Dv, Hk, Hv);
  const std::string ckpt_suffix = "_" + std::to_string(ckpt);
  int32_t steps = T;
  {
    auto buffers = gated_delta_forward_buffers(
        q, k, v, h0, g, beta, y_scratch, 1, hf_scratch, &steps);
    buffers.push_back(gated_delta_buffer("state_cache", state_cache, 1));
    gated_delta_launch(
        "seq_gated_delta_" + suffix + ckpt_suffix + "_save", buffers, Dv, B * Hv);
  }
  {
    std::vector<CrosstlMlxBuffer> buffers = {
        gated_delta_buffer("q", q, 0),
        gated_delta_buffer("k", k, 0),
        gated_delta_buffer("v", v, 0),
        gated_delta_buffer("g", g, 0),
        gated_delta_buffer("b", beta, 0),
        gated_delta_buffer("cot_o", cot_o, 0),
        gated_delta_buffer("cot_h", cot_h, 0),
        gated_delta_buffer("state_cache", state_cache, 0),
        {"T", "int32", &steps, 1, 0},
        gated_delta_buffer("dq", dq_acc, CROSTL_MLX_BUFFER_INOUT),
        gated_delta_buffer("dk", dk_acc, CROSTL_MLX_BUFFER_INOUT),
        gated_delta_buffer("dv", dv_acc, CROSTL_MLX_BUFFER_INOUT),
        gated_delta_buffer("dg", dg_acc, CROSTL_MLX_BUFFER_INOUT),
        gated_delta_buffer("db", db_acc, CROSTL_MLX_BUFFER_INOUT),
        gated_delta_buffer("dh", dh, CROSTL_MLX_BUFFER_INOUT),
    };
    gated_delta_launch(
        "seq_gated_delta_vjp_" + suffix + ckpt_suffix, buffers, Dv, B * Hv);
  }

  if (stage_fp32) {
    dispatch_copy(dq_acc, dq);
    dispatch_copy(dk_acc, dk);
    dispatch_copy(dv_acc, dv);
    dispatch_copy(dg_acc, dg);
    dispatch_copy(db_acc, db);
  }
}

} // namespace mlx::core::fast
