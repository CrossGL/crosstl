// Ports mlx/backend/metal/gated_delta_update.cpp: GatedDeltaUpdate and
// GatedDeltaUpdateVJP's use_fallback and eval_gpu. The host reports no NAX,
// so gated_delta_chunk_size resolves a requested 16-step chunk to the
// eight-step fused chunk kernel, gated_delta_chunk_size_vjp returns the
// sequential backward and gated_delta_ckpt returns 1. Each launch binds the
// pinned kernels' buffers in their order, with the (32, 4, 1) threadgroups
// and grids gated_delta_update.cpp dispatches. The forward fills its output
// with zeros first, and the backward its five float32 accumulators and the
// state gradient, as upstream does; the backward then casts the
// accumulators to a 16-bit gradient type. gated_delta_layout checks each
// launch against the kernels' addressing.
// crosstl_backend.cpp includes this file at its end, so it shares that
// translation unit's copy, cast and dispatch helpers.

namespace mlx::core::fast {
namespace {

// get_type_string's names for the instantiated gated-delta types.
const char* gated_delta_type_string(Dtype dtype) {
  return dtype == float32 ? "float" : dtype == float16 ? "float16_t" :
      dtype == bfloat16 ? "bfloat16_t" : nullptr;
}

const char* gated_delta_storage_type(Dtype dtype) {
  return dtype == float32 ? "float32" : dtype == float16 ? "float16" :
      dtype == bfloat16 ? "bfloat16" : nullptr;
}

int gated_delta_chunk_size(int T) {
  // metal::is_nax_available() is false on the host.
  const int C = env::get_var("GATED_DELTA_CHUNK", T > 8 ? 8 : 1);
  return C == 16 ? 8 : C;
}

int gated_delta_chunk_size_vjp() {
  return 1;
}

int gated_delta_ckpt(int chunk) {
  int c = env::get_var("GATED_DELTA_CKPT", 16);
  if (c != 1 && c != 4 && c != 8 && c != 16) {
    c = 16;
  }
  return chunk == 16 ? c : 1;
}

bool supported_gated_delta_shape(int Hk, int Dk, int Hv, int Dv) {
  if (Dk != 128 || Dv != 128) {
    return false;
  }
  return (Hk == 24 && Hv == 24) || (Hk == 32 && Hv == 32) ||
      (Hk == 16 && Hv == 32) || (Hk == 16 && Hv == 48) ||
      (Hk == 16 && Hv == 16) || (Hk == 16 && Hv == 64);
}

// ensure_row_contiguous: a row-contiguous copy of any other view.
array gated_delta_operand(const array& x) {
  return dense_input(x);
}

array gated_delta_alloc(Shape shape, Dtype dtype) {
  array a(std::move(shape), dtype, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

// fill_gpu(array(0, dtype), a): a broadcast zero copied into every element.
void gated_delta_zero(array& a) {
  array zero(0, a.dtype());
  array fill(a.shape(), a.dtype(), nullptr, {});
  fill.copy_shared_buffer(zero, Strides(a.ndim(), 0), {true, false, false}, 1);
  dispatch_copy(fill, a);
}

uint32_t gated_delta_groups(int64_t value) {
  if (value < 1 || value > 65535) {
    throw std::invalid_argument("CrossTL gated delta grid exceeds the portable group limit.");
  }
  return uint32_t(value);
}

CrosstlMlxBuffer gated_delta_buffer(const char* name, const array& a, uint32_t direction) {
  const char* storage = gated_delta_storage_type(a.dtype());
  if (storage == nullptr) {
    throw std::invalid_argument("CrossTL gated delta binds float32, float16 or bfloat16 arrays.");
  }
  return {name, storage, const_cast<void*>(a.data<void>()), uint64_t(a.size()), direction};
}

void gated_delta_launch(
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
    throw std::runtime_error(std::string("CrossTL native gated delta failed: ") + error);
  }
}

// The forward kernels' buffers 0 through 8, in gated_delta_update.cpp's order.
std::vector<CrosstlMlxBuffer> gated_delta_forward_buffers(
    const array& q,
    const array& k,
    const array& v,
    const array& h0,
    const array& g,
    const array& beta,
    const array& y,
    uint32_t y_direction,
    const array& state_out,
    int32_t& T) {
  return {
      gated_delta_buffer("q", q, 0),
      gated_delta_buffer("k", k, 0),
      gated_delta_buffer("v", v, 0),
      gated_delta_buffer("state_in", h0, 0),
      gated_delta_buffer("g", g, 0),
      gated_delta_buffer("beta", beta, 0),
      gated_delta_buffer("y", y, y_direction),
      gated_delta_buffer("state_out", state_out, 1),
      {"T", "int32", &T, 1, 0}};
}

std::string gated_delta_suffix(const array& q, int Dk, int Dv, int Hk, int Hv) {
  const char* type = gated_delta_type_string(q.dtype());
  if (type == nullptr) {
    throw std::invalid_argument("CrossTL gated delta requires float32, float16 or bfloat16 inputs.");
  }
  return std::string(type) + "_" + std::to_string(Dk) + "_" + std::to_string(Dv) + "_" +
      std::to_string(Hk) + "_" + std::to_string(Hv);
}

void require_gated_delta_shape(int B, int T, int Hk, int Dk, int Hv, int Dv) {
  if (!supported_gated_delta_shape(Hk, Dk, Hv, Dv) || B < 1 || T < 1 ||
      int64_t(B) * Hv > 65535) {
    throw std::invalid_argument(
        "CrossTL gated delta requires an instantiated head shape, a nonempty sequence and at most 65535 batch heads.");
  }
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
  auto q = gated_delta_operand(inputs[0]);
  auto k = gated_delta_operand(inputs[1]);
  auto v = gated_delta_operand(inputs[2]);
  auto g = gated_delta_operand(inputs[3]);
  auto beta = gated_delta_operand(inputs[4]);
  auto h0 = gated_delta_operand(inputs[5]);

  auto& out = outputs[0];
  auto& hf = outputs[1];

  const int B = q.shape(0);
  int32_t T = q.shape(1);
  const int Hk = q.shape(2);
  const int Dk = q.shape(3);
  const int Hv = v.shape(2);
  const int Dv = v.shape(3);
  require_gated_delta_shape(B, T, Hk, Dk, Hv, Dv);

  const int C = gated_delta_chunk_size(T);
  const int ckpt = gated_delta_ckpt(C);
  const std::string suffix = gated_delta_suffix(q, Dk, Dv, Hk, Hv);

  gated_delta_zero(out);
  hf.set_data(allocator::malloc(hf.nbytes()));

  // The forward is inference-only: save_state (function constant 200) is
  // false, which each entry's name implies.
  std::string name;
  uint32_t rows;
  switch (C) {
    case 8:
      name = "gated_delta_fused_chunk_" + suffix + "_" + std::to_string(C);
      rows = uint32_t(Dv / 8);
      break;
    case 1:
    case 0:
      name = "seq_gated_delta_" + suffix + "_" + std::to_string(ckpt);
      rows = uint32_t(Dv);
      break;
    default:
      throw std::runtime_error(
          "NYI: Only sequential and chunk size 8,16 are supported");
  }
  auto buffers = gated_delta_forward_buffers(q, k, v, h0, g, beta, out, 2, hf, T);
  const CrosstlMlxLaunch launch{
      {1, gated_delta_groups(rows / 4), gated_delta_groups(int64_t(B) * Hv)}, {32, 4, 1}};
  gated_delta_launch(name, buffers, out.size(), launch);
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
  // Inputs are q, k, v, g, b, h0, cot_o, cot_h; outputs dq, dk, dv, dg, db, dh0.
  auto q = gated_delta_operand(inputs[0]);
  auto k = gated_delta_operand(inputs[1]);
  auto v = gated_delta_operand(inputs[2]);
  auto g = gated_delta_operand(inputs[3]);
  auto beta = gated_delta_operand(inputs[4]);
  auto h0 = gated_delta_operand(inputs[5]);
  auto cot_o = gated_delta_operand(inputs[6]);
  auto cot_h = gated_delta_operand(inputs[7]);

  const int B = q.shape(0);
  int32_t T = q.shape(1);
  const int Hk = q.shape(2);
  const int Dk = q.shape(3);
  const int Hv = v.shape(2);
  const int Dv = v.shape(3);
  require_gated_delta_shape(B, T, Hk, Dk, Hv, Dv);

  // Without NAX the backward is sequential: one state per step.
  const int C = gated_delta_chunk_size_vjp();
  const int ckpt = gated_delta_ckpt(C);
  const int n_states = T;

  auto& dq = outputs[0];
  auto& dk = outputs[1];
  auto& dv = outputs[2];
  auto& dg = outputs[3];
  auto& db = outputs[4];
  auto& dh = outputs[5];

  // The kernels accumulate their gradients as fp32, so a 16-bit type is
  // staged in float32 accumulators and cast at the end.
  const bool stage_fp32 = q.dtype() != float32;
  std::vector<array> accumulators;
  for (auto* gradient : {&dq, &dk, &dv, &dg, &db}) {
    accumulators.push_back(
        stage_fp32 ? array(gradient->shape(), float32, nullptr, {}) : *gradient);
  }
  for (auto& accumulator : accumulators) {
    gated_delta_zero(accumulator);
  }
  gated_delta_zero(dh);

  array state_cache = gated_delta_alloc({B, Hv, n_states, Dv, Dk}, float32);
  array y_scratch = gated_delta_alloc({B, T, Hv, Dv}, q.dtype());
  array hf_scratch = gated_delta_alloc({B, Hv, Dv, Dk}, float32);

  const std::string suffix = gated_delta_suffix(q, Dk, Dv, Hk, Hv);
  const std::string ckpt_suffix = "_" + std::to_string(ckpt);
  const CrosstlMlxLaunch launch{
      {1, gated_delta_groups(Dv / 4), gated_delta_groups(int64_t(B) * Hv)}, {32, 4, 1}};
  {
    // The save pass: save_state is true, so the pipeline's hash name is the
    // kernel name with "_save".
    auto buffers = gated_delta_forward_buffers(
        q, k, v, h0, g, beta, y_scratch, 1, hf_scratch, T);
    buffers.push_back(gated_delta_buffer("state_cache", state_cache, 1));
    gated_delta_launch(
        "seq_gated_delta_" + suffix + ckpt_suffix + "_save", buffers, y_scratch.size(),
        launch);
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
        {"T", "int32", &T, 1, 0},
        gated_delta_buffer("dq", accumulators[0], 2),
        gated_delta_buffer("dk", accumulators[1], 2),
        gated_delta_buffer("dv", accumulators[2], 2),
        gated_delta_buffer("dg", accumulators[3], 2),
        gated_delta_buffer("db", accumulators[4], 2),
        gated_delta_buffer("dh", dh, 2)};
    gated_delta_launch(
        "seq_gated_delta_vjp_" + suffix + ckpt_suffix, buffers, accumulators[2].size(),
        launch);
  }
  if (stage_fp32) {
    std::vector<array*> gradients = {&dq, &dk, &dv, &dg, &db};
    for (size_t index = 0; index < gradients.size(); ++index) {
      // copy_gpu(acc, gradient, CopyType::General): a cast of the dense
      // accumulator.
      dispatch_cast({accumulators[index]}, *gradients[index]);
    }
  }
}

} // namespace mlx::core::fast
