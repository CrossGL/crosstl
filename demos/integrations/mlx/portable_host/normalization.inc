// Ports mlx/backend/metal/normalization.cpp: RMSNorm, RMSNormVJP, LayerNorm
// and LayerNormVJP::eval_gpu. crosstl_backend.cpp includes this file at its
// end, so these ports share that translation unit's dispatch helpers.

namespace mlx::core::fast {
namespace {

constexpr uint64_t max_norm_elements = (uint64_t(1) << 28) - 1;

// An array's data must lie inside its buffer before its words are bound.
void require_norm_data(const array& value, const char* label) {
  if (value.offset() < 0 || uint64_t(value.offset()) > value.buffer_size() ||
      value.data_size() == 0 ||
      value.data_size() >
          (value.buffer_size() - uint64_t(value.offset())) / value.itemsize() ||
      value.data_size() > max_norm_elements) {
    throw std::invalid_argument(
        std::string("CrossTL ") + label + " operand exceeds its bounds.");
  }
}

// normalization.cpp binds a weight or bias at its only stride, or at zero
// when it has no axis; a negative stride has no unsigned form to bind.
uint32_t norm_stride(const array& value, const char* label) {
  if (value.ndim() != 1) {
    return 0;
  }
  if (value.strides()[0] < 0 || value.strides()[0] > int64_t(max_norm_elements)) {
    throw std::invalid_argument(
        std::string("CrossTL ") + label + " parameters require non-negative strides.");
  }
  return uint32_t(value.strides()[0]);
}

void require_norm_types(
    const std::vector<array>& inputs, const array& out, const char* label) {
  if (!row_kernel_type(out.dtype()) || inputs[0].ndim() == 0) {
    throw std::invalid_argument(
        std::string("CrossTL ") + label +
        " requires float32, float16 or bfloat16 rows.");
  }
  for (const auto& input : inputs) {
    if (input.dtype() != out.dtype()) {
      throw std::invalid_argument(
          std::string("CrossTL ") + label + " operands must share one type.");
    }
  }
}

// RMSNorm and LayerNorm::eval_gpu's set_output: rows whose last axis is
// contiguous and whose row stride is the row length or zero, or a single
// row, run in place; other inputs run from a row-contiguous copy.
array norm_rows(const array& x, array& out) {
  bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
  if (no_copy && x.ndim() > 1) {
    auto s = x.strides()[x.ndim() - 2];
    no_copy &= (s == 0 || s == x.shape().back() || x.shape(-2) == 1);
  }
  if (no_copy) {
    if (x.is_donatable()) {
      out.copy_shared_buffer(x);
    } else {
      out.set_data(
          allocator::malloc(x.data_size() * x.itemsize()),
          x.data_size(),
          x.strides(),
          x.flags());
    }
    return x;
  }
  array x_copy = row_kernel_copy(x);
  out.copy_shared_buffer(x_copy);
  return x_copy;
}

// The VJPs' check_input: row-contiguous operands bind directly.
std::pair<array, bool> norm_row_contiguous(const array& x) {
  if (x.flags().row_contiguous) {
    return {x, false};
  }
  return {row_kernel_copy(x), true};
}

// Single-row kernels run ceil(axis_size / reads) threads rounded up to a
// simdgroup; looped kernels run the pipeline's 1024 threads.
uint32_t norm_width(uint32_t axis_size, int reads, bool looped) {
  return looped ? 1024 : uint32_t(((uint64_t(axis_size) + reads - 1) / reads + 31) / 32 * 32);
}

// Rows the launch covers; the host bounds every launch axis at 65535 groups.
uint32_t norm_row_count(const array& x, const char* label) {
  require_norm_data(x, label);
  const uint64_t axis_size = uint64_t(x.shape().back());
  if (axis_size == 0 || x.data_size() % axis_size) {
    throw std::invalid_argument(
        std::string("CrossTL ") + label + " input does not cover whole rows.");
  }
  return uint32_t(x.data_size() / axis_size);
}

void dispatch_norm(
    const std::string& entry, CrosstlMlxBuffer* buffers, uint32_t count,
    uint64_t logical, uint32_t groups, uint32_t width, const char* label) {
  if (groups == 0 || groups > 65535) {
    throw std::invalid_argument(
        std::string("CrossTL ") + label + " rows exceed the portable group limit.");
  }
  const CrosstlMlxLaunch launch{{groups, 1, 1}, {width, 1, 1}, {0, 0, 0}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, count, logical, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native ") + label + " failed: " + error);
  }
}

// strided_reduce_general_dispatch's column sum over a row-contiguous input
// read as its (rows, axis_size) plan, whatever the input's rank: Reduce's own
// GPU plan for axis 0 of that view.
void sum_norm_rows(Stream s, const array& rows, array& out) {
  const int64_t axis_size = rows.ndim() == 0 ? 1 : rows.shape().back();
  if (!rows.flags().row_contiguous || rows.data_size() != rows.size() || axis_size < 1 ||
      rows.size() % axis_size != 0 || out.size() != size_t(axis_size)) {
    throw std::invalid_argument("CrossTL normalization column sums require dense rows.");
  }
  const int64_t count = rows.size() / axis_size;
  array view({int(count), int(axis_size)}, rows.dtype(), nullptr, {});
  view.copy_shared_buffer(
      rows,
      {axis_size, 1},
      {true, true, count == 1 || axis_size == 1},
      rows.data_size());
  Reduce(s, Reduce::Sum, {0}).eval_gpu({view}, out);
}

} // namespace

bool RMSNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 2 || outputs.size() != 1) {
    throw std::invalid_argument("CrossTL RMS normalization requires two inputs.");
  }
  auto& out = outputs[0];
  require_norm_types(inputs, out, "RMS normalization");
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  const array x = norm_rows(inputs[0], out);
  const array& w = inputs[1];
  require_norm_data(w, "RMS normalization");
  uint32_t axis_size = uint32_t(x.shape().back());
  const uint32_t n_rows = norm_row_count(x, "RMS normalization");
  const bool looped = axis_size > 4096;
  const char* dtype = row_kernel_type(out.dtype());
  const std::string entry = std::string(looped ? "rms_looped" : "rms") + dtype;
  float eps = eps_;
  uint32_t w_stride = norm_stride(w, "RMS normalization");
  CrosstlMlxBuffer buffers[] = {
      {"x", dtype, const_cast<void*>(x.data<void>()), x.data_size(), 0},
      {"w", dtype, const_cast<void*>(w.data<void>()), w.data_size(), 0},
      {"out", dtype, out.data<void>(), x.data_size(), 1},
      {"eps", "float32", &eps, 1, 0},
      {"axis_size", "uint32", &axis_size, 1, 0},
      {"w_stride", "uint32", &w_stride, 1, 0},
  };
  dispatch_norm(
      entry, buffers, 6, x.data_size(), n_rows, norm_width(axis_size, 4, looped),
      "RMS normalization");
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 3 || outputs.size() != 2) {
    throw std::invalid_argument("CrossTL RMS normalization gradients require three inputs.");
  }
  auto [x, copied] = norm_row_contiguous(inputs[0]);
  const array& w = inputs[1];
  auto [g, g_copied] = norm_row_contiguous(inputs[2]);
  array& gx = outputs[0];
  array& gw = outputs[1];
  require_norm_types({x, w, g}, gx, "RMS normalization gradient");
  const bool has_w = w.ndim() != 0;
  if (x.size() == 0) {
    gx.set_data(allocator::malloc(0));
    gw.set_data(allocator::malloc(gw.nbytes()));
    return;
  }
  require_norm_data(w, "RMS normalization gradient");
  require_norm_data(g, "RMS normalization gradient");
  if (x.is_donatable()) {
    gx.copy_shared_buffer(x);
  } else if (g.is_donatable()) {
    gx.copy_shared_buffer(g);
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  uint32_t axis_size = uint32_t(x.shape().back());
  uint32_t n_rows = norm_row_count(x, "RMS normalization gradient");
  const bool looped = axis_size > 4096;
  uint32_t rows_per_group = 1;
  uint32_t n_groups = n_rows;
  if (!looped) {
    rows_per_group = (n_rows + 511) / 512;
    n_groups = (n_rows + rows_per_group - 1) / rows_per_group;
  }
  const char* dtype = row_kernel_type(gx.dtype());
  // Without a weight the kernel never writes gw; one element stands in for
  // the weight scalar normalization.cpp binds there.
  array gw_temp = has_w
      ? array({int(n_groups), x.shape().back()}, gw.dtype(), nullptr, {})
      : array({1}, gw.dtype(), nullptr, {});
  gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
  const std::string entry = std::string(looped ? "vjp_rms_looped" : "vjp_rms") + dtype +
      (has_w ? "_w" : "_now");
  float eps = eps_;
  uint32_t w_stride = norm_stride(w, "RMS normalization gradient");
  CrosstlMlxBuffer buffers[] = {
      {"x", dtype, const_cast<void*>(x.data<void>()), x.data_size(), 0},
      {"w", dtype, const_cast<void*>(w.data<void>()), w.data_size(), 0},
      {"g", dtype, const_cast<void*>(g.data<void>()), g.data_size(), 0},
      {"gx", dtype, gx.data<void>(), x.data_size(), 1},
      {"gw", dtype, gw_temp.data<void>(), gw_temp.size(), 1},
      {"eps", "float32", &eps, 1, 0},
      {"axis_size", "uint32", &axis_size, 1, 0},
      {"w_stride", "uint32", &w_stride, 1, 0},
      {"n_rows", "uint32", &n_rows, 1, 0},
      {"rows_per_group", "uint32", &rows_per_group, 1, 0},
  };
  dispatch_norm(
      entry, buffers, looped ? 8 : 10, x.data_size(), n_groups,
      norm_width(axis_size, 4, looped), "RMS normalization gradient");
  if (has_w) {
    sum_norm_rows(stream(), gw_temp, gw);
  } else {
    gw.set_data(allocator::malloc(gw.nbytes()));
  }
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 3 || outputs.size() != 1) {
    throw std::invalid_argument("CrossTL layer normalization requires three inputs.");
  }
  auto& out = outputs[0];
  require_norm_types(inputs, out, "layer normalization");
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  const array x = norm_rows(inputs[0], out);
  const array& w = inputs[1];
  const array& b = inputs[2];
  require_norm_data(w, "layer normalization");
  require_norm_data(b, "layer normalization");
  uint32_t axis_size = uint32_t(x.shape().back());
  const uint32_t n_rows = norm_row_count(x, "layer normalization");
  const bool looped = axis_size > 6656;
  const char* dtype = row_kernel_type(out.dtype());
  const std::string entry =
      std::string(looped ? "layer_norm_looped" : "layer_norm") + dtype;
  float eps = eps_;
  uint32_t w_stride = norm_stride(w, "layer normalization");
  uint32_t b_stride = norm_stride(b, "layer normalization");
  CrosstlMlxBuffer buffers[] = {
      {"x", dtype, const_cast<void*>(x.data<void>()), x.data_size(), 0},
      {"w", dtype, const_cast<void*>(w.data<void>()), w.data_size(), 0},
      {"b", dtype, const_cast<void*>(b.data<void>()), b.data_size(), 0},
      {"out", dtype, out.data<void>(), x.data_size(), 1},
      {"eps", "float32", &eps, 1, 0},
      {"axis_size", "uint32", &axis_size, 1, 0},
      {"w_stride", "uint32", &w_stride, 1, 0},
      {"b_stride", "uint32", &b_stride, 1, 0},
  };
  dispatch_norm(
      entry, buffers, 8, x.data_size(), n_rows,
      norm_width(axis_size, looped ? 4 : 8, looped), "layer normalization");
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 4 || outputs.size() != 3) {
    throw std::invalid_argument("CrossTL layer normalization gradients require four inputs.");
  }
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[3].is_donatable();
  auto [x, copied] = norm_row_contiguous(inputs[0]);
  donate_x |= copied;
  const array& w = inputs[1];
  auto [g, g_copied] = norm_row_contiguous(inputs[3]);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];
  require_norm_types({x, w, inputs[2], g}, gx, "layer normalization gradient");
  const bool has_w = w.ndim() != 0;
  if (x.size() == 0) {
    gx.set_data(allocator::malloc(0));
    gw.set_data(allocator::malloc(gw.nbytes()));
    gb.set_data(allocator::malloc(gb.nbytes()));
    return;
  }
  require_norm_data(w, "layer normalization gradient");
  require_norm_data(g, "layer normalization gradient");
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  uint32_t axis_size = uint32_t(x.shape().back());
  const uint32_t n_rows = norm_row_count(x, "layer normalization gradient");
  // The bias gradient sums g before the kernel can overwrite it in place.
  if (gb.ndim() == 1 && gb.size() == axis_size) {
    sum_norm_rows(stream(), g, gb);
  } else {
    gb.set_data(allocator::malloc(gb.nbytes()));
  }
  array gw_temp = has_w
      ? array({int(n_rows), x.shape().back()}, gw.dtype(), nullptr, {})
      : array({1}, gw.dtype(), nullptr, {});
  if (has_w && !g_in_gx && donate_g) {
    gw_temp.copy_shared_buffer(g);
  } else {
    gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
  }
  const bool looped = axis_size > 8192;
  const char* dtype = row_kernel_type(gx.dtype());
  const std::string entry =
      std::string(looped ? "vjp_layer_norm_looped" : "vjp_layer_norm") + dtype +
      (has_w ? "_w" : "_now");
  float eps = eps_;
  uint32_t w_stride = norm_stride(w, "layer normalization gradient");
  CrosstlMlxBuffer buffers[] = {
      {"x", dtype, const_cast<void*>(x.data<void>()), x.data_size(), 0},
      {"w", dtype, const_cast<void*>(w.data<void>()), w.data_size(), 0},
      {"g", dtype, const_cast<void*>(g.data<void>()), g.data_size(), 0},
      {"gx", dtype, gx.data<void>(), x.data_size(), 1},
      {"gw", dtype, gw_temp.data<void>(), gw_temp.size(), 1},
      {"eps", "float32", &eps, 1, 0},
      {"axis_size", "uint32", &axis_size, 1, 0},
      {"w_stride", "uint32", &w_stride, 1, 0},
  };
  dispatch_norm(
      entry, buffers, 8, x.data_size(), n_rows,
      norm_width(axis_size, looped ? 4 : 8, looped), "layer normalization gradient");
  if (has_w) {
    sum_norm_rows(stream(), gw_temp, gw);
  } else {
    gw.set_data(allocator::malloc(gw.nbytes()));
  }
}

} // namespace mlx::core::fast
