// Ports mlx/backend/metal/cross_entropy.cpp: CrossEntropy and
// CrossEntropyVJP::eval_gpu. crosstl_backend.cpp includes this file at its
// end, so these ports share that translation unit's dispatch helpers.

namespace mlx::core::fast {
namespace {

constexpr uint64_t max_cross_entropy_elements = (uint64_t(1) << 28) - 1;

// cross_entropy.cpp's ensure_row_contiguous; the copy stays alive as an
// operand of the launch that reads it.
array cross_entropy_rows(const array& x) {
  if (x.flags().row_contiguous) {
    return x;
  }
  return row_kernel_copy(x);
}

// An array's data must lie inside its buffer before its words are bound.
void require_cross_entropy_data(const array& value, Dtype dtype) {
  if (value.dtype() != dtype || value.offset() < 0 ||
      uint64_t(value.offset()) > value.buffer_size() || value.data_size() == 0 ||
      value.data_size() >
          (value.buffer_size() - uint64_t(value.offset())) / value.itemsize() ||
      value.data_size() > max_cross_entropy_elements) {
    throw std::invalid_argument("CrossTL cross-entropy operand exceeds its bounds.");
  }
}

// cross_entropy_dims: one threadgroup per row of ceil(axis_size / 4) threads
// rounded up to a simdgroup, between one simdgroup and the pipelines' 1024.
void dispatch_cross_entropy(
    const std::string& entry, CrosstlMlxBuffer* buffers, uint32_t count,
    const array& in, int axis_size) {
  const uint64_t rows = in.data_size() / uint64_t(axis_size);
  if (!rows || rows > 65535 || rows * uint64_t(axis_size) != in.data_size()) {
    throw std::invalid_argument(
        "CrossTL cross-entropy rows exceed the portable group limit.");
  }
  const uint64_t needed = (uint64_t(axis_size) + 3) / 4;
  const uint32_t width =
      uint32_t(std::clamp<uint64_t>((needed + 31) / 32 * 32, 32, 1024));
  const CrosstlMlxLaunch launch{{uint32_t(rows), 1, 1}, {width, 1, 1}, {0, 0, 0}};
  char error[2048] = {};
  const int status = dispatch_callback.load()(
      entry.c_str(), buffers, count, in.data_size(), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native cross-entropy failed: ") + error);
  }
}

const char* cross_entropy_type(const array& in) {
  const char* dtype = row_kernel_type(in.dtype());
  if (!dtype || in.ndim() == 0 || in.shape().back() < 1) {
    throw std::invalid_argument(
        "CrossTL cross-entropy requires float32, float16 or bfloat16 logits.");
  }
  return dtype;
}

} // namespace

bool CrossEntropy::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 2 || outputs.size() != 1) {
    throw std::invalid_argument("CrossTL cross-entropy requires two inputs.");
  }
  auto& in_pre = inputs[0];
  auto& out = outputs[0];
  if (in_pre.size() == 0) {
    throw std::invalid_argument("[cross_entropy] Received empty array.");
  }
  const char* dtype = cross_entropy_type(in_pre);
  const array in = cross_entropy_rows(in_pre);
  const array targets = cross_entropy_rows(inputs[1]);
  require_cross_entropy_data(in, in_pre.dtype());
  require_cross_entropy_data(targets, int32);
  if (out.dtype() != float32) {
    throw std::invalid_argument("CrossTL cross-entropy writes float32 losses.");
  }
  out.set_data(allocator::malloc(out.nbytes()));
  int axis_size = in.shape().back();
  CrosstlMlxBuffer buffers[] = {
      {"in", dtype, const_cast<void*>(in.data<void>()), in.data_size(), 0},
      {"targets", "int32", const_cast<void*>(targets.data<void>()),
       targets.data_size(), 0},
      {"out", "float32", out.data<void>(), out.size(), 1},
      {"axis_size", "int32", &axis_size, 1, 0},
  };
  dispatch_cross_entropy(
      std::string("cross_entropy_") + dtype, buffers, 4, in, axis_size);
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  if (inputs.size() != 4 || outputs.size() != 1) {
    throw std::invalid_argument("CrossTL cross-entropy gradients require four inputs.");
  }
  auto& in_pre = inputs[0];
  auto& out = outputs[0];
  if (in_pre.size() == 0) {
    throw std::invalid_argument("[cross_entropy] Received empty array.");
  }
  const char* dtype = cross_entropy_type(in_pre);
  const bool donate_in = in_pre.is_donatable() || !in_pre.flags().row_contiguous;
  const array in = cross_entropy_rows(in_pre);
  const array targets = cross_entropy_rows(inputs[1]);
  const array loss = cross_entropy_rows(inputs[2]);
  const array cotan = cross_entropy_rows(inputs[3]);
  require_cross_entropy_data(in, in_pre.dtype());
  require_cross_entropy_data(targets, int32);
  require_cross_entropy_data(loss, float32);
  require_cross_entropy_data(cotan, float32);
  if (out.dtype() != in.dtype()) {
    throw std::invalid_argument(
        "CrossTL cross-entropy gradients take the logits' type.");
  }
  // The host uploads every operand before the launch and writes the
  // gradient back after it, so a donated input may share the output.
  if (donate_in) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  int axis_size = in.shape().back();
  CrosstlMlxBuffer buffers[] = {
      {"in", dtype, const_cast<void*>(in.data<void>()), in.data_size(), 0},
      {"targets", "int32", const_cast<void*>(targets.data<void>()),
       targets.data_size(), 0},
      {"loss", "float32", const_cast<void*>(loss.data<void>()), loss.data_size(), 0},
      {"cotan", "float32", const_cast<void*>(cotan.data<void>()), cotan.data_size(), 0},
      {"out", dtype, out.data<void>(), in.data_size(), 1},
      {"axis_size", "int32", &axis_size, 1, 0},
  };
  dispatch_cross_entropy(
      std::string("cross_entropy_vjp_") + dtype, buffers, 6, in, axis_size);
}

} // namespace mlx::core::fast
