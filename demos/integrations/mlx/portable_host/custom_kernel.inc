// Ports mlx/backend/metal/custom_kernel.cpp: CustomKernel::eval_gpu, which
// runs the kernels mx.fast.metal_kernel builds. MLX compiles the kernel's
// Metal source, metal_kernel.cpp's write_signature output, and binds each
// input, the shape, strides and dimension count the source reads, and every
// output at consecutive buffer slots. The host passes that source with every
// slot's buffer, named by its slot, and the exact thread grid, and the runtime
// translates the source on first use (custom_kernel_packages).
// crosstl_backend.cpp includes this file at its end, so the port shares that
// translation unit's dispatch helpers and copies.

#include <sstream>
#include <string>

namespace mlx::core::fast {
namespace {

constexpr uint64_t custom_kernel_max_elements = (uint64_t(1) << 31) - 1;
constexpr int custom_kernel_max_groups = 65535;
// Every translated target bounds a threadgroup at 1024 threads, the most an
// Apple GPU's pipeline reports as maxTotalThreadsPerThreadgroup.
constexpr int custom_kernel_max_threadgroup = 1024;

// The element extent ``count`` must lie inside the operand's allocation.
void custom_kernel_require_extent(const array& x, uint64_t count) {
  if (count == 0 || x.offset() < 0 || uint64_t(x.offset()) > x.buffer_size() ||
      count > custom_kernel_max_elements ||
      count > (x.buffer_size() - uint64_t(x.offset())) / x.itemsize()) {
    throw std::invalid_argument(
        "CrossTL custom kernel operand is empty or exceeds its allocation.");
  }
}

// The elements an input's view reaches from its data pointer: the whole
// array when it is row contiguous, otherwise its strided span. A host upload
// starts at the data pointer, so it cannot hold negative strides.
uint64_t custom_kernel_span(const array& x) {
  if (x.ndim() == 0 || x.flags().row_contiguous) {
    return x.size();
  }
  uint64_t span = 1;
  for (int axis = 0; axis < x.ndim(); ++axis) {
    if (x.shape(axis) == 0) {
      return 0;
    }
    const auto stride = x.strides()[axis];
    if (stride < 0 || uint64_t(stride) > custom_kernel_max_elements) {
      throw std::invalid_argument(
          "CrossTL custom kernels require non-negative input strides.");
    }
    span += (uint64_t(x.shape(axis)) - 1) * uint64_t(stride);
  }
  return span;
}

std::string custom_kernel_slot(size_t index) {
  return "buffer" + std::to_string(index);
}

} // namespace

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  // cuda_kernel's precompiled modules, shared memory and scalar arguments
  // have no Metal source to translate.
  if (is_precompiled_ || shared_memory_ != 0 || !scalar_arguments_.empty()) {
    throw std::invalid_argument("CrossTL custom kernels run Metal sources only.");
  }
  const auto [tx, ty, tz] = threadgroup_;
  const auto [gx, gy, gz] = grid_;
  if (tx < 1 || ty < 1 || tz < 1 || gx < 1 || gy < 1 || gz < 1) {
    throw std::invalid_argument("CrossTL custom kernels need a non-empty grid.");
  }
  const int64_t tg_size = int64_t(tx) * ty * tz;
  if (tg_size > custom_kernel_max_threadgroup) {
    std::ostringstream msg;
    msg << "Thread group size (" << tg_size << ") is greater than "
        << " the maximum allowed threads per threadgroup ("
        << custom_kernel_max_threadgroup << ").";
    throw std::invalid_argument(msg.str());
  }
  // dispatch_threads(grid, min(threadgroup, grid)): Metal runs the partial
  // edge groups, so the launch carries the exact thread grid.
  const uint32_t group[3] = {
      uint32_t(std::min(tx, gx)), uint32_t(std::min(ty, gy)),
      uint32_t(std::min(tz, gz))};
  const uint32_t extent[3] = {uint32_t(gx), uint32_t(gy), uint32_t(gz)};
  CrosstlMlxLaunch launch{};
  for (int axis = 0; axis < 3; ++axis) {
    const uint32_t groups = (extent[axis] + group[axis] - 1) / group[axis];
    if (groups > custom_kernel_max_groups) {
      throw std::invalid_argument(
          "CrossTL custom kernels launch at most 65535 groups per axis.");
    }
    launch.workgroup_count[axis] = groups;
    launch.workgroup_size[axis] = group[axis];
    launch.thread_grid_size[axis] = extent[axis];
  }

  if (shape_infos_.size() != inputs.size()) {
    throw std::invalid_argument("CrossTL custom kernel inputs differ from its source.");
  }
  // check_input: ensure_row_contiguous copies every other view row-major.
  std::vector<array> checked;
  checked.reserve(inputs.size());
  for (const auto& in : inputs) {
    if (ensure_row_contiguous_ && !in.flags().row_contiguous) {
      checked.push_back(contiguous_copy(in));
    } else {
      checked.push_back(in);
    }
  }
  for (auto& out : outputs) {
    if (!cast_type(out.dtype())) {
      throw std::invalid_argument("CrossTL custom kernel output type is unsupported.");
    }
    if (init_value_) {
      // fill_gpu(array(init_value, dtype), out): a broadcast scalar copied
      // into every element.
      array value(init_value_.value(), out.dtype());
      array fill(out.shape(), out.dtype(), nullptr, {});
      fill.copy_shared_buffer(
          value, Strides(out.ndim(), 0), {true, false, false}, 1);
      dispatch_copy(fill, out);
    } else {
      out.set_data(allocator::malloc(out.nbytes()));
    }
  }

  // The buffers in the slots write_signature assigns, each named by its slot.
  std::vector<std::string> names;
  std::vector<std::vector<int32_t>> shapes;
  std::vector<std::vector<int64_t>> strides;
  std::vector<int32_t> dimensions;
  const size_t slots = 2 * outputs.size() + 4 * inputs.size();
  names.reserve(slots);
  shapes.reserve(inputs.size());
  strides.reserve(inputs.size());
  dimensions.reserve(inputs.size());
  std::vector<CrosstlMlxBuffer> buffers;
  auto bind = [&](const char* dtype, const void* data, uint64_t count,
                  uint32_t direction) {
    names.push_back(custom_kernel_slot(buffers.size()));
    buffers.push_back(
        {names.back().c_str(), dtype, const_cast<void*>(data), count, direction});
  };
  for (size_t i = 0; i < checked.size(); ++i) {
    const auto& in = checked[i];
    const char* dtype = cast_type(in.dtype());
    if (!dtype) {
      throw std::invalid_argument("CrossTL custom kernel input type is unsupported.");
    }
    const uint64_t span = custom_kernel_span(in);
    custom_kernel_require_extent(in, span);
    bind(dtype, in.data<void>(), span, 0);
    if (in.ndim() > 0) {
      const auto& [shape, stride, ndim] = shape_infos_[i];
      if (shape) {
        shapes.emplace_back(in.shape().begin(), in.shape().end());
        bind("int32", shapes.back().data(), shapes.back().size(), 0);
      }
      if (stride) {
        strides.emplace_back(in.strides().begin(), in.strides().end());
        bind("int64", strides.back().data(), strides.back().size(), 0);
      }
      if (ndim) {
        dimensions.push_back(in.ndim());
        bind("int32", &dimensions.back(), 1, 0);
      }
    }
  }
  for (auto& out : outputs) {
    custom_kernel_require_extent(out, out.size());
    bind(
        cast_type(out.dtype()), out.data<void>(), out.size(),
        init_value_ ? CROSTL_MLX_BUFFER_INOUT : 1);
  }
  int32_t math_mode = compile_options_;
  buffers.push_back(
      {"kernel_source", "utf8", const_cast<char*>(source_.data()), source_.size(), 0});
  buffers.push_back({"compile_options", "int32", &math_mode, 1, 0});

  char error[2048] = {};
  const int status = dispatch_callback.load()(
      name_.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
      uint64_t(gx) * uint64_t(gy) * uint64_t(gz), &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native custom kernel failed: ") + error);
  }
}

} // namespace mlx::core::fast
