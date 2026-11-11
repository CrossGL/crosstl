// Ports mlx/backend/metal/compiled.cpp: Compiled::eval_gpu. MLX generates one
// Metal library of elementwise kernels for a fused tape and launches the
// variant its inputs' layout selects; the host generates exactly that
// variant with the same build_kernel and hands its source to the runtime,
// which translates it on first use. crosstl_backend.cpp includes this file at
// its end, so the port shares that translation unit's dispatch helpers.

#include <fmt/format.h>

#include "mlx/backend/common/compiled.h"
#include "mlx/graph_utils.h"

namespace mlx::core {
namespace {

// The headers metal::utils(), unary_ops(), binary_ops() and ternary_ops()
// embed in every compiled library, in that order.
constexpr const char* compiled_preamble =
    "#include \"mlx/backend/metal/kernels/utils.h\"\n"
    "#include \"mlx/backend/metal/kernels/unary_ops.h\"\n"
    "#include \"mlx/backend/metal/kernels/binary_ops.h\"\n"
    "#include \"mlx/backend/metal/kernels/ternary_ops.h\"\n";

// Each host launch covers at most 65535 one-thread groups along an axis.
constexpr uint64_t compiled_max_groups = 65535;
constexpr uint64_t compiled_max_elements = (uint64_t(1) << 31) - 1;

// MLX names a library by its tape, which grows with the fused graph; the
// host names it by the library name's FNV-1a hash, which every platform
// computes alike, so translated paths stay short.
std::string compiled_kernel_name(const std::string& library) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char byte : library) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return fmt::format("compiled_{:016x}", hash);
}

// metal/utils.h get_work_per_thread.
int compiled_work_per_thread(Dtype dtype) {
  return std::max(1, 8 / static_cast<int>(dtype.size()));
}

// Verbatim port of compiled.cpp build_kernel.
void build_compiled_kernel(
    std::string& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool contiguous,
    int ndim,
    bool dynamic_dims,
    bool use_big_index,
    int work_per_thread) {
  NodeNamer namer;
  bool add_indices = false;
  int cnt = 0;

  os += fmt::format(
      "[[host_name(\"{0}\")]]\n[[kernel]] void {0}(\n", kernel_name);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) {
      continue;
    }
    const auto& x = inputs[i];
    auto& xname = namer.get_name(x);
    if (!is_scalar(x) && !contiguous) {
      add_indices = true;
    }
    os += fmt::format(
        "    device const {0}* {1} [[buffer({2})]],\n",
        get_type_string(x.dtype()),
        xname,
        cnt++);
  }

  std::string idx_type = use_big_index ? "int64_t" : "uint";
  if (add_indices) {
    os += fmt::format(
        "    constant const int64_t* in_strides [[buffer({0})]],\n", cnt++);
  }

  for (auto& x : outputs) {
    os += fmt::format(
        "    device {0}* {1} [[buffer({2})]],\n",
        get_type_string(x.dtype()),
        namer.get_name(x),
        cnt++);
  }
  if (!contiguous) {
    os += fmt::format(
        "    constant const int* output_shape [[buffer({0})]],\n", cnt++);
  } else {
    os += fmt::format(
        "    constant const {0}& size [[buffer({1})]],\n", idx_type, cnt++);
  }
  if (dynamic_dims) {
    os += fmt::format("    constant const int& ndim [[buffer({0})]],\n", cnt++);
  }

  os += "    uint3 pos [[thread_position_in_grid]],\n";
  os += "    uint3 grid [[threads_per_grid]]) {\n";

  os += fmt::format("  constexpr int N_ = {0};\n", work_per_thread);
  if (contiguous && use_big_index) {
    os += "  int64_t index = N_ * (pos.x + grid.x * int64_t(pos.y));\n";
  } else if (contiguous) {
    os += "  uint index = N_ * pos.x;\n";
  } else if (work_per_thread > 1) {
    os += fmt::format(
        "  int xshape = output_shape[{0}];\n",
        dynamic_dims ? "ndim - 1" : std::to_string(ndim - 1));
    os += fmt::format(
        "  {0} index = N_ * pos.x + xshape * (pos.y + {0}(grid.y) * pos.z);\n",
        idx_type);
  } else {
    os += fmt::format(
        "  {0} index = pos.x + grid.x * (pos.y + {0}(grid.y) * pos.z);\n",
        idx_type);
  }
  if (work_per_thread > 1 && contiguous) {
    os += "  for (int i = 0; i < N_ && index < size; ++i) {\n";
  }

  std::vector<array> nc_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& x = inputs[i];
    auto& xname = namer.get_name(x);

    if (is_constant(i)) {
      std::ostringstream ss;
      print_constant(ss, x);
      os += fmt::format(
          "  auto tmp_{0} = static_cast<{1}>({2});\n",
          xname,
          get_type_string(x.dtype()),
          ss.str());
    } else if (is_scalar(x)) {
      os += fmt::format(
          "  {0} tmp_{1} = {1}[0];\n", get_type_string(x.dtype()), xname);
    } else if (contiguous) {
      os += fmt::format(
          "  {0} tmp_{1} = {1}[index];\n", get_type_string(x.dtype()), xname);
    } else {
      nc_inputs.push_back(x);
    }
  }

  for (size_t i = 0; i < nc_inputs.size(); ++i) {
    auto& xname = namer.get_name(nc_inputs[i]);
    os += fmt::format("  {0} index_{1} = ", idx_type, xname);
    if (ndim == 1) {
      int offset = i * ndim;
      os += fmt::format(
          "elem_to_loc_1<{0}>(pos.x, in_strides[{1}]);\n", idx_type, offset);
    } else if (ndim == 2) {
      int offset = i * ndim;
      os += fmt::format(
          "elem_to_loc_2<{0}>({{pos.x, pos.y}}, in_strides + {1});\n",
          idx_type,
          offset);
    } else if (ndim == 3) {
      int offset = i * ndim;
      os += fmt::format(
          "elem_to_loc_3<{0}>(pos, in_strides + {1});\n", idx_type, offset);
    } else if (!dynamic_dims) {
      int offset = (i + 1) * ndim;
      os += fmt::format(
          "N_ * pos.x * {0}(in_strides[{1}]) + pos.y * {0}(in_strides[{2}]);\n",
          idx_type,
          offset - 1,
          offset - 2);
    } else {
      os += fmt::format(
          "N_ * pos.x * {0}(in_strides[ndim * {1} + ndim - 1]) + pos.y * {0}(in_strides[ndim * {1} + ndim - 2]);\n",
          idx_type,
          i);
    }
  }

  if (!nc_inputs.empty() && (ndim > 3 || dynamic_dims)) {
    os += "  uint zpos = pos.z;\n";
    if (dynamic_dims) {
      os += "  for (int d = ndim - 3; d >= 0; --d) {\n";
    } else {
      os += fmt::format("  for (int d = {0}; d >= 0; --d) {{\n", ndim - 3);
    }
    os += "    uint l = zpos % output_shape[d];\n";
    for (size_t i = 0; i < nc_inputs.size(); ++i) {
      auto& xname = namer.get_name(nc_inputs[i]);
      os += fmt::format("    index_{0} += ", xname);
      if (dynamic_dims) {
        os +=
            fmt::format("l * {0}(in_strides[{1} * ndim + d]);\n", idx_type, i);
      } else {
        os +=
            fmt::format("l * {0}(in_strides[{1} + d]);\n", idx_type, i * ndim);
      }
    }
    os += "    zpos /= output_shape[d];\n  }\n";
  }

  if (work_per_thread > 1 && !contiguous) {
    os +=
        "  for (int i = 0; i < N_ && (int(N_ * pos.x) + i) < xshape; ++i) {\n";
  }

  for (size_t i = 0; i < nc_inputs.size(); ++i) {
    auto& x = nc_inputs[i];
    auto& xname = namer.get_name(x);
    os += fmt::format(
        "  {0} tmp_{1} = {1}[index_{1}];\n", get_type_string(x.dtype()), xname);
  }

  for (auto& x : tape) {
    os += fmt::format(
        "  {0} tmp_{1} = ", get_type_string(x.dtype()), namer.get_name(x));
    if (is_static_cast(x.primitive())) {
      os += fmt::format(
          "cast_to<{0}>(tmp_{1});\n",
          get_type_string(x.dtype()),
          namer.get_name(x.inputs()[0]));
    } else {
      os += x.primitive().name();
      os += "()(";
      for (size_t i = 0; i + 1 < x.inputs().size(); i++) {
        os += fmt::format("tmp_{0}, ", namer.get_name(x.inputs()[i]));
      }
      os += fmt::format("tmp_{0});\n", namer.get_name(x.inputs().back()));
    }
  }

  for (auto& x : outputs) {
    os += fmt::format("  {0}[index] = tmp_{0};\n", namer.get_name(x));
  }
  if (work_per_thread > 1) {
    for (size_t i = 0; i < nc_inputs.size(); ++i) {
      auto& x = nc_inputs[i];
      auto& xname = namer.get_name(x);
      if (!dynamic_dims) {
        os += fmt::format(
            "  index_{0} += in_strides[{1}];\n", xname, i * ndim + ndim - 1);
      } else {
        os += fmt::format(
            "  index_{0} += in_strides[{1} * ndim + ndim - 1];\n", xname, i);
      }
    }
    os += "  index++;\n  }\n";
  }

  os += "}\n";

  if (cnt > 31) {
    std::ostringstream msg;
    msg << "[compile] Too many inputs/outputs fused in the Metal Compiled "
        << "primitive which exhausted the available argument buffers for "
        << "the kernel. Please file an issue with the function that results "
        << "in this error. The name of the kernel is '" << kernel_name << "'";
    throw std::runtime_error(msg.str());
  }
}

// The element extent ``count`` must lie inside the operand's allocation.
void compiled_require_extent(const array& x, uint64_t count) {
  if (x.offset() < 0 || uint64_t(x.offset()) > x.buffer_size() ||
      count > compiled_max_elements ||
      count > (x.buffer_size() - uint64_t(x.offset())) / x.itemsize()) {
    throw std::invalid_argument("CrossTL compiled operand exceeds its allocation.");
  }
}

// A non-negative strided view's element span from its data pointer.
uint64_t compiled_span(const array& x, const Shape& shape, const Strides& strides) {
  uint64_t span = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    if (strides[axis] < 0 || uint64_t(strides[axis]) > compiled_max_elements) {
      throw std::invalid_argument(
          "CrossTL compiled kernels require non-negative input strides.");
    }
    span += (uint64_t(shape[axis]) - 1) * uint64_t(strides[axis]);
  }
  compiled_require_extent(x, span);
  return span;
}

void launch_compiled(
    const std::string& entry,
    const std::string& source,
    std::vector<CrosstlMlxBuffer>& buffers,
    uint64_t threads,
    const CrosstlMlxLaunch& launch) {
  buffers.push_back(
      {"kernel_source", "utf8", const_cast<char*>(source.data()), source.size(), 0});
  char error[2048] = {};
  int status = dispatch_callback.load()(
      entry.c_str(), buffers.data(), static_cast<uint32_t>(buffers.size()),
      threads, &launch, error, sizeof(error));
  error[sizeof(error) - 1] = '\0';
  buffers.pop_back();
  if (status != 0) {
    throw std::runtime_error(std::string("CrossTL native compiled kernel failed: ") + error);
  }
}

} // namespace

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  require_runtime();
  for (const auto& x : inputs) {
    if (!cast_type(x.dtype())) {
      throw std::invalid_argument("CrossTL compiled kernels do not support this dtype.");
    }
  }
  for (const auto& x : outputs) {
    if (!cast_type(x.dtype())) {
      throw std::invalid_argument("CrossTL compiled kernels do not support this dtype.");
    }
  }
  auto [contiguous, negative_strides, shape, strides] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);
  if (negative_strides || compiled_use_large_index(inputs, outputs, contiguous)) {
    // MLX's large variants index with int64_t from the data pointer, which a
    // host upload starting at that pointer cannot hold for negative offsets.
    throw std::invalid_argument(
        "CrossTL compiled kernels require non-negative strides and 32-bit indices.");
  }
  const int ndim = shape.size();
  const bool dynamic = ndim >= 8;
  // The variant eval_gpu selects, with the build_kernel arguments
  // Compiled::eval_gpu's library builder passes for it.
  std::string suffix = contiguous ? "_contiguous" : "_strided_";
  int work_per_thread = 1;
  if (!contiguous) {
    suffix += dynamic ? "dynamic" : std::to_string(ndim);
    work_per_thread = dynamic ? 2 : ndim > 3 ? 2 : 1;
  } else {
    const auto data_size = outputs[0].data_size();
    work_per_thread = data_size < (size_t(1) << 16)
        ? 1
        : compiled_work_per_thread(outputs[0].dtype());
    if (work_per_thread > 1) {
      suffix += "_n";
    }
  }
  const std::string entry = compiled_kernel_name(kernel_lib_) + suffix;
  std::string source = compiled_preamble;
  build_compiled_kernel(
      source,
      entry,
      inputs_,
      outputs_,
      tape_,
      is_constant_,
      contiguous,
      contiguous || dynamic ? 0 : ndim,
      dynamic,
      /* use_big_index = */ false,
      work_per_thread);

  compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);
  const uint64_t total = contiguous ? outputs[0].data_size() : outputs[0].size();
  if (total == 0) {
    return;
  }
  if (total > compiled_max_elements) {
    throw std::invalid_argument("CrossTL compiled kernels require 32-bit element counts.");
  }

  // The kernel's parameter names: build_kernel names the non-constant inputs,
  // then the outputs, then the constants.
  NodeNamer namer;
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (!is_constant_(i)) {
      namer.get_name(inputs_[i]);
    }
  }
  for (const auto& x : outputs_) {
    namer.get_name(x);
  }
  if (contiguous) {
    // Launch at most 65535 threads at a time; every vector input and output
    // shares the output's linear layout, so each launch offsets them alike.
    const uint64_t per_launch = compiled_max_groups * work_per_thread;
    for (uint64_t first = 0; first < total; first += per_launch) {
      uint32_t size = static_cast<uint32_t>(std::min(per_launch, total - first));
      std::vector<CrosstlMlxBuffer> buffers;
      for (size_t i = 0; i < inputs.size(); ++i) {
        if (is_constant_(i)) {
          continue;
        }
        const auto& x = inputs[i];
        const bool scalar = is_scalar(x);
        if (!scalar && x.data_size() != total) {
          throw std::invalid_argument("CrossTL compiled vector does not match its output.");
        }
        compiled_require_extent(x, scalar ? 1 : total);
        buffers.push_back(
            {namer.get_name(inputs_[i]).c_str(), cast_type(x.dtype()),
             const_cast<char*>(x.data<char>()) + (scalar ? 0 : first * x.itemsize()),
             scalar ? 1 : size, 0});
      }
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto& x = outputs[i];
        compiled_require_extent(x, total);
        buffers.push_back(
            {namer.get_name(outputs_[i]).c_str(), cast_type(x.dtype()),
             x.data<char>() + first * x.itemsize(), size, 1});
      }
      buffers.push_back({"size", "uint32", &size, 1, 0});
      const uint64_t threads = (uint64_t(size) + work_per_thread - 1) / work_per_thread;
      const CrosstlMlxLaunch launch{{static_cast<uint32_t>(threads), 1, 1}, {1, 1, 1}};
      launch_compiled(entry, source, buffers, threads, launch);
    }
    return;
  }

  // compiled.cpp launches (ceil(dim0 / N), dim1, rest) threads over the
  // collapsed shape; the output index assumes that whole grid in one launch.
  uint64_t dim0 = ndim > 0 ? shape[ndim - 1] : 1;
  uint64_t dim1 = ndim > 1 ? shape[ndim - 2] : 1;
  uint64_t rest = outputs[0].size() / (dim0 * dim1);
  dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
  if (dim0 > compiled_max_groups || dim1 > compiled_max_groups ||
      rest > compiled_max_groups) {
    throw std::invalid_argument(
        "CrossTL compiled kernels launch at most 65535 groups per axis.");
  }
  std::vector<CrosstlMlxBuffer> buffers;
  std::vector<int64_t> in_strides;
  int stride_index = 1; // strides[0] is the output's.
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) {
      continue;
    }
    const auto& x = inputs[i];
    uint64_t span = 1;
    if (!is_scalar(x)) {
      span = compiled_span(x, shape, strides[stride_index]);
      in_strides.insert(
          in_strides.end(), strides[stride_index].begin(), strides[stride_index].end());
      stride_index++;
    } else {
      compiled_require_extent(x, 1);
    }
    buffers.push_back(
        {namer.get_name(inputs_[i]).c_str(), cast_type(x.dtype()),
         const_cast<char*>(x.data<char>()), span, 0});
  }
  if (!in_strides.empty()) {
    buffers.push_back(
        {"in_strides", "int64", in_strides.data(), in_strides.size(), 0});
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto& x = outputs[i];
    compiled_require_extent(x, x.size());
    buffers.push_back(
        {namer.get_name(outputs_[i]).c_str(), cast_type(x.dtype()),
         x.data<char>(), x.size(), 1});
  }
  std::vector<int32_t> output_shape(shape.begin(), shape.end());
  buffers.push_back(
      {"output_shape", "int32", output_shape.data(), output_shape.size(), 0});
  int32_t dimensions = ndim;
  if (dynamic) {
    buffers.push_back({"ndim", "int32", &dimensions, 1, 0});
  }
  const CrosstlMlxLaunch launch{
      {static_cast<uint32_t>(dim0), static_cast<uint32_t>(dim1),
       static_cast<uint32_t>(rest)},
      {1, 1, 1}};
  launch_compiled(entry, source, buffers, dim0 * dim1 * rest, launch);
}

} // namespace mlx::core
