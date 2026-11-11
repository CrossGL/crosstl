"""Prepare an explicit synchronous MLX backend backed by CrossTL dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from demos.integrations.mlx.portable_host.packages import (
    BINARY_OPERATIONS,
    COMPARISON_OPERATIONS,
    LOGICAL_OPERATIONS,
    UNARY_OPERATIONS,
)

COMMIT = "9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8"
HERE = Path(__file__).resolve().parent
VIEW_PRIMITIVES = (
    "AsStrided",
    "Broadcast",
    "BroadcastAxes",
    "Copy",
    "CustomTransforms",
    "Depends",
    "ExpandDims",
    "Reshape",
    "Slice",
    "Split",
    "Squeeze",
    "StopGradient",
    "Transpose",
    "Unflatten",
)
MULTI_OUTPUT_VIEWS = {"CustomTransforms", "Depends", "Split"}
# Primitives that MLX's GPU backend (mlx/backend/gpu/primitives.cpp) evaluates
# with the shared host eval instead of a kernel, such as NumberOfElements,
# which shapeless compiled graphs use to count reduced elements.
HOST_EVAL_PRIMITIVES = ("NumberOfElements",)
COPY_PRIMITIVES = ("Contiguous", "Flatten", "Full", "Pad", "SliceUpdate", "View")
# Slices whose start indices are arrays: a translated offset kernel computes
# the start, and the dynamic copy kernels move the elements.
DYNAMIC_SLICE_PRIMITIVES = ("DynamicSlice", "DynamicSliceUpdate")
CAST_PRIMITIVES = ("AsType",)
# Unary primitives MLX instantiates only for complex64 operands.
COMPLEX_PRIMITIVES = ("Conjugate", "Real", "Imag")
# Fast primitives normalization.inc and cross_entropy.inc port; the forward
# ones keep MLX's CPU fallback, and every one of them is a multi-output
# primitive.
FALLBACK_PRIMITIVES = ("LayerNorm", "RMSNorm", "CrossEntropy")
NORMALIZATION_PRIMITIVES = ("LayerNorm", "RMSNorm", "LayerNormVJP", "RMSNormVJP")
CROSS_ENTROPY_PRIMITIVES = ("CrossEntropy", "CrossEntropyVJP")
# gated_delta.inc ports the recurrent gated-delta update and its backward,
# whose use_fallback no_gpu defines outright (GATED_DELTA_DEFINITIONS).
GATED_DELTA_PRIMITIVES = ("GatedDeltaUpdate", "GatedDeltaUpdateVJP")
# fast::ConvertFP8 launches the unary ToFP8 and FromFP8 vectors; sdpa.inc
# ports the forward attention and its fused VJP, whose use_fallback and
# supports_bool_mask no_gpu defines outright (ATTENTION_DEFINITIONS).
FAST_PRIMITIVES = (
    *NORMALIZATION_PRIMITIVES,
    *CROSS_ENTROPY_PRIMITIVES,
    "ConvertFP8",
    "ScaledDotProductAttention",
    "ScaledDotProductAttentionVJP",
    *GATED_DELTA_PRIMITIVES,
    # mx.fast.metal_kernel's kernels, which custom_kernel.inc launches.
    "CustomKernel",
)
# mx.fast.metal_kernel refuses every stream without a Metal back end. The host
# builds an adapted copy of metal_kernel.cpp in its place that also accepts
# the GPU the registered runtime provides, which executes the kernel's Metal
# source translated (custom_kernel.inc); the rest of the file is unchanged.
METAL_KERNEL_SOURCE = "mlx/backend/common/metal_kernel.cpp"
METAL_KERNEL_STREAM = (
    "  if (metal::is_available()) {\n"
    "    // Default an unspecified stream to the GPU rather than the global default\n"
)
METAL_KERNEL_HOST_STREAM = (
    "  if (metal::is_available() || is_available(Device::gpu)) {\n"
    "    // Default an unspecified stream to the GPU rather than the global default\n"
)
# no_gpu's attention definitions that sdpa.inc replaces with
# scaled_dot_product_attention.cpp's.
ATTENTION_DEFINITIONS = (
    """bool fast::ScaledDotProductAttention::use_fallback(
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
  if (force_fused) {
    throw std::invalid_argument(
        "[scaled_dot_product_attention] force_fused=True but no fused "
        "kernel is available in CPU backend.");
  }
  return true;
}
""",
    """bool fast::ScaledDotProductAttention::supports_bool_mask() {
  return false;
}
""",
    """bool fast::ScaledDotProductAttentionVJP::use_fallback(
    const array& q,
    Stream s) {
  return true;
}
""",
)
# no_gpu's gated-delta definitions that gated_delta.inc replaces with
# gated_delta_update.cpp's.
GATED_DELTA_DEFINITIONS = (
    """bool fast::GatedDeltaUpdate::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    const bool has_mask,
    Stream s) {
  return true;
}
""",
    """bool fast::GatedDeltaUpdateVJP::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    Stream s) {
  return true;
}
""",
)


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Unexpected upstream source: {before!r}")
    return text.replace(before, after, 1)


def adapted_sources(original, primitives, events, metal_kernel):
    """Render the complete adaptation without changing the source tree.

    ``original``, ``primitives`` and ``events`` are no_gpu's CMakeLists.txt,
    primitives.cpp and event.cpp, and ``metal_kernel`` is metal_kernel.cpp.
    """
    for primitive in (
        "Arange",
        "ArgReduce",
        "LogSumExp",
        "Softmax",
        "ArgPartition",
        "ArgSort",
        "Partition",
        "Sort",
        "SearchSorted",
        *FAST_PRIMITIVES,
        "RandomBits",
        "Scan",
        "Quantize",
        "QuantizedMatmul",
        "GatherQMM",
        "Matmul",
        "AddMM",
        "GatherMM",
        "SegmentedMM",
        "BlockMaskedMM",
        "Convolution",
        "FFT",
        "Hadamard",
        "Reduce",
        "BitwiseBinary",
        "BitwiseInvert",
        "Concatenate",
        "Select",
        "Gather",
        "GatherAxis",
        "ScatterAxis",
        "Scatter",
        "MaskedScatter",
        "DivMod",
        "Compiled",
        *UNARY_OPERATIONS,
        *VIEW_PRIMITIVES,
        *HOST_EVAL_PRIMITIVES,
        *COPY_PRIMITIVES,
        *DYNAMIC_SLICE_PRIMITIVES,
        *BINARY_OPERATIONS,
        "Power",
        "ArcTan2",
        *CAST_PRIMITIVES,
        *COMPLEX_PRIMITIVES,
        *COMPARISON_OPERATIONS,
        *LOGICAL_OPERATIONS,
        "LogicalNot",
    ):
        if primitive in {"Log2", "Log10", "Rsqrt"}:
            continue
        macro = (
            "NO_GPU_USE_FALLBACK"
            if primitive in FALLBACK_PRIMITIVES
            else (
                "NO_GPU_MULTI"
                if primitive in MULTI_OUTPUT_VIEWS
                or primitive in {"Compiled", "DivMod", "Quantize", *FAST_PRIMITIVES}
                else "NO_GPU"
            )
        )
        primitives = replace_once(
            primitives,
            f"{macro}({primitive})",
            f"// {primitive} is implemented by the registered native dispatch backend.",
        )
    for definition in ATTENTION_DEFINITIONS:
        primitives = replace_once(
            primitives,
            definition,
            "// sdpa.inc ports this ScaledDotProductAttention member.\n",
        )
    for definition in GATED_DELTA_DEFINITIONS:
        primitives = replace_once(
            primitives,
            definition,
            "// gated_delta.inc ports this gated-delta member.\n",
        )
    events = replace_once(
        events,
        "void Event::wait(Stream stream) {",
        """void Event::wait(Stream stream) {
  if (stream.device == Device::gpu) {
    wait();
    return;
  }""",
    )
    events = replace_once(
        events,
        "void Event::signal(Stream stream) {",
        """void Event::signal(Stream stream) {
  if (stream.device == Device::gpu) {
    auto& ec = cast<EventCounter>();
    {
      std::lock_guard lock(ec.mtx);
      ec.value = value();
    }
    ec.cv.notify_all();
    return;
  }""",
    )
    metal_kernel = replace_once(
        metal_kernel, METAL_KERNEL_STREAM, METAL_KERNEL_HOST_STREAM
    )
    # mlx/CMakeLists.txt adds backend/common before backend/no_gpu, so the
    # adapted metal_kernel.cpp can take the original's place in mlx's sources.
    cmake = """option(MLX_CROSTL_HOST "Use the CrossTL synchronous host backend" OFF)
if(MLX_CROSTL_HOST)
  get_filename_component(
    CROSTL_METAL_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/../common/metal_kernel.cpp
    ABSOLUTE)
  get_target_property(CROSTL_MLX_SOURCES mlx SOURCES)
  list(FIND CROSTL_MLX_SOURCES ${CROSTL_METAL_KERNEL} CROSTL_METAL_KERNEL_INDEX)
  if(CROSTL_METAL_KERNEL_INDEX EQUAL -1)
    message(FATAL_ERROR "The CrossTL host cannot find ${CROSTL_METAL_KERNEL}")
  endif()
  list(REMOVE_ITEM CROSTL_MLX_SOURCES ${CROSTL_METAL_KERNEL})
  set_property(TARGET mlx PROPERTY SOURCES ${CROSTL_MLX_SOURCES})
  target_sources(mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_event.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_metal_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/crosstl_primitives.cpp)
else()
""" + original + "\nendif()\n"
    return {
        "CMakeLists.txt": cmake.encode("utf-8"),
        "crosstl_primitives.cpp": primitives.encode("utf-8"),
        "crosstl_event.cpp": events.encode("utf-8"),
        "crosstl_metal_kernel.cpp": metal_kernel.encode("utf-8"),
        "crosstl_backend.cpp": (HERE / "backend.cpp").read_bytes(),
        "crosstl_dispatch.h": (HERE / "dispatch.h").read_bytes(),
        "crosstl_normalization.inc": (HERE / "normalization.inc").read_bytes(),
        "crosstl_compiled.inc": (HERE / "compiled.inc").read_bytes(),
        "crosstl_gather_mm.inc": (HERE / "gather_mm.inc").read_bytes(),
        "crosstl_gather_qmm.inc": (HERE / "gather_qmm.inc").read_bytes(),
        "crosstl_segmented_mm.inc": (HERE / "segmented_mm.inc").read_bytes(),
        "crosstl_block_masked_mm.inc": (HERE / "block_masked_mm.inc").read_bytes(),
        "crosstl_scatter.inc": (HERE / "scatter.inc").read_bytes(),
        "crosstl_conv.inc": (HERE / "conv.inc").read_bytes(),
        "crosstl_cross_entropy.inc": (HERE / "cross_entropy.inc").read_bytes(),
        "crosstl_fft.inc": (HERE / "fft.inc").read_bytes(),
        "crosstl_sdpa.inc": (HERE / "sdpa.inc").read_bytes(),
        "crosstl_hadamard.inc": (HERE / "hadamard.inc").read_bytes(),
        "crosstl_gated_delta.inc": (HERE / "gated_delta.inc").read_bytes(),
        "crosstl_custom_kernel.inc": (HERE / "custom_kernel.inc").read_bytes(),
    }


def require_revision(root):
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=30
    ).strip()
    if head != COMMIT:
        raise ValueError(f"Expected MLX {COMMIT}, got {head}")


def source_record(sources):
    return {
        "commit": COMMIT,
        "files": {
            f"mlx/backend/no_gpu/{name}": hashlib.sha256(data).hexdigest()
            for name, data in sources.items()
        },
    }


def verify_prepared(root):
    """Require exactly the current adapter and otherwise unchanged tracked sources."""
    root = Path(root).resolve()
    require_revision(root)
    changed = (
        subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only", "-z", "HEAD", "--"],
            timeout=30,
        )
        .decode("utf-8")
        .split("\0")
    )
    if set(changed) - {"", "mlx/backend/no_gpu/CMakeLists.txt"}:
        raise ValueError("Unrelated tracked MLX sources were modified")
    originals = [
        subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:{path}"],
            timeout=30,
        ).decode("utf-8")
        for path in (
            *(
                f"mlx/backend/no_gpu/{name}"
                for name in ("CMakeLists.txt", "primitives.cpp", "event.cpp")
            ),
            METAL_KERNEL_SOURCE,
        )
    ]
    sources = adapted_sources(*originals)
    for name, expected in sources.items():
        path = root / "mlx/backend/no_gpu" / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"Prepared MLX source does not match: {name}")
    return source_record(sources)


def prepare(root, output):
    root = Path(root).resolve()
    require_revision(root)
    modified = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True, timeout=30
    )
    if modified.strip():
        raise ValueError("Preparation requires a clean MLX checkout")
    backend = root / "mlx/backend/no_gpu"
    sources = adapted_sources(
        *(
            (backend / name).read_text(encoding="utf-8")
            for name in ("CMakeLists.txt", "primitives.cpp", "event.cpp")
        ),
        (root / METAL_KERNEL_SOURCE).read_text(encoding="utf-8"),
    )
    generated = [backend / name for name in sources if name != "CMakeLists.txt"]
    if any(path.exists() or path.is_symlink() for path in generated):
        raise ValueError("Preparation cannot overwrite an existing backend file")
    if Path(output).exists() or Path(output).is_symlink():
        raise ValueError("Preparation evidence already exists")
    for name, data in sources.items():
        (backend / name).write_bytes(data)
    record = source_record(sources)
    Path(output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.mlx_root, args.output)
