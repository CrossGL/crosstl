"""Bind the explicit MLX callback ABI to CrossTL native package execution."""

from __future__ import annotations

import ctypes
import json
import math
import shutil
import sys
from pathlib import Path

from crosstl.project import build_native_loader_dispatch_request
from crosstl.project.metal_runtime import MetalComputeRuntime
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from crosstl.project.runtime_values import exact_types_within
from crosstl.project.runtime_verification import (
    DirectXRuntimeParityAdapter,
    MetalRuntimeParityAdapter,
    OpenGLRuntimeParityAdapter,
    RuntimeParityExecutor,
    RuntimeTestAdapterSpec,
)
from crosstl.translator.resource_storage import encoded_storage_dtype
from demos.integrations.mlx.portable_host import (
    arange_dispatch,
    arange_packages,
    arg_reduce_dispatch,
    arg_reduce_layout,
    bfloat_storage,
    block_masked_mm_dispatch,
    block_masked_mm_layout,
    column_reduction_layout,
    column_small_layout,
    compiled_dispatch,
    compiled_layout,
    conv_dispatch,
    copy_dispatch,
    copy_layout,
    cross_entropy_dispatch,
    cross_entropy_layout,
    divmod_dispatch,
    dot_dispatch,
    dot_layout,
    dynamic_offset_dispatch,
    dynamic_offset_packages,
    element_storage,
    elementwise_dispatch,
    fft_dispatch,
    fft_packages,
    fp_matmul_layout,
    fp_qmm_layout,
    gather_dispatch,
    gather_mm_dispatch,
    gemm_dispatch,
    gemm_layout,
    gemm_splitk_dispatch,
    gemm_splitk_layout,
    gemv_dispatch,
    gemv_layout,
    half_storage,
    masked_scatter_dispatch,
    masked_scatter_packages,
    matmul_dispatch,
    matmul_layout,
    norm_dispatch,
    norm_layout,
    qmm_layout,
    quantization_dispatch,
    quantization_layout,
    random_dispatch,
    reduction_layout,
    region_dispatch,
    row_reduction_layout,
    scan_dispatch,
    scan_layout,
    sdpa_dispatch,
    sdpa_packages,
    sdpa_vjp_dispatch,
    sdpa_vjp_packages,
    searchsorted_dispatch,
    searchsorted_layout,
    slice_update_layout,
    softmax_dispatch,
    softmax_layout,
    sort_dispatch,
    sort_layout,
    steel_attention_dispatch,
    steel_attention_packages,
    wide_column_dispatch,
    wide_reduction_dispatch,
    wide_reduction_packages,
    wide_row_dispatch,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    CACHE_DIRECTORY as ARG_REDUCE_CACHE,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    ArgReducePackageCache,
)
from demos.integrations.mlx.portable_host.block_masked_mm_packages import (
    MASKED_MM_CACHE,
    MASKED_MV_CACHE,
    MaskedMmPackageCache,
    MaskedMvPackageCache,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    CACHE_DIRECTORY as SMALL_COLUMN_CACHE,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    SmallColumnPackageCache,
)
from demos.integrations.mlx.portable_host.compiled_packages import (
    CACHE_DIRECTORY as COMPILED_CACHE,
)
from demos.integrations.mlx.portable_host.compiled_packages import CompiledPackageCache
from demos.integrations.mlx.portable_host.conv_packages import (
    CACHE_DIRECTORY as CONV_CACHE,
)
from demos.integrations.mlx.portable_host.conv_packages import ConvPackageCache
from demos.integrations.mlx.portable_host.copy_packages import CACHE as COPY_CACHE
from demos.integrations.mlx.portable_host.copy_packages import (
    ENTRIES as LAZY_COPY_ENTRIES,
)
from demos.integrations.mlx.portable_host.copy_packages import CopyPackageCache
from demos.integrations.mlx.portable_host.cross_entropy_packages import (
    CACHE_DIRECTORY as CROSS_ENTROPY_CACHE,
)
from demos.integrations.mlx.portable_host.cross_entropy_packages import (
    CrossEntropyPackageCache,
)
from demos.integrations.mlx.portable_host.divmod_packages import CACHE as DIVMOD_CACHE
from demos.integrations.mlx.portable_host.divmod_packages import DivModPackageCache
from demos.integrations.mlx.portable_host.divmod_packages import (
    signature as divmod_signature,
)
from demos.integrations.mlx.portable_host.dot_packages import (
    CACHE_DIRECTORY as DOT_CACHE,
)
from demos.integrations.mlx.portable_host.dot_packages import DotPackageCache
from demos.integrations.mlx.portable_host.elementwise_packages import (
    CACHE as ELEMENTWISE_CACHE,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ENTRIES as LAZY_ELEMENTWISE_ENTRIES,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ElementwisePackageCache,
)
from demos.integrations.mlx.portable_host.fp_matmul_packages import (
    CACHE_DIRECTORY as FP_QUANTIZED_MATMUL_CACHE,
)
from demos.integrations.mlx.portable_host.fp_matmul_packages import (
    FpQuantizedMatmulPackageCache,
)
from demos.integrations.mlx.portable_host.fp_qmm_packages import (
    CACHE_DIRECTORY as FP_QUANTIZED_QMM_CACHE,
)
from demos.integrations.mlx.portable_host.fp_qmm_packages import (
    FpQuantizedQmmPackageCache,
)
from demos.integrations.mlx.portable_host.gather_mm_packages import (
    GATHER_MM_CACHE,
    GATHER_MV_CACHE,
    GATHER_OFFSETS_CACHE,
    GatherMmPackageCache,
    GatherMvPackageCache,
    GatherOffsetsPackageCache,
)
from demos.integrations.mlx.portable_host.gather_packages import (
    GatherPackageCache,
)
from demos.integrations.mlx.portable_host.gather_packages import (
    signature as gather_signature,
)
from demos.integrations.mlx.portable_host.gemm_packages import (
    CACHE_DIRECTORY as GEMM_CACHE,
)
from demos.integrations.mlx.portable_host.gemm_packages import GemmPackageCache
from demos.integrations.mlx.portable_host.gemm_splitk_packages import (
    CACHE_DIRECTORY as SPLITK_GEMM_CACHE,
)
from demos.integrations.mlx.portable_host.gemm_splitk_packages import SplitKPackageCache
from demos.integrations.mlx.portable_host.gemv_packages import (
    CACHE_DIRECTORY as GEMV_CACHE,
)
from demos.integrations.mlx.portable_host.gemv_packages import GemvPackageCache
from demos.integrations.mlx.portable_host.matmul_packages import (
    CACHE_DIRECTORY as QUANTIZED_MATMUL_CACHE,
)
from demos.integrations.mlx.portable_host.matmul_packages import (
    QuantizedMatmulPackageCache,
)
from demos.integrations.mlx.portable_host.norm_packages import (
    CACHE_DIRECTORY as NORM_CACHE,
)
from demos.integrations.mlx.portable_host.norm_packages import NormPackageCache
from demos.integrations.mlx.portable_host.packages import (
    ABSOLUTE_ENTRIES,
    ARANGE_ENTRIES,
    BFLOAT_ABSOLUTE_ENTRIES,
    BFLOAT_BINARY_ENTRIES,
    BFLOAT_CAST_ENTRIES,
    BFLOAT_COMPARISON_ENTRIES,
    BFLOAT_COPY_ENTRY,
    BFLOAT_ENTRIES,
    BINARY_ENTRIES,
    BITWISE_ENTRIES,
    BITWISE_INVERT_ENTRIES,
    BITWISE_PACKAGE_ENTRIES,
    BOOLEAN_CAST_ENTRIES,
    BOOLEAN_COPY_ENTRY,
    CAST_ENTRIES,
    COMPARISON_ENTRIES,
    COPY_ENTRY,
    ENTRIES,
    HALF_ABSOLUTE_ENTRIES,
    HALF_ARITHMETIC_ENTRIES,
    HALF_BINARY_ENTRIES,
    HALF_CAST_ENTRIES,
    HALF_COMPARISON_ENTRIES,
    HALF_COPY_ENTRY,
    HALF_ENTRIES,
    INTEGER64_ABSOLUTE_ENTRIES,
    INTEGER64_BINARY_ENTRIES,
    INTEGER64_CAST_ENTRIES,
    INTEGER64_COMPARISON_ENTRIES,
    INTEGER64_COPY_ENTRIES,
    INTEGER64_ENTRIES,
    LOGICAL_NOT_ENTRY,
    SELECTION_ENTRIES,
    SLICE_UPDATE_ENTRIES,
    UNARY_ENTRIES,
)
from demos.integrations.mlx.portable_host.qmm_packages import (
    CACHE_DIRECTORY as QUANTIZED_QMM_CACHE,
)
from demos.integrations.mlx.portable_host.qmm_packages import QuantizedQmmPackageCache
from demos.integrations.mlx.portable_host.quantization_packages import (
    QuantizationPackageCache,
)
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES as RANDOM_ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import RandomPackageCache
from demos.integrations.mlx.portable_host.random_packages import (
    load_index as load_random_index,
)
from demos.integrations.mlx.portable_host.reduction_cache import ReductionPackageCache
from demos.integrations.mlx.portable_host.reduction_packages import (
    ALL_ENTRIES as REDUCTION_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    ALL_ROW_ENTRIES,
    COLUMN_ENTRIES,
    HALF_COLUMN_ENTRIES,
    INIT_ENTRIES,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reduction_index,
)
from demos.integrations.mlx.portable_host.reduction_packages import variant_keys
from demos.integrations.mlx.portable_host.scan_packages import ScanPackageCache
from demos.integrations.mlx.portable_host.searchsorted_packages import (
    CACHE_DIRECTORY as SEARCHSORTED_CACHE,
)
from demos.integrations.mlx.portable_host.searchsorted_packages import (
    SearchSortedPackageCache,
)
from demos.integrations.mlx.portable_host.segmented_mm_packages import (
    SEGMENTED_MM_CACHE,
    SegmentedMmPackageCache,
)
from demos.integrations.mlx.portable_host.small_row_packages import (
    SMALL_ROW_ENTRIES,
    SmallRowPackageCache,
)
from demos.integrations.mlx.portable_host.softmax_packages import (
    CACHE_DIRECTORY as SOFTMAX_CACHE,
)
from demos.integrations.mlx.portable_host.softmax_packages import SoftmaxPackageCache
from demos.integrations.mlx.portable_host.sort_packages import (
    CACHE_DIRECTORY as SORT_CACHE,
)
from demos.integrations.mlx.portable_host.sort_packages import SortPackageCache
from demos.integrations.mlx.portable_host.trace_records import write_trace


class Buffer(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("dtype", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("count", ctypes.c_uint64),
        ("output", ctypes.c_uint32),
    ]


DISPATCH_VERSION = 3
# The bytes one Metal dispatch may bind: two 128 MiB passes of MLX's largest
# four-step FFT, with their guards, fit.
METAL_BUFFER_BYTES = 1 << 30


class Launch(ctypes.Structure):
    _fields_ = [
        ("workgroup_count", ctypes.c_uint32 * 3),
        ("workgroup_size", ctypes.c_uint32 * 3),
        ("thread_grid_size", ctypes.c_uint32 * 3),
    ]

    def execution(self):
        count, size = list(self.workgroup_count), list(self.workgroup_size)
        if (
            any(not 1 <= value <= 65535 for value in count)
            or any(not 1 <= value <= 1024 for value in size)
            or math.prod(size) > 1024
        ):
            raise ValueError("Native launch geometry exceeds its bounds")
        execution = {"workgroupCount": count, "workgroupSize": size}
        exact = list(self.thread_grid_size)
        if any(exact):
            if any(value < 1 for value in exact) or any(
                (extent + width - 1) // width != groups
                for extent, width, groups in zip(exact, size, count)
            ):
                raise ValueError("Native exact grid does not match its covering groups")
            execution["threadGridSize"] = exact
        return execution


CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(Buffer),
    ctypes.c_uint32,
    ctypes.c_uint64,
    ctypes.POINTER(Launch),
    ctypes.c_void_p,
    ctypes.c_size_t,
)
ENTRY_AVAILABLE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)
TYPES = {
    "bool_": ctypes.c_uint8,
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "uint16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint32": ctypes.c_uint32,
    "int64": ctypes.c_int64,
    "uint64": ctypes.c_uint64,
}
_installed_runtime = None
INT_TYPES = frozenset({int})
BOOL_TYPES = frozenset({bool})
INTEGRAL_TYPES = frozenset({bool, int})
FLOAT_TYPES = frozenset({float})
_FLOAT_CTYPES = frozenset({ctypes.c_float, ctypes.c_double})
_INTEGRAL_CTYPES = frozenset(
    {
        ctypes.c_bool,
        ctypes.c_int8,
        ctypes.c_uint8,
        ctypes.c_int16,
        ctypes.c_uint16,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_int64,
        ctypes.c_uint64,
    }
)
COPY_GUARD = [0x6A15BEEF] * 32
BOOLEAN_GUARD = [index % 2 == 0 for index in range(32)]
ALL_CAST_ENTRIES = {
    **CAST_ENTRIES,
    **BOOLEAN_CAST_ENTRIES,
    **INTEGER64_CAST_ENTRIES,
    **HALF_CAST_ENTRIES,
    **BFLOAT_CAST_ENTRIES,
}
ALL_BINARY_ENTRIES = {
    **BINARY_ENTRIES,
    **BITWISE_ENTRIES,
    **INTEGER64_BINARY_ENTRIES,
    **HALF_BINARY_ENTRIES,
    **BFLOAT_BINARY_ENTRIES,
}
ALL_COMPARISON_ENTRIES = {
    **COMPARISON_ENTRIES,
    **INTEGER64_COMPARISON_ENTRIES,
    **HALF_COMPARISON_ENTRIES,
    **BFLOAT_COMPARISON_ENTRIES,
}
ALL_ABSOLUTE_ENTRIES = {
    **ABSOLUTE_ENTRIES,
    **INTEGER64_ABSOLUTE_ENTRIES,
    **HALF_ABSOLUTE_ENTRIES,
    **BFLOAT_ABSOLUTE_ENTRIES,
}
ALL_HALF_ENTRIES = (*HALF_ENTRIES, *HALF_ARITHMETIC_ENTRIES)
ALL_COPY_ENTRIES = {
    COPY_ENTRY: "uint32",
    BOOLEAN_COPY_ENTRY: "bool_",
    HALF_COPY_ENTRY: "float16",
    BFLOAT_COPY_ENTRY: "bfloat16",
    **INTEGER64_COPY_ENTRIES,
}


def physical_dtype(dtype, target):
    if dtype == "bfloat16":
        return {"metal": "bfloat16", "directx": "uint16", "opengl": "float32"}[target]
    if dtype == "float16" and target == "opengl":
        return "float32"
    if dtype == "bool_":
        return "bool" if target == "metal" else "uint32"
    return dtype


def boolean_values(values, dtype):
    expected_types = BOOL_TYPES if dtype == "bool" else INT_TYPES
    if not exact_types_within(values, expected_types) or (
        values and not (min(values) >= 0 and max(values) <= 1)
    ):
        raise ValueError("Boolean storage must contain canonical zero or one values")
    # Exact bools or ints already are their canonical storage values.
    return list(values)


def wire_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+infinity" if value > 0 else "-infinity"
    return value


def wire_values(values):
    """``[wire_value(value) for value in values]``, in bulk for exact scalars."""
    if isinstance(values, ctypes.Array) and values._type_ in _FLOAT_CTYPES:
        values = element_storage.array_values(values)
        # A finite sum has no NaN or infinite term; an overflowing one only
        # takes the element-wise path, which leaves finite values unchanged.
        if math.isfinite(sum(values)):
            return values
        return [wire_value(value) for value in values]
    if isinstance(values, ctypes.Array) and values._type_ in _INTEGRAL_CTYPES:
        # Integral storage reads as exact ints (or bools), already wire values.
        return element_storage.array_values(values)
    values = list(values)
    if exact_types_within(values, INTEGRAL_TYPES) or (
        exact_types_within(values, FLOAT_TYPES) and all(map(math.isfinite, values))
    ):
        return values
    return [wire_value(value) for value in values]


class HostRuntime:
    def __init__(
        self,
        directory,
        trace,
        *,
        reductions=None,
        mlx_root=None,
        bitwise=None,
        selection=None,
        absolute=None,
        integer64=None,
        slice_updates=None,
        random=None,
        half=None,
        half_arithmetic=None,
        bfloat=None,
        retain_native_modules=False,
        trace_values=True,
    ):
        if type(retain_native_modules) is not bool:
            raise ValueError("Native module retention must be a Boolean")
        if type(trace_values) is not bool and not (
            type(trace_values) is tuple
            and trace_values
            and all(type(prefix) is str and prefix for prefix in trace_values)
        ):
            raise ValueError(
                "Trace value retention must be a Boolean or entry prefixes"
            )
        self.retain_native_modules = retain_native_modules
        # False keeps each long scalar array as its length and SHA-256 only;
        # a tuple keeps the values of the entries that start with its prefixes.
        self.trace_values = trace_values
        # A callable ``(record, line) -> line`` that checks each dispatch
        # record whole before its returned line is written (write_trace).
        self.trace_audit = None
        self.directory = Path(directory).resolve()
        self.trace = Path(trace).resolve()
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.target = index["target"]
        self.small_rows = (
            SmallRowPackageCache(mlx_root, self.directory / "small-rows", self.target)
            if mlx_root is not None
            else None
        )
        self.small_columns = (
            SmallColumnPackageCache(
                mlx_root, self.directory / SMALL_COLUMN_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # Arg reductions translate each entry at its threadgroup width.
        self.arg_reductions = (
            ArgReducePackageCache(
                mlx_root, self.directory / ARG_REDUCE_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # Softmax and logsumexp rows translate each entry at its width.
        self.softmaxes = (
            SoftmaxPackageCache(mlx_root, self.directory / SOFTMAX_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # Merge-sort stages translate each entry at its launch width.
        self.sorts = (
            SortPackageCache(mlx_root, self.directory / SORT_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # Binary searches translate each entry at its pass size.
        self.searchsorted = (
            SearchSortedPackageCache(
                mlx_root, self.directory / SEARCHSORTED_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # RMS and layer normalization rows translate each entry at its width.
        self.norms = (
            NormPackageCache(mlx_root, self.directory / NORM_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # Fused cross-entropy rows translate each entry at its width.
        self.cross_entropies = (
            CrossEntropyPackageCache(
                mlx_root, self.directory / CROSS_ENTROPY_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.gathers = (
            GatherPackageCache(mlx_root, self.directory / "gather", self.target)
            if mlx_root is not None
            else None
        )
        self.quantization = (
            QuantizationPackageCache(
                mlx_root, self.directory / "quantization", self.target
            )
            if mlx_root is not None
            else None
        )
        self.quantized_matmuls = (
            QuantizedMatmulPackageCache(
                mlx_root, self.directory / QUANTIZED_MATMUL_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.fp_quantized_matmuls = (
            FpQuantizedMatmulPackageCache(
                mlx_root, self.directory / FP_QUANTIZED_MATMUL_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.quantized_qmms = (
            QuantizedQmmPackageCache(
                mlx_root, self.directory / QUANTIZED_QMM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.fp_quantized_qmms = (
            FpQuantizedQmmPackageCache(
                mlx_root, self.directory / FP_QUANTIZED_QMM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.gemvs = (
            GemvPackageCache(mlx_root, self.directory / GEMV_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        self.dots = (
            DotPackageCache(mlx_root, self.directory / DOT_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # Kernels MLX generates for fused tapes translate on first use.
        self.compiled = (
            CompiledPackageCache(mlx_root, self.directory / COMPILED_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        self.gemms = (
            GemmPackageCache(mlx_root, self.directory / GEMM_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        self.splitk_gemms = (
            SplitKPackageCache(
                mlx_root, self.directory / SPLITK_GEMM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # GatherMM's gathered gemv, steel gather and group offset kernels.
        self.gather_mvs = (
            GatherMvPackageCache(
                mlx_root, self.directory / GATHER_MV_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.gather_mms = (
            GatherMmPackageCache(
                mlx_root, self.directory / GATHER_MM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.gather_offsets = (
            GatherOffsetsPackageCache(
                mlx_root, self.directory / GATHER_OFFSETS_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # SegmentedMM's segmented steel GEMM kernels.
        self.segmented_mms = (
            SegmentedMmPackageCache(
                mlx_root, self.directory / SEGMENTED_MM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # BlockMaskedMM's masked gemv and masked steel GEMM kernels.
        self.masked_mvs = (
            MaskedMvPackageCache(
                mlx_root, self.directory / MASKED_MV_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.masked_mms = (
            MaskedMmPackageCache(
                mlx_root, self.directory / MASKED_MM_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # Convolution's unfold and separable depthwise kernels.
        self.convs = (
            ConvPackageCache(mlx_root, self.directory / CONV_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # FFT's Stockham kernels, one per fft.cpp pipeline hash name.
        self.ffts = (
            fft_packages.FftPackageCache(
                mlx_root, self.directory / fft_packages.CACHE_DIRECTORY, self.target
            )
            if mlx_root is not None
            else None
        )
        # The vector attention kernels, one per sdpa_vector pipeline hash name.
        self.sdpas = (
            sdpa_packages.SdpaPackageCache(
                mlx_root, self.directory / sdpa_packages.CACHE_DIRECTORY, self.target
            )
            if mlx_root is not None
            else None
        )
        # The full-attention kernel, one per steel_attention pipeline hash name.
        self.steel_attentions = (
            steel_attention_packages.SteelAttentionPackageCache(
                mlx_root,
                self.directory / steel_attention_packages.CACHE_DIRECTORY,
                self.target,
            )
            if mlx_root is not None
            else None
        )
        # The fused attention VJP's row products, block gradients and
        # reductions, one per pipeline name.
        self.sdpa_vjps = (
            sdpa_vjp_packages.SdpaVjpPackageCache(
                mlx_root,
                self.directory / sdpa_vjp_packages.CACHE_DIRECTORY,
                self.target,
            )
            if mlx_root is not None
            else None
        )
        self.scans = (
            ScanPackageCache(mlx_root, self.directory / "scans", self.target)
            if mlx_root is not None
            else None
        )
        self.divmods = (
            DivModPackageCache(mlx_root, self.directory / DIVMOD_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # Casts and copies that no installed family covers translate on first use.
        self.copies = (
            CopyPackageCache(mlx_root, self.directory / COPY_CACHE, self.target)
            if mlx_root is not None
            else None
        )
        # So do the unary and binary vectors of every other real dtype.
        self.elementwise = (
            ElementwisePackageCache(
                mlx_root, self.directory / ELEMENTWISE_CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # So do the aranges of the 8- and 16-bit integers and 16-bit floats.
        self.aranges = (
            arange_packages.cache(
                mlx_root, self.directory / arange_packages.CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # So do the start-offset kernels MLX generates for dynamic slices.
        self.dynamic_offsets = (
            dynamic_offset_packages.DynamicOffsetPackageCache(
                mlx_root, self.directory / dynamic_offset_packages.CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # So do the masked_assign kernels MLX generates for masked scatters.
        self.masked_scatters = (
            masked_scatter_packages.MaskedScatterPackageCache(
                mlx_root, self.directory / masked_scatter_packages.CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        # So do whole-array reductions of the 8-byte and 16-bit float types.
        self.wide_reductions = (
            wide_reduction_packages.cache(
                mlx_root, self.directory / wide_reduction_packages.CACHE, self.target
            )
            if mlx_root is not None
            else None
        )
        self.descriptors = index["descriptors"]
        if set(self.descriptors) != set(ENTRIES):
            raise ValueError("Packages must contain the exact supported entry set")
        self.random_directory = Path(random).resolve() if random is not None else None
        self.randoms = None
        if self.random_directory is not None:
            if mlx_root is None:
                raise ValueError("Random dispatch requires the pinned MLX source")
            self.descriptors.update(
                load_random_index(self.random_directory, self.target)
            )
            self.randoms = RandomPackageCache(
                mlx_root, self.random_directory / "variants", self.target
            )
        self.bitwise_directory = (
            Path(bitwise).resolve() if bitwise is not None else None
        )
        self.selection_directory = (
            Path(selection).resolve() if selection is not None else None
        )
        self.absolute_directory = (
            Path(absolute).resolve() if absolute is not None else None
        )
        self.integer64_directory = (
            Path(integer64).resolve() if integer64 is not None else None
        )
        self.slice_update_directory = (
            Path(slice_updates).resolve() if slice_updates is not None else None
        )
        self.half_directory = Path(half).resolve() if half is not None else None
        self.bfloat_directory = Path(bfloat).resolve() if bfloat is not None else None
        self.half_arithmetic_directory = (
            Path(half_arithmetic).resolve() if half_arithmetic is not None else None
        )
        for family, directory, entries in (
            ("bitwise", self.bitwise_directory, BITWISE_PACKAGE_ENTRIES),
            ("selection", self.selection_directory, SELECTION_ENTRIES),
            ("absolute", self.absolute_directory, ABSOLUTE_ENTRIES),
            ("integer64", self.integer64_directory, INTEGER64_ENTRIES),
            ("slice-update", self.slice_update_directory, SLICE_UPDATE_ENTRIES),
            ("half", self.half_directory, HALF_ENTRIES),
            ("bfloat", self.bfloat_directory, BFLOAT_ENTRIES),
            (
                "half-arithmetic",
                self.half_arithmetic_directory,
                HALF_ARITHMETIC_ENTRIES,
            ),
        ):
            if directory is None:
                continue
            operators = json.loads(
                (directory / "index.json").read_text(encoding="utf-8")
            )
            if (
                not isinstance(operators, dict)
                or operators.get("family") != family
                or operators.get("target") != self.target
                or not isinstance(operators.get("descriptors"), dict)
                or set(operators.get("descriptors", {})) != set(entries)
                or any(
                    not isinstance(value, dict) or value.get("target") != self.target
                    for value in operators["descriptors"].values()
                )
            ):
                raise ValueError(
                    f"{family.title()} packages must match the target and exact entry set"
                )
            self.descriptors.update(operators["descriptors"])
        directories = (
            [self.directory / "reductions"]
            if reductions is None
            else reductions if isinstance(reductions, (list, tuple)) else [reductions]
        )
        if not directories:
            raise ValueError("At least one reduction package directory is required")
        self.reduction_descriptors = {}
        self.reduction_directories = {}
        self.checked_reductions = {}
        self.dispatch_count = 0
        for directory in directories:
            directory = Path(directory).resolve()
            if not (directory / "index.json").is_file():
                if reductions is None:
                    continue
                raise ValueError("Reduction package index is missing")
            reduction_index = load_reduction_index(directory, self.target)
            keys = variant_keys(reduction_index)
            if keys & (
                self.reduction_descriptors.keys() | self.checked_reductions.keys()
            ):
                raise ValueError("Reduction packages contain duplicate variants")
            if reduction_index.get("dispatchMode") == "on-demand":
                if mlx_root is None:
                    raise ValueError("Checked reductions require the pinned MLX source")
                cache = ReductionPackageCache(
                    mlx_root, directory / "variants", self.target
                )
                self.checked_reductions.update({key: cache for key in keys})
                continue
            descriptors = reduction_index["descriptors"]
            self.reduction_descriptors.update(descriptors)
            self.reduction_directories.update({key: directory for key in descriptors})
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        # Each package artifact is compiled and validated once per session;
        # later dispatches of byte-identical sources reuse the validated module
        # and record the reuse on every validation step.
        if self.target == "opengl":
            adapter = OpenGLRuntimeParityAdapter(
                runtime=OpenGLComputeRuntime(context_backends=("egl",)),
                reuse_validated_modules=True,
            )
        elif self.target == "directx":
            adapter = DirectXRuntimeParityAdapter(
                runtime=DirectXComputeRuntime(), reuse_validated_modules=True
            )
        elif self.target == "metal":
            # An explicit resolver gives this session its own helper, built
            # once, instead of re-keying the shared helper on the toolchain
            # with three xcrun lookups before every dispatch. The largest
            # launches MLX makes, the two passes of a 2**24-point four-step
            # FFT, bind 128 MiB inputs and outputs, past the runtime's
            # default 256 MiB limit; MLX's own Metal allocations are bounded
            # by the device instead.
            adapter = MetalRuntimeParityAdapter(
                runtime=MetalComputeRuntime(
                    tool_resolver=shutil.which, max_buffer_bytes=METAL_BUFFER_BYTES
                ),
                reuse_validated_modules=True,
            )
        else:
            raise ValueError(f"Unsupported host target: {self.target}")
        self.executor = RuntimeParityExecutor(
            RuntimeTestAdapterSpec(
                adapter_id=f"mlx-host-{self.target}",
                target=self.target,
                executor=self.target,
                adapter_kind=f"{self.target}-native-runtime",
            ),
            runtime_adapter=adapter,
        )
        self.callback = CALLBACK(self._dispatch)
        self.entry_available = ENTRY_AVAILABLE(self._entry_available)
        self.library = None

    def _entry_available(self, entry):
        try:
            name = entry.decode("ascii")
            if name in self.descriptors:
                return 1
            if name in LAZY_COPY_ENTRIES and self.copies is not None:
                return 1
            if name in LAZY_ELEMENTWISE_ENTRIES and self.elementwise is not None:
                return 1
            if name in arange_packages.ENTRIES and self.aranges is not None:
                return 1
            if (
                name in dynamic_offset_packages.ENTRIES
                and self.dynamic_offsets is not None
            ):
                return 1
            if (
                name in masked_scatter_packages.ENTRIES
                and self.masked_scatters is not None
            ):
                return 1
            if name in arg_reduce_layout.ENTRIES and self.arg_reductions is not None:
                return 1
            if name in softmax_layout.ENTRIES and self.softmaxes is not None:
                return 1
            if name in sort_layout.ENTRIES and self.sorts is not None:
                return 1
            if name in searchsorted_layout.ENTRIES and self.searchsorted is not None:
                return 1
            if name in norm_layout.ENTRIES and self.norms is not None:
                return 1
            if (
                name in cross_entropy_layout.ENTRIES
                and self.cross_entropies is not None
            ):
                return 1
            if (
                name in wide_reduction_packages.ENTRIES
                and self.wide_reductions is not None
            ):
                return 1
            if name.startswith(scan_layout.ENTRY_PREFIXES) and self.scans is not None:
                scan_layout.signature(name)
                return 1
            if (
                name.startswith(divmod_dispatch.ENTRY_PREFIXES)
                and self.divmods is not None
            ):
                divmod_signature(name)
                return 1
            if name.startswith(block_masked_mm_dispatch.ENTRY_PREFIXES):
                kind = block_masked_mm_layout.kind_of(name)
                return int(block_masked_mm_dispatch.cache_for(self, kind) is not None)
            if conv_dispatch.is_entry(name):
                return int(self.convs is not None)
            if fft_dispatch.is_entry(name):
                return int(self.ffts is not None)
            if sdpa_dispatch.is_entry(name):
                return int(self.sdpas is not None)
            if steel_attention_dispatch.is_entry(name):
                return int(self.steel_attentions is not None)
            if sdpa_vjp_dispatch.is_entry(name):
                return int(self.sdpa_vjps is not None)
            if name.startswith(gather_mm_dispatch.ENTRY_PREFIXES):
                kind = gather_mm_dispatch.family(name)
                return int(gather_mm_dispatch.cache_for(self, kind) is not None)
            if name.startswith(("gather", "scatter")) and self.gathers is not None:
                gather_signature(name)
                return 1
            if (
                name.startswith(matmul_layout.ENTRY_PREFIXES)
                and self.quantized_matmuls is not None
            ):
                matmul_layout.signature(name)
                return 1
            if (
                name.startswith(fp_matmul_layout.ENTRY_PREFIXES)
                and self.fp_quantized_matmuls is not None
            ):
                fp_matmul_layout.signature(name)
                return 1
            if (
                name.startswith(qmm_layout.ENTRY_PREFIXES)
                and self.quantized_qmms is not None
            ):
                qmm_layout.signature(name)
                return 1
            if (
                name.startswith(fp_qmm_layout.ENTRY_PREFIXES)
                and self.fp_quantized_qmms is not None
            ):
                fp_qmm_layout.signature(name)
                return 1
            if name.startswith(gemv_layout.ENTRY_PREFIXES) and self.gemvs is not None:
                gemv_layout.signature(name)
                return 1
            if name.startswith(dot_layout.ENTRY_PREFIXES) and self.dots is not None:
                dot_layout.signature(name)
                return 1
            if (
                name.startswith(compiled_layout.ENTRY_PREFIX)
                and self.compiled is not None
            ):
                compiled_layout.variant(name)
                return 1
            if name.startswith(gemm_layout.ENTRY_PREFIXES) and self.gemms is not None:
                gemm_layout.signature(name)
                return 1
            if (
                name.startswith(gemm_splitk_layout.ENTRY_PREFIXES)
                and self.splitk_gemms is not None
            ):
                gemm_splitk_layout.signature(name)
                return 1
            if (
                name.startswith(quantization_layout.ENTRY_PREFIXES)
                and self.quantization is not None
            ):
                quantization_layout.signature(name)
                return 1
            return 0
        except (AttributeError, UnicodeDecodeError, ValueError):
            return 0

    def install(self):
        global _installed_runtime
        import mlx.core as mx

        if _installed_runtime is not None:
            raise RuntimeError("A CrossTL host runtime is already installed")
        library = (
            Path(mx.__file__).with_name("mlx.dll")
            if sys.platform == "win32"
            else Path(mx.__file__)
        )
        self.library = ctypes.CDLL(str(library))
        # mx.device_info reports the native target as the GPU architecture.
        architecture = self.library.crosstl_mlx_set_device_architecture
        architecture.argtypes = [ctypes.c_char_p]
        architecture.restype = ctypes.c_int
        result = architecture(self.target.encode("ascii"))
        if result:
            raise RuntimeError(f"MLX rejected the device architecture: {result}")
        register = self.library.crosstl_mlx_register_runtime
        register.argtypes = [ctypes.c_uint32, CALLBACK, ENTRY_AVAILABLE]
        register.restype = ctypes.c_int
        result = register(DISPATCH_VERSION, self.callback, self.entry_available)
        if result:
            raise RuntimeError(
                f"MLX rejected the native callback registration: {result}"
            )
        _installed_runtime = self
        mx.set_default_device(mx.gpu)

    def _dispatch(self, entry, buffers, count, threads, launch, error, capacity):
        try:
            if not launch:
                raise ValueError("Native launch geometry is missing")
            self.dispatch(
                entry.decode("ascii"), buffers, count, threads, launch=launch.contents
            )
            return 0
        except Exception as exception:
            message = str(exception).encode("utf-8")
            if error and capacity:
                payload = message[: capacity - 1] + b"\0"
                ctypes.memmove(error, payload, len(payload))
            return 1

    def lazy_copy(self, entry):
        """Whether ``entry`` runs from the lazy copy cache, not a package family."""
        return (
            entry in LAZY_COPY_ENTRIES
            and entry not in self.descriptors
            and self.copies is not None
        )

    def lazy_elementwise(self, entry):
        """Whether ``entry`` runs from the lazy unary or binary cache."""
        return (
            entry in LAZY_ELEMENTWISE_ENTRIES
            and entry not in self.descriptors
            and self.elementwise is not None
        )

    def lazy_arange(self, entry):
        """Whether ``entry`` runs from the lazy arange cache."""
        return (
            entry in arange_packages.ENTRIES
            and entry not in self.descriptors
            and self.aranges is not None
        )

    def lazy_wide_reduction(self, entry):
        """Whether ``entry`` runs from the wide whole-array reduction cache."""
        return (
            entry in wide_reduction_packages.ENTRIES
            and self.wide_reductions is not None
        )

    def dispatch(self, entry, buffers, count, threads, *, launch=None):
        if entry.startswith(compiled_layout.ENTRY_PREFIX):
            return compiled_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if self.lazy_wide_reduction(entry):
            return wide_reduction_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in wide_row_dispatch.ENTRIES:
            return wide_row_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in wide_column_dispatch.ENTRIES:
            return wide_column_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if self.lazy_arange(entry):
            return arange_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in dynamic_offset_packages.ENTRIES:
            return dynamic_offset_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in masked_scatter_packages.ENTRIES:
            return masked_scatter_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in arg_reduce_layout.ENTRIES:
            return arg_reduce_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in softmax_layout.ENTRIES:
            return softmax_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in sort_layout.ENTRIES:
            return sort_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry in searchsorted_layout.ENTRIES:
            return searchsorted_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in norm_layout.ENTRIES:
            return norm_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry in cross_entropy_layout.ENTRIES:
            return cross_entropy_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if self.lazy_copy(entry):
            return copy_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if self.lazy_elementwise(entry):
            return elementwise_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(scan_layout.ENTRY_PREFIXES):
            return scan_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry.startswith(divmod_dispatch.ENTRY_PREFIXES):
            return divmod_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(matmul_layout.ENTRY_PREFIXES):
            return matmul_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(fp_matmul_layout.ENTRY_PREFIXES):
            return matmul_dispatch.dispatch_with(
                self,
                fp_matmul_layout,
                self.fp_quantized_matmuls,
                entry,
                buffers,
                count,
                threads,
                launch,
            )
        if entry.startswith(qmm_layout.ENTRY_PREFIXES):
            return matmul_dispatch.dispatch_with(
                self,
                qmm_layout,
                self.quantized_qmms,
                entry,
                buffers,
                count,
                threads,
                launch,
            )
        if entry.startswith(fp_qmm_layout.ENTRY_PREFIXES):
            return matmul_dispatch.dispatch_with(
                self,
                fp_qmm_layout,
                self.fp_quantized_qmms,
                entry,
                buffers,
                count,
                threads,
                launch,
            )
        if entry.startswith(block_masked_mm_dispatch.ENTRY_PREFIXES):
            return block_masked_mm_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if conv_dispatch.is_entry(entry):
            return conv_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if fft_dispatch.is_entry(entry):
            return fft_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if sdpa_dispatch.is_entry(entry):
            return sdpa_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if steel_attention_dispatch.is_entry(entry):
            return steel_attention_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if sdpa_vjp_dispatch.is_entry(entry):
            return sdpa_vjp_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(gather_mm_dispatch.ENTRY_PREFIXES):
            return gather_mm_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(gemv_layout.ENTRY_PREFIXES):
            return gemv_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry.startswith(dot_layout.ENTRY_PREFIXES):
            return dot_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry.startswith(gemm_layout.ENTRY_PREFIXES):
            return gemm_dispatch.dispatch(self, entry, buffers, count, threads, launch)
        if entry.startswith(gemm_splitk_layout.ENTRY_PREFIXES):
            return gemm_splitk_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(quantization_layout.ENTRY_PREFIXES):
            return quantization_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry in RANDOM_ENTRIES:
            return random_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        if entry.startswith(("gather", "scatter")):
            return gather_dispatch.dispatch(
                self, entry, buffers, count, threads, launch
            )
        initialization = entry in INIT_ENTRIES
        small_row = entry in SMALL_ROW_ENTRIES
        row_reduction = entry in ALL_ROW_ENTRIES or small_row
        small_column = entry in column_small_layout.ALL_SMALL_COLUMN_ENTRIES
        column_reduction = (
            entry in COLUMN_ENTRIES or entry in HALF_COLUMN_ENTRIES or small_column
        )
        shaped_reduction = row_reduction or column_reduction
        layout_module = (
            (column_small_layout if small_column else column_reduction_layout)
            if column_reduction
            else row_reduction_layout
        )
        reduction = entry in REDUCTION_ENTRIES or shaped_reduction or initialization
        if entry not in self.descriptors and not reduction:
            raise ValueError(f"No translated package for {entry}")
        if reduction and launch is None:
            raise ValueError("Native reductions require explicit launch geometry")
        copy = entry in ALL_COPY_ENTRIES
        slice_update = entry in SLICE_UPDATE_ENTRIES
        invert = entry in BITWISE_INVERT_ENTRIES
        absolute = entry in ALL_ABSOLUTE_ENTRIES
        bitwise = entry in BITWISE_PACKAGE_ENTRIES
        binary = entry in ALL_BINARY_ENTRIES
        comparison = entry in ALL_COMPARISON_ENTRIES
        binary_operation = binary or comparison
        selection = entry in SELECTION_ENTRIES
        cast = entry in ALL_CAST_ENTRIES
        if (
            count != (
                1
                if initialization
                else (
                    len(layout_module.signature(entry))
                    if shaped_reduction
                    else (
                        8
                        if copy or slice_update
                        else (
                            5
                            if selection
                            else 4 if binary_operation or reduction else 3
                        )
                    )
                )
            )
            or not buffers
            or not 0
            < threads
            <= (
                copy_layout.MAX_DESTINATION_ELEMENTS
                if copy
                else (
                    reduction_layout.MAX_ELEMENTS
                    if entry in REDUCTION_ENTRIES
                    else (
                        column_reduction_layout.MAX_ELEMENTS
                        if column_reduction
                        else (
                            row_reduction_layout.MAX_ELEMENTS
                            if row_reduction
                            else 65535
                        )
                    )
                )
            )
        ):
            raise ValueError("Invalid or unsupported native dispatch dimensions")
        execution = launch.execution() if launch is not None else None
        if initialization and execution != {
            "workgroupCount": [threads, 1, 1],
            "workgroupSize": [1, 1, 1],
        }:
            raise ValueError(
                "Reduction initialization requires one invocation per output"
            )
        checked_reduction = None
        if reduction and not small_row and not small_column:
            key = f'w{execution["workgroupSize"][0]}/{entry}'
            checked_reduction = getattr(self, "checked_reductions", {}).get(key)
            if checked_reduction is None and key not in self.reduction_descriptors:
                raise ValueError(f"No translated reduction variant for {key}")
            if checked_reduction is None:
                descriptor = self.reduction_descriptors[key]
                package_directory = self.reduction_directories[key] / "package"
        elif not small_row and not small_column:
            descriptor = self.descriptors[entry]
            package_directory = (
                self.bfloat_directory
                if entry in BFLOAT_ENTRIES
                else (
                    (
                        self.half_directory
                        if entry in HALF_ENTRIES
                        else self.half_arithmetic_directory
                    )
                    if entry in ALL_HALF_ENTRIES
                    else (
                        self.slice_update_directory
                        if slice_update
                        else (
                            self.integer64_directory
                            if entry in INTEGER64_ENTRIES
                            else (
                                self.bitwise_directory
                                if bitwise
                                else (
                                    self.selection_directory
                                    if selection
                                    else (
                                        self.absolute_directory
                                        if absolute
                                        else self.directory
                                    )
                                )
                            )
                        )
                    )
                )
            ) / "package"
        logical_not = entry == LOGICAL_NOT_ENTRY
        unary = entry in UNARY_ENTRIES or logical_not or invert or absolute
        if initialization:
            names = {"out"}
        elif shaped_reduction:
            names = set(layout_module.signature(entry))
        elif reduction:
            names = {"in", "out", "in_size", "row_size"}
        elif copy:
            names = set(copy_layout.DTYPES)
        elif slice_update:
            names = set(slice_update_layout.DTYPES)
        elif binary_operation:
            names = {"a", "b", "c", "size"}
        elif selection:
            names = {"a", "b", "c", "d", "size"}
        elif cast:
            names = {"src", "dst", "size"}
        else:
            names = {"in", "size", "out"} if unary else {"start", "step", "out"}
        output_name = (
            "dst"
            if copy or cast
            else "c" if binary_operation else "d" if selection else "out"
        )
        supplied = {}
        for index in range(count):
            buffer = buffers[index]
            if not buffer.name or not buffer.dtype:
                raise ValueError("Native buffer identity is missing")
            name = buffer.name.decode("ascii")
            dtype = buffer.dtype.decode("ascii")
            if (
                name not in names
                or name in supplied
                or dtype not in TYPES
                or not buffer.data
            ):
                raise ValueError("Invalid native buffer")
            if entry in ARANGE_ENTRIES and dtype != entry.removeprefix("arange"):
                raise ValueError("Native arange buffer dtype does not match its entry")
            if initialization:
                expected = threads
                if dtype != INIT_ENTRIES[entry]:
                    raise ValueError(
                        "Reduction initialization dtype does not match its entry"
                    )
            elif reduction:
                expected = {"in": threads, "out": execution["workgroupCount"][1]}.get(
                    name, 1
                )
            else:
                expected = (
                    threads
                    if name == output_name
                    or (unary and name == "in")
                    or (binary_operation and name in {"a", "b"})
                    or (selection and name in {"a", "b", "c"})
                    or (cast and name == "src")
                    else 1
                )
            directions = (
                {copy_layout.INOUT}
                if slice_update and name == output_name
                else (
                    {1, copy_layout.INOUT}
                    if copy and name == output_name
                    else {int(name == output_name)}
                )
            )
            if (
                not copy
                and not slice_update
                and not shaped_reduction
                and buffer.count != expected
            ) or buffer.output not in directions:
                raise ValueError("Native buffer shape or direction does not match")
            if unary and dtype != (
                "uint32"
                if name == "size"
                else (
                    BITWISE_INVERT_ENTRIES[entry]
                    if invert
                    else (
                        ALL_ABSOLUTE_ENTRIES[entry]
                        if absolute
                        else "bool_" if logical_not else "float32"
                    )
                )
            ):
                raise ValueError("Native unary buffer dtype does not match")
            if binary and dtype != (
                "uint32" if name == "size" else ALL_BINARY_ENTRIES[entry]
            ):
                raise ValueError("Native binary buffer dtype does not match")
            if comparison and dtype != (
                "uint32"
                if name == "size"
                else "bool_" if name == "c" else ALL_COMPARISON_ENTRIES[entry]
            ):
                raise ValueError("Native comparison buffer dtype does not match")
            if selection and dtype != (
                "uint32"
                if name == "size"
                else "bool_" if name == "a" else SELECTION_ENTRIES[entry]
            ):
                raise ValueError("Native selection buffer dtype does not match")
            if cast and dtype != (
                "uint32"
                if name == "size"
                else ALL_CAST_ENTRIES[entry][0 if name == "src" else 1]
            ):
                raise ValueError("Native cast buffer dtype does not match")
            if (
                reduction
                and not shaped_reduction
                and not initialization
                and dtype != (
                    "uint64"
                    if name in {"in_size", "row_size"}
                    else REDUCTION_ENTRIES[entry]
                )
            ):
                raise ValueError("Native reduction buffer dtype does not match")
            supplied[name] = buffer
        if set(supplied) != names:
            raise ValueError("Native buffer names do not match the operation")
        shaped_metadata = None
        if shaped_reduction:
            shaped_metadata = layout_module.validate(
                entry, supplied, threads, execution
            )
            if small_column:
                if self.small_columns is None:
                    raise ValueError(
                        "Small column dispatch requires the pinned MLX source root"
                    )
                descriptor, package_directory = self.small_columns.get(
                    column_small_layout.variant(
                        entry, tuple(execution["workgroupSize"])
                    )
                )
            if checked_reduction is not None:
                descriptor, package_directory = checked_reduction.get(
                    entry,
                    threads,
                    execution,
                    layout=dict(
                        metadata=shaped_metadata,
                        inputCount=supplied["in"].count,
                        outputCount=supplied["out"].count,
                    ),
                )
        elif reduction and not initialization:
            reduction_layout.validate(
                supplied, threads, execution, REDUCTION_ENTRIES[entry]
            )
            if checked_reduction is not None:
                descriptor, package_directory = checked_reduction.get(
                    entry, threads, execution
                )
        copy_metadata = (
            copy_layout.validate(
                supplied,
                threads,
                dtype=ALL_COPY_ENTRIES[entry],
            )
            if copy
            else None
        )
        slice_metadata = (
            slice_update_layout.validate(
                supplied, threads, dtype=SLICE_UPDATE_ENTRIES[entry]
            )
            if slice_update
            else None
        )
        grid = (
            execution["workgroupCount"]
            if reduction
            else (copy_metadata["workgroupCount"] if copy else [threads, 1, 1])
        )
        execution = (
            launch.execution()
            if launch is not None
            else {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}
        )
        if not reduction and execution != {
            "workgroupCount": grid,
            "workgroupSize": [1, 1, 1],
        }:
            raise ValueError("Native launch geometry does not match the operation")
        if small_row:
            if self.small_rows is None:
                raise ValueError(
                    "Small-row dispatch requires the pinned MLX source root"
                )
            region_packages = self.small_rows.get(
                entry,
                thread_grid_size=execution["threadGridSize"],
                workgroup_size=execution["workgroupSize"],
            )
            descriptor, package_directory = region_packages[0]
        if (unary or binary_operation or cast or selection) and ctypes.cast(
            supplied["size"].data, ctypes.POINTER(ctypes.c_uint32)
        )[0] != threads:
            raise ValueError("Native operation size does not match the launch")
        if bitwise and entry.startswith(("vv_LeftShift", "vv_RightShift")):
            buffer = supplied["b"]
            counts = ctypes.cast(
                buffer.data, ctypes.POINTER(TYPES[BITWISE_ENTRIES[entry]])
            )
            if any(not 0 <= counts[index] < 32 for index in range(threads)):
                raise ValueError("32-bit shifts require counts in [0, 31]")
        guard = COPY_GUARD
        guarded = (
            copy
            or (unary and self.retain_native_modules)
            or binary_operation
            or cast
            or logical_not
            or reduction
            or invert
            or selection
            or absolute
            or slice_update
        )
        output_dtype = supplied[output_name].dtype.decode("ascii")
        bit_storage = (slice_update and output_dtype == "float32") or entry in (
            *HALF_CAST_ENTRIES,
            *BFLOAT_CAST_ENTRIES,
        )
        if output_dtype == "float16":
            guard = half_storage.pack(half_storage.GUARD, self.target)
        if output_dtype == "bfloat16":
            guard = bfloat_storage.pack(bfloat_storage.GUARD, self.target)
        if output_dtype == "bool_":
            guard = (
                BOOLEAN_GUARD
                if self.target == "metal"
                else [int(value) for value in BOOLEAN_GUARD]
            )
        if not bit_storage and (
            (reduction and output_dtype == "float32")
            or (binary and ALL_BINARY_ENTRIES[entry] == "float32")
            or (cast and ALL_CAST_ENTRIES[entry][1] == "float32")
            or (selection and output_dtype == "float32")
            or (unary and self.retain_native_modules and output_dtype == "float32")
        ):
            guard = [
                ctypes.c_float.from_buffer_copy(ctypes.c_uint32(word)).value
                for word in COPY_GUARD
            ]
        inputs, outputs, destinations = {}, {}, {}
        matched = set()
        binding_names = set()
        for binding in descriptor["bindings"]:
            if "executionInput" in binding.get("provenance", {}):
                continue
            layout = binding["scalarLayout"]
            member = layout.get("memberName", binding["name"])
            if self.target == "directx":
                member = member.removeprefix(entry.rstrip("_") + "_")
            name = {"out_": "out", "in_": "in"}.get(member, member)
            if (
                name not in supplied
                or name in matched
                or binding["name"] in binding_names
            ):
                raise ValueError(
                    "Reflected binding identity does not match the operation"
                )
            matched.add(name)
            binding_names.add(binding["name"])
            buffer = supplied[name]
            dtype = buffer.dtype.decode("ascii")
            storage = physical_dtype(dtype, self.target)
            reflected_storage = encoded_storage_dtype(
                layout,
                target=self.target,
                resource_kind=binding.get("kind", ""),
                logical_dtype=storage,
            )
            if layout["elementType"] != reflected_storage or layout[
                "elementStrideBytes"
            ] != (1 if storage == "bool" else ctypes.sizeof(TYPES[storage])):
                raise ValueError("Native and reflected buffer layouts disagree")
            if layout["elementSizeBytes"] != (
                1 if storage == "bool" else ctypes.sizeof(TYPES[storage])
            ):
                raise ValueError("Native and reflected element sizes disagree")
            ctype = TYPES[dtype]
            view = ctypes.cast(
                buffer.data, ctypes.POINTER(ctype * buffer.count)
            ).contents
            if (
                (comparison or (reduction and dtype == "bfloat16"))
                and buffer.output == 0
                and dtype in {"float32", "bfloat16"}
                and self.target != "metal"
            ):
                word_type = ctypes.c_uint16 if dtype == "bfloat16" else ctypes.c_uint32
                words = ctypes.cast(
                    buffer.data, ctypes.POINTER(word_type * buffer.count)
                ).contents
                mask, normal = (
                    (0x7FFF, 0x80) if dtype == "bfloat16" else (0x7FFFFFFF, 0x00800000)
                )
                if any(0 < (word & mask) < normal for word in words):
                    raise ValueError(
                        "Subnormal float comparison parity is not established for "
                        f"{self.target}; see CrossGL/crosstl#2000"
                    )
            if bit_storage and dtype == "float32":
                values = element_storage.array_values(
                    ctypes.cast(
                        buffer.data, ctypes.POINTER(ctypes.c_uint32 * buffer.count)
                    ).contents
                )
            else:
                values = [0] * buffer.count if buffer.output == 1 else wire_values(view)
            if initialization:
                # Unwritten outputs must differ from the reduction identity.
                initial_value = int("sum" in entry or entry == "init_reduce_orbool_")
                values = [initial_value] * buffer.count
            if (bitwise or absolute) and buffer.output:
                values = [
                    (
                        half_storage.GUARD[0]
                        if dtype == "float16"
                        else (
                            bfloat_storage.GUARD[0]
                            if dtype == "bfloat16"
                            else 1 if dtype == "bool_" else COPY_GUARD[0]
                        )
                    )
                ] * buffer.count
            if selection and buffer.output:
                values = [
                    int(guard[0]) if dtype == "bool_" else guard[0]
                ] * buffer.count
            if dtype == "bool_":
                values = boolean_values(values, "uint32")
                if storage == "bool":
                    values = [bool(value) for value in values]
            if dtype == "float16":
                values = half_storage.pack(values, self.target)
            if dtype == "bfloat16":
                values = bfloat_storage.pack(values, self.target)
            value = {
                "dtype": storage,
                "shape": [buffer.count],
                "values": values,
            }
            if bit_storage and dtype == "float32":
                value["encoding"] = FLOAT32_BITS
            elif dtype == "float16":
                value["encoding"] = half_storage.encoding(self.target)
            elif dtype == "bfloat16" and bfloat_storage.encoding(self.target):
                value["encoding"] = bfloat_storage.encoding(self.target)
            if guarded and buffer.output:
                value["shape"] = [buffer.count + len(guard)]
                value["values"].extend(guard)
                inputs[binding["name"]] = value
            if buffer.output:
                outputs[binding["name"]] = (
                    {key: item for key, item in value.items() if key != "values"}
                    if slice_update
                    else value
                )
                destinations[binding["name"]] = (buffer, ctype)
            else:
                inputs[binding["name"]] = value
        if matched != set(supplied):
            raise ValueError("Reflected bindings do not cover the operation")
        if small_row and self.target != "metal":
            result = region_dispatch.run_regions(
                self, region_packages, inputs, outputs, execution
            )
        else:
            request = build_native_loader_dispatch_request(
                descriptor,
                package_directory,
                inputs,
                outputs,
                execution,
                expected_target=self.target,
                **(
                    {
                        "observe_inputs": [
                            binding["name"]
                            for binding in descriptor["bindings"]
                            if binding.get("provenance", {})
                            .get("sourceResource", {})
                            .get("parameter")
                            in {"in", "in_size", "row_size"}
                        ]
                    }
                    if checked_reduction is not None
                    else {}
                ),
            )
            result = (
                gather_dispatch.execute(self, request)
                if entry in (*ALL_HALF_ENTRIES, *BFLOAT_ENTRIES)
                or self.retain_native_modules
                else self.executor.run(request)
            )
        if result.status != "ok" or set(result.outputs) != set(destinations):
            raise RuntimeError("Native executor did not return the required outputs")
        for name, (buffer, ctype) in destinations.items():
            output = result.outputs[name]
            size = buffer.count + (len(guard) if guarded else 0)
            dtype = buffer.dtype.decode("ascii")
            storage = physical_dtype(dtype, self.target)
            if output["dtype"] != storage or output["shape"] != [size]:
                raise RuntimeError("Native readback layout does not match the output")
            expected_encoding = (
                half_storage.encoding(self.target)
                if dtype == "float16"
                else (
                    bfloat_storage.encoding(self.target)
                    if dtype == "bfloat16"
                    else FLOAT32_BITS if bit_storage else None
                )
            )
            if output.get("encoding") != expected_encoding:
                raise RuntimeError(
                    "Native readback storage encoding does not match the output"
                )
            if len(output["values"]) != size:
                raise RuntimeError("Native readback size does not match the output")
            if dtype == "float16":
                half_storage.unpack(output["values"], self.target)
            elif dtype == "bfloat16":
                bfloat_storage.unpack(output["values"], self.target)
            elif dtype == "bool_":
                boolean_values(output["values"], storage)
            elif (
                bitwise
                or copy
                or absolute
                or slice_update
                or (selection and dtype != "float32")
                or dtype in {"int64", "uint64"}
                or bit_storage
            ):
                bits = ctypes.sizeof(ctype) * 8
                low, high = (
                    (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                    if dtype.startswith("int")
                    else (0, 2**bits - 1)
                )
                readback = output["values"]
                if not exact_types_within(readback, INT_TYPES) or (
                    readback and not (low <= min(readback) and max(readback) <= high)
                ):
                    raise RuntimeError(
                        "Native integer readback is outside its integer type"
                    )
            if guarded and output["values"][buffer.count :] != guard:
                raise RuntimeError("Native operation changed the output buffer guard")
            # A validated copy writes distinct in-bounds destinations, as many
            # as its shape has elements, so one that size writes every element.
            copy_fills = copy and math.prod(copy_metadata["shape"]) == buffer.count
            if (copy or slice_update) and not copy_fills:
                written = set(
                    copy_layout.destination_indices(
                        copy_metadata if copy else slice_metadata
                    )
                )
                initial = inputs[name]["values"]

                def storage_word(value):
                    if dtype == "float32" and not bit_storage:
                        return ctypes.c_uint32.from_buffer_copy(
                            ctypes.c_float(float(value))
                        ).value
                    return value

                # Validated destinations lie in [0, count), so as many distinct
                # written indices as elements leave no untouched storage.
                every_element_written = len(written) == buffer.count and (
                    not written or (min(written) >= 0 and max(written) < buffer.count)
                )
                if not every_element_written and any(
                    storage_word(value) != storage_word(initial[index])
                    for index, value in enumerate(output["values"][: buffer.count])
                    if index not in written
                ):
                    raise RuntimeError(
                        "Native operation changed untouched destination storage"
                    )
            storage_type = (
                ctypes.c_uint32 if bit_storage and dtype == "float32" else ctype
            )
            result_values = output["values"][: buffer.count]
            if dtype == "float16":
                result_values = half_storage.unpack(result_values, self.target)
            if dtype == "bfloat16":
                result_values = bfloat_storage.unpack(result_values, self.target)
            values = (storage_type * buffer.count)()
            # Slice assignment stores each item as the constructor would, in C.
            values[:] = (
                list(map(float, result_values))
                if storage_type is ctypes.c_float
                else result_values
            )
            ctypes.memmove(buffer.data, values, ctypes.sizeof(values))
        write_trace(
            self,
            {
                "entry": entry,
                "target": self.target,
                "threads": threads,
                **execution,
                "dispatchVersion": DISPATCH_VERSION,
                "artifact": descriptor["artifact"],
                "details": result.details,
                **(
                    {
                        (
                            "bfloatStorage"
                            if entry in BFLOAT_ENTRIES
                            else "halfStorage"
                        ): {
                            "logicalType": output_dtype,
                            "physicalType": storage,
                            "encoding": expected_encoding,
                            "values": output["values"][: buffer.count],
                            "guardValues": output["values"][buffer.count :],
                            "logicalWords": list(values),
                        },
                        "inputs": inputs,
                        "packageRoot": str(package_directory),
                    }
                    if entry in (*ALL_HALF_ENTRIES, *BFLOAT_ENTRIES)
                    else {}
                ),
                **(
                    {
                        "sliceUpdateValues": output["values"][: buffer.count],
                        "sliceUpdateGuardValues": output["values"][buffer.count :],
                        "sliceUpdateMetadata": slice_metadata,
                        **(
                            {
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if self.retain_native_modules
                            else {}
                        ),
                        **(
                            {
                                "sliceUpdateStorageWords": output["values"][
                                    : buffer.count
                                ],
                                "sliceUpdateGuardWords": output["values"][
                                    buffer.count :
                                ],
                                "sliceUpdateValues": [
                                    wire_value(
                                        ctypes.c_float.from_buffer_copy(
                                            ctypes.c_uint32(word)
                                        ).value
                                    )
                                    for word in output["values"][: buffer.count]
                                ],
                                "sliceUpdateGuardValues": [
                                    ctypes.c_float.from_buffer_copy(
                                        ctypes.c_uint32(word)
                                    ).value
                                    for word in output["values"][buffer.count :]
                                ],
                            }
                            if bit_storage
                            else {}
                        ),
                    }
                    if slice_update
                    else {}
                ),
                **(
                    {
                        "integer64Values": output["values"][: buffer.count],
                        "integer64GuardValues": output["values"][buffer.count :],
                    }
                    if entry in INTEGER64_ENTRIES
                    else {}
                ),
                **(
                    {
                        "selectionValues": output["values"][: buffer.count],
                        "selectionGuardValues": output["values"][buffer.count :],
                    }
                    if selection
                    else {}
                ),
                **(
                    {"bitwiseValues": output["values"][: buffer.count]}
                    if bitwise
                    else {}
                ),
                **(
                    {"absoluteValues": output["values"][: buffer.count]}
                    if absolute
                    else {}
                ),
                **(
                    {"unaryGuardValues": output["values"][buffer.count :]}
                    if unary and guarded
                    else {}
                ),
                **(
                    {
                        "unaryValues": output["values"][: buffer.count],
                        "inputs": inputs,
                        "packageRoot": str(package_directory),
                    }
                    if unary and self.retain_native_modules
                    else {}
                ),
                **({"initializationValue": initial_value} if initialization else {}),
                **(
                    {
                        "reductionGuardValues": output["values"][buffer.count :],
                        "reductionValues": output["values"][: buffer.count],
                        **(
                            {
                                "reductionLayout": dict(
                                    inputCount=supplied["in"].count,
                                    outputCount=supplied["out"].count,
                                )
                            }
                            if shaped_reduction and checked_reduction is not None
                            else {}
                        ),
                        **(
                            {
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if self.retain_native_modules
                            or checked_reduction is not None
                            else {}
                        ),
                        "reductionMetadata": (
                            {"outputSize": threads}
                            if initialization
                            else (
                                shaped_metadata
                                if shaped_reduction
                                else {
                                    name: ctypes.cast(
                                        supplied[name].data,
                                        ctypes.POINTER(ctypes.c_uint64),
                                    )[0]
                                    for name in ("in_size", "row_size")
                                }
                            )
                        ),
                    }
                    if reduction
                    else {}
                ),
                **(
                    {
                        "copyGuardWords": output["values"][buffer.count :],
                        "copyMetadata": copy_metadata,
                        "copyValues": output["values"][: buffer.count],
                        **(
                            {
                                "inputs": inputs,
                                "packageRoot": str(package_directory),
                            }
                            if self.retain_native_modules
                            else {}
                        ),
                    }
                    if copy
                    else {}
                ),
                **(
                    {"binaryGuardValues": output["values"][buffer.count :]}
                    if binary_operation
                    else {}
                ),
                **(
                    {
                        "binaryValues": output["values"][: buffer.count],
                        "inputs": inputs,
                        "packageRoot": str(package_directory),
                    }
                    if binary_operation and self.retain_native_modules
                    else {}
                ),
                **(
                    {
                        "castGuardValues": output["values"][buffer.count :],
                        "castValues": output["values"][: buffer.count],
                        "inputs": inputs,
                        "packageRoot": str(package_directory),
                    }
                    if cast
                    else {}
                ),
                **(
                    {
                        "booleanGuardValues": output["values"][buffer.count :],
                        "physicalBooleanType": physical_dtype("bool_", self.target),
                    }
                    if output_dtype == "bool_"
                    else {}
                ),
            },
        )
        self.dispatch_count = getattr(self, "dispatch_count", 0) + 1
