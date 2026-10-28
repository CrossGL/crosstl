"""Run unchanged upstream op tests on the translated host's package families.

Each test runs unmodified from the pinned ``python/tests`` tree, once on the
MLX CPU and once through the translated host. The native run installs the
base host packages with the families the dedicated CI job translates: random
bits, Boolean and integer bitwise operations, selection, absolute values,
the 64-bit integer family, reduction initialization and on-demand
whole-array and row reductions. General gather and scatter kernels are specialized
from the pinned MLX sources at dispatch time, and each of their dispatches is
recomputed from its retained inputs and native modules. DivMod's
``binary_two`` kernels are specialized the same way, and every DivMod dispatch
is recomputed from its retained operands. Casts and copies that no installed
family covers (the narrow integers and the 16-bit float pairs) translate from
``copy.metal`` on first use; each is recomputed from its retained words and
must name the package its cache derives again. Unary and binary vectors of the
other real dtypes (the 16-bit floats, the narrow integers, Boolean arithmetic
and the 64-bit bitwise operations) translate from ``unary.metal`` and
``binary.metal`` the same way; each must name its cached package and native
modules, and integer and Boolean ones are recomputed from their operands.
The elementwise kernels MLX generates for fused tapes translate per generated
source; each dispatch must name the package of its traced source and its
native modules. Scans and small row and column
reductions are also translated on first use; each of their dispatches must
name the package its cache derives again from the event. Arg reductions are
translated per threadgroup width on first use and every index is recomputed
from the retained input; OpenGL does not run the arg reduction tests yet
(``TARGET_EXCLUSIONS``). Softmax and logsumexp rows are translated per
threadgroup width on first use; each dispatch must name its cached package,
upload its retained input and keep its output guard, and the upstream tests
compare its values. Merge-sort stages are translated per launch width on
first use and every retained stage is recomputed. RMS and layer
normalization rows and their gradients are translated per threadgroup width
on first use; each dispatch must name its cached package, upload its
retained operands, keep its output guards and leave an absent weight's
gradient placeholder unwritten, and the upstream tests compare their values
with MLX's composed normalization. GEMV, dot-product, steel GEMM, split-K,
GatherMM and SegmentedMM kernels are translated per entry on first use; each
dispatch must name the package its cache derives again and its native
modules, and the upstream tests compare the products with NumPy's. Affine and
block quantization kernels are translated per entry on first use; each
dispatch must name its cached package and native modules, cover complete
quantization groups and keep its output guards, and the upstream tests
compare the values with MLX's CPU quantization. Every
other dispatch must come from one of those indexes, so a test that reaches an uninstalled
family fails the trace instead of passing on an unchecked package.
"""

import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from demos.integrations.mlx.portable_host import (
    arange_evidence,
    arange_packages,
    arg_reduce_evidence,
    arg_reduce_layout,
    compiled_evidence,
    copy_evidence,
    cross_entropy_evidence,
    cross_entropy_layout,
    divmod_evidence,
    dynamic_offset_evidence,
    dynamic_offset_packages,
    elementwise_evidence,
    gather_evidence,
    lazy_package_evidence,
    masked_scatter_evidence,
    masked_scatter_packages,
    norm_evidence,
    norm_layout,
    quantization_evidence,
    random_evidence,
    scatter_axis_evidence,
    scatter_evidence,
    searchsorted_dispatch,
    searchsorted_evidence,
    searchsorted_layout,
    softmax_evidence,
    softmax_layout,
    sort_dispatch,
    sort_evidence,
    sort_layout,
    upstream_subtest_shards,
    upstream_test_seeds,
    wide_column_dispatch,
    wide_column_evidence,
    wide_reduction_evidence,
    wide_reduction_packages,
    wide_row_dispatch,
    wide_row_evidence,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    CACHE_DIRECTORY as ARG_REDUCE_CACHE,
)
from demos.integrations.mlx.portable_host.arg_reduce_packages import (
    ArgReducePackageCache,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    CACHE_DIRECTORY as SMALL_COLUMN_CACHE,
)
from demos.integrations.mlx.portable_host.column_small_packages import (
    SmallColumnPackageCache,
)
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
from demos.integrations.mlx.portable_host.divmod_dispatch import (
    ENTRY_PREFIXES as DIVMOD_PREFIXES,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    CACHE as ELEMENTWISE_CACHE,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ENTRIES as LAZY_ELEMENTWISE_ENTRIES,
)
from demos.integrations.mlx.portable_host.elementwise_packages import (
    ElementwisePackageCache,
)
from demos.integrations.mlx.portable_host.norm_packages import (
    CACHE_DIRECTORY as NORM_CACHE,
)
from demos.integrations.mlx.portable_host.norm_packages import NormPackageCache
from demos.integrations.mlx.portable_host.prepare import COMMIT, verify_prepared
from demos.integrations.mlx.portable_host.random_packages import (
    ENTRIES,
)
from demos.integrations.mlx.portable_host.random_packages import (
    load_index as load_random,
)
from demos.integrations.mlx.portable_host.reduction_packages import (
    load_index as load_reductions,
)
from demos.integrations.mlx.portable_host.reduction_packages import variant_keys
from demos.integrations.mlx.portable_host.runtime import HostRuntime
from demos.integrations.mlx.portable_host.searchsorted_packages import (
    CACHE_DIRECTORY as SEARCHSORTED_CACHE,
)
from demos.integrations.mlx.portable_host.searchsorted_packages import (
    SearchSortedPackageCache,
)
from demos.integrations.mlx.portable_host.small_row_packages import SmallRowPackageCache
from demos.integrations.mlx.portable_host.softmax_packages import (
    CACHE_DIRECTORY as SOFTMAX_CACHE,
)
from demos.integrations.mlx.portable_host.softmax_packages import SoftmaxPackageCache
from demos.integrations.mlx.portable_host.sort_packages import (
    CACHE_DIRECTORY as SORT_CACHE,
)
from demos.integrations.mlx.portable_host.sort_packages import SortPackageCache
from demos.integrations.mlx.portable_host.trace_records import TraceStream
from demos.integrations.mlx.portable_host.upstream_module_groups import (
    ATTENTION_SWEEP_TESTS,
)
from demos.integrations.mlx.portable_host.upstream_module_groups import (
    GROUPS as MODULE_GROUPS,
)
from demos.integrations.mlx.portable_host.upstream_trace_audit import TraceAudit
from demos.integrations.mlx.portable_host.verify import upstream_test_sources
from demos.integrations.mlx.portable_host.verify_bitwise import verify_native_identity
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

# Whole-module tests beyond the core operations: losses, initializers, linear
# algebra, einsum paths, FFT helpers and tree utilities, and the fused
# cross-entropy tests of test_fast. Each reaches the GPU; linear algebra
# factorizations run on the MLX CPU stream as upstream directs, and their
# operands and checks run through the host.
MODULE_TESTS = tuple(
    f"{module}.{case}.test_{name}"
    for module, case, names in (
        (
            "test_losses",
            "TestLosses",
            (
                "binary_cross_entropy",
                # The GPU loss runs the fused cross_entropy.metal rows.
                "cross_entropy",
                "cosine_similarity_loss",
                "gaussian_nll_loss",
                "hinge_loss",
                "huber_loss",
                "kl_div_loss",
                "l1_loss",
                "log_cosh_loss",
                "margin_ranking_loss",
                "mse_loss",
                "nll_loss",
                "smooth_l1_loss",
                "triplet_loss",
            ),
        ),
        (
            "test_einsum",
            "TestEinsum",
            (
                "attention",
                "broadcasting",
                "ellipses",
                "ellipses_broadcast",
                # The einsum_path searches, and opt_einsum's contraction
                # cases through every path they select.
                "longer_paths",
                "multi_input_einsum",
                "opt_einsum_test_cases",
                "scalar_operands",
                "simple_einsum",
                "simple_path",
                "sum_first",
                "two_input_einsum",
            ),
        ),
        ("test_init", "TestInit", ("orthogonal", "sparse_zeros_per_row", "uniform")),
        (
            "test_linalg",
            "TestLinalg",
            (
                "cholesky",
                "cholesky_inv",
                "complex_norm",
                "cross_product",
                "det",
                "inverse",
                "lu",
                "lu_factor",
                "norm",
                "pseudo_inverse",
                "qr_factorization",
                "slogdet",
                "solve",
                "svd_decomposition",
                "tri_inverse",
            ),
        ),
        ("test_tree", "TestTreeUtils", ("merge", "supported_trees")),
        (
            "test_fft",
            "TestFFT",
            ("fftfreq", "fftshift", "fftshift_errors", "ifftshift", "rfftfreq"),
        ),
        # These reach the fused cross-entropy rows of every float dtype from
        # one to the most simdgroups, and the fused gradient rows.
        # test_cross_entropy_grad's composed reference gradient scatters
        # (4, 7, 4096) outputs, which the host's axis scatter splits.
        ("test_fast", "TestFast", ("cross_entropy", "cross_entropy_grad")),
    )
    for name in names
)

# Convolution tests whose every route the host ports: conv1d as conv2d over
# expand_dims views, large filters over few channels, flipped and grouped 1-D
# convolutions, NumPy's 1-D and 2-D convolutions of every float dtype,
# repeated convolutions, the flipped and grouped gradients, the gradient
# shapes of strided and dilated convolutions, unaligned 2-D channels and the
# per-depth-tap 3-D decomposition, which reach the unfold, depthwise,
# padded-channel, general and implicit GEMM and small-depth 3-D routes of
# conv.cpp.
CONV_TESTS = tuple(
    "test_conv.TestConv.test_" + name
    for name in (
        "basic_grad_shapes",
        "conv2d_large_filter_small_channels",
        "conv2d_unaligned_channels",
        "conv_1d_groups_flipped",
        "conv_1d_with_2d",
        "conv_3D_small_kd_decomposition",
        "conv_general_flip_grad",
        "conv_groups_grad",
        "numpy_conv",
        "repeated_conv",
    )
)

# test_array methods that reach the GPU: indexing and its gradients, item
# assignment and functional .at[] updates, slice updates of 2-D contiguous
# arrays, copies, casts, views and the buffer, DLPack and pickle round trips
# of device arrays, and the array equality operators.
ARRAY_TESTS = (
    *(
        "test_array.TestArray.test_" + name
        for name in (
            "add_numpy",
            "api",
            "array_at",
            "array_comparison",
            "array_copy",
            "array_from_noncontiguous_np",
            "array_np_dtype_conversion",
            "array_pickle",
            "array_repr_precision",
            "array_subclass",
            "array_unary_ops",
            "array_view_ref_counting",
            "asarray",
            "buffer_protocol",
            "buffer_protocol_eval_error",
            "buffer_protocol_ref_counting",
            "construction_from_lists",
            "construction_from_lists_wide_ints",
            "dlpack",
            "dlpack_cpu_dtype_mapping",
            "format",
            "getitem_with_list",
            "indexing",
            "indexing_grad",
            "init_from_array",
            "inplace",
            "load_from_pickled_np",
            "logical_overloads",
            "matrix_transpose",
            "memoryless_copy",
            "multi_output_leak",
            "np_array_conversion_copies_by_default",
            "real_imag",
            "setitem",
            "setitem_with_boolean_mask",
            "slice_bounds_with_array_index",
            "slice_update_contiguous_2d",
        )
    ),
    *(
        "test_array.TestEquality.test_" + name
        for name in (
            "array_eq_array",
            "array_eq_scalar",
        )
    ),
    *(
        "test_array.TestInequality.test_" + name
        for name in (
            "array_ne_array",
            "array_ne_scalar",
        )
    ),
)

# test_autograd methods: VJPs and JVPs of the composed ops, including the
# complex ones, scans, sorts, scatters, slices and every slice update
# reduction, custom functions and gradient trees.
AUTOGRAD_TESTS = (
    *(
        "test_autograd.TestAutograd.test_" + name
        for name in (
            "autograd_types",
            "auxiliary_values",
            "captured",
            "complex_abs_grad",
            "complex_exp_vjp",
            "complex_log_vjp",
            "complex_prod_vjp",
            "complex_unary_vjps",
            "complex_vjps",
            "concatenate_vjps",
            "cummax_grad",
            "cummin_grad",
            "cumprod_grad",
            "custom_function",
            "eval_in_grad",
            "grad",
            "grad_kwargs",
            "grad_trees",
            "jvp",
            "jvp_through_bitwise_ops",
            "jvp_with_constant_inputs",
            "jvp_with_partly_traced_inputs",
            "logsumexp_grad",
            "matmul_jvps",
            "power_grad",
            "put_along_axis_grads",
            "reduce_jvp",
            "scatter_add_vjp",
            "scatter_max_vjp",
            "scatter_min_vjp",
            "scatter_vjp",
            "slice_grads",
            "slice_grads_single_element",
            "slice_update_add_vjp",
            "slice_update_max_vjp",
            "slice_update_min_vjp",
            "slice_update_multiply_vjp",
            "sort_grad",
            "split_against_slice",
            "stop_gradient",
            "stop_gradient_computed_indices",
            "topk_grad",
            "transform_releases_captures",
            "update_state",
            "vjp",
        )
    ),
)

# test_compile methods: compiled tapes of every dtype, captured state and
# random keys, compiled gradients, and shapeless graphs, including the ones
# that count reduced elements with NumberOfElements on the host.
COMPILE_TESTS = (
    *(
        "test_compile.TestCompile.test_" + name
        for name in (
            "compile_abs_unsigned",
            "compile_broadcast_only",
            "compile_capture",
            "compile_changing_outputs",
            "compile_changing_outputs_with_state",
            "compile_create_list",
            "compile_different_log_bases",
            "compile_dynamic_dims",
            "compile_equal_nan",
            "compile_float_constant_precision",
            "compile_grad",
            "compile_inf",
            "compile_inputs_with_primitives",
            "compile_kwargs",
            "compile_large_graph_with_broadcasts",
            "compile_many_inputs",
            "compile_many_outputs",
            "compile_multi_output",
            "compile_nonfinite_constants",
            "compile_output_with_siblings",
            "compile_release_on_another_thread",
            "compile_rng",
            "compile_sigmoid_matches_eager",
            "compile_state_capture_with_rng_updates_in_place",
            "compile_tuple_output_in_thread",
            "compile_two_input_grad",
            "compile_types",
            "compile_vjp",
            "compile_with_closure",
            "compile_with_constant",
            "compile_with_long_name",
            "compile_with_none",
            "compiled_subnormal_bool_cast",
            "dtypes",
            "function_creates_array",
            "inf_constant",
            "max_into_equal",
            "multiple_compile_same_capture",
            "outputs_changing",
            "shapeless_compile",
            "shapeless_compile_addmm",
            "shapeless_compile_gather_mm",
            "shapeless_compile_gather_qmm",
            "shapeless_compile_matmul",
            "shapeless_compile_reduce_after_gather",
            "shapeless_compile_slice_update",
            "shapeless_compile_with_broadcasts",
            "shapeless_compile_with_reduction",
            "shapeless_mean",
            "shared_broadcast",
            "simple_compile",
            "transform_over_eval_compiled",
            "vjp_vjp_compiled",
            "vmap_compiled",
        )
    ),
)

UPSTREAM_TESTS = (
    tuple(
        "test_ops.TestOps." + name
        for name in (
            "test_trunc",
            "test_erfinv",
            "test_logical_and",
            "test_logical_or",
            "test_full_ones_zeros",
            "test_atleast_1d",
            "test_atleast_2d",
            "test_atleast_3d",
            "test_tile",
            "test_repeat",
            "test_roll",
            "test_roll_errors",
            "test_tri",
            "test_tril",
            "test_triu",
            "test_as_strided",
            "test_view",
            "test_depends",
            "test_eye",
            "test_diag",
            "test_diagonal",
            "test_divmod",
            # Base families only.
            "test_all",
            "test_cummax_cummin_nan",
            "test_empty_matmuls",
            "test_irregular_alignments",
            "test_scan_invalid_axis",
            "test_scan_strided_bool",
            "test_slice_with_negative_stride",
            "test_take_along_axis",
            "test_unary_ops_from_non_array",
            # These also reach the 64-bit integer family.
            "test_add",
            "test_any",
            "test_bitwise_grad",
            "test_count_nonzero",
            "test_irregular_binary_ops",
            "test_linspace",
            "test_linspace_endpoint",
            "test_mean",
            "test_min_and_max",
            "test_prod",
            # This also reaches lazily translated casts.
            "test_subnormal_bool_cast",
            # These reach lazily translated unary and binary vectors.
            "test_binary_ops",
            "test_bitwise_ops",
            "test_divide",
            "test_floor_divide",
            "test_integer_power",
            "test_remainder",
            "test_scalar_inputs",
            "test_sigmoid",
            "test_tensordot",
            "test_unary_ops",
            "test_zeros_ones_empty_like_dtype",
            # These also reach lazily translated complex64 kernels.
            "test_array_equal",
            "test_complex_ops",
            "test_complex_power",
            "test_conjugate",
            "test_diag_zero_size",
            "test_isfinite",
            "test_isinf",
            "test_isnan",
            "test_isneginf",
            "test_isposinf",
            "test_log",
            "test_log10",
            "test_log1p",
            "test_log2",
            "test_logaddexp",
            "test_maximum",
            "test_minimum",
            "test_outer",
            "test_real_imag",
            "test_sign",
            "test_trig_ops",
            # This reaches the lazily translated ToFP8 and FromFP8 vectors.
            "test_to_from_fp8",
            # These reach lazily translated selections of every GPU dtype.
            "test_clip",
            "test_concatenate",
            "test_meshgrid",
            "test_nan_to_num",
            "test_pad",
            "test_pad_reflect_symmetric",
            "test_slice_update_reversed",
            "test_take",
            # This reaches int32 and float32 replacement axis-scatters.
            "test_put_along_axis",
            # This reaches the lazily translated dot product, its whole-array
            # partial sum, and matrix-vector products.
            "test_inner",
            # This also reaches lazily translated aranges.
            "test_round",
            # These reach lazily translated wide whole-array, small-row and
            # small-column reductions.
            "test_std",
            "test_trace",
            "test_var",
            "test_vecdot",
            # This also reaches row_reduce_looped through the on-demand row index.
            "test_sum",
            # This slices and updates at array start indices, which the
            # translated offset kernel turns into element offsets.
            "test_dynamic_slicing",
        )
    ) + (
        # These reach lazily translated arg reductions of every real dtype, and
        # of the flattened array in a threadgroup of thirteen simdgroups.
        "test_ops.TestOps.test_argmin_argmax",
        "test_reduce.TestReduce.test_arg_reduce",
        "test_bf16.TestBF16.test_arg_reduction_ops",
    ) + (
        # These reach lazily translated whole-array, row and column sums,
        # products and extrema of every integer, float and Boolean dtype, over
        # permuted, expanded and negative-stride views, and empty outputs.
        "test_reduce.TestReduce.test_axis_permutation_sums",
        "test_reduce.TestReduce.test_expand_sums",
        "test_reduce.TestReduce.test_row_reduce_negative_stride",
        "test_reduce.TestReduce.test_dtypes",
        "test_reduce.TestReduce.test_sum_bool",
        "test_reduce.TestReduce.test_edge_case",
        "test_reduce.TestReduce.test_zero_size",
        # These reach complex64 extrema with NaN in either part, and a column
        # sum over 8,192 rows.
        "test_reduce.TestReduce.test_nan_propagation_complex64",
        "test_reduce.TestReduce.test_long_column",
        # This reaches Boolean all and any reductions of signed zeros.
        "test_reduce.TestReduce.test_and_or_negative_zero",
    ) + (
        # These reach lazily translated block and looped softmax and logsumexp
        # rows, precise 16-bit softmax, and transposed and broadcast inputs.
        "test_ops.TestOps.test_softmax",
        "test_ops.TestOps.test_logsumexp",
        "test_ops.TestOps.test_logsumexp_shape",
    ) + (
        # These reach lazily translated single-block sorts and argsorts of
        # contiguous, strided, broadcast and reversed views, and multi-block
        # tile sorts, partitions and merges up to a 2^22-element row.
        "test_ops.TestOps.test_sort",
        "test_ops.TestOps.test_partition",
        "test_ops.TestOps.test_argpartition",
        "test_ops.TestOps.test_sort_nan",
        "test_ops.TestOps.test_argsort_nan",
        # These reach the same sorts through medians of float16, bfloat16
        # and float32 arrays over several axes, with NaN in any position.
        "test_ops.TestOps.test_median",
        "test_ops.TestOps.test_median_nan",
    ) + (
        # This reaches lazily translated binary searches of contiguous,
        # strided, reversed and empty sequences, and of transposed, sliced
        # and broadcast values, through searchsorted.metal.
        "test_ops.TestOps.test_searchsorted",
        # This reaches concatenations of int64 arrays through lazy copies.
        "test_ops.TestOps.test_unstack",
        # This reaches the kernels MLX generates for the fused tapes of
        # mx.unique's compiled helpers, with sorts and scans.
        "test_ops.TestOps.test_unique",
        # This reaches the translated masked_scatter kernels through boolean
        # masks of whole arrays and of leading dimensions.
        "test_ops.TestOps.test_masked_scatter",
    ) + (
        # These reach lazily translated contiguous and strided, forward and
        # reverse, inclusive and exclusive scans of float32, int32, int64 and
        # complex64, log-add-exp scans, and size-one axes of any stride.
        "test_ops.TestOps.test_scans",
        "test_ops.TestOps.test_logcumsumexp",
        "test_ops.TestOps.test_scans_complex_exclusive",
        "test_ops.TestOps.test_scan_size_one_axis",
    ) + (
        # These reach lazily translated single-row and looped RMS and layer
        # normalization of every float dtype, their gradient kernels with and
        # without a weight, and the column sums of the weight gradients.
        "test_fast.TestFast.test_rms_norm",
        "test_fast.TestFast.test_rms_norm_grad",
        "test_fast.TestFast.test_layer_norm_dim_check",
        "test_fast.TestFast.test_layer_norm",
        "test_fast.TestFast.test_slice_into_layer_norm",
        "test_fast.TestFast.test_layer_norm_grad",
        "test_fast.TestFast.test_layer_norm_grad_no_bias",
        "test_fast.TestFast.test_layer_norm_grad_no_params",
        "test_fast.TestFast.test_layer_norm_grad_params",
    ) + tuple(
        # These reach lazily translated GEMV, dot-product, steel GEMM, split-K
        # and AddMM epilogues of every float dtype, complex64 GEMV, unaligned,
        # batched, broadcast, transposed and strided operands and empty
        # products, BlockMaskedMM's masked gemv and steel kernels, GatherMM's
        # gathered gemv, steel gather and group offset kernels and
        # SegmentedMM's kernels with their gradients.
        "test_blas.TestBlas." + name
        for name in (
            "test_matmul",
            "test_matmul_unaligned",
            "test_matvec_unaligned",
            "test_matmul_shapes",
            "test_matmul_dtypes",
            "test_matmul_batched",
            "test_matrix_vector",
            "test_matrix_vector_batched",
            "test_matrix_vector_broadcast",
            "test_matrix_vector_attn",
            "test_matrix_vector_edgecases",
            "test_mismatch_stride_mm",
            "test_gemv_gemm_same_precision",
            "test_complex_gemv",
            "test_addmm",
            "test_addmm_grad",
            "test_dot_product",
            "test_empty_matmul",
            "test_wide_matmul",
            "test_wide_addmm",
            "test_block_masked_matmul",
            "test_gather_matmul",
            "test_gather_matmul_grad",
            "test_gather_mm_sorted",
            "test_gather_mm_sorted_vjp",
            "test_gather_mm_blocks",
            "test_segmented_mm",
            "test_gather_matmul_index_vjp_requires_stop_gradient",
        )
    ) + MODULE_TESTS + CONV_TESTS + ARRAY_TESTS + AUTOGRAD_TESTS + COMPILE_TESTS + tuple(
        name for tests in MODULE_GROUPS.values() for name in tests
    )
)
# Tests a target cannot run yet, with the issue that tracks each. OpenGL's
# software subgroups cannot run an arg reduction wider than one simdgroup.
TARGET_EXCLUSIONS = {
    "opengl": {
        "test_ops.TestOps.test_argmin_argmax": "CrossGL/crosstl#2207",
        "test_reduce.TestReduce.test_arg_reduce": "CrossGL/crosstl#2207",
        "test_bf16.TestBF16.test_arg_reduction_ops": "CrossGL/crosstl#2207",
        # Reverse scans read through wrapped unsigned pointer offsets.
        "test_ops.TestOps.test_scans": "CrossGL/crosstl#2905",
        "test_ops.TestOps.test_scans_complex_exclusive": "CrossGL/crosstl#2905",
    }
}
# Source-specialized indexing kernels by entry prefix, longest prefix first.
INDEXING_PREFIXES = ("gather", "scatter_axis", "scatter")
# Entry-keyed package families; reductions are keyed by width and entry.
FAMILIES = (
    "packages",
    "random",
    "bitwise",
    "selection",
    "absolute",
    "integer64",
    "slice_updates",
)
# Families loaded from a translated index; the random family has its own loader.
INDEXED_FAMILIES = FAMILIES[2:]
REQUIRED_REDUCTIONS = frozenset(("w32/all_reduce_andbool_", "w256/all_reduce_andbool_"))
NEGATIVE_CHECKS = {"missing": "No translated package for vv_Equalint32"}
# Dispatches whose trace records keep their long values for the audits. Random
# records keep their keys and digest the storage the audit recomputes from them
# (``random_dispatch.dispatch``).
RETAINED_VALUES = (
    *arg_reduce_layout.OPERATIONS,
    *softmax_layout.PREFIXES,
    *sort_dispatch.PREFIXES,
    searchsorted_dispatch.PREFIX,
    *norm_layout.PREFIXES,
    cross_entropy_layout.PREFIX,
    *INDEXING_PREFIXES,
    *DIVMOD_PREFIXES,
    # Masked scatter audits recompute the output from the mask and source.
    masked_scatter_packages.ENTRY_PREFIX,
    "all_reduce_",
    "init_reduce_",
    # Wide small-row and small-column audits derive every upload from the
    # values and hash the outputs, also where they do not recompute them.
    *wide_row_dispatch.ENTRIES,
    *wide_column_dispatch.ENTRIES,
)
# Tests that reject their arguments before building a graph, or whose every
# result is empty. They dispatch no kernel on either worker, and the native
# worker must dispatch none for them.
NO_DISPATCH_TESTS = frozenset(
    (
        "test_fast.TestFast.test_layer_norm_dim_check",
        "test_reduce.TestReduce.test_zero_size",
    )
)
# Tests that skip on the MLX CPU because they need a GPU: the CPU worker must
# skip each of them and the native worker must run them. They compare the
# products with NumPy's or the CPU stream's, so the CPU run is not their
# reference.
GPU_ONLY_TESTS = frozenset(
    (
        *(
            "test_blas.TestBlas." + name
            for name in (
                "test_dot_product",
                "test_wide_matmul",
                "test_wide_addmm",
                "test_gather_mm_blocks",
            )
        ),
        # Evaluates GPU and CPU streams that wait on each other.
        "test_eval.TestEval.test_multistream_deadlock",
        # Overrides the two-pass vector kernels' block count (sdpa.inc).
        "test_fast_sdpa.TestFastSDPA.test_sdpa_blocks_env_override",
        # Full attention with head dims 72, 80 and 96, and the shards of
        # test_sdpa, which is too slow on the CPU.
        *(
            "test_fast_sdpa.TestFastSDPA." + name
            for name in (
                "test_sdpa_head_dim_72",
                "test_sdpa_head_dim_80",
                "test_sdpa_head_dim_72_80_sinks",
                "test_sdpa_head_dim_96",
            )
        ),
        *ATTENTION_SWEEP_TESTS,
    )
)
# CI runs each group in its own job. The normalization gradients move tens of
# millions of elements through the host, about as long as every other test, and
# the reduction tests sweep thousands of axis, dtype and layout subcases. The
# two longest sweeps translate a package for each new reduction launch, about
# half an hour each, so each runs in a job of its own. The BLAS tests reach
# about a hundred and fifty matrix kernels, which their jobs translate ahead of
# the run (upstream_matrix_packages); the wide products and the gathered and
# segmented ones each take about as long as the other BLAS tests together.
# The gathered blocks test takes almost as long on its own: it gathers weight
# slices of more than 65,535 elements in parts and casts its operands between
# float32 and the half types in thousands of copy launches.
# The block-masked test sweeps every mask pair and both block sizes over
# thousands of products, a hundred masked and plain matrix kernels, longer
# than the gathered blocks test.
GROUP_PREFIXES = {
    "normalization": "test_fast.",
    "reductions": "test_reduce.",
    "blas": "test_blas.",
}
GROUP_TESTS = {
    "reduction-dtypes": ("test_reduce.TestReduce.test_dtypes",),
    "reduction-permutations": ("test_reduce.TestReduce.test_axis_permutation_sums",),
    "blas-wide": tuple(
        "test_blas.TestBlas." + name for name in ("test_wide_matmul", "test_wide_addmm")
    ),
    "blas-gather": tuple(
        "test_blas.TestBlas." + name
        for name in (
            "test_gather_matmul",
            "test_gather_matmul_grad",
            "test_gather_mm_sorted",
            "test_gather_mm_sorted_vjp",
            "test_segmented_mm",
            "test_gather_matmul_index_vjp_requires_stop_gradient",
        )
    ),
    "blas-gather-blocks": ("test_blas.TestBlas.test_gather_mm_blocks",),
    "blas-masked": ("test_blas.TestBlas.test_block_masked_matmul",),
    # The module tests run in one job; the norm sweep is the longest of them.
    "modules": MODULE_TESTS,
    # The convolution tests translate their conv.metal, steel_conv and GEMM
    # kernels on demand, so they run in a job of their own.
    "conv": CONV_TESTS,
    # The array, autograd and compile modules each run in a job of their own;
    # each translates its lazy vectors, copies and compiled tapes on demand.
    "array": ARRAY_TESTS,
    "autograd": AUTOGRAD_TESTS,
    "compile": COMPILE_TESTS,
    # The GPU methods of the nn, vmap, eval, export, load, optimizer and
    # rotary embedding modules, about a quarter of an hour each natively.
    **MODULE_GROUPS,
}


def _group_of(name):
    for group, tests in GROUP_TESTS.items():
        if name in tests:
            return group
    for group, prefix in GROUP_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return "operations"


GROUPS = {
    group: tuple(name for name in UPSTREAM_TESTS if _group_of(name) == group)
    for group in ("operations", *GROUP_PREFIXES, *GROUP_TESTS)
}
# The CI step bounds a group's workers and its audit at 5,000 seconds; a cold
# native run also translates its lazy vectors and kernels.
TIMEOUTS = {"cpu": 300, "native": 4400, "missing": 60}
# The native worker checks each dispatch record whole as it is written and
# keeps it only in this compact form; its per-record audit (TraceAudit) and
# the full records' count, size and SHA-256 go to TRACE_AUDIT.
COMPACT_TRACE = "native/dispatch.compact.jsonl"
TRACE_AUDIT = "native/trace-audit.json"


def upstream_tests(target, group=None):
    """The upstream tests ``target`` runs in ``group``, in inventory order.

    ``group=None`` selects every group.
    """
    if group is not None and group not in GROUPS:
        raise ValueError(f"Unknown upstream op group: {group}")
    excluded = TARGET_EXCLUSIONS.get(target, {})
    selected = UPSTREAM_TESTS if group is None else GROUPS[group]
    return tuple(name for name in selected if name not in excluded)


def negative_checks(group):
    """The rejection checks a group runs; the operation tests own them."""
    return NEGATIVE_CHECKS if group in {None, "operations"} else {}


def requires_indexing(group):
    """Whether a group's trace must reach a gather or scatter."""
    return group in {None, "operations"}


def package_target(packages):
    return json.loads((Path(packages) / "index.json").read_text())["target"]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def install(args, output):
    trace = Path(COMPACT_TRACE).name if args.worker == "native" else "dispatch.jsonl"
    host = HostRuntime(
        args.packages,
        output / trace,
        random=args.random,
        bitwise=args.bitwise,
        selection=args.selection,
        absolute=args.absolute,
        integer64=args.integer64,
        slice_updates=args.slice_updates,
        reductions=list(args.reductions),
        mlx_root=args.mlx_root,
        # The audits recompute these dispatches from their values; the other
        # families are checked by artifact and native identity only.
        trace_values=RETAINED_VALUES,
    )
    if args.worker == "missing":
        host.descriptors.pop("vv_Equalint32")
        # Without the pinned source the comparison has no package at all.
        host.elementwise = None
    # Lazy copies and integer or Boolean lazy vectors are recomputed from their
    # values; installed families are not.
    host.trace_values = (
        *RETAINED_VALUES,
        *(entry for entry in LAZY_COPY_ENTRIES if host.lazy_copy(entry)),
        *(
            entry
            for entry in LAZY_ELEMENTWISE_ENTRIES
            if host.lazy_elementwise(entry) and elementwise_evidence.recomputable(entry)
        ),
        *(
            entry
            for entry in arange_packages.ENTRIES
            if host.lazy_arange(entry) and arange_evidence.recomputable(entry)
        ),
    )
    host.install()
    return host


def reject(mx, host, output, check):
    try:
        mx.eval(mx.array_equal(mx.array([1, 2, 3]), mx.array([1, 2, 4])))
    except (ValueError, RuntimeError) as error:
        record = {
            "check": check,
            "error": str(error),
            "dispatchCount": host.dispatch_count,
        }
        write_json(output / "rejection.json", record)
        if NEGATIVE_CHECKS[check] not in str(error) or host.dispatch_count:
            raise RuntimeError(
                "The missing package was not refused before dispatch"
            ) from error
        return
    raise RuntimeError("A comparison without its package was accepted")


def worker(args):
    import mlx.core as mx
    import numpy as np

    output = args.output_dir
    output.mkdir(parents=True)
    if mx.is_available(mx.gpu):
        raise RuntimeError("A different GPU backend is already available")
    host = None
    if args.worker == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        host = install(args, output)
    if args.worker in NEGATIVE_CHECKS:
        return reject(mx, host, output, args.worker)
    audit = None
    if host is not None:
        # Every record is checked whole as the host writes it, and only its
        # compact form reaches the disk.
        target, indexes, _ = load_indexes(args)
        audit = trace_audit(np, args, target, indexes, output / "native-modules")
        host.trace_audit = audit
    os.environ["DEVICE"] = "gpu" if host else "cpu"
    os.environ["CI"] = "1"
    sys.path.insert(0, str(args.mlx_root / "python/tests"))
    tests = upstream_tests(package_target(args.packages), args.group)
    records = []
    for name in tests:
        start = host.dispatch_count if host else 0
        seed = (
            None
            if args.seed is None
            else upstream_test_seeds.test_seed(args.seed, name)
        )
        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed % (1 << 32))
        result, subtests = run_upstream_test(name)
        records.append(
            {
                "test": name,
                "testsRun": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skips": len(result.skipped),
                "dispatchCount": host.dispatch_count - start if host else 0,
                "seed": seed,
                **({} if subtests is None else {"subtests": subtests}),
            }
        )
        write_json(output / "upstream.json", {"tests": records})
    if audit is not None:
        (output / Path(TRACE_AUDIT).name).write_text(
            json.dumps(audit.summary(), allow_nan=False), encoding="utf-8"
        )
    validate_upstream({"tests": records}, native=host is not None, inventory=tests)
    if audit is not None and audit.error is not None:
        raise RuntimeError(f"The native trace audit failed: {audit.error}")


def run_upstream_test(name):
    """Run upstream test ``name``, or its shard, and the shard's subtest tally."""
    test, shard, count = upstream_subtest_shards.parse(name)
    suite = unittest.defaultTestLoader.loadTestsFromName(test)
    if shard is None:
        return unittest.TextTestRunner(verbosity=2).run(suite), None
    module = sys.modules[test.split(".", 1)[0]]
    with upstream_subtest_shards.selecting(module, shard, count) as tally:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result, dict(tally)


def validate_upstream(record, *, native, inventory=UPSTREAM_TESTS):
    tests = record.get("tests", [])
    if [item.get("test") for item in tests] != list(inventory):
        raise ValueError("Upstream op test inventory is incomplete")
    for item in tests:
        skips = int(not native and item.get("test") in GPU_ONLY_TESTS)
        expected = {"testsRun": 1, "failures": 0, "errors": 0, "skips": skips}
        count = item.get("dispatchCount")
        dispatches = native and item.get("test") not in NO_DISPATCH_TESTS
        if (
            any(item.get(key) != value for key, value in expected.items())
            or type(count) is not int
            or (count <= 0 if dispatches else count != 0)
        ):
            raise ValueError(f"Unchanged upstream test failed: {item.get('test')}")
        upstream_subtest_shards.validate(
            item["test"], item.get("subtests"), ran=not skips
        )


def load_indexes(args):
    base = json.loads((args.packages / "index.json").read_text())
    target = base["target"]
    indexes = {"packages": base}
    indexes["random"] = {
        "target": target,
        "descriptors": load_random(args.random, target),
    }
    for name in INDEXED_FAMILIES:
        index = json.loads((getattr(args, name) / "index.json").read_text())
        if index.get("target") != target:
            raise ValueError(f"The {name} packages target another backend")
        indexes[name] = index
    reductions = [load_reductions(path, target) for path in args.reductions]
    keys = set().union(*(variant_keys(index) for index in reductions))
    if REQUIRED_REDUCTIONS - keys:
        raise ValueError(
            "Upstream op assertions require reduction packages: "
            + ", ".join(sorted(REQUIRED_REDUCTIONS - keys))
        )
    return target, indexes, reductions


def indexing_auditor(entry):
    """Return the evidence check for a source-specialized indexing entry."""
    # GatherMM's group offsets share the gather prefix; the lazy matrix audit
    # owns them.
    if lazy_package_evidence.family(entry) is not None:
        return None
    auditors = {
        "gather": gather_evidence.audit_event,
        "scatter_axis": scatter_axis_evidence.audit_event,
        "scatter": scatter_evidence.audit_event,
    }
    prefix = next((item for item in INDEXING_PREFIXES if entry.startswith(item)), None)
    return auditors.get(prefix)


def event_audits(np, args, target, installed, modules=None):
    """The per-dispatch evidence checks of a native trace, in audit order.

    Each check is ``(selects, family, audit, role)``. ``selects(entry, event)``
    picks the records the check owns. ``family`` names the dispatches in the
    error for a record that targets another backend; ``None`` leaves the
    target to the check itself. ``audit(event)`` recomputes or derives the
    record's result, package or native modules. ``role`` says what the check
    settles: a ``random`` record's package is tied to it later, ``indexing``
    and ``owned`` checks tie the record to its package and native modules, and
    a ``scan`` check covers the native modules only where it reports so.
    ``modules`` is the native worker's retained module directory, by default
    under the verifier's output directory.
    """
    root, packages = args.mlx_root, args.packages
    if modules is None:
        modules = Path(args.output_dir) / "native/native-modules"

    def lazy(entries, family, check, cache):
        def selects(entry, event):
            return entry in entries and entry not in installed

        return selects, family, lambda event: check(event, cache), "owned"

    def owned(entries, family, check, cache):
        def selects(entry, event):
            return entry in entries

        return selects, family, lambda event: check(event, cache), "owned"

    scans = lazy_package_evidence.LazyPackages(root, packages, target)
    compiled = compiled_evidence.cache(root, packages, target)
    quantizations = quantization_evidence.cache(root, packages, target)

    return (
        (
            lambda entry, event: entry in ENTRIES,
            None,
            lambda event: random_evidence.audit_event(np, event),
            "random",
        ),
        (
            lambda entry, event: indexing_auditor(entry) is not None,
            "indexing dispatch",
            # Recomputes the result from the retained inputs and native modules.
            lambda event: indexing_auditor(event["entry"])(np, event),
            "indexing",
        ),
        (
            lambda entry, event: entry.startswith(DIVMOD_PREFIXES),
            "DivMod dispatch",
            # Recomputes both outputs and checks the retained native modules.
            lambda event: divmod_evidence.audit_event(event),
            "owned",
        ),
        # Each check below derives its cached package again and checks the
        # native modules. Copies, integer and Boolean vectors and integer
        # aranges, offsets, masked scatters, reductions, arg reductions, sorts
        # and searches are also recomputed from their retained inputs; the
        # upstream tests compare the other values.
        lazy(
            LAZY_COPY_ENTRIES,
            "copy dispatch",
            copy_evidence.audit_event,
            CopyPackageCache(root, packages / COPY_CACHE, target),
        ),
        lazy(
            LAZY_ELEMENTWISE_ENTRIES,
            "vector dispatch",
            elementwise_evidence.audit_event,
            ElementwisePackageCache(root, packages / ELEMENTWISE_CACHE, target),
        ),
        lazy(
            arange_packages.ENTRIES,
            "arange dispatch",
            arange_evidence.audit_event,
            arange_packages.cache(root, packages / arange_packages.CACHE, target),
        ),
        owned(
            dynamic_offset_packages.ENTRIES,
            "dynamic offset",
            dynamic_offset_evidence.audit_event,
            dynamic_offset_packages.DynamicOffsetPackageCache(
                root, packages / dynamic_offset_packages.CACHE, target
            ),
        ),
        owned(
            masked_scatter_packages.ENTRIES,
            "masked scatter",
            masked_scatter_evidence.audit_event,
            masked_scatter_packages.MaskedScatterPackageCache(
                root, packages / masked_scatter_packages.CACHE, target
            ),
        ),
        owned(
            wide_reduction_packages.ENTRIES,
            "wide reduction",
            wide_reduction_evidence.audit_event,
            wide_reduction_packages.cache(
                root, packages / wide_reduction_packages.CACHE, target
            ),
        ),
        owned(
            wide_row_dispatch.ENTRIES,
            "wide row reduction",
            lambda event, cache: wide_row_evidence.audit_event(event, cache, modules),
            SmallRowPackageCache(root, packages / "small-rows", target),
        ),
        owned(
            wide_column_dispatch.ENTRIES,
            "wide column reduction",
            wide_column_evidence.audit_event,
            SmallColumnPackageCache(root, packages / SMALL_COLUMN_CACHE, target),
        ),
        owned(
            arg_reduce_layout.ENTRIES,
            "arg reduction",
            arg_reduce_evidence.audit_event,
            ArgReducePackageCache(root, packages / ARG_REDUCE_CACHE, target),
        ),
        owned(
            softmax_layout.ENTRIES,
            "softmax or logsumexp",
            softmax_evidence.audit_event,
            SoftmaxPackageCache(root, packages / SOFTMAX_CACHE, target),
        ),
        owned(
            sort_layout.ENTRIES,
            "sort",
            sort_evidence.audit_event,
            SortPackageCache(root, packages / SORT_CACHE, target),
        ),
        owned(
            searchsorted_layout.ENTRIES,
            "searchsorted",
            searchsorted_evidence.audit_event,
            SearchSortedPackageCache(root, packages / SEARCHSORTED_CACHE, target),
        ),
        owned(
            norm_layout.ENTRIES,
            "normalization",
            norm_evidence.audit_event,
            NormPackageCache(root, packages / NORM_CACHE, target),
        ),
        owned(
            cross_entropy_layout.ENTRIES,
            "cross-entropy",
            cross_entropy_evidence.audit_event,
            CrossEntropyPackageCache(root, packages / CROSS_ENTROPY_CACHE, target),
        ),
        (
            lambda entry, event: lazy_package_evidence.family(entry) is not None,
            None,
            # Derives the package again and checks scans' native modules.
            lambda event: lazy_package_evidence.audit_event(event, scans, modules),
            "scan",
        ),
        (
            lambda entry, event: compiled_evidence.is_compiled(event),
            None,
            # Finds the package of the traced source again.
            lambda event: compiled_evidence.audit_event(event, compiled),
            "owned",
        ),
        (
            lambda entry, event: quantization_evidence.is_quantization(entry),
            None,
            # Checks the complete groups, output guards and native modules.
            lambda event: quantization_evidence.audit_event(event, quantizations),
            "owned",
        ),
    )


def trace_audit(np, args, target, indexes, modules=None):
    """A ``TraceAudit`` of every per-dispatch check for these packages."""
    installed = set().union(*(index["descriptors"] for index in indexes.values()))
    return TraceAudit(event_audits(np, args, target, installed, modules), target)


def audit_trace(
    np, args, trace, target, indexes, reductions, *, require_indexing=True, record=None
):
    """Tie every dispatch to exactly one translated package.

    ``trace`` is any iterable of dispatch records; each is checked once and
    kept only in compact form (``trace_records.compact_trace_values``), so a
    trace of many gigabytes is audited in the memory of its largest record.
    ``record`` receives each compact record in order. ``require_indexing``
    requires the trace to reach a gather or scatter. Returns the number of
    indexing dispatches.
    """
    audit = trace_audit(np, args, target, indexes)
    events = []
    for event in trace:
        compact = audit.check(event)
        if record is not None:
            record(compact)
        events.append(compact)
    return audit_compact_trace(
        args,
        events,
        target,
        indexes,
        reductions,
        audit.summary(),
        require_indexing=require_indexing,
    )


def audit_compact_trace(
    args, trace, target, indexes, reductions, audit, *, require_indexing=True
):
    """Tie compact records, checked one by one by ``audit``, to their packages.

    ``audit`` is a ``TraceAudit.summary()``: the native worker checks each
    record whole as it is written and keeps only the compact ``trace``.
    Returns the number of indexing dispatches.
    """
    if audit.get("error") is not None:
        raise ValueError(f"The native trace audit failed: {audit['error']}")
    events = list(trace)
    owners, identities = list(audit["owners"]), audit["identities"]
    if not len(events) == len(owners) == len(identities):
        raise ValueError("The upstream op trace differs from its audit")
    _audit_packages(args, target, indexes, reductions, events, owners, identities)
    if require_indexing and not audit["indexing"]:
        raise ValueError("Upstream indexing tests did not reach gather or scatter")
    return audit["indexing"]


def _audit_packages(args, target, indexes, reductions, events, owners, identities):
    """Tie compact records to their packages once every record is checked.

    ``owners`` counts the checks that already own each record and
    ``identities`` marks the records whose native identity no check covered.
    """
    random_events = [event for event in events if event["entry"] in ENTRIES]
    if random_events:
        random_evidence.verify_packages(
            random_events, args.mlx_root, args.random, target
        )
    verify_native_identity(
        [event for event, unchecked in zip(events, identities) if unchecked], target
    )
    for name in FAMILIES:
        if name == "random":
            selected = [
                i for i, event in enumerate(events) if event["entry"] in ENTRIES
            ]
        else:
            descriptors = indexes[name]["descriptors"]
            selected = [
                i for i, event in enumerate(events) if event["entry"] in descriptors
            ]
            verify_artifacts(
                [events[i] for i in selected],
                getattr(args, name),
                indexes[name],
                variants=False,
            )
        for i in selected:
            owners[i] += 1
    for path, index in zip(args.reductions, reductions):
        keys = variant_keys(index)
        selected = [
            i
            for i, event in enumerate(events)
            if f'w{event["workgroupSize"][0]}/{event["entry"]}' in keys
        ]
        verify_artifacts(
            [events[i] for i in selected],
            path,
            index,
            variants=True,
            mlx_root=args.mlx_root,
        )
        for i in selected:
            owners[i] += 1
    if any(count != 1 for count in owners):
        raise ValueError("Upstream op trace contains an unknown or ambiguous package")


def run_worker(args, output, mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[4] / "tools/run_bounded_command.py"),
        "--timeout-seconds",
        str(TIMEOUTS[mode]),
        "--label",
        f"MLX upstream ops {mode}",
        "--",
        sys.executable,
        "-m",
        __package__ + ".verify_upstream_ops",
        "--worker",
        mode,
    ]
    for name in ("mlx_root", "packages", *FAMILIES[1:]):
        command.extend(
            ("--" + name.replace("_", "-"), str(getattr(args, name).resolve()))
        )
    for path in args.reductions:
        command.extend(("--reductions", str(path.resolve())))
    if args.group is not None:
        command.extend(("--group", args.group))
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    command.extend(("--output-dir", str(output / mode)))
    with (output / f"{mode}.stdout").open("w") as stdout:
        with (output / f"{mode}.stderr").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    record = {"command": command, "returncode": process.returncode}
    write_json(output / f"{mode}.command.json", record)
    return record


def verify(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(8), "little")
    before = verify_prepared(args.mlx_root)
    target = package_target(args.packages)
    tests = upstream_tests(target, args.group)
    checks = negative_checks(args.group)
    sources = upstream_test_sources(args.mlx_root, tests)
    write_json(output / "adaptation-before.json", before)
    target, indexes, reductions = load_indexes(args)
    evidence = {
        "commit": COMMIT,
        "target": target,
        "passed": False,
        "adaptation": before,
        "upstreamTestSources": sources,
        "group": args.group,
        "seed": args.seed,
        "upstreamTests": list(tests),
        "excludedUpstreamTests": dict(TARGET_EXCLUSIONS.get(target, {})),
        "workers": {},
        "fullTranslatedBackend": False,
        "fullUpstreamSuite": False,
    }
    try:
        for mode in ("cpu", "native", *checks):
            evidence["workers"][mode] = run_worker(args, output, mode)
            write_json(output / "evidence.json", evidence)
        if any(value["returncode"] for value in evidence["workers"].values()):
            raise RuntimeError(
                f"An upstream op worker failed with seed {args.seed}; "
                "inspect retained logs"
            )
        upstream = {}
        for mode in ("cpu", "native"):
            upstream[mode] = json.loads((output / mode / "upstream.json").read_text())
            validate_upstream(upstream[mode], native=mode == "native", inventory=tests)
        # The native worker checked each record as it wrote the compact
        # trace; long groups' full records run to tens of gigabytes.
        audit = json.loads((output / TRACE_AUDIT).read_text(encoding="utf-8"))
        trace = TraceStream(output / COMPACT_TRACE)
        indexing = audit_compact_trace(
            args,
            trace,
            target,
            indexes,
            reductions,
            audit,
            require_indexing=requires_indexing(args.group),
        )
        dispatches = sum(item["dispatchCount"] for item in upstream["native"]["tests"])
        if not dispatches == trace.records == audit["records"]:
            raise ValueError("Upstream op trace count differs")
        rejections = {}
        for mode, error in checks.items():
            item = json.loads((output / mode / "rejection.json").read_text())
            if (
                item.get("check") != mode
                or item.get("dispatchCount") != 0
                or error not in item.get("error", "")
            ):
                raise ValueError("Upstream op rejection evidence is incomplete")
            rejections[mode] = item
        if before != verify_prepared(args.mlx_root) or sources != upstream_test_sources(
            args.mlx_root, tests
        ):
            raise ValueError("MLX source changed during upstream op verification")
        evidence.update(
            passed=True,
            dispatchCount=trace.records,
            trace={key: audit[key] for key in ("records", "bytes", "sha256")},
            compactTrace=trace.summary(),
            indexingDispatchCount=indexing,
            upstream=upstream,
            negativeChecks=rejections,
        )
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        write_json(output / "evidence.json", evidence)
    return evidence


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "mlx-root",
        "packages",
        "random",
        "bitwise",
        "selection",
        "absolute",
        "integer64",
        "slice-updates",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reductions", type=Path, action="append", required=True)
    parser.add_argument("--worker", choices=("cpu", "native", *NEGATIVE_CHECKS))
    parser.add_argument("--group", choices=tuple(GROUPS))
    parser.add_argument(
        "--seed",
        type=upstream_test_seeds.parse_seed,
        help="the run seed test_seed derives each test's MLX seed from; "
        "a random one is recorded when omitted",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse()
    worker(arguments) if arguments.worker else verify(arguments)
