"""Whole upstream MLX test modules that verify_upstream_ops runs by group.

Each tuple names unchanged methods of the pinned ``python/tests`` modules
that reach the GPU through the portable host: every method of these modules
that dispatches a translated kernel and passes on the native host, in the
order unittest loads them. Methods that dispatch nothing run in the
``host-only`` group (HOST_ONLY_TESTS), which must dispatch nothing; methods
that need a kernel family the host does not port yet are not listed.
"""

from demos.integrations.mlx.portable_host import upstream_subtest_shards


def _tests(module, case, names):
    return tuple(f"{module}.{case}.test_{name}" for name in names)


# test_nn: module state, weight files and every layer whose forward and
# backward passes reach the GPU, among them the convolutions, the
# normalization layers, the recurrent GRU, pooling, the quantized embedding
# and both upsampling modes.
NN_TESTS = _tests(
    "test_nn",
    "TestBase",
    (
        "circular_leaks",
        "load_from_weights",
        "model_with_dict",
        "module_attributes",
        "save_npz_weights",
        "save_safetensors_weights",
    ),
) + _tests(
    "test_nn",
    "TestLayers",
    (
        "batch_norm",
        "batch_norm_stats",
        "causal_mask",
        "celu",
        "conv1d",
        "conv2d",
        "elu",
        "gelu",
        "glu",
        "group_norm",
        "gru",
        "hard_shrink",
        "hard_tanh",
        "hardswish",
        "instance_norm",
        "leaky_relu",
        "log_sigmoid",
        "log_softmax",
        "mish",
        "pooling",
        "prelu",
        "quantized_embedding",
        "relu",
        "relu6",
        "sequential",
        "sigmoid",
        "sin_pe",
        "softmax",
        "softmin",
        "softplus",
        "softshrink",
        "softsign",
        "step",
        "upsample",
        "upsample_align_corners_one_dim",
    ),
)

# test_vmap, test_eval and test_threads: vectorized maps of the composed ops,
# sorts, partitions, scatters, convolutions, factorizations and views;
# synchronous, asynchronous and multi-stream evaluation and the buffer
# donation of no-op unary and copy chains; and thread-local
# streams and transforms traced on several threads at once.
TRANSFORM_TESTS = (
    _tests(
        "test_vmap",
        "TestVmap",
        (
            "binary",
            "broadcast_axes_vmap",
            "tree",
            "unary",
            "vmap_argreduce",
            "vmap_argsort",
            "vmap_concatenate",
            "vmap_const_func",
            "vmap_conv",
            "vmap_gather",
            "vmap_indexing",
            "vmap_inverse",
            "vmap_masked_scatter",
            "vmap_matmul",
            "vmap_mean",
            "vmap_pad",
            "vmap_partition",
            "vmap_reduce",
            "vmap_scatter",
            "vmap_sort",
            "vmap_strided_slice_single_element",
            "vmap_svd",
            "vmap_topk",
            "vmap_types",
            "vmap_view",
        ),
    )
    + _tests(
        "test_eval",
        "TestEval",
        (
            "async_eval",
            "async_eval_into_eval",
            "async_eval_into_eval_diff_stream",
            "async_eval_twice",
            "async_eval_with_multiple_streams",
            "donation_for_noops",
            "eval",
            "eval_mixed",
            "eval_slow_fast_multi_stream",
            "multi_output_eval_during_transform",
            "multistream_deadlock",
            "retain_graph",
        ),
    )
    + _tests(
        "test_threads",
        "TestThreads",
        ("concurrent_transforms", "threadlocal_stream"),
    )
)

# test_export_import and test_load: exported and imported functions, among
# them shapeless and quantized ones, and the NumPy, safetensors and GGUF
# round trips of device arrays.
SERIALIZATION_TESTS = _tests(
    "test_export_import",
    "TestExportImport",
    (
        "basic_export_import",
        "export_control_flow",
        "export_conv",
        "export_conv_shapeless",
        "export_import_metadata",
        "export_import_multi_with_constants",
        "export_import_scatter_sum",
        "export_import_shapeless",
        "export_kwarg_ordering",
        "export_matmul_shapeless_batch_and_mid_dim",
        "export_matmul_shapeless_mid_dim",
        "export_quantized_model",
        "export_random_sample",
        "export_scatter_gather",
        "export_searchsorted",
        "export_unique",
        "export_variable_inputs",
        "export_with_kwargs",
    ),
) + _tests(
    "test_load",
    "TestLoad",
    (
        "load_donation",
        "load_f8_e4m3",
        "load_gguf_quantized",
        "non_contiguous",
        "save_and_load",
        "save_and_load_empty",
        "save_and_load_fs",
        "save_and_load_gguf",
        "save_and_load_gguf_metadata_arrays",
        "save_and_load_gguf_metadata_basic",
        "save_and_load_gguf_metadata_mixed",
        "save_and_load_safetensors",
        "save_over_lazily_loaded_npy",
        "save_over_lazily_loaded_safetensors",
        "savez_and_loadz",
    ),
)

# test_optimizers, the random samplers no other verifier runs, the bfloat16
# operator sweeps, and the device, stream, memory, constant and copying
# asarray tests.
TRAINING_TESTS = (
    _tests(
        "test_optimizers",
        "TestOptimizers",
        (
            "adadelta",
            "adafactor",
            "adagrad",
            "adam",
            "compiled_optimizer",
            "lion",
            "muon",
            "optimizers",
            "rmsprop",
            "sgd",
            "update_lr_compiled",
            "weight_decay_keeps_inputs_unchanged",
        ),
    )
    + _tests(
        "test_optimizers",
        "TestSchedulers",
        (
            "clip_grad_norm",
            "compile_with_schedule",
            "cosine_decay",
            "decay_lr",
            "linear_warmup_with_cosine_decay",
            "multi_optimizer_with_parameterless_layers",
            "schedule_joiner",
        ),
    )
    + _tests(
        "test_random",
        "TestRandom",
        (
            "bernoulli",
            "categorical",
            "multivariate_normal",
            "normal_and_laplace",
            "permutation",
            "randint",
            "truncated_normal",
        ),
    )
    + _tests(
        "test_bf16",
        "TestBF16",
        ("binary_ops", "blas_ops", "reduction_ops", "unary_ops"),
    )
    + _tests("test_device", "TestDevice", ("op_on_device",))
    + _tests(
        "test_device",
        "TestStream",
        ("op_on_stream", "stream_context_exited_on_another_thread"),
    )
    + _tests("test_memory", "TestMemory", ("active_memory_count", "memory_info"))
    + _tests("test_constants", "TestConstants", ("newaxis_for_reshaping_arrays",))
    # asarray's copy=True copy of a million int32s, read back through a gather.
    + _tests("test_zero_copy", "TestZeroCopy", ("copy_true_copies",))
)

# test_fast's rotary embeddings and fused transforms, and the antialiased
# upsampling sweeps: thousands of launches each.
ROPE_TESTS = _tests(
    "test_fast",
    "TestFast",
    (
        "fast_transforms",
        "rope",
        "rope_batch",
        "rope_grad",
        "rope_single_batch",
        "rope_with_freqs",
        "rope_with_large_offset",
    ),
) + _tests(
    "test_upsample",
    "TestUpsample",
    ("antialias_1d_smoke", "antialias_non_integer_scale_smoke"),
)

# test_fft: 1-D and N-D transforms, zero-padded through complex64 slice
# updates, norms, strided and transposed inputs, and the vmap and jvp
# transforms; every Stockham, Rader and fused Bluestein size up to 4,096
# points in batches of 1, 3 and 32; four-step powers of two up to 2**24
# points; and multi-upload Bluestein sizes up to 18,247 points. The batched
# Bluestein test transforms 1,023 and 1,024 rows of 1,031 and 1,531 points,
# whose twiddles the generate_bluestein_twiddles kernel tabulates; it runs
# only where mx.metal.is_available() (METAL_PATH_TESTS). The frequency and
# shift tests run with the operations; test_fft_grads compares against
# PyTorch and runs with the other PyTorch-reference tests (TORCH_LAYER_TESTS).
FFT_TESTS = _tests(
    "test_fft",
    "TestFFT",
    (
        "fft",
        "fftn",
        "fft_norm",
        "fft_contiguity",
        "fft_vmap",
        "fft_jvp",
        "fft_throws",
        "fft_shared_mem",
        "fft_big_powers_of_two",
        "fft_large_numbers",
        "fft_into_ifft",
        "batched_bluestein_twiddle_table",
    ),
)

# test_ops: Walsh-Hadamard transforms of 2**k, 12 * 2**k, 20 * 2**k and
# 28 * 2**k points (hadamard.inc) in float32, float16 and bfloat16, strided
# inputs through the copy, the strided two-pass split above 8,192 points, and
# the transform's gradient under nested vmaps. test_hadamard skips itself
# when CI is set, so the worker clears it for that test (CI_CLEARED_TESTS);
# test_hadamard_m_only, the m stage alone, runs only where
# mx.metal.is_available() (METAL_PATH_TESTS).
HADAMARD_TESTS = _tests(
    "test_ops", "TestOps", ("hadamard", "hadamard_grad_vmap", "hadamard_m_only")
)

# test_fast_gated_delta: the recurrent gated-delta update and its backward
# (gated_delta.inc) on 32 heads and on 16 key heads shared by 32 or 48 value
# heads, each checked against MLX's composed fallback. The masked test takes
# the composed fallback on both sides, about eleven thousand launches, and the
# dtype test checks the output types of graphs it never evaluates, so it
# dispatches nothing (NO_DISPATCH_TESTS). test_gated_delta_vjp selects the
# kernels only where mx.metal.is_available() (METAL_PATH_TESTS); its composed
# reference takes about twenty-six thousand launches, half an hour natively on
# Metal, so it runs in a job of its own.
GATED_DELTA_TESTS = _tests(
    "test_fast_gated_delta",
    "TestGatedDelta",
    ("gated_delta_dtypes", "gated_delta_fallback_masked"),
)
GATED_DELTA_VJP_TESTS = _tests(
    "test_fast_gated_delta", "TestGatedDelta", ("gated_delta_vjp",)
)
# The kernels mx.fast.metal_kernel builds from Metal source
# (custom_kernel.inc): plain, templated, strided and helper-calling kernels,
# thread attributes, a source changed under the same name, in one evaluation
# or two, both math modes, inputs of two dtypes, a kernel that does not
# compile, whose failure must leave the arrays evaluated before it intact, and
# custom kernels in exported functions, evaluated or only traced.
CUSTOM_KERNEL_TESTS = (
    _tests(
        "test_fast",
        "TestFast",
        (
            "custom_kernel_args",
            "custom_kernel_attributes",
            "custom_kernel_basic",
            "custom_kernel_caching",
            "custom_kernel_helper",
            "custom_kernel_mixed_dtypes",
            "custom_kernel_same_name_different_source_one_eval",
            "custom_kernel_strides",
            "custom_metal_kernel_math_mode",
        ),
    )
    + _tests("test_eval", "TestEval", ("eval_exception_does_not_corrupt_state",))
    + _tests(
        "test_export_import",
        "TestExportImport",
        (
            "export_custom_metal_kernel_with_math_mode",
            "export_custom_metal_kernel_without_evaluation",
            "export_import_custom_kernel",
        ),
    )
)
# Tests the pinned suite runs only where mx.metal.is_available().
METAL_PATH_TESTS = (
    "test_conv.TestConv.test_conv2d_winograd_batch_tiling",
    "test_conv_transpose.TestConvTranspose.test_conv_transpose_unfold_tiling",
    "test_fast_gated_delta.TestGatedDelta.test_gated_delta_vjp",
    "test_fft.TestFFT.test_batched_bluestein_twiddle_table",
    "test_ops.TestOps.test_hadamard_m_only",
    "test_quantized.TestQuantized.test_gather_qmm_sorted_nax_large_m",
    "test_quantized.TestQuantized.test_qqmm_global_scale_matrix_kernels",
    *(
        name
        for name in CUSTOM_KERNEL_TESTS
        if name.rsplit(".", 1)[1]
        in {
            "test_custom_kernel_caching",
            "test_custom_kernel_mixed_dtypes",
            "test_custom_kernel_same_name_different_source_one_eval",
            "test_custom_metal_kernel_math_mode",
            "test_eval_exception_does_not_corrupt_state",
        }
    ),
)
# Tests whose bodies write their kernel's source in Metal only where
# mx.metal.is_available(), and in CUDA otherwise; the native host executes
# Metal sources translated, so the native worker runs them while
# is_available reports the host's GPU. The export tests take the path that
# evaluates their kernel there and, on the CPU worker, check MLX's refusal
# without a Metal back end (METAL_SOURCE_CPU_TESTS); the others need a GPU.
METAL_SOURCE_TESTS = tuple(
    name for name in CUSTOM_KERNEL_TESTS if name not in METAL_PATH_TESTS
)
METAL_SOURCE_CPU_TESTS = tuple(
    name
    for name in METAL_SOURCE_TESTS
    if name.rsplit(".", 1)[1].startswith("test_export_custom_metal_kernel_")
)

# test_fast_sdpa: queries of at most eight positions, which
# scaled_dot_product_attention.cpp runs through the vector kernels in one pass
# or, for long grouped queries, in two (sdpa.inc), and the inf-score test,
# whose four-wide heads take MLX's composed fallback. The other tests reach
# the steel full-attention kernels (ATTENTION_TESTS) or run only where
# mx.metal.is_available(). test_sdpa_vector_gqa_long draws about 10^8 normal
# samples through the translated random-bits kernel, whose evidence records
# every byte, and does not finish within the native worker's bound.
SDPA_TESTS = _tests(
    "test_fast_sdpa",
    "TestFastSDPA",
    (
        "sdpa_vector_kv_transposed_head_seq",
        "sdpa_vector",
        "sdpa_inf_score",
        "sdpa_few_query",
        "sdpa_vector_batched",
        "sdpa_blocks_env_override",
    ),
)

# test_fast_sdpa: queries of more than eight positions, which
# scaled_dot_product_attention.cpp runs through the steel full-attention
# kernel (sdpa.inc), and the gradients, which run its fused VJP kernels after
# a forward that saves the logsumexp. The host has no NAX, so the padded head
# dims and the NAX kernels never apply. Each head-dim test runs 48 subtests
# of about 70 dispatches, so the tests are split in groups that each fit the
# native worker's bound: the head dims 72 and 80, the head dim 96 with the
# small and gradient tests, and the tests that draw tens of millions of normal
# samples. test_sdpa alone runs a subtest for each of its 240 dtype, layout,
# mask and shape cases, longer than the bound, so each of three sweep groups
# runs every third case (upstream_subtest_shards).
ATTENTION_TESTS = _tests(
    "test_fast_sdpa",
    "TestFastSDPA",
    (
        "sdpa_head_dim_72",
        "sdpa_head_dim_80",
        "sdpa_head_dim_72_80_sinks",
        "sdpa_fully_masked",
    ),
)
ATTENTION_2_TESTS = _tests(
    "test_fast_sdpa",
    "TestFastSDPA",
    (
        "sdpa_head_dim_96",
        "sdpa_broadcast_mask",
        "sdpa_noncontiguous_inputs",
        "sdpa_promote_mask",
        "sdpa_nan_bug",
        "sdpa_grad",
    ),
)
ATTENTION_3_TESTS = _tests(
    "test_fast_sdpa", "TestFastSDPA", ("sdpa_attention_sinks", "sdpa_sliced")
)
SWEEP_SHARDS = 3
ATTENTION_SWEEP_TESTS = upstream_subtest_shards.shard_names(
    "test_fast_sdpa.TestFastSDPA.test_sdpa", SWEEP_SHARDS
)

# Tests that compare the GPU's results with PyTorch's on the CPU: the 1-D,
# 2-D, 3-D, depthwise, general and asymmetrically padded convolutions, the
# transposed ones with and without output padding, and their gradients; then
# batch normalization, AdamW, nearest, linear and cubic upsampling with and
# without antialiasing, bfloat16 conversions from torch tensors and ml_dtypes
# arrays, the cumulative max and min VJPs and the FFT gradients. They skip
# themselves without torch (or ml_dtypes), so only their jobs install the
# pinned CPU wheel and ml_dtypes, and the worker refuses to run them without
# those (TORCH_GROUPS, REFERENCE_MODULE_TESTS).
TORCH_CONV_TESTS = _tests(
    "test_conv",
    "TestConv",
    (
        "asymmetric_padding",
        "torch_conv_1D",
        "torch_conv_1D_grad",
        "torch_conv_2D",
        "torch_conv_2D_grad",
        "torch_conv_3D",
        "torch_conv_3D_grad",
        "torch_conv_depthwise",
        "torch_conv_general",
    ),
) + _tests(
    "test_conv_transpose",
    "TestConvTranspose",
    (
        "torch_conv_tranpose_1d_output_padding",
        "torch_conv_transpose_1D",
        "torch_conv_transpose_1D_grad",
        "torch_conv_transpose_2D",
        "torch_conv_transpose_2D_grad",
        "torch_conv_transpose_2d_output_padding",
        "torch_conv_transpose_3D",
        "torch_conv_transpose_3D_grad",
        "torch_conv_transpose_3d_output_padding",
    ),
)
TORCH_LAYER_TESTS = (
    _tests("test_bf16", "TestBF16", ("conversion", "conversion_ml_dtypes"))
    + _tests("test_autograd", "TestAutograd", ("cummax_cummin_grad_vs_torch",))
    + _tests("test_fft", "TestFFT", ("fft_grads",))
    + _tests("test_nn", "TestLayers", ("batch_norm_matches_torch",))
    + _tests("test_optimizers", "TestOptimizers", ("adamw_matches_pytorch",))
    + _tests(
        "test_upsample",
        "TestUpsample",
        (
            "antialias_upscale_cubic_matches_pytorch",
            "antialias_upscale_linear_is_noop",
        ),
    )
)
# The nearest, linear and cubic upsampling sweeps, with and without
# antialiasing, launch about 36,000 kernels between them (28,000 for the
# antialiased one), so they run in a job of their own.
TORCH_UPSAMPLE_TESTS = _tests(
    "test_upsample", "TestUpsample", ("torch_upsample", "torch_upsample_antialias")
)
TORCH_GROUPS = ("torch-conv", "torch-layers", "torch-upsample")

# nvfp4 dequantization with broadcast, zero-stride scales against contiguous
# ones (quantize_strided); then test_quantized's products whose operands
# QQMatmul and GatherQQMM quantize: the block quantize and quantize-dequantize
# kernels, dispatch_qmv's mxfp8 and nvfp4 routes and gather_qmv's, among them
# the kernels that read the weights' global scales, and qmm's for QQMatmul's globally scaled bfloat16
# matrix products (qqmm_global_scale_matrix_kernels, which runs only where
# mx.metal.is_available(): METAL_PATH_TESTS). GatherQMM's gathered products
# of quantized weights follow: gather_qmv, gather_qvm and gather_qmm for
# affine and block-scaled weights, in float32 and, for the matrix path,
# bfloat16; then the sorted single-row products gather_qmm_rhs multiplies,
# among them GatherQQMM's globally scaled bfloat16 products
# (gather_qqmm_global_scale_matrix_paths).
QQ_MATMUL_TESTS = _tests(
    "test_quantized",
    "TestQuantized",
    (
        "quantize_strided",
        "qqmm",
        "qqmv",
        "qqmm_global_scale_matrix_kernels",
        "gather_qqmm",
        "gather_qmm",
        "gather_matmul_grad",
        "gather_qmm_matrix_path",
        "gather_qmm_sorted_sliced_weight",
        "gather_qmm_sorted_unaligned_k",
        "gather_qmm_grad",
        "gather_qqmm_global_scale_matrix_paths",
    ),
)
# gather_qmm_rhs's sorted products over every quantization mode, about a
# quarter of an hour natively on Metal, and the 32,832-row product, which
# test_gather_qmm_sorted_nax_large_m runs only where mx.metal.is_available()
# (METAL_PATH_TESTS); without NAX it takes gather_qmm_rhs too.
QUANTIZED_SORTED_TESTS = _tests(
    "test_quantized",
    "TestQuantized",
    ("gather_qmm_sorted", "gather_qmm_sorted_nax_large_m"),
)
# The nvfp4 products whose K or N leaves a 16-element tail, sorted through
# gather_qmm_rhs and tiled through gather_qmm, in float32, float16 and
# bfloat16 on the GPU; their operands of up to two million elements take
# over half an hour natively on Metal.
QUANTIZED_FP_TAIL_TESTS = _tests(
    "test_quantized", "TestQuantized", ("fp_gather_qmm_non_multiple_of_32",)
)
# QuantizedMatmul's products that the dedicated qmm verifiers do not model:
# the batched affine and mxfp4 small-M blocks of qmm (test_qmm_small_m_block,
# which skips on MLX's CPU), the nvfp4 16-element K tail differentiated in
# both transpositions, and the gradient of an affine product with respect to
# its scales and biases. About 4,200 launches, in a job of their own.
QUANTIZED_MATMUL_TESTS = _tests(
    "test_quantized",
    "TestQuantized",
    ("qmm_small_m_block", "fp_qmm_non_multiple_of_32_vjp", "vjp_scales_biases"),
)
QUANTIZED_GROUPS = (
    "quantized-products",
    "quantized-sorted",
    "quantized-fp-tails",
    "quantized-matmul",
)

# The three leak tests compare MLX's active memory before and after compiling,
# exporting or vectorizing closures only where mx.metal.is_available(); the
# native worker runs them while it reports the host's GPU, whose allocator
# counts the buffers, and the CPU worker runs them unchanged
# (verify_upstream_ops METAL_SOURCE_TESTS).
HOST_MEMORY_TESTS = (
    "test_compile.TestCompile.test_leaks",
    "test_export_import.TestExportImport.test_leaks",
    "test_vmap.TestVmap.test_leaks",
)
# The device context and GPU device_info tests skip unless
# mx.is_available(mx.gpu), so the CPU worker, whose host registers no runtime,
# skips them, and the native worker runs them without a kernel.
HOST_GPU_TESTS = (
    "test_device.TestDevice.test_device_context",
    "test_device.TestDeviceInfo.test_device_info_gpu",
)
# Methods of the routed modules that run on the portable host but reach no
# kernel: they trace, transform, compile or export graphs whose shapes, types
# and identities they check without evaluating them, reject invalid arguments
# before building a graph, evaluate only on the CPU stream (the linear algebra
# solvers, NumPy loads and their read errors), read constants, devices, streams
# and buffer sizes, or construct modules, initializers, optimizers and
# schedulers. They also convert arrays that hold their data to and from Python
# scalars, lists, NumPy, buffers and CPU DLPack capsules, promote and inspect
# dtypes, print arrays and graphs, refuse float64 work on the GPU stream (the
# test_double methods, which compute in float64 only on the CPU), map trees,
# and refuse zero-copy adoption without Metal. The native worker must dispatch
# nothing for any of them (verify_upstream_ops NO_DISPATCH_TESTS), so a lazy
# graph that starts to evaluate fails the group.
HOST_ONLY_TESTS = (
    _tests(
        "test_array",
        "TestArray",
        (
            "array_basics",
            "array_iteration",
            "array_namespace",
            "array_namespace_asarray",
            "array_np_conversion",
            "array_np_shape_dim_check",
            "array_repr",
            "array_to_list",
            "asarray_copy",
            "bool_conversion",
            "create_from_buffer",
            "double_keeps_precision",
            "dtype_promotion",
            "dtype_python_scalar_promotion",
            "from_dlpack_cpu",
            "from_dlpack_cpu_strided",
            "inplace_preserves_ids",
            "int_type",
            "large_indices",
            "scalar_integer_conversion_overflow",
            "slice_negative_step",
            "to_scalar",
        ),
    )
    # The namespace-information case names its one method plain ``test``.
    + ("test_array.TestArrayNamespsceInfo.test",)
    + _tests(
        "test_array",
        "TestDtypes",
        (
            "can_cast",
            "dtypes",
            "finfo",
            "iinfo",
            "index_conversion",
            "index_conversion_invalid",
            "isdtype",
            "result_type",
            "scalar_conversion",
        ),
    )
    + _tests("test_array", "TestEquality", ("list_equals_array", "tuple_equals_array"))
    + _tests(
        "test_array",
        "TestInequality",
        (
            "dlx_device_type",
            "invalid_op_on_array",
            "list_not_equals_array",
            "obj_inequality_array",
            "tuple_not_equals_array",
        ),
    )
    + _tests("test_array", "TestVersion", ("version",))
    + _tests(
        "test_autograd",
        "TestAutograd",
        (
            "conv_second_order_grad_with_padding",
            "flatten_unflatten_vjps",
            "grad_ids_pre_post",
            "grad_with_copies",
            "grad_with_inplace_update",
            "index_vjp_requires_stop_gradient",
            "jvp_comparison_tangent_dtype",
            "leaks",
            "vjp_types",
        ),
    )
    + _tests("test_bf16", "TestBF16", ("arithmetic_reduction_ops",))
    + _tests(
        "test_compile",
        "TestCompile",
        (
            "compile_donates_input_buffer",
            "compile_rng_across_threads",
            "compile_shapeless_with_broadcast",
            "compile_without_captured_inputs",
            "compiled_preserves_attributes",
            "double_constant",
            "enable_disable",
            "leaks",
            "shapeless_compile_full_like",
            "shapeless_compile_gather",
            "shapeless_compile_unflatten",
            "shapeless_compile_with_reshape",
            "unsupported_input_types",
            "wrap_compiled",
        ),
    )
    + _tests(
        "test_constants",
        "TestConstants",
        ("constants_availability", "constants_values"),
    )
    + _tests("test_device", "TestDefaultDevice", ("mlx_default_device",))
    + _tests("test_device", "TestDevice", ("device", "device_context"))
    + _tests(
        "test_device",
        "TestDeviceInfo",
        (
            "device_count",
            "device_info_cpu",
            "device_info_default",
            "device_info_gpu",
        ),
    )
    + _tests("test_device", "TestStream", ("reentrant_stream_context", "stream"))
    + _tests(
        "test_double",
        "TestDouble",
        (
            "binary_ops",
            "conversion",
            "gemm",
            "get_and_set_item",
            "lapack",
            "linspace",
            "python_float_keeps_double_precision",
            "reductions",
            "type_promotion",
            "unary_ops",
            "where",
        ),
    )
    + _tests(
        "test_eval",
        "TestEval",
        ("async_eval_error_in_synchronize", "async_eval_in_trace"),
    )
    + _tests(
        "test_export_import", "TestExportImport", ("export_with_callback", "leaks")
    )
    + _tests(
        "test_fast", "TestFast", ("cross_entropy_shape_checks", "rope_dims_validation")
    )
    + _tests("test_graph", "TestGraph", ("to_dot",))
    + _tests(
        "test_init",
        "TestInit",
        (
            "constant",
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "he_uniform",
            "normal",
        ),
    )
    + _tests("test_linalg", "TestLinalg", ("solve_triangular",))
    + _tests(
        "test_load",
        "TestLoad",
        (
            "async_load_npy_read_error_across_streams",
            "load_npy_dtype",
            "load_npy_read_error",
        ),
    )
    + _tests("test_memory", "TestMemory", ("array_buffer_size",))
    + _tests(
        "test_nn",
        "TestBase",
        (
            "chaining",
            "freeze_strict_keys",
            "grad_of_module",
            "module_state",
            "module_utilities",
            "parameter_deletion",
            "quantize",
            "quantize_freeze",
            "quantized_sharded_linear_construction",
            "update",
            "update_modules",
        ),
    )
    + _tests(
        "test_nn",
        "TestLayers",
        (
            "alibi",
            "attention",
            "bilinear",
            "conv_transpose_extra_repr",
            "dropout",
            "dropout2d",
            "dropout3d",
            "identity",
            "linear",
            "lstm",
            "norm_eps_validation",
            "rnn",
            "rope",
            "set_dtype",
            "transformer",
            "transformer_custom_modules",
            "transformer_decoder",
            "transformer_decoder_layer",
            "transformer_encoder",
            "transformer_encoder_layer",
        ),
    )
    + _tests(
        "test_optimizers",
        "TestOptimizers",
        ("epsilon_validation", "optimizer_state", "types_conserved"),
    )
    + _tests(
        "test_optimizers",
        "TestSchedulers",
        ("exponential_decay", "init_from_state", "multi_optimizer", "step_decay"),
    )
    + _tests(
        "test_quantized",
        "TestQuantized",
        ("fp_vjp_scales_throws", "mode_error_cases", "qmm_fp_type"),
    )
    + _tests("test_random", "TestRandom", ("complex_normal",))
    + _tests(
        "test_tree",
        "TestTreeUtils",
        (
            "empty_subtree_merge",
            "tree_flatten",
            "tree_map",
            "tree_unflatten_integer_key_collision",
        ),
    )
    + _tests(
        "test_vmap",
        "TestVmap",
        (
            "basics",
            "leaks",
            "mismatch_input_sizes",
            "vmap_flatten",
            "vmap_put_along_axis",
            "vmap_split_vmap",
            "vmap_take_along_axis",
        ),
    )
    # Without Metal, asarray must refuse copy=False; the adoption tests skip.
    + _tests(
        "test_zero_copy",
        "TestZeroCopy",
        (
            "copy_false",
            "copy_false_dtype_conversion_raises",
            "default_shares_or_copies",
        ),
    )
)

# verify_upstream_ops.GROUP_TESTS: each runs in a CI job of its own.
GROUPS = {
    "nn": NN_TESTS,
    "transforms": TRANSFORM_TESTS,
    "serialization": SERIALIZATION_TESTS,
    "training": TRAINING_TESTS,
    "rope": ROPE_TESTS,
    "fft": FFT_TESTS + HADAMARD_TESTS,
    "gated-delta": GATED_DELTA_TESTS,
    "gated-delta-vjp": GATED_DELTA_VJP_TESTS,
    "sdpa": SDPA_TESTS,
    "attention": ATTENTION_TESTS,
    "attention-2": ATTENTION_2_TESTS,
    "attention-3": ATTENTION_3_TESTS,
    **{
        f"attention-sweep-{index}": (name,)
        for index, name in enumerate(ATTENTION_SWEEP_TESTS, start=1)
    },
    "custom-kernels": CUSTOM_KERNEL_TESTS,
    "torch-conv": TORCH_CONV_TESTS,
    "torch-layers": TORCH_LAYER_TESTS,
    "torch-upsample": TORCH_UPSAMPLE_TESTS,
    "quantized-products": QQ_MATMUL_TESTS,
    "quantized-sorted": QUANTIZED_SORTED_TESTS,
    "quantized-fp-tails": QUANTIZED_FP_TAIL_TESTS,
    "quantized-matmul": QUANTIZED_MATMUL_TESTS,
    "host-only": HOST_ONLY_TESTS,
}
