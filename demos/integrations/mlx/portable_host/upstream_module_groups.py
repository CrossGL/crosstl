"""Whole upstream MLX test modules that verify_upstream_ops runs by group.

Each tuple names unchanged methods of the pinned ``python/tests`` modules
that reach the GPU through the portable host: every method of these modules
that dispatches a translated kernel and passes on the native host, in the
order unittest loads them. Methods that dispatch nothing, and methods that
need a kernel family the host does not port yet, are not listed.
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

# test_vmap and test_eval: vectorized maps of the composed ops, sorts,
# partitions, scatters, factorizations and views, and synchronous,
# asynchronous and multi-stream evaluation.
TRANSFORM_TESTS = _tests(
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
) + _tests(
    "test_eval",
    "TestEval",
    (
        "async_eval",
        "async_eval_into_eval",
        "async_eval_into_eval_diff_stream",
        "async_eval_twice",
        "async_eval_with_multiple_streams",
        "eval",
        "eval_mixed",
        "eval_slow_fast_multi_stream",
        "multi_output_eval_during_transform",
        "multistream_deadlock",
        "retain_graph",
    ),
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
# operator sweeps, and the device, stream, memory and constant tests.
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
# points; and multi-upload Bluestein sizes up to 18,247 points. The
# frequency and shift tests run with the operations; test_fft_grads compares
# against PyTorch, which the host environment does not install.
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
    ),
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

# verify_upstream_ops.GROUP_TESTS: each runs in a CI job of its own.
GROUPS = {
    "nn": NN_TESTS,
    "transforms": TRANSFORM_TESTS,
    "serialization": SERIALIZATION_TESTS,
    "training": TRAINING_TESTS,
    "rope": ROPE_TESTS,
    "fft": FFT_TESTS,
    "sdpa": SDPA_TESTS,
    "attention": ATTENTION_TESTS,
    "attention-2": ATTENTION_2_TESTS,
    "attention-3": ATTENTION_3_TESTS,
    **{
        f"attention-sweep-{index}": (name,)
        for index, name in enumerate(ATTENTION_SWEEP_TESTS, start=1)
    },
}
