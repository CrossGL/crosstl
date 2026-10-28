"""Block-scaled matrix-matrix host routing, launch contracts and package configs."""

import ctypes
import re
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import fp_qmm_layout as layout
from demos.integrations.mlx.portable_host import fp_qmm_packages, qmm_layout, runtime
from demos.integrations.mlx.portable_host.quantization_layout import SOURCE_TYPES
from demos.integrations.mlx.tests.host.test_portable_fp_quantized_matmul import (
    CTYPES,
    contiguous,
)

HOST = Path(layout.__file__).parent
# (name, mode, dtype, M, N, K, batches, transpose)
CASES = (
    # K % 32 == 16 admits no split: qmm_t.
    ("tail-rows", "nvfp4", "float32", 50, 128, 1040, 1, True),
    ("tail-unaligned", "nvfp4", "float16", 33, 70, 528, 1, True),
    ("stacked-t", "mxfp8", "bfloat16", 40, 64, 256, 3, True),
    ("rows-n", "nvfp4", "float32", 50, 1040, 512, 1, False),
    ("stacked-n", "mxfp4", "float16", 4, 96, 64, 2, False),
    # 33 partitions of 32: 512 / 8 tiles capped at K / 32.
    ("split", "mxfp4", "float32", 50, 128, 1056, 1, True),
    ("split-half", "mxfp8", "bfloat16", 64, 256, 512, 1, True),
)


def case(name):
    return next(item[1:] for item in CASES if item[0] == name)


def native_binding(name):
    """What backend.cpp binds for one case: entry, counts, parameters and launch."""
    mode, dtype, m, n, k, batches, transpose = case(name)
    group, bits = layout.MODES[mode]
    entry = layout.select(mode, dtype, m, n, k, batches, transpose)
    split = qmm_layout.split_k(m, n, k, group) if "splitk" in entry else 1
    rows, columns = (n, k) if transpose else (k, n)
    words, groups = columns * bits // 32, columns // group
    lead = [batches] if batches > 1 else []
    counts = {
        "x": batches * m * k,
        "w": batches * rows * words,
        "scales": batches * rows * groups,
        "y": split * batches * m * n,
    }
    parameters = {"K": k, "N": n, "M": m}
    if split > 1:
        parameters.update(k_partition_size=k // split, split_k_partition_stride=m * n)
    elif batches > 1:
        parameters.update(
            x_batch_ndims=1,
            x_shape=[*lead, m, k],
            x_strides=contiguous([*lead, m, k]),
            w_batch_ndims=1,
            w_shape=[*lead, rows, words],
            w_strides=contiguous([*lead, rows, words]),
            s_strides=contiguous([*lead, rows, groups]),
        )
    execution = {
        "workgroupCount": [-(-n // 32), -(-m // 32), split * batches],
        "workgroupSize": [32, 2, 2],
    }
    return entry, counts, parameters, execution


def native_buffers(name):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    entry, counts, parameters, execution = native_binding(name)
    memory, buffers = {}, []
    for parameter in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, parameter)
        if parameter in counts:
            values = [0] * counts[parameter]
        else:
            value = parameters[parameter]
            values = [value] if isinstance(value, int) else value
        memory[parameter] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                parameter.encode(),
                dtype.encode(),
                ctypes.addressof(memory[parameter]),
                len(values),
                int(parameter == "y"),
            )
        )
    return entry, buffers, memory, execution


@pytest.mark.parametrize("dtype", tuple(SOURCE_TYPES.values()))
@pytest.mark.parametrize("mode", tuple(layout.MODES))
def test_signature_accepts_exactly_the_routed_instantiations(dtype, mode):
    """fp_quantized.metal's qmm_t, qmm_t_splitk and qmm_n entries without ``_hgs``."""
    group, bits = layout.MODES[mode]
    stem = f"{dtype}_gs_{group}_b_{bits}"
    accepted = [
        *(
            f"{mode}_qmm_t_{stem}_alN_{aligned}_batch_{batch}"
            for aligned in ("true", "false")
            for batch in (0, 1)
        ),
        *(f"{mode}_qmm_t_splitk_{stem}_alN_{aligned}" for aligned in ("true", "false")),
        *(f"{mode}_qmm_n_{stem}_batch_{batch}" for batch in (0, 1)),
    ]
    for entry in accepted:
        info = layout.signature(entry)
        assert entry.startswith(layout.ENTRY_PREFIXES)
        assert (info["groupSize"], info["bits"]) == (group, bits)
        assert layout.workgroup_size(entry) == (32, 2, 2)
        assert layout.transposed(entry) == (info["kernel"] != "qmm_n")
        assert layout.declares_global_scale(entry) == (info["kernel"] == "qmm_t")


@pytest.mark.parametrize(
    "entry",
    (
        "nvfp4_qmm_t_float_gs_32_b_4_alN_true_batch_0",
        "mxfp8_qmm_t_float_gs_32_b_4_alN_true_batch_0",
        "mxfp4_qmm_t_float_gs_32_b_4_batch_0",
        "mxfp4_qmm_t_float_gs_32_b_4_alN_true",
        "mxfp4_qmm_n_float_gs_32_b_4_alN_true_batch_0",
        "mxfp4_qmm_n_float_gs_32_b_4",
        "mxfp4_qmm_t_splitk_float_gs_32_b_4_alN_true_batch_0",
        "nvfp4_qmm_t_bfloat16_t_gs_16_b_4_alN_true_batch_0_hgs",
        "mxfp4_qmm_n_float_gs_32_b_4_batch_0_hgs",
        "affine_qmm_t_float_gs_32_b_4_alN_true_batch_0",
        "mxfp4_gather_qmm_t_float_gs_32_b_4_alN_true",
    ),
)
def test_signature_refuses_entries_quantized_matmul_never_selects(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_selection_follows_quantized_cpp_for_the_tail_shapes():
    """test_fp_qmm_non_multiple_of_32's shapes, as quantized.cpp routes them."""
    select = layout.select
    assert select("nvfp4", "float32", 50, 128, 1040, 1) == (
        "nvfp4_qmm_t_float_gs_16_b_4_alN_true_batch_0"
    )
    assert select("nvfp4", "float16", 33, 64, 528, 1) == (
        "nvfp4_qmm_t_float16_t_gs_16_b_4_alN_true_batch_0"
    )
    assert select("nvfp4", "bfloat16", 50, 128, 16, 1) == (
        "nvfp4_qmm_t_bfloat16_t_gs_16_b_4_alN_true_batch_0"
    )
    assert select("nvfp4", "float32", 50, 1040, 512, 1, transpose=False) == (
        "nvfp4_qmm_n_float_gs_16_b_4_batch_0"
    )
    assert select("mxfp4", "float32", 50, 128, 1056, 1) == (
        "mxfp4_qmm_t_splitk_float_gs_32_b_4_alN_true"
    )
    assert qmm_layout.split_k(50, 128, 1056, 32) == 33
    assert select("mxfp8", "float16", 50, 70, 1056, 2) == (
        "mxfp8_qmm_t_float16_t_gs_32_b_8_alN_false_batch_1"
    )
    with pytest.raises(ValueError, match="vector products"):
        select("mxfp4", "float32", 17, 128, 1056, 1)
    with pytest.raises(ValueError, match="vector products"):
        select("mxfp4", "float32", 3, 128, 1056, 1, transpose=False)


@pytest.mark.parametrize("name", [item[0] for item in CASES])
def test_plan_accepts_backend_launches_and_adds_inert_values(name):
    entry, counts, parameters, execution = native_binding(name)
    mode, _, m, n, k, batches, transpose = case(name)
    metadata = layout.plan(entry, counts, parameters, counts["y"], execution)
    assert (metadata["M"], metadata["N"], metadata["K"]) == (m, n, k)
    assert metadata["batches"] == batches and metadata["mode"] == mode
    assert metadata["overreadPadding"] == {"scales": 0}
    values = metadata["parameters"]
    assert ("global_scale" in values) == (metadata["kernel"] == "qmm_t")
    if metadata["kernel"] == "qmm_t":
        assert values["global_scale"] == [layout.INACTIVE_GLOBAL_SCALE]
    assert "b_strides" not in values and "biases" not in values
    if metadata["splitK"] > 1:
        assert not set(layout.BATCH_PARAMETERS) & set(values)
        assert values["k_partition_size"] * metadata["splitK"] == k
    else:
        assert set(layout.BATCH_PARAMETERS) <= set(values)
    if batches == 1 and metadata["splitK"] == 1:
        rows = n if transpose else k
        assert values["x_shape"] == [m, k] and values["w_batch_ndims"] == 0
        assert values["w_shape"][0] == rows


@pytest.mark.parametrize(
    "fault",
    (
        "workgroup",
        "tiles",
        "threads",
        "groups",
        "depth",
        "partition",
        "batch-missing",
        "unbatched-views",
        "bias-strides",
        "short-scales",
    ),
)
@pytest.mark.parametrize("name", ("tail-rows", "stacked-n", "split"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    entry, counts, parameters, execution = native_binding(name)
    info = layout.signature(entry)
    threads = counts["y"]
    if fault == "workgroup":
        execution["workgroupSize"] = [32, 2, 1]
    elif fault == "tiles":
        execution["workgroupCount"][0] += 1
    elif fault == "threads":
        threads += 1
    elif fault == "groups":
        parameters["K" if info["kernel"] != "qmm_n" else "N"] += 8
    elif fault == "depth":
        execution["workgroupCount"][2] += 1
    elif fault == "partition":
        if info["kernel"] == "qmm_t_splitk":
            parameters["k_partition_size"] += 32
        else:
            parameters["k_partition_size"] = parameters["K"]
    elif fault == "batch-missing":
        if info["batched"]:
            parameters.pop("s_strides")
        else:
            parameters["x_batch_ndims"] = 0
    elif fault == "unbatched-views":
        parameters["w_batch_ndims"] = 0
    elif fault == "bias-strides":
        parameters["b_strides"] = [1]
    else:
        counts["scales"] -= 1
    with pytest.raises(ValueError):
        layout.plan(entry, counts, parameters, threads, execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "names", "biases"))
def test_storage_rejects_native_identity_faults(fault):
    entry, buffers, memory, _ = native_buffers("stacked-t")
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    assert layout.read_storage(entry, supplied)[0]["scales"] > 0
    if fault == "dtype":
        supplied["scales"].dtype = b"bfloat16"
    elif fault == "direction":
        supplied["x"].output = 1
    elif fault == "names":
        supplied.pop("M")
    else:
        supplied["biases"] = supplied["scales"]
    with pytest.raises(ValueError):
        layout.read_storage(entry, supplied)


def test_package_configs_scope_the_affine_assertions_to_the_block_source(tmp_path):
    entries = {
        name: native_binding(name)[0] for name in ("tail-rows", "stacked-n", "split")
    }
    for entry in entries.values():
        affine = re.sub(r"^\w+?_qmm", "affine_qmm", entry).replace("_gs_16_", "_gs_32_")
        configs = {
            target: fp_qmm_packages.project_config(tmp_path, "out", target, entry)
            for target in ("metal", "directx", "opengl")
        }
        for target, config in configs.items():
            assert list(config.include_patterns) == [layout.SOURCE]
            rules = config.entry_workgroup_size_rules[layout.SOURCE]
            assert [int(size) for size in rules[entry]] == [32, 2, 2]
            assert [item.parameter for item in config.metadata_preconditions] == list(
                qmm_layout.bounded_parameters(affine)
            )
            assert {item.source for item in config.metadata_preconditions} == {
                layout.SOURCE
            }
            assert bool(config.workgroup_access_assertions) == (target != "metal")
            assert bool(config.index_range_assertions) == (target == "opengl")
            options = config.source_options["metal"]
            assert options["cooperative_matrix_fragment_mapping"] == "tile_4x4_row_pair"
        gl = configs["opengl"]
        assert len(gl.index_range_assertions) == len(
            qmm_layout.index_assertions(affine)
        )
        assert {item.source for item in gl.index_range_assertions} == {layout.SOURCE}
        (access,) = configs["directx"].workgroup_access_assertions
        (expected,) = qmm_layout.access_assertions(affine)
        assert access.function == expected["function"]
        assert access.source == layout.SOURCE and access.entry_point == entry


def test_runtime_routes_block_matrix_products_through_their_cache(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization, host.quantized_matmuls = None, None
    host.fp_quantized_matmuls, host.quantized_qmms = None, None
    host.fp_quantized_qmms = None
    entry = "nvfp4_qmm_t_float_gs_16_b_4_alN_true_batch_0"
    assert host._entry_available(entry.encode()) == 0
    host.fp_quantized_qmms = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"mxfp4_qmm_t_splitk_float_gs_32_b_4_alN_true") == 1
    hgs = b"mxfp4_qmm_t_float_gs_32_b_4_alN_true_batch_0_hgs"
    assert host._entry_available(hgs) == 0
    calls = []
    monkeypatch.setattr(
        runtime.matmul_dispatch,
        "dispatch_with",
        lambda host, layout, cache, entry, *args: calls.append((layout, cache, entry)),
    )
    host.dispatch(entry, [], 8, 1, launch=None)
    assert calls == [(layout, host.fp_quantized_qmms, entry)]


def test_backend_routes_every_mode_through_the_matrix_matrix_kernels():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    routing = backend[backend.index("void QuantizedMatmul::eval_gpu") :]
    qmm = routing[: routing.index("dispatch_fp_quantized_matvec(")]
    for fragment in (
        "if (M >= vector_limit) {",
        "std::optional<array> qmm_biases;",
        'const char* mode = affine ? "affine" : fp_mode;',
        "qmm_split_k(M, N, K, group_size_)",
        "dispatch_qmm_split_k(\n          x, w, scales, qmm_biases, out, mode,",
        "dispatch_qmm(\n          x, w, scales, qmm_biases, out, mode, transpose_,",
    ):
        assert fragment in qmm
    for name in ("void dispatch_qmm(", "void dispatch_qmm_split_k("):
        start = backend.index(name)
        body = backend[start : backend.index("\n}\n", start)]
        for fragment in (
            "const std::optional<array>& biases,",
            "const char* mode,",
            "std::string(mode) +",
            '{"scales", biases ? storage : "uint8"',
            'buffers.push_back({"biases", storage, data(*biases)',
        ):
            assert fragment in body
    start = backend.index("void dispatch_qmm(")
    body = backend[start : backend.index("\n}\n", start)]
    assert "QuantizedBatchViews(x, w, scales, *biases)" in body
    assert ": QuantizedBatchViews(x, w, scales);" in body
