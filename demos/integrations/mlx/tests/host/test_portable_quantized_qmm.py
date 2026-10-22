"""Affine matrix-matrix host routing, launch contracts and package configs."""

import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import matmul_dispatch, matmul_layout
from demos.integrations.mlx.portable_host import qmm_layout as layout
from demos.integrations.mlx.portable_host import qmm_packages, runtime
from demos.integrations.mlx.portable_host import verify_quantized_qmm as verify
from demos.integrations.mlx.portable_host.verify_quantized_matmul import case_data
from demos.integrations.mlx.tests.host.test_portable_quantized_matmul import (
    CTYPES,
    contiguous,
    span,
)

HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
GATES = {
    "qmm_t": "test_current_quantized_qmm_t",
    "qmm_n": "test_current_quantized_qmm_n",
    "qmm_t_splitk": "test_current_quantized_qmm_t_splitk",
}


def gate(kernel):
    import importlib

    return importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.{GATES[kernel]}"
    )


def native_binding(record):
    """What backend.cpp binds for one verification case: counts and parameters."""
    k, n, m, batches = record["K"], record["N"], record["M"], record["batches"]
    rows, columns = (n, k) if record["transpose"] else (k, n)
    words, groups = columns * record["bits"] // 32, columns // record["groupSize"]
    x_shape = list(record["xShape"])
    x_strides = contiguous(x_shape)
    if record["layout"] == "strided":
        x_strides[0] *= 2
    w_batch = record["wBatchShape"]
    out_batch = x_shape[:-2] if w_batch else []
    w_shape = [*out_batch, rows, words]
    strides = {}
    for name, width in (("w", words), ("s", groups), ("b", groups)):
        dense = contiguous([*w_batch, rows, width])
        strides[name] = [
            0 if extent == 1 and target > 1 else stride
            for extent, target, stride in zip(w_batch, out_batch, dense)
        ] + dense[len(w_batch) :]
    split = record["splitK"]
    counts = {
        "x": span(x_shape, x_strides),
        "w": math.prod(w_batch) * rows * words,
        "scales": math.prod(w_batch) * rows * groups,
        "biases": math.prod(w_batch) * rows * groups,
        "y": split * batches * m * n,
    }
    parameters = {"K": k, "N": n, "M": m}
    if split > 1:
        parameters.update(k_partition_size=k // split, split_k_partition_stride=m * n)
    elif batches > 1:
        parameters.update(
            x_batch_ndims=len(x_shape) - 2,
            x_shape=x_shape,
            x_strides=x_strides,
            w_batch_ndims=len(w_shape) - 2,
            w_shape=w_shape,
            w_strides=strides["w"],
            s_strides=strides["s"],
            b_strides=strides["b"],
        )
    execution = {
        "workgroupCount": [-(-n // 32), -(-m // 32), split * batches],
        "workgroupSize": [32, 2, 2],
    }
    return counts, parameters, execution


def native_buffers(record):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    counts, parameters, execution = native_binding(record)
    entry, memory, buffers = record["entry"], {}, []
    for name in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, name)
        if name in counts:
            values = [0] * counts[name]
        else:
            value = parameters[name]
            values = [value] if isinstance(value, int) else value
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == "y"),
            )
        )
    return buffers, memory, execution


@pytest.mark.parametrize("dtype", ("float", "float16_t", "bfloat16_t"))
@pytest.mark.parametrize("group", (32, 64, 128))
@pytest.mark.parametrize("bits", (2, 3, 4, 5, 6, 8))
def test_signature_accepts_exactly_the_instantiated_entries(dtype, group, bits):
    stem = f"{dtype}_gs_{group}_b_{bits}"
    accepted = [
        *(
            f"affine_qmm_t_{stem}_alN_{aligned}_batch_{batch}"
            for aligned in ("true", "false")
            for batch in (0, 1)
        ),
        *(f"affine_qmm_t_splitk_{stem}_alN_{aligned}" for aligned in ("true", "false")),
        *(f"affine_qmm_n_{stem}_batch_{batch}" for batch in (0, 1)),
    ]
    for entry in accepted:
        info = layout.signature(entry)
        assert entry.startswith(layout.ENTRY_PREFIXES)
        assert not entry.startswith(matmul_layout.ENTRY_PREFIXES)
        assert layout.workgroup_size(entry) == (32, 2, 2)
        assert layout.transposed(entry) == (info["kernel"] != "qmm_n")
    for entry in (
        f"affine_qmm_t_{stem}_batch_0",
        f"affine_qmm_t_{stem}_alN_true",
        f"affine_qmm_t_splitk_{stem}_alN_true_batch_0",
        f"affine_qmm_n_{stem}_alN_true_batch_0",
        f"affine_qmm_n_{stem}",
        f"affine_qmm_t_{stem}_alN_true_batch_0_hgs",
        f"affine_qmm_t_nax_{stem}_alN_true_batch_0",
        f"affine_gather_qmm_t_{stem}_alN_true",
        f"mxfp4_qmm_t_{stem}_alN_true_batch_0",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


@pytest.mark.parametrize("kernel", ("qmm_t", "qmm_n", "qmm_t_splitk"))
def test_signature_accepts_every_gate_entry(kernel):
    module = gate(kernel)
    for entry, info in module.INFO.items():
        signature = layout.signature(entry)
        assert signature["kernel"] == kernel
        assert (signature["dtype"], signature["groupSize"], signature["bits"]) == (
            info["dtype"],
            info["groupSize"],
            info["bits"],
        )
        assert signature["batched"] == (info["variant"] == "batch_1")


def test_split_follows_the_qmm_t_gate_and_quantized_cpp():
    module = gate("qmm_t")
    for m in (18, 30, 33, 64, 129, 600):
        for n in (8, 32, 99, 256, 1024):
            for k in (32, 128, 256, 2048, 8192):
                for group in (32, 64, 128):
                    assert layout.split_k(m, n, k, group) == module._split_k(
                        m, n, k, group
                    )
    # One tile: 512 partitions, at most K / max(group, 32) and dividing K.
    assert layout.split_k(32, 32, 2048, 32) == 64
    assert layout.split_k(32, 32, 2048, 128) == 16
    assert layout.split_k(32, 32, 96, 32) == 3
    assert layout.split_k(30, 8, 256, 64) == 4
    assert layout.split_k(129, 99, 256, 64) == 4
    assert layout.split_k(40, 96, 128, 128) == 1
    # 1,024 tiles leave no split.
    assert layout.split_k(1024, 1024, 4096, 32) == 1


def test_selection_follows_quantized_matmul_for_upstream_shapes():
    select = layout.select
    # test_small_matrix: (3, 10, 256) rows over 2-D weights fold to 30 rows.
    assert select("float32", 64, 4, 30, 8, 256, 1) == (
        "affine_qmm_t_splitk_float_gs_64_b_4_alN_false"
    )
    assert select("float32", 64, 4, 30, 256, 8, 1, False) == (
        "affine_qmm_n_float_gs_64_b_4_batch_0"
    )
    assert select("float32", 64, 4, 10, 256, 8, 3, False) == (
        "affine_qmm_n_float_gs_64_b_4_batch_1"
    )
    # test_non_multiples: 129 rows of 99 outputs split 4 ways; 10 rows of an
    # untransposed K of 33 take qmm_n.
    assert select("float32", 64, 4, 129, 99, 256, 1) == (
        "affine_qmm_t_splitk_float_gs_64_b_4_alN_false"
    )
    assert select("float32", 64, 4, 10, 256, 33, 1, False) == (
        "affine_qmm_n_float_gs_64_b_4_batch_0"
    )
    # Batched transposed products never split; unsplittable K takes qmm_t.
    assert select("float16", 32, 8, 33, 40, 96, 3) == (
        "affine_qmm_t_float16_t_gs_32_b_8_alN_false_batch_1"
    )
    assert select("bfloat16", 128, 6, 40, 96, 128, 1) == (
        "affine_qmm_t_bfloat16_t_gs_128_b_6_alN_true_batch_0"
    )
    # Below get_qmv_batch_limit (18, 12 and 10 rows transposed, 4 otherwise)
    # the products are matrix-vector ones.
    for m, n, k, transpose in (
        (17, 64, 256, True),
        (11, 4096, 256, True),
        (9, 8192, 256, True),
        (3, 256, 64, False),
    ):
        with pytest.raises(ValueError, match="qmv and qvm"):
            select("float32", 64, 4, m, n, k, 1, transpose)
        matmul_layout.select("float32", 64, 4, m, n, k, 1, transpose)
        select("float32", 64, 4, m + 1, n, k, 1, transpose)


def test_verification_cases_cover_every_route():
    entries = [record["entry"] for record in RECORDS]
    assert len(set(entries)) == len(entries)
    kernels = {layout.signature(entry)["kernel"] for entry in entries}
    assert kernels == {"qmm_t", "qmm_t_splitk", "qmm_n"}
    layouts = {(record["layout"], bool(record["wBatchShape"])) for record in RECORDS}
    assert layouts == {("dense", True), ("dense", False), ("strided", False)}
    splits = {
        (record["splitK"], record["dtype"], *record["dispatches"][1:])
        for record in RECORDS
        if record["splitK"] > 1
    }
    two_pass = "col_reduce_2pass_1_32_32_reduce_sum"
    looped = "col_reduce_looped_1_32_32_reduce_sum"
    assert splits == {
        (4, "float32", "col_reduce_small_1_reduce_sumfloat32"),
        (64, "float32", looped + "float32"),
        (512, "float32", two_pass + "float32", looped + "float32"),
        (4, "float16", "col_reduce_small_1_reduce_sumfloat16"),
        (8, "float16", "col_reduce_small_1_reduce_sumfloat16"),
        (16, "bfloat16", "col_reduce_small_1_reduce_sumbfloat16"),
        (64, "float16", looped + "float16"),
        (64, "bfloat16", looped + "bfloat16"),
        (512, "float16", two_pass + "float16", looped + "float16"),
        (384, "bfloat16", two_pass + "bfloat16", looped + "bfloat16"),
    }
    assert verify.small_column_variants() == {
        "col_reduce_small_1_reduce_sumfloat32@32x4",
        "col_reduce_small_1_reduce_sumfloat16@32x4",
        "col_reduce_small_1_reduce_sumfloat16@32x8",
        "col_reduce_small_1_reduce_sumbfloat16@32x8",
    }
    # Partial row tiles, partial column tiles of unaligned N and K tails.
    assert any(record["M"] % 32 for record in RECORDS)
    assert any(record["N"] % 32 for record in RECORDS if record["transpose"])
    assert any(record["K"] % 32 for record in RECORDS if not record["transpose"])
    # More than 65,535 partial products: 64 partitions of 32 by 32 float32
    # outputs, and test_qmm's 8 partitions of 64 by 256 float16 ones.
    partials = {
        record["name"]: record["splitK"] * record["M"] * record["N"]
        for record in RECORDS
    }
    assert partials["qmm-t-split-wide-sum"] == 65536
    assert partials["qmm-t-split-half-wide"] == 131072
    # Each partition is k_partition_size columns of K.
    for record in RECORDS:
        width = record["K"] // record["splitK"]
        assert record["partitions"] == [
            [start, start + width] for start in range(0, record["K"], width)
        ]
    # Every split-K plan runs: the two-pass sum above 256 partitions of fewer
    # than 32,768 outputs, the bfloat16 one over a folded 3-D x.
    plans = {
        layout.partition_sum(record["M"], record["N"], record["splitK"])
        for record in RECORDS
        if record["splitK"] > 1
    }
    assert plans == {"small", "looped", "two-pass"}
    (folded,) = (r for r in RECORDS if r["name"] == "qmm-t-split-bfloat-two-pass")
    assert folded["xShape"] == [2, 6, 24576] and folded["M"] == 12
    assert not hasattr(verify, "REFUSALS")


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_case_data_is_exact_in_every_element_type(record):
    import numpy as np

    data = case_data(record, 5)
    group, bits = record["groupSize"], record["bits"]
    scales = np.abs(data["scales"])
    assert set(np.unique(scales)) <= {1.0, 0.5, 0.25, 0.125}
    # Dequantized weights are (code - bias code) times a power of two, at most
    # 2^bits - 1 in magnitude: exact in bfloat16's eight significant bits.
    offsets = data["biases"] / data["scales"]
    assert np.array_equal(offsets, np.round(offsets))
    assert np.abs(offsets).max() <= (1 << bits) - 1
    assert data["x"].shape[-1] == record["K"]
    assert data["scales"].shape[-1] * group == (
        record["K"] if record["transpose"] else record["N"]
    )


def test_partition_sum_rounds_in_the_column_sum_order():
    import numpy as np

    from demos.integrations.mlx.portable_host.verify_quantized_matmul import (
        partition_sum,
    )

    # 16 partitions of one float16 output: threadgroup rows 0 to 7 add
    # partitions j and j + 8 from zero, then row 0 adds rows 1 to 7 in turn.
    # Every 2048 + 1 rounds back to 2048 (ties to even), so the exact 2056 is
    # never reached.
    record = {
        "dtype": "float16",
        "M": 1,
        "N": 1,
        "partitions": [[k, k + 1] for k in range(16)],
    }
    weights = np.array([2048.0] + [1.0] * 8 + [0.0] * 7).reshape(16, 1)
    assert partition_sum(np, record, np.ones((1, 16)), weights).tolist() == [[2048.0]]
    # Partition 8 is row 0's second addend: 1 + 2048 rounds there instead.
    weights = np.array([1.0] * 8 + [2048.0] + [0.0] * 7).reshape(16, 1)
    assert partition_sum(np, record, np.ones((1, 16)), weights).tolist() == [[2048.0]]
    weights = np.array([0.0] * 8 + [1.0] * 8).reshape(16, 1)
    assert partition_sum(np, record, np.ones((1, 16)), weights).tolist() == [[8.0]]
    with pytest.raises(ValueError, match="small column pass"):
        partition_sum(
            np,
            record | {"partitions": [[k, k + 1] for k in range(32)]},
            np.ones((1, 32)),
            np.ones((32, 1)),
        )


@pytest.mark.parametrize(
    "record",
    [
        record
        for record in RECORDS
        if record["dtype"] != "float32" and record["splitK"] > 1
    ],
    ids=lambda record: record["name"],
)
def test_half_split_references_round_every_partition(record):
    import numpy as np

    from demos.integrations.mlx.portable_host.verify_quantized_matmul import rounded

    seed = [case[0] for case in verify.CASES].index(record["name"])
    expected = case_data(record, seed)["expected"]
    assert np.array_equal(rounded(expected, record["dtype"]), expected)
    single = case_data(record | {"partitions": [[0, record["K"]]]}, seed)["expected"]
    # Rounding each partition and each addition changes these products.
    if record["name"] != "qmm-t-split-half":
        assert (expected != single).sum() > 100


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record):
    counts, parameters, execution = native_binding(record)
    entry = record["entry"]
    threads = counts["y"]
    metadata = layout.plan(entry, counts, parameters, threads, execution)
    assert (
        metadata["M"],
        metadata["N"],
        metadata["K"],
        metadata["batches"],
        metadata["splitK"],
    ) == (record["M"], record["N"], record["K"], record["batches"], record["splitK"])
    # Unbatched qmm kernels reflect the batch parameters they never read.
    if layout.signature(entry)["kernel"] != "qmm_t_splitk":
        assert set(metadata["parameters"]) == {
            *layout.SCALARS,
            *layout.BATCH_PARAMETERS,
        }


FAULTS = (
    "grid-n",
    "grid-m",
    "depth",
    "workgroup",
    "threads",
    "k",
    "group",
    "x-short",
    "w-short",
    "extra",
)


def mutate(fault, counts, parameters, execution):
    count = execution["workgroupCount"]
    if fault == "grid-n":
        count[0] += 1
    elif fault == "grid-m":
        count[1] += 1
    elif fault == "depth":
        count[2] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 2, 1]
    elif fault == "threads":
        counts["y"] -= 1
    elif fault == "k":
        parameters["K"] += 32
    elif fault == "group":
        parameters["K" if "k_partition_size" in parameters else "N"] += 1
    elif fault == "x-short":
        counts["x"] -= 1
    elif fault == "w-short":
        counts["w"] -= 1
    else:
        parameters["final_block_size"] = 1


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize(
    "name", ("qmm-t-stacked", "qmm-t-unsplit", "qmm-t-split-tiles", "qmm-n-k-tail")
)
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    record = next(item for item in RECORDS if item["name"] == name)
    counts, parameters, execution = native_binding(record)
    mutate(fault, counts, parameters, execution)
    threads = counts["y"] if fault != "threads" else counts["y"] + 1
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, threads, execution)


@pytest.mark.parametrize("fault", ("partition", "stride", "split", "missing"))
def test_split_plan_rejects_partitions_quantized_cpp_would_not_make(fault):
    record = next(item for item in RECORDS if item["name"] == "qmm-t-split-small")
    counts, parameters, execution = native_binding(record)
    if fault == "partition":
        parameters["k_partition_size"] //= 2
    elif fault == "stride":
        parameters["split_k_partition_stride"] += 1
    elif fault == "split":
        # Two partitions of 128 are valid MLX shapes, but not qmm_splitk's.
        execution["workgroupCount"][2] = 2
        parameters["k_partition_size"] = 128
        counts["y"] //= 2
    else:
        parameters.pop("split_k_partition_stride")
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)


def test_plan_rejects_batch_views_that_exceed_their_buffers():
    record = next(item for item in RECORDS if item["name"] == "qmm-n-stacked")
    counts, parameters, execution = native_binding(record)
    parameters["w_strides"][0] += 1
    with pytest.raises(ValueError, match="exceeds its buffer"):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)
    counts, parameters, execution = native_binding(record)
    parameters["x_shape"][-1] += 1
    with pytest.raises(ValueError, match="matrix shapes"):
        layout.plan(record["entry"], counts, parameters, counts["y"], execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "scalar", "names", "alias"))
def test_storage_rejects_native_identity_faults(fault):
    record = next(item for item in RECORDS if item["name"] == "qmm-t-split-small")
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    counts, parameters = layout.read_storage(record["entry"], dict(supplied))
    assert parameters["k_partition_size"] == 64
    assert counts["y"] == 4 * 30 * 8
    if fault == "dtype":
        supplied["x"].dtype = b"float16"
    elif fault == "direction":
        supplied["x"].output = 1
    elif fault == "scalar":
        supplied["k_partition_size"].count = 2
    elif fault == "names":
        supplied.pop("split_k_partition_stride")
    else:
        supplied["y"].data = supplied["x"].data
    with pytest.raises(ValueError, match="matrix-matrix"):
        layout.read_storage(record["entry"], supplied)
    assert memory


def test_dispatch_routes_qmm_entries_through_their_cache(monkeypatch):
    record = next(item for item in RECORDS if item["name"] == "qmm-t-unsplit")
    buffers, memory, execution = native_buffers(record)
    calls = []

    class Cache:
        def get(self, entry):
            calls.append(entry)
            raise RuntimeError("stop")

    host = object.__new__(runtime.HostRuntime)
    host.quantized_qmms, host.quantized_matmuls = Cache(), None
    launch = type("Launch", (), {"execution": lambda self: execution})()
    array = (runtime.Buffer * len(buffers))(*buffers)
    with pytest.raises(RuntimeError, match="stop"):
        host.dispatch(record["entry"], array, len(buffers), 40 * 96, launch=launch)
    assert calls == [record["entry"]]
    with pytest.raises(ValueError, match="matrix-matrix dispatch requires"):
        matmul_dispatch.dispatch_with(
            host, layout, None, record["entry"], array, len(buffers), 1, launch
        )
    assert memory


def test_runtime_reports_qmm_entries_available():
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization, host.quantized_matmuls = object(), object()
    host.quantized_qmms = None
    entry = b"affine_qmm_t_float_gs_64_b_4_alN_true_batch_0"
    assert host._entry_available(entry) == 0
    host.quantized_qmms = object()
    assert host._entry_available(entry) == 1
    assert host._entry_available(b"affine_qmm_t_float_gs_64_b_4_batch_0") == 0
    assert host._entry_available(b"affine_qmm_t_splitk_float_gs_64_b_4_alN_true") == 1


@pytest.mark.parametrize("kernel", ("qmm_t", "qmm_n", "qmm_t_splitk"))
def test_package_configs_match_the_qmm_gates(tmp_path, kernel):
    module = gate(kernel)
    work = tmp_path / "work"
    for entry in module.ENTRIES[:: len(module.ENTRIES) // 7]:
        for target in ("metal", "directx", "opengl"):
            config = qmm_packages.project_config(tmp_path, "work/out", target, entry)
            expected = module._config(work, target, entry)
            for field in (
                "root",
                "output_dir",
                "entry_points",
                "entry_workgroup_size_rules",
                "workgroup_access_assertions",
                "index_range_assertions",
                "metadata_preconditions",
                "source_options",
            ):
                assert getattr(config, field) == getattr(expected, field), field
            assert bool(config.index_range_assertions) == (target == "opengl")
            assert bool(config.workgroup_access_assertions) == (target != "metal")
    cache = qmm_packages.QuantizedQmmPackageCache(tmp_path, tmp_path / "c", "opengl")
    assert cache.SOURCE == layout.SOURCE == module.SOURCE
    with pytest.raises(ValueError, match="matrix-matrix target"):
        qmm_packages.QuantizedQmmPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import (
        column_small_packages,
        matmul_packages,
    )

    entries = qmm_packages.verification_entries()
    assert entries == sorted(record["entry"] for record in RECORDS)
    assert verify.small_column_variants() <= set(
        column_small_packages.verification_entries()
    )
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(qmm_packages, "_build_one", build)
    records = qmm_packages.build_entries(
        tmp_path, tmp_path / "packages", "directx", entries, 3
    )
    assert [record["entry"] for record in records] == entries
    assert {call[:2] for call in calls} == {
        ((tmp_path / "packages" / qmm_packages.CACHE_DIRECTORY).resolve(), "directx")
    }
    with pytest.raises(ValueError):
        qmm_packages.build_entries(
            tmp_path, tmp_path, "opengl", ["affine_qmm_t_float_gs_64_b_4"], 1
        )


def test_backend_routes_matrix_products_as_quantized_cpp():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert "does not route matrix-matrix products" not in backend
    for text in (
        "const int split_k = transpose_ && B == 1 ? qmm_split_k(M, N, K, group_size_) : 1;",
        "int split_k = int(std::max<int64_t>(1, 512 / tiles));",
        "const int k_align = group_size > 32 ? group_size : 32;",
        "while (split_k > 1 && (K % (split_k * k_align) != 0)) {",
        'std::string(mode) + "_qmm_t_splitk_" +',
        'std::string(mode) + (transpose ? "_qmm_t_" : "_qmm_n_")',
        'const char* mode = affine ? "affine" : fp_mode;',
        '(N % 32 == 0 ? "_alN_true" : "_alN_false")',
        "temp_shape.insert(temp_shape.begin(), split_k);",
        "{{tiles_n, tiles_m, uint32_t(split_k)}, {32, 2, 2}}",
        "{{tiles_n, tiles_m, uint32_t(B)}, {32, 2, 2}}",
        'dispatch_column_reduce(intermediate, out, plan, {0}, "sum", storage);',
    ):
        assert text in backend, text
    assert "partition_sum_type" not in backend


def test_trace_validation_requires_the_routed_entries():
    trace = [
        {
            "entry": entry,
            "target": "opengl",
            "matmulMetadata": {
                key: record[key] for key in ("M", "N", "K", "batches", "splitK")
            },
        }
        for record in RECORDS
        for entry in record["dispatches"]
    ]
    verify.validate_trace(trace, RECORDS, "opengl")
    split = next(
        index for index, event in enumerate(trace) if "splitk" in event["entry"]
    )
    with pytest.raises(RuntimeError):
        verify.validate_trace(
            trace[: split + 1] + trace[split + 2 :], RECORDS, "opengl"
        )
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, RECORDS, "metal")
    trace[split]["matmulMetadata"]["splitK"] = 2
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, RECORDS, "opengl")
    trace[split]["matmulMetadata"]["splitK"] = RECORDS[5]["splitK"]
    with pytest.raises(RuntimeError, match="cover"):
        verify.validate_trace(
            [event for event in trace if "qmm_n" not in event["entry"]],
            [record for record in RECORDS if "qmm_n" not in record["entry"]],
            "opengl",
        )


def test_ci_translates_and_executes_the_matrix_matrix_products():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["quantized-host"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    translate = steps["Translate quantized matrix-matrix packages"]
    name = "Execute affine quantized matrix-matrix products through MLX"
    step = steps[name]
    for item in (translate, step):
        assert "if" not in item and "continue-on-error" not in item
        assert "set -euo pipefail" in item["run"]
        assert "--packages .mlx-portable-quantized/base/packages" in item["run"]
    assert "portable_host.qmm_packages" in translate["run"]
    assert "--verification" in translate["run"] and "--jobs 3" in translate["run"]
    assert "portable_host.verify_quantized_qmm" in step["run"]
    # The looped partition sum comes from the split-K column packages.
    assert "--reductions .mlx-portable-quantized/column-reductions" in step["run"]
    assert "tee .mlx-portable-quantized/quantized-qmm.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert (
        "test_portable_quantized_qmm.py" in steps["Validate quantized host contracts"][
            "run"
        ]
    )
    # Small partition sums are translated with the split-K sums, which list
    # these cases' variants among their verification entries.
    assert (
        names.index("Build adapted upstream MLX")
        < names.index("Translate quantized matrix-matrix packages")
        < names.index("Translate split-K partition sums")
        < names.index(name)
        < names.index("Retain quantized execution evidence")
    )
    from demos.integrations.mlx.portable_host import column_small_packages

    assert verify.small_column_variants() <= set(
        column_small_packages.verification_entries()
    )
