"""Affine GatherQMM host routing, launch contracts and package configs."""

import ctypes
import importlib
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import gather_qmm_layout as layout
from demos.integrations.mlx.portable_host import (
    gather_qmm_packages,
    lazy_package_evidence,
    matmul_dispatch,
    matmul_layout,
    runtime,
)
from demos.integrations.mlx.portable_host import verify_gather_qmm as verify
from demos.integrations.mlx.tests.host.test_portable_quantized_matmul import (
    CTYPES,
    contiguous,
)

HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
GATES = {
    "qmv_fast": "test_current_quantized_qmv",
    "qmv": "test_current_quantized_qmv_unaligned",
    "qvm": "test_current_quantized_qvm",
    "qmm_t": "test_current_quantized_gather_qmm",
    "qmm_n": "test_current_quantized_gather_qmm",
}


def gate(kernel):
    return importlib.import_module(
        f"demos.integrations.mlx.tests.kernels.{GATES[kernel]}"
    )


def gate_entries(kernel):
    """The gathered entries of ``kernel`` its current-tree gate translates."""
    module = gate(kernel)
    prefix = f"affine_gather_{kernel}_"
    return [
        entry
        for entry, info in module.INFO.items()
        if entry.startswith(prefix) and info.get("variant") == "gather"
    ]


def collapse(shape, strides):
    """mlx/backend/common/utils.cpp collapse_contiguous_dims."""
    marks = []
    if shape:
        if shape[0] != 1:
            marks.append(0)
        size = shape[0]
        for axis in range(1, len(shape)):
            size *= shape[axis]
            if any(
                values[axis] * shape[axis] != values[axis - 1] or size > 2**31 - 1
                for values in strides
            ):
                marks.append(-1)
                size = shape[axis]
            if shape[axis] != 1:
                marks.append(axis)
        marks.append(-1)
    collapsed, views, index = [], [[] for _ in strides], 0
    while True:
        while index < len(marks) and marks[index] == -1:
            index += 1
        if index == len(marks):
            break
        extent, end = shape[marks[index]], index + 1
        while marks[end] != -1:
            extent *= shape[marks[end]]
            end += 1
        collapsed.append(extent)
        for view, values in zip(views, strides):
            view.append(values[marks[end - 1]])
        index = end + 1
    if shape and not collapsed:
        collapsed, views = [1], [[0] for _ in strides]
    return collapsed, views


def flat(value):
    if isinstance(value, (tuple, list)):
        return [item for part in value for item in flat(part)]
    return [value]


def broadcast_strides(shape, target):
    """The strides of a contiguous ``shape`` broadcast to ``target``."""
    dense = contiguous(list(shape))
    padded = [0] * (len(target) - len(shape)) + [
        0 if extent == 1 and want > 1 else stride
        for extent, want, stride in zip(
            shape, target[len(target) - len(shape) :], dense
        )
    ]
    return padded


def native_binding(record, **faults):
    """What gather_qmm.inc binds for one verification case."""
    k, n, m = record["K"], record["N"], record["M"]
    rows, columns = (n, k) if record["transpose"] else (k, n)
    words, groups = columns * record["bits"] // 32, columns // record["groupSize"]
    x_shape = [*record["xBatchShape"], m, k]
    w_batch = record["wBatchShape"]
    w_shape = [*w_batch, rows, words]
    index_shape = record["indexShape"]
    lhs, rhs = record["lhsIndices"], record["rhsIndices"]
    shape, (lhs_strides, rhs_strides) = collapse(
        index_shape,
        [
            broadcast_strides(verify.shape_of(lhs), index_shape),
            broadcast_strides(verify.shape_of(rhs), index_shape),
        ],
    )
    values = {
        "lhs_indices": faults.get("lhs", flat(lhs)),
        "rhs_indices": faults.get("rhs", flat(rhs)),
    }
    counts = {
        "x": math.prod(x_shape),
        "w": math.prod(w_shape),
        "scales": math.prod(w_batch) * rows * groups,
        "biases": math.prod(w_batch) * rows * groups,
        "lhs_indices": len(values["lhs_indices"]),
        "rhs_indices": len(values["rhs_indices"]),
        "y": record["batches"] * m * n,
    }
    if layout.is_matrix(record["entry"]):
        parameters = {"K": k, "N": n, "M": m}
        grid = [-(-n // 32), -(-m // 32), record["batches"]]
    else:
        parameters = {"in_vec_size": k, "out_vec_size": n}
        grid = [m, -(-n // layout.tile_columns(record["entry"])), record["batches"]]
    parameters.update(
        x_batch_ndims=len(x_shape) - 2,
        x_shape=x_shape,
        x_strides=contiguous(x_shape),
        w_batch_ndims=len(w_shape) - 2,
        w_shape=w_shape,
        w_strides=contiguous(w_shape),
        s_strides=contiguous([*w_batch, rows, groups]),
        b_strides=contiguous([*w_batch, rows, groups]),
        batch_ndims=len(shape),
        batch_shape=shape,
        lhs_strides=lhs_strides,
        rhs_strides=rhs_strides,
    )
    parameters.update(faults.get("parameters", {}))
    execution = {
        "workgroupCount": faults.get("grid", grid),
        "workgroupSize": list(layout.workgroup_size(record["entry"])),
    }
    return counts, values, parameters, execution


def native_buffers(record, **faults):
    """ctypes buffers in gather_qmm.inc's binding order, with backing storage."""
    counts, values, parameters, execution = native_binding(record, **faults)
    entry, memory, buffers = record["entry"], {}, []
    for name in layout.parameter_names(entry):
        dtype = layout.parameter_type(entry, name)
        if name in values:
            data = values[name]
        elif name in counts:
            data = [0] * counts[name]
        else:
            value = parameters[name]
            data = [value] if isinstance(value, int) else value
        memory[name] = (CTYPES[dtype] * len(data))(*data)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(data),
                int(name == "y"),
            )
        )
    return buffers, memory, execution


def planned(record, **faults):
    buffers, memory, execution = native_buffers(record, **faults)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    counts, parameters = layout.read_storage(record["entry"], supplied)
    threads = record["batches"] * record["M"] * record["N"]
    return layout.plan(record["entry"], counts, parameters, threads, execution), memory


def gated(kernel):
    if kernel in ("qmm_t", "qmm_n"):
        return [
            entry for entry in gate(kernel).ENTRIES if f"_gather_{kernel}_" in entry
        ]
    return gate_entries(kernel)


# 3 types, 3 group sizes and 6 widths; qmm_t twice, over N's alignment.
@pytest.mark.parametrize(
    ("kernel", "count"),
    (("qmv_fast", 54), ("qmv", 54), ("qvm", 54), ("qmm_t", 108), ("qmm_n", 54)),
)
def test_signature_accepts_every_gate_entry(kernel, count):
    entries = gated(kernel)
    assert len(entries) == count
    for entry in entries:
        info = layout.signature(entry)
        assert info["kernel"] == kernel
        assert entry.startswith(layout.ENTRY_PREFIXES)
        expected = (32, 2, 2) if kernel in ("qmm_t", "qmm_n") else (32, 2, 1)
        assert layout.workgroup_size(entry) == expected
        assert tuple(gate(kernel).WORKGROUP) == expected


@pytest.mark.parametrize(
    "entry",
    (
        "affine_gather_qmm_t_float_gs_64_b_4",
        "affine_gather_qmm_n_float_gs_64_b_4_alN_true",
        "affine_gather_qmv_float_gs_64_b_4_alN_false",
        "affine_gather_qmv_fast_float_gs_64_b_4_batch_1",
        "affine_gather_qmm_rhs_nt_float_gs_64_b_4",
        "affine_gather_qvm_float_gs_16_b_4",
        "affine_qmv_float_gs_64_b_4_batch_1",
        "mxfp4_gather_qmv_float_gs_32_b_4",
        None,
    ),
)
def test_signature_rejects_other_entries(entry):
    with pytest.raises(ValueError):
        layout.signature(entry)


def test_selection_follows_gather_qmm_for_upstream_shapes():
    # test_gather_qmm's affine shapes at its default group size and width.
    expected = {
        (1, 32, 128, True): "affine_gather_qmv_float_gs_64_b_4",
        (32, 32, 256, True): "affine_gather_qmm_t_float_gs_64_b_4_alN_true",
        (1, 32, 256, True): "affine_gather_qmv_float_gs_64_b_4",
        (32, 256, 32, False): "affine_gather_qmm_n_float_gs_64_b_4",
        (1, 256, 32, False): "affine_gather_qvm_float_gs_64_b_4",
        (32, 32, 512, True): "affine_gather_qmm_t_float_gs_64_b_4_alN_true",
        (1, 32, 512, True): "affine_gather_qmv_fast_float_gs_64_b_4",
        (32, 512, 32, False): "affine_gather_qmm_n_float_gs_64_b_4",
        (1, 512, 32, False): "affine_gather_qvm_float_gs_64_b_4",
    }
    for (m, n, k, transpose), entry in expected.items():
        assert layout.select("float32", 64, 4, m, n, k, transpose) == entry
    # The vector limit: 18 rows below 2,048 by 2,048, then 12 and 10; 4
    # untransposed.
    assert layout.select("float16", 32, 4, 17, 64, 2048, True).startswith(
        "affine_gather_qmv_fast_"
    )
    assert "qmm_t" in layout.select("float16", 32, 4, 18, 64, 2048, True)
    assert "qmv_" in layout.select("float16", 32, 4, 11, 64, 4096, True)
    assert "qmm_t" in layout.select("float16", 32, 4, 12, 64, 4096, True)
    assert "qmm_t" in layout.select("float16", 32, 4, 10, 64, 8192, True)
    assert "qvm" in layout.select("float16", 32, 4, 3, 64, 64, False)
    assert "qmm_n" in layout.select("float16", 32, 4, 4, 64, 64, False)
    assert layout.select("bfloat16", 32, 4, 33, 40, 64).endswith("_alN_false")


def test_verification_cases_cover_every_route():
    entries = [record["entry"] for record in RECORDS]
    assert len(set(entries)) == len(entries)
    assert {layout.signature(entry)["kernel"] for entry in entries} == verify.KERNELS
    for record in RECORDS:
        assert record["dispatches"] == [record["entry"]]
        assert (
            layout.select(
                record["dtype"],
                record["groupSize"],
                record["bits"],
                record["M"],
                record["N"],
                record["K"],
                record["transpose"],
            )
            == record["entry"]
        )
    assert gather_qmm_packages.verification_entries() == sorted(entries)


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_case_data_is_exact_in_every_element_type(record):
    import numpy as np

    data = verify.case_data(record, 3)
    assert data["expected"].shape == (*record["indexShape"], record["M"], record["N"])
    assert data["pairs"] == verify.matrix_pairs(np, record)
    # Every dequantized weight and input holds in the element type.
    for name in ("scales", "biases", "x"):
        values = np.asarray(data[name], dtype=np.float64)
        assert np.array_equal(verify.rounded(values, record["dtype"]), values), name


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_replays_the_gathered_offsets(record):
    import numpy as np

    metadata, memory = planned(record)
    assert memory
    assert (
        metadata["M"],
        metadata["N"],
        metadata["K"],
        metadata["batches"],
        metadata["kernel"],
    ) == (
        record["M"],
        record["N"],
        record["K"],
        record["batches"],
        layout.signature(record["entry"])["kernel"],
    )
    assert metadata["matrixIndices"] == verify.matrix_pairs(np, record)
    assert set(metadata["parameters"]) == set(
        layout.parameter_names(record["entry"])
    ) - set(layout.MATRIX_PARAMETERS)


def test_cases_take_every_offset_path():
    parameters = [planned(record)[0]["parameters"] for record in RECORDS]
    assert {value["batch_ndims"] for value in parameters} == {1, 2}
    assert {value["x_batch_ndims"] for value in parameters} == {1, 2}
    assert {value["w_batch_ndims"] for value in parameters} == {0, 1, 2}
    # A scalar index broadcasts with a zero stride.
    assert any(0 in value["rhs_strides"] for value in parameters)
    assert any(0 in value["lhs_strides"] for value in parameters)


def _record(name):
    return next(record for record in RECORDS if record["name"] == name)


@pytest.mark.parametrize(
    ("name", "faults", "message"),
    (
        ("qmv-fast-experts", {"rhs": [2, 0, 3, 1]}, "index names no matrix"),
        ("qmv-fast-experts", {"lhs": [1, 0, 2, 1]}, "index names no matrix"),
        (
            "qmv-fast-experts",
            {"parameters": {"lhs_strides": [2]}},
            "index view exceeds its buffer",
        ),
        (
            "qmv-fast-experts",
            {"parameters": {"batch_shape": [2]}},
            "index batch does not match",
        ),
        (
            "qmm-t-experts",
            {"parameters": {"rhs_strides": [-1]}},
            "index batch does not match",
        ),
        ("qmm-t-experts", {"grid": [2, 1, 3]}, "launch does not match"),
        ("qvm-experts", {"grid": [3, 3, 3]}, "launch does not match"),
        ("qmv-broadcast", {"parameters": {"batch_ndims": 1}}, "index batch"),
        (
            "qmm-n-experts",
            {"parameters": {"w_strides": [520, 13, 1]}},
            "row contiguous",
        ),
        (
            "qmm-n-experts",
            {"parameters": {"x_shape": [2, 33, 41]}},
            "matrix shapes do not match",
        ),
        (
            "qmv-shared-weights",
            {"parameters": {"x_strides": [2000, 512, 256, 1]}},
            "x matrix exceeds its buffer",
        ),
        ("qmm-t-limit", {"grid": [1, 2, 2]}, "launch does not match"),
        (
            "qmm-t-limit",
            {"parameters": {"M": 17}, "grid": [1, 1, 2]},
            "differs from MLX's routing",
        ),
    ),
)
def test_plan_rejects_launches_that_differ_from_mlx(name, faults, message):
    with pytest.raises(ValueError, match=message):
        planned(_record(name), **faults)


def test_storage_rejects_native_identity_faults():
    record = _record("qmm-n-experts")
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    layout.read_storage(record["entry"], supplied)
    for name, field, value in (
        ("lhs_indices", "dtype", b"int32"),
        ("rhs_indices", "output", 1),
        ("batch_shape", "count", layout.MAX_BATCH_DIMENSIONS + 1),
    ):
        faulty = dict(supplied)
        buffer = supplied[name]
        values = {
            key: getattr(buffer, key)
            for key in ("name", "dtype", "data", "count", "output")
        }
        values[field] = value
        if field == "count":
            storage = (ctypes.c_int32 * value)(*([1] * value))
            memory[f"{name}-fault"] = storage
            values["data"] = ctypes.addressof(storage)
        faulty[name] = runtime.Buffer(*values.values())
        with pytest.raises(ValueError):
            layout.read_storage(record["entry"], faulty)
    with pytest.raises(ValueError):
        layout.read_storage(
            record["entry"],
            {key: item for key, item in supplied.items() if key != "rhs_indices"},
        )


def test_dispatch_routes_gathered_entries_through_their_cache():
    record = _record("qmm-t-experts")
    buffers, memory, execution = native_buffers(record)
    calls = []

    class Cache:
        def get(self, entry):
            calls.append(entry)
            raise RuntimeError("stop")

    host = object.__new__(runtime.HostRuntime)
    host.gather_qmms = Cache()
    launch = type("Launch", (), {"execution": lambda self: execution})()
    array = (runtime.Buffer * len(buffers))(*buffers)
    threads = record["batches"] * record["M"] * record["N"]
    with pytest.raises(RuntimeError, match="stop"):
        host.dispatch(record["entry"], array, len(buffers), threads, launch=launch)
    assert calls == [record["entry"]]
    with pytest.raises(ValueError, match="dispatch requires"):
        matmul_dispatch.dispatch_with(
            host, layout, None, record["entry"], array, len(buffers), threads, launch
        )
    assert memory


def test_runtime_reports_gathered_entries_available():
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.quantized_qmms = None
    host.fp_quantized_matmuls = host.fp_quantized_qmms = None
    host.gather_qmms = None
    entry = b"affine_gather_qmm_t_float_gs_64_b_4_alN_true"
    assert host._entry_available(entry) == 0
    host.gather_qmms = object()
    assert host._entry_available(entry) == 1
    assert host._entry_available(b"affine_gather_qvm_bfloat16_t_gs_32_b_2") == 1
    assert host._entry_available(b"affine_gather_qmm_t_float_gs_64_b_4") == 0
    assert host._entry_available(b"affine_gather_qmm_rhs_nt_float_gs_64_b_4") == 0


def test_lazy_evidence_audits_gathered_dispatches(tmp_path):
    assert (
        lazy_package_evidence.family("affine_gather_qmv_fast_float_gs_64_b_4")
        == "gather-quantized"
    )
    assert lazy_package_evidence.family("affine_qmv_float_gs_64_b_4_batch_0") == (
        "quantized-mv"
    )
    lazy = lazy_package_evidence.LazyPackages(tmp_path, tmp_path / "p", "opengl")
    cache = lazy.caches["gather-quantized"]
    assert isinstance(cache, gather_qmm_packages.GatherQmmPackageCache)
    assert cache.directory == (tmp_path / "p" / gather_qmm_packages.CACHE_DIRECTORY)


@pytest.mark.parametrize("kernel", ("qmm_t", "qmm_n"))
def test_matrix_package_configs_match_the_gather_qmm_gate(tmp_path, kernel):
    module = gate(kernel)
    entries = gated(kernel)
    work = tmp_path / "work"
    for entry in entries[:: max(1, len(entries) // 7)]:
        for target in ("metal", "directx", "opengl"):
            config = gather_qmm_packages.project_config(
                tmp_path, "work/out", target, entry
            )
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


@pytest.mark.parametrize("kernel", ("qmv_fast", "qmv", "qvm"))
def test_vector_package_configs_match_the_gather_gates(tmp_path, kernel):
    module = gate(kernel)
    work = tmp_path / "work"
    entries = gated(kernel)
    for entry in entries[:: max(1, len(entries) // 5)]:
        for target in ("metal", "directx", "opengl"):
            config = gather_qmm_packages.project_config(
                tmp_path, "work/out", target, entry
            )
            expected = module._config(work, target, entry)
            for field in (
                "root",
                "output_dir",
                "entry_points",
                "entry_workgroup_size_rules",
                "workgroup_access_assertions",
                "metadata_preconditions",
                "source_options",
            ):
                assert getattr(config, field) == getattr(expected, field), field
            # The gates bound the offsets by their cases; the host by the
            # largest buffer it binds, which plan proves.
            assert [
                (item.source, item.expression, item.value_range.minimum)
                for item in config.index_range_assertions
            ] == [
                (item.source, item.expression, item.value_range.minimum)
                for item in expected.index_range_assertions
            ]
            assert {
                item.value_range.maximum for item in config.index_range_assertions
            } <= {layout.MAX_ELEMENTS}
            assert bool(config.index_range_assertions) == (target == "opengl")
    cache = gather_qmm_packages.GatherQmmPackageCache(tmp_path, tmp_path / "c", "metal")
    assert cache.SOURCE == layout.SOURCE == module.SOURCE
    with pytest.raises(ValueError, match="product target"):
        gather_qmm_packages.GatherQmmPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import matmul_packages

    entries = gather_qmm_packages.verification_entries()
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(gather_qmm_packages, "_build_one", build)
    records = gather_qmm_packages.build_entries(
        tmp_path, tmp_path / "packages", "directx", entries, 3
    )
    assert [record["entry"] for record in records] == entries
    assert {call[:2] for call in calls} == {
        (
            (tmp_path / "packages" / gather_qmm_packages.CACHE_DIRECTORY).resolve(),
            "directx",
        )
    }
    with pytest.raises(ValueError):
        gather_qmm_packages.build_entries(
            tmp_path, tmp_path, "opengl", ["affine_gather_qmm_t_float_gs_64_b_4"], 1
        )


def test_backend_routes_gathered_products_as_quantized_cpp():
    source = (HOST / "gather_qmm.inc").read_text(encoding="utf-8")
    for text in (
        "void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {",
        "if (M == 1 && B >= 16 && right_sorted_ && B / E >= 4) {",
        "const bool matrix = M >= vector_limit;",
        'const std::string mode = affine ? "affine" : fp_mode;',
        'mode + (transpose_ ? "_gather_qmm_t_" : "_gather_qmm_n_") + stem +',
        '(N % 32 == 0 ? "_alN_true" : "_alN_false")',
        '"_gather_qmv_fast_"',
        "const int rows = transpose_ ? 8 : 2 * std::min(group_size_, 32);",
        "lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});",
        "CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 2}}",
        "CrosstlMlxLaunch{{columns, tiles, uint32_t(B)}, {32, 2, 1}}",
        "biases = matmul_operand(inputs[3]);",
    ):
        assert text in source, text
    # The buffers in quantized.cpp's binding order.
    order = [
        source.index(f'{{"{name}", ')
        for name in ("w", "scales", "biases", "x", "lhs_indices", "rhs_indices", "y")
    ]
    assert order == sorted(order)
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.index('#include "mlx/backend/no_gpu/crosstl_gather_mm.inc"') < (
        backend.index('#include "mlx/backend/no_gpu/crosstl_gather_qmm.inc"')
    )
    text = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '"GatherQMM",' in text
    assert '"crosstl_gather_qmm.inc": (HERE / "gather_qmm.inc").read_bytes(),' in text


def test_trace_validation_requires_the_routed_entries():
    import numpy as np

    records = [
        {**record, "pairs": verify.matrix_pairs(np, record)} for record in RECORDS
    ]
    refusals = [
        {"name": "sorted-rhs", "dispatches": [verify.DEFAULT_INDEX_ENTRY]},
        {"name": "rhs-out-of-range", "dispatches": []},
    ]
    trace = [
        {
            "entry": record["entry"],
            "target": "opengl",
            "matmulMetadata": {
                "M": record["M"],
                "N": record["N"],
                "K": record["K"],
                "batches": record["batches"],
                "matrixIndices": record["pairs"],
            },
            "parameters": planned(record)[0]["parameters"],
        }
        for record in records
    ]
    trace.append({"entry": verify.DEFAULT_INDEX_ENTRY, "target": "opengl"})
    verify.validate_trace(trace, records, refusals, "opengl")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, records, refusals, "metal")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace[:-1], records, refusals, "opengl")
    trace[0]["matmulMetadata"]["matrixIndices"] = [[0, 0]] * records[0]["batches"]
    with pytest.raises(RuntimeError, match="launch differs"):
        verify.validate_trace(trace, records, refusals, "opengl")
    trace[0]["matmulMetadata"]["matrixIndices"] = records[0]["pairs"]
    kept = [item for item in records if "qvm" not in item["entry"]]
    with pytest.raises(RuntimeError, match="cover"):
        verify.validate_trace(
            [event for event in trace if "qvm" not in event["entry"]],
            kept,
            refusals,
            "opengl",
        )


def test_ci_translates_and_executes_the_gathered_products():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["quantized-host"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    translate = steps["Translate gathered quantized packages"]
    name = "Execute affine gathered quantized products through MLX"
    step = steps[name]
    for item in (translate, step):
        assert "if" not in item and "continue-on-error" not in item
        assert "set -euo pipefail" in item["run"]
        assert "--packages .mlx-portable-quantized/base/packages" in item["run"]
    assert "portable_host.gather_qmm_packages" in translate["run"]
    assert "--verification" in translate["run"] and "--jobs 3" in translate["run"]
    assert "portable_host.verify_gather_qmm" in step["run"]
    assert "tee .mlx-portable-quantized/gather-qmm.log" in step["run"]
    assert step["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert (
        "test_portable_gather_qmm.py" in steps["Validate quantized host contracts"][
            "run"
        ]
    )
    assert (
        names.index("Build adapted upstream MLX")
        < names.index("Translate gathered quantized packages")
        < names.index(name)
        < names.index("Retain quantized execution evidence")
    )
    assert matmul_layout.MAX_ELEMENTS == layout.MAX_ELEMENTS
