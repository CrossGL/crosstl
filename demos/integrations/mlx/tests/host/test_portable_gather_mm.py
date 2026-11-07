"""GatherMM host routing, gathered launch contracts and their verification cases."""

import copy
import ctypes
import math
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    gather_mm_dispatch,
    gather_mm_layout,
    gather_mm_packages,
    gather_mv_layout,
    prepare,
    runtime,
)
from demos.integrations.mlx.portable_host import verify_gather_mm as verify
from demos.integrations.mlx.portable_host import verify_matmul

HOST = Path(gather_mm_layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
BY_NAME = {record["name"]: record for record in RECORDS}
CTYPES = {
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint32": ctypes.c_uint32,
}


def kind(name):
    return [record for record in RECORDS if record["kind"] == name]


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def row_strides(shape):
    return verify_matmul._row_strides(list(shape))


def bound(record, name):
    """The (shape, strides) gather_mm.inc binds: copies are row-major."""
    shape, strides = record[name]["shape"], record[name]["strides"]
    if record["route"]["copies"].get(name):
        strides = row_strides(shape)
    return list(shape), list(strides)


def index_view(record, name):
    """An index array's stored values and its strides broadcast to the batch."""
    spec = record[name] or {
        "shape": record["a"]["shape"][:-2],
        "values": list(range(math.prod(record["a"]["shape"][:-2]))),
    }
    shape = record["indexShape"]
    source = [1] * (len(shape) - len(spec["shape"])) + list(spec["shape"])
    strides = [
        0 if extent == 1 and target != 1 else stride
        for extent, target, stride in zip(source, shape, row_strides(source))
    ]
    return list(spec["values"]), strides


def mv_binding(record):
    route, (entry,) = record["route"], record["entries"]
    is_mv = record["a"]["shape"][-2] != 1
    mat, vec = ("a", "b") if is_mv else ("b", "a")
    mat_shape, mat_strides = bound(record, mat)
    vec_shape, vec_strides = bound(record, vec)
    lhs, lhs_strides = index_view(record, "lhs")
    rhs, rhs_strides = index_view(record, "rhs")
    mat_values, mat_index_strides = (lhs, lhs_strides) if is_mv else (rhs, rhs_strides)
    vec_values, vec_index_strides = (rhs, rhs_strides) if is_mv else (lhs, lhs_strides)
    ndim = len(record["indexShape"])
    out_shape = [
        *record["indexShape"],
        record["a"]["shape"][-2],
        record["b"]["shape"][-1],
    ]
    batches = math.prod(record["indexShape"])
    counts = {
        "mat": span(mat_shape, mat_strides),
        "in_vec": span(vec_shape, vec_strides),
        "out_vec": batches * route["outVecSize"],
        "vec_indices": len(vec_values),
        "mat_indices": len(mat_values),
    }
    parameters = {
        "in_vec_size": route["K"],
        "out_vec_size": route["outVecSize"],
        "marix_ld": route["matrixLd"],
        "batch_ndim": ndim,
        "batch_shape": out_shape,
        "index_batch_strides": (vec_index_strides + mat_index_strides) or [0, 0],
        "vector_batch_ndim": len(vec_shape) - 2,
        "vector_batch_shape": vec_shape,
        "vector_batch_stride": vec_strides,
        "matrix_batch_ndim": len(mat_shape) - 2,
        "matrix_batch_shape": mat_shape,
        "matrix_batch_stride": mat_strides,
    }
    views = {
        "vec_indices": (ctypes.c_uint32 * len(vec_values))(*vec_values),
        "mat_indices": (ctypes.c_uint32 * len(mat_values))(*mat_values),
    }
    execution = {
        "workgroupCount": [
            -(-route["outVecSize"] // gather_mv_layout.outputs_per_group(entry)),
            1,
            batches,
        ],
        "workgroupSize": list(gather_mv_layout.workgroup_size(entry)),
    }
    return entry, counts, parameters, views, execution


def mm_binding(record):
    route, (entry,) = record["route"], record["entries"]
    info = gather_mm_layout.signature(entry)
    a_shape, a_strides = bound(record, "a")
    b_shape, b_strides = bound(record, "b")
    lhs, lhs_strides = index_view(record, "lhs")
    rhs, rhs_strides = index_view(record, "rhs")
    ndim = len(record["indexShape"])
    m, n, k = route["M"], route["N"], route["K"]
    batches = math.prod(record["indexShape"])
    params = {
        "M": m,
        "N": n,
        "K": k,
        "lda": route["lda"],
        "ldb": route["ldb"],
        "ldd": n,
        "tiles_n": -(-n // info["bn"]),
        "tiles_m": -(-m // info["bm"]),
        "batch_stride_a": lhs_strides[0] if ndim else 0,
        "batch_stride_b": rhs_strides[0] if ndim else 0,
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // info["bk"],
        "batch_ndim": ndim,
    }
    counts = {
        "A": span(a_shape, a_strides),
        "B": span(b_shape, b_strides),
        "C": batches * m * n,
        "lhs_indices": len(lhs),
        "rhs_indices": len(rhs),
    }
    parameters = {
        "params": params,
        "indices_shape": list(record["indexShape"]) or [1],
        "lhs_strides": lhs_strides or [0],
        "rhs_strides": rhs_strides or [0],
        "batch_ndim_a": len(a_shape) - 2,
        "batch_shape_a": a_shape,
        "batch_strides_a": a_strides,
        "batch_ndim_b": len(b_shape) - 2,
        "batch_shape_b": b_shape,
        "batch_strides_b": b_strides,
    }
    views = {
        "lhs_indices": (ctypes.c_uint32 * len(lhs))(*lhs),
        "rhs_indices": (ctypes.c_uint32 * len(rhs))(*rhs),
    }
    execution = {
        "workgroupCount": [params["tiles_n"], params["tiles_m"], batches],
        "workgroupSize": list(gather_mm_layout.workgroup_size(entry)),
    }
    return entry, counts, parameters, views, execution


def rhs_binding(record):
    route = record["route"]
    entry = record["entries"][1]
    info = gather_mm_layout.signature(entry)
    b_shape, b_strides = bound(record, "b")
    m, n, k = route["M"], route["N"], route["K"]
    groups = math.prod(b_shape[:-2])
    rows, _ = index_view(record, "rhs")
    offsets = gather_mm_layout.offsets_reference(rows, groups)
    params = {
        "M": m,
        "N": n,
        "K": k,
        "lda": k,
        "ldb": route["ldb"],
        "ldd": n,
        "tiles_n": -(-n // info["bn"]),
        "tiles_m": gather_mm_layout.rhs_tiles(m, groups),
        "batch_stride_a": 0,
        "batch_stride_b": b_strides[-3],
        "batch_stride_d": 0,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // info["bk"],
        "batch_ndim": 0,
    }
    counts = {"A": m * k, "B": span(b_shape, b_strides), "C": m * n, "offsets": groups}
    parameters = {"params": params, "num_groups": groups}
    views = {"offsets": (ctypes.c_int32 * groups)(*offsets)}
    execution = {
        "workgroupCount": [params["tiles_n"], params["tiles_m"], 1],
        "workgroupSize": list(gather_mm_layout.workgroup_size(entry)),
    }
    return entry, counts, parameters, views, execution


def test_cases_cover_every_gather_route():
    for target in ("metal", "directx", "opengl"):
        assert verify._covered(
            [{**record, "outputHash": "x"} for record in RECORDS], target
        )
    assert {record["kind"] for record in RECORDS} == set(verify.KERNELS)
    assert verify.RHS_UNSUPPORTED == {
        "directx": "CrossGL/crosstl#2546",
        "opengl": "CrossGL/crosstl#2546",
    }
    assert not verify._covered(kind("mm") + kind("rhs"), "metal")


def test_routes_mirror_gather_mm_eval_gpu():
    expected = {
        "matrix-vector-batch": [
            "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"
            "_has_batch_n_align_M_n_align_N_n_align_K_n"
        ],
        "matrix-grid-aligned": [
            "steel_gather_mm_nt_float16_float16_bm64_bn64_bk16_wm2_wn2"
            "_has_batch_t_align_M_t_align_N_t_align_K_t"
        ],
        "matrix-scalar-transposed": [
            "steel_gather_mm_tn_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2"
            "_has_batch_n_align_M_n_align_N_n_align_K_n"
        ],
        "row-transposed-gemv": ["gemv_t_gather_float32_bm1_bn2_sm8_sn4_tm4_tn4"],
        "row-transposed-weights": ["gemv_gather_float16_bm4_bn1_sm1_sn32_tm4_tn4"],
        "column-grid": ["gemv_gather_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4"],
        "sorted-rows": [
            "gather_mm_offsets",
            "steel_gather_mm_rhs_nn_float32_float32_bm16_bn64_bk16_wm1_wn2"
            "_align_N_t_align_K_t",
        ],
        "sorted-broadcast-transposed": [
            "gather_mm_offsets",
            "steel_gather_mm_rhs_nt_float16_float16_bm16_bn64_bk16_wm1_wn2"
            "_align_N_n_align_K_t",
        ],
        "empty-k": [],
    }
    assert {name: BY_NAME[name]["entries"] for name in expected} == expected
    assert BY_NAME["matrix-copied-default"]["copies"] == 1
    assert BY_NAME["matrix-copied-default"]["defaults"] == 1
    assert BY_NAME["sorted-broadcast-transposed"]["copies"] == 1
    assert BY_NAME["sorted-rows"]["copies"] == 0
    assert BY_NAME["sorted-rows"]["route"]["M"] == 8
    # Unsorted single rows and right-sorted products with left indices take
    # gather_mv; MLX sorts only when the left indices are implied.
    plain = gather_mm_layout.route(
        "float32", [4, 1, 8], [8, 8, 1], [2, 8, 3], [24, 3, 1], [4], False
    )
    assert plain["kind"] == "mv"
    for entry in (e for record in RECORDS for e in record["entries"]):
        assert gather_mm_dispatch.is_entry(entry)


def test_signatures_accept_only_the_host_tiles():
    for record in RECORDS:
        for entry in record["entries"]:
            gather_mm_dispatch.family(entry)
    for entry in (
        "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm1_wn2_has_batch_n"
        "_align_M_n_align_N_n_align_K_n",
        "steel_gather_mm_nn_float32_float16_bm64_bn64_bk16_wm2_wn2_has_batch_n"
        "_align_M_n_align_N_n_align_K_n",
        "steel_gather_mm_rhs_tn_float32_float32_bm16_bn64_bk16_wm1_wn2"
        "_align_N_t_align_K_t",
        "steel_gather_mm_rhs_nn_complex64_complex64_bm16_bn64_bk16_wm1_wn2"
        "_align_N_t_align_K_t",
        "gather_mm_offsets_float32",
    ):
        with pytest.raises(ValueError):
            gather_mm_layout.signature(entry)
    for entry in (
        "gemv_gather_float32_bm1_bn8_sm1_sn32_tm4_tn4",
        "gemv_gather_complex64_bm4_bn1_sm1_sn32_tm4_tn4",
        "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
    ):
        with pytest.raises(ValueError):
            gather_mv_layout.signature(entry)
    assert not gather_mm_dispatch.is_entry(
        "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
    )


@pytest.mark.parametrize("transpose", (False, True))
def test_gather_mv_selection_stays_instantiated(transpose):
    for k in (1, 64, 8191, 8192):
        for out in (1, 3, 4, 511, 512, 2047, 2048, 4096):
            for dtype in gather_mv_layout.DTYPES:
                gather_mv_layout.signature(
                    gather_mv_layout.select(dtype, transpose, k, out)
                )


@pytest.mark.parametrize("record", kind("mv"), ids=lambda record: record["name"])
def test_gather_mv_plan_accepts_each_launch(record):
    entry, counts, parameters, views, execution = mv_binding(record)
    metadata = gather_mv_layout.plan(
        entry, counts, parameters, views, counts["out_vec"], execution
    )
    lhs = verify._flat(
        record["lhs"]
        or {
            "shape": record["a"]["shape"][:-2],
            "values": list(range(math.prod(record["a"]["shape"][:-2]))),
        },
        record["indexShape"],
    )
    rhs = verify._flat(record["rhs"], record["indexShape"])
    is_mv = record["a"]["shape"][-2] != 1
    assert metadata["matrixIndices"] == (lhs if is_mv else rhs)
    assert metadata["vectorIndices"] == (rhs if is_mv else lhs)
    assert metadata["inert"] == {"alpha": 1.0, "beta": 0.0}


@pytest.mark.parametrize(
    "fault", ("index-range", "index-read", "matrix", "grid", "entry", "strides")
)
def test_gather_mv_plan_rejects_launches_that_differ_from_mlx(fault):
    record = BY_NAME["column-grid"]
    entry, counts, parameters, views, execution = mv_binding(record)
    if fault == "index-range":
        views["mat_indices"][0] = 3
    elif fault == "index-read":
        counts["vec_indices"] = 1
    elif fault == "matrix":
        counts["mat"] -= 1
    elif fault == "grid":
        execution["workgroupCount"][2] += 1
    elif fault == "entry":
        entry = entry.replace("_bm4_", "_bm8_").replace("_tm4_", "_tm4_")
    else:
        parameters["index_batch_strides"] = parameters["index_batch_strides"][:-1]
    with pytest.raises(ValueError):
        gather_mv_layout.plan(
            entry, counts, parameters, views, counts["out_vec"], execution
        )


@pytest.mark.parametrize("record", kind("mm"), ids=lambda record: record["name"])
def test_gather_mm_plan_accepts_each_launch(record):
    entry, counts, parameters, views, execution = mm_binding(record)
    metadata = gather_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    assert metadata["kind"] == "mm"
    assert metadata["batches"] == math.prod(record["indexShape"])
    assert metadata["rhsIndices"] == verify._flat(record["rhs"], record["indexShape"])


@pytest.mark.parametrize(
    "fault", ("index-range", "index-read", "operand", "params", "grid", "padding")
)
def test_gather_mm_plan_rejects_launches_that_differ_from_mlx(fault):
    record = BY_NAME["matrix-grid-aligned"]
    entry, counts, parameters, views, execution = mm_binding(record)
    if fault == "index-range":
        views["rhs_indices"][1] = 3
    elif fault == "index-read":
        counts["lhs_indices"] = 3
    elif fault == "operand":
        counts["B"] -= 1
    elif fault == "params":
        parameters["params"]["batch_stride_d"] += 1
    elif fault == "grid":
        execution["workgroupCount"][0] += 1
    else:
        parameters["indices_shape"] = [1]
    with pytest.raises(ValueError):
        gather_mm_layout.plan(entry, counts, parameters, views, counts["C"], execution)


def test_gather_mm_scalar_indices_bind_one_unread_element():
    record = BY_NAME["matrix-scalar-transposed"]
    entry, counts, parameters, views, execution = mm_binding(record)
    assert parameters["indices_shape"] == [1]
    assert parameters["lhs_strides"] == parameters["rhs_strides"] == [0]
    gather_mm_layout.plan(entry, counts, parameters, views, counts["C"], execution)
    parameters["lhs_strides"] = [1]
    with pytest.raises(ValueError, match="batch vectors"):
        gather_mm_layout.plan(entry, counts, parameters, views, counts["C"], execution)


@pytest.mark.parametrize("record", kind("rhs"), ids=lambda record: record["name"])
def test_gather_rhs_plan_accepts_each_launch(record):
    entry, counts, parameters, views, execution = rhs_binding(record)
    metadata = gather_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    assert metadata["groups"] == math.prod(record["b"]["shape"][:-2])
    assert metadata["offsets"] == list(views["offsets"])


@pytest.mark.parametrize(
    "fault", ("unsorted", "past-rows", "tiles", "groups", "experts")
)
def test_gather_rhs_plan_rejects_launches_that_differ_from_mlx(fault):
    record = BY_NAME["sorted-rows"]
    entry, counts, parameters, views, execution = rhs_binding(record)
    if fault == "unsorted":
        views["offsets"][1], views["offsets"][2] = 5, 2
    elif fault == "past-rows":
        views["offsets"][3] = 9
    elif fault == "tiles":
        parameters["params"]["tiles_m"] += 1
        execution["workgroupCount"][1] += 1
    elif fault == "groups":
        parameters["num_groups"] = 3
    else:
        counts["B"] -= 1
    with pytest.raises(ValueError):
        gather_mm_layout.plan(entry, counts, parameters, views, counts["C"], execution)


def test_offsets_reference_follows_the_kernel_search():
    assert gather_mm_layout.offsets_reference([0, 0, 1, 1, 1, 3, 3, 3], 4) == [
        0,
        2,
        5,
        5,
    ]
    assert gather_mm_layout.offsets_reference([3], 7) == [0, 0, 0, 0, 1, 1, 1]
    assert gather_mm_layout.offsets_reference([0, 0x80000000, 0xFFFFFFFF], 2) == [0, 1]
    entry = gather_mm_layout.OFFSETS_ENTRY
    execution = {"workgroupCount": [4, 1, 1], "workgroupSize": [1, 1, 1]}
    counts = {"indices": 8, "offsets": 4}
    gather_mm_layout.plan(entry, counts, {"M": 8}, {}, 4, execution)
    with pytest.raises(ValueError, match="past their indices"):
        gather_mm_layout.plan(entry, counts, {"M": 9}, {}, 4, execution)
    with pytest.raises(ValueError, match="grid"):
        gather_mm_layout.plan(
            entry, counts, {"M": 8}, {}, 4, {**execution, "workgroupSize": [4, 1, 1]}
        )


def native_buffers(names, types, counts, parameters, views, output):
    memory, buffers = {}, []
    for name in names:
        dtype = types(name)
        if name in views:
            memory[name] = views[name]
            data, count = ctypes.addressof(views[name]), len(views[name])
        else:
            if name == "params":
                value = list(parameters[name].values())
            else:
                value = [0] * counts[name] if name in counts else parameters[name]
            values = value if isinstance(value, list) else [value]
            memory[name] = (CTYPES[dtype] * len(values))(*values)
            data, count = ctypes.addressof(memory[name]), len(values)
        buffers.append(
            runtime.Buffer(
                name.encode(), dtype.encode(), data, count, int(name == output)
            )
        )
    return {buffer.name.decode(): buffer for buffer in buffers}, memory


@pytest.mark.parametrize(
    "name, binding, layout, output",
    (
        ("column-grid", mv_binding, gather_mv_layout, "out_vec"),
        ("matrix-grid-aligned", mm_binding, gather_mm_layout, "C"),
        ("sorted-broadcast-transposed", rhs_binding, gather_mm_layout, "C"),
    ),
)
def test_storage_reads_every_gathered_binding(name, binding, layout, output):
    entry, counts, parameters, views, execution = binding(BY_NAME[name])
    names = (
        layout.HOST_PARAMETERS
        if layout is gather_mv_layout
        else layout.host_parameters(entry)
    )
    supplied, memory = native_buffers(
        names,
        lambda item: layout.parameter_type(entry, item),
        counts,
        parameters,
        views,
        output,
    )
    read_counts, values, read_views = layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    assert {key: list(value) for key, value in read_views.items()} == {
        key: list(value) for key, value in views.items()
    }
    layout.plan(entry, read_counts, values, read_views, counts[output], execution)
    broken = dict(supplied)
    broken.pop(names[0])
    with pytest.raises(ValueError, match="do not match"):
        layout.read_storage(entry, broken)
    flipped = dict(supplied)
    flipped[names[0]] = runtime.Buffer(
        supplied[names[0]].name,
        supplied[names[0]].dtype,
        supplied[names[0]].data,
        supplied[names[0]].count,
        1,
    )
    with pytest.raises(ValueError, match="direction"):
        layout.read_storage(entry, flipped)
    del memory


def test_dispatch_storage_binds_every_reflected_parameter():
    record = BY_NAME["column-grid"]
    entry, counts, parameters, views, execution = mv_binding(record)
    supplied, memory = native_buffers(
        gather_mv_layout.HOST_PARAMETERS,
        lambda item: gather_mv_layout.parameter_type(entry, item),
        counts,
        parameters,
        views,
        "out_vec",
    )
    metadata = gather_mv_layout.plan(
        entry, counts, parameters, views, counts["out_vec"], execution
    )
    storage = gather_mm_dispatch._storage(
        entry, "mv", supplied, counts, parameters, metadata
    )
    assert set(storage) == set(gather_mv_layout.PARAMETERS)
    assert storage["out_vec"][-1] == gather_mm_dispatch.GUARDS["bfloat16"]
    entry, counts, parameters, views, execution = rhs_binding(BY_NAME["sorted-rows"])
    supplied, memory = native_buffers(
        gather_mm_layout.RHS_PARAMETERS,
        lambda item: gather_mm_layout.parameter_type(entry, item),
        counts,
        parameters,
        views,
        "C",
    )
    metadata = gather_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    storage = gather_mm_dispatch._storage(
        entry, "rhs", supplied, counts, parameters, metadata
    )
    assert set(storage) == set(gather_mm_layout.RHS_PARAMETERS)
    assert storage["offsets"][:4] == [0, 2, 5, 5]
    assert storage["offsets"][-1] == gather_mm_dispatch.INT32_POISON
    del memory


def test_packages_follow_the_kernel_gates():
    root = Path("/pinned")
    mv = BY_NAME["column-grid"]["entries"][0]
    mm = BY_NAME["matrix-grid-aligned"]["entries"][0]
    rhs = BY_NAME["sorted-rows"]["entries"][1]
    for target in ("metal", "directx", "opengl"):
        config = gather_mm_packages.gather_mv_config(root, "out", target, mv)
        assert bool(config.index_range_assertions) == (target == "opengl")
        for entry in (mm, rhs):
            config = gather_mm_packages.gather_mm_config(root, "out", target, entry)
            assert (
                dict(config.specialization_constants)
                == gather_mm_layout.signature(entry)["constants"]
            )
            assert bool(config.workgroup_access_assertions) == (target != "metal")
            assert bool(config.index_range_assertions) == (target == "opengl")
        config = gather_mm_packages.gather_offsets_config(
            root, "out", target, "gather_mm_offsets"
        )
        assert tuple(config.workgroup_size) == (1, 1, 1)
    assert gather_mm_packages.cache_for(mv)[1] == "gather-mv"
    assert gather_mm_packages.cache_for(mm)[1] == "gather-mm"
    assert gather_mm_packages.cache_for("gather_mm_offsets")[1] == "gather-mm-offsets"
    assert set(gather_mm_packages.verification_entries()) == {
        entry for record in RECORDS for entry in record["entries"]
    }


def test_runtime_routes_gather_entries_before_gemv_and_gathers(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, object()
    host.gather_mvs = host.gather_mms = host.gather_offsets = None
    entries = [BY_NAME["column-grid"]["entries"][0], *BY_NAME["sorted-rows"]["entries"]]
    for entry in entries:
        assert host._entry_available(entry.encode()) == 0
    host.gather_mvs = host.gather_mms = host.gather_offsets = object()
    for entry in entries:
        assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"gemv_gather_float32_bm1_bn1_sm1_sn32_tm4_tn4") == 0
    calls = []
    monkeypatch.setattr(
        gather_mm_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    for entry in entries:
        host.dispatch(entry, [], 0, 1, launch=None)
    assert calls == entries


def trace_events(record, target):
    route = record["route"]
    if record["kind"] == "empty":
        return [{"entry": "s_copyfloat32float32", "target": target}]
    events = [{"entry": "arangeuint32", "target": target}] * record["defaults"]
    events += [{"entry": "v_copyfloat32float32", "target": target}] * record["copies"]
    lhs, _ = index_view(record, "lhs")
    lhs_flat = verify._flat(
        record["lhs"] or {"shape": record["a"]["shape"][:-2], "values": lhs},
        record["indexShape"],
    )
    rhs_flat = verify._flat(record["rhs"], record["indexShape"])
    if record["kind"] == "rhs":
        groups = math.prod(record["b"]["shape"][:-2])
        events.append(
            {
                "entry": record["entries"][0],
                "target": target,
                "gatherMetadata": {"groups": groups},
            }
        )
        if target in verify.RHS_UNSUPPORTED:
            return events
        events.append(
            {
                "entry": record["entries"][1],
                "target": target,
                "gatherMetadata": {
                    "offsets": gather_mm_layout.offsets_reference(rhs_flat, groups),
                    "M": route["M"],
                    "ldb": route["ldb"],
                },
            }
        )
        return events
    if record["kind"] == "mv":
        is_mv = record["a"]["shape"][-2] != 1
        metadata = {
            "matrixIndices": lhs_flat if is_mv else rhs_flat,
            "vectorIndices": rhs_flat if is_mv else lhs_flat,
            **{key: route[key] for key in ("K", "outVecSize", "matrixLd")},
        }
    else:
        metadata = {
            "lhsIndices": lhs_flat,
            "rhsIndices": rhs_flat,
            "lda": route["lda"],
            "ldb": route["ldb"],
        }
    return events + [
        {"entry": record["entries"][0], "target": target, "gatherMetadata": metadata}
    ]


def traced(records, target):
    events, placed = [], []
    for record in records:
        produced = trace_events(record, target)
        extra = (
            {"unsupported": verify.RHS_UNSUPPORTED[target]}
            if (record["kind"] == "rhs" and target in verify.RHS_UNSUPPORTED)
            else {"outputHash": record["name"]}
        )
        placed.append(
            {
                **record,
                **extra,
                "dispatchStart": len(events),
                "dispatchCount": len(produced),
            }
        )
        events += produced
    return events, placed


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_trace_validation_requires_the_routed_gathers(target):
    events, records = traced(RECORDS, target)
    verify.validate_trace(events, records, target)
    other = "opengl" if target != "opengl" else "metal"
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(events, records, other)
    for name, change in (
        (
            "matrix-vector-batch",
            lambda event: event["gatherMetadata"]["rhsIndices"].reverse(),
        ),
        ("column-grid", lambda event: event["gatherMetadata"].update(matrixLd=1)),
    ):
        changed = copy.deepcopy(events)
        record = BY_NAME[name]
        placed = next(r for r in records if r["name"] == name)
        change(changed[placed["dispatchStart"] + placed["dispatchCount"] - 1])
        with pytest.raises(RuntimeError, match="differs"):
            verify.validate_trace(changed, records, target)
        assert record["entries"]
    changed = copy.deepcopy(events)
    placed = next(r for r in records if r["name"] == "matrix-copied-default")
    changed[placed["dispatchStart"]]["entry"] = "v_copyfloat32float32"
    with pytest.raises(RuntimeError, match="copies differ"):
        verify.validate_trace(changed, records, target)


def test_backend_ports_gather_mm_eval_gpu():
    port = (HOST / "gather_mm.inc").read_text(encoding="utf-8")
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    tail = backend[backend.rindex("} // namespace mlx::core::fast") :]
    assert '#include "mlx/backend/no_gpu/crosstl_gather_mm.inc"\n' in tail
    start = port.index("void GatherMM::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        "if (a.size() == 0 || b.size() == 0) {",
        "if (M == 1 && right_sorted_) {",
        "dispatch_gather_mv(b, a, rhs_indices, lhs_indices, out, N, K, false);",
        "dispatch_gather_mv(a, b, lhs_indices, rhs_indices, out, M, K, true);",
        "dispatch_gather_mm(a, b, lhs_indices, rhs_indices, out, M, N, K);",
    ]
    assert [body.index(text) for text in order] == sorted(
        body.index(text) for text in order
    )
    for name in (
        *gather_mv_layout.HOST_PARAMETERS,
        *gather_mm_layout.PARAMETERS,
        *gather_mm_layout.RHS_PARAMETERS,
        *gather_mm_layout.OFFSETS_PARAMETERS,
    ):
        assert f'{{"{name}", ' in port, name
    for text in (
        "transpose_mat = (!is_mv) ^ transpose_mat;",
        "index_strides = {0, 0};",
        "std::min<int64_t>(M, (int64_t(M) + bm - 1) / bm + num_groups - 1)",
        "broadcast(a_, broadcast_a);",
        "{{uint32_t(num_groups), 1, 1}, {1, 1, 1}}",
    ):
        assert text in port, text


def test_preparation_replaces_the_gather_mm_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    sources = prepare.adapted_sources("", "", "")
    assert "NO_GPU(GatherMM)" in replaced
    assert sources["crosstl_gather_mm.inc"] == (HOST / "gather_mm.inc").read_bytes()


def test_ci_runs_the_gather_mm_route_on_every_quantized_host_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    steps = workflow["jobs"]["quantized-host"]["steps"]
    names = [item.get("name") for item in steps]
    execute = steps[names.index("Execute GatherMM products through MLX")]
    assert (
        names.index("Execute AddMM products through MLX")
        < names.index("Execute GatherMM products through MLX")
        < names.index("Retain quantized execution evidence")
    )
    assert "if" not in execute and "continue-on-error" not in execute
    assert "set -euo pipefail" in execute["run"]
    assert "portable_host.verify_gather_mm" in execute["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in execute["run"]
    assert "tee .mlx-portable-quantized/gather-mm.log" in execute["run"]
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    validate = steps[names.index("Validate quantized host contracts")]["run"]
    assert "test_portable_gather_mm.py" in validate
