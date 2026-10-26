"""BlockMaskedMM host routing, masked launch contracts and their verification cases."""

import copy
import ctypes
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    block_masked_mm_dispatch,
    block_masked_mm_layout,
    block_masked_mm_packages,
    lazy_package_evidence,
    prepare,
    runtime,
    upstream_matrix_packages,
)
from demos.integrations.mlx.portable_host import verify_block_masked_mm as verify
from demos.integrations.mlx.portable_host import verify_matmul
from demos.integrations.mlx.tests.kernels import test_current_gemm_masked as gemm_gate
from demos.integrations.mlx.tests.kernels import test_current_gemv_masked as gemv_gate

layout = block_masked_mm_layout
HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
BY_NAME = {record["name"]: record for record in RECORDS}
PRODUCTS = [record for record in RECORDS if record["kind"].startswith("masked")]
CTYPES = {
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "bool_": ctypes.c_uint8,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def bound_arrays(record):
    """The (shape, strides) of every array block_masked_mm.inc binds."""
    m, k = record["a"]["shape"][-2:]
    n = record["b"]["shape"][-1]
    batch = record["outShape"][:-2]
    arrays = {}
    for key, dims in (("a", (m, k)), ("b", (k, n))):
        shape, strides = verify._broadcast(
            record[key]["shape"], record[key]["strides"], [*batch, *dims]
        )
        if record["route"]["copies"].get(key):
            strides = verify_matmul._row_strides(shape)
        arrays[key] = (shape, strides)
    for which, stored in record["masks"]["shapes"].items():
        rows, columns = stored[-2:]
        arrays[which] = verify._broadcast(
            stored, verify_matmul._row_strides(stored), [*batch, rows, columns]
        )
    return arrays


def launch(record):
    """``(entry, counts, parameters, execution)`` as block_masked_mm.inc binds them."""
    (entry,) = record["entries"]
    info = layout.signature(entry)
    arrays = bound_arrays(record)
    route = record["route"]
    m, n, k = route["M"], route["N"], route["K"]
    batch = record["outShape"][:-2]
    order = ["a", "b"] + [name for name in ("out", "lhs", "rhs") if name in arrays]
    if batch:
        bstrides = {name: arrays[name][1][:-2] for name in order}
        shape = list(batch)
    else:
        bstrides = {name: [0] for name in order}
        shape = [1]
    batches = verify._prod(shape)
    names = {"out": "out_mask"}
    if info["kind"] == "masked-mm":
        names.update(lhs="lhs_mask", rhs="rhs_mask")
        counts = {
            "A": span(*arrays["a"]),
            "B": span(*arrays["b"]),
            "D": verify._prod(record["outShape"]),
        }
        mask_strides = []
        for which in ("out", "lhs", "rhs"):
            if which in arrays:
                strides = arrays[which][1]
                mask_strides += [strides[-1], strides[-2]]
                counts[names[which]] = span(*arrays[which])
        tiles_n, tiles_m = -(-n // info["bn"]), -(-m // info["bm"])
        params = dict(
            M=m,
            N=n,
            K=k,
            lda=route["lda"],
            ldb=route["ldb"],
            ldd=n,
            tiles_n=tiles_n,
            tiles_m=tiles_m,
            batch_stride_a=bstrides["a"][-1] if batch else 0,
            batch_stride_b=bstrides["b"][-1] if batch else 0,
            batch_stride_d=m * n,
            swizzle_log=0,
            gemm_k_iterations_aligned=k // 16,
            batch_ndim=len(shape),
        )
        parameters = {
            "params": params,
            "batch_shape": shape,
            "batch_strides": [value for name in order for value in bstrides[name]],
            "mask_strides": mask_strides,
        }
        execution = [tiles_n, tiles_m, batches]
        return entry, counts, parameters, _execution(entry, execution)
    is_b_matrix = n != 1
    mat, vec = ("b", "a") if is_b_matrix else ("a", "b")
    mat_mask, vec_mask = ("rhs", "lhs") if is_b_matrix else ("lhs", "rhs")
    transpose_mat = info["kernel"] == "gemv_t"
    out_size = n if is_b_matrix else m
    # gemv_operand on the operand before any copy, as eval_gpu calls it.
    dims = (k, n) if is_b_matrix else (m, k)
    pre = verify._broadcast(
        record[mat]["shape"], record[mat]["strides"], [*batch, *dims]
    )
    ld = layout._check_transpose(*pre, (n if is_b_matrix else m) == 1)[1]
    counts = {
        "mat": span(*arrays[mat]),
        "in_vec": span(*arrays[vec]),
        "out_vec": verify._prod(record["outShape"]),
    }
    mask_strides, mask_batch = [], []
    out_rows = -1 if record["outShape"][-2 if transpose_mat else -1] == 1 else -2
    if "out" in arrays:
        strides = arrays["out"][1]
        mask_strides += [strides[out_rows], strides[-3 - out_rows]]
        mask_batch += bstrides["out"]
        counts["out_mask"] = span(*arrays["out"])
    if mat_mask in arrays:
        strides = arrays[mat_mask][1]
        first = (
            (-2 if not is_b_matrix else -1)
            if transpose_mat
            else (-2 if is_b_matrix else -1)
        )
        mask_strides += [strides[first], strides[-3 - first]]
        mask_batch += bstrides[mat_mask]
        vec_shape = arrays[vec][0]
        strides = arrays[vec_mask][1]
        axis = -2 if transpose_mat else -1
        first = -1 if vec_shape[axis] == 1 else -2
        mask_strides += [strides[first], strides[-3 - first]]
        mask_batch += bstrides[vec_mask]
        counts["mat_mask"] = span(*arrays[mat_mask])
        counts["vec_mask"] = span(*arrays[vec_mask])
    parameters = {
        "in_vec_size": k,
        "out_vec_size": out_size,
        "marix_ld": ld,
        "batch_ndim": len(shape),
        "batch_shape": shape,
        "vector_batch_stride": list(bstrides[vec]),
        "matrix_batch_stride": list(bstrides[mat]),
        "mask_strides": mask_strides,
        "mask_batch_strides": mask_batch,
    }
    tiles = -(-out_size // layout.outputs_per_group(entry))
    return entry, counts, parameters, _execution(entry, [tiles, 1, batches])


def _execution(entry, count):
    return {
        "workgroupCount": list(count),
        "workgroupSize": list(layout.workgroup_size(entry)),
    }


def output_of(entry):
    return layout.OUTPUTS[layout.kind_of(entry)]


def test_layout_names_every_instantiated_entry():
    assert {
        entry: layout.signature(entry)["kernel"] for entry in gemv_gate.ENTRIES
    } == {entry: info["kind"] for entry, info in gemv_gate.INFO.items()}
    for entry in gemm_gate.ENTRIES:
        info = layout.signature(entry.name)
        assert info["model"] == entry and info["block"] == entry.block
    assert len(gemm_gate.ENTRIES) == 768 and len(gemv_gate.ENTRIES) == 528
    for entry in (
        "gemv_outmask_nomask_opmask_nomask_float32_bm2_bn1_sm4_sn8_tm4_tn4_nc0",
        "gemv_outmask_float16_opmask_float16_float32_bm2_bn1_sm4_sn8_tm4_tn4_nc0",
        "gemv_t_outmask_bool__opmask_bool__float32_bm1_bn2_sm8_sn4_tm4_tn1_nc0",
        "gemv_outmask_bool__opmask_bool__float32_bm2_bn1_sm4_sn8_tm4_tn4_nc2",
        "steel_gemm_block_outmask_bool__opmask_bool__nn_float32_float32_bm16_bn16"
        "_bk16_wm2_wn2_MN_naligned_K_naligned",
        "gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


@pytest.mark.parametrize("block", layout.BLOCK_SIZES)
@pytest.mark.parametrize("transpose_mat", (False, True))
def test_gemv_selection_follows_eval_gpu(block, transpose_mat):
    for out_size in (1, 3, 4, 100, 511, 512, 2047, 2048, 5000):
        entry = layout.select_gemv(
            "float16", transpose_mat, out_size, block, "bool_", "nomask", False
        )
        instantiated = verify._instantiated({"entries": [entry]})
        # gemv_masked.metal instantiates no gemv_t tile for fewer than four
        # outputs at matmul.cpp's two simdgroup columns, so MLX cannot run them.
        assert instantiated == (not transpose_mat or out_size >= 4), entry
        if instantiated:
            assert layout.block_size_of(entry) == block
            assert layout.signature(entry)["block"] == block


def test_cases_cover_every_masked_route():
    assert {record["kind"] for record in RECORDS} == set(verify.KERNELS)
    placed = [
        (
            {**record, "outputHash": ""}
            if record["kind"] != "missing"
            else {**record, "refused": True}
        )
        for record in RECORDS
    ]
    assert verify._covered(placed)
    assert not verify._covered(
        [record for record in placed if record["name"] != "steel-copied-a"]
    )


@pytest.mark.parametrize("record", PRODUCTS, ids=lambda record: record["name"])
def test_plan_accepts_each_launch(record):
    entry, counts, parameters, execution = launch(record)
    metadata = layout.plan(
        entry, counts, parameters, {}, counts[output_of(entry)], execution
    )
    assert metadata["kind"] == record["kind"]
    assert metadata["blockSize"] == record["blockSize"]
    assert metadata["batches"] == verify._prod(record["outShape"][:-2])
    event = {"maskedMetadata": metadata}
    assert verify._launch_matches(event, record)


@pytest.mark.parametrize(
    "name, fault, message",
    (
        ("steel-bool-unaligned", "out-mask", "out_mask read exceeds"),
        ("steel-bool-unaligned", "lhs-mask", "lhs_mask read exceeds"),
        ("steel-transposed-grid", "rhs-mask", "rhs_mask read exceeds"),
        ("steel-bool-unaligned", "operand", "A batch exceeds"),
        ("steel-bool-unaligned", "params", "parameters differ"),
        ("steel-bool-unaligned", "grid", "grid"),
        ("steel-bool-unaligned", "routing", "routing"),
        ("steel-bool-unaligned", "output", "output extent differs"),
        ("steel-bool-unaligned", "negative", "mask strides differ"),
        ("steel-transposed-b-batched", "batch-stride", "batch strides differ"),
        ("gemv-t-row", "out-mask", "out_mask read exceeds"),
        ("gemv-t-row", "vec-mask", "vec_mask read exceeds"),
        ("gemv-column-tall", "mat-mask", "mat_mask read exceeds"),
        ("gemv-t-grid", "variant", "batch variant"),
        ("gemv-column-batched", "routing", "routing"),
        ("gemv-t-row", "grid", "grid"),
        ("gemv-column-batched", "operand", "in_vec exceeds"),
        ("gemv-column-batched", "mask-batch", "mask batch strides differ"),
    ),
)
def test_plan_rejects_launches_that_differ_from_mlx(name, fault, message):
    entry, counts, parameters, execution = launch(BY_NAME[name])
    output = output_of(entry)
    if fault.endswith("-mask"):
        counts[fault.replace("-", "_")] -= 1
    elif fault == "operand":
        counts["A" if "A" in counts else "in_vec"] -= 1
    elif fault == "params":
        parameters["params"]["batch_stride_d"] += 1
    elif fault == "grid":
        execution["workgroupCount"][2] += 1
    elif fault == "routing":
        # The same grid with another instantiated tile.
        if "MN_" in entry:
            entry = entry.replace("_MN_naligned", "_MN_taligned")
        else:
            entry = entry.replace("_bm2_bn1_sm4_sn8_", "_bm4_bn1_sm2_sn16_")
            execution["workgroupSize"] = list(layout.workgroup_size(entry))
    elif fault == "output":
        counts[output] += 1
    elif fault == "negative":
        parameters["mask_strides"][0] = -1
    elif fault == "batch-stride":
        parameters["params"]["batch_stride_a"] += 1
    elif fault == "variant":
        entry = entry.replace("_nc1", "_nc0")
    else:
        parameters["mask_batch_strides"].append(0)
    with pytest.raises(ValueError, match=message):
        layout.plan(entry, counts, parameters, {}, counts[output], execution)


def test_gemv_t_reads_its_output_mask_by_block_index():
    """gemv_t_masked ignores the output mask strides matmul.cpp binds."""
    entry, counts, parameters, execution = launch(BY_NAME["gemv-t-row"])
    parameters["mask_strides"][1] = 1000
    layout.plan(entry, counts, parameters, {}, counts["out_vec"], execution)
    entry, counts, parameters, execution = launch(BY_NAME["gemv-column-tall"])
    parameters["mask_strides"][1] = 1000
    with pytest.raises(ValueError, match="out_mask read exceeds"):
        layout.plan(entry, counts, parameters, {}, counts["out_vec"], execution)


def native_buffers(entry, counts, parameters):
    memory, buffers = {}, {}
    output = output_of(entry)
    for name in layout.host_parameters(entry):
        dtype = layout.parameter_type(entry, name)
        if name in counts:
            values = [0] * counts[name]
        elif name == "params":
            values = list(parameters[name].values())
        else:
            value = parameters[name]
            values = [value] if isinstance(value, int) else list(value)
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            len(values),
            int(name == output),
        )
    return buffers, memory


@pytest.mark.parametrize("name", ("steel-transposed-grid", "gemv-t-row"))
def test_storage_reads_every_bound_parameter(name):
    entry, counts, parameters, execution = launch(BY_NAME[name])
    supplied, memory = native_buffers(entry, counts, parameters)
    read_counts, values, views = layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters and views == {}
    layout.plan(entry, read_counts, values, views, counts[output_of(entry)], execution)
    absent = layout.absent_masks(entry)
    assert set(supplied) == set(layout.parameters(entry)) - set(absent)
    broken = dict(supplied)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="do not match"):
        layout.read_storage(entry, broken)
    mask = "lhs_mask" if "lhs_mask" in supplied else "mat_mask"
    retyped = dict(supplied)
    retyped[mask] = runtime.Buffer(mask.encode(), b"float32", supplied[mask].data, 1, 0)
    with pytest.raises(ValueError, match="storage"):
        layout.read_storage(entry, retyped)
    output = output_of(entry)
    overlapping = dict(supplied)
    overlapping[output] = runtime.Buffer(
        output.encode(), supplied[output].dtype, supplied[mask].data, counts[output], 1
    )
    with pytest.raises(ValueError, match="overlaps"):
        layout.read_storage(entry, overlapping)
    del memory


class _Host:
    def __init__(self, target):
        self.target = target


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_dispatch_binds_masks_as_the_target_stores_them(target):
    entry, counts, parameters, execution = launch(BY_NAME["gemv-short"])
    supplied, memory = native_buffers(entry, counts, parameters)
    ctypes.memmove(supplied["mat_mask"].data, bytes([1, 0]), 2)
    storage = block_masked_mm_dispatch._storage(entry, supplied, counts, parameters)
    assert storage["mat_mask"][:2] == [1, 0]
    assert storage["mat_mask"][2:] == [1] * block_masked_mm_dispatch.GUARD_COUNT
    assert storage["out_vec"][-1] == block_masked_mm_dispatch.GUARDS["float16"]
    host = _Host(target)
    bool_type, bool_stride = block_masked_mm_dispatch.BOOL_TYPES[target]
    inert_type, inert_stride = block_masked_mm_dispatch.INERT_TYPES[target]
    absent = set(layout.absent_masks(entry))
    assert absent == {"out_mask"}
    value = block_masked_mm_dispatch._value(
        host,
        entry,
        "mat_mask",
        storage,
        {"scalarLayout": {"elementType": bool_type, "elementStrideBytes": bool_stride}},
        absent,
    )
    assert value["dtype"] == bool_type
    assert value["values"][:2] == ([True, False] if bool_type == "bool" else [1, 0])
    inert = block_masked_mm_dispatch._value(
        host,
        entry,
        "out_mask",
        storage,
        {
            "scalarLayout": {
                "elementType": inert_type,
                "elementStrideBytes": inert_stride,
            }
        },
        absent,
    )
    assert inert == {"dtype": inert_type, "shape": [4], "values": [0, 0, 0, 0]}
    with pytest.raises(ValueError, match="mask layout differs"):
        block_masked_mm_dispatch._value(
            host,
            entry,
            "out_mask",
            storage,
            {"scalarLayout": {"elementType": "uint8", "elementStrideBytes": 1}},
            absent,
        )
    ctypes.memmove(supplied["mat_mask"].data, bytes([2]), 1)
    with pytest.raises(ValueError, match="not canonical"):
        block_masked_mm_dispatch._storage(entry, supplied, counts, parameters)
    del memory


def test_packages_follow_the_kernel_gates(tmp_path):
    root = Path("/pinned")
    for record in PRODUCTS:
        (entry,) = record["entries"]
        for target in ("metal", "directx", "opengl"):
            config = block_masked_mm_packages.cache_for(root, tmp_path, target, entry)
            built = config.project_config("out", entry)
            if layout.kind_of(entry) == "masked-mv":
                gate = gemv_gate._config(tmp_path / "work", target, entry)
                assert (
                    built.entry_workgroup_size_rules == gate.entry_workgroup_size_rules
                )
            else:
                gate = gemm_gate._config(
                    root, "out", target, gemm_gate.model.BY_NAME[entry]
                )
                assert built.workgroup_size_rules == gate.workgroup_size_rules
                assert built.immutable_storage == gate.immutable_storage
                assert (
                    built.workgroup_access_assertions
                    == gate.workgroup_access_assertions
                )
            assert built.entry_points == gate.entry_points
            assert built.source_options == gate.source_options
            assert [item.to_json() for item in built.index_range_assertions] == [
                item.to_json() for item in gate.index_range_assertions
            ]
    assert set(block_masked_mm_packages.verification_entries()) == {
        entry for record in RECORDS for entry in record["entries"]
    }


def test_runtime_routes_masked_entries_before_gemv(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, object()
    host.gather_mvs = host.gather_mms = host.gather_offsets = object()
    host.segmented_mms = host.gemvs = object()
    host.masked_mvs = host.masked_mms = None
    entries = [record["entries"][0] for record in PRODUCTS]
    for entry in entries:
        assert host._entry_available(entry.encode()) == 0
    host.masked_mvs = host.masked_mms = object()
    for entry in entries:
        assert host._entry_available(entry.encode()) == 1
    (missing,) = BY_NAME["gemv-t-uninstantiated"]["route"]["missing"]
    assert host._entry_available(missing.encode()) == 0
    calls = []
    monkeypatch.setattr(
        block_masked_mm_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    for entry in entries:
        host.dispatch(entry, [], 0, 1, launch=None)
    assert calls == entries
    for entry in entries:
        family = lazy_package_evidence.family(entry)
        assert family == layout.kind_of(entry)
        assert family in upstream_matrix_packages.ORDER


def trace_events(record, target):
    copies = [{"entry": "g2_copyfloat32float32", "target": target}] * record["copies"]
    if record["kind"] == "empty":
        return copies + [{"entry": "s_copyfloat32float32", "target": target}]
    if record["kind"] in ("none", "missing"):
        return []
    entry, counts, parameters, execution = launch(record)
    metadata = layout.plan(
        entry, counts, parameters, {}, counts[output_of(entry)], execution
    )
    return copies + [{"entry": entry, "target": target, "maskedMetadata": metadata}]


def traced(records, target):
    events, placed = [], []
    for record in records:
        produced = trace_events(record, target)
        placed.append(
            {
                **record,
                "outputHash": record["name"],
                "refused": record["kind"] == "missing",
                "dispatchStart": len(events),
                "dispatchCount": len(produced),
            }
        )
        events += produced
    return events, placed


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_trace_validation_requires_the_routed_launches(target):
    events, records = traced(RECORDS, target)
    verify.validate_trace(events, records, target)
    other = "opengl" if target != "opengl" else "metal"
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(events, records, other)
    for name, change in (
        ("steel-bool-unaligned", lambda event: event["maskedMetadata"].update(lda=1)),
        ("gemv-t-row", lambda event: event["maskedMetadata"].update(blockSize=64)),
        ("gemv-short", lambda event: event["maskedMetadata"].update(opMask="scale")),
    ):
        changed = copy.deepcopy(events)
        placed = next(r for r in records if r["name"] == name)
        change(changed[placed["dispatchStart"] + placed["dispatchCount"] - 1])
        with pytest.raises(RuntimeError, match="differs"):
            verify.validate_trace(changed, records, target)
    changed = copy.deepcopy(events)
    placed = next(r for r in records if r["name"] == "steel-copied-a")
    changed[placed["dispatchStart"]]["entry"] = "arangeuint32"
    with pytest.raises(RuntimeError, match="copies differ"):
        verify.validate_trace(changed, records, target)


def test_reference_follows_the_masked_model():
    """verify's reference is gemm_masked_model.product on every case kind."""
    import numpy as np

    for name in ("steel-bool-unaligned", "steel-scale-aligned", "steel-copied-a"):
        record = BY_NAME[name]
        data = verify.case_data(np, record, 3)
        a = verify_matmul._view(np, data["a"], record["a"]["view"])
        b = verify_matmul._view(np, data["b"], record["b"]["view"])
        (entry,) = record["entries"]
        found = layout.signature(entry)["model"]
        masks = data["masks"]
        m, k = a.shape
        n = b.shape[1]

        def reader(which, masks=masks):
            return None if which not in masks else (lambda r, c: masks[which][r, c])

        product = gemm_gate.model.product(
            found,
            m,
            n,
            k,
            lambda r, i: a[r, i],
            lambda i, c: b[i, c],
            reader("out"),
            reader("lhs"),
            reader("rhs"),
        )
        expected = np.zeros((m, n))
        for (row, column), value in product.items():
            expected[row, column] = value
        np.testing.assert_array_equal(
            data["expected"],
            verify.rounded(expected + 0.0, record["dtype"]),
        )
    assert verify.case_data(np, BY_NAME["empty-output"], 1)["expected"].shape == (0, 7)


def test_backend_ports_block_masked_mm_eval_gpu():
    port = (HOST / "block_masked_mm.inc").read_text(encoding="utf-8")
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_segmented_mm.inc"\n'
        "// So does the port of BlockMaskedMM's masked products.\n"
        '#include "mlx/backend/no_gpu/crosstl_block_masked_mm.inc"\n'
        "// And the port of the general Scatter, which tiles and orders its launches.\n"
        '#include "mlx/backend/no_gpu/crosstl_scatter.inc"\n'
        "// And the port of Convolution, whose explicit GEMM routes use the steel GEMM.\n"
        '#include "mlx/backend/no_gpu/crosstl_conv.inc"\n'
        "// And the port of the fused cross-entropy, which shares the row helpers.\n"
        '#include "mlx/backend/no_gpu/crosstl_cross_entropy.inc"\n'
        "// And the port of FFT, whose inputs take the strided copy.\n"
        '#include "mlx/backend/no_gpu/crosstl_fft.inc"'
    )
    start = port.index("void BlockMaskedMM::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        "if (a_pre.size() == 0 || b_pre.size() == 0) {",
        "auto [transpose_a, a_cols, a] = gemv_operand(a_pre, M == 1);",
        "auto [transpose_b, b_cols, b] = gemv_operand(b_pre, N == 1);",
        "if (out.ndim() > 2) {",
        "if (std::min(M, N) == 1) {",
        "dispatch_masked_gemv(",
        "dispatch_masked_gemm(",
    ]
    assert [body.index(text) for text in order] == sorted(
        body.index(text) for text in order
    )
    for name in (*layout.GEMV_PARAMETERS, *layout.GEMM_PARAMETERS):
        assert f'{{"{name}", ' in port, name
    for text in (
        "bn = (block_size == 64 && out_vector_len >= 2048) ? 4 : 2;",
        "tm = block_size == 32 ? 4 : 8;",
        "bm = out_vector_len >= 512 ? 4 : 2;",
        "{uint32_t(n_tgp), 1, uint32_t(batch_size_out)}, {32, uint32_t(bn), uint32_t(bm)}};",
        "M, N, K, ops.a_cols, ops.b_cols, N, tn, tm, A_batch_str, B_batch_str,",
        "{32, uint32_t(wn), uint32_t(wm)}};",
    ):
        assert text in port, text


def test_preparation_replaces_the_block_masked_mm_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    sources = prepare.adapted_sources("", "", "")
    assert "NO_GPU(BlockMaskedMM)" in replaced
    assert (
        sources["crosstl_block_masked_mm.inc"]
        == (HOST / "block_masked_mm.inc").read_bytes()
    )


def test_ci_runs_the_block_masked_mm_route_on_every_quantized_host_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    steps = workflow["jobs"]["quantized-host"]["steps"]
    names = [item.get("name") for item in steps]
    step = "Execute BlockMaskedMM products through MLX"
    execute = steps[names.index(step)]
    assert (
        names.index("Execute SegmentedMM products through MLX")
        < names.index(step)
        < names.index("Retain quantized execution evidence")
    )
    assert "if" not in execute and "continue-on-error" not in execute
    assert "set -euo pipefail" in execute["run"]
    assert "portable_host.block_masked_mm_packages" in execute["run"]
    assert "portable_host.verify_block_masked_mm" in execute["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in execute["run"]
    assert "tee .mlx-portable-quantized/block-masked-mm.log" in execute["run"]
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    validate = steps[names.index("Validate quantized host contracts")]["run"]
    assert "test_portable_block_masked_mm.py" in validate
