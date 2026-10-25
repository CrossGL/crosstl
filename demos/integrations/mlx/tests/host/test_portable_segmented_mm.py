"""SegmentedMM host routing, segmented launch contracts and their verification cases."""

import copy
import ctypes
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    gather_mm_dispatch,
    prepare,
    runtime,
    segmented_mm_layout,
    segmented_mm_packages,
    verify_matmul,
)
from demos.integrations.mlx.portable_host import verify_segmented_mm as verify
from demos.integrations.mlx.tests.kernels import gemm_segmented_model as model
from demos.integrations.mlx.tests.kernels.test_current_gemm_segmented import (
    _index_assertions,
    _k_tiles,
)

HOST = Path(segmented_mm_layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
BY_NAME = {record["name"]: record for record in RECORDS}
CTYPES = {
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int64": ctypes.c_int64,
    "uint32": ctypes.c_uint32,
}


def kind(name):
    return [record for record in RECORDS if record["kind"] == name]


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def bound(record, name):
    """The (shape, strides) segmented_mm.inc binds: copies are row-major."""
    shape, strides = record[name]["shape"], record[name]["strides"]
    if record["route"]["copies"].get(name):
        strides = verify_matmul._row_strides(list(shape))
    return list(shape), list(strides)


def bound_segments(record):
    """The segment words the kernel reads: pairs, or boundary words."""
    spec = record["segments"]
    if record["route"]["contiguous"]:
        return [word for pair in spec["pairs"] for word in pair]
    return list(spec["words"])


def binding(record):
    route, (entry,) = record["route"], record["entries"]
    info = segmented_mm_layout.signature(entry)
    a_shape, a_strides = bound(record, "a")
    b_shape, b_strides = bound(record, "b")
    m, n, k = route["M"], route["N"], route["K"]
    batches = len(record["segments"]["pairs"])
    params = {
        "M": m,
        "N": n,
        "K": k,
        "lda": route["lda"],
        "ldb": route["ldb"],
        "ldd": n,
        "tiles_n": -(-n // info["bn"]),
        "tiles_m": -(-m // info["bm"]),
        "batch_stride_a": 0,
        "batch_stride_b": 0,
        "batch_stride_d": m * n,
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": 0,
        "batch_ndim": 0,
    }
    words = bound_segments(record)
    counts = {
        "A": span(a_shape, a_strides),
        "B": span(b_shape, b_strides),
        "C": batches * m * n,
        "segments": len(words),
    }
    views = {"segments": (ctypes.c_uint32 * len(words))(*words)}
    execution = {
        "workgroupCount": [params["tiles_n"], params["tiles_m"], batches],
        "workgroupSize": list(segmented_mm_layout.workgroup_size(entry)),
    }
    return entry, counts, {"params": params}, views, execution


def test_cases_cover_every_segmented_route():
    assert {record["kind"] for record in RECORDS} == set(verify.KERNELS)
    assert verify._covered([{**record, "outputHash": ""} for record in RECORDS])
    for name in ("empty-output", "empty-k", "copied-segments-transposed"):
        assert not verify._covered(
            [
                {**record, "outputHash": ""}
                for record in RECORDS
                if record["name"] != name
            ]
        )
    for record in kind("segmented"):
        for entry in record["entries"]:
            assert segmented_mm_layout.signature(entry)["source"] in model.BY_NAME


def test_routes_mirror_segmented_mm_eval_gpu():
    copies = {name: record["route"]["copies"] for name, record in BY_NAME.items()}
    assert copies["copied-a-boundaries"] == {"a": True, "b": False, "segments": False}
    assert copies["copied-segments-transposed"] == {
        "a": False,
        "b": False,
        "segments": True,
    }
    contiguous = {
        record["name"]: record["route"]["contiguous"] for record in kind("segmented")
    }
    assert contiguous == {
        "pairs-unaligned": True,
        "pairs-aligned": True,
        "boundaries-transposed-a": False,
        "pairs-batched-transposed-b": True,
        # The copy is a dense (S, 2) array of pairs.
        "copied-segments-transposed": True,
        "copied-a-boundaries": False,
    }
    route = BY_NAME["pairs-batched-transposed-b"]["route"]
    assert (route["M"], route["N"], route["K"], route["ldb"]) == (20, 64, 33, 33)
    assert BY_NAME["pairs-batched-transposed-b"]["outShape"] == [2, 3, 20, 64]
    assert BY_NAME["empty-k"]["kind"] == "empty"
    assert BY_NAME["empty-output"]["kind"] == "none"
    with pytest.raises(ValueError, match="shapes differ"):
        segmented_mm_layout.route(
            "float32", [4, 5], [5, 1], [6, 7], [7, 1], ([1, 2], [2, 1], True)
        )


@pytest.mark.parametrize(
    "shape, strides, row_contiguous, expected",
    (
        ((3, 2), (2, 1), True, (True, False)),
        ((4, 2), (1, 1), False, (False, False)),
        # Batch axes must step over the boundaries of the axes after them.
        ((3, 3, 2), (3, 1, 1), False, (False, False)),
        ((1, 4, 2), (9, 1, 1), False, (False, False)),
        # MLX's own check passes (B, S, 2) with stride B only when B == S.
        ((2, 3, 2), (3, 1, 1), False, (True, True)),
        ((2, 3, 2), (2, 1, 1), False, (True, True)),
        ((3, 2), (1, 3), False, (True, True)),
        ((3, 2), (4, 1), False, (True, True)),
    ),
)
def test_segments_layout_follows_check_segments_layout(
    shape, strides, row_contiguous, expected
):
    assert (
        segmented_mm_layout.segments_layout(list(shape), list(strides), row_contiguous)
        == expected
    )


def test_signatures_accept_only_the_host_tile():
    entry = BY_NAME["pairs-unaligned"]["entries"][0]
    info = segmented_mm_layout.signature(entry)
    assert info["constants"] == {
        "segments_contiguous": True,
        "align_M": False,
        "align_N": False,
    }
    assert segmented_mm_layout.workgroup_size(entry) == (32, 2, 2)
    for bad in (
        entry.replace("_bm64_bn64_bk16_wm2_wn2", "_bm32_bn32_bk16_wm2_wn2"),
        entry.replace("float32_float32", "float32_float16"),
        entry + "_align_K_t",
        entry.replace("_segments_contiguous_t", ""),
        None,
    ):
        with pytest.raises(ValueError):
            segmented_mm_layout.signature(bad)
    with pytest.raises(ValueError, match="float32, float16 and bfloat16"):
        segmented_mm_layout.select("complex64", False, False, 4, 4, True)


@pytest.mark.parametrize("bk", (16, 32))
def test_k_loop_replays_the_kernel_loop(bk):
    for start in range(0, 3 * bk + 5, 3):
        for end in range(start, 4 * bk + 7, 5):
            tiles, tail, reach = segmented_mm_layout.k_loop(bk, start, end)
            assert (tiles, tail) == _k_tiles(bk, start, end)
            assert reach == end
    # An inverted segment reads nothing unless its uint32 count wraps a short.
    assert segmented_mm_layout.k_loop(16, 9, 1) == (0, -8, 9)
    assert segmented_mm_layout.k_loop(16, 65530, 0) == (0, 6, 65536)


@pytest.mark.parametrize("record", kind("segmented"), ids=lambda record: record["name"])
def test_plan_accepts_each_launch(record):
    entry, counts, parameters, views, execution = binding(record)
    metadata = segmented_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    assert metadata["kind"] == "segmented"
    assert metadata["segments"] == record["segments"]["pairs"]
    assert metadata["batches"] == len(record["segments"]["pairs"])
    assert metadata["kLoops"] == [
        list(segmented_mm_layout.k_loop(16, *pair)[:2])
        for pair in record["segments"]["pairs"]
    ]


@pytest.mark.parametrize(
    "fault, message",
    (
        ("segment-end", "exceeds the contraction"),
        ("segment-start", "exceeds the contraction"),
        ("segment-read", "segments read exceeds"),
        ("operand", "A exceeds its buffer"),
        ("params", "parameters differ"),
        ("grid", "grid"),
        ("routing", "routing"),
        ("leading", "parameters differ"),
        ("output", "output extent differs"),
    ),
)
def test_plan_rejects_launches_that_differ_from_mlx(fault, message):
    record = BY_NAME["pairs-unaligned"]
    entry, counts, parameters, views, execution = binding(record)
    params = parameters["params"]
    if fault == "segment-end":
        views["segments"][1] = params["K"] + 1
    elif fault == "segment-start":
        views["segments"][2] = params["K"] + 1
    elif fault == "segment-read":
        counts["segments"] -= 1
    elif fault == "operand":
        counts["A"] -= 1
    elif fault == "params":
        params["batch_stride_d"] += 1
    elif fault == "grid":
        execution["workgroupCount"][2] += 1
    elif fault == "routing":
        entry = entry.replace("_align_N_n", "_align_N_t")
    elif fault == "leading":
        params["lda"] = params["K"] - 1
    else:
        counts["C"] += 1
    with pytest.raises(ValueError, match=message):
        segmented_mm_layout.plan(
            entry, counts, parameters, views, counts["C"], execution
        )


def test_plan_replays_inverted_segments():
    record = BY_NAME["pairs-unaligned"]
    entry, counts, parameters, views, execution = binding(record)
    views["segments"][0], views["segments"][1] = 9, 1
    metadata = segmented_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    assert metadata["segments"][0] == [9, 1] and metadata["kLoops"][0] == [0, -8]


def test_single_rows_and_columns_take_any_leading_dimension():
    record = BY_NAME["pairs-unaligned"]
    entry, counts, parameters, views, execution = binding(record)
    params = parameters["params"]
    params.update(M=1, tiles_m=1, batch_stride_d=params["N"], lda=0)
    counts["C"] = 5 * params["N"]
    counts["A"] = params["K"]
    metadata = segmented_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    assert metadata["M"] == 1 and metadata["lda"] == 0
    params["lda"] = -1
    with pytest.raises(ValueError, match="parameters differ"):
        segmented_mm_layout.plan(
            entry, counts, parameters, views, counts["C"], execution
        )


def native_buffers(entry, counts, parameters, views):
    memory, buffers = {}, []
    for name in segmented_mm_layout.PARAMETERS:
        dtype = segmented_mm_layout.parameter_type(entry, name)
        if name in views:
            memory[name] = views[name]
            data, count = ctypes.addressof(views[name]), len(views[name])
        else:
            values = (
                list(parameters[name].values())
                if name == "params"
                else [0] * counts[name]
            )
            memory[name] = (CTYPES[dtype] * len(values))(*values)
            data, count = ctypes.addressof(memory[name]), len(values)
        buffers.append(
            runtime.Buffer(name.encode(), dtype.encode(), data, count, int(name == "C"))
        )
    return {buffer.name.decode(): buffer for buffer in buffers}, memory


def test_storage_reads_every_segmented_binding():
    record = BY_NAME["boundaries-transposed-a"]
    entry, counts, parameters, views, execution = binding(record)
    supplied, memory = native_buffers(entry, counts, parameters, views)
    read_counts, values, read_views = segmented_mm_layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    assert list(read_views["segments"]) == list(views["segments"])
    segmented_mm_layout.plan(
        entry, read_counts, values, read_views, counts["C"], execution
    )
    broken = dict(supplied)
    broken.pop("segments")
    with pytest.raises(ValueError, match="do not match"):
        segmented_mm_layout.read_storage(entry, broken)
    flipped = dict(supplied)
    flipped["A"] = runtime.Buffer(b"A", b"bfloat16", supplied["A"].data, counts["A"], 1)
    with pytest.raises(ValueError, match="direction"):
        segmented_mm_layout.read_storage(entry, flipped)
    retyped = dict(supplied)
    retyped["segments"] = runtime.Buffer(
        b"segments", b"int32", supplied["segments"].data, counts["segments"], 0
    )
    with pytest.raises(ValueError, match="storage"):
        segmented_mm_layout.read_storage(entry, retyped)
    overlapping = dict(supplied)
    overlapping["C"] = runtime.Buffer(
        b"C", b"bfloat16", supplied["A"].data, counts["C"], 1
    )
    with pytest.raises(ValueError, match="overlaps"):
        segmented_mm_layout.read_storage(entry, overlapping)
    del memory


def test_dispatch_storage_binds_every_reflected_parameter():
    record = BY_NAME["copied-segments-transposed"]
    entry, counts, parameters, views, execution = binding(record)
    supplied, memory = native_buffers(entry, counts, parameters, views)
    assert gather_mm_dispatch.family(entry) == "segmented"
    assert gather_mm_dispatch.label_of("segmented") == "SegmentedMM"
    metadata = segmented_mm_layout.plan(
        entry, counts, parameters, views, counts["C"], execution
    )
    storage = gather_mm_dispatch._storage(
        entry, "segmented", supplied, counts, parameters, metadata
    )
    assert set(storage) == set(segmented_mm_layout.PARAMETERS)
    assert storage["segments"][:6] == [0, 40, 9, 1, 20, 36]
    assert storage["segments"][-1] == gather_mm_dispatch.POISON["uint32"]
    assert storage["C"][-1] == gather_mm_dispatch.GUARDS["float16"]
    assert storage["params"] == parameters["params"]
    del memory


def test_packages_follow_the_kernel_gate():
    root = Path("/pinned")
    for record in kind("segmented"):
        (entry,) = record["entries"]
        info = segmented_mm_layout.signature(entry)
        for target in ("metal", "directx", "opengl"):
            config = segmented_mm_packages.segmented_mm_config(
                root, "out", target, entry
            )
            assert dict(config.specialization_constants) == info["constants"]
            assert config.entry_points == {
                segmented_mm_layout.SOURCE: (info["source"],)
            }
            assert [item.to_json() for item in config.immutable_storage] == [
                {
                    "source": segmented_mm_layout.SOURCE,
                    "entryPoint": info["source"],
                    "parameter": "segments",
                }
            ]
            assert bool(config.workgroup_access_assertions) == (target != "metal")
            assert bool(config.index_range_assertions) == (target == "opengl")
        # The kernel gate asserts the same index ranges.
        assert segmented_mm_layout.index_assertions(entry) == _index_assertions(
            model.BY_NAME[info["source"]]
        )
    assert set(segmented_mm_packages.verification_entries()) == {
        entry for record in RECORDS for entry in record["entries"]
    }


def test_runtime_routes_segmented_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, object()
    host.gather_mvs = host.gather_mms = host.gather_offsets = object()
    host.segmented_mms = None
    entries = [record["entries"][0] for record in kind("segmented")]
    for entry in entries:
        assert host._entry_available(entry.encode()) == 0
    host.segmented_mms = object()
    for entry in entries:
        assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(entries[0].replace("bm64", "bm32").encode()) == 0
    calls = []
    monkeypatch.setattr(
        gather_mm_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    for entry in entries:
        host.dispatch(entry, [], 0, 1, launch=None)
    assert calls == entries


def trace_events(record, target):
    copies = [{"entry": "g2_copyuint32uint32", "target": target}] * record["copies"]
    if record["kind"] == "empty":
        return copies + [{"entry": "s_copyfloat32float32", "target": target}]
    if record["kind"] == "none":
        return []
    route = record["route"]
    metadata = {
        "segments": record["segments"]["pairs"],
        "kLoops": [
            list(segmented_mm_layout.k_loop(16, *pair)[:2])
            for pair in record["segments"]["pairs"]
        ],
        "batches": len(record["segments"]["pairs"]),
        "constants": {"segments_contiguous": route["contiguous"]},
        **{key: route[key] for key in ("M", "N", "K", "lda", "ldb")},
    }
    return copies + [
        {"entry": record["entries"][0], "target": target, "gatherMetadata": metadata}
    ]


def traced(records, target):
    events, placed = [], []
    for record in records:
        produced = trace_events(record, target)
        placed.append(
            {
                **record,
                "outputHash": record["name"],
                "dispatchStart": len(events),
                "dispatchCount": len(produced),
            }
        )
        events += produced
    return events, placed


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_trace_validation_requires_the_routed_segments(target):
    events, records = traced(RECORDS, target)
    verify.validate_trace(events, records, target)
    other = "opengl" if target != "opengl" else "metal"
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(events, records, other)
    for name, change in (
        (
            "pairs-unaligned",
            lambda event: event["gatherMetadata"]["segments"].reverse(),
        ),
        (
            "boundaries-transposed-a",
            lambda event: event["gatherMetadata"].update(lda=1),
        ),
        (
            "copied-a-boundaries",
            lambda event: event["gatherMetadata"]["kLoops"][0].reverse(),
        ),
    ):
        changed = copy.deepcopy(events)
        placed = next(r for r in records if r["name"] == name)
        change(changed[placed["dispatchStart"] + placed["dispatchCount"] - 1])
        with pytest.raises(RuntimeError, match="differs"):
            verify.validate_trace(changed, records, target)
    changed = copy.deepcopy(events)
    placed = next(r for r in records if r["name"] == "copied-segments-transposed")
    changed[placed["dispatchStart"]]["entry"] = "arangeuint32"
    with pytest.raises(RuntimeError, match="copies differ"):
        verify.validate_trace(changed, records, target)
    placed = next(r for r in records if r["name"] == "empty-k")
    changed = copy.deepcopy(events)
    del changed[placed["dispatchStart"]]
    with pytest.raises(RuntimeError):
        verify.validate_trace(changed, records, target)


def test_reference_follows_cpu_segmented_mm():
    import numpy as np

    record = BY_NAME["copied-segments-transposed"]
    data = verify.case_data(np, record, 4)
    a = verify_matmul._view(np, data["a"], record["a"]["view"])
    b = verify_matmul._view(np, data["b"], record["b"]["view"])
    expected = data["expected"].astype(np.float64)
    assert expected.shape == (3, 70, 66)
    # The inverted second segment writes zeros, as the CPU's k_end <= k_start.
    assert not expected[1].any()
    np.testing.assert_array_equal(
        expected[2],
        (a[:, 20:36] @ b[20:36]).astype(np.float16).astype(np.float64),
    )
    assert verify.case_data(np, BY_NAME["empty-output"], 7)["expected"].shape == (
        0,
        5,
        7,
    )


def test_backend_ports_segmented_mm_eval_gpu():
    port = (HOST / "segmented_mm.inc").read_text(encoding="utf-8")
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    tail = backend[
        backend.rindex('#include "mlx/backend/no_gpu/crosstl_compiled.inc"') :
    ]
    assert tail.index("crosstl_gather_mm.inc") < tail.index("crosstl_segmented_mm.inc")
    # Only the BlockMaskedMM and general Scatter ports follow it.
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_segmented_mm.inc"\n'
        "// So does the port of BlockMaskedMM's masked products.\n"
        '#include "mlx/backend/no_gpu/crosstl_block_masked_mm.inc"\n'
        "// And the port of the general Scatter, which tiles and orders its launches.\n"
        '#include "mlx/backend/no_gpu/crosstl_scatter.inc"'
    )
    start = port.index("void SegmentedMM::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        "out.set_data(allocator::malloc(out.nbytes()));",
        "if (out.size() == 0) {",
        "if (K == 0) {",
        "require_empty_segments(segments);",
        "dispatch_segmented_mm(a, b, segments, out, M, N, K);",
    ]
    assert [body.index(text) for text in order] == sorted(
        body.index(text) for text in order
    )
    for name in segmented_mm_layout.PARAMETERS:
        assert f'{{"{name}", ' in port, name
    for text in (
        "auto [transpose_a, lda, a] = gemv_operand(a_, false);",
        "auto [transpose_b, ldb, b] = gemv_operand(b_, false);",
        '"_segments_contiguous_" + flag(segments_contiguous)',
        "M, N, K, lda, ldb, N, tiles_n, tiles_m, 0, 0, int64_t(M) * N, 0, 0, 0};",
        "{uint32_t(tiles_n), uint32_t(tiles_m), uint32_t(batch_size_out)},",
        "{32, uint32_t(wn), uint32_t(wm)}};",
    ):
        assert text in port, text


def test_preparation_replaces_the_segmented_mm_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    sources = prepare.adapted_sources("", "", "")
    assert "NO_GPU(SegmentedMM)" in replaced
    assert (
        sources["crosstl_segmented_mm.inc"] == (HOST / "segmented_mm.inc").read_bytes()
    )


def test_ci_runs_the_segmented_mm_route_on_every_quantized_host_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    steps = workflow["jobs"]["quantized-host"]["steps"]
    names = [item.get("name") for item in steps]
    execute = steps[names.index("Execute SegmentedMM products through MLX")]
    assert (
        names.index("Execute GatherMM products through MLX")
        < names.index("Execute SegmentedMM products through MLX")
        < names.index("Retain quantized execution evidence")
    )
    assert "if" not in execute and "continue-on-error" not in execute
    assert "set -euo pipefail" in execute["run"]
    assert "portable_host.verify_segmented_mm" in execute["run"]
    assert "--packages .mlx-portable-quantized/base/packages" in execute["run"]
    assert "tee .mlx-portable-quantized/segmented-mm.log" in execute["run"]
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    validate = steps[names.index("Validate quantized host contracts")]["run"]
    assert "test_portable_segmented_mm.py" in validate
