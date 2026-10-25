"""Convolution host routing, unfold and depthwise launch contracts and their cases."""

import copy
import ctypes
import math
import struct
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    conv_dispatch,
    conv_layout,
    conv_packages,
    gemm_layout,
    lazy_package_evidence,
    prepare,
    runtime,
    upstream_matrix_packages,
)
from demos.integrations.mlx.portable_host import verify_conv as verify
from demos.integrations.mlx.tests.kernels import test_current_conv as gate

layout = conv_layout
HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
BY_NAME = {record["name"]: record for record in RECORDS}
ROUTES = {
    "explicit-1d-strided": "explicit_gemm_conv_ND_gpu",
    "explicit-1d-transpose-tiled": "explicit_gemm_conv_ND_gpu",
    "explicit-1d-splitk": "explicit_gemm_conv_ND_gpu",
    "explicit-2d-unaligned": "explicit_gemm_conv_ND_gpu",
    "explicit-2d-dilated-copied": "explicit_gemm_conv_ND_gpu",
    "explicit-3d-transpose": "explicit_gemm_conv_ND_gpu",
    "group-1d-unaligned": "explicit_gemm_conv_group_ND_gpu",
    "group-1d-transpose-tiled": "explicit_gemm_conv_group_ND_gpu",
    "group-2d-transpose": "explicit_gemm_conv_group_ND_gpu",
    "group-2d-unaligned": "explicit_gemm_conv_group_ND_gpu",
    "depthwise-1d": "depthwise_conv_1D_gpu",
    "depthwise-1d-pointwise-copied": "depthwise_conv_1D_gpu",
    "depthwise-1d-long": "depthwise_conv_1D_gpu",
    "refused-implicit-2d": "implicit_gemm_conv_2D_gpu",
    "refused-depthwise-2d": "depthwise_conv_2D_gpu",
    "empty-batch": "none",
}


def params_bytes(params):
    """``MLXConvParams<N>`` in Metal's natural C layout, as the gate packs it."""
    return gate._params_bytes(params)


def launches(record):
    """``(entry, counts, parameters, threads, execution)`` per conv.inc launch."""
    p = record["params"]
    if record["route"] == "depthwise_conv_1D_gpu":
        (entry,) = record["convEntries"]
        batches, length, channels = record["inShape"]
        steps, taps = record["outShape"][1], record["wtShape"][1]
        counts = {
            "in": batches * length * channels,
            "w": channels * taps,
            "out": batches * steps * channels,
        }
        parameters = {"strides": [length * channels, channels, 1], "kernel_size": taps}
        execution = {
            "workgroupCount": [channels, steps, batches],
            "workgroupSize": [1, 1, 1],
        }
        return [(entry, counts, parameters, counts["out"], execution)]
    if not record["convEntries"]:
        return []
    (entry,) = record["convEntries"]
    points = math.prod(p["wS"])
    total = p["N"] * math.prod(p["oS"])
    rows = verify._unfold_rows(
        math.prod(record["wtShape"]) // p["O"] * p["groups"], total, record["tileRows"]
    )
    result = []
    for offset in range(0, total, rows):
        tile = min(rows, total - offset)
        threads = tile * p["C"] * points
        counts = {"in": math.prod(record["inShape"]), "out": threads}
        parameters = {"params": copy.deepcopy(p), "row_offset": offset}
        execution = {
            "workgroupCount": [p["C"], points, tile],
            "workgroupSize": [1, 1, 1],
        }
        result.append((entry, counts, parameters, threads, execution))
    return result


def test_signature_names_every_ported_entry():
    for dtype in layout.DTYPES:
        for spatial in layout.SPATIAL:
            for transpose in (False, True):
                entry = layout.unfold_entry(transpose, dtype, spatial)
                info = layout.signature(entry)
                assert info == {
                    "kind": "unfold",
                    "transpose": transpose,
                    "dtype": dtype,
                    "spatial": spatial,
                }
                assert layout.host_parameters(entry) == (
                    "in",
                    "out",
                    "params",
                    "row_offset",
                )
        info = layout.signature(layout.dw1_entry(dtype))
        assert info["kind"] == "dw1" and info["dtype"] == dtype
    assert set(gate.UNFOLD_ENTRIES) == {
        layout.unfold_entry(t, d, s)
        for t in (False, True)
        for d in layout.DTYPES
        for s in layout.SPATIAL
    }
    # The int64 kernel serves more than 2^31 - 1 input elements, which no
    # host launch binds.
    for entry in (
        "depthwise_conv_1d_float32_large",
        "naive_unfold_nd_float32_4",
        "naive_unfold_nd_int32_1",
        "depthwise_conv_2d_float32",
        None,
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            layout.signature(entry)


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_cases_route_as_conv_cpp(record):
    assert record["route"] == ROUTES[record["name"]]
    if record["route"] in verify.UNPORTED | {"none"}:
        assert record["launches"] == [] and record["convEntries"] == []


def test_routes_follow_conv_cpp_branches():
    base = BY_NAME["explicit-2d-unaligned"]["params"]

    def routed(**changes):
        return layout.route({**copy.deepcopy(base), **changes})

    assert routed() == "explicit_gemm_conv_ND_gpu"
    assert routed(C=16, O=16) == "implicit_gemm_conv_2D_gpu"
    assert routed(C=4) == "implicit_gemm_conv_2D_gpu"
    assert routed(oS=[16, 16]) == "pad_in_channels_conv_2D_gpu"
    assert routed(oS=[16, 16], wS=[1, 1]) == "implicit_gemm_conv_2D_general_gpu"
    assert routed(idil=[2, 1], C=16, O=16) == "implicit_gemm_conv_2D_general_gpu"
    assert routed(idil=[2, 1]) == "explicit_gemm_conv_ND_gpu"
    assert routed(C=32, O=256, iS=[64, 64], oS=[64, 64]) == "winograd_conv_2D_gpu"
    assert routed(C=32, O=256, iS=[64, 64], oS=[64, 64], flip=True) == (
        "implicit_gemm_conv_2D_gpu"
    )
    one = BY_NAME["explicit-1d-strided"]["params"]
    assert layout.route({**one, "C": 4}) == "implicit_gemm_conv_2D_gpu"
    assert layout.route({**one, "C": 4, "idil": [2]}) == "explicit_gemm_conv_ND_gpu"
    depthwise = BY_NAME["depthwise-1d"]["params"]
    assert layout.route(depthwise) == "depthwise_conv_1D_gpu"
    assert layout.route({**depthwise, "flip": True}) == "implicit_gemm_conv_2D_gpu"
    three = BY_NAME["explicit-3d-transpose"]["params"]
    assert layout.route({**three, "idil": [1, 1, 1]}) == "pad_and_slice_conv_3D_gpu"
    assert (
        layout.route({**three, "idil": [1, 1, 1], "C": 16, "O": 16, "pad": [0, 2, 2]})
        == "small_kd_conv_3D_gpu"
    )
    assert (
        layout.route({**three, "idil": [1, 1, 1], "C": 16, "O": 16})
        == "implicit_gemm_conv_3D_gpu"
    )
    assert (
        layout.route(
            {
                **three,
                "idil": [2, 2, 2],
                "wS": [2, 2, 2],
                "pad": [1, 1, 1],
                "oS": [2 * value for value in three["iS"]],
            }
        )
        == "stride_two_conv_transpose_3D_gpu"
    )


def test_cases_cover_every_ported_route():
    placed = [
        {**record, "outputHash": "x"}
        for record in RECORDS
        if record["route"] not in verify.UNPORTED | {"none"}
    ] + [record for record in RECORDS if record["route"] in verify.UNPORTED | {"none"}]
    assert verify._covered(placed)
    assert not verify._covered(
        [record for record in placed if record["route"] != "depthwise_conv_1D_gpu"]
    )
    assert BY_NAME["explicit-1d-splitk"]["productEntries"][0].startswith(
        "steel_gemm_splitk_"
    )
    assert BY_NAME["explicit-1d-transpose-tiled"]["tiles"] == 9
    assert BY_NAME["group-1d-transpose-tiled"]["tiles"] == 10
    assert BY_NAME["group-2d-unaligned"]["copies"] == 1
    assert BY_NAME["explicit-2d-dilated-copied"]["copies"] == 1


@pytest.mark.parametrize(
    "record",
    [record for record in RECORDS if record["convEntries"]],
    ids=lambda record: record["name"],
)
def test_plan_accepts_each_launch(record):
    planned = launches(record)
    assert planned and [entry for entry, *_ in planned] == [
        entry for entry in record["launches"] if entry in record["convEntries"]
    ]
    for entry, counts, parameters, threads, execution in planned:
        metadata = layout.plan(entry, counts, parameters, threads, execution)
        assert metadata["route"] == record["route"]


def _unfold(name="explicit-2d-unaligned"):
    return launches(BY_NAME[name])[0]


@pytest.mark.parametrize(
    "fault, message",
    (
        (lambda c, p, e: p["params"].update(groups=2), "invalid|routing"),
        (lambda c, p, e: p["params"].update(C=16, O=16), "strides|routing"),
        (lambda c, p, e: p["params"]["in_strides"].__setitem__(0, 1), "strides"),
        (lambda c, p, e: p["params"]["oS"].__setitem__(0, 0), "invalid"),
        (lambda c, p, e: p["params"]["str"].__setitem__(0, 2**30), "signed range"),
        (lambda c, p, e: p.update(row_offset=1), "tile"),
        (lambda c, p, e: c.update(**{"in": c["in"] - 1}), "input exceeds"),
        (lambda c, p, e: e.update(workgroupSize=[32, 8, 1]), "thread grid"),
        (lambda c, p, e: e["workgroupCount"].__setitem__(1, 1), "thread grid"),
        (lambda c, p, e: e.update(threadGridSize=[1, 1, 1]), "geometry"),
    ),
)
def test_unfold_plan_rejects_launches_that_differ_from_conv_cpp(fault, message):
    entry, counts, parameters, threads, execution = copy.deepcopy(_unfold())
    fault(counts, parameters, execution)
    with pytest.raises(ValueError, match=message):
        layout.plan(entry, counts, parameters, threads, execution)


def test_unfold_plan_requires_the_entry_conv_cpp_names():
    entry, counts, parameters, threads, execution = _unfold("group-2d-unaligned")
    with pytest.raises(ValueError, match="invalid"):
        layout.plan(
            entry.replace("_transpose", ""), counts, parameters, threads, execution
        )
    entry, counts, parameters, threads, execution = _unfold("explicit-1d-strided")
    params = {**parameters["params"], "C": 4}
    params["in_strides"] = layout.contiguous([params["N"], *params["iS"], 4])
    params["wt_strides"] = layout.contiguous([params["O"], *params["wS"], 4])
    # conv.cpp sends four input channels through the 2-D implicit GEMM.
    with pytest.raises(ValueError, match="routing"):
        layout.plan(entry, counts, {**parameters, "params": params}, threads, execution)


@pytest.mark.parametrize(
    "fault",
    (
        lambda c, p, e: p["strides"].__setitem__(2, 2),
        lambda c, p, e: p.update(kernel_size=p["kernel_size"] + 1),
        lambda c, p, e: c.update(w=c["w"] - 1),
        lambda c, p, e: c.update(out=c["out"] + 1),
        lambda c, p, e: e["workgroupCount"].__setitem__(2, 1),
        lambda c, p, e: e.update(workgroupSize=[16, 16, 4]),
    ),
)
def test_depthwise_plan_rejects_launches_that_differ_from_conv_cpp(fault):
    entry, counts, parameters, threads, execution = copy.deepcopy(
        launches(BY_NAME["depthwise-1d"])[0]
    )
    fault(counts, parameters, execution)
    with pytest.raises(ValueError):
        layout.plan(entry, counts, parameters, threads, execution)


CTYPES = {
    "float32": ctypes.c_float,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "uint8": ctypes.c_uint8,
}


def native_buffers(entry, counts, parameters):
    """ctypes buffers named and typed as conv.inc binds them."""
    memory, buffers = {}, {}
    for name in layout.host_parameters(entry):
        dtype = layout.parameter_type(entry, name)
        if name == "params":
            data = params_bytes(parameters["params"])
            memory[name] = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
            count = len(data)
        elif name in counts:
            count = counts[name]
            memory[name] = (CTYPES[dtype] * count)()
        else:
            value = parameters[name]
            values = value if isinstance(value, list) else [value]
            memory[name] = (ctypes.c_int32 * len(values))(*values)
            count = len(values)
        buffers[name] = runtime.Buffer(
            name.encode(),
            dtype.encode(),
            ctypes.addressof(memory[name]),
            count,
            int(name == layout.OUTPUT),
        )
    return buffers, memory


@pytest.mark.parametrize(
    "name", ("explicit-3d-transpose", "group-1d-unaligned", "depthwise-1d-long")
)
def test_storage_reads_every_bound_parameter(name):
    entry, counts, parameters, threads, execution = launches(BY_NAME[name])[-1]
    supplied, memory = native_buffers(entry, counts, parameters)
    read_counts, values = layout.read_storage(entry, supplied)
    assert read_counts == counts and values == parameters
    layout.plan(entry, read_counts, values, threads, execution)
    broken = dict(supplied)
    broken.pop("in")
    with pytest.raises(ValueError, match="do not match"):
        layout.read_storage(entry, broken)
    retyped = dict(supplied)
    retyped["in"] = runtime.Buffer(b"in", b"int32", supplied["in"].data, 1, 0)
    with pytest.raises(ValueError, match="storage"):
        layout.read_storage(entry, retyped)
    overlapping = dict(supplied)
    overlapping["out"] = runtime.Buffer(
        b"out", supplied["out"].dtype, supplied["in"].data, counts["out"], 1
    )
    with pytest.raises(ValueError, match="overlaps"):
        layout.read_storage(entry, overlapping)
    del memory


def test_params_decode_mlx_conv_params_for_every_rank():
    for name in (
        "explicit-1d-strided",
        "explicit-2d-unaligned",
        "explicit-3d-transpose",
    ):
        params = BY_NAME[name]["params"]
        data = params_bytes(params)
        spatial = len(params["iS"])
        assert len(data) == layout.PARAMS_SIZES[spatial] == gate.PARAMS_SIZES[spatial]
        assert layout.decode_params(data, spatial) == params
        with pytest.raises(ValueError, match="MLXConvParams"):
            layout.decode_params(data[:-8], spatial)
    corrupted = bytearray(params_bytes(BY_NAME["explicit-1d-strided"]["params"]))
    corrupted[116] = 2
    with pytest.raises(ValueError, match="boolean"):
        layout.decode_params(bytes(corrupted), 1)


class _Host:
    def __init__(self, target):
        self.target = target


def _descriptor(entry, target):
    """A reflected descriptor in the shape the translated packages carry."""
    info = layout.signature(entry)
    bindings = []
    for name in layout.host_parameters(entry):
        if name == "params":
            scalar = {
                "elementType": "record",
                "elementStrideBytes": layout.PARAMS_SIZES[info["spatial"]],
            }
        elif name in layout.ARRAYS[info["kind"]]:
            scalar = {"elementType": "float32", "elementStrideBytes": 4}
        else:
            scalar = {"elementType": "int32", "elementStrideBytes": 4}
        bindings.append(
            {
                "name": f"{name}_binding",
                "kind": "buffer",
                "access": "read_write" if name == layout.OUTPUT else "read",
                "provenance": {"sourceResource": {"parameter": name}},
                "scalarLayout": scalar,
            }
        )
    if target == "directx" and info["kind"] == "dw1":
        bindings.append(
            {
                "name": "CrossGLDispatchInfo",
                "provenance": {"kind": "generated-execution-input"},
                "scalarLayout": {"elementType": "uint32"},
            }
        )
    return {"bindings": bindings}


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_dispatch_binds_every_reflected_parameter(monkeypatch, target):
    entry, counts, parameters, threads, execution = launches(BY_NAME["depthwise-1d"])[0]
    supplied, memory = native_buffers(entry, counts, parameters)
    storage = conv_dispatch._storage(entry, supplied, counts, parameters)
    assert len(storage["out"]) == counts["out"] + conv_dispatch.GUARD_COUNT
    monkeypatch.setattr(
        conv_dispatch,
        "encode_binding",
        lambda dtype, words, *_: {"dtype": dtype, "values": list(words)},
    )
    descriptor = _descriptor(entry, target)
    inputs, observed, output, bound = conv_dispatch._inputs(
        _Host(target), entry, descriptor, storage, parameters
    )
    assert bound == set(layout.host_parameters(entry))
    assert output == "out_binding" and "out_binding" not in observed
    assert inputs["strides_binding"]["values"] == parameters["strides"]
    assert "CrossGLDispatchInfo" not in inputs
    missing = {"bindings": descriptor["bindings"][1:]}
    with pytest.raises(ValueError, match="incomplete"):
        conv_dispatch._inputs(_Host(target), entry, missing, storage, parameters)
    foreign = copy.deepcopy(descriptor)
    foreign["bindings"][0]["provenance"]["sourceResource"]["parameter"] = "wt"
    with pytest.raises(ValueError, match="differs from its source"):
        conv_dispatch._inputs(_Host(target), entry, foreign, storage, parameters)
    del memory


def test_dispatch_packs_the_parameter_record():
    entry, counts, parameters, threads, execution = _unfold("explicit-3d-transpose")
    binding = _descriptor(entry, "metal")["bindings"][2]
    with pytest.raises(ValueError, match="record differs"):
        conv_dispatch._params(
            {
                **binding,
                "scalarLayout": {"elementType": "record", "elementStrideBytes": 8},
            },
            parameters["params"],
            3,
        )


def test_packages_follow_the_conv_gate(tmp_path):
    root = Path("/pinned")
    entries = {entry for record in RECORDS for entry in record["convEntries"]}
    for entry in sorted(entries):
        for target in ("metal", "directx", "opengl"):
            config = conv_packages.ConvPackageCache(root, tmp_path, target)
            built = config.project_config("out", entry)
            assert built.entry_points == {layout.SOURCE: (entry,)}
            (rules,) = built.entry_workgroup_size_rules.values()
            assert list(built.entry_workgroup_size_rules) == [layout.SOURCE]
            assert {name: tuple(map(int, size)) for name, size in rules.items()} == {
                entry: layout.WORKGROUP
            }
            kind = gate._kind(entry)
            assert [
                item.to_json()["expression"] for item in built.index_range_assertions
            ] == (
                list(gate.OPENGL_INDEX_RANGE_ASSERTIONS[kind])
                if target == "opengl"
                else []
            )
            metal = built.source_options["metal"]
            assert metal["max_template_specializations"] == 256
            assert metal["max_template_materialization_work"] == 16384
            assert metal["preserve_resource_origins"] is True
            profile = (
                metal.get("target_options", {})
                .get(target, {})
                .get("binary32_additive_profile")
            )
            assert profile == (
                gate.ADDITIVE_PROFILE
                if gate._profiled_additions(target, entry)
                else None
            )
    assert set(conv_packages.verification_entries()) == entries
    assert set(conv_packages.product_entries()) == {
        entry for record in RECORDS for entry in record["productEntries"]
    }


def _grouped_product(name="group-2d-unaligned"):
    record = BY_NAME[name]
    p = record["params"]
    groups = p["groups"]
    m = p["N"] * math.prod(p["oS"])
    n, k = p["O"] // groups, math.prod(record["wtShape"]) // p["O"]
    (entry,) = record["productEntries"]
    bm, bn, bk, wm, wn = gemm_layout.HOST_TILE
    params = {
        "M": m,
        "N": n,
        "K": k,
        "tiles_n": -(-n // bn),
        "tiles_m": -(-m // bm),
        "swizzle_log": 0,
        "gemm_k_iterations_aligned": k // bk,
        "batch_ndim": 1,
        **gemm_layout.grouped_params(m, n, k, groups),
    }
    counts = {"A": m * k * groups, "B": p["O"] * k, "D": m * p["O"]}
    parameters = {"params": params, "batch_shape": [1], "batch_strides": [0, 0]}
    execution = {
        "workgroupCount": [params["tiles_n"], params["tiles_m"], groups],
        "workgroupSize": [32, wn, wm],
    }
    return entry, counts, parameters, groups * m * n, execution


def test_gemm_plan_accepts_conv_grouped_products():
    for name in ("group-1d-unaligned", "group-2d-transpose", "group-2d-unaligned"):
        entry, counts, parameters, threads, execution = _grouped_product(name)
        metadata = gemm_layout.plan(entry, counts, parameters, threads, execution)
        assert metadata["groups"] == BY_NAME[name]["params"]["groups"]
        assert metadata["batches"] == metadata["groups"]


@pytest.mark.parametrize(
    "fault, message",
    (
        (lambda c, p, e: p["params"].update(ldd=p["params"]["N"]), "differ"),
        (lambda c, p, e: p["params"].update(batch_stride_d=0), "differ"),
        (lambda c, p, e: p["params"].update(lda=p["params"]["K"]), "differ"),
        (lambda c, p, e: p.update(batch_strides=[0, 1]), "conv.cpp"),
        (lambda c, p, e: c.update(A=c["A"] - 1), "A batch"),
        (lambda c, p, e: c.update(B=c["B"] - 1), "B batch"),
        (lambda c, p, e: e["workgroupCount"].__setitem__(2, 2), "differ|extent"),
    ),
)
def test_gemm_plan_rejects_grouped_products_that_differ(fault, message):
    entry, counts, parameters, threads, execution = copy.deepcopy(_grouped_product())
    fault(counts, parameters, execution)
    with pytest.raises(ValueError, match=message):
        gemm_layout.plan(entry, counts, parameters, threads, execution)


def test_gemm_plan_keeps_matmul_batches_ungrouped():
    """A one-batch product launched over more Z groups is no grouped product."""
    entry, counts, parameters, threads, execution = copy.deepcopy(_grouped_product())
    parameters["params"].update(
        ldd=parameters["params"]["N"],
        batch_stride_a=0,
        batch_stride_b=0,
        batch_stride_d=parameters["params"]["M"] * parameters["params"]["N"],
        lda=parameters["params"]["K"],
    )
    with pytest.raises(ValueError):
        gemm_layout.plan(entry, counts, parameters, threads, execution)


def test_runtime_routes_conv_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, object()
    host.gather_mvs = host.gather_mms = host.gather_offsets = object()
    host.segmented_mms = host.gemvs = object()
    host.masked_mvs = host.masked_mms = object()
    host.convs = None
    entries = sorted({entry for record in RECORDS for entry in record["convEntries"]})
    for entry in entries:
        assert host._entry_available(entry.encode()) == 0
    host.convs = object()
    for entry in entries:
        assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"depthwise_conv_1d_float32_large") == 0
    calls = []
    monkeypatch.setattr(conv_dispatch, "dispatch", lambda *args: calls.append(args[1]))
    for entry in entries:
        host.dispatch(entry, [], 0, 1, launch=None)
    assert calls == entries
    for entry in entries:
        assert lazy_package_evidence.family(entry) == "conv"
    assert "conv" in upstream_matrix_packages.ORDER


def trace_events(record, target):
    events = [{"entry": "g3_copyfloat32float32", "target": target}] * (record["copies"])
    planned = iter(launches(record))
    for entry in record["launches"]:
        if entry in record["convEntries"]:
            _, counts, parameters, threads, execution = next(planned)
            metadata = layout.plan(entry, counts, parameters, threads, execution)
            events.append({"entry": entry, "target": target, "convMetadata": metadata})
        else:
            gemm = (
                {"groups": record["params"]["groups"]}
                if record["params"]["groups"] > 1
                else {}
            )
            events.append({"entry": entry, "target": target, "gemmMetadata": gemm})
    return events


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
def test_trace_validation_requires_the_routed_launches(target):
    events, records = traced(RECORDS, target)
    verify.validate_trace(events, records, target)
    other = "opengl" if target != "opengl" else "metal"
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(events, records, other)
    placed = {record["name"]: record for record in records}
    changed = copy.deepcopy(events)
    record = placed["explicit-1d-splitk"]
    changed[record["dispatchStart"] + 1]["entry"] = "steel_gemm_fused_x"
    with pytest.raises(RuntimeError, match="differs from its route"):
        verify.validate_trace(changed, records, target)
    changed = copy.deepcopy(events)
    record = placed["group-2d-transpose"]
    changed[record["dispatchStart"] + 2]["gemmMetadata"]["groups"] = 2
    with pytest.raises(RuntimeError, match="groups differ"):
        verify.validate_trace(changed, records, target)
    changed = copy.deepcopy(events)
    record = placed["depthwise-1d-long"]
    changed[record["dispatchStart"]]["convMetadata"]["route"] = "x"
    with pytest.raises(RuntimeError, match="launch differs"):
        verify.validate_trace(changed, records, target)
    changed = copy.deepcopy(events)
    record = placed["explicit-2d-dilated-copied"]
    del changed[record["dispatchStart"]]
    with pytest.raises(RuntimeError):
        verify.validate_trace(changed, records, target)


def _direct(np, x, w, record):
    """conv_general's definition, one output element at a time."""
    o, p = record["options"], record["params"]
    spatial = len(p["iS"])
    c_group, o_group = p["C"] // o["groups"], p["O"] // o["groups"]
    out = np.zeros(record["outShape"])
    for index in np.ndindex(*record["outShape"]):
        n, position, channel = index[0], index[1:-1], index[-1]
        group = channel // o_group
        total = 0.0
        for tap in np.ndindex(*p["wS"]):
            source = [
                p["wS"][i] - 1 - tap[i] if o["flip"] else tap[i] for i in range(spatial)
            ]
            dilated = [
                position[i] * o["stride"][i] - o["padLo"][i] + source[i] * o["kdil"][i]
                for i in range(spatial)
            ]
            if any(
                value < 0 or value % o["idil"][i] or value // o["idil"][i] >= p["iS"][i]
                for i, value in enumerate(dilated)
            ):
                continue
            point = tuple(value // o["idil"][i] for i, value in enumerate(dilated))
            for c in range(c_group):
                total += x[(n, *point, group * c_group + c)] * w[(channel, *tap, c)]
        out[index] = total
    return out


@pytest.mark.parametrize(
    "name",
    (
        "explicit-1d-strided",
        "explicit-1d-transpose-tiled",
        "explicit-2d-dilated-copied",
        "group-1d-transpose-tiled",
        "group-2d-transpose",
        "depthwise-1d",
    ),
)
def test_reference_follows_the_convolution_definition(name):
    import numpy as np

    record = BY_NAME[name]
    data = verify.case_data(np, record, 3)
    expected = _direct(np, data["x"], data["w"], record)
    assert np.array_equal(verify.reference(np, data["x"], data["w"], record), expected)
    assert np.array_equal(
        data["expected"], verify.rounded(expected + 0.0, record["dtype"])
    )


def test_backend_ports_convolution_eval_gpu():
    port = (HOST / "conv.inc").read_text(encoding="utf-8")
    start = port.index("void Convolution::eval_gpu(")
    body = port[start : port.index("\n}\n", start)]
    order = [
        "require_runtime();",
        "if (out.size() == 0) {",
        "if (out.ndim() == 5) {",
        "conv_3d(",
        "conv_2d(",
        "conv_1d(",
    ]
    assert [body.index(text) for text in order] == sorted(
        body.index(text) for text in order
    )
    for route in verify.UNPORTED:
        assert f'conv_unported("{route}");' in port, route
    for name in (*layout.PARAMETERS["unfold"], *layout.PARAMETERS["dw1"]):
        assert f'{{"{name}", ' in port, name
    for text in (
        '"naive_unfold_nd_", in, in_tile, conv_params, row_offset, tile_rows);',
        '"naive_unfold_transpose_nd_", in, in_tile, conv_params, row_offset, tile_rows);',
        "{conv_groups(conv_params.C), conv_groups(kernel_points), conv_groups(tile_rows)},",
        "{conv_groups(D), conv_groups(Tout), conv_groups(B)}, {1, 1, 1}};",
        'env::get_var("MLX_CONV_UNFOLD_TILE_ROWS", 0)',
        "Shape{1}, Strides{0, 0}, int64_t(implicit_K), int64_t(implicit_N) * implicit_K,",
        "implicit_K, false, true, Shape{1}, Strides{0}, Strides{0});",
    ):
        assert text in port, text
    one = port[port.index("void conv_1d(") :]
    one = one[: one.index("\n}\n")]
    assert one.index("dispatch_depthwise_conv_1d(") < one.index("conv_dispatch_2d(")
    assert one.index("conv_dispatch_2d(") < one.index("conv_explicit(in, wt, out")
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.rstrip().endswith('#include "mlx/backend/no_gpu/crosstl_conv.inc"')
    regular = backend[backend.index("void dispatch_steel_gemm_regular(") :]
    regular = regular[: regular.index("\n}\n")]
    assert "set_data" not in regular
    assert "ldd,\n      int64_t(tiles_n)," in regular


def test_preparation_replaces_the_convolution_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    sources = prepare.adapted_sources("", "", "")
    assert "NO_GPU(Convolution)" in replaced
    assert sources["crosstl_conv.inc"] == (HOST / "conv.inc").read_bytes()


def test_params_layout_matches_the_kernel_gate():
    for spatial in layout.SPATIAL:
        head = struct.Struct(f"<{3 + 7 * spatial}i").size
        assert layout.PARAMS_SIZES[spatial] == gate.PARAMS_SIZES[spatial]
        assert (
            head + (-head % 8) + 24 * (spatial + 2) + 5 <= layout.PARAMS_SIZES[spatial]
        )


def test_ci_runs_the_convolution_routes_on_every_platform():
    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["conv-host"]
    assert {item["target"] for item in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    assert job["if"] == "needs.ci-plan.outputs.heavy == 'true'"
    steps = job["steps"]
    names = [item.get("name") for item in steps]
    step = "Execute convolutions through MLX"
    execute = steps[names.index(step)]
    assert (
        names.index("Build adapted upstream MLX")
        < names.index(step)
        < names.index("Retain convolution execution evidence")
    )
    assert "if" not in execute and "continue-on-error" not in execute
    run = execute["run"]
    assert "set -euo pipefail" in run
    assert run.count("portable_host.conv_packages") == 2
    assert "portable_host.conv_packages products" in run
    assert "portable_host.verify_conv" in run
    assert run.count("--packages .mlx-portable-conv/base/packages") == 3
    assert "tee .mlx-portable-conv/conv.log" in run
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    validate = steps[names.index("Validate convolution host contracts")]
    assert "test_portable_conv.py" in validate["run"]
    assert "conv-host" in workflow["jobs"]["result"]["needs"]
