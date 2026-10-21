"""Steel GEMM host routing, launch contracts and guarded commits."""

import ctypes
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gemm_dispatch as dispatch
from demos.integrations.mlx.portable_host import gemm_layout as layout
from demos.integrations.mlx.portable_host import gemm_packages, runtime
from demos.integrations.mlx.portable_host import verify_matmul as verify
from demos.integrations.mlx.tests.host.test_portable_matmul import (
    CTYPES,
    bound_operands,
    physical_layout,
    span,
)
from demos.integrations.mlx.tests.kernels import gemm_fused_model as model
from demos.integrations.mlx.tests.kernels import test_current_gemm as gate

HOST = Path(layout.__file__).parent
RECORDS = [
    record
    for record in (verify.case_record(case) for case in verify.CASES)
    if record["family"] == "gemm"
]
BY_NAME = {record["name"]: record for record in RECORDS}
FLAGS = ("has_batch", "align_M", "align_N", "align_K")


def host_entry(transpose, dtype, flags):
    has_batch, align_m, align_n, align_k = ("t" if value else "n" for value in flags)
    return (
        f"steel_gemm_fused_{transpose}_{dtype}_{dtype}_bm64_bn64_bk16_wm2_wn2"
        f"_has_batch_{has_batch}_use_out_source_n_do_axpby_n"
        f"_align_M_{align_m}_align_N_{align_n}_align_K_{align_k}"
    )


def native_binding(record):
    """What backend.cpp binds for one verification case."""
    route = record["route"]
    operands = bound_operands(record)
    counts = {
        "A": span(*operands["a"]),
        "B": span(*operands["b"]),
        "D": route["batches"] * route["M"] * route["N"],
    }
    parameters = {
        "params": dict(route["params"]),
        "batch_shape": list(route["batchShape"]),
        "batch_strides": list(route["batchStrides"]),
    }
    execution = {
        "workgroupCount": [
            route["params"]["tiles_n"],
            route["params"]["tiles_m"],
            route["batches"],
        ],
        "workgroupSize": [32, 2, 2],
    }
    return counts, parameters, execution


def native_buffers(record, data=None):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    counts, parameters, execution = native_binding(record)
    entry, memory, buffers = record["entry"], {}, []
    for name in layout.HOST_PARAMETERS:
        dtype = layout.parameter_type(entry, name)
        if name in counts:
            values = (data or {}).get(name, [0] * counts[name])
        elif name == "params":
            values = [parameters[name][field] for field in layout.PARAMS_FIELDS]
        else:
            values = parameters[name]
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == "D"),
            )
        )
    return buffers, memory, execution


def test_signature_accepts_exactly_the_host_tile_entries():
    entries = {
        host_entry(transpose, dtype, flags)
        for transpose in ("nn", "nt", "tn", "tt")
        for dtype in layout.DTYPES
        for flags in itertools.product((False, True), repeat=4)
    }
    assert len(entries) == 192
    for entry in entries:
        info = layout.signature(entry)
        assert info["source"] in model.BY_NAME
        assert model.BY_NAME[info["source"]].workgroup_size == (32, 2, 2)
        assert layout.workgroup_size(entry) == (32, 2, 2)
        assert info["constants"]["use_out_source"] is False
        assert set(info["constants"]) == set(layout.CONSTANT_IDS)
    base = host_entry("nn", "float32", (False,) * 4)
    for entry in (
        base.replace("_bm64_bn64_bk16_wm2_wn2", "_bm32_bn32_bk16_wm2_wn2"),
        base.replace("_bm64_bn64_bk16_wm2_wn2", "_bm64_bn64_bk16_wm1_wn2"),
        base.replace("float32_float32", "complex64_complex64"),
        base.replace("float32_float32", "float32_float16"),
        base.replace("use_out_source_n", "use_out_source_t"),
        base.replace("do_axpby_n", "do_axpby_t"),
        base.replace("_has_batch_n", ""),
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2",
        "steel_gemm_splitk_nn_float32_float32_bm16_bn16_bk16_wm2_wn2",
        base.encode(),
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


def test_constant_ids_match_the_gemm_gate():
    assert layout.CONSTANT_IDS == gate.CONSTANT_IDS
    assert layout.SOURCE == gate.SOURCE


def test_split_k_follows_steel_matmul_axpby():
    assert layout.split_k(16, 16, 256, 1)
    assert layout.split_k(3, 256, 256, 1)
    assert not layout.split_k(16, 16, 256, 2)
    assert not layout.split_k(16, 16, 127, 1)
    assert not layout.split_k(3, 512, 256, 1)
    assert layout.split_k(512, 512, 512, 1)
    # 33 x 32 tiles of 16 exceed the threshold of 1024.
    assert not layout.split_k(528, 512, 4096, 1)
    assert layout.split_k(512, 512, 4096, 1)


def test_selection_names_matmul_cpp_hash():
    entry = layout.select("bfloat16", False, True, 128, 70, 48, 2)
    assert entry == host_entry("nt", "bfloat16", (True, True, False, True))
    assert layout.signature(entry)["transposeB"]
    with pytest.raises(ValueError):
        layout.select("complex64", False, False, 64, 64, 64, 1)


def test_routes_mirror_matmul_eval_gpu():
    routes = {name: record["route"] for name, record in BY_NAME.items()}
    folded = routes["gemm-batch-folded-into-rows"]
    assert (folded["M"], folded["batches"], folded["batchShape"]) == (15, 1, [1])
    assert folded["batchStrides"] == [0, 0]
    two = routes["gemm-two-batch-axes"]
    assert two["batchShape"] == [2, 3]
    assert two["batchStrides"] == [1224, 408, 0, 456]
    assert two["params"]["batch_stride_a"] == 408
    assert two["params"]["batch_stride_b"] == 456
    batched = routes["gemm-batched"]
    assert batched["params"]["batch_stride_a"] == 600
    assert batched["params"]["batch_stride_b"] == 1200
    assert batched["params"]["batch_stride_d"] == 800
    tn = routes["gemm-tn"]["params"]
    assert (tn["lda"], tn["ldb"], tn["ldd"]) == (65, 66, 66)
    tt = routes["gemm-tt"]["params"]
    assert (tt["lda"], tt["ldb"], tt["tiles_m"], tt["tiles_n"]) == (130, 24, 3, 2)
    assert routes["gemm-copied-operand"]["copies"] == {"a": True, "b": False}
    assert routes["gemm-copied-operand"]["params"]["lda"] == 32
    # test_qmv's unbatched reference folds three vectors into M; with
    # K = 256 and N = 512 it is a regular GEMM.
    qmv = layout.route("float32", [3, 256], [256, 1], [256, 512], [1, 256])
    assert qmv["entry"] == host_entry("nt", "float32", (False, False, True, True))
    for a, b, message in (
        (([16, 256], [256, 1]), ([256, 16], [16, 1]), "Split-K"),
        (([3, 256], [256, 1]), ([256, 256], [1, 256]), "Split-K"),
        (([1, 8], [8, 1]), ([8, 5], [5, 1]), "gemv route"),
        (([1, 16], [16, 1]), ([16, 1], [1, 1]), "gemv route"),
        (([2, 8], [8, 1]), ([3, 8, 1], [8, 1, 1]), "ranks"),
    ):
        with pytest.raises(ValueError, match=message):
            layout.route("float32", *a, *b)
    assert layout.matmul_route("float32", [4, 8], [8, 1], [8, 5], [5, 1])[0] == "gemm"
    assert layout.matmul_route("float32", [1, 8], [8, 1], [8, 5], [5, 1])[0] == "gemv"
    with pytest.raises(ValueError, match="dot products"):
        layout.matmul_route("float32", [1, 16], [16, 1], [16, 1], [1, 1])


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record):
    counts, parameters, execution = native_binding(record)
    metadata = layout.plan(record["entry"], counts, parameters, counts["D"], execution)
    route = record["route"]
    assert [metadata[key] for key in ("M", "N", "K", "batches")] == [
        route[key] for key in ("M", "N", "K", "batches")
    ]
    assert metadata["source"] == layout.signature(record["entry"])["source"]
    assert set(metadata["inert"]) == {"ldc", "fdc", "batch_stride_c", "alpha", "beta"}


def mutate(fault, counts, parameters, execution):
    params = parameters["params"]
    if fault == "tiles":
        execution["workgroupCount"][0] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 2, 1]
    elif fault == "batches":
        execution["workgroupCount"][2] += 1
    elif fault == "threads":
        counts["D"] += 1
    elif fault == "a-short":
        counts["A"] -= 1
    elif fault == "b-short":
        counts["B"] -= 1
    elif fault == "negative-stride":
        parameters["batch_strides"][0] = -1
    elif fault == "ldd":
        params["ldd"] += 1
    elif fault == "lda":
        params["lda"] = 1
    elif fault == "swizzle":
        params["swizzle_log"] = 1
    elif fault == "iterations":
        params["gemm_k_iterations_aligned"] += 1
    elif fault == "rank":
        params["batch_ndim"] += 1
    elif fault == "batch-stride":
        params["batch_stride_b"] += 1
    elif fault == "output-stride":
        params["batch_stride_d"] += 1
    elif fault == "dimension":
        params["K"] += 16
    elif fault == "bound":
        counts["A"] = layout.MAX_ELEMENTS + 1
        params["lda"] = layout.MAX_ELEMENTS + 1
    elif fault == "integer":
        params["M"] = float(params["M"])


FAULTS = (
    "tiles",
    "workgroup",
    "batches",
    "threads",
    "a-short",
    "b-short",
    "negative-stride",
    "ldd",
    "lda",
    "swizzle",
    "iterations",
    "rank",
    "batch-stride",
    "output-stride",
    "dimension",
    "bound",
    "integer",
)


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("name", ("gemm-two-batch-axes", "gemm-batched", "gemm-tt"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    record = BY_NAME[name]
    counts, parameters, execution = native_binding(record)
    mutate(fault, counts, parameters, execution)
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, counts["D"], execution)


def test_plan_rejects_entries_mlx_would_not_select():
    record = BY_NAME["gemm-nn-aligned"]
    counts, parameters, execution = native_binding(record)
    # Only the function constants are visible to the plan: MLX's transposition
    # follows the operand strides the host itself derives.
    for other in (
        record["entry"].replace("align_M_t", "align_M_n"),
        record["entry"].replace("has_batch_n", "has_batch_t"),
    ):
        with pytest.raises(ValueError, match="routing"):
            layout.plan(other, counts, parameters, counts["D"], execution)
    # A product MLX would send to split-K.
    params = dict(parameters["params"], M=64, N=64, K=256, tiles_m=1, tiles_n=1)
    params.update(gemm_k_iterations_aligned=16, batch_stride_d=64 * 64, lda=256)
    params.update(ldb=64, ldd=64)
    execution["workgroupCount"] = [1, 1, 1]
    counts = {"A": 64 * 256, "B": 256 * 64, "D": 64 * 64}
    split = dict(parameters, params=params)
    with pytest.raises(ValueError, match="routing"):
        layout.plan(record["entry"], counts, split, counts["D"], execution)


@pytest.mark.parametrize(
    "fault", ("dtype", "direction", "empty", "params", "names", "rank")
)
def test_storage_rejects_native_identity_faults(fault):
    record = BY_NAME["gemm-two-batch-axes"]
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    if fault == "dtype":
        supplied["A"].dtype = b"float32"
    elif fault == "direction":
        supplied["B"].output = 1
    elif fault == "empty":
        supplied["A"].count = 0
    elif fault == "params":
        supplied["params"].count = 13
    elif fault == "rank":
        supplied["batch_shape"].count = layout.MAX_BATCH_DIMENSIONS + 1
    else:
        supplied.pop("batch_strides")
    with pytest.raises(ValueError):
        layout.read_storage(record["entry"], supplied)
    assert memory


def test_storage_reads_params_and_rejects_aliased_outputs():
    record = BY_NAME["gemm-batched"]
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    counts, parameters = layout.read_storage(record["entry"], supplied)
    assert parameters["params"] == record["route"]["params"]
    assert counts["D"] == 3 * 20 * 40
    supplied["D"].data = supplied["A"].data
    with pytest.raises(ValueError, match="overlaps"):
        layout.read_storage(record["entry"], supplied)
    assert memory


def struct_layout(name):
    """The reflected record layout of GEMMParams or GEMMAddMMParams."""
    from crosstl.project.storage_record_layout import storage_record_layout

    if name == "params":
        types = ["int"] * 8 + ["int64_t"] * 3 + ["int"] * 3
        type_name, names = "GEMMParams", layout.PARAMS_FIELDS
    else:
        types = ["int", "int", "int64_t", "float", "float"]
        type_name, names = "GEMMAddMMParams", tuple(layout.INERT_ADDMM)
    return storage_record_layout(
        type_name, list(zip(types, names)), storage_layout="metal-buffer"
    )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "missing", "shape", "trace", "unbound", "readonly", "duplicate"),
)
@pytest.mark.parametrize("name", ("gemm-two-batch-axes", "gemm-tn"))
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, name
):
    record = BY_NAME[name]
    entry, dtype = record["entry"], record["dtype"]
    buffers, memory, execution = native_buffers(record)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in layout.PARAMETERS:
        if fault == "unbound" and parameter == "A":
            continue
        scalar = (
            struct_layout(parameter)
            if parameter in dispatch.RECORD_STRIDES
            else physical_layout(target, dispatch._type(entry, parameter))
        )
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": (
                    "read_write" if parameter == "D" and fault != "readonly" else "read"
                ),
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": scalar,
            }
        )
    if fault == "duplicate":
        descriptor["bindings"].append(dict(descriptor["bindings"][0], name="again"))
    host = SimpleNamespace(
        target=target,
        gemms=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, kwargs))
        return outputs

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    monkeypatch.setattr(
        dispatch,
        "pack_storage_records",
        lambda layout_, records: [len(layout_["structMembers"])] + [0] * 5,
    )
    one = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_D"]
        count = len(memory["D"])
        value["values"][:count] = dispatch.encode_binding(
            dtype, [one] * count, target, physical_layout(target, dtype), "buffer"
        )["values"]
        if fault == "guard":
            value["values"][-1] = value["values"][0]
        elif fault == "missing":
            outputs.clear()
        elif fault == "shape":
            value["shape"] = [1]
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    supplied = (runtime.Buffer * len(buffers))(*buffers)
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    call = lambda: dispatch.dispatch(
        host, entry, supplied, len(buffers), len(memory["D"]), launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["D"])
        return
    call()
    assert list(memory["D"]) == [one] * len(memory["D"])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == set(inputs) - {"binding_D"}
    for parameter in ("A", "B"):
        assert len(inputs[f"binding_{parameter}"]["values"]) == (
            len(memory[parameter]) + dispatch.GUARD_COUNT
        )
    # C is never read without use_out_source, so it holds only poison.
    assert len(inputs["binding_C"]["values"]) == dispatch.GUARD_COUNT
    assert inputs["binding_params"]["dtype"] == "uint32"
    assert inputs["binding_params"]["values"][0] == len(layout.PARAMS_FIELDS)
    assert inputs["binding_addmm_params"]["values"][0] == 5
    assert inputs["binding_batch_strides"]["values"] == record["route"]["batchStrides"]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry
    assert event["gemmMetadata"]["M"] == record["route"]["M"]
    assert event["parameters"]["params"] == record["route"]["params"]
    assert event["boundParameters"] == sorted(layout.PARAMETERS)


def test_dispatch_packs_reflected_gemm_params():
    from crosstl.project import pack_storage_records

    params = BY_NAME["gemm-two-batch-axes"]["route"]["params"]
    binding = {
        "provenance": {"sourceResource": {"parameter": "params"}},
        "scalarLayout": struct_layout("params"),
    }
    value = dispatch._record(binding, params)
    assert value["dtype"] == "uint32" and len(value["values"]) == 18
    assert value["values"] == pack_storage_records(binding["scalarLayout"], [params])
    assert value["values"][:3] == [params["M"], params["N"], params["K"]]
    binding["scalarLayout"] = dict(binding["scalarLayout"], elementStrideBytes=80)
    with pytest.raises(ValueError, match="stride"):
        dispatch._record(binding, params)


@pytest.mark.parametrize("record", RECORDS[:3], ids=lambda record: record["name"])
def test_package_configs_match_the_gemm_gate(tmp_path, record):
    entry = record["entry"]
    info = layout.signature(entry)
    gate_entry = model.BY_NAME[info["source"]]
    for target in ("metal", "directx", "opengl"):
        config = gemm_packages.project_config(tmp_path, "out", target, entry)
        expected = gate._config(tmp_path, "out", target, gate_entry)
        assert config.entry_points == {layout.SOURCE: (info["source"],)}
        assert config.specialization_constants == info["constants"]
        assert config.freeze_specialization_constants is True
        assert config.source_options == expected.source_options
        assert config.workgroup_size_rules == expected.workgroup_size_rules
        assert config.workgroup_access_assertions == (
            expected.workgroup_access_assertions
        )
        assert config.index_range_assertions == expected.index_range_assertions
        assert bool(config.index_range_assertions) == (target == "opengl")
    cache = gemm_packages.GemmPackageCache(tmp_path, tmp_path / "cache", "opengl")
    assert cache.SOURCE == layout.SOURCE
    assert cache.source_entry(entry) == info["source"]
    assert cache.workgroup_size(entry) == (32, 2, 2)
    with pytest.raises(ValueError, match="gemm target"):
        gemm_packages.GemmPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import matmul_packages

    entries = gemm_packages.verification_entries()
    assert entries == [record["entry"] for record in RECORDS]
    assert len(set(entries)) == 9
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(gemm_packages, "_build_one", build)
    records = gemm_packages.build_entries(
        tmp_path, tmp_path / "packages", "opengl", [*entries, entries[0]], 3
    )
    assert [record["entry"] for record in records] == sorted(set(entries))
    assert {call[:2] for call in calls} == {
        ((tmp_path / "packages" / gemm_packages.CACHE_DIRECTORY).resolve(), "opengl")
    }
    with pytest.raises(ValueError):
        gemm_packages.build_entries(
            tmp_path, tmp_path, "opengl", ["steel_gemm_fused_nn_float32"], 1
        )


def test_runtime_routes_gemm_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.gemvs = None
    host.gemms = None
    entry = RECORDS[0]["entry"]
    assert host._entry_available(entry.encode()) == 0
    host.gemms = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(entry.replace("_bm64_", "_bm32_").encode()) == 0
    calls = []
    monkeypatch.setattr(
        runtime.gemm_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(entry, [], 6, 1, launch=None)
    assert calls == [entry]


def test_backend_mirrors_steel_matmul():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    for text in (
        "void dispatch_steel_gemm(",
        "split_tiles <= 1024 && K / 16 >= 8 && K >= std::max(M, N)",
        "dispatch_steel_gemm_splitk(a, b, out, M, N, K, lda, ldb, transpose_a, transpose_b);",
        "constexpr int bm = 64, bn = 64, bk = 16, wm = 2, wn = 2;",
        '"_use_out_source_n_do_axpby_n_align_M_"',
        "A_batch_stride.back(),",
        "{32, uint32_t(wn), uint32_t(wm)}",
        "{uint32_t(tiles_n), uint32_t(tiles_m), uint32_t(batch_size_out)}",
    ):
        assert text in backend, text
    for name in layout.HOST_PARAMETERS:
        assert f'{{"{name}", ' in backend, name


def test_case_data_is_exact_and_rounded():
    import numpy as np

    for seed, record in enumerate(RECORDS):
        data = verify.case_data(np, record, seed)
        exact = np.matmul(
            verify._view(np, data["a"], record["a"]["view"]),
            verify._view(np, data["b"], record["b"]["view"]),
        )
        assert data["expected"].shape == exact.shape
        assert np.abs(exact).max() * 32 < 2**24
        assert np.any(data["expected"] != 0)
        route = record["route"]
        assert math.prod(exact.shape) == route["batches"] * route["M"] * route["N"]
