"""Floating-point matrix-vector host routing, launch contracts and guarded commits."""

import ctypes
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gemv_dispatch as dispatch
from demos.integrations.mlx.portable_host import gemv_layout as layout
from demos.integrations.mlx.portable_host import gemv_packages, prepare, runtime
from demos.integrations.mlx.portable_host import verify_matmul as verify
from demos.integrations.mlx.portable_host.matmul_dispatch import STRIDES

HOST = Path(layout.__file__).parent
CTYPES = {
    "float32": ctypes.c_uint32,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
}
ALL_RECORDS = [verify.case_record(case) for case in verify.CASES]
RECORDS = [record for record in ALL_RECORDS if record["family"] == "gemv"]


def span(shape, strides):
    return 1 + sum((extent - 1) * stride for extent, stride in zip(shape, strides))


def bound_operands(record):
    """The views backend.cpp binds: copies are row-major."""
    operands = {}
    for name in ("a", "b"):
        shape, strides = verify.view_layout(
            record[name]["stored"], record[name]["view"]
        )
        if record["route"]["copies"][name]:
            strides = layout._row_strides(shape)
        operands[name] = (shape, strides)
    return operands


def native_binding(record):
    """What backend.cpp binds for one verification case."""
    route = record["route"]
    operands = bound_operands(record)
    mat, vec = ("b", "a") if route["N"] != 1 else ("a", "b")
    counts = {
        "mat": span(*operands[mat]),
        "in_vec": span(*operands[vec]),
        "out_vec": route["batches"] * route["outVecSize"],
    }
    parameters = {
        "in_vec_size": route["K"],
        "out_vec_size": route["outVecSize"],
        "marix_ld": route["matrixLd"],
        "batch_ndim": len(route["batchShape"]),
        "batch_shape": list(route["batchShape"]),
        "vector_batch_stride": list(route["vectorBatchStride"]),
        "matrix_batch_stride": list(route["matrixBatchStride"]),
    }
    execution = {
        "workgroupCount": [
            -(-route["outVecSize"] // layout.outputs_per_group(record["entry"])),
            1,
            route["batches"],
        ],
        "workgroupSize": list(layout.workgroup_size(record["entry"])),
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
                int(name == "out_vec"),
            )
        )
    return buffers, memory, execution


def test_signature_accepts_exactly_the_instantiated_matmul_entries():
    entries = {
        f"{kind}_{dtype}_bm{bm}_bn{bn}_sm{sm}_sn{sn}_tm{tm}_tn{tn}_nc{nc}_axpby0"
        for kind, blocks in layout.BLOCKS.items()
        for dtype in layout.DTYPES
        for bm, bn, sm, sn, tm, tn in blocks
        for nc in (0, 1)
    }
    assert len(entries) == 72
    for entry in entries:
        info = layout.signature(entry)
        assert layout.workgroup_size(entry) == (32, info["bn"], info["bm"])
    for entry in (
        "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby1",
        "gemv_complex64_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4",
        "gemv_t_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_float32_bm2_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
        "gemv_wide_float16_nv5_kl32",
        b"gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)


@pytest.mark.parametrize("transpose", (False, True))
def test_selection_follows_gemv_axbpy_and_stays_instantiated(transpose):
    seen = set()
    for k in (1, 33, 64, 65, 96, 1024, 8191, 8192, 16384):
        for out in (1, 2, 3, 4, 100, 511, 512, 2047, 2048, 4095, 4096):
            for ndim in (1, 2):
                entry = layout.select("float16", transpose, k, out, ndim)
                info = layout.signature(entry)
                assert info["kind"] == ("gemv_t" if transpose else "gemv")
                assert info["nc"] == (ndim == 2)
                seen.add(entry.split("_nc")[0])
    assert layout.select("float32", True, 8192, 2048, 1).startswith(
        "gemv_t_float32_bm1_bn16_sm4_sn8_tm4_tn4"
    )
    assert layout.select("float32", True, 8191, 2048, 1).startswith(
        "gemv_t_float32_bm1_bn16_sm8_sn4_tm4_tn4"
    )
    assert layout.select("float32", False, 64, 4096, 1).startswith(
        "gemv_float32_bm1_bn1_sm8_sn4"
    )
    assert layout.select("float32", False, 65, 4096, 1).startswith(
        "gemv_float32_bm8_bn1_sm1_sn32"
    )
    assert layout.select("float32", False, 1600, 100, 1).startswith(
        "gemv_float32_bm1_bn8_sm1_sn32"
    )
    # Matmul reaches every instantiated tile but gemv's (4, 1, 1, 32, 1, 4):
    # fewer than 4 outputs over K > 64 always satisfies K >= 16 * outputs.
    assert len(seen) == (5 if transpose else 6)
    with pytest.raises(ValueError):
        layout.select("complex64", transpose, 8, 8, 1)


def test_collapse_contiguous_dims_matches_mlx():
    collapse = layout.collapse_contiguous_dims
    assert collapse([2, 3], [[3, 1], [3, 1]]) == ([6], [[1], [1]])
    assert collapse([2, 3], [[3, 1], [0, 1]]) == ([2, 3], [[3, 1], [0, 1]])
    assert collapse([1, 4, 1], [[4, 1, 1], [0, 0, 0]]) == ([4], [[1], [0]])
    assert collapse([1, 1], [[1, 1], [1, 1]]) == ([1], [[0], [0]])
    assert collapse([2, 1, 3], [[3, 9, 1], [6, 6, 2]]) == ([2, 3], [[3, 1], [6, 2]])
    assert collapse([4, 4], [[4, 1], [4, 1]], size_cap=8) == ([4, 4], [[4, 1], [4, 1]])


def test_routes_mirror_matmul_eval_gpu():
    routes = {record["name"]: record["route"] for record in RECORDS}
    folded = routes["batch-folded-into-rows"]
    assert (folded["M"], folded["batches"], folded["batchShape"]) == (12, 1, [1])
    two = routes["vector-matrix-two-batch-axes"]
    assert two["batchShape"] == [2, 3] and two["matrixBatchStride"] == [0, 2048]
    assert routes["vector-matrix-shared-vector"]["vectorBatchStride"] == [0]
    broadcast = routes["vector-matrix-broadcast-vector"]
    assert broadcast["copies"] == {"a": True, "b": False}
    assert broadcast["vectorBatchStride"] == [48]
    assert routes["strided-vector"]["copies"] == {"a": True, "b": False}
    assert routes["transposed-matrix-column"]["matrixLd"] == 50
    # Upstream test_qvm's references are vector-matrix products, test_qmv's
    # batched ones matrix-vector products over transposed weights.
    qvm = layout.route("float32", [1, 67], [67, 1], [67, 256], [256, 1])
    assert qvm["entry"] == "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
    qmv = layout.route(
        "float32", [8, 1, 256], [256, 256, 1], [8, 256, 512], [131072, 1, 256]
    )
    assert qmv["entry"] == "gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0"
    assert qmv["batches"] == 8 and qmv["matrixBatchStride"] == [131072]
    # Negative strides are copied before routing.
    flipped = layout.route("float32", [1, 8], [8, -1], [8, 4], [4, 1])
    assert flipped["copies"]["a"] and flipped["entry"].startswith("gemv_t_")
    for a, b, message in (
        (([4, 8], [8, 1]), ([8, 5], [5, 1]), "steel GEMM"),
        # test_qmv's unbatched reference folds three vectors into M.
        (([3, 64], [64, 1]), ([64, 256], [1, 64]), "steel GEMM"),
        (([1, 16], [16, 1]), ([16, 1], [1, 1]), "dot products"),
        (([2, 8], [8, 1]), ([3, 8, 1], [8, 1, 1]), "ranks"),
    ):
        with pytest.raises(ValueError, match=message):
            layout.route("float32", *a, *b)


def test_case_views_match_numpy_strides():
    import numpy as np

    for record in RECORDS:
        for name in ("a", "b"):
            stored, view = record[name]["stored"], record[name]["view"]
            values = np.zeros(stored, dtype=np.float32)
            expected = verify._view(np, values, view)
            shape, strides = verify.view_layout(stored, view)
            assert list(expected.shape) == shape
            assert [stride // 4 for stride in expected.strides] == strides


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record):
    counts, parameters, execution = native_binding(record)
    metadata = layout.plan(
        record["entry"], counts, parameters, counts["out_vec"], execution
    )
    route = record["route"]
    assert (
        metadata["K"],
        metadata["outVecSize"],
        metadata["matrixLd"],
        metadata["batches"],
    ) == (route["K"], route["outVecSize"], route["matrixLd"], route["batches"])
    assert set(metadata["inert"]) | set(layout.HOST_PARAMETERS) == set(
        layout.PARAMETERS
    ) - {"bias"}
    assert metadata["inert"]["bias_batch_stride"] == [0] * len(route["batchShape"])


def mutate(fault, counts, parameters, execution):
    if fault == "tiles":
        execution["workgroupCount"][0] += 1
    elif fault == "rows":
        execution["workgroupCount"][1] = 2
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 4, 4]
    elif fault == "threads":
        counts["out_vec"] += 1
    elif fault == "vector-short":
        counts["in_vec"] -= 1
    elif fault == "matrix-short":
        counts["mat"] -= 1
    elif fault == "negative-stride":
        parameters["matrix_batch_stride"][0] = -1
    elif fault == "rank":
        parameters["batch_ndim"] += 1
    elif fault == "batch-extent":
        parameters["batch_shape"][0] += 1
    elif fault == "variant":
        parameters.update(
            batch_ndim=1,
            batch_shape=[math.prod(parameters["batch_shape"])],
            vector_batch_stride=parameters["vector_batch_stride"][-1:],
            matrix_batch_stride=parameters["matrix_batch_stride"][-1:],
        )
    elif fault == "dimension":
        parameters["in_vec_size"] = 2 * parameters["in_vec_size"]
    elif fault == "bound":
        counts["mat"] = layout.MAX_ELEMENTS + 1
    elif fault == "ld":
        parameters["marix_ld"] = -1


@pytest.mark.parametrize(
    "fault",
    (
        "tiles",
        "rows",
        "workgroup",
        "threads",
        "vector-short",
        "matrix-short",
        "negative-stride",
        "rank",
        "batch-extent",
        "variant",
        "dimension",
        "bound",
        "ld",
    ),
)
@pytest.mark.parametrize(
    "name", ("vector-matrix-two-batch-axes", "transposed-weights-batched")
)
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    record = next(item for item in RECORDS if item["name"] == name)
    if fault == "variant" and not layout.signature(record["entry"])["nc"]:
        record = next(
            item for item in RECORDS if item["name"] == "vector-matrix-two-batch-axes"
        )
    counts, parameters, execution = native_binding(record)
    mutate(fault, counts, parameters, execution)
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, counts["out_vec"], execution)


def test_plan_rejects_entries_mlx_would_not_select():
    record = next(item for item in RECORDS if item["name"] == "transposed-weights")
    counts, parameters, execution = native_binding(record)
    other = record["entry"].replace("_bm4_", "_bm8_")
    with pytest.raises(ValueError):
        layout.plan(other, counts, parameters, counts["out_vec"], execution)
    transposed = "gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0"
    execution["workgroupSize"] = [32, 2, 1]
    execution["workgroupCount"][0] = -(-67 // 32)
    with pytest.raises(ValueError, match="routing|buffer"):
        layout.plan(transposed, counts, parameters, counts["out_vec"], execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "empty", "scalar", "names"))
def test_storage_rejects_native_identity_faults(fault):
    record = RECORDS[4]
    buffers, memory, _ = native_buffers(record)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    if fault == "dtype":
        supplied["in_vec"].dtype = b"float16"
    elif fault == "direction":
        supplied["mat"].output = 1
    elif fault == "empty":
        supplied["mat"].count = 0
    elif fault == "scalar":
        supplied["marix_ld"].count = 2
    else:
        supplied.pop("batch_shape")
    with pytest.raises(ValueError):
        layout.read_storage(record["entry"], supplied)
    assert memory


def test_storage_rejects_outputs_that_alias_inputs():
    buffers, memory, _ = native_buffers(RECORDS[0])
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    supplied["out_vec"].data = supplied["in_vec"].data
    supplied["out_vec"].count = 1
    with pytest.raises(ValueError, match="overlaps"):
        layout.read_storage(RECORDS[0]["entry"], supplied)
    assert memory


def physical_layout(target, dtype):
    if dtype in ("float16", "bfloat16") and target != "metal":
        physical = "uint16" if target == "directx" else "float32"
    else:
        physical = dtype
    scalar = {"elementType": physical, "elementStrideBytes": STRIDES[physical]}
    if dtype == "float16" and target == "directx":
        scalar.update(
            storageLayout="hlsl-structured-buffer",
            runtimeSized=True,
            storageEncoding={
                "encoding": "ieee754-binary16",
                "logicalElementType": "float16",
            },
        )
    return scalar


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "missing", "shape", "trace", "unbound", "readonly", "duplicate"),
)
@pytest.mark.parametrize(
    "name", ("vector-matrix-two-batch-axes", "transposed-weights-short-k")
)
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, name
):
    record = next(item for item in RECORDS if item["name"] == name)
    entry, dtype = record["entry"], record["dtype"]
    buffers, memory, execution = native_buffers(record)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in layout.PARAMETERS:
        if fault == "unbound" and parameter == "mat":
            continue
        kind = dispatch._type(entry, parameter)
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": (
                    "read_write"
                    if parameter == "out_vec" and fault != "readonly"
                    else "read"
                ),
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": physical_layout(target, kind),
            }
        )
    if fault == "duplicate":
        descriptor["bindings"].append(dict(descriptor["bindings"][0], name="again"))
    host = SimpleNamespace(
        target=target,
        gemvs=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, kwargs))
        return outputs

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)
    one = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}[dtype]

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_out_vec"]
        count = len(memory["out_vec"])
        value["values"][:count] = dispatch.encode_binding(
            dtype,
            [one] * count,
            target,
            physical_layout(target, dtype),
            "buffer",
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
        host, entry, supplied, len(buffers), len(memory["out_vec"]), launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["out_vec"])
        return
    call()
    assert list(memory["out_vec"]) == [one] * len(memory["out_vec"])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == set(inputs) - {"binding_out_vec"}
    for parameter in ("mat", "in_vec"):
        assert len(inputs[f"binding_{parameter}"]["values"]) == (
            len(memory[parameter]) + dispatch.GUARD_COUNT
        )
    # The bindings gemv_axbpy leaves unbound get inert values.
    assert len(inputs["binding_bias"]["values"]) == dispatch.GUARD_COUNT
    assert inputs["binding_bias_stride"]["values"] == [0]
    assert inputs["binding_bias_batch_stride"]["values"] == [0] * len(
        record["route"]["batchShape"]
    )
    alpha = inputs["binding_alpha"]
    assert alpha["dtype"] == "float32" and len(alpha["values"]) == 1
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry
    assert event["gemvMetadata"]["outVecSize"] == record["route"]["outVecSize"]
    assert event["boundParameters"] == sorted(layout.PARAMETERS)


def test_package_configs_scope_target_options_and_assertions(tmp_path):
    entry = "gemv_t_float16_bm1_bn2_sm8_sn4_tm4_tn4_nc1_axpby0"
    gl = gemv_packages.project_config(tmp_path, "out", "opengl", entry)
    assert [item.expression for item in gl.index_range_assertions] == list(
        layout.INDEX_RANGE_ASSERTIONS["gemv_t"] + layout.INDEX_RANGE_ASSERTIONS["batch"]
    )
    assert {item.value_range.maximum for item in gl.index_range_assertions} == {
        layout.SIGNED_MAX
    }
    for target in ("metal", "directx"):
        config = gemv_packages.project_config(tmp_path, "out", target, entry)
        assert not config.index_range_assertions
        options = config.source_options["metal"]["target_options"]
        assert options["directx"]["software_subgroup_width"] == 32
        assert config.entry_points == {layout.SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[layout.SOURCE].values()
        assert tuple(int(value) for value in rule) == (32, 2, 1)
    cache = gemv_packages.GemvPackageCache(tmp_path, tmp_path / "cache", "opengl")
    assert cache.SOURCE == layout.SOURCE and cache.workgroup_size(entry) == (32, 2, 1)
    with pytest.raises(ValueError, match="matrix-vector target"):
        gemv_packages.GemvPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import matmul_packages

    entries = gemv_packages.verification_entries()
    assert entries == [record["entry"] for record in RECORDS]
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(gemv_packages, "_build_one", build)
    records = gemv_packages.build_entries(
        tmp_path, tmp_path / "packages", "directx", [*entries, entries[0]], 3
    )
    assert [record["entry"] for record in records] == sorted(set(entries))
    assert {call[:2] for call in calls} == {
        ((tmp_path / "packages" / gemv_packages.CACHE_DIRECTORY).resolve(), "directx")
    }
    with pytest.raises(ValueError, match="entries and jobs"):
        gemv_packages.build_entries(tmp_path, tmp_path, "directx", [], 1)
    with pytest.raises(ValueError):
        gemv_packages.build_entries(
            tmp_path, tmp_path, "directx", ["gemv_complex64_bm4_bn1"], 1
        )
    calls.clear()
    argv = ["--mlx-root", str(tmp_path), "--packages", str(tmp_path)]
    assert gemv_packages.main([*argv, "--target", "metal", "--verification"]) == 0
    assert sorted(call[2] for call in calls) == sorted(set(entries))


def test_runtime_routes_gemv_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = None
    host.gemvs = None
    entry = RECORDS[0]["entry"]
    assert host._entry_available(entry.encode()) == 0
    host.gemvs = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(entry.replace("axpby0", "axpby1").encode()) == 0
    assert host._entry_available(b"gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4") == 0
    calls = []
    monkeypatch.setattr(
        runtime.gemv_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(entry, [], 10, 1, launch=None)
    assert calls == [entry]


def test_preparation_replaces_the_matmul_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    prepare.adapted_sources("", "", "")
    assert "NO_GPU(Matmul)" in replaced and "NO_GPU(AddMM)" not in replaced
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert "void Matmul::eval_gpu(" in backend
    for text in (
        '#include "mlx/backend/common/matmul.h"',
        "collapse_batches(a, b)",
        "A_batch_stride.back() == M * K",
        "K >= 8192 && out_vector_len >= 2048",
        "out_vector_len >= 2048 ? 16 : out_vector_len >= 512 ? 4 : 2",
        "bm = out_vector_len >= 4096 ? 8 : 4;",
        "} else if (K >= 16 * out_vector_len) {",
        '(batch_shape.size() == 1 ? "0" : "1") + "_axpby0"',
        "{32, uint32_t(bn), uint32_t(bm)}",
        "dispatch_steel_gemm_splitk(a, b, out, M, N, K, lda, ldb, transpose_a, transpose_b);",
        "does not route single dot products yet",
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
        if record["dtype"] == "float32":
            assert np.array_equal(data["expected"].astype(np.float64), exact)
        assert np.any(data["expected"] != 0)


def trace_events(record, target):
    """The trace events the dispatches of ``record`` write."""
    route = record["route"]
    if record["family"] == "splitk":
        keys = ("M", "N", "K", "partitions")
        return [
            {
                "entry": record["entry"],
                "target": target,
                "splitkMetadata": {key: route[key] for key in keys},
                "parameters": {"params": dict(route["params"])},
                "outputHash": f"partitions-{record['name']}",
            },
            {
                "entry": route["accum"],
                "target": target,
                "splitkMetadata": {"partitions": route["partitions"]},
                "parameters": dict(route["accumParams"]),
                "inputHashes": {"C_split": f"partitions-{record['name']}"},
            },
        ]
    return [trace_event(record, target)]


def trace_event(record, target):
    """The trace event a dispatch of a single-kernel ``record`` writes."""
    route = record["route"]
    if record["family"] == "gemm":
        return {
            "entry": record["entry"],
            "target": target,
            "gemmMetadata": {key: route[key] for key in ("M", "N", "K", "batches")},
            "parameters": {
                "params": dict(route["params"]),
                "batch_shape": list(route["batchShape"]),
                "batch_strides": list(route["batchStrides"]),
            },
        }
    return {
        "entry": record["entry"],
        "target": target,
        "gemvMetadata": {
            "K": route["K"],
            "outVecSize": route["outVecSize"],
            "matrixLd": route["matrixLd"],
            "batches": route["batches"],
        },
        "parameters": {
            "batch_shape": route["batchShape"],
            "vector_batch_stride": route["vectorBatchStride"],
            "matrix_batch_stride": route["matrixBatchStride"],
        },
    }


def test_trace_validation_requires_the_routed_entries():
    trace = [
        event for record in ALL_RECORDS for event in trace_events(record, "opengl")
    ]
    copy = {"entry": "ggn2_dynamic_copyfloat16float16", "target": "opengl"}
    verify.validate_trace([copy, *trace], ALL_RECORDS, "opengl")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, ALL_RECORDS, "metal")
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace[1:], ALL_RECORDS, "opengl")
    trace[0]["parameters"]["vector_batch_stride"] = [1]
    with pytest.raises(RuntimeError):
        verify.validate_trace(trace, ALL_RECORDS, "opengl")
    trace[0]["parameters"]["vector_batch_stride"] = [0]
    gemm = next(
        index for index, record in enumerate(ALL_RECORDS) if record["family"] == "gemm"
    )
    trace[gemm]["parameters"]["params"]["lda"] += 1
    with pytest.raises(RuntimeError, match="launch differs"):
        verify.validate_trace(trace, ALL_RECORDS, "opengl")
    trace[gemm]["parameters"]["params"]["lda"] -= 1
    verify.validate_trace(trace, ALL_RECORDS, "opengl")
    # The accumulation must read the partitions its product wrote.
    accum = next(
        index
        for index, event in enumerate(trace)
        if event["entry"].startswith("steel_gemm_splitk_accum_")
    )
    trace[accum]["inputHashes"]["C_split"] = "other"
    with pytest.raises(RuntimeError, match="launch differs"):
        verify.validate_trace(trace, ALL_RECORDS, "opengl")
    trace[accum]["inputHashes"]["C_split"] = trace[accum - 1]["outputHash"]
    verify.validate_trace(trace, ALL_RECORDS, "opengl")
    # A split-K product without its accumulation is not the routed sequence.
    with pytest.raises(RuntimeError, match="routed entries"):
        verify.validate_trace(trace[:accum] + trace[accum + 1 :], ALL_RECORDS, "opengl")
    # Each family must cover its routes on its own.
    for family in ("gemv", "gemm", "splitk"):
        kept = [record for record in ALL_RECORDS if record["family"] != family] + [
            record for record in ALL_RECORDS if record["family"] == family
        ][:3]
        events = [event for record in kept for event in trace_events(record, "opengl")]
        with pytest.raises(RuntimeError, match="cover"):
            verify.validate_trace(events, kept, "opengl")


def test_ci_runs_the_matmul_route_on_every_quantized_host_platform():
    import re

    import yaml

    root = Path(__file__).resolve().parents[5]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["quantized-host"]
    names = [item.get("name") for item in job["steps"]]
    steps = dict(zip(names, job["steps"]))
    translate = steps["Translate floating-point matrix-vector packages"]
    gemm = steps["Translate steel GEMM packages"]
    splitk = steps["Translate split-K GEMM packages"]
    execute = steps["Execute floating-point matrix products through MLX"]
    assert names.index("Translate split-K GEMM packages") < names.index(
        "Execute floating-point matrix products through MLX"
    )
    for item in (translate, gemm, splitk, execute):
        assert "if" not in item and "continue-on-error" not in item
        assert "set -euo pipefail" in item["run"]
        assert "--packages .mlx-portable-quantized/base/packages" in item["run"]
    assert "portable_host.gemv_packages" in translate["run"]
    assert "portable_host.gemm_packages" in gemm["run"]
    assert "portable_host.gemm_splitk_packages" in splitk["run"]
    for item in (translate, gemm, splitk):
        assert "--verification" in item["run"] and "--jobs 3" in item["run"]
    assert "portable_host.verify_matmul" in execute["run"]
    assert "tee .mlx-portable-quantized/matmul.log" in execute["run"]
    assert execute["env"]["CROSTL_DIRECTX_FORCE_WARP"] == "1"
    assert (
        "test_portable_matmul.py" in steps["Validate quantized host contracts"]["run"]
    )
    order = [
        "Build adapted upstream MLX",
        "Translate floating-point matrix-vector packages",
        "Translate steel GEMM packages",
        "Execute floating-point matrix products through MLX",
        "Retain quantized execution evidence",
    ]
    assert [names.index(item) for item in order] == sorted(
        names.index(item) for item in order
    )
    deadlines = [
        int(value)
        for item in job["steps"]
        for value in re.findall(r"--timeout-seconds (\d+)", item.get("run", ""))
    ]
    assert sum(deadlines) + 900 < job["timeout-minutes"] * 60
