"""Floating-point dot-product host routing, launch contracts and guarded commits."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import dot_dispatch as dispatch
from demos.integrations.mlx.portable_host import dot_layout as layout
from demos.integrations.mlx.portable_host import dot_packages, gemm_layout, runtime
from demos.integrations.mlx.portable_host import verify_matmul as verify

HOST = Path(layout.__file__).parent
CTYPES = {
    "float32": ctypes.c_uint32,
    "float16": ctypes.c_uint16,
    "bfloat16": ctypes.c_uint16,
    "int32": ctypes.c_int32,
}
RECORDS = [
    record
    for record in (verify.case_record(case) for case in verify.CASES)
    if record["family"] == "dot"
]
ONE = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}


def test_signature_accepts_exactly_the_instantiated_entries():
    for dtype in layout.DTYPES:
        entry = layout.select(dtype)
        assert entry == f"dot_product_{dtype}_it32_tg512_sg16"
        assert layout.signature(entry) == {"dtype": dtype}
        assert layout.workgroup_size(entry) == (512, 1, 1)
    for entry in (
        "dot_product_complex64_it32_tg512_sg16",
        "dot_product_float32_it16_tg512_sg16",
        "dot_product_float32_it32_tg256_sg8",
        "dot_product_float32",
        None,
    ):
        with pytest.raises(ValueError):
            layout.signature(entry)
    with pytest.raises(ValueError):
        layout.select("int32")
    assert dot_packages.verification_entries() == [
        layout.select(dtype) for dtype in layout.DTYPES
    ]


def test_groups_and_passes_follow_dot_product_and_all_reduce_dispatch():
    # 512 threads of 32 items: one partial per 16,384 elements.
    assert [layout.groups(k) for k in (1, 16384, 16385, 40000)] == [1, 1, 2, 3]
    assert layout.reduction_passes(4096) == 1 and layout.reduction_passes(4097) == 2


def test_routes_mirror_matmul_eval_gpu():
    route = layout.route("float16", [1, 1001], [1001, 1], [1001, 1], [1, 1])
    assert route["entry"] == "dot_product_float16_it32_tg512_sg16"
    assert (route["K"], route["partials"], route["kernels"]) == (1001, 1, 3)
    family, _ = gemm_layout.matmul_route(
        "float16", [1, 1001], [1001, 1], [1001, 1], [1, 1]
    )
    assert family == "dot"
    # A strided vector is copied first and then takes the dot route.
    copied = layout.route("float32", [1, 33], [66, 2], [33, 1], [1, 1])
    assert copied["copies"] == {"a": True, "b": False}
    # Batched or wider products take the matrix-vector or GEMM routes.
    for a, b in (
        (([2, 1, 8], [8, 8, 1]), ([2, 8, 1], [8, 1, 1])),
        (([1, 8], [8, 1]), ([8, 2], [2, 1])),
    ):
        with pytest.raises(ValueError, match="dot route"):
            layout.route("float32", *a, *b)
        assert gemm_layout.matmul_route("float32", *a, *b)[0] != "dot"


def native_buffers(record, data=None):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    route, entry = record["route"], record["entry"]
    counts = {"a": route["K"], "b": route["K"], "output": route["partials"]}
    memory, buffers = {}, []
    for name in layout.HOST_PARAMETERS:
        dtype = layout.parameter_type(entry, name)
        values = (
            [route["K"]] if name == "n" else (data or {}).get(name, [0] * counts[name])
        )
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == "output"),
            )
        )
    execution = {
        "workgroupCount": [route["partials"], 1, 1],
        "workgroupSize": [512, 1, 1],
    }
    return buffers, memory, execution


def by_name(buffers):
    return {buffer.name.decode(): buffer for buffer in buffers}


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record):
    buffers, memory, execution = native_buffers(record)
    counts, parameters = layout.read_storage(record["entry"], by_name(buffers))
    assert parameters == {"n": record["route"]["K"]}
    metadata = layout.plan(
        record["entry"], counts, parameters, record["route"]["partials"], execution
    )
    assert metadata == {
        "K": record["route"]["K"],
        "partials": record["route"]["partials"],
    }


@pytest.mark.parametrize(
    "fault", ("groups", "size", "threads", "output", "short", "length", "geometry")
)
def test_plan_rejects_launches_that_differ_from_mlx(fault):
    record = next(item for item in RECORDS if item["route"]["partials"] > 1)
    buffers, memory, execution = native_buffers(record)
    counts, parameters = layout.read_storage(record["entry"], by_name(buffers))
    threads = record["route"]["partials"]
    if fault == "groups":
        execution["workgroupCount"] = [threads - 1, 1, 1]
    elif fault == "size":
        execution["workgroupSize"] = [256, 1, 1]
    elif fault == "threads":
        threads += 1
    elif fault == "output":
        counts["output"] += 1
    elif fault == "short":
        counts["b"] -= 1
    elif fault == "length":
        parameters["n"] = 0
    else:
        execution = {"workgroupCount": execution["workgroupCount"]}
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, threads, execution)


@pytest.mark.parametrize("fault", ("dtype", "direction", "missing", "alias", "count"))
def test_storage_rejects_native_identity_faults(fault):
    record = RECORDS[0]
    buffers, memory, execution = native_buffers(record)
    supplied = by_name(buffers)
    if fault == "dtype":
        supplied["a"].dtype = b"float16"
    elif fault == "direction":
        supplied["b"].output = 1
    elif fault == "missing":
        del supplied["n"]
    elif fault == "alias":
        supplied["output"].data = supplied["a"].data
    else:
        supplied["n"].count = 2
    with pytest.raises(ValueError):
        layout.read_storage(record["entry"], supplied)


def physical_layout(target, dtype):
    if dtype in ("float16", "bfloat16") and target != "metal":
        physical = "uint16" if target == "directx" else "float32"
    else:
        physical = dtype
    strides = {"uint16": 2, "float16": 2, "bfloat16": 2, "float32": 4, "int32": 4}
    scalar = {"elementType": physical, "elementStrideBytes": strides[physical]}
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
@pytest.mark.parametrize("name", ("dot-product-tail", "dot-product-groups"))
def test_dispatch_commits_the_partials_only_after_validation(
    tmp_path, monkeypatch, target, fault, name
):
    record = next(item for item in RECORDS if item["name"] == name)
    entry, dtype = record["entry"], record["dtype"]
    buffers, memory, execution = native_buffers(record)
    descriptor = {"artifact": {}, "bindings": []}
    for parameter in layout.PARAMETERS:
        if fault == "unbound" and parameter == "a":
            continue
        kind = layout.parameter_type(entry, parameter)
        descriptor["bindings"].append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": (
                    "read_write"
                    if parameter == "output" and fault != "readonly"
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
        dots=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, kwargs))
        return outputs

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_output"]
        count = len(memory["output"])
        value["values"][:count] = dispatch.encode_binding(
            "float32",
            [ONE["float32"]] * count,
            target,
            physical_layout(target, "float32"),
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
        host, entry, supplied, len(buffers), len(memory["output"]), launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["output"])
        return
    call()
    assert list(memory["output"]) == [ONE["float32"]] * len(memory["output"])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == {"binding_a", "binding_b", "binding_n"}
    for parameter in ("a", "b"):
        assert len(inputs[f"binding_{parameter}"]["values"]) == (
            record["route"]["K"] + dispatch.GUARD_COUNT
        )
    assert inputs["binding_n"]["values"] == [record["route"]["K"]]
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry and event["target"] == target
    assert event["dotMetadata"] == {
        "K": record["route"]["K"],
        "partials": record["route"]["partials"],
    }
    assert event["boundParameters"] == sorted(layout.PARAMETERS)
    assert event["outputValues"] == [ONE["float32"]] * record["route"]["partials"]


def test_package_configs_scope_target_options(tmp_path):
    entry = layout.select("bfloat16")
    for target in ("metal", "directx", "opengl"):
        config = dot_packages.project_config(tmp_path, "out", target, entry)
        assert not config.index_range_assertions
        options = config.source_options["metal"]["target_options"]
        assert options["directx"] == {
            "software_subgroup_width": 32,
            "relative_wave_shuffle_out_of_range": "self",
        }
        assert options["opengl"] == {"software_subgroup_width": 32}
        assert config.entry_points == {layout.SOURCE: (entry,)}
        (rule,) = config.entry_workgroup_size_rules[layout.SOURCE].values()
        assert tuple(int(value) for value in rule) == (512, 1, 1)
    cache = dot_packages.DotPackageCache(tmp_path, tmp_path / "cache", "opengl")
    assert cache.SOURCE == layout.SOURCE and cache.workgroup_size(entry) == (512, 1, 1)
    with pytest.raises(ValueError, match="dot-product target"):
        dot_packages.DotPackageCache(tmp_path, tmp_path, "vulkan")


def test_runtime_routes_dot_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = None
    host.gemvs = host.dots = None
    entry = layout.select("float32")
    assert host._entry_available(entry.encode()) == 0
    host.dots = object()
    assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(b"dot_product_int32_it32_tg512_sg16") == 0
    calls = []
    monkeypatch.setattr(
        runtime.dot_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(entry, [], 4, 1, launch=None)
    assert calls == [entry]


def test_backend_sums_the_partials_and_copies_the_total():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    start = backend.index("void dispatch_dot_product(")
    body = backend[start : backend.index("\n}\n", start)]
    for text in (
        '"_it32_tg512_sg16"',
        "(uint64_t(K) + items_per_thread - 1) / items_per_thread",
        "(threads + thread_group_size - 1) / thread_group_size",
        "array partials({static_cast<int>(blocks)}, float32, nullptr, {});",
        '{"output", "float32", partials.data<void>(), blocks, 1}',
        "{uint32_t(blocks), 1, 1}, {uint32_t(thread_group_size), 1, 1}",
        "partials.size() <= 4096",
        'dispatch_all_reduce(partials, total, "sum", "float32");',
        "dispatch_cast({total}, out);",
    ):
        assert text in body, text
    for name in layout.HOST_PARAMETERS:
        assert f'{{"{name}", ' in body, name
    assert "dispatch_dot_product(a, b, out, K);" in backend
