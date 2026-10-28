"""Split-K steel GEMM host routing, launch contracts and guarded commits."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import gemm_layout
from demos.integrations.mlx.portable_host import gemm_splitk_dispatch as dispatch
from demos.integrations.mlx.portable_host import gemm_splitk_layout as layout
from demos.integrations.mlx.portable_host import gemm_splitk_packages, runtime
from demos.integrations.mlx.portable_host import verify_matmul as verify
from demos.integrations.mlx.tests.host.test_portable_matmul import (
    CTYPES,
    bound_operands,
    physical_layout,
    span,
)
from demos.integrations.mlx.tests.kernels import gemm_splitk_model as model
from demos.integrations.mlx.tests.kernels import test_current_gemm_splitk as gate

HOST = Path(layout.__file__).parent
RECORDS = [
    record
    for record in (verify.case_record(case) for case in verify.CASES)
    if record["family"] == "splitk"
]
BY_NAME = {record["name"]: record for record in RECORDS}
ONE = {"float32": 0x3F800000, "float16": 0x3C00, "bfloat16": 0x3F80}


def native_binding(record, kind="splitk"):
    """What backend.cpp binds for one kernel of a verification case."""
    route = record["route"]
    m, n, partitions = route["M"], route["N"], route["partitions"]
    if kind == "accum":
        counts = {"C_split": partitions * m * n, "D": m * n}
        execution = {"workgroupCount": [n, m, 1], "workgroupSize": [1, 1, 1]}
        return counts, dict(route["accumParams"]), execution
    operands = bound_operands(record)
    counts = {
        "A": span(*operands["a"]),
        "B": span(*operands["b"]),
        "C": partitions * m * n,
    }
    params = route["params"]
    execution = {
        "workgroupCount": [params["tiles_n"], params["tiles_m"], partitions],
        "workgroupSize": [32, 2, 2],
    }
    return counts, {"params": dict(params)}, execution


def entry_of(record, kind):
    return record["entry"] if kind == "splitk" else record["route"]["accum"]


def native_buffers(record, kind="splitk"):
    """ctypes buffers in backend.cpp's binding order, with backing storage."""
    counts, parameters, execution = native_binding(record, kind)
    entry, memory, buffers = entry_of(record, kind), {}, []
    output = layout.OUTPUTS[kind]
    for name in layout.PARAMETERS[kind]:
        dtype = layout.parameter_type(entry, name)
        if name in counts:
            values = [0] * counts[name]
        elif name == "params":
            values = [parameters[name][field] for field in layout.PARAMS_FIELDS]
        else:
            values = [parameters[name]]
        memory[name] = (CTYPES[dtype] * len(values))(*values)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(values),
                int(name == output),
            )
        )
    return buffers, memory, execution


def test_signature_accepts_exactly_the_host_entries():
    accepted = {
        entry.name
        for entry in model.entries()
        if entry.dtype in layout.DTYPES and entry.oname == "float32"
    }
    assert len(accepted) == 3 * 4 * 4 * 4
    for name in accepted:
        info = layout.signature(name)
        assert info["kind"] == "splitk" and info["model"].name == name
        assert layout.workgroup_size(name) == (32, 2, 2)
    for oname in layout.DTYPES:
        name = f"steel_gemm_splitk_accum_{oname}_float32"
        assert layout.signature(name)["kind"] == "accum"
        assert layout.workgroup_size(name) == (1, 1, 1)
    for name in (
        "steel_gemm_splitk_accum_float32_float32_axbpy",
        "steel_gemm_splitk_accum_complex64_complex64",
        "steel_gemm_splitk_nn_complex64_complex64_bm16_bn16_bk16_wm2_wn2"
        "_MN_taligned_K_taligned",
        "steel_gemm_splitk_nn_float32_float32_bm64_bn16_bk16_wm2_wn2"
        "_MN_taligned_K_taligned",
        "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2",
        None,
    ):
        with pytest.raises(ValueError):
            layout.signature(name)


@pytest.mark.parametrize(
    "m, n, k, plan",
    ((16, 16, 256, (16, 1, 16)), (33, 35, 300, (4, 4, 64)), (24, 24, 200, (16, 0, 0))),
)
def test_partition_plan_follows_steel_gemm_splitk_axpby(m, n, k, plan):
    assert layout.partition_plan(m, n, k) == plan
    assert layout.splitk_params(m, n, k, k, n)["split_k_partitions"] == plan[0]


def test_routes_mirror_matmul_eval_gpu():
    routes = {name: record["route"] for name, record in BY_NAME.items()}
    whole = routes["splitk-nn-whole"]
    assert whole["entry"] == (
        "steel_gemm_splitk_nn_float32_float32_bm16_bn16_bk16_wm2_wn2"
        "_MN_taligned_K_taligned"
    )
    assert whole["accum"] == "steel_gemm_splitk_accum_float32_float32"
    assert whole["accumParams"] == {
        "k_partitions": 16,
        "partition_stride": 256,
        "ldd": 16,
    }
    tt = routes["splitk-tt-aligned"]["params"]
    assert (tt["lda"], tt["ldb"], tt["tiles_m"], tt["tiles_n"]) == (64, 512, 2, 3)
    folded = routes["splitk-batch-folded-into-rows"]
    assert (folded["M"], folded["batches"]) == (16, 1)
    copied = routes["splitk-copied-operand"]
    assert copied["copies"] == {"a": True, "b": False}
    assert copied["params"]["lda"] == 208
    assert copied["params"]["split_k_partitions"] == 16
    assert copied["params"]["split_k_partition_size"] == 0
    with pytest.raises(ValueError, match="fused"):
        layout.route("float32", [64, 64], [64, 1], [64, 64], [64, 1])
    with pytest.raises(ValueError, match="gemv"):
        layout.route("float32", [1, 256], [256, 1], [256, 16], [16, 1])
    with pytest.raises(ValueError, match="split-K route"):
        gemm_layout.route("float32", [16, 256], [256, 1], [256, 16], [16, 1])
    family, route = gemm_layout.matmul_route(
        "float32", [16, 256], [256, 1], [256, 16], [16, 1]
    )
    assert family == "splitk" and route == whole | {"copies": route["copies"]}


@pytest.mark.parametrize("kind", ("splitk", "accum"))
@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["name"])
def test_plan_accepts_each_verification_launch(record, kind):
    counts, parameters, execution = native_binding(record, kind)
    output = layout.OUTPUTS[kind]
    metadata = layout.plan(
        entry_of(record, kind), counts, parameters, counts[output], execution
    )
    assert metadata["kind"] == kind
    assert metadata["partitions"] == record["route"]["partitions"]
    assert (metadata["M"], metadata["N"]) == (
        record["route"]["M"],
        record["route"]["N"],
    )


def mutate(fault, counts, parameters, execution):
    params = parameters.get("params")
    if fault == "tiles":
        execution["workgroupCount"][0] += 1
    elif fault == "workgroup":
        execution["workgroupSize"] = [32, 2, 1]
    elif fault == "partitions":
        execution["workgroupCount"][2] += 1
    elif fault == "threads":
        counts["C"] += 1
    elif fault == "a-short":
        counts["A"] -= 1
    elif fault == "b-short":
        counts["B"] -= 1
    elif fault == "ldc":
        params["ldc"] += 1
    elif fault == "lda":
        params["lda"] = 1
    elif fault == "swizzle":
        params["swizzle_log"] = 1
    elif fault == "iterations":
        params["gemm_k_iterations_aligned"] += 1
    elif fault == "partition-size":
        params["split_k_partition_size"] += 16
    elif fault == "partition-stride":
        params["split_k_partition_stride"] += 1
    elif fault == "dimension":
        params["K"] += 16
    elif fault == "unsplit":
        # A product MLX would send to steel_gemm_fused.
        params.update(K=8, lda=8)
    elif fault == "bound":
        counts["A"] = layout.MAX_ELEMENTS + 1
        params["lda"] = layout.MAX_ELEMENTS + 1
    elif fault == "integer":
        params["M"] = float(params["M"])


FAULTS = (
    "tiles",
    "workgroup",
    "partitions",
    "threads",
    "a-short",
    "b-short",
    "ldc",
    "lda",
    "swizzle",
    "iterations",
    "partition-size",
    "partition-stride",
    "dimension",
    "unsplit",
    "bound",
    "integer",
)


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("name", ("splitk-nt-partial", "splitk-tt-aligned"))
def test_plan_rejects_launches_that_differ_from_mlx(fault, name):
    record = BY_NAME[name]
    counts, parameters, execution = native_binding(record)
    mutate(fault, counts, parameters, execution)
    with pytest.raises(ValueError):
        layout.plan(record["entry"], counts, parameters, counts["C"], execution)


@pytest.mark.parametrize(
    "fault", ("partitions", "stride", "ldd", "threads", "split", "grid", "workgroup")
)
def test_accum_plan_rejects_launches_that_differ_from_mlx(fault):
    record = BY_NAME["splitk-mixed-tile"]
    counts, parameters, execution = native_binding(record, "accum")
    threads = counts["D"]
    if fault == "partitions":
        parameters["k_partitions"] = 33
        counts["C_split"] = 33 * counts["D"]
    elif fault == "stride":
        parameters["partition_stride"] += 1
    elif fault == "ldd":
        parameters["ldd"] += 1
    elif fault == "threads":
        threads += 1
    elif fault == "split":
        counts["C_split"] -= 1
    elif fault == "grid":
        execution["workgroupCount"][2] = 2
    else:
        execution["workgroupSize"] = [2, 1, 1]
    with pytest.raises(ValueError):
        layout.plan(record["route"]["accum"], counts, parameters, threads, execution)


def test_plan_rejects_entries_mlx_would_not_select():
    record = BY_NAME["splitk-tt-aligned"]
    counts, parameters, execution = native_binding(record)
    for other in (
        record["entry"].replace("MN_taligned", "MN_naligned"),
        record["entry"].replace("_tt_", "_nn_"),
        record["entry"].replace("K_taligned", "K_naligned"),
        record["entry"].replace("bm32_bn32", "bm16_bn32"),
    ):
        # The plan derives the transposition's leading-dimension bounds first.
        with pytest.raises(ValueError, match="routing|differ"):
            layout.plan(other, counts, parameters, counts["C"], execution)


@pytest.mark.parametrize(
    "fault", ("dtype", "direction", "empty", "params", "names", "scalar", "alias")
)
def test_storage_rejects_native_identity_faults(fault):
    kind = "accum" if fault in ("scalar", "alias") else "splitk"
    record = BY_NAME["splitk-nt-partial"]
    buffers, memory, _ = native_buffers(record, kind)
    supplied = {buffer.name.decode(): buffer for buffer in buffers}
    if fault == "dtype":
        supplied["C"].dtype = b"float16"
    elif fault == "direction":
        supplied["B"].output = 1
    elif fault == "empty":
        supplied["A"].count = 0
    elif fault == "params":
        supplied["params"].count = 14
    elif fault == "scalar":
        supplied["ldd"].count = 2
    elif fault == "alias":
        supplied["D"].data = supplied["C_split"].data
    else:
        supplied.pop("params")
    with pytest.raises(ValueError):
        layout.read_storage(entry_of(record, kind), supplied)
    assert memory


def test_storage_reads_params_and_scalars():
    record = BY_NAME["splitk-tn-wide-tile"]
    for kind in ("splitk", "accum"):
        buffers, memory, _ = native_buffers(record, kind)
        supplied = {buffer.name.decode(): buffer for buffer in buffers}
        counts, parameters = layout.read_storage(entry_of(record, kind), supplied)
        if kind == "splitk":
            assert parameters == {"params": record["route"]["params"]}
            assert counts["C"] == 4 * 48 * 64
        else:
            assert parameters == record["route"]["accumParams"]
            assert counts == {"C_split": 4 * 48 * 64, "D": 48 * 64}
        assert memory


def params_layout(record=True):
    """The reflected GEMMSpiltKParams layout: a record or an int32 array."""
    members = [
        {"name": name, "offsetBytes": 4 * index}
        for index, name in enumerate(layout.PARAMS_FIELDS)
    ]
    if record:
        from crosstl.project.storage_record_layout import storage_record_layout

        return storage_record_layout(
            "GEMMSpiltKParams",
            [("int", name) for name in layout.PARAMS_FIELDS],
            storage_layout="metal-buffer",
        )
    return {
        "elementType": "int32",
        "elementStrideBytes": 52,
        "componentCount": 13,
        "structMembers": members,
    }


def test_dispatch_packs_both_params_reflections():
    from crosstl.project import pack_storage_records

    params = BY_NAME["splitk-nt-partial"]["route"]["params"]
    binding = {"scalarLayout": params_layout()}
    value = dispatch._params(binding, params)
    assert value["dtype"] == "uint32" and len(value["values"]) == 13
    assert value["values"] == pack_storage_records(binding["scalarLayout"], [params])
    array = dispatch._params({"scalarLayout": params_layout(False)}, params)
    assert array == {"dtype": "int32", "shape": [13], "values": list(params.values())}
    for change in ({"elementStrideBytes": 56}, {"componentCount": 12}):
        with pytest.raises(ValueError, match="record differs"):
            dispatch._params({"scalarLayout": params_layout(False) | change}, params)


def descriptor_for(entry, kind, target, fault):
    bindings = []
    for parameter in layout.PARAMETERS[kind]:
        if fault == "unbound" and parameter == layout.PARAMETERS[kind][0]:
            continue
        dtype = layout.parameter_type(entry, parameter)
        scalar = (
            params_layout()
            if parameter == "params"
            else (
                physical_layout(target, dtype)
                if dtype != "int32"
                else {"elementType": "int32", "elementStrideBytes": 4}
            )
        )
        output = parameter == layout.OUTPUTS[kind]
        bindings.append(
            {
                "name": f"binding_{parameter}",
                "kind": "buffer",
                "access": "read_write" if output and fault != "readonly" else "read",
                "provenance": {"sourceResource": {"parameter": parameter}},
                "scalarLayout": scalar,
            }
        )
    if fault == "duplicate":
        bindings.append(dict(bindings[0], name="again"))
    return {"artifact": {}, "bindings": bindings}


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "missing", "shape", "trace", "unbound", "readonly", "duplicate"),
)
@pytest.mark.parametrize("kind", ("splitk", "accum"))
def test_dispatch_commits_the_output_only_after_validation(
    tmp_path, monkeypatch, target, fault, kind
):
    record = BY_NAME["splitk-nt-partial"]
    entry = entry_of(record, kind)
    output = layout.OUTPUTS[kind]
    dtype = layout.parameter_type(entry, output)
    buffers, memory, execution = native_buffers(record, kind)
    descriptor = descriptor_for(entry, kind, target, fault)
    host = SimpleNamespace(
        target=target,
        splitk_gemms=SimpleNamespace(get=lambda name: (descriptor, tmp_path)),
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
        value = outputs[f"binding_{output}"]
        count = len(memory[output])
        value["values"][:count] = dispatch.encode_binding(
            dtype,
            [ONE[dtype]] * count,
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
        host, entry, supplied, len(buffers), len(memory[output]), launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory[output])
        return
    call()
    assert list(memory[output]) == [ONE[dtype]] * len(memory[output])
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == set(inputs) - {f"binding_{output}"}
    guarded = ("A", "B") if kind == "splitk" else ("C_split",)
    for parameter in guarded:
        assert len(inputs[f"binding_{parameter}"]["values"]) == (
            len(memory[parameter]) + dispatch.GUARD_COUNT
        )
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == entry
    assert event["splitkMetadata"]["partitions"] == 4
    assert set(event["inputHashes"]) == set(guarded)
    assert event["boundParameters"] == sorted(layout.PARAMETERS[kind])
    if kind == "accum":
        assert event["parameters"] == record["route"]["accumParams"]
        for name, value in record["route"]["accumParams"].items():
            assert inputs[f"binding_{name}"]["values"] == [value]


@pytest.mark.parametrize("kind", ("splitk", "accum"))
def test_package_configs_match_the_splitk_gate(tmp_path, kind):
    record = BY_NAME["splitk-nt-partial"]
    entry = entry_of(record, kind)
    workgroup = layout.workgroup_size(entry)
    for target in ("metal", "directx", "opengl"):
        config = gemm_splitk_packages.project_config(tmp_path, "out", target, entry)
        expected = gate._config(tmp_path, "out", target, entry, workgroup)
        assert config.entry_points == expected.entry_points == {layout.SOURCE: (entry,)}
        assert config.source_options == expected.source_options
        assert config.workgroup_size_rules == expected.workgroup_size_rules
        assert config.workgroup_access_assertions == (
            expected.workgroup_access_assertions
        )
        assert config.index_range_assertions == expected.index_range_assertions
        assert bool(config.index_range_assertions) == (target == "opengl")
    cache = gemm_splitk_packages.SplitKPackageCache(tmp_path, tmp_path / "c", "metal")
    assert cache.SOURCE == layout.SOURCE == gate.SOURCE
    assert cache.source_entry(entry) == entry
    with pytest.raises(ValueError, match="gemm target"):
        gemm_splitk_packages.SplitKPackageCache(tmp_path, tmp_path, "vulkan")


def test_package_builds_cover_the_verification_entries(tmp_path, monkeypatch):
    import concurrent.futures

    from demos.integrations.mlx.portable_host import matmul_packages

    entries = gemm_splitk_packages.verification_entries()
    assert set(entries) == {
        name
        for record in RECORDS
        for name in (record["entry"], record["route"]["accum"])
    }
    assert len(set(entries)) == 10
    calls = []

    def build(root, directory, target, entry):
        calls.append((Path(directory), target, entry))
        return {"entry": entry}

    monkeypatch.setattr(
        matmul_packages, "ProcessPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(gemm_splitk_packages, "_build_one", build)
    records = gemm_splitk_packages.build_entries(
        tmp_path, tmp_path / "packages", "opengl", entries, 3
    )
    assert [record["entry"] for record in records] == sorted(set(entries))
    assert {call[:2] for call in calls} == {
        (
            (tmp_path / "packages" / gemm_splitk_packages.CACHE_DIRECTORY).resolve(),
            "opengl",
        )
    }


def test_runtime_routes_splitk_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = host.gemvs = host.gemms = None
    host.splitk_gemms = None
    record = RECORDS[0]
    for entry in (record["entry"], record["route"]["accum"]):
        assert host._entry_available(entry.encode()) == 0
    host.splitk_gemms = object()
    for entry in (record["entry"], record["route"]["accum"]):
        assert host._entry_available(entry.encode()) == 1
    assert host._entry_available(record["entry"].replace("bm16", "bm64").encode()) == 0
    calls = []
    monkeypatch.setattr(
        runtime.gemm_splitk_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(record["route"]["accum"], [], 5, 1, launch=None)
    assert calls == [record["route"]["accum"]]


def test_backend_mirrors_steel_gemm_splitk_axpby():
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    for text in (
        "void dispatch_steel_gemm_splitk(",
        "const int bm = M < 40 ? 16 : 32;",
        "const int bn = N < 40 ? 16 : 32;",
        "std::min(std::max(2, next_power_of_2(_tk / (_tm * _tn))), 32);",
        "const int gemm_k_iterations = (K / bk) / split_k_partitions;",
        "array C_split({split_k_partitions, M, N}, float32, nullptr, {});",
        '"_float32_bm"',
        '"aligned_K_"',
        'std::string("steel_gemm_splitk_accum_") + storage + "_float32"',
        "{uint32_t(tn), uint32_t(tm), uint32_t(split_k_partitions)}",
        "{{uint32_t(N), uint32_t(M), 1}, {1, 1, 1}}",
    ):
        assert text in backend, text
    for kind in ("splitk", "accum"):
        for name in layout.PARAMETERS[kind]:
            assert f'{{"{name}", ' in backend, name
