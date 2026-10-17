"""Validate checked whole-array and row reduction packages and cache identity."""

import copy
import hashlib
import json

import pytest

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import reduction_cache as packages
from demos.integrations.mlx.portable_host import reduction_packages, runtime
from demos.integrations.mlx.portable_host.reduction_layout import stage
from demos.integrations.mlx.portable_host.verify_rows import verify_artifacts

ENTRY = "all_reduce_sumfloat32"


def launch(count=129):
    plan = stage(count)
    return {key: plan[key] for key in ("workgroupCount", "workgroupSize")}


@pytest.mark.parametrize(
    "fault",
    [
        "entry",
        "entry-list",
        "zero",
        "overflow",
        "bool",
        "float",
        "null-launch",
        "bool-launch",
        "float-launch",
        "missing-launch",
        "extra-launch",
        "rows",
        "width",
    ],
)
def test_invalid_contract_precedes_source_access(tmp_path, monkeypatch, fault):
    entry, count, execution = ENTRY, 129, launch()
    if fault == "entry":
        entry = "row_reduce_simple_sumfloat32"
    elif fault == "entry-list":
        entry = []
    elif fault in {"zero", "overflow", "bool", "float"}:
        count = {"zero": 0, "overflow": 2**31, "bool": True, "float": 129.0}[fault]
    elif fault == "null-launch":
        execution = None
    elif fault in {"bool-launch", "float-launch"}:
        execution["workgroupCount"][0] = True if fault == "bool-launch" else 1.0
    elif fault == "missing-launch":
        execution.pop("workgroupSize")
    elif fault == "extra-launch":
        execution["unused"] = []
    elif fault == "rows":
        execution["workgroupCount"][1] += 1
    else:
        execution["workgroupSize"][0] += 32
    cache = packages.ReductionPackageCache(tmp_path, tmp_path / "cache", "opengl")
    monkeypatch.setattr(
        cache,
        "_require_source",
        lambda: pytest.fail("Invalid contract accessed source"),
    )
    with pytest.raises(ValueError):
        cache.get(entry, count, execution)
    assert not cache.directory.exists()


@pytest.fixture(params=("metal", "opengl", "directx"))
def packaged(tmp_path, monkeypatch, request):
    root = tmp_path / "source"
    source = root / packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("""template<typename T> kernel void reduce_values(
device const T* in [[buffer(0)]], device T* out [[buffer(1)]],
constant ulong& in_size [[buffer(2)]], constant ulong& row_size [[buffer(3)]],
uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  T value = simd_sum(in[0]);
  if (tid == 0) out[group.y] = value + T(in_size - row_size);
}
template [[host_name("all_reduce_sumfloat32")]] [[kernel]]
decltype(reduce_values<float>) reduce_values<float>;
""")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(packages, "SOURCE_SHA256", digest)
    monkeypatch.setattr(packages, "require_revision", lambda path: None)
    monkeypatch.setattr(packages.subprocess, "check_output", lambda *_args, **_kw: "")
    cache = packages.ReductionPackageCache(root, tmp_path / "cache", request.param)
    descriptor, package = cache.get(ENTRY, 129, launch())
    return cache, descriptor, package


def test_contract_detaches_and_cache_reloads(packaged, monkeypatch):
    cache, descriptor, package = packaged
    execution = launch()
    contract = packages.dispatch_contract(ENTRY, 129, execution)
    execution["workgroupSize"][0] = 32
    assert contract["execution"] == launch()
    monkeypatch.setattr(
        cache, "_build", lambda *_args: pytest.fail("Cache hit rebuilt shader")
    )
    assert cache.get(ENTRY, 129, launch()) == (descriptor, package)
    assert cache.load(ENTRY, 129, launch()) == (descriptor, package)
    with pytest.raises(FileNotFoundError):
        cache.load(ENTRY, 130, launch(130))


def test_equal_width_contracts_have_distinct_packages(packaged):
    cache, descriptor, package = packaged
    other, other_package = cache.get(ENTRY, 130, launch(130))
    assert launch(129) == launch(130)
    assert package != other_package and descriptor != other
    assert cache.load(ENTRY, 129, launch()) == (descriptor, package)
    assert cache.load(ENTRY, 130, launch(130)) == (other, other_package)


@pytest.mark.parametrize(
    "fault", ("identity", "metadata", "region", "artifact", "source", "implementation")
)
def test_cache_rejects_changed_identity_or_content(packaged, monkeypatch, fault):
    cache, descriptor, package = packaged
    index = package.parent / "index.json"
    record = json.loads(index.read_text())
    if fault == "source":
        source = cache.root / packages.SOURCE
        source.write_text(source.read_text() + "\n// changed\n")
    elif fault == "implementation":
        monkeypatch.setattr(
            packages, "translation_implementation_hash", lambda: "0" * 64
        )
    elif fault == "artifact":
        (package / descriptor["artifact"]["packagePath"]).write_text("// changed\n")
    else:
        if fault == "identity":
            record["identity"]["contract"]["count"] += 1
        elif fault == "metadata":
            binding = next(
                binding
                for binding in record["descriptor"]["bindings"]
                if "metadataPrecondition" in binding.get("provenance", {})
            )
            binding["provenance"]["metadataPrecondition"]["maximum"] += 1
        else:
            record["descriptor"].setdefault("provenance", {})["dispatchRegion"] = {}
        index.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.load(ENTRY, 129, launch())


def test_cache_does_not_publish_changed_source(packaged, monkeypatch):
    cache, _, _ = packaged
    original = cache._build
    before = set(cache.directory.iterdir())

    def changed(output, identity):
        original(output, identity)
        source = cache.root / packages.SOURCE
        source.write_text(source.read_text() + "\n// changed during packaging\n")

    monkeypatch.setattr(cache, "_build", changed)
    with pytest.raises(ValueError, match="source hash"):
        cache.get(ENTRY, 130, launch(130))
    assert set(cache.directory.iterdir()) == before


@pytest.mark.parametrize("fault", (None, "in_size", "row_size", "launch"))
def test_package_enforces_checked_dispatch_inputs(packaged, fault):
    cache, descriptor, package = packaged
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        words = {"in": [1.0] * 129, "out": [0.0], "in_size": [129], "row_size": [129]}[
            parameter
        ]
        if parameter == fault:
            words = [130]
        packet = dict(
            dtype=binding["scalarLayout"]["elementType"],
            shape=[len(words)],
            values=words,
        )
        inputs[binding["name"]] = packet
        if parameter == "out":
            outputs[binding["name"]] = copy.deepcopy(packet)
    execution = launch()
    if fault == "launch":
        execution["workgroupCount"][0] += 1
    if fault is None:
        request = build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, execution
        )
        assert request.execution_plan.dispatch.workgroup_size == tuple(
            launch()["workgroupSize"]
        )
        assert not request.execution_plan.diagnostics
    elif fault == "launch" and cache.target == "metal":
        # Metal retains the original source entry; its launch is checked by the cache.
        with pytest.raises(ValueError):
            cache.get(ENTRY, 129, execution)
    else:
        with pytest.raises(ValueError):
            build_native_loader_dispatch_request(
                descriptor, package, inputs, outputs, execution
            )


@pytest.fixture
def catalog(packaged, tmp_path, monkeypatch):
    cache, _, _ = packaged
    monkeypatch.setattr(reduction_packages, "require_revision", lambda root: None)
    directory = tmp_path / "catalog"
    index = reduction_packages.build_packages(
        cache.root,
        directory,
        cache.target,
        widths=(32, 64),
        entries=(ENTRY,),
        on_demand=True,
    )
    assert "descriptors" not in index
    assert not (directory / "variants").exists()
    assert reduction_packages.load_index(directory, cache.target) == index
    return cache.root, directory, index


@pytest.mark.parametrize(
    "fault", ("revision", "sourceHash", "dispatchMode", "family", "descriptors")
)
def test_catalog_rejects_changed_identity(catalog, fault):
    root, directory, index = catalog
    index[fault] = {} if fault == "descriptors" else "changed"
    (directory / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        reduction_packages.load_index(directory, index["target"])


def make_host(catalog, tmp_path, monkeypatch, *, include_source=True, duplicate=False):
    from types import SimpleNamespace

    root, directory, index = catalog
    base = tmp_path / "base"
    base.mkdir()
    (base / "index.json").write_text(
        json.dumps(
            {
                "target": index["target"],
                "descriptors": dict.fromkeys(runtime.ENTRIES, {}),
            }
        )
    )
    for name in (
        "MetalRuntimeParityAdapter",
        "OpenGLRuntimeParityAdapter",
        "DirectXRuntimeParityAdapter",
        "OpenGLComputeRuntime",
        "DirectXComputeRuntime",
        "RuntimeParityExecutor",
    ):
        monkeypatch.setattr(runtime, name, lambda *args, **kwargs: SimpleNamespace())
    return runtime.HostRuntime(
        base,
        tmp_path / "trace.jsonl",
        mlx_root=root if include_source else None,
        reductions=[directory, directory] if duplicate else directory,
    )


@pytest.mark.parametrize("fault", ("missing-source", "duplicate"))
def test_catalog_host_registration_rejects_invalid_configuration(
    catalog, tmp_path, monkeypatch, fault
):
    with pytest.raises(ValueError, match="pinned MLX source|duplicate variants"):
        make_host(
            catalog,
            tmp_path,
            monkeypatch,
            include_source=fault != "missing-source",
            duplicate=fault == "duplicate",
        )


def dispatch_buffers(count):
    import ctypes

    plan = stage(count)
    rows = plan["workgroupCount"][1]
    memory = [
        (ctypes.c_float * count)(*range(count)),
        (ctypes.c_float * rows)(),
        ctypes.c_uint64(count),
        ctypes.c_uint64(plan["rowSize"]),
    ]
    buffers = (runtime.Buffer * 4)(
        *[
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(data),
                length,
                int(name == "out"),
            )
            for name, dtype, length, data in zip(
                ("in", "out", "in_size", "row_size"),
                ("float32", "float32", "uint64", "uint64"),
                (count, rows, 1, 1),
                memory,
            )
        ]
    )
    geometry = runtime.Launch(
        tuple(plan["workgroupCount"]), tuple(plan["workgroupSize"])
    )
    return memory, buffers, geometry


@pytest.mark.parametrize(
    "fault", ("size", "row", "dtype", "input-count", "output-count", "null")
)
def test_host_validates_buffers_before_building_checked_packages(
    catalog, tmp_path, monkeypatch, fault
):
    host = make_host(catalog, tmp_path, monkeypatch)
    cache = next(iter(host.checked_reductions.values()))
    monkeypatch.setattr(
        cache, "get", lambda *args: pytest.fail("Invalid buffers reached translation")
    )
    memory, buffers, geometry = dispatch_buffers(129)
    if fault in {"size", "row"}:
        memory[2 if fault == "size" else 3].value += 1
    elif fault == "dtype":
        buffers[0].dtype = b"uint32"
    elif fault.endswith("count"):
        buffers[0 if fault == "input-count" else 1].count += 1
    else:
        buffers[0].data = None
    with pytest.raises(ValueError):
        host.dispatch(ENTRY, buffers, 4, 129, launch=geometry)
    assert not cache.directory.exists()
    assert not host.trace.exists()


def test_host_selects_and_verifies_exact_count_packages(catalog, tmp_path, monkeypatch):
    root, directory, index = catalog
    host = make_host(catalog, tmp_path, monkeypatch)
    calls = []

    # Preserve the requested output shape and guards without asserting native execution.
    from crosstl.project.runtime_verification import RuntimeExecutorResult

    def execute(request):
        calls.append(request)
        names = {value.name for value in request.fixture.expected_outputs}
        outputs = {
            value.name: value.to_json()
            for value in request.fixture.inputs
            if value.name in names
        }
        return RuntimeExecutorResult(outputs=outputs, details={})

    host.executor.run = execute
    for count in (129, 130):
        memory, buffers, geometry = dispatch_buffers(count)
        host.dispatch(ENTRY, buffers, 4, count, launch=geometry)
    events = [json.loads(line) for line in host.trace.read_text().splitlines()]
    assert len(calls) == len(events) == 2
    assert all(len(call.observed_inputs) == 3 for call in calls)
    assert events[0]["packageRoot"] != events[1]["packageRoot"]
    verify_artifacts(events, directory, index, mlx_root=root)
    monkeypatch.setattr(
        packages.ReductionPackageCache,
        "_build",
        lambda *args: pytest.fail("Verification rebuilt a package"),
    )
    for fault in ("package", "artifact", "count", "metadata", "upload", "missing"):
        changed = copy.deepcopy(events)
        event = changed[0]
        if fault == "package":
            event["packageRoot"] = events[1]["packageRoot"]
        elif fault == "artifact":
            event["artifact"] = events[1]["artifact"]
        elif fault == "count":
            event["threads"] = 131
            event["reductionMetadata"] = {"in_size": 131, "row_size": 131}
        elif fault == "metadata":
            event["reductionMetadata"]["row_size"] += 1
        elif fault == "upload":
            packet = next(
                value
                for value in event["inputs"].values()
                if value["dtype"] == "uint64"
            )
            packet["values"][0] += 1
        else:
            del event["packageRoot"]
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_artifacts(changed, directory, index, mlx_root=root)
    with pytest.raises(ValueError, match="pinned source"):
        verify_artifacts(events, directory, index)


def row_layout(simple=False):
    from demos.integrations.mlx.portable_host import row_reduction_layout
    from demos.integrations.mlx.tests.host.test_portable_rows import _buffers

    entry, buffers, count, execution, memory = _buffers(simple, 2)
    metadata = row_reduction_layout.validate(entry, buffers, count, execution)
    layout = dict(
        metadata=metadata,
        inputCount=buffers["in"].count,
        outputCount=buffers["out"].count,
    )
    return entry, count, execution, layout, buffers, memory


@pytest.mark.parametrize(
    "fault",
    [
        "metadata",
        "scalar",
        "bool",
        "element",
        "span",
        "rows",
        "rank",
        "launch",
        "extra",
    ],
)
def test_row_cache_rejects_invalid_layout_before_source_access(
    tmp_path, monkeypatch, fault
):
    entry, count, execution, layout, _, _ = row_layout()
    if fault == "metadata":
        layout.pop("metadata")
    elif fault == "scalar":
        layout["metadata"]["row_size"] = [65, 65]
    elif fault == "bool":
        layout["metadata"]["shape"][1] = True
    elif fault == "element":
        layout["metadata"]["strides"][1] = -1
    elif fault == "span":
        layout["inputCount"] -= 1
    elif fault == "rows":
        layout["outputCount"] += 1
    elif fault == "rank":
        layout["metadata"]["reduce_ndim"] = [1]
    elif fault == "launch":
        execution["workgroupCount"][1] += 1
    else:
        layout["metadata"]["extra"] = [0]
    cache = packages.ReductionPackageCache(tmp_path, tmp_path / "cache", "opengl")
    monkeypatch.setattr(
        cache, "_require_source", lambda: pytest.fail("Invalid layout accessed source")
    )
    with pytest.raises(ValueError):
        cache.get(entry, count, execution, layout=layout)
    assert not cache.directory.exists()


@pytest.fixture(
    params=[
        (target, simple)
        for target in ("metal", "opengl", "directx")
        for simple in (False, True)
    ]
)
def row_packaged(tmp_path, monkeypatch, request):
    target, simple = request.param
    entry, count, execution, layout, buffers, memory = row_layout(simple)
    if simple:
        parameters = "constant ulong& reduction_size [[buffer(2)]], constant long& out_size [[buffer(3)]]"
        value = "reduction_size + out_size"
    else:
        parameters = """constant long& row_size [[buffer(2)]],
constant long& non_row_reductions [[buffer(3)]], constant int* shape [[buffer(4)]],
constant long* strides [[buffer(5)]], constant int& ndim [[buffer(6)]],
constant int* reduce_shape [[buffer(7)]], constant long* reduce_strides [[buffer(8)]],
constant int& reduce_ndim [[buffer(9)]]"""
        value = (
            "row_size + non_row_reductions + shape[0] + shape[1] + strides[0] + strides[1] + ndim"
            " + reduce_shape[0] + reduce_shape[1] + reduce_strides[0] + reduce_strides[1] + reduce_ndim"
        )
    root = tmp_path / "source"
    source = root / packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(
        f"""template<typename T> kernel void reduce_rows(device const T* in [[buffer(0)]],
device T* out [[buffer(1)]], {parameters}, uint3 group [[threadgroup_position_in_grid]],
uint tid [[thread_index_in_threadgroup]]) {{
    T value = simd_sum(in[0]);
    if (tid == 0) out[group.y] = value + T({value});
}}
template [[host_name("{entry}")]] [[kernel]]
decltype(reduce_rows<float>) reduce_rows<float>;
"""
    )
    monkeypatch.setattr(
        packages, "SOURCE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(packages, "require_revision", lambda path: None)
    monkeypatch.setattr(packages.subprocess, "check_output", lambda *_args, **_kw: "")
    cache = packages.ReductionPackageCache(root, tmp_path / "row-cache", target)
    descriptor, package = cache.get(entry, count, execution, layout=layout)
    return cache, descriptor, package, entry, count, execution, layout, buffers, memory


def test_row_cache_retains_each_metadata_element_and_detaches_layout(
    row_packaged, monkeypatch
):
    cache, descriptor, package, entry, count, execution, layout, _, _ = row_packaged
    changed = copy.deepcopy(layout)
    contract = packages.dispatch_contract(entry, count, execution, layout=changed)
    changed["metadata"][next(iter(changed["metadata"]))][0] += 1
    assert contract["metadata"] == layout["metadata"]
    conditions = [
        item
        for binding in descriptor["bindings"]
        for item in packages._binding_conditions(binding)
    ]
    assert len(conditions) == sum(map(len, layout["metadata"].values()))
    assert all(item["minimum"] == item["maximum"] for item in conditions)
    monkeypatch.setattr(
        cache, "_build", lambda *args: pytest.fail("Cache hit rebuilt shader")
    )
    assert cache.load(entry, count, execution, layout=layout) == (descriptor, package)
    assert cache.get(entry, count, execution, layout=layout) == (descriptor, package)


def test_row_cache_identity_includes_layout_values(row_packaged):
    cache, descriptor, package, entry, count, execution, layout, _, _ = row_packaged
    other_layout = copy.deepcopy(layout)
    other_count = count
    if "strides" in layout["metadata"]:
        other_layout["metadata"]["strides"].reverse()
    else:
        other_layout["metadata"]["reduction_size"][0] += 1
        other_count += layout["outputCount"]
        other_layout["inputCount"] = other_count
    with pytest.raises(FileNotFoundError):
        cache.load(entry, other_count, execution, layout=other_layout)
    other, other_package = cache.get(entry, other_count, execution, layout=other_layout)
    assert other_package != package and other != descriptor
    assert cache.load(entry, count, execution, layout=layout) == (descriptor, package)


@pytest.mark.parametrize("fault", [None, "scalar", "element", "dtype"])
def test_row_dispatch_checks_uploaded_metadata(row_packaged, fault):
    cache, descriptor, package, entry, count, execution, layout, _, _ = row_packaged
    inputs, outputs = {}, {}
    for binding in descriptor["bindings"]:
        name = binding["name"]
        parameter = binding["provenance"]["sourceResource"]["parameter"]
        values = (
            layout["metadata"][parameter]
            if parameter in layout["metadata"]
            else [0.0] * layout["inputCount" if parameter == "in" else "outputCount"]
        )
        packet = dict(
            dtype=binding["scalarLayout"]["elementType"],
            shape=[len(values)],
            values=list(values),
        )
        inputs[name] = packet
        if parameter == "out":
            outputs[name] = copy.deepcopy(packet)
    metadata_packets = [
        packet for packet in inputs.values() if packet["dtype"] != "float32"
    ]
    if fault == "element":
        arrays = [packet for packet in metadata_packets if packet["shape"][0] > 1]
        if arrays:
            arrays[0]["values"][1] += 1
        else:
            metadata_packets[-1]["values"][0] += 1
    elif fault == "scalar":
        next(packet for packet in metadata_packets if packet["shape"] == [1])["values"][
            0
        ] += 1
    elif fault == "dtype":
        metadata_packets[0]["dtype"] = "float32"
    if fault is None:
        request = build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, execution
        )
        assert not request.execution_plan.diagnostics
    else:
        with pytest.raises(ValueError):
            build_native_loader_dispatch_request(
                descriptor, package, inputs, outputs, execution
            )


def test_row_catalog_routes_validated_uploads_and_verifies_trace(
    row_packaged, tmp_path, monkeypatch
):
    from crosstl.project.runtime_verification import RuntimeExecutorResult

    cache, _, _, entry, count, execution, layout, buffers, memory = row_packaged
    monkeypatch.setattr(reduction_packages, "require_revision", lambda root: None)
    directory = tmp_path / "row-catalog"
    index = reduction_packages.build_packages(
        cache.root,
        directory,
        cache.target,
        widths=(32,),
        entries=(entry,),
        family="row",
        on_demand=True,
    )
    host = make_host((cache.root, directory, index), tmp_path, monkeypatch)
    calls = []

    def execute(request):
        calls.append(request)
        names = {value.name for value in request.fixture.expected_outputs}
        return RuntimeExecutorResult(
            outputs={
                value.name: value.to_json()
                for value in request.fixture.inputs
                if value.name in names
            },
            details={},
        )

    host.executor.run = execute
    raw = (runtime.Buffer * len(buffers))(*buffers.values())
    launch = runtime.Launch(
        tuple(execution["workgroupCount"]), tuple(execution["workgroupSize"])
    )
    host.dispatch(entry, raw, len(buffers), count, launch=launch)
    events = [json.loads(line) for line in host.trace.read_text().splitlines()]
    assert len(calls) == len(events) == 1
    assert events[0]["reductionMetadata"] == layout["metadata"]
    assert events[0]["reductionLayout"] == {
        key: layout[key] for key in ("inputCount", "outputCount")
    }
    verify_artifacts(events, directory, index, mlx_root=cache.root)
    monkeypatch.setattr(
        packages.ReductionPackageCache,
        "_build",
        lambda *args: pytest.fail("Trace verification rebuilt shader"),
    )
    for fault in (
        "span",
        "metadata",
        "upload",
        "input-shape",
        "input-values",
        "output",
        "package",
    ):
        changed = copy.deepcopy(events)
        if fault == "span":
            changed[0]["reductionLayout"]["inputCount"] += 1
        elif fault == "metadata":
            changed[0]["reductionMetadata"][next(iter(layout["metadata"]))][0] += 1
        elif fault == "upload":
            packet = next(
                packet
                for packet in changed[0]["inputs"].values()
                if packet["dtype"] in {"int64", "uint64"}
            )
            packet["values"][0] += 1
        elif fault in {"input-shape", "input-values", "output"}:
            packets = [
                packet
                for packet in changed[0]["inputs"].values()
                if packet["dtype"] == "float32"
            ]
            packet = next(
                packet
                for packet in packets
                if packet["shape"]
                == [
                    (
                        layout["outputCount"] + 32
                        if fault == "output"
                        else layout["inputCount"]
                    )
                ]
            )
            if fault == "input-shape":
                packet["shape"][0] -= 1
            else:
                packet["values"].pop()
        else:
            changed[0]["packageRoot"] = str(directory)
        with pytest.raises((ValueError, FileNotFoundError)):
            verify_artifacts(changed, directory, index, mlx_root=cache.root)


def test_row_cache_does_not_publish_changed_metadata(row_packaged, monkeypatch):
    cache, _, _, entry, count, execution, layout, _, _ = row_packaged
    changed = copy.deepcopy(layout)
    if "strides" in changed["metadata"]:
        changed["metadata"]["strides"].reverse()
    else:
        changed["metadata"]["reduction_size"][0] += 1
        count += changed["outputCount"]
        changed["inputCount"] = count
    before = set(cache.directory.iterdir())
    original = cache._build

    def mutate(output, identity):
        original(output, identity)
        if "strides" in changed["metadata"]:
            changed["metadata"]["strides"].reverse()
        else:
            changed["metadata"]["reduction_size"][0] += 1

    monkeypatch.setattr(cache, "_build", mutate)
    with pytest.raises(ValueError):
        cache.get(entry, count, execution, layout=changed)
    assert set(cache.directory.iterdir()) == before
