"""Validate exact whole-array reduction package contracts and cache identity."""

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
