"""Scan caches bind source-owned scalar bounds to the exact native launch."""

import copy
import json

import pytest

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx.portable_host import scan_layout, scan_packages


def contract(strided):
    entry = f"{'strided' if strided else 'contig'}_scan_inclusive_sum_int32_int32"
    axis, stride = 33, 33 if strided else 1
    metadata = {
        "axisSize": axis,
        "stride": stride,
        "strideBlocks": (stride + 31) // 32,
        "storageElements": axis * stride * 3,
        "operation": "sum",
        "sourceType": "int32",
        "outputType": "int32",
    }
    launch = scan_layout.execution(entry, metadata["storageElements"], axis, stride)
    return entry, metadata, launch


@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "extra",
        "operation",
        "source",
        "destination",
        "blocks",
        "bool-blocks",
        "bool-axis",
        "bool-stride",
        "bool-size",
        "axis",
        "size",
        "stride",
        "groups",
        "width",
        "bool-launch",
        "extra-launch",
        "missing-launch",
        "null-launch",
    ),
)
def test_cache_rejects_unchecked_contract_before_source_access(
    tmp_path, monkeypatch, strided, fault
):
    entry, metadata, launch = contract(strided)
    if fault == "missing":
        metadata.pop("axisSize")
    elif fault == "extra":
        metadata["unchecked"] = 1
    elif fault in {"operation", "source", "destination"}:
        key = {
            "operation": "operation",
            "source": "sourceType",
            "destination": "outputType",
        }[fault]
        metadata[key] = "other"
    elif fault in {"blocks", "bool-blocks"}:
        metadata["strideBlocks"] = 3 if fault == "blocks" else True
    elif fault.startswith("bool-") and fault != "bool-launch":
        key = {
            "bool-axis": "axisSize",
            "bool-stride": "stride",
            "bool-size": "storageElements",
        }[fault]
        metadata[key] = True
    elif fault in {"axis", "size", "stride"}:
        key = {"axis": "axisSize", "size": "storageElements", "stride": "stride"}[fault]
        metadata[key] = 0
    elif fault == "groups":
        launch["workgroupCount"][1] += 1
    elif fault == "width":
        launch["workgroupSize"][0] *= 2
    elif fault == "bool-launch":
        launch["workgroupSize"][1] = True
    elif fault == "extra-launch":
        launch["unchecked"] = [1, 1, 1]
    elif fault == "missing-launch":
        launch.pop("workgroupCount")
    else:
        launch = None
    accesses = []
    monkeypatch.setattr(
        scan_packages.ScanPackageCache,
        "_require_source",
        lambda self: accesses.append(self),
    )
    cache = scan_packages.ScanPackageCache(tmp_path, tmp_path / "cache", "opengl")
    with pytest.raises(ValueError):
        cache.get(entry, metadata, launch)
    assert not accesses and not cache.directory.exists()


@pytest.fixture(
    params=[
        (target, strided)
        for target in ("metal", "directx", "opengl")
        for strided in (False, True)
    ]
)
def packaged_scan(tmp_path, monkeypatch, request):
    target, strided = request.param
    entry, metadata, launch = contract(strided)
    source = tmp_path / "source" / scan_packages.SOURCE
    source.parent.mkdir(parents=True)
    extra = ", constant ulong& stride, constant ulong& stride_blocks" if strided else ""
    expression = "axis_size + stride + stride_blocks" if strided else "axis_size"
    source.write_text(
        "template<typename T>\nkernel void scan(device const T* in, device T* out, "
        f"constant ulong& axis_size{extra}, uint id [[thread_position_in_grid]]) {{\n"
        f"    out[id] = simd_sum(in[id]) + T({expression});\n}}\n"
        f'template [[host_name("{entry}")]] [[kernel]] decltype(scan<int>) scan<int>;\n'
    )
    monkeypatch.setattr(
        scan_packages.ScanPackageCache, "_require_source", lambda self: None
    )
    cache = scan_packages.ScanPackageCache(
        source.parents[4], tmp_path / "cache", target
    )
    descriptor, package = cache.get(entry, metadata, launch)
    return cache, descriptor, package, entry, metadata, launch


def test_scan_package_retains_checked_contracts_and_reloads(packaged_scan):
    cache, descriptor, package, entry, metadata, launch = packaged_scan
    assert cache.get(
        entry, dict(reversed(list(metadata.items()))), copy.deepcopy(launch)
    ) == (descriptor, package)
    record = json.loads((package.parent / "index.json").read_text())
    expected = {"axis_size": metadata["axisSize"]}
    if metadata["stride"] > 1:
        expected.update(
            stride=metadata["stride"], stride_blocks=metadata["strideBlocks"]
        )
    assert record["identity"]["metadataValues"] == expected
    assert record["identity"]["dispatchGeometry"] == launch
    conditions = {
        binding["provenance"]["sourceResource"]["parameter"]: binding["provenance"][
            "metadataPrecondition"
        ]
        for binding in descriptor["bindings"]
        if "metadataPrecondition" in binding.get("provenance", {})
    }
    assert set(conditions) == set(expected)
    for name, value in expected.items():
        assert conditions[name] == {
            "dtype": "uint64",
            "minimum": value,
            "maximum": value,
            "sourceResource": {
                "schemaVersion": 1,
                "backend": "metal",
                "entryPoint": entry,
                "parameter": name,
                "parameterIndex": {"axis_size": 2, "stride": 3, "stride_blocks": 4}[
                    name
                ],
            },
        }
    reversed_identity = copy.deepcopy(record["identity"])
    reversed_identity["metadataValues"] = dict(reversed(list(expected.items())))
    assert cache._load(package.parent, reversed_identity) == (descriptor, package)
    metadata["axisSize"] += 1
    launch["workgroupCount"][1] += 1
    assert record["identity"]["metadataValues"] == expected
    assert cache._load(package.parent, record["identity"]) == (descriptor, package)


@pytest.mark.parametrize(
    "fault", ("metadata", "launch", "missing-contract", "wrong-ordinal")
)
def test_scan_cache_rejects_relabelled_package(packaged_scan, fault):
    cache, descriptor, package, entry, metadata, launch = packaged_scan
    if cache.target == "metal" and fault == "launch":
        pytest.skip(
            "Metal preserves the runtime launch rather than specializing a region"
        )
    index = package.parent / "index.json"
    record = json.loads(index.read_text())
    identity = record["identity"]
    if fault == "metadata":
        identity["metadataValues"]["axis_size"] += 1
    elif fault == "launch":
        identity["dispatchGeometry"]["workgroupCount"][1] += 1
    else:
        binding = next(
            b
            for b in record["descriptor"]["bindings"]
            if "metadataPrecondition" in b.get("provenance", {})
        )
        if fault == "missing-contract":
            binding["provenance"].pop("metadataPrecondition")
        else:
            binding["provenance"]["metadataPrecondition"]["sourceResource"][
                "parameterIndex"
            ] += 1
    index.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="descriptor does not match"):
        cache._load(package.parent, identity)


def test_scan_dispatch_rejects_metadata_outside_packaged_bounds(packaged_scan):
    cache, descriptor, package, entry, metadata, launch = packaged_scan
    buffers = {}
    for binding in descriptor["bindings"]:
        name = binding["provenance"]["sourceResource"]["parameter"]
        values = {
            "axis_size": [metadata["axisSize"]],
            "stride": [metadata["stride"]],
            "stride_blocks": [metadata["strideBlocks"]],
        }.get(name, [0] * metadata["storageElements"])
        buffers[binding["name"]] = {
            "dtype": binding["scalarLayout"]["elementType"],
            "shape": [len(values)],
            "values": values,
        }
    output = next(
        b["name"]
        for b in descriptor["bindings"]
        if b["provenance"]["sourceResource"]["parameter"] == "out"
    )
    outputs = {
        output: {
            key: value for key, value in buffers[output].items() if key != "values"
        }
    }
    build_native_loader_dispatch_request(descriptor, package, buffers, outputs, launch)
    axis = next(
        b["name"]
        for b in descriptor["bindings"]
        if b["provenance"]["sourceResource"]["parameter"] == "axis_size"
    )
    buffers[axis]["values"][0] += 1
    with pytest.raises(ValueError, match="precondition"):
        build_native_loader_dispatch_request(
            descriptor, package, buffers, outputs, launch
        )
