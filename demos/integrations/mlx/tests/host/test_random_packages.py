"""Random package variants bind byte counts and launch geometry to their artifacts."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.project import build_native_loader_dispatch_request
from demos.integrations.mlx import random_audit
from demos.integrations.mlx.portable_host import (
    random_evidence,
    random_packages,
    verify_upstream_blocks,
)


def launch(keys=3, width=1):
    return {"workgroupCount": [keys, width, 1], "workgroupSize": [1, 1, 1]}


@pytest.mark.parametrize(
    "fault",
    (
        "keys-bool",
        "keys-zero",
        "keys-overflow",
        "bytes-bool",
        "bytes-zero",
        "bytes-short",
        "bytes-overflow",
        "storage-overflow",
        "groups",
        "width",
        "bool-launch",
        "float-launch",
        "extra-launch",
        "missing-launch",
        "null-launch",
    ),
)
def test_random_cache_rejects_invalid_contract_before_source_access(
    tmp_path, monkeypatch, fault
):
    keys, size, execution = 3, 4, launch()
    if fault == "keys-bool":
        keys = True
    elif fault == "keys-zero":
        keys = 0
    elif fault == "keys-overflow":
        keys = 65536
    elif fault == "bytes-bool":
        size = True
    elif fault == "bytes-zero":
        size = 0
    elif fault == "bytes-short":
        size = 3
    elif fault == "bytes-overflow":
        # 65537 word pairs per key has no exact portable group height.
        size = 524296
    elif fault == "storage-overflow":
        keys, size = 65535, 524280
    elif fault == "groups":
        execution["workgroupCount"][0] += 1
    elif fault == "width":
        execution["workgroupSize"][0] = 32
    elif fault in {"bool-launch", "float-launch"}:
        execution["workgroupSize"][0] = True if fault == "bool-launch" else 1.0
    elif fault == "extra-launch":
        execution["other"] = [1, 1, 1]
    elif fault == "missing-launch":
        execution.pop("workgroupSize")
    else:
        execution = None
    monkeypatch.setattr(
        random_packages,
        "verify_source",
        lambda *_: pytest.fail("Invalid contract accessed source"),
    )
    cache = random_packages.RandomPackageCache(tmp_path, tmp_path / "cache", "opengl")
    with pytest.raises(ValueError):
        cache.get(keys, size, execution)
    assert not cache.directory.exists()


@pytest.fixture(params=("metal", "opengl", "directx"))
def packaged_random(tmp_path, monkeypatch, request):
    root = tmp_path / "source"
    source = root / random_packages.SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("""kernel void rbitsc(device const uint* keys, device char* out,
    constant bool& odd, constant ulong& bytes_per_key,
    uint3 index [[thread_position_in_grid]]) {
    out += index.x * bytes_per_key;
    ulong idx = ulong(index.y) << 2;
    out[idx] = char(keys[index.x] + uint(odd));
}
kernel void rbits(device const uint* keys, device char* out,
    constant bool& odd, constant ulong& bytes_per_key,
    constant int& ndim, constant int* key_shape, constant long* key_strides,
    uint3 index [[thread_position_in_grid]]) {
    out += index.x * bytes_per_key;
    ulong idx = ulong(index.y) << 2;
    long k1_elem = key_strides[0] * index.x;
    long k2_elem = k1_elem + key_strides[1];
    out[idx] = char(keys[k1_elem] + keys[k2_elem] + uint(odd) + ndim + key_shape[0]);
}
""")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(random_packages, "SOURCE_SHA256", digest)

    def verify(supplied):
        assert supplied == root
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Test source changed")

    monkeypatch.setattr(random_packages, "verify_source", verify)
    cache = random_packages.RandomPackageCache(
        root, tmp_path / "cache" / "variants", request.param
    )
    descriptors, package = cache.get(3, 4, launch())
    return cache, descriptors, package


def test_random_contract_is_detached_and_cache_reloads(packaged_random, monkeypatch):
    cache, descriptors, package = packaged_random
    original = launch()
    contract = random_packages.dispatch_contract(3, 4, original)
    original["workgroupCount"][0] = 1
    assert contract["execution"] == launch()
    monkeypatch.setattr(
        random_packages,
        "build_packages",
        lambda *_args, **_kw: pytest.fail("Cache hit rebuilt the package"),
    )
    assert cache.get(3, 4, launch()) == (descriptors, package)
    assert cache.load(3, 4, launch()) == (descriptors, package)
    assert (
        random_packages.load_index(
            package.parent, cache.target, expected_contract=contract
        )
        == descriptors
    )
    record = json.loads((package.parent / "index.json").read_text())
    assert record["contract"] == record["cacheIdentity"]["contract"] == contract
    for entry, descriptor in descriptors.items():
        conditions = [
            b["provenance"]["metadataPrecondition"]
            for b in descriptor["bindings"]
            if "metadataPrecondition" in b.get("provenance", {})
        ]
        assert conditions == [random_packages._condition(entry, contract)]
    with pytest.raises(FileNotFoundError):
        cache.load(3, 5, launch())
    with pytest.raises(FileNotFoundError):
        cache.load(1, 4, launch(1))


@pytest.mark.parametrize(
    "fault", ("identity", "contract", "metadata", "region", "artifact")
)
def test_random_cache_rejects_changed_record_or_artifact(packaged_random, fault):
    cache, descriptors, package = packaged_random
    path = package.parent / "index.json"
    record = json.loads(path.read_text())
    if fault == "identity":
        record["cacheIdentity"]["recipeHash"] = "0" * 64
    elif fault == "contract":
        record["contract"]["bytesPerKey"] = 5
    elif fault == "metadata":
        binding = next(
            b
            for b in record["descriptors"]["rbitsc"]["bindings"]
            if "metadataPrecondition" in b.get("provenance", {})
        )
        binding["provenance"]["metadataPrecondition"]["maximum"] += 1
    elif fault == "region":
        record["descriptors"]["rbitsc"].setdefault("provenance", {})[
            "dispatchRegion"
        ] = {}
    else:
        (package / descriptors["rbitsc"]["artifact"]["packagePath"]).write_text(
            "// changed artifact\n"
        )
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        cache.get(3, 4, launch())


def test_random_cache_rejects_changed_source_and_implementation(
    packaged_random, monkeypatch
):
    cache, descriptors, package = packaged_random
    monkeypatch.setattr(
        random_packages, "translation_implementation_hash", lambda: "0" * 64
    )
    with pytest.raises(FileNotFoundError):
        cache.load(3, 4, launch())
    source = cache.root / random_packages.SOURCE
    source.write_text(source.read_text() + "\n// changed source\n")
    with pytest.raises(ValueError, match="source changed"):
        cache.load(3, 4, launch())


def test_random_cache_separates_equal_grid_byte_contracts(packaged_random):
    cache, descriptors, package = packaged_random
    other, other_package = cache.get(3, 5, launch())
    assert package != other_package
    assert other != descriptors
    assert cache.load(3, 4, launch()) == (descriptors, package)
    assert cache.load(3, 5, launch()) == (other, other_package)


def test_random_cache_translates_wide_outputs_with_exact_group_heights(
    packaged_random,
):
    cache = packaged_random[0]
    wide = {"workgroupCount": [1, 32768, 1], "workgroupSize": [1, 2, 1]}
    with pytest.raises(ValueError, match="launch differs"):
        cache.get(1, 524288, launch(1, 65536))
    descriptors, package = cache.get(1, 524288, wide)
    assert cache.load(1, 524288, wide) == (descriptors, package)
    region = {
        "threadGridSize": [1, 65536, 1],
        "sourceWorkgroupSize": [1, 2, 1],
        "workgroupOffset": [0, 0, 0],
        "workgroupCount": [1, 32768, 1],
        "workgroupSize": [1, 2, 1],
    }
    for descriptor in descriptors.values():
        assert descriptor["provenance"].get("dispatchRegion") == (
            None if cache.target == "metal" else region
        )
        artifact = package / descriptor["artifact"]["packagePath"]
        text = artifact.read_text()
        if cache.target == "opengl":
            assert "local_size_x = 1, local_size_y = 2, local_size_z = 1" in text
        elif cache.target == "directx":
            assert "[numthreads(1, 2, 1)]" in text


def test_random_cache_does_not_publish_changed_translation_inputs(
    packaged_random, monkeypatch
):
    cache, descriptors, package = packaged_random
    build = random_packages.build_packages
    before = set(cache.directory.iterdir())

    def changed(*args, **kwargs):
        result = build(*args, **kwargs)
        source = cache.root / random_packages.SOURCE
        source.write_text(source.read_text() + "\n// changed during translation\n")
        return result

    monkeypatch.setattr(random_packages, "build_packages", changed)
    with pytest.raises(ValueError, match="source changed"):
        cache.get(3, 8, launch())
    assert set(cache.directory.iterdir()) == before


@pytest.mark.parametrize("entry", random_packages.ENTRIES)
def test_random_request_rejects_values_outside_artifact_contract(
    packaged_random, entry
):
    cache, descriptors, package = packaged_random
    descriptor = descriptors[entry]
    case = random_audit.workload(entry, 3, 1)
    inputs, outputs = random_audit.dispatch_values(descriptor, case)
    build_native_loader_dispatch_request(descriptor, package, inputs, outputs, launch())
    binding = next(
        b
        for b in descriptor["bindings"]
        if "metadataPrecondition" in b.get("provenance", {})
    )
    inputs[binding["name"]]["values"] = [5]
    with pytest.raises(ValueError, match="precondition"):
        build_native_loader_dispatch_request(
            descriptor, package, inputs, outputs, launch()
        )
    inputs[binding["name"]]["values"] = [4]
    if cache.target != "metal":
        with pytest.raises(ValueError, match="geometry"):
            build_native_loader_dispatch_request(
                descriptor, package, inputs, outputs, launch(4)
            )


@pytest.mark.parametrize(
    "fault", (None, "root", "artifact", "keys", "bytes", "entry", "target")
)
def test_random_evidence_requires_the_dispatched_cached_artifact(
    packaged_random, fault
):
    cache, descriptors, package = packaged_random
    event = {
        "entry": "rbitsc",
        "target": cache.target,
        "artifact": copy.deepcopy(descriptors["rbitsc"]["artifact"]),
        "packageRoot": str(package),
        "randomMetadata": {"keyCount": 3, "nativeBytesPerKey": 4},
        **launch(),
    }
    if fault == "root":
        event["packageRoot"] = str(package.parent)
    elif fault == "artifact":
        event["artifact"]["hash"]["value"] = "0" * 64
    elif fault == "keys":
        event["randomMetadata"]["keyCount"] = 1
        event["workgroupCount"][0] = 1
    elif fault == "bytes":
        event["randomMetadata"]["nativeBytesPerKey"] = 5
    elif fault == "entry":
        event["entry"] = "unknown"
    elif fault == "target":
        event["target"] = "unknown"
    if fault:
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            random_evidence.verify_packages(
                [event], cache.root, cache.directory.parent, cache.target
            )
    else:
        random_evidence.verify_packages(
            [event], cache.root, cache.directory.parent, cache.target
        )
        assert verify_upstream_blocks.event_packages(
            event, SimpleNamespace(randoms=cache)
        ) == [(descriptors["rbitsc"], Path(event["packageRoot"]))]
