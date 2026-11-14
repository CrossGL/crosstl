"""Check the dispatch evidence verifier against real fixture results and artifacts.

The fixture inventory and its MSTest results come from the descriptor fixture,
and the manifest from translating the fixture HLSL with CrossTL. The evidence
records are written the way the portable runtime writes them, so each mutation
corresponds to a runtime or harness defect the verifier must reject.
"""

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import (  # noqa: E402
    FIXTURE_CONTRACT,
    FIXTURE_POLICY,
    ROOT,
    load_module,
)

EVIDENCE = load_module("computesharp_evidence", ROOT / "evidence.py")
PROFILE = "translated"
DEVICE_CASE = "DiscoveryFixture.Cases.Device (Warp)"
DISPATCH_CASE = "DiscoveryFixture.Cases.Values (1.5,System.UInt32)"
UPLOAD_CASE = "DiscoveryFixture.Cases.Values (-0,System.Int32)"
DISCRETE_CASE = "DiscoveryFixture.Cases.Device (Discrete)"
UNGATED_CASE = "DiscoveryFixture.Cases.Gated (System.Int32)"
GATED_CASE = "DiscoveryFixture.Cases.Gated (System.Single)"
HOST_ONLY_CASE = "DiscoveryFixture.Cases.Expanded expanded"


@pytest.fixture(scope="session")
def translated_fixture(baseline_fixture, tmp_path_factory):
    discovery, trx = baseline_fixture
    artifacts = tmp_path_factory.mktemp("fixture-artifacts")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "translate.py"),
            "--discovery",
            str(discovery),
            "--output",
            str(artifacts),
            "--target",
            "metal",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return discovery, trx, artifacts / "metal" / "manifest.json"


def _records(manifest_path):
    """Consistent evidence for every required fixture case, keyed by role."""
    manifest = json.loads(manifest_path.read_text())
    ((hlsl, entry),) = manifest["shaders"].items()
    assert entry["status"] == "translated"
    target = next(item for item in entry["bindings"] if item["register"] == "u2")
    common = {"backend": "metal", "case": DISPATCH_CASE}
    return {
        "device": {
            "record": "device",
            "backend": "metal",
            "case": DEVICE_CASE,
            "name": "Fixture GPU",
            "api": "Metal",
            "driver": "fixture driver",
            "hardware": True,
            "fp64": False,
            "manifest": str(manifest_path),
            "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "translatorCommit": manifest["translator"]["commit"],
            "bufferGuardBytes": 256,
        },
        "gate": {
            "record": "capability",
            "backend": "metal",
            "case": GATED_CASE,
            "capability": "fixture-feature",
            "available": True,
        },
        "buffer": {
            **common,
            "record": "resource",
            "resourceId": 7,
            "type": "ComputeSharp.ReadWriteBuffer<System.UInt32>",
            "memory": "device",
            "bytes": 260,
        },
        "upload": {
            "record": "resource",
            "backend": "metal",
            "case": UPLOAD_CASE,
            "resourceId": 8,
            "type": "ComputeSharp.UploadBuffer<System.Int32>",
            "memory": "host",
            "bytes": 64,
        },
        "texture": {
            "record": "resource",
            "backend": "metal",
            "case": DISCRETE_CASE,
            "resourceId": 9,
            "type": "ComputeSharp.ReadWriteTexture2D<System.Single>",
            "memory": "device",
            "bytes": 1024,
        },
        "ungated": {
            "record": "capability",
            "backend": "metal",
            "case": UNGATED_CASE,
            "capability": "texture",
            "available": True,
        },
        "upload-release": {"record": "release", "backend": "metal", "resourceId": 8},
        "pipeline": {
            **common,
            "record": "pipeline",
            "shader": entry["type"],
            "hlslSha256": hlsl,
            "artifactSha256": entry["artifactSha256"],
            "threads": entry["threads"],
        },
        "dispatch": {
            **common,
            "record": "dispatch",
            "shader": entry["type"],
            "hlslSha256": hlsl,
            "artifactSha256": entry["artifactSha256"],
            "dispatch": [65, 1, 1],
            "groups": [3, 1, 1],
            "constants": "41000000010000000100000000000000",
            "bindings": [
                {
                    "register": "u2",
                    "namespace": target["namespace"],
                    "slot": target["binding"],
                    "resourceId": 7,
                }
            ],
        },
        "readback": {
            **common,
            "record": "readback",
            "resource": "ReadWriteBuffer<uint>[65]",
            "resourceId": 7,
            "bytes": 260,
            "sha256": "0" * 64,
        },
        "release": {"record": "release", "backend": "metal", "resourceId": 7},
    }


def _write(path, records, compress=False):
    if isinstance(records, dict):
        records = list(records.values())
    text = "".join(json.dumps(record) + "\n" for record in records)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    return path


def _verify(translated_fixture, evidence, fault=None, policy=FIXTURE_POLICY):
    discovery, trx, manifest = translated_fixture
    return EVIDENCE.verify(
        evidence=evidence,
        manifest_path=manifest,
        discovery_dir=discovery,
        trx=trx,
        profile=PROFILE,
        contract_path=FIXTURE_CONTRACT,
        fault=fault,
        policy_path=policy,
    )


@pytest.mark.parametrize("compress", [False, True])
def test_consistent_evidence_is_accepted(translated_fixture, tmp_path, compress):
    name = "evidence.jsonl.gz" if compress else "evidence.jsonl"
    records = _records(translated_fixture[2])
    summary = _verify(translated_fixture, _write(tmp_path / name, records, compress))
    assert summary["status"] == "passed"
    assert summary["records"] == {
        "capability": 2,
        "device": 1,
        "dispatch": 1,
        "pipeline": 1,
        "readback": 1,
        "release": 2,
        "resource": 3,
    }
    assert summary["device"]["hardware"] is True
    assert summary["dispatchedShaders"] == summary["translatedShaders"] == 1
    assert summary["translatedShadersNotDispatched"] == []
    assert summary["casesWithDispatch"] == 1
    assert summary["casesWithEvidence"] == 6
    assert summary["resources"] == {"device": 2, "host": 1}
    assert summary["releasedResources"] == 2
    assert summary["requiredCaseNamesExecuted"] == 7
    assert summary["requiredCaseNamesWithEvidence"] == 6
    assert summary["hostOnlyCaseNames"] == 1
    assert summary["hostOnlyMethodsWithEvidence"] == []
    assert summary["gatedCaseNamesWithCapabilityQuery"] == 1
    assert (
        summary["policySha256"]
        == hashlib.sha256(FIXTURE_POLICY.read_bytes()).hexdigest()
    )


def test_host_only_methods_that_leave_evidence_are_reported(
    translated_fixture, tmp_path
):
    records = _records(translated_fixture[2])
    records["host"] = {**records["upload"], "case": HOST_ONLY_CASE, "resourceId": 10}
    summary = _verify(translated_fixture, _write(tmp_path / "evidence.jsonl", records))
    assert summary["hostOnlyCaseNames"] == 0
    assert summary["hostOnlyMethodsWithEvidence"] == ["DiscoveryFixture.Cases.Expanded"]


def _mutate(records, manifest_path, mutation):
    device, pipeline = records["device"], records["pipeline"]
    dispatch, readback = records["dispatch"], records["readback"]
    if mutation == "no-device":
        del records["device"]
    elif mutation == "no-dispatch":
        del records["dispatch"]
    elif mutation == "backend":
        dispatch["backend"] = "opengl"
    elif mutation == "software-device":
        device["hardware"] = False
    elif mutation == "other-manifest":
        device["manifestSha256"] = "3" * 64
    elif mutation == "no-guards":
        device.pop("bufferGuardBytes")
    elif mutation == "unknown-hlsl":
        pipeline["hlslSha256"] = "1" * 64
    elif mutation == "artifact-hash":
        dispatch["artifactSha256"] = pipeline["artifactSha256"] = "2" * 64
    elif mutation == "artifact-file":
        manifest = json.loads(manifest_path.read_text())
        (entry,) = manifest["shaders"].values()
        artifact = manifest_path.parent.parent / entry["artifact"]
        artifact.write_text(artifact.read_text() + "\n// changed\n")
    elif mutation == "threads":
        pipeline["threads"] = [64, 1, 1]
    elif mutation == "groups":
        dispatch["groups"] = [2, 1, 1]
    elif mutation == "zero-dispatch":
        dispatch["dispatch"] = [0, 1, 1]
        dispatch["groups"] = [0, 1, 1]
    elif mutation == "constants":
        dispatch["constants"] = "4100"
    elif mutation == "register":
        dispatch["bindings"][0]["register"] = "u3"
    elif mutation == "slot":
        dispatch["bindings"][0]["slot"] += 1
    elif mutation == "missing-binding":
        dispatch["bindings"] = []
    elif mutation == "no-resource":
        dispatch["bindings"][0].pop("resourceId")
    elif mutation == "unattributed":
        readback.pop("case")
    elif mutation == "unexecuted-case":
        readback["case"] = "DiscoveryFixture.Cases.Missing"
    elif mutation == "platform-case":
        for record in (pipeline, dispatch):
            record["case"] = "DiscoveryFixture.Cases.Plain"
    elif mutation == "unexpected-fault":
        dispatch["fault"] = "omit-dispatch"
    elif mutation == "readback-hash":
        readback["sha256"] = "abc"
    elif mutation == "unknown-record":
        records["upload-copy"] = {**readback, "record": "upload"}
    elif mutation == "unallocated-binding":
        dispatch["bindings"][0]["resourceId"] = 42
    elif mutation == "host-binding":
        dispatch["bindings"][0]["resourceId"] = 8
    elif mutation == "unallocated-readback":
        del records["buffer"]
    elif mutation == "use-after-release":
        release = records.pop("release")
        items = list(records.items())
        index = [key for key, _ in items].index("dispatch")
        items.insert(index, ("release", release))
        records.clear()
        records.update(items)
    elif mutation == "double-release":
        records["again"] = dict(records["upload-release"])
    elif mutation == "unknown-release":
        records["upload-release"]["resourceId"] = 99
    elif mutation == "duplicate-resource":
        records["texture"]["resourceId"] = 7
    elif mutation == "resource-size":
        records["buffer"]["bytes"] = 0
    elif mutation == "resource-memory":
        records["texture"]["memory"] = "shared"
    elif mutation == "unattributed-resource":
        records["texture"].pop("case")
    elif mutation == "case-without-evidence":
        del records["texture"]
    elif mutation == "gate-without-query":
        del records["gate"]
    elif mutation == "gate-answer":
        records["gate"]["available"] = False
    elif mutation == "capability-without-answer":
        records["ungated"].pop("available")
    elif mutation == "fp64-answer":
        records["fp64"] = {
            **records["ungated"],
            "capability": "fp64",
            "available": True,
        }
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "no-device",
        "no-dispatch",
        "backend",
        "software-device",
        "other-manifest",
        "no-guards",
        "unknown-hlsl",
        "artifact-hash",
        "artifact-file",
        "threads",
        "groups",
        "zero-dispatch",
        "constants",
        "register",
        "slot",
        "missing-binding",
        "no-resource",
        "unattributed",
        "unexecuted-case",
        "platform-case",
        "unexpected-fault",
        "readback-hash",
        "unknown-record",
        "unallocated-binding",
        "host-binding",
        "unallocated-readback",
        "use-after-release",
        "double-release",
        "unknown-release",
        "duplicate-resource",
        "resource-size",
        "resource-memory",
        "unattributed-resource",
        "case-without-evidence",
        "gate-without-query",
        "gate-answer",
        "capability-without-answer",
        "fp64-answer",
    ],
)
def test_inconsistent_evidence_is_rejected(baseline_fixture, tmp_path, mutation):
    # Each mutation gets its own artifacts so a changed file cannot leak.
    discovery, trx = baseline_fixture
    artifacts = tmp_path / "artifacts"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "translate.py"),
            "--discovery",
            str(discovery),
            "--output",
            str(artifacts),
            "--target",
            "metal",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = artifacts / "metal" / "manifest.json"
    records = _records(manifest)
    _mutate(records, manifest, mutation)
    with pytest.raises(EVIDENCE.EvidenceError) as raised:
        _verify(
            (discovery, trx, manifest), _write(tmp_path / "evidence.jsonl", records)
        )
    assert raised.value.errors


def _with_unorm_store_scales(translated_fixture, tmp_path):
    """The fixture artifacts with a UNORM store scale buffer for register u2."""
    discovery, trx, manifest_path = translated_fixture
    artifacts = tmp_path / "artifacts"
    shutil.copytree(manifest_path.parent.parent, artifacts)
    manifest = json.loads(manifest_path.read_text())
    ((_hlsl, entry),) = manifest["shaders"].items()
    slot = 1 + max(item["binding"] for item in entry["bindings"])
    entry["bindings"].append(
        {
            "name": "crosstl_unorm_store_u2",
            "parameter": "crosstl_unorm_store_u2",
            "kind": "constant-buffer",
            "namespace": "buffer",
            "binding": slot,
            "count": 1,
            "register": None,
        }
    )
    copy = artifacts / manifest_path.parent.name / manifest_path.name
    copy.write_text(json.dumps(manifest))
    return (discovery, trx, copy), slot


@pytest.mark.parametrize(
    "mutation", [None, "scales", "slot", "format", "missing", "undeclared"]
)
def test_unorm_store_scales_must_match_the_view_format(
    translated_fixture, tmp_path, mutation
):
    fixture, slot = _with_unorm_store_scales(translated_fixture, tmp_path)
    if mutation == "undeclared":
        fixture = translated_fixture
    records = _records(fixture[2])
    binding = records["dispatch"]["bindings"][0]
    binding.update(
        {"format": 28, "unormScaleSlot": slot, "unormScales": [255, 255, 255, 255]}
    )
    if mutation == "scales":
        binding["unormScales"] = [1, 1, 1, 1]
    elif mutation == "slot":
        binding["unormScaleSlot"] = slot + 1
    elif mutation == "format":
        binding["format"] = 11
    elif mutation == "missing":
        del binding["unormScales"], binding["unormScaleSlot"]
    evidence = _write(tmp_path / "evidence.jsonl", records)
    if mutation is None:
        assert _verify(fixture, evidence)["status"] == "passed"
        return
    with pytest.raises(EVIDENCE.EvidenceError) as raised:
        _verify(fixture, evidence)
    assert any("UNORM store scales" in error for error in raised.value.errors)


@pytest.mark.parametrize(
    "scales, accepted",
    [([1, 1, 1, 1], True), ([255, 255, 255, 255], False), ([0, 0, 0, 0], False)],
)
def test_unorm_scale_control_evidence_must_show_the_perturbation(
    translated_fixture, tmp_path, scales, accepted
):
    fixture, slot = _with_unorm_store_scales(translated_fixture, tmp_path)
    records = _records(fixture[2])
    for record in records.values():
        record["fault"] = "perturb-unorm-scales"
    records["dispatch"]["bindings"][0].update(
        {"format": 28, "unormScaleSlot": slot, "unormScales": scales}
    )
    evidence = _write(tmp_path / "evidence.jsonl", records)
    if accepted:
        summary = _verify(fixture, evidence, fault="perturb-unorm-scales")
        assert summary["status"] == "passed"
        return
    with pytest.raises(EVIDENCE.EvidenceError) as raised:
        _verify(fixture, evidence, fault="perturb-unorm-scales")
    assert any("UNORM store scales" in error for error in raised.value.errors)


def test_negative_control_evidence_must_carry_its_fault(translated_fixture, tmp_path):
    records = _records(translated_fixture[2])
    with pytest.raises(EVIDENCE.EvidenceError):
        _verify(
            translated_fixture,
            _write(tmp_path / "plain.jsonl", records),
            fault="swap-bindings",
        )
    for record in records.values():
        record["fault"] = "swap-bindings"
    summary = _verify(
        translated_fixture,
        _write(tmp_path / "faulted.jsonl", records),
        fault="swap-bindings",
    )
    assert summary["status"] == "passed"


def test_invalid_json_lines_are_rejected(translated_fixture, tmp_path):
    path = tmp_path / "evidence.jsonl"
    path.write_text('{"record": "device"\n')
    with pytest.raises(EVIDENCE.EvidenceError):
        _verify(translated_fixture, path)


def _policy(tmp_path, **changes):
    policy = json.loads(FIXTURE_POLICY.read_text())
    policy.update(changes)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "other"},
        {"capabilityQueries": {}},
        {
            "capabilityQueries": {
                "fixture-feature": {"capability": "fixture-feature"},
                "fp64": {"capability": "fp64"},
            }
        },
        {
            "hostOnlyMethods": {
                "DiscoveryFixture.Cases.Plain": (
                    "Fixture platform case that never runs on a device."
                )
            }
        },
        {
            "hostOnlyMethods": {
                "DiscoveryFixture.Cases.Missing": (
                    "Fixture method that is not in the inventory."
                )
            }
        },
        {"hostOnlyMethods": {"DiscoveryFixture.Cases.Expanded": "host"}},
    ],
)
def test_invalid_evidence_policies_are_rejected(translated_fixture, tmp_path, changes):
    records = _records(translated_fixture[2])
    evidence = _write(tmp_path / "evidence.jsonl", records)
    with pytest.raises(EVIDENCE.EvidenceError):
        _verify(translated_fixture, evidence, policy=_policy(tmp_path, **changes))


def test_port_evidence_policy_matches_the_port_contract():
    contract = json.loads((ROOT / "coverage.json").read_text())
    policy = json.loads(EVIDENCE.POLICY.read_text())
    assert set(policy["capabilityQueries"]) == set(contract["capabilities"])
    for method, reason in policy["hostOnlyMethods"].items():
        entry = contract["methods"].get(method) or contract["classes"].get(
            method.rsplit(".", 1)[0]
        )
        assert entry is not None and entry["category"] == "required-gpu", method
        assert len(reason) >= 20, method
