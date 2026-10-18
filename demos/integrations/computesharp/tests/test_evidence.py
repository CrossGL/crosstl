"""Check the dispatch evidence verifier against real fixture results and artifacts.

The fixture inventory and its MSTest results come from the descriptor fixture,
and the manifest from translating the fixture HLSL with CrossTL. The evidence
records are written the way the portable runtime writes them, so each mutation
corresponds to a runtime or harness defect the verifier must reject.
"""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
from fixture_support import FIXTURE_CONTRACT, ROOT, load_module  # noqa: E402

EVIDENCE = load_module("computesharp_evidence", ROOT / "evidence.py")
PROFILE = "translated"
DEVICE_CASE = "DiscoveryFixture.Cases.Device (Warp)"
DISPATCH_CASE = "DiscoveryFixture.Cases.Values (1.5,System.UInt32)"


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
    manifest = json.loads(manifest_path.read_text())
    ((hlsl, entry),) = manifest["shaders"].items()
    assert entry["status"] == "translated"
    target = next(item for item in entry["bindings"] if item["register"] == "u2")
    common = {"backend": "metal", "case": DISPATCH_CASE}
    return [
        {
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
        {
            **common,
            "record": "pipeline",
            "shader": entry["type"],
            "hlslSha256": hlsl,
            "artifactSha256": entry["artifactSha256"],
            "threads": entry["threads"],
        },
        {
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
        {
            **common,
            "record": "readback",
            "resource": "ReadWriteBuffer<uint>[65]",
            "resourceId": 7,
            "bytes": 260,
            "sha256": "0" * 64,
        },
    ]


def _write(path, records, compress=False):
    text = "".join(json.dumps(record) + "\n" for record in records)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text)
    return path


def _verify(translated_fixture, evidence, fault=None):
    discovery, trx, manifest = translated_fixture
    return EVIDENCE.verify(
        evidence=evidence,
        manifest_path=manifest,
        discovery_dir=discovery,
        trx=trx,
        profile=PROFILE,
        contract_path=FIXTURE_CONTRACT,
        fault=fault,
    )


@pytest.mark.parametrize("compress", [False, True])
def test_consistent_evidence_is_accepted(translated_fixture, tmp_path, compress):
    name = "evidence.jsonl.gz" if compress else "evidence.jsonl"
    records = _records(translated_fixture[2])
    summary = _verify(translated_fixture, _write(tmp_path / name, records, compress))
    assert summary["status"] == "passed"
    assert summary["records"] == {
        "device": 1,
        "dispatch": 1,
        "pipeline": 1,
        "readback": 1,
    }
    assert summary["device"]["hardware"] is True
    assert summary["dispatchedShaders"] == summary["translatedShaders"] == 1
    assert summary["translatedShadersNotDispatched"] == []
    assert summary["casesWithDispatch"] == 1
    assert summary["casesWithEvidence"] == 2


def _mutate(records, manifest_path, mutation):
    device, pipeline, dispatch, readback = records
    if mutation == "no-device":
        records.remove(device)
    elif mutation == "no-dispatch":
        records.remove(dispatch)
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
        records.append({**readback, "record": "upload"})
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


def test_negative_control_evidence_must_carry_its_fault(translated_fixture, tmp_path):
    records = _records(translated_fixture[2])
    with pytest.raises(EVIDENCE.EvidenceError):
        _verify(
            translated_fixture,
            _write(tmp_path / "plain.jsonl", records),
            fault="swap-bindings",
        )
    for record in records:
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
