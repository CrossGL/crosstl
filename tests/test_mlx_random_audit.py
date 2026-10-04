"""Check random-audit oracles, bounded bindings and rejection of incomplete evidence."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx import random_audit as audit
from demos.integrations.mlx.portable_host.random_layout import RandomOutputLayout


@pytest.mark.parametrize("count", range(34))
@pytest.mark.parametrize("keys", (1, 3))
def test_random_output_storage_keeps_every_logical_byte(count, keys):
    layout = RandomOutputLayout(keys, count)
    stride = max(4, count) if count else 0
    assert layout.native_bytes_per_key == stride
    assert layout.native_byte_count == keys * stride
    assert layout.logical_byte_count == keys * count
    raw = bytes((index * 37 + 131) % 256 for index in range(keys * stride))
    values = [value if value < 128 else value - 256 for value in raw]
    assert layout.unpack(values) == b"".join(
        raw[key * stride : key * stride + count] for key in range(keys)
    )


@pytest.mark.parametrize(
    "keys,count", [(0, 3), (-1, 3), (True, 3), (1, -1), (1, 1.0), (1, False)]
)
def test_random_output_rejects_invalid_dimensions(keys, count):
    with pytest.raises(ValueError, match="positive keys"):
        RandomOutputLayout(keys, count)


@pytest.mark.parametrize("bad", [-129, 128, 1.0, True, "1"])
def test_random_output_rejects_nonbyte_readback(bad):
    with pytest.raises(ValueError, match="signed byte"):
        RandomOutputLayout(1, 1).unpack([0, 0, 0, bad])


@pytest.mark.parametrize("size", (0, 1, 3, 5))
def test_random_output_requires_full_native_allocation(size):
    with pytest.raises(ValueError, match="exact signed byte"):
        RandomOutputLayout(1, 1).unpack([0] * size)


def test_partial_byte_profile_preserves_source_counters():
    cases = audit.workloads(byte_tails=True)
    assert len(cases) == len({case["id"] for case in cases}) == 48
    assert {case["logicalBytesPerKey"] for case in cases} == {
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        15,
        17,
        33,
    }
    for case in cases:
        count, keys = case["logicalBytesPerKey"], case["keyCount"]
        assert case["wordCount"] == (count + 3) // 4
        assert case["odd"] == case["wordCount"] % 2
        assert case["bytesPerKey"] == max(4, count)
        assert len(case["expected"]) == max(4, count) * keys
        assert len(case["logicalExpected"]) == count * keys
        assert case["execution"]["workgroupCount"] == [
            keys,
            (case["wordCount"] + 1) // 2,
            1,
        ]
        layout = RandomOutputLayout(keys, count)
        assert list(layout.unpack(case["expected"])) == case["logicalExpected"]
        if count < 4:
            assert case["expected"][:4] == [89, 1, 32, 107]
            assert case["logicalExpected"][:count] == [89, 1, 32][:count]


@pytest.mark.parametrize("count", (False, -1, 0, 1.0, 65535))
def test_partial_byte_workload_rejects_invalid_allocation(count):
    with pytest.raises(ValueError, match="workload"):
        audit.byte_workload("rbitsc", 1, count)


def test_native_module_is_retained_with_executed_identity(tmp_path):
    content = b"MTLB-retention-control"
    module = tmp_path / "temporary.metallib"
    module.write_bytes(content)
    destination = tmp_path / "evidence"
    destination.mkdir()
    digest = hashlib.sha256(content).hexdigest()
    record = audit.retain_native_module(
        {
            "nativeRuntimeDispatch": {"modulePath": str(module)},
            "metalRuntime": {"librarySHA256": digest},
        },
        "metal",
        destination,
    )
    module.unlink()
    assert record == {
        "path": "kernel.metallib",
        "sizeBytes": len(content),
        "sha256": digest,
    }
    assert (destination / record["path"]).read_bytes() == content


@pytest.mark.parametrize("fault", ("path", "missing", "hash", "header"))
def test_invalid_executed_module_cannot_be_retained(tmp_path, fault):
    module = tmp_path / "temporary.metallib"
    content = b"BAD!" if fault == "header" else b"MTLB-retention-control"
    module.write_bytes(content)
    details = {
        "nativeRuntimeDispatch": {"modulePath": str(module)},
        "metalRuntime": {"librarySHA256": hashlib.sha256(content).hexdigest()},
    }
    if fault == "path":
        details["nativeRuntimeDispatch"] = {}
    elif fault == "missing":
        module.unlink()
    elif fault == "hash":
        details["metalRuntime"]["librarySHA256"] = "0" * 64
    with pytest.raises(ValueError, match="module"):
        audit.retain_native_module(details, "metal", tmp_path)
    assert not (tmp_path / "kernel.metallib").exists()


def test_native_random_audit_is_required_on_macos():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/mlx-gather-roundtrip.yml"
        ).read_text()
    )
    events = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "demos/integrations/mlx/random_audit.py" in events[event]["paths"]
        assert "tests/test_mlx_random_audit.py" in events[event]["paths"]
    job = workflow["jobs"]["metal"]
    assert job["runs-on"].startswith("macos-")
    (step,) = (
        step
        for step in job["steps"]
        if step.get("name") == "Validate translated Metal random kernels"
    )
    assert "if" not in step and not step.get("continue-on-error", False)
    assert "set -euo pipefail" in step["run"]
    assert "--timeout-seconds 300" in step["run"]
    assert "python -m demos.integrations.mlx.random_audit" in step["run"]
    assert "--target metal --output-dir .mlx-gather/random" in step["run"]


@pytest.mark.parametrize(
    "key,count,expected",
    [
        ((0, 0), (0, 0), (0x6B200159, 0x99BA4EFE)),
        ((0, 0), (0, 1), (0x375F238F, 0xCDDB151D)),
    ],
)
def test_original_metal_word_controls(key, count, expected):
    assert audit.threefry(key, count) == expected


@pytest.mark.parametrize("bad", [-1, 2**32, True, 1.0, "1"])
def test_reference_rejects_non_uint32_words(bad):
    with pytest.raises(ValueError, match="uint32"):
        audit.threefry((0, bad), (0, 0))


def test_case_coverage_and_source_index_bounds():
    cases = audit.workloads()
    assert len(cases) == len({case["id"] for case in cases}) == 20
    for case in cases:
        assert len(case["expected"]) == case["keyCount"] * case["bytesPerKey"]
        assert len(case["expected"]) + audit.GUARD_COUNT < 65535
        assert all(-128 <= value <= 127 for value in case["expected"])
        assert case["execution"]["workgroupSize"] == [1, 1, 1]
        assert case["execution"]["workgroupCount"] == [
            case["keyCount"],
            (case["wordCount"] + 1) // 2,
            1,
        ]
        if case["entry"] == "rbits":
            for key_index in range(case["keyCount"]):
                assert (
                    tuple(
                        case["keys"][key_index + lane * case["keyCount"]]
                        for lane in (0, 1)
                    )
                    == audit.KEYS[key_index]
                )


def descriptor(target, entry):
    names = {
        "keys": "uint32",
        "out_": "int8" if target == "metal" else "int32",
        "odd": "bool" if target == "metal" else "uint32",
        "bytes_per_key": "uint64",
    }
    if entry == "rbits":
        names.update(ndim="int32", key_shape="int32", key_strides="int64")
    return {
        "target": target,
        "entryPoint": {"name": entry},
        "artifact": {
            "packagePath": "artifacts/random",
            "hash": {"algorithm": "sha256", "value": "a" * 64},
            "sizeBytes": 12,
        },
        "bindings": [
            {
                "name": name,
                "scalarLayout": {
                    "memberName": entry + "_" + name if target == "directx" else name,
                    "elementType": dtype,
                },
            }
            for name, dtype in names.items()
        ],
    }


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("case", audit.workloads(), ids=lambda case: case["id"])
def test_physical_bindings(target, case):
    desc = descriptor(target, case["entry"])
    desc["bindings"].append({"provenance": {"executionInput": "group-count"}})
    inputs, outputs = audit.dispatch_values(desc, case)
    assert set(outputs) == {"out_"}
    assert outputs["out_"]["values"] == [91] * (len(case["expected"]) + 17)
    assert inputs["keys"]["values"] == case["keys"]
    assert inputs["bytes_per_key"]["values"] == [case["bytesPerKey"]]
    assert type(inputs["odd"]["values"][0]) is (bool if target == "metal" else int)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "layout", "dtype"])
def test_rejects_incomplete_binding_contract(fault):
    case = audit.workload("rbitsc", 1, 2)
    desc = descriptor("opengl", "rbitsc")
    if fault == "duplicate":
        desc["bindings"].append(copy.deepcopy(desc["bindings"][0]))
    elif fault == "missing":
        desc["bindings"].pop()
    elif fault == "layout":
        desc["bindings"][1]["scalarLayout"] = None
    else:
        desc["bindings"][1]["scalarLayout"]["elementType"] = "uint32"
    with pytest.raises(ValueError):
        audit.dispatch_values(desc, case)


def native_result(desc, case):
    target = desc["target"]
    artifact = desc["artifact"]
    identity = {key: artifact[key] for key in ("hash", "sizeBytes")}
    actions = ["dispatch-translated-artifact", "collect-runtime-outputs"] + {
        "metal": ["compile-metal-for-native-runtime", "link-metal-for-native-runtime"],
        "opengl": ["validate-glsl-for-opengl-runtime"],
        "directx": ["compile-hlsl-for-directx-runtime"],
    }[target]
    return SimpleNamespace(
        status="ok",
        outputs={
            "out_": {
                "dtype": "int8" if target == "metal" else "int32",
                "shape": [len(case["expected"]) + 17],
                "values": case["expected"] + [91] * 17,
            }
        },
        details={
            "artifactIdentityVerification": {
                "target": target,
                "verificationStatus": "verified",
                "expectedIdentity": identity,
                "observedIdentity": copy.deepcopy(identity),
            },
            "nativeRuntimeDispatch": {
                "artifact": {"target": target, "packagePath": artifact["packagePath"]},
                "dispatch": {
                    **copy.deepcopy(case["execution"]),
                    "entryPoint": desc["entryPoint"]["name"],
                },
            },
            "runtimeParityAdapter": {
                "target": target,
                "runtimeAdapter": target + "-native-runtime",
            },
            "adapterSteps": [
                {"action": action, "status": "passed"} for action in actions
            ],
        },
    )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "zeros",
        "unsigned",
        "guard",
        "type",
        "shape",
        "missing",
        "identity",
        "compiler",
        "geometry",
        "status",
    ],
)
def test_numerical_and_native_evidence_contract(target, fault):
    case = audit.workload("rbitsc", 1, 2)
    desc = descriptor(target, case["entry"])
    result = native_result(desc, case)
    output = result.outputs["out_"]
    if fault == "zeros":
        output["values"][:8] = [0] * 8
    elif fault == "unsigned":
        output["values"][:8] = [value & 255 for value in output["values"][:8]]
    elif fault == "guard":
        output["values"][-1] = 0
    elif fault == "type":
        output["values"][0] = float(output["values"][0])
    elif fault == "shape":
        output["shape"] = [1, len(output["values"])]
    elif fault == "missing":
        result.details = {}
    elif fault == "identity":
        result.details["artifactIdentityVerification"]["observedIdentity"][
            "sizeBytes"
        ] += 1
    elif fault == "compiler":
        result.details["adapterSteps"] = result.details["adapterSteps"][:2]
    elif fault == "geometry":
        result.details["nativeRuntimeDispatch"]["dispatch"]["workgroupCount"][0] += 1
    elif fault == "status":
        result.status = "skipped"
    assert audit.compare_readback(result, "out_", case, desc) is (fault is None)


@pytest.mark.parametrize("wrong", [False, True])
@pytest.mark.parametrize("byte_tails", [False, True])
def test_audit_keeps_complete_failure_evidence(
    tmp_path, monkeypatch, wrong, byte_tails
):
    cases = audit.workloads(byte_tails=byte_tails)
    desc = {entry: descriptor("opengl", entry) for entry in audit.ENTRIES}
    monkeypatch.setattr(audit, "build_packages", lambda *args: desc)
    monkeypatch.setattr(audit, "verify_source", lambda root: None)
    monkeypatch.setattr(
        audit,
        "build_native_loader_dispatch_request",
        lambda d, p, i, o, e, **kw: (d, e),
    )
    closed = []

    def run(request):
        selected, execution = request
        case = cases[run.calls]
        assert execution == case["execution"]
        run.calls += 1
        result = native_result(selected, case)
        module = tmp_path / "temporary.glsl"
        module.write_text("#version 430\nvoid main() {}\n")
        result.details["nativeRuntimeDispatch"]["modulePath"] = str(module)
        result.details["retainedNativeModule"] = audit.retain_native_module(
            result.details, "opengl", native.evidence_directory
        )
        module.unlink()
        if wrong:
            result.outputs["out_"]["values"][0] = 0
        return result

    run.calls = 0
    native = SimpleNamespace(
        run=run,
        runtime_adapter=SimpleNamespace(
            runtime=SimpleNamespace(close=lambda: closed.append(True))
        ),
    )
    monkeypatch.setattr(audit, "executor", lambda target: native)
    output = tmp_path / "audit"
    result = audit.audit(tmp_path, "opengl", output, byte_tails=byte_tails)
    assert result["passed"] is (not wrong)
    assert len(result["cases"]) == run.calls == len(cases)
    assert all(case["passed"] is (not wrong) for case in result["cases"])
    assert len(list(output.glob("*/result.json"))) == len(cases)
    assert len(list(output.glob("*/logical-output.json"))) == (
        0 if wrong else len(cases)
    )
    assert json.loads((output / "evidence.json").read_text()) == result
    assert closed == [True]


def test_cli_returns_failure_for_failed_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit, "audit", lambda *args, **kwargs: {"passed": False, "cases": []}
    )
    assert (
        audit.main(
            [
                "--mlx-root",
                str(tmp_path),
                "--target",
                "opengl",
                "--output-dir",
                str(tmp_path / "audit"),
            ]
        )
        == 1
    )


def test_native_failure_records_every_case_and_closes_runtime(tmp_path, monkeypatch):
    desc = {entry: descriptor("opengl", entry) for entry in audit.ENTRIES}
    monkeypatch.setattr(audit, "build_packages", lambda *args: desc)
    monkeypatch.setattr(audit, "verify_source", lambda root: None)
    monkeypatch.setattr(
        audit, "build_native_loader_dispatch_request", lambda *args, **kwargs: None
    )
    closed = []

    def run(request):
        raise RuntimeError("Native compiler rejected the source")

    native = SimpleNamespace(
        run=run,
        runtime_adapter=SimpleNamespace(
            runtime=SimpleNamespace(close=lambda: closed.append(True))
        ),
    )
    monkeypatch.setattr(audit, "executor", lambda target: native)
    output = tmp_path / "audit"
    result = audit.audit(tmp_path, "opengl", output)
    assert result["passed"] is False
    assert len(result["cases"]) == 20
    assert all(
        case["error"] == "Native compiler rejected the source"
        for case in result["cases"]
    )
    assert closed == [True]
    before = (output / "evidence.json").read_bytes()
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path, "opengl", output)
    assert (output / "evidence.json").read_bytes() == before


def test_contract_checks_run_on_all_platforms():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/mlx-portable-host.yml"
        ).read_text()
    )
    events = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "tests/test_mlx_random_audit.py" in events[event]["paths"]
        assert "demos/integrations/mlx/random_audit.py" in events[event]["paths"]
    (job,) = (
        job
        for job in workflow["jobs"].values()
        if any(
            step.get("name") == "Validate portable host contracts"
            for step in job["steps"]
        )
    )
    assert {case["target"] for case in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    (step,) = (
        step
        for step in job["steps"]
        if step.get("name") == "Validate portable host contracts"
    )
    assert "tests/test_mlx_random_audit.py" in step["run"]
    assert "-n auto" in step["run"]


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("position", (0, 1, 2, 3, 4, 7, 8, 11, 12))
def test_byte_profile_checks_padding_and_guards_before_unpacking(target, position):
    case = audit.byte_workload("rbitsc", 3, 1)
    desc = descriptor(target, "rbitsc")
    result = native_result(desc, case)
    assert audit.compare_readback(result, "out_", case, desc)
    result.outputs["out_"]["values"][position] ^= 1
    assert not audit.compare_readback(result, "out_", case, desc)


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_partial_random_profile_is_required_on_each_platform(target):
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/mlx-gather-roundtrip.yml"
        ).read_text()
    )
    events = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "demos/integrations/mlx/portable_host/**" in events[event]["paths"]
    (step,) = (
        step
        for step in workflow["jobs"][target]["steps"]
        if "random byte tails" in step.get("name", "")
    )
    assert "if" not in step and not step.get("continue-on-error", False)
    assert "set -euo pipefail" in step["run"]
    assert "--timeout-seconds 450" in step["run"]
    assert "--byte-tails" in step["run"]
    assert f"--target {target}" in step["run"]
    assert "python -m demos.integrations.mlx.random_audit" in step["run"]
