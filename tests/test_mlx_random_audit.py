"""Check random-audit oracles, bounded bindings and rejection of incomplete evidence."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx import random_audit as audit


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
def test_audit_keeps_complete_failure_evidence(tmp_path, monkeypatch, wrong):
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
        case = audit.workloads()[run.calls]
        assert execution == case["execution"]
        run.calls += 1
        result = native_result(selected, case)
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
    result = audit.audit(tmp_path, "opengl", output)
    assert result["passed"] is (not wrong)
    assert len(result["cases"]) == run.calls == 20
    assert all(case["passed"] is (not wrong) for case in result["cases"])
    assert len(list(output.glob("*/result.json"))) == 20
    assert json.loads((output / "evidence.json").read_text()) == result
    assert closed == [True]


def test_cli_returns_failure_for_failed_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "audit", lambda *args: {"passed": False, "cases": []})
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
