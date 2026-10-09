"""Large-cast readbacks, batch boundaries and required native CI coverage."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.integrations.mlx.portable_host import verify_cast_batches as verifier


def test_cast_inventory_covers_boundaries_offsets_and_storage_widths():
    cases = verifier.cases()
    assert len(cases) == 11 and len({case["id"] for case in cases}) == 11
    assert {case["count"] for case in cases} == {65535, 65536, 131075}
    assert sum(len(verifier.batch_sizes(case)) for case in cases) == 21
    assert {case["source"] for case in cases} == {"float32", "int32", "uint32", "bool_"}
    assert any(case["source"] == case["destination"] for case in cases)
    for case in cases:
        source, expected = verifier.reference(np, case)
        assert source.shape == expected.shape == tuple(case["shape"])
        assert source.size == case["count"] and case["offset"] > 0
        assert np.shares_memory(source, source.base)
        chunks = [
            source.reshape(-1)[first : first + verifier.BATCH_SIZE]
            for first in range(0, source.size, verifier.BATCH_SIZE)
        ]
        hashes = [hashlib.sha256(chunk.tobytes()).hexdigest() for chunk in chunks]
        assert len(hashes) == len(set(hashes))


@pytest.mark.parametrize(
    "fault", [None, "source", "changed-source", "result", "shape", "dtype"]
)
def test_cast_readback_requires_exact_storage(fault):
    case = verifier.cases()[0]
    source, expected = verifier.reference(np, case)
    before, after, actual = source.copy(), source.copy(), expected.copy()
    if fault == "source":
        before[0] += 1
    elif fault == "changed-source":
        after[-1] += 1
    elif fault == "result":
        actual[-1] += 1
    elif fault == "shape":
        actual = actual.reshape(256, 256)
    elif fault == "dtype":
        actual = actual.astype("int64")
    if fault:
        with pytest.raises(ValueError):
            verifier.record(np, case, before, after, actual)
    else:
        item = verifier.record(np, case, before, after, actual)
        assert item["sourcePreserved"] is True
        assert item["resultHash"] == hashlib.sha256(expected.tobytes()).hexdigest()


def synthetic_case(tmp_path, target):
    case = verifier.cases()[0]
    source, expected = verifier.reference(np, case)
    np.savez(
        tmp_path / (case["id"] + ".npz"), before=source, after=source, actual=expected
    )
    item = verifier.record(np, case, source, source, expected)
    item.update(dispatchStart=0, dispatchCount=2)
    entry = "v_copy" + case["source"] + case["destination"]
    native_entry = {"opengl": "main", "directx": "CSMain"}.get(target, entry)
    events = []
    first = 0
    for size in verifier.batch_sizes(case):
        guard = (
            np.asarray(verifier.COPY_GUARD, dtype="uint32")
            .astype(case["destination"])
            .tolist()
        )
        inputs = {
            "src": {
                "dtype": case["source"],
                "shape": [size],
                "values": source[first : first + size].tolist(),
            },
            "dst": {
                "dtype": case["destination"],
                "shape": [size + len(guard)],
                "values": [0] * size + guard,
            },
            "size": {"dtype": "uint32", "shape": [1], "values": [size]},
        }
        bindings = {
            name: {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "binding": {
                    "metadata": {
                        "scalarLayout": {
                            "memberName": name,
                            "elementType": value["dtype"],
                            "elementStrideBytes": np.dtype(value["dtype"]).itemsize,
                        }
                    }
                },
            }
            for name, value in inputs.items()
        }
        events.append(
            {
                "entry": entry,
                "threads": size,
                "target": target,
                "dispatchVersion": 3,
                "workgroupCount": [size, 1, 1],
                "workgroupSize": [1, 1, 1],
                "inputs": inputs,
                "details": {
                    "request": {
                        "buffers": bindings,
                        "target": target,
                        "entryPoint": native_entry,
                        "dispatch": {
                            "entryPoint": native_entry,
                            "workgroupCount": [size, 1, 1],
                            "workgroupSize": [1, 1, 1],
                        },
                    }
                },
                "castValues": expected[first : first + size].tolist(),
                "castGuardValues": guard,
            }
        )
        first += size
    return case, item, events


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "order",
        "launch",
        "width",
        "version",
        "offset",
        "output",
        "guard",
        "binding",
        "dtype",
        "receipt",
        "request",
        "output-coercion",
    ],
)
@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_cast_audit_rejects_incomplete_or_changed_batches(
    tmp_path, monkeypatch, fault, target
):
    case, item, events = synthetic_case(tmp_path, target)
    audited = []
    monkeypatch.setattr(
        verifier, "audit_native_execution", lambda event: audited.append(event)
    )
    if fault == "missing":
        events.pop()
    elif fault == "order":
        events.reverse()
    elif fault == "launch":
        events[0]["workgroupCount"][0] -= 1
    elif fault == "width":
        events[0]["workgroupSize"] = [32, 1, 1]
    elif fault == "version":
        events[0]["dispatchVersion"] = 2
    elif fault == "offset":
        events[1]["inputs"]["src"]["values"][0] = events[0]["inputs"]["src"]["values"][
            0
        ]
    elif fault == "output":
        events[1]["castValues"][0] += 1
    elif fault == "guard":
        events[1]["castGuardValues"][0] = 0
    elif fault == "binding":
        events[0]["inputs"].pop("size")
    elif fault == "dtype":
        events[0]["inputs"]["src"]["dtype"] = "int64"
    elif fault == "receipt":
        item["resultHash"] = "0" * 64
    elif fault == "request":
        events[0]["details"]["request"]["dispatch"]["workgroupCount"] = [1, 1, 1]
    elif fault == "output-coercion":
        events[0]["castValues"][0] = str(events[0]["castValues"][0])
    if fault:
        with pytest.raises(ValueError):
            verifier.audit_case(np, case, item, events, tmp_path)
    else:
        verifier.audit_case(np, case, item, events, tmp_path)
        assert audited == events


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("bool", [2]),
        ("bool", [1]),
        ("bool", ["false"]),
        ("uint32", [True]),
        ("uint32", [-1]),
        ("uint32", [2**32]),
        ("int32", [2**31]),
        ("int32", [1.5]),
        ("int32", ["1"]),
        ("float32", [True]),
        ("float32", [float("inf")]),
        ("float32", [float("nan")]),
        ("float32", [0.1]),
        ("float32", [1e100]),
        ("uint32", (1,)),
        ("int64", [1]),
    ],
)
def test_cast_physical_storage_is_not_silently_coerced(dtype, values):
    with pytest.raises(ValueError, match="Cast physical values"):
        verifier.physical_values(np, values, dtype)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failure",
        "timeout",
        "missing",
        "passed",
        "commit",
        "target",
        "casesPerPath",
        "dispatchCount",
        "fullUpstreamSuite",
    ],
)
def test_bounded_cast_verification_requires_complete_success(
    tmp_path, monkeypatch, fault
):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "index.json").write_text(json.dumps({"target": "opengl"}))
    output = tmp_path / "evidence"
    summary = {
        "passed": True,
        "commit": verifier.COMMIT,
        "target": "opengl",
        "casesPerPath": 11,
        "dispatchCount": 21,
        "fullUpstreamSuite": False,
    }
    if fault in summary:
        summary[fault] = None
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--timeout-seconds") + 1] == "600"
        assert command[command.index("-m") + 1] == verifier.__name__
        assert command[-1] == str(output)
        assert not kwargs["check"]
        kwargs["stdout"].write("retained output")
        kwargs["stderr"].write("retained error")
        output.mkdir()
        if fault != "missing":
            (output / "evidence.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode={"failure": 1, "timeout": 124}.get(fault, 0))

    monkeypatch.setattr(verifier.subprocess, "run", run)
    if fault:
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            verifier.run_bounded(tmp_path / "mlx", packages, output)
    else:
        assert verifier.run_bounded(tmp_path / "mlx", packages, output) == summary
    assert len(calls) == 1
    assert output.with_suffix(".stdout").read_text() == "retained output"
    assert output.with_suffix(".stderr").read_text() == "retained error"
    assert json.loads(output.with_suffix(".command.json").read_text())[
        "returncode"
    ] == {"failure": 1, "timeout": 124}.get(fault, 0)


@pytest.mark.parametrize("value", [None, 1, 0, "true"])
def test_native_module_retention_requires_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="retention must be a Boolean"):
        verifier.HostRuntime(
            tmp_path / "packages", tmp_path / "trace", retain_native_modules=value
        )


def test_cast_ci_requires_native_execution_on_all_three_platforms():
    workflow = yaml.safe_load(
        Path(".github/workflows/demo-project-testing.yml").read_text()
    )
    job = workflow["jobs"]["portable-host"]
    assert {row["target"] for row in job["strategy"]["matrix"]["include"]} == {
        "metal",
        "directx",
        "opengl",
    }
    step = next(
        step
        for step in job["steps"]
        if step.get("name") == "Execute upstream tests on the translated backend"
    )
    assert not step.get("if") and not step.get("continue-on-error")
    assert "set -euo pipefail" in step["run"]
    assert "--timeout-seconds 4000" in step["run"]
    assert "portable_host.verify --mlx-root mlx-upstream" in step["run"]
    assert "--packages .mlx-portable-host/packages" in step["run"]
    assert "--cast-batches" in step["run"]
    assert "--output-dir .mlx-portable-host/evidence" in step["run"]
