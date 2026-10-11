"""Float scatter preserves storage, indexed updates and native evidence."""

import hashlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crosstl.project.runtime_value_encoding import FLOAT32_BITS
from demos.integrations.mlx.portable_host import (
    gather_dispatch,
    runtime,
    scatter_evidence,
)
from demos.integrations.mlx.portable_host import scatter_workloads as workloads
from demos.integrations.mlx.portable_host import verify_scatter
from demos.integrations.mlx.portable_host.gather_workloads import words
from demos.integrations.mlx.tests.host.test_portable_scatter import (
    buffers,
    scatter_event,
)
from tests.ci_helpers import assert_paths_covered

CASES = list(workloads.float_cases())
ACTIVE = [case for case in CASES if case["layout"] != "empty"]


def test_float_scatter_case_inventory():
    assert len(CASES) == len({case["id"] for case in CASES}) == 81
    assert len(ACTIVE) == 76
    assert {case["operation"] for case in CASES} == {
        "none",
        "sum",
        "prod",
        "min",
        "max",
    }
    assert {case["index_dtype"] for case in CASES} == {
        "int32",
        "uint32",
        "int64",
        "uint64",
    }


def test_float_scatter_preserves_original_integer_workload_bytes():
    digest = hashlib.sha256()
    cases = list(workloads.cases())
    assert len(cases) == 146
    assert [case["id"] for case in cases[144:]] == [
        "int32-none-capacity",
        "uint32-none-capacity",
    ]
    for case in cases[:144]:
        digest.update(case["id"].encode())
        source, indices, updates, expected = workloads.reference(np, case)
        for array in (source, *indices, updates, expected):
            digest.update(str((str(array.dtype), array.shape, array.strides)).encode())
            digest.update(array.tobytes())
    assert (
        digest.hexdigest()
        == "d1c1918c8caba506f06acd2ffec60ef26ddb2e3b6e1b7a7087d6a9a906000ac7"
    )


def test_float_scatter_payload_imports_contiguous_storage():
    def array(value):
        assert value.flags.c_contiguous
        return np.array(value)

    case = next(case for case in CASES if case["layout"] == "payloads")
    source, indices, updates = workloads.arrays(SimpleNamespace(array=array), np, case)
    expected_source, expected_indices, expected_updates = workloads.arrays(np, np, case)
    assert words(np, source) == words(np, expected_source)
    assert words(np, updates) == words(np, expected_updates)
    assert np.array_equal(indices, expected_indices)


def test_float_scatter_requires_distinct_native_ci_profiles():
    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[5]
            / ".github/workflows/demo-project-testing.yml"
        ).read_text()
    )
    job = workflow["jobs"]["scatter-host"]
    rows = job["strategy"]["matrix"]["include"]
    assert {(row["target"], row["profile"]) for row in rows} == {
        ("metal", "integer"),
        ("directx", "integer"),
        ("opengl", "integer"),
        ("metal", "float32"),
        ("directx", "float32"),
        ("opengl", "float32"),
    }
    assert (
        next(
            row["os"]
            for row in rows
            if row["target"] == "metal" and row["profile"] == "float32"
        )
        == "xcode-27"
    )
    steps = {step.get("name"): step for step in job["steps"]}
    assert "Xcode_27.1.app" in steps["Select float scatter Metal toolchain"]["run"]
    step = steps["Execute general scatter host operations"]
    assert "matrix.profile == 'float32' && '--float32'" in step["run"]
    assert "if" not in step and "continue-on-error" not in step
    upload = steps["Retain general scatter execution evidence"]
    assert "matrix.profile" in upload["with"]["name"]
    assert upload["if"] == "always()"
    for trigger in ("pull_request", "push"):
        assert_paths_covered(
            workflow.get("on", workflow.get(True))[trigger]["paths"],
            "demos/integrations/mlx/tests/host/test_portable_float_scatter.py",
        )


@pytest.mark.parametrize("native", (False, True))
def test_float_scatter_records_require_the_complete_profile(native):
    records = []
    for index, case in enumerate(CASES):
        *_, expected = workloads.reference(np, case)
        records.append(
            {
                **case,
                "actual": words(np, expected),
                "shape": list(expected.shape),
                "resultDtype": "mlx.core.float32",
                "inputUnchanged": True,
                "dispatchStart": index if native else 0,
                "dispatchCount": int(native),
            }
        )
    verify_scatter.validate_records(np, records, native=native, float32=True)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records, native=native)
    with pytest.raises(ValueError):
        verify_scatter.validate_records(np, records[:-1], native=native, float32=True)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_float_scatter_reference_matches_independent_indexed_update(case):
    source, indices, updates, expected = workloads.reference(np, case)
    actual = source.copy()
    if case["operation"] == "none":
        actual.view(np.uint32)[tuple(indices)] = updates.view(np.uint32)
    else:
        {"sum": np.add, "prod": np.multiply, "min": np.minimum, "max": np.maximum}[
            case["operation"]
        ].at(actual, tuple(indices), updates)
    assert words(np, actual) == words(np, expected)
    assert source.dtype == updates.dtype == expected.dtype == np.float32
    if case["layout"] == "payloads":
        assert words(np, source) != words(np, expected)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_float_scatter_all_order_intermediates_are_exact(case):
    source, indices, updates, _ = workloads.reference(np, case)
    groups = {}
    for coordinate in np.ndindex(indices[0].shape):
        prefix = tuple(
            int(index[coordinate]) % source.shape[axis]
            for axis, index in enumerate(indices)
        )
        suffix = source.shape[len(prefix) :]
        for offset in np.ndindex(suffix):
            groups.setdefault(prefix + offset, []).append(updates[coordinate + offset])
    for address, updates in groups.items():
        if case["operation"] == "none":
            assert len(set(words(np, np.asarray(updates)))) == 1
            continue
        initial = Fraction(float(source[address]))
        values = [Fraction(float(value)) for value in updates]
        if case["operation"] == "sum":
            assert all((value * 4).denominator == 1 for value in [initial, *values])
            assert 4 * (abs(initial) + sum(map(abs, values))) < 2**24
        elif case["operation"] == "prod":
            if case["layout"] == "alias":
                assert len(values) == 1
                assert (
                    Fraction(float(np.float32(initial * values[0])))
                    == initial * values[0]
                )
            else:
                assert set(map(abs, values)) <= {
                    Fraction(1, 2),
                    Fraction(1),
                    Fraction(2),
                }
                lower = abs(initial) * np.prod(
                    [min(Fraction(1), abs(value)) for value in values], dtype=object
                )
                upper = abs(initial) * np.prod(
                    [max(Fraction(1), abs(value)) for value in values], dtype=object
                )
                assert Fraction(2) ** -126 <= lower <= upper < Fraction(2) ** 127
        else:
            assert all(np.isfinite(float(value)) for value in [initial, *values])


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("case", ACTIVE, ids=lambda case: case["id"])
def test_float_scatter_audit_reconstructs_raw_uploads(tmp_path, target, case):
    event = scatter_event(tmp_path, case, target)
    source, indices, updates, expected = workloads.reference(np, case)
    initial, actual_indices, actual_updates, actual = scatter_evidence.audit_event(
        np, event
    )
    assert words(np, initial) == words(np, source)
    assert all(np.array_equal(a, b) for a, b in zip(indices, actual_indices))
    assert words(np, actual_updates) == words(np, updates)
    assert actual == words(np, expected)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (
        "float-upload",
        "bool-upload",
        "negative-upload",
        "large-upload",
        "upload-encoding",
        "request-encoding",
        "member-type",
        "member-offset",
        "numeric-readback",
        "negative-readback",
        "bool-readback",
        "large-readback",
        "guard",
        "hash",
    ),
)
def test_float_scatter_rejects_corrupt_raw_evidence(tmp_path, target, fault):
    event = scatter_event(tmp_path, CASES[0], target)
    if fault.endswith("upload"):
        event["inputs"]["updates"]["values"][0] = {
            "float-upload": 0.5,
            "bool-upload": True,
            "negative-upload": -1,
            "large-upload": 2**32,
        }[fault]
    elif fault == "upload-encoding":
        if target == "opengl":
            event["inputs"]["out"]["encoding"] = FLOAT32_BITS
        else:
            event["inputs"]["out"].pop("encoding")
    elif fault == "request-encoding":
        if target == "opengl":
            event["details"]["request"]["buffers"]["out"]["encoding"] = FLOAT32_BITS
        else:
            event["details"]["request"]["buffers"]["out"].pop("encoding")
    elif fault.startswith("member"):
        member = event["details"]["request"]["buffers"]["out"]["binding"]["metadata"][
            "scalarLayout"
        ]["structMembers"][0]
        member["physicalType" if fault == "member-type" else "offsetBytes"] = (
            ("float" if target == "opengl" else "uint") if fault == "member-type" else 4
        )
    elif fault.endswith("readback"):
        event["scatterValues"][0] = {
            "numeric-readback": 0.5,
            "negative-readback": -1,
            "bool-readback": False,
            "large-readback": 2**32,
        }[fault]
    elif fault == "guard":
        event["scatterGuardValues"][0] = 0
    else:
        event["outputHash"] = "0" * 64
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize("operation", ("none", "sum", "prod", "min", "max"))
def test_float_scatter_advertises_native_targets(target, operation):
    host = SimpleNamespace(target=target, descriptors={}, gathers=object())
    entry = f"scatterfloat32int64_{operation}_1_updc_true_nwork1_int"
    assert runtime.HostRuntime._entry_available(host, entry.encode()) == 1


@pytest.mark.parametrize("member", ("out", "updates"))
def test_opengl_float_scatter_rejects_exchanged_storage_roles(tmp_path, member):
    event = scatter_event(tmp_path, CASES[0], "opengl")
    value = event["inputs"][member]
    binding = event["details"]["request"]["buffers"][member]
    scalar = binding["binding"]["metadata"]["scalarLayout"]
    dtype = "float32" if member == "out" else "uint32"
    value["dtype"] = binding["dtype"] = scalar["elementType"] = dtype
    if member == "out":
        scalar["structMembers"][0]["physicalType"] = "float"
        value["encoding"] = binding["encoding"] = FLOAT32_BITS
    else:
        value.pop("encoding")
        binding.pop("encoding")
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("word", (0.5, True, -1, 2**32))
def test_opengl_float_scatter_rejects_invalid_output_words(tmp_path, word):
    event = scatter_event(tmp_path, CASES[0], "opengl")
    event["inputs"]["out"]["values"][0] = word
    with pytest.raises(ValueError):
        scatter_evidence.audit_event(np, event)


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault", (None, "encoding", "dtype", "guard", "member", "negative", "float", "bool")
)
def test_float_scatter_dispatch_preserves_raw_initialization(
    tmp_path, monkeypatch, target, fault
):
    entry, supplied, arrays, expected, execution = buffers(CASES[-1])
    event = scatter_event(tmp_path, CASES[-1], target)
    descriptor = {
        "artifact": {},
        "bindings": [
            {"name": name, "scalarLayout": value["binding"]["metadata"]["scalarLayout"]}
            for name, value in event["details"]["request"]["buffers"].items()
        ],
    }
    if fault == "member":
        next(value for value in descriptor["bindings"] if value["name"] == "out")[
            "scalarLayout"
        ]["structMembers"][0][
            "physicalType"
        ] = "float" if target == "opengl" else "uint"

    def request(_descriptor, _directory, inputs, outputs, launch, **kwargs):
        assert inputs["out"]["values"] == words(np, arrays["out"]) + runtime.COPY_GUARD
        assert inputs["updates"]["values"] == words(np, arrays["updates"])
        assert inputs["updates"]["dtype"] == "float32"
        assert inputs["updates"]["encoding"] == FLOAT32_BITS
        if target == "opengl":
            assert inputs["out"]["dtype"] == "uint32"
            assert "encoding" not in inputs["out"]
        else:
            assert inputs["out"]["dtype"] == "float32"
            assert inputs["out"]["encoding"] == FLOAT32_BITS
        return SimpleNamespace(outputs=outputs)

    def execute(_host, request):
        output = {
            **request.outputs["out"],
            "values": words(np, expected) + runtime.COPY_GUARD,
        }
        if fault == "encoding":
            if target == "opengl":
                output["encoding"] = FLOAT32_BITS
            else:
                output.pop("encoding")
        elif fault == "dtype":
            output["dtype"] = "float32" if target == "opengl" else "uint32"
        elif fault == "guard":
            output["values"][-1] = 0
        elif fault in {"negative", "float", "bool"}:
            output["values"][0] = {"negative": -1, "float": 0.5, "bool": False}[fault]
        return SimpleNamespace(status="ok", outputs={"out": output}, details={})

    monkeypatch.setattr(
        gather_dispatch, "build_native_loader_dispatch_request", request
    )
    monkeypatch.setattr(gather_dispatch, "execute", execute)
    host = SimpleNamespace(
        target=target,
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
        gathers=SimpleNamespace(get=lambda *_: (descriptor, tmp_path)),
    )
    table = (runtime.Buffer * len(supplied))(*supplied.values())
    launch = runtime.Launch(tuple(execution["workgroupCount"]), (1, 1, 1))
    initial = arrays["out"].tobytes()
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            gather_dispatch.dispatch(
                host, entry, table, len(table), expected.size, launch
            )
        assert arrays["out"].tobytes() == initial and host.dispatch_count == 0
    else:
        gather_dispatch.dispatch(host, entry, table, len(table), expected.size, launch)
        assert words(np, arrays["out"]) == words(np, expected)
        assert host.dispatch_count == 1
        assert (
            hashlib.sha256(arrays["out"].tobytes()).hexdigest() == event["outputHash"]
        )
